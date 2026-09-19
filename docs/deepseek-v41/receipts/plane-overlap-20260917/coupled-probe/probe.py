"""Bounded demand-read / gate-up / down-plane overlap screen."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
from types import SimpleNamespace

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent GPU guard required before MLX import')
signal.alarm(180)
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'probe.json'
if OUT.exists():
    raise RuntimeError('refusing to overwrite measured evidence')
proof = json.loads((ROOT / 'construction.json').read_text())
before = host_memory_snapshot()
if (not before['box']['ok']
        or before['box']['used_bytes'] + proof['static_incremental_bound_bytes'] > 110000000000
        or before['box']['wired_bytes'] + proof['static_incremental_bound_bytes'] > 100 * 1024**3):
    raise RuntimeError('bounded probe does not fit current physical/wired memory')
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if source != proof['source_commit']:
    raise RuntimeError('source changed after construction accounting')
for name, digest in proof['helper_sha256'].items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('unchanged helper differs: ' + name)
inventory_bytes = (ROOT / 'artifact/manifest.json').read_bytes()
if hashlib.sha256(inventory_bytes).hexdigest() != proof['artifact_manifest_sha256']:
    raise RuntimeError('packed inventory changed')
artifact = json.loads(inventory_bytes)
model = Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
manifest_path = model / 'expert-manifest.json'
if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != artifact['source_manifest_sha256']:
    raise RuntimeError('source manifest changed')
st = (model / 'experts.bin').stat()
if {k: getattr(st, k) for k in artifact['source_identity']} != artifact['source_identity']:
    raise RuntimeError('weight source identity changed')

import numpy as np
import mlx.core as mx
from mtplx.expert_manifest import load_expert_manifest
from mtplx.expert_io import PositionalExpertReader
from mtplx.models.expert_mlx import MlxComponentBank, MlxComponentSlot, _run_component_bank_q4, _clamped_swiglu
from packed_storage import load_layer, remove_raw_scales, make_dispatch, bind_weight_reader, WEIGHT_BYTES
from paired_kernels import make_projection

mx.set_memory_limit(2 * 1024**3)
mx.set_cache_limit(256 * 1024**2)
manifest = load_expert_manifest(manifest_path)
records = {r.expert: r for r in manifest.records if r.layer == 20}
reader = bank = scales = None
report = dict(source_commit=source, construction=proof, before=before,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=[],
    scope='Three real layer20 expert records; real packed kernels; synthetic BF16 rows. Coupled I/O/compute screen, not full-model throughput.')


class OrderedRead:
    """Own all writable views until every submitted reader is terminal."""
    def __init__(self, items):
        jobs = []
        for record, dest in items:
            for i, (name, offset) in enumerate(zip(
                    ('gate_proj.weight', 'up_proj.weight', 'down_proj.weight'),
                    (0, 6266880, 12533760))):
                jobs.append((i == 2, record.sidecar_offset + offset, dest.component_view(name)))
        self.jobs = sorted(jobs, key=lambda j: j[0])  # GU first, stable record order
        self.futures = []
        self.items = len(items)

    def __enter__(self):
        return self

    @staticmethod
    def read(job):
        _, offset, view = job
        reader._readv_range_into('experts.bin', offset, (view,),
            cancel_event=None, deadline_ns=None, pipeline_phase=None)

    def start(self):
        reader.metrics.update(record_requests=self.items, records_read=self.items,
                              sidecar_record_requests=self.items)
        for job in self.jobs[1:]:
            self.futures.append((job[0], reader._fanout_executor.submit(self.read, job)))
        self.read(self.jobs[0])

    def wait(self, *, only_gu=False):
        for is_down, future in self.futures:
            if not only_gu or not is_down:
                future.result()

    def __exit__(self, exc_type, exc, tb):
        failure = None
        for _, future in self.futures:
            try:
                future.result()
            except BaseException as error:
                if failure is None:
                    failure = error
        for _, _, view in self.jobs:
            view.release()
        if exc_type is None:
            if failure is not None:
                raise failure
            reader.metrics.update(records_unhashed=self.items)


def make_stages():
    gu_kernel, down_kernel = make_projection(2304, 5120), make_projection(5120, 2304)
    gs, us, ds = (scales[p] for p in ('gate_proj', 'up_proj', 'down_proj'))

    def gu(x, bindings):
        rows = len(bindings)
        pairs = mx.array([(b.buffer.bank_index, b.expert) for b in bindings], mx.int32)
        x = x.reshape(rows, 1, 1, 5120)
        common = dict(template=[('T', mx.bfloat16)], grid=(32, 576, rows),
            threadgroup=(32, 2, 1), output_shapes=[(rows, 1, 1, 2304)], output_dtypes=[mx.bfloat16])
        g = gu_kernel(inputs=[x, pairs, bank.arrays['gate_proj.weight'], *gs], **common)[0]
        u = gu_kernel(inputs=[x, pairs, bank.arrays['up_proj.weight'], *us], **common)[0]
        return _clamped_swiglu(g, u, 10.0), pairs

    def down(h, pairs):
        rows = int(h.shape[0])
        return down_kernel(inputs=[h, pairs, bank.arrays['down_proj.weight'], *ds],
            template=[('T', mx.bfloat16)], grid=(32, 1280, rows), threadgroup=(32, 2, 1),
            output_shapes=[(rows, 1, 1, 5120)], output_dtypes=[mx.bfloat16])[0].reshape(rows, 5120)

    return gu, down


def run():
    global reader, bank, scales
    reader = PositionalExpertReader(model, bypass_page_cache=True, use_native=False, io_read_fanout=4)
    bank = MlxComponentBank(capacity=3, record=records[0], label='three-plane-overlap')
    experts = proof['experts']
    slots = [MlxComponentSlot(bank, (i+1) % 3, label=f'overlap-{i}') for i in range(3)]
    items = [(records[e], s) for e, s in zip(experts, slots)]
    for record, slot in items:
        reader.read_record_into(manifest, record, slot, verify_hash=True)
    report['record_sha256'] = {str(e): records[e].sha256 for e in experts}
    cases = []
    rng = np.random.default_rng(641)
    for rows in (3, 9, 18):
        bindings = tuple(SimpleNamespace(buffer=slots[i % 3], expert=experts[i % 3]) for i in range(rows))
        x = mx.array(rng.standard_normal((rows, 5120)).astype(np.float32)).astype(mx.bfloat16)
        y = _run_component_bank_q4(x, bindings, group_size=32, bits=4, swiglu_limit=10.0, codec='mxfp4')
        mx.eval(x, y)
        cases.append((rows, x, bindings, np.array(y.view(mx.uint16))))
    del y
    released = remove_raw_scales(bank, mx=mx)
    if released != 3 * 1105920:
        raise RuntimeError('raw scale ownership differs from accounting')
    for slot in slots:
        slot.nbytes = WEIGHT_BYTES
    scales = load_layer(ROOT / 'artifact', artifact['layers'][20], mx=mx)
    dispatch = make_dispatch(scales, mx=mx)
    gu, down = make_stages()
    bind_weight_reader(reader)
    report['packed_layer_bytes'] = artifact['layers'][20]['packed_bytes']
    report['payload_bytes_per_trip'] = 3 * WEIGHT_BYTES
    for rows, x, bindings, reference in cases:
        def trip(mode):
            start = time.perf_counter_ns()
            window = None
            if mode == 'stock':
                reader.read_component_records_into(manifest, items, verify_hash=False)
                y = dispatch(x, bindings)
                mx.eval(y)
            else:
                with OrderedRead(items) as reads:
                    reads.start()
                    if mode == 'ordered_all':
                        reads.wait()
                        y = dispatch(x, bindings)
                        mx.eval(y)
                    else:
                        reads.wait(only_gu=True)
                        hidden, pairs = gu(x, bindings)
                        mx.async_eval(hidden)
                        submitted = time.perf_counter_ns()
                        reads.wait()
                        window = time.perf_counter_ns() - submitted
                        y = down(hidden, pairs)
                        mx.eval(y)
            return y, (time.perf_counter_ns()-start)/1e9, window

        case = dict(rows=rows, unique_experts=3, parity={}, timings=[])
        for mode in ('stock', 'ordered_all', 'overlap'):
            y, _, _ = trip(mode)
            bits = np.array(y.view(mx.uint16))
            equal = bool(np.array_equal(bits, reference))
            case['parity'][mode] = dict(exact_bytes=equal, differing_elements=int(np.count_nonzero(bits != reference)))
            if not equal:
                raise RuntimeError('coupled plane result differs from exact native reference: '+mode)
        for mode in ('stock', 'ordered_all', 'overlap', 'stock', 'overlap', 'ordered_all', 'stock'):
            samples, windows = [], []
            for _ in range(5):
                y, elapsed, window = trip(mode)
                samples.append(elapsed)
                if window is not None:
                    windows.append(window)
            case['timings'].append(dict(arm=mode, samples_s=samples,
                median_s=statistics.median(samples), gu_submit_to_all_reads_ready_ns=windows))
        case['arm_medians_s'] = {mode: statistics.median(t['median_s'] for t in case['timings'] if t['arm']==mode)
                                for mode in ('stock', 'ordered_all', 'overlap')}
        case['latency_reduction_pct'] = {mode:100*(1-case['arm_medians_s'][mode]/case['arm_medians_s']['stock'])
                                         for mode in ('ordered_all', 'overlap')}
        report['cases'].append(case)
        OUT.write_text(json.dumps(report, indent=2)+'\n')
        print('PLANE_CASE', json.dumps({k:v for k,v in case.items() if k!='timings'}), flush=True)
    report['mlx_allocator_peak_bytes'] = int(mx.get_peak_memory())
    report['after'] = host_memory_snapshot()
    report['complete'] = True


try:
    run()
finally:
    if reader is not None:
        reader.close()
    if scales is not None:
        scales.clear()
    if bank is not None:
        bank.close()
    mx.synchronize()
    mx.clear_cache()
    report['active_after_close_bytes'] = int(mx.get_active_memory())
    spec = importlib.util.spec_from_file_location('owned_cache', 'scripts/deepseek_v41/reclaim_file_cache.py')
    reclaim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reclaim)
    paths = [model/'experts.bin']
    paths += [ROOT/'artifact'/c[f]['file'] for c in artifact['layers'][20]['components'].values()
              for f in ('descriptors','payload','bases')]
    reclaimed = [reclaim.reclaim_file(p) for p in paths]
    report['cleanup'] = dict(files=len(reclaimed), cached_page_bytes_before=sum(x['cached_page_bytes_before'] for x in reclaimed),
        cached_page_bytes_after=sum(x['cached_page_bytes_after'] for x in reclaimed))
    OUT.write_text(json.dumps(report, indent=2)+'\n')
if report['active_after_close_bytes'] > 2 * 1024**2:
    raise RuntimeError('probe retains Metal owners after explicit close')
print('PLANE_COMPLETE', json.dumps({k:report[k] for k in ('mlx_allocator_peak_bytes','active_after_close_bytes','cleanup')}), flush=True)
