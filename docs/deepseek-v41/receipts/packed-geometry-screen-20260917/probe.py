"""Bounded real-weight whole-MLP screen for independent packed-kernel changes."""
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
signal.alarm(240)
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
ROOT = Path(__file__).resolve().parent
output = ROOT / 'probe.json'
if output.exists():
    raise RuntimeError('refusing to overwrite probe evidence')
proof = json.loads((ROOT / 'construction.json').read_text())
before = host_memory_snapshot()
if (not before['box']['ok']
    or before['box']['used_bytes'] + proof['static_incremental_bound_bytes'] > 109500000000
    or before['box']['wired_bytes'] + proof['static_incremental_bound_bytes'] > 100 * 1024**3):
    raise RuntimeError('bounded host/Metal/cache/compiler allocation does not fit')
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if source != proof['source_commit']:
    raise RuntimeError('source commit differs from construction proof')
for name, digest in proof['unchanged_helpers_sha256'].items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError(f'control helper changed: {name}')
if hashlib.sha256((ROOT / 'artifact/manifest.json').read_bytes()).hexdigest() != proof['artifact_manifest_sha256']:
    raise RuntimeError('packed inventory changed')
artifact = json.loads((ROOT / 'artifact/manifest.json').read_text())
import numpy as np
import mlx.core as mx
from mtplx.expert_manifest import load_expert_manifest
from mtplx.expert_io import PositionalExpertReader
from mtplx.models.expert_mlx import MlxComponentBank, MlxComponentSlot, _run_component_bank_q4
from packed_storage import load_layer, remove_raw_scales, make_dispatch, WEIGHT_BYTES
from geometry_kernels import make_dispatch as make_variant

mx.set_memory_limit(2 * 1024**3)
mx.set_cache_limit(256 * 1024**2)
model = Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
manifest_path = model / 'expert-manifest.json'
if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != artifact['source_manifest_sha256']:
    raise RuntimeError('source manifest changed')
source_stat = (model / 'experts.bin').stat()
if {k: getattr(source_stat, k) for k in artifact['source_identity']} != artifact['source_identity']:
    raise RuntimeError('source weight identity changed')
manifest = load_expert_manifest(manifest_path)
records = {r.expert: r for r in manifest.records if r.layer == 20}
bank = reader = scales = None
report = dict(scope='real layer20 weights, nonidentity physical slots/expert IDs, whole MLP; synthetic BF16 inputs; not full-model performance',
    source_commit=source, construction=proof, before=before,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    kernel_sha256=hashlib.sha256((ROOT / 'geometry_kernels.py').read_bytes()).hexdigest(), cases=[])


def run():
    global bank, reader, scales
    reader = PositionalExpertReader(model, bypass_page_cache=True, use_native=False, io_read_fanout=4)
    bank = MlxComponentBank(capacity=48, record=records[0], label='packed-geometry-proof')
    slots = [MlxComponentSlot(bank, (i * 17 + 11) % 48, label=f'proof-{i}') for i in range(36)]
    experts = [(i * 73 + 127) % 384 for i in range(36)]
    for expert, slot in zip(experts, slots):
        reader.read_record_into(manifest, records[expert], slot, verify_hash=True)
    report['record_sha256'] = {str(e): records[e].sha256 for e in experts}
    rng = np.random.default_rng(419)
    cases = []
    for rows, unique in [(6, 6), (18, 3), (36, 12), (36, 36)]:
        bindings = tuple(SimpleNamespace(buffer=slots[i % unique], expert=experts[i % unique]) for i in range(rows))
        x = mx.array(rng.standard_normal((rows, 5120)).astype(np.float32)).astype(mx.bfloat16)
        reference = _run_component_bank_q4(x, bindings, group_size=32, bits=4, swiglu_limit=10.0, codec='mxfp4')
        mx.eval(reference, x)
        cases.append((rows, unique, x, bindings, np.array(reference.view(mx.uint16))))
    del reference
    mx.synchronize()
    released = remove_raw_scales(bank, mx=mx)
    if released != 48 * 1105920:
        raise RuntimeError('raw scale release differs from static accounting')
    for slot in slots:
        slot.nbytes = WEIGHT_BYTES
    scales = load_layer(ROOT / 'artifact', artifact['layers'][20], mx=mx)
    dispatches = {
        'control': make_dispatch(scales, mx=mx),
        'r8_sg2': make_variant(scales, results=8, simdgroups=2),
        'r4_sg4': make_variant(scales, results=4, simdgroups=4),
        'float_fp4': make_variant(scales, results=4, simdgroups=2, fp4_float_bits=True),
    }
    report['released_raw_scales_bytes'] = released
    report['resident_packed_layer_bytes'] = artifact['layers'][20]['packed_bytes']
    for rows, unique, x, bindings, reference in cases:
        case = dict(rows=rows, unique_experts=unique, parity={}, timings=[])
        passing = []
        for mode, dispatch in dispatches.items():
            y = dispatch(x, bindings)
            mx.eval(y)
            bits = np.array(y.view(mx.uint16))
            equal = bool(np.array_equal(bits, reference))
            case['parity'][mode] = dict(exact_bytes=equal, differing_elements=int(np.count_nonzero(bits != reference)),
                                       output_sha256=hashlib.sha256(bits.tobytes()).hexdigest())
            if mode == 'control' and not equal:
                raise RuntimeError('retained packed control differs from native arithmetic')
            if equal:
                passing.append(mode)
        for _ in range(20):
            mx.eval(dispatches['control'](x, bindings))
        candidates = [mode for mode in passing if mode != 'control']
        schedule = ['control', *candidates, 'control', *reversed(candidates), 'control', *candidates, 'control']
        for mode in schedule:
            dispatch = dispatches[mode]
            mx.eval(dispatch(x, bindings))
            samples = []
            for _ in range(15):
                start = time.perf_counter_ns()
                mx.eval(dispatch(x, bindings))
                samples.append((time.perf_counter_ns() - start) / 1e9)
            case['timings'].append(dict(arm=mode, median_s=statistics.median(samples), samples_s=samples))
        case['arm_medians_s'] = {mode: statistics.median(t['median_s'] for t in case['timings'] if t['arm'] == mode) for mode in passing}
        case['latency_reduction_pct'] = {mode: 100 * (1 - case['arm_medians_s'][mode] / case['arm_medians_s']['control']) for mode in candidates}
        report['cases'].append(case)
        output.write_text(json.dumps(report, indent=2) + '\n')
        print('GEOMETRY_CASE', json.dumps({key: case[key] for key in ['rows', 'unique_experts', 'parity', 'arm_medians_s', 'latency_reduction_pct']}), flush=True)
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
    cache_spec = importlib.util.spec_from_file_location('owned_cache', 'scripts/deepseek_v41/reclaim_file_cache.py')
    cache = importlib.util.module_from_spec(cache_spec)
    cache_spec.loader.exec_module(cache)
    paths = [model / 'experts.bin']
    paths += [ROOT / 'artifact' / component[field]['file'] for component in artifact['layers'][20]['components'].values()
              for field in ('descriptors', 'payload', 'bases')]
    reclaimed = [cache.reclaim_file(path) for path in paths]
    report['cleanup'] = dict(files=len(reclaimed), cached_page_bytes_before=sum(x['cached_page_bytes_before'] for x in reclaimed),
                             cached_page_bytes_after=sum(x['cached_page_bytes_after'] for x in reclaimed))
    output.write_text(json.dumps(report, indent=2) + '\n')
if report['active_after_close_bytes'] > 2 * 1024**2:
    raise RuntimeError('operator arrays remain after explicit close')
print('GEOMETRY_COMPLETE', json.dumps({key: report[key] for key in ['mlx_allocator_peak_bytes', 'active_after_close_bytes', 'cleanup']}), flush=True)
