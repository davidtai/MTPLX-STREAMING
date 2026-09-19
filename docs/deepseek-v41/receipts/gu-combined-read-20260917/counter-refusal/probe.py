"""Bounded CPU read screen; compare the installed plane reader with GU scatter."""
import gc
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent guard must hold the GPU/service lock')
signal.alarm(120)


class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('the CPU read screen must not import MLX')


sys.meta_path.insert(0, NoMLX())
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
from mtplx.expert_io import ExpertIOMetrics, PositionalExpertReader
from mtplx.expert_manifest import load_expert_manifest

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'probe.json'
ARTIFACT = Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
MANIFEST = ARTIFACT / 'expert-manifest.json'
MANIFEST_SHA = '44c340845990b4ec21342d914204d69eb9e7cf371d6b18944f8dfd2e945df3e9'
WEIGHT = 5898240
GAP = 368640
RECORD = 18800640
ALLOWANCE = 2 * 1024**3
NAMES = ('gate_proj.weight', 'up_proj.weight', 'down_proj.weight')
if OUT.exists() or hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != MANIFEST_SHA:
    raise RuntimeError('output exists or native manifest identity changed')
before = host_memory_snapshot()
if (not before['box']['ok'] or before['box']['used_bytes'] + ALLOWANCE > 110000000000
    or before['box']['wired_bytes'] + ALLOWANCE > 100 * 1024**3):
    raise RuntimeError('bounded CPU read screen cannot fit current memory')
source_identity = {k: getattr((ARTIFACT / 'experts.bin').stat(), k)
                   for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}
if source_identity['st_size'] != 288777830400:
    raise RuntimeError('native expert sidecar size differs')
manifest = load_expert_manifest(MANIFEST)
records = {(r.layer, r.expert): r for r in manifest.records}
if set(records) != {(l, e) for l in range(40) for e in range(384)}:
    raise RuntimeError('native 40x384 record inventory differs')
if any(tuple(s.length for s in r.segments) != (WEIGHT, GAP) * 3 or not r.sha256
       for r in records.values()):
    raise RuntimeError('native component lengths or record digests differ')


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native = load_helper('native_reader')
combined = load_helper('combined_reader')


class Slot:
    def __init__(self):
        self.arrays = {name: bytearray(WEIGHT) for name in NAMES}
        self.arrays['gate_gap'] = bytearray(GAP)

    def component_view(self, name):
        return memoryview(self.arrays[name])

    def digest(self):
        h = hashlib.sha256()
        for name in NAMES:
            h.update(self.arrays[name])
        return h.hexdigest()


slots = tuple(Slot() for _ in range(6))
report = {
    'scope': 'CPU read-batch screen with native reader/fanout and gate/up readiness. No Metal owners, target model, cache pressure or decode TPS claim.',
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'manifest_sha256': MANIFEST_SHA, 'source_identity': source_identity,
    'whole_machine_ceiling_bytes': 110000000000, 'incremental_allowance_bytes': ALLOWANCE,
    'weight_buffer_bytes': 6 * 3 * WEIGHT, 'scratch_buffer_bytes': 6 * GAP,
    'candidate_extra_read_bytes_per_record': GAP,
    'initial_memory': before, 'arms': [], 'complete': False,
}
try:
    for size in (1, 3, 6):
        batches = tuple(tuple(records[(i * 7) % 40, (i * 11 + j * 61) % 384]
                              for j in range(size)) for i in range(128))
        expected = None
        for arm, helper in (('control_before', native), ('combined', combined), ('control_after', native)):
            reader = PositionalExpertReader(ARTIFACT, bypass_page_cache=True,
                                            use_native=False, io_read_fanout=4)
            local = threading.local()
            helper.bind_reader(reader, local)
            try:
                def read_batch(batch):
                    local.part = helper.PlanePart(SimpleNamespace(loads=batch))
                    reader.read_component_records_into(manifest, tuple(zip(batch, slots)), verify_hash=False)

                for batch in batches[:4]:
                    read_batch(batch)
                reader.metrics = ExpertIOMetrics()
                started = time.perf_counter()
                for batch in batches:
                    read_batch(batch)
                wall = time.perf_counter() - started
                if not local.part.gate_up_ready.done():
                    raise RuntimeError('gate/up publication is missing')
                metrics = reader.metrics.as_dict()
                hashes = [s.digest() for s in slots[:size]]
                if expected is None:
                    expected = hashes
                if hashes != expected:
                    raise RuntimeError('candidate weight bytes differ from native plane reads')
                per_record = 3 * WEIGHT + (GAP if arm == 'combined' else 0)
                calls = 2 if arm == 'combined' else 3
                if (metrics['read_bytes'] != len(batches) * size * per_record
                    or metrics['records_read'] != len(batches) * size
                    or metrics['python_preadv_invocations'] != len(batches) * size * calls
                    or any(metrics[k] for k in ('short_reads', 'integrity_errors', 'io_errors', 'cancellations', 'deadline_errors'))):
                    raise RuntimeError('read coverage or error counters differ')
                row = {'records_per_batch': size, 'arm': arm, 'batches': len(batches),
                       'fanout': 4, 'worker_capacity': reader._fanout_pool_workers,
                       'wall_s': wall, 'payload_gb_per_s': len(batches) * size * 3 * WEIGHT / wall / 1e9,
                       'weight_sha256': hashes, 'metrics': metrics,
                       'memory': host_memory_snapshot()}
                report['arms'].append(row)
                print('READ_SCREEN', json.dumps({k: row[k] for k in ('records_per_batch', 'arm', 'wall_s', 'payload_gb_per_s')}), flush=True)
            finally:
                reader.close()
                del local.part
        # Verify the final reference records against the native artifact digest,
        # including all three scale ranges, outside every measured arm.
        raw = tuple(bytearray(n) for n in (WEIGHT, GAP) * 3)
        verifier = PositionalExpertReader(ARTIFACT, bypass_page_cache=True, use_native=False)
        try:
            for index, record in enumerate(batches[-1]):
                views = tuple(memoryview(a) for a in raw)
                try:
                    verifier._readv_range_into('experts.bin', record.sidecar_offset, views,
                                              cancel_event=None, deadline_ns=None)
                finally:
                    for view in views:
                        view.release()
                h = hashlib.sha256()
                weights = hashlib.sha256()
                for i, array in enumerate(raw):
                    h.update(array)
                    if i % 2 == 0:
                        weights.update(array)
                if h.hexdigest() != record.sha256 or weights.hexdigest() != expected[index]:
                    raise RuntimeError('reference weights do not match the full native record')
        finally:
            verifier.close()
            del raw
    after_identity = {k: getattr((ARTIFACT / 'experts.bin').stat(), k) for k in source_identity}
    if after_identity != source_identity:
        raise RuntimeError('expert source identity changed during the read screen')
    report['complete'] = True
finally:
    del slots
    gc.collect()
    report['final_memory'] = host_memory_snapshot()
    report['mlx_imported'] = any(n == 'mlx' or n.startswith('mlx.') for n in sys.modules)
    OUT.write_text(json.dumps(report, indent=2) + '\n')
