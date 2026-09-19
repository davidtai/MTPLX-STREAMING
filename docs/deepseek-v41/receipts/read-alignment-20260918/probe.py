"""Bounded CPU read screen; compare source-matched and page-aligned destination offsets."""
import ctypes
import mmap
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
PAGE = os.sysconf("SC_PAGESIZE")
if PAGE != 16384: raise RuntimeError("expected native16KiB page geometry")


class Slot:
    def __init__(self):
        self.arrays = {name: mmap.mmap(-1,WEIGHT+PAGE) for name in NAMES}
        self.offsets = {name:0 for name in NAMES}
        self.addresses = {name:ctypes.addressof(ctypes.c_ubyte.from_buffer(array))
                          for name,array in self.arrays.items()}
        if any(address%PAGE for address in self.addresses.values()):
            raise RuntimeError('CPU control buffer is not page-aligned')

    def configure(self,record,matched):
        for name,offset in zip(NAMES,(0,6266880,12533760)):
            self.offsets[name]=(record.sidecar_offset+offset)%PAGE if matched else 0

    def component_view(self,name):
        offset=self.offsets[name]
        return memoryview(self.arrays[name])[offset:offset+WEIGHT]

    def digest(self):
        h=hashlib.sha256()
        for name in NAMES:
            with self.component_view(name) as view:h.update(view)
        return h.hexdigest()

    def close(self):
        for array in self.arrays.values():array.close()


slots = tuple(Slot() for _ in range(3))
report = {
    'scope': 'CPU native plane reader: page-aligned mmap destinations versus offsets matching source modulo16KiB. Same three plane reads and fanout. No MLX imports, Metal owners, model or full TPS claim; actual Metal base alignment is not measured by this screen.',
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'manifest_sha256': MANIFEST_SHA, 'source_identity': source_identity,
    'whole_machine_ceiling_bytes': 110000000000, 'incremental_allowance_bytes': ALLOWANCE,
    'weight_buffer_bytes': 3 * 3 * WEIGHT, 'alignment_padding_bytes': 3 * 3 * PAGE,
    'candidate_extra_read_bytes_per_record': 0,
    'initial_memory': before, 'arms': [], 'complete': False,
}
try:
    for size in (3,):
        batches = tuple(tuple(records[(i * 7) % 40, (i * 11 + j * 61) % 384]
                              for j in range(size)) for i in range(128))
        expected = None
        for arm, helper in (('page_before',native),('source_matched1',native),('page_middle',native),('source_matched2',native),('page_after',native)):
            reader = PositionalExpertReader(ARTIFACT, bypass_page_cache=True,
                                            use_native=False, io_read_fanout=4)
            local = threading.local()
            helper.bind_reader(reader, local)
            try:
                def read_batch(batch):
                    for slot,record in zip(slots,batch):slot.configure(record,arm.startswith('source_matched'))
                    local.part = helper.PlanePart(SimpleNamespace(loads=batch))
                    reader.read_component_records_into(manifest, tuple(zip(batch, slots)), verify_hash=False)

                for batch in batches[:4]:
                    read_batch(batch)
                reader.metrics = ExpertIOMetrics()
                # Both the installed record reader and range reader must share
                # this fresh counter object after unmeasured warmup.
                helper.bind_reader(reader, local)
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
                per_record = 3 * WEIGHT
                calls = 3
                report.setdefault('coverage', []).append({'arm': arm, 'size': size, 'wall_s': wall, 'metrics': metrics})
                if (metrics['read_bytes'] != len(batches) * size * per_record
                    or metrics['records_read'] != len(batches) * size
                    or metrics['python_preadv_invocations'] != len(batches) * size * calls
                    or any(metrics[k] for k in ('short_reads', 'integrity_errors', 'io_errors', 'cancellations', 'deadline_errors'))):
                    raise RuntimeError('read coverage or error counters differ')
                row = {'records_per_batch': size, 'arm': arm, 'batches': len(batches),
                       'fanout': 4, 'worker_capacity': reader._fanout_pool_workers,
                       'last_buffer_offsets': [slot.offsets.copy() for slot in slots[:size]],
                       'source_mod_page_counts': {str(mod):sum((record.sidecar_offset+offset)%PAGE==mod for batch in batches for record in batch for offset in (0,6266880,12533760)) for mod in (0,8192)},
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
    for slot in slots:slot.close()
    del slots
    gc.collect()
    report['final_memory'] = host_memory_snapshot()
    report['mlx_imported'] = any(n == 'mlx' or n.startswith('mlx.') for n in sys.modules)
    OUT.write_text(json.dumps(report, indent=2) + '\n')
