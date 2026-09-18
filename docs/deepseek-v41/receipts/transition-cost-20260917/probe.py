"""Attribute the existing 84-to-102 packed bank transition on one real layer."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent GPU guard required before MLX import')
signal.alarm(180)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'probe.json'
if OUT.exists():
    raise RuntimeError('refusing to overwrite evidence')
proof = json.loads((ROOT / 'construction.json').read_text())
if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != proof['source_commit']:
    raise RuntimeError('source commit changed')
for name, expected in proof['helper_sha256'].items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('transition helper changed')
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
before = host_memory_snapshot()
bound = proof['incremental_bound_bytes']
if (not before['box']['ok'] or before['box']['used_bytes'] + bound > 110000000000
    or before['box']['wired_bytes'] + bound > 100 * 1024**3):
    raise RuntimeError('bounded transition cannot fit the current machine')
blob = (ROOT / 'artifact/manifest.json').read_bytes()
if hashlib.sha256(blob).hexdigest() != proof['artifact_manifest_sha256']:
    raise RuntimeError('packed inventory changed')
inventory = json.loads(blob)
MODEL = Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
st = (MODEL / 'experts.bin').stat()
if {k: getattr(st, k) for k in inventory['source_identity']} != inventory['source_identity']:
    raise RuntimeError('weight source identity changed')
if hashlib.sha256((MODEL / 'expert-manifest.json').read_bytes()).hexdigest() != inventory['source_manifest_sha256']:
    raise RuntimeError('native manifest changed')

import mlx.core as mx
from mtplx.expert_manifest import load_expert_manifest
from mtplx.expert_io import PositionalExpertReader
from mtplx.models.expert_mlx import MlxComponentBank, MlxComponentSlot
from packed_storage import load_layer, remove_raw_scales, WEIGHT_BYTES
from bank_growth_final import grow_bank

mx.set_memory_limit(4 * 1024**3)
mx.set_cache_limit(256 * 1024**2)
bank = reader = scales = None
report = {'scope': 'One real layer20 bank at native 84-to-102 geometry; transition timing attribution only, no target execution or throughput claim',
          'construction': proof, 'before': before, 'complete': False}


def weight_hash(slot):
    h = hashlib.sha256()
    for name in ('gate_proj.weight', 'up_proj.weight', 'down_proj.weight'):
        view = slot.component_view(name)
        try:
            h.update(view)
        finally:
            view.release()
    return h.hexdigest()


def run():
    global bank, reader, scales
    manifest = load_expert_manifest(MODEL / 'expert-manifest.json')
    records = {r.expert: r for r in manifest.records if r.layer == 20}
    bank = MlxComponentBank(capacity=84, record=records[0], label='transition-cost')
    reader = PositionalExpertReader(MODEL, bypass_page_cache=True, use_native=False, io_read_fanout=4)
    slots = [MlxComponentSlot(bank, row, label=f'cost-{row}') for row in (0, 41, 83)]
    for expert, slot in zip((127, 200, 273), slots):
        reader.read_record_into(manifest, records[expert], slot, verify_hash=True)
    hashes = [weight_hash(slot) for slot in slots]
    mx.synchronize()
    mx.clear_cache()
    report['before_transition_active_bytes'] = int(mx.get_active_memory())
    report['initial_peak_bytes'] = int(mx.get_peak_memory())
    mx.reset_peak_memory()
    start = time.perf_counter()
    released = remove_raw_scales(bank, mx=mx)
    after_release = time.perf_counter()
    scales = load_layer(ROOT / 'artifact', inventory['layers'][20], mx=mx)
    after_scales = time.perf_counter()
    added = grow_bank(bank, 102, mx=mx)
    after_growth = time.perf_counter()
    mx.synchronize()
    mx.clear_cache()
    end = time.perf_counter()
    report.update(
        timings_s={'release_raw_scales': after_release - start,
                   'load_and_hash_packed_scales': after_scales - after_release,
                   'grow_three_weight_components': after_growth - after_scales,
                   'final_sync_and_clear': end - after_growth,
                   'whole_transition': end - start},
        released_raw_scale_bytes=released,
        packed_layer_bytes=inventory['layers'][20]['packed_bytes'],
        added_weight_bytes=added,
        transition_peak_bytes=int(mx.get_peak_memory()),
        after_transition_active_bytes=int(mx.get_active_memory()),
        whole_allocator_peak_bytes=max(report['initial_peak_bytes'], int(mx.get_peak_memory())),
        native_record_sha256={str(e): records[e].sha256 for e in (127, 200, 273)})
    if released != 84 * 1105920 or added != 18 * WEIGHT_BYTES:
        raise RuntimeError('transition byte accounting changed')
    if [weight_hash(slot) for slot in slots] != hashes:
        raise RuntimeError('old row contents changed during growth')
    report['old_row_weight_sha256'] = dict(zip(('0', '41', '83'), hashes))
    report['old_rows_exact'] = True
    report['after'] = host_memory_snapshot()
    report['complete'] = True
    print('TRANSITION_COST', json.dumps(report['timings_s']), flush=True)


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
    spec = importlib.util.spec_from_file_location('cost_reclaim', 'scripts/deepseek_v41/reclaim_file_cache.py')
    reclaim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reclaim)
    paths = [MODEL / 'experts.bin'] + [ROOT / 'artifact' / c[f]['file']
        for c in inventory['layers'][20]['components'].values() for f in ('descriptors', 'payload', 'bases')]
    rows = [reclaim.reclaim_file(p) for p in paths]
    report['cleanup'] = {k: sum(r[k] for r in rows) for k in ('cached_page_bytes_before', 'cached_page_bytes_after')}
    OUT.write_text(json.dumps(report, indent=2) + '\n')
if report['active_after_close_bytes'] > 2 * 1024**2:
    raise RuntimeError('transition arrays remain after close')
print('TRANSITION_COST_COMPLETE', report['whole_allocator_peak_bytes'], report['active_after_close_bytes'], flush=True)
