"""Bounded real-weight phase transition, slot/expert identity and I/O check."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent GPU guard required')
signal.alarm(180)
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
before = host_memory_snapshot()
BOUND = 8 * 1024**3
if not before['box']['ok'] or before['box']['used_bytes'] + BOUND > 109500000000:
    raise RuntimeError('bounded bank, host and compiler memory do not fit')
root = Path(__file__).resolve().parent
output = root / 'storage-probe.json'
if output.exists():
    raise RuntimeError('refusing to overwrite evidence')
artifact = json.loads((root / 'artifact/manifest.json').read_text())
if not artifact['complete'] or artifact['packed_bytes'] != 3086136060:
    raise RuntimeError('complete inventory changed')
import numpy as np
import mlx.core as mx
from mtplx.expert_manifest import load_expert_manifest
from mtplx.expert_io import PositionalExpertReader
from mtplx.models.expert_mlx import MlxComponentBank, MlxComponentSlot, _run_component_bank_q4
from packed_storage import load_layer, remove_raw_scales, bind_weight_reader, make_dispatch, WEIGHT_BYTES

mx.set_memory_limit(4 * 1024**3)
mx.set_cache_limit(256 * 1024**2)
model = Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
manifest_path = model / 'expert-manifest.json'
if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != artifact['source_manifest_sha256']:
    raise RuntimeError('source manifest changed')
manifest = load_expert_manifest(manifest_path)
records = {r.expert: r for r in manifest.records if r.layer == 20}
source_stat = (model / 'experts.bin').stat()
if {k: getattr(source_stat, k) for k in artifact['source_identity']} != artifact['source_identity']:
    raise RuntimeError('verified source bank identity changed')
reader = PositionalExpertReader(model, bypass_page_cache=True, use_native=False, io_read_fanout=4)
bank = MlxComponentBank(capacity=103, record=records[0], label='packed-persistent-proof')
transient = MlxComponentBank(capacity=48, record=records[0], label='packed-transient-proof')
slots = [MlxComponentSlot(bank, (i * 7 + 11) % 103, label=f'proof-{i}') for i in range(36)]
transients = [MlxComponentSlot(transient, i * 11 + 4, label=f'transient-proof-{i}') for i in range(3)]
experts = [(i * 73 + 127) % 384 for i in range(36)]
transient_experts = [383, 211, 0]
loaded = tuple(zip(experts, slots)) + tuple(zip(transient_experts, transients))
hashes = {}


def weight_hash(slot):
    h = hashlib.sha256()
    for name in ('gate_proj.weight', 'up_proj.weight', 'down_proj.weight'):
        view = slot.component_view(name)
        h.update(view)
        view.release()
    return h.hexdigest()


for expert, slot in loaded:
    reader.read_record_into(manifest, records[expert], slot, verify_hash=True)
    hashes[slot.label] = weight_hash(slot)
mx.synchronize()
rng = np.random.default_rng(419)
cases = []
for rows, unique, pool, expert_ids in (
    (6, 6, slots, experts), (18, 3, transients, transient_experts),
    (36, 12, slots, experts), (36, 36, slots, experts),
):
    bindings = tuple(SimpleNamespace(buffer=pool[i % unique], expert=expert_ids[i % unique]) for i in range(rows))
    x = mx.array(rng.standard_normal((rows, 5120)).astype(np.float32)).astype(mx.bfloat16)
    ref = _run_component_bank_q4(x, bindings, group_size=32, bits=4, swiglu_limit=10.0, codec='mxfp4')
    mx.eval(ref, x)
    cases.append((x, bindings, np.array(ref.view(mx.uint16)), rows, unique))
del ref
mx.synchronize()
mx.clear_cache()
active_before = mx.get_active_memory()
released = remove_raw_scales(bank, mx=mx) + remove_raw_scales(transient, mx=mx)
for _, slot in loaded:
    slot.nbytes = WEIGHT_BYTES
if released != (103 + 48) * 1105920:
    raise RuntimeError('released scale backing differs from static geometry')
scales = load_layer(root / 'artifact', artifact['layers'][20], mx=mx)
dispatch = make_dispatch(scales, mx=mx)
bind_weight_reader(reader)
metrics_before = reader.metrics.as_dict()
for expert, slot in loaded[:36]:
    reader.read_record_into(manifest, records[expert], slot, verify_hash=False)
reader.read_component_records_into(manifest, tuple((records[e], s) for e, s in loaded[36:]), verify_hash=False)
for _, slot in loaded:
    if weight_hash(slot) != hashes[slot.label]:
        raise RuntimeError('weight-only read changed native weight bytes')
metrics_after = reader.metrics.as_dict()
if (metrics_after['read_bytes'] - metrics_before['read_bytes'] != len(loaded) * WEIGHT_BYTES
    or metrics_after['python_preadv_invocations'] - metrics_before['python_preadv_invocations'] != len(loaded) * 3):
    raise RuntimeError('weight-only physical byte/call accounting differs')
parity = []
for x, bindings, reference, rows, unique in cases:
    result = dispatch(x, bindings)
    mx.eval(result)
    bits = np.array(result.view(mx.uint16))
    if not np.array_equal(reference, bits):
        raise RuntimeError(f'packed scale mismatch with nonidentity slots: {rows}/{unique}')
    parity.append({'rows': rows, 'unique': unique, 'exact_bytes': True,
                   'output_sha256': hashlib.sha256(bits.tobytes()).hexdigest()})
    print('STORAGE_PARITY', rows, unique, 'exact', flush=True)
mx.synchronize()
mx.clear_cache()
after_transition = int(mx.get_active_memory())
packed_bytes = artifact['layers'][20]['packed_bytes']
if after_transition > active_before - released + packed_bytes + 32 * 1024**2:
    raise RuntimeError('phase allocation exceeds released raw plus exact packed ownership')
report = {
    'scope': 'physical persistent103/transient48 banks, full384-expert layer20 packed scales; exact synthetic-input MLP outputs and byte ownership, not full-model throughput',
    'source_commit': os.environ['DSV41_SOURCE_COMMIT'],
    'helper_sha256': {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in ('storage_probe.py', 'packed_storage.py', 'paired_kernels.py', 'kernels.py')},
    'artifact_manifest_sha256': hashlib.sha256((root / 'artifact/manifest.json').read_bytes()).hexdigest(),
    'static_increment_bound_bytes': BOUND, 'memory_before': before,
    'released_raw_scale_bytes': released, 'resident_layer_packed_bytes': packed_bytes,
    'active_before_transition_bytes': active_before, 'active_after_transition_bytes': after_transition,
    'read_bytes': metrics_after['read_bytes'] - metrics_before['read_bytes'],
    'preadv_calls': metrics_after['python_preadv_invocations'] - metrics_before['python_preadv_invocations'],
    'all_weight_hashes_exact': True, 'parity': parity,
    'mlx_allocator_peak_bytes': int(mx.get_peak_memory()), 'memory_after': host_memory_snapshot(),
}
del dispatch, scales, result, x, bindings, cases
bank.close()
transient.close()
reader.close()
mx.synchronize()
mx.clear_cache()
report['active_after_close_bytes'] = int(mx.get_active_memory())
if report['active_after_close_bytes'] > 2 * 1024**2:
    raise RuntimeError('bank or scale owner survives explicit close')
report['complete'] = True
output.write_text(json.dumps(report, indent=2) + '\n')
print('STORAGE_COMPLETE', json.dumps({k: report[k] for k in ('released_raw_scale_bytes', 'resident_layer_packed_bytes', 'mlx_allocator_peak_bytes', 'active_after_close_bytes')}), flush=True)
