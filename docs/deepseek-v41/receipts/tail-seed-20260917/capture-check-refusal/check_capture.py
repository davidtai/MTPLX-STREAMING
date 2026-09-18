"""Small native-backbone capture check after the real-weight seed memory win."""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent GPU guard required before MLX import')
signal.alarm(120)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'capture-check.json'
if OUT.exists():
    raise RuntimeError('refusing to overwrite capture evidence')
proof = json.loads((ROOT / 'construction.json').read_text())
if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != proof['source_commit']:
    raise RuntimeError('source commit changed')
if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True):
    raise RuntimeError('tracked source is dirty')
for name, digest in proof['runtime_source_sha256'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('native capture source changed')
for name, digest in proof['helper_sha256'].items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('capture helper changed')
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
before = host_memory_snapshot()
if (not before['box']['ok'] or before['box']['used_bytes'] + 4 * 1024**3 > 110000000000
    or before['box']['wired_bytes'] + 4 * 1024**3 > 100 * 1024**3):
    raise RuntimeError('small capture check cannot fit current memory')
import mlx.core as mx
import numpy as np
from mlx.utils import tree_map
from mtplx.models import deepseek_v41 as dv
from tail_prefill import build_forward

spec = importlib.util.spec_from_file_location('native_prefill_fixture', 'tests/models/test_deepseek_v41_layer_major_prefill.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
mx.set_default_device(mx.gpu)
mx.set_memory_limit(1024**3)
mx.set_cache_limit(64 * 1024**2)


def exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def run():
    for dtype in (mx.float32, mx.bfloat16):
        args = fixture._csa_args(dspark_target_layer_ids=[5, 6, 7])
        model = dv.Model(args)
        fixture._randomize(model, seed=419)
        model.update(tree_map(lambda a: a.astype(dtype), model.parameters()))
        mx.eval(model.parameters())
        ids = mx.array([[(i * 13 + 5) % args.vocab_size for i in range(33)]])
        original_type = type(model.model)
        control_cache = model.make_cache()
        logits, hidden = model(ids, cache=control_cache, return_hidden=True,
                               prefill_chunk=7, prefill_layer_major=True)
        mx.eval(logits, hidden)
        control_logits = np.array(logits)
        control_hidden = np.array(hidden)
        control_state, control_offset = fixture._cache_snapshot(control_cache)
        control_next = model(mx.array([[3]]), cache=control_cache)
        mx.eval(control_next)
        next_bits = np.array(control_next)
        del logits, hidden, control_cache, control_next
        for keep in (7, 17):
            cls = type('CaptureTailProof', (original_type,), {'_forward_layer_major': build_forward(keep)})
            object.__setattr__(model.model, '__class__', cls)
            cache = model.make_cache()
            logits, hidden = model(ids, cache=cache, return_hidden=True,
                                   prefill_chunk=7, prefill_layer_major=True)
            mx.eval(logits, hidden)
            state, offset = fixture._cache_snapshot(cache)
            equal_logits = exact(np.array(logits), control_logits)
            equal_hidden = exact(np.array(hidden), control_hidden[:, -keep:])
            equal_cache = offset == control_offset == 33 and set(state) == set(control_state)
            equal_cache = equal_cache and all(exact(state[k], control_state[k]) for k in state)
            nxt = model(mx.array([[3]]), cache=cache)
            mx.eval(nxt)
            equal_next = exact(np.array(nxt), next_bits)
            row = {'dtype': str(dtype), 'prompt_rows': 33, 'chunk_rows': 7,
                   'keep_rows': keep, 'hidden_shape': list(hidden.shape),
                   'exact_logits': equal_logits, 'exact_hidden_tail': equal_hidden,
                   'exact_cache': bool(equal_cache), 'exact_next_step': equal_next}
            report['cases'].append(row)
            print('TAIL_CAPTURE_CHECK', json.dumps(row), flush=True)
            if not all((equal_logits, equal_hidden, equal_cache, equal_next)):
                raise RuntimeError('tail capture changed the target path')
            del cache, logits, hidden, nxt
        del model, ids
    report['complete'] = True


report = {'scope': 'Four small actual eight-layer native backbone cases after the successful real-weight seed memory screen; target logits/cache/next step and retained hiddens checked bytewise. No full-model throughput claim.',
          'construction': proof, 'before': before, 'cases': [], 'complete': False}
try:
    run()
finally:
    gc.collect()
    mx.synchronize()
    report['allocator_peak_bytes'] = int(mx.get_peak_memory())
    mx.clear_cache()
    report['active_after_close_bytes'] = int(mx.get_active_memory())
    OUT.write_text(json.dumps(report, indent=2) + '\n')
if report['active_after_close_bytes'] > 2 * 1024**2:
    raise RuntimeError('capture check left unexpected tensor owners')
print('TAIL_CAPTURE_COMPLETE', report['allocator_peak_bytes'], report['active_after_close_bytes'], flush=True)
