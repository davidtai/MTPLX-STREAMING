"""Real output-projection ownership screen under a bounded parent GPU window."""
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import time

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent GPU guard required before MLX import')
signal.alarm(120)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'probe.json'
if OUT.exists():
    raise RuntimeError('refusing to overwrite the ownership screen')
proof = json.loads((ROOT / 'installation.json').read_text())
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
before = host_memory_snapshot()
bound = proof['incremental_bound_bytes']
if (not before['box']['ok'] or before['box']['used_bytes'] + bound > 110000000000
    or before['box']['wired_bytes'] + bound > 100 * 1024**3):
    raise RuntimeError('bounded projection screen cannot fit current memory')
MODEL = Path(proof['model_path'])
if hashlib.sha256((MODEL / 'expert-manifest.json').read_bytes()).hexdigest() != proof['model_manifest_sha256']:
    raise RuntimeError('native artifact manifest changed')

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mtplx.expert_manifest import load_expert_manifest
from mtplx.models import deepseek_v41 as dv
from owned_projection import BF16Output, FirstOutput, install_attention
from projection_reader import load_projection_tensors

mx.set_default_device(mx.gpu)
mx.set_memory_limit(2 * 1024**3)
mx.set_cache_limit(128 * 1024**2)
args = dv.ModelArgs.from_dict(json.loads((MODEL / 'config.json').read_text()))


def settle():
    mx.synchronize()
    gc.collect()
    mx.clear_cache()
    return {'active_bytes': int(mx.get_active_memory()),
            'peak_bytes': int(mx.get_peak_memory()),
            'cache_bytes': int(mx.get_cache_memory())}


def bits(a):
    return np.array(a.view(mx.uint8))


def run():
    attn = dv.Attention(args, 0)
    # This operator executes only native output projection. Never load or
    # evaluate unrelated query, KV, index or compressor parameters.
    for name in ('attn_sink', 'wq_a', 'q_norm_weight', 'wq_b', 'wkv',
                 'kv_norm_weight', 'compressor', 'indexer'):
        if name in attn:
            del attn[name]
    attn.wo_a = nn.QuantizedLinear(4096, 8192, bias=False, group_size=32, bits=8, mode='mxfp8')
    attn.wo_b = nn.QuantizedLinear(8192, 5120, bias=False, group_size=32, bits=8, mode='mxfp8')
    manifest = load_expert_manifest(MODEL / 'expert-manifest.json')
    wanted = set(proof['resident_names'])
    kept = tuple(t for t in manifest.resident_tensors if t.tensor in wanted)
    raw = load_projection_tensors(MODEL, manifest, kept, mx=mx)
    weights = {n.removeprefix('layers.0.attn.'): a for n, a in raw.items()}
    if set(dict(tree_flatten(attn.parameters()))) != set(weights):
        raise RuntimeError('output-only parameter inventory differs')
    attn.load_weights(list(weights.items()), strict=True)
    if any(a is not weights[n] for n, a in tree_flatten(attn.parameters())):
        raise RuntimeError('initial output weights remain owned')
    mx.eval(attn.parameters())
    del raw, weights, manifest
    rng = np.random.default_rng(419)
    inputs = {}
    for rows in (1, 6, 8):
        o = mx.array(rng.normal(0, 0.1, (1, rows, 64, 512)).astype(np.float32)).astype(mx.bfloat16)
        qcos, qsin = dv._cos_sin(attn.inv_freq, mx.arange(16384, 16384 + rows))
        mx.eval(o, qcos, qsin)
        inputs[rows] = (o, qcos, qsin, 1, rows)
    del o, qcos, qsin
    report['packed_only'] = settle()
    expected = {}
    for rows, operands in inputs.items():
        out = attn._out_prep_fused(*operands)
        mx.eval(out)
        expected[rows] = bits(out)
        del out
    report['native_warm'] = settle()
    # Exercise a cold first use of the candidate, using the same native builder.
    attn._wo_a_bf16T_cache = None
    attn._wo_a_dense_cache = None
    report['before_installation'] = settle()
    report['installation'] = install_attention(attn)
    if type(attn._out_prep_fused_impl) is not FirstOutput:
        raise RuntimeError('cold first-use route was not installed')
    mx.reset_peak_memory()
    for rows in (6, 1, 8):
        out = attn._out_prep_fused(*inputs[rows])
        mx.eval(out)
        actual = bits(out)
        exact = actual.shape == expected[rows].shape and actual.tobytes() == expected[rows].tobytes()
        report['cases'].append({'rows': rows, 'exact_output': exact,
                                'sha256': hashlib.sha256(actual.tobytes()).hexdigest()})
        if not exact:
            raise RuntimeError('single-owner output projection changed native bytes')
        del out, actual
    if ('wo_a' in attn or type(attn._out_prep_fused_impl) is not BF16Output
        or attn._wo_a_bf16T_cache is not None or attn._wo_a_dense_cache is not None):
        raise RuntimeError('old packed/cache ownership remains installed')
    report['candidate_warm'] = settle()
    released = report['native_warm']['active_bytes'] - report['candidate_warm']['active_bytes']
    report['released_active_bytes'] = released
    if released != 34603008:
        raise RuntimeError('packed projection release differs from exact physical accounting')
    report['scaled_40_layer_release_bytes'] = 40 * released
    report['complete'] = True


report = {'scope': 'Native real layer0 output projection only; one cold ownership transfer and exact M1/M6/M8 outputs. Memory proof, not full-model throughput or capacity evidence.',
          'construction': proof, 'initial_memory': before, 'cases': [], 'complete': False}
try:
    run()
finally:
    report['after_close'] = settle()
    report['final_memory'] = host_memory_snapshot()
    OUT.write_text(json.dumps(report, indent=2) + '\n')
if report['after_close']['active_bytes'] > 2 * 1024**2:
    raise RuntimeError('unexpected live tensors after closing the ownership screen')
print('PROJECTION_OWNERSHIP', json.dumps({k: report[k] for k in ('complete', 'released_active_bytes', 'scaled_40_layer_release_bytes', 'native_warm', 'candidate_warm', 'after_close')}), flush=True)
