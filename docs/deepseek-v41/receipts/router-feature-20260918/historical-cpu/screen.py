"""CPU-only causal router feature screen using the retained W35 trace.

Diagnostic only: W35's prompt and AR rows differ from the accepted M6 workload.
No MLX import, speculative I/O, model loading or production mutation.
"""
import os
os.environ['VECLIB_MAXIMUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import fcntl
import hashlib
import importlib.abc
import json
from pathlib import Path
import struct
import sys
import time
import signal

signal.alarm(120)

class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('CPU screen attempted MLX import')

sys.meta_path.insert(0, NoMLX())
import numpy as np
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot

ROOT = Path(__file__).resolve().parent
REPO = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
TRACE = REPO / '.benchmark-artifacts/deepseek-v41/route-traces-w35'
MODEL = Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
OUT = ROOT / 'screen.json'
if OUT.exists():
    raise RuntimeError('refusing evidence overwrite')
before = host_memory_snapshot()
if not before['box']['ok'] or before['box']['used_bytes'] + 512*1024**2 > 110000000000:
    raise RuntimeError('512MiB CPU envelope does not fit current machine use')
manifest = json.loads((TRACE / 'manifest.json').read_text())
assert manifest['n_rows'] == 2304 and manifest['n_decode'] == 256
assert manifest['stored_hidden'] == 5120 and manifest['n_experts'] == 384
index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
cfg = json.loads((MODEL / 'config.json').read_text())
cfg = cfg.get('text_config', cfg)
temp = float(cfg.get('gate_temp', 1.0) or 1.0)
score_func = cfg.get('scoring_func', 'sqrtsoftplus')
assert score_func == 'sqrtsoftplus'
digests = {}

def open_nocache(path):
    f = open(path, 'rb', buffering=0)
    fcntl.fcntl(f.fileno(), 48, 1)  # Darwin F_NOCACHE
    return f

def bf32(a):
    return (a.astype(np.uint32) << 16).view(np.float32)

def hidden(layer, kind):
    path = TRACE / f'layer{layer:03d}_{kind}.npy'
    with open_nocache(path) as f:
        v = np.lib.format.read_magic(f)
        assert v in ((1, 0), (2, 0))
        reader = np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0
        shape, order, dtype = reader(f)
        assert not order and shape[0] == 2304
        stride = int(np.prod(shape[1:])) * dtype.itemsize
        f.seek(2048*stride, 1)
        raw = f.read(256*stride)
        assert len(raw) == 256*stride
    digests[str(path.relative_to(TRACE)) + ':decode256'] = hashlib.sha256(raw).hexdigest()
    a = np.frombuffer(raw, dtype=dtype).reshape(256, *shape[1:])
    return bf32(a) if kind in ('router_in', 'layer_in') else a.copy()

def tensor(name):
    path = MODEL / index[name]
    with open_nocache(path) as f:
        n = struct.unpack('<Q', f.read(8))[0]
        assert n < 64*1024**2
        header = json.loads(f.read(n))
        e = header[name]
        lo, hi = e['data_offsets']
        assert hi-lo <= 8*1024**2
        f.seek(8+n+lo)
        raw = f.read(hi-lo)
        assert len(raw) == hi-lo
    digests[name] = hashlib.sha256(raw).hexdigest()
    if e['dtype'] == 'BF16':
        a = bf32(np.frombuffer(raw, np.uint16))
    elif e['dtype'] == 'F32':
        a = np.frombuffer(raw, np.float32).copy()
    else:
        raise RuntimeError(e['dtype'])
    return a.reshape(e['shape'])

def biased_scores(x, w, b):
    z = (x @ w.T) / temp
    # Stable softplus; CPU ranking must self-align before predictor claims.
    return np.sqrt(np.logaddexp(np.float32(0), z)) + b

def metrics(scores, ids):
    ranked = np.argsort(-scores, axis=1, kind='stable')
    truth = np.zeros_like(scores, dtype=bool)
    np.put_along_axis(truth, ids.astype(np.int64), True, axis=1)
    correct = np.take_along_axis(truth, ranked, axis=1)
    result = {}
    for k in (1, 2, 4, 6, 10, 12, 16):
        hits = int(correct[:, :k].sum())
        result[str(k)] = {'hits':hits, 'issued':len(ids)*k,
                          'truth':ids.size, 'precision':hits/(len(ids)*k),
                          'recall':hits/ids.size}
    return result

start = time.monotonic()
rows = []
prev_router = hidden(3, 'router_in')
prev_layer = hidden(3, 'layer_in')
prev_norm = tensor('layers.3.ffn_norm.weight')
for layer in range(4, 40):
    router = hidden(layer, 'router_in')
    layer_in = hidden(layer, 'layer_in')
    true = hidden(layer, 'top6')
    weight = tensor(f'layers.{layer}.ffn.gate.weight')
    bias = tensor(f'layers.{layer}.ffn.gate.bias')
    norm = tensor(f'layers.{layer}.ffn_norm.weight')
    assert np.all(np.isfinite(norm)) and np.min(np.abs(prev_norm)) > 1e-8
    features = {'self_alignment': router,
                'existing_pre_attention_mean': prev_layer,
                'post_attention_router': prev_router,
                'post_attention_norm_adjusted': prev_router * (norm / prev_norm)}
    result = {name: metrics(biased_scores(x, weight, bias), true)
              for name, x in features.items()}
    rows.append({'layer':layer, 'metrics':result})
    prev_router, prev_layer, prev_norm = router, layer_in, norm
    if layer % 8 == 7:
        print(json.dumps({'completed_through_layer':layer,'elapsed_s':time.monotonic()-start}), flush=True)

summary = {}
for name in rows[0]['metrics']:
    summary[name] = {}
    for k in rows[0]['metrics'][name]:
        hits = sum(r['metrics'][name][k]['hits'] for r in rows)
        issued = sum(r['metrics'][name][k]['issued'] for r in rows)
        truth = sum(r['metrics'][name][k]['truth'] for r in rows)
        summary[name][k] = {'hits':hits,'issued':issued,'truth':truth,
                            'precision':hits/issued,'recall':hits/truth}
alignment = summary['self_alignment']['6']['recall']
result = {'complete':True,'cpu_only':True,'static_incremental_bound_bytes':512*1024**2,
          'before':before,'after':host_memory_snapshot(),
          'trace_manifest':manifest,'same_prompt_as_acceptance':False,
          'scope':'256 historical AR rows, target layers4..39; candidate screens only',
          'source_commit':'013ba48db733659d287d40e7304e1683a0d7e179',
          'elapsed_s':time.monotonic()-start,'native_self_alignment':alignment,
          'cpu_alignment_usable':alignment > .995,'summary':summary,'rows':rows,
          'consumed_range_sha256':digests}
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'output':str(OUT),'elapsed_s':result['elapsed_s'],
                  'self_alignment':alignment,'at6':{n:v['6'] for n,v in summary.items()}}),flush=True)
