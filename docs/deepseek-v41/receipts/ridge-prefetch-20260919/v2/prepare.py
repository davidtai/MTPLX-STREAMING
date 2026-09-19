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

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent-held guard required for the CPU memory envelope')
ROOT = Path(__file__).resolve().parent
REPO = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
TRACE = REPO / '.benchmark-artifacts/deepseek-v41/route-traces-w35'
MODEL = Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
OUT = ROOT / 'preparation.json'
if OUT.exists():
    raise RuntimeError('refusing evidence overwrite')
before = host_memory_snapshot()
if not before['box']['ok'] or before['box']['used_bytes'] + 1024**3 > 110000000000:
    raise RuntimeError('1GiB CPU envelope does not fit current machine use')
manifest = json.loads((TRACE / 'manifest.json').read_text())
assert manifest['n_rows'] == 2304 and manifest['n_decode'] == 256
assert manifest['stored_hidden'] == 5120 and manifest['n_experts'] == 384
index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
cfg = json.loads((MODEL / 'config.json').read_text())
cfg = cfg.get('text_config', cfg)
temp = float(cfg.get('gate_temp', 1.0) or 1.0)
eps = float(cfg.get('rms_norm_eps', 1e-6))
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
        raw = f.read(2304*stride)
        assert len(raw) == 2304*stride
    digests[str(path.relative_to(TRACE)) + ':all2304'] = hashlib.sha256(raw).hexdigest()
    a = np.frombuffer(raw, dtype=dtype).reshape(2304, *shape[1:])
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

# Appended after the authenticated CPU readers; MLX remains prohibited.
import subprocess

installation = json.loads((ROOT / 'installation.json').read_text())
if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() != installation['source_commit']:
    raise RuntimeError('CPU preparation source changed')
reference_path = Path(installation['ridge_reference_path'])
blob = reference_path.read_bytes()
if hashlib.sha256(blob).hexdigest() != installation['ridge_reference_sha256']:
    raise RuntimeError('frozen ridge selection changed')
reference = json.loads(blob)
del blob
if hashlib.sha256((TRACE / 'manifest.json').read_bytes()).hexdigest() != installation['training_manifest_sha256']:
    raise RuntimeError('prompt training trace changed')
routes = json.loads((ROOT / 'routes.json').read_text())


def scores_from_z(z, bias):
    return np.sqrt(np.logaddexp(np.float32(0), z)) + bias


def fit(z, y, lam):
    mean = z.mean(axis=0)
    scale = np.maximum(z.std(axis=0), np.float32(0.1))
    center = y.mean(axis=0)
    design = (z - mean) / scale
    gram = design.T @ design
    gram.flat[::385] += np.float32(lam * len(z))
    coef = np.linalg.solve(gram, design.T @ (y - center))
    return mean, scale, center, coef


started = time.monotonic()
arrays = {}
rows = []
for layer in (31, 32):
    prior = next(r for r in reference['rows'] if r['layer'] == layer)
    previous = hidden(layer - 1, 'router_in')
    current = hidden(layer, 'router_in')
    weight = tensor(f'layers.{layer}.ffn.gate.weight')
    bias = tensor(f'layers.{layer}.ffn.gate.bias')
    z = (previous @ weight.T) / temp
    target_z = (current @ weight.T) / temp
    baseline = scores_from_z(z, bias)
    residual = scores_from_z(target_z, bias) - baseline
    model = fit(z[1024:2048], residual[1024:2048], prior['selected_ridge'])
    digest = hashlib.sha256(b''.join(v.tobytes() for v in model)).hexdigest()
    if digest != prior['fitted_adapter_sha256']:
        raise RuntimeError(f'layer{layer} prompt adapter does not reproduce its frozen hash')
    for name, value in zip(('mean', 'scale', 'center', 'coef'), model):
        arrays[f'layer{layer}_{name}'] = value
    exact = np.array(routes['captured_scores'][str(layer)], dtype=np.float32)
    softplus = np.maximum(exact - bias, np.float32(1e-6)) ** 2
    exact_z = softplus + np.log(-np.expm1(-softplus))
    mean, scale, center, coef = model
    corrected = exact + (((exact_z - mean) / scale) @ coef + center)
    arrays[f'layer{layer}_scores'] = corrected
    selected = next(r for r in reference['transfer_analysis']['families']['transferred_ridge']['selected_layers']
                    if r['layer'] == layer)
    if selected['config'] != installation['ridge_configs'][str(layer)]:
        raise RuntimeError('issue settings differ from frozen chronological calibration')
    rows.append({'layer': layer, 'fitted_adapter_sha256': digest,
                 'adapter_bytes': sum(v.nbytes for v in model),
                 'score_bytes': corrected.nbytes, 'selected_ridge': prior['selected_ridge'],
                 'config': selected['config'], 'prior_train': selected['train'],
                 'prior_heldout': selected['heldout'],
                 'inverse_score_max_error': float(np.max(np.abs(scores_from_z(exact_z, bias) - exact)))})
    del previous, current, weight, bias, z, target_z, baseline, residual, exact, exact_z, softplus, corrected
for name, digest in digests.items():
    if reference['consumed_range_sha256'].get(name) != digest:
        raise RuntimeError('prompt source range changed: ' + name)
destination = ROOT / 'ridge-parameters.npz'
if destination.exists():
    raise RuntimeError('refusing parameter evidence overwrite')
np.savez(destination, **arrays)
report = {'complete': True, 'cpu_only': True, 'mlx_imported': False,
          'source_commit': installation['source_commit'], 'before': before,
          'after': host_memory_snapshot(), 'elapsed_s': time.monotonic() - started,
          'static_incremental_bound_bytes': 1024**3, 'rows': rows,
          'parameters_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
          'parameters_bytes': destination.stat().st_size,
          'consumed_range_sha256': digests,
          'scope': 'Reproduce two frozen prompt-only adapters and their saved-score application. No new model selection or target execution.'}
OUT.write_text(json.dumps(report, indent=2) + '\n')
print('RIDGE_PREPARATION', json.dumps({k: report[k] for k in ('complete', 'elapsed_s', 'parameters_bytes', 'rows')}), flush=True)
