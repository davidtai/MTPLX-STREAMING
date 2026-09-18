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
OUT = ROOT / 'screen.json'
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

# Appended after the audited CPU-only trace and native gate tensor readers.
from mtplx.expert_streaming import LayerExpertSlotBank
import subprocess

installation=json.loads((ROOT/'installation.json').read_text())
if subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()!=installation['source_commit']:
    raise RuntimeError('source commit differs from CPU screen')
if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=installation['script_sha256']:
    raise RuntimeError('CPU screen source identity differs')
for name,expected in installation['runtime_sources'].items():
    if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('runtime source differs')
if hashlib.sha256((TRACE/'manifest.json').read_bytes()).hexdigest()!=installation['source_manifest_sha256']:
    raise RuntimeError('trace manifest differs')


def scores_from_z(z,bias):
    return np.sqrt(np.logaddexp(np.float32(0),z))+bias


def fit(z,y,lam):
    mean=z.mean(axis=0)
    scale=np.maximum(z.std(axis=0),np.float32(0.1))
    center=y.mean(axis=0)
    design=(z-mean)/scale
    gram=design.T@design
    gram.flat[::385]+=np.float32(lam*len(z))
    coef=np.linalg.solve(gram,design.T@(y-center))
    return mean,scale,center,coef


def correct(z,model):
    mean,scale,center,coef=model
    return ((z-mean)/scale)@coef+center


def proxy_misses(predictions,truth,layer):
    # Causal native policy replay of the captured AR rows. This is not the
    # M6 cache state, complete original prefill, or a prefetch timing simulation.
    bank=LayerExpertSlotBank(expert_count=384,persistent_slots=110,transient_slots=48,
        single_pool=True,cache_policy='transition-window',layer_id=layer)
    for route in truth[:2048]:
        bank.plan((int(v) for v in route),phase='prefill')
    ranked={name:np.argsort(-score,axis=1,kind='stable') for name,score in predictions.items()}
    result={name:{str(k):{'issued':0,'hits':0,'misses':0} for k in (1,2,4,6,10)}
            for name in predictions}
    for i,route in enumerate(truth[2048:]):
        required=set(int(v) for v in route)
        resident=set(bank._expert_to_slot)
        missing=required-resident
        for name,rank in ranked.items():
            for k in (1,2,4,6,10):
                selected=set(int(v) for v in rank[i,:k])-resident
                record=result[name][str(k)]
                record['issued']+=len(selected)
                record['hits']+=len(selected & missing)
                record['misses']+=len(missing)
        plan=bank.plan((int(v) for v in route),phase='decode')
        if set(plan.misses)!=missing:
            raise RuntimeError('cache proxy miss census differs from the native plan')
    return result



def fit_hidden(x, residual, lam):
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), np.float32(0.1))
    center = residual.mean(axis=0)
    design = (x - mean) / scale
    gram = design @ design.T
    gram.flat[::len(x)+1] += np.float32(lam * len(x))
    coef = design.T @ np.linalg.solve(gram, residual - center)
    return mean, scale, center, coef


def fold_hidden(weight, model):
    mean, scale, center, coef = model
    delta = coef / scale[:, None]
    return weight / temp + delta.T, center - mean @ delta


started = time.monotonic()
rows = []
prev = hidden(3, 'router_in')
for layer in range(4, 40):
    current = hidden(layer, 'router_in')
    truth = hidden(layer, 'top6')
    weight = tensor(f'layers.{layer}.ffn.gate.weight')
    bias = tensor(f'layers.{layer}.ffn.gate.bias')
    z = (prev @ weight.T) / temp
    target_z = (current @ weight.T) / temp
    baseline = scores_from_z(z, bias)
    target = scores_from_z(target_z, bias)
    fit_range, validation = slice(1024, 1792), slice(1792, 2048)
    score_residual = target - baseline
    choices = []
    for lam in (0.1, 1.0):
        model = fit(z[fit_range], score_residual[fit_range], lam)
        prediction = baseline[validation] + correct(z[validation], model)
        scored = metrics(prediction, truth[validation])['6']
        choices.append((scored['hits'], -float(np.mean((prediction - target[validation])**2)), lam))
    chosen = max(choices)[2]
    model = fit(z[1024:2048], score_residual[1024:2048], chosen)
    score_ridge = baseline[2048:] + correct(z[2048:], model)
    score_model_hash = hashlib.sha256(b''.join(v.tobytes() for v in model)).hexdigest()
    del model
    raw_residual = target_z - z
    hidden_choices = []
    for lam in (1.0, 10.0):
        model = fit_hidden(prev[fit_range], raw_residual[fit_range], lam)
        folded_weight, folded_bias = fold_hidden(weight, model)
        prediction = scores_from_z(prev[validation] @ folded_weight.T + folded_bias, bias)
        scored = metrics(prediction, truth[validation])['6']
        hidden_choices.append((scored['hits'], -float(np.mean((prediction - target[validation])**2)), lam))
        del model, folded_weight, folded_bias
    hidden_chosen = max(hidden_choices)[2]
    model = fit_hidden(prev[1024:2048], raw_residual[1024:2048], hidden_chosen)
    folded_weight, folded_bias = fold_hidden(weight, model)
    folded_raw = prev[2048:] @ folded_weight.T + folded_bias
    unfolded_raw = z[2048:] + correct(prev[2048:], model)
    max_fold_raw_error = float(np.max(np.abs(folded_raw - unfolded_raw)))
    if not np.isfinite(folded_raw).all() or max_fold_raw_error > 1e-3:
        raise RuntimeError(f'nonfinite or inaccurate folded predictor: {max_fold_raw_error}')
    predictions = {
        'direct': baseline[2048:],
        'prompt_score_ridge': score_ridge,
        'full_hidden_raw_ridge': scores_from_z(folded_raw, bias),
    }
    result = {name: metrics(score, truth[2048:]) for name, score in predictions.items()}
    result['self_alignment'] = metrics(target[2048:], truth[2048:])
    rows.append({
        'layer': layer,
        'score_ridge_lambda': chosen,
        'score_model_sha256': score_model_hash,
        'hidden_ridge_lambda': hidden_chosen,
        'hidden_validation_choices': hidden_choices,
        'metrics': result,
        'miss_proxy': proxy_misses(predictions, truth, layer),
        'folded_runtime_parameter_bytes': folded_weight.nbytes + folded_bias.nbytes,
        'folded_model_sha256': hashlib.sha256(folded_weight.tobytes() + folded_bias.tobytes()).hexdigest(),
        'max_fold_raw_error': max_fold_raw_error,
    })
    prev = current
    del model, folded_weight, folded_bias
    if layer % 8 == 7:
        print(json.dumps({'completed_through_layer': layer, 'elapsed_s': time.monotonic()-started}), flush=True)

summary = {}
for name in rows[0]['metrics']:
    summary[name] = {}
    for k in rows[0]['metrics'][name]:
        counts = {key: sum(r['metrics'][name][k][key] for r in rows) for key in ('hits', 'issued', 'truth')}
        counts.update(precision=counts['hits']/counts['issued'], recall=counts['hits']/counts['truth'])
        summary[name][k] = counts
miss_summary = {}
for name in rows[0]['miss_proxy']:
    miss_summary[name] = {}
    for k in rows[0]['miss_proxy'][name]:
        counts = {key: sum(r['miss_proxy'][name][k][key] for r in rows) for key in ('hits', 'issued', 'misses')}
        counts.update(precision=counts['hits']/max(1,counts['issued']), coverage=counts['hits']/max(1,counts['misses']))
        miss_summary[name][k] = counts
alignment = summary['self_alignment']['6']['recall']
report = {
    'complete': True, 'cpu_only': True, 'source_commit': installation['source_commit'],
    'scope': installation['scope'], 'training': installation['training'],
    'static_incremental_bound_bytes': 1024**3, 'before': before, 'after': host_memory_snapshot(),
    'elapsed_s': time.monotonic()-started, 'native_self_alignment': alignment,
    'cpu_alignment_usable': alignment > .995, 'same_prompt_as_acceptance': False,
    'decode_labels_used_for_training_or_selection': False,
    'all_layer_runtime_parameter_bytes': sum(r['folded_runtime_parameter_bytes'] for r in rows),
    'summary': summary, 'miss_proxy_summary': miss_summary, 'rows': rows,
    'consumed_range_sha256': digests,
}
assert report['all_layer_runtime_parameter_bytes'] == installation['folded_runtime_parameter_bytes']
OUT.write_text(json.dumps(report, indent=2)+'\n')
print('FULL_HIDDEN_ROUTER', json.dumps({k: report[k] for k in (
    'complete', 'elapsed_s', 'native_self_alignment', 'all_layer_runtime_parameter_bytes', 'miss_proxy_summary')}), flush=True)
