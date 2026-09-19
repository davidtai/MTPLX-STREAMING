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
