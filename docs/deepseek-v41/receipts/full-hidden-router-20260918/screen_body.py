
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
