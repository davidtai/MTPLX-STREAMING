"""Causal route prediction on the complete saved M6 trajectory, CPU only.

This is an optimistic coverage screen, not a prefetch simulation. It preserves
the ordinary bank state and gives every correct prediction unlimited lead time.
False predictions are counted as extra traffic. No future route trains a
prediction; the following true route is used only to score it.
"""
import ast
from collections import Counter
import gzip
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import time


class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('MLX forbidden in CPU prediction screen')


sys.meta_path.insert(0, NoMLX())
import numpy as np
from mtplx.expert_streaming import LayerExpertSlotBank, RoutingPhase

ROOT = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
TRACE = ROOT / 'docs/deepseek-v41/receipts/mtp-verify-routes-20260913/mtp-verify-routes-16k-1024-v2.json.gz'
OLD_SCREEN = Path('/tmp/screen_dsv41_cap93_staged_policy.py')
with gzip.open(TRACE, 'rt') as stream:
    data = json.load(stream)
node = next(n for n in ast.parse(OLD_SCREEN.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == 'bank_for')
exec(compile(ast.Module(body=[node], type_ignores=[]), str(OLD_SCREEN), 'exec'))

LAYERS = sorted(data['target_routes_by_layer'], key=int)
assert LAYERS == list(map(str, range(40))) and data['cycles'] == 206
ROUTES = {int(l): [np.unique(row) for row in data['target_routes_by_layer'][l]]
          for l in LAYERS}
N = 384
K = (1, 2, 4, 6)
METHODS = ('previous_route', 'same_layer', 'cross_layer', 'blend')
HALF = 103
stats = np.zeros((2, len(METHODS), len(K), 40, 3), dtype=np.int64)
demands = np.zeros((2, 40), dtype=np.int64)
banks = {int(l): bank_for(l, 102, 'transition-window') for l in LAYERS}
same = np.zeros((40, N, N), dtype=np.float32)
cross = np.zeros_like(same)
same_den = np.zeros((40, N), dtype=np.float32)
cross_den = np.zeros_like(same_den)
previous = [None] * 40
pending = {}
start = time.perf_counter()

for cycle in range(206):
    half = int(cycle >= HALF)
    for layer in range(40):
        # Observe the route now, at its real layer boundary. Predictions for
        # this layer were fixed before reaching this block.
        route = ROUTES[layer][cycle]
        bank = banks[layer]
        plan = bank.plan(data['target_routes_by_layer'][str(layer)][cycle],
                         phase=RoutingPhase.DECODE)
        missing = {int(load.expert) for load in plan.loads}
        demands[half, layer] += len(missing)
        for method, ranked in pending.pop(layer, []):
            for j, k in enumerate(K):
                predicted = set(map(int, ranked[:k]))
                hits = len(predicted & missing)
                stats[half, method, j, layer] += (len(predicted), hits, len(predicted)-hits)

        # Update only with routes observed up through this layer and cycle.
        prior = previous[layer]
        if prior is not None:
            same[layer] *= np.float32(0.98)
            same_den[layer] *= np.float32(0.98)
            same[layer][np.ix_(prior, route)] += 1
            same_den[layer, prior] += 1
        if layer > 0:
            source = ROUTES[layer-1][cycle]  # already observed
            cross[layer] *= np.float32(0.98)
            cross_den[layer] *= np.float32(0.98)
            cross[layer][np.ix_(source, route)] += 1
            cross_den[layer, source] += 1
        previous[layer] = route

        target = layer + 1
        if target == 40 or previous[target] is None:
            continue
        prior = previous[target]
        a = np.zeros(N, dtype=np.float32)
        a[prior] = 1
        b = (same[target, prior] / np.maximum(1, same_den[target, prior, None])).mean(axis=0)
        c = (cross[target, route] / np.maximum(1, cross_den[target, route, None])).mean(axis=0)
        resident = np.array(list(banks[target]._expert_to_slot), dtype=np.int32)
        for i, score in enumerate((a, b, c, 0.5*b + 0.5*c)):
            score[resident] = -1
            # Stable id tie break; only positive evidence can issue a prediction.
            order = np.argsort(-score, kind='stable')
            ranked = order[score[order] > 0][:max(K)].copy()
            pending.setdefault(target, []).append((i, ranked))

elapsed = time.perf_counter() - start
output = []
for i, method in enumerate(METHODS):
    for j, k in enumerate(K):
        train = stats[0, i, j]
        # Select useful layers using the first half only; evaluation is the
        # chronological second half, with no retuning on those scores.
        selected = [l for l in range(1, 40)
                    if train[l, 0] >= 20 and train[l, 1]/train[l, 0] >= 0.8]
        for scope, ids in (('all_layers', list(range(40))), ('train_precision_80', selected)):
            issued, useful, wasted = map(int, stats[1, i, j, ids].sum(axis=0))
            demand = int(demands[1].sum())
            output.append(dict(method=method, k=k, scope=scope, selected_layers=ids,
                heldout_baseline_reads=demand, predicted_reads=issued,
                potentially_early_reads=useful, extra_reads=wasted,
                useful_precision=useful/issued if issued else None,
                optimistic_hidden_read_fraction=useful/demand,
                added_traffic_fraction=wasted/demand))

result = dict(scope=__doc__, elapsed_s=elapsed, capacity=102, policy='transition-window',
    trace=str(TRACE), trace_sha256=hashlib.sha256(TRACE.read_bytes()).hexdigest(),
    bank_initializer_source=str(OLD_SCREEN),
    bank_initializer_sha256=hashlib.sha256(OLD_SCREEN.read_bytes()).hexdigest(),
    initial_state_limit='Historical initial bank expanded using prefill frequencies; first half warms the banks. Not the later packed full-run initial residency.',
    cycles=206, train_cycles=103, heldout_cycles=103,
    baseline_reads_by_half=demands.sum(axis=1).tolist(),
    predictor_arrays_bytes=sum(x.nbytes for x in (same,cross,same_den,cross_den)),
    mlx_imported=any(x=='mlx' or x.startswith('mlx.') for x in sys.modules),
    candidates=output,
    limits=['Unlimited prediction lead time; no I/O contention or bank pollution is modeled.',
            'Actual runtime currently rejects prefetch with transition-window; no production code changed.',
            'Screen results cannot be promoted as latency or throughput evidence.'])
Path('/tmp/dsv41-causal-prefetch-20260917/screen.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='candidates'}, indent=2))
print(json.dumps(sorted(output, key=lambda r:-(r['optimistic_hidden_read_fraction']-r['added_traffic_fraction']))[:8], indent=2))
