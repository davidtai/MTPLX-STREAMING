"""Causal CPU-only policy screen over captured native M6 target routes."""
import gzip
import hashlib
import importlib.abc
import json
from pathlib import Path
import runpy
import sys
import time

class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('CPU cache screen must not import MLX')

sys.meta_path.insert(0, NoMLX())
import numpy as np

ROOT = Path('/tmp/dsv41-online-cache-20260917')
TRACE = Path('docs/deepseek-v41/receipts/mtp-verify-routes-20260913/mtp-verify-routes-16k-1024-v2.json.gz')
RESTORE = Path('docs/deepseek-v41/receipts/memory-budget-110/replay_route_policies.py')
restore = runpy.run_path(str(RESTORE))['restore']


class OnlineLogistic:
    def __init__(self, lags, lr):
        self.lags, self.lr = lags, np.float32(lr)
        self.weights = np.zeros((384 * lags, 384), np.float32)
        self.bias = np.full(384, -2.5, np.float32)
        self.history = []
        self.features = None
        self.scale = np.float32(1)
        self.probability = np.full(384, 1 / (1 + np.exp(2.5)), np.float32)

    def observe(self, current):
        # Train the previous route's forecast only after this route is known.
        # No future expert IDs or acceptance labels enter the predictor.
        if self.features is not None:
            truth = np.zeros(384, np.float32)
            truth[current] = 1
            error = truth - self.probability
            self.weights *= np.float32(0.999)
            self.weights[self.features] += self.lr * self.scale * error
            self.bias += self.lr * np.float32(0.25) * error
        self.history.insert(0, current)
        del self.history[self.lags:]
        self.features = np.concatenate([rows + lag * 384 for lag, rows in enumerate(self.history)])
        self.scale = np.float32(1 / np.sqrt(len(self.features)))
        logits = self.bias + self.scale * self.weights[self.features].sum(axis=0)
        self.probability = 1 / (1 + np.exp(-np.clip(logits, -12, 12)))


def replay_layer(state, sequence, *, capacity, lags=0, lr=0, blend=0):
    bank = restore(state, policy='transition-window', single_pool=True)
    extra = capacity - bank.persistent_slots
    assert extra >= 0
    bank._slot_to_expert.extend([None] * extra)
    bank.persistent_slots = bank._persistent_capacity = capacity
    bank.slot_count += extra
    bank._protected_cap = max(1, int(capacity * .8))
    predictor = OnlineLogistic(lags, lr) if lags else None
    stock_scores = bank._transition_window_scores
    if predictor is not None:
        if blend == 1:
            bank._transition_window_scores = lambda: predictor.probability
        else:
            def mixed_scores():
                stock = stock_scores()
                pred = predictor.probability
                return np.float32(1 - blend) * stock / max(float(stock.max()), 1e-6) + np.float32(blend) * pred / max(float(pred.max()), 1e-6)
            bank._transition_window_scores = mixed_scores
    misses = []
    for step in sequence:
        if predictor is not None:
            predictor.observe(np.asarray(sorted(set(step)), dtype=np.intp))
        plan = bank.try_plan_all_hits(step, phase='decode')
        if plan is None:
            plan = bank.plan(step, phase='decode')
        misses.append(len(plan.misses))
    return dict(train_reads=sum(misses[:100]), heldout_reads=sum(misses[100:]),
                total_reads=sum(misses), per_cycle_reads=misses)


def main():
    with gzip.open(TRACE, 'rt') as f:
        trace = json.load(f)
    assert trace['complete'] and trace['cycles'] == 206 and trace['output_tokens'] == 1024
    digest = hashlib.sha256(json.dumps(trace['generated_ids']).encode()).hexdigest()
    assert digest == '0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac'
    configs = [dict(capacity=100)]
    configs += [dict(capacity=100, lags=lags, lr=lr, blend=blend)
                for lags in (1, 2) for lr in (.1, .5) for blend in (.5, 1)]
    report = dict(scope='CPU causal online cache-policy screen, no throughput claim',
        trace_sha256=hashlib.sha256(TRACE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        token_ids_sha256=digest, initial_state='Captured73-slot prefill bank plus27 empty slots, identical for every candidate',
        selection='100 chronological cycles for candidate selection; report the following106 separately; online updates continue causally',
        arms=[], complete=False)
    for config in configs:
        start = time.perf_counter()
        rows = {layer: replay_layer(trace['initial_banks'][layer], sequence, **config)
                for layer, sequence in trace['target_routes_by_layer'].items()}
        arm = dict(config=config, elapsed_s=time.perf_counter()-start,
            **{key:sum(row[key] for row in rows.values()) for key in ('train_reads','heldout_reads','total_reads')},
            per_layer=rows)
        report['arms'].append(arm)
        print(json.dumps({k:v for k,v in arm.items() if k!='per_layer'}), flush=True)
        (ROOT/'online-screen.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True
    (ROOT/'online-screen.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
