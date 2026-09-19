"""Offline replay of installed policies using the captured warm state."""
import argparse
import importlib.abc
import json
import sys
from collections import Counter
from pathlib import Path

class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('offline replay cannot import MLX')

sys.meta_path.insert(0, NoMLX())
from mtplx.expert_streaming import LayerExpertSlotBank, _ExpertHistory

def restore(state, *, policy=None, decay=None, single_pool=None):
    kwargs = {key: state[key] for key in (
        'expert_count', 'persistent_slots', 'transient_slots',
        'frequency_decay', 'cache_policy', 'single_pool')}
    if policy is not None:
        kwargs['cache_policy'] = policy
    if decay is not None:
        kwargs['frequency_decay'] = decay
    if single_pool is not None:
        kwargs['single_pool'] = single_pool
    bank = LayerExpertSlotBank(**kwargs)
    for key in ('_slot_to_expert', '_pool_clock', '_decode_epoch',
                '_saw_decode_since_prefill'):
        setattr(bank, key, state[key])
    bank._slot_to_expert = list(bank._slot_to_expert)
    for key in ('_expert_to_slot', '_pool_recency'):
        setattr(bank, key, {int(k): v for k, v in state[key].items()})
    for key in ('_protected', '_prefill_seed_candidates'):
        setattr(bank, key, set(state[key]))
    bank._prefill_route_freq = Counter({int(k): v for k, v in state['_prefill_route_freq'].items()})
    bank._history = [_ExpertHistory(**h) for h in state['_history']]
    return bank

def replay(payload, **kwargs):
    per_layer = {}
    for layer, sequence in payload['sequences'].items():
        bank = restore(payload['initial_banks'][layer], **kwargs)
        misses = 0
        for step in sequence:
            # Serving probes the resident-only path first. Its ordered hit
            # touches are part of the real recency policy, unlike plan()'s set.
            plan = bank.try_plan_all_hits(step, phase='decode')
            if plan is None:
                plan = bank.plan(step, phase='decode')
            misses += len(plan.misses)
        per_layer[layer] = misses
    total = sum(per_layer.values())
    return {'misses': total, 'misses_per_token': total / payload['decode_steps'],
            'bytes_per_token': total * payload['record_bytes'] / payload['decode_steps'],
            'per_layer': per_layer}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('trace', type=Path)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    trace = json.loads(a.trace.read_text())
    receipt = json.loads(a.receipt.read_text().splitlines()[0])
    observed = receipt['serve_stream_counters']
    if receipt['pin_working_set']['enabled']:
        raise ValueError('this capture does not contain the pinned working set')
    control = replay(trace)
    if control['misses'] != observed['expert_cache']['expert_misses']:
        raise ValueError(f"control replay mismatch: {control['misses']} vs {observed['expert_cache']['expert_misses']}")
    result = {'purpose': 'offline policy screening; no throughput claim',
              'source_commit': trace['source_commit'], 'observed': observed,
              'control': control, 'control_exact': True, 'candidates': {}}
    for decay in (.8, .9, .95, .97, .98, .99, .995, .999, 1.):
        result['candidates'][f'frequency-{decay}'] = replay(
            trace, policy='frequency', decay=decay, single_pool=False)
    result['candidates']['lru'] = replay(trace, policy='lru', single_pool=False)
    a.out.write_text(json.dumps(result, indent=2) + '\n')

if __name__ == '__main__':
    main()
