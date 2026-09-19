"""CPU-only prefill-derived slot allocation, without future decode lookahead."""
import gzip
import hashlib
import heapq
import json
from pathlib import Path
import runpy
import subprocess
import time

ROOT = Path(__file__).resolve().parent
TRACE = Path('docs/deepseek-v41/receipts/mtp-verify-routes-20260913/mtp-verify-routes-16k-1024-v2.json.gz')
restore = runpy.run_path('docs/deepseek-v41/receipts/memory-budget-110/replay_route_policies.py')['restore']
MINIMUM = 84
MAXIMUM = 128
AVERAGE = 99


def allocation(states, mode):
    if mode == 'prefill_frequency':
        select = runpy.run_path(str(ROOT / 'full/packed/prefill_allocation.py'))['capacities_from_prefill']
        counts = {int(layer): {int(k): v for k, v in state['_prefill_route_freq'].items()}
                  for layer, state in states.items()}
        capacities = select(counts, AVERAGE)
        return {layer: capacities[int(layer)] for layer in states}
    capacities = {layer: MINIMUM for layer in states}
    ranked = {}
    for layer, state in states.items():
        counts = sorted((float(state['_prefill_route_freq'].get(str(e), 0))
                         for e in range(384)), reverse=True)
        total = sum(counts)
        if total <= 0:
            raise ValueError('prefill route statistics are required')
        if mode == 'independent_m6_union':
            counts = [1 - (1 - value / (total / 6)) ** 6 for value in counts]
        ranked[layer] = counts
    heap = [(-ranked[layer][MINIMUM], int(layer), layer) for layer in states]
    heapq.heapify(heap)
    for _ in range((AVERAGE - MINIMUM) * len(states)):
        _, _, layer = heapq.heappop(heap)
        capacities[layer] += 1
        count = capacities[layer]
        if count < MAXIMUM:
            heapq.heappush(heap, (-ranked[layer][count], int(layer), layer))
    assert sum(capacities.values()) == AVERAGE * len(states)
    return capacities


def replay(trace, capacities):
    results = {}
    for layer, sequence in trace['target_routes_by_layer'].items():
        bank = restore(trace['initial_banks'][layer], policy='transition-window', single_pool=True)
        capacity = capacities[layer]
        extra = capacity - bank.persistent_slots
        assert extra >= 0
        bank._slot_to_expert.extend([None] * extra)
        bank.persistent_slots = bank._persistent_capacity = capacity
        bank.slot_count += extra
        bank._protected_cap = max(1, int(capacity * .8))
        counts = [len(bank.plan(step, phase='decode').misses) for step in sequence]
        results[layer] = {'capacity': capacity, 'first_half_misses': sum(counts[:103]),
                          'second_half_misses': sum(counts[103:]), 'total_misses': sum(counts)}
    return {**{key: sum(row[key] for row in results.values())
               for key in ('first_half_misses', 'second_half_misses', 'total_misses')},
            'per_layer': results}


def main():
    started = time.perf_counter()
    trace = json.loads(gzip.decompress(TRACE.read_bytes()))
    assert trace['complete'] and trace['cycles'] == 206
    states = trace['initial_banks']
    plans = {'uniform': {layer: AVERAGE for layer in states}}
    plans.update({mode: allocation(states, mode) for mode in ('prefill_frequency', 'independent_m6_union')})
    report = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'scope': 'Policy demand misses only; excludes physical transient reuse; no throughput claim',
              'selection': 'Prefill frequencies only; no decode rows used to select capacities',
              'initial_state': 'Same captured 73 residents in every arm; extra slots start empty',
              'minimum_capacity': MINIMUM, 'maximum_capacity': MAXIMUM, 'average_capacity': AVERAGE,
              'trace_sha256': hashlib.sha256(TRACE.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'arms': {}}
    for name, capacities in plans.items():
        same = next((other for other in report['arms'] if plans[other] == capacities), None)
        arm = replay(trace, capacities) if same is None else dict(report['arms'][same], identical_allocation_to=same)
        report['arms'][name] = arm
        print(json.dumps({'arm': name, **{k: v for k, v in arm.items() if k != 'per_layer'},
                          'min_capacity': min(capacities.values()), 'max_capacity': max(capacities.values())}), flush=True)
    report['elapsed_s'] = time.perf_counter() - started
    report['complete'] = True
    (ROOT / 'screen.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
