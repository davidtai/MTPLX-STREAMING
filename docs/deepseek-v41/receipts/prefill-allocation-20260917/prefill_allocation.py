"""Select fixed decode capacities from the completed prompt's routing counts."""
import heapq
import math

MINIMUM_CAPACITY = 84
MAXIMUM_CAPACITY = 128
LAYER_COUNT = 40
EXPERT_COUNT = 384


def capacities_from_prefill(counts_by_layer, average_capacity):
    if set(counts_by_layer) != set(range(LAYER_COUNT)):
        raise ValueError('prefill allocation requires all 40 target layers')
    if type(average_capacity) is not int or not MINIMUM_CAPACITY < average_capacity <= 103:
        raise ValueError('average capacity must be admitted above84 through103')
    ranked = {}
    for layer, counts in counts_by_layer.items():
        if any(type(expert) is not int or not 0 <= expert < EXPERT_COUNT for expert in counts):
            raise ValueError('invalid prefill expert ID')
        values = [float(counts.get(expert, 0)) for expert in range(EXPERT_COUNT)]
        if any(not math.isfinite(value) or value < 0 for value in values) or sum(values) <= 0:
            raise ValueError('complete nonnegative prefill statistics are required')
        ranked[layer] = sorted(values, reverse=True)
    capacities = [MINIMUM_CAPACITY] * LAYER_COUNT
    candidates = [(-ranked[layer][MINIMUM_CAPACITY], layer) for layer in range(LAYER_COUNT)]
    heapq.heapify(candidates)
    for _ in range((average_capacity - MINIMUM_CAPACITY) * LAYER_COUNT):
        _, layer = heapq.heappop(candidates)
        capacities[layer] += 1
        count = capacities[layer]
        if count < MAXIMUM_CAPACITY:
            heapq.heappush(candidates, (-ranked[layer][count], layer))
    return tuple(capacities)
