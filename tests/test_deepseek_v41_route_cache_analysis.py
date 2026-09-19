"""CPU checks for the offline batch-cache lower bound."""
from functools import lru_cache
from itertools import combinations, product

from scripts.deepseek_v41.analyze_route_cache import optimal_batch_misses


def brute_force(sequence, capacity, initial=()):
    @lru_cache(None)
    def solve(index, cached):
        if index == len(sequence):
            return 0
        needed = set(sequence[index])
        present = set(cached)
        available = sorted(present | needed)
        keep = min(capacity, len(available))
        return len(needed - present) + min(
            solve(index + 1, subset)
            for subset in combinations(available, keep)
        )
    return solve(0, tuple(sorted(initial)))


def test_bypass_can_retain_a_useful_expert_while_serving_a_cold_route():
    # One persistent slot plus temporary service storage. Loading expert 1
    # need not evict expert 0; the mandatory-admission oracle reports two misses.
    assert optimal_batch_misses([(0,), (1,), (0,)], 1, initial=(0,)) == 1


def test_future_use_greedy_matches_exhaustive_batch_cache_choices():
    routes = tuple(combinations(range(4), 2))
    for sequence in product(routes, repeat=3):
        for capacity in (0, 1, 2, 3):
            for initial in ((), tuple(range(capacity))):
                assert optimal_batch_misses(sequence, capacity, initial) == brute_force(
                    sequence, capacity, initial
                )


def test_duplicates_count_one_record_and_empty_trace_counts_zero():
    assert optimal_batch_misses([(1, 1, 2), (2, 2)], 1) == 2
    assert optimal_batch_misses([], 4, initial=(1, 2)) == 0
