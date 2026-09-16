"""CPU-only screen of nonuniform per-layer target expert-slot allocation.

Replays the installed 2Q policy at fixed total physical slots. Fewer slots
evict from the captured warm bank by its policy; extra slots start empty. A
trace-fitted allocation is an oracle screen, not an installable policy.
"""

import gzip
import json
import runpy
from pathlib import Path

from scripts.deepseek_v41.analyze_route_cache import optimal_batch_misses

HERE = Path(__file__).parent
TRACE = HERE / "mtp-verify-routes-16k-1024-v2.json.gz"
RESTORE = HERE.parent / "memory-budget-110/replay_route_policies.py"
BASE = 73
CHOICES = tuple(range(53, 94, 5))
TRAIN = 100


def replay_capacity(restore, state, sequence, capacity):
    bank = restore(state)
    if capacity < bank.persistent_slots:
        bank._persistent_capacity = capacity
        while bank.occupancy > capacity:
            slot = bank._pool_victim_slot(pinned=set(), allow_protected=True)
            bank.invalidate_expert(bank._slot_to_expert[slot])
        bank._protected_cap = max(1, int(capacity * .8))
    elif capacity > bank.persistent_slots:
        extra = capacity - bank.persistent_slots
        bank._slot_to_expert.extend([None] * extra)
        bank.persistent_slots = capacity
        bank.slot_count += extra
        bank._persistent_capacity = capacity
        bank._protected_cap = max(1, int(capacity * .8))
    train = heldout = 0
    for epoch, step in enumerate(sequence):
        plan = bank.try_plan_all_hits(step, phase="decode")
        if plan is None:
            plan = bank.plan(step, phase="decode")
        if epoch < TRAIN:
            train += len(plan.misses)
        else:
            heldout += len(plan.misses)
    return train, heldout


def optimize(rows, metric):
    # Exact fixed-slot knapsack at five-slot granularity, across all 40 layers.
    dp = {0: (0, ())}
    for row in rows:
        following = {}
        for delta, (cost, chosen) in dp.items():
            for capacity in CHOICES:
                next_delta = delta + (capacity - BASE) // 5
                candidate = (cost + row[capacity][metric], chosen + (capacity,))
                if next_delta not in following or candidate < following[next_delta]:
                    following[next_delta] = candidate
        dp = following
    return dp[0]


def main():
    with gzip.open(TRACE, "rt") as stream:
        trace = json.load(stream)
    restore = runpy.run_path(str(RESTORE))["restore"]
    layers = sorted(trace["target_routes_by_layer"], key=int)
    rows = [
        {capacity: replay_capacity(
            restore, trace["initial_banks"][layer],
            trace["target_routes_by_layer"][layer], capacity)
         for capacity in CHOICES}
        for layer in layers
    ]
    control = tuple(sum(row[BASE][part] for row in rows) for part in (0, 1))
    assert sum(control) == trace["decode_records_read"] == 53999
    train_best, train_choice = optimize(rows, 0)
    fitted_eval = sum(row[capacity][1] for row, capacity in zip(rows, train_choice))
    oracle_best, oracle_choice = optimize(
        [{capacity: (sum(parts),) for capacity, parts in row.items()} for row in rows], 0
    )
    assert sum(train_choice) == sum(oracle_choice) == len(layers) * BASE
    # Stronger relaxation: allow the same total slots to move between layers at
    # every access. Layer-qualified expert IDs preserve record identity while
    # removing all per-layer ownership constraints.
    width = max(
        state["expert_count"] for state in trace["initial_banks"].values()
    )
    steps = len(trace["target_routes_by_layer"][layers[0]])
    global_sequence = [
        [int(layer) * width + int(expert) for expert in trace["target_routes_by_layer"][layer][step]]
        for step in range(steps)
        for layer in layers
    ]
    global_initial = [
        int(layer) * width + int(expert)
        for layer, state in trace["initial_banks"].items()
        for expert in state["_expert_to_slot"]
    ]
    global_oracle = optimal_batch_misses(
        global_sequence, len(layers) * BASE, global_initial
    )
    print(json.dumps({
        "scope": "offline physical-slot allocation screen; no throughput claim",
        "source_commit": trace["source_commit"],
        "total_slots": len(layers) * BASE,
        "control_train_reads": control[0],
        "control_heldout_reads": control[1],
        "control_full_reads": sum(control),
        "train_fitted_train_reads": train_best,
        "train_fitted_heldout_reads": fitted_eval,
        "train_fitted_full_reads": train_best + fitted_eval,
        "train_fitted_slots_by_layer": dict(zip(layers, train_choice)),
        "full_trace_oracle_reads": oracle_best,
        "full_trace_oracle_slots_by_layer": dict(zip(layers, oracle_choice)),
        "global_full_trace_oracle_reads": global_oracle,
        "global_oracle_semantics": (
            "clairvoyant equal-size cache with bypass and slots movable between "
            "layers at every access; lower bound only"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
