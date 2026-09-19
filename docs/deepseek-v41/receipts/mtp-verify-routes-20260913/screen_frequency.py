"""CPU-only causal frequency/admission screen over actual MTP verify routes.

This is a diagnostic policy simulator, not a runtime implementation or TPS claim.
The current runtime policy is independently replayed by replay_route_policies.py.
"""

import gzip
import json
from collections import Counter
from pathlib import Path

TRACE = Path(__file__).with_name("mtp-verify-routes-16k-1024-v2.json.gz")


def replay_layer(sequence, state, *, decay, floor, ratio, seed, count_assignments):
    capacity = state["persistent_slots"]
    resident = {int(e) for e in state["_expert_to_slot"]}
    assert len(resident) == capacity
    prefill = {int(e): count for e, count in state["_prefill_route_freq"].items()}
    prefill_max = max(prefill.values(), default=1)
    score = {e: seed * prefill.get(e, 0) / prefill_max for e in resident}
    seen_at = {e: 0 for e in resident}
    misses = 0
    for epoch, step in enumerate(sequence, 1):
        counts = Counter(step)
        required = set(counts)

        def current(expert):
            return score.get(expert, 0) * decay ** (epoch - seen_at.get(expert, epoch))

        for expert, multiplicity in counts.items():
            score[expert] = current(expert) + (multiplicity if count_assignments else 1)
            seen_at[expert] = epoch
        missing = required - resident
        misses += len(missing)
        for expert in sorted(missing, key=lambda e: (-score[e], e)):
            eligible = resident - required
            if not eligible:
                continue
            victim = min(eligible, key=lambda e: (current(e), e))
            if score[expert] >= max(floor, ratio * current(victim)):
                resident.remove(victim)
                resident.add(expert)
    return misses


def main():
    with gzip.open(TRACE, "rt") as stream:
        trace = json.load(stream)
    output = []
    for assignments in (False, True):
        for decay in (0.85, 0.9, 0.95, 0.98, 0.995):
            for floor in (0, 1.5, 2.5):
                for seed in (0, 2, 5):
                    misses = sum(
                        replay_layer(sequence, trace["initial_banks"][layer],
                                     decay=decay, floor=floor, ratio=1.0,
                                     seed=seed, count_assignments=assignments)
                        for layer, sequence in trace["target_routes_by_layer"].items()
                    )
                    output.append({"assignments": assignments, "decay": decay,
                                   "floor": floor, "seed": seed, "record_reads": misses})
    print(json.dumps({"current_control_record_reads": trace["decode_records_read"],
                      "candidates": sorted(output, key=lambda row: row["record_reads"])[:20]},
                     indent=2))


if __name__ == "__main__":
    main()
