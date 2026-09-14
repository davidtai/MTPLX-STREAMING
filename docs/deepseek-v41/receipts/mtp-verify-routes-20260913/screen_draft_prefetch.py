"""CPU-only held-out screen of draft-route hints for target miss prefetch.

Uses archived, matching-output MTP draft and target traces. A prediction is
ranked before the current target route is observed; the installed target cache
policy is replayed unchanged. This measures hint precision, not TPS or a
deployable prefetch lane.
"""

import gzip
import json
import runpy
from collections import Counter, deque
from pathlib import Path

HERE = Path(__file__).parent
TARGET = HERE / "mtp-verify-routes-16k-1024-v2.json.gz"
DRAFT = HERE.parent / "mtp-route-capture-20260913/mtp-routes-16k-1024.json"
REPLAY = HERE.parent / "memory-budget-110/replay_route_policies.py"


def main():
    with gzip.open(TARGET, "rt") as stream:
        target = json.load(stream)
    draft = json.loads(DRAFT.read_text())
    assert target["complete"] and draft["matches_prior_mtp_ids"]
    assert target["generated_ids"] == draft["generated_ids"]
    assert target["cycles"] == draft["cycles"] == 206
    restore = runpy.run_path(str(REPLAY))["restore"]
    feats = [
        {(stage, int(e)) for stage, rows in enumerate(cycle)
         for row in rows for e in row}
        for cycle in draft["draft_routes_by_cycle_stage_row"]
    ]
    result = Counter()
    for layer, sequence in target["target_routes_by_layer"].items():
        bank = restore(target["initial_banks"][layer])
        seen = 0
        counts = Counter()
        feature_count = Counter()
        joint = {}
        recent = deque(maxlen=8)
        for cycle, step in enumerate(sequence):
            required = set(step)
            resident = set(bank._expert_to_slot)
            misses = required - resident
            if cycle >= 100:
                result["actual_misses"] += len(misses)
                base = {e: counts[e] / seen for e in range(bank.expert_count)}
                recent_counts = Counter(e for row in recent for e in row)
                conditional = dict(base)
                for feature in feats[cycle]:
                    total = feature_count[feature] + 5
                    table = joint.get(feature, {})
                    for e in range(bank.expert_count):
                        conditional[e] += (table.get(e, 0) - feature_count[feature] * base[e]) / total / len(feats[cycle])
                for name, rank in (
                    ("base", lambda e: (base[e], counts[e], -e)),
                    ("recent8", lambda e: (recent_counts[e], base[e], -e)),
                    ("draft", lambda e: (conditional[e], base[e], -e)),
                ):
                    choices = sorted(set(range(bank.expert_count)) - resident,
                                     key=rank, reverse=True)
                    for width in (2, 4, 8, 12):
                        predicted = set(choices[:width])
                        key = f"{name}_top{width}"
                        result[key + "_hits"] += len(predicted & misses)
                        result[key + "_issued"] += width
            plan = bank.try_plan_all_hits(step, phase="decode")
            if plan is None:
                plan = bank.plan(step, phase="decode")
            assert len(plan.misses) == len(misses)
            result["replayed_reads"] += len(plan.misses)
            seen += 1
            counts.update(required)
            recent.append(required)
            for feature in feats[cycle]:
                feature_count[feature] += 1
                joint.setdefault(feature, Counter()).update(required)
    assert result["replayed_reads"] == target["decode_records_read"]
    print(json.dumps({"control_reads": result["replayed_reads"],
                      "heldout_actual_misses": result["actual_misses"],
                      "predictions": {
                          key.removesuffix("_hits"): {
                              "hit_reads": value,
                              "issued_reads": result[key.removesuffix("_hits") + "_issued"],
                              "precision": value / result[key.removesuffix("_hits") + "_issued"],
                              "fraction_of_misses": value / result["actual_misses"],
                          }
                          for key, value in result.items() if key.endswith("_hits")
                      }}, indent=2))


if __name__ == "__main__":
    main()
