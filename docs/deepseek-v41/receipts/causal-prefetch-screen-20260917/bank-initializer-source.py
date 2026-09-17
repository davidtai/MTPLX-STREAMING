"""CPU-only cap-93 policy x staged-verify replay; MLX imports are forbidden."""
import gzip
import importlib.abc
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise RuntimeError("offline replay cannot import MLX")
        return None
sys.meta_path.insert(0, NoMLX())

from mtplx.expert_streaming import LayerExpertSlotBank, RoutingPhase

TRACE = Path("docs/deepseek-v41/receipts/mtp-verify-routes-20260913/mtp-verify-routes-16k-1024-v2.json.gz")
with gzip.open(TRACE, "rt") as stream:
    data = json.load(stream)
layers = sorted(data["target_routes_by_layer"], key=int)
accepts = [0] * 10 + [1] * 13 + [2] * 20 + [3] * 14 + [4] * 18 + [5] * 131
assert len(accepts) == data["cycles"] == 206

def bank_for(layer, cap, policy):
    src = data["initial_banks"][layer]
    bank = LayerExpertSlotBank(
        expert_count=384,
        persistent_slots=cap,
        transient_slots=44,
        frequency_decay=float(src["frequency_decay"]),
        cache_policy=policy,
        single_pool=True,
        layer_id=int(layer),
    )
    original = [int(e) for e in src["_slot_to_expert"]]
    prefill = Counter({int(e): int(n) for e, n in src["_prefill_route_freq"].items()})
    additions = sorted(
        (e for e in prefill if e not in original), key=lambda e: (-prefill[e], e)
    )[: cap - len(original)]
    residents = original + additions
    bank._slot_to_expert[:] = residents
    bank._expert_to_slot = {expert: slot for slot, expert in enumerate(residents)}
    recency = {int(e): int(n) for e, n in src["_pool_recency"].items()}
    clock = int(src["_pool_clock"])
    for expert in additions:
        clock += 1
        recency[expert] = clock
    bank._pool_recency = recency
    bank._pool_clock = clock
    bank._protected = set(residents)
    bank._prefill_route_freq = prefill
    for index, history in enumerate(src["_history"]):
        bank._history[index].score = float(history["score"])
        bank._history[index].score_epoch = int(history["score_epoch"])
        bank._history[index].last_used = int(history["last_used"])
    bank._decode_epoch = int(src["_decode_epoch"])
    bank._saw_decode_since_prefill = bool(src["_saw_decode_since_prefill"])
    return bank

def replay(policy, chunks, order):
    banks = {layer: bank_for(layer, 93, policy) for layer in layers}
    reads = calls = forwarded = 0
    for cycle, accepted in enumerate(order):
        needed = accepted + 1
        lo = 0
        for chunk in chunks:
            if lo >= needed:
                break
            hi = min(6, lo + chunk)
            calls += 1
            forwarded += hi - lo
            for layer in layers:
                route = data["target_routes_by_layer"][layer][cycle][6 * lo : 6 * hi]
                plan = banks[layer].plan(route, phase=RoutingPhase.DECODE)
                reads += len(plan.loads)
            lo = hi
    return reads, calls, forwarded

def summary(values):
    values = sorted(values)
    return {
        "min": values[0],
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": values[-1],
    }

orders = []
for seed in range(8):
    order = list(accepts)
    random.Random(seed).shuffle(order)
    orders.append(order)
rows = []
for policy in ("frequency", "transition-window"):
    for name, chunks in (("full6", (6,)), ("3_then3", (3, 3))):
        values = [replay(policy, chunks, order) for order in orders]
        rows.append({
            "cap": 93,
            "policy": policy,
            "schedule": name,
            "chunks": chunks,
            "reads": summary([value[0] for value in values]),
            "forward_calls": summary([value[1] for value in values]),
            "rows_forwarded": summary([value[2] for value in values]),
        })
out = {
    "mlx_imported": any(name == "mlx" or name.startswith("mlx.") for name in sys.modules),
    "trace": str(TRACE),
    "acceptance_histogram": dict(Counter(accepts)),
    "permutations": len(orders),
    "rows": rows,
}
assert not out["mlx_imported"]
Path("/tmp/dsv41-cap93-staged-policy.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
