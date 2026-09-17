"""CPU-only causal cache capacity screen over the recorded full M6 routes."""
import argparse
import gzip
import importlib.abc
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np


class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise RuntimeError("offline replay cannot import MLX")
        return None


sys.meta_path.insert(0, NoMLX())
from mtplx.expert_streaming import LayerExpertSlotBank, RoutingPhase


parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
trace = Path(
    "docs/deepseek-v41/receipts/mtp-verify-routes-20260913/"
    "mtp-verify-routes-16k-1024-v2.json.gz"
)
with gzip.open(trace, "rt") as stream:
    data = json.load(stream)
layers = sorted(data["target_routes_by_layer"], key=int)


def bank_for(layer, capacity, policy):
    src = data["initial_banks"][layer]
    bank = LayerExpertSlotBank(
        expert_count=384,
        persistent_slots=capacity,
        transient_slots=48,
        frequency_decay=float(src["frequency_decay"]),
        cache_policy=policy,
        single_pool=True,
        layer_id=int(layer),
    )
    original = [int(expert) for expert in src["_slot_to_expert"]]
    prefill = Counter({int(e): int(n) for e, n in src["_prefill_route_freq"].items()})
    additions = sorted(
        (expert for expert in prefill if expert not in original),
        key=lambda expert: (-prefill[expert], expert),
    )[: capacity - len(original)]
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


def cost(layer, capacity, policy):
    bank = bank_for(layer, capacity, policy)
    reads = 0
    for route in data["target_routes_by_layer"][layer]:
        if len(route) != 36:
            raise RuntimeError("expected a complete six-row route")
        reads += len(bank.plan(route, phase=RoutingPhase.DECODE).loads)
    return reads


def allocate(curves, budget):
    # Five physical bank shapes bound the number of kernel specializations.
    states = {0: (0, [])}
    for layer in layers:
        updated = {}
        for used, (reads, capacities) in states.items():
            for capacity in (73, 88, 96, 112, 128):
                total = used + capacity
                if total > budget:
                    continue
                candidate = (reads + curves[layer][capacity], capacities + [capacity])
                if total not in updated or candidate[0] < updated[total][0]:
                    updated[total] = candidate
        states = updated
    reads, capacities = states[budget]
    return {"total_slots": budget, "reads": reads, "capacities": capacities}


if args.out.exists():
    raise RuntimeError("refusing to overwrite replay evidence")
import hashlib
import time
start = time.monotonic()
curves = {}
for policy in ("frequency", "transition-window", "transition-window-tuned"):
    capacities = (80, 90, 91) if policy == "frequency" else (73, 80, 88, 90, 91, 96, 112, 128)
    curves[policy] = {}
    for layer in layers:
        curves[policy][layer] = {cap: cost(layer, cap, policy) for cap in capacities}
    print(policy, {cap: sum(curves[policy][layer][cap] for layer in layers) for cap in (80, 90, 91)}, flush=True)
allocations = {policy: {str(budget): allocate(curves[policy], budget) for budget in (3600, 3640)}
               for policy in ("transition-window", "transition-window-tuned")}
out = {
    "trace": str(trace), "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
    "source_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "scope": "recorded full M6 routes in temporal order; no shuffled acceptance or staged verification",
    "transient_slots": 48, "curves": curves, "allocations": allocations,
    "uniform_reads": {p: {str(cap): sum(curves[p][layer][cap] for layer in layers) for cap in (80, 90, 91)} for p in curves},
    "elapsed_s": time.monotonic() - start,
    "mlx_imported": any(name == "mlx" or name.startswith("mlx.") for name in sys.modules),
}
assert not out["mlx_imported"]
with args.out.open("x") as f:
    json.dump(out, f, indent=2); f.write("\n")
print(json.dumps({"allocations": allocations, "elapsed_s": out["elapsed_s"], "mlx_imported": out["mlx_imported"]}), flush=True)
