#!/usr/bin/env python3
"""F8 host-cost profiler + microbench for the DeepSeek-V4.1 transition-window
decode route policy (LayerExpertSlotBank).

CPU ONLY -- MLX imports are hard-blocked by a meta-path finder; numpy only;
single process; run under ``nice -n 19``.  Drives the REAL LayerExpertSlotBank
transition-window policy over the REAL 206-cycle x 40-layer verify route trace
at 111 persistent + 48 transient (the f1-overlap-sim control config), exactly
the ``bank.plan(route, phase="decode")`` path the runtime runs on the main
thread before any SSD read is submitted.

Loader/replay pieces are copied (minimally) from
.worktrees/dsv41-f1-overlap-sim/scripts/deepseek_v41/overlap_schedule_sim.py
(branch f1/overlap-sim), whose cache replay reproduces the published miss
counts exactly (35,164/8,240 @cap102 transition-window; 53,999 @cap73+48
frequency).  This harness re-asserts those anchors before any timing claim.

Usage (from the worktree root):
  nice -n 19 python3 scripts/deepseek_v41/f8_policy_hostcost_bench.py profile --impl package
  nice -n 19 python3 scripts/deepseek_v41/f8_policy_hostcost_bench.py bench   --impl package --reps 7
  nice -n 19 python3 scripts/deepseek_v41/f8_policy_hostcost_bench.py both    --reps 7
"""
from __future__ import annotations

import argparse
import cProfile
import gzip
import hashlib
import importlib.abc
import importlib.util
import io
import json
import pstats
import sys
import time
from collections import Counter
from pathlib import Path


class _NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise RuntimeError("f8_policy_hostcost_bench is CPU-only; MLX is forbidden")


sys.meta_path.insert(0, _NoMLX())
# Resolve mtplx to the *current worktree* ahead of any editable install.
sys.path.insert(0, str(Path.cwd()))
import numpy as np  # noqa: E402  (after the guard)

TRACE = Path(
    "docs/deepseek-v41/receipts/mtp-verify-routes-20260913/"
    "mtp-verify-routes-16k-1024-v2.json.gz"
)
TRACE_SHA256 = "07b4b720bae831421bbbab6d4cb770ade577096c1e85733ad949350edf66d0da"
ORACLE_PATH = Path("tests/_f8_expert_streaming_oracle.py")

N_LAYERS = 40
PERSISTENT = 111
TRANSIENT = 48
POLICY = "transition-window"
RUN_LAYER_CALLS = 7920  # task's per-run scale (40 layers x 198 measured cycles)


def load_module(impl: str):
    """Return the expert_streaming module for ``impl`` in {package, oracle}."""

    if impl == "package":
        import mtplx.expert_streaming as mod  # noqa: WPS433

        return mod
    if impl == "oracle":
        spec = importlib.util.spec_from_file_location(
            "_f8_expert_streaming_oracle", str(ORACLE_PATH)
        )
        mod = importlib.util.module_from_spec(spec)
        # Register before exec so @dataclass can resolve cls.__module__.
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    raise ValueError(impl)


def load_trace():
    blob = TRACE.read_bytes()
    got = hashlib.sha256(blob).hexdigest()
    if got != TRACE_SHA256:
        raise SystemExit(f"trace sha256 mismatch: {got}")
    trace = json.loads(gzip.decompress(blob))
    if not (trace["complete"] and trace["cycles"] == 206):
        raise SystemExit("unexpected trace shape")
    return trace


def restore(mod, state, *, policy=None, decay=None, single_pool=None):
    """Restore a LayerExpertSlotBank from saved warm state (verbatim logic from
    docs/deepseek-v41/receipts/memory-budget-110/replay_route_policies.py, but
    parameterised by the module so old/new can be restored identically)."""

    kwargs = {
        key: state[key]
        for key in (
            "expert_count",
            "persistent_slots",
            "transient_slots",
            "frequency_decay",
            "cache_policy",
            "single_pool",
        )
    }
    if policy is not None:
        kwargs["cache_policy"] = policy
    if decay is not None:
        kwargs["frequency_decay"] = decay
    if single_pool is not None:
        kwargs["single_pool"] = single_pool
    bank = mod.LayerExpertSlotBank(**kwargs)
    for key in ("_slot_to_expert", "_pool_clock", "_decode_epoch",
                "_saw_decode_since_prefill"):
        setattr(bank, key, state[key])
    bank._slot_to_expert = list(bank._slot_to_expert)
    for key in ("_expert_to_slot", "_pool_recency"):
        setattr(bank, key, {int(k): v for k, v in state[key].items()})
    for key in ("_protected", "_prefill_seed_candidates"):
        setattr(bank, key, set(state[key]))
    bank._prefill_route_freq = Counter(
        {int(k): v for k, v in state["_prefill_route_freq"].items()}
    )
    bank._history = [mod._ExpertHistory(**h) for h in state["_history"]]
    return bank


def make_bank(mod, state, persistent, transient, policy):
    """Restore and grow to ``persistent`` slots -- verbatim from the f1 sim."""

    bank = restore(mod, state, policy=policy, single_pool=True)
    extra = persistent - bank.persistent_slots
    if extra > 0:
        bank._slot_to_expert.extend([None] * extra)
    bank.persistent_slots = bank._persistent_capacity = persistent
    bank.slot_count = persistent + transient
    bank.transient_slots = transient
    if hasattr(bank, "_protected_cap"):
        bank._protected_cap = max(1, int(persistent * 0.8))
    return bank


def build_banks(mod, trace, persistent=PERSISTENT, transient=TRANSIENT, policy=POLICY):
    layers = sorted(trace["target_routes_by_layer"], key=int)
    return {
        int(l): make_bank(mod, trace["initial_banks"][l], persistent, transient, policy)
        for l in layers
    }


def routes_by_layer(trace):
    return {
        int(l): trace["target_routes_by_layer"][l]
        for l in trace["target_routes_by_layer"]
    }


def replay_decode(banks, routes, cyc):
    """Deterministic decode replay -- the exact main-thread policy path.
    Returns total demand-miss records (a sanity anchor)."""

    total = 0
    for c in range(cyc):
        for L in range(N_LAYERS):
            plan = banks[L].plan(routes[L][c], phase="decode")
            total += len(plan.misses)
    return total


def replay_decode_txn(banks, routes, cyc):
    """Transaction-path replay -- what the runtime's _plan_route_transaction
    runs (bank.plan_transaction), committing each accepted route (no rollback).
    This is the receipt's 'cache policy transaction' cost."""

    total = 0
    for c in range(cyc):
        for L in range(N_LAYERS):
            plan, _txn = banks[L].plan_transaction(routes[L][c], phase="decode")
            total += len(plan.misses)
    return total


def anchor_checks(mod, trace):
    """Reproduce two published miss counts EXACTLY (same method as the f1 sim)."""

    out = {}
    misses = routes = 0
    for layer, seq in trace["target_routes_by_layer"].items():
        bank = make_bank(mod, trace["initial_banks"][layer], 102, 48, "transition-window")
        for route in seq:
            misses += len(set(bank.plan(route, phase="decode").misses))
            routes += 1
    out["prefix_readiness_cap102"] = {
        "misses": misses, "routes": routes,
        "expected_misses": 35164, "expected_routes": 8240,
        "exact": misses == 35164 and routes == 8240,
    }
    records = 0
    for layer, seq in trace["target_routes_by_layer"].items():
        bank = make_bank(mod, trace["initial_banks"][layer], 73, 48, "frequency")
        for route in seq:
            plan = bank.try_plan_all_hits(route, phase="decode")
            if plan is None:
                plan = bank.plan(route, phase="decode")
            records += len(plan.misses)
    out["mtp_verify_cap73"] = {
        "records": records, "expected": 53999, "exact": records == 53999,
    }
    return out


def run_profile(impl, trace, path="plan"):
    mod = load_module(impl)
    anchors = anchor_checks(mod, trace)
    for name, a in anchors.items():
        if not a["exact"]:
            raise SystemExit(f"anchor {name} failed to reproduce: {a}")
    routes = routes_by_layer(trace)
    cyc = trace["cycles"]
    calls = cyc * N_LAYERS
    banks = build_banks(mod, trace)
    replay = replay_decode_txn if path == "transaction" else replay_decode

    pr = cProfile.Profile()
    pr.enable()
    total = replay(banks, routes, cyc)
    pr.disable()

    st = pstats.Stats(pr)
    st.sort_stats("cumulative")
    buf = io.StringIO()
    st.stream = buf
    st.print_stats(40)
    print(f"\n=== cProfile ({impl}, path={path}): {calls} layer calls, "
          f"{total} demand records ===")
    print(buf.getvalue())

    # Per-layer-call table for the policy functions of interest.
    rows = []
    for (fname, line, func), (cc, nc, tt, ct, callers) in st.stats.items():
        if "expert_streaming" not in fname and "oracle" not in fname:
            continue
        rows.append((func, nc, tt, ct))
    rows.sort(key=lambda r: r[3], reverse=True)
    print(f"=== per-layer-call cost by policy function ({impl}) ===")
    print(f"{'function':40s}{'ncalls':>10}{'tot_us/call':>14}{'cum_us/call':>14}")
    for func, nc, tt, ct in rows[:25]:
        print(f"{func:40s}{nc:>10}{tt/calls*1e6:>14.3f}{ct/calls*1e6:>14.3f}")
    return {"impl": impl, "anchors": anchors, "records": total, "calls": calls}


def run_bench(impl, trace, reps, path="plan"):
    mod = load_module(impl)
    anchors = anchor_checks(mod, trace)
    for name, a in anchors.items():
        if not a["exact"]:
            raise SystemExit(f"anchor {name} failed to reproduce: {a}")
    routes = routes_by_layer(trace)
    cyc = trace["cycles"]
    calls = cyc * N_LAYERS
    replay = replay_decode_txn if path == "transaction" else replay_decode

    # Warm once (JIT-free, but touch code paths and let numpy settle).
    banks = build_banks(mod, trace)
    warm_records = replay(banks, routes, cyc)

    times = []
    records = None
    for _ in range(reps):
        banks = build_banks(mod, trace)  # fresh state (plan mutates); NOT timed
        t0 = time.perf_counter()
        rec = replay(banks, routes, cyc)
        times.append(time.perf_counter() - t0)
        records = rec
    times.sort()
    best = times[0]
    median = times[len(times) // 2]
    per_call_us = best / calls * 1e6
    per_run_s = per_call_us * 1e-6 * RUN_LAYER_CALLS
    print(f"\n=== microbench ({impl}, path={path}) ===")
    print(f"reps={reps} calls/rep={calls} records={records} (warm {warm_records})")
    print(f"best_replay_s={best:.4f} median_replay_s={median:.4f} "
          f"all={[round(x, 4) for x in times]}")
    print(f"per_layer_call_us (best) = {per_call_us:.3f}")
    print(f"per_run_s (x{RUN_LAYER_CALLS}) = {per_run_s:.4f}")
    return {
        "impl": impl, "reps": reps, "calls": calls, "records": records,
        "best_replay_s": best, "median_replay_s": median,
        "per_layer_call_us": per_call_us, "per_run_s": per_run_s,
        "all_replay_s": times,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["profile", "bench", "both", "anchors"])
    ap.add_argument("--impl", choices=["package", "oracle"], default="package")
    ap.add_argument("--path", choices=["plan", "transaction"], default="plan")
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()

    if any(m == "mlx" or m.startswith("mlx.") for m in sys.modules):
        raise SystemExit("MLX leaked into the process")

    trace = load_trace()
    print(f"trace sha256 OK: {TRACE_SHA256}")
    print(f"oracle sha256: {hashlib.sha256(ORACLE_PATH.read_bytes()).hexdigest()}")
    pkg = load_module("package")
    print("package expert_streaming sha256: "
          + hashlib.sha256(Path(pkg.__file__).read_bytes()).hexdigest())

    if args.mode == "anchors":
        print(json.dumps(anchor_checks(load_module(args.impl), trace), indent=2))
    elif args.mode == "profile":
        run_profile(args.impl, trace, args.path)
    elif args.mode == "bench":
        run_bench(args.impl, trace, args.reps, args.path)
    elif args.mode == "both":
        r_old = run_bench("oracle", trace, args.reps, args.path)
        r_new = run_bench("package", trace, args.reps, args.path)
        speedup = r_old["per_layer_call_us"] / r_new["per_layer_call_us"]
        print(f"\n=== BEFORE (oracle) vs AFTER (package) path={args.path} ===")
        print(f"per_layer_call_us: oracle={r_old['per_layer_call_us']:.3f} "
              f"package={r_new['per_layer_call_us']:.3f} speedup={speedup:.2f}x")
        print(f"per_run_s (x{RUN_LAYER_CALLS}): oracle={r_old['per_run_s']:.4f} "
              f"package={r_new['per_run_s']:.4f} "
              f"saved={r_old['per_run_s'] - r_new['per_run_s']:.4f}")
        assert r_old["records"] == r_new["records"], "record count differs!"
        print(f"records identical: {r_old['records']}")

    if any(m == "mlx" or m.startswith("mlx.") for m in sys.modules):
        raise SystemExit("MLX leaked into the process")


if __name__ == "__main__":
    main()
