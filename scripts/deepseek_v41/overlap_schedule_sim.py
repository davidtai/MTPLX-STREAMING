#!/usr/bin/env python3
"""Discrete-event overlap-schedule simulator for DeepSeek-V4.1 M6 verify decode.

CPU ONLY (MLX imports are hard-blocked by a NoMLX meta-path finder), numpy only,
single process; run under `nice -n 19`.  Full method, provenance and results are
in docs/deepseek-v41/receipts/f1-overlap-sim-20260919/README.md.

F1 hypothesis: the native decode runner is SERIAL per layer call (route -> wait
for demand expert-record reads with the main thread blocked -> GPU compute), so
the SSD is idle for the compute fraction of every layer.  Does issuing layer
L+1's predicted misses onto the SSD while layer L computes hide the read wait?

Replays the saved native M6 route trajectory (206 verify cycles x 40 layers)
through the real LayerExpertSlotBank cache policy (transition-window, 111+48
slots), then feeds the deterministic per-layer demand-miss stream into a
single-server SSD model with a bounded prefetch ring.  The cache replay is
validated exactly against prefix-readiness-20260918 (35,164/8,240 @ cap102) and
mtp-verify-routes-20260913 (53,999 records @ cap73+48) before any timing claim.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.abc
import json
import runpy
import sys
import time
from collections import deque
from pathlib import Path


class _NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise RuntimeError("overlap_schedule_sim is CPU-only; MLX is forbidden")


sys.meta_path.insert(0, _NoMLX())
# Resolve mtplx to the *current worktree* (ahead of any editable install), like
# prefix-readiness-20260918's screen.  Run this script from the worktree root.
sys.path.insert(0, str(Path.cwd()))
import numpy as np  # noqa: E402  (after guard, like the sibling receipts)

# Sibling module (numpy-only, same MLX guard): the router-capture re-scorer that
# builds the REAL 1-ahead predictions replayed by the capture-comparison arm.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rescore_router_capture as rrc  # noqa: E402

# ---------------------------------------------------------------------------
# Inputs (all read-only; hashes are asserted at run time)
# ---------------------------------------------------------------------------
TRACE = Path(
    "docs/deepseek-v41/receipts/mtp-verify-routes-20260913/"
    "mtp-verify-routes-16k-1024-v2.json.gz"
)
TRACE_SHA256 = "07b4b720bae831421bbbab6d4cb770ade577096c1e85733ad949350edf66d0da"
HELPER = Path("docs/deepseek-v41/receipts/memory-budget-110/replay_route_policies.py")

# Timing constants: extension-bank-20260919 full run -- the BEST complete result
# and the exact 111-slot config modelled (198 cycles, 73.7626 s).  Non-read compute
# = verify - read-union (serial premise) distributed UNIFORMLY over layer calls;
# per-layer arrays exist only for a different-capacity 77 s run (see README).
REC_BYTES = 17_694_720          # extension-bank growth.decode_weight_record_bytes ("17.7 MB")
SRC_REC_BYTES = 18_800_640      # trace source_record_bytes (older 73-slot capture)
VERIFY_S = 69.15495437418576
READ_UNION_S = 43.398996549
DRAFT_S = 1.970860714034643
ACCEPT_S = 0.06804832693887874
COMMIT_S = 0.15270599152427167
GROWTH_S = 2.195692875051908
MEAS_CYCLES = 198
MEAS_LAYER_CALLS = 40 * MEAS_CYCLES
C_LAYER = (VERIFY_S - READ_UNION_S) / MEAS_LAYER_CALLS   # non-read compute / layer call
T_DRAFT = DRAFT_S / MEAS_CYCLES
T_ACCEPT = ACCEPT_S / MEAS_CYCLES
T_COMMIT = COMMIT_S / MEAS_CYCLES

MEAS_RECORDS = 31_573           # extension-bank measured records read (198-cycle run)
N_EXPERTS = 384
DEFAULT_RATE = 12.9             # GB/s (decimal); task's "one SSD server at 12.9 GB/s"
MEAS_RATE = 12.873              # read_gb_per_s_window from the extension-bank run
SYNTH_COVERAGE = 0.74           # gate-oracle-class predictor, from w89 AR trace
SYNTH_PRECISION = 0.62

# Extrapolation anchors for the REAL-predictor capture arm: the published control
# of the full 1,024-token run at 111+48 slots / rate 12.9 (this receipt's sibling,
# f1-overlap-sim-20260919 README, control row).  The capture arm measures the
# read-wait-reduction fraction on the first 64 verify cycles and applies it here.
FULL_CONTROL_TOTAL_S = 74.667
FULL_CONTROL_READ_WAIT_S = 43.395
# The real predictor used by the capture arm: which captured score tensor and how
# per-row scores are merged (chosen by held-out coverage; see rescore curve).
REAL_FEATURE = 1                # 1 = post_attention_router (beats pre-attention mean)
REAL_MERGE = "max"             # max-over-rows beats sum-over-rows at every budget
REAL_WIDTH = 8                  # ranked list length; the compute window admits ~2.37


# ---------------------------------------------------------------------------
# Cache replay
# ---------------------------------------------------------------------------
def load_trace():
    blob = TRACE.read_bytes()
    got = hashlib.sha256(blob).hexdigest()
    if got != TRACE_SHA256:
        raise SystemExit(f"trace sha256 mismatch: {got}")
    trace = json.loads(gzip.decompress(blob))
    if not (trace["complete"] and trace["cycles"] == 206):
        raise SystemExit("unexpected trace shape")
    return trace


def make_bank(restore, state, persistent, transient, policy):
    """Restore a LayerExpertSlotBank and (optionally) grow it to `persistent`
    slots, mirroring prefix-readiness-20260918's grow procedure exactly."""
    bank = restore(state, policy=policy, single_pool=True)
    extra = persistent - bank.persistent_slots
    if extra > 0:
        bank._slot_to_expert.extend([None] * extra)
    bank.persistent_slots = bank._persistent_capacity = persistent
    bank.slot_count = persistent + transient
    bank.transient_slots = transient
    if hasattr(bank, "_protected_cap"):
        bank._protected_cap = max(1, int(persistent * 0.8))
    return bank


def anchor_checks(trace, restore):
    """Reproduce two published miss counts EXACTLY before trusting the model."""
    out = {}
    # prefix-readiness-20260918: cap 102 (73+29), transition-window, plain plan().
    misses = routes = 0
    for layer, seq in trace["target_routes_by_layer"].items():
        bank = make_bank(restore, trace["initial_banks"][layer], 102, 48, "transition-window")
        for route in seq:
            misses += len(set(bank.plan(route, phase="decode").misses))
            routes += 1
    out["prefix_readiness_cap102"] = {"misses": misses, "routes": routes,
                                      "expected": [35164, 8240],
                                      "exact": misses == 35164 and routes == 8240}
    # mtp-verify-routes-20260913: cap 73+48, frequency, resident-first probe.
    records = 0
    for layer, seq in trace["target_routes_by_layer"].items():
        bank = make_bank(restore, trace["initial_banks"][layer], 73, 48, "frequency")
        for route in seq:
            plan = bank.try_plan_all_hits(route, phase="decode")
            if plan is None:
                plan = bank.plan(route, phase="decode")
            records += len(plan.misses)
    out["mtp_verify_cap73"] = {"records": records, "expected": 53999,
                               "exact": records == 53999}
    return out


def replay(trace, restore, persistent, transient=48, policy="transition-window",
           want_causal=False):
    """Deterministic replay -> per-(cycle,layer) demand-miss expert lists.

    If want_causal, also builds the online (causal, decay 0.98) cross-layer
    transition predictor from causal-prefetch-screen-20260917 and returns a
    top-16 ranked prediction for target layer L+1 at each (cycle, layer).
    """
    layers = sorted(trace["target_routes_by_layer"], key=int)
    routes = {int(l): trace["target_routes_by_layer"][l] for l in layers}
    banks = {int(l): make_bank(restore, trace["initial_banks"][l], persistent,
                               transient, policy) for l in layers}
    cyc = trace["cycles"]
    misses = [[None] * 40 for _ in range(cyc)]
    if want_causal:
        cross = np.zeros((40, N_EXPERTS, N_EXPERTS), np.float32)
        cross_den = np.zeros((40, N_EXPERTS), np.float32)
        seen = [False] * 40
        causal = {}
    for c in range(cyc):
        for L in range(40):
            raw = routes[L][c]
            plan = banks[L].plan(raw, phase="decode")
            misses[c][L] = list(plan.misses)
            if not want_causal:
                continue
            route = np.unique(raw)
            if L > 0:
                src = np.unique(routes[L - 1][c])
                cross[L] *= np.float32(0.98)
                cross_den[L] *= np.float32(0.98)
                cross[L][np.ix_(src, route)] += 1
                cross_den[L, src] += 1
            seen[L] = True
            T = L + 1
            if T < 40 and seen[T]:
                score = (cross[T, route] /
                         np.maximum(1, cross_den[T, route, None])).mean(axis=0)
                resident = np.fromiter(banks[T]._expert_to_slot.keys(), np.int64)
                score = score.copy()
                score[resident] = -1.0
                order = np.argsort(-score, kind="stable")
                causal[(c, L)] = order[score[order] > 0][:16].tolist()
            else:
                causal[(c, L)] = []
    total = sum(len(misses[c][L]) for c in range(cyc) for L in range(40))
    return misses, (causal if want_causal else None), total


def synthetic_predictions(misses, cyc):
    """For each TARGET (c,layer), a ranked list holding coverage ~0.74 of that
    layer's actual misses plus wrong ids, with the correct experts spread EVENLY
    so any prefix of the list keeps precision ~0.62 (a real predictor does not
    know which are correct, so window truncation must not concentrate hits).
    Keyed by target layer; predict() looks it up for target L+1.
    Clearly SYNTHETIC: no real predictor achieves this on the M6 verify trace."""
    pred = {}
    for c in range(cyc):
        for T in range(40):
            m = misses[c][T]
            n_correct = int(round(SYNTH_COVERAGE * len(m)))
            if n_correct == 0:
                pred[(c, T)] = []
                continue
            emitted = max(n_correct, int(round(n_correct / SYNTH_PRECISION)))
            mset = set(m)
            wrong = [e for e in range(N_EXPERTS) if e not in mset]
            out, ci, wi = [], 0, 0
            for i in range(emitted):
                take = int((i + 1) * n_correct / emitted) > int(i * n_correct / emitted)
                if take and ci < n_correct:
                    out.append(m[ci]); ci += 1
                else:
                    out.append(wrong[wi]); wi += 1
            pred[(c, T)] = out
    return pred


# ---------------------------------------------------------------------------
# Single-server SSD discrete-event model
# ---------------------------------------------------------------------------
def simulate(misses, cyc, *, arm, width, ring_size, rate_gbps, rec_bytes,
             causal=None, synth=None, real=None, pred_from=0, growth_s=GROWTH_S):
    """One SSD server at `rate_gbps` (decimal GB/s), records of `rec_bytes`.

    Per layer call: (1) an in-flight speculative read is NOT preempted -- it
    delays the demand reads; (2) demand reads (misses not already in the ring)
    have priority and block the main thread; (3) during the layer's compute
    window the SSD issues speculative reads for the predicted target layer(s)
    into a bounded prefetch ring (FIFO eviction at capacity).
    Cross-cycle arm also prefetches layers 0-3 during the draft gap.

    `real` supplies the capture-replay arm's ranked 1-ahead predictions keyed by
    the compute-window layer; `pred_from` suppresses prediction for windows below
    it (the capture arm keeps layers 0-3 unpredicted); `growth_s` is the one-time
    boundary cost (0 for the mid-stream 64-cycle capture replay).
    """
    rd = rec_bytes / (rate_gbps * 1e9)          # seconds per record
    t = growth_s                                 # one-time boundary growth (if any)
    ring = {}                                    # (layer,expert) -> completion time
    fifo = deque()                               # eviction order of ring keys
    inflight = None                              # (key, end_time) or None
    issued = useful = 0
    read_wait = 0.0

    def predict(c, L):
        if arm == "control":
            return []
        if L < pred_from:                        # keep the low layers unpredicted
            return []
        if arm == "oracle1":
            return [(L + 1, misses[c][L + 1])] if L + 1 < 40 else []
        if arm == "oracle2":
            return [(T, misses[c][T]) for T in (L + 1, L + 2) if T < 40]
        if arm == "causal":
            return [(L + 1, causal[(c, L)][:width])] if L + 1 < 40 else []
        if arm == "synthetic":
            return [(L + 1, synth[(c, L + 1)])] if L + 1 < 40 else []
        if arm == "real":
            return [(L + 1, real[(c, L)])] if (L + 1 < 40 and (c, L) in real) else []
        if arm == "cross_cycle":
            return []                            # handled in the draft gap only
        raise ValueError(arm)

    def fill(start, end, targets):
        """Run speculative reads back-to-back in [start, end); one may end in
        flight past `end`.  Returns the SSD clock after scheduling."""
        nonlocal issued, inflight
        s = start
        for T, experts in targets:
            for e in experts:
                key = (T, e)
                if key in ring or (inflight and inflight[0] == key):
                    continue
                if len(ring) + (1 if inflight else 0) >= ring_size:
                    while fifo and fifo[0] not in ring:
                        fifo.popleft()           # drop already-consumed keys
                    if not fifo:
                        return s                 # nothing evictable; ring is all in use
                    del ring[fifo.popleft()]     # FIFO-evict the oldest live record
                fin = s + rd
                issued += 1
                if fin <= end:
                    ring[key] = fin
                    fifo.append(key)
                    s = fin
                else:
                    inflight = (key, fin)
                    return s
        return s

    for c in range(cyc):
        # draft gap: main thread drafts; SSD idle (arm cross_cycle prefetches L0-3)
        if arm == "cross_cycle":
            fill(t, t + T_DRAFT, [(L, misses[c][L]) for L in range(4)])
        t += T_DRAFT
        for L in range(40):
            arrive = t
            if inflight is not None:
                key, fin = inflight
                if key not in ring:
                    ring[key] = fin
                    fifo.append(key)
                inflight = None
                demand_start = max(arrive, fin)
            else:
                demand_start = arrive
            # consume ring hits; count remaining demand reads
            need = 0
            for e in misses[c][L]:
                key = (L, e)
                if key in ring:
                    del ring[key]
                    useful += 1
                else:
                    need += 1
            demand_end = demand_start + need * rd
            read_wait += demand_end - arrive
            t = demand_end
            # compute window: SSD prefetches predicted target layer(s)
            fill(demand_end, demand_end + C_LAYER, predict(c, L))
            t = demand_end + C_LAYER
        t += T_ACCEPT + T_COMMIT
    total = t
    demand_records = sum(len(misses[c][L]) for c in range(cyc) for L in range(40))
    wasted = issued - useful
    return {
        "arm": arm, "width": width, "ring_size": ring_size,
        "rate_gbps": rate_gbps,
        "total_decode_s": total,
        "tps_1024": 1024.0 / total,
        "read_wait_exposed_s": read_wait,
        "demand_records": demand_records,
        "spec_issued": issued, "spec_useful": useful, "spec_wasted": wasted,
        "spec_precision": (useful / issued) if issued else None,
        "hidden_fraction": useful / demand_records if demand_records else 0.0,
        "extra_bytes_read": wasted * rec_bytes,
    }


# ---------------------------------------------------------------------------
# REAL-predictor capture arm: replay the 64 captured D5/M6 verify cycles with the
# actual ranked 1-ahead predictions.  Self-contained (numpy on the NPZ only; no
# mtplx, no 206-cycle trace, no bank replay -- the capture stores the real demand
# reads directly).  Layers 4-39 predicted (windows 3-38); layers 0-3 unpredicted.
# ---------------------------------------------------------------------------
def run_capture_comparison(rate, rb, *, capture_path=None,
                           feature=REAL_FEATURE, rule=REAL_MERGE, width=REAL_WIDTH):
    cap = rrc.load_capture(capture_path)
    cap_misses = rrc.demand_misses(cap)                     # (64, 40) demand reads
    real_pred = rrc.real_predictions(cap, feature, rule, width)
    cyc = rrc.CYCLES
    rd = rb / (rate * 1e9)
    common = dict(rate_gbps=rate, rec_bytes=rb, real=real_pred,
                  pred_from=rrc.FIRST_TARGET - 1, growth_s=0.0, width=width)

    rows = []
    for ring in (16, 32, 64):
        for arm in ("control", "real", "oracle1"):
            rows.append(simulate(cap_misses, cyc, arm=arm, ring_size=ring, **common))
    by = {(r["arm"], r["ring_size"]): r for r in rows}

    def frac(arm, ring=32):
        c0 = by[("control", ring)]["read_wait_exposed_s"]
        return (c0 - by[(arm, ring)]["read_wait_exposed_s"]) / c0 if c0 else 0.0

    def extrapolate(arm, ring=32):
        f = frac(arm, ring)
        removed = f * FULL_CONTROL_READ_WAIT_S
        total = FULL_CONTROL_TOTAL_S - removed
        return {"arm": arm, "ring_size": ring,
                "read_wait_reduction_fraction": f,
                "seconds_removed_from_control": removed,
                "implied_full_total_s": total, "implied_tps_1024": 1024.0 / total}

    extrap = {arm: {f"ring{ring}": extrapolate(arm, ring) for ring in (16, 32, 64)}
              for arm in ("real", "oracle1")}
    return {
        "purpose": "REAL-predictor arm: replay the 64 captured D5/M6 verify cycles "
                   "with the actual ranked 1-ahead predictions; no throughput claim, "
                   "no production code changed.",
        "source_commit": _git_head(),
        "capture_path": cap["path"], "capture_sha256": cap["sha256"],
        "captured_slots_per_layer": 105,
        "predictor": {"feature": rrc.FEATURES[feature], "merge_rule": rule,
                      "ranked_width": width,
                      "note": "window-limited: the compute window admits ~2.37 "
                              "records, so budget k>~3 is inert in the sim."},
        "cycles": cyc,
        "record_bytes": rb, "rate_gbps": rate, "seconds_per_record": rd,
        "c_layer_ms": C_LAYER * 1e3,
        "records_per_compute_window": C_LAYER / rd,
        "scope_note": "Layers 0-2 were not instrumented in the capture (0 demand "
                      "reads modelled there); layer 3 carries its 279 captured reads "
                      "but is unpredicted. Layers 0-3 are unpredicted and identical "
                      "across arms, so they do not bias the control-vs-arm delta.",
        "arms": rows,
        "primary_ring": 32,
        "extrapolation_anchor": {"full_control_total_s": FULL_CONTROL_TOTAL_S,
                                 "full_control_read_wait_s": FULL_CONTROL_READ_WAIT_S,
                                 "source": "f1-overlap-sim-20260919 control @ 111+48, rate 12.9"},
        "extrapolation": extrap,
        "curve": {rrc.FEATURES[feature]: {rule: rrc.curve(cap, feature, rule)}},
    }


def _print_capture(res):
    print(f"\nCAPTURE REPLAY  ({res['cycles']} cycles @ {res['captured_slots_per_layer']} "
          f"slots; predictor={res['predictor']['feature']}/{res['predictor']['merge_rule']}; "
          f"window admits {res['records_per_compute_window']:.2f} records)")
    hdr = ("arm", "ring", "total_s", "read_wait_s", "hidden%", "issued", "useful",
           "wasted", "prec")
    print(("{:<10}{:>5}{:>10}{:>12}{:>8}{:>8}{:>8}{:>8}{:>7}").format(*hdr))
    for a in res["arms"]:
        print(("{:<10}{:>5}{:>10.4f}{:>12.4f}{:>8.1f}{:>8}{:>8}{:>8}{:>7}").format(
            a["arm"], a["ring_size"], a["total_decode_s"], a["read_wait_exposed_s"],
            a["hidden_fraction"] * 100, a["spec_issued"], a["spec_useful"],
            a["spec_wasted"],
            "-" if a["spec_precision"] is None else f"{a['spec_precision']:.2f}"))
    for arm, rings in res["extrapolation"].items():
        for rk, e in rings.items():
            print(f"  extrapolate {arm} {rk}: read-wait reduction "
                  f"{e['read_wait_reduction_fraction']*100:.1f}% -> removes "
                  f"{e['seconds_removed_from_control']:.2f} s of {FULL_CONTROL_TOTAL_S} s "
                  f"-> {e['implied_full_total_s']:.2f} s / {e['implied_tps_1024']:.2f} TPS")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("docs/deepseek-v41/receipts/f1-overlap-sim-20260919/results.json"))
    ap.add_argument("--rate-gbps", type=float, default=DEFAULT_RATE)
    ap.add_argument("--rec-bytes", type=int, default=REC_BYTES)
    ap.add_argument("--anchors-only", action="store_true")
    ap.add_argument("--capture-only", action="store_true",
                    help="Run ONLY the REAL-predictor capture arm (numpy on the NPZ; "
                         "no mtplx, no 206-cycle trace) and write --capture-out.")
    ap.add_argument("--capture-out", type=Path,
                    default=Path("docs/deepseek-v41/receipts/f1-real-predictor-20260919/capture-sim.json"))
    args = ap.parse_args()

    if any(m == "mlx" or m.startswith("mlx.") for m in sys.modules):
        raise SystemExit("MLX leaked into the process")

    if args.capture_only:
        res = run_capture_comparison(args.rate_gbps, args.rec_bytes)
        res["mlx_imported"] = any(m == "mlx" or m.startswith("mlx.") for m in sys.modules)
        args.capture_out.parent.mkdir(parents=True, exist_ok=True)
        args.capture_out.write_text(json.dumps(res, indent=2) + "\n")
        _print_capture(res)
        return

    restore = runpy.run_path(str(HELPER))["restore"]
    trace = load_trace()

    t0 = time.perf_counter()
    anchors = anchor_checks(trace, restore)
    for name, a in anchors.items():
        if not a["exact"]:
            raise SystemExit(f"anchor {name} failed to reproduce: {a}")
    if args.anchors_only:
        print(json.dumps(anchors, indent=2))
        return

    # Control config (111+48) with the causal predictor built alongside.
    misses, causal, records111 = replay(trace, restore, 111, want_causal=True)
    synth = synthetic_predictions(misses, trace["cycles"])
    control_delta = (records111 - MEAS_RECORDS) / MEAS_RECORDS

    rate = args.rate_gbps
    rb = args.rec_bytes
    arms = []
    common = dict(rate_gbps=rate, rec_bytes=rb, causal=causal, synth=synth)
    arms.append(simulate(misses, trace["cycles"], arm="control", width=0,
                         ring_size=32, **common))
    for ring in (16, 32, 64):
        arms.append(simulate(misses, trace["cycles"], arm="oracle1", width=0,
                             ring_size=ring, **common))
        arms.append(simulate(misses, trace["cycles"], arm="oracle2", width=0,
                             ring_size=ring, **common))
    for w in (6, 10, 12, 16):
        arms.append(simulate(misses, trace["cycles"], arm="causal", width=w,
                             ring_size=32, **common))
    arms.append(simulate(misses, trace["cycles"], arm="synthetic", width=0,
                         ring_size=32, **common))
    arms.append(simulate(misses, trace["cycles"], arm="cross_cycle", width=0,
                         ring_size=32, **common))

    # Sensitivity: prefetch-ring size (oracles never fill it; the wasteful arms
    # do), SSD rate, and cache capacity.
    ring_rows = []
    for ring in (16, 32, 64):
        for arm, w in (("oracle2", 0), ("synthetic", 0), ("causal", 10)):
            row = simulate(misses, trace["cycles"], arm=arm, width=w,
                           ring_size=ring, **common)
            ring_rows.append(row)
    rate_rows = []
    for r in (11.0, MEAS_RATE, 12.9):
        for arm, w in (("control", 0), ("oracle2", 0), ("causal", 10),
                       ("synthetic", 0)):
            row = simulate(misses, trace["cycles"], arm=arm, width=w, ring_size=32,
                           rate_gbps=r, rec_bytes=rb, causal=causal, synth=synth)
            rate_rows.append(row)
    cap_rows = []
    for cap in (111, 115, 119):
        m_cap, _, rec_cap = replay(trace, restore, cap)
        ctl = simulate(m_cap, trace["cycles"], arm="control", width=0, ring_size=32,
                       rate_gbps=rate, rec_bytes=rb)
        orc = simulate(m_cap, trace["cycles"], arm="oracle2", width=0, ring_size=32,
                       rate_gbps=rate, rec_bytes=rb)
        cap_rows.append({"capacity_persistent": cap, "transient": 48,
                         "records": rec_cap,
                         "control_total_s": ctl["total_decode_s"],
                         "control_read_wait_s": ctl["read_wait_exposed_s"],
                         "oracle2_total_s": orc["total_decode_s"],
                         "oracle2_read_wait_s": orc["read_wait_exposed_s"]})

    result = {
        "purpose": "F1 overlap-schedule discrete-event screen; no throughput claim, "
                   "no production code changed. Prefetch is not modeled to alter the "
                   "cache policy (ring lives outside the pool), so demand misses are "
                   "deterministic and predictor-independent.",
        "source_commit": _git_head(),
        "elapsed_s": time.perf_counter() - t0,
        "mlx_imported": any(m == "mlx" or m.startswith("mlx.") for m in sys.modules),
        "inputs": {
            "trace": str(TRACE), "trace_sha256": TRACE_SHA256,
            "replay_helper": str(HELPER),
            "replay_helper_sha256": hashlib.sha256(HELPER.read_bytes()).hexdigest(),
            "expert_streaming_sha256":
                hashlib.sha256(Path("mtplx/expert_streaming.py").read_bytes()).hexdigest(),
        },
        "timing_model": {
            "record_bytes": rb, "source_record_bytes": SRC_REC_BYTES,
            "rate_gbps_primary": rate, "measured_rate_gbps": MEAS_RATE,
            "c_layer_ms": C_LAYER * 1e3, "t_draft_ms": T_DRAFT * 1e3,
            "t_accept_ms": T_ACCEPT * 1e3, "t_commit_ms": T_COMMIT * 1e3,
            "growth_s": GROWTH_S,
            "compute_distribution": "uniform over layer calls (per-layer arrays only "
                                    "exist for a different-capacity 77 s run)",
            "source": "extension-bank-20260919 full run (111 slots, 198 cycles, 73.7626 s)",
        },
        "anchors": anchors,
        "control_reconciliation": {
            "trace_cycles": trace["cycles"],
            "records_at_111plus48": records111,
            "measured_records_198cycle_run": MEAS_RECORDS,
            "records_delta_fraction": control_delta,
            "note": "Saved trace = 206 cycles @ 73 slots (bd542a39); measured "
                    "31,573/43.399 s best = a DIFFERENT 198-cycle run @ 111 slots "
                    "(d5f15e7a). Replaying the 206-cycle trace @ 111+48 is the closest "
                    "faithful control.",
        },
        "predictor_provenance": {
            "control/oracle1/oracle2/cross_cycle": "REAL trace (deterministic "
                "replay); oracle arms use the true future misses.",
            "causal": "REAL cross-layer transition predictor CPU-reproduced from "
                "causal-prefetch-screen-20260917 (online, causal, decay 0.98).",
            "synthetic": "SYNTHETIC reference (coverage 0.74 / precision 0.62, "
                "gate-oracle strength from the w89 AR trace); labelled, not real.",
        },
        "arms": arms,
        "sensitivity_ring": ring_rows,
        "sensitivity_rate": rate_rows,
        "sensitivity_capacity": cap_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    _print_table(result)


def _git_head():
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _print_table(result):
    print(f"trace cycles={result['control_reconciliation']['trace_cycles']} "
          f"records@111+48={result['control_reconciliation']['records_at_111plus48']} "
          f"(measured 198-cycle {MEAS_RECORDS}, "
          f"delta {result['control_reconciliation']['records_delta_fraction']*100:+.2f}%)")
    hdr = ("arm", "w", "ring", "total_s", "tps", "read_wait_s", "hidden%",
           "issued", "useful", "wasted", "prec")
    print(("{:<12}{:>3}{:>5}{:>10}{:>7}{:>12}{:>8}{:>8}{:>8}{:>8}{:>7}").format(*hdr))
    for a in result["arms"]:
        print(("{:<12}{:>3}{:>5}{:>10.3f}{:>7.2f}{:>12.3f}{:>8.1f}{:>8}{:>8}{:>8}{:>7}")
              .format(a["arm"], a["width"], a["ring_size"], a["total_decode_s"],
                      a["tps_1024"], a["read_wait_exposed_s"],
                      a["hidden_fraction"] * 100, a["spec_issued"], a["spec_useful"],
                      a["spec_wasted"],
                      "-" if a["spec_precision"] is None else f"{a['spec_precision']:.2f}"))


if __name__ == "__main__":
    main()
