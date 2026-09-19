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
             causal=None, synth=None, real=None, pred_from=0, growth_s=GROWTH_S,
             compute_scale=1.0):
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
    cl = C_LAYER * compute_scale                 # per-layer compute window (scaled)
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
            fill(demand_end, demand_end + cl, predict(c, L))
            t = demand_end + cl
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


# ===========================================================================
# What-if knobs (f1-stack-sim-20260919).  All four knobs run on the captured
# demand stream and the REAL post-attention-router / max 1-ahead predictor; no
# production code, no GPU.  Every knob TABLE reports HELD-OUT numbers (capture
# cycles 32-63); any fitted parameter is chosen on cycles 0-31.  The stack
# (knob 4) borrows the f1-real-predictor receipt's extrapolation rule verbatim.
# ===========================================================================
N_PLANES = 3                        # a record is 3 weight planes: gate, up, down
PLANE_BYTES = REC_BYTES // N_PLANES  # 5,898,240 B ~ 5.9 MB per plane
RING_MEM_MB_PER_REC = REC_BYTES / 1e6   # 17.69 MB per resident record (decimal)
STRIPED_RATE = 18.0                 # GB/s, a second striped SSD
STACK_CAPACITIES = (105, 111, 119)  # captured 105, +6, +14 persistent rows/layer
STACK_SCALES = (1.0, 0.8, 0.6)      # compute-window scale (faster compute -> smaller window)
STACK_RATES = (DEFAULT_RATE, STRIPED_RATE)   # 12.9 (one drive), 18.0 (striped pair)
ORACLE2_FULL_S = 51.197             # parent f1-overlap-sim oracle-2-ahead full total = the "51.2 s" target
HELDOUT_LO, HELDOUT_HI = 32, 64     # held-out capture cycles
TRAIN_LO, TRAIN_HI = 0, 32          # threshold-fitting cycles


def scored_predictions(cap, feature=REAL_FEATURE, rule=REAL_MERGE, width=12):
    """{(cycle, window-layer L): [(expert, merged_score), ...]} ranked desc, READY
    residents excluded.  L = target-1 (the layer whose compute window issues the
    prediction for target L+1).  Built from rescore's merge + rank helpers so the
    ranked ids match real_predictions() exactly; scores are exposed for knob 2."""
    merged = rrc.merged_scores(cap["scores"][feature], rule)
    order, m = rrc._ranked(merged, cap["physical"])
    top = order[..., :width]
    pred = {}
    for c in range(rrc.CYCLES):
        for idx in range(rrc.N_TARGET):
            L = idx + rrc.FIRST_TARGET - 1
            row = top[c, idx]
            vals = m[c, idx, row]
            pred[(c, L)] = [(int(e), float(v)) for e, v in zip(row, vals)
                            if np.isfinite(v)]
    return pred


def _slice_stream(cap_misses, scored, lo, hi):
    """Sub-stream of cap_misses[lo:hi] with predictions re-keyed to 0-based cycles.
    The real 1-ahead predictor is within-cycle (target L+1 from cycle c's own
    features), so slicing cycles introduces no cross-cycle leakage."""
    sub = cap_misses[lo:hi]
    sscored = {(c - lo, L): v for (c, L), v in scored.items() if lo <= c < hi}
    return sub, sscored, hi - lo


def simulate_planes(misses, cyc, *, arm, scored=None, n_planes=1, preempt="record",
                    budget_k=None, score_threshold=None, window_stop=False,
                    ring_size=32, rate_gbps=DEFAULT_RATE, rec_bytes=REC_BYTES,
                    pred_from=0, growth_s=0.0, compute_scale=1.0):
    """Plane-granular single-server SSD DES for the REAL 1-ahead predictor.

    A record is `n_planes` planes of rec_bytes/n_planes each.  `preempt` sets how a
    demand read treats the one speculative plane in flight at a layer boundary:
      'record' -- waits the whole record  (== the parent simulate() at n_planes=1),
      'plane'  -- waits at most one plane (~0.46 ms), remaining planes deferred,
      'full'   -- ideal zero-delay bound (the in-flight plane is abandoned).
    A partially-read speculative record that becomes a demand hit reads only its
    remaining planes.  Issue policy (knob 2): `budget_k` caps per-call issuance,
    `score_threshold` gates on the merged predictor score, `window_stop` refuses to
    start a plane that cannot finish inside the compute window.  arm='control'
    issues nothing.  Held-out slicing is the caller's job (pass a sliced stream)."""
    rd = rec_bytes / (rate_gbps * 1e9)
    rp = rd / n_planes                  # seconds per plane
    cl = C_LAYER * compute_scale
    ring = {}                           # (layer,expert) -> planes_done (0 only transiently)
    fifo = deque()                      # FIFO eviction order of ring keys
    inflight = None                     # (key, planes_done_before_current, plane_fin)
    issued = useful = 0
    spec_planes = useful_spec_planes = 0
    read_wait = 0.0
    t = growth_s

    def want(c, L):
        if arm == "control" or L < pred_from or scored is None:
            return []
        if L + 1 >= 40 or (c, L) not in scored:
            return []
        lst = scored[(c, L)]
        if score_threshold is not None:
            lst = [(e, v) for (e, v) in lst if v >= score_threshold]
        if budget_k is not None:
            lst = lst[:budget_k]
        return [(L + 1, [e for (e, _) in lst])]

    def fill(start, end, targets):
        nonlocal issued, inflight, spec_planes
        s = start
        for T, experts in targets:
            for e in experts:
                key = (T, e)
                done = ring.get(key, 0)
                if done >= n_planes:
                    continue
                if inflight is not None and inflight[0] == key:
                    continue
                if window_stop and (end - s) < rp:
                    return s                       # cannot even start a plane
                if key not in ring:
                    if len(ring) + (1 if inflight else 0) >= ring_size:
                        while fifo and fifo[0] not in ring:
                            fifo.popleft()
                        if not fifo:
                            return s               # nothing evictable; ring all in use
                        del ring[fifo.popleft()]
                    ring[key] = 0
                    fifo.append(key)
                    done = 0
                    issued += 1                    # a fresh speculative record
                for _ in range(n_planes - done):
                    if window_stop and (end - s) < rp:
                        return s                   # refuse a plane that would go in-flight
                    plane_fin = s + rp
                    spec_planes += 1
                    if plane_fin <= end:
                        ring[key] += 1
                        s = plane_fin
                    else:
                        inflight = (key, ring[key], plane_fin)
                        return s
        return s

    for c in range(cyc):
        t += T_DRAFT
        for L in range(40):
            arrive = t
            boundary_wait = 0.0
            if inflight is not None:
                key, p_before, plane_fin = inflight
                if preempt == "record":
                    rec_fin = plane_fin + (n_planes - p_before - 1) * rp
                    boundary_wait = max(0.0, rec_fin - arrive)
                    ring[key] = n_planes
                elif preempt == "plane":
                    boundary_wait = max(0.0, plane_fin - arrive)
                    ring[key] = p_before + 1
                else:                              # 'full': zero demand delay
                    if p_before > 0:
                        ring[key] = p_before
                    elif key in ring:              # nothing landed; drop transient entry
                        del ring[key]
                inflight = None
            demand_start = arrive + boundary_wait
            demand_time = 0.0
            for e in misses[c][L]:
                key = (L, e)
                if key in ring:
                    p = ring.pop(key)
                    demand_time += (n_planes - p) * rp   # only the missing planes
                    useful += 1
                    useful_spec_planes += p
                else:
                    demand_time += rd
            demand_end = demand_start + demand_time
            read_wait += demand_end - arrive
            t = demand_end
            fill(demand_end, demand_end + cl, want(c, L))
            t = demand_end + cl
        t += T_ACCEPT + T_COMMIT
    total = t
    demand_records = sum(len(misses[c][L]) for c in range(cyc) for L in range(40))
    wasted = issued - useful
    return {
        "arm": arm, "n_planes": n_planes, "preempt": preempt,
        "budget_k": budget_k, "score_threshold": score_threshold,
        "window_stop": window_stop, "ring_size": ring_size,
        "rate_gbps": rate_gbps, "compute_scale": compute_scale,
        "total_decode_s": total, "read_wait_exposed_s": read_wait,
        "demand_records": demand_records,
        "spec_issued": issued, "spec_useful": useful, "spec_wasted": wasted,
        "spec_precision": (useful / issued) if issued else None,
        "hidden_fraction": useful / demand_records if demand_records else 0.0,
        "spec_bytes_read": spec_planes * (rec_bytes / n_planes),
        "useful_spec_planes": useful_spec_planes,
        "extra_bytes_read": wasted * rec_bytes,
    }


def recycle_analysis(cap_misses, scored, *, ring_sizes=(32, 64, 128), horizon=2,
                     budget_k=3, lo=HELDOUT_LO, hi=HELDOUT_HI):
    """Knob 3 -- waste recycling.  A wasted speculative record (issued for layer T
    at cycle c but not a demand miss then) sits in a per-layer persistent ring
    instead of being FIFO-evicted at the window.  Count how often it becomes a
    demand hit for the SAME layer within the next `horizon` cycles.  These are
    EXTRA hidden reads, disjoint from the 1-ahead same-cycle hits.  Ring capacity
    is `ring_size` records (evict globally oldest); memory = ring_size x 17.7 MB.
    Warm-up uses all earlier cycles; only hits in [lo,hi) are counted."""
    out = {}
    n_cyc = len(cap_misses)
    for R in ring_sizes:
        ring = {}                        # (layer,expert) -> issue_cycle
        order = deque()
        extra = 0
        for c in range(n_cyc):
            for T in range(rrc.FIRST_TARGET, 40):
                pred = {e for e, _ in scored.get((c, T - 1), [])[:budget_k]}
                dmiss = set(cap_misses[c][T])
                for e in dmiss - pred:            # not predicted this cycle
                    key = (T, e)
                    if key in ring and 1 <= (c - ring[key]) <= horizon:
                        if lo <= c < hi:
                            extra += 1
                        del ring[key]
                for e in dmiss & pred:            # 1-ahead hit consumes its record
                    ring.pop((T, e), None)
                for e in pred - dmiss:            # wasted this cycle -> keep for recycling
                    key = (T, e)
                    if key not in ring and len(ring) >= R:
                        while order and order[0] not in ring:
                            order.popleft()
                        if order:
                            del ring[order.popleft()]
                    ring[key] = c
                    order.append(key)
        recs = sum(len(cap_misses[c][T]) for c in range(lo, hi) for T in range(40))
        out[R] = {"ring_records": R, "ring_mem_mb": round(R * RING_MEM_MB_PER_REC, 1),
                  "horizon_cycles": horizon, "budget_k": budget_k,
                  "extra_hidden_reads_heldout": extra,
                  "heldout_demand_records": recs,
                  "extra_hidden_fraction": (extra / recs) if recs else 0.0}
    return out


def _fit_threshold(cap_misses, scored, k, *, npl, pre, rate, rec_bytes, pred_from):
    """Choose the merged-score threshold on TRAIN cycles 0-31 that minimises train
    read wait for budget k; None (no gating) is always a candidate."""
    tr_m, tr_s, tr_c = _slice_stream(cap_misses, scored, TRAIN_LO, TRAIN_HI)
    vals = [v for lst in tr_s.values() for (_, v) in lst[:k]]
    cands = [None]
    if vals:
        va = np.array(vals)
        cands += [float(np.quantile(va, q)) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
    best_th, best_rw = None, None
    for th in cands:
        r = simulate_planes(tr_m, tr_c, arm="real", scored=tr_s, n_planes=npl,
                            preempt=pre, budget_k=k, score_threshold=th,
                            rate_gbps=rate, rec_bytes=rec_bytes, pred_from=pred_from)
        if best_rw is None or r["read_wait_exposed_s"] < best_rw - 1e-12:
            best_th, best_rw = th, r["read_wait_exposed_s"]
    return best_th


def run_stack_sim(restore, trace, cap, *, rec_bytes=REC_BYTES):
    """Compute all four knobs and the stacked grid.  Needs the 206-cycle trace
    (for the capacity control anchors, via the f1 receipt's trace-replay method)
    and the capture NPZ (for the real predictor's held-out read-wait reduction)."""
    cyc_all = trace["cycles"]
    scored = scored_predictions(cap)
    cap_misses = rrc.demand_misses(cap)
    pred_from = rrc.FIRST_TARGET - 1
    ho_m, ho_s, ho_c = _slice_stream(cap_misses, scored, HELDOUT_LO, HELDOUT_HI)
    ho_records = sum(len(ho_m[c][L]) for c in range(ho_c) for L in range(40))
    base = dict(scored=ho_s, ring_size=32, rec_bytes=rec_bytes, pred_from=pred_from)

    def cap_control(rate):
        return simulate_planes(ho_m, ho_c, arm="control", rate_gbps=rate, **base)

    def real_arm(rate=DEFAULT_RATE, scale=1.0, **cfg):
        return simulate_planes(ho_m, ho_c, arm="real", rate_gbps=rate,
                               compute_scale=scale, **cfg, **base)

    # ---- Knob 1: speculative read granularity / preemption (held-out, 12.9) ----
    k1_ctrl = cap_control(DEFAULT_RATE)
    knob1 = []
    for label, npl, pre in (("baseline (record, no preempt)", 1, "record"),
                            ("plane preempt (<=1 plane wait)", N_PLANES, "plane"),
                            ("ideal full-preempt (0 delay)", N_PLANES, "full")):
        r = real_arm(budget_k=3, n_planes=npl, preempt=pre)
        r["label"] = label
        r["readwait_reduction_vs_control"] = (
            (k1_ctrl["read_wait_exposed_s"] - r["read_wait_exposed_s"])
            / k1_ctrl["read_wait_exposed_s"])
        knob1.append(r)

    # ---- Knob 2: issue policy (plane base; threshold fit on 0-31, report 32-63) --
    # Policy configs = k in {1,2,3} (fitted threshold, no stop) plus a window-stop
    # arm at k=3 (where the window otherwise leaves a plane in flight).  The best
    # config is chosen by TRAIN read wait (no held-out peeking); window-stop is a
    # scheduling rule, not a fitted parameter, so selecting it is leak-free.
    tr_m, tr_s, tr_c = _slice_stream(cap_misses, scored, TRAIN_LO, TRAIN_HI)

    def _train_rw(k, th, ws):
        return simulate_planes(tr_m, tr_c, arm="real", scored=tr_s, n_planes=N_PLANES,
                               preempt="plane", budget_k=k, score_threshold=th,
                               window_stop=ws, rate_gbps=DEFAULT_RATE,
                               rec_bytes=rec_bytes, pred_from=pred_from)["read_wait_exposed_s"]

    knob2 = []
    train_best = None                                   # (train_rw, k, th, window_stop)
    policy_arms = []
    for k in (1, 2, 3):
        th = _fit_threshold(cap_misses, scored, k, npl=N_PLANES, pre="plane",
                            rate=DEFAULT_RATE, rec_bytes=rec_bytes, pred_from=pred_from)
        policy_arms.append((f"k={k}, fitted threshold", k, th, False))
    policy_arms.append(("k=3, window-stop (refuse in-flight plane)", 3, None, True))
    for label, k, th, ws_flag in policy_arms:
        r = real_arm(budget_k=k, n_planes=N_PLANES, preempt="plane",
                     score_threshold=th, window_stop=ws_flag)
        r["label"] = label
        r["fitted_threshold"] = th
        r["window_stop"] = ws_flag
        r["readwait_reduction_vs_control"] = (
            (k1_ctrl["read_wait_exposed_s"] - r["read_wait_exposed_s"])
            / k1_ctrl["read_wait_exposed_s"])
        knob2.append(r)
        trw = _train_rw(k, th, ws_flag)
        if train_best is None or trw < train_best[0] - 1e-12:
            train_best = (trw, k, th, ws_flag)
    sel_k, sel_th, sel_ws = train_best[1], train_best[2], train_best[3]

    # ---- Knob 3: waste recycling (held-out extra hidden reads) ----
    knob3 = recycle_analysis(cap_misses, scored, ring_sizes=(32, 64, 128),
                             horizon=2, budget_k=sel_k)
    recycle_ring = 64                                   # ring used for the stack bonus
    recycle_bonus_frac = knob3[recycle_ring]["extra_hidden_fraction"]

    # ---- Best real-predictor configuration for the stack ----
    best_cfg = dict(n_planes=N_PLANES, preempt="plane", window_stop=sel_ws,
                    budget_k=sel_k,
                    score_threshold=sel_th)

    # capacity control anchors via the f1 receipt's trace-replay method
    cap_anchor = {}
    for C in STACK_CAPACITIES:
        m_cap, _, rec_cap = replay(trace, restore, C)
        cell = {"records": rec_cap}
        for R in STACK_RATES:
            for s in STACK_SCALES:
                ctl = simulate(m_cap, cyc_all, arm="control", width=0, ring_size=32,
                               rate_gbps=R, rec_bytes=rec_bytes, compute_scale=s)
                cell[f"{R}|{s}"] = {"control_total_s": ctl["total_decode_s"],
                                    "control_read_wait_s": ctl["read_wait_exposed_s"]}
        cap_anchor[C] = cell

    # held-out real-predictor read-wait-reduction fraction per (rate, scale)
    frac = {}
    for R in STACK_RATES:
        cc = cap_control(R)
        for s in STACK_SCALES:
            arm = real_arm(rate=R, scale=s, **best_cfg)
            f_sim = ((cc["read_wait_exposed_s"] - arm["read_wait_exposed_s"])
                     / cc["read_wait_exposed_s"])
            frac[(R, s)] = {"f_sim": f_sim, "recycle_bonus": recycle_bonus_frac,
                            "f_total": f_sim + recycle_bonus_frac,
                            "real_precision": arm["spec_precision"],
                            "real_hidden_fraction": arm["hidden_fraction"]}

    # stacked grid + extrapolation (SAME RULE as f1-real-predictor receipt)
    stack = []
    for C in STACK_CAPACITIES:
        for R in STACK_RATES:
            for s in STACK_SCALES:
                anc = cap_anchor[C][f"{R}|{s}"]
                fr = frac[(R, s)]
                removed = fr["f_total"] * anc["control_read_wait_s"]
                implied_total = anc["control_total_s"] - removed
                stack.append({
                    "persistent": C, "rate_gbps": R, "compute_scale": s,
                    "control_total_s": anc["control_total_s"],
                    "control_read_wait_s": anc["control_read_wait_s"],
                    "readwait_reduction_fraction": fr["f_total"],
                    "f_sim_planes_policy": fr["f_sim"],
                    "f_recycle_bonus": fr["recycle_bonus"],
                    "seconds_removed": removed,
                    "real_implied_total_s": implied_total,
                    "real_implied_tps": 1024.0 / implied_total,
                    "reaches_51_2_real": implied_total <= ORACLE2_FULL_S,
                    "reaches_51_2_control_alone": anc["control_total_s"] <= ORACLE2_FULL_S,
                })

    reach = [c for c in stack if c["real_implied_total_s"] <= ORACLE2_FULL_S]
    reach.sort(key=lambda c: (STACK_RATES.index(c["rate_gbps"]),   # 12.9 before 18.0
                              STACK_SCALES.index(c["compute_scale"]),  # 1.0 first
                              STACK_CAPACITIES.index(c["persistent"])))  # 105 first
    return {
        "purpose": "F1 stack-sim: four what-if knobs on the REAL 1-ahead predictor. "
                   "CPU-only screen; no throughput claim, no production code, no GPU. "
                   "All knob tables are HELD-OUT (capture cycles 32-63); fitted "
                   "parameters chosen on cycles 0-31.",
        "source_commit": _git_head(),
        "capture_path": cap["path"], "capture_sha256": cap["sha256"],
        "record_bytes": rec_bytes, "plane_bytes": PLANE_BYTES, "n_planes": N_PLANES,
        "held_out_cycles": [HELDOUT_LO, HELDOUT_HI - 1],
        "held_out_demand_records": ho_records,
        "predictor": {"feature": rrc.FEATURES[REAL_FEATURE], "merge_rule": REAL_MERGE},
        "knob1_granularity": {"control": k1_ctrl, "arms": knob1},
        "knob2_issue_policy": {"arms": knob2, "selected_k": sel_k,
                               "selected_threshold": sel_th,
                               "selected_window_stop": sel_ws,
                               "selection": "min TRAIN (0-31) read wait"},
        "knob3_recycle": {"rings": knob3, "stack_ring_used": recycle_ring,
                          "stack_bonus_fraction": recycle_bonus_frac},
        "best_config": best_cfg,
        "stack_capacity_anchors": cap_anchor,
        "stack_fraction_by_rate_scale": {f"{R}|{s}": frac[(R, s)]
                                         for R in STACK_RATES for s in STACK_SCALES},
        "extrapolation_rule": (
            "Same as f1-real-predictor-20260919: measure the real predictor's "
            "read-wait-reduction fraction f on the capture (here HELD-OUT cycles "
            "32-63), then apply f to the full 1,024-token control read wait at that "
            "cell's capacity/rate (from the 206-cycle trace replay, the f1 receipt's "
            "capacity method) and subtract from the full control total. This is an "
            "EXTRAPOLATION: it assumes the held-out hidden fraction holds across the "
            "run and holds the fraction constant across capacity (capacity enters "
            "only through the control anchor)."),
        "target_51_2_s": ORACLE2_FULL_S,
        "stack": stack,
        "cells_reaching_51_2": reach,
        "mlx_imported": any(m == "mlx" or m.startswith("mlx.") for m in sys.modules),
    }


def _print_stack(res):
    print("\nKNOB 1 (granularity/preemption, held-out 32-63, rate 12.9)")
    c = res["knob1_granularity"]["control"]
    print(f"  control read wait {c['read_wait_exposed_s']:.4f} s "
          f"({c['demand_records']} demand records)")
    for a in res["knob1_granularity"]["arms"]:
        print(f"  {a['label']:<32} read_wait {a['read_wait_exposed_s']:.4f} s  "
              f"hidden {a['hidden_fraction']*100:5.1f}%  prec {a['spec_precision']:.2f}  "
              f"rw-reduction {a['readwait_reduction_vs_control']*100:5.1f}%")
    print("\nKNOB 2 (issue policy, plane base; threshold fit 0-31, report 32-63)")
    for a in res["knob2_issue_policy"]["arms"]:
        th = a.get("fitted_threshold", None)
        ths = "none" if th is None else f"{th:.3f}"
        print(f"  {a['label']:<38} thr {ths:>6}  read_wait {a['read_wait_exposed_s']:.4f} s  "
              f"hidden {a['hidden_fraction']*100:5.1f}%  prec {a['spec_precision']:.2f}  "
              f"rw-red {a['readwait_reduction_vs_control']*100:5.1f}%")
    print(f"  selected k={res['knob2_issue_policy']['selected_k']} "
          f"threshold={res['knob2_issue_policy']['selected_threshold']} "
          f"window_stop={res['knob2_issue_policy']['selected_window_stop']}")
    print("\nKNOB 3 (waste recycling, held-out extra hidden reads)")
    for R, d in res["knob3_recycle"]["rings"].items():
        print(f"  ring {R:>3} recs ({d['ring_mem_mb']} MB): "
              f"{d['extra_hidden_reads_heldout']} extra hidden "
              f"({d['extra_hidden_fraction']*100:.2f}% of held-out demand)")
    print("\nKNOB 4 STACK (cap x compute-scale x rate)  best cfg="
          f"{res['best_config']}  recycle bonus "
          f"{res['knob3_recycle']['stack_bonus_fraction']*100:.2f}%")
    hdr = ("persist", "rate", "scale", "ctl_total", "ctl_rw", "f%", "real_total", "real_TPS", "<=51.2")
    print(("{:>8}{:>6}{:>6}{:>11}{:>9}{:>7}{:>11}{:>9}{:>7}").format(*hdr))
    for cell in res["stack"]:
        print(("{:>8}{:>6}{:>6}{:>11.3f}{:>9.3f}{:>7.1f}{:>11.3f}{:>9.2f}{:>7}").format(
            cell["persistent"], cell["rate_gbps"], cell["compute_scale"],
            cell["control_total_s"], cell["control_read_wait_s"],
            cell["readwait_reduction_fraction"] * 100, cell["real_implied_total_s"],
            cell["real_implied_tps"], "Y" if cell["reaches_51_2_real"] else ""))
    print(f"\ncells reaching <=51.2 s (real): {len(res['cells_reaching_51_2'])} "
          f"of {len(res['stack'])}")
    for cell in res["cells_reaching_51_2"]:
        tag = " (control alone)" if cell["reaches_51_2_control_alone"] else ""
        print(f"  persist {cell['persistent']}, rate {cell['rate_gbps']}, "
              f"scale {cell['compute_scale']} -> {cell['real_implied_total_s']:.2f} s / "
              f"{cell['real_implied_tps']:.2f} TPS{tag}")


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
    ap.add_argument("--stack", action="store_true",
                    help="Run the four what-if knobs + stacked grid (needs the "
                         "206-cycle trace AND the capture NPZ) and write --stack-out.")
    ap.add_argument("--stack-out", type=Path,
                    default=Path("docs/deepseek-v41/receipts/f1-stack-sim-20260919/stack-sim.json"))
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

    if args.stack:
        anchors = anchor_checks(trace, restore)
        for name, a in anchors.items():
            if not a["exact"]:
                raise SystemExit(f"anchor {name} failed to reproduce: {a}")
        cap = rrc.load_capture()
        res = run_stack_sim(restore, trace, cap, rec_bytes=args.rec_bytes)
        res["anchors"] = anchors
        args.stack_out.parent.mkdir(parents=True, exist_ok=True)
        args.stack_out.write_text(json.dumps(res, indent=2) + "\n")
        _print_stack(res)
        return

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
