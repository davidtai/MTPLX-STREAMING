#!/usr/bin/env python3
"""F10: row-group pipeline discrete-event simulator for DeepSeek-V4.1 M6 verify.

CPU ONLY (a NoMLX meta-path finder hard-blocks any `mlx` import), numpy only,
single process; run under `nice -n 19`.  Full method, provenance and results are
in docs/deepseek-v41/receipts/f10-rowgroup-pipeline-sim-20260919/README.md.

The idea under test (David's; we simulate it, we do not judge it by intuition):
each routed verify layer is serial today -- GPU stage G (ends in the routing-index
host sync) -> host route prep -> SSD reads of the missing experts (GPU idle but for
hit experts) -> miss-expert compute + next layer's G.  "Row-group pipelining"
splits the M=6 verify rows into two causal groups (A = rows 0-2, B = rows 3-5) and
runs them as a two-stage pipeline staggered by ONE layer on a single host thread,
so one group's GPU stage overlaps the other group's SSD reads.  Every read stays an
exact demand read; routing/experts/arithmetic are row-independent and unchanged.

This module DECOMPOSES the f1 lumped per-layer compute (C_LAYER = 3.2520 ms) into
G + host_pre + host_post + expert-compute, puts hit- AND miss-expert compute on a
FIFO GPU queue (assignment-count x c_assign; addendum), and replays the REAL
LayerExpertSlotBank policy with TWO route observations per layer (group A then B)
so the split's effect on the transition-window cache is measured, not assumed.  The
control (one group, full route, G_full) reproduces the f1 74.667 s / 43.395 s
anchor; the cache replay reuses the sibling f1 sim's validated helpers unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import sys
import time
from pathlib import Path


class _NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise RuntimeError("rowgroup_pipeline_sim is CPU-only; MLX is forbidden")


sys.meta_path.insert(0, _NoMLX())
# Resolve mtplx and the sibling sim to the CURRENT worktree.  Run from the root.
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402  (after the guard, like the sibling receipts)

# The f1 sim is imported (not modified): its NoMLX guard, validated timing
# constants, trace loader, bank-grow helper and the two published anchor checks.
import overlap_schedule_sim as osim  # noqa: E402

# ---- timing skeleton, inherited verbatim from f1 / extension-bank-20260919 ----
REC_BYTES = osim.REC_BYTES                 # 17,694,720 B ("17.7 MB")
C_LAYER_MS = osim.C_LAYER * 1e3            # 3.25201 ms lumped non-read compute/layer
GROWTH_S = osim.GROWTH_S                   # 2.1957 s one-time boundary cost
T_DRAFT = osim.T_DRAFT                      # per-cycle draft (s)
T_ACCEPT = osim.T_ACCEPT
T_COMMIT = osim.T_COMMIT
DEFAULT_RATE = osim.DEFAULT_RATE           # 12.9 GB/s (decimal)
MEAS_RECORDS = osim.MEAS_RECORDS           # 31,573 measured 198-cycle records
MEAS_CYCLES = osim.MEAS_CYCLES             # 198 (retained run); trace has 206

# ---- decomposition parameters (task defaults; host_post calibrated below) ----
G_FULL_MS = 2.05          # full-batch GPU stage (hyper-conn + attn + gate + route sync)
HOST_PRE_MS = 0.35        # host route prep, paid after each barrier (per group)
G_HALF_SWEEP = (1.2, 1.5, 1.8, 2.05)   # half-batch G: small-op chains are launch-bound
C_ASSIGN_SWEEP = (0.025, 0.035, 0.045)  # ms per (row,expert) assignment; central 0.035
C_ASSIGN_CENTRAL = 0.035
RATE_SWEEP = (12.9, 14.0)
CAP_SWEEP = (111, 115)                  # persistent slots (+48 transient)
TRANSIENT = 48
PIPE_EXTRA_MS = 3.0        # extra per-cycle: split/concat + the second set of syncs
THIN_SWEEP = (0.0, 0.05, 0.10, 0.20)   # synthetic byte-lever record thinning
THIN_SEED = 20260919
M_ROWS = 6
EXPERTS_PER_ROW = 6
GROUP_A_ROWS = 3           # ceil(M/2); A = rows 0..2 (ids 0:18), B = rows 3..5 (18:36)


# ---------------------------------------------------------------------------
# Cache replay -> per-(cycle,layer) demand records + assignment counts.
# Uses the f1 sim's exact bank-grow helper so records match its anchors.
# ---------------------------------------------------------------------------
def replay(trace, restore, persistent, *, n_groups, transient=TRANSIENT,
           policy="transition-window"):
    """Deterministic replay of the real LayerExpertSlotBank policy.

    n_groups=1: the single full 36-id route per (cycle,layer) -- the control.
    n_groups=2: split each route into A=ids[0:18] (rows 0-2) and B=ids[18:36]
    (rows 3-5) and call plan() twice per layer (A then B) on the SAME bank, so
    the transition-window policy sees two observations and B sees A's just-admitted
    experts as residents (an expert needed by both groups is read once, by A).

    Returns arrays keyed [group][cycle][layer]:
      rec[g] = demand records (len of plan.misses) for group g's route
      ma[g]  = MISS assignments = # of the group's (row,expert) pairs whose expert
               is in that route's miss set (>= rec because an expert can serve rows)
    plus total records and the zero-miss imbalance tallies.
    """
    routes = {int(lk): trace["target_routes_by_layer"][lk]
              for lk in trace["target_routes_by_layer"]}
    banks = {L: osim.make_bank(restore, trace["initial_banks"][str(L)], persistent,
                               transient, policy) for L in range(40)}
    cyc = trace["cycles"]
    split = GROUP_A_ROWS * EXPERTS_PER_ROW      # 18
    rec = [np.zeros((cyc, 40), np.int32) for _ in range(n_groups)]
    ma = [np.zeros((cyc, 40), np.int32) for _ in range(n_groups)]
    zero = [0] * n_groups
    either_zero = 0
    total = 0
    for c in range(cyc):
        for L in range(40):
            raw = routes[L][c]
            if n_groups == 1:
                subroutes = [raw]
            else:
                subroutes = [raw[:split], raw[split:]]
            zero_here = False
            for g, sub in enumerate(subroutes):
                plan = banks[L].plan(sub, phase="decode")
                mset = set(plan.misses)
                nrec = len(plan.misses)
                rec[g][c, L] = nrec
                ma[g][c, L] = sum(1 for e in sub if e in mset)
                total += nrec
                if nrec == 0:
                    zero[g] += 1
                    zero_here = True
            if zero_here:
                either_zero += 1
    calls = cyc * 40
    return {
        "n_groups": n_groups, "persistent": persistent, "cycles": cyc,
        "rec": rec, "ma": ma, "nassign_per_group": (M_ROWS * EXPERTS_PER_ROW
                                                     if n_groups == 1 else split),
        "total_records": total, "layer_calls": calls,
        "group_calls": calls * n_groups,
        "zero_miss_per_group": zero,
        "zero_miss_group_fraction": [z / calls for z in zero],
        "either_zero_calls": either_zero,
        "either_zero_fraction": either_zero / calls,
        "mean_miss_assign_per_layercall": float(
            sum(int(ma[g].sum()) for g in range(n_groups)) / calls),
    }


def thin_records(rec_arrays, frac, seed):
    """Synthetic byte-lever: keep each demand record with prob (1-frac), fixed seed.
    SSD load only -- miss-assignment (E_miss compute) is unchanged.  Deterministic
    given (seed, traversal order group->cycle->layer)."""
    if frac <= 0:
        return rec_arrays
    rng = np.random.default_rng(seed)
    out = []
    for arr in rec_arrays:
        thinned = np.zeros_like(arr)
        cyc, nl = arr.shape
        for c in range(cyc):
            for L in range(nl):
                n = int(arr[c, L])
                if n:
                    thinned[c, L] = int((rng.random(n) >= frac).sum())
        out.append(thinned)
    return out


# ---------------------------------------------------------------------------
# Discrete-event schedule: one host thread, one FIFO GPU queue, one FIFO SSD.
# ---------------------------------------------------------------------------
def simulate(groups, cyc, *, g_dur_ms, c_assign_ms, rate_gbps, rec_bytes,
             host_pre_ms, host_post_ms, extra_per_cycle_ms, growth_s=GROWTH_S,
             collect_periods=True):
    """Replay the fixed host program (A leads B by one layer) over the resources.

    groups: list of dicts with rec[cyc,40] (already thinned if any), ma[cyc,40],
    nassign (18 or 36).  E_hit = (nassign-ma)*c_assign on the GPU after submit_reads
    (runs while that group's records read); E_miss = ma*c_assign on the GPU after
    finish.  A group's next G depends (via FIFO submission order) on that group's
    E_hit and E_miss completing.  barrier blocks the host on the group's G; finish
    blocks it on the group's read batch.  One-group + G_full + extra=0 == control.
    """
    rd = rec_bytes / (rate_gbps * 1e9)          # seconds per record
    g_dur = g_dur_ms / 1e3
    ca = c_assign_ms / 1e3
    host_pre = host_pre_ms / 1e3
    host_post = host_post_ms / 1e3
    extra = extra_per_cycle_ms / 1e3
    G = len(groups)
    gG = [0.0] * G      # completion time of each group's most recent G stage
    gR = [0.0] * G      # completion time of each group's most recent read batch
    host_t = growth_s
    st = {"gpu_free": 0.0, "ssd_free": 0.0}
    acc = {"gpu_busy": 0.0, "ssd_busy": 0.0, "blk_barrier": 0.0,
           "blk_finish": 0.0, "host_busy": 0.0, "records": 0}
    periods = []
    periods_balanced = []
    periods_imbalanced = []

    def gpu(submit_t, dur):
        if dur <= 0:
            return max(st["gpu_free"], submit_t)
        start = max(st["gpu_free"], submit_t)
        st["gpu_free"] = start + dur
        acc["gpu_busy"] += dur
        return st["gpu_free"]

    def ssd(submit_t, n):
        acc["records"] += n
        if n <= 0:
            return max(st["ssd_free"], submit_t)
        start = max(st["ssd_free"], submit_t)
        st["ssd_free"] = start + n * rd
        acc["ssd_busy"] += n * rd
        return st["ssd_free"]

    for c in range(cyc):
        host_t += T_DRAFT
        # ---- open layer 0 for each group in order ----
        for gi in range(G):
            g = groups[gi]
            ma0 = int(g["ma"][c, 0])
            gG[gi] = gpu(host_t, g_dur)                       # submit G(gi,0)
            w = gG[gi] - host_t                               # barrier(gi,0)
            if w > 0:
                acc["blk_barrier"] += w
                host_t = gG[gi]
            host_t += host_pre
            acc["host_busy"] += host_pre
            gR[gi] = ssd(host_t, int(g["rec"][c, 0]))         # submit_reads(gi,0)
            gpu(host_t, (g["nassign"] - ma0) * ca)            # E_hit(gi,0) on GPU
        marker = host_t
        # ---- layers 1..39 ----
        for L in range(1, 40):
            imbalanced = False
            for gi in range(G):
                g = groups[gi]
                maL = int(g["ma"][c, L])
                maPrev = int(g["ma"][c, L - 1])
                w = gR[gi] - host_t                           # finish(gi,L-1)
                if w > 0:
                    acc["blk_finish"] += w
                    host_t = gR[gi]
                host_t += host_post                           # graph build
                acc["host_busy"] += host_post
                gpu(host_t, maPrev * ca)                      # E_miss(gi,L-1)
                gG[gi] = gpu(host_t, g_dur)                   # G(gi,L) right after
                w = gG[gi] - host_t                           # barrier(gi,L)
                if w > 0:
                    acc["blk_barrier"] += w
                    host_t = gG[gi]
                host_t += host_pre
                acc["host_busy"] += host_pre
                gR[gi] = ssd(host_t, int(g["rec"][c, L]))     # submit_reads(gi,L)
                gpu(host_t, (g["nassign"] - maL) * ca)        # E_hit(gi,L)
                if int(g["rec"][c, L]) == 0:
                    imbalanced = True
            if collect_periods:
                p = host_t - marker
                periods.append(p)
                (periods_imbalanced if imbalanced else periods_balanced).append(p)
                marker = host_t
        # ---- flush E_miss(39), then the head/accept/commit barrier ----
        for gi in range(G):
            g = groups[gi]
            w = gR[gi] - host_t
            if w > 0:
                acc["blk_finish"] += w
                host_t = gR[gi]
            host_t += host_post
            acc["host_busy"] += host_post
            gpu(host_t, int(g["ma"][c, 39]) * ca)
        host_t = max(host_t, st["gpu_free"])   # head waits for the last MoE output
        host_t += T_ACCEPT + T_COMMIT + extra
    total = host_t

    def pct(a, q):
        return float(np.percentile(a, q)) if a else 0.0

    return {
        "total_decode_s": total,
        "tps_1024": 1024.0 / total,
        "read_records": acc["records"],
        "ssd_busy_s": acc["ssd_busy"],
        "gpu_busy_s": acc["gpu_busy"],
        "ssd_busy_frac": acc["ssd_busy"] / total,
        "gpu_busy_frac": acc["gpu_busy"] / total,
        "gpu_bound": acc["gpu_busy"] > acc["ssd_busy"],
        "host_blocked_finish_s": acc["blk_finish"],
        "host_blocked_barrier_s": acc["blk_barrier"],
        "host_busy_s": acc["host_busy"],
        "period_ms_p10": pct(periods, 10) * 1e3,
        "period_ms_p50": pct(periods, 50) * 1e3,
        "period_ms_p90": pct(periods, 90) * 1e3,
        "period_ms_mean_balanced": (float(np.mean(periods_balanced)) * 1e3
                                    if periods_balanced else 0.0),
        "period_ms_mean_imbalanced": (float(np.mean(periods_imbalanced)) * 1e3
                                      if periods_imbalanced else 0.0),
        "n_periods_balanced": len(periods_balanced),
        "n_periods_imbalanced": len(periods_imbalanced),
    }


def calibrate_host_post(single_replay, c_assign_ms=C_ASSIGN_CENTRAL):
    """host_post s.t. mean per-layer non-read critical path == the measured C_LAYER
    at the central c_assign (control reproduces the f1 74.667 s anchor).  Uses the
    single-route mean miss-assignment; host_pre and G_full stay at the task givens.
    """
    mean_ma = single_replay["mean_miss_assign_per_layercall"]
    return C_LAYER_MS - G_FULL_MS - HOST_PRE_MS - mean_ma * c_assign_ms


def control_and_pipeline(single, two, *, g_half_ms, c_assign_ms, rate, rec_bytes,
                         host_post_ms, thin_frac=0.0):
    """Run the control (1 group, G_full) and the two-group pipeline for one cell."""
    cyc = single["cycles"]
    s_rec = thin_records(single["rec"], thin_frac, THIN_SEED)
    t_rec = thin_records(two["rec"], thin_frac, THIN_SEED)
    ctrl_groups = [{"rec": s_rec[0], "ma": single["ma"][0],
                    "nassign": single["nassign_per_group"]}]
    pipe_groups = [{"rec": t_rec[g], "ma": two["ma"][g],
                    "nassign": two["nassign_per_group"]} for g in range(2)]
    ctrl = simulate(ctrl_groups, cyc, g_dur_ms=G_FULL_MS, c_assign_ms=c_assign_ms,
                    rate_gbps=rate, rec_bytes=rec_bytes, host_pre_ms=HOST_PRE_MS,
                    host_post_ms=host_post_ms, extra_per_cycle_ms=0.0)
    pipe = simulate(pipe_groups, cyc, g_dur_ms=g_half_ms, c_assign_ms=c_assign_ms,
                    rate_gbps=rate, rec_bytes=rec_bytes, host_pre_ms=HOST_PRE_MS,
                    host_post_ms=host_post_ms, extra_per_cycle_ms=PIPE_EXTRA_MS)
    pipe["seconds_removed_vs_control"] = ctrl["total_decode_s"] - pipe["total_decode_s"]
    return ctrl, pipe


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(
        "docs/deepseek-v41/receipts/f10-rowgroup-pipeline-sim-20260919/results.json"))
    ap.add_argument("--anchors-only", action="store_true")
    args = ap.parse_args()

    if any(m == "mlx" or m.startswith("mlx.") for m in sys.modules):
        raise SystemExit("MLX leaked into the process")

    import runpy
    restore = runpy.run_path(str(osim.HELPER))["restore"]
    trace = osim.load_trace()

    # Exact cache validation: reuse f1's two published anchors, then our own.
    anchors = osim.anchor_checks(trace, restore)
    for name, a in anchors.items():
        if not a["exact"]:
            raise SystemExit(f"anchor {name} failed: {a}")
    if args.anchors_only:
        print(json.dumps(anchors, indent=2))
        return

    t0 = time.perf_counter()
    # Replays (single + two-group) at each capacity.
    single = {cap: replay(trace, restore, cap, n_groups=1) for cap in CAP_SWEEP}
    two = {cap: replay(trace, restore, cap, n_groups=2) for cap in CAP_SWEEP}
    if single[111]["total_records"] != 31636:
        raise SystemExit(f"control records {single[111]['total_records']} != 31636")

    host_post = calibrate_host_post(single[111])

    # Control-line reconciliation at the central cell for each c_assign.
    control_lines = {}
    for ca in C_ASSIGN_SWEEP:
        ctrl, _ = control_and_pipeline(single[111], two[111], g_half_ms=1.5,
                                       c_assign_ms=ca, rate=DEFAULT_RATE,
                                       rec_bytes=REC_BYTES, host_post_ms=host_post)
        control_lines[ca] = ctrl

    # Main grid: G_half x rate x capacity at the central c_assign.
    grid = []
    for cap in CAP_SWEEP:
        for rate in RATE_SWEEP:
            for gh in G_HALF_SWEEP:
                ctrl, pipe = control_and_pipeline(
                    single[cap], two[cap], g_half_ms=gh, c_assign_ms=C_ASSIGN_CENTRAL,
                    rate=rate, rec_bytes=REC_BYTES, host_post_ms=host_post)
                grid.append({"persistent": cap, "rate_gbps": rate, "g_half_ms": gh,
                             "c_assign_ms": C_ASSIGN_CENTRAL,
                             "control": ctrl, "pipeline": pipe})

    # c_assign sensitivity at the primary cell (G_half 1.5, 12.9, cap 111).
    c_sens = []
    for ca in C_ASSIGN_SWEEP:
        ctrl, pipe = control_and_pipeline(single[111], two[111], g_half_ms=1.5,
                                          c_assign_ms=ca, rate=DEFAULT_RATE,
                                          rec_bytes=REC_BYTES, host_post_ms=host_post)
        c_sens.append({"c_assign_ms": ca, "control": ctrl, "pipeline": pipe})

    # Byte-lever thinning at cap 111 / 12.9 / c 0.035, for two G_half values, to
    # show the SSD->GPU crossover as records are removed.
    thin = []
    for gh in (1.5, 1.8):
        for f in THIN_SWEEP:
            ctrl, pipe = control_and_pipeline(single[111], two[111], g_half_ms=gh,
                                              c_assign_ms=C_ASSIGN_CENTRAL,
                                              rate=DEFAULT_RATE, rec_bytes=REC_BYTES,
                                              host_post_ms=host_post, thin_frac=f)
            thin.append({"g_half_ms": gh, "thin_frac": f,
                         "control": ctrl, "pipeline": pipe})

    result = {
        "purpose": "F10 row-group pipeline DES screen; no throughput claim, no "
                   "production code, no GPU/service touched. Control reproduces the "
                   "f1 74.667 s anchor; the two-group split is replayed through the "
                   "real transition-window cache with two route observations/layer.",
        "source_commit": osim._git_head(),
        "elapsed_s": time.perf_counter() - t0,
        "mlx_imported": any(m == "mlx" or m.startswith("mlx.") for m in sys.modules),
        "inputs": {
            "trace": str(osim.TRACE), "trace_sha256": osim.TRACE_SHA256,
            "replay_helper": str(osim.HELPER),
            "replay_helper_sha256": sha256(osim.HELPER),
            "expert_streaming_sha256": sha256("mtplx/expert_streaming.py"),
            "sibling_sim_sha256": sha256(
                "scripts/deepseek_v41/overlap_schedule_sim.py"),
        },
        "timing_model": {
            "record_bytes": REC_BYTES, "c_layer_ms": C_LAYER_MS,
            "g_full_ms": G_FULL_MS, "host_pre_ms": HOST_PRE_MS,
            "host_post_ms_calibrated": host_post,
            "growth_s": GROWTH_S, "t_draft_ms": T_DRAFT * 1e3,
            "t_accept_ms": T_ACCEPT * 1e3, "t_commit_ms": T_COMMIT * 1e3,
            "pipe_extra_per_cycle_ms": PIPE_EXTRA_MS,
            "c_assign_sweep_ms": C_ASSIGN_SWEEP, "c_assign_central_ms": C_ASSIGN_CENTRAL,
            "note": "Per-layer non-read compute = G_full + host_pre + host_post + "
                    "miss_assign*c_assign; hit_assign*c_assign runs on the GPU during "
                    "reads (hidden except on zero-miss layers). host_post calibrated "
                    "once at c_assign=0.035 so mean non-read compute == C_LAYER. Head "
                    "is inside VERIFY_S->C_LAYER (no separate head term). 206-cycle "
                    "trace replayed with per-198-cycle rates, exactly as the f1 sim; "
                    "TPS=1024/total.",
        },
        "anchors": anchors,
        "replay_stats": {
            cap: {
                "single_records": single[cap]["total_records"],
                "two_group_records": two[cap]["total_records"],
                "two_group_delta_frac": (two[cap]["total_records"]
                                         - single[cap]["total_records"])
                / single[cap]["total_records"],
                "single_zero_miss_frac": single[cap]["zero_miss_group_fraction"][0],
                "two_zero_miss_frac_A": two[cap]["zero_miss_group_fraction"][0],
                "two_zero_miss_frac_B": two[cap]["zero_miss_group_fraction"][1],
                "two_either_zero_frac": two[cap]["either_zero_fraction"],
                "single_mean_miss_assign": single[cap]["mean_miss_assign_per_layercall"],
            } for cap in CAP_SWEEP
        },
        "control_lines_by_c_assign": control_lines,
        "grid": grid,
        "c_assign_sensitivity_primary_cell": c_sens,
        "byte_lever_thinning_primary_cell": thin,
        "variants_skipped": {
            "half_step_B_lags": "B.barrier(L) before A.finish(L) opens B's route on "
                                "L before A's route on L closes -> >1 open decode "
                                "route/layer; the runtime forbids it. Not simulated.",
            "three_group": "Not a trivial generalisation: 3 groups touching layer L "
                           "per iteration need >1 open route (same runtime limit) and "
                           "an unmeasured G_third launch-bound cost. Skipped.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float) + "\n")
    _print(result)


def _print(r):
    tm = r["timing_model"]
    print(f"host_post calibrated = {tm['host_post_ms_calibrated']:.4f} ms "
          f"(G_full {tm['g_full_ms']}, host_pre {tm['host_pre_ms']}, "
          f"C_LAYER {tm['c_layer_ms']:.4f})")
    print("\nCONTROL LINE (cap 111, 12.9 GB/s) vs f1 anchor 74.667 s / 43.395 s:")
    for ca, c in r["control_lines_by_c_assign"].items():
        print(f"  c_assign={ca}: total {c['total_decode_s']:.3f} s  "
              f"TPS {c['tps_1024']:.2f}  SSD {c['ssd_busy_frac']*100:.1f}%  "
              f"GPU {c['gpu_busy_frac']*100:.1f}%  blk_finish {c['host_blocked_finish_s']:.2f} s  "
              f"blk_barrier {c['host_blocked_barrier_s']:.2f} s  "
              f"records {c['read_records']}  period p50 {c['period_ms_p50']:.2f} ms")
    rs = r["replay_stats"]
    print("\nRECORD/IMBALANCE (transition-window):")
    for cap, s in rs.items():
        print(f"  cap {cap}: single {s['single_records']}  two-group "
              f"{s['two_group_records']} ({s['two_group_delta_frac']*100:+.2f}%)  "
              f"zeroMiss A {s['two_zero_miss_frac_A']*100:.1f}% B "
              f"{s['two_zero_miss_frac_B']*100:.1f}% either {s['two_either_zero_frac']*100:.1f}% "
              f"(single {s['single_zero_miss_frac']*100:.1f}%)")
    print("\nGRID (c_assign=0.035): persist rate g_half | ctrl_s pipe_s removed TPS | "
          "SSD% GPU% bound | blkFin blkBar | p10/p50/p90 ms")
    for cell in r["grid"]:
        p, c = cell["pipeline"], cell["control"]
        print(f"  {cell['persistent']} {cell['rate_gbps']:>4} {cell['g_half_ms']:>4} | "
              f"{c['total_decode_s']:.2f} {p['total_decode_s']:.2f} "
              f"{p['seconds_removed_vs_control']:+.2f} {p['tps_1024']:.2f} | "
              f"{p['ssd_busy_frac']*100:.0f} {p['gpu_busy_frac']*100:.0f} "
              f"{'GPU' if p['gpu_bound'] else 'SSD'} | "
              f"{p['host_blocked_finish_s']:.1f} {p['host_blocked_barrier_s']:.1f} | "
              f"{p['period_ms_p10']:.1f}/{p['period_ms_p50']:.1f}/{p['period_ms_p90']:.1f}")
    print("\nc_assign sensitivity (primary cell G_half 1.5, 12.9, cap 111):")
    for row in r["c_assign_sensitivity_primary_cell"]:
        p, c = row["pipeline"], row["control"]
        print(f"  c={row['c_assign_ms']}: ctrl {c['total_decode_s']:.2f}s  pipe "
              f"{p['total_decode_s']:.2f}s removed {p['seconds_removed_vs_control']:+.2f}  "
              f"GPU {p['gpu_busy_frac']*100:.0f}% SSD {p['ssd_busy_frac']*100:.0f}% "
              f"{'GPU-bound' if p['gpu_bound'] else 'SSD-bound'}")
    print("\nByte-lever thinning (cap 111 / 12.9 / c 0.035):")
    for row in r["byte_lever_thinning_primary_cell"]:
        p, c = row["pipeline"], row["control"]
        print(f"  G_half {row['g_half_ms']} -{int(row['thin_frac']*100):>2}% recs: "
              f"ctrl {c['total_decode_s']:.2f}s pipe {p['total_decode_s']:.2f}s "
              f"removed {p['seconds_removed_vs_control']:+.2f}  records {p['read_records']}  "
              f"GPU {p['gpu_busy_frac']*100:.0f}% SSD {p['ssd_busy_frac']*100:.0f}% "
              f"{'GPU-bound' if p['gpu_bound'] else 'SSD-bound'}")


if __name__ == "__main__":
    main()
