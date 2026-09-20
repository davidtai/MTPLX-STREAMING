#!/usr/bin/env python3
"""F14: non-uniform per-layer expert-cache CAPACITY allocation screen (DSV4.1 M6 decode).

CPU ONLY (an ``_NoMLX`` meta-path finder raises on any ``mlx`` import), numpy only,
run under ``nice -n 19``.  This is a retrospective cache-*sizing* screen, NOT a
throughput result and NOT a production-code proposal.

Today every routed layer gets the SAME persistent capacity (108-111 rows) + 48
shared transient slots.  84 rows exist at the prefill->decode boundary; the rest
are "extension rows" allocated per layer AFTER prefill, so per-layer capacity is a
free construction-time choice as long as each layer keeps >= 84 rows and the TOTAL
row count is unchanged.  A slot bank is a PURE cache -- it changes only which
records are read, never a routed expert's output -- so ANY allocation is bit-exact.
Question: how many expert-record reads does a non-uniform per-layer capacity remove
at the same total memory?

The replay REUSES the exact machinery that reproduces the f1-overlap control
(206-cycle trace @ 111+48 -> 31,636 records): ``overlap_schedule_sim.make_bank``
(restore captured 73-slot snapshot, grow to c, protected_cap=int(c*0.8)) + the real
``LayerExpertSlotBank.plan(phase='decode')``.  Each layer is an INDEPENDENT bank, so
total = sum_L misses_L(c_L) is separable and the anchor is a per-layer superposition.

Phases (run in short batches, gate-checked between):
  curves  : misses_L(c) per-cycle for a layer subset over a capacity grid -> .npz
  analyze : load curves, reproduce anchors, optimize (exact DP), causality
            (oracle / split-half / prefill-rule), sensitivity; re-simulate the
            chosen vectors exactly; write results.json.
"""
from __future__ import annotations

import argparse
import importlib.abc
import json
import os
import runpy
import sys
import time
from math import comb, log
from pathlib import Path


class _NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise RuntimeError("per_layer_capacity_sim is CPU-only; MLX is forbidden")


if not any(isinstance(f, _NoMLX) for f in sys.meta_path):
    sys.meta_path.insert(0, _NoMLX())

WT = Path(__file__).resolve().parents[2]          # worktree root
os.chdir(WT)                                        # osim resolves TRACE/HELPER vs cwd
sys.path.insert(0, str(WT))
sys.path.insert(0, str(WT / "scripts" / "deepseek_v41"))
import numpy as np  # noqa: E402
import overlap_schedule_sim as osim  # noqa: E402  (load_trace, make_bank, anchor_checks)

POLICY = "transition-window"
TRANSIENT = 48
CMIN, CMAX = 84, 192
GRID16 = (84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 128, 136, 144, 160, 176, 192)
DENSE = tuple(range(CMIN, CMAX + 1))
C_UNIFORM = (108, 111)
REC_MS = 1.36                                       # task headline: ms per record read
REC_MS_F7 = osim.REC_BYTES / (osim.DEFAULT_RATE * 1e9) * 1e3  # 1.372, cross-ref
RECEIPT = WT / "docs/deepseek-v41/receipts/f14-per-layer-capacity-20260919"


# --------------------------------------------------------------------------- #
def load():
    restore = runpy.run_path(str(osim.HELPER))["restore"]
    trace = osim.load_trace()
    layers = sorted(int(l) for l in trace["target_routes_by_layer"])
    routes = {L: trace["target_routes_by_layer"][str(L)] for L in layers}
    snap = {L: trace["initial_banks"][str(L)] for L in layers}
    return restore, trace, layers, routes, snap, trace["cycles"]


def replay_layer(restore, snap_L, routes_L, c, cyc, protected_cap=None):
    """Per-cycle demand-miss counts for one layer at persistent capacity c."""
    bank = osim.make_bank(restore, snap_L, c, TRANSIENT, POLICY)
    if protected_cap is not None:                   # sensitivity (ii): pin protected cap
        bank._protected_cap = max(1, int(protected_cap))
    out = np.empty(cyc, np.int32)
    for i in range(cyc):
        out[i] = len(bank.plan(routes_L[i], phase="decode").misses)
    return out


# --------------------------------------------------------------------------- #
def phase_curves(layer_lo, layer_hi, caps, out_npz):
    restore, _trace, layers, routes, snap, cyc = load()
    sub = [L for L in layers if layer_lo <= L <= layer_hi]
    caps = list(caps)
    misses = np.zeros((len(sub), len(caps), cyc), np.int16)
    for li, L in enumerate(sub):
        t0 = time.perf_counter()
        for ci, c in enumerate(caps):
            misses[li, ci] = replay_layer(restore, snap[L], routes[L], c, cyc)
        print(f"  L{L:2d}: {len(caps)} caps x {cyc} cyc in "
              f"{time.perf_counter()-t0:.2f}s (c={caps[0]}->{misses[li,0].sum()}, "
              f"c={caps[-1]}->{misses[li,-1].sum()})", flush=True)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, layers=np.array(sub), caps=np.array(caps),
                        misses=misses, cycles=cyc)
    print(f"wrote {out_npz}: layers {sub[0]}..{sub[-1]}, {misses.sum()} total misses")


# --------------------------------------------------------------------------- #
def _load_curves(npzs):
    """Merge curve shards -> tot[L][x], fh[L][x], sh[L][x] on dense integer grid."""
    caps_ref = None
    cyc = None
    tot, fh, sh, percyc = {}, {}, {}, {}
    for p in npzs:
        d = np.load(p)
        caps = tuple(int(x) for x in d["caps"])
        if caps_ref is None:
            caps_ref = caps
        elif caps != caps_ref:
            raise SystemExit(f"cap grid mismatch in {p}")
        cyc = int(d["cycles"])
        for li, L in enumerate(int(x) for x in d["layers"]):
            m = d["misses"][li].astype(np.int64)           # (ncap, cyc)
            tot[L] = m.sum(1)
            fh[L] = m[:, :100].sum(1)                       # cycles 0..99
            sh[L] = m[:, 100:].sum(1)                       # cycles 100..205
            percyc[L] = m
    return caps_ref, cyc, tot, fh, sh, percyc


def _c_index(caps, c):
    return caps.index(c)


def optimize_dp(cost, layers, caps, budget_rows):
    """Exact separable bounded allocation: min sum cost[L][c] s.t. sum c=budget,
    CMIN<=c<=CMAX.  cost[L] indexed like caps (must be the DENSE integer grid)."""
    if caps != DENSE:
        raise SystemExit("optimize_dp requires the dense integer grid")
    n = len(layers)
    xmax = CMAX - CMIN
    B = budget_rows - n * CMIN
    INF = np.inf
    dp = np.full(B + 1, INF)
    dp[0] = 0.0
    parent = np.full((n, B + 1), -1, np.int32)
    for i, L in enumerate(layers):
        f = cost[L].astype(float)                          # length xmax+1
        ndp = np.full(B + 1, INF)
        for x in range(0, min(xmax, B) + 1):
            cand = dp[: B + 1 - x] + f[x]
            seg = ndp[x: B + 1]
            better = cand < seg
            seg[better] = cand[better]
            parent[i, np.nonzero(better)[0] + x] = x
        dp = ndp
    b = B
    xs = [0] * n
    for i in range(n - 1, -1, -1):
        xs[i] = int(parent[i, b])
        b -= xs[i]
    alloc = {L: CMIN + xs[i] for i, L in enumerate(layers)}
    return alloc, float(dp[B])


def alloc_total(cost, alloc, caps):
    return int(sum(cost[L][_c_index(caps, alloc[L])] for L in alloc))


def proportional_alloc(weights, layers, budget_rows):
    """Integer extension rows ~ weights, sum(c)=budget, c in [CMIN,CMAX]."""
    w = np.array([max(0.0, float(weights[L])) for L in layers], float)
    E = budget_rows - len(layers) * CMIN
    xmax = CMAX - CMIN
    if w.sum() <= 0:
        w = np.ones_like(w)
    raw = E * w / w.sum()
    x = np.floor(raw).astype(int)
    x = np.clip(x, 0, xmax)
    resid = E - int(x.sum())
    frac = raw - np.floor(raw)
    order = np.argsort(-frac)                               # give leftover to top frac
    j = 0
    while resid > 0 and j < 10 * len(x):
        k = order[j % len(order)]
        if x[k] < xmax:
            x[k] += 1
            resid -= 1
        j += 1
    while resid < 0:                                        # rare: took too many
        k = int(np.argmax(x))
        x[k] -= 1
        resid += 1
    return {L: CMIN + int(x[i]) for i, L in enumerate(layers)}


def round_to_mult(alloc, layers, budget_rows, m=4):
    """Round each c to a multiple of m, then repair sum to budget within [CMIN,CMAX]."""
    x = {L: int(round((alloc[L] - CMIN) / m) * m) for L in layers}
    x = {L: min(CMAX - CMIN, max(0, x[L])) for L in layers}
    E = budget_rows - len(layers) * CMIN
    # repair in steps of m
    while sum(x.values()) != E:
        diff = E - sum(x.values())
        step = m if diff > 0 else -m
        # move the layer with best marginal room
        cand = [L for L in layers if 0 <= x[L] + step <= CMAX - CMIN]
        if not cand:
            break
        L = cand[0]
        x[L] += step
    return {L: CMIN + x[L] for L in layers}


def prefill_stats(snap, layers):
    """Per-layer prefill-route-frequency statistics (causal, boundary-available)."""
    out = {}
    for L in layers:
        pf = {int(k): float(v) for k, v in (snap[L].get("_prefill_route_freq") or {}).items()}
        tot = sum(pf.values()) or 1.0
        p = np.array([v / tot for v in pf.values()], float)
        p = p[p > 0]
        H = float(-(p * np.log(p)).sum())                  # entropy (nats)
        eff = float(np.exp(H))                             # effective # experts
        vals = np.sort(np.array(list(pf.values()), float))[::-1]
        top84 = float(vals[:84].sum() / tot)               # mass in the base rows
        out[L] = {"distinct": len(pf), "entropy": H, "eff_number": eff,
                  "one_minus_top84_mass": 1.0 - top84}
    return out


def sign_test(deltas):
    imp = int((deltas < 0).sum())
    wor = int((deltas > 0).sum())
    n = imp + wor
    if n == 0:
        return imp, wor, 1.0
    k = min(imp, wor)
    p = min(1.0, 2.0 * sum(comb(n, j) for j in range(k + 1)) / (2.0 ** n))
    return imp, wor, p


def secs(records):
    return records * REC_MS / 1e3


# --------------------------------------------------------------------------- #
def phase_analyze(npzs, out_json):
    restore, _trace, layers, routes, snap, cyc = load()
    caps, cyc2, tot, fh, sh, _pc = _load_curves(npzs)
    assert cyc == cyc2
    got = sorted(tot)
    if got != layers:
        raise SystemExit(f"curves cover {got[0]}..{got[-1]} ({len(got)}), trace has "
                         f"{layers[0]}..{layers[-1]} ({len(layers)})")

    res = {"purpose": "F14 per-layer expert-cache capacity allocation screen; CPU-only, "
                      "numpy-only, no GPU/service/production-code. A slot bank is a pure "
                      "cache: any allocation is bit-exact.",
           "source_commit": osim._git_head(), "policy": POLICY, "transient": TRANSIENT,
           "layers": layers, "cycles": cyc, "cap_grid_dense": [CMIN, CMAX],
           "cap_grid_reported": list(GRID16), "record_ms": REC_MS,
           "record_ms_f7": REC_MS_F7, "mlx_imported": any(
               m == "mlx" or m.startswith("mlx.") for m in sys.modules)}

    # ---- anchors -----------------------------------------------------------
    anc = osim.anchor_checks(_trace, restore)
    u = {C: sum(int(tot[L][_c_index(caps, C)]) for L in layers) for C in C_UNIFORM}
    # independent re-simulation of the uniform 111 baseline (must equal 31,636)
    resim111 = sum(int(replay_layer(restore, snap[L], routes[L], 111, cyc).sum())
                   for L in layers)
    res["anchors"] = {**anc, "uniform_totals": u, "resim_uniform_111": resim111,
                      "expected_111": 31636,
                      "anchor_ok": u[111] == 31636 == resim111}
    print(f"[anchor] cap102={anc['prefix_readiness_cap102']['exact']} "
          f"cap73={anc['mtp_verify_cap73']['exact']} uniform111={u[111]} "
          f"resim111={resim111} (want 31636)")

    # ---- miss-vs-capacity table (reported 16-pt grid) + convexity ----------
    table = {C: [int(tot[L][_c_index(caps, C)]) for L in layers] for C in GRID16}
    d2min = {}
    convex_layers = 0
    for L in layers:
        f = tot[L].astype(float)
        d2 = f[2:] + f[:-2] - 2 * f[1:-1]                  # dense 2nd diff
        d2min[L] = float(d2.min())
        if d2.min() >= -1e-9:
            convex_layers += 1
    res["miss_vs_capacity"] = {"grid": list(GRID16), "per_layer_totals": table}
    res["convexity"] = {"layers_convex_dense": convex_layers, "n_layers": len(layers),
                        "min_2nd_diff_over_layers": float(min(d2min.values())),
                        "note": "2nd difference of total misses over dense integer c; "
                                ">=0 == convex (diminishing returns)."}

    # ---- optimization: oracle (full trace) --------------------------------
    alloc_opt, sens_greedy = {}, {}
    opt_block = {}
    for C in C_UNIFORM:
        budget = len(layers) * C
        a, val = optimize_dp(tot, layers, caps, budget)
        alloc_opt[C] = a
        removed = u[C] - int(round(val))
        # independent exact re-simulation of the chosen vector
        resim = sum(int(replay_layer(restore, snap[L], routes[L], a[L], cyc).sum())
                    for L in layers)
        opt_block[C] = {"uniform_reads": u[C], "opt_reads": int(round(val)),
                        "opt_reads_resim": resim, "records_removed": removed,
                        "pct": 100 * removed / u[C], "seconds_removed": secs(removed),
                        "resim_matches": resim == int(round(val)),
                        "alloc": {int(L): int(a[L]) for L in layers}}
        print(f"[oracle C={C}] uniform={u[C]} opt={int(round(val))} resim={resim} "
              f"removed={removed} ({100*removed/u[C]:.2f}%, {secs(removed):.2f}s @1.36ms)")
    res["optimize_oracle"] = opt_block

    # ---- causality: split-half (fit one half, evaluate the other) ---------
    split = {}
    for C in C_UNIFORM:
        budget = len(layers) * C
        u_fh = sum(int(fh[L][_c_index(caps, C)]) for L in layers)
        u_sh = sum(int(sh[L][_c_index(caps, C)]) for L in layers)
        a_fit_fh, _ = optimize_dp(fh, layers, caps, budget)   # fit cyc 0-99
        a_fit_sh, _ = optimize_dp(sh, layers, caps, budget)   # fit cyc 100-205
        eval_sh = sum(int(sh[L][_c_index(caps, a_fit_fh[L])]) for L in layers)
        eval_fh = sum(int(fh[L][_c_index(caps, a_fit_sh[L])]) for L in layers)
        split[C] = {
            "fit_first_eval_second": {"uniform": u_sh, "alloc_reads": eval_sh,
                                      "removed": u_sh - eval_sh,
                                      "seconds_removed": secs(u_sh - eval_sh)},
            "fit_second_eval_first": {"uniform": u_fh, "alloc_reads": eval_fh,
                                      "removed": u_fh - eval_fh,
                                      "seconds_removed": secs(u_fh - eval_fh)}}
        print(f"[split C={C}] fit0-99/eval100-205 removed={u_sh-eval_sh}; "
              f"fit100-205/eval0-99 removed={u_fh-eval_fh}")
    res["causality_split_half"] = split

    # ---- causality: prefill-only rule -------------------------------------
    pstats = prefill_stats(snap, layers)
    stat_names = ("eff_number", "entropy", "one_minus_top84_mass", "distinct")
    # correlate each stat with the oracle extension (C=108) and baseline per-layer miss
    ext108 = np.array([alloc_opt[108][L] - CMIN for L in layers], float)
    base_miss = np.array([tot[L][_c_index(caps, 108)] for L in layers], float)
    corr = {}
    for s in stat_names:
        v = np.array([pstats[L][s] for L in layers], float)
        corr[s] = {"vs_oracle_ext": float(np.corrcoef(v, ext108)[0, 1]),
                   "vs_baseline_miss": float(np.corrcoef(v, base_miss)[0, 1])}
    best_stat = max(stat_names, key=lambda s: corr[s]["vs_baseline_miss"])
    prefill = {"stats_per_layer": {int(L): pstats[L] for L in layers},
               "stat_correlations": corr, "selected_stat": best_stat,
               "selection_note": "stat FORM selected by correlation with the per-layer "
                                  "baseline miss level (a peek); the rule itself uses only "
                                  "boundary-available _prefill_route_freq and is evaluated "
                                  "on the full decode trace.", "by_C": {}}
    for C in C_UNIFORM:
        budget = len(layers) * C
        w = {L: pstats[L][best_stat] for L in layers}
        a_rule = proportional_alloc(w, layers, budget)
        rule_reads = alloc_total(tot, a_rule, caps)
        resim = sum(int(replay_layer(restore, snap[L], routes[L], a_rule[L], cyc).sum())
                    for L in layers)
        removed = u[C] - rule_reads
        orc = res["optimize_oracle"][C]["records_removed"]
        prefill["by_C"][C] = {"uniform": u[C], "rule_reads": rule_reads,
                              "rule_reads_resim": resim, "removed": removed,
                              "pct": 100 * removed / u[C], "seconds_removed": secs(removed),
                              "frac_of_oracle": (removed / orc) if orc else 0.0,
                              "alloc": {int(L): int(a_rule[L]) for L in layers}}
        print(f"[prefill C={C}] stat={best_stat} removed={removed} "
              f"({100*removed/u[C]:.2f}%, {removed/orc*100 if orc else 0:.0f}% of oracle)")
    res["causality_prefill_rule"] = prefill

    # ---- cross-prompt transfer --------------------------------------------
    # Every other *.json.gz under receipts/ was inspected: the lookahead-adjacent
    # `routes.json.gz` are predictor-experiment variants of the SAME 64-cycle
    # router-feature capture (shared trace/capture sha; key `routes`, not
    # `target_routes_by_layer`); the rest are timing/one-generation/policy-arm
    # artifacts. None is an independent SECOND-PROMPT full decode route trace with
    # per-layer routes + initial_banks. So cross-prompt transfer cannot be tested.
    res["cross_prompt"] = {
        "second_prompt_trace_available": False,
        "note": "Only one full decode route trace exists in the receipts "
                "(mtp-verify-routes 16k/1024, the anchor trace). No independent "
                "second-prompt decode trace is present, so cross-prompt transfer of "
                "the allocation is UNTESTED (stated plainly, per the task)."}

    # ---- sensitivity -------------------------------------------------------
    sens = {}
    # (i) +-8 total rows: re-optimize at budget +-8, gain still positive?
    tot_rows = {}
    for C in C_UNIFORM:
        for d in (-8, +8):
            budget = len(layers) * C + d
            a, val = optimize_dp(tot, layers, caps, budget)
            # uniform reference at the SAME total: spread +-8 over the first |d| layers
            ru = _uniform_plus(tot, layers, caps, C, d)
            tot_rows[f"C{C}{d:+d}"] = {"budget": budget, "opt_reads": int(round(val)),
                                       "uniform_like_reads": ru,
                                       "removed_vs_uniform_like": ru - int(round(val)),
                                       "seconds_removed": secs(ru - int(round(val)))}
    sens["plus_minus_8_rows"] = tot_rows
    # (ii) protected_cap fixed at int(C*0.8) instead of int(c_L*0.8): re-simulate
    prot = {}
    for C in C_UNIFORM:
        a = alloc_opt[C]
        pc = int(C * 0.8)
        # uniform baseline with fixed protected cap
        u_fix = sum(int(replay_layer(restore, snap[L], routes[L], C, cyc,
                                     protected_cap=pc).sum()) for L in layers)
        a_fix = sum(int(replay_layer(restore, snap[L], routes[L], a[L], cyc,
                                     protected_cap=pc).sum()) for L in layers)
        prot[C] = {"protected_cap_fixed": pc, "uniform_fixedprot": u_fix,
                   "alloc_fixedprot": a_fix, "removed_fixedprot": u_fix - a_fix,
                   "seconds_removed": secs(u_fix - a_fix),
                   "removed_scaledprot": res["optimize_oracle"][C]["records_removed"]}
        print(f"[sens prot C={C}] fixed-protcap removed={u_fix-a_fix} "
              f"(vs scaled {res['optimize_oracle'][C]['records_removed']})")
    sens["protected_cap_fixed_vs_scaled"] = prot
    # (iii) round allocations to multiples of 4
    rnd = {}
    for C in C_UNIFORM:
        budget = len(layers) * C
        a4 = round_to_mult(alloc_opt[C], layers, budget, 4)
        reads4 = alloc_total(tot, a4, caps)
        removed = u[C] - reads4
        rnd[C] = {"reads": reads4, "removed": removed, "pct": 100 * removed / u[C],
                  "seconds_removed": secs(removed), "budget_ok": sum(a4.values()) == budget,
                  "alloc": {int(L): int(a4[L]) for L in layers}}
        print(f"[sens round4 C={C}] removed={removed} ({100*removed/u[C]:.2f}%)")
    sens["round_to_multiple_of_4"] = rnd
    res["sensitivity"] = sens

    RECEIPT.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(res, indent=2, default=float) + "\n")
    print(f"wrote {out_json} (mlx_imported={res['mlx_imported']})")


def _uniform_plus(tot, layers, caps, C, d):
    """Reads if the +-d extra/fewer rows were spread one-per-layer (a 'uniform-like'
    reference at the perturbed total)."""
    per = {L: C for L in layers}
    step = 1 if d > 0 else -1
    for i in range(abs(d)):
        L = layers[i % len(layers)]
        per[L] = min(CMAX, max(CMIN, per[L] + step))
    return sum(int(tot[L][_c_index(caps, per[L])]) for L in layers)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    pc = sub.add_parser("curves")
    pc.add_argument("--lo", type=int, required=True)
    pc.add_argument("--hi", type=int, required=True)
    pc.add_argument("--grid", choices=("dense", "grid16"), default="dense")
    pc.add_argument("--out", type=Path, required=True)
    pa = sub.add_parser("analyze")
    pa.add_argument("--npz", type=Path, nargs="+", required=True)
    pa.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if any(m == "mlx" or m.startswith("mlx.") for m in sys.modules):
        raise SystemExit("MLX leaked into the process")
    if a.phase == "curves":
        caps = DENSE if a.grid == "dense" else GRID16
        phase_curves(a.lo, a.hi, caps, a.out)
    else:
        phase_analyze(a.npz, a.out)


if __name__ == "__main__":
    main()
