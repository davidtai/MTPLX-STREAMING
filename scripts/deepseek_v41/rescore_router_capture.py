#!/usr/bin/env python3
"""Re-score the router-feature-20260918 capture across the FULL issue-budget curve.

CPU ONLY (MLX imports are hard-blocked), numpy only, single process; run under
`nice -n 19`.  This is a retrospective screen, not a throughput result and not a
production-code proposal.

Codex's router-feature-20260918 capture recorded, for the first 64 native D5/M6
verify cycles of the exact 16,384/1,024 run, for the 36 target layers 4-39:
  * scores[feature, cycle, target_idx, row, expert]  -- the NEXT layer's gate
    applied to (0) the mean residual before attention and (1) the native router
    input after attention of the CURRENT layer (a 1-layer-ahead prediction).
  * actual[cycle, layer, row, :6]                     -- native top-6 routes.
  * physical[cycle, target_idx, expert]               -- READY physical owners at
    prediction time (persistent + READY transient); these are excluded.
  * reads[cycle, layer, expert] in {0,1}              -- actual physical record
    reads (demand misses); no duplicate reads in the capture.

Codex only evaluated it at a >=85%-precision operating point (per-layer width /
margin fitted on cycles 0-31).  This module produces the WHOLE curve with a
parameter-free ranked predictor: no precision floor and no per-layer selection.

For each input (feature) and merge rule (max-over-rows / sum-over-rows) we build
one per-layer-call prediction list = the rank-ordered union over the <=6 verify
rows (per-expert merged score, READY physical residents excluded), and for issue
budget k in {1,2,3,4,6,8,12} report miss coverage (useful predicted misses /
actual physical misses) and precision (useful / issued) on held-out cycles 32-63
and on all 64, plus a per-layer breakdown for the best merge rule.

It also exposes real_predictions() so overlap_schedule_sim.py can replay the same
ranked predictions through the discrete-event overlap model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import sys
from pathlib import Path


class _NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise RuntimeError("rescore_router_capture is CPU-only; MLX is forbidden")


if not any(isinstance(f, _NoMLX) for f in sys.meta_path):
    sys.meta_path.insert(0, _NoMLX())

import numpy as np  # noqa: E402  (after guard, like the sibling receipts)

# ---------------------------------------------------------------------------
# Capture (read-only; sha256 asserted at load).  The NPZ lives in the read-only
# deepseek-v41 worktree's ignored artifact dir (capture-artifact.json pins it).
# ---------------------------------------------------------------------------
CAPTURE_SHA256 = "5d8dd85c412f0c8e332733843e6eb9ed36ac6c8a8e5a7c71c8c2615e71edca3e"
CAPTURE_CANDIDATES = (
    Path("/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41/"
         ".benchmark-artifacts/deepseek-v41/router-feature-20260918/"
         "full-router-feature-20260918-v2.router-capture.npz"),
)
CYCLES = 64
FIRST_TARGET = 4                       # target layers are 4..39
N_TARGET = 36
ROWS = 6
N_EXPERTS = 384
HELDOUT = slice(32, 64)                # calibration-free held-out half
ALL = slice(0, 64)
FEATURES = ("existing_pre_attention_mean", "post_attention_router")
MERGE_RULES = ("max", "sum")
BUDGETS = (1, 2, 3, 4, 6, 8, 12)


def find_capture(path: Path | None = None) -> Path:
    cands = [path] if path is not None else list(CAPTURE_CANDIDATES)
    checked = []
    for p in cands:
        checked.append(str(p))
        if p and p.exists():
            return p
    raise SystemExit("router capture NPZ not found; checked:\n  " + "\n  ".join(checked))


def load_capture(path: Path | None = None) -> dict:
    p = find_capture(path)
    digest = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            digest.update(blk)
    got = digest.hexdigest()
    if got != CAPTURE_SHA256:
        raise SystemExit(f"capture sha256 mismatch at {p}: {got}")
    z = np.load(p)
    scores = z["scores"]        # (2, 64, 36, 6, 384) f32
    actual = z["actual"]        # (64, 40, 6, 6) i32
    nrows = z["nrows"]          # (64, 40) u8
    persistent = z["persistent"]  # (64, 36, 384) bool
    physical = z["physical"]    # (64, 36, 384) bool  READY owners at predict time
    reads = z["reads"]          # (64, 40, 384) u8   actual physical reads
    assert scores.shape == (2, CYCLES, N_TARGET, ROWS, N_EXPERTS), scores.shape
    assert reads.shape == (CYCLES, 40, N_EXPERTS), reads.shape
    if int(reads.max()) > 1:
        raise SystemExit("capture has duplicate reads; miss==read assumption broken")
    # target-layer demand misses (layers 4..39): (64, 36, 384)
    missing = reads[:, FIRST_TARGET:, :] > 0
    return {
        "path": str(p), "sha256": got, "scores": scores, "actual": actual,
        "nrows": nrows, "persistent": persistent, "physical": physical,
        "reads": reads, "missing": missing,
    }


# ---------------------------------------------------------------------------
# Ranked predictor (parameter-free)
# ---------------------------------------------------------------------------
def merged_scores(scores_feat: np.ndarray, rule: str) -> np.ndarray:
    """(64,36,6,384) -> (64,36,384): merge the per-row gate scores per expert.

    All 36 target layers carry exactly 6 verify rows in every captured cycle
    (asserted at capture time), so the merge is over all 6 rows.
    """
    if rule == "max":
        return scores_feat.max(axis=2)
    if rule == "sum":
        return scores_feat.sum(axis=2)
    raise ValueError(rule)


def _ranked(merged: np.ndarray, physical: np.ndarray) -> np.ndarray:
    """Descending expert order per (cycle, target_idx), READY residents removed."""
    m = merged.astype(np.float64, copy=True)
    m[physical] = -np.inf
    return np.argsort(-m, axis=-1, kind="stable"), m


def issued_mask(merged: np.ndarray, physical: np.ndarray, k: int) -> np.ndarray:
    """Top-k rank-ordered union per (cycle, target_idx), excluding READY residents.

    Captured gate scores are strictly positive, so every non-resident expert is a
    finite candidate; top-k always returns k distinct non-resident experts here.
    """
    order, m = _ranked(merged, physical)
    topk = order[..., :k]
    valid = np.isfinite(np.take_along_axis(m, topk, axis=-1))
    out = np.zeros(merged.shape, bool)
    np.put_along_axis(out, topk, valid, axis=-1)
    return out


def _layerwise(a: np.ndarray, sl: slice) -> np.ndarray:
    """Sum a boolean (64,36,384) over the cycle slice and the expert axis -> (36,)."""
    return a[sl].sum(axis=(0, 2))


def curve(cap: dict, feature: int, rule: str, budgets=BUDGETS) -> dict:
    scores_feat = cap["scores"][feature]
    physical = cap["physical"]
    missing = cap["missing"]
    merged = merged_scores(scores_feat, rule)
    rows = []
    for k in budgets:
        issued = issued_mask(merged, physical, k)
        useful = issued & missing
        rec = {"k": k}
        for name, sl in (("all", ALL), ("heldout", HELDOUT)):
            iss = int(issued[sl].sum())
            usf = int(useful[sl].sum())
            mis = int(missing[sl].sum())
            rec[name] = {
                "issued": iss, "useful": usf, "actual_misses": mis,
                "precision": (usf / iss) if iss else None,
                "miss_coverage": (usf / mis) if mis else 0.0,
            }
        rows.append(rec)
    return {"feature": FEATURES[feature], "merge_rule": rule, "budgets": rows}


def per_layer(cap: dict, feature: int, rule: str, k: int, sl: slice = HELDOUT) -> list:
    scores_feat = cap["scores"][feature]
    physical = cap["physical"]
    missing = cap["missing"]
    merged = merged_scores(scores_feat, rule)
    issued = issued_mask(merged, physical, k)
    useful = issued & missing
    iss = _layerwise(issued, sl)
    usf = _layerwise(useful, sl)
    mis = _layerwise(missing, sl)
    out = []
    for i in range(N_TARGET):
        out.append({
            "layer": i + FIRST_TARGET,
            "issued": int(iss[i]), "useful": int(usf[i]), "actual_misses": int(mis[i]),
            "precision": (float(usf[i] / iss[i]) if iss[i] else None),
            "miss_coverage": (float(usf[i] / mis[i]) if mis[i] else 0.0),
        })
    return out


def coverage_ceiling(cap: dict, sl: slice = HELDOUT) -> dict:
    """Max attainable coverage: fraction of demand misses whose expert is NOT a
    READY resident at prediction time (the honest exclusion caps coverage < 1)."""
    missing = cap["missing"]
    physical = cap["physical"]
    predictable = missing & ~physical
    mis = int(missing[sl].sum())
    return {"actual_misses": mis, "predictable_misses": int(predictable[sl].sum()),
            "excluded_resident_misses": int((missing & physical)[sl].sum()),
            "ceiling_coverage": (float(predictable[sl].sum() / mis) if mis else 0.0)}


# ---------------------------------------------------------------------------
# Ranked predictions for the discrete-event sim
# ---------------------------------------------------------------------------
def real_predictions(cap: dict, feature: int, rule: str, width: int) -> dict:
    """Ranked top-`width` predicted misses for TARGET layer L+1, keyed by the
    compute-window layer L (L in 3..38 -> target 4..39).  READY residents removed.
    The sim caps issuance by the compute window, so width need only exceed the
    window's admission (~2.37 records)."""
    scores_feat = cap["scores"][feature]
    physical = cap["physical"]
    merged = merged_scores(scores_feat, rule)
    order, m = _ranked(merged, physical)
    top = order[..., :width]
    pred = {}
    for c in range(CYCLES):
        for idx in range(N_TARGET):
            L = idx + FIRST_TARGET - 1          # window layer for target idx+4
            row = top[c, idx]
            vals = m[c, idx, row]
            pred[(c, L)] = [int(e) for e, v in zip(row, vals) if np.isfinite(v)]
    return pred


def demand_misses(cap: dict) -> list:
    """cap_misses[cycle][layer] = experts actually read (demand miss) that layer.

    Layers 0-2 were not instrumented in the capture (reads==0 there); layer 3 and
    layers 4-39 carry the real captured physical reads.
    """
    reads = cap["reads"]
    out = []
    for c in range(CYCLES):
        row = []
        for L in range(40):
            row.append([int(e) for e in np.nonzero(reads[c, L] > 0)[0]])
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def analyze(cap: dict) -> dict:
    families = {}
    for f, fname in enumerate(FEATURES):
        families[fname] = {rule: curve(cap, f, rule) for rule in MERGE_RULES}
    # best merge rule = highest held-out coverage at k=8 for the stronger feature.
    def cov8(f, rule):
        row = next(r for r in families[FEATURES[f]][rule]["budgets"] if r["k"] == 8)
        return row["heldout"]["miss_coverage"]
    best = max(((f, rule) for f in range(2) for rule in MERGE_RULES),
               key=lambda fr: cov8(*fr))
    bf, brule = best
    return {
        "capture_path": cap["path"], "capture_sha256": cap["sha256"],
        "cycles": CYCLES, "target_layers": list(range(FIRST_TARGET, 40)),
        "held_out_cycles": [32, 63], "budgets": list(BUDGETS),
        "merge_rules": list(MERGE_RULES),
        "merge_rule_definition":
            "Per-expert merged score = max (resp. sum) of the per-row predicted "
            "gate score over the 6 verify rows; experts ranked by merged score "
            "descending; READY physical residents excluded; top-k issued. This is "
            "the rank-ordered union of the per-row candidate experts.",
        "coverage_ceiling_heldout": coverage_ceiling(cap, HELDOUT),
        "coverage_ceiling_all": coverage_ceiling(cap, ALL),
        "families": families,
        "best_feature": FEATURES[bf], "best_merge_rule": brule,
        "per_layer_best_heldout": {
            f"k{k}": per_layer(cap, bf, brule, k, HELDOUT) for k in (1, 2, 3, 4, 8)},
        "two_ahead_evaluable": False,
        "two_ahead_note":
            "NOT evaluable. The capture stores, at layer L, only the L+1 gate "
            "applied to layer L's features (a single 1-ahead score tensor per "
            "target). No L+2 score tensor exists at layer L, so two-layers-ahead "
            "predictor quality cannot be measured from this capture.",
    }


def _print_curve(res: dict) -> None:
    print(f"capture {res['capture_path']}")
    print(f"  sha256 {res['capture_sha256']}")
    cc = res["coverage_ceiling_heldout"]
    print(f"  held-out actual misses={cc['actual_misses']} "
          f"predictable(non-resident)={cc['predictable_misses']} "
          f"ceiling coverage={cc['ceiling_coverage']*100:.2f}%")
    for fname in FEATURES:
        for rule in MERGE_RULES:
            print(f"\n[{fname} | merge={rule}]  (held-out cycles 32-63)")
            print("  {:>3} {:>8} {:>8} {:>10} {:>12}".format(
                "k", "issued", "useful", "precision", "coverage"))
            for r in res["families"][fname][rule]["budgets"]:
                h = r["heldout"]
                p = "-" if h["precision"] is None else f"{h['precision']*100:6.2f}%"
                print("  {:>3} {:>8} {:>8} {:>10} {:>11.2f}%".format(
                    r["k"], h["issued"], h["useful"], p, h["miss_coverage"] * 100))
    print(f"\nbest: {res['best_feature']} / merge={res['best_merge_rule']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path(
        "docs/deepseek-v41/receipts/f1-real-predictor-20260919/curve.json"))
    args = ap.parse_args()
    if any(m == "mlx" or m.startswith("mlx.") for m in sys.modules):
        raise SystemExit("MLX leaked into the process")
    cap = load_capture(args.capture)
    res = analyze(cap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2) + "\n")
    _print_curve(res)


if __name__ == "__main__":
    main()
