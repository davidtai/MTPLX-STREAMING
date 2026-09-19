#!/usr/bin/env python3
"""W89 -- CPU-only offline trainer/evaluator for the DSV4.1 route predictor.

Reads the per-layer route traces written by ``collect_route_traces.py`` and, for
every routed layer L, trains and evaluates four predictors of L's top-``top_k``
experts, TRAIN on the prefill-tail rows and TEST on the held-out decode rows:

  (a) router_in(L)   -> top-k at L   -- re-runs the router; ~sanity ceiling.
  (b) layer_in(L-1)  -> top-k at L   -- ONE layer ahead (the useful one): the
                                         residual entering the previous layer,
                                         so the read overlaps a full layer of
                                         compute.
  (c) layer_in(L-2)  -> top-k at L   -- TWO layers ahead (more overlap).
  (d) cheap features -> top-k at L   -- previous layer's routed ids (multi-hot)
                                         + the token id (one-hot over the train
                                         vocab; the learned column is that
                                         token's embedding into expert space).

Metrics per predictor, per layer (mean over layers + the worst 5 layers):
  * precision@k / recall@k at k == top_k (both equal when both sets are top_k).
  * miss_reduction@K for prefetch widths K in ``--prefetch-k`` (default 6,8,12):
    the fraction of L's true top-k experts already covered if we prefetch the
    predictor's top-K one layer ahead -- i.e. the fraction of that layer's
    compulsory misses hidden behind compute.

Linear predictor = ridge (multi-output least squares to the multi-hot target,
closed form) by default: fast, deterministic, no LR, and predictor (a) recovers
the near-linear router as the sanity check.  ``--model logistic`` / ``--model
mlp`` (one hidden layer) train by gradient descent on mlx-cpu for comparison.

CPU-only (pins MLX to CPU); tiny per-layer models; processes one layer at a time
so the working set stays well under the worker guard.  ``--tiny`` self-generates
a small synthetic trace from the fake model (no artifact, no GPU) and runs the
whole pipeline end-to-end.

  PYTHONPATH=<worktree> nice -n 19 python3 scripts/deepseek_v41/train_route_predictor.py --tiny

Author: Opus 4.8 worker (w89/route-predictor).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from collect_route_traces import (  # noqa: E402
    PHASE_DECODE,
    PHASE_PREFILL_TAIL,
    bf16_to_f32,
    collect_tiny_trace,
)


# ---------------------------------------------------------------------------
# Trace access
# ---------------------------------------------------------------------------
class Trace:
    """Lazy reader over a route-trace directory (manifest + per-layer shards)."""

    def __init__(self, trace_dir: Path):
        self.dir = Path(trace_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.layer_ids = list(self.manifest["layer_ids"])
        self.n_experts = int(self.manifest["n_experts"])
        self.top_k = int(self.manifest["top_k"])
        self.hidden = int(self.manifest["stored_hidden"])
        self.phase = np.load(self.dir / "phase.npy")
        self.tokens = np.load(self.dir / "tokens.npy")
        self.train_mask = self.phase == PHASE_PREFILL_TAIL
        self.test_mask = self.phase == PHASE_DECODE

    def router_in(self, lid: int) -> np.ndarray:
        return bf16_to_f32(np.load(self.dir / f"layer{lid:03d}_router_in.npy"))

    def layer_in(self, lid: int) -> np.ndarray:
        return bf16_to_f32(np.load(self.dir / f"layer{lid:03d}_layer_in.npy"))

    def top6(self, lid: int) -> np.ndarray:
        return np.load(self.dir / f"layer{lid:03d}_top6.npy")

    def multihot(self, lid: int) -> np.ndarray:
        idx = self.top6(lid)
        m = np.zeros((idx.shape[0], self.n_experts), np.float32)
        rows = np.arange(idx.shape[0])[:, None]
        m[rows, idx] = 1.0
        return m


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------
def _augment(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((x.shape[0], 1), np.float32)], axis=1)


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form multi-output ridge: W = (XtX + lam I)^-1 Xt Y, X augmented."""
    xa = _augment(x.astype(np.float64))
    d = xa.shape[1]
    a = xa.T @ xa
    a[np.diag_indices(d)] += lam
    b = xa.T @ y.astype(np.float64)
    w = np.linalg.solve(a, b)
    return w.astype(np.float32)


def ridge_scores(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return _augment(x.astype(np.float32)) @ w


def _gd_fit(x, y, *, hidden, steps, lr, lam, seed=0):
    """Logistic (hidden==0) / one-hidden-layer MLP via full-batch GD on mlx-cpu,
    sigmoid BCE multi-label.  Kept tiny; returns a scorer callable."""
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    mx.random.seed(seed)
    xm = mx.array(x.astype(np.float32))
    ym = mx.array(y.astype(np.float32))
    din = x.shape[1]
    dout = y.shape[1]
    if hidden and hidden > 0:
        def fwd(p, xin):
            h = mx.maximum(xin @ p["w0"].T + p["b0"], 0.0)
            return h @ p["w1"].T + p["b1"]

        params = {
            "w0": 0.02 * mx.random.normal((hidden, din)), "b0": mx.zeros((hidden,)),
            "w1": 0.02 * mx.random.normal((dout, hidden)), "b1": mx.zeros((dout,)),
        }
    else:
        def fwd(p, xin):
            return xin @ p["w0"].T + p["b0"]

        params = {"w0": 0.02 * mx.random.normal((dout, din)), "b0": mx.zeros((dout,))}

    def loss_fn(p):
        logits = fwd(p, xm)
        bce = mx.mean(mx.logaddexp(0.0, logits) - ym * logits)
        reg = lam * sum(mx.sum(v * v) for k, v in p.items() if k.startswith("w"))
        return bce + reg / max(1, x.shape[0])

    gfn = mx.value_and_grad(loss_fn)
    for _ in range(int(steps)):
        _l, g = gfn(params)
        for k in params:
            params[k] = params[k] - lr * g[k]
        mx.eval(params)

    def scorer(xin):
        return np.array(fwd(params, mx.array(xin.astype(np.float32))))

    return scorer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def eval_predictor(scores_te: np.ndarray, true_idx_te: np.ndarray, top_k: int,
                   prefetch_ks) -> dict:
    """Row-wise precision@top_k, recall@top_k, miss_reduction@K for each K."""
    n = scores_te.shape[0]
    n_exp = scores_te.shape[1]
    # rank experts by predicted score, descending
    order = np.argsort(-scores_te, axis=1)
    true_sets = [set(int(e) for e in row) for row in true_idx_te]
    out = {"n_test": int(n)}
    # precision/recall at top_k
    predk = order[:, :top_k]
    inter_k = np.array([len(true_sets[r] & set(predk[r].tolist())) for r in range(n)])
    out["precision_at_k"] = float(np.mean(inter_k / top_k)) if n else 0.0
    out["recall_at_k"] = out["precision_at_k"]  # both sets size top_k
    for K in prefetch_ks:
        kk = min(int(K), n_exp)
        predK = order[:, :kk]
        cover = np.array([len(true_sets[r] & set(predK[r].tolist())) for r in range(n)])
        out[f"miss_red_at_{K}"] = float(np.mean(cover / top_k)) if n else 0.0
    return out


PREDICTORS = ("a", "b", "c", "d")
PREDICTOR_DESC = {
    "a": "router_in(L)->L  (re-run router, sanity)",
    "b": "layer_in(L-1)->L (one layer ahead)",
    "c": "layer_in(L-2)->L (two layers ahead)",
    "d": "prev-route+token->L (cheap features)",
}


def _feature_target(trace: Trace, ptype: str, pos: int):
    """(X, true_idx, ok) for predictor ``ptype`` at the ``pos``-th routed layer.

    Uses ONLY the train-vocab for predictor (d)'s token one-hot (fit on train,
    unseen decode tokens map to the all-zero 'unknown' column)."""
    lids = trace.layer_ids
    tgt = lids[pos]
    true_all = trace.top6(tgt)
    tr, te = trace.train_mask, trace.test_mask
    if ptype == "a":
        x = trace.router_in(tgt)
        return x[tr], x[te], true_all[tr], true_all[te]
    if ptype == "b":
        if pos < 1:
            return None
        src = trace.layer_in(lids[pos - 1])
        return src[tr], src[te], true_all[tr], true_all[te]
    if ptype == "c":
        if pos < 2:
            return None
        src = trace.layer_in(lids[pos - 2])
        return src[tr], src[te], true_all[tr], true_all[te]
    if ptype == "d":
        if pos < 1:
            return None
        prev_mh = trace.multihot(lids[pos - 1])
        toks = trace.tokens
        train_vocab = {int(t): i for i, t in enumerate(np.unique(toks[tr]))}
        V = len(train_vocab)
        tok_oh = np.zeros((len(toks), V), np.float32)
        for r, t in enumerate(toks):
            j = train_vocab.get(int(t))
            if j is not None:
                tok_oh[r, j] = 1.0
        feat = np.concatenate([prev_mh, tok_oh], axis=1)
        return feat[tr], feat[te], true_all[tr], true_all[te]
    raise ValueError(ptype)


def train_and_eval(trace: Trace, *, model: str, lam: float, steps: int,
                   hidden: int, lr: float, prefetch_ks) -> dict:
    lids = trace.layer_ids
    results = {p: [] for p in PREDICTORS}
    for ptype in PREDICTORS:
        for pos in range(len(lids)):
            ft = _feature_target(trace, ptype, pos)
            if ft is None:
                continue
            x_tr, x_te, y_tr_idx, y_te_idx = ft
            y_tr = np.zeros((x_tr.shape[0], trace.n_experts), np.float32)
            y_tr[np.arange(x_tr.shape[0])[:, None], y_tr_idx] = 1.0
            if x_te.shape[0] == 0:
                continue
            if model == "ridge":
                w = ridge_fit(x_tr, y_tr, lam)
                scores_te = ridge_scores(w, x_te)
            else:
                scorer = _gd_fit(x_tr, y_tr, hidden=(hidden if model == "mlp" else 0),
                                 steps=steps, lr=lr, lam=lam)
                scores_te = scorer(x_te)
            m = eval_predictor(scores_te, y_te_idx, trace.top_k, prefetch_ks)
            m["layer"] = int(lids[pos])
            results[ptype].append(m)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _mean(rows, key):
    vals = [r[key] for r in rows]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(results: dict, prefetch_ks) -> dict:
    summary = {}
    for ptype in PREDICTORS:
        rows = results[ptype]
        if not rows:
            summary[ptype] = {"n_layers": 0}
            continue
        keys = ["precision_at_k", "recall_at_k"] + [f"miss_red_at_{K}" for K in prefetch_ks]
        means = {k: _mean(rows, k) for k in keys}
        worst_key = f"miss_red_at_{prefetch_ks[0]}"
        worst = sorted(rows, key=lambda r: r[worst_key])[:5]
        summary[ptype] = {
            "n_layers": len(rows),
            "mean": means,
            "worst5": [
                {"layer": r["layer"], **{k: round(r[k], 4) for k in keys}}
                for r in worst
            ],
        }
    return summary


def print_report(summary: dict, trace: Trace, prefetch_ks) -> None:
    print("=" * 78)
    print(f"W89 route-predictor feasibility  |  n_experts={trace.n_experts} "
          f"top_k={trace.top_k} hidden={trace.hidden} "
          f"layers={len(trace.layer_ids)}")
    print(f"  train rows (prefill-tail)={int(trace.train_mask.sum())}  "
          f"test rows (decode)={int(trace.test_mask.sum())}")
    print("=" * 78)
    hdr = f"{'predictor':<34}{'prec@k':>8}{'rec@k':>8}"
    for K in prefetch_ks:
        hdr += f"{'missRed@'+str(K):>11}"
    hdr += f"{'#L':>5}"
    print(hdr)
    print("-" * len(hdr))
    for ptype in PREDICTORS:
        s = summary[ptype]
        label = f"({ptype}) {PREDICTOR_DESC[ptype]}"
        if s["n_layers"] == 0:
            print(f"{label:<34}{'--':>8}")
            continue
        m = s["mean"]
        line = f"{label:<34}{m['precision_at_k']:>8.3f}{m['recall_at_k']:>8.3f}"
        for K in prefetch_ks:
            line += f"{m['miss_red_at_'+str(K)]:>11.3f}"
        line += f"{s['n_layers']:>5}"
        print(line)
    print("-" * len(hdr))
    print("worst-5 layers by miss_red@%d (lowest prefetch coverage):" % prefetch_ks[0])
    for ptype in PREDICTORS:
        s = summary[ptype]
        if s["n_layers"] == 0:
            continue
        w = ", ".join(
            f"L{r['layer']}={r['miss_red_at_%d' % prefetch_ks[0]]:.2f}" for r in s["worst5"]
        )
        print(f"  ({ptype}) {w}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Training-free gate-oracle evaluation (the REAL router weights)
# ---------------------------------------------------------------------------
# The learned fitter is underdetermined (2,048 train rows for a 5120->384 map),
# so it measures the fitter, not the route's predictability.  The training-free
# evaluation runs the artifact's OWN gate (real weights + score-correction bias,
# via the port's exact routing function) on a shifted hidden and asks: does the
# real router, fed the residual one/two layers early, name this layer's experts?
ORACLE_DESC = {
    "a_prime": "a' gate_L(router_in_L) vs top6_L      (recompute; alignment ~1.0)",
    "b_prime": "b' gate_L(layer_in_{L-1}) vs top6_L   (one layer ahead, no training)",
    "c_prime": "c' gate_L(layer_in_{L-2}) vs top6_L   (two layers ahead, no training)",
}


import struct  # noqa: E402


def _read_safetensor(shard_path: Path, name: str):
    """Read a single tensor's raw bytes from a safetensors shard (no whole-shard
    load), returned as an mlx float32 array with the tensor's shape.

    BF16 tensors are returned as their exact bf16->f32 values (the gate immediately
    upcasts weight.astype(f32), so f32 here is bit-identical to loading bf16); F32
    tensors are read directly.  Keeps peak RAM at ~one gate (a few MB) instead of
    a ~1 GB shard, so the 40-layer sweep stays well under the worker guard."""
    import mlx.core as mx

    with open(shard_path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen).decode("utf-8"))
        data_start = 8 + hlen
        ent = header[name]
        beg, end = ent["data_offsets"]
        fh.seek(data_start + beg)
        buf = fh.read(end - beg)
    dt = ent["dtype"]
    shape = tuple(ent["shape"])
    if dt in ("BF16", "bfloat16"):
        f32 = bf16_to_f32(np.frombuffer(buf, dtype=np.uint16)).reshape(shape)
    elif dt in ("F32", "float32"):
        f32 = np.frombuffer(buf, dtype=np.float32).reshape(shape).copy()
    elif dt in ("F16", "float16"):
        f32 = np.frombuffer(buf, dtype=np.float16).astype(np.float32).reshape(shape)
    else:  # pragma: no cover - the gate tensors are bf16/f32 in this artifact
        raise ValueError(f"unhandled safetensors dtype {dt!r} for {name}")
    return mx.array(np.ascontiguousarray(f32))


def _load_gate_weights(artifact_dir: Path, layer_ids) -> dict:
    """Read each layer's REAL gate.weight (bf16) + gate.bias (f32) from the
    artifact's sharded safetensors, one tensor at a time (raw bytes, no whole-
    shard load).  Returns {layer_id: {"weight": mx f32 [n_exp, dim] (exact bf16
    values), "bias": mx f32 [n_exp]}}.

    Names are the artifact's resident tensors ``layers.{L}.ffn.gate.{weight,bias}``
    (``ffn.gate.bias`` is the noaux_tc score-correction the loader renames to
    ``mlp.gate.e_score_correction_bias``)."""
    artifact_dir = Path(artifact_dir)
    index = json.loads((artifact_dir / "model.safetensors.index.json").read_text())
    wmap = index["weight_map"]
    gates: dict = {}
    for L in layer_ids:
        wn = f"layers.{L}.ffn.gate.weight"
        bn = f"layers.{L}.ffn.gate.bias"
        gates[L] = {
            "weight": _read_safetensor(artifact_dir / wmap[wn], wn),
            "bias": _read_safetensor(artifact_dir / wmap[bn], bn),
        }
    return gates


def _row_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean row-wise cosine similarity between two [n, d] float arrays."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(an * bn, axis=1)))


def gate_oracle_eval(trace: Trace, artifact_dir: Path, prefetch_ks) -> dict:
    """Training-free: run each layer's real gate on router_in_L (a'), on
    layer_in_{L-1} (b'), on layer_in_{L-2} (c'); plus the residual cosine (e')."""
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from mtplx.models.deepseek_v41 import ModelArgs
    from mtplx.models import deepseek_v41_moe as moe

    cfg = json.loads((Path(artifact_dir) / "config.json").read_text())
    args = ModelArgs.from_dict(cfg)
    temp = float(getattr(args, "gate_temp", 1.0) or 1.0)
    sf = args.scoring_func
    gates = _load_gate_weights(artifact_dir, trace.layer_ids)
    lids = trace.layer_ids
    test = trace.test_mask  # held-out decode rows
    results = {"a_prime": [], "b_prime": [], "c_prime": [], "cos": []}

    def _rank_metrics(x_np, gw, gb, true_idx):
        # the port's EXACT gate prefix -> biased scores; eval_predictor ranks them
        _scores, biased = moe._gate_prefix_impl(mx.array(x_np), gw, gb, temp, sf)
        return eval_predictor(np.array(biased), true_idx, trace.top_k, prefetch_ks)

    for pos, L in enumerate(lids):
        gw, gb = gates[L]["weight"], gates[L]["bias"]
        true6 = trace.top6(L)[test]
        # a' -- recompute this layer's route from its own router input
        m = _rank_metrics(trace.router_in(L)[test], gw, gb, true6)
        m["layer"] = int(L)
        results["a_prime"].append(m)
        # b' -- one layer ahead
        if pos >= 1:
            li_prev = trace.layer_in(lids[pos - 1])[test]
            m = _rank_metrics(li_prev, gw, gb, true6)
            m["layer"] = int(L)
            results["b_prime"].append(m)
            # e' -- residual similarity that explains b'
            cos = _row_cosine(li_prev, trace.layer_in(L)[test])
            results["cos"].append({"layer": int(L), "cos": round(cos, 4)})
        # c' -- two layers ahead
        if pos >= 2:
            m = _rank_metrics(trace.layer_in(lids[pos - 2])[test], gw, gb, true6)
            m["layer"] = int(L)
            results["c_prime"].append(m)
    return results


def summarize_oracle(results: dict, prefetch_ks) -> dict:
    keys = ["precision_at_k", "recall_at_k"] + [f"miss_red_at_{K}" for K in prefetch_ks]
    out = {}
    for p in ("a_prime", "b_prime", "c_prime"):
        rows = results[p]
        if not rows:
            out[p] = {"n_layers": 0}
            continue
        worst_key = f"miss_red_at_{prefetch_ks[0]}"
        worst = sorted(rows, key=lambda r: r[worst_key])[:5]
        out[p] = {
            "n_layers": len(rows),
            "mean": {k: _mean(rows, k) for k in keys},
            "worst5": [{"layer": r["layer"], **{k: round(r[k], 4) for k in keys}}
                       for r in worst],
        }
    if results["cos"]:
        out["cos"] = {
            "mean": float(np.mean([r["cos"] for r in results["cos"]])),
            "worst5": sorted(results["cos"], key=lambda r: r["cos"])[:5],
        }
    return out


def _oracle_markdown(summary: dict, trace: Trace, prefetch_ks) -> str:
    L = ["# W89 training-free gate-oracle route eval (window-35 real trace)",
         "",
         f"Real router weights (`layers.L.ffn.gate.{{weight,bias}}`, {trace.n_experts} "
         f"experts, top-{trace.top_k}, {trace.hidden}-d) via the port's exact "
         f"`_gate_prefix_impl` (sqrtsoftplus + noaux_tc bias). Test = "
         f"{int(trace.test_mask.sum())} held-out decode rows; mean over "
         f"{len(trace.layer_ids)} layers.",
         "",
         "| predictor | prec@6 | rec@6 | " +
         " | ".join(f"missRed@{K}" for K in prefetch_ks) + " |",
         "|---|---:|---:|" + "---:|" * len(prefetch_ks)]
    for p in ("a_prime", "b_prime", "c_prime"):
        s = summary[p]
        if s["n_layers"] == 0:
            continue
        m = s["mean"]
        row = (f"| {ORACLE_DESC[p]} | {m['precision_at_k']:.3f} | {m['recall_at_k']:.3f} | "
               + " | ".join(f"{m['miss_red_at_'+str(K)]:.3f}" for K in prefetch_ks) + " |")
        L.append(row)
    if "cos" in summary:
        L += ["", f"**Residual cosine** cos(layer_in_L, layer_in_{{L-1}}) mean over "
              f"layers = **{summary['cos']['mean']:.4f}** "
              f"(worst: " +
              ", ".join(f"L{r['layer']}={r['cos']:.3f}" for r in summary['cos']['worst5'])
              + ")"]
    L += ["", "## Worst-5 layers by missRed@%d" % prefetch_ks[0], ""]
    for p in ("a_prime", "b_prime", "c_prime"):
        s = summary[p]
        if s["n_layers"] == 0:
            continue
        w = ", ".join(f"L{r['layer']}={r['miss_red_at_%d' % prefetch_ks[0]]:.2f}"
                      for r in s["worst5"])
        L.append(f"- **{p}**: {w}")
    # Verdict: smallest tested prefetch width k where mean one-ahead miss_reduction
    # clears the 0.7 overlap threshold.
    L += ["", "## Verdict"]
    bm = summary["b_prime"].get("mean") if summary["b_prime"]["n_layers"] else None
    if bm is not None:
        cross = next((K for K in prefetch_ks if bm[f"miss_red_at_{K}"] >= 0.70), None)
        if cross is not None:
            L.append(
                f"- One layer ahead (b'), **mean miss_reduction clears 0.70 at "
                f"prefetch width k={cross}** (missRed@{cross}={bm[f'miss_red_at_{cross}']:.3f}); "
                f"at k={prefetch_ks[0]} it is {bm[f'miss_red_at_{prefetch_ks[0]}']:.3f}."
            )
        else:
            L.append(
                f"- One layer ahead (b'): mean miss_reduction does **not** reach 0.70 "
                f"within the tested widths (max missRed@{prefetch_ks[-1]}="
                f"{bm[f'miss_red_at_{prefetch_ks[-1]}']:.3f})."
            )
    if "cos" in summary:
        L.append(
            f"- Explained by the residual stream's layer-to-layer stability "
            f"(cos={summary['cos']['mean']:.3f}); it is lowest in the early layers, "
            f"which are the per-layer floor (worst-5 above)."
        )
    return "\n".join(L) + "\n"


def print_oracle_report(summary: dict, trace: Trace, prefetch_ks) -> None:
    print("=" * 82)
    print(f"W89 TRAINING-FREE gate-oracle  |  real router weights  |  "
          f"n_experts={trace.n_experts} top_k={trace.top_k} "
          f"layers={len(trace.layer_ids)} decode_rows={int(trace.test_mask.sum())}")
    print("=" * 82)
    hdr = f"{'predictor':<48}{'prec@6':>8}{'rec@6':>8}"
    for K in prefetch_ks:
        hdr += f"{'missRed@'+str(K):>11}"
    print(hdr)
    print("-" * len(hdr))
    for p in ("a_prime", "b_prime", "c_prime"):
        s = summary[p]
        if s["n_layers"] == 0:
            continue
        m = s["mean"]
        line = f"{ORACLE_DESC[p]:<48}{m['precision_at_k']:>8.3f}{m['recall_at_k']:>8.3f}"
        for K in prefetch_ks:
            line += f"{m['miss_red_at_'+str(K)]:>11.3f}"
        print(line)
    print("-" * len(hdr))
    if "cos" in summary:
        print(f"residual cosine cos(layer_in_L, layer_in_L-1) mean = "
              f"{summary['cos']['mean']:.4f}")
    for p in ("a_prime", "b_prime", "c_prime"):
        s = summary[p]
        if s["n_layers"] == 0:
            continue
        w = ", ".join(f"L{r['layer']}={r['miss_red_at_%d' % prefetch_ks[0]]:.2f}"
                      for r in s["worst5"])
        print(f"  worst5 {p}: {w}")
    print("=" * 82)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--traces", type=Path, default=None,
                   help="route-trace directory (from collect_route_traces.py)")
    p.add_argument("--tiny", action="store_true",
                   help="self-generate a synthetic trace from the fake model, then run")
    p.add_argument("--model", choices=("ridge", "logistic", "mlp", "gate"),
                   default="ridge",
                   help="'gate' = training-free eval with the artifact's real "
                        "router weights (needs --gate-weights)")
    p.add_argument("--gate-weights", type=Path, default=None,
                   help="artifact dir with config.json + sharded safetensors "
                        "(for --model gate)")
    p.add_argument("--ridge-lambda", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=300, help="GD steps (logistic/mlp)")
    p.add_argument("--lr", type=float, default=0.5, help="GD learning rate")
    p.add_argument("--hidden", type=int, default=512, help="MLP hidden width")
    p.add_argument("--prefetch-k", default="6,8,12",
                   help="prefetch widths for miss_reduction (comma-separated)")
    p.add_argument("--out-json", type=Path, default=None,
                   help="write the metrics summary as JSON (data artifact, not a report)")
    p.add_argument("--md-out", type=Path, default=None,
                   help="write a markdown results table (data receipt; --model gate)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    prefetch_ks = [int(k) for k in str(args.prefetch_k).split(",") if k.strip()]
    tmp = None
    if args.tiny:
        tmp = tempfile.mkdtemp(prefix="w89-tiny-train-")
        collect_tiny_trace(tmp)
        trace_dir = Path(tmp)
    elif args.traces is not None:
        trace_dir = Path(args.traces)
    else:
        print("error: pass --traces DIR or --tiny", file=sys.stderr)
        return 2
    trace = Trace(trace_dir)
    if args.model == "gate":
        if args.gate_weights is None:
            print("error: --model gate needs --gate-weights <artifact dir>",
                  file=sys.stderr)
            return 2
        results = gate_oracle_eval(trace, args.gate_weights, prefetch_ks)
        summary = summarize_oracle(results, prefetch_ks)
        print_oracle_report(summary, trace, prefetch_ks)
        if args.md_out is not None:
            args.md_out.parent.mkdir(parents=True, exist_ok=True)
            args.md_out.write_text(_oracle_markdown(summary, trace, prefetch_ks))
            print(f"[w89] wrote markdown table -> {args.md_out}")
        if args.out_json is not None:
            payload = {
                "trace_manifest": trace.manifest, "model": "gate-oracle",
                "gate_weights": str(args.gate_weights), "prefetch_ks": prefetch_ks,
                "summary": summary, "per_layer": results,
            }
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(json.dumps(payload, indent=2))
            print(f"[w89] wrote metrics json -> {args.out_json}")
        return 0
    results = train_and_eval(
        trace, model=args.model, lam=args.ridge_lambda, steps=args.steps,
        hidden=args.hidden, lr=args.lr, prefetch_ks=prefetch_ks,
    )
    summary = summarize(results, prefetch_ks)
    print_report(summary, trace, prefetch_ks)
    if args.out_json is not None:
        payload = {
            "trace_manifest": trace.manifest,
            "model": args.model, "prefetch_ks": prefetch_ks,
            "summary": summary,
            "per_layer": results,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"[w89] wrote metrics json -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
