"""W95 retune: offline precision / economics of the gate-oracle prefetch set.

Window 39 showed the k=12 one-ahead prefetch ENGAGED (misses 58.8 -> 38.9/token)
but OVER-ISSUED (164 speculative reads/token -> SSD 20% -> 70% busy, speculative
contending with demand -> net -6%).  Economics (coordinator): a hidden miss saves
~2.7 ms of GPU stall; a speculative read costs ~1.8 ms of drive -> the drive pays
~0.67 read per hidden miss, so the issued set must maximise

    objective = hidden_misses - 0.67 * wasted_reads

and needs precision >= ~0.5 to be net-positive.  This computes, on the W89
one-layer-ahead gate oracle (real router weights on the previous layer's stored
residual, the tensor W89 measured), precision / recall / issued-size / objective
across the prefetch width ``k`` and the confidence MARGIN: issue a predicted
expert iff its gate score >= (the predicted 6th-highest score) - margin.

  * margin >= 0 WIDENS beyond the top-6 (keep rank>6 runners-up within margin);
  * margin  < 0 TRIMS the top-6 to the confident ones (score clearly above s6).

Residency is capacity-dependent and not in the trace, so precision/recall/wasted
are vs the TRUE top-6; hidden_misses is proportional (a ~const miss fraction of
the true route), so the ARGMAX operating point is residency-independent.  The
absolute score scale (mean s1, s6, s1-s6 gap) is printed so the chosen margin is
interpretable and portable to the runtime env MTPLX_DSV41_GATE_PREFETCH_MARGIN.

CPU-only; one layer loaded at a time (<1.5 GB); reuses the W89 evaluator's Trace +
gate-weight reader + the port's exact ``_gate_prefix_impl``.  No GPU, read-only.

Run (inside the worktree):
    PYTHONPATH=$PWD nice -n 19 .venv/bin/python3 \
      scripts/deepseek_v41/w95_prefetch_precision.py \
      --traces .worktrees/deepseek-v41/.benchmark-artifacts/deepseek-v41/route-traces-w35 \
      --gate-weights /Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Reuse the W89 evaluator's trace reader + gate-weight loader (same module).
from scripts.deepseek_v41.train_route_predictor import Trace, _load_gate_weights

COST_PER_READ = 0.67  # wasted-read penalty vs a hidden miss (1.8 ms / 2.7 ms)
MIN_LAYER_DEFAULT = 4  # W89: the first ~4 layers are the residual-turnover floor


def _biased_scores(x_np, gw, gb, temp, sf):
    import mlx.core as mx

    from mtplx.models import deepseek_v41_moe as moe

    _scores, biased = moe._gate_prefix_impl(mx.array(x_np), gw, gb, temp, sf)
    return np.array(biased)


def _gate_metrics(biased, true6, k, margin, top_k=6):
    """Confidence-gated issued set per row and its economics vs the true top-6.

    issued = { top-k experts with score >= s_{top_k} - margin }, where s_{top_k}
    is the top_k-th highest score (the routing boundary)."""
    n = biased.shape[0]
    order = np.argsort(-biased, axis=1)
    topk = order[:, :k]                                   # [n, k] candidate ids
    row = np.arange(n)[:, None]
    topk_scores = biased[row, topk]                       # [n, k] scores, desc
    s_boundary = np.sort(biased, axis=1)[:, -top_k][:, None]  # top_k-th highest
    keep = topk_scores >= (s_boundary - margin)           # [n, k] bool
    true_sets = [set(int(e) for e in r) for r in true6]
    issued_tot = hits_tot = wasted_tot = 0
    prec_sum = rec_sum = 0.0
    for i in range(n):
        issued = [int(topk[i, j]) for j in range(k) if keep[i, j]]
        tset = true_sets[i]
        hits = sum(1 for e in issued if e in tset)
        wasted = len(issued) - hits
        issued_tot += len(issued)
        hits_tot += hits
        wasted_tot += wasted
        prec_sum += (hits / len(issued)) if issued else 1.0
        rec_sum += hits / max(1, len(tset))
    return {
        "issued": issued_tot / n,
        "hits": hits_tot / n,
        "wasted": wasted_tot / n,
        "precision": prec_sum / n,
        "recall": rec_sum / n,
        "objective": (hits_tot - COST_PER_READ * wasted_tot) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--gate-weights", type=Path, required=True)
    ap.add_argument("--min-layer", type=int, default=MIN_LAYER_DEFAULT)
    ap.add_argument("--ks", type=int, nargs="+", default=[6, 8, 12])
    ap.add_argument("--out", type=Path, default=None, help="write the markdown table")
    args = ap.parse_args()

    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from mtplx.models.deepseek_v41 import ModelArgs

    cfg = json.loads((args.gate_weights / "config.json").read_text())
    margs = ModelArgs.from_dict(cfg)
    temp = float(getattr(margs, "gate_temp", 1.0) or 1.0)
    sf = margs.scoring_func

    trace = Trace(args.traces)
    gates = _load_gate_weights(args.gate_weights, trace.layer_ids)
    lids = trace.layer_ids
    test = trace.test_mask
    top_k = trace.top_k

    # calibrate the score scale (mean s1, s6, s1-s6) on the one-ahead residual
    s1s, s6s = [], []
    # margins in absolute score units, plus 0 (== issue exactly top-6) and trims
    margins = [-0.30, -0.15, -0.05, 0.0, 0.05, 0.15, 0.30]
    acc = {(k, m): [] for k in args.ks for m in margins}

    for pos, L in enumerate(lids):
        if pos < 1 or int(L) < args.min_layer:
            continue
        gw, gb = gates[L]["weight"], gates[L]["bias"]
        li_prev = trace.layer_in(lids[pos - 1])[test]   # one layer ahead (b')
        true6 = trace.top6(L)[test]
        biased = _biased_scores(li_prev, gw, gb, temp, sf)
        sr = np.sort(biased, axis=1)
        s1s.append(float(np.mean(sr[:, -1])))
        s6s.append(float(np.mean(sr[:, -top_k])))
        for k in args.ks:
            for m in margins:
                acc[(k, m)].append(_gate_metrics(biased, true6, k, m, top_k))
        del biased, li_prev

    def _mean(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    s1 = float(np.mean(s1s)); s6 = float(np.mean(s6s)); gap = s1 - s6
    lines = []
    lines.append(f"# W95 prefetch precision/economics (one-ahead gate oracle, "
                 f"layers >= {args.min_layer})")
    lines.append("")
    lines.append(f"Score scale (mean over layers/decode rows): s1={s1:.4f}, "
                 f"s{top_k}={s6:.4f}, gap(s1-s{top_k})={gap:.4f}. "
                 f"objective = hits - {COST_PER_READ} * wasted (per layer-route; "
                 f"hidden_misses proportional to hits).")
    lines.append("")
    lines.append("| k | margin | issued | hits | wasted | precision | recall | objective |")
    lines.append("|--:|-------:|-------:|-----:|-------:|----------:|-------:|----------:|")
    best = None
    for k in args.ks:
        for m in margins:
            rows = acc[(k, m)]
            r = {kk: _mean(rows, kk) for kk in
                 ("issued", "hits", "wasted", "precision", "recall", "objective")}
            lines.append(
                f"| {k} | {m:+.2f} | {r['issued']:.2f} | {r['hits']:.2f} | "
                f"{r['wasted']:.2f} | {r['precision']:.3f} | {r['recall']:.3f} | "
                f"{r['objective']:.3f} |"
            )
            if r["precision"] >= 0.5 and (best is None or r["objective"] > best[2]["objective"]):
                best = (k, m, r)
    lines.append("")
    if best is not None:
        k, m, r = best
        lines.append(f"**Pick (max objective, precision >= 0.5): k={k}, margin={m:+.2f} "
                     f"(= {m/gap:+.2f} x the s1-s{top_k} gap) -> precision {r['precision']:.3f}, "
                     f"recall {r['recall']:.3f}, issued {r['issued']:.2f}/layer, "
                     f"objective {r['objective']:.3f}.**")
    else:
        lines.append("**No (k, margin) reached precision >= 0.5 on this trace.**")
    text = "\n".join(lines)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
