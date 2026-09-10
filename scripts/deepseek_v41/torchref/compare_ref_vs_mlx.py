#!/usr/bin/env python3
"""Compare the torch reference (ref_forward.py) against the MLX port (mlx_dump.py)
on the 31-token probe and NAME the first diverging layer/sublayer.

Pure numpy; loads the full float32 buffers under
``.benchmark-artifacts/deepseek-v41/w9/{,mlx/}*.npy`` and the committed receipts.
For each comparable tensor: per-position cosine (over the flattened feature axis)
and max-abs difference, plus the global cosine.  Router: per-token top-6 id set
overlap + weight L1.  Engram: exact row-id agreement per token/head.

A sublayer "diverges" when its min-over-positions cosine drops below --cos-thresh
(default 0.999) while its INPUT still agrees -- that localizes the defect to the
sublayer, distinct from quant noise (which shows as a small uniform deficit).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
NPY = REPO / ".benchmark-artifacts" / "deepseek-v41" / "w9"
RECEIPTS = REPO / "docs" / "deepseek-v41" / "receipts"


def load(rel):
    p = NPY / rel
    return np.load(p) if p.exists() else None


def per_pos_cos_maxabs(a, b):
    """a,b: [1,S,...] -> (cos[S], maxabs[S], global_cos)."""
    if a is None or b is None or a.shape != b.shape:
        return None
    A = a.reshape(a.shape[1], -1).astype(np.float64)
    B = b.reshape(b.shape[1], -1).astype(np.float64)
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-30
    cos = num / den
    maxabs = np.abs(A - B).max(1)
    g_num = float((A * B).sum())
    g_den = float(np.linalg.norm(A) * np.linalg.norm(B) + 1e-30)
    return cos, maxabs, g_num / g_den


def summ_line(name, res, thr):
    if res is None:
        return f"{name:28} MISSING/shape-mismatch", False
    cos, maxabs, gcos = res
    mn = float(cos.min())
    worst = int(cos.argmin())
    bad = mn < thr
    flag = "  <== DIVERGES" if bad else ""
    return (f"{name:28} min_cos={mn:.6f} @pos{worst:<2} "
            f"global_cos={gcos:.6f} max|Δ|={maxabs.max():.4g}{flag}"), bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cos-thresh", type=float, default=0.999)
    ap.add_argument("--max-layer", type=int, default=2)
    ap.add_argument("--out", type=Path, default=RECEIPTS / "compare_ref_vs_mlx.json")
    args = ap.parse_args(argv)
    thr = args.cos_thresh

    report = {"cos_thresh": thr, "rows": [], "router": {}, "engram_row_ids": {}, "first_divergence": None}
    first_div = None

    def check(name, ref_rel, mlx_rel, sub=None, layer=None):
        nonlocal first_div
        res = per_pos_cos_maxabs(load(ref_rel), load("mlx/" + mlx_rel))
        line, bad = summ_line(name, res, thr)
        print(line)
        row = {"name": name, "sub": sub, "layer": layer}
        if res is not None:
            cos, maxabs, gcos = res
            row.update(min_cos=float(cos.min()), argmin_pos=int(cos.argmin()),
                       global_cos=float(gcos), max_abs_diff=float(maxabs.max()),
                       per_pos_cos=[float(x) for x in cos])
        report["rows"].append(row)
        if bad and first_div is None:
            first_div = {"name": name, "sub": sub, "layer": layer,
                         "min_cos": row.get("min_cos"), "argmin_pos": row.get("argmin_pos")}

    print(f"=== ref(torch f32) vs mlx(q2 streaming) | cos<{thr} = divergence ===\n")
    check("embed_stream", "embed_stream.npy", "embed_stream.npy", sub="embed", layer=-1)
    for L in range(args.max_layer + 1):
        print(f"--- layer {L} ---")
        if L == 1:
            check(f"engram_L{L}_input(=prev out)", f"engram_L{L}_input.npy", f"engram_L{L}_input.npy", "engram_in", L)
            check(f"engram_L{L}_value", f"engram_L{L}_value.npy", f"engram_L{L}_value.npy", "engram_value", L)
            check(f"engram_L{L}_output", f"engram_L{L}_output.npy", f"engram_L{L}_output.npy", "engram_out", L)
        check(f"attn_L{L}_input", f"attn_L{L}_input.npy", f"attn_L{L}_input.npy", "attn_in", L)
        check(f"attn_L{L}_window_kv", f"attn_L{L}_window_kv.npy", f"attn_L{L}_window_kv.npy", "window_kv", L)
        if (NPY / f"attn_L{L}_compressed_kv.npy").exists():
            check(f"attn_L{L}_compressed_kv", f"attn_L{L}_compressed_kv.npy", f"attn_L{L}_compressed_kv.npy", "compressed_kv", L)
            check(f"attn_L{L}_index_k", f"attn_L{L}_index_k.npy", f"attn_L{L}_index_k.npy", "index_k", L)
        check(f"attn_L{L}_output", f"attn_L{L}_output.npy", f"attn_L{L}_output.npy", "attn_out", L)
        check(f"moe_L{L}_input", f"moe_L{L}_input.npy", f"moe_L{L}_input.npy", "moe_in", L)
        check(f"moe_L{L}_shared_out", f"moe_L{L}_shared_out.npy", f"moe_L{L}_shared_out.npy", "moe_shared", L)
        check(f"moe_L{L}_output", f"moe_L{L}_output.npy", f"moe_L{L}_output.npy", "moe_out", L)
        check(f"layer{L}_output", f"layer{L}_output.npy", f"layer{L}_output.npy", "layer_out", L)

    # ---- router top-6 agreement ----
    mlx = json.loads((RECEIPTS / "mlx_layers012.json").read_text())
    print("\n=== router top-6 agreement (ref golden vs mlx) ===")
    for L in range(args.max_layer + 1):
        rg = RECEIPTS / f"torchref_golden_moe_L{L}.json"
        if not rg.exists():
            continue
        ref_r = json.loads(rg.read_text())["router"]
        mlx_r = mlx["moe"][str(L)]["router"]
        ref_ids = ref_r["topk_ids"]
        mlx_ids = mlx_r["topk_ids"]
        n = min(len(ref_ids), len(mlx_ids))
        exact_set = sum(1 for i in range(n) if set(ref_ids[i]) == set(mlx_ids[i]))
        overlap = [len(set(ref_ids[i]) & set(mlx_ids[i])) for i in range(n)]
        mean_ov = float(np.mean(overlap))
        rw = np.array([ref_r["topk_weights"][i] for i in range(n)])
        # sort each token's (id,weight) by id to compare weights order-independently
        def by_id(ids, ws):
            return np.array([w for _, w in sorted(zip(ids, ws))])
        wdiff = []
        for i in range(n):
            if set(ref_ids[i]) == set(mlx_ids[i]):
                wr = by_id(ref_ids[i], ref_r["topk_weights"][i])
                wm = by_id(mlx_ids[i], mlx_r["topk_weights"][i])
                wdiff.append(float(np.abs(wr - wm).max()))
        wd = float(np.max(wdiff)) if wdiff else None
        print(f"layer {L}: exact top6 set match {exact_set}/{n} tokens; mean overlap {mean_ov:.2f}/6; "
              f"max|Δweight| (matched)={wd}")
        report["router"][str(L)] = {"exact_set_match": exact_set, "n_tokens": n,
                                    "mean_overlap": mean_ov, "max_weight_diff_matched": wd,
                                    "per_token_overlap": overlap}

    # ---- engram row-id agreement (layer 1) ----
    print("\n=== engram row-id agreement (layer 1) ===")
    rg = RECEIPTS / "torchref_golden_engram_L1.json"
    if rg.exists() and "1" in mlx.get("engram", {}):
        ref_rows = np.array(json.loads(rg.read_text())["row_ids"]["values"], dtype=np.int64)
        mlx_rows = np.array(mlx["engram"]["1"]["row_ids"]["values"], dtype=np.int64)
        if ref_rows.shape == mlx_rows.shape:
            eq = int((ref_rows == mlx_rows).sum())
            tot = int(ref_rows.size)
            print(f"row ids exact match: {eq}/{tot} ({100.0*eq/tot:.2f}%)  shape={list(ref_rows.shape)}")
            report["engram_row_ids"] = {"exact": eq, "total": tot, "shape": list(ref_rows.shape),
                                        "all_match": eq == tot}
            if eq != tot:
                mism = np.argwhere(ref_rows != mlx_rows)[:8]
                for idx in mism:
                    b, l, c = idx
                    print(f"  mismatch [tok {l}, col {c}]: ref {ref_rows[b,l,c]} vs mlx {mlx_rows[b,l,c]}")
        else:
            print(f"shape mismatch ref {ref_rows.shape} vs mlx {mlx_rows.shape}")

    report["first_divergence"] = first_div
    print("\n" + "=" * 70)
    if first_div is None:
        print(f"VERDICT: no sublayer cosine < {thr} through layer {args.max_layer} "
              f"(ref and mlx agree to that tolerance).")
    else:
        print(f"VERDICT: FIRST DIVERGENCE at {first_div['name']} "
              f"(sub={first_div['sub']}, layer={first_div['layer']}, min_cos={first_div['min_cos']:.6f}).")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
