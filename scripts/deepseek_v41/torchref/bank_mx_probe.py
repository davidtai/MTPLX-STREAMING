#!/usr/bin/env python3
"""W9 candidate-bank quality probe (MLX): for the FP4-source routed experts the
reference router selects at layers 0-2, requantize into each candidate format and
measure per-expert-weight cosine vs the source fp32, plus mxfp4 bit-exactness,
bytes/record and bank size (40x384).  Also validates that the numpy affine
quant used by bank_ladder.py (torch) matches mx.quantize, so the two agree.

Formats: q2/q3/q4/q6 gs64 affine, q4 gs32 affine, mxfp4 gs32.  CPU only.
Writes docs/deepseek-v41/receipts/bank_mx_probe.json.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx.core as mx
mx.set_default_device(mx.cpu)
import ref_forward as RF

REPO = RF.REPO
RECEIPTS = REPO / "docs" / "deepseek-v41" / "receipts"
UNION = json.loads(Path("/tmp/r0_selected_union.json").read_text())

# per-expert element counts (w1 gate, w3 up: [inter,dim]; w2 down: [dim,inter])
DIM, INTER = 5120, 2304
N_PER_EXPERT = 2 * INTER * DIM + DIM * INTER  # w1+w3+w2


def np_affine_quant_dequant(w, group_size, bits):
    """Reference affine quant->dequant matching mlx affine mode."""
    out, ind = w.shape
    g = w.reshape(out, ind // group_size, group_size)
    wmin = g.min(axis=2, keepdims=True)
    wmax = g.max(axis=2, keepdims=True)
    levels = (1 << bits) - 1
    scale = (wmax - wmin) / levels
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.round((g - wmin) / scale), 0, levels)
    deq = q * scale + wmin
    return deq.reshape(out, ind).astype(np.float32)


def cos(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def mx_quant_dequant(w_np, group_size, bits, mode):
    w = mx.array(w_np)
    q = mx.quantize(w, group_size=group_size, bits=bits, mode=mode)
    deq = mx.dequantize(*q, group_size=group_size, bits=bits, mode=mode)
    return np.array(deq.astype(mx.float32)), q


def bytes_per_record(bits, group_size, mode):
    n = N_PER_EXPERT
    if mode == "mxfp4":
        return n * bits / 8 + (n / group_size) * 1        # e8m0 1-byte scale, no bias
    return n * bits / 8 + (n / group_size) * 4            # bf16 scale + bf16 bias


def main():
    shards = RF.Shards(RF.SRC)
    formats = [
        ("q2_gs64", 2, 64, "affine"), ("q3_gs64", 3, 64, "affine"),
        ("q4_gs64", 4, 64, "affine"), ("q4_gs32", 4, 32, "affine"),
        ("q6_gs64", 6, 64, "affine"), ("mxfp4_gs32", 4, 32, "mxfp4"),
    ]
    result = {"n_pairs": sum(len(v) for v in UNION.values()), "formats": {}, "notes": {}}

    # ---- numpy-affine vs mx.quantize validation (one weight) ----
    L0e = UNION["0"][0]
    w0 = shards.dequant_weight(f"layers.0.ffn.experts.{L0e}.w1.weight").numpy()
    mxq, _ = mx_quant_dequant(w0, 64, 4, "affine")
    npq = np_affine_quant_dequant(w0, 64, 4)
    result["notes"]["numpy_vs_mx_affine_q4gs64_maxabsdiff"] = float(np.abs(mxq - npq).max())
    result["notes"]["numpy_vs_mx_affine_q4gs64_cos"] = cos(mxq, npq)

    # ---- per-format per-expert quality vs source ----
    t0 = time.time()
    for fname, bits, gs, mode in formats:
        cbits, cbank = [], []
        mxfp4_exact = True
        n = 0
        for L, ids in UNION.items():
            for eid in ids:
                base = f"layers.{L}.ffn.experts.{eid}"
                for w in ("w1", "w2", "w3"):
                    src = shards.dequant_weight(f"{base}.{w}.weight").numpy()
                    try:
                        deq, q = mx_quant_dequant(src, gs, bits, mode)
                    except Exception as exc:
                        result["formats"][fname] = {"error": repr(exc)[:200]}
                        deq = None
                        break
                    cbits.append(cos(deq, src))
                    if mode == "mxfp4":
                        mxfp4_exact = mxfp4_exact and bool(np.array_equal(deq, src))
                    n += 1
                if fname in result["formats"] and "error" in result["formats"][fname]:
                    break
            if fname in result["formats"] and "error" in result["formats"][fname]:
                break
        if fname in result["formats"] and "error" in result["formats"][fname]:
            print(f"[probe] {fname}: {result['formats'][fname]['error']}")
            continue
        bpr = bytes_per_record(bits, gs, mode)
        result["formats"][fname] = {
            "bits": bits, "group_size": gs, "mode": mode,
            "mean_cos_vs_source": float(np.mean(cbits)), "min_cos_vs_source": float(np.min(cbits)),
            "n_weights": n, "bytes_per_record": int(bpr),
            "bank_bytes_40x384": int(bpr * 40 * 384),
            "bank_GiB_40x384": round(bpr * 40 * 384 / (1024 ** 3), 2),
        }
        if mode == "mxfp4":
            result["formats"][fname]["bit_exact_vs_source"] = mxfp4_exact
        print(f"[probe] {fname:12} mean_cos_vs_source={np.mean(cbits):.6f} "
              f"bytes/rec={int(bpr):>9} bank={bpr*40*384/(1024**3):.1f}GiB"
              + (f" mxfp4_bit_exact={mxfp4_exact}" if mode == "mxfp4" else ""))
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    (RECEIPTS / "bank_mx_probe.json").write_text(json.dumps(result, indent=2))
    print(f"[probe] numpy==mx affine(q4gs64): maxdiff={result['notes']['numpy_vs_mx_affine_q4gs64_maxabsdiff']:.3g} "
          f"cos={result['notes']['numpy_vs_mx_affine_q4gs64_cos']:.7f}")
    print(f"[probe] DONE {time.time()-t0:.1f}s -> bank_mx_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
