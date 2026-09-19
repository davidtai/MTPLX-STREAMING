#!/usr/bin/env python3
"""Isolate the layer-0 window-KV divergence: is the MLX code faithful to its OWN
q8 weights (=> the ref gap is pure quantization), or does it diverge even at equal
weights (=> a structural bug)?

Feeds the layer-0 attention the reference's captured post-hc-norm input x
(.benchmark-artifacts/.../attn_L0_input.npy, which matched MLX to cos 0.99994) and
compares, all in MLX:
  * kv_q8      = kv_norm(wkv_q8(x)) then RoPE   (the model's own path)
  * kv_f32wt   = kv_norm((x @ dequant(wkv_q8).T)) then RoPE  (same code, weights pre-dequantized)
  * the ref oracle window_kv (torch f32 weights) loaded from npy
So: cos(kv_q8, kv_f32wt) isolates the q8 *weight* error under identical code;
cos(kv_q8, oracle) is the total gap; cos(kv_f32wt, oracle) is the residual code+source-quant gap.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx_dump  # noqa: F401  (runs _force_worktree_mtplx + mx cpu)
import mlx.core as mx

REPO = Path(__file__).resolve().parents[3]
NPY = REPO / ".benchmark-artifacts" / "deepseek-v41" / "w9"
GIB = 1024 ** 3
MODEL = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4").expanduser()


def cos(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(a.shape[1], -1)
    b = np.asarray(b, dtype=np.float64).reshape(b.shape[1], -1)
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30
    return float((num / den).min()), float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming
    from mtplx.models import deepseek_v41 as M

    resident = load_deepseek_v41_streaming(
        MODEL, memory_limit_bytes=int(100 * GIB), max_live_kv_tokens=4096, admit=True,
        admission_receipt=None, expert_cache_limit_bytes=int(2 * GIB), apply_memory_cap=False,
        slot_layout="component-banks", cache_scope="layer", island_layers=(), verify_record_hashes=False)
    model = resident.model
    attn = model.model.layers[0].attn

    x_np = np.load(NPY / "attn_L0_input.npy")            # [1,31,5120] f32 (ref = mlx to 0.99994)
    x = mx.array(x_np)
    positions = mx.arange(0, x.shape[1])
    qcos, qsin = M._cos_sin(attn.inv_freq, positions)

    # q8 path (the model's own wkv)
    kv_q8 = M._rmsnorm(attn.wkv(x), attn.kv_norm_weight, attn.eps)
    kv_q8 = M._rope_last(kv_q8, qcos, qsin)

    # f32-weight path: dequantize the SAME q8 weight, plain matmul, same norm+rope
    wkv = attn.wkv
    if hasattr(wkv, "scales"):
        w = mx.dequantize(wkv.weight, wkv.scales, wkv.biases, group_size=wkv.group_size, bits=wkv.bits)
        wq_kind = f"QuantizedLinear bits={wkv.bits} gs={wkv.group_size}"
    else:
        w = wkv.weight
        wq_kind = "dense (unexpected)"
    kv_f32 = M._rmsnorm(x @ w.T, attn.kv_norm_weight, attn.eps)
    kv_f32 = M._rope_last(kv_f32, qcos, qsin)

    mx.eval(kv_q8, kv_f32)
    kv_q8_np = np.array(kv_q8.astype(mx.float32))
    kv_f32_np = np.array(kv_f32.astype(mx.float32))
    oracle = np.load(NPY / "attn_L0_window_kv.npy")      # torch f32-weight window_kv

    print(f"wkv module: {wq_kind}")
    print(f"cos(kv_q8, kv_f32wt)  [q8-weight error, identical code]  min={cos(kv_q8_np, kv_f32_np)[0]:.6f} global={cos(kv_q8_np, kv_f32_np)[1]:.6f}")
    print(f"cos(kv_f32wt, oracle) [code+source-quant, no q8]         min={cos(kv_f32_np, oracle)[0]:.6f} global={cos(kv_f32_np, oracle)[1]:.6f}")
    print(f"cos(kv_q8, oracle)    [total window_kv gap]              min={cos(kv_q8_np, oracle)[0]:.6f} global={cos(kv_q8_np, oracle)[1]:.6f}")
    # also the wkv q8 weight vs a plain matmul consistency and dynamic range
    print(f"wkv q8 weight: shape {tuple(w.shape)}  |max| {float(mx.max(mx.abs(w))):.4g}")

    # --- amplification check: run MLX's OWN captured input through the same code ---
    xm_np = np.load(NPY / "mlx" / "attn_L0_input.npy")
    print(f"\ncos(mlx_input, oracle_input) = min {cos(xm_np, x_np)[0]:.6f} global {cos(xm_np, x_np)[1]:.6f}")
    xm = mx.array(xm_np)
    kv_from_mlx_in = M._rope_last(M._rmsnorm(attn.wkv(xm), attn.kv_norm_weight, attn.eps), qcos, qsin)
    mx.eval(kv_from_mlx_in)
    kv_from_mlx_in_np = np.array(kv_from_mlx_in.astype(mx.float32))
    print(f"cos(kv[mlx_in], oracle_window_kv)   min {cos(kv_from_mlx_in_np, oracle)[0]:.6f} global {cos(kv_from_mlx_in_np, oracle)[1]:.6f}")
    print(f"cos(kv[mlx_in], kv[oracle_in])      min {cos(kv_from_mlx_in_np, kv_q8_np)[0]:.6f} global {cos(kv_from_mlx_in_np, kv_q8_np)[1]:.6f}")
    full_mlx_wkv = np.load(NPY / "mlx" / "attn_L0_window_kv.npy")
    print(f"cos(kv[mlx_in], full_run_mlx_wkv)   min {cos(kv_from_mlx_in_np, full_mlx_wkv)[0]:.6f} global {cos(kv_from_mlx_in_np, full_mlx_wkv)[1]:.6f}  (should be ~1: same computation)")
    # per-position norms of the pre-norm wkv output, to see if amplification is at low-norm positions
    pre = np.array((xm @ w.T).astype(mx.float32))[0]  # [31,512]
    norms = np.linalg.norm(pre, axis=1)
    print(f"pre-kvnorm per-pos L2 norm: min {norms.min():.4g} @pos{int(norms.argmin())}  max {norms.max():.4g}  median {np.median(norms):.4g}")

    # --- dtype hypothesis: run the SAME code with a bf16-cast input (model's native dtype) ---
    xb = xm.astype(mx.bfloat16)
    kv_bf16 = M._rope_last(M._rmsnorm(attn.wkv(xb), attn.kv_norm_weight, attn.eps), qcos, qsin)
    mx.eval(kv_bf16)
    kv_bf16_np = np.array(kv_bf16.astype(mx.float32))
    print(f"\n[dtype] kv output dtype fp32-input={kv_from_mlx_in.dtype}  bf16-input={kv_bf16.dtype}")
    print(f"[dtype] cos(kv[bf16 in], full_run_mlx_wkv)  min {cos(kv_bf16_np, full_mlx_wkv)[0]:.6f} global {cos(kv_bf16_np, full_mlx_wkv)[1]:.6f}  (if ~1: full run stored bf16)")
    print(f"[dtype] cos(kv[bf16 in], oracle_window_kv)  min {cos(kv_bf16_np, oracle)[0]:.6f} global {cos(kv_bf16_np, oracle)[1]:.6f}")
    try:
        resident._mtplx_expert_runtime.close()
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
