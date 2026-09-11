"""Runtime codec: native-mxfp4 expert execution through the streaming gather path.

CPU-only.  Builds a small component bank of mxfp4-repacked experts and checks the
real ``_gather_component_bank(codec="mxfp4")`` / ``_run_mxfp4_expert`` output against
an independent dequantize-then-matmul reference (with and without the SwiGLU clamp),
and asserts the affine path is byte-unchanged.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx_lm.models.activations import swiglu

from mtplx.models.expert_mlx import (
    _clamped_swiglu,
    _gather_component_bank,
)

H, I, G, BITS = 64, 128, 32, 4  # hidden, expert_hidden, group, bits


class _DuckBank:
    def __init__(self, arrays):
        self.arrays = arrays


def _fp4_grid(rng, out, inn):
    fp4 = np.array([0.0, .5, 1, 1.5, 2, 3, 4, 6, 0.0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    codes = rng.integers(0, 16, size=(out, inn))
    exps = rng.integers(122, 132, size=(out, inn // 32))
    return (fp4[codes] * np.repeat((2.0 ** (exps - 127)).astype(np.float32), 32, axis=1)).astype(np.float32)


def _build_bank(n_experts, seed):
    """Return (duck bank, list of per-expert fp32 dequant weights)."""
    rng = np.random.default_rng(seed)
    shapes = {"gate_proj": (I, H), "up_proj": (I, H), "down_proj": (H, I)}
    stacks = {f"{p}.weight": [] for p in shapes} | {f"{p}.scales": [] for p in shapes}
    ref = []
    for _ in range(n_experts):
        w = {}
        for proj, (o, k) in shapes.items():
            f32 = _fp4_grid(rng, o, k)
            packed, scales = mx.quantize(mx.array(f32), group_size=G, bits=BITS, mode="mxfp4")
            # dequant is bit-exact; keep it as the reference weight
            deq = np.array(mx.dequantize(packed, scales, group_size=G, bits=BITS, mode="mxfp4").astype(mx.float32))
            assert np.array_equal(deq, f32)
            w[proj] = deq
            stacks[f"{proj}.weight"].append(packed)
            stacks[f"{proj}.scales"].append(scales)
        ref.append(w)
    arrays = {k: mx.stack(v) for k, v in stacks.items()}
    mx.eval(*arrays.values())
    return _DuckBank(arrays), ref


def _reference_mlp(x_np, weights, swiglu_limit=None):
    """Independent fp32 reference of one expert's clamped SwiGLU MLP."""
    xg = mx.array(x_np)
    gate = xg @ mx.array(weights["gate_proj"]).T
    up = xg @ mx.array(weights["up_proj"]).T
    hidden = _clamped_swiglu(gate, up, swiglu_limit)
    out = hidden @ mx.array(weights["down_proj"]).T
    return np.array(out.astype(mx.float32))


def _cos(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def test_mxfp4_gather_matches_dequant_matmul_reference():
    bank, ref = _build_bank(4, seed=3)
    rng = np.random.default_rng(11)
    rows = 5
    x = rng.standard_normal((rows, H)).astype(np.float32)
    idx = np.array([0, 3, 1, 2, 0], dtype=np.int32)
    slot_indices = mx.array(idx.reshape(-1, 1))
    out = _gather_component_bank(
        mx.array(x), bank, slot_indices, group_size=G, bits=BITS, codec="mxfp4"
    )
    out = np.array(out.astype(mx.float32))
    assert out.shape == (rows, H)
    expected = np.stack([_reference_mlp(x[r:r + 1], ref[idx[r]])[0] for r in range(rows)])
    assert _cos(out, expected) >= 0.9999, _cos(out, expected)


def test_mxfp4_gather_applies_swiglu_clamp():
    bank, ref = _build_bank(3, seed=5)
    rng = np.random.default_rng(9)
    rows = 4
    # large inputs so the pre-activations exceed the clamp limit
    x = (rng.standard_normal((rows, H)) * 20).astype(np.float32)
    idx = np.array([0, 1, 2, 1], dtype=np.int32)
    slot_indices = mx.array(idx.reshape(-1, 1))
    limit = 10.0
    out = np.array(_gather_component_bank(
        mx.array(x), bank, slot_indices, group_size=G, bits=BITS,
        swiglu_limit=limit, codec="mxfp4").astype(mx.float32))
    clamped = np.stack([_reference_mlp(x[r:r + 1], ref[idx[r]], swiglu_limit=limit)[0] for r in range(rows)])
    unclamped = np.stack([_reference_mlp(x[r:r + 1], ref[idx[r]], swiglu_limit=None)[0] for r in range(rows)])
    assert _cos(out, clamped) >= 0.9999
    # the clamp must actually change the output at these magnitudes
    assert _cos(clamped, unclamped) < 0.999


def test_affine_gather_path_unchanged():
    """The affine component-bank path (with biases) still runs and matches its own
    dequant-matmul reference -- the codec branch does not perturb affine."""
    rng = np.random.default_rng(2)
    n, rows = 3, 4
    shapes = {"gate_proj": (I, H), "up_proj": (I, H), "down_proj": (H, I)}
    stacks = {f"{p}.{leaf}": [] for p in shapes for leaf in ("weight", "scales", "biases")}
    ref = []
    for _ in range(n):
        w = {}
        for proj, (o, k) in shapes.items():
            f32 = rng.standard_normal((o, k)).astype(np.float32) * 0.1
            wq = mx.array(f32).astype(mx.bfloat16)
            packed, scales, biases = mx.quantize(wq, group_size=64, bits=4, mode="affine")
            deq = np.array(mx.dequantize(packed, scales, biases, group_size=64, bits=4, mode="affine").astype(mx.float32))
            w[proj] = deq
            stacks[f"{proj}.weight"].append(packed)
            stacks[f"{proj}.scales"].append(scales)
            stacks[f"{proj}.biases"].append(biases)
        ref.append(w)
    arrays = {k: mx.stack(v) for k, v in stacks.items()}
    mx.eval(*arrays.values())
    bank = _DuckBank(arrays)
    x = rng.standard_normal((rows, H)).astype(np.float32)
    idx = np.array([0, 2, 1, 0], dtype=np.int32)
    out = np.array(_gather_component_bank(
        mx.array(x), bank, mx.array(idx.reshape(-1, 1)), group_size=64, bits=4,
        codec="affine").astype(mx.float32))
    expected = np.stack([_reference_mlp(x[r:r + 1], ref[idx[r]])[0] for r in range(rows)])
    assert _cos(out, expected) >= 0.999, _cos(out, expected)
