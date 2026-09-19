"""W101 / K36: fused decode/verify attention PROJECTION-CHAIN kernels
(``MTPLX_DSV41_ATTN_FUSED_PROJ``; docs/deepseek-v41/W101_ATTN_FUSED_PROJ.md).

W99 §8.5's verdict: after the wo_a cache, lean casts and the K29 core, a decode
attention layer's remaining kernels are dominated by the qkv/out PROJECTION CHAINS
-- the rmsnorm + interleaved-RoPE + head/group layout GLUE between the (kept)
quantized matmuls and the (kept) grouped o-LoRA down-projection.  W101 fuses each
glue region into ONE ``mx.fast.metal_kernel`` and replaces the o-LoRA einsum's
per-token weight Transpose (window-40: out_proj ~4.9 ms/layer, ~25x its bandwidth)
with a batched ``matmul`` over a pre-transposed cached bf16 weight.

ROUNDING-CLASS, GPU-only, small-M.  These CPU-pinned tests prove:
  1. the flag parses (default OFF) and the GPU/small-M gate returns False on CPU;
  2. the fused path FALLS BACK to eager on CPU -> byte-identical to lever-off over
     64 decode steps (the numerics on Metal are the GPU test's job);
  3. the engagement counter is wired (0 fused calls on CPU -> fell back);
  4. the census: the fused qkv/out chains issue fewer non-view kernels than the
     K22-compiled tape and the eager chain (the dispatch collapse);
  5. the fused wo_a weight is TRANSPOSED once + cached (no per-token re-layout);
  6. the kernels module's pure-MLX references match the eager helpers.

Tiny dims, CPU-pinned (worker-tests-must-pin-mlx-cpu.md); no artifact, no GPU.
"""

from __future__ import annotations

import importlib.util
import io
import re
from collections import Counter
from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.cpu)

import numpy as np
import pytest

from mtplx.models import deepseek_v41 as dsv41
from mtplx.models import deepseek_v41_fused_proj_kernels as fp

_REPO = Path(__file__).resolve().parents[1]
_RECT = re.compile(r'\[label ="([^"]+)", shape=rectangle\]')
_VIEW = {"Reshape", "Flatten", "Unflatten", "ExpandDims", "Broadcast",
         "Transpose", "Squeeze", "Slice", "StopGradient", "Arange"}


def _load_bisect():
    spec = importlib.util.spec_from_file_location(
        "w101_bisect", _REPO / "scripts" / "deepseek_v41" / "metal_decode_attn_bisect.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ops(*outs):
    buf = io.StringIO()
    mx.export_to_dot(buf, *[a for a in outs if isinstance(a, mx.array)])
    return Counter(_RECT.findall(buf.getvalue()))


def _kern(counter):
    return sum(v for k, v in counter.items() if k not in _VIEW)


def _decode(model, ops, prompt_ids, steps):
    cache = model.make_cache()
    logits = model(ops.input([list(prompt_ids)]), cache=cache)
    ops.sync(logits)
    tok = ops.argmax_last(logits)
    ids = [tok]
    for _ in range(steps):
        logits = model(ops.input([[tok]]), cache=cache)
        ops.sync(logits)
        tok = ops.argmax_last(logits)
        ids.append(tok)
    return ids


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(dsv41._ATTN_FUSED_PROJ_ENV, raising=False)
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    fp.reset_engagement()
    yield


# --------------------------------------------------------------------------
# 1. flag parsing + GPU/small-M gate
# --------------------------------------------------------------------------
def test_fused_proj_off_by_default():
    assert dsv41._resolve_attn_fused_proj(raw="") is False
    assert dsv41._resolve_attn_fused_proj(raw="0") is False
    assert dsv41._resolve_attn_fused_proj(raw="off") is False
    assert dsv41._resolve_attn_fused_proj(raw="none") is False
    assert dsv41._resolve_attn_fused_proj(raw="1") is True
    assert dsv41._resolve_attn_fused_proj(raw="true") is True
    assert dsv41._resolve_attn_fused_proj(raw="on") is True
    with pytest.raises(ValueError):
        dsv41._resolve_attn_fused_proj(raw="banana")


def test_fused_proj_gate_false_on_cpu(monkeypatch):
    """The gate is False on a CPU-pinned host even when armed (no Metal default
    device) -> the eager fallback runs and no Metal is dispatched."""
    monkeypatch.setenv(dsv41._ATTN_FUSED_PROJ_ENV, "1")
    assert dsv41._fused_proj_use(1) is False       # armed but CPU device
    assert dsv41._fused_proj_use(8) is False
    monkeypatch.setenv(dsv41._ATTN_FUSED_PROJ_ENV, "0")
    assert dsv41._fused_proj_use(1) is False       # off


def test_fused_proj_small_m_cap():
    """The gate mirrors the K29 small-M cap (b*s <= 8): decode + depth-<=7 verify."""
    assert dsv41._DECODE_ATTN_KERNEL_MAX_ROWS == 8


# --------------------------------------------------------------------------
# 2. CPU fallback -> byte-identical to lever-off over 64 decode steps
# --------------------------------------------------------------------------
@pytest.mark.parametrize("compile_on", [False, True])
def test_fused_proj_fallback_byte_identical_64_steps(monkeypatch, compile_on):
    """On CPU the fused path is gated OFF (no Metal), so the eager (or K22-tape)
    chain runs -> lever ON must be byte-identical to lever OFF over 64 steps."""
    bis = _load_bisect()
    monkeypatch.setattr(dsv41, "_ATTN_COMPILE", bool(compile_on))
    model, _args = bis._build_tiny_full_model(seed=5)
    ops = bis._TinyOps()
    prompt_ids = list(range(1, 41))

    monkeypatch.setenv(dsv41._ATTN_FUSED_PROJ_ENV, "0")
    ids_off = _decode(model, ops, prompt_ids, 64)
    monkeypatch.setenv(dsv41._ATTN_FUSED_PROJ_ENV, "1")
    ids_on = _decode(model, ops, prompt_ids, 64)
    assert ids_on == ids_off, "fused-proj must fall back to eager on CPU (byte-identical)"


# --------------------------------------------------------------------------
# 3. engagement counter wiring
# --------------------------------------------------------------------------
def test_engagement_counter_api():
    fp.reset_engagement()
    assert fp.engagement() == {
        "qkv_calls": 0, "out_calls": 0, "rows": 0,
        "qkv_fallbacks": 0, "out_fallbacks": 0,
    }
    fp.note_qkv(5)
    fp.note_qkv(1)
    fp.note_out()
    fp.note_fallback("qkv")
    fp.note_fallback("out")
    e = fp.engagement()
    assert e["qkv_calls"] == 2 and e["rows"] == 6 and e["out_calls"] == 1
    assert e["qkv_fallbacks"] == 1 and e["out_fallbacks"] == 1
    fp.reset_engagement()
    assert fp.engagement()["qkv_calls"] == 0


def test_engagement_zero_after_cpu_decode(monkeypatch):
    """A decode with the lever ON on CPU records ZERO fused calls (it fell back) --
    the receipt can tell 'fused did not run' from 'ran (slowly)'."""
    bis = _load_bisect()
    model, _args = bis._build_tiny_full_model(seed=7)
    ops = bis._TinyOps()
    monkeypatch.setenv(dsv41._ATTN_FUSED_PROJ_ENV, "1")
    fp.reset_engagement()
    _decode(model, ops, list(range(1, 41)), 4)
    e = fp.engagement()
    assert e["qkv_calls"] == 0 and e["out_calls"] == 0, (
        f"CPU decode must not dispatch the fused kernels, got {e}"
    )


# --------------------------------------------------------------------------
# 4. census -- the fused chains issue fewer non-view kernels
# --------------------------------------------------------------------------
def _regions(seed=1):
    from mtplx.models.deepseek_v41 import (
        _rmsnorm, _rope_last, _cos_sin, _lin_arrays, _attn_qkv_prep, _attn_out_prep,
    )
    from mlx.utils import tree_flatten, tree_unflatten
    from mtplx.models.deepseek_v41 import Model, ModelArgs
    args = ModelArgs(
        vocab_size=48, hidden_size=32, num_hidden_layers=8, num_attention_heads=4,
        head_dim=16, qk_rope_head_dim=4, q_lora_rank=16, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=2,
        index_n_heads=2, index_head_dim=8, index_topk=5, sliding_window=8,
        window_size=8, swiglu_limit=0.5, compress_ratios=[0, 0, 2, 2, 2, 1, 1, 1],
        kv_source_layer_ids=[2, 5], index_source_layer_ids=[2, 5, 6],
        candidate_source_layer_id=5, candidate_topk_blocks=3, candidate_block_size=2)
    model = Model(args)
    mx.random.seed(seed)
    new = []
    for name, arr in tree_flatten(model.parameters()):
        if arr.ndim == 1 and ("norm_weight" in name or name.endswith("norm.weight")):
            v = 1.0 + 0.2 * mx.random.normal(arr.shape)
        elif "attn_sink" in name:
            v = 0.5 * mx.random.normal(arr.shape)
        else:
            v = 0.1 * mx.random.normal(arr.shape)
        new.append((name, v.astype(mx.float32)))
    model.update(tree_unflatten(new))
    mx.eval(model.parameters())
    attn = model.model.layers[6].attn
    b, s, H, hd = 1, 1, attn.n_heads, attn.head_dim
    x = 0.1 * mx.random.normal((1, 1, attn.dim))
    qcos, qsin = _cos_sin(attn.inv_freq, mx.array([5]))
    o = 0.1 * mx.random.normal((b, s, H, hd))
    mx.eval(x, qcos, qsin, o)
    # eager qkv/out
    qr = _rmsnorm(attn.wq_a(x), attn.q_norm_weight, attn.eps)
    q = _rope_last(attn.wq_b(qr).reshape(b, s, H, hd), qcos, qsin)
    kv = _rope_last(_rmsnorm(attn.wkv(x), attn.kv_norm_weight, attn.eps), qcos, qsin)
    qkv_e = _kern(_ops(q, qr, kv))
    oo = _rope_last(o, qcos, qsin, inverse=True).reshape(b, s, attn.n_groups, -1)
    out_e = _kern(_ops(attn.wo_b(attn._o_lora_down(oo).reshape(b, s, -1))))
    # K22-compiled qkv/out
    of = dsv41._ATTN_COMPILE
    dsv41._ATTN_COMPILE = True
    dsv41._ATTN_COMPILED.clear()
    try:
        qc = _attn_qkv_prep(attn)(
            x, qcos, qsin, attn.q_norm_weight, attn.kv_norm_weight,
            *_lin_arrays(attn.wq_a), *_lin_arrays(attn.wq_b), *_lin_arrays(attn.wkv))
        qkv_k = _kern(_ops(*qc))
        oc = _attn_out_prep(attn)(o, qcos, qsin, attn._o_lora_dense_weight(),
                                  *_lin_arrays(attn.wo_b))
        out_k = _kern(_ops(oc))
    finally:
        dsv41._ATTN_COMPILE = of
        dsv41._ATTN_COMPILED.clear()
    # fused qkv/out (traced, not eval'd -- Custom nodes on CPU)
    qkv_f = _kern(_ops(*attn._qkv_prep_fused(x, qcos, qsin, b, s, H, hd)))
    out_f = _kern(_ops(attn._out_prep_fused(o, qcos, qsin, b, s)))
    return dict(qkv_e=qkv_e, qkv_k=qkv_k, qkv_f=qkv_f,
                out_e=out_e, out_k=out_k, out_f=out_f)


def test_census_fused_chains_fewer_kernels():
    r = _regions()
    # fused < K22-compiled < eager, for both chains
    assert r["qkv_f"] < r["qkv_k"] < r["qkv_e"], r
    assert r["out_f"] < r["out_k"] < r["out_e"], r
    # the fused chains are exactly the kept matmuls + the fused metal kernels:
    # qkv = 3 quantized/dense matmuls + 3 CustomKernel = 6; out = 2 matmul + 1 = 3.
    assert r["qkv_f"] == 6, f"expected 6 fused qkv kernels (3 matmul + 3 custom), got {r['qkv_f']}"
    assert r["out_f"] == 3, f"expected 3 fused out kernels (2 matmul + 1 custom), got {r['out_f']}"


def test_census_fused_qkv_has_three_custom_kernels():
    """The qkv-prep glue is exactly three fused metal_kernels (rmsnorm, q rope,
    kv rmsnorm+rope); out-prep is one (o de-rope + layout)."""
    from mtplx.models.deepseek_v41 import _cos_sin
    bis = _load_bisect()
    args = bis.real_args()
    attn = bis._build_layer(args, "swa_only")
    b, s, H, hd = 1, 1, attn.n_heads, attn.head_dim
    x = (mx.random.normal((b, s, attn.dim)) * 0.02).astype(mx.bfloat16)
    qcos, qsin = _cos_sin(attn.inv_freq, mx.array([37]))
    o = (mx.random.normal((b, s, H, hd)) * 0.02).astype(mx.bfloat16)
    mx.eval(x, qcos, qsin, o)
    qkv = _ops(*attn._qkv_prep_fused(x, qcos, qsin, b, s, H, hd))
    out = _ops(attn._out_prep_fused(o, qcos, qsin, b, s))
    assert qkv["CustomKernel"] == 3, f"qkv-prep must be 3 fused kernels, got {dict(qkv)}"
    assert out["CustomKernel"] == 1, f"out-prep must be 1 fused kernel, got {dict(out)}"
    # the grouped o-LoRA is a matmul (+wo_b), NOT re-transposing the weight per call:
    # only the small ``o`` transposes remain (the big weight is cached pre-transposed).
    assert out["Matmul"] == 2, f"out-prep must keep 2 matmuls (o-LoRA + wo_b), got {dict(out)}"


# --------------------------------------------------------------------------
# 5. fused wo_a weight is transposed once + cached (no per-token re-layout)
# --------------------------------------------------------------------------
def test_o_lora_fused_weight_transposed_and_cached():
    bis = _load_bisect()
    args = bis.real_args()
    attn = bis._build_layer(args, "swa_only")   # bf16 dense wo_a (tiny) -> not quantized
    w1 = attn._o_lora_fused_weight()
    w2 = attn._o_lora_fused_weight()
    assert w1 is w2, "fused wo_a weight must be cached (identity keyed on packed weight)"
    g, in_per_group, r = w1.shape
    assert g == attn.n_groups and r == attn.o_lora_rank, (
        f"fused wo_a must be [g, in_per_group, o_lora_rank]; got {w1.shape}")
    assert in_per_group == attn.n_heads * attn.head_dim // attn.n_groups
    # values equal the eager [g, r, in] weight transposed to [g, in, r]
    eager = attn._o_lora_dense_weight()          # [g, r, in]
    mx.eval(w1, eager)
    assert bool(mx.all(w1 == mx.swapaxes(eager, 1, 2)).item()), (
        "fused wo_a values must equal the eager weight transposed to [g, in, r]")


def test_o_lora_fused_weight_quantized_is_bf16(monkeypatch):
    """When wo_a is quantized-resident the fused weight is dequantized to bf16 (the
    reference einsum dtype), transposed once -- half the f32 cache's memory."""
    import mlx.nn as nn
    bis = _load_bisect()
    args = bis.real_args()
    attn = bis._build_layer(args, "swa_only")
    # quantize wo_a to q8 (affine) like the resident codec
    attn.wo_a = nn.QuantizedLinear.from_linear(attn.wo_a, group_size=64, bits=8)
    mx.eval(attn.wo_a.parameters())
    w = attn._o_lora_fused_weight()
    assert w.dtype == mx.bfloat16, f"quantized fused wo_a must be bf16, got {w.dtype}"
    assert w.shape == (attn.n_groups, attn.n_heads * attn.head_dim // attn.n_groups,
                       attn.o_lora_rank)


# --------------------------------------------------------------------------
# 6. kernels module -- pure-MLX references match the eager helpers; build/trace
# --------------------------------------------------------------------------
def test_kernel_references_match_eager_helpers():
    from mtplx.models.deepseek_v41 import _rmsnorm, _rope_last, _cos_sin
    mx.random.seed(3)
    H, hd, rd, qlr = 4, 16, 4, 20
    inv = 1.0 / (10000.0 ** (mx.arange(0, rd, 2, dtype=mx.float32) / rd))
    cos, sin = _cos_sin(inv, mx.array([4, 5, 6]))
    mx.eval(cos, sin)
    for dt in (mx.float32, mx.bfloat16):
        x = (mx.random.normal((1, 3, qlr)) * 0.5).astype(dt)
        w = (1.0 + 0.2 * mx.random.normal((qlr,))).astype(dt)
        q = (mx.random.normal((1, 3, H, hd)) * 0.5).astype(dt)
        mx.eval(x, w, q)
        assert bool(mx.array_equal(_rmsnorm(x, w, 1e-6),
                                   fp.rmsnorm_reference(x, w, 1e-6)).item())
        assert bool(mx.array_equal(_rope_last(q, cos, sin),
                                   fp.rope_heads_reference(q, cos, sin)).item())
        assert bool(mx.array_equal(_rope_last(q, cos, sin, inverse=True),
                                   fp.rope_heads_reference(q, cos, sin, inverse=True)).item())


def test_kernels_available_and_trace_single_custom():
    """Each fused kernel traces to exactly ONE CustomKernel (one dispatch); building
    the kernel object needs Metal available (True on this box) but no dispatch."""
    from mtplx.models.deepseek_v41 import _cos_sin
    assert fp.kernel_available() is True
    inv = 1.0 / (10000.0 ** (mx.arange(0, 4, 2, dtype=mx.float32) / 4))
    cos, sin = _cos_sin(inv, mx.array([4, 5, 6]))
    x = mx.zeros((1, 3, 20), dtype=mx.bfloat16)
    w = mx.ones((20,), dtype=mx.bfloat16)
    kv = mx.zeros((1, 3, 16), dtype=mx.bfloat16)
    wkv = mx.ones((16,), dtype=mx.bfloat16)
    q = mx.zeros((1, 3, 4, 16), dtype=mx.bfloat16)
    mx.eval(cos, sin, x, w, kv, wkv, q)
    assert _ops(fp.rmsnorm(x, w, 1e-6))["CustomKernel"] == 1
    assert _ops(fp.rmsnorm_rope(kv, wkv, 1e-6, cos, sin))["CustomKernel"] == 1
    assert _ops(fp.rope_heads(q, cos, sin))["CustomKernel"] == 1
    assert _ops(fp.rope_heads(q, cos, sin, inverse=True))["CustomKernel"] == 1
