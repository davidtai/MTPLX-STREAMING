"""W97: byte-identity + dispatch-cut proof for the ``MTPLX_DSV41_ATTN_WO_A_CACHE``
lever (docs/deepseek-v41/W97_ATTENTION_291MS.md).

The grouped o-LoRA ``wo_a`` down-projection is applied as an einsum, so
``Attention._o_lora_dense_weight`` calls ``mx.dequantize(wo_a)`` to materialise the
dense weight.  Off (default) that dequantize is re-issued **every layer every decode
token**; on, it is computed once and cached (keyed on the packed-weight identity).

This proves:
  1. byte-identity -- 64 decode steps through the production ``Attention._attend``
     produce bit-for-bit identical output with the lever OFF vs ON, both eager and
     under the K22 attention compile tape;
  2. the cut -- with the lever OFF ``mx.dequantize`` is invoked once per decode step
     (per layer), with it ON exactly once for the whole run;
  3. the cache keys on the packed-weight identity -- re-quantising ``wo_a`` (a new
     weight array) rebuilds the cached dense weight.

Tiny dims, CPU-pinned (worker-tests-must-pin-mlx-cpu.md), well under the memory
guard; no artifact, no GPU.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)

import pytest

from mtplx.models import deepseek_v41 as dsv41
from mtplx.models.deepseek_v41 import Attention, ModelArgs
from mtplx.models.deepseek_v41_cache import LayerAttentionCache, SharedAttentionRuntime


# --------------------------------------------------------------------------- #
# tiny swa_only layer (compress_ratio 0 -> no compressor/indexer): the simplest
# path that still exercises the full q/kv projections, the window attention and
# the grouped o-LoRA out-projection (wo_a einsum + wo_b).
# --------------------------------------------------------------------------- #
def _tiny_args() -> ModelArgs:
    return ModelArgs(
        num_hidden_layers=4,
        hidden_size=256,
        num_attention_heads=8,
        head_dim=64,
        qk_rope_head_dim=16,
        q_lora_rank=128,
        o_lora_rank=64,
        o_groups=4,          # in_per_group = 8*64/4 = 128 (gs 64 aligned)
        sliding_window=16,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=8,
        compress_ratios=[0, 0, 0, 0],   # layer 0 -> swa_only
        candidate_source_layer_id=-1,
    )


def _build_quantized_layer(seed: int = 0, bits: int = 8, group_size: int = 64,
                            mode: str = "affine") -> Attention:
    """A single swa_only ``Attention`` with random weights, then ``wo_a`` (and the
    other group-aligned Linears) quantised exactly as the resident loader does."""
    mx.random.seed(seed)
    args = _tiny_args()
    attn = Attention(args, 0)
    assert attn.mode == dsv41.MODE_SWA_ONLY
    attn.set_dtype(mx.bfloat16)
    mx.eval(attn.parameters())
    # Quantise using the resident predicate (quantises wo_a: has to_quantized,
    # group-aligned, not switch_mlp/gate).  Affine q8 (default resident codec).
    predicate = dsv41._make_resident_quant_predicate(mode, group_size)
    nn.quantize(attn, group_size=group_size, bits=bits, class_predicate=predicate)
    mx.eval(attn.parameters())
    assert isinstance(attn.wo_a, nn.QuantizedLinear), "wo_a must be quantised for this test"
    return attn


def _fresh_state(args: ModelArgs):
    cache = LayerAttentionCache(
        window_size=args.window_size, compress_ratio=0, is_kv_source=False
    )
    return cache, SharedAttentionRuntime()


def _run_decode(attn: Attention, args: ModelArgs, xs, drop_cache: bool = True):
    """Run one decode step per row of ``xs`` (position advancing by one), returning
    the stacked outputs.  A fresh cache/shared per call so control and candidate see
    identical state."""
    cache, shared = _fresh_state(args)
    if drop_cache and hasattr(attn, "_wo_a_dense_cache"):
        del attn._wo_a_dense_cache
    outs = []
    for t, x in enumerate(xs):
        out = attn._attend(x, mx.array([t]), cache, shared)
        mx.eval(out)
        outs.append(out)
        shared = SharedAttentionRuntime()  # a real per-forward shared runtime
    return outs


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Never let an ambient arm leak in; each test sets the lever explicitly.
    monkeypatch.delenv(dsv41._ATTN_WO_A_CACHE_ENV, raising=False)
    yield


@pytest.mark.parametrize("compile_on", [False, True])
def test_wo_a_cache_byte_identical_64_steps(monkeypatch, compile_on):
    args = _tiny_args()
    attn = _build_quantized_layer()
    mx.random.seed(123)
    steps = 64
    xs = [(mx.random.normal((1, 1, args.hidden_size)) * 0.05).astype(mx.bfloat16)
          for _ in range(steps)]
    mx.eval(xs)

    monkeypatch.setattr(dsv41, "_ATTN_COMPILE", bool(compile_on))

    monkeypatch.setenv(dsv41._ATTN_WO_A_CACHE_ENV, "0")
    ctrl = _run_decode(attn, args, xs)
    monkeypatch.setenv(dsv41._ATTN_WO_A_CACHE_ENV, "1")
    cand = _run_decode(attn, args, xs)

    for t, (a, b) in enumerate(zip(ctrl, cand)):
        assert a.dtype == b.dtype
        assert bool(mx.all(a == b).item()), f"step {t}: wo_a-cache output not byte-identical"


def test_wo_a_dequant_issued_once_when_cached(monkeypatch):
    args = _tiny_args()
    attn = _build_quantized_layer()
    mx.random.seed(7)
    steps = 16
    xs = [(mx.random.normal((1, 1, args.hidden_size)) * 0.05).astype(mx.bfloat16)
          for _ in range(steps)]
    mx.eval(xs)
    monkeypatch.setattr(dsv41, "_ATTN_COMPILE", False)

    calls = {"n": 0}
    real_dequantize = mx.dequantize

    def counting_dequantize(*a, **k):
        calls["n"] += 1
        return real_dequantize(*a, **k)

    # Patch the name the model module resolves (module uses ``mx.dequantize``).
    monkeypatch.setattr(dsv41.mx, "dequantize", counting_dequantize)

    monkeypatch.setenv(dsv41._ATTN_WO_A_CACHE_ENV, "0")
    calls["n"] = 0
    _run_decode(attn, args, xs)
    off_calls = calls["n"]

    monkeypatch.setenv(dsv41._ATTN_WO_A_CACHE_ENV, "1")
    calls["n"] = 0
    _run_decode(attn, args, xs)
    on_calls = calls["n"]

    # Off: one wo_a dequant per decode step.  On: exactly one for the whole run.
    assert off_calls == steps, f"expected {steps} dequants off, got {off_calls}"
    assert on_calls == 1, f"expected 1 dequant with the cache on, got {on_calls}"


def test_wo_a_cache_rebuilds_on_requantize(monkeypatch):
    args = _tiny_args()
    attn = _build_quantized_layer()
    monkeypatch.setattr(dsv41, "_ATTN_COMPILE", False)
    monkeypatch.setenv(dsv41._ATTN_WO_A_CACHE_ENV, "1")

    w1 = attn._o_lora_dense_weight()
    w1b = attn._o_lora_dense_weight()
    # Same packed weight -> same cached array object returned.
    assert w1b is w1

    # Swap in a different quantised wo_a weight (a reload / re-quantise): the cache
    # keys on the packed-weight identity, so it must rebuild and reflect the new
    # values, never return the stale dense array.
    new_lin = nn.QuantizedLinear(
        attn.wo_a.weight.shape[1] * (32 // attn.wo_a.bits),  # nominal in-features
        attn.wo_a.weight.shape[0],
        bias=False, group_size=attn.wo_a.group_size, bits=attn.wo_a.bits,
    )
    mx.random.seed(99)
    dense = (mx.random.normal((attn.n_groups * args.o_lora_rank,
                               attn.n_heads * args.head_dim // attn.n_groups)) * 0.03)
    packed, scales, biases = mx.quantize(dense, group_size=attn.wo_a.group_size, bits=attn.wo_a.bits)
    attn.wo_a.weight = packed
    attn.wo_a.scales = scales
    attn.wo_a.biases = biases
    mx.eval(attn.wo_a.parameters())

    w2 = attn._o_lora_dense_weight()
    assert w2 is not w1, "cache must rebuild when the packed weight identity changes"
    expected = real = mx.dequantize(
        packed, scales, biases, group_size=attn.wo_a.group_size, bits=attn.wo_a.bits,
        mode=getattr(attn.wo_a, "mode", "affine"),
    ).reshape(attn.n_groups, args.o_lora_rank, -1)
    assert bool(mx.all(w2 == expected).item())


def _astype_count(*outs):
    """Number of AsType (dtype-cast) primitives in the lazy graph rooted at ``outs``
    -- via ``mx.export_to_dot``, the same dispatch proxy dispatch_census.py uses."""
    import io
    import re

    buf = io.StringIO()
    mx.export_to_dot(buf, *[a for a in outs if isinstance(a, mx.array)])
    return len(re.findall(r'\[label ="AsType", shape=rectangle\]', buf.getvalue()))


def test_wo_a_cache_is_f32_and_removes_per_token_astype(monkeypatch):
    """W97 review fix (item 1): ``mx.dequantize`` returns bf16 for both codecs, so the
    earlier lever (which cached that bf16) left the per-token ``.astype(mx.float32)``
    on the weight leg in place.  The cache now stores the f32 promotion, so its dtype
    is f32 AND the per-token ``_o_lora_down`` graph carries no AsType on the wo_a leg
    (only the ``o`` cast survives) -- proven with the export_to_dot primitive counter,
    not assumed."""
    attn = _build_quantized_layer()
    monkeypatch.setattr(dsv41, "_ATTN_COMPILE", False)

    # ``o`` reaching ``_o_lora_down`` is [b, s, n_groups, in_per_group]; make it bf16
    # so the ``o.astype(f32)`` is a REAL cast (the single AsType expected to survive).
    in_per_group = attn.n_heads * attn.head_dim // attn.n_groups
    o = mx.zeros((1, 1, attn.n_groups, in_per_group), dtype=mx.bfloat16)
    mx.eval(o)

    # Cache ON: the dense weight is f32 and the wo_a leg contributes 0 AsType.
    monkeypatch.setenv(dsv41._ATTN_WO_A_CACHE_ENV, "1")
    if hasattr(attn, "_wo_a_dense_cache"):
        del attn._wo_a_dense_cache
    w_on = attn._o_lora_dense_weight()
    mx.eval(w_on)
    assert w_on.dtype == mx.float32, f"cache must be f32, got {w_on.dtype}"
    on_count = _astype_count(attn._o_lora_down(o))
    assert on_count == 1, (
        f"cache ON: only the o.astype should remain, got {on_count} AsType "
        "(the f32 cache must make the weight-leg .astype a graph no-op; the cached f32 "
        "weight is a materialised leaf, so no dequant/astype is in the per-token graph)"
    )

    # Cache OFF (control): the per-token graph re-runs the whole dequant (which carries
    # its own internal AsType) AND the bf16 weight's .astype(f32), so it strictly
    # exceeds the cached-f32 graph -- the per-token weight-leg work the fix removes.
    monkeypatch.setenv(dsv41._ATTN_WO_A_CACHE_ENV, "0")
    if hasattr(attn, "_wo_a_dense_cache"):
        del attn._wo_a_dense_cache
    off_count = _astype_count(attn._o_lora_down(o))
    assert off_count > on_count, (
        f"cache OFF must carry the per-token weight-leg AsType(s) the fix removes: "
        f"off={off_count} on={on_count}"
    )
