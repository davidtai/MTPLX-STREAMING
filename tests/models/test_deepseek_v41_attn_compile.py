"""W41 / kernel-ledger K22 -- attention-chain (and gate-prefix / MoE-combine)
tape collapse (CPU, synthetic, no artifact).

``MTPLX_DSV41_ATTN_COMPILE`` replays two PURE attention chains from an
``mx.compile`` tape instead of rebuilding the graph each decode step -- the
pre-SDPA q/kv projection+norm+rope prep (up to, not including, the KV-cache write
and the SDPA) and the post-SDPA output chain -- plus the pure MoE gate prefix
(score GEMM + sqrtsoftplus + bias) and the MoE combine.  All four are byte-for-
byte the eager body with the flag off / above the row cap; on and in the small
decode/verify row regime they are ``mx.array_equal`` with eager.

Gates (all pin MLX to CPU; tiny random config; no artifact load):
  * flag on vs off ``mx.array_equal`` over decode (n=1), a K+1 verify batch, and
    chunked + layer-major prefill;
  * the tape is inert above the row cap (prefill one-shot flag on == flag off);
  * KV / compress / index cache state is identical on vs off (the tapes are pure,
    the KV-cache write stays outside them);
  * the primitive-count collapse is asserted straight from the W41 census tool
    (``scripts/deepseek_v41/dispatch_census.py``): the compiled attention QKV /
    output / gate-prefix / combine chains each dispatch strictly fewer graph
    primitives than eager, and the whole decode token's primitive count drops;
  * ``_attn_use_compile`` gating (flag/env off, or rows > cap -> eager).

The tiny config uses ``q_lora_rank=16`` (a power-of-2 RMSNorm reduction): the
default hc-compile config's ``q_lora_rank=12`` is a pathological non-power-of-2
reduction size that ``mx.mean`` reassociates under ``mx.compile`` (a tiny-config
artifact -- the real model's q_lora_rank=1280 / head_dim=512 are compile-stable,
measured W41).  The compiled attention prep chain is ``array_equal`` with eager
up to 7 rows and reassociates the projection matmul at >= 8 (measured, matching
W33/K4), so the tests keep every compiled shape at rows <= 7.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

import mtplx.models.deepseek_v41 as dv41  # noqa: E402
from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402

# rows <= 7 are array_equal; the projection matmul reassociates at >= 8 (W33/K4).
_TEST_CAP = 7
_DOT_RECT = re.compile(r'\[label ="([^"]+)", shape=rectangle\]')


def _count_prims(*outs):
    arrs = [a for a in outs if isinstance(a, mx.array)]
    buf = io.StringIO()
    mx.export_to_dot(buf, *arrs)
    return len(_DOT_RECT.findall(buf.getvalue()))


def _csa_args(**over) -> ModelArgs:
    base = dict(
        vocab_size=48, hidden_size=32, num_hidden_layers=8,
        num_attention_heads=4, head_dim=16, qk_rope_head_dim=4,
        q_lora_rank=16, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=2,
        index_n_heads=2, index_head_dim=8, index_topk=5,
        sliding_window=8, window_size=8, swiglu_limit=0.5,
        compress_ratios=[0, 0, 2, 2, 2, 1, 1, 1],
        kv_source_layer_ids=[2, 5], index_source_layer_ids=[2, 5, 6],
        candidate_source_layer_id=5, candidate_topk_blocks=3, candidate_block_size=2,
        rope_scaling={"rope_type": "yarn", "factor": 16, "beta_fast": 32,
                      "beta_slow": 1, "original_max_position_embeddings": 65536},
    )
    base.update(over)
    return ModelArgs(**base)


def _randomize(model, seed=0, scale=0.1):
    mx.random.seed(seed)
    new = []
    for name, arr in tree_flatten(model.parameters()):
        if arr.ndim == 1 and ("norm_weight" in name or name.endswith("norm.weight")):
            v = 1.0 + 0.2 * mx.random.normal(arr.shape)
        elif "attn_sink" in name:
            v = 0.5 * mx.random.normal(arr.shape)
        else:
            v = scale * mx.random.normal(arr.shape)
        new.append((name, v.astype(mx.float32)))
    model.update(tree_unflatten(new))
    mx.eval(model.parameters())


@contextlib.contextmanager
def _ac(flag: bool, max_rows: int = _TEST_CAP):
    """Flip the K22 knobs for a block and always restore them; clear the shared
    compiled-tape cache on entry and exit so a stale trace never leaks across a
    flag flip (the gate-prefix / combine tapes share this cache)."""
    old_f, old_r = dv41._ATTN_COMPILE, dv41._ATTN_COMPILE_MAX_ROWS
    dv41._ATTN_COMPILE = flag
    dv41._ATTN_COMPILE_MAX_ROWS = max_rows
    dv41._ATTN_COMPILED.clear()
    try:
        yield
    finally:
        dv41._ATTN_COMPILE, dv41._ATTN_COMPILE_MAX_ROWS = old_f, old_r
        dv41._ATTN_COMPILED.clear()


def _new_model(seed=1, **over):
    args = _csa_args(**over)
    model = Model(args)
    _randomize(model, seed=seed)
    return model, args


def _prefill_one_shot(model, args, s, seed):
    ids = mx.array(np.random.RandomState(seed).randint(0, args.vocab_size, size=(1, s)))
    cache = model.make_cache()
    mx.eval(model(ids, cache=cache, prefill_chunk=0))
    return cache


# ---------------------------------------------------------------------------
# 1. array_equal: decode (n=1)
# ---------------------------------------------------------------------------
def test_decode_flag_on_off_identical():
    decode_tokens = [3, 17, 5, 29]

    def run(flag):
        model, args = _new_model(seed=1)
        with _ac(flag):
            cache = _prefill_one_shot(model, args, s=12, seed=0)  # >cap -> eager both
            outs = []
            for t in decode_tokens:
                lo = model(mx.array([[t]]), cache=cache)  # n=1 <= cap -> compiled when flag
                mx.eval(lo)
                outs.append(np.array(lo))
        return outs

    off, on = run(False), run(True)
    for i, (a, b) in enumerate(zip(off, on)):
        assert np.array_equal(a, b), f"decode step {i} differs (max {np.max(np.abs(a-b))})"


# ---------------------------------------------------------------------------
# 2. array_equal: a K+1 verify batch (n = K+1)
# ---------------------------------------------------------------------------
def test_verify_batch_flag_on_off_identical():
    K = 3
    verify_ids = mx.array([[7, 2, 41, 13]])

    def run(flag):
        model, args = _new_model(seed=2)
        with _ac(flag):
            cache = _prefill_one_shot(model, args, s=12, seed=1)
            logits = model(verify_ids, cache=cache)  # n=K+1=4 <= cap -> compiled when flag
            mx.eval(logits)
            return np.array(logits)

    assert run(False).shape[1] == K + 1
    assert np.array_equal(run(False), run(True))


# ---------------------------------------------------------------------------
# 3. array_equal: chunked prefill (compiled small chunks) + layer-major
# ---------------------------------------------------------------------------
def _prefill_logits(flag, *, chunk, layer_major, s=20, seed=4, model_seed=3):
    model, args = _new_model(seed=model_seed)
    ids = mx.array(np.random.RandomState(seed).randint(0, args.vocab_size, size=(1, s)))
    with _ac(flag):
        cache = model.make_cache()
        logits = model(ids, cache=cache, prefill_chunk=chunk,
                       prefill_layer_major=(True if layer_major else None))
        mx.eval(logits)
        return np.array(logits)


def test_prefill_chunked_flag_on_off_identical():
    off = _prefill_logits(False, chunk=5, layer_major=False)  # chunk 5 <= cap -> compiled
    on = _prefill_logits(True, chunk=5, layer_major=False)
    assert np.array_equal(off, on), f"chunked prefill differs (max {np.max(np.abs(off-on))})"


def test_prefill_layer_major_flag_on_off_identical():
    off = _prefill_logits(False, chunk=5, layer_major=True)
    on = _prefill_logits(True, chunk=5, layer_major=True)
    assert np.array_equal(off, on), f"layer-major prefill differs (max {np.max(np.abs(off-on))})"


# ---------------------------------------------------------------------------
# 4. the compiled tape is INERT above the row cap (prefill one-shot)
# ---------------------------------------------------------------------------
def test_compile_inert_above_row_cap():
    model_off, args = _new_model(seed=5)
    ids = mx.array(np.random.RandomState(6).randint(0, args.vocab_size, size=(1, 12)))
    with _ac(False):
        lo = np.array(model_off(ids, cache=model_off.make_cache(), prefill_chunk=0))
    model_on, _ = _new_model(seed=5)
    with _ac(True):  # s=12 > cap 7 -> even flag-on falls to eager
        ln = np.array(model_on(ids, cache=model_on.make_cache(), prefill_chunk=0))
    assert np.array_equal(lo, ln)


# ---------------------------------------------------------------------------
# 5. cache state is identical flag on vs off (no cache mutation in the tapes)
# ---------------------------------------------------------------------------
def _cache_snapshot(cache):
    out = {}
    for i, lc in enumerate(cache.layers):
        for nm in ("window", "compress_kv", "index_k"):
            arr = getattr(lc, nm, None)
            if isinstance(arr, mx.array):
                out[f"{i}.{nm}"] = np.array(arr)
    return out


def test_cache_state_identical_flag_on_off():
    def run(flag):
        model, args = _new_model(seed=7)
        with _ac(flag):
            cache = _prefill_one_shot(model, args, s=12, seed=8)
            for t in [11, 4, 33]:
                mx.eval(model(mx.array([[t]]), cache=cache))
            return _cache_snapshot(cache)

    off, on = run(False), run(True)
    assert set(off) == set(on) and off
    for k in off:
        assert np.array_equal(off[k], on[k]), f"cache {k} differs on/off"


# ---------------------------------------------------------------------------
# 6. primitive-count collapse asserted from the W41 census tool
# ---------------------------------------------------------------------------
def _load_census():
    path = Path(__file__).resolve().parents[2] / "scripts" / "deepseek_v41" / "dispatch_census.py"
    spec = importlib.util.spec_from_file_location("dsv41_dispatch_census", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_census_micro_reduction_per_chain():
    census = _load_census()
    micro = census._micro_census(seed=1)
    # every compiled chain dispatches strictly fewer graph primitives than eager
    for name, d in micro.items():
        assert d["compiled"]["primitives"] < d["eager"]["primitives"], (
            name, d["eager"]["primitives"], d["compiled"]["primitives"])
        assert d["reduction"] == d["eager"]["primitives"] - d["compiled"]["primitives"]
    # the attention prep chains carry the bulk of the collapse
    assert micro["attn.qkv_prep"]["reduction"] >= 20
    assert micro["moe.gate_prefix"]["reduction"] >= 1


def test_census_full_model_total_drops():
    census = _load_census()
    before = census._run_full_census(False, _TEST_CAP, seed=1)
    after = census._run_full_census(True, _TEST_CAP, seed=1)
    b = before["total_primitives_per_token"]
    a = after["total_primitives_per_token"]
    assert a < b, (a, b)
    # every attention stage's per-token primitives drop; the stages K22 does not
    # touch (HC, routed switch, shared expert) are unchanged.
    for mode in ("attn.reuse", "attn.full", "attn.swa_only", "attn.reindex"):
        assert after["stages"][mode]["primitives_per_token"] < \
            before["stages"][mode]["primitives_per_token"], mode
    for untouched in ("hc.premix_sinkhorn", "hc.combine", "moe.routed_switch",
                      "moe.shared_expert"):
        assert after["stages"][untouched]["primitives_per_token"] == \
            before["stages"][untouched]["primitives_per_token"], untouched
    # the gate prefix and combine folds do reduce their stages
    assert after["stages"]["moe.gate_topk"]["primitives_per_token"] < \
        before["stages"]["moe.gate_topk"]["primitives_per_token"]
    assert after["stages"]["moe.combine"]["primitives_per_token"] < \
        before["stages"]["moe.combine"]["primitives_per_token"]


# ---------------------------------------------------------------------------
# 7. _attn_use_compile gating
# ---------------------------------------------------------------------------
def test_attn_use_compile_gating():
    with _ac(False):
        assert dv41._attn_use_compile(1) is False  # flag off -> never
    with _ac(True, max_rows=7):
        assert dv41._attn_use_compile(1) is True
        assert dv41._attn_use_compile(7) is True
        assert dv41._attn_use_compile(8) is False  # rows > cap -> eager


def test_env_default_off():
    # the module import-time default is OFF unless the env is truthy.
    assert dv41._ATTN_COMPILE_ENV == "MTPLX_DSV41_ATTN_COMPILE"
    truthy = (os.environ.get(dv41._ATTN_COMPILE_ENV) or "").strip().lower() not in (
        "", "0", "false", "no", "off", "auto")
    assert dv41._ATTN_COMPILE == truthy


# ---------------------------------------------------------------------------
# 8. one compiled tape shared across layers (weights are tape inputs)
# ---------------------------------------------------------------------------
def test_one_qkv_tape_shared_across_layers():
    model, args = _new_model(seed=9)
    with _ac(True):
        cache = _prefill_one_shot(model, args, s=12, seed=9)
        mx.eval(model(mx.array([[5]]), cache=cache))  # one decode token, all layers
        qkv_keys = [k for k in dv41._ATTN_COMPILED if k[0] == "qkv"]
        out_keys = [k for k in dv41._ATTN_COMPILED if k[0] == "out"]
        # all 8 layers share ONE qkv tape and ONE out tape (same codec/geometry).
        assert len(qkv_keys) == 1, qkv_keys
        assert len(out_keys) == 1, out_keys


# ---------------------------------------------------------------------------
# 9. quantized projection path: the tape applies mx.quantized_matmul exactly as
#    nn.QuantizedLinear (the serving residents are q8/mxfp*), bit-exact under
#    mx.compile.  The tiny model keeps projections dense (dims not gs-aligned), so
#    this exercises the _apply_lin quantized branch directly at unit level.
# ---------------------------------------------------------------------------
def test_apply_lin_quantized_matches_module_under_compile():
    import mlx.nn as nn
    from mtplx.models.deepseek_v41 import _lin_desc, _lin_arrays, _apply_lin

    mx.random.seed(3)
    lin = nn.Linear(64, 32, bias=False)
    lin.weight = 0.1 * mx.random.normal((32, 64))
    qlin = lin.to_quantized(group_size=64, bits=8)  # affine q8, carries biases
    x = 0.3 * mx.random.normal((4, 64))
    mx.eval(x, qlin.weight, qlin.scales, qlin.biases)

    desc, arrs = _lin_desc(qlin), _lin_arrays(qlin)
    assert desc[0] == "q" and len(arrs) == 3  # (weight, scales, biases)

    eager = qlin(x)  # the exact nn.QuantizedLinear.__call__ path
    compiled = mx.compile(lambda xx, *a: _apply_lin(desc, a, xx))(x, *arrs)
    mx.eval(eager, compiled)
    assert np.array_equal(np.array(eager), np.array(compiled))
