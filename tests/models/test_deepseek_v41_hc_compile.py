"""W33 / kernel-ledger K4 -- Hyper-Connection tape collapse (CPU, synthetic).
W39 rebase on W32/K3: the tape calls W32's ``_hc_split_sinkhorn`` dispatcher.

DSV4.1 wraps attention and the MoE each in a Hyper-Connection pre/post: a
flatten + rsqrt-norm + small ``fn`` matmul + the ``_hc_split_sinkhorn`` normaliser
(a row-softmax + 20 alternating row/column normalises over a ``[..., hc, hc]``
matrix, 16 floats at decode) + the ``pre_mix`` collapse + RMSNorm on the way in,
and the ``post`` re-mix on the way out -- ~two dozen tiny primitives, twice per
layer.  Uncompiled that is the top per-token *dispatch* source (KERNEL_LEDGER
2.1/4: ~6.4k HC-mix dispatches).  K4 replays those chains from an ``mx.compile``
tape instead of rebuilding the graph from Python each call, behind
``MTPLX_DSV41_HC_COMPILE`` (default OFF).

These gates prove, on a tiny CPU config (no artifact):

  * flag on vs off is ``mx.array_equal`` (f32 CPU) over decode (n=1), a K+1
    verify batch, and chunked + layer-major prefill -- the compiled tape is
    bit-exact to the eager body in the small-row regime the row-cap keeps it in;
  * the compiled tape is inert above the row-cap (prefill one-shot flag on ==
    flag off, both eager);
  * dispatch collapse: eager rebuilds the HC graph (``_hc_split_sinkhorn`` invoked
    twice per layer per token) while a warm compiled tape replays with ZERO
    Python HC-graph construction, and each compiled callable runs exactly once
    per layer per token;
  * the Sinkhorn stays an opaque function boundary (the compiled tape calls the
    module ``_hc_split_sinkhorn`` -- so the K3 worker's Metal kernel drops in when
    ``MTPLX_DSV41_SINKHORN_METAL`` is armed on the GPU);
  * W39: flag-on/off byte-identity holds across ALL FOUR combinations of
    (``MTPLX_DSV41_HC_COMPILE``) x (``MTPLX_DSV41_SINKHORN_METAL``) -- on CPU the
    Metal route is inert (``_sinkhorn_use_kernel`` is False off-GPU), so all four
    are ``mx.array_equal``;
  * ``_hc_use_compile`` gating (flag/env off, or rows > cap -> eager).

Pins MLX to CPU; tiny random config; no artifact load.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

import mtplx.models.deepseek_v41 as dv41  # noqa: E402
from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402

# The tiny config's Hyper-Connection matmul (``[b*s, hc*hidden] @ [hc*hidden,
# mix_hc]`` = ``[., 128] @ [128, 24]``) and the RMS/HC-mix mean reductions match
# the eager kernels BIT-EXACTLY under ``mx.compile`` up to 7 rows and reassociate
# ~5e-7 at >=8 (measured, W33).  Decode (1) and a K=3 verify batch (4) sit well
# under it; the tests keep every compiled shape at rows <= 7.
_TEST_CAP = 7


# ---------------------------------------------------------------------------
# fixtures (mirror tests/models/test_deepseek_v41_layer_major_prefill.py)
# ---------------------------------------------------------------------------
def _csa_args(**over) -> ModelArgs:
    base = dict(
        vocab_size=48, hidden_size=32, num_hidden_layers=8,
        num_attention_heads=4, head_dim=16, qk_rope_head_dim=4,
        q_lora_rank=12, o_lora_rank=8, o_groups=2,
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
def _hc(flag: bool, max_rows: int = _TEST_CAP):
    """Flip the K4 module knobs for a block and always restore them.  Clears the
    compiled-tape cache on entry and exit so a stale trace can never leak across a
    flag flip (each block builds its own fresh tapes)."""
    old_f, old_r = dv41._HC_COMPILE, dv41._HC_COMPILE_MAX_ROWS
    dv41._HC_COMPILE = flag
    dv41._HC_COMPILE_MAX_ROWS = max_rows
    dv41._HC_COMPILED.clear()
    try:
        yield
    finally:
        dv41._HC_COMPILE, dv41._HC_COMPILE_MAX_ROWS = old_f, old_r
        dv41._HC_COMPILED.clear()


@contextlib.contextmanager
def _metal(flag: bool):
    """Arm/disarm ``MTPLX_DSV41_SINKHORN_METAL`` via the env (W32 reads it at use,
    never at import).  On CPU ``_sinkhorn_use_kernel()`` stays False regardless, so
    this is numerically inert here -- exactly what the 2x2 gate asserts."""
    key = dv41._SINKHORN_METAL_ENV
    old = os.environ.get(key)
    if flag:
        os.environ[key] = "1"
    else:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def _new_model(seed=1, **over):
    args = _csa_args(**over)
    model = Model(args)
    _randomize(model, seed=seed)
    return model, args


def _prefill_one_shot(model, args, s, seed):
    """One-shot prefill (chunk 0); s > cap so the HC prep is eager regardless of
    the flag -> the post-prefill cache is identical on both flag settings."""
    ids = mx.array(np.random.RandomState(seed).randint(0, args.vocab_size, size=(1, s)))
    cache = model.make_cache()
    logits = model(ids, cache=cache, prefill_chunk=0)
    mx.eval(logits)
    return cache


# ---------------------------------------------------------------------------
# 1. array_equal: decode (n=1)
# ---------------------------------------------------------------------------
def test_decode_flag_on_off_identical():
    decode_tokens = [3, 17, 5, 29]  # fixed synthetic decode stream (both runs)

    def run(flag):
        model, args = _new_model(seed=1)
        with _hc(flag):
            cache = _prefill_one_shot(model, args, s=12, seed=0)  # >cap -> eager both
            outs = []
            for t in decode_tokens:
                lo = model(mx.array([[t]]), cache=cache)  # n=1 -> compiled when flag
                mx.eval(lo)
                outs.append(np.array(lo))
        return outs

    off = run(False)
    on = run(True)
    for i, (a, b) in enumerate(zip(off, on)):
        assert np.array_equal(a, b), f"decode step {i} logits differ (max {np.max(np.abs(a-b))})"


# ---------------------------------------------------------------------------
# 2. array_equal: a K+1 verify batch (n = K+1)
# ---------------------------------------------------------------------------
def test_verify_batch_flag_on_off_identical():
    K = 3  # standard MTP depth; verify forward is K+1 = 4 query rows
    verify_ids = mx.array([[7, 2, 41, 13]])  # K+1 fixed tokens

    def run(flag):
        model, args = _new_model(seed=2)
        with _hc(flag):
            cache = _prefill_one_shot(model, args, s=12, seed=1)  # >cap -> eager both
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
    with _hc(flag):
        cache = model.make_cache()
        logits = model(ids, cache=cache, prefill_chunk=chunk,
                       prefill_layer_major=(True if layer_major else None))
        mx.eval(logits)
        return np.array(logits)


def test_prefill_chunked_flag_on_off_identical():
    # chunk 5 <= cap 7: every span's HC prep is compiled on flag-on, eager on
    # flag-off -> array_equal proves the compiled small-chunk prefill is bit-exact.
    off = _prefill_logits(False, chunk=5, layer_major=False)
    on = _prefill_logits(True, chunk=5, layer_major=False)
    assert np.array_equal(off, on), f"chunked prefill differs (max {np.max(np.abs(off-on))})"


def test_prefill_layer_major_flag_on_off_identical():
    off = _prefill_logits(False, chunk=5, layer_major=True)
    on = _prefill_logits(True, chunk=5, layer_major=True)
    assert np.array_equal(off, on), f"layer-major prefill differs (max {np.max(np.abs(off-on))})"


# ---------------------------------------------------------------------------
# 4. the compiled tape is INERT above the row-cap (prefill one-shot)
# ---------------------------------------------------------------------------
def test_compile_inert_above_row_cap():
    # s = 12 > cap 7: even with the flag ON the HC prep falls to the eager body,
    # so flag-on one-shot logits must equal flag-off exactly.
    model_off, args = _new_model(seed=5)
    ids = mx.array(np.random.RandomState(6).randint(0, args.vocab_size, size=(1, 12)))
    with _hc(False):
        lo = np.array(model_off(ids, cache=model_off.make_cache(), prefill_chunk=0))
    model_on, _ = _new_model(seed=5)
    with _hc(True):
        ln = np.array(model_on(ids, cache=model_on.make_cache(), prefill_chunk=0))
    assert np.array_equal(lo, ln)


# ---------------------------------------------------------------------------
# 5. cache state is identical flag on vs off (no cache mutation in the tapes)
# ---------------------------------------------------------------------------
def _cache_snapshot(cache):
    out = {}
    for i, lc in enumerate(cache.layers):
        for nm in ("window", "compress_kv", "index_k"):
            a = getattr(lc, nm, None)
            if a is not None:
                out[f"{i}.{nm}"] = np.array(a)
    return out, int(cache.offset)


def test_cache_state_identical_flag_on_off():
    decode_tokens = [11, 4, 22]

    def run(flag):
        model, args = _new_model(seed=6)
        with _hc(flag):
            cache = _prefill_one_shot(model, args, s=10, seed=2)
            for t in decode_tokens:
                mx.eval(model(mx.array([[t]]), cache=cache))
            return _cache_snapshot(cache)

    snap_off, off_off = run(False)
    snap_on, off_on = run(True)
    assert off_off == off_on
    assert set(snap_off) == set(snap_on)
    for k, v in snap_off.items():
        assert np.array_equal(snap_on[k], v), f"cache[{k}] differs flag on vs off"


# ---------------------------------------------------------------------------
# 6. dispatch collapse + one compiled-callable invocation per layer per token.
#
# Two distinct counts:
#   (A) Python HC-graph CONSTRUCTION -- counted on ``_hc_split_sinkhorn`` (the
#       module boundary both the eager body and the compiled trace call).  Eager
#       rebuilds it 2x per layer per token (the ~6.4k-dispatch source, rebuilt
#       from Python every step); the compiled path builds it ONCE for the whole
#       model (one tape shared across all layers -- weights are tape inputs) and a
#       warm decode token then reconstructs NOTHING.
#   (B) compiled-callable INVOCATION -- counted by wrapping ``_hc_compiled``'s
#       return.  A warm decode token invokes each tape exactly once per layer.
# ---------------------------------------------------------------------------
def test_dispatch_collapse_and_once_per_layer():
    model, args = _new_model(seed=7)
    L = args.num_hidden_layers

    build = {"sinkhorn": 0}          # (A) graph construction
    real_sinkhorn = dv41._hc_split_sinkhorn

    def c_sinkhorn(*a, **k):
        build["sinkhorn"] += 1
        return real_sinkhorn(*a, **k)

    inv = {"attn_prep": 0, "ffn_prep": 0, "moe_combine": 0}   # (B) invocations
    real_hc_compiled = dv41._hc_compiled

    def counting_hc_compiled(kind, *consts):
        fn = real_hc_compiled(kind, *consts)

        def wrapped(*a, **k):
            inv[kind] += 1
            return fn(*a, **k)

        return wrapped

    dv41._hc_split_sinkhorn = c_sinkhorn
    dv41._hc_compiled = counting_hc_compiled
    try:
        # ---- eager: every decode token rebuilds the HC graph from Python ----
        with _hc(False):
            cache = _prefill_one_shot(model, args, s=10, seed=3)
            build["sinkhorn"] = 0
            inv.update(attn_prep=0, ffn_prep=0, moe_combine=0)
            n_tokens = 3
            for t in (5, 8, 2):
                mx.eval(model(mx.array([[t]]), cache=cache))
            # 2 HC mixes (attn + ffn) rebuilt per layer per token; no compiled
            # callable is used with the flag off.
            assert build["sinkhorn"] == 2 * L * n_tokens, build
            assert inv == {"attn_prep": 0, "ffn_prep": 0, "moe_combine": 0}, inv

        # ---- compiled: one shared trace, then warm replay builds nothing ----
        with _hc(True):
            cache = _prefill_one_shot(model, args, s=10, seed=3)  # >cap -> eager prefill
            build["sinkhorn"] = 0
            inv.update(attn_prep=0, ffn_prep=0, moe_combine=0)
            # cold token: the rows=1 tapes trace ONCE for the whole model (shared
            # across layers) -> 2 sinkhorn builds total (one per mix tape); the
            # compiled callables are still invoked once per layer.
            mx.eval(model(mx.array([[5]]), cache=cache))
            assert build["sinkhorn"] == 2, build
            assert inv == {"attn_prep": L, "ffn_prep": L, "moe_combine": L}, inv

            # warm replay: HC graph construction is ZERO (the dispatch collapse);
            # each tape is still invoked exactly once per layer per token.
            build["sinkhorn"] = 0
            inv.update(attn_prep=0, ffn_prep=0, moe_combine=0)
            for t in (8, 2, 9):
                mx.eval(model(mx.array([[t]]), cache=cache))
            assert build["sinkhorn"] == 0, build   # <-- the dispatch collapse
            assert inv == {"attn_prep": 3 * L, "ffn_prep": 3 * L, "moe_combine": 3 * L}, inv
    finally:
        dv41._hc_split_sinkhorn = real_sinkhorn
        dv41._hc_compiled = real_hc_compiled


# ---------------------------------------------------------------------------
# 7. _hc_use_compile gating
# ---------------------------------------------------------------------------
def test_hc_use_compile_gating():
    x1 = mx.zeros((1, 1, 4, 32))    # 1 row  (decode)
    x4 = mx.zeros((1, 4, 4, 32))    # 4 rows (verify K+1)
    x9 = mx.zeros((1, 9, 4, 32))    # 9 rows (> cap)
    with _hc(False):
        assert dv41._hc_use_compile(x1) is False   # flag off -> never
    with _hc(True, max_rows=7):
        assert dv41._hc_use_compile(x1) is True
        assert dv41._hc_use_compile(x4) is True
        assert dv41._hc_use_compile(x9) is False   # rows > cap -> eager


# ---------------------------------------------------------------------------
# 8. W39: byte-identity across all four (HC_COMPILE) x (SINKHORN_METAL) combos.
#
# On CPU ``_sinkhorn_use_kernel()`` is always False (default device is not the
# GPU), so W32's Metal route is inert and the SINKHORN_METAL flag can never move
# CPU numerics; K4's compiled tape is bit-exact to eager in the small-row regime.
# So all four combinations must produce identical logits over decode, a K+1
# verify batch, and a chunk-5 (<= cap) prefill.
# ---------------------------------------------------------------------------
def _probe_logits(hc_flag, metal_flag):
    out = {}
    with _metal(metal_flag), _hc(hc_flag):
        # decode: prefill one-shot (> cap -> eager both) then 4 n=1 steps
        model, args = _new_model(seed=11)
        cache = _prefill_one_shot(model, args, s=12, seed=7)
        out["decode"] = []
        for t in (3, 17, 5, 29):
            lo = model(mx.array([[t]]), cache=cache)
            mx.eval(lo)
            out["decode"].append(np.array(lo))
        # verify batch (K+1 = 4 rows) against a fresh prefilled cache
        m2, a2 = _new_model(seed=11)
        c2 = _prefill_one_shot(m2, a2, s=12, seed=7)
        v = m2(mx.array([[7, 2, 41, 13]]), cache=c2)
        mx.eval(v)
        out["verify"] = np.array(v)
        # chunk-5 prefill (<= cap -> compiled per span when HC on)
        m3, a3 = _new_model(seed=11)
        ids = mx.array(np.random.RandomState(4).randint(0, a3.vocab_size, size=(1, 20)))
        pl = m3(ids, cache=m3.make_cache(), prefill_chunk=5)
        mx.eval(pl)
        out["prefill_chunk5"] = np.array(pl)
    return out


def test_flag_2x2_byte_identical_on_cpu():
    combos = [(False, False), (True, False), (False, True), (True, True)]
    ref = _probe_logits(*combos[0])
    for hc_flag, metal_flag in combos[1:]:
        got = _probe_logits(hc_flag, metal_flag)
        tag = f"HC_COMPILE={hc_flag} SINKHORN_METAL={metal_flag}"
        for i, (a, b) in enumerate(zip(ref["decode"], got["decode"])):
            assert np.array_equal(a, b), f"decode[{i}] differs vs baseline ({tag})"
        assert np.array_equal(ref["verify"], got["verify"]), f"verify differs vs baseline ({tag})"
        assert np.array_equal(ref["prefill_chunk5"], got["prefill_chunk5"]), \
            f"chunk-5 prefill differs vs baseline ({tag})"
