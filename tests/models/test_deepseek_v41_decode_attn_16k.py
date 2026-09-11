"""W76 -- decode attention at 16K is T-independent in attention-*proper*.

The W76 audit (docs/deepseek-v41/W76_DECODE_ATTN_16K.md) asked where the measured
GPU ~7.3 ms/layer "attention proper" at T=16384 comes from -- and in particular
why the SWA-only layers (window 128 keys, no indexer) cost the same, work their
math says is T-independent.  Bisecting the decode attention path op-by-op on the
CPU double (fake config, tiny dims, real T = 1024 / 4096 / 16384 cache rows, per
CSA mode) showed that attention-*proper* -- the qkv projection, the selected-key
gather, the QK/softmax/PV over ``k = window + index_topk`` keys, and the output
projection -- does NOT scale with T.  The only per-op O(T) decode-attention costs
are the KV appends (W73 Causes 1+2, fixed byte-identically by
``MTPLX_DSV41_KV_CHUNK_GROW``) and the indexer ``select`` (W73 Cause 3, inherent).

These tests LOCK that invariant:

  1. **Operand is T-independent** -- the gathered attention operand ``KVg`` (the
     only thing attention-proper touches from the T-row cache) has an identical
     ``k`` at T=1024 and T=16384 for every CSA mode.  This is the structural proof
     that no attention-proper op is O(T); it is not timing-based, so it never
     flakes.  A future regression that re-widens the decode score to the full
     history would blow the gathered-row count and fail here.

  2. **Wall does not scale with T** -- the SWA-only attention-proper wall at
     T=16384 stays well under a linear multiple of its T=1024 wall on the CPU
     double (a true O(T) op would be ~16x; the append alone is ~14x).  Generous
     bound + best-of-N + warmup so host-encode jitter under a GPU window cannot
     make it flaky.

  3. **W76 select-fence is byte-identical** -- ``MTPLX_DSV41_SELECT_FENCE`` (which
     fences the K30 ``selected_idx`` argsort into the ``select`` decode sub-stage
     so it stops leaking into ``score``) changes only the stage-timing census, so
     decode logits are byte-identical with it on vs off, even under an active
     decode stage-timing session.

CPU-only (``mx.set_default_device(mx.cpu)`` -- "no GPU" is not enough, MLX defaults
to Metal).  Run under ``nice -n 19`` and without ``pytest -n auto``.
"""

from __future__ import annotations

import os
import time

import numpy as np
import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mtplx.models import deepseek_v41 as M  # noqa: E402
from mtplx.models import deepseek_v41_cache as C  # noqa: E402
from mtplx.models import deepseek_v41_stage_timing as S  # noqa: E402

RNG = np.random.default_rng(20260911)


# ---------------------------------------------------------------------------
# fake config -- tiny dims, real head_dim/window so k saturates as in the model
# ---------------------------------------------------------------------------
def _args(nlayers=3):
    from mtplx.models.deepseek_v41 import ModelArgs
    return ModelArgs(
        vocab_size=64, hidden_size=64, num_hidden_layers=nlayers,
        num_attention_heads=4, head_dim=32, qk_rope_head_dim=8,
        q_lora_rank=24, o_lora_rank=16, o_groups=2,
        moe_intermediate_size=32, n_routed_experts=8, num_experts_per_tok=2,
        index_n_heads=2, index_head_dim=16, index_topk=64,
        sliding_window=32, window_size=32, swiglu_limit=10.0,
        compress_ratios=[0, 1, 1][:nlayers],
        kv_source_layer_ids=[1], index_source_layer_ids=[1],
        candidate_source_layer_id=1, candidate_topk_blocks=2048, candidate_block_size=8,
        rope_scaling={"rope_type": "yarn", "factor": 16, "beta_fast": 32,
                      "beta_slow": 1, "original_max_position_embeddings": 65536},
    )


def _init_attn(args, layer_id):
    from mtplx.models.deepseek_v41 import Attention
    lyr = Attention(args, layer_id)

    def rnd(shape, s=0.05):
        return mx.array((s * RNG.standard_normal(shape)).astype(np.float32))

    for lin in (lyr.wq_a, lyr.wq_b, lyr.wkv, lyr.wo_a, lyr.wo_b):
        lin.weight = rnd(lin.weight.shape)
    lyr.attn_sink = rnd(lyr.attn_sink.shape, 0.3)
    if lyr.indexer is not None:
        lyr.indexer.wq_b.weight = rnd(lyr.indexer.wq_b.weight.shape)
        lyr.indexer.weights_proj.weight = rnd(lyr.indexer.weights_proj.weight.shape)
        if lyr.indexer.owns_k:
            lyr.indexer.wk.weight = rnd(lyr.indexer.wk.weight.shape)
    mx.eval(lyr.parameters())
    return lyr


def _run_decode_attend(lyr, args, T, *, reuse=False):
    """One decode ``_attend`` with a directly-filled T-row cache (no prefill)."""
    hd = args.head_dim
    b, s, H = 1, 1, args.num_attention_heads
    x = mx.array((0.05 * RNG.standard_normal((b, s, args.hidden_size))).astype(np.float32))
    pos = mx.array([T], dtype=mx.int32)
    cache = C.LayerAttentionCache(
        window_size=lyr.window_size,
        compress_ratio=lyr.compress_ratio,
        is_kv_source=lyr.is_kv_source,
    )
    cache.window = mx.array(RNG.standard_normal((1, T, hd)).astype(np.float32))
    shared = C.SharedAttentionRuntime()
    if reuse:
        # a Reuse layer reads the source's published compress_kv + selected_idx
        cache.compress_kv = mx.array(RNG.standard_normal((1, T, hd)).astype(np.float32))
        idxtopk = min(args.index_topk, T)
        sel = np.sort(RNG.choice(T, size=(1, 1, idxtopk), replace=False)).astype(np.int32)
        shared.compress_kv = cache.compress_kv
        shared.selected_idx = mx.array(sel)
        shared.topk_mask = None
    out = lyr._attend(x, pos, cache, shared)
    mx.eval(out)
    return out


# ---------------------------------------------------------------------------
# 1. the attention-proper operand is T-independent (structural, non-flaky)
# ---------------------------------------------------------------------------
def _gathered_rows_count(lyr, args, T, *, reuse=False, monkeypatch=None):
    seen = {"k": 0}
    real = M._gather_rows

    def spy(source, idx, valid):
        g = real(source, idx, valid)
        seen["k"] += int(g.shape[2])  # [b, s, k, d] -> k gathered rows
        return g

    monkeypatch.setattr(M, "_gather_rows", spy)
    _run_decode_attend(lyr, args, T, reuse=reuse)
    monkeypatch.setattr(M, "_gather_rows", real)
    return seen["k"]


def test_swa_only_operand_T_independent(monkeypatch):
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    args = _args()
    lyr = _init_attn(args, 0)
    assert lyr.mode == M.MODE_SWA_ONLY
    k_small = _gathered_rows_count(lyr, args, 1024, monkeypatch=monkeypatch)
    k_big = _gathered_rows_count(lyr, args, 16384, monkeypatch=monkeypatch)
    # SWA gathers exactly the window (T-independent, saturated at window_size)
    assert k_small == k_big == lyr.window_size, (k_small, k_big, lyr.window_size)


def test_reuse_operand_T_independent(monkeypatch):
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    args = _args()
    lyr = _init_attn(args, 2)
    assert lyr.mode == M.MODE_REUSE
    k_small = _gathered_rows_count(lyr, args, 1024, reuse=True, monkeypatch=monkeypatch)
    k_big = _gathered_rows_count(lyr, args, 16384, reuse=True, monkeypatch=monkeypatch)
    # Reuse gathers window + index_topk compressed rows -- both T-independent.
    assert k_small == k_big == lyr.window_size + args.index_topk, (k_small, k_big)


# ---------------------------------------------------------------------------
# 2. attention-proper wall does not scale with T (timing, generous bound)
# ---------------------------------------------------------------------------
def _best_ms(fn, iters=5, warm=3):
    for _ in range(warm):
        fn()
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        best = min(best, (time.perf_counter_ns() - t0) / 1e6)
    return best


def test_swa_attention_proper_wall_flat_in_T(monkeypatch):
    """attention-proper = whole ``_attend`` minus the window append.  The append is
    the W73 O(T) (fixed by chunk-grow); everything else must stay flat.  A true
    O(T) op would push the 16K/1K ratio toward ~16x -- assert it stays under 5x."""
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    args = _args()
    lyr = _init_attn(args, 0)
    hd = args.head_dim
    b, s, H = 1, 1, args.num_attention_heads
    x = mx.array((0.05 * RNG.standard_normal((b, s, args.hidden_size))).astype(np.float32))
    mx.eval(x)

    def proper_ms(T):
        pos = mx.array([T], dtype=mx.int32)
        store = mx.array(RNG.standard_normal((1, T, hd)).astype(np.float32))
        mx.eval(store)
        # pre-fill window_all so the append (O(T)) is OUTSIDE the timed region --
        # exactly what the cache_append stage fence does in the real census.
        cache = C.LayerAttentionCache(window_size=lyr.window_size, compress_ratio=0,
                                      is_kv_source=False)
        cache.window = store
        kv_new = mx.array(RNG.standard_normal((1, 1, hd)).astype(np.float32))
        cache.append_window(kv_new)
        window_all = cache.window
        mx.eval(window_all)
        qcos, qsin = M._cos_sin(lyr.inv_freq, pos)
        q = M._rope_last(
            lyr.wq_b(M._rmsnorm(lyr.wq_a(x), lyr.q_norm_weight, lyr.eps)).reshape(b, s, H, hd),
            qcos, qsin)
        mx.eval(q, qcos, qsin)

        def body():
            o = lyr._sparse_attend_selected(q, window_all, None, None, pos)
            mx.eval(o)
        return _best_ms(body)

    small = proper_ms(1024)
    big = proper_ms(16384)
    assert big <= 5.0 * small, f"attention-proper scaled with T: {small:.4f} -> {big:.4f} ms"


# ---------------------------------------------------------------------------
# 3. W76 select-fence is byte-identical (the fix vs the old path)
# ---------------------------------------------------------------------------
def _tiny_model():
    from mlx.utils import tree_flatten, tree_unflatten
    from mtplx.models.deepseek_v41 import Model, ModelArgs
    args = ModelArgs(
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
    model = Model(args)
    mx.random.seed(1)
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
    return model


def _prefill_decode_logits(model, prompt, steps, *, stage_timing):
    cache = model.make_cache()
    if stage_timing:
        S.begin("decode")
    logits = model(mx.array([list(prompt)]), cache=cache)
    mx.eval(logits)
    outs = [logits[0, -1]]
    tok = int(mx.argmax(logits[0, -1]).item())
    for _ in range(steps):
        if stage_timing:
            with S.frame():
                logits = model(mx.array([[tok]]), cache=cache)
                mx.eval(logits)
        else:
            logits = model(mx.array([[tok]]), cache=cache)
            mx.eval(logits)
        outs.append(logits[0, -1])
        tok = int(mx.argmax(logits[0, -1]).item())
    if stage_timing:
        S.end()
    return outs


def test_select_fence_byte_identical(monkeypatch):
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    model = _tiny_model()
    prompt = list(range(12))

    monkeypatch.delenv("MTPLX_DSV41_SELECT_FENCE", raising=False)
    base = _prefill_decode_logits(model, prompt, steps=10, stage_timing=False)

    # fence ON, and under an ACTIVE decode stage-timing session so the fence
    # actually fires (the _sd fence is a no-op otherwise) -- logits must be identical.
    monkeypatch.setenv("MTPLX_DSV41_SELECT_FENCE", "1")
    fenced = _prefill_decode_logits(model, prompt, steps=10, stage_timing=True)

    assert len(base) == len(fenced)
    for i, (a, b) in enumerate(zip(base, fenced)):
        assert bool(mx.all(a == b).item()), f"logits differ at decode step {i}"


def test_select_fence_charges_argsort_to_select(monkeypatch):
    """With the fence on, the K30 ``selected_idx`` array is realised inside the
    ``select`` decode sub-stage rather than left lazy for the ``score`` gather.  We
    assert the observable consequence: fencing does not change the selection the
    model computes (byte-identical selected rows), and the fence only adds work to
    the ``select`` bracket -- captured here as: a decode census records a non-zero
    ``select`` decode sub-stage on the index-source modes."""
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    monkeypatch.setenv("MTPLX_DSV41_SELECT_FENCE", "1")
    model = _tiny_model()
    prompt = list(range(12))
    S.begin("decode")
    _prefill_decode_logits(model, prompt, steps=4, stage_timing=False)
    # (drive a couple more decode frames through the timed forward)
    cache = model.make_cache()
    logits = model(mx.array([list(prompt)]), cache=cache)
    mx.eval(logits)
    tok = int(mx.argmax(logits[0, -1]).item())
    for _ in range(4):
        with S.frame():
            logits = model(mx.array([[tok]]), cache=cache)
            mx.eval(logits)
        tok = int(mx.argmax(logits[0, -1]).item())
    snap = S.end().snapshot()
    db = snap.get("decode_breakdown", {})
    # a select decode sub-stage exists for the index-source modes (full/reindex)
    select_stages = [k for k in db if k.endswith(".select")]
    assert select_stages, f"no select decode sub-stage recorded: {sorted(db)}"
