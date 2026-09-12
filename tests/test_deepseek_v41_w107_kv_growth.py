"""W107 -- bounded / preallocated KV growth (MTPLX_DSV41_KV_BOUNDED).

David: "controlling kv growth is crucial for everything."  The shipped cache grows
its KV lanes with a per-token ``mx.concatenate`` (:func:`_grow`) -- an O(current
length) copy per lane per token, O(T^2) over the cell.  W80 bounded the window
(ring) and preallocated compress/index only when ``MTPLX_DSV41_WINDOW_RING_MAXKV``
was set, but left the compressor frontier (``comp_state.raw_kv`` / ``raw_score``,
the "main latent KV") growing with ``_grow``.  Under ``MTPLX_DSV41_KV_BOUNDED``
EVERY lane is bounded/preallocated to ``max_kv`` at prefill and written in place
with a donated ``mx.slice_update`` (O(new rows)/token, no per-token realloc):

  * window   -- the W80 bounded sliding-window ring.
  * compress -- preallocated ``_GrowBuffer`` (``ceil(max_kv/ratio)`` rows).
  * index    -- preallocated ``_GrowBuffer`` (``ceil(max_kv/ratio)`` rows).
  * latent   -- the compressor frontier, PREALLOCATED to ``max_kv`` (W107 fixes the
                last O(T) concatenate lane).

These tests prove, on CPU only:

  1. **Preallocation size == :func:`kv_bytes_at_max_kv`** -- the bytes a bounded
     cache actually allocates equal the pure formula W106 will use.
  2. **Append beyond max_kv raises cleanly** -- the preallocated lanes are hard
     bounded (no silent geometric resize past the cap).
  3. **In-place stability across N appends** -- every lane's raw backing keeps its
     data pointer / capacity stable across decode appends (no per-token realloc:
     ``kv_realloc_<lane>`` flat, ``get_active_memory`` flat), while the plain
     ``_grow`` path's memory grows.
  4. **Byte-identical outputs** -- a tiny-model prefill+decode is BIT-identical
     bounded vs the shipped path (selected-key path) and bounded vs the W80 ring
     arm (pure prealloc/in-place isolation), same seed.
  5. **Counters increment** -- the per-lane engagement counters populate.

CPU-only (``mx.set_default_device(mx.cpu)`` -- "no GPU" is not enough, MLX defaults
to Metal).  Tiny dims; run under ``nice -n 19`` and without ``pytest -n auto``.
"""

from __future__ import annotations

import os

import numpy as np
import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mtplx.models import deepseek_v41_cache as C  # noqa: E402

RNG = np.random.default_rng(20260912)


def _row(n, d, rng=RNG):
    return mx.array(rng.standard_normal((1, n, d)).astype(np.float32))


def _eq(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return tuple(a.shape) == tuple(b.shape) and bool(mx.all(a == b).item())


# Tiny synthetic model config (a stand-in for ModelArgs for the pure-formula path).
class _Cfg:
    num_hidden_layers = 8
    window_size = 8
    head_dim = 16
    index_head_dim = 12
    compress_ratios = [0, 0, 2, 2, 2, 1, 1, 1]
    kv_source_layer_ids = [2, 5]


def _bounded_env(monkeypatch, maxkv=64):
    """Arm the bounded lanes with an explicit max_kv and the DEFAULT ring tuning
    (max_verify 8 / slack 8 / headroom 64), so :func:`kv_bytes_at_max_kv`'s defaults
    match what the cache preallocates."""
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED", "1")
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED_MAXKV", str(maxkv))
    for k in ("MTPLX_DSV41_WINDOW_RING_MAX_VERIFY", "MTPLX_DSV41_WINDOW_RING_SLACK",
              "MTPLX_DSV41_WINDOW_RING_HEADROOM", "MTPLX_DSV41_WINDOW_RING_MAXKV",
              "MTPLX_DSV41_WINDOW_RING"):
        monkeypatch.delenv(k, raising=False)


def _prefill_all_lanes(cache, cfg, n):
    """Feed ``n`` prefill rows through every lane of every layer so each bounded
    buffer allocates at its cap (window head_dim, compress head_dim, index
    index_head_dim, latent head_dim via the compressor frontier)."""
    for L, lc in enumerate(cache.layers):
        lc.append_window(_row(n, cfg.head_dim))
        if L in set(cfg.kv_source_layer_ids):
            ratio = cfg.compress_ratios[L]
            if ratio > 1:
                lc.comp_state.push(_row(n, cfg.head_dim), _row(n, cfg.head_dim))
            groups = n // ratio if ratio >= 1 else 0
            if groups:
                lc.append_compress(_row(groups, cfg.head_dim))
                lc.append_index_k(_row(groups, cfg.index_head_dim))
        lc.advance(n)


def _make_cache(cfg):
    return C.DeepseekV41Cache(
        cfg.num_hidden_layers, window_size=cfg.window_size,
        compress_ratios=cfg.compress_ratios,
        kv_source_layer_ids=cfg.kv_source_layer_ids,
    )


# ---------------------------------------------------------------------------
# 1. preallocation size == kv_bytes_at_max_kv
# ---------------------------------------------------------------------------
def test_kv_bytes_formula_matches_preallocation(monkeypatch):
    cfg = _Cfg()
    max_kv = 64
    _bounded_env(monkeypatch, max_kv)
    C.reset_kv_bounded_stats()
    cache = _make_cache(cfg)
    assert all(lc._kv_bounded for lc in cache.layers), "flag not read at construction"
    # one prefill append per lane allocates every bounded buffer at its cap
    _prefill_all_lanes(cache, cfg, n=20)  # 20 < window phys_cap (88) => no transient grow

    # this direct-cache prefill feeds fp32 rows through EVERY window (incl. layer 0),
    # so the whole cache is fp32 -> model_dtype_bytes == store_dtype_bytes == 4.
    expected = C.kv_bytes_at_max_kv(
        cfg, max_kv, model_dtype_bytes=4, store_dtype_bytes=4)
    stats = C.kv_bounded_stats()
    assert stats["alloc_bytes"] == expected, (
        f"alloc_bytes {stats['alloc_bytes']} != kv_bytes_at_max_kv {expected}")

    # sanity: sum the live raw backings directly, too (window 2x ping-pong,
    # latent kv+score, per kv-source layer)
    live = 0
    for L, lc in enumerate(cache.layers):
        live += int(lc._window._bufs[0].nbytes) + int(lc._window._bufs[1].nbytes)
        if lc._compress_kv.raw_backing() is not None:
            live += int(lc._compress_kv.raw_backing().nbytes)
        if lc._index_k.raw_backing() is not None:
            live += int(lc._index_k.raw_backing().nbytes)
        if lc.comp_state is not None:
            for a in lc.comp_state.raw_backings():
                if a is not None:
                    live += int(a.nbytes)
    assert live == expected, f"live raw-backing bytes {live} != formula {expected}"


def test_kv_bytes_breakdown_scales_with_max_kv():
    cfg = _Cfg()
    small = C.kv_bytes_breakdown_at_max_kv(cfg, 64)
    big = C.kv_bytes_breakdown_at_max_kv(cfg, 4096)
    # compress/index/latent scale with max_kv; window does NOT (bounded ring)
    assert big["compress"] > small["compress"]
    assert big["index"] > small["index"]
    assert big["latent"] > small["latent"]
    assert big["window"] == small["window"], "window must not grow with max_kv"
    assert big["total"] == big["window"] + big["compress"] + big["index"] + big["latent"]
    assert C.kv_bytes_at_max_kv(cfg, 4096) == big["total"]
    # defaults: layer-0 window bf16 (2), everything else fp32 (4). An all-fp32 run
    # (model_dtype_bytes=4) only widens layer 0's window (one layer, 2->4).
    f32 = C.kv_bytes_breakdown_at_max_kv(cfg, 4096, model_dtype_bytes=4)
    phys = big["phys_cap"]
    assert f32["window"] - big["window"] == 2 * phys * cfg.head_dim * (4 - 2)
    assert f32["compress"] == big["compress"]     # already fp32
    assert f32["latent"] == big["latent"]         # already fp32


def test_medium_a_all_kv_source_compress_index_fp32():
    """Review MEDIUM-A: compress/index are fp32 on EVERY kv-source layer -- ratio>1
    pools in fp32, ratio==1 follows the fp32 residual (kv-source layers are all L>0).
    (This supersedes MEDIUM-2's ratio==1==bf16 model.)  _Cfg kv_source [2, 5]."""
    cfg = _Cfg()
    max_kv = 4096
    bd = C.kv_bytes_breakdown_at_max_kv(cfg, max_kv)  # defaults: model 2, store 4
    cc2 = C._bounded_comp_cap(max_kv, 2)              # layer 2 (ratio 2)
    cc1 = C._bounded_comp_cap(max_kv, 1)              # layer 5 (ratio 1)
    hd, ihd = cfg.head_dim, cfg.index_head_dim
    assert bd["compress"] == (cc2 + cc1) * hd * 4    # both fp32
    assert bd["index"] == (cc2 + cc1) * ihd * 4
    # window: layer 0 is model dtype (bf16=2), layers 1..N-1 fp32 (4)
    phys = bd["phys_cap"]
    exp_window = 2 * phys * hd * 2 + (cfg.num_hidden_layers - 1) * 2 * phys * hd * 4
    assert bd["window"] == exp_window


# ---------------------------------------------------------------------------
# 2. append beyond max_kv raises cleanly
# ---------------------------------------------------------------------------
def test_append_beyond_max_kv_raises_grow_buffer():
    gb = C._GrowBuffer(bounded_cap=4, counter_lane="compress")
    gb.append(_row(4, 8))                      # fills the cap exactly
    with pytest.raises(ValueError, match="exceed preallocated cap"):
        gb.append(_row(1, 8))                  # one past the cap -> clean raise
    # a multi-row overflow in one call also raises (nothing partially written)
    gb2 = C._GrowBuffer(bounded_cap=4, counter_lane="index")
    with pytest.raises(ValueError, match="exceed preallocated cap"):
        gb2.append(_row(5, 8))


def test_append_beyond_max_kv_raises_latent_frontier():
    cs = C.CompressorState(2, bounded=True, maxkv=4)  # latent cap = 4 + slack(8) = 12
    cs.push(_row(12, 16), _row(12, 16))        # fills the cap
    with pytest.raises(ValueError, match="exceed preallocated cap"):
        cs.push(_row(1, 16), _row(1, 16))


def test_append_beyond_max_kv_raises_via_cache(monkeypatch):
    _bounded_env(monkeypatch, maxkv=8)
    cache = _make_cache(_Cfg())
    lc = cache.layers[5]                        # kv_source, ratio 1 -> comp_cap 8+slack
    # compress cap = ceil(8/1)+8 = 16; feed 16 then overflow
    lc.append_compress(_row(16, 16))
    with pytest.raises(ValueError, match="exceed preallocated cap"):
        lc.append_compress(_row(1, 16))


# ---------------------------------------------------------------------------
# 3. in-place stability across N appends (no per-token realloc)
# ---------------------------------------------------------------------------
def test_inplace_stable_no_realloc_flat_memory(monkeypatch):
    cfg = _Cfg()
    _bounded_env(monkeypatch, maxkv=512)
    C.reset_kv_bounded_stats()
    cache = _make_cache(cfg)
    _prefill_all_lanes(cache, cfg, n=20)
    mx.eval([a for lc in cache.layers for a in lc.eval_backing()])

    # snapshot: realloc counts + raw-backing shapes after the (one-time) prealloc
    reallocs_before = {ln: C.kv_bounded_stats()[f"kv_realloc_{ln}"]
                       for ln in ("window", "compress", "index", "latent")}
    shapes_before = {}
    for L, lc in enumerate(cache.layers):
        shapes_before[(L, "window")] = tuple(lc._window.raw_backing().shape)
        if lc._compress_kv.raw_backing() is not None:
            shapes_before[(L, "compress")] = tuple(lc._compress_kv.raw_backing().shape)
        if lc._index_k.raw_backing() is not None:
            shapes_before[(L, "index")] = tuple(lc._index_k.raw_backing().shape)
        if lc.comp_state is not None:
            shapes_before[(L, "latent")] = tuple(
                lc.comp_state.raw_backings()[0].shape)

    mx.eval([])
    mx.reset_peak_memory()
    base_active = mx.get_active_memory()

    # N decode steps: one row per lane per step, all in place
    N = 64
    for _ in range(N):
        for L, lc in enumerate(cache.layers):
            lc.append_window(_row(1, cfg.head_dim))
            if L in set(cfg.kv_source_layer_ids):
                ratio = cfg.compress_ratios[L]
                if ratio > 1:
                    p = lc.comp_state.push(_row(1, cfg.head_dim), _row(1, cfg.head_dim))
                    if p.shape[1] > 0:
                        lc.append_compress(_row(p.shape[1], cfg.head_dim))
                        lc.append_index_k(_row(p.shape[1], cfg.index_head_dim))
                else:
                    lc.append_compress(_row(1, cfg.head_dim))
                    lc.append_index_k(_row(1, cfg.index_head_dim))
            lc.advance(1)
        mx.eval([a for lc in cache.layers for a in lc.eval_backing()])

    stats = C.kv_bounded_stats()
    # (a) NO lane reallocated during decode: the buffer's data pointer is stable.
    for ln in ("window", "compress", "index", "latent"):
        assert stats[f"kv_realloc_{ln}"] == reallocs_before[ln], (
            f"lane {ln} reallocated during decode "
            f"({reallocs_before[ln]} -> {stats[f'kv_realloc_{ln}']})")
        assert stats[f"kv_appends_{ln}"] > 0

    # (b) raw-backing capacity (shape) is unchanged -- same physical buffer.
    for L, lc in enumerate(cache.layers):
        assert tuple(lc._window.raw_backing().shape) == shapes_before[(L, "window")]
        if (L, "compress") in shapes_before:
            assert tuple(lc._compress_kv.raw_backing().shape) == shapes_before[(L, "compress")]
        if (L, "index") in shapes_before:
            assert tuple(lc._index_k.raw_backing().shape) == shapes_before[(L, "index")]
        if (L, "latent") in shapes_before:
            assert tuple(lc.comp_state.raw_backings()[0].shape) == shapes_before[(L, "latent")]

    # (c) active memory stayed flat across the N decode steps (donated in place):
    # far below what N per-token concatenations of the growing stores would cost.
    peak_growth = mx.get_peak_memory() - base_active
    one_window_buffer = 2 * cfg.window_size * cfg.head_dim * 4  # generous single-buffer bound
    assert peak_growth < 8 * one_window_buffer * cfg.num_hidden_layers, (
        f"peak grew {peak_growth} across {N} in-place decode steps (not flat)")


def test_plain_grow_memory_grows_unlike_bounded():
    """Control: the shipped ``_grow`` (concatenate) frontier's active memory grows
    with N; the bounded frontier's does not."""
    d = 16
    N = 200
    # plain: _grow every step
    plain = C.CompressorState(2, bounded=False)
    plain.push(_row(2, d), _row(2, d))
    mx.eval([plain.raw_kv, plain.raw_score]); mx.reset_peak_memory()
    base = mx.get_active_memory()
    for _ in range(N):
        plain.push(_row(1, d), _row(1, d))
        mx.eval([plain.raw_kv, plain.raw_score])
    plain_growth = mx.get_active_memory() - base

    # bounded: preallocated, in-place
    bounded = C.CompressorState(2, bounded=True, maxkv=N + 16)
    bounded.push(_row(2, d), _row(2, d))
    mx.eval(bounded.raw_backings()); mx.reset_peak_memory()
    base2 = mx.get_active_memory()
    for _ in range(N):
        bounded.push(_row(1, d), _row(1, d))
        mx.eval(bounded.raw_backings())
    bounded_growth = mx.get_active_memory() - base2

    assert bounded_growth < plain_growth, (
        f"bounded frontier should not grow like plain: bounded={bounded_growth} "
        f"plain={plain_growth}")


# ---------------------------------------------------------------------------
# 4. byte-identical outputs (tiny random model, same seed)
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


def _prefill_decode(model, prompt, steps):
    cache = model.make_cache()
    logits = model(mx.array([list(prompt)]), cache=cache)
    mx.eval(logits)
    outs = [logits[0, -1]]
    toks = []
    tok = int(mx.argmax(logits[0, -1]).item()); toks.append(tok)
    for _ in range(steps):
        logits = model(mx.array([[tok]]), cache=cache)
        mx.eval(logits)
        outs.append(logits[0, -1])
        tok = int(mx.argmax(logits[0, -1]).item()); toks.append(tok)
    return outs, toks, cache


def _clear_kv_envs(monkeypatch):
    for k in ("MTPLX_DSV41_KV_BOUNDED", "MTPLX_DSV41_KV_BOUNDED_MAXKV",
              "MTPLX_DSV41_WINDOW_RING", "MTPLX_DSV41_WINDOW_RING_MAX_VERIFY",
              "MTPLX_DSV41_WINDOW_RING_SLACK", "MTPLX_DSV41_WINDOW_RING_HEADROOM",
              "MTPLX_DSV41_WINDOW_RING_MAXKV", "MTPLX_DSV41_KV_CHUNK_GROW"):
        monkeypatch.delenv(k, raising=False)


def test_model_bounded_bit_identical_vs_legacy_selected_path(monkeypatch):
    model = _tiny_model()
    prompt = list(range(20)); steps = 180   # T ~ 200 -> the ring drops, latent grows
    # legacy (shipped): plain _grow, no ring, no bounded -- selected-key path
    _clear_kv_envs(monkeypatch)
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    base, base_toks, base_cache = _prefill_decode(model, prompt, steps)
    assert base_cache.layers[0]._kv_bounded is False

    # bounded: every lane preallocated to max_kv, in place
    _clear_kv_envs(monkeypatch)
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED", "1")
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED_MAXKV", "512")
    C.reset_kv_bounded_stats()
    bnd, bnd_toks, bnd_cache = _prefill_decode(model, prompt, steps)
    assert bnd_cache.layers[0]._kv_bounded is True
    assert isinstance(bnd_cache.layers[0]._window, C._WindowRing)
    assert bnd_cache.layers[0].window_drop_offset > 0, "ring should have dropped rows"

    assert base_toks == bnd_toks
    for i, (a, b) in enumerate(zip(base, bnd)):
        assert bool(mx.all(a == b).item()), f"bounded vs legacy differ at step {i}"


def test_model_bounded_bit_identical_vs_window_ring(monkeypatch):
    """Isolate the W107 prealloc/in-place changes: bounded vs the W80 ring arm share
    the SAME window ring, so they must be BIT-identical on BOTH paths (the only diff
    is the preallocated compress/index/latent backing == byte-identical views)."""
    model = _tiny_model()
    prompt = list(range(20)); steps = 150
    ring_tuning = {"MTPLX_DSV41_WINDOW_RING_MAX_VERIFY": "2",
                   "MTPLX_DSV41_WINDOW_RING_SLACK": "1",
                   "MTPLX_DSV41_WINDOW_RING_HEADROOM": "2"}

    _clear_kv_envs(monkeypatch)
    for k, v in ring_tuning.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING", "1")
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING_MAXKV", "512")
    ring, ring_toks, ring_cache = _prefill_decode(model, prompt, steps)
    assert ring_cache.layers[0]._window_ring is True
    assert ring_cache.layers[0]._kv_bounded is False

    _clear_kv_envs(monkeypatch)
    for k, v in ring_tuning.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED", "1")
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED_MAXKV", "512")
    bnd, bnd_toks, bnd_cache = _prefill_decode(model, prompt, steps)
    assert bnd_cache.layers[0]._kv_bounded is True

    assert ring_toks == bnd_toks
    for i, (a, b) in enumerate(zip(ring, bnd)):
        assert bool(mx.all(a == b).item()), f"bounded vs window_ring differ at step {i}"


# ---------------------------------------------------------------------------
# trim / rollback parity (bounded vs legacy)
# ---------------------------------------------------------------------------
def test_bounded_trim_rollback_parity(monkeypatch):
    cfg = _Cfg()

    def run(bounded):
        _clear_kv_envs(monkeypatch)
        if bounded:
            monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED", "1")
            monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED_MAXKV", "512")
        rng = np.random.default_rng(7)
        lc = C.LayerAttentionCache(window_size=8, compress_ratio=2, is_kv_source=True)
        for _ in range(10):
            lc.append_window(mx.array(rng.standard_normal((1, 1, 16)).astype(np.float32)))
            p = lc.comp_state.push(
                mx.array(rng.standard_normal((1, 1, 16)).astype(np.float32)),
                mx.array(rng.standard_normal((1, 1, 16)).astype(np.float32)))
            if p.shape[1] > 0:
                lc.append_compress(
                    mx.array(rng.standard_normal((1, p.shape[1], 16)).astype(np.float32)))
                lc.append_index_k(
                    mx.array(rng.standard_normal((1, p.shape[1], 12)).astype(np.float32)))
            lc.advance(1)
        m = lc.mark()
        for _ in range(4):
            lc.append_window(mx.array(rng.standard_normal((1, 1, 16)).astype(np.float32)))
            lc.comp_state.push(
                mx.array(rng.standard_normal((1, 1, 16)).astype(np.float32)),
                mx.array(rng.standard_normal((1, 1, 16)).astype(np.float32)))
            lc.advance(1)
        lc.rollback(m)
        return lc

    on, off = run(True), run(False)
    assert _eq(on.window, off.window)
    assert _eq(on.compress_kv, off.compress_kv)
    assert _eq(on.index_k, off.index_k)
    assert _eq(on.comp_state.raw_kv, off.comp_state.raw_kv)
    assert _eq(on.comp_state.raw_score, off.comp_state.raw_score)
    assert on.offset == off.offset
    assert on.comp_state.n_fed == off.comp_state.n_fed


def test_bounded_trim_parity(monkeypatch):
    cfg = _Cfg()

    def run(bounded):
        _clear_kv_envs(monkeypatch)
        if bounded:
            monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED", "1")
            monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED_MAXKV", "256")
        rng = np.random.default_rng(3)
        lc = C.LayerAttentionCache(window_size=8, compress_ratio=2, is_kv_source=True)
        for _ in range(12):
            lc.append_window(mx.array(rng.standard_normal((1, 1, 16)).astype(np.float32)))
            p = lc.comp_state.push(
                mx.array(rng.standard_normal((1, 1, 16)).astype(np.float32)),
                mx.array(rng.standard_normal((1, 1, 16)).astype(np.float32)))
            if p.shape[1] > 0:
                lc.append_compress(
                    mx.array(rng.standard_normal((1, p.shape[1], 16)).astype(np.float32)))
                lc.append_index_k(
                    mx.array(rng.standard_normal((1, p.shape[1], 12)).astype(np.float32)))
            lc.advance(1)
        lc.trim(3)
        return lc

    on, off = run(True), run(False)
    assert _eq(on.window, off.window)
    assert _eq(on.compress_kv, off.compress_kv)
    assert _eq(on.index_k, off.index_k)
    assert _eq(on.comp_state.raw_kv, off.comp_state.raw_kv)
    assert on.offset == off.offset == 9


# ---------------------------------------------------------------------------
# 5. counters increment + engagement
# ---------------------------------------------------------------------------
def test_counters_increment_and_reset(monkeypatch):
    cfg = _Cfg()
    _bounded_env(monkeypatch, maxkv=256)
    C.reset_kv_bounded_stats()
    z = C.kv_bounded_stats()
    assert z["enabled"] is False and z["layers_bounded"] == 0
    for ln in ("window", "compress", "index", "latent"):
        assert z[f"kv_appends_{ln}"] == 0 and z[f"kv_realloc_{ln}"] == 0

    cache = _make_cache(cfg)
    _prefill_all_lanes(cache, cfg, n=16)
    # a few decode steps to drive in-place writes
    for _ in range(20):
        for L, lc in enumerate(cache.layers):
            lc.append_window(_row(1, cfg.head_dim))
            if L in set(cfg.kv_source_layer_ids):
                ratio = cfg.compress_ratios[L]
                if ratio > 1:
                    p = lc.comp_state.push(_row(1, cfg.head_dim), _row(1, cfg.head_dim))
                    if p.shape[1] > 0:
                        lc.append_compress(_row(p.shape[1], cfg.head_dim))
                        lc.append_index_k(_row(p.shape[1], cfg.index_head_dim))
                else:
                    lc.append_compress(_row(1, cfg.head_dim))
                    lc.append_index_k(_row(1, cfg.index_head_dim))
            lc.advance(1)

    s = C.kv_bounded_stats()
    assert s["enabled"] is True
    assert s["layers_bounded"] == cfg.num_hidden_layers
    assert s["maxkv"] == 256
    assert s["alloc_bytes"] > 0
    # every lane engaged: one-time prealloc (realloc) + in-place decode writes
    assert s["kv_appends_window"] > 0 and s["kv_realloc_window"] >= 1
    assert s["kv_appends_compress"] > 0 and s["kv_realloc_compress"] >= 1
    assert s["kv_appends_index"] > 0 and s["kv_realloc_index"] >= 1
    assert s["kv_appends_latent"] > 0 and s["kv_realloc_latent"] >= 1
    # W107F rename: the per-lane append counter is ``kv_appends_<lane>`` now; the old
    # ``kv_inplace_writes_<lane>`` name must be GONE from the stats snapshot.
    for ln in ("window", "compress", "index", "latent"):
        assert f"kv_appends_{ln}" in s
        assert f"kv_inplace_writes_{ln}" not in s
    for ln in ("window", "compress", "index", "latent"):
        assert s[f"rows_{ln}"] > 0

    C.reset_kv_bounded_stats()
    assert C.kv_bounded_stats()["layers_bounded"] == 0


def test_bounded_engages_when_env_set_after_import(monkeypatch):
    """The flag is read at construction (per request), not frozen at import -- a
    make_cache after the env is stamped picks the bounded lanes."""
    _clear_kv_envs(monkeypatch)
    cfg = _Cfg()
    off = _make_cache(cfg)
    assert all(lc._kv_bounded is False for lc in off.layers)

    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED", "1")
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED_MAXKV", "128")
    on = _make_cache(cfg)
    assert all(lc._kv_bounded for lc in on.layers)
    assert isinstance(on.layers[0]._window, C._WindowRing)
    assert isinstance(on.layers[2]._compress_kv, C._GrowBuffer)
    assert on.layers[2].comp_state is not None and on.layers[2].comp_state._bounded


def test_bounded_maxkv_falls_back_to_window_ring_maxkv(monkeypatch):
    _clear_kv_envs(monkeypatch)
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED", "1")
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING_MAXKV", "96")  # no KV_BOUNDED_MAXKV
    assert C._kv_bounded_maxkv() == 96
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED_MAXKV", "77")   # explicit wins
    assert C._kv_bounded_maxkv() == 77


# ---------------------------------------------------------------------------
# Review HIGH-1: DSpark trim/rollback must NOT reallocate the preallocated
# compress/index lanes (length-only truncate, stable backing).
# ---------------------------------------------------------------------------
def _drive_verify_cycle(lc, cfg, k_plus_1):
    """One DSpark-style verify block: append K+1 rows through window + compressor +
    compress/index, advance."""
    lc.append_window(_row(k_plus_1, cfg.head_dim))
    p = lc.comp_state.push(_row(k_plus_1, cfg.head_dim), _row(k_plus_1, cfg.head_dim))
    if p.shape[1] > 0:
        lc.append_compress(_row(p.shape[1], cfg.head_dim))
        lc.append_index_k(_row(p.shape[1], cfg.index_head_dim))
    lc.advance(k_plus_1)


def test_dspark_trim_no_realloc_compress_index(monkeypatch):
    cfg = _Cfg()
    _bounded_env(monkeypatch, maxkv=512)
    C.reset_kv_bounded_stats()
    lc = C.LayerAttentionCache(window_size=cfg.window_size, compress_ratio=2,
                               is_kv_source=True)
    # prefill 40 rows (fills the buffers once)
    lc.append_window(_row(40, cfg.head_dim))
    lc.comp_state.push(_row(40, cfg.head_dim), _row(40, cfg.head_dim))
    lc.append_compress(_row(20, cfg.head_dim))
    lc.append_index_k(_row(20, cfg.index_head_dim))
    lc.advance(40)

    s0 = C.kv_bounded_stats()
    realloc_c0, realloc_i0 = s0["kv_realloc_compress"], s0["kv_realloc_index"]
    alloc0 = s0["alloc_bytes"]
    # backing capacity (shape) is the stable-buffer proxy: mx.slice_update returns a
    # new mx.array WRAPPER each in-place write (functional API) even when it donates
    # the same memory, so id() is not stable; a fresh mx.zeros (a realloc) would
    # change the shape and bump kv_realloc_* / alloc_bytes -- those are the invariants.
    shape_c0 = tuple(lc._compress_kv.raw_backing().shape)
    shape_i0 = tuple(lc._index_k.raw_backing().shape)

    # 8 rejected DSpark cycles: append K+1=4, then trim 2 (reject 2 of the block)
    for _ in range(8):
        _drive_verify_cycle(lc, cfg, k_plus_1=4)
        lc.trim(2)

    s1 = C.kv_bounded_stats()
    assert s1["kv_realloc_compress"] == realloc_c0, (
        f"compress reallocated over verify cycles "
        f"({realloc_c0} -> {s1['kv_realloc_compress']})")
    assert s1["kv_realloc_index"] == realloc_i0, (
        f"index reallocated over verify cycles "
        f"({realloc_i0} -> {s1['kv_realloc_index']})")
    # the backing capacity is unchanged (length-only truncate, not a fresh set())
    assert tuple(lc._compress_kv.raw_backing().shape) == shape_c0
    assert tuple(lc._index_k.raw_backing().shape) == shape_i0
    # alloc_bytes did not grow with the cycles (no fresh mx.zeros per reject)
    assert s1["alloc_bytes"] == alloc0, (
        f"alloc_bytes grew over verify cycles ({alloc0} -> {s1['alloc_bytes']})")


def test_rollback_no_realloc_compress_index(monkeypatch):
    cfg = _Cfg()
    _bounded_env(monkeypatch, maxkv=512)
    C.reset_kv_bounded_stats()
    lc = C.LayerAttentionCache(window_size=cfg.window_size, compress_ratio=2,
                               is_kv_source=True)
    lc.append_window(_row(40, cfg.head_dim))
    lc.comp_state.push(_row(40, cfg.head_dim), _row(40, cfg.head_dim))
    lc.append_compress(_row(20, cfg.head_dim))
    lc.append_index_k(_row(20, cfg.index_head_dim))
    lc.advance(40)

    realloc_c0 = C.kv_bounded_stats()["kv_realloc_compress"]
    shape_c0 = tuple(lc._compress_kv.raw_backing().shape)

    for _ in range(8):
        m = lc.mark()
        _drive_verify_cycle(lc, cfg, k_plus_1=4)
        lc.rollback(m)

    s1 = C.kv_bounded_stats()
    assert s1["kv_realloc_compress"] == realloc_c0
    assert tuple(lc._compress_kv.raw_backing().shape) == shape_c0
    # rollback restored the exact pre-cycle group count
    assert int(lc.compress_kv.shape[1]) == 20


# ---------------------------------------------------------------------------
# Review HIGH-2: the served path must give the bounded lanes a max_kv.
# ---------------------------------------------------------------------------
def test_server_registers_kv_bounded_lever_keys():
    """The two W107 lever env keys are in the served-log snapshot list (a merge
    worker adds RUNNER/DRAFT_HEAD_BF16 to the same list -- ours must survive)."""
    from mtplx.server import openai as O
    assert "MTPLX_DSV41_KV_BOUNDED" in O._DSV41_LEVER_ENV_KEYS
    assert "MTPLX_DSV41_KV_BOUNDED_MAXKV" in O._DSV41_LEVER_ENV_KEYS
    resolved = O._dsv41_resolved_lever_env(
        {"MTPLX_DSV41_KV_BOUNDED": "1", "MTPLX_DSV41_KV_BOUNDED_MAXKV": "17408"})
    assert resolved["MTPLX_DSV41_KV_BOUNDED"] == "1"
    assert resolved["MTPLX_DSV41_KV_BOUNDED_MAXKV"] == "17408"


def test_server_plumbed_maxkv_bounds_the_cache(monkeypatch):
    """The env the server stamps from max_live_kv_tokens is honoured at cache
    construction: every lane preallocates to that cap (this is what the served path
    now delivers -- before HIGH-2 the server stamped nothing and the lanes fell back
    to geometric growth)."""
    _clear_kv_envs(monkeypatch)
    # what the server writes at setup: KV_BOUNDED on + MAXKV == max_live_kv_tokens
    served_max_live_kv = 4096
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED", "1")
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED_MAXKV", str(served_max_live_kv))
    assert C._kv_bounded_maxkv() == served_max_live_kv

    cache = _make_cache(_Cfg())
    assert all(lc._kv_bounded for lc in cache.layers)
    # a ratio-2 kv-source layer preallocates compress to ceil(max_kv/2)+slack and the
    # latent frontier to max_kv+slack -- i.e. it is bounded, not geometric-from-256.
    lc = cache.layers[2]
    lc.append_compress(_row(1, _Cfg.head_dim))
    assert int(lc._compress_kv.raw_backing().shape[1]) == \
        C._bounded_comp_cap(served_max_live_kv, 2)
    lc.comp_state.push(_row(1, _Cfg.head_dim), _row(1, _Cfg.head_dim))
    assert int(lc.comp_state.raw_backings()[0].shape[1]) == \
        C._bounded_latent_cap(served_max_live_kv)


def test_server_hard_sets_maxkv_over_stale_env(monkeypatch):
    """Review MEDIUM-B: the server HARD-SETS MTPLX_DSV41_KV_BOUNDED_MAXKV from the
    authoritative max_live_kv_tokens, so a stale env (shell / profile / earlier ab run)
    does NOT survive (setdefault would have let it win, breaking assert_can_admit or
    over-preallocating)."""
    from mtplx.server.openai import _plumb_kv_bounded_maxkv
    _clear_kv_envs(monkeypatch)
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED_MAXKV", "999")   # stale/smaller value
    _plumb_kv_bounded_maxkv(4096)                               # server's plumbing
    assert C._kv_bounded_maxkv() == 4096, "stale env survived the server hard-set"
    # a larger stale value is also overridden
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED_MAXKV", "99999")
    _plumb_kv_bounded_maxkv(4096)
    assert C._kv_bounded_maxkv() == 4096


# ---------------------------------------------------------------------------
# Review MEDIUM-1: the window ring must shrink back to its base phys_cap after a
# chunked prefill inflates it (formula and steady allocation must agree).
# ---------------------------------------------------------------------------
def test_window_ring_shrinks_back_after_chunked_prefill():
    ring = C._WindowRing(8, 8, 8, 64)          # base_phys_cap = 8+8+8+64 = 88
    base = ring._base_phys_cap
    assert base == 88
    rng = np.random.default_rng(1)
    # chunked prefill: 2048 rows in 256-row chunks -> inflates phys_cap transiently
    for _ in range(8):
        ring.append(mx.array(rng.standard_normal((1, 256, 16)).astype(np.float32)))
    assert ring.phys_cap > base, "prefill should transiently grow the ping-pong buffers"
    # decode steps: the ring must SHRINK back to base (old bug: phys_cap stuck at ~263)
    for _ in range(300):
        ring.append(mx.array(rng.standard_normal((1, 1, 16)).astype(np.float32)))
    assert ring.phys_cap == base, f"ring did not shrink back: phys_cap={ring.phys_cap}"
    assert int(ring.raw_backing().shape[1]) == base
    # live window bytes now match the formula's window term (2 x base x head_dim x dtype)
    live_window = int(ring._bufs[ring._cur].nbytes)
    assert live_window == base * 16 * 4                     # fp32 rows here


def test_window_formula_matches_steady_allocation_after_chunked_prefill(monkeypatch):
    """End to end: after a chunked prefill the summed live window bytes equal the
    kv_bytes formula's window term (the doc §4 number), not the inflated transient."""
    cfg = _Cfg()
    _bounded_env(monkeypatch, maxkv=4096)
    C.reset_kv_bounded_stats()
    cache = _make_cache(cfg)
    rng = np.random.default_rng(2)
    # prefill the window lane of every layer in 64-row chunks (> base phys_cap 88? no;
    # use 128-row chunks to exceed base and force a transient grow)
    for _ in range(6):
        for lc in cache.layers:
            lc.append_window(mx.array(rng.standard_normal((1, 128, cfg.head_dim)).astype(np.float32)))
    # decode until every ring compacts back to base
    for _ in range(400):
        for lc in cache.layers:
            lc.append_window(mx.array(rng.standard_normal((1, 1, cfg.head_dim)).astype(np.float32)))
    for lc in cache.layers:
        assert lc._window.phys_cap == lc._window._base_phys_cap
    live_window = sum(int(lc._window._bufs[0].nbytes) + int(lc._window._bufs[1].nbytes)
                      for lc in cache.layers)
    formula_window = C.kv_bytes_breakdown_at_max_kv(
        cfg, 4096, model_dtype_bytes=4, store_dtype_bytes=4)["window"]
    assert live_window == formula_window, (
        f"steady window bytes {live_window} != formula {formula_window}")


# ---------------------------------------------------------------------------
# Review MEDIUM-3: rollback across a ring compaction must fail loud, not silently
# lose in-window rows (pre-existing W80 divergence).
# ---------------------------------------------------------------------------
def test_window_ring_deep_rollback_across_compaction_raises():
    ring = C._WindowRing(8, 2, 1, 2)   # window_size 8, cap_keep 11, phys_cap 13
    rng = np.random.default_rng(4)
    for _ in range(100):
        ring.append(mx.array(rng.standard_normal((1, 1, 16)).astype(np.float32)))
    assert ring.drop_offset > 0, "need a compaction to have advanced the drop frontier"
    L = ring.logical_len()
    # a shallow rollback within the resident window is fine (DSpark 1 cycle / device
    # route depth 1) -- the raise fires BEFORE any mutation, so state is intact.
    ring.truncate_to_length(L - 1)
    assert ring.rows() == 12
    # a deep rollback whose window reaches below the drop frontier must RAISE
    with pytest.raises(ValueError, match="below the drop frontier"):
        ring.truncate_to_length(20)
    # full reset to 0 is allowed (no history needed)
    ring.truncate_to_length(0)
    assert ring.rows() == 0 and ring.drop_offset == 0


def test_layer_cache_deep_rollback_across_compaction_raises(monkeypatch):
    """The trim/rollback seam surfaces the raise: a mark, then enough decode to
    compact past the mark's window, then rollback -> ValueError (not silent)."""
    _clear_kv_envs(monkeypatch)
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED", "1")
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED_MAXKV", "4096")
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING_HEADROOM", "2")   # tiny -> compaction fast
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING_MAX_VERIFY", "2")
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING_SLACK", "1")
    lc = C.LayerAttentionCache(window_size=8, compress_ratio=0, is_kv_source=False)
    for _ in range(20):
        lc.append_window(_row(1, 16)); lc.advance(1)
    m = lc.mark()                       # mark at offset 20
    for _ in range(60):                 # decode far past the mark's window -> compactions
        lc.append_window(_row(1, 16)); lc.advance(1)
    assert lc._window.drop_offset > 20
    with pytest.raises(ValueError, match="below the drop frontier"):
        lc.rollback(m)


def test_shallow_rollback_over_ring_is_safe(monkeypatch):
    """A depth-1 rollback (DSpark accept/reject one cycle) never trips MEDIUM-3."""
    _clear_kv_envs(monkeypatch)
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED", "1")
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED_MAXKV", "4096")
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING_HEADROOM", "2")
    lc = C.LayerAttentionCache(window_size=8, compress_ratio=0, is_kv_source=False)
    for _ in range(60):
        lc.append_window(_row(1, 16)); lc.advance(1)
    m = lc.mark()
    lc.append_window(_row(1, 16)); lc.advance(1)   # one speculative step
    lc.rollback(m)                                  # reject it -- must not raise
    assert lc.offset == 60


# ---------------------------------------------------------------------------
# Review LOW-1: an over-cap forward must fail BEFORE touching any lane (no
# half-updated cache).
# ---------------------------------------------------------------------------
def test_assert_can_admit_raises_without_mutating(monkeypatch):
    cfg = _Cfg()
    _bounded_env(monkeypatch, maxkv=64)
    cache = _make_cache(cfg)
    _prefill_all_lanes(cache, cfg, n=20)          # offset 20 on every entry
    offs = [lc.offset for lc in cache.layers]
    wlens = [lc.window_len() for lc in cache.layers]
    clens = [_r(lc.compress_kv) for lc in cache.layers]
    # 60 more tokens overflows the latent cap (64 + slack 8 = 72 < 20 + 60 = 80)
    with pytest.raises(ValueError, match="cannot admit"):
        cache.assert_can_admit(60)
    # nothing moved -- the pre-check touched no lane
    assert [lc.offset for lc in cache.layers] == offs
    assert [lc.window_len() for lc in cache.layers] == wlens
    assert [_r(lc.compress_kv) for lc in cache.layers] == clens
    # a within-cap admission does not raise
    cache.assert_can_admit(40)                     # 20 + 40 = 60 <= 72


def _r(a):
    return 0 if a is None else int(a.shape[1])


def test_over_cap_model_forward_raises_and_leaves_cache_clean(monkeypatch):
    """An over-cap prefill fails via the backbone pre-check with the cache untouched
    (offset 0, no lanes written) -- not half-updated."""
    _clear_kv_envs(monkeypatch)
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED", "1")
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED_MAXKV", "16")   # tiny cap
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    model = _tiny_model()
    cache = model.make_cache()
    prompt = mx.array([list(range(40))])           # 40 > 16 + slack -> over-cap
    with pytest.raises(ValueError, match="cannot admit"):
        model(prompt, cache=cache)
    # the pre-check ran before any layer appended: every entry is still empty
    assert all(lc.offset == 0 for lc in cache.layers)
    assert all(lc.window_len() == 0 for lc in cache.layers)


# ---------------------------------------------------------------------------
# Review LOW-2: truncate_to(0) must keep the preallocated buffer (no realloc on
# the next append).
# ---------------------------------------------------------------------------
def test_truncate_to_zero_keeps_prealloc():
    gb = C._GrowBuffer(bounded_cap=32, counter_lane="compress")
    C.reset_kv_bounded_stats()
    gb.append(_row(10, 8))                       # one prealloc
    assert C.kv_bounded_stats()["kv_realloc_compress"] == 1
    cap_shape = tuple(gb.raw_backing().shape)
    gb.truncate_to(0)                            # rollback to empty
    assert gb.rows() == 0
    assert gb.raw_backing() is not None          # buffer kept (LOW-2)
    assert tuple(gb.raw_backing().shape) == cap_shape
    gb.append(_row(5, 8))                        # next append must reuse, not realloc
    s = C.kv_bounded_stats()
    assert s["kv_realloc_compress"] == 1, "truncate_to(0) dropped the prealloc"
    assert s["kv_appends_compress"] >= 1
    assert gb.rows() == 5


def test_full_trim_to_zero_no_realloc_via_cache(monkeypatch):
    """A whole-entry trim to offset 0 keeps every bounded lane's prealloc."""
    cfg = _Cfg()
    _bounded_env(monkeypatch, maxkv=256)
    C.reset_kv_bounded_stats()
    lc = C.LayerAttentionCache(window_size=8, compress_ratio=2, is_kv_source=True)
    lc.append_window(_row(20, 16))
    lc.comp_state.push(_row(20, 16), _row(20, 16))
    lc.append_compress(_row(10, 16))
    lc.append_index_k(_row(10, 12))
    lc.advance(20)
    reallocs = {ln: C.kv_bounded_stats()[f"kv_realloc_{ln}"]
                for ln in ("compress", "index", "latent")}
    lc.trim(20)                                  # trim the whole entry to empty
    assert lc.offset == 0
    for ln in ("compress", "index", "latent"):
        assert C.kv_bounded_stats()[f"kv_realloc_{ln}"] == reallocs[ln], (
            f"{ln} reallocated on trim-to-0")


# ---------------------------------------------------------------------------
# Review MEDIUM-A: the formula (defaults) matches alloc_bytes on a bf16 model
# (layer-0 window bf16, everything else fp32) -- and the ab receipt gate.
# ---------------------------------------------------------------------------
def _tiny_model_bf16():
    from mlx.utils import tree_flatten, tree_unflatten
    model = _tiny_model()
    new = [(n, a.astype(mx.bfloat16)) for n, a in tree_flatten(model.parameters())]
    model.update(tree_unflatten(new))
    mx.eval(model.parameters())
    return model


def test_medium_a_formula_matches_alloc_on_bf16_model(monkeypatch):
    _clear_kv_envs(monkeypatch)
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED", "1")
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED_MAXKV", "256")
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    model = _tiny_model_bf16()
    C.reset_kv_bounded_stats()
    cache = model.make_cache()
    logits = model(mx.array([list(range(20))]), cache=cache); mx.eval(logits)
    tok = int(mx.argmax(logits[0, -1]).item())
    for _ in range(10):
        logits = model(mx.array([[tok]]), cache=cache); mx.eval(logits)
        tok = int(mx.argmax(logits[0, -1]).item())

    alloc = C.kv_bounded_stats()["alloc_bytes"]
    # DEFAULTS model the bf16 reality: layer-0 window bf16 (2), everything else fp32 (4)
    formula = C.kv_bytes_at_max_kv(model.args, 256)
    assert alloc == formula, f"alloc {alloc} != formula {formula} (bf16 model)"
    # the ab-receipt gate is this equality (exact here: prefill 20 < window phys_cap,
    # so no transient window grow -> exactly one ring init per layer, no compaction
    # realloc; the counter is process-global across layers).
    assert C.kv_bounded_stats()["kv_realloc_window"] == len(cache.layers)
    # and the layer-0 window really is the only bf16 store
    assert cache.layers[0]._window.raw_backing().dtype == mx.bfloat16
    assert cache.layers[1]._window.raw_backing().dtype == mx.float32
    assert cache.layers[2]._compress_kv.raw_backing().dtype == mx.float32


# ---------------------------------------------------------------------------
# Review HIGH-A: the session-bank prefix restore must MISS cleanly (trim returns
# != delta, no mutation) when the divergence exceeds the ring's recoverable depth,
# so the served single-request lane falls back to a cold prefill (not a ValueError).
# ---------------------------------------------------------------------------
def _prefill_ring_cache(monkeypatch, tokens, *, maxkv=1024):
    _clear_kv_envs(monkeypatch)
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED", "1")
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED_MAXKV", str(maxkv))
    cfg = _Cfg()
    cache = _make_cache(cfg)
    # decode-style 1-row appends so the ring compacts and the drop frontier advances
    for _ in range(tokens):
        for L, lc in enumerate(cache.layers):
            lc.append_window(_row(1, cfg.head_dim))
            if L in set(cfg.kv_source_layer_ids):
                ratio = cfg.compress_ratios[L]
                if ratio > 1:
                    p = lc.comp_state.push(_row(1, cfg.head_dim), _row(1, cfg.head_dim))
                    if p.shape[1] > 0:
                        lc.append_compress(_row(p.shape[1], cfg.head_dim))
                        lc.append_index_k(_row(p.shape[1], cfg.index_head_dim))
                else:
                    lc.append_compress(_row(1, cfg.head_dim))
                    lc.append_index_k(_row(1, cfg.index_head_dim))
        cache.advance(1)
    return cache


def test_session_bank_deep_prefix_restore_misses_cleanly(monkeypatch):
    from mtplx.session_bank import _trim_cache_ref_to_prefix
    cache = _prefill_ring_cache(monkeypatch, 600)
    assert cache.layers[0]._window.drop_offset > 0, "ring must have compacted"
    offs = [lc.offset for lc in cache.layers]
    wlens = [lc.window_len() for lc in cache.layers]
    clens = [_r(lc.compress_kv) for lc in cache.layers]

    # a deep prefix restore (divergence ~300 >> the ~24-row recoverable window) must
    # MISS -- returns False WITHOUT raising and WITHOUT mutating the cache
    assert _trim_cache_ref_to_prefix(cache, 300) is False
    assert [lc.offset for lc in cache.layers] == offs, "cache mutated on a deep miss"
    assert [lc.window_len() for lc in cache.layers] == wlens
    assert [_r(lc.compress_kv) for lc in cache.layers] == clens


def test_session_bank_shallow_prefix_restore_hits(monkeypatch):
    """A within-window divergence still restores (trim returns delta) -- the miss
    guard must not break the common short-tail regenerate case."""
    from mtplx.session_bank import _trim_cache_ref_to_prefix
    cache = _prefill_ring_cache(monkeypatch, 600)
    # prefix 599 -> divergence of 2 tokens, well inside the resident window
    assert _trim_cache_ref_to_prefix(cache, 599) is True
    assert all(lc.offset == 598 for lc in cache.layers)  # target_offset = prefix - 1


def test_rollback_deep_leaves_engram_untouched(monkeypatch):
    """Review HIGH-A: a failed deep rollback (window raises) must not have already
    rewound the engram -- the ring truncate now runs before the engram trim."""
    _clear_kv_envs(monkeypatch)
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED", "1")
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED_MAXKV", "1024")
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING_HEADROOM", "2")

    class _EngramSpy:
        def __init__(self): self.trims = []
        def trim(self, n): self.trims.append(int(n))

    spy = _EngramSpy()
    lc = C.LayerAttentionCache(window_size=8, compress_ratio=0, is_kv_source=False,
                               engram_state=spy)
    for _ in range(20):
        lc.append_window(_row(1, 16)); lc.advance(1)
    m = lc.mark()
    for _ in range(60):
        lc.append_window(_row(1, 16)); lc.advance(1)
    assert lc._window.drop_offset > 20
    with pytest.raises(ValueError, match="below the drop frontier"):
        lc.rollback(m)
    assert spy.trims == [], "engram was trimmed before the window raise (half-rewound)"


# ---------------------------------------------------------------------------
# Review LOW-A: the over-cap admission check must be hoisted to __call__ so a
# chunk-major prefill fails at offset 0, not at span 2 with a partial prefix.
# ---------------------------------------------------------------------------
def test_over_cap_chunk_major_prefill_raises_at_offset_zero(monkeypatch):
    _clear_kv_envs(monkeypatch)
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED", "1")
    monkeypatch.setenv("MTPLX_DSV41_KV_BOUNDED_MAXKV", "40")   # tiny cap
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    model = _tiny_model()
    cache = model.make_cache()
    prompt = mx.array([list(range(100))])                      # 100 >> 40 + slack
    # chunk 32 forces the chunk-major driver (multiple _forward_span spans); the
    # whole-prompt pre-check in __call__ must raise BEFORE span 0 writes anything.
    with pytest.raises(ValueError, match="cannot admit"):
        model(prompt, cache=cache, prefill_chunk=32)
    assert all(lc.offset == 0 for lc in cache.layers), "a chunk was written before the raise"
    assert all(lc.window_len() == 0 for lc in cache.layers)


# ---------------------------------------------------------------------------
# Review round-2 finding 2: the cell16k_ring CONTROL arm is frozen (no kv_bounded);
# its env set must equal the frozen window-39 basis.  kv_bounded is a CANDIDATE.
# ---------------------------------------------------------------------------
def _load_ab_module():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "scripts" / "deepseek_v41" / "ab_decode_env_levers.py"
    spec = importlib.util.spec_from_file_location("_ab_w107", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _frozen_control_env():
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent / "fixtures" / "dsv41_window39_control_arm_env.json"
    return json.loads(p.read_text())["arm_env_set"]


def test_control_arm_frozen_matches_window39():
    ab = _load_ab_module()
    frozen = _frozen_control_env()
    preset = ab.ARM_PRESETS["cell16k_ring"]
    live_set = {k: v for k, v in preset.items() if v is not None}
    assert live_set == frozen, (
        "cell16k_ring control env diverged from the frozen window-39 basis:\n"
        f"  extra:   {sorted(set(live_set) - set(frozen))}\n"
        f"  missing: {sorted(set(frozen) - set(live_set))}\n"
        f"  changed: {[k for k in live_set if k in frozen and live_set[k] != frozen[k]]}"
    )
    # the control must NOT carry the bounded lever (it is a candidate)
    assert preset.get(ab.KV_BOUNDED_ENV) is None
    assert preset.get(ab.KV_BOUNDED_MAXKV_ENV) is None


def test_bounded_candidate_is_control_plus_only_kv_bounded():
    """cell16k_ring_bounded is the clean bounded candidate: EXACTLY the frozen control
    set plus MTPLX_DSV41_KV_BOUNDED=1 (so the A/B isolates only the bounded lever)."""
    ab = _load_ab_module()
    frozen = _frozen_control_env()
    cand = ab.ARM_PRESETS["cell16k_ring_bounded"]
    cand_set = {k: v for k, v in cand.items() if v is not None}
    assert cand.get(ab.KV_BOUNDED_ENV) == "1"
    # difference from the frozen control is exactly the one bounded key
    assert set(cand_set) - set(frozen) == {ab.KV_BOUNDED_ENV}
    assert {k: cand_set[k] for k in frozen} == frozen


# ---------------------------------------------------------------------------
# Review round-2 finding 1: the KV donation probe must DISCRIMINATE a donating
# in-place write from an O(T) copy (validated on CPU; the orchestrator runs --gpu).
# ---------------------------------------------------------------------------
def _load_probe_module():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "scripts" / "deepseek_v41" / "kv_donation_probe.py"
    spec = importlib.util.spec_from_file_location("_kv_probe", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows_by(rows, prim, hold_view):
    return sorted(
        (r for r in rows if r.get("supported") and r["primitive"] == prim
         and r.get("hold_view") == hold_view),
        key=lambda r: r["N"])


def test_kv_donation_probe_rebind_donates_on_cpu():
    """Round-4 fixed probe: the buffer POINTER is the signal.  In the cache's rebind
    pattern (no lingering alias) mx.slice_update DONATES (0 pointer flips) -- the
    re-review's correction of round-3's false 'copy' verdict; holding a view at the
    write flips the pointer every append (copy)."""
    probe = _load_probe_module()
    res = probe.run(sizes=[2048, 8192], dim=64, dtype_name="fp32", reps=6, use_gpu=False)
    rows = res["rows"]
    prims = {r["primitive"] for r in rows}
    assert {"concat", "slice_update", "setitem", "put_along_axis"} <= prims

    su = _rows_by(rows, "slice_update", False)[-1]
    if not su.get("ptr_available"):
        pytest.skip("MLX build does not expose the buffer-protocol data pointer")
    # slice_update donates in the rebind pattern: pointer stable, ~no allocation
    assert su["ptr_flips"] == 0, "slice_update did not donate in the rebind pattern"
    assert su["peak_delta_bytes"] < 0.5 * su["buffer_bytes"]
    # __setitem__ is identical (also donates) -- so the round-3 switch bought nothing
    si = _rows_by(rows, "setitem", False)[-1]
    assert si["ptr_flips"] == 0
    # holding a live view at the write defeats donation (copies) for BOTH primitives
    assert _rows_by(rows, "slice_update", True)[-1]["ptr_flips"] > 0
    assert _rows_by(rows, "setitem", True)[-1]["ptr_flips"] > 0
    assert "DONATE(ptr-stable)" in res["verdict"] and "COPY(ptr-flips" in res["verdict"]


def test_kv_donation_probe_self_test_and_shape():
    probe = _load_probe_module()
    st = probe.self_test()
    if st.get("ptr_available"):
        assert st["pass"] is True
        assert st["rebind_ptr_flips"] == 0 and st["held_view_ptr_flips"] > 0
    res = probe.run(sizes=[256, 512], dim=32, dtype_name="fp32", reps=4, use_gpu=False)
    for r in res["rows"]:
        if not r.get("supported"):
            continue
        for k in ("ms_per_append", "peak_delta_bytes", "buffer_bytes"):
            assert k in r
        if r["primitive"] != "concat":               # concat is the baseline (no rebind)
            assert "ptr_flips" in r


# ---------------------------------------------------------------------------
# Review round-4: round-3's in-place __setitem__ was REVERTED.  The re-review proved
# at the mlx-fork source that mx.slice_update donates in the cache's rebind pattern
# (self._buf = write(self._buf) drops the old descriptor -> is_donatable), so it is
# pointer-stable and identical to __setitem__; and in-place is UNSAFE because view()
# can return the buffer IDENTITY (_len == cap), so a held lc.window would be mutated
# by the next append.  These tests lock in the safe reverted primitive.
# ---------------------------------------------------------------------------
def test_inplace_write_machinery_removed():
    """The in-place write lever/helpers are gone; lanes carry no _inplace flag."""
    assert not hasattr(C, "_kv_inplace_write_enabled")
    assert not hasattr(C, "_inplace_row_write")
    assert not hasattr(C, "_KV_INPLACE_WRITE_ENV")
    gb = C._GrowBuffer(bounded_cap=8, counter_lane="compress")
    assert not hasattr(gb, "_inplace")
    ring = C._WindowRing(8, 2, 1, 2, counter_lane="window")
    assert not hasattr(ring, "_inplace")


def test_slice_update_append_does_not_mutate_returned_view():
    """The hazard round-3's in-place write would have caused: view() returns the buffer
    IDENTITY when _len == cap, so a held view must NOT change when the next append
    writes.  mx.slice_update (functional; copies when the old buffer is still referenced
    by the view) keeps the held view intact -- the safety the revert restores."""
    gb = C._GrowBuffer(bounded_cap=4, counter_lane="compress")
    gb.append(_row(4, 8))                     # _len == cap 4 -> view() is buffer IDENTITY
    v = gb.view()
    assert v is gb.raw_backing(), "precondition: view() returns the buffer identity here"
    snap = mx.array(v); mx.eval(v, snap)
    gb.truncate_to(3)
    gb.append(_row(1, 8))                     # next append while v still references _buf
    mx.eval(v, gb.view())
    assert bool(mx.all(v == snap).item()), "a previously-returned view was mutated by append"


def test_bounded_lane_backing_pointer_stable_rebind(monkeypatch):
    """mx.slice_update donates in the rebind pattern: raw_backing()'s data pointer is
    stable across steady in-cap decode appends (0 flips), matching the source-level
    finding.  Guarded: skipped if MLX refuses the buffer-protocol pointer."""
    _bounded_env(monkeypatch, maxkv=256)
    lc = C.LayerAttentionCache(window_size=8, compress_ratio=1, is_kv_source=True)
    lc.append_compress(_row(4, 16)); lc.advance(4)   # allocate the compress buffer
    p0 = C._array_data_ptr(lc._compress_kv.raw_backing())
    if p0 is None:
        pytest.skip("MLX build does not expose the buffer-protocol data pointer")
    flips = 0
    for _ in range(30):                       # steady in-cap appends (no realloc)
        lc.append_compress(_row(1, 16)); lc.advance(1)
        p1 = C._array_data_ptr(lc._compress_kv.raw_backing())
        if p1 != p0:
            flips += 1
            p0 = p1
    assert flips == 0, f"slice_update did not donate in the rebind pattern ({flips} flips)"


def test_donation_gate_slice_update_donates_on_cpu(monkeypatch):
    """The round-4 donation gate: driving a bounded cache through decode with
    mx.slice_update, the compress/index/latent buffer pointers NEVER flip (every append
    donates via the rebind pattern) and the window pointer flips exactly once per
    ping-pong compaction (== window_ring.drops).  This is the real-path proof the
    reviewer specified; guarded if the pointer is unavailable."""
    cfg = _Cfg()
    _bounded_env(monkeypatch, maxkv=4096)
    C.reset_kv_bounded_stats()
    C.reset_window_ring_stats()
    cache = _make_cache(cfg)
    if C._array_data_ptr(mx.zeros((1, 2, 2))) is None:
        pytest.skip("MLX build does not expose the buffer-protocol data pointer")
    _prefill_all_lanes(cache, cfg, n=20)               # single append/lane -> 0 window drops
    mx.eval([a for lc in cache.layers for a in lc.eval_backing()])
    cache.sample_ptr_flips()                            # init prev pointers (no flip counted)
    for _ in range(150):                               # enough to force window compactions
        for L, lc in enumerate(cache.layers):
            lc.append_window(_row(1, cfg.head_dim))
            if L in set(cfg.kv_source_layer_ids):
                ratio = cfg.compress_ratios[L]
                if ratio > 1:
                    p = lc.comp_state.push(_row(1, cfg.head_dim), _row(1, cfg.head_dim))
                    if p.shape[1] > 0:
                        lc.append_compress(_row(p.shape[1], cfg.head_dim))
                        lc.append_index_k(_row(p.shape[1], cfg.index_head_dim))
                else:
                    lc.append_compress(_row(1, cfg.head_dim))
                    lc.append_index_k(_row(1, cfg.index_head_dim))
            lc.advance(1)
        mx.eval([a for lc in cache.layers for a in lc.eval_backing()])
        cache.sample_ptr_flips()

    s = C.kv_bounded_stats()
    rs = C.window_ring_stats()
    assert s["ptr_samples"] > 0
    # compress/index/latent donate every append -> pointer never moves
    assert s["ptr_flips_compress"] == 0
    assert s["ptr_flips_index"] == 0
    assert s["ptr_flips_latent"] == 0
    # the window flips exactly once per compaction
    assert rs["drops"] > 0, "need window compactions to exercise the ptr flip"
    assert s["ptr_flips_window"] == rs["drops"], (s["ptr_flips_window"], rs["drops"])
    # one prealloc per lane per layer (no per-token realloc)
    gate = C.kv_donation_gate(s, rs, expected_reallocs=C.expected_bounded_reallocs(cfg))
    assert gate["ok"] is True, gate["reasons"]


def test_donation_gate_unavailable_without_samples():
    z = {"ptr_samples": 0}
    g = C.kv_donation_gate(z, {"drops": 0})
    assert g["ok"] is None and "unavailable" in g["reasons"][0]


def test_expected_bounded_reallocs():
    exp = C.expected_bounded_reallocs(_Cfg())   # kv_source [2,5] ratios {2:2, 5:1}
    assert exp == {"window": 8, "compress": 2, "index": 2, "latent": 2}
