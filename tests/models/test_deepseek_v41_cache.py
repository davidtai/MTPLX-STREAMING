"""W13 -- per-sequence attention STATE for the DeepSeek-V4.1 MLX port.

Every test drives ``mtplx.models.deepseek_v41_cache`` against an independent
numpy transcription of the reference state updates
(``~/models/DeepSeek-V4.1-Flash-src/inference/model.py``) on random data, on the
CPU (``mx.set_default_device(mx.cpu)`` -- a "no GPU" test still defaults MLX to
Metal without this).
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mtplx.models.deepseek_v41_cache import (  # noqa: E402
    CompressorState,
    DeepseekV41Cache,
    LayerAttentionCache,
    SharedAttentionRuntime,
    make_cache,
    ring_view,
    window_topk_idxs,
)

RNG = np.random.default_rng(20260910)


def _rand(*shape):
    return mx.array(RNG.standard_normal(shape).astype(np.float32))


def _np(a):
    return np.asarray(a) if a is not None else None


# ---------------------------------------------------------------------------
# numpy transcriptions of the reference state updates
# ---------------------------------------------------------------------------
def ref_window_ring(kv_all: np.ndarray, window_size: int, length: int):
    """Reference ``window_kv_cache[:bsz]`` built token-by-token with the literal
    reference write (``ring[p % W] = kv[p]`` for every fed position -- model.py
    L708-719) and read as the whole ring (L719)."""
    B, _, hd = kv_all.shape
    ring = np.zeros((B, window_size, hd), dtype=kv_all.dtype)
    for p in range(length):
        ring[:, p % window_size] = kv_all[:, p]
    return ring


def ref_window_topk_idxs(window_size, bsz, seqlen, start_pos):
    """Numpy transcription of ``get_window_topk_idxs`` (model.py L409-426)."""
    win = window_size
    if start_pos == 0:
        end = np.arange(seqlen)[:, None]
        idxs = np.clip(end - win + 1, 0, None) + np.arange(min(seqlen, win))
        idxs = np.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % win + 1
        idxs = np.concatenate([np.arange(oldest, win), np.arange(oldest)])
        idxs = np.where(idxs > start_pos, -1, idxs)[None, :]
    idxs = idxs.astype(np.int32)
    return np.broadcast_to(idxs[None], (bsz, idxs.shape[0], idxs.shape[1]))


def ref_pool_oneshot(kv: np.ndarray, score: np.ndarray, ratio: int):
    """One-shot compressor pooling of every complete group (model.py L473-475)."""
    B, T, hd = kv.shape
    g = T // ratio
    if g == 0:
        return np.zeros((B, 0, hd), dtype=kv.dtype)
    kv_g = kv[:, : g * ratio].reshape(B, g, ratio, hd)
    sc_g = score[:, : g * ratio].reshape(B, g, ratio, hd)
    sc_g = sc_g - sc_g.max(axis=2, keepdims=True)
    w = np.exp(sc_g)
    w = w / w.sum(axis=2, keepdims=True)
    return (kv_g * w).sum(axis=2)


# ---------------------------------------------------------------------------
# 1. window ring rolling across the 128 boundary
# ---------------------------------------------------------------------------
def test_window_ring_rolling_is_chunk_invariant():
    B, hd, W, total = 2, 4, 128, 300
    kv_all = RNG.standard_normal((B, total, hd)).astype(np.float32)
    oracle = ref_window_ring(kv_all, W, total)

    for chunk in (1, 7, 64, 300):
        cache = DeepseekV41Cache(1, window_size=W)
        layer = cache.layers[0]
        p = 0
        while p < total:
            step = min(chunk, total - p)
            layer.append_window(mx.array(kv_all[:, p : p + step]))
            cache.advance(step)
            p += step
        got = _np(layer.ring(cache.offset))
        assert got.shape == oracle.shape, (chunk, got.shape, oracle.shape)
        assert np.array_equal(got, oracle), f"ring mismatch at chunk={chunk}"


def test_window_ring_view_while_still_filling():
    B, hd, W = 1, 3, 128
    kv_all = RNG.standard_normal((B, 40, hd)).astype(np.float32)
    cache = DeepseekV41Cache(1, window_size=W)
    cache.layers[0].append_window(mx.array(kv_all))
    cache.advance(40)
    # the reference reads the whole [B, W, hd] ring; slot == position while
    # filling, empty slots (40..127) zero
    got = _np(cache.layers[0].ring(40))
    exp = np.zeros((B, W, hd), dtype=np.float32)
    exp[:, :40] = kv_all
    assert got.shape == (B, W, hd)
    assert np.array_equal(got, exp)


def test_window_topk_idxs_matches_reference():
    for bsz, seqlen, start_pos in [(2, 10, 0), (1, 200, 0), (3, 1, 5), (2, 1, 200), (1, 1, 127)]:
        got = _np(window_topk_idxs(128, bsz, seqlen, start_pos))
        exp = ref_window_topk_idxs(128, bsz, seqlen, start_pos)
        assert got.shape == exp.shape, (bsz, seqlen, start_pos, got.shape, exp.shape)
        assert np.array_equal(got, exp), (bsz, seqlen, start_pos)


def test_ring_view_none_and_empty():
    assert ring_view(None, 128, 5) is None
    assert ring_view(mx.zeros((1, 4, 2)), 128, 0) is None


# ---------------------------------------------------------------------------
# 2. compressor pooling across odd/even chunk boundaries == one-shot
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chunks", [[13], [1] * 13, [3, 4, 5, 1], [2, 2, 2, 2, 2, 2, 1], [7, 6]])
def test_compressor_pooling_is_chunk_invariant(chunks):
    B, hd, ratio = 2, 4, 2
    total = sum(chunks)
    kv = RNG.standard_normal((B, total, hd)).astype(np.float32)
    sc = RNG.standard_normal((B, total, hd)).astype(np.float32)
    oracle = ref_pool_oneshot(kv, sc, ratio)

    state = CompressorState(ratio)
    pooled_chunks = []
    p = 0
    for c in chunks:
        out = state.push(mx.array(kv[:, p : p + c]), mx.array(sc[:, p : p + c]))
        if out.shape[1] > 0:
            pooled_chunks.append(_np(out))
        p += c
    pooled = np.concatenate(pooled_chunks, axis=1) if pooled_chunks else np.zeros((B, 0, hd), np.float32)

    assert pooled.shape == oracle.shape, (chunks, pooled.shape, oracle.shape)
    assert np.allclose(pooled, oracle, rtol=1e-5, atol=1e-5), chunks
    assert state.n_fed == total
    assert state.n_groups == total // ratio


def test_compressor_state_only_for_ratio_gt_1():
    with pytest.raises(ValueError):
        CompressorState(1)


# ---------------------------------------------------------------------------
# 3. index-K append
# ---------------------------------------------------------------------------
def test_index_k_append_grows_and_concatenates():
    layer = LayerAttentionCache(window_size=128, compress_ratio=2, is_kv_source=True)
    a = _rand(2, 3, 8)
    b = _rand(2, 5, 8)
    layer.append_index_k(a)
    layer.append_index_k(b)
    got = _np(layer.index_k)
    assert got.shape == (2, 8, 8)
    assert np.array_equal(got[:, :3], _np(a))
    assert np.array_equal(got[:, 3:], _np(b))
    # a zero-length append is a no-op
    layer.append_index_k(mx.zeros((2, 0, 8)))
    assert _np(layer.index_k).shape == (2, 8, 8)


# ---------------------------------------------------------------------------
# 4. candidate / top-k publication and reuse-layer read
# ---------------------------------------------------------------------------
def test_shared_runtime_publish_and_reuse_read():
    shared = SharedAttentionRuntime()
    assert shared.compress_kv is None and shared.topk_idxs is None

    # index/candidate source publishes
    ck = _rand(2, 6, 4)
    ik = _rand(2, 6, 8)
    sel = mx.array(RNG.integers(0, 2, (2, 3, 6)).astype(np.bool_))
    cand = mx.array(RNG.integers(0, 2, (2, 3, 6)).astype(np.bool_))
    shared.compress_kv = ck
    shared.index_k = ik
    shared.topk_idxs = sel
    shared.candidates = cand

    # a Reuse layer reads the same published objects (identity, one slot each)
    assert shared.compress_kv is ck
    assert shared.topk_idxs is sel
    assert shared.candidates is cand
    # topk_mask is the boolean-view alias of the same slot
    assert shared.topk_mask is shared.topk_idxs
    new_mask = mx.array(RNG.integers(0, 2, (2, 3, 6)).astype(np.bool_))
    shared.topk_mask = new_mask
    assert shared.topk_idxs is new_mask


def test_cache_hands_a_fresh_shared_runtime():
    cache = DeepseekV41Cache(3)
    r1 = cache.new_shared_runtime()
    r2 = cache.new_shared_runtime()
    assert isinstance(r1, SharedAttentionRuntime) and r1 is not r2


# ---------------------------------------------------------------------------
# 5. trim(n) then re-feed == identical state and identical attention inputs
# ---------------------------------------------------------------------------
def _feed_kv_source_layer(layer, kv_win, comp_kv, comp_sc):
    """Mimic one W10 forward on a kv_source layer: seed the window, push the
    compressor rows, and append whatever groups completed to the compressed and
    index stores (identity stand-ins for W10's RoPE/quant -- the state mechanics
    are what is under test)."""
    layer.append_window(kv_win)
    pooled = layer.comp_state.push(comp_kv, comp_sc)
    if pooled.shape[1] > 0:
        layer.append_compress(pooled)                 # stand-in for rope+quant(pooled)
        layer.append_index_k(pooled[..., :4])          # stand-in for the index keys


def test_trim_then_refeed_restores_identical_state_and_attention_inputs():
    B, hd, W, ratio = 2, 6, 128, 2
    N, M = 10, 7
    tot = N + M
    kv_win = RNG.standard_normal((B, tot, hd)).astype(np.float32)
    comp_kv = RNG.standard_normal((B, tot, hd)).astype(np.float32)
    comp_sc = RNG.standard_normal((B, tot, hd)).astype(np.float32)

    def build(feed_len):
        cache = DeepseekV41Cache(1, window_size=W, compress_ratios=[ratio], kv_source_layer_ids=[0])
        layer = cache.layers[0]
        for p in range(feed_len):
            _feed_kv_source_layer(
                layer,
                mx.array(kv_win[:, p : p + 1]),
                mx.array(comp_kv[:, p : p + 1]),
                mx.array(comp_sc[:, p : p + 1]),
            )
            cache.advance(1)
        return cache

    # reference end state: feed all N+M directly
    direct = build(tot)

    # streamed: feed N, then M, then trim(M) back, then re-feed the same M
    cache = build(N)
    layer = cache.layers[0]
    for p in range(N, tot):
        _feed_kv_source_layer(
            layer,
            mx.array(kv_win[:, p : p + 1]),
            mx.array(comp_kv[:, p : p + 1]),
            mx.array(comp_sc[:, p : p + 1]),
        )
        cache.advance(1)
    assert cache.offset == tot
    trimmed = cache.trim(M)
    assert trimmed == M
    assert cache.offset == N
    # trimmed stores are exactly N tokens deep again
    assert layer.window_len() == N
    assert layer.comp_state.n_fed == N
    for p in range(N, tot):
        _feed_kv_source_layer(
            layer,
            mx.array(kv_win[:, p : p + 1]),
            mx.array(comp_kv[:, p : p + 1]),
            mx.array(comp_sc[:, p : p + 1]),
        )
        cache.advance(1)

    dl, cl = direct.layers[0], cache.layers[0]
    assert cache.offset == direct.offset == tot
    # stored state identical
    assert np.array_equal(_np(cl.window), _np(dl.window))
    assert np.array_equal(_np(cl.compress_kv), _np(dl.compress_kv))
    assert np.array_equal(_np(cl.index_k), _np(dl.index_k))
    assert np.array_equal(_np(cl.comp_state.raw_kv), _np(dl.comp_state.raw_kv))
    assert np.array_equal(_np(cl.comp_state.raw_score), _np(dl.comp_state.raw_score))
    # attention inputs identical
    assert np.array_equal(_np(cl.ring(cache.offset)), _np(dl.ring(direct.offset)))
    assert np.array_equal(
        _np(window_topk_idxs(W, B, 1, cache.offset)),
        _np(window_topk_idxs(W, B, 1, direct.offset)),
    )


def test_trim_crosses_group_boundary_exactly():
    # 17 tokens, ratio 2 -> 8 groups + 1 frontier row; trim 7 -> 5 groups, 0 frontier
    B, hd, ratio = 1, 4, 2
    cache = DeepseekV41Cache(1, window_size=128, compress_ratios=[ratio], kv_source_layer_ids=[0])
    layer = cache.layers[0]
    for p in range(17):
        _feed_kv_source_layer(layer, _rand(B, 1, hd), _rand(B, 1, hd), _rand(B, 1, hd))
        cache.advance(1)
    assert layer.comp_state.n_groups == 8 and _np(layer.compress_kv).shape[1] == 8
    cache.trim(7)
    assert cache.offset == 10
    assert layer.window_len() == 10
    assert layer.comp_state.n_fed == 10
    assert layer.comp_state.n_groups == 5
    assert _np(layer.compress_kv).shape[1] == 5
    assert _np(layer.index_k).shape[1] == 5


def test_mark_rollback_matches_trim():
    B, hd, ratio = 1, 4, 2
    cache = DeepseekV41Cache(1, window_size=128, compress_ratios=[ratio], kv_source_layer_ids=[0])
    layer = cache.layers[0]
    for _ in range(10):
        _feed_kv_source_layer(layer, _rand(B, 1, hd), _rand(B, 1, hd), _rand(B, 1, hd))
        cache.advance(1)
    snap = cache.mark()
    for _ in range(6):
        _feed_kv_source_layer(layer, _rand(B, 1, hd), _rand(B, 1, hd), _rand(B, 1, hd))
        cache.advance(1)
    cache.rollback(snap)
    assert cache.offset == 10
    assert layer.window_len() == 10 and layer.comp_state.n_fed == 10
    assert _np(layer.compress_kv).shape[1] == 5


def test_trim_caps_at_offset():
    cache = DeepseekV41Cache(2)
    for _ in range(4):
        for lc in cache.layers:
            lc.append_window(_rand(1, 1, 4))
        cache.advance(1)
    assert cache.trim(100) == 4
    assert cache.offset == 0


# ---------------------------------------------------------------------------
# 6. engram_state advances and trims in step
# ---------------------------------------------------------------------------
def _small_ngram_state():
    from mtplx.engram_v41 import NgramHashState, n_hash_cols

    max_ng, n_heads = 3, 2
    cols = n_hash_cols(max_ng, n_heads)
    V = 40
    token_map = list(range(V))
    # fixed (not RNG-derived) so two independently built states share one hash
    # config -- the streamed vs one-shot comparison must differ only in feeding
    multipliers = np.array([[13, 8675309, 271828183]], dtype=np.int64)  # [1, max_ng], odd
    primes = np.array([[[101, 103], [107, 109]]], dtype=np.int64)  # [1, max_ng-1, n_heads]
    flat_offsets = np.array([[0, 101, 204, 311]], dtype=np.int64)  # [1, cols]
    assert primes.shape == (1, max_ng - 1, n_heads)
    assert flat_offsets.shape == (1, cols)
    return NgramHashState(
        token_map=token_map,
        multipliers=multipliers,
        primes=primes,
        flat_offsets=flat_offsets,
        pad_compressed=int(token_map[2]),
        max_ngram_size=max_ng,
        n_heads=n_heads,
        layer_ids=(1,),
    )


def _forward(cache, es, ids_chunk, hd=4):
    """One W10-shaped forward: advance the engram history (before the layers) and
    seed each layer's window, then advance the shared offset."""
    B, chunk = ids_chunk.shape
    es.advance(ids_chunk)
    for layer in cache.layers:
        layer.append_window(_rand(B, chunk, hd))
    cache.advance(chunk)


def test_engram_state_advances_and_trims_in_step_with_offset():
    es = _small_ngram_state()
    cache = make_cache(3, engram_state=es)
    assert cache.engram_state is es

    B = 2
    ids = RNG.integers(0, 40, (B, 20)).astype(np.int64)
    # W10 backbone advances the engram history once per forward, before the layers
    p = 0
    for chunk in (5, 1, 1, 1, 4):
        _forward(cache, es, ids[:, p : p + chunk])
        p += chunk
        assert es.length == cache.offset  # advance in step

    # a speculative rollback trims KV and engram history together
    cache.trim(6)
    assert cache.offset == 6
    assert es.length == 6  # trimmed in step
    assert cache.layers[0].window_len() == 6  # KV trimmed in the same step


def test_engram_trim_then_refeed_restores_identical_row_ids():
    B = 1
    ids = RNG.integers(0, 40, (B, 12)).astype(np.int64)

    # direct: 12 tokens in one advance
    es_d = _small_ngram_state()
    cache_d = make_cache(1, engram_state=es_d)
    es_d.advance(ids)
    cache_d.advance(12)
    direct_rows = es_d.current_row_ids(0).copy()

    # streamed: 8, then 4, trim 4, re-feed the last 4
    es_s = _small_ngram_state()
    cache_s = make_cache(1, engram_state=es_s)
    _forward(cache_s, es_s, ids[:, :8])
    _forward(cache_s, es_s, ids[:, 8:12])
    cache_s.trim(4)
    assert cache_s.offset == 8 and es_s.length == 8
    rows = es_s.advance(ids[:, 8:12])
    cache_s.advance(4)
    assert cache_s.offset == 12 and es_s.length == 12
    # the re-fed tail hashes identically (n-gram lookback crossed the boundary)
    assert np.array_equal(rows[:, :, 0, :], direct_rows[:, 8:12])


# ---------------------------------------------------------------------------
# 7. gate / mlx_lm decode-loop contract
# ---------------------------------------------------------------------------
class _FakeModel:
    """Minimal stand-in exposing the ``make_cache`` hook mlx_lm looks for."""

    def __init__(self, n_layers, ratios, kv_sources):
        self._n = n_layers
        self._ratios = ratios
        self._kv = kv_sources

    def make_cache(self):
        c = DeepseekV41Cache(
            self._n, window_size=128, compress_ratios=self._ratios, kv_source_layer_ids=self._kv
        )
        return c


def test_make_prompt_cache_returns_our_cache():
    from mlx_lm.models.cache import make_prompt_cache

    model = _FakeModel(4, [0, 2, 2, 1], [1, 2, 3])
    cache = make_prompt_cache(model)
    assert isinstance(cache, DeepseekV41Cache)
    assert len(cache.layers) == 4
    assert cache.is_trimmable()


def test_gate_decode_loop_reuses_one_object_and_trims():
    # make_cache once, same object every step (the gate's _greedy_decode shape),
    # then a speculative trim, exactly as the verify path uses it.
    model = _FakeModel(2, [0, 2], [1])
    cache = model.make_cache()
    assert cache.offset == 0

    def step(n):
        # a forward touches every layer's window and advances the shared offset
        for li, layer in enumerate(cache.layers):
            layer.append_window(_rand(1, n, 4))
            if layer.is_kv_source:
                pooled = layer.comp_state.push(_rand(1, n, 4), _rand(1, n, 4))
                if pooled.shape[1] > 0:
                    layer.append_compress(pooled)
                    layer.append_index_k(pooled[..., :4])
        cache.advance(n)

    step(6)                     # prefill
    same = cache
    for _ in range(4):          # decode steps reuse the identical object
        step(1)
        assert cache is same
    assert cache.offset == 10

    cache.trim(3)               # reject 3 speculative tokens
    assert cache.offset == 7
    assert cache.layers[0].window_len() == 7
    step(3)                     # re-feed the accepted continuation
    assert cache.offset == 10


def test_make_cache_from_args_like_object():
    class _Args:
        num_hidden_layers = 5
        window_size = 128
        compress_ratios = [0, 2, 2, 1, 0]
        kv_source_layer_ids = [1, 3]

    cache = make_cache(_Args())
    assert len(cache.layers) == 5
    assert cache.layers[1].is_kv_source and cache.layers[1].comp_state is not None
    assert cache.layers[3].is_kv_source and cache.layers[3].comp_state is None  # ratio 1: no frontier
    assert not cache.layers[0].is_kv_source
