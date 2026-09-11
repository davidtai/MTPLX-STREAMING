"""W80 / K34 -- bounded sliding-window RING + preallocated compress/index stores
(MTPLX_DSV41_WINDOW_RING).

The phase-1 window store keeps FULL history append-only (row j == token j), so at
T=16384 the resident window is ~40x16384x512x2 = ~0.7 GB and W76/W78 pinned that
resident churn as the memory-pressure amplifier that inflates the whole 16K decode
step.  Under ``MTPLX_DSV41_WINDOW_RING`` the window store is a BOUNDED ring of
``window_size + max_verify + slack`` rows in a fixed pair of ping-pong buffers
(~128x resident cut), and the compress_kv / index_k stores (which the indexer needs
in FULL) are preallocated and written in place -- removing the per-token full-store
realloc for all three lanes.  A logical ``drop_offset`` threads through
``_window_selected_idx`` / ``_window_attend`` / the SWA mask so they address the
SAME absolute positions as the full store.

These tests prove:

  1. **Byte-identity (selected path)** -- the ring's reachable reads equal the full
     store row-for-row; end to end, a tiny-model prefill+decode is BIT-identical
     with the flag on vs off on the K30 selected-key path (cell16k's path).  The
     masked-full path is reassociation-level (the score-reduction width shrinks --
     greedy-identical, max|d|~1e-6, in-family with the other DSV4.1 score levers).
  2. **Drop-aware verify seam** -- a DSpark accept/reject/rollback sequence over the
     ring reproduces the AR reference exactly (verify authoritative + exact trim
     rollback of the ring's logical length / drop_offset).
  3. **No T-sized array in decode** -- the ring never ``concatenate``s the window and
     allocates nothing per steady decode step (ping-pong reuse); its raw backing
     stays at ``phys_cap`` rows, and cumulative rows-copied is amortized O(N), not
     the plain path's O(N^2).
  4. **eval-fence (item 4)** -- ``eval_backing`` forces the RAW preallocated buffer,
     never a fresh per-token logical slice.

CPU-only (``mx.set_default_device(mx.cpu)`` -- "no GPU" is not enough, MLX defaults
to Metal).  Tiny dims, run under ``nice -n 19`` and without ``pytest -n auto``.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mtplx.models import deepseek_v41_cache as C  # noqa: E402

RNG = np.random.default_rng(20260911)


def _row(n=1, d=16):
    return mx.array(RNG.standard_normal((1, n, d)).astype(np.float32))


def _seeded_row(n, d, rng):
    return mx.array(rng.standard_normal((1, n, d)).astype(np.float32))


def _eq(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return tuple(a.shape) == tuple(b.shape) and bool(mx.all(a == b).item())


def _ring(window=8, mv=2, slack=1, headroom=2):
    return C._WindowRing(window, mv, slack, headroom)


# ---------------------------------------------------------------------------
# _WindowRing: the logical view is a byte-identical suffix of the full history
# ---------------------------------------------------------------------------
def test_ring_view_matches_full_history_suffix():
    ring = _ring(window=8, mv=2, slack=1, headroom=2)   # cap_keep=11, phys_cap=13
    full = None
    rng = np.random.default_rng(1)
    r = _seeded_row(12, 4, rng); ring.append(r); full = r   # prefill chunk
    for _ in range(60):                                     # decode
        r = _seeded_row(1, 4, rng)
        ring.append(r)
        full = mx.concatenate([full, r], axis=1)
        drop = ring.drop_offset
        v = ring.view()
        assert ring.logical_len() == full.shape[1]
        # physical row j == absolute position drop + j
        seg = full[:, drop: drop + v.shape[1]]
        assert _eq(v, seg), "ring view is not the full-history suffix"
        # reachability: the last window_size rows are always resident
        need = min(ring.window_size, full.shape[1])
        assert drop <= full.shape[1] - need


def test_ring_prefill_chunk_then_decode_and_multi_row():
    ring = _ring(window=8, mv=4, slack=2, headroom=4)
    full = None
    rng = np.random.default_rng(2)
    for n in (12, 1, 1, 5, 1, 1, 1, 7, 1):   # mixed chunk widths (prefill + verify)
        r = _seeded_row(n, 4, rng)
        ring.append(r)
        full = r if full is None else mx.concatenate([full, r], axis=1)
        drop = ring.drop_offset
        v = ring.view()
        assert ring.logical_len() == full.shape[1]
        assert _eq(v, full[:, drop: drop + v.shape[1]])


def test_ring_truncate_and_reseat_exact():
    ring = _ring(window=8, mv=4, slack=2, headroom=4)
    full = None
    rng = np.random.default_rng(3)
    r = _seeded_row(20, 4, rng); ring.append(r); full = r
    for _ in range(40):
        r = _seeded_row(1, 4, rng); ring.append(r)
        full = mx.concatenate([full, r], axis=1)
    # verify-style rollback: append 4, then keep only 1 (accept 0)
    mark_len = full.shape[1]
    r = _seeded_row(4, 4, rng); ring.append(r)
    full4 = mx.concatenate([full, r], axis=1)
    ring.truncate_to_length(mark_len + 1)
    kept = full4[:, : mark_len + 1]
    v = ring.view(); drop = ring.drop_offset
    assert ring.logical_len() == mark_len + 1
    assert _eq(v, kept[:, drop: drop + v.shape[1]])
    # reseat (state restore): drop recomputed from logical length
    ring.reseat(mark_len + 1)
    assert ring.drop_offset == (mark_len + 1) - ring.rows()


def test_ring_no_per_step_alloc_and_amortized_rows_copied(monkeypatch):
    C.reset_window_ring_stats()
    ring = _ring(window=8, mv=2, slack=1, headroom=4)   # cap_keep=11, phys_cap=15
    # count mx.zeros allocations attributable to the ring
    real_zeros = mx.zeros
    calls = {"n": 0}

    def counting_zeros(*a, **k):
        calls["n"] += 1
        return real_zeros(*a, **k)

    monkeypatch.setattr(C.mx, "zeros", counting_zeros)
    monkeypatch.setattr(C.mx, "concatenate",
                        lambda *a, **k: pytest.fail("ring must not concatenate"))
    rng = np.random.default_rng(4)
    N = 400
    ring.append(_seeded_row(1, 4, rng))   # first alloc (2 ping-pong buffers)
    allocs_after_init = calls["n"]
    for _ in range(N):
        ring.append(_seeded_row(1, 4, rng))
        # the raw backing never exceeds phys_cap rows -- no T-sized window array
        assert int(ring.raw_backing().shape[1]) == ring.phys_cap
        assert ring.rows() <= ring.phys_cap
    # steady decode allocates NOTHING (ping-pong reuse); only the initial pair
    assert calls["n"] == allocs_after_init, "ring allocated during steady decode"
    st = C.window_ring_stats()
    # amortized O(N): rows_copied ~ c*N, NOT the plain path's ~N^2/2
    assert st["rows_copied"] < 6 * (N + 1)
    assert st["reallocs"] == 0
    # the plain _grow over the same N would copy ~ sum(k) = O(N^2)
    assert st["rows_copied"] < (N * (N + 1)) // 4


# ---------------------------------------------------------------------------
# LayerAttentionCache: reachable reads match the plain backing; trim/rollback
# ---------------------------------------------------------------------------
def _drive_layer_cache(ring_on, monkeypatch, ratio, is_kv, steps=40, prompt=20):
    if ring_on:
        monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING", "1")
        monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING_MAX_VERIFY", "4")
        monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING_SLACK", "2")
        monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING_HEADROOM", "4")
        monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING_MAXKV", "256")
    else:
        for k in ("MTPLX_DSV41_WINDOW_RING", "MTPLX_DSV41_WINDOW_RING_MAX_VERIFY",
                  "MTPLX_DSV41_WINDOW_RING_SLACK", "MTPLX_DSV41_WINDOW_RING_HEADROOM",
                  "MTPLX_DSV41_WINDOW_RING_MAXKV"):
            monkeypatch.delenv(k, raising=False)
    rng = np.random.default_rng(999)
    lc = C.LayerAttentionCache(window_size=8, compress_ratio=ratio, is_kv_source=is_kv)
    lc.append_window(_seeded_row(prompt, 16, rng))
    if is_kv:
        lc.append_compress(_seeded_row(prompt // max(1, ratio), 20, rng))
        lc.append_index_k(_seeded_row(prompt // max(1, ratio), 12, rng))
    lc.advance(prompt)
    for step in range(steps):
        lc.append_window(_seeded_row(1, 16, rng))
        if is_kv and (lc.offset + 1) % max(1, ratio) == 0:
            lc.append_compress(_seeded_row(1, 20, rng))
            lc.append_index_k(_seeded_row(1, 12, rng))
        lc.advance(1)
    return lc


@pytest.mark.parametrize("ratio,is_kv", [(0, False), (1, True), (2, True)])
def test_layer_cache_ring_reachable_reads_match_plain(monkeypatch, ratio, is_kv):
    on = _drive_layer_cache(True, monkeypatch, ratio, is_kv)
    off = _drive_layer_cache(False, monkeypatch, ratio, is_kv)
    assert on._window_ring is True and off._window_ring is False
    assert on.offset == off.offset
    # the window ring holds a SUFFIX; the reachable rows must match the plain store
    drop = on.window_drop_offset
    assert drop > 0, "test should exercise dropping"
    ov = on.window
    assert _eq(ov, off.window[:, drop: drop + ov.shape[1]])
    # the last window_size rows (all that decode reads) are present + identical
    W = on.window_size
    assert _eq(on.window[:, -W:], off.window[:, -W:])
    # compress / index stores are NOT bounded -> fully identical
    assert _eq(on.compress_kv, off.compress_kv)
    assert _eq(on.index_k, off.index_k)


def test_layer_cache_ring_trim_and_rollback_drop_aware(monkeypatch):
    on = _drive_layer_cache(True, monkeypatch, ratio=1, is_kv=True, steps=40)
    off = _drive_layer_cache(False, monkeypatch, ratio=1, is_kv=True, steps=40)
    # a mark, some verify-like appends, then trim back to the mark
    m_on, m_off = on.mark(), off.mark()
    rng = np.random.default_rng(7)
    for lc, seed in ((on, 7), (off, 7)):
        r = np.random.default_rng(seed)
        for _ in range(4):
            lc.append_window(mx.array(r.standard_normal((1, 1, 16)).astype(np.float32)))
            lc.append_compress(mx.array(r.standard_normal((1, 1, 20)).astype(np.float32)))
            lc.append_index_k(mx.array(r.standard_normal((1, 1, 12)).astype(np.float32)))
            lc.advance(1)
    on.rollback(m_on); off.rollback(m_off)
    assert on.offset == off.offset
    drop = on.window_drop_offset
    ov = on.window
    assert _eq(ov, off.window[:, drop: drop + ov.shape[1]]), "rollback not exact"
    assert _eq(on.window[:, -on.window_size:], off.window[:, -off.window_size:])
    assert _eq(on.compress_kv, off.compress_kv)
    # trim seam (the served DSpark path drives entry.trim)
    on.trim(3); off.trim(3)
    assert on.offset == off.offset
    d2 = on.window_drop_offset
    assert _eq(on.window, off.window[:, d2: d2 + on.window.shape[1]])


def test_eval_backing_returns_raw_buffer_not_slice(monkeypatch):
    lc = _drive_layer_cache(True, monkeypatch, ratio=1, is_kv=True, steps=40)
    backing = lc.eval_backing()
    # the window backing is the RAW ping-pong buffer (phys_cap rows), NOT a fresh
    # logical-length slice -- so a per-token settle fence never builds a T-sized copy
    win_raw = lc._window.raw_backing()
    assert any(a is win_raw for a in backing), "eval_backing did not force the raw buffer"
    assert int(win_raw.shape[1]) == lc._window.phys_cap
    # and it is strictly larger than the resident logical rows (a slice would be smaller)
    assert win_raw.shape[1] >= lc._window.rows()


def test_compress_index_preallocated_no_resize(monkeypatch):
    """W80 item 2: under the ring the compress/index stores are preallocated to
    ``maxkv`` and written in place -- so the geometric _GrowBuffer never resizes
    (one buffer allocation, ever) and the logical rows-copied per append is O(new
    rows), NOT O(n_comp).  (The donation finding: on Metal the in-place
    ``slice_update`` is donated only when uniquely referenced; the indexer's
    full-store read blocks that -> the custom-kernel remedy is a GPU-window item.
    The CPU double here proves the LOGICAL O(new-rows) write + no-resize.)"""
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING", "1")
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING_MAXKV", "256")
    C.reset_kv_chunk_grow_stats()
    lc = C.LayerAttentionCache(window_size=8, compress_ratio=1, is_kv_source=True)
    assert isinstance(lc._compress_kv, C._GrowBuffer)
    assert lc._compress_kv._init_cap == 256  # preallocated to maxkv
    rng = np.random.default_rng(11)
    N = 200
    for _ in range(N):
        lc.append_compress(_seeded_row(1, 20, rng))
        lc.append_index_k(_seeded_row(1, 12, rng))
    s = C.kv_chunk_grow_stats()
    # preallocated to 256 > N=200 -> the geometric resize NEVER fires: one buffer
    # per lane (2 total), and rows_copied is exactly one row per append (O(new)).
    assert s["buffers"] == 2, f"expected no resize (2 buffers), got {s['buffers']}"
    assert s["rows_copied"] == 2 * N, f"rows_copied not O(new): {s['rows_copied']}"
    assert int(lc.compress_kv.shape[1]) == N and int(lc.index_k.shape[1]) == N


def test_window_ring_stats_reset_and_snapshot(monkeypatch):
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING", "1")
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING_MAX_VERIFY", "2")
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING_SLACK", "1")
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING_HEADROOM", "2")
    C.reset_window_ring_stats()
    assert C.window_ring_stats()["enabled"] is False
    lc = C.LayerAttentionCache(window_size=8, compress_ratio=0, is_kv_source=False)
    for _ in range(30):
        lc.append_window(_row(1, 16)); lc.advance(1)
    s = C.window_ring_stats()
    assert s["enabled"] is True and s["layers_ring"] >= 1
    assert s["capacity"] == lc._window.cap_keep
    assert s["drops"] >= 1 and s["rows_copied"] >= 30
    C.reset_window_ring_stats()
    assert C.window_ring_stats()["layers_ring"] == 0


# ---------------------------------------------------------------------------
# End-to-end tiny model: selected path bit-identical, masked-full greedy-identical
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


def _ring_env(monkeypatch):
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING", "1")
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING_MAX_VERIFY", "2")
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING_SLACK", "1")
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING_HEADROOM", "2")
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING_MAXKV", "512")


def test_model_selected_path_bit_identical_with_ring(monkeypatch):
    model = _tiny_model()
    prompt = list(range(20)); steps = 280   # T ~ 300, forces many drops
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    monkeypatch.delenv("MTPLX_DSV41_WINDOW_RING", raising=False)
    base, base_toks, base_cache = _prefill_decode(model, prompt, steps)
    _ring_env(monkeypatch)
    ring, ring_toks, ring_cache = _prefill_decode(model, prompt, steps)
    assert ring_cache.layers[0]._window_ring is True
    assert isinstance(ring_cache.layers[0]._window, C._WindowRing)
    assert ring_cache.layers[0].window_drop_offset > 0, "ring should have dropped rows"
    assert base_toks == ring_toks
    for i, (a, b) in enumerate(zip(base, ring)):
        assert bool(mx.all(a == b).item()), f"selected path logits differ at step {i}"


def test_model_masked_full_path_greedy_identical_with_ring(monkeypatch):
    model = _tiny_model()
    prompt = list(range(20)); steps = 200
    monkeypatch.delenv("MTPLX_DSV41_SELECTED_KEYS", raising=False)  # masked-full
    monkeypatch.delenv("MTPLX_DSV41_WINDOW_RING", raising=False)
    base, base_toks, _ = _prefill_decode(model, prompt, steps)
    _ring_env(monkeypatch)
    ring, ring_toks, _ = _prefill_decode(model, prompt, steps)
    assert base_toks == ring_toks, "masked-full path must stay greedy-identical"
    maxd = max(float(mx.max(mx.abs(a - b)).item()) for a, b in zip(base, ring))
    # reassociation-level only (the score-reduction width shrinks); NOT bit-identical
    assert maxd < 1e-5, f"masked-full reassoc too large: {maxd}"


def test_no_T_sized_window_concat_during_decode(monkeypatch):
    """The ring never concatenates the window store during decode (it slice_updates
    a bounded ping-pong buffer); the largest window-lane array is phys_cap rows."""
    model = _tiny_model()
    _ring_env(monkeypatch)
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    cache = model.make_cache()
    logits = model(mx.array([list(range(40))]), cache=cache); mx.eval(logits)
    tok = int(mx.argmax(logits[0, -1]).item())
    # decode to a large T so a plain store would be huge; the ring stays bounded
    for _ in range(60):
        logits = model(mx.array([[tok]]), cache=cache); mx.eval(logits)
        tok = int(mx.argmax(logits[0, -1]).item())
    for lc in cache.layers:
        assert int(lc._window.raw_backing().shape[1]) == lc._window.phys_cap
        assert lc._window.rows() <= lc._window.phys_cap
        # the resident window is far smaller than the logical length (T ~ 100)
        assert lc._window.logical_len() > lc._window.phys_cap


# ---------------------------------------------------------------------------
# DSpark verify accept/reject/rollback over the ring reproduces the AR reference
# ---------------------------------------------------------------------------
def _mtp_args(vocab=64):
    from mtplx.models.deepseek_v41 import ModelArgs
    return ModelArgs(
        vocab_size=vocab, hidden_size=32, num_hidden_layers=5,
        num_attention_heads=4, head_dim=16, qk_rope_head_dim=8,
        q_lora_rank=16, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=2,
        sliding_window=8, window_size=8, hc_mult=4, hc_sinkhorn_iters=2,
        scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, swiglu_limit=0.0,
        n_mtp_layers=3, dspark_block_size=4, dspark_noise_token_id=vocab - 1,
        dspark_target_layer_ids=[2, 3, 4], dspark_markov_rank=12,
        dspark_n_routed_experts=8, dspark_num_experts_per_tok=2,
    )


def _mtp_model(seed=0, vocab=64):
    from mlx.utils import tree_flatten, tree_unflatten
    from mtplx.models.deepseek_v41 import Model
    mx.random.seed(seed)
    args = _mtp_args(vocab)
    model = Model(args, quantize=False, mtp=True)
    filled = []
    for name, value in tree_flatten(model.parameters()):
        leaf = name.split(".")[-1]
        if value.ndim == 1:
            new = mx.random.normal(value.shape) * 0.1 + (
                1.0 if leaf.endswith("norm_weight") or leaf == "scale" else 0.0)
        else:
            new = mx.random.normal(value.shape) * (value.shape[-1] ** -0.5)
        filled.append((name, new.astype(value.dtype)))
    model.update(tree_unflatten(filled))
    mx.eval(model.parameters())
    return model


def test_dspark_verify_reproduces_ar_with_ring(monkeypatch):
    from mtplx.models.deepseek_v41_dspark_decode import dspark_generate, DSparkDecodeStats
    from mtplx.sampling import SamplerConfig
    _ring_env(monkeypatch)
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    model = _mtp_model(seed=3, vocab=32)
    prompt = list(range(6))
    greedy = SamplerConfig(temperature=0.0)
    n = 120
    # AR reference under the SAME ring config
    def ar(model, prompt, n):
        cache = model.make_cache()
        logits = model(mx.array([list(prompt)]), cache=cache); mx.eval(logits)
        out = []; tok = int(mx.argmax(logits[0, -1]).item())
        for _ in range(n):
            out.append(tok)
            logits = model(mx.array([[tok]]), cache=cache); mx.eval(logits)
            tok = int(mx.argmax(logits[0, -1]).item())
        return out
    ref = ar(model, prompt, n)
    stats = DSparkDecodeStats()
    out = dspark_generate(model, prompt, max_tokens=n, sampler=greedy, seed=0,
                          stop_ids=set(), speculative_depth=3, stats=stats)
    assert out[:len(ref)] == ref[:len(out)], "DSpark+ring did not reproduce AR"
    # the run exercised BOTH accepts and rejects (drop-aware rollback path)
    assert stats.accepted_drafts > 0 and stats.rejected_drafts > 0
