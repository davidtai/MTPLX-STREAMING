"""W73 / K32 -- chunk-grown KV append backing (MTPLX_DSV41_KV_CHUNK_GROW).

The phase-1 cache grows every store lane with :func:`_grow` (a full
``mx.concatenate`` of the whole store per token = O(current-length) copy per
layer per token).  W73 measured the window append alone at ~2 ms/layer at
T=16384 on the CPU double (~40x its 1K cost) -- across 40 layers the dominant
per-token O(T) work once K30 selected keys has bounded the attention score
(docs/deepseek-v41/W73_DECODE_16K_AUDIT.md).

Under ``MTPLX_DSV41_KV_CHUNK_GROW`` the lanes grow via :class:`_GrowBuffer`: a
geometric-capacity buffer + logical length, appended with a donated
``mx.slice_update`` in-place write (amortized O(new-rows); the copy-everything
resize fires only on the O(log T) doublings).  These tests prove the two things
that matter:

  1. **Byte-identity** -- the ``buf[:, :length]`` view equals the ``_grow``
     (concatenate) store row-for-row, for single-row (decode) and multi-row
     (prefill chunk) appends, through trim / rollback / mlx_lm state, and --
     end to end -- a real tiny-model prefill+decode produces byte-identical
     logits with the flag on vs off.
  2. **O(T) counter** -- bytes (rows) copied per step does NOT scale with T: the
     chunk path's cumulative rows-copied is ~O(N) (amortized), the plain path's
     is ~O(N^2); the per-step wall stays flat while the plain path's grows.

CPU-only (``mx.set_default_device(mx.cpu)`` -- "no GPU" is not enough, MLX
defaults to Metal).  Run under ``nice -n 19`` and without ``pytest -n auto``.
"""

from __future__ import annotations

import time

import numpy as np
import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mtplx.models import deepseek_v41_cache as C  # noqa: E402

RNG = np.random.default_rng(20260911)


def _row(n=1, d=16):
    return mx.array(RNG.standard_normal((1, n, d)).astype(np.float32))


def _eq(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return tuple(a.shape) == tuple(b.shape) and bool(mx.all(a == b).item())


# ---------------------------------------------------------------------------
# _GrowBuffer byte-identity vs _grow
# ---------------------------------------------------------------------------
def test_grow_buffer_single_row_byte_identical():
    gb = C._GrowBuffer(init_cap=4)
    plain = None
    for _ in range(37):  # spans several geometric doublings from cap 4
        r = _row(1)
        gb.append(r)
        plain = C._grow(plain, r)
        assert _eq(gb.view(), plain)


def test_grow_buffer_multi_row_byte_identical():
    gb = C._GrowBuffer(init_cap=4)
    plain = None
    for n in (5, 3, 9, 1, 20, 7, 2):
        chunk = _row(n)
        gb.append(chunk)
        plain = C._grow(plain, chunk)
        assert _eq(gb.view(), plain)


def test_grow_buffer_set_and_truncate_byte_identical():
    gb = C._GrowBuffer(init_cap=4)
    plain = None
    for n in (6, 4, 10):
        chunk = _row(n)
        gb.append(chunk)
        plain = C._grow(plain, chunk)
    gb.set(plain[:, :12])
    assert _eq(gb.view(), plain[:, :12])
    gb.truncate_to(5)
    assert _eq(gb.view(), plain[:, :5])
    gb.truncate_to(0)
    assert gb.view() is None
    gb.set(None)
    assert gb.view() is None


def test_grow_buffer_empty_and_zero_len_appends():
    gb = C._GrowBuffer()
    assert gb.view() is None
    gb.append(None)
    gb.append(mx.zeros((1, 0, 16)))
    assert gb.view() is None


# ---------------------------------------------------------------------------
# O(T) counter: rows copied per step does not scale with T
# ---------------------------------------------------------------------------
def test_append_rows_copied_amortized_not_O_T():
    N, d = 1500, 32
    r = mx.array(np.zeros((1, 1, d), np.float32))

    C._APPEND_ROWS_COPIED = 0
    gb = C._GrowBuffer(init_cap=256)
    for _ in range(N):
        gb.append(r)
        mx.eval(gb._buf)
    chunk_rows = C._APPEND_ROWS_COPIED

    C._APPEND_ROWS_COPIED = 0
    plain = None
    for _ in range(N):
        plain = C._grow(plain, r)
        C._note_rows_copied(C._rows(plain))
        mx.eval(plain)
    plain_rows = C._APPEND_ROWS_COPIED

    # amortized: geometric growth copies each row O(1) times (< ~4 N total),
    # never the O(N^2/2) the concatenate-everything plain path pays.
    assert chunk_rows < 4 * N, (chunk_rows, N)
    assert plain_rows > N * N // 4, (plain_rows, N)
    # concrete separation at N=1500: ~5.7k vs ~1.13M
    assert plain_rows > 50 * chunk_rows


def test_append_walltime_flat_vs_growing():
    # The donated in-place write keeps per-step wall flat as T grows; the plain
    # concatenate path's per-step wall grows with T.  Lenient factors so the
    # assertion is robust under a loaded box, but the two paths are far apart.
    N, d = 1600, 512
    r = mx.array(np.zeros((1, 1, d), np.float32))

    def early_late(step):
        step()  # warm
        t = time.perf_counter()
        for _ in range(20):
            step()
        early = time.perf_counter() - t
        for _ in range(N - 40):
            step()
        t = time.perf_counter()
        for _ in range(20):
            step()
        late = time.perf_counter() - t
        return early, late

    gb = C._GrowBuffer(init_cap=256)
    ce, cl = early_late(lambda: (gb.append(r), mx.eval(gb._buf)))

    box = {"a": None}

    def plain_step():
        box["a"] = C._grow(box["a"], r)
        mx.eval(box["a"])

    pe, pl = early_late(plain_step)

    # chunk path stays roughly flat (allow 3x slack for scheduler noise)...
    assert cl < 3.0 * ce, (ce, cl)
    # ...while the plain path's late steps are markedly slower than its early ones
    assert pl > 4.0 * pe, (pe, pl)


# ---------------------------------------------------------------------------
# LayerAttentionCache parity: chunk-grow vs default, through trim/rollback/state
# ---------------------------------------------------------------------------
def _drive_layer(cache, ratio, is_kv_source):
    """Feed a deterministic sequence of appends + one trim + one rollback."""
    d_win, d_cmp, d_idx = 16, 20, 12
    marks = []
    for step in range(9):
        cache.append_window(_row(1, d_win))
        if is_kv_source and step % max(1, ratio) == 0:
            cache.append_compress(_row(1, d_cmp))
            cache.append_index_k(_row(1, d_idx))
        cache.advance(1)
        if step == 4:
            marks.append(cache.mark())
    cache.rollback(marks[0])
    for _ in range(3):
        cache.append_window(_row(1, d_win))
        cache.advance(1)


@pytest.mark.parametrize("ratio,is_kv", [(0, False), (1, True), (2, True)])
def test_layer_cache_chunk_grow_matches_default(monkeypatch, ratio, is_kv):
    def _seeded_row(n, d, rng):
        return mx.array(rng.standard_normal((1, n, d)).astype(np.float32))

    def build_and_drive(chunk_on):
        rng = np.random.default_rng(999)  # identical stream both runs
        if chunk_on:
            monkeypatch.setenv("MTPLX_DSV41_KV_CHUNK_GROW", "1")
        else:
            monkeypatch.delenv("MTPLX_DSV41_KV_CHUNK_GROW", raising=False)
        lc = C.LayerAttentionCache(
            window_size=8, compress_ratio=ratio, is_kv_source=is_kv
        )
        mark = None
        for step in range(9):
            lc.append_window(_seeded_row(1, 16, rng))
            if is_kv and step % max(1, ratio) == 0:
                lc.append_compress(_seeded_row(1, 20, rng))
                lc.append_index_k(_seeded_row(1, 12, rng))
            lc.advance(1)
            if step == 4:
                mark = lc.mark()
        # rewind to the mark (the W13 rollback seam -- exercises the property
        # truncate path on every lane for all CSA modes), then feed more.
        lc.rollback(mark)
        for _ in range(2):
            lc.append_window(_seeded_row(1, 16, rng))
            lc.advance(1)
        return lc

    on = build_and_drive(True)
    off = build_and_drive(False)
    assert on._chunk_grow is True and off._chunk_grow is False
    assert _eq(on.window, off.window)
    assert _eq(on.compress_kv, off.compress_kv)
    assert _eq(on.index_k, off.index_k)
    assert on.offset == off.offset
    # mlx_lm state tuple round-trips identically
    for a, b in zip(on.state, off.state):
        assert _eq(a, b)


def test_layer_cache_rollback_parity(monkeypatch):
    def run(chunk_on):
        rng = np.random.default_rng(4242)
        if chunk_on:
            monkeypatch.setenv("MTPLX_DSV41_KV_CHUNK_GROW", "1")
        else:
            monkeypatch.delenv("MTPLX_DSV41_KV_CHUNK_GROW", raising=False)
        lc = C.LayerAttentionCache(window_size=8, compress_ratio=1, is_kv_source=True)
        for _ in range(5):
            lc.append_window(mx.array(rng.standard_normal((1, 1, 16)).astype(np.float32)))
            lc.append_compress(mx.array(rng.standard_normal((1, 1, 20)).astype(np.float32)))
            lc.append_index_k(mx.array(rng.standard_normal((1, 1, 12)).astype(np.float32)))
            lc.advance(1)
        m = lc.mark()
        for _ in range(3):
            lc.append_window(mx.array(rng.standard_normal((1, 1, 16)).astype(np.float32)))
            lc.append_compress(mx.array(rng.standard_normal((1, 1, 20)).astype(np.float32)))
            lc.append_index_k(mx.array(rng.standard_normal((1, 1, 12)).astype(np.float32)))
            lc.advance(1)
        lc.rollback(m)
        return lc

    on, off = run(True), run(False)
    assert _eq(on.window, off.window)
    assert _eq(on.compress_kv, off.compress_kv)
    assert _eq(on.index_k, off.index_k)
    assert on.offset == off.offset


# ---------------------------------------------------------------------------
# End-to-end: a real tiny-model prefill + decode is byte-identical with the flag
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
    return model, args


def _prefill_decode_logits(model, prompt, steps):
    cache = model.make_cache()
    logits = model(mx.array([list(prompt)]), cache=cache)
    mx.eval(logits)
    outs = [logits[0, -1]]
    tok = int(mx.argmax(logits[0, -1]).item())
    for _ in range(steps):
        logits = model(mx.array([[tok]]), cache=cache)
        mx.eval(logits)
        outs.append(logits[0, -1])
        tok = int(mx.argmax(logits[0, -1]).item())
    return outs, cache


def test_model_prefill_decode_byte_identical_with_chunk_grow(monkeypatch):
    model, _ = _tiny_model()
    prompt = list(range(12))

    monkeypatch.delenv("MTPLX_DSV41_KV_CHUNK_GROW", raising=False)
    base, base_cache = _prefill_decode_logits(model, prompt, steps=10)

    monkeypatch.setenv("MTPLX_DSV41_KV_CHUNK_GROW", "1")
    chunk, chunk_cache = _prefill_decode_logits(model, prompt, steps=10)

    # the cache actually engaged the chunk-grown backing
    assert chunk_cache.layers[0]._chunk_grow is True
    assert base_cache.layers[0]._chunk_grow is False
    assert isinstance(chunk_cache.layers[0]._window, C._GrowBuffer)

    # every decode step's logits are BYTE-identical (exact by construction)
    assert len(base) == len(chunk)
    for i, (a, b) in enumerate(zip(base, chunk)):
        assert bool(mx.all(a == b).item()), f"logits differ at decode step {i}"
