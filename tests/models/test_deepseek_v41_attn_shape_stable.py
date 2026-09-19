"""W90 -- MTPLX_DSV41_ATTN_SHAPE_STABLE: shared selected-compressed-KV gather.

A DISPATCH-COUNT cleanup, NOT the in-situ decode-floor fix.  All Reuse / Reindex /
Full layers of a group read the SAME ``(compress_kv, selected_idx)`` pair, so the
shipped K30 path issues the compressed-lane gather (~3 tiny host dispatches) once per
layer; ``MTPLX_DSV41_ATTN_SHAPE_STABLE`` gathers it ONCE per source and shares the
bounded ``[b, s, k, hd]`` result down the group, so only the first layer of a group
issues it (est. ~38 -> ~8 compress gathers/token on the real backbone; 6 -> 3 on this
tiny config; <= ~1% of the token).  Small-M gated (decode/verify only; a prefill-
chunk cache would pin ~17 GB at the 16K cell).

This does NOT explain the mode-independent ~4 ms/layer in-situ floor: the earlier
"O(T) source reference" claim is FALSIFIED -- ``docs/deepseek-v41/receipts/gpu-
windows/window-34/w78-in-model.json`` shows ``attn.swa_only`` (no ``compress_kv``)
at 7.447 ms/layer vs ``attn.reuse`` 6.960 (swa costs MORE while referencing no
compressed store), and the isolated bench references O(T) per dispatch yet is flat.
The floor tracks GPU DVFS (macmon: 71 C not-thermal, ~45% busy, 580-1381 MHz);
``swa_only`` is the decisive per-mode control (see ``W90_ATTN_IN_SITU.md``).

The lever is a pure caching of the already-deterministic K30 gather, so it is
BYTE-IDENTICAL to the shipped selected-key path.  These tests prove, on a tiny CPU
config (no artifact):

  * the env resolver parses truthy / falsy exactly like the sibling levers;
  * ON vs OFF is ``mx.array_equal`` over 64 decode steps (token, logits) AND over a
    small-M ``K+1`` verify batch shape (rows > 1, s > 1);
  * the shared gather cuts the count of O(T) ``compress_kv``-source gathers per
    token (each group re-gathers once, not once per reuse layer) while the total
    gathered rows are identical;
  * OFF (or no ``shared``) is the plain per-layer gather (unchanged shipped path);
  * it composes byte-identically with ``MTPLX_DSV41_SELECT_FENCE`` and
    ``MTPLX_DSV41_KV_CHUNK_GROW``.

CPU-pinned; tiny random config; no artifact load.  Run under ``nice -n 19`` without
``-n auto`` (one test file per pytest process, < 1.5 GB).
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

import mtplx.models.deepseek_v41 as dv41  # noqa: E402
from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures (mirror tests/models/test_deepseek_v41_stage_timing.py)
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


def _new_model(seed=1, **over):
    args = _csa_args(**over)
    model = Model(args)
    _randomize(model, seed=seed)
    return model, args


def _prefill(model, args, s, seed, chunk=0):
    ids = mx.array(np.random.RandomState(seed).randint(0, args.vocab_size, size=(1, s)))
    cache = model.make_cache()
    logits = model(ids, cache=cache, prefill_chunk=chunk)
    mx.eval(logits)
    token = int(mx.argmax(logits[0, -1]).item())
    return cache, token


def _decode(model, cache, token, n):
    """Greedy decode ``n`` steps; return (tokens, logits-per-step)."""
    toks, outs = [], []
    for _ in range(n):
        logits = model(mx.array([[token]]), cache=cache)
        mx.eval(logits)
        token = int(mx.argmax(logits[0, -1]).item())
        toks.append(token)
        outs.append(np.array(logits))
    return toks, outs


# ---------------------------------------------------------------------------
# 1. env resolver parses like the sibling levers
# ---------------------------------------------------------------------------
def test_resolve_attn_shape_stable_parsing(monkeypatch):
    assert dv41._resolve_attn_shape_stable(raw=None) is False  # default OFF (unset)
    for off in ("", "0", "false", "no", "off", "auto", "AUTO", " Off "):
        assert dv41._resolve_attn_shape_stable(raw=off) is False, off
    for on in ("1", "true", "on", "yes", "TRUE", " 1 "):
        assert dv41._resolve_attn_shape_stable(raw=on) is True, on
    # read at use through the env (serving stamps the key after import)
    monkeypatch.setenv(dv41._ATTN_SHAPE_STABLE_ENV, "1")
    assert dv41._resolve_attn_shape_stable() is True
    monkeypatch.setenv(dv41._ATTN_SHAPE_STABLE_ENV, "0")
    assert dv41._resolve_attn_shape_stable() is False


# ---------------------------------------------------------------------------
# 2. byte-identical ON vs OFF over 64 decode steps (the core contract)
# ---------------------------------------------------------------------------
def _run_decode(shape_stable, monkeypatch, *, prefill=48, steps=64,
                select_fence="0", kv_chunk_grow="0"):
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    monkeypatch.setenv("MTPLX_DSV41_ATTN_SHAPE_STABLE", shape_stable)
    monkeypatch.setenv("MTPLX_DSV41_SELECT_FENCE", select_fence)
    monkeypatch.setenv("MTPLX_DSV41_KV_CHUNK_GROW", kv_chunk_grow)
    model, args = _new_model(seed=1)
    cache, token = _prefill(model, args, s=prefill, seed=0, chunk=16)
    return _decode(model, cache, token, steps)


def test_decode_byte_identical_on_off_64_steps(monkeypatch):
    _toks_off, off = _run_decode("0", monkeypatch)
    _toks_on, on = _run_decode("1", monkeypatch)
    assert len(off) == len(on) == 64
    for i, (a, b) in enumerate(zip(off, on)):
        assert np.array_equal(a, b), (
            f"decode step {i} differs (max {np.max(np.abs(a - b))})"
        )


def test_decode_tokens_identical_on_off(monkeypatch):
    toks_off, _ = _run_decode("0", monkeypatch)
    toks_on, _ = _run_decode("1", monkeypatch)
    assert toks_off == toks_on


# ---------------------------------------------------------------------------
# 3. byte-identical composed with SELECT_FENCE and KV_CHUNK_GROW
# ---------------------------------------------------------------------------
def test_byte_identical_composes_with_select_fence(monkeypatch):
    _t0, off = _run_decode("0", monkeypatch, steps=32, select_fence="1")
    _t1, on = _run_decode("1", monkeypatch, steps=32, select_fence="1")
    for i, (a, b) in enumerate(zip(off, on)):
        assert np.array_equal(a, b), f"step {i} differs under select_fence"


def test_byte_identical_composes_with_kv_chunk_grow(monkeypatch):
    _t0, off = _run_decode("0", monkeypatch, steps=32, kv_chunk_grow="1")
    _t1, on = _run_decode("1", monkeypatch, steps=32, kv_chunk_grow="1")
    for i, (a, b) in enumerate(zip(off, on)):
        assert np.array_equal(a, b), f"step {i} differs under kv_chunk_grow"


# ---------------------------------------------------------------------------
# 4. the shared gather cuts O(T) compress-source gathers, same rows gathered
# ---------------------------------------------------------------------------
def _count_gathers(shape_stable, monkeypatch, prefill=256):
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    monkeypatch.setenv("MTPLX_DSV41_ATTN_SHAPE_STABLE", shape_stable)
    monkeypatch.setenv("MTPLX_DSV41_SELECT_FENCE", "0")
    monkeypatch.setenv("MTPLX_DSV41_KV_CHUNK_GROW", "0")
    model, args = _new_model(seed=1)
    cache, token = _prefill(model, args, s=prefill, seed=0, chunk=64)

    counts = {"compress_src": 0, "window_src": 0}
    orig = dv41._gather_rows
    win = args.window_size

    def spy(source, idx, valid):
        n = int(source.shape[1])
        # window source rows ~ current length (>> window_size); compressed source
        # rows ~ length/ratio.  Both scale with T; a bounded pre-gathered operand
        # never re-enters _gather_rows.  Tag by which store this is (window store is
        # the largest; compressed is ~n/2) -- here we simply count sources with
        # rows > window_size (the T-scaling gathers) as "big" and split by the last
        # axis width vs head_dim to tell window (hd) from compressed (hd) -- both hd,
        # so instead count all T-scaling gathers and assert the ON count is strictly
        # smaller (the shared compress gather removed some).
        if n > win:
            counts["compress_src"] += 1  # T-scaling source reference
        return orig(source, idx, valid)

    dv41._gather_rows = spy
    try:
        _decode(model, cache, token, 1)  # one decode token
    finally:
        dv41._gather_rows = orig
    return counts["compress_src"]


def test_shared_gather_reduces_tscale_source_references(monkeypatch):
    off = _count_gathers("0", monkeypatch)
    on = _count_gathers("1", monkeypatch)
    # ON must issue strictly fewer T-scaling gather-source references per token: the
    # compressed lane is gathered once per (compress_kv, selected_idx) source and
    # shared, so the Reuse layers that shipped a per-layer compress gather no longer
    # re-enter _gather_rows against the O(T) store.  (The per-layer window-store
    # gathers are unchanged -- those are the W80 ring's lane.)
    assert on < off, (on, off)


# ---------------------------------------------------------------------------
# 5. OFF is the plain per-layer gather (no shared cache field mutated)
# ---------------------------------------------------------------------------
def test_off_does_not_populate_shared_cache(monkeypatch):
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    monkeypatch.setenv("MTPLX_DSV41_ATTN_SHAPE_STABLE", "0")
    model, args = _new_model(seed=1)
    cache, token = _prefill(model, args, s=64, seed=0, chunk=16)
    seen = {"cached": False}
    orig = dv41._selected_compress_gather

    def spy(compress_kv, comp_idx, shared):
        out = orig(compress_kv, comp_idx, shared)
        if shared is not None and getattr(shared, "_sel_cmp_kvg", None) is not None:
            seen["cached"] = True
        return out

    dv41._selected_compress_gather = spy
    try:
        _decode(model, cache, token, 3)
    finally:
        dv41._selected_compress_gather = orig
    assert seen["cached"] is False  # OFF never caches on shared


def test_on_populates_shared_cache(monkeypatch):
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    monkeypatch.setenv("MTPLX_DSV41_ATTN_SHAPE_STABLE", "1")
    model, args = _new_model(seed=1)
    cache, token = _prefill(model, args, s=64, seed=0, chunk=16)
    seen = {"cached": False}
    orig = dv41._selected_compress_gather

    def spy(compress_kv, comp_idx, shared):
        out = orig(compress_kv, comp_idx, shared)
        if shared is not None and getattr(shared, "_sel_cmp_kvg", None) is not None:
            seen["cached"] = True
        return out

    dv41._selected_compress_gather = spy
    try:
        _decode(model, cache, token, 3)
    finally:
        dv41._selected_compress_gather = orig
    assert seen["cached"] is True  # ON caches the shared gather on the runtime


# ---------------------------------------------------------------------------
# 6. helper is byte-identical to a direct per-layer _gather_rows
# ---------------------------------------------------------------------------
def test_helper_matches_plain_gather(monkeypatch):
    monkeypatch.setenv("MTPLX_DSV41_ATTN_SHAPE_STABLE", "1")
    b, n, d, s, k = 1, 300, 16, 1, 5
    compress_kv = mx.random.normal((b, n, d))
    comp_idx = mx.array(np.array([[[3, 17, -1, 200, 42]]], dtype=np.int32))
    comp_valid = comp_idx >= 0

    class _Shared:  # minimal per-forward runtime double
        pass

    shared = _Shared()
    plain = dv41._gather_rows(compress_kv, comp_idx, comp_valid)
    kvg1, v1 = dv41._selected_compress_gather(compress_kv, comp_idx, shared)  # gathers + caches
    kvg2, v2 = dv41._selected_compress_gather(compress_kv, comp_idx, shared)  # cache hit
    assert np.array_equal(np.array(kvg1), np.array(plain))
    assert np.array_equal(np.array(kvg2), np.array(plain))
    assert np.array_equal(np.array(v1), np.array(comp_valid))
    # a different source array identity misses the cache and re-gathers correctly
    comp_idx2 = mx.array(np.array([[[1, 2, 3, 4, -1]]], dtype=np.int32))
    kvg3, _ = dv41._selected_compress_gather(compress_kv, comp_idx2, shared)
    assert np.array_equal(
        np.array(kvg3), np.array(dv41._gather_rows(compress_kv, comp_idx2, comp_idx2 >= 0))
    )


# ---------------------------------------------------------------------------
# 7. (review) verify-shaped forward (rows>1, s<=8) is byte-identical ON vs OFF
# ---------------------------------------------------------------------------
def _run_with_verify(stable, monkeypatch, *, window_ring="0", layer_major="0",
                     kv_chunk_grow="0"):
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    monkeypatch.setenv("MTPLX_DSV41_ATTN_SHAPE_STABLE", stable)
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING", window_ring)
    monkeypatch.setenv("MTPLX_DSV41_PREFILL_LAYER_MAJOR", layer_major)
    monkeypatch.setenv("MTPLX_DSV41_KV_CHUNK_GROW", kv_chunk_grow)
    model, args = _new_model(seed=1)
    cache, token = _prefill(model, args, s=48, seed=0, chunk=16)
    dec = _decode(model, cache, token, 16)[1]
    # a small-M verify-shaped forward (s=4 rows > 1) on the live cache
    vids = mx.array(np.random.RandomState(7).randint(0, args.vocab_size, size=(1, 4)))
    vlog = model(vids, cache=cache)
    mx.eval(vlog)
    tok = int(mx.argmax(vlog[0, -1]).item())
    dec2 = _decode(model, cache, tok, 8)[1]
    return dec, np.array(vlog), dec2


def test_verify_shaped_forward_byte_identical(monkeypatch):
    d_off, v_off, d2_off = _run_with_verify("0", monkeypatch)
    d_on, v_on, d2_on = _run_with_verify("1", monkeypatch)
    assert np.array_equal(v_off, v_on), "s=4 verify forward differs"
    for i, (a, b) in enumerate(zip(d_off, d_on)):
        assert np.array_equal(a, b), f"pre-verify decode {i} differs"
    for i, (a, b) in enumerate(zip(d2_off, d2_on)):
        assert np.array_equal(a, b), f"post-verify decode {i} differs"


def test_byte_identical_ring_and_layer_major(monkeypatch):
    for combo in (
        {"window_ring": "1"},
        {"window_ring": "1", "layer_major": "1"},
        {"layer_major": "1"},
        {"kv_chunk_grow": "1", "layer_major": "1"},
    ):
        d_off, v_off, d2_off = _run_with_verify("0", monkeypatch, **combo)
        d_on, v_on, d2_on = _run_with_verify("1", monkeypatch, **combo)
        assert np.array_equal(v_off, v_on), combo
        assert all(np.array_equal(a, b) for a, b in zip(d_off, d_on)), combo
        assert all(np.array_equal(a, b) for a, b in zip(d2_off, d2_on)), combo


# ---------------------------------------------------------------------------
# 8. (review) exact hit-count: hits == #Reuse, misses == #Full+#Reindex per token
# ---------------------------------------------------------------------------
def _hit_miss_per_token(monkeypatch, *, window_ring="0", kv_chunk_grow="0"):
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    monkeypatch.setenv("MTPLX_DSV41_ATTN_SHAPE_STABLE", "1")
    monkeypatch.setenv("MTPLX_DSV41_WINDOW_RING", window_ring)
    monkeypatch.setenv("MTPLX_DSV41_KV_CHUNK_GROW", kv_chunk_grow)
    model, args = _new_model(seed=1)
    cache, token = _prefill(model, args, s=64, seed=0, chunk=16)

    stats = {"hit": 0, "miss": 0}
    orig = dv41._selected_compress_gather

    def spy(compress_kv, comp_idx, shared):
        c = getattr(shared, "_sel_cmp_kvg", None)
        if c is not None and c[0] is compress_kv and c[1] is comp_idx:
            stats["hit"] += 1
        else:
            stats["miss"] += 1
        return orig(compress_kv, comp_idx, shared)

    dv41._selected_compress_gather = spy
    try:
        _decode(model, cache, token, 4)  # 4 decode tokens
    finally:
        dv41._selected_compress_gather = orig
    modes = args.layer_modes
    n_reuse = modes.count("reuse")
    n_src = modes.count("full") + modes.count("reindex")  # layers that (re)publish a selection
    return stats, n_reuse, n_src


def test_exact_hit_miss_counts_ring(monkeypatch):
    stats, n_reuse, n_src = _hit_miss_per_token(monkeypatch, window_ring="1")
    # per token: every Reuse layer hits the shared cache; every Full/Reindex layer
    # (re)publishes a selection -> a fresh (compress_kv|selected_idx) identity -> miss.
    assert stats["hit"] == n_reuse * 4, (stats, n_reuse)
    assert stats["miss"] == n_src * 4, (stats, n_src)


def test_exact_hit_miss_counts_chunk_grow(monkeypatch):
    stats, n_reuse, n_src = _hit_miss_per_token(monkeypatch, kv_chunk_grow="1")
    assert stats["hit"] == n_reuse * 4, (stats, n_reuse)
    assert stats["miss"] == n_src * 4, (stats, n_src)


# ---------------------------------------------------------------------------
# 9. (review) small-M gate: a prefill chunk (rows > 8) NEVER caches (no pinning)
# ---------------------------------------------------------------------------
def _run_gate_probe(monkeypatch, *, prefill_s, chunk, layer_major="0", decode_n=2):
    monkeypatch.setenv("MTPLX_DSV41_SELECTED_KEYS", "1")
    monkeypatch.setenv("MTPLX_DSV41_ATTN_SHAPE_STABLE", "1")
    monkeypatch.setenv("MTPLX_DSV41_PREFILL_LAYER_MAJOR", layer_major)
    model, args = _new_model(seed=1)
    seen = {"prefill_calls": 0, "prefill_cached": 0, "peak_pinned_bytes": 0,
            "decode_cached": 0}
    live: dict = {}
    orig = dv41._selected_compress_gather

    def spy(compress_kv, comp_idx, shared):
        rows = int(comp_idx.shape[0]) * int(comp_idx.shape[1])
        out = orig(compress_kv, comp_idx, shared)
        c = None if shared is None else getattr(shared, "_sel_cmp_kvg", None)
        cached = c is not None
        if rows > dv41._DECODE_ATTN_KERNEL_MAX_ROWS:  # a prefill chunk
            seen["prefill_calls"] += 1
            if cached:
                seen["prefill_cached"] += 1
                live[id(shared)] = c[2].nbytes
                seen["peak_pinned_bytes"] = max(seen["peak_pinned_bytes"],
                                                 sum(live.values()))
        elif cached:  # decode / verify
            seen["decode_cached"] += 1
        return out

    dv41._selected_compress_gather = spy
    try:
        cache, token = _prefill(model, args, s=prefill_s, seed=0, chunk=chunk)
        _decode(model, cache, token, decode_n)
    finally:
        dv41._selected_compress_gather = orig
    return seen


def test_prefill_chunk_never_caches_small_m_gate(monkeypatch):
    # one-shot prefill (rows = 40 > 8): gathers, but the small-M gate never caches
    seen = _run_gate_probe(monkeypatch, prefill_s=40, chunk=0)
    assert seen["prefill_calls"] > 0
    assert seen["prefill_cached"] == 0            # nothing pinned at prefill
    assert seen["peak_pinned_bytes"] == 0
    assert seen["decode_cached"] > 0              # decode (rows=1) still shares


def test_layer_major_chunked_prefill_pins_nothing(monkeypatch):
    # the review's pin scenario: layer-major CHUNKED prefill (chunk=16 > 8).  Without
    # the small-M gate each per-chunk shared runtime would pin its [chunk,k,hd] operand
    # for the whole group (OOM at 16K/64K); the gate must keep the pinned bytes at 0.
    seen = _run_gate_probe(monkeypatch, prefill_s=64, chunk=16, layer_major="1")
    assert seen["prefill_calls"] > 0
    assert seen["prefill_cached"] == 0
    assert seen["peak_pinned_bytes"] == 0
    assert seen["decode_cached"] > 0
