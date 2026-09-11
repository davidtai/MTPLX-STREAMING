"""W26 — the per-sequence engram n-gram history round-trips through the
entry-0 cache-state contract, so a session-bank / SSD warm-turn restore no
longer desyncs the engram hashing (layers 1 and 14) from the restored KV.

Before W26 the engram history (``NgramHashState._buf`` / ``_len``) was NOT part
of ``LayerAttentionCache.state``: a KV-only ``snapshot_cache`` / ``restore_cache``
(the session bank's near-prefix store/restore) left the engram fresh, so a warm
turn computed wrong engram-layer hidden states for the suffix, and
``mlx_lm.save_prompt_cache`` raised ``std::bad_cast`` on the raw numpy state
(docs/deepseek-v41/W22_REPORT.md).  W26 serialises the streaming history as one
``mx.array`` on ``NgramHashState.state`` and carries it on the owning entry's
``state`` / ``replace_state``, so both round-trips reinstate it.

The bit-exactness tests drive the *real* MTPLX generation forward over a tiny
random-config model with a real :class:`~mtplx.engram_v41.EngramV41` hook
attached to layer 1 (so decode logits genuinely depend on the engram history),
mirroring ``tests/test_deepseek_v41_served_generation.py`` and
``tests/test_engram_v41.py``'s synthetic-bank construction.

CPU only.  Run under ``nice -n 19``, without ``-n auto``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

import mtplx.deepseek_v41_convert as dc
from mtplx.cache_state import CacheSnapshot, restore_cache, snapshot_cache
from mtplx.engram_v41 import EngramV41, NgramHashState, n_hash_cols
from mtplx.models.deepseek_v41 import DeepseekV41Cache, Model, ModelArgs
from mtplx.mtp_patch import MTPContract
from mtplx.ngram_row_cache import FileRowReader, NGramRowCache, RowGeometry

mx.set_default_device(mx.cpu)

# editable-install CWD-shadowing guard (memory: run the worktree's code)
import mtplx.engram_v41 as _ev
_REPO_ROOT = Path(__file__).resolve().parents[1]
assert Path(_ev.__file__).resolve().is_relative_to(_REPO_ROOT), (_ev.__file__, _REPO_ROOT)


# ---------------------------------------------------------------------------
# tiny random-config model + a real MTPLXRuntime, with a synthetic engram hook
# ---------------------------------------------------------------------------
def _swa_args(**over) -> ModelArgs:
    base = dict(
        vocab_size=48, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, head_dim=16, qk_rope_head_dim=4,
        q_lora_rank=12, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=2,
        sliding_window=6, window_size=6, swiglu_limit=0.5,
        compress_ratios=[0, 0], kv_source_layer_ids=[], index_source_layer_ids=[],
        candidate_source_layer_id=-1,
    )
    base.update(over)
    return ModelArgs(**base)


def _randomize(model: Model, seed: int = 0) -> None:
    mx.random.seed(seed)
    new = []
    for name, arr in tree_flatten(model.parameters()):
        if arr.ndim == 1 and ("norm_weight" in name or name.endswith("norm.weight")):
            v = 1.0 + 0.2 * mx.random.normal(arr.shape)
        elif "attn_sink" in name:
            v = 0.5 * mx.random.normal(arr.shape)
        else:
            v = 0.3 * mx.random.normal(arr.shape)
        new.append((name, v.astype(mx.float32)))
    model.update(tree_unflatten(new))
    mx.eval(model.parameters())


class _FixedTokenizer:
    eos_token_id = None
    eos_token_ids: set[int] = set()

    def decode(self, tokens):
        return " ".join(str(t) for t in tokens)


# one engram layer (layer id 1), max_ngram 3, 2 heads -> 4 hash cols/token
_MAX_NG, _N_HEADS = 3, 2
_COLS = n_hash_cols(_MAX_NG, _N_HEADS)


def _synthetic_ngram_state(vocab: int) -> NgramHashState:
    """A small streaming n-gram hash config whose token map covers ``vocab``."""
    multipliers = np.array([[13, 8675309, 271828183]], dtype=np.int64)
    primes = np.array([[[101, 103], [107, 109]]], dtype=np.int64)
    flat_offsets = np.array([[0, 101, 204, 311]], dtype=np.int64)
    assert flat_offsets.shape == (1, _COLS)
    return NgramHashState(
        token_map=list(range(vocab)),
        multipliers=multipliers, primes=primes, flat_offsets=flat_offsets,
        pad_compressed=2, max_ngram_size=_MAX_NG, n_heads=_N_HEADS, layer_ids=(1,),
    )


def _synthetic_engram_hook(args: ModelArgs, seed: int = 100) -> EngramV41:
    """A real :class:`EngramV41` over a synthetic affine-q8 bank, sized to the
    model's hc-expanded residual (``[B, L, hc_mult, hidden_size]``)."""
    rng = np.random.default_rng(seed)
    n_emb, head_dim = 800, 256
    dim, hc_mult = int(args.hidden_size), int(args.hc_mult)

    f32 = rng.standard_normal((n_emb, head_dim)).astype(np.float32)
    rec = dc.engram_chunk_records(f32)          # affine q8 / gs64 -> 272 B/row
    bank_dir = tempfile.mkdtemp()
    p = Path(bank_dir) / "bank.bin"
    p.write_bytes(np.ascontiguousarray(rec).tobytes())
    cache = NGramRowCache(
        FileRowReader(p, row_bytes=272, num_rows=n_emb),
        RowGeometry(head_dim, 8, 64), num_rows=n_emb, cache_rows=128,
    )
    wkv_w = (rng.standard_normal((dim * (hc_mult + 1), _COLS * head_dim)) * 0.02).astype(np.float32)
    q_w = rng.standard_normal((hc_mult, dim)).astype(np.float32)
    k_w = rng.standard_normal((hc_mult, dim)).astype(np.float32)
    return EngramV41(
        layer_id=1, layer_hash_index=0, row_cache=cache,
        wkv=EngramV41.dense_wkv(mx.array(wkv_w)),
        q_weight=mx.array(q_w), k_weight=mx.array(k_w),
        dim=dim, hc_mult=hc_mult, norm_eps=float(args.rms_norm_eps),
    )


def _runtime(args: ModelArgs, *, seed: int = 0, engram: bool = False):
    from mtplx.runtime import MTPLXRuntime

    model = Model(args, quantize=False)
    _randomize(model, seed=seed)
    if engram:
        # attach AFTER randomize so the hook's q/k/wkv weights are not overwritten
        model.model.engram_hash = _synthetic_ngram_state(args.vocab_size)
        model.model.layers[1].engram_hook = _synthetic_engram_hook(args, seed=seed + 100)
    return MTPLXRuntime(
        model=model, tokenizer=_FixedTokenizer(), model_path=Path("."),
        mtp_enabled=False, contract=MTPContract(),
    )


def _prompt(n, vocab=48, seed=7):
    rng = np.random.default_rng(seed)
    return [int(v) for v in rng.integers(0, vocab, size=n)]


def _decode_logits(rt, cache, dec_tokens):
    return [np.array(rt.forward_ar(mx.array([[t]]), cache=cache).astype(mx.float32))
            for t in dec_tokens]


# ===========================================================================
# 1. NgramHashState.state / replace_state — the primitive, fresh/advance/trim
#    semantics preserved.
# ===========================================================================
def test_ngram_state_serialise_roundtrip_and_semantics():
    st = _synthetic_ngram_state(48)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 48, size=(1, 14)).astype(np.int64)
    st.advance(ids)

    blob = st.state
    assert isinstance(blob, mx.array)
    assert blob.dtype == mx.int32 and tuple(blob.shape) == (1, 14)

    # restore into a fresh state (shared config via fresh()) -> identical buffer
    restored = st.fresh()
    restored.replace_state(blob)
    assert restored.length == 14
    assert np.array_equal(np.asarray(blob).astype(np.int64), restored._buf)

    # continued advance is bit-exact vs the uninterrupted state
    dec = rng.integers(0, 48, size=(1, 6)).astype(np.int64)
    ref = [st.advance(dec[:, i:i + 1]).copy() for i in range(6)]
    got = [restored.advance(dec[:, i:i + 1]).copy() for i in range(6)]
    assert all(np.array_equal(a, b) for a, b in zip(ref, got))

    # a fresh (never-advanced) state serialises to the canonical empty buffer,
    # and restoring it resets the history (fresh()/reset semantics)
    empty = _synthetic_ngram_state(48).state
    assert tuple(empty.shape) == (1, 0)
    reset_target = st.fresh()
    reset_target.advance(ids)
    reset_target.replace_state(empty)
    assert reset_target.length == 0 and reset_target._buf is None

    # trim still works after a restore (the append-only journal is intact)
    t = st.fresh()
    t.replace_state(blob)
    t.advance(dec)
    t.trim(6)
    assert t.length == 14
    again = [t.advance(dec[:, i:i + 1]).copy() for i in range(6)]
    assert all(np.array_equal(a, b) for a, b in zip(ref, again))


# ===========================================================================
# 2. entry-0 cache state carries the engram; snapshot_cache/restore_cache
#    round-trips it, and continued decode is BIT-EXACT vs uninterrupted decode.
# ===========================================================================
def test_near_prefix_restore_then_decode_is_bit_exact_with_engram():
    args = _swa_args()
    rt = _runtime(args, seed=5, engram=True)
    prompt = _prompt(14)
    dec = _prompt(6, seed=21)

    # (a) uninterrupted: prefill + decode
    ref = rt.make_cache()
    rt.forward_ar(mx.array([prompt]), cache=ref)
    ref_logits = _decode_logits(rt, ref, dec)

    # (b) snapshot after prefill, restore into a FRESH cache (fresh engram),
    #     then decode the same tokens.  This is the session-bank warm-turn path.
    live = rt.make_cache()
    rt.forward_ar(mx.array([prompt]), cache=live)
    snap = snapshot_cache(live)

    # entry 0 owns the engram; its state now carries it as a 6th mx.array leaf.
    assert live[0].engram_state is not None and all(
        e.engram_state is None for e in list(live)[1:]
    )
    assert len(tuple(live[0].state)) == 6
    assert isinstance(tuple(live[0].state)[5], mx.array)

    warm = rt.make_cache()
    assert warm.engram_state.length == 0                  # fresh before restore
    restore_cache(warm, snap)
    assert warm.engram_state.length == len(prompt)         # engram reinstated
    warm_logits = _decode_logits(rt, warm, dec)

    assert all(np.array_equal(a, b) for a, b in zip(ref_logits, warm_logits)), (
        "engram-wired warm-turn decode must be bit-exact after a session-bank restore"
    )


def test_kv_only_restore_desyncs_engram_proving_the_fix_matters():
    """Control: dropping the engram leaf from the snapshot (the pre-W26 KV-only
    5-tuple) desyncs the engram, so decode logits DIFFER; the full 6-tuple
    restore makes them identical."""
    args = _swa_args()
    rt = _runtime(args, seed=6, engram=True)
    prompt = _prompt(14, seed=3)
    dec = _prompt(6, seed=31)

    ref = rt.make_cache()
    rt.forward_ar(mx.array([prompt]), cache=ref)
    ref_logits = _decode_logits(rt, ref, dec)

    live = rt.make_cache()
    rt.forward_ar(mx.array([prompt]), cache=live)
    snap = snapshot_cache(live)

    # pre-W26 behaviour: entry-0 state without the engram leaf (KV only).
    kv_only_states = list(snap.states)
    kv_only_states[0] = tuple(kv_only_states[0])[:5]
    kv_only = CacheSnapshot(states=tuple(kv_only_states), meta_states=snap.meta_states)

    warm_bad = rt.make_cache()
    restore_cache(warm_bad, kv_only)
    assert warm_bad.engram_state.length == 0               # engram left fresh
    bad_logits = _decode_logits(rt, warm_bad, dec)
    assert not all(np.array_equal(a, b) for a, b in zip(ref_logits, bad_logits)), (
        "a KV-only restore must desync the engram (this is the W22 bug)"
    )

    # full W26 restore fixes it.
    warm_ok = rt.make_cache()
    restore_cache(warm_ok, snap)
    ok_logits = _decode_logits(rt, warm_ok, dec)
    assert all(np.array_equal(a, b) for a, b in zip(ref_logits, ok_logits))


# ===========================================================================
# 3. the no-engram path is unchanged (state is the plain 5-tuple; round-trips).
# ===========================================================================
def test_no_engram_path_state_is_unchanged():
    args = _swa_args()
    rt = _runtime(args, seed=8, engram=False)
    cache = rt.make_cache()
    rt.forward_ar(mx.array([_prompt(12)]), cache=cache)

    # no entry owns an engram; every entry's state is the pre-W26 5-tuple.
    for entry in cache:
        assert entry.engram_state is None
        assert len(tuple(entry.state)) == 5

    # KV snapshot/restore still round-trips exactly and decode is bit-exact.
    ref_logits = _decode_logits(rt, cache, _prompt(4, seed=41))

    live = rt.make_cache()
    rt.forward_ar(mx.array([_prompt(12)]), cache=live)
    snap = snapshot_cache(live)
    warm = rt.make_cache()
    restore_cache(warm, snap)
    warm_logits = _decode_logits(rt, warm, _prompt(4, seed=41))
    assert all(np.array_equal(a, b) for a, b in zip(ref_logits, warm_logits))


# ===========================================================================
# 4. SSD on-disk prompt cache: the engram now serialises to an mx.array, so
#    mlx_lm.save_prompt_cache no longer raises std::bad_cast on it, and
#    save_prompt_cache -> load_prompt_cache round-trips the engram history.
# ===========================================================================
def _register_layer_cache_for_load():
    """Make ``LayerAttentionCache`` resolvable by ``mlx_lm.load_prompt_cache``
    (which does ``globals()[type(c).__name__].from_state(...)``), the same
    registration pattern ``mtplx.arrays_cache_patch`` uses for its vendored
    cache class."""
    import mlx_lm.models.cache as cm
    from mtplx.models.deepseek_v41_cache import LayerAttentionCache
    cm.LayerAttentionCache = LayerAttentionCache
    return LayerAttentionCache


def _kv_source_entry_with_engram(prompt_len=8):
    """A single kv-source compressor entry (ratio 2) fed enough tokens to
    complete groups, so every KV lane is a real array (no ``None`` holes) and
    ``save_prompt_cache`` can serialise the whole ``state`` tuple."""
    from mtplx.models.deepseek_v41_cache import LayerAttentionCache

    st = _synthetic_ngram_state(48)
    entry = LayerAttentionCache(window_size=6, compress_ratio=2, is_kv_source=True,
                                engram_state=st)
    rng = np.random.default_rng(1)
    ids = rng.integers(0, 48, size=(1, prompt_len)).astype(np.int64)
    # advance the engram history and populate the KV lanes with plausible rows
    entry.engram_state.advance(ids)
    head_dim = 16
    entry.append_window(mx.array(rng.standard_normal((1, prompt_len, head_dim)).astype(np.float32)))
    groups = prompt_len // 2
    entry.append_compress(mx.array(rng.standard_normal((1, groups, head_dim)).astype(np.float32)))
    entry.append_index_k(mx.array(rng.standard_normal((1, groups, 8)).astype(np.float32)))
    entry.comp_state.push(
        mx.array(rng.standard_normal((1, prompt_len, head_dim)).astype(np.float32)),
        mx.array(rng.standard_normal((1, prompt_len, head_dim)).astype(np.float32)),
    )
    entry.offset = prompt_len
    return entry, ids


def test_ssd_save_load_prompt_cache_roundtrips_engram_history():
    from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache

    LayerAttentionCache = _register_layer_cache_for_load()
    entry, ids = _kv_source_entry_with_engram(prompt_len=8)

    # every state leaf is an mx.array now (KV lanes + engram) -> save succeeds,
    # where the pre-W26 raw-numpy engram state raised std::bad_cast.
    state = tuple(entry.state)
    assert len(state) == 6
    assert all(isinstance(v, mx.array) for v in state)

    path = os.path.join(tempfile.mkdtemp(), "sc.safetensors")
    save_prompt_cache(path, [entry])            # must NOT raise
    assert os.path.exists(path)

    loaded = load_prompt_cache(path)
    assert len(loaded) == 1
    loaded_entry = loaded[0]
    assert isinstance(loaded_entry, LayerAttentionCache)
    assert int(loaded_entry.offset) == int(entry.offset)

    # KV lanes round-trip bit-exactly
    for a, b in zip(tuple(entry.state)[:5], tuple(loaded_entry.state)[:5]):
        assert np.array_equal(np.array(a), np.array(b))

    # the engram history round-trips: rehydrate the loaded blob into a fresh
    # NgramHashState carrying the shared hash config (the production restore
    # re-attaches the model's engram config, then fills the saved buffer), and
    # verify continued hashing is bit-exact vs the original.
    rehydrated = _synthetic_ngram_state(48)
    rehydrated.replace_state(loaded_entry.loaded_engram_state)
    assert rehydrated.length == entry.engram_state.length
    assert np.array_equal(rehydrated._buf, entry.engram_state._buf)

    dec = np.random.default_rng(9).integers(0, 48, size=(1, 5)).astype(np.int64)
    ref = [entry.engram_state.advance(dec[:, i:i + 1]).copy() for i in range(5)]
    got = [rehydrated.advance(dec[:, i:i + 1]).copy() for i in range(5)]
    assert all(np.array_equal(a, b) for a, b in zip(ref, got))
