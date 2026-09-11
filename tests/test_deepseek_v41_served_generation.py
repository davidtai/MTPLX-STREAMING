"""Served-order gate for the DeepSeek-V4.1 cache: drive the real MTPLX
generation path over a tiny random-config model on CPU (no artifact).

The streamed artifact loaded and ``/health`` was up, but the first chat
completion died in ``generation.restore_or_prefill_prompt_state -> _prefill ->
_cache_has_recurrent_entries`` with ``TypeError: 'DeepseekV41Cache' object is
not iterable``: the model handed the serve path a bespoke container where the
mlx_lm convention (and every other MTPLX backend) is a *list of per-layer cache
entries*.  These tests pin the cache in the shape the serve path actually
consumes it in -- built through ``MTPLXRuntime.make_cache`` (so the runtime's
``configure_*`` layout hooks run over it), iterated by
``_cache_has_recurrent_entries``, prefilled by
``restore_or_prefill_prompt_state``, decoded by ``forward_ar``, and rewound by
the runtime's own verify/rollback API (``rollback_after_verify`` /
``trim_verified_window_without_snapshot`` / ``_trim_cache_to_offset``) -- the
exact seam MTP verify will drive.  Rollback exactness is asserted as *bit*
equality of both the stored state and the next-token logits, because a rejected
speculative tail that does not fully un-decode silently corrupts every later
token.

CPU-pinned (test-scoped) so MLX fp32 is bit-exact; a tiny shrunk config, random
weights, no checkpoint, no torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mlx.core as mx
import pytest

pytest.importorskip("mlx.core")
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from mtplx import generation  # noqa: E402
from mtplx.cache_state import (  # noqa: E402
    rollback_after_verify,
    snapshot_untrimmable_cache,
    trim_verified_window_without_snapshot,
)
from mtplx.generation import (  # noqa: E402
    _cache_has_recurrent_entries,
    _cache_offset,
    _trim_cache_to_offset,
    generate_ar,
    restore_or_prefill_prompt_state,
)
from mtplx.models.deepseek_v41 import DeepseekV41Cache, Model, ModelArgs  # noqa: E402
from mtplx.mtp_patch import MTPContract  # noqa: E402
from mtplx.sampling import SamplerConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_default_device():
    # Test-scoped CPU pin: a module-level set_default_device would leak into the
    # Metal bit-exactness suites collected later in the same process.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


# ---------------------------------------------------------------------------
# tiny random-config model + a real MTPLXRuntime over it (no artifact)
# ---------------------------------------------------------------------------
def _swa_args(**over) -> ModelArgs:
    """Two pure sliding-window layers -- the smallest text forward."""
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


def _csa_args(**over) -> ModelArgs:
    """The full §0 layer menu: swa, swa, full-r2, reuse, reuse, full-r1(cand),
    reindex, reuse -- so a rollback exercises every lane (window ring, the
    compressor frontier + emitted rows, the indexer's own compressor)."""
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


def _runtime(args: ModelArgs, *, seed: int = 0, engram: bool = False):
    """A real ``MTPLXRuntime`` over the shrunk dense model, AR lane (no MTP
    graft needed to exercise the trunk cache the verify path rewinds)."""
    from mtplx.runtime import MTPLXRuntime

    model = Model(args, quantize=False)
    _randomize(model, seed=seed)
    if engram:
        model.model.engram_hash = _ngram_state(args.vocab_size)
    return MTPLXRuntime(
        model=model,
        tokenizer=_FixedTokenizer(),
        model_path=Path("."),
        mtp_enabled=False,
        contract=MTPContract(),
    )


def _ngram_state(vocab: int):
    """A small streaming n-gram hash config whose token map covers ``vocab`` --
    the per-sequence engram history the backbone advances every forward."""
    from mtplx.engram_v41 import NgramHashState, n_hash_cols

    max_ng, n_heads = 3, 2
    cols = n_hash_cols(max_ng, n_heads)
    multipliers = np.array([[13, 8675309, 271828183]], dtype=np.int64)
    primes = np.array([[[101, 103], [107, 109]]], dtype=np.int64)
    flat_offsets = np.array([[0, 101, 204, 311]], dtype=np.int64)
    assert flat_offsets.shape == (1, cols)
    return NgramHashState(
        token_map=list(range(vocab)),
        multipliers=multipliers,
        primes=primes,
        flat_offsets=flat_offsets,
        pad_compressed=2,
        max_ngram_size=max_ng,
        n_heads=n_heads,
        layer_ids=(1,),
    )


def _prompt(n, vocab=48, seed=7):
    rng = np.random.default_rng(seed)
    return [int(v) for v in rng.integers(0, vocab, size=n)]


# ---------------------------------------------------------------------------
# 1. make_cache is the served list-of-per-layer-caches shape
# ---------------------------------------------------------------------------
def test_runtime_make_cache_is_the_served_list_shape():
    args = _swa_args()
    rt = _runtime(args, seed=1)
    cache = rt.make_cache()  # runs configure_owned_recurrent / configure_tail_owned

    # iterable list of per-layer entries (the exact operations the serve path does)
    assert isinstance(cache, DeepseekV41Cache)
    assert len(cache) == args.num_hidden_layers
    entries = list(cache)
    assert len(entries) == args.num_hidden_layers
    assert cache[0] is entries[0]
    # the failing call from the original traceback now returns cleanly: our
    # entries are all trimmable attention KV, none recurrent.
    assert _cache_has_recurrent_entries(cache) is False
    for entry in cache:
        assert entry.is_trimmable() is True
        assert int(entry.offset) == 0
        assert callable(entry.trim)


# ---------------------------------------------------------------------------
# 2. served order: restore_or_prefill_prompt_state -> decode a few steps
# ---------------------------------------------------------------------------
def test_restore_or_prefill_then_decode_emits_tokens():
    args = _swa_args()
    rt = _runtime(args, seed=2)
    prompt = _prompt(17)

    # the exact call that used to raise (restore_or_prefill -> _prefill ->
    # _cache_has_recurrent_entries) now completes and returns a prefilled cache.
    ps = restore_or_prefill_prompt_state(rt, prompt)
    cache = ps.trunk_cache
    assert len(cache) == args.num_hidden_layers
    assert _cache_offset(cache) == len(prompt)
    assert ps.logits is not None
    vocab = args.vocab_size

    # a few real decode steps through the runtime forward, greedy-sampled.
    # (restore_or_prefill returns final-row logits [B, vocab]; forward_ar returns
    # [B, 1, vocab] -- take the last row from whatever shape came back.)
    def _last_row(logits):
        row = logits if logits.ndim == 2 else logits[:, -1, :]
        return int(mx.argmax(row.astype(mx.float32), axis=-1)[0])

    tokens = []
    logits = ps.logits
    for _ in range(6):
        tok = _last_row(logits)
        assert 0 <= tok < vocab
        tokens.append(tok)
        logits = rt.forward_ar(mx.array([[tok]]), cache=cache)
    assert len(tokens) == 6
    assert _cache_offset(cache) == len(prompt) + 6


def test_generate_ar_over_served_cache_emits_tokens():
    args = _swa_args()
    rt = _runtime(args, seed=3)
    out = generate_ar(
        rt,
        _prompt(20),
        max_tokens=12,
        sampler=SamplerConfig(temperature=0.0),
        stop_token_ids=set(),
    )
    assert len(out.tokens) == 12
    assert all(0 <= t < args.vocab_size for t in out.tokens)


# ---------------------------------------------------------------------------
# 3. trim/rollback through the runtime's own verify API restores state exactly
#    (this is what MTP verify drives on a rejected speculative tail)
# ---------------------------------------------------------------------------
def _primed_cache(rt, ids, prompt_len, total):
    cache = rt.make_cache()
    rt.forward_ar(ids[:, :prompt_len], cache=cache)
    for t in range(prompt_len, total):
        rt.forward_ar(ids[:, t:t + 1], cache=cache)
    return cache


def _field(v):
    return None if v is None else np.array(v)


def _cache_state_fields(cache):
    return [[_field(v) for v in entry.state] for entry in cache]


def _states_bit_equal(ref, got):
    if len(ref) != len(got):
        return False
    for lref, lgot in zip(ref, got):
        for a, b in zip(lref, lgot):
            if a is None or b is None:
                if not (a is None and b is None):
                    return False
            elif a.shape != b.shape or not np.array_equal(a, b):
                return False
    return True


def test_rollback_after_verify_restores_state_and_logits_bit_exact():
    args = _csa_args()
    rt = _runtime(args, seed=4)
    rng = np.random.default_rng(11)
    prompt_len, decoded, k, tail = 10, 6, 3, 5
    total = prompt_len + decoded
    ids = mx.array(rng.integers(0, args.vocab_size, size=(1, total + k + tail)).tolist())

    ref = _primed_cache(rt, ids, prompt_len, total)
    got = _primed_cache(rt, ids, prompt_len, total)

    # the verify shape: K speculative tokens in one forward, then reject them.
    snapshot = snapshot_untrimmable_cache(got)  # all-None: every entry trimmable
    rt.forward_ar(ids[:, total:total + k], cache=got)
    assert _cache_offset(got) == total + k
    rollback_after_verify(got, snapshot, k)

    assert _cache_offset(got) == total
    assert _states_bit_equal(_cache_state_fields(ref), _cache_state_fields(got)), (
        "V4.1 keeps full append-only history, so a rejected tail must un-decode "
        "to bit-identical state on every lane"
    )
    # the headline form: the continuation's logits are bit-identical to the
    # never-decoded path, token after token.
    ref_logits = [np.array(rt.forward_ar(ids[:, t:t + 1], cache=ref))
                  for t in range(total, total + tail)]
    got_logits = [np.array(rt.forward_ar(ids[:, t:t + 1], cache=got))
                  for t in range(total, total + tail)]
    assert all(np.array_equal(a, b) for a, b in zip(ref_logits, got_logits))


def test_snapshot_free_verify_trim_and_near_prefix_trim():
    args = _csa_args()
    rt = _runtime(args, seed=6)
    rng = np.random.default_rng(13)
    prompt_len, decoded, k = 9, 5, 4
    total = prompt_len + decoded
    ids = mx.array(rng.integers(0, args.vocab_size, size=(1, total + k)).tolist())

    # (a) snapshot-free repair (MTPLX_SKIP_VERIFY_SNAPSHOT lane): keep 1 of K.
    got = _primed_cache(rt, ids, prompt_len, total)
    rt.forward_ar(ids[:, total:total + k], cache=got)
    assert trim_verified_window_without_snapshot(got, verified_tokens=k, keep_tokens=1)
    assert _cache_offset(got) == total + 1

    # (b) near-prefix restore trim (session bank / _trim_cache_to_offset).
    got2 = _primed_cache(rt, ids, prompt_len, total)
    rt.forward_ar(ids[:, total:total + k], cache=got2)
    assert _cache_offset(got2) == total + k
    assert _trim_cache_to_offset(got2, total) is True
    assert _cache_offset(got2) == total
    # a cache carrying a non-trimmable entry must NOT be snapshot-free repaired.
    class _Recurrent:  # no trim() -> not trimmable
        offset = total + k
        state = object()

    assert not trim_verified_window_without_snapshot(
        [got2[0], _Recurrent()], verified_tokens=k, keep_tokens=1
    )


# ---------------------------------------------------------------------------
# 4. the shared engram history rides the owning entry's trim through the serve
#    path (the one piece V4.1 adds over deepseek_v4's independent per-layer caches)
# ---------------------------------------------------------------------------
def test_engram_history_rides_entry0_trim_through_the_served_path():
    args = _swa_args(compress_ratios=[0, 0])
    rt = _runtime(args, seed=5, engram=True)

    cache = rt.make_cache()
    owners = [i for i, e in enumerate(cache) if e.engram_state is not None]
    assert owners == [0], "exactly the first entry owns the per-sequence engram"
    assert cache.engram_state is not None  # reachable for the backbone to advance

    # the served loop advances the engram every forward.
    prompt = _prompt(12)
    out = generate_ar(
        rt, prompt, max_tokens=6,
        sampler=SamplerConfig(temperature=0.0), stop_token_ids=set(),
    )
    assert len(out.tokens) == 6

    # a verify rollback rewinds the engram history in lockstep with the offset,
    # through the owning entry -- exactly once, not once per layer.
    fresh = rt.make_cache()
    ids = mx.array([prompt])
    rt.forward_ar(ids, cache=fresh)
    length_before = int(fresh.engram_state.length)
    assert length_before == len(prompt) == _cache_offset(fresh)

    snapshot = snapshot_untrimmable_cache(fresh)
    extra = mx.array(np.random.default_rng(1).integers(0, args.vocab_size, size=(1, 3)).tolist())
    rt.forward_ar(extra, cache=fresh)
    assert int(fresh.engram_state.length) == length_before + 3
    rollback_after_verify(fresh, snapshot, 3)
    assert _cache_offset(fresh) == length_before
    assert int(fresh.engram_state.length) == length_before


# ---------------------------------------------------------------------------
# 5. session/SSD prompt-cache save-restore: what round-trips, and the exact
#    reason near-prefix session restore stays gated off for the engram artifact
#    (see docs/deepseek-v41/W22_REPORT.md "session/SSD save-restore").
# ---------------------------------------------------------------------------
def test_session_snapshot_roundtrips_kv_but_not_engram_documents_the_gate():
    from mtplx.cache_state import restore_cache, snapshot_cache

    args = _swa_args(compress_ratios=[0, 0, 2, 2], num_hidden_layers=4,
                     kv_source_layer_ids=[2], index_source_layer_ids=[2])
    rt = _runtime(args, seed=8, engram=True)
    ids = mx.array([_prompt(14)])

    cache = rt.make_cache()
    rt.forward_ar(ids, cache=cache)
    kv_before = _cache_state_fields(cache)
    off_before = _cache_offset(cache)
    eng_before = int(cache.engram_state.length)
    assert off_before == eng_before == 14

    # the in-memory session-bank store path: snapshot_cache reads entry.state +
    # entry.meta_state for every entry.
    snapshot = snapshot_cache(cache)

    # advance the live cache (as a warm turn's suffix would), then restore.
    rt.forward_ar(mx.array([[5, 6, 7]]), cache=cache)
    restore_cache(cache, snapshot)

    # the KV lanes (window ring, compressed KV, index keys, compressor frontier)
    # and per-entry offsets round-trip bit-exactly -> session KV restore works.
    assert _cache_offset(cache) == off_before
    assert _states_bit_equal(kv_before, _cache_state_fields(cache))

    # and, since W26, the engram history rides entry 0's state as a 6th leaf,
    # so restore_cache rewinds it together with the KV: near-prefix session
    # restore is safe again on layers 1/14 (the KV-only desync is the control in
    # tests/test_deepseek_v41_engram_state.py).
    assert int(cache.engram_state.length) == eng_before


def test_ssd_on_disk_prompt_cache_is_inoperative_for_this_cache_shape():
    # The SSD (on-disk) session cache serializes via mlx_lm.save_prompt_cache,
    # which cannot serialize this cache's nested/None-holed state tuples; it
    # fails closed (raises) rather than corrupting, so the SSD path is
    # effectively unavailable for the V4.1 backend (report: SSD save-restore).
    import os
    import tempfile

    from mlx_lm.models.cache import save_prompt_cache

    args = _swa_args()
    rt = _runtime(args, seed=9)
    cache = rt.make_cache()
    rt.forward_ar(mx.array([_prompt(10)]), cache=cache)
    path = os.path.join(tempfile.mkdtemp(), "sc.safetensors")
    with pytest.raises(Exception):
        save_prompt_cache(path, cache)
