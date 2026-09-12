"""W83: served DeepSeek-V4.1 block-prefix (SSD) restore corrupts the cache.

Served DSV4.1-Flash on the 16,384-token cell produced garbage in the two runs
that reused a 768-token prefix from the SSD session bank (block_prefix mode),
and coherent text only in the run with no reuse (full 16,384 prefill).  Root
cause: the SSD cold-tier ``block_prefix`` restore reconstructs a token-prefix by
slicing persisted tensor blocks along **axis 2** (``codec.decode_payload_prefix``
/ ``_decode_tensor_blocks``), which is the token axis only for a standard mlx_lm
KVCache ``[B, H, T, D]``.  The V4.1 per-layer state tensors are ``[B, seq, D]``
with the token axis at **axis 1**, so the axis-2 slice never trims the sequence:
``decode_payload_prefix`` returns the *full* banked-entry lanes, yet labels them
``cache_snapshot_prefix_len == requested_prefix``.  ``restore_entry_prefix_cache``
then sees ``cache_snapshot_prefix_len == required_cache_prefix_len`` and SKIPS its
corrective ``entry.trim``, so the served cache holds the whole banked entry's KV
while the engine believes it restored the short block boundary -> the suffix
prefill runs on desynced KV -> token soup.

The gate that (wrongly) authorises the axis-2 slice for V4.1 is
``codec.snapshot_supports_prefix_decode``: it excludes ``deepseek-v4`` and Gemma's
rotating cache but not V4.1, whose sequence axis is likewise not axis 2.  The fix
excludes the V4.1 layer-cache meta version too, routing V4.1 block-prefix restores
to the full ``decode_payload`` + ``entry.trim`` path (which is exact for every
lane -- window ring, compress/index groups, compressor frontier, engram).

CPU-pinned, tiny random-config model, no artifact.  These drive the real codec
and the real ``SessionBank.restore_entry_prefix_cache`` -- the exact function the
served near-prefix path calls (``generation.py`` restore loop).
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest

pytest.importorskip("mlx.core")

from mtplx.cache_state import restore_cache, snapshot_cache  # noqa: E402
from mtplx.cache_bank.codec import (  # noqa: E402
    decode_payload,
    decode_payload_prefix,
    encode_payload,
    payload_supports_prefix_decode,
    snapshot_supports_prefix_decode,
)
from mtplx.session_bank import (  # noqa: E402
    SessionBank,
    SessionBankEntry,
    token_prefix_hash,
)
from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402
from mtplx.mtp_patch import MTPContract  # noqa: E402

# Reuse the vetted tiny-model fixtures from the served-generation gate.
from test_deepseek_v41_served_generation import (  # noqa: E402
    _cache_offset,
    _cache_state_fields,
    _csa_args,
    _ngram_state,
    _randomize,
    _states_bit_equal,
    _FixedTokenizer,
)


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _runtime(args: ModelArgs, *, seed: int = 0, engram: bool = False):
    from pathlib import Path

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


def _prompt(n, vocab=48, seed=7):
    rng = np.random.default_rng(seed)
    return [int(v) for v in rng.integers(0, vocab, size=n)]


def _cold_style_entry(cache_snapshot, *, token_ids, cache_snapshot_prefix_len):
    """A SessionBankEntry shaped exactly as the SSD block-prefix lane builds one
    (``session_bank._cold_near_prefix_candidate``): clone snapshot, no live ref,
    ``cache_snapshot_prefix_len`` recording the actually-materialised span."""
    return SessionBankEntry(
        token_ids=tuple(int(t) for t in token_ids),
        token_hash=token_prefix_hash(tuple(int(t) for t in token_ids)),
        model_path=".",
        mtp_enabled=False,
        hidden_variant=None,
        cache_snapshot=cache_snapshot,
        logits=None,
        hidden=None,
        cache_snapshot_prefix_len=cache_snapshot_prefix_len,
    )


def _decode_like_cold_tier(spec, read, *, restore_point):
    """Mirror ``cold_tier._restore_row``'s partial-restore dispatch exactly: the
    axis-2 prefix slice is taken iff ``payload_supports_prefix_decode`` says the
    layout can be block-sliced (has_recurrent is False for V4.1, so the only gate
    is that predicate).  Returns (cache_snapshot, cache_snapshot_prefix_len)."""
    if payload_supports_prefix_decode(spec):
        decoded = decode_payload_prefix(
            spec, read, cache_prefix_len=max(0, restore_point - 1)
        )
        return decoded.cache_snapshot, decoded.cache_snapshot_prefix_len
    decoded = decode_payload(spec, read)
    return decoded.cache_snapshot, None


# ---------------------------------------------------------------------------
# 1. The gate contract: V4.1's per-lane token axis is NOT axis 2, so the codec
#    must refuse to block-slice it (route it to full decode + trim).  This is
#    the one-line invariant the fix restores; it FAILS before the fix.
# ---------------------------------------------------------------------------
def test_snapshot_supports_prefix_decode_refuses_v41_layout():
    args = _csa_args()
    rt = _runtime(args, seed=4, engram=True)
    cache = rt.make_cache()
    rt.forward_ar(mx.array([_prompt(40)]), cache=cache)
    snap = snapshot_cache(cache)
    enc = encode_payload(
        cache_snapshot=snap, logits=None, hidden=None,
        mtp_history_snapshot=None, block_size=256,
    )
    assert payload_supports_prefix_decode(enc.spec) is False, (
        "V4.1 state tensors are [B, seq, head_dim] (token axis = axis 1); the "
        "cold-tier prefix decoder slices axis 2, so block-slicing V4.1 does not "
        "correspond to a token trim and must be refused (full decode + trim)."
    )
    # And directly on a per-entry meta spec, mirroring the deepseek-v4 exclusion.
    assert snapshot_supports_prefix_decode(enc.spec["cache_snapshot"]) is False


# ---------------------------------------------------------------------------
# 2. The mechanism: prove decode_payload_prefix does NOT trim the V4.1 token
#    axis (independent of the fix -- documents WHY the gate must refuse it).
# ---------------------------------------------------------------------------
def test_decode_payload_prefix_does_not_trim_v41_token_axis():
    args = _csa_args()
    rt = _runtime(args, seed=4, engram=True)
    cache = rt.make_cache()
    rt.forward_ar(mx.array([_prompt(40)]), cache=cache)
    snap = snapshot_cache(cache)
    enc = encode_payload(
        cache_snapshot=snap, logits=None, hidden=None,
        mtp_history_snapshot=None, block_size=256,
    )
    read = lambda name: enc.tensors[name]  # noqa: E731
    decoded = decode_payload_prefix(enc.spec, read, cache_prefix_len=15)
    # The window lane of layer 0 was [1, 40, 16]; a real 15-token trim would make
    # it [1, 15, 16].  The axis-2 slicer leaves the token axis (axis 1) at 40.
    win0 = decoded.cache_snapshot.states[0][0]
    assert win0 is not None
    assert int(win0.shape[1]) == 40, (
        "decode_payload_prefix sliced axis 2, not the token axis (axis 1): the "
        "V4.1 window lane is returned untrimmed at its full banked length"
    )


# ---------------------------------------------------------------------------
# 3. End-to-end through the REAL SessionBank.restore_entry_prefix_cache (the
#    exact call generation.py makes on a near-prefix candidate): a block-prefix
#    (SSD) restore of a long banked entry to a shared boundary, then suffix
#    decode, must be bit-identical to the RAM reuse of the SAME banked state
#    (the ground-truth reuse path W22 verified).  The reference is the in-memory
#    banked cache trimmed to the boundary -- NOT a fresh short prefill, which
#    differs from a trim at the ~1e-7 fp level (different prefill kernel shape)
#    and is not the bug.  Before the fix the SSD block-prefix restore diverges
#    grossly (untrimmed lanes / rejected meta): the served-path garbage.
# ---------------------------------------------------------------------------
def _prefill_banked(rt, ids):
    cache = rt.make_cache()
    rt.forward_ar(mx.array([ids]), cache=cache)
    return cache


def test_block_prefix_ssd_restore_matches_ram_reuse():
    args = _csa_args()
    rt = _runtime(args, seed=4, engram=True)

    L_bank = 40          # long banked entry (a prior cell's transcript)
    matched = 16         # block boundary the new prompt shares (restore_point)
    rng = np.random.default_rng(101)
    banked_ids = [int(v) for v in rng.integers(0, args.vocab_size, size=L_bank)]
    suffix = [int(v) for v in rng.integers(0, args.vocab_size, size=24)]
    new_ids = banked_ids[:matched] + suffix  # shares the first `matched` tokens
    seed_and_suffix = new_ids[matched - 1:]  # seed re-forward slot + divergent tail

    # SSD-encode the banked entry via the real codec.
    enc = encode_payload(
        cache_snapshot=snapshot_cache(_prefill_banked(rt, banked_ids)),
        logits=None, hidden=None, mtp_history_snapshot=None, block_size=256,
    )
    read = lambda name: enc.tensors[name]  # noqa: E731

    # Ground truth: RAM reuse of the SAME banked state -- prefill, trim to the
    # boundary's seed slot (== what restore_entry_prefix_cache's trim lands on).
    ref_cache = _prefill_banked(rt, banked_ids)
    ref_cache.trim(L_bank - (matched - 1))
    assert _cache_offset(ref_cache) == matched - 1

    # SSD block-prefix restore through the real bank code path.
    cache_snap, csp_len = _decode_like_cold_tier(enc.spec, read, restore_point=matched)
    entry = _cold_style_entry(
        cache_snap, token_ids=banked_ids, cache_snapshot_prefix_len=csp_len
    )
    bank = SessionBank(max_entries=8, max_bytes=1 << 30, per_session_max_bytes=1 << 30)
    result = bank.restore_entry_prefix_cache(rt, entry, matched, mode="clone")
    assert result is not None, "bank refused the block-prefix restore"
    restored_cache = result[0]

    assert _cache_offset(restored_cache) == matched - 1, (
        f"restored offset {_cache_offset(restored_cache)} != boundary seed slot "
        f"{matched - 1}: the SSD block-prefix restore did not land on the boundary"
    )
    assert _states_bit_equal(
        _cache_state_fields(ref_cache), _cache_state_fields(restored_cache)
    ), "SSD block-prefix restored lanes differ from the RAM reuse of the boundary"

    # The headline: continuation logits match the RAM reuse, token after token.
    def _tail_logits(cache):
        return [np.array(rt.forward_ar(mx.array([[tok]]), cache=cache))
                for tok in seed_and_suffix]

    assert all(
        np.array_equal(a, b)
        for a, b in zip(_tail_logits(ref_cache), _tail_logits(restored_cache))
    ), "decode after SSD block-prefix restore diverges from RAM reuse (garbage)"


# ---------------------------------------------------------------------------
# 4. Control: the FULL-decode + entry.trim path (what the fix routes V4.1 to)
#    already matches RAM reuse bit-exactly, so the fix's target path is correct
#    and safe.  Passes before AND after the fix.
# ---------------------------------------------------------------------------
def test_full_decode_then_trim_matches_ram_reuse():
    args = _csa_args()
    rt = _runtime(args, seed=4, engram=True)
    L_bank, matched = 40, 16
    rng = np.random.default_rng(202)
    banked_ids = [int(v) for v in rng.integers(0, args.vocab_size, size=L_bank)]

    enc = encode_payload(
        cache_snapshot=snapshot_cache(_prefill_banked(rt, banked_ids)),
        logits=None, hidden=None, mtp_history_snapshot=None, block_size=256,
    )
    read = lambda name: enc.tensors[name]  # noqa: E731

    ref_cache = _prefill_banked(rt, banked_ids)
    ref_cache.trim(L_bank - (matched - 1))

    # Force the full-decode path (the fix's route): cache_snapshot_prefix_len=None.
    decoded = decode_payload(enc.spec, read)
    entry = _cold_style_entry(
        decoded.cache_snapshot, token_ids=banked_ids, cache_snapshot_prefix_len=None
    )
    bank = SessionBank(max_entries=8, max_bytes=1 << 30, per_session_max_bytes=1 << 30)
    result = bank.restore_entry_prefix_cache(rt, entry, matched, mode="clone")
    assert result is not None
    restored_cache = result[0]
    assert _cache_offset(restored_cache) == matched - 1
    assert _states_bit_equal(
        _cache_state_fields(ref_cache), _cache_state_fields(restored_cache)
    ), "full-decode + entry.trim must match RAM reuse (the fix's target path)"


# ---------------------------------------------------------------------------
# 5. Hypothesis B (sampled verify/rollback corrupts the cache) -- refuted here
#    on the same tiny model: a sampled (temperature 1) speculative tail rolled
#    back leaves EVERY lane bit-identical to the never-decoded path, and the
#    continuation logits match.  The rollback mechanics are token-driven, not
#    sample-driven, so the sampling temperature cannot corrupt the cache; and
#    the served garbage also occurs on the AR lane (window-30, no DSpark), so B
#    cannot be the cause.  (Greedy exactness is pinned in the served-gen gate;
#    this adds the sampled tail.)
# ---------------------------------------------------------------------------
def test_sampled_verify_rollback_cache_consistent():
    from mtplx.cache_state import rollback_after_verify, snapshot_untrimmable_cache

    args = _csa_args()
    rt = _runtime(args, seed=4)
    rng = np.random.default_rng(303)
    prompt_len, decoded, k, tail = 12, 5, 4, 4
    total = prompt_len + decoded
    ids = mx.array(
        rng.integers(0, args.vocab_size, size=(1, total + k + tail)).tolist()
    )

    def primed():
        c = rt.make_cache()
        rt.forward_ar(ids[:, :prompt_len], cache=c)
        for t in range(prompt_len, total):
            rt.forward_ar(ids[:, t:t + 1], cache=c)
        return c

    ref = primed()
    got = primed()

    # A sampled speculative tail (whatever tokens a temp-1 draft proposes is
    # irrelevant to rollback: the verify forwards k tokens, then rejects them).
    snapshot = snapshot_untrimmable_cache(got)
    rt.forward_ar(ids[:, total:total + k], cache=got)
    assert _cache_offset(got) == total + k
    rollback_after_verify(got, snapshot, k)
    assert _cache_offset(got) == total

    assert _states_bit_equal(_cache_state_fields(ref), _cache_state_fields(got)), (
        "verify rollback must restore every lane bit-exactly regardless of the "
        "sampler -- so the sampled DSpark verify path cannot corrupt the cache"
    )
    ref_logits = [np.array(rt.forward_ar(ids[:, t:t + 1], cache=ref))
                  for t in range(total, total + tail)]
    got_logits = [np.array(rt.forward_ar(ids[:, t:t + 1], cache=got))
                  for t in range(total, total + tail)]
    assert all(np.array_equal(a, b) for a, b in zip(ref_logits, got_logits))
