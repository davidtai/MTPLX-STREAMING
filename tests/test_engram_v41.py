"""Tests for the DeepSeek-V4.1-Flash Engram runtime (mtplx/engram_v41.py).

  * compressed token map size == 99092 on the real tokenizer;
  * streaming NgramHashState vs an independent numpy transcription of the reference
    ``NgramHashState.forward`` (torch is not installed in this venv), over 200 random
    sequences, all 24 row ids per position per layer, incl pad/DEAD at sequence start;
  * rollback: trim N then re-feed -> identical row ids;
  * row dequant parity on the REAL bank: MLX dequant == converter numpy dequant, and cos
    vs the source FP8 rows meets the converter tolerance (>= 0.99994);
  * the EngramV41 hook math vs a numpy transcription of the reference ``Engram.forward``.

CPU only.  Run under ``nice -n 19``, without ``-n auto``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

import mtplx.deepseek_v41_convert as dc
from mtplx.engram_bank import EngramBank
from mtplx.engram_v41 import (
    DEAD,
    EngramV41,
    NgramHashState,
    _StepState,
    build_compressed_token_map,
)
from mtplx.ngram_row_cache import FileRowReader, NGramRowCache, RowGeometry

mx.set_default_device(mx.cpu)

# editable-install CWD-shadowing guard
import mtplx.engram_v41 as _ev
_REPO_ROOT = Path(__file__).resolve().parents[1]
assert Path(_ev.__file__).resolve().is_relative_to(_REPO_ROOT), (_ev.__file__, _REPO_ROOT)

_DEFAULT_ART = "/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4"
ARTIFACT = Path(os.environ.get("DSV41_ARTIFACT_DIR", _DEFAULT_ART))
ENGRAM_DIR = ARTIFACT / "engram"
SRC = Path("/Users/davidtai/models/DeepSeek-V4.1-Flash-src")
MANIFEST = ENGRAM_DIR / "engram-manifest.json"

_have_manifest = MANIFEST.is_file()
_have_tokenizer = (ARTIFACT / "tokenizer.json").is_file()
_have_source = SRC.is_dir()

needs_manifest = pytest.mark.skipif(not _have_manifest, reason="engram manifest not present")
needs_tokenizer = pytest.mark.skipif(
    not (_have_manifest and _have_tokenizer), reason="artifact tokenizer not present"
)
needs_bank = pytest.mark.skipif(
    not (_have_manifest and (ENGRAM_DIR / "engram-L1.bin").is_file()),
    reason="engram bank .bin not present",
)
needs_source = pytest.mark.skipif(
    not (_have_manifest and _have_source), reason="source safetensors not present",
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(ARTIFACT))


@pytest.fixture(scope="module")
def state(tokenizer):
    return NgramHashState.from_manifest_path(ENGRAM_DIR, tokenizer)


# --------------------------------------------------------------------------
# independent numpy transcription of reference engram.py NgramHashState.forward
# (engram.py lines 160-184), used because torch is not installed in this venv.
# --------------------------------------------------------------------------
def _oracle_row_ids(st: NgramHashState, ids: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    B, S = ids.shape
    compressed = st.token_map[ids]
    if mask is not None:
        compressed = np.where(mask, compressed, DEAD)
    pos = np.broadcast_to(np.arange(S)[None, :], (B, S))
    blocked = np.zeros((B, S), dtype=bool)
    toks = []
    for shift in range(st.max_ngram_size):
        src = np.take_along_axis(compressed, np.clip(pos - shift, 0, None), axis=1)
        blocked = blocked | (pos < shift) | (src == DEAD)
        toks.append(np.where(blocked, st.pad_compressed, src))
    toks = np.stack(toks, axis=-1)
    prod = toks[:, :, None, :] * st.multipliers[None, None, :, :]
    rolling = prod[..., 0]
    hashes = []
    for i in range(1, st.max_ngram_size):
        rolling = np.bitwise_xor(rolling, prod[..., i])
        hashes.append(rolling[..., None] % st.primes[:, i - 1][None, None])
    return np.concatenate(hashes, axis=-1) + st.flat_offsets[None, None]


# --------------------------------------------------------------------------
@needs_tokenizer
def test_compressed_token_map_size(tokenizer):
    lookup, size = build_compressed_token_map(tokenizer)
    assert size == 99092
    assert len(lookup) == len(tokenizer)
    # pad id 2 maps to a valid compressed id
    assert 0 <= lookup[2] < size


@needs_tokenizer
def test_hash_parity_streaming_vs_reference(state):
    rng = np.random.default_rng(0)
    B, S = 200, 24                                   # 200 random sequences
    vocab = state.token_map.shape[0]
    ids = rng.integers(0, vocab, size=(B, S)).astype(np.int64)
    mask = rng.random((B, S)) > 0.15                 # ~15% DEAD (image-span) tokens

    ref = _oracle_row_ids(state, ids, mask)          # [B,S,n_layers,cols]

    # streaming: feed in irregular chunks so cross-chunk (decode-split) lookback is exercised
    state.reset()
    outs = []
    cut = 0
    bounds = sorted(set(int(x) for x in rng.integers(1, S, size=6))) + [S]
    for end in bounds:
        if end <= cut:
            continue
        outs.append(state.advance(ids[:, cut:end], mask[:, cut:end]))
        cut = end
    streamed = np.concatenate(outs, axis=1)
    assert streamed.shape == ref.shape
    assert np.array_equal(streamed, ref)

    # start-of-sequence pad handling: position 0 has only the current token; all longer
    # lookbacks are pad. Row ids must be < each layer's table size.
    for li, n_emb in enumerate(state.num_embeddings):
        assert int(streamed[:, :, li].max()) < n_emb
        assert int(streamed[:, :, li].min()) >= 0

    # explicit all-masked (DEAD) column stays valid (pad-filled), matching the oracle
    state.reset()
    hard = state.advance(ids[:1, :5], np.zeros((1, 5), dtype=bool))
    ref_hard = _oracle_row_ids(state, ids[:1, :5], np.zeros((1, 5), dtype=bool))
    assert np.array_equal(hard, ref_hard)


@needs_tokenizer
def test_rollback_trim_refeed(state):
    rng = np.random.default_rng(1)
    vocab = state.token_map.shape[0]
    ids = rng.integers(0, vocab, size=(2, 30)).astype(np.int64)
    mask = rng.random((2, 30)) > 0.1

    state.reset()
    state.advance(ids[:, :20], mask[:, :20])
    tail_ref = state.advance(ids[:, 20:30], mask[:, 20:30])     # positions 20..29
    assert state.length == 30

    # reject the last 10 positions and re-feed the same tokens
    state.trim(10)
    assert state.length == 20
    tail_again = state.advance(ids[:, 20:30], mask[:, 20:30])
    assert np.array_equal(tail_again, tail_ref)

    # a different continuation after trimming yields different ids but stays valid
    state.trim(10)
    other = state.advance(rng.integers(0, vocab, size=(2, 10)).astype(np.int64),
                          np.ones((2, 10), dtype=bool))
    assert other.shape == tail_ref.shape


# --------------------------------------------------------------------------
def _source_rows(shard, wname, sname, rows):
    path = os.path.join(str(SRC), shard)
    header, ds = dc.read_safetensors_header(path)
    ents = dc.tensor_entries(header)
    we, se = ents[wname], ents[sname]
    fd = os.open(path, os.O_RDONLY)
    HD, G = 256, 8
    try:
        wu = np.empty((len(rows), HD), np.uint8)
        su = np.empty((len(rows), G), np.uint8)
        for i, r in enumerate(rows):
            wu[i] = np.frombuffer(dc._pread_exact(fd, ds + we.begin + r * HD, HD), np.uint8)
            su[i] = np.frombuffer(dc._pread_exact(fd, ds + se.begin + r * G, G), np.uint8)
        return dc.dequant_engram_embed(wu, su)
    finally:
        os.close(fd)


@needs_bank
@needs_source
def test_row_dequant_parity_real_bank():
    manifest = json.loads(MANIFEST.read_text())
    for le in manifest["layers"]:
        L, nrows, s = le["layer_id"], le["rows"], le["source"]
        rng = np.random.default_rng(2000 + L)
        ids = sorted(int(x) for x in rng.integers(0, nrows, size=64))
        bank = EngramBank.open(ENGRAM_DIR, L, cache_rows=32)
        try:
            # (1) MLX dequant of gathered rows == converter numpy dequant (bf16 rounding only)
            dq_mlx = np.array(bank.cache.dequantize(np.array(ids)).astype(mx.float32))
            dq_np = bank.dequantize_rows(ids)
            assert dq_mlx.shape == (64, 256)
            assert np.max(np.abs(dq_mlx - dq_np)) < 0.5

            # (2) cos vs source FP8 rows meets the converter tolerance (>= 0.99994)
            src = _source_rows(s["shard_file"], s["weight_tensor"], s["scale_tensor"], ids)
            cos = np.array([
                float((dq_mlx[i] @ src[i]) / (np.linalg.norm(dq_mlx[i]) * np.linalg.norm(src[i])))
                for i in range(len(ids))
            ])
            assert cos.mean() >= 0.99994, (L, cos.mean())
            assert np.median(cos) >= 0.99994, (L, np.median(cos))
            assert cos.min() >= 0.9995, (L, cos.min())   # floor: catch gross corruption
        finally:
            bank.close()


# --------------------------------------------------------------------------
# EngramV41 hook math vs a numpy transcription of reference Engram.forward (model.py 350-364)
# --------------------------------------------------------------------------
def _engram_forward_numpy(x, embed, wkv_w, q_w, k_w, hc_mult, dim, eps, clamp, mask=None):
    B, L = x.shape[0], x.shape[1]
    kv = embed.reshape(B, L, -1) @ wkv_w.T
    key = kv[..., : hc_mult * dim].reshape(B, L, hc_mult, dim)
    value = kv[..., hc_mult * dim:]
    weight = q_w * k_w
    h = x
    rstd = (1.0 / np.sqrt(np.mean(h * h, -1) + eps)) * (1.0 / np.sqrt(np.mean(key * key, -1) + eps))
    dot = np.sum(h * weight * key, -1) * rstd * dim ** -0.5
    gate = 1.0 / (1.0 + np.exp(-np.copysign(np.sqrt(np.maximum(np.abs(dot), clamp)), dot)))
    if mask is not None:
        gate = np.where(mask[..., None], gate, 0.0)
    return h + gate[..., None] * value[:, :, None, :]


def test_engram_module_math(tmp_path):
    rng = np.random.default_rng(9)
    n_emb, cols, head_dim, dim, hc_mult = 800, 24, 256, 16, 4
    eps, clamp = 1e-6, 1e-6

    # synthetic affine bank
    f32 = rng.standard_normal((n_emb, head_dim)).astype(np.float32)
    rec = dc.engram_chunk_records(f32)
    p = tmp_path / "bank.bin"
    p.write_bytes(np.ascontiguousarray(rec).tobytes())
    cache = NGramRowCache(FileRowReader(p, row_bytes=272, num_rows=n_emb),
                          RowGeometry(head_dim, 8, 64), num_rows=n_emb, cache_rows=64)

    B, L = 2, 3
    row_ids = rng.integers(0, n_emb, size=(B, L, 1, cols)).astype(np.int64)  # 1 engram layer here
    x = rng.standard_normal((B, L, hc_mult, dim)).astype(np.float32)
    wkv_w = (rng.standard_normal((dim * (hc_mult + 1), cols * head_dim)) * 0.02).astype(np.float32)
    q_w = rng.standard_normal((hc_mult, dim)).astype(np.float32)
    k_w = rng.standard_normal((hc_mult, dim)).astype(np.float32)
    token_mask = np.array([[True, True, False], [True, False, True]])

    module = EngramV41(
        layer_id=1, layer_hash_index=0, row_cache=cache,
        wkv=EngramV41.dense_wkv(mx.array(wkv_w)),
        q_weight=mx.array(q_w), k_weight=mx.array(k_w),
        dim=dim, hc_mult=hc_mult, norm_eps=eps, clamp_value=clamp,
    )
    cstate = _StepState(row_ids=row_ids, token_mask=mx.array(token_mask))
    token_ids = np.zeros((B, L), np.int64)
    out = np.array(module(mx.array(x), token_ids, cstate).astype(mx.float32))

    # numpy reference uses the identical (bf16-rounded) embed the module dequantizes
    embed_np = np.array(cache.dequantize(row_ids[:, :, 0, :]).astype(mx.float32))
    ref = _engram_forward_numpy(x, embed_np, wkv_w, q_w, k_w, hc_mult, dim, eps, clamp, token_mask)

    assert out.shape == (B, L, hc_mult, dim)
    assert np.allclose(out, ref, atol=2e-2, rtol=1e-3)
    # masked positions pass through untouched (gate == 0 there)
    assert np.allclose(out[0, 2], x[0, 2], atol=1e-5)
    assert np.allclose(out[1, 1], x[1, 1], atol=1e-5)
    # the additive contribution (result - hidden) matches the reference delta
    assert np.allclose(out - x, ref - x, atol=2e-2, rtol=1e-3)
