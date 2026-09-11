"""Tests for the DeepSeek-V4.1-Flash Engram affine-8 disk bank.

Covers: fp8 dequant vs an independent numpy transcription of model.py's math; the
affine record round-trip (bytes <-> mx.quantize); a full synthetic conversion with
resume/idempotence; and reader gather + dequant correctness against the source.

Run under ``nice -n 19``, without ``-n auto``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

import mtplx.deepseek_v41_convert as dc
from mtplx.engram_bank import EngramBank

mx.set_default_device(mx.cpu)

_CONV_PATH = Path(__file__).resolve().parents[1] / "scripts" / "convert_deepseek_v41_engram.py"
_spec = importlib.util.spec_from_file_location("convert_engram", _CONV_PATH)
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)

HEAD_DIM = dc.ENGRAM_HEAD_DIM
GROUPS32 = HEAD_DIM // dc.ENGRAM_FP8_BLOCK  # 8


# --------------------------------------------------------------------------
# synthetic source shard
# --------------------------------------------------------------------------
def _rand_e4m3_bytes(rng, n) -> np.ndarray:
    """Random e4m3fn bytes with the two NaN codes (0x7F, 0xFF) scrubbed to 0."""
    b = rng.integers(0, 256, size=n, dtype=np.uint8)
    b[(b == 0x7F) | (b == 0xFF)] = 0
    return b


def _rand_e8m0_bytes(rng, n) -> np.ndarray:
    """Random e8m0 scale bytes in [124,131] (2**(-3..4)); never 0xFF (NaN)."""
    return rng.integers(124, 132, size=n, dtype=np.uint8)


def _write_safetensors(path: Path, tensors: dict) -> None:
    header, data, offset = {}, bytearray(), 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + len(raw)]}
        data += raw
        offset += len(raw)
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(bytes(data))


def _make_synthetic_shard(tmp_path: Path, layer: int, rows: int, seed: int = 0):
    """Write a synthetic shard with layer L engram embed weight/scale + an index."""
    rng = np.random.default_rng(seed)
    wu8 = _rand_e4m3_bytes(rng, rows * HEAD_DIM).reshape(rows, HEAD_DIM)
    su8 = _rand_e8m0_bytes(rng, rows * GROUPS32).reshape(rows, GROUPS32)
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    shard_file = f"model-000{47 if layer == 1 else 48}-of-00048.safetensors"
    _write_safetensors(src / shard_file, {
        f"layers.{layer}.engram.embed.weight": ("F8_E4M3", (rows, HEAD_DIM), wu8.tobytes()),
        f"layers.{layer}.engram.embed.scale": ("F8_E8M0", (rows, GROUPS32), su8.tobytes()),
    })
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"weight_map": {
        f"layers.{layer}.engram.embed.weight": shard_file,
        f"layers.{layer}.engram.embed.scale": shard_file,
    }}))
    return src, index, wu8, su8


# --------------------------------------------------------------------------
# constants / layout
# --------------------------------------------------------------------------
def test_record_constants_and_prime_layout():
    assert dc.ENGRAM_RECORD_BYTES == 272
    lay = dc.engram_record_layout()
    assert lay["weight"]["length"] == 256 and lay["weight"]["offset"] == 0
    assert lay["scales"]["offset"] == 256 and lay["scales"]["length"] == 8
    assert lay["biases"]["offset"] == 264 and lay["biases"]["length"] == 8
    assert dc.engram_n_hash_cols() == 24

    primes = dc.engram_prime_layout()
    for li in range(len(dc.ENGRAM_LAYER_IDS)):
        _, flat, total = dc.engram_flat_offsets(primes[li])
        assert len(flat) == 24
        assert total == dc.ENGRAM_NUM_EMBEDDINGS[li]

    m1 = dc.compute_engram_hash_multipliers()
    m2 = dc.compute_engram_hash_multipliers()
    assert m1.shape == (2, dc.ENGRAM_MAX_NGRAM_SIZE)
    assert np.array_equal(m1, m2)            # deterministic
    assert bool((m1 % 2 == 1).all())         # odd


# --------------------------------------------------------------------------
# fp8 dequant vs independent numpy transcription of model.py
# --------------------------------------------------------------------------
def _independent_e4m3_decode(u8: np.ndarray) -> np.ndarray:
    out = np.empty(u8.shape, dtype=np.float32)
    flat = u8.ravel()
    dec = np.empty(flat.shape, dtype=np.float32)
    for i, b in enumerate(flat):
        b = int(b)
        sign = -1.0 if (b >> 7) & 1 else 1.0
        exp = (b >> 3) & 0xF
        man = b & 0x7
        if exp == 0:
            dec[i] = sign * (man / 8.0) * (2.0 ** (1 - 7))
        elif exp == 0xF and man == 0x7:
            dec[i] = np.nan
        else:
            dec[i] = sign * (1.0 + man / 8.0) * (2.0 ** (exp - 7))
    return dec.reshape(u8.shape)


def test_dequant_matches_numpy_transcription():
    rng = np.random.default_rng(7)
    rows = 40
    wu8 = _rand_e4m3_bytes(rng, rows * HEAD_DIM).reshape(rows, HEAD_DIM)
    su8 = _rand_e8m0_bytes(rng, rows * GROUPS32).reshape(rows, GROUPS32)

    # independent transcription of ParallelEngramEmbedding.forward
    v = _independent_e4m3_decode(wu8)
    s = (2.0 ** (su8.astype(np.float64) - 127)).astype(np.float32)
    ref = (v.reshape(rows, GROUPS32, dc.ENGRAM_FP8_BLOCK) * s[:, :, None]).reshape(rows, HEAD_DIM)

    got = dc.dequant_engram_embed(wu8, su8)
    assert np.array_equal(got, ref)

    # bad scale shape rejected
    with pytest.raises(ValueError):
        dc.dequant_engram_embed(wu8, su8[:, :4])


# --------------------------------------------------------------------------
# affine record round-trip (bytes <-> mx.quantize)
# --------------------------------------------------------------------------
def test_chunk_record_roundtrip_exact_and_cosine():
    rng = np.random.default_rng(3)
    rows = 50
    v = rng.standard_normal((rows, HEAD_DIM)).astype(np.float32)

    rec = dc.engram_chunk_records(v)
    assert rec.shape == (rows, 272)

    # parse the record's three fields back out
    w = np.ascontiguousarray(rec[:, 0:256]).view("<u4")          # [rows,64]
    s = np.ascontiguousarray(rec[:, 256:264]).view("<u2")        # [rows,4]
    b = np.ascontiguousarray(rec[:, 264:272]).view("<u2")        # [rows,4]

    # exact byte round-trip vs a direct quantize
    packed, scales, biases = dc.quantize_engram_rows(v)
    assert np.array_equal(w, np.array(packed).astype("<u4"))
    assert np.array_equal(s, np.array(scales.view(mx.uint16)).astype("<u2"))
    assert np.array_equal(b, np.array(biases.view(mx.uint16)).astype("<u2"))

    # dequant matches mx.dequantize up to bf16 rounding; cosine vs source high
    dq_mlx = np.array(mx.dequantize(packed, scales, biases, group_size=64, bits=8,
                                    mode="affine").astype(mx.float32))
    dq_np = dc.dequant_affine_record(rec[:, 0:256].astype(np.uint8), s, b)
    assert np.max(np.abs(dq_np - dq_mlx)) < 0.5  # bf16 rounding only
    cos = float((dq_np.ravel() @ v.ravel()) /
                (np.linalg.norm(dq_np) * np.linalg.norm(v)))
    assert cos >= 0.999


# --------------------------------------------------------------------------
# full synthetic conversion + reader
# --------------------------------------------------------------------------
def test_full_convert_and_reader(tmp_path):
    layer, rows = 1, 300
    src, index, wu8, su8 = _make_synthetic_shard(tmp_path, layer, rows, seed=11)
    out = tmp_path / "engram"
    state = tmp_path / "state"
    weight_map = json.loads(index.read_text())["weight_map"]

    entry = conv.convert_layer(layer, src, out, weight_map, state_dir=state,
                               chunk_rows=64, max_rows=0, wait=False, poll=0.0,
                               expected_rows=rows)
    conv.write_manifest(out, [entry])

    binp = out / f"engram-L{layer}.bin"
    assert binp.stat().st_size == rows * 272
    # cleanliness: engram/ holds exactly the shippable files (no partial/.tmp)
    assert sorted(p.name for p in out.iterdir()) == ["engram-L1.bin", "engram-manifest.json"]
    # in-progress bin was moved out of state_dir; only the journal remains
    assert not (state / "engram-L1.bin").exists()
    assert (state / "engram-L1.journal.json").exists()

    ref = dc.dequant_engram_embed(wu8, su8)  # source f32 truth

    bank = EngramBank.open(out, layer)
    assert len(bank) == rows and bank.record_bytes == 272

    idxs = [0, 1, 5, 63, 64, 65, 299, 128, 200]
    w, s, b = bank.gather(idxs)
    assert w.shape == (len(idxs), 64) and s.shape == (len(idxs), 4) and b.shape == (len(idxs), 4)

    dq = bank.dequantize_rows(idxs)
    for i, r in enumerate(idxs):
        a, c = dq[i], ref[r]
        cos = float((a @ c) / (np.linalg.norm(a) * np.linalg.norm(c)))
        assert cos >= 0.999, (r, cos)

    # gather returns exactly what a re-quantize of those source rows produces
    packed, scales, biases = dc.quantize_engram_rows(ref[idxs])
    assert np.array_equal(w, np.array(packed).astype("<u4"))
    assert np.array_equal(s, np.array(scales.view(mx.uint16)).astype("<u2"))

    # out-of-range guard
    with pytest.raises(IndexError):
        bank.gather([rows])
    bank.close()


def test_reader_lru_budget(tmp_path):
    layer, rows = 1, 200
    src, index, _, _ = _make_synthetic_shard(tmp_path, layer, rows, seed=5)
    out = tmp_path / "engram"
    weight_map = json.loads(index.read_text())["weight_map"]
    entry = conv.convert_layer(layer, src, out, weight_map, state_dir=tmp_path / "st",
                               chunk_rows=50, max_rows=0, wait=False, poll=0.0,
                               expected_rows=rows)
    conv.write_manifest(out, [entry])

    bank = EngramBank.open(out, layer, cache_rows=4)  # budget = 4 records
    bank.gather(list(range(20)))
    assert bank.cache_used_bytes <= 4 * bank.record_bytes
    # repeated gather is served from cache and still correct
    w1, _, _ = bank.gather([10, 11, 12])
    w2, _, _ = bank.gather([10, 11, 12])
    assert np.array_equal(w1, w2)
    bank.close()


# --------------------------------------------------------------------------
# resume + idempotence + append-only
# --------------------------------------------------------------------------
def test_resume_and_idempotence(tmp_path, monkeypatch):
    layer, rows = 1, 320
    src, index, wu8, su8 = _make_synthetic_shard(tmp_path, layer, rows, seed=21)
    weight_map = json.loads(index.read_text())["weight_map"]

    # reference: a clean full run in a separate tree
    ref_out = tmp_path / "ref_engram"
    conv.convert_layer(layer, src, ref_out, weight_map, state_dir=tmp_path / "ref_state",
                       chunk_rows=64, max_rows=0, wait=False, poll=0.0, expected_rows=rows)
    ref_bytes = (ref_out / "engram-L1.bin").read_bytes()
    assert len(ref_bytes) == rows * 272

    out = tmp_path / "engram"
    state = tmp_path / "state"

    # fault-inject: fail on the 3rd chunk (chunks 0,1 complete, 2 raises)
    real = dc.engram_chunk_records
    calls = {"n": 0}

    def flaky(values, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("injected fault")
        return real(values, **kw)

    monkeypatch.setattr(dc, "engram_chunk_records", flaky)
    with pytest.raises(RuntimeError, match="injected fault"):
        conv.convert_layer(layer, src, out, weight_map, state_dir=state,
                           chunk_rows=64, max_rows=0, wait=False, poll=0.0,
                           fsync_every=1, expected_rows=rows)

    # partial bin + journal live in state_dir, NOT the artifact
    assert not (out / "engram-L1.bin").exists()
    assert (state / "engram-L1.bin").exists()
    j = json.loads((state / "engram-L1.journal.json").read_text())
    assert j["done_chunks"] == [0, 1]  # 2 chunks committed before the fault
    partial = (state / "engram-L1.bin").read_bytes()
    # completed region already matches the reference (byte-for-byte)
    assert partial[: 2 * 64 * 272] == ref_bytes[: 2 * 64 * 272]

    # resume with the real quantizer: only the remaining chunks are recomputed
    monkeypatch.setattr(dc, "engram_chunk_records", real)
    resume_calls = {"n": 0}

    def counting(values, **kw):
        resume_calls["n"] += 1
        return real(values, **kw)

    monkeypatch.setattr(dc, "engram_chunk_records", counting)
    conv.convert_layer(layer, src, out, weight_map, state_dir=state,
                       chunk_rows=64, max_rows=0, wait=False, poll=0.0,
                       fsync_every=1, expected_rows=rows)
    n_chunks = (rows + 63) // 64
    assert resume_calls["n"] == n_chunks - 2  # chunks 0,1 skipped, not rewritten

    got = (out / "engram-L1.bin").read_bytes()
    assert got == ref_bytes  # byte-identical to a clean full run

    # idempotent: a third run early-skips (final present, correct size)
    monkeypatch.setattr(dc, "engram_chunk_records", real)
    before = (out / "engram-L1.bin").stat().st_mtime_ns
    conv.convert_layer(layer, src, out, weight_map, state_dir=state,
                       chunk_rows=64, max_rows=0, wait=False, poll=0.0, expected_rows=rows)
    assert (out / "engram-L1.bin").read_bytes() == ref_bytes
    assert (out / "engram-L1.bin").stat().st_mtime_ns == before  # untouched


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------
def test_manifest_fields_and_hashing(tmp_path):
    layer, rows = 1, 128
    src, index, _, _ = _make_synthetic_shard(tmp_path, layer, rows, seed=1)
    out = tmp_path / "engram"
    weight_map = json.loads(index.read_text())["weight_map"]
    entry = conv.convert_layer(layer, src, out, weight_map, state_dir=tmp_path / "st",
                               chunk_rows=64, max_rows=0, wait=False, poll=0.0,
                               expected_rows=rows)
    conv.write_manifest(out, [entry])

    m = json.loads((out / "engram-manifest.json").read_text())
    assert m["format"] == "mtplx-engram-manifest-v1"
    assert m["model_key"] == dc.MODEL_KEY
    assert m["source_revision"] == dc.SOURCE_REVISION
    assert m["quant"] == {"bits": 8, "group_size": 64, "mode": "affine",
                          "head_dim": 256, "record_bytes": 272}
    le = m["layers"][0]
    for k in ("layer_id", "file", "rows", "record_bytes", "total_bytes",
              "quant", "record_layout", "source"):
        assert k in le
    assert le["total_bytes"] == rows * 272

    h = m["hashing"]
    assert h["layer_ids"] == [1, 14]
    assert h["n_hash_cols"] == 24
    assert len(h["hash_multipliers"]) == 2 and len(h["hash_multipliers"][0]) == 4
    for li, pl in enumerate(h["per_layer"]):
        assert pl["total_rows"] == dc.ENGRAM_NUM_EMBEDDINGS[li]
        flat = [p for pn in pl["primes"] for p in pn]
        assert sum(flat) == dc.ENGRAM_NUM_EMBEDDINGS[li]
        assert pl["flat_offsets"][0] == 0


# ==========================================================================
# mxfp8 row codec (`--row-codec mxfp8`): EXACT byte repack of the source
# ==========================================================================
def test_mxfp8_record_layout_constants():
    assert conv.MXFP8_RECORD_BYTES == 264
    lay = conv.mxfp8_record_layout()
    assert lay["weight"]["offset"] == 0 and lay["weight"]["length"] == 256
    assert lay["weight"]["dtype"] == "U32" and lay["weight"]["shape"] == [64]
    assert lay["scales"]["offset"] == 256 and lay["scales"]["length"] == 8
    assert lay["scales"]["dtype"] == "U8" and lay["scales"]["shape"] == [8]
    assert "biases" not in lay
    assert lay["record_bytes"] == 264


def _convert_mxfp8_one_layer(tmp_path, layer, rows, seed):
    """Run the mxfp8 pipeline exactly like main(): stage (finalize=False) -> flip -> manifest."""
    src, index, wu8, su8 = _make_synthetic_shard(tmp_path, layer, rows, seed=seed)
    out = tmp_path / "engram"
    state = tmp_path / "state"
    weight_map = json.loads(index.read_text())["weight_map"]
    entry = conv.convert_layer(layer, src, out, weight_map, state_dir=state,
                               chunk_rows=64, max_rows=0, wait=False, poll=0.0,
                               expected_rows=rows, row_codec="mxfp8", finalize=False)
    staged = out / f"engram-L{layer}.bin.new"
    assert staged.is_file()                       # staging lives in the artifact, unfinalized
    conv.finalize_mxfp8(out, [layer])
    conv.write_manifest(out, [entry], row_codec="mxfp8")
    return out, state, wu8, su8, entry


def test_mxfp8_full_convert_and_reader(tmp_path):
    layer, rows = 1, 300
    out, state, wu8, su8, _ = _convert_mxfp8_one_layer(tmp_path, layer, rows, seed=13)

    binp = out / f"engram-L{layer}.bin"
    assert binp.stat().st_size == rows * 264
    # clean artifact: only shippable files, no leftover .bin.new
    assert sorted(p.name for p in out.iterdir()) == ["engram-L1.bin", "engram-manifest.json"]
    assert not (out / "engram-L1.bin.new").exists()
    # journal lives outside the artifact
    assert (state / "engram-L1.journal.json").exists()

    # the bank IS the source bytes, verbatim (exact repack: codes | e8m0 scales)
    raw = np.frombuffer(binp.read_bytes(), np.uint8).reshape(rows, 264)
    assert np.array_equal(raw[:, :256], wu8)
    assert np.array_equal(raw[:, 256:264], su8)

    ref = dc.dequant_engram_embed(wu8, su8)       # model's FP8 dequant (f32 truth)
    bank = EngramBank.open(out, layer, cache_rows=32)
    assert bank.mode == "mxfp8" and bank.record_bytes == 264 and len(bank) == rows

    idxs = [0, 1, 5, 63, 64, 65, 299, 128, 200]
    w, s, b = bank.gather(idxs)
    assert w.shape == (len(idxs), 64) and s.shape == (len(idxs), 8) and b is None
    assert np.array_equal(w.view(np.uint8).reshape(len(idxs), -1)[:, :256], wu8[idxs])
    assert np.array_equal(s, su8[idxs])

    # BIT-EXACT: numpy reference dequant AND the real MLX cache dequant path == source fp32
    assert np.array_equal(bank.dequantize_rows(idxs), ref[idxs])
    dq_mlx = np.array(bank.cache.dequantize(np.array(idxs)).astype(mx.float32))
    assert np.array_equal(dq_mlx, ref[idxs])

    with pytest.raises(IndexError):
        bank.gather([rows])
    bank.close()


def test_mxfp8_manifest_fields(tmp_path):
    layer, rows = 1, 128
    out, _, _, _, _ = _convert_mxfp8_one_layer(tmp_path, layer, rows, seed=2)
    m = json.loads((out / "engram-manifest.json").read_text())
    assert m["format"] == "mtplx-engram-manifest-v1"
    assert m["quant"] == {"bits": 8, "group_size": 32, "mode": "mxfp8",
                          "head_dim": 256, "record_bytes": 264}
    assert "note" in m["dequant"] and "mxfp8" in m["dequant"]["note"]
    le = m["layers"][0]
    assert le["record_bytes"] == 264 and le["total_bytes"] == rows * 264
    assert le["quant"]["mode"] == "mxfp8" and le["quant"]["group_size"] == 32
    assert le["record_layout"] == conv.mxfp8_record_layout()
    assert le["source"]["exact_repack"] is True
    # sha256 present and matches the bank file bytes
    assert len(le["sha256"]) == 64
    import hashlib
    assert le["sha256"] == hashlib.sha256((out / "engram-L1.bin").read_bytes()).hexdigest()
    # hashing block still emitted (codec-independent)
    assert m["hashing"]["n_hash_cols"] == 24


def test_mxfp8_flip_preserves_residents(tmp_path):
    """Rewriting the manifest for the mxfp8 flip must NOT drop a pre-existing residents entry."""
    layer, rows = 1, 128
    src, index, _, _ = _make_synthetic_shard(tmp_path, layer, rows, seed=7)
    out = tmp_path / "engram"
    weight_map = json.loads(index.read_text())["weight_map"]

    # pre-existing affine manifest carrying a W4 residents entry (as the real artifact has)
    e_aff = conv.convert_layer(layer, src, out, weight_map, state_dir=tmp_path / "sa",
                               chunk_rows=64, max_rows=0, wait=False, poll=0.0,
                               expected_rows=rows, row_codec="affine")
    fake_res = {"file": "engram-residents.safetensors", "total_bytes": 123,
                "sha256": "d" * 64, "tensors": [{"name": "x", "dtype": "U32", "shape": [1]}],
                "layers": [{"layer_id": 1}]}
    conv.write_manifest(out, [e_aff], row_codec="affine", residents=fake_res)
    assert json.loads((out / "engram-manifest.json").read_text())["residents"] == fake_res

    # mxfp8 flip: capture residents pre-flip (as main() does), convert, finalize, rewrite manifest
    residents = conv.read_existing_residents(out)
    assert residents == fake_res
    e_mx = conv.convert_layer(layer, src, out, weight_map, state_dir=tmp_path / "sm",
                              chunk_rows=64, max_rows=0, wait=False, poll=0.0,
                              expected_rows=rows, row_codec="mxfp8", finalize=False)
    conv.finalize_mxfp8(out, [layer])
    conv.write_manifest(out, [e_mx], row_codec="mxfp8", residents=residents)

    m = json.loads((out / "engram-manifest.json").read_text())
    assert m["quant"]["mode"] == "mxfp8"
    assert m["residents"] == fake_res                 # preserved verbatim
    keys = list(m.keys())
    assert keys[keys.index("layers") + 1] == "residents"   # canonical position
    assert keys[keys.index("residents") + 1] == "hashing"
    # manifest_sha256 recomputed over the residents-carrying manifest
    import hashlib
    payload = {k: v for k, v in m.items() if k != "manifest_sha256"}
    assert m["manifest_sha256"] == hashlib.sha256(
        json.dumps(payload, indent=2).encode()).hexdigest()


def test_mxfp8_resume_idempotence_and_staging(tmp_path, monkeypatch):
    layer, rows = 1, 320
    src, index, wu8, su8 = _make_synthetic_shard(tmp_path, layer, rows, seed=23)
    weight_map = json.loads(index.read_text())["weight_map"]
    out = tmp_path / "engram"
    state = tmp_path / "state"

    # fault-inject on the 3rd chunk (chunks 0,1 commit, 2 raises)
    real = conv._mxfp8_chunk_records
    calls = {"n": 0}

    def flaky(w, s):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("injected fault")
        return real(w, s)

    monkeypatch.setattr(conv, "_mxfp8_chunk_records", flaky)
    with pytest.raises(RuntimeError, match="injected fault"):
        conv.convert_layer(layer, src, out, weight_map, state_dir=state, chunk_rows=64,
                           max_rows=0, wait=False, poll=0.0, fsync_every=1,
                           expected_rows=rows, row_codec="mxfp8", finalize=False)
    # partial staging .bin.new in the artifact; NO finalized bank; journal outside, [0,1]
    assert (out / "engram-L1.bin.new").exists()
    assert not (out / "engram-L1.bin").exists()
    j = json.loads((state / "engram-L1.journal.json").read_text())
    assert j["done_chunks"] == [0, 1] and j["row_codec"] == "mxfp8"
    # committed region already byte-exact
    partial = (out / "engram-L1.bin.new").read_bytes()
    ref_full = conv._mxfp8_chunk_records(wu8, su8).tobytes()
    assert partial[: 2 * 64 * 264] == ref_full[: 2 * 64 * 264]

    # resume with the real repacker: only the remaining chunks recompute
    monkeypatch.setattr(conv, "_mxfp8_chunk_records", real)
    resume = {"n": 0}

    def counting(w, s):
        resume["n"] += 1
        return real(w, s)

    monkeypatch.setattr(conv, "_mxfp8_chunk_records", counting)
    entry = conv.convert_layer(layer, src, out, weight_map, state_dir=state, chunk_rows=64,
                               max_rows=0, wait=False, poll=0.0, fsync_every=1,
                               expected_rows=rows, row_codec="mxfp8", finalize=False)
    n_chunks = (rows + 63) // 64
    assert resume["n"] == n_chunks - 2            # chunks 0,1 not rewritten
    conv.finalize_mxfp8(out, [layer])
    assert (out / "engram-L1.bin").read_bytes() == ref_full  # byte-identical to a clean run

    # idempotent: a finalize=True re-run early-skips (final present, correct size)
    before = (out / "engram-L1.bin").stat().st_mtime_ns
    conv.convert_layer(layer, src, out, weight_map, state_dir=state, chunk_rows=64,
                       max_rows=0, wait=False, poll=0.0, expected_rows=rows,
                       row_codec="mxfp8", finalize=True)
    assert (out / "engram-L1.bin").stat().st_mtime_ns == before


def test_both_manifests_load_side_by_side(tmp_path):
    """Affine and mxfp8 banks/manifests both open through EngramBank (codec from the manifest)."""
    rows = 200
    src, index, wu8, su8 = _make_synthetic_shard(tmp_path, 1, rows, seed=31)
    weight_map = json.loads(index.read_text())["weight_map"]
    ref = dc.dequant_engram_embed(wu8, su8)
    idxs = [0, 1, 64, 199, 100]

    aff = tmp_path / "affine"
    e_aff = conv.convert_layer(1, src, aff, weight_map, state_dir=tmp_path / "sa",
                               chunk_rows=64, max_rows=0, wait=False, poll=0.0,
                               expected_rows=rows, row_codec="affine")
    conv.write_manifest(aff, [e_aff], row_codec="affine")

    mx8 = tmp_path / "mxfp8"
    e_mx = conv.convert_layer(1, src, mx8, weight_map, state_dir=tmp_path / "sm",
                              chunk_rows=64, max_rows=0, wait=False, poll=0.0,
                              expected_rows=rows, row_codec="mxfp8", finalize=False)
    conv.finalize_mxfp8(mx8, [1])
    conv.write_manifest(mx8, [e_mx], row_codec="mxfp8")

    ba = EngramBank.open(aff, 1)
    bm = EngramBank.open(mx8, 1)
    try:
        assert ba.mode == "affine" and ba.record_bytes == 272
        assert bm.mode == "mxfp8" and bm.record_bytes == 264
        # mxfp8 is exact; affine is close (bf16 requantize) -- both usable, same source
        assert np.array_equal(bm.dequantize_rows(idxs), ref[idxs])
        aff_dq = ba.dequantize_rows(idxs)
        cos = np.array([float((aff_dq[i] @ ref[idxs][i]) /
                              (np.linalg.norm(aff_dq[i]) * np.linalg.norm(ref[idxs][i])))
                        for i in range(len(idxs))])
        assert cos.min() >= 0.999
    finally:
        ba.close()
        bm.close()


# --------------------------------------------------------------------------
# resident Engram projections sidecar (`--mode residents`) -- synthetic
# --------------------------------------------------------------------------
def _bf16_bytes(rng, shape) -> tuple[np.ndarray, bytes]:
    """Random BF16 bit patterns + the f32 they losslessly widen to."""
    f = (rng.standard_normal(shape) * 0.05).astype(np.float32)
    u16 = (f.view(np.uint32) >> 16).astype(np.uint16)          # truncate to bf16
    f_bf = (u16.astype(np.uint32) << 16).view(np.float32)      # exact widen back
    return f_bf, u16.tobytes()


def _make_resident_shard(tmp_path, layer, out_w, in_w, hc_mult, dim, seed=3):
    """Synthetic source shard with layer L's engram wkv (fp8) + q/k (bf16) residents."""
    rng = np.random.default_rng(seed)
    wu8 = _rand_e4m3_bytes(rng, out_w * in_w).reshape(out_w, in_w)
    su8 = _rand_e8m0_bytes(rng, (out_w // 32) * (in_w // 32)).reshape(out_w // 32, in_w // 32)
    q_bf, q_raw = _bf16_bytes(rng, (hc_mult, dim))
    k_bf, k_raw = _bf16_bytes(rng, (hc_mult, dim))
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    shard_file = f"model-000{47 if layer == 1 else 48}-of-00048.safetensors"
    _write_safetensors(src / shard_file, {
        f"layers.{layer}.engram.wkv.weight": ("F8_E4M3", (out_w, in_w), wu8.tobytes()),
        f"layers.{layer}.engram.wkv.scale": ("F8_E8M0", su8.shape, su8.tobytes()),
        f"layers.{layer}.engram.q_weight": ("BF16", (hc_mult, dim), q_raw),
        f"layers.{layer}.engram.k_weight": ("BF16", (hc_mult, dim), k_raw),
    })
    weight_map = {
        f"layers.{layer}.engram.wkv.weight": shard_file,
        f"layers.{layer}.engram.wkv.scale": shard_file,
        f"layers.{layer}.engram.q_weight": shard_file,
        f"layers.{layer}.engram.k_weight": shard_file,
    }
    return src, weight_map, wu8, su8, q_bf, k_bf


def test_convert_residents_sidecar_and_manifest(tmp_path):
    from mtplx.engram_v41 import load_engram_residents
    from mtplx.ngram_row_cache import FileRowReader, NGramRowCache, RowGeometry

    layer, hc_mult, dim = 1, 3, 16
    out_w, in_w = dim * (hc_mult + 1), 128          # 64 x 128, both /32 for the block scale
    src, weight_map, wu8, su8, q_bf, k_bf = _make_resident_shard(
        tmp_path, layer, out_w, in_w, hc_mult, dim)
    out = tmp_path / "engram"

    # a pre-existing manifest (banks-mode output) to be extended in place
    conv.write_manifest(out, [])
    pre = json.loads((out / "engram-manifest.json").read_text())
    assert "residents" not in pre

    entry, parity, sidecar = conv.convert_residents(
        src, out, weight_map, layers=[layer], wait=False, poll=0.0, verify=True)
    backup = tmp_path / "manifest.bak.json"
    conv.update_manifest_with_residents(out, entry, backup_path=backup)

    # -- sidecar file: names, dtypes, shapes -------------------------------
    assert sidecar.name == "engram-residents.safetensors"
    header, _ = dc.read_safetensors_header(str(sidecar))
    names = {k for k in header if k != "__metadata__"}
    assert names == {
        f"layers.{layer}.engram.wkv.weight",
        f"layers.{layer}.engram.wkv.scales",
        f"layers.{layer}.engram.wkv.biases",
        f"layers.{layer}.engram.q_weight",
        f"layers.{layer}.engram.k_weight",
    }
    assert header[f"layers.{layer}.engram.wkv.weight"]["dtype"] == "U32"
    assert header[f"layers.{layer}.engram.wkv.scales"]["dtype"] == "BF16"
    assert header[f"layers.{layer}.engram.q_weight"]["dtype"] == "F32"
    assert header[f"layers.{layer}.engram.q_weight"]["shape"] == [hc_mult, dim]

    # -- parity: q8 wkv roundtrip high-cos, q/k exact ----------------------
    p = parity[layer]
    assert p["wkv_cos_row_min"] >= 0.999, p
    assert p["q_exact"] and p["k_exact"]
    assert p["q_max_abs_err"] == 0.0 and p["k_max_abs_err"] == 0.0

    # -- manifest residents entry + sha recompute --------------------------
    m = json.loads((out / "engram-manifest.json").read_text())
    r = m["residents"]
    assert r["file"] == "engram-residents.safetensors"
    assert r["total_bytes"] == sidecar.stat().st_size
    assert len(r["sha256"]) == 64
    tnames = {t["name"] for t in r["tensors"]}
    assert tnames == names
    # sha over the manifest WITHOUT manifest_sha256, indent=2 -- same recipe as write_manifest
    import hashlib
    payload = {k: v for k, v in m.items() if k != "manifest_sha256"}
    assert m["manifest_sha256"] == hashlib.sha256(
        json.dumps(payload, indent=2).encode()).hexdigest()
    # `residents` sits right after `layers`
    keys = list(m.keys())
    assert keys[keys.index("layers") + 1] == "residents"
    # backup captured the pre-update manifest verbatim
    assert backup.read_text() == json.dumps(pre, indent=2)

    # -- loader: shapes/dtypes + wkv == dequant-and-matmul -----------------
    res = load_engram_residents(out, layer)
    assert res.q_weight.dtype == mx.float32 and res.k_weight.dtype == mx.float32
    assert tuple(res.q_weight.shape) == (hc_mult, dim)
    assert res.dim == dim and res.hc_mult == hc_mult
    # q/k arrays are the exact widened source
    assert np.array_equal(np.array(res.q_weight), q_bf)
    assert np.array_equal(np.array(res.k_weight), k_bf)

    rng = np.random.default_rng(11)
    x = mx.array(rng.standard_normal((2, in_w)).astype(np.float32))
    kv = res.wkv(x)
    assert tuple(kv.shape) == (2, out_w)
    w = mx.dequantize(res.wkv_packed, res.wkv_scales, res.wkv_biases,
                      group_size=res.group_size, bits=res.bits, mode="affine")
    ref = x @ w.T
    assert bool(mx.all(mx.isfinite(kv)).item())
    # quantized_matmul == dequantize-then-matmul up to bf16 accumulation; cosine is the
    # robust wiring check (a transposed/misindexed wkv would collapse it toward 0).
    a, b = np.array(kv).reshape(-1), np.array(ref).reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    assert cos >= 0.999, cos
    rel = float(mx.max(mx.abs(kv - ref)).item()) / (float(mx.max(mx.abs(ref)).item()) + 1e-6)
    assert rel < 5e-2, rel

    # -- build EngramV41 from the residents + a synthetic row cache ---------
    head_dim, cols = 64, in_w // 64                 # embed reshape -> cols*head_dim == in_w
    n_emb = 40
    f32 = (rng.standard_normal((n_emb, head_dim)) * 0.1).astype(np.float32)
    rec = dc.engram_chunk_records(f32, group=head_dim)
    bank_path = tmp_path / "bank.bin"
    bank_path.write_bytes(np.ascontiguousarray(rec).tobytes())
    cache = NGramRowCache(
        FileRowReader(bank_path, row_bytes=rec.shape[1], num_rows=n_emb),
        RowGeometry(head_dim, 8, head_dim), num_rows=n_emb, cache_rows=16)
    module = res.build_module(row_cache=cache, layer_hash_index=0,
                              norm_eps=1e-6, clamp_value=1e-6)
    from mtplx.engram_v41 import _StepState
    B, L = 1, 2
    row_ids = rng.integers(0, n_emb, size=(B, L, 1, cols)).astype(np.int64)
    xx = mx.array(rng.standard_normal((B, L, hc_mult, dim)).astype(np.float32))
    outp = module(xx, np.zeros((B, L), np.int64), _StepState(row_ids=row_ids))
    assert tuple(outp.shape) == (B, L, hc_mult, dim)
    assert bool(mx.all(mx.isfinite(outp)).item())
