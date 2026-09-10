"""Tests for the generic bounded resident-row cache (mtplx/ngram_row_cache.py).

LRU semantics on a synthetic on-disk bank: byte budget, hit/miss stats, eviction never
changes returned values, batched contiguous misses coalesce into one positional read, and
requests larger than the cache still serve correctly.  Plus the MLX affine dequant path
against the converter's numpy dequant.

CPU only.  Run under ``nice -n 19``, without ``-n auto``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

import mtplx  # noqa: F401  (assert below binds the worktree copy)
import mtplx.deepseek_v41_convert as dc
from mtplx.ngram_row_cache import (
    FileRowReader,
    NGramRowCache,
    RowGeometry,
    cache_bytes_from_env,
)

mx.set_default_device(mx.cpu)

# guard the editable-install CWD-shadowing trap: this must be the worktree's module
import mtplx.ngram_row_cache as _nrc
_REPO_ROOT = Path(__file__).resolve().parents[1]
assert Path(_nrc.__file__).resolve().is_relative_to(_REPO_ROOT), (_nrc.__file__, _REPO_ROOT)

ROW_BYTES = 272


def _make_random_bank(tmp_path: Path, n_rows: int, seed: int = 0) -> tuple[Path, np.ndarray]:
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(n_rows, ROW_BYTES), dtype=np.uint8)
    p = tmp_path / "bank.bin"
    p.write_bytes(raw.tobytes())
    return p, raw


def _open(tmp_path, n_rows, *, cache_rows=None, cache_bytes=None, seed=0):
    p, raw = _make_random_bank(tmp_path, n_rows, seed)
    reader = FileRowReader(p, row_bytes=ROW_BYTES, num_rows=n_rows)
    geom = RowGeometry(256, 8, 64)
    cache = NGramRowCache(reader, geom, num_rows=n_rows,
                          cache_rows=cache_rows, cache_bytes=cache_bytes)
    return cache, raw


# --------------------------------------------------------------------------
def test_slot_count_and_budget(tmp_path):
    cache, raw = _open(tmp_path, 100, cache_bytes=10 * ROW_BYTES)
    assert cache.slot_count == 10
    assert cache.budget_bytes == 10 * ROW_BYTES
    cache.gather_bytes(list(range(50)))
    assert cache.resident_rows <= cache.slot_count
    assert cache.resident_bytes <= cache.budget_bytes
    # a sub-row budget still holds one row
    small = NGramRowCache(FileRowReader(tmp_path / "bank.bin", row_bytes=ROW_BYTES, num_rows=100),
                          RowGeometry(256, 8, 64), num_rows=100, cache_bytes=1)
    assert small.slot_count == 1


def test_hit_miss_stats(tmp_path):
    cache, raw = _open(tmp_path, 100, cache_rows=64)
    cache.gather_bytes([1, 2, 3])            # 3 fresh misses
    assert cache.stats["misses"] == 3 and cache.stats["hits"] == 0
    cache.gather_bytes([1, 2, 3])            # all hits
    assert cache.stats["hits"] == 3 and cache.stats["misses"] == 3
    cache.gather_bytes([3, 4])               # 3 hit, 4 miss
    assert cache.stats["hits"] == 4 and cache.stats["misses"] == 4
    # duplicate ids in one request count as one unique row
    s0 = dict(cache.stats)
    cache.gather_bytes([9, 9, 9])
    assert cache.stats["misses"] - s0["misses"] == 1


def test_gathered_bytes_match_disk(tmp_path):
    cache, raw = _open(tmp_path, 200, cache_rows=64, seed=3)
    ids = [0, 1, 5, 63, 64, 65, 199, 128, 200 - 100]
    got = cache.gather_bytes(ids)
    for i, r in enumerate(ids):
        assert np.array_equal(got[i], raw[r]), r
    # order and duplicates preserved
    got2 = cache.gather_bytes([7, 7, 3, 7])
    assert np.array_equal(got2[0], raw[7]) and np.array_equal(got2[1], raw[7])
    assert np.array_equal(got2[2], raw[3]) and np.array_equal(got2[3], raw[7])


def test_eviction_never_changes_values(tmp_path):
    cache, raw = _open(tmp_path, 100, cache_rows=3, seed=7)
    # fill, then force evictions
    for r in range(10):
        out = cache.gather_bytes([r])
        assert np.array_equal(out[0], raw[r])
    assert cache.stats["evictions"] >= 7
    assert cache.resident_bytes <= cache.budget_bytes
    # row 0 was evicted long ago; re-gather must reproduce the exact on-disk bytes
    again = cache.gather_bytes([0])
    assert np.array_equal(again[0], raw[0])
    # every historical row still reads back correctly (values are read-authoritative)
    all_ids = list(range(10))
    block = cache.gather_bytes(all_ids)
    for i, r in enumerate(all_ids):
        assert np.array_equal(block[i], raw[r])


def test_batched_misses_coalesce_reads(tmp_path):
    cache, raw = _open(tmp_path, 500, cache_rows=64, seed=11)
    # one contiguous run of fresh misses -> a single positional read
    cache.gather_bytes([10, 11, 12, 13, 14])
    assert cache.stats["reads"] == 1 and cache.stats["rows_read"] == 5
    # three separated singletons -> three reads
    cache.reset()
    cache.gather_bytes([0, 100, 300])
    assert cache.stats["reads"] == 3
    # two runs in one request -> two reads (out-of-order request still coalesces by row id)
    cache.reset()
    cache.gather_bytes([21, 20, 22, 200, 201])
    assert cache.stats["reads"] == 2 and cache.stats["rows_read"] == 5


def test_request_larger_than_cache(tmp_path):
    # 20 rows through a 4-slot cache: served correctly, residency bounded, reads subchunked
    cache, raw = _open(tmp_path, 100, cache_rows=4, seed=5)
    ids = list(range(20))
    got = cache.gather_bytes(ids)
    for i, r in enumerate(ids):
        assert np.array_equal(got[i], raw[r])
    assert cache.resident_rows <= 4
    assert cache.resident_bytes <= cache.budget_bytes
    # contiguous 20-row run, 4 slots -> ceil(20/4) = 5 reads
    assert cache.stats["reads"] == 5 and cache.stats["rows_read"] == 20


def test_out_of_range(tmp_path):
    cache, raw = _open(tmp_path, 50)
    with pytest.raises(IndexError):
        cache.gather_bytes([50])
    with pytest.raises(IndexError):
        cache.gather_bytes([-1])


def test_dequantize_matches_numpy(tmp_path):
    # valid affine-8 records so the MLX dequant path is meaningful
    rng = np.random.default_rng(2)
    n = 80
    f32 = rng.standard_normal((n, 256)).astype(np.float32)
    rec = dc.engram_chunk_records(f32)              # [n, 272] uint8
    p = tmp_path / "affine.bin"
    p.write_bytes(np.ascontiguousarray(rec).tobytes())
    reader = FileRowReader(p, row_bytes=272, num_rows=n)
    cache = NGramRowCache(reader, RowGeometry(256, 8, 64), num_rows=n, cache_rows=16)

    ids = np.array([[0, 1, 2], [3, 63, 79]])        # exercise shape preservation
    dq = np.array(cache.dequantize(ids).astype(mx.float32))
    assert dq.shape == (2, 3, 256)

    # numpy affine dequant of the same records (converter reference)
    flat = ids.reshape(-1)
    s = np.ascontiguousarray(rec[flat, 256:264]).view("<u2")
    b = np.ascontiguousarray(rec[flat, 264:272]).view("<u2")
    ref = dc.dequant_affine_record(rec[flat, 0:256].astype(np.uint8), s, b).reshape(2, 3, 256)
    assert np.max(np.abs(dq - ref)) < 0.5           # bf16 rounding only


def test_geometry_validation():
    RowGeometry(256, 8, 64)                          # ok
    with pytest.raises(ValueError):
        RowGeometry(255, 8, 64)                      # not a multiple of group_size
    with pytest.raises(ValueError):
        RowGeometry(256, 5, 64)                      # unsupported bits


def test_cache_bytes_from_env(monkeypatch):
    assert cache_bytes_from_env("1GiB") == 1073741824
    monkeypatch.setenv("MTPLX_ENGRAM_CACHE_LIMIT", "1.5 GB")
    assert cache_bytes_from_env("1GiB") == 1500000000
    monkeypatch.delenv("MTPLX_ENGRAM_CACHE_LIMIT", raising=False)
    assert cache_bytes_from_env(4096) == 4096
