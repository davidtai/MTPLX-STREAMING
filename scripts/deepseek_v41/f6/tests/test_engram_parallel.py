"""CPU tests for the F6 parallel Engram row-cache miss path.

Proves the installed concurrent-miss ``gather_bytes`` is bit-for-bit identical
to the stock serial path (returned bytes, ``_lru`` order + slots, ``_free``,
``stats``, ``_arena``) across random batches with duplicates, contiguous runs,
evictions, a gather larger than the cache, out-of-range errors, and an injected
read failure (all-or-nothing: no partially inserted rows). No Metal: the cache
path is pure numpy + ``os.preadv``; ``dequantize`` (the only mx op) is never
called. conftest pins mx to CPU before import regardless.
"""
import inspect

import numpy as np
import pytest

# conftest.py pins mx to CPU and adds the F6 dir to sys.path before this import.
import engram_parallel as ep
from mtplx.ngram_row_cache import FileRowReader, NGramRowCache, RowGeometry

ROW_BYTES = 264  # real Engram record width (mxfp8: 256 code + 8 e8m0 scale bytes)
GEOM = RowGeometry(values_per_row=256, bits=8, group_size=32, mode="mxfp8")  # -> 264 B/row


# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------
def _bank_bytes(num_rows: int) -> np.ndarray:
    """Deterministic distinct content per row: first 4 bytes = row id (LE),
    remaining bytes a per-(row, col) pattern."""
    rows = np.arange(num_rows, dtype=np.uint32)
    data = np.empty((num_rows, ROW_BYTES), dtype=np.uint8)
    data[:, :4] = rows.view(np.uint8).reshape(num_rows, 4)
    cols = np.arange(ROW_BYTES - 4, dtype=np.uint64)
    data[:, 4:] = ((rows.astype(np.uint64)[:, None] * 131 + cols[None, :]) & 0xFF).astype(np.uint8)
    return data


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    num_rows = 8192
    data = _bank_bytes(num_rows)
    path = tmp_path_factory.mktemp("f6bank") / "engram-synth.bin"
    path.write_bytes(data.tobytes())
    return str(path), num_rows, data


def _make_cache(bank, cache_bytes):
    path, num_rows, _ = bank
    reader = FileRowReader(path, row_bytes=ROW_BYTES, num_rows=num_rows)
    cache = NGramRowCache(reader, GEOM, num_rows=num_rows, cache_bytes=cache_bytes)
    cache._arena[:] = 0  # deterministic (prod uses np.empty); makes full-arena compare meaningful
    return cache


def _pair(bank, cache_bytes):
    """A stock (serial) cache and an installed (parallel) cache, identical state."""
    ser = _make_cache(bank, cache_bytes)
    par = _make_cache(bank, cache_bytes)
    ep.install([par], workers=8)
    return ser, par


def _assert_equiv(par, ser, out_par, out_ser):
    assert np.array_equal(out_par, out_ser), "returned bytes differ"
    assert list(par._lru.items()) == list(ser._lru.items()), "LRU order/slots differ"
    assert par._free == ser._free, "_free differs"
    assert par.stats == ser.stats, f"stats differ: {par.stats} vs {ser.stats}"
    assert np.array_equal(par._arena, ser._arena), "arena bytes differ"


def _assert_consistent(cache):
    slots = list(cache._lru.values())
    assert len(slots) == len(set(slots)), "duplicate resident slots"
    lru_set, free_set = set(slots), set(cache._free)
    assert lru_set.isdisjoint(free_set), "a slot is both resident and free"
    assert lru_set | free_set == set(range(cache.slot_count)), "slots not partitioned"
    assert len(cache._lru) + len(cache._free) == cache.slot_count


def _run_batches(bank, cache_bytes, batches):
    ser, par = _pair(bank, cache_bytes)
    for batch in batches:
        out_ser = ser.gather_bytes(batch)
        out_par = par.gather_bytes(batch)
        _assert_equiv(par, ser, out_par, out_ser)
        _assert_consistent(par)
    return ser, par


# --------------------------------------------------------------------------
# equivalence across scenarios
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", list(range(12)))
def test_equiv_random_no_eviction(bank, seed):
    """Large cache (no eviction): random batches with duplicates + contiguous runs."""
    _, num_rows, _ = bank
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(6):
        n = int(rng.integers(1, 200))
        rows = rng.integers(0, num_rows, size=n).tolist()
        # inject a contiguous run and duplicates
        base = int(rng.integers(0, num_rows - 5))
        rows += [base, base + 1, base + 2, base + 3]
        rows += rng.choice(rows, size=min(len(rows), 8)).tolist()
        rng.shuffle(rows)
        batches.append(rows)
    _run_batches(bank, num_rows * ROW_BYTES, batches)


@pytest.mark.parametrize("seed", list(range(12)))
def test_equiv_random_with_evictions(bank, seed):
    """Small cache (16 slots): batches exceed capacity -> evictions must match."""
    _, num_rows, _ = bank
    rng = np.random.default_rng(1000 + seed)
    batches = []
    for _ in range(10):
        n = int(rng.integers(1, 40))
        rows = rng.integers(0, num_rows, size=n).tolist()
        base = int(rng.integers(0, num_rows - 5))
        rows += [base, base + 1, base + 2]
        rng.shuffle(rows)
        batches.append(rows)
    ser, par = _run_batches(bank, 16 * ROW_BYTES, batches)
    assert ser.stats["evictions"] > 0  # the scenario really exercised eviction
    assert par.stats["evictions"] == ser.stats["evictions"]


def test_equiv_contiguous_run_single_read(bank):
    """A fully contiguous batch coalesces into ONE read_run (reads == 1)."""
    _, num_rows, _ = bank
    ser, par = _pair(bank, num_rows * ROW_BYTES)
    batch = list(range(100, 148))  # 48 consecutive rows == 1 contiguous run
    out_ser = ser.gather_bytes(batch)
    out_par = par.gather_bytes(batch)
    _assert_equiv(par, ser, out_par, out_ser)
    assert par.stats["reads"] == 1 and par.stats["rows_read"] == 48
    assert par.stats["misses"] == 48 and par.stats["hits"] == 0


def test_equiv_gather_larger_than_cache(bank):
    """A single gather bigger than the cache: rows copied out as read, evicting
    within the same call. Serial and parallel must agree."""
    _, num_rows, _ = bank
    slot_count = 16
    ser, par = _pair(bank, slot_count * ROW_BYTES)
    batch = list(range(0, 100))  # 100 rows > 16 slots -> in-call eviction
    out_ser = ser.gather_bytes(batch)
    out_par = par.gather_bytes(batch)
    _assert_equiv(par, ser, out_par, out_ser)
    # returned bytes are the true row bytes even for rows evicted later in-call
    _, _, data = bank
    assert np.array_equal(out_par, data[np.array(batch)])
    assert par.resident_rows == slot_count


def test_equiv_duplicates_fill_all_positions(bank):
    _, num_rows, _ = bank
    ser, par = _pair(bank, num_rows * ROW_BYTES)
    batch = [5, 5, 7, 5, 7, 9, 9]
    out_ser = ser.gather_bytes(batch)
    out_par = par.gather_bytes(batch)
    _assert_equiv(par, ser, out_par, out_ser)
    _, _, data = bank
    for i, r in enumerate(batch):
        assert np.array_equal(out_par[i], data[r])


def test_equiv_empty_batch(bank):
    _, num_rows, _ = bank
    ser, par = _pair(bank, num_rows * ROW_BYTES)
    out_ser = ser.gather_bytes([])
    out_par = par.gather_bytes([])
    assert out_par.shape == (0, ROW_BYTES) and out_ser.shape == (0, ROW_BYTES)
    assert par.stats == ser.stats
    st = ep.stats()
    assert st["parallel_calls"] >= 1


def test_hit_then_miss_ordering(bank):
    """Second gather mixes resident (hit) + new (miss) rows; LRU touch order and
    insertion order must match serial exactly."""
    _, num_rows, _ = bank
    ser, par = _pair(bank, num_rows * ROW_BYTES)
    for batch in ([10, 20, 30, 40], [40, 10, 55, 30, 60, 20]):
        out_ser = ser.gather_bytes(batch)
        out_par = par.gather_bytes(batch)
        _assert_equiv(par, ser, out_par, out_ser)


# --------------------------------------------------------------------------
# error behaviour
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [-1, 8192, 99999])
def test_out_of_range_raises_and_no_state_change(bank, bad):
    _, num_rows, _ = bank
    ser, par = _pair(bank, num_rows * ROW_BYTES)
    # prime some state first
    ser.gather_bytes([1, 2, 3])
    par.gather_bytes([1, 2, 3])
    snap_lru = list(par._lru.items())
    snap_free = list(par._free)
    snap_stats = dict(par.stats)
    with pytest.raises(IndexError):
        par.gather_bytes([1, bad, 2])
    with pytest.raises(IndexError):
        ser.gather_bytes([1, bad, 2])
    # no mutation from the failed call (validation precedes any touch/read)
    assert list(par._lru.items()) == snap_lru
    assert list(par._free) == snap_free
    assert par.stats == snap_stats
    assert list(par._lru.items()) == list(ser._lru.items())


class _FailingReader(FileRowReader):
    def __init__(self, *a, fail_rows=(), **k):
        super().__init__(*a, **k)
        self.fail_rows = set(int(r) for r in fail_rows)

    def read_run(self, start, count):
        for r in range(start, start + count):
            if r in self.fail_rows:
                raise EOFError(f"injected read failure at row {r}")
        return super().read_run(start, count)


def test_injected_read_failure_no_partial_insert(bank):
    """A read error propagates and leaves the cache uncorrupted: NO miss row is
    partially inserted (all-or-nothing for the miss batch), the slot partition
    invariant holds, and previously resident rows are untouched."""
    path, num_rows, data = bank
    reader = _FailingReader(path, row_bytes=ROW_BYTES, num_rows=num_rows, fail_rows=[4242])
    cache = NGramRowCache(reader, GEOM, num_rows=num_rows, cache_bytes=num_rows * ROW_BYTES)
    cache._arena[:] = 0
    ep.install([cache], workers=8)

    # prime with a successful all-new gather
    cache.gather_bytes([100, 101, 102, 500, 900])
    _assert_consistent(cache)
    snap_lru = list(cache._lru.items())
    snap_free = list(cache._free)
    snap_stats = dict(cache.stats)
    resident_before = {r: cache._arena[slot].copy() for r, slot in cache._lru.items()}

    # a batch of all-new rows, one of which (4242) fails to read
    with pytest.raises(EOFError):
        cache.gather_bytes([2000, 3000, 4242, 5000])

    # not corrupted: identical LRU (order + slots), identical free list, identical
    # stats (populate never ran), and every previously resident row still correct.
    assert list(cache._lru.items()) == snap_lru, "resident set/order changed after failure"
    assert list(cache._free) == snap_free, "_free changed after failure"
    assert cache.stats == snap_stats, "stats advanced despite failed read"
    _assert_consistent(cache)
    for r, before in resident_before.items():
        assert np.array_equal(cache._arena[cache._lru[r]], before)
        assert np.array_equal(cache._arena[cache._lru[r]], data[r])
    # none of the (would-be) miss rows leaked in
    for r in (2000, 3000, 4242, 5000):
        assert r not in cache._lru


# --------------------------------------------------------------------------
# install validation / routing / idempotency
# --------------------------------------------------------------------------
def test_install_rejects_non_nocache_reader(bank):
    path, num_rows, _ = bank
    reader = FileRowReader(path, row_bytes=ROW_BYTES, num_rows=num_rows, bypass_page_cache=False)
    cache = NGramRowCache(reader, GEOM, num_rows=num_rows, cache_bytes=num_rows * ROW_BYTES)
    with pytest.raises(RuntimeError, match="F_NOCACHE"):
        ep.install([cache])


def test_install_rejects_wrong_reader_type(bank):
    _, num_rows, _ = bank

    class _Fake:
        row_bytes = ROW_BYTES
        num_rows = 8192
        io_cache_mode = "f-nocache"
        bypass_page_cache = True

        def read_run(self, s, c):
            return b"\0" * (c * ROW_BYTES)

    cache = NGramRowCache(_Fake(), GEOM, num_rows=num_rows, cache_bytes=num_rows * ROW_BYTES)
    with pytest.raises(TypeError, match="FileRowReader"):
        ep.install([cache])


def test_install_idempotent_and_routes_through_dequantize(bank):
    _, num_rows, _ = bank
    cache = _make_cache(bank, num_rows * ROW_BYTES)
    s1 = ep.install([cache], workers=8)
    s2 = ep.install([cache], workers=8)  # idempotent: same state, no re-bind error
    assert s1 is s2
    # the bound method is the parallel one, and dequantize() routes through it
    assert cache.gather_bytes.__func__ is ep._parallel_gather_bytes
    assert "self.gather_bytes" in inspect.getsource(NGramRowCache.dequantize)


def test_stats_reports_parallelism(bank):
    _, num_rows, _ = bank
    ser, par = _pair(bank, num_rows * ROW_BYTES)
    # a big all-miss batch of scattered rows -> many concurrent single-row reads
    rng = np.random.default_rng(7)
    batch = rng.choice(num_rows, size=400, replace=False).tolist()
    par.gather_bytes(batch)
    st = ep.stats()  # per-test (conftest resets the registry)
    assert st["installed_caches"] == 1 and st["workers"] == 8
    assert st["rows_read_parallel"] == 400          # all 400 distinct rows missed and were read
    assert 1 <= st["reads_submitted"] <= 400        # sub-runs (contiguous rows coalesce)
    assert 1 <= st["max_inflight"] <= 8             # bounded by the worker count
    assert st["parallel_calls"] == 1 and st["calls_with_reads"] == 1
