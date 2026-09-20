"""CPU tests for F20 engram read lookahead (scripts/deepseek_v41 f6 + f16).

Pins MLX to CPU before any import that pulls in ``mlx.core`` (``NGramRowCache`` and
``f16.pipeline`` both do; MLX defaults to Metal otherwise).  No Metal, no GPU, no
artifact: the cache path is pure numpy + ``os.preadv`` and the driver tests stub the
group run, so nothing here issues an ``mx`` kernel beyond a tiny ``concatenate`` on CPU.

Proves the spec's section-4 contract:
  1. Identical state -- stock serial ``gather_bytes`` == F6 parallel == F6 + F20
     lookahead (returned bytes, ``_lru`` order+slots, ``_arena``, ``_free`` and every
     ``cache.stats`` counter), across two interleaved groups on a shared cache, with and
     without evictions, and across two caches.
  2. Rows a partner inserted between prefetch and collect are ignored; rows evicted in
     between are re-read; a prefetch future that raises falls back to a read at the
     collect; ``clear_pending`` leaves no future un-joined.
  3. ``prefetch_rows`` mutates no cache state (snapshot before/after).
  4. Driver -- lookahead is called once per pipelined forward with BOTH groups' row ids
     before any group slice runs; disabled -> the bound no-op; enabled without the F6
     gather installed -> construction refuses.
"""
from __future__ import annotations

import os
import sys
import types
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts" / "deepseek_v41"
_F6_DIR = _SCRIPTS / "f6"
_PACKED = _REPO / "docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed"
# .f16-site holds greenlet (private dir, NOT the shared venv); honour F16SITE like the
# F16 test does (default = the f16-pipeline worktree's dir).  Needed before f16.pipeline.
_F16_SITE = os.environ.get(
    "F16SITE",
    "/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f16-pipeline/.f16-site",
)
for _p in (str(_SCRIPTS), str(_F6_DIR), str(_PACKED), str(_F16_SITE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# engram_parallel imported TOP-LEVEL (f6 dir), exactly as the F6 stager imports it, so
# the module the F6 install patches gather_bytes from is the same one f16.pipeline
# resolves at enable time.
import engram_parallel as ep  # noqa: E402
from mtplx.ngram_row_cache import FileRowReader, NGramRowCache, RowGeometry  # noqa: E402

from f16 import pipeline as pl  # noqa: E402  (package; scripts dir on path)

ROW_BYTES = 264  # real Engram record width (mxfp8: 256 code + 8 e8m0 scale bytes)
GEOM = RowGeometry(values_per_row=256, bits=8, group_size=32, mode="mxfp8")  # -> 264 B/row


# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------
def _bank_bytes(num_rows: int) -> np.ndarray:
    """Deterministic distinct content per row (row id in the first 4 bytes)."""
    rows = np.arange(num_rows, dtype=np.uint32)
    data = np.empty((num_rows, ROW_BYTES), dtype=np.uint8)
    data[:, :4] = rows.view(np.uint8).reshape(num_rows, 4)
    cols = np.arange(ROW_BYTES - 4, dtype=np.uint64)
    data[:, 4:] = ((rows.astype(np.uint64)[:, None] * 131 + cols[None, :]) & 0xFF).astype(np.uint8)
    return data


@pytest.fixture(autouse=True)
def _reset_states():
    """Per-test install registry + pool lifetime (stats() is per-test)."""
    def _clear():
        for st in list(ep._STATES):
            st.shutdown()
        ep._STATES.clear()

    _clear()
    yield
    _clear()


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    num_rows = 4096
    data = _bank_bytes(num_rows)
    path = tmp_path_factory.mktemp("f20bank") / "engram-synth.bin"
    path.write_bytes(data.tobytes())
    return str(path), num_rows, data


def _make_cache(bank, cache_bytes):
    path, num_rows, _ = bank
    reader = FileRowReader(path, row_bytes=ROW_BYTES, num_rows=num_rows)
    cache = NGramRowCache(reader, GEOM, num_rows=num_rows, cache_bytes=cache_bytes)
    cache._arena[:] = 0  # deterministic (prod uses np.empty); makes full-arena compare meaningful
    return cache


def _install(cache, workers=8):
    ep.install([cache], workers=workers)
    return cache


def _assert_state_identical(a, b, label):
    assert list(a._lru.items()) == list(b._lru.items()), f"{label}: _lru order/slots differ"
    assert list(a._free) == list(b._free), f"{label}: _free differs"
    assert a.stats == b.stats, f"{label}: stats differ {a.stats} vs {b.stats}"
    assert np.array_equal(a._arena, b._arena), f"{label}: arena bytes differ"


# --------------------------------------------------------------------------
# 1. identical-state property (stock serial == F6 parallel == F6 + lookahead)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", list(range(8)))
@pytest.mark.parametrize("slots", [16, 4096])  # 16 -> evictions; 4096 -> none
def test_identical_state_interleaved_groups(bank, seed, slots):
    _, num_rows, data = bank
    rng = np.random.default_rng(seed)
    c_stock = _make_cache(bank, slots * ROW_BYTES)                 # serial reference
    c_f6 = _install(_make_cache(bank, slots * ROW_BYTES))          # parallel, no prefetch
    c_f20 = _install(_make_cache(bank, slots * ROW_BYTES))         # parallel + lookahead

    def mkgroup():
        n = int(rng.integers(1, 40))
        rows = rng.integers(0, num_rows, size=n).tolist()
        base = int(rng.integers(0, num_rows - 4))
        rows += [base, base + 1, base + 2]  # a contiguous run
        rng.shuffle(rows)
        return rows

    for cycle in range(6):
        ga = mkgroup()
        gb = mkgroup()
        gb += rng.choice(ga, size=min(len(ga), 5)).tolist()  # overlap -> partner-insertion path
        rng.shuffle(gb)

        # stock serial: A then B
        os_a = c_stock.gather_bytes(ga)
        os_b = c_stock.gather_bytes(gb)
        # F6 parallel (no prefetch): A then B
        of6_a = c_f6.gather_bytes(ga)
        of6_b = c_f6.gather_bytes(gb)
        # F20: prefetch BOTH groups up front (futures accumulate on the one cache), collect A, B
        ep.prefetch_rows(c_f20, ga)
        ep.prefetch_rows(c_f20, gb)
        of20_a = c_f20.gather_bytes(ga)
        of20_b = c_f20.gather_bytes(gb)
        ep.clear_pending(c_f20)

        for lbl, oa, ob in (("f6", of6_a, of6_b), ("f20", of20_a, of20_b)):
            assert np.array_equal(oa, os_a), f"cycle {cycle} {lbl}: group-A bytes differ"
            assert np.array_equal(ob, os_b), f"cycle {cycle} {lbl}: group-B bytes differ"
        # returned bytes are the true rows (not just equal to each other)
        assert np.array_equal(os_a, data[np.array(ga)])
        assert np.array_equal(os_b, data[np.array(gb)])
        _assert_state_identical(c_f6, c_stock, f"cycle {cycle} f6")
        _assert_state_identical(c_f20, c_stock, f"cycle {cycle} f20")

    if slots == 16:
        assert c_stock.stats["evictions"] > 0  # the scenario really exercised eviction


def test_identical_state_two_caches(bank):
    """Two engram caches (as layers 1 and 14 in the pipeline), each with both groups
    prefetched then collected, stay byte-identical to the serial path."""
    _, num_rows, _ = bank
    rng = np.random.default_rng(123)
    stock = [_make_cache(bank, 16 * ROW_BYTES) for _ in range(2)]
    look = [_install(_make_cache(bank, 16 * ROW_BYTES)) for _ in range(2)]
    for _cycle in range(5):
        ga = rng.integers(0, num_rows, size=int(rng.integers(1, 30))).tolist()
        gb = rng.integers(0, num_rows, size=int(rng.integers(1, 30))).tolist()
        for ci in range(2):
            os_a = stock[ci].gather_bytes(ga)
            os_b = stock[ci].gather_bytes(gb)
            ep.prefetch_rows(look[ci], ga)
            ep.prefetch_rows(look[ci], gb)
            ol_a = look[ci].gather_bytes(ga)
            ol_b = look[ci].gather_bytes(gb)
            ep.clear_pending(look[ci])
            assert np.array_equal(ol_a, os_a)
            assert np.array_equal(ol_b, os_b)
            _assert_state_identical(look[ci], stock[ci], f"cache {ci}")


def test_prefetched_bytes_are_used_not_reread(bank):
    """The lookahead is load-bearing: after prefetch, break the reader; a collect of
    ONLY-prefetched rows still succeeds (no fresh read) with the true bytes.  If the
    collect ignored the prefetch it would fall back to a read and hit the boom reader,
    so this distinguishes 'prefetch used' from 'prefetch silently ignored'."""
    _, num_rows, data = bank
    c = _install(_make_cache(bank, num_rows * ROW_BYTES))
    rows = [10, 11, 12, 500, 900]
    ep.prefetch_rows(c, rows)
    ep._resolve_pending(c)  # join: the reads ran with the ORIGINAL reader

    class _BoomReader:
        row_bytes = ROW_BYTES
        num_rows = 4096
        io_cache_mode = "f-nocache"
        bypass_page_cache = True

        def read_run(self, s, n):
            raise AssertionError("collect performed a fresh read for a prefetched row")

    c.reader = _BoomReader()  # affects only NEW reads; the pending futures captured the old reader
    out = c.gather_bytes(rows)  # all rows prefetched -> remaining is empty -> no fetch, no boom
    ep.clear_pending(c)
    assert np.array_equal(out, data[np.array(rows)]), "prefetched bytes were wrong"


# --------------------------------------------------------------------------
# 2. partner insert / eviction / failed prefetch / clear
# --------------------------------------------------------------------------
def test_partner_inserted_rows_are_ignored(bank):
    """Rows a partner made resident between prefetch and collect are hits at collect;
    the prefetched bytes for them are unused and state matches serial."""
    _, num_rows, _ = bank
    c_stock = _make_cache(bank, num_rows * ROW_BYTES)
    c = _install(_make_cache(bank, num_rows * ROW_BYTES))
    ga = [10, 11, 12]
    gb = [11, 12, 13]  # overlaps A on 11, 12

    c_stock.gather_bytes(ga)
    c_stock.gather_bytes(gb)
    # both groups prefetch (11, 12 are misses at forward start for both); A collects
    # (inserts 10,11,12); B collects (11,12 now hits -> prefetched bytes ignored, 13 miss).
    ep.prefetch_rows(c, ga)
    ep.prefetch_rows(c, gb)
    c.gather_bytes(ga)
    c.gather_bytes(gb)
    ep.clear_pending(c)
    _assert_state_identical(c, c_stock, "partner-insert")


def test_evicted_rows_are_reread(bank):
    """A row resident at forward start (so not prefetched) but evicted by the partner's
    collect is a miss now and is read at the collect; state matches serial."""
    _, num_rows, data = bank
    slots = 4
    c_stock = _make_cache(bank, slots * ROW_BYTES)
    c = _install(_make_cache(bank, slots * ROW_BYTES))
    # forward-start residency: 1,2,3,4 (prime both identically)
    c_stock.gather_bytes([1, 2, 3, 4])
    c.gather_bytes([1, 2, 3, 4])
    ga = [5, 6]      # both miss; evict the oldest (1, 2)
    gb = [1, 7]      # 1 resident at forward start (NOT prefetched), evicted by A -> re-read; 7 miss

    os_a = c_stock.gather_bytes(ga)
    os_b = c_stock.gather_bytes(gb)
    ep.prefetch_rows(c, ga)   # 5, 6
    ep.prefetch_rows(c, gb)   # 1 resident -> skipped; 7 prefetched
    ol_a = c.gather_bytes(ga)  # insert 5,6 -> evict 1,2
    ol_b = c.gather_bytes(gb)  # 1 now a miss (evicted, not prefetched) -> read; 7 from prefetch
    ep.clear_pending(c)

    assert np.array_equal(ol_a, os_a)
    assert np.array_equal(ol_b, os_b)
    assert np.array_equal(ol_b, data[np.array(gb)])  # true bytes for the re-read row
    _assert_state_identical(c, c_stock, "evicted-reread")
    assert c.stats["evictions"] >= 2


def test_failed_prefetch_future_falls_back_to_collect(bank):
    """A prefetch future that raised contributes no rows; those misses are read at the
    collect (a real error would surface there), and the result matches serial."""
    _, num_rows, data = bank
    c_stock = _make_cache(bank, num_rows * ROW_BYTES)
    c = _install(_make_cache(bank, num_rows * ROW_BYTES))
    good = [10, 11, 12]         # prefetched normally (real futures)
    failed = [50, 51]           # only "covered" by a poisoned future -> must be re-read
    extra = [900]               # never prefetched -> read at collect too
    all_rows = good + failed + extra

    ep.prefetch_rows(c, good)
    assert c._f20_pending, "prefetch queued nothing"
    poisoned = Future()
    poisoned.set_exception(EOFError("injected prefetch failure"))
    c._f20_pending.append((poisoned, [(50, 2)]))  # raises on resolve -> 50,51 fall back

    os_all = c_stock.gather_bytes(all_rows)
    ol_all = c.gather_bytes(all_rows)
    ep.clear_pending(c)

    assert np.array_equal(ol_all, os_all)
    assert np.array_equal(ol_all, data[np.array(all_rows)])
    _assert_state_identical(c, c_stock, "failed-prefetch-fallback")


def test_clear_pending_joins_all_futures(bank):
    _, num_rows, _ = bank
    c = _install(_make_cache(bank, num_rows * ROW_BYTES))
    ep.prefetch_rows(c, list(range(100, 148)) + [7, 900, 3000])
    futs = [f for (f, _c) in c._f20_pending]
    assert futs, "no futures to join"
    ep.clear_pending(c)
    assert all(f.done() for f in futs), "clear_pending left a future un-joined"
    assert c._f20_pending == [], "clear_pending did not drop the pending list"


def test_clear_pending_absent_is_noop(bank):
    _, num_rows, _ = bank
    c = _install(_make_cache(bank, num_rows * ROW_BYTES))
    ep.clear_pending(c)              # never prefetched -> attribute absent
    assert c._f20_pending == []


# --------------------------------------------------------------------------
# 3. prefetch_rows mutates nothing
# --------------------------------------------------------------------------
def test_prefetch_mutates_no_cache_state(bank):
    _, num_rows, _ = bank
    c = _install(_make_cache(bank, 16 * ROW_BYTES))
    c.gather_bytes([1, 2, 3, 100, 101])  # prime residency + stats
    snap_lru = list(c._lru.items())
    snap_free = list(c._free)
    snap_stats = dict(c.stats)
    snap_arena = c._arena.copy()

    ep.prefetch_rows(c, [1, 2, 500, 501, 502, 3, 500])  # resident 1,2,3 skipped; dup 500 deduped

    assert list(c._lru.items()) == snap_lru, "_lru changed"
    assert list(c._free) == snap_free, "_free changed"
    assert c.stats == snap_stats, "cache.stats changed"
    assert np.array_equal(c._arena, snap_arena), "arena changed"
    # the lookahead lane's OWN stats advanced (never cache.stats)
    st = ep.stats()
    assert st["lookahead_calls"] == 1
    assert st["lookahead_rows_submitted"] == 3  # only 500,501,502 were non-resident misses
    ep.clear_pending(c)


# --------------------------------------------------------------------------
# 4. driver: call site, disabled no-op, refuse without F6
# --------------------------------------------------------------------------
def _stub_model_with_engram(pairs):
    """A stub model whose backbone.layers carry engram hooks (layer_hash_index +
    row_cache) for each (lhi, cache) pair, plus one non-engram layer."""
    layers = [types.SimpleNamespace(
        engram_hook=types.SimpleNamespace(layer_hash_index=lhi, row_cache=cache))
        for lhi, cache in pairs]
    layers.append(types.SimpleNamespace(engram_hook=None))
    return types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))


def test_enabled_lookahead_prefetches_both_groups_correct_rows(bank):
    _, num_rows, _ = bank
    cache = _install(_make_cache(bank, num_rows * ROW_BYTES))
    model = _stub_model_with_engram([(0, cache), (1, cache)])
    pipe = pl.Pipeline(model, armed=True, engram_lookahead=True)
    assert pipe.engram_lookahead is True
    assert pipe._engram_lookahead == pipe._engram_lookahead_enabled
    assert pipe._f20_prefetch is ep.prefetch_rows  # resolved to the same module F6 installed from

    calls = []
    pipe._f20_prefetch = lambda rc, row_ids: calls.append((id(rc), list(row_ids)))
    n_layers, cols = 2, 3
    cur_a = np.arange(2 * n_layers * cols).reshape(1, 2, n_layers, cols) % num_rows
    cur_b = (np.arange(2 * n_layers * cols).reshape(1, 2, n_layers, cols) + 7) % num_rows
    pipe._engram_lookahead(cur_a, cur_b)

    expected = []
    for cur in (cur_a, cur_b):
        for lhi in (0, 1):
            expected.append((id(cache), np.asarray(cur)[:, :, lhi, :].reshape(-1).tolist()))
    assert calls == expected


def test_run_pipeline_calls_lookahead_before_groups_and_clear_after(bank):
    """The driver calls _engram_lookahead(cur_a, cur_b) once, after the two advances and
    before any group slice runs, and _engram_clear after (even via the finally)."""
    _, num_rows, _ = bank
    model = types.SimpleNamespace(model=types.SimpleNamespace())
    pipe = pl.Pipeline(model, armed=True)  # disabled lookahead; we override the callables
    pipe.backbone._device_route_active = lambda cache, rows: False

    order = []
    seen = {}

    def rec_lookahead(cur_a, cur_b):
        order.append("lookahead")
        seen["a"], seen["b"] = cur_a, cur_b

    def rec_clear():
        order.append("clear")

    def rec_run_groups(gl, gt):
        order.append("run_groups")
        return [(mx.zeros((1, 4, 2)), None), (mx.zeros((1, 2, 2)), None)]

    pipe._engram_lookahead = rec_lookahead
    pipe._engram_clear = rec_clear
    pipe._run_groups = rec_run_groups

    adv = {"n": 0}

    def advance(ids):
        adv["n"] += 1
        return np.full((1, int(ids.shape[1]), 2, 3), adv["n"])

    fake_cache = types.SimpleNamespace(
        offset=0,
        assert_can_admit=lambda n: None,
        engram_state=types.SimpleNamespace(advance=advance),
        advance=lambda n: None,
    )
    ids = mx.array((np.arange(6).reshape(1, 6) % num_rows))
    logits, mh = pipe.pipelined_forward(lambda _ids, _c: None, ids, fake_cache)
    mx.eval(logits)

    assert order == ["lookahead", "run_groups", "clear"]
    assert np.array_equal(seen["a"], np.full((1, 4, 2, 3), 1))  # advance(ids_a) -> group A
    assert np.array_equal(seen["b"], np.full((1, 2, 2, 3), 2))  # advance(ids_b) -> group B
    assert tuple(logits.shape) == (1, 6, 2)
    assert pipe.counters["pipelined_forwards"] == 1


def test_run_pipeline_clears_even_on_group_error(bank):
    """A group error still runs _engram_clear (the finally) before propagating."""
    _, num_rows, _ = bank
    model = types.SimpleNamespace(model=types.SimpleNamespace())
    pipe = pl.Pipeline(model, armed=True)
    pipe.backbone._device_route_active = lambda cache, rows: False

    cleared = {"n": 0}
    pipe._engram_lookahead = lambda a, b: None
    pipe._engram_clear = lambda: cleared.__setitem__("n", cleared["n"] + 1)

    def boom(gl, gt):
        raise RuntimeError("group blew up")

    pipe._run_groups = boom
    fake_cache = types.SimpleNamespace(
        offset=0,
        assert_can_admit=lambda n: None,
        engram_state=types.SimpleNamespace(advance=lambda ids: np.zeros((1, int(ids.shape[1]), 2, 3))),
        advance=lambda n: None,
    )
    ids = mx.array((np.arange(6).reshape(1, 6) % num_rows))
    with pytest.raises(RuntimeError, match="group blew up"):
        pipe.pipelined_forward(lambda _ids, _c: None, ids, fake_cache)
    assert cleared["n"] == 1, "clear did not run on the error path"


def test_disabled_lookahead_is_bound_noop(bank):
    _, num_rows, _ = bank
    cache = _install(_make_cache(bank, num_rows * ROW_BYTES))
    model = _stub_model_with_engram([(0, cache)])
    pipe = pl.Pipeline(model, armed=True)  # engram_lookahead defaults False
    assert pipe.engram_lookahead is False
    assert pipe._engram_lookahead == pipe._engram_lookahead_noop
    assert pipe._engram_clear == pipe._engram_clear_noop
    assert not hasattr(pipe, "_f20_layers")
    # the bound no-ops do nothing and issue no prefetch
    pipe._engram_lookahead(np.zeros((1, 2, 1, 3)), np.zeros((1, 2, 1, 3)))
    pipe._engram_clear()
    assert not getattr(cache, "_f20_pending", None)


def test_enable_without_f6_installed_refuses(bank):
    _, num_rows, _ = bank
    cache = _make_cache(bank, num_rows * ROW_BYTES)  # NOT F6-installed
    model = _stub_model_with_engram([(0, cache)])
    with pytest.raises(RuntimeError, match="F6"):
        pl.Pipeline(model, armed=True, engram_lookahead=True)


def test_enable_without_engram_hooks_refuses(bank):
    model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[
        types.SimpleNamespace(engram_hook=None), types.SimpleNamespace(engram_hook=None)]))
    with pytest.raises(RuntimeError, match="no engram-hook"):
        pl.Pipeline(model, armed=True, engram_lookahead=True)
