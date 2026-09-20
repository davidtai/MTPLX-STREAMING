"""F6: concurrent Engram row-cache miss reads (monkeypatch install).

Installs a drop-in replacement for :meth:`mtplx.ngram_row_cache.NGramRowCache.gather_bytes`
that reads all missing rows of ONE ``gather_bytes`` call CONCURRENTLY through a
dedicated :class:`~concurrent.futures.ThreadPoolExecutor`, then populates the
arena/LRU in EXACTLY the serial order so cache state, eviction order, stats and
returned bytes are bit-for-bit identical to the stock serial path.

No tracked runtime file is edited: the retained runner requires a clean tracked
tree at a pinned commit, so the replacement is a monkeypatch bound onto the live
cache *instances* (an instance attribute shadows the class method; ``dequantize``
calls ``self.gather_bytes`` and so picks it up).

Why this is a win (measured by Fable's timing probe on the retained 13.5-TPS
DeepSeek-V4.1 Q4 decode): in decode nearly every Engram lookup is a miss (new
tokens make new n-grams), so each Engram layer (1 and 14) issues ~144 serial
264-byte ``F_NOCACHE`` ``preadv`` round trips per verify cycle on the MAIN
thread while the GPU and the expert SSD path sit idle -- 6.40 ms and 6.31 ms
mean at the two Engram layer transitions, ~2.52 s of a 74.9 s decode. Reading
those misses concurrently collapses the serial latency chain. The same serial
chain is paid ~180k times in prefill, so a load-time install helps TTFT too.

Identical-state argument
------------------------
The stock ``gather_bytes`` reads misses INSIDE the population loop, interleaved
with ``_alloc_slot``. Reads (positional ``preadv`` of a fixed on-disk record) are
pure and touch no cache state; ``_alloc_slot`` touches no file. So moving ALL the
reads to *before* the population loop cannot change any result: the bytes for a
given ``(start, count)`` run are deterministic, and the population loop is
replayed in the EXACT serial order -- same ``_contiguous_runs`` grouping, same
sub-run split by ``slot_count``, same ``_alloc_slot`` sequence, same
``stats`` increments, same ``_lru``/``_arena``/``_free`` mutations. Hits are
processed first in the same ``positions`` (first-seen) order, misses in the same
``sorted`` order. Result: returned bytes, ``_lru`` order, ``_free``, ``_arena``
and every ``stats`` counter are identical to serial.

Failure semantics (stronger than serial): all reads settle before ANY cache
mutation, so a read error propagates with NO partially inserted rows -- the
resident set is unchanged (only hit LRU-order touches, which stock also does).
Stock serial can leave earlier miss runs inserted before the failing read; the
parallel path is all-or-nothing for the miss batch.

Public API: :func:`install`, :func:`install_from_env`, :func:`stats`.
"""
from __future__ import annotations

import os
import threading
import types
from concurrent.futures import ThreadPoolExecutor, wait as _futures_wait

import numpy as np

# NB: importing NGramRowCache pulls in ``mlx.core`` (module-level in
# ngram_row_cache). Callers that must stay on CPU set the default device before
# importing this module; nothing here touches Metal.
from mtplx.ngram_row_cache import FileRowReader, NGramRowCache, _contiguous_runs

__all__ = [
    "install",
    "install_from_env",
    "stats",
    "ParallelReadState",
    "prefetch_rows",
    "clear_pending",
]

_INSTALL_ATTR = "_f6_parallel"
# Instance attribute (on each NGramRowCache) holding the F20 lookahead's in-flight
# read futures for the current forward.  A list of ``(future, requests)`` where
# ``requests`` is the sub-run ``(start, count)`` list the future covers.  Purely
# on the instance, so cache state stays untouched and the driver/collect share it
# regardless of which module object F6 was installed from.
_PENDING_ATTR = "_f20_pending"
_STATES: "list[ParallelReadState]" = []
_STATES_LOCK = threading.Lock()


class ParallelReadState:
    """Shared read-parallelism state for the caches installed together.

    One pool (and one stats block) is shared by both Engram banks so the reads of
    a single ``gather_bytes`` call fan out across up to ``workers`` threads. The
    pool is dedicated (not a global executor) so Engram reads never queue behind
    unrelated work.
    """

    def __init__(self, *, workers: int = 16, pool: "ThreadPoolExecutor | None" = None) -> None:
        self.workers = int(workers)
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        self._own_pool = pool is None
        self.pool = pool if pool is not None else ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix="f6-engram-read"
        )
        self._lock = threading.Lock()      # guards stats against a concurrent stats() read
        self.caches: "list[NGramRowCache]" = []
        self.stats = {
            "parallel_calls": 0,       # patched gather_bytes invocations
            "calls_with_reads": 0,     # of those, calls that issued >=1 concurrent read
            "reads_submitted": 0,      # sub-run reads dispatched to the pool (at collect)
            "rows_read_parallel": 0,   # rows fetched through the pool (at collect)
            "max_inflight": 0,         # peak concurrent chunk tasks issued by one call
            # F20 lookahead (prefetch_rows): reads issued up front, at forward start.
            # These never touch cache.stats -- they only count the lookahead lane.
            "lookahead_calls": 0,          # prefetch_rows invocations
            "lookahead_rows_submitted": 0, # miss rows dispatched to the pool ahead of the collect
        }

    # -- the concurrent read task -------------------------------------------
    @staticmethod
    def _read_chunk(reader, chunk):
        # read_run allocates its own bytearray/memoryview per call and uses positional
        # os.preadv on the shared fd -> thread-safe, no shared per-call buffer (verified
        # against FileRowReader.read_run). No counters or locks here: this is the hot path.
        return [reader.read_run(start, count) for (start, count) in chunk]

    def fetch(self, reader, requests):
        """Read every ``(start, count)`` run concurrently; return a list of ``bytes``
        aligned to ``requests``.

        The runs are split into at most ``workers`` CONTIGUOUS chunks, one pool task per
        chunk (review 2026-09-19: one ``submit`` per run cost ~12 us x ~144 runs of
        main-thread time per Engram call -- as long as the reads themselves). Contiguous
        chunks keep the results in request order on reassembly.

        Every future is drained before returning; if any read raised, the first error is
        re-raised AFTER all settle, so the caller has not yet mutated the cache (no partial
        insertion, no dangling in-flight read)."""
        n = len(requests)
        if n == 0:
            return []
        size = -(-n // min(self.workers, n))
        futures = [self.pool.submit(self._read_chunk, reader, requests[i:i + size])
                   for i in range(0, n, size)]
        with self._lock:
            self.stats["reads_submitted"] += n
            if len(futures) > self.stats["max_inflight"]:
                self.stats["max_inflight"] = len(futures)   # concurrent chunk tasks, per call
        results: list = []
        error: "BaseException | None" = None
        for fut in futures:
            try:
                results.extend(fut.result())
            except BaseException as exc:  # keep draining the rest, remember the first
                if error is None:
                    error = exc
        if error is not None:
            raise error
        return results

    def note_call(self, *, had_reads: bool, rows: int) -> None:
        # gather_bytes runs on the main thread only; still take the lock so a
        # concurrent stats() read is consistent with the pool-thread counters.
        with self._lock:
            self.stats["parallel_calls"] += 1
            if had_reads:
                self.stats["calls_with_reads"] += 1
                self.stats["rows_read_parallel"] += rows

    def note_lookahead(self, rows: int) -> None:
        """Count one ``prefetch_rows`` call and the miss rows it dispatched (F20
        lookahead lane).  Runs on the generation thread; the lock keeps a
        concurrent ``stats()`` read consistent."""
        with self._lock:
            self.stats["lookahead_calls"] += 1
            self.stats["lookahead_rows_submitted"] += int(rows)

    def shutdown(self) -> None:
        if self._own_pool:
            self.pool.shutdown(wait=True)


def _parallel_gather_bytes(self, row_ids) -> np.ndarray:
    """Instance-bound replacement for ``NGramRowCache.gather_bytes``.

    Mirrors the stock method exactly except that the missing sub-run reads are
    issued concurrently instead of inline in the population loop. All cache-state
    mutation runs on the calling (main) thread in the original serial order.

    Two sources for the miss bytes, and they are interchangeable because a
    positional read of a fixed on-disk record is pure (immutable bytes, no cache
    state):
      * F20 lookahead (``prefetch_rows`` at forward start): rows read ahead of time
        into pool buffers, resolved here into a ``{row: bytes}`` map.  Rows a
        partner group inserted in between are now hits (ignored); rows evicted in
        between are simply not in the map and are read now.
      * the remaining misses: read now through ``state.fetch`` (the stock F6 path).
    With NO lookahead the prefetch map is empty, ``remaining`` is every miss, and
    this is byte-for-byte the stock F6 serial-order collect.  Either way the
    population loop runs over the SAME misses in the SAME order, so ``_lru``,
    ``_arena``, ``_free``, every ``stats`` counter and the returned bytes are
    identical to the stock serial path.
    """
    state: ParallelReadState = self._f6_parallel
    rows = [int(r) for r in row_ids]
    R = len(rows)
    out = np.empty((R, self.row_bytes), dtype=np.uint8)
    if R == 0:
        self.stats["gathers"] += 1
        state.note_call(had_reads=False, rows=0)
        return out

    positions: "dict[int, list[int]]" = {}
    for i, r in enumerate(rows):
        if r < 0 or r >= self.num_rows:
            raise IndexError(f"row {r} out of range [0, {self.num_rows})")
        positions.setdefault(r, []).append(i)

    misses: "list[int]" = []
    for r in positions:
        slot = self._lru.get(r)
        if slot is not None:
            self._lru.move_to_end(r)          # touch (identical to serial)
            self.stats["hits"] += 1
            out[positions[r]] = self._arena[slot]
        else:
            misses.append(r)

    had_reads = False
    parallel_rows = 0
    if misses:
        misses.sort()
        runs = _contiguous_runs(misses)
        # F20 lookahead: bytes for some/all misses may already be in flight from the
        # forward-start prefetch. Resolve those futures into a {row: bytes} map (a
        # failed future contributes nothing -> its rows fall back to a read now).
        prefetched = _resolve_pending(self)
        remaining = [r for r in misses if r not in prefetched] if prefetched else misses
        # Read the rows NOT already covered, grouped + sub-run-split EXACTLY as
        # gather_bytes would (contiguous runs, slot_count). With no lookahead this is
        # every miss, so state.fetch and its stats match the stock parallel path. All
        # reads settle before any cache mutation (all-or-nothing on a read error).
        read_map: "dict[int, np.ndarray]" = {}
        if remaining:
            rem_runs = _contiguous_runs(remaining)
            requests: "list[tuple[int, int]]" = []
            for start, count in rem_runs:
                off = 0
                while off < count:
                    n = min(count - off, self.slot_count)
                    requests.append((start + off, n))
                    off += n
            datas = state.fetch(self.reader, requests)
            had_reads = True
            parallel_rows = sum(c for (_s, c) in requests)
            ri = 0
            for start, count in rem_runs:
                off = 0
                while off < count:
                    n = min(count - off, self.slot_count)
                    block = np.frombuffer(datas[ri], dtype=np.uint8).reshape(n, self.row_bytes)
                    ri += 1
                    for k in range(n):
                        read_map[start + off + k] = block[k]
                    off += n
        # Replay the stock population loop over ALL misses in the EXACT serial order
        # (same runs, same sub-run split, same _alloc_slot sequence, same per-sub-run
        # stats), sourcing each row's bytes from the prefetch map or the just-read map.
        for start, count in runs:
            off = 0
            while off < count:
                n = min(count - off, self.slot_count)
                self.stats["reads"] += 1
                self.stats["rows_read"] += n
                self.stats["misses"] += n
                for k in range(n):
                    r = start + off + k
                    row = prefetched.get(r)
                    if row is None:
                        row = read_map[r]
                    slot = self._alloc_slot()
                    self._arena[slot] = row
                    self._lru[r] = slot           # newest
                    out[positions[r]] = row
                off += n
    self.stats["gathers"] += 1
    state.note_call(had_reads=had_reads, rows=parallel_rows)
    return out


# --------------------------------------------------------------------------
# F20 read lookahead: issue the misses' reads at forward start, collect at the hook
# --------------------------------------------------------------------------
def prefetch_rows(cache, row_ids) -> None:
    """Issue the positional reads for the non-resident rows of ``row_ids`` up front,
    on the generation thread, WITHOUT mutating ANY cache state.

    Called by the F16 pipeline driver at forward start (once per group per engram
    layer), before layer 0 runs.  Computes the misses exactly as ``gather_bytes``
    would (unique in-range rows not in ``_lru`` -- a membership test that never
    reorders the LRU), coalesces them into the same contiguous / slot_count-bounded
    sub-runs, submits those reads to the SAME :class:`ParallelReadState` pool the F6
    collect uses, and remembers the futures on ``cache._f20_pending`` (each paired
    with the sub-runs it covers).  Returns immediately.

    Touches ``_lru``/``_arena``/``_free``/``stats`` not at all; only ``_f20_pending``
    (a private in-flight-read list) and the F6-side stats (``lookahead_calls`` /
    ``lookahead_rows_submitted``) change.  Redundant with another group's prefetch of
    the same row only in wasted I/O (the bytes are identical); the collect decides,
    per row, whether to use a prefetched byte block."""
    state: ParallelReadState = getattr(cache, _INSTALL_ATTR)
    num_rows = cache.num_rows
    lru = cache._lru
    seen: "set[int]" = set()
    misses: "list[int]" = []
    for r in row_ids:
        r = int(r)
        if r in seen:
            continue
        seen.add(r)
        # Residency: membership test only (no move_to_end) -> no LRU/stat mutation.
        # Out-of-range rows are left for the collect's validation to raise, exactly
        # as stock gather_bytes does; prefetch never reads them.
        if r in lru or r < 0 or r >= num_rows:
            continue
        misses.append(r)
    if not misses:
        state.note_lookahead(0)
        return
    misses.sort()
    requests: "list[tuple[int, int]]" = []
    for start, count in _contiguous_runs(misses):
        off = 0
        while off < count:
            n = min(count - off, cache.slot_count)
            requests.append((start + off, n))
            off += n
    # Submit in at most ``workers`` contiguous chunks (mirrors ParallelReadState.fetch;
    # one submit per sub-run costs as much as the reads).  Keep each future with the
    # sub-runs it covers so the collect can decompose it and a per-chunk failure only
    # forces those rows to be re-read.
    reader = cache.reader
    n = len(requests)
    size = -(-n // min(state.workers, n))
    pending = getattr(cache, _PENDING_ATTR, None)
    if pending is None:
        pending = []
        setattr(cache, _PENDING_ATTR, pending)
    for i in range(0, n, size):
        chunk = requests[i:i + size]
        fut = state.pool.submit(ParallelReadState._read_chunk, reader, chunk)
        pending.append((fut, chunk))
    state.note_lookahead(sum(c for (_s, c) in requests))


def _resolve_pending(cache) -> "dict[int, np.ndarray]":
    """Join the cache's pending prefetch futures into a ``{row: bytes[row_bytes]}`` map.

    Called at the collect.  Idempotent and non-destructive: it does NOT clear
    ``_f20_pending`` (both groups' collects resolve the same accumulated futures;
    ``clear_pending`` drops them at forward end).  A future that raised contributes
    NO rows -- those miss rows fall back to a read in the collect, where a genuine
    read error surfaces exactly as the stock path.  Rows a partner already made
    resident are simply not in the collect's miss set, so their map entry is unused."""
    pending = getattr(cache, _PENDING_ATTR, None)
    if not pending:
        return {}
    row_bytes = cache.row_bytes
    out: "dict[int, np.ndarray]" = {}
    for fut, chunk in pending:
        try:
            datas = fut.result()
        except BaseException:  # noqa: BLE001 - a failed prefetch is not fatal here
            continue           # rows re-read in the collect (real error surfaces there)
        for (start, count), data in zip(chunk, datas):
            block = np.frombuffer(data, dtype=np.uint8).reshape(count, row_bytes)
            for k in range(count):
                out[start + k] = block[k]
    return out


def clear_pending(cache) -> None:
    """Drop the cache's leftover prefetch futures, joining them first so no pool read
    is left writing after the forward.  Called by the driver after both groups finish
    (normal or error).  Errors from a failed read are swallowed here -- cleanup only
    needs the pool tasks to have finished; a real read error already surfaced (or will)
    at the collect."""
    pending = getattr(cache, _PENDING_ATTR, None)
    if pending:
        _futures_wait([fut for (fut, _chunk) in pending])
    setattr(cache, _PENDING_ATTR, [])


# --------------------------------------------------------------------------
# install
# --------------------------------------------------------------------------
def _resolve_caches(model_or_caches) -> "list[NGramRowCache]":
    """Resolve the list of Engram :class:`NGramRowCache` from a model, a bank
    list, a single cache, or an iterable of caches."""
    x = model_or_caches
    if isinstance(x, NGramRowCache):
        return [x]
    if isinstance(x, (list, tuple)) and x and all(isinstance(c, NGramRowCache) for c in x):
        return list(x)
    # A model (the retained wrapper) exposes `_engram_banks` (each bank.cache is
    # the NGramRowCache); some call sites hand the wrapper, some the backbone.
    banks = getattr(x, "_engram_banks", None)
    if banks is None:
        inner = getattr(x, "model", None)
        if inner is not None:
            banks = getattr(inner, "_engram_banks", None)
    if not banks:
        raise RuntimeError(
            "F6 parallel install: could not find Engram banks on "
            f"{type(x).__name__} (expected `_engram_banks`, a cache, or a cache list)"
        )
    caches = [bank.cache for bank in banks]
    if not caches or not all(isinstance(c, NGramRowCache) for c in caches):
        raise RuntimeError("F6 parallel install: `_engram_banks[*].cache` are not NGramRowCache")
    return caches


def _validate(cache: NGramRowCache) -> None:
    """Validate the invariants ONCE, at install. No per-call eligibility check
    or fallback ever runs on the hot path (AGENTS.md)."""
    if not isinstance(cache, NGramRowCache):
        raise TypeError(f"F6 parallel install target is not NGramRowCache: {type(cache)!r}")
    reader = cache.reader
    if not isinstance(reader, FileRowReader):
        raise TypeError(
            "F6 parallel install requires a FileRowReader (single-fd positional "
            f"preadv, thread-safe); got {type(reader)!r}"
        )
    if getattr(reader, "io_cache_mode", None) != "f-nocache" or not getattr(
        reader, "bypass_page_cache", False
    ):
        raise RuntimeError(
            "F6 parallel install requires an F_NOCACHE reader "
            f"(io_cache_mode={getattr(reader, 'io_cache_mode', None)!r})"
        )
    if int(reader.row_bytes) != int(cache.row_bytes):
        raise ValueError(
            f"reader row_bytes {reader.row_bytes} != cache row_bytes {cache.row_bytes}"
        )
    if int(cache.row_bytes) <= 0:
        raise ValueError(f"non-positive row_bytes {cache.row_bytes}")
    # The replacement reimplements the serial body, so the internal contract it
    # depends on must be present (isinstance already pins the class; this makes a
    # mismatch fail loudly at install, not at first gather).
    if not callable(getattr(cache, "_alloc_slot", None)):
        raise RuntimeError("NGramRowCache._alloc_slot missing; F6 patch incompatible with this mtplx")
    _need = {"hits", "misses", "reads", "rows_read", "gathers", "evictions"}
    if not _need.issubset(set(getattr(cache, "stats", {}))):
        raise RuntimeError(f"NGramRowCache.stats missing keys {_need - set(cache.stats)}")


def install(model_or_caches, *, workers: int = 16, pool=None) -> ParallelReadState:
    """Bind the concurrent-miss ``gather_bytes`` onto the Engram caches.

    ``model_or_caches`` is the retained model wrapper (``_engram_banks``), a list
    of :class:`NGramRowCache`, or one cache. All resolved caches share one
    :class:`ParallelReadState` (one pool + one stats block), so the two Engram
    banks fan their reads across the same ``workers`` threads.

    Validates class identity, reader type, F_NOCACHE fd and row_bytes once, then
    rebinds the method. Idempotent: a cache already carrying an F6 binding keeps
    its existing state (its state is reused for the returned handle).
    """
    caches = _resolve_caches(model_or_caches)

    existing = [getattr(c, _INSTALL_ATTR, None) for c in caches]
    already = [s for s in existing if s is not None]
    if already and all(s is already[0] for s in existing):
        # fully installed already under one shared state -> no-op, return it
        return already[0]
    if any(existing):  # partial / mixed install is a programming error
        raise RuntimeError("F6 parallel install: caches are partially/inconsistently installed")

    for cache in caches:
        _validate(cache)

    state = ParallelReadState(workers=workers, pool=pool)
    state.caches = list(caches)
    for cache in caches:
        setattr(cache, _INSTALL_ATTR, state)
        # instance attribute shadows the class method; dequantize() -> gather_bytes
        cache.gather_bytes = types.MethodType(_parallel_gather_bytes, cache)
    with _STATES_LOCK:
        _STATES.append(state)
    return state


def install_from_env(model_or_caches, *, site: "str | None" = None) -> "ParallelReadState | None":
    """Install iff ``MTPLX_DSV41_F6_ENGRAM_PARALLEL=1``.

    ``site`` (``"load"`` or ``"decode"``) is passed by the runner stager so the
    single ``MTPLX_DSV41_F6_INSTALL`` env value picks which anchor actually
    installs; when ``site`` is given, install happens only if it matches. Called
    with no ``site`` (item-2 contract), only the parallel-enable env gates it.

    ``MTPLX_DSV41_F6_ENGRAM_WORKERS`` overrides the worker count (default 16).
    ``MTPLX_DSV41_F6_ENGRAM_LOOKAHEAD=1`` is refused loudly: the read-lookahead lane
    (``prefetch_rows`` / ``clear_pending``) now lives in this module, but the F6
    standalone install only rebinds ``gather_bytes`` -- it does NOT wire the driver
    that issues the prefetch at forward start.  The lookahead is driven by the F16
    pipeline via ``MTPLX_DSV41_F20_ENGRAM_LOOKAHEAD=1`` (read in ``f16.install``), which
    refuses unless this F6 gather is already installed on the hook caches.  Setting the
    F6 flag alone would arm an env that installs nothing (AGENTS.md: fail once, clearly).
    """
    if os.environ.get("MTPLX_DSV41_F6_ENGRAM_PARALLEL") != "1":
        return None
    if os.environ.get("MTPLX_DSV41_F6_ENGRAM_LOOKAHEAD") == "1":
        raise RuntimeError(
            "MTPLX_DSV41_F6_ENGRAM_LOOKAHEAD=1 is set but the F6 standalone install "
            "does not drive the Engram read lookahead. Enable it through the F16 "
            "pipeline with MTPLX_DSV41_F20_ENGRAM_LOOKAHEAD=1 (read in f16.install), "
            "which wires prefetch_rows at forward start and requires this F6 parallel "
            "gather already installed. Refusing to arm an env that installs nothing."
        )
    if site is not None and os.environ.get("MTPLX_DSV41_F6_INSTALL") != site:
        return None
    workers = int(os.environ.get("MTPLX_DSV41_F6_ENGRAM_WORKERS", "16"))
    state = install(model_or_caches, workers=workers)
    import atexit
    import json

    print("F6_ENGRAM_INSTALL " + json.dumps({"site": site, "workers": workers,
                                              "caches": len(state.caches)}), flush=True)
    atexit.register(lambda: print("F6_ENGRAM_STATS " + json.dumps(stats()), flush=True))
    return state


def stats() -> dict:
    """Aggregate parallel-read stats across all installs (read once after decode).

    Returns rows read in parallel, call counts, peak in-flight reads, and the
    installed-cache/worker counts. Safe to call from the main thread once
    generation has stopped."""
    agg = {
        "parallel_calls": 0,
        "calls_with_reads": 0,
        "reads_submitted": 0,
        "rows_read_parallel": 0,
        "max_inflight": 0,
        "lookahead_calls": 0,
        "lookahead_rows_submitted": 0,
        "installed_caches": 0,
        "workers": 0,
    }
    with _STATES_LOCK:
        states = list(_STATES)
    for st in states:
        with st._lock:
            for k in ("parallel_calls", "calls_with_reads", "reads_submitted",
                      "rows_read_parallel", "lookahead_calls", "lookahead_rows_submitted"):
                agg[k] += st.stats[k]
            agg["max_inflight"] = max(agg["max_inflight"], st.stats["max_inflight"])
            agg["installed_caches"] += len(st.caches)
            agg["workers"] = max(agg["workers"], st.workers)
    return agg
