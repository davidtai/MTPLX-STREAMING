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
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# NB: importing NGramRowCache pulls in ``mlx.core`` (module-level in
# ngram_row_cache). Callers that must stay on CPU set the default device before
# importing this module; nothing here touches Metal.
from mtplx.ngram_row_cache import FileRowReader, NGramRowCache, _contiguous_runs

__all__ = ["install", "install_from_env", "stats", "ParallelReadState"]

_INSTALL_ATTR = "_f6_parallel"
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
        self._lock = threading.Lock()      # guards the cross-thread counters below
        self._inflight = 0
        self.caches: "list[NGramRowCache]" = []
        self.stats = {
            "parallel_calls": 0,       # patched gather_bytes invocations
            "calls_with_reads": 0,     # of those, calls that issued >=1 concurrent read
            "reads_submitted": 0,      # sub-run reads dispatched to the pool
            "rows_read_parallel": 0,   # rows fetched through the pool
            "max_inflight": 0,         # peak concurrently-executing reads observed
        }

    # -- the concurrent read task -------------------------------------------
    def _read(self, reader, start, count):
        with self._lock:
            self._inflight += 1
            if self._inflight > self.stats["max_inflight"]:
                self.stats["max_inflight"] = self._inflight
        try:
            # read_run allocates its own bytearray/memoryview per call and uses
            # positional os.preadv on the shared fd -> thread-safe, no shared
            # per-call buffer (verified against FileRowReader.read_run).
            return reader.read_run(start, count)
        finally:
            with self._lock:
                self._inflight -= 1

    def fetch(self, reader, requests):
        """Read every ``(start, count)`` run concurrently; return a list of
        ``bytes`` aligned to ``requests``.

        Every future is drained before returning; if any read raised, the first
        error is re-raised AFTER all settle, so the caller has not yet mutated
        the cache (no partial insertion, no dangling in-flight read)."""
        with self._lock:
            self.stats["reads_submitted"] += len(requests)
        futures = [self.pool.submit(self._read, reader, s, c) for (s, c) in requests]
        results: list = [None] * len(futures)
        error: "BaseException | None" = None
        for i, fut in enumerate(futures):
            try:
                results[i] = fut.result()
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

    def shutdown(self) -> None:
        if self._own_pool:
            self.pool.shutdown(wait=True)


def _parallel_gather_bytes(self, row_ids) -> np.ndarray:
    """Instance-bound replacement for ``NGramRowCache.gather_bytes``.

    Mirrors the stock method exactly except that the missing sub-run reads are
    issued concurrently up front (``state.fetch``) instead of inline in the
    population loop. All cache-state mutation runs on the calling (main) thread
    in the original order.
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
        # Sub-run read requests, in the EXACT order the serial loop would issue
        # them (per run, split by slot_count). run count == stock read count.
        requests: "list[tuple[int, int]]" = []
        for start, count in runs:
            off = 0
            while off < count:
                n = min(count - off, self.slot_count)
                requests.append((start + off, n))
                off += n
        # All reads happen here, before any mutation. Raises on the first error.
        datas = state.fetch(self.reader, requests)
        had_reads = True
        parallel_rows = sum(c for (_s, c) in requests)
        # Replay the stock population loop verbatim, sourcing bytes from `datas`.
        ri = 0
        for start, count in runs:
            off = 0
            while off < count:
                n = min(count - off, self.slot_count)
                data = datas[ri]
                ri += 1
                self.stats["reads"] += 1
                self.stats["rows_read"] += n
                self.stats["misses"] += n
                block = np.frombuffer(data, dtype=np.uint8).reshape(n, self.row_bytes)
                for k in range(n):
                    r = start + off + k
                    slot = self._alloc_slot()
                    self._arena[slot] = block[k]
                    self._lru[r] = slot           # newest
                    out[positions[r]] = block[k]
                off += n
    self.stats["gathers"] += 1
    state.note_call(had_reads=had_reads, rows=parallel_rows)
    return out


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
    ``MTPLX_DSV41_F6_ENGRAM_LOOKAHEAD=1`` is refused loudly: the lookahead lane is
    designed but not installed in this build (needs a GPU run to establish the
    advance->hook ordering invariant), and running with an armed-but-absent lane
    is unsafe (AGENTS.md: fail once, clearly, before measured generation).
    """
    if os.environ.get("MTPLX_DSV41_F6_ENGRAM_PARALLEL") != "1":
        return None
    if os.environ.get("MTPLX_DSV41_F6_ENGRAM_LOOKAHEAD") == "1":
        raise RuntimeError(
            "MTPLX_DSV41_F6_ENGRAM_LOOKAHEAD=1 is set but the F6 Engram lookahead "
            "lane is not installed in this build (design filed in the F6 report; "
            "it requires a GPU run to establish the advance->hook ordering "
            "invariant). Refusing to measure with an armed-but-absent lane."
        )
    if site is not None and os.environ.get("MTPLX_DSV41_F6_INSTALL") != site:
        return None
    workers = int(os.environ.get("MTPLX_DSV41_F6_ENGRAM_WORKERS", "16"))
    return install(model_or_caches, workers=workers)


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
        "installed_caches": 0,
        "workers": 0,
    }
    with _STATES_LOCK:
        states = list(_STATES)
    for st in states:
        with st._lock:
            for k in ("parallel_calls", "calls_with_reads", "reads_submitted", "rows_read_parallel"):
                agg[k] += st.stats[k]
            agg["max_inflight"] = max(agg["max_inflight"], st.stats["max_inflight"])
            agg["installed_caches"] += len(st.caches)
            agg["workers"] = max(agg["workers"], st.workers)
    return agg
