"""F2b lane-private HOST ring: R records x 3 planes of page-aligned anonymous RAM.

The runtime keeps ``prefetch_slots == 0`` and never sees this ring; a decode MISS is
simply fulfilled from RAM instead of the SSD by the intercepted reader (see
``reader_intercept.py``). MLX-free: mmap + numpy + threading, so it is unit-testable on
CPU with real objects.

An entry is ONE plane, keyed by its ABSOLUTE file offset in ``experts.bin``
(``record.sidecar_offset + plane_offset``); no layer/expert identity is needed at the
reader. Per-plane states: QUEUED (enqueued, not read) -> READING (a worker is reading)
-> READY (bytes resident); a per-entry refcount is held while a demand copy is in
progress. FIFO recycling reuses the oldest buffer whose entry is neither READING nor
referenced. One lock guards the metadata; the byte copies happen OUTSIDE the lock.

Plane buffer size is the decode plane VIEW length (weights only; scales are resident) --
read at install from a live slot's ``component_view`` length, never hard-coded.
"""
from __future__ import annotations

import mmap
import threading

import numpy as np

# Reference weight-plane offsets inside one packed record (retained bind_reader):
# gate @0, up @6,266,880, down @12,533,760. The ring is keyed by ABSOLUTE offset, so
# these are only the default per-record plane offsets a caller adds to sidecar_offset.
PLANE_OFFSETS = (0, 6_266_880, 12_533_760)

_QUEUED, _READING, _READY = "queued", "reading", "ready"


class F2bCounters:
    """Plain ints, updated off the measured main thread (reader + worker threads) and a
    few on the main thread after the run; dumped ONCE after decode. Not per-token proof
    counters -- aggregate engagement, AGENTS.md."""

    __slots__ = (
        "planes_issued", "planes_completed", "planes_hits", "planes_waits",
        "planes_cancelled", "planes_wasted", "bytes_speculative",
        "records_full", "records_partial",
    )

    def __init__(self) -> None:
        for name in self.__slots__:
            setattr(self, name, 0)

    def as_dict(self) -> dict:
        return {name: int(getattr(self, name)) for name in self.__slots__}


class _Entry:
    __slots__ = ("offset", "buf_index", "length", "state", "event", "refcount",
                 "cancelled", "consumed")

    def __init__(self, offset: int, buf_index: int, length: int) -> None:
        self.offset = offset
        self.buf_index = buf_index
        self.length = length          # exact plane bytes (gate/up != down)
        self.state = _QUEUED
        self.event = threading.Event()
        self.refcount = 0
        self.cancelled = False
        self.consumed = False         # a demand read copied this plane (not wasted)


def _writable(view) -> np.ndarray:
    """A writable uint8 numpy view over a writable buffer (memoryview or mmap)."""
    arr = np.frombuffer(view, dtype=np.uint8)
    if not arr.flags.writeable:
        arr = arr.view()
        arr.flags.writeable = True  # the backing buffer is writable; frombuffer is conservative
    return arr


class HostRing:
    def __init__(self, *, records: int, planes: int = 3, plane_bytes: int,
                 counters: F2bCounters | None = None) -> None:
        # ``plane_bytes`` is the buffer size = the MAX plane length; each entry records
        # its own exact plane length (gate/up 6,266,880 vs down 5,160,960).
        self.records = int(records)
        self.planes = int(planes)
        self.plane_bytes = int(plane_bytes)
        self.capacity = self.records * self.planes
        if self.capacity < 1 or self.plane_bytes < 1:
            raise ValueError("ring needs >=1 plane buffer of >=1 byte")
        self.counters = counters or F2bCounters()
        # Page-aligned anonymous host buffers (mmap.mmap(-1, n)); keep refs alive.
        self._mmaps = [mmap.mmap(-1, self.plane_bytes) for _ in range(self.capacity)]
        self._np = [np.frombuffer(mm, dtype=np.uint8) for mm in self._mmaps]
        self._free = list(range(self.capacity))          # free buffer indices
        self._entries: dict[int, _Entry] = {}            # offset -> entry
        self._fifo: list[int] = []                        # offsets, oldest first
        self._lock = threading.Lock()

    # -- buffer accessors (memoryview of a plane buffer, sliced to n bytes) ----
    def buffer_view(self, buf_index: int, n: int) -> memoryview:
        return memoryview(self._mmaps[buf_index])[:n]

    # -- enqueue (speculative side, main thread) -------------------------------
    def enqueue(self, offset: int, length: int) -> bool:
        """Reserve a QUEUED plane entry for ``offset`` of ``length`` bytes (recycling the
        oldest free-able buffer). Idempotent: an offset already present is left as-is.
        Returns True iff a fresh entry was created (a plane a worker should read)."""
        offset = int(offset)
        length = int(length)
        if length > self.plane_bytes:
            raise ValueError(f"plane length {length} exceeds buffer size {self.plane_bytes}")
        with self._lock:
            if offset in self._entries:
                return False
            buf = self._acquire_buffer_locked()
            if buf is None:
                return False  # every buffer is READING or referenced; drop this plane
            entry = _Entry(offset, buf, length)
            self._entries[offset] = entry
            self._fifo.append(offset)
            return True

    def _acquire_buffer_locked(self) -> int | None:
        if self._free:
            return self._free.pop()
        # FIFO recycle: evict the oldest entry that is not READING and not referenced.
        for i, victim_off in enumerate(self._fifo):
            victim = self._entries.get(victim_off)
            if victim is None:
                continue
            if victim.state is _READING or victim.refcount > 0:
                continue
            # a committed (READY) victim recycled without a demand read consuming it.
            if victim.state is _READY and not victim.consumed:
                self.counters.planes_wasted += 1
            del self._entries[victim_off]
            del self._fifo[i]
            return victim.buf_index
        return None

    # -- worker side (speculative read) ----------------------------------------
    def begin_read(self, offset: int) -> memoryview | None:
        """A worker claims ``offset`` for reading. Returns the buffer view to read into,
        or None if the entry is gone/cancelled/not QUEUED (the worker skips it)."""
        offset = int(offset)
        with self._lock:
            entry = self._entries.get(offset)
            if entry is None or entry.cancelled or entry.state is not _QUEUED:
                return None
            entry.state = _READING
            return self.buffer_view(entry.buf_index, entry.length)

    def end_read(self, offset: int, *, ok: bool) -> None:
        """A worker settles a claimed read. ok -> READY; failure -> evict (a later
        demand read falls back to a normal pread)."""
        offset = int(offset)
        with self._lock:
            entry = self._entries.get(offset)
            if entry is None or entry.state is not _READING:
                return
            if ok:
                entry.state = _READY
                self.counters.planes_completed += 1
                self.counters.bytes_speculative += int(entry.length)
            else:
                # evict so the buffer is reusable and demand re-reads from SSD.
                self._entries.pop(offset, None)
                self._free.append(entry.buf_index)
                try:
                    self._fifo.remove(offset)
                except ValueError:
                    pass
            entry.event.set()

    # -- reader side (demand path; the lane's actual work) ---------------------
    def try_serve(self, offset: int, dest_view: memoryview, *, wait_timeout: float = 5.0) -> bool:
        """Fulfil a demand plane read from RAM if possible. Returns True iff the dest was
        filled from the ring (the caller then issues NO pread); False -> caller preads.

          READY  -> copy ring->dest (np.copyto, GIL released), count a hit.
          READING-> wait for completion, then copy, count a wait.
          QUEUED -> mark cancelled (worker skips it), count a cancel, return False.
          absent -> return False.
        """
        offset = int(offset)
        with self._lock:
            entry = self._entries.get(offset)
            if entry is None:
                return False
            if entry.state is _QUEUED:
                entry.cancelled = True
                self.counters.planes_cancelled += 1
                return False
            if entry.state is _READY:
                entry.refcount += 1
                src = self._np[entry.buf_index]
                length = entry.length
                counter_attr = "planes_hits"
            else:  # _READING
                event = entry.event
                counter_attr = "planes_waits"
        if counter_attr == "planes_hits":
            self._copy_and_release(entry, dest_view, src, length, counter_attr)
            return True
        # READING: wait outside the lock, then re-check and copy.
        event.wait(timeout=wait_timeout)
        with self._lock:
            entry = self._entries.get(offset)
            if entry is None or entry.state is not _READY:
                return False  # failed/evicted while we waited -> pread
            entry.refcount += 1
            src = self._np[entry.buf_index]
            length = entry.length
        self._copy_and_release(entry, dest_view, src, length, "planes_waits")
        return True

    def _copy_and_release(self, entry, dest_view, src, length, counter_attr) -> None:
        try:
            dst = _writable(dest_view)
            n = min(dst.shape[0], int(length))
            np.copyto(dst[:n], src[:n])  # releases the GIL; no memoryview slice assign
            setattr(self.counters, counter_attr, getattr(self.counters, counter_attr) + 1)
        finally:
            with self._lock:
                entry.refcount -= 1
                entry.consumed = True   # a demand read consumed this plane (not wasted)

    # -- introspection ---------------------------------------------------------
    def has(self, offset: int) -> bool:
        with self._lock:
            return int(offset) in self._entries

    def state_of(self, offset: int) -> str | None:
        with self._lock:
            entry = self._entries.get(int(offset))
            return None if entry is None else entry.state
