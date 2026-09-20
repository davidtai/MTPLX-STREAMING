"""F2b lane-private HOST ring: R records x 3 weight planes of page-aligned anonymous RAM.

The runtime keeps ``prefetch_slots == 0`` and never sees this ring; a decode MISS is
simply fulfilled from RAM instead of the SSD by the intercepted reader (see
``reader_intercept.py``). MLX-free: mmap + numpy + threading, unit-testable on CPU.

Record layout (retained mxfp4): each of the three projection slots is 6,266,880 B =
weight 5,898,240 + scale 368,640, so the packed source record is 3 x 6,266,880 =
18,800,640 B (spec.expert_record_bytes). The DECODE reader fills only the three
5,898,240-byte WEIGHT views (scales are resident), at plane offsets 0 / 6,266,880 /
12,533,760; those three weight planes total 3 x 5,898,240 = 17,694,720 B (the decode
weight record, ``rt._representative_record_bytes`` after growth). All three weight planes
are the SAME length (5,898,240) -- install asserts this and that 3 x length equals the
runtime's decode record bytes.

An entry is ONE plane, keyed by its ABSOLUTE ``experts.bin`` offset
(``record.sidecar_offset + plane_offset``); no layer/expert identity at the reader.
States QUEUED -> READING -> READY + a per-entry refcount held while a demand copy runs.
FIFO recycling reuses the oldest buffer whose entry is neither READING nor referenced.
One lock guards the metadata; the byte copies (``np.copyto`` on ``np.frombuffer`` views,
GIL released) happen outside it.
"""
from __future__ import annotations

import mmap
import threading

import numpy as np

# Weight-plane offsets inside one packed record (retained bind_reader): gate @0,
# up @6,266,880, down @12,533,760. Keyed by ABSOLUTE offset (sidecar_offset + these).
PLANE_OFFSETS = (0, 6_266_880, 12_533_760)
# The three decode weight planes are equal-length; their sum is the decode weight record.
WEIGHT_PLANE_BYTES = 5_898_240
WEIGHT_RECORD_BYTES = 3 * WEIGHT_PLANE_BYTES          # 17,694,720

_QUEUED, _READING, _READY = "queued", "reading", "ready"


class F2bCounters:
    """Plain ints, updated off the measured main thread (reader + worker/coordinator
    threads) and a few on the main thread after the run; dumped ONCE after decode.
    Aggregate engagement, not per-token proof counters (AGENTS.md)."""

    __slots__ = (
        "planes_issued", "planes_completed", "planes_hits", "planes_waits",
        "planes_cancelled", "planes_wasted", "planes_length_mismatch",
        "planes_epoch_skipped", "coordinator_batches", "ready_reading_hwm",
        "bytes_speculative", "records_full", "records_partial",
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
        self.length = length
        self.state = _QUEUED
        self.event = threading.Event()
        self.refcount = 0
        self.cancelled = False
        self.consumed = False


def _writable(view) -> np.ndarray:
    arr = np.frombuffer(view, dtype=np.uint8)
    if not arr.flags.writeable:
        arr = arr.view()
        arr.flags.writeable = True
    return arr


class HostRing:
    def __init__(self, *, records: int, planes: int = 3, plane_bytes: int,
                 counters: F2bCounters | None = None, wire: bool = False) -> None:
        self.records = int(records)
        self.planes = int(planes)
        self.plane_bytes = int(plane_bytes)          # buffer size == the (equal) plane length
        self.capacity = self.records * self.planes
        if self.capacity < 1 or self.plane_bytes < 1:
            raise ValueError("ring needs >=1 plane buffer of >=1 byte")
        self.counters = counters or F2bCounters()
        self._mmaps = [mmap.mmap(-1, self.plane_bytes) for _ in range(self.capacity)]
        self._np = [np.frombuffer(mm, dtype=np.uint8) for mm in self._mmaps]
        # Optional: wire (mlock) the plane buffers ONCE at construction so every speculative
        # F_NOCACHE read lands in already-wired pages (the kernel otherwise wires/unwires the
        # destination per I/O while the main thread is submitting GPU work). Same bytes, same
        # accounting (the ring is already charged to admission); failure is loud.
        self.wired_bytes = 0
        if wire:
            self.wired_bytes = self._wire_buffers()
        self._free = list(range(self.capacity))
        self._entries: dict[int, _Entry] = {}
        self._fifo: list[int] = []
        self._active = 0                              # count of READING + READY entries
        self._lock = threading.Lock()

    def _wire_buffers(self) -> int:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.mlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        libc.mlock.restype = ctypes.c_int
        total = 0
        for arr in self._np:
            arr[:] = 0                                   # fault the pages in before wiring
            if libc.mlock(ctypes.c_void_p(arr.ctypes.data), ctypes.c_size_t(arr.nbytes)) != 0:
                err = ctypes.get_errno()
                raise OSError(err, f"mlock of an F2b ring buffer failed after {total} bytes")
            total += int(arr.nbytes)
        return total

    def buffer_view(self, buf_index: int, n: int) -> memoryview:
        return memoryview(self._mmaps[buf_index])[:n]

    # -- enqueue (coordinator thread) ------------------------------------------
    def enqueue(self, offset: int, length: int) -> bool:
        offset = int(offset)
        length = int(length)
        if length > self.plane_bytes:
            raise ValueError(f"plane length {length} exceeds buffer size {self.plane_bytes}")
        with self._lock:
            if offset in self._entries:
                return False
            buf = self._acquire_buffer_locked()
            if buf is None:
                return False
            self._entries[offset] = _Entry(offset, buf, length)
            self._fifo.append(offset)
            return True

    def _acquire_buffer_locked(self) -> int | None:
        if self._free:
            return self._free.pop()
        for i, victim_off in enumerate(self._fifo):
            victim = self._entries.get(victim_off)
            if victim is None:
                continue
            if victim.state is _READING or victim.refcount > 0:
                continue
            if victim.state is _READY:
                self._active -= 1
                if not victim.consumed:
                    self.counters.planes_wasted += 1
            del self._entries[victim_off]
            del self._fifo[i]
            return victim.buf_index
        return None

    # -- discard a QUEUED entry (fix 3: window-stopped / cancelled must not linger) --
    def discard(self, offset: int) -> bool:
        """Remove a QUEUED, unreferenced entry and free its buffer. Returns True if
        discarded. READING/READY/referenced entries are left untouched."""
        offset = int(offset)
        with self._lock:
            entry = self._entries.get(offset)
            if entry is None or entry.state is not _QUEUED or entry.refcount > 0:
                return False
            self._entries.pop(offset, None)
            try:
                self._fifo.remove(offset)
            except ValueError:
                pass
            self._free.append(entry.buf_index)
            return True

    # -- worker side -----------------------------------------------------------
    def begin_read(self, offset: int) -> memoryview | None:
        offset = int(offset)
        with self._lock:
            entry = self._entries.get(offset)
            if entry is None or entry.cancelled or entry.state is not _QUEUED:
                return None
            entry.state = _READING
            self._active += 1
            if self._active > self.counters.ready_reading_hwm:
                self.counters.ready_reading_hwm = self._active
            return self.buffer_view(entry.buf_index, entry.length)

    def end_read(self, offset: int, *, ok: bool) -> None:
        offset = int(offset)
        with self._lock:
            entry = self._entries.get(offset)
            if entry is None or entry.state is not _READING:
                return
            if ok:
                entry.state = _READY                  # still active (READING->READY)
                self.counters.planes_completed += 1
                self.counters.bytes_speculative += int(entry.length)
            else:
                self._active -= 1
                self._entries.pop(offset, None)
                self._free.append(entry.buf_index)
                try:
                    self._fifo.remove(offset)
                except ValueError:
                    pass
            entry.event.set()

    # -- reader side (demand path) ---------------------------------------------
    def try_serve(self, offset: int, dest_view, *, wait_timeout: float = 5.0) -> bool:
        offset = int(offset)
        dest_len = len(dest_view)
        with self._lock:
            entry = self._entries.get(offset)
            if entry is None:
                return False
            if dest_len != entry.length:
                # A demand view whose length disagrees with the cached plane would leave
                # STALE bytes in the slot; refuse and pread (fix 2).
                self.counters.planes_length_mismatch += 1
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
        event.wait(timeout=wait_timeout)
        with self._lock:
            entry = self._entries.get(offset)
            if entry is None or entry.state is not _READY or dest_len != entry.length:
                if entry is not None and dest_len != entry.length:
                    self.counters.planes_length_mismatch += 1
                return False
            entry.refcount += 1
            src = self._np[entry.buf_index]
            length = entry.length
        self._copy_and_release(entry, dest_view, src, length, "planes_waits")
        return True

    def _copy_and_release(self, entry, dest_view, src, length, counter_attr) -> None:
        try:
            dst = _writable(dest_view)
            np.copyto(dst[:length], src[:length])     # exactly `length` bytes; GIL released
            setattr(self.counters, counter_attr, getattr(self.counters, counter_attr) + 1)
        finally:
            with self._lock:
                entry.refcount -= 1
                entry.consumed = True

    # -- introspection ---------------------------------------------------------
    def has(self, offset: int) -> bool:
        with self._lock:
            return int(offset) in self._entries

    def state_of(self, offset: int) -> str | None:
        with self._lock:
            entry = self._entries.get(int(offset))
            return None if entry is None else entry.state
