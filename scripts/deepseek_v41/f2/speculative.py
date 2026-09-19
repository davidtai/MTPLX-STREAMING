"""F2b speculative reader: a coordinator thread + a small worker pool that fill the ring.

The measured main thread does almost nothing: a source layer's wrapper copies the
evaluated prediction and hands ONE item to the coordinator (``submit_prediction``). The
COORDINATOR thread does resident filtering + ranking + offset mapping + enqueue (injected
``plan_fn``); WORKER threads read planes via the SAME primitive the retained lane uses
(``reader._readv_range_into('experts.bin', offset, (view,))``), gate/up/down order.

Window-stop is a per-target EPOCH (fix 1): ``note_demand_imminent(T)`` increments
``epoch[T]``; each queued plane is stamped with the epoch at enqueue; a worker starts a
plane only if its stamp still equals ``epoch[T]``. A stale (window-stopped) or cancelled
plane is DISCARDED from the ring so it does not linger as a phantom hit (fix 3). MLX-free.
"""
from __future__ import annotations

import os
import queue
import threading


class SpeculativePool:
    def __init__(self, reader, ring, *, plane_specs, plan_fn, workers: int = 3,
                 direct_fd: int | None = None):
        self.reader = reader
        # Read primitive bound ONCE (no per-plane branch). ``direct_fd`` = a private
        # F_NOCACHE descriptor on experts.bin: one bare ``os.preadv`` per plane. The retained
        # ``reader._readv_range_into`` costs two passes through the reader's shared Condition
        # (fd lease) plus locked metrics per call -- tens of microseconds of GIL-holding Python
        # per plane that contend with the demand readers AND with the measured main thread
        # while it builds the next layer's graph (the probe charged +0.26 ms/call to that gap).
        self._fd = direct_fd
        self._read_plane = self._read_direct if direct_fd is not None else self._read_via_reader
        self.ring = ring
        self.counters = ring.counters
        self.plane_specs = tuple((int(d), int(n)) for d, n in plane_specs)  # (offset_delta, length)
        self.plan_fn = plan_fn                          # (target, scores_np) -> [sidecar_offset]
        self.workers = int(workers)
        self._coord_q: queue.Queue = queue.Queue()
        self._work_q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._epoch: dict[int, int] = {}
        self._rec_total: dict[int, int] = {}
        self._rec_ok: dict[int, int] = {}
        self._rec_seen: dict[int, int] = {}
        self._closed = False
        self._coordinator = threading.Thread(target=self._coordinate, name="f2b-coord", daemon=True)
        self._coordinator.start()
        self._threads = tuple(
            threading.Thread(target=self._work, name=f"f2b-spec-{i}", daemon=True)
            for i in range(self.workers)
        )
        for t in self._threads:
            t.start()

    # -- main-thread API (cheap) ----------------------------------------------
    def note_demand_imminent(self, target: int) -> None:
        with self._lock:
            self._epoch[int(target)] = self._epoch.get(int(target), 0) + 1

    def current_epoch(self, target: int) -> int:
        with self._lock:
            return self._epoch.get(int(target), 0)

    def submit_prediction(self, target: int, epoch: int, scores) -> None:
        """Hand a target's evaluated prediction to the coordinator (one queue.put)."""
        self._coord_q.put((int(target), int(epoch), scores))

    # -- coordinator thread (ranking + enqueue, off the measured path) --------
    def _coordinate(self) -> None:
        while True:
            item = self._coord_q.get()
            if item is None:
                self._coord_q.task_done()
                return
            target, epoch, scores = item
            try:
                offsets = self.plan_fn(target, scores)
                self.counters.coordinator_batches += 1
                for sidecar_offset in offsets:
                    self._enqueue_record(target, int(sidecar_offset), epoch)
            finally:
                self._coord_q.task_done()

    def _enqueue_record(self, target: int, sidecar_offset: int, epoch: int) -> None:
        rec_key = int(sidecar_offset)
        jobs = [(sidecar_offset + d, n) for d, n in self.plane_specs]
        with self._lock:
            self._rec_total[rec_key] = len(jobs)
            self._rec_ok[rec_key] = 0
            self._rec_seen[rec_key] = 0
        for offset, length in jobs:                     # gate, up, down
            if self.ring.enqueue(offset, length):
                self.counters.planes_issued += 1
                self._work_q.put((target, rec_key, offset, epoch))

    # -- worker threads --------------------------------------------------------
    def _work(self) -> None:
        while True:
            item = self._work_q.get()
            if item is None:
                self._work_q.task_done()
                return
            target, rec_key, offset, epoch = item
            try:
                if epoch != self.current_epoch(target):
                    # window-stopped: never start this plane; do not leave it QUEUED.
                    self.ring.discard(offset)
                    self.counters.planes_epoch_skipped += 1
                    self._settle_record(rec_key, ok=False, started=False)
                    continue
                view = self.ring.begin_read(offset)
                if view is None:
                    self.ring.discard(offset)           # cancelled/gone -> free the slot
                    self._settle_record(rec_key, ok=False, started=False)
                    continue
                ok = True
                try:
                    self._read_plane(offset, view)
                except BaseException:
                    ok = False
                self.ring.end_read(offset, ok=ok)
                self._settle_record(rec_key, ok=ok, started=True)
            finally:
                self._work_q.task_done()

    def _read_via_reader(self, offset: int, view) -> None:
        self.reader._readv_range_into(
            "experts.bin", offset, (view,),
            cancel_event=None, deadline_ns=None, pipeline_phase=None,
        )

    def _read_direct(self, offset: int, view) -> None:
        fd = self._fd
        total = len(view)
        done = os.preadv(fd, (view,), offset)
        while done < total:                                # short read: resume the remainder
            if done <= 0:
                raise OSError("short speculative plane read")
            got = os.preadv(fd, (view[done:],), offset + done)
            if got <= 0:
                raise OSError("short speculative plane read")
            done += got

    def _settle_record(self, rec_key: int, *, ok: bool, started: bool) -> None:
        with self._lock:
            self._rec_seen[rec_key] = self._rec_seen.get(rec_key, 0) + 1
            if started and ok:
                self._rec_ok[rec_key] = self._rec_ok.get(rec_key, 0) + 1
            if self._rec_seen[rec_key] >= self._rec_total.get(rec_key, 0):
                got = self._rec_ok.get(rec_key, 0)
                total = self._rec_total.get(rec_key, 0)
                if total and got >= total:
                    self.counters.records_full += 1
                elif got > 0:
                    self.counters.records_partial += 1
                for d in (self._rec_total, self._rec_ok, self._rec_seen):
                    d.pop(rec_key, None)

    def drain(self) -> None:
        """Wait for coordinator + workers to quiesce (off the measured path)."""
        self._coord_q.join()
        self._work_q.join()

    def shutdown(self, wait: bool = True) -> None:
        if not self._closed:
            self._closed = True
            self._coord_q.put(None)
            for _ in self._threads:
                self._work_q.put(None)
        if wait:
            self._coordinator.join()
            for t in self._threads:
                t.join()
