"""F2b speculative reader: a small private thread pool that fills the host ring.

Workers call the SAME primitive the retained lane uses -- ``reader._readv_range_into
('experts.bin', offset, (view,))`` -- into ring plane buffers, in gate/up/down order per
record. A per-target "demand imminent" flag stops workers from STARTING new planes for a
target once its own forward begins (in-flight planes finish; unstarted ones are dropped,
not left queued). MLX-free.
"""
from __future__ import annotations

import queue
import threading


class SpeculativePool:
    def __init__(self, reader, ring, *, workers: int = 3):
        self.reader = reader
        self.ring = ring
        self.counters = ring.counters
        self._q: queue.Queue = queue.Queue()
        self._closed = False
        self._demand_imminent: set[int] = set()          # target layers whose forward began
        self._lock = threading.Lock()
        # per-record plane bookkeeping for records_full / records_partial (worker threads).
        self._rec_total: dict[int, int] = {}
        self._rec_ok: dict[int, int] = {}
        self._rec_seen: dict[int, int] = {}
        self.workers = int(workers)
        self._threads = tuple(
            threading.Thread(target=self._work, name=f"f2b-spec-{i}", daemon=True)
            for i in range(self.workers)
        )
        for t in self._threads:
            t.start()

    def note_demand_imminent(self, target: int) -> None:
        """Called by the target layer's wrapper right after its mx.eval(indices, merged):
        stop starting new speculative planes for this target."""
        with self._lock:
            self._demand_imminent.add(int(target))

    def enqueue_record(self, target: int, sidecar_offset: int, planes) -> int:
        """Enqueue a predicted record's planes in gate/up/down order. ``planes`` is a
        sequence of (plane_offset_delta, length). Returns the number of planes newly
        queued into the ring."""
        target = int(target)
        rec_key = int(sidecar_offset)
        jobs = [(int(sidecar_offset) + int(delta), int(length)) for delta, length in planes]
        with self._lock:
            self._rec_total[rec_key] = len(jobs)
            self._rec_ok[rec_key] = 0
            self._rec_seen[rec_key] = 0
        issued = 0
        for offset, length in jobs:                # gate, up, down
            if self.ring.enqueue(offset, length):
                self.counters.planes_issued += 1
                self._q.put((target, rec_key, offset))
                issued += 1
        return issued

    def _work(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            target, rec_key, offset = item
            try:
                with self._lock:
                    stop = target in self._demand_imminent
                if stop:
                    # window-stop: do not START this plane; leave the ring entry QUEUED
                    # (a demand read will cancel + pread it, or it recycles unread).
                    self._settle_record(rec_key, ok=False, started=False)
                    continue
                view = self.ring.begin_read(offset)
                if view is None:
                    self._settle_record(rec_key, ok=False, started=False)  # cancelled/gone
                    continue
                ok = True
                try:
                    self.reader._readv_range_into(
                        "experts.bin", offset, (view,),
                        cancel_event=None, deadline_ns=None, pipeline_phase=None,
                    )
                except BaseException:
                    ok = False
                self.ring.end_read(offset, ok=ok)
                self._settle_record(rec_key, ok=ok, started=True)
            finally:
                self._q.task_done()

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
        """Wait for the queue to empty (in-flight reads finish). Off the measured path."""
        self._q.join()

    def shutdown(self, wait: bool = True) -> None:
        if not self._closed:
            self._closed = True
            for _ in self._threads:
                self._q.put(None)
        if wait:
            for t in self._threads:
                t.join()
