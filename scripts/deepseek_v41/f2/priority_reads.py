"""N-worker demand-priority read queue for the F2 prefetch lane's isolated reader.

MLX-free (pure threads + a priority queue), so it is unit-testable on CPU.

This is a drop-in replacement for the reader's native ``_fanout_executor`` (an
unbounded-ordering ``ThreadPoolExecutor``) installed ONLY on the candidate arm.
When the lane is installed, the sole submitter is the bound plane-split reader
(``plane_lane_prefetch.bind_priority_reader``), which always calls
``submit(read, job, priority=...)``: demand plane reads at priority 0, speculative
(ring) plane reads at priority 1 (via the ``IgnoreGUPublication`` witness). So this
queue only ever sees keyword ``priority`` calls -- the native fanout-splitting
paths (``expert_io._scatter_record_fanned`` / ``_read_range_into``) are bypassed by
the reader rebind, so a positionless ``submit(fn, arg)`` is never issued here.

Three properties this reader is responsible for, and where each is guaranteed:

  1. **>= native parallelism (fix #1).** ``workers`` is a constructor argument;
     ``plane_lane_prefetch.install`` passes the native pool's worker count read at
     install (``reader._fanout_pool_workers``, mtplx/expert_io.py:419-429; fanout 4
     -> 15). Codex's screen hard-coded 4 workers, so its candidate arms read with
     LESS parallelism than their controls -- the fix restores >= native.

  2. **Priority controls QUEUED work only; an active read finishes.** ``work``
     pulls the lowest ``(priority, sequence)`` first, so a demand plane read jumps
     ahead of every queued speculative one, but a read already handed to ``fn`` runs
     to completion (the reader keeps writer joins + memoryview ownership through the
     read -- plane_lane_prefetch ``bind_priority_reader``). Sentinels drain LAST
     (``_SHUTDOWN_PRIORITY`` > every real priority) so shutdown never strands a
     queued read.

  3. **A speculative record no longer wanted is skipped; a demanded in-flight one
     is awaited (never re-read).** These are RUNTIME guarantees, not re-implemented
     here (correct-by-design; no redundant reader-level cancellation):
       * skip-when-unwanted: a speculative record whose ring assignment was recycled
         is skipped BEFORE any read is submitted to this queue --
         ``ExpertStreamingRuntime._run_speculative_load`` re-checks the ticket under
         the layer lock and returns early if the assignment recycled
         (mtplx/expert_runtime.py:5297-5303); a settled-but-stale read's commit is
         then refused (``_apply_prefetch_completions`` -> ``bank.commit_prefetch``
         returns False, mtplx/expert_streaming.py:444-445). A ``Future`` cancelled
         before a worker starts it is also skipped here via
         ``set_running_or_notify_cancel``.
       * awaited-not-re-read: a demanded expert whose speculative read is still in
         flight is awaited and committed as a hit by
         ``ExpertStreamingRuntime._reconcile_prefetch_for_route`` (the await at
         mtplx/expert_runtime.py:5044 and the in-place commit at :5052, driven by the
         inflight index registered at :5238), which runs inside ``begin_split_route``
         (mtplx/expert_runtime.py:4168) BEFORE the miss set is planned -- so the true
         route reads the already-issued bytes instead of issuing a duplicate.
"""
from __future__ import annotations

from concurrent.futures import Future
from itertools import count
from queue import PriorityQueue
import threading

# Real read priorities are 0 (demand) and 1 (speculative). Shutdown sentinels use
# a strictly-higher priority so every queued read drains before a worker exits.
_DEMAND_PRIORITY = 0
_SPECULATIVE_PRIORITY = 1
_SHUTDOWN_PRIORITY = 2


def native_worker_count(reader) -> int:
    """The native fanout pool's worker count (the install-time default for N).

    ``reader._fanout_pool_workers`` is fixed at reader construction from the read
    fanout (mtplx/expert_io.py:419-429). Floored at 1 so a fanout-1 reader (which
    has no fanout pool) still yields a usable single-worker default.
    """
    return max(1, int(getattr(reader, "_fanout_pool_workers", 0) or 0))


class PriorityReads:
    def __init__(self, workers: int) -> None:
        workers = int(workers)
        if workers < 1:
            raise ValueError("PriorityReads requires >= 1 worker")
        self.workers = workers
        self.queue: PriorityQueue = PriorityQueue()
        self.sequence = count()
        self.lock = threading.Lock()
        self.closed = False
        self.threads = tuple(
            threading.Thread(
                target=self.work, name=f"f2-packed-read-{i}", daemon=True
            )
            for i in range(workers)
        )
        for thread in self.threads:
            thread.start()

    def submit(self, fn, job, *, priority) -> Future:
        future: Future = Future()
        with self.lock:
            if self.closed:
                raise RuntimeError("read queue is closed")
            # (priority, monotonic sequence) is a total order, so heap comparison
            # never reaches the payload tuple (which holds unorderable objects).
            self.queue.put((int(priority), next(self.sequence), (future, fn, job)))
        return future

    def work(self) -> None:
        while True:
            _priority, _sequence, task = self.queue.get()
            if task is None:
                self.queue.task_done()
                return
            future, fn, job = task
            try:
                # A Future cancelled before a worker claims it is skipped here; one
                # already running cannot be cancelled and finishes normally.
                if future.set_running_or_notify_cancel():
                    try:
                        result = fn(job)
                    except BaseException as error:  # noqa: BLE001 - propagate to caller
                        future.set_exception(error)
                    else:
                        future.set_result(result)
                        result = None
            finally:
                future = fn = job = task = None
                self.queue.task_done()

    def shutdown(self, wait: bool = True) -> None:
        with self.lock:
            if not self.closed:
                self.closed = True
                for _ in self.threads:
                    self.queue.put(
                        (_SHUTDOWN_PRIORITY, next(self.sequence), None)
                    )
        if wait:
            for thread in self.threads:
                thread.join()
