"""One construction-bound reader job per part, executed by its miss worker.

The native slot transaction submits under its completion-error lock, then calls
Future.result outside that lock. Delay physical reading until that existing
wait. The native fill, admission, publication, pin and failure paths stay intact.
Only the fixed plane lane uses this executor: no timed external future consumers.
"""
from concurrent.futures import Future
import hashlib
import inspect
from pathlib import Path


class WorkerRead(Future):
    def __init__(self, fn, args, kwargs):
        super().__init__()
        self.work = fn, args, kwargs

    def result(self, timeout=None):
        work, self.work = self.work, None
        if work is not None and self.set_running_or_notify_cancel():
            fn, args, kwargs = work
            try:
                self.set_result(fn(*args, **kwargs))
            except BaseException as error:
                self.set_exception(error)
        return super().result(timeout=timeout)


class MissWorkerReader:
    def __init__(self, predecessor):
        self.predecessor = predecessor

    def submit(self, fn, *args, **kwargs):
        return WorkerRead(fn, args, kwargs)

    def shutdown(self, *args, **kwargs):
        return self.predecessor.shutdown(*args, **kwargs)


def install(runtime, *, source_sha256):
    slots = runtime.slots
    source = Path(inspect.getfile(type(slots))).read_bytes()
    if (hashlib.sha256(source).hexdigest() != source_sha256
            or not slots._batch_decode_reads or not slots.prefer_sidecar
            or slots.verify_hashes or slots._reader_pool_telemetry is not None
            or runtime._pipeline_ledger is not None
            or runtime.config.decode_miss_records_per_part != 3
            or runtime.config.prefetch_slots
            or type(slots._executor).__name__ != 'ReaderExecutor'
            or runtime.reader.read_component_records_into.__module__ != 'plane_lane'):
        raise RuntimeError('miss-worker reader requires the pinned native plane lane')
    # The installed plane reader already scatters arbitrary record positions.
    # Owned loads vary per route; all their storage/file invariants were proved
    # by the plane installer. An all-hit route still creates no reader job.
    slots._can_batch_component_sidecar = lambda plan, owned: bool(owned)
    slots._executor = MissWorkerReader(slots._executor)
