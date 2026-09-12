"""W109 (T1 / issue I1): DSpark verify SSD queue-depth levers.

Covers ``MTPLX_DSV41_VERIFY_IO_FANOUT`` and ``MTPLX_DSV41_VERIFY_UNION_READ`` end
to end on the runtime split-route path with a synthetic component-bank artifact
(reused from ``test_expert_overlap_split``): env default OFF, union batching into
one submission, fanout into N concurrent groups, a fake recording reader proving
N-way concurrency and one-call union batching, the four engagement counters, and
byte-identity of the emitted switch output between fanout 1 and fanout 8.

CPU-pinned (MLX defaults to Metal; these tests pin CPU), single file per process,
< 1.5 GB RSS.
"""

from __future__ import annotations

import pathlib
import sys
import threading
import time

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

# Reuse the synthetic-bank harness (strictly adjacent sidecar records, component
# slots) from the overlap-split suite.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_expert_overlap_split import (  # noqa: E402
    _open_overlap_runtime,
    _overlap_artifact,
    _drain_split_route,
)

from mtplx.models.expert_mlx import HotExpertSwitchGLU  # noqa: E402


FANOUT_ENV = "MTPLX_DSV41_VERIFY_IO_FANOUT"
UNION_ENV = "MTPLX_DSV41_VERIFY_UNION_READ"


def _metrics(runtime) -> dict:
    return runtime.slots.metrics.as_dict()


class _RecordingReader:
    """Wrap the runtime reader to record call order + observed concurrency.

    Every ``read_record_into`` / ``read_component_records_into`` call bumps a
    shared active gauge (peak recorded) and appends ``(method, num_records)`` to an
    ordered list, then sleeps ``delay_s`` to force any concurrent reads to overlap.
    """

    def __init__(self, runtime, delay_s: float) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls: list[tuple[str, int]] = []
        reader = runtime.reader
        orig_single = reader.read_record_into
        orig_batch = reader.read_component_records_into

        def single(manifest, record, destination, **kwargs):
            self._enter("single", 1)
            try:
                time.sleep(delay_s)
                return orig_single(manifest, record, destination, **kwargs)
            finally:
                self._exit()

        def batch(manifest, items, **kwargs):
            self._enter("batch", len(items))
            try:
                time.sleep(delay_s)
                return orig_batch(manifest, items, **kwargs)
            finally:
                self._exit()

        reader.read_record_into = single
        reader.read_component_records_into = batch

    def _enter(self, method: str, n: int) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append((method, n))

    def _exit(self) -> None:
        with self.lock:
            self.active -= 1

    @property
    def total_records(self) -> int:
        return sum(n for _method, n in self.calls)


# ---------------------------------------------------------------------------
# submission-level grouping (len(pending._miss_futures)) + counters


def test_verify_io_default_off_keeps_per_expert_parts(tmp_path, monkeypatch) -> None:
    """Env unset -> exact current behavior (per-expert parts), counters all zero."""
    monkeypatch.delenv(FANOUT_ENV, raising=False)
    monkeypatch.delenv(UNION_ENV, raising=False)
    root, spec, _m, manifest_path, _e = _overlap_artifact(
        tmp_path, expert_count=6, top_k=4
    )
    runtime = _open_overlap_runtime(root, spec, manifest_path, overlap=False)
    try:
        with runtime.begin_split_route(1, [0, 1, 2, 3], phase="decode") as pending:
            assert len(pending._miss_futures) == 4  # per-expert default
            assert pending._verify_io_timed is False
            _drain_split_route(pending)
        m = _metrics(runtime)
        assert m["verify_io_reads_issued"] == 0
        assert m["verify_io_batches"] == 0
        assert m["verify_io_max_inflight"] == 0
        assert m["verify_io_wait_ns_total"] == 0
    finally:
        runtime.close()


def test_verify_union_read_forces_single_submission(tmp_path, monkeypatch) -> None:
    """UNION_READ=1 collapses the union into ONE submission even with overlap off."""
    monkeypatch.delenv(FANOUT_ENV, raising=False)
    monkeypatch.setenv(UNION_ENV, "1")
    root, spec, _m, manifest_path, _e = _overlap_artifact(
        tmp_path, expert_count=6, top_k=4
    )
    runtime = _open_overlap_runtime(root, spec, manifest_path, overlap=False)
    try:
        with runtime.begin_split_route(1, [0, 1, 2, 3], phase="decode") as pending:
            assert len(pending._miss_futures) == 1  # one batched submission
            assert pending._verify_io_timed is True
            readies = _drain_split_route(pending)
        assert len(readies) == 1
        assert set(readies[0].plan.misses) == {0, 1, 2, 3}
        m = _metrics(runtime)
        assert m["verify_io_batches"] == 1
        assert m["verify_io_reads_issued"] == 4
        assert m["verify_io_max_inflight"] >= 1
    finally:
        runtime.close()


def test_verify_io_fanout_splits_union_into_n_groups(tmp_path, monkeypatch) -> None:
    """FANOUT=2 over 4 misses -> 2 disjoint groups covering the whole union."""
    monkeypatch.setenv(FANOUT_ENV, "2")
    monkeypatch.delenv(UNION_ENV, raising=False)
    root, spec, _m, manifest_path, _e = _overlap_artifact(
        tmp_path, expert_count=6, top_k=4
    )
    runtime = _open_overlap_runtime(root, spec, manifest_path, overlap=False)
    try:
        with runtime.begin_split_route(1, [0, 1, 2, 3], phase="decode") as pending:
            assert len(pending._miss_futures) == 2  # two concurrent groups
            readies = _drain_split_route(pending)
        # The two groups partition the union exactly once.
        covered = set()
        for ready in readies:
            covered |= set(ready.plan.misses)
        assert covered == {0, 1, 2, 3}
        assert sum(len(r.plan.misses) for r in readies) == 4  # disjoint
        m = _metrics(runtime)
        assert m["verify_io_batches"] == 2
        assert m["verify_io_reads_issued"] == 4
    finally:
        runtime.close()


def test_verify_io_fanout_at_or_above_union_is_per_expert(
    tmp_path, monkeypatch
) -> None:
    """FANOUT >= union width -> one part per expert (existing per-expert regime)."""
    monkeypatch.setenv(FANOUT_ENV, "8")
    monkeypatch.setenv(UNION_ENV, "1")  # union+fanout composite (the arm's shape)
    root, spec, _m, manifest_path, _e = _overlap_artifact(
        tmp_path, expert_count=6, top_k=4
    )
    runtime = _open_overlap_runtime(root, spec, manifest_path, overlap=False)
    try:
        with runtime.begin_split_route(1, [0, 1, 2, 3], phase="decode") as pending:
            # min(8, 4) == 4 == unique misses -> per-expert.
            assert len(pending._miss_futures) == 4
            _drain_split_route(pending)
        m = _metrics(runtime)
        assert m["verify_io_batches"] == 4
        assert m["verify_io_reads_issued"] == 4
    finally:
        runtime.close()


# ---------------------------------------------------------------------------
# fake recording reader: N-way concurrency and one-call union batching


def test_fanout_reads_run_concurrently(tmp_path, monkeypatch) -> None:
    """FANOUT=2 -> the two groups' reads are in flight at the same time.

    Adjacent records within each group coalesce into one scatter read, so the only
    concurrency the reader sees is the two GROUPS running at once -- isolating the
    fanout the lever controls (the pre-existing within-part concurrency for
    non-adjacent records would otherwise confound the count).
    """
    monkeypatch.setenv(FANOUT_ENV, "2")
    monkeypatch.delenv(UNION_ENV, raising=False)
    root, spec, _m, manifest_path, _e = _overlap_artifact(
        tmp_path, expert_count=8, top_k=4
    )
    runtime = _open_overlap_runtime(root, spec, manifest_path, overlap=True)
    recorder = _RecordingReader(runtime, delay_s=0.05)
    try:
        # Groups {0,1} and {2,3}; each pair is sidecar-adjacent -> one scatter read
        # per group -> exactly two concurrent reader calls.
        with runtime.begin_split_route(1, [0, 1, 2, 3], phase="decode") as pending:
            assert len(pending._miss_futures) == 2
            _drain_split_route(pending)
        assert recorder.max_active == 2  # two groups read concurrently
        assert len([c for c in recorder.calls if c[0] == "batch"]) == 2
        assert recorder.total_records == 4
        m = _metrics(runtime)
        assert m["verify_io_max_inflight"] == 2
        assert m["verify_io_batches"] == 2
        assert m["verify_io_wait_ns_total"] > 0
    finally:
        runtime.close()


def test_union_read_is_a_single_reader_call(tmp_path, monkeypatch) -> None:
    """UNION_READ=1 (overlap on) -> ONE batched reader call covers the whole union."""
    monkeypatch.delenv(FANOUT_ENV, raising=False)
    monkeypatch.setenv(UNION_ENV, "1")
    root, spec, _m, manifest_path, _e = _overlap_artifact(
        tmp_path, expert_count=8, top_k=4
    )
    runtime = _open_overlap_runtime(root, spec, manifest_path, overlap=True)
    recorder = _RecordingReader(runtime, delay_s=0.01)
    try:
        # Adjacent records -> the batched path coalesces them into one scatter call.
        with runtime.begin_split_route(1, [0, 1, 2, 3], phase="decode") as pending:
            assert len(pending._miss_futures) == 1
            _drain_split_route(pending)
        batch_calls = [c for c in recorder.calls if c[0] == "batch"]
        assert len(batch_calls) == 1  # ONE submission, ONE reader call
        assert batch_calls[0][1] == 4  # covering all four union records
        assert recorder.max_active == 1
    finally:
        runtime.close()


# ---------------------------------------------------------------------------
# byte identity: fanout 1 vs 8 emit bit-identical switch output


def _switch_output(root, spec, manifest_path, monkeypatch, *, fanout: str) -> np.ndarray:
    monkeypatch.setenv(FANOUT_ENV, fanout)
    monkeypatch.setenv(UNION_ENV, "1")
    runtime = _open_overlap_runtime(root, spec, manifest_path, overlap=True)
    try:
        switch = HotExpertSwitchGLU(runtime, 1)
        mx.random.seed(5)
        x = mx.random.normal((1, 1, 64)).astype(mx.bfloat16)
        indices = mx.array([[[0, 1, 2, 3]]], dtype=mx.uint32)
        output = switch(x, indices)
        mx.eval(output)
        engaged = _metrics(runtime)["verify_io_batches"]
        return np.asarray(output.astype(mx.float32)), int(engaged)
    finally:
        runtime.close()


def test_fanout_1_and_8_are_byte_identical(tmp_path, monkeypatch) -> None:
    root, spec, _m, manifest_path, _e = _overlap_artifact(
        tmp_path, expert_count=8, top_k=4
    )
    # Two independent bank builds from the same seed would differ in bytes; reuse
    # the SAME artifact so only the read schedule changes between the arms.
    out1, engaged1 = _switch_output(
        root, spec, manifest_path, monkeypatch, fanout="1"
    )
    out8, engaged8 = _switch_output(
        root, spec, manifest_path, monkeypatch, fanout="8"
    )
    assert engaged1 >= 1  # union (fanout 1) engaged the lever
    assert engaged8 >= 1  # fanout 8 engaged the lever
    assert np.array_equal(out1, out8), "fanout must not change emitted bytes"
