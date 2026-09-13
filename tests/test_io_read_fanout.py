"""W123: io read-fanout on the component-banks scatter path + queue-depth gauge.

Pure-Python I/O (no MLX, no model, no GPU): small temp files drive
:class:`PositionalExpertReader` directly, so these run on the CPU with a tiny RSS.

Invariants locked here (post red-team):
* **HIGH-1 even slicing** -- fanout cuts the record's flat extent, not the
  component boundaries, so fanout 4 vs 8 genuinely give 4 vs 8 near-equal
  ranges (not a null A/B capped at #big-components);
* **byte-identity** -- the fanned read is byte-for-byte the single scatter;
* **HIGH-2 caller participation** -- concurrent fanned records do not serialize,
  and a demand record issues its own first sub-read without waiting on a blocked
  pool;
* **gauge** -- ``read_wall_ns`` is the union of in-flight intervals, giving a
  realized-QD and realized-BW that survive the receipt's per-window delta.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.expert_io import PositionalExpertReader
from mtplx.serve_stream_counters import stream_counters_delta


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


class _ComponentDestination:
    def __init__(self, lengths: tuple[int, ...]) -> None:
        self.buffers = tuple(bytearray(length) for length in lengths)

    def record_views(self, _record: object) -> tuple[memoryview, ...]:
        return tuple(memoryview(buffer) for buffer in self.buffers)

    def payload(self) -> bytes:
        return b"".join(bytes(buffer) for buffer in self.buffers)


def _write_record(tmp_path: Path, lengths: tuple[int, ...]) -> tuple[str, bytes]:
    total = sum(lengths)
    payload = bytes((index * 7 + 3) % 256 for index in range(total))
    relative_name = "bank.sidecar"
    (tmp_path / relative_name).write_bytes(payload)
    return relative_name, payload


def _record(relative_name: str, lengths: tuple[int, ...], total: int):
    manifest = SimpleNamespace(sidecar=SimpleNamespace(file=relative_name))
    record = SimpleNamespace(
        layer=1,
        expert=2,
        logical_bytes=total,
        segments=tuple(SimpleNamespace(length=length) for length in lengths),
        sidecar_offset=0,
        sidecar_length=total,
        sha256=None,
    )
    return manifest, record


# mxfp4-shaped geometry scaled to 4 KiB multiples: 3 "weight" + 3 "scale"
# components. Big components dwarf the scales, which is exactly why component-edge
# grouping capped the old fanout at ~3-4 groups.
_WEIGHT = 32 * 4096   # 131072
_SCALE = 2 * 4096     # 8192
_MXFP4_LIKE = (_WEIGHT, _SCALE, _WEIGHT, _SCALE, _WEIGHT, _SCALE)


# --------------------------------------------------------------------------- #
# HIGH-1: even byte-range partitioner
# --------------------------------------------------------------------------- #


def test_byte_ranges_scale_with_parts_and_stay_balanced() -> None:
    total = sum(_MXFP4_LIKE)  # 417792
    for parts in (2, 4, 8):
        ranges = PositionalExpertReader._fanout_byte_ranges(total, parts)
        assert len(ranges) == parts  # cuts the bytes, not the components
        # contiguous tiling of [0, total)
        cursor = 0
        for lo, hi in ranges:
            assert lo == cursor
            assert hi > lo
            cursor = hi
        assert cursor == total
        # aligned interior cuts
        for lo, _hi in ranges[1:]:
            assert lo % 4096 == 0
        # balanced: max range <= 1.1x mean
        sizes = [hi - lo for lo, hi in ranges]
        assert max(sizes) <= 1.1 * (total / parts)


def test_byte_ranges_disabled_and_tiny() -> None:
    assert PositionalExpertReader._fanout_byte_ranges(1000, 1) == [(0, 1000)]
    assert PositionalExpertReader._fanout_byte_ranges(0, 8) == []


def test_slices_for_byte_range_maps_across_component_edges() -> None:
    views = tuple(memoryview(bytearray(n)) for n in (10, 10, 10))
    cum = [0, 10, 20, 30]
    # a range straddling the first two components -> two slices
    slices = PositionalExpertReader._slices_for_byte_range(views, cum, 5, 15)
    assert [len(s) for s in slices] == [5, 5]
    # a range inside one component -> one slice
    slices = PositionalExpertReader._slices_for_byte_range(views, cum, 22, 28)
    assert [len(s) for s in slices] == [6]


# --------------------------------------------------------------------------- #
# Byte-identity + scaling (fixes the null A/B)
# --------------------------------------------------------------------------- #


def _read_once(tmp_path: Path, lengths: tuple[int, ...], fanout: int):
    relative_name, payload = _write_record(tmp_path, lengths)
    manifest, record = _record(relative_name, lengths, len(payload))
    destination = _ComponentDestination(lengths)
    with PositionalExpertReader(
        tmp_path, use_native=False, io_read_fanout=fanout
    ) as reader:
        reader.read_record_into(
            manifest, record, destination, verify_hash=False, pipeline_phase="decode"
        )
        metrics = reader.metrics.as_dict()
    return payload, destination, metrics


def test_kill_switch_default_is_single_scatter(tmp_path: Path) -> None:
    payload, destination, metrics = _read_once(tmp_path, _MXFP4_LIKE, fanout=1)
    assert destination.payload() == payload
    assert metrics["read_operations"] == 1
    assert metrics["records_read"] == 1
    assert metrics["read_inflight_max"] == 1


def test_fanout_scales_read_count_with_parts(tmp_path: Path) -> None:
    # The red-team's null A/B: with component-edge grouping fanout 4 and 8 gave
    # the SAME ~4 ops. Even byte slicing must give 4 and 8.
    _p4, d4, m4 = _read_once(tmp_path, _MXFP4_LIKE, fanout=4)
    _p8, d8, m8 = _read_once(tmp_path, _MXFP4_LIKE, fanout=8)
    assert m4["read_operations"] == 4
    assert m8["read_operations"] == 8
    assert m4["records_read"] == 1 and m8["records_read"] == 1


def test_fanned_read_is_byte_identical_to_single_scatter(tmp_path: Path) -> None:
    p1, single, _ = _read_once(tmp_path, _MXFP4_LIKE, fanout=1)
    p8, fanned, _ = _read_once(tmp_path, _MXFP4_LIKE, fanout=8)
    assert p1 == p8
    assert fanned.payload() == p8
    for a, b in zip(single.buffers, fanned.buffers):
        assert bytes(a) == bytes(b)


def test_fanned_read_handles_uneven_and_tiny_components(tmp_path: Path) -> None:
    lengths = (4096, 5_000_192, 4096, 4096, 5_000_192, 8192)
    payload, destination, metrics = _read_once(tmp_path, lengths, fanout=4)
    assert destination.payload() == payload
    assert metrics["read_bytes"] == len(payload)


# --------------------------------------------------------------------------- #
# HIGH-2: caller participation + real concurrency + gauge
# --------------------------------------------------------------------------- #


def test_gauge_union_wall_and_realized_qd(tmp_path: Path) -> None:
    # Barrier gated INSIDE the depth gauge (after enter_read) so all `parts`
    # sub-reads are in flight together; a serialized reader would deadlock.
    lengths = _MXFP4_LIKE
    relative_name, payload = _write_record(tmp_path, lengths)
    manifest, record = _record(relative_name, lengths, len(payload))
    destination = _ComponentDestination(lengths)
    parts = 4
    with PositionalExpertReader(
        tmp_path, use_native=False, io_read_fanout=parts
    ) as reader:
        ranges = reader._fanout_byte_ranges(len(payload), parts)
        assert len(ranges) == parts
        barrier = threading.Barrier(parts, timeout=10)
        original_enter = reader.metrics.enter_read

        def barriered_enter() -> None:
            original_enter()
            barrier.wait()

        reader.metrics.enter_read = barriered_enter  # type: ignore[method-assign]
        reader.read_record_into(
            manifest, record, destination, verify_hash=False, pipeline_phase="decode"
        )
        metrics = reader.metrics.as_dict()

    assert destination.payload() == payload
    assert metrics["read_inflight_max"] == parts
    assert metrics["read_wall_ns"] > 0
    # thread-time (sum) >> union wall -> realized QD well above 1
    assert metrics["read_realized_qd"] > 1.5
    assert metrics["read_realized_gb_per_s"] > 0


def test_single_scatter_realized_qd_is_one(tmp_path: Path) -> None:
    _p, _d, metrics = _read_once(tmp_path, (2048, 2048, 2048), fanout=1)
    assert metrics["read_inflight_max"] == 1
    assert metrics["read_realized_qd"] == pytest.approx(1.0, abs=0.2)


def test_two_concurrent_fanned_records_do_not_serialize(tmp_path: Path) -> None:
    # 2 records x fanout 4 = 8 sub-reads; a Barrier(8) inside the gauge only
    # releases if all 8 overlap (caller participation + a pool wide enough).
    lengths = _MXFP4_LIKE
    relative_name, payload = _write_record(tmp_path, lengths)
    manifest, record = _record(relative_name, lengths, len(payload))
    with PositionalExpertReader(
        tmp_path, use_native=False, io_read_fanout=4
    ) as reader:
        barrier = threading.Barrier(8, timeout=10)
        original_enter = reader.metrics.enter_read

        def barriered_enter() -> None:
            original_enter()
            barrier.wait()

        reader.metrics.enter_read = barriered_enter  # type: ignore[method-assign]
        errors: list[BaseException] = []

        def run() -> None:
            try:
                dest = _ComponentDestination(lengths)
                reader.read_record_into(
                    manifest, record, dest, verify_hash=False, pipeline_phase="decode"
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        assert not any(thread.is_alive() for thread in threads)
        assert not errors
        assert reader.metrics.as_dict()["read_inflight_max"] == 8


def test_demand_first_subread_beats_blocked_pool(tmp_path: Path) -> None:
    # The caller thread must issue its own first sub-read even when the fanout
    # pool is fully blocked (a prefetch backlog): caller participation, not the
    # pool, drives the first preadv.
    lengths = _MXFP4_LIKE
    relative_name, payload = _write_record(tmp_path, lengths)
    manifest, record = _record(relative_name, lengths, len(payload))
    destination = _ComponentDestination(lengths)
    with PositionalExpertReader(
        tmp_path, use_native=False, io_read_fanout=4
    ) as reader:
        caller_proceeded = threading.Event()
        release_pool = threading.Event()
        original = reader._readv_range_into

        def gated(name, offset, slices, **kwargs):
            if threading.current_thread().name.startswith("mtplx-io-fanout"):
                release_pool.wait(timeout=10)  # pool sub-reads blocked
            else:
                caller_proceeded.set()  # caller sub-read runs regardless
            return original(name, offset, slices, **kwargs)

        reader._readv_range_into = gated  # type: ignore[method-assign]
        worker = threading.Thread(
            target=lambda: reader.read_record_into(
                manifest, record, destination, verify_hash=False,
                pipeline_phase="decode",
            )
        )
        worker.start()
        try:
            assert caller_proceeded.wait(timeout=5)
        finally:
            release_pool.set()
        worker.join(timeout=10)
        assert not worker.is_alive()
    assert destination.payload() == payload


def test_fanout_error_in_one_range_propagates(tmp_path: Path) -> None:
    lengths = _MXFP4_LIKE
    relative_name, payload = _write_record(tmp_path, lengths)
    manifest, record = _record(relative_name, lengths, len(payload))
    destination = _ComponentDestination(lengths)
    with PositionalExpertReader(
        tmp_path, use_native=False, io_read_fanout=4
    ) as reader:
        boom = RuntimeError("scatter boom")
        calls = {"n": 0}
        lock = threading.Lock()
        original = reader._readv_range_into

        def flaky(*args, **kwargs):
            with lock:
                calls["n"] += 1
                n = calls["n"]
            if n == 2:
                raise boom
            return original(*args, **kwargs)

        reader._readv_range_into = flaky  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="scatter boom"):
            reader.read_record_into(
                manifest, record, destination, verify_hash=False,
                pipeline_phase="decode",
            )


# --------------------------------------------------------------------------- #
# Receipt delta path: the gauge must survive per-window differencing
# --------------------------------------------------------------------------- #


def test_stream_counters_delta_fixes_gauge_window_metrics() -> None:
    # Reproduces the window-50 breakage: read_inflight_max is a PEAK and
    # read_inflight_depth_mean a derived FLOAT -- differencing them gives garbage
    # (max=1, mean negative). The fix takes max from AFTER and recomputes mean /
    # realized-QD / GB-window from monotonic-counter deltas + the union wall.
    before = {
        "io": {
            "read_bytes": 1_000,
            "read_ns": 100,
            "read_wall_ns": 100,
            "read_inflight_max": 3,          # peak already 3 from prefill
            "read_inflight_depth_sum": 5,
            "read_inflight_samples": 5,
            "read_inflight_depth_mean": 1.0,  # derived float
            "read_mib_per_second": 42.0,      # derived float
            "records_read": 5,
        }
    }
    after = {
        "io": {
            "read_bytes": 1_000 + 365_000,
            "read_ns": 100 + 2_450,           # summed thread-time
            "read_wall_ns": 100 + 1_000,      # union (drive-busy) wall
            "read_inflight_max": 4,           # window peak
            "read_inflight_depth_sum": 5 + 3_600,
            "read_inflight_samples": 5 + 900,
            "read_inflight_depth_mean": 4.0,
            "read_mib_per_second": 55.0,
            "records_read": 5 + 900,
        }
    }
    out = stream_counters_delta(before, after, tokens=256)["io"]
    # MEDIUM-2: read_inflight_max is a cumulative peak (prefill pollution) and is
    # intentionally NOT emitted in the window block; read_realized_qd is the
    # window evidence.
    assert "read_inflight_max" not in out
    assert out["read_inflight_depth_mean"] == pytest.approx(4.0)  # 3600/900
    assert out["read_realized_qd"] == pytest.approx(2.45)  # 2450/1000
    assert out["read_gb_per_s_window"] == pytest.approx(365.0)  # bytes/union-ns
    assert out["read_thread_gb_per_s_window"] == pytest.approx(149.0, rel=1e-3)
    assert out["records_read"] == 900


# --------------------------------------------------------------------------- #
# MEDIUM fixes: open-interval wall, issue-time reject counting, pool sizing
# --------------------------------------------------------------------------- #


def test_read_wall_ns_includes_open_interval(tmp_path: Path) -> None:
    # MEDIUM-1: a read still in flight at snapshot time (pool never idled) must
    # still report a nonzero union wall -- otherwise read_gb_per_s_window /
    # read_realized_qd delta to None, the very numbers the lever is judged by.
    import time

    with PositionalExpertReader(tmp_path, use_native=False) as reader:
        reader.metrics.enter_read()
        try:
            time.sleep(0.01)
            snap = reader.metrics.as_dict()
        finally:
            reader.metrics.exit_read()
    assert snap["read_wall_ns"] > 0
    # and the stored counter is not corrupted: after exit it closed exactly once
    assert reader.metrics.read_inflight_current == 0


def test_issue_time_deadline_and_cancel_are_counted(tmp_path: Path) -> None:
    # MEDIUM-3: a pre-`try` (issue-time) deadline/cancel rejection used to raise
    # without incrementing the counter.
    from mtplx.expert_io import ExpertIOCancelled, ExpertIODeadlineExceeded

    relative_name = "d.bin"
    (tmp_path / relative_name).write_bytes(b"0123456789")
    with PositionalExpertReader(tmp_path, use_native=False) as reader:
        with pytest.raises(ExpertIODeadlineExceeded):
            reader._read_range_into(
                relative_name, 0, memoryview(bytearray(10)),
                cancel_event=None, deadline_ns=1,  # already past
            )
        assert reader.metrics.as_dict()["deadline_errors"] == 1
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(ExpertIOCancelled):
            reader._readv_range_into(
                relative_name, 0, (memoryview(bytearray(10)),),
                cancel_event=cancel, deadline_ns=None,
            )
        assert reader.metrics.as_dict()["cancellations"] == 1


def test_fanout_pool_sized_for_concurrent_records(tmp_path: Path) -> None:
    # MEDIUM-4: pool >= (1 + prefetch_inflight_cap) x (fanout - 1) so a demand
    # tail never queues behind prefetch tails.
    for fanout, expected in ((4, 15), (8, 35)):
        with PositionalExpertReader(
            tmp_path, use_native=False, io_read_fanout=fanout
        ) as reader:
            assert reader._fanout_pool_workers == expected
