"""W123: io read-fanout on the component-banks scatter path.

These tests are pure-Python I/O (no MLX, no model, no GPU): they build a small
temp file and drive :class:`PositionalExpertReader` directly, so they run on the
CPU with a tiny RSS.

They lock in the two invariants the fix rests on:

* **byte-identity** -- fanning a record's contiguous sidecar extent into N
  concurrent sub-reads reads exactly the same bytes into exactly the same
  component buffers as the shipped single scatter (kill-switch ``fanout==1``);
* **real concurrency** -- with fanout>1 the sub-reads are in flight at the same
  time (the read-pool depth gauge peaks at the group count), which is what raises
  ``read_ns/wall`` above the QD1 floor.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.expert_io import PositionalExpertReader


class _ComponentDestination:
    """A component-banks slot buffer: one memoryview per record segment."""

    def __init__(self, lengths: tuple[int, ...]) -> None:
        self.buffers = tuple(bytearray(length) for length in lengths)

    def record_views(self, _record: object) -> tuple[memoryview, ...]:
        return tuple(memoryview(buffer) for buffer in self.buffers)

    def payload(self) -> bytes:
        return b"".join(bytes(buffer) for buffer in self.buffers)


def _write_record(tmp_path: Path, lengths: tuple[int, ...]) -> tuple[str, bytes]:
    """Write a deterministic payload covering ``sum(lengths)`` bytes."""

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


# --------------------------------------------------------------------------- #
# _fanout_offset_groups: the pure partitioner.
# --------------------------------------------------------------------------- #


def _views(*lengths: int) -> tuple[memoryview, ...]:
    return tuple(memoryview(bytearray(length)) for length in lengths)


def test_fanout_offset_groups_single_when_disabled() -> None:
    views = _views(10, 20, 30)
    groups = PositionalExpertReader._fanout_offset_groups(views, 1)
    assert len(groups) == 1
    assert groups[0][0] == 0
    assert len(groups[0][1]) == 3


def test_fanout_offset_groups_tiles_contiguously_and_disjointly() -> None:
    lengths = (100, 4, 100, 4, 100, 4)
    views = _views(*lengths)
    groups = PositionalExpertReader._fanout_offset_groups(views, 4)
    # At most ``parts`` groups.
    assert 1 < len(groups) <= 4
    # Reconstruct the covered (offset, length) spans and check they tile
    # [0, total) exactly once, in order, with no gaps or overlaps.
    total = sum(lengths)
    cursor = 0
    covered = 0
    for offset, group_views in groups:
        assert offset == cursor  # contiguous, no gap
        span = sum(len(view) for view in group_views)
        assert span > 0
        cursor += span
        covered += span
    assert cursor == total
    assert covered == total


def test_fanout_offset_groups_never_exceeds_parts() -> None:
    views = _views(*([1000] * 9))
    for parts in (2, 3, 4, 8, 16):
        groups = PositionalExpertReader._fanout_offset_groups(views, parts)
        assert len(groups) <= parts
        assert sum(len(v) for _off, grp in groups for v in grp) == 9000


def test_fanout_offset_groups_drops_empty_views() -> None:
    views = (memoryview(bytearray(10)), memoryview(bytearray(0)),
             memoryview(bytearray(10)))
    groups = PositionalExpertReader._fanout_offset_groups(views, 4)
    kept = [v for _off, grp in groups for v in grp]
    assert all(len(v) for v in kept)
    assert sum(len(v) for v in kept) == 20


# --------------------------------------------------------------------------- #
# Byte-identity: fanout is a scheduling change, never a bytes change.
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
    lengths = (300_000, 4, 300_000, 4, 300_000, 4)  # 3 "weights" + 3 "scales"
    payload, destination, metrics = _read_once(tmp_path, lengths, fanout=1)
    assert destination.payload() == payload
    # fanout==1 -> the exact prior path: one scatter, one range-reader call.
    assert metrics["read_operations"] == 1
    assert metrics["read_inflight_max"] == 1


def test_fanned_read_is_byte_identical_to_single_scatter(tmp_path: Path) -> None:
    lengths = (300_000, 4, 300_000, 4, 300_000, 4)
    payload_single, single, _ = _read_once(tmp_path, lengths, fanout=1)
    payload_fanned, fanned, metrics = _read_once(tmp_path, lengths, fanout=8)
    assert payload_single == payload_fanned
    # Same bytes, same component split, byte-for-byte.
    assert fanned.payload() == payload_fanned
    assert fanned.payload() == single.payload()
    for a, b in zip(single.buffers, fanned.buffers):
        assert bytes(a) == bytes(b)
    # Fanout issued more than one scatter (the concurrency source).
    assert metrics["read_operations"] > 1


def test_fanned_read_handles_uneven_and_tiny_components(tmp_path: Path) -> None:
    lengths = (1, 5_000_000, 1, 1, 5_000_000, 2)
    payload, destination, metrics = _read_once(tmp_path, lengths, fanout=4)
    assert destination.payload() == payload
    assert metrics["read_bytes"] == len(payload)


# --------------------------------------------------------------------------- #
# Real concurrency: the sub-reads are genuinely in flight together.
# --------------------------------------------------------------------------- #


def test_fanout_reads_run_concurrently_depth_equals_group_count(
    tmp_path: Path,
) -> None:
    # Four equal components -> exactly four contiguous groups at fanout==4.
    lengths = (4096, 4096, 4096, 4096)
    relative_name, payload = _write_record(tmp_path, lengths)
    manifest, record = _record(relative_name, lengths, len(payload))
    destination = _ComponentDestination(lengths)

    with PositionalExpertReader(
        tmp_path, use_native=False, io_read_fanout=4
    ) as reader:
        groups = reader._fanout_offset_groups(
            destination.record_views(record), reader.io_read_fanout
        )
        parties = len(groups)
        assert parties == 4  # deterministic for equal components

        # Gate every read INSIDE the depth gauge (after enter_read incremented
        # the in-flight counter) on a shared barrier: it only releases once all
        # ``parties`` reads have entered, so a serialized reader would block and
        # time out. Success therefore proves the reads overlap, and the gauge
        # necessarily peaks at ``parties``.
        barrier = threading.Barrier(parties, timeout=10)
        original_enter = reader.metrics.enter_read

        def barriered_enter() -> None:
            original_enter()
            barrier.wait()

        reader.metrics.enter_read = barriered_enter  # type: ignore[method-assign]

        reader.read_record_into(
            manifest, record, destination, verify_hash=False, pipeline_phase="decode"
        )
        metrics = reader.metrics.as_dict()

    assert destination.payload() == payload  # concurrency did not corrupt bytes
    assert metrics["read_inflight_max"] == parties
    assert metrics["read_inflight_depth_mean"] > 1.0
    assert metrics["read_operations"] == parties


def test_single_scatter_depth_gauge_stays_at_one(tmp_path: Path) -> None:
    lengths = (2048, 2048, 2048)
    _payload, _destination, metrics = _read_once(tmp_path, lengths, fanout=1)
    # No fanout -> exactly one read in flight -> QD1 gauge.
    assert metrics["read_inflight_max"] == 1
    assert metrics["read_inflight_depth_mean"] == pytest.approx(1.0)


def test_fanout_error_in_one_group_propagates(tmp_path: Path) -> None:
    lengths = (1024, 1024, 1024, 1024)
    relative_name, payload = _write_record(tmp_path, lengths)
    manifest, record = _record(relative_name, lengths, len(payload))
    destination = _ComponentDestination(lengths)

    with PositionalExpertReader(
        tmp_path, use_native=False, io_read_fanout=4
    ) as reader:
        boom = RuntimeError("scatter boom")
        calls = {"n": 0}
        original = reader._readv_range_into

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise boom
            return original(*args, **kwargs)

        reader._readv_range_into = flaky  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="scatter boom"):
            reader.read_record_into(
                manifest,
                record,
                destination,
                verify_hash=False,
                pipeline_phase="decode",
            )
