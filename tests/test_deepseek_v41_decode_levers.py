"""Byte-identity + correctness tests for the W24 DSV4.1 decode-streaming levers.

Two runtime levers land in W24, each an explicit switch that defaults OFF and must
NOT change the bytes the reader returns -- it only changes WHEN/HOW records are
read:

  * ``io_read_fanout`` (new, expert_io.py): split one large record's positional
    read into N concurrent contiguous sub-reads.  Tested here on a tiny synthetic
    file: for random (offset, length) reads, fanout in {1,2,4,8} returns bytes
    identical to a single sequential read.
  * ``max_read_chunk_bytes`` (existing read-chunk lever): the chunk size the
    sequential loop reads in must not change the bytes returned.

Plus the pure census-analysis functions (LRU / Belady / reuse distance / Gate 0
verify-union / Gate 1 hot-expert pinning) on synthetic int sequences.

MLX is pinned to the CPU device at import (worker-tests-must-pin-mlx-cpu.md: a
"no GPU" instruction is not enough -- MLX defaults to Metal).  These tests do not
require the real 269 GiB artifact or any GPU.
"""

from __future__ import annotations

import importlib.util
import os
import random
from pathlib import Path

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mtplx.expert_io import PositionalExpertReader
from mtplx.expert_runtime import ExpertStreamingConfig

_REPO = Path(__file__).resolve().parents[1]


def _load_census_module():
    path = _REPO / "scripts" / "deepseek_v41" / "routing_census.py"
    spec = importlib.util.spec_from_file_location("w24_routing_census", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _load_census_module()


# --------------------------------------------------------------------------- #
# io_read_fanout -- byte identity + validation
# --------------------------------------------------------------------------- #
def _make_bank(tmp_path: Path, size: int) -> bytes:
    data = os.urandom(size)
    (tmp_path / "experts.bin").write_bytes(data)
    return data


@pytest.mark.parametrize("fanout", [1, 2, 4, 8])
def test_io_read_fanout_is_byte_identical(tmp_path, fanout):
    # small chunk so even a modest record exercises the sequential chunk loop
    # AND (for fanout>1) the concurrent split.
    size = 200_003  # deliberately not a multiple of anything
    data = _make_bank(tmp_path, size)
    reader = PositionalExpertReader(
        tmp_path,
        max_read_chunk_bytes=4096,
        use_native=False,  # exercise the portable os.preadv path deterministically
        io_read_fanout=fanout,
    )
    try:
        rng = random.Random(1234 + fanout)
        for _ in range(40):
            length = rng.randint(1, size)
            offset = rng.randint(0, size - length)
            buf = bytearray(length)
            reader._read_range_into(
                "experts.bin", offset, memoryview(buf),
                cancel_event=None, deadline_ns=None,
            )
            assert bytes(buf) == data[offset:offset + length], (
                f"fanout={fanout} offset={offset} length={length} mismatch"
            )
        # a full-record read (> chunk, triggers fanout split for fanout>1)
        buf = bytearray(size)
        reader._read_range_into(
            "experts.bin", 0, memoryview(buf),
            cancel_event=None, deadline_ns=None,
        )
        assert bytes(buf) == data
        # the real bytes-read counter (Gate D basis) must be exact
        assert reader.metrics.as_dict()["read_bytes"] >= size
    finally:
        reader.close()


def test_io_read_fanout_matches_control(tmp_path):
    size = 500_000
    data = _make_bank(tmp_path, size)
    control = PositionalExpertReader(tmp_path, max_read_chunk_bytes=8192,
                                     use_native=False, io_read_fanout=1)
    fan = PositionalExpertReader(tmp_path, max_read_chunk_bytes=8192,
                                 use_native=False, io_read_fanout=6)
    try:
        for offset, length in [(0, size), (1, size - 2), (12345, 400000), (size - 10, 10)]:
            a = bytearray(length)
            b = bytearray(length)
            control._read_range_into("experts.bin", offset, memoryview(a),
                                     cancel_event=None, deadline_ns=None)
            fan._read_range_into("experts.bin", offset, memoryview(b),
                                 cancel_event=None, deadline_ns=None)
            assert bytes(a) == bytes(b) == data[offset:offset + length]
    finally:
        control.close()
        fan.close()


def test_io_read_fanout_split_partitions_exactly():
    for total in (1, 7, 100, 18_800_000):
        for parts in (1, 2, 3, 4, 8):
            segs = PositionalExpertReader._fanout_split(total, parts)
            assert sum(l for _, l in segs) == total
            # contiguous, disjoint, covering [0,total)
            pos = 0
            for start, length in segs:
                assert start == pos and length > 0
                pos += length
            assert pos == total


def test_io_read_fanout_validation(tmp_path):
    _make_bank(tmp_path, 100)
    with pytest.raises((ValueError, TypeError)):
        PositionalExpertReader(tmp_path, io_read_fanout=0)
    with pytest.raises((ValueError, TypeError)):
        PositionalExpertReader(tmp_path, io_read_fanout=True)


def test_config_io_read_fanout_default_off_and_validated():
    cfg = ExpertStreamingConfig(model_key="k", memory_limit_bytes=1, max_live_kv_tokens=0)
    assert cfg.io_read_fanout == 1  # default OFF
    cfg2 = ExpertStreamingConfig(model_key="k", memory_limit_bytes=1,
                                 max_live_kv_tokens=0, io_read_fanout=4)
    assert cfg2.io_read_fanout == 4
    with pytest.raises((ValueError, TypeError)):
        ExpertStreamingConfig(model_key="k", memory_limit_bytes=1,
                              max_live_kv_tokens=0, io_read_fanout=0)


# --------------------------------------------------------------------------- #
# max_read_chunk_bytes -- byte identity across chunk sizes (read-chunk lever)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("chunk", [4096, 65536, 1_000_000])
def test_max_read_chunk_bytes_is_byte_identical(tmp_path, chunk):
    size = 300_000
    data = _make_bank(tmp_path, size)
    reader = PositionalExpertReader(tmp_path, max_read_chunk_bytes=chunk,
                                    use_native=False, io_read_fanout=1)
    try:
        buf = bytearray(size)
        reader._read_range_into("experts.bin", 0, memoryview(buf),
                                cancel_event=None, deadline_ns=None)
        assert bytes(buf) == data
    finally:
        reader.close()


# --------------------------------------------------------------------------- #
# Pure census analysis
# --------------------------------------------------------------------------- #
def test_lru_and_belady_and_reuse():
    assert rc.simulate_lru([1, 2, 3, 1], 2) == (0, 4)
    assert rc.simulate_lru([1, 2, 3, 1], 3) == (1, 3)
    assert rc.simulate_belady([1, 2, 3, 1], 2) == (1, 3)
    # Belady is never worse than LRU at the same capacity.
    seq = [random.Random(0).randint(0, 20) for _ in range(300)]
    for cap in (3, 5, 10):
        _, lru_m = rc.simulate_lru(seq, cap)
        _, bel_m = rc.simulate_belady(seq, cap)
        assert bel_m <= lru_m
    assert rc.reuse_distances([1, 2, 3, 1]) == [-1, -1, -1, 2]
    assert rc.simulate_lru([1], 2, warm=[1, 2]) == (1, 0)


def test_gate0_verify_union():
    # layer 0: identical routing every token -> u == top_k; layer 1: disjoint.
    steps = {
        0: [[1, 2, 3, 4, 5, 6]] * 4,
        1: [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]],
    }
    g = rc.verify_union_stats(steps, [2, 4])
    # Wp=2: layer0 union 6, layer1 union 12 -> median 9
    assert g[2]["u_median"] == 9
    # Wp=4: layer0 union 6, layer1 union 24
    assert g[4]["u_max"] == 24
    assert g[2]["dedup_factor_vs_naive"] == pytest.approx(12 / 9)


def test_gate1_pin_beats_lru_on_concentrated_traffic():
    # a highly concentrated layer: pinning the hot expert should beat LRU.
    hot = {0: [[1, 1]] * 20 + [[1, 2]] * 8}
    g = rc.pin_residency_gate(hot, [1], train_frac=0.7)
    assert g[1]["static_pin_miss_rate"] <= g[1]["lru_miss_rate"] + 1e-9
    # Belady is the bound.
    assert g[1]["belady_miss_rate"] <= g[1]["lru_miss_rate"] + 1e-9


def test_gate1_uniform_traffic_has_low_concentration_stdev():
    # every layer identically diffuse -> cross-layer coverage stdev ~ 0 (hy3 dead
    # case): the pin lever should NOT be claimed.
    uniform = {L: [[i % 10, (i + 1) % 10] for i in range(40)] for L in range(6)}
    g = rc.pin_residency_gate(uniform, [4], train_frac=0.7)
    assert g[4]["cross_layer_top_n_coverage_stdev"] == pytest.approx(0.0, abs=1e-9)
