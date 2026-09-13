"""CPU-pinned unit test for the W106 receipt ``memory`` block + background sampler.

Covers ``scripts/deepseek_v41/bench_standard_shape.py``:

  * ``_MemorySampler`` (a daemon-thread 1 Hz RSS + system-used sampler that touches
    no MLX and holds no lock), and
  * ``_MLXMemProbe.memory_block`` -- the block that fixes David's "peak_gb is the
    actual memory usage" misread by carrying ``mlx_peak_gb`` (the old ``peak_gb``),
    ``process_peak_rss_gb``, ``system_used_peak_gb`` and ``system_used_at_decode_start_gb``.

The block machinery is MLX-agnostic (it reads ru_maxrss / ps / vm_stat, never the
allocator), so this test drives it with a FAKE mx that reports a chosen MLX peak.
No MLX import, GPU, model, server, or network. Run under ``nice -n 19`` and without ``pytest -n
auto`` (host-encode sensitivity).
"""

from __future__ import annotations

import importlib.util
import time
import sys

import pytest
from pathlib import Path


_WT = Path(__file__).resolve().parents[1]
_SCRIPTS = _WT / "scripts" / "deepseek_v41"

GIB = 1024**3


def _load(name: str):
    path = _SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"dsv41_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeMx:
    """A stand-in for ``mlx.core`` that reports a fixed allocator peak, so the test
    controls ``mlx_peak_gb`` without allocating on any device."""

    def __init__(self, peak_bytes: int):
        self._peak = int(peak_bytes)

    def get_peak_memory(self) -> int:
        return self._peak

    def reset_peak_memory(self) -> None:  # pragma: no cover - not exercised here
        self._peak = 0


def _bench():
    return _load("bench_standard_shape")


@pytest.mark.skipif(sys.platform != "darwin", reason="live macOS footprint probe")
def test_memory_block_has_all_keys_and_process_rss_ge_mlx_peak():
    bench = _bench()
    # A deliberately TINY fake MLX peak (8 MiB) so process RSS (this pytest process,
    # tens of MiB at least) genuinely exceeds it -- proving the block accounts for
    # more than the MLX allocator (the whole point of the fix), not just passing a
    # tautology.
    probe = bench._MLXMemProbe(_FakeMx(8 * 1024 * 1024))
    sampler = probe.new_sampler(interval_s=0.05)
    sampler.start()
    # Move RSS with a touched ~200 MiB buffer so the sampler + ru_maxrss see it.
    blob = bytearray(200 * 1024 * 1024)
    for i in range(0, len(blob), 4096):
        blob[i] = 1
    time.sleep(0.2)
    sampler.stop()

    block = probe.memory_block(sampler)

    # W106 MEDIUM-2: three distinct, single-meaning process keys (no max() blob).
    for key in (
        "mlx_peak_gb",
        "process_footprint_peak_gib",
        "ru_maxrss_gb",
        "process_footprint_peak_gb",
        "system_used_peak_gb",
        "system_used_at_run_start_gb",
    ):
        assert key in block, f"missing key {key!r} in memory block: {block}"
        assert isinstance(block[key], float)

    # mlx_peak_gb is exactly the fake allocator peak.
    assert block["mlx_peak_gb"] == (8 * 1024 * 1024) / 1e9

    # With a sampler, process_peak_rss_gb IS the sampler peak (bracketed), not a
    # max() with mlx/ru_maxrss.
    assert block["process_footprint_peak_bytes"] == sampler.peak_rss_bytes
    # The sampler caught the ~200 MiB blob, so it dwarfs the 8 MiB fake MLX peak.
    assert block["process_footprint_peak_gb"] > block["mlx_peak_gb"]
    assert block["process_footprint_peak_gib"] > 0.1  # >100 MiB, the touched blob

    # Non-negative envelope figures (system_used is >0 on darwin, 0 elsewhere).
    assert block["system_used_peak_gb"] >= 0.0
    assert block["system_used_at_run_start_gb"] >= 0.0
    assert block["system_used_peak_gb"] >= block["system_used_at_run_start_gb"]

    del blob


def test_memory_block_without_sampler_still_builds():
    """No sampler: footprint/system peaks are unknown; RSS stays separate."""
    probe = _bench()._MLXMemProbe(_FakeMx(4 * 1024 * 1024))
    block = probe.memory_block(None)
    assert block["process_footprint_peak_gb"] is None
    assert block["system_used_peak_gb"] is None
    assert block["system_used_at_run_start_gb"] is None
    assert block["ru_maxrss_gb"] > 0


def test_sampler_thread_is_daemon_and_stops_cleanly():
    """The sampler runs off the hot path (a daemon thread) and joins on stop."""
    bench = _bench()
    probe = bench._MLXMemProbe(_FakeMx(1))
    sampler = probe.new_sampler(interval_s=0.05)
    sampler.start()
    assert sampler._thread is not None
    assert sampler._thread.daemon is True
    assert sampler._thread.is_alive()
    sampler.stop()
    assert sampler._thread is None


# --------------------------------------------------------------------------
# David's directive: the printed headline + the top-level receipt must carry the
# whole-PROCESS number (incl. non-Metal), not only the MLX allocator peak.  These
# cover the new keys/helpers in ab_decode_env_levers.py (loaded by file path).
# --------------------------------------------------------------------------


def _ab():
    return _load("ab_decode_env_levers")


def test_peak_process_gb_reads_process_footprint_from_run():
    ab = _ab()
    # W121: the receipt memory block carries process_footprint_peak_gb (decimal GB).
    run = {"memory": {"mlx_peak_gb": 40.0, "process_footprint_peak_gb": 52.5,
                      "box_used_gb": 63.5, "box_baseline_gb": 11.0}}
    # peak_process_gb is the whole-process phys_footprint peak, NOT the MLX peak.
    assert ab._peak_process_gb(run) == 52.5


def test_peak_process_gb_none_when_no_memory_block():
    ab = _ab()
    assert ab._peak_process_gb({}) is None
    assert ab._peak_process_gb({"memory": {}}) is None


def test_memory_headline_prints_footprint_and_box_used():
    ab = _ab()
    receipt = {
        "peak_gb": 40.0,
        "memory": {
            "mlx_peak_gb": 40.0,
            "process_footprint_peak_gb": 52.5,
            "box_used_gb": 63.5,
            "box_baseline_gb": 11.0,
        },
    }
    line = ab._memory_headline(receipt)
    # Headline uses explicitly named decimal GB fields.
    assert "mlx_peak_gb=40.00" in line
    assert "process_footprint_peak_gb=52.50" in line
    assert "box_used_gb=63.50" in line
    assert "includes file cache" in line


def test_memory_headline_handles_missing_memory_block():
    ab = _ab()
    # No memory block (e.g. a defensive None): the headline still renders.
    line = ab._memory_headline({"peak_gb": 12.0})
    assert "mlx_peak_gb=n/a" in line
    assert "process_footprint_peak_gb=n/a" in line
    assert "box_used_gb=n/a" in line


@pytest.mark.skipif(sys.platform != "darwin", reason="live macOS footprint probe")
def test_sampler_peak_rss_is_high_water_not_exit_value():
    """The sampler must report the PEAK over its window, not the value at stop:
    allocate, let the sampler catch it, drop the reference, and the recorded peak
    must still reflect the allocation (proves peak-over-prefill+decode semantics)."""
    bench = _bench()
    probe = bench._MLXMemProbe(_FakeMx(1))
    sampler = probe.new_sampler(interval_s=0.02)
    sampler.start()
    blob = bytearray(220 * 1024 * 1024)
    for i in range(0, len(blob), 4096):
        blob[i] = 1
    time.sleep(0.15)
    peak_at_alloc = sampler.peak_rss_bytes
    del blob  # drop the allocation; the recorded peak must NOT fall back
    time.sleep(0.15)
    sampler.stop()
    assert sampler.peak_rss_bytes >= peak_at_alloc
    # and the peak genuinely captured the allocation (well above an empty baseline).
    assert sampler.peak_rss_bytes > 100 * 1024 * 1024


def test_memory_block_unstarted_sampler_does_not_substitute_lifetime_rss():
    bench = _bench()
    probe = bench._MLXMemProbe(_FakeMx(4 * 1024 * 1024))
    block = probe.memory_block(probe.new_sampler())
    assert block["process_footprint_peak_gb"] is None
    assert block["ru_maxrss_gb"] > 0
