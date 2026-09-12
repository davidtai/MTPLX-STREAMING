"""CPU-pinned unit test for the W106 receipt ``memory`` block + background sampler.

Covers ``scripts/deepseek_v41/bench_standard_shape.py``:

  * ``_MemorySampler`` (a daemon-thread 1 Hz RSS + system-used sampler that touches
    no MLX and holds no lock), and
  * ``_MLXMemProbe.memory_block`` -- the block that fixes David's "peak_gb is the
    actual memory usage" misread by carrying ``mlx_peak_gb`` (the old ``peak_gb``),
    ``process_peak_rss_gb``, ``system_used_peak_gb`` and ``system_used_at_start_gb``.

The block machinery is MLX-agnostic (it reads ru_maxrss / ps / vm_stat, never the
allocator), so this test drives it with a FAKE mx that reports a chosen MLX peak.
MLX is imported ONLY to pin the default device to CPU
(memory/worker-tests-must-pin-mlx-cpu.md: MLX defaults to Metal). No GPU, no
model, no server, no network. Run under ``nice -n 19`` and without ``pytest -n
auto`` (host-encode sensitivity).
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import mlx.core as mx

# HARD rule: pin MLX to CPU before anything can touch Metal.
mx.set_default_device(mx.cpu)

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

    for key in (
        "mlx_peak_gb",
        "process_peak_rss_gb",
        "system_used_peak_gb",
        "system_used_at_start_gb",
    ):
        assert key in block, f"missing key {key!r} in memory block: {block}"
        assert isinstance(block[key], float)

    # mlx_peak_gb is exactly the fake allocator peak.
    assert block["mlx_peak_gb"] == (8 * 1024 * 1024) / GIB

    # The core invariant: process RSS peak includes the non-Metal footprint the MLX
    # peak omits, so it is >= mlx_peak_gb ...
    assert block["process_peak_rss_gb"] >= block["mlx_peak_gb"]
    # ... and here, with a tiny fake MLX peak, strictly greater (the ~200 MiB blob
    # + the interpreter dwarf 8 MiB), so the block is measuring real RSS.
    assert block["process_peak_rss_gb"] > block["mlx_peak_gb"]

    # Non-negative envelope figures (system_used is >0 on darwin, 0 elsewhere).
    assert block["system_used_peak_gb"] >= 0.0
    assert block["system_used_at_start_gb"] >= 0.0
    assert block["system_used_peak_gb"] >= block["system_used_at_start_gb"]

    del blob


def test_memory_block_without_sampler_still_builds():
    """With no sampler, the block still carries mlx_peak_gb and a process RSS peak
    (from ru_maxrss) >= it; system-used fields are 0 (unsampled)."""
    bench = _bench()
    probe = bench._MLXMemProbe(_FakeMx(4 * 1024 * 1024))
    block = probe.memory_block(None)
    assert set(block) == {
        "mlx_peak_gb",
        "process_peak_rss_gb",
        "system_used_peak_gb",
        "system_used_at_start_gb",
    }
    assert block["process_peak_rss_gb"] >= block["mlx_peak_gb"]
    assert block["system_used_peak_gb"] == 0.0
    assert block["system_used_at_start_gb"] == 0.0


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
