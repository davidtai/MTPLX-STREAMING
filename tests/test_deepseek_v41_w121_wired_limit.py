"""W121: the DSV4.1 CLI/bench memory cap WIRES the Metal working set.

Root cause (live idle-box reconciliation, 2026-09-12): the served path
(``mtplx.server.openai._apply_metal_memory_caps``) sets BOTH ``mx.set_memory_limit``
AND ``mx.set_wired_limit``, so the resident server's Metal pages land in the
kernel ``wire_count`` bucket (visible to ``top`` and the guard).  The DSV4.1
CLI/bench path (``expert_runtime.apply_mlx_memory_cap``) historically set ONLY
``set_memory_limit``, so its IOAccelerator (GPU) pages were NOT wired -- they sat
in the active/inactive LRU and were compressed/swapped under pressure.  In window
46 that hid ~40 GB from the vm_stat guard and the box went over the 110 GB hard
limit before the operator aborted.

``apply_mlx_memory_cap`` now also calls ``set_wired_limit(limit)`` (the SAME value
handed to ``set_memory_limit``), mirroring the server and ``mtplx.glm52_q1t_over10``.

CPU-only: a fake ``mx`` exercises the code path; no Metal, no model, no MLX import.
"""
from __future__ import annotations

import types

from mtplx.expert_runtime import apply_mlx_memory_cap

GIB = 1024**3


def _plan(*, total=100_000, reserve=10_000, io=5_000):
    """Minimal object with the fields reconcile_mlx_memory_cap / apply read."""
    return types.SimpleNamespace(
        total_limit_bytes=total,
        runtime_reserve_bytes=reserve,
        io_staging_bytes=io,
        fixed_bytes=reserve + io + 1,
        persistent_cache_bytes=0,
        mmap_islands_wired=True,
        mmap_island_bytes=0,
    )


class _FakeMX:
    def __init__(self):
        self.mem = None
        self.wired = None
        self.prev_wired = 7  # a nonzero "previous" so the readback is exercised

    def set_memory_limit(self, value):
        self.mem = int(value)
        return 0

    def set_wired_limit(self, value):
        prev = self.prev_wired
        self.wired = int(value)
        self.prev_wired = int(value)
        return prev


def test_cap_sets_wired_limit_equal_to_allocation_limit():
    mx = _FakeMX()
    env: dict[str, str] = {}
    report = apply_mlx_memory_cap(_plan(), mx_module=mx, env=env)
    limit = 100_000 - 10_000 - 5_000  # 85_000
    assert mx.mem == limit
    assert mx.wired == limit  # wired == the allocation cap, so Metal is bounded + wired
    assert report["applied"] is True
    assert report["limit"] == limit
    assert report["wired_limit_applied"] is True
    assert report["wired_limit_bytes"] == limit
    assert report["wired_limit_api"] == "mx.set_wired_limit"
    assert report["previous_wired_limit_bytes"] == 7


def test_cap_wired_limit_falls_back_to_metal_namespace():
    captured = {}

    class _Metal:
        @staticmethod
        def set_memory_limit(value):
            captured["mem"] = int(value)
            return 0

        @staticmethod
        def set_wired_limit(value):
            captured["wired"] = int(value)
            return None

    mx = types.SimpleNamespace(metal=_Metal())
    report = apply_mlx_memory_cap(_plan(), mx_module=mx, env={})
    assert captured["wired"] == 85_000
    assert report["wired_limit_applied"] is True
    assert report["wired_limit_api"] == "mx.metal.set_wired_limit"
    assert report["previous_wired_limit_bytes"] is None  # setter returned None


def test_cap_survives_missing_wired_api_backward_compatible():
    """An older MLX with no set_wired_limit must NOT fail the run; the allocation
    limit still applies and the wired outcome is reported, not raised."""

    class _OldMX:
        def __init__(self):
            self.mem = None

        def set_memory_limit(self, value):
            self.mem = int(value)
            return 0

    mx = _OldMX()
    report = apply_mlx_memory_cap(_plan(), mx_module=mx, env={})
    assert mx.mem == 85_000
    assert report["applied"] is True
    assert report["limit"] == 85_000
    assert report["wired_limit_applied"] is False
    assert report["wired_limit_reason"] == "set_wired_limit_unavailable"


def test_cap_wired_limit_os_refusal_is_reported_not_raised():
    """If the OS/driver refuses the wired limit, the run continues (memory limit
    still applied) and the error is recorded."""

    class _RefusingMX:
        def __init__(self):
            self.mem = None

        def set_memory_limit(self, value):
            self.mem = int(value)
            return 0

        def set_wired_limit(self, value):
            raise RuntimeError("wired limit refused by driver")

    mx = _RefusingMX()
    report = apply_mlx_memory_cap(_plan(), mx_module=mx, env={})
    assert mx.mem == 85_000
    assert report["applied"] is True
    assert report["wired_limit_applied"] is False
    assert "wired limit refused" in report["wired_limit_error"]
