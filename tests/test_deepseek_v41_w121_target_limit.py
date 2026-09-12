"""W121: the target-based MLX allocator limit (David's post-window-47 "rebalance").

Window 47 proved that with set_memory_limit ABOVE the steady active peak the
freed-buffer LRU holds across misses (decode 2.99 -> 5.49); at the plan it clears
on every miss.  So the allocator limit is derived from the TOTAL box target:
  mlx_limit = box_target - baseline   (both decimal GB)
which keeps box_used = baseline + process(active+cache) <= box_target by
construction.  The engine budget (MTPLX_MEMORY_LIMIT_BYTES) is UNCHANGED, so the
resident set / KV / expert-slot count and every byte of output are identical --
only the soft allocator ceiling moves.  Headroom stays an explicit add-on override.

CPU-only: pure helper + a fake mx; no Metal, no model.
"""
from __future__ import annotations

import types

import pytest

from mtplx.expert_runtime import (
    BOX_BASELINE_ENV,
    BOX_TARGET_ENV,
    ExpertStreamingConfigurationError,
    apply_mlx_memory_cap,
    resolve_box_target_mlx_limit_bytes,
)

GB = 1_000_000_000


def _plan(*, total=100_000, reserve=10_000, io=5_000):
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

    def set_memory_limit(self, v):
        self.mem = int(v)
        return 0

    def set_wired_limit(self, v):
        self.wired = int(v)
        return 0


# ---- resolve_box_target_mlx_limit_bytes -------------------------------------
def test_target_unset_returns_none():
    assert resolve_box_target_mlx_limit_bytes(env={}) is None


def test_target_minus_baseline_decimal_gb():
    r = resolve_box_target_mlx_limit_bytes(
        env={BOX_TARGET_ENV: "100", BOX_BASELINE_ENV: "11.06"}
    )
    assert r["mlx_limit_bytes"] == int(round((100 - 11.06) * GB))  # ~88.94 GB
    assert r["box_target_gb"] == 100.0
    assert abs(r["box_baseline_gb"] - 11.06) < 1e-9


def test_target_default_100_when_marker_present():
    # Present-but-"default" arms the path at the 100 GB default.
    r = resolve_box_target_mlx_limit_bytes(
        env={BOX_TARGET_ENV: "default", BOX_BASELINE_ENV: "10"}
    )
    assert r["box_target_gb"] == 100.0
    assert r["mlx_limit_bytes"] == int(round(90 * GB))


def test_target_baseline_from_argument():
    r = resolve_box_target_mlx_limit_bytes(
        env={BOX_TARGET_ENV: "100"}, baseline_bytes=12 * GB
    )
    assert r["mlx_limit_bytes"] == int(round((100 - 12) * GB))


def test_target_armed_without_baseline_raises():
    with pytest.raises(ExpertStreamingConfigurationError, match="baseline"):
        resolve_box_target_mlx_limit_bytes(env={BOX_TARGET_ENV: "100"})


def test_target_below_baseline_raises():
    with pytest.raises(ExpertStreamingConfigurationError, match="no MLX budget"):
        resolve_box_target_mlx_limit_bytes(
            env={BOX_TARGET_ENV: "10", BOX_BASELINE_ENV: "12"}
        )


# ---- apply_mlx_memory_cap with the target armed -----------------------------
def test_apply_uses_target_limit_and_keeps_engine_budget():
    mx = _FakeMX()
    env = {BOX_TARGET_ENV: "100", BOX_BASELINE_ENV: "11"}
    report = apply_mlx_memory_cap(_plan(), mx_module=mx, env=env)
    engine_budget = 100_000 - 10_000 - 5_000  # 85_000 (unchanged plan reconcile)
    mlx_limit = int(round((100 - 11) * GB))  # 89 GB
    assert report["limit_source"] == "box_target"
    assert report["limit"] == mlx_limit
    assert mx.mem == mlx_limit          # allocator limit at the target
    assert mx.wired == mlx_limit        # wired at the same target
    assert report["wired_limit_bytes"] == mlx_limit
    assert report["box_target_gb"] == 100.0
    assert report["box_target_mlx_limit_bytes"] == mlx_limit
    # The ENGINE budget (bounds residents/KV/slots) stays the plan value -> outputs
    # are byte-identical; only the soft ceiling moved up to the target.
    assert env["MTPLX_MEMORY_LIMIT_BYTES"] == str(engine_budget)


def test_apply_target_plus_headroom_override():
    mx = _FakeMX()
    env = {
        BOX_TARGET_ENV: "100",
        BOX_BASELINE_ENV: "11",
        "MTPLX_DSV41_MLX_LIMIT_HEADROOM_GIB": "4",
    }
    report = apply_mlx_memory_cap(_plan(), mx_module=mx, env=env)
    mlx_limit = int(round((100 - 11) * GB)) + 4 * (1024**3)
    assert report["limit"] == mlx_limit
    assert mx.mem == mlx_limit


def test_apply_target_below_engine_budget_raises():
    mx = _FakeMX()
    # A plan whose engine budget (~200 GB) exceeds the target-minus-baseline limit.
    big = _plan(total=200 * GB, reserve=1 * GB, io=0)
    env = {BOX_TARGET_ENV: "100", BOX_BASELINE_ENV: "11"}
    with pytest.raises(ExpertStreamingConfigurationError, match="BELOW the engine budget"):
        apply_mlx_memory_cap(big, mx_module=mx, env=env)


def test_apply_without_target_is_legacy_plan_path():
    mx = _FakeMX()
    env: dict[str, str] = {}
    report = apply_mlx_memory_cap(_plan(), mx_module=mx, env=env)
    assert report["limit_source"] == "plan"
    assert report["limit"] == 85_000
    assert "box_target_gb" not in report
    assert mx.mem == 85_000
