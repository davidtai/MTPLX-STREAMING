"""W121: the target-based MLX allocator limit (David's post-window-47 "rebalance").

Window 47 proved that with set_memory_limit ABOVE the steady active peak the
freed-buffer LRU holds across misses (decode 2.99 -> 5.49); at the plan it clears
on every miss.  So the allocator limit is derived from the TOTAL box target, with a
band reserved for the LRU so the cache can HOLD without pushing box_used past target:

  allocator_limit = box_target - baseline - allocator_cache_limit   (cache = 6 GiB)
  set_cache_limit  = allocator_cache_limit
  engine_budget    = allocator_limit - transient_band               (sizes the slots)

Then box_used = baseline + active + cache <= target at the DECODE PEAK by
construction (active <= engine_budget + transient_band = allocator_limit, cache <=
allocator_cache_limit).  target/baseline are decimal GB; cache/transient band are GiB.

CPU-only: pure helper + a fake mx; no Metal, no model.
"""
from __future__ import annotations

import types

import pytest

from mtplx.expert_runtime import (
    BOX_BASELINE_ENV,
    BOX_TARGET_ENV,
    DEFAULT_ALLOC_CACHE_GIB,
    DEFAULT_TRANSIENT_BAND_GIB,
    ExpertStreamingConfigurationError,
    apply_mlx_memory_cap,
    resolve_box_target_mlx_limit_bytes,
)

GB = 1_000_000_000
GIB = 1024**3
CACHE = int(round(DEFAULT_ALLOC_CACHE_GIB * GIB))   # 6 GiB
BAND = int(round(DEFAULT_TRANSIENT_BAND_GIB * GIB))  # 4.1 GiB


def _plan(*, total=100_000, reserve=10_000, io=5_000, fixed=None,
          persistent_slots=0, persistent_cache=0):
    return types.SimpleNamespace(
        total_limit_bytes=total,
        runtime_reserve_bytes=reserve,
        io_staging_bytes=io,
        fixed_bytes=(reserve + io + 1) if fixed is None else fixed,
        persistent_slots=persistent_slots,
        persistent_cache_bytes=persistent_cache,
        mmap_islands_wired=True,
        mmap_island_bytes=0,
    )


class _FakeMX:
    def __init__(self):
        self.mem = None
        self.wired = None
        self.cache = None

    def set_memory_limit(self, v):
        self.mem = int(v)
        return 0

    def set_wired_limit(self, v):
        self.wired = int(v)
        return 0

    def set_cache_limit(self, v):
        self.cache = int(v)
        return 0


# ---- resolve_box_target_mlx_limit_bytes -------------------------------------
def test_target_unset_returns_none():
    assert resolve_box_target_mlx_limit_bytes(env={}) is None


def test_target_minus_baseline_minus_cache_decimal_gb():
    r = resolve_box_target_mlx_limit_bytes(
        env={BOX_TARGET_ENV: "100", BOX_BASELINE_ENV: "11.06"}
    )
    # target - baseline (decimal GB) - allocator cache (GiB)
    assert r["mlx_limit_bytes"] == int(round((100 - 11.06) * GB)) - CACHE
    assert r["box_target_gb"] == 100.0
    assert abs(r["box_baseline_gb"] - 11.06) < 1e-9
    assert r["allocator_cache_limit_bytes"] == CACHE
    assert r["transient_band_bytes"] == BAND
    # engine budget (sizes the persistent slots) = allocator_limit - transient band
    assert r["engine_budget_bytes"] == r["mlx_limit_bytes"] - BAND


def test_target_default_100_when_marker_present():
    # Present-but-"default" arms the path at the 100 GB default.
    r = resolve_box_target_mlx_limit_bytes(
        env={BOX_TARGET_ENV: "default", BOX_BASELINE_ENV: "10"}
    )
    assert r["box_target_gb"] == 100.0
    assert r["mlx_limit_bytes"] == int(round(90 * GB)) - CACHE


def test_target_baseline_from_argument():
    r = resolve_box_target_mlx_limit_bytes(
        env={BOX_TARGET_ENV: "100"}, baseline_bytes=12 * GB
    )
    assert r["mlx_limit_bytes"] == int(round((100 - 12) * GB)) - CACHE


def test_cache_and_band_are_overridable():
    r = resolve_box_target_mlx_limit_bytes(
        env={
            BOX_TARGET_ENV: "100",
            BOX_BASELINE_ENV: "11",
            "MTPLX_DSV41_MLX_CACHE_LIMIT_GIB": "8",
            "MTPLX_DSV41_TRANSIENT_BAND_GIB": "5",
        }
    )
    assert r["allocator_cache_limit_bytes"] == 8 * GIB
    assert r["transient_band_bytes"] == 5 * GIB
    assert r["mlx_limit_bytes"] == int(round((100 - 11) * GB)) - 8 * GIB
    assert r["engine_budget_bytes"] == r["mlx_limit_bytes"] - 5 * GIB


def test_target_armed_without_baseline_raises():
    with pytest.raises(ExpertStreamingConfigurationError, match="baseline"):
        resolve_box_target_mlx_limit_bytes(env={BOX_TARGET_ENV: "100"})


def test_target_below_baseline_raises():
    with pytest.raises(ExpertStreamingConfigurationError, match="no MLX budget"):
        resolve_box_target_mlx_limit_bytes(
            env={BOX_TARGET_ENV: "10", BOX_BASELINE_ENV: "12"}
        )


def test_target_just_above_baseline_but_under_cache_raises():
    # target - baseline is positive (0.5 GB) but below the 6 GiB cache band -> no budget.
    with pytest.raises(ExpertStreamingConfigurationError, match="no MLX budget"):
        resolve_box_target_mlx_limit_bytes(
            env={BOX_TARGET_ENV: "12.5", BOX_BASELINE_ENV: "12"}
        )


# ---- apply_mlx_memory_cap with the target armed -----------------------------
def test_apply_uses_target_limit_wires_and_caps_and_keeps_engine_budget():
    mx = _FakeMX()
    env = {BOX_TARGET_ENV: "100", BOX_BASELINE_ENV: "11"}
    report = apply_mlx_memory_cap(_plan(), mx_module=mx, env=env)
    engine_budget = 100_000 - 10_000 - 5_000  # 85_000 (unchanged plan reconcile)
    mlx_limit = int(round((100 - 11) * GB)) - CACHE
    assert report["limit_source"] == "box_target"
    assert report["limit"] == mlx_limit
    assert mx.mem == mlx_limit          # allocator limit at target - baseline - cache
    assert mx.wired == mlx_limit        # wired at the same limit
    assert mx.cache == CACHE            # freed-buffer LRU bounded to the reserved band
    assert report["wired_limit_bytes"] == mlx_limit
    assert report["cache_limit_applied"] is True
    assert report["cache_limit_bytes"] == CACHE
    assert report["allocator_cache_limit_bytes"] == CACHE
    assert report["transient_band_bytes"] == BAND
    assert report["engine_budget_bytes"] == mlx_limit - BAND
    assert report["box_target_gb"] == 100.0
    assert report["box_target_mlx_limit_bytes"] == mlx_limit
    # The ENGINE budget env (bounds residents/KV/slots -- sized where the config is
    # built) stays the reconciled plan value here; only the soft ceiling moved.
    assert env["MTPLX_MEMORY_LIMIT_BYTES"] == str(engine_budget)


def test_apply_records_slot_derivation():
    # A plan with concrete slots so record_bytes is derivable.
    mx = _FakeMX()
    plan = _plan(fixed=40 * GB, persistent_slots=100, persistent_cache=200 * (1 << 20))
    env = {BOX_TARGET_ENV: "100", BOX_BASELINE_ENV: "11"}
    report = apply_mlx_memory_cap(plan, mx_module=mx, env=env)
    sd = report["slot_derivation"]
    mlx_limit = int(round((100 - 11) * GB)) - CACHE
    record = (200 * (1 << 20)) // 100
    assert sd["allocator_limit_bytes"] == mlx_limit
    assert sd["fixed_footprint_bytes"] == 40 * GB
    assert sd["transient_band_bytes"] == BAND
    assert sd["engine_budget_bytes"] == mlx_limit - BAND
    assert sd["record_bytes"] == record
    assert sd["persistent_slots_plan"] == 100
    # slots the target would support = (allocator_limit - fixed - band) / record
    expected = max(0, (mlx_limit - BAND) - 40 * GB) // record
    assert sd["persistent_slots_at_target"] == expected


def test_box_used_at_peak_stays_under_target():
    # baseline + active(<=engine_budget+band) + cache(<=cache) <= target, by construction.
    r = resolve_box_target_mlx_limit_bytes(
        env={BOX_TARGET_ENV: "100", BOX_BASELINE_ENV: "11"}
    )
    baseline_b = int(round(11 * GB))
    peak = baseline_b + r["engine_budget_bytes"] + r["transient_band_bytes"] + CACHE
    assert peak <= int(round(100 * GB))


def test_apply_target_plus_headroom_override():
    mx = _FakeMX()
    env = {
        BOX_TARGET_ENV: "100",
        BOX_BASELINE_ENV: "11",
        "MTPLX_DSV41_MLX_LIMIT_HEADROOM_GIB": "4",
    }
    report = apply_mlx_memory_cap(_plan(), mx_module=mx, env=env)
    mlx_limit = int(round((100 - 11) * GB)) - CACHE + 4 * GIB
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
    assert "slot_derivation" not in report
    assert mx.mem == 85_000
    # The legacy path leaves the allocator cache limit untouched.
    assert mx.cache is None
