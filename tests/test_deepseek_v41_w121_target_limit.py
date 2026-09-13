"""W121 target-based MLX allocator limit + HIGH-2 red-team corrections.

  allocator_limit (set_memory_limit == set_wired_limit) = target - baseline - host_overhead
  set_cache_limit  = allocator_cache_limit   (the freed LRU, bounded ALONE)
  engine_budget    = allocator_limit - transient_band(peak - plan) - cache_room

mx.set_memory_limit bounds active+cache JOINTLY, so the allocator limit only reserves the
host overhead the allocator never sees (footprint_peak - mlx_peak); the cache lives WITHIN
that limit.  process_footprint_peak = mlx_peak(<= allocator_limit) + host_overhead, so
box_used = baseline + footprint <= target.  The engine budget (persistent slots == the
plan) leaves room under the allocator limit for the prefill/peak overshoot (peak - plan =
5.54 GiB on W46/47) AND the LRU.  target/baseline are decimal GB; host/cache/band are GiB.

CPU-only: pure helper + a fake mx; no Metal, no model.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from mtplx.expert_runtime import (
    BOX_BASELINE_ENV,
    BOX_TARGET_ENV,
    DEFAULT_ALLOC_CACHE_GIB,
    DEFAULT_HOST_OVERHEAD_GIB,
    DEFAULT_TRANSIENT_BAND_GIB,
    ExpertStreamingConfigurationError,
    apply_mlx_memory_cap,
    resolve_box_target_mlx_limit_bytes,
)

GB = 1_000_000_000
GIB = 1024**3
CACHE = int(round(DEFAULT_ALLOC_CACHE_GIB * GIB))    # 6 GiB
BAND = int(round(DEFAULT_TRANSIENT_BAND_GIB * GIB))  # 5.54 GiB (peak - plan)
HOST = int(round(DEFAULT_HOST_OVERHEAD_GIB * GIB))   # 2.5 GiB (footprint - mlx_peak)


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
        self.mem = int(v); return 0

    def set_wired_limit(self, v):
        self.wired = int(v); return 0

    def set_cache_limit(self, v):
        self.cache = int(v); return 0


# ---- resolve_box_target_mlx_limit_bytes -------------------------------------
def test_target_unset_returns_none():
    assert resolve_box_target_mlx_limit_bytes(env={}) is None


def test_allocator_limit_is_target_minus_baseline_minus_host():
    r = resolve_box_target_mlx_limit_bytes(
        env={BOX_TARGET_ENV: "100", BOX_BASELINE_ENV: "11.06"}
    )
    # HIGH-2: allocator limit reserves the HOST overhead, NOT the cache.
    assert r["mlx_limit_bytes"] == int(round((100 - 11.06) * GB)) - HOST
    assert r["host_overhead_bytes"] == HOST
    assert r["allocator_cache_limit_bytes"] == CACHE
    assert r["transient_band_bytes"] == BAND
    # engine budget leaves room for BOTH the peak overshoot AND the LRU.
    assert r["engine_budget_bytes"] == r["mlx_limit_bytes"] - BAND - CACHE


def test_target_default_100_when_marker_present():
    r = resolve_box_target_mlx_limit_bytes(
        env={BOX_TARGET_ENV: "default", BOX_BASELINE_ENV: "10"}
    )
    assert r["box_target_gb"] == 100.0
    assert r["mlx_limit_bytes"] == int(round(90 * GB)) - HOST


def test_target_baseline_from_argument():
    r = resolve_box_target_mlx_limit_bytes(
        env={BOX_TARGET_ENV: "100"}, baseline_bytes=12 * GB
    )
    assert r["mlx_limit_bytes"] == int(round((100 - 12) * GB)) - HOST


def test_host_cache_band_are_overridable():
    r = resolve_box_target_mlx_limit_bytes(
        env={
            BOX_TARGET_ENV: "100", BOX_BASELINE_ENV: "11",
            "MTPLX_DSV41_HOST_OVERHEAD_GIB": "3",
            "MTPLX_DSV41_MLX_CACHE_LIMIT_GIB": "8",
            "MTPLX_DSV41_TRANSIENT_BAND_GIB": "5",
        }
    )
    assert r["mlx_limit_bytes"] == int(round((100 - 11) * GB)) - 3 * GIB
    assert r["engine_budget_bytes"] == r["mlx_limit_bytes"] - 5 * GIB - 8 * GIB


def test_target_armed_without_baseline_raises():
    with pytest.raises(ExpertStreamingConfigurationError, match="baseline"):
        resolve_box_target_mlx_limit_bytes(env={BOX_TARGET_ENV: "100"})


def test_target_below_baseline_raises():
    with pytest.raises(ExpertStreamingConfigurationError, match="no MLX budget"):
        resolve_box_target_mlx_limit_bytes(
            env={BOX_TARGET_ENV: "10", BOX_BASELINE_ENV: "12"}
        )


def test_target_above_baseline_but_under_host_raises():
    # target - baseline is +2 GB but below the 2.5 GiB host overhead -> no budget.
    with pytest.raises(ExpertStreamingConfigurationError, match="no MLX budget"):
        resolve_box_target_mlx_limit_bytes(
            env={BOX_TARGET_ENV: "14", BOX_BASELINE_ENV: "12"}
        )


def test_band_plus_cache_over_allocator_raises_no_engine_budget():
    # allocator limit positive but band + cache exhaust it -> no engine budget.
    with pytest.raises(ExpertStreamingConfigurationError, match="no engine budget"):
        resolve_box_target_mlx_limit_bytes(
            env={BOX_TARGET_ENV: "20", BOX_BASELINE_ENV: "12"}  # alloc ~5.1 GiB < band+cache
        )


# ---- the reviewer's W47-numbers invariant (MEDIUM-8) ------------------------
def test_w47_numbers_peak_fits_and_cache_room_ge_5gib():
    # W46/47: plan 69.18, active_start 70.62, mlx_peak 74.72 -> peak-plan 5.54,
    # active_start-plan 1.44.  With target 100 / baseline 10.6 the derived engine budget
    # must satisfy: prefill peak (engine + 5.54) <= allocator limit, and the decode cache
    # room (limit - (engine + 1.44)) >= 5 GiB (W47 hr8 held 5.0).
    r = resolve_box_target_mlx_limit_bytes(
        env={BOX_TARGET_ENV: "100", BOX_BASELINE_ENV: "10.6"}
    )
    limit = r["mlx_limit_bytes"]
    engine = r["engine_budget_bytes"]
    peak = engine + int(round(5.54 * GIB))
    assert peak <= limit, "prefill peak must stay under the allocator limit"
    cache_room = limit - (engine + int(round(1.44 * GIB)))
    assert cache_room >= 5 * GIB, "decode LRU must have >= 5 GiB room"
    # and box_used at the peak stays <= target: baseline + mlx_peak(<=limit) + host.
    baseline_b = int(round(10.6 * GB))
    assert baseline_b + limit + HOST <= int(round(100 * GB)) + GIB  # within rounding


# ---- MEDIUM-8: assert the derivation against the REAL window-47 receipt ----------
_W47 = (
    Path(__file__).resolve().parents[1]
    / "docs/deepseek-v41/receipts/gpu-windows/window-47/ar-v2-attn.json"
)


@pytest.mark.skipif(not _W47.exists(), reason="window-47 receipt not present")
def test_derivation_holds_against_window47_receipt():
    m = json.loads(_W47.read_text())["memory"]
    peak = float(m["mlx_peak_gb"])            # 74.565 GiB
    active_start = float(m["mlx_active_gb_at_decode_start"])  # 70.620 GiB
    plan = float(m["plan_limit_gib_derived"])  # 69.178 GiB
    band_measured = peak - plan               # ~5.39 GiB (the unpriced-above-plan band)
    active_overshoot = active_start - plan     # ~1.44 GiB (active at decode start over plan)
    # (ii) the default transient band must COVER the measured peak-plan overshoot, so a
    # prefill peak of engine + band_measured stays under the allocator limit.
    assert DEFAULT_TRANSIENT_BAND_GIB >= band_measured, (
        f"transient band {DEFAULT_TRANSIENT_BAND_GIB} < measured peak-plan {band_measured:.3f}"
    )
    # with the target armed at the same box the window ran, the engine budget derived
    # from the RECEIPT numbers keeps the prefill peak under the limit AND leaves >= 5 GiB
    # of decode cache room (W47 hr8 held 5.0).
    r = resolve_box_target_mlx_limit_bytes(
        env={BOX_TARGET_ENV: "100", BOX_BASELINE_ENV: "10.6"}
    )
    limit = r["mlx_limit_bytes"]
    engine = r["engine_budget_bytes"]
    assert engine + int(round(band_measured * GIB)) <= limit
    assert limit - (engine + int(round(active_overshoot * GIB))) >= 5 * GIB


# ---- apply_mlx_memory_cap ----------------------------------------------------
def test_apply_sets_limits_at_host_reserved_target_wires_and_caches_cache():
    mx = _FakeMX()
    env = {BOX_TARGET_ENV: "100", BOX_BASELINE_ENV: "11"}
    report = apply_mlx_memory_cap(_plan(), mx_module=mx, env=env)
    mlx_limit = int(round((100 - 11) * GB)) - HOST
    assert report["limit_source"] == "box_target"
    assert report["limit"] == mlx_limit
    assert mx.mem == mlx_limit          # allocator limit = target - baseline - host
    assert mx.wired == mlx_limit        # wired at the same limit
    assert mx.cache == CACHE            # LRU bounded ALONE to the cache room
    assert report["cache_limit_applied"] is True
    assert report["host_overhead_bytes"] == HOST
    assert report["engine_budget_bytes"] == mlx_limit - BAND - CACHE
    assert report["box_target_mlx_limit_bytes"] == mlx_limit


def test_apply_records_slot_derivation():
    mx = _FakeMX()
    plan = _plan(fixed=40 * GB, persistent_slots=100, persistent_cache=200 * (1 << 20))
    env = {BOX_TARGET_ENV: "100", BOX_BASELINE_ENV: "11"}
    sd = apply_mlx_memory_cap(plan, mx_module=mx, env=env)["slot_derivation"]
    mlx_limit = int(round((100 - 11) * GB)) - HOST
    engine = mlx_limit - BAND - CACHE
    record = (200 * (1 << 20)) // 100
    assert sd["allocator_limit_bytes"] == mlx_limit
    assert sd["engine_budget_bytes"] == engine
    assert sd["cache_room_bytes"] == CACHE
    assert sd["fixed_footprint_bytes_plan_estimate"] == 40 * GB
    assert sd["record_bytes"] == record
    assert sd["persistent_slots_at_target"] == max(0, (engine - 40 * GB)) // record


def test_apply_target_plus_headroom_override():
    mx = _FakeMX()
    env = {BOX_TARGET_ENV: "100", BOX_BASELINE_ENV: "11",
           "MTPLX_DSV41_MLX_LIMIT_HEADROOM_GIB": "4"}
    report = apply_mlx_memory_cap(_plan(), mx_module=mx, env=env)
    assert report["limit"] == int(round((100 - 11) * GB)) - HOST + 4 * GIB
    assert mx.mem == report["limit"]


def test_apply_target_below_engine_budget_raises():
    mx = _FakeMX()
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
    assert mx.cache is None  # legacy path leaves the cache limit untouched
