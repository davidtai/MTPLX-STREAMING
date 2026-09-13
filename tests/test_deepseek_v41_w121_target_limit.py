"""W121 target-based MLX allocator limit + HIGH-2/HIGH-A/HIGH-B red-team corrections.

  allocator_limit (set_memory_limit == set_wired_limit) = target - baseline - host_overhead
  set_cache_limit  = allocator_cache_limit   (the freed LRU, bounded ALONE)
  engine_budget    = allocator_limit - max(transient_band(peak-plan),
                                           active_overshoot(active_start-plan) + cache_room)

mx.set_memory_limit bounds active+cache JOINTLY, so the allocator limit reserves only the
host overhead the allocator never sees (footprint_peak - mlx_peak; HIGH-A: ~0.5 GiB from
W48's footprint receipt, not the 2.5 RSS estimate).  The engine budget reserves the WORST
of the prefill (engine + band) and decode (engine + active_overshoot + cache) regimes --
they are NOT additive (the LRU is inside mlx_peak).  target/baseline decimal GB;
host/cache/band/active GiB.

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
    DEFAULT_ACTIVE_OVERSHOOT_GIB,
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
ACTIVE = int(round(DEFAULT_ACTIVE_OVERSHOOT_GIB * GIB))  # 1.44 GiB (active_start - plan)
HOST = int(round(DEFAULT_HOST_OVERHEAD_GIB * GIB))   # 0.5 GiB (footprint - mlx_peak)
RESERVE = max(BAND, ACTIVE + CACHE)                  # max(5.54, 7.44) = 7.44 GiB


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
    # HIGH-2/HIGH-A: allocator limit reserves the HOST overhead only.
    assert r["mlx_limit_bytes"] == int(round((100 - 11.06) * GB)) - HOST
    assert r["host_overhead_bytes"] == HOST
    assert r["allocator_cache_limit_bytes"] == CACHE
    assert r["transient_band_bytes"] == BAND
    assert r["active_overshoot_bytes"] == ACTIVE
    # HIGH-A: engine budget = allocator - max(band, active + cache) (NON-additive).
    assert r["engine_reserve_bytes"] == RESERVE
    assert r["engine_budget_bytes"] == r["mlx_limit_bytes"] - RESERVE


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


def test_host_cache_band_active_are_overridable():
    r = resolve_box_target_mlx_limit_bytes(
        env={
            BOX_TARGET_ENV: "100", BOX_BASELINE_ENV: "11",
            "MTPLX_DSV41_HOST_OVERHEAD_GIB": "3",
            "MTPLX_DSV41_MLX_CACHE_LIMIT_GIB": "8",
            "MTPLX_DSV41_TRANSIENT_BAND_GIB": "5",
            "MTPLX_DSV41_ACTIVE_OVERSHOOT_GIB": "2",
        }
    )
    assert r["mlx_limit_bytes"] == int(round((100 - 11) * GB)) - 3 * GIB
    # engine reserve = max(band 5, active 2 + cache 8 = 10) = 10 GiB
    assert r["engine_budget_bytes"] == r["mlx_limit_bytes"] - 10 * GIB


def test_target_armed_without_baseline_raises():
    with pytest.raises(ExpertStreamingConfigurationError, match="baseline"):
        resolve_box_target_mlx_limit_bytes(env={BOX_TARGET_ENV: "100"})


def test_target_below_baseline_raises():
    with pytest.raises(ExpertStreamingConfigurationError, match="no MLX budget"):
        resolve_box_target_mlx_limit_bytes(
            env={BOX_TARGET_ENV: "10", BOX_BASELINE_ENV: "12"}
        )


def test_target_above_baseline_but_under_host_raises():
    # target - baseline is +0.3 GB but below the 0.5 GiB host overhead -> no budget.
    with pytest.raises(ExpertStreamingConfigurationError, match="no MLX budget"):
        resolve_box_target_mlx_limit_bytes(
            env={BOX_TARGET_ENV: "12.3", BOX_BASELINE_ENV: "12"}
        )


def test_reserve_over_allocator_raises_no_engine_budget():
    # allocator ~6.95 GiB positive but the engine reserve (7.44) exhausts it.
    with pytest.raises(ExpertStreamingConfigurationError, match="no engine budget"):
        resolve_box_target_mlx_limit_bytes(
            env={BOX_TARGET_ENV: "20", BOX_BASELINE_ENV: "12"}
        )


# ---- HIGH-A: the reviewer's W48-plan invariant + fill --------------------------
def test_w48_plan_fits_and_fills_target():
    # W48 (target 100, baseline 10.8138): engine 72.961, peak 78.242, active 74.141.
    r = resolve_box_target_mlx_limit_bytes(
        env={BOX_TARGET_ENV: "100", BOX_BASELINE_ENV: "10.8138"}
    )
    limit = r["mlx_limit_bytes"]
    engine = r["engine_budget_bytes"]
    peak_w48 = int(round(78.242 * GIB))
    active_w48 = int(round(74.141 * GIB))
    engine_w48 = int(round(72.961 * GIB))
    assert peak_w48 <= limit, "W48 peak must fit under the allocator limit"
    assert limit - active_w48 >= 5 * GIB, "decode cache room >= 5 GiB"
    assert engine >= engine_w48 - int(round(0.5 * GIB)), "must not hand back fewer slots than W48"
    # fill: the derivation must not leave the box under-used.
    fill = r["box_baseline_bytes"] + engine + r["transient_band_bytes"] + r["host_overhead_bytes"]
    assert fill >= int(round(0.97 * 100 * GB)), "derivation under-fills the target"


# ---- MEDIUM-8: assert against the REAL window-47 receipt ----------------------
_W47 = (
    Path(__file__).resolve().parents[1]
    / "docs/deepseek-v41/receipts/gpu-windows/window-47/ar-v2-attn.json"
)


@pytest.mark.skipif(not _W47.exists(), reason="window-47 receipt not present")
def test_derivation_holds_against_window47_receipt():
    m = json.loads(_W47.read_text())["memory"]
    peak = float(m["mlx_peak_gb"])
    active_start = float(m["mlx_active_gb_at_decode_start"])
    plan = float(m["plan_limit_gib_derived"])
    band_measured = peak - plan
    active_overshoot = active_start - plan
    assert DEFAULT_TRANSIENT_BAND_GIB >= band_measured, (
        f"transient band {DEFAULT_TRANSIENT_BAND_GIB} < measured peak-plan {band_measured:.3f}"
    )
    assert DEFAULT_ACTIVE_OVERSHOOT_GIB >= active_overshoot - 1e-6, (
        f"active overshoot {DEFAULT_ACTIVE_OVERSHOOT_GIB} < measured {active_overshoot:.3f}"
    )
    r = resolve_box_target_mlx_limit_bytes(
        env={BOX_TARGET_ENV: "100", BOX_BASELINE_ENV: "10.6"}
    )
    limit = r["mlx_limit_bytes"]
    engine = r["engine_budget_bytes"]
    assert engine + int(round(band_measured * GIB)) <= limit
    assert limit - (engine + int(round(active_overshoot * GIB))) >= 5 * GIB


# ---- apply_mlx_memory_cap ----------------------------------------------------
def test_apply_sets_limits_wires_and_caches_cache():
    mx = _FakeMX()
    env = {BOX_TARGET_ENV: "100", BOX_BASELINE_ENV: "11"}
    report = apply_mlx_memory_cap(_plan(), mx_module=mx, env=env)
    mlx_limit = int(round((100 - 11) * GB)) - HOST
    assert report["limit_source"] == "box_target"
    assert report["limit"] == mlx_limit
    assert mx.mem == mlx_limit
    assert mx.wired == mlx_limit
    assert mx.cache == CACHE
    assert report["cache_limit_applied"] is True
    assert report["host_overhead_bytes"] == HOST
    assert report["active_overshoot_bytes"] == ACTIVE
    assert report["engine_reserve_bytes"] == RESERVE
    assert report["engine_budget_bytes"] == mlx_limit - RESERVE


def test_apply_records_slot_derivation():
    mx = _FakeMX()
    plan = _plan(fixed=40 * GB, persistent_slots=100, persistent_cache=200 * (1 << 20))
    env = {BOX_TARGET_ENV: "100", BOX_BASELINE_ENV: "11"}
    sd = apply_mlx_memory_cap(plan, mx_module=mx, env=env)["slot_derivation"]
    mlx_limit = int(round((100 - 11) * GB)) - HOST
    engine = mlx_limit - RESERVE
    record = (200 * (1 << 20)) // 100
    assert sd["allocator_limit_bytes"] == mlx_limit
    assert sd["engine_budget_bytes"] == engine
    assert sd["engine_reserve_bytes"] == RESERVE
    assert sd["cache_room_bytes"] == CACHE
    assert sd["active_overshoot_bytes"] == ACTIVE
    assert sd["fixed_footprint_bytes_plan_estimate"] == 40 * GB
    assert sd["record_bytes"] == record
    assert sd["persistent_slots_at_target"] == max(0, (engine - 40 * GB)) // record


def test_apply_refuses_headroom_under_armed_target():
    # HIGH-B: headroom + armed target -> box_used above the target -> refuse.
    mx = _FakeMX()
    env = {BOX_TARGET_ENV: "100", BOX_BASELINE_ENV: "11",
           "MTPLX_DSV41_MLX_LIMIT_HEADROOM_GIB": "8"}
    with pytest.raises(ExpertStreamingConfigurationError, match="cannot be combined with an"):
        apply_mlx_memory_cap(_plan(), mx_module=mx, env=env)


def test_apply_headroom_still_works_without_target():
    # the legacy plan path still honors headroom (only the armed-target combo is refused).
    mx = _FakeMX()
    env = {"MTPLX_DSV41_MLX_LIMIT_HEADROOM_GIB": "4"}
    report = apply_mlx_memory_cap(_plan(), mx_module=mx, env=env)
    assert report["limit"] == 85_000 + 4 * GIB
    assert mx.mem == 85_000 + 4 * GIB


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
    assert mx.cache is None
