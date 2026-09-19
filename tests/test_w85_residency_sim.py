"""CPU-only unit tests for scripts/deepseek_v41/w85_residency_sim.py.

No MLX, no GPU, no model load, no receipts read.  Pins the W85 plan/miss
arithmetic against the measured anchors (W36 real-manifest table, W81 served
split, window-32/W83 served counters, gate_0 verify union).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "w85_residency_sim",
    Path(__file__).resolve().parents[1] / "scripts" / "deepseek_v41" / "w85_residency_sim.py",
)
sim = importlib.util.module_from_spec(_SPEC)
sys.modules["w85_residency_sim"] = sim  # so @dataclass can resolve __module__
_SPEC.loader.exec_module(sim)  # type: ignore[union-attr]


def test_textonly_regime_calibrates_to_W36_table():
    # W36_PERSISTENT_SLOT_CAPACITY.md: real manifest, transient=6, kv=4096,
    # text-only (discounted) residents -> 72->79, 82->93, 92->108 slots/layer.
    to = sim.RESIDENT_BYTES_TEXTONLY
    assert sim.plan_slots(72, 6, 4096, resident_bytes=to).slots_per_layer == 79
    assert sim.plan_slots(82, 6, 4096, resident_bytes=to).slots_per_layer == 93
    assert sim.plan_slots(92, 6, 4096, resident_bytes=to).slots_per_layer == 108


def test_served_regime_matches_W81_split():
    # W81_VERIFY_SWITCH.md: served 60 GiB, full residents 17.37 GiB, ts=48,
    # kv=16384 -> 49 slots/layer (1960 total), 34.32 GiB persistent cache.
    p60 = sim.plan_slots(60, 48, 16384)  # default resident = SERVED
    assert p60.slots_per_layer == 49
    assert abs(p60.expert_cache_gib - 34.32) < 0.05
    assert abs(p60.resident_gib - 17.37) < 0.02
    assert abs(p60.transient_gib - 0.84) < 0.01  # GLOBAL pool, not x40
    # 80 GiB served -> 78 slots/layer (coordinator / W81).
    assert sim.plan_slots(80, 48, 16384).slots_per_layer == 78


def test_transient_is_nearly_free_not_34_gib():
    # Dropping transient 48->8 frees at most ~1 persistent slot/layer.
    base = sim.plan_slots(60, 48, 16384).slots_per_layer
    freed = sim.plan_slots(60, 8, 16384).slots_per_layer - base
    assert 0 <= freed <= 1


def test_program_capacity_feasibility():
    # 49 & 78 slots feasible (<=100 GB buffer); 120 slots infeasible (>110 hard).
    peak49 = sim.peak_gb_estimate(sim.slots_to_total_plan_gib(49))
    peak78 = sim.peak_gb_estimate(sim.slots_to_total_plan_gib(78))
    peak120 = sim.peak_gb_estimate(sim.slots_to_total_plan_gib(120))
    assert peak49 <= 100
    assert peak78 <= 100
    assert peak120 > 110


def test_ar_miss_matches_served_and_target_unreachable():
    # Served AR-16K: hit 0.741 -> ~61.9 miss/token (240 req/token).
    assert abs(sim.misses_per_token(0.741078) - sim.SERVED_AR_16K["miss_per_token"]) < 0.5
    # 20 tok/s @12.5 GiB/s needs hit >= ~0.85; the flat 16K hit (0.741) misses it.
    need_bytes = sim.SSD_CEILING_GIB_S * sim.GiB / 20
    need_miss = need_bytes / sim.BYTES_PER_MISS_ACCOUNTED
    need_hit = 1 - need_miss / sim.AR_REQ_PER_TOKEN
    assert need_hit > 0.84
    assert 0.741 < need_hit  # measured hit is below the requirement


def test_ssd_bound_ceiling_below_20_at_compulsory_floor():
    # At the compulsory 62-miss floor, even the 12.5 GiB/s SSD ceiling caps AR
    # decode well under 20 tok/s.
    bpt = sim.misses_per_token(0.741078) * sim.BYTES_PER_MISS_ACCOUNTED
    assert sim.ssd_bound_tok_s(bpt, bw_gib_s=sim.SSD_CEILING_GIB_S) < 12.0
    assert sim.ssd_bound_tok_s(bpt, bw_gb_s=sim.SSD_REALIZED_SWITCH_GB_S) < 9.0


def test_verify_union_monotone_and_depth5_needs_bigger_transient():
    u4 = sim._verify_union_mean(4)
    u6 = sim._verify_union_mean(6)
    assert sim._verify_union_mean(2) < sim._verify_union_mean(3) < u4 < u6
    assert abs(u4 - 18.69) < 0.1          # gate_0 measured
    assert 23.0 < u6 < 27.0               # extrapolated depth-5 union
    # depth-3 max union 24 -> ts>=24; depth-5 max 36 -> ts=24 breaks it.
    assert sim.VERIFY_ROWS[3] * sim.TOP_K == 24
    assert sim.VERIFY_ROWS[5] * sim.TOP_K == 36


def test_verify_120_target_needs_resident_hit_far_above_served():
    naive = sim._verify_union_mean(4) * sim.N_MOE_LAYERS
    need_frac = 1 - 120 / naive
    assert need_frac > 0.83          # need >=0.84 resident-hit
    served_frac = 1 - 330 / naive    # served ~0.56
    assert served_frac < 0.60
