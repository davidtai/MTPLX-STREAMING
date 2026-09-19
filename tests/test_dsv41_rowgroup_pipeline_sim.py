"""Tiny CPU-only unit tests for the F10 row-group pipeline DES core.

No MLX.  The DES-core tests use hand-built group streams (no mtplx, no trace); the
cache-replay tests are guarded and skip if mtplx/the trace are unavailable.
Run: nice -n 19 python -m pytest tests/test_dsv41_rowgroup_pipeline_sim.py -q
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sim = importlib.import_module("scripts.deepseek_v41.rowgroup_pipeline_sim")

RATE, RB = 12.9, sim.REC_BYTES
RD = RB / (RATE * 1e9)
CYC = 5


def _group(rec, ma, nassign, cyc=CYC):
    return {"rec": np.full((cyc, 40), rec, np.int32),
            "ma": np.full((cyc, 40), ma, np.int32), "nassign": nassign}


def _sim(groups, **kw):
    base = dict(g_dur_ms=2.05, c_assign_ms=0.035, rate_gbps=RATE, rec_bytes=RB,
                host_pre_ms=0.35, host_post_ms=0.70, extra_per_cycle_ms=0.0,
                growth_s=0.0)
    base.update(kw)
    return sim.simulate(groups, CYC, **base)


def test_records_conserved_and_ssd_busy_exact():
    g = [_group(3, 4, 36)]
    r = _sim(g)
    assert r["read_records"] == CYC * 40 * 3
    assert abs(r["ssd_busy_s"] - r["read_records"] * RD) < 1e-9
    assert abs(r["ssd_busy_frac"] - r["ssd_busy_s"] / r["total_decode_s"]) < 1e-12


def test_gpu_busy_equals_analytic_one_and_two_groups():
    # Every assignment is computed once per layer (hit+miss = nassign), plus G.
    g1 = [_group(2, 5, 36)]
    r1 = _sim(g1, g_dur_ms=2.05, c_assign_ms=0.035)
    exp1 = CYC * 40 * (2.05 + 36 * 0.035) / 1e3
    assert abs(r1["gpu_busy_s"] - exp1) < 1e-9
    g2 = [_group(2, 3, 18), _group(2, 3, 18)]
    r2 = _sim(g2, g_dur_ms=1.5, c_assign_ms=0.035)
    exp2 = CYC * 40 * (2 * 1.5 + 36 * 0.035) / 1e3
    assert abs(r2["gpu_busy_s"] - exp2) < 1e-9


def test_schedule_closed_form_when_reads_free():
    # rate huge -> rd ~ 0, c_assign 0: total is purely the schedule skeleton.
    g = [_group(3, 0, 36)]
    r = _sim(g, rate_gbps=1e9, c_assign_ms=0.0, g_dur_ms=2.05,
             host_pre_ms=0.35, host_post_ms=0.70)
    per_cycle = (sim.T_DRAFT + sim.T_ACCEPT + sim.T_COMMIT
                 + 40 * (2.05 + 0.35 + 0.70) / 1e3)
    assert abs(r["total_decode_s"] - CYC * per_cycle) < 1e-6
    assert r["gpu_busy_s"] > 0 and r["ssd_busy_s"] < 1e-6


def test_pipeline_beats_control_when_read_bound():
    # Same total demand records (control 4/layer == two groups 2+2), reads dominate.
    ctrl = _sim([_group(4, 4, 36)], g_dur_ms=2.05)
    pipe = _sim([_group(2, 2, 18), _group(2, 2, 18)], g_dur_ms=1.5,
                extra_per_cycle_ms=3.0)
    assert pipe["total_decode_s"] < ctrl["total_decode_s"]
    # the pipeline exposes less read wait than the fully-serial control
    assert pipe["host_blocked_finish_s"] < ctrl["host_blocked_finish_s"]


def test_monotonic_in_ghalf_and_rate():
    base = [_group(2, 2, 18), _group(2, 2, 18)]
    totals = [_sim(base, g_dur_ms=gh, extra_per_cycle_ms=3.0)["total_decode_s"]
              for gh in (1.2, 1.5, 1.8, 2.05)]
    assert totals == sorted(totals)                     # more GPU work -> slower
    faster = _sim(base, rate_gbps=18.0, extra_per_cycle_ms=3.0)["total_decode_s"]
    slower = _sim(base, rate_gbps=12.9, extra_per_cycle_ms=3.0)["total_decode_s"]
    assert faster < slower                              # faster drive -> faster


def test_gpu_bound_flag_tracks_busy():
    r = _sim([_group(2, 2, 18), _group(2, 2, 18)], g_dur_ms=2.05, c_assign_ms=0.045,
             extra_per_cycle_ms=3.0)
    assert r["gpu_bound"] == (r["gpu_busy_s"] > r["ssd_busy_s"])


def test_zero_miss_group_has_empty_reads():
    # a group with no misses contributes no SSD load but still computes E_hit
    r = _sim([_group(0, 0, 18), _group(3, 3, 18)], g_dur_ms=1.5, extra_per_cycle_ms=3.0)
    assert r["read_records"] == CYC * 40 * 3            # only group B reads
    assert r["gpu_busy_s"] > 0                          # E_hit still runs


def test_thinning_deterministic_and_reduces():
    arrs = [np.full((CYC, 40), 10, np.int32)]
    a = sim.thin_records(arrs, 0.2, sim.THIN_SEED)[0]
    b = sim.thin_records(arrs, 0.2, sim.THIN_SEED)[0]
    assert np.array_equal(a, b)                         # deterministic
    assert a.sum() < arrs[0].sum()                      # fewer records
    assert (a <= arrs[0]).all()
    assert sim.thin_records(arrs, 0.0, sim.THIN_SEED) is arrs   # no-op at 0


# ---------------------------------------------------------------------------
# Cache-replay tests (guarded: need mtplx + the route trace, run from worktree root)
# ---------------------------------------------------------------------------
try:
    import runpy
    _restore = runpy.run_path(str(sim.osim.HELPER))["restore"]
    _trace = sim.osim.load_trace()
except Exception:
    _restore = _trace = None
needs_trace = pytest.mark.skipif(_trace is None,
                                 reason="mtplx/route trace not available")


@needs_trace
def test_single_route_reproduces_f1_records():
    r = sim.replay(_trace, _restore, 111, n_groups=1)
    assert r["total_records"] == 31636                  # exact f1 control anchor


@needs_trace
def test_two_group_split_changes_records_and_raises_zero_miss():
    one = sim.replay(_trace, _restore, 111, n_groups=1)
    two = sim.replay(_trace, _restore, 111, n_groups=2)
    assert two["total_records"] > one["total_records"]  # split reads more (+~2.9%)
    assert two["either_zero_fraction"] > one["zero_miss_group_fraction"][0]


@needs_trace
def test_control_reconciles_f1_anchor_within_2pct():
    one = sim.replay(_trace, _restore, 111, n_groups=1)
    two = sim.replay(_trace, _restore, 111, n_groups=2)
    hp = sim.calibrate_host_post(one)
    ctrl, _ = sim.control_and_pipeline(one, two, g_half_ms=1.5, c_assign_ms=0.035,
                                       rate=12.9, rec_bytes=RB, host_post_ms=hp)
    assert abs(ctrl["total_decode_s"] - 74.667) / 74.667 < 0.02
    assert abs(ctrl["host_blocked_finish_s"] - 43.395) < 0.1   # == f1 read wait


def test_no_mlx_imported():
    assert not any(m == "mlx" or m.startswith("mlx.") for m in sys.modules)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok", name)
            except Exception as exc:                    # noqa: BLE001
                print("FAIL", name, exc)
