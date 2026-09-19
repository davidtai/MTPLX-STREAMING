"""Tiny CPU-only unit tests for the overlap-schedule DES core.

No MLX, no mtplx, no trace file: exercises simulate() on a hand-built miss
stream so the discrete-event invariants are checked in isolation.
Run: nice -n 19 python -m pytest tests/test_dsv41_overlap_schedule_sim.py
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sim = importlib.import_module("scripts.deepseek_v41.overlap_schedule_sim")
rrc = sim.rrc                                   # sibling re-scorer (numpy-only)

try:
    _CAP = rrc.load_capture()                   # None-guarded: skip if NPZ absent
except SystemExit:
    _CAP = None
needs_capture = pytest.mark.skipif(_CAP is None, reason="router capture NPZ not present")

# 3 cycles x 40 layers; a few misses on early layers so predictions have targets.
CYC = 3
MISSES = [[[10 + (c + L) % 5] if L % 3 == 0 else [] for L in range(40)]
          for c in range(CYC)]
RATE, RB = 12.9, sim.REC_BYTES
KW = dict(rate_gbps=RATE, rec_bytes=RB)


def _run(arm, **kw):
    return sim.simulate(MISSES, CYC, arm=arm, width=kw.pop("width", 0),
                        ring_size=kw.pop("ring_size", 32),
                        causal=kw.pop("causal", None),
                        synth=kw.pop("synth", None), **KW)


def test_control_read_wait_is_pure_demand():
    r = _run("control")
    rd = RB / (RATE * 1e9)
    assert r["spec_issued"] == 0 and r["spec_useful"] == 0
    assert abs(r["read_wait_exposed_s"] - r["demand_records"] * rd) < 1e-9
    assert r["hidden_fraction"] == 0.0


def test_issued_equals_useful_plus_wasted():
    for arm in ("oracle1", "oracle2", "cross_cycle"):
        r = _run(arm)
        assert r["spec_issued"] == r["spec_useful"] + r["spec_wasted"]


def test_oracle_is_lossless_and_reduces_read_wait():
    ctl = _run("control")
    for arm in ("oracle1", "oracle2"):
        r = _run(arm)
        assert r["spec_wasted"] == 0            # oracle never wrong
        assert r["spec_precision"] == 1.0
        assert r["read_wait_exposed_s"] <= ctl["read_wait_exposed_s"] + 1e-9
        assert r["total_decode_s"] < ctl["total_decode_s"]


def test_two_ahead_hides_at_least_one_ahead():
    assert _run("oracle2")["hidden_fraction"] >= _run("oracle1")["hidden_fraction"] - 1e-9


def test_synthetic_emits_wrong_and_correct_predictions():
    synth = sim.synthetic_predictions(MISSES, CYC)
    r = _run("synthetic", synth=synth)
    # synthetic predictor is imperfect: it wastes some reads (precision < 1).
    assert r["spec_issued"] > 0 and r["spec_wasted"] > 0
    assert r["spec_precision"] is None or 0.0 <= r["spec_precision"] < 1.0


def test_pred_from_gates_low_layers():
    # 1 cycle; the only miss is at target layer 2, predicted at window layer 1.
    cap_misses = [[[] for _ in range(40)]]
    cap_misses[0][2] = [50]
    real = {(0, 1): [50]}
    kw = dict(width=8, ring_size=32, real=real, growth_s=0.0, **KW)
    # pred_from=0 lets layer 1's window prefetch target 2 -> the miss is hidden.
    open_ = sim.simulate(cap_misses, 1, arm="real", pred_from=0, **kw)
    assert open_["spec_issued"] == 1 and open_["spec_useful"] == 1
    # pred_from=3 keeps layers 0-3 unpredicted -> nothing issued.
    gated = sim.simulate(cap_misses, 1, arm="real", pred_from=3, **kw)
    assert gated["spec_issued"] == 0 and gated["spec_useful"] == 0
    assert gated["read_wait_exposed_s"] > open_["read_wait_exposed_s"]


@needs_capture
def test_capture_curve_monotone_and_best_rule():
    for f in range(2):
        for rule in rrc.MERGE_RULES:
            rows = rrc.curve(_CAP, f, rule)["budgets"]
            cov = [r["heldout"]["miss_coverage"] for r in rows]
            prec = [r["heldout"]["precision"] for r in rows]
            assert cov == sorted(cov)                      # coverage rises with k
            assert prec == sorted(prec, reverse=True)      # precision falls with k
            assert cov[-1] <= rrc.coverage_ceiling(_CAP)["ceiling_coverage"] + 1e-9
    # post-attention router beats the pre-attention mean; max beats sum at k=1.
    def cov(f, rule, k):
        return next(r for r in rrc.curve(_CAP, f, rule)["budgets"]
                    if r["k"] == k)["heldout"]["miss_coverage"]
    assert cov(1, "max", 8) > cov(0, "max", 8)
    assert cov(1, "max", 1) > cov(1, "sum", 1)


@needs_capture
def test_capture_arm_control_real_oracle_ordering():
    res = sim.run_capture_comparison(RATE, RB)
    by = {(a["arm"], a["ring_size"]): a for a in res["arms"]}
    for ring in (16, 32, 64):
        c, r, o = by[("control", ring)], by[("real", ring)], by[("oracle1", ring)]
        assert c["total_decode_s"] > r["total_decode_s"] > o["total_decode_s"]
        assert o["spec_wasted"] == 0 and o["spec_precision"] == 1.0
        assert 0.0 < r["hidden_fraction"] < o["hidden_fraction"]
        assert 0.0 < r["spec_precision"] < 1.0                # real is imperfect
    assert res["captured_slots_per_layer"] == 105             # NOT 111
    e = res["extrapolation"]["real"]["ring32"]
    assert 0.0 < e["seconds_removed_from_control"] < sim.FULL_CONTROL_READ_WAIT_S


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
