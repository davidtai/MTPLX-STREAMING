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


# ---------------------------------------------------------------------------
# What-if knobs (f1-stack-sim-20260919): plane-granular DES core.  Hand-built
# stream so the invariants are checked without MLX/mtplx/the capture.
# ---------------------------------------------------------------------------
CYC2 = 4
# misses on layers >=3 (like the capture); two experts each so predictions can be
# partly right and partly wrong, and the compute window leaves a plane in flight.
MISSES2 = [[[100 + L, 200 + ((c + L) % 5)] if L >= 3 else [] for L in range(40)]
           for c in range(CYC2)]


def _preds():
    """Predictions for window L -> target L+1: one true miss of L+1 plus one wrong
    id, as both the id-list (for simulate) and the scored list (for planes)."""
    ids, scored = {}, {}
    for c in range(CYC2):
        for L in range(3, 39):
            true = MISSES2[c][L + 1][:1]
            lst = true + [900 + L]                       # one correct, one wrong
            ids[(c, L)] = lst
            scored[(c, L)] = [(e, -float(i)) for i, e in enumerate(lst)]
    return ids, scored


def test_planes_reduce_to_record_model():
    """n_planes=1, preempt='record' must reproduce simulate() bit-for-bit for both
    control and the real arm at several ring sizes (locks the plane DES core)."""
    ids, scored = _preds()
    keys = ("total_decode_s", "read_wait_exposed_s", "demand_records",
            "spec_issued", "spec_useful", "spec_wasted")
    for ring in (8, 16, 32):
        old_c = sim.simulate(MISSES2, CYC2, arm="control", width=0, ring_size=ring,
                             pred_from=3, growth_s=0.0, **KW)
        new_c = sim.simulate_planes(MISSES2, CYC2, arm="control", n_planes=1,
                                    preempt="record", ring_size=ring, pred_from=3,
                                    growth_s=0.0, **KW)
        assert all(abs(old_c[k] - new_c[k]) < 1e-12 for k in keys)
        old_r = sim.simulate(MISSES2, CYC2, arm="real", width=8, ring_size=ring,
                             real=ids, pred_from=3, growth_s=0.0, **KW)
        new_r = sim.simulate_planes(MISSES2, CYC2, arm="real", scored=scored,
                                    n_planes=1, preempt="record", budget_k=None,
                                    ring_size=ring, pred_from=3, growth_s=0.0, **KW)
        assert all(abs(old_r[k] - new_r[k]) < 1e-12 for k in keys), (ring, old_r, new_r)


def test_preempt_monotone_and_useful_invariant():
    """Finer preemption never raises read wait (record >= plane >= full) and the
    useful/issued counts are identical across preemption modes (only the demand
    delay changes, not which speculative records land)."""
    _, scored = _preds()
    base = dict(scored=scored, n_planes=3, budget_k=3, ring_size=32,
                pred_from=3, growth_s=0.0, **KW)
    rec = sim.simulate_planes(MISSES2, CYC2, arm="real", preempt="record", **base)
    pln = sim.simulate_planes(MISSES2, CYC2, arm="real", preempt="plane", **base)
    ful = sim.simulate_planes(MISSES2, CYC2, arm="real", preempt="full", **base)
    assert rec["read_wait_exposed_s"] >= pln["read_wait_exposed_s"] - 1e-12
    assert pln["read_wait_exposed_s"] >= ful["read_wait_exposed_s"] - 1e-12
    assert rec["spec_useful"] == pln["spec_useful"] == ful["spec_useful"]
    assert rec["spec_issued"] == pln["spec_issued"] == ful["spec_issued"]


def test_window_stop_kills_inflight_and_recycle_monotone():
    """window_stop removes the in-flight-plane penalty (read wait <= plane preempt,
    and no worse than the ideal full-preempt bound); recycle extra-hidden reads are
    non-decreasing in ring size."""
    _, scored = _preds()
    base = dict(scored=scored, n_planes=3, budget_k=3, ring_size=32,
                pred_from=3, growth_s=0.0, **KW)
    pln = sim.simulate_planes(MISSES2, CYC2, arm="real", preempt="plane", **base)
    ful = sim.simulate_planes(MISSES2, CYC2, arm="real", preempt="full", **base)
    ws = sim.simulate_planes(MISSES2, CYC2, arm="real", preempt="plane",
                             window_stop=True, **base)
    assert ws["read_wait_exposed_s"] <= pln["read_wait_exposed_s"] + 1e-12
    assert ws["read_wait_exposed_s"] >= ful["read_wait_exposed_s"] - 1e-9
    # recycle counts rise (weakly) with a larger persistent per-layer ring
    rec = sim.recycle_analysis(MISSES2, scored, ring_sizes=(2, 8, 64), horizon=2,
                               budget_k=3, lo=0, hi=CYC2)
    extra = [rec[r]["extra_hidden_reads_heldout"] for r in (2, 8, 64)]
    assert extra == sorted(extra)
    assert rec[64]["ring_mem_mb"] == round(64 * sim.RING_MEM_MB_PER_REC, 1)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
