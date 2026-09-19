"""Tiny CPU-only unit tests for the overlap-schedule DES core.

No MLX, no mtplx, no trace file: exercises simulate() on a hand-built miss
stream so the discrete-event invariants are checked in isolation.
Run: nice -n 19 python -m pytest tests/test_dsv41_overlap_schedule_sim.py
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sim = importlib.import_module("scripts.deepseek_v41.overlap_schedule_sim")

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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
