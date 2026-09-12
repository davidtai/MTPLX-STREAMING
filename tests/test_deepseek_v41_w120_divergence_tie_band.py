"""W120 -- magnitude-aware (bf16-ulp) DSpark divergence classifier.

W119 (docs/deepseek-v41/W119_EAGER_VERIFY_PARITY.md) proved the DSpark K+1-row
verify equals the 1-row AR forward BITWISE on CPU fp32; the GPU divergence at
window-45 index 111 (``ar_top2_margin`` 0.125, ``dspark_top2_margin`` 0.0,
max|Δlogit| 1.125, "class" divergent) is bf16 accumulation-order between the s=K+1
and s=1 eager einsum tiles quantized by the bf16 head -- every delta is an INTEGER
number of bf16 ulps.  W120 implements W119's recommended classifier rule so that
case is labelled ``tie_flip`` (rounding-class), while a genuine >ulp divergence
with clear margins stays ``divergent`` (loud).

These gates cover :func:`classify_divergence` (W120 rule) and its helpers
(:func:`_ulp_bf16`, :func:`_peak_contested_logit`, :func:`_tie_ulps_from_env`) in
:mod:`mtplx.models.deepseek_v41_dspark_decode`:

  1. ulp_bf16 lands on the W119 dyadic grid at every operating magnitude.
  2. window-45 (ar 0.125 / dspark 0.0 / Δ 1.125 / peak ~16) -> ``tie_flip``.
  3. a real divergence (clear margins, contested deltas that cannot close them)
     -> ``divergent`` -- both a small-delta case (margins 2.0/1.5, Δ 0.1) and a
     genuine large-delta swap whose deltas are NOT rounding-scale (the gate).
  4. the fixed-band legacy case (low magnitude, floor dominates) still ``tie_flip``.
  5. the near-tie may be on the authoritative VERIFY (dspark) side.
  6. the magnitude-aware band reclassifies a bf16-tie that the old fixed 3e-2 band
     called divergent.
  7. ``MTPLX_DSV41_DIVERGENCE_TIE_ULPS`` (k) is read AT USE; the ``tie_ulps`` arg
     wins; it is DELIBERATELY NOT a decode lever (out of ALL_LEVER_ENVS).
  8. receipt keys are additive JSON scalars; W77 keys are unchanged.

Self-contained: pure classifier (no model, no checkpoint, no GPU); CPU device so
any incidental mx.array conversion is fp32 bit-exact.
"""
import json

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


from mtplx.models.deepseek_v41_dspark_decode import (  # noqa: E402
    DSPARK_DIVERGENCE_TIE_ULPS_DEFAULT,
    DSPARK_DIVERGENCE_TIE_ULPS_ENV,
    DSPARK_TIE_MARGIN_DEFAULT,
    _peak_contested_logit,
    _tie_ulps_from_env,
    _ulp_bf16,
    classify_divergence,
)

VOCAB = 48


def _row(pairs, vocab=VOCAB, floor=-5.0):
    """A 1-D logits row of length ``vocab`` filled with ``floor`` except the
    ``{index: value}`` entries in ``pairs``."""
    r = np.full(vocab, floor, dtype=np.float32)
    for idx, val in pairs.items():
        r[idx] = val
    return r


# ---------------------------------------------------------------------------
# 1. _ulp_bf16 -- the W119 dyadic grid
# ---------------------------------------------------------------------------
def test_ulp_bf16_matches_w119_grid():
    # (|logit|, one bf16 ulp) rows straight out of the W119 verdict table.
    table = {
        8.0: 0.0625,
        16.0: 0.125,
        24.0: 0.125,   # same binade [16, 32)
        32.0: 0.25,
        48.0: 0.25,
        64.0: 0.5,
        96.0: 0.5,
        128.0: 1.0,
        192.0: 1.0,
        256.0: 2.0,
        16.125: 0.125,  # the window-45 peak magnitude
        1.0: 2.0 ** -7,
    }
    for mag, ulp in table.items():
        assert _ulp_bf16(mag) == pytest.approx(ulp, rel=0, abs=1e-12), mag
        # sign-independent
        assert _ulp_bf16(-mag) == pytest.approx(ulp, rel=0, abs=1e-12), mag
    # zero / non-finite -> 0.0 (caller falls back to the fixed floor)
    assert _ulp_bf16(0.0) == 0.0
    assert _ulp_bf16(float("inf")) == 0.0
    assert _ulp_bf16(float("nan")) == 0.0


def test_peak_contested_logit_uses_contested_tokens():
    ar = _row({5: 16.125, 9: 16.0})
    dsp = _row({5: 16.0, 9: 16.0})
    assert _peak_contested_logit(ar, dsp, 5, 9) == pytest.approx(16.125, abs=1e-6)
    # only one row available
    assert _peak_contested_logit(None, dsp, 5, 9) == pytest.approx(16.0, abs=1e-6)
    # no row -> None
    assert _peak_contested_logit(None, None, 5, 9) is None
    # out-of-range token indices -> fall back to the row top-1 magnitude
    assert _peak_contested_logit(ar, None, 999, None) == pytest.approx(16.125, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. window-45 case -> tie_flip (the W119 headline)
# ---------------------------------------------------------------------------
def test_window45_case_is_tie_flip():
    # ar top1 idx5=16.125, top2 idx9=16.0  -> ar_top2_margin 0.125 (1 bf16 ulp at 16)
    # a peak token idx20 carries the vocab-wide max|Δ| = 1.125 (dyadic, 9 ulps@16).
    ar = _row({5: 16.125, 9: 16.0, 20: 16.0})
    # verify row: idx5 and idx9 are bf16-IDENTICAL -> dspark_top2_margin 0.0 (a
    # genuine bf16 tie on the authoritative verify side); idx20 differs by 1.125.
    dsp = _row({5: 16.0, 9: 16.0, 20: 14.875})
    out = classify_divergence(
        index=111, ar_token=5, dspark_token=9,
        ar_logits_row=ar, dspark_logits_row=dsp,
    )
    assert out["class"] == "tie_flip"
    assert out["divergence_index"] == 111
    assert out["ar_top2_margin"] == pytest.approx(0.125, abs=1e-4)
    assert out["dspark_top2_margin"] == pytest.approx(0.0, abs=1e-6)
    assert out["max_abs_logit_delta"] == pytest.approx(1.125, abs=1e-4)
    assert out["peak_contested_logit"] == pytest.approx(16.125, abs=1e-4)
    assert out["ulp_bf16_at_peak"] == pytest.approx(0.125, abs=1e-6)
    assert out["tie_band_used"] == pytest.approx(0.375, abs=1e-6)  # max(0.03, 3*0.125)
    assert out["tie_ulps"] == DSPARK_DIVERGENCE_TIE_ULPS_DEFAULT == 3
    # fires by BOTH rule (a) (dspark margin 0.0 < 0.375) and rule (c).
    assert out["rounding_class_by_delta"] is True
    assert out["deltas_within_tie_band"] is True
    # the OLD fixed-band rule would have mislabelled this (ar margin 0.125 >= 0.03).
    assert out["ar_top2_margin"] > DSPARK_TIE_MARGIN_DEFAULT


# ---------------------------------------------------------------------------
# 3. real divergence -> divergent
# ---------------------------------------------------------------------------
def test_real_divergence_small_delta_is_divergent():
    # Clear margins (ar 2.0, dspark 1.5) with tiny contested-token deltas (Δ 0.1
    # total): rounding cannot close a 1.5 margin -> divergent.  The dominant top-1
    # is a non-contested token so the contested pair can carry a small delta.
    ar = _row({0: 4.0, 1: 2.0, 3: 0.0, 7: 0.0}, floor=0.0)
    dsp = _row({0: 4.0, 1: 2.5, 3: 0.05, 7: 0.05}, floor=0.0)
    out = classify_divergence(
        index=10, ar_token=3, dspark_token=7,
        ar_logits_row=ar, dspark_logits_row=dsp,
    )
    assert out["class"] == "divergent"
    assert out["ar_top2_margin"] == pytest.approx(2.0, abs=1e-4)
    assert out["dspark_top2_margin"] == pytest.approx(1.5, abs=1e-4)
    assert out["delta_at_ar_token"] == pytest.approx(0.05, abs=1e-4)
    assert out["delta_at_dspark_token"] == pytest.approx(0.05, abs=1e-4)
    assert out["rounding_class_by_delta"] is False  # 1.5 !<= 0.1


def test_genuine_swap_large_delta_is_divergent_via_gate():
    # A genuine argmax swap: ar picks 3, verify picks 7, both with a decisive 2.0
    # margin.  A real swap has contested deltas >= its margin, so the UNGATED
    # rule (c) (min margin <= delta sum) is TRUE -- but the closing deltas (2.0) are
    # 64 ulps at |logit|~4, NOT rounding-scale, so the gate keeps it DIVERGENT.
    ar = _row({3: 4.0, 7: 2.0}, floor=-10.0)
    dsp = _row({7: 4.0, 3: 2.0}, floor=-10.0)
    out = classify_divergence(
        index=27, ar_token=3, dspark_token=7,
        ar_logits_row=ar, dspark_logits_row=dsp,
    )
    assert out["class"] == "divergent"
    assert out["rounding_class_by_delta"] is True       # literal W119 signal fires
    assert out["deltas_within_tie_band"] is False       # ... but the gate blocks it
    assert out["ulp_bf16_at_peak"] == pytest.approx(0.03125, abs=1e-6)  # ulp@|4|
    assert out["delta_at_ar_token"] == pytest.approx(2.0, abs=1e-4)
    assert out["delta_at_dspark_token"] == pytest.approx(2.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 4. fixed-band legacy case still tie_flip (floor dominates at low magnitude)
# ---------------------------------------------------------------------------
def test_fixed_band_legacy_case_still_tie_flip():
    # Low-magnitude logits (~1): 3 * ulp_bf16(1.0) = 0.0234 < 0.03, so the fixed
    # DSPARK_TIE_MARGIN_DEFAULT floor dominates and the classic near-tie fires.
    ar = _row({3: 1.0, 7: 0.995})    # ar_top2_margin 0.005
    dsp = _row({7: 1.0, 3: 0.994})   # dspark_top2_margin 0.006
    out = classify_divergence(
        index=3, ar_token=3, dspark_token=7,
        ar_logits_row=ar, dspark_logits_row=dsp,
    )
    assert out["class"] == "tie_flip"
    assert out["ulp_bf16_at_peak"] == pytest.approx(2.0 ** -7, abs=1e-9)  # 0.0078125
    assert out["tie_band_used"] == pytest.approx(DSPARK_TIE_MARGIN_DEFAULT, abs=1e-9)
    assert out["tie_band_used"] == pytest.approx(0.03, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. near-tie on the authoritative VERIFY side (W119 part (a))
# ---------------------------------------------------------------------------
def test_verify_side_near_tie_absolves_when_ar_margin_wide():
    # ar margin 0.5 is ABOVE the band (0.375 at |logit|~16) -- the old ar-only rule
    # would call this divergent -- but the verify (dspark) margin 0.1 is BELOW it,
    # so the flip is rounding-class (the verify forward itself could not separate
    # the two tokens).
    ar = _row({3: 16.5, 7: 16.0})    # ar_top2_margin 0.5
    dsp = _row({7: 16.1, 3: 16.0})   # dspark_top2_margin 0.1
    out = classify_divergence(
        index=8, ar_token=3, dspark_token=7,
        ar_logits_row=ar, dspark_logits_row=dsp,
    )
    assert out["class"] == "tie_flip"
    assert out["ar_top2_margin"] == pytest.approx(0.5, abs=1e-4)
    assert out["dspark_top2_margin"] == pytest.approx(0.1, abs=1e-4)
    assert out["tie_band_used"] == pytest.approx(0.375, abs=1e-6)
    assert out["ar_top2_margin"] > out["tie_band_used"]     # AR side alone: divergent
    assert out["dspark_top2_margin"] < out["tie_band_used"]  # verify side: near-tie


# ---------------------------------------------------------------------------
# 6. magnitude-aware band reclassifies a bf16-tie the old fixed band missed
# ---------------------------------------------------------------------------
def test_magnitude_aware_band_reclassifies_bf16_tie():
    # At |logit|~16 a 1-ulp gap is 0.125 -- 4x the fixed 0.03 band.  The old rule
    # (ar margin 0.125 >= 0.03) said divergent; the magnitude-aware band (0.375)
    # correctly labels it a bf16 tie flip.
    ar = _row({3: 16.125, 7: 16.0})   # margin 0.125 (1 ulp @16)
    dsp = _row({7: 16.125, 3: 16.0})
    out = classify_divergence(
        index=0, ar_token=3, dspark_token=7,
        ar_logits_row=ar, dspark_logits_row=dsp,
    )
    assert out["class"] == "tie_flip"
    assert out["ar_top2_margin"] == pytest.approx(0.125, abs=1e-4)
    assert out["ar_top2_margin"] > DSPARK_TIE_MARGIN_DEFAULT   # old band would fail
    assert out["tie_band_used"] == pytest.approx(0.375, abs=1e-6)


# ---------------------------------------------------------------------------
# 7. k knob -- argument wins; env read AT USE; NOT a decode lever
# ---------------------------------------------------------------------------
def _k_case(**kw):
    # ar/dspark margins 0.2 at |logit|~16 (ulp 0.125): k=3 band 0.375 -> tie_flip;
    # k=0 band 0.03 -> divergent (deltas 0.2 are outside the 0.03 band -> gated).
    ar = _row({3: 16.2, 7: 16.0})
    dsp = _row({7: 16.2, 3: 16.0})
    return classify_divergence(
        index=1, ar_token=3, dspark_token=7,
        ar_logits_row=ar, dspark_logits_row=dsp, **kw,
    )


def test_tie_ulps_default_and_argument(monkeypatch):
    monkeypatch.delenv(DSPARK_DIVERGENCE_TIE_ULPS_ENV, raising=False)
    out = _k_case()
    assert out["tie_ulps"] == 3
    assert out["tie_band_used"] == pytest.approx(0.375, abs=1e-6)
    assert out["class"] == "tie_flip"
    # k=0 collapses the band to the fixed floor -> divergent
    out0 = _k_case(tie_ulps=0)
    assert out0["tie_ulps"] == 0
    assert out0["tie_band_used"] == pytest.approx(DSPARK_TIE_MARGIN_DEFAULT, abs=1e-9)
    assert out0["class"] == "divergent"


def test_tie_ulps_env_read_at_use(monkeypatch):
    monkeypatch.setenv(DSPARK_DIVERGENCE_TIE_ULPS_ENV, "0")
    assert _tie_ulps_from_env(None) == 0
    assert _k_case()["class"] == "divergent"       # env honoured at use
    monkeypatch.setenv(DSPARK_DIVERGENCE_TIE_ULPS_ENV, "10")
    assert _k_case()["tie_ulps"] == 10
    assert _k_case()["tie_band_used"] == pytest.approx(1.25, abs=1e-6)  # max(0.03,10*0.125)
    # the explicit argument WINS over the env
    assert _k_case(tie_ulps=3)["tie_ulps"] == 3
    # invalid env falls back to the default
    monkeypatch.setenv(DSPARK_DIVERGENCE_TIE_ULPS_ENV, "not-an-int")
    assert _tie_ulps_from_env(None) == DSPARK_DIVERGENCE_TIE_ULPS_DEFAULT


def test_env_knob_is_not_a_decode_lever():
    # A classifier setting must not leak into the decode-lever registry (it never
    # changes tokens produced).  Documented in W120_DIVERGENCE_TIE_BAND.md.
    ab = pytest.importorskip("scripts.deepseek_v41.ab_decode_env_levers")
    assert DSPARK_DIVERGENCE_TIE_ULPS_ENV == "MTPLX_DSV41_DIVERGENCE_TIE_ULPS"
    assert DSPARK_DIVERGENCE_TIE_ULPS_ENV not in ab.ALL_LEVER_ENVS


# ---------------------------------------------------------------------------
# 8. missing AR row stays conservative even with a hard verify tie
# ---------------------------------------------------------------------------
def test_missing_ar_row_stays_divergent_even_on_verify_tie():
    # A failed M=1 replay (ar row None) is NOT silently absolved, even when the
    # verify row shows a hard bf16 tie -- there is no reference to confirm the flip.
    dsp = _row({7: 16.0, 3: 16.0})   # dspark_top2_margin 0.0
    out = classify_divergence(
        index=5, ar_token=3, dspark_token=7,
        ar_logits_row=None, dspark_logits_row=dsp,
    )
    assert out["class"] == "divergent"
    assert out["ar_top2_margin"] is None
    assert out["max_abs_logit_delta"] is None
    assert out["dspark_top2_margin"] == pytest.approx(0.0, abs=1e-6)
    assert out["rounding_class_by_delta"] is None
    assert out["delta_at_ar_token"] is None


# ---------------------------------------------------------------------------
# receipt shape: additive JSON scalars, W77 keys preserved
# ---------------------------------------------------------------------------
def test_receipt_is_json_scalars_and_keeps_w77_keys():
    ar = _row({5: 16.125, 9: 16.0, 20: 16.0})
    dsp = _row({5: 16.0, 9: 16.0, 20: 14.875})
    out = classify_divergence(
        index=111, ar_token=5, dspark_token=9,
        ar_logits_row=ar, dspark_logits_row=dsp,
    )
    w77_keys = {
        "divergence_index", "ar_token", "dspark_token", "ar_top2_margin",
        "dspark_top2_margin", "max_abs_logit_delta", "tie_margin", "class",
    }
    w120_keys = {
        "tie_band_used", "tie_ulps", "peak_contested_logit", "ulp_bf16_at_peak",
        "delta_at_ar_token", "delta_at_dspark_token", "rounding_class_by_delta",
        "deltas_within_tie_band",
    }
    assert w77_keys <= set(out)          # every W77 key preserved (no rename)
    assert w120_keys <= set(out)         # every W120 key present
    # the W77 tie_margin key stays the FLOOR param, distinct from tie_band_used
    assert out["tie_margin"] == pytest.approx(DSPARK_TIE_MARGIN_DEFAULT, abs=1e-9)
    # all values are JSON scalars (round-trips cleanly, no numpy / arrays)
    for v in out.values():
        assert v is None or isinstance(v, (int, float, bool, str))
    json.dumps(out)
