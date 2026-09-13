"""W120 -- magnitude-aware (bf16-ulp) contested-token DSpark divergence classifier.

W119 (docs/deepseek-v41/W119_EAGER_VERIFY_PARITY.md) proved the DSpark K+1-row
verify equals the 1-row AR forward BITWISE on CPU fp32; the GPU divergence at
window-45 index 111 is bf16 accumulation-order quantized by the bf16 head. W120
implements the rounding-class rule; the W120 red-team then tightened it:

  * a flip is absolved ONLY when the two forwards differ at the two CONTESTED
    tokens by no more than the band (deltas_within_tie_band) -- a verify row's own
    tight top-2, or a decisive AR top-2 elsewhere, never absolves on its own;
  * the decision uses the CONTESTED margin (|row[ar_token]-row[dspark_token]|), NOT
    the row's top-1/top-2 gap;
  * the band's peak comes from the AR REFERENCE row only (a garbage verify logit
    must not widen it);
  * absolution needs a real AR reference (>=2 logits, both tokens in range);
  * MTPLX_DSV41_DIVERGENCE_TIE_ULPS accepts only a bounded non-negative integer.

Self-contained: pure classifier (no model, no checkpoint, no GPU); CPU device so
any incidental mx.array conversion is fp32/bf16 bit-exact.
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
    DSPARK_DIVERGENCE_TIE_ULPS_MAX,
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


def _cls(ar, dsp, ar_token, dspark_token, index=0, **kw):
    return classify_divergence(
        index=index, ar_token=ar_token, dspark_token=dspark_token,
        ar_logits_row=ar, dspark_logits_row=dsp, **kw,
    )


# ---------------------------------------------------------------------------
# 1. helpers
# ---------------------------------------------------------------------------
def test_ulp_bf16_matches_w119_grid():
    table = {
        8.0: 0.0625, 16.0: 0.125, 24.0: 0.125, 32.0: 0.25, 48.0: 0.25,
        64.0: 0.5, 96.0: 0.5, 128.0: 1.0, 192.0: 1.0, 256.0: 2.0,
        16.125: 0.125, 1.0: 2.0 ** -7,
    }
    for mag, ulp in table.items():
        assert _ulp_bf16(mag) == pytest.approx(ulp, rel=0, abs=1e-12), mag
        assert _ulp_bf16(-mag) == pytest.approx(ulp, rel=0, abs=1e-12), mag
    assert _ulp_bf16(0.0) == 0.0
    assert _ulp_bf16(float("inf")) == 0.0
    assert _ulp_bf16(float("nan")) == 0.0


def test_peak_contested_logit_uses_ar_row_only():
    ar = _row({5: 16.125, 9: 16.0})
    # helper takes the AR row only -- a garbage verify logit can never reach it.
    assert _peak_contested_logit(ar, 5, 9) == pytest.approx(16.125, abs=1e-6)
    assert _peak_contested_logit(None, 5, 9) is None
    # out-of-range tokens fall back to the AR row's top-1 magnitude
    assert _peak_contested_logit(ar, 999, None) == pytest.approx(16.125, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. window-45 case -> tie_flip (plausible reconstruction; see the doc)
# ---------------------------------------------------------------------------
def test_window45_reconstruction_is_tie_flip():
    # Reconstruction of window-45 index-111 with contested tokens at |logit|~16 and
    # rounding-scale contested deltas (the real receipt stores only scalars, so the
    # contested deltas are not recoverable from it -- see the doc's re-check note).
    ar = _row({5: 16.125, 9: 16.0, 20: 16.0})   # ar_top2_margin 0.125
    dsp = _row({5: 16.0, 9: 16.0, 20: 14.875})  # dspark_top2_margin 0.0; max|Δ| 1.125
    out = _cls(ar, dsp, ar_token=5, dspark_token=9, index=111)
    assert out["class"] == "tie_flip"
    assert out["ar_top2_margin"] == pytest.approx(0.125, abs=1e-4)
    assert out["dspark_top2_margin"] == pytest.approx(0.0, abs=1e-6)
    assert out["ar_contested_margin"] == pytest.approx(0.125, abs=1e-4)
    assert out["dspark_contested_margin"] == pytest.approx(0.0, abs=1e-6)
    assert out["max_abs_logit_delta"] == pytest.approx(1.125, abs=1e-4)
    assert out["peak_contested_logit"] == pytest.approx(16.125, abs=1e-4)
    assert out["ulp_bf16_at_peak"] == pytest.approx(0.125, abs=1e-6)
    assert out["tie_band_used"] == pytest.approx(0.375, abs=1e-6)   # max(0.03, 3*0.125)
    assert out["delta_at_ar_token"] == pytest.approx(0.125, abs=1e-4)
    assert out["delta_at_dspark_token"] == pytest.approx(0.0, abs=1e-6)
    assert out["deltas_within_tie_band"] is True


# ---------------------------------------------------------------------------
# 3. red-team divergence cases (verbatim) -- all must class "divergent"
# ---------------------------------------------------------------------------
def test_verify_tie_large_delta_is_divergent():
    # HIGH-1: a tight VERIFY top-2 (dspark tie 0.0) must NOT absolve when the
    # contested deltas are large (1.5/1.5 = 12 ulps @16).
    ar = _row({0: 20.0, 1: 17.0})
    dsp = _row({0: 18.5, 1: 18.5})
    out = _cls(ar, dsp, ar_token=0, dspark_token=1)
    assert out["class"] == "divergent"
    assert out["dspark_top2_margin"] == pytest.approx(0.0, abs=1e-6)  # verify tie
    assert out["delta_at_ar_token"] == pytest.approx(1.5, abs=1e-4)
    assert out["delta_at_dspark_token"] == pytest.approx(1.5, abs=1e-4)
    assert out["deltas_within_tie_band"] is False          # the gate blocks it
    assert out["tie_band_used"] == pytest.approx(0.375, abs=1e-6)


def test_shifted_verify_row_is_divergent():
    # A uniform +40 shift of the verify row: contested margins look tiny (0.125) but
    # every contested delta is 40.  Peak (band) must come from the AR row only.
    ar = _row({0: 16.125, 1: 16.0})
    dsp = (np.asarray(_row({0: 16.125, 1: 16.0})) + 40.0).astype(np.float32)
    out = _cls(ar, dsp, ar_token=0, dspark_token=1)
    assert out["class"] == "divergent"
    assert out["peak_contested_logit"] == pytest.approx(16.125, abs=1e-4)  # AR, not 56
    assert out["tie_band_used"] == pytest.approx(0.375, abs=1e-6)
    assert out["delta_at_ar_token"] == pytest.approx(40.0, abs=1e-4)
    assert out["deltas_within_tie_band"] is False


def test_uncontested_near_tie_is_divergent():
    # HIGH-2: the row's own top-2 gap (0.1, between tok0 and tok2) is a near-tie, but
    # the CONTESTED pair (tok0 vs tok1) is decisive (10.0) -- must be divergent.
    ar = _row({0: 20.0, 2: 19.9, 1: 10.0})
    dsp = _row({0: 20.0, 2: 19.9, 1: 25.0})
    out = _cls(ar, dsp, ar_token=0, dspark_token=1)
    assert out["class"] == "divergent"
    assert out["ar_top2_margin"] == pytest.approx(0.1, abs=1e-4)        # near-tie...
    assert out["ar_contested_margin"] == pytest.approx(10.0, abs=1e-4)  # ...but decisive
    assert out["delta_at_dspark_token"] == pytest.approx(15.0, abs=1e-4)
    assert out["deltas_within_tie_band"] is False


def test_garbage_verify_logit_does_not_self_absolve():
    # MEDIUM-1: a garbage +300 verify logit would (under a both-rows peak) widen the
    # band to 6.0 and absolve a real 3.125 contested delta.  Peak from the AR row
    # only keeps the band at 0.375 -> divergent.
    ar = _row({3: 16.125, 7: 16.0})
    dsp = _row({3: 13.0, 7: 16.0, 9: 300.0})
    out = _cls(ar, dsp, ar_token=3, dspark_token=7)
    assert out["class"] == "divergent"
    assert out["peak_contested_logit"] == pytest.approx(16.125, abs=1e-4)  # not 300
    assert out["tie_band_used"] == pytest.approx(0.375, abs=1e-6)
    assert out["delta_at_ar_token"] == pytest.approx(3.125, abs=1e-4)
    assert out["deltas_within_tie_band"] is False


def test_degenerate_ar_row_is_divergent():
    # MEDIUM-2: a partial/failed replay (empty or 1-element AR row) is not silently
    # absolved, even with a hard verify tie.
    dsp = _row({7: 16.0, 3: 16.0})   # dspark tie 0.0
    for ar in (np.array([], dtype=np.float32), np.array([5.0], dtype=np.float32)):
        out = _cls(ar, dsp, ar_token=3, dspark_token=7)
        assert out["class"] == "divergent"
        assert out["ar_top2_margin"] is None
        assert out["ar_contested_margin"] is None


# ---------------------------------------------------------------------------
# 4. legitimate tie flips still fire (contested deltas are rounding-scale)
# ---------------------------------------------------------------------------
def test_magnitude_aware_band_reclassifies_bf16_tie():
    # A genuine 1-ulp bf16 tie at |logit|~16: contested margins 0.125, contested
    # deltas 0.125 (within band 0.375) -> tie_flip.  The old fixed 0.03 band failed.
    ar = _row({3: 16.125, 7: 16.0})
    dsp = _row({7: 16.125, 3: 16.0})
    out = _cls(ar, dsp, ar_token=3, dspark_token=7)
    assert out["class"] == "tie_flip"
    assert out["ar_contested_margin"] == pytest.approx(0.125, abs=1e-4)
    assert out["ar_contested_margin"] > DSPARK_TIE_MARGIN_DEFAULT   # old band would fail
    assert out["tie_band_used"] == pytest.approx(0.375, abs=1e-6)
    assert out["deltas_within_tie_band"] is True


def test_fixed_band_legacy_case_still_tie_flip():
    # Low magnitude (~1): 3*ulp_bf16(1.0)=0.0234 < 0.03, so the fixed floor governs;
    # contested margin 0.005, deltas 0.006 (within 0.03) -> tie_flip.
    ar = _row({3: 1.0, 7: 0.995})
    dsp = _row({7: 1.0, 3: 0.994})
    out = _cls(ar, dsp, ar_token=3, dspark_token=7)
    assert out["class"] == "tie_flip"
    assert out["ulp_bf16_at_peak"] == pytest.approx(2.0 ** -7, abs=1e-9)
    assert out["tie_band_used"] == pytest.approx(0.03, abs=1e-9)


def test_rounding_class_by_delta_computation():
    # A REAL flip (AR picks 3, verify picks 7) at |logit|~16 (band 0.375) whose
    # contested deltas (0.375 each, within band) close the smaller contested margin.
    # rounding_class_by_delta reports on the CONTESTED margins; the flip is tie_flip.
    ar = _row({3: 16.5, 7: 16.0})       # ar_contested(3,7) = 0.5, AR argmax = 3
    dsp = _row({3: 16.125, 7: 16.375})  # dsp_contested = 0.25, verify argmax = 7
    out = _cls(ar, dsp, ar_token=3, dspark_token=7)
    assert out["ar_contested_margin"] == pytest.approx(0.5, abs=1e-4)
    assert out["dspark_contested_margin"] == pytest.approx(0.25, abs=1e-4)
    assert out["delta_at_ar_token"] == pytest.approx(0.375, abs=1e-4)   # |16.5-16.125|
    assert out["delta_at_dspark_token"] == pytest.approx(0.375, abs=1e-4)  # |16.0-16.375|
    assert out["deltas_within_tie_band"] is True                        # 0.375 <= 0.375
    assert out["rounding_class_by_delta"] is True                       # 0.25 <= 0.75
    assert out["class"] == "tie_flip"


# ---------------------------------------------------------------------------
# 5. bf16 mx rows never raise
# ---------------------------------------------------------------------------
def test_bf16_mx_rows_do_not_raise():
    ar = mx.array(_row({5: 16.125, 9: 16.0, 20: 16.0})).astype(mx.bfloat16)
    dsp = mx.array(_row({5: 16.0, 9: 16.0, 20: 14.875})).astype(mx.bfloat16)
    out = _cls(ar, dsp, ar_token=5, dspark_token=9, index=111)
    assert out["class"] == "tie_flip"          # values are bf16-exact
    assert out["max_abs_logit_delta"] == pytest.approx(1.125, abs=1e-4)


# ---------------------------------------------------------------------------
# 6. k knob -- argument wins; env read AT USE; bounded; NOT a decode lever
# ---------------------------------------------------------------------------
def _k_case(**kw):
    # ar/dspark contested margins 0.2, contested deltas 0.2 at |logit|~16 (ulp 0.125):
    # k=3 -> band 0.375 -> tie_flip; k=0 -> band 0.03 -> divergent (deltas > band).
    ar = _row({3: 16.2, 7: 16.0})
    dsp = _row({7: 16.2, 3: 16.0})
    return _cls(ar, dsp, ar_token=3, dspark_token=7, index=1, **kw)


def test_tie_ulps_default_and_argument(monkeypatch):
    monkeypatch.delenv(DSPARK_DIVERGENCE_TIE_ULPS_ENV, raising=False)
    out = _k_case()
    assert out["tie_ulps"] == 3 and out["tie_band_used"] == pytest.approx(0.375, abs=1e-6)
    assert out["class"] == "tie_flip"
    out0 = _k_case(tie_ulps=0)
    assert out0["tie_ulps"] == 0
    assert out0["tie_band_used"] == pytest.approx(DSPARK_TIE_MARGIN_DEFAULT, abs=1e-9)
    assert out0["class"] == "divergent"


def test_tie_ulps_env_read_at_use(monkeypatch):
    monkeypatch.setenv(DSPARK_DIVERGENCE_TIE_ULPS_ENV, "0")
    assert _tie_ulps_from_env(None) == 0
    assert _k_case()["class"] == "divergent"
    monkeypatch.setenv(DSPARK_DIVERGENCE_TIE_ULPS_ENV, "10")
    assert _k_case()["tie_ulps"] == 10
    assert _k_case()["tie_band_used"] == pytest.approx(1.25, abs=1e-6)  # max(0.03,10*0.125)
    # the explicit argument WINS over the env
    assert _k_case(tie_ulps=3)["tie_ulps"] == 3


def test_tie_ulps_env_bounds_rejected(monkeypatch, capsys):
    D = DSPARK_DIVERGENCE_TIE_ULPS_DEFAULT
    # negatives / non-integers / malformed -> default + WARN (never silent coercion)
    for bad in ("-1", "1_0", "3.0", "nan", "1e20", "  "):
        monkeypatch.setenv(DSPARK_DIVERGENCE_TIE_ULPS_ENV, bad)
        assert _tie_ulps_from_env(None) == D, bad
    out = capsys.readouterr().out
    assert "WARN" in out
    # a valid integer passes; leading + ok; "0" is a valid explicit disable
    for good, exp in (("5", 5), ("+3", 3), ("0", 0)):
        monkeypatch.setenv(DSPARK_DIVERGENCE_TIE_ULPS_ENV, good)
        assert _tie_ulps_from_env(None) == exp, good
    # above the cap -> clamped
    monkeypatch.setenv(DSPARK_DIVERGENCE_TIE_ULPS_ENV, "100")
    assert _tie_ulps_from_env(None) == DSPARK_DIVERGENCE_TIE_ULPS_MAX
    monkeypatch.setenv(DSPARK_DIVERGENCE_TIE_ULPS_ENV, "100000000000000000000")
    assert _tie_ulps_from_env(None) == DSPARK_DIVERGENCE_TIE_ULPS_MAX


def test_tie_ulps_explicit_bounds_rejected():
    D = DSPARK_DIVERGENCE_TIE_ULPS_DEFAULT
    assert _tie_ulps_from_env(3.5) == D        # fractional float rejected
    assert _tie_ulps_from_env(-1) == D         # negative rejected
    assert _tie_ulps_from_env(True) == D       # bool rejected
    assert _tie_ulps_from_env(0) == 0          # 0 is a valid explicit disable
    assert _tie_ulps_from_env(5) == 5
    assert _tie_ulps_from_env(3.0) == 3        # integer-valued float accepted
    assert _tie_ulps_from_env(100) == DSPARK_DIVERGENCE_TIE_ULPS_MAX  # clamped


def test_env_knob_is_not_a_decode_lever():
    ab = pytest.importorskip("scripts.deepseek_v41.ab_decode_env_levers")
    assert DSPARK_DIVERGENCE_TIE_ULPS_ENV == "MTPLX_DSV41_DIVERGENCE_TIE_ULPS"
    assert DSPARK_DIVERGENCE_TIE_ULPS_ENV not in ab.ALL_LEVER_ENVS


# ---------------------------------------------------------------------------
# 7. receipt shape
# ---------------------------------------------------------------------------
def test_receipt_is_json_scalars_and_keeps_w77_keys():
    ar = _row({5: 16.125, 9: 16.0, 20: 16.0})
    dsp = _row({5: 16.0, 9: 16.0, 20: 14.875})
    out = _cls(ar, dsp, ar_token=5, dspark_token=9, index=111)
    w77_keys = {
        "divergence_index", "ar_token", "dspark_token", "ar_top2_margin",
        "dspark_top2_margin", "max_abs_logit_delta", "tie_margin", "class",
    }
    w120_keys = {
        "tie_band_used", "tie_ulps", "peak_contested_logit", "ulp_bf16_at_peak",
        "ar_contested_margin", "dspark_contested_margin", "delta_at_ar_token",
        "delta_at_dspark_token", "rounding_class_by_delta", "deltas_within_tie_band",
        "ar_logit_at_ar_token", "ar_logit_at_dspark_token",
        "dspark_logit_at_ar_token", "dspark_logit_at_dspark_token",
    }
    assert w77_keys <= set(out)
    assert w120_keys <= set(out)
    assert out["tie_margin"] == pytest.approx(DSPARK_TIE_MARGIN_DEFAULT, abs=1e-9)
    for v in out.values():
        assert v is None or isinstance(v, (int, float, bool, str))
    json.dumps(out)
