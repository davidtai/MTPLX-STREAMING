"""W77 -- DSpark-DIRECT greedy divergence classification (tie-flip vs divergent).

The DSpark greedy stream is AR by construction; a first-token difference from the
AR reference is a greedy argmax *flip* that comes from the target forward returning
slightly different logits at the M=1 (AR decode) vs M=K+1 (verify) row count under
the cell16k levers.  David's standing rule: such a flip is acceptable *iff* the AR
top-1/top-2 gap at that position is within the rounding envelope (a near-tie).

These gates cover the classification primitives in
:mod:`mtplx.models.deepseek_v41_dspark_decode`:

  1. :func:`classify_divergence` -- synthetic near-tie -> ``tie_flip``; a wide AR
     margin -> ``divergent``; missing AR logits -> rejected ``unclassified``;
     ``max_abs_logit_delta`` and both top-2 margins computed.
  2. :class:`DivergenceCapture` -- ``observe`` snapshots the verify logits row of
     the FIRST committed token that differs from the reference, at the right
     global index, and stays empty when the streams agree.
  3. end to end on the shrunk CPU model: greedy dspark == AR leaves the capture
     empty (no false positive); a one-token-perturbed reference fires the capture
     at exactly that index, carrying the real verify row.

Self-contained: shrunk seeded config, CPU device (MLX fp32 bit-exact), no
downloads, no checkpoint, no experts.bin.
"""
import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402
from mtplx.models.deepseek_v41_dspark_decode import (  # noqa: E402
    DSPARK_BF16_CLASS_DELTA,
    DSPARK_TIE_MARGIN_DEFAULT,
    DivergenceCapture,
    DSparkDecodeStats,
    _top2_margin,
    classify_divergence,
    dspark_generate,
)
from mtplx.sampling import SamplerConfig  # noqa: E402

DIM = 32
N_LAYERS = 4
VOCAB = 48


def _args(**over):
    kwargs = dict(
        vocab_size=VOCAB,
        hidden_size=DIM,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        head_dim=16,
        qk_rope_head_dim=8,
        q_lora_rank=16,
        o_lora_rank=8,
        o_groups=2,
        moe_intermediate_size=16,
        n_routed_experts=8,
        num_experts_per_tok=2,
        sliding_window=8,
        window_size=8,
        hc_mult=4,
        hc_sinkhorn_iters=2,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=0.0,
        n_mtp_layers=3,
        dspark_block_size=4,
        dspark_noise_token_id=VOCAB - 1,
        dspark_target_layer_ids=[1, 2, 3],
        dspark_markov_rank=12,
        dspark_n_routed_experts=8,
        dspark_num_experts_per_tok=2,
    )
    kwargs.update(over)
    return ModelArgs(**kwargs)


def _seeded_model(seed=0):
    mx.random.seed(seed)
    model = Model(_args(), quantize=False, mtp=True)
    filled = []
    for name, value in tree_flatten(model.parameters()):
        leaf = name.split(".")[-1]
        if value.ndim == 1:
            noise = mx.random.normal(value.shape) * 0.1
            centre = 1.0 if leaf.endswith("norm_weight") or leaf == "scale" else 0.0
            new = noise + centre
        else:
            new = mx.random.normal(value.shape) * (value.shape[-1] ** -0.5)
        filled.append((name, new.astype(value.dtype)))
    model.update(tree_unflatten(filled))
    mx.eval(model.parameters())
    return model


# ---------------------------------------------------------------------------
# 1. classify_divergence (pure)
# ---------------------------------------------------------------------------
def _row(vocab, top1_idx, top1, top2_idx, top2, floor=-5.0):
    r = np.full(vocab, floor, dtype=np.float32)
    r[top1_idx] = top1
    r[top2_idx] = top2
    return r


def test_top2_margin_basic():
    r = _row(10, 3, 2.0, 7, 1.5)
    assert _top2_margin(r) == pytest.approx(0.5, abs=1e-6)
    assert _top2_margin(np.array([1.0])) is None
    assert _top2_margin(None) is None
    # accepts an mx.array
    assert _top2_margin(mx.array(r)) == pytest.approx(0.5, abs=1e-5)


def test_tie_flip_when_ar_margin_below_threshold():
    # AR nearly tied between tokens 3 (top) and 7; a few-ulp perturbation flips it.
    ar = _row(VOCAB, top1_idx=3, top1=2.000, top2_idx=7, top2=1.995)
    dsp = _row(VOCAB, top1_idx=7, top1=2.001, top2_idx=3, top2=1.994)
    out = classify_divergence(
        index=42, ar_token=3, dspark_token=7,
        ar_logits_row=ar, dspark_logits_row=dsp,
    )
    assert out["class"] == "tie_flip"
    assert out["divergence_index"] == 42
    assert out["ar_token"] == 3 and out["dspark_token"] == 7
    assert out["ar_top2_margin"] == pytest.approx(0.005, abs=1e-5)
    assert out["ar_top2_margin"] < out["tie_margin"]
    # rows differ by 0.006 at idx 3 and idx 7 -> max |Δlogit| = 0.006
    assert out["max_abs_logit_delta"] == pytest.approx(0.006, abs=1e-4)
    assert out["tie_margin"] == pytest.approx(DSPARK_TIE_MARGIN_DEFAULT)


def test_divergent_when_ar_margin_above_threshold():
    # AR is decisive (gap 0.5 >> tie_margin) yet the streams differ -> a real,
    # non-rounding divergence, not a tie-break flip.
    ar = _row(VOCAB, top1_idx=3, top1=2.5, top2_idx=7, top2=2.0)
    dsp = _row(VOCAB, top1_idx=7, top1=2.5, top2_idx=3, top2=2.0)
    out = classify_divergence(
        index=10, ar_token=3, dspark_token=7,
        ar_logits_row=ar, dspark_logits_row=dsp,
    )
    assert out["class"] == "divergent"
    assert out["ar_top2_margin"] == pytest.approx(0.5, abs=1e-5)
    assert out["ar_top2_margin"] >= out["tie_margin"]
    assert out["max_abs_logit_delta"] == pytest.approx(0.5, abs=1e-4)


def test_missing_ar_row_is_unclassified():
    dsp = _row(VOCAB, 7, 2.0, 3, 1.999)
    out = classify_divergence(
        index=5, ar_token=3, dspark_token=7,
        ar_logits_row=None, dspark_logits_row=dsp,
    )
    # Missing evidence proves neither a tie nor a non-tie difference.
    assert out["class"] == "unclassified"
    assert out["rows_consistent"] is None
    assert out["unavailable_logits"] == ["ar"]
    assert out["ar_top2_margin"] is None
    assert out["max_abs_logit_delta"] is None
    assert out["dspark_top2_margin"] == pytest.approx(0.001, abs=1e-4)


def test_tie_margin_override_flips_class():
    ar = _row(VOCAB, 3, 2.0, 7, 1.9)  # gap 0.1
    dsp = _row(VOCAB, 7, 2.0, 3, 1.9)
    # default 3e-2 -> 0.1 gap is 'divergent'
    assert classify_divergence(index=0, ar_token=3, dspark_token=7,
                               ar_logits_row=ar, dspark_logits_row=dsp)["class"] == "divergent"
    # a permissive override (0.2) reclassifies the same flip as tie_flip
    assert classify_divergence(index=0, ar_token=3, dspark_token=7,
                               ar_logits_row=ar, dspark_logits_row=dsp,
                               tie_margin=0.2)["class"] == "tie_flip"


def test_bf16_class_delta_documented():
    # The default threshold is exactly 3x the documented bf16-class floor.
    assert DSPARK_TIE_MARGIN_DEFAULT == pytest.approx(3.0 * DSPARK_BF16_CLASS_DELTA)


# ---------------------------------------------------------------------------
# 2. DivergenceCapture.observe (synthetic verify logits)
# ---------------------------------------------------------------------------
def _fake_verify_logits(argmax_ids, vocab=VOCAB):
    """[1, M, vocab] logits whose per-row argmax is argmax_ids[m] (peak 3.0)."""
    m = len(argmax_ids)
    arr = np.full((1, m, vocab), -1.0, dtype=np.float32)
    for i, tid in enumerate(argmax_ids):
        arr[0, i, tid] = 3.0
        arr[0, i, (tid + 1) % vocab] = 2.5  # a runner-up so top-2 margin = 0.5
    return mx.array(arr)


def test_capture_records_first_mismatch():
    # AR reference: prefill token 9 at pos 0, then decode tokens.
    ar_ref = [9, 4, 5, 6, 7, 8]
    cap = DivergenceCapture(ar_ref)
    # cycle 1 commits [4, 5] at global positions 1, 2 (verify rows 0, 1) -- agree
    cap.observe(base_len=0, committed=[4, 5], verify_logits=_fake_verify_logits([4, 5]))
    assert not cap.found
    # cycle 2 commits [20, 30] at global positions 3, 4 -- pos 3 expects 6, got 20
    vlog = _fake_verify_logits([20, 30])
    cap.observe(base_len=2, committed=[20, 30], verify_logits=vlog)
    assert cap.found
    assert cap.index == 3
    assert cap.ar_token == 6 and cap.dspark_token == 20
    assert cap.dspark_top2_margin == pytest.approx(0.5, abs=1e-4)
    # the captured row is verify row 0 of this cycle (the mismatching token)
    assert int(np.argmax(cap.dspark_logits_row)) == 20
    # a later cycle does not overwrite the first capture
    cap.observe(base_len=4, committed=[11], verify_logits=_fake_verify_logits([11]))
    assert cap.index == 3


def test_capture_empty_when_streams_agree():
    ar_ref = [9, 4, 5, 6]
    cap = DivergenceCapture(ar_ref)
    cap.observe(base_len=0, committed=[4, 5, 6], verify_logits=_fake_verify_logits([4, 5, 6]))
    assert not cap.found
    assert classify_divergence  # sanity: importable
    # None reference disables the capture entirely
    cap2 = DivergenceCapture(None)
    cap2.observe(base_len=0, committed=[1, 2], verify_logits=_fake_verify_logits([1, 2]))
    assert not cap2.found


# ---------------------------------------------------------------------------
# 3. end to end on the shrunk CPU model
# ---------------------------------------------------------------------------
def _ar_stream(model, prompt, n):
    return dspark_generate(
        model, prompt, max_tokens=n, sampler=SamplerConfig(temperature=0.0),
        seed=0, speculative_depth=0, stats=DSparkDecodeStats(),
    )


def test_e2e_greedy_matches_ar_capture_empty():
    model = _seeded_model()
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    n = 24
    ar = _ar_stream(model, prompt, n)
    cap = DivergenceCapture(ar)
    dsp = dspark_generate(
        model, prompt, max_tokens=n, sampler=SamplerConfig(temperature=0.0),
        seed=0, speculative_depth=3, stats=DSparkDecodeStats(),
        divergence_capture=cap,
    )
    # gate 1 of the decode lane: greedy dspark == AR on the fp32 CPU double.
    assert dsp == ar
    # so the capture never fires (no false positive on an exact lane).
    assert not cap.found


def test_e2e_perturbed_reference_fires_capture():
    model = _seeded_model()
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    n = 24
    dsp_stream = dspark_generate(
        model, prompt, max_tokens=n, sampler=SamplerConfig(temperature=0.0),
        seed=0, speculative_depth=3, stats=DSparkDecodeStats(),
    )
    # Perturb the AR reference at a known decode position so the capture must fire
    # there and read that cycle's verify logits row (proves the _decode_cycles
    # wiring, not just the synthetic path).
    flip_at = 7
    ref = list(dsp_stream)
    ref[flip_at] = (ref[flip_at] + 1) % VOCAB
    cap = DivergenceCapture(ref)
    dsp2 = dspark_generate(
        model, prompt, max_tokens=n, sampler=SamplerConfig(temperature=0.0),
        seed=0, speculative_depth=3, stats=DSparkDecodeStats(),
        divergence_capture=cap,
    )
    assert dsp2 == dsp_stream  # deterministic, unchanged
    assert cap.found
    assert cap.index == flip_at
    assert cap.dspark_token == dsp_stream[flip_at]
    assert cap.ar_token == ref[flip_at]
    assert cap.dspark_logits_row is not None
    # the captured verify row's argmax IS the committed dspark token (greedy).
    assert int(np.argmax(cap.dspark_logits_row)) == dsp_stream[flip_at]
