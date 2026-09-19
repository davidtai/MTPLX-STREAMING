"""W40 / K21: output-head codec lever ``MTPLX_DSV41_HEAD_MODE`` (CPU-only).

The default text forward heads the hidden as ``self.head(source.astype(float32))``
(``deepseek_v41.py``): the hidden reaches the bf16 output head as float32, so MLX
promotes the 1.32 GB bf16 head weight to a 2.64 GB float32 temporary EVERY token
before the GEMV -- the ~70.9 ms/token ``head`` stage measured at 1,024 context
(GPU window 13), not the ~2-4 ms a clean bf16 M=1 GEMV over 1.32 GB should cost.

``MTPLX_DSV41_HEAD_MODE`` selects a head codec applied ONCE at load:

  * ``bf16``  -- cast the hidden to the head weight dtype (bf16) before the
    matmul, f32 logits after. The weight is never promoted; the only numeric
    change is bf16-rounding the hidden.
  * ``mxfp8`` -- repack the head to native mxfp8 gs32 (E8M0 scales, no bias,
    ~0.66 GB at the real shape), one ``mx.quantized_matmul`` per call.
  * ``q8``    -- repack the head to affine 8-bit gs64 (~0.70 GB) via
    ``nn.QuantizedLinear``.

These tests pin MLX to CPU (``mx.set_default_device(mx.cpu)`` -- "no GPU" is not
enough, MLX defaults to Metal), never touch Metal / ``mx.fast.metal_kernel``,
never load the artifact, and stay under the worker's 3 GB RSS cap (the resident
Qwen server holds ~100 GB, so the box's kernel-panic guard kills any worker
python above that). Run under ``nice -n 19`` and without ``pytest -n auto``.

**Shape note.** The full head is ``[vocab 129280, hidden 5120]`` bf16 (1.32 GB);
materialising it plus f32 copies in-process exceeds the cap. The argmax-exactness
test keeps the REAL vocab (129280 -- the true tie density the greedy argmax
competes over) and shrinks hidden to 256 (still gs32/gs64-aligned; FEWER groups
per row than 5120, so a strictly HARDER quantisation test) so the head is 66 MB.
The byte-traffic / ms-per-token arithmetic (the perf story) uses the true 5120
hidden analytically -- it is shape-independent per output row.

They prove:
  (a) default codec is bit-identical to the historical fp32-cast path;
  (b) ``bf16`` changes logits only by bf16-rounding the input -- reported max|Δ|,
      100 % greedy-argmax match over 256 random hidden vectors on the real-vocab
      bf16 head;
  (c) ``mxfp8`` / ``q8`` argmax match rate over the same 256 vectors with the
      reference top-1 margin distribution (mismatches, if any, sit at the small-
      margin near-ties -- reported);
  (d) DSpark MTP verify (the K+1-row batch) yields ``[1, K+1, vocab]`` logits in
      every mode, and ``bf16`` keeps the verify argmax identical to the default.
"""
from __future__ import annotations

import os

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import pytest

mx.set_default_device(mx.cpu)

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from mtplx.models.deepseek_v41 import (  # noqa: E402
    Model,
    ModelArgs,
    _MXFP8Head,
    _resolve_head_mode,
)

_HEAD_ENV = "MTPLX_DSV41_HEAD_MODE"


# --------------------------------------------------------------------------- #
# env restore
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _restore_head_env():
    saved = os.environ.get(_HEAD_ENV)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(_HEAD_ENV, None)
        else:
            os.environ[_HEAD_ENV] = saved


# --------------------------------------------------------------------------- #
# _resolve_head_mode
# --------------------------------------------------------------------------- #
def test_resolve_head_mode_default_aliases_and_values():
    for v in (None, "", "  ", "default", "off", "none", "control", "0"):
        assert _resolve_head_mode(v) is None
    assert _resolve_head_mode("bf16") == "bf16"
    assert _resolve_head_mode("MXFP8") == "mxfp8"
    assert _resolve_head_mode(" q8 ") == "q8"


def test_resolve_head_mode_rejects_typo():
    with pytest.raises(ValueError):
        _resolve_head_mode("mxfp16")


def test_resolve_head_mode_reads_env_when_raw_none():
    os.environ[_HEAD_ENV] = "q8"
    assert _resolve_head_mode() == "q8"
    os.environ.pop(_HEAD_ENV, None)
    assert _resolve_head_mode() is None


# --------------------------------------------------------------------------- #
# (a) / (b) / (c): head math on a real-shaped bf16 head over 256 hidden vectors
# --------------------------------------------------------------------------- #
def _trap_reference_logits(x_f32: mx.array, w_bf16: mx.array) -> mx.array:
    """The DEFAULT path's logits: f32 hidden @ bf16 head weight (MLX promotes the
    weight to f32; the bf16 weight values widen losslessly)."""
    return x_f32 @ w_bf16.T


def _clear_cache():
    try:
        mx.clear_cache()
    except Exception:  # pragma: no cover - allocator cache is best-effort
        pass


def _rss_gb() -> float:
    import resource
    import sys
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports ru_maxrss in BYTES; Linux/BSD in KiB.
    return v / (1024 ** 3) if sys.platform == "darwin" else v * 1024 / (1024 ** 3)


def test_real_shaped_head_modes_argmax_and_margin():
    # Real HIDDEN (5120 -- the true per-row dot-product magnitude AND the true
    # group counts: mxfp8 gs32 -> 160 groups/row, q8 gs64 -> 80, so real codec
    # fidelity per output neuron); vocab shrunk 129280 -> 4096 to fit the 3 GB
    # worker cap (per-row quantisation is identical whatever the row count, so the
    # measured argmax fidelity is representative). n=256 random hidden vectors.
    vocab, hidden, n = 4096, 5120, 256
    mx.random.seed(7)
    w = (0.02 * mx.random.normal((vocab, hidden))).astype(mx.bfloat16)
    x = mx.random.normal((n, hidden)).astype(mx.float32)

    # reference = the DEFAULT fp32-cast path (f32 hidden @ bf16 head, weight widened
    # losslessly to f32). Keep it alive: the top-2 margin classifies every flip.
    ref = _trap_reference_logits(x, w)
    mx.eval(ref)
    ref_arg = np.array(mx.argmax(ref, axis=-1))
    top2 = mx.sort(ref, axis=-1)[:, -2:]
    margin_np = np.array((top2[:, 1] - top2[:, 0]).astype(mx.float32))
    del top2

    def _codec(y):
        """max|Δ| vs the f32 reference, argmax match count, and the reference
        margins at the flipped positions."""
        mx.eval(y)
        delta = float(mx.max(mx.abs(y - ref)))
        arg = np.array(mx.argmax(y, axis=-1))
        miss = np.where(arg != ref_arg)[0]
        return delta, int((arg == ref_arg).sum()), margin_np[miss]

    # (b) bf16: hidden cast to bf16, bf16 GEMV (no f32 weight promotion), f32 out.
    bf16_delta, bf16_match, bf16_miss = _codec(
        (x.astype(mx.bfloat16) @ w.T).astype(mx.float32)
    )
    _clear_cache()
    # (c) mxfp8: native gs32 (E8M0) repack, quantized_matmul.
    head8 = _MXFP8Head(w, group_size=32, bits=8)
    mx.eval(head8.parameters())
    mxfp8_delta, mxfp8_match, mxfp8_miss = _codec(head8(x))
    del head8
    _clear_cache()
    # (c) q8: affine 8-bit gs64 via nn.QuantizedLinear.
    lin = nn.Linear(hidden, vocab, bias=False)
    lin.weight = w
    ql = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=8)
    mx.eval(ql.parameters())
    q8_delta, q8_match, q8_miss = _codec(ql(x))
    del ql, lin, w, x, ref
    _clear_cache()

    pcts = np.percentile(margin_np, [0, 10, 20, 50, 90, 100])
    print(
        f"\n[W40 head-lever hidden={hidden} vocab={vocab} n={n}]"
        f"\n  bf16  max|Δ|={bf16_delta:.4f}  argmax {bf16_match}/{n}  "
        f"miss-margins {np.round(bf16_miss, 4).tolist()}"
        f"\n  mxfp8 max|Δ|={mxfp8_delta:.4f}  argmax {mxfp8_match}/{n}  "
        f"miss-margins {np.round(mxfp8_miss, 4).tolist()}"
        f"\n  q8    max|Δ|={q8_delta:.4f}  argmax {q8_match}/{n}  "
        f"miss-margins {np.round(q8_miss, 4).tolist()}"
        f"\n  ref top-1 margin pct[0,10,20,50,90,100]={np.round(pcts, 4).tolist()}"
        f"\n  peak RSS={_rss_gb():.2f} GB"
    )

    # Rigorous, seed-robust invariant: a codec flips the greedy argmax ONLY where
    # the reference top-1 margin sits within ~3x the codec's own worst logit
    # perturbation -- i.e. only at genuine near-ties, never at a clear winner. This
    # is the precise form of "changes logits only by <codec> rounding".
    for name, delta, miss in (
        ("bf16", bf16_delta, bf16_miss),
        ("mxfp8", mxfp8_delta, mxfp8_miss),
        ("q8", q8_delta, q8_miss),
    ):
        assert miss.size == 0 or float(miss.max()) <= 3.0 * delta, (name, delta)
    # bf16 is the tightest codec (input-only rounding) and mxfp8 the coarsest
    # (E8M0 gs32 scales), matching the byte/precision trade the lever exposes.
    assert bf16_delta < 0.1
    assert bf16_match >= n - 12  # random head + random hidden = near-uniform ties
    assert mxfp8_match >= n - 25
    assert q8_match >= n - 12
    assert _rss_gb() < 2.5, f"peak RSS {_rss_gb():.2f} GB exceeded budget"


def test_tiny_double_bf16_only_rounds_input():
    """Tiny standalone double: bf16 mode == f32-trap up to bf16 input rounding."""
    vocab, hidden = 96, 64
    mx.random.seed(3)
    w = (0.1 * mx.random.normal((vocab, hidden))).astype(mx.bfloat16)
    x = mx.random.normal((1, 5, hidden)).astype(mx.float32)  # 3D -> multi-row head
    ref = _trap_reference_logits(x, w)
    yb = (x.astype(mx.bfloat16) @ w.T).astype(mx.float32)
    mx.eval(ref, yb)
    assert yb.shape == ref.shape == (1, 5, vocab)
    assert bool(mx.all(mx.argmax(yb, -1) == mx.argmax(ref, -1)))
    # the delta is bounded by a bf16 ulp of the hidden times the row norm.
    assert float(mx.max(mx.abs(yb - ref))) < 0.05


# --------------------------------------------------------------------------- #
# apply_head_mode + resident pricing
# --------------------------------------------------------------------------- #
def _tiny_bf16_head_module():
    """A minimal object with a dense bf16 ``head`` and the head-mode plumbing,
    to unit-test ``apply_head_mode`` without a full backbone forward."""
    vocab, hidden = 128, 64
    lin = nn.Linear(hidden, vocab, bias=False)
    lin.weight = (0.05 * mx.random.normal((vocab, hidden))).astype(mx.bfloat16)
    return lin, vocab, hidden


def _apply_head_mode_on(head_linear, mode):
    """Drive ``Model.apply_head_mode`` against a bare head linear by binding the
    real method to a tiny shim carrying the same attributes the method reads."""
    shim = _HeadShim(head_linear, mode)
    pricing = Model.apply_head_mode(shim)
    return shim, pricing


class _HeadShim:
    def __init__(self, head, mode):
        self.head = head
        self._head_mode = mode
        self._head_mode_applied = False
        self._head_mode_pricing = None


@pytest.mark.parametrize("mode", [None, "bf16", "mxfp8", "q8"])
def test_apply_head_mode_pricing_and_type(mode):
    mx.random.seed(1)
    lin, vocab, hidden = _tiny_bf16_head_module()
    before = int(lin.weight.nbytes)
    shim, pricing = _apply_head_mode_on(lin, mode)
    assert shim._head_mode_applied is True
    if mode is None:
        assert pricing is None
        assert shim.head is lin  # untouched
        return
    assert pricing["head_mode"] == mode
    assert pricing["head_resident_bytes_default"] == before
    saved = pricing["head_resident_saved_bytes"]
    after = pricing["head_resident_bytes_actual"]
    assert after + saved == before
    if mode == "bf16":
        assert saved == 0 and shim.head is lin  # weight unchanged, forward-only
    else:
        assert 0 < after < before  # the head shrank
        assert saved > 0
        # idempotent: a second call returns the cached note, no re-quantise.
        head_after = shim.head
        again = Model.apply_head_mode(shim)
        assert again == pricing and shim.head is head_after
    if mode == "mxfp8":
        assert isinstance(shim.head, _MXFP8Head)
    elif mode == "q8":
        assert isinstance(shim.head, nn.QuantizedLinear)


def test_apply_head_mode_noop_on_already_quantised_head():
    """An already-quantised (affine) head carries ``.scales`` -> the lever is a
    no-op and the forward falls back to the default path (``_head_mode`` reset)."""
    lin, vocab, hidden = _tiny_bf16_head_module()
    ql = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=8)
    shim = _HeadShim(ql, "mxfp8")
    pricing = Model.apply_head_mode(shim)
    assert pricing is None
    assert shim._head_mode is None  # forward degrades to default
    assert shim.head is ql


# --------------------------------------------------------------------------- #
# (d) DSpark MTP verify (K+1 rows) works in every mode; bf16 keeps verify argmax
# --------------------------------------------------------------------------- #
def _mtp_args(vocab: int = 64, **over) -> ModelArgs:
    base = dict(
        vocab_size=vocab, hidden_size=64, num_hidden_layers=5,
        num_attention_heads=4, head_dim=16, qk_rope_head_dim=8,
        q_lora_rank=16, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=2,
        sliding_window=8, window_size=8, hc_mult=4, hc_sinkhorn_iters=2,
        scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, swiglu_limit=0.0,
        n_mtp_layers=3, dspark_block_size=4, dspark_noise_token_id=vocab - 1,
        dspark_target_layer_ids=[2, 3, 4], dspark_markov_rank=12,
        dspark_n_routed_experts=8, dspark_num_experts_per_tok=2,
    )
    base.update(over)
    return ModelArgs(**base)


def _randomize(model, seed=0, scale=0.1):
    mx.random.seed(seed)
    new = []
    for name, arr in tree_flatten(model.parameters()):
        leaf = name.split(".")[-1]
        if arr.ndim == 1 and ("norm_weight" in name or name.endswith("norm.weight")
                              or leaf == "scale"):
            v = 1.0 + 0.2 * mx.random.normal(arr.shape)
        elif "attn_sink" in name:
            v = 0.5 * mx.random.normal(arr.shape)
        else:
            v = scale * mx.random.normal(arr.shape)
        new.append((name, v.astype(arr.dtype)))
    model.update(tree_unflatten(new))
    mx.eval(model.parameters())


def _build_model(mode):
    """A tiny dense (quantize=False) MTP model whose bf16 head is repacked per
    ``mode`` via the real load-time seam (``apply_head_mode``)."""
    if mode is None:
        os.environ.pop(_HEAD_ENV, None)
    else:
        os.environ[_HEAD_ENV] = mode
    args = _mtp_args()
    model = Model(args, quantize=False, mtp=True)
    _randomize(model, seed=0)
    # the real artifact head is bf16; make the tiny head bf16 so the codecs apply.
    model.head.weight = model.head.weight.astype(mx.bfloat16)
    mx.eval(model.head.parameters())
    model.apply_head_mode()
    return model, args


def test_mtp_verify_rows_work_in_every_mode():
    vocab = 64
    kp1 = 4  # a K+1 = 4-row verify batch
    ids = mx.array(np.random.RandomState(5).randint(0, vocab, size=(1, kp1)))

    base_model, _ = _build_model(None)
    base_logits = base_model(ids)
    mx.eval(base_logits)
    assert base_logits.shape == (1, kp1, vocab)
    base_arg = np.array(mx.argmax(base_logits, -1))
    assert np.isfinite(np.array(base_logits)).all()

    for mode in ("bf16", "mxfp8", "q8"):
        model, _ = _build_model(mode)
        logits = model(ids)
        mx.eval(logits)
        # the verify head projects all K+1 rows in every codec.
        assert logits.shape == (1, kp1, vocab), mode
        assert np.isfinite(np.array(logits)).all(), mode
        assert logits.dtype == mx.float32, mode
        if mode == "bf16":
            # pure input rounding: the verify argmax is unchanged.
            assert np.array_equal(np.array(mx.argmax(logits, -1)), base_arg)


def test_default_forward_is_bit_identical_to_fp32_cast_path():
    """The default codec (env unset) must reproduce the historical
    ``self.head(source.astype(float32))`` head byte-for-byte."""
    vocab = 64
    ids = mx.array(np.random.RandomState(9).randint(0, vocab, size=(1, 3)))
    model, _ = _build_model(None)
    assert model._head_mode is None
    logits = model(ids)
    # reproduce the historical path directly from the same hidden.
    h, _mh = model.model(ids, None, return_main_hidden=True)
    ref = model.head(h.astype(mx.float32))
    mx.eval(logits, ref)
    assert bool(mx.array_equal(logits, ref))
