"""W50 / K25: CPU exactness tests for the DSV4.1 prefill score-path levers.

Covers the ``mtplx/models/deepseek_v41.py`` attention score/softmax/value path
under ``MTPLX_DSV41_PREFILL_SCORE_DTYPE`` (f32|bf16 QK^T/PV matmul input dtype)
and ``MTPLX_DSV41_PREFILL_SCORE_KEY_CHUNK`` (split-K online-softmax chunk width):

  * ``score_dtype=f32`` (the default) is BYTE-IDENTICAL to the shipped inline
    ``_sparse_attend`` formula;
  * MLX bf16 matmul accumulates in f32 (a bf16 matmul == round_bf16 of the
    f32-accumulated result), so the bf16 arm's error is exactly the two matmuls'
    input+output bf16 rounding -- LOSSY-by-design, greedy tokens can flip;
  * the split-K online softmax is mathematically identical to one-shot up to
    float reassociation (f32: ~1e-6, greedy-identical), NaN-free on fully-masked
    chunks, and composes with bf16;
  * decode / M=1 (``q.shape[1] == 1``) ignores both levers (byte-identical).

No GPU/Metal, no artifact load, <3 GB RSS.  MLX pinned to CPU per
memory/worker-tests-must-pin-mlx-cpu.md.  Run under ``nice -n 19``, no ``-n auto``.
"""

from __future__ import annotations

import os
import resource

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_unflatten

mx.set_default_device(mx.cpu)

from mtplx.models import deepseek_v41 as dsv41  # noqa: E402
from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402

_HD = dsv41._PREFILL_SCORE_DTYPE_ENV
_KC = dsv41._PREFILL_SCORE_KEY_CHUNK_ENV


class _AttnShim:
    """Binds the real ``_sparse_attend*`` methods to the minimal state they read
    (``softmax_scale`` + per-head ``attn_sink``) -- a unit harness over the actual
    module code, no full ``Attention.__init__`` / projections needed."""

    def __init__(self, head_dim, attn_sink):
        self.softmax_scale = head_dim ** -0.5
        self.attn_sink = attn_sink


# bind the real (unbound) methods so calls exercise the shipped code paths
_AttnShim._sparse_attend = dsv41.Attention._sparse_attend
_AttnShim._sparse_attend_oneshot = dsv41.Attention._sparse_attend_oneshot
_AttnShim._sparse_attend_chunked = dsv41.Attention._sparse_attend_chunked


def _rand_inputs(b, s, H, hd, T, *, drop=0.3, seed=3):
    mx.random.seed(seed)
    q = mx.random.normal((b, s, H, hd)).astype(mx.float32)
    KV = mx.random.normal((b, T, hd)).astype(mx.float32)
    # a realistic causal-ish reachability mask + a per-head learned sink
    attend = mx.random.uniform(shape=(b, s, T)) > drop
    sink = 0.3 * mx.random.normal((H,))
    return q, KV, attend, sink


def _oneshot_reference(shim, q, KV, attend):
    """The pre-W50 inline ``_sparse_attend`` formula, verbatim (the f32 oracle)."""
    b, s, H, _ = q.shape
    scores = mx.einsum("bshd,btd->bsht", q.astype(mx.float32), KV.astype(mx.float32))
    scores = scores * shim.softmax_scale
    scores = mx.where(attend[:, :, None, :], scores, float("-inf"))
    sink = mx.broadcast_to(shim.attn_sink.astype(mx.float32).reshape(1, 1, H, 1), (b, s, H, 1))
    full = mx.concatenate([scores, sink], axis=-1)
    w = mx.softmax(full, axis=-1)[..., : KV.shape[1]]
    return mx.einsum("bsht,btd->bshd", w, KV.astype(mx.float32))


@pytest.fixture(autouse=True)
def _clear_score_env():
    saved = {k: os.environ.get(k) for k in (_HD, _KC)}
    for k in (_HD, _KC):
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --------------------------------------------------------------------------
# (1) f32 path is byte-identical to the shipped inline formula
# --------------------------------------------------------------------------


def test_score_dtype_f32_byte_identical_to_shipped_formula():
    q, KV, attend, sink = _rand_inputs(1, 12, 8, 64, 300)
    shim = _AttnShim(64, sink)
    ref = _oneshot_reference(shim, q, KV, attend)
    got = shim._sparse_attend_oneshot(q, KV, attend, mx.float32)
    assert mx.array_equal(ref, got), "f32 one-shot must be byte-identical to control"
    # and the dispatcher with no env set (or explicit f32) routes to the same path
    got_disp = shim._sparse_attend(q, KV, attend)
    assert mx.array_equal(ref, got_disp)
    os.environ[_HD] = "f32"
    assert mx.array_equal(ref, shim._sparse_attend(q, KV, attend))


# --------------------------------------------------------------------------
# (2) bf16 matmul accumulates in f32 -> error == bf16 output rounding
# --------------------------------------------------------------------------


def test_bf16_matmul_accumulates_in_f32():
    # QK^T-shaped contraction over head_dim=512 (the DSV4.1 latent width).
    mx.random.seed(7)
    q = mx.random.normal((1, 8, 4, 512)).astype(mx.float32).astype(mx.bfloat16)
    KV = mx.random.normal((1, 64, 512)).astype(mx.float32).astype(mx.bfloat16)
    bf = mx.einsum("bshd,btd->bsht", q, KV)
    assert bf.dtype == mx.bfloat16
    # f32-accumulated reference of the SAME bf16 inputs:
    ref = mx.einsum("bshd,btd->bsht", q.astype(mx.float32), KV.astype(mx.float32))
    # a bf16 matmul equals round_bf16(f32-accumulated) EXACTLY (proves f32 accum,
    # not bf16 accumulation whose error would grow with the K=512 reduction):
    assert mx.array_equal(bf, ref.astype(mx.bfloat16))
    err = float(mx.max(mx.abs(bf.astype(mx.float32) - ref)))
    out_round = float(mx.max(mx.abs(ref - ref.astype(mx.bfloat16).astype(mx.float32))))
    assert err <= out_round + 1e-12, (err, out_round)


# --------------------------------------------------------------------------
# (3) bf16 one-shot: lossy, bounded by bf16 rounding
# --------------------------------------------------------------------------


def test_score_bf16_oneshot_delta_is_bf16_scale():
    q, KV, attend, sink = _rand_inputs(1, 12, 8, 64, 300)
    shim = _AttnShim(64, sink)
    f32 = shim._sparse_attend_oneshot(q, KV, attend, mx.float32)
    bf16 = shim._sparse_attend_oneshot(q, KV, attend, mx.bfloat16)
    d = float(mx.max(mx.abs(bf16 - f32)))
    rel = d / (float(mx.max(mx.abs(f32))) + 1e-9)
    # bf16 mantissa is ~2^-8; the output-value delta is at bf16 rounding scale,
    # far above the f32 chunked-reassociation floor (~1e-6) -- it is a real,
    # lossy precision change, not a reorder.
    assert rel < 5e-2, rel
    assert d > 1e-4, "bf16 must actually differ (it is lossy by design)"


# --------------------------------------------------------------------------
# (4) split-K online softmax == one-shot up to f32 reassociation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key_chunk", [1, 7, 32, 64, 128, 512, 4096])
def test_chunked_f32_matches_oneshot(key_chunk):
    q, KV, attend, sink = _rand_inputs(1, 12, 8, 64, 300)
    shim = _AttnShim(64, sink)
    f32 = shim._sparse_attend_oneshot(q, KV, attend, mx.float32)
    ch = shim._sparse_attend_chunked(q, KV, attend, mx.float32, key_chunk)
    d = float(mx.max(mx.abs(ch - f32)))
    # scores are bit-identical (the QK^T reduces over head_dim, not the chunked T
    # axis); only the softmax denom + value sum reassociate -> pure f32 reorder.
    assert d < 1e-4, (key_chunk, d)


def test_chunked_all_masked_chunk_is_nan_free_and_equals_oneshot():
    q, KV, _, sink = _rand_inputs(1, 6, 8, 64, 200)
    shim = _AttnShim(64, sink)
    attend = mx.zeros((1, 6, 200)).astype(mx.bool_)  # nothing reachable -> sink only
    ch = shim._sparse_attend_chunked(q, KV, attend, mx.float32, 64)
    assert bool(mx.all(mx.isfinite(ch))), "fully-masked chunk must not NaN"
    os_ = shim._sparse_attend_oneshot(q, KV, attend, mx.float32)
    assert mx.array_equal(ch, os_)  # both collapse to the value-0 sink -> zeros
    assert float(mx.max(mx.abs(ch))) == 0.0


def test_chunked_composes_with_bf16():
    q, KV, attend, sink = _rand_inputs(1, 12, 8, 64, 300)
    shim = _AttnShim(64, sink)
    bf16 = shim._sparse_attend_oneshot(q, KV, attend, mx.bfloat16)
    both = shim._sparse_attend_chunked(q, KV, attend, mx.bfloat16, 64)
    assert bool(mx.all(mx.isfinite(both)))
    # bf16-chunked lands near bf16 one-shot (bf16 rounding + f32 reassociation)
    assert float(mx.max(mx.abs(both - bf16))) < 5e-2


# --------------------------------------------------------------------------
# (5) decode / M=1 ignores both levers (byte-identical to control)
# --------------------------------------------------------------------------


def test_decode_m1_ignores_score_levers():
    q, KV, attend, sink = _rand_inputs(1, 1, 8, 64, 40)  # s == 1 (decode)
    shim = _AttnShim(64, sink)
    base = shim._sparse_attend_oneshot(q, KV, attend, mx.float32)
    for env in ({_HD: "bf16"}, {_KC: "8"}, {_HD: "bf16", _KC: "8"}):
        for k in (_HD, _KC):
            os.environ.pop(k, None)
        for k, v in env.items():
            os.environ[k] = v
        out = shim._sparse_attend(q, KV, attend)
        assert mx.array_equal(base, out), f"decode M=1 must ignore {env}"


# --------------------------------------------------------------------------
# resolvers
# --------------------------------------------------------------------------


def test_resolvers_parse_and_reject():
    r = dsv41._resolve_prefill_score_dtype
    assert r("") is mx.float32
    assert r("f32") is mx.float32 and r("fp32") is mx.float32 and r("default") is mx.float32
    assert r("bf16") is mx.bfloat16 and r("BF16") is mx.bfloat16 and r("bfloat16") is mx.bfloat16
    with pytest.raises(ValueError):
        r("fp8")
    c = dsv41._resolve_prefill_score_key_chunk
    assert c("") is None and c("0") is None and c("off") is None
    assert c("2048") == 2048 and c("1") == 1
    with pytest.raises(ValueError):
        c("-4")
    with pytest.raises(ValueError):
        c("wide")


# --------------------------------------------------------------------------
# real-shaped random layer: 64 heads x 512, T up to 4096, <3 GB RSS
# --------------------------------------------------------------------------


def test_real_shaped_layer_exactness_and_footprint():
    # DSV4.1 shape: H=64, head_dim=512.  rows=128 prefill queries over T=4096 keys.
    b, s, H, hd, T = 1, 128, 64, 512, 4096
    q, KV, attend, sink = _rand_inputs(b, s, H, hd, T, seed=11)
    shim = _AttnShim(hd, sink)
    f32 = shim._sparse_attend_oneshot(q, KV, attend, mx.float32)
    mx.eval(f32)
    bf16 = shim._sparse_attend_oneshot(q, KV, attend, mx.bfloat16)
    mx.eval(bf16)
    ch = shim._sparse_attend_chunked(q, KV, attend, mx.float32, 512)
    mx.eval(ch)
    d_bf16 = float(mx.max(mx.abs(bf16 - f32)))
    d_ch = float(mx.max(mx.abs(ch - f32)))
    denom = float(mx.max(mx.abs(f32))) + 1e-9
    # chunked f32: reassociation floor, orders of magnitude below the bf16 delta.
    assert d_ch < 1e-3, d_ch
    assert d_ch < d_bf16, (d_ch, d_bf16)
    assert (d_bf16 / denom) < 5e-2, d_bf16 / denom
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)
    # macOS ru_maxrss is bytes; the transient must stay well under the 3 GB budget.
    assert rss_gb < 3.0, f"RSS {rss_gb:.2f} GB exceeded 3 GB"


# --------------------------------------------------------------------------
# end-to-end tiny-model prefill: chunked-f32 is greedy-identical; bf16 is lossy
# --------------------------------------------------------------------------


def _tiny_model():
    args = ModelArgs(
        vocab_size=48, hidden_size=32, num_hidden_layers=8,
        num_attention_heads=4, head_dim=16, qk_rope_head_dim=4,
        q_lora_rank=12, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=2,
        index_n_heads=2, index_head_dim=8, index_topk=5,
        sliding_window=8, window_size=8, swiglu_limit=0.5,
        compress_ratios=[0, 0, 2, 2, 2, 1, 1, 1],
        kv_source_layer_ids=[2, 5], index_source_layer_ids=[2, 5, 6],
        candidate_source_layer_id=5, candidate_topk_blocks=3, candidate_block_size=2,
        rope_scaling={"rope_type": "yarn", "factor": 16, "beta_fast": 32,
                      "beta_slow": 1, "original_max_position_embeddings": 65536},
    )
    model = Model(args)
    mx.random.seed(1)
    new = []
    for name, arr in tree_flatten(model.parameters()):
        if arr.ndim == 1 and ("norm_weight" in name or name.endswith("norm.weight")):
            v = 1.0 + 0.2 * mx.random.normal(arr.shape)
        elif "attn_sink" in name:
            v = 0.5 * mx.random.normal(arr.shape)
        else:
            v = 0.1 * mx.random.normal(arr.shape)
        new.append((name, v.astype(mx.float32)))
    model.update(tree_unflatten(new))
    mx.eval(model.parameters())
    return model


def _logits(model, ids, env):
    for k in (_HD, _KC):
        os.environ.pop(k, None)
    for k, v in env.items():
        os.environ[k] = v
    lg = model(mx.array([ids]))
    mx.eval(lg)
    return lg


def test_tiny_model_prefill_argmax_parity():
    model = _tiny_model()
    ids = list(range(30))
    base = _logits(model, ids, {})                     # control (f32 one-shot)
    assert mx.array_equal(base, _logits(model, ids, {_HD: "f32"}))  # explicit f32 == control

    # chunked f32: greedy-identical on EVERY row (reassociation ~1e-6 << logit gaps)
    ch = _logits(model, ids, {_KC: "8"})
    assert float(mx.max(mx.abs(ch - base))) < 1e-4
    assert bool(mx.all(mx.argmax(ch[0], axis=-1) == mx.argmax(base[0], axis=-1)))

    # bf16: LOSSY by design.  We assert only that it runs finite and its logit
    # delta is at the bf16 scale -- greedy tokens CAN flip on near-ties (they do on
    # this untrained double's intermediate rows), so parity is NOT asserted; a task
    # eval (HumanEval) is the ship gate, not exactness (cf. W40 head_bf16, memory/
    # deepseek-v4-quality-verdict).
    bf = _logits(model, ids, {_HD: "bf16"})
    assert bool(mx.all(mx.isfinite(bf)))
    assert float(mx.max(mx.abs(bf - base))) > 1e-3

    both = _logits(model, ids, {_HD: "bf16", _KC: "8"})
    assert bool(mx.all(mx.isfinite(both)))
