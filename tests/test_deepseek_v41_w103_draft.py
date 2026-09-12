"""W103 DSpark draft-head fp32-cast-trap removal (``MTPLX_DSV41_DRAFT_HEAD_BF16``).

``DSparkBlock.forward_head`` projects the draft hidden through the SHARED trunk
output head as ``head(_rmsnorm(x, ...).astype(mx.float32))``.  With a dense bf16
head (the native artifact keeps the head bf16, and the backbone
``MTPLX_DSV41_HEAD_MODE=bf16`` fix leaves the weight bf16 -- it repairs only
``Model._apply_head``, NOT this draft site), the ``.astype(f32)`` on the INPUT
forces MLX -- which has no mixed-precision matmul -- to promote the whole
``[vocab, hidden]`` bf16 weight to a f32 temporary every draft cycle before the
GEMV.  It is the exact twin of the trap W40/K21 removed on the backbone
(memory/dsv41-head-fp32-cast-trap names this draft site as the follow-up).

``MTPLX_DSV41_DRAFT_HEAD_BF16`` (default OFF, read at use) casts the draft hidden
to the head weight dtype instead (a bf16 GEMV over the resident weight, f32 logits
after) -- exactly the backbone ``bf16`` codec.  The draft head's numerics only set
the acceptance rate (the runtime's greedy target verify is authoritative), so on a
bf16 head this is a rounding-class change; when the head is already f32 (the tiny
CPU double) the cast is a no-op, so the flag on/off is BYTE-IDENTICAL, proven here
over 64 draft steps.

CPU device (MLX fp32 bit-exact), tiny seeded config, no artifact.
"""
import io
import os
import re
from collections import Counter

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

import mtplx.models.deepseek_v41 as dv41  # noqa: E402
import mtplx.models.deepseek_v41_dspark as dsp  # noqa: E402
from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.fixture(autouse=True)
def _restore_flags():
    """Every test flips the module global / env / counters; restore all around each."""
    prev_head = dsp._DRAFT_HEAD_BF16
    prev_env = os.environ.get(dsp._DRAFT_HEAD_BF16_ENV)
    prev_compile = dsp._DRAFT_COMPILE
    dsp._reset_draft_head_calls()
    dsp._DRAFT_COMPILE = False  # isolate the head lever from the K33 tape collapse
    try:
        yield
    finally:
        dsp._DRAFT_HEAD_BF16 = prev_head
        dsp._DRAFT_COMPILE = prev_compile
        dsp._reset_draft_head_calls()
        dsp._DRAFT_COMPILED.clear()
        dv41._ATTN_COMPILED.clear()
        dv41._HC_COMPILED.clear()
        if prev_env is None:
            os.environ.pop(dsp._DRAFT_HEAD_BF16_ENV, None)
        else:
            os.environ[dsp._DRAFT_HEAD_BF16_ENV] = prev_env


def _args(vocab=64, block_size=5, **over):
    kwargs = dict(
        vocab_size=vocab, hidden_size=32, num_hidden_layers=5, num_attention_heads=4,
        head_dim=16, qk_rope_head_dim=8, q_lora_rank=16, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=2,
        sliding_window=8, window_size=8, hc_mult=4, hc_sinkhorn_iters=2,
        scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, swiglu_limit=0.0,
        n_mtp_layers=3, dspark_block_size=block_size, dspark_noise_token_id=vocab - 1,
        dspark_target_layer_ids=[2, 3, 4], dspark_markov_rank=12,
        dspark_n_routed_experts=8, dspark_num_experts_per_tok=2,
    )
    kwargs.update(over)
    return ModelArgs(**kwargs)


def _seeded_model(seed=0, vocab=64, head_bf16=False, **over):
    mx.random.seed(seed)
    args = _args(vocab=vocab, **over)
    model = Model(args, quantize=False, mtp=True)
    filled = []
    for name, value in tree_flatten(model.parameters()):
        leaf = name.split(".")[-1]
        if value.ndim == 1:
            centre = 1.0 if leaf.endswith("norm_weight") or leaf == "scale" else 0.0
            new = mx.random.normal(value.shape) * 0.1 + centre
        else:
            new = mx.random.normal(value.shape) * (value.shape[-1] ** -0.5)
        filled.append((name, new.astype(value.dtype)))
    model.update(tree_unflatten(filled))
    if head_bf16:
        model.head.weight = model.head.weight.astype(mx.bfloat16)
    mx.eval(model.parameters())
    return args, model


_PROMPT = [3, 5, 7, 9, 11, 13, 2, 4, 6, 8, 10, 12, 1, 0, 15, 17, 19]


def _seed_caches(model):
    """One prompt forward + window seed; returns (main_h, caches).  The draft does
    not mutate the caches (it attends over the window without appending), so many
    independent draft steps can reuse one seeding."""
    ids = mx.array(_PROMPT).reshape(1, -1)
    logits, main_hidden = model(ids, return_hidden=True)
    mx.eval(logits, main_hidden)
    caches = model.make_mtp_cache()
    model.mtp.seed_main(main_hidden, caches)
    return main_hidden[:, -1:, :], caches


def _draft_step(model, main_h, caches, primary_token):
    primary = mx.array([int(primary_token)])
    out_ids, dlogits, conf = model.mtp.draft_block(
        main_h, primary, caches, model.model.embed_tokens, model.head
    )
    mx.eval(out_ids, dlogits, conf)
    return out_ids, dlogits, conf


_DOT_RECT = re.compile(r'\[label ="([^"]+)", shape=rectangle\]')


def _prim_ops(*outs):
    buf = io.StringIO()
    mx.export_to_dot(buf, *[a for a in outs if isinstance(a, mx.array)])
    return Counter(_DOT_RECT.findall(buf.getvalue()))


# --------------------------------------------------------------------------- #
# 1. exactness: flag on vs off is byte-identical over 64 draft steps (f32 head)
# --------------------------------------------------------------------------- #
def test_draft_head_flag_byte_identical_over_64_steps():
    """On the tiny f32 head the cast is a no-op, so the bf16 branch is byte-for-byte
    the default branch -- proven over 64 distinct draft steps (draft tokens, logits,
    confidence all identical), and the block is non-degenerate."""
    _, model = _seeded_model(seed=0)
    main_h, caches = _seed_caches(model)

    seen = set()
    n_steps = 64
    for i in range(n_steps):
        primary = i % model.args.vocab_size
        dsp._DRAFT_HEAD_BF16 = False
        o_off, l_off, c_off = _draft_step(model, main_h, caches, primary)
        dsp._DRAFT_HEAD_BF16 = True
        o_on, l_on, c_on = _draft_step(model, main_h, caches, primary)

        assert mx.array_equal(o_off, o_on), f"step {i}: draft tokens diverged"
        assert mx.array_equal(l_off, l_on), f"step {i}: draft logits diverged"
        assert mx.array_equal(c_off, c_on), f"step {i}: confidence diverged"
        assert int(np.asarray(o_off).reshape(-1).size) == model.mtp.block_size + 1
        seen.add(tuple(np.asarray(o_off).reshape(-1).tolist()))

    # premise: the 64 steps are a real (varied) draft signal, not one repeated token
    assert len(seen) > 1, "draft steps were degenerate (all identical outputs)"


# --------------------------------------------------------------------------- #
# 2. engagement counter: one head call per draft_block, on the branch the flag picks
# --------------------------------------------------------------------------- #
def test_draft_head_engagement_counter():
    _, model = _seeded_model(seed=1)
    main_h, caches = _seed_caches(model)

    dsp._reset_draft_head_calls()
    dsp._DRAFT_HEAD_BF16 = True
    for i in range(5):
        _draft_step(model, main_h, caches, i)
    calls = dsp._draft_head_calls()
    assert calls == {"bf16": 5, "default": 0}, calls

    dsp._reset_draft_head_calls()
    dsp._DRAFT_HEAD_BF16 = False
    for i in range(3):
        _draft_step(model, main_h, caches, i)
    calls = dsp._draft_head_calls()
    assert calls == {"bf16": 0, "default": 3}, calls


# --------------------------------------------------------------------------- #
# 3. flag is read at use (env), never frozen at import
# --------------------------------------------------------------------------- #
def test_flag_read_at_use_from_env():
    dsp._DRAFT_HEAD_BF16 = None  # fall through to the env key
    for off in ("", "0", "false", "no", "off", "auto"):
        os.environ[dsp._DRAFT_HEAD_BF16_ENV] = off
        assert dsp._draft_head_bf16_on() is False, off
    for on in ("1", "true", "yes", "on", "bf16"):
        os.environ[dsp._DRAFT_HEAD_BF16_ENV] = on
        assert dsp._draft_head_bf16_on() is True, on
    # module-global pin overrides the env
    os.environ[dsp._DRAFT_HEAD_BF16_ENV] = "0"
    dsp._DRAFT_HEAD_BF16 = True
    assert dsp._draft_head_bf16_on() is True


# --------------------------------------------------------------------------- #
# 4. source dtype: dense float head -> its weight dtype; quantized head -> f32
# --------------------------------------------------------------------------- #
def test_head_source_dtype_by_codec():
    import mlx.nn as nn

    bf16_head = nn.Linear(32, 40, bias=False)
    bf16_head.weight = bf16_head.weight.astype(mx.bfloat16)
    mx.eval(bf16_head.parameters())
    assert dsp._draft_head_source_dtype(bf16_head) == mx.bfloat16

    f32_head = nn.Linear(32, 40, bias=False)
    mx.eval(f32_head.parameters())
    assert dsp._draft_head_source_dtype(f32_head) == mx.float32

    q_head = nn.QuantizedLinear.from_linear(f32_head, group_size=32, bits=8)
    mx.eval(q_head.parameters())
    # a quantized head carries .scales -> stay f32 (quantized_matmul dequants per group)
    assert dsp._draft_head_source_dtype(q_head) == mx.float32


# --------------------------------------------------------------------------- #
# 5. bf16 head: the flag removes the f32 weight-promotion (the trap), keeps counter
# --------------------------------------------------------------------------- #
def test_bf16_head_removes_weight_promotion():
    """On a bf16 head the DEFAULT branch promotes the [vocab, hidden] weight to a
    f32 temporary (a materialized f32 copy of the whole head every cycle); the bf16
    branch does not.  Proven structurally: with the input pre-cast so it needs no
    cast, the only bf16->f32 AsType in the default head matmul is the weight, and it
    is absent in the bf16 branch."""
    _, model = _seeded_model(seed=2, head_bf16=True)
    head = model.head
    W = head.weight
    assert W.dtype == mx.bfloat16
    hidden = mx.random.normal((1, 5, model.args.hidden_size))
    xf = hidden.astype(mx.float32)          # what the DEFAULT branch feeds head()
    xb = hidden.astype(W.dtype)             # what the bf16 branch feeds head()
    Wf32 = W.astype(mx.float32)             # a pre-eval'd f32 weight leaf (control)
    mx.eval(W, xf, xb, Wf32)

    default_ops = _prim_ops(xf @ W.T)                       # promotes W -> f32
    bf16_ops = _prim_ops((xb @ W.T).astype(mx.float32))     # bf16 GEMV, f32 logits
    # xf is a pre-eval'd f32 leaf, so its AsType is elided; the sole remaining
    # bf16->f32 cast in the default matmul is the weight promotion.
    assert default_ops.get("AsType", 0) == 1, default_ops
    # the bf16 branch's only AsType is the tiny f32 logits cast, never the weight.
    assert bf16_ops.get("AsType", 0) == 1, bf16_ops
    # control: an already-f32 weight leaf needs no promotion at all.
    assert _prim_ops(xf @ Wf32.T).get("AsType", 0) == 0

    # and the forward_head lever engages / dtype is f32 logits on both branches
    main_h, caches = _seed_caches(model)
    dsp._reset_draft_head_calls()
    dsp._DRAFT_HEAD_BF16 = True
    o_on, l_on, _ = _draft_step(model, main_h, caches, 7)
    assert dsp._draft_head_calls()["bf16"] == 1
    assert l_on.dtype == mx.float32
    assert int(np.asarray(o_on).reshape(-1).size) == model.mtp.block_size + 1
