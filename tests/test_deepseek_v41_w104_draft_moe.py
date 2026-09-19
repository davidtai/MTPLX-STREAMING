"""W104: the DSpark DRAFT block's resident MoE is a barrier-free resident
``mx.gather_qmm(mode="mxfp4")`` that issues ZERO host syncs per draft cycle.

Subject: ``DSparkBlock.mlp(moe_input)`` only (the 3 MTP stages' resident 128-expert
top-3 MoE).  ``self.mlp`` is a plain ``deepseek_v41_moe.MoE`` whose ``switch_mlp`` is
an mlx-lm ``SwitchGLU`` whose ``SwitchLinear`` leaves are repacked to
``QuantizedSwitchLinear(mode="mxfp4", gs32)`` by ``Model._build_mtp_head``.  It is NOT
rebound to the streamed ``HotExpertSwitchGLU`` (``bind_streamed_switches`` walks only
``model.model.layers``; the MTP stages live in ``model.mtp.layers``), so there is no
``mx.eval(indices)`` routing barrier, no ``.tolist()`` route plan, no layer lock, no
deferred release and no per-call dequant / repack.

W104 traced the draft MoE to this already-barrier-free resident path, so the task's
conditional ``MTPLX_DSV41_DRAFT_RESIDENT_MOE`` lever was NOT built (it would only
duplicate the current path); these tests lock the properties of the EXISTING path and
the existing draft levers (``DRAFT_COMPILE`` K33, ``DRAFT_HEAD_BF16`` W103).  See
``docs/deepseek-v41/W104_DRAFT_RESIDENT_MOE.md`` and the sync census
``scripts/deepseek_v41/w104_draft_moe_sync_census.py``.

CPU device (MLX fp32 bit-exact), tiny seeded config, no artifact, no GPU.
"""
import os

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

import mtplx.models.deepseek_v41 as dv41  # noqa: E402
import mtplx.models.deepseek_v41_dspark as dsp  # noqa: E402
from mtplx.models.deepseek_v41 import (  # noqa: E402
    Model,
    ModelArgs,
    _make_mtp_expert_quant_predicate,
)
from mtplx.models.deepseek_v41_moe import MoE  # noqa: E402

# Source files that mean "a host sync came from the routed MoE path".
_MOE_FILES = ("deepseek_v41_moe.py", "switch_layers.py", "expert_mlx.py")


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
    prev_compile = dsp._DRAFT_COMPILE
    prev_head = dsp._DRAFT_HEAD_BF16
    prev_env = os.environ.get(dsp._DRAFT_HEAD_BF16_ENV)
    dsp._reset_draft_head_calls()
    try:
        yield
    finally:
        dsp._DRAFT_COMPILE = prev_compile
        dsp._DRAFT_HEAD_BF16 = prev_head
        dsp._reset_draft_head_calls()
        dsp._DRAFT_COMPILED.clear()
        dv41._ATTN_COMPILED.clear()
        dv41._HC_COMPILED.clear()
        if prev_env is None:
            os.environ.pop(dsp._DRAFT_HEAD_BF16_ENV, None)
        else:
            os.environ[dsp._DRAFT_HEAD_BF16_ENV] = prev_env


# --------------------------------------------------------------------------- #
# tiny real-structure DSpark head (3 stages, resident 8-expert top-2 MoE == the
# 128-expert top-3 structure at small scale)
# --------------------------------------------------------------------------- #
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


def _seeded_model(seed=0, vocab=64, **over):
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
    mx.eval(model.parameters())
    return args, model


_PROMPT = [3, 5, 7, 9, 11, 13, 2, 4, 6, 8, 10, 12, 1, 0, 15, 17, 19]


def _seed_caches(model):
    ids = mx.array(_PROMPT).reshape(1, -1)
    logits, main_hidden = model(ids, return_hidden=True)
    mx.eval(logits, main_hidden)
    caches = model.make_mtp_cache()
    model.mtp.seed_main(main_hidden, caches)
    mx.eval([c.window for c in caches if c.window is not None])
    return main_hidden[:, -1:, :], caches


# --------------------------------------------------------------------------- #
# a host-sync counter that tags every mx.eval / async_eval / tolist / item with
# the deepest project frame that triggered it (so a sync can be blamed on a file)
# --------------------------------------------------------------------------- #
class _SyncSpy:
    def __init__(self):
        import traceback
        self._tb = traceback
        self.calls = []  # (kind, "file:line")
        self.recording = False
        self._orig = {}

    def _tag(self):
        for fr in reversed(self._tb.extract_stack()[:-2]):
            base = os.path.basename(fr.filename)
            if base == os.path.basename(__file__):
                continue
            if base.startswith("<") or "/mlx/" in fr.filename:
                continue
            return f"{base}:{fr.lineno}"
        return "<unknown>"

    def __enter__(self):
        self._orig = {
            "eval": mx.eval, "async_eval": mx.async_eval,
            "tolist": mx.array.tolist, "item": mx.array.item,
        }

        def wrap(kind, orig):
            def w(*a, **k):
                if self.recording:
                    self.calls.append((kind, self._tag()))
                return orig(*a, **k)
            return w

        mx.eval = wrap("mx.eval", self._orig["eval"])
        mx.async_eval = wrap("mx.async_eval", self._orig["async_eval"])
        mx.array.tolist = wrap("tolist", self._orig["tolist"])
        mx.array.item = wrap("item", self._orig["item"])
        return self

    def __exit__(self, *exc):
        mx.eval = self._orig["eval"]
        mx.async_eval = self._orig["async_eval"]
        mx.array.tolist = self._orig["tolist"]
        mx.array.item = self._orig["item"]

    def moe_calls(self):
        return [c for c in self.calls if c[1].split(":")[0] in _MOE_FILES]


# --------------------------------------------------------------------------- #
# 1. the resident draft MoE issues ZERO host syncs per draft cycle (64 cycles)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("draft_compile", [False, True])
def test_draft_moe_issues_no_host_sync_over_64_cycles(draft_compile):
    _, model = _seeded_model(seed=0)
    main_h, caches = _seed_caches(model)
    embed, head = model.model.embed_tokens, model.head
    dsp._DRAFT_COMPILE = draft_compile
    dsp._DRAFT_COMPILED.clear()

    with _SyncSpy() as spy:
        for i in range(64):
            spy.recording = True
            out_ids, dlogits, conf = model.mtp.draft_block(
                main_h, mx.array([i % model.args.vocab_size]), caches, embed, head
            )
            spy.recording = False
            # the caller's single terminal eval (NOT recorded) -- drains all 3 stages
            # at once so the lazy graph does not grow across the 64 cycles.
            spy._orig["eval"](out_ids, dlogits, conf)

    # HARD (W104 subject): the routed MoE path issued no host sync in any cycle.
    assert spy.moe_calls() == [], (
        f"draft MoE issued host syncs (compile={draft_compile}): {spy.moe_calls()[:8]}"
    )
    # LOCK: the whole draft cycle is lazy -- the only per-cycle sync is the caller's
    # terminal eval above (not recorded).  If this ever fails OUTSIDE the MoE files,
    # a non-MoE draft stage began syncing (not W104's subject); the MoE assert above
    # is the authoritative one.
    assert spy.calls == [], (
        f"a draft-cycle host sync appeared (compile={draft_compile}): {spy.calls[:8]}"
    )


# --------------------------------------------------------------------------- #
# 2. the real resident mxfp4 path (QuantizedSwitchLinear -> gather_qmm(mode=mxfp4))
#    is sync-free on an isolated, gs32-aligned MoE
# --------------------------------------------------------------------------- #
def _mxfp4_moe(seed=3, hidden=64, inter=64, experts=8, topk=3):
    mx.random.seed(seed)
    args = ModelArgs(
        vocab_size=64, hidden_size=hidden, num_hidden_layers=2, num_attention_heads=4,
        head_dim=16, moe_intermediate_size=inter, n_routed_experts=experts,
        num_experts_per_tok=topk, hc_mult=4, scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5, swiglu_limit=10.0, n_shared_experts=1,
        norm_topk_prob=False,
    )
    moe = MoE(40, args)  # DSpark-style resident stage MoE (layer_id 40)
    filled = []
    for name, value in tree_flatten(moe.parameters()):
        new = (mx.random.normal(value.shape) * (value.shape[-1] ** -0.5)
               if value.ndim >= 2 else mx.random.normal(value.shape) * 0.1)
        filled.append((name, new.astype(value.dtype)))
    moe.update(tree_unflatten(filled))
    nn.quantize(moe, group_size=32, bits=4, mode="mxfp4",
                class_predicate=_make_mtp_expert_quant_predicate(32))
    mx.eval(moe.parameters())
    return moe, args


def test_resident_mxfp4_switch_is_gather_qmm_and_sync_free():
    moe, args = _mxfp4_moe()
    # the routed experts really became the mxfp4 gather_qmm leaf (not bf16 gather_mm).
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
    assert isinstance(moe.switch_mlp.gate_proj, QuantizedSwitchLinear)
    assert isinstance(moe.switch_mlp.up_proj, QuantizedSwitchLinear)
    assert isinstance(moe.switch_mlp.down_proj, QuantizedSwitchLinear)
    assert moe.switch_mlp.gate_proj.mode == "mxfp4"
    assert moe.switch_mlp.gate_proj.bits == 4 and moe.switch_mlp.gate_proj.group_size == 32
    assert moe.switch_mlp.gate_proj.get("biases") is None  # mxfp4: E8M0 scales, no bias

    x = mx.random.normal((1, 5, args.hidden_size)).astype(mx.bfloat16)
    mx.eval(x)
    with _SyncSpy() as spy:
        spy.recording = True
        y = moe(x)
        spy.recording = False
        spy._orig["eval"](y)
    assert spy.moe_calls() == [], f"mxfp4 resident MoE synced: {spy.moe_calls()[:8]}"
    assert y.shape == (1, 5, args.hidden_size)


# --------------------------------------------------------------------------- #
# 3. DRAFT_COMPILE brackets the MoE (_moe_compile_window); on==off is byte-identical
#    over 64 draft cycles.  (No new resident path was added -- this locks the
#    existing MoE compile-window bit-exactness, the "identity vs the current path"
#    the task asks for.)
# --------------------------------------------------------------------------- #
def test_draft_compile_moe_window_byte_identical_over_64_cycles():
    _, model = _seeded_model(seed=1)
    main_h, caches = _seed_caches(model)
    embed, head = model.model.embed_tokens, model.head
    dsp._DRAFT_HEAD_BF16 = False  # isolate the K33 compile window from the head lever

    seen = set()
    for i in range(64):
        primary = mx.array([i % model.args.vocab_size])

        dsp._DRAFT_COMPILE = False
        dsp._DRAFT_COMPILED.clear()
        dv41._ATTN_COMPILED.clear()
        dv41._HC_COMPILED.clear()
        o0, l0, c0 = model.mtp.draft_block(main_h, primary, caches, embed, head)
        mx.eval(o0, l0, c0)

        dsp._DRAFT_COMPILE = True
        dsp._DRAFT_COMPILED.clear()
        dv41._ATTN_COMPILED.clear()
        dv41._HC_COMPILED.clear()
        o1, l1, c1 = model.mtp.draft_block(main_h, primary, caches, embed, head)
        mx.eval(o1, l1, c1)

        assert mx.array_equal(o0, o1), f"cycle {i}: draft tokens diverged (MoE window)"
        assert mx.array_equal(l0, l1), f"cycle {i}: draft logits diverged (MoE window)"
        assert mx.array_equal(c0, c1), f"cycle {i}: confidence diverged (MoE window)"
        seen.add(tuple(np.asarray(o0).reshape(-1).tolist()))

    assert len(seen) > 1, "draft cycles were degenerate (all identical outputs)"


# --------------------------------------------------------------------------- #
# 4. engagement counter (one head call per draft cycle) + env read at use.  W103
#    owns the deep version (tests/test_deepseek_v41_w103_draft.py); this keeps the
#    W104 file self-contained for the counter the DSpark receipt surfaces.
# --------------------------------------------------------------------------- #
def test_draft_head_engagement_counter_one_per_cycle():
    _, model = _seeded_model(seed=2)
    main_h, caches = _seed_caches(model)
    embed, head = model.model.embed_tokens, model.head

    dsp._reset_draft_head_calls()
    dsp._DRAFT_HEAD_BF16 = True
    for i in range(6):
        out = model.mtp.draft_block(main_h, mx.array([i]), caches, embed, head)
        mx.eval(*out)
    assert dsp._draft_head_calls() == {"bf16": 6, "default": 0}

    dsp._reset_draft_head_calls()
    dsp._DRAFT_HEAD_BF16 = False
    for i in range(4):
        out = model.mtp.draft_block(main_h, mx.array([i]), caches, embed, head)
        mx.eval(*out)
    assert dsp._draft_head_calls() == {"bf16": 0, "default": 4}


def test_draft_head_bf16_env_read_at_use():
    dsp._DRAFT_HEAD_BF16 = None  # fall through to the env key, read at each call
    for off in ("", "0", "false", "no", "off", "auto"):
        os.environ[dsp._DRAFT_HEAD_BF16_ENV] = off
        assert dsp._draft_head_bf16_on() is False, off
    for on in ("1", "true", "yes", "on", "bf16"):
        os.environ[dsp._DRAFT_HEAD_BF16_ENV] = on
        assert dsp._draft_head_bf16_on() is True, on


def test_draft_compile_env_read_at_use():
    dsp._DRAFT_COMPILE = None  # fall through to the env key, read at each call
    prev = os.environ.get(dsp._DRAFT_COMPILE_ENV)
    try:
        for off in ("", "0", "false", "no", "off", "auto"):
            os.environ[dsp._DRAFT_COMPILE_ENV] = off
            assert dsp._draft_compile_on() is False, off
        for on in ("1", "true", "yes", "on"):
            os.environ[dsp._DRAFT_COMPILE_ENV] = on
            assert dsp._draft_compile_on() is True, on
    finally:
        if prev is None:
            os.environ.pop(dsp._DRAFT_COMPILE_ENV, None)
        else:
            os.environ[dsp._DRAFT_COMPILE_ENV] = prev
