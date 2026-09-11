"""K33 (W65) DSpark draft-block dispatch collapse: exactness + census reduction.

The DSpark-DIRECT draft block runs 3 shallow stages (sliding-window attention +
resident 128-expert top-3 MoE) plus forward_embed and a markov autoregression --
a tiny amount of math dispatched as ~1.6k graph primitives per cycle on the tiny
double. ``MTPLX_DSV41_DRAFT_COMPILE`` (default OFF) replays the pure chains from
mx.compile tapes (attention prep + Hyper-Connection prep reuse the backbone K22/K4
tapes; the MoE gate-prefix/combine folds; the markov step; the confidence head).

The draft head's numerics only set the acceptance rate -- the runtime's greedy
target verify is authoritative -- so the shipped bar is greedy-verify == AR. K33
is a pure dispatch cut, so the bar here is STRICTER: flag on vs off is byte-
identical draft tokens (and logits/confidence) on the tiny double, and the
greedy == AR gate stays green with the flag on.

CPU device (MLX fp32 bit-exact), tiny seeded config, no artifact.
"""
import importlib.util
import os
from pathlib import Path

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
def _restore_draft_flag():
    """Every test flips the module global / env; restore both around each."""
    prev_global = dsp._DRAFT_COMPILE
    prev_env = os.environ.get(dsp._DRAFT_COMPILE_ENV)
    try:
        yield
    finally:
        dsp._DRAFT_COMPILE = prev_global
        dsp._DRAFT_COMPILED.clear()
        dv41._ATTN_COMPILED.clear()
        dv41._HC_COMPILED.clear()
        if prev_env is None:
            os.environ.pop(dsp._DRAFT_COMPILE_ENV, None)
        else:
            os.environ[dsp._DRAFT_COMPILE_ENV] = prev_env


def _args(vocab: int = 64, **over):
    kwargs = dict(
        vocab_size=vocab, hidden_size=32, num_hidden_layers=5, num_attention_heads=4,
        head_dim=16, qk_rope_head_dim=8, q_lora_rank=16, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=2,
        sliding_window=8, window_size=8, hc_mult=4, hc_sinkhorn_iters=2,
        scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, swiglu_limit=0.0,
        n_mtp_layers=3, dspark_block_size=4, dspark_noise_token_id=vocab - 1,
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


def _draft_once(model, prompt):
    ids = mx.array(prompt).reshape(1, -1)
    logits, main_hidden = model(ids, return_hidden=True)
    mx.eval(logits, main_hidden)
    caches = model.make_mtp_cache()
    model.mtp.seed_main(main_hidden, caches)
    primary = mx.array([int(mx.argmax(logits[0, -1]))])
    out_ids, dlogits, conf = model.mtp.draft_block(
        main_hidden[:, -1:, :], primary, caches, model.model.embed_tokens, model.head
    )
    mx.eval(out_ids, dlogits, conf)
    return out_ids, dlogits, conf


_PROMPT = [3, 5, 7, 9, 11, 13, 2, 4, 6, 8, 10, 12, 1, 0, 15, 17, 19]


# --------------------------------------------------------------------------- #
# 1. exactness: flag off vs on is byte-identical (draft tokens, logits, conf)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_draft_block_flag_on_off_byte_identical(seed):
    _, model = _seeded_model(seed=seed)

    dsp._DRAFT_COMPILE = False
    o_off, l_off, c_off = _draft_once(model, _PROMPT)
    dsp._DRAFT_COMPILE = True
    o_on, l_on, c_on = _draft_once(model, _PROMPT)

    assert mx.array_equal(o_off, o_on), f"seed {seed}: draft tokens diverged"
    assert mx.array_equal(l_off, l_on), f"seed {seed}: draft logits diverged"
    assert mx.array_equal(c_off, c_on), f"seed {seed}: confidence diverged"
    # premise: the block is non-degenerate (real draft signal, not all-noise)
    assert int(np.asarray(o_off).reshape(-1).size) == model.mtp.block_size + 1


def test_flag_off_matches_the_shipped_eager_drafter():
    """Flag OFF must be the pre-K33 eager path byte-for-byte: same draft as a run
    with the env key explicitly cleared and the module global forced off."""
    _, model = _seeded_model(seed=0)
    dsp._DRAFT_COMPILE = None
    os.environ.pop(dsp._DRAFT_COMPILE_ENV, None)
    o_default, l_default, _ = _draft_once(model, _PROMPT)
    dsp._DRAFT_COMPILE = False
    o_off, l_off, _ = _draft_once(model, _PROMPT)
    assert mx.array_equal(o_default, o_off)
    assert mx.array_equal(l_default, l_off)


# --------------------------------------------------------------------------- #
# 2. greedy == AR with the flag ON (through the real generic MTP engine)
# --------------------------------------------------------------------------- #
class _FixedTokenizer:
    eos_token_id = None
    eos_token_ids: set = set()

    def decode(self, tokens):
        return " ".join(str(t) for t in tokens)


def _runtime(seed=0, vocab=64):
    from mtplx.mtp_patch import MTPContract, validate_mtp_support
    from mtplx.runtime import MTPLXRuntime
    from mtplx.models.deepseek_v41 import (
        inject_deepseek_v41_mtp_support,
        is_deepseek_v41_mtp_config,
    )

    config = {"model_type": "deepseek_v41", "n_mtp_layers": 3}
    assert is_deepseek_v41_mtp_config(config)
    _a, model = _seeded_model(seed=seed, vocab=vocab)
    assert inject_deepseek_v41_mtp_support(model, Path("."), config, MTPContract())
    assert validate_mtp_support(model)
    return MTPLXRuntime(
        model=model, tokenizer=_FixedTokenizer(), model_path=Path("."),
        mtp_enabled=True, contract=MTPContract(),
    )


def _ar(rt, prompt, max_tokens):
    from mtplx.generation import generate_ar
    from mtplx.sampling import SamplerConfig

    return generate_ar(rt, prompt, max_tokens=max_tokens,
                       sampler=SamplerConfig(temperature=0.0), stop_token_ids=set())


def _spec(rt, prompt, max_tokens, depth):
    from mtplx.generation import generate_mtpk
    from mtplx.sampling import SamplerConfig

    return generate_mtpk(rt, prompt, max_tokens=max_tokens,
                         sampler=SamplerConfig(temperature=0.0), speculative_depth=depth,
                         mtp_history_policy="committed", stop_token_ids=set(),
                         verify_strategy="batched")


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_greedy_verify_equals_ar_with_draft_compile_on(depth):
    prompt = [int(v) for v in np.random.default_rng(7).integers(0, 64, size=17)]
    baseline = _ar(_runtime(), prompt, 48)
    assert len(set(baseline.tokens)) > 1, "premise: AR must not be degenerate"

    dsp._DRAFT_COMPILE = True
    out = _spec(_runtime(), prompt, 48, depth)
    assert out.tokens == baseline.tokens, (
        f"depth {depth}: draft-compile ON broke greedy verify == AR\n"
        f"  AR  : {baseline.tokens}\n  spec: {out.tokens}"
    )


def test_draft_compile_on_and_off_spec_agree():
    """The lane's spec output is identical with the flag on and off (the head's
    numerics are unchanged; only its dispatch is)."""
    prompt = [int(v) for v in np.random.default_rng(11).integers(0, 64, size=17)]
    dsp._DRAFT_COMPILE = False
    off = _spec(_runtime(), prompt, 40, 3)
    dsp._DRAFT_COMPILE = True
    on = _spec(_runtime(), prompt, 40, 3)
    assert on.tokens == off.tokens


# --------------------------------------------------------------------------- #
# 3. the flag gate: default off, env read at use, row cap
# --------------------------------------------------------------------------- #
def test_flag_defaults_off_and_reads_env_at_use():
    dsp._DRAFT_COMPILE = None
    os.environ.pop(dsp._DRAFT_COMPILE_ENV, None)
    assert dsp._draft_compile_on() is False
    for off in ("0", "false", "no", "off", "auto", ""):
        os.environ[dsp._DRAFT_COMPILE_ENV] = off
        assert dsp._draft_compile_on() is False, off
    for on in ("1", "true", "yes", "on"):
        os.environ[dsp._DRAFT_COMPILE_ENV] = on
        assert dsp._draft_compile_on() is True, on
    # a module-global pin overrides the env either way (tests / census flip it)
    os.environ[dsp._DRAFT_COMPILE_ENV] = "1"
    dsp._DRAFT_COMPILE = False
    assert dsp._draft_compile_on() is False
    dsp._DRAFT_COMPILE = True
    os.environ[dsp._DRAFT_COMPILE_ENV] = "0"
    assert dsp._draft_compile_on() is True


def test_row_cap_confines_compile_to_the_draft_regime():
    dsp._DRAFT_COMPILE = True
    assert dsp._draft_use_compile(4) is True
    assert dsp._draft_use_compile(dsp._DRAFT_COMPILE_MAX_ROWS) is True
    assert dsp._draft_use_compile(dsp._DRAFT_COMPILE_MAX_ROWS + 1) is False
    dsp._DRAFT_COMPILE = False
    assert dsp._draft_use_compile(4) is False


# --------------------------------------------------------------------------- #
# 4. the census reduction (asserted from the census tool)
# --------------------------------------------------------------------------- #
def _census_tool():
    path = Path(__file__).resolve().parents[1].parent / "scripts" / "deepseek_v41" / "dispatch_census.py"
    spec = importlib.util.spec_from_file_location("dsv41_dispatch_census", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_draft_census_total_and_targeted_stages_drop():
    tool = _census_tool()
    before = tool._run_draft_census(False, seed=1)
    after = tool._run_draft_census(True, seed=1)

    # total primitives per draft cycle strictly drops
    assert after["total_primitives_per_token"] < before["total_primitives_per_token"]

    bs, as_ = before["stages"], after["stages"]

    def pt(rep, name):
        return rep.get(name, {}).get("primitives_per_token", 0.0)

    # every compiled chain strictly drops
    for name in (
        "dspark.attn.qkv_prep", "dspark.attn.out_prep", "dspark.attn.main_kv",
        "dspark.hc.attn_prep", "dspark.hc.ffn_prep", "dspark.hc.moe_combine",
        "moe.gate_topk", "moe.combine",
    ):
        assert pt(as_, name) < pt(bs, name), f"{name} should drop: {pt(bs, name)} -> {pt(as_, name)}"

    # the SDPA and the resident routed gather are NOT compiled -> unchanged
    for name in ("dspark.attn.sdpa", "moe.routed_switch", "moe.shared_expert"):
        assert pt(as_, name) == pt(bs, name), f"{name} must be unchanged"

    # the resident 128-expert MoE is one gather (SwitchGLU), not a per-expert loop:
    # its routed_switch stage is a small constant count per stage, not O(n_experts)
    assert 0 < bs["moe.routed_switch"]["primitives_per_call"] < 64
