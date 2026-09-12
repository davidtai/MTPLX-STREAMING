"""Regression: the ab A/B harness's DSpark lane must call the public
``dspark_generate`` with a signature it actually accepts.

Window-34 steps 2-3 crashed at the *first* DSpark call: W81's
``scripts/deepseek_v41/ab_decode_env_levers.py`` ``_generate_dspark`` passes
``prefill_callback=`` to :func:`mtplx.models.deepseek_v41_dspark_decode.dspark_generate`,
but that kwarg existed only on the served-lane ``generate_dspark``.  The merge
batch stayed green because no CPU test drove the ab->dspark call end to end --
every existing DSpark test called ``dspark_generate`` directly with a hand-built
kwarg set.  These tests close that gap on the tiny fake model:

  1. the ab ``_generate_dspark`` runs end to end (the exact crashing call);
  2. ``dspark_generate`` fires ``prefill_callback`` once with the served-lane
     payload keys, right after prefill;
  3. an AST + :func:`inspect.signature` guard: every keyword the ab script passes
     to ``dspark_generate`` is a real parameter (what window-34 violated);
  4. a raising callback never aborts decode (telemetry is best-effort).

Self-contained: shrunk seeded config, CPU device (MLX fp32 bit-exact), no
downloads/checkpoint/experts.bin.  MLX is pinned to the CPU per
memory/worker-tests-must-pin-mlx-cpu.md.
"""
import ast
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402
from mtplx.models.deepseek_v41_dspark_decode import dspark_generate  # noqa: E402
from mtplx.sampling import SamplerConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


_WT = Path(__file__).resolve().parents[2]
_AB_PATH = _WT / "scripts" / "deepseek_v41" / "ab_decode_env_levers.py"

DIM = 32
N_LAYERS = 5


def _args(vocab: int = 64, **over):
    # Matches tests/models/test_deepseek_v41_dspark_decode.py's shrunk config so the
    # tiny model carries a working DSpark MTP head on the CPU.
    kwargs = dict(
        vocab_size=vocab,
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
        dspark_noise_token_id=vocab - 1,
        dspark_target_layer_ids=[2, 3, 4],
        dspark_markov_rank=12,
        dspark_n_routed_experts=8,
        dspark_num_experts_per_tok=2,
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
            noise = mx.random.normal(value.shape) * 0.1
            centre = 1.0 if leaf.endswith("norm_weight") or leaf == "scale" else 0.0
            new = noise + centre
        else:
            new = mx.random.normal(value.shape) * (value.shape[-1] ** -0.5)
        filled.append((name, new.astype(value.dtype)))
    model.update(tree_unflatten(filled))
    mx.eval(model.parameters())
    return args, model


def _prompt(n, vocab=64, seed=7):
    rng = np.random.default_rng(seed)
    return [int(v) for v in rng.integers(0, vocab, size=n)]


def _load_ab():
    """Load the ab A/B script by file path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("dsv41_ab_dspark_cb", _AB_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _MemProbe:
    """Minimal probe: ``_generate_dspark`` only calls reset_peak()/peak_bytes()."""

    def reset_peak(self):
        pass

    def peak_bytes(self):
        return 0


def test_ab_generate_dspark_end_to_end_with_prefill_callback():
    """Drive the ab harness's ``_generate_dspark`` end to end -- the exact call
    window-34 crashed on (``dspark_generate(..., prefill_callback=...)``)."""
    ab = _load_ab()
    _args_, model = _seeded_model(seed=0)
    prompt = _prompt(6)
    steps = 4
    out = ab._generate_dspark(
        model=model,
        mx=mx,
        mem_probe=_MemProbe(),
        prompt_ids=prompt,
        steps=steps,
        depth=1,
    )
    assert isinstance(out, dict)
    # prefill token + ``steps`` decode tokens, matching _generate.
    assert len(out["generated"]) == steps + 1
    # the prefill_callback fired inside _generate_dspark (its snapshot key is
    # present even though it is None on a model with no expert-streaming runtime).
    assert "stream_after_prefill" in out


def test_dspark_generate_prefill_callback_fires_once_after_prefill():
    """``dspark_generate`` invokes ``prefill_callback`` exactly once with the
    served-lane payload keys."""
    _args_, model = _seeded_model(seed=0)
    prompt = _prompt(6)
    calls = []
    toks = dspark_generate(
        model,
        prompt,
        max_tokens=5,
        sampler=SamplerConfig(temperature=0.0),
        seed=0,
        speculative_depth=1,
        prefill_callback=lambda info: calls.append(info),
    )
    assert len(toks) == 5
    assert len(calls) == 1
    info = calls[0]
    assert set(info) == {"prompt_tokens", "prompt_eval_time_s"}
    assert info["prompt_tokens"] == len(prompt)
    assert isinstance(info["prompt_eval_time_s"], float)
    assert info["prompt_eval_time_s"] >= 0.0


def test_dspark_generate_without_callback_is_unaffected():
    """Omitting ``prefill_callback`` keeps the prior behaviour (no crash, same
    token count) -- the new param is purely additive."""
    _args_, model = _seeded_model(seed=0)
    prompt = _prompt(6)
    toks = dspark_generate(
        model,
        prompt,
        max_tokens=5,
        sampler=SamplerConfig(temperature=0.0),
        seed=0,
        speculative_depth=1,
    )
    assert len(toks) == 5


def test_ab_dspark_generate_call_kwargs_match_signature():
    """AST + inspect.signature guard: every keyword the ab script passes to
    ``dspark_generate`` is a real parameter (``dspark_generate`` has no **kwargs),
    so a future signature drift like window-34's is caught statically."""
    sig_params = set(inspect.signature(dspark_generate).parameters)
    tree = ast.parse(_AB_PATH.read_text())
    calls = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        if name != "dspark_generate":
            continue
        calls += 1
        for kw in node.keywords:
            if kw.arg is None:  # **kwargs splat -- nothing to check
                continue
            assert kw.arg in sig_params, (
                f"ab script passes dspark_generate({kw.arg}=...) but that is not a "
                f"parameter of dspark_generate {sorted(sig_params)}"
            )
    assert calls >= 1, "expected at least one dspark_generate(...) call in the ab script"


def test_prefill_callback_error_does_not_crash_decode():
    """A raising ``prefill_callback`` must not abort decode -- telemetry is
    best-effort and swallowed by the shared helper."""
    _args_, model = _seeded_model(seed=0)
    prompt = _prompt(6)

    def _boom(_info):
        raise RuntimeError("telemetry boom")

    toks = dspark_generate(
        model,
        prompt,
        max_tokens=5,
        sampler=SamplerConfig(temperature=0.0),
        seed=0,
        speculative_depth=1,
        prefill_callback=_boom,
    )
    assert len(toks) == 5
