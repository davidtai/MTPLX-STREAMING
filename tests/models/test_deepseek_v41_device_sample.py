"""W63 / K32 device-sample AR decode lane gates.

The device-sample lane (:func:`mtplx.models.deepseek_v41_dspark_decode.
run_device_sample_decode`, armed by ``MTPLX_DSV41_DEVICE_SAMPLE=1``) keeps the
sampled token ON DEVICE across the whole decode: the id feeds the next forward's
embedding directly (``mx.take``) and the host reads it with a ONE-STEP LAG
(step t+1's forward already submitted via ``mx.async_eval`` before token t's id
is materialized), so the GPU never idles on the per-token device->host read that
makes DSV4.1 decode dispatch-bound.

Correctness bar (proven here on the CPU double, MLX fp32 bit-exact):
  1. GREEDY is BYTE-IDENTICAL to the classic argmax loop -- both at the model
     level (:func:`run_device_sample_decode` vs an argmax AR reference) and on
     the served lane (:func:`mtplx.generation.generate_ar` with the env on vs
     off), across 256 tokens.
  2. The one-step lag is correct at a STOP token and at ``max_tokens``: the
     emitted ids equal the classic up-to-stop output, exactly one extra forward
     is computed and discarded (bounded), and streaming == committed.
  3. SAMPLED is not token-for-token equal to the host numpy path (documented
     seed-mapping + narrower-nucleus deviation, W63_DEVICE_SAMPLE.md), but its
     support is a SUBSET of the host top-k / host support -- the device never
     draws a token the host would not.
  4. The lane engages ONLY for the deepseek_v41 model and only when armed, so no
     other model's path is touched, and it is OFF by default.

Self-contained: shrunk seeded config, CPU device, no downloads, no checkpoint,
no experts.bin, tiny RSS.
"""
import os
from pathlib import Path

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


@pytest.fixture(autouse=True)
def _clear_device_sample_env():
    saved = os.environ.get("MTPLX_DSV41_DEVICE_SAMPLE")
    os.environ.pop("MTPLX_DSV41_DEVICE_SAMPLE", None)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("MTPLX_DSV41_DEVICE_SAMPLE", None)
        else:
            os.environ["MTPLX_DSV41_DEVICE_SAMPLE"] = saved


from mtplx.models.deepseek_v41 import (  # noqa: E402
    Model,
    ModelArgs,
    inject_deepseek_v41_mtp_support,
)
from mtplx.models.deepseek_v41_dspark_decode import (  # noqa: E402
    device_sample_eligible,
    device_sample_enabled,
    run_device_sample_decode,
)
from mtplx.sampling import SamplerConfig  # noqa: E402

DIM = 32
N_LAYERS = 5


def _args(vocab: int = 64, **over):
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


class _FixedTokenizer:
    eos_token_id = None
    eos_token_ids: set = set()

    def decode(self, tokens):
        return " ".join(str(t) for t in tokens)


def _runtime(seed=0, vocab=64):
    from mtplx.mtp_patch import MTPContract
    from mtplx.runtime import MTPLXRuntime

    _args_, model = _seeded_model(seed=seed, vocab=vocab)
    cfg = {"model_type": "deepseek_v41", "n_mtp_layers": 3}
    assert inject_deepseek_v41_mtp_support(model, Path("."), cfg, MTPContract())
    return MTPLXRuntime(
        model=model,
        tokenizer=_FixedTokenizer(),
        model_path=Path("."),
        mtp_enabled=True,
        contract=MTPContract(),
    )


def _prompt(n, vocab=64, seed=7):
    rng = np.random.default_rng(seed)
    return [int(v) for v in rng.integers(0, vocab, size=n)]


def _ar_reference(model, prompt, n):
    """Independent greedy AR reference at the model level (argmax decode)."""
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)
    tok = int(mx.argmax(logits[0, -1]).item())
    out = [tok]
    for _ in range(n - 1):
        logits = model(mx.array([[tok]]), cache=cache)
        tok = int(mx.argmax(logits[0, -1]).item())
        out.append(tok)
    return out


def _device_sample_model(model, prompt, n_more, *, sampler=None, stop_ids=None,
                         on_token=None):
    """Device-sample decode at the model level: prefill, host-sample the first
    token (argmax for greedy), then run the one-step-lag pipeline for n_more."""
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)
    first = int(mx.argmax(logits[0, -1]).item())

    def _forward_row(ids):
        return model(ids, cache=cache)[0, -1]

    more, finish, extra = run_device_sample_decode(
        forward_row=_forward_row,
        first_token=first,
        n_more=int(n_more),
        sampler=sampler,
        stop_ids=stop_ids,
        on_token=on_token,
    )
    return [first] + more, finish, extra


GREEDY = SamplerConfig(temperature=0.0)


# --------------------------------------------------------------------------- #
# 1. greedy == AR, byte-for-byte over 256 tokens (model level)
# --------------------------------------------------------------------------- #
def test_device_sample_greedy_reproduces_ar_over_256_tokens():
    _args_, model = _seeded_model(seed=0)
    prompt = _prompt(17)
    ref = _ar_reference(model, prompt, 256)
    assert len(ref) == 256
    assert len(set(ref)) > 1, "premise: AR output must not be degenerate"

    out, finish, extra = _device_sample_model(model, prompt, 255, stop_ids=set())
    assert out == ref, (
        "device-sample greedy diverged from AR at "
        f"{next((i for i, (a, b) in enumerate(zip(out, ref)) if a != b), None)}"
    )
    assert len(out) == 256
    assert finish == "length"
    # exactly one extra forward computed + discarded (the classic loop breaks
    # before forwarding its final token; the lag pipeline computes that one step).
    assert extra == 1


# --------------------------------------------------------------------------- #
# 2. served generate_ar greedy: env on == env off, byte-for-byte, streamed too
# --------------------------------------------------------------------------- #
def test_generate_ar_greedy_device_sample_is_byte_identical():
    from mtplx.generation import generate_ar

    rt = _runtime(seed=0)
    prompt = _prompt(17)

    os.environ.pop("MTPLX_DSV41_DEVICE_SAMPLE", None)
    base = generate_ar(rt, prompt, max_tokens=64, sampler=GREEDY, stop_token_ids=set())
    assert not any(
        isinstance(e, dict) and e.get("device_sample") for e in base.stats.events
    ), "device-sample lane must be OFF by default"

    os.environ["MTPLX_DSV41_DEVICE_SAMPLE"] = "1"
    streamed: list[int] = []
    ds = generate_ar(
        rt, prompt, max_tokens=64, sampler=GREEDY, stop_token_ids=set(),
        token_callback=lambda d: streamed.extend(d),
    )
    assert list(ds.tokens) == list(base.tokens)
    assert streamed == list(ds.tokens), "streamed delta must equal committed tokens"
    assert ds.finish_reason == base.finish_reason == "length"
    assert any(
        isinstance(e, dict) and e.get("device_sample") for e in ds.stats.events
    ), "device-sample lane must record its engagement event"


# --------------------------------------------------------------------------- #
# 3. one-step lag is correct at a stop token (model level)
# --------------------------------------------------------------------------- #
def test_device_sample_stop_lag_matches_classic_up_to_stop():
    _args_, model = _seeded_model(seed=0)
    prompt = _prompt(17)
    ref = _ar_reference(model, prompt, 64)
    stop_tok = ref[9]
    first_stop_idx = ref.index(stop_tok)
    classic_up_to_stop = ref[: first_stop_idx + 1]

    streamed: list[int] = []
    out, finish, extra = _device_sample_model(
        model, prompt, 63, stop_ids={stop_tok}, on_token=streamed.append
    )
    assert out == classic_up_to_stop
    assert finish == "stop"
    assert out[-1] == stop_tok
    assert out.count(stop_tok) == 1
    # streamed on_token deltas exclude the first (prefill) token, include the stop.
    assert streamed == out[1:]
    # one extra forward computed + discarded at the stop.
    assert extra == 1


# --------------------------------------------------------------------------- #
# 4. served stop path byte-identical to the classic served path
# --------------------------------------------------------------------------- #
def test_generate_ar_stop_lag_byte_identical():
    from mtplx.generation import generate_ar

    rt = _runtime(seed=0)
    prompt = _prompt(17)
    os.environ.pop("MTPLX_DSV41_DEVICE_SAMPLE", None)
    full = list(
        generate_ar(rt, prompt, max_tokens=64, sampler=GREEDY, stop_token_ids=set()).tokens
    )
    stop = full[9]
    base = list(
        generate_ar(rt, prompt, max_tokens=64, sampler=GREEDY, stop_token_ids={stop}).tokens
    )

    os.environ["MTPLX_DSV41_DEVICE_SAMPLE"] = "1"
    ds = list(
        generate_ar(rt, prompt, max_tokens=64, sampler=GREEDY, stop_token_ids={stop}).tokens
    )
    assert ds == base
    assert ds[-1] == stop and ds.count(stop) == 1


# --------------------------------------------------------------------------- #
# 5. length finish honours n_more exactly; extra forward bounded to 1
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_more", [1, 2, 5, 40])
def test_device_sample_length_honours_n_more_and_extra_is_one(n_more):
    _args_, model = _seeded_model(seed=1)
    prompt = _prompt(17)
    ref = _ar_reference(model, prompt, n_more + 1)
    out, finish, extra = _device_sample_model(model, prompt, n_more, stop_ids=set())
    assert out == ref
    assert len(out) == n_more + 1
    assert finish == "length"
    assert extra == 1


# --------------------------------------------------------------------------- #
# 6. eligibility rules (greedy always; sampled needs top_k>1, no penalties)
# --------------------------------------------------------------------------- #
def test_device_sample_eligibility():
    ok, reason = device_sample_eligible(SamplerConfig(temperature=0.0))
    assert ok and reason == "greedy"
    # sampler=None means greedy too.
    ok, reason = device_sample_eligible(None)
    assert ok and reason == "greedy"

    ok, reason = device_sample_eligible(
        SamplerConfig(temperature=0.8, top_p=0.95, top_k=40)
    )
    assert ok and reason == "sampled"

    ok, _ = device_sample_eligible(SamplerConfig(temperature=0.8, top_k=1))
    assert not ok, "sampled with top_k<=1 must be ineligible for the device lane"
    ok, _ = device_sample_eligible(SamplerConfig(temperature=0.8, top_k=0))
    assert not ok

    ok, _ = device_sample_eligible(
        SamplerConfig(temperature=0.8, top_k=40, presence_penalty=0.5)
    )
    assert not ok, "presence/frequency penalties are host-only"


# --------------------------------------------------------------------------- #
# 7. sampled: valid completion, support is a subset of the host top-k / support
# --------------------------------------------------------------------------- #
def test_device_sample_sampled_support_is_subset_of_host():
    from mtplx.generation import _distribution_from_mlx_logits, _mx_lazy_sample

    _args_, model = _seeded_model(seed=2)
    prompt = _prompt(17)
    sampler = SamplerConfig(temperature=0.9, top_p=0.95, top_k=20)

    # valid completion end to end
    out, finish, extra = _device_sample_model(
        model, prompt, 32, sampler=sampler, stop_ids=set()
    )
    assert len(out) == 33
    assert all(0 <= t < 64 for t in out)
    assert extra == 1

    # For a fixed logits row, every device draw is inside the host support /
    # host top-k (the device nucleus is a prefix-subset of the host nucleus).
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)
    row = logits[0, -1]
    dist = _distribution_from_mlx_logits(row, sampler)
    if hasattr(dist, "token_ids"):  # SparseDistribution
        host_dense = np.asarray(dist.to_dense())
    else:
        host_dense = np.asarray(dist)
    host_support = set(np.nonzero(host_dense > 1e-12)[0].tolist())
    probs = np.asarray(mx.softmax(row.astype(mx.float32) / 0.9))
    host_topk = set(np.argsort(-probs)[: int(sampler.top_k)].tolist())

    key = mx.random.key(4242)
    drawn: set[int] = set()
    for _ in range(3000):
        key, sub = mx.random.split(key)
        drawn.add(int(_mx_lazy_sample(row, sampler, sub).item()))
    assert drawn <= host_topk, "device draw escaped the host top-k candidate set"
    assert drawn <= host_support, "device draw escaped the host nucleus support"


# --------------------------------------------------------------------------- #
# 8. the lane never engages for a non-deepseek_v41 model (other paths untouched)
# --------------------------------------------------------------------------- #
def test_generate_ar_non_dsv41_model_never_engages():
    from mtplx.generation import generate_ar

    rt = _runtime(seed=0)
    prompt = _prompt(17)
    os.environ["MTPLX_DSV41_DEVICE_SAMPLE"] = "1"
    original = rt.model.model_type
    rt.model.model_type = "some_other_model"
    try:
        out = generate_ar(rt, prompt, max_tokens=16, sampler=GREEDY, stop_token_ids=set())
    finally:
        rt.model.model_type = original
    assert not any(
        isinstance(e, dict) and e.get("device_sample") for e in out.stats.events
    ), "device-sample lane must not engage for a non-deepseek_v41 model"


def test_device_sample_enabled_reads_env():
    os.environ.pop("MTPLX_DSV41_DEVICE_SAMPLE", None)
    assert device_sample_enabled() is False
    os.environ["MTPLX_DSV41_DEVICE_SAMPLE"] = "1"
    assert device_sample_enabled() is True
    os.environ["MTPLX_DSV41_DEVICE_SAMPLE"] = "off"
    assert device_sample_enabled() is False
