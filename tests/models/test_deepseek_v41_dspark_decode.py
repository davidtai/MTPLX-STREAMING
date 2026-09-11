"""DSpark-DIRECT decode lane gates (worker W57).

The DSpark-DIRECT loop (:mod:`mtplx.models.deepseek_v41_dspark_decode`) drives
W23's DSpark drafter through a lean speculative loop that *bypasses* the generic
native-MTP machinery (``generate_mtpk`` etc.).  Its correctness bar is identical
to the generic lane's: greedy speculative decode is byte-for-byte AR, and a
``K=0`` sampled configuration is exactly AR sampling under the same seed
(distribution-matching speculative sampling), regardless of draft quality --
because the target verify is authoritative (memory: deepseek-v4-mtplx-port /
spec-decode-cycle-anatomy).

Gates:
  1. greedy dspark == AR over 256 tokens at depth 1/2/3 (the extended P3.0 bar),
     and K=0 greedy == AR.
  2. sampled K=0 == ``generate_ar`` byte-for-byte (same seed); sampled K>0 has
     sane acceptance statistics and returns a valid completion.
  3. accept + reject both exercised, counters populate per depth.
  4. cache rollback is exact: the target cache offset after a run equals
     prompt + generated (no speculative drift).
  5. the served lane (``generate_dspark``) end to end on a stub runtime: greedy
     == ``generate_ar``, stop tokens / max_tokens / usage counts honoured, accept
     stats in ``stats.to_dict()``, streamed tokens == committed tokens.

Self-contained: shrunk seeded config, CPU device (MLX fp32 bit-exact), no
downloads, no checkpoint, no experts.bin.
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


from mtplx.models.deepseek_v41 import (  # noqa: E402
    Model,
    ModelArgs,
    inject_deepseek_v41_mtp_support,
)
from mtplx.models.deepseek_v41_dspark_decode import (  # noqa: E402
    DSparkDecodeStats,
    dspark_generate,
    generate_dspark,
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


GREEDY = SamplerConfig(temperature=0.0)


# --------------------------------------------------------------------------- #
# 1. greedy == AR (the extended P3.0 bar: 256 tokens)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_dspark_direct_greedy_reproduces_ar_over_256_tokens(depth):
    _args_, model = _seeded_model(seed=0)
    prompt = _prompt(17)
    ref = _ar_reference(model, prompt, 256)
    assert len(ref) == 256
    assert len(set(ref)) > 1, "premise: AR output must not be degenerate"

    stats = DSparkDecodeStats()
    out = dspark_generate(
        model, prompt, max_tokens=256, sampler=GREEDY, seed=0,
        speculative_depth=depth, stats=stats,
    )
    assert out == ref, (
        f"depth {depth}: DSpark-DIRECT greedy diverged from AR at "
        f"{next((i for i, (a, b) in enumerate(zip(out, ref)) if a != b), None)}"
    )
    assert len(out) == 256


# --------------------------------------------------------------------------- #
# 2. sampled: K=0 == generate_ar exactly; K>0 distribution-matching + sane
# --------------------------------------------------------------------------- #
def test_dspark_direct_sampled_k0_equals_generate_ar():
    """K=0 is pure AR sampling: under one seed it must reproduce generate_ar
    byte-for-byte (same numpy RNG draws through _sample_from_logits)."""
    from mtplx.generation import generate_ar

    rt = _runtime(seed=0)
    model = rt.model
    prompt = _prompt(17)
    sampler = SamplerConfig(temperature=0.8, top_p=0.95, top_k=40)
    ar = generate_ar(rt, prompt, max_tokens=48, sampler=sampler, seed=123,
                     stop_token_ids=set())
    out = dspark_generate(model, prompt, max_tokens=48, sampler=sampler, seed=123,
                          stop_ids=set(), speculative_depth=0)
    assert out == list(ar.tokens)


def test_dspark_direct_sampled_kpos_is_sane():
    """K>0 sampled returns a valid completion with populated, consistent accept
    statistics (distribution-matching is guaranteed by the point-mass
    speculative-sampling construction; here we assert the run is well-formed)."""
    _args_, model = _seeded_model(seed=3, vocab=8)
    prompt = _prompt(17, vocab=8)
    sampler = SamplerConfig(temperature=1.0, top_p=1.0, top_k=0)
    stats = DSparkDecodeStats()
    out = dspark_generate(model, prompt, max_tokens=64, sampler=sampler, seed=5,
                          speculative_depth=3, stats=stats)
    assert len(out) == 64
    assert all(0 <= t < 8 for t in out)
    assert stats.drafted_tokens == sum(stats.drafted_by_depth) > 0
    assert stats.accepted_drafts == sum(stats.accepted_by_depth)
    assert stats.rejected_drafts == stats.drafted_tokens - stats.accepted_drafts


# --------------------------------------------------------------------------- #
# 3. accept + reject both exercised; counters populate per depth
# --------------------------------------------------------------------------- #
def test_dspark_direct_exercises_accept_and_reject():
    _args_, model = _seeded_model(seed=3, vocab=8)
    prompt = _prompt(17, vocab=8)
    ref = _ar_reference(model, prompt, 64)
    stats = DSparkDecodeStats()
    out = dspark_generate(model, prompt, max_tokens=64, sampler=GREEDY, seed=0,
                          speculative_depth=3, stats=stats)
    assert out == ref
    assert stats.accepted_drafts > 0, "premise: no draft was ever accepted"
    assert stats.rejected_drafts > 0, "premise: no draft was ever rejected"
    assert len(stats.drafted_by_depth) == 3
    assert len(stats.accepted_by_depth) == 3
    assert stats.cycles == stats.verify_calls > 0
    # every emitted token is either a bonus (all-accept) or a correction
    assert stats.bonus_tokens + stats.correction_tokens == stats.cycles


# --------------------------------------------------------------------------- #
# 4. cache rollback is exact (no speculative drift)
# --------------------------------------------------------------------------- #
def test_dspark_direct_cache_offset_matches_committed_length():
    """After a run the target cache holds exactly prompt + all-but-last committed
    tokens (the last emitted token is the un-forwarded next primary), proving the
    reject rollback trims the speculative tail exactly."""
    _args_, model = _seeded_model(seed=0)
    prompt = _prompt(17)

    # Reproduce the loop's cache so we can read its final offset.
    from mtplx.models.deepseek_v41_dspark_decode import _decode_cycles, _target_forward
    from mtplx.generation import _sample_from_logits

    cache = model.make_cache()
    mtp_caches = model.make_mtp_cache()
    fwd = _target_forward(model)
    logits, main_hidden = fwd(mx.array([prompt]), cache)
    mx.eval(logits, main_hidden)
    model.mtp.seed_main(main_hidden, mtp_caches)
    primary, _ = _sample_from_logits(logits[0, -1], GREEDY, np.random.default_rng(0))
    primary = int(primary)
    stats = DSparkDecodeStats()
    rest, _finish = _decode_cycles(
        model=model, forward=fwd, cache=cache, mtp_caches=mtp_caches,
        primary=primary, main_h=main_hidden[:, -1:, :], max_tokens=40 - 1,
        sampler=GREEDY, rng=np.random.default_rng(0), stop_ids=set(),
        k_request=3, confidence_threshold=None, stats=stats,
        token_callback=None, abort_check=None,
    )
    tokens = [primary] + rest
    # cache holds the prompt plus every committed token except the final
    # un-forwarded primary (which is next-cycle input, never forwarded).
    assert cache[0].offset == len(prompt) + len(tokens) - 1
    # DSpark stage windows advanced by the same committed count (ring-capped).
    for c in mtp_caches:
        assert c.offset == len(prompt) + len(tokens) - 1


# --------------------------------------------------------------------------- #
# 5. served lane end to end (stub runtime)
# --------------------------------------------------------------------------- #
def test_served_dspark_greedy_equals_generate_ar_and_streams():
    from mtplx.generation import generate_ar

    rt = _runtime(seed=0)
    prompt = _prompt(17)
    ar = generate_ar(rt, prompt, max_tokens=64, sampler=GREEDY, stop_token_ids=set())

    streamed: list[int] = []
    out = generate_dspark(
        rt, prompt, max_tokens=64, sampler=GREEDY, seed=0, stop_token_ids=set(),
        speculative_depth=3, token_callback=lambda d: streamed.extend(d),
    )
    assert list(out.tokens) == list(ar.tokens)
    assert streamed == list(out.tokens), "streamed delta must equal committed tokens"
    assert out.finish_reason == "length"
    assert len(out.tokens) == 64

    d = out.stats.to_dict()
    for key in ("accepted_drafts", "rejected_drafts", "drafted_tokens",
                "verify_calls", "accepted_by_depth", "drafted_by_depth",
                "correction_tokens", "bonus_tokens"):
        assert key in d, f"missing accept-stat key in served stats: {key}"
    assert d["generated_tokens"] == 64
    assert d["verify_calls"] > 0
    assert out.stats.events[0]["lane"] == "dspark_direct"


def test_served_dspark_honours_stop_and_usage():
    rt = _runtime(seed=0)
    prompt = _prompt(17)
    # Run once to find a token the greedy stream actually emits, then stop on it.
    out0 = generate_dspark(rt, prompt, max_tokens=32, sampler=GREEDY, seed=0,
                           stop_token_ids=set(), speculative_depth=3)
    assert len(out0.tokens) == 32
    stop_tok = int(out0.tokens[5])
    out = generate_dspark(rt, prompt, max_tokens=32, sampler=GREEDY, seed=0,
                          stop_token_ids={stop_tok}, speculative_depth=3)
    assert stop_tok in out.tokens
    assert out.tokens[-1] == stop_tok, "generation must stop AT the stop token"
    assert out.finish_reason == "stop"
    # tokens before the stop match the un-stopped run (byte-identical prefix)
    assert list(out.tokens) == list(out0.tokens[: len(out.tokens)])


def test_served_dspark_requires_mtp_runtime():
    class _FakeRT:
        mtp_enabled = False
        model = None
        tokenizer = _FixedTokenizer()

    with pytest.raises(RuntimeError, match="MTP-enabled"):
        generate_dspark(_FakeRT(), [1, 2, 3], max_tokens=4, sampler=GREEDY, seed=0)


# --------------------------------------------------------------------------- #
# 6. confidence early-stop is a pure latency lever (output unchanged)
# --------------------------------------------------------------------------- #
def test_dspark_direct_confidence_early_stop_is_lossless():
    _args_, model = _seeded_model(seed=3, vocab=8)
    prompt = _prompt(17, vocab=8)
    ref = _ar_reference(model, prompt, 48)
    # An aggressive threshold trims the verify width but must not change output.
    stats = DSparkDecodeStats()
    out = dspark_generate(model, prompt, max_tokens=48, sampler=GREEDY, seed=0,
                          speculative_depth=3, confidence_threshold=0.9, stats=stats)
    assert out == ref


# --------------------------------------------------------------------------- #
# 7. server dispatch wiring (mode normalization + the DSpark-DIRECT gate)
# --------------------------------------------------------------------------- #
def test_server_generation_mode_accepts_dspark():
    from fastapi import HTTPException

    from mtplx.server.openai import _normalize_generation_mode

    assert _normalize_generation_mode("dspark") == "dspark"
    assert _normalize_generation_mode("mtp") == "mtp"
    assert _normalize_generation_mode("ar") == "ar"
    with pytest.raises(HTTPException):
        _normalize_generation_mode("bogus")


def test_server_dspark_direct_selected_gate(tmp_path, monkeypatch):
    import argparse
    import json as _json
    import types

    from mtplx.server.openai import _dspark_direct_selected

    (tmp_path / "config.json").write_text(_json.dumps({"model_type": "deepseek_v41"}))
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text(_json.dumps({"model_type": "qwen4_exp"}))

    def _state(model_dir, mtp_enabled):
        return types.SimpleNamespace(
            args=argparse.Namespace(model=str(model_dir)),
            runtime=types.SimpleNamespace(mtp_enabled=mtp_enabled),
        )

    dsv41 = _state(tmp_path, True)
    # explicit dspark mode selects the lane on a dsv41 MTP runtime
    assert _dspark_direct_selected(dsv41, "dspark") is True
    # mtp mode only selects it with the env flag
    monkeypatch.delenv("MTPLX_DSV41_DSPARK_DIRECT", raising=False)
    assert _dspark_direct_selected(dsv41, "mtp") is False
    monkeypatch.setenv("MTPLX_DSV41_DSPARK_DIRECT", "1")
    assert _dspark_direct_selected(dsv41, "mtp") is True
    assert _dspark_direct_selected(dsv41, "ar") is False
    # never for a non-dsv41 model, and never without an MTP runtime
    assert _dspark_direct_selected(_state(other, True), "dspark") is False
    assert _dspark_direct_selected(_state(tmp_path, False), "dspark") is False
