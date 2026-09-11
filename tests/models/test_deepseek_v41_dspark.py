"""DSpark MTP gates for the DeepSeek-V4.1 backend: spec == AR + the streaming bar.

The DSpark 3-stage draft head (``mtplx.models.deepseek_v41_dspark``, worker W23)
is a *pure latency* optimisation: greedy speculative decode must emit the exact
same token sequence as autoregressive decode.  The draft head's numerics only
set the acceptance rate -- the runtime's greedy target verify is authoritative --
so the bar is argmax equality, NOT bit-exactness (memory: deepseek-v4-mtplx-port
/ spec-decode-cycle-anatomy; PORT_PLAN P3.0 done-when).

Gates, in order:

  1. **spec == AR through the real engine.**  ``generate_mtpk`` at depth 1/2/3
     emits the identical greedy sequence as ``generate_ar`` over 64 tokens.  This
     runs the actual ``mtplx.generation`` machine (prefill, per-depth DSpark draft
     chain, batched verify, accept, reject, rollback), not a hand-rolled loop, so
     it also gates the registry/runtime/injector wiring.

  2. **accept + reject both exercised** (small vocab makes an untrained draft head
     agree often enough for both), and the engine's acceptance counters populate.

  3. **the forward + MTP surface contract** the runtime drives every MTP backend
     through (return_hidden -> main_hidden, emit_logits/logits_keep, the reject of
     input_embeddings, degrade-to-AR).

  4. **the streaming-bank verify bar** (OPTIMIZATION_LEDGER Gate 0 / R2): a K+1-row
     verify must reach each backbone MoE layer in ONE switch call carrying all
     rows (the precondition for record dedup), and the routed-expert union u per
     layer is measured -- wider verify only widens u, which is why K>3 / tree
     verify are dead on the streaming bank.

  5. **the resident MTP name mapping** (``Model.sanitize`` opt-in ``mtp=True``):
     every ``mtp.{i}.*`` checkpoint name maps onto a real DSpark head parameter.

Self-contained: shrunk seeded config, no downloads, no checkpoint, no torch.  CPU
device so MLX fp32 is bit-exact.
"""
import importlib.util
import os
import sys
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
    _map_mtp_residents,
    inject_deepseek_v41_mtp_support,
    is_deepseek_v41_mtp_config,
)

# A shrunk all-sliding-window backbone (compress_ratios default to 0) with a
# 3-stage DSpark head whose target layers are the last three backbone layers.
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
        # DSpark head
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


def _seeded_model(seed=0, vocab=64, mtp=True, **over):
    mx.random.seed(seed)
    args = _args(vocab=vocab, **over)
    model = Model(args, quantize=False, mtp=mtp)
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
    from mtplx.mtp_patch import MTPContract, validate_mtp_support
    from mtplx.runtime import MTPLXRuntime

    config = {"model_type": "deepseek_v41", "n_mtp_layers": 3}
    assert is_deepseek_v41_mtp_config(config)
    _args_, model = _seeded_model(seed=seed, vocab=vocab)
    assert inject_deepseek_v41_mtp_support(model, Path("."), config, MTPContract())
    assert validate_mtp_support(model)
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


def _ar(rt, prompt, max_tokens):
    from mtplx.generation import generate_ar
    from mtplx.sampling import SamplerConfig

    return generate_ar(
        rt, prompt, max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.0), stop_token_ids=set(),
    )


def _spec(rt, prompt, max_tokens, depth, verify_strategy="batched"):
    from mtplx.generation import generate_mtpk
    from mtplx.sampling import SamplerConfig

    return generate_mtpk(
        rt, prompt, max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=depth, mtp_history_policy="committed",
        stop_token_ids=set(), verify_strategy=verify_strategy,
    )


# --------------------------------------------------------------------------- #
# 1. spec == AR through the real engine  (the P3.0 done-when)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_dspark_greedy_verify_reproduces_ar_over_64_tokens(depth):
    prompt = _prompt(17)
    baseline = _ar(_runtime(), prompt, 64)
    assert len(baseline.tokens) == 64
    assert len(set(baseline.tokens)) > 1, "premise: AR output must not be degenerate"

    out = _spec(_runtime(), prompt, 64, depth)
    assert out.tokens == baseline.tokens, (
        f"depth {depth}: DSpark speculative decode diverged from AR\n"
        f"  AR  : {baseline.tokens}\n"
        f"  spec: {out.tokens}"
    )


# seed 3 gives a strong mixed signal (11 accepts / 34 rejects over 48 tokens at
# vocab 8); the untrained separate-net DSpark head agrees with the trunk only
# occasionally, so a degenerate seed (0/5) can accept nothing -- pin a mixed one.
_MIXED_SEED = 3


def test_dspark_exercises_both_accept_and_reject():
    prompt = _prompt(17, vocab=8)
    baseline = _ar(_runtime(seed=_MIXED_SEED, vocab=8), prompt, 48)
    out = _spec(_runtime(seed=_MIXED_SEED, vocab=8), prompt, 48, 3)
    stats = out.stats.to_dict()
    assert out.tokens == baseline.tokens
    assert stats["accepted_drafts"] > 0, "premise: no draft was ever accepted"
    assert stats["rejected_drafts"] > 0, "premise: no draft was ever rejected"


def test_dspark_acceptance_counters_populate_per_depth():
    depth = 3
    out = _spec(_runtime(seed=_MIXED_SEED, vocab=8), _prompt(17, vocab=8), 48, depth)
    stats = out.stats.to_dict()
    assert stats["runtime_mtp_enabled"] is True
    assert len(stats["drafted_by_depth"]) == depth
    assert len(stats["accepted_by_depth"]) == depth
    assert sum(stats["drafted_by_depth"]) == stats["drafted_tokens"] > 0
    assert sum(stats["accepted_by_depth"]) > 0
    assert stats["mtp_forward_calls"] > 0 and stats["make_mtp_cache_calls"] > 0


# --------------------------------------------------------------------------- #
# 2. the forward + MTP surface contract
# --------------------------------------------------------------------------- #
def test_forward_surface_matches_the_runtime_contract():
    args, model = _seeded_model()
    ids = mx.array(_prompt(9)).reshape(1, 9)
    plain = np.array(model(ids))
    logits, main_hidden = model(ids, return_hidden=True)
    # main_hidden is the concat of the 3 target-layer hiddens
    assert main_hidden.shape == (1, 9, args.hidden_size * len(args.dspark_target_layer_ids))
    assert np.array_equal(np.array(logits), plain)
    kept = model(ids, logits_keep=1)
    assert kept.shape == (1, 1, args.vocab_size)
    assert int(np.array(kept)[:, 0].argmax()) == int(plain[:, -1].argmax())
    # the draft consumes main_hidden; mtp_forward returns (logits, hidden)
    mcache = model.make_mtp_cache()
    dl, dh = model.mtp_forward(main_hidden, ids[:, -1:], mtp_cache=mcache,
                               return_hidden=True, mtp_depth=1)
    assert dl.shape == (1, 1, args.vocab_size)
    assert dh.shape[0] == 1 and dh.shape[-1] == main_hidden.shape[-1]


def test_surface_rejects_input_embeddings_and_degrades_without_head():
    _, model = _seeded_model()
    with pytest.raises(ValueError, match="input_embeddings"):
        model(mx.array([[1, 2, 3]]), input_embeddings=mx.zeros((1, 3, 32)))
    # a model built without the head reports no MTP (degrade-to-AR)
    from mtplx.mtp_patch import MTPContract

    _, ar_only = _seeded_model(mtp=False)
    assert not ar_only.has_mtp
    assert inject_deepseek_v41_mtp_support(
        ar_only, Path("."), {"model_type": "deepseek_v41", "n_mtp_layers": 3}, MTPContract()
    ) is False


# --------------------------------------------------------------------------- #
# 3. the streaming-bank verify bar (OPTIMIZATION_LEDGER Gate 0 / R2)
# --------------------------------------------------------------------------- #
def test_kplus1_verify_reaches_each_moe_layer_in_one_call_and_bounds_the_union():
    """On the streaming bank a K+1-row verify gathers the UNION of the rows'
    routed experts, so record dedup (one gather per unique expert per layer per
    cycle) is the enabler.  Its precondition is that all K+1 rows reach each MoE
    layer in ONE switch call; this gate confirms that and measures the union u
    per layer for a K+1 (=4) row verify -- the quantity Gate 0 bounds (median
    u <= 10 at K=3 on the real 384-expert model; here u <= rows*top_k = 8)."""
    args, model = _seeded_model()
    cache = model.make_cache()

    calls_per_layer: list[int] = []
    unions: list[int] = []
    for layer in model.layers:
        gate = layer.mlp.gate
        state = {"calls": 0}

        def wrapped(x, image_mask=None, _gate=gate, _state=state):
            _state["calls"] += 1
            weights, indices = _gate(x, image_mask)
            _state["rows"] = int(indices.shape[0])
            _state["union"] = int(np.unique(np.array(indices)).size)
            return weights, indices

        layer.mlp.gate = wrapped
        layer.mlp._probe = state

    K = 3
    verify_ids = mx.array(_prompt(K + 1)).reshape(1, K + 1)  # a K+1-row verify forward
    _ = model(verify_ids, cache=cache)
    mx.eval(_)

    for layer in model.layers:
        state = layer.mlp._probe
        calls_per_layer.append(state["calls"])
        assert state["rows"] == K + 1, "all K+1 verify rows must reach the MoE in one call"
        unions.append(state["union"])

    assert all(c == 1 for c in calls_per_layer), (
        "each MoE layer must be reached exactly once per verify forward -- the "
        "precondition for record dedup across the union"
    )
    top_k = args.num_experts_per_tok
    assert all(1 <= u <= (K + 1) * top_k for u in unions)
    # a dedup'd gather issues |union| records; a naive per-row gather issues
    # (K+1)*top_k -- the dedup saving this asserts is available.
    assert min(unions) <= (K + 1) * top_k


# --------------------------------------------------------------------------- #
# 4. resident MTP name mapping (Model.sanitize opt-in mtp=True)
# --------------------------------------------------------------------------- #
def test_mtp_resident_name_mapping_covers_the_head_parameter_tree():
    """Every ``mtp.{i}.*`` checkpoint name the opt-in load path keeps must map onto
    a real DSpark head parameter path.  Build the inverse from the model's own
    tree and confirm the mapping is exact for the dense tensors + stacked experts."""
    args, model = _seeded_model()
    head_paths = {
        name for name, _ in tree_flatten(model.parameters()) if name.startswith("mtp.")
    }
    assert head_paths, "the head must expose mtp.* parameters"

    stages = args.n_mtp_layers
    last = stages - 1
    # synthesize representative checkpoint names for each stage (mirrors W18)
    src: dict[str, mx.array] = {}
    for i in range(stages):
        for proj in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            src[f"mtp.{i}.attn.{proj}.weight"] = mx.zeros((1,))
        src[f"mtp.{i}.attn.attn_sink"] = mx.zeros((1,))
        src[f"mtp.{i}.attn.q_norm.weight"] = mx.zeros((1,))
        src[f"mtp.{i}.attn.kv_norm.weight"] = mx.zeros((1,))
        for hc in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                   "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
            src[f"mtp.{i}.{hc}"] = mx.zeros((1,))
        src[f"mtp.{i}.attn_norm.weight"] = mx.zeros((1,))
        src[f"mtp.{i}.ffn_norm.weight"] = mx.zeros((1,))
        src[f"mtp.{i}.ffn.gate.weight"] = mx.zeros((1,))
        src[f"mtp.{i}.ffn.gate.bias"] = mx.zeros((1,))
        for w in ("w1", "w2", "w3"):
            src[f"mtp.{i}.ffn.shared_experts.{w}.weight"] = mx.zeros((1,))
            for e in range(args.dspark_n_routed_experts):
                src[f"mtp.{i}.ffn.experts.{e}.{w}.weight"] = mx.zeros((2, 3))
        if i == 0:
            src[f"mtp.{i}.main_proj.weight"] = mx.zeros((1,))
            src[f"mtp.{i}.main_norm.weight"] = mx.zeros((1,))
        if i == last:
            src[f"mtp.{i}.norm.weight"] = mx.zeros((1,))
            src[f"mtp.{i}.markov_head.embed.weight"] = mx.zeros((1,))
            src[f"mtp.{i}.markov_head.head.weight"] = mx.zeros((1,))
            src[f"mtp.{i}.confidence_head.proj.weight"] = mx.zeros((1,))

    mapped = _map_mtp_residents(src)
    unknown = sorted(k for k in mapped if k not in head_paths)
    assert not unknown, f"mapped names with no matching head parameter: {unknown}"
    # every non-shared head parameter (embed/head are the trunk's, not under mtp.)
    # must be covered by the mapping
    missing = sorted(head_paths - set(mapped))
    assert not missing, f"head parameters left unmapped: {missing}"
    # experts stacked over the expert axis
    ge = mapped[f"mtp.layers.0.mlp.switch_mlp.gate_proj.weight"]
    assert ge.shape[0] == args.dspark_n_routed_experts


# --------------------------------------------------------------------------- #
# 5. the loader opt-in mtp=True path
# --------------------------------------------------------------------------- #
import types  # noqa: E402


def _fake_manifest(names):
    return types.SimpleNamespace(
        resident_tensors=[
            types.SimpleNamespace(tensor=n, length=8, shard="s0", shape=(1,), dtype="F32")
            for n in names
        ]
    )


def test_loader_partition_keeps_mtp_only_when_opted_in():
    from mtplx.models import deepseek_v41_loader as L

    man = _fake_manifest([
        "model.embed_tokens.weight", "layers.0.attn.wq_a.weight",
        "vision.foo", "mtp.0.attn.wq_a.weight", "mtp.0.ffn.experts.0.w1.weight",
    ])
    ar = L.partition_text_residents(man)
    mtp = L.partition_text_residents(man, with_mtp=True)
    assert ar.skipped_mtp_count == 2 and not any(
        t.tensor.startswith("mtp.") for t in ar.kept
    )
    assert mtp.skipped_mtp_count == 0 and sum(
        t.tensor.startswith("mtp.") for t in mtp.kept
    ) == 2
    # vision is always skipped
    assert not any(t.tensor.startswith("vision.") for t in mtp.kept)


def test_resolve_with_mtp_is_opt_in_and_fails_loud_when_unsatisfiable(monkeypatch):
    from mtplx.models import deepseek_v41_loader as L

    man = _fake_manifest(["mtp.0.attn.wq_a.weight"])
    cfg = {"model_type": "deepseek_v41", "num_nextn_predict_layers": 3}
    monkeypatch.delenv("MTPLX_DSV41_MTP", raising=False)
    assert L.resolve_with_mtp(cfg, man, None) is False  # default off
    assert L.resolve_with_mtp(cfg, man, True) is True
    assert L.resolve_with_mtp(cfg, man, False) is False
    monkeypatch.setenv("MTPLX_DSV41_MTP", "1")
    assert L.resolve_with_mtp(cfg, man, None) is True  # env opt-in
    monkeypatch.setenv("MTPLX_DSV41_MTP", "0")
    assert L.resolve_with_mtp(cfg, man, None) is False
    # opting in against an artifact that cannot honour it is a loud error
    with pytest.raises(L.ResidentLoadError, match="no MTP stages"):
        L.resolve_with_mtp({"model_type": "deepseek_v41"}, man, True)
    with pytest.raises(L.ResidentLoadError, match="no mtp"):
        L.resolve_with_mtp(cfg, _fake_manifest(["layers.0.attn.wq_a.weight"]), True)
