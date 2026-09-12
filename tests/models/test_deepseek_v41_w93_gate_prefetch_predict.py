"""W93 -- gate-oracle prefetch PREDICTOR + wiring, on the tiny real DSV4.1 model.

Model-level proof of the predictor half of the one-layer-ahead gate-oracle
prefetch (docs/deepseek-v41/W93_GATE_PREFETCH.md), on the tiny CPU DSV4.1 ``Model``
from tests/models/test_deepseek_v41_stage_timing.py (real backbone, resident
SwitchGLU, no artifact).  These lock:

  1. **link install eligibility** -- ``install_gate_prefetch_links`` wires a layer
     to the NEXT routed layer's gate only for consecutive routed layers whose
     target ``L >= MIN_LAYER`` and ``L != last`` (W89's early floor + the last-
     layer skip);
  2. **offline replay (the trace-format decision)** -- during a real decode the
     prediction the DecoderLayer stashes for layer ``L`` equals the top-k of
     layer ``L``'s OWN gate applied to the residual ENTERING ``L-1`` (W89's
     ``gate_L(layer_in_{L-1})``), for the exact ``k`` and gate the runtime uses;
  3. **exact by construction (AR)** -- 64 greedy decode steps are ``mx.array_
     equal`` with the flag on vs off: computing + stashing the prediction feeds
     no logit;
  4. **verify inertness (T>1)** -- the predictor is skipped for any multi-row
     forward (prefill / the DSpark M=4 verify), so a DSpark sequence is
     byte-identical on/off by construction.

Pins MLX to CPU; tiny random config; no artifact.  Run under ``nice -n 19`` and
without ``pytest -n auto``.
"""

from __future__ import annotations

import types

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

import pytest  # noqa: E402

import mtplx.models.deepseek_v41 as dv  # noqa: E402
from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402
from mtplx.models.deepseek_v41_moe import gate_predict_topk  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

FLAG = "MTPLX_DSV41_GATE_PREFETCH"
MIN_LAYER_FLAG = "MTPLX_DSV41_GATE_PREFETCH_MIN_LAYER"
# The tiny model routes 8 experts; a prefetch width < 8 keeps the top-k a proper
# SUBSET so the replay/next-gate assertions are discriminating (k>=8 would trivially
# select all experts). The real profile uses k=10/12 over 384 experts.
KPRED = 3


# --- tiny model fixtures (mirror test_deepseek_v41_stage_timing.py) ----------
def _csa_args(**over) -> ModelArgs:
    base = dict(
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
    base.update(over)
    return ModelArgs(**base)


def _randomize(model, seed=0, scale=0.1):
    mx.random.seed(seed)
    new = []
    for name, arr in tree_flatten(model.parameters()):
        if arr.ndim == 1 and ("norm_weight" in name or name.endswith("norm.weight")):
            v = 1.0 + 0.2 * mx.random.normal(arr.shape)
        elif "attn_sink" in name:
            v = 0.5 * mx.random.normal(arr.shape)
        else:
            v = scale * mx.random.normal(arr.shape)
        new.append((name, v.astype(mx.float32)))
    model.update(tree_unflatten(new))
    mx.eval(model.parameters())


def _new_model(seed=1, **over):
    args = _csa_args(**over)
    model = Model(args)
    _randomize(model, seed=seed)
    return model, args


def _fake_runtime(k, layers):
    """A minimal non-Module stand-in exposing only what the predictor reads."""
    routed = tuple(range(len(layers)))
    return types.SimpleNamespace(
        spec=types.SimpleNamespace(routed_layer_indices=routed),
        config=types.SimpleNamespace(prefetch_slots=k),
        prefetch_experts=lambda *a, **k: 0,
        note_gate_prefetch_predicted=lambda *a, **k: None,
    )


def _arm(model, k):
    """Attach a fake runtime to every layer's switch + install the next-gate
    links, so the predictor engages on the resident path (which never consumes
    the stash -- exactly the point of the byte-identity test)."""
    layers = model.model.layers
    rt = _fake_runtime(k, layers)
    for layer in layers:
        layer.mlp.switch_mlp.runtime = rt
    dv.install_gate_prefetch_links(model, rt)
    return rt


def _prefill(model, args, s, seed):
    ids = mx.array(np.random.RandomState(seed).randint(0, args.vocab_size, size=(1, s)))
    cache = model.make_cache()
    logits = model(ids, cache=cache, prefill_chunk=0)
    mx.eval(logits)
    return cache, int(mx.argmax(logits[0, -1]).item())


def _decode(model, cache, first_token, steps):
    """Greedy AR decode; return the per-step logits (evaluated)."""
    out = []
    token = first_token
    for _ in range(steps):
        logits = model(mx.array([[token]]), cache=cache)
        mx.eval(logits)
        out.append(logits[0, -1])
        token = int(mx.argmax(logits[0, -1]).item())
    return out


# ---------------------------------------------------------------------------
# 1. link install eligibility
# ---------------------------------------------------------------------------
def test_install_links_only_eligible_layers(monkeypatch) -> None:
    monkeypatch.delenv(MIN_LAYER_FLAG, raising=False)  # default 4
    model, _ = _new_model(num_hidden_layers=8)
    rt = _fake_runtime(10, model.model.layers)
    installed = dv.install_gate_prefetch_links(model, rt)
    # routed = 0..7, min_layer=4, last=7 -> targets 4,5,6 -> sources 3,4,5.
    linked = {
        i for i, layer in enumerate(model.model.layers)
        if getattr(layer.mlp, "_mtplx_gate_prefetch_next", None) is not None
    }
    assert linked == {3, 4, 5}, linked
    assert installed == 3
    for src in (3, 4, 5):
        link = model.model.layers[src].mlp._mtplx_gate_prefetch_next
        assert link.next_layer == src + 1
        assert link.next_gate is model.model.layers[src + 1].mlp.gate


def test_min_layer_env_moves_floor(monkeypatch) -> None:
    monkeypatch.setenv(MIN_LAYER_FLAG, "2")
    model, _ = _new_model(num_hidden_layers=8)
    rt = _fake_runtime(10, model.model.layers)
    dv.install_gate_prefetch_links(model, rt)
    linked = {
        i for i, layer in enumerate(model.model.layers)
        if getattr(layer.mlp, "_mtplx_gate_prefetch_next", None) is not None
    }
    # min_layer=2 -> targets 2..6 -> sources 1..5.
    assert linked == {1, 2, 3, 4, 5}, linked


# ---------------------------------------------------------------------------
# 2. offline replay: the stashed prediction == top-k gate(layer_in_{L-1})
# ---------------------------------------------------------------------------
def test_stashed_prediction_matches_gate_oracle(monkeypatch) -> None:
    monkeypatch.setenv(FLAG, str(KPRED))
    monkeypatch.delenv(MIN_LAYER_FLAG, raising=False)
    model, args = _new_model(num_hidden_layers=8)
    _arm(model, KPRED)

    # capture the residual ENTERING each layer during a real decode step.
    captured: dict[int, mx.array] = {}
    orig = dv.DecoderLayer._maybe_stash_gate_prefetch

    def capture(self, h):
        captured[self.layer_id] = h
        return orig(self, h)

    monkeypatch.setattr(dv.DecoderLayer, "_maybe_stash_gate_prefetch", capture)

    cache, token = _prefill(model, args, s=6, seed=3)
    logits = model(mx.array([[token]]), cache=cache)
    mx.eval(logits)

    k = KPRED
    checked = 0
    for src in (3, 4, 5):  # the eligible source layers (targets 4,5,6)
        switch = model.model.layers[src].mlp.switch_mlp
        pending = getattr(switch, "_mtplx_gate_prefetch_pending", None)
        assert pending is not None, f"layer {src} did not stash a prediction"
        target, predicted = pending
        assert target == src + 1
        # independent recompute of W89's b' predictor: layer L's OWN gate applied
        # to mean-over-hc of the residual entering L-1.
        next_gate = model.model.layers[target].mlp.gate
        collapsed = mx.mean(captured[src].astype(mx.float32), axis=2)
        expected = gate_predict_topk(next_gate, collapsed, k)
        mx.eval(predicted, expected)
        assert set(predicted.reshape(-1).tolist()) == set(
            expected.reshape(-1).tolist()
        ), f"layer {src}: issued set != top-{k} gate(layer_in_{{{src}}})"
        assert predicted.shape[-1] == k
        checked += 1
    assert checked == 3


def test_predictor_uses_next_gate_not_own(monkeypatch) -> None:
    # Guard against wiring the WRONG gate: layer L's route must be predicted by
    # layer L's gate, applied to L-1's input -- not by L-1's own gate.
    monkeypatch.setenv(FLAG, str(KPRED))
    monkeypatch.delenv(MIN_LAYER_FLAG, raising=False)
    model, args = _new_model(num_hidden_layers=8)
    _arm(model, KPRED)
    captured: dict[int, mx.array] = {}
    orig = dv.DecoderLayer._maybe_stash_gate_prefetch

    def capture(self, h):
        captured[self.layer_id] = h
        return orig(self, h)

    monkeypatch.setattr(dv.DecoderLayer, "_maybe_stash_gate_prefetch", capture)
    cache, token = _prefill(model, args, s=6, seed=5)
    mx.eval(model(mx.array([[token]]), cache=cache))

    src = 4
    _target, predicted = model.model.layers[src].mlp.switch_mlp._mtplx_gate_prefetch_pending
    collapsed = mx.mean(captured[src].astype(mx.float32), axis=2)
    with_next = gate_predict_topk(model.model.layers[src + 1].mlp.gate, collapsed, KPRED)
    with_own = gate_predict_topk(model.model.layers[src].mlp.gate, collapsed, KPRED)
    mx.eval(predicted, with_next, with_own)
    assert set(predicted.reshape(-1).tolist()) == set(with_next.reshape(-1).tolist())
    # the two gates are different random matrices, so the sets differ -- proving
    # the mechanism used the NEXT gate (the assertion above), not the own gate.
    assert set(with_next.reshape(-1).tolist()) != set(with_own.reshape(-1).tolist())


# ---------------------------------------------------------------------------
# 3. exact by construction: 64 AR decode steps byte-identical on/off
# ---------------------------------------------------------------------------
def test_ar_decode_byte_identical_flag_on_off(monkeypatch) -> None:
    monkeypatch.delenv(MIN_LAYER_FLAG, raising=False)

    # flag OFF (no runtime, no links) -> pure demand path.
    monkeypatch.delenv(FLAG, raising=False)
    model_off, args = _new_model(seed=7, num_hidden_layers=8)
    cache_off, tok_off = _prefill(model_off, args, s=8, seed=11)
    logits_off = _decode(model_off, cache_off, tok_off, steps=64)

    # flag ON + armed switches -> the predictor computes + stashes every step.
    monkeypatch.setenv(FLAG, str(KPRED))
    model_on, args2 = _new_model(seed=7, num_hidden_layers=8)
    _arm(model_on, KPRED)
    cache_on, tok_on = _prefill(model_on, args2, s=8, seed=11)
    logits_on = _decode(model_on, cache_on, tok_on, steps=64)

    assert tok_off == tok_on
    for step, (a, b) in enumerate(zip(logits_off, logits_on)):
        assert mx.array_equal(a, b), f"step {step}: logits differ with flag on"
    # sanity: the predictor actually engaged on the armed model.
    pend = model_on.model.layers[4].mlp.switch_mlp._mtplx_gate_prefetch_pending
    assert pend is not None and pend[0] == 5


# ---------------------------------------------------------------------------
# 4. verify inertness: the predictor is skipped for any multi-row forward
# ---------------------------------------------------------------------------
def test_predictor_inert_on_multirow_forward(monkeypatch) -> None:
    monkeypatch.setenv(FLAG, str(KPRED))
    monkeypatch.delenv(MIN_LAYER_FLAG, raising=False)
    model, args = _new_model(num_hidden_layers=8)
    _arm(model, KPRED)
    cache, token = _prefill(model, args, s=6, seed=9)
    # clear any stash the prefill's per-layer forward may have left.
    for layer in model.model.layers:
        layer.mlp.switch_mlp._mtplx_gate_prefetch_pending = None
    # a T=4 forward (the DSpark verify row batch shape): predictor must skip.
    verify_ids = mx.array([[token, token, token, token]])
    mx.eval(model(verify_ids, cache=cache))
    for i, layer in enumerate(model.model.layers):
        pend = getattr(layer.mlp.switch_mlp, "_mtplx_gate_prefetch_pending", None)
        assert pend is None, f"layer {i} stashed a prediction on a T>1 verify forward"
