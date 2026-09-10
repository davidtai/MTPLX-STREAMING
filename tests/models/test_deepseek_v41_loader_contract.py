"""Loader/engram interface contract for the DeepSeek-V4.1 text Model.

Validates the surface the streaming loader (mtplx/models/deepseek_v41_loader.py)
and the engram module (mtplx/engram_v41.py) rely on: the ``engram_bank_path``
constructor arg, the ``mlp.switch_mlp`` streamed-expert seam, q8-resident
parameterisation, ``sanitize`` name remapping with a strict q8 load, and the
three-argument engram hook with advance/rollback threading.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mtplx.models.deepseek_v41 import DeepseekV41Cache, Model, ModelArgs


def _q8_args(**over):
    # group-64-aligned so nn.quantize produces q8 residents; a Full layer (1)
    # with compressor+indexer and a Full/candidate layer (2) exercise every name.
    base = dict(
        vocab_size=128, hidden_size=64, num_hidden_layers=3,
        num_attention_heads=2, head_dim=64, qk_rope_head_dim=16, q_lora_rank=64,
        o_lora_rank=64, o_groups=2, moe_intermediate_size=64, n_routed_experts=8,
        num_experts_per_tok=2, index_n_heads=2, index_head_dim=64, index_topk=4,
        sliding_window=8, window_size=8, swiglu_limit=0.5,
        compress_ratios=[0, 2, 1], kv_source_layer_ids=[1, 2],
        index_source_layer_ids=[1, 2], candidate_source_layer_id=2,
        candidate_topk_blocks=2, candidate_block_size=2,
        rope_scaling={"rope_type": "yarn", "factor": 16, "beta_fast": 32,
                      "beta_slow": 1, "original_max_position_embeddings": 65536},
    )
    base.update(over)
    return ModelArgs(**base)


def _ckpt_name(model_path: str) -> str:
    """Inverse of Model.sanitize for a resident parameter path (test-side)."""
    if model_path == "model.norm_weight":
        return "norm.weight"
    if model_path.startswith("head."):
        return model_path
    if model_path.startswith("model.embed_tokens."):
        return "embed." + model_path[len("model.embed_tokens."):]
    rest = model_path[len("model."):] if model_path.startswith("model.") else model_path
    rest = rest.replace(".mlp.", ".ffn.")
    rest = rest.replace("gate.e_score_correction_bias", "gate.bias")
    rest = rest.replace("norm_weight", "norm.weight")
    return rest


class _NoParamSwitch(nn.Module):
    """Stand-in for the streamed switch after bind_streamed_switches (no params)."""

    def __call__(self, x, indices):  # pragma: no cover - not called here
        raise RuntimeError("streamed switch is unbound in this test")


def test_construct_interface():
    model = Model(_q8_args(), engram_bank_path="/artifact/engram", quantize=True)
    assert model.engram_bank_path == "/artifact/engram"
    assert model.model_type == "deepseek_v41"
    # streamed-expert seam on every layer (hy3 convention)
    for layer in model.model.layers:
        assert hasattr(layer.mlp, "switch_mlp")
        assert layer.engram_hook is None
    # q8 residents were created
    paths = [p for p, _ in tree_flatten(model.parameters())]
    assert sum(1 for p in paths if p.endswith(".scales")) > 0


def test_sanitize_roundtrip_and_drops():
    model = Model(_q8_args(), quantize=True)
    params = dict(tree_flatten(model.parameters()))
    for path in params:
        if "switch_mlp" in path:
            continue  # streamed, not a resident tensor
        ckpt = _ckpt_name(path)
        assert list(model.sanitize({ckpt: params[path]}).keys()) == [path], (ckpt, path)
    # dropped: VL bias, vision/aligner/image, mtp.*
    dropped = model.sanitize({
        "layers.0.ffn.gate.bias_vl": mx.zeros((8,)),
        "vision.blocks.0.norm1.weight": mx.zeros((4,)),
        "mtp.0.attn.wq_a.weight": mx.zeros((4, 4)),
    })
    assert dropped == {}


def test_strict_q8_load():
    model = Model(_q8_args(), quantize=True)
    # emulate bind_streamed_switches: streamed experts carry no resident params
    for layer in model.model.layers:
        layer.mlp.switch_mlp = _NoParamSwitch()
    mx.eval(model.parameters())

    params = dict(tree_flatten(model.parameters()))
    resident_paths = set(params)
    # a checkpoint-named resident dict built from the model's own arrays
    ckpt = {_ckpt_name(p): v for p, v in params.items()}
    ckpt["layers.0.ffn.gate.bias_vl"] = mx.zeros((8,))  # must be dropped
    ckpt["vision.blocks.0.norm1.weight"] = mx.zeros((4,))

    sanitized = model.sanitize(ckpt)
    assert set(sanitized) == resident_paths, (
        resident_paths.symmetric_difference(sanitized)
    )
    model.load_weights(list(sanitized.items()), strict=True)  # raises if incomplete


def test_engram_hook_wiring_and_rollback():
    model = Model(_q8_args(), quantize=False)

    class FakeState:
        def __init__(self):
            self.advanced = []
            self.trimmed = []

        def advance(self, ids, token_mask=None):
            self.advanced.append(tuple(int(d) for d in ids.shape))

        def trim(self, n):
            self.trimmed.append(int(n))

        def current_row_ids(self, i):  # pragma: no cover - fake hook ignores it
            return None

    calls = []

    def hook(hidden, token_ids, cache_state):
        calls.append((tuple(hidden.shape), tuple(token_ids.shape), cache_state))
        return hidden  # no-op engram

    model.model.layers[1].engram_hook = hook
    cache = model.make_cache()
    cache.engram_state = FakeState()

    prompt = mx.array([[(i * 3 + 1) % model.args.vocab_size for i in range(12)]])
    model(prompt, cache)
    assert cache.engram_state.advanced == [(1, 12)]  # advanced once per step
    hc, dim = model.args.hc_mult, model.args.hidden_size
    assert calls[-1][0] == (1, 12, hc, dim)  # hc-expanded stream
    assert calls[-1][1] == (1, 12)           # token ids [B, L]
    assert calls[-1][2] is cache.engram_state

    mark = cache.mark()
    model(mx.array([[7]]), cache)
    assert cache.engram_state.advanced[-1] == (1, 1)
    cache.rollback(mark)
    assert cache.engram_state.trimmed == [1]  # engram history rolled back in step
    assert cache.offset == 12
