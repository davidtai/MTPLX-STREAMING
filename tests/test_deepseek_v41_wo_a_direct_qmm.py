"""Focused gates for the measured packed-MXFP8 target ``wo_a`` route."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.models import deepseek_v41 as dsv41


@pytest.fixture(autouse=True)
def _cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _attention():
    args = dsv41.ModelArgs(
        vocab_size=48,
        hidden_size=32,
        num_hidden_layers=8,
        num_attention_heads=4,
        head_dim=16,
        qk_rope_head_dim=4,
        q_lora_rank=16,
        o_lora_rank=8,
        o_groups=2,
        moe_intermediate_size=16,
        n_routed_experts=8,
        num_experts_per_tok=2,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=5,
        sliding_window=8,
        window_size=8,
        swiglu_limit=0.5,
        compress_ratios=[0, 0, 2, 2, 2, 1, 1, 1],
        kv_source_layer_ids=[2, 5],
        index_source_layer_ids=[2, 5, 6],
        candidate_source_layer_id=5,
        candidate_topk_blocks=3,
        candidate_block_size=2,
    )
    attn = dsv41.Attention(args, 0)
    mx.random.seed(41)
    attn.wo_a.weight = (0.1 * mx.random.normal(attn.wo_a.weight.shape)).astype(mx.bfloat16)
    attn.wo_a = nn.QuantizedLinear.from_linear(
        attn.wo_a, group_size=32, bits=8, mode="mxfp8"
    )
    mx.eval(attn.wo_a.parameters())
    return attn


def test_direct_qmm_flag_is_strictly_parsed():
    assert dsv41._resolve_attn_wo_a_direct(raw="") is False
    assert dsv41._resolve_attn_wo_a_direct(raw="off") is False
    assert dsv41._resolve_attn_wo_a_direct(raw="1") is True
    assert dsv41._resolve_attn_wo_a_direct(raw="true") is True
    with pytest.raises(ValueError, match="WO_A_DIRECT"):
        dsv41._resolve_attn_wo_a_direct(raw="maybe")


def test_direct_qmm_is_prebound_and_uses_no_dense_cache(monkeypatch):
    attn = _attention()
    receipt = attn.install_wo_a_direct_route()
    route = attn._out_prep_fused_impl
    assert receipt == {
        "mode": "direct_mxfp8_gather_qmm",
        "groups": 2,
        "rank": 8,
        "input_per_group": 32,
    }

    x = mx.random.normal((2, 6, 32)).astype(mx.bfloat16)
    wo = attn.wo_a
    dense_t = mx.contiguous(mx.swapaxes(mx.dequantize(
        wo.weight,
        wo.scales,
        wo.biases,
        group_size=wo.group_size,
        bits=wo.bits,
        mode=wo.mode,
    ).astype(mx.bfloat16).reshape(2, 8, 32), 1, 2))
    ref = mx.matmul(x, dense_t)

    monkeypatch.setattr(
        attn,
        "_wo_a_quant",
        lambda: (_ for _ in ()).throw(AssertionError("hot eligibility lookup")),
    )
    got = route.project_grouped(x)
    mx.eval(ref, got)
    assert tuple(got.shape) == (2, 6, 8)
    rel = float(mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)))) / (
        float(mx.max(mx.abs(ref.astype(mx.float32)))) + 1e-12
    )
    assert rel < 5e-2
    assert attn._out_prep_fused_impl is route
    assert getattr(attn, "_wo_a_bf16T_cache", None) is None
