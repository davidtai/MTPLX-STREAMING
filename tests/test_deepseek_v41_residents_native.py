"""W18: MLX-native resident repacks (mxfp8 dense, mxfp4 MTP experts).

Proves the block-scale layout claim David asked for: the source ``F8_E4M3`` +
``F8_E8M0`` 32x32 block-scale dense tensors repack *bit-exactly* to ``mxfp8``
gs32, and the FP4 MTP experts repack bit-exactly to ``mxfp4`` gs32 --
``mx.dequantize`` of the repacked tensor is byte-for-byte the reference fp32
dequant (``np.array_equal``).

Two tiers:
  * synthetic (always runs, no artifact): construct F8_E4M3/E8M0 block tensors
    and FP4 tensors from the codec's own LUTs and prove the round-trip, plus the
    negative case (a value that cannot map exactly is rejected).
  * real source slice (opt-in, ``MTPLX_DSV41_SRC`` -> the Hub source dir):
    every FP8 dense family at layer 0 + an MTP expert, proven bit-exact.

CPU-pinned at import (mx.quantize/dequantize run on the CPU stream; no GPU).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

import pytest

from mtplx import deepseek_v41_convert as dc

NATIVE_GS = dc.NATIVE_GROUP_SIZE  # 32


# ---------------------------------------------------------------------------
# synthetic: build an exact F8_E4M3 + E8M0 block tensor and repack it
# ---------------------------------------------------------------------------
def _random_e4m3_bytes(shape, rng) -> np.ndarray:
    """Uniform E4M3 code bytes, excluding the NaN pattern (0x7F/0xFF)."""
    b = rng.integers(0, 256, size=shape, dtype=np.uint8)
    b = np.where((b & 0x7F) == 0x7F, np.uint8(0), b)  # avoid NaN codes
    return b


def _e8m0_scale_bytes(shape, rng, lo=110, hi=124) -> np.ndarray:
    """E8M0 scale bytes in a moderate exponent band (2**(byte-127))."""
    return rng.integers(lo, hi, size=shape, dtype=np.uint8)


def test_mxfp8_exact_repack_synthetic():
    rng = np.random.default_rng(18)
    O, I = 64, 128  # 2x4 blocks of 32x32
    weight_u8 = _random_e4m3_bytes((O, I), rng)
    scale_u8 = _e8m0_scale_bytes((O // 32, I // 32), rng)
    codes, scales = dc.repack_fp8_block_to_mxfp8(weight_u8, scale_u8, "synthetic.wq")[:2]
    assert codes.dtype == mx.uint32 and scales.dtype == mx.uint8
    assert scales.shape == (O, I // NATIVE_GS)  # per-(row, 32-col) E8M0
    ref = dc.dequant_fp8_block(weight_u8, scale_u8)
    deq = np.array(
        mx.dequantize(codes, scales, group_size=NATIVE_GS, bits=8, mode="mxfp8").astype(mx.float32)
    )
    assert np.array_equal(deq, ref)  # bit-exact


def test_mxfp8_wide_block_maps_to_gs32():
    """A 1x128 source block scale (one scale per 128 cols, as some checkpoints
    use) still maps exactly: every 32-col mxfp8 group inside the block shares the
    block's scale, and mxfp8 reconstructs each group bit-exactly."""
    rng = np.random.default_rng(1128)
    O, I = 32, 128
    weight_u8 = _random_e4m3_bytes((O, I), rng)
    scale_u8 = _e8m0_scale_bytes((1, 1), rng)  # single 32x128 block
    # dequant_fp8_block derives block sizes from the shape ratio (32x128 here)
    ref = dc.dequant_fp8_block(weight_u8, scale_u8)
    codes, scales = dc.quantize_mxfp8_exact(ref, "synthetic.wide")
    deq = np.array(
        mx.dequantize(codes, scales, group_size=NATIVE_GS, bits=8, mode="mxfp8").astype(mx.float32)
    )
    assert np.array_equal(deq, ref)


def test_mxfp4_exact_repack_synthetic():
    rng = np.random.default_rng(4)
    O, I = 64, 128
    packed = rng.integers(0, 256, size=(O, I // 2), dtype=np.uint8)  # two E2M1 nibbles/byte
    scale_u8 = _e8m0_scale_bytes((O, I // 32), rng)
    ref = dc.dequant_fp4(packed, scale_u8)
    codes, scales = dc.quantize_mxfp4_exact(ref, "synthetic.expert")
    assert codes.dtype == mx.uint32 and scales.dtype == mx.uint8
    deq = np.array(
        mx.dequantize(codes, scales, group_size=NATIVE_GS, bits=4, mode="mxfp4").astype(mx.float32)
    )
    assert np.array_equal(deq, ref)


def test_mxfp8_rejects_inexact():
    """A value with more precision than E4M3 can hold (a bf16-source projection)
    must be rejected by the exactness gate, not silently shipped lossy."""
    rng = np.random.default_rng(7)
    vals = (rng.standard_normal((32, 64)).astype(np.float32) * 0.017)  # generic bf16-ish
    # not representable as E4M3*2^k in general -> the gate must raise
    with pytest.raises(ValueError, match="NOT bit-exact"):
        dc.quantize_mxfp8_exact(vals, "bf16.indexer.wk")


def test_native_disposition_map():
    def ent(name, dtype, shape=(64, 64)):
        return dc.TensorEntry(name, dtype, shape, 0, 0)

    assert dc.native_resident_disposition(ent("layers.0.attn.wq_a.weight", "F8_E4M3")) == "mxfp8"
    assert dc.native_resident_disposition(ent("layers.0.attn.indexer.wq_b.weight", "F8_E4M3")) == "mxfp8"
    assert dc.native_resident_disposition(ent("mtp.0.attn.wkv.weight", "F8_E4M3")) == "mxfp8"
    assert dc.native_resident_disposition(ent("mtp.0.ffn.experts.3.w1.weight", "I8")) == "mxfp4"
    assert dc.native_resident_disposition(ent("layers.0.ffn.experts.3.w1.weight", "I8")) == "drop"
    assert dc.native_resident_disposition(ent("layers.1.engram.embed.weight", "F8_E4M3")) == "drop"
    assert dc.native_resident_disposition(ent("layers.0.attn.wq_a.scale", "F8_E8M0")) == "drop"
    assert dc.native_resident_disposition(ent("embed.weight", "BF16")) == "keep"
    assert dc.native_resident_disposition(ent("head.weight", "BF16")) == "keep"
    assert dc.native_resident_disposition(ent("layers.0.attn.indexer.wk.weight", "BF16")) == "keep"
    assert dc.native_resident_disposition(ent("layers.0.attn.compressor.wkv.weight", "BF16")) == "keep"
    assert dc.native_resident_disposition(ent("layers.0.hc_attn_fn", "F32", (16, 256))) == "keep"


# ---------------------------------------------------------------------------
# real source slice (opt-in)
# ---------------------------------------------------------------------------
def _src_dir():
    p = os.environ.get("MTPLX_DSV41_SRC", str(Path.home() / "models" / "DeepSeek-V4.1-Flash-src"))
    return Path(p) if (Path(p) / "model.safetensors.index.json").is_file() else None


@pytest.mark.skipif(_src_dir() is None, reason="Hub source not present (set MTPLX_DSV41_SRC)")
def test_real_layer0_fp8_families_bit_exact():
    import json

    src = _src_dir()
    wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    families = [
        "layers.0.attn.wq_a", "layers.0.attn.wq_b", "layers.0.attn.wkv",
        "layers.0.attn.wo_a", "layers.0.attn.wo_b",
        "layers.0.ffn.shared_experts.w1", "layers.0.ffn.shared_experts.w2",
        "layers.0.ffn.shared_experts.w3",
    ]
    for base in families:
        shard = wm[base + ".weight"]
        hdr, ds = dc.read_safetensors_header(str(src / shard))
        ents = dc.tensor_entries(hdr)
        we, se = ents[base + ".weight"], ents[base + ".scale"]
        assert we.dtype == "F8_E4M3" and se.dtype == "F8_E8M0"
        fd = os.open(str(src / shard), os.O_RDONLY)
        try:
            w = np.frombuffer(dc.read_tensor_raw(fd, ds, we), np.uint8).reshape(we.shape)
            s = np.frombuffer(dc.read_tensor_raw(fd, ds, se), np.uint8).reshape(se.shape)
        finally:
            os.close(fd)
        codes, scales, ref = dc.repack_fp8_block_to_mxfp8(w, s, base)  # raises if inexact
        deq = np.array(
            mx.dequantize(codes, scales, group_size=NATIVE_GS, bits=8, mode="mxfp8").astype(mx.float32)
        )
        assert np.array_equal(deq, ref), base


@pytest.mark.skipif(_src_dir() is None, reason="Hub source not present (set MTPLX_DSV41_SRC)")
def test_real_mtp_expert_bit_exact():
    import json

    src = _src_dir()
    wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    base = "mtp.0.ffn.experts.0.w1"
    shard = wm[base + ".weight"]
    hdr, ds = dc.read_safetensors_header(str(src / shard))
    ents = dc.tensor_entries(hdr)
    we, se = ents[base + ".weight"], ents[base + ".scale"]
    fd = os.open(str(src / shard), os.O_RDONLY)
    try:
        w = np.frombuffer(dc.read_tensor_raw(fd, ds, we), np.uint8).reshape(we.shape)
        s = np.frombuffer(dc.read_tensor_raw(fd, ds, se), np.uint8).reshape(se.shape)
    finally:
        os.close(fd)
    codes, scales, ref = dc.repack_fp4_to_mxfp4(w, s, base)
    deq = np.array(
        mx.dequantize(codes, scales, group_size=NATIVE_GS, bits=4, mode="mxfp4").astype(mx.float32)
    )
    assert np.array_equal(deq, ref)


# ---------------------------------------------------------------------------
# both codecs construct, strict-load their own residents, and run a forward
# ---------------------------------------------------------------------------
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402


def _small_args(**over):
    base = dict(
        vocab_size=128, hidden_size=64, num_hidden_layers=3, num_attention_heads=2,
        head_dim=64, qk_rope_head_dim=16, q_lora_rank=64, o_lora_rank=64, o_groups=2,
        moe_intermediate_size=64, n_routed_experts=8, num_experts_per_tok=2,
        index_n_heads=2, index_head_dim=64, index_topk=4, sliding_window=8, window_size=8,
        swiglu_limit=0.5, compress_ratios=[0, 2, 1], kv_source_layer_ids=[1, 2],
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


class _StubSwitch(nn.Module):
    """Streamed-expert seam after bind: no resident params; returns zeros so a
    forward exercises the mxfp8 dense/shared projections without a bank."""

    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    def __call__(self, x, indices):
        top_k = indices.shape[-1]
        return mx.zeros((x.shape[0], top_k, self._dim), dtype=mx.float32)


@pytest.mark.parametrize(
    "quantization",
    [
        {"group_size": 64, "bits": 8, "mode": "affine"},   # original artifact codec
        {"group_size": 32, "bits": 8, "mode": "mxfp8"},    # native exact-repack codec
    ],
    ids=["affine_q8", "mxfp8"],
)
def test_both_codecs_strict_load_and_forward(quantization):
    args = _small_args()
    model = Model(args, quantize=True, quantization=quantization)
    # emulate bind_streamed_switches: routed experts stream (no resident params)
    for layer in model.model.layers:
        layer.mlp.switch_mlp = _StubSwitch(args.hidden_size)
    mx.eval(model.parameters())

    params = dict(tree_flatten(model.parameters()))
    resident_paths = set(params)
    assert not any(".switch_mlp." in p for p in resident_paths)

    # the mode's shape: mxfp8 residents carry NO biases; affine q8 does
    if quantization["mode"] == "mxfp8":
        assert not any(p.endswith(".biases") for p in resident_paths)
        # embed/head + the bf16-source projections stay dense (no scales)
        assert not any("embed_tokens.scales" in p for p in resident_paths)
        assert not any(p == "head.scales" for p in resident_paths)
        assert not any("indexer.wk.scales" in p for p in resident_paths)
    else:
        assert any(p.endswith(".biases") for p in resident_paths)
        assert any("embed_tokens.scales" in p for p in resident_paths)

    # a checkpoint-named resident dict built from the model's own arrays + drops
    ckpt = {_ckpt_name(p): v for p, v in params.items()}
    ckpt["layers.0.ffn.gate.bias_vl"] = mx.zeros((8,))          # must be dropped
    ckpt["vision.blocks.0.norm1.weight"] = mx.zeros((4,))        # must be dropped
    ckpt["mtp.0.attn.wq_a.weight"] = mx.zeros((4, 4))            # must be dropped
    sanitized = model.sanitize(ckpt)
    assert set(sanitized) == resident_paths, resident_paths.symmetric_difference(sanitized)
    model.load_weights(list(sanitized.items()), strict=True)     # raises if incomplete

    # forward smoke test: mxfp8/affine dense projections must run on CPU
    ids = mx.array([[0, 3, 7, 1, 5, 2, 9, 4]])
    logits = model(ids)
    mx.eval(logits)
    assert logits.shape == (1, 8, args.vocab_size)
    assert bool(mx.isfinite(logits).all())
