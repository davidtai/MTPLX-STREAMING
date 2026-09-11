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
