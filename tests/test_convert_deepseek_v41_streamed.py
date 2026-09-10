"""Unit tests for the DeepSeek-V4.1-Flash streaming converter primitives."""

from __future__ import annotations

import numpy as np
import pytest

from mtplx import deepseek_v41_convert as dc
from mtplx.expert_manifest import _COMPONENTS, _expected_component_shape
from mtplx.expert_streaming_models import get_model_spec


# --------------------------------------------------------------------------
# decode LUTs
# --------------------------------------------------------------------------
def test_e4m3_lut_known_values():
    lut = dc.E4M3_LUT
    assert lut[0x00] == 0.0
    assert lut[0x38] == 1.0          # S=0 exp=7 man=0 -> 2^0
    assert lut[0x7E] == 448.0        # max normal
    assert lut[0xB8] == -1.0         # sign bit set
    assert np.isnan(lut[0x7F]) and np.isnan(lut[0xFF])
    # subnormal: 0x01 -> (1/8) * 2^-6
    assert abs(lut[0x01] - (1 / 8) * 2 ** -6) < 1e-12


def test_e8m0_lut_known_values():
    lut = dc.E8M0_LUT
    assert lut[127] == 1.0
    assert lut[128] == 2.0
    assert lut[126] == 0.5
    assert np.isnan(lut[0xFF])


# --------------------------------------------------------------------------
# FP4 (E2M1) dequant vs a straight FP4_TABLE transcription  (task requirement)
# --------------------------------------------------------------------------
def test_dequant_fp4_matches_direct_table_indexing():
    rng = np.random.default_rng(1234)
    out, in_dim = 8, 128
    packed = rng.integers(0, 256, size=(out, in_dim // 2), dtype=np.uint8)
    # e8m0 scale bytes in a benign exponent range (avoid 0xFF NaN)
    scale = rng.integers(120, 135, size=(out, in_dim // 32), dtype=np.uint8)

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    expected = np.empty((out, in_dim), dtype=np.float32)
    expected[:, 0::2] = dc.FP4_TABLE[low]       # low nibble first
    expected[:, 1::2] = dc.FP4_TABLE[high]
    scale_f = (2.0 ** (scale.astype(np.float32) - 127.0)).astype(np.float32)
    expected *= np.repeat(scale_f, 32, axis=1)

    got = dc.dequant_fp4(packed, scale)
    assert got.shape == (out, in_dim)
    np.testing.assert_array_equal(got, expected)


def test_dequant_fp4_rejects_scale_shape_mismatch():
    packed = np.zeros((4, 16), dtype=np.uint8)
    with pytest.raises(ValueError):
        dc.dequant_fp4(packed, np.zeros((4, 3), dtype=np.uint8))


# --------------------------------------------------------------------------
# FP8-block dequant vs a straight transcription
# --------------------------------------------------------------------------
def test_dequant_fp8_block_matches_direct():
    rng = np.random.default_rng(99)
    o, i = 64, 96
    weight = rng.integers(0, 256, size=(o, i), dtype=np.uint8)
    weight[weight == 0x7F] = 0x38  # scrub NaN codes so equality is well-defined
    weight[weight == 0xFF] = 0xB8
    scale = rng.integers(120, 135, size=(o // 32, i // 32), dtype=np.uint8)

    w = dc.E4M3_LUT[weight]
    s = (2.0 ** (scale.astype(np.float32) - 127.0)).astype(np.float32)
    s = np.repeat(np.repeat(s, 32, axis=0), 32, axis=1)
    expected = w * s

    got = dc.dequant_fp8_block(weight, scale)
    assert got.shape == (o, i)
    np.testing.assert_array_equal(got, expected)


# --------------------------------------------------------------------------
# record geometry matches the manifest contract
# --------------------------------------------------------------------------
def test_components_match_manifest_order():
    assert dc.COMPONENTS == _COMPONENTS


def test_record_bytes_matches_port_plan():
    # 8,847,360 packed + 2,211,840 scale/bias = 11,059,200 (2.5 bpw, 158.2 GiB bank)
    assert dc.EXPERT_RECORD_BYTES == 11_059_200
    assert dc.EXPERT_RECORD_BYTES % dc.ALIGNMENT == 0
    packed = 3 * dc.HIDDEN_SIZE * dc.MOE_INTERMEDIATE * dc.EXPERT_BITS // 8
    scale_bias = (3 * dc.HIDDEN_SIZE * dc.MOE_INTERMEDIATE // dc.GROUP_SIZE) * 2 * 2
    assert packed == 8_847_360
    assert scale_bias == 2_211_840


def test_component_shapes_match_manifest_expectation():
    spec = get_model_spec(dc.MODEL_KEY)
    total = 0
    for component in dc.COMPONENTS:
        dtype, shape, length = _expected_component_shape(spec, component)
        total += length
        if component.endswith(".weight"):
            assert dtype == "U32"
        else:
            assert dtype == "BF16"
    assert total == dc.EXPERT_RECORD_BYTES


# --------------------------------------------------------------------------
# mx.quantize path: shapes, dtypes, byte lengths, and a full 9-segment record
# --------------------------------------------------------------------------
def test_quantize_affine_shapes_and_bytes():
    rng = np.random.default_rng(7)
    # gate_proj geometry [out=2304, in=5120] but shrink for test speed
    w = rng.standard_normal((128, 320)).astype(np.float32)
    packed, scales, biases = dc.quantize_affine(w, bits=2, group_size=64)
    assert str(packed.dtype).endswith("uint32")
    assert packed.shape == (128, 320 * 2 // 32)
    assert scales.shape == (128, 320 // 64)
    b_w, b_s, b_b = dc.component_bytes(packed, scales, biases)
    assert len(b_w) == packed.size * 4
    assert len(b_s) == scales.size * 2
    assert len(b_b) == biases.size * 2


def test_full_expert_record_is_exact_size():
    rng = np.random.default_rng(3)
    blobs = []
    for proj in ("gate_proj", "up_proj", "down_proj"):
        if proj == "down_proj":
            out, in_dim = dc.HIDDEN_SIZE, dc.MOE_INTERMEDIATE
        else:
            out, in_dim = dc.MOE_INTERMEDIATE, dc.HIDDEN_SIZE
        w = rng.standard_normal((out, in_dim)).astype(np.float32)
        packed, scales, biases = dc.quantize_affine(w, bits=2, group_size=64)
        blobs.extend(dc.component_bytes(packed, scales, biases))
    record = b"".join(blobs)
    assert len(record) == dc.EXPERT_RECORD_BYTES


def test_quantize_dequantize_cosine_reasonable():
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(11)
    w = rng.standard_normal((256, 512)).astype(np.float32)
    packed, scales, biases = dc.quantize_affine(w, bits=2, group_size=64)
    wdq = mx.dequantize(packed, scales, biases, group_size=64, bits=2)
    mx.eval(wdq)
    a = np.asarray(w, dtype=np.float32).ravel()
    b = np.array(wdq.astype(mx.float32)).ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    # Gaussian noise is the worst case for 2-bit; real expert rows do better.
    assert cos > 0.80


# --------------------------------------------------------------------------
# resident classification
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,shape,dtype,expected",
    [
        ("layers.0.attn.wq_a.weight", (1280, 5120), "F8_E4M3", "quantize"),
        ("layers.0.ffn.shared_experts.w1.weight", (2304, 5120), "F8_E4M3", "quantize"),
        ("embed.weight", (129280, 5120), "BF16", "quantize"),
        ("vision.blocks.0.mlp.w1.weight", (5632, 1024), "BF16", "quantize"),
        ("layers.0.ffn.gate.weight", (384, 5120), "BF16", "keep"),      # router exact
        ("layers.0.ffn.gate.bias", (384,), "F32", "keep"),
        ("layers.0.attn_norm.weight", (5120,), "BF16", "keep"),
        ("layers.0.hc_attn_fn", (24, 20480), "F32", "keep"),            # 2-D but hc_*
        ("layers.0.attn.attn_sink", (64,), "F32", "keep"),
        ("layers.0.attn.wq_a.scale", (40, 160), "F8_E8M0", "drop"),
    ],
)
def test_resident_disposition(name, shape, dtype, expected):
    entry = dc.TensorEntry(name, dtype, shape, 0, 1)
    assert dc.resident_disposition(entry) == expected


def test_expert_and_engram_and_mtp_predicates():
    assert dc.is_bank_expert("layers.7.ffn.experts.100.w2.weight")
    assert dc.is_bank_expert("layers.39.ffn.experts.383.w3.scale")
    assert not dc.is_bank_expert("layers.7.ffn.shared_experts.w2.weight")
    assert dc.is_mtp_expert("mtp.0.ffn.experts.5.w1.weight")
    assert not dc.is_bank_expert("mtp.0.ffn.experts.5.w1.weight")
    assert dc.is_engram("layers.1.engram.embed.weight")


# --------------------------------------------------------------------------
# raw bytes -> f32 decoding
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# MTP routed experts: affine q8/gs32 near-exactness from an FP4 source grid,
# and the reason mxfp4/mxfp8 were rejected (not bit-exact in mlx 0.32.0).
# --------------------------------------------------------------------------
def _fp4_grid(rng, out, in_dim):
    """Synthetic tensor drawn from the exact E2M1 * per-32-col E8M0 source grid."""
    codes = rng.integers(0, 16, size=(out, in_dim))
    exps = rng.integers(122, 132, size=(out, in_dim // 32))
    return (dc.FP4_TABLE[codes] * np.repeat((2.0 ** (exps - 127)).astype(np.float32), 32, axis=1)).astype(np.float32)


def test_mtp_affine_q8_gs32_near_exact():
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(20)
    ref = _fp4_grid(rng, 256, 512)
    pk, sc, bi = dc.quantize_affine(ref, bits=8, group_size=32)
    deq = np.array(mx.dequantize(pk, sc, bi, group_size=32, bits=8, mode="affine").astype(mx.float32))
    a = ref.ravel(); b = deq.ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos > 0.9999, cos  # near-exact requant of the FP4 draft-head grid


def test_mxfp4_is_not_bit_exact_in_mlx_032():
    """Documents why MTP experts use affine q8/gs32, not mxfp4: mlx 0.32.0's
    mxfp4 re-derives group scales with headroom and is NOT a bit-exact repack."""
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(21)
    ref = _fp4_grid(rng, 128, 256)
    q = mx.quantize(mx.array(ref), group_size=32, bits=4, mode="mxfp4")
    deq = np.array(mx.dequantize(*q, group_size=32, bits=4, mode="mxfp4").astype(mx.float32))
    assert not np.array_equal(deq, ref)  # not exact -> affine q8 is the correct choice


def test_raw_bf16_roundtrips_to_f32():
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    vals = np.array([[1.0, -2.5, 0.0], [3.25, 100.0, -0.5]], dtype=np.float32)
    bf = mx.array(vals).astype(mx.bfloat16)
    raw = np.array(bf.view(mx.uint16)).astype("<u2").tobytes()
    entry = dc.TensorEntry("x", "BF16", (2, 3), 0, len(raw))
    got = dc.raw_to_f32(entry, raw)
    # bf16 has 8 mantissa bits; these values are all exactly representable
    np.testing.assert_array_equal(got, vals)
