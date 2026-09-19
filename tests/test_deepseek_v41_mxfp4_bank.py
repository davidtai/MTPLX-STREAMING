"""W16 native-mxfp4 routed-expert bank: record math + lossless-repack invariants.

CPU-pinned (MLX defaults to Metal; these must run on CPU per the worker-test
rule).  The self-contained tests need only mlx + numpy + the converter
primitives.  The spec/artifact tests skip cleanly until W15's mxfp4 spec entry
lands and the bank is built (heavy artifact check gated on MTPLX_RUN_HEAVY_W16=1).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)  # worker-test rule: pin MLX to CPU

from mtplx import deepseek_v41_convert as dc

MODEL_KEY = "deepseek-v41-flash-expert-mxfp4"
GROUP, BITS = 32, 4
# geometry: 3 projections of 5120*2304 params each
PARAMS = 3 * dc.HIDDEN_SIZE * dc.MOE_INTERMEDIATE  # 35,389,440
EXPECT_RECORD_BYTES = 18_800_640
EXPECT_BANK_BYTES = 40 * 384 * EXPECT_RECORD_BYTES  # 288,777,830,400


def mxfp4_record_bytes(hidden=dc.HIDDEN_SIZE, inter=dc.MOE_INTERMEDIATE,
                       bits=BITS, group=GROUP) -> int:
    params = 3 * hidden * inter
    return params * bits // 8 + params // group  # E8M0 1-byte scale/group, no bias


# --------------------------------------------------------------------------
# record / bank / bytes-per-token math
# --------------------------------------------------------------------------
def test_record_bytes_matches_spec_math():
    assert mxfp4_record_bytes() == EXPECT_RECORD_BYTES
    # packed 4-bit + 1-byte E8M0 per 32, no bias
    assert PARAMS * BITS // 8 == 17_694_720
    assert PARAMS // GROUP == 1_105_920
    assert 17_694_720 + 1_105_920 == EXPECT_RECORD_BYTES


def test_bank_and_bytes_per_token():
    assert EXPECT_BANK_BYTES == 288_777_830_400
    # 268.95 GiB
    assert abs(EXPECT_BANK_BYTES / 1024**3 - 268.95) < 0.05
    # cold bytes/token at top-6 * 40 layers = 4.20 GiB
    bpt = 6 * 40 * EXPECT_RECORD_BYTES
    assert bpt == 4_512_153_600
    assert abs(bpt / 1024**3 - 4.20) < 0.01


def test_mxfp4_is_smaller_than_affine_q4_gs64():
    # native mxfp4 (1-byte scale, no bias) < affine q4 gs64 (bf16 scale+bias)
    affine_q4_gs64 = PARAMS * 4 // 8 + (PARAMS // 64) * 2 * 2
    assert affine_q4_gs64 == 19_906_560
    assert EXPECT_RECORD_BYTES < affine_q4_gs64


# --------------------------------------------------------------------------
# lossless-repack invariant (the whole reason mxfp4 was chosen)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_mxfp4_repack_is_bit_exact_on_synthetic_fp4(seed):
    """dequant_fp4(source) -> mx.quantize(mxfp4) -> mx.dequantize must be exact."""
    rng = np.random.default_rng(seed)
    out_, in_ = 8, 128  # in_ divisible by 32
    packed = rng.integers(0, 256, size=(out_, in_ // 2), dtype=np.uint8)
    scale = rng.integers(120, 135, size=(out_, in_ // 32), dtype=np.uint8)  # avoid 0xFF (NaN)
    src_f32 = dc.dequant_fp4(packed, scale)

    w = mx.array(np.ascontiguousarray(src_f32, dtype=np.float32))
    pk, sc = mx.quantize(w, group_size=GROUP, bits=BITS, mode="mxfp4")
    mx.eval(pk, sc)
    # framing: packed uint32 [out, in/8], scales uint8 [out, in/32], no biases
    assert pk.dtype == mx.uint32 and tuple(pk.shape) == (out_, in_ // 8)
    assert sc.dtype == mx.uint8 and tuple(sc.shape) == (out_, in_ // GROUP)

    deq = np.array(mx.dequantize(pk, sc, group_size=GROUP, bits=BITS, mode="mxfp4").astype(mx.float32))
    assert np.array_equal(deq, src_f32), f"maxabs={np.abs(deq - src_f32).max()}"


def test_mxfp4_quantize_returns_no_bias_leaf():
    w = mx.array(np.random.randn(4, 64).astype(np.float32))
    q = mx.quantize(w, group_size=GROUP, bits=BITS, mode="mxfp4")
    assert len(q) == 2  # (packed, scales) — affine returns 3 (with biases)


# --------------------------------------------------------------------------
# spec entry (W15) — skips until the mxfp4 spec lands
# --------------------------------------------------------------------------
def _get_spec():
    from mtplx.expert_streaming_models import get_model_spec
    try:
        return get_model_spec(MODEL_KEY)
    except Exception:
        return None


def test_mxfp4_spec_entry_when_present():
    spec = _get_spec()
    if spec is None:
        pytest.skip("W15 mxfp4 spec entry not yet committed")
    assert spec.key == MODEL_KEY
    assert spec.quant_bits == BITS and spec.quant_group_size == GROUP
    assert spec.expert_codec == "mxfp4"
    assert spec.expert_record_bytes == EXPECT_RECORD_BYTES
    assert spec.routed_expert_bytes == EXPECT_BANK_BYTES
    assert spec.swiglu_limit == 10.0  # DeepSeek-V4.1-Flash clamps experts at +/-10


# --------------------------------------------------------------------------
# built artifact — heavy, gated
# --------------------------------------------------------------------------
ARTIFACT = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4").expanduser()


@pytest.mark.skipif(os.environ.get("MTPLX_RUN_HEAVY_W16") != "1",
                    reason="heavy artifact check; set MTPLX_RUN_HEAVY_W16=1")
def test_built_bank_manifest_identity():
    import json
    man = ARTIFACT / "expert-manifest.json"
    if not man.is_file():
        pytest.skip("mxfp4 bank not built yet")
    m = json.loads(man.read_text())
    assert m["model_key"] == MODEL_KEY
    # identity is stamped from the spec (build_mxfp4_manifest: source_repo=spec.quant_model)
    from mtplx.expert_streaming_models import get_model_spec
    spec = get_model_spec(MODEL_KEY)
    assert m["source_repo"] == spec.quant_model
    assert m["source_revision"] == spec.quant_revision
    assert m["artifact"]["record_count"] == 40 * 384
    assert (ARTIFACT / "experts.bin").stat().st_size == EXPECT_BANK_BYTES
