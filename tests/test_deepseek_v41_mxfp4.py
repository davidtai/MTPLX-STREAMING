"""Native-mxfp4 routed-expert codec: manifest admission + spec byte math.

CPU-only, no big data -- the manifest records are synthesized metadata (the
converter builds identical records from the real bank).  Exercises the mxfp4
mode end to end through ``validate_structure`` + ``validate_expert_manifest_spec``
and pins the record byte layout the streaming runtime consumes.
"""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx

mx.set_default_device(mx.cpu)

import pytest

from mtplx import expert_manifest as EM
from mtplx.expert_manifest import (
    EMPTY_SHA256,
    ExpertManifest,
    ExpertRecord,
    ShardInfo,
    SidecarInfo,
    MXFP4_ALIGNMENT,
    TensorSegment,
    expert_components_for_mode,
    validate_expert_manifest_spec,
)
from mtplx.expert_streaming_models import get_model_spec

RECORD_BYTES = 18_800_640
# Per-projection mxfp4 leaf layout for the pinned DeepSeek-V4.1-Flash geometry
# (hidden 5120, expert_hidden 2304): (component, dtype, shape, length).
_PROJ = {
    "gate_proj": ((2304, 640), (2304, 160)),  # w1: [2304, 5120]
    "up_proj": ((2304, 640), (2304, 160)),    # w3: [2304, 5120]
    "down_proj": ((5120, 288), (5120, 72)),   # w2: [5120, 2304]
}


def _record(layer: int, expert: int, base: int) -> ExpertRecord:
    segments = []
    cursor = base
    for proj, (wshape, sshape) in _PROJ.items():
        wlen = 4 * wshape[0] * wshape[1]
        slen = sshape[0] * sshape[1]
        segments.append(TensorSegment(
            component=f"{proj}.weight",
            tensor=f"layers.{layer}.ffn.experts.{expert}.{proj}.weight",
            shard="experts.bin", offset=cursor, length=wlen, dtype="U32", shape=wshape))
        cursor += wlen
        segments.append(TensorSegment(
            component=f"{proj}.scales",
            tensor=f"layers.{layer}.ffn.experts.{expert}.{proj}.scales",
            shard="experts.bin", offset=cursor, length=slen, dtype="U8", shape=sshape))
        cursor += slen
    assert cursor - base == RECORD_BYTES
    return ExpertRecord(layer=layer, expert=expert, logical_bytes=RECORD_BYTES,
                        segments=tuple(segments), sidecar_offset=base,
                        sidecar_length=RECORD_BYTES)


def _pilot_manifest(spec, layers=(0, 1)):
    records = []
    idx = 0
    for layer in layers:
        for expert in range(spec.expert_count):
            records.append(_record(layer, expert, idx * RECORD_BYTES))
            idx += 1
    routed = len(records) * RECORD_BYTES
    size = routed
    sidecar = ShardInfo(name="experts.bin", size=size, header_bytes=0,
                        header_sha256=EMPTY_SHA256, sha256=EMPTY_SHA256, kind="sidecar")
    return ExpertManifest(
        model_key=spec.key, source_repo=spec.quant_model,
        source_revision=spec.quant_revision, quant_bits=4, quant_group_size=32,
        quant_mode="mxfp4", artifact_tensor_bytes=routed, resident_tensor_bytes=0,
        routed_expert_bytes=routed, shards=(sidecar,), resident_tensors=(),
        records=tuple(records),
        sidecar=SidecarInfo(file="experts.bin", alignment=MXFP4_ALIGNMENT, size=size,
                            sha256=EMPTY_SHA256),
    ).with_digest()


def test_expert_components_for_mode_mxfp4():
    assert expert_components_for_mode("mxfp4") == (
        "gate_proj.weight", "gate_proj.scales",
        "up_proj.weight", "up_proj.scales",
        "down_proj.weight", "down_proj.scales",
    )


def test_mxfp4_pilot_manifest_admits_against_pilot_spec():
    spec = get_model_spec("deepseek-v41-flash-expert-mxfp4")
    pilot = replace(spec, routed_layer_start=0, routed_layer_count=2)
    manifest = _pilot_manifest(pilot)
    manifest.validate_structure()  # structural: mode/bits/group/components/bytes
    validate_expert_manifest_spec(manifest, pilot, require_pinned_tensor_bytes=False)
    assert manifest.quant_mode == "mxfp4"
    assert manifest.routed_expert_bytes == 2 * spec.expert_count * RECORD_BYTES


def test_mxfp4_roundtrips_through_manifest_json():
    spec = get_model_spec("deepseek-v41-flash-expert-mxfp4")
    pilot = replace(spec, routed_layer_start=0, routed_layer_count=1)
    manifest = _pilot_manifest(pilot, layers=(0,))
    restored = ExpertManifest.from_dict(manifest.to_dict())
    assert restored.quant_mode == "mxfp4"
    assert restored.records[0].segments[0].component == "gate_proj.weight"
    assert restored.records[0].segments[1].dtype == "U8"


def test_mxfp4_record_with_bias_leaf_is_rejected():
    """A record carrying an extra bias leaf (wrong component order) must not admit."""
    spec = get_model_spec("deepseek-v41-flash-expert-mxfp4")
    pilot = replace(spec, routed_layer_start=0, routed_layer_count=1)
    good = _pilot_manifest(pilot, layers=(0,))
    payload = good.to_dict()
    rec0 = payload["records"][0]
    rec0["segments"].append({
        "component": "gate_proj.biases", "tensor": "x", "shard": "experts.bin",
        "offset": 0, "length": 2, "dtype": "BF16", "shape": [1],
    })
    rec0["logical_bytes"] = RECORD_BYTES + 2
    with pytest.raises(EM.ExpertManifestError):
        ExpertManifest.from_dict(payload, verify_digest=False)


def test_mxfp4_wrong_group_size_is_rejected():
    """mxfp4 manifests must group in 32s (structural guard)."""
    spec = get_model_spec("deepseek-v41-flash-expert-mxfp4")
    pilot = replace(spec, routed_layer_start=0, routed_layer_count=1)
    payload = _pilot_manifest(pilot, layers=(0,)).to_dict()
    payload["quantization"]["group_size"] = 64
    with pytest.raises(EM.ExpertManifestError):
        ExpertManifest.from_dict(payload, verify_digest=False)


def test_affine_q2_spec_unchanged_by_mxfp4_addition():
    """The affine DeepSeek-V4.1 Q2 spec byte math is untouched."""
    q2 = get_model_spec("deepseek-v41-flash-expert-q2")
    assert q2.expert_codec == "affine"
    assert q2.expert_record_bytes == 11_059_200
    assert q2.routed_expert_bytes == 169_869_312_000
    mxfp4 = get_model_spec("deepseek-v41-flash-expert-mxfp4")
    assert mxfp4.resident_bytes == q2.resident_bytes  # identical residents
    assert mxfp4.expert_record_bytes == RECORD_BYTES
