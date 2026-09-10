"""W3 strict spec validation for the DeepSeek-V4.1-Flash Q2 streamed bank.

Pins and proves, against the real artifact manifest:
  - total_tensor_bytes == 195_033_235_352 (measured header-inventory sum;
    resident 25_163_923_352 + routed 169_869_312_000),
  - resident/routed/record geometry under require_pinned_tensor_bytes=True,
  - the source-identity divergence (shipped manifest carries pre-publish
    ``local/...`` identity) is the ONLY thing that separates the HF-pinned spec
    from strict admission.

CPU only; reads the manifest JSON, never the 169 GiB bank.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from mtplx.expert_manifest import (
    ExpertManifestError,
    load_expert_manifest,
    validate_expert_manifest_spec,
)
from mtplx.expert_streaming_models import get_model_spec

ARTIFACT = Path(
    os.environ.get(
        "DSV41_ARTIFACT",
        os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2"),
    )
)
MANIFEST = ARTIFACT / "expert-manifest.json"
KEY = "deepseek-v41-flash-expert-q2"

TOTAL_TENSOR_BYTES = 195_033_235_352
RESIDENT_TENSOR_BYTES = 25_163_923_352
ROUTED_EXPERT_BYTES = 169_869_312_000
EXPERT_RECORD_BYTES = 11_059_200

pytestmark = pytest.mark.skipif(
    not MANIFEST.is_file(), reason=f"artifact manifest not present at {MANIFEST}"
)


@pytest.fixture(scope="module")
def spec():
    return get_model_spec(KEY)


@pytest.fixture(scope="module")
def manifest():
    return load_expert_manifest(MANIFEST)


def test_spec_bytes_are_pinned_to_measured_inventory(spec):
    assert spec.total_tensor_bytes == TOTAL_TENSOR_BYTES
    assert spec.routed_expert_bytes == ROUTED_EXPERT_BYTES
    assert spec.resident_bytes == RESIDENT_TENSOR_BYTES
    assert spec.expert_record_bytes == EXPERT_RECORD_BYTES
    # derivation: resident + routed == total (no delta vs the manifest)
    assert spec.resident_bytes + spec.routed_expert_bytes == spec.total_tensor_bytes


def test_spec_kv_and_indexer_pins(spec):
    # Phase-1 bf16 global CSA2 KV: 3*(1280//2) + 1280 = 3200 B/token.
    assert spec.kv_bytes_per_token == 3_200
    # index_source_layer_ids from the artifact text_config.
    assert spec.full_indexer_layers == (2, 8, 14, 20, 24, 28, 32, 36)
    assert spec.top_k == 6
    assert spec.expert_count == 384
    assert spec.total_layers == 40
    assert spec.routed_layer_count == 40
    assert spec.mtp_included is False


def test_quant_model_pinned_to_public_hf_repo(spec):
    assert spec.quant_model == "OpensourceWTF/DeepSeek-V4.1-Flash-MTPLX-streaming-q2"
    assert spec.quant_revision == "b64980a16283647bb213ab475335f38f516e0d9e"


def test_manifest_bytes_match_pinned_spec(spec, manifest):
    assert manifest.artifact_tensor_bytes == spec.total_tensor_bytes
    assert manifest.resident_tensor_bytes == spec.resident_bytes
    assert manifest.routed_expert_bytes == spec.routed_expert_bytes
    assert manifest.model_key == spec.key


def test_strict_validation_passes_on_source_rebased_spec(spec, manifest):
    # The shipped manifest carries the pre-publish source identity; rebase the
    # spec's quant_model/quant_revision to it (this is exactly the identity a
    # republished manifest will carry) and run STRICT validation with
    # require_pinned_tensor_bytes=True.  Everything -- bytes, geometry, record
    # keys, component shapes -- must pass.
    spec_rebased = replace(
        spec,
        quant_model=manifest.source_repo,
        quant_revision=manifest.source_revision,
    )
    validate_expert_manifest_spec(
        manifest, spec_rebased, require_pinned_tensor_bytes=True
    )


def test_hf_pinned_spec_fails_only_on_source_identity(spec, manifest):
    # Proves the byte/geometry pins are correct: the HF-pinned spec fails strict
    # validation ONLY because the shipped manifest still carries pre-publish
    # ``local/...`` source identity.  This is the documented publish blocker.
    assert manifest.source_repo != spec.quant_model
    with pytest.raises(ExpertManifestError) as excinfo:
        validate_expert_manifest_spec(
            manifest, spec, require_pinned_tensor_bytes=True
        )
    assert "source identity" in str(excinfo.value)
