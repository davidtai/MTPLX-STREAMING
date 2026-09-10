"""W3 loader + memory-plan + admission tests for DeepSeek-V4.1 Q2 streaming.

CPU only, no Metal, no full-bank load:
  - text-only resident filter (exact kept/skipped counts and bytes),
  - manifest admission on the real artifact (model_key, record bytes, sample
    record digests, and a real receipt via the admission code path with a
    trusted bank digest so the 169 GiB bank is never hashed),
  - plan_expert_memory slot counts at the 100 GiB knob for 4K/16K/64K,
  - loader end-to-end with a test double (runtime open with admission receipt +
    bind_streamed_switches over the real 40 routed layers + engram ctor arg),
  - the /health-relevant facts the loader determines.

Every artifact-backed test skips when the artifact is absent.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from deepseek_v41_test_double import (  # noqa: E402
    Model as DoubleModel,
    ModelArgs as DoubleArgs,
    model_classes as double_classes,
)

from mtplx.expert_admission import (  # noqa: E402
    TrustedFileDigest,
    admit_expert_artifact,
)
from mtplx.expert_manifest import load_expert_manifest  # noqa: E402
from mtplx.expert_streaming_models import (  # noqa: E402
    get_model_spec,
    plan_expert_memory,
)
from mtplx.models import deepseek_v41_loader as loader  # noqa: E402
from mtplx.models.expert_mlx import bind_streamed_switches  # noqa: E402

ARTIFACT = Path(
    os.environ.get(
        "DSV41_ARTIFACT",
        os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2"),
    )
)
MANIFEST = ARTIFACT / "expert-manifest.json"
BANK = ARTIFACT / "experts.bin"
KEY = "deepseek-v41-flash-expert-q2"

# Measured header-inventory figures (W3), asserted exactly.
KEPT_COUNT, KEPT_BYTES = 1616, 8_673_203_648
SKIPPED_COUNT, SKIPPED_BYTES = 4110, 16_490_719_704
SKIPPED_MTP_COUNT, SKIPPED_MTP_BYTES = 3584, 15_974_347_224
SKIPPED_VISION_COUNT, SKIPPED_VISION_BYTES = 526, 516_372_480
RECORD_BYTES = 11_059_200

KNOB = 100 * 1024**3
RESERVE = 7 * 1024**3
WINDOW = loader.SWA_WINDOW_BYTES  # 5_242_880

pytestmark = pytest.mark.skipif(
    not MANIFEST.is_file(), reason=f"artifact manifest not present at {MANIFEST}"
)


@pytest.fixture(scope="module")
def spec():
    return get_model_spec(KEY)


@pytest.fixture(scope="module")
def manifest():
    return load_expert_manifest(MANIFEST)


def _rebased_spec(spec, manifest):
    """Spec with source identity rebased to the shipped manifest.

    The shipped manifest carries pre-publish ``local/...`` identity; a
    republished manifest will carry the HF identity the spec pins.  Tests that
    must run admission against the artifact as-shipped rebase the spec to the
    manifest identity, which is exactly the post-republish state.
    """

    return replace(
        spec,
        quant_model=manifest.source_repo,
        quant_revision=manifest.source_revision,
    )


def _stat_receipt(manifest):
    banks = []
    for part in manifest.sidecar.parts:
        st = os.stat(ARTIFACT / part.file)
        banks.append(
            {
                "file": part.file,
                "sha256": part.sha256,
                "st_dev": st.st_dev,
                "st_ino": st.st_ino,
                "st_size": st.st_size,
                "st_mtime_ns": st.st_mtime_ns,
                "st_ctime_ns": st.st_ctime_ns,
            }
        )
    return {
        "schema": 1,
        "artifact_root": str(ARTIFACT.resolve()),
        "manifest_sha256": manifest.manifest_sha256,
        "banks": banks,
    }


# --------------------------------------------------------------------------
# text-only resident filter
# --------------------------------------------------------------------------
def test_text_only_filter_counts_and_bytes(manifest):
    part = loader.partition_text_residents(manifest)
    assert (part.kept_count, part.kept_bytes) == (KEPT_COUNT, KEPT_BYTES)
    assert (part.skipped_count, part.skipped_bytes) == (SKIPPED_COUNT, SKIPPED_BYTES)
    assert (part.skipped_mtp_count, part.skipped_mtp_bytes) == (
        SKIPPED_MTP_COUNT,
        SKIPPED_MTP_BYTES,
    )
    assert (part.skipped_vision_count, part.skipped_vision_bytes) == (
        SKIPPED_VISION_COUNT,
        SKIPPED_VISION_BYTES,
    )
    # partition covers exactly the manifest residents
    assert part.kept_count + part.skipped_count == len(manifest.resident_tensors)
    assert part.kept_bytes + part.skipped_bytes == manifest.resident_tensor_bytes


def test_text_only_filter_predicate_never_keeps_vision_or_mtp(manifest):
    for tensor in manifest.resident_tensors:
        keep = loader.is_text_resident(tensor.tensor)
        skipped = tensor.tensor.startswith(("mtp.", "vision.", "aligner.", "image_"))
        assert keep == (not skipped)


# --------------------------------------------------------------------------
# manifest admission
# --------------------------------------------------------------------------
def test_admission_model_key_and_record_bytes(spec, manifest):
    assert manifest.model_key == spec.key == KEY
    # every record carries the pinned affine Q2 geometry
    for record in manifest.records[:: len(manifest.records) // 12 or 1]:
        assert record.logical_bytes == RECORD_BYTES == spec.expert_record_bytes
    assert manifest.routed_expert_bytes == spec.routed_expert_bytes


@pytest.mark.skipif(not BANK.is_file(), reason="expert bank not present")
def test_admission_sample_record_digests(manifest):
    # Read a handful of records from experts.bin (a few * ~10.5 MiB, not the
    # 158 GiB bank) and confirm their sha256 matches the manifest record hash.
    sample_indices = [0, 1, len(manifest.records) // 2, len(manifest.records) - 1]
    fd = os.open(BANK, os.O_RDONLY)
    try:
        for idx in sample_indices:
            record = manifest.records[idx]
            assert record.sha256 is not None
            digest = hashlib.sha256()
            for segment in record.segments:
                assert segment.shard == "experts.bin"
                digest.update(os.pread(fd, segment.length, segment.offset))
            assert digest.hexdigest() == record.sha256
    finally:
        os.close(fd)


@pytest.mark.skipif(not BANK.is_file(), reason="expert bank not present")
def test_real_admission_writes_receipt(spec, manifest, tmp_path, monkeypatch):
    # Drive the REAL admission code (admit_expert_artifact) against the artifact
    # with a trusted bank digest so the 169 GiB bank is never hashed.  It writes
    # a revision/digest-bound receipt into a scratch receipt_root outside the
    # artifact.  Uses the source-rebased spec (emulating the republished
    # manifest identity) because the shipped manifest's identity is pre-publish.
    import mtplx.expert_streaming_models as esm

    monkeypatch.setitem(esm.MODEL_SPECS, KEY, _rebased_spec(spec, manifest))
    st = os.stat(BANK)
    trusted = {
        manifest.sidecar.parts[0].file: TrustedFileDigest(
            sha256=manifest.sidecar.parts[0].sha256,
            st_dev=st.st_dev,
            st_ino=st.st_ino,
            st_size=st.st_size,
            st_mtime_ns=st.st_mtime_ns,
            st_ctime_ns=st.st_ctime_ns,
        )
    }
    receipt = admit_expert_artifact(
        ARTIFACT, receipt_root=tmp_path, trusted_bank_digests=trusted
    )
    assert receipt["manifest_sha256"] == manifest.manifest_sha256
    assert receipt["banks"][0]["file"] == "experts.bin"
    assert receipt["banks"][0]["sha256"] == manifest.sidecar.parts[0].sha256
    receipt_file = Path(receipt["receipt_path"])
    assert receipt_file.is_file()
    assert receipt_file.parent == tmp_path.resolve()


# --------------------------------------------------------------------------
# memory plan
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ctx", [4096, 16384, 65536])
def test_memory_plan_text_only_slots(spec, ctx):
    skip = SKIPPED_BYTES  # drop mtp + vision residents for text-only AR
    plan = plan_expert_memory(
        spec,
        total_limit_bytes=KNOB,
        context_tokens=ctx,
        runtime_reserve_bytes=RESERVE,
        additional_resident_bytes=WINDOW,
        resident_discount_bytes=skip,
    )
    assert plan.fits_fixed
    assert plan.slots_per_layer == 205
    assert plan.persistent_slots == 205 * spec.routed_layer_count
    # resident is the text-only backbone plus the SWA window
    assert plan.resident_bytes == (spec.resident_bytes - skip) + WINDOW == KEPT_BYTES + WINDOW
    # resident + reserve + kv + transient fits the knob
    assert plan.resident_bytes + RESERVE + plan.kv_bytes + plan.transient_bytes <= KNOB


def test_memory_plan_full_resident_is_conservative(spec):
    # Without the text-only skip (what ExpertStreamingRuntime.open currently
    # prices, reserving all 25.16 GB of residents), the bank still fits but buys
    # fewer slots -- the skip is worth +37 slots/layer.
    plan = plan_expert_memory(
        spec,
        total_limit_bytes=KNOB,
        context_tokens=4096,
        runtime_reserve_bytes=RESERVE,
        additional_resident_bytes=WINDOW,
    )
    assert plan.fits_fixed
    assert plan.slots_per_layer == 168
    assert 205 - plan.slots_per_layer == 37


# --------------------------------------------------------------------------
# loader end-to-end with the test double
# --------------------------------------------------------------------------
def test_open_runtime_and_bind_end_to_end(spec, manifest):
    runtime = loader.open_deepseek_v41_runtime(
        ARTIFACT,
        memory_limit_bytes=KNOB,
        max_live_kv_tokens=4096,
        runtime_reserve_bytes=RESERVE,
        spec=_rebased_spec(spec, manifest),
        expert_cache_limit_bytes=0,  # tiny host allocation for the CPU test
        admission_receipt=_stat_receipt(manifest),
        apply_memory_cap=False,  # never touch the MLX/Metal memory cap
    )
    try:
        assert runtime.spec.key == KEY
        assert runtime.plan.fits_fixed
        assert runtime.reader.backend in {"native", "preadv"}

        # Construct the test double via the loader's model-class resolver hook
        # and the config that ships in the artifact.
        from mlx_lm.utils import load_config

        config = load_config(ARTIFACT)
        assert config["model_type"] == "deepseek_v41"
        model_cls, args_cls = double_classes()
        model = model_cls(
            args_cls.from_dict(config),
            engram_bank_path=loader.engram_bank_path_for(ARTIFACT),
        )

        bound = bind_streamed_switches(model, runtime)
        assert bound == runtime.spec.routed_layer_count == 40
        # every routed layer's switch seam is now a streamed switch
        for i in runtime.spec.routed_layer_indices:
            assert type(model.model.layers[i].mlp.switch_mlp).__name__ == (
                "HotExpertSwitchGLU"
            )
        # engram bank path handed to the model constructor
        assert model.engram_bank_path == (ARTIFACT.resolve() / "engram")

        # text-only resident plan the loader would materialize
        part = loader.partition_text_residents(runtime.manifest)
        assert part.kept_count == KEPT_COUNT
        assert part.kept_bytes == KEPT_BYTES
    finally:
        runtime.close()


def test_construct_is_wired_and_guarded_until_w1(spec, manifest):
    # The production construct entry resolves the real W1 model classes by
    # default; until mtplx.models.deepseek_v41 lands it raises a clear,
    # actionable error rather than an opaque ImportError.
    runtime = loader.open_deepseek_v41_runtime(
        ARTIFACT,
        memory_limit_bytes=KNOB,
        max_live_kv_tokens=4096,
        runtime_reserve_bytes=RESERVE,
        spec=_rebased_spec(spec, manifest),
        expert_cache_limit_bytes=0,
        admission_receipt=_stat_receipt(manifest),
        apply_memory_cap=False,
    )
    try:
        with pytest.raises(loader.ResidentLoadError) as excinfo:
            loader.construct_deepseek_v41_resident_model(ARTIFACT, runtime)
        assert "mtplx.models.deepseek_v41" in str(excinfo.value)
    finally:
        runtime.close()


# --------------------------------------------------------------------------
# /health-relevant facts (CPU-level equivalent; a full CLI /health needs
# Metal + W1's model, so it is not run here)
# --------------------------------------------------------------------------
def test_health_relevant_facts(spec):
    # The serve path forces generation_mode "ar" for a streamed artifact and
    # reports the model key; this artifact is AR-only (no MTP carried), and the
    # loader/runtime surface the model key the /health payload uses.
    assert spec.key == KEY
    assert spec.mtp_included is False
    assert spec.mtp_layer_index is None
