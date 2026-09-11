"""W36 regression: the loader's component-bank allocator plan must agree with the
slot pool plan ``ExpertStreamingRuntime.open`` builds.

``open_deepseek_v41_runtime`` is a *second* runtime-open entry (the P1.7
streamed==resident gate and the CPU end-to-end proof, plus the W24/W28 decode-lever
and bench scripts). It wires its own component-bank allocator through
``_component_bank_allocator_for``. That allocator sizes each per-layer bank's
capacity from its own memory plan, while ``ExpertStreamingRuntime.open`` enumerates
the persistent slot indices (``range(pool_plan.slots_per_layer)``) it hands the
allocator. If the two plans disagree on ``slots_per_layer``, the pool asks the
allocator for a persistent slot index the bank cannot hold and the load dies with
``ValueError('persistent slot is outside planned capacity')`` (issue W36 —
window-11 ab_decode_levers.py failed on every arm, control included, at 82 GiB).

W21 (effe73663) added ``text_only_resident_discount`` (the 8.31 GiB of MTP + vision
residents a phase-1 text-only AR forward never wires) to open()'s pool plan and to
runtime.py's production pre-flight allocator, but NOT to this loader's allocator.
The two plans then drifted apart by +11..+12 slots/layer for the shipped
DeepSeek-V4.1-Flash mxfp4 artifact at the 72/82/92 GiB envelopes. bench_standard_shape.py
did not hit it only because its 72/92 GiB runs predate that commit; any
component-banks load through ``open_deepseek_v41_runtime`` since W21 fails the same way.

CPU only, no GPU. Reads ONLY the real artifact's expert-manifest.json (metadata);
never opens experts.bin and never allocates a bank, so RSS stays tiny.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.cpu)

import pytest

from mtplx.expert_manifest import load_expert_manifest
from mtplx.expert_runtime import (
    proj_quant_plan_discount,
    proj_requant_plan_discount,
    resolve_island_placement,
    text_only_resident_discount,
)
from mtplx.expert_streaming_models import get_model_spec
from mtplx.models.deepseek_v41_loader import (
    DEFAULT_RUNTIME_RESERVE_BYTES,
    SWA_WINDOW_BYTES,
    _component_bank_allocator_for,
    build_streaming_config,
)

GIB = 1024**3
MXFP4 = "deepseek-v41-flash-expert-mxfp4"
ARTIFACT = Path(
    os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4")
)
_MANIFEST = ARTIFACT / "expert-manifest.json"

pytestmark = pytest.mark.skipif(
    not _MANIFEST.exists(),
    reason=f"real DSV4.1-Flash mxfp4 manifest not present at {_MANIFEST}",
)


def _cfg(limit_gib: float, max_kv: int = 4096):
    # The exact overrides open_deepseek_v41_runtime / the W24 + bench scripts pass.
    return build_streaming_config(
        get_model_spec(MXFP4),
        memory_limit_bytes=int(limit_gib * GIB),
        max_live_kv_tokens=int(max_kv),
        runtime_reserve_bytes=DEFAULT_RUNTIME_RESERVE_BYTES,
        expert_cache_limit_bytes=None,
        cache_scope="layer",
        slot_layout="component-banks",
        island_layers=(),
        verify_record_hashes=False,
    )


def _pool_plan(cfg, spec, manifest):
    """The plan ExpertStreamingRuntime.open builds for its slot pool (line-for-line
    the resident_discount_bytes expression at expert_runtime.py open())."""
    resolved = resolve_island_placement(cfg, ARTIFACT, spec=spec)
    return resolved.memory_plan(
        spec,
        additional_resident_bytes=SWA_WINDOW_BYTES,
        resident_discount_bytes=(
            proj_quant_plan_discount(manifest, resolved.proj_quant)
            + proj_requant_plan_discount(manifest, resolved.proj_requant)
            + text_only_resident_discount(manifest, spec)
        ),
        layer_record_bytes=(
            manifest.record_bytes_by_layer() if spec.is_mixed_official else None
        ),
    )


def test_text_only_discount_is_material_for_this_artifact():
    """Guard the premise: the shipped mxfp4 artifact really carries MTP/vision
    residents the text-only discount removes (else the regression proves nothing)."""
    manifest = load_expert_manifest(_MANIFEST)
    spec = get_model_spec(MXFP4)
    discount = text_only_resident_discount(manifest, spec)
    assert discount > 6 * GIB, discount  # ~8.31 GiB of MTP + vision residents


@pytest.mark.parametrize("limit_gib", [72.0, 82.0, 92.0])
def test_loader_allocator_plan_agrees_with_pool_plan(limit_gib):
    """The loader's component-bank allocator plan and open()'s pool plan must agree
    on slots_per_layer / persistent_slots. Before the W36 fix the allocator omitted
    text_only_resident_discount and was ~11 slots/layer short (82 vs 93 at 82 GiB),
    which crashed the load; here they must be byte-for-byte equal."""
    spec = get_model_spec(MXFP4)
    manifest = load_expert_manifest(_MANIFEST)
    cfg = _cfg(limit_gib)

    allocator = _component_bank_allocator_for(cfg, spec, ARTIFACT, _MANIFEST, manifest)
    assert allocator is not None  # component-banks layout wires the allocator
    alloc_plan = allocator.plan

    pool_plan = _pool_plan(cfg, spec, manifest)

    assert alloc_plan.slots_per_layer == pool_plan.slots_per_layer, (
        f"@{limit_gib} GiB allocator {alloc_plan.slots_per_layer} slots/layer != "
        f"pool {pool_plan.slots_per_layer}; the pool enumerates persistent slot "
        f"index {alloc_plan.slots_per_layer} which the bank cannot hold"
    )
    assert alloc_plan.persistent_slots == pool_plan.persistent_slots
    assert alloc_plan.persistent_cache_bytes == pool_plan.persistent_cache_bytes
    # And the discount really is applied to the allocator's fixed side.
    expected_resident = (
        spec.resident_bytes
        + SWA_WINDOW_BYTES
        - proj_quant_plan_discount(manifest, cfg.proj_quant)
        - proj_requant_plan_discount(manifest, cfg.proj_requant)
        - text_only_resident_discount(manifest, spec)
    )
    assert alloc_plan.resident_bytes == expected_resident
