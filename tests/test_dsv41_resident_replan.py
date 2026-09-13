"""Effective resident pricing survives static KV admission; no MLX allocation."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.expert_runtime import ExpertStreamingConfig, ExpertStreamingRuntime
from mtplx.expert_streaming_models import (
    MIXED_OFFICIAL_CODEC,
    ExpertStreamingModelSpec,
    get_model_spec,
)


def _tiny_spec(mixed):
    record_bytes = {0: 6912, 1: 3456 if mixed else 6912}
    spec = ExpertStreamingModelSpec(
        key="tiny-resident-replan", display_name="resident replan fixture",
        source_model="fixture", source_revision="fixture",
        quant_model="fixture", quant_revision="fixture",
        total_tensor_bytes=4096 + 4 * sum(record_bytes.values()),
        total_layers=2, routed_layer_start=0, routed_layer_count=2,
        expert_count=4, top_k=1, hidden_size=64, expert_hidden_size=64,
        quant_bits=4, quant_group_size=64, quant_parameter_bytes=2,
        router_storage="bfloat16", router_matmul_dtype="float32", router_bytes=0,
        kv_bytes_per_token=16, mtp_layer_index=None, mtp_included=False,
        expert_codec=MIXED_OFFICIAL_CODEC if mixed else "affine",
    )
    return spec, record_bytes if mixed else None


def _runtime(spec, layer_bytes, adjustment, *, memory_bytes=1024**2):
    config = ExpertStreamingConfig(
        model_key=spec.key, memory_limit_bytes=memory_bytes,
        max_live_kv_tokens=128, runtime_reserve_bytes=0,
        transient_slots=spec.top_k, slot_layout="component-banks",
    )
    plan = config.memory_plan(
        spec, additional_resident_bytes=max(0, adjustment),
        resident_discount_bytes=max(0, -adjustment), layer_record_bytes=layer_bytes,
    )
    # The real constructor allocates policy metadata, not tensor buffers. No
    # route is executed; the fake slot pool supplies only its health check.
    runtime = ExpertStreamingRuntime(
        Path("."), spec, config,
        SimpleNamespace(record_bytes_by_layer=lambda: layer_bytes), plan,
        reader=None, slots=SimpleNamespace(raise_if_unhealthy=lambda: None),
        single_slot_pool=True,
    )
    return runtime, plan


@pytest.mark.parametrize("mixed", (False, True), ids=("uniform", "mixed"))
@pytest.mark.parametrize("adjustment", (-1024, 0, 1024), ids=("discount", "unchanged", "extra"))
def test_kv_admission_and_release_preserve_effective_residents(mixed, adjustment):
    spec, layer_bytes = _tiny_spec(mixed)
    runtime, initial = _runtime(spec, layer_bytes, adjustment)
    try:
        admission = runtime.admit_kv_tokens(8)
        assert runtime._live_kv_tokens == 8
        assert runtime.plan is initial
        assert initial.resident_bytes == 4096 + adjustment
        assert initial.kv_bytes == 128 * spec.kv_bytes_per_token
        assert not runtime._derived_cache_policy
        admission.release()
        assert runtime._live_kv_tokens == 0
        assert runtime.plan is initial
    finally:
        runtime._split_executor.shutdown(wait=True)


def test_measured_deepseek_text_residents_support_static_admission():
    spec = get_model_spec("deepseek-v41-flash-expert-mxfp4")
    effective_residents = 15_103_104_448
    runtime, initial = _runtime(
        spec, None, effective_residents - spec.resident_bytes,
        memory_bytes=50 * 1024**3,
    )
    try:
        assert initial.resident_bytes == effective_residents
        with runtime.admit_kv_tokens(128):
            assert runtime.plan is initial
            assert initial.kv_bytes == 128 * spec.kv_bytes_per_token
        assert runtime.plan.resident_bytes == effective_residents
    finally:
        runtime._split_executor.shutdown(wait=True)
