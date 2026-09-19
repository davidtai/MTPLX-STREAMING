"""Protect target defaults against measured 16K live-byte envelopes; no MLX.

The uncached calibration and source 607995471 Python coding receipts used the
same resident/kernel geometry with 13 and 27 expert slots per layer respectively.
Only the persistent expert allocation changes in this projection. We add the
FULL allocator cache and Python capacity instead of treating a sampled OS peak
or MLX's active-only peak as a physical upper bound. Future shapes/implementations
still require guarded measurement; this locks the observed regression.
"""
from __future__ import annotations

import pytest

from mtplx import expert_runtime as runtime
from mtplx.expert_streaming_models import get_model_spec

GIB = 1024**3
GB = 1_000_000_000
RESIDENT_BYTES = 15_103_104_448
FIXED_BYTES = 23_521_727_936
RECORD_BYTES = 18_800_640
LAYERS = 40
MAX_KV = 17_664

# calibration-resident-uncached.jsonl and python-16k-1024-band16.jsonl;
# source measurements include all prefill/decode active allocation peaks.
MEASURED_ENVELOPES = (
    pytest.param("46.7479", 13, 40_300_779_044, id="16k-32-calibration"),
    pytest.param("46.7607", 27, 50_834_415_512, id="16k-1024-python"),
)


@pytest.mark.parametrize("baseline_gb,measured_slots,active_peak", MEASURED_ENVELOPES)
@pytest.mark.parametrize("session_gib", (0, 2), ids=("benchmark", "serving-bank"))
def test_default_slot_capacity_leaves_measured_peak_and_full_caches_under_110gb(
    baseline_gb, measured_slots, active_peak, session_gib
):
    budget = runtime.resolve_box_target_mlx_limit_bytes(
        {runtime.BOX_TARGET_ENV: "default", runtime.BOX_BASELINE_ENV: baseline_gb,
         runtime.BOX_SESSION_BANK_ENV: str(session_gib)},
        memsize_bytes=128 * GIB,
    )
    spec = get_model_spec("deepseek-v41-flash-expert-mxfp4")
    # Use the real uniform-slot planner with the calibrated effective residents
    # (text-only discount plus installed wo_a cache), including its slot rounding.
    config = runtime.ExpertStreamingConfig(
        model_key=spec.key, memory_limit_bytes=budget["engine_budget_bytes"],
        max_live_kv_tokens=MAX_KV, slot_layout="component-banks",
        runtime_reserve_bytes=7 * GIB, transient_slots=48,
    )
    plan = config.memory_plan(
        spec,
        resident_discount_bytes=spec.resident_bytes - RESIDENT_BYTES,
    )
    # The calibrated active peak already includes real KV. The fixed-storage
    # admission correction now reserves its configured maximum before sizing slots.
    assert plan.fixed_bytes == FIXED_BYTES + MAX_KV * spec.kv_bytes_per_token
    assert spec.expert_record_bytes == RECORD_BYTES
    assert spec.routed_layer_count == LAYERS
    measured_persistent_bytes = measured_slots * LAYERS * RECORD_BYTES
    slot_delta = plan.persistent_cache_bytes - measured_persistent_bytes
    projected_active_peak = active_peak + slot_delta
    envelope = (
        budget["box_baseline_bytes"] + budget["host_overhead_bytes"]
        + projected_active_peak + budget["allocator_cache_limit_bytes"]
        + budget["session_bank_reserve_bytes"]
    )
    assert envelope <= 110 * GB, (
        f"defaults admit {plan.slots_per_layer} slots/layer but the measured active "
        f"envelope plus full caches and baseline needs {envelope:,} bytes"
    )
