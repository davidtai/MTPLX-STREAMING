"""W93 Lane A: gate-oracle prefetch CONFIG allocation + plan arithmetic.

Locks the review's CRITICAL finding: when MTPLX_DSV41_GATE_PREFETCH is armed the
env is AUTHORITATIVE on BOTH config paths (the ab-bench loader and the served
profile), so a profile-seeded prefetch_slots=0 cannot leave the lever measuring
control-vs-control. Also pins the GLOBAL-ring plan arithmetic at the profile
numbers. CPU-only; no artifact; run under ``nice -n 19`` without ``-n auto``.
"""

from __future__ import annotations

import os
import types

import mlx.core as mx

mx.set_default_device(mx.cpu)

import pytest  # noqa: E402

from mtplx.expert_runtime import ExpertStreamingConfig  # noqa: E402
from mtplx.expert_streaming_models import get_model_spec  # noqa: E402
from mtplx.models.deepseek_v41_loader import (  # noqa: E402
    build_streaming_config,
    resolve_gate_prefetch_ring_slots,
)
from mtplx.expert_profiles import build_expert_streaming_config  # noqa: E402

_KEY = "deepseek-v41-flash-expert-mxfp4"
_MEM = 80 * 1024**3


@pytest.fixture(autouse=True)
def _clean_env():
    saved = {
        k: os.environ.get(k)
        for k in ("MTPLX_DSV41_GATE_PREFETCH", "MTPLX_DSV41_GATE_PREFETCH_MIN_LAYER")
    }
    for k in saved:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# CRITICAL: env authoritative on the resolver (a seeded 0 must not win)
# ---------------------------------------------------------------------------
def test_resolver_env_authoritative_over_seeded_zero():
    assert resolve_gate_prefetch_ring_slots(0) == 0  # off -> unchanged
    assert resolve_gate_prefetch_ring_slots(8) == 8
    os.environ["MTPLX_DSV41_GATE_PREFETCH"] = "10"
    assert resolve_gate_prefetch_ring_slots(0) == 20  # seeded 0 does NOT win
    assert resolve_gate_prefetch_ring_slots(8) == 20  # never shrinks below 2*k
    assert resolve_gate_prefetch_ring_slots(24) == 24  # keeps a larger caller value
    os.environ["MTPLX_DSV41_GATE_PREFETCH"] = "12"
    assert resolve_gate_prefetch_ring_slots(0) == 24
    os.environ["MTPLX_DSV41_GATE_PREFETCH"] = "20"
    assert resolve_gate_prefetch_ring_slots(0) == 32  # capped at 32


# ---------------------------------------------------------------------------
# CRITICAL: the ab-bench loader path arms the ring over a profile-seeded 0
# ---------------------------------------------------------------------------
def test_loader_config_arms_ring_over_profile_zero():
    spec = get_model_spec(_KEY)
    os.environ["MTPLX_DSV41_GATE_PREFETCH"] = "10"
    # the ab bench / profile seeds an explicit prefetch_slots=0
    cfg = build_streaming_config(
        spec, memory_limit_bytes=_MEM, max_live_kv_tokens=4096, prefetch_slots=0
    )
    assert cfg.prefetch_slots == 20, "env must arm the ring over a seeded 0"
    # a larger explicit caller value is respected (never shrunk).
    cfg_big = build_streaming_config(
        spec, memory_limit_bytes=_MEM, max_live_kv_tokens=4096, prefetch_slots=28
    )
    assert cfg_big.prefetch_slots == 28
    # off -> stays 0 (shipped profile byte-identical)
    os.environ.pop("MTPLX_DSV41_GATE_PREFETCH")
    cfg0 = build_streaming_config(
        spec, memory_limit_bytes=_MEM, max_live_kv_tokens=4096, prefetch_slots=0
    )
    assert cfg0.prefetch_slots == 0


# ---------------------------------------------------------------------------
# CRITICAL: the served path arms the ring, gated to DeepSeek-V4.1 profiles
# ---------------------------------------------------------------------------
def _fake_profile(model_key, mem):
    return types.SimpleNamespace(
        model_key=model_key,
        config={
            "memory_limit_bytes": mem,
            "max_live_kv_tokens": 4096,
            "prefetch_slots": 0,
        },
        process_ceiling_bytes=110 * 1024**3,
        island_layer_count=None,
    )


def test_served_config_arms_ring():
    os.environ["MTPLX_DSV41_GATE_PREFETCH"] = "10"
    cfg = build_expert_streaming_config(_fake_profile(_KEY, _MEM))
    assert cfg.prefetch_slots == 20
    os.environ.pop("MTPLX_DSV41_GATE_PREFETCH")
    assert build_expert_streaming_config(_fake_profile(_KEY, _MEM)).prefetch_slots == 0


def test_served_env_does_not_arm_other_models():
    # the DSV4.1-named env must NOT arm another model's ring (hy3 shares the knob).
    os.environ["MTPLX_DSV41_GATE_PREFETCH"] = "10"
    cfg = build_expert_streaming_config(_fake_profile("hy3-expert-q2", 40 * 1024**3))
    assert cfg.prefetch_slots == 0


# ---------------------------------------------------------------------------
# plan arithmetic at the profile numbers: the ring is GLOBAL (records TOTAL)
# ---------------------------------------------------------------------------
def test_plan_prefetch_bytes_are_global_not_per_layer():
    spec = get_model_spec(_KEY)
    config = ExpertStreamingConfig(
        model_key=_KEY, memory_limit_bytes=_MEM, max_live_kv_tokens=4096,
        prefetch_slots=20,
    )
    plan = config.memory_plan(spec)
    assert plan.prefetch_ring_slots == 20
    # GLOBAL: 20 records TOTAL, NOT 20 * routed_layer_count.
    assert plan.prefetch_bytes == 20 * spec.expert_record_bytes
    assert plan.prefetch_bytes < spec.routed_layer_count * spec.expert_record_bytes
    baseline = ExpertStreamingConfig(
        model_key=_KEY, memory_limit_bytes=_MEM, max_live_kv_tokens=4096,
    ).memory_plan(spec)
    assert baseline.prefetch_ring_slots == 0 and baseline.prefetch_bytes == 0


def test_ring_comes_out_of_persistent_budget_by_at_most_one_slot():
    # review MEDIUM-d: the 0.36 GiB ring is NOT free reserve -- it reduces the
    # persistent budget. slots_per_layer stays unchanged ONLY when the floor-
    # division slack absorbs it (the shipped -75 profile: 49 unchanged); on a
    # floor boundary it can drop by exactly one slot/layer. The robust invariant
    # is: the whole global ring costs the LRU AT MOST one slot per layer (vs the
    # per-layer ring, which would cost ~2*k). The receipt prints both values.
    spec = get_model_spec(_KEY)
    no_ring = ExpertStreamingConfig(
        model_key=_KEY, memory_limit_bytes=_MEM, max_live_kv_tokens=4096,
    ).memory_plan(spec)
    with_ring = ExpertStreamingConfig(
        model_key=_KEY, memory_limit_bytes=_MEM, max_live_kv_tokens=4096,
        prefetch_slots=20,
    ).memory_plan(spec)
    assert with_ring.persistent_cache_bytes <= no_ring.persistent_cache_bytes
    delta = no_ring.slots_per_layer - with_ring.slots_per_layer
    assert 0 <= delta <= 1, f"global ring cost the LRU {delta} slots/layer (expect <=1)"
