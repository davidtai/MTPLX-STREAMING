"""W95 -- the single ``MTPLX_DSV41_RUNNER=v2`` runner switch.

v2 is the ONE user switch for the rebuilt streaming decode path
(docs/deepseek-v41/W95_RUNNER_DESIGN.md).  Its first-deliverable scope (post
window-38, which withdrew the host-sync-drain premise) is to HIDE THE SSD MISS
WAITS by composing, without the user stacking their individual env keys:

  * the W93 gate-oracle one-layer-ahead prefetch (arms the global ring at
    ``k = _RUNNER_V2_GATE_PREFETCH_K`` when no explicit ``MTPLX_DSV41_GATE_PREFETCH``),
  * the W87 single scan-resistant slot pool (admit every miss -- no first-seen
    rejection into the discard-after-one-use transient tier).

These tests lock:

  A. **Arming (config).** ``MTPLX_DSV41_RUNNER=v2`` alone arms the prefetch width /
     global ring; an explicit ``MTPLX_DSV41_GATE_PREFETCH`` always wins; unset is
     byte-identical off (k=0, ring unchanged).
  B. **Arming (runtime open).** A runtime opened under v2 turns the single slot
     pool on (``_single_slot_pool``) and builds the prefetch ring, without the
     user setting ``MTPLX_DSV41_SINGLE_SLOT_POOL``.
  C. **Exactness.** The composed v2 residency policy (single pool + a warmed
     prefetch ring) changes NO gathered value -- a switch route that streams a
     miss is byte-identical under v2 vs the shipped two-tier path, at M=1 (AR)
     and M=4 (the DSpark verify row batch).  Residency never changes the math.

CPU-pinned; tiny synthetic artifact (``_integrated_hy3_artifact``); no GPU, no
real weights.  Run under ``nice -n 19`` and without ``pytest -n auto``.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.cpu)

import pytest  # noqa: E402

import mtplx.models.deepseek_v41 as dv  # noqa: E402
from mtplx.expert_manifest import load_expert_manifest  # noqa: E402
from mtplx.expert_runtime import (  # noqa: E402
    ExpertStreamingConfig,
    ExpertStreamingRuntime,
)
from mtplx.expert_streaming_models import get_model_spec  # noqa: E402
from mtplx.models.deepseek_v41 import (  # noqa: E402
    _resolve_gate_prefetch_k,
    _runner_v2_enabled,
    _RUNNER_V2_GATE_PREFETCH_K,
)
from mtplx.models.deepseek_v41_loader import (  # noqa: E402
    build_streaming_config,
    resolve_gate_prefetch_ring_slots,
)
from mtplx.models.expert_mlx import (  # noqa: E402
    HotExpertSwitchGLU,
    make_mlx_component_bank_allocator,
)

_REAL_EVAL = mx.eval
_KEY = "deepseek-v41-flash-expert-mxfp4"
_MEM = 80 * 1024**3
_RUNNER_KEYS = (
    "MTPLX_DSV41_RUNNER",
    "MTPLX_DSV41_GATE_PREFETCH",
    "MTPLX_DSV41_GATE_PREFETCH_MIN_LAYER",
    "MTPLX_DSV41_SINGLE_SLOT_POOL",
)


@pytest.fixture(autouse=True)
def _clean_env():
    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    saved = {k: os.environ.get(k) for k in _RUNNER_KEYS}
    for k in _RUNNER_KEYS:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        mx.set_default_device(prev_dev)


# ---------------------------------------------------------------------------
# A. arming (config level, no artifact)
# ---------------------------------------------------------------------------
def test_runner_v2_arms_gate_prefetch_width():
    assert _runner_v2_enabled() is False
    assert _resolve_gate_prefetch_k() == 0  # off -> byte-identical
    os.environ["MTPLX_DSV41_RUNNER"] = "v2"
    assert _runner_v2_enabled() is True
    assert _resolve_gate_prefetch_k() == _RUNNER_V2_GATE_PREFETCH_K
    # the global ring is sized 2*k over a profile-seeded 0
    assert resolve_gate_prefetch_ring_slots(0) == 2 * _RUNNER_V2_GATE_PREFETCH_K


def test_explicit_gate_prefetch_wins_over_v2_default():
    os.environ["MTPLX_DSV41_RUNNER"] = "v2"
    os.environ["MTPLX_DSV41_GATE_PREFETCH"] = "8"
    assert _resolve_gate_prefetch_k() == 8  # explicit sub-key wins
    assert resolve_gate_prefetch_ring_slots(0) == 16
    # an explicit raw arg is respected as-is (never overridden by the v2 default)
    assert _resolve_gate_prefetch_k("0") == 0


def test_runner_v2_arms_ring_in_loader_config():
    spec = get_model_spec(_KEY)
    # off -> shipped profile byte-identical (seeded 0 stays 0)
    cfg_off = build_streaming_config(
        spec, memory_limit_bytes=_MEM, max_live_kv_tokens=4096, prefetch_slots=0
    )
    assert cfg_off.prefetch_slots == 0
    # v2 arms the ring over the profile-seeded 0
    os.environ["MTPLX_DSV41_RUNNER"] = "v2"
    cfg_v2 = build_streaming_config(
        spec, memory_limit_bytes=_MEM, max_live_kv_tokens=4096, prefetch_slots=0
    )
    assert cfg_v2.prefetch_slots == 2 * _RUNNER_V2_GATE_PREFETCH_K
    # a larger explicit caller value is never shrunk
    cfg_big = build_streaming_config(
        spec, memory_limit_bytes=_MEM, max_live_kv_tokens=4096, prefetch_slots=40
    )
    assert cfg_big.prefetch_slots == 40


def test_runner_v2_arms_overlap_miss_reads():
    """v2 issues all of a layer's demand misses as ONE part (higher SSD queue
    depth than the per-expert default, W96 D2).  Off -> default False; explicit
    caller value wins.  Byte-identical (scheduling, tests/test_expert_overlap_split)."""
    spec = get_model_spec(_KEY)
    cfg_off = build_streaming_config(
        spec, memory_limit_bytes=_MEM, max_live_kv_tokens=4096, prefetch_slots=0
    )
    assert cfg_off.overlap_miss_reads is False
    os.environ["MTPLX_DSV41_RUNNER"] = "v2"
    cfg_v2 = build_streaming_config(
        spec, memory_limit_bytes=_MEM, max_live_kv_tokens=4096, prefetch_slots=0
    )
    assert cfg_v2.overlap_miss_reads is True
    cfg_explicit = build_streaming_config(
        spec, memory_limit_bytes=_MEM, max_live_kv_tokens=4096, prefetch_slots=0,
        overlap_miss_reads=False,
    )
    assert cfg_explicit.overlap_miss_reads is False


# ---------------------------------------------------------------------------
# B/C. runtime arming + exactness (tiny real component-bank runtime)
# ---------------------------------------------------------------------------
def _open_runtime(tmp_path, *, runner_v2, expert_count=8, top_k=2,
                  resident_slots=2, transient=8, prefetch=10):
    """Open a tiny real streamed runtime.  ``runner_v2`` sets the env BEFORE open
    so ``ExpertStreamingRuntime.open`` reads it for the single-pool gate."""
    from tests.test_streamed_models import _integrated_hy3_artifact

    if runner_v2:
        os.environ["MTPLX_DSV41_RUNNER"] = "v2"
    else:
        os.environ.pop("MTPLX_DSV41_RUNNER", None)
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    root, config, spec, manifest_path = _integrated_hy3_artifact(
        tmp_path, expert_count=expert_count, top_k=top_k
    )
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    ring_pad = (transient + prefetch + 16) * spec.expert_record_bytes
    sc = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(resident_slots) + ring_pad,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        transient_slots=transient,
        slot_layout="component-banks",
        prefetch_slots=prefetch,
        resource_telemetry=True,
    )
    plan = sc.memory_plan(spec)
    rt = ExpertStreamingRuntime.open(
        root, manifest_path, sc, spec=spec,
        buffer_allocator=make_mlx_component_bank_allocator(
            plan, spec, load_expert_manifest(manifest_path)
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    return rt, spec


def _switch(rt, spec):
    return HotExpertSwitchGLU(rt, spec.routed_layer_start)


def _route_once(rt, spec, experts):
    sw = _switch(rt, spec)
    for e in experts:
        idx = mx.array([[e] * spec.top_k], dtype=mx.int32)
        x = mx.zeros((1, 1, spec.hidden_size), dtype=mx.bfloat16)
        _REAL_EVAL(sw(x, idx))
    rt.flush_deferred_slot_releases(evaluate=True)


def _inputs(rows, top_k, hidden, experts):
    mx.random.seed(95 + rows)
    x = (0.3 * mx.random.normal((rows, 1, hidden))).astype(mx.bfloat16)
    flat = [experts[i % len(experts)] for i in range(rows * top_k)]
    idx = mx.array(flat, dtype=mx.int32).reshape((rows, top_k))
    _REAL_EVAL(x, idx)
    return x, idx


def test_runner_v2_open_arms_single_pool(tmp_path):
    rt_off, _ = _open_runtime(tmp_path / "off", runner_v2=False)
    try:
        assert rt_off._single_slot_pool is False
    finally:
        rt_off.close()
    rt_v2, _ = _open_runtime(tmp_path / "v2", runner_v2=True)
    try:
        assert rt_v2._single_slot_pool is True
        assert rt_v2.config.prefetch_slots > 0  # ring built
    finally:
        rt_v2.close()


@pytest.mark.parametrize("rows", [1, 4])
def test_runner_v2_switch_byte_identical_to_shipped(tmp_path, rows):
    """A route that streams a miss is byte-identical under v2 (single pool +
    prefetch ring) vs the shipped two-tier path.  Residency never changes math."""
    warm = [0, 1]        # resident_slots=2 -> only these two stay persistent
    miss = [6, 7]        # never resident -> streamed on the route

    rt_off, spec = _open_runtime(tmp_path / "off", runner_v2=False)
    try:
        _route_once(rt_off, spec, warm)
        x, idx = _inputs(rows, spec.top_k, spec.hidden_size, miss)
        out_off = _switch(rt_off, spec)(x, idx)
        _REAL_EVAL(out_off)
        rt_off.flush_deferred_slot_releases(evaluate=True)
    finally:
        rt_off.close()

    rt_v2, spec2 = _open_runtime(tmp_path / "v2", runner_v2=True)
    try:
        # warm the ring with the exact experts the true route needs, so v2 serves
        # them from the ring (single pool + committed prefetch), not two-tier demand
        _route_once(rt_v2, spec2, warm)
        rt_v2.prefetch_experts(spec2.routed_layer_start, miss)
        x2, idx2 = _inputs(rows, spec2.top_k, spec2.hidden_size, miss)
        out_v2 = _switch(rt_v2, spec2)(x2, idx2)
        _REAL_EVAL(out_v2)
        rt_v2.flush_deferred_slot_releases(evaluate=True)
    finally:
        rt_v2.close()

    assert mx.array_equal(out_off, out_v2), (
        f"M={rows}: v2 residency policy changed the gathered value"
    )
