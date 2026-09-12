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
import numpy as np

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
    _resolve_gate_prefetch_margin,
    _runner_v2_enabled,
    _RUNNER_V2_GATE_PREFETCH_K,
    _RUNNER_V2_GATE_PREFETCH_MARGIN,
    _RUNNER_V2_RING_SLOTS,
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
    "MTPLX_DSV41_GATE_PREFETCH_MARGIN",
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
    assert _resolve_gate_prefetch_k() == _RUNNER_V2_GATE_PREFETCH_K  # 6 (retuned)
    # v2 sizes the global ring for the DSpark verify union (48), which also covers
    # the AR k=6 -- not merely 2*k_AR.
    assert resolve_gate_prefetch_ring_slots(0) == _RUNNER_V2_RING_SLOTS


def test_explicit_gate_prefetch_wins_over_v2_default():
    os.environ["MTPLX_DSV41_RUNNER"] = "v2"
    os.environ["MTPLX_DSV41_GATE_PREFETCH"] = "8"
    assert _resolve_gate_prefetch_k() == 8  # explicit AR width wins
    # the ring is still v2-sized for the verify union regardless of the AR width
    assert resolve_gate_prefetch_ring_slots(0) == _RUNNER_V2_RING_SLOTS
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
    assert cfg_v2.prefetch_slots == _RUNNER_V2_RING_SLOTS  # verify-union sized
    # a larger explicit caller value is never shrunk
    cfg_big = build_streaming_config(
        spec, memory_limit_bytes=_MEM, max_live_kv_tokens=4096, prefetch_slots=64
    )
    assert cfg_big.prefetch_slots == 64


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
                  resident_slots=2, transient=8, prefetch=10, overlap=None):
    """Open a tiny real streamed runtime.  ``runner_v2`` sets the env BEFORE open
    so ``ExpertStreamingRuntime.open`` reads it for the single-pool gate.
    ``overlap`` (default = ``runner_v2``) sets the config's overlap_miss_reads --
    the v2 composition is single pool + prefetch ring + overlap_miss_reads."""
    from tests.test_streamed_models import _integrated_hy3_artifact

    if runner_v2:
        os.environ["MTPLX_DSV41_RUNNER"] = "v2"
    else:
        os.environ.pop("MTPLX_DSV41_RUNNER", None)
    if overlap is None:
        overlap = bool(runner_v2)
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
        overlap_miss_reads=overlap,
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


def _inputs(rows, top_k, hidden, experts, step=0):
    mx.random.seed(95 + rows + 1000 * step)
    x = (0.3 * mx.random.normal((rows, 1, hidden))).astype(mx.bfloat16)
    flat = [experts[i % len(experts)] for i in range(rows * top_k)]
    idx = mx.array(flat, dtype=mx.int32).reshape((rows, top_k))
    _REAL_EVAL(x, idx)
    return x, idx


def _settle_prefetch(rt):
    """Wait out every in-flight speculative read, deterministically."""
    lock = getattr(rt, "_prefetch_lock", None)
    if lock is None:
        return
    with lock:
        pending = tuple(getattr(rt, "_prefetch_futures", ()) or ())
    for future in pending:
        try:
            future.result()
        except BaseException:
            pass


def test_runner_v2_receipt_block(tmp_path):
    """The snapshot carries a rolled-up ``runner`` block when v2 is armed (mode,
    the composed knobs, and the SSD-hiding counters the paired window reads);
    absent -- shipped snapshot byte-unchanged -- when off."""
    rt_off, _ = _open_runtime(tmp_path / "off", runner_v2=False, prefetch=0)
    try:
        assert "runner" not in rt_off.resource_telemetry_snapshot(mx_module=mx)
    finally:
        rt_off.close()
    rt_v2, spec = _open_runtime(
        tmp_path / "v2", runner_v2=True, resident_slots=2, transient=8, prefetch=6
    )
    try:
        _route_once(rt_v2, spec, [0, 1])           # warm two persistent residents
        rt_v2.prefetch_experts(spec.routed_layer_start, [6, 7])  # ring-warm two misses
        _settle_prefetch(rt_v2)
        x, idx = _inputs(1, spec.top_k, spec.hidden_size, [6, 7])
        _REAL_EVAL(_switch(rt_v2, spec)(x, idx))    # stream the misses (ring hits)
        rt_v2.flush_deferred_slot_releases(evaluate=True)
        block = rt_v2.resource_telemetry_snapshot(mx_module=mx)["runner"]
        assert block["mode"] == "v2"
        assert block["single_pool"] is True
        assert block["overlap_miss_reads"] is True
        assert block["ring_slots"] > 0
        for key in (
            "prefetch_k", "prefetch_margin", "ring_slots", "byte_budget",
            "budget_skips", "expert_misses", "bytes_read", "hit_rate",
            "demand_bytes_read", "speculative_bytes_read", "prefetch_issued",
            "prefetch_hit_on_true_route", "prefetch_wasted", "pool_loads",
            "pool_promotions",
        ):
            assert key in block, f"runner block missing {key}"
        assert block["expert_misses"] >= 1  # experts 6,7 were streamed
        assert block["prefetch_k"] == _RUNNER_V2_GATE_PREFETCH_K  # 6 (retuned)
        assert block["prefetch_margin"] == _RUNNER_V2_GATE_PREFETCH_MARGIN  # -0.05
        assert block["byte_budget"] == 0.5  # v2 default demand-priority budget
    finally:
        rt_v2.close()


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


def _route_seq(n_steps, n_experts, top_k, seed=95):
    """A deterministic decode-like route sequence: mostly M=1 (AR) with periodic
    M=4 / M=6 (the DSpark depth-3 / depth-5 verify row batches), over a rotating
    expert set that churns a small resident pool (misses + evictions + re-hits)."""
    rng = np.random.RandomState(seed)
    seq = []
    for i in range(n_steps):
        if i % 37 == 36:
            rows = 6            # DSpark depth-5 verify shape (K+1 = 6 rows)
        elif i % 17 == 16:
            rows = 4            # DSpark depth-3 verify shape
        else:
            rows = 1            # AR decode
        experts = sorted(int(e) for e in rng.choice(n_experts, size=top_k, replace=False))
        seq.append((rows, experts))
    return seq


def test_runner_v2_256_step_byte_identical_streamed(tmp_path):
    """256 decode-like routes over a REAL streamed runtime: v2 (single pool +
    prefetch ring + overlap_miss_reads) is byte-identical to the shipped two-tier
    path at EVERY step, across a churning resident pool (misses / evictions /
    re-hits) and the M=4 / M=6 DSpark verify shapes -- residency evolution never
    changes a gathered value.  Also proves v2 actually ENGAGES: the ring serves
    hits (not a silent no-op)."""
    n_experts, top_k = 8, 2
    seq = _route_seq(256, n_experts, top_k)

    rt_off, spec = _open_runtime(
        tmp_path / "off", runner_v2=False, expert_count=n_experts, top_k=top_k,
        resident_slots=3, transient=8, prefetch=0,
    )
    rt_v2, spec2 = _open_runtime(
        tmp_path / "v2", runner_v2=True, expert_count=n_experts, top_k=top_k,
        resident_slots=3, transient=8, prefetch=6,
    )
    try:
        assert rt_v2._single_slot_pool is True
        assert rt_v2.config.overlap_miss_reads is True
        assert rt_off._single_slot_pool is False
        sw_off = _switch(rt_off, spec)
        sw_v2 = _switch(rt_v2, spec2)
        for i, (rows, experts) in enumerate(seq):
            # v2: prefetch this step's experts a beat early so the true route can
            # consume them as ring hits (the SSD-hiding mechanism engaged).
            rt_v2.prefetch_experts(spec2.routed_layer_start, experts)
            _settle_prefetch(rt_v2)
            x, idx = _inputs(rows, top_k, spec.hidden_size, experts, step=i)
            o_off = sw_off(x, idx)
            _REAL_EVAL(o_off)
            rt_off.flush_deferred_slot_releases(evaluate=True)
            o_v2 = sw_v2(x, idx)
            _REAL_EVAL(o_v2)
            rt_v2.flush_deferred_slot_releases(evaluate=True)
            assert mx.array_equal(o_off, o_v2), (
                f"step {i} (rows={rows}, experts={experts}): v2 diverged from shipped"
            )
        assert rt_v2.counters.prefetch_hit_on_true_route > 0, "ring never served a hit"
    finally:
        rt_off.close()
        rt_v2.close()


# ---------------------------------------------------------------------------
# D. retune: confidence margin + DSpark-verify union prediction
# ---------------------------------------------------------------------------
def test_confidence_margin_trims_to_confident():
    """margin >= 0 keeps the full top-6; margin < 0 TRIMS to the confident subset
    (threshold above the 6th score) -> higher issued-set precision. A read of x +
    the frozen gate weights, so it never changes a routed output."""
    from mtplx.models.deepseek_v41_moe import _gate_prefix_impl, gate_predict_topk
    from tests.models.test_deepseek_v41_w93_lane_d_predictor import _make_gate

    gate = _make_gate(dim=32, n_routed=16, topk=6, seed=3)
    mx.random.seed(7)
    x = (0.5 * mx.random.normal((4, 32))).astype(mx.bfloat16)
    _REAL_EVAL(x)
    _, biased = _gate_prefix_impl(
        x, gate.weight, gate.e_score_correction_bias,
        float(gate.gate_temp), str(gate.score_func),
    )
    s = np.sort(np.array(biased), axis=1)
    gap = float(np.mean(s[:, -1] - s[:, -6]))  # mean (s1 - s6)

    def kept(margin):
        return int((np.array(gate_predict_topk(gate, x, 6, margin=margin)) >= 0).sum())

    n0 = kept(0.0)
    assert n0 == 4 * 6                 # margin 0 keeps the full top-6 per row
    assert kept(gap) == n0             # a widening margin still caps at k=6
    assert kept(-0.5 * gap) < n0       # a trim margin issues fewer (confident set)
    base = np.array(gate_predict_topk(gate, x, 6, margin=0.0))
    trim = np.array(gate_predict_topk(gate, x, 6, margin=-0.5 * gap))
    for r in range(4):
        assert {int(v) for v in trim[r] if v >= 0} <= {int(v) for v in base[r]}


def test_runner_v2_verify_forward_predicts_union(monkeypatch):
    """Under v2 an M=(K+1) verify forward predicts the per-row route for the next
    layer (NOT inert, unlike an AR-only GATE_PREFETCH which stays inert on T>1).
    Model-level with the predict suite's fake runtime."""
    from tests.models.test_deepseek_v41_w93_gate_prefetch_predict import (
        _arm, _new_model, _prefill,
    )

    monkeypatch.setenv("MTPLX_DSV41_RUNNER", "v2")
    model, args = _new_model(num_hidden_layers=8)
    _arm(model, _RUNNER_V2_GATE_PREFETCH_K)
    cache, token = _prefill(model, args, s=6, seed=9)
    for layer in model.model.layers:
        layer.mlp.switch_mlp._mtplx_gate_prefetch_pending = None
    # a T=6 verify row batch (the DSpark depth-5 shape)
    verify_ids = mx.array([[token] * 6])
    mx.eval(model(verify_ids, cache=cache))
    stashed = [
        i for i, layer in enumerate(model.model.layers)
        if getattr(layer.mlp.switch_mlp, "_mtplx_gate_prefetch_pending", None)
        is not None
    ]
    assert stashed, "v2 verify forward stashed no prediction (inert on T>1)"
    # the prediction is per-row (the union source has T=6 rows)
    pred = model.model.layers[stashed[0]].mlp.switch_mlp._mtplx_gate_prefetch_pending[1]
    _REAL_EVAL(pred)
    assert pred.shape[0] == 6, "verify prediction must be per-row (T=6)"
