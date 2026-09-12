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
                  resident_slots=2, transient=8, prefetch=10, overlap=None,
                  exact_pool=False):
    """Open a tiny real streamed runtime.  ``runner_v2`` sets the env BEFORE open
    so ``ExpertStreamingRuntime.open`` reads it for the single-pool gate.
    ``overlap`` (default = ``runner_v2``) sets the config's overlap_miss_reads --
    the v2 composition is single pool + prefetch ring + overlap_miss_reads.

    ``exact_pool`` caps the persistent cache at exactly ``resident_slots`` per layer
    via ``expert_cache_limit_bytes`` (the stricter secondary cap in
    ``plan_expert_memory``).  Without it the ``ring_pad`` headroom + a small expert
    count let the planner fit ALL experts resident (slots_per_layer == expert_count),
    so a churn test never evicts -- the review's all-hit finding."""
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
        expert_cache_limit_bytes=(
            resident_slots * spec.expert_record_bytes if exact_pool else None
        ),
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


def _churn_schedule(n_steps):
    """Per-step row count: mostly M=1 (AR) with periodic M=4 / M=6 (the DSpark
    depth-3 / depth-5 verify row batches)."""
    return [
        6 if i % 37 == 36 else (4 if i % 17 == 16 else 1) for i in range(n_steps)
    ]


def _churn_inputs(rows, top_k, hidden, step, n_experts):
    """A deterministic decode-like route with PER-ROW varied experts from a ROTATING
    disjoint window over ``n_experts`` -- consecutive steps churn a small resident
    pool (misses + evictions + re-hits) and each verify row batch touches up to
    ``rows*top_k`` distinct experts (not the same two every row).  Returns
    ``(x, idx, unique_experts)``."""
    mx.random.seed(95 + rows + 1000 * step)
    x = (0.3 * mx.random.normal((rows, 1, hidden))).astype(mx.bfloat16)
    base = (step * rows * top_k) % n_experts
    flat = [(base + j) % n_experts for j in range(rows * top_k)]
    idx = mx.array(flat, dtype=mx.int32).reshape((rows, top_k))
    _REAL_EVAL(x, idx)
    return x, idx, sorted({int(e) for e in flat})


def test_runner_v2_256_step_byte_identical_streamed(tmp_path):
    """256 decode-like routes over a REAL streamed runtime: v2 (single pool +
    prefetch ring + overlap_miss_reads) is byte-identical to the shipped two-tier
    path at EVERY step -- residency evolution never changes a gathered value
    (gathers are row-independent, so identity is EXPECTED).  With an EXACT 3-slot
    pool over 16 experts and per-row-varied rotating routes the pool genuinely
    churns (misses / EVICTIONS / re-hits), and predicting the NEXT step one beat
    ahead makes the ring both serve hits AND recycle some mispredictions
    (prefetch_wasted).  This is the review's all-hit finding fixed: slots_per_layer
    was silently 8 (ring_pad headroom over 8 experts), every M=4/M=6 row routed to
    the same two experts, and evictions / prefetch_wasted were 0."""
    n_experts, top_k, resident_slots = 16, 2, 3
    schedule = _churn_schedule(256)

    rt_off, spec = _open_runtime(
        tmp_path / "off", runner_v2=False, expert_count=n_experts, top_k=top_k,
        resident_slots=resident_slots, transient=8, prefetch=0, exact_pool=True,
    )
    rt_v2, spec2 = _open_runtime(
        tmp_path / "v2", runner_v2=True, expert_count=n_experts, top_k=top_k,
        resident_slots=resident_slots, transient=8, prefetch=6, exact_pool=True,
    )
    try:
        # the review's silent override: assert the pool is REALLY resident_slots
        # (not the planner-inflated slots_per_layer that made the churn all-hit).
        assert rt_off.plan.slots_per_layer == resident_slots
        assert rt_v2.plan.slots_per_layer == resident_slots
        assert rt_v2._single_slot_pool is True
        assert rt_v2.config.overlap_miss_reads is True
        assert rt_off._single_slot_pool is False
        layer = spec.routed_layer_start
        sw_off = _switch(rt_off, spec)
        sw_v2 = _switch(rt_v2, spec2)
        for i, rows in enumerate(schedule):
            # v2: predict the NEXT step's experts one beat ahead (the gate-oracle
            # one-token-ahead contract) so the ring both serves hits and recycles
            # some mispredicted commits unused (prefetch_wasted).
            nxt_rows = schedule[i + 1] if i + 1 < len(schedule) else 1
            _, _, nxt_experts = _churn_inputs(
                nxt_rows, top_k, spec.hidden_size, i + 1, n_experts
            )
            rt_v2.prefetch_experts(layer, nxt_experts)
            _settle_prefetch(rt_v2)
            x, idx, _ = _churn_inputs(rows, top_k, spec.hidden_size, i, n_experts)
            o_off = sw_off(x, idx)
            _REAL_EVAL(o_off)
            rt_off.flush_deferred_slot_releases(evaluate=True)
            o_v2 = sw_v2(x, idx)
            _REAL_EVAL(o_v2)
            rt_v2.flush_deferred_slot_releases(evaluate=True)
            assert mx.array_equal(o_off, o_v2), (
                f"step {i} (rows={rows}): v2 diverged from shipped"
            )
        c = rt_v2.counters
        assert c.prefetch_hit_on_true_route > 0, "ring never served a hit"
        assert c.evictions > 0, "resident pool never evicted (all-hit again)"
        assert c.prefetch_wasted > 0, "no mispredicted prefetch was ever recycled"
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


def test_v2_issue_site_drops_sentinels_no_crash(tmp_path):
    """W95f CRITICAL regression: under v2 the retuned margin (-0.05) trims rank
    6..k of EVERY row to the -1 sentinel (gate_predict_topk: the threshold sits
    ABOVE the 6th score, so the 6th-ranked candidate is always trimmed).  The
    real issue site (_issue_gate_prefetch -> runtime.prefetch_experts ->
    GlobalPrefetchRing._key) rejects -1 ("expert id must be at least 0") and
    would kill the decode step on the first issued prediction.  The fix drops the
    sentinels (and dedups the verify's per-row union) before the ring sees them.

    The W93 predict suite faked prefetch_experts with a lambda, so it never
    reached the ring -- this drives the REAL ring on a real streamed runtime."""
    from mtplx.models.deepseek_v41_moe import gate_predict_topk
    from mtplx.models.expert_mlx import _issue_gate_prefetch
    from tests.models.test_deepseek_v41_w93_lane_d_predictor import _make_gate

    rt, spec = _open_runtime(
        tmp_path, runner_v2=True, expert_count=16, top_k=2,
        resident_slots=2, transient=8, prefetch=12,
    )
    try:
        layer = spec.routed_layer_start
        # 1. the retuned margin really produces sentinels, every row, and a real
        #    margin-trimmed prediction issues through the real ring without raising
        #    (this is exactly the array shape the DSpark verify hands the issue
        #    site; at HEAD prefetch_experts ValueErrors on the -1).
        gate = _make_gate(dim=32, n_routed=spec.expert_count, topk=6, seed=3)
        mx.random.seed(7)
        gx = (0.5 * mx.random.normal((4, 32))).astype(mx.bfloat16)
        _REAL_EVAL(gx)
        real_pred = gate_predict_topk(
            gate, gx, 6, margin=_RUNNER_V2_GATE_PREFETCH_MARGIN
        )
        _REAL_EVAL(real_pred)
        rp = np.array(real_pred)
        assert (rp < 0).any(), "margin -0.05 produced no sentinel to regress on"
        assert int((rp < 0).sum(axis=1).min()) >= 1, "every row must carry >=1 sentinel"
        _issue_gate_prefetch(rt, (layer, real_pred))  # must NOT raise

        # 2. deterministic count: [6, 7, -1, -1, -1, -1] -> ids {6, 7} -> 2.
        before = int(rt.counters.prefetch_predicted)
        pred = mx.array([[6, 7, -1, -1, -1, -1]], dtype=mx.int32)
        _issue_gate_prefetch(rt, (layer, pred))  # must NOT raise
        assert int(rt.counters.prefetch_predicted) - before == 2, (
            rt.counters.prefetch_predicted
        )
        _settle_prefetch(rt)
    finally:
        rt.close()


# ---------------------------------------------------------------------------
# E. W95f -- the speculative-byte budget (demand accounting + recovering window)
# ---------------------------------------------------------------------------
def test_v2_cold_demand_misses_increment_demand_bytes(tmp_path):
    """W95f: a COLD demand miss (expert never resident, never prefetched) advances
    ``demand_bytes_read`` at the plan site, giving the speculative-byte budget a
    real demand denominator. At HEAD the only writer was the reconcile FALLBACK, so
    cold misses left the denominator at 0 and the budget throttled nothing."""
    rt, spec = _open_runtime(
        tmp_path, runner_v2=True, expert_count=16, top_k=2,
        resident_slots=2, transient=8, prefetch=6,
    )
    try:
        layer = spec.routed_layer_start
        rec = rt._record_bytes_for_layer(layer)
        assert rt.demand_bytes_read == 0  # nothing streamed yet
        # route two cold-miss experts (never resident, never prefetched) -> two
        # demand SSD reads.  (Fails at HEAD: demand_bytes_read stays 0.)
        x, idx = _inputs(1, spec.top_k, spec.hidden_size, [8, 9])
        _REAL_EVAL(_switch(rt, spec)(x, idx))
        rt.flush_deferred_slot_releases(evaluate=True)
        assert rt.demand_bytes_read == 2 * rec, rt.demand_bytes_read
    finally:
        rt.close()


def test_v2_byte_budget_recovers_after_latch(tmp_path):
    """W95f: after speculation runs far ahead of demand (emulating the post-fallback
    state the review found), the budget backs off for the CURRENT decode token but
    a LATER token issues prefetch again -- a recovering window, not a one-way latch.
    At HEAD the cumulative-since-open comparison stayed latched off for the life of
    the process (demand only grew via another fallback that was never issued)."""
    rt, spec = _open_runtime(
        tmp_path, runner_v2=True, expert_count=16, top_k=2,
        resident_slots=2, transient=8, prefetch=6,
    )
    try:
        layer = spec.routed_layer_start
        rec = rt._record_bytes_for_layer(layer)
        # emulate one fallback then heavy speculation: demand = 1 record, spec = 10x.
        rt.demand_bytes_read = rec
        rt.speculative_bytes_read = 10 * rec
        skips0 = rt._prefetch_budget_skips
        assert rt.prefetch_experts(layer, [8, 9]) == 0  # over budget -> skip
        assert rt._prefetch_budget_skips == skips0 + 1
        # cross a decode-token boundary with a real route: the window re-snapshots
        # its marks to the current cumulative totals, emptying the window.
        x, idx = _inputs(1, spec.top_k, spec.hidden_size, [0, 1])
        _REAL_EVAL(_switch(rt, spec)(x, idx))
        rt.flush_deferred_slot_releases(evaluate=True)
        assert rt._decode_token_index >= 1  # a token boundary elapsed
        # window is empty now -> prefetch issues again (real speculative reads).
        # (Fails at HEAD: still latched, returns 0.)
        issued = rt.prefetch_experts(layer, [10, 11])
        _settle_prefetch(rt)
        assert issued > 0, "budget stayed latched after a decode-token boundary"
    finally:
        rt.close()


def test_v2_receipt_blocks_reach_harness_and_daemon(tmp_path):
    """W95f (review HIGH-3): the runner + gate_prefetch receipt blocks reach BOTH
    the harness receipt (ab_decode_env_levers._runner_receipt_blocks) and the served
    daemon's stream-counter path (serve_stream_counters.snapshot_stream_counters),
    carrying budget_skips, the committed+awaited denominator, and per-decode-token
    normalisations.  At HEAD they lived only in resource_telemetry_snapshot, whose
    callers were the other bench scripts + tests -- ab_decode_env_levers and the
    daemon never logged them."""
    import importlib.util
    from types import SimpleNamespace
    from mtplx.serve_stream_counters import snapshot_stream_counters

    rt, spec = _open_runtime(
        tmp_path / "v2", runner_v2=True, expert_count=16, top_k=2,
        resident_slots=2, transient=8, prefetch=6,
    )
    try:
        sw = _switch(rt, spec)
        for i in range(3):  # a few misses so counters + decode_steps are non-trivial
            x, idx = _inputs(1, spec.top_k, spec.hidden_size, [8 + i, 9 + i], step=i)
            _REAL_EVAL(sw(x, idx))
            rt.flush_deferred_slot_releases(evaluate=True)

        # (a) the served daemon's stream-counter path surfaces both blocks.
        ssc = snapshot_stream_counters(rt)
        assert "runner" in ssc and "gate_prefetch" in ssc
        assert "budget_skips" in ssc["runner"]

        # (b) the runner block carries the review's additions: the committed+awaited
        #     denominator, a derived hit rate, and per-decode-token normalisations.
        runner = rt.resource_telemetry_snapshot(mx_module=mx)["runner"]
        for key in (
            "budget_skips", "prefetch_committed", "prefetch_awaited_inflight",
            "prefetch_hit_rate", "decode_steps", "expert_misses_per_token",
            "speculative_bytes_per_token", "demand_bytes_per_token",
        ):
            assert key in runner, f"runner block missing {key}"
        assert runner["decode_steps"] >= 1
        assert 0.0 <= runner["prefetch_hit_rate"] <= 1.0  # committed+awaited denom

        # (c) the harness embeds both onto a fake run's receipt.
        _path = (
            Path(__file__).resolve().parents[1]
            / "scripts" / "deepseek_v41" / "ab_decode_env_levers.py"
        )
        _spec = importlib.util.spec_from_file_location("dsv41_ab_w95f", _path)
        ab = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(ab)
        blocks = ab._runner_receipt_blocks(SimpleNamespace(_mtplx_expert_runtime=rt))
        assert "runner" in blocks and "gate_prefetch" in blocks
        assert "budget_skips" in blocks["runner"]
    finally:
        rt.close()

    # (d) OFF: neither block leaks into snapshot() or the stream-counter path
    #     (the shipped snapshot stays byte-unchanged).
    rt_off, _ = _open_runtime(tmp_path / "off", runner_v2=False, prefetch=0)
    try:
        snap_off = rt_off.snapshot()
        assert "runner" not in snap_off and "gate_prefetch" not in snap_off
        ssc_off = snapshot_stream_counters(rt_off)
        assert "runner" not in ssc_off and "gate_prefetch" not in ssc_off
    finally:
        rt_off.close()
