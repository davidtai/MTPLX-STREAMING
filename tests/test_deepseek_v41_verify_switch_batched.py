"""W81 -- batched single-fence verify switch for unique > transient capacity.

W66 collapsed a small-M DECODE verify to ONE routing barrier when the whole
route (hits + misses) fit transient capacity in one ``begin_split_route``.  On
the real model (window 31, receipt
docs/deepseek-v41/receipts/gpu-windows/window-31/dspark-16k-cell16k.json) the
DSpark 4-row verify routes rows*top_k = 4*6 = 24 assignments with ~20 unique
experts, but the bench runtime's transient capacity was ``spec.top_k`` (=6, the
loader's unset-``transient_slots`` default), so ~20 > 6 declined W66 and the route
fell to the bounded ``route_waves`` loop -- which fences EVERY transient-bounded
split wave part (moe.routed_switch 733.8 ms/token, ~3.6 begin_split_route per
layer-verify; W66's ``hot.verify_single_barrier_split`` never fired).

W81 keeps W66 for unique <= capacity and, for unique > capacity, admits the SAME
capacity-bounded waves ``route_waves`` would produce, but gathers each wave's hits
+ all miss parts deferred (async) behind a SINGLE ``synchronous_fence`` -- ONE
fence per batch, never one per wave part.  Pin-safety (W44 / issue #120): the layer
lock is not reentrant and non-final waves must recycle transient slots, so each
NON-final wave fences + releases before the next re-enters; only the FINAL wave
defers its whole release to the next routing barrier.  Net: (num_waves - 1) blocking
fences + the one routing barrier.

Byte-identity is asserted flag on (batched) vs off (bounded loop) at capacity=6 for
1-miss, multi-miss and all-miss 4-row routes with ~20 unique experts (the real
profile's effective slot count).  Engagement counters prove the batched fast path
takes >=95% of layer-verifies.  Sync/admission counts before (legacy) vs after
(batched) are compared.  CPU-pinned; no GPU; fake component-bank runtime; peak RSS
well under the guard.  Run under ``nice -n 19``.
"""

from __future__ import annotations

import math
import os

import mlx.core as mx
import pytest

pytest.importorskip("mlx.core")

mx.set_default_device(mx.cpu)

import mtplx.models.expert_mlx as expert_mlx  # noqa: E402
from mtplx.expert_manifest import load_expert_manifest  # noqa: E402
from mtplx.expert_runtime import (  # noqa: E402
    ExpertStreamingConfig,
    ExpertStreamingRuntime,
)
from mtplx.models.expert_mlx import (  # noqa: E402
    HotExpertSwitchGLU,
    make_mlx_component_bank_allocator,
)

FLAG = "MTPLX_DSV41_VERIFY_SINGLE_BARRIER"
_REAL_EVAL = mx.eval

# 4 rows x top_k 6 = 24 assignments; 20 unique experts ([0..19], four repeated to
# fill 24) reproduces the real 4-row verify shape.  transient capacity 6 forces
# ceil(20/6) = 4 capacity-bounded waves, so the batched multi-wave path runs.
_TOP_K = 6
_ROWS = 4
_EXPERT_COUNT = 24
_CAPACITY = 6
_RESIDENT_SLOTS = 24  # LRU room; the warm set decides how many stay resident
_N_UNIQUE = 20
_ROUTE_EXPERTS = [i % _N_UNIQUE for i in range(_ROWS * _TOP_K)]  # len 24, 20 unique
_UNIQUE_EXPERTS = sorted(set(_ROUTE_EXPERTS))
_EXPECTED_WAVES = math.ceil(_N_UNIQUE / _CAPACITY)  # 4

# warm list -> intended resident-hit set over the unique experts
_WARM_FOR_KIND = {
    "1miss": _UNIQUE_EXPERTS[:-1],          # 19 resident, 1 miss
    "multimiss": _UNIQUE_EXPERTS[: _N_UNIQUE // 2],  # 10 resident, 10 miss
    "allmiss": [],                          # nothing resident, 20 miss
}


@pytest.fixture(autouse=True)
def _cpu_and_flag():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    saved = os.environ.get(FLAG)
    try:
        yield
    finally:
        mx.set_default_device(prev)
        if saved is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = saved


def _open_runtime(tmp_path, *, resident_slots=_RESIDENT_SLOTS, transient=_CAPACITY):
    from tests.test_streamed_models import _integrated_hy3_artifact

    root, _config, spec, manifest_path = _integrated_hy3_artifact(
        tmp_path, expert_count=_EXPERT_COUNT, top_k=_TOP_K
    )
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    sc = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(resident_slots),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        transient_slots=transient,
        slot_layout="component-banks",
    )
    plan = sc.memory_plan(spec)
    assert plan.transient_slots == transient
    rt = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        sc,
        spec=spec,
        buffer_allocator=make_mlx_component_bank_allocator(
            plan, spec, load_expert_manifest(manifest_path)
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    return rt, spec


def _warm(rt, spec, experts):
    """Route each expert once (fenced) so it is persistent-resident (a hit)."""
    os.environ.pop(FLAG, None)
    sw = HotExpertSwitchGLU(rt, spec.routed_layer_start)
    for e in experts:
        idx = mx.array([[e] * spec.top_k], dtype=mx.int32)
        x = mx.zeros((1, 1, spec.hidden_size), dtype=mx.bfloat16)
        _REAL_EVAL(sw(x, idx))
    rt.flush_deferred_slot_releases(evaluate=True)


def _inputs(rows, top_k, hidden):
    mx.random.seed(81 + rows)
    x = (0.3 * mx.random.normal((rows, 1, hidden))).astype(mx.bfloat16)
    flat = [_ROUTE_EXPERTS[i % len(_ROUTE_EXPERTS)] for i in range(rows * top_k)]
    idx = mx.array(flat, dtype=mx.int32).reshape((rows, top_k))
    _REAL_EVAL(x, idx)
    return x, idx


def _run(rt, spec, x, idx, *, flag):
    os.environ[FLAG] = "1" if flag else "0"
    sw = HotExpertSwitchGLU(rt, spec.routed_layer_start)
    out = sw(x, idx)
    _REAL_EVAL(out)
    rt.flush_deferred_slot_releases(evaluate=True)
    return out


def _misses(rt, spec):
    resident = rt.peek_resident_experts(
        spec.routed_layer_start, tuple(_UNIQUE_EXPERTS)
    )
    return [e for e in _UNIQUE_EXPERTS if e not in resident]


# ---------------------------------------------------------------------------
# byte-identity: batched (flag on) vs bounded loop (flag off), unique(20) > cap(6)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rows", [2, 4, 8])
@pytest.mark.parametrize("kind", ["1miss", "multimiss", "allmiss"])
def test_batched_byte_identical_on_vs_off(tmp_path, rows, kind) -> None:
    warm = _WARM_FOR_KIND[kind]
    dir_off = tmp_path / "off"
    dir_on = tmp_path / "on"
    dir_off.mkdir()
    dir_on.mkdir()

    rt, spec = _open_runtime(dir_off)
    try:
        _warm(rt, spec, warm)
        x, idx = _inputs(rows, spec.top_k, spec.hidden_size)
        out_off = _run(rt, spec, x, idx, flag=False)
    finally:
        rt.close()

    rt2, spec2 = _open_runtime(dir_on)
    try:
        _warm(rt2, spec2, warm)
        x2, idx2 = _inputs(rows, spec2.top_k, spec2.hidden_size)
        out_on = _run(rt2, spec2, x2, idx2, flag=True)
        assert out_off.shape == out_on.shape == (rows, spec2.top_k, spec2.hidden_size)
        assert mx.array_equal(
            out_off, out_on
        ), f"{kind} M={rows}: batched fast path != bounded loop"
    finally:
        rt2.close()


# ---------------------------------------------------------------------------
# engagement: the batched fast path takes >=95% of 4-row layer-verifies
# ---------------------------------------------------------------------------
def _counter(probe, event):
    return probe.snapshot()["stages"].get(event, {}).get("count", 0)


def test_batched_engagement_at_least_95pct(tmp_path) -> None:
    import mtplx.expert_route_probe as probe

    saved_enabled = probe.ENABLED
    probe.ENABLED = True
    try:
        rt, spec = _open_runtime(tmp_path, resident_slots=_CAPACITY)
        try:
            # Keep the persistent cache at capacity (6) < 20 unique, so most of the
            # route misses every verify and the split/batched path (not W61 all-hit)
            # is what must engage.
            n_verifies = 40
            os.environ[FLAG] = "1"
            sw = HotExpertSwitchGLU(rt, spec.routed_layer_start)
            before_split = _counter(probe, "hot.verify_single_barrier_split")
            before_allhit = _counter(probe, "hot.verify_single_barrier")
            for i in range(n_verifies):
                x, idx = _inputs(_ROWS, spec.top_k, spec.hidden_size)
                out = sw(x, idx)
                _REAL_EVAL(out)
                rt.flush_deferred_slot_releases(evaluate=True)
            engaged_split = _counter(probe, "hot.verify_single_barrier_split") - before_split
            engaged_allhit = _counter(probe, "hot.verify_single_barrier") - before_allhit
            engaged = engaged_split + engaged_allhit
            assert engaged >= math.ceil(0.95 * n_verifies), (
                f"fast path engaged on {engaged}/{n_verifies} layer-verifies "
                f"(split={engaged_split}, all_hit={engaged_allhit}); expected >=95%"
            )
            # and the NEW batched path is what carried them (unique 20 > cap 6)
            assert engaged_split >= math.ceil(0.95 * n_verifies), (
                f"batched split path engaged on {engaged_split}/{n_verifies}"
            )
        finally:
            rt.close()
    finally:
        probe.ENABLED = saved_enabled


# ---------------------------------------------------------------------------
# sync / admission counts: batched vs legacy, per layer-verify
# ---------------------------------------------------------------------------
def _count(rt, spec, x, idx, *, flag):
    """Return (blocking mx.eval count, begin_split_route admission count)."""
    os.environ[FLAG] = "1" if flag else "0"
    n = {"eval": 0, "admit": 0}
    real_eval = expert_mlx.mx.eval
    real_bsr = rt.begin_split_route

    def counting_eval(*a, **k):
        n["eval"] += 1
        return real_eval(*a, **k)

    def counting_bsr(*a, **k):
        n["admit"] += 1
        return real_bsr(*a, **k)

    sw = HotExpertSwitchGLU(rt, spec.routed_layer_start)
    expert_mlx.mx.eval = counting_eval
    rt.begin_split_route = counting_bsr
    try:
        out = sw(x, idx)
    finally:
        expert_mlx.mx.eval = real_eval
        rt.begin_split_route = real_bsr
    real_eval(out)
    rt.flush_deferred_slot_releases(evaluate=True)
    return n["eval"], n["admit"]


def test_batched_one_fence_per_wave_vs_legacy(tmp_path) -> None:
    # all-miss so every wave is a split; unique 20 > cap 6 -> 4 waves.
    rt, spec = _open_runtime(tmp_path, resident_slots=_CAPACITY)
    try:
        _warm(rt, spec, [])  # nothing resident
        misses = len(_misses(rt, spec))
        assert misses == _N_UNIQUE, f"expected {_N_UNIQUE} misses, saw {misses}"
        x, idx = _inputs(_ROWS, spec.top_k, spec.hidden_size)

        fast_eval, fast_admit = _count(rt, spec, x, idx, flag=True)
        # Batched: one begin_split_route per wave (ceil(20/6)=4), one routing
        # barrier + one fence per NON-final wave = 1 + (waves-1) blocking evals.
        assert fast_admit == _EXPECTED_WAVES, (
            f"batched did {fast_admit} admissions, expected {_EXPECTED_WAVES} waves"
        )
        assert fast_eval == 1 + (_EXPECTED_WAVES - 1), (
            f"batched did {fast_eval} blocking evals, expected "
            f"{1 + (_EXPECTED_WAVES - 1)} (1 barrier + {_EXPECTED_WAVES - 1} "
            "non-final fences)"
        )
    finally:
        rt.close()

    # legacy bounded loop pays strictly more blocking evals (fences per wave part).
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    rt2, spec2 = _open_runtime(legacy_dir)
    try:
        _warm(rt2, spec2, [])
        x2, idx2 = _inputs(_ROWS, spec2.top_k, spec2.hidden_size)
        legacy_eval, legacy_admit = _count(rt2, spec2, x2, idx2, flag=False)
        assert legacy_eval > 1 + (_EXPECTED_WAVES - 1), (
            f"legacy did {legacy_eval} blocking evals; batched fence-per-wave "
            f"should be strictly fewer than the legacy fence-per-part path"
        )
    finally:
        rt2.close()


# ---------------------------------------------------------------------------
# receipt counters emitted on the batched path
# ---------------------------------------------------------------------------
def test_batched_receipt_counters_emitted(tmp_path) -> None:
    import mtplx.expert_route_probe as probe

    saved_enabled = probe.ENABLED
    probe.ENABLED = True
    try:
        rt, spec = _open_runtime(tmp_path, resident_slots=_CAPACITY)
        try:
            _warm(rt, spec, [])
            x, idx = _inputs(_ROWS, spec.top_k, spec.hidden_size)
            b_split = _counter(probe, "hot.verify_single_barrier_split")
            b_batched = _counter(probe, "hot.verify_single_barrier_batched")
            _run(rt, spec, x, idx, flag=True)
            assert _counter(probe, "hot.verify_single_barrier_split") == b_split + 1
            # unique(20) > cap(6) -> the multi-wave batched counter also ticks once
            assert (
                _counter(probe, "hot.verify_single_barrier_batched") == b_batched + 1
            )
        finally:
            rt.close()
    finally:
        probe.ENABLED = saved_enabled
