"""W66 -- split single-barrier fast path for the small-M DECODE switch.

W61 collapsed an ALL-HIT small-M verify (2..8 rows) to one routing barrier by
pinning the whole route with ``try_all_hit_route`` and gathering rows*top_k once.
A verify with ANY miss declined that probe and fell back to the multi-barrier
bounded ``route_waves`` loop, which fences every transient-bounded split wave
(window 25: ~630 ms per 4-row verify).

W66 is the split counterpart: when the WHOLE route (hits + misses) fits transient
capacity in one transaction, the layer admits it in ONE ``begin_split_route``
submission -- hits pinned in persistent slots, the missing experts streamed into
transient together -- gathers every rows*top_k assignment through the shared
component-bank helper (deferred / async-submitted), and defers the single route's
release to the next routing barrier.  Net: exactly ONE routing barrier
(``mx.eval(indices)``) per layer, no per-wave synchronous fence.  Because hits and
misses share one pinned set, a decode miss can never be promoted into (and recycle)
a slot whose gather is still pending; and because the gather is row-independent,
the single deferred route is byte-identical to the fenced bounded loop.  A route
whose unique experts exceed transient capacity cannot be one transaction and falls
through to the bounded loop unchanged.

Gated on ``MTPLX_DSV41_VERIFY_SINGLE_BARRIER`` (default ON), receipt counter
``hot.verify_single_barrier_split``.  Byte-identity is asserted flag on vs off on a
tiny synthetic component-bank runtime (fake bank; peak RSS < 200 MB) for 1-miss,
multi-miss and all-miss layers at M=2,4,8.  A counting test asserts the fast path
costs 1 routing barrier + ceil(misses / transient_capacity) admissions per layer,
versus the fenced-per-wave bounded loop.  CPU-pinned; no GPU.  Run under
``nice -n 19``.
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

# A 4-unique route ([0,1,2,3]) with transient_slots=4 engages the single-barrier
# split path at every M; resident_slots=6 keeps the warmed hits persistent so the
# miss count is exactly len({0,1,2,3}) - len(warm).
_ROUTE_EXPERTS = [0, 1, 2, 3]
_CAPACITY = 4
_EXPERT_COUNT = 6
_RESIDENT_SLOTS = 6

# warm list -> intended unique-miss count over _ROUTE_EXPERTS
_WARM_FOR_KIND = {
    "1miss": [0, 1, 2],
    "multimiss": [0, 1],
    "allmiss": [],
}
_EXPECTED_MISSES = {"1miss": 1, "multimiss": 2, "allmiss": 4}


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


def _open_runtime(tmp_path):
    from tests.test_streamed_models import _integrated_hy3_artifact

    root, _config, spec, manifest_path = _integrated_hy3_artifact(
        tmp_path, expert_count=_EXPERT_COUNT, top_k=2
    )
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    sc = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(_RESIDENT_SLOTS),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        transient_slots=_CAPACITY,
        slot_layout="component-banks",
    )
    plan = sc.memory_plan(spec)
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
    mx.random.seed(31 + rows)
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
        spec.routed_layer_start, tuple(_ROUTE_EXPERTS)
    )
    return [e for e in _ROUTE_EXPERTS if e not in resident]


# ---------------------------------------------------------------------------
# byte-identity: flag on vs off, 1-miss / multi-miss / all-miss at M=2,4,8
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rows", [2, 4, 8])
@pytest.mark.parametrize("kind", ["1miss", "multimiss", "allmiss"])
def test_split_byte_identical_on_vs_off(tmp_path, rows, kind) -> None:
    warm = _WARM_FOR_KIND[kind]
    dir_off = tmp_path / "off"
    dir_on = tmp_path / "on"
    dir_off.mkdir()
    dir_on.mkdir()

    rt, spec = _open_runtime(dir_off)
    try:
        _warm(rt, spec, warm)
        assert len(_misses(rt, spec)) == _EXPECTED_MISSES[kind]
        x, idx = _inputs(rows, spec.top_k, spec.hidden_size)
        out_off = _run(rt, spec, x, idx, flag=False)
    finally:
        rt.close()

    # fresh runtime so the on-run's residency matches the off-run's start.
    rt2, spec2 = _open_runtime(dir_on)
    try:
        _warm(rt2, spec2, warm)
        x2, idx2 = _inputs(rows, spec2.top_k, spec2.hidden_size)
        out_on = _run(rt2, spec2, x2, idx2, flag=True)
        assert out_off.shape == out_on.shape == (rows, spec2.top_k, spec2.hidden_size)
        assert mx.array_equal(
            out_off, out_on
        ), f"{kind} M={rows}: split fast path != bounded loop"
    finally:
        rt2.close()


# ---------------------------------------------------------------------------
# counting: 1 routing barrier + ceil(misses / capacity) admissions per layer
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


@pytest.mark.parametrize("kind", ["1miss", "multimiss", "allmiss"])
def test_split_host_syncs_per_layer(tmp_path, kind) -> None:
    warm = _WARM_FOR_KIND[kind]
    rt, spec = _open_runtime(tmp_path)
    try:
        _warm(rt, spec, warm)
        misses = len(_misses(rt, spec))
        assert misses == _EXPECTED_MISSES[kind]
        x, idx = _inputs(4, spec.top_k, spec.hidden_size)

        fast_eval, fast_admit = _count(rt, spec, x, idx, flag=True)
        # The fast path fires: exactly one routing barrier (mx.eval(indices)),
        # every gather deferred (async_eval, not a blocking eval), and the whole
        # route admitted in ceil(misses / capacity) begin_split_route submissions
        # -- one submission here since unique <= transient capacity.
        expected_admissions = math.ceil(misses / _CAPACITY)
        assert fast_eval == 1, f"{kind}: fast path did {fast_eval} blocking evals"
        assert fast_admit == expected_admissions, (
            f"{kind}: fast path did {fast_admit} admissions, "
            f"expected ceil({misses}/{_CAPACITY})={expected_admissions}"
        )
        # host syncs per layer = 1 routing barrier + ceil(misses / capacity).
        host_syncs = fast_eval + fast_admit
        assert host_syncs == 1 + expected_admissions
    finally:
        rt.close()


def test_split_beats_bounded_loop_on_syncs(tmp_path) -> None:
    # The bounded route_waves loop (flag off) fences every transient-bounded
    # split wave, so a miss verify pays several blocking evals; the split fast
    # path pays exactly one.
    rt, spec = _open_runtime(tmp_path)
    try:
        _warm(rt, spec, _WARM_FOR_KIND["allmiss"])
        x, idx = _inputs(4, spec.top_k, spec.hidden_size)
        fast_eval, _ = _count(rt, spec, x, idx, flag=True)
        legacy_eval, _ = _count(rt, spec, x, idx, flag=False)
        assert fast_eval == 1, f"fast path did {fast_eval} blocking evals, expected 1"
        assert legacy_eval > fast_eval, (
            f"legacy did {legacy_eval} evals, fast did {fast_eval} "
            "(expected the fenced loop to pay more)"
        )
    finally:
        rt.close()


# ---------------------------------------------------------------------------
# receipt counter is emitted on the split path
# ---------------------------------------------------------------------------
def _counter(probe, event):
    return probe.snapshot()["stages"].get(event, {}).get("count", 0)


def test_split_receipt_counter_emitted(tmp_path) -> None:
    import mtplx.expert_route_probe as probe

    saved_enabled = probe.ENABLED
    probe.ENABLED = True  # count()/snapshot() are import-time gated otherwise
    try:
        rt, spec = _open_runtime(tmp_path)
        try:
            _warm(rt, spec, _WARM_FOR_KIND["multimiss"])
            x, idx = _inputs(4, spec.top_k, spec.hidden_size)
            before = _counter(probe, "hot.verify_single_barrier_split")
            _run(rt, spec, x, idx, flag=True)
            after = _counter(probe, "hot.verify_single_barrier_split")
            assert after == before + 1, (
                "expected one hot.verify_single_barrier_split tick"
            )
        finally:
            rt.close()
    finally:
        probe.ENABLED = saved_enabled
