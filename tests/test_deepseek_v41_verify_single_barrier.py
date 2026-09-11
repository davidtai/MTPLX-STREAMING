"""W61 -- verify single-barrier fast path for the small-M DECODE switch.

The DSpark K+1 verify forward (2..8 rows) otherwise splits its rows*top_k
assignments across several transient-bounded route_waves, each paying its own
device->host fence (window 25: a 4-row verify ~630 ms in moe.routed_switch vs
~74 ms for M=1). All-hit experts live in persistent slots (no transient bound), so
env ``MTPLX_DSV41_VERIFY_SINGLE_BARRIER`` (default ON) pins the whole route with one
``try_all_hit_route`` and gathers rows*top_k in ONE wave via the K27 sorted
gather_qmm, deferred-released once -> exactly ONE routing barrier per layer.

Byte-identity is asserted flag on vs off on a REAL component-bank runtime for
all-hit (M=2,4,8) and for the miss cases (which fall through to the bounded loop, so
the two paths coincide). A counting test asserts the fast path costs exactly one
blocking host sync (the ``mx.eval(indices)`` barrier) per layer at M=4 all-hit,
versus the several the split-wave path pays. CPU-pinned; no GPU. Run under
``nice -n 19``.
"""

from __future__ import annotations

import os

import mlx.core as mx
import pytest

pytest.importorskip("mlx.core")

mx.set_default_device(mx.cpu)

import mtplx.models.expert_mlx as expert_mlx  # noqa: E402
from mtplx.expert_manifest import load_expert_manifest  # noqa: E402
from mtplx.expert_runtime import ExpertStreamingConfig, ExpertStreamingRuntime  # noqa: E402
from mtplx.models.expert_mlx import (  # noqa: E402
    HotExpertSwitchGLU,
    make_mlx_component_bank_allocator,
)

FLAG = "MTPLX_DSV41_VERIFY_SINGLE_BARRIER"
_REAL_EVAL = mx.eval


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


def _open_runtime(tmp_path, *, expert_count=4, top_k=2, resident_slots=4, transient=2):
    from tests.test_streamed_models import _integrated_hy3_artifact

    root, config, spec, manifest_path = _integrated_hy3_artifact(
        tmp_path, expert_count=expert_count, top_k=top_k
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


def _warm_all(rt, spec, experts):
    """Route each expert once (fenced) so it is persistent-resident (all-hit)."""
    os.environ.pop(FLAG, None)
    sw = HotExpertSwitchGLU(rt, spec.routed_layer_start)
    hidden = spec.hidden_size
    for e in experts:
        idx = mx.array([[e] * spec.top_k], dtype=mx.int32)  # 1 row, top_k copies of e
        x = mx.zeros((1, 1, hidden), dtype=mx.bfloat16)
        out = sw(x, idx)
        _REAL_EVAL(out)
    rt.flush_deferred_slot_releases(evaluate=True)


def _inputs(rows, top_k, hidden, experts):
    mx.random.seed(31 + rows)
    x = (0.3 * mx.random.normal((rows, 1, hidden))).astype(mx.bfloat16)
    # each of the rows*top_k assignments routes to experts[i % len]
    flat = [experts[i % len(experts)] for i in range(rows * top_k)]
    idx = mx.array(flat, dtype=mx.int32).reshape((rows, top_k))
    _REAL_EVAL(x, idx)
    return x, idx


def _run(rt, spec, x, idx, *, flag):
    if flag:
        os.environ[FLAG] = "1"
    else:
        os.environ[FLAG] = "0"
    sw = HotExpertSwitchGLU(rt, spec.routed_layer_start)
    out = sw(x, idx)
    _REAL_EVAL(out)
    rt.flush_deferred_slot_releases(evaluate=True)
    return out


# ---------------------------------------------------------------------------
# byte-identity: all-hit at M=2,4,8, flag on vs off
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rows", [2, 4, 8])
def test_all_hit_byte_identical_on_vs_off(tmp_path, rows) -> None:
    rt, spec = _open_runtime(tmp_path, expert_count=4, top_k=2, resident_slots=4, transient=2)
    try:
        experts = list(range(spec.expert_count))
        _warm_all(rt, spec, experts)
        x, idx = _inputs(rows, spec.top_k, spec.hidden_size, experts)
        out_off = _run(rt, spec, x, idx, flag=False)
        out_on = _run(rt, spec, x, idx, flag=True)
        assert out_off.shape == out_on.shape == (rows, spec.top_k, spec.hidden_size)
        assert mx.array_equal(out_off, out_on), f"M={rows}: fast path != legacy"
    finally:
        rt.close()


# ---------------------------------------------------------------------------
# byte-identity: split / all-miss fall through to the bounded loop -> identical
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rows", [2, 4])
@pytest.mark.parametrize("kind", ["split", "all_miss"])
def test_miss_cases_byte_identical_on_vs_off(tmp_path, rows, kind) -> None:
    # resident_slots=2 < expert_count=4, so experts 2,3 are never persistent ->
    # any route touching them misses; the fast path returns None and falls through.
    dir_off = tmp_path / "off"
    dir_on = tmp_path / "on"
    dir_off.mkdir()
    dir_on.mkdir()
    rt, spec = _open_runtime(dir_off, expert_count=4, top_k=2, resident_slots=2, transient=2)
    try:
        if kind == "split":
            _warm_all(rt, spec, [0, 1])          # 0,1 resident; 2,3 miss
        x, idx = _inputs(rows, spec.top_k, spec.hidden_size, [0, 1, 2, 3])
        out_off = _run(rt, spec, x, idx, flag=False)
    finally:
        rt.close()
    # fresh runtime so the on-run's residency matches the off-run's start.
    rt2, spec2 = _open_runtime(dir_on, expert_count=4, top_k=2, resident_slots=2, transient=2)
    try:
        if kind == "split":
            _warm_all(rt2, spec2, [0, 1])
        x2, idx2 = _inputs(rows, spec2.top_k, spec2.hidden_size, [0, 1, 2, 3])
        out_on = _run(rt2, spec2, x2, idx2, flag=True)
        assert mx.array_equal(out_off, out_on), f"{kind} M={rows}: fast fall-through != legacy"
    finally:
        rt2.close()


# ---------------------------------------------------------------------------
# counting: exactly ONE blocking host sync per layer at M=4 all-hit
# ---------------------------------------------------------------------------
def _count_blocking_evals(rt, spec, x, idx, *, flag):
    if flag:
        os.environ[FLAG] = "1"
    else:
        os.environ[FLAG] = "0"
    n = {"eval": 0}
    real = expert_mlx.mx.eval

    def counting_eval(*a, **k):
        n["eval"] += 1
        return real(*a, **k)

    sw = HotExpertSwitchGLU(rt, spec.routed_layer_start)
    expert_mlx.mx.eval = counting_eval
    try:
        out = sw(x, idx)
    finally:
        expert_mlx.mx.eval = real
    real(out)
    rt.flush_deferred_slot_releases(evaluate=True)
    return n["eval"]


def test_one_host_sync_per_layer_at_m4_all_hit(tmp_path) -> None:
    # transient=2 so the legacy path splits the M=4 (rows*top_k=8, 4 unique) route
    # into several fenced waves; the fast path pins the whole route once.
    rt, spec = _open_runtime(tmp_path, expert_count=4, top_k=2, resident_slots=4, transient=2)
    try:
        experts = list(range(spec.expert_count))
        _warm_all(rt, spec, experts)
        x, idx = _inputs(4, spec.top_k, spec.hidden_size, experts)
        fast = _count_blocking_evals(rt, spec, x, idx, flag=True)
        legacy = _count_blocking_evals(rt, spec, x, idx, flag=False)
        # exactly one blocking sync (the mx.eval(indices) routing barrier).
        assert fast == 1, f"fast path did {fast} blocking evals, expected 1"
        # the split-wave legacy path pays more (barrier + a fence per split wave).
        assert legacy > 1, f"legacy path did {legacy} (expected >1 from split fences)"
    finally:
        rt.close()
