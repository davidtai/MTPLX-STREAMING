"""W93 lane C -- gate-oracle prefetch ISSUE-ORDER + speculative-cap + reconcile
timeout/failure fallback, on a REAL streamed runtime.

Locks the lane-C half of the DSV4.1 one-layer-ahead gate-oracle prefetch fix
(docs/deepseek-v41/W93_GATE_PREFETCH.md), reusing the barrier tests'
fake-but-real component-bank runtime (``_integrated_hy3_artifact`` +
``ExpertStreamingRuntime`` + ``HotExpertSwitchGLU``) with a ``prefetch_slots``
ring armed:

  1. **issue order (HIGH-3)** -- the next layer's speculative prefetch is issued
     only AFTER this layer's own demand misses have been submitted; asserted via
     the reader's call order (demand reader vs speculative reader);
  2. **speculative cap (HIGH-3)** -- no more than N speculative reads run in
     flight at once (``MTPLX_DSV41_GATE_PREFETCH_MAX_INFLIGHT``, default 4);
  3. **reconcile timeout fallback (MED-a)** -- a never-completing speculative
     read makes the demand route fall back to a demand load; the route completes
     bounded and the gather is byte-identical to a pure-demand route;
  4. **I/O failure fallback (MED-a)** -- a failing speculative read likewise
     falls back to a demand load; byte-identical.

CPU-pinned; tiny synthetic artifact; no GPU, no real weights.  Run under
``nice -n 19`` and without ``pytest -n auto``.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import mlx.core as mx
import pytest

pytest.importorskip("mlx.core")

mx.set_default_device(mx.cpu)

from mtplx.expert_manifest import load_expert_manifest  # noqa: E402
from mtplx.expert_runtime import (  # noqa: E402
    ExpertStreamingConfig,
    ExpertStreamingRuntime,
)
from mtplx.models.expert_mlx import (  # noqa: E402
    HotExpertSwitchGLU,
    make_mlx_component_bank_allocator,
)

_REAL_EVAL = mx.eval


@pytest.fixture(autouse=True)
def _cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(prev)


def _open_runtime(
    tmp_path, *, expert_count=8, top_k=2, resident_slots=8, transient=4, prefetch=10
):
    from tests.test_streamed_models import _integrated_hy3_artifact

    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    root, config, spec, manifest_path = _integrated_hy3_artifact(
        tmp_path, expert_count=expert_count, top_k=top_k
    )
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    ring_pad = (transient + prefetch + 16) * spec.expert_record_bytes
    sc = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=(
            fixed + spec.persistent_cache_bytes(resident_slots) + ring_pad
        ),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        transient_slots=transient,
        slot_layout="component-banks",
        prefetch_slots=prefetch,
        resource_telemetry=True,
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


def _layer(spec):
    return spec.routed_layer_start


def _switch(rt, spec):
    return HotExpertSwitchGLU(rt, _layer(spec))


def _route_once(rt, spec, experts):
    """Route ``experts`` once (fenced) so they become persistent-resident."""
    sw = _switch(rt, spec)
    for e in experts:
        idx = mx.array([[e] * spec.top_k], dtype=mx.int32)
        x = mx.zeros((1, 1, spec.hidden_size), dtype=mx.bfloat16)
        _REAL_EVAL(sw(x, idx))
    rt.flush_deferred_slot_releases(evaluate=True)


def _inputs(rows, top_k, hidden, experts):
    mx.random.seed(93 + rows)
    x = (0.3 * mx.random.normal((rows, 1, hidden))).astype(mx.bfloat16)
    flat = [experts[i % len(experts)] for i in range(rows * top_k)]
    idx = mx.array(flat, dtype=mx.int32).reshape((rows, top_k))
    _REAL_EVAL(x, idx)
    return x, idx


def _settle_prefetch(rt):
    """Wait out every in-flight speculative read, deterministically."""
    with rt._prefetch_lock:
        pending = tuple(rt._prefetch_futures)
    for future in pending:
        try:
            future.result()
        except BaseException:
            pass


# ---------------------------------------------------------------------------
# 1. issue order (HIGH-3): the L+1 prefetch is submitted AFTER L's demand misses
#    -- asserted via the demand reader vs speculative reader call order.
# ---------------------------------------------------------------------------
def test_prefetch_issued_after_demand_misses(tmp_path) -> None:
    # resident 0,1; the current route needs cold 4,5 (demand misses); the stashed
    # prediction targets cold 6,7 (the speculative reads). The fix moves the issue
    # point to AFTER begin_split_route has submitted the demand misses.
    rt, spec = _open_runtime(
        tmp_path, expert_count=8, top_k=2, resident_slots=2, transient=4, prefetch=10
    )
    layer = _layer(spec)
    order: list[str] = []           # byte-reader call order (the instruction's test)
    order_lock = threading.Lock()
    submit: list[str] = []          # runtime submission order (gen-thread, exact)
    snap = {"at_issue": None}       # reader log snapshot at speculative submission
    try:
        _route_once(rt, spec, [0, 1])  # warm BEFORE instrumenting (no log pollution)
        real_ensure_part = rt.slots.ensure_route_part
        real_ensure = rt.slots.ensure_route
        real_load_spec = rt.slots.load_speculative
        real_begin = rt.begin_split_route
        real_prefetch = rt.prefetch_experts

        def logged_ensure_part(*a, **k):
            with order_lock:
                order.append("demand")
            return real_ensure_part(*a, **k)

        def logged_ensure(*a, **k):
            with order_lock:
                order.append("demand")
            return real_ensure(*a, **k)

        def logged_load_spec(*a, **k):
            with order_lock:
                order.append("spec")
            return real_load_spec(*a, **k)

        def logged_begin(*a, **k):
            submit.append("submit_demand")
            return real_begin(*a, **k)

        def logged_prefetch(*a, **k):
            submit.append("submit_spec")
            # Snapshot the byte-reader log at the instant the speculative prefetch
            # is issued: with the fix, the layer's demand reads have already run
            # and no speculative read has (the issue point is after the route).
            with order_lock:
                snap["at_issue"] = list(order)
            return real_prefetch(*a, **k)

        rt.slots.ensure_route_part = logged_ensure_part
        rt.slots.ensure_route = logged_ensure
        rt.slots.load_speculative = logged_load_spec
        rt.begin_split_route = logged_begin
        rt.prefetch_experts = logged_prefetch
        try:
            x, idx = _inputs(1, spec.top_k, spec.hidden_size, [4, 5])
            sw = _switch(rt, spec)
            sw._mtplx_gate_prefetch_pending = (
                layer,
                mx.array([[6, 7]], dtype=mx.int32),
            )
            out = sw(x, idx)
            _REAL_EVAL(out)
            rt.flush_deferred_slot_releases(evaluate=True)
            _settle_prefetch(rt)  # let the speculative reads actually run
        finally:
            rt.slots.ensure_route_part = real_ensure_part
            rt.slots.ensure_route = real_ensure
            rt.slots.load_speculative = real_load_spec
            rt.begin_split_route = real_begin
            rt.prefetch_experts = real_prefetch
        # (a) runtime submission order (deterministic, generation thread): the
        #     demand-miss submission (begin_split_route) precedes the speculative
        #     issue (prefetch_experts).
        assert "submit_demand" in submit and "submit_spec" in submit, submit
        assert submit.index("submit_demand") < submit.index("submit_spec"), (
            f"prefetch issued before begin_split_route submitted misses: {submit}"
        )
        # (b) at the instant the prefetch is issued, the layer's demand reads have
        #     already happened and no speculative read has yet.
        assert snap["at_issue"] is not None, "prefetch was never issued"
        assert "demand" in snap["at_issue"], (
            f"demand misses not yet read when prefetch issued: {snap['at_issue']}"
        )
        assert "spec" not in snap["at_issue"], (
            f"a speculative read preceded the demand misses: {snap['at_issue']}"
        )
        # (c) reader call order overall: every demand read precedes every spec read.
        assert "demand" in order and "spec" in order, order
        first_spec = order.index("spec")
        last_demand = max(i for i, tag in enumerate(order) if tag == "demand")
        assert last_demand < first_spec, (
            f"speculative read ran before the layer's demand misses: {order}"
        )
    finally:
        rt.close()


# ---------------------------------------------------------------------------
# 2. speculative cap (HIGH-3): no more than N speculative reads in flight.
# ---------------------------------------------------------------------------
def test_speculative_reads_capped_in_flight(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_DSV41_GATE_PREFETCH_MAX_INFLIGHT", "2")
    # transient=16 -> the fraction budget alone would allow >2 reads; the cap (2)
    # is the binding constraint, proving the cap -- not the fraction -- limits it.
    rt, spec = _open_runtime(
        tmp_path, expert_count=8, top_k=2, resident_slots=2, transient=16, prefetch=10
    )
    layer = _layer(spec)
    # cap applied at construction (executor max_workers == in-flight read bound).
    assert rt._prefetch_max_reads == 2, (
        f"speculative cap not applied: {rt._prefetch_max_reads}"
    )
    active = {"n": 0, "max": 0}
    active_lock = threading.Lock()
    release = threading.Event()
    real_load = rt.slots.load_speculative

    def gated_load(l, load):
        with active_lock:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        try:
            release.wait(timeout=10.0)
            return real_load(l, load)
        finally:
            with active_lock:
                active["n"] -= 1

    rt.slots.load_speculative = gated_load
    try:
        _route_once(rt, spec, [0, 1])
        # predict 6 cold experts -> 6 speculative reads queued to a 2-worker pool.
        issued = rt.prefetch_experts(layer, [2, 3, 4, 5, 6, 7])
        assert issued >= 3, f"expected several speculative reads, got {issued}"
        # give the workers time to pile up at the gate; concurrency floors at N=2.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with active_lock:
                cur = active["n"]
            if cur >= 2:
                break
            time.sleep(0.01)
        with active_lock:
            reached = active["max"]
        assert reached <= 2, f"more than 2 speculative reads in flight: {reached}"
        assert reached == 2, f"cap should permit exactly 2 concurrent, saw {reached}"
    finally:
        release.set()
        _settle_prefetch(rt)
        rt.slots.load_speculative = real_load
        rt.close()


# ---------------------------------------------------------------------------
# 3. reconcile timeout fallback (MED-a): a never-completing speculative read ->
#    the demand route times out awaiting it, falls back to a demand load, and the
#    gather is byte-identical to a pure-demand route.
# ---------------------------------------------------------------------------
def test_reconcile_timeout_falls_back_to_demand(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_DSV41_GATE_PREFETCH_RECONCILE_TIMEOUT_S", "0.3")
    # control: expert 6 streamed on the demand route itself (no speculation).
    rt0, spec0 = _open_runtime(
        tmp_path / "ctl", expert_count=8, top_k=2, resident_slots=2, transient=8,
        prefetch=10,
    )
    try:
        _route_once(rt0, spec0, [0, 1])
        x0, idx0 = _inputs(1, spec0.top_k, spec0.hidden_size, [6])
        out_demand = _switch(rt0, spec0)(x0, idx0)
        _REAL_EVAL(out_demand)
        rt0.flush_deferred_slot_releases(evaluate=True)
    finally:
        rt0.close()
    # candidate: a speculative read for 6 that never completes.
    rt, spec = _open_runtime(
        tmp_path / "cand", expert_count=8, top_k=2, resident_slots=2, transient=8,
        prefetch=10,
    )
    layer = _layer(spec)
    stuck = threading.Event()
    real_load = rt.slots.load_speculative

    def stuck_load(l, load):
        stuck.wait(timeout=15.0)  # never released until teardown -> read hangs
        return real_load(l, load)

    rt.slots.load_speculative = stuck_load
    try:
        _route_once(rt, spec, [0, 1])
        assert rt.prefetch_experts(layer, [6]) == 1
        bank = rt._banks[layer]
        assert bank.prefetch_ticket(6) is not None  # in flight, blocked
        before = rt.demand_bytes_read
        # the true route needs 6: begin_split_route -> reconcile awaits (0.3 s) ->
        # timeout -> invalidate ticket -> demand load. Bounded, not an unbounded park.
        t0 = time.time()
        x, idx = _inputs(1, spec.top_k, spec.hidden_size, [6])
        out = _switch(rt, spec)(x, idx)
        _REAL_EVAL(out)
        rt.flush_deferred_slot_releases(evaluate=True)
        elapsed = time.time() - t0
        assert elapsed < 5.0, f"route did not fall back promptly ({elapsed:.2f}s)"
        assert mx.array_equal(out_demand, out), "timeout fallback changed the gather"
        assert rt.demand_bytes_read > before, "demand fallback bytes not accounted"
    finally:
        stuck.set()
        _settle_prefetch(rt)
        rt.slots.load_speculative = real_load
        rt.close()


# ---------------------------------------------------------------------------
# 4. I/O failure fallback (MED-a): a failing speculative read -> demand load,
#    byte-identical.
# ---------------------------------------------------------------------------
def test_reconcile_read_failure_falls_back_to_demand(tmp_path) -> None:
    # control: demand-served expert 6.
    rt0, spec0 = _open_runtime(
        tmp_path / "ctl", expert_count=8, top_k=2, resident_slots=2, transient=8,
        prefetch=10,
    )
    try:
        _route_once(rt0, spec0, [0, 1])
        x0, idx0 = _inputs(1, spec0.top_k, spec0.hidden_size, [6])
        out_demand = _switch(rt0, spec0)(x0, idx0)
        _REAL_EVAL(out_demand)
        rt0.flush_deferred_slot_releases(evaluate=True)
    finally:
        rt0.close()
    # candidate: the speculative read for 6 blocks in flight, then FAILS.
    rt, spec = _open_runtime(
        tmp_path / "cand", expert_count=8, top_k=2, resident_slots=2, transient=8,
        prefetch=10,
    )
    layer = _layer(spec)
    gate = threading.Event()
    real_load = rt.slots.load_speculative

    def failing_load(l, load):
        gate.wait(timeout=15.0)
        raise OSError("synthetic speculative read failure")

    rt.slots.load_speculative = failing_load
    try:
        _route_once(rt, spec, [0, 1])
        assert rt.prefetch_experts(layer, [6]) == 1
        bank = rt._banks[layer]
        assert bank.prefetch_ticket(6) is not None  # in flight, blocked
        before = rt.demand_bytes_read
        # release the read to FAIL shortly after the reconcile begins awaiting it.
        releaser = threading.Timer(0.15, gate.set)
        releaser.start()
        x, idx = _inputs(1, spec.top_k, spec.hidden_size, [6])
        out = _switch(rt, spec)(x, idx)
        _REAL_EVAL(out)
        rt.flush_deferred_slot_releases(evaluate=True)
        releaser.join()
        assert mx.array_equal(out_demand, out), "failure fallback changed the gather"
        assert rt.demand_bytes_read > before, "demand fallback bytes not accounted"
    finally:
        gate.set()
        _settle_prefetch(rt)
        rt.slots.load_speculative = real_load
        rt.close()
