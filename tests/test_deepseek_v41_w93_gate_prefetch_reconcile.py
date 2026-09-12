"""W93 -- gate-oracle prefetch RECONCILE + ring, on a REAL streamed runtime.

Switch-level proof of the runtime half of the DSV4.1 one-layer-ahead gate-oracle
prefetch (docs/deepseek-v41/W93_GATE_PREFETCH.md).  Reuses the barrier tests'
fake-but-real component-bank runtime (``_integrated_hy3_artifact`` +
``ExpertStreamingRuntime`` + ``HotExpertSwitchGLU``) with a ``prefetch_slots``
ring armed.  These lock:

  1. **Exact by construction** -- the switch gather is byte-identical with a
     gate-oracle prediction stashed (ring warmed / reads issued) vs not, at M=1
     (AR) and M=4 (the MTP verify row batch), because the gather always uses the
     TRUE ``indices`` and the ring only warms the cache;
  2. **No new host sync** -- ``_run`` performs exactly ONE generation-thread
     ``mx.eval`` whether or not a prediction rides its indices barrier;
  3. **hit_on_true_route** -- a prefetched expert the next true route needs
     resolves as a ring hit and increments the counter (no re-read);
  4. **wasted** -- a prefetched expert no route needs is evicted round-robin and
     counts as wasted;
  5. **awaited_inflight** -- a true route needing a still-in-flight prefetch
     awaits that read and commits it instead of issuing a duplicate;
  6. **pin/lock safety** -- prefetch never evicts a persistent resident, and
     ``prefetch_experts`` skips (returns 0, never blocks) when the layer lock is
     held (the deferred-split ~3827 hazard).

CPU-pinned; tiny synthetic artifact; no GPU, no real weights.  Run under
``nice -n 19`` and without ``pytest -n auto``.
"""

from __future__ import annotations

import threading
from pathlib import Path

import mlx.core as mx
import pytest

pytest.importorskip("mlx.core")

mx.set_default_device(mx.cpu)

import mtplx.models.expert_mlx as expert_mlx  # noqa: E402
from mtplx.expert_manifest import load_expert_manifest  # noqa: E402
from mtplx.expert_runtime import (  # noqa: E402
    ExpertStreamingConfig,
    ExpertStreamingRuntime,
    RoutingPhase,
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
    # budget the persistent tier + a comfortable pad for the transient and W93
    # prefetch-ring slots (each ~one expert record).
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
# 1. exact by construction: a stashed prediction never changes the gather
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rows", [1, 4])
def test_gather_byte_identical_with_prediction_stashed(tmp_path, rows) -> None:
    experts = [0, 1, 2, 3]
    x = idx = None
    # cold run (no prediction)
    rt, spec = _open_runtime(tmp_path / "a", expert_count=8, top_k=2)
    try:
        _route_once(rt, spec, experts)
        x, idx = _inputs(rows, spec.top_k, spec.hidden_size, experts)
        out_off = _switch(rt, spec)(x, idx)
        _REAL_EVAL(out_off)
        rt.flush_deferred_slot_releases(evaluate=True)
    finally:
        rt.close()
    # identical run WITH a gate-oracle prediction stashed on the switch (issues a
    # speculative prefetch for experts 4,5 -- never consumed by THIS gather).
    rt2, spec2 = _open_runtime(tmp_path / "b", expert_count=8, top_k=2)
    try:
        _route_once(rt2, spec2, experts)
        x2, idx2 = _inputs(rows, spec2.top_k, spec2.hidden_size, experts)
        sw = _switch(rt2, spec2)
        sw._mtplx_gate_prefetch_pending = (
            _layer(spec2),
            mx.array([[4, 5]], dtype=mx.int32),
        )
        out_on = sw(x2, idx2)
        _REAL_EVAL(out_on)
        rt2.flush_deferred_slot_releases(evaluate=True)
        assert mx.array_equal(out_off, out_on), f"M={rows}: prediction changed gather"
        # the prediction was actually issued (mechanism engaged, not a no-op)
        _settle_prefetch(rt2)
        assert rt2.counters.prefetch_issued >= 1
    finally:
        rt2.close()


# ---------------------------------------------------------------------------
# 1b. verify shape (M=4): a route SERVED FROM THE RING is byte-identical to the
#     same route served on demand -- the ring changes no gathered value, so a
#     DSpark verify batch that consumes prefetched experts is exact.
# ---------------------------------------------------------------------------
def test_verify_m4_ring_served_byte_identical_to_demand(tmp_path) -> None:
    experts = [6, 7]  # never persistent (resident_slots=2 warms 0,1)
    # demand: experts streamed on the true route itself.
    rt, spec = _open_runtime(
        tmp_path / "d", expert_count=8, top_k=2, resident_slots=2, transient=8,
        prefetch=10,
    )
    try:
        _route_once(rt, spec, [0, 1])
        x, idx = _inputs(4, spec.top_k, spec.hidden_size, experts)
        out_demand = _switch(rt, spec)(x, idx)
        _REAL_EVAL(out_demand)
        rt.flush_deferred_slot_releases(evaluate=True)
    finally:
        rt.close()
    # ring-served: the exact same experts are prefetched (committed) first, so the
    # M=4 true route consumes them as ring hits.
    rt2, spec2 = _open_runtime(
        tmp_path / "r", expert_count=8, top_k=2, resident_slots=2, transient=8,
        prefetch=10,
    )
    try:
        _route_once(rt2, spec2, [0, 1])
        assert rt2.prefetch_experts(_layer(spec2), experts) == 2
        _settle_prefetch(rt2)
        before = rt2.counters.prefetch_hit_on_true_route
        x2, idx2 = _inputs(4, spec2.top_k, spec2.hidden_size, experts)
        out_ring = _switch(rt2, spec2)(x2, idx2)
        _REAL_EVAL(out_ring)
        rt2.flush_deferred_slot_releases(evaluate=True)
        assert rt2.counters.prefetch_hit_on_true_route > before, "ring not consumed"
        assert mx.array_equal(out_demand, out_ring), "ring-served M=4 != demand"
    finally:
        rt2.close()


# ---------------------------------------------------------------------------
# 2. no new host sync: one generation-thread mx.eval whether or not a prediction
#    rides the indices barrier
# ---------------------------------------------------------------------------
def _count_main_thread_evals(rt, spec, x, idx, *, pending):
    n = {"eval": 0}
    real = expert_mlx.mx.eval
    main = threading.main_thread()

    def counting_eval(*a, **k):
        if threading.current_thread() is main:
            n["eval"] += 1
        return real(*a, **k)

    sw = _switch(rt, spec)
    if pending is not None:
        sw._mtplx_gate_prefetch_pending = pending
    expert_mlx.mx.eval = counting_eval
    try:
        out = sw(x, idx)
    finally:
        expert_mlx.mx.eval = real
    real(out)
    rt.flush_deferred_slot_releases(evaluate=True)
    return n["eval"]


def test_prediction_adds_no_host_sync(tmp_path) -> None:
    experts = [0, 1, 2, 3]
    rt, spec = _open_runtime(tmp_path, expert_count=8, top_k=2)
    try:
        _route_once(rt, spec, experts)
        x, idx = _inputs(1, spec.top_k, spec.hidden_size, experts)
        without = _count_main_thread_evals(rt, spec, x, idx, pending=None)
        with_pred = _count_main_thread_evals(
            rt, spec, x, idx,
            pending=(_layer(spec), mx.array([[4, 5]], dtype=mx.int32)),
        )
        # The fenced all-hit path's own generation-thread eval count (indices
        # barrier + wave fence) is the baseline; the claim (item 2) is that the
        # prediction rides the indices barrier and adds ZERO extra host syncs.
        assert without >= 1
        assert with_pred == without, (
            f"prediction added host syncs: {with_pred} vs {without}"
        )
    finally:
        rt.close()


# ---------------------------------------------------------------------------
# 3. hit_on_true_route: a prefetched expert the next route needs is a ring hit
# ---------------------------------------------------------------------------
def test_prefetch_hit_on_true_route(tmp_path) -> None:
    layer = None
    # resident_slots small so experts 6,7 are never persistent -> the only way
    # they are resident for the true route is via the prefetch ring.
    rt, spec = _open_runtime(
        tmp_path, expert_count=8, top_k=2, resident_slots=2, transient=4, prefetch=10
    )
    layer = _layer(spec)
    try:
        _route_once(rt, spec, [0, 1])  # 0,1 persistent; 6,7 cold
        before = rt.counters.prefetch_hit_on_true_route
        issued = rt.prefetch_experts(layer, [6, 7])
        assert issued == 2, f"expected 2 speculative reads, got {issued}"
        _settle_prefetch(rt)
        # a true route needing 6,7 (M=1, both slots) -> ring hits.
        x, idx = _inputs(1, spec.top_k, spec.hidden_size, [6, 7])
        out = _switch(rt, spec)(x, idx)
        _REAL_EVAL(out)
        rt.flush_deferred_slot_releases(evaluate=True)
        gained = rt.counters.prefetch_hit_on_true_route - before
        assert gained >= 1, f"prefetch commits were not consumed as hits ({gained})"
        # committed reads accounted, bytes non-zero, snapshot census present.
        snap = rt.resource_telemetry_snapshot()
        gp = snap["gate_prefetch"]
        assert gp["issued"] >= 2 and gp["bytes_prefetched"] >= 2 * spec.expert_record_bytes
        assert gp["hit_on_true_route"] >= 1
        assert "census" in gp and gp["census"].startswith("gate_prefetch k=")
    finally:
        rt.close()


# ---------------------------------------------------------------------------
# 4. wasted: a prefetched expert no route needs is evicted round-robin -> wasted
# ---------------------------------------------------------------------------
def test_prefetch_wasted_on_unused_eviction(tmp_path) -> None:
    # prefetch ring width 2; predict 2 experts, then predict 2 fresh experts ->
    # the first pair is recycled without ever being consumed by a true route.
    rt, spec = _open_runtime(
        tmp_path, expert_count=8, top_k=2, resident_slots=2, transient=4, prefetch=2
    )
    layer = _layer(spec)
    try:
        _route_once(rt, spec, [0, 1])
        assert rt.prefetch_experts(layer, [4, 5]) == 2
        _settle_prefetch(rt)
        rt.prefetch_experts(layer, [])  # flush completions -> commit 4,5
        _settle_prefetch(rt)
        # fresh predictions recycle the (never-consumed) ring slots holding 4,5.
        assert rt.prefetch_experts(layer, [6, 7]) == 2
        _settle_prefetch(rt)
        rt.prefetch_experts(layer, [])
        assert rt.counters.prefetch_wasted >= 2, (
            f"unused ring evictions not counted as wasted "
            f"({rt.counters.prefetch_wasted})"
        )
    finally:
        rt.close()


# ---------------------------------------------------------------------------
# 5. awaited_inflight: a true route awaits a still-in-flight prefetch (no dup)
# ---------------------------------------------------------------------------
def test_awaits_inflight_prefetch_instead_of_reissue(tmp_path) -> None:
    rt, spec = _open_runtime(
        tmp_path, expert_count=8, top_k=2, resident_slots=2, transient=4, prefetch=10
    )
    layer = _layer(spec)
    gate = threading.Event()
    real_load = rt.slots.load_speculative

    def gated_load(l, load):
        gate.wait(timeout=5.0)  # hold the read in flight until released
        return real_load(l, load)

    rt.slots.load_speculative = gated_load
    try:
        _route_once(rt, spec, [0, 1])
        issued = rt.prefetch_experts(layer, [6])
        assert issued == 1
        # 6's read is now in flight (blocked on the gate); it is not yet committed.
        bank = rt._banks[layer]
        assert bank.prefetch_ticket(6) is not None
        assert 6 not in bank.published_experts([6])
        # release the read shortly after the reconcile begins to await it.
        releaser = threading.Timer(0.15, gate.set)
        releaser.start()
        lock = rt._layer_locks[layer]
        lock.acquire()
        try:
            rt._reconcile_prefetch_for_route(layer, (6,))
        finally:
            lock.release()
        releaser.join()
        assert rt.counters.prefetch_awaited_inflight >= 1, "inflight read not awaited"
        # after awaiting, 6 is committed and would hit-resolve (no duplicate read).
        assert 6 in bank.published_experts([6])
    finally:
        gate.set()
        rt.slots.load_speculative = real_load
        rt.close()


# ---------------------------------------------------------------------------
# 6a. pin/lock: prefetch never evicts a persistent resident
# ---------------------------------------------------------------------------
def test_prefetch_never_evicts_persistent_resident(tmp_path) -> None:
    rt, spec = _open_runtime(
        tmp_path, expert_count=8, top_k=2, resident_slots=4, transient=4, prefetch=4
    )
    layer = _layer(spec)
    try:
        _route_once(rt, spec, [0, 1, 2, 3])  # persistent
        bank = rt._banks[layer]
        assert set([0, 1, 2, 3]).issubset(set(bank.resident_experts))
        # fill and churn the ring with cold experts.
        for pair in ([4, 5], [6, 7], [4, 6]):
            rt.prefetch_experts(layer, pair)
            _settle_prefetch(rt)
            rt.prefetch_experts(layer, [])
        # the earned persistent residents are untouched by ring turnover.
        assert set([0, 1, 2, 3]).issubset(set(bank.resident_experts))
    finally:
        rt.close()


# ---------------------------------------------------------------------------
# 6b. pin/lock: prefetch skips (returns 0, never blocks) when the layer lock is
#     held -- the deferred-split ~3827 hazard
# ---------------------------------------------------------------------------
def test_prefetch_skips_when_layer_lock_held(tmp_path) -> None:
    rt, spec = _open_runtime(tmp_path, expert_count=8, top_k=2, prefetch=10)
    layer = _layer(spec)
    try:
        _route_once(rt, spec, [0, 1])
        lock = rt._layer_locks[layer]
        lock.acquire()  # simulate a deferred split holding the layer transaction
        try:
            issued = rt.prefetch_experts(layer, [4, 5])
        finally:
            lock.release()
        assert issued == 0, "prefetch must skip (not block) when the lock is held"
    finally:
        rt.close()
