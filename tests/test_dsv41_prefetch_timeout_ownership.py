"""CPU-only regression for a timed-out prefetch outliving its ring assignment."""
from __future__ import annotations

import threading
from concurrent.futures import Future

import pytest

from mtplx.expert_manifest import save_expert_manifest
from mtplx.expert_runtime import ExpertStreamingConfig, ExpertStreamingRuntime
from mtplx.expert_streaming import RoutingPhase

from test_expert_slots_runtime import _global_artifact


def _open_runtime(tmp_path, monkeypatch):
    root, spec, manifest, expected = _global_artifact(tmp_path, expert_count=3)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    monkeypatch.setenv("MTPLX_DSV41_GATE_PREFETCH_MAX_INFLIGHT", "2")
    config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=spec.resident_bytes + 10 * spec.expert_record_bytes,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        verify_artifact_headers=False,
        prefetch_slots=2,
        transient_slots=2,
        speculative_io_fraction=1.0,
    )
    return ExpertStreamingRuntime.open(
        root, manifest_path, config, spec=spec, apply_memory_cap=False
    ), expected


def test_timed_out_writer_cannot_overwrite_new_published_ring_expert(
    tmp_path, monkeypatch
):
    # Two tiny CPU slots and two workers reproduce the production topology.
    # Layer 1 occupies the first ring slot and is protected from layer 2's
    # replacement; the second slot is the contested assignment.
    runtime, expected = _open_runtime(tmp_path, monkeypatch)
    entered = threading.Event()
    release_old = threading.Event()
    callbacks = {expert: threading.Event() for expert in (0, 1)}
    submitted = {}
    real_load = runtime.slots.load_speculative
    real_submit = runtime._prefetch_executor.submit
    real_finish = runtime._finish_prefetch_load

    def pause_after_ticket_check(layer, load):
        if (layer, load.expert) == (2, 0):
            # _run_speculative_load already checked the ticket. A scheduling
            # pause here leaves the Future running but the physical slot free.
            entered.set()
            assert release_old.wait(2), "test did not release old worker"
        return real_load(layer, load)

    def capture_submit(fn, layer, load, ticket):
        future = real_submit(fn, layer, load, ticket)
        submitted[(layer, load.expert)] = future
        return future

    def finish(layer, expert, ticket, future):
        try:
            return real_finish(layer, expert, ticket, future)
        finally:
            callbacks[expert].set()

    def settle(expert):
        submitted[(2, expert)].result(timeout=2)
        assert callbacks[expert].wait(2), "prefetch callback did not settle"
        runtime.prefetch_experts(2, ())

    monkeypatch.setattr(runtime.slots, "load_speculative", pause_after_ticket_check)
    monkeypatch.setattr(runtime._prefetch_executor, "submit", capture_submit)
    monkeypatch.setattr(runtime, "_finish_prefetch_load", finish)
    runtime._prefetch_reconcile_timeout_s = 0
    try:
        protected = runtime._banks[1].plan_prefetch((0,))[0]
        real_load(1, protected)
        assert runtime._banks[1].commit_prefetch(0)

        assert runtime.prefetch_experts(2, (0,)) == 1
        assert entered.wait(2), "old worker did not pass its ticket check"
        old_future = submitted[(2, 0)]
        old_ring_slot = runtime._prefetch_ring._inflight[(2, 0)][0]
        with runtime._layer_locks[2]:
            runtime._reconcile_prefetch_for_route(2, (0,))
        assert not old_future.done(), "fixture did not exercise a live timeout"

        # Retaining an unpublished ring assignment must still allow immediate
        # demand I/O into a disjoint slot while its original worker is pending.
        with runtime._layer_locks[2]:
            demand = runtime._banks[2].plan((0,), phase=RoutingPhase.DECODE)
            assert demand.loads and old_ring_slot not in demand.slots
            ready = runtime.slots.ensure_route(2, demand)
            try:
                assert bytes(ready.bindings[0].buffer) == expected[(2, 0)]
                assert not old_future.done()
            finally:
                ready.release(synchronize=False)

        issued = runtime.prefetch_experts(2, (1,))
        if issued == 0:
            # Correct retention may skip the new prediction until the old
            # physical writer is terminal; it must become usable afterward.
            release_old.set()
            settle(0)
            assert runtime.prefetch_experts(2, (1,)) == 1
            settle(1)
        else:
            assert issued == 1
            settle(1)
            release_old.set()
            settle(0)

        published = runtime._prefetch_ring.published(2, (1,))
        assert 1 in published
        physical = runtime.slots._physical(2, published[1])
        actual_owner = (physical.layer, physical.expert)
        assert actual_owner == (2, 1), (
            f"stale timed-out writer replaced a published expert: "
            f"published={published}, physical_owner={actual_owner}"
        )
        assert bytes(physical.buffer) == expected[(2, 1)]
        plan = runtime._banks[2].plan((1,), phase=RoutingPhase.DECODE)
        assert plan.loads == (), "the new expert should be a ring hit"
        ready = runtime.slots.ensure_route(2, plan)
        ready.release(synchronize=False)
    finally:
        release_old.set()
        runtime.close(timeout=2)


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_terminal_prefetch_failure_releases_assignment(tmp_path, monkeypatch, outcome):
    runtime, _expected = _open_runtime(tmp_path, monkeypatch)
    try:
        bank = runtime._banks[2]
        bank.plan_prefetch((0,))
        ticket = bank.prefetch_ticket(0)
        future = Future()
        if outcome == "failed":
            future.set_exception(OSError("controlled speculative read failure"))
        else:
            assert future.cancel()
        # Pause the ordinary done-callback handoff: reconcile must independently
        # recognize terminal failure and release the assignment for another read.
        with runtime._prefetch_lock:
            runtime._prefetch_futures.add(future)
            runtime._prefetch_inflight_futures[(2, 0)] = future
        with runtime._layer_locks[2]:
            runtime._reconcile_prefetch_for_route(2, (0,))
            assert bank.prefetch_ticket(0) is None
            assert len(bank.plan_prefetch((0,))) == 1
            new_ticket = bank.prefetch_ticket(0)
            assert new_ticket != ticket
        runtime._finish_prefetch_load(2, 0, ticket, future)
        with runtime._layer_locks[2]:
            runtime._apply_prefetch_completions(2, bank)
        assert bank.prefetch_ticket(0) == new_ticket
        assert not bank.published_experts((0,))
    finally:
        runtime.close(timeout=2)


@pytest.mark.parametrize("succeeded", [True, False])
def test_completion_enqueued_after_reset_cannot_touch_new_ticket(
    tmp_path, monkeypatch, succeeded
):
    runtime, expected = _open_runtime(tmp_path, monkeypatch)
    before_append = threading.Event()
    allow_append = threading.Event()
    errors = []
    callback_thread = None

    class PauseBeforeCompletionAppend:
        def __init__(self):
            self.lock = threading.Lock()
            self.callback_entries = 0

        def __enter__(self):
            if threading.current_thread() is callback_thread:
                self.callback_entries += 1
                if self.callback_entries == 2:
                    before_append.set()
                    assert allow_append.wait(2), "test did not resume callback"
            self.lock.acquire()
            return self

        def __exit__(self, *_):
            self.lock.release()

    try:
        bank = runtime._banks[2]
        old_load = bank.plan_prefetch((0,))[0]
        old_ticket = bank.prefetch_ticket(0)
        # A callback runs only after its Future is terminal: the completed writer
        # no longer owns a pool lifecycle claim when the registry entry disappears.
        runtime.slots.load_speculative(2, old_load)
        future = Future()
        if succeeded:
            future.set_result(None)
        else:
            future.set_exception(OSError("controlled terminal failure"))
        runtime._prefetch_futures.add(future)
        runtime._prefetch_inflight_futures[(2, 0)] = future
        monkeypatch.setattr(runtime, "_prefetch_lock", PauseBeforeCompletionAppend())

        def finish():
            try:
                runtime._finish_prefetch_load(2, 0, old_ticket, future)
            except BaseException as exc:
                errors.append(exc)

        callback_thread = threading.Thread(target=finish)
        callback_thread.start()
        assert before_append.wait(2)
        assert not runtime._prefetch_futures
        assert not runtime._prefetch_completions

        runtime.reset()
        new_load = bank.plan_prefetch((0,))[0]
        new_ticket = bank.prefetch_ticket(0)
        assert new_load.slot == old_load.slot
        assert new_ticket != old_ticket

        allow_append.set()
        callback_thread.join(2)
        assert not callback_thread.is_alive()
        assert not errors
        with runtime._layer_locks[2]:
            runtime._apply_prefetch_completions(2, bank)
        assert bank.prefetch_ticket(0) == new_ticket
        assert not bank.published_experts((0,))
        runtime.slots.load_speculative(2, new_load)
        assert bank.commit_prefetch(0, ticket=new_ticket)
        physical = runtime.slots._physical(2, new_load.slot)
        assert bytes(physical.buffer) == expected[(2, 0)]
    finally:
        allow_append.set()
        if callback_thread is not None:
            callback_thread.join(2)
        runtime.close(timeout=2)
