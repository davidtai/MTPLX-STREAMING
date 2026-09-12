"""W93 LANE B -- GlobalPrefetchRing self-starvation + receipt-integrity fixes.

Locks the lane-B half of the adversarial-review fix of the DSV4.1 gate-oracle
prefetch (docs/deepseek-v41/W93_GATE_PREFETCH.md), all CPU-pinned on the
fake component-bank runtime (no GPU, no real weights):

  * **HIGH-2 (self-starving ring)** -- a speculative read that settles AFTER its
    own layer's reconcile must not pin a shared ring slot for a whole token:
    ``prefetch_experts`` drains EVERY layer's settled completions (not only the
    target) before planning, and the dropped/skipped predictions are now counted
    (``dropped_no_slot`` / ``skipped_lock_held`` / ``skipped_backlog``).
  * **MED-b (receipt integrity)** -- an awaited-inflight commit is folded into the
    reported ``committed`` so ``hit_rate <= 1.0``; a wasted read is charged to the
    VICTIM's layer (the layer that predicted it), not the evicting caller.
  * **Lane-C ring rules** -- ``plan_prefetch(target)`` never evicts an entry of
    layer ``target - 1`` (its true route has not run yet), and honours an
    optional ``is_slot_pinned`` hook.

Run under ``nice -n 19`` and without ``pytest -n auto`` (one test file per
process).  Ring-class tests are pure; runtime tests use the barrier tests'
fake-but-real component-bank artifact, extended to multiple routed layers.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import pytest

pytest.importorskip("mlx.core")

mx.set_default_device(mx.cpu)

from mlx.utils import tree_flatten  # noqa: E402

from mtplx.expert_manifest import (  # noqa: E402
    build_expert_manifest,
    load_expert_manifest,
    save_expert_manifest,
)
from mtplx.expert_runtime import (  # noqa: E402
    ExpertStreamingConfig,
    ExpertStreamingRuntime,
)
from mtplx.expert_streaming import GlobalPrefetchRing  # noqa: E402
from mtplx.expert_streaming_models import ExpertStreamingModelSpec  # noqa: E402
from mtplx.models.expert_mlx import (  # noqa: E402
    HotExpertSwitchGLU,
    make_mlx_component_bank_allocator,
)
from mtplx.models.hy3_mlx import Model as Hy3Model  # noqa: E402
from mtplx.models.hy3_mlx import ModelArgs as Hy3Args  # noqa: E402

_REAL_EVAL = mx.eval


@pytest.fixture(autouse=True)
def _cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(prev)


# ===========================================================================
# Pure GlobalPrefetchRing tests (no runtime, no MLX device work)
# ===========================================================================
def _commit_all(ring: GlobalPrefetchRing, layer: int, loads) -> None:
    for load in loads:
        assert ring.commit_prefetch(layer, load.expert)


def test_target_minus_one_entry_never_evicted() -> None:
    """``plan_prefetch(target)`` must never recycle a slot held by layer
    ``target - 1`` -- that layer's true route has not run yet and is about to
    consume its prediction (the lane-C ring rule)."""
    ring = GlobalPrefetchRing(ring_size=2, base=100, expert_count=32)
    for layer in (4, 5):
        ring.note_decode(layer)
    # Fill both slots with layer 4's committed predictions.
    loads = ring.plan_prefetch(4, [10, 11])
    assert len(loads) == 2
    _commit_all(ring, 4, loads)
    # Prefetch for target=5: target-1 == 4, so 4's entries are protected and the
    # ring has no evictable slot -> zero loads, both predictions dropped.
    assert ring.plan_prefetch(5, [20, 21]) == ()
    assert ring.published(4, [10, 11]) == {10: 100, 11: 101}
    assert ring.prefetch_skip_snapshot()["dropped_no_slot"] == {5: 2}


def test_pinned_victim_is_skipped() -> None:
    """A slot the caller's ``is_slot_pinned`` reports pinned is never a ring
    victim; unpinning one slot frees exactly one load."""
    ring = GlobalPrefetchRing(ring_size=2, base=50, expert_count=32)
    for layer in (1, 3):  # non-adjacent so target-1 protection is inert here
        ring.note_decode(layer)
    _commit_all(ring, 1, ring.plan_prefetch(1, [10, 11]))
    pinned = {50, 51}
    assert ring.plan_prefetch(3, [20, 21], is_slot_pinned=lambda s: s in pinned) == ()
    # release one physical slot -> one prediction gets a slot, the other does not.
    loads = ring.plan_prefetch(3, [20, 21], is_slot_pinned=lambda s: s == 50)
    assert len(loads) == 1
    assert loads[0].slot == 51


def test_wasted_attributed_to_victim_layer_not_evicting_layer() -> None:
    """A committed-but-never-consumed entry evicted by another layer's prefetch
    is charged to the VICTIM's layer (MED-b)."""
    ring = GlobalPrefetchRing(ring_size=2, base=0, expert_count=32)
    for layer in (1, 3):
        ring.note_decode(layer)
    _commit_all(ring, 1, ring.plan_prefetch(1, [10, 11]))
    # Layer 3 (target-1 == 2, disjoint) evicts layer 1's unused entries.
    assert len(ring.plan_prefetch(3, [20, 21])) == 2
    assert ring.consume_wasted_by_layer() == {1: 2}
    # int accessor stays intact for the bank wrapper / legacy callers.
    assert ring.consume_wasted() == 0


def test_dropped_no_slot_counted_per_target_layer() -> None:
    """A prediction that finds no eligible slot increments ``dropped_no_slot``
    for the TARGET layer."""
    ring = GlobalPrefetchRing(ring_size=2, base=0, expert_count=32)
    ring.note_decode(7)
    # Two inflight reads fill the ring; a third target's predictions are dropped.
    assert len(ring.plan_prefetch(7, [10, 11])) == 2  # both inflight, unrecyclable
    assert ring.plan_prefetch(7, [12, 13]) == ()
    assert ring.prefetch_skip_snapshot()["dropped_no_slot"] == {7: 2}


def test_cross_layer_wrap_recycles_consumed_slots_over_40_layers() -> None:
    """A shared ring parametrized with 40 routed layers recycles each layer's
    consumed slots as decode wraps forward: every one-step-ahead prediction lands
    (no self-starvation), target-1 is protected throughout, and nothing is
    miscounted as wasted or dropped."""
    n_layers = 40
    expert_count = 64
    ring = GlobalPrefetchRing(ring_size=4, base=0, expert_count=expert_count)

    def experts(layer: int) -> list[int]:
        return [(2 * layer) % expert_count, (2 * layer + 1) % expert_count]

    max_resident = 0
    for cur in range(n_layers):
        ring.note_decode(cur)
        target = cur + 1
        if target < n_layers:
            # Predict target's route during cur's forward.
            loads = ring.plan_prefetch(target, experts(target))
            assert len(loads) == 2, f"layer {target} starved: {loads}"
            _commit_all(ring, target, loads)
            # target-1 protection: cur's committed prediction survives planning
            # target, since cur's own true route has not run yet.
            if cur >= 1:
                assert ring.published(cur, experts(cur)) == {
                    e: s for e, s in ring.published(cur, experts(cur)).items()
                }
                assert len(ring.published(cur, experts(cur))) == 2, (
                    f"target-1 layer {cur} evicted while planning {target}"
                )
        # cur's own true route now runs and consumes cur's prediction.
        consumed = ring.published(cur, experts(cur))
        ring.mark_used(cur, consumed.keys())
        resident = len(ring._key_to_slot) + len(ring._inflight)
        max_resident = max(max_resident, resident)
        assert resident <= ring.ring_size, f"ring overflowed at layer {cur}"

    assert ring._cursor >= ring.ring_size, "round-robin never wrapped"
    assert max_resident == ring.ring_size, "ring never filled -- no real wrap"
    assert ring.consume_wasted_by_layer() == {}, "consumed slots miscounted wasted"
    assert ring.prefetch_skip_snapshot()["dropped_no_slot"] == {}


# ===========================================================================
# Runtime tests on the fake component-bank runtime
# ===========================================================================
def _hy3_artifact(tmp_path: Path, *, expert_count=8, top_k=2, routed_layer_count=1):
    """A tiny hy3 component-bank artifact with ``routed_layer_count`` routed
    layers (layer 0 dense, layers 1..N routed).  Mirrors the barrier tests'
    single-layer ``_integrated_hy3_artifact`` extended across layers."""
    total_layers = routed_layer_count + 1
    args = Hy3Args(
        model_type="hy_v3",
        hidden_size=64,
        num_hidden_layers=total_layers,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=expert_count,
        num_experts_per_tok=top_k,
        num_shared_experts=1,
        first_k_dense_replace=1,
        rms_norm_eps=1e-5,
        vocab_size=128,
        max_position_embeddings=128,
        head_dim=16,
        router_scaling_factor=2.0,
    )
    model = Hy3Model(args)
    weights = dict(tree_flatten(model.parameters()))
    shapes = {
        "gate_proj.weight": (expert_count, 64, 8),
        "gate_proj.scales": (expert_count, 64, 1),
        "gate_proj.biases": (expert_count, 64, 1),
        "up_proj.weight": (expert_count, 64, 8),
        "up_proj.scales": (expert_count, 64, 1),
        "up_proj.biases": (expert_count, 64, 1),
        "down_proj.weight": (expert_count, 64, 8),
        "down_proj.scales": (expert_count, 64, 1),
        "down_proj.biases": (expert_count, 64, 1),
    }
    for layer in range(1, total_layers):
        for component, shape in shapes.items():
            dtype = mx.uint32 if component.endswith("weight") else mx.bfloat16
            value = (
                mx.ones(shape, dtype=dtype)
                if component.endswith("scales")
                else mx.zeros(shape, dtype=dtype)
            )
            weights[f"model.layers.{layer}.mlp.switch_mlp.{component}"] = value
    mx.eval(weights)
    root = Path(tmp_path) / "hy3"
    root.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    config = asdict(args)
    config["model_type"] = "hy_v3"
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    total_bytes = sum(int(value.nbytes) for value in weights.values())
    spec = ExpertStreamingModelSpec(
        key="tiny-hy3-q4",
        display_name="Tiny Hy3 Q4",
        source_model="test/tiny-hy3",
        source_revision="source",
        quant_model="test/tiny-hy3-q4",
        quant_revision="quant",
        total_tensor_bytes=total_bytes,
        total_layers=total_layers,
        routed_layer_start=1,
        routed_layer_count=routed_layer_count,
        expert_count=expert_count,
        top_k=top_k,
        hidden_size=64,
        expert_hidden_size=64,
        quant_bits=4,
        quant_group_size=64,
        quant_parameter_bytes=2,
        router_storage="float32",
        router_matmul_dtype="float32",
        router_bytes=expert_count * 64 * 4 + expert_count * 4,
        kv_bytes_per_token=0,
        mtp_layer_index=total_layers,
        mtp_included=False,
    )
    manifest = build_expert_manifest(root, spec)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    return root, spec, manifest_path


def _open_runtime(
    tmp_path,
    *,
    expert_count=8,
    top_k=2,
    resident_slots=8,
    transient=4,
    prefetch=10,
    routed_layer_count=1,
):
    root, spec, manifest_path = _hy3_artifact(
        tmp_path,
        expert_count=expert_count,
        top_k=top_k,
        routed_layer_count=routed_layer_count,
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


def _settle_prefetch(rt) -> None:
    with rt._prefetch_lock:
        pending = tuple(rt._prefetch_futures)
    for future in pending:
        try:
            future.result()
        except BaseException:
            pass


def _route_once(rt, spec, layer, experts) -> None:
    sw = HotExpertSwitchGLU(rt, layer)
    for e in experts:
        idx = mx.array([[e] * spec.top_k], dtype=mx.int32)
        x = mx.zeros((1, 1, spec.hidden_size), dtype=mx.bfloat16)
        _REAL_EVAL(sw(x, idx))
    rt.flush_deferred_slot_releases(evaluate=True)


def _decode_inputs(rows, top_k, hidden, experts):
    mx.random.seed(93 + rows)
    x = (0.3 * mx.random.normal((rows, 1, hidden))).astype(mx.bfloat16)
    flat = [experts[i % len(experts)] for i in range(rows * top_k)]
    idx = mx.array(flat, dtype=mx.int32).reshape((rows, top_k))
    _REAL_EVAL(x, idx)
    return x, idx


def test_late_completion_of_other_layer_does_not_pin_shared_ring(tmp_path) -> None:
    """HIGH-2: a layer-A read that settles after A's own route must not pin a
    slot of the SHARED ring so that a later layer B cannot prefetch.

    ring_size 2; issue layer A with a gated slow reader; run layer A's reconcile
    needing NEITHER prediction; release the gate AFTER that route so A's reads
    settle unapplied; then two predictions for a non-adjacent layer B must both
    issue (they can only do so once A's stuck completions are drained -- 0 before
    the fix, 2 after)."""
    layer_a, layer_b = 1, 3  # non-adjacent: B's target-1 (2) never shields A
    rt, spec = _open_runtime(
        tmp_path,
        expert_count=8,
        top_k=2,
        resident_slots=2,
        transient=4,
        prefetch=2,  # ring_size 2
        routed_layer_count=3,
    )
    gate = threading.Event()
    real_load = rt.slots.load_speculative

    def gated_load(layer, load):
        gate.wait(timeout=5.0)
        return real_load(layer, load)

    rt.slots.load_speculative = gated_load
    try:
        # 1. layer A: two gated speculative reads fill the 2-slot ring (inflight).
        assert rt.prefetch_experts(layer_a, [6, 7]) == 2
        bank_a = rt._banks[layer_a]
        assert bank_a.prefetch_ticket(6) is not None
        assert bank_a.prefetch_ticket(7) is not None
        # 2. layer A's route needs NEITHER 6 nor 7 (reads still gated/inflight).
        lock_a = rt._layer_locks[layer_a]
        lock_a.acquire()
        try:
            rt._reconcile_prefetch_for_route(layer_a, (0, 1))
        finally:
            lock_a.release()
        # 3. release the gate AFTER the route -> reads settle, queue as A's
        #    completions, but A's slots stay INFLIGHT (A is not revisited).
        gate.set()
        _settle_prefetch(rt)
        with rt._prefetch_lock:
            assert len(rt._prefetch_completions.get(layer_a, [])) == 2
        assert bank_a.prefetch_ticket(6) is not None  # still inflight in the ring
        # 4. two predictions for layer B must both issue -- only possible if
        #    prefetch_experts drained A's settled completions first.
        assert rt.prefetch_experts(layer_b, [4, 5]) == 2
        # the evicted, never-consumed A entries are charged to layer A, not B.
        snap = rt.resource_telemetry_snapshot()["gate_prefetch"]
        assert snap["per_layer"][str(layer_a)]["wasted"] == 2
        assert snap["per_layer"].get(str(layer_b), {}).get("wasted", 0) == 0
    finally:
        gate.set()
        rt.slots.load_speculative = real_load
        rt.close()


def test_hit_rate_at_most_one_with_awaited_commits(tmp_path) -> None:
    """MED-b: an awaited-inflight commit is a settled+published read and must be
    folded into ``committed`` so ``hit_rate <= 1``.  A route that consumes one
    async-committed prefetch AND one awaited-inflight prefetch produces two hits
    against one async commit -- hit_rate would read 2.0 without the fold."""
    rt, spec = _open_runtime(
        tmp_path, expert_count=8, top_k=2, resident_slots=2, transient=4, prefetch=10
    )
    layer = spec.routed_layer_start
    gate = threading.Event()
    gate.set()  # open: expert 6's read passes straight through
    real_load = rt.slots.load_speculative

    def gated_load(l, load):
        gate.wait(timeout=5.0)
        return real_load(l, load)

    rt.slots.load_speculative = gated_load
    try:
        _route_once(rt, spec, layer, [0, 1])  # 0,1 persistent; 6,7 cold
        # expert 6: prefetch + settle -> its completion is queued (async path).
        assert rt.prefetch_experts(layer, [6]) == 1
        _settle_prefetch(rt)
        # expert 7: gate it so its read blocks in flight.
        gate.clear()
        assert rt.prefetch_experts(layer, [7]) == 1  # applies 6 (committed++), 7 inflight
        # the true route needs BOTH 6 (committed) and 7 (awaited in flight).
        releaser = threading.Timer(0.15, gate.set)
        releaser.start()
        x, idx = _decode_inputs(1, spec.top_k, spec.hidden_size, [6, 7])
        out = HotExpertSwitchGLU(rt, layer)(x, idx)
        _REAL_EVAL(out)
        rt.flush_deferred_slot_releases(evaluate=True)
        releaser.join()
        snap = rt.resource_telemetry_snapshot()["gate_prefetch"]
        assert snap["awaited_inflight"] >= 1, "no awaited-inflight commit occurred"
        assert snap["hit_on_true_route"] >= 2, "both prefetched experts should hit"
        # the fix: committed includes the awaited commit, so hit <= committed <= 1.
        assert snap["hit_on_true_route"] <= snap["committed"], (
            f"hit {snap['hit_on_true_route']} > committed {snap['committed']}"
        )
        assert snap["hit_rate"] <= 1.0, f"hit_rate {snap['hit_rate']} exceeds 1.0"
        assert "hit=" in snap["census"] and "rate " in snap["census"]
    finally:
        gate.set()
        rt.slots.load_speculative = real_load
        rt.close()


def test_skip_and_drop_counters_exposed_in_gate_prefetch_block(tmp_path) -> None:
    """HIGH-2 receipt: the three reasons a prediction never issues --
    ``dropped_no_slot`` (ring full), ``skipped_lock_held`` (~3827 hazard) and
    ``skipped_backlog`` (read backlog full) -- are counted and surfaced in the
    gate_prefetch block (top-level and per-layer)."""
    rt, spec = _open_runtime(
        tmp_path, expert_count=8, top_k=2, resident_slots=2, transient=4, prefetch=2
    )
    layer = spec.routed_layer_start
    gate = threading.Event()  # closed: reads block, so the ring stays inflight-full
    real_load = rt.slots.load_speculative

    def gated_load(l, load):
        gate.wait(timeout=5.0)
        return real_load(l, load)

    rt.slots.load_speculative = gated_load
    try:
        _route_once(rt, spec, layer, [0, 1])
        # dropped_no_slot: fill the 2-slot ring with inflight reads, then a fresh
        # pair finds no evictable slot.
        assert rt.prefetch_experts(layer, [4, 5]) == 2
        assert rt.prefetch_experts(layer, [6, 7]) == 0

        # skipped_lock_held: the layer lock is held (deferred-split hazard).
        lock = rt._layer_locks[layer]
        lock.acquire()
        try:
            assert rt.prefetch_experts(layer, [3]) == 0
        finally:
            lock.release()

        # skipped_backlog: two reads are in flight; drop the ceiling below that.
        rt._prefetch_backlog_limit = 1
        assert rt.prefetch_experts(layer, [2]) == 0

        gp = rt.resource_telemetry_snapshot()["gate_prefetch"]
        assert gp["dropped_no_slot"] >= 2
        assert gp["skipped_lock_held"] >= 1
        assert gp["skipped_backlog"] >= 1
        per_layer = gp["per_layer"][str(layer)]
        assert per_layer["dropped_no_slot"] >= 2
        assert per_layer["skipped_lock_held"] >= 1
        assert per_layer["skipped_backlog"] >= 1
        assert "dropped=" in gp["census"] and "skipped_lock=" in gp["census"]
    finally:
        gate.set()
        _settle_prefetch(rt)
        rt.slots.load_speculative = real_load
        rt.close()
