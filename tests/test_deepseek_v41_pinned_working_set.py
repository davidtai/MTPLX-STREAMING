"""W64 (R3-pin) -- post-prefill PINNED WORKING SET.

CPU-pinned, pure-Python bank policy + a tiny MLX slot-memory gather model. No GPU,
no artifact, no real ``experts.bin``; MLX pinned to the CPU device
(memory/worker-tests-must-pin-mlx-cpu.md). Run under ``nice -n 19`` and without
``pytest -n auto``.

The lever (env ``MTPLX_DSV41_PIN_WORKING_SET``, default off): after prefill, rank
each layer's resident experts by prefill routing frequency and pin the top-K so
their persistent slot is never recycled by decode LRU admission. That makes the
pinned slots STATIC for the whole decode -- the safety property the W44 barrier-
free device route needs (its deferred gather cannot race a slot recycle for a
pinned expert; W44_DEVICE_ROUTE.md §8).

These tests lock:
  1. pinned experts survive a churn sequence that evicts the unpinned free tail;
  2. decode misses still admit into the free tail (pinned untouched);
  3. ``pin_working_set(experts=all)`` (the ``pin_ws`` arm) freezes the layer:
     no cold decode expert can take a persistent slot, ``pinned_static`` holds,
     ``route_all_pinned`` is True for any resident route;
  4. the working set is ranked by prefill routing frequency;
  5. a memory-forced capacity eviction MAY evict a pinned expert (memory is the
     hard constraint) and unpins it;
  6. the served-expert gather is BYTE-IDENTICAL with pinning on vs off over a
     churny decode sequence -- modelling the physical bank, so a wrong slot
     (a recycle-before-read) would surface as a wrong expert;
  7. the runtime pin hook pins after prefill, refreshes on cadence, and the
     snapshot / telemetry reports pinned-count-per-layer + all-pinned-hit rate.
"""

from __future__ import annotations

import os
import threading

import mlx.core as mx
import pytest

pytest.importorskip("mlx.core")

mx.set_default_device(mx.cpu)

from mtplx.expert_runtime import (  # noqa: E402
    ExpertStreamingRuntime,
    parse_pin_refresh_tokens,
    parse_pin_working_set,
)
from mtplx.expert_streaming import LayerExpertSlotBank, RoutingPhase  # noqa: E402

PIN_ENV = "MTPLX_DSV41_PIN_WORKING_SET"
REFRESH_ENV = "MTPLX_DSV41_PIN_REFRESH_TOKENS"


@pytest.fixture(autouse=True)
def _cpu_and_clean_env():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    saved = {k: os.environ.get(k) for k in (PIN_ENV, REFRESH_ENV)}
    for k in (PIN_ENV, REFRESH_ENV):
        os.environ.pop(k, None)
    try:
        yield
    finally:
        mx.set_default_device(prev)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _bank(*, persistent_slots=4, transient_slots=4, expert_count=64, policy="frequency"):
    # frequency_decay=1.0 so scores are exact route counts (deterministic ranking).
    return LayerExpertSlotBank(
        expert_count=expert_count,
        persistent_slots=persistent_slots,
        transient_slots=transient_slots,
        frequency_decay=1.0,
        cache_policy=policy,
    )


# ---------------------------------------------------------------------------
# 1. pinned experts survive churn; the unpinned free tail is evicted.
# ---------------------------------------------------------------------------
def test_pinned_survive_churn_unpinned_evicted():
    bank = _bank(persistent_slots=4, transient_slots=4)
    # Prefill routes admit the four warm experts into persistent slots; 0 and 1
    # are the prompt-frequent ones (higher prefill count).
    bank.prepare_prefill_seed([0, 0, 0, 1, 1, 2, 3])
    for e in (0, 1, 2, 3):
        bank.plan([e], phase="prefill")
    assert set(bank.resident_experts) == {0, 1, 2, 3}

    # Pin the top-2 (0, 1) and leave a 2-slot free tail (slots holding 2, 3).
    pinned = bank.pin_working_set(top_k=2, free_tail=2)
    assert pinned == (0, 1)
    assert bank.pinned_experts == frozenset({0, 1})
    assert bank.pinned_static is True

    # Churn: 40 distinct cold experts, each routed twice so it clears the
    # admission floor and takes a persistent slot.
    for cold in range(10, 50):
        bank.plan([cold], phase="decode")
        bank.plan([cold], phase="decode")

    resident = set(bank.resident_experts)
    # The pinned pair is still resident after all that churn ...
    assert {0, 1} <= resident
    # ... and the churn only ever recycled the 2-slot free tail (2 and 3 gone).
    assert 2 not in resident and 3 not in resident
    assert len(resident) == 4  # 2 pinned + 2 free-tail residents


# ---------------------------------------------------------------------------
# 2. decode misses still admit into the free tail.
# ---------------------------------------------------------------------------
def test_miss_admits_into_free_tail():
    bank = _bank(persistent_slots=3, transient_slots=4, policy="lru")
    bank.prepare_prefill_seed([0, 1, 2])
    for e in (0, 1, 2):
        bank.plan([e], phase="prefill")
    bank.pin_working_set(experts=[0, 1])  # pin two, one free-tail slot (holds 2)
    free_tail_slot = bank._expert_to_slot[2]

    # A cold decode miss must admit into the free tail (persistent), evicting 2,
    # never a pinned expert.
    plan = bank.plan([9], phase="decode")
    assert plan.loads and plan.loads[0].persistent is True
    assert plan.loads[0].slot == free_tail_slot
    assert plan.evictions[0].previous_expert == 2
    resident = set(bank.resident_experts)
    assert {0, 1, 9} == resident  # pinned kept, free tail recycled


# ---------------------------------------------------------------------------
# 3. pin_ws arm: pin all keys -> a fully static layer.
# ---------------------------------------------------------------------------
def test_pin_ws_all_keys_freezes_layer():
    bank = _bank(persistent_slots=3, transient_slots=4)
    bank.prepare_prefill_seed([5, 6, 7])
    for e in (5, 6, 7):
        bank.plan([e], phase="prefill")
    pinned = bank.pin_working_set(experts=bank.resident_experts)
    assert set(pinned) == {5, 6, 7}
    assert bank.pinned_static is True
    assert bank.route_all_pinned([5, 6]) is True
    assert bank.route_all_pinned([5, 99]) is False

    # Every persistent slot is pinned, so a hot cold expert cannot take one --
    # it is served transiently instead (byte-exact, just not cached).
    for _ in range(5):
        plan = bank.plan([42], phase="decode")
    assert all(not load.persistent for load in plan.loads)
    assert set(bank.resident_experts) == {5, 6, 7}  # frozen


# ---------------------------------------------------------------------------
# 4. the working set is ranked by prefill routing frequency.
# ---------------------------------------------------------------------------
def test_pin_ranks_by_prefill_frequency():
    bank = _bank(persistent_slots=4, transient_slots=4)
    # Expert 3 is the most prompt-frequent, then 2, then 1, then 0.
    bank.prepare_prefill_seed([3, 3, 3, 3, 2, 2, 2, 1, 1, 0])
    for e in (0, 1, 2, 3):
        bank.plan([e], phase="prefill")
    pinned = bank.pin_working_set(top_k=2, free_tail=2)
    assert set(pinned) == {2, 3}  # the two highest prefill-frequency experts


# ---------------------------------------------------------------------------
# 5. capacity eviction (memory hard constraint) may evict a pinned expert.
# ---------------------------------------------------------------------------
def test_capacity_eviction_may_evict_pinned_and_unpins():
    bank = _bank(persistent_slots=4, transient_slots=4)
    bank.prepare_prefill_seed([0, 1, 2, 3])
    for e in (0, 1, 2, 3):
        bank.plan([e], phase="prefill")
    bank.pin_working_set(experts=[0, 1, 2, 3])  # everything pinned

    # Normal admission never returns a pinned victim ...
    assert bank.peek_victim() is None
    # ... but a memory-forced capacity eviction (respect_pins=False) can.
    victim = bank.peek_victim(respect_pins=False)
    assert victim is not None
    expert, slot = victim
    assert expert in {0, 1, 2, 3}
    bank.invalidate_expert(expert)
    # The evicted expert is dropped from the pinned set (no longer resident).
    assert expert not in bank.pinned_experts
    assert expert not in set(bank.resident_experts)


# ---------------------------------------------------------------------------
# 6. served-expert gather is byte-identical with pinning on vs off.
# ---------------------------------------------------------------------------
class _SlotMemory:
    """Models the physical component bank: slot -> resident expert. A load writes
    its expert into its slot; the gather reads whichever expert currently occupies
    the resolved slot -- so a wrong slot (a recycle-before-read) would surface as a
    wrong expert, exactly the W44 window-19 corruption this lever prevents."""

    def __init__(self, n_slots: int) -> None:
        self.slot: list[int | None] = [None] * n_slots

    def apply(self, plan) -> None:
        for load in plan.loads:
            self.slot[load.slot] = load.expert

    def read(self, plan) -> list[int | None]:
        return [self.slot[s] for s in plan.slots]


def _sig(expert: int) -> mx.array:
    """Deterministic per-expert 'weight' signature (the gather's output)."""
    return ((expert + 1) * mx.arange(1, 5)).astype(mx.bfloat16)


def _drive(sequence, *, pin_after, pin_kind):
    """Route ``sequence`` through a real bank, modelling the physical slot memory,
    and return the concatenated per-assignment gather output. ``pin_after`` routes
    are executed as prefill+pin; the rest are decode. Every read must hold the
    requested expert (assert), so the output is a pure function of the request."""
    bank = _bank(persistent_slots=4, transient_slots=6)
    mem = _SlotMemory(bank.slot_count)
    outputs: list[mx.array] = []
    # Prefill warmup: seed + admit the first ``pin_after`` singleton routes.
    warm = sequence[:pin_after]
    flat = [e for route in warm for e in route]
    bank.prepare_prefill_seed(flat)
    for route in warm:
        plan = bank.plan(route, phase="prefill")
        mem.apply(plan)
    if pin_kind == "all":
        bank.pin_working_set(experts=bank.resident_experts)
    elif pin_kind == "topk":
        bank.pin_working_set(top_k=2, free_tail=2)
    # Decode: churn (some routes miss -> admit/evict) modelling the gather read.
    for route in sequence[pin_after:]:
        plan = bank.plan(route, phase="decode")
        mem.apply(plan)
        served = mem.read(plan)
        # The resolved slot must hold the requested expert -- correctness.
        assert served == list(plan.experts)
        for expert in served:
            outputs.append(_sig(int(expert)))
    return mx.concatenate(outputs) if outputs else mx.zeros(0)


def test_served_gather_byte_identical_pin_on_off():
    # A churny decode: warm experts 0..3 revisited, plus cold misses that force
    # eviction under both policies. top_k pinning changes WHICH slot each expert
    # lands in and the hit/miss split, but never which expert is served.
    seq = (
        [[0], [1], [2], [3]]            # prefill warmup (pin_after = 4)
        + [[0], [10], [1], [11], [2], [12], [0], [1], [13], [10], [11], [0]]
    )
    off = _drive(seq, pin_after=4, pin_kind=None)
    on_topk = _drive(seq, pin_after=4, pin_kind="topk")
    on_all = _drive(seq, pin_after=4, pin_kind="all")
    mx.eval(off, on_topk, on_all)
    assert off.tolist() == on_topk.tolist()  # byte-identical served output
    assert off.tolist() == on_all.tolist()


# ---------------------------------------------------------------------------
# 7. runtime pin hook + telemetry (real ExpertStreamingRuntime methods).
# ---------------------------------------------------------------------------
def _runtime_with_banks(layers, **bank_kw):
    """A real ExpertStreamingRuntime carrying only the state the W64 pin methods
    touch -- exercises the actual runtime code against real per-layer banks
    without opening an artifact or loading any experts.bin."""
    rt = object.__new__(ExpertStreamingRuntime)
    rt._banks = {layer: _bank(**bank_kw) for layer in layers}
    rt._global_bank = None
    rt._layer_locks = {layer: threading.Lock() for layer in layers}
    rt._pin_last_epoch = {}
    rt._pin_telemetry_lock = threading.Lock()
    rt._pin_decode_routes = 0
    rt._pin_all_pinned_routes = 0
    # _mark_device_route_dirty consults these; empty -> it is a no-op.
    rt._device_route_lut = {}
    rt._device_route_lut_dirty = {}
    return rt


def test_runtime_pin_hook_pins_after_prefill_and_reports_telemetry():
    os.environ[PIN_ENV] = "all"
    rt = _runtime_with_banks([1], persistent_slots=3, transient_slots=4)
    bank = rt._banks[1]
    # Prefill fills the persistent slots.
    bank.prepare_prefill_seed([0, 1, 2])
    for e in (0, 1, 2):
        bank.plan([e], phase="prefill")

    # Off phase check: a PREFILL hook call must not pin.
    rt.pin_working_set_hook(1, [0, 1, 2], RoutingPhase.PREFILL)
    assert bank.pinned_count == 0

    # First DECODE route pins the whole resident set (pin_ws arm) and records an
    # all-pinned route (0,1 are resident+pinned before the route executes).
    rt.pin_working_set_hook(1, [0, 1], RoutingPhase.DECODE)
    assert bank.pinned_experts == frozenset({0, 1, 2})
    assert rt.layer_pinned_static(1) is True
    assert rt.route_all_pinned(1, [0, 1]) is True

    # A route touching a non-pinned expert is not an all-pinned route.
    bank.plan([0], phase="decode")
    rt.pin_working_set_hook(1, [0, 99], RoutingPhase.DECODE)

    tel = rt.pinned_working_set_telemetry()
    assert tel["enabled"] is True
    assert tel["pinned_by_layer"] == {"1": 3}
    assert tel["pinned_total"] == 3
    assert tel["static_layer_count"] == 1
    assert tel["decode_routes"] == 2
    assert tel["all_pinned_routes"] == 1
    assert tel["all_pinned_hit_rate"] == pytest.approx(0.5)


def test_runtime_pin_hook_off_is_noop():
    # Lever unset -> the hook never pins and never counts (byte-identical path).
    rt = _runtime_with_banks([1])
    bank = rt._banks[1]
    for e in (0, 1, 2, 3):
        bank.plan([e], phase="prefill")
    rt.pin_working_set_hook(1, [0, 1], RoutingPhase.DECODE)
    assert bank.pinned_count == 0
    tel = rt.pinned_working_set_telemetry()
    assert tel["enabled"] is False
    assert tel["decode_routes"] == 0
    assert tel["pinned_by_layer"] == {}


def test_runtime_pin_refresh_cadence():
    os.environ[PIN_ENV] = "2"  # pin the top-2 resident experts
    os.environ[REFRESH_ENV] = "3"  # re-rank every 3 decode epochs
    rt = _runtime_with_banks([1], persistent_slots=4, transient_slots=6)
    bank = rt._banks[1]
    bank.prepare_prefill_seed([0, 0, 1, 1, 2, 3])
    for e in (0, 1, 2, 3):
        bank.plan([e], phase="prefill")

    epoch0 = bank._decode_epoch
    rt.pin_working_set_hook(1, [0], RoutingPhase.DECODE)  # first decode -> pins
    assert bank.pinned_count == 2
    first_epoch = rt._pin_last_epoch[1]
    assert first_epoch >= epoch0

    # Within the refresh window: no re-pin (epoch gate holds).
    bank.plan([0], phase="decode")
    rt.pin_working_set_hook(1, [0], RoutingPhase.DECODE)
    assert rt._pin_last_epoch[1] == first_epoch

    # Past the window: the next hook re-ranks (epoch advances).
    for _ in range(4):
        bank.plan([0], phase="decode")
    rt.pin_working_set_hook(1, [0], RoutingPhase.DECODE)
    assert rt._pin_last_epoch[1] > first_epoch


def test_env_parsers():
    assert parse_pin_working_set(None) is None
    assert parse_pin_working_set("") is None
    assert parse_pin_working_set("0") is None
    assert parse_pin_working_set("off") is None
    assert parse_pin_working_set("all") == ("all", None)
    assert parse_pin_working_set("keys") == ("all", None)
    assert parse_pin_working_set("1.0") == ("all", None)
    assert parse_pin_working_set("0.5") == ("frac", 0.5)
    assert parse_pin_working_set("50%") == ("frac", 0.5)
    assert parse_pin_working_set("8") == ("slots", 8.0)
    assert parse_pin_working_set("nonsense") is None
    assert parse_pin_working_set("1.5") is None  # out of (0, 1]
    assert parse_pin_refresh_tokens(None) == 0
    assert parse_pin_refresh_tokens("0") == 0
    assert parse_pin_refresh_tokens("4") == 4
    assert parse_pin_refresh_tokens("bad") == 0
