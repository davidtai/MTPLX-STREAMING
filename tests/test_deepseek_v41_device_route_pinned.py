"""W71 (KERNEL_LEDGER K24 revived) -- the barrier-free device route GUARDED BY
the W64 pins.

CPU-pinned, tiny synthetic switch / fake bank / real DSV4.1 backbone with fake
switches; no GPU, no artifact, no real ``experts.bin``; MLX pinned to the CPU
device (memory/worker-tests-must-pin-mlx-cpu.md). Run under ``nice -n 19`` and
without ``pytest -n auto``.

The lever (env ``MTPLX_DSV41_DEVICE_ROUTE_PINNED``, default off): issue the
``gather_qmm`` over a PINNED-ONLY device expert->slot LUT (``lut[e] = slot`` only
for pinned experts, ``-1`` otherwise) WITHOUT ``mx.eval(indices)`` -- zero host
syncs on an all-pinned layer -- and defer the all-pinned verification to a batched
``async_eval`` read the runtime flushes at the token boundary. Because a pinned
slot is never recycled by normal decode admission (W64), the deferred gather over
pinned slots CANNOT race a slot recycle (the W44 window-19 failure); the one case
W64 lets a pinned slot move -- a memory-forced capacity eviction -- unpins the
expert, which the flush sees (it checks the CURRENT pinned set) and recomputes on
the fenced path. Any not-all-pinned layer (an unpinned-resident or non-resident
expert) reads a void row and is likewise recovered fenced -- byte-identical.

These tests lock:
  1. all-pinned device-route output is bitwise-identical to the fenced switch, and
     an all-pinned ``_run`` performs ZERO host syncs, at M=1 and M=4;
  2. the pinned LUT is built from PINNED experts ONLY (an unpinned-but-resident
     expert reads -1), and refreshes only when the PIN set changes;
  3. the deferred flush flags exactly the non-pinned experts per layer -- including
     a pinned expert force-evicted mid-token -- and its telemetry counts the
     barrier-free (kept) vs recovered (refenced) layers; a flush is ONE batched sync;
  4. the whole DSV4.1 backbone decode/verify forward is byte-identical to the fully
     fenced path -- output + per-layer KV/compress/index cache + engram state -- across
     all-pinned / partial / miss / forced-eviction layer patterns at M=1 and M=4, and
     the routing barriers paid equal the number of NOT-all-pinned layers;
  5. (W44 race adapted) a pinned slot is NOT recycled under decode churn, so a
     deferred gather over pinned slots is isolated from the recycle that corrupted
     the unshelved W44 device route.
"""

from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

pytest.importorskip("mlx.core")

mx.set_default_device(mx.cpu)

import mtplx.models.expert_mlx as expert_mlx  # noqa: E402
from mtplx.expert_runtime import ExpertStreamingRuntime  # noqa: E402
from mtplx.expert_streaming import LayerExpertSlotBank, RoutingPhase  # noqa: E402
from mtplx.models.deepseek_v41 import Model  # noqa: E402
from mtplx.models.expert_mlx import (  # noqa: E402
    HotExpertSwitchGLU,
    current_expert_routing_phase,
    expert_routing_phase,
)

# Reuse the W44 switch-level fake bank / gather / runtime double + inputs, and the
# recovery test's cache fingerprint helpers (sibling-test imports, as the recovery
# suite already imports from test_deepseek_v41_served_generation).
from tests.test_deepseek_v41_device_route import (  # noqa: E402
    EXPERT_COUNT,
    HIDDEN,
    TOP_K,
    _DeviceRuntime,
    _FAKE_BANK,
    _REAL_EVAL,
    _REAL_GATHER,
    _REAL_TOLIST,
    _fake_gather,
    _fenced_reference,
    _inputs,
)
from tests.test_deepseek_v41_device_route_recovery import (  # noqa: E402
    _assert_cache_equal,
    _cache_fingerprint,
)
from tests.test_deepseek_v41_served_generation import (  # noqa: E402
    _csa_args,
    _ngram_state,
    _randomize,
)

DRP = "MTPLX_DSV41_DEVICE_ROUTE_PINNED"
DR = "MTPLX_DSV41_DEVICE_ROUTE"


@pytest.fixture(autouse=True)
def _cpu_and_flags():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    saved = {k: os.environ.get(k) for k in (DRP, DR)}
    for k in (DRP, DR):
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


# ===========================================================================
# Switch-level: a component-bank DECODE runtime double with the W71 pinned
# surface layered on top of the W44 residency double.
# ===========================================================================
class _PinnedDeviceRuntime(_DeviceRuntime):
    """``resident[layer] = {expert: slot}`` (slot == the fenced ``bank_index``);
    ``pinned[layer]`` is the never-recycled subset. The pinned LUT maps ONLY the
    pinned experts (everything else -> -1); the flush checks the CURRENT pinned
    set, so a force-eviction (unpin) flips an all-pinned route to a recompute."""

    def __init__(self, events, resident, pinned) -> None:
        super().__init__(events, resident)
        self._pinned = {int(k): set(int(e) for e in v) for k, v in pinned.items()}
        self._pinned_lut: dict[int, mx.array] = {}
        self._pinned_snap: dict[int, frozenset] = {}
        self._pinned_dirty: dict[int, bool] = {}
        self._pinned_lut_builds: dict[int, int] = {}

    # pin-state mutation --------------------------------------------------
    def pin(self, layer, experts) -> None:
        self._pinned.setdefault(int(layer), set()).update(int(e) for e in experts)
        self._pinned_dirty[int(layer)] = True

    def force_evict(self, layer, expert) -> None:
        """Memory-forced capacity eviction: unpin + drop residency + invalidate."""
        layer, expert = int(layer), int(expert)
        self._pinned.get(layer, set()).discard(expert)
        self._resident.get(layer, {}).pop(expert, None)
        self._pinned_dirty[layer] = True
        self._lut_dirty[layer] = True

    # W71 device-route surface -------------------------------------------
    def device_route_pinned_lut(self, layer, *, mx_module=None):
        layer = int(layer)
        cached = self._pinned_lut.get(layer)
        if cached is not None and not self._pinned_dirty.get(layer, False):
            return cached
        table = [-1] * EXPERT_COUNT
        snap = set()
        res = self._resident.get(layer, {})
        for e in self._pinned.get(layer, set()):
            slot = res.get(int(e))
            if slot is not None and 0 <= e < EXPERT_COUNT:
                table[int(e)] = int(slot)
                snap.add(int(e))
        arr = mx.array(table, dtype=mx.int32)
        mx.eval(arr)
        self._pinned_lut[layer] = arr
        self._pinned_snap[layer] = frozenset(snap)
        self._pinned_dirty[layer] = False
        self._pinned_lut_builds[layer] = self._pinned_lut_builds.get(layer, 0) + 1
        return arr

    def device_route_pinned_snapshot(self, layer):
        return self._pinned_snap.get(int(layer), frozenset())

    def enqueue_device_route_probe(self, layer, indices, snapshot, *, pinned=False):
        self._probes.append((int(layer), indices, snapshot, bool(pinned)))

    def flush_device_route_probes(self):
        probes = self._probes
        self._probes = []
        misses = []
        for layer, indices, snapshot, pinned in probes:
            ids = [int(v) for v in indices.reshape(-1).tolist()]
            if pinned:
                pinned_now = self._pinned.get(layer, set())
                missed = tuple(sorted({e for e in ids if e not in pinned_now}))
            else:
                missed = tuple(sorted({e for e in ids if e not in snapshot}))
            if missed:
                misses.append((layer, missed))
        return misses


def _run_switch_pinned(runtime, x, idx, *, pinned, layer=0):
    """Run the REAL HotExpertSwitchGLU with the pinned device route on/off."""
    if pinned:
        os.environ[DRP] = "1"
    else:
        os.environ.pop(DRP, None)
    prev = expert_mlx._gather_component_bank
    expert_mlx._gather_component_bank = _fake_gather
    try:
        out = HotExpertSwitchGLU(runtime, layer)(x, idx)
    finally:
        expert_mlx._gather_component_bank = prev
    _REAL_EVAL(out)
    return out


# ---------------------------------------------------------------------------
# 1. all-pinned: byte-identical to fenced + zero host syncs (M=1, M=4)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rows", [1, 4])
def test_all_pinned_device_route_bitwise_identical_and_zero_sync(rows) -> None:
    experts = [(i % EXPERT_COUNT) for i in range(rows * TOP_K)]
    x, idx = _inputs(rows, experts)
    fenced = _fenced_reference(x, idx)

    # every routed expert resident (identity slots) AND pinned -> all-pinned route.
    resident = {0: {e: e + 1 for e in set(experts)}}
    pinned = {0: set(experts)}
    rt = _PinnedDeviceRuntime([], resident, pinned)
    rt.register_component_bank(0, _FAKE_BANK)
    rt.device_route_pinned_lut(0)  # warm the pinned LUT

    counts = {"eval": 0, "tolist": 0}

    def c_eval(*a, **k):
        counts["eval"] += 1
        return _REAL_EVAL(*a, **k)

    def c_tolist(self, *a, **k):
        counts["tolist"] += 1
        return _REAL_TOLIST(self, *a, **k)

    os.environ[DRP] = "1"
    expert_mlx._gather_component_bank = _fake_gather
    expert_mlx.mx.eval = c_eval
    mx.array.tolist = c_tolist
    try:
        out = HotExpertSwitchGLU(rt, 0)(x, idx)
    finally:
        expert_mlx.mx.eval = _REAL_EVAL
        mx.array.tolist = _REAL_TOLIST
        expert_mlx._gather_component_bank = _REAL_GATHER

    # zero blocking host syncs on the all-pinned layer.
    assert counts["eval"] == 0, counts
    assert counts["tolist"] == 0, counts

    _REAL_EVAL(out)
    assert out.shape == fenced.shape == (rows, 1, TOP_K, HIDDEN)
    assert mx.array_equal(out, fenced), f"rows={rows}: pinned device route != fenced"
    # the deferred pinned probe finds no miss on an all-pinned layer.
    assert rt.flush_device_route_probes() == []


# ---------------------------------------------------------------------------
# 2 + 3. pinned-only LUT; flush flags exactly the non-pinned experts; recovery
#        matches fenced (all-pinned / unpinned-resident / miss), M=1 and M=4.
# ---------------------------------------------------------------------------
def _pinned_scenario(kind, rows):
    """3-layer 'token'. Every routed expert is RESIDENT (identity slot e+1); the
    PIN set varies so an unpinned-but-resident expert reads a void LUT row."""
    experts_per_layer = {
        0: [(i % EXPERT_COUNT) for i in range(rows * TOP_K)],
        1: [((i + 3) % EXPERT_COUNT) for i in range(rows * TOP_K)],
        2: [((i + 5) % EXPERT_COUNT) for i in range(rows * TOP_K)],
    }
    resident, pinned = {}, {}
    for layer, experts in experts_per_layer.items():
        uniq = set(experts)
        resident[layer] = {e: e + 1 for e in uniq}  # all resident, always
        if kind == "all_pinned":
            pinned[layer] = set(uniq)
        elif kind == "none_pinned":
            pinned[layer] = set()
        else:  # partial: layer 1 all-pinned; 0/2 leave one resident expert unpinned
            if layer == 1:
                pinned[layer] = set(uniq)
            else:
                pinned[layer] = uniq - {sorted(uniq)[0]}
    return experts_per_layer, resident, pinned


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("kind", ["all_pinned", "partial", "none_pinned"])
def test_pinned_route_sequence_reconciles_to_fenced(kind, rows) -> None:
    experts_per_layer, resident, pinned = _pinned_scenario(kind, rows)
    rt = _PinnedDeviceRuntime([], resident, pinned)
    for layer in experts_per_layer:
        rt.register_component_bank(layer, _FAKE_BANK)

    fenced_seq, device_seq, inputs = {}, {}, {}
    for layer, experts in experts_per_layer.items():
        x, idx = _inputs(rows, experts)
        inputs[layer] = (x, idx)
        fenced_seq[layer] = _fenced_reference(x, idx)
        device_seq[layer] = _run_switch_pinned(rt, x, idx, pinned=True, layer=layer)

    # (3) the deferred pinned probes flag exactly the NOT-pinned experts per layer
    #     (an unpinned-but-resident expert counts as a miss for the pinned route).
    misses = dict(rt.flush_device_route_probes())
    expected = {}
    for layer, experts in experts_per_layer.items():
        missed = tuple(sorted({e for e in experts if e not in pinned[layer]}))
        if missed:
            expected[layer] = missed
    assert misses == expected, (kind, rows, misses, expected)

    # all-pinned layers are already byte-identical; not-all-pinned differ (void row).
    for layer in experts_per_layer:
        if layer in expected:
            assert not mx.array_equal(device_seq[layer], fenced_seq[layer]), (
                f"{kind} L{layer}: not-all-pinned optimistic output matched fenced"
            )
        else:
            assert mx.array_equal(device_seq[layer], fenced_seq[layer]), (
                f"{kind} L{layer}: all-pinned device output != fenced"
            )

    # (recovery) the backbone forces the flagged layers onto the fenced path; the
    # experts are resident, so the fenced re-run is byte-identical to fenced.
    reconciled = dict(device_seq)
    for layer in expected:
        x, idx = inputs[layer]
        reconciled[layer] = _run_switch_pinned(rt, x, idx, pinned=False, layer=layer)
    for layer in experts_per_layer:
        assert mx.array_equal(reconciled[layer], fenced_seq[layer]), (
            f"{kind} L{layer}: reconciled != fenced"
        )


def test_pinned_lut_is_pinned_only_and_refreshes_on_pin_change() -> None:
    # experts 1,2,3 resident; only 1,2 pinned -> LUT maps 1,2; 3 (resident,
    # unpinned) reads -1, same as a truly absent expert.
    rt = _PinnedDeviceRuntime([], {0: {1: 2, 2: 3, 3: 4}}, {0: {1, 2}})
    a = rt.device_route_pinned_lut(0)
    assert int(a[1].item()) == 2 and int(a[2].item()) == 3
    assert int(a[3].item()) == -1, "an unpinned-but-resident expert must read -1"
    assert int(a[5].item()) == -1
    assert rt.device_route_pinned_snapshot(0) == frozenset({1, 2})
    b = rt.device_route_pinned_lut(0)
    assert a is b, "pinned LUT rebuilt with no pin-set change"
    assert rt._pinned_lut_builds[0] == 1

    rt.pin(0, [3])  # pin set changed -> dirty
    c = rt.device_route_pinned_lut(0)
    assert c is not a and rt._pinned_lut_builds[0] == 2
    assert int(c[3].item()) == 4 and rt.device_route_pinned_snapshot(0) == frozenset(
        {1, 2, 3}
    )


# ===========================================================================
# 3 (real runtime). The production pinned-LUT / flush / telemetry against real
# per-layer banks -- no artifact, no experts.bin.
# ===========================================================================
def _bank(*, persistent_slots=4, transient_slots=6, expert_count=32, policy="frequency"):
    return LayerExpertSlotBank(
        expert_count=expert_count,
        persistent_slots=persistent_slots,
        transient_slots=transient_slots,
        frequency_decay=1.0,
        cache_policy=policy,
    )


def _real_runtime(layers, *, expert_count=32, **bank_kw):
    """A real ExpertStreamingRuntime carrying only the state the W71 pinned-device
    surface touches -- exercises the actual runtime code against real banks."""
    rt = object.__new__(ExpertStreamingRuntime)
    rt.spec = SimpleNamespace(expert_count=expert_count)
    rt._banks = {l: _bank(expert_count=expert_count, **bank_kw) for l in layers}
    rt._global_bank = None
    rt._layer_locks = {l: threading.Lock() for l in layers}
    rt._device_route_lut = {}
    rt._device_route_lut_snapshot = {}
    rt._device_route_lut_dirty = {}
    rt._device_route_pinned_lut = {}
    rt._device_route_pinned_snapshot = {}
    rt._device_route_pinned_lut_dirty = {}
    rt._device_route_probes = []
    rt._device_route_pinned_flushes = 0
    rt._device_route_pinned_barrier_free_layers = 0
    rt._device_route_pinned_recovered_layers = 0
    return rt


def test_real_runtime_pinned_lut_and_force_evict_flush_and_telemetry() -> None:
    rt = _real_runtime([0], persistent_slots=4, transient_slots=6)
    bank = rt._banks[0]
    bank.prepare_prefill_seed([0, 0, 1, 1, 2, 3])
    for e in (0, 1, 2, 3):
        bank.plan([e], phase="prefill")
    bank.pin_working_set(experts=[0, 1])  # pin 0,1; 2,3 resident but unpinned

    lut = rt.device_route_pinned_lut(0)
    slot0, slot1 = bank._expert_to_slot[0], bank._expert_to_slot[1]
    assert int(lut[0].item()) == slot0 and int(lut[1].item()) == slot1
    assert int(lut[2].item()) == -1 and int(lut[3].item()) == -1  # resident, unpinned
    assert rt.device_route_pinned_snapshot(0) == frozenset({0, 1})

    # A pinned route [0, 1] flushes clean (barrier-free); telemetry counts it.
    rt.enqueue_device_route_probe(
        0, mx.array([[0, 1]], dtype=mx.int32), frozenset({0, 1}), pinned=True
    )
    assert rt.flush_device_route_probes() == []

    # A free-tail LRU churn (an UNPINNED residency change) must NOT dirty the
    # pinned LUT -- a pinned slot never moves, so the pinned view stays exact.
    rt._mark_device_route_dirty(0)
    assert rt._device_route_pinned_lut_dirty.get(0) is not True
    assert rt.device_route_pinned_lut(0) is lut  # not rebuilt

    # Force-eviction (memory hard constraint): unpin expert 0 through the real
    # policy path, which invalidates the pinned LUT because expert 0 was pinned.
    slot = rt._invalidate_policy_expert(0, 0)  # unpins + drops residency
    assert slot is not None
    assert rt._device_route_pinned_lut_dirty[0] is True
    lut2 = rt.device_route_pinned_lut(0)
    assert int(lut2[0].item()) == -1  # expert 0 no longer pinned
    assert rt.device_route_pinned_snapshot(0) == frozenset({1})

    rt.enqueue_device_route_probe(
        0, mx.array([[0, 1]], dtype=mx.int32), frozenset({0, 1}), pinned=True
    )
    misses = rt.flush_device_route_probes()
    assert misses == [(0, (0,))], misses  # the force-evicted pin is recovered

    os.environ[DRP] = "1"
    try:
        tel = rt.device_route_pinned_telemetry()
    finally:
        os.environ.pop(DRP, None)
    assert tel["enabled"] is True
    assert tel["flushes"] == 2
    assert tel["barrier_free_layers"] == 1  # the first (all-pinned) flush
    assert tel["recovered_layers"] == 1  # the force-evicted flush
    assert tel["barrier_free_layers_per_flush"] == pytest.approx(0.5)


def test_real_runtime_flush_is_one_batched_sync() -> None:
    """The '+1 flush': one flush call issues exactly ONE mx.eval regardless of how
    many probed layers it verifies (the batched span-end verify sync, never one per
    layer -- which would reintroduce the ~40 barriers the route exists to remove)."""
    rt = _real_runtime([0, 1, 2], persistent_slots=4, transient_slots=6)
    for lid in (0, 1, 2):
        bank = rt._banks[lid]
        bank.prepare_prefill_seed([0, 1, 2, 3])
        for e in (0, 1, 2, 3):
            bank.plan([e], phase="prefill")
        bank.pin_working_set(experts=[0, 1, 2, 3])
        rt.enqueue_device_route_probe(
            lid, mx.array([[0, 1]], dtype=mx.int32), frozenset({0, 1}), pinned=True
        )

    calls = {"n": 0}
    real_eval = mx.eval

    def counting_eval(*a, **k):
        calls["n"] += 1
        return real_eval(*a, **k)

    mx.eval = counting_eval
    try:
        misses = rt.flush_device_route_probes()
    finally:
        mx.eval = real_eval
    assert misses == []  # all three layers all-pinned
    assert calls["n"] == 1, f"flush issued {calls['n']} syncs, expected 1 batched"


# ===========================================================================
# 4. Backbone end-to-end: byte-identity + cache/engram + barrier arithmetic.
# ===========================================================================
class _FakePinnedRT:
    """The pinned-device-route surface the backbone + fake switch use, over a PIN
    map this test controls. Only ``barrier()`` (a fenced ``mx.eval(indices)``)
    counts toward routing barriers -- the pinned device path issues none. The pin
    map is pre-established (out-of-band), so ``pin_working_set`` is a no-op here."""

    def __init__(self, pinned, force_evict=None):
        self.pinned = {int(k): set(int(e) for e in v) for k, v in pinned.items()}
        self._device_route_force_fenced = frozenset()
        self.probes: list = []
        self.barriers = 0
        # {trigger_layer: (target_layer, expert)} -- unpin target's expert when
        # trigger layer's device route runs (models a mid-token force-eviction).
        self.force_evict = {int(k): v for k, v in (force_evict or {}).items()}

    def flush_device_route_probes(self):
        pr, self.probes = self.probes, []
        misses = []
        for lid, ids, _pinned in pr:
            pinned_now = self.pinned.get(lid, set())
            missed = tuple(sorted({e for e in ids if e not in pinned_now}))
            if missed:
                misses.append((lid, missed))
        return misses

    def set_device_route_force_fenced(self, layers):
        self._device_route_force_fenced = frozenset(int(x) for x in layers)

    def pin_working_set(self, layer=None):
        return {}  # pins pre-established in __init__ (out-of-band boundary call)

    def unpin(self, layer, expert):
        self.pinned.get(int(layer), set()).discard(int(expert))

    def barrier(self, indices):
        self.barriers += 1
        mx.eval(indices)


class _FakePinnedSwitch(nn.Module):
    """Deterministic expert-id-dependent stand-in. pinned device path: gather with
    the pinned expert id, but a NON-pinned expert clamps to id 0 (a void row) and
    enqueues a probe (no barrier). fenced path: gather with the true id (correct),
    pay one barrier. An all-pinned device layer == fenced byte-for-byte."""

    def __init__(self, layer_id, rt):
        super().__init__()
        self.layer_id = int(layer_id)
        self.runtime = rt

    def __call__(self, xf, indices):
        rt = self.runtime
        device = (
            os.environ.get(DRP) == "1"
            and current_expert_routing_phase(token_count=int(xf.shape[0]))
            is RoutingPhase.DECODE
            and self.layer_id not in rt._device_route_force_fenced
        )
        ids = [[int(e) for e in row] for row in indices.tolist()]
        pinned = rt.pinned.get(self.layer_id, set())
        if device:
            eff = [[e if e in pinned else 0 for e in row] for row in ids]
            rt.probes.append(
                (self.layer_id, [e for row in ids for e in row], True)
            )
            # A later layer's route force-evicts an earlier layer's pinned expert.
            if self.layer_id in rt.force_evict:
                tl, te = rt.force_evict[self.layer_id]
                rt.unpin(tl, te)
        else:
            eff = ids
            rt.barrier(indices)
        eff_arr = mx.array(eff, dtype=mx.float32)[..., None]  # [n, top_k, 1]
        routed = xf[:, None, :].astype(mx.float32) * (1.0 + 0.03 * eff_arr)
        return routed.astype(xf.dtype)


def _install(model, rt):
    for layer in model.model.layers:
        layer.mlp.switch_mlp = _FakePinnedSwitch(layer.layer_id, rt)


def _run(pattern, m, *, seed=3, engram=False):
    """Build a fresh seeded model, install pinned fake switches, run one span of
    ``m`` tokens (M=1 AR / M=4 verify), return (logits, cache_fingerprint, barriers)."""
    args = _csa_args()
    model = Model(args)
    _randomize(model, seed=seed)
    if engram:
        model.model.engram_hash = _ngram_state(args.vocab_size)

    n_layers = len(model.model.layers)
    routed = set(range(args.n_routed_experts))
    force_evict = None
    if pattern == "fenced":
        pinned = {lid: set(routed) for lid in range(n_layers)}
    elif pattern == "all_pinned":
        pinned = {lid: set(routed) for lid in range(n_layers)}
    elif pattern == "partial":
        pinned = {lid: set(routed) for lid in range(n_layers)}
        pinned[3] = routed - {0}  # layer 3 has one unpinned routed expert
    elif pattern == "multi":
        pinned = {lid: set(routed) for lid in range(n_layers)}
        for lid in (2, 5, 6):
            pinned[lid] = set()  # not-all-pinned (no pins)
    elif pattern == "none":
        pinned = {lid: set() for lid in range(n_layers)}  # every layer refenced
    elif pattern == "forced_evict":
        pinned = {lid: set(routed) for lid in range(n_layers)}
        force_evict = {n_layers - 1: (2, 0)}  # last layer unpins layer 2's expert 0
    else:
        raise ValueError(pattern)

    rt = _FakePinnedRT(pinned, force_evict=force_evict)
    _install(model, rt)

    cache = model.make_cache()
    rng = np.random.default_rng(11)
    ids = mx.array(rng.integers(0, args.vocab_size, size=(1, m)), dtype=mx.int32)

    if pattern == "fenced":
        os.environ.pop(DRP, None)
    else:
        os.environ[DRP] = "1"

    def forward():
        return model(ids, cache=cache)

    if m == 1:
        logits = forward()
    else:
        with expert_routing_phase(RoutingPhase.DECODE):
            logits = forward()
    mx.eval(logits)
    return logits, _cache_fingerprint(cache), rt.barriers


@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize(
    "pattern,expected_barriers",
    [
        ("all_pinned", 0),
        ("partial", 1),
        ("multi", 3),
        ("none", 8),
        ("forced_evict", 1),
    ],
)
def test_pinned_backbone_matches_fenced(pattern, expected_barriers, m):
    ref_logits, ref_cache, _ = _run("fenced", m, seed=3)
    dev_logits, dev_cache, dev_bar = _run(pattern, m, seed=3)

    assert dev_logits.shape == ref_logits.shape
    assert mx.array_equal(dev_logits, ref_logits), (
        f"{pattern} M={m}: pinned device-route output != fenced"
    )
    _assert_cache_equal(dev_cache, ref_cache)
    # barriers/token == number of NOT-all-pinned layers (the recovery pass fences
    # exactly those; all-pinned layers pay none). The single batched span-end
    # verify sync is the '+1 flush', asserted in test_real_runtime_flush_is_one_...
    assert dev_bar == expected_barriers, (
        f"{pattern} M={m}: {dev_bar} routing barriers, expected {expected_barriers}"
    )


@pytest.mark.parametrize("m", [1, 4])
def test_pinned_backbone_preserves_engram_state(m):
    ref_logits, ref_cache, _ = _run("fenced", m, seed=5, engram=True)
    dev_logits, dev_cache, dev_bar = _run("forced_evict", m, seed=5, engram=True)
    assert mx.array_equal(dev_logits, ref_logits), f"M={m}: output != fenced (engram)"
    _assert_cache_equal(dev_cache, ref_cache)  # includes engram _buf/_len
    assert dev_bar == 1


# ===========================================================================
# 5. W44 race adapted: a pinned slot is NOT recycled under decode churn, so a
#    deferred gather over pinned slots is isolated from the recycle that
#    corrupted the unshelved W44 device route.
# ===========================================================================
def test_pinned_slot_not_recycled_under_churn():
    """W44 §8: the shelved device route's deferred gather read an UNpinned slot a
    mid-decode admission recycled in place -> garbage. W64 pins make the read slot
    stable: the pinned expert keeps its exact bank row through a churn that recycles
    only the free tail, so a deferred gather over that row reads the same expert it
    was issued for -- the safety property the pinned device route relies on."""
    bank = _bank(persistent_slots=4, transient_slots=6, policy="lru")
    bank.prepare_prefill_seed([0, 0, 1, 1, 2, 3])
    for e in (0, 1, 2, 3):
        bank.plan([e], phase="prefill")
    bank.pin_working_set(top_k=2, free_tail=2)  # pin 0,1; free tail holds 2,3
    pinned_slot_0 = bank._expert_to_slot[0]
    pinned_slot_1 = bank._expert_to_slot[1]

    # Churn: 20 distinct cold experts, each routed twice to clear the admission
    # floor and take a persistent slot -- exactly the decode LRU churn that
    # recycled the W44 read slot.
    for cold in range(10, 30):
        bank.plan([cold], phase="decode")
        bank.plan([cold], phase="decode")

    # The pinned experts kept their EXACT slots (a deferred gather issued over
    # ``pinned_slot_*`` still reads expert 0/1) ...
    assert bank._expert_to_slot.get(0) == pinned_slot_0
    assert bank._expert_to_slot.get(1) == pinned_slot_1
    assert bank._slot_to_expert[pinned_slot_0] == 0
    assert bank._slot_to_expert[pinned_slot_1] == 1
    # ... and the churn only ever recycled the unpinned free tail.
    resident = set(bank.resident_experts)
    assert {0, 1} <= resident
    assert 2 not in resident and 3 not in resident

    # A route over the pinned pair is all-pinned for the whole decode -> the pinned
    # device route may keep it barrier-free (its deferred gather cannot race).
    assert bank.route_all_pinned([0, 1]) is True
