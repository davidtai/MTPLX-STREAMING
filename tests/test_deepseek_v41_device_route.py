"""W44 (KERNEL_LEDGER K24) -- barrier-free all-hit device route.

CPU-pinned, tiny synthetic streamed switch + fake bank. No GPU, no artifact, no
real weights; MLX pinned to the CPU device (memory/worker-tests-must-pin-mlx-cpu.md).
Run under ``nice -n 19`` and without ``pytest -n auto``.

The lever (env ``MTPLX_DSV41_DEVICE_ROUTE``, default off): issue the all-hit
``gather_qmm`` over a device-side expert->slot LUT (``lut[indices]``) WITHOUT
``mx.eval(indices)`` -- zero host syncs on an all-hit layer -- and defer residency
verification to an ``async_eval`` read the runtime flushes later. All-hit layers
are byte-identical to the fenced path (the LUT slot equals the fenced
``bank_index``); a missing expert reads a void row and is caught by the deferred
probe, whose caller recovers it on the fenced path (see W44_DEVICE_ROUTE.md).

These tests lock:
  1. all-hit device-route output is bitwise-identical to the fenced switch, M=1 & M=4;
  2. an all-hit device-route ``_run`` performs ZERO host syncs (no mx.eval / .tolist);
  3. the deferred probe correctly flags exactly the non-resident experts per layer;
  4. recovery (admit the misses, re-run) is bitwise-identical to fenced, so a
     reconciled all-hit/mixed/all-miss sequence matches fenced at M=1 and M=4;
  5. the device LUT refreshes only when residency changes.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

pytest.importorskip("mlx.core")

mx.set_default_device(mx.cpu)

import mtplx.models.expert_mlx as expert_mlx  # noqa: E402
from mtplx.expert_runtime import RouteWave  # noqa: E402
from mtplx.expert_streaming import RoutingPhase  # noqa: E402
from mtplx.models.expert_mlx import HotExpertSwitchGLU  # noqa: E402

DEVICE_ROUTE_FLAG = "MTPLX_DSV41_DEVICE_ROUTE"

_REAL_EVAL = mx.eval
_REAL_TOLIST = mx.array.tolist
_REAL_GATHER = expert_mlx._gather_component_bank

EXPERT_COUNT = 8
TOP_K = 2
HIDDEN = 2
_FAKE_BANK = object()


@pytest.fixture(autouse=True)
def _cpu_and_flag():
    import os

    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    saved = os.environ.get(DEVICE_ROUTE_FLAG)
    os.environ.pop(DEVICE_ROUTE_FLAG, None)
    try:
        yield
    finally:
        mx.set_default_device(prev)
        if saved is None:
            os.environ.pop(DEVICE_ROUTE_FLAG, None)
        else:
            os.environ[DEVICE_ROUTE_FLAG] = saved


def _fake_gather(x, bank, slot_indices, *, group_size, bits, swiglu_limit=None, codec="affine"):
    """Deterministic, slot-sensitive stand-in: output depends on the slot index,
    so a wrong slot (a void miss row) shows as a bit difference and a matching
    slot is byte-identical."""
    rows = int(x.shape[0])
    s = slot_indices.reshape((rows, 1)).astype(mx.float32)
    return (x.astype(mx.float32) * 2.0 + s * 10.0 + 1.0).astype(x.dtype)


class _DeviceRuntime:
    """Component-bank DECODE runtime double with a controllable residency map and
    the W44 device-route surface. ``resident[layer] = {expert: slot}`` (slot ==
    the fenced ``bank_index``). Non-resident experts are LUT -1 (a miss)."""

    def __init__(self, events, resident) -> None:
        self.events = events
        self.spec = SimpleNamespace(
            top_k=TOP_K,
            hidden_size=HIDDEN,
            quant_group_size=64,
            quant_bits=4,
            expert_count=EXPERT_COUNT,
        )
        self.manifest = SimpleNamespace(sidecar=None)
        self.config = SimpleNamespace(
            slot_layout="component-banks",
            resource_telemetry=False,
        )
        self._pipeline_ledger = None
        self._resident = {int(k): dict(v) for k, v in resident.items()}
        self._bank_registered: set[int] = set()
        self._lut: dict[int, mx.array] = {}
        self._lut_snapshot: dict[int, frozenset] = {}
        self._lut_dirty: dict[int, bool] = {}
        self._lut_builds: dict[int, int] = {}
        self._probes: list = []

    # residency mutation (models admission on recovery) ------------------
    def admit(self, layer, experts) -> None:
        layer = int(layer)
        for e in experts:
            # true slot = e+1 (slot 0 is intentionally never a real expert, so a
            # miss's clamp-to-0 can never alias a resident expert's output).
            self._resident.setdefault(layer, {})[int(e)] = int(e) + 1
        self._lut_dirty[layer] = True

    # device-route surface ----------------------------------------------
    def register_component_bank(self, layer, bank) -> None:
        if bank is not None:
            self._bank_registered.add(int(layer))

    def component_bank_for_layer(self, layer):
        return _FAKE_BANK if int(layer) in self._bank_registered else None

    def device_route_lut(self, layer, *, mx_module=None):
        layer = int(layer)
        cached = self._lut.get(layer)
        if cached is not None and not self._lut_dirty.get(layer, False):
            return cached
        table = [-1] * EXPERT_COUNT
        snap = set()
        for e, slot in self._resident.get(layer, {}).items():
            if 0 <= e < EXPERT_COUNT:
                table[e] = int(slot)
                snap.add(int(e))
        arr = mx.array(table, dtype=mx.int32)
        mx.eval(arr)
        self._lut[layer] = arr
        self._lut_snapshot[layer] = frozenset(snap)
        self._lut_dirty[layer] = False
        self._lut_builds[layer] = self._lut_builds.get(layer, 0) + 1
        return arr

    def device_route_snapshot(self, layer):
        return self._lut_snapshot.get(int(layer), frozenset())

    def enqueue_device_route_probe(self, layer, indices, snapshot) -> None:
        self._probes.append((int(layer), indices, snapshot))

    def flush_device_route_probes(self):
        probes = self._probes
        self._probes = []
        misses = []
        for layer, indices, snapshot in probes:
            ids = [int(v) for v in indices.reshape(-1).tolist()]
            missed = tuple(sorted({e for e in ids if e not in snapshot}))
            if missed:
                misses.append((layer, missed))
        return misses

    # fenced surface (only try_all_hit is exercised -- reference + recovery) --
    def observe_route(self, *_a, **_k):
        return None

    def prepare_prefill_seed(self, *_a, **_k):
        return ()

    def peek_resident_experts(self, layer, expert_ids):
        res = self._resident.get(int(layer), {})
        return frozenset(e for e in expert_ids if e in res)

    def route_waves(self, expert_ids, **_k):
        experts = tuple(expert_ids)
        return (RouteWave(positions=tuple(range(len(experts))), experts=experts),)

    def try_all_hit_route(self, layer, experts, *, phase, **_k):
        res = self._resident.get(int(layer), {})
        experts = tuple(experts)
        if any(e not in res for e in experts):
            return None
        bindings = tuple(
            SimpleNamespace(
                expert=int(e),
                buffer=SimpleNamespace(bank=_FAKE_BANK, bank_index=int(res[e])),
            )
            for e in experts
        )
        return SimpleNamespace(
            plan=SimpleNamespace(hits=experts),
            bindings=bindings,
            release=lambda **_kw: self.events.append("release"),
        )

    def begin_split_route(self, *_a, **_k):  # not reached in these tests
        raise AssertionError("device-route tests must not hit the split path")


def _inputs(rows, layer_experts):
    """x = [rows, 1, HIDDEN]; indices = [rows, 1, TOP_K] from ``layer_experts``
    (a flat list of rows*TOP_K expert ids)."""
    mx.random.seed(7 + rows)
    x = (0.5 * mx.random.normal((rows, 1, HIDDEN))).astype(mx.bfloat16)
    idx = mx.array(layer_experts, dtype=mx.int32).reshape((rows, 1, TOP_K))
    mx.eval(x, idx)
    return x, idx


def _full_resident():
    # every expert resident in its identity slot -> the fenced reference is
    # all-hit for any route, output = fake_gather over the true slots (== e).
    return {0: {e: e + 1 for e in range(EXPERT_COUNT)}}


def _run_switch(runtime, x, idx, *, device_route, layer=0):
    import os

    if device_route:
        os.environ[DEVICE_ROUTE_FLAG] = "1"
    else:
        os.environ.pop(DEVICE_ROUTE_FLAG, None)
    prev = expert_mlx._gather_component_bank
    expert_mlx._gather_component_bank = _fake_gather
    try:
        out = HotExpertSwitchGLU(runtime, layer)(x, idx)
    finally:
        expert_mlx._gather_component_bank = prev
    _REAL_EVAL(out)
    return out


def _fenced_reference(x, idx):
    """Run the real fenced all-hit switch path against a fully-resident runtime."""
    rt = _DeviceRuntime([], _full_resident())
    return _run_switch(rt, x, idx, device_route=False)


# ---------------------------------------------------------------------------
# 1 + 2. all-hit: byte-identical to fenced, and zero host syncs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rows", [1, 4])
def test_all_hit_device_route_bitwise_identical_and_zero_sync(rows) -> None:
    experts = [(i % EXPERT_COUNT) for i in range(rows * TOP_K)]
    x, idx = _inputs(rows, experts)
    fenced = _fenced_reference(x, idx)

    # residency covers every routed expert (all-hit), identity slots.
    resident = {0: {e: e + 1 for e in set(experts)}}
    rt = _DeviceRuntime([], resident)
    rt.register_component_bank(0, _FAKE_BANK)   # bank captured by a prior route
    rt.device_route_lut(0)                       # warm the LUT (built on cache change)

    # census: count blocking host syncs during the device-route _run.
    counts = {"eval": 0, "tolist": 0}

    def c_eval(*a, **k):
        counts["eval"] += 1
        return _REAL_EVAL(*a, **k)

    def c_tolist(self, *a, **k):
        counts["tolist"] += 1
        return _REAL_TOLIST(self, *a, **k)

    import os

    os.environ[DEVICE_ROUTE_FLAG] = "1"
    expert_mlx._gather_component_bank = _fake_gather
    expert_mlx.mx.eval = c_eval
    mx.array.tolist = c_tolist
    try:
        out = HotExpertSwitchGLU(rt, 0)(x, idx)
    finally:
        expert_mlx.mx.eval = _REAL_EVAL
        mx.array.tolist = _REAL_TOLIST
        expert_mlx._gather_component_bank = _REAL_GATHER
    # the switch itself issued no blocking sync on this all-hit layer.
    assert counts["eval"] == 0, counts
    assert counts["tolist"] == 0, counts

    _REAL_EVAL(out)
    assert out.shape == fenced.shape == (rows, 1, TOP_K, HIDDEN)
    assert mx.array_equal(out, fenced), f"rows={rows}: device route != fenced"

    # the deferred probe finds no miss on an all-hit layer.
    assert rt.flush_device_route_probes() == []


# ---------------------------------------------------------------------------
# 3 + 4. miss detection + recovery byte-identity, over sequences
# ---------------------------------------------------------------------------
def _scenario(kind, rows):
    """Return (route per layer, resident-per-layer) for a 3-layer 'token'."""
    experts_per_layer = {
        # layer -> flat list of rows*TOP_K expert ids
        0: [(i % EXPERT_COUNT) for i in range(rows * TOP_K)],
        1: [((i + 3) % EXPERT_COUNT) for i in range(rows * TOP_K)],
        2: [((i + 5) % EXPERT_COUNT) for i in range(rows * TOP_K)],
    }
    resident = {}
    for layer, experts in experts_per_layer.items():
        uniq = set(experts)
        if kind == "all_hit":
            res = {e: e + 1 for e in uniq}
        elif kind == "all_miss":
            res = {}
        else:  # mixed: layer 1 all-hit, layers 0/2 miss one expert each
            if layer == 1:
                res = {e: e + 1 for e in uniq}
            else:
                drop = sorted(uniq)[0]
                res = {e: e + 1 for e in uniq if e != drop}
        resident[layer] = res
    return experts_per_layer, resident


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("kind", ["all_hit", "mixed", "all_miss"])
def test_device_route_sequence_reconciles_to_fenced(kind, rows) -> None:
    experts_per_layer, resident = _scenario(kind, rows)
    rt = _DeviceRuntime([], resident)
    for layer in experts_per_layer:
        rt.register_component_bank(layer, _FAKE_BANK)

    fenced_seq = {}
    device_seq = {}
    inputs = {}
    for layer, experts in experts_per_layer.items():
        x, idx = _inputs(rows, experts)
        inputs[layer] = (x, idx)
        fenced_seq[layer] = _fenced_reference(x, idx)
        device_seq[layer] = _run_switch(rt, x, idx, device_route=True, layer=layer)

    # (3) the deferred probes flag exactly the non-resident experts per layer.
    misses = dict(rt.flush_device_route_probes())
    expected = {}
    for layer, experts in experts_per_layer.items():
        missed = tuple(sorted({e for e in experts if e not in resident[layer]}))
        if missed:
            expected[layer] = missed
    assert misses == expected, (kind, rows, misses, expected)

    # all-hit layers are already byte-identical; miss layers differ until recovery.
    for layer in experts_per_layer:
        if layer in expected:
            assert not mx.array_equal(device_seq[layer], fenced_seq[layer]), (
                f"{kind} L{layer}: optimistic miss output unexpectedly matched fenced"
            )
        else:
            assert mx.array_equal(device_seq[layer], fenced_seq[layer]), (
                f"{kind} L{layer}: all-hit device output != fenced"
            )

    # (4) recovery: admit the flagged misses and re-run -> byte-identical to fenced.
    reconciled = dict(device_seq)
    for layer, missed in expected.items():
        rt.admit(layer, missed)
        x, idx = inputs[layer]
        reconciled[layer] = _run_switch(rt, x, idx, device_route=True, layer=layer)
    for layer in experts_per_layer:
        assert mx.array_equal(reconciled[layer], fenced_seq[layer]), (
            f"{kind} L{layer}: reconciled != fenced"
        )


# ---------------------------------------------------------------------------
# 5. the LUT is rebuilt only when residency changes
# ---------------------------------------------------------------------------
def test_lut_refreshes_only_on_residency_change() -> None:
    rt = _DeviceRuntime([], {0: {1: 2, 2: 3}})
    a = rt.device_route_lut(0)
    b = rt.device_route_lut(0)
    assert a is b, "LUT rebuilt with no residency change"
    assert rt._lut_builds[0] == 1
    rt.admit(0, [5])            # residency changed -> dirty
    c = rt.device_route_lut(0)
    assert c is not a
    assert rt._lut_builds[0] == 2
    assert int(c[5].item()) == 6 and int(a[5].item()) == -1
