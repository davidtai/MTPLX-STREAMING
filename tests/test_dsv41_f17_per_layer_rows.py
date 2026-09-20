"""CPU tests for the F17 per-layer extension-row lever.

Covers (1) ``allocation.allocate`` modes/validation/determinism, (2) the
``stage_f17_runner`` byte-exact round-trip on the REAL archived ``extension.py``
plus double-apply refusal, and (3) executing BOTH the retained and the staged
``grow_rows`` against a minimal fake runtime: uniform mode must reproduce the
retained result exactly, and a non-uniform vector must produce the requested
per-layer capacities while leaving every plan/accounting total unchanged.

Pure CPU: no GPU, no Metal, no real MLX ops (grow_rows takes a stub ``mx`` and
fake bank/slot/policy objects injected via ``sys.modules``).
"""
from __future__ import annotations

import mlx.core as mx  # noqa: F401  (suite convention: pin MLX to the CPU device)

mx.set_default_device(mx.cpu)

import contextlib
import importlib.util
import sys
import threading
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_F17_DIR = _ROOT / "scripts" / "deepseek_v41" / "f17"
_ARCHIVED_EXTENSION = (
    _ROOT
    / "docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed/extension.py"
)

if str(_F17_DIR) not in sys.path:
    sys.path.insert(0, str(_F17_DIR))

import allocation  # noqa: E402
import stage_f17_runner as stager  # noqa: E402

# The F14 oracle allocation (added rows), clamped to the >=4 extension-lane floor
# and rebalanced to the exact total; regenerated in the report from results.json.
ORACLE_C108 = (
    "85,58,63,18,36,29,4,13,22,14,7,19,33,32,4,34,4,13,4,58,"
    "4,4,4,19,4,4,4,31,4,10,17,32,31,38,14,11,25,36,59,59"
)
ORACLE_C107 = (
    "81,56,60,17,34,28,4,12,21,13,7,18,32,31,4,33,4,12,4,55,"
    "4,4,4,18,4,4,4,30,4,10,16,31,30,36,13,10,24,34,57,57"
)


# --------------------------------------------------------------------------- #
# tiny fake runtime that satisfies allocation.allocate                        #
# --------------------------------------------------------------------------- #
def _alloc_runtime(n=40, freqs=None):
    spec = types.SimpleNamespace(routed_layer_indices=tuple(range(n)))
    banks = {
        i: types.SimpleNamespace(
            _prefill_route_freq=(freqs[i] if freqs is not None else Counter())
        )
        for i in range(n)
    }
    return types.SimpleNamespace(spec=spec, _banks=banks)


@pytest.fixture(autouse=True)
def _clear_f17_env(monkeypatch):
    monkeypatch.delenv(allocation.ALLOC_ENV, raising=False)
    monkeypatch.delenv(allocation.MAX_ADDED_ENV, raising=False)


# --------------------------------------------------------------------------- #
# 1. allocation.allocate                                                      #
# --------------------------------------------------------------------------- #
def test_uniform_default_matches_capacity_minus_old():
    added = allocation.allocate(_alloc_runtime(), capacity=108, old=84)
    assert set(added) == set(range(40))
    assert set(added.values()) == {24}
    assert sum(added.values()) == 40 * 24


def test_uniform_explicit_env(monkeypatch):
    monkeypatch.setenv(allocation.ALLOC_ENV, "uniform")
    added = allocation.allocate(_alloc_runtime(), capacity=111, old=84)
    assert set(added.values()) == {27}
    assert sum(added.values()) == 40 * 27


@pytest.mark.parametrize("vec,capacity", [(ORACLE_C108, 108), (ORACLE_C107, 107)])
def test_vector_mode_exact(monkeypatch, vec, capacity):
    monkeypatch.setenv(allocation.ALLOC_ENV, "vector:" + vec)
    added = allocation.allocate(_alloc_runtime(), capacity=capacity, old=84)
    want = [int(x) for x in vec.split(",")]
    assert [added[i] for i in range(40)] == want
    assert sum(added.values()) == 40 * (capacity - 84)
    assert min(added.values()) >= allocation.MIN_ADDED


def test_vector_rejects_bad_count(monkeypatch):
    monkeypatch.setenv(allocation.ALLOC_ENV, "vector:1,2,3")
    with pytest.raises(ValueError, match="exactly 40"):
        allocation.allocate(_alloc_runtime(), capacity=108, old=84)


def test_vector_rejects_sub_min(monkeypatch):
    bad = ",".join(["24"] * 39 + ["0"])  # a zero-capacity bank
    monkeypatch.setenv(allocation.ALLOC_ENV, "vector:" + bad)
    with pytest.raises(ValueError, match=r">= 4"):
        allocation.allocate(_alloc_runtime(), capacity=108, old=84)


def test_vector_rejects_wrong_sum(monkeypatch):
    bad = ",".join(["25"] * 40)  # sums to 1000 != 960
    monkeypatch.setenv(allocation.ALLOC_ENV, "vector:" + bad)
    with pytest.raises(ValueError, match="must sum to 960"):
        allocation.allocate(_alloc_runtime(), capacity=108, old=84)


def test_unknown_mode_refused(monkeypatch):
    monkeypatch.setenv(allocation.ALLOC_ENV, "belady")
    with pytest.raises(ValueError, match="not a recognised"):
        allocation.allocate(_alloc_runtime(), capacity=108, old=84)


def _diffuse_concentrated_freqs():
    """First 3 layers spread across all experts (diffuse); rest concentrated."""
    freqs = []
    for i in range(40):
        c = Counter()
        if i < 3:
            for e in range(256):
                c[e] = 1
        else:
            for e in range(40):
                c[e] = 100
        freqs.append(c)
    return freqs


def test_prefill_rule_bounds_quantum_and_sum(monkeypatch):
    monkeypatch.setenv(allocation.ALLOC_ENV, "prefill_rule")
    added = allocation.allocate(
        _alloc_runtime(freqs=_diffuse_concentrated_freqs()), capacity=108, old=84
    )
    vals = [added[i] for i in range(40)]
    assert sum(vals) == 40 * 24
    assert all(v % allocation.QUANTUM == 0 for v in vals)
    assert all(allocation.MIN_ADDED <= v <= allocation.DEFAULT_MAX_ADDED for v in vals)
    # diffuse layers must receive strictly more than the concentrated ones
    assert min(vals[:3]) > max(vals[3:])


def test_prefill_rule_deterministic(monkeypatch):
    monkeypatch.setenv(allocation.ALLOC_ENV, "prefill_rule")
    freqs = _diffuse_concentrated_freqs()
    a = allocation.allocate(_alloc_runtime(freqs=freqs), capacity=108, old=84)
    b = allocation.allocate(_alloc_runtime(freqs=freqs), capacity=108, old=84)
    assert [a[i] for i in range(40)] == [b[i] for i in range(40)]


def test_prefill_rule_respects_max_added_env(monkeypatch):
    monkeypatch.setenv(allocation.ALLOC_ENV, "prefill_rule")
    monkeypatch.setenv(allocation.MAX_ADDED_ENV, "40")
    added = allocation.allocate(
        _alloc_runtime(freqs=_diffuse_concentrated_freqs()), capacity=108, old=84
    )
    vals = [added[i] for i in range(40)]
    assert max(vals) <= 40
    assert sum(vals) == 40 * 24


def test_prefill_rule_empty_freq_fails(monkeypatch):
    monkeypatch.setenv(allocation.ALLOC_ENV, "prefill_rule")
    with pytest.raises(RuntimeError, match="empty _prefill_route_freq"):
        allocation.allocate(_alloc_runtime(), capacity=108, old=84)


def test_capacity_not_greater_than_old_refused():
    with pytest.raises(ValueError, match="capacity>old"):
        allocation.allocate(_alloc_runtime(), capacity=84, old=84)


# --------------------------------------------------------------------------- #
# 2. stager round-trip on the REAL archived extension.py                      #
# --------------------------------------------------------------------------- #
def _archived_source():
    return _ARCHIVED_EXTENSION.read_text()


def test_stager_roundtrip_and_marker():
    src = _archived_source()
    staged = stager.stage(src)  # stage() asserts the byte-exact reverse internally
    assert staged != src
    assert stager._MARKER in staged
    # reversing all six edits recovers the input byte-for-byte
    recovered = staged
    for _label, old, new in reversed(stager._EDITS):
        recovered = recovered.replace(new, old)
    assert recovered == src


def test_stager_double_apply_refused():
    staged = stager.stage(_archived_source())
    with pytest.raises(RuntimeError, match="already applied"):
        stager.stage(staged)


def test_staged_extension_compiles():
    compile(stager.stage(_archived_source()), "<staged-extension>", "exec")


def test_stager_cli_refuses_in_place(tmp_path):
    p = tmp_path / "extension.py"
    p.write_text(_archived_source())
    with pytest.raises(RuntimeError, match="must differ"):
        stager.main(["--extension", str(p), "--out", str(p)])


def test_stager_scalar_route_capacity_untouched():
    """The report-only scalar pool._persistent_route_capacity stays uniform."""
    staged = stager.stage(_archived_source())
    assert "pool._persistent_route_capacity=capacity" in staged  # scalar line kept
    assert "pool._persistent_route_capacities={layer:old+_f17_added[layer]" in staged


# --------------------------------------------------------------------------- #
# 3. execute retained vs staged grow_rows on a fake runtime                   #
# --------------------------------------------------------------------------- #
_W = 1000  # fake per-row weight bytes (any positive int; accounting is self-consistent)


class _FakeArray:
    def __init__(self, nbytes):
        self.nbytes = nbytes


class _FakeBank:
    def __init__(self, capacity, record=None, label=None):
        self.capacity = capacity
        self.label = label
        self.record_bytes = _W
        # After remove_raw_scales a bank holds exactly capacity*_W weight bytes.
        self.arrays = {"weight": _FakeArray(capacity * _W)}


class _FakeSlot:
    def __init__(self, bank, index, label=None):
        self.bank = bank
        self.bank_index = index
        self.label = label


class _FakePhysical:
    def __init__(self, label, buffer):
        self.label = label
        self.buffer = buffer
        self.pins = 0
        self.pin_claims = 0
        self.state = "READY"


class _SlotState:
    LOADING = object()  # a sentinel distinct from every physical slot's state


def _remove_raw_scales(bank, mx=None):  # fake already sized at weight bytes
    return 0


@dataclass
class _FakePlan:
    slots_per_layer: int
    persistent_slots: int
    persistent_cache_bytes: int
    transient_slots: int
    transient_bytes: int
    cache_scope: str
    persistent_slots_by_layer: tuple
    prefetch_ring_slots: int
    persistent_budget_bytes: int
    expert_cache_limit_bytes: object
    allocated_bytes: int
    unallocated_bytes: int
    total_limit_bytes: int


@dataclass
class _FakeConfig:
    slot_layout: str
    cache_policy: str
    split_route_release: str
    memory_limit_bytes: int
    expert_cache_limit_bytes: object


class _FakeAllocator:
    def __init__(self, banks, plan):
        self.banks = banks
        self.slots = {}
        self.plan = plan


class _FakeMetrics:
    def as_dict(self):
        return {"active_routes": 0}


class _FakePool:
    def __init__(self, banks, plan, persistent, transient, record_map, ensure_locks, allocated_bytes):
        self._allocator = _FakeAllocator(banks, plan)
        self._persistent = persistent
        self._transient = transient
        self._prefetch = {}
        self._record_map = record_map
        self._ensure_locks = ensure_locks
        self._lifecycle = contextlib.nullcontext()
        self._closed = False
        self._closing = False
        self._cleanup_owners = []
        self.metrics = _FakeMetrics()
        self.allocated_bytes = allocated_bytes
        self.plan = plan
        self._persistent_route_capacity = plan.slots_per_layer
        self._persistent_route_capacities = {}

    def _drain_completion_fences(self):
        pass


@dataclass
class _FakePolicy:
    persistent_slots: int
    _persistent_capacity: int
    slot_count: int
    _protected_cap: int
    prefetch_slots: int
    _slot_to_expert: list
    _prefill_route_freq: Counter


class _FakeRuntime:
    def __init__(self, pool, plan, config, spec, banks, layer_locks):
        self.slots = pool
        self.plan = plan
        self.config = config
        self.spec = spec
        self._banks = banks
        self._layer_locks = layer_locks
        self._prefetch_ring = None
        self._global_bank = None
        self._single_slot_pool = True
        self._device_route_lut = {}
        self._device_route_lut_snapshot = {}
        self._device_route_pinned_lut = {}
        self._device_route_pinned_snapshot = {}
        self._device_route_lut_dirty = {}
        self._device_route_pinned_lut_dirty = {}

    def flush_deferred_slot_releases(self, evaluate=False):
        pass

    def _drain_prefetch_loads(self):
        pass

    def _raise_if_unhealthy(self):
        pass


def _build_runtime(*, old=84, n=40, transient=48, freqs=None):
    layers = tuple(range(n))
    banks = {}
    persistent = {}
    for layer in layers:
        pbank = _FakeBank(old, label=f"persistent-{layer}")
        banks[("persistent", layer)] = pbank
        for row in range(old):
            buf = _FakeSlot(pbank, row, label=f"p-{layer}-{row}")
            persistent[(layer, row)] = _FakePhysical(f"p-{layer}-{row}", buf)
    tbank = _FakeBank(transient, label="transient")
    banks[("transient", -1)] = tbank
    transient_slots = [
        _FakePhysical(f"t-{i}", _FakeSlot(tbank, i)) for i in range(transient)
    ]
    expected_bytes = (n * old + transient) * _W
    slack = 8 * 1024 ** 3
    plan = _FakePlan(
        slots_per_layer=old,
        persistent_slots=n * old,
        persistent_cache_bytes=n * old * _W,
        transient_slots=transient,
        transient_bytes=transient * _W,
        cache_scope="layer",
        persistent_slots_by_layer=(),
        prefetch_ring_slots=0,
        persistent_budget_bytes=n * old * _W,
        expert_cache_limit_bytes=None,
        allocated_bytes=expected_bytes,
        unallocated_bytes=slack,
        total_limit_bytes=expected_bytes + slack,
    )
    config = _FakeConfig(
        slot_layout="component-banks",
        cache_policy="transition-window",
        split_route_release="deferred",
        memory_limit_bytes=expected_bytes + slack,
        expert_cache_limit_bytes=None,
    )
    pool = _FakePool(
        banks, plan, persistent, transient_slots,
        record_map={(layer, 0): object() for layer in layers},
        ensure_locks={layer: threading.Lock() for layer in layers},
        allocated_bytes=expected_bytes,
    )
    spec = types.SimpleNamespace(
        hidden_size=5120, expert_hidden_size=2304, top_k=6,
        expert_codec="mxfp4", routed_layer_indices=layers,
    )
    policies = {
        layer: _FakePolicy(
            persistent_slots=old, _persistent_capacity=old, slot_count=old + transient,
            _protected_cap=max(1, int(old * 0.8)), prefetch_slots=0,
            _slot_to_expert=[None] * old,
            _prefill_route_freq=(freqs[layer] if freqs is not None else Counter()),
        )
        for layer in layers
    }
    return _FakeRuntime(
        pool, plan, config, spec, policies,
        layer_locks={layer: threading.Lock() for layer in layers},
    )


@contextlib.contextmanager
def _fake_runtime_deps():
    """Inject fake mtplx/packed_storage modules so grow_rows' inner imports resolve."""
    expert_mlx = types.ModuleType("mtplx.models.expert_mlx")
    expert_mlx.MlxComponentBank = _FakeBank
    expert_mlx.MlxComponentSlot = _FakeSlot
    expert_slots = types.ModuleType("mtplx.expert_slots")
    expert_slots._PhysicalSlot = _FakePhysical
    expert_slots.ExpertSlotState = _SlotState
    packed_storage = types.ModuleType("packed_storage")
    packed_storage.remove_raw_scales = _remove_raw_scales
    packed_storage.WEIGHT_BYTES = _W
    mtplx = types.ModuleType("mtplx")
    models = types.ModuleType("mtplx.models")
    models.expert_mlx = expert_mlx
    mtplx.models = models
    mtplx.expert_slots = expert_slots
    injected = {
        "mtplx": mtplx,
        "mtplx.models": models,
        "mtplx.models.expert_mlx": expert_mlx,
        "mtplx.expert_slots": expert_slots,
        "packed_storage": packed_storage,
    }
    saved = {k: sys.modules.get(k) for k in injected}
    sys.modules.update(injected)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _load_module(name, text):
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(text, f"<{name}>", "exec"), module.__dict__)
    return module


_FAKE_MX = types.SimpleNamespace(synchronize=lambda: None)


def _run_grow(module_text, *, capacity, env=None, monkeypatch, freqs=None):
    if env is None:
        monkeypatch.delenv(allocation.ALLOC_ENV, raising=False)
    else:
        monkeypatch.setenv(allocation.ALLOC_ENV, env)
    module = _load_module("f17_ext_under_test", module_text)
    rt = _build_runtime(freqs=freqs)
    with _fake_runtime_deps():
        report = module.grow_rows(rt, capacity=capacity, layout="extension", mx=_FAKE_MX)
    return rt, report


def _state(rt):
    pool = rt.slots
    rows = {layer: 0 for (layer, _row) in pool._persistent}
    for (layer, _row) in pool._persistent:
        rows[layer] += 1
    return {
        "route_caps": dict(pool._persistent_route_capacities),
        "route_cap_scalar": pool._persistent_route_capacity,
        "pool_allocated_bytes": pool.allocated_bytes,
        "rows_per_layer": rows,
        "plan_persistent_slots": rt.plan.persistent_slots,
        "plan_persistent_cache_bytes": rt.plan.persistent_cache_bytes,
        "plan_total_limit_bytes": rt.plan.total_limit_bytes,
        "plan_slots_per_layer": rt.plan.slots_per_layer,
        "plan_by_layer": rt.plan.persistent_slots_by_layer,
        "policy": {
            layer: (
                rt._banks[layer].persistent_slots,
                rt._banks[layer]._persistent_capacity,
                rt._banks[layer].slot_count,
                rt._banks[layer]._protected_cap,
                len(rt._banks[layer]._slot_to_expert),
            )
            for layer in range(40)
        },
    }


def test_grow_rows_uniform_matches_retained(monkeypatch):
    retained_text = _archived_source()
    staged_text = stager.stage(retained_text)
    rt_ret, rep_ret = _run_grow(retained_text, capacity=108, env=None, monkeypatch=monkeypatch)
    rt_stg, rep_stg = _run_grow(staged_text, capacity=108, env="uniform", monkeypatch=monkeypatch)
    assert _state(rt_ret) == _state(rt_stg)
    # every layer uniform at 108, totals as the uniform path produces
    st = _state(rt_stg)
    assert set(st["route_caps"].values()) == {108}
    assert st["route_cap_scalar"] == 108
    assert st["plan_persistent_slots"] == 40 * 108
    assert st["plan_slots_per_layer"] == 108
    assert st["plan_by_layer"] == ()  # kept empty -> run_full gates pass
    for key in ("physical_allocated_bytes", "capacity", "layers", "added_payload_bytes"):
        assert rep_ret[key] == rep_stg[key]
    assert rep_stg["existing_row_owners_unchanged"] is True


def test_grow_rows_vector_per_layer_totals_unchanged(monkeypatch):
    staged_text = stager.stage(_archived_source())
    # a clear non-uniform split summing to 40*24 = 960 (== uniform total)
    added = [44, 4] + [24] * 38
    assert sum(added) == 40 * 24 and min(added) >= 4
    vec = ",".join(str(x) for x in added)
    rt_uni, _ = _run_grow(staged_text, capacity=108, env="uniform", monkeypatch=monkeypatch)
    rt_vec, rep = _run_grow(staged_text, capacity=108, env="vector:" + vec, monkeypatch=monkeypatch)
    st = _state(rt_vec)
    # per-layer capacities are exactly 84 + added_L
    assert st["route_caps"][0] == 84 + 44
    assert st["route_caps"][1] == 84 + 4
    assert st["route_caps"][2] == 84 + 24
    assert st["rows_per_layer"][0] == 84 + 44
    assert st["rows_per_layer"][1] == 84 + 4
    # per-layer policy fields track c_L
    assert st["policy"][0] == (128, 128, 128 + 48, max(1, int(128 * 0.8)), 128)
    assert st["policy"][1] == (88, 88, 88 + 48, max(1, int(88 * 0.8)), 88)
    # every total is identical to the uniform run (admission math untouched)
    su = _state(rt_uni)
    assert st["plan_persistent_slots"] == su["plan_persistent_slots"] == 40 * 108
    assert st["plan_persistent_cache_bytes"] == su["plan_persistent_cache_bytes"]
    assert st["plan_total_limit_bytes"] == su["plan_total_limit_bytes"]
    assert st["pool_allocated_bytes"] == su["pool_allocated_bytes"]
    assert st["plan_slots_per_layer"] == 108  # uniform-equivalent scalar
    assert st["plan_by_layer"] == ()          # still empty -> gates pass
    assert rep["existing_row_owners_unchanged"] is True
    assert rep["physical_allocated_bytes"] == st["pool_allocated_bytes"]


def test_grow_rows_executes_f14_oracle_vector(monkeypatch):
    """The deliverable oracle string runs end-to-end through the staged grow_rows."""
    staged_text = stager.stage(_archived_source())
    rt, rep = _run_grow(
        staged_text, capacity=108, env="vector:" + ORACLE_C108, monkeypatch=monkeypatch
    )
    want = [int(x) for x in ORACLE_C108.split(",")]
    st = _state(rt)
    for layer in range(40):
        assert st["route_caps"][layer] == 84 + want[layer]
        assert st["rows_per_layer"][layer] == 84 + want[layer]
    assert st["plan_persistent_slots"] == 40 * 108  # total unchanged
    assert rep["physical_allocated_bytes"] == st["pool_allocated_bytes"]


def test_shape_mode_rescales_a_profile_to_any_admitted_capacity(monkeypatch):
    """``shape:<40 weights>`` apportions a fixed per-layer profile to whatever capacity the
    live admission picked: exact totals, multiples of 4, floors respected, deterministic."""
    import types

    from allocation import ALLOC_ENV, allocate

    oracle = [85,58,63,18,36,29,4,13,22,14,7,19,33,32,4,34,4,13,4,58,4,4,4,19,4,4,4,31,4,10,
              17,32,31,38,14,11,25,36,59,59]
    weights = ",".join(str(v - 4) for v in oracle)
    monkeypatch.setenv(ALLOC_ENV, "shape:" + weights)
    runtime = types.SimpleNamespace(spec=types.SimpleNamespace(routed_layer_indices=tuple(range(40))))
    for capacity in (105, 106, 107, 108, 111):
        added = allocate(runtime, capacity=capacity)
        values = [added[layer] for layer in range(40)]
        assert sum(values) == 40 * (capacity - 84)
        assert all(v >= 4 and v % 4 == 0 for v in values)
        assert values == [allocate(runtime, capacity=capacity)[layer] for layer in range(40)]
        assert values[0] == max(values) and values[20] == 4      # L0 heaviest, L20 at the floor
