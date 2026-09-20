"""CPU tests for the F33 host trims (scripts/deepseek_v41/f16).

Two construction-time, bit-exact host trims for the F16 two-group verify pipeline:

  * ``flat_indices`` -- a derived-run step that evaluates ``indices.reshape(-1)`` inside
    the routing barrier (``flat_indices = indices.reshape(-1)`` / ``mx.eval(indices,
    flat_indices)``) and reads that view in the experts tuple (``tuple(flat_indices
    .tolist())``), removing the second scheduler round trip per group slice.  Tested at the
    source level and the build (compile) level across all four {reads, barrier} x {flat
    off, flat on} modes, each with and without stamps; flat OFF is byte-identical to the
    stock derivation (no digest regression); the barrier+flat head is asserted verbatim.
  * ``fast_observe`` (``f16/host_trims.py``) -- a per-bank, STATE-IDENTICAL replacement of
    ``LayerExpertSlotBank._observe_transition_window`` whose 2-D ``np.ix_`` fancy add
    becomes a 1-D fancy add on the flattened counts view.  Proven against the REAL bank
    over 2,000 seeded random decode routes (plans + every policy array compared after each
    ``plan()``); both variants timed (us/call; no speed assertion).  Refusals: wrong
    policy, wrong shape/dtype/contiguity, empty banks, double install, validate-first.

Pins MLX to CPU before importing ``f16.pipeline`` (greenlet + ``mlx.core``);
``mtplx.expert_streaming`` imports no MLX, so the real ``LayerExpertSlotBank`` runs on
pure numpy.  No GPU/Metal, no artifact.
"""
from __future__ import annotations

import inspect
import os
import sys
import textwrap
import time
import types
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)  # BEFORE importing f16.pipeline (imports mlx.core + greenlet)

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts" / "deepseek_v41"
_PACKED = _REPO / "docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed"
# .f16-site holds greenlet (private dir, NOT the shared venv); honour F16SITE like the
# F16/F18/F20/F24 tests (default = the F16 pipeline worktree's dir).
_F16_SITE = os.environ.get(
    "F16SITE",
    "/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f16-pipeline/.f16-site",
)
for _p in (str(_SCRIPTS), str(_PACKED), str(_F16_SITE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import greenlet  # noqa: E402, F401  (from .f16-site; f16.pipeline imports it)

from mtplx.expert_streaming import (  # noqa: E402
    LayerExpertSlotBank,
    RoutingPhase,
    TRANSITION_WINDOW_CACHE_POLICY,
)

from f16 import host_trims as ht  # noqa: E402
from f16 import install as f16_install  # noqa: E402
from f16 import pipeline as pl  # noqa: E402


def _sched_source() -> str:
    import plane_lane
    import projection_install

    base = textwrap.dedent(inspect.getsource(plane_lane.PackedDecode.run))
    return projection_install.scheduled_run_source(base)


# ---------------------------------------------------------------------------
# 1. flat_indices source derivation: both replacements once + round-trip
# ---------------------------------------------------------------------------
def _reverse_flat(flat: str) -> str:
    """Reverse the two F33 replacements line-by-line (indent-agnostic) so the result can
    be compared against the un-derived scheduled source."""
    out: list[str] = []
    lines = flat.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        indent = ln[: len(ln) - len(ln.lstrip())]
        if ln.strip() == "flat_indices = indices.reshape(-1)":
            assert lines[i + 1].strip() == "mx.eval(indices, flat_indices)"
            out.append(indent + "mx.eval(indices)")
            i += 2
            continue
        if ln.strip() == "experts = tuple(flat_indices.tolist())":
            out.append(indent + "experts = tuple(int(e) for e in indices.reshape(-1).tolist())")
            i += 1
            continue
        out.append(ln)
        i += 1
    return "\n".join(out)


def test_flat_indices_source_replaces_both_lines_once_and_roundtrips():
    sched = _sched_source()
    flat = pl.f16_flat_indices_source(sched)
    assert flat.count("flat_indices = indices.reshape(-1)") == 1
    assert flat.count("mx.eval(indices, flat_indices)") == 1
    assert flat.count("experts = tuple(flat_indices.tolist())") == 1
    strips = [ln.strip() for ln in flat.splitlines()]
    # the two rewritten originals are gone as standalone lines
    assert "mx.eval(indices)" not in strips
    assert "experts = tuple(int(e) for e in indices.reshape(-1).tolist())" not in strips
    # the reshape line is immediately followed by the combined eval
    lines = flat.splitlines()
    ri = next(i for i, ln in enumerate(lines) if ln.strip() == "flat_indices = indices.reshape(-1)")
    assert lines[ri + 1].strip() == "mx.eval(indices, flat_indices)"
    # round-trip: reversing both replacements recovers the scheduled source line-for-line
    # (like the sibling derivations, _replace_line normalises the source's trailing newline
    # via "\n".join, so the comparison is on splitlines, not the raw string).
    assert _reverse_flat(flat).splitlines() == sched.splitlines()


def test_flat_indices_source_refuses_missing_or_duplicate_anchor():
    sched = _sched_source()
    # double apply: both anchors are gone after the first derivation -> refuse
    flat = pl.f16_flat_indices_source(sched)
    with pytest.raises(RuntimeError, match="flat-eval anchor is not unique"):
        pl.f16_flat_indices_source(flat)
    # a source missing the eval anchor
    missing = "\n".join(ln for ln in sched.splitlines() if ln.strip() != "mx.eval(indices)")
    with pytest.raises(RuntimeError, match="flat-eval anchor is not unique"):
        pl.f16_flat_indices_source(missing)
    # a source with the experts anchor duplicated (eval is still unique, so flat-eval
    # succeeds and flat-experts is the one that finds two)
    dup: list[str] = []
    for ln in sched.splitlines():
        dup.append(ln)
        if ln.strip() == "experts = tuple(int(e) for e in indices.reshape(-1).tolist())":
            dup.append(ln)
    with pytest.raises(RuntimeError, match="flat-experts anchor is not unique"):
        pl.f16_flat_indices_source("\n".join(dup))


@pytest.mark.parametrize("dtype", [mx.uint32, mx.int32])
def test_flat_tolist_equals_old_tuple_elementwise_and_is_python_ints(dtype):
    """The trim reads ``tuple(flat_indices.tolist())`` where ``flat_indices =
    indices.reshape(-1)``; on a uint32/int32 MLX array this must equal the retained
    ``tuple(int(e) for e in indices.reshape(-1).tolist())`` element-for-element, and the
    elements must be plain Python ints (so the routed expert ids are unchanged)."""
    raw = np.array([[3, 0, 383, 128], [7, 255, 42, 6]], dtype=np.int64)
    indices = mx.array(raw).astype(dtype)
    mx.eval(indices)
    flat_indices = indices.reshape(-1)
    mx.eval(indices, flat_indices)  # the exact eval the derived run performs
    old = tuple(int(e) for e in indices.reshape(-1).tolist())
    new = tuple(flat_indices.tolist())
    assert new == old
    assert all(type(x) is int for x in new), "tolist() did not return plain Python ints"
    assert new == tuple(raw.reshape(-1).tolist())


# ---------------------------------------------------------------------------
# 2. barrier+flat exact derived text (the order the orchestrator specified)
# ---------------------------------------------------------------------------
def test_barrier_flat_derived_head_is_exact():
    sched = _sched_source()
    derived = pl.f16_barrier_run_source(sched, flat_indices=True)
    strips = [ln.strip() for ln in derived.splitlines()]
    ri = strips.index("flat_indices = indices.reshape(-1)")
    assert strips[ri + 1] == "self._f16_barrier(flat_indices)"
    assert strips[ri + 2] == "mx.eval(indices, flat_indices)"
    assert "experts = tuple(flat_indices.tolist())" in strips
    assert derived.count("self._f16_yield()") == 1
    # the stock lines are gone
    assert "mx.eval(indices)" not in strips
    assert "self._f16_barrier(indices)" not in strips
    assert "experts = tuple(int(e) for e in indices.reshape(-1).tolist())" not in strips


# ---------------------------------------------------------------------------
# 3. all four {reads, barrier} x {flat off, flat on} modes, each +/- stamps: compile
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("barrier", [False, True])
@pytest.mark.parametrize("stamps", [False, True])
def test_all_modes_derive_and_compile(flat, barrier, stamps):
    sched = _sched_source()
    base = (pl.f16_barrier_run_source(sched, flat_indices=flat) if barrier
            else pl.f16_run_source(sched, flat_indices=flat))
    derived = pl.f16_stamp_run_source(base, flat_indices=flat) if stamps else base

    assert derived.count("self._f16_yield()") == 1
    if flat:
        assert derived.count("flat_indices = indices.reshape(-1)") == 1
        assert derived.count("mx.eval(indices, flat_indices)") == 1
        assert derived.count("experts = tuple(flat_indices.tolist())") == 1
        assert "experts = tuple(int(e) for e in indices.reshape(-1).tolist())" not in derived
    else:
        assert "flat_indices" not in derived
        assert any(ln.strip() == "mx.eval(indices)" for ln in derived.splitlines())
    if barrier:
        want = "self._f16_barrier(flat_indices)" if flat else "self._f16_barrier(indices)"
        assert derived.count(want) == 1
    else:
        assert "self._f16_barrier" not in derived
    if stamps:
        assert derived.count("self._f16_stamp3()") == 1
        assert derived.count("self._f16_stamproute(experts, parts, pending)") == 1

    # compiles in plane_lane's namespace (the identical helpers the run closes over)
    import plane_lane

    ns = dict(plane_lane.__dict__)
    exec(compile(derived, "<f33_test>", "exec"), ns)  # noqa: S102
    assert callable(ns["run"])


# ---------------------------------------------------------------------------
# 4. build level: flat on/off compile distinctly, flat OFF == stock (no regression)
# ---------------------------------------------------------------------------
def test_build_yield_and_barrier_flat_on_off():
    off_fn, off = pl.build_yield_run(flat_indices=False)
    on_fn, on = pl.build_yield_run(flat_indices=True)
    assert callable(off_fn) and callable(on_fn)
    assert off["flat_indices"] is False and on["flat_indices"] is True
    assert on_fn.__code__.co_code != off_fn.__code__.co_code
    # flat OFF matches the no-arg default build EXACTLY (existing digest is preserved)
    assert off["f16_yield_run_sha256"] == pl.build_yield_run()[1]["f16_yield_run_sha256"]
    assert on["f16_yield_run_sha256"] != off["f16_yield_run_sha256"]

    b_off_fn, b_off = pl.build_barrier_run(flat_indices=False)
    b_on_fn, b_on = pl.build_barrier_run(flat_indices=True)
    assert b_on_fn.__code__.co_code != b_off_fn.__code__.co_code
    assert b_off["f16_barrier_run_sha256"] == pl.build_barrier_run()[1]["f16_barrier_run_sha256"]
    assert b_on["f16_barrier_run_sha256"] != b_off["f16_barrier_run_sha256"]


@pytest.mark.parametrize("barrier", [False, True])
def test_build_stamped_flat_on_off_compiles(barrier):
    off_fn, off = pl.build_stamped_run(barrier=barrier, flat_indices=False)
    on_fn, on = pl.build_stamped_run(barrier=barrier, flat_indices=True)
    assert callable(off_fn) and callable(on_fn)
    assert off["flat_indices"] is False and on["flat_indices"] is True
    assert on_fn.__code__.co_code != off_fn.__code__.co_code
    key = "f16_stamped_barrier_run_sha256" if barrier else "f16_stamped_reads_run_sha256"
    assert off[key] == pl.build_stamped_run(barrier=barrier)[1][key]


def test_flat_off_is_the_stock_derivation():
    """Flat OFF is byte-identical to the stock derivation (no regression for the existing
    reads/barrier digests): the stock lines are present, no ``flat_indices`` appears, and
    removing the barrier + yield inserts recovers the scheduled source."""
    sched = _sched_source()
    reads = pl.f16_run_source(sched)          # default flat off
    barrier = pl.f16_barrier_run_source(sched)
    assert "flat_indices" not in reads and "flat_indices" not in barrier
    assert any(ln.strip() == "mx.eval(indices)" for ln in reads.splitlines())
    assert any(ln.strip() == "self._f16_barrier(indices)" for ln in barrier.splitlines())
    recovered = [ln for ln in barrier.splitlines()
                 if ln.strip() not in ("self._f16_barrier(indices)", "self._f16_yield()")]
    assert recovered == sched.splitlines()


# ---------------------------------------------------------------------------
# 5. fast_observe: state-identical against the REAL bank over 2,000 routes
# ---------------------------------------------------------------------------
class _Runtime:
    def __init__(self, banks):
        self._banks = banks


def _real_transition_bank(transient_slots: int = 24) -> LayerExpertSlotBank:
    # transition-window requires single_pool=True and exactly 384 experts; transient_slots
    # must cover the route width.  The F24 sibling test uses transient_slots=8, which caps a
    # route at 8 unique experts; the F33 equivalence drives 6-24-expert routes through
    # plan(), so it widens transient_slots to 24 (otherwise every other construction arg
    # matches F24's real transition bank).
    return LayerExpertSlotBank(
        expert_count=384,
        persistent_slots=8,
        transient_slots=transient_slots,
        cache_policy=TRANSITION_WINDOW_CACHE_POLICY,
        single_pool=True,
        layer_id=0,
    )


def test_fast_observe_is_state_identical_over_2000_routes():
    ref = _real_transition_bank()
    fast = _real_transition_bank()
    frag = ht.install_fast_observe(_Runtime({0: fast}))
    assert frag == {"fast_observe_banks": 1}
    # the replacement is an instance attribute shadowing the class method
    assert "_observe_transition_window" in fast.__dict__
    assert "_observe_transition_window" not in ref.__dict__

    rng = np.random.RandomState(20260920)
    for i in range(2000):
        k = int(rng.randint(6, 25))  # 6..24 unique experts
        route = [int(x) for x in rng.choice(384, size=k, replace=False)]
        pr = ref.plan(list(route), phase=RoutingPhase.DECODE)
        pf = fast.plan(list(route), phase=RoutingPhase.DECODE)
        assert pr.misses == pf.misses, f"misses differ at route {i}"
        assert pr.hits == pf.hits, f"hits differ at route {i}"
        assert pr.loads == pf.loads, f"loads differ at route {i}"
        assert pr.evictions == pf.evictions, f"evictions differ at route {i}"
        assert np.array_equal(ref._transition_counts, fast._transition_counts), \
            f"_transition_counts differ at route {i}"
        assert np.array_equal(ref._transition_denominators, fast._transition_denominators), \
            f"_transition_denominators differ at route {i}"
        assert np.array_equal(ref._transition_window_frequency, fast._transition_window_frequency), \
            f"_transition_window_frequency differ at route {i}"
        assert ref._transition_previous == fast._transition_previous, \
            f"_transition_previous differ at route {i}"
        assert list(ref._transition_window) == list(fast._transition_window), \
            f"window differ at route {i}"
    # the outer-product update was genuinely exercised (many cells, repeated accumulation)
    assert float(ref._transition_counts.max()) > 1.0
    assert int((ref._transition_counts != 0).sum()) > 1000


def _measure_observe(*, fast: bool, iters: int = 40000) -> float:
    bank = _real_transition_bank()
    if fast:
        ht.install_fast_observe(_Runtime({0: bank}))
    bank._transition_previous = tuple(range(6, 18))  # 12 previous experts
    cur = tuple(range(0, 18))                          # 18 current experts (216-cell add)
    t0 = time.perf_counter()
    for _ in range(iters):
        bank._observe_transition_window(cur)
    return (time.perf_counter() - t0) / iters * 1e6


def test_fast_observe_timing_is_reported():
    """Time both variants (no speed assertion; host-encode sensitive).  The numbers are
    printed for the F33 report."""
    pinned_us = _measure_observe(fast=False)
    fast_us = _measure_observe(fast=True)
    print(f"F33_OBSERVE_TIMING pinned_us_per_call={pinned_us:.4f} fast_us_per_call={fast_us:.4f}")
    assert pinned_us > 0.0 and fast_us > 0.0


# ---------------------------------------------------------------------------
# 6. fast_observe refusals: policy, shapes/dtype/contiguity, empty, validate-first, double
# ---------------------------------------------------------------------------
def _fake_tw_bank(*, counts, denoms):
    bank = types.SimpleNamespace()
    bank.cache_policy = TRANSITION_WINDOW_CACHE_POLICY
    bank._transition_counts = counts
    bank._transition_denominators = denoms
    return bank


@pytest.mark.parametrize("policy", ["frequency", "lru", "transition-window-tuned"])
def test_fast_observe_refuses_non_plain_policy(policy):
    bank = LayerExpertSlotBank(
        expert_count=384, persistent_slots=8, transient_slots=8,
        cache_policy=policy, single_pool=True, layer_id=0,
    )
    with pytest.raises(RuntimeError, match="cache_policy"):
        ht.install_fast_observe(_Runtime({0: bank}))


def _good_counts():
    return np.zeros((384, 384), dtype=np.float32)


def _good_denoms():
    return np.zeros(384, dtype=np.float32)


@pytest.mark.parametrize("counts,denoms,match", [
    (None, _good_denoms(), "_transition_counts is None"),
    (np.zeros((383, 384), dtype=np.float32), _good_denoms(), "_transition_counts shape"),
    (np.zeros((384, 384), dtype=np.float64), _good_denoms(), "_transition_counts dtype"),
    (np.zeros((384, 768), dtype=np.float32)[:, ::2], _good_denoms(), "C-contiguous"),
    (_good_counts(), None, "_transition_denominators shape"),
    (_good_counts(), np.zeros(383, dtype=np.float32), "_transition_denominators shape"),
])
def test_fast_observe_refuses_wrong_counts_or_denoms(counts, denoms, match):
    bank = _fake_tw_bank(counts=counts, denoms=denoms)
    with pytest.raises(RuntimeError, match=match):
        ht.install_fast_observe(_Runtime({0: bank}))


def test_fast_observe_refuses_empty_banks():
    with pytest.raises(RuntimeError, match="empty"):
        ht.install_fast_observe(_Runtime({}))


def test_fast_observe_validates_all_before_applying_any():
    """A bad bank sandwiched between good ones must leave NO bank patched (validate-first
    then apply)."""
    good_a = _real_transition_bank()
    bad = _fake_tw_bank(counts=None, denoms=_good_denoms())
    good_b = _real_transition_bank()
    with pytest.raises(RuntimeError, match="_transition_counts is None"):
        ht.install_fast_observe(_Runtime({0: good_a, 7: bad, 14: good_b}))
    assert "_observe_transition_window" not in good_a.__dict__
    assert "_observe_transition_window" not in good_b.__dict__


def test_fast_observe_refuses_double_install():
    bank = _real_transition_bank()
    ht.install_fast_observe(_Runtime({0: bank}))
    with pytest.raises(RuntimeError, match="already"):
        ht.install_fast_observe(_Runtime({0: bank}))


# ---------------------------------------------------------------------------
# 7. install() integration: report keys for flat_indices + fast_observe
# ---------------------------------------------------------------------------
class _StubPipeline:
    """Stands in for the armed ``Pipeline`` so ``install(armed=True)`` runs without a
    staged model.  Exposes exactly the attributes the install report reads."""

    def __init__(self, model, *, armed, split="fixed4", handoff="reads",
                 engram_lookahead=False):
        self._skip = 1
        self.engram_lookahead = False
        self.split = split


def _patch_gates(monkeypatch):
    monkeypatch.setattr(f16_install, "Pipeline", _StubPipeline)
    monkeypatch.setattr(f16_install, "verify_source_pins", lambda: {"stub": "pin"})
    monkeypatch.setattr(f16_install, "_assert_scheduled_lane", lambda target: 3)
    monkeypatch.setattr(f16_install, "_collect_runners", lambda target: {})
    monkeypatch.setattr(f16_install, "_wrap_issue_next_for_trailer", lambda runners: 0)
    monkeypatch.setattr(f16_install, "pipeline_a_rows", lambda: 4)
    for env in ("MTPLX_DSV41_DEVICE_ROUTE", "MTPLX_DSV41_DEVICE_ROUTE_PINNED"):
        monkeypatch.delenv(env, raising=False)


def _armed_target(banks):
    runtime = types.SimpleNamespace(_banks=banks, _global_bank=None)
    target = types.SimpleNamespace(
        model=types.SimpleNamespace(layers=[]),
        _mtplx_expert_runtime=runtime,
    )
    return target, runtime


def test_install_flat_on_fast_on_reports_and_binds(monkeypatch):
    _patch_gates(monkeypatch)  # NOTE: bind_yield_run is REAL here so the flat sha is real
    banks = {0: _real_transition_bank(), 3: _real_transition_bank()}
    target, _ = _armed_target(banks)
    report = f16_install.install(target, armed=True, flat_indices=True, fast_observe=True)
    assert report["flat_indices"] is True
    assert report["fast_observe"] is True
    assert report["fast_observe_banks"] == 2
    # the derived run's sha stays under its existing key and reflects the flat variant
    assert report["f16_yield_run_sha256"] == pl.build_yield_run(flat_indices=True)[1]["f16_yield_run_sha256"]
    assert report["f16_yield_run_sha256"] != pl.build_yield_run(flat_indices=False)[1]["f16_yield_run_sha256"]
    assert all("_observe_transition_window" in b.__dict__ for b in banks.values())


def test_install_flat_off_fast_off_reports_false_and_leaves_banks(monkeypatch):
    _patch_gates(monkeypatch)
    # flat off takes the stock reads bind call -> the F24-style one-arg stub is valid.
    monkeypatch.setattr(f16_install, "bind_yield_run",
                        lambda runners: {"retained_plane_lane_sha256": "stub"})
    banks = {0: _real_transition_bank(), 3: _real_transition_bank()}
    target, _ = _armed_target(banks)
    report = f16_install.install(target, armed=True)  # flat off, fast off
    assert report["fast_observe"] is False
    assert report["fast_observe_banks"] == 0
    assert all("_observe_transition_window" not in b.__dict__ for b in banks.values())


def test_install_barrier_flat_binds_the_flat_barrier_run(monkeypatch):
    _patch_gates(monkeypatch)  # bind_barrier_run is REAL
    banks = {0: _real_transition_bank()}
    target, _ = _armed_target(banks)
    report = f16_install.install(target, armed=True, handoff="barrier", flat_indices=True)
    assert report["flat_indices"] is True
    assert report["handoff"] == "barrier"
    assert report["f16_barrier_run_sha256"] == pl.build_barrier_run(flat_indices=True)[1]["f16_barrier_run_sha256"]
    assert report["f16_barrier_run_sha256"] != pl.build_barrier_run(flat_indices=False)[1]["f16_barrier_run_sha256"]
