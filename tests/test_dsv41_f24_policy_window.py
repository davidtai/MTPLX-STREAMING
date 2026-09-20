"""CPU tests for the F24 transition-window policy horizon (scripts/deepseek_v41/f16/install.py).

Pins MLX to CPU before importing ``f16.install`` (it pulls in ``f16.pipeline`` ->
greenlet + ``mlx.core``); ``mtplx.expert_streaming`` itself imports no MLX, so the real
``LayerExpertSlotBank`` runs on pure numpy.  No GPU/Metal, no artifact.  Covers the spec:

  * ``install_policy_window`` widens every per-layer bank's transition window from the
    shipped 16 to the requested value and reports the bank count; ``None`` leaves every
    bank untouched and reports ``{None, 0}``;
  * validate-first-then-apply: an out-of-contract window (15, 257, 3.5, True, "abc") or a
    mismatched bank (limit != 16, a non-transition-window policy, a non-deque window,
    ``counts`` None) raises and leaves NO bank modified (no partial application);
  * ``install()`` merges the fragment in the ARMED branch only (not armed -> the report is
    unchanged and every bank is untouched);
  * ``install_from_env`` reads ``MTPLX_DSV41_F24_POLICY_WINDOW`` at use (unset/empty ->
    None; ``int`` otherwise; a non-integer string raises ValueError naming the variable);
  * a REAL transition-window ``LayerExpertSlotBank`` ships at 16 and, once widened, keeps
    more than 16 routes -- exactly ``min(limit, n_plans)`` -- after N decode ``plan()`` calls.
"""
from __future__ import annotations

import collections
import os
import sys
import types
from pathlib import Path

import numpy as np  # noqa: F401  (kept for parity with the sibling harness; bank is pure numpy)
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)  # BEFORE importing f16.install -> f16.pipeline imports mlx.core

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts" / "deepseek_v41"
_PACKED = _REPO / "docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed"
# .f16-site holds greenlet (private dir, NOT the shared venv); honour F16SITE like the
# F16/F18/F20 tests (default = the f16-pipeline worktree's dir).  Needed before f16.pipeline.
_F16_SITE = os.environ.get(
    "F16SITE",
    "/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f16-pipeline/.f16-site",
)
for _p in (str(_SCRIPTS), str(_PACKED), str(_F16_SITE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mtplx.expert_streaming import (  # noqa: E402
    LayerExpertSlotBank,
    RoutingPhase,
    TRANSITION_WINDOW_CACHE_POLICY,
)

from f16 import install as f16_install  # noqa: E402  (package; scripts dir on path)

_POLICY_ENV = "MTPLX_DSV41_F24_POLICY_WINDOW"


# ---------------------------------------------------------------------------
# fakes: a minimal bank + runtime with exactly the four fields the validator reads
# ---------------------------------------------------------------------------
def _fake_bank(*, cache_policy=TRANSITION_WINDOW_CACHE_POLICY, limit=16,
               window="deque", counts="set"):
    """A pure-Python stand-in exposing exactly the fields ``install_policy_window``
    reads.  ``window="deque"`` -> a fresh deque; any other value is used verbatim (e.g.
    a list, to model a missing deque).  ``counts="set"`` -> a non-None sentinel; None
    models the missing-counts bank."""
    bank = types.SimpleNamespace()
    bank.cache_policy = cache_policy
    bank._transition_window_limit = limit
    bank._transition_window = collections.deque() if window == "deque" else window
    bank._transition_counts = object() if counts == "set" else counts
    return bank


def _fake_runtime(banks):
    return types.SimpleNamespace(_banks=banks)


def _good_banks(layers=(0, 3, 14)):
    return {layer: _fake_bank() for layer in layers}


# ---------------------------------------------------------------------------
# 1. install_policy_window: widen + count, None passthrough, empty-banks refuse
# ---------------------------------------------------------------------------
def test_widens_every_bank_and_reports_count():
    banks = _good_banks((0, 3, 14, 27))
    frag = f16_install.install_policy_window(_fake_runtime(banks), 32)
    assert frag == {"policy_window": 32, "policy_window_banks": 4}
    assert all(b._transition_window_limit == 32 for b in banks.values())


@pytest.mark.parametrize("window", [16, 24, 40, 256])
def test_widen_to_any_in_range(window):
    banks = _good_banks((1, 2))
    frag = f16_install.install_policy_window(_fake_runtime(banks), window)
    assert frag == {"policy_window": window, "policy_window_banks": 2}
    assert all(b._transition_window_limit == window for b in banks.values())


def test_none_leaves_banks_and_reports_none():
    banks = _good_banks()
    frag = f16_install.install_policy_window(_fake_runtime(banks), None)
    assert frag == {"policy_window": None, "policy_window_banks": 0}
    assert all(b._transition_window_limit == 16 for b in banks.values())


def test_empty_banks_raises():
    with pytest.raises(RuntimeError, match="_banks is empty"):
        f16_install.install_policy_window(_fake_runtime({}), 32)


# ---------------------------------------------------------------------------
# 2. validate-first-then-apply: out-of-contract window value -> ValueError, no mutation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [15, 257, 0, -1, 1000, 3.5, True, "abc", "3.5"])
def test_invalid_window_raises_valueerror_no_mutation(bad):
    banks = _good_banks((0, 3, 14))
    with pytest.raises(ValueError):
        f16_install.install_policy_window(_fake_runtime(banks), bad)
    assert all(b._transition_window_limit == 16 for b in banks.values()), (
        "a bank was modified on a rejected window value"
    )


# ---------------------------------------------------------------------------
# 3. validate-first-then-apply: a mismatched bank -> RuntimeError, NO bank modified
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_kwargs,match", [
    (dict(cache_policy="frequency"), "cache_policy"),
    (dict(cache_policy="lru"), "cache_policy"),
    (dict(cache_policy="transition-window-tuned"), "cache_policy"),  # the tuned variant is not the target
    (dict(limit=32), "_transition_window_limit"),
    (dict(limit=8), "_transition_window_limit"),
    (dict(window=[]), "not a collections.deque"),
    (dict(window=None), "not a collections.deque"),
    (dict(counts=None), "_transition_counts is None"),
])
def test_bad_bank_raises_runtimeerror_no_mutation(bad_kwargs, match):
    # One bad bank SANDWICHED between good banks; no bank may be modified (the apply
    # loop runs only after every bank validates), so both good banks keep limit 16.
    good_a = _fake_bank()
    bad = _fake_bank(**bad_kwargs)
    good_b = _fake_bank()
    banks = {0: good_a, 7: bad, 14: good_b}
    with pytest.raises(RuntimeError, match=match):
        f16_install.install_policy_window(_fake_runtime(banks), 32)
    assert good_a._transition_window_limit == 16
    assert good_b._transition_window_limit == 16
    if "limit" not in bad_kwargs:  # the bad bank's own limit (when it had a valid one) is untouched too
        assert bad._transition_window_limit == 16


# ---------------------------------------------------------------------------
# 4. install() integration: armed merges the fragment; not armed leaves the report
# ---------------------------------------------------------------------------
class _StubPipeline:
    """Stands in for the armed ``Pipeline`` so ``install(armed=True)`` runs without the
    real staged model.  Exposes exactly the attributes the install report reads."""

    def __init__(self, model, *, armed, split="fixed4", handoff="reads",
                 engram_lookahead=False):
        self._skip = 1
        self.engram_lookahead = False
        self.split = split


def _patch_armed_gates(monkeypatch):
    """Patch the F16 correctness gates (exercised by the F16 suite) to no-ops so the
    F24 policy-window step is the only thing under test in the armed ``install()``."""
    monkeypatch.setattr(f16_install, "Pipeline", _StubPipeline)
    monkeypatch.setattr(f16_install, "verify_source_pins", lambda: {"stub": "pin"})
    monkeypatch.setattr(f16_install, "_assert_scheduled_lane", lambda target: 3)
    monkeypatch.setattr(f16_install, "_collect_runners", lambda target: {})
    monkeypatch.setattr(f16_install, "bind_yield_run",
                        lambda runners: {"retained_plane_lane_sha256": "stub"})
    monkeypatch.setattr(f16_install, "_wrap_issue_next_for_trailer", lambda runners: 0)
    monkeypatch.setattr(f16_install, "pipeline_a_rows", lambda: 4)
    for env in ("MTPLX_DSV41_DEVICE_ROUTE", "MTPLX_DSV41_DEVICE_ROUTE_PINNED"):
        monkeypatch.delenv(env, raising=False)


def _armed_target(banks):
    runtime = types.SimpleNamespace(_banks=banks, _global_bank=None)
    return types.SimpleNamespace(
        model=types.SimpleNamespace(layers=[]),
        _mtplx_expert_runtime=runtime,
    ), runtime


def test_install_armed_applies_and_reports(monkeypatch):
    _patch_armed_gates(monkeypatch)
    target, runtime = _armed_target(_good_banks((0, 3, 14)))
    report = f16_install.install(target, armed=True, policy_window=32)
    assert report["installed"] is True and report["armed"] is True
    assert report["policy_window"] == 32
    assert report["policy_window_banks"] == 3
    assert all(b._transition_window_limit == 32 for b in runtime._banks.values())


def test_install_armed_none_reports_none_and_leaves_banks(monkeypatch):
    _patch_armed_gates(monkeypatch)
    target, runtime = _armed_target(_good_banks((0, 3, 14)))
    report = f16_install.install(target, armed=True, policy_window=None)
    assert report["policy_window"] is None
    assert report["policy_window_banks"] == 0
    assert all(b._transition_window_limit == 16 for b in runtime._banks.values())


def test_install_not_armed_ignores_policy_window():
    banks = _good_banks((0, 3))
    target = types.SimpleNamespace(
        model=types.SimpleNamespace(),
        _mtplx_expert_runtime=types.SimpleNamespace(_banks=banks),
    )
    report = f16_install.install(target, armed=False, policy_window=32)
    assert report == {"installed": False, "armed": False, "reason": "MTPLX_DSV41_F16 != 1"}
    assert "policy_window" not in report
    assert all(b._transition_window_limit == 16 for b in banks.values())


# ---------------------------------------------------------------------------
# 5. install_from_env: read MTPLX_DSV41_F24_POLICY_WINDOW at use
# ---------------------------------------------------------------------------
def _clear_f16_env(monkeypatch):
    monkeypatch.delenv("MTPLX_DSV41_F16", raising=False)
    monkeypatch.delenv("MTPLX_DSV41_F16_COUNTERS", raising=False)


def _capture_install(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        f16_install, "install",
        lambda target, **kwargs: captured.update(kwargs) or {"installed": False, "armed": False},
    )
    return captured


def test_env_unset_passes_none(monkeypatch):
    _clear_f16_env(monkeypatch)
    monkeypatch.delenv(_POLICY_ENV, raising=False)
    captured = _capture_install(monkeypatch)
    f16_install.install_from_env(types.SimpleNamespace())
    assert captured["policy_window"] is None


def test_env_empty_passes_none(monkeypatch):
    _clear_f16_env(monkeypatch)
    monkeypatch.setenv(_POLICY_ENV, "")
    captured = _capture_install(monkeypatch)
    f16_install.install_from_env(types.SimpleNamespace())
    assert captured["policy_window"] is None


@pytest.mark.parametrize("raw,expected", [("32", 32), ("16", 16), ("256", 256), ("5", 5)])
def test_env_int_passes_through(monkeypatch, raw, expected):
    # The env read only converts; the [16, 256] range check lives in the armed helper
    # ("5" would be refused there), so a plain int -- in range or not -- passes through.
    _clear_f16_env(monkeypatch)
    monkeypatch.setenv(_POLICY_ENV, raw)
    captured = _capture_install(monkeypatch)
    f16_install.install_from_env(types.SimpleNamespace())
    assert captured["policy_window"] == expected


@pytest.mark.parametrize("raw", ["abc", "3.5", "32x", "1e3", "  "])
def test_env_noninteger_raises_with_var_name(monkeypatch, raw):
    _clear_f16_env(monkeypatch)
    monkeypatch.setenv(_POLICY_ENV, raw)
    with pytest.raises(ValueError, match=_POLICY_ENV):
        f16_install.install_from_env(types.SimpleNamespace())


# ---------------------------------------------------------------------------
# 6. a REAL transition-window LayerExpertSlotBank (CPU only, pure numpy)
# ---------------------------------------------------------------------------
def _real_transition_bank():
    # transition-window requires single_pool=True and exactly 384 experts (its dense
    # transition table is bounded to 384x384); transient_slots covers the route width.
    return LayerExpertSlotBank(
        expert_count=384,
        persistent_slots=8,
        transient_slots=8,
        cache_policy=TRANSITION_WINDOW_CACHE_POLICY,
        single_pool=True,
        layer_id=0,
    )


def test_real_bank_ships_at_16_transition_window():
    """Grounds the validator's constants against the real bank: the shipped
    transition-window bank has cache_policy 'transition-window', limit 16, a deque
    window and non-None counts -- exactly what install_policy_window requires."""
    bank = _real_transition_bank()
    assert bank.cache_policy == TRANSITION_WINDOW_CACHE_POLICY
    assert bank._transition_window_limit == 16
    assert isinstance(bank._transition_window, collections.deque)
    assert bank._transition_counts is not None


@pytest.mark.parametrize("limit,n_plans,expected", [
    (32, 40, 32),   # the spec case: widen to 32, 40 decode plans -> exactly 32 (>16)
    (40, 40, 40),   # limit exactly fills over the run
    (48, 40, 40),   # fewer plans than the limit -> window = n_plans, still well past 16
])
def test_real_bank_widened_keeps_more_than_16(limit, n_plans, expected):
    bank = _real_transition_bank()
    frag = f16_install.install_policy_window(_fake_runtime({0: bank}), limit)
    assert frag == {"policy_window": limit, "policy_window_banks": 1}
    assert bank._transition_window_limit == limit
    for i in range(n_plans):
        route = [(2 * i) % 384, (2 * i + 1) % 384]  # 2 distinct valid ids <= transient_slots
        bank.plan(route, phase=RoutingPhase.DECODE)
    assert len(bank._transition_window) > 16, "widened window did not retain more than 16 routes"
    assert len(bank._transition_window) == expected, "window did not settle at min(limit, n_plans)"
