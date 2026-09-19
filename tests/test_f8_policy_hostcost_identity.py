"""F8 identity proof: the host-cost-optimised transition-window policy in
``mtplx.expert_streaming`` is DECISION-IDENTICAL to the pre-optimisation code.

The pre-optimisation module is preserved verbatim at
``tests/_f8_expert_streaming_oracle.py`` (sha256 asserted below) and imported as
the reference ORACLE.  Every test drives the oracle and the live package in
lockstep and asserts identical RoutePlan output (hits, admitted vs rejected
misses, per-expert slot assignment, victim-slot evictions), identical bank state
(slot map, residency, recency, protected set, pins, seed set, history), and --
for the transition-window policy -- bit-exact float32 scores.  It also proves
the published miss/read anchors still reproduce exactly, and that the numpy
mirrors (``_last_used_arr`` / ``_prefill_freq_arr``) stay consistent with the
Python objects through plan, transaction rollback, prefill-seed update and reset.

CPU-ONLY: MLX is hard-blocked by a meta-path finder (the policy layer is pure
Python/numpy).  Run this module in isolation, e.g.

    nice -n 19 .venv/bin/python -m pytest tests/test_f8_policy_hostcost_identity.py -q
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.abc
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest


class _NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise RuntimeError("F8 identity tests are CPU-only; MLX is forbidden")


sys.meta_path.insert(0, _NoMLX())

import numpy as np  # noqa: E402  (after the guard)

ROOT = Path(__file__).resolve().parents[1]
ORACLE_PATH = ROOT / "tests" / "_f8_expert_streaming_oracle.py"
ORACLE_SHA256 = "de1771923bdaaf275f0ea834dbb1407336c288724b088a811a61e235f0571c39"
TRACE = (
    ROOT / "docs" / "deepseek-v41" / "receipts" / "mtp-verify-routes-20260913"
    / "mtp-verify-routes-16k-1024-v2.json.gz"
)
TRACE_SHA256 = "07b4b720bae831421bbbab6d4cb770ade577096c1e85733ad949350edf66d0da"

N_LAYERS = 40


def _mlx_imported() -> bool:
    return any(m == "mlx" or m.startswith("mlx.") for m in sys.modules)


def _load_oracle():
    blob = ORACLE_PATH.read_bytes()
    got = hashlib.sha256(blob).hexdigest()
    assert got == ORACLE_SHA256, f"oracle drifted: {got}"
    spec = importlib.util.spec_from_file_location("_f8_expert_streaming_oracle", str(ORACLE_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register so @dataclass can resolve __module__
    spec.loader.exec_module(mod)
    return mod


import mtplx.expert_streaming as NEW  # noqa: E402
OLD = _load_oracle()


def _restore(mod, state, *, policy=None, decay=None, single_pool=None):
    """Restore a bank from saved warm state (verbatim from the receipt helper
    docs/deepseek-v41/receipts/memory-budget-110/replay_route_policies.py, but
    parameterised by module so old and new restore identically)."""

    kwargs = {
        k: state[k]
        for k in ("expert_count", "persistent_slots", "transient_slots",
                  "frequency_decay", "cache_policy", "single_pool")
    }
    if policy is not None:
        kwargs["cache_policy"] = policy
    if decay is not None:
        kwargs["frequency_decay"] = decay
    if single_pool is not None:
        kwargs["single_pool"] = single_pool
    bank = mod.LayerExpertSlotBank(**kwargs)
    for key in ("_slot_to_expert", "_pool_clock", "_decode_epoch", "_saw_decode_since_prefill"):
        setattr(bank, key, state[key])
    bank._slot_to_expert = list(bank._slot_to_expert)
    for key in ("_expert_to_slot", "_pool_recency"):
        setattr(bank, key, {int(k): v for k, v in state[key].items()})
    for key in ("_protected", "_prefill_seed_candidates"):
        setattr(bank, key, set(state[key]))
    bank._prefill_route_freq = Counter({int(k): v for k, v in state["_prefill_route_freq"].items()})
    bank._history = [mod._ExpertHistory(**h) for h in state["_history"]]
    return bank


def _make_bank(mod, state, persistent, transient, policy):
    bank = _restore(mod, state, policy=policy, single_pool=True)
    extra = persistent - bank.persistent_slots
    if extra > 0:
        bank._slot_to_expert.extend([None] * extra)
    bank.persistent_slots = bank._persistent_capacity = persistent
    bank.slot_count = persistent + transient
    bank.transient_slots = transient
    if hasattr(bank, "_protected_cap"):
        bank._protected_cap = max(1, int(persistent * 0.8))
    return bank


def _plan_sig(plan):
    """Every decision-bearing RoutePlan field, class-agnostic."""

    return (
        plan.phase.value,
        tuple(plan.experts),
        tuple(plan.slots),
        tuple(plan.hits),
        tuple(plan.misses),
        tuple((l.expert, l.slot, l.persistent, l.generation) for l in plan.loads),
        tuple((e.slot, e.previous_expert, e.next_expert) for e in plan.evictions),
        tuple(plan.generations),
        plan.pool_loads,
        plan.scan_inserts,
        plan.promotions,
        tuple(plan.prefetch_hits),
        tuple(plan.prefetch_first_hits),
    )


_ARR_NAMES = ("_transition_counts", "_transition_denominators", "_transition_window_frequency")


def _state_dict(bank):
    """Non-array public + policy state (exact-comparable via ==).  Excludes the
    new-only numpy mirrors, which the oracle lacks (checked separately)."""

    tw = getattr(bank, "_transition_window", None)
    return {
        "slot_to_expert": list(bank._slot_to_expert),
        "expert_to_slot": dict(bank._expert_to_slot),
        "protected": sorted(bank._protected),
        "pool_recency": dict(bank._pool_recency),
        "pool_clock": bank._pool_clock,
        "decode_epoch": bank._decode_epoch,
        "pinned": sorted(bank._pinned),
        "seed_candidates": sorted(bank._prefill_seed_candidates),
        "saw_decode": bank._saw_decode_since_prefill,
        "prefill_route_freq": dict(bank._prefill_route_freq),
        "last_used": [h.last_used for h in bank._history],
        "score": [h.score for h in bank._history],
        "score_epoch": [h.score_epoch for h in bank._history],
        "transition_previous": bank._transition_previous,
        "transition_window": list(tw) if tw is not None else None,
    }


def _light_dict(bank):
    """Cheap per-call decision state for the 8240-call full-trace loop."""

    tw = getattr(bank, "_transition_window", None)
    return {
        "slot_to_expert": list(bank._slot_to_expert),
        "expert_to_slot": dict(bank._expert_to_slot),
        "protected": sorted(bank._protected),
        "pool_recency": dict(bank._pool_recency),
        "pool_clock": bank._pool_clock,
        "decode_epoch": bank._decode_epoch,
        "transition_previous": bank._transition_previous,
        "transition_window": list(tw) if tw is not None else None,
    }


def _arrays_equal(old, new):
    for name in _ARR_NAMES:
        ao, an = getattr(old, name), getattr(new, name)
        if ao is None or an is None:
            assert (ao is None) == (an is None), name
        else:
            assert np.array_equal(ao, an), name


def _assert_mirror_consistent(bank, where):
    """The new module's numpy mirrors must equal the Python objects exactly."""

    exp_last = np.fromiter((h.last_used for h in bank._history), dtype=np.int64,
                           count=bank.expert_count)
    assert np.array_equal(bank._last_used_arr, exp_last), f"_last_used_arr drift @ {where}"
    exp_freq = np.zeros(bank.expert_count, dtype=np.int64)
    for e, c in bank._prefill_route_freq.items():
        exp_freq[int(e)] = c
    assert np.array_equal(bank._prefill_freq_arr, exp_freq), f"_prefill_freq_arr drift @ {where}"


def _full_check(old_bank, new_bank, where):
    assert _state_dict(old_bank) == _state_dict(new_bank), f"state diverged @ {where}"
    _arrays_equal(old_bank, new_bank)
    _assert_mirror_consistent(new_bank, where)


def _snapshot(bank):
    arrs = tuple(None if getattr(bank, n) is None else getattr(bank, n).copy()
                 for n in _ARR_NAMES)
    return (_state_dict(bank), arrs)


def _assert_snapshot(bank, snap, where):
    state, arrs = snap
    assert _state_dict(bank) == state, f"rollback not exact @ {where}"
    for name, a in zip(_ARR_NAMES, arrs):
        cur = getattr(bank, name)
        if a is None:
            assert cur is None, name
        else:
            assert np.array_equal(cur, a), f"{name} rollback @ {where}"


# ---------------------------------------------------------------------------
# 1. Full 206x40 trace, step-by-step identity at 111+48 transition-window.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trace():
    blob = TRACE.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == TRACE_SHA256
    t = json.loads(gzip.decompress(blob))
    assert t["complete"] and t["cycles"] == 206
    return t


def test_full_trace_step_identity(trace):
    """Replay all 206 cycles x 40 layers through OLD and NEW at 111+48
    transition-window; assert per-layer-call RoutePlan identity, bank-state
    identity, and bit-exact float32 scores; totals must match the anchor."""

    assert not _mlx_imported()
    routes = {int(l): trace["target_routes_by_layer"][l] for l in trace["target_routes_by_layer"]}
    old_banks = {L: _make_bank(OLD, trace["initial_banks"][str(L)], 111, 48, "transition-window")
                 for L in range(N_LAYERS)}
    new_banks = {L: _make_bank(NEW, trace["initial_banks"][str(L)], 111, 48, "transition-window")
                 for L in range(N_LAYERS)}
    cyc = trace["cycles"]
    total_old = total_new = 0
    calls = 0
    for c in range(cyc):
        for L in range(N_LAYERS):
            route = routes[L][c]
            po = old_banks[L].plan(route, phase="decode")
            pn = new_banks[L].plan(route, phase="decode")
            assert _plan_sig(po) == _plan_sig(pn), f"plan diverged @ cycle {c} layer {L}"
            # bit-exact float32 scores (transition_previous is set post-decode)
            so = old_banks[L]._transition_window_scores()
            sn = new_banks[L]._transition_window_scores()
            assert so.dtype == sn.dtype == np.float32
            assert np.array_equal(so, sn), f"scores not bit-exact @ cycle {c} layer {L}"
            assert _light_dict(old_banks[L]) == _light_dict(new_banks[L]), \
                f"state diverged @ cycle {c} layer {L}"
            # Full check (history, counts arrays, mirrors) periodically -- cheap
            # to skip per-call since scores bit-exactness already exercises the
            # score inputs and _observe_transition_window is unchanged code.
            if calls % 400 == 0:
                _full_check(old_banks[L], new_banks[L], f"cycle {c} layer {L}")
            total_old += len(po.misses)
            total_new += len(pn.misses)
            calls += 1
    # Full state + array + mirror check on every layer bank at the end.
    for L in range(N_LAYERS):
        _full_check(old_banks[L], new_banks[L], f"final layer {L}")
    assert total_old == total_new == 31636
    assert not _mlx_imported()


# ---------------------------------------------------------------------------
# 2. Published anchors reproduce exactly on OLD and NEW.
# ---------------------------------------------------------------------------
def test_published_anchors_old_and_new(trace):
    for mod, tag in ((OLD, "old"), (NEW, "new")):
        misses = routes = 0
        for layer, seq in trace["target_routes_by_layer"].items():
            bank = _make_bank(mod, trace["initial_banks"][layer], 102, 48, "transition-window")
            for route in seq:
                misses += len(set(bank.plan(route, phase="decode").misses))
                routes += 1
        assert (misses, routes) == (35164, 8240), f"cap102 anchor failed ({tag})"

        records = 0
        for layer, seq in trace["target_routes_by_layer"].items():
            bank = _make_bank(mod, trace["initial_banks"][layer], 73, 48, "frequency")
            for route in seq:
                plan = bank.try_plan_all_hits(route, phase="decode")
                if plan is None:
                    plan = bank.plan(route, phase="decode")
                records += len(plan.misses)
        assert records == 53999, f"cap73 anchor failed ({tag})"
    assert not _mlx_imported()


# ---------------------------------------------------------------------------
# 3. lexsort tie-break equivalence: force each rank level incl. -expert and
#    negative last_used, on a hand-built bank; NEW admissions == OLD admissions.
# ---------------------------------------------------------------------------
def _fresh_pair(persistent, transient, policy, expert_count=384, single_pool=True):
    ko = OLD.LayerExpertSlotBank(expert_count=expert_count, persistent_slots=persistent,
                                 transient_slots=transient, cache_policy=policy,
                                 single_pool=single_pool)
    kn = NEW.LayerExpertSlotBank(expert_count=expert_count, persistent_slots=persistent,
                                 transient_slots=transient, cache_policy=policy,
                                 single_pool=single_pool)
    return ko, kn


def test_lexsort_tie_break_equivalence():
    """Craft decode histories so candidate experts collide at each rank level:
    equal scores (forces last_used), equal (score,last_used) (forces freq), and
    equal (score,last_used,freq) (forces -expert); include never-used experts
    (last_used == -1, i.e. negative) among residents and misses.  The admitted
    set, victim ordering and slot map must match OLD exactly."""

    rng = random.Random(20260919)
    for trial in range(200):
        persistent = rng.randint(8, 40)
        transient = 48
        ko, kn = _fresh_pair(persistent, transient, "transition-window")
        # Warm both identically with a handful of decode routes so scores,
        # window frequency and last_used are populated (and some collide).
        pool = list(range(0, 60))
        for _ in range(rng.randint(1, 20)):
            width = rng.randint(2, min(transient, len(pool)))
            route = rng.sample(pool, width)
            # inject duplicates sometimes (unique count stays <= transient)
            if rng.random() < 0.3:
                route = route + rng.sample(route, rng.randint(1, len(route)))
                rng.shuffle(route)
            po = ko.plan(route, phase="decode")
            pn = kn.plan(route, phase="decode")
            assert _plan_sig(po) == _plan_sig(pn), f"warmup plan diverged trial {trial}"
        _full_check(ko, kn, f"warmup trial {trial}")
        # Now a final decode route mixing residents (some tied) and fresh misses.
        residents = [e for e in ko._slot_to_expert if e is not None]
        fresh = [e for e in range(60, 90)]
        width = rng.randint(2, min(transient, len(residents) + len(fresh)))
        chosen = rng.sample(residents, min(len(residents), rng.randint(0, width))) if residents else []
        chosen += rng.sample(fresh, min(len(fresh), width - len(chosen)))
        if not chosen:
            chosen = rng.sample(fresh, 2)
        rng.shuffle(chosen)
        po = ko.plan(chosen, phase="decode")
        pn = kn.plan(chosen, phase="decode")
        assert _plan_sig(po) == _plan_sig(pn), f"tie-break plan diverged trial {trial}"
        so = ko._transition_window_scores()
        sn = kn._transition_window_scores()
        assert np.array_equal(so, sn), f"tie-break scores diverged trial {trial}"
        _full_check(ko, kn, f"tie-break trial {trial}")
    assert not _mlx_imported()


# ---------------------------------------------------------------------------
# 4. Randomised lockstep property test across every policy: routes, capacities
#    8..128, pins, empty slots, transaction rollback, all-hit probes, reset.
# ---------------------------------------------------------------------------
POLICIES = ["frequency", "lru", "transition-window", "transition-window-tuned"]


def _rand_route(rng, expert_count, transient):
    width = rng.randint(1, min(transient, expert_count))
    route = rng.sample(range(expert_count), width)
    if rng.random() < 0.25 and width > 1:  # duplicates, unique count unchanged
        route = route + rng.sample(route, rng.randint(1, width))
        rng.shuffle(route)
    return route


@pytest.mark.parametrize("policy", POLICIES)
def test_property_random_lockstep(policy):
    """Fuzz OLD vs NEW through a random op sequence; compare after every op."""

    assert not _mlx_imported()
    rng = random.Random(hash(policy) & 0xFFFF)
    for trial in range(60):
        if policy in ("transition-window", "transition-window-tuned"):
            expert_count = 384
            single_pool = True
        else:
            expert_count = rng.randint(16, 128)
            single_pool = rng.random() < 0.5
        persistent = rng.randint(8, min(128, expert_count))
        transient = rng.randint(8, 32)
        ko = OLD.LayerExpertSlotBank(expert_count=expert_count, persistent_slots=persistent,
                                     transient_slots=transient, cache_policy=policy,
                                     single_pool=single_pool)
        kn = NEW.LayerExpertSlotBank(expert_count=expert_count, persistent_slots=persistent,
                                     transient_slots=transient, cache_policy=policy,
                                     single_pool=single_pool)
        _full_check(ko, kn, f"{policy} init trial {trial}")

        for step in range(rng.randint(10, 40)):
            op = rng.choice([
                "decode", "decode", "decode", "prefill", "seed",
                "txn_commit", "txn_rollback", "allhits_rollback",
                "pin", "invalidate", "reset",
            ])
            where = f"{policy} trial {trial} step {step} op {op}"
            if op == "decode":
                r = _rand_route(rng, expert_count, transient)
                po = ko.plan(r, phase="decode")
                pn = kn.plan(r, phase="decode")
                assert _plan_sig(po) == _plan_sig(pn), where
            elif op == "prefill":
                r = _rand_route(rng, expert_count, transient)
                po = ko.plan(r, phase="prefill")
                pn = kn.plan(r, phase="prefill")
                assert _plan_sig(po) == _plan_sig(pn), where
            elif op == "seed":
                r = _rand_route(rng, expert_count, transient)
                assert tuple(ko.prepare_prefill_seed(r)) == tuple(kn.prepare_prefill_seed(r)), where
            elif op == "txn_commit":
                r = _rand_route(rng, expert_count, transient)
                po, _to = ko.plan_transaction(r, phase="decode")
                pn, _tn = kn.plan_transaction(r, phase="decode")
                assert _plan_sig(po) == _plan_sig(pn), where
            elif op == "txn_rollback":
                before = _snapshot(ko)
                r = _rand_route(rng, expert_count, transient)
                po, to = ko.plan_transaction(r, phase="decode")
                pn, tn = kn.plan_transaction(r, phase="decode")
                assert _plan_sig(po) == _plan_sig(pn), where
                to.rollback_completion()
                tn.rollback_completion()
                _assert_snapshot(ko, before, where)
            elif op == "allhits_rollback":
                r = _rand_route(rng, expert_count, transient)
                ro = ko.try_plan_all_hits_transaction(r, phase="decode")
                rn = kn.try_plan_all_hits_transaction(r, phase="decode")
                assert (ro is None) == (rn is None), where
                if ro is not None:
                    assert _plan_sig(ro[0]) == _plan_sig(rn[0]), where
                    if rng.random() < 0.5:
                        ro[1].rollback_completion()
                        rn[1].rollback_completion()
            elif op == "pin":
                resident = [e for e in ko._slot_to_expert if e is not None]
                if resident and rng.random() < 0.5:
                    subset = rng.sample(resident, rng.randint(1, len(resident)))
                    assert tuple(ko.pin_working_set(experts=subset)) == \
                           tuple(kn.pin_working_set(experts=subset)), where
                else:
                    tk = rng.randint(0, persistent)
                    assert tuple(ko.pin_working_set(top_k=tk)) == \
                           tuple(kn.pin_working_set(top_k=tk)), where
            elif op == "invalidate":
                resident = [e for e in ko._slot_to_expert if e is not None]
                if resident:
                    e = rng.choice(resident)
                    assert ko.invalidate_expert(e) == kn.invalidate_expert(e), where
            elif op == "reset":
                ko.reset()
                kn.reset()
            _full_check(ko, kn, where)
    assert not _mlx_imported()


# ---------------------------------------------------------------------------
# 5. Focused reset + rollback mirror-consistency on the transition path.
# ---------------------------------------------------------------------------
def test_reset_and_rollback_mirror_consistency():
    rng = random.Random(4242)
    ko, kn = _fresh_pair(64, 48, "transition-window")
    for _ in range(30):
        r = _rand_route(rng, 384, 48)  # one route fed to both banks
        ko.plan(r, phase="decode")
        kn.plan(r, phase="decode")
    _full_check(ko, kn, "pre-reset")
    # rollback of a decode transaction restores mirrors exactly
    before = _snapshot(kn)
    r = _rand_route(rng, 384, 48)
    _po, to = ko.plan_transaction(r, phase="decode")
    _pn, tn = kn.plan_transaction(r, phase="decode")
    to.rollback_completion()
    tn.rollback_completion()
    _assert_snapshot(kn, before, "reset-test rollback")
    _assert_mirror_consistent(kn, "post-rollback")
    # reset zeroes both mirrors
    kn.reset()
    _assert_mirror_consistent(kn, "post-reset")
    assert np.array_equal(kn._last_used_arr, np.full(384, -1, dtype=np.int64))
    assert not kn._prefill_freq_arr.any()
    assert not _mlx_imported()
