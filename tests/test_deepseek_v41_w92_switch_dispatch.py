"""W92 -- AR-decode streamed-switch dispatch/sync census + the switch-lean arm.

Window 33 (arm ``cell16k_ring``, receipt
docs/deepseek-v41/receipts/gpu-windows/window-33/ar-16k-cell16k-ring.json) ran the
M=1 AR-decode switch through the SHIPPED synchronous wave fence: per all-hit layer
the route probe attributes ``hot.allhit_fence_eval`` == ``hot.all_hit`` -- a SECOND
blocking ``mx.eval(wave_output)`` on top of the one ``mx.eval(indices)`` routing
barrier -- and on a miss layer a blocking fence per split-route wave part.  The
gather itself is already minimal: ``_gather_component_bank`` issues ONE grouped
``mx.gather_qmm`` per component (gate/up/down) over the routed slots, never one per
expert (proven by ``hot.switch_gather_qmm`` == 3 * ``hot.all_hit`` on an all-hit
layer).  ``cell16k_ring`` never armed the deferral (``MTPLX_DSV41_SWITCH_FASTPATH`` /
``_SUBMIT`` unset in the receipt), so the AR lane paid the extra per-layer sync.

The ``switch_lean`` arm arms the K23 variant-B fast-path (defer the release to the
next routing barrier + async-submit the gather so the GPU is fed without the
blocking round-trip -- pure defer without submit lost -13% at 1K, W42 window-14).
Net: exactly ONE small eval (indices only) per streamed layer, gathers issued
without further blocking syncs.  It is a pure fence/release-timing reorder, so the
switch output is byte-identical, and it is pin-safe: ``try_all_hit_route`` pins the
whole route and ``defer_slot_release`` holds those pins until the covering flush, so
an eviction attempt on a deferred route's slot is refused (``ExpertSlotError``) and
redirected -- no admission can recycle a slot whose gather is still pending.

CPU-pinned (mx.set_default_device(mx.cpu)); no GPU; fake component-bank runtime
(``_integrated_hy3_artifact``); peak RSS well under the guard.  Run under
``nice -n 19`` and without ``pytest -n auto``.
"""

from __future__ import annotations

import os

import mlx.core as mx
import pytest

pytest.importorskip("mlx.core")

mx.set_default_device(mx.cpu)

import mtplx.models.expert_mlx as expert_mlx  # noqa: E402
import mtplx.expert_route_probe as route_probe  # noqa: E402
from mtplx.expert_manifest import load_expert_manifest  # noqa: E402
from mtplx.expert_runtime import ExpertStreamingConfig, ExpertStreamingRuntime  # noqa: E402
from mtplx.expert_slots import ExpertSlotError  # noqa: E402
from mtplx.models.expert_mlx import (  # noqa: E402
    HotExpertSwitchGLU,
    make_mlx_component_bank_allocator,
)

FASTPATH = "MTPLX_DSV41_SWITCH_FASTPATH"
SUBMIT = "MTPLX_DSV41_SWITCH_SUBMIT"
VERIFY = "MTPLX_DSV41_VERIFY_SINGLE_BARRIER"
_SWITCH_ENVS = (FASTPATH, SUBMIT, VERIFY)
_REAL_EVAL = mx.eval

_TOP_K = 6
_EXPERT_COUNT = 24


@pytest.fixture(autouse=True)
def _cpu_and_clean_env():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    saved = {k: os.environ.get(k) for k in _SWITCH_ENVS}
    for k in _SWITCH_ENVS:
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


def _open_runtime(tmp_path, *, resident_slots, transient=_TOP_K, kv_tokens=0):
    from tests.test_streamed_models import _integrated_hy3_artifact

    root, _config, spec, manifest_path = _integrated_hy3_artifact(
        tmp_path, expert_count=_EXPERT_COUNT, top_k=_TOP_K
    )
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    kv_bytes = int(getattr(spec, "kv_bytes_per_token", 0) or 0) * kv_tokens
    sc = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=(
            fixed + spec.persistent_cache_bytes(resident_slots) + kv_bytes
        ),
        # No explicit cache cap: fixed slot storage is sized after maximum KV.
        max_live_kv_tokens=kv_tokens,
        runtime_reserve_bytes=0,
        transient_slots=transient,
        slot_layout="component-banks",
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


def _arm(lean: bool) -> None:
    if lean:
        os.environ[FASTPATH] = "1"
        os.environ[SUBMIT] = "1"
    else:
        os.environ.pop(FASTPATH, None)
        os.environ.pop(SUBMIT, None)


def _m1(spec, experts):
    idx = mx.array([list(experts[: spec.top_k])], dtype=mx.int32)  # 1 row, top_k
    mx.random.seed(920 + int(experts[0]))
    x = (0.3 * mx.random.normal((1, 1, spec.hidden_size))).astype(mx.bfloat16)
    _REAL_EVAL(x, idx)
    return x, idx


def _warm(rt, spec, experts):
    sw = HotExpertSwitchGLU(rt, spec.routed_layer_start)
    for e in experts:
        idx = mx.array([[e] * spec.top_k], dtype=mx.int32)
        x = mx.zeros((1, 1, spec.hidden_size), dtype=mx.bfloat16)
        _REAL_EVAL(sw(x, idx))
    rt.flush_deferred_slot_releases(evaluate=True)


# ---------------------------------------------------------------------------
# 1. byte-identity: AR M=1, 64 steps, hit/miss mix, control vs switch_lean
# ---------------------------------------------------------------------------
def _ar_sequence(rt, spec, *, lean, steps=64):
    """Run ``steps`` M=1 decode routes over a rotating expert window (so the LRU
    evicts and re-misses -> a hit/miss mix) and return the stacked outputs."""
    _arm(lean)
    sw = HotExpertSwitchGLU(rt, spec.routed_layer_start)
    outs = []
    for step in range(steps):
        base = step % (_EXPERT_COUNT - _TOP_K)  # window slides across all experts
        experts = [base + j for j in range(_TOP_K)]
        x, idx = _m1(spec, experts)
        out = sw(x, idx)
        _REAL_EVAL(out)
        rt.flush_deferred_slot_releases(evaluate=True)
        outs.append(out)
    return outs


def test_ar_64_step_hit_miss_mix_byte_identical(tmp_path) -> None:
    # resident_slots < expert_count so the rotating window forces real misses,
    # giving a hit/miss mix over the 64 steps (not all-hit).
    dir_off = tmp_path / "off"
    dir_on = tmp_path / "on"
    dir_off.mkdir()
    dir_on.mkdir()

    rt_off, spec = _open_runtime(dir_off, resident_slots=10)
    try:
        outs_off = _ar_sequence(rt_off, spec, lean=False)
    finally:
        rt_off.close()

    rt_on, spec2 = _open_runtime(dir_on, resident_slots=10)
    try:
        outs_on = _ar_sequence(rt_on, spec2, lean=True)
    finally:
        rt_on.close()

    assert len(outs_off) == len(outs_on) == 64
    for i, (a, b) in enumerate(zip(outs_off, outs_on)):
        assert a.shape == b.shape == (1, _TOP_K, spec.hidden_size)
        assert mx.array_equal(a, b), f"AR step {i}: switch_lean != control"


# ---------------------------------------------------------------------------
# 2. dispatch + sync census, before (control) vs after (switch_lean), per path
# ---------------------------------------------------------------------------
def _census_one_switch(rt, spec, x, idx, *, lean):
    """Return (blocking mx.eval count, {probe stage: delta}) for ONE switch call."""
    _arm(lean)
    saved_enabled = route_probe.ENABLED
    real = expert_mlx.mx.eval
    n = {"eval": 0}

    def counting_eval(*a, **k):
        n["eval"] += 1
        return real(*a, **k)

    route_probe.ENABLED = True
    expert_mlx.mx.eval = counting_eval
    try:
        before = {k: v["count"] for k, v in route_probe.snapshot()["stages"].items()}
        out = HotExpertSwitchGLU(rt, spec.routed_layer_start)(x, idx)
        stages = {}
        for k, v in route_probe.snapshot()["stages"].items():
            d = v["count"] - before.get(k, 0)
            if d:
                stages[k] = d
    finally:
        expert_mlx.mx.eval = real
        route_probe.ENABLED = saved_enabled
    real(out)
    rt.flush_deferred_slot_releases(evaluate=True)
    return n["eval"], stages


def test_all_hit_switch_is_one_gather_qmm_per_component(tmp_path) -> None:
    # all 6 routed experts resident -> all-hit.  Prove the all-hit gather is 3
    # dispatches (gate/up/down grouped over the routed slots), never per-expert.
    rt, spec = _open_runtime(tmp_path, resident_slots=_TOP_K)
    try:
        _warm(rt, spec, list(range(_TOP_K)))
        x, idx = _m1(spec, list(range(_TOP_K)))
        _evals, stages = _census_one_switch(rt, spec, x, idx, lean=False)
        assert stages.get("hot.all_hit") == 1, stages
        # hot.allhit_gather_qmm is scoped to the all-hit branch only (a delta on
        # the shared counter), so the per-all-hit-call ratio is exactly 3.
        assert stages.get("hot.allhit_gather_qmm") == 3, (
            f"all-hit switch must be 3 gather_qmm (gate/up/down), saw {stages}"
        )
        # the whole-pass counter agrees on a pure all-hit call.
        assert stages.get("hot.switch_gather_qmm") == 3, stages
    finally:
        rt.close()


def test_split_layer_does_not_inflate_allhit_gather_qmm(tmp_path) -> None:
    # On a MISS (split) layer the all-hit branch is not taken, so the whole-pass
    # counter increments (hits group + miss group = 6) but the all-hit-scoped
    # counter must stay 0 -- otherwise gather_qmm_per_all_hit_call would read the
    # inflated 15-30 the reviewer saw on the 16K cell instead of 3.
    rt, spec = _open_runtime(tmp_path, resident_slots=_TOP_K, transient=_TOP_K)
    try:
        _warm(rt, spec, [0, 1, 2])  # 0,1,2 resident; 6,7,8 miss
        x, idx = _m1(spec, [0, 1, 2, 6, 7, 8])
        _evals, stages = _census_one_switch(rt, spec, x, idx, lean=False)
        assert stages.get("hot.split_route") == 1, stages
        assert stages.get("hot.all_hit", 0) == 0, stages
        assert stages.get("hot.allhit_gather_qmm", 0) == 0, (
            f"split layer must not increment the all-hit-scoped counter, saw {stages}"
        )
        assert stages.get("hot.switch_gather_qmm", 0) > 0, stages  # total still counts
    finally:
        rt.close()


def test_all_hit_sync_count_before_vs_after(tmp_path) -> None:
    # shipped (control): 2 blocking evals per all-hit layer (indices + wave fence),
    # and the fence is SYNCED.  switch_lean: 1 eval (indices only), fence DEFERRED.
    rt, spec = _open_runtime(tmp_path, resident_slots=_TOP_K)
    try:
        _warm(rt, spec, list(range(_TOP_K)))
        x, idx = _m1(spec, list(range(_TOP_K)))

        off_evals, off_stages = _census_one_switch(rt, spec, x, idx, lean=False)
        on_evals, on_stages = _census_one_switch(rt, spec, x, idx, lean=True)

        assert off_evals == 2, f"control all-hit expected 2 blocking evals, got {off_evals}"
        assert on_evals == 1, f"switch_lean all-hit expected 1 blocking eval, got {on_evals}"
        # the removed sync is the wave fence: synced (control) -> deferred (lean).
        assert off_stages.get("hot.allhit_fence_eval") == 1, off_stages
        assert off_stages.get("hot.allhit_defer", 0) == 0, off_stages
        assert on_stages.get("hot.allhit_fence_eval", 0) == 0, on_stages
        assert on_stages.get("hot.allhit_defer") == 1, on_stages
        assert on_stages.get("hot.allhit_defer_submit") == 1, on_stages  # variant B
        # exactly one routing barrier per layer, both paths.
        assert off_stages.get("hot.eval_indices") == on_stages.get("hot.eval_indices") == 1
    finally:
        rt.close()


def test_miss_sync_count_before_vs_after(tmp_path) -> None:
    # 3 misses + 3 hits in one wave.  Control fences each part (indices + hit fence
    # + 3 miss fences = 5 evals); switch_lean defers every part behind the one
    # routing barrier (1 eval), async-submitting the gathers.
    rt, spec = _open_runtime(tmp_path, resident_slots=_TOP_K, transient=_TOP_K)
    try:
        _warm(rt, spec, [0, 1, 2])  # only 0,1,2 resident; 6,7,8 will miss
        route = [0, 1, 2, 6, 7, 8]
        x, idx = _m1(spec, route)
        off_evals, off_stages = _census_one_switch(rt, spec, x, idx, lean=False)
    finally:
        rt.close()

    # fresh runtime so the on-run starts from the same residency as the off-run.
    dir_on = tmp_path / "on"
    dir_on.mkdir()
    rt2, spec2 = _open_runtime(dir_on, resident_slots=_TOP_K, transient=_TOP_K)
    try:
        _warm(rt2, spec2, [0, 1, 2])
        x2, idx2 = _m1(spec2, [0, 1, 2, 6, 7, 8])
        on_evals, on_stages = _census_one_switch(rt2, spec2, x2, idx2, lean=True)
    finally:
        rt2.close()

    assert off_stages.get("hot.split_route") == 1, off_stages
    assert off_evals > on_evals, (
        f"switch_lean miss ({on_evals} evals) must pay fewer blocking evals than "
        f"control ({off_evals})"
    )
    assert on_evals == 1, f"switch_lean miss expected 1 blocking eval, got {on_evals}"


def test_w81_batched_split_byte_identical_under_switch_lean(tmp_path) -> None:
    # The W42 fastpath fake (_FastpathRuntime in test_deepseek_v41_switch_fastpath)
    # returns a single route wave, so its M=4 split cases never exercise the W81
    # multi-wave batched path.  Cover that composition on the REAL runtime here:
    # a 4-row verify with 20 unique experts and transient capacity 6 forces
    # ceil(20/6)=4 capacity-bounded waves (the batched split), and switch_lean must
    # leave the routed output byte-identical to the fenced control.
    top_k = 6
    n_unique = 20
    route = [i % n_unique for i in range(4 * top_k)]  # 24 assignments, 20 unique
    warm = sorted(set(route))[:n_unique // 2]         # half resident -> hit/miss mix

    def run(lean):
        from tests.test_streamed_models import _integrated_hy3_artifact
        d = tmp_path / ("on" if lean else "off")
        d.mkdir()
        root, _c, spec, mp = _integrated_hy3_artifact(d, expert_count=24, top_k=top_k)
        fixed = spec.resident_bytes + spec.transient_scratch_bytes
        sc = ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=fixed + spec.persistent_cache_bytes(24),
            max_live_kv_tokens=0, runtime_reserve_bytes=0,
            transient_slots=6, slot_layout="component-banks",
        )
        rt = ExpertStreamingRuntime.open(
            root, mp, sc, spec=spec,
            buffer_allocator=make_mlx_component_bank_allocator(
                sc.memory_plan(spec), spec, load_expert_manifest(mp)),
            device_synchronize=mx.synchronize, apply_memory_cap=False)
        try:
            _warm(rt, spec, warm)
            _arm(lean)
            mx.random.seed(81)
            x = (0.3 * mx.random.normal((4, 1, spec.hidden_size))).astype(mx.bfloat16)
            idx = mx.array(route, dtype=mx.int32).reshape((4, top_k))
            _REAL_EVAL(x, idx)
            out = HotExpertSwitchGLU(rt, spec.routed_layer_start)(x, idx)
            _REAL_EVAL(out)
            rt.flush_deferred_slot_releases(evaluate=True)
            return out
        finally:
            os.environ.pop(FASTPATH, None)
            os.environ.pop(SUBMIT, None)
            rt.close()

    out_off = run(False)
    out_on = run(True)
    assert out_off.shape == out_on.shape == (4, top_k, 64)
    assert mx.array_equal(out_off, out_on), "W81 batched split under switch_lean != control"


def test_miss_route_byte_identical(tmp_path) -> None:
    dir_off = tmp_path / "off"
    dir_on = tmp_path / "on"
    dir_off.mkdir()
    dir_on.mkdir()
    rt, spec = _open_runtime(dir_off, resident_slots=_TOP_K, transient=_TOP_K)
    try:
        _warm(rt, spec, [0, 1, 2])
        x, idx = _m1(spec, [0, 1, 2, 6, 7, 8])
        _arm(False)
        out_off = HotExpertSwitchGLU(rt, spec.routed_layer_start)(x, idx)
        _REAL_EVAL(out_off)
        rt.flush_deferred_slot_releases(evaluate=True)
    finally:
        rt.close()
    rt2, spec2 = _open_runtime(dir_on, resident_slots=_TOP_K, transient=_TOP_K)
    try:
        _warm(rt2, spec2, [0, 1, 2])
        x2, idx2 = _m1(spec2, [0, 1, 2, 6, 7, 8])
        _arm(True)
        out_on = HotExpertSwitchGLU(rt2, spec2.routed_layer_start)(x2, idx2)
        _REAL_EVAL(out_on)
        rt2.flush_deferred_slot_releases(evaluate=True)
        assert mx.array_equal(out_off, out_on), "miss route: switch_lean != control"
    finally:
        rt2.close()


# ---------------------------------------------------------------------------
# 3. pin-safety: an eviction attempt on a deferred route's slot is refused /
#    redirected while the release is pending, and allowed after the flush.
# ---------------------------------------------------------------------------
def test_control_evicts_below_the_pin_set_when_unpinned(tmp_path) -> None:
    # Contrast baseline for the pin test below (proves it is NOT vacuous): with the
    # SAME residency but NO deferral (fenced), the pin set holds no pins, so a
    # forced eviction below the unpinned count succeeds -- it evicts pin-set
    # experts. The deferred run must instead refuse (next test).
    rt, spec = _open_runtime(tmp_path, resident_slots=8, transient=_TOP_K)
    layer = spec.routed_layer_start
    try:
        _warm(rt, spec, list(range(_TOP_K)))     # pin-set 0..5 warmed first
        _warm(rt, spec, [6, 7])                   # 2 non-pin experts
        _arm(False)                               # fenced: no pins held
        out = HotExpertSwitchGLU(rt, layer)(*_m1(spec, list(range(_TOP_K))))
        _REAL_EVAL(out)
        rt.flush_deferred_slot_releases(evaluate=True)
        assert rt.snapshot(mx_module=mx)["slots"]["pins"] == 0
        # occupancy 8 -> cap 2: with no pins this SUCCEEDS (evicts 6, incl pin-set).
        rt._evict_layer_bank_to_capacity(layer, capacity=2)
        resident = set(
            int(e) for e in rt.peek_resident_experts(layer, tuple(range(_EXPERT_COUNT)))
        )
        assert len(resident) <= 2, resident
        assert not all(e in resident for e in range(_TOP_K)), (
            "control (no pins) must evict pin-set experts down to cap 2"
        )
    finally:
        rt.close()


def test_deferred_all_hit_route_refuses_eviction_until_flush(tmp_path) -> None:
    # Non-vacuous pin test: same residency as the control above, but the all-hit
    # route is DEFERRED (switch_lean), so its 6 slots are pinned. A forced eviction
    # below the pinned count cannot be satisfied -- every pinned slot's invalidate
    # is refused (ExpertSlotError -> peek_victim exhausted) -- so it RAISES and the
    # whole pin set survives; the control above shows the un-pinned run evicts them.
    rt, spec = _open_runtime(tmp_path, resident_slots=8, transient=_TOP_K)
    layer = spec.routed_layer_start
    pin_set = list(range(_TOP_K))
    try:
        _warm(rt, spec, pin_set)   # pin-set 0..5 warmed FIRST (oldest in LRU order)
        _warm(rt, spec, [6, 7])    # 2 non-pin experts (newer)
        _arm(True)                 # switch_lean -> the all-hit route defers, pins 0..5
        sw = HotExpertSwitchGLU(rt, layer)
        out = sw(*_m1(spec, pin_set))
        _REAL_EVAL(out)            # settle the gather; the release is still deferred

        pending = getattr(rt, "_deferred_slot_releases", [])
        assert len(pending) == 1, "the all-hit route must be deferred, not fenced"
        ready = pending[0][0]
        bindings = list(getattr(ready, "bindings", ()))
        assert bindings and rt.snapshot(mx_module=mx)["slots"]["pins"] == _TOP_K

        # (a) a direct invalidate of a pinned slot is REFUSED while pending.
        b0 = bindings[0]
        with pytest.raises(ExpertSlotError):
            rt.slots.invalidate(b0.layer, b0.logical_slot, expert=b0.expert)

        # (b) a forced eviction below the pinned count RAISES (only the 2 non-pin
        # experts can go; the 6 pinned slots all refuse) -- the exact case the
        # control run above satisfied. The pin set survives intact.
        from mtplx.expert_runtime import ExpertStreamingConfigurationError
        with pytest.raises(ExpertStreamingConfigurationError):
            rt._evict_layer_bank_to_capacity(layer, capacity=2)
        resident = set(
            int(e) for e in rt.peek_resident_experts(layer, tuple(range(_EXPERT_COUNT)))
        )
        assert all(e in resident for e in pin_set), (
            f"eviction recycled a pinned deferred slot; resident={sorted(resident)}"
        )

        # (c) after the covering flush the pins drop; the same eviction now succeeds
        # and the slot is invalidatable -- the pins, not LRU order, blocked it.
        rt.flush_deferred_slot_releases(evaluate=True)
        assert rt.snapshot(mx_module=mx)["slots"]["pins"] == 0
        rt._evict_layer_bank_to_capacity(layer, capacity=2)  # no raise now
    finally:
        os.environ.pop(FASTPATH, None)
        os.environ.pop(SUBMIT, None)
        rt.close()


def test_reroute_after_flush_byte_identical_to_control(tmp_path) -> None:
    # After a deferred all-hit route is flushed, a re-route to the same experts is
    # byte-for-byte identical to the fenced control path (the deferral is a pure
    # timing reorder; releasing the pins changes nothing about the gather math).
    dir_off = tmp_path / "off"
    dir_on = tmp_path / "on"
    dir_off.mkdir()
    dir_on.mkdir()
    route = list(range(_TOP_K))

    rt, spec = _open_runtime(dir_off, resident_slots=_TOP_K)
    try:
        _warm(rt, spec, route)
        _arm(False)
        sw = HotExpertSwitchGLU(rt, spec.routed_layer_start)
        _REAL_EVAL(sw(*_m1(spec, route)))
        rt.flush_deferred_slot_releases(evaluate=True)
        out_ctrl = sw(*_m1(spec, route))
        _REAL_EVAL(out_ctrl)
    finally:
        rt.close()

    rt2, spec2 = _open_runtime(dir_on, resident_slots=_TOP_K)
    try:
        _warm(rt2, spec2, route)
        _arm(True)
        sw2 = HotExpertSwitchGLU(rt2, spec2.routed_layer_start)
        _REAL_EVAL(sw2(*_m1(spec2, route)))       # deferred
        rt2.flush_deferred_slot_releases(evaluate=True)  # covering flush
        out_lean = sw2(*_m1(spec2, route))        # re-route after flush
        _REAL_EVAL(out_lean)
        rt2.flush_deferred_slot_releases(evaluate=True)
        assert mx.array_equal(out_ctrl, out_lean), "re-route after flush != control"
    finally:
        rt2.close()


# ---------------------------------------------------------------------------
# 4. CRITICAL regression: a deferred split route must not deadlock a KV boundary.
# ---------------------------------------------------------------------------
def test_deferred_split_route_does_not_deadlock_kv_admission(tmp_path) -> None:
    # Under the arm, an M=1 miss route defers _DeferredSplitClose, which keeps the
    # layer lock held until the next covering flush. Maximum KV is already priced,
    # so admission must complete without taking the deferred route's layer lock.
    import threading

    rt, spec = _open_runtime(tmp_path, resident_slots=10, transient=_TOP_K, kv_tokens=64)
    layer = spec.routed_layer_start
    try:
        assert not rt._derived_cache_policy
        _arm(False)
        sw = HotExpertSwitchGLU(rt, layer)
        for e in (0, 1, 2):
            _REAL_EVAL(sw(*_m1(spec, [e] * _TOP_K)))
            rt.flush_deferred_slot_releases(evaluate=True)
        _arm(True)  # switch_lean
        _REAL_EVAL(sw(*_m1(spec, [0, 1, 2, 6, 7, 8])))  # 3 hits + 3 misses -> split
        assert len(getattr(rt, "_deferred_slot_releases", [])) == 1
        assert rt._layer_locks[layer].locked(), "deferred split must hold the layer lock"

        done = threading.Event()
        box = {}

        def admit():
            try:
                with rt.admit_kv_tokens(1):
                    pass
                box["ok"] = True
            except BaseException as exc:  # noqa: BLE001
                box["err"] = repr(exc)
            finally:
                done.set()

        threading.Thread(target=admit, daemon=True).start()
        assert done.wait(5.0), (
            "admit_kv_tokens deadlocked on a deferred split route's layer lock"
        )
        assert box.get("ok"), box
    finally:
        os.environ.pop(FASTPATH, None)
        os.environ.pop(SUBMIT, None)
        rt.flush_deferred_slot_releases(evaluate=True)
        rt.close()


def test_deferred_split_kv_admission_completes_same_thread(tmp_path) -> None:
    # Same-thread variant: the generation thread ITSELF hits the KV boundary while
    # holding a deferred split's layer lock (the real production shape). Static
    # admission must leave that pending gather and its covering flush untouched.
    # A SIGALRM watchdog bounds the same-thread call so a regression fails loudly
    # instead of hanging the suite (SIGALRM fires only on the main thread, where
    # pytest runs without -n auto).
    import signal

    rt, spec = _open_runtime(tmp_path, resident_slots=10, transient=_TOP_K, kv_tokens=64)
    layer = spec.routed_layer_start
    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        assert not rt._derived_cache_policy
        _arm(False)
        sw = HotExpertSwitchGLU(rt, layer)
        for e in (0, 1, 2):
            _REAL_EVAL(sw(*_m1(spec, [e] * _TOP_K)))
            rt.flush_deferred_slot_releases(evaluate=True)
        _arm(True)
        _REAL_EVAL(sw(*_m1(spec, [0, 1, 2, 6, 7, 8])))  # deferred split -> holds lock
        assert rt._layer_locks[layer].locked()

        def _timeout(_signum, _frame):
            raise TimeoutError("admit_kv_tokens self-deadlocked on the same thread")

        signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, 5.0)
        try:
            with rt.admit_kv_tokens(1):  # same thread as the deferral
                pass
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        # Admission preserves the deferral; normal generation owns its flush.
        assert len(getattr(rt, "_deferred_slot_releases", [])) == 1
        assert rt._layer_locks[layer].locked()
        rt.flush_deferred_slot_releases(evaluate=True)
        assert not rt._layer_locks[layer].locked()
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        os.environ.pop(FASTPATH, None)
        os.environ.pop(SUBMIT, None)
        rt.flush_deferred_slot_releases(evaluate=True)
        rt.close()


def test_device_route_lut_falls_back_when_layer_lock_held(tmp_path) -> None:
    # DEVICE_ROUTE reaches device_route_lut BEFORE the covering flush; a pending
    # deferred split holds the layer lock, so a blocking rebuild there would
    # self-deadlock the generation thread. The builder must non-blocking-acquire
    # and return None (fenced fallback) on contention, not hang.
    import signal

    rt, spec = _open_runtime(tmp_path, resident_slots=10, transient=_TOP_K)
    layer = spec.routed_layer_start
    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        _arm(False)
        sw = HotExpertSwitchGLU(rt, layer)
        for e in (0, 1, 2):
            _REAL_EVAL(sw(*_m1(spec, [e] * _TOP_K)))
            rt.flush_deferred_slot_releases(evaluate=True)
        assert rt.device_route_lut(layer) is not None  # builds cleanly (lock free)

        _arm(True)
        _REAL_EVAL(sw(*_m1(spec, [0, 1, 2, 6, 7, 8])))  # deferred split -> holds lock
        assert rt._layer_locks[layer].locked()
        rt._mark_device_route_dirty(layer)  # residency changed -> LUT must rebuild

        def _timeout(_signum, _frame):
            raise TimeoutError("device_route_lut blocked on the held layer lock")

        signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, 5.0)
        try:
            # contended rebuild must return None (fenced fallback), never hang.
            assert rt.device_route_lut(layer) is None
            assert rt.device_route_pinned_lut(layer) is None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

        # after the covering flush the lock frees and the LUT rebuilds.
        rt.flush_deferred_slot_releases(evaluate=True)
        assert rt.device_route_lut(layer) is not None
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        os.environ.pop(FASTPATH, None)
        os.environ.pop(SUBMIT, None)
        rt.flush_deferred_slot_releases(evaluate=True)
        rt.close()


# ---------------------------------------------------------------------------
# 5. the all-hit deferral never fires in PREFILL (no covering eval between groups)
# ---------------------------------------------------------------------------
def test_all_hit_deferral_never_fires_in_prefill(tmp_path) -> None:
    from mtplx.models.expert_mlx import current_expert_routing_phase
    from mtplx.expert_streaming import RoutingPhase

    rt, spec = _open_runtime(tmp_path, resident_slots=_TOP_K, transient=_TOP_K)
    layer = spec.routed_layer_start
    try:
        _warm(rt, spec, list(range(_TOP_K)))
        _arm(True)  # switch_lean armed
        # A multi-token forward is PREFILL: x shape [1, T>1, H] -> T tokens.
        rows = 4
        mx.random.seed(92)
        x = (0.3 * mx.random.normal((1, rows, spec.hidden_size))).astype(mx.bfloat16)
        idx = mx.array([[e for e in range(_TOP_K)] for _ in range(rows)], dtype=mx.int32)
        _REAL_EVAL(x, idx)
        assert current_expert_routing_phase(token_count=rows) is RoutingPhase.PREFILL
        _evals, stages = _census_one_switch(rt, spec, x, idx, lean=True)
        # even armed, prefill must NOT defer the all-hit release.
        assert stages.get("hot.allhit_defer", 0) == 0, (
            f"prefill must fence, never defer the all-hit release; saw {stages}"
        )
        assert len(getattr(rt, "_deferred_slot_releases", [])) == 0, (
            "prefill left a deferred release pending (no covering eval between groups)"
        )
    finally:
        os.environ.pop(FASTPATH, None)
        os.environ.pop(SUBMIT, None)
        rt.flush_deferred_slot_releases(evaluate=True)
        rt.close()
