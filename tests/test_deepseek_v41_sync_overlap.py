"""W28 -- host-sync census + byte-identity for the DSV4.1 shared-overlap lever.

CPU-pinned, tiny synthetic streamed switch + fake bank (reuses the
``_BankOverlap*`` doubles from ``tests/test_streamed_models.py``).  No GPU, no
artifact, no real weights.

The finding (KERNEL_LEDGER K1 Rank 1): the DSV4.1 streamed forward forces one
``mx.eval(indices)`` routing barrier per streamed layer (~40/token), each a
device->host round-trip with the GPU idle during it.  The lever (behind
``MTPLX_DSV41_SHARED_OVERLAP``, default off) hands the shared expert -- which
depends only on ``x``, never on the routed indices -- to the streamed switch as
``shared_work``, so the switch dispatches it (via ``mx.async_eval``) into that
idle window instead of serialising it after the routed gather.  Pure execution
reorder, so the routed output and the shared output must be bitwise-identical.

These tests lock:

  1. exactly ONE routing barrier (``hot.eval_indices``) per streamed layer, and
     the lever neither deletes it nor adds a blocking sync (the a3b caution:
     removing the decision sync measured +3.27% slower, so overlap-fill, not
     deletion) -- the shared is dispatched as a non-blocking ``async_eval`` and,
     with the hoist active, BEFORE the barrier;
  2. the streamed switch returns bitwise-identical routed + shared with the
     lever off vs on;
  3. W11's ``MoE.__call__`` produces bitwise-identical output with the lever off
     vs on, for both AR (single-row) and verify (multi-row) shapes.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import mlx.core as mx
import pytest

pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from mtplx.models import expert_mlx  # noqa: E402
from mtplx.models.expert_mlx import HotExpertSwitchGLU  # noqa: E402
from mtplx.models.deepseek_v41_moe import MoE  # noqa: E402
from tests.test_streamed_models import (  # noqa: E402
    _bank_overlap_inputs,
    _BankOverlapPending,
    _BankOverlapRuntime,
)

OVERLAP_FLAG = "MTPLX_DSV41_SHARED_OVERLAP"
HOIST_FLAG = "MTPLX_HY3_SHARED_HOIST"

# Pristine references captured once, so a census counter never wraps a counter
# left in place by an earlier call (mx and expert_mlx.mx are the same module
# object, so setattr on one is visible to the other).
_REAL_EVAL = mx.eval
_REAL_ASYNC = mx.async_eval
_REAL_TOLIST = mx.array.tolist
_REAL_BRACKET = expert_mlx._route_probe.bracket
_REAL_BANK_Q4 = expert_mlx._run_component_bank_q4


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


# ---------------------------------------------------------------------------
# 1. host-sync census: one streamed layer, control vs overlap
# ---------------------------------------------------------------------------
def _census(monkeypatch, mode: str, *, deferred: bool):
    """Drive one ``HotExpertSwitchGLU`` decode call and count host syncs.

    Returns ``(counts, order, shared_calls, routed, shared)`` where ``counts``
    tallies blocking ``mx.eval`` / non-blocking ``mx.async_eval`` / device->host
    ``.tolist`` and ``order`` records the interleaving of the routing barrier and
    the shared dispatch.  ``mode`` is 'control' (shipped: plain ``__call__``,
    shared never handed to the switch), 'overlap_nohoist' (shared handed in,
    hoist flag off) or 'overlap_hoist' (shared handed in, hoist armed).
    """
    for key in (HOIST_FLAG, OVERLAP_FLAG):
        monkeypatch.delenv(key, raising=False)
    if mode == "overlap_hoist":
        monkeypatch.setenv(OVERLAP_FLAG, "1")

    counts = {"eval": 0, "async_eval": 0, "tolist": 0}
    order: list[str] = []
    shared_calls = {"n": 0}

    def c_eval(*a, **k):
        counts["eval"] += 1
        order.append("eval")
        return _REAL_EVAL(*a, **k)

    def c_async(*a, **k):
        counts["async_eval"] += 1
        order.append("async_eval")
        return _REAL_ASYNC(*a, **k)

    def c_tolist(self, *a, **k):
        counts["tolist"] += 1
        return _REAL_TOLIST(self, *a, **k)

    # Count the routing barrier specifically via the existing stage bracket, so
    # the "one barrier per layer" claim is measured, not inferred from raw evals.
    @contextmanager
    def c_bracket(stage: str):
        if stage == "hot.eval_indices":
            order.append("barrier")
        with _REAL_BRACKET(stage):
            yield

    def shared_work() -> mx.array:
        shared_calls["n"] += 1
        order.append("shared")
        return mx.ones((1, 1, 2), dtype=mx.bfloat16)

    # Restore from the pristine originals in ``finally`` so back-to-back census
    # calls in one test never stack counters (setattr on mx is global).
    expert_mlx.mx.eval = c_eval
    expert_mlx.mx.async_eval = c_async
    mx.array.tolist = c_tolist
    expert_mlx._route_probe.bracket = c_bracket
    expert_mlx._run_component_bank_q4 = lambda selected, *a, **k: selected
    try:
        events: list[str] = []
        runtime = _BankOverlapRuntime(events, _BankOverlapPending(events))
        runtime.config.deferred_pin_release = deferred
        x, indices = _bank_overlap_inputs()
        switch = HotExpertSwitchGLU(runtime, 1)
        if mode == "control":
            routed = switch(x, indices)  # plain __call__ -> shared_work=None
            shared = None
        else:
            routed, shared = switch.run_with_shared_overlap(x, indices, shared_work)
    finally:
        expert_mlx.mx.eval = _REAL_EVAL
        expert_mlx.mx.async_eval = _REAL_ASYNC
        mx.array.tolist = _REAL_TOLIST
        expert_mlx._route_probe.bracket = _REAL_BRACKET
        expert_mlx._run_component_bank_q4 = _REAL_BANK_Q4

    if shared is None:
        _REAL_EVAL(routed)
    else:
        _REAL_EVAL(routed, shared)
    return counts, order, shared_calls["n"], routed, shared


def test_exactly_one_routing_barrier_per_streamed_layer(monkeypatch) -> None:
    """The ``mx.eval(indices)`` barrier the ledger counts as ~40/token fires
    exactly once per streamed layer -- in every mode.  The lever must not add a
    second one."""
    for mode in ("control", "overlap_nohoist", "overlap_hoist"):
        counts, order, *_ = _census(monkeypatch, mode, deferred=True)
        assert order.count("barrier") == 1, f"{mode}: {order}"


@pytest.mark.parametrize("deferred", [True, False])
def test_overlap_hoist_adds_no_blocking_sync(monkeypatch, deferred) -> None:
    """Overlap-fill must not delete the barrier (a3b: -3.27% when the sync was
    removed) and must not add a blocking ``mx.eval``: the shared is dispatched
    as a non-blocking ``async_eval``, and with the hoist it lands BEFORE the
    barrier so it fills the barrier's GPU-idle round-trip."""
    c_ctl, _o_ctl, n_ctl, _r, _s = _census(monkeypatch, "control", deferred=deferred)
    c_ov, o_ov, n_ov, _r2, shared = _census(
        monkeypatch, "overlap_hoist", deferred=deferred
    )

    assert n_ctl == 0  # control never hands the shared to the switch
    assert n_ov == 1, "shared must be computed exactly once (no double-compute)"
    assert shared is not None
    # Same number of BLOCKING host syncs as the shipped path -- the barrier is
    # preserved, nothing new blocks.
    assert c_ov["eval"] == c_ctl["eval"], (c_ctl, c_ov)
    # The shared branch is added as one non-blocking async dispatch.
    assert c_ov["async_eval"] == c_ctl["async_eval"] + 1, (c_ctl, c_ov)
    # ... and it is submitted before the routing barrier (fills its idle window).
    assert "shared" in o_ov and "barrier" in o_ov
    assert o_ov.index("shared") < o_ov.index("barrier"), o_ov
    # Exactly one device->host route materialization (.tolist) either way.
    assert c_ov["tolist"] == c_ctl["tolist"] == 1


# ---------------------------------------------------------------------------
# 2. streamed switch: routed + shared bitwise-identical, off vs on
# ---------------------------------------------------------------------------
def _drive_switch(monkeypatch, *, overlap: bool, deferred: bool):
    for key in (HOIST_FLAG, OVERLAP_FLAG):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        expert_mlx, "_run_component_bank_q4", lambda selected, *a, **k: selected
    )
    events: list[str] = []
    runtime = _BankOverlapRuntime(events, _BankOverlapPending(events))
    runtime.config.deferred_pin_release = deferred
    x, indices = _bank_overlap_inputs()
    switch = HotExpertSwitchGLU(runtime, 1)

    def shared_work() -> mx.array:
        # A non-trivial shared branch that actually depends on x, so a stale or
        # reordered value would show up as a bit difference.
        return (x * mx.array(3.0, dtype=mx.bfloat16)) + mx.array(1.0, dtype=mx.bfloat16)

    if overlap:
        monkeypatch.setenv(OVERLAP_FLAG, "1")
        routed, shared = switch.run_with_shared_overlap(x, indices, shared_work)
    else:
        routed = switch(x, indices)
        shared = shared_work()
    mx.eval(routed, shared)
    return routed, shared


@pytest.mark.parametrize("deferred", [True, False])
def test_streamed_switch_routed_and_shared_bitwise_identical(
    monkeypatch, deferred
) -> None:
    routed_off, shared_off = _drive_switch(monkeypatch, overlap=False, deferred=deferred)
    routed_on, shared_on = _drive_switch(monkeypatch, overlap=True, deferred=deferred)
    assert mx.array_equal(routed_off, routed_on), "routed output changed under overlap"
    assert mx.array_equal(shared_off, shared_on), "shared output changed under overlap"


# ---------------------------------------------------------------------------
# 3. W11 MoE.__call__: bitwise-identical output, flag off vs on
# ---------------------------------------------------------------------------
def _tiny_moe_args() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=8,
        n_routed_experts=6,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        n_shared_experts=1,
        swiglu_limit=10.0,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
    )


def _randomize_moe(moe: MoE, seed: int = 0) -> None:
    mx.random.seed(seed)
    new = []
    for name, arr in tree_flatten(moe.parameters()):
        if name.endswith("e_score_correction_bias"):
            v = 0.1 * mx.random.normal(arr.shape)
        else:
            v = 0.3 * mx.random.normal(arr.shape)
        new.append((name, v.astype(arr.dtype)))
    moe.update(tree_unflatten(new))
    mx.eval(moe.parameters())


def _moe_output(monkeypatch, x: mx.array, *, overlap: bool) -> mx.array:
    for key in (HOIST_FLAG, OVERLAP_FLAG):
        monkeypatch.delenv(key, raising=False)
    if overlap:
        monkeypatch.setenv(OVERLAP_FLAG, "1")
    args = _tiny_moe_args()
    moe = MoE(0, args)
    _randomize_moe(moe, seed=11)
    out = moe(x)
    mx.eval(out)
    return out


@pytest.mark.parametrize("n_tokens", [1, 4])
def test_moe_call_order_bitwise_identical_off_vs_on(monkeypatch, n_tokens) -> None:
    """The resident ``SwitchGLU`` has no ``run_with_shared_overlap``, so the
    lever falls back to the shipped ordering: the refactored combine must be
    bitwise-identical for AR (n=1) and verify (n=4) shapes."""
    mx.random.seed(7)
    x = 0.5 * mx.random.normal((n_tokens, 8)).astype(mx.bfloat16)
    mx.eval(x)
    out_off = _moe_output(monkeypatch, x, overlap=False)
    out_on = _moe_output(monkeypatch, x, overlap=True)
    assert out_off.shape == out_on.shape == (n_tokens, 8)
    assert mx.array_equal(out_off, out_on), "MoE output changed under the lever"
