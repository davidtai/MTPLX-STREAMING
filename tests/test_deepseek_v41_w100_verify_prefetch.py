"""W100 -- the DSpark verify path issues gate-oracle prefetch, provably.

Three claims, on a CPU-pinned real streamed runtime + fake bank (the W95 harness):

  1. A verify-shaped (M=K+1, DECODE) route under ``MTPLX_DSV41_RUNNER=v2`` issues
     speculative prefetch reads AND the new verify-phase counter
     (``prefetch_issued_verify`` / ``prefetch_committed_verify``) increments, while
     an AR (M=1) route leaves the verify counter at 0 -- so the paired GPU window
     can tell verify-phase prefetch from the AR total.

  2. The switch output is byte-identical with the prefetch prediction stashed vs
     not -- prefetch only changes what is resident, never the gathered math.

  3. The harness wall accounting helper (``_dspark_decode_wall_accounting``)
     excludes the re-prefill from ``decode_wall_s`` and keeps the whole-call figure
     as ``pass_wall_s``.

Run this file in its own process (``pytest tests/test_deepseek_v41_w100_verify_prefetch.py``),
under ``nice -n 19``, no ``-n auto``: MLX defaults to Metal, so the module pins CPU
before any array op.
"""

import os

import mlx.core as mx

mx.set_default_device(mx.cpu)  # MLX defaults to Metal; pin CPU before any array op.

import pytest

from mtplx.models.deepseek_v41 import _GatePrefetchLink
from tests.test_deepseek_v41_w95_runner_v2 import (
    _inputs,
    _open_runtime,
    _route_once,
    _settle_prefetch,
    _switch,
)


@pytest.fixture(autouse=True)
def _isolate_runner_env():
    """`_open_runtime` sets/pops MTPLX_DSV41_RUNNER in os.environ; restore it."""
    mx.set_default_device(mx.cpu)
    saved = os.environ.get("MTPLX_DSV41_RUNNER")
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("MTPLX_DSV41_RUNNER", None)
        else:
            os.environ["MTPLX_DSV41_RUNNER"] = saved


def _runner_block(rt):
    return rt.resource_telemetry_snapshot(mx_module=mx).get("runner", {})


def _run_route_with_prediction(rt, spec, *, rows, route_experts, predicted):
    """Warm residents, stash a NEXT-layer prediction of ``predicted`` on the switch,
    then run one ``rows``-row route over ``route_experts`` -- exactly the sequence a
    verify (rows=K+1) / AR (rows=1) forward drives through the switch, minus the
    backbone. Returns the switch output."""
    layer = spec.routed_layer_start
    _route_once(rt, spec, [0, 1])  # residents 0,1 -> route over them is all-hit
    sw = _switch(rt, spec)
    if predicted is not None:
        pend = mx.array([list(predicted)] * rows, dtype=mx.int32)
        sw._mtplx_gate_prefetch_pending = _GatePrefetchLink(layer, None, pend)
    x, idx = _inputs(rows, spec.top_k, spec.hidden_size, route_experts)
    out = sw(x, idx)
    mx.eval(out)
    rt.flush_deferred_slot_releases(evaluate=True)
    return out


# --------------------------------------------------------------------------
# 1. verify issues prefetch + the verify-phase counter increments; AR does not
# --------------------------------------------------------------------------
def test_verify_route_issues_prefetch_and_bumps_verify_counter(tmp_path):
    rt, spec = _open_runtime(
        tmp_path / "verify", runner_v2=True, expert_count=8, top_k=2,
        resident_slots=2, transient=8, prefetch=10,
    )
    try:
        layer = spec.routed_layer_start
        # a 6-row (K+1, depth-5) verify over resident experts, predicting a
        # non-resident, non-route union for the next layer.
        _run_route_with_prediction(
            rt, spec, rows=6, route_experts=[0, 1], predicted=[4, 5],
        )
        block = _runner_block(rt)
        assert block.get("prefetch_issued", 0) > 0, "verify issued no prefetch at all"
        assert block.get("prefetch_issued_verify", 0) > 0, (
            "verify-phase prefetch counter did not increment"
        )
        # commit path: settle the reads and apply them (a follow-up flush applies
        # this layer's settled completions), then the verify commit is attributed.
        _settle_prefetch(rt)
        rt.prefetch_experts(layer, [])
        block = _runner_block(rt)
        assert block.get("prefetch_committed_verify", 0) > 0, (
            "verify-phase prefetch never committed"
        )
        # the verify counter is a subset of the merged total.
        assert block["prefetch_issued_verify"] <= block["prefetch_issued"]
        assert block["prefetch_committed_verify"] <= block["prefetch_committed"]
    finally:
        rt.close()


def test_ar_route_leaves_verify_counter_zero(tmp_path):
    rt, spec = _open_runtime(
        tmp_path / "ar", runner_v2=True, expert_count=8, top_k=2,
        resident_slots=2, transient=8, prefetch=10,
    )
    try:
        _run_route_with_prediction(
            rt, spec, rows=1, route_experts=[0, 1], predicted=[4, 5],
        )
        block = _runner_block(rt)
        assert block.get("prefetch_issued", 0) > 0, "AR issued no prefetch (setup broke)"
        assert block.get("prefetch_issued_verify", 0) == 0, (
            "AR (M=1) prefetch was miscounted as verify-phase"
        )
        _settle_prefetch(rt)
        rt.prefetch_experts(spec.routed_layer_start, [])
        assert _runner_block(rt).get("prefetch_committed_verify", 0) == 0
    finally:
        rt.close()


# --------------------------------------------------------------------------
# 2. byte-identity: prefetch stashed vs not changes residency, never the math
# --------------------------------------------------------------------------
@pytest.mark.parametrize("rows", [1, 6])
def test_output_byte_identical_prefetch_on_vs_off(tmp_path, rows):
    # A miss route (experts 6,7 are never resident) so the switch actually gathers
    # streamed bytes; identical input on both runtimes.
    miss = [6, 7]

    rt_off, spec = _open_runtime(
        tmp_path / f"off{rows}", runner_v2=True, expert_count=8, top_k=2,
        resident_slots=2, transient=8, prefetch=10,
    )
    try:
        out_off = _run_route_with_prediction(
            rt_off, spec, rows=rows, route_experts=miss, predicted=None,
        )
        mx.eval(out_off)
    finally:
        rt_off.close()

    rt_on, spec2 = _open_runtime(
        tmp_path / f"on{rows}", runner_v2=True, expert_count=8, top_k=2,
        resident_slots=2, transient=8, prefetch=10,
    )
    try:
        # predict a union that OVERLAPS the route (6,7) plus extra (2,3): even
        # pre-warming the route's own experts must not change the gathered value.
        out_on = _run_route_with_prediction(
            rt_on, spec2, rows=rows, route_experts=miss, predicted=[2, 3, 6, 7],
        )
        mx.eval(out_on)
    finally:
        rt_on.close()

    assert mx.array_equal(out_off, out_on), (
        f"rows={rows}: stashing a gate-oracle prefetch changed the gathered output"
    )


# --------------------------------------------------------------------------
# 3. the harness wall accounting helper excludes the re-prefill
# --------------------------------------------------------------------------
def test_dspark_wall_accounting_excludes_prefill():
    from scripts.deepseek_v41.ab_decode_env_levers import (
        _dspark_decode_wall_accounting,
    )

    # window-39 shape: 297.36 s whole call, prefill (TTFT) 203.96 s, decode 93.4 s,
    # 257 generated tokens. decode-only rate must be ~2.75 tok/s, not 0.86.
    acct = _dspark_decode_wall_accounting(
        pass_start=0.0, decode_start=203.96, pass_end=297.36, generated_tokens=257,
    )
    assert acct["pass_wall_s"] == pytest.approx(297.36)
    assert acct["decode_wall_s"] == pytest.approx(93.4)
    assert acct["decode_tok_s"] == pytest.approx(257 / 93.4, rel=1e-6)
    # the misreported whole-call rate the helper replaces:
    assert 257 / acct["pass_wall_s"] == pytest.approx(0.864, abs=1e-3)

    # no prefill callback -> fall back to the whole-call wall (never crash / None).
    fallback = _dspark_decode_wall_accounting(
        pass_start=0.0, decode_start=None, pass_end=100.0, generated_tokens=50,
    )
    assert fallback["decode_wall_s"] == fallback["pass_wall_s"] == 100.0
    assert fallback["decode_tok_s"] == pytest.approx(0.5)

    # a non-positive decode wall reports None rather than dividing by zero.
    degenerate = _dspark_decode_wall_accounting(
        pass_start=0.0, decode_start=100.0, pass_end=100.0, generated_tokens=10,
    )
    assert degenerate["decode_wall_s"] == 0.0
    assert degenerate["decode_tok_s"] is None
