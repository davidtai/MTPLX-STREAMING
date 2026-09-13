"""Unit tests for the W125 host decode timeline probe (mtplx.dsv41_decode_timeline).

CPU-only, stdlib-only: the probe imports nothing but the standard library, so these
tests never load MLX or a model. They drive the probe's public API with a *fake
runner* -- a scripted sequence of the marks the real model + expert runtime emit
per decode token -- against a controllable clock, and assert the aggregated phase
arithmetic. A separate test asserts the probe is a pure no-op when the env flag is
unset, and another that the stamped overhead stays under the 0.5 ms/token budget.
"""

from __future__ import annotations

import importlib

import pytest


def _fresh(monkeypatch, armed: bool):
    """Import a fresh copy of the module with the env flag set/unset, so the
    import-time env read (``_ARMED_ENV``) reflects ``armed``."""
    if armed:
        monkeypatch.setenv("MTPLX_DSV41_DECODE_TIMELINE", "1")
    else:
        monkeypatch.delenv("MTPLX_DSV41_DECODE_TIMELINE", raising=False)
    import mtplx.dsv41_decode_timeline as mod

    return importlib.reload(mod)


class _Clock:
    """A settable monotonic-ns clock the tests advance by hand."""

    def __init__(self) -> None:
        self.t = 0

    def __call__(self) -> int:
        return self.t


class FakeRunner:
    """Plays the probe's public marks in the order the real DSV4.1 v2 decode path
    emits them, with an explicitly scripted clock so every phase delta is exact.

    ``layer_miss`` drives the full fenced-route sequence (barrier -> reconcile ->
    miss issue -> miss ready -> dispatch); ``layer_hit`` drives an all-hit layer
    (barrier -> all_hit -> dispatch, no miss I/O). Times are absolute ns.
    """

    def __init__(self, tl, clock: _Clock) -> None:
        self.tl = tl
        self.clk = clock
        # Real perf_counter_ns is a large boot-relative value, never 0 (0.0 is the
        # probe's "unset" sentinel). Offset the scripted clock so no stamp is 0;
        # deltas -- which is all the arithmetic asserts -- are unchanged.
        self.base = 10 ** 12

    def _at(self, t: int):
        self.clk.t = self.base + int(t)

    def token_begin(self, t: int, n_layers: int) -> None:
        self._at(t)
        self.tl.token_begin(n_layers)

    def token_head_done(self, t: int) -> None:
        self._at(t)
        self.tl.token_head_done()

    def forward_end(self, t: int) -> None:
        self._at(t)
        self.tl.forward_end()

    def layer_miss(
        self,
        layer: int,
        *,
        l_start: int,
        attn_start: int,
        attn_end: int,
        moe_start: int,
        barrier: int,
        reconcile_start: int,
        reconcile_end: int,
        miss_issue: int,
        miss_wait_start: int,
        miss_ready: int,
        dispatched: int,
        l_end: int,
    ) -> None:
        tl = self.tl
        self._at(l_start); tl.layer_start(layer)
        self._at(attn_start); tl.attn_start(layer)
        self._at(attn_end); tl.attn_end(layer)
        self._at(moe_start); tl.moe_start(layer)
        self._at(barrier); tl.barrier_done(layer)
        # reconcile duration accumulator (start captured before the call)
        self._at(reconcile_start); rc = tl.now()
        self._at(reconcile_end); tl.add_reconcile(layer, rc); tl.reconcile_done(layer)
        self._at(miss_issue); tl.miss_issue(layer)
        # exposed miss wait accumulator
        self._at(miss_wait_start); mw = tl.now()
        self._at(miss_ready); tl.add_miss_wait(layer, mw); tl.miss_ready(layer)
        self._at(dispatched); tl.expert_dispatched(layer)
        self._at(l_end); tl.layer_end(layer)

    def layer_hit(
        self,
        layer: int,
        *,
        l_start: int,
        attn_start: int,
        attn_end: int,
        moe_start: int,
        barrier: int,
        all_hit: int,
        dispatched: int,
        l_end: int,
    ) -> None:
        tl = self.tl
        self._at(l_start); tl.layer_start(layer)
        self._at(attn_start); tl.attn_start(layer)
        self._at(attn_end); tl.attn_end(layer)
        self._at(moe_start); tl.moe_start(layer)
        self._at(barrier); tl.barrier_done(layer)
        self._at(all_hit); tl.all_hit(layer)
        self._at(dispatched); tl.expert_dispatched(layer)
        self._at(l_end); tl.layer_end(layer)


# --------------------------------------------------------------------------- #
# No-op when unset
# --------------------------------------------------------------------------- #
def test_noop_when_env_unset(monkeypatch):
    tl = _fresh(monkeypatch, armed=False)
    assert tl.env_armed() is False
    # configure must not arm the probe
    tl.configure(40)
    assert tl.enabled() is False
    # every hot-path mark is a silent no-op and must not raise
    tl.token_begin(40)
    tl.layer_start(0)
    tl.attn_start(0)
    tl.attn_end(0)
    tl.moe_start(0)
    tl.barrier_done(0)
    start = tl.now()
    assert start == 0  # now() returns 0 when not recording
    tl.add_reconcile(0, start)
    tl.reconcile_done(0)
    tl.miss_issue(0)
    tl.add_miss_wait(0, start)
    tl.miss_ready(0)
    tl.all_hit(0)
    tl.expert_dispatched(0)
    tl.layer_end(0)
    tl.token_head_done()
    tl.forward_end()
    snap = tl.snapshot()
    assert snap == {"enabled": False, "env_armed": False}


def test_env_armed_but_unconfigured_is_disabled(monkeypatch):
    tl = _fresh(monkeypatch, armed=True)
    assert tl.env_armed() is True
    # armed by env, but not yet sized -> still disabled until configure runs
    assert tl.enabled() is False
    # marks before configure are harmless no-ops
    tl.layer_start(0)
    tl.barrier_done(0)
    assert tl.snapshot()["enabled"] is False


# --------------------------------------------------------------------------- #
# Phase arithmetic (fake runner, scripted clock)
# --------------------------------------------------------------------------- #
def test_phase_arithmetic_single_miss_layer(monkeypatch):
    tl = _fresh(monkeypatch, armed=True)
    tl.configure(2)                 # real-clock calibration happens here
    clk = _Clock()
    monkeypatch.setattr(tl, "_perf", clk)   # scripted clock for the marks
    fr = FakeRunner(tl, clk)

    fr.token_begin(1000, 2)
    fr.layer_miss(
        0,
        l_start=1000, attn_start=1000, attn_end=1500,
        moe_start=1500, barrier=2500,
        reconcile_start=2500, reconcile_end=2700,
        miss_issue=2800, miss_wait_start=2800, miss_ready=4000,
        dispatched=4200, l_end=4400,
    )
    fr.token_head_done(4600)
    fr.forward_end(4700)

    snap = tl.snapshot()
    assert snap["enabled"] is True
    assert snap["tokens_recorded"] == 1
    assert snap["n_layers"] == 2

    ms = 1e6

    # per-layer phase deltas (layer 0), in ms
    pl = snap["per_layer"]
    assert pl["attn"]["0"]["mean_ms"] == pytest.approx(500 / ms)
    assert pl["gate_to_barrier"]["0"]["mean_ms"] == pytest.approx(1000 / ms)
    assert pl["barrier_to_issue"]["0"]["mean_ms"] == pytest.approx(300 / ms)
    assert pl["miss_wait"]["0"]["mean_ms"] == pytest.approx(1200 / ms)  # 4000-2800
    assert pl["dispatch"]["0"]["mean_ms"] == pytest.approx(200 / ms)    # 4200-4000
    assert pl["combine"]["0"]["mean_ms"] == pytest.approx(200 / ms)     # 4400-4200
    assert pl["moe_total"]["0"]["mean_ms"] == pytest.approx(2900 / ms)  # 4400-1500
    assert pl["layer_total"]["0"]["mean_ms"] == pytest.approx(3400 / ms)  # 4400-1000

    pt = snap["per_token"]
    # accumulators
    assert pt["reconcile_total"]["mean_ms"] == pytest.approx(200 / ms)   # add_reconcile
    assert pt["miss_wait_total"]["mean_ms"] == pytest.approx(1200 / ms)  # add_miss_wait
    assert pt["attn_total"]["mean_ms"] == pytest.approx(500 / ms)
    assert pt["routing_barrier_total"]["mean_ms"] == pytest.approx(1000 / ms)
    # host_gap per token == sum(expert_dispatched - attn_end) = 4200-1500 = 2700
    assert pt["host_gap"]["mean_ms"] == pytest.approx(2700 / ms)
    # head dispatch = head_done - token_start = 4600-1000
    assert pt["head_dispatch"]["mean_ms"] == pytest.approx(3600 / ms)
    assert pt["miss_layers"]["mean"] == pytest.approx(1.0)  # one miss layer
    assert pt["hit_layers"]["mean"] == pytest.approx(0.0)
    # single token: no consecutive-start delta, so token_total has no sample
    assert pt["token_total"]["n"] == 0


def test_all_hit_layer_has_no_miss_phase(monkeypatch):
    tl = _fresh(monkeypatch, armed=True)
    tl.configure(1)
    clk = _Clock()
    monkeypatch.setattr(tl, "_perf", clk)
    fr = FakeRunner(tl, clk)

    fr.token_begin(0, 1)
    fr.layer_hit(
        0,
        l_start=0, attn_start=0, attn_end=100,
        moe_start=100, barrier=300, all_hit=320,
        dispatched=350, l_end=400,
    )
    fr.token_head_done(450)
    fr.forward_end(500)

    snap = tl.snapshot()
    pl = snap["per_layer"]
    # no miss issue/ready recorded -> the miss_wait phase has zero samples
    assert pl["miss_wait"] == {}
    assert pl["barrier_to_issue"] == {}
    assert snap["per_token"]["hit_layers"]["mean"] == pytest.approx(1.0)
    assert snap["per_token"]["miss_layers"]["mean"] == pytest.approx(0.0)
    assert snap["per_token"]["miss_wait_total"]["mean_ms"] == pytest.approx(0.0)
    # host_gap still measured: dispatched - attn_end = 350 - 100 = 250
    assert snap["per_token"]["host_gap"]["mean_ms"] == pytest.approx(250 / 1e6)


def test_percentiles_and_token_total_multi_token(monkeypatch):
    tl = _fresh(monkeypatch, armed=True)
    tl.configure(1)
    clk = _Clock()
    monkeypatch.setattr(tl, "_perf", clk)
    fr = FakeRunner(tl, clk)

    # token 0 starts at 0, token 1 at 1000, token 2 at 3000
    # -> token_total samples: 1000 (t0->t1) and 2000 (t1->t2)
    starts = [0, 1000, 3000]
    # give each token's single layer a distinct attn span: 100, 200, 300
    attn_spans = [100, 200, 300]
    for i, t0 in enumerate(starts):
        fr.token_begin(t0, 1)
        fr.layer_hit(
            0,
            l_start=t0, attn_start=t0, attn_end=t0 + attn_spans[i],
            moe_start=t0 + attn_spans[i], barrier=t0 + attn_spans[i] + 10,
            all_hit=t0 + attn_spans[i] + 15, dispatched=t0 + attn_spans[i] + 20,
            l_end=t0 + attn_spans[i] + 25,
        )
        fr.token_head_done(t0 + attn_spans[i] + 30)
    fr.forward_end(6000)

    snap = tl.snapshot()
    assert snap["tokens_recorded"] == 3
    ms = 1e6

    tot = snap["per_token"]["token_total"]
    assert tot["n"] == 2                          # two consecutive-start deltas
    assert tot["mean_ms"] == pytest.approx(1500 / ms)   # (1000 + 2000) / 2
    # p50 of [1000, 2000] with linear interpolation == mean
    assert tot["p50_ms"] == pytest.approx(1500 / ms)
    # p95 == 1000 + (2000-1000)*0.95
    assert tot["p95_ms"] == pytest.approx(1950 / ms)

    # per-layer attn percentiles over 3 tokens: [100, 200, 300]
    attn = snap["per_layer"]["attn"]["0"]
    assert attn["n"] == 3
    assert attn["mean_ms"] == pytest.approx(200 / ms)
    assert attn["p50_ms"] == pytest.approx(200 / ms)
    assert attn["p95_ms"] == pytest.approx(290 / ms)  # 100 + 200*0.95


def test_capacity_clamp_does_not_crash(monkeypatch):
    tl = _fresh(monkeypatch, armed=True)
    tl.configure(1, max_tokens=2)
    clk = _Clock()
    monkeypatch.setattr(tl, "_perf", clk)
    fr = FakeRunner(tl, clk)

    for i in range(5):  # more tokens than capacity (2)
        fr.token_begin(i * 1000, 1)
        fr.layer_hit(
            0,
            l_start=i * 1000, attn_start=i * 1000, attn_end=i * 1000 + 50,
            moe_start=i * 1000 + 50, barrier=i * 1000 + 60, all_hit=i * 1000 + 65,
            dispatched=i * 1000 + 70, l_end=i * 1000 + 75,
        )
        fr.token_head_done(i * 1000 + 80)
    fr.forward_end(9000)

    snap = tl.snapshot()
    assert snap["tokens_seen"] == 5
    assert snap["tokens_recorded"] == 2
    assert snap["tokens_dropped_over_capacity"] == 3


def test_overhead_within_budget(monkeypatch):
    # Uses the REAL clock (no patch) so the calibrated mark cost is a genuine
    # measurement, and drives a realistic mark count per token.
    tl = _fresh(monkeypatch, armed=True)
    tl.configure(40)
    n_layers = 40
    tokens = 64
    for t in range(tokens):
        tl.token_begin(n_layers)
        for l in range(n_layers):
            tl.layer_start(l)
            tl.attn_start(l)
            tl.attn_end(l)
            tl.moe_start(l)
            tl.barrier_done(l)
            s = tl.now()
            tl.add_reconcile(l, s)
            tl.reconcile_done(l)
            tl.miss_issue(l)
            s = tl.now()
            tl.add_miss_wait(l, s)
            tl.miss_ready(l)
            tl.expert_dispatched(l)
            tl.layer_end(l)
        tl.token_head_done()
        tl.forward_end()

    snap = tl.snapshot()
    ov = snap["overhead"]
    assert ov["marks_total"] > 0
    assert ov["mark_cost_ns_calibrated"] > 0
    assert ov["overhead_ms_per_token"] is not None
    # Budget: the probe must add < 0.5 ms/token.
    assert ov["overhead_ms_per_token"] < 0.5
    assert ov["within_budget"] is True


def test_reset_clears_between_arms(monkeypatch):
    tl = _fresh(monkeypatch, armed=True)
    tl.configure(1)
    clk = _Clock()
    monkeypatch.setattr(tl, "_perf", clk)
    fr = FakeRunner(tl, clk)

    fr.token_begin(0, 1)
    fr.layer_hit(
        0, l_start=0, attn_start=0, attn_end=100, moe_start=100,
        barrier=200, all_hit=210, dispatched=220, l_end=230,
    )
    fr.forward_end(300)
    assert tl.snapshot()["tokens_recorded"] == 1

    tl.reset()
    snap = tl.snapshot()
    assert snap["tokens_recorded"] == 0
    assert snap["tokens_seen"] == 0
    # a fresh arm records cleanly after reset
    fr.token_begin(0, 1)
    fr.layer_hit(
        0, l_start=0, attn_start=0, attn_end=50, moe_start=50,
        barrier=100, all_hit=110, dispatched=120, l_end=130,
    )
    fr.forward_end(200)
    assert tl.snapshot()["tokens_recorded"] == 1
