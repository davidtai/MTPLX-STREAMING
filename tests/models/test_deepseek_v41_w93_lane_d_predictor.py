"""W93 lane D -- gate-oracle PREDICTOR hardening (adversarial-review LOW-1/2/3).

Locks the four lane-D fixes on the DSV4.1 one-layer-ahead gate-oracle prefetch
(docs/deepseek-v41/W93_GATE_PREFETCH.md).  Everything here is CPU-pinned, uses a
tiny random gate / tiny model / tiny synthetic streamed artifact, and never
touches the GPU.  Run under ``nice -n 19`` and without ``pytest -n auto``.

  LOW-1  ``gate_predict_topk`` scores through the SAME (compiled-or-eager) prefix
         the real router runs -- so the predicted top-k SET is the router's own
         selection exactly (a'=1.0), in both compile regimes, and the compile-
         armed path rides the router's SHARED ``_gate_prefix`` tape (no standalone
         f32 gate-weight copy per layer per token).
  LOW-2  the DecoderLayer scores the bf16-ROUNDED ``layer_in`` (bit-identical to
         W89's stored trace tensor via ``bf16_bits``), not the un-rounded f32
         mean -- so the measured missRed@10 0.736 applies to the tensor scored.
  LOW-3  the pending stash rides a plain ``_GatePrefetchLink`` holder, so the
         predicted-id array never registers in the streamed switch's nn.Module
         dict (a raw tuple would, under mlx 0.32.2 ``Module.__setattr__``); proven
         stash->clear->stash on ONE ``HotExpertSwitchGLU``/link instance.
  guard  ``install_gate_prefetch_links`` skips ``DenseIslandSwitchGLU`` sources
         (a resident-island source has no ``mx.eval(indices)`` barrier to ride and
         never consumes the stash).

Plus a 64-decode-step byte-identity check where the prefetch ring ACTUALLY serves
hits, on the real streamed runtime (switch level; see the model-level note at the
foot of the file for the piece that needs the runner rebuild / consumer merge).
"""

from __future__ import annotations

import types

import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)

import pytest  # noqa: E402

import mtplx.models.deepseek_v41 as dv  # noqa: E402
from mtplx.models.deepseek_v41_moe import (  # noqa: E402
    Gate,
    gate_predict_topk,
    _gate_prefix_impl,
)
from mlx.utils import tree_flatten  # noqa: E402

# Tiny-model + fake-runtime helpers (underscore names -> not re-collected here).
from tests.models.test_deepseek_v41_w93_gate_prefetch_predict import (  # noqa: E402
    _new_model,
    _fake_runtime,
)

FLAG = "MTPLX_DSV41_GATE_PREFETCH"
MIN_LAYER_FLAG = "MTPLX_DSV41_GATE_PREFETCH_MIN_LAYER"


@pytest.fixture(autouse=True)
def _cpu_and_clean_compile():
    """Pin CPU and isolate the shared attn-compile flags/cache per test."""
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    saved = (dv._ATTN_COMPILE, dv._ATTN_COMPILE_MAX_ROWS, dict(dv._ATTN_COMPILED))
    try:
        yield
    finally:
        dv._ATTN_COMPILE, dv._ATTN_COMPILE_MAX_ROWS, comp = saved
        dv._ATTN_COMPILED.clear()
        dv._ATTN_COMPILED.update(comp)
        mx.set_default_device(prev)


def _make_gate(dim, n_routed, topk, *, seed=0, temp=1.3, sf="sqrtsoftplus") -> Gate:
    """A tiny random DSV4.1 ``Gate`` with bf16 weight + f32 correction bias (the
    artifact dtypes), constructed off a ``SimpleNamespace`` of the args it reads."""
    args = types.SimpleNamespace(
        hidden_size=dim, num_experts_per_tok=topk, scoring_func=sf,
        gate_temp=temp, norm_topk_prob=True, routed_scaling_factor=1.5,
        n_routed_experts=n_routed,
    )
    g = Gate(0, args)
    mx.random.seed(seed)
    g.weight = (0.1 * mx.random.normal((n_routed, dim))).astype(mx.bfloat16)
    g.e_score_correction_bias = (0.05 * mx.random.normal((n_routed,))).astype(mx.float32)
    mx.eval(g.weight, g.e_score_correction_bias)
    return g


def _sets(idx: mx.array):
    mx.eval(idx)
    return [set(row) for row in idx.tolist()]


# ---------------------------------------------------------------------------
# LOW-1: a' = 1.0 -- the predictor reproduces the router's OWN selection exactly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("compile_on", [False, True])
def test_predictor_matches_router_selection_aprime_one(compile_on) -> None:
    """``gate_predict_topk(gate, x, gate.topk)`` == the router's ``indices`` SET,
    row-for-row, for 32 random rows -- in BOTH the eager and compile-armed
    regimes.  This is the a'=1.0 alignment check: because scoring goes through the
    same prefix the router runs, the sets are bit-identical, not merely byte-equal
    via the K22 compiled==eager claim."""
    dv._ATTN_COMPILED.clear()
    dv._ATTN_COMPILE = bool(compile_on)
    dv._ATTN_COMPILE_MAX_ROWS = 4096
    gate = _make_gate(32, 8, 2, seed=3)
    mx.random.seed(11)
    x = mx.random.normal((32, 32)).astype(mx.float32)

    _weights, router_idx = gate(x)                        # the real router
    pred = gate_predict_topk(gate, x, gate.topk)          # predictor at router k
    router_sets, pred_sets = _sets(router_idx), _sets(pred)
    assert pred_sets == router_sets, (
        f"compile_on={compile_on}: predicted top-{gate.topk} set != router's own "
        f"selection (a' != 1.0)"
    )

    # a WIDER prefetch width always CONTAINS the router's true route (superset),
    # so a hit warms exactly the experts the demand route reads.
    wide = gate_predict_topk(gate, x, 5)
    wide_sets = _sets(wide)
    assert all(r.issubset(w) for r, w in zip(router_sets, wide_sets))


def test_gate_predict_topk_rides_shared_compiled_tape() -> None:
    """LOW-1: with ATTN_COMPILE armed, the predictor scores through the router's
    SHARED ``_gate_prefix`` compiled tape (same ``(gate_prefix, score_func,
    gate_temp)`` cache key) instead of the old unconditional eager
    ``_gate_prefix_impl`` that materialised a [n_routed, dim] f32 gate-weight copy
    per call.  The compiled selection is byte-identical to the eager one."""
    gate = _make_gate(32, 8, 3, seed=5)
    mx.random.seed(1)
    x = mx.random.normal((4, 32)).astype(mx.float32)
    key = ("gate_prefix", str(gate.score_func), float(gate.gate_temp))

    # eager selection (flag off) -- also the reference for byte-identity.
    dv._ATTN_COMPILED.clear()
    dv._ATTN_COMPILE = False
    eager = gate_predict_topk(gate, x, 3)
    assert key not in dv._ATTN_COMPILED, "eager path must not build a tape"

    # compile-armed: the predictor now rides (and populates) the shared tape.
    dv._ATTN_COMPILE = True
    dv._ATTN_COMPILE_MAX_ROWS = 4096
    compiled = gate_predict_topk(gate, x, 3)
    assert key in dv._ATTN_COMPILED, "predictor did not ride the shared _gate_prefix tape"
    # the router keys the SAME tape (proves it is shared, not a private one).
    _w, _idx = gate(x)
    assert key in dv._ATTN_COMPILED
    assert _sets(compiled) == _sets(eager), "compiled selection != eager selection"


# ---------------------------------------------------------------------------
# LOW-2: the DecoderLayer scores the bf16-rounded layer_in, not the f32 mean
# ---------------------------------------------------------------------------
def test_stash_scores_bf16_rounded_layer_in(monkeypatch) -> None:
    """On a residual whose collapsed mean is a row where bf16-rounding CHANGES
    the top-k selection, the DecoderLayer's stashed prediction follows the
    bf16-rounded route (W89's stored ``bf16_bits`` trace tensor) -- NOT the
    un-rounded f32 route.  Reverting the round would flip this assertion."""
    monkeypatch.setenv(FLAG, "3")
    k = 3
    gate = _make_gate(32, 8, k, seed=1)

    # find a row where f32 vs bf16-rounded top-k diverge, verified SINGLE-row.
    mx.random.seed(7)
    rows = mx.random.normal((2048, 32)).astype(mx.float32)
    pf_b = _sets(gate_predict_topk(gate, rows, k))
    pb_b = _sets(gate_predict_topk(gate, rows.astype(mx.bfloat16).astype(mx.float32), k))
    row = None
    for i in range(rows.shape[0]):
        if pf_b[i] == pb_b[i]:
            continue
        x1 = rows[i : i + 1]
        f32_set = set(gate_predict_topk(gate, x1, k).reshape(-1).tolist())
        bf16_set = set(
            gate_predict_topk(gate, x1.astype(mx.bfloat16).astype(mx.float32), k)
            .reshape(-1).tolist()
        )
        if f32_set != bf16_set:
            row, f32_route, bf16_route = i, f32_set, bf16_set
            break
    assert row is not None, "no bf16-rounding-sensitive row found (calibration)"

    # residual entering L-1: [B=1, T=1, hc=1, dim]; mean over the hc axis == x1.
    x1 = rows[row : row + 1]
    h = x1.reshape(1, 1, 1, 32).astype(mx.float32)

    # drive the real DecoderLayer stash through a minimal `self` (it reads only
    # `self.mlp`), with a fake runtime that has a prefetch ring armed.
    link = dv._GatePrefetchLink(next_layer=5, next_gate=gate)
    switch = types.SimpleNamespace(
        runtime=types.SimpleNamespace(
            config=types.SimpleNamespace(prefetch_slots=10)
        )
    )
    mlp = types.SimpleNamespace(_mtplx_gate_prefetch_next=link, switch_mlp=switch)
    fake_self = types.SimpleNamespace(mlp=mlp)

    dv.DecoderLayer._maybe_stash_gate_prefetch(fake_self, h)
    pending = switch._mtplx_gate_prefetch_pending
    assert pending is link
    stashed = set(pending[1].reshape(-1).tolist())
    assert stashed == bf16_route, "predictor did not score the bf16-rounded layer_in"
    assert stashed != f32_route, "predictor scored the un-rounded f32 mean (LOW-2)"


# ---------------------------------------------------------------------------
# LOW-3: the stash never pollutes the switch nn.Module dict, across tokens
# ---------------------------------------------------------------------------
def test_stash_on_link_never_pollutes_switch_dict(tmp_path) -> None:
    """stash -> clear -> stash on ONE ``HotExpertSwitchGLU``/link instance.  The
    switch holds only a plain-object reference (``__setattr__`` else branch), so
    ``"..._pending" in switch`` (module-dict membership) stays False every token,
    while the base consumer's read (``pending[1]`` / unpack) still works.  The
    contrast block shows the OLD raw-tuple pattern WOULD register the id array."""
    from tests.test_deepseek_v41_w93_gate_prefetch_reconcile import (
        _open_runtime,
        _switch,
        _layer,
    )

    rt, spec = _open_runtime(tmp_path, expert_count=8, top_k=2, prefetch=10)
    key = "_mtplx_gate_prefetch_pending"
    try:
        sw = _switch(rt, spec)  # ONE real HotExpertSwitchGLU, reused below
        link = dv._GatePrefetchLink(_layer(spec), next_gate=None)

        # --- contrast: the OLD raw-(int, mx.array)-tuple stash pollutes the dict.
        sw._mtplx_gate_prefetch_pending = (_layer(spec), mx.array([[4, 5]], dtype=mx.int32))
        assert key in sw, "expected the raw-tuple pattern to register in the module dict"
        sw._mtplx_gate_prefetch_pending = None  # clear pops it (the None branch)
        assert key not in sw

        # --- the fix: three tokens' worth of stash/clear on the same instance.
        for token, ids in enumerate(
            [
                mx.array([[4, 5]], dtype=mx.int32),
                mx.array([[6, 7]], dtype=mx.int32),
                mx.array([[1, 2]], dtype=mx.int32),
            ]
        ):
            link.pending_ids = ids
            sw._mtplx_gate_prefetch_pending = link                     # stash
            assert key not in sw, f"token {token}: stash registered in switch dict"
            assert not any(
                key in nm for nm, _ in tree_flatten(sw.parameters())
            ), f"token {token}: stash leaked into switch.parameters()"

            got = sw._mtplx_gate_prefetch_pending                      # consumer read
            assert got is link
            assert got[0] == link.next_layer                           # tuple-compat [0]
            assert set(got[1].reshape(-1).tolist()) == set(ids.reshape(-1).tolist())
            unpacked_layer, unpacked_ids = got                         # 2-unpack
            assert unpacked_layer == link.next_layer and unpacked_ids is ids

            sw._mtplx_gate_prefetch_pending = None                     # clear
            assert getattr(sw, key, None) is None
            assert key not in sw, f"token {token}: clear left a dict entry"
    finally:
        rt.close()


# ---------------------------------------------------------------------------
# install guard: only streamed sources are linked (skip DenseIslandSwitchGLU)
# ---------------------------------------------------------------------------
def test_install_skips_dense_island_source(monkeypatch) -> None:
    """A source layer whose ``switch_mlp`` is a ``DenseIslandSwitchGLU`` is not
    linked (no ``mx.eval(indices)`` barrier to ride, never consumes the stash),
    while its streamed neighbours are."""
    from mtplx.models.expert_mlx import DenseIslandSwitchGLU

    class _FakeIslandSwitch(DenseIslandSwitchGLU):  # valid empty Module, right type
        def __init__(self):
            nn.Module.__init__(self)

    monkeypatch.delenv(MIN_LAYER_FLAG, raising=False)  # default 4 -> sources {3,4,5}
    model, _ = _new_model(num_hidden_layers=8)
    # make layer 4 (a normal source) a dense-island source; 3 and 5 stay streamed.
    model.model.layers[4].mlp.switch_mlp = _FakeIslandSwitch()

    rt = _fake_runtime(10, model.model.layers)
    installed = dv.install_gate_prefetch_links(model, rt)

    linked = {
        i
        for i, layer in enumerate(model.model.layers)
        if getattr(layer.mlp, "_mtplx_gate_prefetch_next", None) is not None
    }
    assert linked == {3, 5}, f"dense-island source 4 must be skipped; got {linked}"
    assert installed == 2


# ---------------------------------------------------------------------------
# 64-decode-step byte-identity WHERE THE RING ACTUALLY SERVES HITS (switch level)
# ---------------------------------------------------------------------------
def test_switch_decode_64_steps_ring_served_byte_identical(tmp_path) -> None:
    """Drive 64 successive M=1 (decode-shaped) routes of cold experts through the
    real ``HotExpertSwitchGLU``.  In the WARMED run each step's route is
    pre-warmed via the runtime's public ``prefetch_experts`` API (so the true
    route consumes ring hits); the COLD run serves the identical routes on demand.
    Every step's gather is byte-identical and the ring genuinely serves hits over
    the run -- the prefetch changes no value, only where the bytes came from."""
    from tests.test_deepseek_v41_w93_gate_prefetch_reconcile import (
        _open_runtime,
        _switch,
        _layer,
        _inputs,
        _route_once,
        _settle_prefetch,
        _REAL_EVAL,
    )

    steps = 64
    cold = [2, 3, 4, 5, 6, 7]  # resident_slots=2 warms 0,1 -> these stay cold

    def run(warm: bool):
        sub = tmp_path / ("warm" if warm else "cold")
        rt, spec = _open_runtime(
            sub, expert_count=8, top_k=2, resident_slots=2, transient=2, prefetch=10
        )
        outs = []
        try:
            _route_once(rt, spec, [0, 1])
            hits0 = rt.counters.prefetch_hit_on_true_route
            for s in range(steps):
                pair = [cold[(2 * s) % len(cold)], cold[(2 * s + 1) % len(cold)]]
                x, idx = _inputs(1, spec.top_k, spec.hidden_size, pair)
                if warm:
                    rt.prefetch_experts(_layer(spec), pair)  # public prefetch API
                    _settle_prefetch(rt)
                out = _switch(rt, spec)(x, idx)              # gather on the TRUE route
                _REAL_EVAL(out)
                rt.flush_deferred_slot_releases(evaluate=True)
                outs.append(out)
            return outs, rt.counters.prefetch_hit_on_true_route - hits0
        finally:
            rt.close()

    cold_outs, _ = run(warm=False)
    warm_outs, ring_hits = run(warm=True)

    assert len(cold_outs) == len(warm_outs) == steps
    for s, (a, b) in enumerate(zip(cold_outs, warm_outs)):
        assert mx.array_equal(a, b), f"step {s}: ring-served gather != demand gather"
    assert ring_hits > 0, "the ring never served a hit over the 64-step run"


# ---------------------------------------------------------------------------
# MODEL-LEVEL ring-served variant -- needs the runner rebuild + consumer merge
# ---------------------------------------------------------------------------
@pytest.mark.skip(
    reason=(
        "MERGE/INFRA DEPENDENCY: driving the tiny DSV4.1 *model's* 64-step decode "
        "through the real HotExpertSwitchGLU (so the stash is consumed on the "
        "switch's own mx.eval(indices) barrier and the ring serves hits INSIDE a "
        "full forward) requires (a) the tiny model wired to a real "
        "ExpertStreamingRuntime via bind_streamed_switches on a streamed artifact "
        "-- runner/loader infra outside lane D's file ownership -- and (b) the "
        "expert_mlx.py consume site (lane B/C). Until then, the ring-serves-hits "
        "byte-identity property is proven at the switch level by "
        "test_switch_decode_64_steps_ring_served_byte_identical, and the model-"
        "level predictor byte-identity on/off by the predict suite's "
        "test_ar_decode_byte_identical_flag_on_off. The assertion that needs the "
        "merge: prefetch_hit_on_true_route > 0 accrued during model(...) decode "
        "with logits mx.array_equal to the flag-off run."
    )
)
def test_model_decode_ring_served_byte_identical_NEEDS_MERGE() -> None:  # pragma: no cover
    raise AssertionError("skipped: see reason (runner rebuild + consumer merge)")
