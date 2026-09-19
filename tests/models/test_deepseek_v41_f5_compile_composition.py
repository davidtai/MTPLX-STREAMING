"""F5 compile-lever composition with the installed packed decode lane (CPU, tiny).

These gates prove, on the tiny CSA config (no artifact, MLX pinned to CPU), the
composition claims the F5 window rests on -- that the compile levers still drive the
installed ``switch._run`` with the right arguments and compute the shared expert
exactly once:

  * a stand-in switch reproduces the streamed-switch + PackedDecode.run CONTRACT
    (``_run(x, indices, *, shared_work) -> (output, shared)``, ``__call__`` ->
    ``_run(shared_work=None) -> output``, ``run_with_shared_overlap`` ->
    ``_run(shared_work=cb) -> (output, shared)`` -- expert_mlx.py:2509-2531,
    plane_lane.py:216/286);
  * CONTROL / HC_COMPILE / ATTN_COMPILE all route the routed switch through
    MoE.__call__ -> _run_routed_shared_overlap -> run_with_shared_overlap ->
    ``_run(shared_work=cb)``: the lane computes the shared expert ONCE (overlapped);
  * K35 (SMALL_STAGES_FUSED) bypasses MoE.__call__ (DecoderLayer._fused_small_decode)
    and calls ``switch_mlp(xf, indices)`` -> ``_run(shared_work=None)``: the lane does
    NOT compute the shared expert (seg2 already did, via the raw weights), so the K1
    verify_shared_overlap ``shared_work`` is a NO-OP under K35 -- computed once, not
    twice, not never;
  * lever on == off is ``mx.array_equal`` at M in {2,4,6,7,8} for HC/ATTN/K35 (the
    tiny-CPU byte-identical claim; NB the module comment 2769-2772 + the HC screen
    3.7e-4 show this is NOT bit-identity at NATIVE width -- the window classifier is
    what protects that); ATTN_CORE_COMPILE is reported as max|delta| (rounding-class);
  * engagement counters move as claimed per arm; exactly one routing barrier per
    routed-layer call is preserved (K35 adds none);
  * per-layer primitive census at M=6 for each arm (dispatch collapse magnitude);
  * the F5 decode-only enable helper flips the import-bound globals / read-at-use env
    and the SEPARATE caps-at-8 option raises the 7-row caps without touching defaults.

Pins MLX to CPU (worker-tests-must-pin-mlx-cpu.md); tiny random config; no artifact.
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

import mtplx.models.deepseek_v41 as dv41  # noqa: E402
from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402
from mtplx.models.expert_mlx import run_switch_with_shared_overlap  # noqa: E402

# The F5 helper under test.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "deepseek_v41" / "f5_compile"))
import f5_decode_levers  # noqa: E402

_DOT_RECT = re.compile(r'\[label ="([^"]+)", shape=rectangle\]')


def _count_prims(*outs) -> int:
    arrs = [a for a in outs if isinstance(a, mx.array)]
    if not arrs:
        return 0
    buf = io.StringIO()
    mx.export_to_dot(buf, *arrs)
    return len(_DOT_RECT.findall(buf.getvalue()))


def _csa_args(**over) -> ModelArgs:
    base = dict(
        vocab_size=48, hidden_size=32, num_hidden_layers=8,
        num_attention_heads=4, head_dim=16, qk_rope_head_dim=4,
        q_lora_rank=16, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=2,
        index_n_heads=2, index_head_dim=8, index_topk=5,
        sliding_window=8, window_size=8, swiglu_limit=0.5,
        compress_ratios=[0, 0, 2, 2, 2, 1, 1, 1],
        kv_source_layer_ids=[2, 5], index_source_layer_ids=[2, 5, 6],
        candidate_source_layer_id=5, candidate_topk_blocks=3, candidate_block_size=2,
        rope_scaling={"rope_type": "yarn", "factor": 16, "beta_fast": 32,
                      "beta_slow": 1, "original_max_position_embeddings": 65536},
    )
    base.update(over)
    return ModelArgs(**base)


def _randomize(model, seed=0, scale=0.1):
    mx.random.seed(seed)
    new = []
    for name, arr in tree_flatten(model.parameters()):
        if arr.ndim == 1 and ("norm_weight" in name or name.endswith("norm.weight")):
            v = 1.0 + 0.2 * mx.random.normal(arr.shape)
        elif "attn_sink" in name:
            v = 0.5 * mx.random.normal(arr.shape)
        else:
            v = scale * mx.random.normal(arr.shape)
        new.append((name, v.astype(mx.float32)))
    model.update(tree_unflatten(new))
    mx.eval(model.parameters())


def _new_model(seed=1, **over):
    args = _csa_args(**over)
    model = Model(args)
    _randomize(model, seed=seed)
    return model, args


class StandInPackedSwitch:
    """Reproduce the streamed switch + PackedDecode.run contract on the tiny model.

    ``_run(x, indices, *, shared_work)`` -> ``(output, shared)`` mirrors
    plane_lane.PackedDecode.run: it fences ``indices`` once (the per-layer routing
    barrier), delegates the routed compute to the wrapped resident switch, and runs
    ``shared_work`` iff not None.  ``__call__`` and ``run_with_shared_overlap`` are
    the exact expert_mlx.py:2509-2531 wrappers, so ``getattr(sw,
    'run_with_shared_overlap')`` selects the overlap branch in
    run_switch_with_shared_overlap."""

    def __init__(self, inner):
        self.inner = inner
        self.barriers = 0
        self.calls: list[dict] = []   # per _run: {"rows", "shared_work_none"}

    def _run(self, x, indices, *, shared_work):
        self.barriers += 1
        mx.eval(indices)   # the routing barrier (plane_lane.py:219 / expert_mlx.py:2271)
        routed = self.inner(x, indices)
        shared = shared_work() if shared_work is not None else None
        rows = int(x.reshape(-1, x.shape[-1]).shape[0])
        self.calls.append({"rows": rows, "shared_work_none": shared_work is None})
        return routed, shared

    def __call__(self, x, indices):
        output, _ = self._run(x, indices, shared_work=None)
        return output

    def run_with_shared_overlap(self, x, indices, shared_work):
        output, shared = self._run(x, indices, shared_work=shared_work)
        assert shared is not None
        return output, shared


def _install_standins(model, *, overlap=True):
    """Wrap every routed layer's switch with the stand-in and select the retained
    verify_shared_overlap route (deepseek_v41_moe.MoE.install_streamed_shared_route)."""
    stand = {}
    for i, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if not hasattr(mlp, "switch_mlp"):
            continue
        sw = StandInPackedSwitch(mlp.switch_mlp)
        mlp.switch_mlp = sw
        if hasattr(mlp, "install_streamed_shared_route"):
            mlp.install_streamed_shared_route(overlap=overlap)
        stand[i] = sw
    return stand


def _prefill(model, args, s=12, seed=0):
    ids = mx.array(np.random.RandomState(seed).randint(0, args.vocab_size, size=(1, s)))
    cache = model.make_cache()
    mx.eval(model(ids, cache=cache, prefill_chunk=0))
    return cache


def _verify_ids(M, seed):
    return mx.array(np.random.RandomState(seed).randint(0, 48, size=(1, M)))


# --- lever context managers (mirror the K4/K35 unit tests) ------------------
@contextlib.contextmanager
def _hc(flag, max_rows=7):
    of, orr = dv41._HC_COMPILE, dv41._HC_COMPILE_MAX_ROWS
    dv41._HC_COMPILE, dv41._HC_COMPILE_MAX_ROWS = flag, max_rows
    dv41._HC_COMPILED.clear()
    try:
        yield
    finally:
        dv41._HC_COMPILE, dv41._HC_COMPILE_MAX_ROWS = of, orr
        dv41._HC_COMPILED.clear()


@contextlib.contextmanager
def _attn(flag, max_rows=32):
    of, orr = dv41._ATTN_COMPILE, dv41._ATTN_COMPILE_MAX_ROWS
    dv41._ATTN_COMPILE, dv41._ATTN_COMPILE_MAX_ROWS = flag, max_rows
    dv41._ATTN_COMPILED.clear()
    try:
        yield
    finally:
        dv41._ATTN_COMPILE, dv41._ATTN_COMPILE_MAX_ROWS = of, orr
        dv41._ATTN_COMPILED.clear()


@contextlib.contextmanager
def _env_flag(key, flag):
    old = os.environ.get(key)
    if flag:
        os.environ[key] = "1"
    else:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


@contextlib.contextmanager
def _fused(flag, max_rows=7):
    orr = dv41._SMALL_STAGES_MAX_ROWS
    dv41._SMALL_STAGES_MAX_ROWS = max_rows
    dv41._SMALL_STAGES_COMPILED.clear()
    with _env_flag(dv41._SMALL_STAGES_FUSED_ENV, flag):
        try:
            yield
        finally:
            dv41._SMALL_STAGES_MAX_ROWS = orr
            dv41._SMALL_STAGES_COMPILED.clear()


_ARMS = ("control", "hc_compile", "attn_compile", "small_stages")


@contextlib.contextmanager
def _arm(name):
    if name == "control":
        with _hc(False), _attn(False), _fused(False):
            yield
    elif name == "hc_compile":
        with _hc(True), _attn(False), _fused(False):
            yield
    elif name == "attn_compile":
        with _hc(False), _attn(True), _fused(False):
            yield
    elif name == "small_stages":
        with _hc(False), _attn(False), _fused(True):
            yield
    else:  # pragma: no cover
        raise ValueError(name)


# ---------------------------------------------------------------------------
# 1. the packed-lane _run contract is reached with the right shared_work per arm
# ---------------------------------------------------------------------------
def test_switch_run_contract_and_shared_work():
    for M in (2, 4, 6, 7, 8):
        # control / HC / ATTN -> lane gets shared_work (overlap); K35 -> None.
        for arm, expect_none in (("control", False), ("hc_compile", False),
                                 ("attn_compile", False), ("small_stages", None)):
            model, args = _new_model(seed=3)
            stand = _install_standins(model, overlap=True)
            with _arm(arm):
                cache = _prefill(model, args)
                for sw in stand.values():
                    sw.calls.clear()
                    sw.barriers = 0
                mx.eval(model(_verify_ids(M, seed=M), cache=cache))
            # every routed layer called _run exactly once this forward
            for i, sw in stand.items():
                assert len(sw.calls) == 1, (arm, M, i, len(sw.calls))
                assert sw.calls[0]["rows"] == M, (arm, M, sw.calls[0])
                fused_active = (arm == "small_stages") and (M <= 7)
                want_none = fused_active
                assert sw.calls[0]["shared_work_none"] is want_none, (arm, M, i)


# ---------------------------------------------------------------------------
# 2. lever on == off, M in {2,4,6,7,8}.  Per the F5 rule: mx.array_equal where the
#    ledger claims byte-identity (HC/K35 at rows<=cap; M=8 both eager -> equal),
#    else report max|delta| (ATTN_COMPILE is rounding-class even on tiny CPU --
#    ~5-7e-7 here -- and rounding-class at native width per deepseek_v41.py:2769).
# ---------------------------------------------------------------------------
def test_lever_on_equals_off_M2_to_M8():
    attn_deltas = {}
    for M in (2, 4, 6, 7, 8):
        def run(arm):
            model, args = _new_model(seed=2)
            _install_standins(model, overlap=True)
            with _arm(arm):
                cache = _prefill(model, args)
                lo = model(_verify_ids(M, seed=100 + M), cache=cache)
                mx.eval(lo)
                return np.array(lo)
        base = run("control")
        # HC and K35: byte-identical on tiny at the admitted rows (and trivially at
        # M=8, where both fall to the eager body above the 7-row cap).
        for arm in ("hc_compile", "small_stages"):
            got = run(arm)
            assert np.array_equal(base, got), (
                f"{arm} M={M} not byte-identical to control on tiny CPU "
                f"(max|delta|={np.max(np.abs(base - got))})")
        # ATTN_COMPILE: rounding-class -> record max|delta|, assert a tiny band.
        got = run("attn_compile")
        d = float(np.max(np.abs(base - got)))
        attn_deltas[M] = d
        assert np.isfinite(d) and d < 1e-5, (M, d)
    print("ATTN_COMPILE on-vs-off max|delta| by M (rounding-class):", attn_deltas)


# ---------------------------------------------------------------------------
# 3. ATTN_CORE_COMPILE does NOT engage on the tiny CSA config -- its selected-key
#    core tape needs the native (n_heads=64/CSA index_topk) geometry.  Document
#    that here: compiled==0, delta==0 on tiny.  Native engagement is a GPU-window
#    concern; the receipt's attn_core_compile_engagement proves it there.
# ---------------------------------------------------------------------------
def test_attn_core_compile_inert_on_tiny():
    report = {}
    for M in (2, 6, 8):
        def run(flag):
            model, args = _new_model(seed=7)
            _install_standins(model, overlap=True)
            with _hc(False), _attn(False), _fused(False), \
                    _env_flag("MTPLX_DSV41_ATTN_CORE_COMPILE", flag):
                dv41._reset_attn_core_compile_calls()
                cache = _prefill(model, args)
                lo = model(_verify_ids(M, seed=200 + M), cache=cache)
                mx.eval(lo)
                return np.array(lo), dv41._attn_core_compile_calls()
        off, _ = run(False)
        on, calls = run(True)
        report[M] = {"compiled": calls["compiled"], "eager": calls["eager"],
                     "max_abs_delta": float(np.max(np.abs(off - on)))}
    print("ATTN_CORE_COMPILE on tiny (expected inert):", report)
    for M, r in report.items():
        assert r["compiled"] == 0, (M, r)          # tape never fires on tiny
        assert r["max_abs_delta"] == 0.0, (M, r)   # so on==off exactly


# ---------------------------------------------------------------------------
# 4. engagement counters per arm (small_stages fused vs eager) at M=6
# ---------------------------------------------------------------------------
def test_engagement_counters_per_arm_M6():
    model, args = _new_model(seed=8)
    _install_standins(model, overlap=True)
    L = args.num_hidden_layers
    # K35 ON at M=6 -> all layers fused, none eager (one verify forward).
    with _fused(True):
        cache = _prefill(model, args)   # eager (s=12 > cap)
        dv41._reset_small_stages_calls()
        mx.eval(model(_verify_ids(6, seed=6), cache=cache))
        on = dv41._small_stages_calls()
    assert on["fused"] == L and on["eager"] == 0, on
    # K35 OFF -> none fused.
    with _fused(False):
        cache = _prefill(model, args)
        dv41._reset_small_stages_calls()
        mx.eval(model(_verify_ids(6, seed=6), cache=cache))
        off = dv41._small_stages_calls()
    assert off["fused"] == 0 and off["eager"] == L, off
    # M=8 > cap -> K35 forced eager even when armed.
    with _fused(True):
        cache = _prefill(model, args)
        dv41._reset_small_stages_calls()
        mx.eval(model(_verify_ids(8, seed=8), cache=cache))
        big = dv41._small_stages_calls()
    assert big["fused"] == 0 and big["eager"] == L, big


# ---------------------------------------------------------------------------
# 5. exactly one routing barrier per routed-layer call; K35 adds none
# ---------------------------------------------------------------------------
def test_one_routing_barrier_per_layer():
    counts = {}
    for arm in _ARMS:
        model, args = _new_model(seed=9)
        stand = _install_standins(model, overlap=True)
        with _arm(arm):
            cache = _prefill(model, args)
            for sw in stand.values():
                sw.barriers = 0
            mx.eval(model(_verify_ids(6, seed=6), cache=cache))
        total = sum(sw.barriers for sw in stand.values())
        per_layer = {i: sw.barriers for i, sw in stand.items()}
        assert all(b == 1 for b in per_layer.values()), (arm, per_layer)
        counts[arm] = total
    # every arm issues the SAME number of routing barriers (one per routed layer).
    assert len(set(counts.values())) == 1, counts


# ---------------------------------------------------------------------------
# 6. shared expert computed EXACTLY once (not twice, not never), K35 vs control
# ---------------------------------------------------------------------------
class _CountingShared:
    """Count module-level shared-expert calls while forwarding w1/w3/w2/etc."""

    def __init__(self, inner):
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "calls", 0)

    def __call__(self, x):
        object.__setattr__(self, "calls", self.calls + 1)
        return self.inner(x)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "inner"), name)


def test_shared_expert_computed_exactly_once():
    def run(arm):
        model, args = _new_model(seed=4)
        stand = _install_standins(model, overlap=True)
        shared_counters = []
        for layer in model.model.layers:
            mlp = layer.mlp
            if hasattr(mlp, "shared_experts"):
                cs = _CountingShared(mlp.shared_experts)
                mlp.shared_experts = cs
                shared_counters.append(cs)
        with _arm(arm):
            cache = _prefill(model, args)
            for cs in shared_counters:
                object.__setattr__(cs, "calls", 0)
            for sw in stand.values():
                sw.calls.clear()
            lo = model(_verify_ids(6, seed=6), cache=cache)
            mx.eval(lo)
        module_shared_calls = sum(cs.calls for cs in shared_counters)
        lane_computed_shared = sum(
            0 if sw.calls[0]["shared_work_none"] else 1 for sw in stand.values())
        return np.array(lo), module_shared_calls, lane_computed_shared, len(stand)

    ctrl, ctrl_mod, ctrl_lane, n = run("control")
    k35, k35_mod, k35_lane, _ = run("small_stages")

    # control: the lane computes the shared expert once per layer (module call via
    # the overlap shared_work lambda); K35: the lane computes 0 (shared_work=None),
    # seg2's fused tape computes it via the raw weights (0 module __call__s).
    assert ctrl_lane == n and ctrl_mod == n, (ctrl_lane, ctrl_mod, n)
    assert k35_lane == 0 and k35_mod == 0, (k35_lane, k35_mod)
    # ...and the outputs are identical -> under K35 the shared expert was computed
    # exactly ONCE and correctly (a double-add or a missing add would diverge).
    assert np.array_equal(ctrl, k35), (
        f"K35 output != control (max|delta|={np.max(np.abs(ctrl - k35))}) -- "
        "shared expert double-counted or dropped")


# ---------------------------------------------------------------------------
# 7. per-layer primitive census at M=6 for each arm (dispatch collapse)
# ---------------------------------------------------------------------------
def test_primitive_census_M6_per_arm():
    from mtplx.models.deepseek_v41 import (
        _hc_attn_prep_impl, _small_compiled,
    )
    model, args = _new_model(seed=4)
    L = next(l for l in model.model.layers if l.attn.mode == dv41.MODE_FULL)
    hc, dim = L.hc_mult, args.hidden_size
    consts = (hc, L.hc_iters, L.norm_eps, L.hc_eps)
    mx.random.seed(0)
    # M=6 HC stream.
    h = (0.3 * mx.random.normal((1, 6, hc, dim))).astype(mx.float32)
    pre_mix = mx.concatenate(
        [mx.ones((1, 6, 1)), mx.zeros((1, 6, hc - 1))], axis=-1).astype(mx.float32)
    mx.eval(h, pre_mix)
    census = {}
    # eager HC attn-prep vs K4 compiled tape vs K35 seg1.
    eager = _hc_attn_prep_impl(h, pre_mix, L.hc_attn_fn, L.hc_attn_base,
                               L.hc_attn_scale, L.attn_norm_weight, *consts)
    census["eager_attn_prep"] = _count_prims(*eager)
    with _hc(True):
        hc_comp = dv41._hc_compiled(
            "attn_prep", hc, L.hc_iters, L.norm_eps, L.hc_eps,
            dv41._sinkhorn_use_kernel(), dv41._hc_premix_use_kernel())(
            h, pre_mix, L.hc_attn_fn, L.hc_attn_base, L.hc_attn_scale, L.attn_norm_weight)
        census["hc_compile_attn_prep"] = _count_prims(*hc_comp)
    with _fused(True):
        seg1 = _small_compiled("seg1", L)(
            h, pre_mix, L.hc_attn_fn, L.hc_attn_base, L.hc_attn_scale, L.attn_norm_weight)
        census["small_stages_seg1"] = _count_prims(*seg1)
        mx.eval(seg1)
    mx.eval(eager, hc_comp)
    print("M=6 per-layer attn-prep primitive census:", census)
    # each compiled arm collapses the eager prep chain.
    assert census["hc_compile_attn_prep"] < census["eager_attn_prep"], census
    assert census["small_stages_seg1"] < census["eager_attn_prep"], census


# ---------------------------------------------------------------------------
# 8. F5 decode-only enable helper: import-bound globals + read-at-use env
# ---------------------------------------------------------------------------
def test_f5_enable_decode_levers_mechanism():
    # start from the retained "all off" state.
    with _hc(False), _attn(False), _fused(False), \
            _env_flag("MTPLX_DSV41_ATTN_CORE_COMPILE", False), \
            _env_flag("MTPLX_DSV41_HC_PREMIX_KERNEL", False):
        assert dv41._HC_COMPILE is False and dv41._ATTN_COMPILE is False
        rep = f5_decode_levers.enable_decode_levers(
            dv41, "hc_compile,attn_compile,small_stages,attn_core_compile")
        try:
            # import-bound levers -> module globals flipped True
            assert dv41._HC_COMPILE is True and dv41._ATTN_COMPILE is True
            assert rep["applied"]["hc_compile"]["mechanism"] == "module_global"
            # read-at-use levers -> env set
            assert os.environ.get("MTPLX_DSV41_SMALL_STAGES_FUSED") == "1"
            assert os.environ.get("MTPLX_DSV41_ATTN_CORE_COMPILE") == "1"
            assert dv41._small_stages_fused_enabled() is True
            assert rep["applied"]["small_stages"]["mechanism"] == "env"
            assert rep["caps_at_8"] is False
        finally:
            # restore globals the context managers do not own
            dv41._HC_COMPILE = False
            dv41._ATTN_COMPILE = False
            os.environ.pop("MTPLX_DSV41_SMALL_STAGES_FUSED", None)
            os.environ.pop("MTPLX_DSV41_ATTN_CORE_COMPILE", None)


# ---------------------------------------------------------------------------
# 9. caps-at-8 SEPARATE option raises the 7-row caps and never touches defaults
# ---------------------------------------------------------------------------
def test_f5_caps_at_8_option():
    assert dv41._HC_COMPILE_MAX_ROWS == 7 and dv41._SMALL_STAGES_MAX_ROWS == 7, \
        "shipped defaults must be 7 before the option runs"
    hc0, ss0 = dv41._HC_COMPILE_MAX_ROWS, dv41._SMALL_STAGES_MAX_ROWS
    try:
        rep = f5_decode_levers.enable_decode_levers(dv41, "hc_compile", caps_at_8=True)
        assert dv41._HC_COMPILE_MAX_ROWS == 8 and dv41._SMALL_STAGES_MAX_ROWS == 8
        assert rep["caps_at_8"] is True
        assert rep["caps_before"] == {"hc_compile_max_rows": 7, "small_stages_max_rows": 7}
        assert rep["caps_after"] == {"hc_compile_max_rows": 8, "small_stages_max_rows": 8}
        assert "rounding-class" in rep["caps_at_8_exactness"]
    finally:
        dv41._HC_COMPILE_MAX_ROWS, dv41._SMALL_STAGES_MAX_ROWS = hc0, ss0
        dv41._HC_COMPILE = False
        dv41._HC_COMPILED.clear()
        dv41._SMALL_STAGES_COMPILED.clear()
    # defaults restored / untouched for the rest of the suite.
    assert dv41._HC_COMPILE_MAX_ROWS == 7 and dv41._SMALL_STAGES_MAX_ROWS == 7
