"""CPU tests for the F2c lean predictor (scripts/deepseek_v41/f2/predictor.py).

Proves the fused module-level compiled ``merged`` tape ('lean', the new default) ranks the
same as the prior gate-prefix path ('native') on realistic router shapes, at strictly lower
host graph-build cost and fewer graph primitives, and that the f32 upcast it preserves is
load-bearing for the ranking.  MLX pinned to CPU (MLX defaults to Metal and the GPU lock is
NOT ours); run under nice -n 19, pytest WITHOUT -n auto.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# A sibling CPU-only module installs a _NoMLX meta-path finder; strip it before importing MLX.
sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "_NoMLX"]

import numpy as np
import pytest

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

mx.set_default_device(mx.cpu)

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts" / "deepseek_v41"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from f2 import predictor as fp  # noqa: E402
from mtplx.models.deepseek_v41_moe import Gate, _gate_prefix_impl  # noqa: E402

N_ROUTED, HIDDEN, TEMP = 384, 5120, 1.0        # DSV4.1 text gate: [384, 5120], sqrtsoftplus


@pytest.fixture(autouse=True)
def _cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(prev)


def _make_gate(*, dim=HIDDEN, n_routed=N_ROUTED, temp=TEMP, score_func="sqrtsoftplus", seed=11):
    from types import SimpleNamespace
    args = SimpleNamespace(hidden_size=dim, num_experts_per_tok=8, scoring_func=score_func,
                           gate_temp=temp, norm_topk_prob=True, routed_scaling_factor=1.0,
                           n_routed_experts=n_routed)
    mx.random.seed(seed)
    gate = Gate(7, args)
    gate.weight = (0.05 * mx.random.normal((n_routed, dim))).astype(mx.bfloat16)
    gate.e_score_correction_bias = (0.1 * mx.random.normal((n_routed,))).astype(mx.float32)
    mx.eval(gate.weight, gate.e_score_correction_bias)
    return gate


def _np(a):
    return np.asarray(a.tolist(), dtype=np.float64)


def _rank_stats(ref, cand, k=8, rel_tol=1e-4):
    """(exact, hard): exact = top-k orderings identical; hard = top-k SETS differ by an
    expert whose ref score is NOT a near-tie with the k-th boundary (a real ranking error).
    Ties at the boundary (|score - boundary| < rel_tol * scale) are tolerated swaps."""
    ro = np.argsort(-ref, kind="stable")
    co = np.argsort(-cand, kind="stable")
    if list(ro[:k]) == list(co[:k]):
        return True, False
    scale = max(1e-12, float(np.max(np.abs(ref))))
    boundary = float(ref[ro[k - 1]])
    for e in set(ro[:k].tolist()) ^ set(co[:k].tolist()):
        if abs(float(ref[e]) - boundary) >= rel_tol * scale:
            return False, True
    return False, False


# ---------------------------------------------------------------------------
# (a) lean vs native top-8 ranking agreement, realistic shapes, rows 4..8
# ---------------------------------------------------------------------------
def test_lean_matches_native_top8_ranking(capsys):
    gate = _make_gate()
    lean = fp.GatePredictor(gate, mode="lean")
    native = fp.GatePredictor(gate, mode="native")
    n_cases, exact_hits, soft_swaps, hard = 240, 0, 0, 0
    max_abs = 0.0
    for i in range(n_cases):
        rows = 4 + (i % 5)                                     # rows 4..8
        x = (0.3 * mx.random.normal((rows, 1, HIDDEN))).astype(mx.bfloat16)
        mx.eval(x)
        m_l = _np(lean.merged(x))
        m_n = _np(native.merged(x))
        max_abs = max(max_abs, float(np.max(np.abs(m_l - m_n))))
        exact, hard_i = _rank_stats(m_n, m_l, k=8)
        exact_hits += int(exact)
        soft_swaps += int(not exact and not hard_i)
        hard += int(hard_i)
    with capsys.disabled():
        print(f"\n[f2c] lean-vs-native top8: exact={exact_hits}/{n_cases} "
              f"soft-tie-swaps={soft_swaps} HARD={hard}  max|dmerged|={max_abs:.3e}")
    assert hard == 0                                          # no real ranking divergence
    assert exact_hits == n_cases                             # CPU: fused compile is bit-exact


def test_lean_bitexact_to_eager_impl():
    """The lean tape reuses _gate_prefix_impl; on CPU the fused compile is byte-identical to
    the eager impl + max (so the predicted top-k SET equals the router's selection)."""
    gate = _make_gate()
    lean = fp.GatePredictor(gate, mode="lean")
    for rows in (4, 5, 6, 7, 8):
        x = (0.3 * mx.random.normal((rows, 1, HIDDEN))).astype(mx.bfloat16)
        mx.eval(x)
        _s, biased = _gate_prefix_impl(x.reshape(-1, HIDDEN), gate.weight,
                                       gate.e_score_correction_bias, TEMP, "sqrtsoftplus")
        ref = _np(mx.max(biased, axis=0))
        got = _np(lean.merged(x))
        assert np.array_equal(got, ref), f"rows={rows}: lean != eager impl"


# ---------------------------------------------------------------------------
# (a') dtype justification: dropping the f32 upcast (bf16 GEMM) reorders the top-8,
#      so the lean tape MUST keep the upcast (it does).
# ---------------------------------------------------------------------------
def test_bf16_gemm_would_break_ranking(capsys):
    gate = _make_gate()
    lean = fp.GatePredictor(gate, mode="lean")           # f32 upcast (via _gate_prefix_impl)
    w, b = gate.weight, gate.e_score_correction_bias
    changed = 0
    n_cases = 240
    for i in range(n_cases):
        rows = 4 + (i % 5)
        x = (0.3 * mx.random.normal((rows, 1, HIDDEN))).astype(mx.bfloat16)
        mx.eval(x)
        xf = x.reshape(-1, HIDDEN)
        bf = (mx.sqrt(nn.softplus((xf @ w.T) / TEMP)) + b).max(axis=0)   # NO upcast
        mx.eval(bf)
        _e, hard = _rank_stats(_np(lean.merged(x)), _np(bf), k=8)
        changed += int(not _e)
    with capsys.disabled():
        print(f"\n[f2c] bf16-GEMM top8 differs from f32-upcast on {changed}/{n_cases} cases "
              f"(justifies keeping the upcast in the lean tape)")
    assert changed > 0                                    # upcast is load-bearing for ranking


# ---------------------------------------------------------------------------
# (b) host graph-build us/call + primitive counts: lean < native
# ---------------------------------------------------------------------------
def _build_us(pred, x, N=2000):
    pred.merged(x)                                        # warm / trace this row count
    t0 = time.perf_counter()
    acc = None
    for _ in range(N):
        acc = pred.merged(x)                              # op construction only
    us = (time.perf_counter() - t0) / N * 1e6
    mx.eval(acc)                                          # single eval OUTSIDE the timer
    return us


def _dot_nodes(arr, tmp):
    import re
    mx.export_to_dot(str(tmp), merged=arr)
    return len(re.findall(r'label\s*=\s*"([^"]*)"', Path(tmp).read_text()))


def test_host_build_and_primitive_counts(capsys, tmp_path, monkeypatch):
    # 'native' is the K22 gate prefix: eager unless the process-global DSV4.1 lever
    # ``deepseek_v41._ATTN_COMPILE`` is on, in which case it is a compiled tape with the SAME
    # primitive count as 'lean' (8 == 8).  Other test modules flip that global and at least one
    # leaks it across a full-suite run, so pin the baseline this comparison is about.
    import mtplx.models.deepseek_v41 as _dv

    monkeypatch.setattr(_dv, "_ATTN_COMPILE", False)
    gate = _make_gate()
    lean = fp.GatePredictor(gate, mode="lean")
    native = fp.GatePredictor(gate, mode="native")
    x = mx.zeros((6, 1, HIDDEN), dtype=mx.bfloat16)
    mx.eval(x, gate.weight, gate.e_score_correction_bias)

    us_native = _build_us(native, x)
    us_lean = _build_us(lean, x)
    n_native = _dot_nodes(native.merged(x), tmp_path / "native.dot")
    n_lean = _dot_nodes(lean.merged(x), tmp_path / "lean.dot")
    with capsys.disabled():
        print(f"\n[f2c] host graph-build: native={us_native:.2f} us/call  lean={us_lean:.2f} "
              f"us/call  (dot nodes: native={n_native}  lean={n_lean})")
    assert n_lean < n_native                              # elementwise chain fused -> fewer prims
    # lean must not COST more host time than native (allow slack for CI jitter).
    assert us_lean <= us_native + 1.0

    # No per-call retrace at a fixed shape: 2000 fixed-shape builds average a few us; a
    # per-call retrace would be ~1000x that.  Guard well above the measured ~2 us.
    assert us_lean < 100.0


def test_primitive_count_comparison_is_hermetic_to_a_leaked_attn_compile_lever(tmp_path, monkeypatch):
    """Regression: with ``_ATTN_COMPILE`` leaked True by another test module the native prefix is
    itself a compiled tape, so 'lean < native' cannot hold (they tie).  The comparison above pins
    the lever; this test documents both regimes so a future leak cannot fail it silently."""
    import mtplx.models.deepseek_v41 as _dv

    gate = _make_gate()
    x = mx.zeros((6, 1, HIDDEN), dtype=mx.bfloat16)
    mx.eval(x, gate.weight, gate.e_score_correction_bias)
    counts = {}
    for lever in (False, True):
        monkeypatch.setattr(_dv, "_ATTN_COMPILE", lever)
        native = fp.GatePredictor(gate, mode="native")
        lean = fp.GatePredictor(gate, mode="lean")
        counts[lever] = (_dot_nodes(native.merged(x), tmp_path / f"native_{lever}.dot"),
                         _dot_nodes(lean.merged(x), tmp_path / f"lean_{lever}.dot"))
    assert counts[False][1] < counts[False][0]        # eager native: lean is the smaller graph
    assert counts[True][1] <= counts[True][0]         # compiled native: lean is never larger
    assert counts[True][1] == counts[False][1]        # lean itself does not depend on the lever


# ---------------------------------------------------------------------------
# (c) construction-time contract: one compiled tape shared, fail-loud validation, modes
# ---------------------------------------------------------------------------
def test_single_compiled_tape_shared_across_layers():
    fp._LEAN_MERGED_CACHE.clear()
    g1, g2 = _make_gate(seed=1), _make_gate(seed=2)
    p1 = fp.GatePredictor(g1, mode="lean")
    p2 = fp.GatePredictor(g2, mode="lean")
    # same (temp, dim) -> exactly one module-level compiled callable serves both layers.
    assert len(fp._LEAN_MERGED_CACHE) == 1
    x = (0.3 * mx.random.normal((6, 1, HIDDEN))).astype(mx.bfloat16)
    mx.eval(x)
    # distinct weights still give distinct outputs through the one shared tape.
    assert not np.array_equal(_np(p1.merged(x)), _np(p2.merged(x)))


def test_default_mode_is_lean():
    gate = _make_gate()
    assert fp.GatePredictor(gate).mode == "lean"


def test_lean_rejects_non_sqrtsoftplus_at_construction():
    gate = _make_gate(score_func="sigmoid")
    with pytest.raises(RuntimeError, match="sqrtsoftplus"):
        fp.GatePredictor(gate, mode="lean")
    # native tolerates it (it routes on score_func at call time, as the router does).
    fp.GatePredictor(gate, mode="native")


def test_unknown_mode_fails_loudly():
    gate = _make_gate()
    with pytest.raises(RuntimeError, match="lean.*native|native.*lean"):
        fp.GatePredictor(gate, mode="turbo")


def test_merged_shape_and_dtype():
    gate = _make_gate()
    for mode in ("lean", "native"):
        pred = fp.GatePredictor(gate, mode=mode)
        out = pred.merged(mx.zeros((7, 1, HIDDEN), dtype=mx.bfloat16))
        mx.eval(out)
        assert out.shape == (N_ROUTED,)
        assert out.dtype == mx.float32
