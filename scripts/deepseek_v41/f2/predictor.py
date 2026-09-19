"""F2b next-layer predictor (parameter-free) + host ranking + source/target geometry.

The device side computes ``merged = max over rows of the target layer's native biased
gate score`` -> [n_routed] f32.  Two construction-time modes select HOW that score is
produced (the host ranking below is identical for both):

  * ``lean`` (default): one module-level ``mx.compile``d tape ``(tokens, weight, bias)
    -> merged[n_routed]`` that reuses the model's own ``_gate_prefix_impl`` (sqrtsoftplus
    branch, f32 upcast preserved) and rides the ``.max(axis=0)`` reduction on the SAME
    tape.  ``mx.compile`` fuses the divide + softplus + sqrt + bias-add elementwise chain
    into a single Metal kernel and prunes the unused raw scores; the reshape rides the
    tape too, so the wrapper hands the raw view straight in.  Confirmed byte-identical to
    the eager impl on CPU (so the predicted top-k SET matches the router's), at strictly
    fewer kernels and lower host graph-build cost than ``native``.
  * ``native``: the prior path -- the shared K22 compiled ``_gate_prefix`` when the router
    itself would compile for this row count, else the eager ``_gate_prefix_impl``, then
    ``.max(axis=0)`` OUTSIDE any tape.  Kept for A/B.

The host side ranks that evaluated prediction, drops the target layer's resident experts
AND any expert already in the ring, and returns the top ``k=3`` -- bit-identical to the
offline scorer ``f2_predictor.merge_rank_exclude``.  The prediction is ADVISORY (it only
chooses which expert planes to read speculatively; it never feeds model arithmetic).
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from mtplx.models.deepseek_v41_moe import (
    _attn_compile_gate,
    _gate_prefix,
    _gate_prefix_impl,
)

DEFAULT_TOP_K = 3
# Verify predictor geometry: targets 4..39 predicted from sources 3..38 (f1-real-predictor).
FIRST_TARGET_LAYER = 4

# Module-level cache of the lean compiled tape, keyed by the gate constants that enter the
# graph (temperature, router dim).  All source layers share one temperature and one dim, so
# in production this holds a single compiled callable; every source layer's weight/bias are
# TRACED INPUTS, so that one tape serves them all.  Shape-specialized (mx.compile's default):
# one trace per distinct row count.  In decode the router input is 4..8 rows, so at most a
# handful of traces are built lazily and then cached -- amortized to ~0 over the ~7k calls a
# run makes.  Shapeless=True also compiles correctly here, but the shape-specialized tape is
# byte-identical to the eager impl per shape (measured) -- the stronger ranking guarantee --
# and its retrace is provably bounded, so it is preferred.
_LEAN_MERGED_CACHE: dict[tuple, "callable"] = {}


def _lean_merged_for(temp: float, dim: int):
    """Build/fetch the module-level ``mx.compile``d ``(tokens, weight, bias) ->
    merged[n_routed]`` for gate temperature ``temp`` and router ``dim`` (sqrtsoftplus only).

    The body reuses the pinned ``_gate_prefix_impl`` so the arithmetic (f32 upcast, GEMM /
    temp, ``sqrt(softplus(.))``, + correction bias) is exactly the router's; ``mx.compile``
    fuses the elementwise chain and prunes the unused raw scores, and the max-over-rows
    reduction rides the same tape.  ``dim`` is captured so the reshape lives INSIDE the tape
    (no per-call host reshape); the f32 upcast is kept because a bf16 GEMM reorders the top-8
    on a large fraction of inputs (measured), which would corrupt the ranking.
    """
    key = (float(temp), int(dim))
    fn = _LEAN_MERGED_CACHE.get(key)
    if fn is None:
        t, d = key

        def _impl(tokens, weight, bias):
            xf = tokens.reshape(-1, d)
            _scores, biased = _gate_prefix_impl(xf, weight, bias, t, "sqrtsoftplus")
            return biased.max(axis=0)

        fn = mx.compile(_impl)
        _LEAN_MERGED_CACHE[key] = fn
    return fn


class GatePredictor:
    """The target (next) layer's native biased-gate score, max-reduced over rows.

    Everything is bound once at construction (weight, bias, temperature, dim, and the
    prebound ``merged`` callable), so ``merged()`` is a single call with no per-call branch
    on invariant module/model metadata (AGENTS.md: validate at construction, no
    eligible-or-fallback branch in the enabled hot path).  ``mode`` is chosen once at
    construction from ``MTPLX_DSV41_F2B_PREDICTOR`` (read once in ``install``); an unknown
    mode, or ``lean`` on a non-sqrtsoftplus gate, fails loudly HERE, before any generation.
    """

    def __init__(self, gate, *, mode: str = "lean") -> None:
        self.gate = gate
        self.mode = str(mode)
        self.weight = gate.weight
        self.bias = gate.e_score_correction_bias
        self.temp = float(gate.gate_temp)
        self.dim = int(gate.dim)
        self.score_func = str(gate.score_func)
        if self.mode == "lean":
            if self.score_func != "sqrtsoftplus":
                raise RuntimeError(
                    f"F2b lean predictor supports score_func='sqrtsoftplus' only; gate "
                    f"{getattr(gate, 'layer_id', '?')} has {self.score_func!r} -- use "
                    f"MTPLX_DSV41_F2B_PREDICTOR=native or add the score_func to the tape"
                )
            compiled = _lean_merged_for(self.temp, self.dim)
            weight, bias = self.weight, self.bias
            self._merged = lambda tokens: compiled(tokens, weight, bias)
        elif self.mode == "native":
            self._merged = self._native_merged
        else:
            raise RuntimeError(
                f"unknown MTPLX_DSV41_F2B_PREDICTOR mode {self.mode!r}; want 'lean' or 'native'"
            )

    def _native_merged(self, tokens: mx.array) -> mx.array:
        """Prior path: the shared K22 compiled gate prefix when the router would compile for
        this row count, else the eager impl, then ``.max(axis=0)`` outside the tape."""
        gate = self.gate
        xf = tokens.reshape(-1, self.dim)
        if _attn_compile_gate(int(xf.shape[0])):
            _scores, biased = _gate_prefix(gate)(
                xf, gate.weight, gate.e_score_correction_bias
            )
        else:
            _scores, biased = _gate_prefix_impl(
                xf, gate.weight, gate.e_score_correction_bias,
                float(gate.gate_temp), str(gate.score_func),
            )
        return biased.max(axis=0)

    def merged(self, tokens: mx.array) -> mx.array:
        """Evaluate the target layer's ``max-over-rows biased gate score`` for ``tokens``
        (the source layer's post-attention router input, any leading shape reshaped to
        ``[-1, dim]`` on the bound path).  Returns f32 ``[n_routed]`` (lazy)."""
        return self._merged(tokens)


def rank_targets(merged, skip, k: int = DEFAULT_TOP_K) -> list[int]:
    """Top-``k`` experts of the evaluated ``merged`` [n_routed] prediction, in descending
    score order, excluding any expert for which ``skip(expert)`` is True (resident in the
    target bank, or already in the ring). ``merged`` may be an mlx array (already forced by
    the wrapper's ``mx.eval(indices, merged)``) or a numpy array.
    """
    scores = np.asarray(merged.tolist() if hasattr(merged, "tolist") and not isinstance(merged, np.ndarray)
                        else merged, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    ids: list[int] = []
    for value in order:
        expert = int(value)
        if skip(expert):
            continue
        ids.append(expert)
        if len(ids) >= int(k):
            break
    return ids


def select_prefetch_sources(routed_layers, *, first_target: int = FIRST_TARGET_LAYER):
    """Source layers that predict: ``L`` predicts ``L+1`` iff ``L+1`` is routed and
    ``L+1 >= first_target``. Retained 0..39 with first_target=4 -> sources 3..38."""
    routed = {int(x) for x in routed_layers}
    first_target = int(first_target)
    return sorted(L for L in routed if (L + 1) in routed and (L + 1) >= first_target)
