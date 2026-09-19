"""F2b next-layer predictor (parameter-free) + host ranking + source/target geometry.

Reused unchanged from the runtime-ring design: the device side computes
``merged = max over rows of the target layer's native biased gate score`` -> [n_routed]
f32, using the model's own ``_gate_prefix`` / ``_gate_prefix_impl`` (exactly what
``Gate.__call__`` ranks) on the SOURCE layer's post-attention router input. The host side
ranks that evaluated prediction, drops the target layer's resident experts AND any expert
already in the ring, and returns the top ``k=3`` -- bit-identical to the offline scorer
``f2_predictor.merge_rank_exclude``.
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


class GatePredictor:
    """The target (next) layer's native biased-gate score, max-reduced over rows."""

    def __init__(self, gate) -> None:
        self.gate = gate

    def merged(self, tokens: mx.array) -> mx.array:
        gate = self.gate
        rows = int(tokens.shape[0])
        if _attn_compile_gate(rows):
            _scores, biased = _gate_prefix(gate)(
                tokens, gate.weight, gate.e_score_correction_bias
            )
        else:
            _scores, biased = _gate_prefix_impl(
                tokens, gate.weight, gate.e_score_correction_bias,
                float(gate.gate_temp), str(gate.score_func),
            )
        return biased.max(axis=0)


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
