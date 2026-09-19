"""F2 live next-layer expert predictor -- device biased-gate max, host top-k issue.

Parameter-free (fix #3): no learned correction, no margin threshold. For a source
layer L (3..38) predicting target layer L+1 (4..39):

  * DEVICE (``GatePredictor.merged``): the target layer's OWN native biased gate
    score -- exactly the tensor ``Gate.__call__`` ranks (mtplx/models/
    deepseek_v41_moe.py: ``biased = scores + e_score_correction_bias``, L296) --
    applied to the SOURCE layer's post-attention router input (the tensor handed to
    the source switch), reduced by MAX over the verify rows to ``[n_routed]`` f32.
    Rides the model's own ``_gate_prefix`` compiled tape when the router would
    compile at this row count, else the eager ``_gate_prefix_impl`` -- the identical
    transform, no redundant f32 gate-weight copy (deepseek_v41_moe.gate_predict_topk
    docstring).

  * HOST (``Issue.__call__``): rank the already-evaluated prediction, drop the
    target layer's READY owners (persistent, transient OR committed ring), take the
    top ``k=3`` remaining experts in descending score order, and issue them once via
    ``runtime.prefetch_experts(target, ids, verify=True)``.

Fix #2 -- the split that keeps the barrier count at 1 routing + 1 miss-drain:
``prepare`` does ONLY the device prediction and evaluates it on the source route's
existing indices barrier (``mx.eval(indices, merged)``); ALL host ranking, READY
filtering and issue happen in ``__call__``, which the runner calls AFTER
``begin_split_route`` has submitted the demand reads. Codex's screen did the numpy
ranking inside ``prepare`` (on the critical path before demand reads were issued);
this moves it after.

The ranking is bit-identical to the offline scorer's ``merge_rank_exclude``
(scripts/deepseek_v41/f2_predictor.py): merge=max over rows, exclude READY residents,
stable descending top-k. ``max`` is the f1-real-predictor winner over ``sum``.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from mtplx.expert_slots import ExpertSlotState
from mtplx.models.deepseek_v41_moe import (
    _attn_compile_gate,
    _gate_prefix,
    _gate_prefix_impl,
)

DEFAULT_TOP_K = 3


class GatePredictor:
    """The target (next) layer's native biased-gate score, max-reduced over rows.

    ``gate`` is the live target-layer MoE ``Gate`` module
    (``model.model.layers[target].mlp.gate``): ``.weight`` [n_routed, hidden] bf16,
    ``.e_score_correction_bias`` [n_routed] f32, ``.gate_temp``, ``.score_func``.
    """

    def __init__(self, gate) -> None:
        self.gate = gate

    def merged(self, tokens: mx.array) -> mx.array:
        """``tokens`` = [rows, hidden] post-attention router input -> [n_routed] f32."""
        gate = self.gate
        rows = int(tokens.shape[0])
        if _attn_compile_gate(rows):
            _scores, biased = _gate_prefix(gate)(
                tokens, gate.weight, gate.e_score_correction_bias
            )
        else:
            _scores, biased = _gate_prefix_impl(
                tokens,
                gate.weight,
                gate.e_score_correction_bias,
                float(gate.gate_temp),
                str(gate.score_func),
            )
        # MAX over the verify rows: an expert one row favours strongly is warmed,
        # rather than diluted by a SUM across rows (f1-real-predictor).
        return biased.max(axis=0)


class Issue:
    """One source layer's prefetch issue, bound at install to its target = L+1."""

    def __init__(self, runtime, target: int, predictor: GatePredictor, *, top_k: int = DEFAULT_TOP_K) -> None:
        self.runtime = runtime
        self.target = int(target)
        self.predictor = predictor
        self.top_k = int(top_k)
        # The physical slot OBJECTS are fixed for the runtime's life; only their
        # .state/.layer/.expert fields mutate. Snapshot the three tiers' containers
        # once (persistent dict, transient tuple, prefetch-ring dict) so READY-owner
        # exclusion reads live fields without rebuilding the container per call.
        slots = runtime.slots
        self._slots = (
            tuple(slots._persistent.values())
            + tuple(slots._transient)
            + tuple(slots._prefetch.values())
        )
        self._merged: mx.array | None = None

    def prepare(self, tokens: mx.array, indices: mx.array) -> None:
        """Build the device prediction and evaluate it on the indices barrier.

        Exactly one host sync for the layer's routing: the same ``mx.eval`` the
        runner would spend on ``indices`` now also forces ``merged`` (a pure read of
        the frozen target gate + this layer's input -- never an ancestor of the
        routed output, so it can only warm the cache). No host ranking here.
        """
        self._merged = self.predictor.merged(tokens)
        mx.eval(indices, self._merged)

    def __call__(self) -> int:
        """Rank the evaluated prediction, drop READY owners, issue the top-k.

        Called after ``begin_split_route`` has submitted the demand reads, so this
        never delays a demand read. Returns the number of speculative reads issued
        (0 if throttled/skipped by the runtime).
        """
        merged = self._merged
        self._merged = None
        if merged is None:
            return 0
        # ``merged`` was forced in prepare; ``.tolist()`` reads settled bytes on the
        # host without a new barrier (same pattern as the runner's indices.tolist()).
        scores = np.asarray(merged.tolist(), dtype=np.float64)
        ready = {
            slot.expert
            for slot in self._slots
            if slot.state is ExpertSlotState.READY and slot.layer == self.target
        }
        order = np.argsort(-scores, kind="stable")
        ids: list[int] = []
        for value in order:
            expert = int(value)
            if expert in ready:
                continue
            ids.append(expert)
            if len(ids) >= self.top_k:
                break
        return self.runtime.prefetch_experts(self.target, ids, verify=True)
