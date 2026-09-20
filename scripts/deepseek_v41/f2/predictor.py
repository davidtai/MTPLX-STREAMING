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
  * ``cpu`` (F2d): the prediction is computed ENTIRELY on the host, off the measured
    routing barrier and off the GPU.  The wrapper's barrier is just ``mx.eval(indices)``;
    the target gate's weight/bias/temperature are materialized as numpy f32 ONCE at
    install (``HostGatePredictor``), and the coordinator thread upcasts the raw bf16
    router-input words (bit-exact) and runs the SAME ``sqrt(softplus((x @ W.T)/temp)) +
    bias`` max-over-rows in numpy (MLX's ``softplus`` is ``logaddexp(x, 0)``, so the host
    uses ``np.logaddexp``).  This removes the predictor's f32 upcast + GEMM from the
    barrier's ``mx.eval`` entirely -- the ~0.26 ms/call the F2b probe charged to it.

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


def words_from_mx(x: "mx.array") -> np.ndarray:
    """The 'cpu' mode main-thread hand-off copy: a FRESH, independent numpy ``uint16``
    array holding the raw bf16 words of ``x`` (already materialized after the routing
    barrier, so ``memoryview`` reads the evaluated buffer with NO MLX op).

    Fresh per call by construction: ``.copy()`` allocates a new buffer that owns its data
    and shares nothing with ``x``.  The coordinator therefore holds the ONLY reference to
    these bytes, so the next call for the same source layer (or MLX reusing ``x``'s buffer)
    cannot alias or overwrite them -- the hand-off is race-safe without any timing window or
    sequence check.  ~1 us/call for a 6x5120 bf16 input (measured), the whole main-thread
    cost of the 'cpu' hand-off besides one ``queue.put``.
    """
    return np.frombuffer(memoryview(x), dtype=np.uint16).copy()


class HostGatePredictor:
    """CPU-only ('cpu' mode) twin of :class:`GatePredictor`: the target (next) layer's
    native biased-gate score, max-reduced over rows, computed ENTIRELY on the host so the
    ADVISORY prediction never touches the GPU or the measured routing barrier.

    The target gate's parameters are materialized on the host ONCE at construction (install
    time -- the GPU window holds the lock, so the ``bf16 -> f32`` upcast may run on Metal
    here; a single deterministic ``np.asarray(w.astype(mx.float32))``, never a per-call
    ``.astype``): weight as numpy ``float32 [n_routed, dim]``, bias as ``float32
    [n_routed]``, and the scalar temperature.  ``score_func`` must be ``sqrtsoftplus``
    (asserted HERE, before any generation -- AGENTS.md: validate at construction, fail once).

    ``scores_from_words`` is a pure numpy transform of the raw router-input words, running
    the SAME arithmetic the device path runs (``sqrt(softplus((x_f32 @ W_f32.T)/temp)) +
    bias`` then ``max`` over rows).  MLX's ``softplus`` is ``logaddexp(x, 0)`` so the host
    uses the numerically safe ``np.logaddexp(0, z)``; the only divergence from the device
    score is float32 GEMM accumulation order (measured max|delta| ~2e-7, top-k SET
    unchanged).  The f32 GEMM releases the GIL, so it does not block the main thread.
    """

    def __init__(self, gate) -> None:
        self.score_func = str(gate.score_func)
        if self.score_func != "sqrtsoftplus":
            raise RuntimeError(
                f"F2b cpu predictor supports score_func='sqrtsoftplus' only; gate "
                f"{getattr(gate, 'layer_id', '?')} has {self.score_func!r} -- use "
                f"MTPLX_DSV41_F2B_PREDICTOR=native or add the score_func to the host path"
            )
        self.dim = int(gate.dim)
        self.temp = float(gate.gate_temp)
        # Materialize ONCE: f32 weight [n_routed, dim] + f32 bias [n_routed] on the host.
        self.weight = np.ascontiguousarray(
            np.asarray(gate.weight.astype(mx.float32)), dtype=np.float32
        )
        self.bias = np.ascontiguousarray(
            np.asarray(gate.e_score_correction_bias.astype(mx.float32)), dtype=np.float32
        )
        self.n_routed = int(self.weight.shape[0])
        if self.weight.shape != (self.n_routed, self.dim) or self.bias.shape != (self.n_routed,):
            raise RuntimeError(
                f"F2b cpu predictor bad param shapes: weight {self.weight.shape}, "
                f"bias {self.bias.shape}, dim {self.dim}"
            )
        # Host bytes charged to memory admission (weight dominates; ~7.9 MB per target layer).
        self.host_bytes = int(self.weight.nbytes + self.bias.nbytes)

    def scores_from_words(self, words: np.ndarray, rows: int) -> np.ndarray:
        """merged[n_routed] f32 from the raw bf16 router-input words (``uint16``, length
        ``rows*dim``).  Runs on the coordinator thread.  Bit-exact ``bf16 -> f32`` upcast
        (bf16 == the high 16 bits of f32), then the gate arithmetic + max over rows."""
        x32 = (words.astype(np.uint32) << 16).view(np.float32).reshape(int(rows), self.dim)
        z = (x32 @ self.weight.T) / self.temp
        biased = np.sqrt(np.logaddexp(np.float32(0.0), z)) + self.bias
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
