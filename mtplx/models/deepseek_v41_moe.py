"""DeepSeek-V4.1-Flash MoE submodule (worker W11).

A faithful, line-by-line MLX transliteration of the DeepSeek reference
``inference/model.py`` MoE stack -- :class:`Gate` (L792-828), :class:`Expert`
(L830-851) and :class:`MoE` (L854-904) -- *modified for the MTPLX expert kernel*:
the routed experts do not run as a ``nn.ModuleList`` of :class:`Expert`, they run
through the MTPLX expert-streaming seam ``switch_mlp`` (the hy3 convention that
:func:`mtplx.models.expert_mlx.bind_streamed_switches` rebinds).  The class names,
constructor arg order (``layer_id, args``) and ``forward`` signature / return
shape all match the reference so W10's ``Block`` calls this exactly as the
reference ``Block`` does (reference ``Block.__init__`` builds ``self.ffn =
MoE(layer_id, args)`` at L931 and ``Block.forward`` calls ``self.ffn(x,
image_mask)`` at L992).

``args`` here is *this port's* MLX :class:`~mtplx.models.deepseek_v41.ModelArgs`
(the ``text_config`` field names), taken in the reference's ``(layer_id, args)``
order.  The released 40-layer config carries ``scoring_func="sqrtsoftplus"``,
``topk_method="noaux_tc"``, ``num_experts_per_tok=6``, ``n_routed_experts=384``,
``routed_scaling_factor=1.5``, ``swiglu_limit=10.0`` and no group routing
(``n_group``/``topk_group`` are absent, so the reference Gate does a plain top-k
over all experts -- there is no group-limited selection to transliterate).

Seam contract (also in docs/deepseek-v41/W11_REPORT.md):

* ``self.switch_mlp`` is the routed-expert seam.  It is called ``switch_mlp(xf,
  indices)`` with ``xf`` = ``[n_tokens, hidden]`` and ``indices`` =
  ``[n_tokens, top_k]`` int32 (the reference top-6 expert ids, in
  ``(scores+bias)``-descending order), and must return ``[n_tokens, top_k,
  hidden]`` -- the *unweighted* routed outputs.  :meth:`MoE.__call__` applies the
  reference-normalised routing weights and the shared expert outside the seam.
* Resident / unit-test default: mlx-lm :class:`~mlx_lm.models.switch_layers.
  SwitchGLU` carrying :class:`ClampedSwiGLU` as its ``activation`` seam, so the
  routed experts get the reference's clamped SwiGLU.  ``bind_streamed_switches``
  replaces this whole module with a streamed switch
  (``HotExpertSwitchGLU`` / ``MappedExpertSwitchGLU`` / ``DenseIslandSwitchGLU``)
  that gathers the Q2 records from ``experts.bin``.
* CLAMP INJECTION / KNOWN GAP: the streamed switches call
  ``mlx_lm.models.activations.swiglu(gate, up)`` directly and expose **no
  activation hook** (``bind_streamed_switches`` reads nothing off the switch it
  replaces).  So on the *streamed* path the ``swiglu_limit`` clamp is currently
  NOT applied.  At ``swiglu_limit=10.0`` this is numerically inert on the real
  layer-0 records (the pre-activation projections never reach +/-10 -- proven in
  tests/models/test_deepseek_v41_moe.py::test_streamed_clamp_is_inert_on_real_records),
  so streamed output is bit-for-bit the clamped output here.  The resident path
  (this module's ``SwitchGLU`` + :class:`ClampedSwiGLU`) *does* clamp.  Making the
  streamed switch clamp for the general case is a one-line-per-call-site change in
  ``mtplx/models/expert_mlx.py`` (add a per-runtime ``swiglu_limit`` on the spec
  and clamp between the up/gate qmm and ``swiglu``); it is out of W11's allowlist
  and unnecessary for this artifact, so it is documented, not applied.

Resident parameter names match the loader's sanitize (see
``mtplx/models/deepseek_v41.py`` ``_sanitize_name``): ``ffn.gate.weight`` ->
``mlp.gate.weight`` (bf16), ``ffn.gate.bias`` ->
``mlp.gate.e_score_correction_bias`` (f32; ``bias_vl`` dropped on the text path),
``ffn.shared_experts.w{1,2,3}.{weight,scales,biases}`` ->
``mlp.shared_experts.w{1,2,3}...`` (resident q8, gs64 affine).  The routed
``ffn.experts.*`` are streamed and never resident, so ``switch_mlp`` is skipped by
the resident quantiser and excluded from the strict 1,616-key text load.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.switch_layers import SwiGLU, SwitchGLU

from . import deepseek_v41_stage_timing as _stime
from .expert_mlx import run_switch_with_shared_overlap


# ---------------------------------------------------------------------------
# K22 (W41): fold the MoE gate's PURE prefix and the MoE combine into compiled
# tapes under the same ``MTPLX_DSV41_ATTN_COMPILE`` flag/row-cap as the attention
# chains (deepseek_v41._attn_use_compile).  Both are pure and bit-exact to eager
# in the decode/verify row regime (measured, W41): the gate prefix is the score
# GEMM + sqrtsoftplus + correction bias (the argpartition/argsort/top-k selection
# that produces the fenced routing barrier stays OUTSIDE), and the combine is the
# weighted routed sum + shared add.  The deepseek_v41 <- deepseek_v41_moe import
# is one-directional, so the flag/cap/cache are reached by a lazy import here (no
# cycle at module load; cheap after the first call).
def _attn_compile_gate(rows: int) -> bool:
    from . import deepseek_v41 as _dv
    return _dv._attn_use_compile(rows)


def _compiled(key, builder):
    """Fetch/build the compiled tape for ``key`` from deepseek_v41's shared
    ``_ATTN_COMPILED`` cache, so a test that clears that cache resets these too."""
    from . import deepseek_v41 as _dv
    fn = _dv._ATTN_COMPILED.get(key)
    if fn is None:
        fn = mx.compile(builder())
        _dv._ATTN_COMPILED[key] = fn
    return fn


def _gate_prefix_impl(x, weight, bias, temp, score_func):
    """The MoE gate's pure prefix -> (scores, biased).  Byte-identical to
    :meth:`Gate.__call__`'s L810-822 body."""
    scores = (x.astype(mx.float32) @ weight.astype(mx.float32).T) / temp
    if score_func == "softmax":
        scores = mx.softmax(scores, axis=-1)
    elif score_func == "sigmoid":
        scores = mx.sigmoid(scores)
    else:  # sqrtsoftplus
        scores = mx.sqrt(nn.softplus(scores))
    return scores, scores + bias


def _gate_prefix(gate):
    key = ("gate_prefix", str(gate.score_func), float(gate.gate_temp))
    temp, sf = float(gate.gate_temp), str(gate.score_func)
    return _compiled(key, lambda: (lambda x, weight, bias: _gate_prefix_impl(x, weight, bias, temp, sf)))


def gate_predict_topk(gate: "Gate", x: mx.array, k: int, margin: float = 0.0) -> mx.array:
    """W93 gate-oracle prefetch prediction: the SET of the top-``k`` expert ids
    layer L's router assigns to input ``x``, using the port's EXACT routing
    transform.

    This is the shipped :meth:`Gate.__call__` scoring (model.py L810-822: score
    GEMM / gate_temp, ``sqrtsoftplus``, + ``e_score_correction_bias``) truncated
    to the top ``k`` of ``(scores + bias)``.  ``k`` is the prefetch width (10/12),
    WIDER than the gate's shipped top-6; prefetch needs the SET, not
    ``torch.topk``'s descending order or the routing weights, so the ``argsort``
    and weight gather that :meth:`Gate.__call__` performs (L823-824) are
    deliberately dropped.  A pure read of ``x`` and the (frozen) gate weights: it
    has no side effect and is never an ancestor of the layer's own output, so
    evaluating it can only warm the expert cache, never change a logit
    (W93_GATE_PREFETCH.md §2).

    Scoring goes through the SAME code path :meth:`Gate.__call__` takes for this
    row count (W93 review LOW-1): the shared K22 compiled ``_gate_prefix`` tape
    when the router would compile (``_attn_compile_gate`` -> ``ATTN_COMPILE`` armed,
    decode/verify rows), else the eager ``_gate_prefix_impl``.  Two payoffs:

    * **exact alignment with the router (a'=1.0).**  Because the ``biased`` scores
      here are produced by the byte-for-byte same (compiled *or* eager) prefix the
      router runs, the predicted top-``k`` SET is bit-identical to the router's own
      selection on the same input in *every* regime -- not merely byte-equal via
      the K22 compiled==eager claim.  At ``k == gate.topk`` this reproduces
      :meth:`Gate.__call__`'s ``indices`` set exactly.
    * **no redundant f32 weight copy.**  The old body called ``_gate_prefix_impl``
      unconditionally, so even inside a compile-armed decode window it eagerly
      materialised a ``[n_routed, dim]`` f32 copy of the gate weight
      (384x5120 -> 7.9 MiB) *per layer per token*.  Riding the router's compiled
      tape fuses that upcast into the score GEMM (and reuses the single cached
      tape the router already built -- same ``("gate_prefix", score_func,
      gate_temp)`` key), so the predictor allocates no standalone f32 weight copy.

    Returns ``[n, k]`` int32.
    """
    xf = x.reshape(-1, gate.dim)
    if _attn_compile_gate(int(xf.shape[0])):
        _scores, biased = _gate_prefix(gate)(
            xf, gate.weight, gate.e_score_correction_bias
        )
    else:
        _scores, biased = _gate_prefix_impl(
            xf,
            gate.weight,
            gate.e_score_correction_bias,
            float(gate.gate_temp),
            str(gate.score_func),
        )
    width = int(biased.shape[-1])
    k = max(1, min(int(k), width))
    part = mx.argpartition(-biased, kth=k - 1, axis=-1)[..., :k].astype(mx.int32)
    if margin:
        # W95 confidence gate (retune): keep only candidates whose score is
        # >= (the top-6 boundary score) - ``margin``, where the boundary is the
        # 6th-highest score (DSV4.1 routes top-6).  ``margin`` < 0 TRIMS to the
        # confident subset (threshold ABOVE the 6th -> raises the issued-set
        # precision, cutting the wasted speculative reads that made k=12 net-slower
        # in window 39); ``margin`` > 0 WIDENS below the boundary.  Gated-out
        # entries become -1 (the issue site drops them).  Purely a read of ``x`` +
        # the frozen gate weights -> never an ancestor of the routed output, so
        # this only changes WHICH experts are pre-warmed, never a logit.
        boundary_rank = min(6, k)
        boundary = mx.sort(biased, axis=-1)[..., -boundary_rank]  # [n] 6th-highest
        threshold = (boundary - float(margin))[..., None]         # [n, 1]
        part_scores = mx.take_along_axis(biased, part, axis=-1)   # [n, k]
        part = mx.where(part_scores >= threshold, part, mx.array(-1, dtype=mx.int32))
    return part


def _moe_combine_impl(routed, weights, shared):
    """The MoE combine: weighted routed sum (f32 accumulator) + shared add.
    Byte-identical to the eager ``(routed*weights).sum(-2) + shared``."""
    return (routed.astype(mx.float32) * weights[..., None]).sum(axis=-2) + shared


def _moe_combine(routed, weights, shared):
    return _compiled(("moe_combine",), lambda: _moe_combine_impl)(routed, weights, shared)


def _moe_combine_dispatch(routed, weights, shared, rows: int):
    """Compiled combine at decode/verify row counts, eager otherwise (the eager
    branch is byte-for-byte the original ``(routed*weights).sum(-2) + shared``)."""
    if _attn_compile_gate(rows):
        return _moe_combine(routed, weights, shared)
    return (routed.astype(mx.float32) * weights[..., None]).sum(axis=-2) + shared


class ClampedSwiGLU(SwiGLU):
    """``SwitchGLU`` activation carrying the reference ``Expert``'s ``swiglu_limit``
    clamp (model.py L845-848).

    mlx-lm's :class:`~mlx_lm.models.switch_layers.SwitchGLU` calls
    ``self.activation(x_up, x_gate)`` -- so the first argument is the *up* branch
    (``w3``) and the second is the *gate* branch (``w1``), the opposite of what the
    names suggest.  The reference clamp is asymmetric (model.py L846-847)::

        up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)  # two-sided
        gate = torch.clamp(gate, max=self.swiglu_limit)                      # upper tail only

    Both cuts land on the pre-activation projections, before ``silu``.  At
    ``limit <= 0`` this defers to the stock fused ``swiglu`` kernel unchanged.
    Holds no parameters, so the weight tree and load path are untouched.
    """

    def __init__(self, limit: float = 0.0) -> None:
        super().__init__()
        self.limit = float(limit or 0.0)

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        # x == up branch, gate == gate branch (SwitchGLU arg order).
        if self.limit > 0:
            x = mx.clip(x, -self.limit, self.limit)      # up: two-sided (L846)
            gate = mx.minimum(gate, self.limit)          # gate: upper tail (L847)
        return super().__call__(x, gate)                 # swiglu(gate, x) = silu(gate)*x


class Gate(nn.Module):
    """Reference ``Gate`` (model.py L792-828): sqrtsoftplus scoring, ``noaux_tc``
    correction bias to *select* experts, unbiased scores to *weight* them, then
    ``norm_topk_prob`` and ``route_scale``.

    Constructor takes ``(layer_id, args)`` in the reference's order (model.py
    L796).  The released text config has no group routing, so this is a plain
    top-k over all ``n_routed_experts`` -- there is no ``n_group``/``topk_group``
    branch in the reference to transliterate.  The image-span routing bias
    (``bias_vl``, model.py L807/L819-820) is dropped: this is the text path, so
    ``image_mask`` is always ``None``.

    Returns ``(weights, indices)`` in the reference's order (model.py L828).
    ``indices`` is int32 ``[n, top_k]`` in ``(scores+bias)``-descending order (the
    order ``torch.topk`` produces at L823); ``weights`` is f32 ``[n, top_k]``.
    """

    def __init__(self, layer_id: int, args) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.hidden_size                                  # L799
        self.topk = args.num_experts_per_tok                        # L800 (n_activated)
        self.score_func = args.scoring_func                          # L801
        self.gate_temp = float(getattr(args, "gate_temp", 1.0) or 1.0)  # L802
        self.norm_topk_prob = args.norm_topk_prob                    # L803
        self.route_scale = args.routed_scaling_factor                # L804
        self.n_routed = args.n_routed_experts
        # gate.weight [n_routed, dim], bf16 in the artifact (L805).
        self.weight = mx.zeros((self.n_routed, self.dim), dtype=mx.bfloat16)
        # noaux_tc correction bias, f32 (reference `self.bias`, L806); the
        # loader renames ffn.gate.bias -> mlp.gate.e_score_correction_bias.
        self.e_score_correction_bias = mx.zeros((self.n_routed,), dtype=mx.float32)

    def __call__(
        self, x: mx.array, image_mask: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array]:
        # L810-822 the PURE gate prefix: score GEMM / temp, scoring function,
        # correction bias.  K22 folds it into one compiled tape at decode/verify
        # (byte-identical to this eager body off / above the row cap); the
        # data-dependent top-k selection below stays eager (it builds the fenced
        # routing barrier).  ``x`` here is [n, dim], so ``x.shape[0]`` is the row
        # count the row-cap gates on.
        if _attn_compile_gate(int(x.shape[0])):
            scores, biased = _gate_prefix(self)(
                x, self.weight, self.e_score_correction_bias
            )
        else:
            # L810: scores = linear(x.float(), weight.float()) / gate_temp
            scores = (x.astype(mx.float32) @ self.weight.astype(mx.float32).T) / self.gate_temp
            # L811-817: scoring function
            if self.score_func == "softmax":
                scores = mx.softmax(scores, axis=-1)
            elif self.score_func == "sigmoid":
                scores = mx.sigmoid(scores)
            else:  # sqrtsoftplus
                scores = mx.sqrt(nn.softplus(scores))
            # L818-822: the correction bias steers selection only (bias_vl unused
            # on the text path).
            biased = scores + self.e_score_correction_bias
        # L823: indices = (scores + bias).topk(topk)[1] -- top-k, sorted desc.
        # mx has no index-returning top-k, so argpartition the top-k set then
        # argsort it by -biased to reproduce torch.topk's descending order.
        part = mx.argpartition(-biased, kth=self.topk - 1, axis=-1)[..., : self.topk]
        order = mx.argsort(-mx.take_along_axis(biased, part, axis=-1), axis=-1)
        indices = mx.take_along_axis(part, order, axis=-1).astype(mx.int32)
        # L824: weights = scores.gather(1, indices) -- the *unbiased* scores.
        weights = mx.take_along_axis(scores, indices, axis=-1)
        # L825-826: norm_topk_prob (the +1e-20 is the training constant, not norm_eps).
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        # L827: route_scale
        weights = weights * self.route_scale
        return weights, indices


class Expert(nn.Module):
    """Reference ``Expert`` (model.py L830-851): one clamped-SwiGLU FFN.

    Constructor matches the reference ``Expert(dim, inter_dim, dtype=None,
    swiglu_limit=0.0)`` (model.py L834).  ``w1`` = gate_proj, ``w3`` = up_proj,
    ``w2`` = down_proj (DeepSeek's convention, L836-838).  ``dtype`` is accepted
    for signature parity (the reference passes ``float4`` to routed experts) but
    is unused here: in this port only the *shared* expert is an :class:`Expert`
    (routed experts run through the streamed ``switch_mlp`` seam), and the shared
    expert's ``w{1,2,3}`` are dense ``nn.Linear`` at construction, converted to
    resident q8 by the model's ``nn.quantize`` (mlx-lm ``QuantizedLinear``, so
    ``w1(x)`` is an ``mx.quantized_matmul`` at serve time).
    """

    def __init__(
        self, dim: int, inter_dim: int, dtype=None, swiglu_limit: float = 0.0
    ) -> None:
        super().__init__()
        self.swiglu_limit = float(swiglu_limit or 0.0)               # L839
        self.w1 = nn.Linear(dim, inter_dim, bias=False)              # gate_proj (L836)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)             # down_proj (L837)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)            # up_proj   (L838)

    def __call__(self, x: mx.array, weights: Optional[mx.array] = None) -> mx.array:
        dtype = x.dtype                                              # L842
        gate = self.w1(x).astype(mx.float32)                        # L843
        up = self.w3(x).astype(mx.float32)                          # L844
        if self.swiglu_limit > 0:                                    # L845
            up = mx.clip(up, -self.swiglu_limit, self.swiglu_limit)  # L846 two-sided
            gate = mx.minimum(gate, self.swiglu_limit)              # L847 upper tail
        x = nn.silu(gate) * up                                      # L848
        if weights is not None:                                     # L849-850
            x = weights * x
        return self.w2(x.astype(dtype))                            # L851


class MoE(nn.Module):
    """Reference ``MoE`` (model.py L854-904): top-``num_experts_per_tok`` routed
    experts plus one shared expert, modified for the MTPLX expert kernel.

    Constructor takes ``(layer_id, args)`` in the reference's order (model.py
    L858).  The reference builds ``self.experts = nn.ModuleList([Expert(...) ...])``
    (L874-886) and dispatches them token-by-token in ``forward`` (L895-901); this
    port replaces that whole routed-expert path with the MTPLX streaming seam
    ``self.switch_mlp`` and applies the per-expert routing weights outside the
    seam.  The shared expert (L887-888) and the gate (L873) are unchanged.
    """

    def __init__(self, layer_id: int, args) -> None:
        super().__init__()
        self.layer_id = layer_id                                     # L860
        self.dim = args.hidden_size                                  # L861
        self.n_routed_experts = args.n_routed_experts               # L868
        self.n_activated_experts = args.num_experts_per_tok         # L870
        self.gate = Gate(layer_id, args)                            # L873
        # L874-886 routed Experts -> the MTPLX streamed-switch seam.  Resident /
        # test default is SwitchGLU with the reference's clamped activation;
        # bind_streamed_switches rebinds this to a bank-backed streamed switch.
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=ClampedSwiGLU(args.swiglu_limit),
        )
        assert args.n_shared_experts == 1                            # L887
        self.shared_experts = Expert(                               # L888
            args.hidden_size, args.moe_intermediate_size, swiglu_limit=args.swiglu_limit
        )
        # Resolve the legacy environment arm once at construction.  The streamed
        # binder may replace this route after it installs ``switch_mlp`` when the
        # immutable runtime config selects verify_shared_overlap.  Keeping the
        # callable prebound avoids an environment read and invariant check in
        # every routed layer forward.
        legacy_shared_overlap = (
            os.environ.get("MTPLX_DSV41_SHARED_OVERLAP") == "1"
        )
        self._routed_shared_route = (
            self._run_routed_shared_overlap
            if legacy_shared_overlap
            else self._run_routed_then_shared
        )
        self._shared_overlap_route_installed = legacy_shared_overlap

    def install_streamed_shared_route(self, *, overlap: bool) -> None:
        """Bind the routed/shared execution order after switch installation."""

        self._shared_overlap_route_installed = bool(overlap)
        self._routed_shared_route = (
            self._run_routed_shared_overlap
            if overlap
            else self._run_routed_then_shared
        )

    def _run_routed_shared_overlap(
        self,
        xf: mx.array,
        indices: mx.array,
    ) -> tuple[mx.array, mx.array]:
        with _stime.stage("moe.routed_switch") as _st:
            routed, shared = run_switch_with_shared_overlap(
                self.switch_mlp,
                xf,
                indices,
                lambda: self.shared_experts(xf).astype(mx.float32),
            )
            _st.add(routed, shared)
        return routed, shared

    def _run_routed_then_shared(
        self,
        xf: mx.array,
        indices: mx.array,
    ) -> tuple[mx.array, mx.array]:
        with _stime.stage("moe.routed_switch") as _st:
            routed = self.switch_mlp(xf, indices)                   # [n, top_k, dim]
            _st.add(routed)
        with _stime.stage("moe.shared_expert") as _st:
            shared = self.shared_experts(xf).astype(mx.float32)
            _st.add(shared)
        return routed, shared

    def combine_routed(
        self, routed: mx.array, weights: mx.array, xf: mx.array
    ) -> mx.array:
        """Reference weighted routed-sum + shared expert (MoE.forward L893-903),
        given a precomputed ``routed = switch_mlp(xf, indices)`` and its routing
        ``weights``.  Returns the flat ``[n, dim]`` f32 result (the caller
        reshapes / casts).

        Split out of :meth:`__call__` so W30's layer-major prefill can keep the
        **resident** gate and shared expert per chunk (matmul batch size == the
        chunk, byte-identical to chunk-major) while batching only the *streamed*
        ``switch_mlp`` across chunks (the read-the-bank-once part).  The gate's
        ``xf @ weight.T`` and this shared ``Expert`` are NOT invariant to the row
        (M) batch size, so batching them across chunks reassociates their fp32
        reductions and flips greedy argmax on the real model (W30 addendum); the
        streamed per-expert gather is M-invariant, so only it is batched."""
        # L893-901: the streamed switch returns the *unweighted* per-expert
        # outputs [n, top_k, dim]; the reference multiplies each expert output by
        # its weight inside the loop (L900) -- done here in one weighted sum, in
        # f32 to match the reference's f32 accumulator (L893).
        # Routed through the same compiled/eager combine dispatch as __call__ so
        # the layer-major path stays byte-identical to chunk-major under K4/K22.
        # W47: bracket the shared expert + combine so layer-major prefill reports
        # the same moe.shared_expert / moe.combine stages as chunk-major
        # (MoE.__call__); no-op off / decode (combine_routed is layer-major only).
        with _stime.stage("moe.shared_expert") as _st:
            shared = self.shared_experts(xf).astype(mx.float32)
            _st.add(shared)
        with _stime.stage("moe.combine") as _st:
            out = _moe_combine_dispatch(routed, weights, shared, int(xf.shape[0]))
            _st.add(out)
        return out

    def __call__(self, x: mx.array, image_mask: Optional[mx.array] = None) -> mx.array:
        # reference MoE.forward, L889-904.
        shape = x.shape                                              # L890
        xf = x.reshape(-1, self.dim)                                # L891
        # L892: gate (text path -> image_mask None); reference order (weights, indices).
        # W37 "gate + top-k + routing barrier": fencing ``indices`` here is the
        # per-layer routing barrier (``mx.eval(indices)``) -- the streamed switch's
        # own internal barrier is then a no-op, so this stage owns the host sync
        # the ledger prices at ~40/token and moe.routed_switch owns only the
        # subsequent miss I/O + gather_qmm.  On the resident (test) path there is
        # no internal barrier; the fence attributes the gate compute here anyway.
        with _stime.stage("moe.gate_topk") as _st:
            weights, indices = self.gate(xf)
            _st.add(weights, indices)
        # L893-901: routed experts.  The streamed switch returns the *unweighted*
        # per-expert outputs [n, top_k, dim]; the reference multiplies each
        # expert output by its weight inside the loop (L900) -- done here in one
        # weighted sum, in f32 to match the reference's f32 accumulator (L893).
        # L903: shared expert every token passes through, added in f32.
        # W28/M6: construction selects one prebound execution route.  The
        # overlap route hands shared work to the streamed switch after demand
        # reads are submitted; the control computes the same shared branch after
        # routed output.  Both feed the unchanged f32 combine below.
        routed, shared = self._routed_shared_route(xf, indices)
        with _stime.stage("moe.combine") as _st:
            y = _moe_combine_dispatch(routed, weights, shared, int(xf.shape[0]))
            _st.add(y)
        # L904: return y.type_as(x).view(shape)
        return y.astype(x.dtype).reshape(shape)
