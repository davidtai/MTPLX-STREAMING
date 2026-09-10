"""DeepSeek-V4.1 MoE: routed switch-streaming seam + always-on shared expert.

STUB owned by worker W11 (``feat/deepseek-v41-w11``) per
``docs/deepseek-v41/PORT_CONTRACT.md``.  Until W11's branch lands this is the
faithful port the W10 backbone imports; on integration W11's module (Gate/Expert/
MoE mirroring reference ``model.py`` L797-904 exactly) replaces it.

Reference math (``model.py``):
- ``Gate`` (L797-823): ``scores = sqrt(softplus(x @ w.T / gate_temp))``; top-k of
  ``scores + bias`` (noaux_tc); weights gathered from the UNBIASED scores;
  ``norm_topk_prob`` divide by ``sum(+1e-20)``; ``*= route_scale``.
- ``Expert`` (L826-851): SwiGLU with ``swiglu_limit`` — up (``w3``) clamped
  two-sided, gate (``w1``) clamped from above; ``silu(gate)*up`` in fp32.
- ``MoE`` (L854-904): top-k routed experts + one shared expert every token passes.

Contract surface consumed by W10 + ``sanitize`` + tests:
- ``MoE(args, layer_id)``; ``moe(x[b,s,dim]) -> [b,s,dim]``.
- ``moe.gate`` (``.weight``, ``.e_score_correction_bias``, ``.topk``),
  ``moe.switch_mlp`` (``SwitchGLU`` seam, replaced by ``bind_streamed_switches``),
  ``moe.shared_experts`` (``.w1``/``.w2``/``.w3``).

HARD W11 FINDING (see PORT_CONTRACT.md): the STREAMED q2 expert path in
``expert_mlx.py`` runs a plain ``swiglu`` with NO ``swiglu_limit`` clamp, so at
serve time the routed clamp the reference applies to every expert is dropped.
W11 must make the clamp reach the streamed path (coordinate re: expert_mlx.py).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.switch_layers import SwitchGLU

# Reference-faithful leaves reused from the V4 backend (verified line-by-line
# against inference/model.py in W1_REPORT / the parity numpy oracle):
#   ClampedSwiGLU  -> reference Expert clamp (up two-sided, gate upper), silu(gate)*up
#   MoEGate        -> reference Gate: sqrtsoftplus + noaux_tc top-k + norm + route_scale
from mtplx.models.deepseek_v4 import ClampedSwiGLU, MoEGate


class _SharedExpert(nn.Module):
    """The always-on shared SwiGLU expert (reference ``Expert``, model.py L826-851),
    named ``w1``/``w2``/``w3`` to match the checkpoint.  ``w3`` (up) is clamped
    two-sided, ``w1`` (gate) only from above; the product runs in fp32."""

    def __init__(self, args):
        super().__init__()
        self.limit = args.swiglu_limit
        self.w1 = nn.Linear(args.hidden_size, args.moe_intermediate_size, bias=False)
        self.w3 = nn.Linear(args.hidden_size, args.moe_intermediate_size, bias=False)
        self.w2 = nn.Linear(args.moe_intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        gate = self.w1(x).astype(mx.float32)
        up = self.w3(x).astype(mx.float32)
        if self.limit and self.limit > 0:
            gate = mx.minimum(gate, self.limit)
            up = mx.clip(up, -self.limit, self.limit)
        return self.w2((nn.silu(gate) * up).astype(dtype))


class MoE(nn.Module):
    """Top-6 routed experts + one shared expert (reference ``MoE``, model.py
    L854-904).  ``switch_mlp`` is the seam the streaming runtime rebinds
    (``bind_streamed_switches``); the resident ``SwitchGLU`` is the test-only
    fallback.  The gate (``sqrtsoftplus``/``noaux_tc``) is reused from V4."""

    def __init__(self, args, layer_id: int):
        super().__init__()
        self.gate = MoEGate(args, layer_id)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=ClampedSwiGLU(args.swiglu_limit),
        )
        self.shared_experts = _SharedExpert(args)

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        indices, weights = self.gate(xf)
        routed = self.switch_mlp(xf, indices)  # [n, topk, dim]
        y = (routed * weights[..., None].astype(routed.dtype)).sum(axis=-2)
        y = y + self.shared_experts(xf)
        return y.reshape(shape)


# Reference alias: W10 constructs ``MoE`` but keep the descriptive name importable
# for any caller/test that reaches for it.
DeepseekV41MoE = MoE
