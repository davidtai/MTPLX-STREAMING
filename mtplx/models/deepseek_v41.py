"""DeepSeek-V4.1-Flash text-only autoregressive forward (phase 1).

This module is the MTPLX runtime for the ``deepseek_v41`` / ``deepseek_v41_text``
checkpoint: a 40-layer MoE backbone whose attention is CSA2 (compressed sparse
attention v2) -- a per-layer sliding window plus, on compressing layers, an
indexer-selected set of compressed KV rows shared down the stack.  Phase 1 covers
text AR decode (P1.0-P1.3): config + per-layer mode table, the Hyper-Connection
dense block, MLA attention with the o-LoRA grouped output and the sliding window,
and the CSA2 compressor / indexer / candidate prefilter with its cross-layer
shared runtime and a trim/rollback cache.  MTP/DSpark, vision and the native
FP4/FP8 KV quantisation are out of scope here.

Reuse: the arithmetic that is byte-identical to the older ``deepseek_v4`` backend
is imported from it (YaRN inverse frequencies, interleaved RoPE, the Sinkhorn
loop, the Hyper-Connection ``mixes->pre/post/comb`` split and ``post`` mix, the
routed-expert ``sqrtsoftplus``/``noaux_tc`` gate, and the clamped-SwiGLU
activation).  Everything the V4.1 reference does differently -- the ``pre_mix``
threaded across sublayers, ``sparse_attn`` with an attention sink, the ring
window, and the whole CSA2 cross-layer machinery -- is written fresh here from
``inference/model.py`` (the DeepSeek MIT reference), which is the authority
wherever it disagrees with the port plan.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.base import BaseModelArgs

# Pure arithmetic reused from the V4 backend (imported, not copied), each verified
# line-by-line against inference/model.py and re-checked by the W6 numpy parity
# oracle (tests/models/test_deepseek_v41_parity.py):
#   _yarn_inv_freq         <- precompute_freqs_cis YaRN ramp (model.py L367-395)
#   _apply_interleaved_rope<- apply_rotary_emb adjacent-pair rotation (L399-414)
#   _sinkhorn_ops          <- the stock Sinkhorn alternating-normalisation loop
#                             (kernel.py hc_split_sinkhorn's normalise body); the
#                             CPU / flag-off route and the kernel's parity oracle.
#   _sinkhorn_kernel_apply <- the whole Sinkhorn loop as one mx.fast.metal_kernel
#                             dispatch (deepseek_v4._sinkhorn_metal_kernel, W32/K3);
#                             the GPU flag-on route.  Shapes are identical between
#                             V4 and V4.1 ([..., hc, hc]), so the kernel source is
#                             REUSED here rather than copied; the pre/post/comb
#                             split is carried in ``hc_split_sinkhorn`` below with
#                             this origin note.
#   _hc_post_impl          <- Block.hc_post (post*x + sum_j comb[j,k]*residual[j])
from mtplx.models.deepseek_v4 import (
    _apply_interleaved_rope,
    _hc_post_impl,
    _sinkhorn_kernel_apply,
    _sinkhorn_ops,
    _yarn_inv_freq,
)

# The MoE (routed switch-streaming seam + shared expert) and the per-sequence KV
# cache are ported in sibling modules (workers W11 / W13); see
# docs/deepseek-v41/PORT_CONTRACT.md.  W13's cache owns the window ring
# (append-only history + ring_view), the CompressorState pooling frontier, the
# compress_kv/index_k append stores and the per-cache SharedAttentionRuntime; W10
# reads/writes it through the methods below.  _LayerCache / _SharedRuntime /
# DeepseekV41Cache are re-exported here for the loader + parity tests.
from mtplx.models.deepseek_v41_cache import (
    DeepseekV41Cache,
    _LayerCache,
    _SharedRuntime,
    _grow,
    make_cache as _make_cache,
)
from mtplx.models.deepseek_v41_moe import MoE
from mtplx.models import deepseek_v41_stage_timing as _stime

# Opt-in per-stage attribution (MTPLX_ROUTE_STAGE_PROBE=1). When disabled the only
# cost is one module-level boolean check, so it is safe on the hot path; the A/B
# env-lever census (scripts/deepseek_v41/ab_decode_env_levers.py) reads its
# snapshot to confirm the Sinkhorn kernel actually engaged per arm (W38/K3).
from mtplx import expert_route_probe as _route_probe

# ---------------------------------------------------------------------------
# Hyper-Connection Sinkhorn normalisation (kernel-ledger K3, W32)
# ---------------------------------------------------------------------------
#: Env toggle for the one-dispatch Metal Sinkhorn on the V4.1 Hyper-Connection.
#:
#: Every backbone layer runs the Sinkhorn alternating-normalisation loop twice
#: per token (the attention HC mix and the ffn HC mix, :meth:`DecoderLayer._mixes`
#: -> :func:`hc_split_sinkhorn`), so at 40 layers that is 80 Sinkhorn calls per
#: token.  As stock MLX ops (:func:`_sinkhorn_ops`) each call is ~119 tiny graph
#: primitives -- one row-softmax, then 39 alternating row/column normalisations of
#: ``reduce_sum`` + ``add eps`` + ``divide`` -- on a ``[..., 4, 4]`` tensor that is
#: 16 floats at decode, all of it host build/encode overhead the GPU never notices.
#:
#: Set truthy to route the whole loop through ``deepseek_v4``'s
#: :func:`_sinkhorn_metal_kernel` instead (via :func:`_sinkhorn_kernel_apply`): one
#: threadgroup thread per matrix carries the 16 floats in registers and runs all 40
#: normalisation passes internally, collapsing each call to a single dispatch.  The
#: arithmetic is the *identical* fp32 order as :func:`_sinkhorn_ops` (the V4 lane's
#: parity gate is 1e-6 + argmax-exact), so this is a pure dispatch-count lever.
#:
#: Default OFF until the GPU parity window measures it (V4 measured AR +29.3% on
#: the same kernel).  The kernel path is taken ONLY when the flag is truthy *and*
#: Metal is available *and* the default device is the GPU; CPU (and any no-Metal
#: build) always takes the stock recurrence, so the flag is inert off-GPU.  Read at
#: use, never frozen at import, because the serving harness stamps optimization
#: keys after this module is imported.
_SINKHORN_METAL_ENV = "MTPLX_DSV41_SINKHORN_METAL"

#: Process-cumulative engagement counters (W38/K3): how many Sinkhorn calls took
#: the Metal kernel branch vs the stock recurrence.  Always-on and near-free (one
#: int add), so the A/B env-lever census can tell whether the kernel *actually*
#: engaged during a decode arm or silently fell back to the recurrence -- a
#: byte-identical, barely-faster arm is equally consistent with "kernel ran and is
#: bit-exact" and "kernel never ran".  Under ``mx.compile`` the wrapper runs only
#: during the (cold) trace, so read these cumulatively over a whole arm (which
#: always traces at least once, and prefill runs eager), not as a warm-window
#: delta.  Reset with :func:`_reset_sinkhorn_kernel_calls`.
_SINKHORN_KERNEL_CALLS = 0
_SINKHORN_RECURRENCE_CALLS = 0


def _reset_sinkhorn_kernel_calls() -> None:
    """Zero the Sinkhorn engagement counters (per-arm reset for the A/B census)."""
    global _SINKHORN_KERNEL_CALLS, _SINKHORN_RECURRENCE_CALLS
    _SINKHORN_KERNEL_CALLS = 0
    _SINKHORN_RECURRENCE_CALLS = 0


def _sinkhorn_kernel_calls() -> dict:
    """Snapshot of the engagement counters (kernel vs recurrence Sinkhorn calls)."""
    return {
        "kernel": int(_SINKHORN_KERNEL_CALLS),
        "recurrence": int(_SINKHORN_RECURRENCE_CALLS),
    }


def _sinkhorn_metal_enabled() -> bool:
    """Whether ``MTPLX_DSV41_SINKHORN_METAL`` requests the Metal Sinkhorn.

    Evaluated on every call (not cached at import) so the serving harness's
    late environment stamp is honoured.
    """
    return _env_truthy(_SINKHORN_METAL_ENV)


def _sinkhorn_use_kernel() -> bool:
    """The kernel path is live only when requested *and* on the GPU.

    A truthy flag on a CPU default device (every worker test pins
    ``mx.set_default_device(mx.cpu)``) or a no-Metal build stays on the stock
    recurrence, so the flag can never change CPU numerics.
    """
    if not _sinkhorn_metal_enabled():
        return False
    if not mx.metal.is_available():
        return False
    return mx.default_device() == mx.gpu


def _sinkhorn_normalise(comb: mx.array, hc: int, iters: int, eps: float) -> mx.array:
    """Doubly-stochastic Sinkhorn normalise of the ``[..., hc, hc]`` comb tensor.

    Dispatches to the one-launch Metal kernel when :func:`_sinkhorn_use_kernel`
    (flag on + GPU), otherwise the stock alternating-normalisation recurrence.
    Both accept any leading dims (``rows = b*s`` = any n): the kernel flattens the
    leading axes to one matrix index and reshapes back, so decode, one-shot
    prefill, chunked prefill and the layer-major path all compose unchanged.

    The kernel is validated only for fp32 (V4's proven lane; production ``comb`` is
    fp32 because :meth:`DecoderLayer._mixes` casts it), and it already computes the
    whole schedule in fp32 registers regardless of I/O dtype.  A non-fp32 ``comb``
    (only ever a test/edge case) is therefore upcast to fp32 for the kernel and the
    result cast back, so the Metal path never processes a bf16 buffer and stays at
    its measured precision; the recurrence stays dtype-native.  Engagement is
    counted either way (see :data:`_SINKHORN_KERNEL_CALLS`).
    """
    global _SINKHORN_KERNEL_CALLS, _SINKHORN_RECURRENCE_CALLS
    if _sinkhorn_use_kernel():
        _SINKHORN_KERNEL_CALLS += 1
        _route_probe.count("hc.sinkhorn_kernel")
        if comb.dtype != mx.float32:
            out = _sinkhorn_kernel_apply(comb.astype(mx.float32), hc, iters, eps)
            return out.astype(comb.dtype)
        return _sinkhorn_kernel_apply(comb, hc, iters, eps)
    _SINKHORN_RECURRENCE_CALLS += 1
    _route_probe.count("hc.sinkhorn_recurrence")
    return _sinkhorn_ops(comb, iters, eps)


def hc_split_sinkhorn(
    mixes: mx.array,
    scale: mx.array,
    base: mx.array,
    hc: int,
    iters: int,
    eps: float,
):
    """V4.1 pre/post/comb split with the Sinkhorn route selected per device.

    This is the module's canonical Sinkhorn-split boundary: **both** the eager
    :meth:`DecoderLayer._mixes` and the compiled :func:`_hc_mixes_split` (K4 HC
    tape) call it by this module-global name, so the K3 Metal kernel drops into
    the compiled tape as well as the eager path, and a test can monkeypatch it to
    census graph construction (tests/models/test_deepseek_v41_hc_compile.py).

    Byte-for-byte the split of ``deepseek_v4.hc_split_sinkhorn`` (origin:
    ``inference/kernel.py`` ``hc_split_sinkhorn_kernel`` L371-427, transcribed in
    ``deepseek_v4``); the only change is that the always-stock
    ``_sinkhorn_ops(comb, iters, eps)`` tail is replaced by
    :func:`_sinkhorn_normalise`, which takes the Metal kernel when it is armed on
    the GPU and the identical stock recurrence otherwise.  Returns ``(pre, post,
    comb)`` with shapes ``[..., hc]``, ``[..., hc]``, ``[..., hc, hc]``.
    """
    pre = mx.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * mx.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]
    comb = comb.reshape(*comb.shape[:-1], hc, hc)  # [..., j, k]
    comb = _sinkhorn_normalise(comb, hc, iters, eps)
    return pre, post, comb


# ---------------------------------------------------------------------------
# Per-layer CSA2 mode (§0 of docs/deepseek-v41/PORT_PLAN.md)
# ---------------------------------------------------------------------------
#: A pure sliding-window attention layer: ``compress_ratios[L] == 0``, no
#: compressor and no indexer (encoder-local layers 0/1 and the MTP tail).
MODE_SWA_ONLY = "swa_only"
#: Owns this group's global compressed KV *and* its index keys/queries; the first
#: layer of a ``compress_ratio`` run and a member of ``kv_source_layer_ids``.
MODE_FULL = "full"
#: Owns its own indexer queries but reuses an earlier layer's compressed KV; a
#: member of ``index_source_layer_ids`` that is not a ``kv_source`` layer.
MODE_REINDEX = "reindex"
#: Reuses both the compressed KV and the selected top-K of an earlier source.
MODE_REUSE = "reuse"


@dataclass
class ModelArgs(BaseModelArgs):
    """DeepSeek-V4.1-Flash text config (field names are the ``text_config`` keys).

    Defaults are the released 40-layer shapes; tests build small instances by
    overriding the shape fields.  ``__post_init__`` mirrors the HF ``rope_scaling``
    block into the flat YaRN fields the RoPE tables read and precomputes the
    per-layer mode table.
    """

    model_type: str = "deepseek_v41"
    vocab_size: int = 129280
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    # moe
    moe_intermediate_size: int = 2304
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    #: V4.1 has no hash-routed layers (engram replaces them); kept so the reused
    #: ``MoEGate`` always takes its score-routed branch.
    num_hash_layers: int = 0
    # attention (MQA-shaped MLA)
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    sliding_window: int = 128
    window_size: int = 128
    # CSA2 compressor / indexer
    compress_ratios: List[int] = field(default_factory=list)
    compress_rope_theta: float = 160000.0
    kv_source_layer_ids: List[int] = field(default_factory=list)
    index_source_layer_ids: List[int] = field(default_factory=list)
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer_id: int = -1
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0
    # hyper-connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # norm / rope / yarn
    rms_norm_eps: float = 1e-20
    rope_theta: float = 10000.0
    max_position_embeddings: int = 1048576
    rope_scaling: Optional[dict] = None
    original_seq_len: int = 65536
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    # engram (hooked externally; only the layer ids are read here)
    engram_layer_ids: List[int] = field(default_factory=list)
    tie_word_embeddings: bool = False
    # DSpark MTP draft head (worker W23; consumed by mtplx.models.deepseek_v41_dspark).
    # ``n_mtp_layers`` == ``num_nextn_predict_layers`` == 3 in the released config;
    # the draft head is built only on the opt-in ``mtp=True`` load path.
    n_mtp_layers: int = 0
    num_nextn_predict_layers: int = 0
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: List[int] = field(default_factory=list)
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 128
    dspark_num_experts_per_tok: int = 3
    dspark_n_activated_experts: int = 0

    def __post_init__(self):
        rs = self.rope_scaling or {}
        if rs:
            self.original_seq_len = int(
                rs.get("original_max_position_embeddings", self.original_seq_len)
            )
            self.rope_factor = float(rs.get("factor", self.rope_factor))
            self.beta_fast = int(rs.get("beta_fast", self.beta_fast))
            self.beta_slow = int(rs.get("beta_slow", self.beta_slow))
        self.window_size = int(self.sliding_window or self.window_size)
        if not self.compress_ratios:
            self.compress_ratios = [0] * self.num_hidden_layers
        self.layer_modes = _derive_layer_modes(self)
        # DSpark: ``n_mtp_layers`` and ``num_nextn_predict_layers`` are the same
        # count under two config spellings; keep both fields agreeing so either
        # source builds the head.
        stages = int(self.n_mtp_layers or self.num_nextn_predict_layers or 0)
        self.n_mtp_layers = stages
        self.num_nextn_predict_layers = stages
        if not self.dspark_num_experts_per_tok:
            self.dspark_num_experts_per_tok = int(self.dspark_n_activated_experts or 3)

    @classmethod
    def from_dict(cls, params: dict) -> "ModelArgs":
        """Accept either the top-level ``deepseek_v41`` config (with a nested
        ``text_config``) or a bare ``deepseek_v41_text`` dict.  The outer runtime
        key ``deepseek_v41`` wins over the sub-config's ``deepseek_v41_text``."""
        outer_type = params.get("model_type")
        inner = params.get("text_config")
        merged = dict(inner) if inner is not None else dict(params)
        if inner is not None and outer_type:
            merged["model_type"] = outer_type
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in merged.items() if k in fields})


def _derive_layer_modes(args: ModelArgs) -> List[str]:
    """The §0 mode per backbone layer from the three CSA2 source-id sets.

    ``compress_ratios[L] == 0`` is a pure sliding-window layer; otherwise a layer
    in ``kv_source_layer_ids`` owns the group's KV (Full), a layer only in
    ``index_source_layer_ids`` owns its indexer queries (Reindex), and the rest
    reuse an earlier source (Reuse).
    """
    kv = set(args.kv_source_layer_ids)
    idx = set(args.index_source_layer_ids)
    modes: List[str] = []
    for layer_id in range(args.num_hidden_layers):
        ratio = args.compress_ratios[layer_id]
        if ratio == 0:
            modes.append(MODE_SWA_ONLY)
        elif layer_id in kv:
            modes.append(MODE_FULL)
        elif layer_id in idx:
            modes.append(MODE_REINDEX)
        else:
            modes.append(MODE_REUSE)
    return modes


# ---------------------------------------------------------------------------
# Numeric leaves (fp32, transcribed from inference/model.py)
# ---------------------------------------------------------------------------
def _rmsnorm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Reference ``RMSNorm.forward`` (model.py L288-293): normalise in fp32, then
    scale by ``weight`` and cast back to the input dtype."""
    dtype = x.dtype
    xf = x.astype(mx.float32)
    var = mx.mean(mx.square(xf), axis=-1, keepdims=True)
    xf = xf * mx.rsqrt(var + eps)
    return (weight.astype(mx.float32) * xf).astype(dtype)


def _swa_inv_freq(args: ModelArgs) -> mx.array:
    """Plain (no-YaRN) inverse frequencies for a pure sliding-window layer."""
    rd = args.qk_rope_head_dim
    return 1.0 / (args.rope_theta ** (mx.arange(0, rd, 2, dtype=mx.float32) / rd))


def _compress_inv_freq(args: ModelArgs) -> mx.array:
    """YaRN inverse frequencies at ``compress_rope_theta`` for a compressing layer."""
    return _yarn_inv_freq(
        args.qk_rope_head_dim,
        args.compress_rope_theta,
        args.original_seq_len,
        args.rope_factor,
        args.beta_fast,
        args.beta_slow,
    )


def _cos_sin(inv_freq, positions: mx.array):
    """``cos``/``sin`` tables ``[len(positions), rope_head_dim//2]`` in fp32.

    ``inv_freq`` is a numpy constant (kept off the parameter tree); it is lifted
    to MLX here."""
    freq = mx.array(np.asarray(inv_freq, dtype=np.float32))
    ang = positions.astype(mx.float32)[:, None] * freq[None, :]
    return mx.cos(ang), mx.sin(ang)


def _rope_last(x: mx.array, cos: mx.array, sin: mx.array, inverse: bool = False) -> mx.array:
    """RoPE the last ``2*half`` dims of ``x`` as adjacent complex pairs, matching
    reference ``apply_rotary_emb``.  ``cos``/``sin`` are ``[s, half]`` and are
    reshaped to broadcast against ``x``'s leading axes; ``inverse`` conjugates
    (cos, -sin), which removes the query rotation from the attention output."""
    rd = cos.shape[-1] * 2
    head = x[..., :-rd]
    tail = x[..., -rd:]
    # x is [batch, seq, *head_axes, rope]; cos/sin are [seq, half] and must align
    # the seq axis (from the right, past any head axes) with x's seq axis.
    extra = tail.ndim - 3  # head-like axes between seq and the rope pair
    shape = [cos.shape[0]] + [1] * extra + [cos.shape[-1]]
    c = cos.reshape(shape)
    s = sin.reshape(shape)
    if inverse:
        s = -s
    roped = _apply_interleaved_rope(tail, c, s)
    if head.shape[-1] == 0:
        return roped
    return mx.concatenate([head, roped], axis=-1)


# The cross-layer attention runtime (reference SharedAttentionRuntime) is W13's
# ``_SharedRuntime`` (imported above): one slot each for the group's compress_kv /
# index_k, the index source's ``topk_mask`` (alias of ``topk_idxs``) and the
# candidate-source's ``candidates``.  W10 creates one per forward via
# ``cache.new_shared_runtime()``.


# ---------------------------------------------------------------------------
# CSA2: compressor, indexer, candidate prefilter
# ---------------------------------------------------------------------------
class Compressor(nn.Module):
    """Gated pooling of ``compress_ratio`` tokens into one pre-RoPE KV latent
    (reference ``Compressor``, model.py L429-485).

    ``ratio == 1`` is a plain per-token projection (no gate); ``ratio > 1`` pools
    each window with a softmax over ``wgate`` scores.  Prefill drops the trailing
    ``s % ratio`` remainder (the reference parks it in decode state; here the
    partial-group state lives in the cache).  Returns the normalised latent before
    RoPE, which the indexer needs unrotated.
    """

    def __init__(self, args: ModelArgs, ratio: int):
        super().__init__()
        self.ratio = ratio
        self.norm_weight = mx.ones((args.head_dim,))
        self.eps = args.rms_norm_eps
        self.wkv = nn.Linear(args.hidden_size, args.head_dim, bias=False)
        if ratio > 1:
            self.wgate = nn.Linear(args.hidden_size, args.head_dim, bias=False)

    def pool(self, x: mx.array, comp_state):
        """The pre-RoPE, normed compressed latents this call completes.

        ``ratio == 1`` is a plain per-token projection with no state (reference
        ``Compressor.forward`` L461-462): every token yields one latent, so a
        whole-chunk prefill and a one-token decode step are the same code.

        ``ratio > 1`` (reference L463-485) projects ``wkv``/``wgate`` in fp32 and
        hands the rows to W13's :class:`~mtplx.models.deepseek_v41_cache.CompressorState`,
        which retains the running frontier and returns the softmax-gated pooled
        latents (reference L475 / L482) for whatever groups this call completed --
        one code path for the prefill ``floor(s/ratio)`` groups + parked remainder
        and the decode "pool when the group just filled".  Returns ``None`` when
        this call completed no group (a still-filling decode step).
        """
        if self.ratio == 1:
            return _rmsnorm(self.wkv(x), self.norm_weight, self.eps)
        xf = x.astype(mx.float32)
        pooled = comp_state.push(self.wkv(xf), self.wgate(xf))  # [b, g, head_dim] fp32
        if pooled.shape[1] == 0:
            return None
        return _rmsnorm(pooled, self.norm_weight, self.eps)


class Indexer(nn.Module):
    """The side-attention that scores compressed rows so each query keeps only
    ``index_topk`` of them (reference ``Indexer``, model.py L488-580).

    A ``kv_source`` layer (``owns_k``) turns the compressor latent into the index
    keys; every indexer layer projects its own queries, scores queries against the
    keys (ReLU'd per head, weighted by ``weights_proj``), masks unreachable and
    (downstream of the candidate source) out-of-candidate rows, and returns the
    per-query selected-row mask.
    """

    def __init__(self, args: ModelArgs, owns_k: bool):
        super().__init__()
        self.owns_k = owns_k
        self.n_heads = args.index_n_heads
        self.index_head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.softmax_scale = args.index_head_dim ** -0.5
        self.eps = args.rms_norm_eps
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.index_head_dim, bias=False)
        self.weights_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        if owns_k:
            self.wk = nn.Linear(args.head_dim, self.index_head_dim, bias=False)
            self.k_norm_weight = mx.ones((self.index_head_dim,))

    def keys(self, latent: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        """Index keys from a pre-RoPE compressor latent: wk -> k_norm -> RoPE tail.
        Only defined on ``owns_k`` layers (reference model.py L544-546)."""
        k = _rmsnorm(self.wk(latent), self.k_norm_weight, self.eps)
        return _rope_last(k, cos, sin)

    def select(self, x, qr, index_k, q_cos, q_sin, compress_lens, n_comp,
               candidates=None, set_candidates=False, cand_topk=0, cand_block=0):
        """Return (topk_mask [b,s,n_comp] bool, candidates_or_None).

        ``compress_lens`` is ``[s]`` int (reachable compressed-row count per query),
        ``n_comp`` the compressed rows available.  When ``set_candidates`` this is
        the candidate source and it also returns the candidate-block mask."""
        b, s, _ = x.shape
        H, D = self.n_heads, self.index_head_dim
        q = self.wq_b(qr).reshape(b, s, H, D)
        q = _rope_last(q, q_cos, q_sin)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        score = mx.einsum("bshd,btd->bsht", q.astype(mx.float32), index_k.astype(mx.float32))
        score = mx.maximum(score, 0.0) * weights.astype(mx.float32)[..., None]
        score = mx.sum(score, axis=2)  # [b, s, n_comp]

        reach = mx.arange(n_comp)[None, None, :] < compress_lens[None, :, None]
        score = mx.where(reach, score, -mx.inf)

        cand_out = None
        if set_candidates:
            cand_out = _select_candidate_blocks(score, compress_lens, cand_topk, cand_block)
        elif candidates is not None:
            score = mx.where(candidates, score, -mx.inf)

        topk = min(self.index_topk, n_comp)
        mask = _topk_rows(score, topk)
        mask = mask & reach  # unreachable rows are never selected
        return mask, cand_out


def _topk_rows(score: mx.array, k: int) -> mx.array:
    """Boolean ``[..., n]`` mask of the ``k`` highest ``score`` entries per row.

    Matches ``score.topk(k).indices`` set membership; ties (equal scores, common
    when every head ReLU'd to 0) are broken toward the lowest index, as the
    reference selection does, so the streamed and one-shot row-length paths agree.
    """
    n = score.shape[-1]
    if k >= n:
        return score > -mx.inf
    ranked = mx.sort(score, axis=-1)[..., ::-1]
    thr = ranked[..., k - 1:k]  # [..., 1], the kth largest per row
    gt = score > thr
    eq = score == thr
    n_gt = mx.sum(gt.astype(mx.int32), axis=-1, keepdims=True)
    tie_rank = mx.cumsum(eq.astype(mx.int32), axis=-1) - 1
    return gt | (eq & (tie_rank < (k - n_gt)))


def _select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    """Level-one block prefilter (reference ``select_candidate_blocks`` L583-610).

    Keeps the ``topk_blocks`` highest-scoring blocks of ``block_size`` compressed
    positions per query (each block scored by its best position); the block holding
    the query's newest position is pinned in.  Returns a ``[b,s,n_comp]`` bool mask.
    """
    b, s, width = logits.shape
    pad = (-width) % block_size
    if pad:
        logits = mx.concatenate(
            [logits, mx.full((b, s, pad), -mx.inf, dtype=logits.dtype)], axis=-1
        )
    num_blocks = logits.shape[-1] // block_size
    scores = mx.max(logits.reshape(b, s, num_blocks, block_size), axis=-1)  # [b,s,num_blocks]

    last = (compress_lens - 1) // block_size  # [s]
    pin = mx.arange(num_blocks)[None, None, :] == last[None, :, None]
    scores = mx.where(pin, mx.array(mx.inf, dtype=scores.dtype), scores)

    kb = min(topk_blocks, num_blocks)
    keep_blocks = _topk_rows(scores, kb)
    keep_blocks = keep_blocks & (scores > -mx.inf)  # drop unreachable filler picks
    keep = mx.repeat(keep_blocks, block_size, axis=-1)[..., :width]
    return keep


# ---------------------------------------------------------------------------
# Attention (MLA + o-LoRA + sliding window + CSA2)
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    """MLA attention: one shared KV latent (``head_dim`` 512, 1 KV head, 64 query
    heads) over a sliding window plus, on CSA2 layers, an indexer-selected set of
    compressed rows -- a single softmax with a per-head attention sink -- then the
    grouped o-LoRA output projection (reference ``Attention``, model.py L613-789).
    """

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.n_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.window_size = args.window_size
        self.eps = args.rms_norm_eps
        self.softmax_scale = args.head_dim ** -0.5
        self.compress_ratio = args.compress_ratios[layer_id]
        self.mode = args.layer_modes[layer_id]
        self.is_kv_source = layer_id in args.kv_source_layer_ids
        self.is_index_source = layer_id in args.index_source_layer_ids
        self.is_candidate_source = layer_id == args.candidate_source_layer_id
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        # test-only capture of the selected compressed rows / candidate mask
        self.capture_selection = False
        self.last_selection = None
        self.last_candidates = None

        self.attn_sink = mx.zeros((self.n_heads,))
        self.wq_a = nn.Linear(self.dim, args.q_lora_rank, bias=False)
        self.q_norm_weight = mx.ones((args.q_lora_rank,))
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm_weight = mx.ones((self.head_dim,))
        in_per_group = self.n_heads * self.head_dim // self.n_groups
        # wo_a is a weight holder (block-diagonal over o_groups, applied as an
        # einsum, not a plain GEMM); an nn.Linear so nn.quantize can make it q8.
        self.wo_a = nn.Linear(in_per_group, self.n_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=False)

        self.compressor = Compressor(args, self.compress_ratio) if self.is_kv_source else None
        self.indexer = Indexer(args, owns_k=self.is_kv_source) if self.is_index_source else None

        # RoPE inverse frequencies are a derived constant, not a checkpoint
        # tensor: store as numpy so MLX does not register it as a parameter.
        if self.compress_ratio:
            self.inv_freq = np.asarray(_compress_inv_freq(args), dtype=np.float32)
        else:
            self.inv_freq = np.asarray(_swa_inv_freq(args), dtype=np.float32)

    def _sparse_attend(self, q, KV, attend):
        """One softmax over the concatenated KV with a per-head sink (value 0),
        equivalent to reference ``sparse_attn``.  q: [b,s,H,hd], KV: [b,T,hd],
        attend: [b,s,T] bool.

        Prefill (query rows ``q.shape[1] > 1``) reads the W50/K25 score-path
        levers -- ``MTPLX_DSV41_PREFILL_SCORE_PATH`` (oneshot|lean),
        ``MTPLX_DSV41_PREFILL_SCORE_DTYPE`` (f32|bf16, the QK^T/PV matmul input
        dtype) and ``MTPLX_DSV41_PREFILL_SCORE_KEY_CHUNK`` (the split-K online
        softmax chunk width, off by default).  ``lean`` is the W50 pass-cut f32
        one-shot path (window-20 showed the score stage is pass/bandwidth-bound, not
        matmul-FLOP-bound: bf16 was -34%, split-K -16%); it forces f32 and ignores
        the dtype/chunk knobs.  Decode / M=1 (``s == 1``) always runs the shipped
        f32 one-shot path, byte-identical to control regardless of the env.

        W60/K29: on the GPU with ``MTPLX_DSV41_DECODE_ATTN_KERNEL=1`` the M=1 decode
        and the small-M (``K+1``) verify batch route through the fused decode
        attention Metal kernel (score + mask + sink softmax + PV in ONE dispatch);
        an unsupported mask shape or a CPU-pinned host falls back to the eager path.
        The hook is a small guarded early return -- the prefill score path (W58's
        one-shot/lean/chunked, W59's key selection) is untouched (rows > the
        small-M cap never enter it)."""
        if _decode_attn_kernel_use(q):
            out = self._decode_attn_kernel(q, KV, attend)
            if out is not None:
                return out
        if q.shape[1] <= 1:
            return self._sparse_attend_oneshot(q, KV, attend, mx.float32)
        path = _resolve_prefill_score_path()
        if path == "lean":
            # f32-only pass-cut one-shot (scale-into-q + folded sink, no concat/slice)
            return self._sparse_attend_oneshot(
                q, KV, attend, mx.float32, fuse_scale=True, fold_sink=True
            )
        score_dtype = _resolve_prefill_score_dtype()
        key_chunk = _resolve_prefill_score_key_chunk()
        if key_chunk is not None:
            return self._sparse_attend_chunked(q, KV, attend, score_dtype, key_chunk)
        return self._sparse_attend_oneshot(q, KV, attend, score_dtype)

    def _decode_attn_kernel(self, q, KV, attend):
        """W60/K29 fused decode / verify MLA attention (one Metal dispatch per
        layer).  Returns the ``[b,s,H,hd]`` f32 output, or ``None`` when the mask
        shape is unsupported so the caller falls back to the eager one-shot.

        Mode-agnostic: it consumes the already-assembled ``[b,s,T]`` boolean (or
        additive-f32) ``attend`` mask shared across heads -- exactly what all four
        CSA modes produce here (swa_only's window mask, and full/reindex/reuse's
        ``concatenate([window_mask, comp_attend])``) -- so it never re-derives the
        CSA candidate selection.  MLA: ``KV`` is the one shared latent, passed as
        both key and value.  A per-head mask (``ndim != 3``) or a shape mismatch is
        the sole eager-fallback path (returns ``None``); genuine kernel errors are
        left to propagate ([[dont-rationalize-broken-as-normal]])."""
        from mtplx.models import deepseek_v41_attn_kernels as _k29
        if attend is not None:
            if attend.ndim != 3 or tuple(attend.shape) != (
                int(q.shape[0]), int(q.shape[1]), int(KV.shape[1])
            ):
                _k29.note_fallback()  # W60 telemetry: armed-but-eager (mask shape)
                return None  # unsupported mask shape -> eager
        return _k29.fused_decode_attention(
            q, KV, KV, attend=attend, attn_sink=self.attn_sink,
            scale=self.softmax_scale, T=int(KV.shape[1]),
        )

    def _sparse_attend_oneshot(self, q, KV, attend, score_dtype,
                               *, fuse_scale=False, fold_sink=False):
        """One-shot softmax over the full concatenated KV (the shipped path).

        ``score_dtype`` casts the QK^T / PV matmul *inputs*; MLX matmul accumulates
        in f32 internally and rounds the result back to ``score_dtype``, so the
        scale, mask, per-head sink and softmax stay in f32 exactly as the reference
        oracle -- the only numerical change under ``bf16`` is the two matmuls'
        input+output bf16 rounding.  ``score_dtype == float32`` with both toggles
        off inserts no extra ops and is byte-identical to control.

        W50 pass-cut toggles (the f32 ``lean`` path; window-20 found the score stage
        pass/bandwidth-bound, not FLOP-bound, so cut passes over the ``[rows,H,T]``
        transient, not FLOPs).  Both are reassociation-level vs control (greedy-
        identical), never bit-identical:
          * ``fuse_scale`` -- scale ``q`` once (``[rows,H,512]``) instead of the
            scores (``[rows,H,T]``): drops one pass over the T-wide transient.
          * ``fold_sink`` -- softmax done manually with the per-head value-0 sink in
            the denominator (reference ``_k_sparse_attn``): no sink concatenate
            (``[rows,H,T+1]`` alloc) and no post-softmax slice (``[rows,H,T]`` copy),
            and the normalize divides the ``[rows,H,512]`` output, not the T-wide w.

        The sub-brackets (``stage_attn``) decompose ``attn.<mode>.score`` into
        qk_matmul / cast / scale_mask_sink / softmax / pv_matmul for the next
        prefill window (no-op / byte-identical when not timing)."""
        b, s, H, _ = q.shape
        Tk = KV.shape[1]
        # The DSpark draft head reuses this method via ``DSparkAttention`` (which
        # has no CSA ``mode``); label its stage_attn sub-brackets ``attn.dspark.*``.
        mode = getattr(self, "mode", "dspark")
        scale = self.softmax_scale
        with _stime.stage_attn("attn." + mode + ".score.qk_matmul") as _st:
            qd = (q * scale) if fuse_scale else q
            scores = mx.einsum(
                "bshd,btd->bsht", qd.astype(score_dtype), KV.astype(score_dtype)
            )
            _st.add(scores)
        if score_dtype != mx.float32:
            with _stime.stage_attn("attn." + mode + ".score.cast") as _st:
                scores = scores.astype(mx.float32)
                _st.add(scores)
        # W58 / K28: prefill-only fused mask + per-head value-0 sink + f32 softmax
        # in ONE Metal dispatch (reads the [rows,H,T] transient twice, writes the
        # normalised probabilities once; no T-wide masked_scores/ex/concat
        # intermediate).  Armed only on the GPU with the flag on and s > 1 (decode
        # /M=1 always runs the eager path, byte-identical); composes with the lean
        # path (scale is folded into q, so the kernel scales by 1.0) and the plain
        # one-shot (kernel applies softmax_scale, folding the scale pass too).  The
        # kernel folds the sink like ``fold_sink`` regardless, so K28-on is
        # reassociation-level vs control (<=1e-6, greedy-identical), NOT byte-
        # identical -- the split-K/chunked path is NOT routed here (W58 report).
        if s > 1 and _prefill_softmax_kernel_use():
            from mtplx.kernels.dsv41_fused_softmax import fused_prefill_softmax
            k_scale = 1.0 if fuse_scale else scale
            with _stime.stage_attn("attn." + mode + ".score.fused_softmax_kernel") as _st:
                p = fused_prefill_softmax(
                    scores, attend=attend, attn_sink=self.attn_sink, scale=k_scale,
                )
                _st.add(p)
            with _stime.stage_attn("attn." + mode + ".score.pv_matmul") as _st:
                o = mx.einsum("bsht,btd->bshd", p, KV.astype(mx.float32))
                _st.add(o)
            return o
        with _stime.stage_attn("attn." + mode + ".score.scale_mask_sink") as _st:
            if not fuse_scale:
                scores = scores * scale
            scores = mx.where(attend[:, :, None, :], scores, float("-inf"))
            _st.add(scores)
        sink = self.attn_sink.astype(mx.float32).reshape(1, 1, H, 1)
        with _stime.stage_attn("attn." + mode + ".score.softmax") as _st:
            if fold_sink:
                # manual softmax with the value-0 sink in the denom (reference
                # _k_sparse_attn L149-153): normalize AFTER PV (divide [rows,H,512]).
                m = mx.maximum(mx.max(scores, axis=-1, keepdims=True), sink)
                ex = mx.exp(scores - m)                    # masked -> exp(-inf) = 0
                denom = mx.sum(ex, axis=-1, keepdims=True) + mx.exp(sink - m)
                _st.add(ex, denom)
            else:
                sink_b = mx.broadcast_to(sink, (b, s, H, 1))
                full = mx.concatenate([scores, sink_b], axis=-1)
                w = mx.softmax(full, axis=-1)[..., :Tk]  # drop the sink column
                _st.add(w)
        with _stime.stage_attn("attn." + mode + ".score.pv_matmul") as _st:
            if fold_sink:
                o = mx.einsum("bsht,btd->bshd", ex, KV.astype(mx.float32)) / denom
            elif score_dtype == mx.float32:
                o = mx.einsum("bsht,btd->bshd", w, KV.astype(mx.float32))
            else:
                o = mx.einsum(
                    "bsht,btd->bshd", w.astype(score_dtype), KV.astype(score_dtype)
                ).astype(mx.float32)
            _st.add(o)
        return o

    def _sparse_attend_chunked(self, q, KV, attend, score_dtype, key_chunk):
        """Two-pass split-K online softmax (gemma4 D512 two-pass / K6) over
        ``key_chunk``-wide key blocks: caps the score transient at
        ``[rows, H, key_chunk]`` instead of the full ``[rows, H, T]``.

        Mathematically identical to :meth:`_sparse_attend_oneshot`: the per-element
        QK^T dot products are bit-identical (the matmul reduces over ``head_dim``,
        NOT the chunked ``T`` axis), so only the softmax-denominator sum and the
        value accumulation reassociate across chunks -- in f32 the sole difference
        from one-shot is float reassociation.  The per-head value-0 sink seeds the
        running state (``m = attn_sink`` finite, ``denom = exp(0) = 1``, ``acc =
        0``), so a fully-masked chunk (all scores ``-inf``) produces ``corr =
        exp(m-m) = 1`` and ``p = 0`` -- never a ``-inf − (−inf)`` NaN."""
        b, s, H, hd = q.shape
        T = KV.shape[1]
        # DSparkAttention (draft head) reuses this method and has no CSA ``mode``.
        mode = getattr(self, "mode", "dspark")
        qd = q.astype(score_dtype)
        KVd = KV.astype(score_dtype)
        m = mx.broadcast_to(
            self.attn_sink.astype(mx.float32).reshape(1, 1, H, 1), (b, s, H, 1)
        )
        denom = mx.ones((b, s, H, 1), dtype=mx.float32)   # sink: exp(attn_sink - m) = 1
        acc = mx.zeros((b, s, H, hd), dtype=mx.float32)   # sink value is 0
        # Sub-brackets accumulate by name across all chunks (stage_attn is a no-op
        # off the prefill probe; the per-chunk online-softmax rescale of ``acc`` is
        # an extra O(rows*H*hd) pass EVERY chunk -- the cost W47/window-20 exposes).
        for c0 in range(0, T, key_chunk):
            c1 = min(c0 + key_chunk, T)
            KVc = KVd[:, c0:c1, :]
            att_c = attend[:, :, c0:c1]
            with _stime.stage_attn("attn." + mode + ".score.qk_matmul") as _st:
                sc = mx.einsum("bshd,btd->bsht", qd, KVc).astype(mx.float32) * self.softmax_scale
                _st.add(sc)
            with _stime.stage_attn("attn." + mode + ".score.scale_mask_sink") as _st:
                sc = mx.where(att_c[:, :, None, :], sc, float("-inf"))
                _st.add(sc)
            with _stime.stage_attn("attn." + mode + ".score.online_softmax") as _st:
                m_c = mx.max(sc, axis=-1, keepdims=True)
                m_new = mx.maximum(m, m_c)
                corr = mx.exp(m - m_new)
                p = mx.exp(sc - m_new)
                denom = denom * corr + mx.sum(p, axis=-1, keepdims=True)
                _st.add(p, denom, corr)
            with _stime.stage_attn("attn." + mode + ".score.pv_matmul") as _st:
                if score_dtype == mx.float32:
                    pv = mx.einsum("bsht,btd->bshd", p, KVc)
                else:
                    pv = mx.einsum(
                        "bsht,btd->bshd", p.astype(score_dtype), KVc
                    ).astype(mx.float32)
                acc = acc * corr + pv   # O(rows*H*hd) rescale of the running output
                _st.add(acc)
            m = m_new
        with _stime.stage_attn("attn." + mode + ".score.combine") as _st:
            out = acc / denom
            _st.add(out)
        return out

    def _window_selected_idx(self, positions, T):
        """This layer's sliding-window keys as gather indices ``[s, W]`` into the
        full-history window store (row j == absolute token j), with a ``[s, W]``
        valid mask.  Reproduces the reference ``get_window_topk_idxs`` (model.py
        L409-426) in the port's absolute-position frame: query at position ``p``
        attends ``{max(0, p-W+1) .. p}`` (``idxs = clamp(p-W+1, 0) + arange(W)``,
        future slots ``idx > p`` marked invalid) -- exactly the set the
        :meth:`_window_attend` causal band mask keeps, so the selected-gather path
        attends the identical window keys as the masked-full path."""
        W = self.window_size
        qp = positions.reshape(-1, 1)                     # [s, 1] absolute positions
        base = mx.maximum(qp - (W - 1), 0)
        idx = base + mx.arange(W).reshape(1, W)           # [s, W]
        valid = (idx <= qp) & (idx < T)                    # not future / in store
        return idx.astype(mx.int32), valid

    def _sparse_attend_selected(self, q, window_all, compress_kv, comp_idx,
                                positions):
        """K30 (W59) selected-key gather attention -- the faithful,
        non-transliterated form of :meth:`_sparse_attend`, for prefill (rows > 1),
        decode (rows == 1) and the ``K+1`` verify batch alike.

        Instead of scoring the full concatenated ``[b, T+n_comp, hd]`` history and
        masking (score transient ``[rows, H, T+n_comp]`` growing with T -- at decode
        the whole compressed history is re-scored every token), this gathers only
        the keys/values each query attends -- its sliding window
        (``_window_selected_idx``) plus the indexer's ``index_topk`` selected
        compressed rows (``comp_idx``, published by the index source) -- into one
        compact ``[rows, k, hd]`` operand and runs a single softmax over ``k``
        keys, exactly as the reference ``sparse_attn`` gathers ``kv[topk_idxs]``.
        ``k = window + min(index_topk, n_comp)`` saturates independent of T (640 /
        128), so the per-layer decode attention becomes T-independent.

        The KV is shared across all 64 query heads (MLA, 1 KV head), so the gather
        is ``[rows, k, hd]`` (not per head) -- the least-traffic layout.

        **K30 x K29 composition:** when ``MTPLX_DSV41_DECODE_ATTN_KERNEL`` is also
        armed on a Metal host at decode / verify (``b*s <= 8``), the compact gathered
        operands are handed to W60's fused decode-attention kernel -- each query row
        becomes its own single-query "batch" (``[rows, 1, H, hd]`` / KV
        ``[rows, k, hd]``), so the kernel's ``row // S`` batch map indexes each
        query's own gathered KV, scoring + sink-softmax + PV over ``k`` keys in ONE
        dispatch (mask-free but for the ``[rows, k]`` valid-count edge).  Otherwise
        (CPU host, kernel off, or prefill rows > 8) the eager gathered softmax runs
        here: the reference ``_k_sparse_attn`` value-0 sink form (max includes the
        sink, normalize after PV), all f32.  Either way, vs the masked-full path the
        only difference is float reassociation over a different key ordering
        (greedy-identical, never bit-identical)."""
        b, s, H, hd = q.shape
        mode = getattr(self, "mode", "dspark")
        with _stime.stage_attn("attn." + mode + ".score.gather") as _st:
            win_idx, win_valid = self._window_selected_idx(positions, window_all.shape[1])
            win_idx = mx.broadcast_to(win_idx[None], (b, s, win_idx.shape[-1]))
            win_valid = mx.broadcast_to(win_valid[None], (b, s, win_valid.shape[-1]))
            kvg_win = _gather_rows(window_all, win_idx, win_valid)   # [b,s,W,hd]
            if compress_kv is not None and comp_idx is not None:
                comp_valid = comp_idx >= 0
                kvg_cmp = _gather_rows(compress_kv, comp_idx, comp_valid)  # [b,s,Ck,hd]
                KVg = mx.concatenate([kvg_win, kvg_cmp], axis=2)
                valid = mx.concatenate([win_valid, comp_valid], axis=2)
            else:
                KVg = kvg_win
                valid = win_valid
            _st.add(KVg)
        # K30 x K29: feed the gathered-k operands to the fused decode kernel when
        # both levers are armed (GPU, decode/verify small-M).  Each query row is its
        # own batch (S=1) so the kernel indexes its own [k,hd] gathered KV; the
        # [rows,k] valid mask carries the only edge (the -1 pad slots).
        if _decode_attn_kernel_use(q):
            from mtplx.models import deepseek_v41_attn_kernels as _k29
            k = KVg.shape[2]
            out = _k29.fused_decode_attention(
                q.reshape(b * s, 1, H, hd),
                KVg.reshape(b * s, k, hd),
                KVg.reshape(b * s, k, hd),
                attend=valid.reshape(b * s, k),
                attn_sink=self.attn_sink,
                scale=self.softmax_scale,
                T=k,
            )
            return out.reshape(b, s, H, hd)
        scale = self.softmax_scale
        with _stime.stage_attn("attn." + mode + ".score.qk_matmul") as _st:
            scores = mx.einsum(
                "bshd,bskd->bshk", q.astype(mx.float32), KVg.astype(mx.float32)
            ) * scale                                       # [b,s,H,k]
            _st.add(scores)
        with _stime.stage_attn("attn." + mode + ".score.scale_mask_sink") as _st:
            scores = mx.where(valid[:, :, None, :], scores, float("-inf"))
            _st.add(scores)
        sink = self.attn_sink.astype(mx.float32).reshape(1, 1, H, 1)
        with _stime.stage_attn("attn." + mode + ".score.softmax") as _st:
            # reference _k_sparse_attn L149-153: value-0 sink in the denominator,
            # a finite max floor so an all-invalid row yields all-zero (not NaN).
            m = mx.maximum(mx.max(scores, axis=-1, keepdims=True), sink)
            ex = mx.exp(scores - m)                         # masked -> exp(-inf) = 0
            denom = mx.sum(ex, axis=-1, keepdims=True) + mx.exp(sink - m)
            _st.add(ex, denom)
        with _stime.stage_attn("attn." + mode + ".score.pv_matmul") as _st:
            o = mx.einsum("bshk,bskd->bshd", ex, KVg.astype(mx.float32)) / denom
            _st.add(o)
        return o

    def _window_attend(self, positions, T, b, s, shared):
        """The causal sliding-window attend mask ``[b, s, T]``.

        Identical across every backbone layer of one forward (same ``positions``
        object, same lockstep window length ``T``, same ``window_size``).  Under
        ``MTPLX_DSV41_ATTN_WIN_MEMO`` (K24) it is computed once per forward and
        memoized on the per-forward ``shared`` runtime, reused for the other
        layers -- byte-identical (the reused array is the same object; reuse fires
        only when ``positions`` is the same object and ``(T, window_size, b, s)``
        match, else it recomputes, so a forward whose layers differ is never
        wrong).  With the flag off it is exactly the original per-layer build."""
        if _ATTN_WIN_MEMO and shared is not None:
            memo = getattr(shared, "_win_attend_memo", None)
            if (memo is not None and memo[0] is positions and memo[1] == T
                    and memo[2] == self.window_size and memo[3] == (b, s)):
                return memo[4]
        wpos = mx.arange(T)
        qp = positions[:, None]
        wp = wpos[None, :]
        win_attend = (wp <= qp) & (wp > qp - self.window_size)  # [s, T]
        attend = mx.broadcast_to(win_attend[None], (b, s, T))
        if _ATTN_WIN_MEMO and shared is not None:
            shared._win_attend_memo = (positions, T, self.window_size, (b, s), attend)
        return attend

    def _publish_compressed(self, x, positions, layer_cache, shared, qcos, qsin):
        """Full/kv_source layer: pool this call's compressed latents (prefill chunk
        or one decode step, both via W13's CompressorState frontier), RoPE the new
        latents at their group positions, derive index keys, append both to the
        layer cache (``append_compress`` / ``append_index_k``) and publish to the
        shared runtime.  ``group j`` stands for the first token of its group, so it
        takes position ``j * ratio`` (reference ``_compress_kv`` L748-758)."""
        ratio = self.compress_ratio
        # W13's CompressorState (comp_state) retains the frontier; ratio==1 has none.
        latent_pre = self.compressor.pool(x, layer_cache.comp_state)
        if latent_pre is not None and latent_pre.shape[1] > 0:
            n_prev = 0 if layer_cache.compress_kv is None else layer_cache.compress_kv.shape[1]
            n_new = latent_pre.shape[1]
            group_pos = mx.arange(n_prev, n_prev + n_new) * ratio
            gcos, gsin = _cos_sin(self.inv_freq, group_pos)
            compress_new = _rope_last(latent_pre, gcos, gsin)
            index_new = self.indexer.keys(latent_pre, gcos, gsin)
            layer_cache.append_compress(compress_new)
            layer_cache.append_index_k(index_new)
        shared.compress_kv = layer_cache.compress_kv
        shared.index_k = layer_cache.index_k

    def _compressed(self, x, qr, positions, layer_cache, shared, qcos, qsin):
        """Returns (compress_kv [b,n_comp,hd], compress_attend [b,s,n_comp] bool)
        for the concatenated-KV attention, running the CSA2 mode dispatch."""
        ratio = self.compress_ratio
        if self.is_kv_source:
            # W47: compress/index KV build (pool + RoPE + indexer keys) + the
            # compress/index cache appends -- the "cache append" work on kv_source
            # layers, distinct from the window append in _attend.
            with _stime.stage_prefill("attn." + self.mode + ".compress_append") as _sp, \
                    _stime.stage_decode("attn." + self.mode + ".compress_append") as _sd:
                self._publish_compressed(x, positions, layer_cache, shared, qcos, qsin)
                _sp.add(shared.compress_kv, shared.index_k)
                _sd.add(shared.compress_kv, shared.index_k)
        compress_kv = shared.compress_kv
        index_k = shared.index_k
        if compress_kv is None:
            # No compressed rows exist yet (a span shorter than one ``ratio`` group,
            # e.g. the first token of a ratio-2 layer under fine chunking, or a
            # prompt below ``ratio``).  Every query's reachable count is 0, so the
            # compressed branch is entirely masked out -- attend over the window
            # only, exactly as one-shot does when ``compress_lens == 0``.
            if self.capture_selection:
                self.last_selection = None
                self.last_candidates = shared.candidates
            return None
        n_comp = compress_kv.shape[1]
        compress_lens = (positions + 1) // ratio  # [s] reachable compressed rows

        # W47 indexer/candidate selection: the data-dependent CSA row pick (top-k
        # over the compressed rows).  Reuse layers read the source's mask (cheap).
        # W47 prefill sub-stage AND W73 decode sub-stage: the indexer's data-dependent
        # CSA row pick.  On an index source it scores ALL n_comp compressed rows and
        # sorts (O(n_comp)); Reuse layers just read the source's mask (cheap).  At
        # decode the single ``attn.<mode>`` bracket hides it, so peel it out via
        # stage_decode too (kept out of the flat sum; one bracket no-ops per kind).
        with _stime.stage_prefill("attn." + self.mode + ".select") as _sp, \
                _stime.stage_decode("attn." + self.mode + ".select") as _sd:
            if self.is_index_source:
                set_c = self.is_candidate_source
                cand = None if set_c else shared.candidates
                mask, cand_out = self.indexer.select(
                    x, qr, index_k, qcos, qsin, compress_lens, n_comp,
                    candidates=cand, set_candidates=set_c,
                    cand_topk=self.candidate_topk_blocks, cand_block=self.candidate_block_size,
                )
                shared.topk_mask = mask
                if set_c:
                    shared.candidates = cand_out
                # K30 (W59): publish the selection as gather indices too, once per
                # index source (the Reuse layers below reuse it, like topk_mask).
                if _resolve_selected_keys():
                    shared.selected_idx = _mask_to_topk_idx(
                        mask, min(self.indexer.index_topk, n_comp)
                    )
            else:  # Reuse: read the source's selection
                mask = shared.topk_mask
            # W76: fence ``mask`` always; additionally fence the K30
            # ``selected_idx`` argsort under MTPLX_DSV41_SELECT_FENCE so its O(T)
            # cost is charged to ``select`` rather than leaking into the ``score``
            # stage that first forces it (byte-identical -- same array, no-op in
            # production; a stage-timing attribution fix only).
            if (_resolve_select_fence() and self.is_index_source
                    and shared.selected_idx is not None):
                _sp.add(mask, shared.selected_idx)
                _sd.add(mask, shared.selected_idx)
            else:
                _sp.add(mask)
                _sd.add(mask)
        if self.capture_selection:
            self.last_selection = mask
            self.last_candidates = shared.candidates
        return compress_kv, mask

    def __call__(self, x, positions, layer_cache, shared):
        # Decode: one ``attn.<mode>`` stage.  Prefill (W47): the finer sub-stages
        # inside ``_attend`` fire instead (via ``stage_prefill``), so the outer
        # bracket is skipped to avoid double-counting.  Off / decode: the sub-stage
        # brackets are no-ops, so ``_attend`` runs inside this single bracket.
        if _stime.is_prefill():
            return self._attend(x, positions, layer_cache, shared)
        with _stime.stage("attn." + self.mode) as _st:
            out = self._attend(x, positions, layer_cache, shared)
            _st.add(out)
        return out

    def _attend(self, x, positions, layer_cache, shared):
        b, s, _ = x.shape
        H, hd, rd = self.n_heads, self.head_dim, self.rope_head_dim
        mode = self.mode
        qcos, qsin = _cos_sin(self.inv_freq, positions)

        # K22 attention-chain compile: the pure projection/norm/rope prep that
        # produces (q, qr, kv_new) is one compiled tape at decode/verify row
        # counts (the KV-cache write, the window mask, the indexer's data-dependent
        # selection and the SDPA all stay OUTSIDE it).  Byte-for-byte the eager
        # body with the flag off / above the row cap.  W47: prefill timing forces
        # eager (``_attn_use_compile`` is False under ``is_prefill``) so this
        # bracket fences the real projection chain, not a tape.
        with _stime.stage_prefill("attn." + mode + ".qkv_proj") as _st:
            if _attn_use_compile(b * s):
                q, qr, kv_new = _attn_qkv_prep(self)(
                    x, qcos, qsin, self.q_norm_weight, self.kv_norm_weight,
                    *_lin_arrays(self.wq_a), *_lin_arrays(self.wq_b), *_lin_arrays(self.wkv),
                )
            else:
                qr = _rmsnorm(self.wq_a(x), self.q_norm_weight, self.eps)
                q = self.wq_b(qr).reshape(b, s, H, hd)
                q = _rope_last(q, qcos, qsin)

                kv_new = _rmsnorm(self.wkv(x), self.kv_norm_weight, self.eps)
                kv_new = _rope_last(kv_new, qcos, qsin)  # window kv roped at its own token positions
            _st.add(q, qr, kv_new)
        # W13's window store keeps the post-RoPE rows append-only (row i == token i).
        # Phase 1 attends over the full history and realises the reference sliding
        # window (get_window_topk_idxs L409-426 / _window_kv L700-720) as a causal
        # window mask over absolute positions -- equivalent to the reference ring
        # for every query that can still reach a slot; ``ring()`` is the bounded
        # phase-2 view.
        # W47 prefill sub-stage AND W73 decode sub-stage: the window KV append is
        # O(current-length) under the phase-1 concatenate backing (the K32
        # MTPLX_DSV41_KV_CHUNK_GROW lever makes it amortized O(1)).  At decode the
        # single ``attn.<mode>`` bracket hides it, so peel it out via stage_decode
        # (kept out of the flat sum, like the prefill attn/switch breakdowns) -- one
        # of the two brackets is a no-op in each session kind, so at most one fires.
        with _stime.stage_prefill("attn." + mode + ".cache_append") as _sp, \
                _stime.stage_decode("attn." + mode + ".cache_append") as _sd:
            layer_cache.append_window(kv_new)
            window_all = layer_cache.window
            _sp.add(window_all)
            _sd.add(window_all)
        # K30 (W59): gather only the selected keys per query instead of scoring the
        # full history and masking -- for prefill (rows > 1), decode (s == 1) AND the
        # K+1 verify batch (W59 decode extension).  The window store and the indexer
        # selection are built the same way; only what is handed to the attention math
        # changes.  At decode the shipped path re-scores the whole compressed history
        # [1,64,T] every token; K30 makes it T-independent (k = window + index_topk).
        # Composes with W60's K29 fused decode kernel (see _sparse_attend_selected).
        use_selected = _resolve_selected_keys()
        # K24 (W45): the sliding-window attend mask is identical across every layer
        # of this forward; memoize it on the per-forward shared runtime under
        # MTPLX_DSV41_ATTN_WIN_MEMO (byte-identical, a pure host-dispatch cut).
        attend = None if use_selected else self._window_attend(
            positions, window_all.shape[1], b, s, shared
        )
        KV = window_all
        sel_compress_kv = None
        sel_comp_idx = None

        if self.compress_ratio:
            comp = self._compressed(
                x, qr, positions, layer_cache, shared, qcos, qsin
            )
            if comp is not None:
                compress_kv, comp_attend = comp
                if use_selected:
                    # K30: the index source published its selection as gather
                    # indices on shared.selected_idx; Reuse layers read it.
                    sel_compress_kv = compress_kv
                    sel_comp_idx = shared.selected_idx
                else:
                    KV = mx.concatenate([window_all, compress_kv], axis=1)
                    attend = mx.concatenate([attend, comp_attend], axis=-1)

        # W47 score+softmax+value + output projection: the masked-full path's
        # [rows, H, T] score transient grows with T; the K30 selected-gather path
        # (use_selected) bounds it at [rows, H, window + index_topk].
        with _stime.stage_prefill("attn." + mode + ".score") as _st:
            if use_selected:
                o = self._sparse_attend_selected(
                    q, window_all, sel_compress_kv, sel_comp_idx, positions
                )
            else:
                o = self._sparse_attend(q, KV, attend)
            # K22: the post-attention output chain -- query-RoPE removal, the grouped
            # o-LoRA down-projection and the ``wo_b`` up-projection -- is pure and
            # fixed-shape (the SDPA output ``o`` is [b,s,H,hd]); one compiled tape at
            # decode/verify.  ``w_ol`` (the dequantized grouped ``wo_a`` weight) is a
            # per-forward constant, derived by the same path as eager and fed as an
            # input, so the einsum inside the tape is bit-exact to :meth:`_o_lora_down`.
            # W50: the output-projection tail is bracketed separately (attn_breakdown)
            # so the score sub-stages sum to the SDPA proper, not SDPA + projection.
            with _stime.stage_attn("attn." + mode + ".score.out_proj") as _sp:
                if _attn_use_compile(b * s):
                    out = _attn_out_prep(self)(
                        o, qcos, qsin, self._o_lora_dense_weight(), *_lin_arrays(self.wo_b)
                    )
                else:
                    o = _rope_last(o, qcos, qsin, inverse=True)
                    o = o.reshape(b, s, self.n_groups, -1)
                    o = self._o_lora_down(o)
                    out = self.wo_b(o.reshape(b, s, -1))
                _sp.add(out)
            _st.add(out)
        return out

    def _o_lora_dense_weight(self):
        """The grouped ``wo_a`` weight as a dense ``[g, o_lora_rank, in_per_group]``
        f32-castable array -- dequantized when q8/native-resident, else the raw
        ``nn.Linear`` weight.  Extracted so :meth:`_o_lora_down` (eager) and the K22
        compiled output tape derive the einsum weight through the *identical* path
        (bit-exact either way): the dequant is weight-only, no dependence on ``o``."""
        wo = self.wo_a
        if isinstance(wo, nn.QuantizedLinear):
            # Mode-aware: affine q8 carries biases; the native float codecs
            # (mxfp8/mxfp4/nvfp4) have ``biases is None`` -- mx.dequantize takes
            # the mode and a ``None`` bias directly.
            w = mx.dequantize(
                wo.weight, wo.scales, wo.biases,
                group_size=wo.group_size, bits=wo.bits, mode=getattr(wo, "mode", "affine"),
            )
        else:
            w = wo.weight
        return w.reshape(self.n_groups, self.o_lora_rank, -1)

    def _o_lora_down(self, o):
        """Grouped ``wo_a`` down-projection: reshape the [out=n_groups*o_lora_rank,
        in_per_group] weight to [g, o_lora_rank, in] and einsum each group over its
        own heads (reference model.py L785-787).  Dequantized when q8-resident."""
        w = self._o_lora_dense_weight()
        return mx.einsum("bsgd,grd->bsgr", o.astype(mx.float32), w.astype(mx.float32))


# ---------------------------------------------------------------------------
# Attention-chain tape collapse (kernel-ledger K22, W41)
# ---------------------------------------------------------------------------
# W37/window-13 stage timing put decode attention at attn.reuse 50.2 ms/tok (30
# layers -> 1.7 ms/layer for an M=1 MLA step) + swa_only 9.8 (2 layers, 4.9
# ms/layer!) + full 8.9 (4) + reindex 8.0 (4).  A single-row attention step
# costing 1.7-4.9 ms is a dispatch-chain problem -- dozens of tiny projection /
# RMSNorm / RoPE ops the GPU never notices -- the regime where whole-chain
# ``mx.compile`` pays and single fusions do not ([[b1-decode-dispatch-removal-
# hides]]).  K22 replays the two PURE attention chains from an ``mx.compile`` tape
# instead of rebuilding the graph from Python each call, behind
# ``MTPLX_DSV41_ATTN_COMPILE`` (default OFF -- the decode win is a GPU-window
# measurement).  Following W33/K4's fixed-shape + row-cap design (shapeless is not
# viable), the two chains are:
#  1. QKV prep: q down/up projection + q-RMSNorm + RoPE and the kv down projection
#     + kv-RMSNorm + RoPE -> (q, qr, kv_new).  PURE: the KV-cache write
#     (``append_window``), the window mask, the indexer's data-dependent CSA
#     selection and the SDPA all stay OUTSIDE the tape.
#  2. Output prep: query-RoPE removal + grouped o-LoRA down-projection einsum +
#     ``wo_b`` up-projection -> attention output.  PURE (the SDPA that mutates
#     nothing runs between the two tapes).
# Projection weights arrive as tape INPUTS (one tape shared across all 40 layers
# -- they share every shape, differing only in values), and each projection is
# applied by ``_apply_lin`` exactly as ``nn.Linear`` / ``nn.QuantizedLinear`` do,
# so the tape is bit-identical to the eager module call whether the residents are
# dense (tiny-config / native-BF16-kept projections) or quantized (q8 gs64 /
# mxfp8 / mxfp4 / nvfp4 residents -- ``mx.quantized_matmul`` replays as one
# primitive, so compile never reassociates it).  ``mx.unflatten``/``mx.flatten``
# replace the ``.reshape(b,s,...)`` calls so the tape reads no dynamic ``.shape``.
#: Env toggle for the K22 attention-chain tape collapse.  Default OFF (the win is
#: a GPU-window measurement).  Read through the module global so tests/operators
#: can flip it after import.
_ATTN_COMPILE_ENV = "MTPLX_DSV41_ATTN_COMPILE"
_ATTN_COMPILE = (os.environ.get(_ATTN_COMPILE_ENV) or "").strip().lower() not in (
    "", "0", "false", "no", "off", "auto",
)
#: Row count (``b*s``) at or below which the compiled attention tapes fire; above
#: it the eager body runs (byte-identical).  Confines compile to the tiny
#: repeating decode (1) / verify (K+1) shapes, where it is ``mx.array_equal`` with
#: eager and where the per-primitive host encode dominates.  Module global so
#: tests can retarget it.
_ATTN_COMPILE_MAX_ROWS = 32
#: One compiled tape per ``(kind, structural-signature)``; the signature carries
#: each projection's ``_lin_desc`` (dense vs the quant codec) plus the head/group
#: geometry, so a quantized-resident model gets its own bit-exact tape.
_ATTN_COMPILED: dict = {}

#: W45 (kernel-ledger K24): dedup the sliding-window attend mask across the
#: backbone layers of one decode/verify/chunk forward.  The mask
#: ``(wp <= qp) & (wp > qp - window_size)`` broadcast to ``[b, s, T]`` is a pure
#: function of ``(positions, window length T, window_size)`` -- all invariant
#: across the layers of one ``_forward_span`` (positions and the per-forward
#: ``shared`` runtime are created once and handed to every layer, and every
#: layer's window store grows in lockstep to the same ``T``), yet each layer
#: rebuilds it from scratch (~arange + 2 compares + subtract + and + broadcast,
#: ~11 graph nodes) -- the census measured this as the largest remaining
#: mode-invariant per-layer dispatch chunk after K22.  Memoizing it on the
#: per-forward ``shared`` runtime computes it ONCE and reuses the identical array
#: for the other ``n_layers - 1`` layers: a pure host-dispatch reduction,
#: byte-identical (the reused mask is the same array; reuse fires only when the
#: query positions are the *same object* and ``(T, window_size, b, s)`` match, so
#: any forward whose layers differ falls back to per-layer recompute -- never
#: wrong).  Default OFF; read through the module global so tests/operators flip it
#: after import (the serving harness stamps keys after import,
#: [[env-flags-read-at-use-not-import]]).
_ATTN_WIN_MEMO_ENV = "MTPLX_DSV41_ATTN_WIN_MEMO"
_ATTN_WIN_MEMO = (os.environ.get(_ATTN_WIN_MEMO_ENV) or "").strip().lower() not in (
    "", "0", "false", "no", "off", "auto",
)


# --- W50 / K25: prefill score-path precision + split-K online softmax --------
#: Selects the prefill (rows > 1) score-path implementation.  ``oneshot``
#: (unset/default) is the shipped full-T one-shot softmax.  ``lean`` is the W50
#: pass-cut f32 one-shot (window-20 found the score stage pass/bandwidth-bound, not
#: matmul-FLOP-bound -- bf16 was -34%, split-K -16%): it scales ``q`` once instead
#: of the T-wide scores and folds the value-0 sink into the denominator (no sink
#: concat / post-softmax slice), cutting three passes over the ``[rows,H,T]``
#: transient.  ``lean`` forces f32 and ignores the DTYPE / KEY_CHUNK knobs; it is
#: reassociation-level vs control (greedy-identical), never bit-identical.  Read at
#: call time ([[env-flags-read-at-use-not-import]]).
_PREFILL_SCORE_PATH_ENV = "MTPLX_DSV41_PREFILL_SCORE_PATH"
_PREFILL_SCORE_PATH_ONESHOT_ALIASES = ("", "default", "off", "none", "control", "oneshot", "one_shot")
_PREFILL_SCORE_PATH_LEAN_ALIASES = ("lean", "fused", "passcut", "pass_cut")


def _resolve_prefill_score_path(raw=None):
    """Resolve ``MTPLX_DSV41_PREFILL_SCORE_PATH`` to ``"oneshot"`` (unset/default,
    the shipped path) or ``"lean"`` (the W50 f32 pass-cut one-shot).  Read at use;
    an unrecognised non-empty value raises (fail fast)."""
    val = os.environ.get(_PREFILL_SCORE_PATH_ENV) if raw is None else raw
    val = (val or "").strip().lower()
    if val in _PREFILL_SCORE_PATH_ONESHOT_ALIASES:
        return "oneshot"
    if val in _PREFILL_SCORE_PATH_LEAN_ALIASES:
        return "lean"
    raise ValueError(
        f"{_PREFILL_SCORE_PATH_ENV}={val!r} is not one of ('oneshot', 'lean') "
        "(or empty/'default' for the current one-shot path)"
    )


#: Casts the prefill (rows > 1) QK^T / PV matmul *inputs* to a cheaper dtype.  MLX
#: matmul accumulates in f32 internally and rounds the result back to the input
#: dtype (verified: a bf16 matmul equals ``round_bf16(f32-accumulated result)``),
#: so ``bf16`` keeps the ~2× bf16 matmul throughput while the scale / mask / sink /
#: softmax stay in f32 exactly as the reference oracle -- the only numerical change
#: is bf16 rounding of the two matmuls' inputs+outputs.  ``f32`` (unset/default) is
#: byte-identical to the shipped path.  Decode (``q.shape[1] == 1``) never reads
#: this lever.  Read at call time (the serving harness stamps keys after import,
#: [[env-flags-read-at-use-not-import]]).
_PREFILL_SCORE_DTYPE_ENV = "MTPLX_DSV41_PREFILL_SCORE_DTYPE"
_PREFILL_SCORE_DTYPE_DEFAULT_ALIASES = (
    "", "default", "off", "none", "control", "f32", "fp32", "float32",
)
_PREFILL_SCORE_DTYPE_BF16_ALIASES = ("bf16", "bfloat16")


def _resolve_prefill_score_dtype(raw=None):
    """Resolve ``MTPLX_DSV41_PREFILL_SCORE_DTYPE`` to the mx dtype for the prefill
    QK^T / PV matmul inputs.  Unset / ``f32`` -> ``mx.float32`` (byte-identical to
    control); ``bf16`` -> ``mx.bfloat16``.  Read at use, never frozen at import.
    An unrecognised non-empty value raises so a mistyped precision lever fails fast
    rather than silently running the default through a whole benchmark window."""
    val = os.environ.get(_PREFILL_SCORE_DTYPE_ENV) if raw is None else raw
    val = (val or "").strip().lower()
    if val in _PREFILL_SCORE_DTYPE_DEFAULT_ALIASES:
        return mx.float32
    if val in _PREFILL_SCORE_DTYPE_BF16_ALIASES:
        return mx.bfloat16
    raise ValueError(
        f"{_PREFILL_SCORE_DTYPE_ENV}={val!r} is not one of ('f32', 'bf16') "
        "(or empty/'default' for the current f32 behaviour)"
    )


#: Key-chunk width for the two-pass split-K online softmax (gemma4 D512 two-pass /
#: K6) over the concatenated KV.  Unset / 0 -> one-shot (full-T score transient,
#: the shipped path).  A positive int caps the score transient at
#: ``[rows, H, key_chunk]``; mathematically identical to one-shot up to float
#: reassociation of the softmax denom + value sum.  Read at call time.
_PREFILL_SCORE_KEY_CHUNK_ENV = "MTPLX_DSV41_PREFILL_SCORE_KEY_CHUNK"


def _resolve_prefill_score_key_chunk(raw=None):
    """Resolve ``MTPLX_DSV41_PREFILL_SCORE_KEY_CHUNK`` to a positive int chunk
    width, or ``None`` (off = one-shot, the default).  An unrecognised /
    non-positive value raises (fail fast) except the explicit ``off`` aliases."""
    val = os.environ.get(_PREFILL_SCORE_KEY_CHUNK_ENV) if raw is None else raw
    val = (val or "").strip().lower()
    if val in ("", "0", "off", "none", "default"):
        return None
    try:
        n = int(val)
    except ValueError:
        raise ValueError(
            f"{_PREFILL_SCORE_KEY_CHUNK_ENV}={val!r} is not a positive integer "
            "(or empty/0/'off' for the one-shot path)"
        )
    if n <= 0:
        raise ValueError(
            f"{_PREFILL_SCORE_KEY_CHUNK_ENV}={val!r} must be > 0 "
            "(or empty/0/'off' for the one-shot path)"
        )
    return n


# --- W59 / K30: selected-key (gather) prefill attention ----------------------
#: Under ``MTPLX_DSV41_SELECTED_KEYS`` the prefill (rows > 1) attention gathers
#: only the keys/values each query actually attends -- its sliding window plus the
#: indexer's ``index_topk`` selected compressed rows -- into a compact
#: ``[rows, k, head_dim]`` operand and runs one softmax over ``k`` keys, exactly as
#: the reference ``sparse_attn`` (kernel.py ``sparse_attn_kernel``: gather
#: ``kv[topk_idxs]`` then FlashAttention over ``k = cdiv(topk, block)*block``
#: columns).  The shipped port instead computes the full ``[rows, H, T]`` score
#: over the entire window+compressed history and masks -- a faithful but wasteful
#: transliteration (masked keys contribute exactly 0 to the softmax), whose score
#: transient grows with T while the reference's ``k`` saturates at
#: ``window + index_topk``.  Selected-gather vs masked-full are mathematically
#: identical up to float reassociation of the softmax sum (greedy-identical, never
#: bit-identical).  Prefill only; decode (``q.shape[1] == 1``) is untouched.  Read
#: at use, never frozen at import ([[env-flags-read-at-use-not-import]]).
_SELECTED_KEYS_ENV = "MTPLX_DSV41_SELECTED_KEYS"


def _resolve_selected_keys(raw=None) -> bool:
    """Whether ``MTPLX_DSV41_SELECTED_KEYS`` arms the K30 selected-key gather
    prefill attention (default OFF).  Read at call time so the serving harness can
    stamp the key after importing this module."""
    val = os.environ.get(_SELECTED_KEYS_ENV) if raw is None else raw
    return (val or "").strip().lower() not in ("", "0", "false", "no", "off", "auto")


#: W76: fence the K30 ``selected_idx`` publication (``_mask_to_topk_idx``, an
#: ``argsort`` over ``n_comp``) INTO the ``attn.<mode>.select`` decode sub-stage.
#: The shipped code fences only ``mask`` (``_sd.add(mask)``); the separate
#: ``shared.selected_idx = _mask_to_topk_idx(mask, ...)`` array it publishes is
#: left lazy, so its ``argsort`` (O(n_comp log n_comp), an index-source-layer O(T)
#: cost) is not forced until a downstream gather in the ``score`` stage reads it --
#: mis-attributing that O(T) select cost into decode attention-*proper* (the W76
#: audit's stage-timing artifact).  Fencing ``selected_idx`` in the select bracket
#: charges the argsort where it belongs.  This is a **stage-timing attribution
#: fix**: the extra ``add`` is a no-op outside a recording decode session (the
#: ``_sd`` fence is ``_NullFence`` in production and under prefill), and the array
#: fenced is the *same* object the gather would force -- so the token, cache and
#: logits are BYTE-IDENTICAL on or off.  Read at use, never frozen at import
#: ([[env-flags-read-at-use-not-import]]).
_SELECT_FENCE_ENV = "MTPLX_DSV41_SELECT_FENCE"


def _resolve_select_fence(raw=None) -> bool:
    """Whether ``MTPLX_DSV41_SELECT_FENCE`` charges the K30 ``selected_idx``
    argsort to ``attn.<mode>.select`` instead of leaking it into ``score`` (default
    OFF -- the shipped attribution).  Read at call time (serving stamps the key
    after import)."""
    val = os.environ.get(_SELECT_FENCE_ENV) if raw is None else raw
    return (val or "").strip().lower() not in ("", "0", "false", "no", "off", "auto")


def _mask_to_topk_idx(mask: mx.array, k: int) -> mx.array:
    """Convert a boolean top-k row mask ``[b, s, n]`` (exactly ``min(k, reachable)``
    True per row) into ``[b, s, k]`` int32 indices of the True positions in
    ascending order, padded with ``-1`` where a row has fewer than ``k`` selected.

    Reproduces the reference's ``score.topk(k).indices.sort().values`` *as a
    function of the same selection the port already computed*: the True positions
    are exactly ``_topk_rows``' selected set, so the gathered key set is identical
    to the masked-full path's unmasked set (softmax reassociation-level equal).
    Position order within the row is irrelevant to the softmax; ascending is chosen
    to match the reference.  One argsort over ``n`` per row -- run once per index
    source (published on ``shared.selected_idx``, reused down the stack), O(n) in
    memory and ~n/(H*head_dim) cheaper than the score it replaces."""
    b, s, n = mask.shape
    ar = mx.arange(n)
    # True positions sort by their own index (0..n-1); False positions by n+index,
    # so every True lands ahead of every False.  All keys distinct -> deterministic.
    keys = mx.where(mask, ar.reshape(1, 1, n), (n + ar).reshape(1, 1, n))
    order = mx.argsort(keys, axis=-1)[..., :k].astype(mx.int32)   # [b, s, k]
    count = mx.sum(mask.astype(mx.int32), axis=-1, keepdims=True)  # [b, s, 1]
    valid = mx.arange(k).reshape(1, 1, k) < count
    return mx.where(valid, order, mx.array(-1, dtype=mx.int32))


def _gather_rows(source: mx.array, idx: mx.array, valid: mx.array) -> mx.array:
    """Gather ``source`` rows named by ``idx`` into a compact per-query operand.

    ``source`` is ``[b, n, d]`` (the shared 1-KV-head window / compressed history),
    ``idx`` ``[b, s, k]`` int (``-1`` for a pad slot), ``valid`` ``[b, s, k]`` bool.
    Returns ``[b, s, k, d]``.  Uses a single flat ``take`` over ``b*n`` rows
    (invalid slots clamped to row 0, masked out later by the caller), so it never
    materialises the ``[b, s, n, d]`` broadcast that a naive ``take_along_axis``
    would -- the only traffic is the ``b*s*k`` gathered ``d``-vectors, the
    least-traffic layout given the KV is shared across all query heads."""
    b, n, d = source.shape
    s, k = idx.shape[1], idx.shape[2]
    idx_c = mx.where(valid, idx, 0)
    offs = (mx.arange(b) * n).reshape(b, 1, 1)
    flat = (idx_c + offs).reshape(-1)
    g = mx.take(source.reshape(b * n, d), flat, axis=0)   # [b*s*k, d]
    return g.reshape(b, s, k, d)


#: W58 / K28: fuse the prefill mask + per-head value-0 sink + f32 softmax over the
#: ``[rows,64,T]`` score transient into ONE ``mx.fast.metal_kernel`` dispatch that
#: reads the raw scores twice and writes the normalised probabilities once (vs the
#: eager ~4-6 T-wide passes + one or two T-wide intermediates).  Prefill-only
#: (gated ``q.shape[1] > 1``), one-shot path only (the split-K/chunked path is NOT
#: routed through the kernel -- see :mod:`mtplx.kernels.dsv41_fused_softmax`), and
#: GPU-only (a CPU-pinned host falls back to the eager path, byte-identical to the
#: chosen score path).  Reassociation-level vs the eager f32 softmax (the
#: threadgroup tree reorders the max/denom/value sums): expect ``max|Δ| <= 1e-6``,
#: greedy-argmax identical -- NOT byte-identical (same class as ``score_chunked`` /
#: ``score_lean``).  Read at use, never frozen at import
#: ([[env-flags-read-at-use-not-import]]).
_PREFILL_SOFTMAX_KERNEL_ENV = "MTPLX_DSV41_PREFILL_SOFTMAX_KERNEL"


def _resolve_prefill_softmax_kernel(raw=None) -> bool:
    """Resolve ``MTPLX_DSV41_PREFILL_SOFTMAX_KERNEL`` to a bool (default OFF).

    Truthy: ``1/true/on/yes``.  Off: unset / ``0/false/off/no/none/default``.  An
    unrecognised non-empty value raises (fail fast), matching the other DSV4.1
    prefill levers."""
    val = os.environ.get(_PREFILL_SOFTMAX_KERNEL_ENV) if raw is None else raw
    val = (val or "").strip().lower()
    if val in ("", "0", "false", "off", "no", "none", "default"):
        return False
    if val in ("1", "true", "on", "yes"):
        return True
    raise ValueError(
        f"{_PREFILL_SOFTMAX_KERNEL_ENV}={val!r} is not a boolean "
        "(1/true/on/yes or empty/0/off for the eager softmax)"
    )


def _prefill_softmax_kernel_use() -> bool:
    """Whether the K28 fused-softmax kernel should run on THIS call: the flag is
    armed AND a Metal GPU is the default device.  A CPU-pinned worker test (or a
    no-Metal host) returns ``False`` so the eager path runs and no Metal is
    dispatched -- the GPU route is proven by a spy in the tests."""
    if not _resolve_prefill_softmax_kernel():
        return False
    try:
        if not mx.metal.is_available() or mx.default_device() != mx.gpu:
            return False
    except Exception:
        return False
    return True


# --- W60 / K29: fused decode / verify MLA attention Metal kernel -------------
#: One ``mx.fast.metal_kernel`` per layer for the M=1 (decode) / small-M (K+1
#: verify) attention step: QK^T score + CSA/causal mask + per-head value-0 sink +
#: f32 softmax + PV in ONE dispatch, online-softmax over key tiles so the ``[64,T]``
#: score row never materialises (``mtplx/models/deepseek_v41_attn_kernels.py``).
#: Default OFF (the decode win is a GPU-window measurement), GPU-only (a CPU-pinned
#: host falls back to the eager one-shot, byte-identical to control).  Mode-agnostic
#: -- it reads the assembled ``[b,s,T]`` boolean ``attend`` mask, so all four CSA
#: modes (swa_only / full / reindex / reuse) route through it identically; an
#: unsupported mask shape falls back to eager.  Reassociation-level vs the eager f32
#: path (the tile reduction reorders the max/denom/value sums): ``max|Δ| <= 1e-6``,
#: greedy-argmax identical -- NOT byte-identical.  Read at use, never frozen at
#: import ([[env-flags-read-at-use-not-import]]).
_DECODE_ATTN_KERNEL_ENV = "MTPLX_DSV41_DECODE_ATTN_KERNEL"

#: Max query rows (``b*s``) the decode kernel serves: M=1 decode and the ``K+1``
#: verify batch (8 covers MTP depth up to 7).  Above it the eager prefill score
#: path runs (W58/W59's domain) -- the hook never diverts prefill.
_DECODE_ATTN_KERNEL_MAX_ROWS = 8


def _resolve_decode_attn_kernel(raw=None) -> bool:
    """Resolve ``MTPLX_DSV41_DECODE_ATTN_KERNEL`` to a bool (default OFF).

    Truthy: ``1/true/on/yes``.  Off: unset / ``0/false/off/no/none/default``.  An
    unrecognised non-empty value raises (fail fast), matching the other DSV4.1
    levers."""
    val = os.environ.get(_DECODE_ATTN_KERNEL_ENV) if raw is None else raw
    val = (val or "").strip().lower()
    if val in ("", "0", "false", "off", "no", "none", "default"):
        return False
    if val in ("1", "true", "on", "yes"):
        return True
    raise ValueError(
        f"{_DECODE_ATTN_KERNEL_ENV}={val!r} is not a boolean "
        "(1/true/on/yes or empty/0/off for the eager attention)"
    )


def _decode_attn_kernel_use(q) -> bool:
    """Whether the K29 fused decode attention should run on THIS call: the flag is
    armed, a Metal GPU is the default device, AND the query is small-M (decode /
    verify, ``b*s <= _DECODE_ATTN_KERNEL_MAX_ROWS``).  A CPU-pinned worker test (or
    a no-Metal host) returns ``False`` so the eager path runs and no Metal is
    dispatched -- the GPU route is proven by a spy in the tests."""
    if not _resolve_decode_attn_kernel():
        return False
    try:
        if not mx.metal.is_available() or mx.default_device() != mx.gpu:
            return False
    except Exception:
        return False
    try:
        rows = int(q.shape[0]) * int(q.shape[1])
    except Exception:
        return False
    return rows <= _DECODE_ATTN_KERNEL_MAX_ROWS


def _lin_desc(linear):
    """Structural descriptor of a projection linear (dense ``nn.Linear`` or
    ``nn.QuantizedLinear``) -- the compile-cache discriminator; carries no arrays."""
    if isinstance(linear, nn.QuantizedLinear):
        has_b = linear.get("biases") is not None
        return ("q", int(linear.group_size), int(linear.bits),
                str(getattr(linear, "mode", "affine")), has_b)
    return ("d",)


def _lin_arrays(linear):
    """The raw arrays a projection linear needs as tape inputs, in
    :func:`_lin_desc` order: dense -> (weight,); quantized -> (weight, scales[,
    biases])."""
    if isinstance(linear, nn.QuantizedLinear):
        arrs = [linear.weight, linear.scales]
        biases = linear.get("biases")
        if biases is not None:
            arrs.append(biases)
        return tuple(arrs)
    return (linear.weight,)


def _lin_n(desc) -> int:
    """Number of arrays :func:`_lin_arrays` yields for ``desc``."""
    if desc[0] == "q":
        return 3 if desc[4] else 2
    return 1


def _apply_lin(desc, arrs, x):
    """Apply the linear described by ``desc`` to ``x`` using ``arrs`` -- exactly
    what ``nn.Linear.__call__`` (``x @ w.T``) / ``nn.QuantizedLinear.__call__``
    (``mx.quantized_matmul``) do (bias=False throughout), so the tape is
    bit-identical to the eager module call and compile never reassociates the
    quantized matmul (one primitive)."""
    if desc[0] == "q":
        _, gs, bits, mode, has_b = desc
        biases = arrs[2] if has_b else None
        return mx.quantized_matmul(
            x, arrs[0], scales=arrs[1], biases=biases,
            transpose=True, group_size=gs, bits=bits, mode=mode,
        )
    return x @ arrs[0].T


def _attn_use_compile(rows: int) -> bool:
    """Is this forward in the row regime the K22 attention tapes are kept for?

    Reads the module globals at call time so a test/operator can flip them after
    import.  In *decode* stage timing this does NOT force-eager: ``_attend`` is a
    single ``attn.<mode>`` stage with no sub-stage fences, so a compiled prep tape
    never splits a fenced bracket.  In *prefill* stage timing (W47) it DOES force
    eager (``_stime.is_prefill()``): the prefill path fences finer attention
    sub-stages (qkv_proj / score / ...), which a compiled prep tape would swallow,
    so timing always measures the eager attention chain -- the shipped prefill
    path (chunk rows >> the row cap are eager anyway)."""
    if not _ATTN_COMPILE or _stime.is_prefill():
        return False
    return int(rows) <= _ATTN_COMPILE_MAX_ROWS


def _attn_qkv_prep_impl(x, qcos, qsin, q_norm_w, kv_norm_w, warrs,
                        dq, db, dk, H, hd, eps):
    """Pure pre-SDPA projection prep -> (q, qr, kv_new).  Byte-identical to the
    eager body: ``mx.unflatten(.,-1,(H,hd))`` is the same contiguous split as
    ``.reshape(b,s,H,hd)`` without reading b,s; ``qr`` is threaded out for the
    indexer."""
    nq, nb = _lin_n(dq), _lin_n(db)
    aq = warrs[:nq]
    ab = warrs[nq:nq + nb]
    ak = warrs[nq + nb:]
    qr = _rmsnorm(_apply_lin(dq, aq, x), q_norm_w, eps)
    q = mx.unflatten(_apply_lin(db, ab, qr), -1, (H, hd))
    q = _rope_last(q, qcos, qsin)
    kv_new = _rmsnorm(_apply_lin(dk, ak, x), kv_norm_w, eps)
    kv_new = _rope_last(kv_new, qcos, qsin)
    return q, qr, kv_new


def _attn_out_prep_impl(o, qcos, qsin, w_ol, wb_arrs, dwob, n_groups):
    """Pure post-SDPA output chain: remove the query RoPE, grouped o-LoRA
    down-projection einsum (dense ``w_ol`` fed as input, derived by the same path
    as eager :meth:`Attention._o_lora_down`), then the ``wo_b`` up-projection.
    Byte-identical to the eager tail."""
    o = _rope_last(o, qcos, qsin, inverse=True)
    o = mx.unflatten(mx.flatten(o, -2, -1), -1, (n_groups, -1))
    o = mx.einsum("bsgd,grd->bsgr", o.astype(mx.float32), w_ol.astype(mx.float32))
    o = mx.flatten(o, -2, -1)
    return _apply_lin(dwob, wb_arrs, o)


def _attn_qkv_prep(attn: "Attention"):
    """Build/fetch the compiled QKV-prep tape for ``attn`` (shared across every
    layer with the same projection codec + head geometry)."""
    dq, db, dk = _lin_desc(attn.wq_a), _lin_desc(attn.wq_b), _lin_desc(attn.wkv)
    H, hd, eps = attn.n_heads, attn.head_dim, attn.eps
    key = ("qkv", dq, db, dk, int(H), int(hd), float(eps))
    fn = _ATTN_COMPILED.get(key)
    if fn is None:
        def impl(x, qcos, qsin, q_norm_w, kv_norm_w, *warrs):
            return _attn_qkv_prep_impl(
                x, qcos, qsin, q_norm_w, kv_norm_w, warrs, dq, db, dk, H, hd, eps
            )
        fn = mx.compile(impl)
        _ATTN_COMPILED[key] = fn
    return fn


def _attn_out_prep(attn: "Attention"):
    """Build/fetch the compiled output-prep tape for ``attn``."""
    dwob = _lin_desc(attn.wo_b)
    n_groups = attn.n_groups
    key = ("out", dwob, int(n_groups), int(attn.o_lora_rank), int(attn.head_dim),
           int(attn.n_heads))
    fn = _ATTN_COMPILED.get(key)
    if fn is None:
        def impl(o, qcos, qsin, w_ol, *wb_arrs):
            return _attn_out_prep_impl(o, qcos, qsin, w_ol, wb_arrs, dwob, n_groups)
        fn = mx.compile(impl)
        _ATTN_COMPILED[key] = fn
    return fn


# ---------------------------------------------------------------------------
# Hyper-Connection tape collapse (kernel-ledger K4) -- carry of V4's HC-compile
# ---------------------------------------------------------------------------
# V4 measured this stack (HC-tape collapse + fused CSA) at AR +31.3% /
# -26.1% dispatches (docs/deepseek-v41/KERNEL_LEDGER.md K4; deepseek_v4.py
# ``_HC_COMPILE``/``_hc_compiled``).  The Hyper-Connection pre/post chain around
# each sublayer is ~two dozen tiny elementwise/small-reduction primitives per
# call -- the Sinkhorn normaliser alone is ~39 ``reduce_sum`` + ~39 divide +
# a row-softmax over a ``[..., hc, hc]`` matrix that is 16 floats at decode --
# run 2x per layer x40 layers.  Uncompiled that is the top per-token *dispatch*
# source (KERNEL_LEDGER 2.1 / 4: the ~6.4k HC-mix dispatches, cf. V4's 6,794).
# ``mx.compile`` replays a prebuilt tape instead of rebuilding the graph from
# Python each call and fuses the elementwise triples into single kernels.
#
# What DSV4.1 carries vs V4:
#  * The Sinkhorn stays an OPAQUE function boundary -- these tapes call the
#    module ``hc_split_sinkhorn`` (owned by the K3 worker, W32), whose
#    ``_sinkhorn_normalise`` tail takes K3's Metal kernel when
#    ``MTPLX_DSV41_SINKHORN_METAL`` is armed on the GPU and the identical stock
#    recurrence otherwise (always on CPU) -- so K3's kernel drops in at the tail
#    of the tape without touching this file (V4 note: the kernel is opaque to
#    ``mx.compile`` but sits inside the traced tape).
#  * Layer weights arrive as tape INPUTS (not captured), so one compiled tape is
#    shared across all ``2 * n_layers`` Hyper-Connections -- they share every
#    shape and differ only in weight values.
#  * ``mx.flatten(x, -2, -1)`` replaces ``_mixes``'s ``reshape(*x.shape[:-2],
#    hc*dim)``: identical memory layout / values, but it reads no dynamic
#    ``.shape`` (reshape-from-shape bakes the first trace's dims).
#
# Why NOT ``shapeless=True`` (measured, W33): (1) ``hc_split_sinkhorn`` contains
# ``comb.reshape(*comb.shape[:-1], hc, hc)``; under a shapeless trace MLX raises
# ``[Primitive::output_shapes] Slice cannot infer output shapes`` -- and that
# function is the K3 worker's, not to be edited here.  (2) Even where a tape
# traces shapeless, the batched matmul reassociates ~1e-6 at batch>1, and
# ``mx.compile`` matches eager BIT-EXACTLY (``mx.array_equal``) only in the small
# row regime (decode n=1 / verify n=K+1); at prefill-chunk row counts the
# ``flat @ fn.T`` matmul and the RMS/HC-mix mean reductions reassociate.  So this
# carries V4's fixed-shape + row-cap design (V4 uses no shapeless either): the
# compiled path fires only for ``rows <= _HC_COMPILE_MAX_ROWS`` -- decode/verify,
# where it is bit-exact and where the per-primitive host encode dominates -- and
# prefill chunks fall through to the eager body (byte-identical either way).
#: Env toggle for the K4 Hyper-Connection tape collapse.  Default OFF -- the
#: decode/dispatch win is a GPU-window measurement (KERNEL_LEDGER KG-f), so the
#: eager per-call graph stays the serving default until measured.  Read through
#: the module global (``deepseek_v41._HC_COMPILE``) so tests/operators can flip
#: it after import.
_HC_COMPILE_ENV = "MTPLX_DSV41_HC_COMPILE"
# ``_env_truthy`` is defined further down; inline its semantics for this
# import-time read (unset/0/false/no/off/auto -> OFF).
_HC_COMPILE = (os.environ.get(_HC_COMPILE_ENV) or "").strip().lower() not in (
    "", "0", "false", "no", "off", "auto",
)
#: Row count (``prod(x.shape[:-2])`` = ``b*s``) at or below which the compiled HC
#: tape is used; above it the eager body runs.  Confines compile to the tiny,
#: repeating decode/verify shapes -- where it is ``mx.array_equal`` with eager and
#: where dispatch host-encode dominates -- and keeps prefill (large chunks, where
#: the matmul/mean reductions reassociate ~1e-6 and per-primitive overhead is
#: already amortised over real work) on the eager path.  Module global so tests
#: can retarget it.
_HC_COMPILE_MAX_ROWS = 32


def _hc_mixes_split(x, fn, base, scale, hc, iters, norm_eps, hc_eps):
    """``DecoderLayer._mixes`` as a pure function of arrays.

    Byte-identical to :meth:`DecoderLayer._mixes` (``mx.flatten(x, -2, -1)`` is
    the same contiguous merge as its ``reshape(*x.shape[:-2], hc*dim)``, just
    without the dynamic-shape read).  :func:`hc_split_sinkhorn` is the opaque
    Sinkhorn boundary (W32/K3): its ``_sinkhorn_normalise`` tail dispatches to the
    Metal kernel when ``MTPLX_DSV41_SINKHORN_METAL`` is armed on the GPU and to the
    identical stock recurrence otherwise (always so on CPU), so the K3 kernel path
    drops into this tape unchanged.  ``fn``/``base``/``scale`` are the layer's raw
    HC weights (tape inputs)."""
    xf = x.astype(mx.float32)
    flat = mx.flatten(xf, -2, -1)
    rsqrt = mx.rsqrt(mx.mean(mx.square(flat), axis=-1, keepdims=True) + norm_eps)
    mixes = (flat @ fn.astype(mx.float32).T) * rsqrt
    return hc_split_sinkhorn(mixes, scale, base, hc, iters, hc_eps)


def _hc_pre_collapse(x, pre_mix):
    """``DecoderLayer._hc_pre`` as a pure function of arrays."""
    y = mx.sum(pre_mix[..., None] * x.astype(mx.float32), axis=2)
    return y.astype(x.dtype)


def _hc_attn_prep_impl(h, pre_mix, attn_fn, attn_base, attn_scale, attn_norm_w,
                       hc, iters, norm_eps, hc_eps):
    """Pre-attention Hyper-Connection prep: the attn HC mix + the ``pre_mix``
    collapse + attn RMSNorm that produce the attention input.  Pure; the
    attention call (which writes this layer's KV) stays OUTSIDE the tape."""
    attn_pre, attn_post, attn_comb = _hc_mixes_split(
        h, attn_fn, attn_base, attn_scale, hc, iters, norm_eps, hc_eps
    )
    x = _hc_pre_collapse(h, pre_mix)
    attn_input = _rmsnorm(x, attn_norm_w, norm_eps)
    return attn_input, attn_pre, attn_post, attn_comb


def _hc_ffn_prep_impl(attn_out, residual, attn_pre, attn_post, attn_comb,
                      ffn_fn, ffn_base, ffn_scale, ffn_norm_w,
                      hc, iters, norm_eps, hc_eps):
    """Post-attention Hyper-Connection ``post`` + the ffn HC mix + ``attn_pre``
    collapse + ffn RMSNorm that produce the routed-expert input.  Pure; the MoE
    (streamed switch) call stays OUTSIDE the tape.  Returns ``(moe_input,
    moe_residual, ffn_post, ffn_comb, ffn_pre)`` -- ``moe_residual`` is the
    post-attention stream :meth:`DecoderLayer.moe_combine` folds the MoE output
    back into."""
    h = _hc_post_impl(attn_out, residual, attn_post, attn_comb)
    ffn_pre, ffn_post, ffn_comb = _hc_mixes_split(
        h, ffn_fn, ffn_base, ffn_scale, hc, iters, norm_eps, hc_eps
    )
    x = _hc_pre_collapse(h, attn_pre)
    moe_input = _rmsnorm(x, ffn_norm_w, norm_eps)
    return moe_input, h, ffn_post, ffn_comb, ffn_pre


#: One compiled tape per ``(kind, consts)`` pair.  ``mx.compile`` keys its own
#: cache on the *identity* of the wrapped function, so the wrapper is built once
#: and reused; the structural constants (``hc``, ``iters``, ``norm_eps``,
#: ``hc_eps``) are closed over, not passed, because they are not arrays and would
#: be invisible to that cache key.  Fixed-shape (not shapeless) -- MLX keeps one
#: tape per distinct activation shape, which for decode/verify is a handful of
#: tiny repeating shapes.
_HC_COMPILED: dict = {}


def _hc_compiled(kind: str, *consts):
    key = (kind, consts)
    fn = _HC_COMPILED.get(key)
    if fn is None:
        # ``consts`` for the mix tapes is ``(hc, iters, norm_eps, hc_eps,
        # sinkhorn_route)``; the trailing route bool is a CACHE-KEY discriminator
        # only (it never enters the arithmetic -- ``hc_split_sinkhorn`` reads the
        # route itself at trace time), so a runtime flip of
        # ``MTPLX_DSV41_SINKHORN_METAL`` on the GPU re-traces the tape with W32's
        # kernel instead of replaying a stale recurrence tape (V4 keys its ``pre``
        # tape on the same bool).  On CPU the route is always the recurrence, so
        # the key is stable and one tape serves every flag combination.
        if kind == "attn_prep":
            hc, iters, norm_eps, hc_eps = consts[:4]

            def impl(h, pre_mix, attn_fn, attn_base, attn_scale, attn_norm_w):
                return _hc_attn_prep_impl(
                    h, pre_mix, attn_fn, attn_base, attn_scale, attn_norm_w,
                    hc, iters, norm_eps, hc_eps,
                )
        elif kind == "ffn_prep":
            hc, iters, norm_eps, hc_eps = consts[:4]

            def impl(attn_out, residual, attn_pre, attn_post, attn_comb,
                     ffn_fn, ffn_base, ffn_scale, ffn_norm_w):
                return _hc_ffn_prep_impl(
                    attn_out, residual, attn_pre, attn_post, attn_comb,
                    ffn_fn, ffn_base, ffn_scale, ffn_norm_w,
                    hc, iters, norm_eps, hc_eps,
                )
        elif kind == "moe_combine":
            impl = _hc_post_impl  # already a pure array function; no consts
        else:  # pragma: no cover - programming error
            raise ValueError(f"unknown Hyper-Connection tape {kind!r}")
        fn = mx.compile(impl)
        _HC_COMPILED[key] = fn
    return fn


def _hc_use_compile(x: mx.array) -> bool:
    """Is ``x`` in the row regime the compiled HC tape is kept for?

    Reads the module globals ``_HC_COMPILE`` / ``_HC_COMPILE_MAX_ROWS`` at call
    time (not captured) so a test or operator can flip either knob after import.
    ``x`` is a ``[..., hc, dim]`` HC stream, so ``prod(x.shape[:-2])`` is ``b*s``.

    While a W37 stage-timing decode forward is in flight the eager body is forced
    (``_stime.recording()``): the compiled tape is one opaque call, so the
    per-stage ``mx.eval`` fences cannot split its premix/Sinkhorn/combine phases.
    Timing therefore always measures the eager Hyper-Connection path -- exactly
    the shipped ``control`` arm -- regardless of ``MTPLX_DSV41_HC_COMPILE``.
    """
    if not _HC_COMPILE or _stime.recording():
        return False
    rows = 1
    for d in x.shape[:-2]:
        rows *= int(d)
    return rows <= _HC_COMPILE_MAX_ROWS


# ---------------------------------------------------------------------------
# Decoder block (Hyper-Connections around attention + MoE)
# ---------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    """One backbone layer: attention and MoE each wrapped in a Hyper-Connection
    pre/post, with the ``pre_mix`` threaded across sublayers (reference ``Block``,
    model.py L907-994).  ``engram`` is a hook slot (``None`` = no-op) that another
    worker attaches on the engram layers."""

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.rms_norm_eps
        self.hc_eps = args.hc_eps
        self.hc_mult = args.hc_mult
        self.hc_iters = args.hc_sinkhorn_iters
        self.attn = Attention(args, layer_id)
        self.mlp = MoE(layer_id, args)  # `mlp` = switch seam; W11 uses reference (layer_id, args) order
        self.attn_norm_weight = mx.ones((args.hidden_size,))
        self.ffn_norm_weight = mx.ones((args.hidden_size,))
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * args.hidden_size
        self.hc_attn_fn = mx.zeros((mix_hc, hc_dim))
        self.hc_attn_base = mx.zeros((mix_hc,))
        self.hc_attn_scale = mx.zeros((3,))
        self.hc_ffn_fn = mx.zeros((mix_hc, hc_dim))
        self.hc_ffn_base = mx.zeros((mix_hc,))
        self.hc_ffn_scale = mx.zeros((3,))
        #: Engram hook, attached by the engram worker on layers in
        #: ``engram_layer_ids`` (default None = no-op).  Called at the reference's
        #: pre-attention insertion point as
        #: ``engram_hook(hidden[B,L,hc_mult,dim], token_ids[B,L], cache_state) ->
        #: hidden`` and returns the UPDATED residual stream (h + gate*value).
        self.engram_hook = None

    def _mixes(self, x, fn, base, scale):
        """Reference ``Block.hc_mixes``: rsqrt-normalise the flattened hc stream
        (``norm_eps``), project by ``fn``, then split into pre/post/comb with the
        Sinkhorn-normalised comb (``hc_eps``)."""
        xf = x.astype(mx.float32)
        flat = xf.reshape(*xf.shape[:-2], self.hc_mult * xf.shape[-1])
        rsqrt = mx.rsqrt(mx.mean(mx.square(flat), axis=-1, keepdims=True) + self.norm_eps)
        mixes = (flat @ fn.astype(mx.float32).T) * rsqrt
        return hc_split_sinkhorn(mixes, scale, base, self.hc_mult, self.hc_iters, self.hc_eps)

    def _hc_pre(self, x, pre_mix):
        """Collapse the hc copies into one sublayer input with the threaded
        ``pre_mix`` (reference ``Block.hc_pre``)."""
        y = mx.sum(pre_mix[..., None] * x.astype(mx.float32), axis=2)
        return y.astype(x.dtype)

    def attn_and_moe_input(self, h, pre_mix, positions, layer_cache, shared):
        """The layer up to and including the MoE input projection: the attention
        Hyper-Connection (which *writes this layer's KV*), then the ffn HC mix and
        the ``_hc_pre`` + RMSNorm that produce the routed-expert input.

        Returns ``(moe_input, carry, ffn_pre)`` where ``carry`` is everything
        :meth:`moe_combine` needs to fold the MoE output back in (the post-attn
        residual and the ffn post/comb mixes) and ``ffn_pre`` is the next layer's
        ``pre_mix``.  W30's layer-major prefill runs this half for every chunk of a
        layer (in order, so the KV writes stay causal) before issuing one shared
        MoE call; the one-shot / chunk-major path composes it back in
        :meth:`__call__` byte-for-byte.

        When ``MTPLX_DSV41_HC_COMPILE`` is on and the row count is in the small
        decode/verify regime (:func:`_hc_use_compile`), the two Hyper-Connection
        prep chains around the attention call run as compiled tapes (K4); the
        attention call itself -- which mutates the KV cache -- stays outside them.
        The eager branch below is byte-for-byte the original body."""
        if _hc_use_compile(h):
            # Trailing bool keys the mix tapes on the active Sinkhorn route (W32),
            # so an armed GPU kernel drops in / a runtime flip re-traces; on CPU it
            # is always False (recurrence), so it never perturbs the numerics.
            consts = (self.hc_mult, self.hc_iters, self.norm_eps, self.hc_eps,
                      _sinkhorn_use_kernel())
            x, attn_pre, attn_post, attn_comb = _hc_compiled("attn_prep", *consts)(
                h, pre_mix,
                self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale,
                self.attn_norm_weight,
            )
            x = self.attn(x, positions, layer_cache, shared)
            moe_input, residual, ffn_post, ffn_comb, ffn_pre = _hc_compiled(
                "ffn_prep", *consts
            )(
                x, h, attn_pre, attn_post, attn_comb,
                self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale,
                self.ffn_norm_weight,
            )
            return moe_input, (residual, ffn_post, ffn_comb), ffn_pre

        residual = h
        # W37 attention Hyper-Connection prep: HC pre-mix + Sinkhorn, the pre_mix
        # collapse and the attention input RMSNorm (the "attention input" that
        # feeds self.attn).  The attention call itself is timed inside Attention
        # under attn.<mode>; its combine ("hc.combine") folds it back below.
        with _stime.stage("hc.premix_sinkhorn") as _st:
            attn_pre, attn_post, attn_comb = self._mixes(
                h, self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale
            )
            x = self._hc_pre(h, pre_mix)
            x = _rmsnorm(x, self.attn_norm_weight, self.norm_eps)
            _st.add(x, attn_pre, attn_post, attn_comb)
        x = self.attn(x, positions, layer_cache, shared)
        with _stime.stage("hc.combine") as _st:
            h = _hc_post_impl(x, residual, attn_post, attn_comb)
            _st.add(h)

        residual = h
        with _stime.stage("hc.premix_sinkhorn") as _st:
            ffn_pre, ffn_post, ffn_comb = self._mixes(
                h, self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale
            )
            x = self._hc_pre(h, attn_pre)
            moe_input = _rmsnorm(x, self.ffn_norm_weight, self.norm_eps)
            _st.add(moe_input, ffn_pre, ffn_post, ffn_comb)
        return moe_input, (residual, ffn_post, ffn_comb), ffn_pre

    @staticmethod
    def moe_combine(moe_output, carry):
        """Fold the routed-expert output back into the residual stream (the ffn
        Hyper-Connection ``post``), given the ``carry`` from
        :meth:`attn_and_moe_input`."""
        residual, ffn_post, ffn_comb = carry
        if _hc_use_compile(residual):
            return _hc_compiled("moe_combine")(
                moe_output, residual, ffn_post, ffn_comb
            )
        with _stime.stage("hc.combine") as _st:
            out = _hc_post_impl(moe_output, residual, ffn_post, ffn_comb)
            _st.add(out)
        return out

    def __call__(self, h, pre_mix, positions, layer_cache, shared):
        moe_input, carry, ffn_pre = self.attn_and_moe_input(
            h, pre_mix, positions, layer_cache, shared
        )
        x = self.mlp(moe_input)
        h = self.moe_combine(x, carry)
        return h, ffn_pre


# ---------------------------------------------------------------------------
# Token-chunked prefill (W20)
# ---------------------------------------------------------------------------
# The whole 16,384-token prompt fed through one forward materialises, on a
# ratio-2 CSA layer, the attention score `[b, s, H, T]` with T = window(s) +
# compressed(s//2); at the standard shape that single fp32 buffer is
# 16385 * 64 * 24577 * 4 = 103,089,701,120 bytes, over the 86.5 GiB Metal buffer
# cap (see docs/deepseek-v41/W20_REPORT.md).  MLX is lazy, so the buffer is only
# forced at the layer's MoE `mx.eval(indices)` -- which is why the crash surfaces
# in the streamed switch (expert_mlx.py) even though the tensor is the attention
# score.  Token-chunking the forward bounds every per-query prefill transient
# (attention score, indexer score, routed MoE rows) to `chunk` query rows while
# the KV / compressed / index caches accumulate across chunks EXACTLY (W13's
# CompressorState pools group-locally, so pooling is independent of how the rows
# were chunked; the window is a causal mask over absolute positions).
#: Env override for the prefill query-chunk.  A positive int forces that chunk;
#: "0" or a negative value disables chunking (one-shot); unset or "auto" picks
#: the shape-aware size from :func:`_derive_prefill_chunk`.  A per-call
#: ``prefill_chunk`` argument to the forward beats the env.
_PREFILL_CHUNK_ENV = "MTPLX_DSV41_PREFILL_CHUNK"
#: Env override (in GB) for the per-chunk transient budget the auto-derivation
#: targets; the default keeps the dominant transient under ~8 GB.
_PREFILL_CHUNK_TARGET_ENV = "MTPLX_DSV41_PREFILL_CHUNK_TARGET_GB"
_PREFILL_CHUNK_TARGET_DEFAULT_GB = 8.0
#: Env toggle (W30 / kernel-ledger K16) for **layer-major** chunked prefill.
#: Default OFF -> W20's chunk-major driver (each chunk through all layers), which
#: re-streams ~the whole routed bank per chunk (~13x at 16K).  Set truthy to run
#: layer-major (iterate every layer over all chunks before the next layer), so a
#: layer's routed experts stream ONCE across the whole prompt instead of once per
#: chunk.  Only engages when chunking is active (chunk < s); one-shot and decode
#: are byte-identical either way.  Left off by default because the ~13x TTFT win
#: is a GPU-window measurement (KERNEL_LEDGER KG-b), not yet flipped in serving.
_PREFILL_LAYER_MAJOR_ENV = "MTPLX_DSV41_PREFILL_LAYER_MAJOR"
#: Env override (in GB) for the routed-expert transient budget the layer-major
#: MoE row-cap targets; the concatenated MoE call over all chunks is split so its
#: ``rows * top_k * hidden * 4`` fp32 routed-output transient stays under this.
#: Shares the ~8 GB default with the attention-chunk budget above.
_PREFILL_MOE_ROWS_TARGET_ENV = "MTPLX_DSV41_PREFILL_MOE_TARGET_GB"


def _prefill_chunk_target_bytes() -> float:
    raw = os.environ.get(_PREFILL_CHUNK_TARGET_ENV)
    gb = _PREFILL_CHUNK_TARGET_DEFAULT_GB
    if raw:
        try:
            gb = float(raw)
        except ValueError:
            gb = _PREFILL_CHUNK_TARGET_DEFAULT_GB
    return max(1.0, gb) * 1e9


def _prefill_score_bytes_per_row(args: "ModelArgs", s: int) -> int:
    """Bytes of the dominant per-query prefill transient for a context of ``s``
    tokens: one row ``[H, T]`` of the fp32 attention score.

    ``T`` is the window store (all ``s`` tokens during phase-1 prefill) plus, on
    the CSA layer with the smallest positive compress ratio ``r``, that layer's
    ``s // r`` compressed rows -- the layer whose full ``[b, s, H, T]`` score
    overflows the Metal buffer cap.  Per query row it is ``H * T * 4``.
    """
    H = int(args.num_attention_heads)
    ratios = [int(r) for r in (args.compress_ratios or []) if int(r) > 0]
    min_ratio = min(ratios) if ratios else 0
    n_comp = (s // min_ratio) if min_ratio else 0
    T = s + n_comp
    return H * T * 4


def _derive_prefill_chunk(args: "ModelArgs", s: int, target_bytes: float) -> int:
    """The largest query-chunk whose dominant transient stays under
    ``target_bytes`` (shape-aware: it shrinks as the context -- and thus ``T`` --
    grows, so 16K, 64K and beyond stay bounded)."""
    per_row = _prefill_score_bytes_per_row(args, s)
    if per_row <= 0:
        return s
    chunk = int(target_bytes // per_row)
    return max(1, min(chunk, s))


def _resolve_prefill_chunk(args: "ModelArgs", s: int, override) -> int:
    """The prefill query-chunk for a forward over ``s`` tokens.

    Precedence: explicit ``override`` argument > ``MTPLX_DSV41_PREFILL_CHUNK`` env
    > shape-aware auto-derivation.  The caller treats a value ``<= 0`` or ``>= s``
    as one-shot, so decode (``s == 1``) and any prompt below one chunk keep the
    original single-pass forward byte-for-byte.
    """
    if override is not None:
        return int(override)
    raw = os.environ.get(_PREFILL_CHUNK_ENV)
    if raw is not None:
        token = raw.strip().lower()
        if token not in ("", "auto"):
            try:
                return int(token)
            except ValueError:
                pass
    return _derive_prefill_chunk(args, s, _prefill_chunk_target_bytes())


def _env_truthy(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() not in ("", "0", "false", "no", "off", "auto")


def _resolve_prefill_layer_major(override) -> bool:
    """Whether to run W30/K16 layer-major chunked prefill.

    Precedence: explicit ``prefill_layer_major`` argument > the
    ``MTPLX_DSV41_PREFILL_LAYER_MAJOR`` env > default OFF (chunk-major)."""
    if override is not None:
        return bool(override)
    return _env_truthy(_PREFILL_LAYER_MAJOR_ENV)


def _derive_moe_row_cap(args: "ModelArgs", target_bytes: float) -> int:
    """Largest number of rows a single layer-major MoE (``switch_mlp``) call may
    carry so its ``rows * top_k * hidden * 4`` fp32 routed-output transient stays
    under ``target_bytes``.  At 16,384 tokens the whole prompt (~16 K rows) is far
    under the ~65 K-row cap, so a layer streams its bank exactly once; only beyond
    the cap is the concatenated call split (re-reading the split's expert union),
    keeping the transient bounded at any context."""
    top_k = int(args.num_experts_per_tok)
    hidden = int(args.hidden_size)
    per_row = top_k * hidden * 4
    if per_row <= 0:
        return 1 << 62
    return max(1, int(target_bytes // per_row))


def _prefill_moe_row_target_bytes() -> float:
    raw = os.environ.get(_PREFILL_MOE_ROWS_TARGET_ENV)
    gb = _PREFILL_CHUNK_TARGET_DEFAULT_GB
    if raw:
        try:
            gb = float(raw)
        except ValueError:
            gb = _PREFILL_CHUNK_TARGET_DEFAULT_GB
    return max(1.0, gb) * 1e9


# ---------------------------------------------------------------------------
# Backbone + top-level model
# ---------------------------------------------------------------------------
class DeepseekV41Backbone(nn.Module):
    """embed -> expand to hc_mult copies -> 40 CSA2 blocks -> collapse -> norm.
    Holds the per-forward :class:`_SharedRuntime` orchestration."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hc_mult = args.hc_mult
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm_weight = mx.ones((args.hidden_size,))
        #: DSpark MTP target layers (reference ``Transformer.forward`` L1265-1266):
        #: the DSpark head reads the *attention input* (pre-layer, mean over the
        #: hc copies) of these backbone layers, concatenated, as its ``main_hidden``.
        self._mtp_target_layer_ids = tuple(
            int(i) for i in (getattr(args, "dspark_target_layer_ids", None) or ())
        )
        #: Engram row-id prototype (an :class:`~mtplx.engram_v41.NgramHashState`),
        #: attached by :meth:`Model.attach_engram`; ``None`` when engram is not
        #: wired.  It is a *config-only* template -- each KV cache gets its own
        #: streaming clone (:meth:`NgramHashState.fresh`), so decode history is
        #: per-sequence.  Mirrors the reference ``Transformer.engram_hash``.
        self.engram_hash = None

    def __call__(self, input_ids, cache=None, *, prefill_chunk=None,
                 prefill_layer_major=None, return_main_hidden: bool = False):
        b, s = input_ids.shape
        if cache is None:
            # a bare forward (no persistent cache) still needs a per-layer cache
            # built from the config (so kv_source layers get their CompressorState)
            # and its own engram history when the hooks are attached
            engram_state = self.engram_hash.fresh() if self.engram_hash is not None else None
            cache = _make_cache(self.args, engram_state=engram_state)

        chunk = _resolve_prefill_chunk(self.args, s, prefill_chunk)
        if chunk <= 0 or chunk >= s:
            # one-shot (decode, short prompts, or chunking disabled): byte-for-byte
            # the original single-pass forward.  Decode (s == 1) captures the DSpark
            # main_hidden per step through this path.  W47: a one-shot prefill is a
            # single chunk (index 0); tagging is a no-op for decode.
            _stime.set_schedule("one_shot")
            with _stime.chunk(0):
                return self._forward_span(
                    input_ids, cache, return_main_hidden=return_main_hidden
                )

        if _resolve_prefill_layer_major(prefill_layer_major):
            # W30 / kernel-ledger K16: iterate every layer over all chunks before
            # the next layer, so a layer's routed experts stream ONCE across the
            # whole prompt instead of once per chunk (~13x -> 1x bank read at 16K).
            return self._forward_layer_major(
                input_ids, cache, chunk, return_main_hidden=return_main_hidden
            )

        # Token-chunked prefill: each span of at most ``chunk`` query tokens flows
        # through all layers, appending to the SAME accumulating cache (window /
        # compressed KV / index keys / compressor frontier) and advancing the
        # offset, so span k attends over every earlier token via the causal window
        # mask and the reachable compressed rows -- identical to one-shot, but the
        # per-query transients are bounded to ``chunk`` rows.  Each span is
        # evaluated before the next builds its graph (MLX is lazy; without the
        # eval the score buffers would not free between spans).
        #
        # main_hidden must span the WHOLE prompt: the spec engine seeds the draft
        # history with `prompt_hidden[:, :-1, :]` against `prompt_ids[1:]`
        # (generation._append_mtp_history asserts equal lengths), so each span
        # captures its own target-layer hiddens and they are concatenated in
        # position order. The DSpark DRAFT still reads only the FINAL hidden state
        # of the prompt -- `mtp_forward` slices `h[:, -1:, :]` -- so "drafts from
        # the last span" holds without dropping the earlier spans the history needs.
        _stime.set_schedule("chunk_major")
        outputs: List[mx.array] = []
        main_parts: List[Optional[mx.array]] = []
        start = 0
        chunk_idx = 0
        while start < s:
            end = min(start + chunk, s)
            # W47: time the whole chunk (all layers + the cache-state eval) under
            # its index, so ``by_chunk`` shows cost vs chunk index (growing T).
            with _stime.chunk(chunk_idx):
                span = self._forward_span(
                    input_ids[:, start:end], cache, return_main_hidden=return_main_hidden
                )
                if return_main_hidden:
                    h_span, mh_span = span
                    main_parts.append(mh_span)
                else:
                    h_span = span
                    mh_span = None
                self._eval_cache_state(cache, h_span, mh_span)
            outputs.append(h_span)
            start = end
            chunk_idx += 1
        out = mx.concatenate(outputs, axis=1)
        if not return_main_hidden:
            return out
        main_hidden = (
            None
            if any(p is None for p in main_parts)
            else mx.concatenate(main_parts, axis=1)
        )
        return out, main_hidden

    def _forward_span(self, input_ids, cache, *, return_main_hidden: bool = False):
        """Run one contiguous span of query tokens through every layer, appending
        to ``cache`` and advancing its offset.  This is the whole original forward
        body; the one-shot path is exactly this over the full prompt.

        When ``return_main_hidden`` it also returns the DSpark ``main_hidden`` for
        this span -- the concatenated ``dspark_target_layer_ids`` hiddens (the
        attention input, mean over the hc copies; reference L1265-1266).  The
        chunked caller runs this only on the span carrying the position it drafts
        from, so the returned tensor's last row is the final prompt token."""
        b, s = input_ids.shape
        positions = mx.arange(cache.offset, cache.offset + s)

        with _stime.stage("embed") as _st:
            h = self.embed_tokens(input_ids)  # [b, s, dim]
            h = mx.broadcast_to(h[:, :, None, :], (b, s, self.hc_mult, h.shape[-1]))
            # identity one-hot mix over the hc copies (reference make_identity_pre_mix)
            pre_mix = mx.concatenate(
                [mx.ones((b, s, 1)), mx.zeros((b, s, self.hc_mult - 1))], axis=-1
            ).astype(mx.float32)
            _st.add(h, pre_mix)

        # Engram row-id state (owned by the engram worker) is advanced once per
        # span before any engram layer reads it; the n-gram lookback reads the full
        # accumulated history, so per-span advance == one-shot advance (text-only
        # has no image mask).
        engram_state = getattr(cache, "engram_state", None)
        if engram_state is not None:
            with _stime.stage("engram.advance"):
                engram_state.advance(input_ids)  # numpy rolling hash; wall-timed

        shared = cache.new_shared_runtime()
        want_main = return_main_hidden and bool(self._mtp_target_layer_ids)
        main_hiddens: List[mx.array] = []
        # W44 device-route cold recovery: when the barrier-free device path is
        # armed for this (decode / verify) span, snapshot each layer's pre-token
        # cache length (mark) and the span's initial (h, pre_mix) so, if the
        # token-end probe flush shows a miss, we can rewind every layer's cache and
        # re-run the whole span with the miss layers forced onto the fenced path --
        # final token, cache, and engram byte-identical to fenced.
        dr = self._device_route_active(cache, s)
        if dr and os.environ.get("MTPLX_DSV41_DEVICE_ROUTE_PINNED") == "1":
            # W71: establish the pinned working set out-of-band, ONCE, at the
            # prefill->decode boundary -- a pure host ranking over the already-
            # resident set (no gather, no routing barrier), so the pinned-only LUT
            # is populated before this span's first device gather. The switch hook
            # rides the fenced ``mx.eval(indices)`` the device route removes, so it
            # cannot pin; this call does (W64_PINNED_WORKING_SET.md §6). Idempotent
            # (epoch-gated per layer, refresh-aware) and a no-op unless
            # MTPLX_DSV41_PIN_WORKING_SET is armed.
            _dr_runtime = self._device_route_runtime()
            _pin = getattr(_dr_runtime, "pin_working_set", None)
            if callable(_pin):
                _pin()
        dr_marks = [lc.mark() for lc in cache.layers] if dr else None
        dr_initial = (h, pre_mix) if dr else None
        for layer in self.layers:
            if layer.engram_hook is not None and engram_state is not None:
                h = layer.engram_hook(h, input_ids, engram_state)
            # DSpark reads the attention INPUT (pre-layer) of the target layers,
            # mean over the hc copies (reference L1265-1266).
            if want_main and layer.layer_id in self._mtp_target_layer_ids:
                main_hiddens.append(mx.mean(h.astype(mx.float32), axis=2).astype(h.dtype))
            h, pre_mix = layer(h, pre_mix, positions, cache.layers[layer.layer_id], shared)
        if dr:
            h, pre_mix, main_hiddens = self._device_route_recover(
                cache, input_ids, positions, engram_state,
                dr_initial, dr_marks, want_main, main_hiddens, h, pre_mix,
            )
        # Advance AFTER recovery: the rollback is length-based and the engram
        # (advanced once, above) is rewound only by an offset delta, which is zero
        # while offset is still pre-token -- so recovery never disturbs the engram.
        cache.advance(s)

        # final collapse of the hc copies with the last pre_mix, then RMSNorm
        with _stime.stage("final_norm") as _st:
            h = mx.sum(pre_mix[..., None] * h.astype(mx.float32), axis=2).astype(h.dtype)
            out = _rmsnorm(h, self.norm_weight, self.args.rms_norm_eps)
            _st.add(out)
        if not return_main_hidden:
            return out
        main_hidden = mx.concatenate(main_hiddens, axis=-1) if main_hiddens else None
        return out, main_hidden

    # -----------------------------------------------------------------------
    # W44 / KERNEL_LEDGER K24 -- barrier-free device route + cold recovery.
    # See docs/deepseek-v41/W44_DEVICE_ROUTE.md.
    # -----------------------------------------------------------------------
    def _device_route_runtime(self):
        """The shared expert-streaming runtime backing the streamed switches (the
        one that queues/flushes device-route probes), or None for the resident /
        dense path where device route never engages."""

        for layer in self.layers:
            switch = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
            runtime = getattr(switch, "runtime", None)
            if runtime is not None and callable(
                getattr(runtime, "flush_device_route_probes", None)
            ):
                return runtime
        return None

    def _device_route_active(self, cache, s: int) -> bool:
        """Arm the device-route recovery for this span iff a device-route env flag
        is set (W44 ``MTPLX_DSV41_DEVICE_ROUTE`` or W71
        ``MTPLX_DSV41_DEVICE_ROUTE_PINNED``), a streamed runtime is present, and
        the routing phase is DECODE -- exactly the condition under which the switch
        takes the barrier-free device path (so a prefill span never arms it and
        never leaves orphan probes)."""

        if (
            os.environ.get("MTPLX_DSV41_DEVICE_ROUTE") != "1"
            and os.environ.get("MTPLX_DSV41_DEVICE_ROUTE_PINNED") != "1"
        ):
            return False
        if self._device_route_runtime() is None:
            return False
        # Local import keeps the module import graph acyclic at load time.
        from mtplx.expert_streaming import RoutingPhase
        from mtplx.models.expert_mlx import current_expert_routing_phase

        return current_expert_routing_phase(token_count=int(s)) is RoutingPhase.DECODE

    def _device_route_recover(
        self, cache, input_ids, positions, engram_state,
        dr_initial, dr_marks, want_main, main_hiddens, h_final, pre_mix_final,
    ):
        """Make the device-route span byte-identical to the fenced path even when
        cold-token misses occurred, paying only ``m`` routing barriers (``m`` =
        miss layers).

        Flush the token's probes; if any layer missed, rewind EVERY layer to its
        pre-token cache length (``rollback`` -- window / compressed / index /
        compressor frontier; the engram is untouched -- its offset delta is zero
        pre-advance) and re-run the whole span from the span's initial (h, pre_mix)
        on a FRESH shared runtime, forcing ONLY the miss layers onto the fenced
        path (barrier + admit + gather -> correct + admitted); every other layer
        stays on the device path (0 barriers). So barriers = miss layers, and
        re-running the (already-correct) prefix costs compute but zero extra
        barriers -- and it sidesteps the cross-layer CSA candidate/index coupling
        a partial-from-``m1`` restart would have to reconstruct on a fresh
        ``shared``. Repeat until a pass has no misses (bounded; the final fallback
        forces the fenced path for every layer). Returns the corrected
        ``(h, pre_mix, main_hiddens)``; cache + engram end exactly as fenced."""

        runtime = self._device_route_runtime()
        flush = runtime.flush_device_route_probes
        misses = flush()
        if not misses:
            return h_final, pre_mix_final, main_hiddens  # all-hit token: already exact

        n = len(self.layers)
        target_ids = self._mtp_target_layer_ids
        h, pre_mix = h_final, pre_mix_final
        max_passes = 4
        passes = 0
        while misses:
            passes += 1
            if passes > max_passes:
                # Safety net: force the fenced path everywhere. Guarantees
                # termination (no device miss survives a fully fenced re-run).
                force_fenced = set(range(n))
            else:
                force_fenced = {int(lid) for lid, _ in misses}
            # Rewind every layer to pre-token (length-based; engram untouched).
            for lid in range(n):
                cache.layers[lid].rollback(dr_marks[lid])
            shared = cache.new_shared_runtime()  # fresh: clean CSA candidate state
            h, pre_mix = dr_initial
            new_main: List[mx.array] = []
            runtime.set_device_route_force_fenced(force_fenced)
            try:
                for lid in range(n):
                    layer = self.layers[lid]
                    if layer.engram_hook is not None and engram_state is not None:
                        h = layer.engram_hook(h, input_ids, engram_state)
                    if want_main and layer.layer_id in target_ids:
                        new_main.append(
                            mx.mean(h.astype(mx.float32), axis=2).astype(h.dtype)
                        )
                    h, pre_mix = layer(
                        h, pre_mix, positions, cache.layers[lid], shared
                    )
            finally:
                runtime.set_device_route_force_fenced(())
            if want_main and target_ids:
                main_hiddens = new_main
            misses = flush()
        return h, pre_mix, main_hiddens

    @staticmethod
    def _eval_cache_state(cache, *extra):
        """Force the accumulated cache stores (and this span's output) so the
        span's transient buffers free before the next span is built.  Load-bearing
        for the memory bound: MLX is lazy, so an unevaluated chain of appends keeps
        every span's score alive at once."""
        arrays = [a for a in extra if a is not None]
        # getattr with a default keeps this robust to W22's cache-container
        # reshaping (mlx_lm per-layer protocol): it forces whatever stored arrays
        # are present, and simply skips any renamed field.
        for lc in cache.layers:
            for name in ("window", "compress_kv", "index_k"):
                a = getattr(lc, name, None)
                if a is not None:
                    arrays.append(a)
            cs = getattr(lc, "comp_state", None)
            if cs is not None:
                for name in ("raw_kv", "raw_score"):
                    a = getattr(cs, name, None)
                    if a is not None:
                        arrays.append(a)
        if arrays:
            mx.eval(arrays)

    # -----------------------------------------------------------------------
    # W30 / kernel-ledger K16 -- layer-major chunked prefill
    # -----------------------------------------------------------------------
    def _forward_layer_major(self, input_ids, cache, chunk, *,
                             return_main_hidden: bool = False):
        """Layer-major chunked prefill (K16): iterate every layer over all chunks
        before the next layer, so each layer's routed-expert bank is streamed
        ONCE across the whole prompt instead of once per chunk.

        Correctness vs the chunk-major driver (:meth:`_forward_span` per span):

        * **Attention stays causal + per chunk.** Within a layer the chunks run in
          order 0..C-1; chunk ``c`` appends its post-RoPE KV to the same
          append-only layer store and reads the accumulated window, so it attends
          over chunks ``< c`` exactly as one-shot -- and the ``[chunk, H, T]`` score
          transient that motivated W20 stays bounded to one chunk (never
          concatenated).  Each chunk's attention half is evaluated before the next
          chunk builds its graph, so only one score is live at a time.
        * **The MoE reads the bank once.** After a layer's C attention halves, the
          chunks' routed-expert inputs are concatenated and fed to ``mlp`` in one
          ``switch_mlp`` call (row-capped, below), so ``partition_route_waves``
          gathers each of the layer's experts exactly once for the whole prompt.
        * **Hyper-Connection state is resident per chunk.** Every chunk keeps its
          own ``[b, chunk, hc_mult, hidden]`` stream and ``pre_mix`` across the
          whole layer loop (all C together are ``hidden * hc_mult * s * bf16`` --
          0.67 GB at 16 K, well under 1 GB); the ffn ``carry`` is transient within
          a layer.  A per-chunk :class:`SharedAttentionRuntime` threads each
          chunk's compressed-KV / index selection down the stack exactly as its
          span would.
        * **Engram + DSpark unchanged in order.** The engram history is advanced
          once per chunk in position order up front (identical ``_buf``/``_len`` to
          chunk-major) and each chunk's row ids are replayed to the engram hook via
          a per-chunk view, so layers 1/14 write the same residual for the same
          rows.  ``main_hidden`` captures the target-layer input per chunk and
          concatenates in position order, so the DSpark draft seed spans the whole
          prompt (its ``[:, -1:, :]`` slice is still the final prompt token)."""
        _stime.set_schedule("layer_major")
        b, s = input_ids.shape
        offset0 = int(cache.offset)
        spans = [(start, min(start + chunk, s)) for start in range(0, s, chunk)]
        n_chunks = len(spans)

        engram_state = getattr(cache, "engram_state", None)
        want_main = return_main_hidden and bool(self._mtp_target_layer_ids)

        # Per-chunk resident state, built once.  The engram history is advanced in
        # position order here (so `_buf`/`_len` end identical to chunk-major) and
        # each chunk's returned row ids are captured for the hook replay below --
        # the shared `_current` only holds the last advance, so we never read it.
        hs: List[mx.array] = []
        pre_mixes: List[mx.array] = []
        positions_all: List[mx.array] = []
        engram_currents: List[Optional[np.ndarray]] = []
        shareds = [cache.new_shared_runtime() for _ in range(n_chunks)]
        main_hiddens: List[List[mx.array]] = [[] for _ in range(n_chunks)]
        for c, (start, end) in enumerate(spans):
            ids_c = input_ids[:, start:end]
            n_c = end - start
            positions_all.append(mx.arange(offset0 + start, offset0 + end))
            # W47: embed + engram.advance per chunk (tagged by chunk index), so the
            # layer-major flat stages match chunk-major.  ``stage``/``chunk`` are
            # no-ops off / decode; this loop is layer-major-only regardless.
            with _stime.chunk(c):
                with _stime.stage("embed") as _st:
                    h_c = self.embed_tokens(ids_c)
                    h_c = mx.broadcast_to(
                        h_c[:, :, None, :], (b, n_c, self.hc_mult, h_c.shape[-1])
                    )
                    _st.add(h_c)
                hs.append(h_c)
                pre_mixes.append(
                    mx.concatenate(
                        [mx.ones((b, n_c, 1)), mx.zeros((b, n_c, self.hc_mult - 1))],
                        axis=-1,
                    ).astype(mx.float32)
                )
                if engram_state is not None:
                    with _stime.stage("engram.advance"):
                        engram_currents.append(engram_state.advance(ids_c))
                else:
                    engram_currents.append(None)

        row_cap = _derive_moe_row_cap(self.args, _prefill_moe_row_target_bytes())

        for layer in self.layers:
            lc = cache.layers[layer.layer_id]
            is_target = want_main and layer.layer_id in self._mtp_target_layer_ids
            moe_inputs: List[mx.array] = []
            carries: List[tuple] = []
            for c, (start, end) in enumerate(spans):
                # W47: tag this (layer, chunk) attention half with the chunk index
                # so ``by_chunk`` accumulates attention/HC/engram per chunk across
                # every layer (the MoE is batched below, outside any chunk tag).
                with _stime.chunk(c):
                    h_c = hs[c]
                    if layer.engram_hook is not None and engram_state is not None:
                        h_c = layer.engram_hook(
                            h_c, input_ids[:, start:end],
                            _ChunkEngramView(engram_currents[c]),
                        )
                    if is_target:
                        main_hiddens[c].append(
                            mx.mean(h_c.astype(mx.float32), axis=2).astype(h_c.dtype)
                        )
                    moe_in_c, carry_c, ffn_pre_c = layer.attn_and_moe_input(
                        h_c, pre_mixes[c], positions_all[c], lc, shareds[c]
                    )
                    moe_inputs.append(moe_in_c)
                    carries.append(carry_c)
                    pre_mixes[c] = ffn_pre_c
                    # Free this chunk's attention score before the next chunk's
                    # graph is built (only one [chunk, H, T] transient live at once).
                    self._eval_layer_transients(lc, moe_in_c, ffn_pre_c)

            # One routed-expert call per layer over every chunk's rows -> the bank
            # is streamed once.  Split only if the row cap (routed-output transient
            # budget) would be exceeded; at 16 K the whole prompt is one call.
            moe_outputs = self._layer_major_moe(layer, moe_inputs, spans, row_cap)
            for c in range(n_chunks):
                hs[c] = layer.moe_combine(moe_outputs[c], carries[c])
            mx.eval(hs)

        cache.advance(s)

        outputs: List[mx.array] = []
        main_parts: List[Optional[mx.array]] = []
        for c in range(n_chunks):
            with _stime.chunk(c), _stime.stage("final_norm") as _st:
                h = mx.sum(
                    pre_mixes[c][..., None] * hs[c].astype(mx.float32), axis=2
                ).astype(hs[c].dtype)
                out_c = _rmsnorm(h, self.norm_weight, self.args.rms_norm_eps)
                _st.add(out_c)
            outputs.append(out_c)
            main_parts.append(
                mx.concatenate(main_hiddens[c], axis=-1) if main_hiddens[c] else None
            )
        out = mx.concatenate(outputs, axis=1)
        if not return_main_hidden:
            return out
        main_hidden = (
            None
            if any(p is None for p in main_parts)
            else mx.concatenate(main_parts, axis=1)
        )
        return out, main_hidden

    def _layer_major_moe(self, layer, moe_inputs, spans, row_cap):
        """Run this layer's MoE over all chunks, reading the streamed bank once.

        Byte-identity vs chunk-major requires care: the MoE's **resident** parts —
        the router ``gate`` (``xf @ weight.T``) and the shared ``Expert`` — are NOT
        invariant to the row (M) batch size (their fp32 reductions reassociate when
        rows are batched, and a batched gate flips a greedy near-tie in the top-k
        selection on the real model, W30 addendum).  Only the **streamed**
        ``switch_mlp`` per-expert gather is M-invariant.  So this computes the gate
        and shared expert **per chunk** (M == the chunk, exactly as chunk-major)
        and batches **only** the ``switch_mlp`` call across chunks (the read-once
        part) — row-capped so the routed-output transient stays bounded.

        Returns the per-chunk MoE outputs ``[b, chunk, hidden]`` (the input to the
        ffn Hyper-Connection ``moe_combine``)."""
        mlp = layer.mlp
        dim = mlp.dim
        n_chunks = len(moe_inputs)
        lengths = [end - start for start, end in spans]

        # Per chunk (resident, batch == chunk): flatten, route, keep xf for the
        # shared expert.  Byte-identical to chunk-major's per-chunk gate/shared.
        xfs: List[mx.array] = []
        weights: List[mx.array] = []
        indices: List[mx.array] = []
        for c in range(n_chunks):
            xf_c = moe_inputs[c].reshape(-1, dim)
            with _stime.stage("moe.gate_topk") as _st:
                w_c, idx_c = mlp.gate(xf_c)
                _st.add(w_c, idx_c)
            xfs.append(xf_c)
            weights.append(w_c)
            indices.append(idx_c)

        # Group consecutive chunks into ``switch_mlp`` calls of at most `row_cap`
        # rows (bank read once per group; one group at 16 K).  A chunk that alone
        # exceeds the cap still forms its own group.
        groups: List[List[int]] = []
        cur: List[int] = []
        cur_rows = 0
        for c in range(n_chunks):
            rows_c = int(xfs[c].shape[0])
            if cur and cur_rows + rows_c > row_cap:
                groups.append(cur)
                cur, cur_rows = [], 0
            cur.append(c)
            cur_rows += rows_c

        if cur:
            groups.append(cur)

        routed_parts: List[Optional[mx.array]] = [None] * n_chunks
        for grp in groups:
            if len(grp) == 1:
                cat_xf, cat_idx = xfs[grp[0]], indices[grp[0]]
            else:
                cat_xf = mx.concatenate([xfs[c] for c in grp], axis=0)
                cat_idx = mx.concatenate([indices[c] for c in grp], axis=0)
            with _stime.stage("moe.routed_switch") as _st:
                routed = mlp.switch_mlp(cat_xf, cat_idx)  # [rows, top_k, dim]; bank once
                _st.add(routed)
            pos = 0
            for c in grp:
                n_c = int(xfs[c].shape[0])
                routed_parts[c] = routed[pos:pos + n_c]
                pos += n_c

        outputs: List[mx.array] = []
        for c in range(n_chunks):
            b = int(moe_inputs[c].shape[0])
            y_c = mlp.combine_routed(routed_parts[c], weights[c], xfs[c])
            outputs.append(y_c.astype(moe_inputs[c].dtype).reshape(b, lengths[c], dim))
        return outputs

    @staticmethod
    def _eval_layer_transients(lc, *extra):
        """Force one chunk's MoE input (and the just-written layer stores) so its
        ``[chunk, H, T]`` attention score frees before the next chunk builds its
        graph.  Scoped to the current layer's store -- earlier layers are already
        evaluated under the layer-major loop."""
        arrays = [a for a in extra if a is not None]
        for name in ("window", "compress_kv", "index_k"):
            a = getattr(lc, name, None)
            if a is not None:
                arrays.append(a)
        cs = getattr(lc, "comp_state", None)
        if cs is not None:
            for name in ("raw_kv", "raw_score"):
                a = getattr(cs, name, None)
                if a is not None:
                    arrays.append(a)
        if arrays:
            mx.eval(arrays)


class _ChunkEngramView:
    """A per-chunk stand-in for the shared :class:`~mtplx.engram_v41.NgramHashState`
    that the layer-major loop hands the engram hook.

    The hook reads only ``current_row_ids(layer_hash_index)`` and (optionally)
    ``token_mask``.  The shared state's ``_current`` cache holds just the *last*
    advance, so under layer-major (where all chunks are advanced up front) it can
    no longer identify a given chunk's rows; this view carries that chunk's
    captured ``advance`` return (``[B, L, n_layers, n_hash_cols]``) so layers 1/14
    write the same residual for the same rows as chunk-major."""

    __slots__ = ("_current", "token_mask")

    def __init__(self, current, token_mask=None):
        self._current = current
        self.token_mask = token_mask

    def current_row_ids(self, layer_hash_index: int):
        if self._current is None:
            raise RuntimeError("no engram positions captured for this chunk")
        return self._current[:, :, layer_hash_index, :]


def _sanitize_name(name: str) -> str:
    """One checkpoint resident tensor name -> this module's parameter path."""
    # RMSNorm weights are bare-array attributes: X.norm.weight -> X.norm_weight
    name = name.replace("norm.weight", "norm_weight")
    # MoE router correction bias (noaux_tc); ffn -> the hy3 switch-seam name mlp
    name = name.replace("ffn.gate.bias", "ffn.gate.e_score_correction_bias")
    name = name.replace(".ffn.", ".mlp.")
    if name.startswith("layers."):
        return "model." + name
    if name.startswith("embed."):
        return "model.embed_tokens." + name[len("embed."):]
    if name == "norm_weight":
        return "model.norm_weight"
    return name  # head.{weight,scales,biases} stay as-is


#: The default resident format when the artifact carries no ``quantization``
#: block: q8 gs64 affine (the original streamed-artifact codec: every dense
#: projection plus the token/output embeddings).
_RESIDENT_QUANT = {"group_size": 64, "bits": 8, "mode": "affine"}

#: MLX-native float codecs (mx.quantize modes).  For these the artifact repacks
#: each source tensor at its *native* precision, so only the tensors the source
#: actually stores as FP8 are quantised; the projections the source keeps in
#: BF16 (below) and the BF16 token/output embeddings stay dense.
_NATIVE_QUANT_MODES = ("mxfp8", "mxfp4", "nvfp4")

#: Dense projection modules the DeepSeek-V4.1-Flash source keeps in **BF16**
#: (never FP8): the indexer key/weight projections and the compressor kv/gate
#: projections.  Under a native codec these stay dense bf16 (an exact repack of
#: FP8 to mxfp8 is impossible for a bf16 source), so the class predicate must
#: exclude them -- otherwise ``nn.quantize`` would lossily requantise a bf16
#: tensor and the strict resident load would demand nonexistent ``.scales``.
_NATIVE_KEEP_BF16_SUFFIXES = (
    ".attn.indexer.wk", ".attn.indexer.weights_proj",
    ".attn.compressor.wkv", ".attn.compressor.wgate",
)


def _resolve_resident_quant(quantization) -> dict:
    """The resident quant params (group_size/bits/mode) from a config
    ``quantization`` block, defaulting to the original q8 gs64 affine codec.

    Only the block's top-level default is read; per-module overrides in the
    block are for the streamed / MTP experts (``mtp.*`` is not in the text
    parameter tree), never the resident projections handled here.
    """
    if not quantization:
        return dict(_RESIDENT_QUANT)
    return {
        "group_size": int(quantization.get("group_size", _RESIDENT_QUANT["group_size"])),
        "bits": int(quantization.get("bits", _RESIDENT_QUANT["bits"])),
        "mode": str(quantization.get("mode", _RESIDENT_QUANT["mode"])),
    }


def _make_resident_quant_predicate(mode: str, group_size: int):
    """Build the ``nn.quantize`` class predicate for the resident codec.

    Common to every codec: never quantise the streamed routed experts
    (``switch_mlp``), the MoE router gate (bf16 in the checkpoint), or a module
    whose input dim is not ``group_size``-aligned (tiny test configs).  Under a
    native float codec, additionally keep the BF16-source projections
    (:data:`_NATIVE_KEEP_BF16_SUFFIXES`) and the BF16 token/output embeddings
    dense; under affine they are quantised too (the original q8 artifact).
    """
    native = mode in _NATIVE_QUANT_MODES

    def predicate(path: str, module: nn.Module):
        if not hasattr(module, "to_quantized"):
            return False
        if "switch_mlp" in path or path.endswith("mlp.gate"):
            return False
        weight = getattr(module, "weight", None)
        if weight is not None and weight.shape[-1] % group_size != 0:
            return False  # tiny test configs whose dims are not group-aligned stay dense
        if native:
            if path.endswith("embed_tokens") or path == "head" or path.endswith(".head"):
                return False  # embed/head are BF16 at source -> stay dense
            if any(path.endswith(suffix) for suffix in _NATIVE_KEEP_BF16_SUFFIXES):
                return False
        return True

    return predicate


def _is_resident_quant_module(path: str, module: nn.Module) -> bool:
    """Back-compat shim: the affine q8 gs64 predicate (dense projections plus the
    token/output embeddings)."""
    return _make_resident_quant_predicate(_RESIDENT_QUANT["mode"], _RESIDENT_QUANT["group_size"])(
        path, module
    )


def _make_mtp_dense_quant_predicate(group_size: int):
    """Native-mxfp8 predicate for the DSpark head's DENSE tensors (``main_proj``,
    ``attn.*``, ``ffn.shared_experts.*``).  Skips the routed experts (``switch_mlp``,
    mxfp4 -- a separate pass), the MoE router gate (bf16), and the markov/confidence
    heads and stage norms (kept bf16/f32 verbatim, W18_REPORT)."""

    def predicate(path: str, module: nn.Module):
        if not hasattr(module, "to_quantized"):
            return False
        if "switch_mlp" in path or path.endswith("mlp.gate"):
            return False
        if "markov_head" in path or "confidence_head" in path:
            return False
        weight = getattr(module, "weight", None)
        if weight is not None and weight.shape[-1] % group_size != 0:
            return False
        return True

    return predicate


def _make_mtp_expert_quant_predicate(group_size: int):
    """Native-mxfp4 predicate for the DSpark head's RESIDENT routed experts
    (``switch_mlp`` only): the 128 experts per stage the artifact ships as mxfp4
    gs32 (config per-module overrides).  Everything else is left to the dense pass."""

    def predicate(path: str, module: nn.Module):
        if not hasattr(module, "to_quantized"):
            return False
        if "switch_mlp" not in path:
            return False
        weight = getattr(module, "weight", None)
        if weight is not None and weight.shape[-1] % group_size != 0:
            return False
        return True

    return predicate


# --- W40 / K21: output-head codec lever (MTPLX_DSV41_HEAD_MODE) --------------
_HEAD_MODE_ENV = "MTPLX_DSV41_HEAD_MODE"

#: Recognised output-head codecs. ``bf16`` casts the final hidden to the (bf16)
#: head weight dtype BEFORE the matmul -- a bf16 GEMV, f32 logits after -- which
#: removes the default path's fp32-cast trap: the hidden reaches ``self.head`` as
#: float32 (``source.astype(mx.float32)``), so MLX promotes the 1.32 GB bf16 head
#: weight to a 2.64 GB float32 temporary EVERY token before the GEMV.  ``mxfp8``
#: repacks the head at load to native mxfp8 gs32 (E8M0 scales, ~0.66 GB) and
#: ``q8`` to affine 8-bit gs64 (~0.70 GB), both projected through
#: ``quantized_matmul`` with the weight quantised ONCE (not per call).  Unset /
#: empty / ``default`` keeps the current (byte-identical) behaviour.
_HEAD_MODES = ("bf16", "mxfp8", "q8")
_HEAD_MODE_DEFAULT_ALIASES = ("", "default", "off", "none", "control", "0")


def _resolve_head_mode(raw=None) -> Optional[str]:
    """Resolve ``MTPLX_DSV41_HEAD_MODE`` to a codec name or ``None`` (default).

    Read at model construction (a load-time lever -- the weight repack happens
    once at load, not per call), never frozen at import
    (memory/env-flags-read-at-use-not-import).  An unrecognised non-empty value
    raises so a mistyped accuracy lever fails fast rather than silently running
    the default path through a whole benchmark window."""
    val = os.environ.get(_HEAD_MODE_ENV) if raw is None else raw
    val = (val or "").strip().lower()
    if val in _HEAD_MODE_DEFAULT_ALIASES:
        return None
    if val in _HEAD_MODES:
        return val
    raise ValueError(
        f"{_HEAD_MODE_ENV}={val!r} is not one of {_HEAD_MODES} "
        "(or empty/'default' for the current behaviour)"
    )


class _MXFP8Head(nn.Module):
    """Load-time native-mxfp8 (gs32, E8M0 scales, no bias) repack of a dense
    ``[vocab, hidden]`` output head: one ``mx.quantized_matmul`` per call, the
    weight quantised ONCE in ``__init__`` (never per token).  Registered as a
    module so the packed weight + scales flow through ``parameters()`` /
    ``mx.eval`` like any resident.  ``group_size`` (32, mlx 0.32.2's only mxfp8
    group size) must divide ``hidden``."""

    def __init__(self, weight: mx.array, *, group_size: int = 32, bits: int = 8):
        super().__init__()
        packed, scales = mx.quantize(weight, group_size=group_size, bits=bits, mode="mxfp8")
        self.weight = packed
        self.scales = scales
        self.group_size = int(group_size)
        self.bits = int(bits)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.quantized_matmul(
            x, self.weight, scales=self.scales, transpose=True,
            group_size=self.group_size, bits=self.bits, mode="mxfp8",
        )


def _logits_rows_to_keep(logits_rows) -> Optional[int]:
    """Map a K19 ``logits_rows`` selector to a ``keep the last N rows`` count.

    ``"last"`` -> ``1`` (only the final position's logits, the single row decode
    seeds from); ``"all"`` / ``None`` -> ``None`` (head every row, the pre-K19
    default); a positive int (or its string form) -> that many trailing rows.
    """
    if logits_rows is None:
        return None
    if isinstance(logits_rows, str):
        token = logits_rows.strip().lower()
        if token == "last":
            return 1
        if token == "all":
            return None
        try:
            value = int(token)
        except ValueError:
            raise ValueError(
                "logits_rows must be 'last', 'all', or a positive int; "
                f"got {logits_rows!r}"
            )
        return max(1, value)
    return max(1, int(logits_rows))


def _resolve_logits_keep(logits_keep, logits_rows) -> Optional[int]:
    """Normalise the two last-row head selectors into a single trailing-row
    count (``int >= 1``) or ``None`` (every row).

    ``logits_keep`` is the runtime ``forward_ar`` contract (an int row count,
    shared with every other MTPLX backend); ``logits_rows`` is the explicit K19
    alias (``"last"`` / ``"all"`` / int). ``logits_keep=1`` and
    ``logits_rows="last"`` are the same slice. When both are supplied they must
    resolve to the same count, so a caller can never silently ask for two
    different widths.
    """
    keep = None if logits_keep is None else max(1, int(logits_keep))
    if logits_rows is not None:
        rows = _logits_rows_to_keep(logits_rows)
        if keep is not None and rows != keep:
            raise ValueError(
                f"logits_rows={logits_rows!r} and logits_keep={logits_keep!r} "
                "select different row counts"
            )
        keep = rows
    return keep


class Model(nn.Module):
    """DeepSeek-V4.1-Flash text AR model.  ``model.model.layers[i].mlp.switch_mlp``
    is the streamed-expert seam; ``head`` is the (untied) output projection.

    ``quantize`` (default True) converts the resident projections to the codec
    named by ``quantization`` so the streamed-artifact residents load strictly;
    tests that compare against the dense oracle pass ``quantize=False``.
    ``quantization`` is the artifact's config ``quantization`` block (its
    top-level ``group_size``/``bits``/``mode``): ``None`` or ``mode="affine"``
    selects the original q8 gs64 codec (dense projections + embed/head); a native
    float mode (``mxfp8``/``mxfp4``/``nvfp4``) selects the exact-repack codec
    (FP8-source projections quantised, BF16-source projections + embed/head kept
    dense).  ``engram_bank_path`` is stored for the engram worker's wiring (this
    module does not build engram).
    """

    def __init__(self, args: ModelArgs, *, engram_bank_path=None, quantize: bool = True,
                 quantization=None, mtp: bool = False):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.engram_bank_path = engram_bank_path
        self.model = DeepseekV41Backbone(args)
        self.head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        #: The DSpark draft head (worker W23), built only on the opt-in ``mtp``
        #: load path when the config declares MTP stages; ``None`` on the text-only
        #: AR path (which leaves construction and quantisation untouched).
        self.mtp = None
        self.resident_quant = _resolve_resident_quant(quantization) if quantize else None
        if quantize:
            qcfg = self.resident_quant
            nn.quantize(
                self,
                group_size=qcfg["group_size"],
                bits=qcfg["bits"],
                mode=qcfg["mode"],
                class_predicate=_make_resident_quant_predicate(qcfg["mode"], qcfg["group_size"]),
            )
        #: W40 / K21 output-head codec lever, resolved at construction from
        #: ``MTPLX_DSV41_HEAD_MODE``.  The weight repack itself is deferred to
        #: :meth:`apply_head_mode` (the real bf16 head weight is only present
        #: AFTER the resident loader loads it); the loader calls that post-load.
        self._head_mode = _resolve_head_mode()
        self._head_mode_applied = False
        self._head_mode_pricing: Optional[dict] = None
        if mtp and int(getattr(args, "n_mtp_layers", 0) or 0) > 0:
            self._build_mtp_head(quantize=quantize, quantization=quantization)

    def _build_mtp_head(self, *, quantize: bool, quantization) -> None:
        """Construct the DSpark draft head after the backbone quantise (so the
        backbone predicate never touches the MTP tensors).

        When ``quantize`` and the artifact carries a native float codec, the MTP
        stages' resident tensors are quantised in place: the dense projections
        (``main_proj`` / ``attn.*`` / ``ffn.shared_experts.*``) and the 128 routed
        experts (``switch_mlp``, mxfp4 gs32) -- unlike the backbone, whose routed
        experts stream from the bank, the DSpark experts are RESIDENT, so they are
        quantised here through mlx-lm's :class:`SwitchGLU` carrying the reference
        clamped SwiGLU -- the task's ``resident SwitchGLU with the +/-10 clamp``
        alternative to ``mx.gather_qmm(mode="mxfp4")`` (both exist in mlx 0.32.2;
        SwitchGLU's quantised matmul is the resident-expert path here).  See
        W23_REPORT for the artifact-verification gap (the 12 GiB CPU cap forbids
        loading the bank).
        """
        from .deepseek_v41_dspark import DSparkHead

        self.mtp = DSparkHead(self.args)
        if quantize:
            qcfg = self.resident_quant
            mode = qcfg["mode"]
            if mode in _NATIVE_QUANT_MODES:
                # Native repack: MTP dense at mxfp8 (the artifact default), the 128
                # routed experts at mxfp4 gs32 (config per-module overrides).
                nn.quantize(
                    self.mtp,
                    group_size=32,
                    bits=8,
                    mode="mxfp8",
                    class_predicate=_make_mtp_dense_quant_predicate(32),
                )
                nn.quantize(
                    self.mtp,
                    group_size=32,
                    bits=4,
                    mode="mxfp4",
                    class_predicate=_make_mtp_expert_quant_predicate(32),
                )

    def apply_head_mode(self) -> Optional[dict]:
        """Repack the output head per ``MTPLX_DSV41_HEAD_MODE`` (W40 / K21), ONCE,
        after the resident weights are loaded -- the loader calls this post-load,
        when ``self.head.weight`` is the real bf16 head rather than the freshly
        constructed placeholder.  Returns a resident-pricing note (or ``None`` for
        the default codec / a head that cannot be recodec'd) that the loader
        merges into the resident load report so the planner sees the reduced
        footprint (1.32 GB -> ~0.66/0.70 GB for mxfp8/q8).  Idempotent.

        ``bf16`` changes no weight (the resident head stays bf16 1.32 GB); it is a
        forward-only fix and its saving is per-token traffic, not footprint.
        ``mxfp8`` / ``q8`` shrink the resident head and are applied here so the
        quantisation runs once, never per token."""
        if self._head_mode_applied:
            return self._head_mode_pricing
        self._head_mode_applied = True
        mode = self._head_mode
        if mode is None:
            return None
        head = self.head
        weight = getattr(head, "weight", None)
        # Only a dense float head (the native artifact keeps the head bf16) can be
        # recodec'd; an already-quantised head (the affine artifact quantises the
        # head to q8 at load, so it carries ``.scales``) or a non-float weight is
        # a no-op, and the forward falls back to the default path.
        dense = (
            weight is not None
            and getattr(head, "scales", None) is None
            and weight.dtype in (mx.bfloat16, mx.float16, mx.float32)
        )
        if not dense:
            self._head_mode = None
            return None
        before_bytes = int(weight.nbytes)
        if mode == "bf16":
            after_bytes = before_bytes  # resident unchanged; per-token traffic cut
        elif mode == "mxfp8":
            self.head = _MXFP8Head(weight, group_size=32, bits=8)
            mx.eval(self.head.parameters())
            after_bytes = int(self.head.weight.nbytes) + int(self.head.scales.nbytes)
        elif mode == "q8":
            self.head = nn.QuantizedLinear.from_linear(head, group_size=64, bits=8)
            mx.eval(self.head.parameters())
            after_bytes = (
                int(self.head.weight.nbytes)
                + int(self.head.scales.nbytes)
                + int(self.head.biases.nbytes)
            )
        else:  # pragma: no cover - _resolve_head_mode already validated the value
            return None
        self._head_mode_pricing = {
            "head_mode": mode,
            "head_resident_bytes_default": before_bytes,
            "head_resident_bytes_actual": after_bytes,
            "head_resident_saved_bytes": before_bytes - after_bytes,
        }
        return self._head_mode_pricing

    def _apply_head(self, source: mx.array) -> mx.array:
        """Project the (final-normed, hyper-connection-merged) hidden through the
        output head under the active codec (W40 / K21).

        Default (``self._head_mode is None``) is byte-identical to the historical
        ``self.head(source.astype(mx.float32))``: it casts the hidden to float32,
        so the matmul promotes the bf16 head weight to a float32 temporary every
        token (the fp32-cast trap this lever removes)."""
        mode = self._head_mode
        if mode == "bf16":
            # bf16 GEMV over the resident bf16 weight, f32 logits after: the only
            # numeric change is bf16-rounding the hidden (no weight promotion).
            return self.head(source.astype(self.head.weight.dtype)).astype(mx.float32)
        if mode in ("mxfp8", "q8"):
            # weight repacked once at load; f32 hidden -> f32 logits, the weight
            # is dequantised per group inside quantized_matmul, never promoted to
            # a full f32 temporary.
            return self.head(source.astype(mx.float32)).astype(mx.float32)
        return self.head(source.astype(mx.float32))

    def __call__(self, input_ids, cache=None, *, return_hidden: bool = False,
                 emit_logits: bool = True, logits_keep=None, logits_rows=None,
                 input_embeddings=None, hidden_variant=None, prefill_chunk=None,
                 prefill_layer_major=None, **kwargs):
        """Target forward and the MTPLX runtime's ``forward_ar`` surface.

        Plain ``model(ids)`` / ``model(ids, cache=cache)`` is unchanged (returns
        logits).  The extra keywords are the uniform contract
        :meth:`mtplx.runtime.MTPLXRuntime.forward_ar` drives every MTP backend
        through: ``return_hidden`` also returns the DSpark ``main_hidden`` (the
        concatenated target-layer hiddens the draft head consumes); ``emit_logits``
        / ``logits_keep`` skip or restrict the ``lm_head`` matmul; ``hidden_variant``
        is accepted and ignored (V4.1's draft input is one defined tensor);
        ``prefill_chunk`` is W20's token-chunked-prefill knob threaded to the
        backbone; ``prefill_layer_major`` opts a chunked prefill into W30/K16's
        layer-major schedule (bank read once across chunks).  ``input_embeddings``
        (a vision splice) is rejected -- the text path has none.

        **K19 (head only the last row at prefill).** ``logits_rows`` is the
        explicit last-row selector -- ``"last"`` heads only the final position
        (equivalent to the runtime's ``logits_keep=1``), ``"all"``/``None`` keeps
        every row.  At a 16,384-token prefill the full-row head builds a
        ``[1, 16384, vocab]`` f32 logits transient (8.47 GB at vocab 129,280) and
        runs the output GEMM over 16,383 rows the AR path never reads -- decode
        seeds only from the last token.  Slicing the (already final-norm /
        hyper-connection-merged) hidden to the trailing rows before the head drops
        both.  The head lives here, outside the backbone's per-chunk loop, so
        under W20 chunked prefill it is still invoked exactly once -- intermediate
        chunks never touch it -- and the narrowing composes with chunking for free.
        MTP verify (the K+1-row decode-verify batch) is *not* prefill and passes
        neither selector, so it keeps every row unchanged.  The default
        (both ``None``) heads every row and is byte-identical to the pre-K19
        forward.
        """
        if input_embeddings is not None:
            raise ValueError(
                "the DeepSeek-V4.1 text backend does not support input_embeddings "
                "(no vision splice path)"
            )
        # W37 stage timing: arm per-forward recording iff this is a decode step
        # (one query row).  A no-op unless a session is armed; prefill forwards
        # (s > 1) never record, so their multi-GB transients are never fenced.
        _stime_probe = _stime.active()
        if _stime_probe is not None:
            _stime_probe.enter_forward(int(input_ids.shape[1]))
        keep_last = _resolve_logits_keep(logits_keep, logits_rows)
        h, main_hidden = self.model(
            input_ids, cache, prefill_chunk=prefill_chunk,
            prefill_layer_major=prefill_layer_major, return_main_hidden=True
        )
        logits = None
        if emit_logits:
            # Row-independent GEMM: head(h)[:, -k:] == head(h[:, -k:]) exactly, so
            # narrowing the head input never changes the surviving rows' logits.
            with _stime.stage("head") as _st:
                source = h if keep_last is None else h[:, -keep_last:, :]
                logits = self._apply_head(source)
                _st.add(logits)
        if not return_hidden:
            return logits
        return logits, main_hidden

    @property
    def layers(self):
        return self.model.layers

    @property
    def lm_head(self):
        """The output projection under its ``lm_head`` alias.

        DeepSeek-V4.1 names the untied output projection ``head`` (with the W40 /
        K21 ``MTPLX_DSV41_HEAD_MODE`` codec repack applied in
        :meth:`apply_head_mode`).  The served MTP machinery
        (``mtplx.draft_lm_head._install_draft_lm_head``,
        ``mtplx.mtp_patch``) resolves the output projection as ``lm_head`` --
        this property routes that lookup to the same module the AR head path
        (:meth:`_apply_head`) uses, so the draft head is requantized from the
        real (bf16 / mxfp8 / q8) head weight with no fp32-cast reintroduced. It
        is a plain alias, not a new submodule, so ``parameters()`` is unchanged
        (the head is counted once, under ``head``)."""
        return self.head

    # -- DSpark MTP (speculative draft head) -------------------------------
    @property
    def mtp_blocks(self) -> list:
        """The DSpark draft stages (``mtplx.mtp_patch.validate_mtp_support``
        probes this)."""
        return list(getattr(self.mtp, "layers", [])) if self.mtp is not None else []

    @property
    def has_mtp(self) -> bool:
        return bool(self.mtp_blocks)

    def hc_hidden(self, inputs, cache=None):
        """The pre-draft state the DSpark head consumes: the concatenated
        target-layer hiddens (``main_hidden``).  Mirror of the V4 ``hc_hidden``
        surface, adapted to DSpark's target-layer taps."""
        _logits, main_hidden = self.model(inputs, cache, return_main_hidden=True)
        return main_hidden

    def make_mtp_cache(self):
        """One :class:`DSparkStageCache` per DSpark stage (each its own
        sliding-window KV of the main hiddens).  The runtime iterates this list
        and trims each entry on rollback (the W13/PORT_CONTRACT ``trim`` seam);
        the draft consumes all stages together in one ``draft_block``."""
        from .deepseek_v41_dspark import DSparkStageCache

        win = int(self.args.window_size)
        head_dim = int(self.args.head_dim)
        return [DSparkStageCache(win, head_dim) for _ in self.mtp_blocks]

    def _resolve_mtp_caches(self, cache, mtp_cache):
        if mtp_cache is not None:
            if not isinstance(mtp_cache, (list, tuple)):
                raise TypeError("mtp_cache must be the make_mtp_cache() list")
            if cache is not None:
                raise TypeError("pass either cache= or mtp_cache=, not both")
            return list(mtp_cache)
        if cache is None:
            return self.make_mtp_cache()
        return list(cache) if isinstance(cache, (list, tuple)) else [cache]

    def mtp_forward(self, h, input_ids, index: int = 0, cache=None, *, mtp_cache=None,
                    concat_order=None, return_hidden: bool = False,
                    mtp_hidden_variant=None, position_offset=None, mtp_depth=None):
        """DSpark draft for one runtime depth step (the uniform
        ``MTPLXRuntime.draft_mtp`` surface).

        DSpark drafts a whole block (``block_size`` tokens) in one
        ``forward_spec``; this adapter serves that block across the runtime's
        per-depth draft chain -- on the first depth of a cycle (or a fresh
        per-depth cache) it runs the 3-stage net + markov autoregression and
        stashes the block on the stage cache, and deeper depths read the next
        block column.  The runtime's greedy target verify is authoritative
        (``draft cache conditions acceptance only``), so any consistent draft
        here stays lossless.  ``concat_order``/``mtp_hidden_variant`` are
        Qwen-shaped knobs with no V4.1 counterpart (accepted and ignored);
        ``position_offset`` is accepted (the default "cache" position mode passes
        ``None``, and DSpark takes its RoPE offset from its own stage cache)."""
        if self.mtp is None:
            raise RuntimeError("this model carries no DSpark MTP head")
        caches = self._resolve_mtp_caches(cache, mtp_cache)
        embed, head = self.model.embed_tokens, self.head
        depth = 1 if mtp_depth is None else int(mtp_depth)
        b = int(input_ids.shape[0])
        tok = input_ids.reshape(b, -1)[:, -1]  # [b]
        main_h = h[:, -1:, :]                    # [b, 1, D_main]

        pending = getattr(caches[0], "_dspark_pending", None) if caches else None
        if pending is None or depth <= 1:
            out_ids, logits, conf = self.mtp.draft_block(main_h, tok, caches, embed, head)
            if caches:
                caches[0]._dspark_pending = (out_ids, logits, conf, depth)
            col = 0
        else:
            out_ids, logits, conf, base = pending
            col = depth - base
            if col < 0 or col >= int(logits.shape[1]):
                out_ids, logits, conf = self.mtp.draft_block(main_h, tok, caches, embed, head)
                if caches:
                    caches[0]._dspark_pending = (out_ids, logits, conf, depth)
                col = 0
        row_logits = logits[:, col : col + 1, :]  # [b, 1, vocab]
        if not return_hidden:
            return row_logits
        # The fed-back hidden is ignored by the stash path; shape it like
        # main_hidden so the runtime's [:, -1:, :] slice type-checks.
        return row_logits, main_h

    def mtp_update_cache(self, h, input_ids, index: int = 0, *, mtp_cache=None,
                         concat_order=None, mtp_hidden_variant=None,
                         position_offset=None, mtp_depth=None, input_embeddings=None):
        """Append committed main hiddens to the DSpark stage windows and drop any
        pending draft block; returns the last committed main hidden.  Keeps the
        draft's window KV in step with the tokens the target committed."""
        if input_embeddings is not None:
            raise ValueError(
                "deepseek_v41 DSpark has no vision splice path (input_embeddings)"
            )
        caches = self._resolve_mtp_caches(None, mtp_cache)
        self.mtp.seed_main(h, caches)
        if caches:
            caches[0]._dspark_pending = None
        return h[:, -1:, :]

    def make_cache(self):
        # W13's factory builds one LayerAttentionCache per layer from the config
        # (window ring, CompressorState frontier on ratio>1 kv_source layers) and a
        # per-sequence SharedAttentionRuntime; each sequence gets its own streaming
        # engram history (the config template is shared).
        engram_state = (
            self.model.engram_hash.fresh() if self.model.engram_hash is not None else None
        )
        return _make_cache(self.args, engram_state=engram_state)

    def stage_timing_report(self):
        """W37 decode stage-timing census, or ``None`` when no session is armed.

        Reads the module-level :mod:`deepseek_v41_stage_timing` probe installed
        for the current decode loop (see ``scripts/deepseek_v41/
        ab_decode_env_levers.py --stage-timing``): per-stage mean ms/token and
        counts over the decode steps, plus ``stage_sum_ms`` (their total) and
        ``frame_wall_ms`` (the reference wall the stages tile).  When the route-
        stage probe (``MTPLX_ROUTE_STAGE_PROBE``) is also armed, its snapshot is
        merged under ``route_stage`` for the switch-internal breakdown (the
        confirmed ~40 ``hot.eval_indices`` barriers/token, and -- streamed only --
        the miss-I/O and gather brackets).  The fences inflate absolute time, so
        the ratios between stages are the signal, not the totals."""
        report = _stime.report()
        if report is None:
            return None
        try:
            from mtplx import expert_route_probe as _route_probe

            if getattr(_route_probe, "ENABLED", False):
                report["route_stage"] = _route_probe.snapshot()
        except Exception:  # pragma: no cover - route probe is best-effort context
            pass
        return report

    def attach_engram(self, engram_dir, *, tokenizer=None, cache_bytes=None):
        """Build the real Engram hooks (layers 1 and 14) from the on-disk artifact.

        The serve-path loader calls this once, after the residents are loaded, so
        the cheap unit-test constructor stays free of the heavy engram I/O (the
        tokenizer walk + the 104 GiB row banks + the 319 MiB resident sidecar).
        Concretely, for every engram layer in ``engram-manifest.json``'s
        ``hashing.layer_ids`` it opens that layer's affine-q8 row bank
        (:class:`~mtplx.engram_bank.EngramBank`, byte budget from
        ``MTPLX_ENGRAM_CACHE_LIMIT``), loads the resident ``wkv``/``q_weight``/
        ``k_weight`` sidecar (:func:`~mtplx.engram_v41.load_engram_residents`) and
        builds the :class:`~mtplx.engram_v41.EngramV41` hook
        (:meth:`EngramResidents.build_module`), attaching it to
        ``self.model.layers[layer_id].engram_hook``.  It also builds one
        :class:`~mtplx.engram_v41.NgramHashState` prototype (its compressed token
        map comes from the artifact tokenizer) and stashes it on the backbone; each
        :meth:`make_cache` clones a fresh per-sequence copy.

        ``engram_dir`` is ``<artifact>/engram``; ``tokenizer`` defaults to the
        artifact's HuggingFace tokenizer (``engram_dir``'s parent), and
        ``cache_bytes`` to the ``MTPLX_ENGRAM_CACHE_LIMIT`` byte budget.
        """
        import json as _json
        from pathlib import Path as _Path

        from ..engram_bank import EngramBank
        from ..engram_v41 import (
            NgramHashState,
            load_engram_residents,
            load_engram_tokenizer,
        )
        from ..ngram_row_cache import cache_bytes_from_env

        engram_dir = _Path(engram_dir)
        manifest = _json.loads((engram_dir / "engram-manifest.json").read_text())
        layer_ids = tuple(int(x) for x in manifest["hashing"]["layer_ids"])
        if cache_bytes is None:
            cache_bytes = cache_bytes_from_env()
        if tokenizer is None:
            # the HF tokenizer lives at the artifact root (engram_dir's parent);
            # its compressed token map drives every engram hash multiplier
            tokenizer = load_engram_tokenizer(engram_dir.parent)

        # one hash-state prototype (config only); make_cache clones per sequence
        self.model.engram_hash = NgramHashState.from_manifest(manifest, tokenizer)

        # keep the banks alive on the model: EngramBank.__del__ would close the
        # NGramRowCache the hooks hold, so we must not let them be collected
        self._engram_banks = []
        for layer_id in layer_ids:
            bank = EngramBank.open(engram_dir, layer_id, cache_bytes=cache_bytes)
            residents = load_engram_residents(engram_dir, layer_id)
            hook = residents.build_module(
                row_cache=bank.cache,
                layer_hash_index=layer_ids.index(layer_id),
                norm_eps=self.args.rms_norm_eps,
            )
            self.model.layers[layer_id].engram_hook = hook
            self._engram_banks.append(bank)
        return layer_ids

    def sanitize(self, weights: dict) -> dict:
        """Map the DeepSeek checkpoint's resident tensor names onto this module's
        parameter paths (see W1_REPORT for the full table).

        ``layers.N.ffn.*`` -> ``model.layers.N.mlp.*`` (the hy3 switch-seam name),
        ``ffn.gate.bias`` -> ``mlp.gate.e_score_correction_bias`` (dropping the
        text-unused ``ffn.gate.bias_vl``), the bare 1-D residents onto their array
        attributes, and ``embed``/``head``/``norm`` onto their module paths.  The
        routed ``ffn.experts.*`` are streamed (never in the resident dict).
        """
        out = {}
        keep_mtp = self.mtp is not None
        mtp_items: dict[str, object] = {}
        for name, value in weights.items():
            if name.startswith(("vision.", "aligner.", "image_")):
                continue
            if name.startswith("mtp."):
                # Text-only AR (no DSpark head built): drop the MTP residents, as
                # phase 1 did.  On the opt-in ``mtp=True`` path the head is built,
                # so map ``mtp.{i}.*`` onto the DSpark head's parameter paths.
                if keep_mtp:
                    mtp_items[name] = value
                continue
            if name.endswith(".bias_vl"):
                continue  # VL routing bias, unused on the text path
            out[_sanitize_name(name)] = value
        if mtp_items:
            out.update(_map_mtp_residents(mtp_items))
        return out


# ---------------------------------------------------------------------------
# DSpark MTP resident name mapping + MTPLX runtime binding (worker W23)
# ---------------------------------------------------------------------------
#: DeepSeek shared-expert / routed-expert FFN weight -> mlx-lm SwitchGLU proj.
_MTP_W_TO_PROJ = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}


def _map_mtp_residents(items: dict) -> dict:
    """Map the checkpoint ``mtp.{i}.*`` residents onto the DSpark head parameter
    paths ``mtp.layers.{i}.*`` (worker W23's opt-in load path).

    Dense tensors are renamed with the same transforms the backbone
    :func:`_sanitize_name` applies (``norm.weight`` -> ``norm_weight``,
    ``ffn.gate.bias`` -> ``mlp.gate.e_score_correction_bias``, ``ffn.`` ->
    ``mlp.``).  The 128 per-expert routed tensors ``mtp.{i}.ffn.experts.{e}.w{j}``
    (RESIDENT mxfp4, W18_REPORT) are STACKED over the expert axis into the
    mlx-lm :class:`SwitchGLU` projections (``gate_proj``/``up_proj``/``down_proj``).

    NOTE (W23_REPORT): the DENSE name mapping is gated against the model's own
    parameter tree by ``tests/models/test_deepseek_v41_dspark.py``.  The mxfp4
    stacked-expert load is NOT verified end to end against the real artifact --
    the 12 GiB CPU cap forbids loading the 376 GiB bank -- so it is committed as
    an implemented-but-unverified path (see PORT_CONTRACT W23)."""
    import re

    experts: dict[tuple, dict[int, object]] = {}
    out: dict[str, object] = {}
    for name, value in items.items():
        m = re.match(r"mtp\.(\d+)\.(.+)$", name)
        if not m:
            continue
        stage, rest = m.group(1), m.group(2)
        exp = re.match(r"ffn\.experts\.(\d+)\.(w[123])\.(.+)$", rest)
        if exp:
            eidx, w, leaf = int(exp.group(1)), exp.group(2), exp.group(3)
            experts.setdefault((stage, _MTP_W_TO_PROJ[w], leaf), {})[eidx] = value
            continue
        rest = rest.replace("ffn.gate.bias", "ffn.gate.e_score_correction_bias")
        rest = rest.replace("norm.weight", "norm_weight")
        if rest.startswith("ffn."):
            rest = "mlp." + rest[len("ffn."):]
        out[f"mtp.layers.{stage}.{rest}"] = value
    for (stage, proj, leaf), by_idx in experts.items():
        ordered = [by_idx[i] for i in sorted(by_idx)]
        out[f"mtp.layers.{stage}.mlp.switch_mlp.{proj}.{leaf}"] = mx.stack(ordered, axis=0)
    return out


def is_deepseek_v41_mtp_config(config: dict) -> bool:
    """Does this artifact declare a DeepSeek-V4.1 DSpark draft head?

    Keys on ``model_type in {deepseek_v41, deepseek_v41_text}`` (top-level or the
    nested ``text_config``) plus a positive stage count (``n_mtp_layers`` /
    ``num_nextn_predict_layers``).  Weight presence is decided later by the load
    path (the head is built only on the opt-in ``mtp=True`` path); a config that
    declares stages but ships no built head degrades to AR via the injector."""
    cfg = config or {}

    def _mt(d):
        return str((d or {}).get("model_type") or "").lower()

    types = {_mt(cfg), _mt(cfg.get("text_config"))}
    if not ({"deepseek_v41", "deepseek_v41_text"} & types):
        return False

    def _stages(d):
        d = d or {}
        return int(d.get("n_mtp_layers") or d.get("num_nextn_predict_layers") or 0)

    return max(_stages(cfg), _stages(cfg.get("text_config"))) > 0


def inject_deepseek_v41_mtp_support(model, path=None, config=None, contract=None) -> bool:
    """Enable the DSpark speculative lane on an already-loaded DeepSeek-V4.1 model.

    Like the sibling native draft head (:func:`mtplx.models.deepseek_v4.
    inject_deepseek_v4_mtp_support`), there is nothing to graft: the DSpark head
    binds through the opt-in ``mtp=True`` load path from the checkpoint's
    ``mtp.{0,1,2}.*`` tensors, and :class:`Model` already carries the runtime's
    draft surface (``__call__(return_hidden=...)``, :meth:`Model.mtp_forward`,
    :meth:`Model.mtp_update_cache`, :meth:`Model.make_mtp_cache`).  This publishes
    that fact in the shape ``mtplx.mtp_patch.validate_mtp_support`` checks (the
    :class:`~mtplx.models.deepseek_v41_dspark.DSparkHead` already answers
    ``.layers``), and returns False -- the degrade-to-autoregressive signal --
    for a checkpoint whose head was not built.  The generic Qwen graft
    (``inject_mtp_support``) cannot serve this backend (it builds a qwen3_5
    ``_MTPModule`` and grafts a sidecar), so this needs its own runtime dispatch
    arm; ``is_deepseek_v4_mtp_config`` never matches a V4.1 config."""
    if not is_deepseek_v41_mtp_config(config or {}):
        return False
    return bool(getattr(model, "mtp_blocks", None))
