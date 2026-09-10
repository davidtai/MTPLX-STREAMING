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
from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.switch_layers import SwitchGLU

# Arithmetic reused verbatim from the V4 backend (imported, not copied).  Each is
# a pure function of arrays / an activation with no V4-specific state; the line
# refs to inference/model.py they transcribe are in W1_REPORT.md.
from mtplx.models.deepseek_v4 import (
    ClampedSwiGLU,
    MoEGate,
    _apply_interleaved_rope,
    _hc_post_impl,
    _yarn_inv_freq,
    hc_split_sinkhorn,
)

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


def _cos_sin(inv_freq: mx.array, positions: mx.array):
    """``cos``/``sin`` tables ``[len(positions), rope_head_dim//2]`` in fp32."""
    ang = positions.astype(mx.float32)[:, None] * inv_freq[None, :]
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


# ---------------------------------------------------------------------------
# Shared cross-layer attention runtime (reference SharedAttentionRuntime)
# ---------------------------------------------------------------------------
class _SharedRuntime:
    """One slot each for what a source layer hands down the stack this forward:
    the group's compressed KV and index keys, the selected compressed-row mask,
    and the layer-20 candidate-block mask.  Layers run in order and every source
    writes before its consumers read, so one slot is enough (reference
    ``SharedAttentionRuntime``)."""

    def __init__(self):
        self.compress_kv = None   # [b, n_comp, head_dim]
        self.index_k = None       # [b, n_comp, index_head_dim]
        self.topk_mask = None     # [b, s, n_comp] bool: selected compressed rows
        self.candidates = None    # [b, s, n_comp] bool: layer-20 candidate blocks


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

    def prefill(self, x: mx.array):
        """Pool a whole ``start_pos == 0`` chunk; returns (latents, remainder_state).

        ``remainder_state`` is (kv_tail, score_tail) for the trailing partial group,
        or ``None`` when the chunk length is a multiple of ratio / ratio == 1."""
        ratio = self.ratio
        if ratio == 1:
            return _rmsnorm(self.wkv(x), self.norm_weight, self.eps), None
        b, s, _ = x.shape
        xf = x.astype(mx.float32)
        kv = self.wkv(xf)
        score = self.wgate(xf)
        remainder = s % ratio
        cutoff = s - remainder
        state = None
        if remainder:
            state = (kv[:, cutoff:], score[:, cutoff:])
            kv = kv[:, :cutoff]
            score = score[:, :cutoff]
        kv = kv.reshape(b, -1, ratio, kv.shape[-1])
        score = score.reshape(b, -1, ratio, score.shape[-1])
        pooled = mx.sum(kv * mx.softmax(score, axis=2), axis=2)
        return _rmsnorm(pooled, self.norm_weight, self.eps), state

    def step(self, x, state):
        """Advance one decode token; returns (latent_or_None, new_state).

        ``ratio == 1`` yields one latent every step; ``ratio > 1`` accumulates raw
        (kv, score) rows until the group fills, then pools them (reference decode
        path, model.py L476-485).  ``state`` is (kv_acc, score_acc) or None."""
        if self.ratio == 1:
            return _rmsnorm(self.wkv(x), self.norm_weight, self.eps), None
        xf = x.astype(mx.float32)
        kv = self.wkv(xf)
        score = self.wgate(xf)
        if state is not None and state[0] is not None:
            kv = mx.concatenate([state[0], kv], axis=1)
            score = mx.concatenate([state[1], score], axis=1)
        if kv.shape[1] == self.ratio:
            pooled = mx.sum(kv * mx.softmax(score, axis=1), axis=1, keepdims=True)
            return _rmsnorm(pooled, self.norm_weight, self.eps), None
        return None, (kv, score)


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
class _GroupedOLoraDown(nn.Module):
    """The block-diagonal ``wo_a`` down-projection over ``o_groups`` (reference
    model.py L785-787).  Stored as one ``[n_groups*o_lora_rank, in_per_group]``
    matrix (checkpoint ``wo_a.weight``) and reshaped to ``[g, o_lora_rank, in]``
    so each group projects only its own heads."""

    def __init__(self, n_groups: int, o_lora_rank: int, in_per_group: int):
        super().__init__()
        self.n_groups = n_groups
        self.o_lora_rank = o_lora_rank
        self.weight = mx.zeros((n_groups * o_lora_rank, in_per_group))

    def __call__(self, o: mx.array) -> mx.array:  # o: [b, s, n_groups, in_per_group]
        w = self.weight.reshape(self.n_groups, self.o_lora_rank, -1)
        return mx.einsum("bsgd,grd->bsgr", o.astype(mx.float32), w.astype(mx.float32))


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
        self.wo_a = _GroupedOLoraDown(self.n_groups, self.o_lora_rank, in_per_group)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=False)

        self.compressor = Compressor(args, self.compress_ratio) if self.is_kv_source else None
        self.indexer = Indexer(args, owns_k=self.is_kv_source) if self.is_index_source else None

        if self.compress_ratio:
            self.inv_freq = _compress_inv_freq(args)
        else:
            self.inv_freq = _swa_inv_freq(args)

    def _sparse_attend(self, q, KV, attend):
        """One softmax over the concatenated KV with a per-head sink (value 0),
        equivalent to reference ``sparse_attn``.  q: [b,s,H,hd], KV: [b,T,hd],
        attend: [b,s,T] bool."""
        b, s, H, _ = q.shape
        scores = mx.einsum("bshd,btd->bsht", q.astype(mx.float32), KV.astype(mx.float32))
        scores = scores * self.softmax_scale
        scores = mx.where(attend[:, :, None, :], scores, float("-inf"))
        sink = mx.broadcast_to(self.attn_sink.astype(mx.float32).reshape(1, 1, H, 1), (b, s, H, 1))
        full = mx.concatenate([scores, sink], axis=-1)
        w = mx.softmax(full, axis=-1)[..., : KV.shape[1]]  # drop the sink column (value 0)
        return mx.einsum("bsht,btd->bshd", w, KV.astype(mx.float32))

    def _publish_compressed(self, x, positions, layer_cache, shared, qcos, qsin):
        """Full/kv_source layer: compress this chunk (prefill) or advance the
        partial group (decode), RoPE new latents at their group positions, derive
        index keys, append both to the layer cache and publish to the shared
        runtime."""
        ratio = self.compress_ratio
        if int(positions[0]) == 0:
            latent_pre, layer_cache.comp_state = self.compressor.prefill(x)
        else:
            latent_pre, layer_cache.comp_state = self.compressor.step(
                x, layer_cache.comp_state
            )
        if latent_pre is not None and latent_pre.shape[1] > 0:
            n_prev = 0 if layer_cache.compress_kv is None else layer_cache.compress_kv.shape[1]
            n_new = latent_pre.shape[1]
            group_pos = mx.arange(n_prev, n_prev + n_new) * ratio
            gcos, gsin = _cos_sin(self.inv_freq, group_pos)
            compress_new = _rope_last(latent_pre, gcos, gsin)
            index_new = self.indexer.keys(latent_pre, gcos, gsin)
            layer_cache.compress_kv = _grow(layer_cache.compress_kv, compress_new)
            layer_cache.index_k = _grow(layer_cache.index_k, index_new)
        shared.compress_kv = layer_cache.compress_kv
        shared.index_k = layer_cache.index_k

    def _compressed(self, x, qr, positions, layer_cache, shared, qcos, qsin):
        """Returns (compress_kv [b,n_comp,hd], compress_attend [b,s,n_comp] bool)
        for the concatenated-KV attention, running the CSA2 mode dispatch."""
        ratio = self.compress_ratio
        if self.is_kv_source:
            self._publish_compressed(x, positions, layer_cache, shared, qcos, qsin)
        compress_kv = shared.compress_kv
        index_k = shared.index_k
        n_comp = compress_kv.shape[1]
        compress_lens = (positions + 1) // ratio  # [s] reachable compressed rows

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
        else:  # Reuse: read the source's selection
            mask = shared.topk_mask
        if self.capture_selection:
            self.last_selection = mask
            self.last_candidates = shared.candidates
        return compress_kv, mask

    def __call__(self, x, positions, layer_cache, shared):
        b, s, _ = x.shape
        H, hd, rd = self.n_heads, self.head_dim, self.rope_head_dim
        qcos, qsin = _cos_sin(self.inv_freq, positions)

        qr = _rmsnorm(self.wq_a(x), self.q_norm_weight, self.eps)
        q = self.wq_b(qr).reshape(b, s, H, hd)
        q = _rope_last(q, qcos, qsin)

        kv_new = _rmsnorm(self.wkv(x), self.kv_norm_weight, self.eps)
        kv_new = _rope_last(kv_new, qcos, qsin)  # window kv roped at its own token positions
        layer_cache.window = _grow(layer_cache.window, kv_new)
        window_all = layer_cache.window
        wpos = mx.arange(window_all.shape[1])
        qp = positions[:, None]
        wp = wpos[None, :]
        win_attend = (wp <= qp) & (wp > qp - self.window_size)  # [s, Tw]
        attend = mx.broadcast_to(win_attend[None], (b, s, window_all.shape[1]))
        KV = window_all

        if self.compress_ratio:
            compress_kv, comp_attend = self._compressed(
                x, qr, positions, layer_cache, shared, qcos, qsin
            )
            KV = mx.concatenate([window_all, compress_kv], axis=1)
            attend = mx.concatenate([attend, comp_attend], axis=-1)

        o = self._sparse_attend(q, KV, attend)
        o = _rope_last(o, qcos, qsin, inverse=True)
        o = o.reshape(b, s, self.n_groups, -1)
        o = self.wo_a(o)
        return self.wo_b(o.reshape(b, s, -1))


def _grow(rows, new):
    """Append ``new`` rows along the sequence axis of an append-only cache."""
    if rows is None:
        return new
    return mx.concatenate([rows, new], axis=1)


# ---------------------------------------------------------------------------
# MoE (routed switch seam + shared expert)
# ---------------------------------------------------------------------------
class _SharedExpert(nn.Module):
    """The always-on shared SwiGLU expert (reference ``Expert``, model.py L830-851),
    named ``w1``/``w2``/``w3`` to match the checkpoint.  ``w3`` (up) is clamped
    two-sided, ``w1`` (gate) only from above; the product runs in fp32."""

    def __init__(self, args: ModelArgs):
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


class DeepseekV41MoE(nn.Module):
    """Top-6 routed experts + one shared expert (reference ``MoE``, model.py
    L854-904).  ``switch_mlp`` is the seam the streaming runtime rebinds
    (``bind_streamed_switches``); the resident ``SwitchGLU`` is the test-only
    fallback.  The gate (``sqrtsoftplus``/``noaux_tc``) is reused from the V4
    backend."""

    def __init__(self, args: ModelArgs, layer_id: int):
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
        self.mlp = DeepseekV41MoE(args, layer_id)  # `mlp` = the switch seam name
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
        #: Hook attached by the engram worker on layers in ``engram_layer_ids``;
        #: ``engram(hidden[b,s,hc,dim], token_ids[b,s]) -> hidden``.  None = no-op.
        self.engram = None

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

    def __call__(self, h, pre_mix, positions, layer_cache, shared):
        residual = h
        attn_pre, attn_post, attn_comb = self._mixes(
            h, self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale
        )
        x = self._hc_pre(h, pre_mix)
        x = _rmsnorm(x, self.attn_norm_weight, self.norm_eps)
        x = self.attn(x, positions, layer_cache, shared)
        h = _hc_post_impl(x, residual, attn_post, attn_comb)

        residual = h
        ffn_pre, ffn_post, ffn_comb = self._mixes(
            h, self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale
        )
        x = self._hc_pre(h, attn_pre)
        x = _rmsnorm(x, self.ffn_norm_weight, self.norm_eps)
        x = self.mlp(x)
        h = _hc_post_impl(x, residual, ffn_post, ffn_comb)
        return h, ffn_pre


# ---------------------------------------------------------------------------
# Cache with the trim/rollback seam
# ---------------------------------------------------------------------------
class _LayerCache:
    """Append-only window / compressed-KV / index-key rows for one layer, plus a
    mark/rollback seam.  Full history is kept and the sliding window is realised
    by the attention mask (equivalent to the reference ring buffer for the
    positions any query can still reach); phase 2 swaps it for a bounded ring."""

    def __init__(self):
        self.window = None
        self.compress_kv = None
        self.index_k = None
        self.comp_state = None  # (kv_acc, score_acc) partial compressor group, or None

    def mark(self):
        def n(a):
            return 0 if a is None else a.shape[1]
        # comp_state holds immutable arrays, so the tuple itself is the snapshot
        return (n(self.window), n(self.compress_kv), n(self.index_k), self.comp_state)

    def rollback(self, mark):
        nw, nc, ni, comp_state = mark
        self.window = _truncate(self.window, nw)
        self.compress_kv = _truncate(self.compress_kv, nc)
        self.index_k = _truncate(self.index_k, ni)
        self.comp_state = comp_state


def _truncate(rows, n):
    if rows is None or n == 0:
        return None if n == 0 else rows
    return rows[:, :n]


class DeepseekV41Cache:
    """Per-model KV cache: one ``_LayerCache`` per layer plus the running token
    offset.  ``mark``/``rollback`` undo a decoded tail (the trim/rollback seam of
    the V4 ``DeepseekV4Cache``), restoring identical logits."""

    def __init__(self, n_layers: int):
        self.layers = [_LayerCache() for _ in range(n_layers)]
        self.offset = 0

    def mark(self):
        return (self.offset, [lc.mark() for lc in self.layers])

    def rollback(self, mark):
        self.offset, layer_marks = mark
        for lc, m in zip(self.layers, layer_marks):
            lc.rollback(m)


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

    def __call__(self, input_ids, cache=None):
        b, s = input_ids.shape
        if cache is None:
            cache = DeepseekV41Cache(len(self.layers))
        positions = mx.arange(cache.offset, cache.offset + s)

        h = self.embed_tokens(input_ids)  # [b, s, dim]
        h = mx.broadcast_to(h[:, :, None, :], (b, s, self.hc_mult, h.shape[-1]))
        # identity one-hot mix over the hc copies (reference make_identity_pre_mix)
        pre_mix = mx.concatenate(
            [mx.ones((b, s, 1)), mx.zeros((b, s, self.hc_mult - 1))], axis=-1
        ).astype(mx.float32)

        shared = _SharedRuntime()
        for layer in self.layers:
            if layer.engram is not None:
                h = layer.engram(h, input_ids)
            h, pre_mix = layer(h, pre_mix, positions, cache.layers[layer.layer_id], shared)
        cache.offset += s

        # final collapse of the hc copies with the last pre_mix, then RMSNorm
        h = mx.sum(pre_mix[..., None] * h.astype(mx.float32), axis=2).astype(h.dtype)
        return _rmsnorm(h, self.norm_weight, self.args.rms_norm_eps)


class Model(nn.Module):
    """DeepSeek-V4.1-Flash text AR model.  ``model.layers[i].mlp.switch_mlp`` is
    the streamed-expert seam; ``head`` is the (untied) output projection."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = DeepseekV41Backbone(args)
        self.head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, input_ids, cache=None):
        h = self.model(input_ids, cache)
        return self.head(h.astype(mx.float32))

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return DeepseekV41Cache(len(self.model.layers))
