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
import numpy as np

from mlx_lm.models.base import BaseModelArgs

# Pure arithmetic reused from the V4 backend (imported, not copied), each verified
# line-by-line against inference/model.py and re-checked by the W6 numpy parity
# oracle (tests/models/test_deepseek_v41_parity.py):
#   _yarn_inv_freq         <- precompute_freqs_cis YaRN ramp (model.py L367-395)
#   _apply_interleaved_rope<- apply_rotary_emb adjacent-pair rotation (L399-414)
#   hc_split_sinkhorn      <- kernel.py hc_split_sinkhorn (pre/post/comb + Sinkhorn)
#   _hc_post_impl          <- Block.hc_post (post*x + sum_j comb[j,k]*residual[j])
from mtplx.models.deepseek_v4 import (
    _apply_interleaved_rope,
    _hc_post_impl,
    _yarn_inv_freq,
    hc_split_sinkhorn,
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
        # W13's window store keeps the post-RoPE rows append-only (row i == token i).
        # Phase 1 attends over the full history and realises the reference sliding
        # window (get_window_topk_idxs L409-426 / _window_kv L700-720) as a causal
        # window mask over absolute positions -- equivalent to the reference ring
        # for every query that can still reach a slot; ``ring()`` is the bounded
        # phase-2 view.
        layer_cache.append_window(kv_new)
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
        o = self._o_lora_down(o)
        return self.wo_b(o.reshape(b, s, -1))

    def _o_lora_down(self, o):
        """Grouped ``wo_a`` down-projection: reshape the [out=n_groups*o_lora_rank,
        in_per_group] weight to [g, o_lora_rank, in] and einsum each group over its
        own heads (reference model.py L785-787).  Dequantized when q8-resident."""
        wo = self.wo_a
        if isinstance(wo, nn.QuantizedLinear):
            w = mx.dequantize(wo.weight, wo.scales, wo.biases, group_size=wo.group_size, bits=wo.bits)
        else:
            w = wo.weight
        w = w.reshape(self.n_groups, self.o_lora_rank, -1)
        return mx.einsum("bsgd,grd->bsgr", o.astype(mx.float32), w.astype(mx.float32))


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
        #: Engram row-id prototype (an :class:`~mtplx.engram_v41.NgramHashState`),
        #: attached by :meth:`Model.attach_engram`; ``None`` when engram is not
        #: wired.  It is a *config-only* template -- each KV cache gets its own
        #: streaming clone (:meth:`NgramHashState.fresh`), so decode history is
        #: per-sequence.  Mirrors the reference ``Transformer.engram_hash``.
        self.engram_hash = None

    def __call__(self, input_ids, cache=None):
        b, s = input_ids.shape
        if cache is None:
            # a bare forward (no persistent cache) still needs a per-layer cache
            # built from the config (so kv_source layers get their CompressorState)
            # and its own engram history when the hooks are attached
            engram_state = self.engram_hash.fresh() if self.engram_hash is not None else None
            cache = _make_cache(self.args, engram_state=engram_state)
        positions = mx.arange(cache.offset, cache.offset + s)

        h = self.embed_tokens(input_ids)  # [b, s, dim]
        h = mx.broadcast_to(h[:, :, None, :], (b, s, self.hc_mult, h.shape[-1]))
        # identity one-hot mix over the hc copies (reference make_identity_pre_mix)
        pre_mix = mx.concatenate(
            [mx.ones((b, s, 1)), mx.zeros((b, s, self.hc_mult - 1))], axis=-1
        ).astype(mx.float32)

        # Engram row-id state (owned by the engram worker) is advanced once per
        # step before any engram layer reads it; text-only has no image mask.
        engram_state = getattr(cache, "engram_state", None)
        if engram_state is not None:
            engram_state.advance(input_ids)

        shared = cache.new_shared_runtime()
        for layer in self.layers:
            if layer.engram_hook is not None and engram_state is not None:
                h = layer.engram_hook(h, input_ids, engram_state)
            h, pre_mix = layer(h, pre_mix, positions, cache.layers[layer.layer_id], shared)
        cache.advance(s)

        # final collapse of the hc copies with the last pre_mix, then RMSNorm
        h = mx.sum(pre_mix[..., None] * h.astype(mx.float32), axis=2).astype(h.dtype)
        return _rmsnorm(h, self.norm_weight, self.args.rms_norm_eps)


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


#: q8 gs64 affine is the resident format for every projection the checkpoint
#: stores quantized (attention, shared expert, compressor, indexer, embed, head).
_RESIDENT_QUANT = {"group_size": 64, "bits": 8, "mode": "affine"}


def _is_resident_quant_module(path: str, module: nn.Module) -> bool:
    """Whether ``nn.quantize`` should quantize this module to the resident q8.

    Quantize the dense projections and the token/output embeddings; keep the MoE
    router gate (bf16 in the checkpoint) and the streamed routed experts
    (``switch_mlp``, served from the bank, never resident) unquantized.  Norms,
    hyper-connection vectors and the attention sink are bare arrays, not modules,
    so ``nn.quantize`` never sees them.
    """
    if not hasattr(module, "to_quantized"):
        return False
    if "switch_mlp" in path or path.endswith("mlp.gate"):
        return False
    in_features = getattr(module, "weight", None)
    if in_features is not None and in_features.shape[-1] % _RESIDENT_QUANT["group_size"] != 0:
        return False  # tiny test configs whose dims are not group-aligned stay dense
    return True


class Model(nn.Module):
    """DeepSeek-V4.1-Flash text AR model.  ``model.model.layers[i].mlp.switch_mlp``
    is the streamed-expert seam; ``head`` is the (untied) output projection.

    ``quantize`` (default True) converts the resident projections to q8 gs64
    affine so the streamed-artifact residents load strictly; tests that compare
    against the dense oracle pass ``quantize=False``.  ``engram_bank_path`` is
    stored for the engram worker's wiring (this module does not build engram).
    """

    def __init__(self, args: ModelArgs, *, engram_bank_path=None, quantize: bool = True):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.engram_bank_path = engram_bank_path
        self.model = DeepseekV41Backbone(args)
        self.head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        if quantize:
            nn.quantize(
                self,
                group_size=_RESIDENT_QUANT["group_size"],
                bits=_RESIDENT_QUANT["bits"],
                mode=_RESIDENT_QUANT["mode"],
                class_predicate=_is_resident_quant_module,
            )

    def __call__(self, input_ids, cache=None):
        h = self.model(input_ids, cache)
        return self.head(h.astype(mx.float32))

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        # W13's factory builds one LayerAttentionCache per layer from the config
        # (window ring, CompressorState frontier on ratio>1 kv_source layers) and a
        # per-sequence SharedAttentionRuntime; each sequence gets its own streaming
        # engram history (the config template is shared).
        engram_state = (
            self.model.engram_hash.fresh() if self.model.engram_hash is not None else None
        )
        return _make_cache(self.args, engram_state=engram_state)

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
        for name, value in weights.items():
            if name.startswith(("vision.", "aligner.", "image_", "mtp.")):
                continue
            if name.endswith(".bias_vl"):
                continue  # VL routing bias, unused on the text path
            out[_sanitize_name(name)] = value
        return out
