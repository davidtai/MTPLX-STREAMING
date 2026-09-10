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
