"""DeepSeek-V4.1-Flash DSpark MTP draft head (worker W23).

A faithful MLX transliteration of the DeepSeek reference ``inference/model.py``
DSpark classes (L1020-1156) plus the ``Transformer.forward_spec`` orchestration
(L1274-1282).  The class names and method structure mirror the reference; each
method carries its reference line reference.

DSpark is a *3-stage* speculative draft head stored under the checkpoint's
``mtp.{0,1,2}.*`` namespace.  It differs from the DeepSeek-V4 single MTP block
(:class:`mtplx.models.deepseek_v4.DeepseekV4MTP`) in three ways this module
transliterates verbatim:

* it consumes the backbone's *target-layer* hidden states
  (``dspark_target_layer_ids``, the attention **input** of those layers, mean
  over the Hyper-Connection copies), not the final pre-head state, projected by a
  stage-0 ``main_proj`` + ``main_norm`` (reference ``DSparkBlock.forward_embed``
  L1128-1135);
* each of the 3 stages is a full V4.1 decoder block -- ``DSparkAttention``
  (a pure sliding-window MLA attention seeded from the *main* hiddens' KV,
  reference L1032-1074) wrapped in Hyper-Connections around its own 128-expert
  MoE (top-3 routed, mxfp4-resident) -- so ``main_x`` threads through all three
  (reference ``Transformer.forward_spec`` L1278-1279);
* the last stage carries a ``markov_head`` (a low-rank per-token logit bias) and a
  ``confidence_head``, and produces ``dspark_block_size`` draft tokens in one call
  by autoregressing the cheap markov correction over the block (reference
  ``DSparkBlock.forward_head`` L1137-1156).

Losslessness bar (memory: deepseek-v4-mtplx-port / spec-decode-cycle-anatomy):
the draft head's numerics only set the *acceptance rate*, never correctness.
The runtime's greedy verify (target argmax over the K+1-row forward) guarantees
MTP output == AR output regardless of draft quality, so this module is NOT
held to bit-exactness -- greedy verify == AR argmax is the bar.

Reuse: the whole V4.1 backbone (``Attention`` projections + sparse-attend + the
grouped o-LoRA, ``DecoderLayer``'s Hyper-Connection ``_mixes``/``_hc_pre`` and
the reused ``_hc_post_impl``, and the ``MoE`` switch seam) is imported from
:mod:`mtplx.models.deepseek_v41`; only the DSpark-specific leaves are new here.

Expert execution: the 3 MTP stages' 128 routed experts are RESIDENT mxfp4 gs32
tensors (W18_REPORT), executed through mlx-lm's quantised :class:`SwitchGLU`
carrying the reference clamped SwiGLU (the ``MoE`` seam this module reuses) --
``mx.gather_qmm`` in mlx 0.32.2 has no ``mode=`` argument, so the resident
mxfp4 path is the SwitchGLU quantised matmul, not a bespoke gather_qmm(mode=).
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mtplx.models.deepseek_v41 import (
    Attention,
    DecoderLayer,
    MODE_SWA_ONLY,
    ModelArgs,
    MoE,
    _cos_sin,
    _hc_post_impl,
    _rmsnorm,
    _rope_last,
    _swa_inv_freq,
)


# ---------------------------------------------------------------------------
# DSpark config (the ``dspark_*`` / ``n_mtp_layers`` fields; read off ModelArgs
# with the released-artifact defaults so a bare ModelArgs still constructs a head)
# ---------------------------------------------------------------------------
def dspark_block_size(args: ModelArgs) -> int:
    return int(getattr(args, "dspark_block_size", 0) or 0)


def n_mtp_layers(args: ModelArgs) -> int:
    #: released config: ``n_mtp_layers`` (== ``num_nextn_predict_layers`` == 3).
    return int(
        getattr(args, "n_mtp_layers", None)
        or getattr(args, "num_nextn_predict_layers", 0)
        or 0
    )


def dspark_target_layer_ids(args: ModelArgs) -> Tuple[int, ...]:
    return tuple(int(i) for i in getattr(args, "dspark_target_layer_ids", ()) or ())


def _dspark_markov_rank(args: ModelArgs) -> int:
    return int(getattr(args, "dspark_markov_rank", 256) or 256)


def _dspark_noise_token_id(args: ModelArgs) -> int:
    return int(getattr(args, "dspark_noise_token_id", 0) or 0)


def _mtp_moe_args(args: ModelArgs) -> ModelArgs:
    """A shallow ModelArgs clone with the MTP MoE shapes (128 routed experts,
    top-3) so the reused :class:`MoE` builds the DSpark stage's own switch seam.

    The MTP stage's ``moe_intermediate_size``/``hidden_size`` equal the
    backbone's (2304 / 5120, W18_REPORT), only the routed-expert count and the
    activated-expert count differ (128 / top-3 vs the backbone's 384 / top-6).
    """
    a = copy.copy(args)
    a.n_routed_experts = int(
        getattr(args, "dspark_n_routed_experts", 128) or 128
    )
    a.num_experts_per_tok = int(
        getattr(args, "dspark_num_experts_per_tok", None)
        or getattr(args, "dspark_n_activated_experts", 3)
        or 3
    )
    return a


def _sample(logits: mx.array, temperature: float) -> mx.array:
    """Reference ``sample`` (model.py L1285-1292): argmax at ``temperature == 0``
    (the greedy draft), else Gumbel-max.  The draft is greedy on the spec path;
    the Gumbel branch is kept for parity with the reference sampler."""
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)
    logits = logits / max(temperature, 1e-5)
    probs = mx.softmax(logits.astype(mx.float32), axis=-1)
    g = -mx.log(-mx.log(mx.random.uniform(shape=probs.shape) + 1e-20) + 1e-20)
    return mx.argmax(mx.log(probs + 1e-20) + g, axis=-1)


# ---------------------------------------------------------------------------
# DSpark leaves
# ---------------------------------------------------------------------------
class DSparkMarkovHead(nn.Module):
    """Reference ``DSparkMarkovHead`` (model.py L1077-1086): a low-rank
    token-conditioned logit bias.  ``embed`` maps a token id to a
    ``dspark_markov_rank`` vector; ``head`` projects that back to a full-vocab
    logit bias.  Returns ``(logits_bias, embed)`` -- the embed feeds the
    confidence head."""

    def __init__(self, vocab_size: int, markov_rank: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, markov_rank)
        # reference ParallelHead == Linear(rank -> vocab), no bias (L1081).
        self.head = nn.Linear(markov_rank, vocab_size, bias=False)

    def __call__(self, token_ids: mx.array) -> Tuple[mx.array, mx.array]:
        embed = self.embed(token_ids)
        logits = self.head(embed)
        return logits, embed


class DSparkConfidenceHead(nn.Module):
    """Reference ``DSparkConfidenceHead`` (model.py L1089-1097): a scalar
    confidence per draft position from ``[hidden ; markov_embed]``, computed in
    fp32 (the reference stores ``proj`` bf16 but casts to fp32 for the score)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False)

    def __call__(self, hidden: mx.array, markov_embed: mx.array) -> mx.array:
        h = mx.concatenate([hidden, markov_embed], axis=-1)
        w = self.proj.weight.astype(mx.float32)
        return (h.astype(mx.float32) @ w.T).squeeze(-1)


class DSparkStageCache:
    """Per-stage sliding-window KV of the *main* hiddens (reference
    ``DSparkAttention.window_kv_cache``, model.py L474/L1046-1065).

    No MTP-stage cache is contracted in ``docs/deepseek-v41/PORT_CONTRACT.md``
    (only the backbone attention cache, W13), so W23 owns this one.  It stores
    the last ``window_size`` post-RoPE main-KV rows and the running ``offset``
    (absolute main-token count), and answers the ``mlx_lm`` rollback seam
    (``trim`` / ``mark`` / ``rollback`` / ``offset`` / ``is_trimmable``) the
    speculative verify rewinds through, exactly as the W13 seam does for the
    backbone (PORT_CONTRACT "Rollback seam (serve / gate / speculative
    verify)").
    """

    def __init__(self, window_size: int, head_dim: int):
        self.window_size = int(window_size)
        self.head_dim = int(head_dim)
        self.window: Optional[mx.array] = None  # [b, <=window_size, head_dim]
        self.offset = 0

    def append_main(self, main_kv: mx.array) -> None:
        """Append this call's post-RoPE main-KV rows; keep only the last
        ``window_size`` (the reference ring) and advance ``offset``."""
        if self.window is None:
            self.window = main_kv
        else:
            self.window = mx.concatenate([self.window, main_kv], axis=1)
        if self.window.shape[1] > self.window_size:
            self.window = self.window[:, -self.window_size :, :]
        self.offset += int(main_kv.shape[1])

    def trim(self, n: int) -> int:
        """Restore to ``n`` main tokens earlier (mlx_lm ``trim`` convention:
        returns the count trimmed).  Rewinds ``offset`` and drops the last ``n``
        window rows."""
        n = int(min(max(n, 0), self.offset))
        if n and self.window is not None:
            keep = self.window.shape[1] - n
            self.window = self.window[:, : max(keep, 0), :] if keep > 0 else None
        self.offset -= n
        return n

    def mark(self):
        return (self.offset, self.window)

    def rollback(self, mark) -> None:
        self.offset, self.window = mark

    def is_trimmable(self) -> bool:
        return True


class DSparkAttention(Attention):
    """Reference ``DSparkAttention`` (model.py L1032-1074): a pure
    sliding-window MLA attention whose *keys/values come from the backbone's
    main hiddens* (seeded across decode) while *queries come from the draft
    hiddens*.  ``compress_ratio == 0`` -- no CSA2 compressor/indexer -- so it
    reuses the base :class:`Attention` projections, ``_sparse_attend`` (the
    per-head sink softmax) and ``_o_lora_down`` (the grouped o-LoRA), and
    overrides only the forward.

    Not inherited from :meth:`Attention.__init__`, because that indexes
    ``compress_ratios[layer_id]`` at a backbone layer id; the MTP stages have no
    backbone layer, so the SWA-only projection set is built here directly (the
    same tensors :meth:`Attention.__init__` builds on a ``compress_ratio == 0``
    layer, minus the absent compressor/indexer).
    """

    def __init__(self, args: ModelArgs):
        nn.Module.__init__(self)
        self.dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.n_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.window_size = args.window_size
        self.eps = args.rms_norm_eps
        self.softmax_scale = args.head_dim ** -0.5
        self.compress_ratio = 0
        #: DSpark attention is always a pure sliding window (``compress_ratio ==
        #: 0``); the base :class:`Attention` reads ``self.mode`` for its
        #: stage-timing labels (``_sparse_attend_oneshot`` L715), and the MTP
        #: stages have no ``layer_modes`` entry, so it is set here directly (a
        #: label only -- no numeric effect).  Without it a draft-block forward
        #: (T > 1 rows -> the score path) raised ``'DSparkAttention' object has
        #: no attribute 'mode'`` and broke the whole draft (W57).
        self.mode = MODE_SWA_ONLY
        self.compressor = None
        self.indexer = None
        self.capture_selection = False
        self.last_selection = None
        self.last_candidates = None
        import numpy as _np

        self.attn_sink = mx.zeros((self.n_heads,))
        self.wq_a = nn.Linear(self.dim, args.q_lora_rank, bias=False)
        self.q_norm_weight = mx.ones((args.q_lora_rank,))
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm_weight = mx.ones((self.head_dim,))
        in_per_group = self.n_heads * self.head_dim // self.n_groups
        self.wo_a = nn.Linear(in_per_group, self.n_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=False)
        # DSpark attention is a base-theta sliding window (reference L1386-style
        # SWA, no YaRN); the RoPE table is a derived constant, kept off the
        # parameter tree as numpy.
        self.inv_freq = _np.asarray(_swa_inv_freq(args), dtype=_np.float32)

    def __call__(
        self, x: mx.array, main_x: mx.array, cache: DSparkStageCache, *, seed_only: bool = False
    ) -> mx.array:
        """Reference ``DSparkAttention.forward`` (model.py L1033-1074).

        ``main_x``: ``[b, S, dim]`` the projected main hiddens (S == prompt len at
        seeding, 1 per decode step).  ``x``: ``[b, block_size, dim]`` the draft
        hiddens.  When ``seed_only`` (reference ``start_pos == 0`` branch,
        L1044-1052), only the window KV is seeded from ``main_x`` and ``x`` is
        returned unchanged.  Otherwise the draft queries attend over the window
        (main KV) concatenated with the block's own draft KV (reference
        L1054-1073), through the per-head sink softmax + grouped o-LoRA.
        """
        b, S, _ = main_x.shape
        # main KV from this stage's wkv, roped at the main tokens' positions
        main_pos = mx.arange(cache.offset, cache.offset + S)
        mcos, msin = _cos_sin(self.inv_freq, main_pos)
        main_kv = _rmsnorm(self.wkv(main_x), self.kv_norm_weight, self.eps)
        main_kv = _rope_last(main_kv, mcos, msin)
        if seed_only:
            # commit the main KV to the window (reference L1044-1065 ring write);
            # the update-cache path drives this for the tokens the target committed.
            cache.append_main(main_kv)
            return x

        # Draft: attend over the committed window PLUS this cycle's (uncommitted)
        # main KV, capped to window_size, WITHOUT appending it -- the draft must
        # not mutate the window (mtp_update_cache commits accepted tokens later;
        # the draft cache conditions acceptance only).
        window = cache.window  # [b, <=window_size, head_dim] committed main rows
        win_all = main_kv if window is None else mx.concatenate([window, main_kv], axis=1)
        if win_all.shape[1] > self.window_size:
            win_all = win_all[:, -self.window_size :, :]
        Wp = int(win_all.shape[1])

        b, T, _ = x.shape  # T == block_size
        base = cache.offset + S  # draft rows follow the current main token(s)
        dpos = mx.arange(base, base + T)
        dcos, dsin = _cos_sin(self.inv_freq, dpos)
        qr = _rmsnorm(self.wq_a(x), self.q_norm_weight, self.eps)
        q = self.wq_b(qr).reshape(b, T, self.n_heads, self.head_dim)
        q = _rope_last(q, dcos, dsin)
        kv = _rmsnorm(self.wkv(x), self.kv_norm_weight, self.eps)
        kv = _rope_last(kv, dcos, dsin)

        KV = mx.concatenate([win_all, kv], axis=1)  # [b, Wp+T, head_dim]
        # reference get_dspark_topk_idxs (L1020-1029): every draft query attends
        # to all valid window rows + all block draft rows.
        attend = mx.ones((b, T, Wp + T), dtype=mx.bool_)
        o = self._sparse_attend(q, KV, attend)  # [b, T, H, head_dim]
        o = _rope_last(o, dcos, dsin, inverse=True)
        o = o.reshape(b, T, self.n_groups, -1)
        o = self._o_lora_down(o)
        return self.wo_b(o.reshape(b, T, -1))


class DSparkBlock(DecoderLayer):
    """Reference ``DSparkBlock(Block)`` (model.py L1100-1156): one DSpark stage.

    Reuses :class:`DecoderLayer`'s Hyper-Connection machinery (``_mixes`` /
    ``_hc_pre`` and the reused ``_hc_post_impl``, the ``attn_norm``/``ffn_norm``
    weights) but swaps the backbone attention for :class:`DSparkAttention` and
    the backbone MoE for the MTP stage's 128-expert MoE.  Stage 0 owns
    ``main_proj`` + ``main_norm``; the last stage owns ``norm`` + ``markov_head``
    + ``confidence_head`` and the shared token embedding + output head (wired in
    by :class:`DSparkHead`).
    """

    def __init__(self, args: ModelArgs, stage_id: int, n_stages: int):
        nn.Module.__init__(self)
        self.stage_id = stage_id
        self.n_stages = n_stages
        self.norm_eps = args.rms_norm_eps
        self.hc_eps = args.hc_eps
        self.hc_mult = args.hc_mult
        self.hc_iters = args.hc_sinkhorn_iters
        self.block_size = dspark_block_size(args)
        self.noise_token_id = _dspark_noise_token_id(args)
        self.temperature = 0.0  # greedy draft on the spec path
        self.attn = DSparkAttention(args)
        self.mlp = MoE(args.num_hidden_layers + stage_id, _mtp_moe_args(args))
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
        self.engram_hook = None

        target_ids = dspark_target_layer_ids(args)
        if stage_id == 0:  # reference L1111-1114
            n_targets = max(1, len(target_ids))
            self.main_proj = nn.Linear(args.hidden_size * n_targets, args.hidden_size, bias=False)
            self.main_norm_weight = mx.ones((args.hidden_size,))
        if stage_id == n_stages - 1:  # reference L1115-1118
            self.norm_weight = mx.ones((args.hidden_size,))
            self.markov_head = DSparkMarkovHead(args.vocab_size, _dspark_markov_rank(args))
            self.confidence_head = DSparkConfidenceHead(args.hidden_size + _dspark_markov_rank(args))
        # The trunk token embedding + output head are NOT stored on the stage
        # (that would alias the backbone modules into the MTP parameter tree and
        # double-quantise them); they are passed in at call time, exactly as
        # DeepseekV4MTP takes ``embed_tokens``/``lm_head`` as forward arguments.

    def main_project(self, main_hidden: mx.array) -> mx.array:
        """Reference ``main_x = main_norm(main_proj(main_hidden))`` (model.py
        L1130), stage 0 only.  Split out so the seeding path can build ``main_x``
        without also embedding a draft block."""
        return _rmsnorm(self.main_proj(main_hidden), self.main_norm_weight, self.norm_eps)

    def forward_embed(
        self, main_hidden: mx.array, input_ids: mx.array, embed: nn.Module
    ) -> Tuple[mx.array, mx.array]:
        """Reference ``DSparkBlock.forward_embed`` (model.py L1128-1135), stage 0
        only.  Projects the concatenated target-layer hiddens to ``main_x`` and
        builds the draft input ``[real_token, noise, ..., noise]`` of length
        ``block_size``, embedded (through the shared trunk embedding) and expanded
        to ``hc_mult`` copies."""
        main_x = self.main_project(main_hidden)
        b = int(input_ids.shape[0])
        first = input_ids.reshape(b, 1)
        noise = mx.full((b, self.block_size - 1), self.noise_token_id, dtype=first.dtype)
        draft_input_ids = mx.concatenate([first, noise], axis=1)  # [b, block_size]
        x = embed(draft_input_ids)  # [b, block_size, dim]
        x = mx.broadcast_to(x[:, :, None, :], (b, self.block_size, self.hc_mult, x.shape[-1]))
        return x, main_x

    def __call__(
        self, h: mx.array, pre_mix: mx.array, main_x: mx.array, cache: DSparkStageCache,
        *, seed_only: bool = False,
    ) -> Tuple[mx.array, mx.array]:
        """Reference ``DSparkBlock.forward`` -> ``Block.forward`` (model.py
        L1122-1126 / L978-994): Hyper-Connection pre/post around the DSpark
        attention (fed ``main_x``) and the MTP MoE.  ``seed_only`` runs only the
        attention's window-seeding branch (reference L1123-1125)."""
        if seed_only:
            # reference L1123-1125: prefill just seeds the window KV cache.
            x = self._hc_pre(h, pre_mix)
            x = _rmsnorm(x, self.attn_norm_weight, self.norm_eps)
            self.attn(x, main_x, cache, seed_only=True)
            return h, pre_mix

        residual = h
        attn_pre, attn_post, attn_comb = self._mixes(
            h, self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale
        )
        x = self._hc_pre(h, pre_mix)
        x = _rmsnorm(x, self.attn_norm_weight, self.norm_eps)
        x = self.attn(x, main_x, cache, seed_only=False)
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

    def forward_head(
        self, x: mx.array, pre_mix: mx.array, input_ids: mx.array, head: nn.Module
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Reference ``DSparkBlock.forward_head`` (model.py L1137-1156), last
        stage only.  Collapses the hc copies, projects to logits through the
        shared output head, then autoregresses the cheap markov correction over
        the block: draft token ``i+1`` is sampled from ``head_logits[:, i]`` plus
        the markov bias of draft token ``i``.  Returns ``(output_ids
        [b, block_size+1], logits [b, block_size, vocab], confidence
        [b, block_size])``."""
        x = self._hc_pre(x, pre_mix)  # [b, block_size, dim]
        base_logits = head(_rmsnorm(x, self.norm_weight, self.norm_eps).astype(mx.float32))

        b = int(input_ids.shape[0])
        out_cols: List[mx.array] = [input_ids.reshape(b)]
        logit_cols: List[mx.array] = []
        markov_embeds: List[mx.array] = []
        for i in range(self.block_size):
            logits_bias, markov_embed = self.markov_head(out_cols[i])
            li = base_logits[:, i, :] + logits_bias
            logit_cols.append(li)
            markov_embeds.append(markov_embed)
            out_cols.append(_sample(li, self.temperature).reshape(b))
        output_ids = mx.stack(out_cols, axis=1)  # [b, block_size+1]
        logits = mx.stack(logit_cols, axis=1)  # [b, block_size, vocab]
        markov_embed = mx.stack(markov_embeds, axis=1)  # [b, block_size, rank]
        confidence = self.confidence_head(x, markov_embed)  # [b, block_size]
        return output_ids, logits, confidence


class DSparkHead(nn.Module):
    """The 3-stage DSpark draft head (reference ``Transformer.mtp`` +
    ``forward_spec``, model.py L1207-1213 / L1274-1282).

    ``self.layers`` are the ``DSparkBlock`` stages (parameter paths
    ``layers.{0,1,2}.*`` so the loader maps ``mtp.{0,1,2}.*`` onto them).  The
    trunk token embedding and output head are passed into :meth:`seed_main` /
    :meth:`draft_block` at call time (not stored on the stages), so the draft's
    logits land in the trunk vocab without aliasing the backbone modules into the
    MTP parameter tree -- exactly as ``DeepseekV4MTP`` takes them as arguments.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.block_size = dspark_block_size(args)
        self.target_layer_ids = dspark_target_layer_ids(args)
        stages = n_mtp_layers(args)
        self.layers = [DSparkBlock(args, i, stages) for i in range(stages)]
        self.hc_mult = args.hc_mult

    def _identity_pre_mix(self, b: int, s: int) -> mx.array:
        return mx.concatenate(
            [mx.ones((b, s, 1)), mx.zeros((b, s, self.hc_mult - 1))], axis=-1
        ).astype(mx.float32)

    def seed_main(self, main_hidden: mx.array, caches: List[DSparkStageCache]) -> None:
        """Reference ``forward_spec`` at ``start_pos == 0`` (model.py L1276-1281):
        append each stage's window KV from committed main hiddens (its own
        ``wkv`` over the shared ``main_x``).  No draft is produced; the draft
        block is embedded only when :meth:`draft_block` runs, so this needs no
        token embedding."""
        main_x = self.layers[0].main_project(main_hidden)
        for stage, cache in zip(self.layers, caches):
            stage.attn(main_x, main_x, cache, seed_only=True)

    def draft_block(
        self, main_hidden: mx.array, input_ids: mx.array, caches: List[DSparkStageCache],
        embed: nn.Module, head: nn.Module,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Reference ``forward_spec`` at ``start_pos > 0`` (model.py L1276-1282):
        embed -> 3 stages (threading ``main_x``) -> the last stage's head.
        Returns ``(output_ids [b, block_size+1], logits [b, block_size, vocab],
        confidence [b, block_size])``."""
        x, main_x = self.layers[0].forward_embed(main_hidden, input_ids, embed)
        b, s = x.shape[0], x.shape[1]
        pre_mix = self._identity_pre_mix(b, s)
        for stage, cache in zip(self.layers, caches):
            x, pre_mix = stage(x, pre_mix, main_x, cache, seed_only=False)
        return self.layers[-1].forward_head(x, pre_mix, input_ids, head)
