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
carrying the reference clamped SwiGLU (the ``MoE`` seam this module reuses).  The
resident SwitchGLU quantised matmul **is** ``mx.gather_qmm(mode="mxfp4")``: on this
box (mlx 0.32.2) both ``mx.gather_qmm`` and ``mx.quantized_matmul`` take
``mode: str = 'affine'`` with an optional ``biases`` (mxfp4 gs32 = uint32 codes +
uint8 E8M0 scales, no bias), and ``mlx_lm``'s ``QuantizedSwitchLinear.__call__`` calls
exactly ``mx.gather_qmm(..., mode="mxfp4")`` per projection -- they are the same op,
not alternatives (W104 corrects the earlier "no ``mode=`` argument" note).  The path
is barrier-free: the MTP stages live in ``model.mtp.layers``, never in
``model.model.layers``, so ``bind_streamed_switches`` cannot rebind them to the
streamed ``HotExpertSwitchGLU`` -- no ``mx.eval(indices)`` routing barrier, no
``.tolist()`` route plan, no layer lock, no deferred release, no per-call dequant /
repack.  ``self.mlp(moe_input)`` issues ZERO host syncs per draft cycle (W104 sync
census, ``scripts/deepseek_v41/w104_draft_moe_sync_census.py``); the one sync per cycle
is the decode driver's terminal ``mx.eval``.
"""

from __future__ import annotations

import contextlib
import copy
import os
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mtplx.models import deepseek_v41 as _dv41
from mtplx.models import deepseek_v41_stage_timing as _stime
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
# DSpark draft-block dispatch collapse (kernel-ledger K33, W65)
# ---------------------------------------------------------------------------
# The DSpark-DIRECT draft block (W57 / W65) runs 3 shallow stages -- each a full
# V4.1 decoder block (sliding-window attention + a RESIDENT 128-expert top-3 MoE)
# over the ``block_size`` draft rows -- plus ``forward_embed`` and a markov
# autoregression over ``block_size`` sequential steps.  That is a tiny amount of
# math dispatched as dozens of small graph primitives per stage: pure dispatch
# count, the same regime the backbone's K22 attention tapes and K4 Hyper-Connection
# tapes address ([[b1-decode-dispatch-removal-hides]]).  K33 replays the draft
# block's PURE chains from ``mx.compile`` tapes instead of rebuilding the graph
# from Python each cycle, behind ``MTPLX_DSV41_DRAFT_COMPILE`` (default OFF -- the
# decode win is a GPU-window measurement):
#   * the attention prep (draft QKV + output) REUSES the backbone K22 tapes
#     (``deepseek_v41._attn_qkv_prep`` / ``_attn_out_prep`` -- byte-identical, same
#     projection codec, one tape shared across the stages), plus a small local
#     main-KV tape for the window chain;
#   * the two Hyper-Connection prep chains + the moe-combine post REUSE the backbone
#     K4 tapes (``deepseek_v41._hc_compiled`` -- a DSpark stage IS structurally the
#     backbone layer, so the same pure array functions apply byte-for-byte);
#   * the MoE gate prefix + combine folds fire by arming the backbone ATTN_COMPILE
#     window around the resident switch call (the resident 128-expert top-3 MoE is
#     ALREADY one ``mx.gather_qmm`` over the ``block_size*top_k`` rows via mlx-lm's
#     ``SwitchGLU``, not a Python per-expert loop -- W65 census);
#   * the markov autoregression folds each step's embed+head+add+argmax into one
#     compiled tape (argmax stays lazy -- no per-step host sync) and the markov
#     embed for the confidence head is gathered ONCE over the sampled block instead
#     of per step; the confidence head is a single batched matmul.
# Every tape is fixed-shape + row-cap (``<= _DRAFT_COMPILE_MAX_ROWS``), following
# W33/K4/K22: byte-identical to the eager body in the tiny draft-row regime, and
# the eager body runs unchanged above the cap and with the flag off, so the shipped
# greedy-verify == AR contract is untouched.
_DRAFT_COMPILE_ENV = "MTPLX_DSV41_DRAFT_COMPILE"
_DRAFT_COMPILE_OFF_ALIASES = ("", "0", "false", "no", "off", "auto")
#: Module-global override.  ``None`` -> read the env key at USE (never frozen at
#: import, so a harness that stamps the key after importing this module still arms
#: the tape -- [[env-flags-read-at-use-not-import]]); a bool pins it (the W65
#: census / exactness tests flip this directly, like the K22 census flips
#: ``deepseek_v41._ATTN_COMPILE``).
_DRAFT_COMPILE: Optional[bool] = None
#: Row count (``b * block_size``) at or below which the draft tapes fire; above it
#: the eager body runs (byte-identical), confining compile to the tiny repeating
#: draft shape where per-primitive host encode dominates.
_DRAFT_COMPILE_MAX_ROWS = 32
#: Local tape cache for the DSpark-specific chains (main-KV / markov step /
#: confidence).  The attention-prep + Hyper-Connection chains reuse deepseek_v41's
#: shared ``_ATTN_COMPILED`` / ``_HC_COMPILED`` caches, so a test that clears those
#: resets them; this dict holds only the leaves those caches do not carry.
_DRAFT_COMPILED: dict = {}


def _draft_compile_on() -> bool:
    """The draft-compile switch: the module-global pin if set, else the env key read
    at use (never frozen at import)."""
    if _DRAFT_COMPILE is not None:
        return bool(_DRAFT_COMPILE)
    raw = (os.environ.get(_DRAFT_COMPILE_ENV) or "").strip().lower()
    return raw not in _DRAFT_COMPILE_OFF_ALIASES


# ---------------------------------------------------------------------------
# DSpark draft-head fp32-cast trap removal (W103, mirror of the backbone
# MTPLX_DSV41_HEAD_MODE=bf16 fix / W40-K21)
# ---------------------------------------------------------------------------
# ``DSparkBlock.forward_head`` projects the draft hidden through the SHARED trunk
# output head via ``head(_rmsnorm(x, ...).astype(mx.float32))``.  When the trunk
# head is a dense bf16 ``nn.Linear`` (the native artifact keeps it bf16 -- and the
# backbone's ``MTPLX_DSV41_HEAD_MODE=bf16`` fix leaves the *weight* bf16, it only
# repairs ``Model._apply_head``, NOT this draft call site), the ``.astype(f32)`` on
# the INPUT forces MLX -- which has no mixed-precision matmul -- to promote the
# whole 1.324 GB bf16 head weight to a 2.648 GB f32 temporary EVERY draft cycle
# before the GEMV (6.6 GB traffic/cycle vs 1.3).  It is the exact twin of the trap
# W40/K21 removed on the backbone and of Hy3's lm_head fp32-cast trap
# ([[dsv41-head-fp32-cast-trap]] names this draft site as the follow-up), and it is
# ~83 ms/cycle of ``dspark.head`` in GPU window 39 (which ran HEAD_MODE=bf16 --
# so the backbone head was already fixed and this draft head was still trapped).
#
# ``MTPLX_DSV41_DRAFT_HEAD_BF16`` (default OFF, read at use never frozen at import):
# cast the draft hidden to the head weight's own dtype before the matmul (a bf16
# GEMV over the resident bf16 weight, no promotion), f32 logits after -- exactly the
# backbone ``bf16`` codec.  For a quantised head (packed weight + scales) the source
# dtype stays f32 (``quantized_matmul`` dequantises per group, never promoting), so
# the flag is a no-op there.  The draft head's numerics only set the acceptance
# rate -- the runtime's greedy target verify is authoritative (greedy verify == AR),
# so bf16-rounding the draft hidden is a rounding-class change, never a correctness
# one.  Byte-identical when the head is already f32 (the ``.astype`` is then a
# no-op): the tiny CPU census/tests exercise that identity.
_DRAFT_HEAD_BF16_ENV = "MTPLX_DSV41_DRAFT_HEAD_BF16"
_DRAFT_HEAD_BF16_OFF_ALIASES = ("", "0", "false", "no", "off", "auto")
#: Module-global override (``None`` -> read the env key at use; a bool pins it, the
#: W103 census / exactness tests flip this directly).
_DRAFT_HEAD_BF16: Optional[bool] = None
#: Process-cumulative engagement counters (mirrors the W38/K3 Sinkhorn counters):
#: how many ``forward_head`` calls took the bf16-source (trap-free) branch vs the
#: default f32-cast branch.  Always-on, one int add.  Reset with
#: :func:`_reset_draft_head_calls`.
_DRAFT_HEAD_BF16_CALLS = 0
_DRAFT_HEAD_DEFAULT_CALLS = 0


def _reset_draft_head_calls() -> None:
    """Zero the draft-head engagement counters (per-arm reset for the A/B census)."""
    global _DRAFT_HEAD_BF16_CALLS, _DRAFT_HEAD_DEFAULT_CALLS
    _DRAFT_HEAD_BF16_CALLS = 0
    _DRAFT_HEAD_DEFAULT_CALLS = 0


def _draft_head_calls() -> dict:
    """Snapshot of the draft-head engagement counters (bf16 vs default forward)."""
    return {
        "bf16": int(_DRAFT_HEAD_BF16_CALLS),
        "default": int(_DRAFT_HEAD_DEFAULT_CALLS),
    }


def _draft_head_bf16_on() -> bool:
    """The draft-head fp32-trap switch: the module-global pin if set, else the env
    key read at use (never frozen at import)."""
    if _DRAFT_HEAD_BF16 is not None:
        return bool(_DRAFT_HEAD_BF16)
    raw = (os.environ.get(_DRAFT_HEAD_BF16_ENV) or "").strip().lower()
    return raw not in _DRAFT_HEAD_BF16_OFF_ALIASES


def _draft_head_source_dtype(head: nn.Module) -> "mx.Dtype":
    """The dtype to cast the head input to so the matmul never promotes a dense
    float head weight to a per-call f32 temporary.  A dense float head (weight in
    a float dtype, no ``.scales``) -> its own weight dtype (bf16/f16/f32 GEMV, no
    promotion); a quantised head (packed weight + scales) -> f32 (``quantized_matmul``
    dequantises per group, never promoting).  On the tiny f32 census head this
    returns f32, so the cast is a no-op and the bf16 branch is byte-identical."""
    w = getattr(head, "weight", None)
    if (
        w is not None
        and getattr(head, "scales", None) is None
        and w.dtype in (mx.bfloat16, mx.float16, mx.float32)
    ):
        return w.dtype
    return mx.float32


def _draft_use_compile(rows: int) -> bool:
    """Is this ``rows``-row draft chain in the regime the K33 tapes are kept for?
    Reads the switch + cap at call time so a test/operator can flip either after
    import."""
    return _draft_compile_on() and int(rows) <= _DRAFT_COMPILE_MAX_ROWS


@contextlib.contextmanager
def _moe_compile_window(active: bool):
    """Arm the backbone ATTN_COMPILE window (K22 gate-prefix + combine folds) around
    the resident MoE switch call so the draft stage's MoE gate prefix + combine run
    as the compiled tapes (byte-identical, W41), then restore the prior state.  Only
    the pure gate-prefix / combine folds read this module global; the resident
    ``SwitchGLU`` gather is unaffected.  A scoped toggle, not a permanent flip:
    ``deepseek_v41._attn_use_compile`` still gates on its own row cap + prefill
    guard, so nothing outside this ``with`` body changes behaviour."""
    if not active:
        yield
        return
    saved = _dv41._ATTN_COMPILE
    _dv41._ATTN_COMPILE = True
    try:
        yield
    finally:
        _dv41._ATTN_COMPILE = saved


def _draft_kv_prep(attn: "DSparkAttention"):
    """Compiled KV prep tape ``rope(rmsnorm(wkv(m)))`` -- reused for the window seed
    (main positions) and, structurally, the draft KV; cos/sin are tape inputs.
    Byte-identical to the eager ``_rope_last(_rmsnorm(wkv(m), kv_norm, eps), cos,
    sin)`` (``_apply_lin`` replays ``nn.Linear`` / ``nn.QuantizedLinear`` exactly)."""
    dk = _dv41._lin_desc(attn.wkv)
    key = ("dspark_kv", dk, int(attn.head_dim), float(attn.eps))
    fn = _DRAFT_COMPILED.get(key)
    if fn is None:
        eps = float(attn.eps)

        def impl(m, cos, sin, kv_norm_w, *warrs):
            kv = _rmsnorm(_dv41._apply_lin(dk, warrs, m), kv_norm_w, eps)
            return _rope_last(kv, cos, sin)

        fn = mx.compile(impl)
        _DRAFT_COMPILED[key] = fn
    return fn


def _draft_markov_step(mh: "DSparkMarkovHead"):
    """Compiled markov step: ``(li, markov_embed, next_token)`` from one draft
    token.  Folds the per-step embed + head-matmul + base-logit add + argmax
    (greedy) into one tape; ``argmax`` stays lazy (no per-step host sync).
    Byte-identical to the eager ``base_row + markov_head.head(embed(token))`` then
    ``argmax`` -- the embedding gather and the ``rank -> vocab`` matmul are single
    primitives compile never reassociates."""
    key = ("dspark_markov_step",)
    fn = _DRAFT_COMPILED.get(key)
    if fn is None:

        def impl(token, base_row, embed_w, head_w):
            me = mx.take(embed_w, token, axis=0)          # markov_head.embed(token)
            lb = me @ head_w.T                            # markov_head.head(me), bias=False
            li = base_row + lb
            nxt = mx.argmax(li, axis=-1)                  # _sample(li, 0.0)
            return li, me, nxt

        fn = mx.compile(impl)
        _DRAFT_COMPILED[key] = fn
    return fn


def _draft_confidence(ch: "DSparkConfidenceHead"):
    """Compiled confidence head: ``(concat([hidden, markov_embed]).f32 @ w.T).squeeze``.
    Already batched over the block rows; the tape folds the concat + cast + matmul +
    squeeze.  Byte-identical to :meth:`DSparkConfidenceHead.__call__`."""
    key = ("dspark_confidence",)
    fn = _DRAFT_COMPILED.get(key)
    if fn is None:

        def impl(hidden, markov_embed, w):
            h = mx.concatenate([hidden, markov_embed], axis=-1)
            return (h.astype(mx.float32) @ w.astype(mx.float32).T).squeeze(-1)

        fn = mx.compile(impl)
        _DRAFT_COMPILED[key] = fn
    return fn


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

    def detach_prefill_backings(self):
        self.window = mx.take(self.window, mx.arange(self.window.shape[1]), axis=1)
        return [self.window]

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
        # main KV from this stage's wkv, roped at the main tokens' positions.  In
        # the draft (non-seed) cycle S == 1 (one committed main token), so the K33
        # main-KV tape fires; the seed path (prefill / commit, S up to prompt len)
        # stays eager (above the row cap).
        main_pos = mx.arange(cache.offset, cache.offset + S)
        mcos, msin = _cos_sin(self.inv_freq, main_pos)
        main_use = (not seed_only) and _draft_use_compile(b * S)
        with _stime.stage("dspark.attn.main_kv") as _st:
            if main_use:
                main_kv = _draft_kv_prep(self)(
                    main_x, mcos, msin, self.kv_norm_weight, *_dv41._lin_arrays(self.wkv)
                )
            else:
                main_kv = _rmsnorm(self.wkv(main_x), self.kv_norm_weight, self.eps)
                main_kv = _rope_last(main_kv, mcos, msin)
            _st.add(main_kv)
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
        d_use = _draft_use_compile(b * T)
        # QKV prep: reuse the backbone K22 tape (q = rope(unflatten(wq_b(rmsnorm(
        # wq_a(x))))); kv = rope(rmsnorm(wkv(x)))) -- byte-identical to the eager
        # body; the returned ``qr`` (indexer intermediate) is unused by DSpark.
        with _stime.stage("dspark.attn.qkv_prep") as _st:
            if d_use:
                q, _qr, kv = _dv41._attn_qkv_prep(self)(
                    x, dcos, dsin, self.q_norm_weight, self.kv_norm_weight,
                    *_dv41._lin_arrays(self.wq_a), *_dv41._lin_arrays(self.wq_b),
                    *_dv41._lin_arrays(self.wkv),
                )
            else:
                qr = _rmsnorm(self.wq_a(x), self.q_norm_weight, self.eps)
                q = self.wq_b(qr).reshape(b, T, self.n_heads, self.head_dim)
                q = _rope_last(q, dcos, dsin)
                kv = _rmsnorm(self.wkv(x), self.kv_norm_weight, self.eps)
                kv = _rope_last(kv, dcos, dsin)
            _st.add(q, kv)

        KV = mx.concatenate([win_all, kv], axis=1)  # [b, Wp+T, head_dim]
        # reference get_dspark_topk_idxs (L1020-1029): every draft query attends
        # to all valid window rows + all block draft rows.
        attend = mx.ones((b, T, Wp + T), dtype=mx.bool_)
        with _stime.stage("dspark.attn.sdpa") as _st:
            o = self._sparse_attend(q, KV, attend)  # [b, T, H, head_dim]
            _st.add(o)
        # Output prep: reuse the backbone K22 tape (query-RoPE removal + grouped
        # o-LoRA down einsum + wo_b) -- byte-identical to the eager tail.
        with _stime.stage("dspark.attn.out_prep") as _st:
            if d_use:
                out = _dv41._attn_out_prep(self)(
                    o, dcos, dsin, self._o_lora_dense_weight(), *_dv41._lin_arrays(self.wo_b)
                )
            else:
                o = _rope_last(o, dcos, dsin, inverse=True)
                o = o.reshape(b, T, self.n_groups, -1)
                o = self._o_lora_down(o)
                out = self.wo_b(o.reshape(b, T, -1))
            _st.add(out)
        return out


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
        with _stime.stage("dspark.forward_embed") as _st:
            main_x = self.main_project(main_hidden)
            b = int(input_ids.shape[0])
            first = input_ids.reshape(b, 1)
            noise = mx.full((b, self.block_size - 1), self.noise_token_id, dtype=first.dtype)
            draft_input_ids = mx.concatenate([first, noise], axis=1)  # [b, block_size]
            x = embed(draft_input_ids)  # [b, block_size, dim]
            x = mx.broadcast_to(x[:, :, None, :], (b, self.block_size, self.hc_mult, x.shape[-1]))
            _st.add(x, main_x)
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

        use = _draft_use_compile(int(h.shape[0]) * int(h.shape[1]))
        # Trailing bools key the mix tapes on the active Sinkhorn route (W32/K3) AND
        # the fused-premix-kernel route (W91/K35): the shared backbone K4 tape's
        # ``_hc_mixes_split`` routes its split+Sinkhorn through
        # ``_hc_premix_sinkhorn``, so a runtime flip of either GPU kernel re-traces
        # this draft-block tape too.  On CPU both are always the reference, so they
        # never perturb the numerics.
        hc_consts = (self.hc_mult, self.hc_iters, self.norm_eps, self.hc_eps,
                     _dv41._sinkhorn_use_kernel(), _dv41._hc_premix_use_kernel())

        residual = h
        # Attention Hyper-Connection prep (mix + Sinkhorn + pre_mix collapse + attn
        # RMSNorm) -> the attention input.  K33 reuses the backbone K4 tape.
        with _stime.stage("dspark.hc.attn_prep") as _st:
            if use:
                x, attn_pre, attn_post, attn_comb = _dv41._hc_compiled(
                    "attn_prep", *hc_consts
                )(
                    h, pre_mix, self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale,
                    self.attn_norm_weight,
                )
            else:
                attn_pre, attn_post, attn_comb = self._mixes(
                    h, self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale
                )
                x = self._hc_pre(h, pre_mix)
                x = _rmsnorm(x, self.attn_norm_weight, self.norm_eps)
            _st.add(x, attn_pre, attn_post, attn_comb)
        x = self.attn(x, main_x, cache, seed_only=False)  # writes nothing (draft)
        # Post-attention HC + ffn HC prep -> the routed-expert input (+ the residual
        # the moe combine folds back).  K33 reuses the backbone K4 tape.
        with _stime.stage("dspark.hc.ffn_prep") as _st:
            if use:
                moe_input, moe_residual, ffn_post, ffn_comb, ffn_pre = _dv41._hc_compiled(
                    "ffn_prep", *hc_consts
                )(
                    x, residual, attn_pre, attn_post, attn_comb,
                    self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale,
                    self.ffn_norm_weight,
                )
            else:
                moe_residual = _hc_post_impl(x, residual, attn_post, attn_comb)
                ffn_pre, ffn_post, ffn_comb = self._mixes(
                    moe_residual, self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale
                )
                xin = self._hc_pre(moe_residual, attn_pre)
                moe_input = _rmsnorm(xin, self.ffn_norm_weight, self.norm_eps)
            _st.add(moe_input, moe_residual, ffn_post, ffn_comb, ffn_pre)
        # Resident 128-expert top-3 MoE (one gather_qmm over block_size*top_k rows
        # via SwitchGLU); arm the K22 gate-prefix + combine folds around it.
        with _moe_compile_window(use):
            x = self.mlp(moe_input)
        with _stime.stage("dspark.hc.moe_combine") as _st:
            if use:
                h = _dv41._hc_compiled("moe_combine")(x, moe_residual, ffn_post, ffn_comb)
            else:
                h = _hc_post_impl(x, moe_residual, ffn_post, ffn_comb)
            _st.add(h)
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
        b = int(input_ids.shape[0])
        use = _draft_use_compile(b * self.block_size)
        greedy = float(self.temperature) <= 0.0
        with _stime.stage("dspark.head") as _st:
            global _DRAFT_HEAD_BF16_CALLS, _DRAFT_HEAD_DEFAULT_CALLS
            x = self._hc_pre(x, pre_mix)  # [b, block_size, dim]
            hidden_n = _rmsnorm(x, self.norm_weight, self.norm_eps)
            if _draft_head_bf16_on():
                # W103: cast the hidden to the head weight dtype (a bf16 GEMV over
                # the resident bf16 weight, f32 logits after) so a dense bf16 head
                # is not promoted to a per-cycle f32 temporary -- the fp32-cast trap.
                _DRAFT_HEAD_BF16_CALLS += 1
                base_logits = head(
                    hidden_n.astype(_draft_head_source_dtype(head))
                ).astype(mx.float32)
            else:
                # Default (byte-identical to the historical draft head): cast the
                # hidden to f32, which promotes a bf16 head weight to a f32 temporary.
                _DRAFT_HEAD_DEFAULT_CALLS += 1
                base_logits = head(hidden_n.astype(mx.float32))
            _st.add(x, base_logits)

        out_cols: List[mx.array] = [input_ids.reshape(b)]
        logit_cols: List[mx.array] = []
        with _stime.stage("dspark.markov") as _st:
            if use and greedy:
                # Fold each step's embed + head-matmul + add + argmax into one tape
                # (argmax stays lazy -- no per-step host sync); the markov embed for
                # the confidence head is gathered ONCE over the sampled block below.
                step = _draft_markov_step(self.markov_head)
                embed_w = self.markov_head.embed.weight
                head_w = self.markov_head.head.weight
                for i in range(self.block_size):
                    li, _me, nxt = step(out_cols[i], base_logits[:, i, :], embed_w, head_w)
                    logit_cols.append(li)
                    out_cols.append(nxt.reshape(b))
            else:
                markov_embeds: List[mx.array] = []
                for i in range(self.block_size):
                    logits_bias, markov_embed = self.markov_head(out_cols[i])
                    li = base_logits[:, i, :] + logits_bias
                    logit_cols.append(li)
                    markov_embeds.append(markov_embed)
                    out_cols.append(_sample(li, self.temperature).reshape(b))
            output_ids = mx.stack(out_cols, axis=1)  # [b, block_size+1]
            logits = mx.stack(logit_cols, axis=1)  # [b, block_size, vocab]
            _st.add(output_ids, logits)

        with _stime.stage("dspark.confidence") as _st:
            if use and greedy:
                # Batch the markov embed over the sampled draft block (one gather),
                # not per step -- byte-identical to stacking the per-step embeds.
                markov_embed = self.markov_head.embed(output_ids[:, : self.block_size])
                confidence = _draft_confidence(self.confidence_head)(
                    x, markov_embed, self.confidence_head.proj.weight
                )
            else:
                markov_embed = mx.stack(markov_embeds, axis=1)  # [b, block_size, rank]
                confidence = self.confidence_head(x, markov_embed)  # [b, block_size]
            _st.add(confidence)
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
