"""DSpark-DIRECT decode lane for DeepSeek-V4.1-Flash (worker W57).

A self-contained speculative-decode loop that drives W23's DSpark 3-stage draft
head (:mod:`mtplx.models.deepseek_v41_dspark`) against the V4.1 target forward,
*bypassing MTPLX's generic native-MTP machinery* (``generate_mtpk``,
``draft_lm_head`` install, ``mtp_patch``, ``runtime.py``'s streaming-block MTP
injection and ``model_scheduler``'s MTP cycle).  Window 21 measured that generic
pathway serving V4.1 MTP at 2.45--2.94 tok/s vs served AR 2.2--5.0 -- it costs
more than it accepts.  Decode on this lane is dispatch-bound (~160 ms for a
1-row forward; a K+1-row verify forward costs little more -- window 17), so a lean
DSpark loop turns accepted tokens into near-free throughput.

Algorithm (what this implements; the DeepSeek reference ``inference/model.py``
carries the DSpark *forward* but explicitly leaves the decode loop out of scope,
see its L129-131, so acceptance/rollback are standard speculative decoding):

* **Draft length.**  DSpark drafts a whole block of ``dspark_block_size`` tokens
  in one ``forward_spec`` (reference ``DSparkBlock.forward_embed`` builds
  ``[real_token, noise, ..., noise]`` and ``forward_head`` autoregresses the
  markov correction over the block).  We propose ``K = min(speculative_depth,
  block_size)`` of them per cycle.  The confidence head (a per-draft-token scalar)
  drives an optional early stop: we keep only the leading run whose confidence
  clears ``confidence_threshold`` (a pure latency lever -- it changes the verify
  width, never the output, since verify is authoritative).

* **Verify.**  One target forward over the ``1 + K`` rows ``[primary, d1, ..., dK]``
  yields ``K+1`` logit rows; row ``i`` is the target's distribution for the token
  *after* block position ``i`` (row 0 = the token after ``primary``).

* **Acceptance.**  Greedy: accept the longest draft prefix whose tokens equal the
  target argmax at the preceding row; the correction (or, on a full accept, the
  bonus) is the target argmax at the first unaccepted row -- so greedy decode is
  byte-for-byte identical to AR regardless of draft quality.  Sampled (temperature
  > 0): standard speculative sampling (Leviathan/Chen) with the target
  distribution.  The DSpark draft is greedy (``DSparkBlock.temperature == 0``), so
  its proposal is the deterministic point mass ``q = delta_d``; accepting ``d``
  with probability ``min(1, p(d)/q(d)) = p(d)`` and, on rejection, sampling the
  correction from ``norm(max(0, p - q))`` gives output marginal exactly ``p``
  (the AR sampler distribution).  A ``K=0`` configuration is therefore exactly AR
  sampling under the same seed.

* **Rollback.**  The verify forward appends ``K+1`` rows to the target cache; we
  keep the committed prefix ``[primary, d1..da]`` (``a+1`` rows) and trim the
  ``K-a`` rejected tail with ``mtplx.cache_state.trim_verified_window_to_prefix``
  (the V4.1 cache is all-trimmable -- ``LayerAttentionCache.is_trimmable`` --
  so window/compressed/index/compressor-frontier/engram rewind together, no
  re-forward).  The DSpark stage windows (W23's ``DSparkStageCache``, the W26 leaf
  engram-free sliding-window KV of the *main* hiddens) are seeded with the
  committed tokens' main hiddens via ``DSparkHead.seed_main``.

Reuse (coordinator directive, W57): the acceptance / verify / cache-rollback
STRUCTURE is the proven DeepSeek-V4 native-MTP K3 loop (``generate_mtp1`` /
``generate_mtpk`` in :mod:`mtplx.generation`, PR #216, 25.86 tok/s @ K3) and its
cache primitives (``snapshot_untrimmable_cache`` + ``trim_verified_window_to_prefix``
/ ``rollback_after_verify``); the speculative-sampling accept math is the shared
``mtplx.sampling`` helpers those loops use.  What this lane drops is the generic
machinery around that structure.  What differs in V4.1's DSpark vs V4's single
MTP block: 3 stages threading ``main_x``, the noise-token block embed, and the
markov/confidence heads -- all owned by W23's drafter, which this loop calls.
"""

from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

import mlx.core as mx
import numpy as np

from mtplx.cache_state import (
    snapshot_untrimmable_cache,
    trim_verified_window_to_prefix,
)

# Optional W37 stage-timing probe: stage()/frame() are no-ops unless a session is
# armed (ab_decode --decode-mode dspark --stage-timing), so wrapping the loop is
# free otherwise.
try:  # pragma: no cover - import guard
    from mtplx.models import deepseek_v41_stage_timing as _stime
except Exception:  # pragma: no cover
    _stime = None


def _stage(name: str):
    return _stime.stage(name) if _stime is not None else contextlib.nullcontext()


def _arm_stage_recording() -> None:
    if _stime is not None:
        try:
            _stime.arm_recording()
        except Exception:
            pass


def _frame():
    return _stime.frame() if _stime is not None else contextlib.nullcontext()


#: K29 (fused decode/verify attention, b*s<=8) and K30 (selected-key gather) env
#: flags. The DSpark verify is a small-M (K+1) forward -- exactly K29's decode/verify
#: case and K30's selected-key case -- but both default OFF, so the streamed switch's
#: rows>1 prefill attention path ran the verify (window-25: attn ~830 ms/verify vs
#: ~50 ms at M=1). The lane arms both by default so the verify uses the decode
#: attention branch; they are greedy-identical to the eager path (float
#: reassociation, never bit-identical), so both the served verify and the offline
#: AR-reference must share the setting -- the lane arms them for the WHOLE run.
_K29_ENV = "MTPLX_DSV41_DECODE_ATTN_KERNEL"
_K30_ENV = "MTPLX_DSV41_SELECTED_KEYS"
_DSPARK_DECODE_KERNEL_ENVS = (_K29_ENV, _K30_ENV)


def _dspark_decode_kernels_disabled() -> bool:
    return os.environ.get("MTPLX_DSV41_DSPARK_DECODE_KERNELS", "").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }


def _dspark_verify_k29_enabled() -> bool:
    """Whether the lane arms K29 (the fused decode-attention kernel). Default ON;
    ``MTPLX_DSV41_DSPARK_VERIFY_K29=0`` keeps K30 (selected keys) but leaves K29
    unarmed, so window 29 can separate the kernel's cost (K29 is itself -38% vs
    eager at M=1, so it may be hurting the wider verify)."""
    return os.environ.get("MTPLX_DSV41_DSPARK_VERIFY_K29", "").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def dspark_decode_kernel_env_defaults() -> dict:
    """The env vars the lane sets by default for a DSpark run: K30 (selected keys)
    always, K29 (fused decode attention) unless ``MTPLX_DSV41_DSPARK_VERIFY_K29=0``.
    Single source used by both the served context and the bench harness (setdefault),
    so an operator's explicit setting always wins."""
    if _dspark_decode_kernels_disabled():
        return {}
    armed = {_K30_ENV: "1"}
    if _dspark_verify_k29_enabled():
        armed[_K29_ENV] = "1"
    return armed


@contextlib.contextmanager
def arm_dspark_decode_kernels():
    """Arm K29/K30 for the duration of a DSpark run (default ON): set each env flag
    to "1" only where the operator has not already set it, restoring after. The
    small-M verify then routes through the fused decode attention + selected-key
    gather instead of the prefill path. ``MTPLX_DSV41_DSPARK_DECODE_KERNELS=0``
    opts out of both; ``MTPLX_DSV41_DSPARK_VERIFY_K29=0`` keeps K30 but not K29."""
    defaults = dspark_decode_kernel_env_defaults()
    saved = {}
    try:
        for key, val in defaults.items():
            if key not in os.environ:
                saved[key] = None
                os.environ[key] = val
        yield
    finally:
        for key, prev in saved.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


def _verify_decode_phase_enabled() -> bool:
    """Route the K+1-row verify forward through the DECODE expert-routing phase.

    Default ON. The streamed switch keys its phase off token_count
    (``current_expert_routing_phase``): >1 row -> PREFILL (the wave/admission
    machinery + dense-expert re-reads, seconds per call), ==1 -> DECODE
    (persistent-slot small-M gather). An MTP verify batch is decode traffic
    regardless of width (mtplx.runtime._expert_routing_context routes
    ``decode_verify`` as DECODE), so the K+1-row verify must run under DECODE or it
    pays the prefill cost every cycle. Only the routing MACHINERY changes, never
    the gathered experts or the matmul, so this is byte-identical to PREFILL and
    the greedy verify stays authoritative. ``MTPLX_DSV41_DSPARK_VERIFY_DECODE_PHASE=0``
    forces PREFILL for the window-25 A/B."""
    raw = os.environ.get("MTPLX_DSV41_DSPARK_VERIFY_DECODE_PHASE", "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _verify_routing_context(enabled: bool):
    """Context that marks the target verify forward as decode traffic so the
    streamed expert switch uses the DECODE phase (persistent-slot small-M gather)
    instead of PREFILL. No-op on a non-streamed model (unit-test double) and when
    disabled. Both markers are set: ``attention_phase("decode_verify")`` steers
    the served ``rt.forward_ar`` routing context, and ``expert_routing_phase(DECODE)``
    forces the phase on the bench path's direct ``model(...)`` call."""
    if not enabled:
        return contextlib.nullcontext()
    try:
        from mtplx.attention_context import attention_phase
        from mtplx.expert_streaming import RoutingPhase
        from mtplx.models.expert_mlx import expert_routing_phase
    except Exception:  # pragma: no cover - streamed runtime not importable
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def _combined():
        with attention_phase("decode_verify"), expert_routing_phase(RoutingPhase.DECODE):
            yield

    return _combined()


#: DSpark MTP residents actually materialized on the ``with_mtp=True`` load path
#: (dense mxfp8 + 3x128 mxfp4 experts + heads ~= 6.7-7.4 GiB, W18 /
#: OPTIMIZATION_LEDGER §1.1). The streaming loader's planner applies
#: ``text_only_resident_discount`` unconditionally (it frees the MTP+vision
#: residents for expert slots), so a ``with_mtp`` load over-reserves slots by
#: exactly the MTP residents it then loads. The bench harness reserves this out of
#: the memory budget so the slot pool (expert cache) shrinks and the plan fits.
DSPARK_MTP_RESIDENT_BYTES = int(7.4 * (1024 ** 3))


def dspark_bench_loader_overrides(
    *,
    want_dspark: bool,
    memory_limit_bytes: int,
    expert_cache_limit_bytes: Optional[int],
    mtp_resident_bytes: int = DSPARK_MTP_RESIDENT_BYTES,
    reprice: bool = True,
) -> tuple[Optional[bool], int, Optional[int]]:
    """Loader kwargs for a DSpark-DIRECT (or ``--with-mtp``) bench load (W57).

    Returns ``(with_mtp, memory_limit_bytes, expert_cache_limit_bytes)``. When the
    head is wanted: ``with_mtp=True`` and, if ``reprice``, both budgets reduced by
    the MTP residents so the planner's default text-only discount does not
    over-commit expert slots.  ``reprice=False`` (the ``--no-reprice`` A/B arm)
    loads the head at the FULL budget so window 26 can separate the budget effect
    (slots) from any head-load code-path effect: at the same budget the streamed
    slot plan is identical (the head's 128 experts stay resident, the streamed
    runtime keeps its 40 backbone layers -- deepseek_v41_loader.
    construct_deepseek_v41_resident_model), so a no-reprice slowdown is a code path,
    not slots.  For a non-head run: ``with_mtp=None`` and budgets unchanged.  Pure
    function, unit-tested on CPU with no model.
    """
    if not want_dspark:
        return None, int(memory_limit_bytes), expert_cache_limit_bytes
    if not reprice:
        return True, int(memory_limit_bytes), expert_cache_limit_bytes
    gib = 1024 ** 3
    reserved_memory = max(gib, int(memory_limit_bytes) - int(mtp_resident_bytes))
    reserved_cache = (
        None
        if expert_cache_limit_bytes is None
        else max(0, int(expert_cache_limit_bytes) - int(mtp_resident_bytes))
    )
    return True, reserved_memory, reserved_cache


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
@dataclass
class DSparkDecodeStats:
    """Per-run DSpark-direct decode counters.

    ``drafted_by_depth[i]`` / ``accepted_by_depth[i]`` count the draft proposed /
    accepted at depth ``i`` (0-indexed) across all cycles; their sums are
    ``drafted_tokens`` / ``accepted_drafts``.  ``cycles`` == ``verify_calls`` (one
    K+1-row verify per cycle).  ``correction_tokens`` counts cycles that emitted a
    reject-correction; ``bonus_tokens`` counts cycles that emitted an all-accept
    bonus.
    """

    speculative_depth: int = 0
    cycles: int = 0
    verify_calls: int = 0
    drafted_tokens: int = 0
    accepted_drafts: int = 0
    rejected_drafts: int = 0
    correction_tokens: int = 0
    bonus_tokens: int = 0
    generated_tokens: int = 0
    drafted_by_depth: List[int] = field(default_factory=list)
    accepted_by_depth: List[int] = field(default_factory=list)
    # Coarse per-phase decode wall (seconds, summed over cycles) -- the per-cycle
    # cost table (draft = 3 resident MTP stage forwards; verify = the K+1-row
    # target forward; accept = the greedy/spec decision; commit = trim + seed).
    draft_time_s: float = 0.0
    verify_time_s: float = 0.0
    accept_time_s: float = 0.0
    commit_time_s: float = 0.0
    verify_decode_phase: bool = True

    def _ensure_depth(self, k: int) -> None:
        if len(self.drafted_by_depth) < k:
            self.drafted_by_depth.extend([0] * (k - len(self.drafted_by_depth)))
            self.accepted_by_depth.extend([0] * (k - len(self.accepted_by_depth)))

    def accept_rate(self) -> float:
        return (self.accepted_drafts / self.drafted_tokens) if self.drafted_tokens else 0.0

    def tokens_per_cycle(self) -> float:
        return (self.generated_tokens / self.cycles) if self.cycles else 0.0

    def accept_rate_by_depth(self) -> List[Optional[float]]:
        return [
            (self.accepted_by_depth[i] / self.drafted_by_depth[i])
            if self.drafted_by_depth[i]
            else None
            for i in range(len(self.drafted_by_depth))
        ]

    def _per_cycle_ms(self, total_s: float) -> Optional[float]:
        return (1000.0 * total_s / self.cycles) if self.cycles else None

    def to_dict(self) -> dict:
        return {
            "speculative_depth": self.speculative_depth,
            "cycles": self.cycles,
            "verify_calls": self.verify_calls,
            "drafted_tokens": self.drafted_tokens,
            "accepted_drafts": self.accepted_drafts,
            "rejected_drafts": self.rejected_drafts,
            "correction_tokens": self.correction_tokens,
            "bonus_tokens": self.bonus_tokens,
            "generated_tokens": self.generated_tokens,
            "drafted_by_depth": list(self.drafted_by_depth),
            "accepted_by_depth": list(self.accepted_by_depth),
            "accept_rate": self.accept_rate(),
            "tokens_per_cycle": self.tokens_per_cycle(),
            "accept_rate_by_depth": self.accept_rate_by_depth(),
            "verify_decode_phase": self.verify_decode_phase,
            "per_cycle": {
                "draft_ms": self._per_cycle_ms(self.draft_time_s),
                "verify_ms": self._per_cycle_ms(self.verify_time_s),
                "accept_ms": self._per_cycle_ms(self.accept_time_s),
                "commit_ms": self._per_cycle_ms(self.commit_time_s),
            },
            "phase_time_s": {
                "draft": self.draft_time_s,
                "verify": self.verify_time_s,
                "accept": self.accept_time_s,
                "commit": self.commit_time_s,
            },
        }


# ---------------------------------------------------------------------------
# W77: greedy divergence classification (tie-flip vs genuine divergence)
# ---------------------------------------------------------------------------
# The DSpark-DIRECT greedy stream is byte-for-byte AR *by construction* -- the
# verify argmax is authoritative -- so any first-token difference from the AR
# reference comes from the target FORWARD returning slightly different logits at
# the same committed context depending on the row count: AR runs a 1-row (M=1)
# decode forward per token, the verify runs a K+1-row (M>1) forward, and under
# cell16k levers (HEAD_MODE=bf16 head GEMV, ATTN_COMPILE shape-specialised tapes,
# SELECTED_KEYS rows>1 gather-softmax) those two shapes take different matmul /
# softmax kernels whose float reassociation is in the bf16 rounding class (never
# bit-identical on Metal; see W40_HEAD_LEVER / K30 notes).  A greedy argmax only
# *flips* when the AR top-1/top-2 logit gap at that position is within that
# rounding envelope -- David's standing rule: "inexact is fine if it's from
# tie-breaker flips" ([[dsv41-inexact-ok-if-tie-flips]]).
#
# Threshold justification (documented per the coordinator ask):
#   * HEAD_MODE=bf16 (W40/K21) casts the final hidden to bf16 before the head
#     GEMV; bf16 carries a 7-bit mantissa, so its unit round-off is 2**-8 ~=
#     3.9e-3.  A single logit therefore carries a bf16-class perturbation of
#     ~|logit| * 2**-8 plus the M=1-vs-M>1 accumulation-order difference of the
#     head GEMM.
#   * The W77 CPU per-lever probe measures the actual max |Δlogit| between the
#     M=1 and M=K+1 forwards for each cell16k lever (see
#     docs/deepseek-v41/W77_DSPARK_DIVERGENCE.md); the bf16-class floor below is
#     that measured order of magnitude (~1e-2 logit units).
#   * A greedy flip caused purely by rounding needs the top-2 gap to be *within*
#     that perturbation, so the tie-flip threshold is set to 3x the bf16-class
#     floor (~3e-2 logit units): a top-2 gap below it is a genuine near-tie that
#     a few bf16 ulps flip (class "tie_flip", acceptable); a gap above it means
#     the argmax changed for a reason larger than rounding (class "divergent" --
#     a real lane bug or a non-rounding lever), which must stay loud.
DSPARK_BF16_CLASS_DELTA = 1.0e-2
#: Default AR top-2 logit gap (in logit units) below which a greedy divergence is
#: classed a tie-break flip.  3x the bf16-class floor.  Overridable per-arm via
#: ``ab_decode_env_levers.py --dspark-tie-margin``.
DSPARK_TIE_MARGIN_DEFAULT = 3.0 * DSPARK_BF16_CLASS_DELTA  # 3e-2

# ---------------------------------------------------------------------------
# W120: magnitude-aware (bf16-ulp) tie band + contested-token classifier
# ---------------------------------------------------------------------------
# W119 (docs/deepseek-v41/W119_EAGER_VERIFY_PARITY.md) PROVED the K+1-row verify
# equals the 1-row AR forward BITWISE on CPU fp32; the GPU divergence at window-45
# index 111 (ar_top2_margin 0.125, dspark_top2_margin 0.0, max|Δlogit| 1.125) is
# bf16 accumulation-order between the s=K+1 and s=1 eager einsum tiles quantized by
# the bf16 head -- every delta is an INTEGER number of bf16 ulps.  The old fixed
# 3e-2 band on the AR top-2 gap mislabels it "divergent": at the cell16k operating
# magnitudes (~16-256) one bf16 ulp is 0.125-2.0 -- 4-66x the constant.
#
# A greedy flip from ``ar_token`` to ``dspark_token`` is rounding-class iff BOTH
# hold (W120 rule; the W120 red-team tightened the earlier draft):
#   1. the two forwards differ at the two CONTESTED tokens by no more than the band
#      (``deltas_within_tie_band``) -- a >band perturbation is NOT rounding.  A
#      verify row's own tight top-2, or a decisive AR top-2 elsewhere, is NEVER on
#      its own a reason to absolve; and
#   2. those rounding-scale deltas actually account for the flip -- EITHER a
#      CONTESTED margin (``|row[ar_token] - row[dspark_token]|``, NOT the row's
#      top-1/top-2 gap) is itself below the band (a), OR the deltas can close the
#      smaller contested margin (c): ``min(ar_contested, dspark_contested) <=
#      |Δ(ar_token)| + |Δ(dspark_token)|``.
# ``tie_band = max(tie_margin, k * ulp_bf16(peak))``, ``ulp_bf16(x) =
# 2**(floor(log2|x|)) * 2**-7``, ``peak = max(|ar[ar_token]|, |ar[dspark_token]|)``
# taken from the AR REFERENCE row ONLY (a garbage verify logit must not widen the
# band).  Absolution requires a real AR reference (>= 2 logits, both contested
# tokens in range); otherwise the class is "divergent" (loud, conservative).
#
# ``k`` is read from MTPLX_DSV41_DIVERGENCE_TIE_ULPS at USE.  It is a CLASSIFIER
# setting (it changes only how a divergence receipt is LABELLED, never the tokens
# produced), so it is DELIBERATELY NOT in ab_decode_env_levers.ALL_LEVER_ENVS (the
# decode-lever registry); its value is stamped in the divergence block as tie_ulps.
DSPARK_DIVERGENCE_TIE_ULPS_ENV = "MTPLX_DSV41_DIVERGENCE_TIE_ULPS"
#: Default ``k`` (ulp multiplier) for the magnitude-aware tie band.
DSPARK_DIVERGENCE_TIE_ULPS_DEFAULT = 3
#: Hard cap on ``k`` -- a huge band would absolve every divergence (a k of 64 at
#: |logit| 256 is already a 128-logit band, well past any real bf16 perturbation).
DSPARK_DIVERGENCE_TIE_ULPS_MAX = 64


def _row_to_np(row) -> Optional["np.ndarray"]:
    """1-D float64 view of a logits row, or ``None``.  Casts an ``mx.array`` (incl.
    bf16, which numpy has no native dtype for) through float32 first, so no row
    DTYPE and no absent (``None``) row makes the classifier raise.

    A row with ``ndim > 1`` is a CALLER bug (the contested-token indexing assumes a
    single vocab vector): RAISE ``ValueError`` rather than silently ``reshape(-1)`` a
    ``[rows, vocab]`` block into one vector, which would index garbage contested
    logits (LOW).  Both real callers already pass a 1-D ``reshape(-1)`` row."""
    if row is None:
        return None
    if isinstance(row, mx.array):
        row = np.asarray(row.astype(mx.float32))
    arr = np.asarray(row)
    if arr.ndim > 1:
        raise ValueError(
            f"classify_divergence expects a 1-D logits row, got shape {arr.shape}"
        )
    return arr.reshape(-1).astype(np.float64)


def _logit_at(row_np: Optional["np.ndarray"], tok: Optional[int]) -> Optional[float]:
    """The logit at token ``tok`` of a 1-D float64 row, or ``None`` when the row is
    absent or ``tok`` is out of range."""
    if row_np is None or tok is None:
        return None
    t = int(tok)
    if 0 <= t < row_np.size:
        return float(row_np[t])
    return None


def _ulp_bf16(x: float) -> float:
    """One unit-in-the-last-place of a bfloat16 value of magnitude ``x``.

    bf16 carries a 7-bit mantissa, so for a logit of magnitude ``|x|`` one ulp is
    ``2**(floor(log2|x|)) * 2**-7``.  Returns ``0.0`` for a zero / non-finite ``x``
    (no finite binade) so the caller falls back to the fixed tie-band floor.
    """
    ax = abs(float(x))
    if not (ax > 0.0) or not math.isfinite(ax):
        return 0.0
    return float(2.0 ** (math.floor(math.log2(ax)) - 7))


def _tie_ulps_from_env(explicit: Optional[int]) -> int:
    """Multiplier ``k`` for the magnitude-aware tie band.  ``explicit`` wins;
    otherwise ``MTPLX_DSV41_DIVERGENCE_TIE_ULPS`` is read AT USE (default 3, per
    [[env-flags-read-at-use-not-import]]).

    Only a non-negative integer in ``[0, DSPARK_DIVERGENCE_TIE_ULPS_MAX]`` is
    accepted; a negative / non-integer / malformed value is REJECTED with a WARN
    line and the default (never silently coerced -- ``"-1"`` is not 0, ``"1_0"`` is
    not 10, ``"3.0"``/``"nan"`` are not 3); a value above the cap is clamped with a
    WARN.
    """
    default = DSPARK_DIVERGENCE_TIE_ULPS_DEFAULT
    cap = DSPARK_DIVERGENCE_TIE_ULPS_MAX

    def _bounded(v: int, source: str) -> int:
        if v < 0:
            print(f"[dsv41] WARN: {source} k={v} < 0; using default {default}",
                  flush=True)
            return default
        if v > cap:
            print(f"[dsv41] WARN: {source} k={v} > cap {cap}; clamping to {cap}",
                  flush=True)
            return cap
        return v

    if explicit is not None:
        # Accept an int (not bool) or an integer-valued float; reject a fractional
        # float or any other type rather than truncating silently.
        if isinstance(explicit, bool):
            print(f"[dsv41] WARN: tie_ulps={explicit!r} is not an int; "
                  f"using default {default}", flush=True)
            return default
        if isinstance(explicit, int):
            return _bounded(explicit, "tie_ulps")
        if isinstance(explicit, float):
            if math.isfinite(explicit) and float(explicit).is_integer():
                return _bounded(int(explicit), "tie_ulps")
            print(f"[dsv41] WARN: tie_ulps={explicit!r} is not an integer; "
                  f"using default {default}", flush=True)
            return default
        print(f"[dsv41] WARN: tie_ulps={explicit!r} is not an int; "
              f"using default {default}", flush=True)
        return default

    raw = os.environ.get(DSPARK_DIVERGENCE_TIE_ULPS_ENV, "").strip()
    if not raw:
        return default
    body = raw[1:] if raw[:1] in "+-" else raw
    if not body.isdigit():  # rejects "1_0", "3.0", "nan", "1e20", " 3 " (post-strip ok)
        print(f"[dsv41] WARN: {DSPARK_DIVERGENCE_TIE_ULPS_ENV}={raw!r} is not an "
              f"integer; using default {default}", flush=True)
        return default
    try:
        return _bounded(int(raw), DSPARK_DIVERGENCE_TIE_ULPS_ENV)
    except ValueError:  # e.g. a unicode digit that isdigit() accepts but int() rejects
        print(f"[dsv41] WARN: {DSPARK_DIVERGENCE_TIE_ULPS_ENV}={raw!r} is not an "
              f"integer; using default {default}", flush=True)
        return default


def _peak_contested_logit(ar_row, ar_token, dspark_token) -> Optional[float]:
    """W119 ``peak_contested_logit`` from the AR REFERENCE row ONLY (never the
    verify row -- a garbage verify logit must not widen the band): ``max(
    |ar[ar_token]|, |ar[dspark_token]|)``.  Falls back to the AR row's top-1
    magnitude when neither contested index is in range.  ``None`` if the AR row is
    absent / empty."""
    flat = _row_to_np(ar_row)
    if flat is None or flat.size == 0:
        return None
    mags: List[float] = []
    for tok in (ar_token, dspark_token):
        if tok is not None and 0 <= int(tok) < flat.size:
            mags.append(abs(float(flat[int(tok)])))
    if not mags:
        mags.append(float(np.max(np.abs(flat))))
    return max(mags)


def _top2_margin(row) -> Optional[float]:
    """Logit gap between the top-1 and top-2 entries of a 1-D logits row (numpy
    array or mx.array, incl. bf16).  ``None`` for a row with fewer than 2 entries.

    This is the row's own greedy decision gap -- a diagnostic receipt value.  The
    W120 class decision uses the CONTESTED margin (between ``ar_token`` and
    ``dspark_token``), not this top-2 gap (see :func:`classify_divergence`)."""
    flat = _row_to_np(row)
    if flat is None or flat.size < 2:
        return None
    top = np.argpartition(flat, -2)[-2:]
    two = np.sort(flat[top])
    return float(two[1] - two[0])


def classify_divergence(
    *,
    index: int,
    ar_token: Optional[int],
    dspark_token: Optional[int],
    ar_logits_row=None,
    dspark_logits_row=None,
    tie_margin: float = DSPARK_TIE_MARGIN_DEFAULT,
    tie_ulps: Optional[int] = None,
) -> dict:
    """Classify the FIRST greedy divergence of a DSpark stream from its AR
    reference at position ``index`` (W77 primitive, W120 contested-token rule).

    ``ar_logits_row`` is the AR forward's full logits vector at ``index`` (the
    faithful M=1 replay); ``dspark_logits_row`` is the verify forward's logits row
    that produced the committed DSpark token there (captured, zero extra forwards).
    Either row may be ``None`` (unavailable) -- the classifier degrades to the
    signals it has and never raises on an absent row or an unusual dtype
    (``mx.array`` rows, incl. bf16, are cast); a malformed row with ndim > 1 is a
    caller bug and DOES raise (see :func:`_row_to_np`).

    W120 rule (see the module block above and W120_DIVERGENCE_TIE_BAND.md).  The
    flip ``ar_token -> dspark_token`` is ``"tie_flip"`` iff it is absolvable AND
    (near-tie by band OR rounding-class by delta):

      * **absolvable** -- a REAL AR reference (``ar_top2_margin`` computable, i.e.
        >= 2 logits) AND both CONTESTED margins computable (both tokens in range on
        both rows) AND both forwards differ at the two contested tokens by no more
        than the band (``deltas_within_tie_band``).  A >band perturbation is NOT
        rounding, so a verify row's own tight top-2 (or a decisive AR top-2 at other
        tokens) is never on its own a reason to absolve.  ADDITIONALLY the rows must
        be SELF-CONSISTENT with the tokens they are credited with
        (``rows_consistent``: ``ar_token`` is an argmax of the AR row and
        ``dspark_token`` is an argmax of the verify row -- ties allowed, since the
        authoritative verify side is an exact bf16 tie in the very case W120 exists
        to absolve; a token strictly below its row's max means that row did NOT
        produce it, so the flip is not rounding), and all four contested logits must
        be FINITE (a NaN / inf contested logit never absolves -- ``max()`` over a NaN
        delta is order-dependent, so ``deltas_within_tie_band`` is forced to False,
        not True, whenever any contested logit is non-finite).
      * **band** ``tie_band = max(tie_margin, k * ulp_bf16(peak))`` with ``peak =
        max(|ar[ar_token]|, |ar[dspark_token]|)`` from the AR REFERENCE row ONLY.
      * **(a) near_tie_by_band** -- ``min(ar_contested_margin,
        dspark_contested_margin) < tie_band`` (CONTESTED margins
        ``|row[ar_token] - row[dspark_token]|``, NOT the row top-1/top-2 gap).
      * **(c) rounding_class_by_delta** -- ``min(ar_contested_margin,
        dspark_contested_margin) <= |Δ(ar_token)| + |Δ(dspark_token)|`` (the
        measured contested deltas can close the smaller contested margin).

    Otherwise ``class`` is ``"divergent"`` (loud, conservative -- a failed / partial
    M=1 replay is never silently absolved).

    ``k`` (the ulp multiplier) defaults to :data:`DSPARK_DIVERGENCE_TIE_ULPS_DEFAULT`
    and is overridable via the ``tie_ulps`` argument or, at USE, the
    ``MTPLX_DSV41_DIVERGENCE_TIE_ULPS`` env (a CLASSIFIER setting, NOT a decode
    lever -- kept out of ALL_LEVER_ENVS, stamped here instead).

    Returns a receipt-ready dict (all JSON scalars, no arrays).  W120 ADDS keys
    (``tie_band_used``, ``tie_ulps``, ``peak_contested_logit``,
    ``ulp_bf16_at_peak``, ``ar_contested_margin``, ``dspark_contested_margin``,
    ``delta_at_ar_token``, ``delta_at_dspark_token``, ``rounding_class_by_delta``,
    ``deltas_within_tie_band``, ``rows_consistent``, ``ar_logit_at_ar_token``,
    ``ar_logit_at_dspark_token``, ``dspark_logit_at_ar_token``,
    ``dspark_logit_at_dspark_token`` -- the last four so a serialized receipt is
    self-decidable without the rows); every W77 key (``divergence_index``,
    ``ar_token``, ``dspark_token``, ``ar_top2_margin``, ``dspark_top2_margin``,
    ``max_abs_logit_delta``, ``tie_margin``, ``class``) is kept unchanged.
    """
    ar_np = _row_to_np(ar_logits_row)
    dsp_np = _row_to_np(dspark_logits_row)
    ar_margin = _top2_margin(ar_np)       # row top-2 gaps: DIAGNOSTIC receipt keys
    dspark_margin = _top2_margin(dsp_np)  # (NOT used for the class decision)

    # Contested logits (at ar_token and dspark_token) on each row.
    ar_l_at_ar = _logit_at(ar_np, ar_token)
    ar_l_at_dsp = _logit_at(ar_np, dspark_token)
    dsp_l_at_ar = _logit_at(dsp_np, ar_token)
    dsp_l_at_dsp = _logit_at(dsp_np, dspark_token)

    # Contested margins -- the gap between the TWO contested tokens on each row
    # (HIGH-2: the row's own top-1/top-2 gap can be at other tokens entirely).
    ar_contested_margin: Optional[float] = (
        abs(ar_l_at_ar - ar_l_at_dsp)
        if ar_l_at_ar is not None and ar_l_at_dsp is not None else None
    )
    dspark_contested_margin: Optional[float] = (
        abs(dsp_l_at_dsp - dsp_l_at_ar)
        if dsp_l_at_dsp is not None and dsp_l_at_ar is not None else None
    )

    # Per-token AR-vs-DSpark deltas at the two contested tokens + vocab-wide max.
    max_abs_delta: Optional[float] = None
    delta_at_ar_token: Optional[float] = None
    delta_at_dspark_token: Optional[float] = None
    if ar_np is not None and dsp_np is not None and ar_np.shape == dsp_np.shape and ar_np.size:
        max_abs_delta = float(np.max(np.abs(ar_np - dsp_np)))
        if ar_l_at_ar is not None and dsp_l_at_ar is not None:
            delta_at_ar_token = abs(ar_l_at_ar - dsp_l_at_ar)
        if ar_l_at_dsp is not None and dsp_l_at_dsp is not None:
            delta_at_dspark_token = abs(ar_l_at_dsp - dsp_l_at_dsp)

    # (b) Magnitude-aware band from the AR REFERENCE row's contested logits ONLY
    # (MEDIUM-1: a garbage verify logit must not widen the band).
    k = _tie_ulps_from_env(tie_ulps)
    peak = _peak_contested_logit(ar_logits_row, ar_token, dspark_token)
    ulp_at_peak = _ulp_bf16(peak) if peak is not None else None
    tie_band = float(tie_margin)
    if ulp_at_peak is not None:
        tie_band = max(tie_band, float(k) * float(ulp_at_peak))

    # All four contested logits must be FINITE.  A NaN / inf contested logit makes a
    # contested delta non-finite, and ``max()`` over a NaN is ORDER-DEPENDENT
    # (``max(0.0, nan)`` is 0.0 but ``max(nan, 0.0)`` is nan), so an unchecked band
    # test could absolve a NaN row (MEDIUM).
    contested_logits_finite = all(
        v is not None and math.isfinite(v)
        for v in (ar_l_at_ar, ar_l_at_dsp, dsp_l_at_ar, dsp_l_at_dsp)
    )

    # The closing deltas must themselves be rounding-class (both <= band).  A >band
    # perturbation is not rounding (HIGH-1); a non-finite delta is FORCED to False
    # (never True), not left to the order-dependent ``max`` comparison (MEDIUM).
    deltas_within_tie_band: Optional[bool] = None
    if delta_at_ar_token is not None and delta_at_dspark_token is not None:
        if not (math.isfinite(delta_at_ar_token) and math.isfinite(delta_at_dspark_token)):
            deltas_within_tie_band = False
        else:
            deltas_within_tie_band = bool(
                max(delta_at_ar_token, delta_at_dspark_token) <= tie_band
            )

    # (c) rounding_class_by_delta -- W119 literal rule on the CONTESTED margins.
    rounding_class_by_delta: Optional[bool] = None
    if (
        ar_contested_margin is not None
        and dspark_contested_margin is not None
        and delta_at_ar_token is not None
        and delta_at_dspark_token is not None
    ):
        rounding_class_by_delta = bool(
            min(ar_contested_margin, dspark_contested_margin)
            <= (delta_at_ar_token + delta_at_dspark_token)
        )

    # rows_consistent (MEDIUM): each row must actually PRODUCE the token it is
    # credited with -- ``ar_token`` an argmax of the AR row AND ``dspark_token`` an
    # argmax of the verify row.  TIES are allowed: the authoritative verify side is
    # an exact bf16 tie in the very case W120 exists to absolve (dspark_top2_margin
    # 0.0), and DSpark's tie-break legitimately commits one of the tied maxima, so a
    # token that TIES for its row's max is consistent; a token STRICTLY BELOW the max
    # means that row did not produce it (a fabricated / mismatched capture) and the
    # flip is not rounding.  Non-finite logits fail here too (``== nanmax`` is False).
    rows_consistent = bool(
        ar_np is not None and ar_np.size
        and dsp_np is not None and dsp_np.size
        and contested_logits_finite
        and ar_l_at_ar == float(np.max(ar_np))
        and dsp_l_at_dsp == float(np.max(dsp_np))
    )

    # Absolution eligibility (MEDIUM-2): a real AR reference (a top-2 margin exists,
    # i.e. >= 2 logits), both contested margins computable, the rows self-consistent
    # with their tokens, all four contested logits finite, and the deltas gate
    # computable AND within the band.
    absolvable = bool(
        rows_consistent
        and contested_logits_finite
        and ar_margin is not None
        and ar_contested_margin is not None
        and dspark_contested_margin is not None
        and deltas_within_tie_band is not None
        and deltas_within_tie_band
    )
    near_tie_by_band = bool(
        absolvable
        and min(ar_contested_margin, dspark_contested_margin) < tie_band
    )
    tie_flip_by_delta = bool(absolvable and rounding_class_by_delta)

    cls = "tie_flip" if (near_tie_by_band or tie_flip_by_delta) else "divergent"
    return {
        "divergence_index": int(index),
        "ar_token": None if ar_token is None else int(ar_token),
        "dspark_token": None if dspark_token is None else int(dspark_token),
        "ar_top2_margin": ar_margin,
        "dspark_top2_margin": dspark_margin,
        "max_abs_logit_delta": max_abs_delta,
        "tie_margin": float(tie_margin),
        # W120 additive keys (the W77 keys above are unchanged):
        "tie_band_used": float(tie_band),
        "tie_ulps": int(k),
        "peak_contested_logit": None if peak is None else float(peak),
        "ulp_bf16_at_peak": None if ulp_at_peak is None else float(ulp_at_peak),
        "ar_contested_margin": ar_contested_margin,
        "dspark_contested_margin": dspark_contested_margin,
        "delta_at_ar_token": delta_at_ar_token,
        "delta_at_dspark_token": delta_at_dspark_token,
        "rounding_class_by_delta": rounding_class_by_delta,
        "deltas_within_tie_band": deltas_within_tie_band,
        "rows_consistent": rows_consistent,
        # raw contested logits so a serialized receipt is self-decidable:
        "ar_logit_at_ar_token": ar_l_at_ar,
        "ar_logit_at_dspark_token": ar_l_at_dsp,
        "dspark_logit_at_ar_token": dsp_l_at_ar,
        "dspark_logit_at_dspark_token": dsp_l_at_dsp,
        "class": cls,
    }


class DivergenceCapture:
    """Watches a greedy DSpark stream against an AR reference and snapshots the
    verify logits row of the FIRST committed token that differs from AR.

    Zero extra forwards: the row is read out of the ``verify_logits`` the cycle
    already computed.  Only meaningful for greedy decode (temperature 0), where
    the emitted token at local block index ``m`` is the argmax of
    ``verify_logits[0, m]`` (an accepted draft) or, for the last emitted token,
    the correction/bonus argmax at that row.  ``ar_reference`` includes the prompt
    prefill token at position 0, aligning 1:1 with :func:`dspark_generate`'s
    returned ids, so ``new_tokens[i]`` is global position ``i + 1``.
    """

    def __init__(self, ar_reference: Optional[Sequence[int]]):
        self.ar_reference: Optional[List[int]] = (
            [int(t) for t in ar_reference] if ar_reference is not None else None
        )
        self.index: Optional[int] = None
        self.ar_token: Optional[int] = None
        self.dspark_token: Optional[int] = None
        #: full logits row (np.float32 [vocab]) of the verify forward at the
        #: first diverging committed token -- kept in-process only, never
        #: serialised (the AB harness reads scalars off it via classify_divergence).
        self.dspark_logits_row = None
        self.dspark_top2_margin: Optional[float] = None

    @property
    def found(self) -> bool:
        return self.index is not None

    def observe(self, *, base_len: int, committed: Sequence[int], verify_logits) -> None:
        """Compare this cycle's actually-committed tokens against the AR
        reference and snapshot the verify row of the first mismatch.

        ``base_len`` is ``len(new_tokens)`` before this cycle appended, so the
        m-th committed token is at global position ``base_len + m + 1`` and was
        produced by ``verify_logits[0, m]``.
        """
        if self.found or self.ar_reference is None or verify_logits is None:
            return
        for m, tok in enumerate(committed):
            gpos = base_len + m + 1
            if gpos >= len(self.ar_reference):
                return
            if int(tok) != int(self.ar_reference[gpos]):
                row = np.asarray(verify_logits[0, m].astype(mx.float32)).reshape(-1)
                self.index = gpos
                self.dspark_token = int(tok)
                self.ar_token = int(self.ar_reference[gpos])
                self.dspark_logits_row = row
                self.dspark_top2_margin = _top2_margin(row)
                return


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _is_stop(token: int, stop_ids: Optional[set]) -> bool:
    return bool(stop_ids) and int(token) in stop_ids


def _confidence_threshold_from_env(explicit: Optional[float]) -> Optional[float]:
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get("MTPLX_DSV41_DSPARK_CONF_THRESHOLD", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _effective_draft_len(conf_row: mx.array, k: int, threshold: Optional[float]) -> int:
    """Early stop: keep the leading run of drafts whose (sigmoid) confidence
    clears ``threshold``.  ``threshold is None`` keeps all ``k`` (the confidence
    head is still computed by the draft head; this only trims what we verify)."""
    if threshold is None or k <= 0:
        return k
    conf = np.asarray(mx.sigmoid(conf_row.astype(mx.float32)))
    conf = conf.reshape(-1)
    keep = 0
    for i in range(k):
        if float(conf[i]) >= threshold:
            keep += 1
        else:
            break
    # Always verify at least one draft so a cycle still makes forward progress
    # off the draft; a fully-below-threshold row degrades to a 2-row verify.
    return max(1, keep)


# ---------------------------------------------------------------------------
# W63 / K32: device-side sampling AR decode (one-step-lag software pipeline)
# ---------------------------------------------------------------------------
# The plain AR / greedy decode loop ends every token on a device->host round
# trip: logits -> argmax/sample on device -> the id is read to the host
# (mx.eval / .item()) -> stop/detok check -> the id is fed back as the next
# input embedding.  That read is a full GPU drain per token, and the next
# step's graph cannot be encoded until it returns, so the decode is
# dispatch-bound (~160 ms/token at 1K, [[dsv41-decode-lever-ledger]]).
#
# This lane keeps the sampled token ON DEVICE: it stays a lazy mx.array fed
# straight into the next forward, whose embedding lookup is ``mx.take`` on that
# id array (no host round trip to become the next input).  The host reads token
# ids with a ONE-STEP LAG -- step t+1's forward is already submitted
# (mx.async_eval on the step outputs) before token t's id is materialized -- so
# the GPU never idles on the host read.
#
# GREEDY (the DSV4.1 benchmark shape) stays BYTE-IDENTICAL to the classic argmax
# loop: argmax is deterministic, so the same integer id feeds the next forward
# either way (proven on the CPU double, tests/models/
# test_deepseek_v41_device_sample.py).
#
# SAMPLED reuses the shipped device shaped sampler ``_mx_lazy_sample``
# (temp -> top-k -> top-p -> categorical) -- the same one the qwen4_exp
# MTPLX_AR_PIPELINE lane uses, so no other model's sampler is touched.  It is NOT
# token-for-token equal to the host numpy path, by two documented deviations
# (W63_DEVICE_SAMPLE.md): (1) it draws from an ``mx.random`` device key, not the
# numpy generator -- a seed-mapping change; (2) its top-p nucleus is taken over
# the RENORMALIZED top-k softmax, so when top-k truncates tail mass it keeps a
# NARROWER nucleus than the host's full-vocab-softmax nucleus.  Both are
# conservative: the sampled support is a subset of the host top-k / host support
# (verified on the CPU double), so the device never draws a token the host would
# not.  Sampled callers that need the exact host distribution keep the default
# (device sample off) and stay on the classic path.
_DEVICE_SAMPLE_ENV = "MTPLX_DSV41_DEVICE_SAMPLE"


def device_sample_enabled() -> bool:
    """True when the W63 device-sample AR decode lane is armed
    (``MTPLX_DSV41_DEVICE_SAMPLE=1``; default off)."""
    return os.environ.get(_DEVICE_SAMPLE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _sampler_is_greedy(sampler) -> bool:
    if sampler is None:
        return True
    return float(getattr(sampler, "temperature", 0.0) or 0.0) <= 0.0


def device_sample_eligible(sampler) -> tuple[bool, str]:
    """Return ``(eligible, reason)`` for the device-sample lane.

    Greedy (``sampler is None`` or ``temperature <= 0``) is always eligible.
    Sampled requires ``temperature > 0`` and ``top_k > 1`` (the device shaped
    sampler ``mtplx.generation._mx_lazy_sample`` covers exactly
    temp -> top-k -> top-p -> categorical); presence/frequency penalties are
    host-only (they need the running token Counter), so a penalised request
    stays on the classic path.
    """
    if _sampler_is_greedy(sampler):
        return True, "greedy"
    top_k = int(getattr(sampler, "top_k", 0) or 0)
    if top_k <= 1:
        return False, "sampled device lane requires top_k > 1"
    if getattr(sampler, "presence_penalty", 0.0) or getattr(
        sampler, "frequency_penalty", 0.0
    ):
        return False, "presence/frequency penalties are host-only"
    return True, "sampled"


def run_device_sample_decode(
    *,
    forward_row: Callable[[mx.array], mx.array],
    first_token: int,
    n_more: int,
    sampler=None,
    seed: int = 0,
    key: Optional[mx.array] = None,
    stop_ids: Optional[set] = None,
    on_token: Optional[Callable[[int], None]] = None,
    abort_check: Optional[Callable[[], bool]] = None,
    timing: Optional[dict] = None,
) -> tuple[List[int], str, int]:
    """One-step-lag device-sample AR decode.

    ``forward_row(ids_2d)`` runs one target forward over a ``[1, T]`` token-id
    array (device-side; its embedding lookup is ``mx.take`` on the id array, so
    the sampled token never round-trips to the host to become the next input)
    and returns the last-position logits row ``[vocab]``.  ``first_token`` is the
    already-emitted token whose forward has NOT run yet; this loop emits up to
    ``n_more`` further tokens (excluding ``first_token``).

    The sampled/greedy token stays a lazy device array fed straight into the
    next ``forward_row``; the host reads token *t* with a ONE-STEP LAG (step
    *t+1*'s forward is already submitted via ``mx.async_eval`` before *t*'s id is
    materialized), so the GPU never idles on the host read.  At a stop token, at
    ``n_more``, or on abort the loop has already submitted ONE extra forward (the
    just-emitted token's step) whose sampled successor is discarded: **at most
    one wasted forward per completion**, returned as the third tuple element (the
    classic loop breaks before forwarding its final token, so the device lane
    computes and drops exactly that one step -- greedy output is unaffected).

    Returns ``(more_tokens, finish_reason, extra_forward_steps)``.
    """
    import time as _time

    greedy = _sampler_is_greedy(sampler)
    if not greedy and key is None:
        key = mx.random.key(int(seed) & 0x7FFFFFFF)

    def _next_from_row(row: mx.array) -> mx.array:
        nonlocal key
        if greedy:
            return mx.argmax(row, axis=-1).reshape(1)
        from mtplx.generation import _mx_lazy_sample

        key, sub = mx.random.split(key)
        return _mx_lazy_sample(row, sampler, sub).reshape(1)

    more: List[int] = []
    if n_more <= 0:
        return more, "length", 0
    if _is_stop(first_token, stop_ids):
        return more, "stop", 0

    def _step(tok_lazy: mx.array):
        row = forward_row(tok_lazy.reshape(1, 1))
        return row, _next_from_row(row)

    finish_reason = "length"
    extra = 0
    # Prime the pipeline: forward first_token, sample its successor (lazy),
    # submit without blocking.
    _row_lazy, tok_lazy = _step(mx.array([int(first_token)]))
    mx.async_eval(tok_lazy)
    while True:
        if abort_check is not None and abort_check():
            finish_reason = "abort"
            extra = 1  # the in-flight tok_lazy step is discarded unread
            break
        # Submit step t+1 BEFORE reading token t, so the GPU stays busy across
        # the host materialization.
        b = _time.perf_counter()
        row_next, tok_next = _step(tok_lazy)
        mx.async_eval(tok_next)
        if timing is not None:
            timing["build_s"] = timing.get("build_s", 0.0) + (
                _time.perf_counter() - b
            )
        w = _time.perf_counter()
        v = int(tok_lazy.item())  # lagged read of token t (already in flight)
        if timing is not None:
            timing["wait_s"] = timing.get("wait_s", 0.0) + (_time.perf_counter() - w)
        more.append(v)
        if on_token is not None:
            on_token(v)
        if _is_stop(v, stop_ids):
            finish_reason = "stop"
            extra = 1  # forward(v) already ran; its successor is discarded
            break
        if len(more) >= n_more:
            finish_reason = "length"
            extra = 1
            break
        _row_lazy, tok_lazy = row_next, tok_next
    return more, finish_reason, extra


def _target_forward(model):
    """Default target forward: ``(logits, main_hidden)`` over the given rows,
    threading the shared V4.1 cache."""

    def _fwd(ids: mx.array, cache) -> tuple:
        return model(ids, cache=cache, return_hidden=True)

    return _fwd


# ---------------------------------------------------------------------------
# core loop
# ---------------------------------------------------------------------------
def _decode_cycles(
    *,
    model,
    forward: Callable[[mx.array, Any], tuple],
    cache,
    mtp_caches: Sequence[Any],
    primary: int,
    main_h: mx.array,
    max_tokens: int,
    sampler,
    rng: np.random.Generator,
    stop_ids: Optional[set],
    k_request: int,
    confidence_threshold: Optional[float],
    stats: DSparkDecodeStats,
    token_callback: Optional[Callable[[List[int]], None]],
    abort_check: Optional[Callable[[], bool]],
    verify_decode_phase: bool = True,
    divergence_capture: Optional["DivergenceCapture"] = None,
) -> tuple[List[int], str]:
    """Run DSpark-direct cycles from ``primary`` (already emitted) + its predictor
    hidden ``main_h``.  Returns ``(new_tokens, finish_reason)`` where ``new_tokens``
    excludes ``primary``."""
    import time as _time

    from mtplx.generation import _sample_from_logits
    from mtplx.sampling import (
        acceptance_probability as _accept_prob,
        residual_distribution as _residual,
        sample_from_distribution as _sample_dist,
    )
    from mtplx.sampling import SparseDistribution

    head = model.head
    embed = model.model.embed_tokens
    dspark = model.mtp
    block_size = int(getattr(dspark, "block_size", 0) or 0)
    k_cap = min(int(k_request), block_size) if block_size else int(k_request)
    greedy = float(getattr(sampler, "temperature", 0.0)) <= 0.0
    stats.speculative_depth = k_cap
    stats._ensure_depth(k_cap)

    def _target_p(row: mx.array):
        from mtplx.generation import _distribution_from_mlx_logits

        return _distribution_from_mlx_logits(row, sampler)

    stats.verify_decode_phase = bool(verify_decode_phase)
    new_tokens: List[int] = []
    finish_reason = "length"
    if _is_stop(primary, stop_ids):
        return new_tokens, "stop"

    while len(new_tokens) < max_tokens:
        if abort_check is not None and abort_check():
            finish_reason = "abort"
            break
        # arm the W37 probe (if a --stage-timing session is active) so BOTH the
        # draft and the verify record from cycle 0, not just after the first verify.
        _arm_stage_recording()

        # ---- draft a block, apply the confidence early stop --------------
        # The 3 DSpark stages run RESIDENT mxfp4 experts (SwitchGLU), never the
        # streamed switch, so drafting stays on resident weights (no phase issue).
        # The draft block's PURE chains (attention prep, Hyper-Connection prep, MoE
        # gate/combine, markov, confidence) collapse to mx.compile tapes under
        # MTPLX_DSV41_DRAFT_COMPILE (K33/W65, default OFF, byte-identical: draft
        # tokens are identical with the flag on/off, so greedy verify == AR holds
        # regardless). One host sync per cycle (the mx.eval below), never per markov
        # step -- the markov argmax stays lazy inside draft_block.
        k_eff = 0
        drafts: List[int] = []
        _t = _time.perf_counter()
        if k_cap > 0:
            primary_arr = mx.array([int(primary)])
            with _stage("dspark.draft"):
                out_ids, _dlogits, conf = dspark.draft_block(
                    main_h, primary_arr, list(mtp_caches), embed, head
                )
                mx.eval(out_ids, conf)
            out_np = np.asarray(out_ids).reshape(-1)
            k_eff = _effective_draft_len(conf, k_cap, confidence_threshold)
            drafts = [int(out_np[1 + i]) for i in range(k_eff)]
        stats.draft_time_s += _time.perf_counter() - _t

        # ---- verify: one forward over [primary, d1..d_keff] --------------
        # Route the K+1-row verify through the DECODE expert-routing phase (an MTP
        # verify batch is decode traffic regardless of width) so the streamed
        # switch does a persistent-slot small-M gather instead of the PREFILL
        # wave/admission/dense re-read that costs seconds per cycle. The W37 frame
        # records the verify's internal model stages when a probe is armed.
        block_ids = [int(primary)] + drafts
        before = snapshot_untrimmable_cache(cache)
        _t = _time.perf_counter()
        with _verify_routing_context(verify_decode_phase), _frame(), _stage("dspark.verify"):
            verify_logits, verify_hidden = forward(mx.array([block_ids]), cache)
            mx.eval(verify_logits, verify_hidden)
        stats.verify_time_s += _time.perf_counter() - _t
        stats.cycles += 1
        stats.verify_calls += 1

        # ---- acceptance --------------------------------------------------
        _t = _time.perf_counter()
        accepted = 0
        emitted: List[int] = []
        if greedy:
            argmax_rows = np.asarray(mx.argmax(verify_logits[0], axis=-1)).reshape(-1)
            for i in range(k_eff):
                stats.drafted_by_depth[i] += 1
                stats.drafted_tokens += 1
                if int(argmax_rows[i]) == drafts[i]:
                    accepted += 1
                    stats.accepted_by_depth[i] += 1
                    stats.accepted_drafts += 1
                else:
                    break
            correction = int(argmax_rows[accepted])
            emitted = drafts[:accepted] + [correction]
        else:
            vocab = int(verify_logits.shape[-1])
            reject_at = None
            correction = None
            for i in range(k_eff):
                stats.drafted_by_depth[i] += 1
                stats.drafted_tokens += 1
                target_p = _target_p(verify_logits[0, i])
                d = drafts[i]
                # The DSpark draft is greedy (DSparkBlock.temperature == 0), so its
                # proposal is the deterministic point mass q = delta_d; standard
                # speculative sampling with q = delta_d accepts d w.p. min(1, p(d))
                # and draws the correction from norm(max(0, p - q)) -> output ~ p.
                q = SparseDistribution.one_hot(d, vocab)
                ap = _accept_prob(target_p, q, d)
                if float(rng.random()) <= ap:
                    accepted += 1
                    stats.accepted_by_depth[i] += 1
                    stats.accepted_drafts += 1
                else:
                    correction = int(_sample_dist(_residual(target_p, q), rng))
                    reject_at = i
                    break
            if reject_at is None:
                # all k_eff accepted -> bonus from target p at the last row (this
                # is also the exact K=0 == AR path: k_eff==0 -> sample row 0).
                bonus, _ = _sample_from_logits(verify_logits[0, accepted], sampler, rng)
                correction = int(bonus)
            emitted = drafts[:accepted] + [int(correction)]

        # A cycle reaches (evaluates) ``accepted`` drafts plus, when it did not
        # accept the whole block, the one that broke the run -- later block
        # drafts are discarded unread.  drafted_by_depth[i] counts cycles that
        # reached depth i, so drafted_tokens == sum(drafted_by_depth) and each
        # reached draft is either accepted or the single reject.
        if accepted < k_eff:
            stats.rejected_drafts += 1
            stats.correction_tokens += 1
        else:
            stats.bonus_tokens += 1
        stats.accept_time_s += _time.perf_counter() - _t

        # ---- commit: keep [primary, d1..da] in the target cache ----------
        _t = _time.perf_counter()
        with _stage("dspark.commit"):
            kept = trim_verified_window_to_prefix(
                cache, before, verified_tokens=len(block_ids), keep_tokens=accepted + 1
            )
            if not kept:
                # V4.1 caches are all-trimmable, so this should not happen; a
                # non-trimmable entry would need the snapshot+re-forward repair.
                raise RuntimeError(
                    "dspark-direct: verify tail could not be trimmed (non-trimmable "
                    "cache entry); this lane requires an all-trimmable V4.1 cache"
                )
            # seed the DSpark stage windows with the committed tokens' main hiddens
            dspark.seed_main(verify_hidden[:, : accepted + 1, :], list(mtp_caches))
            mx.eval([c.window for c in mtp_caches if getattr(c, "window", None) is not None])
        stats.commit_time_s += _time.perf_counter() - _t

        # ---- emit + stop handling ---------------------------------------
        # Emitted tokens are appended in order up to max_tokens; the stop token
        # is included (generate_ar appends it then breaks -- the terminal stop is
        # stripped from the decoded text later, not from the token count).
        stopped = False
        base_len = len(new_tokens)
        delta: List[int] = []
        for tok in emitted:
            if len(new_tokens) >= max_tokens:
                break
            new_tokens.append(int(tok))
            delta.append(int(tok))
            if _is_stop(tok, stop_ids):
                stopped = True
                break
        stats.generated_tokens = len(new_tokens)
        # W77: for greedy decode, snapshot the verify logits row of the first
        # committed token that differs from the AR reference (zero extra
        # forwards; verify_logits[0, m] produced committed token m).
        if greedy and divergence_capture is not None and delta:
            divergence_capture.observe(
                base_len=base_len, committed=delta, verify_logits=verify_logits
            )
        if token_callback is not None and delta:
            token_callback(delta)
        if stopped:
            finish_reason = "stop"
            break
        if len(new_tokens) >= max_tokens:
            finish_reason = "length"
            break

        # ---- advance: next primary is the correction/bonus --------------
        primary = int(correction)
        main_h = verify_hidden[:, accepted : accepted + 1, :]

    return new_tokens, finish_reason


# ---------------------------------------------------------------------------
# public: model-level loop (unit-test surface)
# ---------------------------------------------------------------------------
def _fire_prefill_callback(
    prefill_callback: Optional[Callable[[dict], None]],
    prompt_ids: Sequence[int],
    prompt_eval_time_s: float,
) -> None:
    """Invoke a prefill callback with the standard
    ``{"prompt_tokens", "prompt_eval_time_s"}`` payload right after the prompt
    prefill.  Shared by the self-contained :func:`dspark_generate` and the served
    :func:`generate_dspark` so both lanes fire it identically.  Telemetry must
    never crash decode, so any callback error is swallowed."""
    if prefill_callback is None:
        return
    try:
        prefill_callback(
            {"prompt_tokens": len(prompt_ids), "prompt_eval_time_s": prompt_eval_time_s}
        )
    except Exception:  # pragma: no cover - defensive; telemetry must not crash decode
        pass


def dspark_generate(
    model,
    prompt_ids: Sequence[int],
    *,
    max_tokens: int,
    sampler,
    seed: int = 0,
    stop_ids: Optional[set] = None,
    speculative_depth: Optional[int] = None,
    confidence_threshold: Optional[float] = None,
    verify_decode_phase: Optional[bool] = None,
    stats: Optional[DSparkDecodeStats] = None,
    forward: Optional[Callable[[mx.array, Any], tuple]] = None,
    token_callback: Optional[Callable[[List[int]], None]] = None,
    prefill_callback: Optional[Callable[[dict], None]] = None,
    completion_callback: Optional[Callable[[], None]] = None,
    abort_check: Optional[Callable[[], bool]] = None,
    divergence_capture: Optional["DivergenceCapture"] = None,
) -> List[int]:
    """Self-contained DSpark-direct decode over ``model`` (W23 drafter + V4.1
    target forward).  ``speculative_depth=0`` is pure AR (greedy argmax / sampled
    == :func:`mtplx.generation.generate_ar` under the same seed).  Fills ``stats``
    (created if omitted) with drafted/accepted/cycle counters and returns the
    generated token ids (excluding the prompt).

    The model must carry a DSpark head (``model.mtp`` built via the ``mtp=True``
    load path) and an all-trimmable V4.1 cache.

    ``prefill_callback`` (optional) fires exactly once right after the prompt
    prefill, before the decode cycles, with ``{"prompt_tokens",
    "prompt_eval_time_s"}`` -- the same payload the served :func:`generate_dspark`
    lane emits, so the ab harness's prefill->decode boundary snapshot works on
    either lane.

    ``completion_callback`` runs once before a successful return, while target
    and draft caches are still alive. Benchmark observers freeze their clock
    before reading memory here. Callback errors propagate so an invalid
    measurement cannot silently produce a successful receipt.
    """
    import time

    if getattr(model, "mtp", None) is None:
        raise RuntimeError("dspark_generate requires a model with a DSpark MTP head")

    def complete(ids):
        if completion_callback is not None:
            completion_callback()
        return ids

    if max_tokens <= 0:
        return complete([])
    stats = stats if stats is not None else DSparkDecodeStats()
    block_size = int(getattr(model.mtp, "block_size", 0) or 0)
    k_request = block_size if speculative_depth is None else int(speculative_depth)
    confidence_threshold = _confidence_threshold_from_env(confidence_threshold)
    rng = np.random.default_rng(seed)
    fwd = forward if forward is not None else _target_forward(model)

    cache = model.make_cache()
    mtp_caches = model.make_mtp_cache()

    prompt_arr = mx.array([[int(t) for t in prompt_ids]])
    _prefill_started = time.perf_counter()
    logits, main_hidden = fwd(prompt_arr, cache)
    mx.eval(logits, main_hidden)
    _fire_prefill_callback(
        prefill_callback, prompt_ids, time.perf_counter() - _prefill_started
    )
    # seed the DSpark windows with the whole prompt's main hiddens (ring keeps
    # the last window_size), exactly as the reference forward_spec start_pos==0.
    model.mtp.seed_main(main_hidden, mtp_caches)

    from mtplx.generation import _sample_from_logits

    primary, _ = _sample_from_logits(logits[0, -1], sampler, rng)
    primary = int(primary)
    main_h = main_hidden[:, -1:, :]

    tokens: List[int] = [primary]
    if token_callback is not None:
        token_callback([primary])
    stats.generated_tokens = 1
    if _is_stop(primary, stop_ids) or max_tokens <= 1:
        return complete(tokens[:max_tokens])

    rest, _finish = _decode_cycles(
        model=model,
        forward=fwd,
        cache=cache,
        mtp_caches=mtp_caches,
        primary=primary,
        main_h=main_h,
        max_tokens=max_tokens - 1,
        sampler=sampler,
        rng=rng,
        stop_ids=stop_ids,
        k_request=k_request,
        confidence_threshold=confidence_threshold,
        stats=stats,
        token_callback=token_callback,
        abort_check=abort_check,
        verify_decode_phase=(
            _verify_decode_phase_enabled()
            if verify_decode_phase is None
            else bool(verify_decode_phase)
        ),
        divergence_capture=divergence_capture,
    )
    tokens.extend(rest)
    stats.generated_tokens = len(tokens)
    return complete(tokens)


# ---------------------------------------------------------------------------
# public: served lane (runtime GenerationOutput surface)
# ---------------------------------------------------------------------------
def generate_dspark(
    rt,
    prompt_ids: Sequence[int],
    *,
    max_tokens: int,
    sampler,
    seed: int = 0,
    stop_token_ids: Optional[set] = None,
    token_callback: Optional[Callable[[List[int]], None]] = None,
    speculative_depth: Optional[int] = None,
    confidence_threshold: Optional[float] = None,
    verify_decode_phase: Optional[bool] = None,
    trace_label: Optional[str] = None,
    trace_metadata: Optional[dict] = None,
    prefill_callback: Optional[Callable[[dict], None]] = None,
    abort_check: Optional[Callable[[], bool]] = None,
):
    """DSpark-DIRECT served decode: the lean lane behind ``--generation-mode
    dspark`` / ``MTPLX_DSV41_DSPARK_DIRECT=1`` on a deepseek_v41 MTP runtime.

    Prefills through ``rt.forward_ar`` (cold, no session bank -- this lane bypasses
    the generic MTP warm-prefix/history machinery on purpose), runs
    :func:`_decode_cycles`, streams committed tokens through ``token_callback``, and
    returns a :class:`mtplx.generation.GenerationOutput` carrying the accept stats
    (accepted/drafted/rejected, by-depth, verify_calls) the server's
    ``mtplx_openai_generation`` telemetry surfaces.
    """
    import time

    from mtplx.generation import (
        GenerationOutput,
        GenerationStats,
        _decode,
        _default_stop_tokens,
        _finish_reason_from_tokens,
        _sample_from_logits,
        _strip_terminal_stop,
    )

    if not bool(getattr(rt, "mtp_enabled", False)):
        raise RuntimeError("generate_dspark requires an MTP-enabled runtime")
    model = rt.model
    if getattr(model, "mtp", None) is None:
        raise RuntimeError("generate_dspark requires a model with a DSpark MTP head")
    stop_ids = (
        _default_stop_tokens(rt.tokenizer) if stop_token_ids is None else set(stop_token_ids)
    )
    block_size = int(getattr(model.mtp, "block_size", 0) or 0)
    requested = block_size if speculative_depth is None else int(speculative_depth)
    confidence_threshold = _confidence_threshold_from_env(confidence_threshold)
    rng = np.random.default_rng(seed)
    stats = DSparkDecodeStats()

    cache = model.make_cache()
    mtp_caches = model.make_mtp_cache()

    started_all = time.perf_counter()
    prefill_started = time.perf_counter()
    logits, main_hidden = rt.forward_ar(
        mx.array([[int(t) for t in prompt_ids]]),
        cache=cache,
        return_hidden=True,
        logits_keep=1,
    )
    mx.eval(logits, main_hidden)
    prompt_eval_time = time.perf_counter() - prefill_started
    _fire_prefill_callback(prefill_callback, prompt_ids, prompt_eval_time)
    model.mtp.seed_main(main_hidden, mtp_caches)

    def _fwd(ids: mx.array, c) -> tuple:
        return rt.forward_ar(ids, cache=c, return_hidden=True)

    primary, _ = _sample_from_logits(logits[0, -1], sampler, rng)
    primary = int(primary)
    main_h = main_hidden[:, -1:, :]
    tokens: List[int] = [primary]
    if token_callback is not None:
        token_callback([primary])
    stats.generated_tokens = 1

    decode_started = time.perf_counter()
    finish_reason = "stop"
    if not (_is_stop(primary, stop_ids) or max_tokens <= 1):
        # Arm K29/K30 for the verify cycles so the small-M verify uses the fused
        # decode attention + selected-key gather, not the prefill attention path.
        with arm_dspark_decode_kernels():
            rest, finish_reason = _decode_cycles(
                model=model,
                forward=_fwd,
                cache=cache,
                mtp_caches=mtp_caches,
                primary=primary,
                main_h=main_h,
                max_tokens=max_tokens - 1,
                sampler=sampler,
                rng=rng,
                stop_ids=stop_ids,
                k_request=requested,
                confidence_threshold=confidence_threshold,
                stats=stats,
                token_callback=token_callback,
                abort_check=abort_check,
                verify_decode_phase=(
                    _verify_decode_phase_enabled()
                    if verify_decode_phase is None
                    else bool(verify_decode_phase)
                ),
            )
        tokens.extend(rest)
    else:
        tokens = tokens[:max_tokens]
    decode_wall = time.perf_counter() - decode_started
    elapsed = time.perf_counter() - started_all
    stats.generated_tokens = len(tokens)

    finish = _finish_reason_from_tokens(
        tokens, stop_token_ids=stop_ids, max_tokens=max_tokens
    )
    decode_tok_s = (len(tokens) / decode_wall) if decode_wall > 0 else 0.0
    gen_stats = GenerationStats(
        mode="mtpk",
        generated_tokens=len(tokens),
        elapsed_s=elapsed,
        tok_s=(len(tokens) / elapsed) if elapsed > 0 else 0.0,
        decode_elapsed_s=decode_wall,
        decode_tok_s=decode_tok_s,
        end_to_end_tok_s=(len(tokens) / elapsed) if elapsed > 0 else 0.0,
        benchmark_mode="dspark_direct",
        runtime_mtp_enabled=True,
        prompt_eval_time_s=prompt_eval_time,
        new_prefill_tokens=len(prompt_ids),
        target_forward_time_s=prompt_eval_time,
        verify_calls=stats.verify_calls,
        accepted_drafts=stats.accepted_drafts,
        rejected_drafts=stats.rejected_drafts,
        drafted_tokens=stats.drafted_tokens,
        correction_tokens=stats.correction_tokens,
        bonus_tokens=stats.bonus_tokens,
        speculative_depth=stats.speculative_depth,
        requested_speculative_depth=requested,
        accepted_by_depth=list(stats.accepted_by_depth),
        drafted_by_depth=list(stats.drafted_by_depth),
        peak_memory_bytes=mx.get_peak_memory(),
        events=[{"lane": "dspark_direct", "dspark": stats.to_dict()}],
    )
    return GenerationOutput(
        tokens=tokens,
        text=_decode(rt.tokenizer, _strip_terminal_stop(tokens, stop_ids)),
        stats=gen_stats,
        final_state=None,
        finish_reason=finish,
    )
