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
    abort_check: Optional[Callable[[], bool]] = None,
) -> List[int]:
    """Self-contained DSpark-direct decode over ``model`` (W23 drafter + V4.1
    target forward).  ``speculative_depth=0`` is pure AR (greedy argmax / sampled
    == :func:`mtplx.generation.generate_ar` under the same seed).  Fills ``stats``
    (created if omitted) with drafted/accepted/cycle counters and returns the
    generated token ids (excluding the prompt).

    The model must carry a DSpark head (``model.mtp`` built via the ``mtp=True``
    load path) and an all-trimmable V4.1 cache.
    """
    if getattr(model, "mtp", None) is None:
        raise RuntimeError("dspark_generate requires a model with a DSpark MTP head")
    if max_tokens <= 0:
        return []
    stats = stats if stats is not None else DSparkDecodeStats()
    block_size = int(getattr(model.mtp, "block_size", 0) or 0)
    k_request = block_size if speculative_depth is None else int(speculative_depth)
    confidence_threshold = _confidence_threshold_from_env(confidence_threshold)
    rng = np.random.default_rng(seed)
    fwd = forward if forward is not None else _target_forward(model)

    cache = model.make_cache()
    mtp_caches = model.make_mtp_cache()

    prompt_arr = mx.array([[int(t) for t in prompt_ids]])
    logits, main_hidden = fwd(prompt_arr, cache)
    mx.eval(logits, main_hidden)
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
        return tokens[:max_tokens]

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
    )
    tokens.extend(rest)
    stats.generated_tokens = len(tokens)
    return tokens


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
    if prefill_callback is not None:
        try:
            prefill_callback({"prompt_tokens": len(prompt_ids), "prompt_eval_time_s": prompt_eval_time})
        except Exception:
            pass
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
