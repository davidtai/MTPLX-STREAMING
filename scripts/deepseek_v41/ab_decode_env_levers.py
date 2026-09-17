#!/usr/bin/env python3
"""W28/W30 A/B arms: DeepSeek-V4.1-Flash env-flag levers (K1 shared overlap, K16 layer-major prefill) with the host-sync census.

Kept separate from ab_decode_levers.py (W24), whose arms are ExpertStreamingConfig
overrides (I/O fanout, overlap reads, inflight bytes); this file's arms are model env flags.

The orchestrator runs this INSIDE ``scripts/deepseek_v41/gpu_window.sh`` (holding
the GPU flock, Qwen unloaded, memory-guarded).  ``control`` is the shipped path
(every lever OFF), byte-for-byte; ``shared_overlap`` arms
``MTPLX_DSV41_SHARED_OVERLAP=1`` -- W11's MoE hands its shared expert to the
streamed switch as ``shared_work`` (KERNEL_LEDGER K1), so the switch dispatches
it into the GPU-idle window of the per-layer ``mx.eval(indices)`` routing barrier
+ miss I/O instead of serialising it after the routed gather.  The lever is a
pure execution reorder, so decoded tokens must be byte-identical; a differing
token-id sha256 FAILS the arm.

This is the small self-contained arm the W28 task calls for (feat/deepseek-v41-w24
had not landed on the integration branch).  When W24's ``ab_decode_levers.py``
merges, fold the ``shared_overlap`` preset into its ``ARM_PRESETS`` (a one-line
env-threading addition) and delete this file.

Per arm it reports: prefill tok/s, TTFT, decode tok/s, wall, peak GB, the decoded
token ids + their sha256, and -- with ``--syncs`` -- the per-decoded-token host
sync census off the route-stage probe (``hot.eval_indices`` = the routing barrier
the ledger prices at ~40/token) plus any GPU-overlap telemetry the runtime
exposes.  No GPU work at import; ``--help`` is CPU-safe.

Reuses the proven cell harness in ``bench_standard_shape.py`` (loader, prompt
builder, MLX ops, memory + gather probes) so the numbers are apples-to-apples
with the standard-shape receipts.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import time
import types
from pathlib import Path

import numpy as np  # CPU-only (no mlx); safe for the --dry-run path

DEFAULT_MODEL = Path(
    "~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4"
).expanduser()
GIB = 1024 ** 3
DEFAULT_BOS_ID = 0

# W113: repo-root-relative path to the standard chat-templated 16K cell ids file
# (schema mtplx-server-cell-prompt-ids-v1, cell=sweep, target_tokens=16384,
# seed=20260829).  The cell-prompt guard defaults --prompt-ids-file to this and
# refuses to MEASURE a 16K cell on the raw builder without it (see the W113 block).
STANDARD_CELL16K_PROMPT_IDS = (
    "docs/deepseek-v41/receipts/gpu-windows/window-28b/ar-16k/"
    "prompt-ids-deepseek-v41.json"
)
# W113 LOW-a: the seed + expected PROMPT-ids sha the auto-default pins the standard
# cell to (sha of json.dumps(ids) over the sweep/16384/20260829 entry) so a
# swapped/edited file is refused rather than silently measured.
STANDARD_CELL16K_PROMPT_SEED = 20260829
STANDARD_CELL16K_PROMPT_SHA256 = (
    "1a45b35bae742fae0e26d4f40ee0dc1093a2038e5b514460a4f02e9e56d74565"
)

# W121: the MLX plan is derived from David's TOTAL box target, not a static forecast.
# The runtime (mtplx.expert_runtime.apply_mlx_memory_cap) sets the allocator/wired limit
# = target - baseline - host reserve and set_cache_limit(cache); the bench sizes the
# ENGINE budget (persistent expert slots) = allocator_limit - transient_band so the slots
# FILL the target while active + transient stays under the allocator limit.  The old
# budget-total forecast (total - system_used_at_start(vm_stat) - non_metal - kv_growth -
# safety - plan_overshoot) and its constants/flags were removed (W121); the cache limit
# and transient band default in the runtime (MTPLX_DSV41_MLX_CACHE_LIMIT_GIB=2,
# MTPLX_DSV41_TRANSIENT_BAND_GIB=10).


# --------------------------------------------------------------------------
# W118 review MEDIUM-2: allocator readback proof helpers.  Read on the MAIN thread,
# OUTSIDE the timed region, so they never perturb tok/s.  MLX-optional (a fake mx in
# tests): every getter returns None when the accessor is missing.
# --------------------------------------------------------------------------


def _mlx_getter(mx, name):
    """The first callable ``name`` on ``mx`` or ``mx.metal`` (accessor moved between
    the two across MLX versions), else None."""
    for owner in (mx, getattr(mx, "metal", None)):
        fn = getattr(owner, name, None)
        if callable(fn):
            return fn
    return None


def _mlx_call_int(mx, name):
    fn = _mlx_getter(mx, name)
    if fn is None:
        return None
    try:
        return int(fn())
    except Exception:  # pragma: no cover - defensive
        return None


def _device_max_working_set_bytes(mx):
    """``max_recommended_working_set_size`` from ``mx.metal.device_info()`` (bytes),
    or None -- the ceiling the allocator's gc_limit_ is min()'d against (0.95x)."""
    fn = _mlx_getter(mx, "device_info")
    if fn is None:
        return None
    try:
        info = fn()
    except Exception:  # pragma: no cover - defensive
        return None
    if not isinstance(info, dict):
        return None
    for key in (
        "max_recommended_working_set_size",
        "max_recommended_working_set",
        "recommended_max_working_set_size",
    ):
        val = info.get(key)
        if val:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    return None


def _mlx_headroom_readback_keys(
    mx, *, mlx_peak_bytes, active_start_bytes, active_end_bytes, cache_end_bytes
):
    """W118 review MEDIUM-2: the allocator readback proof keys merged into the receipt
    ``memory`` block.  ``mlx_limit_gib_readback`` = mx.get_memory_limit() (what
    apply_mlx_memory_cap actually set); ``mlx_gc_limit_gib_effective`` = min(readback,
    0.95 x device max_recommended_working_set_size) -- the gc_limit_ the allocator
    clears the cache against on a miss; ``mlx_peak_over_limit_gb`` = mlx_peak - readback
    (POSITIVE means the run went over the soft limit).  The thrash signature is
    ``mlx_active_gb_at_decode_end`` >= gc_limit - ~1 with ``mlx_cache_gb_at_decode_end``
    ~ 0 (the cache is being cleared on every miss)."""

    readback = _mlx_call_int(mx, "get_memory_limit")
    max_wss = _device_max_working_set_bytes(mx)
    gc_limit = None
    if readback is not None:
        gc_limit = readback if max_wss is None else min(readback, int(0.95 * max_wss))

    def _gib(b):
        return None if b is None else int(b) / GIB

    return {
        "mlx_limit_readback_bytes": readback,
        "mlx_gc_limit_effective_bytes": gc_limit,
        "mlx_peak_over_limit_bytes": (
            None
            if (readback is None or mlx_peak_bytes is None)
            else int(mlx_peak_bytes) - readback
        ),
        "mlx_limit_gib_readback": _gib(readback),
        "mlx_gc_limit_gib_effective": _gib(gc_limit),
        "mlx_active_gb_at_decode_start": _gib(active_start_bytes),
        "mlx_active_gb_at_decode_end": _gib(active_end_bytes),
        "mlx_cache_gb_at_decode_end": _gib(cache_end_bytes),
        "mlx_peak_over_limit_gb": (
            None
            if (readback is None or mlx_peak_bytes is None)
            else (int(mlx_peak_bytes) - readback) / GIB
        ),
    }


def _ab_memory_block(block: dict, readback: dict) -> dict:
    """Preserve sampled OS observations and explicit units in schema-v2 receipts."""
    out = dict(block)
    for name in ("process_footprint_peak", "system_used_peak", "mlx_peak"):
        value = block.get(name + "_bytes")
        out[name + "_gb"] = None if value is None else value / 1_000_000_000
        out[name + "_gib"] = None if value is None else value / GIB
    # The readback helper predates schema v2; its *_gb values were binary GiB.
    for name in ("mlx_active_gb_at_decode_start", "mlx_active_gb_at_decode_end",
                 "mlx_cache_gb_at_decode_end"):
        gib = readback.get(name)
        out[name.replace("_gb_", "_bytes_")] = None if gib is None else round(gib * GIB)
        out[name.replace("_gb_", "_gib_")] = gib
        out[name] = None if gib is None else gib * GIB / 1_000_000_000
    for name in (
        "mlx_limit_readback",
        "mlx_gc_limit_effective",
        "mlx_peak_over_limit",
    ):
        value = readback.get(name + "_bytes")
        out[name + "_bytes"] = value
        out[name + "_gb"] = None if value is None else value / 1_000_000_000
        out[name + "_gib"] = None if value is None else value / GIB
    # Compatibility aliases retained for older receipt readers. Both now name
    # the value their key claims: the allocator limit readback and the derived
    # GC threshold, respectively.
    out["mlx_limit_gib_readback"] = out["mlx_limit_readback_gib"]
    out["mlx_gc_limit_gib_readback"] = out["mlx_gc_limit_effective_gib"]
    out["mlx_limit_gib_effective"] = None
    return out


def _memory_cap_block(runtime):
    """W121 MEDIUM-6: the ``apply_mlx_memory_cap`` report, lifted into the receipt on
    EVERY path (explicit --memory-limit-gib as well as the box target) -- applied limit,
    wired_limit_applied / cache_limit_applied, slot_derivation, box-target components.
    Reads ``runtime.memory_cap_report``, else falls back to
    ``runtime.resource_telemetry_snapshot()['memory_cap']`` (same object).  ``None`` only
    when the runtime has neither (a stub runtime, or apply_memory_cap disabled)."""

    rep = getattr(runtime, "memory_cap_report", None)
    if isinstance(rep, dict):
        return rep
    snap = getattr(runtime, "resource_telemetry_snapshot", None)
    if callable(snap):
        try:
            mc = snap().get("memory_cap")
        except Exception:  # pragma: no cover - defensive
            mc = None
        if isinstance(mc, dict):
            return mc
    return None


def _apply_effective_limit(mem, cap) -> None:
    """Set ``mlx_limit_gib_effective`` to the limit ACTUALLY passed to set_memory_limit
    (MEDIUM-6 / window-49): the apply_mlx_memory_cap report's ``limit`` -- NOT the
    get_memory_limit readback, which the OS clamps to 0.95 * maxWorkingSet (window 49:
    passed 77.18, readback 70.18).  Falls back to the clamped readback only when no cap
    report is available at all, so the field is never silently wrong on any path."""

    if not isinstance(mem, dict):
        return
    if isinstance(cap, dict) and cap.get("limit") is not None:
        mem["mlx_limit_gib_effective"] = int(cap["limit"]) / GIB
    elif mem.get("mlx_limit_gib_effective") is None:
        # last-resort: the clamped readback (better than None when no cap report).
        mem["mlx_limit_gib_effective"] = mem.get(
            "mlx_limit_readback_gib",
            mem.get("mlx_gc_limit_gib_readback"),
        )


def _resolve_receipt_baseline_gb(args):
    """The macOS+agent baseline (decimal GB) for the receipt's box_used, on EVERY path.
    Window-49 fix: box_used was 0 on the explicit --memory-limit-gib path because the
    baseline was read ONLY from the target plan.  Prefer the target plan (when armed),
    else the env MTPLX_DSV41_BOX_BASELINE_GB that gpu_window.sh exports from its in-window
    USED_START, else --box-baseline-gb.  The baseline is retained for the explicitly
    labelled baseline-plus-process estimate; measured ``box_used_gb`` comes from
    ``vm_stat`` and already includes the process and file cache."""

    tp = getattr(args, "_dsv41_target_plan", None)
    if tp and tp.get("box_baseline_gb") is not None:
        return float(tp["box_baseline_gb"])
    raw = os.environ.get("MTPLX_DSV41_BOX_BASELINE_GB")
    if raw and str(raw).strip() and str(raw).strip().lower() != "default":
        try:
            return float(str(raw).strip())
        except ValueError:
            pass
    v = getattr(args, "box_baseline_gb", None)
    return None if v is None else float(v)


def _inject_box_used(mem, args) -> None:
    """Keep measured whole-machine usage separate from baseline+process estimate."""
    if not isinstance(mem, dict):
        return
    baseline_gb = _resolve_receipt_baseline_gb(args)
    fp = mem.get("process_footprint_peak_gb")
    mem["box_baseline_gb"] = baseline_gb
    mem["baseline_plus_process_peak_estimate_gb"] = (
        None if baseline_gb is None or fp is None else round(baseline_gb + fp, 4)
    )
    mem["box_used_gb"] = mem.get("system_used_peak_gb")
    mem["box_used_source"] = "sampled_vm_stat_including_file_cache"


# W77: AR top-1/top-2 logit gap (logit units) below which a greedy DSpark
# divergence is classed a tie-break flip rather than a genuine divergence.
# 3x the bf16-class per-logit floor (~1e-2, the W40/K21 HEAD_MODE=bf16 head-GEMV
# rounding measured by the W77 CPU per-lever probe) -- see
# mtplx.models.deepseek_v41_dspark_decode.DSPARK_TIE_MARGIN_DEFAULT and
# docs/deepseek-v41/W77_DSPARK_DIVERGENCE.md.  Duplicated here (not imported) so
# the parser stays CPU-safe / mlx-free for --dry-run.
DSPARK_TIE_MARGIN_DEFAULT = 3.0e-2
OVERLAP_ENV = "MTPLX_DSV41_SHARED_OVERLAP"
PROBE_ENV = "MTPLX_ROUTE_STAGE_PROBE"
STAGE_TIMING_ENV = "MTPLX_DSV41_STAGE_TIMING"
BARRIER_STAGE = "hot.eval_indices"

LAYER_MAJOR_ENV = "MTPLX_DSV41_PREFILL_LAYER_MAJOR"
SINKHORN_METAL_ENV = "MTPLX_DSV41_SINKHORN_METAL"   # K3, merged @ 8982b93c9
HC_COMPILE_ENV = "MTPLX_DSV41_HC_COMPILE"           # K4, landing
SMALL_STAGES_FUSED_ENV = "MTPLX_DSV41_SMALL_STAGES_FUSED"  # K35 (W91): collapse the
# whole per-layer small-stage set (both HC premix/combine incl. the Sinkhorn, the
# MoE gate+top-k, the shared expert, the MoE combine) into three compiled per-layer
# graphs separated only by the two un-fused calls (attention KV write, routed
# switch gather). Extends K4 (HC-only) + K22 (gate-prefix/combine-only) to fold in
# the gate top-k + shared expert + MoE combine. Byte-identical over its whole
# admitted range on CPU (cap = _SMALL_STAGES_MAX_ROWS = 7, the mx.compile bit-exact
# regime) but ROUNDING-CLASS ON GPU: window-37 measured NULL (2.13 vs 2.17 tok/s)
# with the token-id sha DIFFERING (n=1 mx.compile reassociates the fp32 GEMM/
# reductions on Metal, flipping a greedy near-tie; [[dsv41-inexact-ok-if-tie-flips]]).
# No measured GPU win -> kept OUT of every composite arm; only the isolation A/B arm
# small_stages_fused carries it. Engagement in receipt.small_stages_engagement.
HC_PREMIX_KERNEL_ENV = "MTPLX_DSV41_HC_PREMIX_KERNEL"  # K35 (W91): GPU-only fused
# HC-premix Sinkhorn kernel (folds the pre/post/comb split + affine + sigmoid into
# the K3 Sinkhorn dispatch). ROUNDING-CLASS (1e-6, argmax-exact), like K3; default
# off, inert on CPU, and NOT in any composite arm until a GPU parity receipt exists.
# Pinned by every _preset (force-unset) so a parent-shell export cannot silently arm
# the rounding-class kernel; the receipt's arm_env records it.
SWITCH_FASTPATH_ENV = "MTPLX_DSV41_SWITCH_FASTPATH"  # K23 (W42): defer the
# per-all-hit-layer switch fence + async-dispatch split waves (hy3's shipped
# deferred-release mechanism, promoted for the DSV4.1 lane whose config leaves it
# fenced). Removes the second per-layer device->host sync; byte-identical.
SWITCH_SUBMIT_ENV = "MTPLX_DSV41_SWITCH_SUBMIT"  # K23 variant B (W42 window-14):
# pure defer measured -13.4% -- the all-hit deferred branch submitted no GPU work,
# so the device idled until the next barrier (DSV4.1's backbone has no
# MTPLX_HY3_SUBMIT_CADENCE equivalent). Paired with the fast-path, this async_evals
# each all-hit wave output (non-blocking submit, keeps the GPU fed); byte-identical.
HEAD_MODE_ENV = "MTPLX_DSV41_HEAD_MODE"             # W40 / K21, output-head codec
ATTN_COMPILE_ENV = "MTPLX_DSV41_ATTN_COMPILE"       # K22, W41 landing
ATTN_WIN_MEMO_ENV = "MTPLX_DSV41_ATTN_WIN_MEMO"     # K24, W45: window-mask memo
DRAFT_COMPILE_ENV = "MTPLX_DSV41_DRAFT_COMPILE"     # K33 (W65): DSpark draft-block
# tape collapse -- replays the draft block's PURE chains (attention prep + Hyper-
# Connection prep reuse the K22/K4 tapes; the MoE gate-prefix/combine folds; the
# markov step; the confidence head) from mx.compile tapes instead of rebuilding the
# graph from Python each cycle.  Byte-identical (draft tokens/logits/confidence),
# fixed-shape + row-cap.  Only touches the DSpark-DIRECT draft path (--decode-mode
# dspark), so it composes with the decode levers on the target verify forward.
DRAFT_HEAD_BF16_ENV = "MTPLX_DSV41_DRAFT_HEAD_BF16"  # W103: DSpark draft-head
# fp32-cast trap removal -- cast the draft hidden to the resident bf16 head dtype
# (a bf16 GEMV, f32 logits after) instead of casting to f32 and promoting the head
# weight to a per-cycle f32 temporary. Draft-head only (composes with the verify
# forward); rounding-class on the draft logits (greedy verify == AR regardless).
VERIFY_SINGLE_BARRIER_ENV = "MTPLX_DSV41_VERIFY_SINGLE_BARRIER"  # K31 (W61):
# small-M (2..8-row) DECODE verify -- pin the whole route all-hit and gather
# rows*top_k in ONE wave (K27 sorted gather) with one deferred release, so a
# K+1 verify pays ONE routing barrier per layer instead of one per split wave
# (window 25: ~630 ms/verify). Byte-identical; DEFAULT ON in the code
# (os.environ.get(..., "1")), so the census/window sees it live; this arm pins it
# explicitly and a "0" baseline measures the delta.
DEVICE_ROUTE_ENV = "MTPLX_DSV41_DEVICE_ROUTE"  # K24 (W44): barrier-free all-hit
# route -- gather over a device expert->slot LUT WITHOUT mx.eval(indices); a cold
# miss is repaired by a per-layer span re-run so the token stays byte-identical.
# 40 -> m+1 host syncs/token (warm 1); cold pays one extra span of compute
# (W44_DEVICE_ROUTE.md). Byte-identical, but tracked separately from the pure
# per-forward reorders because of that cold-token recovery cost.
PIN_WORKING_SET_ENV = "MTPLX_DSV41_PIN_WORKING_SET"  # W64 (R3-pin): post-prefill
# pinned working set -- top-K resident experts/layer marked never-recyclable so
# a later device route can gather them barrier-free WITHOUT racing an LRU recycle
# (the W44 window-19 failure). "all"/"keys" pins the whole resident set (the
# ``pin_ws`` arm -> fully static layers); a fraction or slot count pins the top
# set and leaves a free tail for misses. Byte-identical (cache policy only); its
# effect is the all-pinned-hit-layer FRACTION reported under receipt
# ``pin_working_set`` (W64_PINNED_WORKING_SET.md).
PIN_REFRESH_TOKENS_ENV = "MTPLX_DSV41_PIN_REFRESH_TOKENS"  # W64: re-rank every N
# decode epochs (0/unset = pin once after prefill).
DEVICE_ROUTE_PINNED_ENV = "MTPLX_DSV41_DEVICE_ROUTE_PINNED"  # W71 (K24 revived):
# the barrier-free device route GUARDED BY the W64 pins -- gather over a PINNED-only
# expert->slot LUT, defer the all-pinned check, and recompute any not-all-pinned
# layer on the fenced path. Because a pinned slot never recycles on normal decode
# admission (W64), the deferred gather over pinned slots cannot race a recycle (the
# W44 window-19 failure); a force-evicted pin is caught by the flush + recomputed.
# Byte-identical to fenced; the ``device_route_pinned`` arm pairs it with pin_ws
# (pin all keys) so every all-hit layer is an all-pinned layer. Its effect is the
# barrier-free-layers-per-token in receipt ``device_route_pinned``
# (W71_DEVICE_ROUTE_PINNED.md).
PREFILL_DENSE_ENV = "MTPLX_DSV41_PREFILL_DENSE_EXPERTS"  # K26 (W51): prefill-only
# "dequantize once, matmul dense" expert path. At the 16K layer-major prefill the
# mxfp4 gs32 gather_qmm is ALU/dequant-bound (W47: ~2.9 TFLOPS); this dequantizes
# each expert with >= PREFILL_DENSE_MIN_ROWS routed rows to bf16 ONCE and runs
# gate/up/down as dense bf16 matmuls. Prefill-only (never engages at decode M=1);
# NOT bit-identical to gather (fp32 matmul accumulation order), within the W51 CPU
# tolerance. The two value knobs below tune the row threshold and dequant batch.
PREFILL_DENSE_MIN_ROWS_ENV = "MTPLX_DSV41_PREFILL_DENSE_MIN_ROWS"  # per-expert rows
# a wave must carry before it densifies (default ~128); below it keeps gather_qmm.
PREFILL_DENSE_BATCH_ENV = "MTPLX_DSV41_PREFILL_DENSE_BATCH"  # experts dequantized
# per bounded batch (default 8; ~71 MB bf16/expert -> ~0.57 GB transient peak).
PREFILL_DENSE_MATMUL_DTYPE_ENV = "MTPLX_DSV41_PREFILL_DENSE_MATMUL_DTYPE"  # W51
# window-20 A/B: "f32" runs the dense dequant + gate/up/down matmuls in float32
# (W50 saw bf16 score matmuls 34% slower than f32 at 16K -- the dense matmul may pay
# the same slow bf16 kernel); default bf16.
SCORE_DTYPE_ENV = "MTPLX_DSV41_PREFILL_SCORE_DTYPE"        # W50 / K25: prefill
# QK^T/PV matmul input dtype ("bf16" -> the bf16 matmul, f32 softmax; unset = f32,
# byte-identical).  LOSSY-by-design; window-20 measured it -34% on the GPU (the
# bf16<->f32 casts of the [rows,64,T] transient + a slow large-N bf16 matmul kernel
# outweigh any FLOP saving -- the score stage is pass/bandwidth-bound, NOT FLOP-bound).
SCORE_KEY_CHUNK_ENV = "MTPLX_DSV41_PREFILL_SCORE_KEY_CHUNK"  # W50 / K25: split-K
# online-softmax key-chunk width (caps the [rows,H,T] score transient at
# [rows,H,chunk]).  f32-exact up to reassociation; window-20 measured it -16% but
# -11 GB peak -- a PEAK-GB lever (the per-chunk O(rows*64*512) output rescale costs
# time), not a throughput lever.
LAYOUT_FIX_ENV = "MTPLX_DSV41_LAYOUT_FIX"          # W56 / K27 F1: sorted routed gather
DOWN_K_PAD_ENV = "MTPLX_DSV41_DOWN_K_PAD"          # W56 / K27 F2: down-proj K pad to 2560
SCORE_PATH_ENV = "MTPLX_DSV41_PREFILL_SCORE_PATH"          # W50: prefill score
# implementation -- "lean" is the f32 pass-cut one-shot (scale q once instead of the
# T-wide scores; fold the value-0 sink into the denom, no concat/slice), cutting
# three passes over the [rows,64,T] transient.  Reassociation-level vs control
# (greedy-identical); the throughput play once window-20 showed the stage is
# pass/bandwidth-bound.  Unset = "oneshot" (byte-identical).
SELECTED_KEYS_ENV = "MTPLX_DSV41_SELECTED_KEYS"    # W59 / K30: selected-key gather --
# gather only the window + index_topk selected compressed rows each query attends
# into a compact [rows, k, 512] operand and score over k (~640) keys, vs the shipped
# masked-full [rows,64,T] score that grows with T.  Reassociation-level vs control
# (greedy-identical); the score-stage FLOP/byte play (24x FLOPs, 50x score bytes at
# 16K prefill; 46x decode KV read at 16K).  Prefill + decode + verify.  Composes with
# K29 (decode kernel consumes the gathered-k operands).  Unset = masked-full.
SOFTMAX_KERNEL_ENV = "MTPLX_DSV41_PREFILL_SOFTMAX_KERNEL"  # W58 / K28: fuse the
# prefill mask + per-head value-0 sink + f32 softmax over the [rows,64,T] transient
# into ONE mx.fast.metal_kernel dispatch (2 device reads + 1 write, no T-wide
# masked_scores/ex/concat intermediate).  Prefill + one-shot only, GPU-only (CPU
# falls back to eager).  Reassociation-level vs control (greedy-identical, <=1e-6),
# NOT byte-identical.  Composes with the lean path (see prefill_lean_k28).
DECODE_ATTN_KERNEL_ENV = "MTPLX_DSV41_DECODE_ATTN_KERNEL"  # W60 / K29: fuse the
# M=1 decode / K+1 verify MLA attention step -- QK^T score + CSA/causal mask +
# per-head value-0 sink + f32 softmax + PV -- into ONE mx.fast.metal_kernel dispatch
# per layer, online-softmax over key tiles (no [64,T] score row).  Decode + small-M
# verify only (prefill untouched), GPU-only (CPU falls back to the eager one-shot).
# Reassociation-level vs control (greedy-identical, <=1e-6), NOT byte-identical.
# LEFT OUT of stack_a until the MTPLX_GPU_PARITY window is clean (W60 report).
MLX_MAX_MB_PER_BUFFER_ENV = "MLX_MAX_MB_PER_BUFFER"  # K14 (W63): MLX command-buffer
# byte cap -- caps MB per Metal command buffer, changing commit granularity and
# host-encode/GPU overlap (and prefill peak). An MLX passthrough (NOT an MTPLX
# lever): MLX reads it once at Metal init, so a genuine A/B runs ONE arm per
# process with the value exported before launch (the a3b K14 verdict was ~dead:
# 200 -> -0.9%, but the Qwen lane measured +1.6% at 500 MB). The arm pins the
# value and the receipt records it so a re-falsifier in the DSV4.1 streaming
# regime is reproducible.
KV_CHUNK_GROW_ENV = "MTPLX_DSV41_KV_CHUNK_GROW"  # W73 / K32: chunk-grown KV append.
# The phase-1 cache re-concatenates the WHOLE window / compressed-KV / index-key
# store on every token (O(current-length) copy per layer per token); at T=16384 the
# window append alone is ~2 ms/layer on the CPU double (~40x its 1K cost) and, across
# 40 layers, the dominant per-token O(T) work once K30 selected keys has already
# bounded the attention score.  This arms a geometric-capacity buffer + logical
# length + a donated mx.slice_update in-place write -> amortized O(new-rows) per token
# (the copy-everything resize fires only on the O(log T) doublings).  BYTE-IDENTICAL
# (the buf[:, :length] view equals the concatenated store).  A decode-shape lever
# (fixes the 16K decode append; prefill grows in bulk anyway).
SELECT_FENCE_ENV = "MTPLX_DSV41_SELECT_FENCE"  # W76: fence K30 selected_idx argsort.
# The K30 selected-key publication `shared.selected_idx = _mask_to_topk_idx(mask, ...)`
# is an argsort over n_comp (an index-source-layer O(T) cost). The shipped select
# bracket fences only `mask`, so the argsort stays lazy and is forced later by the
# first downstream compress-gather in the `score` stage -- mis-charging that O(T)
# select cost into decode attention-*proper* (W76 stage-timing artifact). This arms
# fencing selected_idx into `attn.<mode>.select`, so the argsort is timed where it
# belongs. A stage-timing ATTRIBUTION lever: byte-identical token/cache/logits (the
# fenced array is the same object the gather forces; the extra fence is a no-op in
# production / prefill), so it changes only the decode census, never the output.
WINDOW_RING_ENV = "MTPLX_DSV41_WINDOW_RING"  # W80 / K34: bounded window ring.
# The window store is a genuine sliding window of window_size (128); the decode
# gather / SWA mask only ever read the last 128 rows, yet the phase-1 store keeps
# FULL history (~0.7 GB at 16K across 40 layers).  W76/W78 pinned that resident
# churn as the memory-pressure amplifier that inflates the whole 16K decode step.
# This arms a BOUNDED ring of window_size + max_verify + slack (~136) rows in fixed
# ping-pong buffers (a ~128x cut), plus preallocated compress/index stores (the
# indexer needs them in full, so they can't be bounded, but the per-token realloc
# is removed).  A logical drop_offset addresses the same absolute positions, and
# dropped rows are always beyond the causal window, so the SELECTED-key decode path
# (cell16k) is BYTE-IDENTICAL; the masked-full path is reassociation-level (the
# score-reduction width shrinks -- greedy-identical, max|d|~1e-6, in-family with the
# other DSV4.1 score levers).  A DECODE-shape lever (prefill fills it chunk-wise).
WINDOW_RING_MAX_VERIFY_ENV = "MTPLX_DSV41_WINDOW_RING_MAX_VERIFY"  # widest verify block
WINDOW_RING_SLACK_ENV = "MTPLX_DSV41_WINDOW_RING_SLACK"            # safety margin
WINDOW_RING_HEADROOM_ENV = "MTPLX_DSV41_WINDOW_RING_HEADROOM"      # appends per compaction
WINDOW_RING_MAXKV_ENV = "MTPLX_DSV41_WINDOW_RING_MAXKV"            # compress/index prealloc
ATTN_SHAPE_STABLE_ENV = "MTPLX_DSV41_ATTN_SHAPE_STABLE"  # W90 / K36: shared selected-
# compress gather -- a DISPATCH-COUNT cleanup, NOT the in-situ floor fix.  All
# Reuse/Reindex/Full layers of a group read the SAME (compress_kv, selected_idx)
# pair, so the shipped K30 path issues the compressed-lane gather (~3 tiny host
# dispatches) once per layer; this gathers it ONCE per source and shares the bounded
# [b,s,k,hd] result, so only the first layer of a group issues it (est. ~38 -> ~8
# compress gathers/token on the real backbone; tiny config 6 -> 3).  Expected <= ~1%
# of the token.  BYTE-IDENTICAL (a pure caching of the deterministic K30 gather; same
# rows, same order) -- the byte-identity summary must show it clean.  Small-M gated
# (decode/verify only; a prefill-chunk cache would pin ~17 GB at 16K).  Composes with
# the ring (cell16k_ring_stable).
#
# NOT the fix for the mode-independent ~4 ms/layer in-situ decode floor.  The earlier
# "O(T) source reference" claim is FALSIFIED: window-34/w78-in-model.json shows
# attn.swa_only (no compress_kv) at 7.447 ms/layer vs attn.reuse 6.960 -- swa costs
# MORE while referencing no compressed store -- and the isolated bench references O(T)
# per dispatch yet is flat.  macmon on a live 16K decode read 71 C (NOT thermal),
# ~45% GPU-busy (gpu_usage_ratio), freq swinging 580-1381 MHz: the GPU DVFS-downclocks
# in the gaps between B=1 bursts.  The decisive per-mode control is swa_only; see
# docs/deepseek-v41/W90_ATTN_IN_SITU.md and the --utilization telemetry.
# W87: merge each layer's persistent + transient slot tiers into ONE scan-resistant
# resident pool (prefill AND decode misses admitted, probationary at the eviction
# end, promoted on a later hit).  Memory is UNCHANGED (allocation identical); it
# warms the pool during the 16K prefill so decode starts on the prompt tail instead
# of cold, and widens the verify single-fence capacity to persistent+transient.
# A pure cache change -> BYTE-IDENTICAL tokens/logits to cell16k_ring; the byte-
# identity summary must show it clean.  Composes with the window ring (independent).
SINGLE_SLOT_POOL_ENV = "MTPLX_DSV41_SINGLE_SLOT_POOL"
GATE_PREFETCH_ENV = "MTPLX_DSV41_GATE_PREFETCH"  # W93: gate-oracle one-ahead prefetch width k
GATE_PREFETCH_MIN_LAYER_ENV = "MTPLX_DSV41_GATE_PREFETCH_MIN_LAYER"  # W93: skip targets below this
# W95: the single v2 runner switch -- ONE key composes the W93 gate-oracle prefetch +
# the W87 single scan-resistant pool (no stacked sub-keys; the user sets only this).
# Byte-identical to control's CLASS (residency-only: prefetch warms the cache on the
# TRUE route, the pool only changes which loads happen). See W95_RUNNER_DESIGN.md.
RUNNER_ENV = "MTPLX_DSV41_RUNNER"  # W95: "v2" = the composed SSD-hiding runner
# W107: the master bounded-KV switch (David: "controlling kv growth is crucial for
# everything").  ON => EVERY KV lane is bounded/preallocated to max_kv at prefill and
# written in place (O(new rows)/token, no per-token concatenate/realloc): the W80
# window ring + preallocated compress/index + the PREALLOCATED compressor frontier
# (the "main latent KV", the one lane W80 left growing with a per-token _grow == O(T^2)
# over the cell). CPU parity is covered; Metal parity remains unvalidated after
# differing recorded greedy outputs. OFF by default, and the shared runtime-open
# boundary rejects full-model use. Named bounded presets remain for dry-run/config
# inspection; isolated cache classes support the numerical investigation. MAXKV is
# stamped from the resolved cell max_kv in _run_arm (falls back to WINDOW_RING_MAXKV).
KV_BOUNDED_ENV = "MTPLX_DSV41_KV_BOUNDED"
KV_BOUNDED_MAXKV_ENV = "MTPLX_DSV41_KV_BOUNDED_MAXKV"
# W107 round-4: the round-3 MTPLX_DSV41_KV_INPLACE_WRITE lever was REMOVED -- the
# re-review proved mx.slice_update already donates in the cache's rebind pattern, so
# the in-place __setitem__ switch bought nothing and was unsafe (view() identity). The
# append primitive is fixed at mx.slice_update; there is no write-primitive lever.
# W110 (BENCH-ONLY DIAGNOSTIC -- not a perf lever): gate the per-record sha256
# re-check on the DECODE/verify streaming path. Decode-path hashing has been OFF on
# every ab/bench path (arg default False) and OFF in the served profile
# (deepseek-v41-mxfp4-75: verify_record_hashes=false), so there is nothing to REMOVE.
# The salvaged value runs the OTHER direction: "1" turns hashing ON to MEASURE its
# io-thread cost (arm cell16k_ring_v2_hash), with the records_hashed / hash_thread_ns
# counters. Honoured only by the loader/bench builder (build_streaming_config); the
# served profile builder (expert_profiles.build_expert_streaming_config) deliberately
# does NOT read it, so this env is NOT registered as a served lever. See W110 doc.
VERIFY_RECORD_HASHES_ENV = "MTPLX_DSV41_VERIFY_RECORD_HASHES"

# W97: cache the dequantized grouped o-LoRA (wo_a) weight per layer instead of
# re-issuing mx.dequantize(wo_a) every decode token (the released wo_a is 8x1024x4096
# = 33.55M params).  mx.dequantize returns bf16 for BOTH codecs (67 MB); _o_lora_down
# then promotes it to a fresh 134 MB f32 array per token per layer.  The lever caches
# that f32 promotion once (bf16->f32 is lossless), so the reference dequantizes it
# ONCE at convert (docs/deepseek-v41/W97_ATTENTION_291MS.md).  BYTE-IDENTICAL (the
# cached f32 array is the exact promotion of the dequantize output; the einsum's
# per-token .astype(f32) becomes a no-op) -> the byte-identity summary must show it
# clean.  Read at use (never import-frozen), so it works regardless of the lazy dsv41
# import.  Holds a dense f32 wo_a copy resident per layer (40 x 134 MB ~= 5.4 GB for
# BOTH codecs), so it is opt-in AND priced into the memory plan (deepseek_v41_loader).
WO_A_CACHE_ENV = "MTPLX_DSV41_ATTN_WO_A_CACHE"

# W97: fixed-shape mx.compile of the decode-attention CORE (QK + mask + sink softmax
# + PV over the gathered [b,s,k,hd] operand) -- one geometry-keyed tape at decode/
# small-M verify, the scattered elementwise fused (~13 -> ~8 kernels; the gather
# stays outside; docs/deepseek-v41/W97_ATTENTION_291MS.md).  ROUNDING-CLASS, NOT
# byte-identical: the n=1 compile reassociates the fp32 einsum/reductions (the K35
# lesson; measured max|Δ| ~9e-10 on CPU), so its arms are flagged in the byte-
# identity summary and gated separately from the exact levers.  The K29 fused decode
# kernel (DECODE_ATTN_KERNEL) collapses the SAME core to ONE dispatch on the GPU and
# wins the early return before this path -- it is the lower-dispatch option; this is
# the portable (CPU+GPU) fallback / A-B.  Read at use (never import-frozen).
ATTN_CORE_COMPILE_ENV = "MTPLX_DSV41_ATTN_CORE_COMPILE"

# W99: lean the decode-attention casts -- BYTE-IDENTICAL removal of the two genuinely
# redundant f32 casts (the eager core casts KVg to f32 twice -> once; the per-head
# sink is re-cast per token -> cached) plus the per-token numpy->device relift of the
# layer's RoPE inv_freq (cached).  The bulk of the casts are reference f32 numerics
# and the concatenates are structural RoPE (interleave + head-rejoin) -- NOT reducible
# byte-identically (docs/deepseek-v41/W97_ATTENTION_291MS.md §8).  Read at use; OFF by
# default.  Composes with the wo_a cache as the byte-identical cell16k_ring_lean stack.
ATTN_LEAN_CASTS_ENV = "MTPLX_DSV41_ATTN_LEAN_CASTS"

# W101 / K36: fused decode/verify attention PROJECTION-CHAIN kernels -- the qkv/out
# rmsnorm + interleaved-RoPE + head/group layout GLUE between the (kept) quantized
# matmuls and the (kept) grouped o-LoRA einsum, each fused into ONE metal_kernel.
# ROUNDING-CLASS (fused rmsnorm reassociates the fp32 sum; the o-LoRA einsum reads
# bf16 wo_a), GPU-only, small-M.  Independent of K29 (the SEPARATE core); composes
# with the wo_a cache + lean casts (docs/deepseek-v41/W101_ATTN_FUSED_PROJ.md).
ATTN_FUSED_PROJ_ENV = "MTPLX_DSV41_ATTN_FUSED_PROJ"
# Packed MXFP8 target ``wo_a`` route, installed once after strict resident load.
ATTN_WO_A_DIRECT_ENV = "MTPLX_DSV41_ATTN_WO_A_DIRECT"

# W115: the DSpark lane's per-verify K29 knob.  ``_run_arm`` (and the served
# ``arm_dspark_decode_kernels``) ``os.environ.setdefault`` K29 (DECODE_ATTN_KERNEL) +
# K30 (SELECTED_KEYS) to "1" for EVERY ``--decode-mode dspark`` arm, so the K+1 verify
# runs the fused decode-attention CORE by default; ``MTPLX_DSV41_DSPARK_VERIFY_K29=0``
# drops K29 from that setdefault (K30 stays), leaving the eager per-row gathered core.
# This is the honest A/B knob (W60: K29 is itself -38% vs eager at M=1) -- pin it "0"
# AND DECODE_ATTN_KERNEL="0" on an eager arm so the setdefault cannot re-arm it.
DSPARK_VERIFY_K29_ENV = "MTPLX_DSV41_DSPARK_VERIFY_K29"

# W118 / H7: raise ONLY the MLX allocator soft limit (mx.set_memory_limit) by N GiB
# ABOVE the residency plan, WITHOUT changing what is resident or the expert-cache slot
# plan -- so bytes/routing/outputs are byte-identical.  Read at use by
# expert_runtime.apply_mlx_memory_cap (the served ExpertStreamingRuntime.open path);
# the value is a GiB count ("8"), not a boolean.  Every real window runs the model OVER
# its own MLX limit (plan 69.2 -> mlx_peak 74.3), and MLX 0.32.2 treats set_memory_limit
# as soft: over-limit allocations take the cache-release / scheduler-wait path (the
# allocator-pressure regime).  docs/deepseek-v41/W118_MLX_LIMIT_HEADROOM.md.
MLX_LIMIT_HEADROOM_ENV = "MTPLX_DSV41_MLX_LIMIT_HEADROOM_GIB"

# Every lever env key, in a stable order. Each preset names ALL of them (None =
# force-unset) so applying an arm fully determines the flags regardless of what a
# prior arm in the same process left set -- the arms are independent. The eight
# boolean levers (OVERLAP/LAYER_MAJOR/SINKHORN_METAL/HC_COMPILE/SWITCH_FASTPATH/
# SWITCH_SUBMIT/ATTN_COMPILE/ATTN_WIN_MEMO) are per-forward, byte-identical
# execution reorders; DEVICE_ROUTE is byte-identical too but carries a cold-token
# recovery cost; HEAD_MODE is a LOAD-TIME codec taking a value
# ("bf16"/"mxfp8"/"q8") whose arms are lossy-by-design (bf16 rounding / 8-bit
# weight), so the head-* arms are NOT byte-identical to control -- the final
# byte-identity summary flags them (expected, cf. W40_HEAD_LEVER.md).
ALL_LEVER_ENVS = (
    OVERLAP_ENV,
    LAYER_MAJOR_ENV,
    SINKHORN_METAL_ENV,
    HC_COMPILE_ENV,
    SWITCH_FASTPATH_ENV,
    SWITCH_SUBMIT_ENV,
    ATTN_COMPILE_ENV,
    ATTN_WIN_MEMO_ENV,
    DRAFT_COMPILE_ENV,
    DEVICE_ROUTE_ENV,
    VERIFY_SINGLE_BARRIER_ENV,
    PREFILL_DENSE_ENV,
    PREFILL_DENSE_MIN_ROWS_ENV,
    PREFILL_DENSE_BATCH_ENV,
    PREFILL_DENSE_MATMUL_DTYPE_ENV,
    HEAD_MODE_ENV,
    SCORE_DTYPE_ENV,
    SCORE_KEY_CHUNK_ENV,
    SCORE_PATH_ENV,
    SOFTMAX_KERNEL_ENV,
    DECODE_ATTN_KERNEL_ENV,
    LAYOUT_FIX_ENV,
    DOWN_K_PAD_ENV,
    SELECTED_KEYS_ENV,
    PIN_WORKING_SET_ENV,
    PIN_REFRESH_TOKENS_ENV,
    MLX_MAX_MB_PER_BUFFER_ENV,
    DEVICE_ROUTE_PINNED_ENV,
    KV_CHUNK_GROW_ENV,
    SELECT_FENCE_ENV,
    WINDOW_RING_ENV,
    WINDOW_RING_MAX_VERIFY_ENV,
    WINDOW_RING_SLACK_ENV,
    WINDOW_RING_HEADROOM_ENV,
    WINDOW_RING_MAXKV_ENV,
    ATTN_SHAPE_STABLE_ENV,
    # W91 / K35 (appended; coordinate with any concurrent list extension):
    SMALL_STAGES_FUSED_ENV,
    HC_PREMIX_KERNEL_ENV,
    SINGLE_SLOT_POOL_ENV,
    GATE_PREFETCH_ENV,
    GATE_PREFETCH_MIN_LAYER_ENV,
    # W95 (appended; coordinate with any concurrent list extension):
    RUNNER_ENV,
    # W104 (appended; coordinate with any concurrent list extension):
    DRAFT_HEAD_BF16_ENV,
    # W107 (appended; coordinate with any concurrent list extension):
    KV_BOUNDED_ENV,
    KV_BOUNDED_MAXKV_ENV,
    # W97 (appended; coordinate with any concurrent list extension):
    WO_A_CACHE_ENV,
    ATTN_CORE_COMPILE_ENV,
    # W99 (appended):
    ATTN_LEAN_CASTS_ENV,
    # W101 (appended):
    ATTN_FUSED_PROJ_ENV,
    ATTN_WO_A_DIRECT_ENV,
    # W118 (appended; coordinate with any concurrent list extension): the MLX
    # allocator-limit headroom lever (GiB above the plan; residency-only, byte-identical).
    MLX_LIMIT_HEADROOM_ENV,
    # NOTE (W107 round-4): MTPLX_DSV41_KV_INPLACE_WRITE was DE-REGISTERED (the round-3
    # in-place write was reverted to slice_update + a donation gate), so it is no longer
    # in this list -- the served-log snapshot dropped it too (superset invariant holds).
    # W115 (appended; coordinate with any concurrent list extension): the DSpark
    # per-verify K29 knob (governs the setdefault that arms the verify decode core).
    DSPARK_VERIFY_K29_ENV,
    # NOTE: VERIFY_RECORD_HASHES_ENV is DELIBERATELY NOT in this list. It is a
    # BENCH-ONLY diagnostic env (honoured on the loader/bench builder, NOT on the
    # served profile builder) -- keeping it out of ALL_LEVER_ENVS also keeps it out
    # of the served-log lever snapshot (openai._DSV41_LEVER_ENV_KEYS, which must be a
    # superset), where it would be a DEAD served lever. See W110 doc + _preset.
)


def _preset(
    *, overlap=None, layer_major=None, sinkhorn=None, hc=None, small_stages=None,
    hc_premix_kernel=None, fastpath=None,
    submit=None, attn=None, win_memo=None, draft=None, draft_head_bf16=None,
    device_route=None,
    verify_single=None,
    prefill_dense=None, prefill_dense_min_rows=None, prefill_dense_batch=None,
    prefill_dense_matmul_dtype=None,
    head=None, score_dtype=None, score_key_chunk=None, score_path=None,
    layout_fix=None, down_k_pad=None, selected_keys=None,
    softmax_kernel=None, decode_attn_kernel=None, mlx_max_mb_per_buffer=None,
    kv_chunk_grow=None, select_fence=None,
    window_ring=None, window_ring_max_verify=None, window_ring_slack=None,
    window_ring_headroom=None, window_ring_maxkv=None,
    attn_shape_stable=None,
    pin_working_set=None, pin_refresh=None, device_route_pinned=None,
    single_slot_pool=None,
    gate_prefetch=None,
    gate_prefetch_min_layer=None,
    runner=None,
    kv_bounded=None, kv_bounded_maxkv=None,
    wo_a_cache=None, attn_core_compile=None,
    attn_lean_casts=None, attn_fused_proj=None, attn_wo_a_direct=None,
    dspark_verify_k29=None,
    verify_record_hashes=None,
    mlx_limit_headroom=None,
) -> dict:
    """A preset that pins EVERY lever key (None = force-unset). ``head`` takes a
    codec value ("bf16"/"mxfp8"/"q8"), ``prefill_dense_matmul_dtype`` takes
    "f32"/"bf16", ``prefill_dense_min_rows`` / ``_batch`` an integer string (None =
    use the code default), ``score_dtype`` a "bf16" value (W50/K25, prefill score
    matmul dtype), ``score_key_chunk`` a positive-int string (W50/K25 split-K chunk
    width), ``score_path`` a "lean" value (W50 f32 pass-cut one-shot),
    ``softmax_kernel`` a "1"/None boolean (W58/K28, the fused mask+sink+softmax
    Metal kernel), ``decode_attn_kernel`` a "1"/None boolean (W60/K29, the fused
    decode/verify MLA attention Metal kernel), ``mlx_max_mb_per_buffer`` a
    positive-int string (K14/W63, the MLX command-buffer MB cap passthrough),
    ``kv_chunk_grow`` a "1"/None boolean (W73/K32, the chunk-grown KV append
    backing), ``select_fence`` a "1"/None boolean (W76, fence the K30 selected_idx
    argsort into the select decode sub-stage -- a stage-timing attribution lever);
    the rest a "1"/None boolean."""
    return {
        OVERLAP_ENV: overlap,
        LAYER_MAJOR_ENV: layer_major,
        SINKHORN_METAL_ENV: sinkhorn,
        HC_COMPILE_ENV: hc,
        SMALL_STAGES_FUSED_ENV: small_stages,
        HC_PREMIX_KERNEL_ENV: hc_premix_kernel,
        SWITCH_FASTPATH_ENV: fastpath,
        SWITCH_SUBMIT_ENV: submit,
        ATTN_COMPILE_ENV: attn,
        ATTN_WIN_MEMO_ENV: win_memo,
        DRAFT_COMPILE_ENV: draft,
        DRAFT_HEAD_BF16_ENV: draft_head_bf16,
        DEVICE_ROUTE_ENV: device_route,
        VERIFY_SINGLE_BARRIER_ENV: verify_single,
        PREFILL_DENSE_ENV: prefill_dense,
        PREFILL_DENSE_MIN_ROWS_ENV: prefill_dense_min_rows,
        PREFILL_DENSE_BATCH_ENV: prefill_dense_batch,
        PREFILL_DENSE_MATMUL_DTYPE_ENV: prefill_dense_matmul_dtype,
        HEAD_MODE_ENV: head,
        SCORE_DTYPE_ENV: score_dtype,
        SCORE_KEY_CHUNK_ENV: score_key_chunk,
        SCORE_PATH_ENV: score_path,
        SOFTMAX_KERNEL_ENV: softmax_kernel,
        DECODE_ATTN_KERNEL_ENV: decode_attn_kernel,
        LAYOUT_FIX_ENV: layout_fix,
        DOWN_K_PAD_ENV: down_k_pad,
        SELECTED_KEYS_ENV: selected_keys,
        PIN_WORKING_SET_ENV: pin_working_set,
        PIN_REFRESH_TOKENS_ENV: pin_refresh,
        MLX_MAX_MB_PER_BUFFER_ENV: mlx_max_mb_per_buffer,
        DEVICE_ROUTE_PINNED_ENV: device_route_pinned,
        KV_CHUNK_GROW_ENV: kv_chunk_grow,
        SELECT_FENCE_ENV: select_fence,
        WINDOW_RING_ENV: window_ring,
        WINDOW_RING_MAX_VERIFY_ENV: window_ring_max_verify,
        WINDOW_RING_SLACK_ENV: window_ring_slack,
        WINDOW_RING_HEADROOM_ENV: window_ring_headroom,
        WINDOW_RING_MAXKV_ENV: window_ring_maxkv,
        ATTN_SHAPE_STABLE_ENV: attn_shape_stable,
        SINGLE_SLOT_POOL_ENV: single_slot_pool,
        GATE_PREFETCH_ENV: gate_prefetch,
        GATE_PREFETCH_MIN_LAYER_ENV: gate_prefetch_min_layer,
        RUNNER_ENV: runner,
        KV_BOUNDED_ENV: kv_bounded,
        KV_BOUNDED_MAXKV_ENV: kv_bounded_maxkv,
        WO_A_CACHE_ENV: wo_a_cache,
        ATTN_CORE_COMPILE_ENV: attn_core_compile,
        ATTN_LEAN_CASTS_ENV: attn_lean_casts,
        ATTN_FUSED_PROJ_ENV: attn_fused_proj,
        ATTN_WO_A_DIRECT_ENV: attn_wo_a_direct,
        DSPARK_VERIFY_K29_ENV: dspark_verify_k29,
        VERIFY_RECORD_HASHES_ENV: verify_record_hashes,
        MLX_LIMIT_HEADROOM_ENV: mlx_limit_headroom,
    }


ARM_PRESETS = {
    "control": _preset(),                                    # shipped: all levers OFF
    "shared_overlap": _preset(overlap="1"),                 # W28 K1 lever ON
    "layer_major": _preset(layer_major="1"),                # W30 K16 lever ON
    "sinkhorn_metal": _preset(sinkhorn="1"),                # K3 lever ON (8982b93c9)
    "hc_compile": _preset(hc="1"),                          # K4 lever ON (landing)
    "switch_fastpath": _preset(fastpath="1"),               # W42 K23: pure defer (−13.4% @ w14)
    "switch_fastpath_b": _preset(fastpath="1", submit="1"),  # W42 K23 var B: defer + async submit
    "attn_compile": _preset(attn="1"),                      # K22 lever ON (W41 landing)
    "attn_win_memo": _preset(win_memo="1"),                 # K24 lever ON (W45): window-mask memo
    "draft_compile": _preset(draft="1"),                    # K33 lever ON (W65): DSpark draft-block tape collapse (--decode-mode dspark)
    "device_route": _preset(device_route="1"),              # W44 K24: barrier-free all-hit
    # W64 R3-pin: pin the WHOLE resident set per layer (pin_working_set="all") ->
    # every layer fully static (pinned_static). The all-pinned-hit rate then
    # equals the all-hit rate (every hit route is a pinned route), so the receipt
    # ``pin_working_set.all_pinned_hit_rate`` reads out the layer fraction a
    # barrier-free device route could take race-free. Byte-identical (cache policy
    # only) -- decode ids must match control; only residency trajectory differs.
    "pin_ws": _preset(pin_working_set="all"),
    # W71 (K24 revived): the barrier-free device route guarded by the W64 pins.
    # Pins the WHOLE resident set per layer (pin_working_set="all" -> every layer
    # fully static) and arms the pinned device path + device route, so every
    # all-hit layer is an all-pinned layer taken barrier-free at zero recycle risk
    # (device_route="1" arms the backbone recovery; device_route_pinned="1" selects
    # the pinned-only LUT + all-pinned deferred check). Byte-identical to control:
    # the decode ids must match; only the residency trajectory + host-sync count
    # differ. ``device_route_pinned.barrier_free_layers_per_flush`` reads the win.
    "device_route_pinned": _preset(
        pin_working_set="all", device_route="1", device_route_pinned="1"
    ),
    "verify_single_barrier": _preset(verify_single="1"),    # W61 K31: 1 barrier/layer for small-M verify (default ON)
    # W51 K26: prefill dense experts, armed on the 16K layer-major schedule it
    # targets (the read-once bank pass W47 measured the ALU-bound gather on).
    # Not byte-identical to control (fp32 matmul accumulation order), so the
    # byte-identity summary flags it -- like the head-* arms, expected.
    "prefill_dense_experts": _preset(layer_major="1", prefill_dense="1"),
    # W51 window-20 follow-ups: sweep the row threshold and dequant batch, and the
    # f32-matmul variant (W50's slow-bf16-kernel hypothesis).  All ride layer-major.
    "dense_min32": _preset(
        layer_major="1", prefill_dense="1", prefill_dense_min_rows="32"
    ),
    "dense_batch16": _preset(
        layer_major="1", prefill_dense="1", prefill_dense_batch="16"
    ),
    "dense_f32": _preset(
        layer_major="1", prefill_dense="1", prefill_dense_matmul_dtype="f32"
    ),
    # W56 K27 F1: sorted routed gather -> fused gather_qmm_rhs (prefill switch).
    # Byte-identical on CPU; on Metal it swaps gather_qmv -> gather_qmm_rhs_nax
    # (K26 FP class). Armed on the 16K layer-major schedule where the switch cost is.
    "layout_fix": _preset(layer_major="1", layout_fix="1"),
    # W56 K27 F2: down-proj K padded 2304->2560 so the mxfp4 fast gather_qmv engages
    # (needs K%512==0). Byte-identical (zero-column pad). Helps the per-row gather
    # (decode + sub-threshold prefill waves). NOTE: the fast kernel engages only once
    # the streamed down bank is actually laid out 2560-wide at admission (the
    # expert_io admission-contract change flagged in W56); with today's unpadded
    # bank this arm is a byte-identical no-op that confirms parity.
    "down_k_pad": _preset(down_k_pad="1"),
    # Both K27 levers together on the 16K layer-major schedule.
    "k27_stack": _preset(layer_major="1", layout_fix="1", down_k_pad="1"),
    "both": _preset(overlap="1", layer_major="1"),          # shared_overlap + layer_major
    "all_levers": _preset(
        overlap="1", layer_major="1", sinkhorn="1", hc="1",
        fastpath="1", submit="1", attn="1", win_memo="1",
    ),  # everything on -> the fast-path here is variant B (defer + async submit)
    # W41/W45: the measured-positive / byte-identical levers stacked -- head bf16
    # (fixes the fp32-cast trap) + Sinkhorn kernel + attention-chain compile + the
    # W45 window-mask memo.  device_route is LEFT OUT (W44/window-19: NOT exact on
    # the real model -- the barrier-free gather reads an unpinned slot that a
    # mid-decode admission recycles in place before the deferred gather runs;
    # KERNEL_LEDGER K24). Re-add only once the MTPLX_GPU_PARITY window is clean.
    # The K23 fast-path also stays OUT (W42 window-14 pure defer -13.4%). overlap /
    # layer_major / hc also OFF.
    "stack_a": _preset(head="bf16", sinkhorn="1", attn="1", win_memo="1"),
    # W59: stack_a + the K30 selected-key gather (now also covering decode/verify).
    # All byte-identical/reassociation-level levers -- head bf16 (K21) + Sinkhorn
    # (K3) + attn compile (K22) + window memo (K24) + selected keys (K30).  K30 is
    # reassociation-level (greedy-identical), so stack_b as a whole is greedy-
    # identical, not byte-identical (like stack_a once head_bf16 is in).
    "stack_b": _preset(head="bf16", sinkhorn="1", attn="1", win_memo="1", selected_keys="1"),
    "head_bf16": _preset(head="bf16"),                      # W40 K21: fix fp32-cast trap
    "head_mxfp8": _preset(head="mxfp8"),                    # W40 K21: native mxfp8 gs32 head
    "head_q8": _preset(head="q8"),                          # W40 K21: affine q8 gs64 head
    # W50 K25: prefill score-path precision.  score_bf16 casts the QK^T/PV matmul
    # inputs to bf16 (LOSSY-by-design, like head_bf16 -- NOT byte-identical);
    # score_chunked is the split-K online softmax (f32, greedy-identical, caps the
    # score transient); score_bf16_chunked composes both.
    "score_bf16": _preset(score_dtype="bf16"),           # W50: -34% on GPU (window-20)
    "score_chunked": _preset(score_key_chunk="2048"),    # W50: -16% but -11 GB (peak lever)
    "score_bf16_chunked": _preset(score_dtype="bf16", score_key_chunk="2048"),
    # W50 (post window-20): the f32 pass-cut one-shot -- cuts passes over the
    # [rows,64,T] transient, not FLOPs (the score stage is pass/bandwidth-bound).
    # Reassociation-level vs control (greedy-identical), NOT byte-identical.
    "score_lean": _preset(score_path="lean"),
    # W50+W51: the stacked prefill candidate on the 16K layer-major schedule --
    # dense-experts (K26, the ALU-bound switch) + bf16 score matmuls (K25) + the
    # split-K online softmax (K25).  window-20: 346.5 s (~baseline).  LOSSY.
    "prefill_fast": _preset(
        layer_major="1", prefill_dense="1", score_dtype="bf16", score_key_chunk="2048"
    ),
    # W50 (post window-20): the f32 successor to prefill_fast -- dense experts +
    # the lean pass-cut score path (bf16 dropped, it lost on the GPU).  LOSSY
    # (dense fp32 accumulation order + score reassociation), task-eval gated.
    "prefill_lean": _preset(layer_major="1", prefill_dense="1", score_path="lean"),
    # W59 K30: prefill selected-key gather.  Gather only the window + index_topk
    # selected compressed rows each query attends into a compact [rows, k, 512]
    # operand (k ~= 640, T-independent) and score over k keys, vs the shipped
    # masked-full [rows,64,T] score.  Reassociation-level vs control (greedy-
    # identical, NOT byte-identical -- softmax sum reassociates over a different key
    # order).  Score-stage FLOPs 24x, peak score bytes ~50x at 16K; expected to cut
    # the reuse-layer attention (~144-184 s of TTFT at 16K) ~24x.  Prefill only.
    "selected_keys": _preset(selected_keys="1"),
    # W59: the K30 gather stacked onto the current best f32 prefill candidate
    # (prefill_lean = layer-major + dense experts + lean score path).  Pins every
    # key.  LOSSY vs control (dense fp32 accumulation order + score reassociation),
    # task-eval gated like prefill_lean.
    "prefill_lean_sel": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1"
    ),
    # W58 K28: the fused mask + per-head value-0 sink + f32 softmax Metal kernel
    # (2 device reads + 1 write of the [rows,64,T] transient, no T-wide
    # masked_scores/ex/concat intermediate).  Standalone (one-shot f32 + kernel),
    # to isolate the fused-softmax delta against control.  Reassociation-level
    # (greedy-identical), NOT byte-identical.
    "softmax_kernel": _preset(softmax_kernel="1"),
    # W58: the W50 prefill_lean stack + the K28 fused-softmax kernel on the 16K
    # layer-major schedule -- dense experts (K26) + lean pass-cut score path (K25)
    # + fused mask/sink/softmax (K28).  LOSSY (dense fp32 accumulation order +
    # score reassociation), task-eval gated.
    "prefill_lean_k28": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", softmax_kernel="1"
    ),
    # W58: the full prefill stack for the next window's headline A/B on the 16K
    # layer-major schedule -- dense experts (K26) + lean pass-cut score path (K25)
    # + sorted routed gather / gather_qmm_rhs (K27 layout_fix) + the K28 fused
    # mask/sink/softmax kernel.  LOSSY (dense fp32 accumulation order + score
    # reassociation), task-eval gated.  ``prefill_best_nok28`` is its no-kernel
    # twin (everything but K28), so the pair isolates the fused-softmax delta on
    # top of the otherwise-identical full stack.
    "prefill_best": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", layout_fix="1",
        softmax_kernel="1",
    ),
    "prefill_best_nok28": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", layout_fix="1",
    ),
    # W63: prefill_best_nok28 + the K30 selected-key gather -- the current best
    # f32 prefill stack (layer-major dense experts + lean pass-cut score path +
    # K27 sorted routed gather) with the score-WIDTH lever added, no K28 kernel.
    # LOSSY vs control (dense fp32 accumulation order + score reassociation),
    # task-eval gated like prefill_best_nok28. Pins every key.
    "prefill_best_sel": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", layout_fix="1",
        selected_keys="1",
    ),
    # K14 (W63): the MLX command-buffer MB cap passthrough (500 MB -- the value the
    # Qwen lane measured +1.6% at). NOT an MTPLX lever: MLX reads MLX_MAX_MB_PER_BUFFER
    # once at Metal init, so a genuine A/B runs this arm in its OWN process with the
    # value exported before launch (the harness pins + records it; an in-process arm
    # switch after MLX init does not rebind the buffer). Re-falsifies the a3b K14
    # verdict (~dead there) in the DSV4.1 streaming regime. Byte-identical.
    "mlx_buffer_500": _preset(mlx_max_mb_per_buffer="500"),
    # W60 K29: the fused decode / verify MLA attention Metal kernel -- QK^T score +
    # CSA/causal mask + per-head value-0 sink + f32 softmax + PV in ONE dispatch per
    # layer (online-softmax over key tiles, no [64,T] score row).  Standalone, to
    # isolate the fused-decode-attention delta against control on the 1K decode
    # shape (mode-agnostic: all four CSA modes route through it).  Decode + small-M
    # verify only (prefill untouched).  Reassociation-level (greedy-identical),
    # NOT byte-identical.  NOT in stack_a until the MTPLX_GPU_PARITY window is clean.
    "decode_attn_kernel": _preset(decode_attn_kernel="1"),
    # W73 K32: chunk-grown KV append -- standalone, to isolate the append delta
    # against control on the 16K decode shape (the window / compressed-KV / index-key
    # stores grow via a geometric buffer + donated slice_update instead of a full
    # concatenate per token).  BYTE-IDENTICAL to control (the buf[:, :length] view
    # equals the concatenated store), so the byte-identity summary must show it clean.
    "kv_chunk_grow": _preset(kv_chunk_grow="1"),
    # W73: the measured 16K arm (prefill_lean_sel = layer-major + dense experts +
    # lean score path + K30 selected keys) PLUS the K32 chunk-grow append fix -- the
    # direct A/B that isolates the decode-append O(T) cut at 16K on the real model.
    # Selected keys already bounds the attention score, so this arm's decode delta vs
    # prefill_lean_sel is exactly the window/compressed-KV/index-key append lanes
    # going from O(T)/O(n_comp) to amortized O(1).  LOSSY vs control only through the
    # inherited prefill_lean_sel levers (dense fp32 accumulation + score reassoc);
    # the K32 addition itself is byte-identical.
    "prefill_lean_sel_chunk": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        kv_chunk_grow="1",
    ),
    # W76: the K30 selected_idx argsort-fence -- standalone, to isolate its decode
    # census effect against control (with selected keys on so the index source
    # actually publishes selected_idx).  A stage-timing ATTRIBUTION lever only:
    # byte-identical token/cache/logits, it moves the ~O(n_comp log n_comp) argsort
    # cost from the `attn.<mode>.score` sub-stage into `attn.<mode>.select` where it
    # belongs, so the decode census's attention-*proper* is no longer inflated by an
    # index-source O(T) cost.  Not a production speedup -- it changes only what the
    # census measures, not what runs -- so it is NOT in cell16k.
    "select_fence": _preset(selected_keys="1", select_fence="1"),
    # W76: prefill_lean_sel + the argsort-fence -- the direct census A/B against
    # prefill_lean_sel (the measured 16K arm), isolating how much of that arm's
    # per-mode decode attention-proper was the mis-attributed selected_idx argsort.
    "prefill_lean_sel_fence": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        select_fence="1",
    ),
    # the standard cell: 16,384-token Qwen-PR sweep prompt (1K task + filler), real
    # prefill, then decode; prefill stack + decode stack together.  Prefill lane =
    # prefill_lean_sel_chunk (layer_major + dense experts + lean score path +
    # selected keys + chunk-grown KV append) plus the K27 layout_fix; decode lane =
    # stack_a (head bf16 + Sinkhorn + attn compile + window memo).
    "cell16k": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        kv_chunk_grow="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
    ),
    # W80 / K34: the bounded window ring in ISOLATION, with selected keys on so the
    # decode gather path (the ring's bit-identical composition -- see below) is the
    # one measured.  The window store is a bounded ring (~window_size+max_verify+
    # slack rows) and the compress/index stores are preallocated, removing the
    # per-token full-store realloc for all three lanes and cutting the 16K resident
    # window ~128x (0.7 GB -> ~5.5 MB).  BYTE-IDENTICAL to selected-keys control (the
    # dropped rows are always beyond the causal window, so every reachable gather /
    # SWA read is unchanged); the byte-identity summary must show it clean.  On the
    # masked-full path (selected keys off) it is reassociation-level instead (the
    # score-reduction width shrinks -- greedy-identical, ~1e-6), so the isolation arm
    # pins selected keys on.
    "window_ring": _preset(selected_keys="1", window_ring="1"),
    # W80: cell16k + the window ring -- the standard 16,384-token cell with the
    # bounded window store and preallocated compress/index stores replacing the
    # phase-1 full-history / chunk-grow append.  The direct A/B against cell16k that
    # isolates the ring's decode effect at 16K: the per-token append lanes stop
    # reallocating and the resident window collapses, which -- per W76/W78's
    # memory-pressure finding -- should recover the ballooned context-independent
    # decode stages far beyond the ~58 ms/tok raw append.  Byte-identical to cell16k
    # (both run selected keys), so the byte-identity summary must show it matching
    # cell16k's class (cell16k itself is lossy vs control ONLY through head=bf16 +
    # the dense/lean prefill reassoc; the ring adds NO new lossiness).
    # W107 (review round-2 finding 2): cell16k_ring is THE paired CONTROL in every
    # window (39-42), so its env set is FROZEN -- it must NOT carry kv_bounded (a
    # round-1 mistake defaulted it on, changing the timing basis vs windows 39-41).
    # The bounded lever has unvalidated Metal parity and remains a CANDIDATE
    # (cell16k_ring_bounded below); this control matches window-39's arm_env exactly.
    "cell16k_ring": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
    ),
    # W107 (review round-2): the CLEAN bounded-KV candidate -- cell16k_ring's EXACT key
    # set plus kv_bounded="1".  This is the paired candidate for the bounded lever
    # (A/B: cell16k_ring vs cell16k_ring_bounded), replacing the round-1 "flip
    # KV_BOUNDED=0 on the control" A/B (the control is now frozen without the lever).
    "cell16k_ring_bounded": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1", kv_bounded="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
    ),
    # W104 (was W81): cell16k_ring + BOTH DSpark draft-head levers -- K33 draft-block
    # tape collapse (draft="1") AND the W103 draft-head fp32-cast fix
    # (draft_head_bf16="1").  Exact key set of cell16k_ring plus those two; both touch
    # the DSpark-DIRECT draft head only (--decode-mode dspark), so they compose with
    # the ring and never change the verify math.  Runs at the profile's
    # transient_slots by default, so the verify switch is single-admission, letting
    # this arm measure the full draft-head delta on top of a single-barrier verify.
    # NB (W104): before W104 this arm pinned draft="1" ONLY; DRAFT_COMPILE in
    # isolation is still the standalone ``draft_compile`` arm.
    "cell16k_ring_draft": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        draft="1", draft_head_bf16="1",
    ),
    # W81 (window 34 stacking): cell16k_ring + W64 working-set pin + W71 barrier-free
    # pinned device route (pin_working_set="all" + device_route + device_route_pinned).
    # Exact key set of cell16k_ring plus the three pin/route flags.  The pinned device
    # route is exact only on an all-pinned route and recovers any non-pinned expert
    # fenced, so it is byte-identical to the fenced path; stacking it on the ring +
    # the profile transient_slots measures the barrier-free decode route at 16K.
    "cell16k_ring_pinned": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        pin_working_set="all", device_route="1", device_route_pinned="1",
    ),
    # W90: the shared selected-compress gather in ISOLATION (selected keys on so
    # the shared K30 gather path is the one measured).  A DISPATCH-COUNT cleanup: all
    # Reuse/Reindex/Full layers of a group read the SAME (compress_kv, selected_idx)
    # pair; the shipped path re-issues the compressed-lane gather (~3 tiny host
    # dispatches) once per layer, this issues it once per source and shares the bounded
    # [b,s,k,hd] operand (est. ~38 -> ~8 compress gathers/token real; tiny config 6->3;
    # <= ~1% of the token).  NOT the in-situ floor fix -- that floor is mode-independent
    # (window-34: swa_only 7.447 >= reuse 6.960 ms/layer, and swa has no compress_kv)
    # and tracks GPU DVFS (macmon: 71 C, ~45% busy, 580-1381 MHz), not this reference;
    # the decisive control is swa_only + the --utilization trace.  BYTE-IDENTICAL to
    # selected-keys control (a pure caching of the deterministic K30 gather).
    "attn_shape_stable": _preset(selected_keys="1", attn_shape_stable="1"),
    # W90: cell16k_ring + the shared selected-compress gather.  The ring bounds
    # the per-layer window store; this shares the compressed-lane gather so the non-swa
    # layers stop re-issuing it.  Both are byte-identical dispatch/residency cleanups --
    # NOT the mode-independent in-situ floor (that is GPU DVFS-downclock between B=1
    # bursts; see docs/deepseek-v41/W90_ATTN_IN_SITU.md).  Exact key set of cell16k_ring
    # plus attn_shape_stable="1".  BYTE-IDENTICAL to cell16k_ring (both run selected
    # keys; the shared gather adds no new lossiness), so the byte-identity summary must
    # show it matching cell16k_ring's class (lossy vs control ONLY through head=bf16 +
    # the dense/lean prefill reassoc, cf. cell16k).
    "cell16k_ring_stable": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        attn_shape_stable="1",
    ),
    # W91 / K35: the small-stages fusion in ISOLATION -- the ONLY arm carrying K35
    # (small_stages="1"), an A/B against control for the lever alone (collapses the
    # per-layer small stages -- HC premix/combine, MoE gate+top-k, shared expert, MoE
    # combine -- into three compiled per-layer graphs).  ROUNDING-CLASS ON GPU: it is
    # byte-identical on CPU (mx.compile of the segments is mx.array_equal to eager at
    # rows<=7) but NOT on Metal -- window-37 measured cell16k_ring+K35 at 2.13 tok/s
    # vs the paired reference 2.17 (NULL) with the token-id sha DIFFERING (n=1
    # mx.compile reassociates the fp32 GEMM/reductions on Metal, flipping a greedy
    # near-tie; [[dsv41-inexact-ok-if-tie-flips]]).  So K35 is a rounding-class lever
    # with NO measured GPU win: kept OUT of every composite/candidate arm (no
    # cell16k_ring_fused) -- this isolation arm is the only place it appears.  Pairs
    # with sinkhorn="1" (the K3 Sinkhorn kernel backs the fused premix on the GPU).
    "small_stages_fused": _preset(small_stages="1", sinkhorn="1"),
    # W87 (window 35): cell16k_ring + the single-slot pool (MTPLX_DSV41_SINGLE_SLOT_
    # POOL).  EXACT key set of cell16k_ring plus single_slot_pool="1".  Merges each
    # layer's persistent + transient tiers into ONE scan-resistant resident pool so
    # the 16K prefill leaves the prompt tail resident and decode starts warm instead
    # of re-streaming its own working set (measured two-tier: persistent learns
    # nothing from prefill; decode hit rate 0.741 at 49 slots/layer).  Allocation is
    # UNCHANGED (bytes identical to cell16k_ring), so this is a pure residency change
    # -> BYTE-IDENTICAL tokens/logits; the direct A/B vs cell16k_ring isolates the
    # cold-start recovery, which reads from the receipt's cold_start block
    # (decode_hit_rate_first_64_steps vs steady_state, populated by BOTH arms).
    "cell16k_ring_pool": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        single_slot_pool="1",
    ),
    # W97: cache the dequantized grouped o-LoRA wo_a per layer instead of re-issuing
    # mx.dequantize(wo_a) every decode token (the largest per-token attention traffic
    # item: 304 MB/layer q8 / 420 MB mxfp4; the reference dequantizes it once at
    # convert).  ISOLATION arm (selected_keys on so the decode attention path is the
    # shipped one).  BYTE-IDENTICAL to selected-keys control (the cached array is
    # exactly the dequantize output; the einsum's .astype(f32) is unchanged) -- the
    # byte-identity summary must show it clean.  Holds a dense wo_a copy resident per
    # layer (q8 ~5.4 GB / native ~2.7 GB across 40), so mind the memory limit.
    "wo_a_cache": _preset(selected_keys="1", wo_a_cache="1"),
    # W97: cell16k_ring + the wo_a-dequant cache ONLY.  Exact key set of cell16k_ring
    # plus wo_a_cache="1"; the direct A/B vs cell16k_ring isolates the per-token
    # mx.dequantize(wo_a) + f32-astype cost (40 dequant dispatches + the ~10.7 GB/token
    # f32 astype write+read the earlier bf16 cache left in place, BOTH codecs).
    # BYTE-IDENTICAL to cell16k_ring -- the byte-identity summary must show it clean.
    # The f32 cache is ~5.4 GB resident (40 x 134 MB, both codecs), priced into the
    # memory plan (deepseek_v41_loader reserves it as fixed resident when armed, so the
    # expert-cache allowance shrinks by it).  Watch peak memory: cell16k_ring already
    # peaks ~65 GB at the 16K cell, so run this arm at a SMALLER cell if it OOMs -- do
    # NOT raise --memory-limit-gib, which would raise the expert allowance by the same
    # amount and re-open the overshoot.
    "cell16k_ring_wo_a_cache": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        wo_a_cache="1",
    ),
    # W97: fixed-shape mx.compile of the decode-attention core in ISOLATION
    # (selected_keys on so _sparse_attend_selected -- and thus the core -- is the
    # path).  ROUNDING-CLASS (n=1 compile reassociates the fp32 einsum/reductions),
    # so the byte-identity summary MUST flag it (token-id sha differs vs control on a
    # greedy near-tie flip -- [[dsv41-inexact-ok-if-tie-flips]]); the direct A/B vs
    # selected_keys isolates the core-tape dispatch collapse (13 -> 8 kernels).
    "attn_core_compile": _preset(selected_keys="1", attn_core_compile="1"),
    # W97: cell16k_ring + the core compile ONLY.  Exact key set of cell16k_ring plus
    # attn_core_compile="1".  ROUNDING-CLASS (adds the n=1 core reassociation on top
    # of cell16k_ring's head=bf16 loss), so NOT byte-identical -- expected; the A/B vs
    # cell16k_ring isolates the core collapse.  The K29 fused decode kernel
    # (decode_attn_kernel) is the lower-dispatch alternative (core -> 1 dispatch) and,
    # if also armed, wins the early return before this path.
    "cell16k_ring_attn_core": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        attn_core_compile="1",
    ),
    # W97: cell16k_ring + the wo_a-dequant cache (exact) + the core compile (rounding-
    # class) stacked -- the full W97 attention-dispatch program.  ROUNDING-CLASS via
    # the core compile; the wo_a cache is byte-identical on its own.  Watch peak
    # memory (the wo_a cache holds a dense wo_a copy resident per layer, ~5.4/2.7 GB).
    "cell16k_ring_wo_a_core": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        wo_a_cache="1", attn_core_compile="1",
    ),
    # W97 follow-on: cell16k_ring + the wo_a-dequant cache (exact) + the K29 FUSED
    # decode-attention kernel (decode_attn_kernel="1"): the SINGLE-DISPATCH core
    # (score+mask+sink softmax+PV in one metal_kernel) instead of the mx.compile core.
    # K29 is ROUNDING-CLASS (its tile reduction reassociates the fp32 softmax), so the
    # byte-identity summary flags it (token-id sha differs on a greedy tie flip); this
    # arm lets window 40/41 measure the 1-dispatch core against the ~8-kernel compile
    # core (cell16k_ring_wo_a_core) and the eager baseline (cell16k_ring_wo_a_cache).
    "cell16k_ring_wo_a_k29": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        wo_a_cache="1", decode_attn_kernel="1",
    ),
    # W99: lean the decode-attention casts in ISOLATION (selected_keys on so the eager
    # core -- where the redundant KVg double-cast lives -- is the path).  BYTE-IDENTICAL
    # to selected_keys control (dedupes the KVg f32 cast, caches the f32 sink + the
    # device inv_freq); the byte-identity summary must show it clean.
    "attn_lean_casts": _preset(selected_keys="1", attn_lean_casts="1"),
    # W99: cell16k_ring + the wo_a cache + lean casts -- the BYTE-IDENTICAL W97/W99
    # attention stack (both levers are exact; the only loss vs control is
    # cell16k_ring's own head=bf16).  A/B vs cell16k_ring isolates the exact-lever
    # dispatch savings (wo_a per-token dequant removed + KVg/sink/inv_freq casts leaned).
    "cell16k_ring_lean": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        wo_a_cache="1", attn_lean_casts="1",
    ),
    # W99: cell16k_ring_lean + the K29 fused decode core (1-dispatch score+softmax+PV).
    # ROUNDING-CLASS via K29 (flagged in the byte-identity summary); the lowest-dispatch
    # attention arm (exact wo_a cache + lean casts + the fused core).
    "cell16k_ring_lean_k29": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        wo_a_cache="1", attn_lean_casts="1", decode_attn_kernel="1",
    ),
    # W101: fuse the qkv/out projection-chain GLUE in ISOLATION (selected_keys on so
    # the fused out-prep's o-derope + bf16 o-LoRA einsum is the path; core runs
    # eager -- fused-proj is INDEPENDENT of K29).  ROUNDING-CLASS (fused rmsnorm
    # reassociates the fp32 sum; the o-LoRA einsum reads bf16 wo_a) -- the byte-
    # identity summary flags it; the A/B vs attn_lean_casts / selected_keys isolates
    # the projection-chain dispatch collapse.  GPU-only, small-M (b*s <= 8).
    "attn_fused_proj": _preset(selected_keys="1", attn_fused_proj="1"),
    # W101: cell16k_ring + the wo_a cache + lean casts + K29 (core -> 1 dispatch) +
    # fused proj -- the FULL attention dispatch stack (qkv/out glue fused, core K29,
    # per-token dequant + redundant casts gone).  ROUNDING-CLASS via K29 + fused
    # proj (flagged in the byte-identity summary; token-id sha differs on a greedy
    # tie flip -- [[dsv41-inexact-ok-if-tie-flips]]).  The lowest-dispatch attention
    # arm; the direct A/B vs cell16k_ring_lean_k29 isolates the projection-chain
    # fusion on top of the already-collapsed core.  Note the fused path holds a bf16
    # wo_a copy resident per layer (~2.7 GB) -- watch peak memory at the 16K cell.
    "cell16k_ring_fused": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        wo_a_cache="1", attn_lean_casts="1", decode_attn_kernel="1",
        attn_fused_proj="1",
    ),
    # W92 (switch dispatch census): the minimal AR-decode host-sync-reduction arm in
    # ISOLATION -- the K23 variant-B fast-path (defer + async submit) plus verify
    # single-barrier (default ON, pinned explicit).  Window 33 (arm cell16k_ring)
    # ran the AR all-hit switch through the shipped SYNCHRONOUS wave fence
    # (hot.allhit_fence_eval == hot.all_hit): a SECOND blocking mx.eval(wave_output)
    # per all-hit layer on top of the one mx.eval(indices) routing barrier, plus a
    # blocking fence per split-route wave part on miss layers.  This arm defers that
    # release to the next layer's routing barrier (the covering eval) and async-
    # submits the gather so the GPU is fed WITHOUT the blocking round-trip (variant B;
    # pure defer without submit lost -13% at 1K -- W42 window-14 -- because the lazy
    # graph then accrued and the device idled until the next barrier drained it).
    # Net: exactly ONE small eval (indices only) per streamed layer.  Byte-identical
    # to control (pure fence/release-timing reorder; the gather math is unchanged);
    # pin-safe (try_all_hit_route pins the whole route and defer_slot_release holds
    # those pins until the covering flush, so no admission can recycle a slot whose
    # gather is still pending -- the W44 slot-recycle hazard cannot arise).
    "switch_lean": _preset(fastpath="1", submit="1", verify_single="1"),
    # W92: cell16k_ring + the switch-lean keys (K23 variant B + verify single-barrier).
    # The direct A/B against cell16k_ring that isolates the AR-decode host-sync cut at
    # 16K: same prefill + decode stack, only the per-all-hit-layer wave fence and the
    # per-split-wave miss fences are deferred to the next routing barrier (async-
    # submitted meanwhile).  Byte-identical to cell16k_ring (cell16k_ring is itself
    # lossy vs control only through head=bf16 + the dense/lean prefill reassoc; the
    # switch keys add NO new lossiness -- they are a fence/release-timing reorder).
    # EXPECTATION: small.  At 16K only ~30% of layer-calls are all-hit (window 33:
    # 3,054/10,240); the other ~70% are split layers that block on SSD regardless of
    # fences, so the removable exposed cost is <= ~2.5% (<= ~11 ms/token), inside
    # single-prompt seed noise.  The +2.86% variant-B figure is the 1K regime (window
    # 16, ab-1024-fastpath-b.json), not 16K.  Primarily a host-sync-hygiene lever.
    "cell16k_ring_switch": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        fastpath="1", submit="1", verify_single="1",
    ),
    # W93: the gate-oracle one-layer-ahead prefetch in ISOLATION at k=10 (W89's
    # b' predictor clears the 0.70 overlap threshold at width 10, missRed@10
    # 0.736). During layer L-1's decode the residual entering L-1 is scored by
    # layer L's own router and L's predicted top-10 experts stream from SSD into a
    # bounded ring, so L's true-route misses are pre-warmed. BYTE-IDENTICAL to
    # control: the MoE still gathers on the TRUE route (the prediction only warms
    # the cache; a mispredict wastes a read), and the prediction rides L-1's
    # existing indices barrier (no new host sync). The gate_prefetch receipt block
    # reads the hit rate. Sizes the ring (prefetch_slots=10) via the loader flag.
    "gate_prefetch": _preset(gate_prefetch="10"),
    # W93: cell16k_ring + the gate-oracle prefetch at k=10 -- the standard 16K cell
    # (ring + measured decode/prefill stack) with one-layer-ahead expert prefetch
    # stacked on. The direct A/B against cell16k_ring that isolates how much of the
    # ~19% AR I/O the one-ahead prefetch hides. Byte-identical to cell16k_ring
    # (same lossy class -- head=bf16 + dense/lean prefill reassoc; the prefetch
    # adds NO new lossiness), so the byte-identity summary must show it matching
    # cell16k_ring's class.
    "cell16k_ring_prefetch": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        gate_prefetch="10",
    ),
    # W95: the single v2 runner switch (docs/deepseek-v41/W95_RUNNER_DESIGN.md).
    # ONE key (MTPLX_DSV41_RUNNER=v2) composes the W93 gate-oracle one-layer-ahead
    # prefetch (k=12 default, ring 2*k) AND the W87 single scan-resistant pool
    # (admit every miss) -- NOT the individual sub-keys stacked. First deliverable
    # (post window-38, which withdrew the host-sync-drain premise): HIDE THE SSD
    # MISS WAITS. BYTE-IDENTICAL to control's class -- residency-only (prefetch
    # warms the cache on the TRUE route; the pool only changes which loads happen).
    "runner_v2": _preset(runner="v2"),
    # W95: cell16k_ring + the single v2 switch -- the standard 16K cell (ring +
    # measured decode/prefill stack) with the composed SSD-hiding runner on ONE key.
    # The paired A/B against cell16k_ring measures misses/token, bytes/token and the
    # SSD-bound ms down (target >=60% at equal hit rate) and AR tok/s up (2.05 ->
    # ~2.9, attention untouched). Byte-identical to cell16k_ring's class (head=bf16
    # + dense/lean prefill reassoc; the runner adds NO new lossiness).
    "cell16k_ring_v2": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2",
    ),
    # W104: cell16k_ring_v2 + BOTH DSpark draft-head levers (K33 draft-block tape
    # collapse + the W103 draft-head fp32-cast fix).  Exact key set of cell16k_ring_v2
    # plus draft="1" + draft_head_bf16="1"; both touch the DSpark draft head only, so
    # they compose with the v2 runner's SSD-hiding verify path and never change the
    # verify math.  The direct A/B vs cell16k_ring_v2 isolates the two draft-head
    # levers on the standard 16K cell with the v2 runner armed.  (W104 traced the draft
    # MoE to an already barrier-free resident gather_qmm(mode="mxfp4") -- zero host
    # syncs -- so there is NO draft-MoE lever to stack here; see
    # docs/deepseek-v41/W104_DRAFT_RESIDENT_MOE.md.)
    "cell16k_ring_v2_draft": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2", draft="1", draft_head_bf16="1",
    ),
    # W97F composite: cell16k_ring_v2 + the byte-identical W97/W99 lean attention
    # stack (wo_a f32 cache + leaned casts) + the W101/K36 fused projection-chain glue.
    # EXACT KEY SET (13 keys) = cell16k_ring_v2's twelve keys
    #   layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
    #   window_ring="1", layout_fix="1", kv_bounded="1", head="bf16", sinkhorn="1",
    #   attn="1", win_memo="1", runner="v2"
    # PLUS the three W97/W99/W101 attention keys
    #   wo_a_cache="1", attn_lean_casts="1", attn_fused_proj="1".
    # wo_a_cache + attn_lean_casts are the BYTE-IDENTICAL (exact) W97/W99 lean stack
    # (cell16k_ring_lean); attn_fused_proj is the W101/K36 GPU-only small-M (b*s<=8)
    # projection glue.  The K29 fused decode core (decode_attn_kernel) is DELIBERATELY
    # NOT armed -- the eager core stays -- so the direct A/B vs cell16k_ring_v2 isolates
    # the v2 SSD-hiding runner combined with the attention-dispatch reductions (lean
    # stack + fused proj) without the fused core. Projection caches replace one
    # another by route; the plan reserves the larger fp32 prefill representation.
    "cell16k_ring_v2_attn": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2",
        wo_a_cache="1", attn_lean_casts="1", attn_fused_proj="1",
    ),
    # W122 prefetch-width pair for the SSD roofline: cell16k_ring_v2_attn with the
    # gate-oracle predict width PINNED explicitly (off vs wide), to isolate whether
    # the v2 auto-armed prefetch is net-positive or is drowning the demand reads.
    # MOTIVATION (W122 roofline census): under cell16k_ring_v2_attn the v2 runner
    # auto-arms the gate-oracle at k=6 (MTPLX_DSV41_GATE_PREFETCH UNSET -> the runner
    # default _RUNNER_V2_GATE_PREFETCH_K), and the census found prefetch is DOUBLING
    # the SSD traffic without hiding the misses it costs:
    #   * 1.428 GB/token total SSD read vs 0.711 GB/token DEMAND -- the speculative
    #     reads are ~= the demand reads (a 2x traffic multiplier), yet
    #   * the ring COMMIT rate is only 59% (41% of prefetched records are evicted
    #     unconsumed -- wasted bandwidth), and
    #   * the io reader pool runs at an effective queue depth of 1 (QD1): a
    #     speculative read in flight BLOCKS the next demand read behind it, so the
    #     prefetch is not just wasted bandwidth, it serializes ahead of the reads the
    #     token actually needs.
    # These two arms pin the width so a clean A/B (both vs cell16k_ring_v2_attn's
    # k=6 auto-arm) measures: pf0 = prefetch fully OFF (does removing the 2x traffic /
    # the QD1 head-of-line block recover decode?), pf8 = prefetch WIDER (does more
    # lookahead raise the 59% commit rate enough to pay for the extra traffic?).
    # SAME rounding class as cell16k_ring_v2_attn (both inherit attn_fused_proj, the
    # only rounding-class key in the set); the added gate_prefetch key is ITSELF
    # byte-identical -- the gate-oracle only WARMS the expert cache on the layer's TRUE
    # route (a mispredict wastes a read, a hit saves a wait); the MoE still gathers on
    # the true route, so neither width changes the routed math.
    # NOTE (why the presets are REQUIRED, not ambient env): _apply_arm_env pops every
    # key whose preset value is None, so an ambient MTPLX_DSV41_GATE_PREFETCH exported
    # in the parent shell is CLEARED by cell16k_ring_v2_attn (its gate_prefetch=None)
    # -- the width can only be pinned by a preset that carries the key.
    # pf0: MTPLX_DSV41_GATE_PREFETCH=0 -> deepseek_v41._resolve_gate_prefetch_k returns
    # 0 (a non-positive explicit value is OFF and, being explicit, WINS over the v2
    # auto-arm) -> the ring is not issued, byte-identical routing with zero speculative
    # traffic.
    "cell16k_ring_v2_attn_pf0": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2",
        wo_a_cache="1", attn_lean_casts="1", attn_fused_proj="1",
        gate_prefetch="0",
    ),
    # pf8: MTPLX_DSV41_GATE_PREFETCH=8 -> _resolve_gate_prefetch_k returns 8 (explicit,
    # wins over the k=6 auto-arm) -> a WIDER one-layer-ahead predict set than the v2
    # default; tests whether more lookahead lifts the 59% commit rate above the extra
    # 2x traffic cost.
    "cell16k_ring_v2_attn_pf8": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2",
        wo_a_cache="1", attn_lean_casts="1", attn_fused_proj="1",
        gate_prefetch="8",
    ),
    # W123 routing-barrier pair for the critical-path audit: cell16k_ring_v2_attn with
    # the two EXISTING byte-identical barrier levers that are OFF in the cell arms.
    # MOTIVATION (W123 critical-path census): per routed MoE layer the token pays
    # ~1 ms mx.eval(indices) routing barrier + ~2 ms host (tolist / prefetch-reconcile
    # await / planning) + ~1.15 ms GPU; x40 routed layers = ~169 ms, the whole token.
    # Two shipped levers attack the ~1 ms eval(indices) barrier and neither is armed by
    # cell16k_ring_v2_attn:
    #   * MTPLX_DSV41_SHARED_OVERLAP (=overlap kwarg -> OVERLAP_ENV): a PURE per-forward
    #     execution reorder (expert_mlx.py run_with_shared_overlap): the resident shared
    #     expert depends only on x, not on the routed indices, so it is dispatched INTO
    #     the eval(indices) sync's GPU-idle bubble (async_eval) instead of after the
    #     split route. Same shared_mlp(x), same combine -> BITWISE-IDENTICAL output.
    #   * MTPLX_DSV41_DEVICE_ROUTE (=device_route kwarg -> DEVICE_ROUTE_ENV): the K24/W44
    #     barrier-free all-hit path (NOT the pinned variant). It gathers lut[indices] on
    #     the DEVICE with no mx.eval(indices) and defers verification to ONE batched
    #     token-boundary flush; a cold miss reads a void row and that layer is recomputed
    #     on the fenced path, so the emitted token stays byte-identical (a cold-token
    #     RECOVERY cost, not a numeric divergence -- see the OVERLAP/DEVICE_ROUTE notes
    #     at ~L306 / ~L556). NET barrier removal banks only on all-hit layers.
    # NEITHER env is a rounding-class key (ROUNDING_CLASS_ENVS), so these arms keep
    # cell16k_ring_v2_attn's rounding-class status (attn_fused_proj) UNCHANGED. As with
    # every arm, _apply_arm_env force-unsets the key the preset leaves None, so the base
    # arm CLEARS an ambient MTPLX_DSV41_SHARED_OVERLAP / _DEVICE_ROUTE -- the levers can
    # only be pinned by a preset carrying the key.
    "cell16k_ring_v2_attn_ovl": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2",
        wo_a_cache="1", attn_lean_casts="1", attn_fused_proj="1",
        overlap="1",
    ),
    "cell16k_ring_v2_attn_dr": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2",
        wo_a_cache="1", attn_lean_casts="1", attn_fused_proj="1",
        device_route="1",
    ),
    "cell16k_ring_v2_attn_ovl_dr": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2",
        wo_a_cache="1", attn_lean_casts="1", attn_fused_proj="1",
        overlap="1", device_route="1",
    ),
    # W97F composite (DSpark): cell16k_ring_v2_draft + the SAME three attention keys as
    # cell16k_ring_v2_attn.  EXACT KEY SET (15 keys) = cell16k_ring_v2_draft's twelve
    #   layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
    #   window_ring="1", layout_fix="1", kv_bounded="1", head="bf16", sinkhorn="1",
    #   attn="1", win_memo="1", runner="v2", draft="1", draft_head_bf16="1"
    # PLUS wo_a_cache="1", attn_lean_casts="1", attn_fused_proj="1".
    # The draft + draft_head_bf16 keys drive the DSpark draft head (--decode-mode
    # dspark); the three attention keys apply to the shared trunk attention exactly as
    # in cell16k_ring_v2_attn.  The direct A/B vs cell16k_ring_v2_draft isolates the
    # W97/W99/W101 attention stack under the DSpark decode lane.
    "cell16k_ring_v2_draft_attn": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2", draft="1", draft_head_bf16="1",
        wo_a_cache="1", attn_lean_casts="1", attn_fused_proj="1",
    ),
    # W118 pair for window 46: cell16k_ring_v2_attn + the MLX allocator-limit headroom
    # lever at 8 GiB (H7).  EXACT KEY SET = cell16k_ring_v2_attn's keys PLUS
    # mlx_limit_headroom="8" (MTPLX_DSV41_MLX_LIMIT_HEADROOM_GIB).  The headroom raises
    # ONLY the mx.set_memory_limit soft cap above the residency plan -- residents,
    # expert-cache slots and the prefetch ring are UNCHANGED, so the direct A/B vs
    # cell16k_ring_v2_attn is BYTE-IDENTICAL (same plan_limit_gib_effective) and
    # isolates whether lifting the allocator over-limit path (plan 69.2 < mlx_peak 74.3)
    # frees the in-model attention/verify from the allocator-pressure regime.  Pin the
    # plan with --memory-plan-from so both arms run the SAME plan_limit.
    "cell16k_ring_v2_attn_hr8": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2",
        wo_a_cache="1", attn_lean_casts="1", attn_fused_proj="1",
        mlx_limit_headroom="8",
    ),
    # W118 pair for window 46 (DSpark): cell16k_ring_v2_draft_attn + headroom 8.  EXACT
    # KEY SET = cell16k_ring_v2_draft_attn's keys PLUS mlx_limit_headroom="8".  The
    # headroom is a whole-process allocator cap (not a decode lever), so it composes
    # with the DSpark draft/verify path exactly as on the AR arm; the direct A/B vs
    # cell16k_ring_v2_draft_attn isolates the headroom under the DSpark decode lane.
    "cell16k_ring_v2_draft_attn_hr8": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2", draft="1", draft_head_bf16="1",
        wo_a_cache="1", attn_lean_casts="1", attn_fused_proj="1",
        mlx_limit_headroom="8",
    ),
    # W107F pair for window 44: cell16k_ring_v2_attn + kv_bounded="1".  The A/B arm that
    # ISOLATES the bounded-KV lever on top of the full attention stack.  W121 HIGH-4:
    # bounded KV is byte-identical on CPU but has UNVALIDATED Metal parity. Recorded
    # outputs differ around token 33; layout-dependent rounding is only a hypothesis.
    # Window 47/48 also changed cache capacity (66 -> 71 slots/layer), so that pair
    # does not isolate the numerical cause. Full-model installation is rejected;
    # this preset is retained for dry-run/config inspection. EXACT KEY SET =
    # cell16k_ring_v2_attn (layer_major, prefill_dense, score_path=lean, selected_keys,
    # window_ring, layout_fix, head=bf16, sinkhorn, attn, win_memo, runner=v2, wo_a_cache,
    # attn_lean_casts, attn_fused_proj) PLUS kv_bounded="1".
    "cell16k_ring_v2_attn_bounded": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1", kv_bounded="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2",
        wo_a_cache="1", attn_lean_casts="1", attn_fused_proj="1",
    ),
    # W107F pair for window 44 (DSpark): cell16k_ring_v2_draft_attn + kv_bounded="1" --
    # the DSpark bounded-KV candidate (unvalidated Metal parity, W121 HIGH-4;
    # full-model installation rejected). EXACT KEY SET = cell16k_ring_v2_draft_attn (its 16 keys
    # incl. runner=v2, draft, draft_head_bf16, wo_a_cache, attn_lean_casts,
    # attn_fused_proj) PLUS kv_bounded="1".
    "cell16k_ring_v2_draft_attn_bounded": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1", kv_bounded="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2", draft="1", draft_head_bf16="1",
        wo_a_cache="1", attn_lean_casts="1", attn_fused_proj="1",
    ),
    # W115: the K29 verify-core A/B.  cell16k_ring_v2_draft_attn runs the K+1 verify
    # through the K29 fused decode-attention CORE -- NOT because the preset arms it (it
    # does not) but because _run_arm setdefaults DECODE_ATTN_KERNEL="1" for every
    # --decode-mode dspark arm (dspark_decode_kernel_env_defaults).  This arm pins K29
    # OFF: decode_attn_kernel="0" is the RUNTIME knob (an explicit "0" beats the
    # setdefault, which only fills unset keys) and dspark_verify_k29="0" drops K29 from
    # the setdefault entirely -- BOTH, so nothing can re-arm it.  Every other key is
    # identical to cell16k_ring_v2_draft_attn.
    #
    # SCOPE CAVEAT: DECODE_ATTN_KERNEL is a WHOLE-ARM runtime knob, not verify-only.
    # It also gates the DSpark DRAFT attention (deepseek_v41_dspark.py:591 _sparse_attend
    # -> _decode_attn_kernel_use) and the AR reference decode (M=1) via the same gate.
    # So this arm flips K29 for draft + AR + verify at once.  W60: K29 is -38% vs eager
    # at M=1, so this arm's DRAFT may be FASTER and its headline tok/s / stats.draft_ms
    # are CONFOUNDED by the draft flip -- do NOT read them as the verify-core delta.
    # ONLY dspark.per_cycle_ms.verify_ms (and verify_stage_timing's attn stages) isolate
    # the verify core.  Proof reads rows AND calls: control
    # decode_attn_kernel_engagement has calls>0 with rows>1 (verify rows engaged); this
    # arm has calls==0.  Run at DSpark depth <= 7 (K+1 <= 8): a depth-8 verify is >8 rows
    # and runs eager on BOTH arms while control's calls stays >0 from the M=1 draft, so
    # calls alone would mislead.  greedy-identical to AR either way (verify authoritative;
    # this arm's core is byte-identical eager, K29 is rounding-class).
    "cell16k_ring_v2_draft_attn_eager": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2", draft="1", draft_head_bf16="1",
        wo_a_cache="1", attn_lean_casts="1", attn_fused_proj="1",
        decode_attn_kernel="0", dspark_verify_k29="0",
    ),
    # Current memory-bounded DSpark candidate: the measured pf0 target route,
    # compiled draft chains and bf16 draft head, with K29 explicitly off.
    "cell16k_ring_v2_draft_attn_pf0": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2", draft="1", draft_head_bf16="1",
        wo_a_cache="1", attn_lean_casts="1", attn_fused_proj="1",
        decode_attn_kernel="0", dspark_verify_k29="0", gate_prefetch="0",
    ),
    # Same bounded DSpark route with the shared output head repacked once to
    # affine q8. Target and draft both execute the installed q8 head directly.
    # The slot plan deliberately keeps pricing the dense construction head: the
    # component banks are materialized before the post-load repack releases it.
    # This tie-break-class candidate stays opt-in until the exact lane measures it.
    "cell16k_ring_v2_draft_attn_pf0_head_q8": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="q8", sinkhorn="1", attn="1", win_memo="1",
        runner="v2", draft="1", draft_head_bf16="1",
        wo_a_cache="1", attn_lean_casts="1", attn_fused_proj="1",
        decode_attn_kernel="0", dspark_verify_k29="0", gate_prefetch="0",
    ),
    # Exact-layout real-weight one-layer screen: direct packed MXFP8 wo_a plus
    # the common wo_b was 1.22x at M=1 and 1.19x at M=6, while removing the
    # 2.684 GB target BF16 cache.
    "cell16k_ring_v2_draft_attn_pf0_woa_direct": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2", draft="1", draft_head_bf16="1",
        wo_a_cache="1", attn_lean_casts="1", attn_fused_proj="1",
        attn_wo_a_direct="1",
        decode_attn_kernel="0", dspark_verify_k29="0", gate_prefetch="0",
    ),
    # W110 (BENCH-ONLY DIAGNOSTIC): cell16k_ring_v2 + decode-path per-record sha256
    # turned ON (MTPLX_DSV41_VERIFY_RECORD_HASHES=1, env-authoritative over the ab
    # harness's --verify-record-hashes default False).  This is NOT a perf lever:
    # decode hashing is already OFF on every ab/bench path and OFF in the served
    # profile, so there was nothing to remove.  The A/B cell16k_ring_v2_hash vs
    # cell16k_ring_v2 MEASURES the io-thread cost of hashing (records_hashed /
    # hash_thread_ns) should a future policy ever require it -- the reverse of the
    # withdrawn W109 §1.b "drop hashing" framing.  Byte-identical class either way
    # (hashing never changes bytes, so token_ids_sha256 must match); the
    # resolved_plan.verify_record_hashes stamp proves the two arms actually differ.
    "cell16k_ring_v2_hash": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2", verify_record_hashes="1",
    ),
}

# W97 (review item 7): the rounding-class env keys, documented in ONE place with the
# reason.  An arm whose preset arms ANY of these keys has decoded tokens that are
# EXPECTED to differ from control by rounding (the lever reassociates the fp32
# attention core / softmax, so a greedy near-tie can flip -- [[dsv41-inexact-ok-if-
# tie-flips]]). These are per-arm metadata, not a pairwise parity exemption. The
# summary compares effective levers in BOTH receipts: an unchanged rounding lever
# cannot explain a new mismatch, and unvalidated bounded KV is never exempted.
#
# ROUNDING_CLASS_ARMS is DERIVED from ARM_PRESETS (not a hand list), so a new arm is
# classified automatically the moment its preset names one of these keys.  Each key's
# reason (why reassociation, not a bug):
#   DECODE_ATTN_KERNEL (K29): the fused decode/verify MLA kernel's online-softmax tile
#       reduction reorders the fp32 softmax -- greedy-identical <=1e-6, NOT byte-ident
#       (W60; GPU-only, CPU falls back to the eager one-shot).
#   ATTN_CORE_COMPILE (W97): the n=1 fixed-shape mx.compile of the decode-attention
#       core reassociates the fp32 einsum/reductions (the K35 lesson; measured max|Δ|
#       ~9e-10 on CPU) -- NOT byte-identical, on CPU AND GPU.
#   SMALL_STAGES_FUSED (K35): the fused per-layer small-stage graphs reassociate the
#       fp32 GEMM/reductions on Metal (window-37: token-id sha DIFFERS on GPU;
#       byte-identical only within the CPU mx.compile bit-exact regime).
#   HC_PREMIX_KERNEL (K35): the fused HC-premix Sinkhorn kernel (rounding-class 1e-6,
#       argmax-exact); pinned force-unset by every preset today, listed so a future
#       arm that turns it on is classified automatically.
#   MTPLX_DSV41_DRAFT_HEAD_BF16: a bf16 DSpark draft head would round the draft logits;
#       no current preset arms it (no such constant yet), listed by name so a future
#       arm classifies without a code change.
# DELIBERATELY EXCLUDED (kept FAIL so a genuine exact-lever regression is caught):
#   HC_COMPILE (K4) and SINKHORN_METAL (K3) are classed byte-identical execution
#       reorders on this CPU A/B path (K4 is a byte-identical HC-premix compile; K3
#       falls back to eager on CPU), and MANY exact composite arms carry sinkhorn="1"
#       (cell16k_ring_wo_a_cache, cell16k_ring_lean, stack_*) -- excusing them would
#       mask a real exact-lever divergence.  HEAD_MODE=bf16/mxfp8/q8 is a LOSSY-by-
#       design LOAD-TIME codec, a different class (flagged separately, W40_HEAD_LEVER),
#       not a rounding reorder -- so head=bf16 on an otherwise-exact arm is NOT what
#       makes it rounding-class.  ATTN_LEAN_CASTS (W99) is a byte-identical cast dedupe.
ROUNDING_CLASS_ENVS = (
    DECODE_ATTN_KERNEL_ENV,
    ATTN_CORE_COMPILE_ENV,
    SMALL_STAGES_FUSED_ENV,
    HC_PREMIX_KERNEL_ENV,
    "MTPLX_DSV41_DRAFT_HEAD_BF16",
    ATTN_FUSED_PROJ_ENV,  # W101: metal_kernel glue + cached pre-transposed wo_a (GPU numerics: rounding-class, 0 greedy flips / 65)
    ATTN_WO_A_DIRECT_ENV,  # packed MXFP8 gather_qmm changes the reduction path
)


#: Preset values that mean a lever is UNSET / turned OFF (never a rounding-class
#: reason).  W115: an arm may pin a rounding-class key to an explicit "0" to defeat a
#: setdefault (cell16k_ring_v2_draft_attn_eager pins DECODE_ATTN_KERNEL="0"); an OFF
#: lever must not be listed as a reason its tokens round.  This is the INTERSECTION of
#: OFF across every rounding-class runtime resolver: the bool resolvers
#: (_resolve_decode_attn_kernel/_attn_core_compile/_attn_fused_proj) treat "none"/
#: "default" as OFF but the _env_truthy resolvers (SMALL_STAGES_FUSED, HC_PREMIX_KERNEL)
#: and _draft_head_bf16_on treat "none"/"default" as ON and "auto" as OFF -- so ONLY
#: these six are OFF everywhere.  "none"/"auto"/"default" are deliberately excluded
#: (test_off_values_are_off_in_every_runtime_resolver guards this).
_LEVER_OFF_VALUES = frozenset({None, "", "0", "false", "off", "no"})


def _rounding_class_keys(arm: str) -> list:
    """The rounding-class env keys (see ``ROUNDING_CLASS_ENVS``) an arm's preset
    actually arms -- the reason its tokens are EXPECTED to differ from control by
    rounding.  Empty list for an exact arm.  Derived from ``ARM_PRESETS``, never
    hand-listed, so a new rounding-class arm is classified automatically.  A key pinned
    to an explicit OFF value ("0"/None/...) is NOT a reason (W115: an eager arm pins the
    verify core off with DECODE_ATTN_KERNEL="0")."""
    preset = ARM_PRESETS.get(arm, {})
    return [
        k for k in ROUNDING_CLASS_ENVS
        if str(preset.get(k) if preset.get(k) is not None else "").strip().lower()
        not in _LEVER_OFF_VALUES
    ]


def _is_rounding_class(arm: str) -> bool:
    """True when ``arm`` arms any rounding-class env key (see ``_rounding_class_keys``)."""
    return bool(_rounding_class_keys(arm))


# Derived, not hand-listed: every arm whose preset arms a rounding-class env key.
ROUNDING_CLASS_ARMS = frozenset(a for a in ARM_PRESETS if _is_rounding_class(a))


def _pairwise_rounding_class_keys(base, candidate) -> list:
    """Known rounding levers whose effective setting changed between these runs.

    Recorded runtime env wins over presets (including explicit None/OFF). Presets
    supply missing keys in older receipts. Historical per-arm ``rounding_class``
    labels cannot establish the cause of a difference between two composite arms.
    Bounded KV has no validated Metal error bound, so it cannot claim an exemption.
    """
    def effective_env(receipt):
        env = dict(ARM_PRESETS.get(receipt.get("arm"), {}))
        env.update(receipt.get("arm_env") or {})
        return env

    def state(key, value):
        raw = str(value or "").strip().lower()
        if key == KV_BOUNDED_ENV:
            return raw in ("1", "true", "yes", "on")
        if key in (
            DECODE_ATTN_KERNEL_ENV,
            ATTN_CORE_COMPILE_ENV,
            ATTN_FUSED_PROJ_ENV,
            ATTN_WO_A_DIRECT_ENV,
        ):
            # Match the strict bool resolvers without importing the Metal model.
            if raw in ("", "0", "false", "off", "no", "none", "default"):
                return False
            if raw in ("1", "true", "yes", "on"):
                return True
            return None  # an invalid/unknown setting cannot excuse a mismatch
        # SMALL_STAGES_FUSED, HC_PREMIX_KERNEL and DRAFT_HEAD_BF16 use this
        # permissive resolver (not the strict bool resolver's none/default aliases).
        return raw not in ("", "0", "false", "off", "no", "auto")

    left, right = effective_env(base), effective_env(candidate)
    if state(KV_BOUNDED_ENV, left.get(KV_BOUNDED_ENV)) or state(
        KV_BOUNDED_ENV, right.get(KV_BOUNDED_ENV)
    ):
        return []
    changed = []
    for key in ROUNDING_CLASS_ENVS:
        before, after = state(key, left.get(key)), state(key, right.get(key))
        if before is not None and after is not None and before != after:
            changed.append(key)
    return changed


def _load_bench_module():
    """Import ``bench_standard_shape`` (a sibling script) for its cell harness.

    Its top level imports only the standard library, so this is CPU-safe.
    """
    path = Path(__file__).resolve().parent / "bench_standard_shape.py"
    spec = importlib.util.spec_from_file_location("_dsv41_bench_standard_shape", path)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise ImportError(f"cannot load bench harness at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_dspark_verify_chunks(raw: str) -> tuple[int, ...]:
    try:
        chunks = tuple(int(part.strip()) for part in str(raw).split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "DSpark verify chunks must be comma-separated positive integers"
        ) from exc
    if not chunks or any(value <= 0 for value in chunks):
        raise argparse.ArgumentTypeError(
            "DSpark verify chunks must be comma-separated positive integers"
        )
    return chunks


def _parse_persistent_slots_by_layer(raw: str) -> tuple[int, ...]:
    try:
        capacities = tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "persistent layer capacities must be comma-separated integers"
        ) from exc
    if not capacities or any(capacity < 0 for capacity in capacities):
        raise argparse.ArgumentTypeError(
            "persistent layer capacities must be comma-separated nonnegative integers"
        )
    return capacities


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--context-tokens", type=int, default=1024, choices=(1024, 16384))
    p.add_argument("--decode-tokens", type=int, default=256)
    # W90: GPU DVFS / utilization telemetry + post-prefill cooldown.
    p.add_argument(
        "--utilization", action="store_true",
        help="sample macmon (gpu freq/power/busy, temps) in a background thread over "
             "the timed DECODE and write a 'utilization' block (min/mean/max + series) "
             "+ a one-line census (W90: the GPU DVFS-downclock floor discriminator).",
    )
    p.add_argument(
        "--util-interval-ms", type=int, default=2000,
        help="macmon sampling interval in ms for --utilization (default 2000, ~2 s).",
    )
    p.add_argument(
        "--cooldown-s", type=float, default=0.0,
        help="idle N seconds AFTER prefill and BEFORE the timed decode (TTFT, from "
             "the prefill, is unaffected); logged as a 'cooldown' block (W90).",
    )
    p.add_argument(
        "--decode-mode",
        choices=("ar", "dspark"),
        default="ar",
        help=(
            "Decode lane. 'ar' is greedy target-only autoregression (the default, "
            "byte-identical baseline for the levers). 'dspark' runs the "
            "DSpark-DIRECT speculative loop (W57, mtplx.models."
            "deepseek_v41_dspark_decode.dspark_generate); it also runs the AR "
            "lane and ASSERTS the greedy token ids are byte-identical, and records "
            "tokens/cycle + accept-by-depth in the receipt under 'dspark'."
        ),
    )
    p.add_argument(
        "--dspark-depth",
        type=int,
        default=3,
        help="draft block width K per DSpark-DIRECT cycle (--decode-mode dspark)",
    )
    p.add_argument(
        "--dspark-verify-chunks",
        type=_parse_dspark_verify_chunks,
        default=None,
        metavar="ROWS[,ROWS...]",
        help=(
            "Construction-time partition of the K+1 target verify rows. The "
            "default is one full verify; e.g. --dspark-depth 5 with 3,3 stops "
            "before the second target forward after an early rejection."
        ),
    )
    p.add_argument(
        "--dspark-require-lossless",
        action="store_true",
        help=(
            "W77: restore the hard abort when the DSpark-DIRECT greedy stream "
            "diverges from AR (for exactness-class arms whose ship bar IS byte "
            "identity). Default OFF: a divergence is classified (tie_flip vs "
            "divergent) into the receipt and both streams decode to full length."
        ),
    )
    p.add_argument(
        "--dspark-require-tie-class",
        action="store_true",
        help=(
            "Require a DSpark stream to be byte-identical to AR or have its "
            "first mismatch classified as an index-matched tie_flip. A genuine "
            "or suspect divergence is still written to the receipt, then makes "
            "the runner exit nonzero."
        ),
    )
    p.add_argument(
        "--dspark-tie-margin",
        type=float,
        default=DSPARK_TIE_MARGIN_DEFAULT,
        help=(
            "W77: AR top-2 logit gap (logit units) below which a greedy DSpark "
            "divergence is classed a tie-break flip (rounding-class near-tie) "
            f"rather than 'divergent'. Default {DSPARK_TIE_MARGIN_DEFAULT} (3x the "
            "bf16-class head-GEMV floor)."
        ),
    )
    p.add_argument(
        "--device-sample",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "W63 / K32: run the AR (--decode-mode ar) reference decode with the "
            "device-side sampling one-step-lag pipeline (the sampled token stays "
            "on device and feeds the next embedding directly; the host reads ids "
            "one step behind an already-submitted forward). Greedy is "
            "byte-identical to the classic argmax loop. Default follows "
            "MTPLX_DSV41_DEVICE_SAMPLE (off)."
        ),
    )
    p.add_argument(
        "--with-mtp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Force the model to load with the DSpark MTP head (with_mtp=True) and "
            "reprice its residents. Implied by --decode-mode dspark. On "
            "--decode-mode ar this measures 'AR + head loaded' so window 25 can "
            "A/B it against plain 'AR' (--no-with-mtp) and isolate the head's "
            "expert-cache cost from the verify routing phase."
        ),
    )
    p.add_argument(
        "--reprice",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Legacy compatibility flag. The loader always prices the selected "
            "MTP weights and caches before allocating expert slots; neither "
            "--reprice nor --no-reprice changes the total memory envelope."
        ),
    )
    p.add_argument(
        "--arms",
        nargs="+",
        default=["control", "shared_overlap"],
        help="preset names from ARM_PRESETS (control, shared_overlap, layer_major, "
        "sinkhorn_metal, hc_compile, switch_fastpath, switch_fastpath_b, "
        "attn_compile, attn_win_memo, draft_compile, device_route, "
        "prefill_dense_experts, "
        "dense_min32, dense_batch16, dense_f32, both, all_levers, stack_a, "
        "stack_b, head_bf16, head_mxfp8, head_q8, score_bf16, score_chunked, "
        "score_bf16_chunked, score_lean, prefill_fast, prefill_lean, "
        "selected_keys, prefill_lean_sel, softmax_kernel, prefill_lean_k28, "
        "prefill_best, prefill_best_nok28, decode_attn_kernel)",
    )
    p.add_argument("--out", type=Path, required=True, help="append-only JSONL receipt")
    # Prompt build: mirrors bench_standard_shape.py exactly, so that
    # ``bench._prompt_args(args, ctx)`` reads the same fields with the same
    # defaults and the reused ``build_prompt`` produces the identical standard-
    # shape prompt (deterministic prefill_bench coding-agent prompt + reference
    # BOS). Missing these was the W11 crash (_prompt_args hit args.prompt).
    p.add_argument(
        "--prompt",
        default=None,
        help="literal prompt text; overrides the prefill_bench builder "
        "(default None = build the deterministic prefill_bench prompt).",
    )
    p.add_argument("--prompt-format", default="raw", choices=("raw", "chat"))
    p.add_argument(
        "--bos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="prepend the reference BOS (id --bos-id, default 0). The artifact "
        "tokenizer does not add it; the reference always does (W8_REPORT.md).",
    )
    p.add_argument("--bos-id", type=int, default=DEFAULT_BOS_ID)
    p.add_argument(
        "--prompt-ids-file",
        default=None,
        help="run every arm on the EXACT server token ids exported by "
        "scripts/fable/server_cell_bench.py --prompt-ids-out (the Qwen-PR sized "
        "cells), so the in-process A/B uses the SAME ids as the served cell. "
        "Overrides the prefill_bench builder AND --bos (the exported ids already "
        "are what the server saw; DSV4.1 served ids carry NO BOS). Selects "
        "(cell=sweep, target_tokens==--context-tokens, seed==--prompt-seed). "
        "Default None keeps the built prompt so old receipts stay comparable.",
    )
    p.add_argument(
        "--prompt-seed",
        type=int,
        default=None,
        help="which seed's ids to take from --prompt-ids-file (e.g. 20260829).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="CPU-only test double: apply each arm's env and build the standard-"
        "shape prompt with bench's fake tokenizer -- no model load, no MLX/Metal. "
        "Proves argument resolution, prompt build and per-arm env application.",
    )
    p.add_argument(
        "--syncs",
        type=int,
        default=0,
        metavar="N",
        help="extra probe pass of N decode steps per arm with the route-stage "
        "probe ON, to census host syncs/token (0 = skip; the probe inflates "
        "timing, so it never touches the reported tok/s pass)",
    )
    p.add_argument(
        "--stage-timing",
        action="store_true",
        default=False,
        help="extra W37 decode pass with MTPLX_DSV41_STAGE_TIMING fences per "
        "stage (embed / attention-by-CSA-mode / engram / HC / gate+barrier / "
        "routed switch / shared / combine / head / sample), recorded into the "
        "receipt as ``stage_timing`` (mean ms/token per stage + counts). The "
        "fences inflate absolute time, so this pass never touches the reported "
        "tok/s; the ratios between stages are the signal.",
    )
    p.add_argument(
        "--stage-timing-steps",
        type=int,
        default=None,
        metavar="N",
        help="decode steps for the --stage-timing pass (default: --decode-tokens).",
    )
    p.add_argument(
        "--warm-repeat",
        action="store_true",
        default=False,
        help="after the measured cold pass, re-run prefill+decode of the SAME "
        "prompt a second time in the same process (expert cache + engram warm, "
        "fresh KV / engram history via model.make_cache()) and record the second "
        "pass's decode/prefill tok/s + TTFT as ``warm_*``; bounds the no-miss "
        "ceiling. Token ids must match the cold pass (recorded, not asserted).",
    )
    p.add_argument(
        "--prefill-stage-timing",
        action="store_true",
        default=False,
        help="extra W47 fenced PREFILL pass -> receipt ``prefill_stage_timing``: "
        "per (chunk, layer-type) brackets for attention (qkv_proj / select / "
        "score / cache_append / compress_append), HC, gate+top-k, streamed-switch "
        "breakdown (admission / route-plan / miss-submit) + gather total, shared "
        "expert, combine, engram, with per-chunk aggregation.  Reads the schedule "
        "(chunk-major vs layer-major) from the arm's MTPLX_DSV41_PREFILL_LAYER_MAJOR "
        "and the chunk from MTPLX_DSV41_PREFILL_CHUNK.  Fences inflate absolute "
        "time; ratios are the signal.  Use with --context-tokens 16384.",
    )
    # W62: the plan DERIVES from David's TOTAL box budget
    # (MTPLX_DSV41_BOX_BUDGET_GB, default 100) minus the macOS floor, the measured
    # host overhead, and the allocator cache limit
    # (mtplx.deepseek_v41_memory_profile.derive_plan_from_budget) -- no more
    # hand-picked 82/92 that drove the box into the panic zone (window 28).
    # --memory-limit-gib stays as the explicit override.
    p.add_argument(
        "--memory-limit-gib",
        type=float,
        default=None,
        help="explicit plan ceiling GiB (override); default: derive from "
        "the 110 decimal GB target (legacy --box-budget-gib is explicit).",
    )
    p.add_argument(
        "--box-budget-gib",
        type=float,
        default=None,
        help="Explicit legacy total box-use budget in GiB. When omitted, "
        "the default allocation uses --box-target-gb (110 decimal GB).",
    )
    # Target and baseline are decimal GB; cache/reserve overrides are GiB.
    p.add_argument("--box-target-gb", type=float, default=None, metavar="GB",
        help="Total physical RAM target in decimal GB (default 110 or "
        "MTPLX_DSV41_BOX_TARGET_GB). Reserves system baseline and Python capacity; "
        "conflicts with --memory-limit-gib.")
    p.add_argument("--box-baseline-gb", type=float, default=None, metavar="GB",
        help="Pre-load physical system usage in decimal GB, including file cache. "
        "The GPU guard's live baseline takes precedence; otherwise measured now.")
    p.add_argument("--host-overhead-gib", type=float, default=None, metavar="GIB",
        help="Host allocation reserve in GiB (default at least 2, increased to "
        "cover full Python cache capacity and index metadata).")
    p.add_argument(
        "--allocator-cache-gib",
        type=float,
        default=None,
        metavar="GIB",
        help="MLX freed-buffer cache bound in GiB (set_cache_limit); reserved out of "
        "the engine budget so the LRU holds across misses (target-mode default 2, "
        "MTPLX_DSV41_MLX_CACHE_LIMIT_GIB).",
    )
    p.add_argument(
        "--active-overshoot-gib",
        type=float,
        default=None,
        metavar="GIB",
        help="active memory at decode start ABOVE the plan (active_start - plan) in GiB; "
        "with the cache room it bounds the decode-regime reserve max(band, active + cache) "
        "(target-mode default 1.45, MTPLX_DSV41_ACTIVE_OVERSHOOT_GIB).",
    )
    p.add_argument(
        "--transient-band-gib",
        type=float,
        default=None,
        metavar="GIB",
        help="prefill/decode transient band in GiB (mlx_peak - active_at_decode_start) "
        "reserved below the allocator limit when sizing the persistent slots "
        "(target-mode default 10, MTPLX_DSV41_TRANSIENT_BAND_GIB).",
    )
    p.add_argument(
        "--runtime-reserve-gib",
        type=float,
        default=None,
        metavar="GIB",
        help=(
            "in-plan MLX transient reserve (default 7 GiB). The 110 GB staged "
            "capacity screen uses 3 GiB for cap 89 and may use 2 GiB for cap 91 "
            "only after measured headroom; values below 2 GiB are refused."
        ),
    )
    p.add_argument(
        "--mlx-limit-headroom-gib",
        type=float,
        default=None,
        metavar="GIB",
        help="W118 (H7): raise ONLY the MLX allocator soft limit (mx.set_memory_limit) "
        "by this many GiB ABOVE the derived limit, WITHOUT changing what is resident or "
        "the expert-cache slot plan -- so bytes/routing/outputs are byte-identical. Maps "
        "to MTPLX_DSV41_MLX_LIMIT_HEADROOM_GIB (read at use by apply_mlx_memory_cap). "
        "Default: the env / preset value, else 0.",
    )
    p.add_argument(
        "--memory-plan-from",
        type=Path,
        default=None,
        metavar="PATH",
        help="PIN the target + measured components from a derived-target-plan.json "
        "sidecar (written by an earlier target run) instead of resolving live, so every "
        "A/B arm / invocation uses the SAME engine budget (reproducible residency).",
    )
    p.add_argument(
        "--memory-profile",
        action="store_true",
        default=False,
        help="capture the W62 memory profile (mlx/process/box/plan snapshots) at "
        "load end, after prefill, and every --memory-profile-every decode "
        "tokens; writes receipt['memory_profile'] + a rendered table.",
    )
    p.add_argument(
        "--memory-profile-every",
        type=int,
        default=64,
        metavar="N",
        help="decode-token interval for per-token memory snapshots (default 64).",
    )
    p.add_argument("--expert-cache-limit-gib", type=float, default=None)
    p.add_argument(
        "--apply-memory-cap", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--slot-layout", default="component-banks")
    p.add_argument(
        "--transient-slots",
        type=int,
        default=None,
        metavar="N",
        help=(
            "transient (streaming) slots per layer. DEFAULT: resolve from the "
            "served profile deepseek-v41-mxfp4-75 so the in-process bench runs "
            "the SAME slot plan as production, instead of the loader's unset "
            "default (spec.top_k=6) that starved the W66/W81 verify fast path in "
            "window 31. Pass an explicit N to override (e.g. --transient-slots 24 "
            "fits a 4-row verify's <=24 unique in one admission at 60 GiB)."
        ),
    )
    p.add_argument(
        "--cache-policy",
        choices=(
            "frequency",
            "lru",
            "transition-window",
            "transition-window-tuned",
        ),
        default=None,
        help=(
            "expert-cache admission policy. DEFAULT: resolve from --expert-profile; "
            "transition-window is the bounded DeepSeek-V4.1 causal-policy arm."
        ),
    )
    p.add_argument(
        "--persistent-slots-by-layer",
        type=_parse_persistent_slots_by_layer,
        default=None,
        metavar="N0,N1,...",
        help=(
            "construction-time persistent capacities ordered by routed layer; "
            "the exact vector must fit the resolved expert-cache byte budget"
        ),
    )
    p.add_argument(
        "--decode-miss-records-per-part",
        type=int,
        default=None,
        metavar="N",
        help=(
            "construction-time decode scheduling arm: submit every miss part "
            "immediately but expose completion after at most N records. Requires "
            "the v2 overlap-miss route. Default None keeps one layer-wide part."
        ),
    )
    p.add_argument(
        "--verify-shared-overlap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "construction-time M6 arm: enqueue the resident shared MLP after "
            "verify demand reads are submitted and before waiting for them. "
            "Default OFF is the unchanged routed-then-shared control."
        ),
    )
    p.add_argument(
        "--expert-profile",
        default="deepseek-v41-mxfp4-75",
        help=(
            "profile whose plan fields (transient_slots, split_route_release, "
            "prefetch_slots, cache_policy) seed the in-process runtime when the matching flag "
            "is unset; 'none' disables profile resolution (loader defaults)."
        ),
    )
    p.add_argument("--max-kv", type=int, default=4096)
    p.add_argument("--admit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--admission-receipt", type=Path, default=None)
    p.add_argument(
        "--verify-record-hashes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--seed", type=int, default=0)
    # W113: cell-prompt guard + EOS surfacing.
    p.add_argument(
        "--allow-raw-prompt",
        action="store_true",
        default=False,
        help="ESCAPE HATCH (diagnostics): let a cell16k_* arm / --context-tokens "
        "16384 run MEASURE the RAW prefill_bench builder prompt instead of the "
        "standard chat-templated cell. Skips BOTH the W113 guard refusal and the "
        "ctx-16384 --prompt-ids-file auto-default; stamps prompt_source='raw-"
        "builder' loudly. The raw prompt's greedy first token is EOS, so a served "
        "path returns EMPTY.",
    )
    p.add_argument(
        "--stop-on-eos",
        action="store_true",
        default=False,
        help="served-parity: stop the AR decode (and, in --decode-mode dspark, the "
        "speculative decode) at the EOS id and report decode_tok_s over the tokens "
        "ACTUALLY generated. Default OFF keeps the full fixed-step decode so the "
        "throughput numbers are unchanged. Mirrors the serve_bench_1k W18 guard.",
    )
    p.add_argument(
        "--eos-id",
        type=int,
        default=None,
        help="override the EOS token id used by --stop-on-eos and the EOS-surfacing "
        "receipt fields (default: resolve from the tokenizer files, no model load).",
    )
    return p


def _apply_arm_env(arm: str) -> None:
    if arm not in ARM_PRESETS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARM_PRESETS)}")
    for key, value in ARM_PRESETS[arm].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# --------------------------------------------------------------------------
# W113 cell-prompt guard + prompt provenance + EOS surfacing
# --------------------------------------------------------------------------
# Windows 39-42 MEASURED the RAW prefill_bench builder prompt (BOS + a 96x
# ``# file_N.py`` filler ladder + DEFAULT_FINAL_REQUEST = 16,385 tokens, no chat
# template, no generation prompt) whose greedy first token is EOS (id 1); the
# classic/device decode loops had no EOS check, so 256 FORCED post-EOS filler
# tokens were timed and a served (EOS-honouring) path would have returned an
# EMPTY answer.  The standard 16K cell is the chat-templated ids file
# STANDARD_CELL16K_PROMPT_IDS, reached ONLY via --prompt-ids-file
# (bench._prompt_ids_override: exactly 16,384 ids, ending <｜Assistant｜></think>,
# no BOS re-prepend).  A receipt tells the two apart by ``prompt_source`` /
# ``prompt_tokens`` (16,384 file vs 16,385 raw+BOS) / ``prompt_chat_templated``.
# The guard (real-measurement path) refuses to run a cell16k_* arm or a
# --context-tokens 16384 run on the raw builder unless --allow-raw-prompt, and
# defaults --prompt-ids-file to the standard file so launchers get the cell.


def _repo_root() -> Path:
    """Repo/worktree root: ``scripts/deepseek_v41/<this>.py`` -> ``parents[2]``."""
    return Path(__file__).resolve().parents[2]


def _standard_cell16k_prompt_path() -> Path:
    """Absolute path to the standard 16K cell ids file (repo-root relative)."""
    return _repo_root() / STANDARD_CELL16K_PROMPT_IDS


def _cell16k_arm(name) -> bool:
    """True for the standard-cell arms.  W113 LOW-c: the bare ``cell16k`` preset
    (no trailing underscore) is a 16K-cell arm too, so match it as well as the
    ``cell16k_*`` family."""
    s = str(name)
    return s == "cell16k" or s.startswith("cell16k_")


_SPECIAL_IDS_CACHE: dict = {}


def _special_token_ids(model_path) -> dict:
    """Resolve special-token ids from the tokenizer files WITHOUT a model/MLX.

    Reads ``tokenizer_config.json`` (bos/eos/pad token contents) and
    ``tokenizer.json`` (the ``added_tokens`` content->id map) under ``model_path``
    and returns ``{bos, eos, pad, assistant, user, system, think, end_think}`` for
    the ids that resolve.  CPU-only, read-only, cached by path; ``{}`` if the files
    cannot be read (best-effort -- callers degrade to ``None`` flags).  This is the
    "detect via tokenizer special ids, no model" path W113 needs when
    --prompt-ids-file skips the tokenizer load.
    """
    key = str(model_path)
    if key in _SPECIAL_IDS_CACHE:
        return _SPECIAL_IDS_CACHE[key]
    out: dict = {}
    try:
        base = Path(model_path).expanduser()
        cfg = json.loads((base / "tokenizer_config.json").read_text())
        tok = json.loads((base / "tokenizer.json").read_text())

        def _content(v):
            return v.get("content") if isinstance(v, dict) else v

        by_content: dict = {}
        for t in (tok.get("added_tokens") or []):
            c = t.get("content")
            if c is not None and t.get("id") is not None:
                by_content[c] = int(t["id"])
        wanted = {
            "bos": _content(cfg.get("bos_token")),
            "eos": _content(cfg.get("eos_token")),
            "pad": _content(cfg.get("pad_token")),
            "assistant": "<｜Assistant｜>",
            "user": "<｜User｜>",
            "system": "<｜System｜>",
            "think": "<think>",
            "end_think": "</think>",
        }
        for name, content in wanted.items():
            if content is not None and content in by_content:
                out[name] = by_content[content]
        # config-level id fields win when present (both None for this model).
        if cfg.get("eos_token_id") is not None:
            out["eos"] = int(cfg["eos_token_id"])
        if cfg.get("bos_token_id") is not None:
            out["bos"] = int(cfg["bos_token_id"])
    except Exception:  # pragma: no cover - defensive (missing/unreadable files)
        out = {}
    _SPECIAL_IDS_CACHE[key] = out
    return out


def _resolve_eos_id(args):
    """The EOS token id: ``--eos-id`` override, else the tokenizer-file eos id,
    else ``None`` (EOS surfacing then records ``None`` best-effort)."""
    ov = getattr(args, "eos_id", None)
    if ov is not None:
        return int(ov)
    eid = _special_token_ids(getattr(args, "model", None)).get("eos")
    return int(eid) if eid is not None else None


def _require_eos_id_for_stop(stop_on_eos, eos_id) -> None:
    """W113 MEDIUM-3: refuse ``--stop-on-eos`` when no EOS id could be resolved.

    Without this the flag silently no-ops (nothing stops the decode) while the
    receipt still stamps ``stop_on_eos: true`` -- a served-parity run that did not
    behave like the served path.  Raised in ``_run_arm`` before the model load.
    """
    if stop_on_eos and eos_id is None:
        raise SystemExit(
            "[ab] --stop-on-eos needs an EOS id but none could be resolved from "
            "the tokenizer files (tokenizer_config.json + tokenizer.json) under "
            "--model; pass --eos-id <id> (DeepSeek-V4.1 EOS is id 1)."
        )


def _prompt_chat_templated(prompt_ids, special):
    """Best-effort: is ``prompt_ids`` chat-templated WITH a generation prompt?

    ``True`` when the ids contain the ``<｜Assistant｜>`` special id AND end with a
    generation prompt -- nothing follows the final assistant marker except the
    think open/close markers (the assistant turn is opened but not yet answered,
    e.g. ``... <｜Assistant｜></think>``).  ``False`` for the raw builder prompt (no
    assistant marker) or a prompt whose assistant turn already has content after
    the marker.  ``None`` when the assistant special id is unavailable.
    """
    assistant = (special or {}).get("assistant")
    if assistant is None:
        return None
    ids = [int(t) for t in prompt_ids]
    if assistant not in ids:
        return False
    last = max(i for i, t in enumerate(ids) if t == assistant)
    opener = {
        x
        for x in ((special or {}).get("think"), (special or {}).get("end_think"))
        if x is not None
    }
    return all(t in opener for t in ids[last + 1:])


def _prompt_provenance(args, prompt_ids, prompt_meta) -> dict:
    """The W113 receipt stamps that tell the standard cell from the raw builder.

    ``prompt_source`` is ``"prompt-ids-file"`` when --prompt-ids-file fed the exact
    served ids, else ``"raw-builder"`` (the prefill_bench builder).
    ``prompt_ids_sha256`` is the sha of the PROMPT ids (NOT the generated ids --
    that is ``token_ids_sha256``).  ``prompt_chat_templated`` is detected from the
    tokenizer special ids (no model).  ``prompt_build`` carries the resolver's own
    metadata block (its native ``prompt_source`` field is "prefill_bench"/"literal"
    for the builder or "prompt-ids-file" for the override -- a finer label than the
    two-value top-level one).
    """
    src = (
        "prompt-ids-file"
        if getattr(args, "prompt_ids_file", None)
        else "raw-builder"
    )
    special = _special_token_ids(getattr(args, "model", None))
    ids_sha = hashlib.sha256(
        json.dumps([int(t) for t in prompt_ids]).encode("utf-8")
    ).hexdigest()
    return {
        "prompt_source": src,
        "prompt_ids_file": getattr(args, "prompt_ids_file", None),
        "prompt_ids_sha256": ids_sha,
        "prompt_seed": getattr(args, "prompt_seed", None),
        "prompt_tokens": len(prompt_ids),
        "prompt_chat_templated": _prompt_chat_templated(prompt_ids, special),
        "allow_raw_prompt": bool(getattr(args, "allow_raw_prompt", False)),
        "prompt_build": prompt_meta,
    }


def _eos_surfacing(generated_ids, eos_id) -> dict:
    """EOS surfacing over a generated id stream (W113).

    ``first_token_eos`` -- the FIRST generated token is EOS (a served path would
    then return an EMPTY answer).  ``eos_index`` -- the first position of the EOS
    id in the stream, or ``None``.  ``tokens_before_eos`` -- ``eos_index`` if
    present, else the whole stream length.  ``answer_valid`` -- ``not
    first_token_eos``: the answer is non-empty iff the first token is not EOS.
    This is CAP-INDEPENDENT -- it does not change whether --stop-on-eos truncated
    the stream or the full fixed-step decode ran -- unlike the withdrawn
    ``eos_index > 0.5*N`` rule, which flipped a correct SHORT answer (e.g. a valid
    60-token answer) to invalid once --stop-on-eos shrank N.  ``answer_truncated``
    -- ``eos_index is None``: the decode hit the token cap without the model
    emitting EOS (the answer may be cut off).  ``post_eos_tokens_timed`` -- when
    EOS is present, the number of FORCED post-EOS tokens that were still timed
    (``n_generated - eos_index - 1``; the wasted filler a served path would never
    produce -- 256 on the windows 39-42 raw prompt, whose EOS was at index 0);
    ``0`` when EOS is absent.  All fields are ``None`` when ``eos_id`` is unknown.
    """
    ids = [int(t) for t in (generated_ids or [])]
    total = len(ids)
    if eos_id is None:
        return {
            "first_token_eos": None,
            "eos_index": None,
            "tokens_before_eos": None,
            "answer_valid": None,
            "answer_truncated": None,
            "post_eos_tokens_timed": None,
            "eos_id": None,
            "n_generated": total,
        }
    eos_id = int(eos_id)
    eos_index = next((i for i, t in enumerate(ids) if t == eos_id), None)
    first_token_eos = bool(ids and ids[0] == eos_id)
    return {
        "first_token_eos": first_token_eos,
        "eos_index": eos_index,
        "tokens_before_eos": int(eos_index if eos_index is not None else total),
        "answer_valid": not first_token_eos,
        "answer_truncated": eos_index is None,
        "post_eos_tokens_timed": (
            (total - eos_index - 1) if eos_index is not None else 0
        ),
        "eos_id": eos_id,
        "n_generated": total,
    }


def _warn_if_first_token_eos(arm, lane, surf) -> None:
    """The loud W113 warning when the first generated token is EOS."""
    if surf.get("first_token_eos"):
        print(
            "[ab] " + "!" * 8 + " WARNING: first generated token is EOS -- answer "
            "would be EMPTY on a served path " + "!" * 8
            + f" (arm {arm!r}, {lane})",
            flush=True,
        )


def _verify_standard_cell_prompt(path) -> None:
    """W113 LOW-a: pin the auto-defaulted standard cell to its expected prompt sha.

    Selects the ``(cell=sweep, target_tokens=16384, seed=STANDARD_CELL16K_PROMPT_
    SEED)`` entry and refuses (``SystemExit``) if the sha of its prompt ids does
    not match ``STANDARD_CELL16K_PROMPT_SHA256`` -- so a swapped/edited standard
    file is caught before it is silently measured.  Only the auto-defaulted file is
    pinned; an explicit --prompt-ids-file is the operator's own choice.
    """
    try:
        data = json.loads(Path(path).read_text())
        entry = next(
            e
            for e in (data.get("prompts") or [])
            if str(e.get("cell")) == "sweep"
            and int(e.get("target_tokens") or 0) == 16384
            and e.get("seed") == STANDARD_CELL16K_PROMPT_SEED
        )
        ids = [int(t) for t in entry["token_ids"]]
    except (StopIteration, KeyError, ValueError, TypeError,
            json.JSONDecodeError, OSError) as exc:
        raise SystemExit(
            f"[ab] W113: cannot verify the standard cell file {path} "
            f"(cell=sweep, target_tokens=16384, seed={STANDARD_CELL16K_PROMPT_SEED}"
            f"): {exc!r}. Pass --prompt-ids-file explicitly or --allow-raw-prompt."
        )
    sha = hashlib.sha256(json.dumps(ids).encode("utf-8")).hexdigest()
    if sha != STANDARD_CELL16K_PROMPT_SHA256:
        raise SystemExit(
            f"[ab] W113: the standard cell file {path} prompt-ids sha {sha} does "
            f"NOT match the pinned {STANDARD_CELL16K_PROMPT_SHA256} -- the file was "
            "edited/swapped. Pass --prompt-ids-file explicitly (if intended) or "
            "--allow-raw-prompt."
        )


def _apply_cell_prompt_guard(args) -> None:
    """W113 cell-prompt guard for the REAL measurement path (see the block note).

    - Auto-default --prompt-ids-file to the standard 16K cell file when
      --context-tokens 16384 and that file exists (repo-root relative), so a
      launcher gets the standard cell without passing the path.
    - REFUSE to run any ``cell16k_*`` arm or any --context-tokens 16384 run without
      --prompt-ids-file (raise ``SystemExit`` naming the standard file).
    - --allow-raw-prompt is the loud diagnostics escape hatch: it skips BOTH the
      refusal and the auto-default and runs the raw builder.

    Precedence: an explicit --prompt-ids-file wins over everything; then
    --allow-raw-prompt (raw builder, loud); then the ctx-16384 auto-default; then
    the refusal.  No-op when the run is neither a cell16k_* arm nor 16384-ctx.
    """
    ctx16k = int(getattr(args, "context_tokens", 0) or 0) == 16384
    cell_arms = [a for a in (getattr(args, "arms", None) or []) if _cell16k_arm(a)]
    if not (ctx16k or cell_arms):
        return
    if getattr(args, "prompt_ids_file", None):  # explicit file always wins
        return
    if getattr(args, "allow_raw_prompt", False):
        print(
            "[ab] " + "!" * 8 + " --allow-raw-prompt: measuring the RAW builder "
            "prompt on a 16K cell " + "!" * 8 + "\n"
            "[ab] WARNING: this is the DIAGNOSTIC raw prefill_bench prompt (no chat "
            "template, no generation prompt); its greedy first token is EOS so a "
            "served path returns EMPTY. NOT the standard cell. prompt_source is "
            "stamped 'raw-builder'.",
            flush=True,
        )
        return
    std = _standard_cell16k_prompt_path()
    if ctx16k and std.exists():
        # W113 LOW-a: pin the file to its expected prompt sha (refuse on mismatch),
        # then stamp the seed so the receipt's prompt_seed is not left null.
        _verify_standard_cell_prompt(std)
        args.prompt_ids_file = str(std)
        if getattr(args, "prompt_seed", None) is None:
            args.prompt_seed = STANDARD_CELL16K_PROMPT_SEED
        print(
            f"[ab] W113: defaulted --prompt-ids-file to the standard 16K cell {std} "
            f"(--prompt-seed {args.prompt_seed}, prompt-ids sha pinned; "
            "--allow-raw-prompt for the raw builder)",
            flush=True,
        )
        return
    reason = []
    if cell_arms:
        reason.append(f"cell16k_* arms {cell_arms}")
    if ctx16k:
        reason.append("--context-tokens 16384")
    raise SystemExit(
        "[ab] REFUSED (W113 cell-prompt guard): " + " and ".join(reason) + " require "
        "the standard chat-templated 16K cell prompt, but --prompt-ids-file was not "
        f"given and the standard file was not found at {std}. Pass --prompt-ids-file "
        f"<{STANDARD_CELL16K_PROMPT_IDS}> --prompt-seed 20260829 (schema "
        "mtplx-server-cell-prompt-ids-v1, cell=sweep, target_tokens=16384), or "
        "--allow-raw-prompt to measure the diagnostic raw builder (its greedy first "
        "token is EOS -> empty served answer)."
    )


def _arm_env_snapshot() -> dict:
    """Every lever env key's current value (None = unset)."""
    return {key: os.environ.get(key) for key in ALL_LEVER_ENVS}


def _device_sample_resolved(args) -> bool:
    """W63 / K32: True when the AR reference decode runs the device-sample
    one-step-lag pipeline. ``--device-sample`` overrides; the ``None`` default
    follows ``MTPLX_DSV41_DEVICE_SAMPLE``. The env truthiness is inlined here
    (mirrors ``deepseek_v41_dspark_decode.device_sample_enabled``) so --dry-run
    resolves it WITHOUT importing the model module (which imports mlx.core)."""
    flag = getattr(args, "device_sample", None)
    if flag is not None:
        return bool(flag)
    return os.environ.get("MTPLX_DSV41_DEVICE_SAMPLE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _dry_run_arm(args, arm, bench) -> dict:
    """CPU-only arm double (no model, no MLX/Metal): apply the arm's env, then
    build the standard-shape prompt with bench's fake tokenizer so the prompt-
    build metadata is byte-for-byte identical to bench_standard_shape's
    ``--dry-run`` for the same context cell."""
    build_prompt = bench._load_build_prompt()
    tokenizer = bench._FakeTokenizer()
    prompt_ids, prompt_meta = bench._resolve_prompt(
        args, tokenizer, build_prompt, args.context_tokens
    )
    return {
        "arm": arm,
        "dry_run": True,
        # W97 (review item 7): rounding-class flag + the reason keys the arm arms
        # (see ROUNDING_CLASS_ENVS).  True => a token-id sha mismatch is EXPECTED.
        "rounding_class": _is_rounding_class(arm),
        "rounding_class_keys": _rounding_class_keys(arm),
        "overlap_env": os.environ.get(OVERLAP_ENV),
        "arm_env": _arm_env_snapshot(),
        # K14 (W63): the MLX command-buffer MB cap this arm pins (None = MLX
        # default). MLX binds it at init, so a real A/B exports it per process;
        # the receipt records the arm's pinned value for reproducibility.
        "mlx_max_mb_per_buffer": os.environ.get(MLX_MAX_MB_PER_BUFFER_ENV),
        # W63 / K32: whether the AR decode would run the device-sample pipeline.
        "device_sample": _device_sample_resolved(args),
        "context_tokens": int(args.context_tokens),
        "decode_tokens": int(args.decode_tokens),
        # W113 prompt provenance stamps (prompt_source / prompt_ids_file /
        # prompt_ids_sha256 / prompt_seed / prompt_tokens / prompt_chat_templated /
        # allow_raw_prompt / prompt_build).  In --dry-run this is always the raw
        # builder (the guard/auto-default run only on the real path), so
        # prompt_source == "raw-builder" and prompt_chat_templated is False.
        **_prompt_provenance(args, prompt_ids, prompt_meta),
        # W37 pass toggles resolved offline (no model / MLX): proves the flags
        # thread through argument resolution before a GPU window burns on them.
        "stage_timing": bool(getattr(args, "stage_timing", False)),
        "stage_timing_steps": (
            int(args.stage_timing_steps)
            if getattr(args, "stage_timing_steps", None) is not None
            else int(args.decode_tokens)
        ),
        "warm_repeat": bool(getattr(args, "warm_repeat", False)),
        "prefill_stage_timing": bool(getattr(args, "prefill_stage_timing", False)),
        "prefill_layer_major": os.environ.get(LAYER_MAJOR_ENV),
        "prefill_chunk_env": os.environ.get("MTPLX_DSV41_PREFILL_CHUNK"),
    }


# --------------------------------------------------------------------------
# W121: box-target -> engine-budget derivation.  David ("rebalance it so kv isnt
# terrible" / "track the memory usage so we can efficiently use it") replaced the
# W106 budget-total forecast (total - system_used_at_start(vm_stat) - non_metal
# - kv_growth - safety - plan_overshoot) with the target model: the runtime sets the
# allocator/wired limit = target - baseline - host reserve and set_cache_limit(cache)
# (mtplx.expert_runtime.apply_mlx_memory_cap), and the bench sizes the ENGINE budget
# (persistent expert slots) = allocator_limit - transient_band so the slots FILL the
# target while active + transient stays under the allocator limit.  No static forecasts.
# --------------------------------------------------------------------------


def _gb_to_gib(gb: float) -> float:
    """Decimal GB -> GiB (the boundary conversion for the deprecated ``-gb``
    aliases). 1 GB = 1e9 bytes; 1 GiB = 2**30 bytes."""

    return float(gb) * 1_000_000_000 / GIB


def _resolve_gib_flag(args, gib_attr, gb_attr, default, flag_label):
    """Resolve a GiB quantity from the canonical ``-gib`` flag, else the deprecated
    ``-gb`` alias (decimal GB, converted to GiB with a warning), else ``default``.
    Refuses if BOTH are set (ambiguous)."""

    gib = getattr(args, gib_attr, None)
    gb = getattr(args, gb_attr, None)
    if gib is not None and gb is not None:
        raise ValueError(
            f"pass only one of {flag_label}-gib / {flag_label}-gb (the -gb form is "
            "a deprecated alias); got both"
        )
    if gib is not None:
        return float(gib)
    if gb is not None:
        conv = _gb_to_gib(float(gb))
        print(
            f"[ab] WARN: {flag_label}-gb is DEPRECATED (decimal GB); converting "
            f"{float(gb):g} GB -> {conv:.4g} GiB. Use {flag_label}-gib.",
            flush=True,
        )
        return conv
    return default


def _parse_headroom_gib_value(raw, *, source_label: str) -> float:
    """W118 review MEDIUM-3: parse a headroom GiB value (a flag float or an env/preset
    string) and REJECT non-finite (nan/inf), negative, and underscore-obfuscated
    ("1_0") values with a clear ValueError.  ``float("nan") < 0`` is False, so nan/inf
    would otherwise pass validation and crash in-window; ``float("1_0")`` is 10.0 under
    PEP 515, which silently misreads an operator typo."""

    if isinstance(raw, str):
        s = raw.strip()
        if "_" in s:
            raise ValueError(
                f"{source_label} must be a plain number, got {raw!r} "
                "(underscores are not allowed)"
            )
        value = float(s)  # ValueError on 'abc'
    else:
        value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{source_label} must be finite, got {raw!r}")
    if value < 0:
        raise ValueError(f"{source_label} must be non-negative, got {value}")
    return value


def _resolve_mlx_limit_headroom_gib(args) -> float:
    """W118 (H7): the MLX allocator-limit headroom in GiB for THIS arm.

    Precedence: the explicit ``--mlx-limit-headroom-gib`` flag, else the env the arm
    preset stamps (``MTPLX_DSV41_MLX_LIMIT_HEADROOM_GIB``, read at use -- the runtime
    reads the SAME env in apply_mlx_memory_cap), else 0 (today).  Read at use so it
    reflects the arm whose env is currently applied.  A bad value (negative / nan /
    inf / "1_0" / "abc") raises a CLEAN SystemExit (MEDIUM-3), so it fails before the
    in-window crash rather than after."""

    flag = getattr(args, "mlx_limit_headroom_gib", None)
    try:
        if flag is not None:
            return _parse_headroom_gib_value(
                flag, source_label="--mlx-limit-headroom-gib"
            )
        raw = os.environ.get(MLX_LIMIT_HEADROOM_ENV)
        if raw in (None, ""):
            return 0.0
        return _parse_headroom_gib_value(raw, source_label=MLX_LIMIT_HEADROOM_ENV)
    except ValueError as exc:
        raise SystemExit(f"[ab] {exc}") from None


def _resolve_box_target_gb(args):
    """Explicit flag, environment, or 110 decimal GB; explicit legacy plans opt out."""

    v = getattr(args, "box_target_gb", None)
    if v is not None:
        return float(v)
    raw = os.environ.get("MTPLX_DSV41_BOX_TARGET_GB")
    if raw is None:
        if (getattr(args, "memory_limit_gib", None) is not None or
                getattr(args, "box_budget_gib", None) is not None):
            return None
        from mtplx.expert_runtime import DEFAULT_BOX_TARGET_GB
        return float(DEFAULT_BOX_TARGET_GB)
    s = str(raw).strip()
    if s == "" or s.lower() == "default":
        from mtplx.expert_runtime import DEFAULT_BOX_TARGET_GB
        return float(DEFAULT_BOX_TARGET_GB)
    return float(s)


def _resolve_box_baseline_gb(args):
    """Pre-load physical system usage, including file cache, in decimal GB.

    Prefer the guard's current baseline, then an explicit flag, otherwise read
    the OS. Pinned plans separately reject a larger live baseline.
    """

    raw = os.environ.get("MTPLX_DSV41_BOX_BASELINE_GB")
    if raw and str(raw).strip() and str(raw).strip().lower() != "default":
        env_v = float(str(raw).strip())
        flag_v = getattr(args, "box_baseline_gb", None)
        if flag_v is not None and abs(float(flag_v) - env_v) > 1e-6:
            print(
                f"[ab] MEDIUM-4: using the in-window baseline "
                f"MTPLX_DSV41_BOX_BASELINE_GB={env_v:g} GB (gpu_window measured) over "
                f"--box-baseline-gb {float(flag_v):g} GB",
                flush=True,
            )
        return env_v
    v = getattr(args, "box_baseline_gb", None)
    if v is not None:
        return float(v)
    from mtplx.deepseek_v41_memory_profile import box_memory_snapshot
    snapshot = box_memory_snapshot()
    baseline = snapshot.get("used_bytes") if snapshot.get("ok") else None
    if baseline is None or baseline <= 0:
        raise SystemExit("cannot measure pre-load memory baseline")
    return baseline / 1_000_000_000


def _target_sidecar_path(args):
    # MEDIUM-5: per-ARM sidecar name (keyed off the arm's --out stem) so a multi-arm
    # invocation does not overwrite one shared derived-target-plan.json.
    out = getattr(args, "out", None)
    if out is None:
        return None
    out = Path(out)
    stem = out.name
    for suf in (".jsonl", ".json"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return out.parent / f"{stem}.target-plan.json"


def _write_target_plan_sidecar(args, resolved) -> None:
    """Write the resolved target components to a sidecar so a later run can PIN them
    with --memory-plan-from (reproducible residency).  Best-effort."""

    path = _target_sidecar_path(args)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in resolved.items() if k != "pinned_from"}
        path.write_text(json.dumps(payload, indent=2))
    except Exception:  # pragma: no cover - best-effort sidecar
        pass


def _resolve_target_plan(args):
    """Resolve the W121 target-based plan components, or ``None`` when no target is
    armed.  Stamps MTPLX_DSV41_BOX_TARGET_GB / _BASELINE_GB into ``os.environ`` so the
    runtime (expert_runtime.apply_mlx_memory_cap) derives the SAME allocator/wired limit
    = target - baseline - host reserve and set_cache_limit(cache).  --memory-plan-from
    PINS the target + measured components from a sidecar an earlier run wrote, so every
    A/B arm uses the SAME engine budget (reproducible residency)."""

    from mtplx.expert_runtime import (
        BOX_ACTIVE_OVERSHOOT_ENV,
        BOX_ALLOC_CACHE_ENV,
        BOX_BASELINE_ENV,
        BOX_HOST_OVERHEAD_ENV,
        BOX_SESSION_BANK_ENV,
        BOX_TARGET_ENV,
        BOX_TRANSIENT_BAND_ENV,
        resolve_box_target_mlx_limit_bytes,
    )

    pin_path = getattr(args, "memory_plan_from", None)
    comps = None
    if pin_path:
        # MEDIUM-5: a --memory-plan-from that does not exist must RAISE, never silently
        # fall through to a live derivation (that would run a DIFFERENT plan than the
        # operator asked to pin).
        pp = Path(pin_path)
        if not pp.exists():
            raise FileNotFoundError(
                f"--memory-plan-from {pin_path} does not exist (refusing to fall back "
                "to a live derivation)"
            )
        comps = json.loads(pp.read_text())
        comps["pinned_from"] = str(pin_path)

    if comps is not None:
        target_gb = comps["box_target_gb"]
        baseline_gb = comps["box_baseline_gb"]
        live_baseline_gb = _resolve_box_baseline_gb(args)
        if live_baseline_gb > baseline_gb:
            raise ValueError(
                "live baseline exceeds pinned baseline; the pinned allocation "
                "cannot fit its box target on this run")
        cache_gib = comps.get("allocator_cache_limit_gib")
        band_gib = comps.get("transient_band_gib")
        host_gib = comps.get("host_overhead_gib")
        active_gib = comps.get("active_overshoot_gib")
        os.environ[BOX_SESSION_BANK_ENV] = str(
            comps.get("session_bank_capacity_bytes", 0) / GIB)
        if "python_cache_budget" in comps:
            os.environ["MTPLX_ENGRAM_CACHE_LIMIT"] = str(
                comps["python_cache_budget"]["engram_per_layer_bytes"])
    else:
        target_gb = _resolve_box_target_gb(args)
        if target_gb is None:
            return None
        baseline_gb = _resolve_box_baseline_gb(args)
        cache_gib = getattr(args, "allocator_cache_gib", None)
        band_gib = getattr(args, "transient_band_gib", None)
        host_gib = getattr(args, "host_overhead_gib", None)
        active_gib = getattr(args, "active_overshoot_gib", None)

    # Stamp the env the runtime reads (decimal GB target/baseline; GiB host/cache/band/active).
    os.environ[BOX_TARGET_ENV] = f"{float(target_gb):.17g}"
    if baseline_gb is not None:
        os.environ[BOX_BASELINE_ENV] = f"{float(baseline_gb):.17g}"
    if cache_gib is not None:
        os.environ[BOX_ALLOC_CACHE_ENV] = f"{float(cache_gib):.17g}"
    if band_gib is not None:
        os.environ[BOX_TRANSIENT_BAND_ENV] = f"{float(band_gib):.17g}"
    if host_gib is not None:
        os.environ[BOX_HOST_OVERHEAD_ENV] = f"{float(host_gib):.17g}"
    if active_gib is not None:
        os.environ[BOX_ACTIVE_OVERSHOOT_ENV] = f"{float(active_gib):.17g}"

    # resolve_box_target_mlx_limit_bytes raises (actionable) if the baseline is missing.
    r = resolve_box_target_mlx_limit_bytes(os.environ)
    resolved = {
        "memory_plan_source": "box_target",
        "box_target_gb": r["box_target_gb"],
        "box_baseline_gb": r["box_baseline_gb"],
        "host_overhead_gib": r["host_overhead_gib"],
        "allocator_cache_limit_gib": r["allocator_cache_limit_gib"],
        "transient_band_gib": r["transient_band_gib"],
        "active_overshoot_gib": r["active_overshoot_gib"],
        "allocator_limit_bytes": int(r["mlx_limit_bytes"]),
        "engine_budget_bytes": int(r["engine_budget_bytes"]),
        "engine_budget_gib": r["engine_budget_bytes"] / GIB,
        "python_cache_budget": r["python_cache_budget"],
        "session_bank_capacity_bytes": r["session_bank_capacity_bytes"],
        "session_bank_reserve_bytes": r["session_bank_reserve_bytes"],
    }
    if comps is not None:
        resolved["pinned_from"] = comps.get("pinned_from")
    else:
        _write_target_plan_sidecar(args, resolved)
    return resolved


def _resolve_derivation(args, *, bench=None, max_kv=None):
    """The plan->limit derivation for this run (W121 target model).

    Precedence:
      1. The box TARGET (--box-target-gb, else MTPLX_DSV41_BOX_TARGET_GB) -> the ENGINE
         budget = allocator - max(transient, active overshoot + cache) sizes the
         persistent expert slots to FILL the target, and the target env is stamped so
         the runtime sets the allocator/wired limit + set_cache_limit.  --memory-plan-from
         pins the target + measured components so every A/B arm uses the SAME budget.
      2. --memory-limit-gib -> explicit plan.
      3. otherwise the legacy W62 --box-budget derivation.

    Returns the W62 ``BudgetDerivation`` (memory_limit_bytes / reserve / cache plumb
    into the loader unchanged); the target components are stashed on
    ``args._dsv41_target_plan`` for the receipt (separate from measured physical box usage)."""

    from mtplx.deepseek_v41_memory_profile import derive_plan_from_budget

    runtime_reserve_gib = getattr(args, "runtime_reserve_gib", None)
    if runtime_reserve_gib is None:
        runtime_reserve_gib = 7.0
    runtime_reserve_gib = _parse_headroom_gib_value(
        runtime_reserve_gib, source_label="--runtime-reserve-gib"
    )
    if runtime_reserve_gib < 2.0:
        raise ValueError("--runtime-reserve-gib must be at least 2 GiB")

    target_plan = _resolve_target_plan(args)
    args._dsv41_target_plan = target_plan
    if target_plan is not None:
        # MEDIUM-5: the target OVERRIDES --memory-limit-gib; a conflicting explicit
        # engine budget is ambiguous, so REFUSE rather than silently pick one.
        if getattr(args, "memory_limit_gib", None) is not None:
            raise SystemExit(
                "[ab] --memory-limit-gib conflicts with the armed box target "
                "(--box-target-gb / MTPLX_DSV41_BOX_TARGET_GB): the target derives the "
                "engine budget. Pass only one."
            )
        override = target_plan["engine_budget_gib"]
        print(
            "[ab] memory plan from box target: "
            f"target {target_plan['box_target_gb']:g} GB "
            f"- baseline {target_plan['box_baseline_gb']:.4g} GB "
            f"- host overhead {target_plan['host_overhead_gib']:g} GiB "
            f"= allocator limit {target_plan['allocator_limit_bytes'] / GIB:.4g} GiB; "
            f"engine budget = allocator - max(band "
            f"{target_plan['transient_band_gib']:g}, active {target_plan['active_overshoot_gib']:g} "
            f"+ cache {target_plan['allocator_cache_limit_gib']:g}) GiB = {override:.4g} GiB "
            "(sizes the persistent expert slots)"
            + (
                f" [PINNED from {target_plan['pinned_from']}]"
                if target_plan.get("pinned_from")
                else ""
            ),
            flush=True,
        )
    else:
        override = getattr(args, "memory_limit_gib", None)

    target_fields = {} if target_plan is None else {
        "macos_floor_gib": target_plan["box_baseline_gb"] * 1e9 / GIB,
        "host_overhead_gib": target_plan["host_overhead_gib"],
        "cache_limit_gib": target_plan["allocator_cache_limit_gib"],
    }
    derivation = derive_plan_from_budget(
        box_budget_gib=(getattr(args, "box_budget_gib", None) if target_plan is None
                        else target_plan["box_target_gb"] * 1e9 / GIB),
        override_memory_limit_gib=override,
        runtime_reserve_gib=runtime_reserve_gib,
        **target_fields,
    )
    return derivation


# Plan fields the served profile sets that the loader would otherwise default
# (the W81 finding: transient_slots defaulted to spec.top_k=6, not the profile's
# 48).  Seeded into the in-process runtime so a bench A/B is on the production
# plan. ``transient_slots`` and ``cache_policy`` also take explicit flags;
# ``verify_shared_overlap`` is always an explicit construction-time arm.
_PROFILE_PLAN_FIELDS = (
    "transient_slots",
    "split_route_release",
    "prefetch_slots",
    "bypass_page_cache",
    "cache_policy",
)


def _resolve_plan_overrides(args) -> dict:
    """ExpertStreamingConfig plan overrides from the profile / explicit flags.

    Precedence per field: explicit flag > profile config value > loader default
    (leave unset).  Returns only the fields we actually pin, so an empty dict
    means "loader defaults" (matches the pre-W81 behaviour when profile='none').
    """

    overrides: dict = {}
    explicit_transient = getattr(args, "transient_slots", None)
    if explicit_transient is not None:
        overrides["transient_slots"] = int(explicit_transient)
    explicit_policy = getattr(args, "cache_policy", None)
    if explicit_policy is not None:
        overrides["cache_policy"] = str(explicit_policy)
    explicit_capacities = getattr(args, "persistent_slots_by_layer", None)
    if explicit_capacities is not None:
        overrides["persistent_slots_by_layer"] = tuple(explicit_capacities)
    explicit_miss_part = getattr(args, "decode_miss_records_per_part", None)
    if explicit_miss_part is not None:
        overrides["decode_miss_records_per_part"] = int(explicit_miss_part)
    overrides["verify_shared_overlap"] = bool(
        getattr(args, "verify_shared_overlap", False)
    )

    profile_name = str(getattr(args, "expert_profile", "none") or "none")
    if profile_name != "none":
        try:
            from mtplx.expert_profiles import load_expert_profiles

            profiles = load_expert_profiles()
            profile = profiles.get(profile_name)
        except Exception:
            profile = None
        if profile is not None:
            cfg = dict(getattr(profile, "config", {}) or {})
            for field in _PROFILE_PLAN_FIELDS:
                if field in overrides:
                    continue  # explicit flag already won
                if field in cfg:
                    overrides[field] = cfg[field]
    return overrides


def _runtime_gate_prefetch_k(runtime) -> int | None:
    """The gate-oracle predict width ``k`` the RUNTIME actually resolved, read from
    the runtime's own runner receipt block (``resource_telemetry_snapshot()['runner']
    ['prefetch_k']``, built by deepseek_v41._runner_snapshot).  Returns ``None`` when
    the runtime cannot produce the block (e.g. an explicit GATE_PREFETCH-only arm with
    no v2 runner), so the caller falls back to ``prefetch_slots//2``.

    W110: this is the fix for the window-41 receipt bug -- reading the width from the
    runtime, not from ``os.environ`` (which misses the v2 auto-arm), and not from
    ``prefetch_slots//2`` (the v2 ring is sized to buffer the verify union, not 2*k).
    """

    try:
        snap = runtime.resource_telemetry_snapshot()
    except Exception:
        return None
    if not isinstance(snap, dict):
        return None
    runner = snap.get("runner")
    if isinstance(runner, dict) and runner.get("prefetch_k") is not None:
        try:
            return int(runner["prefetch_k"])
        except (TypeError, ValueError):
            return None
    return None


def _resolved_plan(runtime, args) -> dict | None:
    """The runtime's ACTUAL slot plan, for the receipt/census header.

    Records the plan the bench really ran (not the requested override), so an
    A/B is attributable to a slot plan and window 32 can compare bench vs served.
    """

    plan = getattr(runtime, "plan", None)
    if plan is None:
        return None
    spec = getattr(runtime, "spec", None)
    config = getattr(runtime, "config", None)
    record_bytes = int(getattr(spec, "expert_record_bytes", 0) or 0)
    transient_slots = int(getattr(plan, "transient_slots", 0) or 0)
    persistent_slots = int(getattr(plan, "persistent_slots", 0) or 0)
    routed_layers = int(getattr(spec, "routed_layer_count", 0) or 0)
    prefetch_slots = int(getattr(config, "prefetch_slots", 0) or 0)
    slots_per_layer = int(getattr(plan, "slots_per_layer", 0) or 0)
    layer_capacities = tuple(
        getattr(plan, "persistent_slots_by_layer", ()) or ()
    )
    # W93 (review CRITICAL): an EXPLICIT gate-oracle lever must have ACTUALLY armed
    # the GLOBAL ring on this cell -- otherwise the A/B is control-vs-control. Fail
    # loudly rather than silently benchmarking an unarmed ring. (Explicit-lever guard
    # only; the v2 auto-arm is handled below.)
    gate_env = os.environ.get(GATE_PREFETCH_ENV)
    gate_env_armed = bool(gate_env) and gate_env not in ("0", "")
    if gate_env_armed and prefetch_slots <= 0:
        raise AssertionError(
            f"{GATE_PREFETCH_ENV}={gate_env!r} is armed but the runtime built NO "
            "prefetch ring (prefetch_slots=0). The gate-oracle lever would measure "
            "control-vs-control -- check build_streaming_config / the served "
            "profile arm the env-authoritative ring."
        )
    # W110 (receipt fix): report the ACTUAL armed state, read from the runtime object
    # the loader built -- NOT os.environ. The v2 runner AUTO-ARMS the gate-oracle ring
    # (build_streaming_config sets prefetch_slots for MTPLX_DSV41_RUNNER=v2) with
    # MTPLX_DSV41_GATE_PREFETCH UNSET, so the old env-only ``gate_armed`` reported
    # armed=False / k=0 while the runner actually prefetched at k=6 (window 41:
    # prefetch_committed=10513). A built ring (config.prefetch_slots>0) IS the armed
    # signal; the predict WIDTH is the width the runtime resolved (its runner receipt
    # block's ``prefetch_k``), NOT prefetch_slots//2 -- the v2 ring is sized 2*24=48
    # to double-buffer the ~24-expert verify union one layer ahead, so //2 would
    # misreport the width as 24 instead of 6.
    gate_armed = prefetch_slots > 0
    gate_k = _runtime_gate_prefetch_k(runtime) if gate_armed else 0
    if gate_k is None:
        # No runner block (e.g. an explicit GATE_PREFETCH-only arm, ring = 2*k):
        # //2 recovers the explicit predict width.
        gate_k = prefetch_slots // 2
    # W93 (review MEDIUM-d): the 0.36 GiB ring comes out of the PERSISTENT budget,
    # not free reserve. Show slots_per_layer WITHOUT vs WITH the ring so the LRU
    # effect (kept the same only via floor-division slack) is auditable per run.
    slots_per_layer_no_ring = None if layer_capacities else slots_per_layer
    if prefetch_slots > 0 and config is not None and spec is not None:
        try:
            import dataclasses

            no_ring = dataclasses.replace(config, prefetch_slots=0)
            slots_per_layer_no_ring = int(
                getattr(no_ring.memory_plan(spec), "slots_per_layer", slots_per_layer)
            )
        except Exception:
            slots_per_layer_no_ring = slots_per_layer
    return {
        "memory_limit_bytes": getattr(plan, "total_limit_bytes", None),
        "transient_slots": transient_slots,
        "persistent_slots": persistent_slots,
        # review MEDIUM-d: LRU depth WITH the ring vs the hypothetical no-ring plan.
        "slots_per_layer": None if layer_capacities else slots_per_layer,
        "uniform_equivalent_slots_per_layer": slots_per_layer,
        "persistent_slots_by_layer": {
            str(layer): capacity for layer, capacity in layer_capacities
        },
        "slots_per_layer_no_ring": slots_per_layer_no_ring,
        "expert_record_bytes": record_bytes,
        "transient_bytes_per_layer": transient_slots * record_bytes,
        "transient_bytes_total": transient_slots * record_bytes,
        "transient_bytes_scope": "shared_across_layers",
        "split_route_release": getattr(config, "split_route_release", None),
        "runtime_reserve_bytes": getattr(config, "runtime_reserve_bytes", None),
        "cache_policy": getattr(config, "cache_policy", None),
        "decode_miss_records_per_part": getattr(
            config,
            "decode_miss_records_per_part",
            None,
        ),
        "verify_shared_overlap": bool(
            getattr(config, "verify_shared_overlap", False)
        ),
        "verify_shared_overlap_bound_layers": int(
            getattr(runtime, "_verify_shared_overlap_bound_layers", 0)
        ),
        "single_slot_pool": bool(
            getattr(runtime, "_single_slot_pool", False)
        ),
        # GLOBAL ring: prefetch_slots records TOTAL (shared); k = predict width.
        "prefetch_slots": prefetch_slots,
        # W110: the ACTUAL armed state / predict width the runtime ran (v2 auto-arm
        # included), not the explicit env. ``gate_prefetch_env`` keeps the raw env
        # for provenance (None on a v2-auto-armed run).
        "gate_prefetch_armed": gate_armed,
        "gate_prefetch_k": gate_k,
        "gate_prefetch_env": gate_env,
        "gate_prefetch_ring_bytes": prefetch_slots * record_bytes,
        # W110 (guard/stamp): the ACTUAL decode-path per-record sha256 state the
        # runtime ran, so a hash-vs-parent A/B can never be control-vs-control
        # silently (parent stamps False, cell16k_ring_v2_hash stamps True).
        "verify_record_hashes": bool(getattr(config, "verify_record_hashes", False)),
        "io_cache_mode": getattr(getattr(runtime, "reader", None), "cache_mode", None),
        "io_read_fanout": getattr(getattr(runtime, "reader", None), "io_read_fanout", None),
        "source": (
            "explicit" if getattr(args, "transient_slots", None) is not None
            else f"profile:{getattr(args, 'expert_profile', 'none')}"
            if getattr(args, "_dsv41_plan_overrides", None)
            else "loader-default(top_k)"
        ),
    }


def _receipt_plan_limit_bytes(receipt) -> int | None:
    """Canonical engine budget, preserving old target/legacy receipt support."""
    for block, key in (("resolved_plan", "memory_limit_bytes"),
                       ("memory_cap", "engine_budget_bytes")):
        value = (receipt.get(block) or {}).get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    memory = receipt.get("memory") or {}
    for key in ("plan_limit_gib_effective", "plan_limit_gib_derived"):
        value = memory.get(key)
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value > 0):
            return int(value * GIB)
    return None


def _load_model(args, bench, mx):
    from mtplx.deepseek_v41_memory_profile import apply_allocator_cache_limit
    from mtplx.models.deepseek_v41_dspark_decode import dspark_bench_loader_overrides
    from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming

    admission_receipt = None
    if args.admission_receipt is not None:
        admission_receipt = json.loads(Path(args.admission_receipt).read_text())
    max_kv = bench.resolve_max_kv([args.context_tokens], args.decode_tokens, args.max_kv)
    cache_limit = (
        None
        if args.expert_cache_limit_gib is None
        else int(args.expert_cache_limit_gib * GIB)
    )
    derivation = _resolve_derivation(args, bench=bench, max_kv=max_kv)
    print("[ab] memory derivation: " + derivation.formula(), flush=True)
    # W97: the fixed resident reserve the plan prices (SWA window + the f32 wo_a
    # cache when MTPLX_DSV41_ATTN_WO_A_CACHE is armed).  The arm env is already set
    # (_apply_arm_env ran), so this reflects THIS arm; the expert-cache allowance
    # shrinks by the reserve rather than the process running over plan.
    try:
        from mtplx.models.deepseek_v41_loader import (
            SWA_WINDOW_BYTES as _swa_bytes,
            deepseek_v41_additional_resident_bytes as _addl_resident,
        )

        _addl = _addl_resident()
        _wo_a_reserve = _addl - _swa_bytes
        print(
            f"[ab] additional backbone reserve: SWA {_swa_bytes / GIB:.3f} GiB"
            + (
                f" + wo_a cache {_wo_a_reserve / GIB:.3f} GiB"
                if _wo_a_reserve
                else ""
            )
            + f" = {_addl / GIB:.3f} GiB (MTP weights/caches priced separately by loader)",
            flush=True,
        )
    except Exception:  # pragma: no cover - display only, never fail the run
        pass
    # --decode-mode dspark (or --with-mtp on AR) loads with the DSpark head
    # (with_mtp=True) and reprices the MTP residents against the expert cache so
    # the plan still fits.
    want_head = (getattr(args, "decode_mode", "ar") == "dspark") or bool(
        getattr(args, "with_mtp", None)
    )
    with_mtp, memory_limit_bytes, cache_limit = dspark_bench_loader_overrides(
        want_dspark=want_head,
        memory_limit_bytes=derivation.memory_limit_bytes,
        expert_cache_limit_bytes=cache_limit,
        reprice=bool(getattr(args, "reprice", True)),
    )
    # W81: run the in-process bench on the SAME slot plan the served daemon uses.
    # The served path builds its config from the profile deepseek-v41-mxfp4-75
    # (transient_slots=48, split_route_release=deferred), but this CLI loader left
    # transient_slots unset -> plan default spec.top_k (=6), which starved the
    # W66/W81 verify fast path in window 31.  Seed the profile's plan fields when
    # the matching flag is unset (--transient-slots / profile 'none' disable it).
    plan_overrides = _resolve_plan_overrides(args)
    args._dsv41_plan_overrides = plan_overrides
    if plan_overrides:
        print(
            "[ab] plan overrides from profile "
            f"{getattr(args, 'expert_profile', 'none')!r}: "
            + json.dumps(plan_overrides),
            flush=True,
        )
    resident = load_deepseek_v41_streaming(
        args.model,
        memory_limit_bytes=memory_limit_bytes,
        max_live_kv_tokens=int(max_kv),
        runtime_reserve_bytes=derivation.runtime_reserve_bytes,
        admit=args.admit,
        admission_receipt=admission_receipt,
        expert_cache_limit_bytes=cache_limit,
        apply_memory_cap=args.apply_memory_cap,
        slot_layout=args.slot_layout,
        cache_scope="layer",
        island_layers=(),
        verify_record_hashes=args.verify_record_hashes,
        with_mtp=with_mtp,
        **plan_overrides,
    )
    # W62 (2): bound the MLX allocator's freed-buffer cache from the plan so
    # decode/prefill transients that are freed do not accumulate past the reserve
    # (the served path's _configure_mlx_cache_limit machinery, applied to this
    # CLI lane which set only the active-allocation memory limit).
    # ResidentModel is a frozen dataclass, so carry the derivation/cache report
    # on the (mutable) args namespace rather than on the model.
    args._dsv41_derivation = derivation
    args._dsv41_cache_report = None
    # W121: on the target path the runtime's apply_mlx_memory_cap ALREADY set
    # set_cache_limit to the reserved allocator-cache band (target - baseline - cache);
    # don't re-set it here (a second set with the profile default would clobber an
    # --allocator-cache-gib override).  Only the legacy plan path needs this post-load
    # cache bound (the CLI lane historically set only the active-allocation limit).
    if args.apply_memory_cap and getattr(args, "_dsv41_target_plan", None) is None:
        cache_report = apply_allocator_cache_limit(
            derivation.cache_limit_bytes, mx_module=mx
        )
        print("[ab] allocator cache limit: " + json.dumps(cache_report), flush=True)
        args._dsv41_cache_report = cache_report
    if with_mtp and getattr(resident.model, "mtp", None) is None:
        raise RuntimeError(
            "--decode-mode dspark needs the DSpark MTP head, but the loaded model "
            "has none (with_mtp did not build it -- the artifact ships no mtp.* "
            "residents, or the config declares no MTP stages). Load a DSpark "
            "artifact or drop --decode-mode dspark."
        )
    # W121: no post-load non-Metal re-measure / budget abort -- the target model
    # (target - baseline - cache) is measured
    # against the target by the receipt's memory block + the gpu_window.sh guard, not
    # forecast from a vm_stat baseline before load.
    return resident


def _memory_profile_collector(args, mx, runtime, resident):
    """(callback, snapshots) for the W62 profile, or (None, None) when off.

    The callback captures a snapshot at ``load_end`` immediately, then is handed
    to ``_generate`` for the ``after_prefill`` and per-N-decode captures.  Only
    the load-end snapshot carries the derivation (all share one plan)."""

    if not getattr(args, "memory_profile", False):
        return None, None
    from mtplx.deepseek_v41_memory_profile import memory_profile_snapshot

    plan = getattr(runtime, "plan", None)
    derivation = getattr(args, "_dsv41_derivation", None)
    snaps: list = []

    def _cb(phase, token=None):
        snaps.append(
            memory_profile_snapshot(
                phase=phase,
                token=token,
                plan=plan,
                runtime=runtime,
                mx_module=mx,
                derivation=derivation if phase == "load_end" else None,
            )
        )

    _cb("load_end")
    return _cb, snaps


def _stream_counters_snapshot(model):
    """Best-effort snapshot of the expert-streaming counters, or None.

    W81: lets the in-process bench report the SAME serve_stream_counters block the
    served daemon does (expert hits/misses/bytes/loads per token), so David's hit
    rate + bandwidth-per-token show up on every A/B receipt.  Guarded: a stub
    runtime (or a build without the snapshot) just omits the block.
    """
    try:
        from mtplx.serve_stream_counters import snapshot_stream_counters

        rt = getattr(model, "_mtplx_expert_runtime", None)
        if rt is None:
            return None
        return snapshot_stream_counters(rt, model=model)
    except Exception:
        return None


def _runner_receipt_blocks(model) -> dict:
    """W95f (review HIGH-3): lift the runtime's ``runner`` (v2) and ``gate_prefetch``
    receipt blocks onto every A/B receipt (AR and DSpark), so the SSD-hiding counters
    the paired window reads -- prefetch hit/wasted, demand vs speculative bytes,
    budget_skips, margin, ring size, per-decode-token normalisations -- travel with
    the cell receipt.  They lived only in resource_telemetry_snapshot, whose callers
    were the other benchmark scripts + tests, NOT this harness.  Best-effort: a stub
    runtime or the flags-off shipped path just omits the blocks (empty dict)."""
    try:
        rt = getattr(model, "_mtplx_expert_runtime", None)
        if rt is None:
            return {}
        es = getattr(rt, "expert_streaming", None) or rt
        snap = getattr(es, "resource_telemetry_snapshot", None)
        if not callable(snap):
            return {}
        full = snap()
        if not isinstance(full, dict):
            return {}
        return {key: full[key] for key in ("runner", "gate_prefetch") if key in full}
    except Exception:
        return {}


def _cold_reset_expert_streaming(model) -> bool:
    """W87 HIGH-3: cold-reset the expert-streaming residency + counters between the
    AR reference pass and the DSpark pass of a --decode-mode dspark cell, so the
    DSpark prefill measures a COLD bank (the AR pass otherwise leaves it warm, which
    confounds the single-slot-pool warming A/B).  Best-effort; append-only."""
    try:
        rt = getattr(model, "_mtplx_expert_runtime", None)
        if rt is None:
            return False
        # W87 HIGH-1: the loader attaches the BARE ExpertStreamingRuntime (which HAS
        # reset()), not an MTPLXRuntime (whose .expert_streaming has it); resolve
        # either shape, else the DSpark cold reset was a silent no-op.
        es = getattr(rt, "expert_streaming", None) or rt
        reset = getattr(es, "reset", None)
        if callable(reset):
            reset()
            return True
    except Exception:
        return False
    return False


def _stream_counters_block(run, decode_tokens, resolved_plan):
    """DECODE-scoped serve_stream_counters delta for the receipt (or None).

    ``run`` carries ``stream_after_prefill`` / ``stream_end`` snapshots bracketing
    the decode loop (prefill excluded, so the hit rate is the DECODE hit rate, not
    the cold prefill first-touch rate).  Attaches the resolved slot plan so the
    hit rate is attributable to a capacity.
    """
    try:
        from mtplx.serve_stream_counters import stream_counters_delta

        before = run.get("stream_after_prefill")
        after = run.get("stream_end")
        if not before or not after:
            return None
        # Both AR and DSpark include the first, prefill-produced token. Early
        # EOS can shorten either pass; normalize decode I/O by actual output.
        generated = run.get("generated")
        if generated is not None:
            decode_tokens = max(0, len(generated) - 1)
        block = stream_counters_delta(
            before, after, tokens=int(decode_tokens), phase="decode"
        )
        if not block:
            return None
        if resolved_plan is not None:
            block["slot_plan"] = resolved_plan
        return block
    except Exception:
        return None


_MACMON_MOD = None


def _macmon():
    """Load the shared W90 macmon sampler (``util_macmon.py``, same dir) once.

    Loaded by file path so it works whether this script is run directly or imported
    by a test (``scripts/`` is not a package)."""
    global _MACMON_MOD
    if _MACMON_MOD is None:
        import importlib.util

        path = Path(__file__).resolve().parent / "util_macmon.py"
        spec = importlib.util.spec_from_file_location("dsv41_util_macmon", path)
        _MACMON_MOD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MACMON_MOD)
    return _MACMON_MOD


def _generate(*, model, ops, mem_probe, prompt_ids, steps, mem_profile=None,
              mem_profile_every=64, device_sample=False, cooldown_s=0.0,
              util_sampler=None, stage_timing=False, stop_on_eos=False,
              eos_id=None):
    """Greedy prefill + ``steps`` decode; captures the decoded token ids.

    W113: ``stop_on_eos`` (default off) stops the decode at ``eos_id`` for a
    served-parity run and returns ``decode_steps_run`` = the number of DECODE
    tokens actually generated (excludes the prefill argmax token that begins
    ``generated``), so the caller can report decode_tok_s over the tokens actually
    produced.  Default off runs the full fixed ``steps`` decode (numbers
    unchanged) and ``decode_steps_run == steps``.  When the prefill's own first
    token is already EOS and ``stop_on_eos``, the decode loop is skipped entirely
    (a served path would emit nothing), so ``decode_steps_run == 0``.

    W90: ``cooldown_s`` idles AFTER prefill and BEFORE the timed decode (TTFT, from
    the prefill, is unaffected); ``util_sampler`` (a ``util_macmon.UtilizationSampler``
    or ``None``) samples macmon in a background thread over the DECODE loop only.
    Both land in the returned dict (``cooldown`` / ``utilization``) -- the W90
    discriminator for the GPU DVFS-downclock floor.

    ``mem_profile`` (optional) is a ``callable(phase, token=None)`` that captures
    a W62 memory snapshot; it is called ``after_prefill`` and every
    ``mem_profile_every`` decode tokens.  ``None`` leaves the loop unchanged.

    ``device_sample`` (W63 / K32) swaps the per-token host round-trip decode loop
    for the device-sample one-step-lag pipeline
    (:func:`run_device_sample_decode`): the greedy token stays a lazy device
    array fed straight into the next forward, and the host reads ids one step
    behind an already-submitted forward. Greedy output is byte-identical to the
    classic argmax loop; the AR-reference byte-identity gate (this arm's
    ``token_ids_sha256`` vs the dspark ids) therefore still holds."""
    mem_probe.reset_peak()
    # W106: sample process phys_footprint + system used memory off the hot path (daemon thread,
    # 1 Hz, no MLX calls) over the whole generation, so the receipt's memory block
    # carries the real envelope, not just the MLX allocator peak (peak_gb).
    _mem_sampler = mem_probe.new_sampler()
    _mem_sampler.start()
    try:
        t0 = time.perf_counter()
        cache = model.make_cache()
        # AR consumes only the final prompt row. Do not allocate a full
        # prompt-by-vocabulary output before the timed decode.
        logits = model(ops.input([list(prompt_ids)]), cache=cache, logits_keep=1)
        ops.sync(logits)
        ttft_s = time.perf_counter() - t0
        token = ops.argmax_last(logits)
        generated = [token]
        if mem_profile is not None:
            mem_profile("after_prefill")
        # W90: idle after prefill, before the timed decode (TTFT already captured).
        cooldown_block = None
        if cooldown_s and float(cooldown_s) > 0:
            cooldown_block = _macmon().cooldown(float(cooldown_s), label="ab")
        extra_forward_steps = 0
        # W113: DECODE tokens actually generated (excludes the prefill argmax token
        # that begins ``generated``).  == steps on the default full run; fewer under
        # --stop-on-eos.
        decode_steps_run = 0
        _stop_eos = int(eos_id) if (stop_on_eos and eos_id is not None) else None

        # W92 switch-dispatch census: arm the route-stage probe scoped to the DECODE
        # loop (prefill excluded) so the receipt reports per-layer host syncs
        # (hot.eval_indices), all-hit fences deferred vs synced (hot.allhit_defer vs
        # hot.allhit_fence_eval), and gather_qmm dispatches per switch call
        # (hot.allhit_gather_qmm).  ENABLED is read at use, so setting it here arms the
        # module even if the launch env did not.
        # W94: clear the probe counters HERE -- immediately BEFORE the after_prefill
        # ("before") snapshot below, and NOWHERE between it and the end snapshot -- so the
        # DECODE-scoped route_probe_counts / route_probe_sums_ns delta (end - after_prefill,
        # computed in _stream_counters_block via serve_stream_counters.stream_counters_delta)
        # is EXACT.  The old order cleared AFTER the before-snapshot, so "before" still held
        # the prefill accumulation while "after" held decode-only; stages the prefill
        # dominates (e.g. hot.begin_split_route) then deltaed NEGATIVE, making the
        # "eval(indices) time" a lower bound only (window-37 ar-ring-ref sums_ns).
        _route_probe = None
        _route_prev_enabled = None
        if stage_timing:
            try:
                from mtplx import expert_route_probe as _route_probe

                _route_prev_enabled = _route_probe.ENABLED
                _route_probe.ENABLED = True
                _route_probe._SUMS.clear()
                _route_probe._COUNTS.clear()
            except Exception:
                _route_probe = None

        # W125: arm + size + clear the host decode timeline for THIS decode pass, so
        # a multi-arm process never accumulates across arms. Model.__call__ drives the
        # per-token/per-layer marks; the harness only configures and snapshots. A
        # no-op unless MTPLX_DSV41_DECODE_TIMELINE=1.
        _decode_tl = None
        try:
            from mtplx import dsv41_decode_timeline as _decode_tl

            if _decode_tl.env_armed():
                # MEDIUM (red-team): size to the actual decode length so a
                # 1024-token run is not truncated at the 512 default. --device-sample
                # runs steps+1 forwards (one-step-lag), so add one there.
                _tl_maxtok = int(steps) + (1 if device_sample else 0)
                _decode_tl.configure(
                    len(model.layers), max_tokens=max(1, _tl_maxtok)
                )
                _decode_tl.reset()
            else:
                _decode_tl = None
        except Exception:
            _decode_tl = None

        # W81: bracket the DECODE loop for the serve_stream_counters block (prefill
        # excluded so the hit rate is the decode hit rate).  Taken AFTER the route-probe
        # clear above, so its route_probe_* baseline is zero and the decode delta is exact.
        _sc_after_prefill = _stream_counters_snapshot(model)

        # W118 review MEDIUM-2: allocator active/cache readback around the timed decode
        # (main thread, OUTSIDE the timed region: active_start just before the timer
        # starts, active/cache_end just after it stops).
        _active_start_bytes = _active_end_bytes = _cache_end_bytes = None
        _util_cm = util_sampler if util_sampler is not None else contextlib.nullcontext()
        # W90: the sampler's macmon Popen/terminate happen on the context enter/exit; take
        # decode_start AFTER enter and decode_wall_s BEFORE exit, so tok/s excludes them.
        with _util_cm:  # W90: macmon utilization sampled over the DECODE loop only
            _active_start_bytes = _mlx_call_int(getattr(mem_probe, "_mx", None), "get_active_memory")
            decode_start = time.perf_counter()
            try:  # W92: restore the probe ENABLED flag even if the decode loop raises
                if device_sample:
                    from mtplx.models.deepseek_v41_dspark_decode import run_device_sample_decode

                    def _forward_row(ids):
                        # ids is a device-side [1, 1] token-id array; the model's embedding
                        # lookup consumes it directly (mx.take) -- no host round trip.
                        return model(ids, cache=cache)[0, -1]

                    more, _finish, extra_forward_steps = run_device_sample_decode(
                        forward_row=_forward_row,
                        first_token=int(token),
                        n_more=int(steps),
                        sampler=None,  # greedy (byte-identical to the classic argmax loop)
                        # W113: served-parity early stop (default off -> empty set ->
                        # full fixed-step decode, byte-identical to the classic loop).
                        stop_ids=({_stop_eos} if _stop_eos is not None else set()),
                    )
                    generated.extend(int(t) for t in more)
                    decode_steps_run = len(more)
                else:
                    every = max(1, int(mem_profile_every))
                    # W113: when the prefill's own first token is already EOS, a served
                    # path emits nothing -- skip the decode loop entirely.
                    if not (_stop_eos is not None and int(token) == _stop_eos):
                        for step in range(int(steps)):
                            logits = model(ops.input([[token]]), cache=cache)
                            ops.sync(logits)
                            token = ops.argmax_last(logits)
                            generated.append(token)
                            decode_steps_run += 1
                            if mem_profile is not None and (step + 1) % every == 0:
                                mem_profile("decode", token=step + 1)
                            # W113: served-parity early stop at EOS (default off).
                            if _stop_eos is not None and int(token) == _stop_eos:
                                break
            finally:
                # W92: restore the probe ENABLED flag even if the decode loop raised, so
                # a failed arm never leaves the module armed for the rest of the process
                # (the snapshot below reads _COUNTS regardless of ENABLED).
                if _route_probe is not None and _route_prev_enabled is not None:
                    _route_probe.ENABLED = bool(_route_prev_enabled)
            decode_wall_s = time.perf_counter() - decode_start
            # W118 review MEDIUM-2: active/cache at decode end (timer stopped).
            _active_end_bytes = _mlx_call_int(getattr(mem_probe, "_mx", None), "get_active_memory")
            _cache_end_bytes = _mlx_call_int(getattr(mem_probe, "_mx", None), "get_cache_memory")
        _sc_end = _stream_counters_snapshot(model)
        # W125: snapshot the host decode timeline (per-token/per-layer phase
        # percentiles + host-gap accounting + stamped probe overhead). None unless
        # the probe was armed for this pass.
        _decode_timeline_block = (
            _decode_tl.snapshot()
            if (_decode_tl is not None and _decode_tl.enabled())
            else None
        )
        switch_dispatch = None
        if _route_probe is not None:
            _snap = _route_probe.snapshot()
            _stg = _snap.get("stages", {}) if isinstance(_snap, dict) else {}

            def _c(name):
                return int(_stg.get(name, {}).get("count", 0))

            _all_hit = _c("hot.all_hit")
            _synced = _c("hot.allhit_fence_eval")
            _deferred = _c("hot.allhit_defer")
            _gather_qmm_total = _c("hot.switch_gather_qmm")   # all paths
            _allhit_gather_qmm = _c("hot.allhit_gather_qmm")  # all-hit branch only
            _decode_steps = max(1, int(steps))
            switch_dispatch = {
                # per-layer host round-trips over this DECODE pass (cumulative).
                "eval_indices": _c("hot.eval_indices"),      # the one routing barrier/layer
                "route_host_tolist": _c("hot.route_host"),   # .tolist()+int() host read/layer
                # all-hit switch: fences DEFERRED (to the next routing barrier) vs SYNCED
                # (the shipped blocking mx.eval(wave_output)). The whole win is the ratio.
                "all_hit": _all_hit,
                "allhit_fence_synced": _synced,
                "allhit_fence_deferred": _deferred,
                "allhit_defer_submit": _c("hot.allhit_defer_submit"),
                "allhit_deferred_pct": (
                    round(100.0 * _deferred / _all_hit, 2) if _all_hit else None
                ),
                # miss/split switch: begin_split_route admissions + split-route layer-calls.
                "split_route": _c("hot.split_route"),
                "begin_split_route": _c("hot.begin_split_route"),
                # dispatch census. switch_gather_qmm_total counts EVERY gather_qmm on
                # the pass (all-hit + split parts + prefill waves); allhit_gather_qmm is
                # the all-hit branch only, so gather_qmm_per_all_hit_call is exactly 3
                # (gate/up/down grouped over the routed slots -- never per-expert).
                "switch_gather_qmm_total": _gather_qmm_total,
                "allhit_gather_qmm": _allhit_gather_qmm,
                "gather_qmm_per_all_hit_call": (
                    round(_allhit_gather_qmm / _all_hit, 3) if _all_hit else None
                ),
                "note": "cumulative over this AR DECODE pass. Per layer the host round-trip "
                        "is one mx.eval(indices) barrier; allhit_fence_synced counts the "
                        "SECOND blocking eval the shipped path pays per all-hit layer "
                        "(switch_lean defers it -> allhit_fence_deferred). "
                        "allhit_gather_qmm/all_hit is 3 (gate/up/down); "
                        "switch_gather_qmm_total also includes split parts + prefill waves.",
            }
            print(
                "[ab] switch dispatch: "
                f"all_hit={_all_hit} fence_synced={_synced} fence_deferred={_deferred} "
                f"({switch_dispatch['allhit_deferred_pct']}% deferred) "
                f"allhit_gather_qmm={_allhit_gather_qmm} "
                f"(~{switch_dispatch['gather_qmm_per_all_hit_call']}/all-hit call) "
                f"gather_qmm_total={_gather_qmm_total} "
                f"split_route={switch_dispatch['split_route']} "
                f"eval_indices={switch_dispatch['eval_indices']}",
                flush=True,
            )
    finally:
        _mem_sampler.stop()
    return {
        "generated": [int(t) for t in generated],
        "ttft_s": ttft_s,
        "decode_wall_s": decode_wall_s,
        "peak_gb": mem_probe.peak_bytes() / 1_000_000_000,
        # W106: full memory envelope (mlx_peak_gb == peak_gb, process phys_footprint,
        # and sampled whole-machine physical use including file cache).  W118
        # review MEDIUM-2: merge the allocator readback proof keys (limit readback,
        # gc_limit, active/cache at decode start/end, peak-over-limit).
        "memory": _ab_memory_block(
            mem_probe.memory_block(_mem_sampler),
            _mlx_headroom_readback_keys(
                getattr(mem_probe, "_mx", None),
                mlx_peak_bytes=mem_probe.peak_bytes(),
                active_start_bytes=_active_start_bytes,
                active_end_bytes=_active_end_bytes,
                cache_end_bytes=_cache_end_bytes,
            ),
        ),
        "extra_forward_steps": int(extra_forward_steps),
        # W113: DECODE tokens actually generated (== steps unless --stop-on-eos).
        "decode_steps_run": int(decode_steps_run),
        "stream_after_prefill": _sc_after_prefill,
        "stream_end": _sc_end,
        # W95f: the v2 runner + gate_prefetch receipt blocks (present only when armed).
        **_runner_receipt_blocks(model),
        "cooldown": cooldown_block,
        "utilization": (
            util_sampler.summarize() if util_sampler is not None else None
        ),
        "switch_dispatch": switch_dispatch,
        # W125: host-side decode timeline telemetry (additive; None unless
        # MTPLX_DSV41_DECODE_TIMELINE=1). See docs W125 for phase definitions.
        "decode_timeline": _decode_timeline_block,
    }


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4g}"


def _dspark_divergence_rule(d: dict) -> str:
    """Which W120 rule fired for a tie_flip (or why one did not), from the receipt
    scalars.  Uses the CONTESTED margins + the magnitude-aware ``tie_band_used`` --
    NOT the legacy ``ar_top2_margin < tie_margin`` (false for a W120 tie_flip)."""
    band = d.get("tie_band_used", d.get("tie_margin"))
    arc = d.get("ar_contested_margin")
    dsc = d.get("dspark_contested_margin")
    within = d.get("deltas_within_tie_band")
    # rows_consistent gates absolution: a row whose argmax is not its credited token
    # did not PRODUCE that token, so no rule fires regardless of the margins.
    if d.get("rows_consistent") is False:
        return "none (row argmax != credited token -> row did not produce it)"
    # A rule only "fires" if the delta gate passed (mirrors the class logic): a
    # small contested margin does NOT absolve when a contested delta exceeds the band
    # (or is non-finite, which forces deltas_within_tie_band to False).
    if within is False:
        return "none (contested delta > band / non-finite -> not rounding)"
    fired = []
    if within and band is not None and arc is not None and dsc is not None and min(arc, dsc) < band:
        fired.append("near_tie_by_band(a)")
    if within and d.get("rounding_class_by_delta"):
        fired.append("rounding_class_by_delta(c)")
    return "+".join(fired) if fired else "none"


def _dspark_tie_class_gate_passes(dspark: dict | None) -> bool:
    """Accept a complete identity receipt or a proven, index-matched tie flip."""
    if not isinstance(dspark, dict):
        return False
    identical = dspark.get("byte_identical_vs_ar")
    divergence = dspark.get("divergence")
    if identical is True:
        return divergence is None
    if identical is not False or not isinstance(divergence, dict):
        return False
    return (
        divergence.get("class") == "tie_flip"
        and divergence.get("capture_index_matches_first") is True
    )


def _print_dspark_divergence(arm: str, d: dict) -> None:
    """W120 census line for a classified DSpark divergence.  ``tie_flip`` is a
    one-line note (acceptable, rounding-class); ``divergent`` is LOUD (the flip is
    larger than the bf16 rounding envelope -- a real lane bug or a non-rounding
    lever), so the operator sees it in the arm log even though the arm no longer
    aborts.  Prints the magnitude-aware ``tie_band_used``, BOTH contested margins,
    and which rule fired (the old ``ar_top2_margin < tie_margin`` line was false for
    W120 tie_flips)."""
    i = d["divergence_index"]
    margins = (
        f"ar_contested={_fmt(d.get('ar_contested_margin'))} "
        f"dsp_contested={_fmt(d.get('dspark_contested_margin'))} "
        f"tie_band_used={_fmt(d.get('tie_band_used'))} "
        f"(ar_top2={_fmt(d['ar_top2_margin'])} dsp_top2={_fmt(d['dspark_top2_margin'])}) "
        f"Δ@ar_tok={_fmt(d.get('delta_at_ar_token'))} Δ@dsp_tok={_fmt(d.get('delta_at_dspark_token'))} "
        f"max|Δlogit|={_fmt(d['max_abs_logit_delta'])} rows_consistent={d.get('rows_consistent')}"
    )
    if d["class"] == "tie_flip":
        print(
            f"[ab] dspark divergence @ {i} class=tie_flip (acceptable) "
            f"rule={_dspark_divergence_rule(d)} {margins} "
            f"ar_tok={d['ar_token']} dsp_tok={d['dspark_token']} (arm {arm!r})",
            flush=True,
        )
    else:
        print(
            "[ab] " + "!" * 8 + " DIVERGENT " + "!" * 8 + "\n"
            f"[ab] DSpark greedy stream != AR @ {i} class=DIVERGENT (arm {arm!r}): "
            f"NOT a tie-break flip -- rule={_dspark_divergence_rule(d)} {margins}; "
            f"ar_tok={d['ar_token']} dsp_tok={d['dspark_token']}. "
            "Investigate the lane (or run --dspark-require-lossless to gate).",
            flush=True,
        )


def _ar_logits_row_at_index(*, model, ops, mx, prompt_ids, ar_tokens, index):
    """W77: faithfully replay the AR (M=1) decode forward whose greedy argmax is
    ``ar_tokens[index]`` and return its FULL logits row as a np.float32 vector.

    ``index == 0`` is the prompt-prefill argmax (the shared prefill logits).  For
    ``index >= 1`` this re-prefills the prompt and steps ``index`` M=1 forwards
    feeding ``ar_tokens[0..index-1]`` -- the exact one-row decode shape AR used, so
    the captured logits carry the same M=1 kernel rounding (not a re-prefill of the
    whole prefix, which would be a different matmul shape).  Bounded by ``index <=
    decode_tokens`` M=1 forwards + one prefill; only ever run once, on divergence.
    """
    cache = model.make_cache()
    logits = model(ops.input([list(prompt_ids)]), cache=cache, logits_keep=1)
    ops.sync(logits)
    for j in range(int(index)):
        logits = model(ops.input([[int(ar_tokens[j])]]), cache=cache)
        ops.sync(logits)
    return np.asarray(logits[0, -1].astype(mx.float32)).reshape(-1)


def _dspark_decode_wall_accounting(
    *, pass_start: float, decode_start: "float | None", pass_end: float,
    generated_tokens: int,
) -> dict:
    """Split a DSpark pass's wall clock into the re-prefill and the decode loop.

    ``dspark_generate`` re-prefills the prompt and then runs the decode cycles in
    one call, so timing the whole call folds the re-prefill (the pass's TTFT) into
    what was reported as ``decode_wall_s`` -- W100: window 39 divided 257 tokens by
    a 297 s wall (0.86 tok/s) whose decode phase was only ~93 s (2.75 tok/s). This
    helper takes ``pass_start`` (before ``dspark_generate``), ``decode_start`` (the
    instant the prefill callback fired -- prefill done, before the first draft), and
    ``pass_end`` (after the call), and returns the DECODE-ONLY wall plus the full
    pass wall so nothing is lost:

    * ``pass_wall_s`` -- the whole call (prefill + decode), the old ``decode_wall_s``.
    * ``decode_wall_s`` -- ``pass_end - decode_start`` (excludes the re-prefill);
      falls back to the full pass wall when no prefill callback fired.
    * ``decode_tok_s`` -- ``generated_tokens / decode_wall_s`` (None if the wall is
      non-positive).
    """
    pass_wall_s = max(0.0, pass_end - pass_start)
    if decode_start is None:
        decode_wall_s = pass_wall_s
    else:
        decode_wall_s = max(0.0, pass_end - decode_start)
    decode_tok_s = (generated_tokens / decode_wall_s) if decode_wall_s > 0 else None
    return {
        "pass_wall_s": pass_wall_s,
        "decode_wall_s": decode_wall_s,
        "decode_tok_s": decode_tok_s,
    }


def _reset_dspark_engagement_counters() -> dict:
    """W115: zero the decode-attention-core and fused-projection engagement counters
    before a dspark headline pass so the census is SCOPED to that pass.  The top-level
    ``*_engagement`` receipt blocks are AR-scoped (read after ``_generate``); without a
    dspark-scoped re-read the K+1 VERIFY rows are never counted (the window-43 receipt's
    ``fused_proj_engagement.rows = 10240`` was the AR pass only).  Best-effort: a module
    absent on an older build is skipped; the returned dict says which were reset so
    :func:`_capture_dspark_engagement` reports only those."""
    ok: dict = {}
    try:
        from mtplx.models import deepseek_v41 as _dsv41
        _dsv41._reset_attn_core_compile_calls()
        ok["dsv41"] = True
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        from mtplx.models import deepseek_v41_attn_kernels as _k29
        _k29.reset_engagement()
        ok["k29"] = True
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        from mtplx.models import deepseek_v41_fused_proj_kernels as _fp
        _fp.reset_engagement()
        ok["fp"] = True
    except Exception:  # pragma: no cover - defensive
        pass
    return ok


def _capture_dspark_engagement(reset_ok: dict) -> dict:
    """W115: read the verify-scoped engagement counters after a dspark headline pass.

    ``decode_attn_kernel_engagement`` (K29 dispatches -- the GPU verify core, ON by
    default via the _run_arm setdefault, so ``calls == 0`` proves an eager arm turned it
    off), ``attn_core_compile_engagement`` (the compiled selected-key core), and
    ``fused_proj_engagement`` (W101 qkv/out calls + rows -- now INCLUDING the verify
    rows, the window-43 gap)."""
    out: dict = {}
    if reset_ok.get("dsv41"):
        from mtplx.models import deepseek_v41 as _dsv41
        out["attn_core_compile_engagement"] = _dsv41._attn_core_compile_calls()
    if reset_ok.get("k29"):
        from mtplx.models import deepseek_v41_attn_kernels as _k29
        out["decode_attn_kernel_engagement"] = _k29.engagement()
    if reset_ok.get("fp"):
        from mtplx.models import deepseek_v41_fused_proj_kernels as _fp
        out["fused_proj_engagement"] = _fp.engagement()
    return out


def _generate_dspark(*, model, mx, mem_probe, prompt_ids, steps, depth,
                     verify_chunks=None, stage_timing=False, ar_reference=None,
                     stop_ids=None):
    """Greedy DSpark-DIRECT prefill + ``steps`` decode; captures tokens and the
    per-cycle accept + phase-timing statistics.  Total tokens == steps + 1 to match
    ``_generate`` (prefill token + ``steps`` decode tokens).

    W91: the HEADLINE (tok/s, tokens, stats, peak) is an UNTIMED pass so fused
    decode levers (K35 small-stages) are active for the number the receipt reports
    -- arming the W37 probe forces ``_small_stages_use`` eager (the recording
    guard), which would measure K35 OFF.  When ``stage_timing`` a SEPARATE, timed
    second pass re-runs the decode purely for the VERIFY forward's internal stage
    breakdown (attention, moe.routed_switch) + W61 engagement; its tok/s is
    discarded.  Mirrors the AR path (untimed ``_generate`` + a separate
    ``_stage_timing_pass``)."""
    from mtplx.models.deepseek_v41_dspark_decode import (
        DivergenceCapture,
        DSparkDecodeStats,
        dspark_generate,
    )
    from mtplx.sampling import SamplerConfig

    mem_probe.reset_peak()
    # W106: 1 Hz off-hot-path phys_footprint + system-used sampler over the headline pass (see
    # _generate). Stopped right after the headline peak_gb is captured, before the
    # optional timed stage-timing pass, so the memory block matches that peak_gb.
    _mem_sampler = mem_probe.new_sampler()
    _mem_sampler.start()
    # W118 review MEDIUM-2: allocator active/cache readback around the timed pass.
    _active_start_bytes = _active_end_bytes = _cache_end_bytes = None
    try:
        stats = DSparkDecodeStats()
        # W77: when an AR reference is supplied, capture (zero extra forwards) the
        # verify logits row of the first committed token that diverges from it.
        capture = DivergenceCapture(ar_reference) if ar_reference is not None else None
        route_probe = None
        route_prev_enabled = None
        # W81: snapshot the expert-streaming counters at the prefill->decode boundary
        # (prefill_callback fires after prefill, before the decode cycles) and again
        # after the pass, so the receipt carries a DECODE-scoped serve_stream_counters
        # block (hit rate + streamed bytes/token) matching the served daemon's.
        _sc: dict = {}

        def _stream_prefill_cb(_info):
            _sc["after_prefill"] = _stream_counters_snapshot(model)
            # W100: mark the prefill->decode boundary so decode_wall_s can exclude the
            # re-prefill (this callback fires after prefill, before the decode cycles).
            _sc["decode_start"] = time.perf_counter()

        def _stream_complete_cb():
            nonlocal _active_end_bytes, _cache_end_bytes
            # Freeze generation timing before any diagnostic work, while the
            # DSpark-owned target/draft caches are still live.
            _sc["pass_end"] = time.perf_counter()
            _active_end_bytes = _mlx_call_int(mx, "get_active_memory")
            _cache_end_bytes = _mlx_call_int(mx, "get_cache_memory")
            _mem_sampler.stop()

        # W91: HEADLINE pass is UNTIMED (no W37 recording armed) so fused decode levers
        # (K35 small-stages) are ACTIVE for the tok/s the receipt reports.  Arming
        # ``_stime`` forces ``_small_stages_use`` eager (the recording guard), so a timed
        # headline would measure K35 OFF while the arm env says ON.  Mirrors the AR path
        # (untimed ``_generate`` headline + a separate ``_stage_timing_pass``): the
        # per-stage attribution is a SECOND, timed pass below.
        # W115: zero the decode-core / fused-proj engagement counters right before the
        # UNTIMED headline pass so the dspark receipt block reports THIS pass's verify
        # engagement (the K29 core the verify rows drove + the W101 projections),
        # captured right after it.  Scoped to the headline pass for two reasons: (a) the
        # optional timed stage-timing pass re-runs the decode and would DOUBLE the
        # counts, and (b) that pass forces the COMPILE levers (K35 small-stages, the
        # core-compile tape) eager via their recording guard -- K29 itself has no such
        # guard and still dispatches under timing, but the double-count alone is reason
        # enough to capture before it.  Closes the window-43 "fused_proj = AR only" gap.
        _eng_reset = _reset_dspark_engagement_counters()
        # W118 review MEDIUM-2: allocator active readback just before the timed pass.
        _active_start_bytes = _mlx_call_int(mx, "get_active_memory")
        t0 = time.perf_counter()
        toks = dspark_generate(
            model,
            [int(t) for t in prompt_ids],
            max_tokens=int(steps) + 1,
            sampler=SamplerConfig(temperature=0.0),
            seed=0,
            speculative_depth=int(depth),
            verify_chunks=verify_chunks,
            stats=stats,
            divergence_capture=capture,
            prefill_callback=_stream_prefill_cb,
            completion_callback=_stream_complete_cb,
            stop_ids=stop_ids,  # W113: served-parity early stop (--stop-on-eos)
        )
        _sc["end"] = _stream_counters_snapshot(model)
        # W115: capture the verify-scoped engagement from the headline pass.
        _dspark_engagement = _capture_dspark_engagement(_eng_reset)
        # W118 review MEDIUM-2: active/cache at decode end (timed pass complete).
        # W100: exclude the re-prefill from decode_wall_s (the decode loop only, from
        # the prefill->decode boundary). Keep the whole-call wall as pass_wall_s.
        _wall_acct = _dspark_decode_wall_accounting(
            pass_start=t0,
            decode_start=_sc.get("decode_start"),
            pass_end=_sc.get("pass_end", time.perf_counter()),
            # W113 LOW-b: DECODE-only token count (exclude the prefill/first token)
            # so dspark decode_tok_s uses the SAME denominator as the AR lane
            # (decode_steps_run), instead of steps+1.  Also correct under
            # --stop-on-eos, where len(toks) is the truncated stream.
            generated_tokens=max(0, len(toks) - 1),
        )
        peak_gb = mem_probe.peak_bytes() / 1_000_000_000  # headline peak, captured before the timed pass
    finally:
        _mem_sampler.stop()
    # W118 review MEDIUM-2: merge the allocator readback proof keys into the block.
    _dspark_memory_block = _ab_memory_block(
        mem_probe.memory_block(_mem_sampler),
        _mlx_headroom_readback_keys(
            mx,
            mlx_peak_bytes=mem_probe.peak_bytes(),
            active_start_bytes=_active_start_bytes,
            active_end_bytes=_active_end_bytes,
            cache_end_bytes=_cache_end_bytes,
        ),
    )
    report = None
    w61 = None
    if stage_timing:
        # SECOND pass: arm the W37 probe + the route-stage probe (hot.* per-layer
        # counters incl. W61's hot.verify_single_barrier engagement), clearing it so
        # the census is scoped to this pass, then re-run the decode purely for the
        # verify forward's internal stage breakdown + W61 engagement.  Its tok/s is
        # discarded -- K35 is forced eager here (the recording guard), which is
        # exactly what fenced per-stage attribution needs.  ENABLED is read at use,
        # so setting it here arms the module even if the launch env did not.
        from mtplx.models import deepseek_v41_stage_timing as stime
        try:
            from mtplx import expert_route_probe as route_probe

            route_prev_enabled = route_probe.ENABLED
            route_probe.ENABLED = True
            route_probe._SUMS.clear()
            route_probe._COUNTS.clear()
        except Exception:
            route_probe = None
        stime.begin()
        try:
            dspark_generate(
                model,
                [int(t) for t in prompt_ids],
                max_tokens=int(steps) + 1,
                sampler=SamplerConfig(temperature=0.0),
                seed=0,
                speculative_depth=int(depth),
                verify_chunks=verify_chunks,
                stats=DSparkDecodeStats(),
                stop_ids=stop_ids,  # W113: match the headline pass
            )
            report = model.stage_timing_report()
        finally:
            stime.end()  # never leave the probe armed if the timed pass raises
    if route_probe is not None:
        snap = route_probe.snapshot()
        stg = snap.get("stages", {}) if isinstance(snap, dict) else {}

        def _count(name):
            return int(stg.get(name, {}).get("count", 0))

        # W61 engages (one routing barrier/layer) on an ALL-HIT verify; W66/W81
        # engage the split single-barrier / batched path on a MISS verify.  W81
        # adds a per-verify engagement census: every small-M (2..8 row) DECODE
        # component-bank route is a candidate (hot.verify_candidate); it either
        # engages (all-hit W61 or split W66/W81) or declines with a reason.
        _candidate = _count("hot.verify_candidate")
        _eng_all_hit = _count("hot.verify_single_barrier")
        _eng_split = _count("hot.verify_single_barrier_split")
        _eng_batched = _count("hot.verify_single_barrier_batched")
        _engaged = _eng_all_hit + _eng_split
        _decline_reasons = {
            "flag_off": _count("hot.verify_decline.flag_off"),
            "codec": _count("hot.verify_decline.codec"),
            "shadow_bank": _count("hot.verify_decline.shadow_bank"),
            "assignment_shape": _count("hot.verify_decline.assignment_shape"),
            "no_defer_seam": _count("hot.verify_decline.no_defer_seam"),
            "other": _count("hot.verify_decline.other"),
        }
        _declined = sum(_decline_reasons.values())
        w61 = {
            "verify_single_barrier": _eng_all_hit,
            "verify_single_barrier_split": _eng_split,
            "verify_single_barrier_batched": _eng_batched,
            "all_hit": _count("hot.all_hit"),
            "try_all_hit": _count("hot.try_all_hit"),
            "eval_indices": _count("hot.eval_indices"),
            "begin_split_route": _count("hot.begin_split_route"),
            # W81 per-verify engagement census (the next window reads this line):
            "verify_candidates": _candidate,
            "verify_engaged": _engaged,
            "verify_declined": _declined,
            "verify_engaged_pct": (
                round(100.0 * _engaged / _candidate, 2) if _candidate else None
            ),
            "verify_decline_reasons": _decline_reasons,
            "note": "cumulative over this dspark pass (prefill + drafts + verifies); "
                    "verify_* counters are verify-shape-only (2..8 rows, DECODE, "
                    "component-banks). engaged = all_hit(W61) + split(W66/W81); "
                    "split includes batched (unique > transient capacity).",
        }
        # Human-readable census line so the receipt scrape and console both show it.
        print(
            "[ab] verify engagement: "
            f"{_engaged}/{_candidate} engaged "
            f"({w61['verify_engaged_pct']}%) "
            f"[all_hit={_eng_all_hit} split={_eng_split} batched={_eng_batched}] "
            f"declined={_declined} reasons={_decline_reasons}",
            flush=True,
        )
        route_probe.ENABLED = bool(route_prev_enabled)
    out = {
        "generated": [int(t) for t in toks],
        # W100: decode-only wall (re-prefill excluded); pass_wall_s keeps the old
        # whole-call figure; decode_tok_s = generated / decode_wall_s.
        "decode_wall_s": _wall_acct["decode_wall_s"],
        "pass_wall_s": _wall_acct["pass_wall_s"],
        "decode_tok_s": _wall_acct["decode_tok_s"],
        "peak_gb": peak_gb,  # W91: headline (untimed) peak, not the timed 2nd pass
        # W106: full memory envelope for the headline pass (see _generate).
        "memory": _dspark_memory_block,
        "stats": stats.to_dict(),
        "stream_after_prefill": _sc.get("after_prefill"),
        "stream_end": _sc.get("end"),
        # W115: verify-scoped engagement from the headline pass (see the receipt block).
        "engagement": _dspark_engagement,
        # W95f: the v2 runner + gate_prefetch receipt blocks (present only when armed).
        **_runner_receipt_blocks(model),
    }
    if report is not None:
        out["verify_stage_timing"] = report
    if w61 is not None:
        out["w61_engagement"] = w61
    if capture is not None and capture.found:
        # In-process only: the full logits row stays out of the receipt; the AB
        # harness reads scalars off it via classify_divergence.
        out["divergence"] = {
            "index": capture.index,
            "ar_token": capture.ar_token,
            "dspark_token": capture.dspark_token,
            "dspark_logits_row": capture.dspark_logits_row,
            "dspark_top2_margin": capture.dspark_top2_margin,
        }
    return out


def _overlap_telemetry(runtime) -> dict | None:
    """Best-effort GPU-overlap census off the runtime slot metrics.

    ``overlap_gpu_dispatch_ns`` (work issued while miss reads were open) and
    ``overlap_exposed_wait_ns`` (residual blocking wait the overlap could not
    hide) populate only when ``overlap_miss_reads`` is armed; return None
    otherwise so the arm never fabricates an idle figure.
    """
    metrics = getattr(getattr(runtime, "slots", None), "metrics", None)
    snap = getattr(metrics, "snapshot", None)
    data = snap() if callable(snap) else getattr(metrics, "__dict__", None)
    if not isinstance(data, dict):
        return None
    dispatch = data.get("overlap_gpu_dispatch_ns")
    exposed = data.get("overlap_exposed_wait_ns")
    if not dispatch and not exposed:
        return None
    total = (dispatch or 0) + (exposed or 0)
    return {
        "overlap_gpu_dispatch_ns": dispatch,
        "overlap_exposed_wait_ns": exposed,
        "gpu_idle_share": (exposed / total) if total else None,
    }


def _pin_telemetry(runtime) -> dict | None:
    """W64 pinned-working-set telemetry off the runtime (pinned count per layer +
    the all-pinned-hit rate per decode route). None when the runtime lacks the
    W64 surface (older build) or the lever left it empty."""
    getter = getattr(runtime, "pinned_working_set_telemetry", None)
    if not callable(getter):
        return None
    try:
        tel = getter()
    except Exception:
        return None
    if not isinstance(tel, dict):
        return None
    # Keep the full block only when the lever ran; otherwise a compact off marker.
    if not tel.get("enabled") and not tel.get("decode_routes"):
        return {"enabled": False}
    return tel


def _device_route_pinned_telemetry(runtime) -> dict | None:
    """W71 pinned-device-route barrier-free-layer telemetry off the runtime
    (barrier-free / recovered layers per flush). None when the runtime lacks the
    W71 surface (older build); a compact off marker when the lever never ran."""
    getter = getattr(runtime, "device_route_pinned_telemetry", None)
    if not callable(getter):
        return None
    try:
        tel = getter()
    except Exception:
        return None
    if not isinstance(tel, dict):
        return None
    if not tel.get("enabled") and not tel.get("flushes"):
        return {"enabled": False}
    return tel


def _peak_process_gb(run) -> float | None:
    """The whole-PROCESS phys_footprint peak (incl. the non-Metal Python heap +
    expert-reader buffers) from the in-process 1 Hz sampler, for the top-level receipt
    key ``peak_process_gb`` (decimal GB).  David's "peak memory must include non-Metal
    parts" fix: the legacy ``peak_gb`` is the MLX allocator peak only.  ``None`` when a
    pass produced no memory block."""

    mem = (run or {}).get("memory") or {}
    val = mem.get("process_footprint_peak_gb")
    return None if val is None else float(val)


def _memory_headline(receipt) -> str:
    """Print measured peaks in decimal GB, with unavailable readings explicit."""
    mem = receipt.get("memory") or {}
    def fmt(key):
        value = mem.get(key)
        return "n/a" if value is None else f"{value:.2f}"
    return (
        f"mlx_peak_gb={fmt('mlx_peak_gb')}"
        f" process_footprint_peak_gb={fmt('process_footprint_peak_gb')}"
        f" box_used_gb={fmt('box_used_gb')} (includes file cache)"
        f" baseline_plus_process_estimate_gb={fmt('baseline_plus_process_peak_estimate_gb')}"
    )


# --------------------------------------------------------------------------
# W106 output persistence (David: "store the output so we can audit it").  Every
# bench run persists the FULL generated output -- in the receipt (token_ids +
# decoded_text + head/tail) and as a text sidecar next to the receipt -- so a
# rounding-class result can be text-spot-checked, not just compared by sha.
# Decoding reuses the ALREADY-LOADED bench tokenizer and is fully guarded: a
# tokenizer failure records None and never kills the measured run.
# --------------------------------------------------------------------------
_TEXT_HEAD_CHARS = 600
_TEXT_TAIL_CHARS = 600
_DIVERGENCE_CONTEXT_CHARS = 200


def _decode_ids(tok, ids):
    """Decode token ids to text.  Returns ``(text_or_None, error_or_None)``: never a
    SILENT empty string.  Tries the tokenizer's ``decode`` then the underlying HF
    ``_tokenizer.decode``; a raise records its repr.

    LOW (round 4): an empty result is only an ERROR when it is a genuine decode
    failure.  If ``decode(ids)`` is empty but ``decode(ids, skip_special_tokens=
    False)`` is NON-empty, the ids render only as SPECIAL tokens (e.g. an
    all-EOS/pad stream) -- a LEGITIMATELY empty decoded text, not a failure: return
    that with-specials rendering (so the audit shows what was produced) and no
    error.  Only when BOTH are empty is it recorded as an error."""

    if ids is None:
        return None, "no ids"
    if tok is None:
        return None, "no tokenizer available for output decode"
    ids_int = [int(t) for t in ids]
    last_err = None
    candidates = (
        ("decode", getattr(tok, "decode", None)),
        ("_tokenizer.decode", getattr(getattr(tok, "_tokenizer", None), "decode", None)),
    )
    for label, fn in candidates:
        if not callable(fn):
            continue
        try:
            text = fn(ids_int)
        except Exception as exc:
            last_err = f"{label} raised {exc!r}"
            continue
        if text:  # non-empty string -> success
            return text, None
        # Empty: distinguish a LEGITIMATE special-tokens-only decode from a broken
        # one.  If the with-specials rendering is non-empty, the ids are special
        # tokens -> legit empty; surface that rendering, no error.
        try:
            with_specials = fn(ids_int, skip_special_tokens=False)
        except Exception:
            with_specials = None
        if with_specials:
            return with_specials, None
        last_err = f"{label} returned empty for {len(ids_int)} ids"
    return None, (last_err or "no usable decode method on the tokenizer")


def _text_output_fields(tok, ids) -> dict:
    """The receipt text-audit fields for one id stream: the FULL id list, the full
    decoded text + head/tail (first/last 600 chars), and ``decoded_text_error`` (the
    reason decode produced no text, or None on success).  A failure is LOUD."""

    ids_list = [int(t) for t in (ids or [])]
    text, err = _decode_ids(tok, ids_list)
    if err is not None:
        print(
            f"[ab] WARN: output decode produced no text ({err}); "
            f"token_ids present ({len(ids_list)}), decoded_text_error recorded",
            flush=True,
        )
    if text is None:
        head = tail = None
    else:
        head = text[:_TEXT_HEAD_CHARS]
        tail = text[-_TEXT_TAIL_CHARS:]
    return {
        "token_ids": ids_list,
        "decoded_text": text,
        "decoded_text_head": head,
        "decoded_text_tail": tail,
        "decoded_text_error": err,
    }


def _divergence_context(tok, ids, token_index, span=_DIVERGENCE_CONTEXT_CHARS):
    """The decoded text ``span`` chars either side of the character offset that the
    divergence TOKEN index maps to (decode the prefix to find the offset).  None
    when the stream cannot be decoded."""

    full, _ = _decode_ids(tok, ids)
    if full is None:
        return None
    prefix, _ = _decode_ids(tok, list(ids)[: max(0, int(token_index))])
    offset = len(prefix) if prefix is not None else 0
    return full[max(0, offset - span): offset + span]


def _receipt_stem(out_path) -> Path:
    """The receipt path with a trailing ``.jsonl``/``.json`` stripped, so
    ``<stem>.output.txt`` sits beside the receipt."""

    p = Path(out_path)
    if p.suffix in (".jsonl", ".json"):
        return p.with_suffix("")
    return p


def _suffixed(path: Path, n: int) -> Path:
    """``path`` for n==1, else ``<stem>-n<suffix>`` (e.g. ``x.output-2.txt``)."""

    return path if n == 1 else path.with_name(f"{path.stem}-{n}{path.suffix}")


def _reserve_paired(paths):
    """Find the SMALLEST n for which every path in ``paths`` (at suffix n) is free,
    and atomically reserve them all (O_CREAT|O_EXCL empty files); the SAME n is
    applied to every path so a receipt's sidecars stay paired
    (memory/never-overwrite-a-measurement).  Returns the reserved paths (in order),
    or None on exhaustion/failure."""

    n = 1
    while n < 100000:
        cands = [_suffixed(p, n) for p in paths]
        reserved = []
        clash = False
        for cand in cands:
            try:
                fd = os.open(cand, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                clash = True
                break
            os.close(fd)
            reserved.append(cand)
        if clash:
            for r in reserved:  # release partial reservations before trying n+1
                try:
                    os.unlink(r)
                except OSError:
                    pass
            n += 1
            continue
        return cands
    return None


def _write_reserved(path: Path, content: str) -> None:
    """Atomically write ``content`` over an already-reserved ``path`` (tmp + rename)."""

    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(content)
    os.replace(tmp, path)


def _sidecar_text(*, arm, kind, stream, divergence) -> str:
    sha = stream.get("token_ids_sha256")
    tok_s = stream.get("decode_tok_s")
    text = stream.get("decoded_text")
    header = "\n".join(
        [
            f"# arm: {arm}",
            f"# stream: {kind}",
            f"# token_ids_sha256: {sha}",
            f"# decode_tok_s: {tok_s}",
            "# divergence: "
            + (json.dumps(divergence) if divergence is not None else "none"),
            "",
            "",
        ]
    )
    body = (
        text if text is not None
        else "<decode unavailable (no tokenizer / decode failed)>"
    )
    return header + body + "\n"


def _abort_receipt_row(arm, exc, args=None) -> dict:
    """W106 MEDIUM-C: the ledger row for an arm that aborted before producing a
    receipt (a budget/re-measure abort or a floor refusal).  Carries the arm, the
    failure reason + stage, and (best-effort) the budget derivation captured so far
    so the ledger shows why."""

    stage = getattr(exc, "dsv41_stage", "run_arm")
    row = {
        "arm": arm,
        "aborted": True,
        "reason": str(exc),
        "stage": stage,
        "exception": type(exc).__name__,
        # LOW: a human note so the ledger row is self-explanatory (this arm produced
        # no measurement; the run exited 4 -- chain later arms with `&&`, not `;`).
        "note": (
            "arm aborted before producing a receipt; no measurement recorded. "
            "The bench exited 4 at this arm -- with `&&` chaining the launcher stops "
            "here rather than re-loading + re-aborting every later step."
        ),
    }
    tp = getattr(args, "_dsv41_target_plan", None) if args is not None else None
    if tp is not None:
        row["memory"] = {
            "memory_plan_source": "box_target",
            "box_target_gb": tp.get("box_target_gb"),
            "box_baseline_gb": tp.get("box_baseline_gb"),
            "engine_budget_gib": round(tp.get("engine_budget_gib", 0.0), 4),
        }
    return row


def _append_receipt_row(out_path, row) -> None:
    """Append one JSONL row to the append-only receipt (MEDIUM-C)."""

    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(out_path).open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError as exc:  # pragma: no cover - defensive
        print(f"[ab] WARN: could not append abort row ({exc!r})", flush=True)


def _write_output_sidecars(out_path, receipt) -> None:
    """Persist the FULL decoded output beside the receipt (MEDIUM-3):
    ``<stem>.<sha12>.output.txt`` for the measured stream and, for a DSpark run,
    ``<stem>.<sha12>.ar-reference.output.txt`` for the AR comparison stream.  The
    sha[:12] of the AR pass is in BOTH names (pairs them, and disambiguates arms),
    and a single ``-n`` suffix is applied to BOTH when a name is taken, so the pair
    never splits.  Fully guarded."""

    try:
        stem = _receipt_stem(out_path)
        arm = receipt.get("arm")
        sha12 = str(receipt.get("token_ids_sha256") or "nosha")[:12]
        base = f"{stem.name}.{sha12}"
        dsp = receipt.get("dspark")
        if isinstance(dsp, dict):
            primary_stream, primary_kind, div = dsp, "dspark", dsp.get("divergence")
        else:
            primary_stream, primary_kind, div = receipt, "ar", None

        want = [stem.with_name(base + ".output.txt")]
        if isinstance(dsp, dict):
            want.append(stem.with_name(base + ".ar-reference.output.txt"))

        reserved = _reserve_paired(want)
        if reserved is None:
            print("[ab] WARN: output sidecar names exhausted; skipping", flush=True)
            return

        _write_reserved(
            reserved[0],
            _sidecar_text(arm=arm, kind=primary_kind, stream=primary_stream,
                          divergence=div),
        )
        print(f"[ab] output sidecar: {reserved[0]}", flush=True)
        if isinstance(dsp, dict):
            _write_reserved(
                reserved[1],
                _sidecar_text(arm=arm, kind="ar-reference", stream=receipt,
                              divergence=div),
            )
            print(f"[ab] output sidecar: {reserved[1]}", flush=True)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[ab] WARN: output sidecar step failed ({exc!r})", flush=True)


def _run_arm(args, arm, bench, mx) -> dict:
    _apply_arm_env(arm)
    # W118 (H7): map an explicit --mlx-limit-headroom-gib to the env AFTER the arm
    # preset is applied (a non-hr arm's preset POPs the key), so the flag overrides the
    # preset and the runtime (apply_mlx_memory_cap, read at use) sees the same value the
    # harness prices into the forecast.  Unset flag -> the preset/env value stands.
    _hr_flag = getattr(args, "mlx_limit_headroom_gib", None)
    if _hr_flag is not None:
        # LOW: warn when the flag overrides a preset that did NOT arm the headroom, so
        # a mismatched-arm run (e.g. --arms cell16k_ring_v2_attn --mlx-limit-headroom-gib
        # 8) is not silently a different lever set than the named arm implies.
        _preset_hr = ARM_PRESETS.get(arm, {}).get(MLX_LIMIT_HEADROOM_ENV)
        if _preset_hr in (None, ""):
            print(
                f"[ab] WARN: --mlx-limit-headroom-gib {float(_hr_flag):g} overrides arm "
                f"{arm!r} which does NOT arm the headroom (preset value None); this arm "
                "is no longer byte-identical to the same-named arm at headroom 0.",
                flush=True,
            )
        os.environ[MLX_LIMIT_HEADROOM_ENV] = f"{float(_hr_flag):g}"
    # W107: a bounded arm that did not pin an explicit MTPLX_DSV41_KV_BOUNDED_MAXKV
    # (the presets do not know the CLI --max-kv) preallocates every KV lane to the
    # resolved cell max_kv.  Stamp it here, after the arm env is applied and BEFORE
    # make_cache (per request), so the cache reads it at construction.  Read-at-use,
    # not import.  A bounded arm with neither key set falls back to geometric growth
    # (the kv_realloc_* counters then flag it).
    #
    # W113 fix: SKIP the stamp on the --dry-run path.  The CPU-only dry-run double
    # never builds the cache (nothing reads KV_BOUNDED_MAXKV), and
    # ``bench.resolve_max_kv`` RAISES for a large cell with no explicit --max-kv (the
    # default 4096 is below the 16704 the 16K cell needs), which would abort a dry-run
    # of a KV-bounded arm before its early return below.  Guarding the stamp keeps the
    # dry-run resolution double CPU-safe for bounded arms at any --context-tokens.
    if (
        not getattr(args, "dry_run", False)
        and (os.environ.get(KV_BOUNDED_ENV) or "").strip().lower()
        in ("1", "true", "yes", "on")
    ):
        if not (os.environ.get(KV_BOUNDED_MAXKV_ENV) or "").strip():
            _bounded_max_kv = bench.resolve_max_kv(
                [args.context_tokens], args.decode_tokens, args.max_kv
            )
            os.environ[KV_BOUNDED_MAXKV_ENV] = str(int(_bounded_max_kv))
    if getattr(args, "decode_mode", "ar") == "dspark":
        # Arm K29 (fused decode/verify attention, b*s<=8) + K30 (selected keys) for
        # the WHOLE arm so both the AR reference (_generate) and the dspark verify
        # use the decode attention branch consistently -- they are greedy-identical
        # to the eager path, so byte_identical_vs_ar holds only if both share the
        # setting. setdefault respects an arm that set them explicitly;
        # MTPLX_DSV41_DSPARK_DECODE_KERNELS=0 opts out.
        from mtplx.models.deepseek_v41_dspark_decode import (
            dspark_decode_kernel_env_defaults,
        )

        for _k, _v in dspark_decode_kernel_env_defaults().items():
            os.environ.setdefault(_k, _v)
        # MEDIUM-7: a bounded-KV DSpark run verifies K+1 rows against the compressor
        # frontier, which the bounded latent lane preallocates with only
        # _BOUNDED_LATENT_SLACK rows of verify margin.  depth + 1 > that slack would
        # overrun the preallocated cap mid-verify -- refuse with a clean error.
        if (os.environ.get(KV_BOUNDED_ENV) or "").strip().lower() in ("1", "true", "yes", "on"):
            from mtplx.models.deepseek_v41_cache import _BOUNDED_LATENT_SLACK
            _depth = int(getattr(args, "dspark_depth", 0) or 0)
            if _depth + 1 > _BOUNDED_LATENT_SLACK:
                raise SystemExit(
                    f"[ab] --dspark-depth {_depth} with bounded KV needs depth + 1 "
                    f"({_depth + 1}) <= the bounded latent verify slack "
                    f"{_BOUNDED_LATENT_SLACK}; lower --dspark-depth or unset "
                    f"MTPLX_DSV41_KV_BOUNDED."
                )
    if getattr(args, "dry_run", False):
        return _dry_run_arm(args, arm, bench)
    build_prompt = bench._load_build_prompt()
    _tok = None if getattr(args, "prompt_ids_file", None) else _tokenizer(args, bench)
    prompt_ids, prompt_meta = bench._resolve_prompt(
        args, _tok, build_prompt, args.context_tokens
    )
    # W113: resolve the EOS id (tokenizer files, no model) for the EOS-surfacing
    # receipt fields and the optional --stop-on-eos served-parity early stop.
    eos_id = _resolve_eos_id(args)
    stop_on_eos = bool(getattr(args, "stop_on_eos", False))
    # W113 MEDIUM-3: --stop-on-eos with no resolvable EOS id must refuse (not
    # silently no-op while stamping stop_on_eos:true).  Before the model load.
    _require_eos_id_for_stop(stop_on_eos, eos_id)
    # W106 output persistence (window-43 fix): the prompt path leaves _tok None when
    # --prompt-ids-file supplies the prompt ids -- but the OUTPUT still needs a
    # tokenizer to be decoded for the audit.  Always obtain one for decoding
    # (reusing _tok when present), guarded, so a decode is never silently skipped.
    _out_tok = _tok
    if _out_tok is None:
        try:
            _out_tok = _tokenizer(args, bench)
        except Exception as exc:  # pragma: no cover - defensive
            _out_tok = None
            print(
                "[ab] WARN: could not load a tokenizer for OUTPUT decode "
                f"({exc!r}); receipts will carry decoded_text_error",
                flush=True,
            )
    resident = _load_model(args, bench, mx)
    model = resident.model
    runtime = getattr(model, "_mtplx_expert_runtime", None)
    # W38/K3 Sinkhorn engagement: zero the module counters after model load so the
    # receipt reports THIS arm's kernel-vs-recurrence Sinkhorn split. A truthy
    # ``MTPLX_DSV41_SINKHORN_METAL`` arm whose ``kernel_calls`` is 0 means the
    # Metal kernel silently fell back to the recurrence (which would still be
    # byte-identical and ~as fast) -- exactly the "did it actually run?" question.
    # Read cumulatively over the whole arm because under ``MTPLX_DSV41_HC_COMPILE``
    # the Python wrapper runs only during the (cold) trace, not per warm token.
    try:
        from mtplx.models import deepseek_v41 as _dsv41
        _dsv41._reset_sinkhorn_kernel_calls()
        # W91/K35 fused small-stages + fused-premix-kernel engagement counters
        # (same "did it actually run?" question): zero them after model load so the
        # receipt reports THIS arm's real fused-layer forwards vs eager fallbacks.
        _dsv41._reset_small_stages_calls()
        _dsv41._reset_hc_premix_kernel_calls()
        # W97 (review item 3): zero the decode-attention-core compile engagement so
        # the receipt reports THIS arm's compiled-tape calls vs eager fallbacks.
        _dsv41._reset_attn_core_compile_calls()
    except Exception:  # pragma: no cover - defensive
        _dsv41 = None
    # W60/K29 engagement: zero the fused-decode-attention counters after model load
    # so the receipt reports THIS arm's real kernel dispatches vs armed-but-eager
    # fallbacks -- distinguishes "kernel did not run" (calls 0) from "ran (slowly)".
    try:
        from mtplx.models import deepseek_v41_attn_kernels as _k29
        _k29.reset_engagement()
    except Exception:  # pragma: no cover - defensive
        _k29 = None
    # W101/K36 engagement: zero the fused projection-chain counters after model load
    # so the receipt reports THIS arm's real fused-kernel dispatches (per phase) vs
    # armed-but-eager fallbacks.
    try:
        from mtplx.models import deepseek_v41_fused_proj_kernels as _fp
        _fp.reset_engagement()
    except Exception:  # pragma: no cover - defensive
        _fp = None
    # W73/K32 chunk-grow engagement: zero the cache telemetry after model load so
    # the receipt reports THIS arm's layer-backing choice + append counts.  enabled
    # == 0 means the flag did not reach cache construction (env timing / wrong
    # class); enabled > 0 with logical rows_copied ~flat while the cache_append
    # census stage is still O(T) means the slice_update did not donate on Metal.
    try:
        from mtplx.models import deepseek_v41_cache as _dsv41_cache
        _dsv41_cache.reset_kv_chunk_grow_stats()
        # W80 / K34: zero the window-ring telemetry too, so the receipt reports THIS
        # arm's ring engagement + drop/copy counts.
        _reset_ring = getattr(_dsv41_cache, "reset_window_ring_stats", None)
        if callable(_reset_ring):
            _reset_ring()
        # W107: zero the per-lane bounded-KV telemetry so the receipt reports THIS
        # arm's bounded engagement + per-lane in-place/realloc counts.
        _reset_bounded = getattr(_dsv41_cache, "reset_kv_bounded_stats", None)
        if callable(_reset_bounded):
            _reset_bounded()
    except Exception:  # pragma: no cover - defensive
        _dsv41_cache = None
    try:
        ops = bench._MLXOps(mx)
        mem_probe = bench._MLXMemProbe(mx)
        mem_profile_cb, mem_profile_snaps = _memory_profile_collector(
            args, mx, runtime, resident
        )
        device_sample = _device_sample_resolved(args)
        # W90: macmon utilization over the decode + optional post-prefill cooldown.
        util_sampler = None
        if getattr(args, "utilization", False):
            util_sampler = _macmon().UtilizationSampler(
                interval_ms=int(getattr(args, "util_interval_ms", 2000))
            )
        run = _generate(
            model=model,
            ops=ops,
            mem_probe=mem_probe,
            prompt_ids=prompt_ids,
            steps=args.decode_tokens,
            mem_profile=mem_profile_cb,
            mem_profile_every=int(getattr(args, "memory_profile_every", 64)),
            device_sample=device_sample,
            cooldown_s=float(getattr(args, "cooldown_s", 0.0) or 0.0),
            util_sampler=util_sampler,
            stage_timing=bool(getattr(args, "stage_timing", False)),
            stop_on_eos=stop_on_eos,
            eos_id=eos_id,
        )
        if util_sampler is not None:
            print(f"[ab] {arm}: {util_sampler.census()}", flush=True)
        ids = run["generated"]
        # W113: the number of DECODE tokens actually generated (excludes the prefill
        # argmax token).  == args.decode_tokens on the default full fixed-step run;
        # fewer only under --stop-on-eos.  decode_tok_s is reported over it so a
        # served-parity run's rate reflects the tokens actually produced.
        _decode_generated = int(run.get("decode_steps_run", args.decode_tokens))
        _ar_eos = _eos_surfacing(ids, eos_id)
        _warn_if_first_token_eos(arm, "AR", _ar_eos)
        receipt = {
            "arm": arm,
            # W97 (review item 7): True when this arm's tokens are EXPECTED to differ
            # from control by rounding (a rounding-class attention lever), so the
            # byte-identity summary reads a sha mismatch as "expected", not FAIL.
            # ``rounding_class_keys`` are the reason keys (see ROUNDING_CLASS_ENVS).
            "rounding_class": _is_rounding_class(arm),
            "rounding_class_keys": _rounding_class_keys(arm),
            "overlap_env": os.environ.get(OVERLAP_ENV),
            "arm_env": _arm_env_snapshot(),
            # K14 (W63): the MLX command-buffer MB cap in effect for this arm
            # (None = MLX default). MLX binds it at Metal init, so run one arm per
            # process with it exported for a real A/B; recorded for reproducibility.
            "mlx_max_mb_per_buffer": os.environ.get(MLX_MAX_MB_PER_BUFFER_ENV),
            # W63 / K32: whether the AR reference decode used the device-sample
            # one-step-lag pipeline (greedy byte-identical to the classic loop).
            "device_sample": bool(device_sample),
            "device_sample_extra_forwards": int(run.get("extra_forward_steps", 0)),
            "context_tokens": int(args.context_tokens),
            "decode_tokens": int(args.decode_tokens),
            # W113: DECODE tokens actually generated (== decode_tokens unless
            # --stop-on-eos stopped early); decode_tok_s is reported over it.
            "decode_tokens_generated": _decode_generated,
            "stop_on_eos": stop_on_eos,
            "prompt_tokens": len(prompt_ids),
            "ttft_s": run["ttft_s"],
            "prefill_tok_s": (len(prompt_ids) / run["ttft_s"])
            if run["ttft_s"] > 0
            else None,
            "decode_wall_s": run["decode_wall_s"],
            "decode_tok_s": (_decode_generated / run["decode_wall_s"])
            if (run["decode_wall_s"] > 0 and _decode_generated > 0)
            else None,
            # Schema v2: peak_gb is the MLX peak in decimal GB;
            # peak_process_gb is the whole-PROCESS footprint incl.
            # the non-Metal footprint (David's fix). Both come off run["memory"].
            "peak_gb": run["peak_gb"],
            "peak_process_gb": _peak_process_gb(run),
            # W106: the memory envelope (mlx_peak_gb == the peak_gb above, plus the
            # process phys_footprint peak and the whole-box used-memory peak/at-start the
            # gpu_window.sh guard measures). peak_gb alone is the MLX allocator peak
            # of this process -- it excludes the Python heap, the expert-reader
            # buffers, other processes, and the OS cache, so it is NOT the box usage.
            "memory": run.get("memory"),
            # W90: GPU DVFS/utilization over the decode + optional post-prefill
            # cooldown (the discriminator for the mode-independent in-situ floor).
            "utilization": run.get("utilization"),
            "cooldown": run.get("cooldown"),
            "token_ids_sha256": hashlib.sha256(
                json.dumps(ids).encode()
            ).hexdigest(),
            "first_token_ids": ids[:16],
            # W106 output persistence: the FULL generated ids + decoded text (+
            # head/tail) for the AR pass, so a rounding-class result is text-
            # auditable, not just sha-comparable. Decoded with the loaded bench
            # tokenizer; None if unavailable (guarded).
            **_text_output_fields(_out_tok, ids),
            "overlap_telemetry": _overlap_telemetry(runtime)
            if runtime is not None
            else None,
            # W64 R3-pin: pinned count per layer + all-pinned-hit rate per token
            # (the layer fraction a barrier-free device route could take race-free).
            "pin_working_set": _pin_telemetry(runtime)
            if runtime is not None
            else None,
            # W71: barrier-free layers per token the pinned device route actually
            # kept (vs recovered fenced) over this arm -- the K24-revived win read.
            "device_route_pinned": _device_route_pinned_telemetry(runtime)
            if runtime is not None
            else None,
            # W60/K29 fused-decode-attention engagement (calls/rows/split_calls/
            # fallbacks over the whole arm); calls 0 on a decode_attn_kernel arm
            # means the kernel never ran (all eager) rather than ran-and-was-slow.
            "decode_attn_kernel_engagement": (
                _k29.engagement() if _k29 is not None else None
            ),
            # W97 (review item 3): decode-attention-core compile engagement --
            # ``compiled`` selected-key core calls that ran the fixed-shape mx.compile
            # tape vs ``eager`` calls (lever off / above the small-M cap).  compiled 0
            # on an attn_core_compile arm means the tape never ran (all eager), so a
            # measured delta cannot be credited to it -- proves the tape engaged.
            "attn_core_compile_engagement": (
                _dsv41._attn_core_compile_calls() if _dsv41 is not None else None
            ),
            # W101/K36 fused projection-chain engagement (qkv_calls/out_calls/rows/
            # fallbacks over the whole arm); qkv_calls 0 on a fused arm means the
            # fused kernels never ran (all eager) rather than ran-and-was-slow.
            "fused_proj_engagement": (
                _fp.engagement() if _fp is not None else None
            ),
            # W81: the ACTUAL slot plan this arm ran (transient/persistent slot
            # counts + bytes + source), so an A/B is attributable to a capacity and
            # the in-process bench is comparable to the served profile plan.
            "resolved_plan": _resolved_plan(runtime, args),
            "resident_load_report": dict(
                getattr(model, "_mtplx_resident_load_report", resident.report.as_dict())
            ),
            # W81: DECODE-scoped expert-streaming counters for the AR reference
            # decode (hit rate + streamed bytes/token), matching the served
            # daemon's serve_stream_counters. David: hit rate + bandwidth/token.
            "serve_stream_counters": _stream_counters_block(
                run, args.decode_tokens, _resolved_plan(runtime, args)
            ),
            # W92 switch-dispatch census (present only with --stage-timing): per-layer
            # host syncs + all-hit fences deferred vs synced + gather_qmm/switch-call.
            "switch_dispatch": run.get("switch_dispatch"),
            # W125 host-side decode timeline (present only with
            # MTPLX_DSV41_DECODE_TIMELINE=1): per-token/per-layer phase mean/p50/p95
            # + host-gap accounting (barrier round-trip + reconcile await + exposed
            # miss wait) + the stamped probe overhead. Runs on the REAL compiled v2
            # path (unlike --stage-timing, which forces eager) so the ratios apply
            # to the headline tok/s.
            "decode_timeline": run.get("decode_timeline"),
            # W113 EOS surfacing (AR top level): first_token_eos / eos_index /
            # tokens_before_eos / answer_valid / eos_id / n_generated (see
            # _eos_surfacing).  A first_token_eos=True means a served path returns
            # an EMPTY answer -- the loud warning above fires too.
            **_ar_eos,
            # W113 prompt provenance stamps (prompt_source / prompt_ids_file /
            # prompt_ids_sha256 [PROMPT ids, not the generated token_ids_sha256] /
            # prompt_seed / prompt_tokens / prompt_chat_templated / allow_raw_prompt
            # / prompt_build).  Tells the standard chat-templated cell apart from the
            # raw builder that windows 39-42 measured.
            **_prompt_provenance(args, prompt_ids, prompt_meta),
        }
        if mem_profile_snaps is not None:
            from mtplx.deepseek_v41_memory_profile import (
                format_memory_profile_table,
            )

            mem_profile_cb("decode", token=_decode_generated)
            _deriv = getattr(args, "_dsv41_derivation", None)
            receipt["memory_profile"] = {
                "cache_limit_report": getattr(args, "_dsv41_cache_report", None),
                "derivation": _deriv.as_dict() if _deriv is not None else None,
                "snapshots": mem_profile_snaps,
                "table": format_memory_profile_table(mem_profile_snaps),
            }
            print("[ab] memory profile:\n" + receipt["memory_profile"]["table"],
                  flush=True)
        if getattr(args, "decode_mode", "ar") == "dspark":
            # DSpark-DIRECT lane: greedy speculative decode MUST reproduce the AR
            # ids byte-for-byte (verify is authoritative); assert it in the
            # receipt and record tokens/cycle + accept-by-depth.
            # W87 HIGH-3: cold-reset residency so the DSpark prefill is not warmed
            # by the AR reference pass (which would confound the pool A/B).
            _dspark_cold_reset = _cold_reset_expert_streaming(model)
            dsp = _generate_dspark(
                model=model, mx=mx, mem_probe=mem_probe,
                prompt_ids=prompt_ids, steps=args.decode_tokens,
                depth=args.dspark_depth,
                verify_chunks=getattr(args, "dspark_verify_chunks", None),
                stage_timing=bool(getattr(args, "stage_timing", False)),
                ar_reference=ids,
                # W113: under --stop-on-eos both lanes stop at EOS so the AR-vs-
                # DSpark byte-identity comparison stays like-for-like (default off
                # -> stop_ids None -> full fixed-step decode, numbers unchanged).
                stop_ids=({int(eos_id)} if (stop_on_eos and eos_id is not None)
                          else None),
            )
            dsp_ids = dsp["generated"]
            _dsp_eos = _eos_surfacing(dsp_ids, eos_id)
            _warn_if_first_token_eos(arm, "DSpark", _dsp_eos)
            byte_identical = dsp_ids == ids
            st = dsp["stats"]
            receipt["dspark"] = {
                "depth": int(args.dspark_depth),
                "verify_chunks": st["verify_chunks"],
                "divergence_policy": (
                    "lossless"
                    if getattr(args, "dspark_require_lossless", False)
                    else "tie_or_identical"
                    if getattr(args, "dspark_require_tie_class", False)
                    else "classify"
                ),
                # W91: the tok/s/tokens/stats/peak below come from an UNTIMED headline
                # pass (fused decode levers active); --stage-timing adds a SEPARATE
                # timed attribution pass (verify_stage_timing) that does not feed tok/s.
                "headline_pass": "untimed",
                "cold_reset_before_pass": _dspark_cold_reset,
                "byte_identical_vs_ar": byte_identical,
                # W100: decode-only wall + rate (re-prefill excluded); pass_wall_s
                # is the whole-call figure the pre-W100 decode_wall_s reported.
                "decode_wall_s": dsp["decode_wall_s"],
                "pass_wall_s": dsp.get("pass_wall_s"),
                "decode_tok_s": dsp.get("decode_tok_s"),
                "peak_gb": dsp["peak_gb"],  # MLX allocator peak only (legacy)
                "peak_process_gb": _peak_process_gb(dsp),  # process phys_footprint peak
                # W106: memory envelope for the dspark headline pass (see AR above).
                "memory": dsp.get("memory"),
                "tokens_per_cycle": st["tokens_per_cycle"],
                "accept_rate": st["accept_rate"],
                "accept_rate_by_depth": st["accept_rate_by_depth"],
                "drafted_by_depth": st["drafted_by_depth"],
                "accepted_by_depth": st["accepted_by_depth"],
                "cycles": st["cycles"],
                "verify_calls": st["verify_calls"],
                "drafted_tokens": st["drafted_tokens"],
                "accepted_drafts": st["accepted_drafts"],
                "rejected_drafts": st["rejected_drafts"],
                # the per-cycle cost table (draft/verify/accept/commit ms) + the
                # verify routing phase actually used.
                "verify_decode_phase": st["verify_decode_phase"],
                "per_cycle_ms": st["per_cycle"],
                "phase_time_s": st["phase_time_s"],
                "token_ids_sha256": hashlib.sha256(
                    json.dumps(dsp_ids).encode()
                ).hexdigest(),
                # W113 EOS surfacing (dspark block): first_token_eos / eos_index /
                # tokens_before_eos / answer_valid / eos_id / n_generated over the
                # DSpark stream (see _eos_surfacing).
                **_dsp_eos,
            }
            # W106 output persistence: the FULL DSpark stream (ids + decoded text)
            # AND the AR comparison stream it is verified against, both under the
            # dspark block so the divergence is text-auditable from the receipt.
            receipt["dspark"].update(_text_output_fields(_out_tok, dsp_ids))
            receipt["dspark"]["ar_reference"] = _text_output_fields(_out_tok, ids)
            if dsp.get("verify_stage_timing") is not None:
                # W37 internal breakdown of the verify forward (attention +
                # moe.routed_switch): the census that shows the routing phase.
                receipt["dspark"]["verify_stage_timing"] = dsp["verify_stage_timing"]
            if dsp.get("w61_engagement") is not None:
                # W61 single-barrier fast-path engagement (+ eval_indices barrier
                # count) from the route-stage probe.
                receipt["dspark"]["w61_engagement"] = dsp["w61_engagement"]
            # W115: verify-SCOPED engagement (headline pass) so THIS arm's K+1 verify
            # rows are counted -- decode_attn_kernel_engagement.calls == 0 proves an
            # eager arm turned the verify core off (draft_attn: > 0), and fused_proj_
            # engagement.rows now reflects the verify rows (NOT the AR pass at the
            # receipt top level -- the window-43 gap).
            if dsp.get("engagement"):
                receipt["dspark"].update(dsp["engagement"])
            # W81: DECODE-scoped expert-streaming counters for the DSpark decode
            # (hit rate + streamed bytes/token + slot plan), matching the served
            # daemon's DSpark serve_stream_counters so bench vs served is readable.
            _dspark_ssc = _stream_counters_block(
                dsp, args.decode_tokens, _resolved_plan(runtime, args)
            )
            if _dspark_ssc is not None:
                receipt["dspark"]["serve_stream_counters"] = _dspark_ssc
            # W77: byte-identity is the ship bar only for exact-by-construction
            # arms. A greedy argmax flip at a near-tie caused by rounding-class
            # deltas (bf16 head, M=1 vs M=K+1 matmul kernels, compiled attention)
            # is acceptable ([[dsv41-inexact-ok-if-tie-flips]]). Instead of
            # aborting, classify the FIRST divergence into the receipt and keep
            # both fully-decoded streams; --dspark-require-lossless restores the
            # hard abort for exactness-class arms.
            if not byte_identical:
                from mtplx.models.deepseek_v41_dspark_decode import (
                    classify_divergence,
                )

                first = next(
                    (i for i, (a, b) in enumerate(zip(dsp_ids, ids)) if a != b),
                    min(len(dsp_ids), len(ids)),
                )
                # W106: the decoded text ~200 chars either side of the divergence
                # point, for BOTH streams, so the flip is readable in the receipt.
                receipt["dspark"]["divergence_context"] = {
                    "ar": _divergence_context(_out_tok, ids, first),
                    "dspark": _divergence_context(_out_tok, dsp_ids, first),
                }
                ar_tok = ids[first] if first < len(ids) else None
                dsp_tok = dsp_ids[first] if first < len(dsp_ids) else None
                cap = dsp.get("divergence")
                # The verify logits row is captured for free during the dspark
                # pass; only the AR row needs a (one-time) faithful M=1 replay.
                dsp_row = cap.get("dspark_logits_row") if cap else None
                ar_row = None
                try:
                    ar_row = _ar_logits_row_at_index(
                        model=model, ops=ops, mx=mx, prompt_ids=prompt_ids,
                        ar_tokens=ids, index=first,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"[ab] WARN: AR logits replay at {first} failed: {exc!r}",
                          flush=True)
                divergence = classify_divergence(
                    index=first, ar_token=ar_tok, dspark_token=dsp_tok,
                    ar_logits_row=ar_row, dspark_logits_row=dsp_row,
                    tie_margin=float(getattr(args, "dspark_tie_margin",
                                             DSPARK_TIE_MARGIN_DEFAULT)),
                )
                # Reconcile the capture's index (from the pass) with the list
                # compare; they agree for a decode-position divergence, but record
                # the capture index AND an explicit mismatch flag so a divergence
                # whose captured verify row is NOT the flip's row is loud (the rows
                # fed to classify_divergence would then not be the compared position).
                if cap is not None:
                    cap_idx = cap.get("index")
                    divergence["capture_index"] = cap_idx
                    mismatch = cap_idx is not None and int(cap_idx) != int(first)
                    divergence["capture_index_matches_first"] = not mismatch
                    if mismatch:
                        print(
                            f"[ab] WARN: dspark capture index {cap_idx} != list-compare "
                            f"first-divergence index {first} (arm {arm!r}); the captured "
                            "verify row may not be the flip's row -- classification "
                            "rows are suspect.",
                            flush=True,
                        )
                receipt["dspark"]["divergence"] = divergence
                _print_dspark_divergence(arm, divergence)
                if getattr(args, "dspark_require_lossless", False):
                    raise AssertionError(
                        "DSpark-DIRECT greedy decode diverged from AR at index "
                        f"{first} (arm {arm!r}); speculative lane is not lossless "
                        "[--dspark-require-lossless]"
                    )
            else:
                receipt["dspark"]["divergence"] = None
                receipt["dspark"]["divergence_context"] = None
        if getattr(args, "warm_repeat", False):
            receipt["warm"] = _warm_repeat_pass(
                model=model, ops=ops, mem_probe=mem_probe,
                prompt_ids=prompt_ids, steps=args.decode_tokens,
                cold_ids=ids,
                # W113 MEDIUM-2: stop the warm pass at the same point as the cold
                # pass so denominators match and token_ids_match holds.
                stop_on_eos=stop_on_eos, eos_id=eos_id,
            )
        # W113 MEDIUM-2: the --stage-timing and --syncs passes below deliberately
        # IGNORE --stop-on-eos -- they run the full requested step count for a
        # fenced per-stage / host-sync census whose absolute tok/s is discarded
        # (not a headline rate), so an early stop would only shrink the census
        # sample.  Only the headline AR/DSpark and warm passes honour --stop-on-eos.
        if getattr(args, "stage_timing", False):
            steps = (
                int(args.stage_timing_steps)
                if args.stage_timing_steps is not None
                else int(args.decode_tokens)
            )
            # W90: the fenced stage-timing pass gets the same cooldown + a FRESH
            # sampler (a UtilizationSampler is single-use, so it cannot be the one
            # _generate consumed); its utilization/cooldown ride in the report dict
            # under __w90_utilization / __w90_cooldown.
            st_sampler = None
            if getattr(args, "utilization", False):
                st_sampler = _macmon().UtilizationSampler(
                    interval_ms=int(getattr(args, "util_interval_ms", 2000))
                )
            receipt["stage_timing"] = _stage_timing_pass(
                model=model, ops=ops, prompt_ids=prompt_ids, steps=steps,
                cooldown_s=float(getattr(args, "cooldown_s", 0.0) or 0.0),
                util_sampler=st_sampler,
            )
            if st_sampler is not None:
                print(f"[ab] {arm} stage_timing: {st_sampler.census()}", flush=True)
            _print_decode_stage_summary(
                arm, int(args.context_tokens), receipt["stage_timing"]
            )
        if getattr(args, "prefill_stage_timing", False):
            receipt["prefill_stage_timing"] = _prefill_stage_timing_pass(
                model=model, ops=ops, prompt_ids=prompt_ids,
            )
        if args.syncs > 0:
            receipt["sync_census"] = _sync_census(
                model=model, ops=ops, mem_probe=mem_probe,
                prompt_ids=prompt_ids, steps=args.syncs,
            )
        if _dsv41 is not None:
            eng = _dsv41._sinkhorn_kernel_calls()
            receipt["sinkhorn_engagement"] = {
                "kernel_calls": eng["kernel"],
                "recurrence_calls": eng["recurrence"],
                # >0 kernel with 0 recurrence == the Metal Sinkhorn actually ran;
                # 0 kernel with >0 recurrence == it fell back to the recurrence.
                "engaged": eng["kernel"] > 0 and eng["recurrence"] == 0,
                "sinkhorn_metal_env": os.environ.get(SINKHORN_METAL_ENV),
                "hc_compile_env": os.environ.get(HC_COMPILE_ENV),
                "note": "cumulative over this arm (prefill + decode + census)",
            }
            # W91/K35 engagement: fused vs eager DecoderLayer forwards, and the
            # fused-premix-kernel calls (GPU only).  ``engaged`` = the fused path
            # actually ran (>0 fused forwards); distinguishes "K35 ran" from "armed
            # but forced eager" (the --stage-timing recording guard, or the flag
            # never reaching the child).  Under mx.compile the premix-kernel wrapper
            # runs only at (cold) trace, so ``premix_kernel_calls`` reads ~2xshapes,
            # not once/token -- read cumulatively, like sinkhorn_engagement.
            ss = _dsv41._small_stages_calls()
            pk = _dsv41._hc_premix_kernel_calls()
            receipt["small_stages_engagement"] = {
                "fused_layer_forwards": ss["fused"],
                "eager_layer_forwards": ss["eager"],
                "engaged": ss["fused"] > 0,
                "premix_kernel_calls": pk["kernel"],
                "premix_reference_calls": pk["reference"],
                "small_stages_fused_env": os.environ.get(SMALL_STAGES_FUSED_ENV),
                "hc_premix_kernel_env": os.environ.get("MTPLX_DSV41_HC_PREMIX_KERNEL"),
                "note": "cumulative over this arm (prefill + decode + census); the "
                        "HEADLINE tok/s pass is UNTIMED so K35 is active there -- the "
                        "timed stage-timing pass forces eager (recording guard)",
            }
        if _dsv41_cache is not None:
            # W73/K32 chunk-grow engagement (cumulative over the arm). ``enabled``
            # false => the flag never reached cache construction (env timing / wrong
            # class). ``enabled`` true with a flat ``rows_copied`` while the
            # ``cache_append`` census stage stays O(T) => slice_update did not donate
            # in-place on Metal (append still O(cap)); the append is ~3% of decode
            # regardless (see W73_DECODE_16K_AUDIT.md).
            stats = _dsv41_cache.kv_chunk_grow_stats()
            stats["env"] = os.environ.get(KV_CHUNK_GROW_ENV)
            stats["note"] = (
                "cumulative over this arm; rows_copied is the STRATEGY's logical "
                "cost, compare vs the measured cache_append census stage"
            )
            receipt["kv_chunk_grow"] = stats
            # W80 / K34 window-ring engagement + drop/copy telemetry.  enabled ==
            # False means the flag did not reach cache construction (env timing /
            # wrong class).  enabled == True: `capacity` is the steady logical keep
            # (window_size + max_verify + slack), `phys_capacity` the ping-pong
            # buffer rows, `drops`/`rows_dropped` the compactions/logical rows
            # dropped, `rows_copied` the physical rows written (appends + compaction
            # carries) -- flat-per-token (amortized O(1)) is the win, vs the phase-1
            # O(T) concatenate.  `reallocs` should be ~0 in decode (a prefill chunk
            # wider than the ping-pong buffers is the only grow).
            ring_stats_fn = getattr(_dsv41_cache, "window_ring_stats", None)
            if callable(ring_stats_fn):
                rstats = ring_stats_fn()
                rstats["env"] = os.environ.get(WINDOW_RING_ENV)
                rstats["maxkv_env"] = os.environ.get(WINDOW_RING_MAXKV_ENV)
                rstats["note"] = (
                    "cumulative over this arm; capacity = window_size + max_verify + "
                    "slack (bounded); rows_copied flat-per-token == amortized O(1)"
                )
                receipt["window_ring"] = rstats
            # W107 per-lane bounded-KV engagement (cumulative over the arm).
            # ``enabled`` false => MTPLX_DSV41_KV_BOUNDED never reached cache
            # construction.  enabled true: for each lane ``kv_realloc_<lane>`` should
            # be its one-time prealloc count (window 1, compress 1, index 1, latent 2
            # == kv+score) and STAY there -- a growing ``kv_realloc_*`` over the cell
            # means the lane was not preallocated (max_kv unset / prefill chunk wider
            # than the cap).  ``kv_appends_<lane>`` is the O(new-rows) decode
            # path; ``alloc_bytes`` should ~= kv_bytes_at_max_kv(config, max_kv).
            bounded_stats_fn = getattr(_dsv41_cache, "kv_bounded_stats", None)
            if callable(bounded_stats_fn):
                bstats = bounded_stats_fn()
                bstats["env"] = os.environ.get(KV_BOUNDED_ENV)
                bstats["maxkv_env"] = os.environ.get(KV_BOUNDED_MAXKV_ENV)
                # W107 round-4 DONATION GATE: prove the bounded lanes donate on the real
                # path (mx.slice_update pointer-stable in the rebind pattern).
                # sample_ptr_flips() is stamped by the fenced stage-timing pass; the gate
                # passes iff ptr_flips_window == window_ring.drops, compress/index/latent
                # flips == 0, and kv_realloc_<lane> == one prealloc/layer.
                try:
                    _cfg2 = getattr(model, "args", None)
                    _rs = (_dsv41_cache.window_ring_stats()
                           if hasattr(_dsv41_cache, "window_ring_stats") else {})
                    if _cfg2 is not None and hasattr(_dsv41_cache, "kv_donation_gate"):
                        _exp = _dsv41_cache.expected_bounded_reallocs(_cfg2)
                        bstats["donation_gate"] = _dsv41_cache.kv_donation_gate(
                            bstats, _rs, expected_reallocs=_exp)
                except Exception:  # pragma: no cover - defensive
                    pass
                # W107 (review MEDIUM-A): receipt gate -- compare the memory-plan
                # formula W106 will use against the bytes actually allocated, so a
                # dtype-model drift is caught at runtime.  Exact when the window did
                # not transiently grow during (chunked) prefill, i.e. kv_realloc_window
                # == num_layers (one ring init per layer, no compaction realloc); a
                # chunked prefill grows-then-shrinks the window (MEDIUM-1), so the
                # cumulative alloc_bytes then exceeds the steady formula (expected).
                _bmaxkv = os.environ.get(KV_BOUNDED_MAXKV_ENV)
                _cfg = getattr(model, "args", None)
                bytes_fn = getattr(_dsv41_cache, "kv_bytes_at_max_kv", None)
                if _cfg is not None and _bmaxkv and callable(bytes_fn):
                    try:
                        _formula = int(bytes_fn(_cfg, int(_bmaxkv)))
                        bstats["kv_bytes_formula"] = _formula
                        bstats["formula_matches_alloc"] = bool(
                            int(bstats.get("alloc_bytes", 0)) == _formula
                        )
                    except Exception:  # pragma: no cover - defensive
                        bstats["kv_bytes_formula"] = None
                        bstats["formula_matches_alloc"] = None
                bstats["note"] = (
                    "cumulative over this arm; kv_realloc_<lane> == one-time prealloc "
                    "(>1 growing == not preallocated-bounded); kv_appends_<lane> "
                    "== O(new-rows) decode path; formula_matches_alloc exact iff "
                    "kv_realloc_window == num_layers (no transient prefill grow)"
                )
                receipt["kv_bounded"] = bstats
        # Preserve the armed baseline and its explicitly labelled process estimate,
        # while reporting sampled whole-machine use from vm_stat as box_used_gb.
        # MEDIUM-6: lift the apply_mlx_memory_cap report (applied limit, wired/cache
        # applied, slot_derivation, target components) into the receipt, and take
        # mlx_limit_gib_effective from ITS applied limit -- mlx 0.32.2 has no
        # get_memory_limit readback, so the readback-derived value was None.
        _cap = _memory_cap_block(runtime)
        if _cap is not None:
            receipt["memory_cap"] = _cap
        if isinstance(receipt.get("memory"), dict):
            _inject_box_used(receipt["memory"], args)
            _apply_effective_limit(receipt["memory"], _cap)
        _dsp = receipt.get("dspark")
        if isinstance(_dsp, dict) and isinstance(_dsp.get("memory"), dict):
            _inject_box_used(_dsp["memory"], args)
            _apply_effective_limit(_dsp["memory"], _cap)
            if _cap is not None:
                _dsp["memory_cap"] = _cap
        return receipt
    finally:
        if runtime is not None:
            close = getattr(runtime, "close", None)
            if callable(close):
                close()


def _tokenizer(args, bench):
    from mlx_lm.utils import load_tokenizer

    return load_tokenizer(Path(args.model))


def _warm_repeat_pass(*, model, ops, mem_probe, prompt_ids, steps, cold_ids,
                      stop_on_eos=False, eos_id=None) -> dict:
    """Second prefill+decode of the SAME prompt in the same process.

    A fresh ``model.make_cache()`` resets the KV window and hands a fresh engram
    history clone, but the expert-bank LRU and any engram row cache stay warm from
    the cold pass, so this pass bounds the no-miss decode ceiling.  Greedy decode
    is deterministic, so the warm token ids must match the cold pass -- recorded
    (``token_ids_match`` + both sha256), never asserted, so a mismatch is reported
    instead of crashing the arm.

    W113 MEDIUM-2: ``stop_on_eos`` / ``eos_id`` are threaded through so the warm
    pass stops at the SAME point as the (also stopped) cold pass -- otherwise the
    warm pass runs the full ``steps`` while the cold pass stopped early, mixing
    denominators (warm_decode_tok_s over ``steps`` vs the cold rate over the tokens
    it generated) and breaking ``token_ids_match``.  ``warm_decode_tok_s`` is over
    the tokens ACTUALLY generated (``decode_steps_run``)."""
    run = _generate(
        model=model, ops=ops, mem_probe=mem_probe,
        prompt_ids=prompt_ids, steps=steps,
        stop_on_eos=stop_on_eos, eos_id=eos_id,
    )
    warm_ids = run["generated"]
    warm_generated = int(run.get("decode_steps_run", steps))
    cold = [int(t) for t in cold_ids]
    warm_sha = hashlib.sha256(json.dumps(warm_ids).encode()).hexdigest()
    cold_sha = hashlib.sha256(json.dumps(cold).encode()).hexdigest()
    return {
        "warm_ttft_s": run["ttft_s"],
        "warm_prefill_tok_s": (len(prompt_ids) / run["ttft_s"])
        if run["ttft_s"] > 0
        else None,
        "warm_decode_wall_s": run["decode_wall_s"],
        # W113: over the tokens actually generated (== steps unless --stop-on-eos).
        "warm_decode_tokens_generated": warm_generated,
        "warm_decode_tok_s": (warm_generated / run["decode_wall_s"])
        if (run["decode_wall_s"] > 0 and warm_generated > 0)
        else None,
        "warm_peak_gb": run["peak_gb"],
        "warm_token_ids_sha256": warm_sha,
        "cold_token_ids_sha256": cold_sha,
        "token_ids_match": warm_ids == cold,
    }


#: CSA mode -> the human class label used in the W73 decode summary.
_CSA_MODE_CLASS = {
    "swa_only": "SWA-only",
    "full": "Full",
    "reindex": "Reindex",
    "reuse": "Reuse",
}


def _print_decode_stage_summary(arm: str, context_tokens: int, report: dict) -> None:
    """W73: print the decode stage census so the GPU window can read the 16K
    attribution at a glance -- attention per CSA mode class (SWA-only / Full /
    Reindex / Reuse) plus the KV-append / compressed-append / indexer-select
    sub-stages peeled out of the single ``attn.<mode>`` bracket
    (``decode_breakdown``, kept out of the flat sum, so it decomposes the flat
    ``attn.<mode>`` totals rather than adding to them).  The fences inflate absolute
    time, so the RATIOS between rows are the signal, not the totals."""
    if not report or not report.get("enabled") or report.get("kind") != "decode":
        return
    stages = report.get("stages", {})
    breakdown = report.get("decode_breakdown", {})

    def _mm(d, key):
        v = (d.get(key) or {}).get("mean_ms_per_token")
        return v if v is not None else 0.0

    print(
        f"[ab] --- W73 decode stage census: arm={arm} ctx={context_tokens} "
        f"tokens={report.get('tokens')} (fenced; ratios are the signal) ---"
    )
    print(
        f"[ab]   frame_wall={report.get('frame_wall_ms_per_token', 0):.3f} ms/tok  "
        f"stage_sum={report.get('stage_sum_ms_per_token', 0):.3f} ms/tok"
    )
    # attention per CSA mode class (flat attn.<mode> totals) + its append/select
    # decomposition (from decode_breakdown).
    for mode, label in _CSA_MODE_CLASS.items():
        attn_key = f"attn.{mode}"
        if attn_key not in stages and not any(
            k.startswith(attn_key + ".") for k in breakdown
        ):
            continue
        attn_ms = _mm(stages, attn_key)
        cnt = (stages.get(attn_key) or {}).get("count", 0)
        cache_ms = _mm(breakdown, f"{attn_key}.cache_append")
        comp_ms = _mm(breakdown, f"{attn_key}.compress_append")
        sel_ms = _mm(breakdown, f"{attn_key}.select")
        print(
            f"[ab]   {label:9s} attn={attn_ms:8.3f} ms/tok (layers/tok={cnt:3d})  "
            f"| KV-append={cache_ms:7.3f}  compress-append={comp_ms:7.3f}  "
            f"indexer-select={sel_ms:7.3f}  ms/tok"
        )
    # totals across modes for the append lanes (the W73 O(T) suspects).
    tot_cache = sum(
        _mm(breakdown, k) for k in breakdown if k.endswith(".cache_append")
    )
    tot_comp = sum(
        _mm(breakdown, k) for k in breakdown if k.endswith(".compress_append")
    )
    tot_sel = sum(_mm(breakdown, k) for k in breakdown if k.endswith(".select"))
    print(
        f"[ab]   TOTAL/tok  KV-append={tot_cache:7.3f}  "
        f"compress-append={tot_comp:7.3f}  indexer-select={tot_sel:7.3f}  ms/tok "
        f"(append+select ~= {tot_cache + tot_comp + tot_sel:.1f} ms/tok; the O(T) "
        f"port artifact -- NOT the decode headline, cf. routed_switch/attn.reuse)"
    )
    # W73/K32 chunk-grow engagement (proves the flag reached cache construction).
    try:
        from mtplx.models import deepseek_v41_cache as _c
        s = _c.kv_chunk_grow_stats()
        print(
            f"[ab]   kv_chunk_grow: enabled={s['enabled']} "
            f"layers_chunk_grown={s['layers_chunk_grown']} layers_plain={s['layers_plain']} "
            f"buffers={s['buffers']} appends={s['appends']} logical_rows_copied={s['rows_copied']} "
            f"(flat rows_copied but O(T) cache_append above == slice_update not donating on Metal)"
        )
    except Exception:  # pragma: no cover - defensive
        pass


def _stage_timing_pass(*, model, ops, prompt_ids, steps, cooldown_s=0.0,
                       util_sampler=None) -> dict:
    """W37 fenced decode pass -> ``model.stage_timing_report()``.

    Prefill once (probe unarmed -> untouched), then arm the stage-timing session
    around the decode loop only, wrapping each step in a ``frame`` and the
    argmax/host round-trip in a ``sample`` stage.  The route-stage probe counters
    (when ``MTPLX_ROUTE_STAGE_PROBE`` is armed) are cleared before the loop so the
    merged ``route_stage`` census reflects this window, not the prefill + prior
    passes.  Fences inflate absolute time, so nothing here feeds the reported
    tok/s.

    W90: ``cooldown_s`` idles after the prefill and before the decode loop;
    ``util_sampler`` (a ``util_macmon.UtilizationSampler`` or ``None``) samples
    macmon over the decode loop only.  A ``__w90_cooldown`` / ``__w90_utilization``
    key is stashed on the returned report dict for the receipt."""
    from mtplx.models import deepseek_v41_stage_timing as stime

    cache = model.make_cache()
    logits = model(ops.input([list(prompt_ids)]), cache=cache, logits_keep=1)
    ops.sync(logits)
    token = ops.argmax_last(logits)
    # W90: idle after prefill, before the fenced decode loop.
    cooldown_block = None
    if cooldown_s and float(cooldown_s) > 0:
        cooldown_block = _macmon().cooldown(float(cooldown_s), label="w78-in-model")
    # Reset the route probe window (best-effort; its snapshot is cumulative).
    try:
        from mtplx import expert_route_probe as route_probe

        if getattr(route_probe, "ENABLED", False):
            route_probe._SUMS.clear()
            route_probe._COUNTS.clear()
    except Exception:
        pass
    # W107 round-4 donation gate: init the pointer baseline after prefill (this pass is
    # already fenced, so reading a pointer per token does not perturb the headline tok/s).
    _ptr_sample = getattr(cache, "sample_ptr_flips", None)
    if callable(_ptr_sample):
        _ptr_sample()
    _util_cm = util_sampler if util_sampler is not None else contextlib.nullcontext()
    stime.begin()
    with _util_cm:  # W90: macmon utilization over the fenced decode loop only
        for _ in range(int(steps)):
            with stime.frame():
                logits = model(ops.input([[token]]), cache=cache)
                with stime.stage("sample"):
                    token = ops.argmax_last(logits)
            if callable(_ptr_sample):
                _ptr_sample()  # count per-lane buffer-pointer flips (donation gate)
    report = model.stage_timing_report()
    stime.end()
    report = report if report is not None else {"enabled": False}
    if cooldown_block is not None:
        report["__w90_cooldown"] = cooldown_block
    if util_sampler is not None:
        report["__w90_utilization"] = util_sampler.summarize()
    return report


def _prefill_stage_timing_pass(*, model, ops, prompt_ids) -> dict:
    """W47 fenced PREFILL pass -> ``model.stage_timing_report()`` (kind=prefill).

    One prefill forward of the whole prompt under a prefill session.  The backbone
    picks the schedule from ``MTPLX_DSV41_PREFILL_LAYER_MAJOR`` and the chunk from
    ``MTPLX_DSV41_PREFILL_CHUNK`` (the arm's env), and records per (chunk, layer
    type): attention split (qkv_proj / select / score / cache_append /
    compress_append), HC, gate+top-k, streamed-switch breakdown, shared expert,
    combine, engram -- plus per-chunk walls.  The route-stage probe window is
    cleared first so the merged ``route_stage`` reflects this forward.  Fences
    inflate absolute time; the reported tok/s pass never runs the probe."""
    from mtplx.models import deepseek_v41_stage_timing as stime

    cache = model.make_cache()
    try:
        from mtplx import expert_route_probe as route_probe

        if getattr(route_probe, "ENABLED", False):
            route_probe._SUMS.clear()
            route_probe._COUNTS.clear()
    except Exception:
        pass
    stime.begin(kind="prefill")
    logits = model(ops.input([list(prompt_ids)]), cache=cache, logits_keep=1)
    ops.sync(logits)
    report = model.stage_timing_report()
    stime.end()
    return report if report is not None else {"enabled": False}


def _sync_census(*, model, ops, mem_probe, prompt_ids, steps) -> dict:
    """A short probe pass (route stage probe must be ENABLED via PROBE_ENV set
    before mtplx import) that counts the routing barrier per decoded token."""
    from mtplx import expert_route_probe as probe

    if not getattr(probe, "ENABLED", False):
        return {"enabled": False, "note": f"set {PROBE_ENV}=1 before launch"}
    # prefill first (routes every layer once), then snapshot and decode.
    cache = model.make_cache()
    logits = model(ops.input([list(prompt_ids)]), cache=cache, logits_keep=1)
    ops.sync(logits)
    token = ops.argmax_last(logits)
    before = int(probe._COUNTS.get(BARRIER_STAGE, 0))
    # W38/K3: also delta the Sinkhorn route stages so the census shows whether the
    # Metal kernel or the recurrence carried this decode (eager path only -- under
    # a warm HC-compile tape the Python wrapper does not re-run, so this delta is 0
    # and the arm-level ``sinkhorn_engagement`` counter is the authoritative read).
    sk_before = int(probe._COUNTS.get("hc.sinkhorn_kernel", 0))
    rec_before = int(probe._COUNTS.get("hc.sinkhorn_recurrence", 0))
    for _ in range(int(steps)):
        logits = model(ops.input([[token]]), cache=cache)
        ops.sync(logits)
        token = ops.argmax_last(logits)
    after = int(probe._COUNTS.get(BARRIER_STAGE, 0))
    barriers = after - before
    sk = int(probe._COUNTS.get("hc.sinkhorn_kernel", 0)) - sk_before
    rec = int(probe._COUNTS.get("hc.sinkhorn_recurrence", 0)) - rec_before
    return {
        "enabled": True,
        "decode_steps": int(steps),
        "routing_barriers_total": barriers,
        "routing_barriers_per_token": (barriers / steps) if steps else None,
        "sinkhorn_kernel_calls_decode": sk,
        "sinkhorn_recurrence_calls_decode": rec,
        "stages": probe.snapshot().get("stages", {}),
    }


def _run_dry(args, bench) -> int:
    """CPU-only path: run every requested arm through the dry-run double.

    No MLX/Metal import, no model load; writes the same append-only JSONL
    receipt (with ``prompt_build`` + ``arm_env`` per arm) so the harness'
    argument resolution, prompt build and per-arm env application are all
    exercised offline.
    """
    for arm in args.arms:
        print(
            f"[ab] DRY-RUN arm={arm} ctx={args.context_tokens} "
            f"decode={args.decode_tokens}"
        )
        receipt = _run_arm(args, arm, bench, mx=None)
        with args.out.open("a") as fh:
            fh.write(json.dumps(receipt) + "\n")
        print(
            f"[ab]   prompt_tokens={receipt['prompt_tokens']} "
            f"arm_env={receipt['arm_env']}"
        )
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.dspark_depth) < 0:
        print("ab_decode_env_levers: --dspark-depth must be >= 0", file=sys.stderr)
        return 2
    verify_chunks = getattr(args, "dspark_verify_chunks", None)
    if verify_chunks is not None and sum(verify_chunks) != int(args.dspark_depth) + 1:
        print(
            "ab_decode_env_levers: --dspark-verify-chunks must sum to "
            f"--dspark-depth + 1 ({int(args.dspark_depth) + 1})",
            file=sys.stderr,
        )
        return 2
    if args.dspark_require_tie_class and args.decode_mode != "dspark":
        print(
            "ab_decode_env_levers: --dspark-require-tie-class requires "
            "--decode-mode dspark",
            file=sys.stderr,
        )
        return 2
    bench = _load_bench_module()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        return _run_dry(args, bench)

    # W113 cell-prompt guard: refuse to MEASURE a 16K cell (any cell16k_* arm or
    # --context-tokens 16384) on the raw builder, and default --prompt-ids-file to
    # the standard chat-templated cell so launchers get it.  Real path only -- the
    # --dry-run resolution double never measures throughput (it exercises the raw
    # builder with the fake tokenizer on purpose), so it is exempt.  May raise
    # SystemExit (a clear error naming the standard file) or set
    # args.prompt_ids_file before any arm/model load.
    _apply_cell_prompt_guard(args)

    if args.syncs > 0 or args.stage_timing or args.prefill_stage_timing:
        # The route-stage probe reads its ENABLED flag at import, so arm it before
        # any mtplx import happens inside the arm run.  --stage-timing /
        # --prefill-stage-timing arm it too, so model.stage_timing_report() can
        # merge the route-stage census (hot.eval_indices barrier count) under
        # ``route_stage`` alongside the fenced DSV4.1 stages.
        os.environ[PROBE_ENV] = "1"
    if args.stage_timing or args.prefill_stage_timing:
        # Advisory marker in the receipt env snapshot; the fenced session is armed
        # in-process by deepseek_v41_stage_timing.begin(), not by this env.
        os.environ[STAGE_TIMING_ENV] = "1"
    import mlx.core as mx

    mx.random.seed(int(args.seed))

    receipts = []
    for arm in args.arms:
        print(f"[ab] arm={arm} ctx={args.context_tokens} decode={args.decode_tokens}")
        try:
            receipt = _run_arm(args, arm, bench, mx)
        except (RuntimeError, ValueError) as exc:
            # W106 MEDIUM-C: record the failure in the ledger (an abort row on
            # args.out) and exit with a distinct code (4), so a budget/re-measure
            # abort or a floor refusal is not a silent gap in the receipts.
            row = _abort_receipt_row(arm, exc, args)
            _append_receipt_row(args.out, row)
            print(
                f"[ab]   ABORTED arm={arm} stage={row['stage']} "
                f"reason={row['reason']}",
                flush=True,
            )
            return 4
        receipts.append(receipt)
        with args.out.open("a") as fh:
            fh.write(json.dumps(receipt) + "\n")
        # W106 output persistence: the FULL decoded output as a text sidecar beside
        # the receipt (never overwriting an existing one), so David can audit it.
        _write_output_sidecars(args.out, receipt)
        print(
            f"[ab]   decode_tok_s={receipt['decode_tok_s']} "
            f"{_memory_headline(receipt)} "
            f"sha={receipt['token_ids_sha256'][:12]}"
        )

    # Control-vs-overlap summary: byte-identity is a recorded fact, not a claim.
    parity_failed = False
    if getattr(args, "dspark_require_tie_class", False):
        for receipt in receipts:
            if not _dspark_tie_class_gate_passes(receipt.get("dspark")):
                parity_failed = True
                print(
                    f"[ab] FAIL: {receipt['arm']} DSpark divergence is not an "
                    "index-matched tie_flip (--dspark-require-tie-class)",
                    flush=True,
                )
    if len(receipts) >= 2:
        base = receipts[0]
        # Compare actual engine budgets in bytes. Schema-2 memory samples no
        # longer carry the old GiB plan fields; absent metadata is unknown, not
        # evidence that all arms used the same budget. Intentional budget/cache
        # tradeoffs remain reportable, with their differing budgets explicit.
        _plan_limits = [_receipt_plan_limit_bytes(r) for r in receipts]
        _known_plan_limits = [value for value in _plan_limits if value is not None]
        if len(set(_known_plan_limits)) > 1:
            print(
                "[ab] WARN: arms ran DIFFERENT plan_limit_bytes values "
                f"({_plan_limits}); throughput also reflects this budget change. "
                "Pin the plan with --memory-plan-from for an equal-budget comparison.",
                flush=True,
            )
        if len(_known_plan_limits) != len(_plan_limits):
            print(
                "[ab] WARN: plan_limit_bytes unknown for one or more arms "
                f"({_plan_limits}); equal budgets cannot be established.", flush=True,
            )
        elif len(set(_known_plan_limits)) == 1:
            print(f"[ab] plan reproducibility: all arms ran plan_limit_bytes={_plan_limits[0]}",
                  flush=True)
        for cand in receipts[1:]:
            identical = cand["token_ids_sha256"] == base["token_ids_sha256"]
            d_base = base["decode_tok_s"] or 0.0
            d_cand = cand["decode_tok_s"] or 0.0
            delta = ((d_cand - d_base) / d_base * 100.0) if d_base else None
            print(
                f"[ab] {cand['arm']} vs {base['arm']}: "
                f"byte_identical={identical} "
                f"decode_tok_s {d_base:.3f} -> {d_cand:.3f} "
                f"({'+' if (delta or 0) >= 0 else ''}{delta:.2f}%)"
                if delta is not None
                else f"[ab] {cand['arm']} vs {base['arm']}: byte_identical={identical}"
            )
            if not identical:
                # An unchanged rounding lever cannot explain a difference caused
                # by another change. Use the pair's effective env, not the
                # candidate's broad per-arm metadata. Bounded KV is unvalidated.
                keys = _pairwise_rounding_class_keys(base, cand)
                if keys:
                    keys_str = ", ".join(keys)
                    # A receipt's DSpark divergence compares that arm's AR and
                    # speculative streams, not this base/candidate pair. Preserve
                    # it in the receipt without misattributing its logit margins.
                    print(
                        f"[ab] {cand['arm']}: token-id sha differs -- expected "
                        f"(rounding-class: {keys_str})"
                    )
                else:
                    parity_failed = True
                    print(
                        f"[ab] FAIL: {cand['arm']} changed the decoded tokens "
                        "(no validated changed rounding lever explains this pair)"
                    )
    return 1 if parity_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
