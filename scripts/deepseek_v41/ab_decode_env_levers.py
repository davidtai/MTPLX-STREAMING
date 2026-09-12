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

# W106 item 3: derive the MLX plan limit from David's TOTAL box budget while
# COMPENSATING for the non-Metal requirements (the box has a 110 GB hard ceiling
# and his budget is 100 GB TOTAL for everything):
#
#   plan_limit_gib = total - system_used_at_start - non_metal_overhead
#                          - kv_growth_to_max_kv - safety
#
# Unlike the W62 --box-budget-gib path (fixed profile constants), this MEASURES
# system_used_at_start (vm_stat, the gpu_window.sh formula) and the non-Metal
# process overhead (process RSS - mx active) and prices the KV growth to --max-kv,
# so the plan compensates for everything the MLX allocator peak does NOT see.
# Every term lands in the receipt ``memory`` block.  See docs/deepseek-v41/
# W106_WINDOW_MEMORY_ACCOUNTING.md for the once-only definition of each term.
#
# Conservative pre-load estimate of the non-Metal process overhead (Python heap +
# positional-expert bank read buffers + engram host-side row LRU + tokenizer).
# The plan limit must be fixed BEFORE the model loads (the loader takes it), so
# the derivation uses this estimate first, then re-measures the real overhead
# after load (process RSS - mx active) and lowers the MLX active limit if the
# measurement exceeds it (two-phase).  Mirrors the W62 profile constant
# HOST_OVERHEAD_GIB (mtplx.deepseek_v41_memory_profile.HOST_OVERHEAD_GIB = 10).
DEFAULT_NON_METAL_OVERHEAD_GIB = 10.0
# Safety headroom subtracted from the budget (flag --memory-safety-gb).
DEFAULT_MEMORY_SAFETY_GIB = 3.0
# Floor below which a derived plan limit is refused (flag --memory-budget-floor-gib).
DEFAULT_MEMORY_BUDGET_FLOOR_GIB = 20.0
# If the post-load re-measured overhead exceeds the pre-load estimate by more than
# this, the two-phase step ABORTS before decode (HIGH-1: the MLX limit is never
# lowered post-load).
_BUDGET_REMEASURE_TOLERANCE_GIB = 0.5

# W106 HIGH-2: RSS-vs-mlx_peak semantics on Metal are UNVERIFIED until one real GPU
# window produces a receipt whose gpu_window.sh tree RSS can be compared against
# this process's mlx_peak. Recorded on every memory block so a reader does not
# treat process_peak_rss_gb and mlx_peak_gb as interchangeable.
_RSS_SEMANTICS_NOTE = (
    "UNVERIFIED on Metal: process_peak_rss_gb (phys_footprint) vs mlx_peak_gb "
    "(allocator peak) have not been cross-checked against a real gpu_window.sh "
    "tree-RSS receipt; unified memory may double-count. Compare a real-window "
    "receipt before treating either as the box figure."
)
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
# over the cell).  Byte-identical to the ring arm's CLASS by construction (pure
# prealloc/in-place; the ring's drop_offset already proved the window byte-identity).
# Default ON for the cell16k_ring* arms; MTPLX_DSV41_KV_BOUNDED=0 disables.  MAXKV is
# stamped from the resolved cell max_kv in _run_arm (falls back to WINDOW_RING_MAXKV).
KV_BOUNDED_ENV = "MTPLX_DSV41_KV_BOUNDED"
KV_BOUNDED_MAXKV_ENV = "MTPLX_DSV41_KV_BOUNDED_MAXKV"
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

# W115: verify-attention fast path -- auto-arm the fused decode SDPA core (K29 kernel
# on GPU, W97 core-compile tape on CPU/GPU) + the W101 fused projections for the K+1
# DSpark verify batch (1 < rows <= VERIFY_ATTN_MAX_ROWS, default 8), independent of
# the M=1 decode-core levers, so the batched verify stops running the eager per-row
# core at ~M x the M=1 cost.  ROUNDING-CLASS (the fused/compiled core reassociates the
# fp32 softmax vs eager -- greedy-identical), phase-scoped to the decode_verify forward
# (docs/deepseek-v41/W115_VERIFY_ATTN_FASTPATH.md).
VERIFY_ATTN_FASTPATH_ENV = "MTPLX_DSV41_VERIFY_ATTN_FASTPATH"
VERIFY_ATTN_MAX_ROWS_ENV = "MTPLX_DSV41_VERIFY_ATTN_MAX_ROWS"

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
    # W115 (appended; coordinate with any concurrent list extension): verify-attention
    # fast path lever + row cap.
    VERIFY_ATTN_FASTPATH_ENV,
    VERIFY_ATTN_MAX_ROWS_ENV,
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
    attn_lean_casts=None, attn_fused_proj=None,
    verify_attn_fastpath=None, verify_attn_max_rows=None,
    verify_record_hashes=None,
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
        VERIFY_ATTN_FASTPATH_ENV: verify_attn_fastpath,
        VERIFY_ATTN_MAX_ROWS_ENV: verify_attn_max_rows,
        VERIFY_RECORD_HASHES_ENV: verify_record_hashes,
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
    "cell16k_ring": _preset(
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
        window_ring="1", layout_fix="1", kv_bounded="1",
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
        window_ring="1", layout_fix="1", kv_bounded="1",
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
        window_ring="1", layout_fix="1", kv_bounded="1",
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
        window_ring="1", layout_fix="1", kv_bounded="1",
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
        window_ring="1", layout_fix="1", kv_bounded="1",
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
        window_ring="1", layout_fix="1", kv_bounded="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        attn_core_compile="1",
    ),
    # W97: cell16k_ring + the wo_a-dequant cache (exact) + the core compile (rounding-
    # class) stacked -- the full W97 attention-dispatch program.  ROUNDING-CLASS via
    # the core compile; the wo_a cache is byte-identical on its own.  Watch peak
    # memory (the wo_a cache holds a dense wo_a copy resident per layer, ~5.4/2.7 GB).
    "cell16k_ring_wo_a_core": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1", kv_bounded="1",
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
        window_ring="1", layout_fix="1", kv_bounded="1",
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
        window_ring="1", layout_fix="1", kv_bounded="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        wo_a_cache="1", attn_lean_casts="1",
    ),
    # W99: cell16k_ring_lean + the K29 fused decode core (1-dispatch score+softmax+PV).
    # ROUNDING-CLASS via K29 (flagged in the byte-identity summary); the lowest-dispatch
    # attention arm (exact wo_a cache + lean casts + the fused core).
    "cell16k_ring_lean_k29": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1", kv_bounded="1",
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
        window_ring="1", layout_fix="1", kv_bounded="1",
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
        window_ring="1", layout_fix="1", kv_bounded="1",
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
        window_ring="1", layout_fix="1", kv_bounded="1",
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
        window_ring="1", layout_fix="1", kv_bounded="1",
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
        window_ring="1", layout_fix="1", kv_bounded="1",
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
    # stack + fused proj) without the fused core.  Watch peak memory (wo_a cache + the
    # fused path each hold a per-layer wo_a copy resident at the 16K cell).
    "cell16k_ring_v2_attn": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1", kv_bounded="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2",
        wo_a_cache="1", attn_lean_casts="1", attn_fused_proj="1",
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
        window_ring="1", layout_fix="1", kv_bounded="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2", draft="1", draft_head_bf16="1",
        wo_a_cache="1", attn_lean_casts="1", attn_fused_proj="1",
    ),
    # W115: cell16k_ring_v2_draft_attn + the verify-attention fast path lever. EXACT
    # KEY SET = cell16k_ring_v2_draft_attn's fifteen keys PLUS verify_attn_fastpath="1".
    # The base arm deliberately keeps the eager SDPA CORE (K29 + core-compile OFF), so
    # its K+1 verify pays ~M x the M=1 core cost; this arm auto-arms the fused decode
    # core + fused projections for the verify rows (1 < rows <= 8) ONLY, phase-scoped
    # to decode_verify.  The direct A/B vs cell16k_ring_v2_draft_attn isolates the
    # batched-verify core: eval_indices sums down, verify_ms down, and fused_proj_
    # engagement (dspark block) counts the verify rows.  ROUNDING-CLASS (fused/compiled
    # core reassociates the fp32 softmax vs eager -- greedy verify stays authoritative,
    # so byte-identical-to-AR is NOT the bar; the dspark divergence classifier gates it).
    "cell16k_ring_v2_draft_attn_vfast": _preset(
        layer_major="1", prefill_dense="1", score_path="lean", selected_keys="1",
        window_ring="1", layout_fix="1", kv_bounded="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2", draft="1", draft_head_bf16="1",
        wo_a_cache="1", attn_lean_casts="1", attn_fused_proj="1",
        verify_attn_fastpath="1",
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
        window_ring="1", layout_fix="1", kv_bounded="1",
        head="bf16", sinkhorn="1", attn="1", win_memo="1",
        runner="v2", verify_record_hashes="1",
    ),
}

# W97 (review item 7): the rounding-class env keys, documented in ONE place with the
# reason.  An arm whose preset arms ANY of these keys has decoded tokens that are
# EXPECTED to differ from control by rounding (the lever reassociates the fp32
# attention core / softmax, so a greedy near-tie can flip -- [[dsv41-inexact-ok-if-
# tie-flips]]).  A token-id sha mismatch on such an arm is "expected (rounding-class)",
# NOT a broken exact lever, so the byte-identity summary must not FAIL it; every other
# arm keeps the FAIL (an exact lever that changed the tokens is a bug).
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
#   VERIFY_ATTN_FASTPATH (W115): auto-arms the fused/compiled decode SDPA core for the
#       K+1 verify batch, which reassociates the fp32 softmax vs the eager per-row core
#       (rounding-class 1e-6, greedy-identical) -- so the verify tokens can differ from
#       control by rounding, exactly like DECODE_ATTN_KERNEL / ATTN_CORE_COMPILE.
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
    VERIFY_ATTN_FASTPATH_ENV,  # W115: fused/compiled decode SDPA core for the K+1 verify batch (rounding-class softmax reassoc, greedy-identical)
)


def _rounding_class_keys(arm: str) -> list:
    """The rounding-class env keys (see ``ROUNDING_CLASS_ENVS``) an arm's preset
    actually arms -- the reason its tokens are EXPECTED to differ from control by
    rounding.  Empty list for an exact arm.  Derived from ``ARM_PRESETS``, never
    hand-listed, so a new rounding-class arm is classified automatically."""
    preset = ARM_PRESETS.get(arm, {})
    return [k for k in ROUNDING_CLASS_ENVS if preset.get(k) not in (None, "")]


def _is_rounding_class(arm: str) -> bool:
    """True when ``arm`` arms any rounding-class env key (see ``_rounding_class_keys``)."""
    return bool(_rounding_class_keys(arm))


# Derived, not hand-listed: every arm whose preset arms a rounding-class env key.
ROUNDING_CLASS_ARMS = frozenset(a for a in ARM_PRESETS if _is_rounding_class(a))


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
            "When the DSpark head is loaded, reprice its ~7.4 GiB residents out of "
            "the memory budget (default). --no-reprice loads the head at the FULL "
            "budget so window 26 can separate the budget/slots effect from a "
            "head-load code-path effect: the streamed slot plan is identical at the "
            "same budget, so a --no-reprice slowdown is a code path, not slots."
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
        "--box-budget-gib.",
    )
    p.add_argument(
        "--box-budget-gib",
        type=float,
        default=None,
        help="TOTAL box-use budget GiB the plan is derived from (default: "
        "MTPLX_DSV41_BOX_BUDGET_GB env, else 100).",
    )
    # W106 item 3 / MEDIUM-1: derive the plan limit from the TOTAL box budget while
    # COMPENSATING for the non-Metal requirements.  Canonical flags are GiB
    # (`-gib`); the `-gb` spellings are DEPRECATED aliases that convert decimal GB
    # -> GiB at the boundary (see _resolve_gib_flag).  When given, --memory-budget-
    # total-* OVERRIDES --memory-limit-gib (derive, don't take it literally).
    p.add_argument(
        "--memory-budget-total-gib",
        type=float,
        default=None,
        metavar="GIB",
        help="TOTAL box budget in GiB for EVERYTHING; derive the MLX plan limit as "
        "total - system_used_at_start - non_metal_overhead - kv_growth_to_max_kv "
        "- safety (compensates for the non-Metal requirements). Overrides "
        "--memory-limit-gib. Refuses to start if the derived limit is below "
        "--memory-budget-floor-gib.",
    )
    p.add_argument(
        "--memory-budget-total-gb",
        type=float,
        default=None,
        metavar="GB",
        help="DEPRECATED alias of --memory-budget-total-gib; the value is decimal "
        "GB and is converted to GiB (x1e9/2^30).",
    )
    p.add_argument(
        "--memory-safety-gib",
        type=float,
        default=None,
        metavar="GIB",
        help=f"safety headroom in GiB subtracted in the budget derivation (default "
        f"{DEFAULT_MEMORY_SAFETY_GIB:g}).",
    )
    p.add_argument(
        "--memory-safety-gb",
        type=float,
        default=None,
        metavar="GB",
        help="DEPRECATED alias of --memory-safety-gib (decimal GB -> GiB).",
    )
    p.add_argument(
        "--memory-budget-floor-gib",
        type=float,
        default=DEFAULT_MEMORY_BUDGET_FLOOR_GIB,
        metavar="GIB",
        help=f"refuse to start if the budget-derived plan limit is below this floor "
        f"in GiB (default {DEFAULT_MEMORY_BUDGET_FLOOR_GIB:g}).",
    )
    p.add_argument(
        "--non-metal-overhead-gib",
        type=float,
        default=None,
        metavar="GIB",
        help="conservative pre-load estimate in GiB of the non-Metal process "
        "overhead (python heap + expert-reader buffers + engram LRU + tokenizer) "
        f"used in the budget derivation (default {DEFAULT_NON_METAL_OVERHEAD_GIB:g}); "
        "the real value is re-measured after load (and aborts if it blows budget).",
    )
    p.add_argument(
        "--non-metal-overhead-gb",
        type=float,
        default=None,
        metavar="GB",
        help="DEPRECATED alias of --non-metal-overhead-gib (decimal GB -> GiB).",
    )
    # W106 LOW-4: pre-flight the budget derivation from a dry snapshot (no model
    # load, no MLX) so the floor refusal happens BEFORE the guarded GPU window opens
    # (a raise inside _load_model happens after Qwen is already unloaded).
    p.add_argument(
        "--memory-plan-preflight",
        action="store_true",
        default=False,
        help="print the --memory-budget-total-* plan derivation from a dry snapshot "
        "(no model load, no MLX) and exit 0 (plan >= floor) or 3 (below floor), so "
        "the budget can be checked BEFORE the guarded GPU window opens.",
    )
    p.add_argument(
        "--preflight-freed-gib",
        type=float,
        default=None,
        metavar="GIB",
        help="GiB the guarded window will FREE by booting out the resident agent "
        "(com.tea.qwen); subtracted from the pre-flight 'now' baseline so it derives "
        "from the expected in-window baseline. Default: best-effort read-only "
        "auto-detect of the agent RSS, else 0 with a caveat. Pre-flight only.",
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
        "--expert-profile",
        default="deepseek-v41-mxfp4-75",
        help=(
            "profile whose plan fields (transient_slots, split_route_release, "
            "prefetch_slots) seed the in-process runtime when the matching flag "
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
    return p


def _apply_arm_env(arm: str) -> None:
    if arm not in ARM_PRESETS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARM_PRESETS)}")
    for key, value in ARM_PRESETS[arm].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


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
        "prompt_tokens": len(prompt_ids),
        "prompt_build": prompt_meta,
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
# W106 item 3: budget-total -> plan-limit derivation (compensates for the
# non-Metal requirements).  Pure math in derive_budget_total_plan(); the
# measurements are taken by the caller and INJECTED, so this is unit-testable on
# CPU with no MLX, no model and no vm_stat.
# --------------------------------------------------------------------------


class BudgetTotalDerivation:
    """The plan limit derived from a TOTAL box budget, and every term of it.

    ``plan_limit_gib`` is what the MLX plan is fixed to; ``non_metal_overhead_gb``
    is the pre-load estimate actually used to derive it, and
    ``non_metal_overhead_measured_gb`` is the post-load re-measurement (None until
    measured).  ``memory_keys()`` renders the receipt ``memory``-block keys.

    A PLAIN immutable-by-convention class (not ``@dataclass``): this module is a
    script loaded by file path in tests, and ``@dataclass`` under ``from __future__
    import annotations`` needs the module registered in ``sys.modules`` to resolve
    its string annotations -- which a file-path load does not do.
    """

    __slots__ = (
        "source", "budget_total_gb", "system_used_at_start_gb",
        "non_metal_overhead_gb", "kv_growth_to_max_kv_gb", "safety_gb",
        "floor_gib", "plan_limit_gib", "non_metal_overhead_measured_gb",
        "plan_limit_gib_effective", "kv_estimator", "rss_semantics",
    )

    def __init__(
        self,
        *,
        source,
        budget_total_gb,
        system_used_at_start_gb,
        non_metal_overhead_gb,
        kv_growth_to_max_kv_gb,
        safety_gb,
        floor_gib,
        plan_limit_gib,
        non_metal_overhead_measured_gb=None,
        plan_limit_gib_effective=None,
        kv_estimator=None,
        rss_semantics="unmeasured",
    ):
        self.source = source
        self.budget_total_gb = budget_total_gb
        self.system_used_at_start_gb = system_used_at_start_gb
        self.non_metal_overhead_gb = non_metal_overhead_gb
        self.kv_growth_to_max_kv_gb = kv_growth_to_max_kv_gb
        self.safety_gb = safety_gb
        self.floor_gib = floor_gib
        self.plan_limit_gib = plan_limit_gib
        self.non_metal_overhead_measured_gb = non_metal_overhead_measured_gb
        self.plan_limit_gib_effective = plan_limit_gib_effective
        # W107 follow-up: which KV-growth estimator priced budget_kv_growth_to_max_kv_gb
        # -- "w107" (mtplx.models.deepseek_v41_cache.kv_bytes_at_max_kv, the exact
        # per-lane helper) or "local" (the conservative fallback in this module).
        # None on the explicit path (no KV growth term is priced).
        self.kv_estimator = kv_estimator
        # HIGH-A: "ok" (footprint>=active, measured valid), "inverted" (footprint <
        # mx active -> Metal not in phys_footprint, overhead unmeasurable), or
        # "unmeasured" (pre-load / footprint unavailable).
        self.rss_semantics = rss_semantics

    def replace(self, **changes) -> "BudgetTotalDerivation":
        """A copy with the named fields overridden (dataclasses.replace-style)."""
        current = {name: getattr(self, name) for name in self.__slots__}
        current.update(changes)
        return BudgetTotalDerivation(**current)

    def formula(self) -> str:
        return (
            f"plan_limit = budget_total({self.budget_total_gb:.4g}) "
            f"- system_used_at_start({self.system_used_at_start_gb:.4g}) "
            f"- non_metal_overhead({self.non_metal_overhead_gb:.4g}) "
            f"- kv_growth_to_max_kv({self.kv_growth_to_max_kv_gb:.4g}) "
            f"- safety({self.safety_gb:.4g}) "
            f"= {self.plan_limit_gib:.4g} GiB (floor {self.floor_gib:.4g})"
        )

    def memory_keys(self) -> dict:
        """The budget keys merged into the receipt ``memory`` block.  Always the
        SAME key set (nulls where a term does not apply) so receipts are
        self-describing regardless of which plan source ran."""

        return {
            "memory_plan_source": self.source,
            "budget_total_gb": (
                None if self.budget_total_gb is None
                else round(self.budget_total_gb, 4)
            ),
            "plan_limit_gib_derived": round(self.plan_limit_gib, 4),
            "plan_limit_gib_effective": (
                None if self.plan_limit_gib_effective is None
                else round(self.plan_limit_gib_effective, 4)
            ),
            "budget_system_used_at_start_gb": round(self.system_used_at_start_gb, 4),
            "budget_non_metal_overhead_gb": round(self.non_metal_overhead_gb, 4),
            "budget_non_metal_overhead_measured_gb": (
                None if self.non_metal_overhead_measured_gb is None
                else round(self.non_metal_overhead_measured_gb, 4)
            ),
            "budget_kv_growth_to_max_kv_gb": round(self.kv_growth_to_max_kv_gb, 4),
            "budget_kv_estimator": self.kv_estimator,
            "budget_safety_gb": round(self.safety_gb, 4),
            "budget_floor_gib": round(self.floor_gib, 4),
            "rss_semantics": self.rss_semantics,
        }


def derive_budget_total_plan(
    *,
    budget_total_gb: float,
    system_used_at_start_gb: float,
    non_metal_overhead_gb: float,
    kv_growth_to_max_kv_gb: float,
    safety_gb: float = DEFAULT_MEMORY_SAFETY_GIB,
    floor_gib: float = DEFAULT_MEMORY_BUDGET_FLOOR_GIB,
    kv_estimator: str | None = None,
) -> BudgetTotalDerivation:
    """Derive the MLX plan limit from David's TOTAL box budget, compensating for
    the non-Metal requirements.  All measurements are injected (pure math):

        plan_limit = total - system_used_at_start - non_metal_overhead
                           - kv_growth_to_max_kv - safety

    Raises ``ValueError`` (a clear, actionable message) when the derived plan
    limit is below ``floor_gib`` -- refusing to start rather than opening a GPU
    window on a plan too small to hold the model.
    """

    for name, value in (
        ("budget_total_gb", budget_total_gb),
        ("system_used_at_start_gb", system_used_at_start_gb),
        ("non_metal_overhead_gb", non_metal_overhead_gb),
        ("kv_growth_to_max_kv_gb", kv_growth_to_max_kv_gb),
        ("safety_gb", safety_gb),
        ("floor_gib", floor_gib),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value!r}")

    plan_limit = (
        float(budget_total_gb)
        - float(system_used_at_start_gb)
        - float(non_metal_overhead_gb)
        - float(kv_growth_to_max_kv_gb)
        - float(safety_gb)
    )
    if plan_limit < float(floor_gib):
        exc = ValueError(
            f"--memory-budget-total-gib {budget_total_gb:.4g} derives a plan limit "
            f"of {plan_limit:.4g} GiB, BELOW the floor of {floor_gib:.4g} GiB: "
            f"plan_limit = {budget_total_gb:.4g} "
            f"- system_used_at_start {system_used_at_start_gb:.4g} "
            f"- non_metal_overhead {non_metal_overhead_gb:.4g} "
            f"- kv_growth_to_max_kv {kv_growth_to_max_kv_gb:.4g} "
            f"- safety {safety_gb:.4g}. Raise --memory-budget-total-gib, lower "
            f"--memory-safety-gib / --non-metal-overhead-gib, reduce --max-kv, or "
            f"lower --memory-budget-floor-gib (default "
            f"{DEFAULT_MEMORY_BUDGET_FLOOR_GIB:.4g})."
        )
        exc.dsv41_stage = "budget_derivation"  # W106 MEDIUM-C ledger stage
        raise exc
    return BudgetTotalDerivation(
        source="budget",
        budget_total_gb=float(budget_total_gb),
        system_used_at_start_gb=float(system_used_at_start_gb),
        non_metal_overhead_gb=float(non_metal_overhead_gb),
        kv_growth_to_max_kv_gb=float(kv_growth_to_max_kv_gb),
        safety_gb=float(safety_gb),
        floor_gib=float(floor_gib),
        plan_limit_gib=plan_limit,
        kv_estimator=kv_estimator,
    )


def _explicit_plan_derivation(plan_limit_gib: float) -> BudgetTotalDerivation:
    """A BudgetTotalDerivation for the NON-budget path (explicit --memory-limit-gib
    or the legacy --box-budget default): ``memory_plan_source == "explicit"``, the
    budget terms null, so receipts still carry the full budget key set."""

    return BudgetTotalDerivation(
        source="explicit",
        budget_total_gb=None,
        system_used_at_start_gb=0.0,
        non_metal_overhead_gb=0.0,
        kv_growth_to_max_kv_gb=0.0,
        safety_gb=0.0,
        floor_gib=0.0,
        plan_limit_gib=float(plan_limit_gib),
        plan_limit_gib_effective=float(plan_limit_gib),
    )


# Released DeepSeek-V4.1-Flash text shapes the KV estimator falls back to when a
# config field is absent (mtplx/models/deepseek_v41.py ModelArgs defaults).
_KV_CONFIG_DEFAULTS = {
    "num_hidden_layers": 40,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "index_head_dim": 128,
    "window_size": 128,
    "sliding_window": 128,
    "compress_ratios": [],
    # W106 LOW-1: only a FEW layers hold the compressed/index KV lanes (the released
    # DeepSeek-V4.1-Flash kv_source_layer_ids); the rest keep only the window ring.
    # Real config.json values override this default.
    "kv_source_layer_ids": [2, 8, 14, 20],
}


def _read_kv_config_dims(model_path) -> dict:
    """Read the KV-relevant config dims from the artifact's ``config.json`` WITHOUT
    importing MLX or loading weights (CPU-safe).  Handles the flat and the nested
    ``text_config`` spellings; missing fields fall back to the released shapes."""

    dims = dict(_KV_CONFIG_DEFAULTS)
    try:
        cfg_path = Path(model_path).expanduser() / "config.json"
        raw = json.loads(cfg_path.read_text())
    except (OSError, ValueError, TypeError):
        return dims
    if not isinstance(raw, dict):
        return dims
    src = dict(raw)
    text_cfg = raw.get("text_config")
    if isinstance(text_cfg, dict):
        src.update(text_cfg)  # text_config keys win (the model's real field names)
    for key in dims:
        if key in src and src[key] is not None:
            dims[key] = src[key]
    return dims


def _kv_bytes_at_max_kv(config, max_kv: int) -> int:
    """LOCAL, APPROXIMATE estimate of the bytes the DeepSeek-V4.1 KV lanes grow to
    at ``max_kv`` live tokens (batch 1, bf16).  Deliberately CONSERVATIVE (rounds
    every lane UP: it prices the window lane at the full ``max_kv`` because the
    bounded ring is opt-in, and treats an unknown compress_ratio as 1 = no
    pooling), so the plan errs on the safe side of the 100 GB budget.

    FALLBACK only.  As of the W107 merge the budget derivation prefers the exact
    per-lane helper ``mtplx.models.deepseek_v41_cache.kv_bytes_at_max_kv`` (see
    ``_kv_growth_estimate``); this local estimate runs only when that import is
    unavailable (a config-only, MLX-less environment).  The receipt records which
    ran as ``budget_kv_estimator`` ("w107" | "local").

    Per the cache module (mtplx/models/deepseek_v41_cache.py docstring + ModelArgs)
    each layer holds, at bf16 (2 bytes/element), batch 1:
      * a sliding-window ring   [1, window_size, head_dim]           (every layer;
        bounded ONLY under MTPLX_DSV41_WINDOW_RING -- else it grows append-only to
        the sequence length, so priced here at max_kv rows, conservatively)
      * a compressed/latent KV  [1, ceil(max_kv/ratio), head_dim]    (kv-source layers)
      * a decoupled rope key    [1, ceil(max_kv/ratio), qk_rope_head_dim] (kv-source)
      * an index-key lane       [1, ceil(max_kv/ratio), index_head_dim]  (kv-source)
    A layer is a kv-source when it is in ``kv_source_layer_ids`` (authoritative when
    present; else layers with a non-zero ``compress_ratios`` entry) -- LOW-1: only a
    few layers, not all 40.  ``ratio == 1`` is per-token (no pooling), ``ratio > 1``
    pools that many tokens.
    """

    def _cfg(name, default):
        if isinstance(config, dict):
            value = config.get(name, default)
        else:
            value = getattr(config, name, default)
        return default if value is None else value

    max_kv = int(max_kv)
    if max_kv <= 0:
        return 0
    n_layers = int(_cfg("num_hidden_layers", 40))
    head_dim = int(_cfg("head_dim", 512))
    rope_dim = int(_cfg("qk_rope_head_dim", 64))
    index_dim = int(_cfg("index_head_dim", 128))
    window = int(_cfg("sliding_window", 0)) or int(_cfg("window_size", 128))
    ratios = list(_cfg("compress_ratios", []) or [])
    kv_src = {int(x) for x in (_cfg("kv_source_layer_ids", []) or [])}
    bf16 = 2

    def _rows_for_ratio(ratio: int) -> int:
        if ratio <= 1:
            return max_kv
        return -(-max_kv // ratio)  # ceil

    def _is_kv_source(layer: int) -> bool:
        # LOW-1: kv_source_layer_ids is authoritative when present; else fall back
        # to a non-zero compress_ratios entry.
        if kv_src:
            return layer in kv_src
        return layer < len(ratios) and int(ratios[layer]) != 0

    total = 0
    for layer in range(n_layers):
        # Window ring: conservatively priced at max_kv rows (ring is opt-in), EVERY
        # layer.
        total += max_kv * head_dim * bf16
        if not _is_kv_source(layer):
            continue  # not a kv-source layer: only the window ring above
        # ratio from compress_ratios when known/positive, else 1 (no pooling).
        ratio = int(ratios[layer]) if (layer < len(ratios) and int(ratios[layer]) > 0) else 1
        rows = _rows_for_ratio(ratio)
        total += rows * head_dim * bf16      # latent / compressed KV
        total += rows * rope_dim * bf16      # decoupled rope key
        total += rows * index_dim * bf16     # index key
    return int(total)


def _kv_growth_estimate(dims: dict, max_kv: int) -> tuple[int, str]:
    """The KV-growth-to-``max_kv`` budget term (bytes) and the estimator that
    produced it (``"w107"`` | ``"local"``), for the receipt ``budget_kv_estimator``.

    Prefer the EXACT per-lane helper
    ``mtplx.models.deepseek_v41_cache.kv_bytes_at_max_kv`` (W107): it prices the
    bounded lanes exactly as the cache preallocates them -- the window ring bounded
    and INDEPENDENT of ``max_kv``, and compress / index / latent only on the
    ``kv_source`` layers -- so the derived plan matches what the bounded arm
    actually allocates.  Fall back to the LOCAL conservative estimate
    (:func:`_kv_bytes_at_max_kv`) only if that import is unavailable (a config-only,
    MLX-less environment).  ``dims`` is the flat config dict from
    :func:`_read_kv_config_dims`; the W107 helper reads it as attributes, so it is
    wrapped in a ``SimpleNamespace``."""

    try:
        from mtplx.models.deepseek_v41_cache import (
            kv_bytes_at_max_kv as _w107_kv_bytes_at_max_kv,
        )
    except Exception:
        return int(_kv_bytes_at_max_kv(dims, int(max_kv))), "local"
    cfg = types.SimpleNamespace(**dims)
    return int(_w107_kv_bytes_at_max_kv(cfg, int(max_kv))), "w107"


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


def _measure_system_used_at_start_bytes(args, bench) -> int:
    """The system-wide used-memory baseline (vm_stat, the SAME formula the
    gpu_window.sh guard uses -- factored in bench._system_used_bytes).  Measured
    ONCE at process start and cached on ``args`` so every arm derives from the
    same baseline."""

    cached = getattr(args, "_dsv41_system_used_at_start_bytes", None)
    if cached is not None:
        return int(cached)
    used = int(bench._system_used_bytes())
    args._dsv41_system_used_at_start_bytes = used
    return used


def _budget_total_gib(args):
    """The resolved TOTAL box budget in GiB from --memory-budget-total-gib (or the
    deprecated -gb alias), or None when neither is set."""

    return _resolve_gib_flag(
        args, "memory_budget_total_gib", "memory_budget_total_gb", None,
        "--memory-budget-total",
    )


def _detect_qwen_rss_bytes(label="com.tea.qwen"):
    """Best-effort, READ-ONLY estimate of the resident agent's RSS in bytes via
    ``launchctl print`` (pid) + ``ps -o rss=`` -- neither mutates anything.  Used
    ONLY by the pre-flight (below) to estimate what the guarded window will free by
    booting the agent out; the in-window run measures the real post-bootout baseline
    itself.  Returns None on any failure.  (Not invoked by the tests, which pass
    --preflight-freed-gib explicitly.)"""

    try:
        uid = os.getuid()
        out = subprocess.run(
            ["/bin/launchctl", "print", f"gui/{uid}/{label}"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        pid = None
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("pid = "):
                pid = s.split("=", 1)[1].strip()
                break
        if not pid or not pid.isdigit():
            return None
        rss = subprocess.run(
            ["/bin/ps", "-o", "rss=", "-p", pid],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return int(rss) * 1024 if rss.isdigit() else None
    except Exception:  # pragma: no cover - defensive
        return None


def _derive_budget_total(args, bench, max_kv, *, system_used_gb=None):
    """Compute the item-3 ``BudgetTotalDerivation`` from the resolved flags + the
    measured system-used baseline + the KV estimate, or return None when no budget
    flag was given.  Shared by ``_resolve_derivation`` and the pre-flight.
    ``system_used_gb`` overrides the measured baseline (the pre-flight passes a
    freed-adjusted value)."""

    budget_total = _budget_total_gib(args)
    if budget_total is None:
        return None
    if bench is None or max_kv is None:
        raise ValueError(
            "--memory-budget-total-gib needs the bench module and resolved max_kv "
            "to price the KV growth"
        )
    if system_used_gb is None:
        system_used_gb = _measure_system_used_at_start_bytes(args, bench) / GIB
    non_metal_gb = _resolve_gib_flag(
        args, "non_metal_overhead_gib", "non_metal_overhead_gb",
        DEFAULT_NON_METAL_OVERHEAD_GIB, "--non-metal-overhead",
    )
    safety_gb = _resolve_gib_flag(
        args, "memory_safety_gib", "memory_safety_gb",
        DEFAULT_MEMORY_SAFETY_GIB, "--memory-safety",
    )
    floor_gib = float(
        getattr(args, "memory_budget_floor_gib", DEFAULT_MEMORY_BUDGET_FLOOR_GIB)
    )
    dims = _read_kv_config_dims(getattr(args, "model", None))
    # W97F Fix 1 (preserved through the W106 refactor): price the KV growth with the
    # EXACT W107 per-lane helper (deepseek_v41_cache.kv_bytes_at_max_kv) when it can be
    # imported, falling back to the LOCAL estimator (_kv_bytes_at_max_kv, which carries
    # W106's kv_source_layer_ids fix) only in an MLX-less config-only environment.  The
    # estimator that actually ran is recorded on the plan as budget_kv_estimator.
    kv_growth_bytes, kv_estimator = _kv_growth_estimate(dims, int(max_kv))
    kv_growth_gb = kv_growth_bytes / GIB
    return derive_budget_total_plan(
        budget_total_gb=float(budget_total),
        system_used_at_start_gb=system_used_gb,
        non_metal_overhead_gb=non_metal_gb,
        kv_growth_to_max_kv_gb=kv_growth_gb,
        safety_gb=safety_gb,
        floor_gib=floor_gib,
        kv_estimator=kv_estimator,
    )


def _preflight_memory_plan(args, bench) -> int:
    """W106 LOW-4 / HIGH-B pre-flight: derive the budget plan from a DRY snapshot (no
    model load, no MLX) and exit 0 (plan >= floor) or 3 (below floor), BEFORE the
    guarded GPU window opens so a floor refusal never fires after Qwen is already
    unloaded.

    HIGH-B: the "now" baseline still has the resident agent (com.tea.qwen, ~45 GiB)
    and any bench worker resident, but the guarded window BOOTS THAT OUT before the
    step runs.  ``--preflight-freed-gib N`` (default: best-effort read-only
    auto-detect of the agent RSS, else 0 with a caveat) is subtracted so the
    pre-flight derives from the EXPECTED IN-WINDOW baseline, not the crowded "now"
    baseline.  Both baselines are printed; the real in-window derivation (measured
    after bootout) is authoritative."""

    # LOW: a both-set flag error must exit 3, not traceback.
    try:
        budget = _budget_total_gib(args)
    except ValueError as exc:
        print(f"[ab] memory-plan preflight: REFUSED -- {exc}", flush=True)
        return 3
    if budget is None:
        print(
            "[ab] memory-plan preflight: no --memory-budget-total-gib/-gb given; "
            "plan source is explicit (--memory-limit-gib / legacy box-budget). OK.",
            flush=True,
        )
        return 0

    max_kv = bench.resolve_max_kv(
        [args.context_tokens], args.decode_tokens, args.max_kv
    )
    used_now_gb = int(bench._system_used_bytes()) / GIB

    # Resolve how much the window will free by booting out the resident agent.
    freed = getattr(args, "preflight_freed_gib", None)
    if freed is not None:
        freed_gb = float(freed)
        freed_src = "explicit --preflight-freed-gib"
    else:
        det = _detect_qwen_rss_bytes()
        if det is not None:
            freed_gb = det / GIB
            freed_src = "auto-detected com.tea.qwen RSS (launchctl+ps, read-only)"
        else:
            freed_gb = 0.0
            freed_src = ("0 -- could NOT detect the resident agent; pass "
                         "--preflight-freed-gib to model the bootout")
    baseline_in_window = max(0.0, used_now_gb - freed_gb)
    print(
        f"[ab] memory-plan preflight: system used now {used_now_gb:.2f} GiB; "
        f"expected in-window {baseline_in_window:.2f} GiB "
        f"(freed {freed_gb:.2f} GiB via {freed_src})",
        flush=True,
    )

    try:
        bt = _derive_budget_total(
            args, bench, max_kv, system_used_gb=baseline_in_window
        )
    except ValueError as exc:
        print(f"[ab] memory-plan preflight: REFUSED -- {exc}", flush=True)
        return 3
    print(
        "[ab] memory-plan preflight: OK (plan >= floor)\n"
        f"[ab]   max_kv={max_kv}\n"
        f"[ab]   {bt.formula()}\n"
        "[ab]   NB: the in-window derivation (measured after the agent is booted "
        "out) is authoritative; this is a pre-check.",
        flush=True,
    )
    return 0


def _resolve_derivation(args, *, bench=None, max_kv=None):
    """The plan->limit derivation for this run.

    Precedence:
      1. ``--memory-budget-total-gib`` (or the deprecated -gb alias) -> derive the
         plan limit from the TOTAL box budget, COMPENSATING for the non-Metal
         requirements (item 3).  This OVERRIDES ``--memory-limit-gib``.
      2. ``--memory-limit-gib`` -> explicit plan (source "explicit").
      3. otherwise the legacy W62 --box-budget derivation (source "explicit").

    Returns the W62 ``BudgetDerivation`` (its memory_limit_bytes/reserve/cache
    plumb into the loader unchanged); the item-3 ``BudgetTotalDerivation`` is
    stashed on ``args._dsv41_budget_total`` for the receipt.
    """

    from mtplx.deepseek_v41_memory_profile import derive_plan_from_budget

    bt = _derive_budget_total(args, bench, max_kv)
    if bt is not None:
        args._dsv41_budget_total = bt
        override = bt.plan_limit_gib
    else:
        override = getattr(args, "memory_limit_gib", None)
        args._dsv41_budget_total = None

    derivation = derive_plan_from_budget(
        box_budget_gib=getattr(args, "box_budget_gib", None),
        override_memory_limit_gib=override,
    )
    if args._dsv41_budget_total is None:
        # Non-budget path: record the actually-used plan limit as "explicit" so the
        # receipt memory block carries the full budget key set (nulls elsewhere).
        args._dsv41_budget_total = _explicit_plan_derivation(derivation.plan_gib)
    return derivation


def _budget_memory_keys(args) -> dict:
    """The item-3 budget keys to merge into a receipt ``memory`` block."""

    bt = getattr(args, "_dsv41_budget_total", None)
    if bt is None:
        return _explicit_plan_derivation(0.0).memory_keys()
    return bt.memory_keys()


def _memory_block_extra_keys(args) -> dict:
    """The W106 keys merged into every receipt ``memory`` block: the item-3 budget
    derivation terms plus the HIGH-2 ``rss_semantics_note``."""

    keys = _budget_memory_keys(args)
    keys["rss_semantics_note"] = _RSS_SEMANTICS_NOTE
    return keys


# Plan fields the served profile sets that the loader would otherwise default
# (the W81 finding: transient_slots defaulted to spec.top_k=6, not the profile's
# 48).  Seeded into the in-process runtime so a bench A/B is on the production
# plan.  ``transient_slots`` also takes the explicit ``--transient-slots`` flag.
_PROFILE_PLAN_FIELDS = ("transient_slots", "split_route_release", "prefetch_slots")


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
    slots_per_layer_no_ring = slots_per_layer
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
        "transient_slots": transient_slots,
        "persistent_slots": persistent_slots,
        # review MEDIUM-d: LRU depth WITH the ring vs the hypothetical no-ring plan.
        "slots_per_layer": slots_per_layer,
        "slots_per_layer_no_ring": slots_per_layer_no_ring,
        "expert_record_bytes": record_bytes,
        "transient_bytes_per_layer": transient_slots * record_bytes,
        "transient_bytes_total": transient_slots * record_bytes * routed_layers,
        "split_route_release": getattr(config, "split_route_release", None),
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
        "source": (
            "explicit" if getattr(args, "transient_slots", None) is not None
            else f"profile:{getattr(args, 'expert_profile', 'none')}"
            if getattr(args, "_dsv41_plan_overrides", None)
            else "loader-default(top_k)"
        ),
    }


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
    _bt = getattr(args, "_dsv41_budget_total", None)
    if _bt is not None and _bt.source == "budget":
        print("[ab] budget-total derivation: " + _bt.formula(), flush=True)
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
            f"[ab] additional resident reserve: SWA {_swa_bytes / GIB:.3f} GiB"
            + (
                f" + wo_a f32 cache {_wo_a_reserve / GIB:.3f} GiB"
                if _wo_a_reserve
                else ""
            )
            + f" = {_addl / GIB:.3f} GiB (priced into the plan)",
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
    if args.apply_memory_cap:
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
    # W106 item 3 (two-phase): the plan limit had to be fixed BEFORE load with the
    # conservative non_metal_overhead estimate; now the model is resident, re-MEASURE
    # the real non-Metal overhead from the CURRENT process footprint and record
    # estimate-vs-measured.  HIGH-1 fix: NEVER call mx.set_memory_limit post-load --
    # residents are already allocated so it cannot shrink anything, and a limit
    # below active memory would silently perturb the measured decode.  If the
    # measured overhead blows the budget, ABORT here (before decode) instead.
    _remeasure_non_metal_overhead(args, mx)
    return resident


def _remeasure_non_metal_overhead(args, mx) -> None:
    """Phase 2 of the item-3 budget derivation.  Measure the real non-Metal process
    overhead as ``current phys_footprint (mach task_info) - mx active - mx cache``
    (NOT ru_maxrss, a lifetime high-water) and record estimate-vs-measured, and:
      * subtract the MLX freed-buffer CACHE (``get_cache_memory``) as well as active
        -- the cache is load-transient Metal memory the allocator will reuse, NOT
        non-Metal overhead; counting it caused false aborts (HIGH-A).  We do NOT
        call ``mx.clear_cache()`` (it would perturb the first decode token's
        allocations); subtracting the cache is the non-perturbing equivalent.
      * NEVER call ``mx.set_memory_limit`` post-load (HIGH-1).
      * if ``footprint < active`` (Metal not in phys_footprint on this platform) the
        overhead is UNMEASURABLE: record ``None`` + ``rss_semantics="inverted"`` +
        a WARN, and do NOT abort (never a bogus 0.0 that hides the inversion).
      * otherwise ABORT (raise, before decode) when the measured overhead exceeds
        the estimate by more than the tolerance (the real footprint would then
        exceed the budget).
    Measurement itself is guarded (a read failure records None, no abort)."""

    bt = getattr(args, "_dsv41_budget_total", None)
    if bt is None or bt.source != "budget" or mx is None:
        return
    try:
        from mtplx.deepseek_v41_memory_profile import (
            mlx_memory_snapshot,
            process_rss_snapshot,
        )

        snap = process_rss_snapshot()
        footprint = snap.get("phys_footprint_bytes") or snap.get("resident_bytes")
        mlx = mlx_memory_snapshot(mx_module=mx)
        active = int(mlx.get("active_bytes", 0) or 0)
        cache = int(mlx.get("cache_bytes", 0) or 0)
    except Exception:  # pragma: no cover - defensive
        return

    if footprint is None:
        print(
            "[ab] budget-total re-measure: process footprint unavailable "
            "(non-darwin / mach); keeping the pre-load estimate, MLX limit unchanged",
            flush=True,
        )
        args._dsv41_budget_total = bt.replace(
            non_metal_overhead_measured_gb=None,
            plan_limit_gib_effective=bt.plan_limit_gib,
            rss_semantics="unmeasured",
        )
        return

    footprint = int(footprint)
    if footprint < active:
        # Metal is not counted in phys_footprint on this platform -> the non-Metal
        # overhead cannot be derived by subtraction. Record it as inverted (never a
        # misleading 0.0) and do not abort.
        print(
            f"[ab] budget-total re-measure: WARN phys_footprint "
            f"{footprint / GIB:.2f} GiB < mx active {active / GIB:.2f} GiB "
            "(Metal not in footprint); non_metal_overhead UNMEASURABLE, recorded "
            "None (rss_semantics=inverted); MLX limit unchanged, no abort",
            flush=True,
        )
        args._dsv41_budget_total = bt.replace(
            non_metal_overhead_measured_gb=None,
            plan_limit_gib_effective=bt.plan_limit_gib,
            rss_semantics="inverted",
        )
        return

    measured_gb = max(0.0, (footprint - active - cache) / GIB)
    overage_gb = measured_gb - bt.non_metal_overhead_gb
    # plan_limit is NEVER lowered post-load (HIGH-1).
    args._dsv41_budget_total = bt.replace(
        non_metal_overhead_measured_gb=measured_gb,
        plan_limit_gib_effective=bt.plan_limit_gib,
        rss_semantics="ok",
    )
    print(
        f"[ab] budget-total re-measure: non_metal_overhead measured "
        f"{measured_gb:.2f} GiB (phys_footprint {footprint / GIB:.2f} - mx active "
        f"{active / GIB:.2f} - mx cache {cache / GIB:.2f}); estimate "
        f"{bt.non_metal_overhead_gb:.2f} GiB; MLX limit unchanged "
        "(set_memory_limit is NOT called post-load)",
        flush=True,
    )
    if overage_gb > _BUDGET_REMEASURE_TOLERANCE_GIB:
        exc = RuntimeError(
            "budget-total re-measure ABORT (before decode): measured non-Metal "
            f"overhead {measured_gb:.2f} GiB exceeds the pre-load estimate "
            f"{bt.non_metal_overhead_gb:.2f} GiB by {overage_gb:.2f} GiB, so the real "
            f"footprint would exceed --memory-budget-total-gib "
            f"{bt.budget_total_gb:.4g} by ~{overage_gb:.2f} GiB. Re-run with "
            f"--non-metal-overhead-gib >= {measured_gb:.2f}, a lower --max-kv, or a "
            "higher budget; refusing to run the decode over budget."
        )
        exc.dsv41_stage = "budget_remeasure"  # W106 MEDIUM-C ledger stage
        raise exc


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
        return snapshot_stream_counters(rt)
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
              util_sampler=None, stage_timing=False):
    """Greedy prefill + ``steps`` decode; captures the decoded token ids.

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
    # W106: sample process RSS + system used memory off the hot path (daemon thread,
    # 1 Hz, no MLX calls) over the whole generation, so the receipt's memory block
    # carries the real envelope, not just the MLX allocator peak (peak_gb).
    _mem_sampler = mem_probe.new_sampler()
    _mem_sampler.start()
    try:
        t0 = time.perf_counter()
        cache = model.make_cache()
        logits = model(ops.input([list(prompt_ids)]), cache=cache)
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
        # W81: bracket the DECODE loop for the serve_stream_counters block (prefill
        # excluded so the hit rate is the decode hit rate).
        _sc_after_prefill = _stream_counters_snapshot(model)
        extra_forward_steps = 0

        # W92 switch-dispatch census: arm the route-stage probe scoped to the DECODE
        # loop (prefill excluded) so the receipt reports per-layer host syncs
        # (hot.eval_indices), all-hit fences deferred vs synced (hot.allhit_defer vs
        # hot.allhit_fence_eval), and gather_qmm dispatches per switch call
        # (hot.allhit_gather_qmm).  ENABLED is read at use, so setting it here arms the
        # module even if the launch env did not; counters cleared to scope to decode.
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

        _util_cm = util_sampler if util_sampler is not None else contextlib.nullcontext()
        # W90: the sampler's macmon Popen/terminate happen on the context enter/exit; take
        # decode_start AFTER enter and decode_wall_s BEFORE exit, so tok/s excludes them.
        with _util_cm:  # W90: macmon utilization sampled over the DECODE loop only
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
                        stop_ids=set(),
                    )
                    generated.extend(int(t) for t in more)
                else:
                    every = max(1, int(mem_profile_every))
                    for step in range(int(steps)):
                        logits = model(ops.input([[token]]), cache=cache)
                        ops.sync(logits)
                        token = ops.argmax_last(logits)
                        generated.append(token)
                        if mem_profile is not None and (step + 1) % every == 0:
                            mem_profile("decode", token=step + 1)
            finally:
                # W92: restore the probe ENABLED flag even if the decode loop raised, so
                # a failed arm never leaves the module armed for the rest of the process
                # (the snapshot below reads _COUNTS regardless of ENABLED).
                if _route_probe is not None and _route_prev_enabled is not None:
                    _route_probe.ENABLED = bool(_route_prev_enabled)
            decode_wall_s = time.perf_counter() - decode_start
        _sc_end = _stream_counters_snapshot(model)
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
        "peak_gb": mem_probe.peak_bytes() / GIB,
        # W106: full memory envelope (mlx_peak_gb == peak_gb, + process RSS peak and
        # the whole-box used-memory peak the gpu_window.sh guard aborts on).
        "memory": mem_probe.memory_block(_mem_sampler),
        "extra_forward_steps": int(extra_forward_steps),
        "stream_after_prefill": _sc_after_prefill,
        "stream_end": _sc_end,
        # W95f: the v2 runner + gate_prefetch receipt blocks (present only when armed).
        **_runner_receipt_blocks(model),
        "cooldown": cooldown_block,
        "utilization": (
            util_sampler.summarize() if util_sampler is not None else None
        ),
        "switch_dispatch": switch_dispatch,
    }


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4g}"


def _print_dspark_divergence(arm: str, d: dict) -> None:
    """W77 census line for a classified DSpark divergence.  ``tie_flip`` is a
    one-line note (acceptable, rounding-class); ``divergent`` is LOUD (the flip is
    larger than the bf16 rounding envelope -- a real lane bug or a non-rounding
    lever), so the operator sees it in the arm log even though the arm no longer
    aborts."""
    i = d["divergence_index"]
    if d["class"] == "tie_flip":
        print(
            f"[ab] dspark divergence @ {i} class=tie_flip (acceptable) "
            f"ar_top2_margin={_fmt(d['ar_top2_margin'])} < tie_margin={_fmt(d['tie_margin'])} "
            f"max|Δlogit|={_fmt(d['max_abs_logit_delta'])} "
            f"ar_tok={d['ar_token']} dsp_tok={d['dspark_token']} (arm {arm!r})",
            flush=True,
        )
    else:
        print(
            "[ab] " + "!" * 8 + " DIVERGENT " + "!" * 8 + "\n"
            f"[ab] DSpark greedy stream != AR @ {i} class=DIVERGENT (arm {arm!r}): "
            f"NOT a tie-break flip -- ar_top2_margin={_fmt(d['ar_top2_margin'])} "
            f">= tie_margin={_fmt(d['tie_margin'])}, max|Δlogit|={_fmt(d['max_abs_logit_delta'])}, "
            f"dspark_top2_margin={_fmt(d['dspark_top2_margin'])}; "
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
    logits = model(ops.input([list(prompt_ids)]), cache=cache)
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
    """W115: zero the decode-attention-core, fused-projection and verify-fast-path
    engagement counters before a dspark headline pass so the census is scoped to that
    pass (the top-level ``*_engagement`` blocks are AR-scoped).  Best-effort: a module
    that is absent on an older build is skipped, and the returned dict records which
    modules were reset so :func:`_capture_dspark_engagement` only reports those."""
    ok: dict = {}
    try:
        from mtplx.models import deepseek_v41 as _dsv41
        _dsv41._reset_attn_core_compile_calls()
        _dsv41._reset_verify_attn_fastpath_calls()
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

    ``verify_attn_fastpath_engagement`` (calls/rows/fallbacks the W115 lever governed),
    ``attn_core_compile_engagement`` (compiled vs eager cores -- the verify uses the
    compiled core on CPU / when K29 declines), ``decode_attn_kernel_engagement`` (K29
    dispatches -- the GPU verify core), and ``fused_proj_engagement`` (W101 qkv/out
    calls + rows -- now INCLUDING the verify rows, the window-43 gap)."""
    out: dict = {}
    if reset_ok.get("dsv41"):
        from mtplx.models import deepseek_v41 as _dsv41
        out["verify_attn_fastpath_engagement"] = _dsv41._verify_attn_fastpath_engagement()
        out["attn_core_compile_engagement"] = _dsv41._attn_core_compile_calls()
    if reset_ok.get("k29"):
        from mtplx.models import deepseek_v41_attn_kernels as _k29
        out["decode_attn_kernel_engagement"] = _k29.engagement()
    if reset_ok.get("fp"):
        from mtplx.models import deepseek_v41_fused_proj_kernels as _fp
        out["fused_proj_engagement"] = _fp.engagement()
    return out


def _generate_dspark(*, model, mx, mem_probe, prompt_ids, steps, depth,
                     stage_timing=False, ar_reference=None):
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
    # W106: 1 Hz off-hot-path RSS + system-used sampler over the headline pass (see
    # _generate). Stopped right after the headline peak_gb is captured, before the
    # optional timed stage-timing pass, so the memory block matches that peak_gb.
    _mem_sampler = mem_probe.new_sampler()
    _mem_sampler.start()
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

        # W115: zero the attention-core / projection / verify-fast-path engagement
        # counters right before the UNTIMED headline pass so the dspark receipt block
        # reports THIS pass's real verify engagement (the fused decode core + W101
        # projections the verify rows drove) rather than the AR pass's -- the receipt's
        # top-level fused_proj_engagement is AR-scoped, so without this the verify rows
        # would never be counted (the window-43 "rows = AR only" gap).  Captured right
        # after the headline pass, before the timed stage-timing pass (which forces the
        # cores eager, adding eager fallbacks that would pollute the census).
        _eng_reset = _reset_dspark_engagement_counters()
        # W91: HEADLINE pass is UNTIMED (no W37 recording armed) so fused decode levers
        # (K35 small-stages) are ACTIVE for the tok/s the receipt reports.  Arming
        # ``_stime`` forces ``_small_stages_use`` eager (the recording guard), so a timed
        # headline would measure K35 OFF while the arm env says ON.  Mirrors the AR path
        # (untimed ``_generate`` headline + a separate ``_stage_timing_pass``): the
        # per-stage attribution is a SECOND, timed pass below.
        t0 = time.perf_counter()
        toks = dspark_generate(
            model,
            [int(t) for t in prompt_ids],
            max_tokens=int(steps) + 1,
            sampler=SamplerConfig(temperature=0.0),
            seed=0,
            speculative_depth=int(depth),
            stats=stats,
            divergence_capture=capture,
            prefill_callback=_stream_prefill_cb,
        )
        _sc["end"] = _stream_counters_snapshot(model)
        # W115: capture the verify-scoped engagement (fused decode core + W101
        # projections + verify-fast-path) from the headline pass, before the timed pass.
        _dspark_engagement = _capture_dspark_engagement(_eng_reset)
        # W100: exclude the re-prefill from decode_wall_s (the decode loop only, from
        # the prefill->decode boundary). Keep the whole-call wall as pass_wall_s.
        _wall_acct = _dspark_decode_wall_accounting(
            pass_start=t0,
            decode_start=_sc.get("decode_start"),
            pass_end=time.perf_counter(),
            generated_tokens=len(toks),
        )
        peak_gb = mem_probe.peak_bytes() / GIB  # headline peak, captured before the timed pass
    finally:
        _mem_sampler.stop()
    _dspark_memory_block = mem_probe.memory_block(_mem_sampler)
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
                stats=DSparkDecodeStats(),
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
    """The whole-PROCESS peak RSS (incl. the non-Metal Python heap + expert-reader
    buffers) from the in-process 1 Hz sampler, for the top-level receipt key
    ``peak_process_gb``.  David's "peak memory must include non-Metal parts" fix:
    the legacy ``peak_gb`` is the MLX allocator peak only.  ``None`` when a pass
    produced no memory block."""

    mem = (run or {}).get("memory") or {}
    val = mem.get("process_peak_rss_gb")
    return None if val is None else float(val)


def _memory_headline(receipt) -> str:
    """The ``[ab]`` console peak-memory fragment.  Prints the legacy MLX peak AND
    the whole-process RSS peak + box used-memory peak (David's non-Metal fix), so
    the operator sees the real footprint, not just the MLX allocator figure.
    ``peak_gb`` stays MLX-only for old-receipt comparability; the process/system
    figures come from the in-process sampler over prefill+decode (peak, not exit)."""

    mem = receipt.get("memory") or {}
    peak_gb = receipt.get("peak_gb", 0.0) or 0.0
    return (
        f"peak_gb={peak_gb:.2f}"
        f" mlx_peak_gb={mem.get('mlx_peak_gb', peak_gb):.2f}"
        f" process_peak_rss_gb={mem.get('process_peak_rss_gb', 0.0):.2f}"
        f" system_used_peak_gb={mem.get('system_used_peak_gb', 0.0):.2f}"
        f" (sys start {mem.get('system_used_at_start_gb', 0.0):.2f})"
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
    """Decode token ids to text with the bench's already-loaded tokenizer.  Guarded:
    returns None on any failure or when no tokenizer is available (e.g.
    --prompt-ids-file), so a decode never kills a run."""

    if tok is None or ids is None:
        return None
    decode = getattr(tok, "decode", None)
    if not callable(decode):
        return None
    try:
        return decode([int(t) for t in ids])
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[ab] WARN: token decode failed ({exc!r}); text output omitted",
              flush=True)
        return None


def _text_output_fields(tok, ids) -> dict:
    """The receipt text-audit fields for one id stream: the FULL id list, the full
    decoded text, and its head/tail (first/last 600 chars)."""

    ids_list = [int(t) for t in (ids or [])]
    text = _decode_ids(tok, ids_list)
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
    }


def _divergence_context(tok, ids, token_index, span=_DIVERGENCE_CONTEXT_CHARS):
    """The decoded text ``span`` chars either side of the character offset that the
    divergence TOKEN index maps to (decode the prefix to find the offset).  None
    when the stream cannot be decoded."""

    full = _decode_ids(tok, ids)
    if full is None:
        return None
    prefix = _decode_ids(tok, list(ids)[: max(0, int(token_index))])
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
    }
    bt = getattr(args, "_dsv41_budget_total", None) if args is not None else None
    if bt is not None:
        try:
            row["memory"] = bt.memory_keys()
        except Exception:  # pragma: no cover - defensive
            pass
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
    # W107: a bounded arm that did not pin an explicit MTPLX_DSV41_KV_BOUNDED_MAXKV
    # (the presets do not know the CLI --max-kv) preallocates every KV lane to the
    # resolved cell max_kv.  Stamp it here, after the arm env is applied and BEFORE
    # make_cache (per request), so the cache reads it at construction.  Read-at-use,
    # not import.  A bounded arm with neither key set falls back to geometric growth
    # (the kv_realloc_* counters then flag it).
    if (os.environ.get(KV_BOUNDED_ENV) or "").strip().lower() in ("1", "true", "yes", "on"):
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
    if getattr(args, "dry_run", False):
        return _dry_run_arm(args, arm, bench)
    build_prompt = bench._load_build_prompt()
    _tok = None if getattr(args, "prompt_ids_file", None) else _tokenizer(args, bench)
    prompt_ids, prompt_meta = bench._resolve_prompt(
        args, _tok, build_prompt, args.context_tokens
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
        # W115: zero the verify-fast-path engagement so the AR-pass receipt reports 0
        # (M=1 AR has no verify batch); the dspark block re-zeros + reports the verify.
        _dsv41._reset_verify_attn_fastpath_calls()
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
        )
        if util_sampler is not None:
            print(f"[ab] {arm}: {util_sampler.census()}", flush=True)
        ids = run["generated"]
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
            "prompt_tokens": len(prompt_ids),
            "ttft_s": run["ttft_s"],
            "prefill_tok_s": (len(prompt_ids) / run["ttft_s"])
            if run["ttft_s"] > 0
            else None,
            "decode_wall_s": run["decode_wall_s"],
            "decode_tok_s": (args.decode_tokens / run["decode_wall_s"])
            if run["decode_wall_s"] > 0
            else None,
            # peak_gb is the MLX allocator peak ONLY (kept as-is for old receipts'
            # comparability); peak_process_gb is the whole-PROCESS peak RSS incl.
            # the non-Metal footprint (David's fix). Both come off run["memory"].
            "peak_gb": run["peak_gb"],
            "peak_process_gb": _peak_process_gb(run),
            # W106: the memory envelope (mlx_peak_gb == the peak_gb above, plus the
            # process RSS peak and the whole-box used-memory peak/at-start the
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
            **_text_output_fields(_tok, ids),
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
            # W115: verify-attention fast-path engagement over the AR pass (calls 0 --
            # AR is M=1, no verify batch; the DSpark verify count is in the dspark
            # block).  Present at the top level so the receipt schema always carries it.
            "verify_attn_fastpath_engagement": (
                _dsv41._verify_attn_fastpath_engagement() if _dsv41 is not None else None
            ),
            # W81: the ACTUAL slot plan this arm ran (transient/persistent slot
            # counts + bytes + source), so an A/B is attributable to a capacity and
            # the in-process bench is comparable to the served profile plan.
            "resolved_plan": _resolved_plan(runtime, args),
            # W81: DECODE-scoped expert-streaming counters for the AR reference
            # decode (hit rate + streamed bytes/token), matching the served
            # daemon's serve_stream_counters. David: hit rate + bandwidth/token.
            "serve_stream_counters": _stream_counters_block(
                run, args.decode_tokens, _resolved_plan(runtime, args)
            ),
            # W92 switch-dispatch census (present only with --stage-timing): per-layer
            # host syncs + all-hit fences deferred vs synced + gather_qmm/switch-call.
            "switch_dispatch": run.get("switch_dispatch"),
        }
        if mem_profile_snaps is not None:
            from mtplx.deepseek_v41_memory_profile import (
                format_memory_profile_table,
            )

            mem_profile_cb("decode", token=int(args.decode_tokens))
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
                stage_timing=bool(getattr(args, "stage_timing", False)),
                ar_reference=ids,
            )
            dsp_ids = dsp["generated"]
            byte_identical = dsp_ids == ids
            st = dsp["stats"]
            receipt["dspark"] = {
                "depth": int(args.dspark_depth),
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
                "peak_process_gb": _peak_process_gb(dsp),  # whole-process RSS peak
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
            }
            # W106 output persistence: the FULL DSpark stream (ids + decoded text)
            # AND the AR comparison stream it is verified against, both under the
            # dspark block so the divergence is text-auditable from the receipt.
            receipt["dspark"].update(_text_output_fields(_tok, dsp_ids))
            receipt["dspark"]["ar_reference"] = _text_output_fields(_tok, ids)
            if dsp.get("verify_stage_timing") is not None:
                # W37 internal breakdown of the verify forward (attention +
                # moe.routed_switch): the census that shows the routing phase.
                receipt["dspark"]["verify_stage_timing"] = dsp["verify_stage_timing"]
            if dsp.get("w61_engagement") is not None:
                # W61 single-barrier fast-path engagement (+ eval_indices barrier
                # count) from the route-stage probe.
                receipt["dspark"]["w61_engagement"] = dsp["w61_engagement"]
            # W115: verify-SCOPED engagement (from the headline pass) so this arm's
            # verify rows are counted -- verify_attn_fastpath_engagement (the lever's
            # calls/rows/fallbacks), and the decode-core / fused-proj engagement now
            # reflecting the verify rows (NOT the AR pass at the receipt top level).
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
                    "ar": _divergence_context(_tok, ids, first),
                    "dspark": _divergence_context(_tok, dsp_ids, first),
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
                # the capture index too so a mismatch is visible.
                if cap is not None:
                    divergence["capture_index"] = cap.get("index")
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
            )
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
            # than the cap).  ``kv_inplace_writes_<lane>`` is the O(new-rows) decode
            # path; ``alloc_bytes`` should ~= kv_bytes_at_max_kv(config, max_kv).
            bounded_stats_fn = getattr(_dsv41_cache, "kv_bounded_stats", None)
            if callable(bounded_stats_fn):
                bstats = bounded_stats_fn()
                bstats["env"] = os.environ.get(KV_BOUNDED_ENV)
                bstats["maxkv_env"] = os.environ.get(KV_BOUNDED_MAXKV_ENV)
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
                    "(>1 growing == not preallocated-bounded); kv_inplace_writes_<lane> "
                    "== O(new-rows) decode path; formula_matches_alloc exact iff "
                    "kv_realloc_window == num_layers (no transient prefill grow)"
                )
                receipt["kv_bounded"] = bstats
        # W106 item 3: merge the budget-total derivation terms into every memory
        # block (memory_plan_source + the derived plan limit + each term), so the
        # receipt records how the plan compensated for the non-Metal requirements.
        _extra_keys = _memory_block_extra_keys(args)
        if isinstance(receipt.get("memory"), dict):
            receipt["memory"].update(_extra_keys)
        _dsp = receipt.get("dspark")
        if isinstance(_dsp, dict) and isinstance(_dsp.get("memory"), dict):
            _dsp["memory"].update(_extra_keys)
        return receipt
    finally:
        if runtime is not None:
            close = getattr(runtime, "close", None)
            if callable(close):
                close()


def _tokenizer(args, bench):
    from mlx_lm.utils import load_tokenizer

    return load_tokenizer(Path(args.model))


def _warm_repeat_pass(*, model, ops, mem_probe, prompt_ids, steps, cold_ids) -> dict:
    """Second prefill+decode of the SAME prompt in the same process.

    A fresh ``model.make_cache()`` resets the KV window and hands a fresh engram
    history clone, but the expert-bank LRU and any engram row cache stay warm from
    the cold pass, so this pass bounds the no-miss decode ceiling.  Greedy decode
    is deterministic, so the warm token ids must match the cold pass -- recorded
    (``token_ids_match`` + both sha256), never asserted, so a mismatch is reported
    instead of crashing the arm."""
    run = _generate(
        model=model, ops=ops, mem_probe=mem_probe,
        prompt_ids=prompt_ids, steps=steps,
    )
    warm_ids = run["generated"]
    cold = [int(t) for t in cold_ids]
    warm_sha = hashlib.sha256(json.dumps(warm_ids).encode()).hexdigest()
    cold_sha = hashlib.sha256(json.dumps(cold).encode()).hexdigest()
    return {
        "warm_ttft_s": run["ttft_s"],
        "warm_prefill_tok_s": (len(prompt_ids) / run["ttft_s"])
        if run["ttft_s"] > 0
        else None,
        "warm_decode_wall_s": run["decode_wall_s"],
        "warm_decode_tok_s": (steps / run["decode_wall_s"])
        if run["decode_wall_s"] > 0
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
    logits = model(ops.input([list(prompt_ids)]), cache=cache)
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
    _util_cm = util_sampler if util_sampler is not None else contextlib.nullcontext()
    stime.begin()
    with _util_cm:  # W90: macmon utilization over the fenced decode loop only
        for _ in range(int(steps)):
            with stime.frame():
                logits = model(ops.input([[token]]), cache=cache)
                with stime.stage("sample"):
                    token = ops.argmax_last(logits)
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
    logits = model(ops.input([list(prompt_ids)]), cache=cache)
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
    logits = model(ops.input([list(prompt_ids)]), cache=cache)
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
    bench = _load_bench_module()

    # W106 LOW-4: pre-flight the budget plan BEFORE anything else (no MLX, no model,
    # no --out needed), so a floor refusal happens before the GPU window opens.
    if getattr(args, "memory_plan_preflight", False):
        return _preflight_memory_plan(args, bench)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        return _run_dry(args, bench)

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
    if len(receipts) >= 2:
        base = receipts[0]
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
                # W97 (review item 7): a rounding-class attention lever (the n=1 core
                # compile / K29 tile reduction reassociates the fp32 softmax) can flip
                # a greedy near-tie -- that is EXPECTED, not a broken exact lever, so
                # it must not read as FAIL.  Every other arm keeps the FAIL.  Read the
                # machine label off the receipt (fall back to deriving it, so an older
                # receipt without the field still classifies).
                cand_rc = cand.get("rounding_class")
                if cand_rc is None:
                    cand_rc = _is_rounding_class(cand["arm"])
                if cand_rc:
                    # Name the reason keys so the label is machine-checkable, not a
                    # bare "expected".
                    keys = cand.get("rounding_class_keys") or _rounding_class_keys(
                        cand["arm"]
                    )
                    keys_str = ", ".join(keys) if keys else "?"
                    # When the receipt already classified a divergence (the dspark
                    # decode path records one), add the first divergence index and the
                    # control top-2 logit margin there -- a tiny margin corroborates a
                    # rounding tie ([[dsv41-inexact-ok-if-tie-flips]]).
                    div = (cand.get("dspark") or {}).get("divergence")
                    div_str = ""
                    if isinstance(div, dict) and div.get("divergence_index") is not None:
                        div_str = (
                            f"; first divergence @ {div['divergence_index']}, "
                            f"control top-2 logit margin {_fmt(div.get('ar_top2_margin'))} "
                            f"(cand {_fmt(div.get('dspark_top2_margin'))})"
                        )
                    print(
                        f"[ab] {cand['arm']}: token-id sha differs -- expected "
                        f"(rounding-class: {keys_str}){div_str}"
                    )
                else:
                    print(
                        f"[ab] FAIL: {cand['arm']} changed the decoded tokens "
                        "(the lever must be a pure execution reorder)"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
