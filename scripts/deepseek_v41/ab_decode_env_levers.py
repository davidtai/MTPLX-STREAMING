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
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

DEFAULT_MODEL = Path(
    "~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4"
).expanduser()
GIB = 1024 ** 3
DEFAULT_BOS_ID = 0
OVERLAP_ENV = "MTPLX_DSV41_SHARED_OVERLAP"
PROBE_ENV = "MTPLX_ROUTE_STAGE_PROBE"
STAGE_TIMING_ENV = "MTPLX_DSV41_STAGE_TIMING"
BARRIER_STAGE = "hot.eval_indices"

LAYER_MAJOR_ENV = "MTPLX_DSV41_PREFILL_LAYER_MAJOR"
SINKHORN_METAL_ENV = "MTPLX_DSV41_SINKHORN_METAL"   # K3, merged @ 8982b93c9
HC_COMPILE_ENV = "MTPLX_DSV41_HC_COMPILE"           # K4, landing
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
    KV_CHUNK_GROW_ENV,
)


def _preset(
    *, overlap=None, layer_major=None, sinkhorn=None, hc=None, fastpath=None,
    submit=None, attn=None, win_memo=None, device_route=None,
    verify_single=None,
    prefill_dense=None, prefill_dense_min_rows=None, prefill_dense_batch=None,
    prefill_dense_matmul_dtype=None,
    head=None, score_dtype=None, score_key_chunk=None, score_path=None,
    layout_fix=None, down_k_pad=None, selected_keys=None,
    softmax_kernel=None, decode_attn_kernel=None, kv_chunk_grow=None,
) -> dict:
    """A preset that pins EVERY lever key (None = force-unset). ``head`` takes a
    codec value ("bf16"/"mxfp8"/"q8"), ``prefill_dense_matmul_dtype`` takes
    "f32"/"bf16", ``prefill_dense_min_rows`` / ``_batch`` an integer string (None =
    use the code default), ``score_dtype`` a "bf16" value (W50/K25, prefill score
    matmul dtype), ``score_key_chunk`` a positive-int string (W50/K25 split-K chunk
    width), ``score_path`` a "lean" value (W50 f32 pass-cut one-shot),
    ``softmax_kernel`` a "1"/None boolean (W58/K28, the fused mask+sink+softmax
    Metal kernel), ``decode_attn_kernel`` a "1"/None boolean (W60/K29, the fused
    decode/verify MLA attention Metal kernel), ``kv_chunk_grow`` a "1"/None boolean
    (W73/K32, the chunk-grown KV append backing); the rest a "1"/None boolean."""
    return {
        OVERLAP_ENV: overlap,
        LAYER_MAJOR_ENV: layer_major,
        SINKHORN_METAL_ENV: sinkhorn,
        HC_COMPILE_ENV: hc,
        SWITCH_FASTPATH_ENV: fastpath,
        SWITCH_SUBMIT_ENV: submit,
        ATTN_COMPILE_ENV: attn,
        ATTN_WIN_MEMO_ENV: win_memo,
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
        KV_CHUNK_GROW_ENV: kv_chunk_grow,
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
    "device_route": _preset(device_route="1"),              # W44 K24: barrier-free all-hit
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
}


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
        "attn_compile, attn_win_memo, device_route, prefill_dense_experts, "
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
    # GPU-window defaults (agent booted out -> ~82 GiB planner budget).
    p.add_argument("--memory-limit-gib", type=float, default=82.0)
    p.add_argument("--expert-cache-limit-gib", type=float, default=None)
    p.add_argument(
        "--apply-memory-cap", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--slot-layout", default="component-banks")
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
        "overlap_env": os.environ.get(OVERLAP_ENV),
        "arm_env": _arm_env_snapshot(),
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


def _load_model(args, bench, mx):
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
    # --decode-mode dspark (or --with-mtp on AR) loads with the DSpark head
    # (with_mtp=True) and reprices the MTP residents against the expert cache so
    # the plan still fits.
    want_head = (getattr(args, "decode_mode", "ar") == "dspark") or bool(
        getattr(args, "with_mtp", None)
    )
    with_mtp, memory_limit_bytes, cache_limit = dspark_bench_loader_overrides(
        want_dspark=want_head,
        memory_limit_bytes=int(args.memory_limit_gib * GIB),
        expert_cache_limit_bytes=cache_limit,
        reprice=bool(getattr(args, "reprice", True)),
    )
    resident = load_deepseek_v41_streaming(
        args.model,
        memory_limit_bytes=memory_limit_bytes,
        max_live_kv_tokens=int(max_kv),
        admit=args.admit,
        admission_receipt=admission_receipt,
        expert_cache_limit_bytes=cache_limit,
        apply_memory_cap=args.apply_memory_cap,
        slot_layout=args.slot_layout,
        cache_scope="layer",
        island_layers=(),
        verify_record_hashes=args.verify_record_hashes,
        with_mtp=with_mtp,
    )
    if with_mtp and getattr(resident.model, "mtp", None) is None:
        raise RuntimeError(
            "--decode-mode dspark needs the DSpark MTP head, but the loaded model "
            "has none (with_mtp did not build it -- the artifact ships no mtp.* "
            "residents, or the config declares no MTP stages). Load a DSpark "
            "artifact or drop --decode-mode dspark."
        )
    return resident


def _generate(*, model, ops, mem_probe, prompt_ids, steps):
    """Greedy prefill + ``steps`` decode; captures the decoded token ids."""
    mem_probe.reset_peak()
    t0 = time.perf_counter()
    cache = model.make_cache()
    logits = model(ops.input([list(prompt_ids)]), cache=cache)
    ops.sync(logits)
    ttft_s = time.perf_counter() - t0
    token = ops.argmax_last(logits)
    generated = [token]

    decode_start = time.perf_counter()
    for _ in range(int(steps)):
        logits = model(ops.input([[token]]), cache=cache)
        ops.sync(logits)
        token = ops.argmax_last(logits)
        generated.append(token)
    decode_wall_s = time.perf_counter() - decode_start
    return {
        "generated": [int(t) for t in generated],
        "ttft_s": ttft_s,
        "decode_wall_s": decode_wall_s,
        "peak_gb": mem_probe.peak_bytes() / GIB,
    }


def _generate_dspark(*, model, mx, mem_probe, prompt_ids, steps, depth, stage_timing=False):
    """Greedy DSpark-DIRECT prefill + ``steps`` decode; captures tokens and the
    per-cycle accept + phase-timing statistics.  Total tokens == steps + 1 to match
    ``_generate`` (prefill token + ``steps`` decode tokens).  When ``stage_timing``
    the W37 probe is armed around the decode cycles so the receipt also carries the
    VERIFY forward's internal model stages (attention, moe.routed_switch breakdown,
    which reveals whether rows>1 took the prefill routing phase)."""
    from mtplx.models.deepseek_v41_dspark_decode import (
        DSparkDecodeStats,
        dspark_generate,
    )
    from mtplx.sampling import SamplerConfig

    mem_probe.reset_peak()
    stats = DSparkDecodeStats()
    stime = None
    if stage_timing:
        from mtplx.models import deepseek_v41_stage_timing as stime

        stime.begin()
    t0 = time.perf_counter()
    toks = dspark_generate(
        model,
        [int(t) for t in prompt_ids],
        max_tokens=int(steps) + 1,
        sampler=SamplerConfig(temperature=0.0),
        seed=0,
        speculative_depth=int(depth),
        stats=stats,
    )
    wall = time.perf_counter() - t0
    report = None
    if stime is not None:
        report = model.stage_timing_report()
        stime.end()
    out = {
        "generated": [int(t) for t in toks],
        "decode_wall_s": wall,
        "peak_gb": mem_probe.peak_bytes() / GIB,
        "stats": stats.to_dict(),
    }
    if report is not None:
        out["verify_stage_timing"] = report
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


def _run_arm(args, arm, bench, mx) -> dict:
    _apply_arm_env(arm)
    if getattr(args, "decode_mode", "ar") == "dspark":
        # Arm K29 (fused decode/verify attention, b*s<=8) + K30 (selected keys) for
        # the WHOLE arm so both the AR reference (_generate) and the dspark verify
        # use the decode attention branch consistently -- they are greedy-identical
        # to the eager path, so byte_identical_vs_ar holds only if both share the
        # setting. setdefault respects an arm that set them explicitly;
        # MTPLX_DSV41_DSPARK_DECODE_KERNELS=0 opts out.
        from mtplx.models.deepseek_v41_dspark_decode import (
            _DSPARK_DECODE_KERNEL_ENVS,
            _dspark_decode_kernels_disabled,
        )

        if not _dspark_decode_kernels_disabled():
            for _k in _DSPARK_DECODE_KERNEL_ENVS:
                os.environ.setdefault(_k, "1")
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
    try:
        ops = bench._MLXOps(mx)
        mem_probe = bench._MLXMemProbe(mx)
        run = _generate(
            model=model,
            ops=ops,
            mem_probe=mem_probe,
            prompt_ids=prompt_ids,
            steps=args.decode_tokens,
        )
        ids = run["generated"]
        receipt = {
            "arm": arm,
            "overlap_env": os.environ.get(OVERLAP_ENV),
            "arm_env": _arm_env_snapshot(),
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
            "peak_gb": run["peak_gb"],
            "token_ids_sha256": hashlib.sha256(
                json.dumps(ids).encode()
            ).hexdigest(),
            "first_token_ids": ids[:16],
            "overlap_telemetry": _overlap_telemetry(runtime)
            if runtime is not None
            else None,
            # W60/K29 fused-decode-attention engagement (calls/rows/split_calls/
            # fallbacks over the whole arm); calls 0 on a decode_attn_kernel arm
            # means the kernel never ran (all eager) rather than ran-and-was-slow.
            "decode_attn_kernel_engagement": (
                _k29.engagement() if _k29 is not None else None
            ),
        }
        if getattr(args, "decode_mode", "ar") == "dspark":
            # DSpark-DIRECT lane: greedy speculative decode MUST reproduce the AR
            # ids byte-for-byte (verify is authoritative); assert it in the
            # receipt and record tokens/cycle + accept-by-depth.
            dsp = _generate_dspark(
                model=model, mx=mx, mem_probe=mem_probe,
                prompt_ids=prompt_ids, steps=args.decode_tokens,
                depth=args.dspark_depth,
                stage_timing=bool(getattr(args, "stage_timing", False)),
            )
            dsp_ids = dsp["generated"]
            byte_identical = dsp_ids == ids
            st = dsp["stats"]
            receipt["dspark"] = {
                "depth": int(args.dspark_depth),
                "byte_identical_vs_ar": byte_identical,
                "decode_wall_s": dsp["decode_wall_s"],
                "decode_tok_s": (
                    (len(dsp_ids) / dsp["decode_wall_s"])
                    if dsp["decode_wall_s"] > 0
                    else None
                ),
                "peak_gb": dsp["peak_gb"],
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
            if dsp.get("verify_stage_timing") is not None:
                # W37 internal breakdown of the verify forward (attention +
                # moe.routed_switch): the census that shows the routing phase.
                receipt["dspark"]["verify_stage_timing"] = dsp["verify_stage_timing"]
            # Byte-identity is a hard correctness gate, not a soft metric: a
            # differing greedy sequence means the speculative lane is broken.
            if not byte_identical:
                first = next(
                    (i for i, (a, b) in enumerate(zip(dsp_ids, ids)) if a != b),
                    min(len(dsp_ids), len(ids)),
                )
                raise AssertionError(
                    "DSpark-DIRECT greedy decode diverged from AR at index "
                    f"{first} (arm {arm!r}); speculative lane is not lossless"
                )
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
            receipt["stage_timing"] = _stage_timing_pass(
                model=model, ops=ops, prompt_ids=prompt_ids, steps=steps,
            )
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
        f"(KV_CHUNK_GROW cuts the two append rows to ~O(1))"
    )


def _stage_timing_pass(*, model, ops, prompt_ids, steps) -> dict:
    """W37 fenced decode pass -> ``model.stage_timing_report()``.

    Prefill once (probe unarmed -> untouched), then arm the stage-timing session
    around the decode loop only, wrapping each step in a ``frame`` and the
    argmax/host round-trip in a ``sample`` stage.  The route-stage probe counters
    (when ``MTPLX_ROUTE_STAGE_PROBE`` is armed) are cleared before the loop so the
    merged ``route_stage`` census reflects this window, not the prefill + prior
    passes.  Fences inflate absolute time, so nothing here feeds the reported
    tok/s."""
    from mtplx.models import deepseek_v41_stage_timing as stime

    cache = model.make_cache()
    logits = model(ops.input([list(prompt_ids)]), cache=cache)
    ops.sync(logits)
    token = ops.argmax_last(logits)
    # Reset the route probe window (best-effort; its snapshot is cumulative).
    try:
        from mtplx import expert_route_probe as route_probe

        if getattr(route_probe, "ENABLED", False):
            route_probe._SUMS.clear()
            route_probe._COUNTS.clear()
    except Exception:
        pass
    stime.begin()
    for _ in range(int(steps)):
        with stime.frame():
            logits = model(ops.input([[token]]), cache=cache)
            with stime.stage("sample"):
                token = ops.argmax_last(logits)
    report = model.stage_timing_report()
    stime.end()
    return report if report is not None else {"enabled": False}


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
        receipt = _run_arm(args, arm, bench, mx)
        receipts.append(receipt)
        with args.out.open("a") as fh:
            fh.write(json.dumps(receipt) + "\n")
        print(
            f"[ab]   decode_tok_s={receipt['decode_tok_s']} "
            f"peak_gb={receipt['peak_gb']:.2f} sha={receipt['token_ids_sha256'][:12]}"
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
                print(
                    f"[ab] FAIL: {cand['arm']} changed the decoded tokens "
                    "(the lever must be a pure execution reorder)"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
