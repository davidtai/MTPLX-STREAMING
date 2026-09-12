#!/usr/bin/env python3
"""W78 -- Metal decode-attention op bisect for DeepSeek-V4.1-Flash.

W76 bisected the 16K decode attention on the **CPU** backend and found no per-op
O(T) inside "attention proper" (docs/deepseek-v41/W76_DECODE_ATTN_16K.md).  But
the CPU double does not reproduce Metal kernel behaviour -- ``mx.take`` / gather
over a large source, non-contiguous cache views forcing copies, argsort/topk
kernels, compiled-shape specialisation, the ATTN_WIN_MEMO memo, and the K30
selected-key gather from the T-row cache all behave differently on the GPU.  This
script reconstructs the **real** DeepSeek-V4.1 decode-attention path at real dims
(64 heads x 512, window 128, index_topk 512, the four CSA mode classes) with
RANDOM weights and a RANDOM pre-filled cache -- no model load, no artifact, no
expert bank -- and times, on the GPU, per decode step (M=1) with ``mx.eval``
fences:

  (a) the whole ``Attention._attend`` per CSA mode, and
  (b) each sub-op peeled: qkv projections, window cache append, compressed KV /
      index-key append (``_publish_compressed``), the indexer select + K30
      ``_mask_to_topk_idx`` argsort, the selected-key gather (the ``mx.take`` from
      the T-row cache), the score+softmax+PV, and the o-LoRA out projection.

It prints a ms/step table per op per T and the 16K/1K ratio, and writes a JSON
receipt (``--out``).  The peeled sub-ops call the **production** methods directly
(``Attention._attend`` / ``_sparse_attend_selected`` / ``_publish_compressed`` /
``Indexer.select`` / ``_mask_to_topk_idx`` / ``_window_selected_idx`` /
``_gather_rows``) -- no re-implementation -- so the numbers are the served path.

Levers (so the GPU window can attribute a cost):
  * ``--no-selected-keys``  -- run the masked-full path (score over the whole T,
    ``KV.astype(f32)`` over the whole cache) instead of the K30 selected-key gather.
  * ``--no-win-memo``       -- turn off MTPLX_DSV41_ATTN_WIN_MEMO.
  * ``--compare-no-compile``-- turn off MTPLX_DSV41_ATTN_COMPILE (eager qkv/out).
  * ``--ballast-gib N``      -- hold N GiB of resident Metal buffers for the whole
    run (reproduce the loaded model's 60-88 GB allocator/residency pressure that
    the empty microbench lacks -- window 31 measured every op flat in T here but
    3-7x slower inside the loaded process).
  * ``--ballast-churn``      -- also alloc+free a ~16 MB transient per step between
    ops (mimic the per-token [1,T,512] concat-output churn).

``--in-model`` (window-32 follow-up) is a separate mode: it measures the attention
ops IN SITU on the REAL loaded model with a live cache that grows one token per
step (reusing the ab_decode_env_levers loader + decode stage-timing census, not a
re-implementation), under three pipeline conditions -- (1) full model, (2) the
routed expert switch stubbed to a no-op (shared kept), (3) attention stubbed --
to test whether attention alone is ~2 ms or the census 6.6 ms/layer in situ, and
whether the async expert-gather drain is what lands in the attention fence.  It is
a GPU-window mode (``--in-model --gpu``); ``--in-model --tiny`` validates the stub
plumbing on a fake CPU model.

``--in-model --unfenced`` (window-94) is the UNFENCED attribution.  The fenced
per-stage census above brackets each stage with ``mx.eval``, so every bracket's
wall is a GPU pipeline drain + refill -- ~8 ms per attention bracket in EVERY config
(window 36/37) while the same kernels cost ~2 ms isolated: those absolute per-stage
numbers are LATENCY, not compute (ratios only).  This mode drops all per-stage
fences and measures the WHOLE-TOKEN frame wall (the only ``mx.eval`` per token is
the production loop's own sampler / next-token id) with components stubbed out, so
the delta full - stubbed is a component's TRUE in-situ cost -- including whatever
latency it causes (the DVFS downclock in the sync gap, the SSD wait) but excluding
probe latency.  Five passes: (1) full; (2) switch stubbed with the routing barrier
KEPT; (3) attention stubbed; (4) switch stubbed with the barrier REMOVED; (5)
attention AND switch stubbed (the small-per-layer-stages floor).  Derived: switch
total (1)-(4) = SSD-bound (1)-(2) + sync/barrier (2)-(4); attention (1)-(3); small
stages (5); sum-of-parts vs full + residual.  ``--tiny`` validates all five passes
on CPU.

Exact GPU-window command for the 16,384-token cell (60 GiB plan, --max-kv 17408,
cell16k_ring arm), run through the flock (scripts/deepseek_v41/gpu_window.sh) like
every Metal exec on this box.  With PY the venv python3 and MODEL the streaming
artifact, the invocation is:

    nice -n 19 $PY scripts/deepseek_v41/metal_decode_attn_bisect.py
      --in-model --unfenced --gpu --model $MODEL --arms cell16k_ring
      --context-tokens 16384 --memory-limit-gib 60 --max-kv 17408
      --in-model-steps 64 --utilization --out $OUT/in-model-16k-unfenced.json

(No --prompt-ids-file: the standard-cell builder resolves the same 16,384-token
prompt the ab cell16k_ring reference uses, so pass (1)'s token sha is comparable to
that AR receipt run at --decode-tokens 64.  The fully-escaped gpu_window.sh wrapper
is in docs/deepseek-v41/W94_UNFENCED_ATTRIBUTION.md.)

Default device is **CPU** (so an accidental worker run never touches the Metal
GPU during a benchmark window); pass ``--gpu`` in the exclusive GPU window.  The
``--tiny`` mode (tiny dims, CPU, T in {256,1024}) is the unit-test path that proves
the script executes and the peeled ops sum to within 20% of the whole.

Budget on the GPU window: <= 3 min total, <= 6 GB (one layer per mode, built and
freed sequentially; a 16K real-dim cache is ~120 MB/layer).
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Optional


# --- arm the cell16k attention-relevant env BEFORE importing the model module --
# ``_ATTN_COMPILE`` / ``_ATTN_WIN_MEMO`` are frozen as module globals at import;
# the rest (SELECTED_KEYS / SELECT_FENCE / KV_CHUNK_GROW) are read at use / at
# cache construction.  Set the cell16k defaults here; per-run toggles below
# override the globals or the (read-at-use) env after import.
_CELL16K_ENV = {
    "MTPLX_DSV41_SELECTED_KEYS": "1",
    "MTPLX_DSV41_ATTN_COMPILE": "1",
    "MTPLX_DSV41_ATTN_WIN_MEMO": "1",
    "MTPLX_DSV41_KV_CHUNK_GROW": "1",
    "MTPLX_DSV41_SELECT_FENCE": "1",
}
for _k, _v in _CELL16K_ENV.items():
    os.environ.setdefault(_k, _v)

import mlx.core as mx  # noqa: E402

from mtplx.models import deepseek_v41 as dsv41  # noqa: E402
from mtplx.models.deepseek_v41 import (  # noqa: E402
    Attention,
    ModelArgs,
    MODE_FULL,
    MODE_REINDEX,
    MODE_REUSE,
    MODE_SWA_ONLY,
    _cos_sin,
    _gather_rows,
    _lin_arrays,
    _mask_to_topk_idx,
    _resolve_selected_keys,
    _rmsnorm,
    _rope_last,
)
from mtplx.models.deepseek_v41_cache import LayerAttentionCache, SharedAttentionRuntime  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# One representative backbone layer per CSA mode, from the released 40-layer
# config (docs/deepseek-v41/PORT_PLAN.md §0; verified against the artifact
# config.json: compress_ratios 2 for layers 2-19 / 1 for 20-39, kv_source
# [2,8,14,20], index_source [2,8,14,20,24,28,32,36], candidate_source 20):
#   swa_only : layer 0  (compress_ratio 0, no compressor/indexer)
#   full     : layer 2  (ratio 2, kv_source + index_source -- owns compress_kv,
#                        index_k AND its own compressor frontier)
#   reindex  : layer 24 (ratio 1, index_source only -- owns indexer queries,
#                        reuses layer 20's ratio-1 compressed KV: n_comp == T)
#   reuse    : layer 3  (ratio 2, reuses layer 2's ratio-2 KV + selection:
#                        n_comp == T//2)
def _mode_layer(args: ModelArgs, mode: str) -> int:
    """The FIRST backbone layer of ``mode`` in this config -- the representative
    layer we build.  For the released config that is swa=0, full=2, reindex=24,
    reuse=3; for the tiny config swa=0, full=2, reindex=6, reuse=3.  (The reindex
    rep is always in the ratio-1 region and reuse in the ratio-2 region for both
    configs, which fixes the source ratio in ``_SOURCE_RATIO``.)"""
    for lid, m in enumerate(args.layer_modes):
        if m == mode:
            return lid
    raise ValueError(f"no layer with mode {mode!r} in this config")


def _mode_layer_map(args: ModelArgs) -> Dict[str, int]:
    return {m: _mode_layer(args, m) for m in _MODES}


# The compress ratio of the layer that PUBLISHED the compressed KV a reuse /
# reindex layer reads (its "source"): reuse(3) reads full(2)=ratio2,
# reindex(24) reads full(20)=ratio1.  Sets the dtype (ratio>1 pools in f32,
# ratio==1 is a bf16 per-token projection) and n_comp (T//ratio) of the shared
# store we pre-fill.
_SOURCE_RATIO = {MODE_REUSE: 2, MODE_REINDEX: 1}

_REAL_COMPRESS_RATIOS = (
    [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]  # 40 backbone + 3 MTP (unused)
)
_REAL_KV_SOURCE = [2, 8, 14, 20]
_REAL_INDEX_SOURCE = [2, 8, 14, 20, 24, 28, 32, 36]

_MODES = [MODE_SWA_ONLY, MODE_FULL, MODE_REINDEX, MODE_REUSE]


def real_args() -> ModelArgs:
    """The released-shape config (values from the artifact config.json).  Only the
    attention/indexer/compressor fields matter here; the MoE/HC fields are left at
    their defaults and never exercised (we only build one ``Attention`` layer)."""
    return ModelArgs(
        num_hidden_layers=40,
        hidden_size=5120,
        num_attention_heads=64,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=1280,
        o_lora_rank=1024,
        o_groups=8,
        sliding_window=128,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
        compress_ratios=list(_REAL_COMPRESS_RATIOS),
        compress_rope_theta=160000.0,
        kv_source_layer_ids=list(_REAL_KV_SOURCE),
        index_source_layer_ids=list(_REAL_INDEX_SOURCE),
        candidate_source_layer_id=20,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
    )


def tiny_args() -> ModelArgs:
    """A tiny-dims config that still derives all four CSA modes (same source-id
    shape as the existing DSV4.1 tests): swa=0, full=2, reindex=6, reuse=3."""
    # Dims are kept small (fast, well under the memory guard) but large enough that
    # each fenced op does real work -- so the per-fence CPU sync overhead does not
    # inflate the peeled sum past the 20% band the test checks.
    return ModelArgs(
        num_hidden_layers=8,
        hidden_size=512,
        num_attention_heads=16,
        head_dim=128,
        qk_rope_head_dim=16,
        q_lora_rank=256,
        o_lora_rank=128,
        o_groups=4,
        sliding_window=32,
        index_n_heads=8,
        index_head_dim=32,
        index_topk=32,
        compress_ratios=[0, 0, 2, 2, 2, 1, 1, 1],
        compress_rope_theta=160000.0,
        kv_source_layer_ids=[2, 5],
        index_source_layer_ids=[2, 5, 6],
        candidate_source_layer_id=-1,
    )


# ---------------------------------------------------------------------------
# Build a random layer + pre-filled cache/shared for one mode at length T
# ---------------------------------------------------------------------------
def _build_layer(args: ModelArgs, mode: str) -> Attention:
    """A single ``Attention`` layer for ``mode`` with RANDOM weights, cast to
    bf16 so the activations/cache carry the served bf16 dtype (the model.py rmsnorm
    stores at the activation dtype), which is what makes the masked-full path's
    ``KV.astype(f32)`` a real O(T) cast and the selected path's per-key cast O(k)."""
    layer_id = _mode_layer(args, mode)
    attn = Attention(args, layer_id)
    assert attn.mode == mode, f"layer {layer_id} derived {attn.mode!r}, wanted {mode!r}"
    attn.set_dtype(mx.bfloat16)
    mx.eval(attn.parameters())
    return attn


def _prefill_full(attn: Attention, args: ModelArgs, shared: SharedAttentionRuntime,
                  cache: LayerAttentionCache, X: mx.array, positions_pre: mx.array) -> None:
    """kv_source (full) layer: run the real window projection + the real
    ``_publish_compressed`` over the T prefill tokens.  This fills the window store
    (bf16, T rows), the compressed KV / index-key stores (f32, T//ratio rows) AND
    the compressor frontier ``comp_state.raw_kv`` (f32, T rows) exactly as decode
    would have -- so the decode-step ``comp_state.push`` concatenate is over a real
    T-row frontier (an O(T) cost KV_CHUNK_GROW does NOT cover)."""
    qcos, qsin = _cos_sin(attn.inv_freq, positions_pre)
    kv_bulk = _rmsnorm(attn.wkv(X), attn.kv_norm_weight, attn.eps)
    kv_bulk = _rope_last(kv_bulk, qcos, qsin)
    cache.append_window(kv_bulk)
    attn._publish_compressed(X, positions_pre, cache, shared, qcos, qsin)
    frontier = cache.comp_state.raw_kv if cache.comp_state is not None else None
    _fence(cache.window, shared.compress_kv, shared.index_k, frontier)


def _prefill_window_only(attn: Attention, shared: SharedAttentionRuntime,
                         cache: LayerAttentionCache, X: mx.array,
                         positions_pre: mx.array) -> None:
    """swa_only / reindex / reuse: only the window store is this layer's own; the
    compressed stores (if any) are the source layer's, filled separately."""
    qcos, qsin = _cos_sin(attn.inv_freq, positions_pre)
    kv_bulk = _rmsnorm(attn.wkv(X), attn.kv_norm_weight, attn.eps)
    kv_bulk = _rope_last(kv_bulk, qcos, qsin)
    cache.append_window(kv_bulk)
    mx.eval(cache.window)


def _fill_shared_source(args: ModelArgs, mode: str, shared: SharedAttentionRuntime,
                        b: int, T: int) -> None:
    """Random-fill the shared compressed KV / index keys a reindex / reuse layer
    reads, at the dtype + count its source layer would have published (ratio>1 ->
    f32 pooled; ratio==1 -> bf16 per-token).  The values are irrelevant to timing
    (the layer only gathers/scores them); the dtype and n_comp == T//ratio are what
    set the gather/score/select memory traffic."""
    ratio = _SOURCE_RATIO[mode]
    n_comp = T // ratio
    dtype = mx.float32 if ratio > 1 else mx.bfloat16
    shared.compress_kv = (mx.random.normal((b, n_comp, args.head_dim)) * 0.02).astype(dtype)
    shared.index_k = (mx.random.normal((b, n_comp, args.index_head_dim)) * 0.02).astype(dtype)
    if mode == MODE_REUSE:
        # Reuse reads the index source's already-published selection: a boolean
        # topk mask (read by _compressed) and the K30 gather indices (read by the
        # gather).  Build a legitimate selection: the first k rows selected.
        k = min(args.index_topk, n_comp)
        idx = mx.broadcast_to(mx.arange(k, dtype=mx.int32)[None, None, :], (b, 1, k))
        shared.selected_idx = idx
        mask = mx.arange(n_comp)[None, None, :] < k
        shared.topk_mask = mx.broadcast_to(mask, (b, 1, n_comp))
    # reindex (24) is downstream of the candidate source (20): a candidate-block
    # mask is in force.  full (2) is upstream -> no candidate mask (leave None).
    if mode == MODE_REINDEX and args.candidate_source_layer_id >= 0 \
            and _mode_layer(args, mode) > args.candidate_source_layer_id:
        cand = mx.random.uniform(shape=(b, 1, n_comp)) < 0.5
        shared.candidates = cand
    mx.eval(*[a for a in (shared.compress_kv, shared.index_k, shared.selected_idx,
                          shared.topk_mask, shared.candidates) if a is not None])


def build_case(args: ModelArgs, attn: Attention, mode: str, T: int, b: int = 1):
    """Return (cache, shared, x_decode, positions_decode) for a decode step at
    absolute position T on a cache pre-filled to T tokens."""
    layer_id = _mode_layer(args, mode)
    cache = LayerAttentionCache(
        window_size=args.window_size,
        compress_ratio=args.compress_ratios[layer_id],
        is_kv_source=(layer_id in args.kv_source_layer_ids),
    )
    shared = SharedAttentionRuntime()
    positions_pre = mx.arange(T)
    X = (mx.random.normal((b, T, args.hidden_size)) * 0.02).astype(mx.bfloat16)
    if attn.is_kv_source:  # full
        _prefill_full(attn, args, shared, cache, X, positions_pre)
    else:
        _prefill_window_only(attn, shared, cache, X, positions_pre)
        if mode in _SOURCE_RATIO:  # reindex / reuse read a source's stores
            _fill_shared_source(args, mode, shared, b, T)
    x_dec = (mx.random.normal((b, 1, args.hidden_size)) * 0.02).astype(mx.bfloat16)
    positions_dec = mx.array([T])
    mx.eval(x_dec, positions_dec)
    return cache, shared, x_dec, positions_dec


# ---------------------------------------------------------------------------
# The peeled decode step -- mirrors Attention._attend, one production call per op
# ---------------------------------------------------------------------------
def _fence(*arrays) -> None:
    arrs = [a for a in arrays if isinstance(a, mx.array)]
    if arrs:
        mx.eval(arrs)


# --- W78 window-31 follow-up: allocator/residency-pressure ballast ------------
# The empty-process microbench measures every op flat in T, but inside the loaded
# model at 16K the same ops run 3-7x slower (census cache_append 1.18 ms/layer vs
# ~0.17 here; attn.reuse 6.6 vs ~2.0).  Hypothesis: per-token fresh T-sized buffers
# (the [1,T,512] concat outputs, ~16 MB x 40 layers) allocated inside a process
# already holding 60-88 GB of resident Metal buffers hit an allocator/residency
# slow path.  ``--ballast-gib N`` reproduces the resident set (N GiB held for the
# whole run); ``--ballast-churn`` reproduces the per-token alloc/free of a ~16 MB
# transient BETWEEN the timed ops, so the op's own fresh allocation pays whatever
# the churned + pressured allocator charges.  Both default OFF -> the default path
# is byte-for-byte the pre-follow-up microbench.
_BALLAST_CHUNK_BYTES = 256 * 1024 * 1024          # ~256 MB resident chunks
_BALLAST_CHUNK_ELEMS = _BALLAST_CHUNK_BYTES // 4  # f32
_CHURN_BYTES = 16 * 1024 * 1024                   # ~16 MB, the [1,T,512]@16K size
_CHURN_ELEMS = _CHURN_BYTES // 4                  # f32


def alloc_ballast(gib: float) -> list:
    """Allocate ``gib`` GiB of resident Metal buffers as ~256 MB f32 arrays, force
    them (``mx.eval``), and return the list so the caller holds the references for
    the whole run.  ``gib <= 0`` -> no allocation (default path unchanged)."""
    if gib <= 0:
        return []
    n_chunks = max(1, round(gib * 1024 / (_BALLAST_CHUNK_BYTES / (1024 * 1024))))
    chunks = []
    for _ in range(n_chunks):
        # a real materialised buffer (fill, not a lazy zeros special-case)
        chunks.append(mx.ones((_BALLAST_CHUNK_ELEMS,), dtype=mx.float32))
        mx.eval(chunks[-1])
    return chunks


def _churn(on: bool) -> None:
    """Allocate + force + free a ~16 MB transient -- the per-token concat-output
    churn.  Called BETWEEN the timed ops so its own time is never attributed to an
    op; it only changes the allocator state the next op's allocation sees."""
    if not on:
        return
    tmp = mx.ones((_CHURN_ELEMS,), dtype=mx.float32)
    mx.eval(tmp)
    del tmp


def _mem_gib(name: str):
    """Best-effort MLX memory counter in GiB (``mx.get_active_memory`` etc., with
    the older ``mx.metal.*`` fallback); ``None`` when unavailable (e.g. CPU)."""
    for mod in (mx, getattr(mx, "metal", None)):
        fn = getattr(mod, name, None) if mod is not None else None
        if fn is not None:
            try:
                return fn() / (1024 ** 3)
            except Exception:
                pass
    return None


def decode_step_peeled(attn: Attention, cache: LayerAttentionCache,
                       shared: SharedAttentionRuntime, x: mx.array,
                       positions: mx.array, use_selected: bool,
                       t: Dict[str, int], measure_gather: bool = True,
                       churn: bool = False):
    """One decode step, fencing after each production sub-op; accumulates ns into
    ``t``.  A faithful mirror of ``Attention._attend`` (selected + masked-full
    paths), calling the production methods -- so ``sum(peeled) ~= whole`` and each
    op's cost is the served cost.  With ``churn`` on, a ~16 MB transient is
    allocated + freed BETWEEN the timed ops (never inside a timed block)."""
    b, s, _ = x.shape
    H, hd = attn.n_heads, attn.head_dim
    mode = attn.mode
    qcos, qsin = _cos_sin(attn.inv_freq, positions)

    # -- qkv_proj (compiled tape when ATTN_COMPILE armed, else eager) --
    t0 = time.perf_counter_ns()
    if dsv41._attn_use_compile(b * s):
        q, qr, kv_new = dsv41._attn_qkv_prep(attn)(
            x, qcos, qsin, attn.q_norm_weight, attn.kv_norm_weight,
            *_lin_arrays(attn.wq_a), *_lin_arrays(attn.wq_b), *_lin_arrays(attn.wkv),
        )
    else:
        qr = _rmsnorm(attn.wq_a(x), attn.q_norm_weight, attn.eps)
        q = attn.wq_b(qr).reshape(b, s, H, hd)
        q = _rope_last(q, qcos, qsin)
        kv_new = _rmsnorm(attn.wkv(x), attn.kv_norm_weight, attn.eps)
        kv_new = _rope_last(kv_new, qcos, qsin)
    _fence(q, qr, kv_new)
    t["qkv_proj"] += time.perf_counter_ns() - t0
    _churn(churn)

    # -- cache_append (window) --
    t0 = time.perf_counter_ns()
    cache.append_window(kv_new)
    window_all = cache.window
    _fence(window_all)
    t["cache_append"] += time.perf_counter_ns() - t0
    _churn(churn)

    attend = None
    if not use_selected:
        # -- mask build (masked-full path only) --
        t0 = time.perf_counter_ns()
        attend = attn._window_attend(positions, window_all.shape[1], b, s, shared)
        _fence(attend)
        t["mask_build"] += time.perf_counter_ns() - t0
        _churn(churn)

    KV = window_all
    sel_compress_kv = None
    sel_comp_idx = None

    if attn.compress_ratio:
        ratio = attn.compress_ratio
        # -- compress_append (_publish_compressed on kv_source; else read shared) --
        t0 = time.perf_counter_ns()
        if attn.is_kv_source:
            attn._publish_compressed(x, positions, cache, shared, qcos, qsin)
        compress_kv = shared.compress_kv
        index_k = shared.index_k
        frontier = cache.comp_state.raw_kv if cache.comp_state is not None else None
        _fence(compress_kv, index_k, frontier)
        t["compress_append"] += time.perf_counter_ns() - t0
        _churn(churn)

        if compress_kv is not None:
            n_comp = compress_kv.shape[1]
            compress_lens = (positions + 1) // ratio
            # -- select (indexer.select + K30 argsort on index source; reuse reads) --
            t0 = time.perf_counter_ns()
            if attn.is_index_source:
                set_c = attn.is_candidate_source
                cand = None if set_c else shared.candidates
                mask, cand_out = attn.indexer.select(
                    x, qr, index_k, qcos, qsin, compress_lens, n_comp,
                    candidates=cand, set_candidates=set_c,
                    cand_topk=attn.candidate_topk_blocks,
                    cand_block=attn.candidate_block_size,
                )
                shared.topk_mask = mask
                if set_c:
                    shared.candidates = cand_out
                if _resolve_selected_keys():
                    shared.selected_idx = _mask_to_topk_idx(
                        mask, min(attn.indexer.index_topk, n_comp)
                    )
            else:
                mask = shared.topk_mask
            sel_arrs = [mask]
            if attn.is_index_source and shared.selected_idx is not None:
                sel_arrs.append(shared.selected_idx)
            _fence(*sel_arrs)
            t["select"] += time.perf_counter_ns() - t0
            _churn(churn)

            if use_selected:
                sel_compress_kv = compress_kv
                sel_comp_idx = shared.selected_idx
            else:
                KV = mx.concatenate([window_all, compress_kv], axis=1)
                attend = mx.concatenate([attend, mask], axis=-1)

    # -- gather (isolated attribution; NOT summed -- attend re-runs it) --
    if use_selected and measure_gather:
        t0 = time.perf_counter_ns()
        win_idx, win_valid = attn._window_selected_idx(positions, window_all.shape[1])
        win_idx = mx.broadcast_to(win_idx[None], (b, s, win_idx.shape[-1]))
        win_valid = mx.broadcast_to(win_valid[None], (b, s, win_valid.shape[-1]))
        kvg_win = _gather_rows(window_all, win_idx, win_valid)
        if sel_compress_kv is not None and sel_comp_idx is not None:
            comp_valid = sel_comp_idx >= 0
            kvg_cmp = _gather_rows(sel_compress_kv, sel_comp_idx, comp_valid)
            KVg = mx.concatenate([kvg_win, kvg_cmp], axis=2)
        else:
            KVg = kvg_win
        _fence(KVg)
        t["gather_iso"] += time.perf_counter_ns() - t0

    # -- attend (gather + score + softmax + PV; the production entry) --
    # W90: mirror Attention._attend exactly -- pass shared (so the W90 shared-gather
    # lever is exercised on the peeled path too) and the window drop_offset (0 on the
    # plain/grow backing, the ring's _drop under WINDOW_RING).
    t0 = time.perf_counter_ns()
    if use_selected:
        o = attn._sparse_attend_selected(
            q, window_all, sel_compress_kv, sel_comp_idx, positions,
            cache.window_drop_offset, shared=shared,
        )
    else:
        o = attn._sparse_attend(q, KV, attend)
    _fence(o)
    t["attend"] += time.perf_counter_ns() - t0
    _churn(churn)

    # -- out_proj (o-LoRA down + wo_b up; compiled tape or eager) --
    t0 = time.perf_counter_ns()
    if dsv41._attn_use_compile(b * s):
        out = dsv41._attn_out_prep(attn)(
            o, qcos, qsin, attn._o_lora_dense_weight(), *_lin_arrays(attn.wo_b)
        )
    else:
        o = _rope_last(o, qcos, qsin, inverse=True)
        o = o.reshape(b, s, attn.n_groups, -1)
        o = attn._o_lora_down(o)
        out = attn.wo_b(o.reshape(b, s, -1))
    _fence(out)
    t["out_proj"] += time.perf_counter_ns() - t0
    _churn(churn)
    return out


def decode_step_whole(attn: Attention, cache: LayerAttentionCache,
                      shared: SharedAttentionRuntime, x: mx.array,
                      positions: mx.array, churn: bool = False) -> int:
    """One decode step through the production ``Attention._attend`` (the served
    entry the decode loop calls), fenced once.  Returns elapsed ns.  Under
    ``churn`` a ~16 MB transient is allocated + freed once per step BEFORE the
    timed call (so the allocator is in the churned state ``_attend`` would meet
    inside the real per-layer loop)."""
    _churn(churn)
    t0 = time.perf_counter_ns()
    out = attn._attend(x, positions, cache, shared)
    mx.eval(out)
    return time.perf_counter_ns() - t0


# ---------------------------------------------------------------------------
# Measurement driver
# ---------------------------------------------------------------------------
# The sub-op keys, in _attend order.  gather_iso / score are attribution rows and
# are kept OUT of the peeled sum (attend already contains the gather + score).
_SUM_KEYS = ["qkv_proj", "cache_append", "mask_build", "compress_append", "select",
             "attend", "out_proj"]
_ALL_KEYS = ["qkv_proj", "cache_append", "mask_build", "compress_append", "select",
             "gather_iso", "score", "attend", "out_proj"]


def _reset_forward_shared_cache(shared: SharedAttentionRuntime) -> None:
    """W90 (review fix 5): drop the per-forward selected-compress gather cache so
    each isolated timed iteration re-issues the gather.  A real per-forward starts
    with a fresh ``SharedAttentionRuntime``; without this reset,
    ``MTPLX_DSV41_ATTN_SHAPE_STABLE`` would cache on iteration 1 and every later
    iteration would HIT -- timing a shared-gather-free step that never happens in
    situ (the Reuse mode would read artificially fast).  No-op when the lever is off
    (the attribute is never set)."""
    if getattr(shared, "_sel_cmp_kvg", None) is not None:
        shared._sel_cmp_kvg = None


def measure_mode_T(args: ModelArgs, attn: Attention, mode: str, T: int,
                   use_selected: bool, iters: int, warmup: int,
                   churn: bool = False) -> Dict[str, float]:
    """Time whole + peeled for one (mode, T).  whole and peeled use freshly
    pre-filled caches so each runs a genuine decode step (the step mutates the
    cache); returns ms/step per op + the whole.

    W90: each timed iteration resets the per-forward shared gather cache
    (:func:`_reset_forward_shared_cache`) so ``MTPLX_DSV41_ATTN_SHAPE_STABLE`` is
    measured as a genuine first-touch per step, not a cross-iteration cache hit."""
    # whole
    cache, shared, x, pos = build_case(args, attn, mode, T)
    for _ in range(warmup):
        _reset_forward_shared_cache(shared)
        decode_step_whole(attn, cache, shared, x, pos, churn=churn)
    whole_ns = 0
    for _ in range(iters):
        _reset_forward_shared_cache(shared)
        whole_ns += decode_step_whole(attn, cache, shared, x, pos, churn=churn)
    whole_ms = whole_ns / 1e6 / iters
    del cache, shared, x, pos

    # peeled
    cache, shared, x, pos = build_case(args, attn, mode, T)
    twarm: Dict[str, int] = {k: 0 for k in _ALL_KEYS}
    for _ in range(warmup):
        _reset_forward_shared_cache(shared)
        decode_step_peeled(attn, cache, shared, x, pos, use_selected, twarm, churn=churn)
    t: Dict[str, int] = {k: 0 for k in _ALL_KEYS}
    for _ in range(iters):
        _reset_forward_shared_cache(shared)
        decode_step_peeled(attn, cache, shared, x, pos, use_selected, t, churn=churn)
    del cache, shared, x, pos

    out: Dict[str, float] = {"whole": whole_ms}
    for k in _SUM_KEYS:
        out[k] = t[k] / 1e6 / iters
    out["gather_iso"] = t["gather_iso"] / 1e6 / iters
    # derived score = attend - gather (both selected-path); floor at 0
    out["score"] = max(0.0, out["attend"] - out["gather_iso"]) if use_selected else 0.0
    out["peeled_sum"] = sum(out[k] for k in _SUM_KEYS)
    return out


# ---------------------------------------------------------------------------
# W102 --fence-per-layer: the isolated 40-layer pipeline vs the per-layer barrier
# ---------------------------------------------------------------------------
# The isolated microbench above times ONE representative layer per mode fenced per
# step (~2 ms/layer), while the in-situ model measures ~7.3 ms/layer for the SAME
# attention (docs/deepseek-v41 W90/W97): in the served model every backbone layer
# ends in a routing barrier -- ``mx.eval(indices)`` + ``indices.reshape(-1).tolist()``
# in the streamed expert switch (mtplx/expert_runtime.py: the ~40-syncs/token route
# host-readback) -- so the host-encode of each layer's ~69 tiny attention kernels
# cannot overlap GPU execution.  Two explanations of the ~5 ms/layer gap are open:
# (H1) GPU-side per-kernel overhead of the tiny kernels; (H2) exposed host-encode
# because of the per-layer barrier.  This probe decides it WITHOUT the model: it runs
# ALL backbone layers' production ``Attention._attend`` back-to-back per decode step
# under two conditions differing ONLY in the sync discipline --
#   * unfenced (pipelined): one ``mx.eval`` at the end of the step (all layers'
#     kernels encoded back-to-back, GPU pipelines them, one host drain) -- the way
#     the isolated bench reads ~2 ms/layer; and
#   * fenced: after EACH layer, the served routing barrier (:func:`_fence_layer_output`)
#     so the host blocks on that layer before it can encode the next -- the ~40
#     syncs/token the served path pays.
# Same layers, same ops, same shapes, same count -- the fence changes only WHEN the
# host blocks, never any value (so both loops are bit-identical; the test asserts it).
# If fenced >> unfenced -> H2 (host-encode exposure); if fenced ~= unfenced -> H1.
_FENCE_KINDS = ("eval", "eval_tolist")
_FENCE_READBACK_ELEMS = 8          # route indices are [n, top_k]; a tiny dependent read
_FENCE_DEFAULT_STEPS = 64
_FENCE_DEFAULT_WARMUP = 8
_FENCE_SEED = 1234                 # seed both builds identically -> bit-identical inputs


def _backbone_modes(args: ModelArgs) -> List[str]:
    """The per-position CSA mode of every backbone layer (``layer_modes`` trimmed to
    ``num_hidden_layers`` -- the 3 MTP layers are not decoded here)."""
    return list(args.layer_modes)[: args.num_hidden_layers]


def _build_fence_pipeline(args: ModelArgs, T: int, *, b: int = 1,
                          seed: int = _FENCE_SEED):
    """Build one (attn, cache, shared, x, positions) per backbone layer at decode
    position ``T`` -- the full-depth stack the pipeline sweeps per step.  Each layer
    is its mode's representative (built by :func:`_build_layer` / :func:`build_case`,
    so attn and cache always agree) with its own weights + pre-filled cache, i.e. a
    genuine per-layer attention forward at real decode geometry.  ``seed`` seeds the
    MLX RNG so two builds with the same seed draw byte-identical weights + inputs
    (every random array here flows through ``mx.random``), which is what lets the
    fenced and unfenced loops be compared for bit-identity."""
    mx.random.seed(int(seed))
    pipeline = []
    for mode in _backbone_modes(args):
        attn = _build_layer(args, mode)
        cache, shared, x, pos = build_case(args, attn, mode, T, b=b)
        pipeline.append((attn, cache, shared, x, pos))
    return pipeline


def _fence_layer_output(out: mx.array, fence_kind: str) -> None:
    """Force a full host sync on one layer's attention output the way the served
    routing barrier does.  The streamed expert switch reads its route before it can
    stream -- ``mx.eval(indices)`` then ``indices.reshape(-1).tolist()``
    (mtplx/expert_runtime.py) -- one device->host round-trip per backbone layer per
    token.  ``eval`` reproduces the eval-only barrier (what ``_ZeroSwitch`` keeps);
    ``eval_tolist`` adds the tiny dependent ``.tolist()`` readback so the host is
    blocked the same way the route read blocks it.  NEVER mutates ``out`` (a read of
    a small slice), so the fenced loop stays bit-identical to the unfenced one."""
    mx.eval(out)
    if fence_kind == "eval_tolist":
        k = _FENCE_READBACK_ELEMS if out.size >= _FENCE_READBACK_ELEMS else int(out.size)
        # tiny dependent host readback -- mirrors indices.reshape(-1).tolist()
        _ = out.reshape(-1)[:k].tolist()


def _fence_step(pipeline, *, fence_kind: str, fenced: bool) -> list:
    """One decode step through every layer's production ``Attention._attend``.  When
    ``fenced``, a per-layer host sync (:func:`_fence_layer_output`) follows each layer
    so its host-encode cannot overlap the next; otherwise NO per-layer sync and a
    single ``mx.eval`` over all outputs at the step end (the pipelined path).  Either
    way every layer runs the identical ops; returns the per-layer outputs (the fenced
    path has already forced them, the pipelined path forces them in the closing eval).
    Resets each layer's per-forward shared gather cache first, exactly as
    :func:`measure_mode_T` does, so ``MTPLX_DSV41_ATTN_SHAPE_STABLE`` is a genuine
    first-touch per step."""
    outs = []
    for (attn, cache, shared, x, pos) in pipeline:
        _reset_forward_shared_cache(shared)
        out = attn._attend(x, pos, cache, shared)
        if fenced:
            _fence_layer_output(out, fence_kind)
        outs.append(out)
    if not fenced:
        mx.eval(*outs)
    return outs


def _utilization_sampler(interval_ms: int):
    """A macmon :class:`UtilizationSampler` from the sibling ``util_macmon.py``
    (CPU-safe, stdlib-only; graceful no-op if macmon is absent), or ``None`` if the
    module cannot be loaded.  Loaded by file path so it works under the test's
    spec-import of this script too."""
    try:
        from pathlib import Path
        import importlib.util as _il
        path = Path(__file__).resolve().parent / "util_macmon.py"
        spec = _il.spec_from_file_location("_w102_util_macmon", path)
        if spec is None or spec.loader is None:
            return None
        mod = _il.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.UtilizationSampler(interval_ms=int(interval_ms))
    except Exception:
        return None


def _run_fence_loop(pipeline, *, steps: int, warmup: int, fence_kind: str,
                    fenced: bool, util_on: bool = False,
                    util_interval_ms: int = 2000):
    """Run ``warmup`` (untimed) + ``steps`` (timed) pipeline sweeps and return
    ``(ms_per_layer, mean_ms_per_step, step_ms, last_outs, util_summary)``.  The
    per-step wall is ``perf_counter_ns`` around the whole sweep (all layers + the
    sync discipline); warmup is excluded from every number.  ``last_outs`` is the
    final step's per-layer outputs (kept for the bit-identity check)."""
    n_layers = len(pipeline)
    for _ in range(int(warmup)):
        _fence_step(pipeline, fence_kind=fence_kind, fenced=fenced)
    sampler = _utilization_sampler(util_interval_ms) if util_on else None
    step_ns: List[int] = []
    last_outs: list = []
    cm = sampler if sampler is not None else contextlib.nullcontext()
    with cm:
        for i in range(int(steps)):
            t0 = time.perf_counter_ns()
            outs = _fence_step(pipeline, fence_kind=fence_kind, fenced=fenced)
            step_ns.append(time.perf_counter_ns() - t0)
            if i == int(steps) - 1:
                last_outs = outs
    step_ms = [ns / 1e6 for ns in step_ns]
    mean_ms = statistics.fmean(step_ms) if step_ms else 0.0
    ms_per_layer = (mean_ms / n_layers) if n_layers else 0.0
    util_summary = sampler.summarize(keep_series=False) if sampler is not None else None
    return ms_per_layer, mean_ms, step_ms, last_outs, util_summary


def fence_probe(args: ModelArgs, *, T: int, steps: int, warmup: int,
                fence_kind: str = "eval_tolist", seed: int = _FENCE_SEED,
                util_on: bool = False, util_interval_ms: int = 2000) -> dict:
    """Run the unfenced (pipelined) and fenced (per-layer barrier) loops back to
    back and return both ms/layer figures + the bit-identity verdict.  The two loops
    build from the SAME ``seed`` (byte-identical weights + inputs) and differ ONLY in
    the sync discipline, so their outputs must match bit-for-bit -- the discriminator
    for H1 (per-kernel GPU overhead) vs H2 (exposed host-encode from the per-layer
    routing barrier)."""
    if fence_kind not in _FENCE_KINDS:
        raise ValueError(f"--fence-kind must be one of {_FENCE_KINDS}, got {fence_kind!r}")
    n_layers = len(_backbone_modes(args))

    pipeline = _build_fence_pipeline(args, T, seed=seed)
    ms_u, mean_u, _step_u, outs_u, util_u = _run_fence_loop(
        pipeline, steps=steps, warmup=warmup, fence_kind=fence_kind,
        fenced=False, util_on=util_on, util_interval_ms=util_interval_ms)
    del pipeline
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass

    pipeline = _build_fence_pipeline(args, T, seed=seed)  # same seed -> identical build
    ms_f, mean_f, _step_f, outs_f, util_f = _run_fence_loop(
        pipeline, steps=steps, warmup=warmup, fence_kind=fence_kind,
        fenced=True, util_on=util_on, util_interval_ms=util_interval_ms)
    del pipeline
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass

    bit_identical = (
        len(outs_u) == len(outs_f)
        and all(bool(mx.array_equal(a, b).item()) for a, b in zip(outs_u, outs_f))
    )
    util = None
    if util_u is not None or util_f is not None:
        util = {"unfenced": util_u, "fenced": util_f}

    return {
        "isolated_ms_per_layer_unfenced": ms_u,
        "isolated_ms_per_layer_fenced": ms_f,
        "fence_kind": fence_kind,
        "n_layers": n_layers,
        "context_tokens": int(T),
        "steps": int(steps),
        "warmup_steps": int(warmup),
        "readback_elems": (_FENCE_READBACK_ELEMS if fence_kind == "eval_tolist" else 0),
        "mean_ms_per_step_unfenced": mean_u,
        "mean_ms_per_step_fenced": mean_f,
        "fenced_over_unfenced": (ms_f / ms_u) if ms_u > 0 else None,
        "outputs_bit_identical": bool(bit_identical),
        "utilization": util,
        # underscore-prefixed: the raw output arrays for a direct bit-identity check;
        # run() strips these from the JSON receipt.
        "_outputs_unfenced": outs_u,
        "_outputs_fenced": outs_f,
    }


def print_fence_probe(receipt: dict) -> None:
    """One-block summary of the fence-per-layer probe from a receipt that carries it."""
    fp = receipt.get("fence_probe")
    if not fp:
        return
    print(f"\n=== W102 isolated fence-per-layer probe "
          f"[{receipt['device']}, fence_kind={fp['fence_kind']}, "
          f"n_layers={fp['n_layers']}, ctx={fp['context_tokens']}, "
          f"steps={fp['steps']} warmup={fp['warmup_steps']}] ===")
    print(f"unfenced (pipelined)  ms/layer: {fp['isolated_ms_per_layer_unfenced']:.4f}"
          f"   (mean {fp['mean_ms_per_step_unfenced']:.3f} ms/step)")
    r = fp.get("fenced_over_unfenced")
    rtxt = f"   fenced/unfenced: {r:.2f}x" if isinstance(r, (int, float)) else ""
    print(f"fenced   (per-layer)  ms/layer: {fp['isolated_ms_per_layer_fenced']:.4f}"
          f"   (mean {fp['mean_ms_per_step_fenced']:.3f} ms/step){rtxt}")
    print(f"outputs bit-identical: {fp['outputs_bit_identical']}")


def run(args_cfg: dict) -> dict:
    """Programmatic entry (used by the unit test).  ``args_cfg`` keys: tiny, gpu,
    Ts, iters, warmup, use_selected, win_memo, compile."""
    tiny = args_cfg.get("tiny", False)
    gpu = args_cfg.get("gpu", False)
    if gpu and not tiny:
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)

    use_selected = args_cfg.get("use_selected", True)
    os.environ["MTPLX_DSV41_SELECTED_KEYS"] = "1" if use_selected else "0"
    dsv41._ATTN_WIN_MEMO = bool(args_cfg.get("win_memo", True))
    dsv41._ATTN_COMPILE = bool(args_cfg.get("compile", True))
    ballast_gib = float(args_cfg.get("ballast_gib", 0.0))
    ballast_churn = bool(args_cfg.get("ballast_churn", False))

    args = tiny_args() if tiny else real_args()
    Ts: List[int] = list(args_cfg.get("Ts", [256, 1024] if tiny else [1024, 4096, 16384]))
    iters = int(args_cfg.get("iters", 5 if tiny else 20))
    warmup = int(args_cfg.get("warmup", 2 if tiny else 3))

    # Hold the ballast for the whole run (resident-set pressure), then record what
    # is actually resident so the window can see how close it is to the child cap.
    try:
        mx.reset_peak_memory()
    except Exception:
        pass
    _ballast = alloc_ballast(ballast_gib)
    mem_after_ballast_gib = _mem_gib("get_active_memory")

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mode in _MODES:
        attn = _build_layer(args, mode)
        results[mode] = {}
        for T in Ts:
            results[mode][str(T)] = measure_mode_T(
                args, attn, mode, T, use_selected, iters, warmup, churn=ballast_churn
            )
        del attn
        gc.collect()
        # Default path: free the pool between modes as before.  With ballast held,
        # skip it -- clearing the pool would evict the resident pressure we are
        # deliberately holding (the whole point of --ballast-gib).
        if ballast_gib <= 0:
            try:
                mx.clear_cache()
            except Exception:
                pass

    mem = {
        "ballast_gib": ballast_gib,
        "n_ballast_chunks": len(_ballast),
        "active_after_ballast_gib": mem_after_ballast_gib,
        "active_end_gib": _mem_gib("get_active_memory"),
        "peak_gib": _mem_gib("get_peak_memory"),
    }
    del _ballast
    gc.collect()

    receipt = {
        "worker": "W78",
        "script": "scripts/deepseek_v41/metal_decode_attn_bisect.py",
        "device": "gpu" if (gpu and not tiny) else "cpu",
        "tiny": tiny,
        "dims": {
            "head_dim": args.head_dim, "n_heads": args.num_attention_heads,
            "window": args.window_size, "index_topk": args.index_topk,
            "index_n_heads": args.index_n_heads, "index_head_dim": args.index_head_dim,
            "hidden": args.hidden_size,
        },
        "mode_layer": _mode_layer_map(args),
        "env": {
            "MTPLX_DSV41_SELECTED_KEYS": os.environ.get("MTPLX_DSV41_SELECTED_KEYS"),
            "MTPLX_DSV41_ATTN_COMPILE": dsv41._ATTN_COMPILE,
            "MTPLX_DSV41_ATTN_WIN_MEMO": dsv41._ATTN_WIN_MEMO,
            "MTPLX_DSV41_KV_CHUNK_GROW": os.environ.get("MTPLX_DSV41_KV_CHUNK_GROW"),
            "MTPLX_DSV41_SELECT_FENCE": os.environ.get("MTPLX_DSV41_SELECT_FENCE"),
        },
        "path": "selected_keys" if use_selected else "masked_full",
        "ballast_gib": ballast_gib,
        "ballast_churn": ballast_churn,
        "memory": mem,
        "Ts": Ts,
        "iters": iters,
        "warmup": warmup,
        "results": results,
    }

    # W102: additive fence-per-layer probe.  Off by default -> the receipt above is
    # byte-for-byte the pre-W102 census.  On -> add the pipelined-vs-fenced ms/layer
    # discriminator (both loops in this one run) at the top level + a detail block.
    if bool(args_cfg.get("fence_per_layer", False)):
        fk = str(args_cfg.get("fence_kind", "eval_tolist"))
        T_ctx = int(args_cfg.get("context_tokens", 256 if tiny else 16384))
        fsteps = int(args_cfg.get("fence_steps", _FENCE_DEFAULT_STEPS))
        fwarm = int(args_cfg.get("fence_warmup", _FENCE_DEFAULT_WARMUP))
        util_on = bool(args_cfg.get("utilization", False)) and (gpu and not tiny)
        util_interval_ms = int(args_cfg.get("util_interval_ms", 2000))
        probe = fence_probe(
            args, T=T_ctx, steps=fsteps, warmup=fwarm, fence_kind=fk,
            util_on=util_on, util_interval_ms=util_interval_ms,
        )
        receipt["isolated_ms_per_layer_unfenced"] = probe["isolated_ms_per_layer_unfenced"]
        receipt["isolated_ms_per_layer_fenced"] = probe["isolated_ms_per_layer_fenced"]
        receipt["fence_kind"] = probe["fence_kind"]
        # detail block, minus the raw output arrays (not JSON-serialisable receipt data)
        receipt["fence_probe"] = {k: v for k, v in probe.items() if not k.startswith("_")}

    return receipt


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------
def _ratio(hi: Optional[float], lo: Optional[float]) -> str:
    if not hi or not lo or lo <= 0:
        return "  -  "
    return f"{hi / lo:5.2f}"


def print_tables(receipt: dict) -> None:
    Ts = receipt["Ts"]
    Tlo, Thi = Ts[0], Ts[-1]
    print(f"\n=== W78 Metal decode-attention op bisect "
          f"[{receipt['device']}, path={receipt['path']}, "
          f"compile={receipt['env']['MTPLX_DSV41_ATTN_COMPILE']}, "
          f"win_memo={receipt['env']['MTPLX_DSV41_ATTN_WIN_MEMO']}] ===")
    mem = receipt.get("memory", {})
    print(f"dims: {receipt['dims']}   iters={receipt['iters']} warmup={receipt['warmup']}")
    print(f"ballast_gib={receipt.get('ballast_gib', 0.0)} churn={receipt.get('ballast_churn', False)}  "
          f"active_after_ballast={mem.get('active_after_ballast_gib')} "
          f"active_end={mem.get('active_end_gib')} peak={mem.get('peak_gib')} GiB")
    hdr = "op".ljust(16) + "".join(f"{('T=' + str(T)):>11}" for T in Ts) + f"{'ratio(hi/lo)':>14}"
    rows = ["qkv_proj", "cache_append", "mask_build", "compress_append", "select",
            "gather_iso", "score", "attend", "out_proj", "peeled_sum", "whole"]
    for mode in _MODES:
        print(f"\n-- {mode} (layer {receipt['mode_layer'][mode]}) --  ms/step")
        print(hdr)
        print("-" * len(hdr))
        res = receipt["results"][mode]
        for op in rows:
            vals = [res[str(T)].get(op) for T in Ts]
            if all((v is None or v == 0.0) for v in vals) and op in ("mask_build", "score", "gather_iso"):
                continue
            cells = "".join((f"{v:>11.4f}" if v is not None else f"{'-':>11}") for v in vals)
            r = _ratio(res[str(Thi)].get(op), res[str(Tlo)].get(op))
            print(op.ljust(16) + cells + f"{r:>14}")


# ===========================================================================
# --in-model: the peel on the REAL loaded model with a live growing cache
# ===========================================================================
# Window 32: heap size + allocator churn (ballast) do NOT reproduce the census
# 6.6 ms/layer for Reuse at 16K -- the isolated bench stays flat ~2 ms.  The real
# decode differs in the live cache/state after a 16K prefill, T growing by one
# each step across 40 layers, and the surrounding pipeline (async expert gathers
# whose drain lands inside the next attention fence).  This mode measures the
# attention ops IN SITU on the real layers under three pipeline conditions:
#   (1) full model, real decode, T advancing one/step;
#   (2) same, with the routed expert switch stubbed to a no-op (shared kept);
#   (3) same, with attention stubbed to a no-op (measures the rest of the step).
# It reuses the proven ab_decode_env_levers loader + census pass (no re-impl); the
# per-mode attention ms/layer come straight from the production decode stage-timing
# probe (``attn.<mode>`` mean_ms), exactly the census window 30 read.


def _load_ab_module():
    """Import the sibling ab_decode_env_levers.py for its model loader + census
    pass (``_load_model`` / ``_stage_timing_pass`` / bench harness).  Its top level
    imports only the standard library, so this is CPU-safe."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent / "ab_decode_env_levers.py"
    spec = importlib.util.spec_from_file_location("_w78_ab_decode_env_levers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import ab module at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _ZeroSwitch:
    """Stub for a layer's ``mlp.switch_mlp`` (the streamed routed-expert seam):
    returns the unweighted per-expert output as zeros ``[n, top_k, dim]``, so the
    MoE combine yields just the shared expert (routed contribution zeroed).  Skips
    the expensive routed gather + gather_qmm entirely.  cell16k does not use the
    shared-overlap path, so the plain ``(xf, indices)`` call is the only one.

    ``keep_routing_barrier`` (window-94): the REAL streamed switch pays a per-layer
    host round-trip -- ``mx.eval(indices)`` (expert_mlx.py: the ``hot.eval_indices``
    barrier the ledger prices at ~40/token) -- to read the route before it can
    stream.  The zeros return alone NEVER touches ``indices`` on host (only its
    static ``shape[-1]``), so a plain stub silently drops that barrier along with
    the gather.  The unfenced attribution needs to SPLIT those: pass 2 keeps the
    barrier (``keep_routing_barrier=True`` -> the SSD/gather cost is what the
    full-vs-2 delta then isolates) and pass 4 drops it (``False`` -> the 2-vs-4
    delta is the ~40 host syncs alone).  Default ``False`` so the fenced 3-pass
    mode's ``_ZeroSwitch()`` is byte-for-byte the pre-window-94 stub (that mode
    already owns the barrier at the ``moe.gate_topk`` fence)."""

    __slots__ = ("keep_routing_barrier",)

    def __init__(self, keep_routing_barrier: bool = False):
        self.keep_routing_barrier = bool(keep_routing_barrier)

    def __call__(self, x, indices, *args, **kwargs):
        if self.keep_routing_barrier:
            # The per-layer routing host-sync the streamed switch pays, kept while
            # the SSD gather + gather_qmm are skipped.  Faithful to the production
            # barrier: one host round-trip per streamed layer per token.
            mx.eval(indices)
        return mx.zeros((x.shape[0], indices.shape[-1], x.shape[-1]), dtype=x.dtype)


class _ZeroAttn:
    """Stub for a layer's ``attn`` (``Attention.__call__``): returns a zero tensor
    of the attention-input shape, so the Hyper-Connection folds a zero attention
    contribution.  Skips qkv/append/gather/score/out and the ``attn.<mode>`` stage
    bracket.  The KV cache offset still advances (backbone ``cache.advance`` is
    independent of attention), so the decode loop stays valid."""

    def __call__(self, x, positions, layer_cache, shared):
        return mx.zeros_like(x)


def _apply_stub(model, which: str, *, keep_routing_barrier: bool = False):
    """Swap in the expert / attention stub on every backbone layer; returns a
    restore list.  ``which`` in {"experts", "attn"}.  ``keep_routing_barrier``
    (experts only) controls whether the expert stub keeps the per-layer
    ``mx.eval(indices)`` routing barrier (window-94; ignored for "attn")."""
    saved = []
    for layer in model.layers:
        if which == "experts":
            saved.append((layer.mlp, "switch_mlp", layer.mlp.switch_mlp))
            layer.mlp.switch_mlp = _ZeroSwitch(keep_routing_barrier=keep_routing_barrier)
        elif which == "attn":
            saved.append((layer, "attn", layer.attn))
            layer.attn = _ZeroAttn()
        else:
            raise ValueError(which)
    return saved


#: Restore sentinel: the attribute was NOT an instance override before the stub (a
#: class method), so restore = delete the instance shadow the stub set, revealing the
#: class method again (W97 attn sub-op stubs; see :func:`_apply_attn_subop_stub`).
_RESTORE_DELETE = object()


def _restore_stub(saved):
    for obj, name, orig in saved:
        if orig is _RESTORE_DELETE:
            try:
                delattr(obj, name)
            except AttributeError:
                pass
        else:
            setattr(obj, name, orig)


def _summarize_report(report: dict) -> dict:
    """Pull the per-mode attention ms/layer and the key non-attention stages out of
    a decode stage-timing report (``mean_ms`` is per-layer-per-token == ms/layer)."""
    if not report or not report.get("enabled", False):
        return {"enabled": False}
    stages = report.get("stages", {})

    def _mean(name):
        s = stages.get(name)
        return None if s is None else s.get("mean_ms")

    dec = report.get("decode_breakdown", {}) or {}

    def _dmean(name):
        s = dec.get(name)
        return None if s is None else s.get("mean_ms")

    out = {
        "enabled": True,
        "frame_wall_ms_per_token": report.get("frame_wall_ms_per_token"),
        "stage_sum_ms_per_token": report.get("stage_sum_ms_per_token"),
        "tokens": report.get("tokens"),
        "attn_ms_per_layer": {m: _mean("attn." + m) for m in _MODES},
        "decode_breakdown_ms_per_layer": {
            m: {
                "cache_append": _dmean("attn." + m + ".cache_append"),
                "compress_append": _dmean("attn." + m + ".compress_append"),
                "select": _dmean("attn." + m + ".select"),
            }
            for m in _MODES
        },
        "non_attn_ms_per_layer": {
            "moe.gate_topk": _mean("moe.gate_topk"),
            "moe.routed_switch": _mean("moe.routed_switch"),
            "moe.shared_expert": _mean("moe.shared_expert"),
            "moe.combine": _mean("moe.combine"),
            "hc.premix_sinkhorn": _mean("hc.premix_sinkhorn"),
            "hc.combine": _mean("hc.combine"),
        },
        "per_token": {
            "embed": _mean("embed"),
            "final_norm": _mean("final_norm"),
            "head": _mean("head"),
            "sample": _mean("sample"),
        },
    }
    return out


def _tiny_full_args():
    """A tiny full-model config that derives all four CSA modes (mirrors
    tests/models/test_deepseek_v41_stage_timing.py ``_csa_args``, the proven
    full-Model + stage-timing config)."""
    return ModelArgs(
        vocab_size=48, hidden_size=32, num_hidden_layers=8,
        num_attention_heads=4, head_dim=16, qk_rope_head_dim=4,
        q_lora_rank=12, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=2,
        index_n_heads=2, index_head_dim=8, index_topk=5,
        sliding_window=8, window_size=8, swiglu_limit=0.5,
        compress_ratios=[0, 0, 2, 2, 2, 1, 1, 1],
        kv_source_layer_ids=[2, 5], index_source_layer_ids=[2, 5, 6],
        candidate_source_layer_id=5, candidate_topk_blocks=3, candidate_block_size=2,
        rope_scaling={"rope_type": "yarn", "factor": 16, "beta_fast": 32,
                      "beta_slow": 1, "original_max_position_embeddings": 65536},
    )


def _build_tiny_full_model(seed: int = 1):
    """Full tiny ``Model`` with random weights (no artifact) for CPU validation."""
    from mlx.utils import tree_flatten, tree_unflatten
    from mtplx.models.deepseek_v41 import Model
    args = _tiny_full_args()
    model = Model(args)
    mx.random.seed(seed)
    new = []
    for name, arr in tree_flatten(model.parameters()):
        if arr.ndim == 1 and ("norm_weight" in name or name.endswith("norm.weight")):
            v = 1.0 + 0.2 * mx.random.normal(arr.shape)
        elif "attn_sink" in name:
            v = 0.5 * mx.random.normal(arr.shape)
        else:
            v = 0.1 * mx.random.normal(arr.shape)
        new.append((name, v.astype(mx.float32)))
    model.update(tree_unflatten(new))
    mx.eval(model.parameters())
    return model, args


class _TinyOps:
    """The ops surface ``ab._stage_timing_pass`` needs, for the CPU tiny model."""

    def input(self, ids_2d):
        return mx.array(ids_2d)

    def argmax_last(self, logits) -> int:
        return int(mx.argmax(logits[0, -1]).item())

    def sync(self, logits) -> None:
        mx.eval(logits)


def _in_model_ab_args(ab, cfg: dict, steps: int, arm: str):
    """Build the ab_decode_env_levers args namespace for a GPU in-model pass from
    cfg, exactly as the ab bench path constructs it."""
    argv = [
        "--model", str(cfg["model"]),
        "--context-tokens", str(int(cfg.get("context_tokens", 16384))),
        "--decode-tokens", str(steps),
        "--max-kv", str(int(cfg.get("max_kv", 17408))),
        "--memory-limit-gib", str(float(cfg.get("memory_limit_gib", 60.0))),
        "--arms", arm,
        "--out", "/dev/null",  # unused: we write our own receipt
    ]
    if cfg.get("prompt_ids_file"):
        argv += ["--prompt-ids-file", str(cfg["prompt_ids_file"])]
    if cfg.get("prompt_seed") is not None:
        argv += ["--prompt-seed", str(int(cfg["prompt_seed"]))]
    return ab.build_parser().parse_args(argv)


def _resolve_in_model_prompt(ab, bench, args):
    """Resolve the in-model prompt exactly as ab_decode_env_levers.py's bench path:
    the exported ids when ``--prompt-ids-file`` is given, else the standard
    1K/16K prefill_bench builder run with the model's tokenizer.

    Window-35 step 1 (no ``--prompt-ids-file``) crashed here because the in-model
    path passed ``tokenizer=None`` into the builder (``'NoneType' has no attribute
    'encode'``); loading the tokenizer with ``ab._tokenizer`` -- the same call the
    ab bench lane makes -- is the fix.  The tokenizer is loaded from ``args.model``
    (the streaming artifact carries the tokenizer files); no ids file means no
    tokenizer is needed at all, matching the ab path's guard."""
    build_prompt = bench._load_build_prompt()
    tokenizer = (
        None
        if getattr(args, "prompt_ids_file", None)
        else ab._tokenizer(args, bench)
    )
    return bench._resolve_prompt(args, tokenizer, build_prompt, args.context_tokens)


def _in_model_setup(cfg: dict, steps: int) -> dict:
    """Load the model + resolve the standard cell prompt for an in-model run.

    Shared by both in-model modes -- the fenced 3-pass census (:func:`run_in_model`)
    and the unfenced 5-pass attribution (:func:`run_in_model_unfenced`) -- so both
    drive the SAME loader, arm env, prompt and ops (window-94 refactor; the fenced
    mode is byte-for-byte unchanged).  ``--tiny`` builds the fake CPU model with the
    clean stage-timing config; ``--gpu`` loads the real streaming artifact via the
    ab loader under the requested arm.  Returns the handles both modes need."""
    ab = _load_ab_module()
    tiny = cfg.get("tiny", False)
    arm = cfg.get("arms", "cell16k")

    if tiny:
        # CPU fake-config validation: clean defaults (the proven stage-timing test
        # config), NOT the cell16k GPU levers.  Only the stub plumbing is exercised.
        mx.set_default_device(mx.cpu)
        for k in ("MTPLX_DSV41_SELECTED_KEYS", "MTPLX_DSV41_KV_CHUNK_GROW",
                  "MTPLX_DSV41_SELECT_FENCE"):
            os.environ[k] = "0"
        dsv41._ATTN_COMPILE = False
        dsv41._ATTN_WIN_MEMO = False
        model, args = _build_tiny_full_model(seed=int(cfg.get("seed", 1)))
        ops = _TinyOps()
        prompt_ids = list(range(1, 1 + int(cfg.get("prompt_len", 48))))
        dims = {"hidden": args.hidden_size, "n_layers": args.num_hidden_layers,
                "head_dim": args.head_dim, "n_heads": args.num_attention_heads}
        prompt_meta = {"prompt_source": "tiny_synthetic", "prompt_tokens": len(prompt_ids)}
        device = "cpu"
    else:
        mx.set_default_device(mx.gpu)
        bench = ab._load_bench_module()
        args = _in_model_ab_args(ab, cfg, steps, arm)
        ab._apply_arm_env(arm)  # cell16k: exactly the census arm
        # This script imports dsv41 at top (before the arm), so the import-frozen
        # globals must be re-synced from the arm env (the ab script imports dsv41
        # lazily AFTER the arm, so it never needs this).
        def _envon(key):
            return (os.environ.get(key) or "").strip().lower() not in (
                "", "0", "false", "no", "off", "auto")
        dsv41._ATTN_COMPILE = _envon("MTPLX_DSV41_ATTN_COMPILE")
        dsv41._ATTN_WIN_MEMO = _envon("MTPLX_DSV41_ATTN_WIN_MEMO")
        if hasattr(dsv41, "_HC_COMPILE"):
            dsv41._HC_COMPILE = _envon("MTPLX_DSV41_HC_COMPILE")
        prompt_ids, prompt_meta = _resolve_in_model_prompt(ab, bench, args)
        resident = ab._load_model(args, bench, mx)
        model = resident.model
        ops = bench._MLXOps(mx)
        dims = {"hidden": args.context_tokens, "context_tokens": args.context_tokens,
                "max_kv": args.max_kv, "memory_limit_gib": cfg.get("memory_limit_gib", 60.0)}
        device = "gpu"

    return {"ab": ab, "tiny": tiny, "arm": arm, "model": model, "ops": ops,
            "prompt_ids": prompt_ids, "prompt_meta": prompt_meta, "dims": dims,
            "device": device}


def run_in_model(cfg: dict) -> dict:
    """The three in-situ passes on the real (or tiny) model.  Reuses the ab loader
    + census pass; applies the expert / attention stubs around passes (2) / (3)."""
    tiny = cfg.get("tiny", False)
    steps = int(cfg.get("steps", 4 if tiny else 30))
    setup = _in_model_setup(cfg, steps)
    ab, arm = setup["ab"], setup["arm"]
    model, ops = setup["model"], setup["ops"]
    prompt_ids, prompt_meta = setup["prompt_ids"], setup["prompt_meta"]
    dims, device = setup["dims"], setup["device"]

    cooldown_s = float(cfg.get("cooldown_s", 0.0) or 0.0)
    util_on = bool(cfg.get("utilization", False))
    util_interval_ms = int(cfg.get("util_interval_ms", 2000))

    def _pass(label, *, cooldown_s=0.0, util=False):
        sampler = None
        if util:
            sampler = ab._macmon().UtilizationSampler(interval_ms=util_interval_ms)
        rep = ab._stage_timing_pass(
            model=model, ops=ops, prompt_ids=list(prompt_ids), steps=steps,
            cooldown_s=cooldown_s, util_sampler=sampler,
        )
        entry = {"label": label, "report": rep, "summary": _summarize_report(rep)}
        # W90: peel the utilization / cooldown blocks out of the report onto the pass.
        entry["utilization"] = rep.pop("__w90_utilization", None)
        entry["cooldown"] = rep.pop("__w90_cooldown", None)
        if sampler is not None:
            print(f"[w78-in-model] {label}: {sampler.census()}", flush=True)
        return entry

    passes = {}
    # (1) full model -- the W90 discriminator pass carries the cooldown + macmon
    # utilization telemetry (the GPU DVFS-downclock floor read).
    passes["full"] = _pass("full_model", cooldown_s=cooldown_s, util=util_on)
    # (2) routed expert switch stubbed (keep shared)
    saved = _apply_stub(model, "experts")
    try:
        passes["expert_stub"] = _pass("expert_switch_stubbed")
    finally:
        _restore_stub(saved)
    # (3) attention stubbed (measure the rest of the step)
    saved = _apply_stub(model, "attn")
    try:
        passes["attn_stub"] = _pass("attention_stubbed")
    finally:
        _restore_stub(saved)

    receipt = {
        "worker": "W78",
        "script": "scripts/deepseek_v41/metal_decode_attn_bisect.py",
        "mode": "in_model",
        "device": device,
        "tiny": tiny,
        "arm": arm,
        "steps": steps,
        "dims": dims,
        "prompt": prompt_meta,
        "ballast_gib": 0.0,
        "memory": {
            "active_end_gib": _mem_gib("get_active_memory"),
            "peak_gib": _mem_gib("get_peak_memory"),
        },
        # W90: the GPU DVFS/utilization + cooldown telemetry of the full pass (the
        # window-36 --cooldown-s 180-vs-0 discriminator for the mode-independent floor).
        "utilization": passes["full"].get("utilization"),
        "cooldown": passes["full"].get("cooldown"),
        "passes": passes,
    }
    return receipt


def print_in_model(receipt: dict) -> None:
    print(f"\n=== W78 --in-model [{receipt['device']}, arm={receipt['arm']}, "
          f"steps={receipt['steps']}] ===")
    print(f"dims: {receipt['dims']}   prompt: {receipt['prompt']}")
    mem = receipt.get("memory", {})
    print(f"active_end={mem.get('active_end_gib')} peak={mem.get('peak_gib')} GiB")
    labels = [("full", "(1) full"), ("expert_stub", "(2) expert-stub"),
              ("attn_stub", "(3) attn-stub")]
    hdr = "attn ms/layer".ljust(16) + "".join(f"{lbl:>18}" for _k, lbl in labels)
    print("\n-- attention ms/layer (decode stage-timing attn.<mode> mean) --")
    print(hdr)
    print("-" * len(hdr))
    for mode in _MODES:
        cells = ""
        for key, _lbl in labels:
            s = receipt["passes"][key]["summary"]
            v = (s.get("attn_ms_per_layer") or {}).get(mode) if s.get("enabled") else None
            cells += (f"{v:>18.4f}" if isinstance(v, (int, float)) else f"{'-':>18}")
        print(mode.ljust(16) + cells)
    print("\n-- whole step / non-attention (ms/token, ms/layer) --")
    rows = [("frame_wall_ms_per_token", "frame_wall/tok", "frame_wall_ms_per_token"),
            ("moe.routed_switch", "moe.routed/lyr", None),
            ("moe.shared_expert", "moe.shared/lyr", None),
            ("hc.premix_sinkhorn", "hc.premix/lyr", None),
            ("hc.combine", "hc.combine/lyr", None)]
    for key, lbl, top in rows:
        cells = ""
        for pk, _lbl in labels:
            s = receipt["passes"][pk]["summary"]
            if not s.get("enabled"):
                cells += f"{'-':>18}"
                continue
            if top:
                v = s.get(top)
            else:
                v = (s.get("non_attn_ms_per_layer") or {}).get(key)
            cells += (f"{v:>18.4f}" if isinstance(v, (int, float)) else f"{'-':>18}")
        print(lbl.ljust(16) + cells)


# ===========================================================================
# --in-model --unfenced: whole-token frame-wall attribution (window 94)
# ===========================================================================
# The FENCED per-stage census (run_in_model / deepseek_v41_stage_timing) brackets
# each stage with mx.eval, so every bracket's wall is a GPU pipeline drain + refill
# -- ~8 ms per attention bracket in EVERY config (window 36/37) while the same
# kernels cost ~2 ms isolated.  Those absolute per-stage numbers are LATENCY, not
# compute (ratios only).  This mode drops all per-stage fences and measures the
# WHOLE-TOKEN frame wall with components stubbed out, so the delta full - stubbed is
# a component's TRUE in-situ cost -- including whatever latency it causes (the DVFS
# downclock in the sync gap, the SSD wait) but excluding probe latency.  The only
# mx.eval per token is the production decode loop's own (the sampler / next-token
# id), exactly ab_decode_env_levers._generate's classic argmax loop.
#
# Five passes (mean/median ms/token over the steps, first `warmup_steps` excluded):
#   (1) full            -- the served decode.
#   (2) expert_stub     -- routed switch stubbed, routing barrier KEPT
#                          (_ZeroSwitch(keep_routing_barrier=True): the per-layer
#                          mx.eval(indices) host sync stays; SSD gather + gather_qmm
#                          skipped).  (1)-(2) = the switch's SSD/gather cost.
#   (3) attn_stub       -- attention stubbed (zeros; KV offset still advances).
#                          (1)-(3) = attention incl. any latency it causes.
#   (4) expert_stub_nobarrier -- routed switch stubbed, barrier REMOVED
#                          (_ZeroSwitch(keep_routing_barrier=False), the stub never
#                          touches indices on host).  (2)-(4) = the ~40 host syncs
#                          alone; (1)-(4) = the whole switch (SSD + gather + syncs).
#   (5) small_stages_floor -- attention zeros AND switch stubbed no-barrier: the
#                          floor of the small per-layer stages (norms / HC / Sinkhorn
#                          / gate / shared / combine) + head + sample.
# Derived: switch total (1)-(4) = SSD-bound (1)-(2) + sync (2)-(4); attention (1)-(3);
# small stages (5); sum-of-parts (attn+switch+floor) vs full, and the residual.


def _unfenced_decode_pass(*, ab, model, ops, prompt_ids, steps, warmup_steps=8,
                          cooldown_s=0.0, util_sampler=None, arm_route_probe=False):
    """One UNFENCED in-situ decode pass.

    Prefill once (untimed for the per-step table -- its wall is TTFT), then run
    ``steps`` production decode steps with NO stage recorder armed and NO per-stage
    fences: the only ``mx.eval`` per token is the production loop's own
    (``ops.sync`` + the argmax host read = the sampler / next-token id).  Records the
    per-step wall (``perf_counter_ns`` only -- no GPU sync of its own, so it never
    perturbs the measurement), the decoded ids, the decode-scoped expert-streaming
    counter delta (prefill excluded), and -- via ``util_sampler`` entered over the
    decode loop -- the macmon telemetry.  Mirrors ab._generate's classic argmax loop
    exactly, so pass (1)'s ids are the served-path ids.

    ``arm_route_probe`` (window-94, pass (1) only): arm the route-stage probe over
    the decode loop so the pass records the DECODE ``mx.eval(indices)`` barrier time
    DIRECTLY -- the ``hot.eval_indices`` sum the ledger prices at ~40/token -- so
    window-38's receipt carries the eval(indices) time.  The probe's ``bracket``
    only wraps the ``mx.eval`` the production switch already runs with a
    ``perf_counter`` (it does NOT add a fence), so the frame wall is unperturbed, and
    it does NOT force the eager path (that is gated on ``stime.recording()``, not the
    route probe).  Cleared HERE, immediately before the before-snapshot and nowhere
    between it and the after-snapshot, so the recorded sums/counts ARE the exact
    decode deltas (baseline 0), never negative -- the same discipline as the
    ab_decode_env_levers._generate fix."""
    from mtplx.models import deepseek_v41_stage_timing as stime
    # No stage session may be armed: an armed probe forces the eager attention / HC
    # path (recording() gates _attn_use_compile / _hc_use_compile), which would
    # change the code path away from the served (compiled) one.
    if stime.is_active():
        stime.end()

    cache = model.make_cache()
    t0 = time.perf_counter()
    logits = model(ops.input([list(prompt_ids)]), cache=cache)
    ops.sync(logits)
    ttft_s = time.perf_counter() - t0
    token = ops.argmax_last(logits)
    generated = [token]

    cooldown_block = None
    if cooldown_s and float(cooldown_s) > 0:
        cooldown_block = ab._macmon().cooldown(float(cooldown_s), label="w94-unfenced")

    # W94: arm + CLEAR the route-stage probe here -- immediately before the
    # before-snapshot and nowhere between it and the after-snapshot -- so the decode
    # route_probe sums/counts are the exact decode deltas (baseline 0, no negatives).
    _rp = None
    _rp_prev_enabled = None
    route_probe_counts = None
    route_probe_sums_ns = None
    if arm_route_probe:
        try:
            from mtplx import expert_route_probe as _rp
            _rp_prev_enabled = _rp.ENABLED
            _rp.ENABLED = True
            _rp._SUMS.clear()
            _rp._COUNTS.clear()
        except Exception:
            _rp = None

    # Decode-scoped expert-streaming counter bracket (prefill excluded, so the hit
    # rate / bytes are the DECODE traffic, not the cold prefill first-touch).
    sc_before = ab._stream_counters_snapshot(model)

    _util_cm = util_sampler if util_sampler is not None else contextlib.nullcontext()
    step_ns: List[int] = []
    try:
        with _util_cm:  # macmon sampled over the DECODE loop only
            decode_start = time.perf_counter()
            for _ in range(int(steps)):
                s0 = time.perf_counter_ns()
                logits = model(ops.input([[token]]), cache=cache)
                ops.sync(logits)                 # the one production per-token eval
                token = ops.argmax_last(logits)  # host read of the next-token id
                step_ns.append(time.perf_counter_ns() - s0)
                generated.append(token)
            decode_wall_s = time.perf_counter() - decode_start
    finally:
        # Baseline was cleared before the before-snapshot, so these ARE the exact
        # decode deltas.  Restore ENABLED even if the decode raised (never leave the
        # module armed for the rest of the process).
        if _rp is not None:
            route_probe_counts = {k: int(v) for k, v in dict(_rp._COUNTS).items()}
            route_probe_sums_ns = {k: int(v) for k, v in dict(_rp._SUMS).items()}
            if _rp_prev_enabled is not None:
                _rp.ENABLED = bool(_rp_prev_enabled)
    sc_after = ab._stream_counters_snapshot(model)

    stream = None
    try:
        from mtplx.serve_stream_counters import stream_counters_delta
        if sc_before is not None and sc_after is not None:
            stream = stream_counters_delta(
                sc_before, sc_after, tokens=int(steps), phase="decode"
            )
    except Exception:
        stream = None

    return {
        "step_ms": [ns / 1e6 for ns in step_ns],
        "ttft_s": ttft_s,
        "decode_wall_s": decode_wall_s,
        "generated": generated,
        "stream": stream,
        "cooldown": cooldown_block,
        "warmup_steps": int(warmup_steps),
        "route_probe_counts": route_probe_counts,
        "route_probe_sums_ns": route_probe_sums_ns,
    }


def _unfenced_pass_summary(pass_data: dict, util_summary: Optional[dict], *,
                           ssd_bandwidth_gibs: float) -> dict:
    """Fold one pass's raw per-step walls + telemetry into the summary row: mean /
    median ms/token over the post-warmup steps, tok/s (from the post-warmup mean),
    the macmon busy% / MHz, and the decode-scoped stream counters (misses / bytes /
    hit-rate per token) with an independent SSD-bound time estimate."""
    step_ms = pass_data.get("step_ms") or []
    wu = int(pass_data.get("warmup_steps", 8))
    warm = step_ms[wu:] if len(step_ms) > wu else list(step_ms)
    if not warm:
        warm = list(step_ms) or [0.0]
    mean_ms = statistics.fmean(warm)
    median_ms = statistics.median(warm)
    tok_s = (1000.0 / mean_ms) if mean_ms > 0 else None
    decode_wall_s = pass_data.get("decode_wall_s", 0.0)
    n = len(step_ms)
    decode_wall_tok_s = (n / decode_wall_s) if decode_wall_s > 0 else None

    def _u(field, key="mean"):
        d = (util_summary or {}).get(field)
        return d.get(key) if isinstance(d, dict) else None

    busy = _u("gpu_usage_ratio")
    gpu_busy_pct = busy * 100.0 if isinstance(busy, (int, float)) else None

    ec = ((pass_data.get("stream") or {}).get("expert_cache")) or {}
    bytes_per_tok = ec.get("bytes_read_per_token")
    ssd_est_ms = None
    if isinstance(bytes_per_tok, (int, float)) and bytes_per_tok > 0 and ssd_bandwidth_gibs > 0:
        ssd_est_ms = (bytes_per_tok / (ssd_bandwidth_gibs * (1024 ** 3))) * 1000.0

    return {
        "mean_ms_per_token": mean_ms,
        "median_ms_per_token": median_ms,
        "tok_s": tok_s,
        "steps": n,
        "warmup_steps": wu,
        "measured_steps": len(warm),
        "decode_wall_s": decode_wall_s,
        "decode_wall_tok_s": decode_wall_tok_s,
        "ttft_s": pass_data.get("ttft_s"),
        "gpu_busy_pct": gpu_busy_pct,
        "gpu_freq_mhz": _u("gpu_freq_mhz"),
        "gpu_power_w": _u("gpu_power_w"),
        "misses_per_token": ec.get("misses_per_token"),
        "bytes_read_per_token": bytes_per_tok,
        "records_streamed_per_token": ec.get("records_streamed_per_token"),
        "hit_rate": ec.get("hit_rate"),
        "ssd_bound_estimate_ms": ssd_est_ms,
    }


def _unfenced_attribution(passes: dict, *, ssd_bandwidth_gibs: float) -> dict:
    """The delta attribution over the five unfenced frame walls (ms/token).

    switch total (1)-(4) = SSD-bound (1)-(2) + sync/barrier (2)-(4); attention is
    (1)-(3); the small-stages floor is pass (5).  The sum-of-parts is
    attention + switch-total + floor, and the residual is full minus that sum (the
    interaction / overlap that only appears when the components coexist -- 0 under
    perfect additivity)."""
    def _ms(key):
        return (passes.get(key, {}).get("summary") or {}).get("mean_ms_per_token")

    p1, p2, p3, p4, p5 = (
        _ms("full"), _ms("expert_stub"), _ms("attn_stub"),
        _ms("expert_stub_nobarrier"), _ms("small_stages_floor"),
    )

    def _sub(a, b):
        return (a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None

    attention = _sub(p1, p3)
    switch_total = _sub(p1, p4)
    ssd_bound = _sub(p1, p2)
    sync_barrier = _sub(p2, p4)
    small_stages = p5
    sop = None
    if all(isinstance(x, (int, float)) for x in (attention, switch_total, small_stages)):
        sop = attention + switch_total + small_stages
    residual = _sub(p1, sop)

    full_summary = passes.get("full", {}).get("summary") or {}
    return {
        "full_ms_per_token": p1,
        "attention_ms_per_token": attention,
        "switch_total_ms_per_token": switch_total,
        "switch_ssd_bound_ms_per_token": ssd_bound,
        "switch_sync_barrier_ms_per_token": sync_barrier,
        # W94: the eval(indices) barrier time measured DIRECTLY on pass (1) (route
        # probe hot.eval_indices sum / steps) -- an independent cross-check of the
        # sync/barrier delta (2)-(4).  None off the streamed runtime (e.g. --tiny).
        "eval_indices_ms_per_token_measured": full_summary.get("eval_indices_ms_per_token"),
        "eval_indices_barriers_per_token": full_summary.get("eval_indices_barriers_per_token"),
        "small_stages_floor_ms_per_token": small_stages,
        "sum_of_parts_ms_per_token": sop,
        "unattributed_residual_ms_per_token": residual,
        "ssd_bound_independent_estimate_ms": full_summary.get("ssd_bound_estimate_ms"),
        "misses_per_token": full_summary.get("misses_per_token"),
        "bytes_read_per_token": full_summary.get("bytes_read_per_token"),
        "hit_rate": full_summary.get("hit_rate"),
        "ssd_bandwidth_gibs": ssd_bandwidth_gibs,
    }


#: (receipt key, pass label, which-stub, keep_routing_barrier, also-stub-attn).
_UNFENCED_PASSES = [
    ("full", "full_model", None, False, False),
    ("expert_stub", "expert_switch_stubbed_barrier", "experts", True, False),
    ("attn_stub", "attention_stubbed", "attn", False, False),
    ("expert_stub_nobarrier", "expert_switch_stubbed_no_barrier", "experts", False, False),
    ("small_stages_floor", "small_stages_floor", "experts", False, True),
]


def run_in_model_unfenced(cfg: dict) -> dict:
    """The five UNFENCED whole-token frame-wall passes on the real (or tiny) model.

    Reuses the shared loader (:func:`_in_model_setup`); each pass swaps in the
    relevant stub, runs :func:`_unfenced_decode_pass` (a fresh macmon sampler per
    pass so the single-use sampler is never reused), restores the stub, and folds the
    telemetry into a summary row.  Only pass (1) carries the cooldown before its timed
    decode and its token sha is the served-path comparison."""
    tiny = cfg.get("tiny", False)
    steps = int(cfg.get("steps", 8 if tiny else 64))
    warmup_steps = int(cfg.get("warmup_steps", 2 if tiny else 8))
    ssd_bw = float(cfg.get("ssd_bandwidth_gibs", 4.4) or 4.4)
    setup = _in_model_setup(cfg, steps)
    ab = setup["ab"]
    model, ops = setup["model"], setup["ops"]
    prompt_ids, prompt_meta = setup["prompt_ids"], setup["prompt_meta"]
    dims, device, arm = setup["dims"], setup["device"], setup["arm"]

    cooldown_s = float(cfg.get("cooldown_s", 0.0) or 0.0)
    util_on = bool(cfg.get("utilization", False))
    util_interval_ms = int(cfg.get("util_interval_ms", 2000))

    def _run_pass(label, *, cooldown_s=0.0, arm_route_probe=False):
        sampler = (ab._macmon().UtilizationSampler(interval_ms=util_interval_ms)
                   if util_on else None)
        data = _unfenced_decode_pass(
            ab=ab, model=model, ops=ops, prompt_ids=prompt_ids, steps=steps,
            warmup_steps=warmup_steps, cooldown_s=cooldown_s, util_sampler=sampler,
            arm_route_probe=arm_route_probe,
        )
        util_summary = sampler.summarize() if sampler is not None else None
        if sampler is not None:
            print(f"[w94-unfenced] {label}: {sampler.census()}", flush=True)
        summary = _unfenced_pass_summary(data, util_summary, ssd_bandwidth_gibs=ssd_bw)
        # W94: the DECODE eval(indices) barrier time (hot.eval_indices sum), summed
        # over the pass and divided by steps -- the per-token routing-barrier time,
        # measured directly (armed on pass (1) only).
        sums = data.get("route_probe_sums_ns") or {}
        counts = data.get("route_probe_counts") or {}
        eval_ns = sums.get("hot.eval_indices")
        summary["eval_indices_ms_per_token"] = (
            (eval_ns / 1e6 / max(1, int(steps))) if isinstance(eval_ns, (int, float)) else None
        )
        summary["eval_indices_barriers_per_token"] = (
            (counts.get("hot.eval_indices") / max(1, int(steps)))
            if isinstance(counts.get("hot.eval_indices"), (int, float)) else None
        )
        return {
            "label": label,
            "summary": summary,
            "utilization": util_summary,
            "cooldown": data.get("cooldown"),
            "token_ids_sha256": hashlib.sha256(
                json.dumps(data["generated"]).encode()
            ).hexdigest(),
            "n_token_ids": len(data["generated"]),
            "token_ids": list(data["generated"]),
            # W94: exact decode-scoped route-stage deltas (pass (1) only; None else).
            "route_probe_counts": data.get("route_probe_counts"),
            "route_probe_sums_ns": data.get("route_probe_sums_ns"),
        }

    passes: Dict[str, dict] = {}
    for key, label, which, keep_barrier, stub_attn in _UNFENCED_PASSES:
        saved = []
        if which == "experts":
            saved.append(_apply_stub(model, "experts", keep_routing_barrier=keep_barrier))
        elif which == "attn":
            saved.append(_apply_stub(model, "attn"))
        if stub_attn:
            saved.append(_apply_stub(model, "attn"))
        try:
            # Pass (1) alone carries the cooldown before its timed decode (matching
            # the fenced full pass) AND arms the route probe to record the eval(indices)
            # barrier time directly; the stub passes reuse the already-warm state and
            # never run the real switch, so its barrier is theirs to skip.
            passes[key] = _run_pass(
                label,
                cooldown_s=(cooldown_s if key == "full" else 0.0),
                arm_route_probe=(key == "full"),
            )
        finally:
            for s in reversed(saved):
                _restore_stub(s)

    attribution = _unfenced_attribution(passes, ssd_bandwidth_gibs=ssd_bw)

    receipt = {
        "worker": "W94",
        "script": "scripts/deepseek_v41/metal_decode_attn_bisect.py",
        "mode": "in_model_unfenced",
        "device": device,
        "tiny": tiny,
        "arm": arm,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "ssd_bandwidth_gibs": ssd_bw,
        "dims": dims,
        "prompt": prompt_meta,
        "ballast_gib": 0.0,
        "memory": {
            "active_end_gib": _mem_gib("get_active_memory"),
            "peak_gib": _mem_gib("get_peak_memory"),
        },
        # Pass-1 telemetry hoisted to the top for the receipt reader.
        "utilization": passes["full"].get("utilization"),
        "cooldown": passes["full"].get("cooldown"),
        # Pass (1) only: the served-path token ids + sha.  The sha is over
        # [prefill argmax] + `steps` decode ids, so it matches an ab receipt run at
        # --decode-tokens `steps`; the raw ids let a longer ab receipt be compared by
        # prefix (greedy is deterministic).  Stub-pass ids are garbage (not the model).
        "token_ids_sha256": passes["full"].get("token_ids_sha256"),
        "n_token_ids": passes["full"].get("n_token_ids"),
        "token_ids": passes["full"].get("token_ids"),
        # W94: pass (1)'s exact decode-scoped route-stage deltas (cleared before the
        # before-snapshot, so non-negative) -- window-38's eval(indices) time direct.
        "route_probe_counts": passes["full"].get("route_probe_counts"),
        "route_probe_sums_ns": passes["full"].get("route_probe_sums_ns"),
        "passes": passes,
        "attribution": attribution,
    }
    return receipt


def print_in_model_unfenced(receipt: dict) -> None:
    print(f"\n=== W94 --in-model --unfenced [{receipt['device']}, arm={receipt['arm']}, "
          f"steps={receipt['steps']} warmup={receipt['warmup_steps']}] ===")
    print(f"dims: {receipt['dims']}   prompt: {receipt['prompt']}")
    mem = receipt.get("memory", {})
    print(f"active_end={mem.get('active_end_gib')} peak={mem.get('peak_gib')} GiB")
    print(f"pass(1) token_ids_sha256={receipt.get('token_ids_sha256')}  "
          f"({receipt.get('n_token_ids')} ids = 1 prefill + {receipt['steps']} decode; "
          f"compare vs an ab receipt run at --decode-tokens {receipt['steps']})")

    order = [("full", "(1) full"),
             ("expert_stub", "(2) switch-stub +barrier"),
             ("attn_stub", "(3) attn-stub"),
             ("expert_stub_nobarrier", "(4) switch-stub no-barrier"),
             ("small_stages_floor", "(5) small-stages floor")]

    def _f(v, fmt):
        return fmt.format(v) if isinstance(v, (int, float)) else "-"

    hdr = ("pass".ljust(28) + f"{'ms/tok(mean)':>13}{'ms/tok(med)':>13}"
           f"{'tok/s':>9}{'busy%':>8}{'MHz':>8}{'miss/tok':>10}")
    print("\n-- unfenced whole-token frame wall (no per-stage fences; the signal) --")
    print(hdr)
    print("-" * len(hdr))
    for key, lbl in order:
        s = receipt["passes"][key]["summary"]
        print(lbl.ljust(28)
              + f"{_f(s.get('mean_ms_per_token'), '{:.3f}'):>13}"
              + f"{_f(s.get('median_ms_per_token'), '{:.3f}'):>13}"
              + f"{_f(s.get('tok_s'), '{:.3f}'):>9}"
              + f"{_f(s.get('gpu_busy_pct'), '{:.0f}'):>8}"
              + f"{_f(s.get('gpu_freq_mhz'), '{:.0f}'):>8}"
              + f"{_f(s.get('misses_per_token'), '{:.2f}'):>10}")

    a = receipt["attribution"]
    print("\n-- derived attribution (delta of unfenced frame walls, ms/token) --")
    print(f"  attention              (1)-(3) = {_f(a['attention_ms_per_token'], '{:.3f}')}")
    print(f"  switch total           (1)-(4) = {_f(a['switch_total_ms_per_token'], '{:.3f}')}")
    print(f"    of which SSD-bound   (1)-(2) = {_f(a['switch_ssd_bound_ms_per_token'], '{:.3f}')}"
          f"   [indep. est {_f(a['ssd_bound_independent_estimate_ms'], '{:.3f}')} ms "
          f"= bytes/tok {_f(a['bytes_read_per_token'], '{:.0f}')} / {a['ssd_bandwidth_gibs']} GiB/s;"
          f" misses/tok={_f(a['misses_per_token'], '{:.2f}')} hit={_f(a['hit_rate'], '{:.3f}')}]")
    print(f"    of which sync/barrier (2)-(4) = {_f(a['switch_sync_barrier_ms_per_token'], '{:.3f}')}"
          f"   (~40 host syncs/token; eval(indices) measured direct on (1) = "
          f"{_f(a.get('eval_indices_ms_per_token_measured'), '{:.3f}')} ms/tok over "
          f"{_f(a.get('eval_indices_barriers_per_token'), '{:.1f}')} barriers/tok)")
    print(f"  small stages floor     (5)     = {_f(a['small_stages_floor_ms_per_token'], '{:.3f}')}")
    print("  " + "-" * 44)
    print(f"  sum of parts (attn+switch+floor) = {_f(a['sum_of_parts_ms_per_token'], '{:.3f}')}")
    print(f"  full (1)                         = {_f(a['full_ms_per_token'], '{:.3f}')}")
    print(f"  unattributed residual (1 - sum)  = {_f(a['unattributed_residual_ms_per_token'], '{:.3f}')}")


# ===========================================================================
# --in-model --unfenced --attn-subops: WITHIN-attention sub-op attribution (W97)
# ===========================================================================
# The 5-pass mode above attributes the whole 291 ms/token attention as (1)-(3).
# W97 (docs/deepseek-v41/W97_ATTENTION_291MS.md) found the attention MATH is tiny
# (337 MFLOP/layer, ~0.03 ms/layer of compute; the port is MLA-absorbed like the
# reference -- one shared 512-latent scored against all 64 heads, no per-head K/V
# up-projection) while the per-token TRAFFIC is dominated by the grouped o-LoRA
# ``wo_a`` weight: the port re-issues ``mx.dequantize(wo_a)`` every layer every
# token (304 MB f32 q8 / 420 MB mxfp4 per layer/token), which the isolated bf16
# bench (dense ``wo_a``, no dequant) never sees.  The remaining gap is the
# host-encode of attention's ~100 small dispatches, EXPOSED because the per-layer
# routing barrier drains the pipeline every layer (W96) so nothing hides them --
# the bench's tight loop pipelines them.
#
# This sub-mode locates the 291 ms WITHIN attention: it stubs ONE attention sub-op
# at a time (shape-preserving pass) on the real model and measures the unfenced
# whole-token frame wall, so full - <subop> is that sub-op's true in-situ cost
# (its own kernels PLUS the host-encode of its dispatches, which is the signal).
#
# Sub-ops (this port is ABSORBED, so the reference's "K/V up-projection per head"
# does not exist -- documented N/A; the shared-latent gather is inside attn_core):
#   qkv_proj      -- q down/up (wq_a,wq_b) + kv (wkv) projections zeroed.
#   attn_core     -- the selected-key gather + QK^T + sink softmax + PV (the whole
#                    _sparse_attend[_selected]) -> zeros.
#   wo_a_dequant  -- _o_lora_dense_weight precomputed once and returned: full - this
#                    = the per-token mx.dequantize(wo_a) DISPATCH cost.  NOTE (review
#                    item 6): the stub returns the PRE-ASTYPE array -- with the
#                    WO_A_CACHE lever off (the default here) _o_lora_dense_weight
#                    returns the bf16 dequant, and _o_lora_down still runs
#                    .astype(f32) EVERY token, so this stub is BLIND to the f32
#                    astype (the ~10.7 GB/token write+read the WO_A_CACHE lever, which
#                    caches the f32 array, also removes -- review item 1).  To bench
#                    the whole wo_a fix (dequant + astype) arm MTPLX_DSV41_ATTN_WO_A_CACHE
#                    on the arm instead of relying on this stub.
#   out_proj      -- the whole output projection (grouped wo_a einsum + wo_b) zeroed
#                    (includes the dequant); full - out_proj minus full - wo_a_dequant
#                    isolates the einsum+wo_b from the dequant.
#   rope          -- _rope_last (q rope + inverse output rope) -> identity.
#
# CONTAMINATION (review item 6): the zero-returning stubs (attn_core / qkv_proj /
# out_proj, and the inherited W94 _ZeroAttn) CHANGE the decoded tokens, so a stubbed
# pass routes DIFFERENT experts than full -> different misses / SSD bytes.  Since the
# unfenced frame wall is switch/SSD dominated, ``full - <subop>`` then mixes the
# sub-op's own cost with a changed switch workload.  Each pass therefore records its
# switch workload (misses/bytes per token, + the route-stage hot counters under
# MTPLX_ROUTE_STAGE_PROBE=1) and its token sha; the attribution flags any sub-op whose
# sha != full and prints the switch delta beside its cost so a contaminated row is
# visible, not silently credited to attention.
#
# Run with the attention compile tapes forced OFF: the compiled qkv/out tapes
# evaluate their weight-array inputs (``_lin_arrays``/``_o_lora_dense_weight``) at
# the _attend call site BEFORE the tape runs, so a builder stub could not skip that
# work -- the eager path is the faithful, patchable microscope, and its higher
# dispatch count is exactly what surfaces each sub-op's exposed host-encode.  The
# 5-pass compiled arm still owns the absolute 291 ms; this locates where it sits.
_ATTN_SUBOPS = ["qkv_proj", "attn_core", "wo_a_dequant", "out_proj", "rope"]

# W97 (review item 6): the per-layer expert-switch hot-path route-probe events whose
# per-pass delta attributes the switch workload (all_hit waves / split-route
# admissions / eval_indices routing barriers).  Populated only under
# MTPLX_ROUTE_STAGE_PROBE=1 (the probe adds per-bracket timing and would perturb the
# unfenced wall, so it is off by default; the misses/bytes per token are always on).
_ROUTE_STAGE_EVENTS = ("hot.all_hit", "hot.begin_split_route", "hot.eval_indices")


def _apply_attn_subop_stub(model, which: str):
    """Monkeypatch ONE attention sub-op to a shape-preserving pass on every backbone
    layer; returns a ``(obj, attr, orig)`` restore list (``_restore_stub`` restores
    it).  Caller must have forced ``dsv41._ATTN_COMPILE = False`` first (see above)."""
    # Class methods (not instance overrides) are restored by DELETING the instance
    # shadow the stub sets (revealing the class method again); submodules (wq_*, wo_b)
    # and the module-level _rope_last are restored to their original object.
    saved = []
    if which == "rope":
        saved.append((dsv41, "_rope_last", dsv41._rope_last))
        dsv41._rope_last = lambda x, cos, sin, inverse=False: x
        return saved
    for layer in model.layers:
        attn = layer.attn
        H, hd, g, olr, dim = (attn.n_heads, attn.head_dim, attn.n_groups,
                              attn.o_lora_rank, attn.dim)
        if which == "qkv_proj":
            qlr = int(attn.q_norm_weight.shape[0])
            for name, out in (("wq_a", qlr), ("wq_b", H * hd), ("wkv", hd)):
                saved.append((attn, name, getattr(attn, name)))  # submodule -> setattr
                setattr(attn, name, lambda x, out=out: mx.zeros(
                    (x.shape[0], x.shape[1], out), dtype=x.dtype))
        elif which == "attn_core":
            for name in ("_sparse_attend", "_sparse_attend_selected"):
                saved.append((attn, name, _RESTORE_DELETE))  # class method
            attn._sparse_attend = lambda q, KV, attend, H=H, hd=hd: mx.zeros(
                (q.shape[0], q.shape[1], H, hd), dtype=mx.float32)
            attn._sparse_attend_selected = lambda q, *a, H=H, hd=hd, **k: mx.zeros(
                (q.shape[0], q.shape[1], H, hd), dtype=mx.float32)
        elif which == "wo_a_dequant":
            w = attn._o_lora_dense_weight()   # dequantize ONCE (the fix's ceiling)
            mx.eval(w)
            saved.append((attn, "_o_lora_dense_weight", _RESTORE_DELETE))  # class method
            attn._o_lora_dense_weight = lambda w=w: w
        elif which == "out_proj":
            saved.append((attn, "_o_lora_down", _RESTORE_DELETE))  # class method
            attn._o_lora_down = lambda o, g=g, olr=olr: mx.zeros(
                (o.shape[0], o.shape[1], g, olr), dtype=mx.float32)
            saved.append((attn, "wo_b", attn.wo_b))  # submodule -> setattr
            attn.wo_b = lambda o, dim=dim: mx.zeros(
                (o.shape[0], o.shape[1], dim), dtype=o.dtype)
        else:
            raise ValueError(which)
    return saved


def run_in_model_attn_subops(cfg: dict) -> dict:
    """Within-attention sub-op attribution: a full (eager-attention) baseline plus
    one pass per :data:`_ATTN_SUBOPS`, each with that sub-op stubbed.  Unfenced
    whole-token frame wall; ``full - <subop>`` is the sub-op's in-situ cost."""
    tiny = cfg.get("tiny", False)
    steps = int(cfg.get("steps", 8 if tiny else 64))
    warmup_steps = int(cfg.get("warmup_steps", 2 if tiny else 8))
    ssd_bw = float(cfg.get("ssd_bandwidth_gibs", 4.4) or 4.4)
    setup = _in_model_setup(cfg, steps)
    ab = setup["ab"]
    model, ops = setup["model"], setup["ops"]
    prompt_ids, prompt_meta = setup["prompt_ids"], setup["prompt_meta"]
    dims, device, arm = setup["dims"], setup["device"], setup["arm"]

    util_on = bool(cfg.get("utilization", False))
    util_interval_ms = int(cfg.get("util_interval_ms", 2000))

    def _run_pass(label):
        sampler = (ab._macmon().UtilizationSampler(interval_ms=util_interval_ms)
                   if util_on else None)
        # W97 (review item 6): the sub-op stubs (attn_core/qkv_proj/out_proj return
        # zeros) CHANGE the decoded tokens, so ``full - <subop>`` also folds in a
        # changed expert-streaming workload (different routing -> different misses /
        # SSD bytes; the unfenced frame wall is switch/SSD dominated).  Capture the
        # route-stage hot counters (all_hit / split_route / eval_indices) around this
        # pass so the switch workload is attributable -- best-effort, populated only
        # under MTPLX_ROUTE_STAGE_PROBE=1 (the probe adds per-bracket timing, so it is
        # off by default and would perturb the wall; the decode-scoped misses/bytes
        # per token in the summary are the always-available switch signal).
        try:
            from mtplx import expert_route_probe as _rp
            _rp_enabled = bool(getattr(_rp, "ENABLED", False))
            _before = ({k: _rp.peek(k) for k in _ROUTE_STAGE_EVENTS}
                       if _rp_enabled else None)
        except Exception:
            _rp, _rp_enabled, _before = None, False, None
        data = _unfenced_decode_pass(
            ab=ab, model=model, ops=ops, prompt_ids=prompt_ids, steps=steps,
            warmup_steps=warmup_steps, cooldown_s=0.0, util_sampler=sampler,
        )
        route_stage = None
        if _rp_enabled and _before is not None:
            route_stage = {k: int(_rp.peek(k) - _before[k]) for k in _ROUTE_STAGE_EVENTS}
        util_summary = sampler.summarize() if sampler is not None else None
        if sampler is not None:
            print(f"[w97-subops] {label}: {sampler.census()}", flush=True)
        summary = _unfenced_pass_summary(data, util_summary, ssd_bandwidth_gibs=ssd_bw)
        return {
            "label": label,
            "summary": summary,
            "utilization": util_summary,
            "token_ids_sha256": hashlib.sha256(
                json.dumps(data["generated"]).encode()).hexdigest(),
            "n_token_ids": len(data["generated"]),
            # The switch workload this pass actually ran (item 6): the decode-scoped
            # miss/byte counters are the always-available signal; ``route_stage`` (the
            # hot.all_hit / split_route / eval_indices deltas) is present only with the
            # route probe armed.  full vs a stub differing here means ``full - stub``
            # is contaminated by a changed switch/SSD workload, not pure attention.
            "switch_workload": {
                "misses_per_token": summary.get("misses_per_token"),
                "bytes_read_per_token": summary.get("bytes_read_per_token"),
                "hit_rate": summary.get("hit_rate"),
                "route_stage": route_stage,
                "route_probe_enabled": _rp_enabled,
            },
        }

    # Force the attention compile tapes OFF for the whole sub-op microscope so every
    # sub-op is a reachable eager patch; restore the caller's setting afterwards.
    saved_compile = dsv41._ATTN_COMPILE
    dsv41._ATTN_COMPILE = False
    passes: Dict[str, dict] = {}
    try:
        passes["full"] = _run_pass("full_eager_attention")
        for sub in _ATTN_SUBOPS:
            st = _apply_attn_subop_stub(model, sub)
            try:
                passes[sub] = _run_pass(sub + "_stubbed")
            finally:
                _restore_stub(st)
    finally:
        dsv41._ATTN_COMPILE = saved_compile

    def _ms(key):
        return (passes.get(key, {}).get("summary") or {}).get("mean_ms_per_token")

    full = _ms("full")
    full_sha = passes.get("full", {}).get("token_ids_sha256")
    full_sw = (passes.get("full", {}).get("switch_workload") or {})

    def _dsw(sub_sw, key):
        a, b = sub_sw.get(key), full_sw.get(key)
        return (a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None

    attribution = {"full_eager_ms_per_token": full, "subops": {}}
    for sub in _ATTN_SUBOPS:
        v = _ms(sub)
        sub_sha = passes.get(sub, {}).get("token_ids_sha256")
        sub_sw = (passes.get(sub, {}).get("switch_workload") or {})
        # W97 (review item 6): the stub changed the tokens iff its sha != full's;
        # when it did, the ``cost`` also includes a changed switch/SSD workload, so
        # record the switch delta (misses/bytes per token) next to the cost.
        attribution["subops"][sub] = {
            "stubbed_ms_per_token": v,
            "cost_ms_per_token": (full - v) if isinstance(full, (int, float))
            and isinstance(v, (int, float)) else None,
            "token_sha_differs_from_full": (sub_sha != full_sha),
            "switch_delta": {
                "d_misses_per_token": _dsw(sub_sw, "misses_per_token"),
                "d_bytes_read_per_token": _dsw(sub_sw, "bytes_read_per_token"),
            },
        }

    return {
        "worker": "W97",
        "script": "scripts/deepseek_v41/metal_decode_attn_bisect.py",
        "mode": "in_model_unfenced_attn_subops",
        "device": device, "tiny": tiny, "arm": arm, "steps": steps,
        "warmup_steps": warmup_steps, "ssd_bandwidth_gibs": ssd_bw,
        "dims": dims, "prompt": prompt_meta,
        "note": "attention compile tapes forced OFF for the sub-op microscope; "
                "the 5-pass compiled arm owns the absolute attention (1)-(3).",
        "memory": {"active_end_gib": _mem_gib("get_active_memory"),
                   "peak_gib": _mem_gib("get_peak_memory")},
        "utilization": passes["full"].get("utilization"),
        "passes": passes,
        "attribution": attribution,
    }


def print_in_model_attn_subops(receipt: dict) -> None:
    print(f"\n=== W97 --in-model --unfenced --attn-subops [{receipt['device']}, "
          f"arm={receipt['arm']}, steps={receipt['steps']}] ===")
    print(f"dims: {receipt['dims']}   {receipt['note']}")

    def _f(v, fmt):
        return fmt.format(v) if isinstance(v, (int, float)) else "-"

    hdr = "pass".ljust(24) + f"{'ms/tok(mean)':>13}{'busy%':>8}{'MHz':>8}"
    print("\n-- unfenced whole-token frame wall, attention eager --")
    print(hdr)
    print("-" * len(hdr))
    for key in ["full"] + _ATTN_SUBOPS:
        s = receipt["passes"][key]["summary"]
        print(("full" if key == "full" else key + "-stub").ljust(24)
              + f"{_f(s.get('mean_ms_per_token'), '{:.3f}'):>13}"
              + f"{_f(s.get('gpu_busy_pct'), '{:.0f}'):>8}"
              + f"{_f(s.get('gpu_freq_mhz'), '{:.0f}'):>8}")
    a = receipt["attribution"]
    full_sha = (receipt["passes"].get("full") or {}).get("token_ids_sha256")
    print("\n-- within-attention sub-op cost (full - <subop stubbed>, ms/token) --")
    print(f"  full (eager attention)   = {_f(a['full_eager_ms_per_token'], '{:.3f}')}"
          f"   [token sha {str(full_sha)[:10]}]")
    print("  cost / Δmiss/tok / Δbytes/tok vs full  (item 6: a stub that CHANGED the "
          "tokens also changed the switch/SSD workload -> cost is contaminated)")
    for sub in _ATTN_SUBOPS:
        d = a["subops"][sub]
        sw = d.get("switch_delta") or {}
        changed = d.get("token_sha_differs_from_full")
        flag = "  <-- TOKENS CHANGED: cost includes a changed switch workload" if changed else ""
        print(f"  {sub:16s} cost = {_f(d['cost_ms_per_token'], '{:.3f}')}"
              f"   Δmiss/tok={_f(sw.get('d_misses_per_token'), '{:+.2f}')}"
              f"   Δbytes/tok={_f(sw.get('d_bytes_read_per_token'), '{:+.3e}')}"
              f"{flag}")
    print("  (out_proj cost - wo_a_dequant cost = the grouped wo_a einsum + wo_b; "
          "K/V up-projection N/A -- this port is MLA-absorbed)")
    print("  (wo_a_dequant stub returns the PRE-ASTYPE bf16 array, so its cost is the "
          "per-token dequant DISPATCH only -- NOT the f32 astype; arm "
          "MTPLX_DSV41_ATTN_WO_A_CACHE to bench the whole wo_a fix)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """The in-model / microbench CLI parser (extracted from ``main`` so tests can
    parse the exact launcher argv without running a pass)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", action="store_true",
                   help="run on the Metal GPU (default: CPU, so a worker run is safe)")
    p.add_argument("--tiny", action="store_true",
                   help="tiny dims + CPU + T in {256,1024} (the unit-test path)")
    p.add_argument("--no-selected-keys", action="store_true",
                   help="masked-full path (score+astype over the whole T) instead of K30")
    p.add_argument("--no-win-memo", action="store_true",
                   help="turn off MTPLX_DSV41_ATTN_WIN_MEMO")
    p.add_argument("--compare-no-compile", action="store_true",
                   help="turn off MTPLX_DSV41_ATTN_COMPILE (eager qkv/out projections)")
    p.add_argument("--ballast-gib", type=float, default=0.0,
                   help="hold N GiB of resident Metal buffers for the whole run "
                        "(reproduce the loaded model's allocator/residency pressure)")
    p.add_argument("--ballast-churn", action="store_true",
                   help="also alloc+free a ~16 MB transient per step between ops "
                        "(mimic the per-token [1,T,512] concat churn)")
    p.add_argument("--T", type=int, nargs="+", default=None,
                   help="override the T sweep (default 1024 4096 16384; tiny 256 1024)")
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    # W102 --fence-per-layer (isolated mode): the pipelined-vs-per-layer-barrier
    # ms/layer discriminator (H1 GPU per-kernel overhead vs H2 exposed host-encode).
    p.add_argument("--fence-per-layer", action="store_true",
                   help="isolated mode: also run the full-depth attention pipeline "
                        "once unfenced (one eval/step, pipelined) and once fenced "
                        "(the served per-layer routing barrier after each layer), and "
                        "add isolated_ms_per_layer_unfenced / _fenced + fence_kind to "
                        "the receipt. Additive: the census output is unchanged. Context "
                        "from --context-tokens (default 16384; tiny 256).")
    p.add_argument("--fence-kind", choices=list(_FENCE_KINDS), default="eval_tolist",
                   help="the per-layer host sync for --fence-per-layer: 'eval' = "
                        "mx.eval(out) only (the eval-only barrier); 'eval_tolist' = "
                        "mx.eval + a tiny dependent .tolist() readback, mirroring the "
                        "served route mx.eval(indices)+indices.reshape(-1).tolist() "
                        "(default eval_tolist)")
    p.add_argument("--fence-steps", type=int, default=_FENCE_DEFAULT_STEPS,
                   help="--fence-per-layer: timed decode steps per loop (default 64)")
    p.add_argument("--fence-warmup", type=int, default=_FENCE_DEFAULT_WARMUP,
                   help="--fence-per-layer: warmup steps excluded from both ms/layer "
                        "numbers (default 8)")
    # --in-model (window-32 follow-up): peel on the REAL loaded model, live cache.
    p.add_argument("--in-model", action="store_true",
                   help="measure the attention ops in situ on the real loaded model "
                        "(three passes: full / expert-stub / attn-stub). GPU-window "
                        "mode; --tiny validates the plumbing on a fake CPU model.")
    p.add_argument("--unfenced", action="store_true",
                   help="--in-model: run the FIVE unfenced whole-token frame-wall "
                        "passes (window 94) instead of the three fenced per-stage "
                        "passes -- no stage recorder, no per-stage fences (only the "
                        "production per-token sampler eval); delta full-stubbed is the "
                        "component's true in-situ cost. GPU-window mode; --tiny validates.")
    p.add_argument("--attn-subops", action="store_true",
                   help="--in-model --unfenced: WITHIN-attention sub-op attribution "
                        "(W97) instead of the 5 component passes -- a full (eager-"
                        "attention) baseline plus one unfenced pass per attention "
                        "sub-op stubbed (qkv_proj / attn_core / wo_a_dequant / "
                        "out_proj / rope); full-<subop> is that sub-op's in-situ cost. "
                        "Attention compile tapes forced OFF. GPU-window; --tiny validates.")
    p.add_argument("--warmup-steps", type=int, default=None,
                   help="--in-model --unfenced: decode steps excluded as warmup from "
                        "the mean/median ms/token (default 8 gpu / 2 tiny)")
    p.add_argument("--ssd-bandwidth-gibs", type=float, default=4.4,
                   help="--in-model --unfenced: SSD read bandwidth (GiB/s) for the "
                        "independent SSD-bound estimate bytes/tok / BW (default 4.4, the "
                        "measured M5 Max SSD threshold; the receipt carries misses+bytes "
                        "so any bandwidth can be re-applied)")
    p.add_argument("--model", type=str, default=None,
                   help="--in-model --gpu: the streaming artifact path "
                        "(default: ~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4)")
    p.add_argument("--arms", type=str, default="cell16k",
                   help="--in-model: the ab_decode_env_levers arm to apply (default cell16k)")
    p.add_argument("--prompt-ids-file", type=str, default=None,
                   help="--in-model --gpu: the exported server prompt ids fixture")
    p.add_argument("--prompt-seed", type=int, default=None,
                   help="--in-model --gpu: which seed's ids from --prompt-ids-file")
    p.add_argument("--context-tokens", type=int, default=16384,
                   help="--in-model --gpu: prefill length (default 16384)")
    p.add_argument("--memory-limit-gib", type=float, default=60.0,
                   help="--in-model --gpu: active-allocation memory limit (default 60)")
    p.add_argument("--max-kv", type=int, default=17408,
                   help="--in-model --gpu: max live KV tokens (default 17408)")
    p.add_argument("--in-model-steps", type=int, default=None,
                   help="--in-model: decode steps per pass (fenced default 30 gpu / 4 "
                        "tiny; unfenced default 64 gpu / 8 tiny)")
    # W90: post-prefill cooldown + macmon utilization telemetry (--in-model).
    p.add_argument("--cooldown-s", type=float, default=0.0,
                   help="--in-model: idle N s AFTER prefill, BEFORE the timed decode "
                        "(TTFT unaffected); window-36 runs 180 vs 0 as the discriminator")
    p.add_argument("--utilization", action="store_true",
                   help="--in-model: sample macmon (gpu freq/power/busy, temps) over the "
                        "timed decode into a 'utilization' block + one-line census (W90)")
    p.add_argument("--util-interval-ms", type=int, default=2000,
                   help="macmon sampling interval in ms for --utilization (default 2000)")
    p.add_argument("--out", type=str, default=None, help="write the JSON receipt here")
    return p


def _in_model_cfg(a) -> dict:
    """Build the in-model run cfg from parsed args (shared by ``main`` + tests)."""
    cfg = {
        "tiny": a.tiny,
        "arms": a.arms,
        "model": a.model or os.path.expanduser(
            "~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4"),
        "prompt_ids_file": a.prompt_ids_file,
        "prompt_seed": a.prompt_seed,
        "context_tokens": a.context_tokens,
        "memory_limit_gib": a.memory_limit_gib,
        "max_kv": a.max_kv,
        "cooldown_s": a.cooldown_s,
        "utilization": a.utilization,
        "util_interval_ms": a.util_interval_ms,
        "unfenced": getattr(a, "unfenced", False),
        "attn_subops": getattr(a, "attn_subops", False),
        "ssd_bandwidth_gibs": getattr(a, "ssd_bandwidth_gibs", 4.4),
    }
    if a.in_model_steps is not None:
        cfg["steps"] = a.in_model_steps
    if getattr(a, "warmup_steps", None) is not None:
        cfg["warmup_steps"] = a.warmup_steps
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    p = build_parser()
    a = p.parse_args(argv)

    if getattr(a, "unfenced", False) and not a.in_model:
        p.error("--unfenced is an --in-model mode; pass --in-model --unfenced")
    if getattr(a, "attn_subops", False) and not (a.in_model and getattr(a, "unfenced", False)):
        p.error("--attn-subops is an --in-model --unfenced mode; pass all three")

    if a.in_model:
        if not (a.gpu or a.tiny):
            p.error("--in-model requires --gpu (real model) or --tiny (CPU fake model)")
        if a.gpu and a.tiny:
            p.error("--in-model: choose --gpu OR --tiny, not both")
        cfg = _in_model_cfg(a)
        if getattr(a, "attn_subops", False):
            receipt = run_in_model_attn_subops(cfg)
            print_in_model_attn_subops(receipt)
        elif getattr(a, "unfenced", False):
            receipt = run_in_model_unfenced(cfg)
            print_in_model_unfenced(receipt)
        else:
            receipt = run_in_model(cfg)
            print_in_model(receipt)
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            with open(a.out, "w") as f:
                json.dump(receipt, f, indent=2, default=str)
            print(f"\nreceipt -> {a.out}")
        return 0

    cfg = {
        "tiny": a.tiny,
        "gpu": a.gpu,
        "use_selected": not a.no_selected_keys,
        "win_memo": not a.no_win_memo,
        "compile": not a.compare_no_compile,
        "ballast_gib": a.ballast_gib,
        "ballast_churn": a.ballast_churn,
        "fence_per_layer": a.fence_per_layer,
        "fence_kind": a.fence_kind,
        "fence_steps": a.fence_steps,
        "fence_warmup": a.fence_warmup,
        "context_tokens": a.context_tokens,
        "utilization": a.utilization,
        "util_interval_ms": a.util_interval_ms,
    }
    if a.T is not None:
        cfg["Ts"] = a.T
    if a.iters is not None:
        cfg["iters"] = a.iters
    if a.warmup is not None:
        cfg["warmup"] = a.warmup

    receipt = run(cfg)
    print_tables(receipt)
    print_fence_probe(receipt)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(receipt, f, indent=2, default=str)
        print(f"\nreceipt -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
