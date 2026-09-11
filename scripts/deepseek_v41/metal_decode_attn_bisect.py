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

Default device is **CPU** (so an accidental worker run never touches the Metal
GPU during a benchmark window); pass ``--gpu`` in the exclusive GPU window.  The
``--tiny`` mode (tiny dims, CPU, T in {256,1024}) is the unit-test path that proves
the script executes and the peeled ops sum to within 20% of the whole.

Budget on the GPU window: <= 3 min total, <= 6 GB (one layer per mode, built and
freed sequentially; a 16K real-dim cache is ~120 MB/layer).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
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
    t0 = time.perf_counter_ns()
    if use_selected:
        o = attn._sparse_attend_selected(q, window_all, sel_compress_kv, sel_comp_idx, positions)
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


def measure_mode_T(args: ModelArgs, attn: Attention, mode: str, T: int,
                   use_selected: bool, iters: int, warmup: int,
                   churn: bool = False) -> Dict[str, float]:
    """Time whole + peeled for one (mode, T).  whole and peeled use freshly
    pre-filled caches so each runs a genuine decode step (the step mutates the
    cache); returns ms/step per op + the whole."""
    # whole
    cache, shared, x, pos = build_case(args, attn, mode, T)
    for _ in range(warmup):
        decode_step_whole(attn, cache, shared, x, pos, churn=churn)
    whole_ns = 0
    for _ in range(iters):
        whole_ns += decode_step_whole(attn, cache, shared, x, pos, churn=churn)
    whole_ms = whole_ns / 1e6 / iters
    del cache, shared, x, pos

    # peeled
    cache, shared, x, pos = build_case(args, attn, mode, T)
    twarm: Dict[str, int] = {k: 0 for k in _ALL_KEYS}
    for _ in range(warmup):
        decode_step_peeled(attn, cache, shared, x, pos, use_selected, twarm, churn=churn)
    t: Dict[str, int] = {k: 0 for k in _ALL_KEYS}
    for _ in range(iters):
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
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
    p.add_argument("--out", type=str, default=None, help="write the JSON receipt here")
    a = p.parse_args(argv)

    cfg = {
        "tiny": a.tiny,
        "gpu": a.gpu,
        "use_selected": not a.no_selected_keys,
        "win_memo": not a.no_win_memo,
        "compile": not a.compare_no_compile,
        "ballast_gib": a.ballast_gib,
        "ballast_churn": a.ballast_churn,
    }
    if a.T is not None:
        cfg["Ts"] = a.T
    if a.iters is not None:
        cfg["iters"] = a.iters
    if a.warmup is not None:
        cfg["warmup"] = a.warmup

    receipt = run(cfg)
    print_tables(receipt)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(receipt, f, indent=2, default=str)
        print(f"\nreceipt -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
