#!/usr/bin/env python3
"""W116 -- DSpark verify-attention ROWS census for DeepSeek-V4.1-Flash.

Window-43 (16K cell, DSpark depth 5) measured the K+1 verify's attention at ~372
ms/cycle = ~9.3 ms/layer for the ``rows = K+1 = 6`` verify batch, vs the AR M=1
in-model attention ~1.6 ms/layer -- a ~6x premium.  The selected bytes are tiny
(k = sliding_window 128 + index_topk 512 = 640 keys x hd 512 x 2 B x 6 rows x 40
layers ~= 157 MB/cycle ~= 0.3 ms at 500 GB/s), so the 6x is NOT bandwidth -- it is
per-row gather / dispatch / core work in the small-M gathered attention core
(:meth:`Attention._sparse_attend_selected`, ``mtplx/models/deepseek_v41.py``):
each of the K+1 rows is handed its OWN gathered ``[k, hd]`` KV operand (the K29
kernel maps ``row -> its own [k,hd]``), so the core reads ``rows x k`` keys even
though the rows' windows overlap by ``W - 1`` and their compressed selections are
nearly identical.

This probe reconstructs ONE isolated production ``Attention`` layer at the exact
cell shape (T = 16,384 KV, sliding window 128, index_topk 512, head_dim 512, the
real 64-head config, mxfp8 gs32 projections like W105) and, for
``rows in {1, 2, 4, 6, 8}``, measures -- fenced (``mx.eval`` per op) AND pipelined
(one closing ``mx.eval`` over a chain, per-call), median of >= 7 -- each sub-op of
the verify path in execution order:

  1. ``qkv_proj``   -- the W101 fused (rows <= 8) / compiled / eager QKV projection
                       chain (rmsnorm + RoPE + head layout around the mxfp8 matmuls).
  2. ``select``     -- the indexer top-k over the index lane for each row
                       (:meth:`Indexer.select` + :func:`_mask_to_topk_idx`), on the
                       index-source modes; the reuse mode just reads the source's
                       published selection (~0).
  3. ``gather``     -- the window + selected-compressed key gather (KVg build):
                       :meth:`_window_selected_idx` + :func:`_gather_rows` (window)
                       + :func:`_selected_compress_gather` (compressed) + concat.
  4. the attention core THREE ways on the compact ``[rows, k, hd]`` operand:
       ``core_k29``  -- (a) W60/K29 fused decode kernel (each row its own S=1 batch).
       ``core_eager``-- (b) the eager gathered f32 einsum core
                        (:func:`_attn_core_impl`, the shipped eager block).
       ``core_sdpa`` -- (c) ``mx.fast.scaled_dot_product_attention`` on the gathered
                        ``[rows, k, hd]`` operand with the per-row valid mask + the
                        per-head value-0 sink (``sinks=``) -- a "what a batched
                        kernel would cost" proxy for the per-row layout.
  5. ``out_proj``   -- the W101 fused / compiled / eager output chain (query-RoPE
                       removal + grouped o-LoRA down (cached ``wo_a``) + ``wo_b`` up).

For every sub-op at every ``rows`` it reports fenced + pipelined ms, the scaling
ratio vs ``rows=1``, and the MLX-graph node count (``mx.export_to_dot`` inspection;
flat node count + rising time => same dispatch, more per-row work).  It derives the
implied ms/cycle for 40 layers at ``rows=6``, and a ``k_union`` accounting -- the
union of the rows' selected keys vs ``rows x k`` -- that quantifies the read
amplification a shared-tile W117 kernel removes.

Default device is **CPU** (an accidental worker run never touches Metal during a
benchmark window); ``--gpu`` runs the real cell in the exclusive GPU window,
``--cpu-smoke`` runs the tiny-dims plumbing check (T=256, k=32, hd=32) the unit
test drives.  No model load, no artifact -- one random-weight layer per mode, built
and freed sequentially; peak GPU memory << 20 GiB (see the docstring accounting in
docs/deepseek-v41/W116_VERIFY_ATTN_ROWS_CENSUS.md).

GPU window command (queued behind the flock, like every Metal exec on this box):

    GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((24*1024**3)) \\
      bash scripts/deepseek_v41/gpu_window.sh nice -n 19 \\
      .venv/bin/python3 scripts/deepseek_v41/verify_attn_rows_census.py \\
      --gpu --out <receipt>.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

# Arm the cell16k attention env the isolated bench freezes at import BEFORE importing
# the model module (``_ATTN_COMPILE`` / ``_ATTN_WIN_MEMO`` are frozen as module
# globals at import; the rest are read at use).  setdefault: a caller env still wins.
# Fused projections + lean casts are the W101/W99 verify-path arms the cell runs, so
# the projection sub-op is measured on the served small-M path.
_CELL16K_ENV = {
    "MTPLX_DSV41_SELECTED_KEYS": "1",
    "MTPLX_DSV41_ATTN_COMPILE": "1",
    "MTPLX_DSV41_ATTN_WIN_MEMO": "1",
    "MTPLX_DSV41_KV_CHUNK_GROW": "1",
    "MTPLX_DSV41_SELECT_FENCE": "1",
    "MTPLX_DSV41_ATTN_FUSED_PROJ": "1",
    "MTPLX_DSV41_ATTN_LEAN_CASTS": "1",
}
for _k, _v in _CELL16K_ENV.items():
    os.environ.setdefault(_k, _v)

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

# Reuse the isolated bench's layer / case builders (the DENSE-bf16 baseline) so this
# probe builds the exact production ``Attention`` methods.  The worktree root must
# lead sys.path so ``import mtplx`` resolves to THIS worktree's deepseek_v41 (the
# editable install points at the main checkout -- editable-install CWD shadowing).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKTREE_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
for _p in (_SCRIPT_DIR, _WORKTREE_ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _WORKTREE_ROOT)
import metal_decode_attn_bisect as bench  # noqa: E402

from mtplx.models import deepseek_v41 as dsv41  # noqa: E402
from mtplx.models.deepseek_v41 import ModelArgs, _mask_to_topk_idx  # noqa: E402
from mtplx.models import deepseek_v41_attn_kernels as _k29  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
#: The verify-batch row counts to sweep: M=1 (AR decode control) .. K+1=8 (MTP
#: depth up to 7).  6 = the window-43 DSpark depth-5 verify (primary + d1..d5).
_ROWS = [1, 2, 4, 6, 8]

#: Sub-op keys in execution order.  The three cores are attribution rows on the SAME
#: gathered operand (they are alternatives, not summed).
_SUBOPS = ["qkv_proj", "select", "gather", "core_k29", "core_eager", "core_sdpa",
           "out_proj"]
_CORES = ["core_k29", "core_eager", "core_sdpa"]

_MODES = list(bench._MODES)  # swa_only, full, reindex, reuse


def _smoke_args() -> ModelArgs:
    """Tiny-dims config that still derives all four CSA modes and hits the task's
    plumbing target k=32 (window 16 + index_topk 16), hd=32, at T=256.  Same
    source-id shape as the DSV4.1 tests (swa=0, full=2, reindex=6, reuse=3)."""
    return ModelArgs(
        num_hidden_layers=8,
        hidden_size=128,
        num_attention_heads=4,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=64,
        o_lora_rank=32,
        o_groups=2,
        sliding_window=16,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=16,
        compress_ratios=[0, 0, 2, 2, 2, 1, 1, 1],
        compress_rope_theta=160000.0,
        kv_source_layer_ids=[2, 5],
        index_source_layer_ids=[2, 5, 6],
        candidate_source_layer_id=-1,
    )


# The five attention projections nn.quantize targets (matches W105 / the artifact
# resident layout: attention projections mxfp8 gs32, indexer/compressor left bf16).
_PROJ_NAMES = {"wq_a", "wq_b", "wkv", "wo_a", "wo_b"}


def _quantize_attn_projections(attn, codec: str) -> dict:
    """Quantize the five attention projections in place to ``codec`` (``"mxfp8"``
    gs32 = the artifact codec; ``"bf16"`` = dense, no quant), leaving the bf16
    indexer/compressor projections dense -- exactly the in-model resident layout
    W105 priced.  Returns the projection-byte accounting."""
    if codec == "bf16":
        total = sum(int(getattr(attn, pn).weight.nbytes) for pn in _PROJ_NAMES)
        return {"proj_bytes": total, "codec": "bf16", "quantized": False}
    spec = {"mxfp8": (8, 32, "mxfp8"), "mxfp4": (4, 32, "mxfp4")}[codec]
    bits, gs, mode = spec

    def predicate(path: str, module) -> bool:
        return isinstance(module, nn.Linear) and path in _PROJ_NAMES

    nn.quantize(attn, group_size=gs, bits=bits, mode=mode, class_predicate=predicate)
    mx.eval(attn.parameters())
    total = 0
    qnames = []
    for pn in _PROJ_NAMES:
        lin = getattr(attn, pn)
        if isinstance(lin, nn.QuantizedLinear):
            qnames.append(pn)
            total += int(lin.weight.nbytes) + int(lin.scales.nbytes)
            bset = lin.get("biases")
            if bset is not None:
                total += int(bset.nbytes)
        else:
            total += int(lin.weight.nbytes)
    return {"proj_bytes": total, "codec": codec, "quantized": True,
            "quantized_names": sorted(qnames)}


# ---------------------------------------------------------------------------
# Production sub-op dispatch (mirrors Attention._attend / _sparse_attend_selected)
# ---------------------------------------------------------------------------
def _qkv(attn, x, qcos, qsin, b, s):
    """The verify QKV projection chain, dispatched exactly as ``_attend`` does:
    the W101 fused kernels (GPU, rows <= 8), else the compiled prep tape, else
    eager.  Returns (q, qr, kv_new)."""
    H, hd = attn.n_heads, attn.head_dim
    if dsv41._fused_proj_use(b * s):
        return attn._qkv_prep_fused(x, qcos, qsin, b, s, H, hd)
    if dsv41._attn_use_compile(b * s):
        return dsv41._attn_qkv_prep(attn)(
            x, qcos, qsin, attn.q_norm_weight, attn.kv_norm_weight,
            *dsv41._lin_arrays(attn.wq_a), *dsv41._lin_arrays(attn.wq_b),
            *dsv41._lin_arrays(attn.wkv),
        )
    qr = dsv41._rmsnorm(attn.wq_a(x), attn.q_norm_weight, attn.eps)
    q = attn.wq_b(qr).reshape(b, s, H, hd)
    q = dsv41._rope_last(q, qcos, qsin)
    kv_new = dsv41._rmsnorm(attn.wkv(x), attn.kv_norm_weight, attn.eps)
    kv_new = dsv41._rope_last(kv_new, qcos, qsin)
    return q, qr, kv_new


def _out(attn, o, qcos, qsin, b, s):
    """The verify output chain, dispatched exactly as ``_attend`` does."""
    if dsv41._fused_proj_use(b * s):
        return attn._out_prep_fused(o, qcos, qsin, b, s)
    if dsv41._attn_use_compile(b * s):
        return dsv41._attn_out_prep(attn)(
            o, qcos, qsin, attn._o_lora_dense_weight(), *dsv41._lin_arrays(attn.wo_b)
        )
    o = dsv41._rope_last(o, qcos, qsin, inverse=True)
    o = o.reshape(b, s, attn.n_groups, -1)
    o = attn._o_lora_down(o)
    return attn.wo_b(o.reshape(b, s, -1))


def _select(attn, x, qr, positions, qcos, qsin, shared):
    """The indexer top-k selection sub-op (mirrors ``_compressed``'s select block).
    Index-source modes score all ``n_comp`` compressed rows and argsort to gather
    indices; reuse reads the source's published ``selected_idx``.  Padded to the
    FULL ``index_topk`` (the K29 / core-compile fixed-k consumer), so k is
    context-independent."""
    ratio = attn.compress_ratio
    compress_kv = shared.compress_kv
    index_k = shared.index_k
    n_comp = compress_kv.shape[1]
    s = int(positions.shape[0])
    compress_lens = (positions + 1) // ratio
    if attn.is_index_source:
        set_c = attn.is_candidate_source
        cand = None if set_c else shared.candidates
        mask, _cand_out = attn.indexer.select(
            x, qr, index_k, qcos, qsin, compress_lens, n_comp,
            candidates=cand, set_candidates=set_c,
            cand_topk=attn.candidate_topk_blocks, cand_block=attn.candidate_block_size,
        )
        return _mask_to_topk_idx(mask, attn.indexer.index_topk)  # [b, s, index_topk]
    # Reuse layer: read the index source's published selection.  Built in isolation
    # here the pre-filled selection is [b, 1, k] (the decode-shaped seed from
    # build_case); the verify batch is s = rows, and every reuse row reads the same
    # source selection, so broadcast to [b, s, k] to match the gather's row count.
    sel = shared.selected_idx
    if sel is not None and int(sel.shape[1]) != s:
        sel = mx.broadcast_to(sel[:, :1, :], (int(sel.shape[0]), s, int(sel.shape[2])))
    return sel


def _gather(attn, window_all, sel_compress_kv, sel_comp_idx, positions, win_drop):
    """The window + selected-compressed key gather (KVg build), mirroring
    :meth:`_sparse_attend_selected`.  ``shared=None`` forces the plain per-layer
    gather (no cross-layer caching), so the pipelined chain re-issues it every
    iteration.  Returns (KVg [b,s,k,hd], valid [b,s,k])."""
    b = 1
    s = int(positions.shape[0])
    win_idx, win_valid = attn._window_selected_idx(positions, window_all.shape[1], win_drop)
    win_idx = mx.broadcast_to(win_idx[None], (b, s, win_idx.shape[-1]))
    win_valid = mx.broadcast_to(win_valid[None], (b, s, win_valid.shape[-1]))
    kvg_win = dsv41._gather_rows(window_all, win_idx, win_valid)
    if sel_compress_kv is not None and sel_comp_idx is not None:
        kvg_cmp, comp_valid = dsv41._selected_compress_gather(
            sel_compress_kv, sel_comp_idx, None)
        KVg = mx.concatenate([kvg_win, kvg_cmp], axis=2)
        valid = mx.concatenate([win_valid, comp_valid], axis=2)
    else:
        KVg, valid = kvg_win, win_valid
    return KVg, valid


def _core_k29(attn, q, KVg, valid):
    """(a) W60/K29 fused decode kernel -- each of the b*s rows its own S=1 batch
    with its own gathered ``[k, hd]`` KV, exactly as ``_sparse_attend_selected``
    hands it when K29 is armed.  Called directly (env-independent)."""
    b, s, H, hd = q.shape
    k = KVg.shape[2]
    out = _k29.fused_decode_attention(
        q.reshape(b * s, 1, H, hd),
        KVg.reshape(b * s, k, hd),
        KVg.reshape(b * s, k, hd),
        attend=valid.reshape(b * s, k),
        attn_sink=attn.attn_sink,
        scale=attn.softmax_scale,
        T=k,
    )
    return out.reshape(b, s, H, hd)


def _core_eager(attn, q, KVg, valid):
    """(b) the eager gathered f32 einsum core -- the shipped eager block
    (:func:`_attn_core_impl`, byte-for-byte)."""
    return dsv41._attn_core_impl(q, KVg, valid, attn.attn_sink, attn.softmax_scale)


def _core_sdpa(attn, q, KVg, valid):
    """(c) ``mx.fast.scaled_dot_product_attention`` on the per-row gathered operand
    -- B = rows batches, N_q = H query heads, T_q = 1, N_kv = 1, T_kv = k -- with the
    per-row additive valid mask and the per-head value-0 sink (``sinks=``).  A "what
    a batched kernel would cost" proxy for the per-row layout (still reads rows x k;
    NOT the shared-tile W117 core).  Numerically a proxy only (MLX's SDPA softmax /
    sink reassociates vs the reference _k_sparse_attn core)."""
    b, s, H, hd = q.shape
    rows = b * s
    k = KVg.shape[2]
    q4 = q.reshape(rows, H, hd)[:, :, None, :]          # [rows, H, 1, hd]
    kv4 = KVg.reshape(rows, k, hd)[:, None, :, :]        # [rows, 1, k, hd]
    m = valid.reshape(rows, k)
    add_mask = mx.where(m[:, None, None, :], mx.array(0.0, mx.float32),
                        mx.array(float("-inf"), mx.float32))  # [rows,1,1,k]
    out = mx.fast.scaled_dot_product_attention(
        q4.astype(mx.float32), kv4.astype(mx.float32), kv4.astype(mx.float32),
        scale=attn.softmax_scale, mask=add_mask,
        sinks=attn.attn_sink.astype(mx.float32),
    )
    return out.reshape(b, s, H, hd)


# ---------------------------------------------------------------------------
# Case build + measurement
# ---------------------------------------------------------------------------
def _on_gpu() -> bool:
    """True only when the census is running on the Metal GPU (``--gpu``).  The metal
    cores gate on this so a CPU-pinned run never dispatches Metal."""
    try:
        return mx.metal.is_available() and mx.default_device() == mx.gpu
    except Exception:
        return False


def _inv_freq(attn):
    return attn._lean_inv_freq() if dsv41._resolve_attn_lean_casts() else attn.inv_freq


def build_verify_case(args: ModelArgs, attn, mode: str, T: int, rows: int) -> dict:
    """Build the fixed inputs for the verify sub-op census at ``rows`` query rows on
    a cache pre-filled to T tokens.  The rows' window KV is appended ONCE (untimed)
    so the window store holds T+rows physical rows -- the real verify geometry -- and
    every timed sub-op then runs on static inputs (no cache mutation in the timed
    loop, so no length drift across iters)."""
    cache, shared, _x1, _p1 = bench.build_case(args, attn, mode, T)
    b = 1
    x = (mx.random.normal((b, rows, args.hidden_size)) * 0.02).astype(mx.bfloat16)
    positions = mx.arange(T, T + rows)
    mx.eval(x, positions)
    qcos, qsin = dsv41._cos_sin(_inv_freq(attn), positions)
    mx.eval(qcos, qsin)

    q, qr, kv_new = _qkv(attn, x, qcos, qsin, b, rows)
    mx.eval(q, qr, kv_new)
    cache.append_window(kv_new)              # establish the T+rows verify window
    window_all = cache.window
    win_drop = cache.window_drop_offset
    mx.eval(window_all)

    sel_compress_kv = None
    sel_comp_idx = None
    if attn.compress_ratio and shared.compress_kv is not None:
        sel_comp_idx = _select(attn, x, qr, positions, qcos, qsin, shared)
        sel_compress_kv = shared.compress_kv
        if sel_comp_idx is not None:
            mx.eval(sel_comp_idx)

    KVg, valid = _gather(attn, window_all, sel_compress_kv, sel_comp_idx, positions, win_drop)
    mx.eval(KVg, valid)
    o = _core_eager(attn, q, KVg, valid)     # representative attention output for out_proj
    mx.eval(o)

    return {
        "cache": cache, "shared": shared, "x": x, "positions": positions,
        "qcos": qcos, "qsin": qsin, "q": q, "qr": qr, "kv_new": kv_new,
        "window_all": window_all, "win_drop": win_drop,
        "sel_compress_kv": sel_compress_kv, "sel_comp_idx": sel_comp_idx,
        "KVg": KVg, "valid": valid, "o": o, "b": b, "rows": rows,
        "k": int(KVg.shape[2]),
    }


def _k_union(attn, case: dict) -> dict:
    """Host-side read-amplification accounting: the UNION of the rows' selected keys
    (window physical rows + selected compressed rows) vs ``rows x k``.  A shared-tile
    W117 kernel reads ``k_union`` keys once; the per-row core reads ``rows x k``."""
    positions = case["positions"]
    window_all = case["window_all"]
    win_drop = case["win_drop"]
    rows = case["rows"]
    win_idx, win_valid = attn._window_selected_idx(positions, window_all.shape[1], win_drop)
    wi = win_idx.tolist()
    wv = win_valid.tolist()
    win_set = set()
    per_row_win = []
    for r in range(len(wi)):
        rowset = {wi[r][j] for j in range(len(wi[r])) if wv[r][j]}
        per_row_win.append(len(rowset))
        win_set |= rowset
    comp_set = set()
    per_row_comp = []
    sci = case["sel_comp_idx"]
    if sci is not None:
        ci = sci.tolist()[0] if sci.ndim == 3 else sci.tolist()
        for r in range(len(ci)):
            rowset = {ci[r][j] for j in range(len(ci[r])) if ci[r][j] >= 0}
            per_row_comp.append(len(rowset))
            comp_set |= rowset
    k_union = len(win_set) + len(comp_set)
    per_row_k = [(per_row_win[r] + (per_row_comp[r] if per_row_comp else 0))
                 for r in range(rows)]
    rows_times_k = sum(per_row_k)
    return {
        "k_union": int(k_union),
        "window_union": int(len(win_set)),
        "compress_union": int(len(comp_set)),
        "rows_times_k": int(rows_times_k),
        "per_row_k": per_row_k,
        "amplification": round(rows_times_k / k_union, 4) if k_union else None,
    }


def _op_count(out) -> Optional[int]:
    """MLX-graph node count for a lazy (un-eval'd) sub-op output (or list of
    outputs), via ``mx.export_to_dot``.  Counts DOT node definitions (``[label=``
    lines, not ``->`` edges).  Best-effort: returns ``None`` if graph export is
    unavailable."""
    outs = _aslist(out)
    s = ""
    try:  # documented form: a file PATH string
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".dot")
        os.close(fd)
        try:
            mx.export_to_dot(path, *outs)
            with open(path) as f:
                s = f.read()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    except Exception:
        s = ""
    if not s.strip():
        try:  # fallback: a file-like object
            import io
            buf = io.StringIO()
            mx.export_to_dot(buf, *outs)
            s = buf.getvalue()
        except Exception:
            return None
    if not s.strip():
        return None
    return sum(1 for ln in s.splitlines() if "[label=" in ln and "->" not in ln)


def _aslist(o):
    """A sub-op closure returns an array or a list/tuple of the arrays that must ALL
    be forced (e.g. qkv returns q, qr, kv_new -- kv_new is independent of q, so
    forcing q alone would skip the wkv KV projection)."""
    return list(o) if isinstance(o, (list, tuple)) else [o]


def _fenced_ms(fn: Callable[[float], object], repeats: int, warmup: int) -> float:
    """Median ms over ``repeats`` single fenced calls (``mx.eval`` per call, forcing
    every output of the sub-op)."""
    for _ in range(warmup):
        mx.eval(*_aslist(fn(0.0)))
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        outs = _aslist(fn(0.0))
        mx.eval(*outs)
        samples.append(time.perf_counter_ns() - t0)
    return statistics.median(samples) / 1e6


def _pipelined_ms(fn: Callable[[float], object], chain: int, repeats: int,
                  warmup: int) -> float:
    """Median per-call ms over ``repeats`` chains of ``chain`` calls with one
    closing ``mx.eval``.  Each call adds a DISTINCT perturbation to its input
    (``1e-6 * (i+1)`` -- distinct and well-represented even in bf16, so it is not
    canonicalised to ``+0`` and the ``chain`` outputs are distinct subgraphs) so
    MLX cannot CSE the chain into one submission while still sharing the common
    leaves the way real pipelining does (queued-vs-eager microbench).  Its <=1.6e-5
    magnitude is far below the inputs' ~0.02 scale, so it never changes a sub-op's
    output shape (top-k counts, gather widths) -- only the pipelining is measured."""
    def _perturb(i):
        return 1e-6 * (i + 1)

    def _chain_outs():
        outs = []
        for i in range(chain):
            outs.extend(_aslist(fn(_perturb(i))))
        return outs
    for _ in range(warmup):
        mx.eval(*_chain_outs())
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        outs = _chain_outs()
        mx.eval(*outs)
        samples.append((time.perf_counter_ns() - t0) / chain)
        del outs
    return statistics.median(samples) / 1e6


def _subop_closures(attn, case: dict) -> Dict[str, Callable[[float], object]]:
    """Build the perturbable closure for each sub-op (``fn(p)`` adds tiny ``p`` to
    one representative input, then runs the production sub-op)."""
    b, rows = case["b"], case["rows"]
    x, qr, positions = case["x"], case["qr"], case["positions"]
    qcos, qsin = case["qcos"], case["qsin"]
    q, o = case["q"], case["o"]
    window_all, win_drop = case["window_all"], case["win_drop"]
    sck, sci = case["sel_compress_kv"], case["sel_comp_idx"]
    KVg, valid = case["KVg"], case["valid"]
    shared = case["shared"]

    def add(a, p):
        return a if p == 0.0 else (a + p)

    cl: Dict[str, Callable[[float], object]] = {}
    # qkv returns (q, qr, kv_new) -- ALL forced (kv_new is independent of q).
    cl["qkv_proj"] = lambda p: list(_qkv(attn, add(x, p), qcos, qsin, b, rows))
    # gather returns (KVg, valid) -- both forced.
    cl["gather"] = lambda p: list(_gather(
        attn, add(window_all, p), None if sck is None else add(sck, p), sci,
        positions, win_drop))
    cl["core_eager"] = lambda p: _core_eager(attn, add(q, p), KVg, valid)
    cl["out_proj"] = lambda p: _out(attn, add(o, p), qcos, qsin, b, rows)
    # core_k29 (mx.fast.metal_kernel) and core_sdpa (the GPU-cost proxy) DISPATCH ON
    # METAL even when the default device is CPU (metal_kernel runs on the GPU whenever
    # Metal is available), so they are added ONLY on a GPU run.  On the CPU-pinned
    # census / unit test they are recorded absent (no Metal is ever touched) -- the
    # eager core carries the CPU plumbing check.
    if _on_gpu():
        cl["core_k29"] = lambda p: _core_k29(attn, add(q, p), KVg, valid)
        cl["core_sdpa"] = lambda p: _core_sdpa(attn, add(q, p), KVg, valid)
    # ``select`` is a sub-op only on index-source layers (full / reindex); reuse /
    # swa_only read the source's published selection (~0), so they carry no select.
    if attn.is_index_source and case["sel_compress_kv"] is not None:
        cl["select"] = lambda p: _select(
            attn, x, add(qr, p), positions, qcos, qsin, shared)
    return cl


def _probe(fn: Callable[[float], object]) -> Optional[str]:
    """Run one guarded call; return ``None`` if it evaluates, else a short reason.
    Lets a device-gated sub-op (K29 Metal-only; SDPA at an unsupported head_dim)
    be recorded as absent rather than crashing the census."""
    try:
        mx.eval(*_aslist(fn(0.0)))
        return None
    except Exception as e:  # noqa: BLE001 -- record the reason, don't propagate
        return f"{type(e).__name__}: {str(e)[:160]}"


def measure_mode_rows(args, attn, mode, T, rows, repeats, chain, warmup) -> dict:
    """Fenced + pipelined ms, op-count, and k_union accounting for every sub-op at
    ``rows`` query rows."""
    case = build_verify_case(args, attn, mode, T, rows)
    closures = _subop_closures(attn, case)
    subops: Dict[str, dict] = {}
    for name in _SUBOPS:
        fn = closures.get(name)
        if fn is None:
            if name in ("core_k29", "core_sdpa") and not _on_gpu():
                reason = "gpu-only (Metal core; skipped on CPU-pinned run)"
            elif name == "select":
                reason = "not-index-source (reads source selection)"
            else:
                reason = "not-applicable-for-mode"
            subops[name] = {"present": False, "reason": reason}
            continue
        reason = _probe(fn)
        if reason is not None:
            subops[name] = {"present": False, "reason": reason}
            continue
        subops[name] = {
            "present": True,
            "fenced_ms": round(_fenced_ms(fn, repeats, warmup), 5),
            "pipelined_ms": round(_pipelined_ms(fn, chain, repeats, warmup), 5),
            "op_count": _op_count(fn(0.0)),
        }
    out = {
        "rows": rows,
        "k": case["k"],
        "n_comp": (int(case["sel_compress_kv"].shape[1])
                   if case["sel_compress_kv"] is not None else 0),
        "k_union": _k_union(attn, case),
        "subops": subops,
    }
    # free the case (one layer's transients) before the next rows value
    for key in ("cache", "shared", "KVg", "valid", "q", "qr", "o", "window_all",
                "sel_compress_kv", "kv_new", "x"):
        case[key] = None
    del case, closures
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass
    return out


def _mem_gib(name: str):
    for mod in (mx, getattr(mx, "metal", None)):
        fn = getattr(mod, name, None) if mod is not None else None
        if fn is not None:
            try:
                return fn() / (1024 ** 3)
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(cfg: dict) -> dict:
    smoke = bool(cfg.get("cpu_smoke", False))
    gpu = bool(cfg.get("gpu", False)) and not smoke
    mx.set_default_device(mx.gpu if gpu else mx.cpu)

    # Re-sync the import-frozen dsv41 globals from the (setdefault-armed) env.
    def _envon(key):
        return (os.environ.get(key) or "").strip().lower() not in (
            "", "0", "false", "no", "off", "auto")
    dsv41._ATTN_COMPILE = _envon("MTPLX_DSV41_ATTN_COMPILE")
    dsv41._ATTN_WIN_MEMO = _envon("MTPLX_DSV41_ATTN_WIN_MEMO")

    args = _smoke_args() if smoke else bench.real_args()
    T = int(cfg.get("T", 256 if smoke else 16384))
    rows_list = list(cfg.get("rows", [1, 2, 4] if smoke else _ROWS))
    modes = list(cfg.get("modes", ["full"] if smoke else _MODES))
    codec = cfg.get("codec", "bf16" if smoke else "mxfp8")
    repeats = int(cfg.get("repeats", 3 if smoke else 9))
    chain = int(cfg.get("chain", 4 if smoke else 16))
    warmup = int(cfg.get("warmup", 1 if smoke else 3))

    try:
        mx.reset_peak_memory()
    except Exception:
        pass

    results: Dict[str, dict] = {}
    proj_acct: Dict[str, dict] = {}
    for mode in modes:
        attn = bench._build_layer(args, mode)
        proj_acct[mode] = _quantize_attn_projections(attn, codec)
        results[mode] = {}
        for rows in rows_list:
            results[mode][str(rows)] = measure_mode_rows(
                args, attn, mode, T, rows, repeats, chain, warmup)
        del attn
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

    receipt = {
        "worker": "W116",
        "script": "scripts/deepseek_v41/verify_attn_rows_census.py",
        "git_sha": _git_sha(),
        "device": "gpu" if gpu else "cpu",
        "cpu_smoke": smoke,
        "codec": codec,
        "dims": {
            "T": T, "head_dim": args.head_dim, "n_heads": args.num_attention_heads,
            "window": args.window_size, "index_topk": args.index_topk,
            "index_n_heads": args.index_n_heads, "index_head_dim": args.index_head_dim,
            "hidden": args.hidden_size, "n_layers": args.num_hidden_layers,
        },
        "mode_layer": bench._mode_layer_map(args),
        "rows": rows_list,
        "modes": modes,
        "repeats": repeats,
        "chain": chain,
        "warmup": warmup,
        "env": {k: os.environ.get(k) for k in _CELL16K_ENV},
        "proj_accounting": proj_acct,
        "results": results,
        "memory": {
            "active_end_gib": _mem_gib("get_active_memory"),
            "peak_gib": _mem_gib("get_peak_memory"),
        },
    }
    receipt["derived"] = _derive(receipt)
    return receipt


def _derive(receipt: dict) -> dict:
    """Per-sub-op scaling vs rows=1 and the implied 40-layer ms/cycle at rows=6,
    per mode, for the fenced core numbers."""
    n_layers = int(receipt["dims"]["n_layers"])
    out: Dict[str, dict] = {}
    for mode, per_rows in receipt["results"].items():
        row_keys = sorted(per_rows.keys(), key=lambda r: int(r))
        base = per_rows.get("1")
        scaling: Dict[str, dict] = {}
        for name in _SUBOPS:
            b1 = (base or {}).get("subops", {}).get(name, {}) if base else {}
            b1f = b1.get("fenced_ms") if b1.get("present") else None
            ratios = {}
            for rk in row_keys:
                e = per_rows[rk]["subops"].get(name, {})
                fv = e.get("fenced_ms") if e.get("present") else None
                if b1f is None or fv is None:
                    ratios[rk] = None
                elif b1f > 0:
                    ratios[rk] = round(fv / b1f, 3)
                else:  # rows=1 fenced below timer resolution -> self-ratio is 1.0
                    ratios[rk] = 1.0 if fv == b1f else None
            scaling[name] = {"fenced_ratio_vs_rows1": ratios,
                             "fenced_ms_rows1": b1f}
        # implied ms/cycle at rows=6 for the cheapest core + the fixed sub-ops
        r6 = per_rows.get("6")
        implied = {}
        if r6:
            so = r6["subops"]
            for core in _CORES:
                ce = so.get(core, {})
                cv = ce.get("fenced_ms") if ce.get("present") else None
                fixed = 0.0
                for name in ("qkv_proj", "select", "gather", "out_proj"):
                    e = so.get(name, {})
                    if e.get("present") and e.get("fenced_ms"):
                        fixed += e["fenced_ms"]
                if cv is not None:
                    implied[core] = round((cv + fixed) * n_layers, 3)
        out[mode] = {"scaling": scaling, "implied_ms_per_cycle_rows6_x40": implied}
    return out


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", _SCRIPT_DIR, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def print_tables(receipt: dict) -> None:
    d = receipt["dims"]
    print(f"\n=== W116 verify-attention ROWS census "
          f"[{receipt['device']}, codec={receipt['codec']}, T={d['T']}, "
          f"heads={d['n_heads']}x{d['head_dim']}, window={d['window']}, "
          f"index_topk={d['index_topk']}] ===")
    print(f"rows={receipt['rows']} repeats={receipt['repeats']} "
          f"chain={receipt['chain']} warmup={receipt['warmup']}   "
          f"peak={receipt['memory'].get('peak_gib')} GiB")
    rows_list = receipt["rows"]
    for mode in receipt["modes"]:
        per_rows = receipt["results"][mode]
        r0 = per_rows[str(rows_list[0])]
        print(f"\n-- {mode} (layer {receipt['mode_layer'][mode]})  "
              f"k={r0['k']} n_comp={r0['n_comp']} --  fenced ms / (pipelined ms) / ops")
        hdr = "sub-op".ljust(12) + "".join(f"{('rows=' + str(r)):>22}" for r in rows_list)
        print(hdr)
        print("-" * len(hdr))
        for name in _SUBOPS:
            cells = ""
            any_present = False
            for r in rows_list:
                e = per_rows[str(r)]["subops"].get(name, {})
                if e.get("present"):
                    any_present = True
                    txt = f"{e['fenced_ms']:.4f}/{e['pipelined_ms']:.4f}/{e['op_count']}"
                else:
                    txt = "-"
                cells += f"{txt:>22}"
            if any_present:
                print(name.ljust(12) + cells)
        ku = {str(r): per_rows[str(r)]["k_union"] for r in rows_list}
        print("k_union".ljust(12) + "".join(
            f"{(str(ku[str(r)]['k_union']) + '/' + str(ku[str(r)]['rows_times_k']) + ' x' + str(ku[str(r)]['amplification'])):>22}"
            for r in rows_list))
        imp = receipt["derived"][mode]["implied_ms_per_cycle_rows6_x40"]
        if imp:
            print("implied ms/cycle @rows6 x40 layers: " + ", ".join(
                f"{c}={v}" for c, v in imp.items()))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", action="store_true",
                   help="run the real cell on Metal (default: CPU-pinned)")
    p.add_argument("--cpu-smoke", action="store_true",
                   help="tiny-dims plumbing check (T=256, k=32, hd=32), CPU")
    p.add_argument("--modes", type=str, default=None,
                   help="comma-list of CSA modes (default: all four / smoke: full)")
    p.add_argument("--rows", type=int, nargs="+", default=None,
                   help="query-row counts to sweep (default: 1 2 4 6 8)")
    p.add_argument("--codec", type=str, default=None,
                   choices=["bf16", "mxfp8", "mxfp4"],
                   help="projection codec (default: mxfp8 gpu / bf16 smoke)")
    p.add_argument("--T", type=int, default=None, help="KV context length")
    p.add_argument("--repeats", type=int, default=None,
                   help="median over this many samples (>= 7 for the GPU cell)")
    p.add_argument("--chain", type=int, default=None,
                   help="calls per pipelined chain")
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--out", type=str, default=None, help="write the JSON receipt here")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    cfg: dict = {"gpu": a.gpu, "cpu_smoke": a.cpu_smoke}
    if a.modes is not None:
        cfg["modes"] = [m.strip() for m in a.modes.split(",") if m.strip()]
    if a.rows is not None:
        cfg["rows"] = a.rows
    if a.codec is not None:
        cfg["codec"] = a.codec
    if a.T is not None:
        cfg["T"] = a.T
    if a.repeats is not None:
        cfg["repeats"] = a.repeats
    if a.chain is not None:
        cfg["chain"] = a.chain
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
