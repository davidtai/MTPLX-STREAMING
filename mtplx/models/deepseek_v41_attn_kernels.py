"""K29 — fused decode / verify MLA attention Metal kernel for DeepSeek-V4.1.

Computes the whole M=1 (decode) / small-M (K+1 verify) MLA attention step — QK^T
scores, the CSA/causal mask, the per-head value-0 attention sink, the f32 softmax,
and PV — with an **online softmax over key tiles** so the ``[64, T]`` per-head
score row NEVER materialises (T up to 16K+ streams through threadgroup memory).

Motivation (W60 / KERNEL_LEDGER K29).  Windows 13–16 put decode at 1,024 context
at ~160 ms/token, **dispatch-bound**: attention costs ~60 ms of it (``attn.reuse``
50 ms fenced over 30 layers ≈ 1.5–1.9 ms per M=1 layer) with ~110 Metal primitives
per layer after the K22/K24 compiles, while the actual math per layer at M=1 is
tiny (64 heads × head_dim 512 against ~1K–16K compressed keys).  W45 proved the
shipped eager SDPA (``_sparse_attend``: two einsums + softmax + mask + sink-concat
+ slice) is an **irreducible ~22-primitive reduction a shapeless ``mx.compile``
cannot fold**.  K29 collapses that whole tail into ONE (or, split-K, two)
dispatch(es) per layer.

MLX's fused SDPA cannot be used: head_dim 512 is unsupported by
``mx.fast.scaled_dot_product_attention`` and the value-0 sink semantics differ (W50
measured 2.1e-3 divergence).

**Occupancy (W60 window-26 redesign).**  At M=1 the score/PV is only ``rows·H``
independent ``(row, head)`` reductions — 64 threadgroups per layer at decode, a
tiny fraction of the GPU, so a single-threadgroup-per-``(row,head)`` kernel is
**latency-bound** (window-26: the ``decode_attn_kernel`` arm was −45% vs stack_a,
byte-identical tokens).  The kernel therefore **splits the T keys across ``G``
threadgroups per ``(row, head)``** (flash-decoding split-K), giving ``rows·H·G``
threadgroups (≥512 at M=1), then a cheap second kernel combines the ``G`` partials
per ``(row, head)`` with the value-0 sink.  ``G`` is chosen so the split covers the
GPU (``_choose_splits``); tiny ``T`` (≤ ``_SINGLE_KERNEL_MAX_T``) keeps the
single-dispatch path.

**Precision (W60 window-26 fix).**  ``mx.fast.metal_kernel`` compiles with fast
math, under which the bare ``metal::exp`` is the *fast* approximation (~1e-3
relative error — window-26 parity showed max abs-delta 4e-4…2.1e-3, argmax
mismatches on verify, NOT the ~1e-6 f32-reassociation class).  Every ``exp`` here is
``metal::precise::exp`` (full f32), matching the ``metal::precise::rsqrt`` /
``metal::precise::exp`` the rest of the codebase uses for the same reason.  An f32
numpy simulation of the exact reduction (naive sequential dot + online combine)
matches MLX's blocked matmul softmax to ~1e-7 — i.e., the reduction ORDER is not
the leak, only the transcendental was.

Semantics preserved exactly (reference ``scripts/deepseek_v41/torchref/
ref_forward.py`` ``sparse_attn`` and the model :meth:`Attention._sparse_attend_oneshot`):

  * MLA — one shared KV latent (``k_cache`` and ``v_cache`` are the SAME array), so
    ``score[t] = scale · dot(q[head], KV[t])`` over all ``head_dim`` dims (RoPE
    already baked into q and the cached latents) and ``out[head] = Σ_t p_t · KV[t]``;
  * a boolean CSA/causal ``attend`` mask ``[b, s, T]`` shared across heads (True =
    keep) — the wrapper converts it to the additive ``{0, -inf}`` form;
  * the per-head **value-0 attention sink**: adds ``exp(attn_sink[head] − m)`` to
    the denominator and nothing to the numerator, so a fully-masked row collapses to
    the sink alone (all probabilities 0 → zero output), never a NaN;
  * f32 throughout; ``metal::precise::exp``.

Exactness class: **reassociation-level** vs the eager f32 path (the online / split
reductions reorder the max / denom / value sums), expected ``max|Δ| ≤ 1e-6`` and
greedy-argmax identical — NOT byte-identical.

CPU-safe pure-MLX references (:func:`decode_attention_reference`, one-shot;
:func:`decode_attention_reference_tiled`, the exact online-tile algorithm;
:func:`decode_attention_reference_splitk`, the exact split-K partition + combine)
run with no Metal so the plumbing/spy and CPU-parity tests never build a kernel on
the CPU-pinned worker box.
"""

from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx

#: Threads per ``(row, head[, split])`` group.  Power of two (tree reductions) and
#: the key-tile width (one key per lane per tile).
_TG_DEFAULT = 128

#: Head_dim below/at which one fused threadgroup per ``(row,head)`` is fine — no
#: split-K (the combine dispatch is not worth it for a handful of keys).
_SINGLE_KERNEL_MAX_T = 256

#: Split-K knobs (W60 window-26 occupancy redesign).
_KEYS_PER_SPLIT = 512      #: target keys per split threadgroup
_SPLITS_MAX = 32           #: cap on G (bounds the intermediate buffer + combine cost)
_OCC_TARGET_TG = 512       #: threadgroups we want the split kernel to launch

#: Finite sentinel for "no key seen yet" / a masked lane (keeps the online combine
#: NaN-free: ``NEG − NEG = 0`` → ``exp = 1`` → ``0·1 = 0``).  ``NEG_HALF`` is the
#: "is this a real score" cutoff (a genuine score is ``O(10)`` ≫ NEG_HALF).
_NEG = "-3.0e38f"
_NEG_HALF = "-1.5e38f"


# ---------------------------------------------------------------------------
# engagement telemetry (W60 window-26): count real kernel dispatches so an A/B
# receipt can tell "kernel did not run" from "ran (slowly)".
# ---------------------------------------------------------------------------
_ENGAGE_COUNT = 0
_ENGAGE_ROWS = 0
_ENGAGE_SPLIT_CALLS = 0
_FALLBACK_COUNT = 0


def reset_engagement() -> None:
    """Zero the per-run engagement counters (call before each A/B arm)."""
    global _ENGAGE_COUNT, _ENGAGE_ROWS, _ENGAGE_SPLIT_CALLS, _FALLBACK_COUNT
    _ENGAGE_COUNT = _ENGAGE_ROWS = _ENGAGE_SPLIT_CALLS = _FALLBACK_COUNT = 0


def note_fallback() -> None:
    """Record one decode/verify layer-step that fell back to the eager path (the
    kernel was armed + on-GPU but the mask shape was unsupported).  Lets the A/B
    receipt tell "kernel did not run (all fallbacks)" from "ran (slowly)"."""
    global _FALLBACK_COUNT
    _FALLBACK_COUNT += 1


def engagement() -> dict:
    """Counters since the last reset: ``calls`` (layer-steps the kernel actually
    dispatched), ``rows`` (Σ ``b·s`` over those), ``split_calls`` (calls that took
    the split-K two-kernel path), ``fallbacks`` (armed-but-eager layer-steps)."""
    return {
        "calls": _ENGAGE_COUNT,
        "rows": _ENGAGE_ROWS,
        "split_calls": _ENGAGE_SPLIT_CALLS,
        "fallbacks": _FALLBACK_COUNT,
    }


# ---------------------------------------------------------------------------
# Metal source — one fused kernel (single threadgroup per (row,head))
# ---------------------------------------------------------------------------
# Assembled by ``.replace()`` on a plain template (the Metal body is dense with
# literal ``{`` / ``}`` blocks; doubling braces for an f-string is error-prone).
# ``H`` / ``T`` / ``S`` / ``scale`` are RUNTIME scalars; ``TG`` / ``HD`` compile-time.
_SINGLE_TEMPLATE = r"""
    using namespace metal;
    constexpr uint TG = %%TG%%;
    constexpr uint HD = %%HD%%;
    constexpr float NEG = %%NEG%%;
    constexpr float NEG_HALF = %%NEG_HALF%%;

    const uint gid  = threadgroup_position_in_grid.x;   // one group per (row,head)
    const uint lane = thread_position_in_threadgroup.x;
    const uint Hc = uint(H);
    const uint Tc = uint(T);
    const uint Sc = uint(S);
    const uint row  = gid / Hc;
    const uint head = gid % Hc;
    const uint b_idx = row / Sc;
    const float sc = float(scale);
    const uint q_base  = gid * HD;
    const uint kv_base = b_idx * Tc * HD;
    const uint m_base  = row * Tc;

    threadgroup float q_sh[HD];
    threadgroup float acc_sh[HD];
    threadgroup float s_sh[TG];
    threadgroup float p_sh[TG];
    threadgroup float red_sh[TG];

    for (uint i = lane; i < HD; i += TG) { q_sh[i] = q[q_base + i]; acc_sh[i] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_run = NEG;
    float d_run = 0.0f;

    for (uint tile0 = 0; tile0 < Tc; tile0 += TG) {
        uint t = tile0 + lane;
        float score = NEG;
        if (t < Tc) {
            bool keep = true;
            float add = 0.0f;
            %%MASK%%
            if (keep) {
                uint krow = kv_base + t * HD;
                float dot = 0.0f;
                for (uint d = 0; d < HD; ++d) { dot += q_sh[d] * k[krow + d]; }
                score = dot * sc + add;
            }
        }
        s_sh[lane] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        red_sh[lane] = s_sh[lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = TG >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) { red_sh[lane] = metal::max(red_sh[lane], red_sh[lane + stride]); }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float m_tile = red_sh[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float m_new = metal::max(m_run, m_tile);
        float corr  = metal::precise::exp(m_run - m_new);

        float sv = s_sh[lane];
        float p = (sv <= NEG_HALF) ? 0.0f : metal::precise::exp(sv - m_new);
        p_sh[lane] = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        red_sh[lane] = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = TG >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) { red_sh[lane] = red_sh[lane] + red_sh[lane + stride]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float tile_denom = red_sh[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        d_run = d_run * corr + tile_denom;

        for (uint i = lane; i < HD; i += TG) {
            float a = acc_sh[i] * corr;
            float pv = 0.0f;
            for (uint kk = 0; kk < TG; ++kk) {
                uint tt = tile0 + kk;
                if (tt < Tc) { pv += p_sh[kk] * v[kv_base + tt * HD + i]; }
            }
            acc_sh[i] = a + pv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        m_run = m_new;
    }

    float denom = d_run;
    %%SINK%%
    for (uint i = lane; i < HD; i += TG) {
        out[q_base + i] = (denom > 0.0f) ? (acc_sh[i] / denom) : 0.0f;
    }
"""

# ---------------------------------------------------------------------------
# Metal source — split-K pass 1: partial (m, denom, acc) over one key range.
# grid = TG · (rows·H·G); one threadgroup per (row, head, split).  NO sink, NO
# normalize -- the combine kernel folds the sink and divides.
# ---------------------------------------------------------------------------
_SPLIT_TEMPLATE = r"""
    using namespace metal;
    constexpr uint TG = %%TG%%;
    constexpr uint HD = %%HD%%;
    constexpr uint G  = %%G%%;
    constexpr float NEG = %%NEG%%;
    constexpr float NEG_HALF = %%NEG_HALF%%;

    const uint gid  = threadgroup_position_in_grid.x;   // (row*H+head)*G + split
    const uint lane = thread_position_in_threadgroup.x;
    const uint Hc = uint(H);
    const uint Tc = uint(T);
    const uint Sc = uint(S);
    const uint rh   = gid / G;            // row*H + head
    const uint g    = gid % G;            // split index
    const uint row  = rh / Hc;
    const uint head = rh % Hc;
    const uint b_idx = row / Sc;
    const float sc = float(scale);
    const uint q_base  = rh * HD;
    const uint kv_base = b_idx * Tc * HD;
    const uint m_base  = row * Tc;

    // this split's contiguous key range [c0, c1)
    const uint chunk = (Tc + G - 1) / G;
    const uint c0 = g * chunk;
    const uint c1 = metal::min(c0 + chunk, Tc);

    threadgroup float q_sh[HD];
    threadgroup float acc_sh[HD];
    threadgroup float s_sh[TG];
    threadgroup float p_sh[TG];
    threadgroup float red_sh[TG];

    for (uint i = lane; i < HD; i += TG) { q_sh[i] = q[q_base + i]; acc_sh[i] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_run = NEG;
    float d_run = 0.0f;

    for (uint tile0 = c0; tile0 < c1; tile0 += TG) {
        uint t = tile0 + lane;
        float score = NEG;
        if (t < c1) {
            bool keep = true;
            float add = 0.0f;
            %%MASK%%
            if (keep) {
                uint krow = kv_base + t * HD;
                float dot = 0.0f;
                for (uint d = 0; d < HD; ++d) { dot += q_sh[d] * k[krow + d]; }
                score = dot * sc + add;
            }
        }
        s_sh[lane] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        red_sh[lane] = s_sh[lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = TG >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) { red_sh[lane] = metal::max(red_sh[lane], red_sh[lane + stride]); }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float m_tile = red_sh[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float m_new = metal::max(m_run, m_tile);
        float corr  = metal::precise::exp(m_run - m_new);

        float sv = s_sh[lane];
        float p = (sv <= NEG_HALF) ? 0.0f : metal::precise::exp(sv - m_new);
        p_sh[lane] = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        red_sh[lane] = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = TG >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) { red_sh[lane] = red_sh[lane] + red_sh[lane + stride]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float tile_denom = red_sh[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        d_run = d_run * corr + tile_denom;

        for (uint i = lane; i < HD; i += TG) {
            float a = acc_sh[i] * corr;
            float pv = 0.0f;
            for (uint kk = 0; kk < TG; ++kk) {
                uint tt = tile0 + kk;
                if (tt < c1) { pv += p_sh[kk] * v[kv_base + tt * HD + i]; }
            }
            acc_sh[i] = a + pv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        m_run = m_new;
    }

    // write this split's partial state (un-normalised, no sink)
    if (lane == 0) { part_m[gid] = m_run; part_d[gid] = d_run; }
    for (uint i = lane; i < HD; i += TG) { part_acc[gid * HD + i] = acc_sh[i]; }
"""

# ---------------------------------------------------------------------------
# Metal source — split-K pass 2: combine the G partials per (row,head), fold the
# value-0 sink, normalise.  grid = TG · (rows·H); one threadgroup per (row,head).
# ---------------------------------------------------------------------------
_COMBINE_TEMPLATE = r"""
    using namespace metal;
    constexpr uint TG = %%TG%%;
    constexpr uint HD = %%HD%%;
    constexpr uint G  = %%G%%;
    constexpr float NEG = %%NEG%%;

    const uint gid  = threadgroup_position_in_grid.x;   // row*H + head
    const uint lane = thread_position_in_threadgroup.x;
    const uint Hc = uint(H);
    const uint head = gid % Hc;
    const uint q_base = gid * HD;
    const uint pbase  = gid * G;

    threadgroup float mparts[G];
    threadgroup float dparts[G];
    threadgroup float wparts[G];

    if (lane < G) { mparts[lane] = part_m[pbase + lane]; dparts[lane] = part_d[pbase + lane]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // global max over the G split maxima (G is tiny; every lane computes it)
    float m = NEG;
    for (uint g = 0; g < G; ++g) { m = metal::max(m, mparts[g]); }
    if (lane < G) { wparts[lane] = metal::precise::exp(mparts[lane] - m); }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float denom = 0.0f;
    for (uint g = 0; g < G; ++g) { denom += dparts[g] * wparts[g]; }
    %%SINK%%

    for (uint i = lane; i < HD; i += TG) {
        float a = 0.0f;
        for (uint g = 0; g < G; ++g) { a += part_acc[(pbase + g) * HD + i] * wparts[g]; }
        out[q_base + i] = (denom > 0.0f) ? (a / denom) : 0.0f;
    }
"""

_MASK_BLOCK = (
    "float mv = mask[m_base + t];\n"
    "                keep = (mv >= NEG_HALF);\n"
    "                add = mv;"
)
_SINK_BLOCK_SINGLE = "denom = denom + metal::precise::exp(sink[head] - m_run);"
_SINK_BLOCK_COMBINE = "denom = denom + metal::precise::exp(sink[head] - m);"


def _sub(src: str, *, tg: int, hd: int, g: int | None = None) -> str:
    src = src.replace("%%TG%%", str(int(tg))).replace("%%HD%%", str(int(hd)))
    if g is not None:
        src = src.replace("%%G%%", str(int(g)))
    src = src.replace("%%NEG_HALF%%", _NEG_HALF).replace("%%NEG%%", _NEG)
    return src


def _build_source(*, tg: int, hd: int, has_mask: bool, sink_on: bool) -> str:
    """Single-dispatch fused kernel (one threadgroup per ``(row,head)``)."""
    src = _sub(_SINGLE_TEMPLATE, tg=tg, hd=hd)
    src = src.replace("%%MASK%%", _MASK_BLOCK if has_mask else "")
    src = src.replace("%%SINK%%", _SINK_BLOCK_SINGLE if sink_on else "")
    return src


def _build_split_source(*, tg: int, hd: int, g: int, has_mask: bool) -> str:
    """Split-K pass 1 (partials over one key range; no sink, no normalize)."""
    src = _sub(_SPLIT_TEMPLATE, tg=tg, hd=hd, g=g)
    src = src.replace("%%MASK%%", _MASK_BLOCK if has_mask else "")
    return src


def _build_combine_source(*, tg: int, hd: int, g: int, sink_on: bool) -> str:
    """Split-K pass 2 (combine G partials + value-0 sink + normalize)."""
    src = _sub(_COMBINE_TEMPLATE, tg=tg, hd=hd, g=g)
    src = src.replace("%%SINK%%", _SINK_BLOCK_COMBINE if sink_on else "")
    return src


@lru_cache(maxsize=None)
def _k29_kernel(*, tg: int, hd: int, has_mask: bool, sink_on: bool):
    if not mx.metal.is_available():
        return None
    inputs = ["q", "k", "v", "H", "T", "S", "scale"]
    if has_mask:
        inputs.append("mask")
    if sink_on:
        inputs.append("sink")
    suffix = f"tg{int(tg)}_hd{int(hd)}_{'m' if has_mask else 'nm'}_{'s' if sink_on else 'ns'}"
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv41_fused_decode_attn_{suffix}",
        input_names=inputs,
        output_names=["out"],
        source=_build_source(tg=tg, hd=hd, has_mask=has_mask, sink_on=sink_on),
    )


@lru_cache(maxsize=None)
def _k29_split_kernel(*, tg: int, hd: int, g: int, has_mask: bool):
    if not mx.metal.is_available():
        return None
    inputs = ["q", "k", "v", "H", "T", "S", "scale"]
    if has_mask:
        inputs.append("mask")
    suffix = f"tg{int(tg)}_hd{int(hd)}_g{int(g)}_{'m' if has_mask else 'nm'}"
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv41_fused_decode_attn_split_{suffix}",
        input_names=inputs,
        output_names=["part_m", "part_d", "part_acc"],
        source=_build_split_source(tg=tg, hd=hd, g=g, has_mask=has_mask),
    )


@lru_cache(maxsize=None)
def _k29_combine_kernel(*, tg: int, hd: int, g: int, sink_on: bool):
    if not mx.metal.is_available():
        return None
    inputs = ["part_m", "part_d", "part_acc", "H"]
    if sink_on:
        inputs.append("sink")
    suffix = f"tg{int(tg)}_hd{int(hd)}_g{int(g)}_{'s' if sink_on else 'ns'}"
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv41_fused_decode_attn_combine_{suffix}",
        input_names=inputs,
        output_names=["out"],
        source=_build_combine_source(tg=tg, hd=hd, g=g, sink_on=sink_on),
    )


def kernel_available() -> bool:
    """Whether a Metal GPU is present so the kernel can build/dispatch."""
    return bool(mx.metal.is_available())


def _choose_splits(T: int, n_rows_heads: int) -> int:
    """Pick ``G`` (key-splits per ``(row,head)``) for occupancy: enough
    ``rows·H·G`` threadgroups to cover the GPU AND ~``_KEYS_PER_SPLIT`` keys per
    split, capped at ``_SPLITS_MAX`` and at ``T``.  ``T <= _SINGLE_KERNEL_MAX_T``
    keeps the single-dispatch path (``G == 1``)."""
    if T <= _SINGLE_KERNEL_MAX_T:
        return 1
    g_occ = max(1, math.ceil(_OCC_TARGET_TG / max(1, n_rows_heads)))
    g_load = max(1, math.ceil(T / _KEYS_PER_SPLIT))
    g = min(_SPLITS_MAX, max(g_occ, g_load))
    return max(1, min(g, T))


# ---------------------------------------------------------------------------
# shape plumbing
# ---------------------------------------------------------------------------
def _prepare_q(q: mx.array):
    if q.dtype != mx.float32:
        q = q.astype(mx.float32)
    if q.ndim == 4:
        b, s, H, hd = (int(d) for d in q.shape)
        return mx.contiguous(q.reshape(b * s, H, hd)), b, s, H, hd
    if q.ndim == 3:
        rows, H, hd = (int(d) for d in q.shape)
        return mx.contiguous(q), None, None, H, hd
    raise ValueError(f"q must be 3D [rows,H,hd] or 4D [b,s,H,hd], got {q.shape}")


def _prepare_kv(kv: mx.array, *, b: int, hd: int):
    if kv.dtype != mx.float32:
        kv = kv.astype(mx.float32)
    if kv.ndim == 2:
        kv = kv.reshape(1, kv.shape[0], kv.shape[1])
    if kv.ndim != 3:
        raise ValueError(f"kv must be [b,T,hd] or [T,hd], got {kv.shape}")
    kb, T, khd = (int(d) for d in kv.shape)
    if khd != hd:
        raise ValueError(f"kv head_dim {khd} != q head_dim {hd}")
    if kb != b:
        raise ValueError(f"kv batch {kb} != q batch {b}")
    return mx.contiguous(kv), T


def _prepare_mask(attend, *, rows: int, T: int):
    if attend is None:
        return None
    if attend.dtype == mx.bool_:
        add = mx.where(
            attend,
            mx.zeros(attend.shape, dtype=mx.float32),
            mx.full(attend.shape, float("-inf"), dtype=mx.float32),
        )
    else:
        add = attend.astype(mx.float32)
    return mx.contiguous(add.reshape(rows, T))


def fused_decode_attention(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    attend=None,
    attn_sink=None,
    scale: float = 1.0,
    T=None,
    *,
    tg: int = _TG_DEFAULT,
    n_splits=None,
):
    """Fused decode / verify MLA attention (K29).

    Parameters
    ----------
    q : ``[b, s, H, hd]`` (or ``[rows, H, hd]``) query, RoPE already applied.
    k_cache, v_cache : the cached KV latent ``[b, T, hd]`` (or ``[T, hd]``); MLA →
        the SAME array, passed as both.
    attend : boolean ``[b, s, T]`` / ``[rows, T]`` mask (True = keep), an additive
        f32 mask, or ``None``.  Shared across heads.
    attn_sink : per-head ``[H]`` f32 value-0 sink scores, or ``None``.
    scale : the softmax scale (``head_dim ** -0.5``).
    T : optional key count (defaults to ``k_cache.shape[-2]``).
    tg : threads per group (power of two; key-tile width).
    n_splits : split-K ``G`` (key-splits per ``(row,head)``); ``None`` = auto
        (:func:`_choose_splits`), ``1`` = force the single-dispatch fused kernel.

    Returns
    -------
    ``out`` ``[b, s, H, hd]`` (or ``[rows, H, hd]`` when ``q`` was 3D) f32.
    """
    global _ENGAGE_COUNT, _ENGAGE_ROWS, _ENGAGE_SPLIT_CALLS
    q3, b, s, H, hd = _prepare_q(q)
    rows = q3.shape[0]
    b_eff = rows if b is None else b
    S_arg = rows // b_eff if b_eff else rows

    k3, Tk = _prepare_kv(k_cache, b=b_eff, hd=hd)
    v3, Tv = _prepare_kv(v_cache, b=b_eff, hd=hd)
    if Tk != Tv:
        raise ValueError(f"k_cache T {Tk} != v_cache T {Tv}")
    T_use = int(Tk if T is None else T)
    if T_use > Tk:
        raise ValueError(f"requested T {T_use} exceeds cached rows {Tk}")

    has_mask = attend is not None
    mask = _prepare_mask(attend, rows=rows, T=T_use) if has_mask else None
    sink_on = attn_sink is not None
    sink = attn_sink.astype(mx.float32).reshape(H) if sink_on else None

    n_rh = rows * H
    G = _choose_splits(T_use, n_rh) if n_splits is None else max(1, int(n_splits))
    G = max(1, min(G, T_use))
    _NO_METAL = (  # the factory returns None off-Metal — caller owns the eager fallback
        "K29 fused decode attention requires a Metal GPU (kernel_available() is False)"
    )

    if G == 1:
        kernel = _k29_kernel(tg=int(tg), hd=int(hd), has_mask=has_mask, sink_on=sink_on)
        if kernel is None:
            raise RuntimeError(_NO_METAL)
        inputs = [q3, k3, v3, int(H), int(T_use), int(S_arg), float(scale)]
        if has_mask:
            inputs.append(mask)
        if sink_on:
            inputs.append(sink)
        _ENGAGE_COUNT += 1
        _ENGAGE_ROWS += rows
        (out3,) = kernel(
            inputs=inputs,
            grid=(int(tg) * n_rh, 1, 1),
            threadgroup=(int(tg), 1, 1),
            output_shapes=[(rows, H, hd)],
            output_dtypes=[mx.float32],
        )
    else:
        split = _k29_split_kernel(tg=int(tg), hd=int(hd), g=int(G), has_mask=has_mask)
        combine = _k29_combine_kernel(tg=int(tg), hd=int(hd), g=int(G), sink_on=sink_on)
        if split is None or combine is None:
            raise RuntimeError(_NO_METAL)
        sinputs = [q3, k3, v3, int(H), int(T_use), int(S_arg), float(scale)]
        if has_mask:
            sinputs.append(mask)
        n_part = n_rh * G
        _ENGAGE_COUNT += 1
        _ENGAGE_ROWS += rows
        _ENGAGE_SPLIT_CALLS += 1
        part_m, part_d, part_acc = split(
            inputs=sinputs,
            grid=(int(tg) * n_part, 1, 1),
            threadgroup=(int(tg), 1, 1),
            output_shapes=[(n_part,), (n_part,), (n_part, hd)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
        cinputs = [part_m, part_d, part_acc, int(H)]
        if sink_on:
            cinputs.append(sink)
        (out3,) = combine(
            inputs=cinputs,
            grid=(int(tg) * n_rh, 1, 1),
            threadgroup=(int(tg), 1, 1),
            output_shapes=[(rows, H, hd)],
            output_dtypes=[mx.float32],
        )

    if b is not None:
        return out3.reshape(b, s, H, hd)
    return out3


# ---------------------------------------------------------------------------
# CPU-safe pure-MLX references (no Metal — spy/plumbing + CPU parity)
# ---------------------------------------------------------------------------
def decode_attention_reference(q, k_cache, v_cache, *, attend=None, attn_sink=None,
                               scale: float = 1.0, T=None):
    """One-shot pure-MLX softmax-with-sink MLA attention — the exact operation the
    K29 kernel targets, without any Metal.  Mirrors ``_sparse_attend_oneshot``'s
    ``fold_sink`` math; masked keys → 0, a fully-masked row → 0 output."""
    q = q.astype(mx.float32)
    KV = k_cache.astype(mx.float32)
    if KV.ndim == 2:
        KV = KV.reshape(1, KV.shape[0], KV.shape[1])
    if q.ndim == 3:
        q = q.reshape(1, *q.shape)
        squeeze = True
    else:
        squeeze = False
    b, s, H, hd = (int(d) for d in q.shape)
    Tk = KV.shape[1]
    T_use = Tk if T is None else int(T)
    KV = KV[:, :T_use, :]
    scores = mx.einsum("bshd,btd->bsht", q, KV) * mx.array(scale, dtype=mx.float32)
    if attend is not None:
        att = attend
        if att.dtype != mx.bool_:
            att = att >= mx.array(float("-1.5e38"), dtype=mx.float32)
        att = att[:, :, :T_use]
        scores = mx.where(att[:, :, None, :], scores, mx.array(float("-inf"), dtype=mx.float32))
    if attn_sink is not None:
        sink = attn_sink.astype(mx.float32).reshape(1, 1, H, 1)
    else:
        sink = mx.full((1, 1, H, 1), float("-inf"), dtype=mx.float32)
    m = mx.maximum(mx.max(scores, axis=-1, keepdims=True), sink)
    ex = mx.exp(scores - m)
    denom = mx.sum(ex, axis=-1, keepdims=True) + mx.exp(sink - m)
    out = mx.einsum("bsht,btd->bshd", ex, KV) / denom
    if squeeze:
        return out.reshape(s, H, hd)
    return out


def _online_combine(q, KV, attend, attn_sink, scale, T_use, ranges):
    """Shared pure-MLX online-softmax combine over a list of key ranges (used by the
    tiled and split-K references): the same ``corr = exp(m_run − m_new)`` fold the
    Metal threadgroup does.  ``ranges`` is a list of ``(c0, c1)`` key spans covering
    ``[0, T_use)`` (contiguous, in any partition), which reassociate exactly like the
    tile / split combine."""
    b, s, H, hd = (int(d) for d in q.shape)
    NEG = mx.array(-3.0e38, dtype=mx.float32)
    att = None
    if attend is not None:
        att = attend if attend.dtype == mx.bool_ else (attend >= mx.array(-1.5e38, dtype=mx.float32))
    m_run = mx.full((b, s, H, 1), -3.0e38, dtype=mx.float32)
    d_run = mx.zeros((b, s, H, 1), dtype=mx.float32)
    acc = mx.zeros((b, s, H, hd), dtype=mx.float32)
    for c0, c1 in ranges:
        if c1 <= c0:
            continue
        KVc = KV[:, c0:c1, :]
        sc = mx.einsum("bshd,btd->bsht", q, KVc) * mx.array(scale, dtype=mx.float32)
        if att is not None:
            sc = mx.where(att[:, :, c0:c1][:, :, None, :], sc, NEG)
        valid = sc > mx.array(-1.5e38, dtype=mx.float32)
        m_tile = mx.max(mx.where(valid, sc, NEG), axis=-1, keepdims=True)
        m_new = mx.maximum(m_run, m_tile)
        corr = mx.exp(m_run - m_new)
        p = mx.where(valid, mx.exp(sc - m_new), mx.zeros_like(sc))
        d_run = d_run * corr + mx.sum(p, axis=-1, keepdims=True)
        acc = acc * corr + mx.einsum("bsht,btd->bshd", p, KVc)
        m_run = m_new
    if attn_sink is not None:
        sink = attn_sink.astype(mx.float32).reshape(1, 1, H, 1)
        denom = d_run + mx.exp(sink - m_run)
    else:
        denom = d_run
    return mx.where(denom > 0, acc / denom, mx.zeros_like(acc))


def _prep_ref(q, k_cache):
    q = q.astype(mx.float32)
    KV = k_cache.astype(mx.float32)
    if KV.ndim == 2:
        KV = KV.reshape(1, KV.shape[0], KV.shape[1])
    squeeze = q.ndim == 3
    if squeeze:
        q = q.reshape(1, *q.shape)
    return q, KV, squeeze


def decode_attention_reference_tiled(q, k_cache, v_cache, *, attend=None,
                                     attn_sink=None, scale: float = 1.0, T=None,
                                     tile: int = _TG_DEFAULT):
    """The exact online-tile algorithm the single-dispatch kernel runs (contiguous
    ``tile``-wide spans in order), in pure MLX."""
    q, KV, squeeze = _prep_ref(q, k_cache)
    b, s, H, hd = (int(d) for d in q.shape)
    T_use = KV.shape[1] if T is None else int(T)
    ranges = [(c0, min(c0 + tile, T_use)) for c0 in range(0, T_use, tile)]
    out = _online_combine(q, KV, attend, attn_sink, scale, T_use, ranges)
    return out.reshape(s, H, hd) if squeeze else out


def decode_attention_reference_splitk(q, k_cache, v_cache, *, attend=None,
                                      attn_sink=None, scale: float = 1.0, T=None,
                                      n_splits: int = 8):
    """The exact split-K partition + combine the two-kernel path runs: ``G``
    contiguous ``ceil(T/G)``-wide key ranges, each reduced independently, then merged
    with the value-0 sink.  Mathematically identical to the one-shot reference (the
    online combine is associative up to float reassociation) — the CPU proof that the
    split-K reassociation is greedy-safe before any GPU run."""
    q, KV, squeeze = _prep_ref(q, k_cache)
    b, s, H, hd = (int(d) for d in q.shape)
    T_use = KV.shape[1] if T is None else int(T)
    G = max(1, min(int(n_splits), T_use))
    chunk = (T_use + G - 1) // G
    ranges = [(g * chunk, min(g * chunk + chunk, T_use)) for g in range(G)]
    out = _online_combine(q, KV, attend, attn_sink, scale, T_use, ranges)
    return out.reshape(s, H, hd) if squeeze else out
