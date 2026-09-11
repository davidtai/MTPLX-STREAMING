"""K29 — fused decode / verify MLA attention Metal kernel for DeepSeek-V4.1.

One ``mx.fast.metal_kernel`` dispatch **per layer** computes the whole M=1 (decode)
or small-M (K+1 verify) MLA attention step — QK^T scores, the CSA/causal mask, the
per-head value-0 attention sink, the f32 softmax, and PV — with an **online
softmax over key tiles** so the ``[64, T]`` per-head score row NEVER materialises
(T up to 16K+ streams through threadgroup memory).

Motivation (W60 / KERNEL_LEDGER K29).  Windows 13–16 put decode at 1,024 context
at ~160 ms/token, **dispatch-bound**: attention costs ~60 ms of it (``attn.reuse``
50 ms fenced over 30 layers ≈ 1.5–1.9 ms per M=1 layer, plus ~18 ms on the other
10 layers) with ~110 Metal primitives per layer after the K22/K24 compiles, while
the actual math per layer at M=1 is tiny (64 heads × head_dim 512 against ~1K–16K
compressed keys).  A single-row attention step costing 1.5–4.9 ms is a
dispatch-chain problem ([[b1-decode-dispatch-removal-hides]]).  The shipped eager
decode path (:meth:`Attention._sparse_attend_oneshot`, the ``fold_sink=False``
one-shot) dispatches, per layer, an ``einsum`` QK^T (matmul → reshape), a
scale, a ``mx.where`` mask, a sink ``concatenate``, a ``softmax`` (max/exp/sum/div),
a slice, and an ``einsum`` PV — ~15–25 Metal primitives.  K29 collapses that whole
tail (score + mask + sink softmax + PV) into **ONE** dispatch.

MLX's fused SDPA cannot be used here: head_dim 512 is unsupported by
``mx.fast.scaled_dot_product_attention`` and the value-0 sink semantics differ
(W50 measured 2.1e-3 divergence against a plain SDPA).

Semantics preserved exactly (reference ``scripts/deepseek_v41/torchref/
ref_forward.py`` ``sparse_attn`` and the model :meth:`Attention._sparse_attend_oneshot`):

  * MLA — one shared KV latent (``k_cache`` and ``v_cache`` are the SAME array), so
    ``score[t] = scale · dot(q[head], KV[t])`` over all ``head_dim`` dims (RoPE is
    already baked into q and the cached latents upstream) and ``out[head] = Σ_t p_t
    · KV[t]``;
  * a boolean CSA/causal ``attend`` mask ``[b, s, T]`` shared across heads (True =
    keep) — the wrapper converts it to the additive ``{0, -inf}`` form; a masked key
    contributes nothing;
  * the per-head **value-0 attention sink**: a virtual key whose score is
    ``attn_sink[head]`` and whose value is 0.  It adds ``exp(attn_sink[head] - m)``
    to the denominator and nothing to the numerator, so a fully-masked row collapses
    to the sink alone (all probabilities 0 → zero output), never a NaN;
  * f32 throughout; ``metal::exp`` (precise, matching ``mx.exp``).

Exactness class: **reassociation-level** vs the eager f32 path (the online tile
reduction reorders the max / denom / value sums), expected ``max|Δ| ≤ 1e-6`` and
greedy-argmax identical — the same float class as the K25/K28 tree softmaxes, NOT
byte-identical.

Design (one threadgroup per ``(row, head)``, ``TG`` lanes, one key per lane per
tile):

  1. cooperatively load ``q[row,head,:]`` (head_dim floats) into threadgroup memory
     and zero the running ``acc[head_dim]``;
  2. for each ``TG``-wide key tile: each lane computes its key's masked score
     ``scale·dot(q, KV[t]) + add`` (a head_dim dot), a threadgroup **tree** reduces
     the tile max, all lanes fold the tile into the running ``(m, denom)`` online
     softmax (``corr = exp(m_run - m_new)``; ``acc *= corr``; ``denom = denom·corr +
     Σ_tile exp(s - m_new)``), then the head_dim axis is split across lanes to add
     ``Σ_tile p_k · KV[k]`` into ``acc``;
  3. fold the per-head sink into the denominator once and write ``out = acc / denom``.

CPU-safe pure-MLX references live alongside the kernel
(:func:`decode_attention_reference`, one-shot; :func:`decode_attention_reference_tiled`,
the exact online-tile algorithm) so the plumbing/spy and CPU-parity tests run with
no Metal (``mx.fast.metal_kernel`` builds and dispatches on the GPU regardless of
the default device — never built on the CPU-pinned worker box).
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

#: Threads per ``(row, head)`` reduction group.  Power of two (the tile max / denom
#: tree reduction halves the active lane count each step) and the tile width (one
#: key per lane per tile).  128 = 4 SIMD groups; each lane scans ``ceil(T/128)``
#: keys (≤128 at T=16384).  Threadgroup memory = ``2·head_dim`` (q + acc) + ``3·TG``
#: (score, prob, reduce) floats = ``2·512·4 + 3·128·4`` ≈ 5.6 KiB (≪ 32 KiB).
_TG_DEFAULT = 128

#: Finite sentinel for "no key seen yet" / a masked lane.  A large *finite* negative
#: (not -inf) keeps the online combine NaN-free: an all-masked lane's state is
#: ``(NEG, 0)`` and ``NEG - NEG = 0`` → ``exp = 1`` → ``0·1 = 0`` rather than
#: ``0·exp(nan)``.  ``NEG_HALF`` is the "is this a real score" cutoff (a genuine
#: score is ``O(10)`` ≫ NEG_HALF; a masked lane holds exactly NEG).
_NEG = "-3.0e38f"
_NEG_HALF = "-1.5e38f"


# ---------------------------------------------------------------------------
# Metal source
# ---------------------------------------------------------------------------
# Assembled by ``.replace()`` on a plain template (not an f-string): the Metal
# body is dense with literal ``{`` / ``}`` C++ blocks, and doubling every brace for
# an f-string is error-prone.  Only the four ``%%TOKEN%%`` slots vary structurally;
# ``H`` / ``T`` / ``S`` / ``scale`` are RUNTIME scalar inputs so one compiled kernel
# serves every context length and batch.
_SOURCE_TEMPLATE = r"""
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
    const uint b_idx = row / Sc;             // KV latent is per-batch, shared over s
    const float sc = float(scale);
    const uint q_base  = gid * HD;           // q[row,head,:] (q flat == (row*H+head)*HD)
    const uint kv_base = b_idx * Tc * HD;    // KV[b_idx, 0, :]
    const uint m_base  = row * Tc;           // mask[row,:] (shared over heads)

    threadgroup float q_sh[HD];
    threadgroup float acc_sh[HD];
    threadgroup float s_sh[TG];
    threadgroup float p_sh[TG];
    threadgroup float red_sh[TG];

    // ---- load q into threadgroup memory; zero the running value accumulator -----
    for (uint i = lane; i < HD; i += TG) {
        q_sh[i]   = q[q_base + i];
        acc_sh[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_run = NEG;      // running softmax max (over keys seen so far)
    float d_run = 0.0f;     // running denominator (relative to m_run)

    // ---- stream the T keys in TG-wide tiles (online softmax, no [64,T] alloc) ----
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
                for (uint d = 0; d < HD; ++d) {
                    dot += q_sh[d] * k[krow + d];
                }
                score = dot * sc + add;
            }
        }
        s_sh[lane] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // tile max (tree reduce; masked/out-of-range lanes hold NEG)
        red_sh[lane] = s_sh[lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = TG >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) {
                red_sh[lane] = metal::max(red_sh[lane], red_sh[lane + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float m_tile = red_sh[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float m_new = metal::max(m_run, m_tile);
        float corr  = metal::exp(m_run - m_new);   // rescale prior state to m_new

        // per-lane probability at the new max (masked/out-of-range → 0)
        float sv = s_sh[lane];
        float p = (sv <= NEG_HALF) ? 0.0f : metal::exp(sv - m_new);
        p_sh[lane] = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // tile denominator (tree reduce; p_sh kept intact for the PV pass)
        red_sh[lane] = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = TG >> 1; stride > 0; stride >>= 1) {
            if (lane < stride) {
                red_sh[lane] = red_sh[lane] + red_sh[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float tile_denom = red_sh[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        d_run = d_run * corr + tile_denom;

        // rescale acc by corr, then add this tile's Σ p_k · V[k] (head_dim split
        // across lanes: each lane owns dims {lane, lane+TG, ...}).
        for (uint i = lane; i < HD; i += TG) {
            float a = acc_sh[i] * corr;
            float pv = 0.0f;
            for (uint kk = 0; kk < TG; ++kk) {
                uint tt = tile0 + kk;
                if (tt < Tc) {
                    pv += p_sh[kk] * v[kv_base + tt * HD + i];
                }
            }
            acc_sh[i] = a + pv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        m_run = m_new;
    }

    // ---- fold the per-head value-0 sink into the denominator, normalise, write ---
    float denom = d_run;
    %%SINK%%
    for (uint i = lane; i < HD; i += TG) {
        out[q_base + i] = (denom > 0.0f) ? (acc_sh[i] / denom) : 0.0f;
    }
"""

_MASK_BLOCK = (
    "float mv = mask[m_base + t];\n"
    "                keep = (mv >= NEG_HALF);\n"
    "                add = mv;"
)
_SINK_BLOCK = "denom = denom + metal::exp(sink[head] - m_run);"


def _build_source(*, tg: int, hd: int, has_mask: bool, sink_on: bool) -> str:
    """Assemble the Metal source for one structural variant (mask on/off, sink
    on/off) at a fixed ``TG`` / ``HD``."""
    src = _SOURCE_TEMPLATE
    src = src.replace("%%TG%%", str(int(tg)))
    src = src.replace("%%HD%%", str(int(hd)))
    src = src.replace("%%NEG_HALF%%", _NEG_HALF)   # before %%NEG%% (substring safety)
    src = src.replace("%%NEG%%", _NEG)
    src = src.replace("%%MASK%%", _MASK_BLOCK if has_mask else "")
    src = src.replace("%%SINK%%", _SINK_BLOCK if sink_on else "")
    return src


@lru_cache(maxsize=None)
def _k29_kernel(*, tg: int, hd: int, has_mask: bool, sink_on: bool):
    """Build (and cache) the K29 kernel for one variant, or ``None`` if Metal is
    unavailable (CPU-only test hosts)."""
    if not mx.metal.is_available():
        return None
    inputs = ["q", "k", "v", "H", "T", "S", "scale"]
    if has_mask:
        inputs.append("mask")
    if sink_on:
        inputs.append("sink")
    suffix = (
        f"tg{int(tg)}_hd{int(hd)}"
        f"_{'m' if has_mask else 'nm'}"
        f"_{'s' if sink_on else 'ns'}"
    )
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv41_fused_decode_attn_{suffix}",
        input_names=inputs,
        output_names=["out"],
        source=_build_source(tg=tg, hd=hd, has_mask=has_mask, sink_on=sink_on),
    )


def kernel_available() -> bool:
    """Whether a Metal GPU is present so the kernel can build/dispatch.  The DSV4.1
    flag route also requires the default device be the GPU; callers
    (:func:`deepseek_v41._decode_attn_kernel_use`) check that separately so a
    CPU-pinned worker test never dispatches Metal."""
    return bool(mx.metal.is_available())


# ---------------------------------------------------------------------------
# shape plumbing
# ---------------------------------------------------------------------------
def _prepare_q(q: mx.array):
    """Return (q3 ``[rows, H, hd]`` contiguous f32, b, s, H, hd)."""
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
    """Return the KV latent as a contiguous f32 ``[b, T, hd]`` and T."""
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
    """Convert the model's boolean ``attend`` (True = keep) into the additive
    ``{0, -inf}`` ``[rows, T]`` f32 mask the kernel reads, or ``None`` (no mask).
    An already-additive f32 mask is passed through (contiguous)."""
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
):
    """Fused decode / verify MLA attention (K29).

    Parameters
    ----------
    q : ``[b, s, H, hd]`` (or ``[rows, H, hd]``) query, RoPE already applied.
        ``s`` is 1 for decode or ``K+1`` for a verify batch.
    k_cache, v_cache : the cached KV latent ``[b, T, hd]`` (or ``[T, hd]``).  For
        MLA these are the SAME array (one latent serves key and value); pass it as
        both.
    attend : boolean ``[b, s, T]`` / ``[rows, T]`` CSA/causal mask (True = keep), an
        additive f32 mask, or ``None`` (attend to every key).  Shared across heads.
    attn_sink : per-head ``[H]`` f32 value-0 sink scores, or ``None``.
    scale : the softmax scale (``head_dim ** -0.5``), multiplied into each raw score
        inside the kernel.
    T : optional key count (defaults to ``k_cache.shape[-2]``); a smaller ``T``
        attends to only the first ``T`` cached rows.
    tg : threads per ``(row, head)`` group (power of two; also the key-tile width).

    Returns
    -------
    ``out`` ``[b, s, H, hd]`` (or ``[rows, H, hd]`` when ``q`` was 3D) f32 — the
    attention output the o-projection consumes.
    """
    q3, b, s, H, hd = _prepare_q(q)
    rows = q3.shape[0]
    seq = 1 if s is None else s
    b_eff = rows if b is None else b
    # rows == b*s; the kernel maps row → batch by row // S.
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

    kernel = _k29_kernel(tg=int(tg), hd=int(hd), has_mask=has_mask, sink_on=sink_on)
    if kernel is None:  # no Metal — the caller owns the eager fallback
        raise RuntimeError(
            "K29 fused decode attention requires a Metal GPU "
            "(kernel_available() is False)"
        )

    inputs = [q3, k3, v3, int(H), int(T_use), int(S_arg), float(scale)]
    if has_mask:
        inputs.append(mask)
    if sink_on:
        inputs.append(sink)

    n_groups = rows * H
    (out3,) = kernel(
        inputs=inputs,
        grid=(int(tg) * int(n_groups), 1, 1),
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
    K29 kernel targets, written without any Metal so it runs on the CPU-pinned
    worker box.  Mirrors :meth:`Attention._sparse_attend_oneshot`'s ``fold_sink``
    math (``m = max(max(scores·scale), sink); o = Σ exp(s-m)·V / (Σ exp(s-m) +
    exp(sink-m))``); masked keys → 0, a fully-masked row → 0 output."""
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
    ex = mx.exp(scores - m)                                  # masked → exp(-inf)=0
    denom = mx.sum(ex, axis=-1, keepdims=True) + mx.exp(sink - m)
    out = mx.einsum("bsht,btd->bshd", ex, KV) / denom
    if squeeze:
        return out.reshape(s, H, hd)
    return out


def decode_attention_reference_tiled(q, k_cache, v_cache, *, attend=None,
                                     attn_sink=None, scale: float = 1.0, T=None,
                                     tile: int = _TG_DEFAULT):
    """The EXACT online-tile algorithm the K29 kernel runs, in pure MLX: the same
    per-tile ``(m, denom, acc)`` combine (``corr = exp(m_run - m_new)``) the Metal
    threadgroup does, so a CPU test can prove the tiling reassociation matches the
    one-shot reference (≤1e-6) with no Metal.  Vectorised over ``(b, s, H)``."""
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
    NEG = mx.array(-3.0e38, dtype=mx.float32)

    if attend is not None:
        att = attend
        if att.dtype != mx.bool_:
            att = att >= mx.array(float("-1.5e38"), dtype=mx.float32)
    m_run = mx.full((b, s, H, 1), -3.0e38, dtype=mx.float32)
    d_run = mx.zeros((b, s, H, 1), dtype=mx.float32)
    acc = mx.zeros((b, s, H, hd), dtype=mx.float32)
    for c0 in range(0, T_use, tile):
        c1 = min(c0 + tile, T_use)
        KVc = KV[:, c0:c1, :]
        sc = mx.einsum("bshd,btd->bsht", q, KVc) * mx.array(scale, dtype=mx.float32)
        if attend is not None:
            ac = att[:, :, c0:c1]
            sc = mx.where(ac[:, :, None, :], sc, NEG)   # masked lane → NEG sentinel
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
    out = mx.where(denom > 0, acc / denom, mx.zeros_like(acc))
    if squeeze:
        return out.reshape(s, H, hd)
    return out
