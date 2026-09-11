"""K28 — fused mask + attention-sink softmax over the DSV4.1 prefill score tensor.

One ``mx.fast.metal_kernel`` that reads the raw ``[rows, H, T]`` f32 score tensor
**once per reduction**, applies the CSA candidate/causal mask and the per-head
value-0 attention sink, and writes the normalised softmax probabilities
``p = exp(s - m) / denom`` (masked keys → 0) in a single dispatch.

Motivation (W58 / KERNEL_LEDGER K28).  At 16,384 tokens the DSV4.1 attention
score path materialises a ``[rows=1024, 64 heads, T]`` f32 transient (up to ~6 GiB
at T≈24576).  The shipped eager softmax walks that transient ~4–6 times — the
``mx.where`` mask (read+write), then ``max`` (read), ``exp`` (read+write),
``sum`` (read) for the W50 *lean* fold-sink path, or a ``concatenate`` /
``softmax`` / slice trio for the default one-shot path — plus one or two full
T-wide intermediate allocations (``masked_scores`` / ``ex``).  Window-22 put the
reuse-layer ``softmax`` stage at 84.5 s (one-shot) / ~53 s (lean) of the 184 s
attention cost.  This kernel collapses mask + sink + softmax into **two device
reads and one write** of the transient with **zero T-wide intermediates**: an
online (max, denom) reduction (read #1) then a normalise-and-write pass (read #2).

Semantics preserved exactly (reference ``scripts/deepseek_v41/torchref/
ref_forward.py`` ``_k_sparse_attn`` and the model ``_sparse_attend_oneshot``
``fold_sink`` path):

  * ``s = scores · scale`` (``scale`` folded into q upstream on the lean path, so
    the wrapper passes ``scale=1.0`` there; otherwise the kernel applies it);
  * an additive/boolean mask (the model passes a boolean CSA/causal ``attend``;
    the wrapper converts it to the additive ``{0, -inf}`` form the kernel reads —
    ``mv >= FINITE_MIN`` keeps the key, ``score += mv`` supports a true additive
    bias);
  * the per-head **value-0 attention sink**: a virtual extra key whose score is
    ``attn_sink[head]`` and whose value is 0 — it adds ``exp(attn_sink[head] - m)``
    to the denominator and contributes nothing to the numerator.  The running max
    is taken over the unmasked keys **and** the sink (matching the model's
    ``m = max(max(scores), sink)``), so a fully-masked row collapses to the sink
    alone (denom 1, all probabilities 0 → zero attention output), never a NaN;
  * f32 throughout; ``metal::exp`` (precise, matching ``mx.exp``).

Exactness class: **reassociation-level** vs the eager f32 path (the threadgroup
tree reduction reorders the max / denom / value sums), expected ``max|Δ| ≤ 1e-6``
and greedy-argmax identical — the same float class as W50's split-K
(``score_chunked``) and the lean path (``score_lean``), NOT byte-identical.

Split-K hook: with ``normalize=False, return_stats=True`` the kernel writes the
**un-normalised** ``ex = exp(s - m)`` and emits the per-(row,head) ``(m, denom)``
as extra outputs, so a future split-K driver can merge chunks online.  The model
integration (``_sparse_attend_oneshot``) uses only the normalised one-shot path;
the chunked / split-K composition is scoped out and documented (W58 report).
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

#: Threadgroup size (threads per (row, head) reduction group).  Power of two: the
#: (max, denom) tree reduction halves the active lane count each step.  256 = 8
#: SIMD groups; each lane scans ~T/256 keys (≤96 at T=24576).  Threadgroup memory
#: use is 2·TG·4 B = 2 KiB (well under the 32 KiB limit).
_TG_DEFAULT = 256

#: Finite sentinel for "no key seen yet" and the additive-mask cutoff.  Using a
#: large *finite* negative (not -inf) keeps the online/​tree combine NaN-free: an
#: all-masked lane's state is ``(NEG, 0)`` and ``NEG - NEG = 0`` (``exp`` → 1), so
#: ``0·1 = 0`` rather than ``0·exp(nan)``.  A masked additive entry is stored as
#: ``-inf`` by the wrapper; ``mv >= FINITE_MIN`` (FINITE_MIN == NEG) rejects it.
_NEG = "-3.0e38f"


def _build_source(*, tg: int, has_mask: bool, sink_on: bool,
                  normalize: bool, return_stats: bool) -> str:
    """Assemble the Metal source for one structural variant.

    Conditional blocks are injected in Python (as ``mtplx/kernels/sdpa_2pass.py``
    does) rather than via ``#if`` — ``mx.fast.metal_kernel`` template constants are
    C++ template parameters, not preprocessor macros, so the preprocessor cannot
    branch on them.  ``H`` / ``T`` / ``scale`` are *runtime* scalar inputs so one
    compiled kernel serves every chunk width."""

    # --- mask block: reads the per-row additive mask (shared across heads) ------
    if has_mask:
        mask_decl = "float mv1 = mask[mbase + t];"
        mask_valid = "bool valid = (mv1 >= FINITE_MIN);"
        mask_add = "float add = mv1;"
        mask_decl2 = "float mv2 = mask[mbase + t];"
        mask_valid2 = "bool valid = (mv2 >= FINITE_MIN);"
        mask_add2 = "float add = mv2;"
    else:  # no-mask fast path: every key valid, no additive bias
        mask_decl = ""
        mask_valid = "bool valid = true;"
        mask_add = "float add = 0.0f;"
        mask_decl2 = ""
        mask_valid2 = "bool valid = true;"
        mask_add2 = "float add = 0.0f;"

    # --- sink fold: incorporate the per-head value-0 sink into (M, D) -----------
    if sink_on:
        sink_fold = """
        {
            float sv = sink[head];
            float Mf = metal::max(M, sv);
            D = D * metal::exp(M - Mf) + metal::exp(sv - Mf);   // sink state (sv, 1)
            M = Mf;
        }
        """
    else:
        sink_fold = ""

    # --- write block: normalised p = ex / D, or un-normalised ex (split-K) ------
    # Assigns the pre-declared ``pv`` (must NOT redeclare it -- the store reads the
    # outer ``pv`` after the if/else).
    if normalize:
        write_expr = "pv = (D > 0.0f) ? (ex / D) : 0.0f;"
    else:
        write_expr = "pv = ex;"

    if return_stats:
        stats_write = """
        if (lane == 0) {
            stats_m[gid] = M;
            stats_d[gid] = D;
        }
        """
    else:
        stats_write = ""

    return f"""
        using namespace metal;
        constexpr uint TG = {int(tg)};
        constexpr float NEG = {_NEG};
        constexpr float FINITE_MIN = {_NEG};

        const uint gid = threadgroup_position_in_grid.x;   // one group per (row,head)
        const uint lane = thread_position_in_threadgroup.x;
        const uint Hc = uint(H);
        const uint Tc = uint(T);
        const uint row = gid / Hc;
        const uint head = gid % Hc;
        const uint sbase = gid * Tc;    // scores / out base for this (row,head)
        const uint mbase = row * Tc;    // mask base for this row (shared over heads)
        const float sc = float(scale);

        threadgroup float tg_m[TG];
        threadgroup float tg_d[TG];

        // ---- pass 1 (device read #1): online (max, denom) over strided keys ----
        float lm = NEG;
        float ld = 0.0f;
        for (uint t = lane; t < Tc; t += TG) {{
            {mask_decl}
            {mask_valid}
            {mask_add}
            if (valid) {{
                float s = scores[sbase + t] * sc + add;
                float nm = metal::max(lm, s);
                ld = ld * metal::exp(lm - nm) + metal::exp(s - nm);
                lm = nm;
            }}
        }}
        tg_m[lane] = lm;
        tg_d[lane] = ld;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- tree reduction of the (m, d) online-softmax states ----------------
        for (uint stride = TG >> 1; stride > 0; stride >>= 1) {{
            if (lane < stride) {{
                float m1 = tg_m[lane];
                float d1 = tg_d[lane];
                float m2 = tg_m[lane + stride];
                float d2 = tg_d[lane + stride];
                float M = metal::max(m1, m2);
                float D = d1 * metal::exp(m1 - M) + d2 * metal::exp(m2 - M);
                tg_m[lane] = M;
                tg_d[lane] = D;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        // every lane reads the reduced state and folds the sink identically
        float M = tg_m[0];
        float D = tg_d[0];
        {sink_fold}

        // ---- pass 2 (device read #2 + write): normalise and store --------------
        for (uint t = lane; t < Tc; t += TG) {{
            {mask_decl2}
            {mask_valid2}
            {mask_add2}
            float pv;
            if (valid) {{
                float s = scores[sbase + t] * sc + add;
                float ex = metal::exp(s - M);
                {write_expr}
            }} else {{
                pv = 0.0f;
            }}
            out[sbase + t] = pv;
        }}
        {stats_write}
    """


@lru_cache(maxsize=None)
def _k28_kernel(*, tg: int, has_mask: bool, sink_on: bool,
                normalize: bool, return_stats: bool):
    """Build (and cache) the K28 kernel for one structural variant, or ``None`` if
    Metal is unavailable (CPU-only test hosts)."""
    if not mx.metal.is_available():
        return None
    inputs = ["scores", "H", "T", "scale"]
    if has_mask:
        inputs.append("mask")
    if sink_on:
        inputs.append("sink")
    outputs = ["out"]
    if return_stats:
        outputs += ["stats_m", "stats_d"]
    suffix = (
        f"tg{tg}"
        f"_{'m' if has_mask else 'nm'}"
        f"_{'s' if sink_on else 'ns'}"
        f"_{'norm' if normalize else 'raw'}"
        f"_{'stats' if return_stats else 'nostats'}"
    )
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv41_fused_softmax_{suffix}",
        input_names=inputs,
        output_names=outputs,
        source=_build_source(
            tg=tg, has_mask=has_mask, sink_on=sink_on,
            normalize=normalize, return_stats=return_stats,
        ),
    )


def kernel_available() -> bool:
    """Whether a Metal GPU is present so the kernel can build/dispatch.

    The DSV4.1 flag route also requires the default device be the GPU; callers
    (``deepseek_v41._prefill_softmax_kernel_use``) check that separately so a
    CPU-pinned worker test never dispatches Metal."""
    return bool(mx.metal.is_available())


def _prepare_scores(scores: mx.array):
    """Return (scores_3d[rows,H,T] contiguous f32, b_or_None, s_or_None, H, T)."""
    if scores.dtype != mx.float32:
        scores = scores.astype(mx.float32)
    if scores.ndim == 4:
        b, s, H, T = (int(d) for d in scores.shape)
        scores3 = mx.contiguous(scores.reshape(b * s, H, T))
        return scores3, b, s, H, T
    if scores.ndim == 3:
        rows, H, T = (int(d) for d in scores.shape)
        return mx.contiguous(scores), None, None, H, T
    raise ValueError(f"scores must be 3D [rows,H,T] or 4D [b,s,H,T], got {scores.shape}")


def _prepare_mask(attend, *, rows: int, T: int):
    """Convert the model's boolean ``attend`` (True = keep) into the additive
    ``{0, -inf}`` [rows, T] f32 mask the kernel reads, or ``None`` for the no-mask
    fast path.  Already-additive f32 masks are passed through (contiguous)."""
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
    add = add.reshape(rows, T)
    return mx.contiguous(add)


def fused_prefill_softmax(
    scores: mx.array,
    *,
    attend=None,
    attn_sink=None,
    scale: float = 1.0,
    tg: int = _TG_DEFAULT,
    normalize: bool = True,
    return_stats: bool = False,
):
    """Fused mask + attention-sink softmax (K28).

    Parameters
    ----------
    scores : ``[b, s, H, T]`` or ``[rows, H, T]`` f32 raw QK^T scores (contiguous
        preferred; up-cast/​made-contiguous otherwise).
    attend : boolean ``[b, s, T]`` / ``[rows, T]`` CSA/causal mask (True = keep),
        or an additive f32 mask, or ``None`` (no-mask fast path).  Shared across
        heads (broadcast over ``H`` in the kernel).
    attn_sink : per-head ``[H]`` f32 sink scores, or ``None`` (no sink).
    scale : multiplied into each raw score inside the kernel.  Pass ``1.0`` when
        the scale is already folded into q upstream (the lean path).
    normalize : ``True`` writes ``p = exp(s - m)/denom``; ``False`` writes the
        un-normalised ``ex`` (split-K).
    return_stats : also return per-(row,head) ``(m, denom)`` (split-K online merge).

    Returns
    -------
    ``p`` with the same shape/dtype (f32) as ``scores``.  When ``return_stats`` is
    set, ``(p, m, denom)`` with ``m``/``denom`` shaped ``[..., H]`` (leading dims of
    ``scores`` minus the T axis).
    """
    scores3, b, s, H, T = _prepare_scores(scores)
    rows = scores3.shape[0]
    n_groups = rows * H

    has_mask = attend is not None
    mask = _prepare_mask(attend, rows=rows, T=T) if has_mask else None
    sink_on = attn_sink is not None
    sink = attn_sink.astype(mx.float32).reshape(H) if sink_on else None

    kernel = _k28_kernel(
        tg=int(tg), has_mask=has_mask, sink_on=sink_on,
        normalize=bool(normalize), return_stats=bool(return_stats),
    )
    if kernel is None:  # no Metal — caller is responsible for the eager fallback
        raise RuntimeError("K28 fused softmax requires a Metal GPU (kernel_available() is False)")

    inputs = [scores3, int(H), int(T), float(scale)]
    if has_mask:
        inputs.append(mask)
    if sink_on:
        inputs.append(sink)

    out_shapes = [(rows, H, T)]
    out_dtypes = [mx.float32]
    if return_stats:
        out_shapes += [(n_groups,), (n_groups,)]
        out_dtypes += [mx.float32, mx.float32]

    result = kernel(
        inputs=inputs,
        grid=(int(tg) * int(n_groups), 1, 1),
        threadgroup=(int(tg), 1, 1),
        output_shapes=out_shapes,
        output_dtypes=out_dtypes,
    )
    if return_stats:
        p3, m_flat, d_flat = result
    else:
        (p3,) = result

    # reshape back to the caller's layout
    if b is not None:
        p = p3.reshape(b, s, H, T)
        stats_shape = (b, s, H)
    else:
        p = p3
        stats_shape = (rows, H)
    if return_stats:
        return p, m_flat.reshape(stats_shape), d_flat.reshape(stats_shape)
    return p
