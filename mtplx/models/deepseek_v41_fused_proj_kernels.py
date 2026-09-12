"""W101 (kernel-ledger K36): fused decode/verify attention PROJECTION-CHAIN glue
kernels for DeepSeek-V4.1-Flash.

Context
-------
W97-W99 (``docs/deepseek-v41/W97_ATTENTION_291MS.md``) took the exact-numerics
decode-attention dispatch ceiling to ~55-67 ``metal`` kernels per backbone layer:
the ``wo_a`` cache removed the per-token dequant, the lean casts deduped the
redundant f32 casts, and the K29 fused kernel collapsed the SDPA *core* (QK^T +
mask + sink softmax + PV) to ONE dispatch.  W99 §8.5's verdict: the remaining
count is dominated by the qkv/out **projection chains** -- the ``rmsnorm`` +
interleaved-RoPE + head/group layout **glue** between the (already single-dispatch)
``mx.quantized_matmul`` projections -- and reaching <=25 kernels/layer needs FUSED
custom Metal kernels for those chains.  That is this module.

Design (task W101): keep the quantized matmuls (``mx.quantized_matmul`` /
``gather_qmm`` -- MLX's tuned kernels, one dispatch each) and the grouped o-LoRA
einsum (one ``Matmul`` dispatch); fuse every *glue* region between them into a
single ``mx.fast.metal_kernel`` dispatch:

  * :func:`rmsnorm` -- the post-``wq_a`` q-latent RMSNorm (reference ``RMSNorm``,
    model.py L288-293: normalise in fp32, scale by fp32 weight, store at the input
    dtype).  One threadgroup per row; the sum-of-squares is a threadgroup tree
    reduction (reassociated vs ``mx.mean`` -> ROUNDING-CLASS, ~1e-6, far tighter
    than ``mx.fast.rms_norm``'s ~2-ULP bf16 divergence).
  * :func:`rope_heads` -- interleaved-complex-pair RoPE on the last ``rope_dim``
    dims of each of ``H`` heads (reference ``apply_rotary_emb`` L392-406), forward
    (q after ``wq_b``) or ``inverse`` (o before the o-LoRA down, removing the query
    rotation).  Pure elementwise (one thread per output scalar); no reduction.
  * :func:`rmsnorm_rope` -- the post-``wkv`` KV-latent RMSNorm **and** the k_pe RoPE
    on the last ``rope_dim`` of the 512-d latent, FUSED in one pass (one threadgroup
    per row), so the KV-prep glue is a single dispatch.

The grouped o-LoRA einsum stays ``mx.einsum`` (one tuned ``Matmul`` dispatch); the
caller feeds it the **bf16** ``wo_a`` operand and accumulates in fp32 internally
(MLX matmul always accumulates fp32), matching the reference bf16 einsum (model.py
L784-787) -- half the per-token ``wo_a`` read (67 MB vs the port's 134 MB f32
materialisation) and no f32 write.  ROUNDING-CLASS vs the port's f32 einsum; a
custom grouped GEMV is NOT written (``mx.einsum`` is already one dispatch and MLX's
matmul is faster than a hand GEMV -- a slower custom kernel at fewer dispatches is
still a loss).

Precision
---------
Following the K29 fix (W60 window-26), every kernel does its math in ``float`` and
uses ``metal::precise::`` for the transcendental-free arithmetic here (only mul/
add/rsqrt), reading inputs promoted to fp32 and writing at the requested output
dtype.  NONE of these kernels is byte-identical to the eager chain (the RMSNorm
tree reduction reassociates the fp32 sum; the bf16 einsum operand rounds ``wo_a``):
they are ROUNDING-CLASS, GPU-only, and gated behind ``MTPLX_DSV41_ATTN_FUSED_PROJ``
with an eager fallback on CPU / when the flag is off.  The caller owns the
device/flag/small-M gate; this module just builds and dispatches.

Engagement telemetry mirrors the K29 module (``reset_engagement`` / ``engagement``)
so an A/B receipt can tell "fused ran" from "fell back to eager", split by phase
(qkv-prep vs out-prep).
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

#: Default threads per group (key/reduction tile width; power of two).
_TG_DEFAULT = 128


# ---------------------------------------------------------------------------
# engagement telemetry (W101): count real fused-kernel dispatches per phase so an
# A/B receipt can tell "fused ran" from "armed but fell back to eager".
# ---------------------------------------------------------------------------
_ENGAGE_QKV = 0        #: qkv-prep layer-steps the fused kernels dispatched
_ENGAGE_OUT = 0        #: out-prep layer-steps the fused kernels dispatched
_ENGAGE_ROWS = 0       #: Σ rows (b·s) over fused layer-steps
_FALLBACK_QKV = 0      #: qkv-prep layer-steps that fell back to eager (armed+GPU)
_FALLBACK_OUT = 0      #: out-prep layer-steps that fell back to eager


def reset_engagement() -> None:
    """Zero the per-run engagement counters (call before each A/B arm)."""
    global _ENGAGE_QKV, _ENGAGE_OUT, _ENGAGE_ROWS, _FALLBACK_QKV, _FALLBACK_OUT
    _ENGAGE_QKV = _ENGAGE_OUT = _ENGAGE_ROWS = _FALLBACK_QKV = _FALLBACK_OUT = 0


def note_qkv(rows: int) -> None:
    """Record one qkv-prep layer-step the fused kernels dispatched (``rows`` = b·s)."""
    global _ENGAGE_QKV, _ENGAGE_ROWS
    _ENGAGE_QKV += 1
    _ENGAGE_ROWS += int(rows)


def note_out() -> None:
    """Record one out-prep layer-step the fused kernels dispatched."""
    global _ENGAGE_OUT
    _ENGAGE_OUT += 1


def note_fallback(phase: str) -> None:
    """Record one layer-step that fell back to eager (armed + on-GPU but an
    unsupported shape).  ``phase`` is ``"qkv"`` or ``"out"``."""
    global _FALLBACK_QKV, _FALLBACK_OUT
    if phase == "qkv":
        _FALLBACK_QKV += 1
    else:
        _FALLBACK_OUT += 1


def engagement() -> dict:
    """Counters since the last reset: ``qkv_calls`` / ``out_calls`` (layer-steps the
    fused kernels dispatched, per phase), ``rows`` (Σ ``b·s`` over qkv calls),
    ``qkv_fallbacks`` / ``out_fallbacks`` (armed-but-eager layer-steps)."""
    return {
        "qkv_calls": _ENGAGE_QKV,
        "out_calls": _ENGAGE_OUT,
        "rows": _ENGAGE_ROWS,
        "qkv_fallbacks": _FALLBACK_QKV,
        "out_fallbacks": _FALLBACK_OUT,
    }


def kernel_available() -> bool:
    """Whether a Metal GPU is present so the kernels can build/dispatch."""
    return bool(mx.metal.is_available())


# ---------------------------------------------------------------------------
# Metal sources.  ``%%TG%%`` / ``%%D%%`` / ``%%RD%%`` / ``%%H%%`` / ``%%HD%%`` are
# compile-time constants (one lru_cache'd kernel per shape); ``S`` (seq rows per
# batch, for the cos/sin row map) is a runtime scalar.  Assembled by ``.replace``
# (the Metal body is dense with literal braces).
# ---------------------------------------------------------------------------

# RMSNorm: one threadgroup per row of a [rows, D] matrix.  Reference RMSNorm
# (model.py L288-293): var = mean(x_f32^2) over D; x *= rsqrt(var+eps); out =
# (weight_f32 * x) stored at the output dtype.  The sum-of-squares is a threadgroup
# tree reduction (reassociated -> rounding-class).
_RMSNORM_TEMPLATE = r"""
    using namespace metal;
    constexpr uint TG = %%TG%%;
    constexpr uint D  = %%D%%;

    const uint row  = threadgroup_position_in_grid.x;
    const uint lane = thread_position_in_threadgroup.x;
    const float epsf = float(eps);
    const uint base = row * D;

    threadgroup float red[TG];

    // partial sum of squares over this lane's strided slice of the row
    float ss = 0.0f;
    for (uint i = lane; i < D; i += TG) {
        float v = float(x[base + i]);
        ss += v * v;
    }
    red[lane] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = TG >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) { red[lane] = red[lane] + red[lane + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = red[0] / float(D);
    float inv = metal::precise::rsqrt(mean + epsf);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lane; i < D; i += TG) {
        float v = float(x[base + i]) * inv;
        out[base + i] = static_cast<T>(float(weight[i]) * v);
    }
"""

# RMSNorm + interleaved RoPE on the last RD dims, FUSED (one threadgroup per row of
# a [rows, D] matrix; D == head_dim for the KV latent).  Computes the normed row
# into threadgroup memory, then rotates the last RD dims as adjacent complex pairs
# (reference apply_rotary_emb, forward direction).  cos/sin are [S, RD/2]; row r
# uses seq index r % S.
_RMSNORM_ROPE_TEMPLATE = r"""
    using namespace metal;
    constexpr uint TG = %%TG%%;
    constexpr uint D  = %%D%%;
    constexpr uint RD = %%RD%%;
    constexpr uint HALF = RD / 2;

    const uint row  = threadgroup_position_in_grid.x;
    const uint lane = thread_position_in_threadgroup.x;
    const uint Sc = uint(S);
    const uint s_idx = (Sc > 0u) ? (row % Sc) : 0u;
    const float epsf = float(eps);
    const uint base = row * D;
    const uint cs_base = s_idx * HALF;

    threadgroup float red[TG];
    threadgroup float nrm[D];   // the normed row (pre-RoPE), shared for the pair reads

    float ss = 0.0f;
    for (uint i = lane; i < D; i += TG) {
        float v = float(x[base + i]);
        ss += v * v;
    }
    red[lane] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = TG >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) { red[lane] = red[lane] + red[lane + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = metal::precise::rsqrt(red[0] / float(D) + epsf);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lane; i < D; i += TG) {
        nrm[i] = float(weight[i]) * (float(x[base + i]) * inv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // head part (< D-RD): copy through; tail: interleaved-pair RoPE
    const uint tail0 = D - RD;
    for (uint i = lane; i < D; i += TG) {
        float o;
        if (i < tail0) {
            o = nrm[i];
        } else {
            uint j = i - tail0;          // 0..RD-1 within the rope tail
            uint p = j >> 1;             // pair index
            float c = float(cos[cs_base + p]);
            float sn = float(sin[cs_base + p]);
            float v0 = nrm[tail0 + 2u * p];
            float v1 = nrm[tail0 + 2u * p + 1u];
            o = (j & 1u) ? (v0 * sn + v1 * c) : (v0 * c - v1 * sn);
        }
        out[base + i] = static_cast<T>(o);
    }
"""

# Interleaved RoPE on the last RD dims of each of H heads in a [rows, H, HD] tensor.
# Pure elementwise: one thread per output scalar.  ``%%SIGN%%`` is ``+`` (forward)
# or ``-`` (inverse: sin -> -sin, removes the query rotation from the attn output).
# cos/sin are [S, RD/2]; row r uses seq index r % S.
_ROPE_HEADS_TEMPLATE = r"""
    using namespace metal;
    constexpr uint H  = %%H%%;
    constexpr uint HD = %%HD%%;
    constexpr uint RD = %%RD%%;
    constexpr uint HALF = RD / 2;
    constexpr uint TAIL0 = HD - RD;

    const uint gid = thread_position_in_grid.x;   // one thread per (row,head,d)
    const uint total = uint(rows) * H * HD;
    if (gid >= total) return;

    const uint d    = gid % HD;
    const uint tmp  = gid / HD;
    const uint head = tmp % H;
    const uint row  = tmp / H;
    const uint Sc = uint(S);
    const uint s_idx = (Sc > 0u) ? (row % Sc) : 0u;
    const uint head_base = (row * H + head) * HD;

    float o;
    if (d < TAIL0) {
        o = float(x[head_base + d]);
    } else {
        uint j = d - TAIL0;
        uint p = j >> 1;
        uint cs = s_idx * HALF + p;
        float c = float(cos[cs]);
        float sn = %%SIGN%% float(sin[cs]);
        float v0 = float(x[head_base + TAIL0 + 2u * p]);
        float v1 = float(x[head_base + TAIL0 + 2u * p + 1u]);
        o = (j & 1u) ? (v0 * sn + v1 * c) : (v0 * c - v1 * sn);
    }
    out[gid] = static_cast<T>(o);
"""


def _sub(src: str, **kw) -> str:
    for k, v in kw.items():
        src = src.replace("%%" + k + "%%", str(v))
    return src


@lru_cache(maxsize=None)
def _rmsnorm_kernel(*, tg: int, d: int):
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv41_fp_rmsnorm_tg{tg}_d{d}",
        input_names=["x", "weight", "eps"],
        output_names=["out"],
        source=_sub(_RMSNORM_TEMPLATE, TG=tg, D=d),
    )


@lru_cache(maxsize=None)
def _rmsnorm_rope_kernel(*, tg: int, d: int, rd: int):
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv41_fp_rmsnorm_rope_tg{tg}_d{d}_rd{rd}",
        input_names=["x", "weight", "eps", "cos", "sin", "S"],
        output_names=["out"],
        source=_sub(_RMSNORM_ROPE_TEMPLATE, TG=tg, D=d, RD=rd),
    )


@lru_cache(maxsize=None)
def _rope_heads_kernel(*, h: int, hd: int, rd: int, inverse: bool):
    if not mx.metal.is_available():
        return None
    sign = "-" if inverse else "+"
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv41_fp_rope_h{h}_hd{hd}_rd{rd}_{'inv' if inverse else 'fwd'}",
        input_names=["x", "cos", "sin", "rows", "S"],
        output_names=["out"],
        source=_sub(_ROPE_HEADS_TEMPLATE, H=h, HD=hd, RD=rd, SIGN=sign),
    )


_NO_METAL = "W101 fused-proj kernels require a Metal GPU (kernel_available() is False)"


def rmsnorm(x: mx.array, weight: mx.array, eps: float, *, tg: int = _TG_DEFAULT,
            out_dtype=None) -> mx.array:
    """Fused RMSNorm of ``x`` ``[..., D]`` over the last axis (one dispatch).

    ``weight`` is ``[D]``.  Reference math: fp32 mean-of-squares, ``rsqrt(var+eps)``,
    scale by fp32 ``weight``, store at ``out_dtype`` (default ``x.dtype``).  The
    sum-of-squares reduction is reassociated (threadgroup tree) -> ROUNDING-CLASS."""
    d = int(x.shape[-1])
    rows = int(x.size // d)
    out_dtype = x.dtype if out_dtype is None else out_dtype
    # Native dtype in, promoted to float inside the kernel (``float(x[i])``); the
    # metal_kernel makes its inputs row-contiguous, so no explicit cast/contiguous
    # dispatch is added around the fused kernel.  ``reshape`` is a view.
    xr = x.reshape(rows, d)
    kernel = _rmsnorm_kernel(tg=int(tg), d=d)
    if kernel is None:
        raise RuntimeError(_NO_METAL)
    (out,) = kernel(
        inputs=[xr, weight, float(eps)],
        template=[("T", out_dtype)],
        grid=(int(tg) * rows, 1, 1),
        threadgroup=(int(tg), 1, 1),
        output_shapes=[(rows, d)],
        output_dtypes=[out_dtype],
    )
    return out.reshape(x.shape)


def rmsnorm_rope(x: mx.array, weight: mx.array, eps: float, cos: mx.array,
                 sin: mx.array, *, tg: int = _TG_DEFAULT, out_dtype=None) -> mx.array:
    """Fused RMSNorm + interleaved RoPE on the last ``rope_dim`` dims of ``x``
    ``[..., D]`` (one dispatch; the KV-latent prep).

    ``cos`` / ``sin`` are ``[S, rope_dim//2]`` fp32 (``S`` = seq rows per batch; row
    ``r`` uses seq index ``r % S``).  RMSNorm over the full ``D``, then the last
    ``rope_dim`` are rotated as adjacent complex pairs (forward).  ROUNDING-CLASS."""
    d = int(x.shape[-1])
    rows = int(x.size // d)
    rd = int(cos.shape[-1]) * 2
    out_dtype = x.dtype if out_dtype is None else out_dtype
    s = rows if cos.shape[0] == 0 else int(cos.shape[0])
    xr = x.reshape(rows, d)
    kernel = _rmsnorm_rope_kernel(tg=int(tg), d=d, rd=rd)
    if kernel is None:
        raise RuntimeError(_NO_METAL)
    (out,) = kernel(
        inputs=[xr, weight, float(eps), cos, sin, int(s)],
        template=[("T", out_dtype)],
        grid=(int(tg) * rows, 1, 1),
        threadgroup=(int(tg), 1, 1),
        output_shapes=[(rows, d)],
        output_dtypes=[out_dtype],
    )
    return out.reshape(x.shape)


def rope_heads(x: mx.array, cos: mx.array, sin: mx.array, *, inverse: bool = False,
               out_dtype=None) -> mx.array:
    """Interleaved RoPE on the last ``rope_dim`` dims of each of ``H`` heads in
    ``x`` ``[..., H, HD]`` (one dispatch; pure elementwise).

    ``cos`` / ``sin`` are ``[S, rope_dim//2]`` fp32.  ``inverse`` conjugates the
    rotation (removes the query rotation from the attention output).  Output at
    ``out_dtype`` (default ``x.dtype``); the head part (< ``HD-rope_dim``) copies
    through.  ROUNDING-CLASS (bf16 store)."""
    if x.ndim < 2:
        raise ValueError(f"rope_heads expects [..., H, HD], got {x.shape}")
    h, hd = int(x.shape[-2]), int(x.shape[-1])
    rows = int(x.size // (h * hd))
    rd = int(cos.shape[-1]) * 2
    out_dtype = x.dtype if out_dtype is None else out_dtype
    s = rows if cos.shape[0] == 0 else int(cos.shape[0])
    xr = x.reshape(rows, h, hd)
    kernel = _rope_heads_kernel(h=h, hd=hd, rd=rd, inverse=bool(inverse))
    if kernel is None:
        raise RuntimeError(_NO_METAL)
    total = rows * h * hd
    tg = 256
    grid = ((total + tg - 1) // tg) * tg
    (out,) = kernel(
        inputs=[xr, cos, sin, int(rows), int(s)],
        template=[("T", out_dtype)],
        grid=(int(grid), 1, 1),
        threadgroup=(int(tg), 1, 1),
        output_shapes=[(rows, h, hd)],
        output_dtypes=[out_dtype],
    )
    return out.reshape(x.shape)


# ---------------------------------------------------------------------------
# CPU-safe pure-MLX references (no Metal) -- unit-test oracles the GPU kernels are
# gated ROUNDING-CLASS against, and the shape/plumbing check on a CPU-pinned host.
# These are the SAME math as the eager chain in Attention._attend.
# ---------------------------------------------------------------------------
def rmsnorm_reference(x, weight, eps, out_dtype=None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    xf = x.astype(mx.float32)
    var = mx.mean(mx.square(xf), axis=-1, keepdims=True)
    xf = xf * mx.rsqrt(var + eps)
    return (weight.astype(mx.float32) * xf).astype(out_dtype)


def _rope_last_reference(x, cos, sin, inverse=False, out_dtype=None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    rd = int(cos.shape[-1]) * 2
    head = x[..., :-rd]
    tail = x[..., -rd:]
    extra = tail.ndim - 3
    shape = [cos.shape[0]] + [1] * extra + [cos.shape[-1]]
    c = cos.reshape(shape).astype(mx.float32)
    s = sin.reshape(shape).astype(mx.float32)
    if inverse:
        s = -s
    t = tail.astype(mx.float32)
    tshape = t.shape
    t = t.reshape(*tshape[:-1], tshape[-1] // 2, 2)
    x0, x1 = t[..., 0], t[..., 1]
    r0 = x0 * c - x1 * s
    r1 = x0 * s + x1 * c
    roped = mx.stack([r0, r1], axis=-1).reshape(tshape)
    if head.shape[-1] == 0:
        return roped.astype(out_dtype)
    return mx.concatenate([head.astype(mx.float32), roped], axis=-1).astype(out_dtype)


def rmsnorm_rope_reference(x, weight, eps, cos, sin, out_dtype=None):
    normed = rmsnorm_reference(x, weight, eps, out_dtype=mx.float32)
    return _rope_last_reference(normed, cos, sin, inverse=False, out_dtype=out_dtype)


def rope_heads_reference(x, cos, sin, inverse=False, out_dtype=None):
    return _rope_last_reference(x, cos, sin, inverse=inverse, out_dtype=out_dtype)
