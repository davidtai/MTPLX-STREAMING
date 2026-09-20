"""F35: Metal kernels that run DeepSeek-V4.1 expert projections straight from eschamoe K=3 trellis codes.

Interface mirrors the retained packed lane's ``make_projection`` (kernels.py): one (token, expert) *assignment*
per grid.z / grid.y row, ``ids[assignment]`` = the expert's bank index, weights read from a bank array.  The bank
holds eschamoe tiles: ``code[capacity, IN/16, OUT/16, 16*K] int16`` (a 16x16 tile of 256 weights per 48 int16).
The activation must already be in the trellis domain (``xh = T128(x)`` over 128-blocks of the input dim); the
output is in the trellis domain too (caller applies ``T128`` and the per-output ``rout`` scale).  Decode per weight
is the eschamoe hash codebook (bit-exact with ``escha_decode_ref.py``):
    r = (window * 3417055213) & 0xffffffff;  val = f16((r & 0x8fff) ^ 0x3b60) + f16(((r >> 16) & 0x8fff) ^ 0x3b60)

Two kernels:
  * ``qmv``  -- one simdgroup per OUTPUT element, lanes stride the contraction (the higgs/dusterbloom layout that
               ``mtplx/eschamoe.py::escha_qmv`` uses); simplest, reference-correct, redundant tile fetches.
  * ``tile`` -- one simdgroup per (assignment, output tile column): each lane assembles its 64-bit window once per
               input tile (warp-assembly decode, 8 weights per lane), keeps 8 per-lane accumulators and reduces the
               16 outputs once at the end -> every code byte is fetched exactly once per assignment.
Both take fp32 activations and produce fp32 outputs; accumulation order differs from a dense matmul, so results are
rounding-class vs the reference (asserted to ~1e-3 relative in the GPU harness).
"""
from __future__ import annotations

import mlx.core as mx

MAGIC = 3417055213
K_BITS = 3
NW = 16 * K_BITS  # int16 words per 16x16 tile

_KERNELS: dict = {}

# eschamoe codebook constants as a u32[4] buffer (kept identical to mtplx/eschamoe.py::_CB)
CB = mx.array([MAGIC, 0, 0x8FFF8FFF, 0x3B603B60], dtype=mx.uint32)


def _qmv_source(IN: int, OUT: int) -> str:
    TK, TN, K = IN // 16, OUT // 16, K_BITS
    return f"""
        threadgroup float x_sh[{IN}];
        uint row = threadgroup_position_in_grid.y;                  // assignment
        uint tid = thread_position_in_threadgroup.x;
        uint sg = tid >> 5; uint lane = tid & 31u;
        for (uint i = tid; i < {IN}u; i += 128u) x_sh[i] = xh[row * {IN}u + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint o = threadgroup_position_in_grid.x * 4u + sg;          // output element
        if (o >= {OUT}u) return;
        uint tn = o >> 4; uint c = o & 15u; uint cb2 = (c >> 3) & 1u; uint c7 = c & 7u;
        const device short* base = code + ulong(ids[row]) * {TK * TN * NW}ul;
        uint q = lane & 3u; uint rh = (lane >> 2) & 1u;
        uint t = 4u * (4u * c7 + q) + 2u * cb2 + rh; uint r0 = 8u * rh + 2u * q;
        uint b0 = 2u * t * {K}u + {K + 256 * K}u - 16u; uint b2 = b0 + {K + 16}u;
        uint i0 = (b0 / 32u) % {8 * K}u; uint i1w = (b2 - 1u) / 32u; uint s1 = (i1w + 1u) * 32u - b2; uint i1 = i1w % {8 * K}u;
        float acc = 0.0f;
        for (uint tk = lane >> 3; tk < {TK}u; tk += 4u) {{
            const device short* tile = base + (tk * {TN}u + tn) * {NW}u;
            uint w0 = uint(ushort(tile[2u * i0])) | (uint(ushort(tile[2u * i0 + 1u])) << 16);
            uint wb = uint(ushort(tile[2u * i1])) | (uint(ushort(tile[2u * i1 + 1u])) << 16);
            ulong pair = (ulong(w0) << 32) | ulong(wb);
            uint w1 = uint(pair >> s1);
            uint x0 = ((w1 >> {K}u) & 0xFFFFu) * cb[0] + cb[1]; x0 = (x0 & cb[2]) ^ cb[3];
            uint x1 = (w1 & 0xFFFFu) * cb[0] + cb[1]; x1 = (x1 & cb[2]) ^ cb[3];
            half2 h0 = as_type<half2>(x0); half2 h1 = as_type<half2>(x1);
            float v0 = float(half(float(h0.x) + float(h0.y)));
            float v1 = float(half(float(h1.x) + float(h1.y)));
            acc = fma(x_sh[tk * 16u + r0], v0, acc);
            acc = fma(x_sh[tk * 16u + r0 + 1u], v1, acc);
        }}
        acc = simd_sum(acc);
        if (lane == 0u) out[row * {OUT}u + o] = acc;
    """


def _tile_body(IN: int, OUT: int, base_expr: str) -> str:
    """Shared body of the tile kernel; ``base_expr`` is the C expression for the (assignment) bank base pointer.

    Everything after ``base`` is IDENTICAL between the contiguous-bank ``tile`` variant and the stride-aware
    ``tile_strided`` variant — only the pointer arithmetic that locates this assignment's projection code differs.
    One simdgroup per (assignment, output tile column tn).  Lane L assembles its 64-bit window v once per input
    tile (eschamoe warp assembly: lane_a/lane_b/lane_p tables) and decodes its 8 weights m=0..7 whose tile positions
    (row dr, col dc) come from the per-(lane, m) tables ``pos_r``/``pos_c``.  acc[m] accumulates x[dr] * w over all
    input tiles; at the end the 16 columns are reduced across lanes through threadgroup memory."""
    TK, TN, K = IN // 16, OUT // 16, K_BITS
    return f"""
        uint row = threadgroup_position_in_grid.y;                  // assignment
        uint tn = threadgroup_position_in_grid.x;                   // output tile column (16 outputs)
        uint lane = thread_index_in_simdgroup;
        threadgroup float red[16][32];
        const device short* base = {base_expr};
        const device float* xp = xh + row * {IN}u;
        uint wa = uint(lane_a[lane] >> 1); uint wb = uint(lane_b[lane] >> 1); uint lp = uint(lane_p[lane]);
        float acc[8];
        for (uint m = 0; m < 8u; ++m) acc[m] = 0.0f;
        uint dr[8]; uint dc[8];
        for (uint m = 0; m < 8u; ++m) {{ dr[m] = uint(pos_r[lane * 8u + m]); dc[m] = uint(pos_c[lane * 8u + m]); }}
        #pragma clang loop unroll(disable)
        for (uint tk = 0; tk < {TK}u; ++tk) {{
            const device short* tile = base + (tk * {TN}u + tn) * {NW}u;
            uint rd8 = uint(ushort(tile[wa])) | (uint(ushort(tile[wa + 1u])) << 16);
            uint rd9 = uint(ushort(tile[wb])) | (uint(ushort(tile[wb + 1u])) << 16);
            // The lane's 8 windows sit at bit offsets 3m (m = 0..7) of v = (rd9:rd8) >> lp.  Windows 0..5 end at
            // bit 31 at most, 6..7 need bits 18..40: two 32-bit funnel-shifted words cover all eight without any
            // 64-bit shift in the loop (Apple GPUs emulate ulong shifts with several 32-bit ops).
            // lane_p (lp) is one of {0, 8, 16, 24}: funnel-shift the two 32-bit halves accordingly.
            uint v0 = (lp == 0u) ? rd8 : ((rd8 >> lp) | (rd9 << (32u - lp)));      // bits 0..31 of v
            uint v1 = (lp < 16u) ? ((rd8 >> (lp + 16u)) | (rd9 << (16u - lp)))     // bits 16..47 of v
                                 : (rd9 >> (lp - 16u));
            for (uint m = 0; m < 6u; ++m) {{
                uint window = (v0 >> ({K}u * m)) & 0xffffu;
                uint rr = window * cb[0]; rr = (rr & cb[2]) ^ cb[3];
                half2 h = as_type<half2>(rr);
                float w = float(half(float(h.x) + float(h.y)));
                acc[m] = fma(xp[tk * 16u + dr[m]], w, acc[m]);
            }}
            for (uint m = 6u; m < 8u; ++m) {{
                uint window = (v1 >> ({K}u * m - 16u)) & 0xffffu;
                uint rr = window * cb[0]; rr = (rr & cb[2]) ^ cb[3];
                half2 h = as_type<half2>(rr);
                float w = float(half(float(h.x) + float(h.y)));
                acc[m] = fma(xp[tk * 16u + dr[m]], w, acc[m]);
            }}
        }}
        for (uint c = 0; c < 16u; ++c) red[c][lane] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint m = 0; m < 8u; ++m) red[dc[m]][lane] += acc[m];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 16u) {{
            float s = 0.0f;
            for (uint l = 0; l < 32u; ++l) s += red[lane][l];
            out[row * {OUT}u + tn * 16u + lane] = s;
        }}
    """


def _tile_source(IN: int, OUT: int) -> str:
    """Contiguous per-projection bank ``code[capacity, IN/16, OUT/16, 48]``: base = ids[row] * (one projection)."""
    return _tile_body(IN, OUT, f"code + ulong(ids[row]) * {(IN // 16) * (OUT // 16) * NW}ul")


# --- tcq3 whole-record bank geometry (F38 transcode_bank.py record: 13,290,496 B = 6,645,248 int16 words) ---
# Each cache-row slot holds ONE whole tcq3 record; a projection's code segment starts at a fixed word offset.
# stride = 6,645,248 words; gate code @ 0, up code @ 2,214,144, down code @ 4,428,288 (routs sit between them).
TCQ3_RECORD_WORDS = 6_645_248
TCQ3_CODE_WORD_OFFSETS = {"gate_proj": 0, "up_proj": 2_214_144, "down_proj": 4_428_288}
# Per-projection code segment length in int16 words (gate/up: IN=5120,OUT=2304; down: IN=2304,OUT=5120 -> both 2,211,840)
TCQ3_CODE_WORDS = {"gate_proj": 2_211_840, "up_proj": 2_211_840, "down_proj": 2_211_840}


def _tile_strided_source(IN: int, OUT: int, row_stride_words: int, proj_word_offset: int) -> str:
    """Stride-aware bank: ONE cache row per record holds all three projections + routs contiguously.

    base = code + ids[row] * row_stride_words + proj_word_offset  (both in int16 words).  The kernel body is the
    verbatim ``_tile_body`` string used by ``_tile_source`` — only the base pointer arithmetic changes."""
    return _tile_body(IN, OUT, f"code + ulong(ids[row]) * {row_stride_words}ul + {proj_word_offset}ul")


def make_tcq_projection(n: int, k: int, variant: str = "tile"):
    """Kernel for OUT=n, IN=k (DeepSeek-V4.1 expert geometry: (2304, 5120) gate/up, (5120, 2304) down)."""
    if (n, k) not in ((2304, 5120), (5120, 2304)):
        raise ValueError("only the native target expert geometry is admitted")
    key = (variant, n, k)
    if key in _KERNELS:
        return _KERNELS[key]
    if variant == "qmv":
        kern = mx.fast.metal_kernel(name=f"dsv41_tcq_qmv_{n}_{k}", input_names=["xh", "ids", "code", "cb"],
                                    output_names=["out"], source=_qmv_source(k, n))
    elif variant == "tile":
        kern = mx.fast.metal_kernel(name=f"dsv41_tcq_tile_{n}_{k}",
                                    input_names=["xh", "ids", "code", "cb", "lane_a", "lane_b", "lane_p", "pos_r", "pos_c"],
                                    output_names=["out"], source=_tile_source(k, n))
    else:
        raise ValueError(variant)
    _KERNELS[key] = kern
    return kern


def run_tcq_projection(kern, variant: str, xh: mx.array, ids: mx.array, code: mx.array, n: int, tables=None) -> mx.array:
    """xh [rows, IN] f32 (trellis domain), ids [rows] u32 bank indices, code bank int16 -> out [rows, OUT] f32."""
    rows = int(xh.shape[0])
    if variant == "qmv":
        (out,) = kern(inputs=[xh, ids, code, CB], output_shapes=[(rows, n)], output_dtypes=[mx.float32],
                      grid=(((n + 3) // 4) * 128, rows, 1), threadgroup=(128, 1, 1))
    else:
        lane_a, lane_b, lane_p, pos_r, pos_c = tables
        (out,) = kern(inputs=[xh, ids, code, CB, lane_a, lane_b, lane_p, pos_r, pos_c],
                      output_shapes=[(rows, n)], output_dtypes=[mx.float32],
                      grid=(32 * (n // 16), rows, 1), threadgroup=(32, 1, 1))
    return out


def make_tcq_projection_strided(n: int, k: int, component: str,
                                row_stride_words: int = TCQ3_RECORD_WORDS):
    """Stride-aware ``tile`` kernel for one projection ``component`` reading from a WHOLE-RECORD bank.

    OUT=n, IN=k (gate/up: n=2304,k=5120; down: n=5120,k=2304).  ``code`` at run time is the whole-record int16 bank
    ``[capacity, row_stride_words]``; the projection's code starts at ``TCQ3_CODE_WORD_OFFSETS[component]``.
    """
    if (n, k) not in ((2304, 5120), (5120, 2304)):
        raise ValueError("only the native target expert geometry is admitted")
    if component not in TCQ3_CODE_WORD_OFFSETS:
        raise ValueError(f"unknown projection component {component!r}")
    proj_word_offset = TCQ3_CODE_WORD_OFFSETS[component]
    key = ("tile_strided", n, k, row_stride_words, proj_word_offset)
    if key in _KERNELS:
        return _KERNELS[key]
    kern = mx.fast.metal_kernel(
        name=f"dsv41_tcq_tile_strided_{n}_{k}_{component}",
        input_names=["xh", "ids", "code", "cb", "lane_a", "lane_b", "lane_p", "pos_r", "pos_c"],
        output_names=["out"], source=_tile_strided_source(k, n, row_stride_words, proj_word_offset))
    _KERNELS[key] = kern
    return kern


def run_tcq_projection_strided(kern, xh: mx.array, ids: mx.array, code: mx.array, n: int, tables) -> mx.array:
    """xh [rows, IN] f32 (trellis domain), ids [rows] u32 slot indices into the whole-record bank,
    code int16 whole-record bank [capacity, TCQ3_RECORD_WORDS] -> out [rows, OUT] f32 ( = xh @ W_q )."""
    rows = int(xh.shape[0])
    lane_a, lane_b, lane_p, pos_r, pos_c = tables
    (out,) = kern(inputs=[xh, ids, code, CB, lane_a, lane_b, lane_p, pos_r, pos_c],
                  output_shapes=[(rows, n)], output_dtypes=[mx.float32],
                  grid=(32 * (n // 16), rows, 1), threadgroup=(32, 1, 1))
    return out
