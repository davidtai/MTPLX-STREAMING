"""GPU (Metal) trellis encoder for the eschamoe K=3 format  (DSV4.1 transcode lane, F38).

The CPU encoder (``tcq_encode.py``) needs ~6 ms per 16x16 tile; the expert bank has 2.1e9 tiles.  This module runs
the same search on the GPU with one SIMD group (32 lanes) per tile and NO threadgroup barriers (the 8192-state exact
Viterbi would need a 32 KB state exchange per symbol, i.e. 512 threadgroup barriers per tile; the beam search keeps
its state in registers and 3 KB of simdgroup-private scratch).

Kernels (all take ``targets`` [T, 256] fp32 = the tile's quantization targets in cycle order):
  beam    one simdgroup per tile.  Beam width W (8 beams per lane for W=256).  Per symbol each lane expands its
          beams x 8 symbols (emission em(w,t) = DEC[w]^2 - 2 t DEC[w], DEC = the hash codebook, as on the CPU),
          the simdgroup picks a cost threshold by bisection between the global minimum candidate and a PROVEN upper
          bound of the W-th best (the max over beams of each beam's best successor: those W candidates are all <= it),
          keeps every candidate at or under the threshold (count in [W/2, W], truncated to W in lane order when the
          bisection budget runs out) and records (parent, symbol) per kept slot for the traceback.  The start states
          are the fixed states 0..W-1 at cost 0: the seam repair re-solves symbols 0..7 exactly afterwards, so the
          open-chain start hardly matters (the trellis forgets a state in 5 symbols).
  trace   one thread per tile: follows parent[] from the best final slot -> new3 [T, 256] (3 new bits per symbol).
  seam    == tcq_encode.repair_seam(span=8): with state_0 pinned to the wrap state (from symbols 251..255) and state_8
          pinned to the boundary state (from symbols 3..7), only 11 of the 24 new bits of symbols 0..7 are free; the
          simdgroup enumerates all 2048 patterns and keeps the cheapest (the CPU's fixed-ends Viterbi optimum).
  pack    new3 -> 48 int16 code words per tile using the decoder's bit placement (``cycle_tables()['newbit_pos']``).

Driver: :func:`encode_projection_gpu` takes ``W_esch`` [in, out] fp32 (eschamoe orientation) and returns the code
[in/16, out/16, 48] int16 plus ``rout`` fp16 exactly as ``tcq_encode.encode_projection`` / ``serialize`` expect.  The
scales are computed on the GPU with the CPU formula (rin = 1, rout = column RMS of B_in W / codebook RMS), rounded to
fp16 BEFORE the targets are formed so the quantizer optimizes for the scale the runtime will apply.

Requires the GPU lock (all callers run inside ``gpu_window.sh``).  Import is GPU-free.
"""
from __future__ import annotations

import math
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tcq_encode as tq  # noqa: E402  (CPU reference: tables, codebook, scale algebra)

MAGIC = 3417055213
K_BITS = 3
NW = 16 * K_BITS
INF_STR = "3.0e38f"

_KERNELS: dict = {}
_TABLES: dict = {}


# ------------------------------------------------------------------ tables

def tables() -> dict:
    """cycle_order (uint16 [256]), sym_of_bit / k_of_bit (uint8 [768]) as mx arrays, plus dec_rms."""
    if "t" in _TABLES:
        return _TABLES["t"]
    ct = tq.cycle_tables(3)
    newbit = ct["newbit_pos"]                         # [256, 3] tile-bit index of symbol p's k-th new bit
    sym_of_bit = np.zeros(768, np.uint8)
    k_of_bit = np.zeros(768, np.uint8)
    for p in range(256):
        for k in range(3):
            b = int(newbit[p, k])
            sym_of_bit[b] = p
            k_of_bit[b] = k
    out = {
        "cycle_order": mx.array(ct["cycle_order"].astype(np.uint16)),
        "cycle_order_np": ct["cycle_order"],
        "sym_of_bit": mx.array(sym_of_bit),
        "k_of_bit": mx.array(k_of_bit),
        "dec_rms": tq.codebook_rms(),
        "ct": ct,
    }
    _TABLES["t"] = out
    return out


# ------------------------------------------------------------------ kernel sources

_EM = """
            uint rr = (w) * {MAGIC}u; rr = (rr & 0x8FFF8FFFu) ^ 0x3B603B60u;
            half2 hh = as_type<half2>(rr);
            float v = float(half(float(hh.x) + float(hh.y)));
            float vv = v * v; float tv = t2 * v; float em = vv - tv;
"""


def _beam_source(W: int, rounds: int) -> str:
    BL = W // 32
    NC = BL * 8
    em = _EM.format(MAGIC=MAGIC)
    return f"""
        uint tile = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        threadgroup float  sc_cost[{W}];
        threadgroup ushort sc_state[{W}];
        threadgroup uchar  sc_par[{W}];
        threadgroup uchar  sc_sym[{W}];
        const device float* tp = targets + ulong(tile) * 256ul;
        device uchar* par_out = parent + ulong(tile) * {256 * W}ul;
        device uchar* sym_out = symbol + ulong(tile) * {256 * W}ul;
        const float INF = {INF_STR};
        float bc[{BL}]; uint bs[{BL}];
        for (uint i = 0; i < {BL}u; ++i) {{ bc[i] = 0.0f; bs[i] = lane * {BL}u + i; }}
        uint ntrunc = 0u; uint nshort = 0u;
        #pragma clang loop unroll(disable)
        for (uint p = 0; p < 256u; ++p) {{
            float t2 = 2.0f * tp[p];
            float cc[{NC}];
            float ulane = -INF; float lmin = INF;
            for (uint b = 0; b < {BL}u; ++b) {{
                float bb = bc[b]; float bestem = INF; uint st = bs[b];
                for (uint n = 0; n < 8u; ++n) {{
                    uint w = st | (n << 13);
                    {em}
                    float c = bb + em;
                    cc[b * 8u + n] = c;
                    bestem = min(bestem, em);
                    lmin = min(lmin, c);
                }}
                if (bb < INF) ulane = max(ulane, bb + bestem);
            }}
            float lo = simd_min(lmin);
            float hi = simd_max(ulane);
            float T = hi; bool found = false;
            for (uint r = 0; r < {rounds}u; ++r) {{
                float mid = 0.5f * (lo + hi);
                uint cnt = 0u;
                for (uint k = 0; k < {NC}u; ++k) cnt += (cc[k] <= mid) ? 1u : 0u;
                cnt = simd_sum(cnt);
                if (cnt > {W}u) {{ hi = mid; }}
                else if (cnt < {W // 2}u) {{ lo = mid; }}
                else {{ T = mid; found = true; break; }}
            }}
            if (!found) T = hi;
            uint n_mine = 0u;
            for (uint k = 0; k < {NC}u; ++k) n_mine += (cc[k] <= T) ? 1u : 0u;
            uint off = simd_prefix_exclusive_sum(n_mine);
            uint total = simd_sum(n_mine);
            uint k2 = off;
            for (uint k = 0; k < {NC}u; ++k) {{
                if (cc[k] <= T && k2 < {W}u) {{
                    uint b = k >> 3; uint n = k & 7u;
                    uint w = bs[b] | (n << 13);
                    sc_cost[k2] = cc[k]; sc_state[k2] = ushort(w >> 3);
                    sc_par[k2] = uchar(lane * {BL}u + b); sc_sym[k2] = uchar(n);
                    ++k2;
                }}
            }}
            simdgroup_barrier(mem_flags::mem_threadgroup);
            uint nb = min(total, {W}u);
            ntrunc += (total > {W}u) ? 1u : 0u;
            nshort += (total < {W // 2}u) ? 1u : 0u;
            for (uint i = 0; i < {BL}u; ++i) {{
                uint slot = lane * {BL}u + i;
                if (slot < nb) {{
                    bc[i] = sc_cost[slot]; bs[i] = uint(sc_state[slot]);
                    par_out[p * {W}u + slot] = sc_par[slot]; sym_out[p * {W}u + slot] = sc_sym[slot];
                }} else {{ bc[i] = INF; bs[i] = 0u; }}
            }}
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float mymin = INF; uint myslot = 0xFFFFu;
        for (uint i = 0; i < {BL}u; ++i) {{ if (bc[i] < mymin) {{ mymin = bc[i]; myslot = lane * {BL}u + i; }} }}
        float gmin = simd_min(mymin);
        uint cand = (mymin == gmin) ? myslot : 0xFFFFu;
        uint bestslot = simd_min(cand);
        if (lane == 0u) {{
            best[tile] = ushort(bestslot);
            stats[tile * 3u + 0u] = ntrunc; stats[tile * 3u + 1u] = nshort; stats[tile * 3u + 2u] = as_type<uint>(gmin);
        }}
    """


def _trace_source(W: int) -> str:
    return f"""
        uint tile = thread_position_in_grid.x;
        if (tile >= ntiles[0]) return;
        uint slot = uint(best[tile]);
        const device uchar* par = parent + ulong(tile) * {256 * W}ul;
        const device uchar* sy = symbol + ulong(tile) * {256 * W}ul;
        device uchar* o = new3 + ulong(tile) * 256ul;
        for (int p = 255; p >= 0; --p) {{
            o[p] = sy[uint(p) * {W}u + slot];
            slot = uint(par[uint(p) * {W}u + slot]);
        }}
    """


def _seam_source() -> str:
    em = _EM.format(MAGIC=MAGIC)
    return f"""
        uint tile = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        device uchar* n3 = new3 + ulong(tile) * 256ul;
        const device float* tp = targets + ulong(tile) * 256ul;
        uint s_a = (uint(n3[255]) << 10) | (uint(n3[254]) << 7) | (uint(n3[253]) << 4) | (uint(n3[252]) << 1) | (uint(n3[251]) >> 2);
        uint o3 = uint(n3[3]); uint n4 = uint(n3[4]); uint n5 = uint(n3[5]); uint n6 = uint(n3[6]); uint n7 = uint(n3[7]);
        float t2s[8];
        for (uint p = 0; p < 8u; ++p) t2s[p] = 2.0f * tp[p];
        float best = {INF_STR}; uint bestc = 0u;
        for (uint j = 0; j < 64u; ++j) {{
            uint c = lane * 64u + j;
            uint ns[8];
            ns[0] = c & 7u; ns[1] = (c >> 3) & 7u; ns[2] = (c >> 6) & 7u; ns[3] = ((c >> 9) & 3u) | (o3 & 4u);
            ns[4] = n4; ns[5] = n5; ns[6] = n6; ns[7] = n7;
            uint s = s_a; float cost = 0.0f;
            for (uint p = 0; p < 8u; ++p) {{
                uint w = s | (ns[p] << 13);
                float t2 = t2s[p];
                {em}
                cost += em;
                s = (s >> 3) | (ns[p] << 10);
            }}
            if (cost < best) {{ best = cost; bestc = c; }}
        }}
        float g = simd_min(best);
        uint cand = (best == g) ? bestc : 0xFFFFFFFFu;
        uint bc = simd_min(cand);
        if (lane == 0u) {{
            n3[0] = uchar(bc & 7u); n3[1] = uchar((bc >> 3) & 7u); n3[2] = uchar((bc >> 6) & 7u);
            n3[3] = uchar(((bc >> 9) & 3u) | (o3 & 4u));
        }}
    """


def _pack_source() -> str:
    return f"""
        uint tile = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        const device uchar* n3 = new3 + ulong(tile) * 256ul;
        device short* o = words + ulong(tile) * {NW}ul;
        for (uint i = lane; i < {NW}u; i += 32u) {{
            uint wv = 0u;
            for (uint j = 0; j < 16u; ++j) {{
                uint b = i * 16u + j;
                uint bit = (uint(n3[sym_of_bit[b]]) >> uint(k_of_bit[b])) & 1u;
                wv |= bit << j;
            }}
            o[i] = short(ushort(wv));
        }}
    """


def kernels(W: int = 256, rounds: int = 10) -> dict:
    key = (W, rounds)
    if key in _KERNELS:
        return _KERNELS[key]
    if W % 32 or W < 32 or W > 256:
        raise ValueError("beam width must be a multiple of 32 in [32, 256] (8 bits of slot index)")
    k = {
        "beam": mx.fast.metal_kernel(name=f"dsv41_tcq_beam_{W}_{rounds}", input_names=["targets"],
                                     output_names=["parent", "symbol", "best", "stats"], source=_beam_source(W, rounds)),
        "trace": mx.fast.metal_kernel(name=f"dsv41_tcq_trace_{W}", input_names=["parent", "symbol", "best", "ntiles"],
                                      output_names=["new3"], source=_trace_source(W)),
        "pack": mx.fast.metal_kernel(name="dsv41_tcq_pack", input_names=["new3", "sym_of_bit", "k_of_bit"],
                                     output_names=["words"], source=_pack_source()),
    }
    _KERNELS[key] = k
    return k


# ------------------------------------------------------------------ drivers

def beam_new3(targets: mx.array, W: int = 256, rounds: int = 10) -> tuple:
    """targets [T,256] f32 (cycle order) -> open-chain new3 [T,256] uint8 (before the seam repair), stats [T,3]."""
    k = kernels(W, rounds)
    T = int(targets.shape[0])
    parent, symbol, best, stats = k["beam"](
        inputs=[targets], output_shapes=[(T, 256, W), (T, 256, W), (T,), (T, 3)],
        output_dtypes=[mx.uint8, mx.uint8, mx.uint16, mx.uint32],
        grid=(32 * T, 1, 1), threadgroup=(32, 1, 1))
    tg = 256
    (new3,) = k["trace"](inputs=[parent, symbol, best, mx.array([T], dtype=mx.uint32)],
                         output_shapes=[(T, 256)], output_dtypes=[mx.uint8],
                         grid=(((T + tg - 1) // tg) * tg, 1, 1), threadgroup=(tg, 1, 1))
    return new3, stats


def seam_repair(new3: mx.array, targets: mx.array) -> mx.array:
    """Tail-biting repair of symbols 0..7 (in place semantics: returns the repaired copy)."""
    k = kernels()
    T = int(targets.shape[0])
    # the seam kernel edits new3 in place: hand it a fresh copy as the OUTPUT buffer initialised from the input
    src = _seam_source()
    kern = _KERNELS.setdefault(("seam_io",), mx.fast.metal_kernel(
        name="dsv41_tcq_seam_io", input_names=["new3_in", "targets"], output_names=["new3"],
        source="""
        {
            uint tile0 = threadgroup_position_in_grid.x; uint lane0 = thread_index_in_simdgroup;
            for (uint i = lane0; i < 256u; i += 32u) new3[ulong(tile0) * 256ul + i] = new3_in[ulong(tile0) * 256ul + i];
            simdgroup_barrier(mem_flags::mem_device);
        }
        """ + src))
    (out,) = kern(inputs=[new3, targets], output_shapes=[(T, 256)], output_dtypes=[mx.uint8],
                  grid=(32 * T, 1, 1), threadgroup=(32, 1, 1))
    return out


def pack_words(new3: mx.array) -> mx.array:
    """new3 [T,256] uint8 -> words [T,48] int16 (== tcq_encode.pack_new3)."""
    k = kernels()
    tb = tables()
    T = int(new3.shape[0])
    (words,) = k["pack"](inputs=[new3, tb["sym_of_bit"], tb["k_of_bit"]], output_shapes=[(T, NW)],
                         output_dtypes=[mx.int16], grid=(32 * T, 1, 1), threadgroup=(32, 1, 1))
    return words


def encode_tiles(targets: mx.array, W: int = 256, rounds: int = 10, batch: int = 4096) -> tuple:
    """targets [N,256] f32 -> (words [N,48] int16, new3 [N,256] uint8, stats [N,3] uint32); batched launches."""
    N = int(targets.shape[0])
    words_l, new3_l, stats_l = [], [], []
    for b0 in range(0, N, batch):
        tb = targets[b0:b0 + batch]
        n3, st = beam_new3(tb, W, rounds)
        n3 = seam_repair(n3, tb)
        wd = pack_words(n3)
        mx.eval(wd, n3, st)
        words_l.append(wd); new3_l.append(n3); stats_l.append(st)
    return mx.concatenate(words_l), mx.concatenate(new3_l), mx.concatenate(stats_l)


def _had128() -> mx.array:
    return mx.array(tq._had128())


def t128_last(x: mx.array) -> mx.array:
    lead = x.shape[:-1]
    IC = x.shape[-1]
    return (x.reshape(*lead, IC // 128, 128) @ _had128()).reshape(*lead, IC)


def prepare_targets(W_esch: mx.array, alpha: float = 1.0) -> tuple:
    """W_esch [in,out] f32 -> (targets [nI*nJ,256] f32 in cycle order, rout16 [out] f16, nI, nJ).

    Mirrors tcq_encode.compute_scales / target_what / matrix_to_cycle_targets with rout rounded to fp16 first.
    """
    tb = tables()
    n_in, n_out = int(W_esch.shape[0]), int(W_esch.shape[1])
    HW = t128_last(W_esch.T).T                                  # B_in @ W  [in,out]
    col_rms = mx.sqrt(mx.mean(HW * HW, axis=0))                 # [out]
    rout = mx.maximum(col_rms / max(tb["dec_rms"], 1e-12) * alpha, 1e-8)
    rout16 = rout.astype(mx.float16)
    Wsc = W_esch / rout16.astype(mx.float32)[None, :]
    W_hat = t128_last(t128_last(Wsc.T).T)                       # B_in Wsc B_out
    nI, nJ = n_in // 16, n_out // 16
    t = W_hat.reshape(nI, 16, nJ, 16).transpose(0, 2, 1, 3).reshape(nI * nJ, 256)
    targets = mx.take(t, tb["cycle_order"].astype(mx.int32), axis=1)
    return targets, rout16, nI, nJ


def encode_projection_gpu(W_esch: mx.array, W: int = 256, rounds: int = 10, batch: int = 4096) -> dict:
    """W_esch [in,out] f32 (eschamoe orientation) -> code [nI,nJ,48] int16, rout f16 [out], new3, stats, timings."""
    t0 = time.perf_counter()
    targets, rout16, nI, nJ = prepare_targets(W_esch)
    mx.eval(targets, rout16)
    t1 = time.perf_counter()
    words, new3, stats = encode_tiles(targets, W, rounds, batch)
    t2 = time.perf_counter()
    return {"code": words.reshape(nI, nJ, NW), "rout": rout16, "new3": new3, "stats": stats, "targets": targets,
            "nI": nI, "nJ": nJ, "prep_s": t1 - t0, "encode_s": t2 - t1}


def decode_effective(code: mx.array, rout16: mx.array, decoder) -> mx.array:
    """code [nI,nJ,48] -> effective weight [in,out] f32 = B_in W_recon B_out diag(rout) (rin = 1)."""
    W_recon = decoder(code, K_BITS).astype(mx.float32)          # [in,out]
    E = t128_last(t128_last(W_recon.T).T)
    return E * rout16.astype(mx.float32)[None, :]
