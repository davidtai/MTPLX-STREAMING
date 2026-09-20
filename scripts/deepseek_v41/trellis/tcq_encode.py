"""Trellis-coded quantizer (TCQ) that re-encodes fp32 weights into the EschaLabs ``eschamoe`` K=3
(3 bits/weight) format whose bit-exact decoder lives in ``mtplx/eschamoe.py``  (DSV4.1 F34, CPU).

Format recap (K=3, one 16x16 tile = 256 weights = 48 int16 = 768 bits):
  * codebook: ``DEC[w] = f16((w*M & 0x8fff)^0x3b60) + f16(((w*M>>16)&0x8fff)^0x3b60)`` for a 16-bit
    window ``w`` (M=3417055213).  Context-dependent -> a trellis code, not a lookup table.
  * The decoder's own gather table (``eschamoe_gather.npz``: ``word_of_K3``/``bit_of_K3`` [256,16])
    maps each output slot (row*16+col in the tile) to the 16 tile bits of its window.  DERIVED from
    that table (see :func:`cycle_tables`), the 256 windows form a single CIRCULAR chain: in cycle
    order consecutive 16-bit windows overlap by 13 bits (step 3), wrapping 255->0.  Equivalently a
    tail-biting convolutional code: state = the 13 shared bits, 8 branches (3 new bits) per symbol.
    Each symbol's 3 NEW bits (window places 13,14,15) cover all 768 tile bits exactly once, so the
    bit->tile packing is conflict-free and fully determined by the decoder (no guessing).

Encoding a tile = choose 3 bits/symbol along the cycle to minimize sum (DEC[window]-target)^2:
  * :func:`viterbi_encode`  — exact per-tile trellis (8192 states, 8 branches), free initial state
    (open chain) plus optional wrap passes to relax the single ring seam; vectorized over a batch of
    tiles, uint8 backpointers, RSS-bounded by the batch size.
  * :func:`beam_encode`     — top-B beam variant (cheaper), for the full-shape serialization.

Forward-chain scale/target derivation is in :func:`compute_scales` / :func:`target_what` /
:func:`effective_weight`; see the report for the algebra.

CPU-ONLY (numpy).  ``mtplx.eschamoe`` (the vendor decoder) is imported lazily by the verifier only.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

_MAGIC = np.uint32(3417055213)
_MASK = np.uint16(0x8FFF)
_XOR = np.uint16(0x3B60)
GATHER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "mtplx", "eschamoe_gather.npz",
)

# ------------------------------------------------------------------ codebook

def build_dec_table() -> np.ndarray:
    """The 65536-entry codebook as fp32 (values are fp16; identical to mtplx.eschamoe._build_dec_table)."""
    w = np.arange(65536, dtype=np.uint32)
    r = (w * _MAGIC) & np.uint32(0xFFFFFFFF)
    lo = ((r & 0xFFFF).astype(np.uint16) & _MASK) ^ _XOR
    hi = (((r >> 16) & 0xFFFF).astype(np.uint16) & _MASK) ^ _XOR
    val = lo.view(np.float16).astype(np.float32) + hi.view(np.float16).astype(np.float32)
    return val.astype(np.float16).astype(np.float32)


def codebook_rms(dec: np.ndarray | None = None) -> float:
    dec = build_dec_table() if dec is None else dec
    return float(np.sqrt(np.mean(dec.astype(np.float64) ** 2)))


# ------------------------------------------------------------------ cycle / trellis tables

_CYCLE_CACHE: dict = {}


def cycle_tables(K: int = 3, gather_path: str = GATHER_PATH) -> dict:
    """Derive the circular-trellis tables for K from the decoder's own gather table.

    Returns dict with:
      cycle_order [256]  slot index (=row*16+col) visited at each cycle position p
      G_cycle    [256,16] tile-bit index (0..16K*16-1) of window place-values 0..15, in cycle order
      newbit_pos [256,3]  tile-bit indices of each symbol's 3 NEW bits (window places 13,14,15)
    Asserts the clean circular shift-by-K chain and that new bits tile the buffer exactly once.
    """
    if K in _CYCLE_CACHE:
        return _CYCLE_CACHE[K]
    if K != 3:
        raise NotImplementedError("F34 targets K=3 (down-style 3-bit tiles)")
    z = np.load(gather_path)
    wo = z[f"word_of_K{K}"].astype(np.int64)  # [256,16]
    bo = z[f"bit_of_K{K}"].astype(np.int64)
    g = wo * 16 + bo                          # [256,16] tile-bit index per slot, place values 0..15
    # successor s -> s' where high-13 of window s (places 3..15) == low-13 of window s' (places 0..12)
    key_hi = {tuple(g[s, 3:16].tolist()): s for s in range(256)}
    succ = -np.ones(256, np.int64)
    for sp in range(256):
        k = tuple(g[sp, 0:13].tolist())
        if k in key_hi:
            succ[key_hi[k]] = sp
    assert (succ >= 0).all() and len(set(succ.tolist())) == 256, "successor is not a permutation"
    order = [0]
    cur = 0
    for _ in range(255):
        cur = int(succ[cur])
        order.append(cur)
    assert len(set(order)) == 256 and int(succ[order[-1]]) == 0, "not a single 256-cycle"
    order = np.array(order, np.int64)
    G = g[order]  # [256,16] in cycle order
    for p in range(256):
        assert np.array_equal(G[p, 3:16], G[(p + 1) % 256, 0:13]), f"chain breaks at p={p}"
    newbit = G[:, 13:16].astype(np.int64)  # [256,3]
    flat = newbit.reshape(-1)
    assert len(set(flat.tolist())) == 16 * K * 16 and flat.min() == 0, "new bits do not tile buffer"
    out = {"cycle_order": order, "G_cycle": G, "newbit_pos": newbit, "K": K, "nwords": 16 * K}
    _CYCLE_CACHE[K] = out
    return out


# ------------------------------------------------------------------ Hadamard / forward chain

_HAD128 = None


def _had128() -> np.ndarray:
    global _HAD128
    if _HAD128 is None:
        h = np.array([[1.0]], np.float32)
        while h.shape[0] < 128:
            h = np.block([[h, h], [h, -h]])
        _HAD128 = (h / math.sqrt(128.0)).astype(np.float32)
    return _HAD128


def t128_last(x: np.ndarray) -> np.ndarray:
    """Normalized 128-block Walsh-Hadamard over the LAST axis (== right-multiply by B, B symmetric)."""
    x = x.astype(np.float32)
    lead = x.shape[:-1]
    IC = x.shape[-1]
    assert IC % 128 == 0, IC
    return (x.reshape(*lead, IC // 128, 128) @ _had128()).reshape(*lead, IC)


def t128_axis0(M: np.ndarray) -> np.ndarray:
    """B_in @ M  (Hadamard on axis 0)."""
    return t128_last(M.T).T


def t128_axis1(M: np.ndarray) -> np.ndarray:
    """M @ B_out  (Hadamard on axis 1)."""
    return t128_last(M)


def compute_scales(W_esch: np.ndarray, dec_rms: float, alpha: float = 1.0) -> tuple:
    """rin = ones; rout[j] = RMS_i (B_in @ W_esch)[i,j] / dec_rms * alpha.  W_esch is [in, out]."""
    n_in, n_out = W_esch.shape
    rin = np.ones(n_in, np.float32)
    HW = t128_axis0(W_esch)                                   # B_in @ W_esch  [in,out]
    col_rms = np.sqrt(np.mean(HW.astype(np.float64) ** 2, axis=0))  # [out]
    rout = (col_rms / max(dec_rms, 1e-12) * alpha).astype(np.float32)
    rout = np.where(rout < 1e-8, 1e-8, rout).astype(np.float32)
    return rin, rout


def target_what(W_esch: np.ndarray, rin: np.ndarray, rout: np.ndarray) -> np.ndarray:
    """W_hat = B_in @ (diag(1/rin) W_esch diag(1/rout)) @ B_out  (the matrix the codebook quantizes)."""
    Wsc = (1.0 / rin)[:, None] * W_esch * (1.0 / rout)[None, :]
    return t128_axis1(t128_axis0(Wsc)).astype(np.float32)


def effective_weight(W_hat_q: np.ndarray, rin: np.ndarray, rout: np.ndarray) -> np.ndarray:
    """E = diag(rin) B_in W_hat_q B_out diag(rout)  (reconstructed weight in the ORIGINAL space)."""
    E = t128_axis1(t128_axis0(W_hat_q.astype(np.float32)))
    return (rin[:, None] * E * rout[None, :]).astype(np.float32)


# ------------------------------------------------------------------ tile <-> cycle target extraction

def matrix_to_cycle_targets(W_hat: np.ndarray, ct: dict) -> tuple:
    """W_hat [in,out] -> targets [ntiles,256] in cycle order, and (nI,nJ)."""
    n_in, n_out = W_hat.shape
    assert n_in % 16 == 0 and n_out % 16 == 0
    nI, nJ = n_in // 16, n_out // 16
    # [in,out] -> [nI,16,nJ,16] -> [nI,nJ,16,16] -> [ntiles, 256] (slot = row*16+col)
    t = W_hat.reshape(nI, 16, nJ, 16).transpose(0, 2, 1, 3).reshape(nI * nJ, 256)
    return t[:, ct["cycle_order"]].astype(np.float32), (nI, nJ)


# ------------------------------------------------------------------ Viterbi (exact trellis)

def _trellis_consts(dec: np.ndarray):
    w = np.arange(65536, dtype=np.int64)
    low13 = (w & 0x1FFF).astype(np.int64)          # predecessor state of window w
    dec2 = (dec.astype(np.float64) ** 2).astype(np.float32)
    return low13, dec.astype(np.float32), dec2


_FAST_CACHE: dict = {}


def _trellis_consts_fast(dec: np.ndarray):
    """Contiguous per-state branch tables for the vectorized beam (:func:`beam_encode_fast`).

    ``DECg2[s, r] = DEC[s + r*8192]`` and ``DEC2g2[s, r] = DEC[s + r*8192]**2`` as C-contiguous
    ``[8192, 8]``.  A state's 8 branch windows are then ONE contiguous row (32 B) instead of eight
    values 32 KiB apart, so the per-step gather is a cache-friendly row gather.  Values are byte-
    identical to ``DEC[state + r*8192]`` used by :func:`beam_encode`, so the fast beam is bit-exact.
    """
    key = id(dec)
    if key not in _FAST_CACHE:
        DEC = dec.astype(np.float32)
        DEC2 = (DEC.astype(np.float64) ** 2).astype(np.float32)
        DECg2 = np.ascontiguousarray(DEC.reshape(8, 8192).T)     # [8192,8], row s = branches r=0..7
        DEC2g2 = np.ascontiguousarray(DEC2.reshape(8, 8192).T)
        _FAST_CACHE[key] = (DEC, DEC2, DECg2, DEC2g2)
    return _FAST_CACHE[key]


def _forward_pass(targets: np.ndarray, cost: np.ndarray, bp: np.ndarray,
                  low13: np.ndarray, DEC: np.ndarray, DEC2: np.ndarray) -> np.ndarray:
    """One forward Viterbi sweep; writes branch backpointers into ``bp`` [S,8192,B], returns cost.

    Emission for window w = DEC2[w] - 2 t DEC[w] (the +t^2 offset is common to all paths at a step
    and dropped).  cost[:,low13] equals cost tiled 8x over the top-3 bits, done as a view-add.
    """
    B, S = targets.shape
    for p in range(S):
        nc = DEC2[None, :] - (2.0 * targets[:, p])[:, None] * DEC[None, :]   # [B,65536] emission
        nc.reshape(B, 8, 8192)[:] += cost[:, None, :]                        # + predecessor cost (tiled)
        nc = nc.reshape(B, 8192, 8)                                          # group by next-state (w=8s'+r)
        r = nc.argmin(axis=2).astype(np.uint8)                              # [B,8192] branch (=w&7)
        cost = np.take_along_axis(nc, r[:, :, None].astype(np.int64), axis=2)[:, :, 0]
        bp[p] = r.T
    return cost


def _viterbi_batch(targets: np.ndarray, dec: np.ndarray, s_star: np.ndarray | None = None) -> tuple:
    """Circular-trellis Viterbi over a batch of streams of arbitrary length.  Returns (new3, boundary).

    ``s_star`` None -> free initial state, free-end backtrack (open-chain optimum), boundary = None.
    ``s_star`` given -> forced start AND end at s_star (tail-biting): the packed tile then decodes
    EXACTLY to the chosen windows; optimal among paths through boundary state s_star.
    """
    low13, DEC, DEC2 = _trellis_consts(dec)
    B, S = targets.shape
    bp = np.empty((S, 8192, B), np.uint8)
    rows = np.arange(B)
    if s_star is None:
        cost = _forward_pass(targets, np.zeros((B, 8192), np.float32), bp, low13, DEC, DEC2)
        state = cost.argmin(axis=1).astype(np.int64)                    # free end
    else:
        s_star = np.asarray(s_star, np.int64)
        init = np.full((B, 8192), 1e30, np.float32)
        init[rows, s_star] = 0.0
        _forward_pass(targets, init, bp, low13, DEC, DEC2)              # forced start = s_star
        state = s_star.copy()                                          # forced end = s_star
    out = np.empty((B, S), np.uint8)
    for p in range(S - 1, -1, -1):
        w = (state << 3) | bp[p][state, rows].astype(np.int64)
        out[:, p] = (w >> 13).astype(np.uint8)
        state = w & 0x1FFF
    return out, s_star


def viterbi_encode(targets: np.ndarray, dec: np.ndarray, batch: int = 128,
                   repair_span: int = 8, on_batch=None) -> tuple:
    """Open-chain exact Viterbi + seam repair (the exact, tail-biting-consistent encoder).

    targets [N,256] (cycle order) -> (new3 [N,256] uint8, s0 [N]).  Vectorized over ``batch`` tiles
    (uint8 backpointers) so RSS stays bounded.  ``on_batch`` (if given) is called before each batch
    (e.g. a GPU-window gate poll).
    """
    N, S = targets.shape
    assert S == 256
    ct = cycle_tables(3)
    out = np.empty((N, 256), np.uint8)
    for b0 in range(0, N, batch):
        if on_batch is not None:
            on_batch()
        tb = targets[b0:b0 + batch]
        new3, _ = _viterbi_batch(tb, dec)
        out[b0:b0 + batch] = repair_seam(new3, tb, dec, ct, span=repair_span)
    sstar = decoded_windows(out, ct)[:, 0] & 0x1FFF
    return out, sstar


# ------------------------------------------------------------------ beam search (cheaper) + seam repair

def _viterbi_fixed_ends(targets: np.ndarray, dec: np.ndarray, s_start: np.ndarray,
                        s_end: np.ndarray) -> np.ndarray:
    """Optimal path of length R from state ``s_start`` to state ``s_end`` (both [B]).  new3 [B,R]."""
    low13, DEC, DEC2 = _trellis_consts(dec)
    B, R = targets.shape
    bp = np.empty((R, 8192, B), np.uint8)
    init = np.full((B, 8192), 1e30, np.float32)
    init[np.arange(B), np.asarray(s_start, np.int64)] = 0.0
    _forward_pass(targets, init, bp, low13, DEC, DEC2)
    out = np.empty((B, R), np.uint8)
    state = np.asarray(s_end, np.int64).copy()
    rows = np.arange(B)
    for p in range(R - 1, -1, -1):
        w = (state << 3) | bp[p][state, rows].astype(np.int64)
        out[:, p] = (w >> 13).astype(np.uint8)
        state = w & 0x1FFF
    return out


def repair_seam(new3: np.ndarray, targets: np.ndarray, dec: np.ndarray, ct: dict,
                span: int = 8) -> np.ndarray:
    """Make an open-chain solution tail-biting-consistent by re-optimizing the first ``span`` symbols.

    The ring seam corrupts the ~5 symbols whose window reads wrapped bits.  Re-solve symbols
    [0,span) with the start pinned to the ACTUAL wrap state and the end pinned to the (unchanged)
    boundary state at ``span`` — so symbols >= span and the packed wrap bits are untouched, and the
    packed decode then equals the intended windows.
    """
    act = decoded_windows(new3, ct)
    s_a = act[:, 0] & 0x1FFF                              # actual wrap state (low-13 of window_0)
    s_span = act[:, span] & 0x1FFF                        # boundary state at symbol `span` (fixed)
    fixed = _viterbi_fixed_ends(targets[:, :span], dec, s_a, s_span)
    new3 = new3.copy()
    new3[:, :span] = fixed
    return new3


def beam_encode(targets: np.ndarray, dec: np.ndarray, beam: int = 256,
                batch: int = 512, repair_span: int = 8, on_batch=None) -> tuple:
    """Open-chain top-``beam`` beam search + seam repair (the cheap, tail-biting-consistent encoder).

    targets [N,256] (cycle order) -> (new3 [N,256] uint8, s0 [N]) where s0 is the closed boundary
    state (== low-13 of window_0).  Vectorized over ``batch`` tiles.  ``on_batch`` (if given) is
    called before each batch (e.g. a GPU-window gate poll).
    """
    low13, DEC, DEC2 = _trellis_consts(dec)
    N, S = targets.shape
    assert S == 256
    ct = cycle_tables(3)
    out = np.empty((N, 256), np.uint8)
    sstar = np.empty(N, np.int64)
    top = (np.arange(8, dtype=np.int64) << 13)
    arange8 = np.arange(8, dtype=np.uint8)
    for b0 in range(0, N, batch):
        if on_batch is not None:
            on_batch()
        tb = targets[b0:b0 + batch]
        B = tb.shape[0]
        rows = np.arange(B)
        # step 0: full trellis (free init), then keep top-`beam` next-states
        nc = np.zeros((B, 8192), np.float32)[:, low13] + (DEC2[None, :] - 2.0 * tb[:, 0][:, None] * DEC[None, :])
        nc = nc.reshape(B, 8192, 8)
        r0 = nc.argmin(axis=2)
        c0 = np.take_along_axis(nc, r0[:, :, None], axis=2)[:, :, 0]
        sel = np.argpartition(c0, beam - 1, axis=1)[:, :beam]
        beam_states = sel.astype(np.int64)
        beam_cost = np.take_along_axis(c0, sel, axis=1).astype(np.float32)
        parent = np.empty((S, B, beam), np.int32)
        new3rec = np.empty((S, B, beam), np.uint8)
        parent[0] = -1
        new3rec[0] = ((beam_states << 3 | np.take_along_axis(r0, sel, axis=1)) >> 13).astype(np.uint8)
        flat_new3 = np.broadcast_to(arange8, (B, beam, 8)).reshape(B, beam * 8)
        flat_parent = np.broadcast_to(np.arange(beam, dtype=np.int32)[:, None], (beam, 8)).reshape(beam * 8)
        for p in range(1, S):
            w_cand = beam_states[:, :, None] + top[None, None, :]
            total = beam_cost[:, :, None] + (DEC2[w_cand] - 2.0 * tb[:, p][:, None, None] * DEC[w_cand])
            nxt = w_cand >> 3
            flat_cost = total.reshape(B, beam * 8)
            keep = np.argpartition(flat_cost, beam - 1, axis=1)[:, :beam]
            beam_cost = np.take_along_axis(flat_cost, keep, axis=1).astype(np.float32)
            beam_states = np.take_along_axis(nxt.reshape(B, beam * 8), keep, axis=1)
            new3rec[p] = np.take_along_axis(flat_new3, keep, axis=1)
            parent[p] = flat_parent[keep]
        slot = beam_cost.argmin(axis=1)
        ob = np.empty((B, S), np.uint8)
        for p in range(S - 1, -1, -1):
            ob[:, p] = new3rec[p][rows, slot]
            slot = parent[p][rows, slot]
        ob = repair_seam(ob, tb, dec, ct, span=repair_span)   # per-batch: bounds RSS
        out[b0:b0 + B] = ob
        sstar[b0:b0 + B] = decoded_windows(ob, ct)[:, 0] & 0x1FFF
    return out, sstar


def beam_encode_fast(targets: np.ndarray, dec: np.ndarray, beam: int = 256,
                     batch: int = 1024, repair_span: int = 8, on_batch=None) -> tuple:
    """Bit-exact, faster re-implementation of :func:`beam_encode` (DSV4.1 F37).

    Same open-chain top-``beam`` search + seam repair, producing BYTE-IDENTICAL codes to
    :func:`beam_encode` (asserted in the tests and on the F34 artifact).  The speedups are purely
    mechanical and value-preserving:

      * step 0 drops the ``zeros[:, low13]`` all-zero [B,65536] gather (predecessor cost is 0);
      * per step the 8 branch emissions are read as ONE contiguous [B,beam,8] row gather from the
        ``[8192,8]`` tables (:func:`_trellis_consts_fast`) instead of an element gather of
        ``DEC[state + r*8192]`` scattered 32 KiB apart;
      * the kept symbol and parent slot are ``keep & 7`` / ``keep >> 3`` of the top-k index
        (the flat layout is ``j = slot*8 + r``), removing the ``nxt`` array, the broadcast
        ``flat_new3``/``flat_parent`` tables and two ``take_along_axis`` gathers.

    The emission arithmetic (``beam_cost + (DEC2[w] - 2 t DEC[w])``) and the ``np.argpartition`` call
    are IDENTICAL to :func:`beam_encode`, so ``total`` and therefore ``keep`` are bit-identical.
    ``repair_seam`` is the same call.  targets [N,256] (cycle order) -> (new3 [N,256] uint8, s0 [N]).
    """
    DEC, DEC2, DECg2, DEC2g2 = _trellis_consts_fast(dec)
    N, S = targets.shape
    assert S == 256
    ct = cycle_tables(3)
    out = np.empty((N, 256), np.uint8)
    sstar = np.empty(N, np.int64)
    for b0 in range(0, N, batch):
        if on_batch is not None:
            on_batch()
        tb = targets[b0:b0 + batch]
        B = tb.shape[0]
        rows = np.arange(B)
        # step 0: free init (predecessor cost 0), collapse to best next-state, keep top-`beam`
        nc = (DEC2[None, :] - 2.0 * tb[:, 0][:, None] * DEC[None, :]).reshape(B, 8192, 8)
        r0 = nc.argmin(axis=2)
        c0 = np.take_along_axis(nc, r0[:, :, None], axis=2)[:, :, 0]
        sel = np.argpartition(c0, beam - 1, axis=1)[:, :beam]
        beam_states = sel.astype(np.int32)
        beam_cost = np.take_along_axis(c0, sel, axis=1).astype(np.float32)
        parent = np.empty((S, B, beam), np.int32)
        new3rec = np.empty((S, B, beam), np.uint8)
        parent[0] = -1
        new3rec[0] = ((sel << 3 | np.take_along_axis(r0, sel, axis=1)) >> 13).astype(np.uint8)
        for p in range(1, S):
            # 8 branch emissions per state as a contiguous row gather -> [B,beam,8]
            d2 = DEC2g2[beam_states]
            d1 = DECg2[beam_states]
            total = (beam_cost[:, :, None] + (d2 - (2.0 * tb[:, p])[:, None, None] * d1)).reshape(B, beam * 8)
            keep = np.argpartition(total, beam - 1, axis=1)[:, :beam]
            beam_cost = np.take_along_axis(total, keep, axis=1).astype(np.float32)
            parent_i = keep >> 3                                   # slot: flat j = slot*8 + r
            r = (keep & 7).astype(np.int32)                        # symbol (3 new bits)
            new3rec[p] = r.astype(np.uint8)
            parent[p] = parent_i.astype(np.int32)
            # next state = (old_state >> 3) | (r << 10); old_state = beam_states[parent_i]
            beam_states = (np.take_along_axis(beam_states, parent_i, axis=1) >> 3) | (r << 10)
        slot = beam_cost.argmin(axis=1)
        ob = np.empty((B, S), np.uint8)
        for p in range(S - 1, -1, -1):
            ob[:, p] = new3rec[p][rows, slot]
            slot = parent[p][rows, slot]
        ob = repair_seam(ob, tb, dec, ct, span=repair_span)   # per-batch: bounds RSS
        out[b0:b0 + B] = ob
        sstar[b0:b0 + B] = decoded_windows(ob, ct)[:, 0] & 0x1FFF
    return out, sstar


def simulate_windows(new3: np.ndarray, s0: np.ndarray) -> tuple:
    """Open-chain windows from initial state s0 [N] and new3 [N,256]; returns (win [N,256], end [N]).

    ``end == s0`` iff the code is tail-biting-consistent (then the packed decode equals these windows).
    """
    N, S = new3.shape
    state = np.asarray(s0, np.int64).copy()
    win = np.empty((N, S), np.int64)
    for p in range(S):
        w = state | (new3[:, p].astype(np.int64) << 13)
        win[:, p] = w
        state = w >> 3
    return win, state


# ------------------------------------------------------------------ packing / own decode

def pack_new3(new3: np.ndarray, ct: dict) -> np.ndarray:
    """new3 [N,256] uint8 (cycle order) -> code words [N, 16K] int16.  Conflict-free by construction."""
    N = new3.shape[0]
    K = ct["K"]
    nbits = 16 * K * 16                                    # 768 for K=3
    newbit = ct["newbit_pos"]                              # [256,3] tile-bit indices
    bits = np.zeros((N, nbits), np.uint8)
    for j in range(3):
        bits[:, newbit[:, j]] = (new3 >> j) & 1            # place-value 13+j bit of each symbol
    bits = bits.reshape(N, 16 * K, 16)                     # word, bit-within-word
    words = (bits.astype(np.uint32) << np.arange(16, dtype=np.uint32)[None, None, :]).sum(axis=2)
    return words.astype(np.uint16).astype(np.int16)        # int16 view (u16 wrap for bit 15)


_DECODE_GATHER: dict = {}


def _decode_gather(K: int):
    if K not in _DECODE_GATHER:
        z = np.load(GATHER_PATH)
        _DECODE_GATHER[K] = (z[f"word_of_K{K}"].astype(np.int64), z[f"bit_of_K{K}"].astype(np.int64))
    return _DECODE_GATHER[K]


def decode_numpy(code: np.ndarray, K: int, dec: np.ndarray | None = None,
                 chunk: int = 8192) -> np.ndarray:
    """Own decoder mirroring mtplx.eschamoe.decode_expert_weights.  code int16 [..,nI,nJ,16K] -> W fp16.

    Windows are gathered in tile chunks (``chunk`` tiles) so the [T,256,16] scratch stays bounded.
    """
    dec = build_dec_table() if dec is None else dec
    dec16 = dec.astype(np.float16)
    word_of, bit_of = _decode_gather(K)                   # [256,16]
    wo = word_of.reshape(-1)
    bo = bit_of.reshape(-1).reshape(1, 256, 16).astype(np.uint32)
    pow16 = (np.uint32(1) << np.arange(16, dtype=np.uint32))
    lead = code.shape[:-3]
    nI, nJ, nw = code.shape[-3], code.shape[-2], code.shape[-1]
    assert nw == 16 * K
    T = int(np.prod(lead)) * nI * nJ if lead else nI * nJ
    u16 = (code.astype(np.int64) & 0xFFFF).reshape(T, nw).astype(np.uint32)
    win = np.empty((T, 256), np.int64)
    for c0 in range(0, T, chunk):
        w = u16[c0:c0 + chunk][:, wo].reshape(-1, 256, 16)
        bits = (w >> bo) & np.uint32(1)
        win[c0:c0 + chunk] = (bits * pow16).sum(axis=-1).astype(np.int64)
    Wf = dec16[win].reshape(*lead, nI, nJ, 16, 16)
    L = len(lead)
    perm = (*range(L), L + 0, L + 2, L + 1, L + 3)
    return np.ascontiguousarray(np.transpose(Wf, perm).reshape(*lead, nI * 16, nJ * 16))


def build_expert_code(new3: np.ndarray, nI: int, nJ: int, ct: dict) -> np.ndarray:
    """new3 [nI*nJ,256] (tiles row-major over (ti,tj)) -> code int16 [nI,nJ,16K]."""
    words = pack_new3(new3, ct)                            # [ntiles,16K]
    return words.reshape(nI, nJ, ct["nwords"])


# ------------------------------------------------------------------ full projection encode + serialize

def decoded_windows(new3: np.ndarray, ct: dict) -> np.ndarray:
    """Windows each symbol *actually* decodes to after circular packing.  new3 [N,256] -> win [N,256].

    Packs the new bits then reads them back through the decoder's gather (G_cycle): this is exactly
    what mtplx.eschamoe sees.  For a tail-biting-consistent code this equals :func:`simulate_windows`.
    """
    N = new3.shape[0]
    nbits = ct["nwords"] * 16
    newbit = ct["newbit_pos"]
    bits = np.zeros((N, nbits), np.uint8)
    for j in range(3):
        bits[:, newbit[:, j]] = (new3 >> j) & 1
    G = ct["G_cycle"]                                      # [256,16] tile-bit index per place value
    win = np.zeros((N, 256), np.int64)
    for k in range(16):
        win |= (bits[:, G[:, k]].astype(np.int64) << k)
    return win


def encode_projection(W_esch: np.ndarray, method: str, dec: np.ndarray, ct: dict, *,
                      beam: int = 256, alpha: float = 1.0, row_slice: slice | None = None,
                      on_batch=None) -> dict:
    """Encode one projection (W_esch [in,out]).  Optionally restrict to a row subset for quality.

    All quality metrics are computed on ``W_eff`` (reconstructed from the ACTUAL decode of the packed
    code), so they are honest regardless of any residual seam.  ``consistent`` reports, per tile,
    whether the packed decode equals the encoder's intended windows (tail-biting closed the ring).
    """
    rin, rout = compute_scales(W_esch, codebook_rms(dec), alpha=alpha)
    W_hat = target_what(W_esch, rin, rout)
    if row_slice is not None:
        W_hat, W_esch_used, rin_used = W_hat[row_slice], W_esch[row_slice], rin[row_slice]
    else:
        W_esch_used, rin_used = W_esch, rin
    targets, (nI, nJ) = matrix_to_cycle_targets(W_hat, ct)
    t0 = time.perf_counter()
    if method == "viterbi":
        new3, sstar = viterbi_encode(targets, dec, on_batch=on_batch)
    elif method == "beam":
        new3, sstar = beam_encode(targets, dec, beam=beam, on_batch=on_batch)
    else:
        raise ValueError(method)
    enc_s = time.perf_counter() - t0
    code = build_expert_code(new3, nI, nJ, ct)             # [nI,nJ,16K]
    W_recon = decode_numpy(code, ct["K"], dec).astype(np.float32)   # ACTUAL decode (ground truth)
    W_eff = effective_weight(W_recon, rin_used, rout)
    sim_win, _ = simulate_windows(new3, sstar)
    act_win = decoded_windows(new3, ct)
    consistent = (sim_win == act_win).all(axis=1)          # [ntiles] ring closed per tile
    sse = ((dec[act_win] - targets) ** 2).sum(axis=1)      # [ntiles] actual per-tile SSE
    return {
        "code": code, "rin": rin, "rout": rout, "W_recon": W_recon, "W_eff": W_eff,
        "W_esch_used": W_esch_used, "rin_used": rin_used, "nI": nI, "nJ": nJ,
        "encode_s": enc_s, "n_tiles": nI * nJ, "consistent": consistent,
        "sse_tiles": sse, "sstar": sstar,
    }


def serialize(path: str, experts: list, K: int = 3) -> dict:
    """Write the vendor-field .npz for a set of sample experts of ONE projection kind.

    experts: list of dicts each with keys code [nI,nJ,16K] int16, rin [in] , rout [out], and
    provenance fields.  Writes escha_code/escha_rin/escha_rout/escha_config; returns a summary.
    """
    codes = np.stack([e["code"] for e in experts], axis=0).astype(np.int16)  # [E,nI,nJ,16K]
    rin = np.stack([e["rin"] for e in experts], axis=0).astype(np.float16)
    rout = np.stack([e["rout"] for e in experts], axis=0).astype(np.float16)
    E, nI, nJ, nw = codes.shape
    in_p, out_p = nI * 16, nJ * 16
    config = np.array([16, K, 2, 1, E, in_p, out_p, in_p, out_p], np.int32)
    np.savez(path, escha_code=codes, escha_rin=rin, escha_rout=rout, escha_config=config)
    return {"E": E, "nI": nI, "nJ": nJ, "in_p": in_p, "out_p": out_p,
            "bytes_code": int(codes.nbytes), "bytes_rin": int(rin.nbytes), "bytes_rout": int(rout.nbytes)}
