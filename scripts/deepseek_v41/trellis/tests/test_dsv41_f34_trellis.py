"""CPU tests for the DSV4.1 F34 trellis (TCQ) tools.  MLX pinned to CPU by conftest.py."""
import itertools
import json
import os

import numpy as np
import pytest

import tcq_encode as enc
import tcq_verify as ver

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIM_JSON = "/Users/davidtai/escha-extract/escham_decode_primitive.json"


# ---------------------------------------------------------------- codebook

def test_codebook_matches_reference_primitive():
    dec = enc.build_dec_table()
    assert dec.shape == (65536,)
    # independent recompute of the primitive for a spread of windows
    M = np.uint64(3417055213)
    for w in [0, 1, 2, 7, 255, 4096, 40000, 65535]:
        r = np.uint32((np.uint64(w) * M) & np.uint64(0xFFFFFFFF))
        lo = ((r & 0xFFFF).astype(np.uint16) & np.uint16(0x8FFF)) ^ np.uint16(0x3B60)
        hi = (((r >> 16) & 0xFFFF).astype(np.uint16) & np.uint16(0x8FFF)) ^ np.uint16(0x3B60)
        val = np.float32(lo.view(np.float16)) + np.float32(hi.view(np.float16))
        assert np.float16(val).astype(np.float32) == dec[w]
    with open(PRIM_JSON) as f:
        prim = json.load(f)
    assert np.allclose(dec[:16], np.array(prim["dec_0_15"], np.float32), atol=0, rtol=0)


def test_codebook_matches_vendor():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    vend = ver.vendor_decoder()
    vend_dec = np.array(vend._build_dec_table().astype(mx.float32))
    assert np.array_equal(enc.build_dec_table(), vend_dec)


# ---------------------------------------------------------------- mxfp4 source cross-check

def test_mxfp4_numpy_matches_mlx():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import mxfp4_source as src
    rng = np.random.default_rng(1)
    x = mx.array((rng.standard_normal((7, 256)) * 2.5).astype(np.float32))
    wq, scales = mx.quantize(x, group_size=32, bits=4, mode="mxfp4")
    deq = np.array(mx.dequantize(wq, scales, group_size=32, bits=4, mode="mxfp4").astype(mx.float32))
    mine = src.dequantize_numpy(np.array(wq), np.array(scales))
    assert np.array_equal(mine, deq)


# ---------------------------------------------------------------- cycle tables

def test_cycle_tables_valid():
    ct = enc.cycle_tables(3)  # all structural asserts run inside
    assert ct["cycle_order"].shape == (256,)
    assert sorted(ct["cycle_order"].tolist()) == list(range(256))
    assert ct["newbit_pos"].shape == (256, 3)
    assert len(set(ct["newbit_pos"].reshape(-1).tolist())) == 768


# ---------------------------------------------------------------- pack -> decode round trip

@pytest.mark.parametrize("nI,nJ", [(1, 1), (2, 3)])
def test_pack_decode_roundtrip_bit_exact(nI, nJ):
    ct = enc.cycle_tables(3)
    rng = np.random.default_rng(42)
    new3 = rng.integers(0, 8, size=(nI * nJ, 256)).astype(np.uint8)
    code = enc.build_expert_code(new3, nI, nJ, ct)
    assert code.shape == (nI, nJ, 48) and code.dtype == np.int16
    W_vendor, bit_exact, nmis = ver.decode_and_check(code, 3)
    assert bit_exact, f"{nmis} mismatches vendor vs own decode"
    assert W_vendor.shape == (nI * 16, nJ * 16)


# ---------------------------------------------------------------- Viterbi optimality (toy)

def _path_cost(s0, seq, targets, dec):
    state = int(s0)
    tot = 0.0
    for p, n3 in enumerate(seq):
        w = state | (int(n3) << 13)
        tot += (float(dec[w]) - float(targets[p])) ** 2
        state = w >> 3
    return tot


def _ends_at(s0, seq):
    state = int(s0)
    for n3 in seq:
        state = (state | (int(n3) << 13)) >> 3
    return state


def test_viterbi_core_optimal_open_chain_length4():
    # tests the production forward DP (_forward_pass) + a free-end backtrack vs brute force
    dec = enc.build_dec_table()
    low13, DEC, DEC2 = enc._trellis_consts(dec)
    rng = np.random.default_rng(7)
    for trial in range(6):
        targets = (rng.standard_normal(4).astype(np.float32) * 0.5)
        s0 = int(rng.integers(0, 8192))
        init = np.full((1, 8192), 1e12, np.float32)
        init[0, s0] = 0.0
        bp = np.empty((4, 8192, 1), np.uint8)
        cost = enc._forward_pass(targets[None, :], init.copy(), bp, low13, DEC, DEC2)
        state = int(cost.argmin())                        # free end
        seq = [0] * 4
        for p in range(3, -1, -1):
            w = (state << 3) | int(bp[p][state, 0])
            seq[p] = w >> 13
            state = w & 0x1FFF
        assert state == s0                                # backtracked to fixed start
        vit = _path_cost(s0, seq, targets, dec)
        brute = min(_path_cost(s0, sq, targets, dec) for sq in itertools.product(range(8), repeat=4))
        assert abs(vit - brute) < 1e-5, f"trial {trial}: viterbi {vit} vs brute {brute}"


def test_viterbi_tailbiting_closes_ring_and_optimal():
    # length 5 (>=5 so any boundary state is reachable): ring closes and cost is optimal for s*
    dec = enc.build_dec_table()
    rng = np.random.default_rng(9)
    L = 5
    for trial in range(5):
        targets = (rng.standard_normal(L).astype(np.float32) * 0.5)
        s = int(rng.integers(0, 8192))
        new3, _ = enc._viterbi_batch(targets[None, :], dec, s_star=np.array([s]))
        _, end = enc.simulate_windows(new3, np.array([s]))
        assert int(end[0]) == s, "ring did not close"      # tail-biting consistency
        vit = _path_cost(s, new3[0], targets, dec)
        best = min((_path_cost(s, sq, targets, dec) for sq in itertools.product(range(8), repeat=L)
                    if _ends_at(s, sq) == s), default=None)
        assert best is not None and abs(vit - best) < 1e-5, f"trial {trial}: {vit} vs {best}"


def test_encoders_consistent_and_viterbi_not_worse_on_average():
    dec = enc.build_dec_table()
    ct = enc.cycle_tables(3)
    rng = np.random.default_rng(3)
    targets = (rng.standard_normal((8, 256)).astype(np.float32) * 0.3)
    n3_v, s_v = enc.viterbi_encode(targets, dec)
    n3_b, s_b = enc.beam_encode(targets, dec, beam=64)
    # tail-biting Viterbi and seam-repaired beam both close the ring (actual decode == intended)
    assert (enc.simulate_windows(n3_v, s_v)[0] == enc.decoded_windows(n3_v, ct)).all()
    assert (enc.simulate_windows(n3_b, s_b)[0] == enc.decoded_windows(n3_b, ct)).all()
    sse_v = ((dec[enc.decoded_windows(n3_v, ct)] - targets) ** 2).sum(axis=1)
    sse_b = ((dec[enc.decoded_windows(n3_b, ct)] - targets) ** 2).sum(axis=1)
    assert sse_v.mean() <= sse_b.mean() + 1e-3, (sse_v.mean(), sse_b.mean())


# ---------------------------------------------------------------- forward-chain derivation

def test_effective_weight_reproduces_reference_unquantized():
    rng = np.random.default_rng(11)
    W = (rng.standard_normal((128, 256)).astype(np.float32) * 0.1)
    dec_rms = enc.codebook_rms()
    rin, rout = enc.compute_scales(W, dec_rms)
    W_hat = enc.target_what(W, rin, rout)
    E = enc.effective_weight(W_hat, rin, rout)  # should reproduce W (orthonormal round trip)
    assert np.allclose(E, W, atol=1e-3, rtol=1e-3), np.abs(E - W).max()


def test_chain_reproduces_dequantized_weights():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    dec = enc.build_dec_table()
    ct = enc.cycle_tables(3)
    rng = np.random.default_rng(5)
    W = (rng.standard_normal((128, 256)).astype(np.float32) * 0.1)
    rin, rout = enc.compute_scales(W, enc.codebook_rms(dec))
    W_hat = enc.target_what(W, rin, rout)
    targets, (nI, nJ) = enc.matrix_to_cycle_targets(W_hat, ct)
    new3, _s = enc.beam_encode(targets, dec, beam=128)
    code = enc.build_expert_code(new3, nI, nJ, ct)
    # full verify: bit-exact round trip + chain reproduces x @ effective_weight to fp noise (spec test 4)
    r = ver.verify_expert(code, 3, rin, rout, W, dec, seed=0)
    assert r["bit_exact"] and r["chain"]["rel_vs_eff"] < 2e-3, r
