"""CPU tests for DSV4.1 F37: the vectorized beam encoder and the trellis-ladder effective weight.

Runs in the main .venv (numpy + mlx, no torch).  ``bank_ladder`` imports torch, so the format-table
test parses its source with ``ast`` instead of importing it.  MLX pinned to CPU.
"""
import ast
import os
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

ROOT = Path(__file__).resolve().parents[1]
TRELLIS = ROOT / "scripts" / "deepseek_v41" / "trellis"
sys.path.insert(0, str(TRELLIS))

import tcq_encode as enc          # noqa: E402
import tcq_verify as ver          # noqa: E402
import trellis_ladder as TL       # noqa: E402


# ---------------------------------------------------------------- (1) fast beam == F34 beam

@pytest.mark.parametrize("seed,scale", [(0, 0.3), (1, 1.2), (2, 3.0)])
def test_beam_fast_bit_exact_vs_beam_on_synthetic_256_tiles(seed, scale):
    """The vectorized beam reproduces F34's beam codes BIT-EXACTLY on a synthetic 256-tile problem."""
    dec = enc.build_dec_table()
    rng = np.random.default_rng(seed)
    targets = (rng.standard_normal((256, 256)).astype(np.float32) * scale)
    o_ref, s_ref = enc.beam_encode(targets, dec, beam=256)
    o_fast, s_fast = enc.beam_encode_fast(targets, dec, beam=256)
    assert np.array_equal(o_ref, o_fast), f"{(o_ref != o_fast).sum()} code mismatches"
    assert np.array_equal(s_ref, s_fast)


def test_beam_fast_bit_exact_across_batch_sizes():
    """Batch size must not change the codes (per-row argpartition is batch-independent)."""
    dec = enc.build_dec_table()
    rng = np.random.default_rng(7)
    targets = (rng.standard_normal((300, 256)).astype(np.float32) * 1.0)
    a, _ = enc.beam_encode_fast(targets, dec, beam=256, batch=64)
    b, _ = enc.beam_encode_fast(targets, dec, beam=256, batch=512)
    assert np.array_equal(a, b)


def test_c_beam_is_valid_equal_quality_beam():
    """The C beam step (tcq_beam.c via ctypes) is a VALID beam-256 of equal quality to beam_encode_fast.

    It is NOT bit-exact: the eschamoe codebook has only 10746 distinct fp16 values, so accumulated
    float32 costs tie often, and the C deterministic (cost, index) tie rule resolves ties differently
    than numpy's argpartition — a measure-zero-quality difference, not a functional bug.  The check is
    therefore per-tile SSE parity (both find essentially the same-cost beam solution)."""
    import tcq_beam_c as C
    dec = enc.build_dec_table()
    ct = enc.cycle_tables(3)
    rng = np.random.default_rng(11)
    targets = (rng.standard_normal((300, 256)).astype(np.float32) * 1.0)
    of, _ = enc.beam_encode_fast(targets, dec, beam=256)
    oc, sc = C.beam_encode_c(targets, dec, beam=256, batch=128)
    assert oc.shape == of.shape and oc.dtype == np.uint8
    # the C code decodes to a tail-biting-consistent packing (actual decode == intended windows)
    assert (enc.simulate_windows(oc, sc)[0] == enc.decoded_windows(oc, ct)).all()
    sse_f = ((dec[enc.decoded_windows(of, ct)] - targets) ** 2).sum(1).mean()
    sse_c = ((dec[enc.decoded_windows(oc, ct)] - targets) ** 2).sum(1).mean()
    assert abs(sse_c - sse_f) / sse_f < 1e-3, (sse_f, sse_c)


# ---------------------------------------------------------------- (2) effective weight == F34 chain

def test_effective_weight_matches_f34_verifier_chain():
    """TL.effective_weight_hf equals the F34 effective-weight and the eschamoe forward chain.

    Encodes a tiny source weight, decodes the packed code with the VENDOR decoder (bit-exact check),
    and confirms x @ effective_weight reproduces the T128/rin/rout chain to fp noise (F34 test 4)."""
    dec = enc.build_dec_table()
    rng = np.random.default_rng(3)
    # HF source [out, in]; in,out multiples of 128 (T128) and 16 (tiling)
    w_hf = (rng.standard_normal((256, 128)).astype(np.float32) * 0.1)
    rec = TL.encode_source_weight(w_hf)
    # bit-exact round trip vs the vendor decoder
    _, bit_exact, nmis = ver.decode_and_check(rec["code"], 3, dec)
    assert bit_exact, f"{nmis} vendor-vs-own decode mismatches"
    # TL.effective_weight_hf == enc.effective_weight(decode).T  (byte identical)
    W_recon = enc.decode_numpy(rec["code"], 3, dec).astype(np.float32)          # [in,out]
    rin = np.asarray(rec["rin"], np.float32); rout = np.asarray(rec["rout"], np.float32)
    E_esch = enc.effective_weight(W_recon, rin, rout)                           # [in,out]
    E_hf = TL.effective_weight_hf(rec)                                          # [out,in]
    assert np.array_equal(E_hf, np.ascontiguousarray(E_esch.T))
    # forward-chain: eschamoe chain output reproduces x @ effective_weight to fp noise
    W_esch = np.ascontiguousarray(w_hf.T)                                       # [in,out]
    chk = ver.forward_chain_check(W_recon, rin, rout, W_esch, W_eff=E_esch, seed=0)
    assert chk["rel_vs_eff"] < 2e-3, chk
    # and the ladder consumes E_hf as x @ E_hf.T == x @ E_esch (the eschamoe forward)
    x = rng.standard_normal((5, 128)).astype(np.float32)
    assert np.allclose(x @ E_hf.T, x @ E_esch, atol=1e-4, rtol=1e-4)


def test_tcq3_bank_reader_reproduces_effective_weight_on_f34_sample():
    """Tcq3Bank reads a 1-record tcq3 artifact (built from the F34 sample npz) and returns the same
    effective HF weight as effective_weight_hf (vendor decode + T128/rout chain, rin=1)."""
    import json
    import tempfile
    npz_path = "/Users/davidtai/projects/OpenSourceWTF/reports/dsv41-f34-trellis/dsv41_f34_L20E3_down_proj_full_beam.npz"
    if not os.path.exists(npz_path):
        pytest.skip("F34 sample npz not present")
    z = np.load(npz_path)
    code = np.ascontiguousarray(z["escha_code"][0].astype(np.int16))      # [144,320,48]
    rout = np.ascontiguousarray(z["escha_rout"][0].astype(np.float16))    # [5120]
    nI, nJ, _ = code.shape
    out_f = nJ * 16
    d = tempfile.mkdtemp(prefix="tcq3bank_")
    with open(os.path.join(d, "experts.bin"), "wb") as f:
        f.write(code.tobytes()); f.write(rout.tobytes())
    manifest = {"artifact": "test", "quantization": {"mode": "tcq3", "bits": 3, "K": 3, "rin": 1},
                "records": [{"layer": 20, "expert": 3, "logical_bytes": code.nbytes + rout.nbytes,
                             "segments": [
                                 {"component": "down_proj.code", "offset": 0, "length": code.nbytes,
                                  "dtype": "I16", "shape": [nI, nJ, 48]},
                                 {"component": "down_proj.rout", "offset": code.nbytes, "length": rout.nbytes,
                                  "dtype": "F16", "shape": [out_f]}]}]}
    with open(os.path.join(d, "expert-manifest.json"), "w") as f:
        json.dump(manifest, f)
    bank = TL.Tcq3Bank(d)
    got = bank.effective_hf(20, 3, "down_proj")                          # [out,in]
    ref = TL.effective_weight_hf({"code": code, "rin": np.ones(nI * 16, np.float32),
                                  "rout": np.asarray(rout, np.float32)})
    assert got.shape == (out_f, nI * 16)
    assert np.array_equal(got, ref)


def test_effective_weight_cosine_in_trellis_band():
    """A random tiny weight's effective-weight cosine vs source is finite and < 1 (genuinely lossy)."""
    rng = np.random.default_rng(5)
    w_hf = (rng.standard_normal((128, 256)).astype(np.float32) * 0.1)
    rec = TL.encode_source_weight(w_hf)
    cos = TL.expert_cosine(TL.effective_weight_hf(rec), w_hf)
    assert 0.90 < cos < 1.0, cos


# ---------------------------------------------------------------- (3) ladder format table unchanged

def test_bank_ladder_six_stored_formats_unchanged():
    """bank_ladder.FORMATS still holds source + the six stored formats, unchanged (parsed via ast so
    the test needs no torch).  tcq3 is added only at runtime, never baked into FORMATS."""
    src = (ROOT / "scripts" / "deepseek_v41" / "torchref" / "bank_ladder.py").read_text()
    tree = ast.parse(src)
    formats = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "FORMATS" for t in node.targets):
            formats = ast.literal_eval(node.value)
    assert formats == [
        ("source", None, None, None), ("q2_gs64", 2, 64, "affine"), ("q3_gs64", 3, 64, "affine"),
        ("q4_gs64", 4, 64, "affine"), ("q4_gs32", 4, 32, "affine"), ("q6_gs64", 6, 64, "affine"),
        ("mxfp4_gs32", 4, 32, "mxfp4")], formats
    assert not any(f[0] == "tcq3_beam256" for f in formats)
