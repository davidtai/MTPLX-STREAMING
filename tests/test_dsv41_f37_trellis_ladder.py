"""CPU tests for DSV4.1 F37: the vectorized beam encoder and the trellis-ladder effective weight.

Runs in the main .venv (numpy + mlx, no torch).  ``bank_ladder`` imports torch, so the format-table
test parses its source with ``ast`` instead of importing it.  MLX pinned to CPU.
"""
import ast
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
