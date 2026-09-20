"""Verify a serialized eschamoe-K3 sample against the FP4 reference  (DSV4.1 F34, CPU).

Loads the serialized ``.npz``, decodes it with the EXISTING vendor decoder
(``mtplx/eschamoe.py`` :func:`decode_expert_weights`, CPU device), asserts BIT-EXACT equality with
the encoder's own dequantization (:func:`tcq_encode.decode_numpy`), then reports per projection:
cosine / relative-RMS / max|err| of the reconstructed effective weight vs ``W_ref``, and a
forward-chain check (``x @ W_ref`` vs the eschamoe chain on random ``x``).

The repo's ``mtplx`` is editable-installed to the *main* worktree, which has no ``eschamoe.py``;
this module therefore loads the decoder from the current branch's file by path (it is the identical
file that ``mtplx.eschamoe`` names on ``feat/eschamoe-native``).  CPU-ONLY.
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np

import tcq_encode as enc

_WORKTREE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_ESCHAMOE_PY = os.path.join(_WORKTREE_ROOT, "mtplx", "eschamoe.py")
_VENDOR = None


def vendor_decoder():
    """Load mtplx/eschamoe.py from THIS branch as a standalone module (see module docstring)."""
    global _VENDOR
    if _VENDOR is None:
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        spec = importlib.util.spec_from_file_location("eschamoe_vendor", _ESCHAMOE_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _VENDOR = mod
    return _VENDOR


def metrics(A: np.ndarray, B: np.ndarray) -> dict:
    """cosine, relative RMS (||A-B||/||B||), and max abs error, over the flattened arrays."""
    a = A.astype(np.float64).reshape(-1)
    b = B.astype(np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")
    rms = float(np.linalg.norm(a - b) / (nb if nb > 0 else 1.0))
    return {"cosine": cos, "rel_rms": rms, "max_abs": float(np.abs(a - b).max())}


def decode_and_check(code: np.ndarray, K: int, dec: np.ndarray | None = None,
                     chunk_rows: int = 16) -> tuple:
    """Vendor-decode ``code`` [nI,nJ,16K] and assert bit-exact vs our own decode.

    Decodes in tile-row chunks so the full-projection scratch stays bounded.
    Returns (W_vendor fp32 [nI*16,nJ*16], bit_exact: bool, n_mismatch: int).
    """
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    vend = vendor_decoder()
    nI, nJ, _ = code.shape
    W_vendor = np.empty((nI * 16, nJ * 16), np.float32)
    nmis = 0
    for c0 in range(0, nI, chunk_rows):
        cc = np.ascontiguousarray(code[c0:c0 + chunk_rows].astype(np.int16))
        wv = np.array(vend.decode_expert_weights(mx.array(cc), K).astype(mx.float32))
        wm = enc.decode_numpy(cc, K, dec).astype(np.float32)
        nmis += int((wv != wm).sum())
        W_vendor[c0 * 16:(c0 + cc.shape[0]) * 16] = wv
    return W_vendor, nmis == 0, nmis


def forward_chain_check(W_hat_q: np.ndarray, rin: np.ndarray, rout: np.ndarray,
                        W_esch: np.ndarray, W_eff: np.ndarray | None = None,
                        n_x: int = 64, seed: int = 0) -> dict:
    """eschamoe chain vs x@W_ref on random x, using the already-decoded weight (no re-decode).

    ``W_hat_q`` [in,out] fp32 is the vendor-decoded weight.  Reports rel error of the chain output
    vs (a) x @ W_esch (the quantization error) and (b) x @ effective_weight (should be ~fp noise:
    validates the chain algebra end-to-end).
    """
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    vend = vendor_decoder()
    n_in, n_out = W_esch.shape
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_x, n_in)).astype(np.float32)
    xh = vend.t128(mx.array(x), pre=mx.array(rin.astype(np.float32)))
    y = np.array(vend.t128(xh @ mx.array(W_hat_q.astype(np.float16)), post=mx.array(rout.astype(np.float32))))
    ref = x @ W_esch
    if W_eff is None:
        W_eff = enc.effective_weight(W_hat_q, rin, rout)
    ref_eff = x @ W_eff
    return {
        "rel_vs_ref": float(np.linalg.norm(y - ref) / (np.linalg.norm(ref) + 1e-12)),
        "rel_vs_eff": float(np.linalg.norm(y - ref_eff) / (np.linalg.norm(ref_eff) + 1e-12)),
    }


def verify_expert(code: np.ndarray, K: int, rin: np.ndarray, rout: np.ndarray,
                  W_esch: np.ndarray, dec: np.ndarray | None = None, seed: int = 0) -> dict:
    """Full per-projection verification given the code, scales, and fp32 reference (eschamoe orient)."""
    W_vendor, bit_exact, nmis = decode_and_check(code, K, dec)
    if not bit_exact:
        raise AssertionError(f"round-trip NOT bit-exact: {nmis} mismatches (vendor vs own decode)")
    W_eff = enc.effective_weight(W_vendor, rin, rout)
    m = metrics(W_eff, W_esch)
    chain = forward_chain_check(W_vendor, rin, rout, W_esch, W_eff=W_eff, seed=seed)
    return {"bit_exact": bit_exact, "n_mismatch": nmis, **m, "chain": chain}


def load_serialized(npz_path: str) -> dict:
    z = np.load(npz_path)
    cfg = z["escha_config"]
    return {"code": z["escha_code"], "rin": z["escha_rin"], "rout": z["escha_rout"],
            "config": cfg, "K": int(cfg[1]), "E": int(cfg[4]),
            "in_p": int(cfg[7]), "out_p": int(cfg[8])}


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Verify serialized eschamoe-K3 sample vs FP4 reference")
    ap.add_argument("npz")
    ap.add_argument("sidecar")
    args = ap.parse_args()
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import mxfp4_source as src
    dec = enc.build_dec_table()
    ser = load_serialized(args.npz)
    with open(args.sidecar) as f:
        side = json.load(f)
    print(f"loaded {args.npz}: E={ser['E']} K={ser['K']} in_p={ser['in_p']} out_p={ser['out_p']}")
    for i, prov in enumerate(side["experts"]):
        W_ref = src.load_projection(prov["layer"], prov["expert"], prov["component"])
        W_esch = src.eschamoe_orientation(W_ref)
        r = verify_expert(ser["code"][i], ser["K"], np.array(ser["rin"][i], np.float32),
                          np.array(ser["rout"][i], np.float32), W_esch, dec)
        print(f"[{i}] L{prov['layer']}E{prov['expert']} {prov['component']}: "
              f"bit_exact={r['bit_exact']} cos={r['cosine']:.5f} relRMS={r['rel_rms']:.4f} "
              f"max|e|={r['max_abs']:.4f} chain rel_vs_ref={r['chain']['rel_vs_ref']:.4f} "
              f"rel_vs_eff={r['chain']['rel_vs_eff']:.2e}")


if __name__ == "__main__":
    _cli()
