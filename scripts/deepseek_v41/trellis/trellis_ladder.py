"""F37: trellis (eschamoe K=3, beam-256) EFFECTIVE-weight cache for the torch-reference bank ladder.

CPU-only.  Given a source expert weight ``w`` (fp32 ``[out, in]``, HF orientation, exactly what the
ladder's R0 baseline consumes via ``shards.dequant_weight``), this module:

  * encodes it into the eschamoe K=3 trellis (``rin=ones``, ``rout`` from the codebook RMS, beam-256
    via :func:`tcq_encode.beam_encode_fast`) and caches ``code`` + ``rin`` + ``rout`` on disk
    (~3 bits/weight, ~13 MB/expert; the dense effective weight would be ~47 MB/projection);
  * on load, decodes the packed code and applies the T128/rin/rout chain to return the EFFECTIVE
    fp32 weight ``E`` in HF orientation ``[out, in]`` — i.e. ``x @ E.T`` reproduces the eschamoe
    forward ``T128(T128(x)·W)·rout`` that the trellis bank would run.

Bit-exactness of the round-trip decode vs the vendor ``mtplx.eschamoe`` decoder is covered by the
F34/F35 tests; this module reuses ``tcq_encode.decode_numpy`` (asserted identical to the vendor
decoder there).  No GPU: callers must have run ``mx.set_default_device(mx.cpu)``.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

import tcq_encode as enc

W_NAMES = ("w1", "w2", "w3")          # ladder expert weights: w1=gate, w2=down, w3=up (SwiGLU)

_DEC = None
_CT = None


def _tables():
    global _DEC, _CT
    if _DEC is None:
        _DEC = enc.build_dec_table()
        _CT = enc.cycle_tables(3)
    return _DEC, _CT


def source_sha(w_np: np.ndarray) -> str:
    """Content hash of a source weight (fp32 [out,in]); guards the cache against a changed source."""
    return hashlib.sha256(np.ascontiguousarray(w_np, dtype=np.float32).tobytes()).hexdigest()


def encode_source_weight(w_np: np.ndarray, *, on_batch=None, beam: int = 256, batch: int = 512) -> dict:
    """fp32 HF weight ``[out, in]`` -> eschamoe K=3 beam-256 record (code + scales).

    ``on_batch`` is threaded into the beam so the encode PAUSES between tile batches when a GPU
    window is active (see :mod:`encode_worker`).
    """
    dec, ct = _tables()
    W_esch = np.ascontiguousarray(w_np.T)                      # [in, out] eschamoe orientation
    rin, rout = enc.compute_scales(W_esch, enc.codebook_rms(dec))
    W_hat = enc.target_what(W_esch, rin, rout)
    targets, (nI, nJ) = enc.matrix_to_cycle_targets(W_hat, ct)
    new3, _ = enc.beam_encode_fast(targets, dec, beam=beam, batch=batch, on_batch=on_batch)
    code = enc.build_expert_code(new3, nI, nJ, ct)             # int16 [nI,nJ,48]
    return {"code": code, "rin": rin.astype(np.float16), "rout": rout.astype(np.float16),
            "in_p": int(W_esch.shape[0]), "out_p": int(W_esch.shape[1])}


def effective_weight_hf(rec: dict) -> np.ndarray:
    """Cache/encode record -> EFFECTIVE fp32 weight in HF orientation ``[out, in]``.

    Decodes the packed code (== the vendor decoder) then applies ``E = B_in W_recon B_out D_out``
    (rin=ones); returns ``E.T`` so ``x @ (E.T).T = x @ E`` == the eschamoe chain output.
    """
    dec, _ = _tables()
    W_recon = enc.decode_numpy(rec["code"], 3, dec).astype(np.float32)     # [in, out] fp16 values
    E = enc.effective_weight(W_recon, np.asarray(rec["rin"], np.float32),
                             np.asarray(rec["rout"], np.float32))          # [in, out]
    return np.ascontiguousarray(E.T)                                       # [out, in]


def expert_cosine(effective_hf: np.ndarray, source_hf: np.ndarray) -> float:
    """cosine(effective, source) over the flattened HF weight (per-projection quality vs source)."""
    a = effective_hf.astype(np.float64).reshape(-1)
    b = source_hf.astype(np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


# ---------------------------------------------------------------- on-disk cache (per projection)

def cache_path(cache_dir: str, L: int, eid: int, w_name: str) -> str:
    return os.path.join(cache_dir, f"L{L}_E{eid}_{w_name}.npz")


def save_cache(path: str, rec: dict, src_sha: str, cos: float) -> None:
    tmp = path + ".tmp.npz"
    np.savez(tmp, code=rec["code"], rin=rec["rin"], rout=rec["rout"],
             meta=np.array([rec["in_p"], rec["out_p"]], np.int64),
             src_sha=np.array(src_sha), cos=np.array(cos, np.float64))
    os.replace(tmp, path)


def load_cache(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return {"code": z["code"], "rin": z["rin"], "rout": z["rout"],
            "in_p": int(z["meta"][0]), "out_p": int(z["meta"][1]),
            "src_sha": str(z["src_sha"]), "cos": float(z["cos"])}


class TrellisCache:
    """(L, eid, w_name, source w [out,in]) -> EFFECTIVE HF weight [out,in], from disk (encode-on-miss).

    Used by the ladder's ``tcq3_beam256`` bank branch.  On a hit whose stored ``src_sha`` matches the
    live source it decodes the cache; otherwise it encodes inline (deterministic, same beam) and
    caches it, so the ladder is always correct even for experts the pre-encode pool did not cover.
    """

    def __init__(self, cache_dir: str, pause_cb=None):
        self.cache_dir = cache_dir
        self.pause_cb = pause_cb
        os.makedirs(cache_dir, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def effective(self, L: int, eid: int, w_name: str, w_np: np.ndarray) -> np.ndarray:
        w_np = np.ascontiguousarray(w_np, dtype=np.float32)
        path = cache_path(self.cache_dir, L, eid, w_name)
        if os.path.exists(path):
            rec = load_cache(path)
            if rec["src_sha"] == source_sha(w_np):
                self.hits += 1
                return effective_weight_hf(rec)
        # miss (or stale): encode inline and cache
        self.misses += 1
        rec = encode_source_weight(w_np, on_batch=self.pause_cb)
        eff = effective_weight_hf(rec)
        save_cache(path, rec, source_sha(w_np), expert_cosine(eff, w_np))
        return eff
