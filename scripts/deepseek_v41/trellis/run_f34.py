#!/usr/bin/env python3
"""DSV4.1 F34 sample runner (CPU-only).  Encodes the trellis sample, verifies it against the FP4
reference with the EXISTING eschamoe decoder, serializes ONE full projection, and writes results
incrementally to ``reports/dsv41-f34-trellis/f34_results.json``.

Gate-aware: before every heavy chunk (bank read + encode) it waits while ``/tmp/dsv41-fable-window
.active`` exists, polling every 60 s, so a guarded GPU window is never disturbed.

Usage:  run_f34.py [subsets|full|all]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)

import mxfp4_source as src  # noqa: E402
import tcq_encode as enc  # noqa: E402
import tcq_verify as ver  # noqa: E402

WINDOW = "/tmp/dsv41-fable-window.active"
REPORT_DIR = "/Users/davidtai/projects/OpenSourceWTF/reports/dsv41-f34-trellis"
RESULTS = os.path.join(REPORT_DIR, "f34_results.json")
SAMPLE = [(1, 3), (1, 250), (20, 3), (20, 250), (33, 3), (33, 250)]
COMPONENTS = ("gate_proj", "up_proj", "down_proj")
ROWS = 256              # subset: input-dim rows 0..255 of the eschamoe-orient weight [in,out]
FULL = (20, 3, "down_proj")


def wait_for_gate(poll: int = 60) -> None:
    waited = 0
    while os.path.exists(WINDOW):
        print(f"[gate] {WINDOW} active; waiting {poll}s (total {waited}s)", flush=True)
        time.sleep(poll)
        waited += poll


def affine_cosine(W: np.ndarray, bits: int, gs: int = 64) -> float:
    """Per-group affine (scale+zero) quantization cosine vs W, groups of ``gs`` along axis 0 (in-dim)."""
    n_in, n_out = W.shape
    ng = n_in // gs
    Wg = W.T.reshape(n_out, ng, gs).astype(np.float32)     # [out, ng, gs]
    mn = Wg.min(axis=2, keepdims=True)
    mx_ = Wg.max(axis=2, keepdims=True)
    scale = (mx_ - mn) / (2 ** bits - 1)
    scale = np.where(scale == 0, 1.0, scale)
    deq = (np.round((Wg - mn) / scale) * scale + mn).reshape(n_out, n_in).T
    a = W.reshape(-1).astype(np.float64)
    b = deq.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _load_results() -> dict:
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    return {"subsets": {}, "full": None, "meta": {}}


def _save(results: dict) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    tmp = RESULTS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=1)
    os.replace(tmp, RESULTS)


def run_subsets(dec, ct, results: dict) -> None:
    rowsl = slice(0, ROWS)
    for (L, E) in SAMPLE:
        for comp in COMPONENTS:
            key = f"L{L}_E{E}_{comp}"
            if key in results["subsets"]:
                print(f"[skip] {key} (already done)", flush=True)
                continue
            wait_for_gate()
            t0 = time.perf_counter()
            W_ref = src.load_projection(L, E, comp, verify_sha=False)   # HF [out,in]
            W_esch = src.eschamoe_orientation(W_ref)                    # [in,out]
            rec = {"in": int(W_esch.shape[0]), "out": int(W_esch.shape[1]),
                   "read_s": time.perf_counter() - t0}
            for method in ("viterbi", "beam"):
                r = enc.encode_projection(W_esch, method, dec, ct, beam=256, row_slice=rowsl,
                                          on_batch=wait_for_gate)
                vr = ver.verify_expert(r["code"], ct["K"], r["rin_used"], r["rout"],
                                       r["W_esch_used"], dec)
                rec[method] = {
                    "cosine": vr["cosine"], "rel_rms": vr["rel_rms"], "max_abs": vr["max_abs"],
                    "bit_exact": bool(vr["bit_exact"]), "chain": vr["chain"],
                    "encode_s": r["encode_s"], "n_tiles": int(r["n_tiles"]),
                    "ms_per_tile": r["encode_s"] / int(r["n_tiles"]) * 1000.0,
                    "consistent_frac": float(np.mean(r["consistent"])),
                }
            W_sub = r["W_esch_used"]
            rec["affine"] = {f"q{b}_gs64": affine_cosine(W_sub, b, 64) for b in (3, 4, 6)}
            results["subsets"][key] = rec
            _save(results)
            print(f"[done] {key}: viterbi cos={rec['viterbi']['cosine']:.5f} "
                  f"beam cos={rec['beam']['cosine']:.5f} affine q3={rec['affine']['q3_gs64']:.5f} "
                  f"(read {rec['read_s']:.1f}s, vit {rec['viterbi']['encode_s']:.1f}s)", flush=True)


def run_full(dec, ct, results: dict) -> None:
    if results.get("full"):
        print("[skip] full projection (already done)", flush=True)
        return
    L, E, comp = FULL
    wait_for_gate()
    t0 = time.perf_counter()
    W_ref = src.load_projection(L, E, comp, verify_sha=True)            # HF [out,in]; verify hash
    W_esch = src.eschamoe_orientation(W_ref)
    read_s = time.perf_counter() - t0
    r = enc.encode_projection(W_esch, "beam", dec, ct, beam=256,        # FULL projection, beam
                              on_batch=wait_for_gate)
    vr = ver.verify_expert(r["code"], ct["K"], r["rin"], r["rout"], W_esch, dec)
    # serialize the vendor fields (E=1)
    expert = {"code": r["code"], "rin": r["rin"], "rout": r["rout"]}
    npz_path = os.path.join(REPORT_DIR, f"dsv41_f34_L{L}E{E}_{comp}_full_beam.npz")
    summ = enc.serialize(npz_path, [expert], K=ct["K"])
    rec_sha = src.get_record(L, E).sha256
    n_weights = W_esch.size
    total_bytes = summ["bytes_code"] + summ["bytes_rin"] + summ["bytes_rout"]
    sidecar = {
        "artifact": os.path.basename(npz_path),
        "experts": [{"layer": L, "expert": E, "component": comp, "record_sha256": rec_sha,
                     "in": int(W_esch.shape[0]), "out": int(W_esch.shape[1])}],
        "encoder": {"method": "beam", "beam": 256, "K": ct["K"], "repair_span": 8,
                    "scales": "rin=ones, rout=RMS(H.W)/codebook_rms", "alpha": 1.0},
        "codebook_rms": enc.codebook_rms(dec),
        "bytes": {"code": summ["bytes_code"], "rin": summ["bytes_rin"], "rout": summ["bytes_rout"],
                  "total": total_bytes},
        "bits_per_weight_incl_scales": total_bytes * 8.0 / n_weights,
        "bits_per_weight_code_only": summ["bytes_code"] * 8.0 / n_weights,
        "timings": {"read_s": read_s, "encode_s": r["encode_s"], "n_tiles": int(r["n_tiles"]),
                    "ms_per_tile": r["encode_s"] / int(r["n_tiles"]) * 1000.0},
        "verify": {"bit_exact": bool(vr["bit_exact"]), "n_mismatch": int(vr["n_mismatch"]),
                   "cosine": vr["cosine"], "rel_rms": vr["rel_rms"], "max_abs": vr["max_abs"],
                   "chain": vr["chain"]},
        "consistent_frac": float(np.mean(r["consistent"])),
        "affine_full": {f"q{b}_gs64": affine_cosine(W_esch, b, 64) for b in (3, 4, 6)},
    }
    with open(npz_path.replace(".npz", ".json"), "w") as f:
        json.dump(sidecar, f, indent=1)
    results["full"] = {"npz": os.path.basename(npz_path), "sidecar": sidecar}
    _save(results)
    print(f"[done] FULL {comp}: bit_exact={vr['bit_exact']} cos={vr['cosine']:.5f} "
          f"bpw={sidecar['bits_per_weight_incl_scales']:.3f} bytes={total_bytes} "
          f"encode={r['encode_s']:.1f}s", flush=True)


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    dec = enc.build_dec_table()
    ct = enc.cycle_tables(3)
    results = _load_results()
    results["meta"] = {"codebook_rms": enc.codebook_rms(dec), "rows_subset": ROWS,
                       "sample": [list(x) for x in SAMPLE], "full": list(FULL)}
    _save(results)
    if what in ("subsets", "all"):
        run_subsets(dec, ct, results)
    if what in ("full", "all"):
        run_full(dec, ct, results)
    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
