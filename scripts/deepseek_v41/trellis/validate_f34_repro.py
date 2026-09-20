#!/usr/bin/env python3
"""Prove beam_encode_fast reproduces F34's beam codes BIT-EXACTLY on real DSV4.1 experts (CPU-only).

Two checks (run in the main .venv, the venv that produced the F34 artifact):
  subsets  re-encode the input-row-0..255 subset of >=2 F34 sample projections with BOTH the F34
           beam (:func:`tcq_encode.beam_encode`) and the vectorized :func:`beam_encode_fast`, assert
           identical codes;
  artifact re-encode the FULL L20/E3/down_proj projection with beam_encode_fast and assert its packed
           code equals the stored artifact ``reports/dsv41-f34-trellis/dsv41_f34_L20E3_down_proj_full_beam.npz``.

Reads the mxfp4 bank (gated: pauses while a GPU window/lock is active).  Usage: validate_f34_repro.py [subsets|artifact|all]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

import mxfp4_source as src        # noqa: E402
import tcq_encode as enc          # noqa: E402

ART = "/Users/davidtai/projects/OpenSourceWTF/reports/dsv41-f34-trellis/dsv41_f34_L20E3_down_proj_full_beam.npz"
GPU_LOCK = "/tmp/mtplx-gpu-exclusive.lock"
WINDOW = "/tmp/dsv41-fable-window.active"
SUBSETS = [(1, 3, "gate_proj"), (20, 3, "down_proj")]     # two F34 sample subsets


def gpu_busy():
    if os.path.exists(WINDOW):
        return True
    try:
        r = subprocess.run(["lsof", GPU_LOCK], capture_output=True, timeout=15)
        return r.returncode == 0 and len(r.stdout.splitlines()) > 1
    except Exception:
        return False


def wait_while_busy(what):
    while gpu_busy():
        print(f"[gate] GPU busy; holding {what}", flush=True)
        time.sleep(30)


def _targets(W_esch, dec, ct, row_slice=None):
    rin, rout = enc.compute_scales(W_esch, enc.codebook_rms(dec))
    W_hat = enc.target_what(W_esch, rin, rout)
    if row_slice is not None:
        W_hat = W_hat[row_slice]
    return enc.matrix_to_cycle_targets(W_hat, ct)


def run_subsets(dec, ct):
    ok = True
    for (L, E, comp) in SUBSETS:
        wait_while_busy(f"read L{L}E{E} {comp}")
        W_esch = src.eschamoe_orientation(src.load_projection(L, E, comp))
        targets, _ = _targets(W_esch, dec, ct, row_slice=slice(0, 256))
        wait_while_busy(f"encode L{L}E{E} {comp}")
        o_ref, s_ref = enc.beam_encode(targets, dec, beam=256, on_batch=lambda: wait_while_busy("beam"))
        o_fast, s_fast = enc.beam_encode_fast(targets, dec, beam=256, on_batch=lambda: wait_while_busy("beam_fast"))
        same = np.array_equal(o_ref, o_fast) and np.array_equal(s_ref, s_fast)
        ok = ok and same
        print(f"[subset] L{L}E{E} {comp} rows0:256 ({targets.shape[0]} tiles): "
              f"beam_fast==beam {same} (mism {int((o_ref != o_fast).sum())})", flush=True)
    return ok


def run_artifact(dec, ct):
    if not os.path.exists(ART):
        print(f"[artifact] SKIP (missing {ART})", flush=True)
        return None
    stored = np.load(ART)["escha_code"][0]       # [nI,nJ,48]
    wait_while_busy("read L20E3 down_proj (full)")
    W_esch = src.eschamoe_orientation(src.load_projection(20, 3, "down_proj", verify_sha=True))
    targets, (nI, nJ) = _targets(W_esch, dec, ct)
    t0 = time.time()
    new3, _ = enc.beam_encode_fast(targets, dec, beam=256, on_batch=lambda: wait_while_busy("beam_fast"))
    code = enc.build_expert_code(new3, nI, nJ, ct)
    same = np.array_equal(code.astype(np.int16), stored.astype(np.int16))
    print(f"[artifact] L20E3 down_proj FULL ({nI*nJ} tiles, {time.time()-t0:.0f}s): "
          f"beam_fast code == stored {same} (mism {int((code != stored).sum())})", flush=True)
    return same


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    dec = enc.build_dec_table()
    ct = enc.cycle_tables(3)
    res = {}
    if what in ("subsets", "all"):
        res["subsets_bit_exact"] = run_subsets(dec, ct)
    if what in ("artifact", "all"):
        res["artifact_bit_exact"] = run_artifact(dec, ct)
    print("RESULT " + json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
