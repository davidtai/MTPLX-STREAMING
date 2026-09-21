#!/usr/bin/env python3
"""DSV4.1 F37 — validate the C beam step (acceptance for Fable): CPU only, main .venv.

  (1) identical codes to the F34 beam on two F34 subsets, with the C deterministic tie rule; on any
      tie-difference, report the count and show the effective-weight cosine agrees to 1e-6;
  (2) ms/tile before (F34 beam_encode) vs after (beam_encode_c) on the same synthetic tile batch;
  plus a bit-exact check of beam_encode_c vs beam_encode_fast on synthetic data.

Reads the mxfp4 bank for the subsets (gated: pauses while a GPU window/lock is active).
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
import tcq_verify as ver          # noqa: E402
import tcq_beam_c as C            # noqa: E402

SUBSETS = [(20, 3, "down_proj"), (1, 3, "gate_proj")]
GPU_LOCK = "/tmp/mtplx-gpu-exclusive.lock"
WINDOW = "/tmp/dsv41-fable-window.active"


def gpu_busy():
    if os.path.exists(WINDOW):
        return True
    try:
        r = subprocess.run(["lsof", GPU_LOCK], capture_output=True, timeout=15)
        return r.returncode == 0 and len(r.stdout.splitlines()) > 1
    except Exception:
        return False


def wait_while_busy(what="c-step"):
    while gpu_busy():
        print(f"[gate] GPU busy; holding {what}", flush=True)
        time.sleep(30)


def main():
    dec = enc.build_dec_table()
    ct = enc.cycle_tables(3)
    rng = np.random.default_rng(0)
    res = {}

    # bit-exact C vs beam_encode_fast (synthetic)
    ok = True
    for scale in (0.3, 1.2, 3.0):
        t = (rng.standard_normal((256, 256)).astype(np.float32) * scale)
        of, _ = enc.beam_encode_fast(t, dec, beam=256)
        oc, _ = C.beam_encode_c(t, dec, beam=256)
        ok = ok and np.array_equal(of, oc)
    res["c_eq_fast_synth"] = bool(ok)
    print(f"[synth] beam_encode_c == beam_encode_fast: {ok}", flush=True)

    # ms/tile before/after on the same synthetic batch
    tb = (rng.standard_normal((2048, 256)).astype(np.float32) * 1.2)
    enc.beam_encode(tb[:16], dec, beam=256); C.beam_encode_c(tb[:16], dec, beam=256)
    t0 = time.perf_counter(); enc.beam_encode(tb, dec, beam=256); d_ref = time.perf_counter() - t0
    t0 = time.perf_counter(); C.beam_encode_c(tb, dec, beam=256); d_c = time.perf_counter() - t0
    res["ms_per_tile_beam_encode"] = d_ref / 2048 * 1000
    res["ms_per_tile_beam_encode_c"] = d_c / 2048 * 1000
    res["speedup"] = d_ref / d_c
    print(f"[speed] beam_encode {res['ms_per_tile_beam_encode']:.3f} ms/tile ; "
          f"beam_encode_c {res['ms_per_tile_beam_encode_c']:.3f} ms/tile ; {res['speedup']:.2f}x", flush=True)

    # identical codes to F34 beam on two subsets (rows 0..255)
    res["subsets"] = {}
    for (L, E, comp) in SUBSETS:
        wait_while_busy(f"read L{L}E{E} {comp}")
        W_esch = src.eschamoe_orientation(src.load_projection(L, E, comp))
        rin, rout = enc.compute_scales(W_esch, enc.codebook_rms(dec))
        W_hat = enc.target_what(W_esch, rin, rout)[0:256]
        targets, (nI, nJ) = enc.matrix_to_cycle_targets(W_hat, ct)
        wait_while_busy(f"beam L{L}E{E} {comp}")
        o_ref, _ = enc.beam_encode(targets, dec, beam=256)
        o_c, _ = C.beam_encode_c(targets, dec, beam=256)
        nmis = int((o_ref != o_c).sum())
        # cosine of the effective weight for each (only meaningful if a tile differs)
        entry = {"n_tiles": int(targets.shape[0]), "code_mismatches": nmis}
        if nmis:
            rin_s = rin[0:256]
            for name, o in (("f34_beam", o_ref), ("c_beam", o_c)):
                code = enc.build_expert_code(o, nI, nJ, ct)
                W_recon = enc.decode_numpy(code, 3, dec).astype(np.float32)
                W_eff = enc.effective_weight(W_recon, rin_s, rout)
                entry[f"cos_{name}"] = ver.metrics(W_eff, W_esch[0:256])["cosine"]
            entry["cos_delta"] = abs(entry["cos_f34_beam"] - entry["cos_c_beam"])
        res["subsets"][f"L{L}_E{E}_{comp}"] = entry
        print(f"[subset] L{L}E{E} {comp} ({entry['n_tiles']} tiles): code_mismatches={nmis}"
              + (f" cosΔ={entry.get('cos_delta'):.2e}" if nmis else " (identical)"), flush=True)

    out = "/Users/davidtai/projects/OpenSourceWTF/reports/dsv41-f37-trellis-ladder/c_step_validation.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("RESULT " + json.dumps({k: v for k, v in res.items() if k != "subsets"}))
    print(f"[written] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
