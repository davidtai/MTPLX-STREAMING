#!/usr/bin/env python3
"""DSV4.1 F37 — beam WIDTH vs quality sweep on the F34 subsets (CPU only, for the GPU encoder).

For two F34 sample projections (rows 0..255), encode with beam = 16/32/64/128/256 through the EXISTING
F34 harness (`tcq_encode.encode_projection`, method="beam": tail-biting seam repair, actual vendor-
layout decode), and report per width: cosine vs FP4 of the effective weight, mean per-tile SSE
relative to the beam-256 SSE, and ms/tile.  Writes `reports/dsv41-f37-trellis-ladder/beam_width_sweep.json`.

Reads the mxfp4 bank (gated: pauses while a GPU window/lock is active).  Run in the main .venv.
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

OUT = "/Users/davidtai/projects/OpenSourceWTF/reports/dsv41-f37-trellis-ladder/beam_width_sweep.json"
SUBSETS = [(20, 3, "down_proj"), (20, 3, "gate_proj")]
WIDTHS = [16, 32, 64, 128, 256]
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


def wait_while_busy(what="beam"):
    while gpu_busy():
        print(f"[gate] GPU busy; holding {what}", flush=True)
        time.sleep(30)


def main():
    dec = enc.build_dec_table()
    ct = enc.cycle_tables(3)
    report = {"subsets": {}, "widths": WIDTHS, "rows": 256}
    for (L, E, comp) in SUBSETS:
        wait_while_busy(f"read L{L}E{E} {comp}")
        W_esch = src.eschamoe_orientation(src.load_projection(L, E, comp))
        key = f"L{L}_E{E}_{comp}"
        rows = {}
        for w in WIDTHS:
            wait_while_busy(f"beam{w} {key}")
            r = enc.encode_projection(W_esch, "beam", dec, ct, beam=w, row_slice=slice(0, 256),
                                      on_batch=lambda: wait_while_busy("beam"))
            cos = ver.metrics(r["W_eff"], r["W_esch_used"])["cosine"]
            rows[w] = {"cosine": cos, "mean_sse": float(r["sse_tiles"].mean()),
                       "ms_per_tile": r["encode_s"] / r["n_tiles"] * 1000.0,
                       "n_tiles": int(r["n_tiles"]), "consistent_frac": float(np.mean(r["consistent"]))}
            print(f"[{key}] beam={w:3d} cos={cos:.5f} mean_sse={rows[w]['mean_sse']:.5f} "
                  f"ms/tile={rows[w]['ms_per_tile']:.3f}", flush=True)
        base = rows[256]
        for w in WIDTHS:
            rows[w]["sse_rel_to_256"] = rows[w]["mean_sse"] / base["mean_sse"]
            rows[w]["cos_delta_vs_256"] = rows[w]["cosine"] - base["cosine"]
        report["subsets"][key] = rows

    # smallest width within 0.0005 cosine of beam-256 (worst case over subsets)
    smallest = None
    for w in WIDTHS:
        if all(abs(report["subsets"][k][w]["cos_delta_vs_256"]) <= 5e-4 for k in report["subsets"]):
            smallest = w
            break
    report["smallest_width_within_0.0005_cosine"] = smallest
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nsmallest width within 0.0005 cosine of beam-256 (both subsets): {smallest}")
    print(f"[written] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
