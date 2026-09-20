"""F38 harness: GPU trellis encoder correctness + throughput, run INSIDE gpu_window.sh (single venv-python process).

Checks, in order (each recorded in the JSON receipt argv[1]):
  pack    GPU pack == tcq_encode.pack_new3 bit for bit on random symbols.
  seam    GPU seam repair vs tcq_encode.repair_seam on random tiles: same symbols-0..7 cost (ties aside) and the
          repaired code is tail-biting consistent (simulate_windows == decoded_windows).
  synth   throughput on synthetic targets: us per tile for the beam+trace / seam / pack kernels.
  real    L20/E3 down_proj (the F34 sample): cosine + rel-RMS of the effective weight vs the FP4 reference through
          the vendor decoder, per-tile SSE vs the stored CPU beam-256 code, ring consistency, threshold stats.
  sweep   beam width 64 / 128 / 256 on the same projection (cosine + us/tile).
Env: TCQ_W (default 256), TCQ_ROUNDS (10), TCQ_BATCH (2048), TCQ_SKIP_REAL=1 to skip the bank read.
"""
from __future__ import annotations

import json
import os
import sys
import time

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import tcq_encode as tq  # noqa: E402
import tcq_encode_metal as gm  # noqa: E402
import mxfp4_source as src  # noqa: E402
from mtplx import eschamoe  # noqa: E402

F34_NPZ = "/Users/davidtai/projects/OpenSourceWTF/reports/dsv41-f34-trellis/dsv41_f34_L20E3_down_proj_full_beam.npz"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    d = (a.astype(np.float64) - b.astype(np.float64)).ravel()
    return float(np.sqrt(np.mean(d * d)) / np.sqrt(np.mean(b.astype(np.float64).ravel() ** 2)))


def tile_sse(new3: np.ndarray, targets: np.ndarray, dec: np.ndarray, ct: dict) -> np.ndarray:
    win = tq.decoded_windows(new3, ct)
    return ((dec[win] - targets) ** 2).sum(axis=1)


def main() -> int:
    out_path = sys.argv[1]
    W = int(os.environ.get("TCQ_W", "256"))
    rounds = int(os.environ.get("TCQ_ROUNDS", "10"))
    batch = int(os.environ.get("TCQ_BATCH", "2048"))
    rec: dict = {"W": W, "rounds": rounds, "batch": batch, "mlx": mx.__version__, "checks": {}}
    dec = tq.build_dec_table()
    ct = tq.cycle_tables(3)
    rng = np.random.default_rng(0)

    def save():
        with open(out_path, "w") as f:
            json.dump(rec, f, indent=1)

    # ---- pack parity
    new3 = rng.integers(0, 8, size=(512, 256), dtype=np.uint8)
    words_gpu = np.array(gm.pack_words(mx.array(new3)))
    words_cpu = tq.pack_new3(new3, ct)
    rec["checks"]["pack_bit_exact"] = bool(np.array_equal(words_gpu, words_cpu))
    print("pack bit-exact:", rec["checks"]["pack_bit_exact"], flush=True)
    save()

    # ---- seam parity on random tiles (open-chain symbols random -> repair)
    T = 256
    targets = (rng.standard_normal((T, 256)) * tq.codebook_rms(dec)).astype(np.float32)
    n3 = rng.integers(0, 8, size=(T, 256), dtype=np.uint8)
    rep_gpu = np.array(gm.seam_repair(mx.array(n3), mx.array(targets)))
    rep_cpu = tq.repair_seam(n3, targets, dec, ct, span=8)
    assert np.array_equal(rep_gpu[:, 8:], n3[:, 8:]), "GPU seam touched symbols >= 8"

    def seam_cost(r):
        win = tq.decoded_windows(r, ct)[:, :8]
        return ((dec[win] - targets[:, :8]) ** 2).sum(axis=1)

    cg, cc = seam_cost(rep_gpu), seam_cost(rep_cpu)
    sim_end = tq.simulate_windows(rep_gpu, tq.decoded_windows(rep_gpu, ct)[:, 0] & 0x1FFF)[1]
    consistent = bool((sim_end == (tq.decoded_windows(rep_gpu, ct)[:, 0] & 0x1FFF)).all())
    rec["checks"]["seam"] = {
        "identical_tiles": int((rep_gpu == rep_cpu).all(axis=1).sum()), "tiles": T,
        "max_cost_diff": float(np.abs(cg - cc).max()), "gpu_cheaper_or_equal": int((cg <= cc + 1e-4).sum()),
        "ring_consistent": consistent,
    }
    print("seam:", rec["checks"]["seam"], flush=True)
    save()

    # ---- synthetic throughput
    N = 8192
    tsyn = mx.array((rng.standard_normal((N, 256)) * tq.codebook_rms(dec)).astype(np.float32))
    mx.eval(tsyn)
    for pass_ in range(2):
        t0 = time.perf_counter()
        n3s, st = gm.beam_new3(tsyn[:batch], W, rounds); mx.eval(n3s, st)
        t1 = time.perf_counter()
        n3r = gm.seam_repair(n3s, tsyn[:batch]); mx.eval(n3r)
        t2 = time.perf_counter()
        wd = gm.pack_words(n3r); mx.eval(wd)
        t3 = time.perf_counter()
    # multi-batch steady state
    t4 = time.perf_counter()
    words, new3_all, stats = gm.encode_tiles(tsyn, W, rounds, batch)
    mx.eval(words)
    t5 = time.perf_counter()
    rec["checks"]["synth"] = {
        "beam_trace_us_per_tile": (t1 - t0) / batch * 1e6, "seam_us_per_tile": (t2 - t1) / batch * 1e6,
        "pack_us_per_tile": (t3 - t2) / batch * 1e6, "steady_us_per_tile": (t5 - t4) / N * 1e6,
        "ntrunc_total": int(np.array(stats[:, 0]).sum()), "nshort_total": int(np.array(stats[:, 1]).sum()),
    }
    us = rec["checks"]["synth"]["steady_us_per_tile"]
    rec["checks"]["synth"]["bank_hours_at_this_rate"] = 2.123e9 * us / 1e6 / 3600.0
    print("synth:", rec["checks"]["synth"], flush=True)
    save()
    if os.environ.get("TCQ_SKIP_REAL") == "1":
        return 0

    # ---- real projection: L20/E3 down_proj vs the FP4 reference and the stored CPU beam code
    W_ref = src.load_projection(20, 3, "down_proj")                       # [out,in] f32 (numpy)
    W_esch_np = src.eschamoe_orientation(W_ref)                            # [in,out]
    W_esch = mx.array(W_esch_np)
    res = gm.encode_projection_gpu(W_esch, W, rounds, batch)
    mx.eval(res["code"], res["rout"], res["new3"], res["stats"])
    W_eff = np.array(gm.decode_effective(res["code"], res["rout"], eschamoe.decode_expert_weights))
    new3_np = np.array(res["new3"])
    targets_np = np.array(res["targets"])
    sse_gpu = tile_sse(new3_np, targets_np, dec, ct)
    # ring consistency of the GPU code
    s0 = tq.decoded_windows(new3_np, ct)[:, 0] & 0x1FFF
    ring_ok = bool((tq.simulate_windows(new3_np, s0)[1] == s0).all())
    # vendor decoder vs own decode of the packed words (pack <-> decoder agreement)
    W_recon_vendor = np.array(eschamoe.decode_expert_weights(res["code"], 3).astype(mx.float32))
    W_recon_own = tq.decode_numpy(np.array(res["code"]), 3, dec).astype(np.float32)
    real = {
        "cosine_vs_fp4": cosine(W_eff, W_esch_np), "rel_rms_vs_fp4": rel_rms(W_eff, W_esch_np),
        "ring_consistent": ring_ok, "vendor_decode_equals_own": bool(np.array_equal(W_recon_vendor, W_recon_own)),
        "ntrunc_total": int(np.array(res["stats"][:, 0]).sum()), "nshort_total": int(np.array(res["stats"][:, 1]).sum()),
        "prep_s": res["prep_s"], "encode_s": res["encode_s"], "n_tiles": int(res["nI"] * res["nJ"]),
        "us_per_tile": res["encode_s"] / (res["nI"] * res["nJ"]) * 1e6,
        "mean_tile_sse": float(sse_gpu.mean()),
    }
    if os.path.exists(F34_NPZ):
        z = np.load(F34_NPZ)
        code_cpu = mx.array(z["escha_code"][0]); rout_cpu = mx.array(z["escha_rout"][0])
        W_eff_cpu = np.array(gm.decode_effective(code_cpu, rout_cpu, eschamoe.decode_expert_weights))
        real["cpu_beam256_cosine_vs_fp4"] = cosine(W_eff_cpu, W_esch_np)
        real["cpu_beam256_rel_rms_vs_fp4"] = rel_rms(W_eff_cpu, W_esch_np)
        real["identical_code_tiles_vs_cpu"] = int((np.array(res["code"]).reshape(-1, 48) == z["escha_code"][0].reshape(-1, 48)).all(axis=1).sum())
        real["rout_max_rel_diff_vs_cpu"] = float(np.abs(np.array(res["rout"]).astype(np.float32) - z["escha_rout"][0].astype(np.float32)).max() / np.abs(z["escha_rout"][0].astype(np.float32)).max())
    rec["checks"]["real_L20E3_down"] = real
    print("real:", real, flush=True)
    save()

    # ---- beam width sweep on the same projection
    sweep = {}
    for Ws in (64, 128, 256):
        r2 = gm.encode_projection_gpu(W_esch, Ws, rounds, batch)
        mx.eval(r2["code"])
        W_eff2 = np.array(gm.decode_effective(r2["code"], r2["rout"], eschamoe.decode_expert_weights))
        sweep[str(Ws)] = {"cosine_vs_fp4": cosine(W_eff2, W_esch_np), "rel_rms": rel_rms(W_eff2, W_esch_np),
                          "us_per_tile": r2["encode_s"] / (r2["nI"] * r2["nJ"]) * 1e6,
                          "ntrunc_total": int(np.array(r2["stats"][:, 0]).sum())}
        print("sweep", Ws, sweep[str(Ws)], flush=True)
        save()
    rec["checks"]["sweep"] = sweep
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
