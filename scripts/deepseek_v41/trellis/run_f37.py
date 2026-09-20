#!/usr/bin/env python3
"""DSV4.1 F37 runner: put tcq3 (eschamoe K=3, beam-256) on the torch-reference bank ladder.

Stages (run in the torchref venv, ``.venv-torchref/bin/python``; CPU-only):

  discover  run the SOURCE fp4 forward for layers 0..max, collect the ~191 (layer,expert) pairs the
            31-token probe routes through, and write ``cache/routed_experts.json`` + ``cache/jobs.json``
            (573 projection jobs).
  encode    spawn a 4-process ``nice -n 19`` single-thread-BLAS pool (:mod:`encode_worker`) that
            encodes those projections into the trellis cache, PAUSING whenever the GPU lock is held or
            ``/tmp/dsv41-fable-window.active`` exists; resumable (skips already-cached projections).
  ladder    run every stored format + tcq3_beam256 (effective weights from the cache; encode-on-miss)
            for --max-layer, and write ``torchref_bank_ladder_tcq3.json`` (stored-schema + tcq3 rows).

``--stage all`` (default) runs all three.  The runner itself waits while the GPU is busy before its
own forward passes.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    os.setpriority(os.PRIO_PROCESS, 0, 19)     # absolute nice 19 (idempotent; survives a nice prefix)
except (AttributeError, OSError):
    pass
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "torchref"))

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

import bank_ladder as BL          # noqa: E402
import trellis_ladder as TL       # noqa: E402

REPORT_DIR = Path("/Users/davidtai/projects/OpenSourceWTF/reports/dsv41-f37-trellis-ladder")
CACHE_DIR = REPORT_DIR / "cache"
RECEIPT = REPORT_DIR / "torchref_bank_ladder_tcq3.json"
GPU_LOCK = "/tmp/mtplx-gpu-exclusive.lock"
WINDOW = "/tmp/dsv41-fable-window.active"


def gpu_busy() -> bool:
    if os.path.exists(WINDOW):
        return True
    try:
        r = subprocess.run(["lsof", GPU_LOCK], capture_output=True, timeout=15)
        return r.returncode == 0 and len(r.stdout.splitlines()) > 1
    except Exception:
        return False


def wait_while_busy(what: str, poll: int = 30) -> float:
    paused = 0.0
    while gpu_busy():
        print(f"[gate] GPU busy; holding {what} ({paused:.0f}s)", flush=True)
        time.sleep(poll)
        paused += poll
    return paused


# --------------------------------------------------------------------- stages

def discover(ctx, max_layer):
    wait_while_busy("route discovery")
    t0 = time.time()
    R0 = BL.forward(ctx, BL.ExpertBank(ctx["shards"], None, None, None), max_layer)   # source
    routed = sorted(BL.routed_experts(R0, max_layer))
    jobs = [[L, eid, wn] for (L, eid) in routed for wn in TL.W_NAMES]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "routed_experts.json").write_text(json.dumps(
        {"max_layer": max_layer, "n_experts": len(routed), "n_projections": len(jobs),
         "routed": routed}, indent=1))
    (CACHE_DIR / "jobs.json").write_text(json.dumps(jobs))
    print(f"[discover] {len(routed)} experts, {len(jobs)} projections through L0..{max_layer} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return R0, routed, jobs


def run_pool(jobs_path, nproc=4, batch=512):
    jobs = json.loads(Path(jobs_path).read_text())
    remaining = [j for j in jobs if not os.path.exists(TL.cache_path(str(CACHE_DIR), *j))]
    print(f"[encode] {len(jobs)} projections, {len(remaining)} not yet cached; {nproc} workers", flush=True)
    if not remaining:
        return {"wall_s": 0.0, "paused_s": 0.0, "workers": []}
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[v] = "1"
    t0 = time.time()
    procs = []
    for i in range(nproc):
        cmd = [sys.executable, str(HERE / "encode_worker.py"), "--jobs", str(jobs_path),
               "--cache", str(CACHE_DIR), "--shard", str(i), "--nshards", str(nproc), "--batch", str(batch)]
        procs.append(subprocess.Popen(cmd, env=env))
    rc = [p.wait() for p in procs]
    wall = time.time() - t0
    stats = []
    for i in range(nproc):
        f = CACHE_DIR / f"_worker{i}.json"
        if f.exists():
            stats.append(json.loads(f.read_text()))
    paused = max([s.get("paused_s", 0.0) for s in stats], default=0.0)
    print(f"[encode] pool done in {wall:.0f}s (max paused {paused:.0f}s), rc={rc}", flush=True)
    return {"wall_s": wall, "paused_s": paused, "workers": stats, "rc": rc}


def tcq3_quality(routed):
    """Aggregate per-projection effective-vs-source cosine and byte sizes from the cache."""
    cosines, code_b, rin_b, rout_b = [], 0, 0, 0
    per_expert_code = per_expert_scale = 0
    n_weights = 0
    sample_expert = routed[0] if routed else None
    for (L, eid) in routed:
        for wn in TL.W_NAMES:
            p = TL.cache_path(str(CACHE_DIR), L, eid, wn)
            if not os.path.exists(p):
                continue
            rec = TL.load_cache(p)
            cosines.append(rec["cos"])
            cb = int(np.asarray(rec["code"]).nbytes)
            rb = int(np.asarray(rec["rin"]).nbytes)
            ob = int(np.asarray(rec["rout"]).nbytes)
            code_b += cb; rin_b += rb; rout_b += ob
            if sample_expert is not None and (L, eid) == tuple(sample_expert):
                per_expert_code += cb; per_expert_scale += rb + ob
                n_weights += rec["in_p"] * rec["out_p"]
    if not cosines:
        return None
    bytes_per_record = per_expert_code + per_expert_scale
    return {
        "bits": 3, "group_size": None, "mode": "tcq3_beam256",
        "mean_cos_vs_source": float(np.mean(cosines)), "min_cos_vs_source": float(np.min(cosines)),
        "n_projections_cached": len(cosines),
        "bytes_per_record": bytes_per_record,
        "bank_bytes_40x384": bytes_per_record * 40 * 384,
        "bank_GiB_40x384": round(bytes_per_record * 40 * 384 / 2**30, 2),
        "bits_per_weight_incl_scales": (per_expert_code + per_expert_scale) * 8.0 / max(n_weights, 1),
    }


def run_ladder(ctx, R0, routed, max_layer, encode_stats=None):
    tcq = TL.TrellisCache(str(CACHE_DIR), pause_cb=lambda: wait_while_busy("tcq3 encode-on-miss"))
    formats = list(BL.FORMATS) + [("tcq3_beam256", 3, None, "tcq3")]
    runs = {"source": R0}
    t0 = time.time()
    for fname, bits, gs, mode in formats:
        if fname == "source":
            continue
        wait_while_busy(f"{fname} forward")
        bank = BL.ExpertBank(ctx["shards"], bits, gs, mode, tcq_cache=tcq if mode == "tcq3" else None)
        runs[fname] = BL.forward(ctx, bank, max_layer)
        print(f"[ladder] {fname} forward done ({time.time()-t0:.0f}s)", flush=True)
    report = BL.build_report(runs, formats, max_layer, ctx["probe"])
    q = tcq3_quality(routed)
    if q is not None:
        report["expert_quality_vs_source"]["tcq3_beam256"] = q
    report["tcq3_meta"] = {
        "encoder": "eschamoe K=3 beam-256 (tcq_encode.beam_encode_fast), rin=ones, rout=RMS(H.W)/codebook_rms",
        "cache_dir": str(CACHE_DIR), "tcq_cache_hits": tcq.hits, "tcq_cache_misses": tcq.misses,
        "encode": encode_stats,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(report, indent=2))
    print(f"[ladder] DONE -> {RECEIPT}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-layer", type=int, default=7)
    ap.add_argument("--stage", choices=["discover", "encode", "ladder", "all"], default="all")
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ctx = R0 = routed = None
    if a.stage in ("discover", "all"):
        ctx = BL.setup()
        R0, routed, jobs = discover(ctx, a.max_layer)
    enc_stats = None
    if a.stage in ("encode", "all"):
        enc_stats = run_pool(CACHE_DIR / "jobs.json", nproc=a.nproc, batch=a.batch)
    if a.stage in ("ladder", "all"):
        if ctx is None:
            ctx = BL.setup()
        if routed is None:
            routed = [tuple(x) for x in json.loads((CACHE_DIR / "routed_experts.json").read_text())["routed"]]
        routed = [(L, e) for (L, e) in routed if L <= a.max_layer]     # experts used at this depth
        if R0 is None:
            wait_while_busy("source forward (ladder)")
            R0 = BL.forward(ctx, BL.ExpertBank(ctx["shards"], None, None, None), a.max_layer)
        run_ladder(ctx, R0, routed, a.max_layer, encode_stats=enc_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
