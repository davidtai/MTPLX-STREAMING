#!/usr/bin/env python3
"""F37 encode-pool worker (CPU-only, one shard of the projection job list).

Runs as a ``nice -n 19`` subprocess (4 at most, launched by :mod:`run_f37`).  For each
``(layer, expert, w_name)`` job in its shard it loads the SOURCE fp4 weight from the torch-reference
shards (``~/models/DeepSeek-V4.1-Flash-src`` — the exact weight the ladder's R0 baseline uses),
encodes it into the eschamoe K=3 beam-256 trellis and writes ``code``+``rin``+``rout``+cosine to the
cache.  Between beam tile batches it PAUSES (sleep 30 s, re-check) whenever the GPU exclusive lock is
held by anyone or ``/tmp/dsv41-fable-window.active`` exists, so its CPU load never disturbs another
session's timing window.

Single-threaded (BLAS env + torch 1 thread), RSS-bounded (one projection at a time, beam batch 512).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.nice(19)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                                   # trellis
sys.path.insert(0, str(HERE.parent / "torchref"))               # ref_forward

import numpy as np
import torch
import mlx.core as mx

mx.set_default_device(mx.cpu)
torch.set_num_threads(1)
torch.set_grad_enabled(False)

import ref_forward as RF          # noqa: E402
import trellis_ladder as TL       # noqa: E402

GPU_LOCK = "/tmp/mtplx-gpu-exclusive.lock"
WINDOW = "/tmp/dsv41-fable-window.active"

_PAUSED = {"s": 0.0}


def gpu_busy() -> bool:
    """True if another session's timing window is active or the GPU exclusive lock is held."""
    if os.path.exists(WINDOW):
        return True
    try:
        r = subprocess.run(["lsof", GPU_LOCK], capture_output=True, timeout=15)
        return r.returncode == 0 and len(r.stdout.splitlines()) > 1   # header + >=1 holder
    except Exception:
        return False


def pause_cb(poll: int = 30) -> None:
    """Called before each beam tile batch: finish the current batch, then sleep/re-check while busy."""
    while gpu_busy():
        time.sleep(poll)
        _PAUSED["s"] += poll


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)         # json list of [L, eid, w_name]
    ap.add_argument("--cache", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()

    jobs = json.loads(Path(a.jobs).read_text())
    mine = [tuple(j) for i, j in enumerate(jobs) if i % a.nshards == a.shard]
    os.makedirs(a.cache, exist_ok=True)
    shards = RF.Shards(RF.SRC)
    t0 = time.time()
    done = skipped = 0
    for (L, eid, wn) in mine:
        path = TL.cache_path(a.cache, L, eid, wn)
        if os.path.exists(path):                     # resume: already encoded
            skipped += 1
            continue
        pause_cb()                                   # do not even start a projection during a window
        w = shards.dequant_weight(f"layers.{L}.ffn.experts.{eid}.{wn}.weight")   # torch f32 [out,in]
        w_np = np.ascontiguousarray(w.numpy(), dtype=np.float32)
        del w
        rec = TL.encode_source_weight(w_np, on_batch=pause_cb, batch=a.batch)
        eff = TL.effective_weight_hf(rec)
        cos = TL.expert_cosine(eff, w_np)
        TL.save_cache(path, rec, TL.source_sha(w_np), cos)
        done += 1
        print(f"[w{a.shard}] L{L}E{eid}.{wn} cos={cos:.5f} ({done} done, {time.time()-t0:.0f}s, paused {_PAUSED['s']:.0f}s)", flush=True)

    status = {"shard": a.shard, "done": done, "skipped": skipped,
              "wall_s": time.time() - t0, "paused_s": _PAUSED["s"], "n_assigned": len(mine)}
    Path(a.cache, f"_worker{a.shard}.json").write_text(json.dumps(status))
    print(f"[w{a.shard}] DONE {status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
