"""F12 microbench: serial vs parallel read+hash on the REAL packed-scale artifact.

CPU only, no MLX. Uses the EXACT production per-file algorithm
(O_RDONLY|O_NOFOLLOW, F_NOCACHE=1, F_RDAHEAD 45->0, 8 MiB chunked preadv, sha256)
but on plain ``bytearray`` buffers, to isolate the file open/read/hash cost that
F12 moves off the main thread.

MEMORY SAFETY (hard rules): it reads one file at a time PER WORKER and drops the
buffer immediately after the digest check, and a shared byte budget caps total
concurrent buffer memory (default 180 MB; the artifact's worst-case 16-consecutive
window is ~158 MB, so the cap does not bind in practice). Peak process RSS is
measured (resource.ru_maxrss) and printed per pass. NO metadata pass pre-allocates
buffers, and NOTHING is allocated before the run gate passes.

RUN GATE (checked immediately before EACH pass): run ONLY when BOTH
  (a) the flag file /tmp/dsv41-fable-window.active does NOT exist, and
  (b) `lsof /tmp/mtplx-gpu-exclusive.lock` shows no holder.
The initial gate polls every 60 s up to 90 min. If the flag REAPPEARS between
passes, the series stops (a resumed window must not be overlapped). Run under
``nice -n 19``.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import resource
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_CHUNK = 8 * 1024 ** 2
_F_RDAHEAD = 45
_LOCK = "/tmp/mtplx-gpu-exclusive.lock"
_FLAG = "/tmp/dsv41-fable-window.active"
_BUDGET = 180 * 1024 * 1024
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_FIELDS = ("descriptors", "payload", "bases")


def _window_open() -> bool:
    if os.path.exists(_FLAG):
        return False
    r = subprocess.run(["lsof", _LOCK], capture_output=True, text=True)
    return not r.stdout.strip()


def _await_window(poll: float = 60.0, timeout: float = 90 * 60) -> None:
    deadline = time.time() + timeout
    while not _window_open():
        if time.time() > deadline:
            raise SystemExit("Fable window / GPU lock busy longer than timeout; refusing SSD bench")
        why = "fable-window flag present" if os.path.exists(_FLAG) else "gpu exclusive lock held"
        print(f"  [{why}; waiting {poll:.0f}s ...]", flush=True)
        time.sleep(poll)


class _ByteBudget:
    """Cap total concurrent buffer bytes; a lone file may exceed the cap alone."""

    def __init__(self, limit: int):
        self._limit = limit
        self._used = 0
        self._cv = threading.Condition()

    def reserve(self, n: int) -> None:
        with self._cv:
            while self._used and self._used + n > self._limit:
                self._cv.wait()
            self._used += n

    def release(self, n: int) -> None:
        with self._cv:
            self._used -= n
            self._cv.notify_all()


def _read_hash(path, nbytes, sha256, budget: _ByteBudget):
    """Allocate one file buffer, read+hash+verify, drop it. Returns (read_ns, hash_ns)."""
    budget.reserve(nbytes)
    buf = None
    try:
        buf = bytearray(nbytes)
        mv = memoryview(buf)
        t0 = time.perf_counter_ns()
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            fcntl.fcntl(fd, _F_RDAHEAD, 0)
            cursor = 0
            while cursor < nbytes:
                end = min(cursor + _CHUNK, nbytes)
                count = os.preadv(fd, [mv[cursor:end]], cursor)
                if count <= 0:
                    raise RuntimeError("short read")
                cursor += count
        finally:
            os.close(fd)
        t1 = time.perf_counter_ns()
        digest = hashlib.sha256(mv).hexdigest()
        t2 = time.perf_counter_ns()
        if digest != sha256:
            raise RuntimeError(f"digest mismatch on {path}")
        mv.release()
        return t1 - t0, t2 - t1
    finally:
        buf = None  # drop the buffer before releasing the budget slot
        budget.release(nbytes)


def _peak_rss_mb() -> float:
    # Darwin: ru_maxrss is in bytes (Linux would be KiB).
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _load_meta(artifact: Path):
    manifest = json.loads((artifact / "manifest.json").read_bytes())
    meta = []
    total = 0
    for entry in manifest["layers"]:
        for proj in _PROJECTIONS:
            for field in _FIELDS:
                m = entry["components"][proj][field]
                meta.append((artifact / m["file"], m["bytes"], m["sha256"]))
                total += m["bytes"]
    return meta, total


def _run(meta, budget, workers=None):
    read_ns = hash_ns = 0
    t0 = time.perf_counter()
    if workers is None:
        for path, nb, sha in meta:
            r, h = _read_hash(path, nb, sha, budget)
            read_ns += r
            hash_ns += h
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="f12-bench") as ex:
            futures = [ex.submit(_read_hash, p, nb, sha, budget) for p, nb, sha in meta]
            for f in futures:
                r, h = f.result()
                read_ns += r
                hash_ns += h
    return time.perf_counter() - t0, read_ns / 1e9, hash_ns / 1e9


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="F12 read+hash microbench (CPU/SSD, memory-safe)")
    ap.add_argument("--artifact", default="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/"
                    ".worktrees/deepseek-v41/benchmarks/raw/deepseek-v41-resident-scales/20260917")
    ap.add_argument("--workers", default="4,8,16")
    ap.add_argument("--out", default=None, help="write the table to this file too")
    args = ap.parse_args(argv)

    artifact = Path(args.artifact)
    meta, total = _load_meta(artifact)       # metadata only; no buffers pre-allocated
    gb = total / 1e9
    budget = _ByteBudget(_BUDGET)
    passes = [("serial", None)] + [(f"parallel-{int(w)}", int(w)) for w in args.workers.split(",")]

    rows = []
    header = f"{'pass':<13}{'wall_s':>9}{'GB/s':>9}{'read_agg_s':>12}{'hash_agg_s':>12}{'peakRSS_MB':>12}"
    print(f"artifact={artifact}")
    print(f"files={len(meta)} bytes={total} ({gb:.3f} GB)  budget_MB={_BUDGET // (1024*1024)}")
    print(header, flush=True)

    started = False
    for label, workers in passes:
        if started:
            if not _window_open():
                print("  [fable window reopened mid-series; stopping]", flush=True)
                break
        else:
            _await_window()
        wall, read_s, hash_s = _run(meta, budget, workers)
        started = True
        row = (f"{label:<13}{wall:>9.3f}{gb / wall:>9.2f}{read_s:>12.3f}"
               f"{hash_s:>12.3f}{_peak_rss_mb():>12.1f}")
        rows.append(row)
        print(row, flush=True)

    if args.out:
        Path(args.out).write_text(
            f"artifact={artifact}\nfiles={len(meta)} bytes={total} ({gb:.3f} GB)  "
            f"budget_MB={_BUDGET // (1024*1024)}\n" + header + "\n" + "\n".join(rows) + "\n"
            f"overall_peak_RSS_MB={_peak_rss_mb():.1f}\n")
    print(f"overall_peak_RSS_MB={_peak_rss_mb():.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
