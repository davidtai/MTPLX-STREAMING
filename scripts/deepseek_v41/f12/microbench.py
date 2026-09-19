"""F12 microbench: serial vs parallel read+hash on the REAL packed-scale artifact.

CPU only, no MLX. Buffers are plain ``bytearray`` (production writes into the mx
buffer; here we isolate the file open / preadv / sha256 cost that F12 moves off the
main thread). Uses the EXACT production algorithm: O_RDONLY|O_NOFOLLOW, F_NOCACHE=1,
F_RDAHEAD 45->0, 8 MiB chunked preadv, sha256 compare.

Reads ~3.086 GB from the same SSD that GPU benchmark windows measure, so before
EACH pass it waits for /tmp/mtplx-gpu-exclusive.lock to be free (poll 60 s, up to
40 min) and never runs while the lock is held. Run it under ``nice -n 19``.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_CHUNK = 8 * 1024 ** 2
_F_RDAHEAD = 45
_LOCK = "/tmp/mtplx-gpu-exclusive.lock"
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_FIELDS = ("descriptors", "payload", "bases")


def _lock_held() -> bool:
    r = subprocess.run(["lsof", _LOCK], capture_output=True, text=True)
    return bool(r.stdout.strip())


def _wait_for_lock_free(poll: float = 60.0, timeout: float = 40 * 60) -> None:
    deadline = time.time() + timeout
    while _lock_held():
        if time.time() > deadline:
            raise SystemExit("GPU exclusive lock held longer than timeout; refusing SSD bench")
        print(f"  [lock held; waiting {poll:.0f}s ...]", flush=True)
        time.sleep(poll)


def _read_hash(path, mv, nbytes, sha256):
    """Production per-file body; returns (read_ns, hash_ns)."""
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
    return t1 - t0, t2 - t1


def _load_files(artifact: Path):
    manifest = json.loads((artifact / "manifest.json").read_bytes())
    files = []
    total = 0
    for entry in manifest["layers"]:
        for proj in _PROJECTIONS:
            for field in _FIELDS:
                m = entry["components"][proj][field]
                buf = bytearray(m["bytes"])
                files.append((artifact / m["file"], memoryview(buf), m["bytes"], m["sha256"]))
                total += m["bytes"]
    return files, total


def _run_serial(files):
    read_ns = hash_ns = 0
    t0 = time.perf_counter()
    for path, mv, nb, sha in files:
        r, h = _read_hash(path, mv, nb, sha)
        read_ns += r
        hash_ns += h
    return time.perf_counter() - t0, read_ns / 1e9, hash_ns / 1e9


def _run_parallel(files, workers):
    t0 = time.perf_counter()
    read_ns = hash_ns = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="f12-bench") as ex:
        futures = [ex.submit(_read_hash, p, mv, nb, sha) for p, mv, nb, sha in files]
        for f in futures:
            r, h = f.result()
            read_ns += r
            hash_ns += h
    return time.perf_counter() - t0, read_ns / 1e9, hash_ns / 1e9


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="F12 read+hash microbench (CPU/SSD)")
    ap.add_argument("--artifact", default="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/"
                    ".worktrees/deepseek-v41/benchmarks/raw/deepseek-v41-resident-scales/20260917")
    ap.add_argument("--workers", default="4,8,16")
    ap.add_argument("--out", default=None, help="write the table to this file too")
    args = ap.parse_args(argv)

    artifact = Path(args.artifact)
    files, total = _load_files(artifact)
    gb = total / 1e9
    worker_counts = [int(w) for w in args.workers.split(",")]

    rows = []
    header = f"{'pass':<13}{'wall_s':>9}{'GB/s':>9}{'read_agg_s':>13}{'hash_agg_s':>13}"

    def record(label, wall, read_s, hash_s):
        rows.append(f"{label:<13}{wall:>9.3f}{gb / wall:>9.2f}{read_s:>13.3f}{hash_s:>13.3f}")
        print(rows[-1], flush=True)

    print(f"artifact={artifact}")
    print(f"files={len(files)} bytes={total} ({gb:.3f} GB)")
    print(header, flush=True)

    print("[lock check before serial]", flush=True)
    _wait_for_lock_free()
    record("serial", *_run_serial(files))
    for nw in worker_counts:
        print(f"[lock check before parallel-{nw}]", flush=True)
        _wait_for_lock_free()
        record(f"parallel-{nw}", *_run_parallel(files, nw))

    if args.out:
        Path(args.out).write_text(
            f"artifact={artifact}\nfiles={len(files)} bytes={total} ({gb:.3f} GB)\n"
            + header + "\n" + "\n".join(rows) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
