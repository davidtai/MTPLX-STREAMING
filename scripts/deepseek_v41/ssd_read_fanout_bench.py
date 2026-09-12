#!/usr/bin/env python3
"""W109 (T1 / issue I1): SSD realized-bandwidth vs read fanout microbench.

Measures the realized GB/s of reading fixed-size expert records at random file
offsets as a function of the number of CONCURRENT reads in flight (the "fanout"),
using the SAME read primitive as the streamed decode runtime: positional
``os.preadv`` (or the native ``pread_exact_into`` backend when the extension is
importable) into a caller-owned buffer, with the page cache bypassed via
``F_NOCACHE`` -- see ``mtplx/expert_io.py`` (``os.preadv`` at the Python path,
``native_read_into`` at the native path, ``fcntl(F_NOCACHE)`` on the fd). This is
the DSpark verify demand-read path; the ``MTPLX_DSV41_VERIFY_IO_FANOUT`` lever
raises exactly this cross-record concurrency, so this bench quantifies the
realized-bandwidth-vs-queue-depth curve the lever climbs.

Reads only; never writes to the target file. Prints a JSON report.

    # validate on a small temp file (no model, no real bank) -- SAFE any time:
    python3 ssd_read_fanout_bench.py --self-test

    # the orchestrator runs this against the real expert bank IN THE LOCK GAP:
    python3 ssd_read_fanout_bench.py <bank_file> --record-bytes 18800000 \
        --num-offsets 128 --fanouts 1,2,4,8,16 --repeats 3

The same random offset set is reused for every fanout level (seeded) so the sweep
is a fair A/B; ``F_NOCACHE`` means a reused offset still hits the drive, not the
page cache. The fastest of ``--repeats`` runs is reported per fanout level.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def _load_native_reader():
    """The runtime's optional native pread backend, or None."""
    try:
        from mtplx_native_expert_io import pread_exact_into

        return pread_exact_into
    except Exception:
        return None


def _read_record(fd: int, offset: int, buffer: bytearray, native) -> int:
    """Fill ``buffer`` from ``[offset, offset+len)`` -- the runtime's primitive.

    Loops on partial reads exactly as ``_read_range_into_seq`` does. Returns bytes
    read. Raises on a short read (a record that ran off the end of the file).
    """
    view = memoryview(buffer)
    want = len(view)
    got = 0
    while got < want:
        target = view[got:]
        if native is not None:
            n = int(native(fd, offset + got, target))
        else:
            n = int(os.preadv(fd, [target], offset + got))
        if n <= 0:
            raise IOError(
                f"short read at offset {offset + got}: wanted {want - got} bytes"
            )
        got += n
    return got


def _run_fanout(
    fd: int,
    offsets: list[int],
    record_bytes: int,
    fanout: int,
    native,
) -> tuple[float, int]:
    """Read every offset once, with ``fanout`` reads concurrently in flight.

    Returns (wall_seconds, bytes_read). Each worker owns its own buffer so no two
    concurrent reads share a destination. ``os.preadv`` with an explicit offset is
    thread-safe on a shared fd (see ``expert_io.py``)."""
    if fanout <= 1:
        buffer = bytearray(record_bytes)
        start = time.perf_counter()
        total = 0
        for offset in offsets:
            total += _read_record(fd, offset, buffer, native)
        return time.perf_counter() - start, total

    # A per-WORKER-THREAD buffer (thread-local): each pool thread runs one task at a
    # time, so its buffer is never touched by a concurrent read -- no data race, and
    # peak memory is bounded at fanout * record_bytes (not num_offsets * record_bytes).
    local = threading.local()

    def task(offset: int) -> int:
        buffer = getattr(local, "buffer", None)
        if buffer is None:
            buffer = bytearray(record_bytes)
            local.buffer = buffer
        return _read_record(fd, offset, buffer, native)

    start = time.perf_counter()
    total = 0
    with ThreadPoolExecutor(max_workers=fanout) as pool:
        for got in pool.map(task, offsets):
            total += got
    return time.perf_counter() - start, total


def _measure(
    path: str,
    *,
    record_bytes: int,
    num_offsets: int,
    fanouts: list[int],
    seed: int,
    repeats: int,
    nocache: bool,
    backend: str,
) -> dict:
    file_size = os.path.getsize(path)
    if file_size < record_bytes:
        raise ValueError(
            f"file {path} is {file_size} bytes; smaller than record_bytes "
            f"{record_bytes}"
        )
    max_offset = file_size - record_bytes
    rng = random.Random(seed)
    # A fixed, seeded offset set, reused for every fanout level and repeat so the
    # sweep is a fair comparison. Offsets are byte positions (not record-aligned);
    # the real bank's records ARE aligned, but the drive does not care and this
    # keeps the bench model-agnostic.
    offsets = [rng.randint(0, max_offset) for _ in range(num_offsets)]

    native = _load_native_reader()
    if backend == "preadv":
        native = None
    elif backend == "native" and native is None:
        raise RuntimeError("native backend requested but mtplx_native_expert_io "
                           "is not importable")
    backend_name = "native" if native is not None else "python-preadv"

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    cache_mode = "buffered"
    try:
        if nocache:
            try:
                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                cache_mode = "f-nocache"
            except (OSError, AttributeError):
                cache_mode = "buffered(nocache-unavailable)"
        results: dict[str, dict] = {}
        for fanout in fanouts:
            best_wall = None
            best_bytes = 0
            for _ in range(max(1, repeats)):
                wall, total = _run_fanout(
                    fd, offsets, record_bytes, fanout, native
                )
                if best_wall is None or wall < best_wall:
                    best_wall = wall
                    best_bytes = total
            gb = best_bytes / 1e9
            gib = best_bytes / (1024 ** 3)
            results[str(fanout)] = {
                "wall_s": best_wall,
                "bytes": best_bytes,
                "records": num_offsets,
                "gb_per_s": (gb / best_wall) if best_wall else None,
                "gib_per_s": (gib / best_wall) if best_wall else None,
                "records_per_s": (num_offsets / best_wall) if best_wall else None,
            }
    finally:
        os.close(fd)

    best_fanout = max(
        results,
        key=lambda k: (results[k]["gb_per_s"] or 0.0),
    )
    return {
        "path": os.path.abspath(path),
        "file_size": file_size,
        "record_bytes": record_bytes,
        "num_offsets": num_offsets,
        "fanouts": fanouts,
        "seed": seed,
        "repeats": repeats,
        "backend": backend_name,
        "cache_mode": cache_mode,
        "fanout": results,
        "best_fanout": int(best_fanout),
        "peak_gb_per_s": results[best_fanout]["gb_per_s"],
        "peak_gib_per_s": results[best_fanout]["gib_per_s"],
    }


def _parse_fanouts(text: str) -> list[int]:
    out = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value < 1:
            raise argparse.ArgumentTypeError("fanout must be >= 1")
        out.append(value)
    if not out:
        raise argparse.ArgumentTypeError("no fanouts given")
    return out


def _self_test() -> dict:
    """Build a small temp file and run a tiny sweep. No model, no real bank."""
    record_bytes = 1 << 16  # 64 KiB records
    num_records = 64
    with tempfile.NamedTemporaryFile(
        prefix="mtplx-ssd-fanout-selftest-", suffix=".bin", delete=False
    ) as handle:
        temp_path = handle.name
        handle.write(os.urandom(record_bytes * num_records + 4096))
    try:
        report = _measure(
            temp_path,
            record_bytes=record_bytes,
            num_offsets=48,
            fanouts=[1, 2, 4, 8, 16],
            seed=1234,
            repeats=2,
            nocache=True,
            backend="auto",
        )
    finally:
        os.unlink(temp_path)
    report["self_test"] = True
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="file to read (never written)")
    parser.add_argument(
        "--record-bytes", type=int, default=18_800_000,
        help="bytes per record read (DSV4.1 expert record ~= 18.80 MB)",
    )
    parser.add_argument("--num-offsets", type=int, default=128)
    parser.add_argument(
        "--fanouts", type=_parse_fanouts, default=[1, 2, 4, 8, 16],
        help="comma-separated concurrency levels, e.g. 1,2,4,8,16",
    )
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--no-nocache", dest="nocache", action="store_false",
        help="do NOT set F_NOCACHE (default: bypass the page cache, as the runtime does)",
    )
    parser.add_argument(
        "--backend", choices=("auto", "preadv", "native"), default="auto",
    )
    parser.add_argument("--output", help="write JSON here (default: stdout)")
    parser.add_argument(
        "--self-test", action="store_true",
        help="build a small temp file, run a tiny sweep, and exit (SAFE any time)",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        report = _self_test()
    else:
        if not args.path:
            parser.error("a file path is required (or use --self-test)")
        report = _measure(
            args.path,
            record_bytes=args.record_bytes,
            num_offsets=args.num_offsets,
            fanouts=args.fanouts,
            seed=args.seed,
            repeats=args.repeats,
            nocache=args.nocache,
            backend=args.backend,
        )

    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
