"""Tiny microbench: serial vs parallel Engram miss reads on the REAL bank file.

Measures exactly what F6 parallelizes -- the per-``gather_bytes`` miss-read chain:
sort the row ids, coalesce contiguous runs, read each run via the production
``FileRowReader.read_run`` (F_NOCACHE ``preadv``). Serial does it inline; parallel
fans the runs across a pool (the same ``ParallelReadState.fetch`` the install
uses). No LRU/arena/dequant work -- just the I/O that stalls the main thread.

CPU/IO only, no Metal. Keep it tiny: a measured GPU window may be running and it
reads the same bank. Default 144 rows/call, 50 calls, configs serial/8/16/32.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import mlx.core as mx

mx.set_default_device(mx.cpu)  # importing FileRowReader pulls in mlx; no Metal here

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engram_parallel as ep  # noqa: E402
from mtplx.ngram_row_cache import FileRowReader, _contiguous_runs  # noqa: E402


def _requests(rows, slot_count):
    misses = sorted(set(int(r) for r in rows))
    reqs = []
    for start, count in _contiguous_runs(misses):
        off = 0
        while off < count:
            n = min(count - off, slot_count)
            reqs.append((start + off, n))
            off += n
    return reqs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--rows", type=int, default=384006168, help="num_rows in the bank")
    ap.add_argument("--row-bytes", type=int, default=264)
    ap.add_argument("--rows-per-call", type=int, default=144)
    ap.add_argument("--calls", type=int, default=50)
    ap.add_argument("--workers", type=int, nargs="*", default=[8, 16, 32])
    ap.add_argument("--seed", type=int, default=20260919)
    args = ap.parse_args(argv)

    reader = FileRowReader(args.bank, row_bytes=args.row_bytes, num_rows=args.rows)
    slot_count = 67108864 // args.row_bytes  # the bounded 64 MiB budget
    print(f"bank={args.bank} rows={args.rows} row_bytes={args.row_bytes} "
          f"io={reader.io_cache_mode} rows/call={args.rows_per_call} calls={args.calls}")

    rng = np.random.default_rng(args.seed)
    # fresh random rows per call (decode misses are ~always new n-grams)
    batches = [rng.integers(0, args.rows, size=args.rows_per_call).tolist() for _ in range(args.calls)]
    reqs_per_call = [_requests(b, slot_count) for b in batches]

    def bench(fn):
        # one warm call (page-in the fd path), then timed
        fn(reqs_per_call[0])
        per = []
        for reqs in reqs_per_call:
            t0 = time.perf_counter()
            fn(reqs)
            per.append((time.perf_counter() - t0) * 1e3)
        return per

    def serial(reqs):
        for s, c in reqs:
            reader.read_run(s, c)

    rows = []
    per = bench(serial)
    rows.append(("serial", per, 1))
    for w in args.workers:
        state = ep.ParallelReadState(workers=w)
        try:
            per = bench(lambda reqs, st=state: st.fetch(reader, reqs))
            rows.append((f"pool-{w}", per, state.stats["max_inflight"]))
        finally:
            state.shutdown()

    print(f"{'config':>10} {'mean_ms':>9} {'p50_ms':>8} {'min_ms':>8} {'max_inflight':>13}")
    base = None
    out = []
    for name, per, mif in rows:
        mean = statistics.mean(per)
        p50 = statistics.median(per)
        mn = min(per)
        if base is None:
            base = mean
        print(f"{name:>10} {mean:9.3f} {p50:8.3f} {mn:8.3f} {mif:13d}   ({base/mean:.2f}x)")
        out.append({"config": name, "mean_ms": mean, "p50_ms": p50, "min_ms": mn,
                    "max_inflight": mif, "speedup_vs_serial": base / mean})
    reader.close()
    print("JSON " + json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
