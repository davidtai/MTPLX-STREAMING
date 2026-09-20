"""F19 read order: serve demand plane reads in SUBMISSION order with a small fanout pool.

The retained packed lane issues every plane of every in-flight miss part to the reader's fanout pool
(15 workers at io_read_fanout=4).  The SSD saturates at 3 concurrent 5.9 MB plane reads
(measured 2026-09-20: 9.8 / 13.7 / 14.0 / 14.0 GB/s at 1 / 2 / 3 / 4+), so the extra workers buy no
throughput; they only make every in-flight plane share the device, which delays the OLDEST read --
the one the generation thread waits on next (two-group verify pipeline: the partner's older reads;
any lane: the first miss part of a route, whose early gate/up compute could already run).

This module swaps ``reader._fanout_executor`` for a pool of ``workers`` threads.  ThreadPoolExecutor
queues FIFO, so with a pool no larger than the device's saturation point, planes complete in the
order they were submitted.  Bytes, offsets, destinations and hashing are untouched: same reads,
different service order.

Install point: a staged line in packed_phase.py immediately BEFORE ``plane_lane.install`` (its
``bind_reader`` captures ``reader._fanout_executor.submit`` once), at the quiescent post-prefill
growth boundary, so prefill I/O keeps the retained pool.  No MLX; CPU-testable.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

ENV = "MTPLX_DSV41_F19_FANOUT_WORKERS"


def install(reader, *, workers: int) -> dict:
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 64:
        raise RuntimeError(f"F19 fanout workers must be an integer in [1, 64]; got {workers!r}")
    old = getattr(reader, "_fanout_executor", None)
    if old is None:
        raise RuntimeError("F19 needs the reader's fanout executor (io_read_fanout > 1)")
    before = getattr(reader, "_fanout_pool_workers", None)
    new = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mtplx-io-fanout-f19")
    reader._fanout_executor = new            # reader.close() shuts down whichever pool is installed
    reader._fanout_pool_workers = workers
    old.shutdown(wait=True)                  # quiescent boundary: drains nothing in practice
    return {"installed": True, "workers_before": before, "workers": workers}


def install_from_env(reader) -> dict:
    raw = os.environ.get(ENV)
    if not raw:
        return {"installed": False}
    report = install(reader, workers=int(raw))
    print("F19_READ_ORDER_INSTALL " + json.dumps(report, sort_keys=True), flush=True)
    return report
