#!/usr/bin/env python3
"""W24 routing-locality census for DeepSeek-V4.1-Flash streaming (CPU).

Runs the standardized ``mtplx.prefill_bench`` programming prompt (default 1,024
context tokens, BOS prepended) through the real streaming loader path on the MLX
CPU device, then greedily decodes ``--decode-tokens`` (default 64) tokens.  A
non-invasive switch wrapper records, for every (token, layer), the routed expert
ids selected by the MoE gate.  From that trace it computes the locality census
that decides which decode-time cache/IO levers are worth building:

  * unique routed experts per layer over the decode window;
  * reuse-distance distribution (per assignment);
  * per-layer LRU hit rate at 115 / 205 / 384 slots (configurable);
  * theoretical SSD bytes/token at each slot budget (record size from manifest);
  * Belady/oracle best-achievable hit rate at each budget;
  * a held-out trained-quota gate (frequency slot allocation vs uniform) so the
    per-layer frequency-allocation lever is only claimed if DSV4.1 shows real
    cross-layer concentration variance (hy3-q4's did NOT -- see
    mmap-willneed-unwired.md 2026-07-21 refutation).
  * an MTP verify-window dedup projection (union of consecutive-token expert sets
    at widths 1..D) for the cost model's MTP contribution.

MEMORY / CONCURRENCY CONTRACT (David 2026-09-11, six workers share a 100 GB box;
a guard kills any worker python above 14 GB):

  * CPU only (``mx.set_default_device(mx.cpu)`` at import).
  * The single real-artifact load runs ONLY while holding
    ``/tmp/dsv41-cpu-model-load.lock`` (LOCK_EX) -- this module takes the lock
    itself (``--no-self-lock`` to defer to an outer wrapper).
  * ``memory_limit`` <= 12 GiB, expert cache <= 2 GiB, text-only residents,
    ``apply_memory_cap`` on; an in-process RSS watchdog aborts at ``--rss-abort-gib``
    (default 12).
  * Writes the census JSON incrementally to ``--out`` (append-only receipt).

No GPU work at import; ``--help`` is CPU-safe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MODEL = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4").expanduser()
GIB = 1024 ** 3
DEFAULT_SLOT_BUDGETS = (115, 205, 384)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--context-tokens", type=int, default=1024, choices=(1024, 16384))
    p.add_argument("--decode-tokens", type=int, default=64)
    p.add_argument("--prompt-format", default="raw", choices=("raw", "chat"))
    p.add_argument("--bos-id", type=int, default=0)
    p.add_argument("--no-bos", dest="bos", action="store_false", default=True)
    p.add_argument("--slot-budgets", type=int, nargs="+", default=list(DEFAULT_SLOT_BUDGETS))
    p.add_argument("--mtp-widths", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--slot-layout", default="component-banks")
    p.add_argument("--memory-limit-gib", type=float, default=12.0)
    p.add_argument("--expert-cache-limit-gib", type=float, default=2.0)
    p.add_argument("--max-kv", type=int, default=4096)
    p.add_argument("--rss-abort-gib", type=float, default=12.0)
    p.add_argument("--self-lock", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lock-path", type=Path, default=Path("/tmp/dsv41-cpu-model-load.lock"))
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    raise SystemExit("W24 routing_census: not yet implemented (skeleton)")


if __name__ == "__main__":
    raise SystemExit(main())
