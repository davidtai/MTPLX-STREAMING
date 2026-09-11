#!/usr/bin/env python3
"""W24 A/B harness for DeepSeek-V4.1-Flash decode-time streaming levers.

The orchestrator runs this INSIDE ``scripts/deepseek_v41/gpu_window.sh`` (holding
the GPU flock).  One lever per arm, controlled by explicit runtime switches that
default OFF (so the ``control`` arm is the shipped path, byte-for-byte).  Each arm
runs the standardized ``mtplx.prefill_bench`` prompt (default 1,024 context
tokens, BOS) + ``--decode-tokens`` (default 256) greedy tokens and records, into an
append-only receipt:

  * decode tok/s (and prefill tok/s, TTFT);
  * bytes read off SSD (from the expert-runtime IO counters);
  * expert cache hit rate;
  * peak GB;
  * the decoded token ids (sha) so the control-vs-candidate byte-identity check is
    part of the receipt, not a claim.

Arms are named ``<lever>=<value>``; ``control`` is every lever OFF.  See
``docs/deepseek-v41/W24_REPORT.md`` for the ranked lever list and the exact
commands.  No GPU work at import; ``--help`` is safe on CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MODEL = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4").expanduser()
GIB = 1024 ** 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--context-tokens", type=int, default=1024, choices=(1024, 16384))
    p.add_argument("--decode-tokens", type=int, default=256)
    p.add_argument("--arms", nargs="+", default=["control"],
                   help="lever arms to run, e.g. control io_fanout=6 cache_policy=freqdecay")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prompt-format", default="raw", choices=("raw", "chat"))
    p.add_argument("--bos-id", type=int, default=0)
    p.add_argument("--no-bos", dest="bos", action="store_false", default=True)
    p.add_argument("--slot-layout", default="component-banks")
    p.add_argument("--memory-limit-gib", type=float, default=100.0)
    p.add_argument("--expert-cache-limit-gib", type=float, default=None)
    p.add_argument("--apply-memory-cap", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cpu", action="store_true", default=False)
    p.add_argument("--max-kv", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    raise SystemExit("W24 ab_decode_levers: not yet implemented (skeleton)")


if __name__ == "__main__":
    raise SystemExit(main())
