"""Print the F16/F18 stamp summary written by :mod:`f16.stamps`.

    python -m f16.stamp_readout <stem>

Reads ``<stem>.summary.json`` (or recomputes it from ``<stem>.raw.json.gz`` when the
summary is absent) and prints, per role and overall, the sum / mean / p50 / p90 of each
derived interval (encode, parked_at_barrier, gpu_wait, host_pre, hit_submit,
parked_at_reads, miss_wait, post, build), plus the per-forward total and the
across-forward gap.  Pure host; never imports MLX or touches the GPU.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from . import stamps as _stamps


def _load_summary(stem: str) -> dict:
    summary_path = Path(stem + ".summary.json")
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    raw_path = Path(stem + ".raw.json.gz")
    if not raw_path.exists():
        raise SystemExit(f"stamp_readout: no {summary_path} or {raw_path}")
    with gzip.open(raw_path, "rt") as fh:
        raw = json.load(fh)
    return _stamps.summarize(raw.get("records", []))


def _fmt(stat: dict) -> str:
    if not stat or stat.get("n", 0) == 0:
        return "n=0"

    def us(v):
        return "-" if v is None else f"{v / 1000.0:.2f}us"

    return (f"n={stat['n']:>5} sum={us(stat['sum']):>12} "
            f"mean={us(stat['mean']):>10} p50={us(stat['p50']):>10} "
            f"p90={us(stat['p90']):>10}")


def _print_block(title: str, block: dict, interval_names) -> None:
    print(f"\n[{title}] records={block.get('n_records', 0)}")
    for name in interval_names:
        stat = block.get(name)
        if stat is not None:
            print(f"  {name:<18} {_fmt(stat)}")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        raise SystemExit("usage: python -m f16.stamp_readout <stem>")
    stem = argv[0]
    summary = _load_summary(stem)
    interval_names = summary.get("interval_names", [])
    print(f"F16_STAMP_READOUT stem={stem} records={summary.get('n_records', 0)} "
          f"forwards={summary.get('n_forwards', 0)} ({summary.get('units', '')})")
    for role_name, block in summary.get("per_role", {}).items():
        _print_block(f"role={role_name}", block, interval_names)
    _print_block("overall", summary.get("overall", {}), interval_names)
    print(f"\n  total_per_forward   {_fmt(summary.get('total_per_forward_ns', {}))}")
    print(f"  across_forward_gap  {_fmt(summary.get('across_forward_gap_ns', {}))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
