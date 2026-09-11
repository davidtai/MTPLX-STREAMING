#!/usr/bin/env python3
"""Paired AR-vs-DSpark HumanEval(164) lane comparison for DeepSeek-V4.1-Flash.

The two served decode lanes (W57) — AR and DSpark-DIRECT
(``--load-mtp --generation-mode dspark --depth 3``) — are each gated by ONE
HumanEval(164) pass@1 cell (``humaneval_cell.py``). DSpark's greedy path is
supposed to reproduce AR token-for-token, but David's served sampler is
temperature 1 (not greedy), so the two lanes *will* draw different completions
and a handful of tasks flip pass<->fail between them purely from sampling. This
tool pairs the two cells so those tie-flip-class divergences can be inspected
one task at a time instead of being buried in a single aggregate delta.

Input is the two append-only receipts written by ``humaneval_cell.py`` (one per
lane). Each receipt carries a self-contained ``per_task`` pass map, so this needs
nothing else — no re-serve, no re-open of the underlying code_eval_gate reports.

Output (JSON + a human summary):
  * each lane's strict pass@1, completed-task pass@1, truncation count;
  * ``strict_pass_at_1`` delta (DSpark - AR);
  * the per-task diff list: every task the two lanes DISAGREE on, split into
    ``ar_only_pass`` (AR passed, DSpark failed) and ``dspark_only_pass``, each
    entry carrying both lanes' ``finish_reason`` so a divergence caused by a
    truncated DSpark draft reads differently from a genuine content flip;
  * ``both_pass`` / ``both_fail`` counts and the agreement rate over shared
    tasks; and any task present in only one receipt.

Stdlib only; no MLX, no network, no model. Pure and CPU-safe: ``compare_lanes``
is directly unit-tested.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _per_task_map(receipt: dict) -> dict[str, dict]:
    """task_id -> {passed, finish_reason, status} from a receipt's per_task list.

    Faithful to a one-sample cell (n=1: one row per task). If a receipt ever
    carries >1 sample per task, ``passed`` is OR-ed across the samples (a task
    passes if any sample passed) and the first row's finish_reason is kept.
    """

    out: dict[str, dict] = {}
    for row in receipt.get("per_task") or []:
        task_id = row.get("task_id")
        if task_id is None:
            continue
        if task_id in out:
            out[task_id]["passed"] = out[task_id]["passed"] or bool(row.get("passed"))
        else:
            out[task_id] = {
                "passed": bool(row.get("passed")),
                "finish_reason": row.get("finish_reason"),
                "status": row.get("status"),
            }
    return out


def _lane_summary(receipt: dict) -> dict:
    metrics = receipt.get("metrics") or {}
    return {
        "lane": receipt.get("lane"),
        "serve_flags": receipt.get("serve_flags"),
        "dry_run": bool(receipt.get("dry_run")),
        "git_rev": receipt.get("git_rev"),
        "seed": receipt.get("seed"),
        "max_tokens": receipt.get("max_tokens"),
        "tasks": metrics.get("tasks"),
        "passed": metrics.get("passed"),
        "strict_pass_at_1": metrics.get("strict_pass_at_1"),
        "completed_task_pass_at_1": metrics.get("completed_task_pass_at_1"),
        "truncated_tasks": metrics.get("truncated_tasks"),
        "decode_levers": (receipt.get("decode_levers") or {}).get("resolved"),
    }


def compare_lanes(ar_receipt: dict, dspark_receipt: dict) -> dict:
    """Pair two lane receipts into a comparison summary (pure)."""

    ar_map = _per_task_map(ar_receipt)
    ds_map = _per_task_map(dspark_receipt)
    shared = sorted(set(ar_map) & set(ds_map))
    ar_only_tasks = sorted(set(ar_map) - set(ds_map))
    ds_only_tasks = sorted(set(ds_map) - set(ar_map))

    both_pass = 0
    both_fail = 0
    ar_only_pass: list[dict] = []
    dspark_only_pass: list[dict] = []
    for task_id in shared:
        ar_p = ar_map[task_id]["passed"]
        ds_p = ds_map[task_id]["passed"]
        if ar_p and ds_p:
            both_pass += 1
        elif not ar_p and not ds_p:
            both_fail += 1
        elif ar_p and not ds_p:
            ar_only_pass.append(_diff_entry(task_id, ar_map[task_id], ds_map[task_id]))
        else:
            dspark_only_pass.append(
                _diff_entry(task_id, ar_map[task_id], ds_map[task_id])
            )

    disagreements = len(ar_only_pass) + len(dspark_only_pass)
    agreement = ((both_pass + both_fail) / len(shared)) if shared else None
    ar_rate = (ar_receipt.get("metrics") or {}).get("strict_pass_at_1")
    ds_rate = (dspark_receipt.get("metrics") or {}).get("strict_pass_at_1")
    delta = (ds_rate - ar_rate) if (ar_rate is not None and ds_rate is not None) else None

    return {
        "schema": "mtplx.dsv41_humaneval_lane_compare/1",
        "ar": _lane_summary(ar_receipt),
        "dspark": _lane_summary(dspark_receipt),
        "delta_strict_pass_at_1": delta,
        "comparison": {
            "shared_tasks": len(shared),
            "both_pass": both_pass,
            "both_fail": both_fail,
            "disagreements": disagreements,
            "agreement_rate": agreement,
            "ar_only_pass": ar_only_pass,
            "dspark_only_pass": dspark_only_pass,
            "ar_only_tasks": ar_only_tasks,
            "dspark_only_tasks": ds_only_tasks,
        },
    }


def _diff_entry(task_id: str, ar_row: dict, ds_row: dict) -> dict:
    return {
        "task_id": task_id,
        "ar": {"passed": ar_row["passed"], "finish_reason": ar_row.get("finish_reason")},
        "dspark": {
            "passed": ds_row["passed"],
            "finish_reason": ds_row.get("finish_reason"),
        },
    }


def _load_receipt(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _format_summary(summary: dict) -> str:
    ar = summary["ar"]
    ds = summary["dspark"]
    comp = summary["comparison"]
    lines = [
        "DeepSeek-V4.1-Flash HumanEval(164) — AR vs DSpark paired lane comparison",
        f"  AR      lane={ar['lane']}  strict_pass@1={_fmt(ar['strict_pass_at_1'])} "
        f"({ar['passed']}/{ar['tasks']})  "
        f"completed_task_pass@1={_fmt(ar['completed_task_pass_at_1'])}  "
        f"truncated={ar['truncated_tasks']}",
        f"  DSpark  lane={ds['lane']}  strict_pass@1={_fmt(ds['strict_pass_at_1'])} "
        f"({ds['passed']}/{ds['tasks']})  "
        f"completed_task_pass@1={_fmt(ds['completed_task_pass_at_1'])}  "
        f"truncated={ds['truncated_tasks']}",
        f"  delta strict_pass@1 (DSpark - AR) = {_fmt(summary['delta_strict_pass_at_1'])}",
        f"  shared={comp['shared_tasks']} both_pass={comp['both_pass']} "
        f"both_fail={comp['both_fail']} disagreements={comp['disagreements']} "
        f"agreement_rate={_fmt(comp['agreement_rate'])}",
    ]
    if comp["ar_only_pass"]:
        lines.append("  AR passed, DSpark failed (tie-flip class):")
        for e in comp["ar_only_pass"]:
            lines.append(
                f"    {e['task_id']}  ar.finish={e['ar']['finish_reason']} "
                f"dspark.finish={e['dspark']['finish_reason']}"
            )
    if comp["dspark_only_pass"]:
        lines.append("  DSpark passed, AR failed (tie-flip class):")
        for e in comp["dspark_only_pass"]:
            lines.append(
                f"    {e['task_id']}  ar.finish={e['ar']['finish_reason']} "
                f"dspark.finish={e['dspark']['finish_reason']}"
            )
    if comp["ar_only_tasks"] or comp["dspark_only_tasks"]:
        lines.append(
            f"  NOTE unmatched tasks: only-in-AR={len(comp['ar_only_tasks'])} "
            f"only-in-DSpark={len(comp['dspark_only_tasks'])} "
            "(the two cells did not score the same task set)"
        )
    if ar["dry_run"] or ds["dry_run"]:
        lines.append("  WARNING: a receipt is dry_run=true (synthetic; not a real cell)")
    return "\n".join(lines)


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ar", required=True, type=Path, help="the AR-lane humaneval_cell receipt."
    )
    parser.add_argument(
        "--dspark",
        required=True,
        type=Path,
        help="the DSpark-lane humaneval_cell receipt.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the comparison summary JSON here (append-only: refuses to "
        "overwrite an existing file).",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.ar, args.dspark):
        if not Path(path).is_file():
            print(f"humaneval_lane_compare: receipt not found: {path}", file=sys.stderr)
            return 2
    ar_receipt = _load_receipt(args.ar)
    dspark_receipt = _load_receipt(args.dspark)
    summary = compare_lanes(ar_receipt, dspark_receipt)
    summary["inputs"] = {"ar": str(args.ar), "dspark": str(args.dspark)}

    if args.out is not None:
        out_path = Path(args.out)
        if out_path.exists():
            print(
                f"humaneval_lane_compare: refusing to overwrite {out_path} "
                "(append-only)",
                file=sys.stderr,
            )
            return 2
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"[humaneval_lane_compare] summary -> {out_path}", flush=True)

    print(_format_summary(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
