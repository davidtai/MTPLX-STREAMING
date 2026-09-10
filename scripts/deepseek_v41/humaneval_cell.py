#!/usr/bin/env python3
"""One HumanEval(164) pass@1 cell for DeepSeek-V4.1-Flash q2 streaming.

Drives the repo's existing correctness harness against an already-served
OpenAI-compatible endpoint (``humaneval_cell.sh`` owns the serve lifecycle),
scores ONE pass@1 cell at David's sampler, and writes an append-only receipt
that reports **strict pass@1**, **completed-task pass@1** and the **truncation
rate** (memory/eval-truncation-is-not-failure.md).

Harness reuse (do not reimplement scoring):
  * ``scripts/code_eval_gate.py`` — the server driver the Qwen3.8 PRs used
    (``evals/litellm_hy3/run_humaneval.sh`` calls it). Generation + sandboxed
    scoring + pass@k live there and in ``mtplx/benchmarks/code_eval.py``. This
    module builds that driver's argv, runs it, then reads its report JSON and
    derives the two truncation-aware metrics it does not itself compute.

Sampler (memory/humaneval-one-seed.md, memory/follow-the-specific-setup.md):
David's served sampler is temperature 1, top-p 0.95, top-k 20
(docs/perf/qwen38-475-battery/README.md) — NOT greedy — one seed, one sample per
task (``n=1``). The output cap is set so it does NOT bind
(memory/eval-truncation-is-not-failure.md): ``--max-tokens`` defaults to 32768
for the sampled xhigh-thinking cell; a bound cap shows up as a nonzero
truncation rate rather than being silently scored as failures.

Receipts are APPEND-ONLY (memory/never-overwrite-a-measurement.md): one fresh
UTC-stamped directory per invocation under
``<out-dir>/humaneval_cell/<utc-stamp>/`` (never reused) holding the derived
receipt, the full ``code_eval_gate`` report JSON, and the completions sidecar;
the writer refuses to overwrite an existing receipt file.

``--dry-run`` proves argument parsing, the metric derivation, the receipt path
and the append-only guard on CPU from a synthetic report — no server, no model,
no code execution. This module does no GPU/model work at import; ``--help`` and
``--dry-run`` are CPU-safe.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_DATASET = Path(
    "/Users/davidtai/projects/OpenSourceWTF/benchmark-archive/datasets/HumanEval.jsonl"
)
DEFAULT_OUT_DIR = Path(".benchmark-artifacts/deepseek-v41")
STEP = "humaneval_cell"
# David's served sampler (docs/perf/qwen38-475-battery/README.md): temperature 1,
# top-p 0.95, top-k 20 -- not greedy. Non-binding cap (eval-truncation-is-not-
# failure.md: 32768 for the sampled xhigh cell).
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
DEFAULT_MAX_TOKENS = 32768
DEFAULT_SEED = 42
_TRUNCATED_FINISH = "length"


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="served endpoint, e.g. http://127.0.0.1:18183 (no /v1 suffix; the "
        "driver appends the path). Required unless --dry-run.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="served model id (from /v1/models). Required unless --dry-run.",
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--endpoint", choices=("chat", "completions"), default="chat")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="top-k sampling; sent to the server via --extra-body top_k=<k>. "
        "Pass a value <= 0 to omit it (some engines don't read it).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="output cap. Default 32768 so it does NOT bind for the sampled "
        "xhigh-thinking cell (eval-truncation-is-not-failure.md).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="one seed, one sample per task (n=1); humaneval-one-seed.md.",
    )
    parser.add_argument("--limit", type=int, default=None, help="score first N tasks.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--score-workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=1200.0)
    parser.add_argument("--execution-timeout-s", type=float, default=15.0)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--spec-key",
        default=None,
        help="served spec/model key (from /health model_key), for provenance.",
    )
    parser.add_argument("--manifest-sha", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="CPU-only: derive the metrics from a synthetic report and write a "
        "receipt; no server, no model, no code execution.",
    )
    return parser


# --------------------------------------------------------------------------
# provenance + append-only receipt writing
# --------------------------------------------------------------------------


def _git_rev(cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=str(cwd),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def fresh_out_dir(base: Path, step: str = STEP) -> Path:
    """A never-reused UTC-stamped receipt directory ``<base>/<step>/<stamp>/``."""

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    parent = Path(base) / step
    candidate = parent / stamp
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = parent / f"{stamp}-{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def receipt_filename(receipt: dict) -> str:
    seed = receipt.get("seed", "NA")
    cap = receipt.get("max_tokens", "NA")
    stamp = receipt.get("utc_compact", "NA")
    return f"{STEP}__seed{seed}__cap{cap}__{stamp}.json"


def write_receipt(out_dir: Path, receipt: dict) -> Path:
    path = Path(out_dir) / receipt_filename(receipt)
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing measurement receipt: {path}"
        )
    path.write_text(json.dumps(receipt, indent=2))
    return path


# --------------------------------------------------------------------------
# metric derivation (pure; unit-tested directly)
# --------------------------------------------------------------------------


def compute_cell_metrics(report: dict) -> dict:
    """Derive strict / completed-task pass@1 and the truncation rate.

    ``report`` is a ``scripts.code_eval_gate`` report. Its ``rows`` carry, per
    (task, sample), ``passed`` and ``finish_reason``. With one sample per task
    (n=1) pass@1 per task is just ``passed``. A row whose ``finish_reason`` is
    ``"length"`` was truncated by the output cap and is EXCLUDED from the
    completed-task rate (eval-truncation-is-not-failure.md) rather than counted
    as a wrong answer.
    """

    rows = report.get("rows") or []
    total = len(rows)
    passed = sum(1 for r in rows if r.get("passed"))
    truncated_rows = [r for r in rows if r.get("finish_reason") == _TRUNCATED_FINISH]
    truncated = len(truncated_rows)
    completed_rows = [r for r in rows if r.get("finish_reason") != _TRUNCATED_FINISH]
    completed = len(completed_rows)
    completed_passed = sum(1 for r in completed_rows if r.get("passed"))

    by_finish: dict[str, int] = {}
    for r in rows:
        fr = str(r.get("finish_reason"))
        by_finish[fr] = by_finish.get(fr, 0) + 1

    return {
        "tasks": total,
        "passed": passed,
        "strict_pass_at_1": (passed / total) if total else 0.0,
        "completed_tasks": completed,
        "completed_task_pass_at_1": (completed_passed / completed)
        if completed
        else 0.0,
        "truncated_tasks": truncated,
        "truncation_rate": (truncated / total) if total else 0.0,
        "request_errors": sum(
            1 for r in rows if r.get("status") == "request_error"
        ),
        "by_status": report.get("summary", {}).get("by_status"),
        "by_finish_reason": by_finish,
    }


# --------------------------------------------------------------------------
# real run: drive scripts/code_eval_gate.py, then derive
# --------------------------------------------------------------------------


def _import_code_eval_gate(worktree: Path):
    scripts_dir = str(worktree / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import code_eval_gate  # noqa: E402  (top-level script, added to sys.path)

    return code_eval_gate


def _gate_argv(args, report_path: Path, completions_path: Path) -> list[str]:
    argv = [
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--suite",
        "humaneval",
        "--dataset-path",
        str(args.dataset_path),
        "--endpoint",
        args.endpoint,
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-tokens",
        str(args.max_tokens),
        "--seed",
        str(args.seed),
        "--n",
        "1",
        "--workers",
        str(args.workers),
        "--score-workers",
        str(args.score_workers),
        "--retries",
        str(args.retries),
        "--timeout-s",
        str(args.timeout_s),
        "--execution-timeout-s",
        str(args.execution_timeout_s),
        "--output-json",
        str(report_path),
        "--save-completions",
        str(completions_path),
        "--progress",
        "--allow-code-execution",
    ]
    if args.top_k and int(args.top_k) > 0:
        argv += ["--extra-body", f"top_k={int(args.top_k)}"]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    if args.api_key:
        argv += ["--api-key", args.api_key]
    return argv


def run_real(args) -> int:
    if not args.base_url or not args.model:
        print(
            "humaneval_cell: --base-url and --model are required (or use "
            "--dry-run); humaneval_cell.sh fills them from the served endpoint",
            file=sys.stderr,
        )
        return 2
    if not Path(args.dataset_path).is_file():
        print(
            f"humaneval_cell: dataset not found at {args.dataset_path}",
            file=sys.stderr,
        )
        return 2

    worktree = Path(__file__).resolve().parents[2]
    code_eval_gate = _import_code_eval_gate(worktree)

    out_dir = fresh_out_dir(args.out_dir)
    report_path = out_dir / "code_eval_gate_report.json"
    completions_path = out_dir / "completions.jsonl"
    argv = _gate_argv(args, report_path, completions_path)

    print(f"[humaneval_cell] driving code_eval_gate -> {report_path}", flush=True)
    rc = code_eval_gate.main(argv)
    if not report_path.is_file():
        print(
            f"[humaneval_cell] code_eval_gate produced no report (rc={rc}); "
            "not writing a derived receipt",
            file=sys.stderr,
        )
        return rc or 1

    report = json.loads(report_path.read_text())
    metrics = compute_cell_metrics(report)
    receipt = _assemble_receipt(
        args,
        worktree=worktree,
        metrics=metrics,
        report=report,
        report_path=report_path,
        completions_path=completions_path,
        gate_rc=rc,
        dry_run=False,
    )
    path = write_receipt(out_dir, receipt)
    _print_summary(path, metrics)
    # A request error means the cell did not actually score; surface it.
    if metrics["request_errors"]:
        return 1
    return 0


def _assemble_receipt(
    args,
    *,
    worktree: Path,
    metrics: dict,
    report: dict | None,
    report_path: Path | None,
    completions_path: Path | None,
    gate_rc,
    dry_run: bool,
) -> dict:
    now = time.gmtime()
    provenance = (report or {}).get("provenance", {})
    params = (report or {}).get("params", {})
    return {
        "step": STEP,
        "cell": "humaneval-164-pass@1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", now),
        "utc_compact": time.strftime("%Y%m%dT%H%M%SZ", now),
        "git_rev": _git_rev(worktree),
        "dry_run": bool(dry_run),
        "label": args.label,
        "harness": {
            "driver": "scripts/code_eval_gate.py",
            "scorer": "mtplx/benchmarks/code_eval.py",
            "endpoint": args.endpoint,
        },
        "served_model": args.model,
        "base_url": args.base_url,
        "spec_key": args.spec_key,
        "manifest_sha256": args.manifest_sha,
        "sampler": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k if (args.top_k and args.top_k > 0) else None,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "n": 1,
            "greedy": False,
        },
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "dataset_path": str(args.dataset_path),
        "dataset_sha256": provenance.get("dataset_sha256"),
        "limit": args.limit,
        "metrics": metrics,
        "gate_return_code": gate_rc,
        "gate_params": params,
        "gate_report_json": str(report_path) if report_path else None,
        "completions_sidecar": str(completions_path) if completions_path else None,
    }


def _print_summary(path: Path, metrics: dict) -> None:
    print(f"[humaneval_cell] receipt -> {path}", flush=True)
    print(
        "[humaneval_cell] "
        f"strict_pass@1={metrics['strict_pass_at_1']:.4f} "
        f"({metrics['passed']}/{metrics['tasks']})  "
        f"completed_task_pass@1={metrics['completed_task_pass_at_1']:.4f} "
        f"({metrics['completed_tasks']} completed)  "
        f"truncation_rate={metrics['truncation_rate']:.4f} "
        f"({metrics['truncated_tasks']} truncated)",
        flush=True,
    )


# --------------------------------------------------------------------------
# dry run: synthetic report -> metrics -> receipt
# --------------------------------------------------------------------------


def _synthetic_report() -> dict:
    """A tiny code_eval_gate-shaped report: one truncated row, rest completed."""

    rows = []
    for i in range(10):
        finish = _TRUNCATED_FINISH if i == 0 else "stop"
        passed = i in (2, 3, 4, 5, 6, 7)  # completed passers
        rows.append(
            {
                "task_id": f"HumanEval/{i}",
                "sample": 0,
                "status": "passed" if passed else "failed",
                "passed": passed,
                "finish_reason": finish,
            }
        )
    return {
        "rows": rows,
        "summary": {"by_status": {"passed": 6, "failed": 4}},
        "provenance": {"dataset_sha256": "dryrun"},
        "params": {"temperature": DEFAULT_TEMPERATURE},
    }


def run_dry(args) -> int:
    worktree = Path(__file__).resolve().parents[2]
    report = _synthetic_report()
    metrics = compute_cell_metrics(report)
    out_dir = fresh_out_dir(args.out_dir)
    receipt = _assemble_receipt(
        args,
        worktree=worktree,
        metrics=metrics,
        report=report,
        report_path=None,
        completions_path=None,
        gate_rc=None,
        dry_run=True,
    )
    path = write_receipt(out_dir, receipt)
    print(f"[humaneval_cell] DRY-RUN receipt -> {path}", flush=True)
    _print_summary(path, metrics)
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        return run_dry(args)
    return run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
