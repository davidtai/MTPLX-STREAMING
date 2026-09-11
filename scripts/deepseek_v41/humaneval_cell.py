#!/usr/bin/env python3
"""One HumanEval(164) pass@1 cell for DeepSeek-V4.1-Flash streaming.

Drives the repo's existing correctness harness against an already-served
OpenAI-compatible endpoint (``humaneval_cell.sh`` owns the serve lifecycle),
scores ONE pass@1 cell at David's sampler, and writes an append-only receipt
that reports **strict pass@1**, **completed-task pass@1** and the **truncation
rate** (memory/eval-truncation-is-not-failure.md).

Harness reuse (do not reimplement message construction or scoring):
  * ``scripts/code_eval_gate.py`` — the SAME server driver the Qwen3.8 125B
    MTPLX PRs used (#475/#478/#482/#485/#488). Its ``build_messages`` is the
    construction to keep: the HumanEval system prompt ("You are a Python
    programming assistant. Complete the function ... single fenced Python code
    block ...") and the user turn ``Complete this function:\n\n```python\n{task
    .prompt}\n``` ``. Generation + sandboxed scoring + pass@k live there and in
    ``mtplx/benchmarks/code_eval.py``. This module builds that driver's argv,
    runs it, then reads its report JSON and derives the two truncation-aware
    metrics it does not itself compute, plus a self-contained per-task pass map
    so an AR-vs-DSpark lane comparison (``humaneval_lane_compare.py``) needs
    nothing but the two receipts.

Sampler (memory/humaneval-one-seed.md, memory/follow-the-specific-setup.md):
David's served sampler is temperature 1, top-p 0.95, top-k 20, seed 20260829
(docs/perf/qwen38-475-battery/README.md) — NOT greedy — one seed, one sample per
task (``n=1``). The served path renders DeepSeek-V4.1's real chat template with
BOS and thinking OFF by default (W52), so the answer is direct and a modest
non-binding output cap suffices: ``--max-tokens`` defaults to 2048 and a bound
cap shows up as a nonzero truncation rate rather than being silently scored as
failures (memory/eval-truncation-is-not-failure.md).

Lanes (W57): there are two served decode lanes to gate. ``--lane ar`` is the
served AR baseline (the profile defaults the byte-identical decode levers: head
bf16, Sinkhorn kernel, attention compile, window memo). ``--lane dspark`` is the
DSpark-DIRECT lane (``mtplx serve --load-mtp --generation-mode dspark --depth
3``). ``humaneval_cell.sh`` sets the serve flags; this module only records the
lane, the depth, the raw serve flags and the daemon's resolved decode-lever env
line into the receipt so a cell is self-describing.

Receipts are APPEND-ONLY (memory/never-overwrite-a-measurement.md): one fresh
UTC-stamped directory per invocation under
``<out-dir>/humaneval_cell/<utc-stamp>/`` (never reused) holding the derived
receipt (named ``humaneval_cell__<lane>__seed<S>__cap<T>__<stamp>.json`` so AR
and DSpark cells never collide), the full ``code_eval_gate`` report JSON, and
the completions sidecar; the writer refuses to overwrite an existing receipt
file.

``--dry-run`` proves argument parsing, the metric derivation, the decode-lever
parse, the receipt path and the append-only guard on CPU from a synthetic
report — no server, no model, no code execution. This module does no GPU/model
work at import; ``--help`` and ``--dry-run`` are CPU-safe.
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
# top-p 0.95, top-k 20, seed 20260829 -- not greedy. Non-binding cap: the served
# chat template runs thinking OFF by default (W52), so 2048 tokens is ample for a
# single fenced solution and a bound cap surfaces as truncation, never a silent
# fail (eval-truncation-is-not-failure.md).
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
DEFAULT_MAX_TOKENS = 2048
DEFAULT_SEED = 20260829
DEFAULT_LANE = "ar"
DEFAULT_DEPTH = 3
_TRUNCATED_FINISH = "length"
# The daemon prints this once at startup (mtplx/server/openai.py, W46):
#   "[4/6] DeepSeek-V4.1 decode levers (resolved env): HEAD_MODE=bf16 ...".
_LEVERS_MARKER = "decode levers (resolved env):"


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
        help="output cap. Default 2048 (non-binding for the thinking-OFF served "
        "chat path, W52); a bound cap surfaces as truncation "
        "(eval-truncation-is-not-failure.md).",
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
        "--lane",
        choices=("ar", "dspark"),
        default=DEFAULT_LANE,
        help="decode lane being gated (W57). Recorded in the receipt and in the "
        "receipt filename so AR and DSpark cells never collide.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_DEPTH,
        help="DSpark draft depth (dspark lane); recorded for provenance.",
    )
    parser.add_argument(
        "--serve-flags",
        default=None,
        help="the lane's raw serve flags, recorded verbatim in the receipt "
        "(e.g. '--load-mtp --generation-mode dspark --depth 3').",
    )
    parser.add_argument(
        "--server-log",
        type=Path,
        default=None,
        help="daemon log to scrape the 'decode levers (resolved env)' line from "
        "(the last matching line is parsed into the receipt).",
    )
    parser.add_argument(
        "--decode-levers-line",
        default=None,
        help="explicit decode-levers log line; overrides --server-log. Mainly "
        "for CPU tests.",
    )
    parser.add_argument(
        "--expert-memory-limit",
        default=None,
        help="the --expert-memory-limit passed to the served daemon, recorded "
        "for provenance (e.g. '60GiB').",
    )
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
    lane = receipt.get("lane") or DEFAULT_LANE
    seed = receipt.get("seed", "NA")
    cap = receipt.get("max_tokens", "NA")
    stamp = receipt.get("utc_compact", "NA")
    return f"{STEP}__{lane}__seed{seed}__cap{cap}__{stamp}.json"


def write_receipt(out_dir: Path, receipt: dict) -> Path:
    path = Path(out_dir) / receipt_filename(receipt)
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing measurement receipt: {path}"
        )
    path.write_text(json.dumps(receipt, indent=2))
    return path


# --------------------------------------------------------------------------
# decode-lever env capture (pure; unit-tested directly)
# --------------------------------------------------------------------------


def parse_decode_levers(line: str | None) -> dict | None:
    """Parse the daemon's 'decode levers (resolved env)' line into a dict.

    The line is ``... decode levers (resolved env): HEAD_MODE=bf16
    SINKHORN_METAL=1 ... PREFILL_LAYER_MAJOR=<unset>``. Any prefix (a log
    timestamp, the ``[4/6]`` stage tag) is ignored; each ``KEY=value`` token
    after the marker becomes an entry, with ``<unset>`` mapped to ``None`` to
    match the daemon's own rendering. Returns ``{"raw", "resolved"}`` or
    ``None`` if the line is empty / carries no ``KEY=value`` tokens.
    """

    if not line:
        return None
    idx = line.find(_LEVERS_MARKER)
    tail = line[idx + len(_LEVERS_MARKER) :] if idx >= 0 else line
    resolved: dict[str, str | None] = {}
    for token in tail.split():
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        if not key:
            continue
        resolved[key] = None if value == "<unset>" else value
    if not resolved:
        return None
    return {"raw": line.strip(), "resolved": resolved}


def _scrape_levers_from_log(log_path: Path) -> str | None:
    """The LAST 'decode levers (resolved env)' line in a daemon log, or None."""

    try:
        text = Path(log_path).read_text(errors="replace")
    except OSError:
        return None
    last: str | None = None
    for line in text.splitlines():
        if _LEVERS_MARKER in line:
            last = line
    return last


def resolve_decode_levers(args) -> dict | None:
    """Resolve the decode-lever env for the receipt: explicit line wins, then log."""

    explicit = getattr(args, "decode_levers_line", None)
    if explicit:
        return parse_decode_levers(explicit)
    log_path = getattr(args, "server_log", None)
    if log_path:
        return parse_decode_levers(_scrape_levers_from_log(Path(log_path)))
    return None


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


def per_task_rows(report: dict) -> list[dict]:
    """A self-contained per-task pass map, so a lane comparison needs only receipts.

    One entry per (task, sample) with ``task_id``, ``passed``, ``finish_reason``
    and ``status`` -- exactly what ``humaneval_lane_compare.py`` diffs between the
    AR and DSpark cells without having to re-open either code_eval_gate report.
    """

    out: list[dict] = []
    for r in report.get("rows") or []:
        out.append(
            {
                "task_id": r.get("task_id"),
                "sample": r.get("sample"),
                "passed": bool(r.get("passed")),
                "finish_reason": r.get("finish_reason"),
                "status": r.get("status"),
            }
        )
    return out


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
        decode_levers=resolve_decode_levers(args),
        per_task=per_task_rows(report),
        gate_rc=rc,
        dry_run=False,
    )
    path = write_receipt(out_dir, receipt)
    _print_summary(path, receipt, metrics)
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
    decode_levers: dict | None,
    per_task: list[dict],
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
        "lane": args.lane,
        "depth": args.depth if args.lane == "dspark" else None,
        "serve_flags": args.serve_flags,
        "expert_memory_limit": args.expert_memory_limit,
        "decode_levers": decode_levers,
        "harness": {
            "driver": "scripts/code_eval_gate.py",
            "scorer": "mtplx/benchmarks/code_eval.py",
            "message_construction": "code_eval_gate.build_messages",
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
        "per_task": per_task,
        "gate_return_code": gate_rc,
        "gate_params": params,
        "gate_report_json": str(report_path) if report_path else None,
        "completions_sidecar": str(completions_path) if completions_path else None,
    }


def _print_summary(path: Path, receipt: dict, metrics: dict) -> None:
    print(f"[humaneval_cell] receipt -> {path}", flush=True)
    levers = receipt.get("decode_levers") or {}
    lever_note = ""
    if levers.get("resolved"):
        lever_note = "  levers=" + ",".join(
            f"{k}={v}" for k, v in levers["resolved"].items()
        )
    print(
        f"[humaneval_cell] lane={receipt.get('lane')} "
        f"strict_pass@1={metrics['strict_pass_at_1']:.4f} "
        f"({metrics['passed']}/{metrics['tasks']})  "
        f"completed_task_pass@1={metrics['completed_task_pass_at_1']:.4f} "
        f"({metrics['completed_tasks']} completed)  "
        f"truncation_rate={metrics['truncation_rate']:.4f} "
        f"({metrics['truncated_tasks']} truncated)"
        f"{lever_note}",
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


# A representative resolved-env line so the DRY receipt shows the lever field's
# shape without a server (dry_run:true marks it synthetic).
_DRY_LEVERS_LINE = (
    "[4/6] DeepSeek-V4.1 decode levers (resolved env): HEAD_MODE=bf16 "
    "SINKHORN_METAL=1 ATTN_COMPILE=1 ATTN_WIN_MEMO=1 SWITCH_FASTPATH=1 "
    "SWITCH_SUBMIT=1 DEVICE_ROUTE=1 HC_COMPILE=1 SHARED_OVERLAP=<unset> "
    "PREFILL_LAYER_MAJOR=<unset>"
)


def run_dry(args) -> int:
    worktree = Path(__file__).resolve().parents[2]
    report = _synthetic_report()
    metrics = compute_cell_metrics(report)
    decode_levers = resolve_decode_levers(args) or parse_decode_levers(_DRY_LEVERS_LINE)
    out_dir = fresh_out_dir(args.out_dir)
    receipt = _assemble_receipt(
        args,
        worktree=worktree,
        metrics=metrics,
        report=report,
        report_path=None,
        completions_path=None,
        decode_levers=decode_levers,
        per_task=per_task_rows(report),
        gate_rc=None,
        dry_run=True,
    )
    path = write_receipt(out_dir, receipt)
    print(f"[humaneval_cell] DRY-RUN receipt -> {path}", flush=True)
    _print_summary(path, receipt, metrics)
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        return run_dry(args)
    return run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
