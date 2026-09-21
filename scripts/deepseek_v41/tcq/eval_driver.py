"""F39 quality-gate driver: full HumanEval(164) + MBPP through the SERVED DeepSeek-V4.1 lane, per expert bank,
at David's sampler, one model load per bank, paired mxfp4-vs-tcq3 on the IDENTICAL code path.

David's verdict rule (memory: humaneval-one-seed / follow-the-specific-setup / eval-truncation-is-not-failure):
the lossy tcq3 bank is judged by full HumanEval(164) + MBPP on his SERVED setup at his sampler
(temperature 1.0, top-p 0.95, top-k 20, seed 20260829, n=1, NOT greedy, no tripwire), paired against the mxfp4 bank
on the same driver. One HE cell per candidate; a bound output cap shows up as a truncation rate, never as failures.

This does NOT reimplement message construction or scoring — it reuses the repo's existing halves:
  * ``mtplx serve``            — the OpenAI-compatible server (a free high port, never :8080; SSD session cache OFF
                                 so the shared prod bank never warms a correctness cell). The AR lane is the served
                                 default; the DSpark-direct lane is ``--load-mtp --generation-mode dspark --depth N``.
  * ``scripts/code_eval_gate.py`` — the driver half (posts prompts, gets completions, hands them to
                                 ``mtplx.benchmarks.code_eval`` for sandboxed pass@1). It owns HumanEval AND MBPP.
This module is the orchestrator: one ``mtplx serve`` per ``--bank``, both suites against it, an append-only receipt
(the code_eval_gate report + completions + derived metrics + timings + full provenance), ``--limit N`` smoke, and
``--resume`` (skip a (bank, suite) whose receipt already exists).  ``--bank mxfp4`` and ``--bank tcq3`` take the SAME
path; only the served model dir and the tcq3 arm flag differ, so the control is a true pair.

tcq3 serving: ``--bank tcq3`` serves the tcq3 artifact with ``MTPLX_DSV41_TCQ3=1`` and the F39 ``tcq`` package on
PYTHONPATH, so the server installs the tcq3 loader + decode at construction (the same manifest/spec seam
``tcq.loader_install`` patches; the served streaming decode of a routed tcq3 expert is
``tcq_runtime.decode_expert_to_bf16`` — decode once per routed expert, then the stock matmul).

CPU-safe: ``--dry-run`` (and ``--help``) build the serve command, the code_eval_gate argv and the receipt paths from
arguments alone — no serve, no model, no network, no code execution. GPU/serving happens ONLY on a real run inside
``gpu_window.sh`` when the lock is free (and, for tcq3, when the real bank is complete).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# ---- David's served sampler (memory: humaneval-one-seed.md / qwen38-475-battery README) ----
SAMPLER = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 20260829, "max_tokens": 2048, "endpoint": "chat"}

DATASETS = {
    "humaneval": "/Users/davidtai/projects/OpenSourceWTF/benchmark-archive/datasets/HumanEval.jsonl",
    "mbpp": "/Users/davidtai/projects/OpenSourceWTF/benchmark-archive/datasets/sanitized-mbpp.json",
}
MODEL_DIRS = {
    "mxfp4": "/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4",
    "tcq3": "/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-tcq3",
}
TCQPKG = "/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f39-tcq3-runtime/scripts/deepseek_v41"
# The mtplx serve entrypoint (confirmed by the serve-path scoping; overridable). Runs the WORKTREE's code.
DEFAULT_SERVE_ENTRY = "mtplx serve"
CODE_EVAL_GATE = "/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/scripts/code_eval_gate.py"


def receipt_dir(out_dir: str, bank: str, suite: str, stamp: str) -> Path:
    """Append-only receipt dir: <out>/eval/<bank>/<suite>/<utc-stamp>/  (never reused; memory never-overwrite)."""
    return Path(out_dir) / "eval" / bank / suite / stamp


def latest_completed(out_dir: str, bank: str, suite: str) -> Path | None:
    """The newest completed receipt for (bank, suite), or None. A receipt is complete when report.json exists."""
    base = Path(out_dir) / "eval" / bank / suite
    if not base.is_dir():
        return None
    done = [d for d in sorted(base.iterdir()) if (d / "report.json").is_file()]
    return done[-1] if done else None


def build_serve_command(*, bank: str, model_dir: str, host: str, port: int, lane: str, depth: int,
                        memory_limit_gib: int | None = None, serve_entry: str = DEFAULT_SERVE_ENTRY):
    """The ``mtplx serve`` argv + the env overrides for a bank.  Never :8080; SSD session cache OFF.

    Returns (argv: list[str], env: dict[str, str]).  mxfp4 and tcq3 differ ONLY in the model dir and the tcq3 env.
    """
    if 8080 == port:
        raise ValueError("refuse to serve a correctness cell on :8080 (shared prod endpoint)")
    argv = list(serve_entry.split()) + [
        "--model", model_dir, "--host", host, "--port", str(port),
        "--no-auth", "--ssd-session-cache", "off",
    ]
    if lane == "dspark":
        argv += ["--load-mtp", "--generation-mode", "dspark", "--depth", str(depth)]
    elif lane != "ar":
        raise ValueError(f"unknown lane {lane!r} (ar|dspark)")
    if memory_limit_gib is not None:
        argv += ["--expert-memory-limit", f"{memory_limit_gib}GiB"]
    env = {}
    if bank == "tcq3":
        env["MTPLX_DSV41_TCQ3"] = "1"                      # server installs the tcq3 loader + decode at construction
        env["GPU_WINDOW_CANDIDATE_MODEL_DIR"] = model_dir  # tcq.install._load_tcq_resources reads this for the routs
        env["PYTHONPATH_APPEND_TCQ"] = TCQPKG              # caller prepends TCQPKG so `import tcq.*` resolves
    elif bank != "mxfp4":
        raise ValueError(f"unknown bank {bank!r} (mxfp4|tcq3)")
    return argv, env


def build_gate_argv(*, suite: str, dataset_path: str, base_url: str, model: str, report_path: str,
                    completions_path: str, limit: int | None, workers: int, sampler: dict = SAMPLER) -> list[str]:
    """The scripts/code_eval_gate.py argv at David's sampler (temp 1, top-p 0.95, top-k 20 via extra-body, seed,
    n=1, chat endpoint, non-binding 2048 cap, sandboxed scoring).  Identical for both banks."""
    argv = [
        "--base-url", base_url, "--model", model,
        "--suite", suite, "--dataset-path", dataset_path,
        "--endpoint", sampler["endpoint"],
        "--temperature", str(sampler["temperature"]), "--top-p", str(sampler["top_p"]),
        "--max-tokens", str(sampler["max_tokens"]), "--seed", str(sampler["seed"]),
        "--n", "1", "--workers", str(workers),
        "--output-json", report_path, "--save-completions", completions_path,
        "--allow-code-execution", "--progress",
    ]
    if sampler.get("top_k"):
        argv += ["--extra-body", f"top_k={int(sampler['top_k'])}"]
    if limit is not None:
        argv += ["--limit", str(limit)]
    return argv


def _derive_metrics(report: dict) -> dict:
    """Strict / completed-task pass@1 + truncation rate (memory eval-truncation-is-not-failure), like humaneval_cell."""
    rows = report.get("rows") or []
    total = len(rows)
    passed = sum(1 for r in rows if r.get("passed"))
    completed = [r for r in rows if r.get("finish_reason") != "length"]
    completed_passed = sum(1 for r in completed if r.get("passed"))
    truncated = sum(1 for r in rows if r.get("finish_reason") == "length")
    return {
        "tasks": total, "passed": passed,
        "strict_pass_at_1": (passed / total) if total else 0.0,
        "completed_tasks": len(completed),
        "completed_task_pass_at_1": (completed_passed / len(completed)) if completed else 0.0,
        "truncated_tasks": truncated, "truncation_rate": (truncated / total) if total else 0.0,
    }


def write_receipt(rdir: Path, *, bank: str, suite: str, lane: str, depth: int, model_dir: str, base_url: str,
                  gate_argv: list[str], serve_argv: list[str], report: dict | None, wall_s: float | None) -> Path:
    """Write the append-only receipt (provenance + derived metrics).  Refuses to overwrite an existing receipt."""
    rdir.mkdir(parents=True, exist_ok=False)
    receipt = {
        "bank": bank, "suite": suite, "lane": lane, "depth": depth, "model_dir": model_dir,
        "base_url": base_url, "sampler": SAMPLER, "serve_command": serve_argv, "gate_argv": gate_argv,
        "wall_s": wall_s,
        "metrics": _derive_metrics(report) if report else None,
        "pass_at_1": (report or {}).get("pass@1"),
        "n_rows": len(((report or {}).get("rows")) or []),
    }
    (rdir / "receipt.json").write_text(json.dumps(receipt, indent=1))
    return rdir / "receipt.json"


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def plan(args) -> list[dict]:
    """The (bank, suite) work plan with resolved commands + receipt dirs — the CPU-testable core (no side effects)."""
    suites = ["humaneval", "mbpp"] if args.suite == "both" else [args.suite]
    stamp = _utc_stamp()
    model_dir = args.tcq3_model if args.bank == "tcq3" else args.mxfp4_model
    base_url = f"http://{args.host}:{args.port}"
    serve_argv, serve_env = build_serve_command(bank=args.bank, model_dir=model_dir, host=args.host, port=args.port,
                                                lane=args.lane, depth=args.depth, serve_entry=args.serve_entry)
    items = []
    for suite in suites:
        completed = latest_completed(args.out_dir, args.bank, suite) if args.resume else None
        rdir = receipt_dir(args.out_dir, args.bank, suite, stamp)
        gate_argv = build_gate_argv(suite=suite, dataset_path=getattr(args, f"{suite}_dataset"),
                                    base_url=base_url, model=args.served_model_name,
                                    report_path=str(rdir / "report.json"),
                                    completions_path=str(rdir / "completions.jsonl"),
                                    limit=args.limit, workers=args.workers)
        items.append({"suite": suite, "receipt_dir": str(rdir), "gate_argv": gate_argv,
                      "skip_resumed": str(completed) if completed else None})
    return {"bank": args.bank, "model_dir": model_dir, "serve_command": serve_argv, "serve_env": serve_env,
            "base_url": base_url, "items": items}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", choices=("mxfp4", "tcq3"), required=True)
    ap.add_argument("--suite", choices=("humaneval", "mbpp", "both"), default="both")
    ap.add_argument("--lane", choices=("ar", "dspark"), default="ar")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--limit", type=int, help="smoke: first N tasks per suite")
    ap.add_argument("--out-dir", default="/Users/davidtai/projects/OpenSourceWTF/reports/dsv41-f39-tcq3-runtime/eval-receipts")
    ap.add_argument("--mxfp4-model", default=MODEL_DIRS["mxfp4"])
    ap.add_argument("--tcq3-model", default=MODEL_DIRS["tcq3"])
    ap.add_argument("--humaneval-dataset", dest="humaneval_dataset", default=DATASETS["humaneval"])
    ap.add_argument("--mbpp-dataset", dest="mbpp_dataset", default=DATASETS["mbpp"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18183, help="free high port; never 8080")
    ap.add_argument("--served-model-name", default="deepseek-v41-flash")
    ap.add_argument("--serve-entry", default=DEFAULT_SERVE_ENTRY)
    ap.add_argument("--workers", type=int, default=1, help="sequential by default (one prompt at a time)")
    ap.add_argument("--resume", action="store_true", help="skip a (bank, suite) whose receipt already exists")
    ap.add_argument("--dry-run", action="store_true", help="CPU-only: print the plan, no serve/model/execution")
    args = ap.parse_args()

    p = plan(args)
    if args.dry_run:
        print(json.dumps(p, indent=1))
        print(f"DRY RUN: bank={p['bank']} suites={[i['suite'] for i in p['items']]} "
              f"serve_env={p['serve_env']} (no serve, no model, no execution)")
        return 0
    raise SystemExit("eval_driver real run must execute inside gpu_window.sh with the GPU lock held; "
                     "serving + scoring is GPU work (use --dry-run for CPU plumbing). "
                     "The guarded runner is scripts/deepseek_v41/tcq/eval_window.sh (pending the free-GPU window).")


if __name__ == "__main__":
    raise SystemExit(main())
