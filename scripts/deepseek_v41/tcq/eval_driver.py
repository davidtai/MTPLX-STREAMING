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
# The F39 worktree root holds the seam-edited mtplx (mtplx/models/expert_mlx.py) AND the tcq package.  It is
# prepended to the served process's PYTHONPATH AHEAD of the editable install so BOTH banks import THIS mtplx
# (memory: editable-install shadowing has silently run the wrong mtplx before -- eval_driver asserts mtplx.__file__).
MTPLX_WORKTREE = os.path.dirname(os.path.dirname(TCQPKG))  # .../.worktrees/dsv41-f39-tcq3-runtime
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
        env["MTPLX_DSV41_TCQ3"] = "1"                      # the serve site hook installs the tcq3 loader + decode
        env["GPU_WINDOW_CANDIDATE_MODEL_DIR"] = model_dir  # candidate model dir (also read by tcq.install helpers)
        env["PYTHONPATH_APPEND_TCQ"] = TCQPKG              # so `import tcq.*` resolves in the served process
        env["PYTHONPATH_APPEND_TCQ_SITE"] = os.path.join(TCQPKG, "tcq", "tcq_serve_site")  # sitecustomize auto-arm
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
                  gate_argv: list[str], serve_argv: list[str], report: dict | None, wall_s: float | None,
                  mtplx_file: str | None = None, git_rev: str | None = None) -> Path:
    """Write the append-only receipt (provenance + derived metrics).  Refuses to overwrite an existing receipt."""
    rdir.mkdir(parents=True, exist_ok=True)
    if (rdir / "receipt.json").exists():
        raise FileExistsError(f"receipt exists (append-only; never overwrite a measurement): {rdir / 'receipt.json'}")
    receipt = {
        "bank": bank, "suite": suite, "lane": lane, "depth": depth, "model_dir": model_dir,
        "base_url": base_url, "sampler": SAMPLER, "serve_command": serve_argv, "gate_argv": gate_argv,
        "served_mtplx_file": mtplx_file, "mtplx_git_rev": git_rev, "wall_s": wall_s,
        "metrics": _derive_metrics(report) if report else None,
        "pass_at_1": (report or {}).get("pass@1"),
        "n_rows": len(((report or {}).get("rows")) or []),
    }
    (rdir / "receipt.json").write_text(json.dumps(receipt, indent=1))
    return rdir / "receipt.json"


def serve_pythonpath(mtplx_worktree: str, serve_env: dict) -> list:
    """The served process's PYTHONPATH order: MY worktree root FIRST (its edited mtplx wins over the editable
    install), then the tcq package + serve site hook (tcq3), then the inherited PYTHONPATH.  Same for both banks."""
    pp = [mtplx_worktree]
    for k in ("PYTHONPATH_APPEND_TCQ", "PYTHONPATH_APPEND_TCQ_SITE"):
        if serve_env.get(k):
            pp.append(serve_env[k])
    return pp


def mtplx_under_worktree(mtplx_file: str, mtplx_worktree: str) -> bool:
    """True iff a served ``mtplx.__file__`` resolves under ``<worktree>/mtplx`` (the editable-shadowing guard)."""
    if not mtplx_file:
        return False
    root = os.path.realpath(os.path.join(mtplx_worktree, "mtplx"))
    return os.path.realpath(mtplx_file).startswith(root + os.sep)


def _served_mtplx_file(venv_python: str, pythonpath: list, cwd: str) -> str:
    """Resolve the mtplx the served process WILL import, under the exact PYTHONPATH (a CPU preflight; no model)."""
    import subprocess
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(pythonpath + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    out = subprocess.run([venv_python, "-c", "import mtplx, sys; sys.stdout.write(mtplx.__file__)"],
                         env=env, cwd=cwd, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"could not import mtplx under the serve PYTHONPATH: {out.stderr.strip()[:400]}")
    return out.stdout.strip()


def _git_rev(worktree: str) -> str:
    import subprocess
    try:
        return subprocess.run(["git", "-C", worktree, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


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
            "base_url": base_url, "items": items, "mtplx_worktree": args.mtplx_worktree,
            "serve_pythonpath": serve_pythonpath(args.mtplx_worktree, serve_env)}


def _wait_health(base_url: str, *, timeout_s: float) -> None:
    """Poll <base_url>/health until 200 or timeout (real run only; needs the served endpoint)."""
    import urllib.request
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(2)
    raise RuntimeError(f"serve /health not ready within {timeout_s}s ({last})")


def run(args) -> int:
    """Real run (GPU work; execute ONLY inside gpu_window.sh with the lock held): serve the bank ONCE, run both
    suites through code_eval_gate against it, write append-only receipts, stop the server.  Not CPU-testable
    (needs the served endpoint); the CPU-tested pieces are plan()/build_*()/write_receipt()."""
    import subprocess
    p = plan(args)
    _pp_keys = ("PYTHONPATH_APPEND_TCQ", "PYTHONPATH_APPEND_TCQ_SITE")
    clean_pp = serve_pythonpath(args.mtplx_worktree, p["serve_env"])   # my worktree's mtplx FIRST, then tcq/site
    serve_env = dict(os.environ)
    serve_env.update({k: v for k, v in p["serve_env"].items() if k not in _pp_keys})
    full_pp = clean_pp + ([serve_env["PYTHONPATH"]] if serve_env.get("PYTHONPATH") else [])
    serve_env["PYTHONPATH"] = ":".join(full_pp)
    # editable-install-shadowing guard: the served process MUST import THIS worktree's (seam-edited) mtplx.
    mtplx_file = _served_mtplx_file(args.venv_python, clean_pp, args.worktree)
    git_rev = _git_rev(args.mtplx_worktree)
    if not mtplx_under_worktree(mtplx_file, args.mtplx_worktree):
        raise SystemExit(f"REFUSE: served mtplx resolves to {mtplx_file}, not under {args.mtplx_worktree}/mtplx "
                         "(editable-install shadowing). The seam edit / tcq3 decode would not be the served code.")
    print(f"[eval_driver] served mtplx = {mtplx_file} (git {git_rev[:12]}); serving {p['bank']} "
          f"({p['model_dir']}): {' '.join(p['serve_command'])}", flush=True)
    proc = subprocess.Popen(p["serve_command"], env=serve_env, cwd=args.worktree)
    try:
        _wait_health(p["base_url"], timeout_s=args.serve_timeout_s)
        for item in p["items"]:
            if item["skip_resumed"]:
                print(f"[eval_driver] resume: skip {p['bank']}/{item['suite']} ({item['skip_resumed']})", flush=True)
                continue
            rdir = Path(item["receipt_dir"])
            rdir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            rc = subprocess.run([args.venv_python, CODE_EVAL_GATE, *item["gate_argv"]],
                                env=serve_env, cwd=args.worktree).returncode
            wall = time.time() - t0
            rep_path = rdir / "report.json"
            report = json.loads(rep_path.read_text()) if rep_path.is_file() else None
            write_receipt(rdir, bank=p["bank"], suite=item["suite"], lane=args.lane, depth=args.depth,
                          model_dir=p["model_dir"], base_url=p["base_url"], gate_argv=item["gate_argv"],
                          serve_argv=p["serve_command"], report=report, wall_s=wall,
                          mtplx_file=mtplx_file, git_rev=git_rev)
            print(f"[eval_driver] {p['bank']}/{item['suite']}: gate rc={rc} wall={wall:.1f}s -> {rdir}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except Exception:  # noqa: BLE001
            proc.kill()
    return 0


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
    ap.add_argument("--worktree", default="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-run-d5f15e7a",
                    help="run worktree = cwd for gpu_window.sh")
    ap.add_argument("--mtplx-worktree", default=MTPLX_WORKTREE,
                    help="worktree whose (seam-edited) mtplx must be the served one; prepended to PYTHONPATH + asserted")
    ap.add_argument("--venv-python", default="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python")
    ap.add_argument("--serve-timeout-s", type=float, default=1800.0, help="wait for /health (a large model load)")
    ap.add_argument("--resume", action="store_true", help="skip a (bank, suite) whose receipt already exists")
    ap.add_argument("--dry-run", action="store_true", help="CPU-only: print the plan, no serve/model/execution")
    args = ap.parse_args()

    if args.dry_run:
        p = plan(args)
        print(json.dumps(p, indent=1))
        print(f"DRY RUN: bank={p['bank']} suites={[i['suite'] for i in p['items']]} "
              f"serve_env={p['serve_env']} (no serve, no model, no execution)")
        return 0
    if os.environ.get("EVAL_DRIVER_ALLOW_RUN") != "1":
        raise SystemExit("eval_driver real run is GPU work (serve + generate): launch it INSIDE gpu_window.sh "
                         "with the GPU lock held and set EVAL_DRIVER_ALLOW_RUN=1 to confirm. Use --dry-run for the "
                         "CPU plumbing.")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
