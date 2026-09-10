# DeepSeek-V4.1-Flash q2 streaming — guarded GPU-window drivers

Scripts the **orchestrator** launches for the real-model runs. The harness
author does NOT run any of these (they touch the GPU, the lock, launchctl, and
:8080). `gpu_window.sh` is the only script that takes the exclusive lock and
stops/restores the resident agent; the others are STEPS that run *inside* it.

The whole GPU phase below is staged and ready; it launches the moment the
faithful port passes its **CPU probe** (the streamed prefill still had a defect
at W8 — see `docs/deepseek-v41/W8_REPORT.md` — so this is gated, not assumed).

Paths below assume the **integration** worktree:

    WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
    PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3

## Launch rules (hard)

- Launch each `gpu_window.sh` command as the DIRECT command of a
  `run_in_background: true` Bash call. NEVER a shell `&` one-liner (a reaped
  process group leaves the resident agent DOWN).
  See memory/guarded-window-launch-protocol.md.
- Do NOT stop the launching agent mid-window: it kills `gpu_window.sh` AND its
  lock holder, and the resident agent is left down (restore never runs).
- Only one window at a time. `gpu_window.sh` blocks on the lock and prints a
  queue notice; it never signals whoever holds the lock
  (memory/never-signal-flock-queue.md).
- Run each command from `$WT` so `.benchmark-artifacts/...` receipts land in the
  worktree, or pass an absolute `--out-dir`.
- The **CPU probe (step 0) does NOT run inside `gpu_window.sh`**: CPU-heavy work
  inside a held GPU window wastes it and, on a shared box, would void it
  (memory/cpu-heavy-work-voids-flock-windows.md). It is the *precondition* for
  opening the window, not a window step.

## Launch order (one command per line)

    # 0. CPU probe (PRECONDITION -- no lock, resident agent stays UP; the port
    #    workers' deliverable). The faithful port must produce non-degenerate,
    #    golden-matching output on CPU at the standard 1,024 + BOS input before
    #    any GPU is spent. Representative: the per-layer hidden-state dump on CPU
    #    at 1,024 + BOS vs the W9 torch goldens.
    cd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41 && env PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41 /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3 scripts/deepseek_v41/dump_hidden_states.py --cpu --no-apply-memory-cap --context-tokens 1024 --bos --out /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41/.benchmark-artifacts/deepseek-v41/probe/cpu_1024_bos.json

    # 1. P1.7 streamed==resident argmax gate at the standard input (1,024 + BOS)
    cd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41 && bash scripts/deepseek_v41/gpu_window.sh env PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41 /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3 scripts/deepseek_v41/gate_stream_equals_resident.py --context-tokens 1024 --bos --pinned-layers 20 --out-dir /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41/.benchmark-artifacts/deepseek-v41

    # 2. Standard-shape decode benchmark, both cells (1,024 and 16,384), greedy
    cd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41 && bash scripts/deepseek_v41/gpu_window.sh env PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41 /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3 scripts/deepseek_v41/bench_standard_shape.py --context-tokens 1024 16384 --steps 256 --out-dir /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41/.benchmark-artifacts/deepseek-v41

    # 3. serve + health smoke check (free high port, never :8080)
    cd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41 && bash scripts/deepseek_v41/gpu_window.sh bash scripts/deepseek_v41/serve_health.sh

    # 4. one HumanEval(164) pass@1 cell at David's sampler (served, free high port)
    cd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41 && bash scripts/deepseek_v41/gpu_window.sh bash scripts/deepseek_v41/humaneval_cell.sh

`serve_health.sh` and `humaneval_cell.sh` derive their worktree from the script
path and set `PYTHONPATH`/`cwd` to it themselves (editable-install engagement
guard), so they take no `env` prefix.

Rotate the gate across layers for full 40-layer coverage (one window each):
`--pinned-layers 1,14` (engram layers), `--pinned-layers 2,8` (kv_source),
`--pinned-layers 20` (candidate/index), etc. — a subset per run because the full
158 GiB bank cannot be pinned in 100 GiB.

The benchmark scripts each take `--dry-run`: a CPU-only test double (no model, no
Metal, no server) that proves argument parsing, the prompt build, the receipt
paths and the append-only guard. `humaneval_cell.sh --dry-run` prints the plan
and drives `humaneval_cell.py --dry-run` with no serve and no code execution.
`tests/test_deepseek_v41_bench_scripts.py` covers all of it on CPU.

## What each receipt proves

All receipts are **append-only** (memory/never-overwrite-a-measurement.md): one
fresh UTC-stamped directory per invocation under
`.benchmark-artifacts/deepseek-v41/<step>/<utc-stamp>/`, never reused, and each
writer refuses to overwrite an existing receipt file. Every receipt carries the
git rev, spec key, manifest SHA-256, and the prompt-build metadata.

**Gate receipt** —
`.benchmark-artifacts/deepseek-v41/<utc-stamp>/gate_stream_equals_resident.json`:

- `verdict` = `PASS` iff the greedy argmax token sequence is byte-identical
  whether the pinned subset's experts were served **resident** (Run B islands)
  or **streamed** from the Q2 bank (Run A) — the P1.7 streamed==resident gate.
- `runs.*.argmax_tokens`, `runs.*.gathered_records` (manifest SHA per routed
  `(step, layer, expert)`), `spec_key`, `manifest_sha256`, `pinned_layers_run_b`,
  `git_rev` — provenance binding the verdict to an exact artifact + tree.
- Built at the standard input via `--context-tokens 1024 --bos` (same
  `build_prompt` the bench and the dump use).

**Standard-shape bench** —
`.benchmark-artifacts/deepseek-v41/bench_standard_shape/<utc-stamp>/`
`bench_standard_shape__ctx<cells>__steps<N>__seed<S>__<stamp>.json`:

- One entry in `cells[]` per prefill cell (1,024 and 16,384 by default), each
  with `prompt_build` (the shared prefill_bench + BOS metadata — `prompt_source`,
  `input_tokens`, `bos_prepended`, ...) and a `repeats[]` list. Per repeat:
  `prefill_tok_s`, `ttft_s`, `decode_tok_s`, `peak_mlx_gb` + `process_rss_gb`
  (peak GB), `wall_s`, `expert_records_gathered`, `engram_rows_gathered`, and the
  first 200 chars of the decoded text. `fastest_of` gives the max decode tok/s /
  min TTFT with the range (memory/report-fastest-of-seeds.md).
- Greedy, single prompt, no batching (memory/dsv41-standard-benchmark-shape.md,
  follow-the-specific-setup.md). Three seeds are not needed for a greedy run;
  `--repeats` is for later speed windows. Expert/engram counts are read off the
  runtime's own counters *between* cells, never by wrapping the decode path (that
  would inflate the tok/s this bench measures).

**Serve health** — stdout of `serve_health.sh` plus its server log
(`$TMPDIR/dsv41-serve-health/serve-<stamp>.log`): `/health` on a free high port,
`generation_mode` (expected `ar`) + `model_key`, the `/v1/models` id, one short
chat completion with tok/s, and a clean stop. It never binds :8080.

**HumanEval cell** —
`.benchmark-artifacts/deepseek-v41/humaneval_cell/<utc-stamp>/`
`humaneval_cell__seed<S>__cap<T>__<stamp>.json`, plus the full
`code_eval_gate_report.json` and `completions.jsonl` sidecar:

- ONE HumanEval(164) pass@1 cell (memory/humaneval-one-seed.md) driven against
  `mtplx serve` on a free high port, reusing the repo's existing harness —
  `scripts/code_eval_gate.py` (the driver the Qwen3.8 PRs used) over
  `mtplx/benchmarks/code_eval.py` (scoring/pass@k). It is NOT reimplemented here.
- David's sampler (docs/perf/qwen38-475-battery/README.md): temperature 1,
  top-p 0.95, top-k 20 — not greedy — one seed, `n=1`. The output cap defaults to
  32768 so it does NOT bind (memory/eval-truncation-is-not-failure.md).
- `metrics`: `strict_pass_at_1`, `completed_task_pass_at_1` (excludes rows whose
  `finish_reason == "length"`), and `truncation_rate` — the three the memory rule
  requires. `sampler`, `dataset_sha256`, `spec_key` (served `model_key`),
  `git_rev` for provenance. The SSD SessionBank cold tier is served OFF so the
  shared prod bank never warms the cell (memory/ssd-session-bank-warms-benchmarks).

## Preconditions

- Worker **W1**'s `mtplx/models/deepseek_v41.py` (`Model`, `ModelArgs`) must
  exist, or the loader raises `ResidentLoadError`; the drivers report that
  cleanly and exit non-zero.
- The **faithful port must pass its CPU probe (step 0)** — a non-degenerate,
  golden-matching CPU forward at 1,024 + BOS — before the GPU phase is worth
  running (W8 found a streamed-prefill defect; root-causing is W9/W10).
- The local `expert-manifest.json` must carry the HF identity so admission
  passes (W3 rebased it). The gate/bench accept
  `--no-admit --admission-receipt <file>` if admission must be injected.

## Environment knobs (gpu_window.sh)

- `GPU_WINDOW_MIN_AVAIL_GB` (default 100) — memory that must be available after
  the resident-agent stop before the step runs.
- `GPU_WINDOW_CHILD_RSS_CAP_BYTES` (default 100 GiB) — the step child is killed
  and the agent restored if it exceeds this.
- `GPU_WINDOW_STOP_TIMEOUT` (180), `GPU_WINDOW_RESTORE_TIMEOUT` (300),
  `GPU_WINDOW_LOCK_TIMEOUT` (0 = block forever), `GPU_WINDOW_QWEN_LABEL`
  (com.tea.qwen), `GPU_WINDOW_QWEN_PLIST` (override plist discovery),
  `MTPLX_GPU_LOCK` (/tmp/mtplx-gpu-exclusive.lock).

Bench/HumanEval knobs: `bench_standard_shape.py --context-tokens/--steps/--repeats
/--max-kv/--memory-limit-gib`; `humaneval_cell.sh` reads `HUMANEVAL_DATASET`
(default the canonical `benchmark-archive/datasets/HumanEval.jsonl`, 164 tasks),
`HUMANEVAL_LIMIT` (empty = all 164), `DSV41_OUT_DIR`, and the `serve_health.sh`
serve knobs (`DSV41_MODEL`, `DSV41_HOST`, `DSV41_HEALTH_TIMEOUT`).
