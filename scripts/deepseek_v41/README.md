# DeepSeek-V4.1-Flash q2 streaming — guarded GPU-window drivers (W5)

Scripts the **orchestrator** launches for the first real-model runs. The harness
author does NOT run any of these (they touch the GPU, the lock, launchctl, and
:8080). `gpu_window.sh` is the only script that takes the exclusive lock and
stops/restores the resident agent; the others are STEPS that run *inside* it.

Paths below assume the W5 worktree:

    WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w5
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

## Launch order (one command per line)

    # 1. P1.7 streamed==resident argmax gate (pin layer 20 resident vs stream it)
    cd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w5 && bash scripts/deepseek_v41/gpu_window.sh env PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w5 /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3 scripts/deepseek_v41/gate_stream_equals_resident.py --pinned-layers 20 --out-dir /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w5/.benchmark-artifacts/deepseek-v41

    # 2. serve + health smoke check (free high port, never :8080)
    cd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w5 && bash scripts/deepseek_v41/gpu_window.sh bash scripts/deepseek_v41/serve_health.sh

Rotate the gate across layers for full 40-layer coverage (one window each):
`--pinned-layers 1,14` (engram layers), `--pinned-layers 2,8` (kv_source),
`--pinned-layers 20` (candidate/index), etc. — a subset per run because the full
158 GiB bank cannot be pinned in 100 GiB.

## What each receipt proves

**Gate receipt** —
`.benchmark-artifacts/deepseek-v41/<utc-stamp>/gate_stream_equals_resident.json`
(append-only; one new stamped dir per run, never overwritten):

- `verdict` = `PASS` iff the 32-step greedy argmax token sequence is byte-identical
  whether the pinned subset's experts were served **resident** (Run B islands) or
  **streamed** from the Q2 bank (Run A). That is the P1.7 streamed==resident
  correctness gate — the streaming path does not change what the model decodes.
- `runs.streamed.argmax_tokens` / `runs.resident.argmax_tokens` — the two token
  sequences that were compared.
- `runs.*.gathered_records` — the digest (manifest SHA-256) of every routed
  expert record actually gathered in each run, keyed by `(step, layer, expert)`.
  Because only a subset can be pinned per run, these prove the *same* experts were
  exercised on both sides even where residency differed. Union the pinned subsets
  across rotated runs for total 40-layer coverage.
- `spec_key`, `manifest_sha256`, `memory_limit_bytes`, `expert_cache_limit_bytes`,
  `git_rev`, `pinned_layers_run_b` — provenance to bind the verdict to an exact
  artifact + tree.

**Serve health** — stdout of `serve_health.sh` plus its server log
(`$TMPDIR/dsv41-serve-health/serve-<stamp>.log`). It proves the shipped artifact
serves: `/health` comes up on a free high port, the reported `generation_mode`
(expected `ar` for a streamed artifact) and `model_key`, the served model id from
`/v1/models`, one short chat completion with a token count + tok/s (server-reported
and wall-clock), and a clean stop. No JSON receipt file; the printed lines are the
evidence. It never binds :8080.

## Preconditions

- Worker **W1**'s `mtplx/models/deepseek_v41.py` (`Model`, `ModelArgs`) must exist,
  or the loader raises `ResidentLoadError`; both drivers then report that cleanly
  and exit non-zero.
- The local `expert-manifest.json` must carry the HF identity so admission passes
  (W3 rebased it locally). The gate accepts `--no-admit --admission-receipt <file>`
  if admission must be injected.

## Environment knobs (gpu_window.sh)

- `GPU_WINDOW_MIN_FREED_GB` (default 60) — free-memory rise required after
  bootout to confirm the resident agent released (it frees ~86 GB; the gate sits
  below that to tolerate vm_stat noise). Raise it to match the actual agent.
- `GPU_WINDOW_CHILD_RSS_CAP_BYTES` (default 100 GiB) — memory guard; the step
  child is killed and the agent restored if it exceeds this.
- `GPU_WINDOW_STOP_TIMEOUT` (180), `GPU_WINDOW_RESTORE_TIMEOUT` (300),
  `GPU_WINDOW_LOCK_TIMEOUT` (0 = block forever), `GPU_WINDOW_QWEN_LABEL`
  (com.tea.qwen), `GPU_WINDOW_QWEN_PLIST` (override plist discovery),
  `MTPLX_GPU_LOCK` (/tmp/mtplx-gpu-exclusive.lock).
