# W5 — Guarded GPU-window drivers for the first DeepSeek-V4.1-Flash q2 runs

Branch `feat/deepseek-v41-w5`. Scripts only — the harness author executed **no**
GPU/MLX/launchctl/lock code; all runtime execution is the orchestrator's, later.
Deliverables in `scripts/deepseek_v41/`:

| File | Role |
|---|---|
| `gpu_window.sh` | Serialized exclusive-GPU window: lock → verify wired knob → bootout resident agent → run one step → restore. |
| `gate_stream_equals_resident.py` | P1.7 streamed==resident argmax gate against the W3 loader. |
| `serve_health.sh` | Serve the artifact on a free high port (never :8080), /health smoke check, one chat completion, clean stop. |
| `README.md` | Exact orchestrator launch order + what each receipt proves. |

The drivers follow the repo's established guarded-window shape rather than a new
one; where the reference machinery is Python (`mtplx.qwen_guard`,
`scripts/run_with_qwen_stopped.py`) the shell drivers reuse the same lock file,
the same `bootout`/`bootstrap` verbs, and the same "hold the lock, run the child,
restore before releasing" ordering.

---

## 1. `gpu_window.sh`

Runs one step (its argv) inside an exclusive-GPU window and restores the resident
agent (`com.tea.qwen` on :8080) on **every** exit path (success, failure,
timeout, memory-guard kill, Ctrl-C) via an `EXIT/INT/TERM` trap.

Phases, each printing a UTC-timestamped `[gpu_window]` line:

0. **Lock (fcntl advisory).** macOS ships no `flock(1)` (`which flock` → not
   found; the repo's own shell scripts only *read* the lock via `lsof`, e.g.
   `tools/mixofficial_governor.sh:19,29`). So the wrapper re-execs itself under a
   small inline Python holder that takes `fcntl.flock(LOCK_EX|LOCK_NB)` in a poll
   loop on `/tmp/mtplx-gpu-exclusive.lock` — the **same lock and same primitive**
   `mtplx/qwen_guard.py:27` (`DEFAULT_MLX_LOCK_PATH`) and `:408`
   (`fcntl.flock(lock.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)`) use. Blocking wait with
   a printed queue notice; it **never** SIGSTOP/SIGKILLs the holder
   (memory/never-signal-flock-queue.md). The holder keeps the fd open for the
   whole window and releases it only after the child restores the agent and exits
   (mirrors `scripts/run_with_qwen_stopped.py`'s hold-lock→run-child→restore
   flow).
1. **Wired-memory knob.** Reads `sysctl -n iogpu.wired_limit_mb` and requires it
   in `(0, 102400]` MB (≤ 100 GiB). READ ONLY — never writes the sysctl, never
   raises it (memory/never-exceed-the-memory-knob.md; the 112 GiB panic-#3 cause
   in qwen-serve-crash-loop-guards.md). Fails closed on `0` (no limit) or missing.
2. **Discover plist + pid.** `launchctl print gui/<uid>/com.tea.qwen`; parses the
   `path = …` (plist) and `pid = …` lines. Overridable via `GPU_WINDOW_QWEN_PLIST`.
   If the agent is not loaded at entry, it is left stopped on exit (as found).
3. **Bootout + confirm release.** `launchctl bootout gui/<uid>/com.tea.qwen` —
   `bootout`, **not** `kickstart -k` (kickstart races the port yield and leaves it
   down; memory/guarded-window-launch-protocol.md). Borrowed verb:
   `mtplx/qwen_guard.py:1136` (bootout). Polls until the captured pid is gone
   **and** available memory (vm_stat free+inactive+speculative+purgeable) has
   risen by `GPU_WINDOW_MIN_FREED_GB` (default 60; the agent frees ~86 GB). On
   timeout it fails loudly with no hidden retries.
4. **Run the step under an RSS memory guard.** Backgrounds the step, tracks peak
   RSS via `ps -o rss=,state=` (state check reaps zombies cleanly), and if RSS
   exceeds `GPU_WINDOW_CHILD_RSS_CAP_BYTES` (default 100 GiB) kills the child and
   restores — the external memory guard hy3-benchmark-panic-protocol.md mandates.
5. **Restore + release (trap).** `launchctl bootstrap gui/<uid> <plist>` (borrowed
   verb: `mtplx/qwen_guard.py:1078`), polls for the service to reappear, then the
   Python holder releases the lock. Agent is restored **before** the lock frees,
   so a queued window never acquires mid-reload.

Borrows: lock path + primitive `mtplx/qwen_guard.py:27,408`; bootout/bootstrap
`mtplx/qwen_guard.py:1136,1078`; read-only-lock-no-signal rule
`tools/mixofficial_governor.sh:19,29`; hold-lock/run-child/restore ordering
`scripts/run_with_qwen_stopped.py` (whole file). Generalizes past `qwen_guard`'s
Qwen3.6-hardcoded pattern by discovering the plist from `launchctl print` (the
current :8080 agent is not Qwen3.6), and adds the knob check, free-memory poll,
and child-RSS guard that `qwen_guard` does not carry.

## 2. `gate_stream_equals_resident.py`

The P1.7 streamed==resident argmax gate, adapted to what the W3 loader exposes.
Builds the model **twice** through
`mtplx.models.deepseek_v41_loader.load_deepseek_v41_streaming`:

- **Run A (streamed):** `slot_layout=component-banks`, `island_layers=()` — every
  routed layer streams its experts from `experts.bin`.
- **Run B (resident):** `slot_layout=component-banks`,
  `island_layers=<--pinned-layers>` — that subset is held resident
  (`DenseIslandSwitchGLU`, bound by `mtplx/models/expert_mlx.py:2607`); all other
  layers still stream. Both runs use the **same** slot layout, so residency is the
  only variable.

Each run greedily decodes `--steps` (default 32) tokens from a fixed prompt; the
gate **PASSES** iff the two argmax sequences are byte-identical.

**Subset, not all 40 layers (task-mandated + documented in the docstring):**
pinning every routed layer is impossible in 100 GiB — the bank is ~158 GiB
(169,869,312,000 routed bytes; ~4.15 GiB/layer resident → ~20 layers max). So the
gate pins a subset per run and additionally records **the digest of every routed
expert record actually gathered** (manifest SHA-256 of each selected
`(layer, expert)`), captured by wrapping each bound `switch_mlp` — both
`HotExpertSwitchGLU` and `DenseIslandSwitchGLU` are called `switch_mlp(x, indices)`
(`mtplx/models/expert_mlx.py:1701` and the island dispatch), so selections are
captured for pinned and streamed layers alike. Rotating `--pinned-layers` across
runs covers all 40 layers; the digests prove the same experts were exercised
regardless of where they were served.

Receipt: append-only `.benchmark-artifacts/deepseek-v41/<utc-stamp>/gate_stream_equals_resident.json`
(a fresh stamped dir per run via `_out_dir`, never overwritten —
memory/never-overwrite-a-measurement.md) carrying `git_rev`, `spec_key`
(`runtime.spec.key`), `manifest_sha256` (`runtime.manifest.manifest_sha256`),
`memory_limit_bytes` / `expert_cache_limit_bytes` (`runtime.config`), both
`argmax_tokens`, `match`/`verdict`, `pinned_layers_run_b`, and per-run
`gathered_records`.

Loader/runtime API used (all confirmed to import on CPU): `load_deepseek_v41_streaming`,
`ExpertStreamingConfig` overrides `slot_layout`/`cache_scope`/`island_layers`/
`verify_record_hashes` (`mtplx/expert_runtime.py:160,163,172,156`), `island_layers
require component-banks + cache_scope=layer` (`:371,369`), `model._mtplx_expert_runtime`,
`runtime.manifest.records[*].{layer,expert,sha256}` (`mtplx/expert_manifest.py:516,847`),
`mlx_lm.utils.load_tokenizer`, `mlx_lm.models.cache.make_prompt_cache`.

Assumptions (stated in the docstring): W1's `mtplx/models/deepseek_v41.py` exists;
the model follows `model(ids[None], cache=c) -> logits` (falls back to cacheless
re-feed on `TypeError`); local manifest carries the HF identity so admission
passes (W3). A missing-W1 / admission failure is reported cleanly (exit 3), not a
raw traceback.

## 3. `serve_health.sh`

Runs **inside** `gpu_window.sh` (does not take the lock or touch launchctl/:8080
itself). Serves the WORKTREE's code (`PYTHONPATH` + `cd` = worktree; campaign venv
in the main checkout) and asserts `mtplx.__file__` is under the worktree before
serving (memory/editable-install-cwd-shadowing.md — modeled on
`evals/litellm_hy3/serve.sh:14` campaign-venv + `run_humaneval_guarded.sh:17`
PYTHONPATH). Steps: pick the first free port in 18080–18299 (skips 8080; a Python
bind probe) → `mtplx serve --model ~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2
--host 127.0.0.1 --port <free> --no-auth` (flags verified via `mtplx serve --help`;
AR is forced by the streamed artifact per W3) → wait `/health` (600 s, or report
the server log tail if it dies) → print `generation_mode` + `model_key` (recursive
JSON scan) + the `/v1/models` id → one greedy chat completion, printing usage +
server-reported and wall-clock tok/s → clean SIGTERM/SIGKILL stop in a trap. It
never binds :8080.

## 4. Checks run (CPU only, `nice -n 19`, MLX default device forced to CPU; no GPU, no lock, no launchctl, no :8080)

- `bash -n scripts/deepseek_v41/gpu_window.sh` → OK; `serve_health.sh` → OK.
- `shellcheck` — not installed on this box (`which shellcheck` → not found).
- `python -m py_compile` on `gate_stream_equals_resident.py`, the embedded
  `gpu_window.sh` fcntl-holder heredoc, and all six `serve_health.sh` inline
  Python blocks → all compile.
- `gate_stream_equals_resident.py --help` → renders. Pure-helper unit checks
  (`_parse_layers`, `build_parser` defaults incl. `--no-admit` /
  `--no-verify-record-hashes`, `_out_dir` stamped dir) → all pass.
- Import of the gate's real runtime deps on CPU
  (`mx.set_default_device(mx.cpu)`): `mtplx` resolves to the worktree via
  `PYTHONPATH`, `mtplx.models.deepseek_v41_loader.load_deepseek_v41_streaming`,
  `mtplx.models.expert_mlx`, `mlx_lm.utils.load_tokenizer`,
  `mlx_lm.models.cache.make_prompt_cache` → all import.
- `mtplx serve --help` confirms `--model --host --port --no-auth
  --generation-mode` exist.
- Functional CPU checks: the free-port finder returns `18080`; the health/response
  JSON parsers print `generation_mode`, `model_key`, usage, server-reported +
  wall-clock tok/s, and the completion text against sample payloads.

Not run (out of scope for the harness author): `gpu_window.sh` itself, anything
touching the lock/launchctl/:8080, and any MLX GPU execution. Those are the
orchestrator's, per `README.md`.
