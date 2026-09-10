# W14 — GPU-phase bench + HumanEval harness (staged, CPU-verified)

Branch `feat/deepseek-v41-w14` off `feat/deepseek-v41-streaming` @ `4e276eb2`
(W8 merged). CPU only, `nice -n 19`, no GPU/Metal, no lock, no launchctl, no
`:8080`, `~/models` read-only. The goal was to have the **whole GPU phase ready
to launch** the moment the faithful port passes its CPU probe, so nothing serial
remains. None of these scripts were run on the GPU by the harness author; they
were exercised on CPU via `--dry-run` and unit tests.

## Delivered (allowlist only)

| file | what |
| --- | --- |
| `scripts/deepseek_v41/bench_standard_shape.py` | David's standard-shape greedy decode bench (1,024 + 16,384 cells) through the serve-path loader |
| `scripts/deepseek_v41/humaneval_cell.py` | one HumanEval(164) pass@1 cell driver; derives strict / completed-task pass@1 + truncation rate |
| `scripts/deepseek_v41/humaneval_cell.sh` | serve-on-free-high-port + engagement guard + drive `humaneval_cell.py`, clean stop |
| `scripts/deepseek_v41/README.md` | launch order (probe → gate → bench → serve_health → humaneval) from the integration worktree |
| `tests/test_deepseek_v41_bench_scripts.py` | 20 CPU unit tests (parsing, receipts, append-only guard, metrics, dry-run) |
| `docs/deepseek-v41/W14_REPORT.md` | this report |

No model/loader/engram files were touched.

## Exact sampler settings (HumanEval cell)

From David's served battery `docs/perf/qwen38-475-battery/README.md:9`
("temperature 1, top-p 0.95, top-k 20"), NOT greedy, one seed, one sample/task
(`n=1`, memory/humaneval-one-seed.md). The output cap is set so it does NOT bind
(memory/eval-truncation-is-not-failure.md): default `--max-tokens 32768` for the
sampled xhigh-thinking cell — a bound cap then shows up as a nonzero
`truncation_rate`, never as silent failures.

Defaults in `humaneval_cell.py` (`build_parser`):
`--temperature 1.0`, `--top-p 0.95`, `--top-k 20` (sent as `--extra-body
top_k=20`), `--max-tokens 32768`, `--seed 42`, `--endpoint chat`, `--n 1`.
Reported metrics (`compute_cell_metrics`): `strict_pass_at_1` (all 164),
`completed_task_pass_at_1` (rows whose `finish_reason != "length"`),
`truncation_rate`.

## Harness reused (file:line — not reimplemented)

- **HumanEval driver** (server generation + retries + report):
  `scripts/code_eval_gate.py` — `run` at `scripts/code_eval_gate.py:501`,
  `main` at `:677`, `build_payload` at `:152`, `generate_one` at `:308`. This is
  the driver the Qwen3.8 PRs used (`evals/litellm_hy3/run_humaneval.sh:34` calls
  it). `humaneval_cell.py` builds its argv (`_gate_argv`) and calls
  `code_eval_gate.main(...)`, then reads the report JSON.
- **HumanEval scorer** (loading, program assembly, sandboxed exec, pass@k):
  `mtplx/benchmarks/code_eval.py` — `load_humaneval:109`, `extract_code:186`,
  `build_program:204`, `run_candidate:258`, `pass_at_k:341`, `summarize:361`.
- **Dataset**: the canonical 164-task `HumanEval.jsonl` at
  `/Users/davidtai/projects/OpenSourceWTF/benchmark-archive/datasets/HumanEval.jsonl`
  (`wc -l` = 164), the `HUMANEVAL_DATASET` default in `humaneval_cell.sh`.
- **Prompt build** (shared with the gate + the hidden-state dump): the importable
  `build_prompt` at `scripts/deepseek_v41/dump_hidden_states.py:47`, over
  `mtplx.prefill_bench._prompt_build_for_context` at `mtplx/prefill_bench.py:456`
  (David's standardized prefill_bench programming prompt) + BOS id 0 prepend, so
  every bench receipt carries the same `prompt_build` metadata as the gate
  (`scripts/deepseek_v41/gate_stream_equals_resident.py:356`) and the dump.
  `bench_standard_shape.py` imports it by file path (`scripts/` is not a package).
- **Serve-path loader**: `load_deepseek_v41_streaming` at
  `mtplx/models/deepseek_v41_loader.py:529` (streamed experts + engram attached;
  `slot_layout=component-banks`, `island_layers=()`), the same entry the gate and
  dump use. Model forward `Model.__call__` at `mtplx/models/deepseek_v41.py:921`,
  `make_cache` at `:929`.
- **Serve lifecycle** (port finder in 18080–18299 skipping 8080; editable-install
  engagement assertion): copied from `serve_health.sh:55,82` into
  `humaneval_cell.sh:88,115`.
- **Cheap counters, read off the hot path** (so the decode tok/s the bench
  measures is not inflated): expert records from `runtime.snapshot()["cache"]`
  (`ExpertStreamingRuntime.snapshot` at `mtplx/expert_runtime.py:3441`; the
  `CacheCounters` dict `mtplx/expert_streaming.py:116`, `expert_requests`), and
  engram rows from each hook's `row_cache.stats` (`NGramRowCache` at
  `mtplx/ngram_row_cache.py:250`, `hits`+`misses`). The P1.7 gate wraps
  `switch_mlp` because it needs the per-step selection; a rate bench must not.

## bench_standard_shape.py — what one invocation records

For each `--context-tokens` cell (default 1,024 and 16,384), one greedy
generation of `--steps` decode tokens (default 256). The prefill forward produces
the first token (that is the TTFT); the decode loop then runs `steps` forwards.
Per cell/repeat: `prefill_tok_s` (prompt tokens / TTFT), `ttft_s`, `decode_tok_s`
(steps / decode wall), `peak_mlx_gb` (`mx.get_peak_memory`) + `process_rss_gb`
(RSS), `wall_s`, `expert_records_gathered`, `engram_rows_gathered`, and the first
200 chars of the decoded text. `fastest_of` reports the max decode tok/s / min
TTFT with the range (memory/report-fastest-of-seeds.md). Greedy is deterministic
so three seeds are not needed; `--repeats` is for later speed windows.
`--max-kv` auto-sizes to cover the largest cell + decode.

## Receipts (append-only)

Every script writes one fresh UTC-stamped directory per invocation under
`.benchmark-artifacts/deepseek-v41/<step>/<utc-stamp>/` (never reused; a same-
second collision gets a `-N` suffix) and refuses to overwrite an existing receipt
file (`write_receipt` raises `FileExistsError`) — memory/never-overwrite-a-
measurement.md. The receipt filename is cap/seed/window-suffixed
(`bench_standard_shape__ctx1024-16384__steps256__seed0__<stamp>.json`,
`humaneval_cell__seed42__cap32768__<stamp>.json`). Each carries `git_rev`,
`spec_key`, `manifest_sha256` and the prompt-build metadata. The HumanEval cell
also keeps the full `code_eval_gate` report JSON and a completions sidecar, and
is served with the SSD SessionBank cold tier OFF (`--ssd-session-cache off`) so
the shared prod bank never warms it (memory/ssd-session-bank-warms-benchmarks.md).

## Dry-run / test double

Every script takes `--dry-run`: a CPU-only test double (no model, no Metal, no
server, no code execution) that exercises the SAME timing/receipt/metric code
paths via a fake tokenizer, a serve-path-shaped fake model whose forward bumps
the same runtime + engram counters, and a synthetic `code_eval_gate` report. This
proves argument parsing, the real prefill_bench + BOS prompt build (a fake
tokenizer yields ctx+1 input tokens), the receipt paths, and the append-only
guard. `humaneval_cell.sh --dry-run` prints the plan and drives the `.py`
dry-run.

## CPU checks run (nice -n 19, no `-n auto`, no MLX imported)

- `python -m py_compile` on all three Python files → OK.
- `bash -n` on `humaneval_cell.sh` → OK; its 2 inline Python heredocs compile.
- `pytest tests/test_deepseek_v41_bench_scripts.py` → **20 passed in 0.07s**.
- `bench_standard_shape.py --dry-run --steps 4` → receipt with both cells; ctx
  1,024 → `input_tokens=1025` (BOS), expert/engram gather deltas nonzero.
- `humaneval_cell.py --dry-run` → strict pass@1 0.6000 (6/10), completed-task
  pass@1 0.6667 (9 completed), truncation_rate 0.1000 (1 truncated) — the
  truncation-aware split verified on a synthetic report.
- `humaneval_cell.sh --dry-run` → prints the serve/score plan, writes a receipt,
  no serve.

## Preconditions before the GPU phase (not the harness author's to clear)

Worker W1's `mtplx/models/deepseek_v41.py` must exist (else `ResidentLoadError`,
reported cleanly). The faithful port must pass its **CPU probe** — a
non-degenerate, golden-matching CPU forward at 1,024 + BOS — before the GPU phase
is worth running: W8 withdrew the "Q2 quality" claim and recorded a
single-forward **streamed-prefill defect** (`docs/deepseek-v41/W8_REPORT.md`);
root-causing is W9 (torch goldens) / W10 (faithful transliteration). The local
manifest must carry the HF identity so admission passes (W3), or
`--no-admit --admission-receipt` injects it.
