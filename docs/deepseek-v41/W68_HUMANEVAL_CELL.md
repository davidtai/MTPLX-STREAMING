# W68 — HumanEval(164) quality cell, per decode lane (AR + DSpark)

Scope: bring the DeepSeek-V4.1-Flash HumanEval quality gate up to date with the
served path as it stands after W52 (real chat template + BOS, thinking OFF by
default) and W57 (two served decode lanes: AR and DSpark-DIRECT), and add a
paired AR-vs-DSpark comparison so tie-flip-class divergences can be inspected
one task at a time.

The quality gate for this program is **one HumanEval(164) pass@1 cell per
candidate at David's sampler** (temperature 1, top-p 0.95, top-k 20, seed
20260829, non-binding cap), served through `mtplx serve`, never greedy
(`memory/humaneval-one-seed.md`).

Allowed-writes touched:
`scripts/deepseek_v41/humaneval_cell.py`, `scripts/deepseek_v41/humaneval_cell.sh`,
`scripts/deepseek_v41/humaneval_lane_compare.py` (new),
`tests/test_deepseek_v41_bench_scripts.py`,
`tests/test_deepseek_v41_humaneval_cell_served.py` (new), this report.

---

## 1. What the cell is

Message construction and scoring are **reused verbatim** from
`scripts/code_eval_gate.py` — the same driver the Qwen3.8 125B MTPLX PRs
(#475/#478/#482/#485/#488) used for their quality cells:

- **System prompt:** "You are a Python programming assistant. Complete the
  function you are given. Reply with a single fenced Python code block
  containing the complete function definition …" (`_HUMANEVAL_SYSTEM`).
- **User turn:** ``Complete this function:\n\n```python\n{task.prompt}\n``` ``
  (`build_messages`).
- **Scoring:** `mtplx/benchmarks/code_eval.py` — extract the fenced block, run
  each candidate in a fresh sandboxed subprocess against the reference tests,
  pass@1. `--endpoint chat` posts to `/v1/chat/completions`, so the served path
  renders DeepSeek-V4.1's real chat template with BOS and thinking OFF (W52).

`humaneval_cell.py` drives that gate, then derives the truncation-aware metrics
`code_eval_gate` does not itself compute:

- **strict pass@1** — passers / 164.
- **completed-task pass@1** — passers / (164 − truncated), excluding rows whose
  `finish_reason == "length"` (`memory/eval-truncation-is-not-failure.md`).
- **truncation rate** — truncated / 164. With the non-binding 2048-token cap and
  thinking OFF, this should be ~0; a nonzero value means the cap bound and the
  strict number is understated, not that the model failed.

### Sampler (David's served sampler)

| param | value |
|---|---|
| temperature | 1.0 (never greedy) |
| top-p | 0.95 |
| top-k | 20 (rides `code_eval_gate --extra-body top_k=20`) |
| seed | 20260829 |
| n | 1 (one seed, one sample per task) |
| max_tokens | 2048 (non-binding for the thinking-OFF chat path) |

### Receipts (append-only)

One fresh UTC-stamped directory per invocation under
`<DSV41_RECEIPT_DIR>/humaneval_cell/<stamp>/`, holding:

- `humaneval_cell__<lane>__seed<S>__cap<T>__<stamp>.json` — the derived receipt.
  The **lane is in the filename** so AR and DSpark cells never collide
  (`memory/never-overwrite-a-measurement.md`).
- `code_eval_gate_report.json` — the full driver report.
- `completions.jsonl` — the raw/extracted completion sidecar (offline rescore).

Each receipt records the **lane**, the raw **serve flags**, the
**`--expert-memory-limit`** (if any), and the daemon's resolved **decode-lever
env** — scraped from the server log's `[4/6] DeepSeek-V4.1 decode levers
(resolved env): …` startup line (`mtplx/server/openai.py`, W46) — plus a
self-contained **`per_task`** pass map that the lane comparison consumes without
re-opening either driver report.

---

## 2. Window commands (both lanes)

Each cell is a **step run inside `gpu_window.sh`**, which holds the exclusive GPU
lock and has already booted out the resident agent. The step never takes the
lock or touches `:8080` itself; it serves on a free high port (18080–18299),
asserts `/health` + editable-install engagement, runs the cell, and stops the
server cleanly. Run the two lanes **sequentially** — one server at a time keeps
the box under the 100 GiB knob / 110 GB hard limit (`memory/box-110gb-hard-limit`).

### AR lane (served default; the profile arms the byte-identical decode levers)

```bash
cd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w68
DSV41_LANE=ar \
DSV41_RECEIPT_DIR=docs/deepseek-v41/receipts/w68-humaneval-ar \
bash scripts/deepseek_v41/gpu_window.sh \
     bash scripts/deepseek_v41/humaneval_cell.sh
```

Serves: `mtplx serve --model …streaming-mxfp4 --host 127.0.0.1 --port <free>
--no-auth --ssd-session-cache off` (AR is the streamed artifact's default; the
profile child_env arms head bf16 + Sinkhorn kernel + attention compile + window
memo).

### DSpark-DIRECT lane (W57)

```bash
cd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w68
DSV41_LANE=dspark \
DSV41_RECEIPT_DIR=docs/deepseek-v41/receipts/w68-humaneval-dspark \
bash scripts/deepseek_v41/gpu_window.sh \
     bash scripts/deepseek_v41/humaneval_cell.sh
```

Serves: `mtplx serve …--no-auth --ssd-session-cache off --load-mtp
--generation-mode dspark --depth 3`.

### Env knobs

| var | effect |
|---|---|
| `DSV41_LANE=ar\|dspark` | which decode lane (default `ar`) |
| `DSV41_DEPTH=3` | DSpark draft depth (dspark lane) |
| `DSV41_RECEIPT_DIR=<dir>` | append-only receipt root (`DSV41_OUT_DIR` still honoured as an alias) |
| `DSV41_MEMORY_LIMIT_GIB=<N>` | → `mtplx serve --expert-memory-limit <N>GiB` |
| `DSV41_MAX_LIVE_KV_TOKENS=<N>` | → `mtplx serve --expert-max-live-kv-tokens <N>` |
| `DSV41_WORKERS=<N>` | concurrent in-flight requests (default 4) |
| `HUMANEVAL_LIMIT=<N>` | score only the first N tasks (smoke) |

HumanEval prompts are short (~150–500 tokens) and the output cap is 2048, so the
profile default `max_live_kv_tokens` (16,384) already admits four concurrent
slots; `DSV41_MEMORY_LIMIT_GIB` / `DSV41_MAX_LIVE_KV_TOKENS` are there for parity
with `served_cell_bench.sh`, not because the cell needs them.

### Dry run (CPU wiring proof, no serve/model/execution)

```bash
DSV41_LANE=dspark bash scripts/deepseek_v41/humaneval_cell.sh --dry-run
```

---

## 3. Expected wall time (~3–6 tok/s)

Decode work per lane: **164 tasks × ~200 output tokens ≈ 32,800 tokens.**

| served decode rate | decode time | ≈ |
|---|---|---|
| 3 tok/s | 32,800 / 3 ≈ 10,930 s | ~3.0 h |
| 6 tok/s | 32,800 / 6 ≈ 5,470 s | ~1.5 h |

So budget **~1.5–3 h of decode per lane**, plus a few minutes of model admission
(residents load + admit) and short-prompt prefill (fast). W57 measured served AR
at **2.2–5.0 tok/s** and served DSpark at **2.45–2.94 tok/s** on the Qwen-PR
prompt, so both lanes sit inside the 3–6 tok/s planning band; DSpark is roughly
break-even, not a large speedup, on this shape.

Decode on this box is memory-bandwidth-bound and concurrency does **not** multiply
aggregate throughput (`memory/mtplx-v2-concurrency-ceiling.md`), so `DSV41_WORKERS`
(default 4) trims tail latency and overlaps prefill/scoring but does not shrink the
decode wall much below the single-stream figure above. Plan **one GPU window per
lane, run sequentially** (never two servers concurrently — 110 GB hard limit).

---

## 4. Paired AR-vs-DSpark comparison

After both cells have written receipts, pair them (CPU, no serve):

```bash
PYTHONPATH=$PWD /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3 \
  scripts/deepseek_v41/humaneval_lane_compare.py \
  --ar     docs/deepseek-v41/receipts/w68-humaneval-ar/humaneval_cell/<stamp>/humaneval_cell__ar__seed20260829__cap2048__<stamp>.json \
  --dspark docs/deepseek-v41/receipts/w68-humaneval-dspark/humaneval_cell/<stamp>/humaneval_cell__dspark__seed20260829__cap2048__<stamp>.json \
  --out    docs/deepseek-v41/receipts/w68-humaneval-ar-vs-dspark.json
```

It reports each lane's strict / completed-task pass@1, the `strict_pass@1` delta
(DSpark − AR), and the **per-task diff list** — every task the two lanes disagree
on, split into `ar_only_pass` and `dspark_only_pass`, each entry carrying both
lanes' `finish_reason`. Because David's sampler is temperature 1 (not greedy),
some tasks will flip pass↔fail purely from sampling; the diff list is what lets a
human tell a sampling tie-flip from a genuine lane divergence (e.g. a DSpark
`finish_reason == "length"` next to an AR `stop`). Example human summary:

```
DeepSeek-V4.1-Flash HumanEval(164) — AR vs DSpark paired lane comparison
  AR      lane=ar  strict_pass@1=0.7500 (3/4)  completed_task_pass@1=1.0000  truncated=0
  DSpark  lane=dspark  strict_pass@1=0.7500 (3/4)  completed_task_pass@1=1.0000  truncated=0
  delta strict_pass@1 (DSpark - AR) = 0.0000
  shared=4 both_pass=2 both_fail=0 disagreements=2 agreement_rate=0.5000
  AR passed, DSpark failed (tie-flip class):
    HumanEval/9  ar.finish=stop dspark.finish=stop
  DSpark passed, AR failed (tie-flip class):
    HumanEval/32  ar.finish=stop dspark.finish=length
```

The `--out` summary is append-only (refuses to overwrite an existing file).

---

## 5. CPU test coverage

No GPU, no Metal, no model; importing `code_eval_gate` does not import MLX. Run
under `nice -n 19`, no `pytest -n auto`:

```bash
PYTHONPATH=$PWD nice -n 19 .venv/bin/python3 -m pytest \
  tests/test_deepseek_v41_bench_scripts.py \
  tests/test_deepseek_v41_humaneval_cell_served.py -q
```

- `test_deepseek_v41_bench_scripts.py` — parser defaults (David's sampler), the
  gate argv, the truncation-aware metric derivation, the append-only guard, the
  lane-suffixed receipt filename, and the dry-run receipt.
- `test_deepseek_v41_humaneval_cell_served.py` — the module driven **end to end
  against a stub HTTP server** in-process (real HTTP round-trip + the real
  sandbox scoring path): a correct-solution cell scores strict pass@1 = 1.0 with
  the lane and parsed decode-lever env in the receipt; a `finish_reason=length`
  cell is counted as truncated, not failed. Plus the decode-lever parser and the
  `humaneval_lane_compare.compare_lanes` diff (tie-flip list, unmatched task
  sets, pass@1 delta, append-only CLI).
