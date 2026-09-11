# W53 — DeepSeek-V4.1 served-decode overhead: where the tok/s goes

**Question (from the window-21 gap).** The in-process bench reports **6.24 tok/s**
with the lever stack (`scripts/deepseek_v41/ab_decode_env_levers.py`, greedy,
1,024-token prompt, 256 decode). The served path (`mtplx serve …
DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4`, profile `deepseek-v41-mxfp4-75`,
`/v1/chat/completions` streaming, temp 1.0 / top_p 0.95 / top_k 20,
max_tokens 1024, natural stop) reports AR decode **4.96 / 2.67 / 2.19 tok/s**
for 137 / 189 / 252-token generations and MTP **2.45 / 2.78 / 2.94** for
156 / 312 / 240 tokens. So served decode is 1.3–3× slower than the bench, and
AR appears to *fall with generated length*. The task hypothesis: **O(n)
per-token work in the server loop** (re-detokenize, re-encode, stop-scan over
full text, per-token cache stores, logprob bookkeeping) or per-token GPU work
(sampling over the 129,280-vocab logits).

**Verdict.** The hypothesis is **not borne out**. There is **no O(n)-in-
generated-length work in the served generation thread** for these lengths, and
the per-token host overhead the served path adds over the greedy bench is
**sub-millisecond to low-single-digit-ms** — it cannot explain a 40+ ms/token
gap. The measured behaviour is dominated by **content-dependent expert
streaming** in this mxfp4 SSD-streaming model, not by the server loop.

---

## Definitions (one name per metric, used throughout)

- **decode_tok_s** — `completion_tokens / decode_s`, where `decode_s` is the
  client-observed span from first token (TTFT) to completion
  (`scripts/fable/server_cell_bench.py`; server-stamped `decode_s` in the
  window-21 receipt).
- **per-token decode time** — `decode_s / completion_tokens` (ms/token).
- **greedy bench** — `_generate()` in `ab_decode_env_levers.py:464`: bare
  `model(input)` forward + `mx.eval` + `mx.argmax(...).item()`. No sampler, no
  detokenize, no stop-scan, no `generate_ar` bookkeeping.
- **served AR** — `generate_ar()` (`mtplx/generation.py:6851`) reached through
  `_run_generation_dispatched()` (`mtplx/server/openai.py:24659`, call site
  `openai.py:24888`).
- **O(n)** — cost that grows with the number of tokens generated so far.

---

## 1. Per-token operation table (served AR loop)

The served AR loop is the classic loop in `generate_ar`
(`mtplx/generation.py`, `for step in range(_classic_start, max_tokens)` at
line 7259). The pipelined lane (`MTPLX_AR_PIPELINE`, line ~7118) and the async
double-buffer (`MTPLX_ASYNC_AR`, line ~7107) are both DARK by default and gate
on model hooks DeepSeek-V4.1 does not publish, so the classic loop is what runs.

The streaming callback `token_callback` is `record_tokens`
(`openai.py:24742`) → `on_tokens` (`openai.py:32175`): each does
`token_times.append` + `queue.put` — **O(1), host, non-blocking** (the queue is
an **unbounded** `asyncio.Queue`, `_LoopFedStreamQueue`, `openai.py:623`, put
via `call_soon_threadsafe(put_nowait)`). Detokenize / stop-scan / SSE assembly
run on the **consumer** side, so they never back-pressure the generation thread
and never enter `decode_tok_s`.

| # | Per-token op (served AR) | Cite | O(1)/O(n) in gen length | Host/GPU | Bench does it? |
|---|---|---|---|---|---|
| 1 | `_loop_guard.observe(tokens)` | `generation.py:7261`, `loop_guard.py:293` | O(1) — **off by default** (`_loop_guard_enabled`→False, `openai.py:22823`); even armed, scans `tokens[-window:]` (window≤2048), so O(window) not O(n) | Host | No |
| 2 | `_thinking_guard.observe(tokens)` | `generation.py:7273` | O(1) — None (thinking off) | Host | No |
| 3 | `logits_row = logits[0]` | `generation.py:7290` | O(1) | GPU view | (bench slices `logits[0,-1]`) |
| 4 | constraint mask | `generation.py:7296` | n/a — no constraint | — | No |
| 5 | **sampling** `_sample_from_logits` → `sparse_distribution_from_mlx_logits` | `generation.py:5427`, `fast_sampling.py:567`, `fast_sampling.py:380` | **O(1)** — device argpartition + logsumexp over the **full 129,280 vocab** + one `mx.eval`; only the k≤20 support crosses to host | GPU + 1 host round-trip | **No** (bench uses `mx.argmax`) |
| 6 | `Counter(tokens)` for penalties | `generation.py:7307` | O(n) **but not on this config** — gated on presence/frequency penalty, which the served sampler does not set | Host | No |
| 7 | `tokens.append` + `emit_token` | `generation.py:7312`, `:7080` | O(1) (unbounded queue) | Host | No |
| 8 | `emit_trace()` → `maybe_emit` | `generation.py:7048`, `:2126` | O(1) — live sink is ~1 Hz; **`trace_totals()` built a ~55-key dict every token** (see §3 fix) | Host | No |
| 9 | `events.append({...})` | `generation.py:7314` | O(1) time, O(n) memory | Host | No |
| 10 | `_trim_repeated_suffix(tokens)` | `generation.py:7322`, `:3316` | **O(1) here** — `_detect_repeated_token_suffix` early-returns while `len < min_tokens` (**768**); all cells ≤252 tokens never reach it. When it does run it is bounded by `max_block=96` | Host | No |
| 11 | stream-gate `emit_limit`/`_tail_candidate` | `generation.py:3101`,`:3162` | O(1) — bounded by `max_block=96` and `span_floor=96`, independent of n | Host | No |
| 12 | `_is_stop(token, stop_token_ids)` | `generation.py:7336` | O(1) set membership | Host | No |
| 13 | `rt.forward_ar([[token]], cache)` | `generation.py:7344` | O(context) KV read (grows slowly); **content-dependent expert SSD streaming** | GPU + SSD | Yes (bare `model()`) |
| 14 | `_eval(logits_next)` | `generation.py:7365` | O(1) | GPU sync | Yes (`ops.sync`) |

**Consumer side (does not touch `decode_tok_s`):** `_IncrementalTokenDecoder`
(`openai.py:26234`) is incremental (O(1)/token). `_StopSequenceStreamMonitor`
(`openai.py:557`) is incremental and a no-op with no client stop sequences
(natural-stop cells set none). `_decode(tokenizer, tokens)` at loop end
(`generation.py`) is a single full detokenize — **O(n) once**, amortized O(1).

**MTP loop (`generate_mtpk`, `generation.py:8232`, main loop
`while len(tokens) < max_tokens` at line ~10281).** Same `record_tokens` emit
path (O(1)). Per accepted-token: draft read/forward, one verify forward over
the K+1 window, accept/sample compare over ≤K rows, KV commit — all **bounded
by depth K, O(1) in generated length**. No O(n) host op. MTP was left
documented rather than blind-instrumented: the loop is ~1,300 lines with many
branch exits (draft readers, target-prefix verify, adaptive width, compiled
verify) and blind stage laps there are high-risk. The `StageTimer` is
loop-agnostic and drops into the same `begin`/`lap`/`add`/`tick_token` points.

---

## 2. Root cause — content/expert streaming, not the loop

**Decisive evidence (window-21 AR-1K receipt,
`docs/deepseek-v41/receipts/gpu-windows/window-21/ar-1k/`).** The three AR
cells are the **same 1,024-token prompt** at **three different seeds**:

| seed | completion_tokens | decode_s | decode_tok_s | per-token ms |
|---|---|---|---|---|
| 20260829 | 137 | 27.43 | 4.96 | 200 |
| 20260830 | 189 | 70.30 | 2.67 | 372 |
| 20260831 | 252 | 114.50 | 2.19 | 454 |

Same prompt, seed the only difference → **2.3× spread in per-token time**
(200 → 454 ms). A deterministic O(n) host loop produces the *same* per-token
curve regardless of seed; it cannot manufacture a 2.3× seed-dependent spread.
The spread is **content**: different sampled tokens route to different experts,
and this artifact streams mxfp4 experts from SSD, so per-token cost tracks the
expert-cache/streaming hit rate of the tokens that happen to be drawn.

Corroboration:
- **MTP does not fall** (2.45 → 2.78 → 2.94 for 156 → 312 → 240 tokens); a
  server-loop O(n) bug would hit MTP too. Its variance is also seed/content.
- Every candidate O(n) op is **gated off** below these lengths (repetition trim
  at 768; loop guard off / windowed at 2048; no penalties → no `Counter`; no
  client stops → stop-monitor inert; incremental detok).
- The falloff is not fixed-overhead amortization: a fixed per-request tail would
  make **short** generations slower per token, the opposite of what is seen.

**The bench↔served gap is apples-to-oranges.** The greedy bench
(`ab_decode_env_levers.py:464`) calls `model()` directly with `mx.argmax` — it
is not `generate_ar` and it decodes a *different (greedy) token stream* than the
sampled served runs, hitting different experts. So "6.24 vs 4.96" mixes three
things: (a) sampler + bookkeeping overhead, (b) greedy-vs-sampled expert
routing, (c) the `rt.forward_ar` wrapper vs a bare `model()` call. Only (a) is a
server-loop cost, and §4 shows (a) is sub-ms.

---

## 3. What was fixed

No provable O(n) work exists to remove. Two changes were made, both small and
safe, plus the measurement instrument (§ below):

1. **Removed avoidable per-token host allocation — lazy trace totals.**
   `emit_trace()` (called every token) built `trace_totals()` — a ~55-key dict —
   unconditionally, even when no live sink is attached and no file trace is
   enabled, in which case `maybe_emit` is a guaranteed no-op and the dict is
   discarded. Now guarded by `_DecodeTrace.wants_totals()` (`generation.py:2111`)
   and skipped on that path. **Flag:** `MTPLX_AR_LAZY_TRACE_TOTALS` (default on;
   `=0` restores the unconditional build). **Exactness:** byte-identical — the
   totals are never consumed when skipped; a test asserts identical tokens/text
   with the flag on vs off (`tests/test_serve_stage_timing.py::
   test_lazy_trace_totals_is_byte_identical`). Impact is microseconds/token,
   i.e. **<1%** at a ~160 ms/token GPU cadence — honest and minor.

2. **No sampler change.** The served sparse sampler is already the optimized
   path: materializing full-vocab rows on the host was a measured 15–19%
   serve-lane regression (`fast_sampling.py:380` docstring), and the argpartition/
   logsumexp must see the full vocab to pick top-k/top-p correctly. Pruning
   before the top-k selection would change the distribution, which the exactness
   contract forbids, so it was **not** done.

---

## Instrument added — per-token stage timer (the real deliverable)

`mtplx/serve_stage_timing.py` — a near-zero-overhead probe. Armed by
`MTPLX_SERVE_STAGE_TIMING=1`; when off, every method is one boolean test and the
loop keeps its historical timing byte-for-byte (asserted by
`test_stage_timer_disabled_is_a_noop` and
`test_generate_ar_stage_timing_off_by_default`). It records only host
`perf_counter` deltas and issues **no `mx.eval`/GPU sync of its own** — a
stage's wall time is whatever the loop already forces to materialize inside it
(`forward`/`eval` reuse the deltas the loop already measures).

Wired into `generate_ar` at stages `guards` / `sample` / `emit` / `stopcheck` /
`forward` / `eval`. The summary lands on `GenerationStats.serve_stage_timing`
(`generation.py:2957`), is added to the **`mtplx_openai_generation`** log event
(`openai.py`, guarded by non-empty), and is written to a JSON receipt when
`MTPLX_SERVE_STAGE_TIMING_RECEIPT=<dir|file>` is set (append-only, request- and
timestamp-suffixed).

Example table via the CPU stub (`tests/test_serve_stage_timing.py`, trivial
forward so `sample` dominates; on the real GPU model `forward` will dominate and
this table attributes the sampler's true share):

```
tokens=64  per_token_wall≈0.166 ms   sample 90.9% | forward 5.6% | eval 0.6%
           guards 0.8% | emit 0.4% | stopcheck 0.3%
```

Tests: `tests/test_serve_stage_timing.py` (10 cases — timer no-op-when-off,
table shape, env read, receipt to dir/file, empty/absent-env no-ops,
`generate_ar` off-by-default, `generate_ar` populates table, lazy-totals
exactness). All green under `mx.cpu`, no experts, <3 GB RSS.

---

## 4. Sampler cost on CPU (129,280-vocab row) — can it be tens of ms?

`scripts/deepseek_v41/sampler_cpu_cost.py` (pins `mx.cpu`, no GPU/experts),
200 iters over a DeepSeek-V4.1-shaped 129,280-vocab f32 logits row, sampler
temp 1.0 / top_p 0.95 / top_k 20:

| arm | median ms | Δ vs greedy |
|---|---|---|
| `greedy_argmax` (`mx.argmax`, what the bench does) | 0.130 | — |
| `sparse_device_topk` (**served path**: argpartition+logsumexp over full vocab + draw) | 0.869 | **+0.74** |
| `dense_full_vocab` (unpruned host reference: full softmax + argsort every token) | 1.393 | +1.26 |

**Answer: no — the sampler cannot account for tens of ms/token.** Its
*arithmetic* is **0.74 ms/token** over greedy on CPU, and the in-tree GPU
measurement for this same sparse path is **+0.25 ms/token** inside the busy
decode stream (`fast_sampling.py:380` docstring). Even the pathological dense
full-vocab argsort is 1.26 ms. The sparse path is ~1.6× faster than dense,
matching the documented 15–19% serve-lane figure. The 40+ ms/token bench↔served
gap is therefore **not** the sampler; it is expert routing (greedy-vs-sampled)
plus the `forward_ar` wrapper — which the §-above stage timer resolves exactly in
a real window (compare `forward` share vs `sample` share).

---

## 5. Expected recovery and the re-run command

**Expected served tok/s recovery from server-loop fixes: <1%.** The decode is
GPU + expert-streaming bound; the server loop's host work is sub-ms to
low-single-ms per token and mostly gated off. The lazy-totals fix removes
microseconds/token. The real lever for served decode remains **expert
residency/streaming** (island placement, residency sets, prefetch — per the
existing DSV4.1 decode-lever ledger), not the generation loop. The stage timer
now makes that provable per window instead of inferred.

**Re-run the 1K served cells with the stage timer armed** (from the integration
worktree, holding the GPU flock; never `:8080`):

```bash
export MTPLX_SERVE_STAGE_TIMING=1
export MTPLX_SERVE_STAGE_TIMING_RECEIPT=docs/deepseek-v41/receipts/w53-served-cells/stage-timing
# AR (profile levers ship via child_env: HEAD_MODE=bf16 + SINKHORN_METAL + ATTN_COMPILE)
DSV41_CONTEXTS=1024 \
DSV41_RECEIPT_DIR=docs/deepseek-v41/receipts/w53-served-cells \
  nice -n 19 bash scripts/deepseek_v41/served_cell_bench.sh
# MTP
DSV41_CONTEXTS=1024 \
DSV41_SERVE_EXTRA_ARGS="--generation-mode mtp" \
DSV41_RECEIPT_DIR=docs/deepseek-v41/receipts/w53-served-cells-mtp \
  nice -n 19 bash scripts/deepseek_v41/served_cell_bench.sh
```

The `mtplx_openai_generation` events in the server log then carry a
`serve_stage_timing` block (per-token `forward` / `sample` / `eval` / `emit` /
`guards` / `stopcheck` shares), and one JSON receipt per request lands under the
receipt dir — the exact per-stage attribution of the served decode window.

**Sampler CPU microbench (no GPU):**

```bash
PYTHONPATH=$PWD nice -n 19 .venv/bin/python3 \
  scripts/deepseek_v41/sampler_cpu_cost.py --iters 200 --vocab 129280
```

---

## 6. Per-request expert-streaming counters (attribute the seed spread directly)

§2 shows the seed spread is expert routing. To let a served window **prove**
that from its own log instead of inferring it, each `mtplx_openai_generation`
event now carries a `serve_stream_counters` block: the **decode-phase** delta
(loop start → loop end, so prefill — which is seed-independent — is excluded) of
three counter sources, plus per-completion-token averages. Same block on the AR
and MTP events. Implemented in `mtplx/serve_stream_counters.py`; wired in
`generate_ar` and `generate_mtpk` (snapshot bracketing the decode loop) and
surfaced by `_run_generation` (`openai.py`).

Sources (all best-effort; a source that is not present is simply omitted, and a
runtime with no streaming attached yields `{}` — no crash, CPU-stub tested):

| block | source | key fields (delta over decode) |
|---|---|---|
| `expert_cache` | `rt.expert_streaming_snapshot()["cache"]` (`CacheCounters`, `expert_streaming.py:116`) | `expert_hits`, `expert_misses`, `hit_rate`, `miss_rate`, `records_streamed` (persistent+transient loads), `bytes_read`, `route_calls`, `misses_per_token`, `records_streamed_per_token`, `bytes_read_per_token` |
| `incremental_misses` | `…snapshot()["incremental_misses"]` | `routes`, `parts`, `routes_per_token`, `parts_per_token` |
| `engram_row_cache` | Σ `rt.model._engram_banks[*].cache.stats` (`NGramRowCache`, `ngram_row_cache.py:284`) | `hits`, `misses`, `hit_rate`, `miss_rate`, `rows_read`, `gathers`, `evictions`, `misses_per_token`, `rows_read_per_token` |
| `route_probe_counts` / `route_probe_sums_ns` | `mtplx.expert_route_probe` module counters, **only when `MTPLX_ROUTE_STAGE_PROBE=1`** | per-(phase,stage) call-count and ns-sum deltas (e.g. split-route vs all-hit layer calls) |

The snapshots are plain counter reads — no `mx.eval`, no GPU sync, no I/O — so
the probe adds a few dict copies per request. Because the block is emitted
whenever a streaming runtime is attached (the DeepSeek loader builds an
`ExpertStreamingRuntime`), a window comparison across the three seeds reads the
miss rate / bytes-per-token straight off each event; the expectation from §2 is
that the slow seeds show a higher `expert_cache.miss_rate` and
`bytes_read_per_token`.

**Served command that produces the counters (window, holding the GPU flock;
never `:8080`):**

```bash
export MTPLX_SERVE_STAGE_TIMING=1                                   # per-token stage table
export MTPLX_SERVE_STAGE_TIMING_RECEIPT=docs/deepseek-v41/receipts/w53-served-cells/stage-timing
export MTPLX_ROUTE_STAGE_PROBE=1                                    # adds the route_probe_* sub-blocks
# expert_cache + engram_row_cache need no env — they emit whenever a streaming
# runtime is attached (always, on this artifact).
DSV41_CONTEXTS=1024 \
DSV41_RECEIPT_DIR=docs/deepseek-v41/receipts/w53-served-cells \
  nice -n 19 bash scripts/deepseek_v41/served_cell_bench.sh
# MTP: add DSV41_SERVE_EXTRA_ARGS="--generation-mode mtp" and a *-mtp receipt dir.
```

Each `mtplx_openai_generation` event in the server log then carries
`serve_stream_counters` (and `serve_stage_timing`); with the receipt env set, a
`mtplx_serve_stage_timing` JSON receipt per request carries both blocks under
`stream_counters` / `stages`.
