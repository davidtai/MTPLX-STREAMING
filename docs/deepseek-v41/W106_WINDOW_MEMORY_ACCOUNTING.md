# W106 — window memory accounting (budget, non-Metal peak, tree-kill, output)

The box has a **110 GB hard ceiling** and David's budget is **100 GB TOTAL for
everything** (agent + workers + the GPU step, together). W106 makes the
DeepSeek-V4.1 bench + guard account for that honestly: the plan is *derived from*
the total budget while compensating for the non-Metal requirements, the receipts
report the whole-process peak (not just the MLX allocator peak), the guard kills
the entire step process tree on abort, and every run persists its full output for
a text audit.

Files:
- `scripts/deepseek_v41/ab_decode_env_levers.py` — the A/B bench (items 3, the
  peak-memory directive, output persistence).
- `scripts/deepseek_v41/bench_standard_shape.py` — the shared memory probe /
  sampler (`_MLXMemProbe`, `_MemorySampler`, `_system_used_bytes`); item 2.
- `scripts/deepseek_v41/gpu_window.sh` — the guarded GPU window (items 1, 4).

---

## Definitions (every memory term, once)

These names are used verbatim in the receipts, the console lines, and the guard
log. Each is defined here **once**.

### Peak / envelope figures (receipt `memory` block + top level)

- **`mlx_peak_gb`** (== the legacy top-level **`peak_gb`**) — the MLX allocator
  peak of *this* process (`mx.get_peak_memory`). It EXCLUDES the Python heap, the
  positional-expert bank read buffers, the engram host-side row LRU, the
  tokenizer, other processes, and the OS file cache. It is **not** the box usage.
  `peak_gb` is kept as-is for old receipts' comparability and is documented as
  MLX-only.
- **`process_peak_rss_gb`** (top-level alias **`peak_process_gb`**) — the whole
  **process** peak RSS: `max(ru_maxrss, the 1 Hz in-process sampler's peak,
  mlx_peak)`. This INCLUDES the non-Metal footprint the MLX peak omits, so it is
  always ≥ `mlx_peak_gb`. This is David's "peak memory must include the non-Metal
  parts" figure. It is a PEAK over prefill+decode (the sampler brackets both),
  never the value at exit.
- **`system_used_peak_gb`** — the whole-**box** used-memory peak over the run:
  `(wired down + anonymous + occupied-by-compressor) pages × page size` from
  `vm_stat` — the SAME formula the `gpu_window.sh` phase-4 guard aborts on
  (anonymous, not active: active counts the file-backed page cache the 269 GiB
  mmap'd expert bank fills and the OS reclaims on demand). Shared helper:
  `bench_standard_shape._system_used_bytes`.
- **`system_used_at_start_gb`** — the same box figure sampled once when the
  in-process sampler starts (the decode baseline).

Two independent sources record the process peak, and **both** are kept:
1. the **in-process sampler** (in the receipt, always present — even when the run
   is not under `gpu_window.sh`), and
2. the wrapper's **process-tree RSS** (in the `gpu_window.sh` log): the SUM of RSS
   across STEP_PID and every descendant, plus the MAX single-process RSS.

### Budget-derivation terms (item 3; receipt `memory` block)

- **`budget_total_gb`** — `--memory-budget-total-gb N`: the TOTAL box budget for
  everything.
- **`budget_system_used_at_start_gb`** — the box used-memory baseline measured
  ONCE at process start (`bench._system_used_bytes`, the wrapper's formula), i.e.
  everything already resident before this bench allocates.
- **`budget_non_metal_overhead_gb`** — the process RSS *above* the MLX allocator's
  own accounting (Python heap + expert-reader buffers + engram host LRU +
  tokenizer). Used as a **conservative pre-load estimate** (default 10 GiB,
  `--non-metal-overhead-gb`) because the plan limit must be fixed before the model
  loads; see the two-phase note below.
- **`budget_non_metal_overhead_measured_gb`** — the same overhead **re-measured
  after load** as `process RSS − mx active memory` (None until measured).
- **`budget_kv_growth_to_max_kv_gb`** — the bytes the KV lanes grow to at
  `--max-kv`, from config dims × max_kv × bf16 (see the KV estimator note).
- **`budget_safety_gb`** — `--memory-safety-gb` headroom (default 3 GiB).
- **`budget_floor_gib`** — `--memory-budget-floor-gib` (default 20 GiB): the run
  refuses to start if the derived plan limit is below this.
- **`plan_limit_gib_derived`** — the plan limit the derivation produced (pre-load).
- **`plan_limit_gib_effective`** — the plan limit actually in force after the
  two-phase re-measure (== derived unless the MLX active limit was lowered).
- **`memory_plan_source`** — `"budget"` when `--memory-budget-total-gb` drove the
  plan, else `"explicit"` (`--memory-limit-gib` taken literally / legacy
  `--box-budget` default). The full budget key set is always present, with nulls
  on the explicit path, so receipts are self-describing.

### Guard terms (`gpu_window.sh`)

- **step tree RSS (sum)** — SUM of RSS across STEP_PID and every descendant.
- **max single process** — the MAX single-process RSS in that tree.
- **step process tree** — STEP_PID (the `bash -c "…"` chain) AND every descendant
  (the python benchmark, its `sleep`/subprocess grandchildren, …).

---

## Item 3 — `--memory-budget-total-gb`: derive the plan, compensate for non-Metal

Instead of taking `--memory-limit-gib` literally, derive the MLX plan limit from
the TOTAL box budget:

```
plan_limit_gib = budget_total
               − system_used_at_start      (measured at process start, vm_stat)
               − non_metal_overhead        (process RSS − mx active; see two-phase)
               − kv_growth_to_max_kv        (config dims × max_kv × bf16)
               − safety                     (default 3 GiB)
```

`--memory-budget-total-gb` **overrides** `--memory-limit-gib` (derive, don't take
the literal). The derived limit is fed into the existing W62
`derive_plan_from_budget(..., override_memory_limit_gib=…)` so the runtime
reserve / allocator-cache plumbing is unchanged; only the plan ceiling changes.

**Floor refusal.** If `plan_limit_gib < --memory-budget-floor-gib` (default 20)
the run refuses to start with a clear, actionable error (naming every term and how
to raise the budget / lower safety / reduce max-kv). It never opens a GPU window
on a plan too small to hold the model.

**Two-phase non-Metal overhead.** The plan limit must be fixed *before* the model
loads (the loader takes `memory_limit_bytes`). So:
1. **Estimate** — derive with the conservative `--non-metal-overhead-gb` estimate
   (default 10 GiB, mirroring the W62 `HOST_OVERHEAD_GIB` profile constant) and
   fix the plan.
2. **Load** the model.
3. **Re-measure** the real overhead as `process RSS − mx active memory` and record
   it (`budget_non_metal_overhead_measured_gb`). If it exceeds the estimate by
   more than 0.5 GiB, **lower the MLX active-allocation limit** by the overage so
   the TOTAL still fits the budget, and record the lowered limit
   (`plan_limit_gib_effective`). Best-effort + guarded — it never crashes the run.

**KV growth estimator.** `_kv_bytes_at_max_kv(config, max_kv)` is a LOCAL,
deliberately conservative estimate: per layer it prices the sliding-window ring at
the full `max_kv` (the bounded ring is opt-in), and on kv-source layers the
latent/compressed KV `[max_kv/ratio, head_dim]`, the decoupled rope key
`[max_kv/ratio, qk_rope_head_dim]`, and the index key `[max_kv/ratio,
index_head_dim]`, all bf16. Config dims are read from the artifact `config.json`
(flat or `text_config`-nested) without importing MLX or loading weights.
**TODO(W107):** replace with
`mtplx.models.deepseek_v41_cache.kv_bytes_at_max_kv` — a sibling worker is adding
the exact per-lane helper on another branch; do not depend on it, just swap this
call site when it lands.

**Receipt keys** (in the `memory` block, alongside the item-2 envelope keys):
`memory_plan_source`, `budget_total_gb`, `plan_limit_gib_derived`,
`plan_limit_gib_effective`, `budget_system_used_at_start_gb`,
`budget_non_metal_overhead_gb`, `budget_non_metal_overhead_measured_gb`,
`budget_kv_growth_to_max_kv_gb`, `budget_safety_gb`, `budget_floor_gib`.

**Bench command line — item 3 on the 16K cell** (do not run here; a GPU window is
held by another worktree):

```
.venv/bin/python3 scripts/deepseek_v41/ab_decode_env_levers.py \
  --context-tokens 16384 --decode-tokens 256 --max-kv 17408 \
  --memory-budget-total-gb 100 \
  --out <receipt>.jsonl
```

(`--memory-safety-gb`, `--memory-budget-floor-gib`, `--non-metal-overhead-gb`
default to 3 / 20 / 10 GiB.)

---

## Item 4 — tree-kill the whole step process tree on abort

On any abort — the child-tree RSS cap, the system-used ceiling, or a TERM/INT to
the wrapper — `gpu_window.sh` kills the ENTIRE process tree of the step, not just
STEP_PID.

The step is a `bash -c "a; b; c"` chain whose python benchmark underneath holds
the memory. Killing only STEP_PID (pre-W106) left the python orphaned (reparented
to launchd), still holding GPU/host memory, and a chained next command could still
start after the abort.

- `_step_tree_pids <root>` — every pid in the tree from a single `ps` snapshot,
  root first, robust to pid-reuse cycles.
- `_kill_step_tree` — snapshot the tree pids BEFORE signalling (killing
  reparents/removes members), `TERM` them all, poll up to
  `GPU_WINDOW_KILL_GRACE_SECONDS` (default 2 s), then `KILL` any survivor; reap
  STEP_PID. `_kill_step_child` (the RSS-cap / ceiling abort call sites) delegates
  to it, and the `teardown` trap tree-kills too.

Qwen is still restored and the exclusive lock released from the EXIT trap exactly
as before (restore before release, so a queued window never races the reload).

---

## Peak-memory directive — report the whole-process peak, not only MLX

David: "fix the peak memory measurement so it includes non-Metal parts."

- The `[ab]` console headline prints `peak_gb` (MLX allocator peak, unchanged for
  comparability) alongside `mlx_peak_gb`, `process_peak_rss_gb` and
  `system_used_peak_gb` (helper `_memory_headline`).
- Both the AR and the DSpark receipts carry a top-level **`peak_process_gb`** (the
  whole-process RSS peak incl. non-Metal) next to the MLX-only `peak_gb`.
- The in-process sampler (item 2) brackets prefill+decode and keeps the
  high-water mark, so the receipt has a real peak even when NOT run under
  `gpu_window.sh`; the wrapper's process-tree RSS is the second source (recorded
  in the window log). The value is the PEAK over the run, not the value at exit.

---

## Output persistence — store the output so we can audit it

Every bench run persists its FULL generated output (rounding-class results must be
text-spot-checkable, not just sha-comparable):

**In the receipt** — for the AR pass (top level) and, under the `dspark` block,
for BOTH the DSpark stream and the AR comparison stream it verifies against:
`token_ids` (the full id list), `decoded_text`, `decoded_text_head` (first 600
chars), `decoded_text_tail` (last 600 chars). For DSpark,
`dspark.divergence_context = {ar, dspark}` is the decoded text ~200 chars either
side of the divergence index in each stream (null when identical).

**As a text sidecar** beside the receipt (`<stem>` = `--out` with `.jsonl`
stripped):
- `<stem>.output.txt` — header (arm, sha, decode_tok_s, divergence) + the full
  decoded text of the measured stream (the DSpark stream in dspark mode, else AR).
- `<stem>.ar-reference.output.txt` — the AR comparison stream (dspark runs only).

Sidecars are written **atomically** (tmp + `os.replace`) and **never overwrite**
an existing sidecar (suffix `-2`, `-3`, matching the receipts /
never-overwrite-a-measurement). Decoding reuses the tokenizer already loaded in
the bench (no extra model load) and is fully guarded: a tokenizer failure records
`None` / `"<decode unavailable>"` and never kills the measured run.

---

## Tests (CPU, MLX pinned to CPU; `nice -n 19`, one file per process)

- `tests/test_deepseek_v41_w106_memory_budget.py` — the item-3 derivation math
  (injected measurements), floor refusal, the exact receipt key set, the KV
  estimator, the config reader, and `_resolve_derivation` end-to-end.
- `tests/test_deepseek_v41_w106_memory_block.py` — the item-2 sampler + memory
  block, plus the directive's `_peak_process_gb` / `_memory_headline` and a
  sampler high-water (peak-not-exit) test.
- `tests/test_deepseek_v41_w106_receipt_output.py` — decode guards, token_ids /
  head / tail, divergence-context spans, atomic no-clobber sidecar writes.
- `tests/test_gpu_window_memory_accounting.sh` — the item-1 tree-RSS accounting
  plus the item-4 tree-kill (a fake `bash -c` step spawns a `sleep` grandchild;
  after abort neither the python child nor the grandchild survives), in
  `GPU_WINDOW_TEST_MODE=1` with a temp lock + fake `vm_stat` (no sysctl, no
  launchctl, no real GPU lock, no Metal).
