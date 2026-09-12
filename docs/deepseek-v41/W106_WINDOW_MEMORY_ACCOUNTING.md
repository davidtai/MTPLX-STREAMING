# W106 — window memory accounting (budget, non-Metal peak, tree-kill, output)

The box has a **110 GB hard ceiling** and David's budget is **100 GB TOTAL for
everything** (agent + workers + the GPU step, together). W106 makes the
DeepSeek-V4.1 bench + guard account for that honestly: the plan is *derived from*
the total budget while compensating for the non-Metal requirements, the receipts
report the whole-process peak (not just the MLX allocator peak), the guard kills
the entire step process tree on abort, and every run persists its full output for
a text audit.

Files:
- `scripts/deepseek_v41/ab_decode_env_levers.py` — the A/B bench (item 3, the
  peak-memory directive, output persistence).
- `scripts/deepseek_v41/bench_standard_shape.py` — the shared memory probe /
  sampler (`_MLXMemProbe`, `_MemorySampler`, `_system_used_bytes`,
  `_phys_footprint_bytes`); item 2 + MEDIUM-2.
- `scripts/deepseek_v41/gpu_window.sh` — the guarded GPU window (items 1, 4).

## Units — defined ONCE

- **All flags** that name a memory quantity are **GiB** and end in `-gib`
  (`--memory-budget-total-gib`, `--memory-safety-gib`, `--non-metal-overhead-gib`,
  `--memory-budget-floor-gib`). The `-gb` spellings are **deprecated aliases**:
  their value is **decimal GB** and is converted to GiB at the boundary
  (`GiB = GB × 1e9 / 2^30`), with a warning. Passing both forms of one quantity is
  refused.
- **All receipt memory values** (`*_gb` and `*_gib` keys alike) are **GiB**
  (bytes / 2^30). The legacy `_gb` key spellings are kept for old-receipt
  comparability; the value is GiB.
- **`gpu_window.sh` guard caps** are GiB: 1 GiB = 2^30 bytes. Their `~GB`
  equivalents are printed alongside for the operator.
- 1 GiB ≈ 1.0737 GB. So 100 GB ≈ 93.13 GiB, 102 GiB ≈ 109.5 GB, 93 GiB ≈ 99.9 GB.

## Definitions (every memory term, once)

### Peak / envelope figures (receipt `memory` block + top level)

- **`mlx_peak_gb`** (== the legacy top-level **`peak_gb`**) — the MLX allocator
  peak of *this* process (`mx.get_peak_memory`). EXCLUDES the Python heap, the
  positional-expert bank read buffers, the engram host LRU, the tokenizer, other
  processes, and the OS file cache. It is **not** the box usage. Kept as-is for
  old-receipt comparability; documented as MLX-only.
- **`sampler_peak_rss_gb`** — the 1 Hz in-process sampler's peak of the CURRENT
  process footprint (`phys_footprint`, mach `task_info`), **bracketed over
  prefill+decode**; `null` when no sampler ran. (MEDIUM-2.)
- **`ru_maxrss_gb`** — the process LIFETIME peak RSS (`ru_maxrss`), which also
  spans model load and earlier arms. (MEDIUM-2.)
- **`process_peak_rss_gb`** (top-level alias **`peak_process_gb`**) — the process
  peak over the RUN: the **sampler peak when it is non-zero, else `ru_maxrss`**
  (LOW: a sampler that ran but got no reading yields 0, which falls back to
  `ru_maxrss` rather than a misleading 0.0). A single, decomposable meaning (no
  `max()` blob). David's "peak memory must include the non-Metal parts" figure; a
  PEAK over the window, never the value at exit.
- **`system_used_peak_gb`** — the whole-**box** used-memory peak over the run:
  `(wired down + anonymous + occupied-by-compressor) pages × page size` from
  `vm_stat` — the SAME formula the `gpu_window.sh` phase-4 guard aborts on
  (anonymous, not active). Shared helper `bench._system_used_bytes`.
- **`system_used_at_start_gb`** — the same box figure sampled once when the
  sampler starts (the decode baseline).
- **`rss_semantics_note`** — a fixed note (HIGH-2): RSS-vs-`mlx_peak` semantics on
  Metal are **UNVERIFIED** until one real GPU-window receipt lets the
  `gpu_window.sh` tree RSS be compared against this process's `mlx_peak` (unified
  memory may double-count). Do not treat `process_peak_rss_gb` and `mlx_peak_gb`
  as interchangeable until then.

Two independent sources record the process peak, and **both** are kept: the
in-process sampler (in the receipt, always present) and the wrapper's
process-tree RSS (in the `gpu_window.sh` log).

### Budget-derivation terms (item 3; receipt `memory` block)

- **`budget_total_gb`** — `--memory-budget-total-gib N` (or the `-gb` alias): the
  TOTAL box budget for everything, in GiB.
- **`budget_system_used_at_start_gb`** — the box used-memory baseline measured ONCE
  at process start (`bench._system_used_bytes`, the wrapper's formula).
- **`budget_non_metal_overhead_gb`** — the process footprint *above* the MLX
  allocator (Python heap + expert-reader buffers + engram LRU + tokenizer). Used
  as a **conservative pre-load estimate** (default 10 GiB,
  `--non-metal-overhead-gib`) because the plan must be fixed before the model
  loads; see the two-phase note.
- **`budget_non_metal_overhead_measured_gb`** — the same overhead **re-measured
  after load** as `current phys_footprint (mach task_info) − mx active − mx CACHE`
  (HIGH-A: the MLX freed-buffer cache is load-transient Metal memory, not non-Metal
  overhead; NOT `ru_maxrss`); `null` until measured / when unmeasurable.
- **`rss_semantics`** — the state of that re-measurement (HIGH-A): `"ok"`
  (footprint ≥ active, measured valid), `"inverted"` (footprint < mx active, so
  Metal is not in `phys_footprint` on this platform and the overhead is
  unmeasurable — recorded `null`, never a bogus 0.0, and never aborts), or
  `"unmeasured"` (pre-load / footprint unavailable). Distinct from the fixed
  `rss_semantics_note`.
- **`budget_kv_growth_to_max_kv_gb`** — the bytes the KV lanes grow to at
  `--max-kv`, from config dims × max_kv × bf16 (see the KV estimator note).
- **`budget_safety_gb`** — `--memory-safety-gib` headroom (default 3 GiB).
- **`budget_floor_gib`** — `--memory-budget-floor-gib` (default 20 GiB).
- **`plan_limit_gib_derived`** — the plan limit the derivation produced (pre-load).
- **`plan_limit_gib_effective`** — the plan limit in force after the two-phase
  re-measure. Always equals `plan_limit_gib_derived` (HIGH-1: the MLX limit is
  never lowered post-load).
- **`memory_plan_source`** — `"budget"` when `--memory-budget-total-gib`/`-gb`
  drove the plan, else `"explicit"`. The full budget key set is always present
  (nulls on the explicit path), so receipts are self-describing.

### Guard terms (`gpu_window.sh`)

- **step tree RSS (sum)** — SUM of RSS across STEP_PID and every descendant.
- **max single process** — the MAX single-process RSS in that tree.
- **step process tree** — STEP_PID (the `bash -c "…"` chain) AND every descendant.

## Item 3 — `--memory-budget-total-gib`: derive the plan, compensate for non-Metal

Instead of taking `--memory-limit-gib` literally, derive the MLX plan limit from
the TOTAL box budget:

```
plan_limit_gib = budget_total
               − system_used_at_start      (measured at process start, vm_stat)
               − non_metal_overhead        (pre-load estimate; re-measured after load)
               − kv_growth_to_max_kv        (config dims × max_kv × bf16)
               − safety                     (default 3 GiB)
```

`--memory-budget-total-gib` (or the deprecated `-gb` alias) **overrides**
`--memory-limit-gib`. The derived limit is fed into the W62
`derive_plan_from_budget(..., override_memory_limit_gib=…)`, so the runtime
reserve / allocator-cache plumbing is unchanged; only the plan ceiling changes.

**Floor refusal.** If `plan_limit_gib < --memory-budget-floor-gib` (default 20) the
run refuses to start with a clear, actionable error (naming every term). Use
`--memory-plan-preflight` (below) to hit this **before** the GPU window opens.

**Two-phase non-Metal overhead (HIGH-1).** The plan limit must be fixed *before*
the model loads. So:
1. **Estimate** — derive with the conservative `--non-metal-overhead-gib` estimate
   (default 10 GiB) and fix the plan.
2. **Load** the model.
3. **Re-measure** the real overhead as `current phys_footprint (mach task_info) −
   mx active − mx CACHE` and record it (`budget_non_metal_overhead_measured_gb` +
   `rss_semantics`). Subtracting the MLX freed-buffer **cache** (HIGH-A) is
   essential: it is load-transient Metal memory the allocator reuses, and counting
   it caused false aborts inside an open window. `mx.clear_cache()` is deliberately
   NOT called (it would perturb the first decode token); subtracting the cache is
   the non-perturbing equivalent. If `phys_footprint < mx active` the overhead is
   unmeasurable on this platform → `rss_semantics="inverted"`, measured `null`
   (never a bogus 0.0), no abort. The MLX active limit is **never** lowered
   post-load (residents are already allocated; a limit below active memory would
   route later allocations onto the over-limit path and perturb the decode).
   Otherwise, if the measured overhead exceeds the estimate by more than 0.5 GiB —
   the real footprint would blow the budget — the run **ABORTS with a clear error
   before the decode starts** (recorded via MEDIUM-C).

**KV growth estimator (LOW-1).** `_kv_bytes_at_max_kv(config, max_kv)` is a LOCAL
estimate: every layer's sliding-window ring is priced at the full `max_kv`, and
only the **kv-source layers** (`kv_source_layer_ids`, released `[2,8,14,20]`; else
layers with a non-zero `compress_ratios`) add the latent `[max_kv/ratio, head_dim]`,
rope `[max_kv/ratio, qk_rope_head_dim]`, and index `[max_kv/ratio, index_head_dim]`
lanes, all bf16. It is "conservative" ONLY while the sliding-window ring is off (it
prices the window lane at the full `max_kv`); it models **no fp32 latent frontier**
and so is NOT conservative in general — e.g. it estimated ~774 MB (61 MB non-window)
where W107's exact helper reports ~320 MB for the same cell. Config dims are read
from `config.json` (flat or `text_config`-nested) without importing MLX.
**TODO(W107):** the int-branch already swaps this for
`mtplx.models.deepseek_v41_cache.kv_bytes_at_max_kv` when it lands — this local
estimate is a placeholder, not the authority.

**Receipt keys** (in the `memory` block, alongside the peak/envelope keys):
`memory_plan_source`, `budget_total_gb`, `plan_limit_gib_derived`,
`plan_limit_gib_effective`, `budget_system_used_at_start_gb`,
`budget_non_metal_overhead_gb`, `budget_non_metal_overhead_measured_gb`,
`budget_kv_growth_to_max_kv_gb`, `budget_safety_gb`, `budget_floor_gib`,
`rss_semantics`, `rss_semantics_note`.

**Abort ledger (MEDIUM-C).** If an arm raises (a budget re-measure abort or a floor
refusal), `main()` appends an abort row `{"arm", "aborted": true, "reason",
"stage", "exception"}` (stage: `budget_remeasure` / `budget_derivation` /
`run_arm`; plus the budget snapshot) to `--out` and exits with code **4**, so the
ledger records the failure rather than leaving a silent gap.

**Pre-flight (LOW-4 / HIGH-B).** `--memory-plan-preflight` derives the plan from a
dry snapshot (no model load, no MLX) and exits 0 (plan ≥ floor) or 3 (below floor),
so the budget is checked before the guarded window opens. The "now" baseline still
has the resident agent (~45 GiB) + workers resident, but the window boots the agent
out first, so `--preflight-freed-gib N` (default: best-effort READ-ONLY auto-detect
of the com.tea.qwen RSS via `launchctl print` + `ps`, else 0 with a caveat) is
subtracted; both the "now" and "expected in-window" baselines are printed, and the
in-window derivation (measured after bootout) is authoritative. A both-flag-forms
error exits 3 (not a traceback). NB: a `bash -c "a; b; c"` step chain continues to
`b` after `a` fails, so the pre-flight (a separate step that exits nonzero) is the
reliable gate, not an in-chain failure.

**Bench command line — item 3 on the 16K cell** (do not run here; a GPU window is
held by another worktree). Canonical GiB form (~100 GB budget):

```
.venv/bin/python3 scripts/deepseek_v41/ab_decode_env_levers.py \
  --context-tokens 16384 --decode-tokens 256 --max-kv 17408 \
  --memory-budget-total-gib 93 \
  --out <receipt>.jsonl
```

(`--memory-budget-total-gb 100` — the decimal-GB alias — resolves to the same
~93.13 GiB. `--memory-safety-gib` / `--memory-budget-floor-gib` /
`--non-metal-overhead-gib` default to 3 / 20 / 10 GiB. Add `--memory-plan-preflight`
to check the derivation and exit without loading the model.)

## Item 4 — tree-kill the whole step process tree on abort

On any abort — the child-tree RSS cap, the system-used ceiling, or a TERM/INT to
the wrapper — `gpu_window.sh` kills the ENTIRE process tree of the step, not just
STEP_PID (the `bash -c` chain), so no python descendant is orphaned and a chained
next command never starts after the abort.

- `_step_tree_pids <root>` — every pid in the tree from a single `ps` snapshot.
- `_kill_step_tree` — snapshot the tree pids BEFORE signalling, `TERM` them all,
  poll up to `GPU_WINDOW_KILL_GRACE_SECONDS` (default 2), then `KILL` any survivor;
  reap STEP_PID. `_kill_step_child` (the RSS-cap / ceiling abort sites) and the
  `teardown` trap both delegate to it. `GPU_WINDOW_KILL_GRACE_SECONDS` is validated
  as a non-negative integer (LOW-2: it drives `× 4` bash arithmetic) — a
  non-integer warns and falls back to 2.

Qwen is still restored and the lock released from the EXIT trap (restore before
release).

## Guard caps (HIGH-2, MEDIUM-A, MEDIUM-B) — under David's limits

- System used-memory **ceiling = 102 GiB** (~109.5 GB, under the 110 GB hard
  line). Default was 105 GiB ≈ 112.7 GB, which was OVER the hard line.
- Child-tree **RSS cap = 93 GiB** (~100 GB = David's "100 GB total"). This cap is
  LIVE for the first time under W106 (pre-W106 the poll read the ~0 `bash -c`
  shell RSS). Both caps are printed explicitly (GiB + ~GB) at step start.
- **MEDIUM-A:** `GPU_WINDOW_TOTAL_MEM_CEILING_GB`, `GPU_WINDOW_MIN_AVAIL_GB` and
  `GPU_WINDOW_FOREIGN_WORKER_RSS_GB` are integer-validated **before phase 0** (a
  fractional value would leave the `$(( GB × 1024³ ))` byte var unset and, under
  `set -u`, kill the window after Qwen is booted out) — a non-integer warns and
  falls back to the default, like `GPU_WINDOW_KILL_GRACE_SECONDS`. (These
  historical `*_GB` env names carry **GiB**; see the Units section.)
- **MEDIUM-B:** the effective child-tree cap is related to the measured baseline:
  when `used_start + child_cap > ceiling`, it is lowered (never raised, never
  refused) to `ceiling − used_start` and logged as "effective child-tree cap X
  GiB"; the phase-4 poll aborts on that effective cap.

## Peak-memory directive — report the whole-process peak, not only MLX

- The `[ab]` headline prints `peak_gb` (MLX allocator peak, unchanged) alongside
  `mlx_peak_gb`, `process_peak_rss_gb` and `system_used_peak_gb`.
- Both the AR and DSpark receipts carry a top-level `peak_process_gb`.
- The in-process sampler brackets prefill+decode (peak, not exit) and samples
  `phys_footprint`, so the receipt has a real peak even without `gpu_window.sh`.

## Output persistence — store the output so we can audit it

**In the receipt** — AR pass (top level) and, under the `dspark` block, both the
DSpark stream and the AR comparison stream: `token_ids` (full), `decoded_text`,
`decoded_text_head` (first 600), `decoded_text_tail` (last 600). For DSpark,
`dspark.divergence_context = {ar, dspark}` is the decoded text ~200 chars either
side of the divergence index in each stream (null when identical).

**As text sidecars** beside the receipt (`<stem>` = `--out` with `.jsonl`
stripped), with the AR pass **sha[:12] in both names** and a **single paired
`-n` suffix** (MEDIUM-3), so the pair never splits:
- `<stem>.<sha12>.output.txt` — the measured stream (DSpark in dspark mode, else
  AR): header (arm, sha, decode_tok_s, divergence) + full decoded text.
- `<stem>.<sha12>.ar-reference.output.txt` — the AR comparison stream (dspark
  runs only).
Written atomically (tmp + `os.replace`) and **never** overwriting an existing
sidecar. Decoding reuses the already-loaded tokenizer and is fully guarded (a
failure records `null` / `"<decode unavailable>"` and never kills the run).

## Tests (CPU, MLX pinned to CPU; `nice -n 19`, one file per process)

- `tests/test_deepseek_v41_w106_memory_budget.py` — derivation math (injected
  measurements), floor refusal, the exact receipt key set, the KV estimator
  (kv_source_layer_ids), the config reader, `_resolve_derivation`; the HIGH-1/HIGH-A
  re-measure (never `set_memory_limit`/`clear_cache`; subtracts the mx cache so no
  false abort; `inverted` → `None`, never 0.0); the HIGH-B pre-flight freed baseline
  (rc 0 with `--preflight-freed-gib`, rc 3 without, rc 3 on both-flag-forms); the
  MEDIUM-1 `-gib`/`-gb` resolution; the HIGH-2 `rss_semantics_note`; and the
  MEDIUM-C abort ledger row (stage + reason + append).
- `tests/test_deepseek_v41_w106_memory_block.py` — the item-2 sampler + the
  MEDIUM-2 decomposed keys (`sampler_peak_rss_gb` / `ru_maxrss_gb` /
  `process_peak_rss_gb`), the LOW 0-sampler-peak fallback to `ru_maxrss`,
  `_peak_process_gb` / `_memory_headline`, and a high-water (peak-not-exit) test.
- `tests/test_deepseek_v41_w106_receipt_output.py` — decode guards, token_ids /
  head / tail, divergence-context spans, and the MEDIUM-3 paired sha-named
  sidecars.
- `tests/test_gpu_window_memory_accounting.sh` — item-1 tree-RSS accounting, the
  item-4 tree-kill (a `sleep` grandchild does not survive the abort), the MEDIUM-B
  effective-cap lowering (52 GiB) + 102 GiB ceiling, the LOW-2 non-integer-grace
  fallback, and the MEDIUM-A fractional-ceiling validation (warn + fall back to 102,
  clean exit) — in `GPU_WINDOW_TEST_MODE=1` with a hermetic temp lock (no sysctl,
  no launchctl, no real GPU lock, no Metal).
- `scripts/deepseek_v41/test_gpu_window_guard.sh` — the phase-4 guard math + the
  HIGH-2 102 GiB default ceiling.
