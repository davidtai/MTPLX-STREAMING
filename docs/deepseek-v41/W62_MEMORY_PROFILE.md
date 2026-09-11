# W62 — DeepSeek-V4.1 explicit, self-limiting memory profile

David's directive: *"fix the memory profile if we're stuck in paging."*  Window
28 ran a hand-picked **92 GiB** plan, the box sat at **102 GB** with 0.1 GB free
and 7 GB already in the compressor, and a 1K decode step that takes ~2 min at 82
GiB ran >20 min — the process was paging.  This turns the plan from a
hand-picked number into a value **derived from a TOTAL box budget**, bounds the
one memory sink the plan never priced (the MLX allocator's freed-buffer cache),
and instruments every phase so a window can see where the bytes went.

## 1. Where the bytes go — plan vs peak vs box

During a GPU window the resident Qwen server is booted out, so the DeepSeek-V4.1
process plus macOS is *all* that runs.  Every term below is resident
simultaneously:

```
box_total  =  macOS_floor        (~6 GiB kept free/file-backed, or the
                                   compressor starts — measured at window 28)
           +  mlx_active         (resident weights + KV + live transients;
                                   bounded by set_memory_limit = plan − reserve)
           +  mlx_cache          (MLX allocator's FREED-buffer retention;
                                   was UNBOUNDED on this lane — item 2)
           +  host_overhead      (python heap, positional bank read buffers,
                                   engram host-side row LRU ~2 GiB, tokenizer;
                                   NOT counted by mx.get_peak_memory)
```

The crucial gap David measured: **8–12 GB of process RSS lives outside the
plan.**  `mx.get_peak_memory` only sees MLX buffers, so it read 75.9–77.7 GB at
an 82 GiB plan while the box sat at ~88 GB.  Two things are outside its view:

* **host_overhead** — non-MLX process memory (heap, bank read buffers, the
  engram host-side row LRU, tokenizer). ~10 GiB.
* **mlx_cache** — the allocator keeps freed buffers for reuse. `set_memory_limit`
  (the only cap this CLI/bench lane set) bounds *active* allocation, not the
  freed cache, whose default tracks the memory limit (~0.75× RAM). Freed decode
  and prefill transients accumulated there for the process lifetime — real RSS
  the plan never priced.

The runtime **reserve** (7 GiB, inside the plan) is headroom for *prefill
transients*, and it holds (item 4): the K30 selected-key attention makes the
16K score transient T-independent, and the K26 dense-expert prefill batch is
bounded to ~0.57 GB.

## 2. The derivation — plan from a TOTAL budget

`mtplx/deepseek_v41_memory_profile.py :: derive_plan_from_budget`:

```
plan  =  box_budget − macOS_floor − host_overhead − allocator_cache_limit
```

At David's default budget of **100 GB**:

```
plan = 100 − 6 (macOS floor) − 10 (host overhead) − 6 (allocator cache) = 78 GiB
     = 83,751,862,272 bytes
allocator cache limit = 6 GiB = 6,442,450,944 bytes   (mx.set_cache_limit)
runtime reserve       = 7 GiB = 7,516,192,768 bytes   (in-plan, for transients)
```

Each subtracted term maps to a distinct RSS component so the box total closes to
the budget without double-counting: `box = macOS_floor + mlx_active(≤ plan −
reserve) + mlx_cache(≤ cache_limit) + host_overhead`.

Knobs (env, all optional; unset → the profile constants above):

| Env | Meaning | Default |
|---|---|---|
| `MTPLX_DSV41_BOX_BUDGET_GB` | David's TOTAL box-use budget | `100` |
| `MTPLX_DSV41_MACOS_FLOOR_GB` | macOS + file-cache floor | `6` |
| `MTPLX_DSV41_HOST_OVERHEAD_GB` | process RSS above MLX | `10` |
| `MTPLX_DSV41_MLX_CACHE_LIMIT_GB` | freed-buffer cache bound | `6` |

`--memory-limit-gib` stays as an explicit **override** (the plan is pinned; the
cache-limit fix still applies).  Both bench scripts print the derivation at load
(`[bench] memory derivation: plan = budget(100) − macOS_floor(6) −
host_overhead(10) − cache_limit(6) = 78 GiB`) and the served child inherits the
budget knob via `apply_expert_profile_child_env` (advisory; it never stamps the
load-bearing `MTPLX_MEMORY_LIMIT_BYTES`).

## 3. The plan breakdown at 78 GiB

`plan_breakdown(runtime.plan)` on the real spec (text-only AR, no MTP head):

| Term | Bytes | GiB | Note |
|---|---:|---:|---|
| residents (text-only wired) | — | ~8–10 | full residents 25.16 GB minus the MTP+vision text-only discount; exact value is `plan.resident_bytes` |
| engram host-side row LRU | 2,147,483,648 | 2.00 | lives in **host_overhead**, not `resident_bytes` |
| KV planned (1K) | 3,276,800 | 0.003 | `1024 × 3200 B/token` |
| KV planned (16K) | 52,428,800 | 0.049 | `16384 × 3200 B/token` |
| runtime reserve | 7,516,192,768 | 7.00 | prefill-transient headroom |
| transient service slots | 66,355,200 | 0.06 | `top_k` service banks |
| expert cache (remainder) | ~50.9 GB | ~47 | fills the plan after the fixed side |

`fits_fixed = True`; ~115 persistent expert slots per layer at the 78 GiB plan.
KV is negligible at both context cells, so **the recommended plan is 78 GiB for
1K and 16K alike** — the 16K prefill's larger transient is bounded by K30/K26
(item 4) and covered by the 7 GiB reserve; only the ~49 MB extra KV differs.

## 4. Prefill transients are bounded (checked by inspection)

* **16K score transient — K30 selected-key gather.**
  `mtplx/models/deepseek_v41.py` (`_resolve_selected_keys`, `MTPLX_DSV41_SELECTED_KEYS`):
  the masked-full path's `[rows, H, T]` score transient grows with context; the
  K30 path gathers only the selected keys per query, bounding it to
  `[rows, H, window + index_topk]` — **T-independent**. Armed by the `dspark`
  arm's `setdefault`.
* **Dense-expert prefill batch — K26.**
  `mtplx/models/expert_mlx.py` (`_gather_component_bank_dense`,
  `MTPLX_DSV41_PREFILL_DENSE_EXPERTS`): at the default batch of 8, at most 8
  dequantized bf16 expert copies (~71 MB each) are live at once (~**0.57 GB**),
  with an `mx.eval` between batches so none survives the call.

Both fit well inside the 7 GiB reserve, so the derivation prices them as a
reserve line (`prefill_transient_gib`), and `derive_plan_from_budget` guards
that `plan > reserve + prefill_transient`.

## 5. The allocator overhang fix

`apply_allocator_cache_limit(cache_limit_bytes, mx_module)` calls
`mx.set_cache_limit` from the plan (6 GiB) — the same mechanism the served path
already runs in `_configure_mlx_cache_limit` (`mtplx/server/openai.py`), applied
to the CLI/bench lane that previously set only `set_memory_limit`. Freed decode
and prefill transients now return to the OS past 6 GiB instead of accumulating
for the process lifetime, keeping mlx_cache inside the reserve. Both bench
scripts apply it after load (guarded by `--apply-memory-cap`) and record the
report in the receipt.

## 6. Instrumentation

`memory_profile_snapshot(phase, token, plan, runtime, mx_module, derivation)`
gathers, at a named phase:

* **MLX** — `get_active_memory` / `get_cache_memory` / `get_peak_memory`.
* **process** — current RSS + `phys_footprint` via `mach` `task_info`
  (`TASK_VM_INFO`, **psutil-free**), plus peak `ru_maxrss`.
* **box** — `vm_stat` wired / anon-active / compressor / free.
* **plan** — the breakdown above, plus the live expert-cache allocated bytes
  from `runtime.snapshot()`.

Wired into both bench scripts under `--memory-profile` (snapshots at load end,
after prefill, and every `--memory-profile-every` decode tokens), rendered into
the receipt as a table via `format_memory_profile_table`. The served
`mtplx_openai_generation` event carries `peak/active/cache_memory_bytes` at
request end (all three scheduler lanes).

## 7. The window command

Derives 78 GiB from the default 100 GB budget — no hand-picked ceiling:

```bash
cd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w62
PYTHONPATH=$PWD nice -n 19 .venv/bin/python3 \
  scripts/deepseek_v41/ab_decode_env_levers.py \
  --arms control \
  --context-tokens 1024 --decode-tokens 256 \
  --memory-profile \
  --out docs/deepseek-v41/receipts/w62_memprofile_1k.jsonl
```

Standard-shape (1K + 16K in one process):

```bash
PYTHONPATH=$PWD nice -n 19 .venv/bin/python3 \
  scripts/deepseek_v41/bench_standard_shape.py \
  --context-tokens 1024 16384 --steps 256 \
  --memory-profile \
  --out-dir docs/deepseek-v41/receipts
```

To A/B a different budget without editing a script:
`MTPLX_DSV41_BOX_BUDGET_GB=96 … ab_decode_env_levers.py …` (→ 74 GiB plan), or
pin the old behaviour with `--memory-limit-gib 82`.

## Recommended plans under a 100 GB budget

| Context | Plan | `memory_limit_bytes` | cache limit | reserve |
|---|---:|---:|---:|---:|
| 1K (1,024) | **78 GiB** | 83,751,862,272 | 6 GiB | 7 GiB |
| 16K (16,384) | **78 GiB** | 83,751,862,272 | 6 GiB | 7 GiB |

Same plan both cells; 16K only adds ~49 MB of KV and a K30/K26-bounded prefill
transient that the reserve already covers. Expected box total: macOS floor (~6)
+ mlx_active (≤ plan − reserve ≈ 71) + mlx_cache (≤ 6) + host_overhead (~10) ≈
**93 GB**, a deliberate margin below the 100 GB budget and well clear of the
~110 GB panic zone.
