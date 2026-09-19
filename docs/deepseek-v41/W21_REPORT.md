# W21 report — DeepSeek-V4.1-Flash mxfp4 serve profile + text-only planner pricing

Scope: make `mtplx serve --model ~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4`
resolve a correct, well-tuned config **with no flags** on the MTPLX runner; price
only the residents a text-only AR forward wires; handle the mxfp4 selfcheck
signature; and document the served-path measurement with the repo's tooling.

Branch `feat/deepseek-v41-w21` off integration `5d6dd8ad`. CPU-only; no GPU lock,
no `:8080`, no launchd; peak worker RSS **77 MB** (`/usr/bin/time -l`, the pytest
sweep). No real-artifact model load (manifest JSON reads only, for byte census).

Allowed-writes touched: `mtplx/expert_runtime.py`, `mtplx/runtime.py`,
`mtplx/kernel_selfcheck.py`, `mtplx/models/deepseek_v41_loader.py`,
`mtplx/data/expert_profiles.json`, `mtplx/expert_cli.py` (registry-derived
`--expert-profile` choices — server config for default-profile selection),
`tests/test_deepseek_v41_serve_profile.py`, `tests/test_expert_profiles.py` (the
registry-set assertion, a mechanical consequence of adding a profile),
`scripts/deepseek_v41/README.md`, this report.

---

## 1. Resolved default config (`mtplx serve --model <mxfp4>`, no flags)

`--expert-profile auto` selects the new promoted profile **`deepseek-v41-mxfp4-75`**
by model key (`deepseek-v41-flash-expert-mxfp4`) and `build_expert_streaming_config`
resolves it to:

| Field | Value | Reason |
|---|---|---|
| `memory_limit_bytes` | **82 GiB** (88,046,829,568) | Coordinator 2026-09-10: 82 GiB planner default on the 100 GB box (5 workers share it; ≤ the 100 GiB `iogpu.wired_limit_mb`, never raised). |
| `runtime_reserve_bytes` | **7 GiB** (7,516,192,768) | The promoted-streaming reserve (docs/advanced/ssd-streamed-moe.md). W16 measured peak 59.5 GB at a 72 GiB limit → the 7 GiB reserve was never breached (12.5 GiB total headroom). |
| `weight_envelope_bytes` | 75 GiB (80,530,636,800) | Profile-name convention = envelope = limit − reserve. `envelope + reserve == process_ceiling` (82 GiB). |
| `max_live_kv_tokens` | **16384** | David's 16,384-token prefill cell; KV stays bf16 (`kv_quant=None`) → 52.4 MB at 16k (3,200 B/token). |
| `cache_policy` / `cache_scope` | **lru / layer** | W24 routing census has NOT landed on this branch (`feat/deepseek-v41-w24`, no `routing_census*.json`); per the task, LRU/layer is the safe default. Frequency may win once the census lands. |
| `slot_layout` | **component-banks** | mxfp4 is non-affine; only the component-banks dispatch reads it. Derived from the codec in `__post_init__` even with no explicit flag. |
| `transient_slots` | **48** | Streamed-service bank sized for the prefill chunk (default 2,048 tokens); the promoted GLM t158 streamed config's precedent (docs/advanced/ssd-streamed-moe.md). 48 × 18,800,640 B = 0.84 GiB fixed. |
| `bypass_page_cache` | **true** | F_NOCACHE on — the 269 GiB bank must not thrash the OS page cache. |
| `max_read_chunk_bytes` | 8 MiB | GLM streamed precedent; records (17.9 MiB) read in coalesced chunks. |
| `split_route_release` | deferred | Overlap resident-expert compute with miss reads (GLM precedent). |
| `prefetch_slots` | 0 | No routing census yet; prefetch is a census-driven perf lane. |
| `expert_cache_limit_bytes` | **None → derived** | Single-memory-knob (remainder) policy: the runtime recomputes the expert-cache allowance from `memory_limit` at each KV boundary. `derived_expert_cache_policy == True`. |
| islands | **off** | mxfp4 cannot serve dense/mmap islands (`island_layers=()`, `island_layer_count=None`, `mmap_island_layers=()`); `ExpertStreamingRuntime.open` rejects islands for non-affine banks. |
| `verify_record_hashes` / `verify_sidecar_hash_at_open` | false / false | The admission receipt covers bank integrity; open-time hashing is a diagnostic. |
| `streamed_codec` | none | The bank is native mxfp4; no rANS container. |
| child_env `MTPLX_ENGRAM_CACHE_LIMIT` | **2GiB** | Engram resident-row LRU over the 2×101 GB SSD row banks (layers 1, 14); also the loader default. |
| child_env `MTPLX_SESSION_NEAR_PREFIX_RESTORE` | **0** | W22: a KV-only near-prefix restore desyncs the engram hash at layers 1/14 → silent wrong output on warm turns. |
| child_env `MTPLX_SESSION_STORE_ON_PREFILL` | **0** | W22: the SSD prompt-cache store-on-prefill fails closed for this model. |

hy3/glm promoted profiles are **byte-identical** (only a profile was added; test
`test_hy3_promoted_configs_unchanged`).

## 2. Memory plan at 82 GiB with text-only pricing

The W3-flagged hook is fixed: `ExpertStreamingRuntime.open` (pool plan) **and**
`runtime.py`'s pre-flight plan (component-bank slot allocator) now subtract
`text_only_resident_discount(manifest, spec)` — the residents a text-only AR
forward never wires — so they stay in lock-step (bank capacity == pool).

Residents skipped (measured from the shipped mxfp4 `expert-manifest.json`):

| Class | Tensors | Bytes | GiB |
|---|---:|---:|---:|
| **Text kept** (`embed/head/layers/norm`) | 1,246 | **9,729,152,448** | 9.061 |
| Skipped `mtp.*` (mxfp8 dense + mxfp4 experts) | 2,401 | 7,949,968,776 | 7.404 |
| Skipped vision/aligner/image | 266 | 970,536,960 | 0.904 |
| **Total residents** (== `spec.resident_bytes`) | 3,913 | **18,649,658,184** | 17.369 |

**Discount** (MTP off) = mtp + vision = **8,920,505,736 B (8.31 GiB)**. When a
future MTP serve path sets `spec.mtp_included=True`, only vision is discounted.

Plan (`plan_expert_memory`, record 18,800,640 B, 40 layers × 384 experts, top-k 6):

| Context | KV bytes (bf16) | Slots/layer | Persistent slots | Expert cache (GiB) | Fixed (GiB) | Fits | Unalloc (GiB) |
|---:|---:|---:|---:|---:|---:|:--:|---:|
| 0 (open) | 0 | **92** | 3,680 | 64.435 | 16.906 | yes | 0.659 |
| 1,024 | 3,276,800 | 92 | 3,680 | 64.435 | 16.909 | yes | 0.656 |
| 16,384 | 52,428,800 | 92 | 3,680 | 64.435 | 16.955 | yes | 0.610 |
| 65,536 | 209,715,200 | 92 | 3,680 | 64.435 | 17.102 | yes | 0.464 |

Fixed-side breakdown (open-time, MLX-priced):

| Term | Bytes | GiB |
|---|---:|---:|
| Text-only resident + SWA window (5 MiB) | 9,734,395,328 | 9.066 |
| Runtime reserve | 7,516,192,768 | 7.000 |
| Transient (48 slots) | 902,430,720 | 0.840 |
| KV (16,384 bf16) | 52,428,800 | 0.049 |
| **Expert cache (remainder)** | 69,187,015,680 | **64.435** |
| — of which 92 slots/layer × 40 | 3,680 slots | |

Text-only pricing is worth **+11 slots/layer** (92 vs 81 at full-resident
pricing). Engram row cache (2 GiB) is a **host/SSD-backed LRU** over the 2×101 GB
banks, budgeted outside the MLX expert plan. 92 of 384 experts stay resident-cached
per layer; the 269 GiB bank streams — cache is not the throughput lever at B=1
(W16: cold misses), but MTP verify rows + longer sessions need it maximized, which
is what the text-only discount + derived remainder policy do.

## 3. Served-path measurement (repo tooling)

Full commands in `scripts/deepseek_v41/README.md` (§ "W21 — mxfp4 served profile").
All default to the mxfp4 artifact and run inside `gpu_window.sh`:

- **A. serve health + one completion** — `serve_health.sh` (`mtplx serve`, no flags
  → auto profile, HTTP); standard `mtplx bench serve` for the health/metrics smoke.
- **B. David's shape, greedy** — `bench_standard_shape.py --context-tokens 1024
  16384 --steps 256 --memory-limit-gib 82`; reports prefill tok/s, TTFT, decode
  tok/s, peak GB (peak_mlx_gb + process_rss_gb), wall.
- **C. HumanEval(164) at David's sampler** — `humaneval_cell.sh` (temperature 1,
  top-p 0.95, top-k 20, non-binding cap; HTTP, profile-resolved).

**Standard-bench gap (stated, not worked around):** the generic `mtplx bench
run/prefill-ladder --suite/--profile` cannot express David's shape — `mtplx bench
run` (bare) is the manifest-backend scaffold; the promoted battery is an MTP
depth-sweep over a native model (DSV4.1 streamed is AR-only); neither has a
streamed-AR-MoE served-decode-at-fixed-shape harness with the DSV4.1 prompt build +
expert/engram counters. The DSV4.1 drivers exist for exactly that reason; no new
runner was written. `mtplx bench serve` is used as-is for the health smoke.

## 4. Selfcheck (item 2)

`kernel_selfcheck._expert_quant_signature(spec)` already returns **None** for the
mxfp4 codec (the non-affine branch) and never raises. Confirmed correct: the
`expert_gather` lane hard-codes `mx.gather_qmm(mode="affine")`, so it cannot
validate an mxfp4 bank; None cleanly skips the lane. Docstring now names mxfp4
explicitly; tests lock None-for-mxfp4 and the unchanged affine signature.

## 5. Tests

`tests/test_deepseek_v41_serve_profile.py` — **23 passed** (CPU-only, no artifact):
profile auto-selected by model key + resolved config; planner 92 slots/layer at 82
GiB text-only (+11 over full-resident); KV bf16 = 52.4 MB @16k; discount 0 for
hy3-shaped manifests and MTP-aware; skip-prefixes == the loader's filter; mxfp4
selfcheck None without raising + affine unchanged; engram 2 GiB default + env
override; session-bank kill switches off for mxfp4 and untouched for hy3;
registry-derived `--expert-profile` choices. Regression: `test_expert_profiles.py`
+ `test_expert_streaming_models.py` green (68 total, 100%).

## 6. What remains flagged

- **Served throughput not yet measured.** The profile is **memory-plan / census
  promoted**, not throughput-promoted. Evidence: W16 loader-path decode 4.79 tok/s
  @72 GiB, 4.85 @92 GiB (peak 59.5/79.8 GB). The served numbers at 82 GiB await the
  GPU window (held elsewhere); commands are staged (§3). Promote `evidence_commit`/
  `evidence_receipts` to the real bench receipts once the window runs.
- **`bench_standard_shape.py` config delta.** It measures via the serve-path
  *loader* at `--memory-limit-gib` (loader defaults: frequency cache, top_k
  transient, page cache on), not the full profile object. `--memory-limit-gib 82`
  aligns the envelope + slot count (92/layer) but not cache_policy / transient-48 /
  F_NOCACHE. A true profile-config shape number needs an HTTP decode-through-serve
  cell; no standard streamed-AR served-shape harness exists (none written here).
- **Loader secondary allocator.** `deepseek_v41_loader._component_bank_allocator_for`
  (the P1.7 gate / CPU-proof entry, **not** the production serve path) still prices
  full residents — outside this task's loader allowlist ("engram cache limit
  plumbing only"), and harmless because its callers use `expert_cache_limit=0`
  (0 slots). The production path (`runtime.py`) is fixed. Follow-up when that file
  is next in-scope.
- **Session bank off until W26.** `MTPLX_SESSION_NEAR_PREFIX_RESTORE=0` +
  `MTPLX_SESSION_STORE_ON_PREFILL=0` **lift** when W26 serialises the engram hash
  history into the cache state (then warm turns can restore KV without desyncing the
  layer-1/14 engram hashing).
- **W24 routing census.** LRU/layer is the placeholder default; revisit
  cache_policy/scope + prefetch when `feat/deepseek-v41-w24` lands its census.
- **HF manifest identity for `--download`.** The mxfp4 `quant_revision` is
  `unpublished-mxfp4-repack`; a fresh `--download` admission needs the manifest
  re-uploaded with the pinned HF identity (David's call, per W3's Q2 pattern). Local
  admission matches by construction.
- **82 GiB reading.** Used the coordinator's explicit "82 GiB" = 88.05 GB. If David
  meant 82 GB (76.4 GiB), it is a one-line change to `memory_limit_bytes`/
  `process_ceiling_bytes`/`weight_envelope_bytes` in the profile.
- **Process ceiling vs a busy box.** `process_ceiling` 82 GiB (88 GB) must fit
  installed **and** launch-available RAM; with 5 workers resident, `auto` selection
  fails loud. Serve in a dedicated window (no CPU workers), as the memory rules require.

---

## W21 follow-up (rebased onto integration 106fdd48e = W21 + W23 + W26)

Three coordinator-directed changes on top of the merged W23 (DSpark MTP) + W26
(engram history rides the entry-0 cache state):

### 1. Session bank re-enabled (W26)

W26's `restore_cache` now rewinds the engram alongside the KV, so an in-memory
near-prefix restore no longer desyncs the layer-1/14 engram hash. The profile
therefore **drops** `MTPLX_SESSION_NEAR_PREFIX_RESTORE=0` and
`MTPLX_SESSION_STORE_ON_PREFILL=0` from `child_env` (both return to the engine
default = on); only `MTPLX_ENGRAM_CACHE_LIMIT=2GiB` remains. The SSD prompt-cache
cold tier stays off **separately** (serve `--ssd-session-cache off`; None-KV lanes
+ LayerAttentionCache registration are still unsupported) — not via these knobs.

### 2. `--generation-mode mtp` → `with_mtp=True` serve glue (no env)

`mtplx serve --model <mxfp4> --generation-mode mtp` now serves the DSpark native
MTP head with no `MTPLX_DSV41_MTP` step:

- **expert_cli**: `--generation-mode mtp` is allowed **only** for the native-MTP
  artifact (`is_deepseek_v41_mtp_config` + the manifest ships `mtp.*` residents);
  every external-MTP (hy3/glm) streamed profile keeps the AR-only rule. The
  streamed load kwargs return `mtp=native_mtp` instead of a forced `False`.
- **runtime.py**: a native streamed-MTP load skips the external-MTP
  (`_streamed_mtp_backend`, which needs `mtp_artifacts`) path, and threads
  `with_mtp` to the loader via `construct_resident_model`. The head-injection
  dispatch (`is_deepseek_v41_mtp_config` → `inject_deepseek_v41_mtp_support`)
  publishes it.
- **resident_loader**: `construct_resident_model(with_mtp=)` → the DeepSeek loader.

`--generation-mode ar` (default) is unchanged: `mtp=False`, text-only load, no MTP
residents. Every change is gated on the native-MTP predicate, so the AR and
hy3/glm paths are byte-identical (149 CPU tests green, incl. the AR-only regression).

### 3. MTP planner pricing

When MTP is served the loader keeps **all** `mtp.*` residents
(`partition_text_residents(with_mtp=True).kept` measured = 17,679,121,224 B), so
runtime.py swaps the spec to `mtp_included=True` before the pre-flight and pool
plans; `text_only_resident_discount` then discounts **vision only** (970,536,960 B)
and prices the MTP residents. The swap is a valid, side-effect-free spec change
(`mtp_included` gates only the discount; `plan_expert_memory` does not branch on it).

| Serve mode | Resident priced (GiB) | Discount (B) | Slots/layer | Expert cache (GiB) | Fits (82 GiB) |
|---|---:|---:|---:|---:|:--:|
| AR (mtp off) | 9.066 | 8,920,505,736 (mtp+vision) | **92** | 64.435 | yes |
| **MTP (mtp on)** | 16.470 | 970,536,960 (vision) | **82** | 57.431 | yes |

MTP prices **+7.404 GiB (+7.95 GB)** of residents and costs **10 slots/layer**,
still fitting the 82 GiB envelope. Note on the coordinator's "+6.7 GiB": that is
the DSpark head's **active** 3×128 mxfp4 experts; the loader (and therefore the
plan, to avoid under-reserving) keeps **every** `mtp.*` resident the artifact
ships (7.95 GB), which is the number priced here.
