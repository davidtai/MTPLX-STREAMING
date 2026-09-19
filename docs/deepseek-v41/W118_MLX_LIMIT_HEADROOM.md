# W118 — MLX allocator-limit headroom (H7)

Raise **only** the MLX allocator soft limit (`mx.set_memory_limit`) by N GiB **above**
the residency plan, **without changing what is resident or the expert-cache slot
plan** — so bytes, routing and outputs are byte-identical. The lever tests hypothesis
**H7**: the in-model attention (and everything else) pays allocator-pressure time
because the runtime's MLX limit is set at the *plan* while the steady-state peak sits
~5.5 GiB above it.

## H7 — the finding it acts on

W112's attention-contention probe (receipt
`docs/deepseek-v41/receipts/gpu-windows/window-44b/attn-contention-probe.json`)
isolated the 40-layer attention at the 16K cell:

- **1.28 ms/layer** pipelined (baseline).
- +0.35 under concurrent SSD reads + expert gathers; +0.28 fenced per-layer barrier;
  thermal nil; eager o-LoRA re-layout +0.9.
- **BUT under allocator pressure** (MLX memory limit 60 GiB with ~58 GiB ballast
  resident, cache limit 6 GiB) = **6.58 ms/layer, 5.2× baseline** — the (f) regime.

Every real window runs the model **OVER its own MLX limit**:

| window | plan (GiB) | mlx_peak (GiB) | plan_overshoot ≈ |
| ------ | ---------- | -------------- | ---------------- |
| 43     | 60         | 65.5           | 5.5              |
| 44     | 69.2       | 74.3           | 5.5              |

The "plan_overshoot" (≈ 5.5 GiB) is KV + prefill transients + expert-cache slots above
the residents.

**Mechanism (mlx 0.32.2 `allocator.cpp`).** `set_memory_limit(limit)` sets `block_limit_`
and `gc_limit_ = min(limit, 0.95 × recommendedMaxWorkingSetSize)`. On a cache **miss**
with `active + cache + size ≥ gc_limit_`, `MetalAllocator::malloc` calls
`release_cached_buffers(...)`; once `active ≥ gc_limit_` the release argument exceeds the
pool, so the **entire** buffer cache is cleared, and every later allocation is a fresh
`newBuffer` (page zero-fill) + residency-set insert. There is **no scheduler wait and no
error** — it is **cache-clear-on-miss thrash**, the (f) allocator-pressure regime (5.2×
in-model attention). Setting the limit at the plan while the steady-state peak lands
~5.5 GiB above it keeps `gc_limit_` below the peak, so the process is permanently in that
path. Raising the limit lifts `gc_limit_` above the peak and the miss path stops clearing
the cache.

**H7 status: plausible for prefill, unproven for decode.** `mlx_peak > limit` is measured
for the whole window (prefill sets the peak); whether the *decode* steps are also over
the limit — the thing that would make the lever help decode tok/s — is what the window-46
A/B measures (`mlx_active_gb_at_decode_end` vs `mlx_gc_limit_gib_effective`, plus attention
stage ms/tok and `verify_ms`).

## What the lever changes — and what it does NOT

The lever is `MTPLX_DSV41_MLX_LIMIT_HEADROOM_GIB` (read at use, default `0` = today).
It adds N GiB to the value handed to `set_memory_limit`, in exactly one place.

**Changes (only these numbers):**

- The value passed to `mx.set_memory_limit` — from `plan_limit` to
  `plan_limit + N·GiB`. This is the **soft** allocator ceiling; raising it above the
  steady-state peak (74.3 < 69.2 + 8) lifts `gc_limit_` above the peak so the miss path
  stops clearing the cache.
- `memory.mlx_limit_gib_effective` (receipt) — `plan_limit_gib_effective + N`.
- `memory.mlx_limit_headroom_gib` (receipt) — `N`.
- `memory.budget_forecast_system_peak_gb` (receipt) — the headroom is **priced into
  the forecast box peak** (see below).
- **Proof keys (review MEDIUM-2), read on the main thread outside the timed region:**
  `memory.mlx_limit_gib_readback` (`mx.get_memory_limit()` after load),
  `memory.mlx_gc_limit_gib_effective` (`min(readback, 0.95 × device
  max_recommended_working_set_size)` — the `gc_limit_` the allocator clears against),
  `memory.mlx_active_gb_at_decode_start` / `_at_decode_end`,
  `memory.mlx_cache_gb_at_decode_end`, and `memory.mlx_peak_over_limit_gb`
  (`mlx_peak − readback`; positive means the run went over the soft limit). The
  **thrash signature** is `mlx_active_gb_at_decode_end ≥ gc_limit − ~1` with
  `mlx_cache_gb_at_decode_end ≈ 0`.

**Does NOT change (asserted via the `resolved_plan` / receipt keys):**

- **Residents** — the weights, KV lanes and the SWA/wo_a reserve. Bounded by
  `MTPLX_MEMORY_LIMIT_BYTES`, which is stamped at the **plan** value, unchanged.
- **Transient slots** — the expert-cache slot plan (`transient_slots`,
  `split_route_release`).
- **The prefetch ring** (`prefetch_slots`).
- `plan_limit_gib_derived` / `plan_limit_gib_effective` — the plan limit that bounds
  residency. **Equal for control vs `_hrN`**, which is exactly what the harness's
  plan-equality guard checks before it trusts a byte-identity comparison.
- The allocator **cache** limit (`mx.set_cache_limit`) — resolved separately by
  `deepseek_v41_memory_profile.apply_allocator_cache_limit` and **left as is** (it is
  not coupled to the headroom).

Because the plan (and thus every resident byte, every routed expert and every KV row)
is untouched, greedy decode is **byte-identical**: the `token_ids_sha256` matches.

## The exact code path that sets the limit

`mtplx/expert_runtime.py`:

- `reconcile_mlx_memory_cap(plan, env)` computes `plan_limit` from the plan
  (`total_limit_bytes − runtime_reserve_bytes − io_staging_bytes − paged_band`).
  **The headroom is never read here** — the plan limit is identical with or without it,
  so residents/slots cannot move.
- `resolve_mlx_limit_headroom_bytes(env)` (new) reads
  `MTPLX_DSV41_MLX_LIMIT_HEADROOM_GIB` **at use** (default 0; rejects negative /
  non-numeric).
- `apply_mlx_memory_cap(plan, mx_module, env)`:
  1. `plan_limit = reconcile_mlx_memory_cap(plan, env)`
  2. `env["MTPLX_MEMORY_LIMIT_BYTES"] = str(plan_limit)` — the **engine budget** that
     bounds residency, stamped at the plan value (unchanged).
  3. `limit = plan_limit + resolve_mlx_limit_headroom_bytes(env)`
  4. `set_memory_limit(limit)` — the **only** consumer of the headroom.

`apply_mlx_memory_cap` is called from `ExpertStreamingRuntime.open`
(`mtplx/expert_runtime.py`), which the DeepSeek-V4.1 loader
(`mtplx/models/deepseek_v41_loader.py`) drives on **both** the bench and the served
path — so the headroom is a live served lever (registered in
`server.openai._DSV41_LEVER_ENV_KEYS`, kept a superset of the A/B
`ALL_LEVER_ENVS`).

## The W106 budget forecast prices the headroom (review HIGH-1)

The raised `set_memory_limit` lets the allocator **retain** freed buffers up to the
cache limit above the plan — it does **not** add a fresh N GiB on top of the overshoot.
So the extra box-peak the headroom can add is **bounded**, and the forecast term is
(`scripts/deepseek_v41/ab_decode_env_levers.py`, `BudgetTotalDerivation` /
`_headroom_forecast_extra_gib`):

```
allocator_extra      = max(plan_overshoot, min(headroom, plan_overshoot + cache_limit))
forecast_system_peak = baseline + plan_limit + non_metal_overhead + allocator_extra
```

with `cache_limit` = the `mx.set_cache_limit` ceiling (default 6 GiB, threaded from
`MTPLX_DSV41_MLX_CACHE_LIMIT_GB`). This is **not** `plan_overshoot + headroom` — that
double-count refused the documented window-46 run (forecast 97.7 > 93). At the pinned
window-44 plan (69) and budget 93, with overshoot 6, cache 6, headroom 8:
`allocator_extra = max(6, min(8, 12)) = 8`, forecast `= 8.7 + 69 + 6.0 + 8 = 91.7 ≤ 93`
(**accepted**). With headroom 0, `allocator_extra = plan_overshoot`, so the lever-off
forecast is unchanged.

Two derivation paths, both self-consistent:

- **Derive-from-budget** (`derive_budget_total_plan`): `allocator_extra` is **subtracted
  from `plan_limit`** alongside kv/safety, so under a fixed budget it comes off the
  residents and the forecast still lands at `budget − kv − safety ≤ budget`.
- **Pinned plan** (`--memory-plan-from`, the A/B path): `plan_limit` is fixed, the arm's
  headroom is added on top, and the forecast **rises by `allocator_extra − plan_overshoot`**
  (2 GiB in the example above, not the full headroom). `_validate_pinned_plan` refuses the
  pin (before the window opens) if `live + plan + overhead + allocator_extra > budget`.

Receipt keys (in `memory`): `mlx_limit_headroom_gib`, `mlx_limit_gib_effective`,
`budget_cache_limit_gib`, `budget_headroom_forecast_extra_gib`,
`budget_forecast_system_peak_gb` (inclusive of `allocator_extra`), and the sidecar
(`derived-plan.json`) round-trips `mlx_limit_headroom_gib` + `cache_limit_gib`.

**Auto-pin (review MEDIUM-1).** On the derive path, arms after the first in one
invocation **auto-pin** arm 1's `derived-plan.json`, so a multi-arm run shares one
`plan_limit` even when headroom differs — without it, a per-arm re-derive would drop the
plan by `allocator_extra` on the hr arm and confound the A/B.

**Pre-flight (review HIGH-2).** `--memory-plan-preflight` runs before the arm presets are
applied, so it prices `max(flag, max preset headroom over the arms)` — otherwise a
preset-carried `*_hr8` would read as 0 and the pre-flight over-budget check be blind.

**Finite guards (review MEDIUM-3).** `nan`/`inf`/`1_0`/`abc` are rejected — as
`ExpertStreamingConfigurationError` in the runtime, and as a clean `SystemExit` in the
harness (before the in-window crash), rather than passing validation
(`float('nan') < 0` is `False`) or being silently misread (`float('1_0')` is `10.0`).

## The bench flag and the arms

- Flag: `--mlx-limit-headroom-gib N` → maps to `MTPLX_DSV41_MLX_LIMIT_HEADROOM_GIB`
  (applied after the arm preset, so an explicit flag overrides a preset and reaches the
  runtime). Priced into the budget forecast.
- Arms (append headroom 8 to the W97F composites; every other lever key identical):
  - `cell16k_ring_v2_attn_hr8` = `cell16k_ring_v2_attn` + headroom 8.
  - `cell16k_ring_v2_draft_attn_hr8` = `cell16k_ring_v2_draft_attn` + headroom 8.

## The GPU A/B (window 46)

Run at the **pinned plan 69** (window 44's plan) and **budget 93**, so every arm shares
one `plan_limit` and the comparison is meaningful. Derive + pin once **with the control
arm**, then run each pair against the sidecar. The `*_hr8` arms carry headroom 8 in the
**preset** — do **not** pass `--mlx-limit-headroom-gib` (that would apply headroom to the
control arm too):

```
# 1. Derive + pin the plan (headroom 0) with the CONTROL arm at budget 93.
python scripts/deepseek_v41/ab_decode_env_levers.py \
  --arms cell16k_ring_v2_attn \
  --memory-budget-total-gib 93 --out <dir>/w46.jsonl ...
# 2. AR A/B: control vs +headroom 8 (preset), pinning the SAME plan.
python .../ab_decode_env_levers.py --arms cell16k_ring_v2_attn cell16k_ring_v2_attn_hr8 \
  --memory-plan-from <dir>/derived-plan.json --out <dir>/w46.jsonl ...
# 3. DSpark d3 A/B: draft_attn vs +headroom 8 (preset).
python .../ab_decode_env_levers.py \
  --arms cell16k_ring_v2_draft_attn cell16k_ring_v2_draft_attn_hr8 \
  --decode-mode dspark --memory-plan-from <dir>/derived-plan.json --out <dir>/w46.jsonl ...
```

(A single-invocation `--arms control hr8` also works: MEDIUM-1 auto-pins the hr8 arm to
the control's derived plan, so both share `plan_limit`.)

**Proof the lever engaged and is safe:**

- `memory.mlx_limit_gib_readback` on the `_hr8` arm = plan + 8 (control = plan + 0), and
  `memory.mlx_gc_limit_gib_effective` rises with it.
- `memory.mlx_peak_over_limit_gb` goes from **positive** (control, over the soft limit)
  toward **≤ 0** on `_hr8`; `mlx_active_gb_at_decode_end` drops below
  `mlx_gc_limit_gib_effective` with `mlx_cache_gb_at_decode_end` no longer pinned near 0
  (the cache stops being cleared on every miss).
- The plan-equality guard prints `all arms ran plan_limit=<plan>` (equal
  `plan_limit_gib_effective`), i.e. residency did not move.

Report **prefill tok/s + TTFT** alongside decode tok/s for every arm (prefill is where
`mlx_peak > limit` is already proven, so the headroom's prefill effect is the more likely
win).

**Proof it worked (H7):** the in-model **attention stage ms/tok** and **`verify_ms`**
come **down** on the `_hr8` arm (out of the 5.2× allocator-pressure regime), with
`token_ids_sha256` **byte-identical** to the base arm.

## Served path (LOW: box-fit note)

The env is a **live served lever** — `ExpertStreamingRuntime.open` runs
`apply_mlx_memory_cap` on the served daemon too, so setting
`MTPLX_DSV41_MLX_LIMIT_HEADROOM_GIB` there raises `mx.set_memory_limit` in the served
process. But the served daemon does **not** run the W106 budget forecast, so on serve the
headroom is **unpriced against the box budget**. The exposure is bounded: the allocator
can only retain up to the served **cache limit** (`_configure_mlx_cache_limit`,
~6 GiB tier) above the active peak, so the worst-case extra box-peak is
`min(headroom, cache_limit)` — the same bound the bench forecast prices. Keep the served
headroom ≤ the box's free margin; it is off (0) by default.

## Files

- `mtplx/expert_runtime.py` — `resolve_mlx_limit_headroom_bytes` + the headroom in
  `apply_mlx_memory_cap`.
- `scripts/deepseek_v41/ab_decode_env_levers.py` — the env constant,
  `ALL_LEVER_ENVS` / `_preset` registration, the `_hr8` arms, the `--mlx-limit-headroom-gib`
  flag, the `BudgetTotalDerivation` headroom term (forecast + sidecar + receipt keys),
  and the pin/derive/pre-flight wiring.
- `mtplx/server/openai.py` — `_DSV41_LEVER_ENV_KEYS` registration (W46/W90 superset).
- `tests/test_deepseek_v41_w118_mlx_headroom.py` — CPU tests (fake `mx`).
