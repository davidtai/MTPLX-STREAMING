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
the residents. **MLX 0.32.2 treats `set_memory_limit` as a *soft* limit**: allocations
beyond it go through the over-limit path (cache release / scheduler wait) — the (f)
allocator-pressure regime. Setting the limit at the plan while the steady-state peak
lands ~5.5 GiB above it keeps the process permanently in that path.

## What the lever changes — and what it does NOT

The lever is `MTPLX_DSV41_MLX_LIMIT_HEADROOM_GIB` (read at use, default `0` = today).
It adds N GiB to the value handed to `set_memory_limit`, in exactly one place.

**Changes (only these numbers):**

- The value passed to `mx.set_memory_limit` — from `plan_limit` to
  `plan_limit + N·GiB`. This is the **soft** allocator ceiling; raising it above the
  steady-state peak (74.3 < 69.2 + 8) lifts the process out of the over-limit path.
- `memory.mlx_limit_gib_effective` (receipt) — `plan_limit_gib_effective + N`.
- `memory.mlx_limit_headroom_gib` (receipt) — `N`.
- `memory.budget_forecast_system_peak_gb` (receipt) — the headroom is **priced into
  the forecast box peak** (see below).

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

## The W106 budget forecast prices the headroom

The raised `set_memory_limit` lets the allocator retain up to N GiB **above** the plan,
so the headroom counts toward the forecast whole-box peak
(`scripts/deepseek_v41/ab_decode_env_levers.py`, `BudgetTotalDerivation`):

```
forecast_system_peak = baseline + plan_limit + plan_overshoot
                                 + non_metal_overhead + mlx_limit_headroom
```

Two derivation paths, both self-consistent:

- **Derive-from-budget** (`derive_budget_total_plan`): the headroom is **subtracted
  from `plan_limit`** alongside overshoot/kv/safety, so under a fixed budget it comes
  off the residents and the forecast still lands at `budget − kv − safety ≤ budget`.
- **Pinned plan** (`--memory-plan-from`, the A/B path): `plan_limit` is fixed, the
  arm's headroom is added on top, and the forecast **rises by exactly the headroom**.
  `_validate_pinned_plan` refuses the pin (before the window opens) if
  `live + plan + overshoot + overhead + headroom > budget`.

Receipt keys (in `memory`): `mlx_limit_headroom_gib`, `mlx_limit_gib_effective`,
`budget_forecast_system_peak_gb` (now inclusive of the headroom), and the sidecar
(`derived-plan.json`) round-trips `mlx_limit_headroom_gib`.

## The bench flag and the arms

- Flag: `--mlx-limit-headroom-gib N` → maps to `MTPLX_DSV41_MLX_LIMIT_HEADROOM_GIB`
  (applied after the arm preset, so an explicit flag overrides a preset and reaches the
  runtime). Priced into the budget forecast.
- Arms (append headroom 8 to the W97F composites; every other lever key identical):
  - `cell16k_ring_v2_attn_hr8` = `cell16k_ring_v2_attn` + headroom 8.
  - `cell16k_ring_v2_draft_attn_hr8` = `cell16k_ring_v2_draft_attn` + headroom 8.

## The GPU A/B (window 46)

Run at the **pinned plan 69** (window 44's plan), so every arm shares one
`plan_limit` and the comparison is meaningful. Derive + pin once, then run each arm
against the sidecar:

```
# 1. Derive + pin the plan (headroom 0) with the base arm.
python scripts/deepseek_v41/ab_decode_env_levers.py \
  --arms cell16k_ring_v2_attn \
  --memory-budget-total-gib 100 --out <dir>/w46.jsonl ...
# 2. AR A/B: base vs +headroom 8, pinning the SAME plan.
python .../ab_decode_env_levers.py --arms cell16k_ring_v2_attn cell16k_ring_v2_attn_hr8 \
  --memory-plan-from <dir>/derived-plan.json --out <dir>/w46.jsonl ...
# 3. DSpark d3 A/B: draft_attn vs +headroom 8.
python .../ab_decode_env_levers.py \
  --arms cell16k_ring_v2_draft_attn cell16k_ring_v2_draft_attn_hr8 \
  --decode-mode dspark --memory-plan-from <dir>/derived-plan.json --out <dir>/w46.jsonl ...
```

**Proof the lever engaged and is safe:**

- `memory.mlx_limit_gib_effective` on the `_hr8` arm = plan + 8 (control = plan + 0).
- `memory.mlx_peak_gb` sits **below** the effective limit (the over-limit path is no
  longer taken).
- The plan-equality guard prints `all arms ran plan_limit=<plan>` (equal
  `plan_limit_gib_effective`), i.e. residency did not move.

**Proof it worked (H7):** the in-model **attention stage ms/tok** and **`verify_ms`**
come **down** on the `_hr8` arm (out of the 5.2× allocator-pressure regime), with
`token_ids_sha256` **byte-identical** to the base arm.

## Files

- `mtplx/expert_runtime.py` — `resolve_mlx_limit_headroom_bytes` + the headroom in
  `apply_mlx_memory_cap`.
- `scripts/deepseek_v41/ab_decode_env_levers.py` — the env constant,
  `ALL_LEVER_ENVS` / `_preset` registration, the `_hr8` arms, the `--mlx-limit-headroom-gib`
  flag, the `BudgetTotalDerivation` headroom term (forecast + sidecar + receipt keys),
  and the pin/derive/pre-flight wiring.
- `mtplx/server/openai.py` — `_DSV41_LEVER_ENV_KEYS` registration (W46/W90 superset).
- `tests/test_deepseek_v41_w118_mlx_headroom.py` — CPU tests (fake `mx`).
