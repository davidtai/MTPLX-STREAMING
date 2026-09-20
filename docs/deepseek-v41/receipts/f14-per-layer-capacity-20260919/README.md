# F14 — non-uniform per-layer expert-cache CAPACITY (DeepSeek-V4.1 M6 decode)

CPU-only screen. No MLX, no GPU, no service touched: `per_layer_capacity_sim.py`
installs a `NoMLX` meta-path finder that raises on any `mlx` import, uses numpy
only, runs under `nice -n 19`. A slot bank is a **pure cache** — it changes only
which records are read, never a routed expert's output — so **any allocation is
bit-exact**. Not a throughput result and not a production-code proposal.

## Question

Every routed layer gets the same persistent capacity today (108–111 rows) + 48
shared transient slots. 84 rows exist at the prefill→decode boundary (here a
73-slot capture); the rest are **extension rows** allocated per layer *after*
prefill, so per-layer capacity is a free construction-time choice as long as each
layer keeps ≥ 84 rows and the **total row count is unchanged**. Fable's GPU stamp
probe shows very uneven per-layer miss cost (L0 11.6 ms … L20 2.9 ms of read wait
per layer call). How many expert-record reads does a non-uniform per-layer
capacity remove at the same total memory?

## Method / fidelity

The replay **reuses the exact machinery** that reproduces the f1-overlap control:
`overlap_schedule_sim.make_bank` (restore the captured 73-slot snapshot, grow to
`c`, `_protected_cap = int(c·0.8)`, transient 48) + the real
`LayerExpertSlotBank.plan(phase='decode')`. Each of the 40 layers (0–39) is an
**independent** bank, so `total = Σ_L misses_L(c_L)` is **separable** and the
anchor is a per-layer superposition. `misses_L(c)` was simulated at **every integer
capacity 84–192** (superset of the requested 16-point grid), so the allocation is
optimized **exactly** (no interpolation) and re-simulating the chosen vector is an
identity check (confirmed: `resim == opt` for every allocation).

**Anchors reproduced exactly** (per-layer replay through the real bank):
prefix-readiness cap102 → 35,164 misses / 8,240 routes ✔; mtp-verify cap73
(frequency) → 53,999 records ✔; **uniform 111+48 → 31,636 records** (= f1-overlap
control) ✔, and an independent re-simulation also = 31,636. Uniform 108 → 32,763.

## Commands (from the worktree root, under `nice -n 19`)

```
WT=.worktrees/dsv41-f14-capacity ; PY=<repo>/.venv/bin/python
# curves in 4 batches of 10 layers (~20 s each, flag-gated):
for lohi in 0:9 10:19 20:29 30:39; do
  nice -n 19 $PY scripts/deepseek_v41/per_layer_capacity_sim.py curves \
    --lo ${lohi%:*} --hi ${lohi#*:} --grid dense --out /tmp/curves_${lohi/:/_}.npz ; done
# analyze (anchors + DP optimize + causality + sensitivity + exact re-sim):
nice -n 19 $PY scripts/deepseek_v41/per_layer_capacity_sim.py analyze \
  --npz /tmp/curves_0_9.npz /tmp/curves_10_19.npz /tmp/curves_20_29.npz /tmp/curves_30_39.npz \
  --out docs/deepseek-v41/receipts/f14-per-layer-capacity-20260919/results.json
```

`results.json` carries the full 16-point miss table, all allocation vectors, the
split-half and prefill-rule allocations, and every sensitivity number.
`curves_dense.npz` is the raw per-(layer, capacity, cycle) miss array. `sha256.json`
hashes inputs + outputs (trace `07b4b720…`).

## Results

**Total misses vs UNIFORM capacity** (Σ over 40 layers):

| c | 84 | 100 | 108 | 111 | 116 | 128 | 144 | 176 | 192 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| records | 43,826 | 35,981 | 32,763 | 31,636 | 29,899 | 26,275 | 22,069 | 16,202 | 14,174 |

Per-layer curves are **monotone non-increasing but non-convex** (0/40 strictly
convex; worst 2nd-diff −17), so an **exact DP** (separable bounded allocation) is
used, not marginal greedy (greedy left ~1% on a 20-layer subset).

**Oracle (full-trace) reallocation at the same total memory:**

| C_uniform | uniform reads | oracle reads | removed | % | seconds @1.36 ms | ≈ uniform-equiv |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 108 | 32,763 | 31,165 | **1,598** | 4.88 | **2.17** | ≈ uniform-112 |
| 111 | 31,636 | 30,003 | **1,633** | 5.16 | **2.22** | ≈ uniform-116 |

Allocation range [84, 171]. Rows flow to the high-hot-set layers the probe flagged
expensive (L0→171, L2→149, L38/L39→145, L1→144/158) and away from the cheap ones
(L20/L22/L24/L25→84, the floor). Non-uniform at 108 total rows matches **uniform
~112** and at 111 matches **uniform ~116** — worth ~+4–5 uniform rows/layer at zero
extra memory.

**Causality (allocation must be knowable before decode):**

| Choice | C=108 removed | C=111 removed | note |
| --- | ---: | ---: | --- |
| oracle (full trace, non-causal ceiling) | 1,598 (2.17 s) | 1,633 (2.22 s) | upper bound |
| split-half: fit cyc 0–99 → eval 100–205 | 698 (0.95 s) | 683 (0.93 s) | held-out |
| split-half: fit cyc 100–205 → eval 0–99 | 387 (0.53 s) | 382 (0.52 s) | held-out |
| **prefill-only rule (causal, deployable)** | **581 (0.79 s)** | **623 (0.85 s)** | 36–38 % of oracle |

The prefill rule sets extension rows ∝ a prefill-diffuseness statistic of
`_prefill_route_freq`; **`1 − top-84 mass`** correlates 0.925 with the per-layer
decode miss level (effective-#-experts 0.899, entropy 0.886, raw distinct-count
only 0.458 — near-saturated). Split-half shows the per-layer *need* is stable
within a run (a within-run oracle transfers 382–698 reads to the held-out half);
the prefill proxy captures ~37 % of the ceiling.

**Cross-prompt:** only one full decode route trace exists in the receipts; the
other `.json.gz` are predictor-experiment variants of the same 64-cycle capture.
Cross-prompt transfer is **untested** (stated plainly).

**Sensitivity** (oracle gain, records removed):

| Perturbation | C=108 | C=111 |
| --- | ---: | ---: |
| baseline | 1,598 | 1,633 |
| −8 total rows | 1,609 | 1,660 |
| +8 total rows | 1,596 | 1,625 |
| `_protected_cap` fixed at int(C·0.8) (not scaled per-layer) | 1,598 | 1,633 |
| allocation rounded to multiples of 4 rows | 1,492 | 1,533 |

The gain is invariant to ±8 total rows, **identical** whether `_protected_cap`
scales with per-layer capacity or is pinned (so it is a pure-capacity effect, not a
protected-segment artifact), and keeps 93–94 % after rounding to multiples of 4.

## Recommendation & risk

Allocate per-layer extension rows **proportional to prefill diffuseness**
(`1 − top-84 mass` of `_prefill_route_freq`, or the near-equivalent
effective-#-experts / entropy), rounded to multiples of 4, at the same total
memory. Expected causal gain **≈ 580–620 records / ~0.8 s per 1,024-token run
(≈1.8–2.0 %)**, bit-exact; oracle ceiling ≈ 1,600 records / ~2.2 s (≈5 %). **Risks:**
(1) proxy — captures only ~37 % of the ceiling; a short decode warm-up window or a
fitted (non-proportional) rule would recover more (split-half shows the need is
learnable). (2) **One prompt** — cross-prompt transfer untested. (3) 73-slot capture
grown to ≥84 with empty extension rows (matches real construction) but boundary
residents are fixed at 73. (4) Read-count simulation, not throughput — host overlap
could hide part of the read-wait saving.

## Limits

Screen only; no production code changed, no GPU/service touched, no throughput
proof. Deterministic replay of one captured decode trace through the real cache.
