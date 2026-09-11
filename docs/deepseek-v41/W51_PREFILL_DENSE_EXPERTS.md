# W51 — DeepSeek-V4.1-Flash prefill dense-experts path (KERNEL_LEDGER K26)

**Task class:** kernel/runtime surgery (a prefill-only expert dispatch), CPU-proven,
default OFF. Branch `feat/deepseek-v41-w51`. Companion to the K16 layer-major prefill
(W30) and the W47 stage-timing probe that motivated it. No GPU ran here; the realized
prefill delta is a GPU-window question (KG-k, below).

## Motivation (W47, GPU window 19)

The W47 16,384-token layer-major stage timing
(`receipts/gpu-windows/window-19/prefill-16384-stage-timing.json`, `W47_PREFILL_STAGE_TIMING.md`)
put the streamed switch at:

| stage | value |
|---|---|
| `moe.routed_switch` (fenced total) | **105 s** over 40 layer-major calls (~2.6 s/layer) |
| ↳ `switch.miss_submit` | 6.2 s |
| ↳ `switch.route_plan` | 4.0 s |
| ↳ `gather_qmm` compute (residual) | **~95 s** |
| bank read (269 GiB once, 13 GB/s) | ~22 s (overlaps the gather) |

Per layer: 98,304 routed rows (16,384 tokens × top-6) × 3 projections (gate/up
5120→2304, down 2304→5120) = **6.96 TFLOP**; ×40 = **278 TFLOP in ~95 s ≈ 2.9 TFLOPS**.
That is the mxfp4 gs32 `gather_qmm` at large M being **ALU/dequant-bound**
(`[[metal-sub4bit-alu-bound]]`: sub-4-bit dequant ALU/occupancy binds before bandwidth),
against a dense bf16 matmul on this box at **~15–25 TFLOPS**. The switch compute, not
the SSD bank read, is the dominant term of the 370 s layer-major TTFT.

The W17 microbench (`receipts/gpu-windows/window-17/gather-qmm-microbench.json`) shows
the **opposite** verdict at small M — bf16 dense is memory-bound reading ~3.8× the bytes:

| arm | M=1 (6 rows) | M=4 (24 rows) |
|---|---|---|
| `mxfp4_switch` (gather_qmm) | 417 µs | 982 µs |
| `bf16_dense` | 1035 µs | 3198 µs |

So dense **loses** at decode/verify M and **wins** only once M is large enough to amortize
the bf16 weight read into an ALU-bound matmul. K26 is exactly that: prefill-only, above a
per-expert row threshold.

## Mechanism

`MTPLX_DSV41_PREFILL_DENSE_EXPERTS=1` (default OFF) arms a prefill-only path in the
streamed switch (`mtplx/models/expert_mlx.py`):

1. `HotExpertSwitchGLU._run` sets `_dense_prefill_active = phase is PREFILL and
   codec == "mxfp4" and env-on` (read at use, `[[env-flags-read-at-use-not-import]]`).
   It is threaded into `_dispatch_component_bank(..., dense_prefill=...)` from the PREFILL
   split-route wave only — the decode all-hit / device-route / shadow callers keep the
   default `False`, so the path is **structurally unreachable at decode M=1**.
2. `_run_component_bank_dense_prefill` groups the wave's assignment rows by expert (bank
   slot) from the **host-side** `binding.buffer.bank_index` — no device→host sync, and each
   expert's rows are already together within a wave (route plan). For every expert with
   ≥ `MTPLX_DSV41_PREFILL_DENSE_MIN_ROWS` rows (default **128**):
   - dequantize gate/up/down **once** with the exact call
     `mx.dequantize(weight, scales, group_size=32, bits=4, mode="mxfp4")`
     (E8M0 scales, no bias — the same leaves `_gather_component_bank`'s `mode="mxfp4"`
     gather feeds `gather_qmm`), giving three dense `[out, in]` bf16 matrices;
   - run `gate = x·Wgᵀ`, `up = x·Wuᵀ`, `_clamped_swiglu(gate, up, 10.0)` (unchanged),
     `y = hidden·Wdᵀ` as **dense bf16 matmuls** over that expert's rows;
   - scatter `y` back into the router's original row order.
   Experts below the threshold are served in one grouped `gather_qmm` call.
3. **Bounded transient:** each dequantized expert is ~3 × 2 × 5120 × 2304 B ≈ **70.8 MB**
   bf16. Experts are dequantized in batches of `MTPLX_DSV41_PREFILL_DENSE_BATCH` (default
   **8**) with an `mx.eval` between batches, so at most ~8 experts' dequantized copies are
   live (~**0.57 GB**), and none survives the call — no dequantized weight is kept across
   layers.

When **no** expert clears the threshold (small-M / decode-shaped waves) the function is a
pure `gather_qmm` fall-through, byte-for-byte.

## Cost model (16K layer-major, 384 experts, top-6, ~256 rows/expert avg)

At 16 K almost every expert clears the 128-row threshold, so essentially the whole routed
bank is dequantized per layer:

| term | value |
|---|---|
| dequant bf16 written / layer | 384 × 70.8 MB ≈ **27 GB** |
| dequant bf16 written / prefill (×40) | ≈ **1.09 TB** (269 GiB mxfp4 @ ~0.53 B/w → 2 B/w bf16) |
| dequant DRAM round-trip @ 614 GB/s | ~1.8 s write + ~1.8 s read ≈ **3.5 s** total |
| matmul FLOPs | 278 TFLOP (unchanged) |
| matmul time @ 15–25 TFLOPS (dense) | **11–19 s** (vs 95 s @ 2.9 TFLOPS gather) |
| **switch compute** | **95 s → ~15–22 s** (dense matmul + dequant DRAM) |
| **`moe.routed_switch`** | **105 s → ~25–35 s** (bank read 22 s now co-binds) |
| **16K layer-major TTFT** | **370 s → ~295 s (44 → ~55 tok/s)** |

Expected saving ≈ **70–80 s off the 16 K prefill**. The 269 GiB packed bank read (~22 s) is
unchanged — paid by both paths. Peak stays ≤ ~0.57 GB extra, under the 60 GiB plan / 100 GiB
knob (`[[never-exceed-the-memory-knob]]`, `[[box-110gb-hard-limit]]`).

Caveats: (a) if routing skew leaves many experts under 128 rows the win shrinks (those keep
gather_qmm); (b) the dense path adds the ~3.5 s dequant DRAM traffic gather does not — net
still strongly positive if the 2.9-TFLOPS ALU-bound gather estimate holds; (c) all figures
are analytical — the GPU window measures the real crossover M and realized dense TFLOPS.

## Exactness

`gather_qmm(mxfp4)` vs dequantize+bf16-matmul is **not** bit-identical (fp32 accumulation
order differs). The dequant itself is lossless: an FP4 code × 2^E8M0 lands exactly in bf16.
CPU test on random mxfp4 gs32 weights
(`tests/models/test_deepseek_v41_prefill_dense_experts.py`, hidden 256 / inter 128, 436 rows
over 5 experts, MLX pinned to CPU), reporting max |Δ| and relative error vs a **float64
reference** (the exact dequantized weights, matmul + clamped-SwiGLU in fp64):

| path | max \|Δ\| vs fp64 | rel (to peak) |
|---|---|---|
| `gather_qmm` (flag off) | 9.40e-4 | 2.00e-2 |
| dense bf16 (flag on) | **2.81e-4** | **5.96e-3** |
| dense **vs** gather | 1.22e-3 | 2.56e-2 |

The dense path is *closer* to the fp64 truth than `gather_qmm` (lossless dequant +
fp32-accumulated dense matmul). The dense-vs-gather delta (max |Δ| 1.22e-3, ~2.6e-2 relative
to the output peak) is the bf16 accumulation-order divergence — David's documented FP class
(cf. #171 vk_k split-K, the K16/K24 compile divergences), not a correctness bug. Tests also
assert:
- the mixed dense+gather (default-128) output matches the pure-gather (flag-off) output
  within that bf16 bound ("flag on == flag off" gate);
- the scatter maps every row back to its own expert's fp64 MLP (router order preserved);
- below-threshold and M=1 (decode-shaped, one row per expert) waves are `mx.array_equal`
  to the pure gather — **the flag cannot perturb decode**.

## GPU window 20 result and follow-up

**Measured** (integration `7a789731d`, 16,384-token prompt, layer-major, 60 GiB plan, chunk
1024; `receipts/gpu-windows/window-20/prefill-16384-ladder.json`):

| arm | TTFT | vs layer_major | tokens identical? | peak |
|---|---|---|---|---|
| `layer_major` (control) | 352.8 s | — | — | 76.6 GB |
| `prefill_dense_experts` | **332.2 s** | **−20 s (+6% prefill tok/s)** | yes (token-sha) | 76.6 GB |

−20 s, not the modelled −70–80 s. The model assumed dense bf16 runs at 15–25 TFLOPS; the
shortfall says it does not. Two suspects — the window rules the first out.

### (a) Threshold coverage — NOT the bottleneck

From W24's routing census (`routing_census_1024.json`, `prefill_distinct_experts` per layer:
mean **261/384**, median 259.5, min 224, max 313 at 1024 tokens), the per-layer load at 16 K is
98,304 assignments (uniform 256 rows/expert). An occupancy estimate of the fraction of rows in
experts **below** a threshold at 16 K:

| model | rows <128 | rows <32 | rows <16 |
|---|---|---|---|
| equal-prob core (Neff≈261, ~376 rows each) | ~0% | ~0% | ~0% |
| skew-aware, distinct@16K→320 | ~7.7% | ~1.9% | ~1.0% |
| skew-aware, distinct@16K→384 | ~10.0% | ~4.0% | ~2.0% |

So **threshold 128 already routes ~90–97% of rows dense**; 32 captures >96%, 16 >98%. Lowering
the threshold recovers a few % of rows at most — coverage is not why the win is small. (The new
per-layer counters below measure the true split directly, superseding this estimate.) A row
threshold near **32** is nonetheless a safe default: it captures essentially all routable rows
while staying above the small-M region where the W17 microbench shows dense losing to gather.

### (b) The bf16 dense matmul kernel — the likely leak

W50 measured bf16 score matmuls **34% slower than f32** at 16 K on this box. The dense
gate/up/down probably pay the same slow bf16 path, eating the ALU advantage. Ruled out as the
leak: **the dequant is not double-writing** — `mx.dequantize(mode="mxfp4")` returns **bf16
directly** (verified; default output dtype is bf16, no f32 intermediate), and K26 now dequantizes
**straight to the compute dtype** (`dtype=` on `mx.dequantize`), so there is no bf16→f32 or
f32→bf16 cast and no doubled write in either variant. The `dense_f32` arm
(`MTPLX_DSV41_PREFILL_DENSE_MATMUL_DTYPE=f32`) runs dequant + all three matmuls in f32 to test
whether the f32 kernel recovers the modelled win (CPU exactness: f32 variant max|Δ| vs fp64
**1.39e-4**, even closer to truth than bf16's 2.81e-4; within 2.05e-2 of gather).

### Instrumentation added (into the W47 prefill stage-timing receipt)

Nested brackets inside the dense path decompose the residual switch cost, and per-layer counters
record the dense/gather split — both no-ops off a `--prefill-stage-timing` session:

- `switch_breakdown`: `switch.dense.group_rows`, `switch.dense.dequant`, `switch.dense.matmul`,
  `switch.dense.scatter`, `switch.gather_qmm_fallback` (dequant/matmul fire once per densified
  expert, so `mean_ms` is per-expert).
- `switch_tallies`: `dense.calls`, `dense.experts_total`, `dense.experts_dense`,
  `dense.experts_under_threshold`, `dense.rows_dense`, `dense.rows_gather` (summed across the 40
  layer-major calls; divide by `dense.calls` for per-layer means).

So the next 16 K window shows exactly where the ~95 s went (dequant vs matmul vs scatter) and how
many rows/experts actually took the dense path at threshold 128/32.

### New A/B arms

`dense_min32` (threshold 32), `dense_batch16` (dequant batch 16), `dense_f32` (f32 matmul) — all
ride the layer-major schedule; keys pinned in every preset.

## How to run

CPU tests (no GPU, no `experts.bin`; ≤3 GB RSS):

```
PYTHONPATH=$PWD nice -n 19 .venv/bin/python3 -m pytest \
  tests/models/test_deepseek_v41_prefill_dense_experts.py \
  tests/test_deepseek_v41_ab_env_levers.py -p no:cacheprovider -q
```

GPU-window A/B (arms pin layer-major + dense on; token-identical to control at W20; add
`--prefill-stage-timing` for the switch breakdown + tallies):

```
... scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 16384 \
    --arms control layer_major prefill_dense_experts dense_min32 dense_f32 \
    --out <receipt.jsonl>
```

Tuning envs: `MTPLX_DSV41_PREFILL_DENSE_MIN_ROWS` (row threshold, default 128),
`MTPLX_DSV41_PREFILL_DENSE_BATCH` (experts per bounded dequant batch, default 8),
`MTPLX_DSV41_PREFILL_DENSE_MATMUL_DTYPE` (`bf16` default / `f32`).

## Files touched

- `mtplx/models/expert_mlx.py` — env constants + `_prefill_dense_experts_enabled` /
  `_positive_env_int` / `_prefill_dense_matmul_dtype`; `_dequantize_mxfp4_slot` (dtype-direct);
  `_run_component_bank_dense_prefill` (nested brackets + tallies + f32 variant);
  `_dispatch_component_bank(dense_prefill=...)`; `HotExpertSwitchGLU._run` gate + wiring.
- `mtplx/models/deepseek_v41_stage_timing.py` — `tally()` + `switch_tallies` export (additive).
- `scripts/deepseek_v41/ab_decode_env_levers.py` — four env keys pinned in every preset; arms
  `prefill_dense_experts`, `dense_min32`, `dense_batch16`, `dense_f32`.
- `tests/models/test_deepseek_v41_prefill_dense_experts.py` — CPU exactness + f32 variant +
  bracket/tally export + no-op tests.
- `tests/test_deepseek_v41_ab_env_levers.py` — arms + env-key dry-run coverage extended.
- `docs/deepseek-v41/KERNEL_LEDGER.md` — K26 entry (+ W20 result / follow-up).

## Open GPU-window question (KG-k)

Window 20 gave −20 s (not −70–80 s), token-identical, same 76.6 GB peak. The coverage estimate
says the threshold is not the cause (~90–97% of rows already route dense at 128). Remaining
questions for the next window: (a) do the new `switch_tallies` confirm >90% dense coverage at
threshold 128 (and does `dense_min32` change it materially); (b) does `dense_f32` recover the
modelled win — i.e. is the bf16 matmul the W50 slow-kernel path; (c) the
`switch.dense.{dequant,matmul,scatter}` split — how much of the residual is dequant DRAM vs
matmul; (d) `dense_batch16` peak/throughput trade. All CPU-provable pieces (numerics, counters,
no-op-off-session) are locked; only the realized GPU timing is open.
