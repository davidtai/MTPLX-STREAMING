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

## How to run

CPU tests (no GPU, no `experts.bin`; ≤3 GB RSS):

```
PYTHONPATH=$PWD nice -n 19 .venv/bin/python3 -m pytest \
  tests/models/test_deepseek_v41_prefill_dense_experts.py \
  tests/test_deepseek_v41_ab_env_levers.py -p no:cacheprovider -q
```

GPU-window A/B (arm pins layer-major + dense on; not byte-identical to control, expected):

```
... scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 16384 \
    --arms control layer_major prefill_dense_experts --out <receipt.jsonl>
```

Tuning envs: `MTPLX_DSV41_PREFILL_DENSE_MIN_ROWS` (per-expert row threshold, default 128),
`MTPLX_DSV41_PREFILL_DENSE_BATCH` (experts dequantized per bounded batch, default 8).

## Files touched

- `mtplx/models/expert_mlx.py` — env constants + `_prefill_dense_experts_enabled` /
  `_positive_env_int`; `_dequantize_mxfp4_slot`; `_run_component_bank_dense_prefill`;
  `_dispatch_component_bank(dense_prefill=...)`; `HotExpertSwitchGLU._run` gate + wiring.
- `scripts/deepseek_v41/ab_decode_env_levers.py` — three env keys pinned in every preset;
  arm `prefill_dense_experts`.
- `tests/models/test_deepseek_v41_prefill_dense_experts.py` — new CPU exactness + no-op tests.
- `tests/test_deepseek_v41_ab_env_levers.py` — arm + env-key dry-run coverage extended.
- `docs/deepseek-v41/KERNEL_LEDGER.md` — K26 entry.

## Open GPU-window question (KG-k)

Realized 16 K prefill delta is unmeasured: the analytical model assumes ~all experts clear
128 rows and dense runs at 15–25 TFLOPS on the real 5120×2304 experts. The window measures
(a) the actual per-expert row distribution at 16 K (how many clear the threshold), (b) the
realized dense TFLOPS and thus the crossover M, (c) `moe.routed_switch` and TTFT vs the
`layer_major` control at David's standard shape, and (d) whether the bf16 divergence flips
any prompt-token argmax (it should not — prefill feeds the same greedy head).
