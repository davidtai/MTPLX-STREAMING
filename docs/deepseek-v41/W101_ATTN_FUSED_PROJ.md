# W101 (K36) — Fused decode-attention PROJECTION-CHAIN kernels

Branch `w101/attn-fused-projections` (base `cdabd7db8`, the W97/W98/W99 attention
work). Lever `MTPLX_DSV41_ATTN_FUSED_PROJ` (default OFF, rounding-class, GPU-only,
small-M). Kernels module `mtplx/models/deepseek_v41_fused_proj_kernels.py`.

## 0. Verdict up front

W99 §8.5 left the exact-numerics decode-attention ceiling at ~55–67 kernels/layer and
said the remainder is the **qkv/out projection chains** — the rmsnorm + interleaved-RoPE
+ head/group layout **glue** between the (already single-dispatch) `mx.quantized_matmul`
projections and the grouped o-LoRA down-projection. W101 fuses that glue.

**The single largest win is the OUT-PREP `wo_a` re-layout, not compute.** Window-40's
in-model attribution (below) priced `out_proj` at **~4.9 ms/layer** — ~25× the 67 MB bf16
weight read (~0.2 ms). Graph inspection (§3) shows the cause: `mx.einsum("bsgd,grd->bsgr",
o, wo_a)` **re-lays-out (Transpose) the `[8,1024,4096]` = 33.55M-param `wo_a` operand every
token** (3 `Transpose` nodes in the einsum graph; a ~67 MB copy per layer per token).
Replacing the einsum with a batched `mx.matmul` over a weight **pre-transposed once and
cached** (`[g, in_per_group, o_lora_rank]`) removes that per-token copy — the weight is read
directly and only the tiny `o` is transposed. This is bit-identical to the reference bf16
einsum (model.py L784-787) and rounding-class vs the port's f32 einsum.

Measured dispatch collapse (graph primitives = a **lower bound** on Metal dispatches;
`--attn-proj` census, tiny real-structure model, counts shape-independent):

| chain | eager | K22-compiled (cell16k_ring baseline) | **W101 fused** |
|---|---:|---:|---:|
| qkv-prep | 33 | 17 | **6** (3 qmm + 3 fused kernels) |
| out-prep | 11 | 7 | **3** (2 matmul + 1 fused kernel) |
| **projection chains** | **44** | **24** | **9** |

The projection chains (this lever's scope) drop to **9 dispatches**. The whole-layer ≤25
target is NOT reached by fused-proj alone — the remainder is the attention **core** (a
separate lever, and ~0 ms in-model, §1) plus **structural** ops (gather / window-index /
cos-sin / KV-store) that are explicitly out of scope (§6).

## 1. Window-40 in-model attribution (the cost ranking this ordering follows)

GPU window 40 step 1 — in-model unfenced whole-token frame wall, `cell16k_ring` @16384, 64
steps, attention compile tapes OFF. Receipt: `docs/deepseek-v41/receipts/gpu-windows/window-40/attn-subops.json`.
`full = 503 ms/tok`; cost = `full − stubbed` (the stubs change decoded tokens, so the
expert-miss workload differs per pass — treat as **rankings**, not exact ms):

| sub-op | cost ms/tok | ≈ ms/layer | note |
|---|---:|---:|---|
| **out_proj** | **195** | **4.9** | largest; ≫ its 67 MB bf16 bandwidth (~0.2 ms) → a per-token re-layout (§3). GPU sits 18% busy / 338 MHz with it stubbed |
| **qkv_proj** | **148** | **3.7** | the projection-chain glue (rmsnorm + layout) |
| rope | 56 | 1.4 | the interleave/rejoin concats — folded into the qkv/out kernels here |
| attn_core | −34 | ≈0 | within the ±40 ms noise band → ~0 |
| wo_a_dequant | −26 | ≈0 | within noise → ~0 (the W97 cache already removes it) |

**Ordering W101 follows:** (1) out-prep first (biggest, and it is a re-layout not compute);
(2) qkv-prep; (3) rope folded into both kernels. The core is **not** composed with — it
measures ~0 in-model (matching window-27's SHELVED K29 verdict), so fused-proj is kept
**independent of K29**.

## 2. Per-dispatch list of ONE SIMPLE (swa_only) decode layer — BEFORE, classified

Eager decode `_attend`, real dims (`H=64, head_dim=512, rope=64, hidden=5120,
q_lora_rank=1280, o_lora_rank=1024, o_groups=8, k=640`). Class: **[M]** matmul kept ·
**[G]** fusable glue · **[S]** structural (out of scope) · **[R]** removable re-layout.

| # | op / helper | non-view kernels | class | disposition |
|---|---|---|---|---|
| 1 | `_cos_sin` (pos→f32, ×freq, cos, sin) | ~4 | **[S]** | per-layer; not a projection chain |
| 2 | `wq_a` (`quantized_matmul` 5120→1280) | 1 | **[M]** | kept |
| 3 | q-latent `_rmsnorm` (astype, square, sum, +eps, rsqrt, ×2 mul, astype) | ~7 | **[G]** | → fused kernel **A** (`rmsnorm`) |
| 4 | `wq_b` (`quantized_matmul` 1280→32768) | 1 | **[M]** | kept |
| 5 | q `_rope_last` (split, interleave mul×4 / sub / add, stack+rejoin concat×2, astype) | ~9 | **[G]** | → fused kernel **B** (`rope_heads`) |
| 6 | `wkv` (`quantized_matmul` 5120→512) | 1 | **[M]** | kept |
| 7 | kv-latent `_rmsnorm` | ~7 | **[G]** | → fused kernel **C** (`rmsnorm_rope`) |
| 8 | kv `_rope_last` (last 64 of the 512 latent) | ~5 | **[G]** | → fused kernel **C** (fused with 7) |
| 9 | `append_window` (concat kv_new into the store) | ~1 | **[S]** | KV store — out of scope |
| 10 | `_window_selected_idx` (maximum, add, ≤/</≥, astype int32) | ~5 | **[S]** | gather-index build — out of scope |
| 11 | `_gather_rows` (where, +offset, `take`) | ~3 | **[S]** | gather — out of scope |
| 12 | **core** (QK matmul, scale, mask where, max, sink maximum, exp, sum, sink exp, sub, divide, PV matmul) | ~13 | **[S]** | the attention core — separate lever (K29→1); ~0 ms in-model |
| 13 | o `_rope_last(inverse=True)` (de-rotate the query RoPE) | ~5 | **[G]** | → fused kernel **D** (`rope_heads` inverse) |
| 14 | o-LoRA down `einsum("bsgd,grd→bsgr")` | 1 matmul **+ 3 Transpose** | **[M]+[R]** | matmul kept; the per-token **weight Transpose is [R]** (§3) → matmul over a pre-transposed cached weight |
| 15 | `wo_b` (`quantized_matmul` 8192→5120) | 1 | **[M]** | kept |

The **[G]** glue (rows 3, 5, 7, 8, 13) and the **[R]** re-layout (row 14) are what W101
removes. The **[M]** matmuls stay MLX's tuned kernels; the **[S]** rows are structural and
out of scope.

## 3. Graph inspection — the o-LoRA einsum re-lays out `wo_a` every token

`mx.export_to_dot` of `mx.einsum("bsgd,grd->bsgr", o, w)` with `w = [8,1024,4096]`:

```
einsum(bsgd,grd->bsgr) graph ops: {'Transpose': 3, 'Reshape': 2, 'Matmul': 1}
```

Three `Transpose` nodes. One transposes the **weight** `[g, r, in] → [g, in, r]` so the
batched matmul can contract `in` — a **33.55M-element / 67 MB copy per token per layer** (2.7
GB/token over 40 layers), plus its dispatch. Re-specifying the einsum as `bsgd,gdr->bsgr`
with a pre-transposed `w` still emits 3 Transpose (einsum re-derives its own contraction
layout). The fix is to leave einsum entirely:

```
matmul(o[g,rows,in], wT[g,in,r]) graph ops: {'Reshape': 2, 'Transpose': 2, 'Matmul': 1}
```

with `wT = swapaxes(w,1,2)` **cached once** (`_o_lora_fused_weight`, materialised via
`mx.eval`, keyed on the packed-weight identity). The only Transposes left are on the tiny
`o` (`[rows≤8, g=8, in]`); the 67 MB weight is read directly. Numerics: `matmul(bf16, fp32
accumulate)` is **bit-identical** to the reference bf16 einsum and `max|Δ| = 2.6e-3` vs the
port's f32 einsum (rounding-class). This is the mechanism that turns the 4.9 ms/layer
out_proj back into a ~bandwidth-bound read. **A custom grouped GEMV was NOT written** — a
single batched `mx.matmul` expresses the grouped o-LoRA, and MLX's matmul beats a hand GEMV.

## 4. The fusion plan (four `mx.fast.metal_kernel`s + one cached-weight matmul)

Keep every `mx.quantized_matmul` (wq_a/wq_b/wkv/wo_b — MLX's tuned kernels, one dispatch
each). Fuse the glue:

- **A `rmsnorm`** — post-`wq_a` q-latent RMSNorm. One threadgroup/row; fp32 mean-of-squares
  tree reduction, `metal::precise::rsqrt`, ×fp32 weight, store at the activation dtype
  (reference `RMSNorm`, model.py L288-293).
- **B `rope_heads`** — post-`wq_b` interleaved-complex-pair RoPE on the last 64 dims of each
  of 64 heads. Pure elementwise (one thread/output scalar); folds the rope interleave +
  head-rejoin concats (reference `apply_rotary_emb` L392-406).
- **C `rmsnorm_rope`** — post-`wkv` KV-latent RMSNorm **and** the k_pe RoPE on the last 64
  dims of the 512-d latent, **fused in one pass** (one threadgroup/row).
- **D `rope_heads(inverse=True)`** — out-prep query-RoPE removal + `[b,s,H,hd]→[b,s,g,in]`
  group layout.
- **o-LoRA down**: batched `mx.matmul` over the pre-transposed cached bf16 weight (§3), **not**
  `mx.einsum`. `wo_b` stays the quantized matmul.

Reads are native dtype (bf16 activations promoted to `float` in-kernel via `float(x[i])`);
outputs `static_cast<T>` at the activation dtype — the proven idiom in
`mtplx/kernels/qsa_indexer_prepare.py` and `deepseek_v4.py`'s bf16 kernels. `metal::precise::`
arithmetic throughout (matching the K29 fix).

### Numerics class — rounding-class, by construction

- fused RMSNorm reassociates the fp32 sum-of-squares (threadgroup tree) — ~1e-6 class, far
  tighter than `mx.fast.rms_norm`'s measured ~2-ULP (1.56e-2) bf16 divergence, which is why
  a custom kernel is used and not `mx.fast.rms_norm`;
- the fused **C** kernel keeps the normed KV latent in **fp32 through the RoPE**, where the
  eager/reference path rounds it to bf16 first — slightly *more* accurate, still rounding-class;
- the o-LoRA matmul reads bf16 `wo_a` (reference dtype) and accumulates fp32.

**Never byte-identical** (no bit-for-bit test claims it). Composes with the W97 wo_a cache
(the fused path uses its OWN bf16 transposed cache, so it never calls `_o_lora_dense_weight`
— no conflict, and only the bf16 copy ≈2.7 GB is resident, HALF the f32 cache) and the W99
lean casts; **independent of K29** (touches different code).

### Which matmul kernels the fused path dispatches (mxfp8 gs32) — composes with W105

The artifact's attention projections are stored **native mxfp8, group_size 32, bits 8**
(config.json `quantization` default `mode: "mxfp8"`, `group_size: 32`, `bits: 8`; U8-packed
weights, **E8M0 (uint8) scales, `biases is None`** — a native float codec, NOT affine q8
gs64). The fused glue kernels A/B/C/D sit **between** these matmul dispatches; the fused path
does not change any matmul — it keeps `nn.QuantizedLinear.__call__`, so per decode layer it
issues:

| projection | call in the fused path | matmul kernel + mode |
|---|---|---|
| `wq_a` (5120→1280) | `self.wq_a(x)` in `_qkv_prep_fused` | `mx.quantized_matmul(mode="mxfp8", group_size=32, bits=8)` |
| `wq_b` (1280→32768) | `self.wq_b(qr)` in `_qkv_prep_fused` | `mx.quantized_matmul(mode="mxfp8", group_size=32, bits=8)` |
| `wkv` (5120→512) | `self.wkv(x)` in `_qkv_prep_fused` | `mx.quantized_matmul(mode="mxfp8", group_size=32, bits=8)` |
| o-LoRA down (grouped) | `mx.matmul(o, wT)` in `_out_prep_fused` | **bf16 `mx.matmul`** (fp32 accum) over `wo_a` **dequantized mxfp8→bf16** once + cached pre-transposed (§3); NOT `quantized_matmul`, NOT `einsum` |
| `wo_b` (8192→5120) | `self.wo_b(o)` in `_out_prep_fused` | `mx.quantized_matmul(mode="mxfp8", group_size=32, bits=8)` |

So the fused decode layer dispatches **4 `mx.quantized_matmul(mode="mxfp8", gs32, b8)` GEMVs**
(wq_a, wq_b, wkv, wo_b) + **1 bf16 `mx.matmul`** (grouped o-LoRA) + the **4 fused glue
metal_kernels** (A `rmsnorm`, B `rope_heads`, C `rmsnorm_rope`, D `rope_heads` inverse). **No
kernel change is needed** for the mxfp8 mode: the mode/group_size/bits flow from each
`nn.QuantizedLinear`'s own attributes (via `__call__`, and via `getattr(wo, "mode", "affine")`
in `_o_lora_fused_weight`'s dequant), so on the real mxfp8 artifact the fused path already
dispatches mxfp8 matmuls, while on the tiny test config (dense `nn.Linear`, dims not
gs32-aligned) it dispatches dense `x @ w.T`.

**Composition with W105.** W105 measures whether the **mxfp8 small-M (`b*s≤8`) GEMV** is
itself the dominant decode-attention cost. Those are exactly the 4 `mx.quantized_matmul(
mode="mxfp8")` dispatches this fused path **keeps** (plus, for W105's accounting, note the
o-LoRA leg is NOT one of them — it is the bf16 `mx.matmul`). W101's scope is the layout/norm/
rope **glue between** the matmuls: W101 removes those dispatches; W105 prices the matmul
dispatches W101 keeps. The two results add — decode-attention cost ≈ (mxfp8 GEMVs, W105's
subject) + (fused glue + bf16 o-LoRA matmul, W101's subject).

## 5. Measured before/after kernel counts (`scripts/deepseek_v41/dispatch_census.py --attn-proj`)

Non-view graph primitives (a **lower bound** on Metal dispatches — `Concatenate`/`Slice`/
reduction copies each cost ≥1 real dispatch). Tiny real-structure model; counts are
shape-independent, so they equal the artifact's per-layer counts. Full-layer K22 = 122
matches W99 §8.4 exactly (method validated).

### Region (definitive, structure-fixed)

| chain | eager | K22-compiled | **fused** | fused breakdown |
|---|---:|---:|---:|---|
| qkv-prep | 33 | 17 | **6** | 3 quantized matmuls + 3 fused kernels (A,B,C) |
| out-prep | 11 | 7 | **3** | o-LoRA matmul + wo_b matmul + 1 fused kernel (D) |

### Whole decode layer (per-call, 1 token)

| mode | eager | K22 (cell16k_ring) | K22 + fused-proj | + K29 core |
|---|---:|---:|---:|---:|
| swa_only (simple) | 75 | 55 | **40** | 28 |
| reuse (simple) | 82 | 62 | **47** | 35 |
| reindex (full) | 144 | 124 | 109 | 97 |
| full | 142 | 122 | **106** | 94 |

**The ≤25 whole-layer target is not reached** (best simple-layer = 28 with K29, 40 without).
The projection chains — this lever's scope — reach **9 dispatches** (from 24 K22 / 44 eager).
The residual ~28–40 is the **core** (K29 → 1, but ~0 ms in-model so not relied on) plus the
**structural** ops (gather, window-index, cos/sin, KV-store) that §6 keeps out of scope. Per
the window-40 ranking, the two dominant *cost* items (out_proj re-layout, qkv glue) are the
ones fused-proj removes.

## 6. Scope

W101 touches ONLY the decode-attention projection chains (qkv-prep, out-prep). No MoE, no
routing, no indexer, no core. The structural ops (gather/sort/KV-store, cos/sin, window-index)
are classified above but left to their own frontiers.

## 7. Numerics labels (GPU — tiny model on Metal)

Measured under the exclusive GPU lock (`MTPLX_DSV41_GPU_TESTS=1`), mlx 0.32.2, tiny
real-structure model, from `tests/test_deepseek_v41_attn_fused_proj_w101_gpu.py`. Receipts:
`.benchmark-artifacts/deepseek-v41/w101_gpu_numerics_2026-09-12_run1.log` (**run1**,
2026-09-12T12:08:50Z: rmsnorm/rmsnorm_rope kernels, whole-`_attend`, greedy identity — 9
passed, 4 failed, see below) and the 2026-09-12T13:40:21Z `gpu_window.sh` rerun (`gap1` window,
step 2: `pytest tests/test_deepseek_v41_attn_fused_proj_w101_gpu.py -x -q`) — **13 passed, 0
failed**, the current state of this worktree's test file. CPU fallback is byte-identical to
lever-off over 64 decode steps (the fused path is GPU-only; CPU runs the eager chain — proven
in the CPU test, `13 passed`). **Rounding-class by construction; no bit-for-bit test, so never
byte-identical** (§4).

**Note on the `rope_heads` numbers (§7.1 below).** At run1, `test_rope_heads_kernel_vs_reference`
(4 parametrizations: fwd/inverse x bf16/f32) crashed before computing a delta: `q` was built as
a flat 3-D `(3, H, hd)` tensor, but `rope_heads_reference` → `_rope_last_reference` only
supports the production 4-D `[b, s, H, hd]` layout, so its `cos`/`sin` reshape could not
broadcast (`ValueError: [broadcast_shapes] Shapes (3,64,32) and (3,32) cannot be broadcast`) —
this is the run1 "4 failed". That shape bug is fixed in this worktree's dirty test file (`q =
mx.random.normal((1, 3, H, hd))`, the 4-D layout — see §7.1's closing note), and the
2026-09-12T13:40:21Z rerun passed all 13 tests, including the 4 `rope_heads` parametrizations.
But that rerun's pytest invocation (`-x -q`, no `-s`) captures stdout, and pytest discards
captured stdout for passing tests, so the per-test `[W101 kernel] rope_heads ... max|Δ|=...`
print lines are not in that log either. **Net: no log on disk currently holds the 4 `rope_heads`
max|Δ| numbers** — run1 crashed before printing them (no fallback value), and the only run
where the fixed test passed did not preserve prints. Table below labels those 4 cells
accordingly; producing real numbers needs a GPU rerun with `-s`, out of scope for this
mechanical doc-fill pass (no GPU use).

### 7.1 Per-kernel fused kernel vs pure-MLX reference (max|Δ|)

| fused kernel | bf16 max\|Δ\| (band 1.5e-2) | f32 max\|Δ\| (band 1e-3) |
|---|---:|---:|
| **A `rmsnorm`** (q-latent; d = 1280 / 512 / 20) | 0.0 / 0.0 / 0.0 | 4.77e-7 / 2.38e-7 / 0.0 |
| **C `rmsnorm_rope`** (512-d KV latent + k_pe RoPE) | 7.78e-3 | 7.15e-7 |
| **B/D `rope_heads`** (fwd, `[b=1,s=3,H=64,hd=512]`) | not captured † | not captured † |
| **D `rope_heads`** (inverse, same shape) | not captured † | not captured † |

† 13/13 passed at the 2026-09-12T13:40:21Z rerun (all 4 `rope_heads` parametrizations included),
but that run's stdout was captured (no `-s`), so no `max|Δ|` was printed to the log; run1
crashed on this test pre-fix (see note above) before computing a delta, so there is no run1
fallback value either. Both bands are unchanged (1.5e-2 bf16 / 1e-3 f32) and the assertions in
the fixed test enforce them — the 13-pass result is proof both deltas are `<=` their band, just
not a captured numeric value.

RMSNorm (kernel A) is **exact in bf16** (max\|Δ\| = 0.0 — the fp32 mean-of-squares tree +
`metal::precise::rsqrt` + bf16 store lands on the identical bf16 value as the MLX reference)
and pure-reassociation in f32 (≤4.77e-7). `rmsnorm_rope` (C) is 7.78e-3 bf16 (≈ one bf16 ULP
on the RoPE'd 512-latent) / 7.15e-7 f32. All inside the reassociation band — rounding-class as
designed. (The `rope_heads` per-kernel row is exercised at the **production 4-D layout**
`[b,s,H,hd]`, the only layout the eager `_rope_last` / the oracle are defined for; the flat
`[rows,H,hd]` layout is accepted by the kernel alone — this worktree's dirty test file switched
the standalone `rope_heads` test from the flat 3-D shape to `[b=1,s=3,H=64,hd=512]` for exactly
this reason.)

### 7.2 Whole fused `_attend` vs eager (lever OFF), real 1024-context decode geometry, per mode

| mode | max\|Δ\| | engagement (qkv / out / rows, fallbacks) | band |
|---|---:|---|---:|
| swa_only (simple) | **4.73e-4** | 1 / 1 / 1, 0 fallbacks | ≤ 5e-2 |
| reuse (simple)    | **9.48e-5** | 1 / 1 / 1, 0 fallbacks | ≤ 5e-2 |
| full (indexer)    | **2.02e-4** | 1 / 1 / 1, 0 fallbacks | ≤ 3e-1 |

All three are ~1e-4 — far below even the tight swa_only/reuse 5e-2 band. The `full` (indexer
top-k) mode did **not** trigger a compressed-key selection flip at this geometry (2.02e-4, not
the ~1e-1 a whole-key swap would produce). The engagement counter shows the fused qkv **and**
out kernels dispatched on every call (`qkv_calls = out_calls = 1`, `rows = 1` for the `b*s=1`
decode step, `0` fallbacks) — proof the GPU fused path ran, not the eager CPU fallback.

### 7.3 Greedy-argmax identity over 64 decode steps (tiny full model, lever ON vs OFF)

**65 / 65 identical** (prompt-final token + 64 decode steps), **0 flips** — the fused decode
path reproduces the lever-off greedy trajectory token-for-token on this model. No flips to
label (a rounding-class lever is *allowed* near-tie flips; there were none here).

### 7.4 Numerics class + dispatch count

**Rounding-class, never byte-identical.** The fused RMSNorm reassociates the fp32
sum-of-squares (tree), kernel C keeps the KV latent in fp32 through the RoPE, and the o-LoRA
leg reads bf16 `wo_a` (the reference dtype) + fp32-accumulates. Every measured delta (per-kernel
≤ 7.78e-3 bf16 / ≤ 4.77e-7 f32; whole `_attend` ≤ 4.73e-4; 65/65 greedy identity) sits inside
the reassociation / bf16 bands — the lever is rounding-class as designed (§4), not bit-exact.

**Dispatch count.** This GPU test captures the **engagement counter** (fused kernels
dispatched: `qkv_calls = out_calls = 1` per layer-step, 0 fallbacks — proof the fused path ran
on Metal), **not** a raw Metal command-buffer count. The before/after dispatch **collapse** is
the §5 census (`scripts/deepseek_v41/dispatch_census.py --attn-proj`; shape-independent graph
primitives, a lower bound on Metal dispatches): projection chains **44 eager → 24 K22 → 9
fused** (qkv-prep 33→17→6, out-prep 11→7→3); whole simple decode layer **75 → 55 → 40**. Those
counts are the census's and are unchanged by this run.

## 8. Arms + tests

- Lever `MTPLX_DSV41_ATTN_FUSED_PROJ` (default OFF, rounding-class, GPU-only, small-M `b*s≤8`),
  read at use. Engagement counter `fused_proj_engagement` (qkv_calls / out_calls / rows /
  fallbacks) lands in the ab decode receipts beside `decode_attn_kernel_engagement`.
- Arms (`scripts/deepseek_v41/ab_decode_env_levers.py`):
  - `attn_fused_proj` — isolation (`selected_keys` + fused proj; core eager — independent of K29);
  - `cell16k_ring_fused` — the full attention dispatch stack (cell16k_ring + wo_a cache + lean
    casts + K29 + fused proj). Note K29 measures ~0 ms in-model; the win is the projection chains.
- Tests: `tests/test_deepseek_v41_attn_fused_proj_w101.py` (CPU: flag/gate/fallback/engagement/
  census kernel drop/transposed-cached weight/references) and
  `tests/test_deepseek_v41_attn_fused_proj_w101_gpu.py` (GPU numerics, under the lock).

## 9. Evidence base

- Window-40 attribution: `docs/deepseek-v41/receipts/gpu-windows/window-40/attn-subops.json`.
- W97/W99: `docs/deepseek-v41/W97_ATTENTION_291MS.md` §7 (dispatch census), §8 (casts/concats,
  and §8.5 the "fuse the projection chains" recommendation this window executes).
- Reference math: model.py L288-293 (RMSNorm), L392-406 (apply_rotary_emb), L784-787 (bf16
  o-LoRA einsum).
- `mtplx/kernels/qsa_indexer_prepare.py` — the proven bf16-activation rmsnorm+rope metal_kernel
  idiom (`float(x[i])` reads, `metal::precise::rsqrt`, `static_cast<T>` writes).
