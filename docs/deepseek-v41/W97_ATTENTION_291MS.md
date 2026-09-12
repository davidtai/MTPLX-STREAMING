# W97 — DeepSeek-V4.1-Flash in-model decode attention: the 291 ms/token

Branch `w97/attention-291ms` (base `503beb96e`, window-38 receipts). CPU-only analysis
+ a window-runnable sub-op bisect; no GPU touched by this worker.

## 0. Verdict up front (read this first)

The window-38 unfenced attribution prices in-model decode **attention at 290.8 ms/token**
(`full 486.8 − attn-stubbed 195.9`), ~7.3 ms/layer over the 40 backbone layers, while the
isolated Metal microbench of the same entry points measures ~2.0 ms/layer. The task framed
the ~5 ms/layer gap as "~3–7 GFLOP or ~3 GB of traffic per layer somewhere" and floated
un-absorbing (up-projecting 640 keys × 64 heads ≈ 10.7 GFLOP/layer) as the prime suspect.

**That hypothesis is false for this port, and the gap is not one big compute op.** The
per-op table below shows attention does **337 MFLOP/layer** total (≈0.03 ms/layer of
compute) — the port is already MLA-**absorbed** exactly like the reference (one shared
512-d KV latent scored against all 64 query heads; no per-head K/V up-projection anywhere).
There is no 3–7 GFLOP op to find.

The 291 ms decomposes as:

1. **The one real per-token traffic defect (attention-local, fixable): the grouped o-LoRA
   `wo_a` down-projection re-issues `mx.dequantize(wo_a)` every layer every token, then
   promotes the result to f32 every token.** At the released dims `wo_a` is `[8, 1024, 4096]`
   = 33.55M params. `mx.dequantize` returns the **scale dtype = bf16** for BOTH resident
   codecs (affine-q8 has bf16 scales/biases; the native mxfp4/mxfp8/nvfp4 banks are bf16-scale)
   — probe: `[8192,4096]` q8 gs64 → **bf16, 67.1 MB**. `_o_lora_down` (and the K22 out tape)
   then run `.astype(mx.float32)`, materialising a fresh **134 MB f32** array **per token per
   layer** for both codecs and running the einsum in f32 — the per-token cost is the dequant
   (67 MB bf16 write) **plus** the f32 astype (**134 MB write + 134 MB read = ~10.7 GB/token**
   over 40 layers). The reference (model.py L784-787) dequantizes `wo_a` **once at convert time
   to bf16** and runs the einsum in bf16 (67 MB/layer, 2.68 GB/token). At the gappy in-situ
   bandwidth this is ~1 ms/layer and, more importantly, 40 extra dequant **dispatches**/token —
   the single largest attention compute/traffic item, and one the bf16 isolated bench (dense
   `wo_a`, no dequant) structurally cannot see. **Confirmed by the W96 as-is audit (finding
   L9).** Fixed here (§5) by caching the **f32** array (so the per-token dequant AND astype
   both vanish), byte-identically, behind `MTPLX_DSV41_ATTN_WO_A_CACHE`.

2. **The dominant remainder (~200 ms/token) is exposed host-encode latency, not attention
   compute.** Attention issues ~100–270 tiny dispatches/layer (norms, RoPE, gather, the
   grouped einsum, the projections). In the isolated bench these **pipeline** — host-encode
   overlaps GPU execution — so 40 layers cost ~2 ms/layer. In the served decode the per-layer
   routing barrier `mx.eval(indices)` **drains the pipeline every layer** (W96 finding #1/#2:
   ≥315 ms/token, ≥7.9 ms/barrier, GPU 58% *used* despite 94% *active*), so attention's
   dispatches cannot hide — each pays full host-encode serially and the GPU idles between
   them. Stubbing attention removes those dispatches (and their exposed encode), which is
   why `full − attn-stub` reads 291 ms. **This is an architecture-level issue (the barrier),
   already the subject of the `w95/barrier-free-resident` line — it is NOT inside the
   attention module and is out of scope for an attention-local fix.** The `wo_a` cache helps
   it (40 fewer dispatches/token) but does not remove it.

The `--attn-subops` extension of the W94 tool (§4) lets window 39 confirm this split
empirically in one run.

## 1. Per-op FLOP / byte table — one decode token, per backbone layer, released dims

Dims: `H=64, head_dim=512, rope_head_dim=64, hidden=5120, q_lora_rank=1280, o_lora_rank=1024,
o_groups=8, window=128, index_topk=512 → k=640 selected keys (bounded, context-independent)`.
Weight bytes shown for the two resident codecs (q8 gs64 affine = default `_RESIDENT_QUANT`;
mxfp4 gs32 = the native DSV4.1 bank) and the reference (bf16, dequantized once at convert).

| op | FLOP | wt q8 | wt mxfp4 | wt bf16 (ref) |
|---|---:|---:|---:|---:|
| wq_a  (q down 5120→1280)            |  13.11 MFLOP | 6.96 MB | 3.48 MB | 13.11 MB |
| wq_b  (q up 1280→32768)             |  83.89 MFLOP | 44.56 MB | 22.28 MB | 83.89 MB |
| wkv   (kv 5120→512, shared latent)  |   5.24 MFLOP | 2.79 MB | 1.39 MB | 5.24 MB |
| QK^T  (absorbed [H,k], f32)         |  41.94 MFLOP | – | – | – |
| PV    (absorbed [H,hd], f32)        |  41.94 MFLOP | – | – | – |
| KVg gather (640×512 bf16)           |   0          | 0.66 MB | 0.66 MB | 0.66 MB |
| wo_a einsum (grouped bsgd,grd→bsgr) |  67.11 MFLOP | (weight below) | | |
| wo_b  (8192→5120)                   |  83.89 MFLOP | 44.56 MB | 22.28 MB | 83.89 MB |
| **TOTAL matmul FLOP**               | **337.12 MFLOP** | | | |

337 MFLOP/layer × 40 = 13.5 GFLOP/token — at any plausible M5 Max f32 rate this is
sub-millisecond for the whole token. **Attention is nowhere near compute-bound.**

### The `wo_a` o-LoRA down-projection — the per-token materialization

`wo_a` = `[o_groups·o_lora_rank, in_per_group]` = `[8192, 4096]` = 33.55M params.

`mx.dequantize` returns the **scale dtype = bf16** for both resident codecs, so the dequant
output is **67.1 MB bf16**; `_o_lora_down` (and the K22 out tape) then `.astype(mx.float32)` it
to **134.2 MB f32** every token before the einsum. So the per-token cost is the same for both
codecs — a dequant plus an f32 astype — differing only in the packed read size:

| path | per token / layer | × 40 layers |
|---|---|---:|
| **port, q8 gs64** (default) | dequant read ~35.7 MB (packed+scales) + dequant write **67 MB bf16** + astype write **134 MB f32** + einsum read **134 MB f32** | **~14 GB/token** |
| **port, mxfp4 gs32** (native bank) | dequant read ~18.9 MB (4-bit packed+scales) + dequant write **67 MB bf16** + astype write **134 MB f32** + einsum read **134 MB f32** | **~13.4 GB/token** |
| **reference** (bf16, dequantized once) | einsum read (bf16) **67 MB** (no per-token dequant / astype) | **2.68 GB/token** |

Of that per-token cost the **f32 astype alone is 134 MB write + 134 MB read = 268 MB/layer ×
40 = ~10.7 GB/token for BOTH codecs** — the traffic the earlier lever left in place (it cached
the *bf16* dequant and the per-token astype still fired). The fix (§5) caches the **f32** array,
so both the dequant dispatch and the astype vanish.

Verified with `mx.dequantize` on the real shapes: q8 gs64 → **bf16 67.11 MB** output; mxfp4/mxfp8
gs32 → **bf16 67.11 MB** output (then `_o_lora_down`'s `.astype(f32)` promotes it to 134.22 MB).

### The compressor frontier (W96 finding #6) — real but O(T), only 4 layers

`CompressorState.push` (`deepseek_v41_cache.py:648-649`) `mx.concatenate`s the **full fp32
`[1,T,512]` history** for BOTH `raw_kv` and `raw_score` every token, on the 4 kv-source
layers (2, 8, 14, 20): 4.2 MB/token/layer × 4 = 17 MB/token at T=1024; **67 MB/token/layer
× 4 = 268 MB/token at T=16384**. Only the last `ratio` (=2) rows are needed to complete the
next group; the history is retained only for speculative trim/rollback (unused at AR decode).
This is O(T) and confined to 4 layers, so it is NOT the constant-in-context 5 ms/layer — but
it is a genuine defect worth a separate frontier-window fix.

## 2. Diff vs the reference (`/Users/davidtai/models/DeepSeek-V4.1-Flash-src/inference/model.py`)

| aspect | reference (`Attention.forward` L765-789) | port (`Attention._attend`) | same? |
|---|---|---|---|
| KV scoring | `sparse_attn(q[b,s,64,512], kv[b,T,512])` — **absorbed**, one shared latent per position, 64 query heads | `_sparse_attend[_selected]`: `einsum bshd,bskd→bshk`, KVg shared across heads — **absorbed** | ✅ identical shape/algebra; **no un-absorb, no per-head K/V up-projection in either** |
| decode key set | window 128 + index_topk 512 selected | window 128 + index_topk 512 selected (K30 gather) | ✅ bounded k=640, context-independent |
| score/softmax/PV dtype | f32 | f32 | ✅ |
| q/kv/o_b projections | fp8 GEMM | `mx.quantized_matmul` (mxfp4/q8), single primitive | ✅ (port reads *less* weight) |
| **o-LoRA `wo_a`** | **dequantized ONCE at convert to bf16; einsum in bf16** | **`mx.dequantize` per token → f32; einsum in f32** | ❌ **the divergence** — port does 5–6× the `wo_a` traffic and 40 dequant dispatches/token |
| compressor frontier | reference keeps only the `ratio` decode-state rows (L449-456) | concatenates the full fp32 history every token (kv-source layers) | ❌ O(T) copy (W96 #6) |

The reference's "K/V up-projection per head" that the task hypothesised **does not exist** in
either path — both score the compressed latent directly (MLA absorption). The `wo_a`
dequant-precision divergence is the real per-token compute/traffic gap.

## 3. Why the isolated bench (`metal_decode_attn_bisect.py`) misses ~5 ms/layer

Two structural differences between `build_case`'s layer and the loader's real construction:

1. **`_build_layer` calls `attn.set_dtype(mx.bfloat16)`** (metal_decode_attn_bisect.py:250) —
   plain **bf16** `nn.Linear` weights, never quantised. So `wo_a` is a dense `nn.Linear` and
   `_o_lora_dense_weight()` takes the `else` branch: **no `mx.dequantize` at all**, ever. The
   bench never issues the 40 dequant dispatches or the 67 MB bf16 dequant write per layer.
   (It still pays the f32 einsum via `_o_lora_down`'s `.astype(f32)`, so it sees the einsum
   read but not the dequant.) The projections are bf16 dense matmuls (bandwidth-bound, fast at
   M=1) rather than mxfp4 `quantized_matmul` (ALU-bound at M=1) — another construction diff.

2. **No routing barrier between the bench's layers.** The bench times one `Attention._attend`
   in a tight loop with `mx.eval` fences; there is no per-layer `mx.eval(indices)` draining the
   pipeline. Attention's ~100 small dispatches therefore **pipeline** (host-encode overlaps GPU
   execution) → ~2 ms/layer. In the served decode the barrier drains every layer, so the same
   dispatches are **serialised** behind exposed host-encode and the GPU idles between them —
   the ~5 ms/layer that only exists in situ. This is the W96 thesis (68% busy at 1.13 GHz;
   ≥315 ms/token inside `mx.eval(indices)`) and is why the fenced per-stage census reads
   ~7.5 ms/layer (drain latency) while the kernels are ~2 ms.

Neither difference is a bug in the bench — they are exactly the two things the released model
does that a single-layer tight loop cannot reproduce.

## 4. Window-39 command — within-attention sub-op attribution (`--attn-subops`)

The W94 unfenced tool now has a within-attention microscope: a full (eager-attention)
baseline plus one unfenced whole-token frame-wall pass per attention sub-op stubbed to a
shape-preserving pass, so `full − <subop>` is that sub-op's true in-situ cost (its kernels
**plus** the exposed host-encode of its dispatches). Sub-ops (this port is absorbed, so the
reference's per-head K/V up-projection sub-op is N/A): `qkv_proj`, `attn_core` (gather + QK^T +
sink softmax + PV), `wo_a_dequant` (the exact `MTPLX_DSV41_ATTN_WO_A_CACHE` fix — `full − this`
= the per-token dequant cost), `out_proj` (whole grouped `wo_a` einsum + `wo_b`; `out_proj −
wo_a_dequant` isolates the einsum+wo_b from the dequant), `rope`. The attention compile tapes
are forced OFF for the microscope (the compiled qkv/out tapes evaluate their weight-array
inputs at the `_attend` call site before any builder stub could run, so the eager path is the
faithful, patchable microscope; the 5-pass compiled arm still owns the absolute 291 ms).

Run through the flock (`scripts/deepseek_v41/gpu_window.sh`) like every Metal exec on this box;
with `PY` the venv python3, `MODEL` the streaming artifact, `OUT` the window dir:

```
nice -n 19 $PY scripts/deepseek_v41/metal_decode_attn_bisect.py \
  --in-model --unfenced --attn-subops --gpu --model $MODEL --arms cell16k_ring \
  --context-tokens 16384 --memory-limit-gib 60 --max-kv 17408 \
  --in-model-steps 64 --warmup-steps 8 --utilization \
  --out $OUT/in-model-16k-attn-subops.json
```

Validate the plumbing first on CPU (no GPU, no artifact):

```
$PY scripts/deepseek_v41/metal_decode_attn_bisect.py \
  --in-model --unfenced --attn-subops --tiny --in-model-steps 6 --warmup-steps 2
```

**Expected read (the thesis):** if the gap is exposed host-encode (dispatch-bound), each
sub-op's cost tracks its dispatch count and `wo_a_dequant` alone is ~1 ms/layer (the traffic
+ 40 dispatches); if it were a big compute op, one sub-op would dominate disproportionately.
The tiny CPU run already shows `rope` costing ~1.4 ms on a 32-d model — pure dispatch, no
compute — which is the microscope working.

## 5. The fix — `MTPLX_DSV41_ATTN_WO_A_CACHE` (byte-identical, default OFF)

`Attention._o_lora_dense_weight` (`deepseek_v41.py`) now memoises the dequantized `wo_a`
weight per layer when the lever is on, **promoted to f32**, keyed on the packed-weight identity
(a re-quantize / reload rebuilds it), materialised once via `mx.eval` so later tokens reference
the buffer rather than a lazy dequantize node. `mx.dequantize` returns **bf16** for both codecs
and `_o_lora_down` / the K22 out tape promote it to f32 with `.astype(mx.float32)`; caching that
exact f32 promotion (**bf16→f32 is lossless**) is **byte-identical to control** — no fp math is
reordered — and makes the per-token `.astype(mx.float32)` a graph no-op (verified: 0 `AsType` on
the weight leg). It therefore removes BOTH the per-token dequant dispatch (40/token) AND the
per-token f32 astype (write+read ~10.7 GB/token, both codecs) that the earlier bf16 cache left
in place. Default OFF and opt-in because the **f32** cache holds a dense f32 copy of every
layer's `wo_a` resident — **40 × [8192,4096]·4 = 40 × 134 MB ≈ 5.4 GB for BOTH codecs** (the
codec only affects the packed source, not the cached f32 size) — so it must be priced into the
memory plan (`mtplx/models/deepseek_v41_loader.py`) and minded against the box memory budget
([[never-exceed-the-memory-knob]]; the 60 GB runner already peaks ~65 GB at window-38).

A lower-memory, reference-matching variant (cache once as **bf16** and run the einsum in bf16,
2.7 GB and half the einsum-read traffic) is a rounding-class change vs the current f32 einsum —
the reference numerics — and is left as a follow-up decision for Fable/David given the
memory/precision trade.

### Tests (CPU, tiny, nice -n 19)

- `tests/test_deepseek_v41_wo_a_cache_w97.py`
  - `test_wo_a_cache_byte_identical_64_steps[compile off|on]` — 64 decode steps through the
    production `_attend` are bit-for-bit identical with the lever OFF vs ON, eager and under
    the K22 attention compile tape.
  - `test_wo_a_dequant_issued_once_when_cached` — `mx.dequantize` is invoked once per decode
    step with the lever OFF (N per N steps) and **exactly once** for the whole run with it ON
    (the per-token work is gone).
  - `test_wo_a_cache_rebuilds_on_requantize` — the cache keys on the packed-weight identity;
    swapping `wo_a` rebuilds it.
- `tests/models/test_metal_decode_attn_bisect.py` (extended) — the `--attn-subops` mode:
  all-passes-run, `cost == full − stubbed`, per-seam apply/restore is byte-identical, the
  absorbed-port sub-op list (no K/V up-projection), CLI gating, and `main` writes the receipt.

## 6. Evidence base

- `docs/deepseek-v41/receipts/gpu-windows/window-38/unfenced-attribution.json` — attention
  290.835 ms/token; full 486.8; SSD-bound 328.7 ms est; GPU 58% used / 94% active, 990 MHz.
- W96 as-is audit (`w95/barrier-free-resident:docs/deepseek-v41/W96_RUNNER_AS_IS_AUDIT.md`) —
  findings #1/#2 (≈140 host syncs/token, ≥315 ms in `mx.eval(indices)`), #6 (frontier concat),
  L9 (`mx.dequantize(wo_a)` per layer per token).
- `mtplx/models/deepseek_v41.py` `Attention` (absorbed MLA), `_o_lora_dense_weight`/`_o_lora_down`.
- Reference `Attention.forward` L765-789 (bf16 `wo_a` einsum, dequantized once at convert).
- CPU arithmetic table + `mx.dequantize` dtype/byte probe (this worker; §1).

## 7. Follow-on (window 40): decode-attention dispatch COUNT — the core collapse

Since attention compute is ~0.03 ms/layer and the cost is the host-encode of ~100
tiny dispatches that cannot pipeline behind the per-layer routing sync (§0), the
lever is dispatch **count**. Measured with the W91 pattern (`mx.export_to_dot` graph
primitives = the dispatch proxy; `mx.compile` collapses elementwise chains into
`Compiled` nodes) on the tiny real-structure model, but at the **real decode
geometry** for the core (primitive count is shape-independent).

### 7.1 Dispatch table — per decode attention layer (M=1, `ATTN_COMPILE` on, selected-keys)

Whole-layer graph primitives per CSA mode (one decode `_attend`, K22 qkv/out tapes on):

| mode | graph prims | non-view kernels | top kernel ops |
|---|---:|---:|---|
| swa_only | 162 | 69 | AsType 15, Matmul 7, Concatenate 6, +core elementwise |
| reindex  | 162 | 69 | (same shape as swa/reuse at M=1 selected) |
| reuse    | 162 | 69 | AsType 15, Matmul 7, Concatenate 6 |
| full     | 283 | 122 | AsType 25, Matmul 10, Concatenate 10, Gather 2, +indexer sort/cumsum |

`mx.compile` cannot fuse Matmul / Concatenate / reductions / the data-dependent
gather+sort, so the **whole layer stays ~69–122 dispatches even with qkv/out
compiled** — the ≤15/layer target is NOT reachable by compiling alone. The single
largest kernel op is **`AsType` (15 simple / 25 full)** — redundant f32 casts
scattered through score/PV/rmsnorm/o-LoRA (a separate cast-reduction lever).

### 7.2 The compilable/kernelizable chunk — the fixed-shape CORE

The selected-key core (QK^T + mask + per-head value-0 sink + f32 softmax + PV over
the gathered `[b,s,k,hd]` operand) has FIXED shapes at decode (`k` = 640 for the
compress modes / 128 for swa_only; `hd` 512; `H` 64). Census (`dispatch_census.py
--attn-core`):

| core path | graph prims | non-view kernels | dispatches | Δ vs eager |
|---|---:|---:|---:|---|
| eager | 26 | 13 (2 matmul + scale/where/max/max/2×sub/2×exp/sum/add/divide) | ~13 | — (exact) |
| `mx.compile` (`MTPLX_DSV41_ATTN_CORE_COMPILE`) | 16 | 8 (2 matmul + 3 fused `Compiled` + Max + Maximum + Sum) | ~8 | **rounding-class**, max\|Δ\| 9.3e-10 (CPU) |
| K29 fused kernel (`MTPLX_DSV41_DECODE_ATTN_KERNEL`) | — | — | **1** (score+mask+softmax+PV, one `metal_kernel`) | rounding-class, GPU-only |

**Which is lower: the K29 fused kernel (1 dispatch) beats the `mx.compile` core
(~8).** K29 wins the early return in `_sparse_attend_selected` before the compile
path, so the two never both apply; K29 is GPU-only (CPU falls back to eager) and the
compile core is the portable CPU+GPU fallback / A-B. Both are **rounding-class**
(neither is byte-identical: the n=1 compile and the K29 tile reduction each
reassociate the fp32 einsum/reductions — the K35 lesson), gated separately from the
exact levers (`wo_a_cache` is the only byte-identical W97 attention lever).

The core compile is keyed on geometry (`_ATTN_CORE_COMPILED`), verified bounded: 64
decode steps build ≤ a handful of tapes (one per distinct `k`), never one per token.
Combining K29 (core → 1) with the `wo_a` cache (out-proj dequant → 0/token) and the
existing qkv/out K22 tapes brings a simple (swa/reuse) layer's *substantial-compute*
dispatches (matmuls + fused blocks + reductions + gather) to ~10–12; the
index-source layers (full/reindex) still carry the irreducible indexer `Sort`/
`CumSum`/top-k.

### 7.3 Arms, lever, tests

- Lever `MTPLX_DSV41_ATTN_CORE_COMPILE` (default OFF, rounding-class):
  `Attention._sparse_attend_selected` routes the fixed-shape core through the
  geometry-keyed compiled tape at decode/small-M verify; above the small-M cap the
  eager block runs (byte-for-byte control).
- Arms: `attn_core_compile` (isolation), `cell16k_ring_attn_core` (= cell16k_ring +
  core compile), `cell16k_ring_wo_a_core` (= cell16k_ring + `wo_a` cache + core
  compile — the full W97 attention-dispatch program). All three carry the core
  compile → **rounding-class**, so the byte-identity summary flags them (token-id
  sha will differ vs control on a greedy near-tie flip — [[dsv41-inexact-ok-if-tie-flips]]).
- Census: `nice -n 19 $PY scripts/deepseek_v41/dispatch_census.py --attn-core`
  (self-contained, CPU). Window-40 GPU A/B (through the flock):
  ```
  nice -n 19 $PY scripts/deepseek_v41/metal_decode_attn_bisect.py \
    --in-model --unfenced --attn-subops --gpu --model $MODEL --arms cell16k_ring_wo_a_core \
    --context-tokens 16384 --memory-limit-gib 60 --max-kv 17408 \
    --in-model-steps 64 --warmup-steps 8 --utilization --out $OUT/w40-wo_a_core.json
  ```
  and the paired AR receipt at arms `cell16k_ring` / `cell16k_ring_attn_core` /
  `cell16k_ring_wo_a_core` for tok/s + token-sha (label the rounding-class flip).
- Tests: `tests/test_deepseek_v41_attn_core_compile_w97.py` — core prims eager>compiled
  (2 matmuls survive, elementwise fuses); compile-cache bounded over 64 steps (no
  per-token retrace); greedy-argmax identical + labelled logit max\|Δ\| over 64
  decode steps on a tiny full model (rounding-class band).

## 8. W99 — the casts and concats: trace, classify, lean (byte-identical)

The §7 census shows the decode attention layer issues 15 (simple) / 25 (full) `AsType`
kernels and 6 / 10 `Concatenate` kernels. W99 traced each on the T=1 decode path (tiny
real-structure model, real dims, `WINDOW_RING` = the served backing) and classified them:
(a) redundant, (b) load-bearing reference f32, (c) load-bearing kernel/index dtype.

### 8.1 `AsType` (f32 casts) — trace + classification

| source (helper) | count/layer | dtype | class | reason |
|---|---:|---|---|---|
| `_rmsnorm` (q-norm, kv-norm; ×2 calls) | ~4 | bf16→f32→bf16 | **(b)** | reference `RMSNorm` normalises in f32 (model.py L288-293); folds into the K22 qkv `Compiled` node |
| `_apply_interleaved_rope` store (q, kv, o; ×3) | ~3 | f32→bf16 | **(b)** | reference `apply_rotary_emb` rotates in f32, stores at model dtype (L232-244); folds into the tapes |
| `_cos_sin` `positions.astype(f32)` | 1 | int→f32 | **(b)** | angle math is f32 (positions differ per token — not hoistable) |
| core `q.astype(f32)` | 1 | bf16→f32 | **(b)** | reference sparse-attn scores in f32 |
| core `KVg.astype(f32)` (QK **and** PV) | 2 | bf16→f32 | **(a)** | **same array cast twice — compute once** |
| core `attn_sink.astype(f32)` | 1 | f32→f32 reshape | **(a)** | per-layer constant re-cast every token — **cache per layer** |
| `_o_lora_down` `o`/`w`.astype(f32) | ~2 | →f32 | **(b)** | the port's o-LoRA einsum is f32 (folds into the K22 out tape) |
| `_window_selected_idx` `.astype(int32)` | 1 | →int32 | **(c)** | `mx.take` requires int index |
| (also, not a graph primitive) `_cos_sin` `mx.array(np.asarray(inv_freq))` | — | numpy→device | **(a)** | per-token host→device **upload** of a per-layer constant — **lift once, cache** |

### 8.2 `Concatenate` — trace + classification

All 6 (simple) come from **RoPE**, ×3 calls (q, kv_new, o), 2 each: `_apply_interleaved_rope`'s
`mx.stack([r0,r1])` (the complex-pair interleave) and `_rope_last`'s `concatenate([head, roped])`
(re-joining the non-rope head to the rotated tail). Full adds 2 (`_sparse_attend_selected`'s
window+compressed KVg/valid concat) + the indexer lanes. **All class (b)/structural**: the
interleave is the rope layout; the head-rejoin is byte-identically removable only by roping the
full head with an identity-padded (cos=1,sin=0) table, which costs 8× the rope elementwise and
2 padding-concats to build — a net loss. The window+compressed concat is 1 op vs 2 einsums or 2
slice-writes if split — also a loss.

### 8.3 What W99 removes (byte-identical, `MTPLX_DSV41_ATTN_LEAN_CASTS`, default OFF)

The three class-(a) items: the eager core's **double `KVg` f32 cast → one**, the **per-layer f32
sink cached**, and the **per-token `inv_freq` numpy→device relift → lifted once and cached**.
BYTE-IDENTICAL by construction (no fp math reordered) — tested bit-for-bit over 64 decode steps,
eager AND compiled, and composing with the wo_a cache.

### 8.4 Measured before/after (real dims, one decode `_attend`, K22 tapes on, WINDOW_RING)

| mode | config | graph prims | non-view kernels | AsType | Concat |
|---|---|---:|---:|---:|---:|
| swa_only/reuse | K22 (baseline) | 162 | 69 | 15 | 6 |
| swa_only/reuse | K22 + lean (byte-identical) | 160 | 67 | 13 | 6 |
| swa_only/reuse | K22 + lean + core-compile (rounding-class) | 151 | 63 | 14 | 6 |
| full | K22 (baseline) | 283 | 122 | 25 | 10 |
| full | K22 + lean | 282 | 121 | 24 | 10 |

### 8.5 Verdict — the targets are NOT reachable by cast/concat lean

The ≤40 (byte-identical) and ≤25 (with K29) **per-layer** targets are **not achievable** by
leaning casts/concats. The trace shows the honest reason: **~13 of the 15 casts are class-(b)
reference f32** (rmsnorm, softmax, cos/sin, RoPE store, o-LoRA) and the **6 concats are
structural RoPE** — none reducible byte-identically. Byte-identical lean removes only the 2
class-(a) casts (KVg dedupe) + the per-token inv_freq upload (a host-encode, not a graph
primitive), taking a simple layer 69 → 67 kernels. K29 collapses the ~13-kernel core to 1 (69 →
~57), still short of 25. **The whole-layer count is dominated by the qkv/out projection chains
(matmuls + f32 rmsnorm + RoPE interleave/rejoin) that none of these levers touch.** Reaching
≤25/layer would require FUSED custom Metal kernels for qkv-prep (proj+rmsnorm+rope) and out-prep
(rope+grouped-einsum+wo_b) — a single dispatch each — i.e. the K29 approach applied to the
projection chains, a separate kernel-authoring program (not byte-identical, GPU-only). Recommend
that as the next step; the exact-numerics ceiling here is ~55–67 kernels/layer.

### 8.6 Arms + tests

- Lever `MTPLX_DSV41_ATTN_LEAN_CASTS` (default OFF, **byte-identical**).
- Arms: `attn_lean_casts` (isolation), `cell16k_ring_lean` (= cell16k_ring + wo_a cache +
  lean casts — the byte-identical W97/W99 stack), `cell16k_ring_lean_k29` (+ K29, rounding-class),
  and `cell16k_ring_wo_a_k29` (= cell16k_ring + wo_a cache + K29, the 1-dispatch core arm).
- Tests: `tests/test_deepseek_v41_lean_casts_w99.py` — bit-for-bit identical logits + greedy ids
  over 64 decode steps (eager + compiled); composes byte-identically with the wo_a cache; the
  eager core issues fewer `AsType`; inv_freq/sink cached once. Arm assertions in
  `tests/test_deepseek_v41_ab_env_levers.py`.
