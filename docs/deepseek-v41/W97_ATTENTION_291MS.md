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
   `wo_a` down-projection re-issues `mx.dequantize(wo_a)` every layer every token.** At the
   released dims `wo_a` is `[8, 1024, 4096]` = 33.55M params; the port dequantizes it to a
   fresh **134 MB f32** (q8 gs64) / **67 MB bf16 → 134 MB f32** (native mxfp4) array **per
   token per layer** and runs the einsum in f32 — **304 MB/layer (q8) / 420 MB/layer
   (mxfp4)**, ~12–17 GB/token over 40 layers. The reference (model.py L784-787) dequantizes
   `wo_a` **once at convert time to bf16** and runs the einsum in bf16 (67 MB/layer,
   2.68 GB/token). At the gappy in-situ bandwidth this is ~1 ms/layer and, more importantly,
   40 extra dequant **dispatches**/token — the single largest attention compute/traffic
   item, and one the bf16 isolated bench (dense `wo_a`, no dequant) structurally cannot see.
   **Confirmed by the W96 as-is audit (finding L9).** Fixed here (§5), byte-identically,
   behind `MTPLX_DSV41_ATTN_WO_A_CACHE`.

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

| path | per token / layer | × 40 layers |
|---|---|---:|
| **port, q8 gs64** (default) | dequant read 35.7 MB + dequant write **134 MB f32** + einsum read **134 MB f32** = **304 MB** | **12.16 GB/token** |
| **port, mxfp4 gs32** (native bank) | dequant 84.9 MB + astype bf16→f32 201 MB + einsum read 134 MB = **420 MB** | **16.82 GB/token** |
| **reference** (bf16, dequantized once) | einsum read (bf16) **67 MB** (no per-token dequant / astype) | **2.68 GB/token** |

Verified with `mx.dequantize` on the real shapes: q8 gs64 → f32 134.22 MB output; mxfp4/mxfp8
gs32 → bf16 67.11 MB output (then `_o_lora_down`'s `.astype(f32)` promotes it to 134 MB).

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
   bench never issues the 40 dequant dispatches or the 134 MB f32 dequant write per layer.
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
array per layer when the lever is on, keyed on the packed-weight identity (a re-quantize /
reload rebuilds it), materialised once via `mx.eval` so later tokens reference the buffer
rather than a lazy dequantize node. **Byte-identical to control**: the cached array is exactly
what `mx.dequantize` returned; the reshape and `_o_lora_down`'s `.astype(f32)` are unchanged, so
no fp math is reordered (this holds for both q8 and native codecs). It removes the per-token
dequant dispatch (40/token) and its dequant-write traffic (~5.4 GB/token q8 / 2.7 GB/token
native). Default OFF and opt-in because it holds a dense copy of every layer's `wo_a` resident
(q8: 40 × 134 MB ≈ 5.4 GB; native: 40 × 67 MB ≈ 2.7 GB) — mind the box memory budget
([[never-exceed-the-memory-knob]], the 60 GB runner limit; window-38 already peaks 65 GB).

A lower-memory, reference-matching variant (dequantize once to **bf16** and run the einsum in
bf16, 2.7 GB and half the einsum-read traffic) is a rounding-class change vs the current f32
einsum — the reference numerics — and is left as a follow-up decision for Fable/David given
the memory/precision trade.

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
