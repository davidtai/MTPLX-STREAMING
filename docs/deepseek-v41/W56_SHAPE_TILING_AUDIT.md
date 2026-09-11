# W56 — DeepSeek-V4.1-Flash shape / tiling audit (KERNEL_LEDGER K27)

Worker: Opus 4.8, branch `feat/deepseek-v41-w56` (worktree `.worktrees/dsv41-w56`).
CPU-only static analysis + CPU byte-identity tests; **no GPU / Metal executed** (a
window may be running). David's ask: *"make sure you do right-shape and right-size
optimizations and tilings."*

This audits every hot matmul / gather_qmm / quantized_matmul / einsum / softmax /
reduction on the decode (M=1 and the M=K+1 verify) and prefill (rows=chunk, T up to
16,384) paths against the **actual mlx 0.32.2 Metal kernel dispatch** on this box's
GPU. It is the shape/tiling companion to K25 (score precision, W50) and K26 (dense
experts, W51): those price *precision* and *dequant*; this prices *which kernel the
operand shapes select and whether the tiles are aligned or ragged*.

**Headline:** most of the speculative fixes are **refuted** — the score/PV/o-LoRA
einsums already lower to byte-identical, fully tile-aligned batched GEMMs, and every
big dense/quantized weight dim (hidden 5120, inter 2304, head_dim 512, vocab 129280)
is a multiple of the 64-wide steel/gather tiles. The audit found **one** first-order
shape defect: the routed-expert switch calls `gather_qmm` with **unsorted** rows, so
mlx takes the per-row `gather_qmv` instead of the fused weight-streamed-once
`gather_qmm_rhs` — the mechanism behind the W47 "2.9 TFLOPS / 105 s" prefill switch.
Two second-order shape facts (down-proj ragged K, verify-batch dtype) are documented
with microbench arms.

---

## 1. The device and the MLX 0.32.2 dispatch facts (cited)

Box GPU: **`applegpu_g17s`** (Apple M5 Max — arch gen **17**, size char **`s`**), NAX
(neural-accelerator matmul) **available**. Confirmed by the arch string the repo's own
`tests/test_qsa_indexer_select_metal.py` pins (`applegpu_g17s`); re-confirm in-window
with `mx.metal.device_info()["architecture"]`. Source paths below are the v0.32.2 tag
of the mlx tree (`mlx-profiler/mlx`, `git show v0.32.2:mlx/backend/metal/…`).

### Dense matmul — `Matmul::eval_gpu` (`backend/metal/matmul.cpp`)
- `M==1` (decode) → **gemv**; `M==1 && N==1` → dot_product. `min(M,N)==1` → gemv (`:1507,1545`).
- `M in 2..≤5`, dtype∈{fp16,bf16}, weight transposed (`b_transposed`), `K%4==0`, vec4-aligned,
  `arch_gen≥15` → **`gemv_wide`**: streams the weight ONCE for up to 5 vectors (`:1299-1353, 1520`).
  **f32 is excluded** (`out.dtype()` must be fp16/bf16, `:1332`). g17 is gen 17 ≥ 15 → available.
- else `steel_matmul_axpby` (`:844`):
  - `use_nax = is_nax_available() && !complex && (enable_tf32() || dtype!=f32)` (`:917-919`) —
    **bf16/fp16 use NAX; f32 uses the classic steel path unless `MLX_ENABLE_TF32`**.
  - non-nax split-K (`:925`): `!use_nax && B==1 && ceil(M/16)*ceil(N/16) ≤ 2048 && K/16≥8 && K≥max(M,N)`
    (`min_tmn_threshold=2048` for size `s`/`d`, `:921`).
  - nax split-K (`:948`): `use_nax && B==1 && (K ≥ 3·max(M,N) || (max(M,N)≤1024 && K > 2·max(M,N)))`.
  - regular **NAX** tiles (`steel_matmul_regular_axpby_nax`, `:206-216`): for size `s`
    **`bm=64, bn=128, bk=256`** (`bk=64` when `K≥8192 && K>M+N`), `wm=2, wn=4`; the *aligned*
    (fast, bounds-check-free) kernel needs **`M%64==0 && N%128==0 && K%256==0`** (`:235-237`).
  - regular **non-nax** (f32) tiles, size `s` = "Medium device" branch → `bm=64,bn=64,bk=16` (`:165-171`).
- `check_transpose` (`:27-43`): an operand that is **neither row- nor col-contiguous**
  (last-dim stride ≠ 1 AND second-last stride ≠ 1) forces a `contiguous_copy_gpu`.
- Batch collapse (`:1493-1502, 878-907`): `[batch,M,K]` contiguous @ **broadcast** weight
  (`B_batch_stride==0`) collapses to one `[batch·M, K]` GEMM; a **per-batch** weight stays batched.

### Quantized — `QuantizedMatmul` / `GatherQMM::eval_gpu` (`backend/metal/quantized.cpp`)
- `ensure_row_contiguous_matrix` forces x/w/scales row-contiguous, else a copy (`:1790-1795, 65-82`).
- vector limit (transpose) `= get_qmv_batch_limit(K,N)` (`:1804,1898`): gen 17, size≠`d`
  → **13** when D,O>4096, **25** when ≤4096, **33** when ≤2048 (`:85-95`). `M ≥ limit` → GEMM
  (`qmm`/`gather_qmm`); `M < limit` → GEMV (`qmv`/`gather_qmv`, or `qmv_wide`/`gather_qmm_rhs`).
- `qmm` → **`qmm_nax`** if `nax && transpose && K%64==0 && bf16`: `bm=32(M≤32)/64, bn=64, bk=64`,
  aligned `N%64==0` (`:1039-1058, 834-842`).
- `gather_qmm` → **`gather_qmm_nax`** (same gate): `bm=bn=bk=64`, aligned `N%64==0` (`:1236-1255, 937-949`).
- **`gather_qmm_rhs`** (fused, streams each expert weight ONCE over its contiguous row block):
  fires ONLY when **`M==1 && B≥16 && right_sorted_ && B/E≥4`** (`:1905`). `right_sorted_ =
  sorted_indices && lhs_indices is None` (`ops.cpp:5632-5633`). → `gather_qmm_rhs_nax`
  `bm=(M/E<64)?32:64, bn=64, bk=64` (`:1495-1503`).
- `gather_qmv` (M<limit) `bn=8, bk=32`; **fast** kernel iff `N%8==0 && K%qmv_fast_k_alignment(bits)==0`
  (`:1329-1337`). `qmv_fast_k_alignment` = `pack_factor·(bits==2?1:2)·32` (`:147-148`) →
  **mxfp4 (bits 4): 512**, mxfp8 (bits 8): 256. `use_narrow_qmv` is **nvfp4-only** (`:487-488`), so
  mxfp4 experts / mxfp8 head never take it.

### Model dim alignment against those tiles
| dim | value | %64 | %128 | %256 | %512 |
|---|---|---|---|---|---|
| hidden | 5120 | 0 | 0 | 0 | 0 |
| inter | 2304 | 0 | 0 | 0 | **256** |
| head_dim | 512 | 0 | 0 | 0 | 0 |
| q_lora_rank | 1280 | 0 | 0 | 0 | **256** |
| o_lora_rank | 1024 | 0 | 0 | 0 | 0 |
| q up (H·hd) | 32768 | 0 | 0 | 0 | 0 |
| vocab | 129280 | 0 | 0 | 0 | **256** |
| prefill T / chunk | 1024·16384 | 0 | 0 | 0 | 0 |

Every GEMM **N/K** that feeds a steel/gather/qmm tile is a multiple of 64/128/256 → the
*aligned* kernel variants are selected. The only sub-alignment is **`%512`** for `inter`,
`q_lora_rank`, `vocab` — which matters **only** for the mxfp4 `gather_qmv`/`qmv` **fast**
K-alignment (512), not for any GEMM tile. See F2.

---

## 2. Audit table (hot ops, decode M=1 / verify M=4 / prefill rows=chunk, T≤16384)

`site` = `mtplx/models/…`. "kernel" = the mlx 0.32.2 kernel the shapes select on g17s.

| # | op | site | shapes (M×K→N, batch) | dtype | layout / tile verdict | kernel selected | est. cost | proposed fix |
|---|---|---|---|---|---|---|---|---|
| F1 | routed switch gather (gate/up/down) | `expert_mlx.py:1749` `_gather_component_bank` | x`[rows,1,1,5120]`, w`[384,2304,5120]`/`[384,5120,2304]`, `rhs_indices[rows,1]`; rows≈98 304 @16K | bf16 act / mxfp4 gs32 | **rows UNSORTED, `sorted_indices` unset → `right_sorted_`=False**; M=1<13 → per-row `gather_qmv`, no cross-row weight reuse | `gather_qmv` (thrash) | **~105 s prefill** (W47, 2.9 TFLOPS) | **sort rows by slot, `sorted_indices=True`, unsort → `gather_qmm_rhs_nax`** (weight streamed once/expert). `MTPLX_DSV41_LAYOUT_FIX` |
| F2 | expert **down**-proj (decode) | `expert_mlx.py:1770` | x`[6,1,1,2304]`, w`[384,5120,2304]` | bf16/mxfp4 | **K=inter=2304, 2304%512=256 → misses fast `gather_qmv`** (N=5120%8 ok); gate/up (K=5120%512=0) hit fast | `gather_qmv` (slow, ragged-K) | ⅓ of decode switch (~74 ms/tok) | pad K→2560 in bank (zeros exact) — **load-time, documented not implemented**; microbench arm `down_align` |
| F3 | attn score QK^T + PV | `deepseek_v41.py:719,750` `_sparse_attend_oneshot` | q`[1,s,64,512]`·KV`[1,T,512]`→`[1,s,64,T]`; M=s·64=65 536, K=512, N=T | f32 (bf16 arm) | **already a byte-identical, fully aligned batched GEMM** (M%64,N%64/128,K%256 all 0). Output-transient-bound (M·N=1.07e9 f32=4.3 GB), NOT FLOP-bound | f32→steel medium `64/64/16`; bf16→nax `64/128/256` | score 153 s @16K | **no einsum→matmul win** (proven byte-identical, §3). Real lever = the W50 `lean` pass-cut (done) / K6 |
| F4 | o-LoRA grouped down | `deepseek_v41.py:1025` `_o_lora_down` | `bsgd,grd->bsgr`; batch g=8, M=b·s, K=4096, N=1024 | f32 | einsum lowers to byte-identical batched GEMM (K,N %256 ok) | batched steel medium | small | **no win** (proven byte-identical, §3) |
| F5 | attn/head projections (verify M=4) | `deepseek_v41.py:944,996`; `_MXFP8Head:2542` | wq_b`[1280→32768]`, wo_b`[8192→5120]`, head`[5120→129280]`; M=4 | bf16 resident | **bf16 M=4 → `gemv_wide` (weight streamed ONCE for the 4 rows)**; f32 would re-stream 4× | `gemv_wide` (bf16) | verify weight reads | keep verify acts **bf16** (already, K21); microbench arm confirms gemv_wide |
| F6 | dense-quant prefill expert matmul (K26, default off) | `expert_mlx.py:1701` | x`[rows,5120]`@`[2304,5120]`ᵀ; rows 128–1024 | bf16/f32 | dense matmul tiles aligned (N 2304%128=0, K 5120%256=0) — **not a tile defect**; W51 shortfall is bf16-slow-vs-f32 (W50), a precision/kernel issue | nax `64/128/256` | K26 territory | orthogonal to K27; `dense` microbench confirms the W50 dtype gap |
| — | gate score GEMM | `deepseek_v41_moe.py:215` | xf`[n,5120]`@`[384,5120]`ᵀ→`[n,384]` | f32 | N=384%64=0; decode M=1→gemv; prefill M≥13→ f32 split-K (small-MN large-K) | gemv / splitk | small | none (aligned) |
| — | HC mix | `deepseek_v41.py:1395` | `[rows,20480]`@`[24,20480]`ᵀ→`[rows,24]` | f32 | **N=24 ragged** (%64≠0) but N tiny → negligible; dispatch-bound (K3/K4 own it) | gemv / ragged steel | ~0 | none (K3/K4) |
| — | Sinkhorn 4×4 | `deepseek_v41.py:163` | comb`[rows,4,4]`, 20 iters | f32 | tiny; a **dispatch-count** problem (~6.4k/tok), not tiles | elementwise | K3 lever | K3 Metal kernel |
| — | indexer select score | `deepseek_v41.py:539` | q`[1,s,32,128]`·k`[1,T,128]`→`[1,s,32,T]` | f32 | same aligned batched-GEMM shape as F3 (K=128%16=0) | batched steel | < F3 | none (aligned) |
| — | engram wkv gather+GEMV | `deepseek_v41.py:2984` (attach) | 264 B rows, top gather + small GEMV | mixed | latency op (~0 tok/s), K20 | gather+gemv | ~0 | K20 (out of scope) |

---

## 3. Refutations (proven on CPU, `mx.set_default_device(mx.cpu)`)

The audit's speculative "einsum→matmul / batch-the-heads / de-stride" fixes were tested
for byte-identity at the real layout — **all three are no-ops** because mlx's einsum
already lowers to the optimal batched GEMM:

| rewrite | `mx.array_equal` vs shipped einsum | verdict |
|---|---|---|
| score QK^T `bshd,btd->bsht` → `matmul(q.reshape(1,s·H,hd), KVᵀ)` | **True** (max\|Δ\|=0.0) | einsum already this GEMM; no win |
| PV `bsht,btd->bshd` → `matmul(w.reshape(1,s·H,T), KV)` | **True** (0.0) | ″ |
| o-LoRA `bsgd,grd->bsgr` → transpose-to-batch + `matmul` | **True** (0.0) | ″ |

So the score cost is **not** a copy/layout problem — it is the 4.3 GB `[65536,T]` output
transient (already addressed by the W50 `lean` f32 pass-cut and K6, and why bf16 *loses*:
it adds a cast pass over that transient with no FLOP relief at K=512). **Do not chase an
einsum rewrite.**

The **F1** sorted-gather rewrite IS byte-identical on CPU (a permutation + its inverse
over an M-independent per-row matmul): tested at 6 / 24 / 3072 / 8192 rows, mxfp4 and
affine, flag on == off, max\|Δ\|=0.0 (`tests/models/test_deepseek_v41_w56_layout_fix.py`).
On Metal it swaps `gather_qmv → gather_qmm_rhs_nax` — a kernel reassociation in the same
documented FP class as K26 (measured, not bit-identical, in a window).

---

## 4. Implemented — `MTPLX_DSV41_LAYOUT_FIX` (default OFF)

Single site: `expert_mlx.py:_gather_component_bank` (every switch caller — resident,
streamed, device-route, dense-prefill fall-through — funnels through it). When the flag
is set AND the wave has ≥ `MTPLX_DSV41_LAYOUT_FIX_MIN_ROWS` rows (default **2048**, so
decode 6 / verify 24 / small waves keep the **exact shipped unsorted call** and the
served M=1 path is untouched), it:
1. `perm = argsort(slot)` on device; `x = take(x, perm)`, `slot = take(slot, perm)`;
2. runs gate/up/down with `sorted_indices=True` (routes to `gather_qmm_rhs_nax` when
   `B/E≥4`, i.e. prefill; otherwise still correct via `gather_qmv`);
3. `output = take(output, argsort(perm))` — unsort to router order.

Tunables (read at use, [[env-flags-read-at-use-not-import]]):
- `MTPLX_DSV41_LAYOUT_FIX` = master (default off).
- `MTPLX_DSV41_LAYOUT_FIX_MIN_ROWS` (default 2048) — the sort threshold.
- `MTPLX_DSV41_GATHER_ROWS_PER_CALL` (default 0 = one call) — chunk the sorted wave
  (memory bound + microbench sweep; each chunk is a contiguous sorted sub-range so
  `sorted_indices` stays valid and per-row math is unchanged).
- Pre-existing and still exposed: `MTPLX_DSV41_PREFILL_CHUNK` (512/1024/2048),
  `MTPLX_DSV41_PREFILL_SCORE_KEY_CHUNK`, `MTPLX_DSV41_PREFILL_DENSE_{MIN_ROWS,BATCH,MATMUL_DTYPE}`.

CPU tests (`tests/models/test_deepseek_v41_w56_layout_fix.py`, 15 cases, **all pass**,
pinned to CPU): flag on/off byte-identical for decode / verify-batch / chunk-major /
layer-major waves (mxfp4 + affine); default-threshold leaves small waves untouched;
rows-per-call 512/1000/4096 byte-identical.

**F2 (pad down-proj K→2560) is NOT implemented**: it repacks the mxfp4 bank at load time
(outside the mechanical-reorder allowlist and the streamed-bank build). It is documented
and armed as a microbench case (`down_align`) so a window prices it before any bank change.

---

## 5. Microbench — `scripts/deepseek_v41/shape_tiling_microbench.py`

CPU-safe `--help` / `--dry-run` (print the case/shape/variant plan; **no Metal touch** —
mlx is imported lazily and no array op runs). A real run executes on the default device
(Metal, in-window only) with `--memory-limit-gib` (default 8) capping the working set;
warmup/iters median, TFLOPS per variant, JSON receipt. Cases at the real shapes:
`score` (einsum vs matmul, f32 vs bf16), `score_keychunk` (0/512/1024/2048),
`olora` (einsum vs bmm), `gather` (unsorted vs sorted, rows 6/12/24/48/96/chunk·6),
`down_align` (ragged 2304 vs padded 2560), `dense` (bf16 vs f32, rows 128/256/1024).

### Exact window command (run in-window ONLY; flock held, qwen unloaded)
```
scripts/deepseek_v41/gpu_window.sh \
  env PYTHONPATH="$PWD" \
    /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3 \
    scripts/deepseek_v41/shape_tiling_microbench.py \
    --seq-t 16384 --score-chunk 1024 --warmup 3 --iters 15 --memory-limit-gib 8 \
    --out docs/deepseek-v41/receipts/gpu-windows/window-22/shape_tiling_microbench.json
```
(`--score-chunk 512` / `2048` reruns the score/gather chunk sweep; `--only gather` isolates F1.)

The K27 A/B on the served model (the number that decides F1) uses the standard 16K
layer-major shape ([[dsv41-standard-benchmark-shape]]), paired in one window:
```
# control
scripts/deepseek_v41/gpu_window.sh env PYTHONPATH="$PWD" MTPLX_DSV41_PREFILL_LAYER_MAJOR=1 \
  MTPLX_DSV41_PREFILL_CHUNK=1024 .venv/bin/python3 scripts/deepseek_v41/bench_standard_shape.py …
# +layout fix
scripts/deepseek_v41/gpu_window.sh env PYTHONPATH="$PWD" MTPLX_DSV41_PREFILL_LAYER_MAJOR=1 \
  MTPLX_DSV41_PREFILL_CHUNK=1024 MTPLX_DSV41_LAYOUT_FIX=1 .venv/bin/python3 scripts/deepseek_v41/bench_standard_shape.py …
```
Pass gate: 16K prefill TTFT down materially, decode byte-identical (flag never engages at
M=1), peak under the 100 GiB knob; token-sha divergence on prefill is the documented
`gather_qmm_rhs` FP class (report it).

---

## 6. Top findings (ranked by estimated seconds / ms saved)

1. **F1 — sorted routed gather (prefill switch).** Unsorted `gather_qmm` → per-row
   `gather_qmv` is the W47 105 s / 2.9 TFLOPS switch. `sorted_indices=True` routes to the
   fused `gather_qmm_rhs_nax` (each expert weight streamed once over its ~256-row block).
   Implemented behind `MTPLX_DSV41_LAYOUT_FIX`; CPU byte-identical; **window measures the
   seconds.** Estimated switch **105 s → ~25–45 s** (bank read co-binds), the biggest K27 lever.
2. **F2 — down-proj ragged K=2304.** Misses the mxfp4 fast `gather_qmv` (needs K%512==0),
   so ⅓ of the ~74 ms/tok decode switch runs the slow kernel. Fix = pad K→2560 (zeros
   exact) at load; documented + `down_align` microbench arm to price it first.
3. **F5 — verify M=4 wants bf16, not f32.** bf16 M=4 dense projections hit `gemv_wide`
   (weight streamed once for all 4 rows); f32 re-streams 4×. Keep verify acts bf16 (K21).
   `gather` rows 6–96 microbench quantifies the verify-batch regime.
4. **F3 refuted — score einsum is not a copy/tile defect.** Already a byte-identical,
   fully-aligned batched GEMM; it is output-transient-bound, which is why bf16 loses
   (W50) and the f32 `lean` pass-cut is the real lever. Saves chasing a dead rewrite.
5. **F6 — dense-prefill (K26) is aligned, not a tiling loss.** Its N/K hit the nax tiles;
   the W51 −20 s (vs modelled −70) is the bf16-slow-vs-f32 kernel (W50), not raggedness —
   the `dense` microbench (bf16 vs f32, rows 128/256/1024) isolates it, orthogonal to K27.

---

## 7. Verification
- `tests/models/test_deepseek_v41_w56_layout_fix.py` — 15 CPU cases, all pass (byte-identity).
- `tests/models/test_deepseek_v41_prefill_dense_experts.py` — still passes (no regression).
- `shape_tiling_microbench.py --help` / `--dry-run` — CPU-safe, no Metal.
- Byte-identity of the F3/F4 refutations and the F1 sorted gather verified on CPU
  (`mx.array_equal`, max\|Δ\|=0.0).
