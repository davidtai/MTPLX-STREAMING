# W58 / K28 — Fused mask + attention-sink softmax Metal kernel

Status: IMPLEMENTED + CPU-proven (algorithm + plumbing + dispatch), default OFF.
GPU numeric parity is gated behind `MTPLX_GPU_PARITY=1` for a window run.
Worker `feat/deepseek-v41-w58` (off `feat/deepseek-v41-streaming` tip 53aa725c2).
CPU-only static/analysis work; **no GPU/Metal executed** (a GPU window holds the
exclusive lock; `mx.fast.metal_kernel` builds *and* runs on the GPU even under the
CPU default device, so every test here pins `mx.cpu` and spies the kernel).

Env: `MTPLX_DSV41_PREFILL_SOFTMAX_KERNEL=1` (default off).
Kernel: `mtplx/kernels/dsv41_fused_softmax.py`.
Integration: `mtplx/models/deepseek_v41.py::Attention._sparse_attend_oneshot`.
Tests: `tests/models/test_deepseek_v41_fused_softmax.py` (19 CPU + 1 GPU-gated).
Arms: `softmax_kernel`, `prefill_lean_k28` in `scripts/deepseek_v41/ab_decode_env_levers.py`.

---

## 0. What K28 is

At 16,384 tokens the DSV4.1 CSA2 attention score path materialises a
`[rows=1024, 64 heads, T]` f32 transient (up to **~6 GiB** at T≈24,576). The
shipped eager softmax walks that transient repeatedly:

- **one-shot (default):** `scores*scale` → `mx.where(attend, ·, -inf)` →
  `concatenate([scores, sink])` → `mx.softmax(full)` → slice `[..., :T]`.
- **lean (W50 `score_lean`, `fuse_scale`+`fold_sink`):** `mx.where(attend, ·, -inf)`
  → `max` → `exp` → `sum` (+ sink term), normalise after PV.

Window-22 (`receipts/gpu-windows/window-22/prefill-16384-round2.json`) put the
30 reuse-layer attention cost at **184 s** across 17 chunks, of which
**`attn.reuse.score.softmax = 84.5 s`** (one-shot) and `scale_mask_sink = 14.7 s`
were the mask+softmax terms (qk 49.5 s, pv 17.8 s, out_proj 17.4 s). The W50 lean
path drops the softmax stage to **~53 s** (`prefill_lean` TTFT 265 s).

K28 collapses **mask + per-head value-0 sink + f32 softmax** into **one
`mx.fast.metal_kernel` dispatch** that reads the raw scores twice and writes the
normalised probabilities once, with **zero T-wide intermediates** (no
`masked_scores`, no `ex`, no concatenated-sink allocation, no slice copy). One
threadgroup per `(row, head)`; `TG=256` lanes scan the length-`T` score vector
strided; a threadgroup tree reduction produces the online-softmax `(max, denom)`
state; a second pass writes `p = exp(s - m) / denom` (masked keys → 0).

- **Threads per row(-head):** `TG = 256` lanes/group; grid `= 256 · rows · H`
  threads, one group per `(row, head)` (`threadgroup_position_in_grid.x`).
- **Reduction scheme:** each lane runs an **online (max, denom) accumulation**
  over its strided keys (`nm=max(lm,s); ld = ld·exp(lm-nm)+exp(s-nm)`), seeded
  with a finite sentinel `NEG=-3e38f` (so an all-masked lane's `(NEG, 0)` state is
  a true identity and never forms `-inf−(−inf)`). The `(m, d)` states combine up a
  power-of-two **threadgroup tree** with the log-sum-exp merge
  `M=max(m1,m2); D=d1·exp(m1-M)+d2·exp(m2-M)`. The per-head value-0 sink folds in
  once at the group level (`M=max(M, sink[h]); D=D·exp(M-M')+exp(sink[h]-M')`),
  matching the model's `m = max(max(scores), sink)`.

## 1. Semantics preserved (exactly, to reassociation)

Read against `scripts/deepseek_v41/torchref/ref_forward.py::_k_sparse_attn` and the
model `_sparse_attend_oneshot` `fold_sink` path:

- `scores = q·kᵀ·scale` — scale folded into q on the lean path (kernel passed
  `scale=1.0`); otherwise the kernel applies `softmax_scale` in-register (fusing
  the scale pass too).
- additive/boolean mask — the model passes a **boolean** CSA/causal `attend`
  (True = keep); the wrapper converts it to the additive `{0, -inf}` form the
  kernel reads (`mv >= FINITE_MIN` keeps the key, `score += mv` supports a true
  additive bias). A **no-mask fast path** (all keys valid) is a separate compiled
  variant.
- per-head **value-0 attention sink** — a virtual extra key with score
  `attn_sink[head]` and value 0: it adds `exp(attn_sink[head] - m)` to the
  denominator and contributes nothing to the numerator. A fully-masked row
  collapses to the sink alone (denom 1, all p = 0 → zero attention output),
  **NaN-free** (reference "all-invalid rows → zero output").
- f32 throughout; `metal::exp` (precise, matching `mx.exp` — not `fast::exp`).

**Exactness class: reassociation-level, NOT byte-identical** — the threadgroup
tree reorders the max/denom sums, and the kernel writes the *normalised* `p` so
the model's PV becomes `o = p·V` (division folded per-element) instead of lean's
`o = (ex·V)/denom` (division after the sum). Same float class as W50
`score_chunked` / `score_lean`. Expected **max|Δ| ≤ 1e-6, greedy-argmax
identical**.

Validated on CPU (a float32 numpy simulation of the exact kernel algorithm —
online per-lane accumulation + tree combine + sink fold + normalise — vs the eager
`mx`-style reference): **max|Δ| = 1.86e-9, argmax parity exact**, fully-masked
rows finite and all-zero, no-mask path matches. The GPU parity test then confirms
the *Metal execution* matches at the two documented shapes.

## 2. Byte / pass accounting (per chunk-layer, 16K layer-major, rows=1024)

Let `S` = the `[rows, H, T]` f32 transient (up to ~6 GiB at the largest reuse
chunk-layer). T-wide memory operations on the mask+softmax stages K28 replaces:

| Path | mask+softmax T-wide traffic | T-wide intermediates allocated |
|---|---|---|
| eager **one-shot** (scale·, where, concat, softmax, slice) | ~**10 S** (5R+5W) | masked_scores, concat(T+1), softmax(T+1), slice → up to ~4 S peak |
| eager **lean** (where; max, exp, sum) | ~**6 S** (4R+2W) | masked_scores (S) + ex (S) → up to ~2 S peak |
| **K28** (2 reads + 1 write) | **3 S** (2R+1W) | **none** beyond the output `p` (S), which PV needs anyway |

- vs lean: **6 S → 3 S = 50 % less** mask+softmax memory traffic, and the two
  large intermediates (`masked_scores`, `ex`) are gone — a peak-GB relief of up to
  ~2 S (≈ up to ~12 GiB at the 6 GiB chunk-layer) that eases the 100 GiB knob.
- vs one-shot: **10 S → 3 S = 70 % less**, and the concat/softmax/slice trio's
  `(T+1)`-wide allocations disappear.
- Mask reads are the `[rows, T]` additive mask (1/H = 1/64 of S) — negligible.

**Cost side (honest):** the kernel evaluates **~3 `exp`/element** (online rescale
in pass 1 does two, pass 2 recomputes one) vs eager's **1 `exp`/element** (it
stores `ex`). This is the memory-vs-ALU trade inherent to fusing (fewer bytes, no
stored `ex`, more transcendental ops). **The net win requires the mask+softmax
stage to be memory/pass-bound** — which is exactly what window-20 measured for the
score path (bf16 −34 %, split-K −16 %: it is *not* FLOP-bound). If a window shows
`exp`-ALU binding instead, the documented fallback is the **3-pass variant**
(separate max reduction, then sum, then write: 2 `exp`/element, 4 S memory) — a
one-flag change to `_build_source` (drop the online rescale in pass 1). K28 as
shipped chooses the memory-minimal online form because the premise is
bandwidth-bound.

## 3. Expected seconds saved at 16K (ESTIMATE — GPU-window-gated)

Assuming the stage is memory/pass-bound (window-20), time ∝ T-wide passes on the
mask+softmax stages:

- **`prefill_lean_k28` vs `prefill_lean`:** lean mask+softmax ≈ 53 s (softmax) plus
  the `where` mask stage; 6 S → 3 S ⇒ **est. −25 to −31 s** off the ~53–63 s, i.e.
  prefill_lean TTFT **265 s → ~235–240 s (≈ −10 to −11 %)**.
- **`softmax_kernel` vs `control` (one-shot):** eager mask+softmax ≈ 99 s
  (84.5 + 14.7); 10 S → 3 S ⇒ up to **~−50 to −65 s** if fully memory-bound —
  capped by the 3×-`exp` cost, so treat the lower half of that band as the
  realistic expectation until measured.

These are estimates in the **same FP/roofline class the K25 note warns about**
(window-20 overturned the K25 FLOP roofline). They are **not** a claimed win: the
gate is a paired in-window A/B.

## 4. GPU gate (Pass condition)

Run inside a GPU window (qwen unloaded, memory-guarded, through the flock):

- **Parity (this worker's test):**
  `MTPLX_GPU_PARITY=1 MTPLX_PARITY_RECEIPT=<path>.json \`
  `PYTHONPATH=$PWD nice -n 19 .venv/bin/python3 -m pytest \`
  `tests/models/test_deepseek_v41_fused_softmax.py::test_fused_softmax_parity_gpu -s`
  — compares the kernel to the eager f32 softmax-with-sink on random
  `[64,64,4096]` and `[8,64,16384]` scores (+ a `scale=1.0` case), reporting
  **max|Δ| + argmax mismatch** and writing the receipt. **Pass if** max|Δ| ≤ 1e-6
  and argmax mismatch 0 on every arm.
- **Throughput (integration A/B):** `softmax_kernel` vs `control` and
  `prefill_lean_k28` vs `prefill_lean` at the 16K standard shape, layer-major,
  paired in-window. **Pass if** TTFT strictly down AND greedy tokens identical to
  the eager arm (reassociation-level, so byte-identity is not required; the
  standard-shape byte-identity summary will flag it like `score_lean`).

## 5. Integration, flags, arms

- **Flag:** `MTPLX_DSV41_PREFILL_SOFTMAX_KERNEL` → `_resolve_prefill_softmax_kernel`
  (default off, fail-fast on a bad value, read-at-use). `_prefill_softmax_kernel_use`
  gates on flag AND `mx.metal.is_available()` AND `mx.default_device() == mx.gpu`,
  so a **CPU-pinned host (or no-Metal box) falls back to the eager path,
  byte-identical** to the selected score path — proven at the method level
  (`array_equal`) and end-to-end through a tiny real prefill.
- **Scope:** prefill only (`q.shape[1] > 1`; decode/M=1 always eager,
  byte-identical) and the **one-shot** path only. It composes with the lean path
  (scale folded into q → kernel `scale=1.0`) and the plain one-shot (kernel applies
  `softmax_scale`). The **chunked/split-K path is NOT routed through the kernel**
  (see §6).
- **Arms** (all lever keys pinned; dry-run test extended):
  - `softmax_kernel` — one-shot f32 + K28, to isolate the fused-softmax delta vs
    control.
  - `prefill_lean_k28` — the W50 `prefill_lean` stack (layer-major + dense experts
    K26 + lean score path K25) + K28.

## 6. Split-K hook (designed, scoped out of the model)

The kernel exposes `normalize=False, return_stats=True`: it then writes the
**un-normalised** `ex = exp(s - m)` and emits per-`(row,head)` `(m, denom)` as two
extra outputs, so a future split-K driver can merge chunk states online (the W50
`score_chunked` composition). The **model integration deliberately uses only the
normalised one-shot path**; wiring the chunked path through K28 (and merging the
stats across key chunks) is left to a follow-up and is not exercised here beyond
the CPU output-plumbing spy. This satisfies the task's "expose optional max/denom
outputs **or** scope to one-shot and say so" — we do both: the hook exists, the
model scope is one-shot.

## 7. Test evidence (CPU, no Metal dispatched)

`tests/models/test_deepseek_v41_fused_softmax.py` — **19 passed, 1 skipped**
(the GPU parity test; auto-skips without `MTPLX_GPU_PARITY=1`), peak RSS < 0.2 GB:

- flag resolver (default off / truthy / read-at-use / fail-fast) and the GPU gate;
- model dispatch: prefill (`s>1`) routes through the kernel wrapper on a spied GPU;
  the lean path passes `scale=1.0`; **decode (`s==1`) never calls the kernel**;
  flag-on-CPU one-shot AND lean are **byte-identical** to control (`array_equal`);
- wrapper plumbing via a spy kernel (no real Metal build): grid `= TG·rows·H`,
  threadgroup `(TG,1,1)`, output shapes `[(rows,H,T)]` f32, the `[rows,T]` additive
  mask (bool → `{0,-inf}`) and `[H]` sink inputs, the no-mask / no-sink / additive
  / return-stats / 3D-input variants;
- the source builder assembles for every structural variant (no shadowed `pv`
  redeclaration, `TG` constexpr present).

Regression sweep (CPU): `test_deepseek_v41_ab_env_levers` (+ the 2 new arms),
`test_deepseek_v41_prefill_score_precision`, `chunked_prefill`, `stage_timing`,
`parity`, `layer_major_prefill`, `no_mlx_imports`, `bench_scripts` — all green.

## 8. Verdict

K28 is a **memory-traffic lever** for the 16K prefill mask+softmax stage: it cuts
the T-wide passes on that stage roughly in half (lean) to 70 % (one-shot) and
removes the two large intermediates, at the cost of ~3× the `exp` work. It is
correct to reassociation level (CPU-proven) and safely gated (default off,
GPU-only, prefill+one-shot). **The throughput win is a GPU-window claim, not a
measured one** — the paired A/Bs in §4 decide it, and the 3-pass fallback is ready
if the window shows `exp`-ALU binding rather than the assumed bandwidth bound.
