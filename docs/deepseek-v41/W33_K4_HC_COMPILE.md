# W33 — HC-tape collapse (kernel-ledger K4): compile the Hyper-Connection chains

Branch `feat/deepseek-v41-w33` off `feat/deepseek-v41-streaming` @ `f9ac84875`.
Implementation commit **`3b89fb61f955c6dcb8c2c9e7a9d57523cba33b97`**.
Files: `mtplx/models/deepseek_v41.py`,
`tests/models/test_deepseek_v41_hc_compile.py` (new), this report,
`docs/deepseek-v41/KERNEL_LEDGER.md` (§K4 status line). CPU-only, tiny synthetic
config, no artifact loaded. Peak RSS of the model-suite run incl. this file:
**0.57 GB** (`/usr/bin/time -l` maximum resident set size 573,243,392 B).

## Verdict

DSV4.1 wraps attention and the MoE each in a Hyper-Connection pre/post: a
`flatten` + rsqrt-norm + small `fn` matmul + `hc_split_sinkhorn` (row-softmax +
20 alternating row/column normalises over a `[..., hc, hc]` matrix — 16 floats at
decode) + the `pre_mix` collapse + RMSNorm on the way in, and the `post` re-mix
on the way out — **~two dozen tiny primitives, twice per layer, ×40 layers**.
Uncompiled that is the **top per-token dispatch source** (KERNEL_LEDGER §2.1/§4:
~6.4k HC-mix dispatches, cf. V4's 6,794). K4 replays those chains from an
`mx.compile` tape instead of rebuilding the graph from Python every call, behind
`MTPLX_DSV41_HC_COMPILE` (**default OFF** — the decode/dispatch win is the KG-f
GPU-window measurement; V4 measured the stack at **AR +31.3 %, −26.1 %
dispatches**).

**Implemented and proven on CPU:** flag on vs off is `mx.array_equal` (f32 CPU)
over decode (n=1), a K+1 verify batch, and chunked + layer-major prefill; the
tape is inert above the row-cap; cache state is identical on/off; the dispatch
collapse is asserted (eager rebuilds the HC graph 2×/layer/token, a warm compiled
tape rebuilds **zero**), one compiled-callable invocation per layer per token.
The Sinkhorn stays an **opaque function boundary** — the tapes call the module
`hc_split_sinkhorn`, so the K3 worker's (W32) Metal kernel drops in at the tail of
the tape with **no change to this file**. Attention and the MoE streamed switch
stay **outside** the tapes, so the tapes are pure (no cache mutation).

## 1. What changed (structure, not math)

The eager per-call graph stays the default; the compiled path is opt-in and
byte-identical to the eager body with the flag off.

* **New module-level pure functions** (`_hc_mixes_split`, `_hc_pre_collapse`,
  `_hc_attn_prep_impl`, `_hc_ffn_prep_impl`) mirror `DecoderLayer._mixes` /
  `_hc_pre` / the two RMSNorms / `_hc_post_impl`. They take the layer's HC
  weights (`hc_*_fn/base/scale`, the norm weights) as **tape inputs**, so
  `mx.compile` caches **one** tape per `(kind, consts)` shared across all
  `2 * n_layers` Hyper-Connections — they share every shape and differ only in
  weight values (V4's design). `mx.flatten(x, -2, -1)` replaces `_mixes`'s
  `reshape(*x.shape[:-2], hc*dim)`: identical contiguous merge / identical
  values, but it reads no dynamic `.shape` (a reshape-from-shape bakes the first
  trace's dims under a shapeless trace).
* **`_hc_compiled(kind, *consts)`** builds and caches `mx.compile(impl)` once per
  key (the structural constants `hc, iters, norm_eps, hc_eps` are closed over,
  invisible to MLX's own shape/dtype cache otherwise). Fixed-shape (not
  shapeless — see §4).
* **`DecoderLayer.attn_and_moe_input` / `moe_combine`** gain a flag-gated
  compiled branch (`_hc_use_compile`), with the eager body kept **verbatim**. So
  `DecoderLayer.__call__` — and the W30 layer-major / W20 chunk-major drivers
  that compose these two methods — are unchanged with the flag off. The attention
  call (which writes this layer's KV) runs **between** the two compiled tapes; the
  MoE call runs between `attn_and_moe_input` and `moe_combine` — never captured.

## 2. Per-layer decode dispatch inventory (analytical, M=1)

Each `mx` call = 1 dispatch; `reshape`/`flatten`/`slice`/`transpose` = metadata
(0). The non-MoE, non-attention chain, per layer per decode token:

| Segment (per layer) | ops | which reassociate under compile |
|---|---:|---|
| attn HC mix — rsqrt-norm (`astype·square·mean·+eps·rsqrt`) | 5 | `mean` (large axis) |
| attn HC mix — `fn` matmul (`fn.astype·matmul·*rsqrt`) | 3 | `matmul` |
| attn HC mix — affine split pre/post/comb (`*scale·+base·sigmoid`, ×3) | 10 | none (elementwise) |
| attn HC mix — **Sinkhorn** (softmax + 39×(sum·+eps·divide)) | **119** | none (small hc-axis reductions are bit-exact) |
| attn `_hc_pre` collapse (`astype·*·sum(axis=2)·astype`) | 4 | none |
| attn RMSNorm (`astype·square·mean·+eps·rsqrt·*·wcast·*·astype`) | 9 | `mean` (large axis) |
| attn `_hc_post` (`astype·astype·*·einsum·+·astype`) | 6 | none |
| ffn HC mix (rsqrt-norm + matmul + affine + Sinkhorn) | 137 | `mean`, `matmul` |
| ffn `_hc_pre` collapse | 4 | none |
| ffn RMSNorm | 9 | `mean` |
| ffn `_hc_post` (via `moe_combine`) | 6 | none |
| **Total HC per layer** | **~312** | |

× 40 layers ≈ **~12.5k** HC ops/token, of which the Sinkhorn alone is
`119 × 2 × 40 ≈ 9.5k` primitives — matching KERNEL_LEDGER §4's "~6.4k dispatch
for HC-mix normalization" once the ~39 eps-adds per Sinkhorn are counted as folds
(`39 reduce_sum + 39 divide + 1 softmax ≈ 79`, cf. V4's ~80/mix and 6,794 total).

### After K4 (compile), per layer

`mx.compile` **fuses the elementwise triples** (`divide(add(sum, eps))` → one
fused kernel each, the sigmoids/muls/adds/collapse/post into single kernels) and
**replays a prebuilt tape** — the per-token Python **graph construction drops to
zero** on a warm decode step (measured, §5, test 6). What `mx.compile` does **not**
fuse: reductions and the matmul. So the residual per-layer floor is the
Sinkhorn's `39 reduce_sum + 39 fused-divide + softmax` (≈79/mix), the two
`mean` reductions, the two `_hc_pre` sums, the two `_hc_post` einsums and the two
matmuls.

| | Sinkhorn / mix | HC ops / layer | who removes the rest |
|---|---:|---:|---|
| eager | ~119 | ~312 | — |
| **K4 (compile)** | **~79** | **~180** | elementwise fused; per-token graph rebuild → 0 (warm) |
| + K3 (W32 Sinkhorn kernel) | **1** | ~60 | one Metal dispatch collapses the 20-iter recurrence |

So **~312 → ~180 HC dispatches/layer** from K4's elementwise fusion (−~42 % of
the HC portion), plus the elimination of per-token host-side graph construction.
Blended over the whole decode step (~6.8k dispatches, Sinkhorn-dominated) that is
the ballpark of V4's **measured −26.1 %**; the realized GPU delta on DSV4.1 is
KG-f. K3 (W32) then collapses the Sinkhorn to 1 dispatch each — the tapes call
`hc_split_sinkhorn` unchanged, so K3's kernel lands at the tail of the K4 tape.

## 3. Exactness — where `mx.compile` is bit-exact to eager

The strict gate is `mx.array_equal` (f32 CPU), stricter than V4's `1e-6` parity
bar. Measured op-by-op under `mx.compile` (W33, dims 16→5120):

| op | bit-exact vs eager |
|---|---|
| `hc_split_sinkhorn` (softmax + normalises over hc=4) | **always** (all rows/dims) |
| affine split (sigmoids), `_hc_pre` sum (hc), `_hc_post` einsum (hc) | **always** |
| RMSNorm / HC-mix `mean` (reduce over `dim` / `hc*dim`) | **rows ≤ 4** at any dim; reassociates ≥ ~64 |
| `fn` matmul `flat @ fn.T` | rows ≤ 4 at small dim; reassociates ≥ 8 (and at M=1 for large dim) |

On the tiny test config (hc·dim = 128) the **full model** is `array_equal` on/off
for **s ≤ 7** and diverges ~5e-7 at s ≥ 8. Decode (1) and a K=3 verify batch (4)
sit well inside the bit-exact regime; the row-cap keeps the compiled path there.

## 4. What V4 fused that DSV4.1 does NOT carry, and why

1. **`shapeless=True` — rejected (two measured reasons).** (a) `hc_split_sinkhorn`
   (owned by the K3 worker, W32 — not to be edited here) contains
   `comb.reshape(*comb.shape[:-1], hc, hc)`; under a shapeless trace MLX raises
   `[Primitive::output_shapes] Slice cannot infer output shapes`, so a tape
   containing it cannot be compiled shapeless at all. (b) Even where a tape traces
   shapeless, the batched matmul reassociates ~1e-6 at **batch > 1**, and
   `mx.compile` is `array_equal` with eager only in the **small-row regime**.
   So this carries V4's **fixed-shape + row-cap** design (V4 uses `shapeless`
   nowhere — it uses `_HC_COMPILE_MAX_ROWS`): the compiled path fires only for
   `rows ≤ _HC_COMPILE_MAX_ROWS` (decode/verify), where it is bit-exact and where
   the per-primitive host encode dominates; prefill chunks fall to the eager body
   (byte-identical either way). The KERNEL_LEDGER K4 note's "shapeless where the
   row count varies" is not reachable through W32's un-modifiable Sinkhorn.
2. **The head Hyper-Connection.** V4 compiles a third `"head"` tape
   (`_hc_head_impl`: `sigmoid(mixes*scale + base)+eps` collapse with its own
   `fn/base/scale`, `ParallelHead.hc_head`). **DSV4.1 has no head HC** — its final
   collapse is a plain `sum(pre_mix[...,None] * h, axis=2)` + RMSNorm over the
   *threaded* `pre_mix` (backbone `_forward_span`), 2–3 ops with no `fn`, no
   sigmoid, no Sinkhorn. Nothing to compile; not carried because it does not
   exist here.
3. **The "fused CSA attention" half of V4's K4 headline.** Kept **out** of the
   tapes: the attention call mutates the KV cache (a compiled tape must be pure),
   `head_dim` 512 is not an MLX fused-SDPA dim so attention materialises the full
   score (that is **K6 / gemma4's two-pass split-K**, a separate prefill lever),
   and a hand MLA kernel is explicitly **Dead-here** in the ledger (V4:
   neutral-to-negative, tipped near-ties). DSV4.1 keeps the default fused SDPA;
   the compile lever is HC-only.
4. **The matmul inside the tape is not bit-exact at large rows.** V4 gates its
   compiled `pre` at `1e-6`; the row-cap is what lets DSV4.1 meet the stricter
   `array_equal` gate (the cap confines the tape to rows where the matmul and the
   `mean` reductions match eager exactly).

## 5. Tests (`tests/models/test_deepseek_v41_hc_compile.py`, CPU, tiny, no artifact)

8 tests, all `mx.set_default_device(mx.cpu)`, `_HC_COMPILE_MAX_ROWS = 7` (the
config's bit-exact ceiling):

1. `test_decode_flag_on_off_identical` — 4 decode steps (n=1), flag on vs off,
   `array_equal` at each step (prefill one-shot > cap → eager both → identical
   cache).
2. `test_verify_batch_flag_on_off_identical` — a K+1 = 4-row verify forward, flag
   on (compiled) vs off, `array_equal`.
3. `test_prefill_chunked_flag_on_off_identical` — chunk 5 (≤ cap → compiled
   per span on flag-on), flag on vs off, `array_equal`.
4. `test_prefill_layer_major_flag_on_off_identical` — same under W30 layer-major.
5. `test_compile_inert_above_row_cap` — one-shot s=12 > cap, flag on == flag off
   (the tape is bypassed above the cap).
6. `test_cache_state_identical_flag_on_off` — window / compress_kv / index_k and
   offset identical on/off after prefill + 3 decode steps (the tapes mutate no
   cache).
7. `test_dispatch_collapse_and_once_per_layer` — **eager** rebuilds the HC graph
   `2 × L × n_tokens` times (`hc_split_sinkhorn`); the **compiled** path builds it
   **once** for the whole model (shared tape) and a warm decode token rebuilds
   **zero**; each compiled callable (`attn_prep`/`ffn_prep`/`moe_combine`) is
   invoked exactly `L` times per token.
8. `test_hc_use_compile_gating` — flag off → never; rows > cap → eager; rows ≤
   cap & flag on → compiled.

**Verification (CPU, `nice -n 19`, no `-n auto`):**

- `test_deepseek_v41_hc_compile.py` — **8 passed**.
- Full `tests/models/test_deepseek_v41_*.py` + `tests/test_deepseek_v41_*.py` —
  **201 passed, 39 skipped** (artifact-gated), no failures. The K4 flag is
  default OFF and the eager body is byte-identical, so every existing gate is
  unchanged.

Attribution check clean: `scripts/check_ai_attribution.py --range
f9ac84875..3b89fb61f` → exit 0 (no AI attribution).

## 6. Not changed

- `hc_split_sinkhorn` / `_sinkhorn_ops` (W32 / K3 territory) — untouched; the
  tapes call `hc_split_sinkhorn` so the K3 kernel drops in with no change here.
- `deepseek_v41_cache.py`, the MoE / streamed switch, the attention kernels,
  the W30 layer-major and W20 chunk-major drivers, one-shot / decode paths —
  unchanged; all byte-identical with the flag off.

## 7. Default OFF — the follow-up

`MTPLX_DSV41_HC_COMPILE` is opt-in so every existing gate stays green unchanged
and the eager per-call graph stays the serving default. The dispatch/decode win
is a **GPU-window measurement** (KERNEL_LEDGER **KG-f**: HC-compile + the CSA
attention default, argmax parity + decode +, expected smaller after K3). KG-f
runs this flag on against off *after* K3 (KG-c) exposes GPU time; on a pass a
one-line default flip lands it in serving.
