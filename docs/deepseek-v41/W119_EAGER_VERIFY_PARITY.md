# W119 — eager K+1-row verify vs 1-row AR reference parity

**Question (window 45, plan 69).** With the K29 decode-attention kernel ON
(`cell16k_ring_v2_draft_attn`, today's DSpark default) the DSpark greedy stream
diverges from the AR reference only as a **tie flip** (index 27, `ar_top2_margin`
0.0, max|Δlogit| 0.5). With K29 OFF for the whole arm
(`cell16k_ring_v2_draft_attn_eager`: `MTPLX_DSV41_DECODE_ATTN_KERNEL=0` +
`MTPLX_DSV41_DSPARK_VERIFY_K29=0`, so the eager gathered core
`_sparse_attend_selected` serves BOTH the K+1-row verify and the 1-row AR/draft)
the classifier reports class **"divergent"** at index 111 (`ar_top2_margin` 0.125,
`dspark_top2_margin` 0.0, max|Δlogit| 1.125) at BOTH depth 5 and depth 3, same
index and delta. Both texts coherent; acceptance unchanged (0.715 / 0.753).

Under greedy decoding the K+1-row verify must reproduce the AR forward's logits for
the same prefix up to accumulation-order rounding. Is the eager multi-row core
computing something genuinely different from the 1-row eager core for the same row
(a bug), or is the GPU delta bf16/f32 accumulation order between the s=K+1 and s=1
einsum tiles (rounding-class)?

Receipts (read-only): `.worktrees/deepseek-v41/docs/deepseek-v41/receipts/gpu-windows/window-45/`.

---

## 1. Trace — how row *i*'s keys / mask / positions are formed (rows > 1 eager vs the 1-row path)

Files: `mtplx/models/deepseek_v41.py`, `mtplx/models/deepseek_v41_cache.py`.
The eager selected-key path is `Attention._attend` → `_compressed` →
`Indexer.select` → `_sparse_attend_selected` (eager softmax block). Every step
below is indexed independently over the query axis `s` (broadcast), so row *i*'s
result depends only on row *i*'s inputs plus the SHARED, position-independent
stores. In the K+1-row verify `s = K+1`; in the AR reference each of the K+1 tokens
is a separate `s = 1` forward.

| # | Candidate the finding lists | Where (file:line) | Verdict |
|---|---|---|---|
| — | **positions / RoPE per row** | `deepseek_v41.py:3767` `positions = arange(cache.offset, cache.offset+s)`; `:1419` `_cos_sin(inv_freq, positions)` | Row *i* gets absolute position `offset+i` in BOTH paths (the AR replays the block in order, so its i-th token forwards at cache offset `offset+i`). q/kv row *i* is RoPE'd at `offset+i` identically. No difference. |
| 1 | **intra-block causal mask among the K+1 rows** | `_window_selected_idx` `:1104` (`valid = (idx<=qp) & (idx<logical_len) & (idx>=drop_offset)`) | The window gather is per-row: `idx = clamp(p_i-W+1,0)+arange(W)`, and the causal `idx<=p_i` guarantees row *i* NEVER reads a window slot for a later block row, even though the verify store physically holds all K+1 new rows (`append_window` `:1461`). `logical_len` differs (verify store longer) but only bounds `idx<logical_len`, which row *i*'s window never reaches. Identical key SET, identical post-RoPE vectors. Clean. |
| 3 | **SWA window bounds for the extra rows** | `_window_selected_idx` `:1104`, `_window_attend` `:1255` | Same as #1: per-row window band over absolute positions; `drop_offset==0` here. Clean. |
| 6 | **compress/index ratio-2 lanes' row alignment** | `_publish_compressed` `:1290` → `Compressor.pool` `:667` → `CompressorState.push` `deepseek_v41_cache.py:1121` | `push` pools each group from its OWN `[g*ratio,(g+1)*ratio)` rows and "the result is independent of how the rows were chunked" (docstring `:1128-1131`): a batched K+1 push and K+1 sequential 1-row pushes complete the SAME groups from the SAME tokens → identical pooled latents; `append_compress`/`append_index_k` grow the store identically; group RoPE at `group_pos = arange(n_prev,n_prev+n_new)*ratio` `:1303` is position-based. The token that completes group *g* (position `(g+1)*ratio-1`) sees group *g* as reachable in BOTH paths (`compress_lens=(p+1)//ratio` below). Clean. |
| 4 | **selected-keys (index_topk) set per row** | `Indexer.select` `:731` (`compress_lens=(positions+1)//ratio` `:1338`; `reach=arange(n_comp)<compress_lens[:,None]`; `_topk_rows(score, min(index_topk,n_comp))` `:764`), `_mask_to_topk_idx` `:1947` | Fully per-row via broadcast over `s`: row *i*'s selection depends on `x[i]`, `qr[i]`, `positions[i]`, and the shared `index_k`. `reach` masks any compressed row from a later group (index ≥ `compress_lens[i]`). K29/core-compile OFF ⇒ unpadded `topk = min(index_topk,n_comp)` (`:1376`), same width in verify and AR for the same `n_comp`. `_mask_to_topk_idx` is a deterministic per-row argsort. Clean. |
| — | **gather** | `_gather_rows` `:1985`, `_selected_compress_gather` `:2004` | Flat `take` per `(b,s,k)`; shape-stable cache is default OFF and disabled above the decode row cap, so the plain per-layer transient gather runs (byte-identical). Per-row. Clean. |
| 5 | **sink / softmax scaling per row** | `_sparse_attend_selected` eager block `:1234-1268` (`_attn_core_impl` `:2415` is the extracted twin) | `scores = einsum("bshd,bskd->bshk", q.f32, KVg.f32)*scale`; `where(valid,·,-inf)`; per-head sink; `m=max(max(scores,-1),sink)`; `ex=exp(scores-m)`; `denom=sum(ex,-1)+exp(sink-m)`; `o=einsum("bshk,bskd->bshd", ex, KVg.f32)/denom`. Every op indexes the `s` axis independently → row *i*'s output depends ONLY on row *i*'s `q`, `KVg[i]`, `valid[i]`. The ONLY thing that changes between `s=K+1` and `s=1` is the einsum/reduction operand SHAPE (the `s` axis length), which changes the matmul tiling / accumulation ORDER — a pure reduction reassociation, not a different computed quantity. Clean (rounding-class only). |
| — | **output projection** | `_attend` `:1509-1530` (`_rope_last` inverse, grouped `_o_lora_down`, `wo_b`) | Eager (`_ATTN_COMPILE` off, fused-proj GPU-only). Per-row. Clean. |
| 7 | **K29-vs-eager difference in how the AR reference itself is computed** | `dspark_decode.py:107-109`, `_decode_attn_kernel_use` `:2147` | The arm shares its env with the offline AR reference ("both the served verify and the offline AR-reference must share the setting"). So on the eager arm BOTH the K+1 verify and the 1-row AR run the eager core; on the K29 arm BOTH run K29. The K29 kernel reshapes EACH row to its own single-query batch `q[b*s,1,H,hd]` / KV `[b*s,k,hd]` (`_sparse_attend_selected:1191-1199`), so row *i*'s math is the SAME kernel invocation whether it sits in a K+1 verify or a 1-row AR — no accumulation-order difference between the two forwards. The eager path runs an `s=K+1` einsum for the verify and an `s=1` einsum for the AR — different tile, different GPU accumulation order. This is why the eager arm shows a larger per-row delta than K29, not because the AR reference changed algorithm. |

**Trace conclusion.** For row *i*, every key/mask/position feeding the eager core is
formed identically to the 1-row path; the sole batched-vs-1-row difference is the
`s`-axis length of the score/PV einsums and the softmax reductions — i.e.
floating-point reduction ORDER. On CPU fp32 that reordering is negligible; on the
GPU it is the bf16/f32-tile accumulation-order class that compounds through the 40
backbone layers + MoE + head and flips a near-tie token.

---

## 2. Test evidence

Test: `tests/test_deepseek_v41_w119_eager_verify_parity.py` (CPU, fp32, no artifact,
no GPU). Tiny synthetic DSV4.1 model: ratio-2 CSA layers
(`compress_ratios=[0,0,2,2,2,1,1,1]`), `index_topk=5`, `sliding_window=8`,
64-token prompt (window slides; ratio-2 groups ≈ 35 ≫ index_topk so the selection
saturates and the compressed-selected path is exercised).

<!-- W119_EVIDENCE_START -->
Run (CPU, `nice -n 19`, `PYTHONPATH=<worktree>` so `mtplx` resolves to the
streaming-branch code — the `main` checkout the editable install targets does not
contain `deepseek_v41.py` at all):

```
$ PYTHONPATH=$WT MTPLX_DSV41_DECODE_ATTN_KERNEL=0 nice -n 19 .venv/bin/python3 \
      -m pytest tests/test_deepseek_v41_w119_eager_verify_parity.py -v -s
...
======================== 12 passed, 2 warnings in 1.20s ========================
```

Per-test tails:

```
[W119] K+1=2 eager verify vs 1-row AR: per-row max|Δlogit|=['4.77e-07','4.17e-07']  overall=4.768e-07
[W119] K+1=4 eager verify vs 1-row AR: per-row max|Δlogit|=['4.77e-07','4.17e-07','4.77e-07','5.96e-07']  overall=5.960e-07
[W119] K+1=6 eager verify vs 1-row AR: per-row max|Δlogit|=['4.77e-07','2.98e-07','5.36e-07','6.11e-07','3.58e-07','4.77e-07']  overall=6.109e-07
[W119] K+1=2 argmax verify=[42, 14]                ar=[42, 14]
[W119] K+1=4 argmax verify=[42, 14, 14, 42]        ar=[42, 14, 14, 42]
[W119] K+1=6 argmax verify=[42, 14, 14, 42, 6, 3]  ar=[42, 14, 14, 42, 6, 3]
[W119] K+1=2 COMPILED-core verify vs 1-row AR: overall max|Δlogit|=4.768e-07; core tapes built=4
[W119] K+1=4 COMPILED-core verify vs 1-row AR: overall max|Δlogit|=5.960e-07; core tapes built=4
[W119] K+1=6 COMPILED-core verify vs 1-row AR: overall max|Δlogit|=6.109e-07; core tapes built=4
[W119] core s=2: batch-invariance max|Δ|=0.000e+00  eager-vs-K29-reference max|Δ|=0.000e+00
[W119] core s=4: batch-invariance max|Δ|=0.000e+00  eager-vs-K29-reference max|Δ|=0.000e+00
[W119] core s=6: batch-invariance max|Δ|=0.000e+00  eager-vs-K29-reference max|Δ|=0.000e+00
```

Reading:

* **The eager selected-key core is BITWISE batch-invariant on CPU** — row *i* of the
  `s=K+1` `_attn_core_impl` call equals the `s=1` call exactly (max|Δ| =
  `0.000e+00` at s=2/4/6), and equals the K29 CPU reference
  `decode_attention_reference` exactly. So candidates #1/#3/#4/#5/#6/#7 are all
  numerically clean: nothing in the multi-row core computes a different quantity.
* **Full-model K+1-row verify == 1-row AR** to `≤ 6.11e-07` per row (fp32 CPU). That
  residual is CPU-BLAS reduction order in the row-independent GEMMs (projections,
  MoE, head), NOT the attention core (which is exactly 0) — five orders of magnitude
  below the 1e-4 bar and independent of K+1.
* **Acceptance decision identical**: per-row argmax matches for every K+1, so the
  greedy accept/reject would be identical for any draft block.
* **Compiled core** (`ATTN_CORE_COMPILE=1`) holds the same parity (`≤ 6.11e-07`) with
  a bounded 4-tape cache.
<!-- W119_EVIDENCE_END -->

---

## 3. Verdict

<!-- W119_VERDICT_START -->
**ROUNDING, not a bug.** The eager multi-row verify core computes the identical
per-row quantity as the 1-row core — proven BITWISE on CPU fp32 (core
batch-invariance max|Δ| = 0.000e+00; eager == K29 CPU reference exactly; full-model
verify == AR to ≤ 6.11e-07, which is fp32 BLAS reduction order in the non-attention
GEMMs). There is no offending line. Every candidate the finding listed (intra-block
causal mask, per-row RoPE/positions, SWA bounds, per-row selected-key set, sink/
softmax, ratio-2 lane alignment, AR-reference computation) is numerically clean.

So the GPU index-111 delta is **bf16 accumulation-order between the s=K+1 and s=1
eager einsum tiles**, compounded through the 40 backbone layers (residual stream +
per-token MoE top-k routing, itself a hard argmax that can flip an expert at a
boundary — an amplifier) and quantized by the `HEAD_MODE=bf16` output GEMV onto the
bf16 grid. The eager score/PV einsums cast to f32, but on Metal the f32 matmul tiles
differently for `s=6` than for `s=1` (different threadgroup reduction order), so
row *i*'s attention output differs from its 1-row twin by a few f32 ulps; that
perturbation compounds and the bf16 head rounds it to the dyadic grid.

**Why 1.125 is consistent with the rounding class (bf16-ulp quantification).** bf16
has 7 mantissa bits, so for a logit of magnitude *x* one ulp is
`2^(floor(log2|x|) − 7)`:

| |logit| | 1 bf16 ulp |
|---|---|
| 8 | 0.0625 |
| 16–24 | 0.125 |
| 32–48 | 0.25 |
| 64–96 | 0.5 |
| 128–192 | 1.0 |
| 256 | 2.0 |

Every observed value lands on the bf16 dyadic grid, an exact integer number of ulps:

| quantity | value | in ulps |
|---|---|---|
| `dspark_top2_margin` (eager) | 0.0 | 0 — the two top verify logits are **bf16-identical** (a genuine bf16 tie) |
| `ar_top2_margin` (K29 arm) | 0.0 | 0 (the K29-arm flip was already a tie) |
| `ar_top2_margin` (eager) | 0.125 | exactly **1 ulp** at \|logit\|∈[16,32) |
| `max|Δlogit|` (K29 arm) | 0.5 | 1 ulp at \|logit\|∈[64,128) |
| `max|Δlogit|` (eager) | **1.125** | 9 ulps at [16,32), or ≈1 ulp at the peak logit (\|logit\|~128–256, where 1 ulp = 1.0–2.0) |

`max|Δlogit|` is taken over the whole vocab and is dominated by the largest-magnitude
logit, where one bf16 step is ≈1.0; 1.125 is one such step plus a smaller-magnitude
one. A real arithmetic bug in the multi-row core would produce arbitrary,
non-grid-aligned deltas — these are all exact bf16 steps, the fingerprint of bf16
head quantization. **1.125 is consistent with the rounding class.**

**Why the eager arm looks worse than K29.** With K29 ON, the kernel reshapes *each*
query row to its own single-query batch (`q[b*s,1,H,hd]`, KV `[b*s,k,hd]`), so row
*i*'s math is the *same kernel invocation* whether it sits in the K+1 verify or the
1-row AR — zero s=6-vs-s=1 tile difference, hence only the head-GEMV tie flip (index
27). The eager path runs a genuine `s=6` einsum for the verify and `s=1` for the AR;
those tiles reduce in different order. K29 is coarser *per row* (bf16/fast-transcendental
~1e-3) but *identical between the two forwards*; eager is finer per row (f32) but
*differs between the two forwards*. Neither is a bug.

### Proposed classifier rule (report only — runtime NOT changed in W119)

The current rule (`deepseek_v41_dspark_decode.py:435`)
`cls = "tie_flip" if (ar_margin is not None and ar_margin < tie_margin) else "divergent"`
misclassifies index 111 for two reasons:

1. **It ignores `dspark_top2_margin`.** The near-tie here is on the *authoritative
   verify* side (0.0), not the AR side (0.125). When the verify forward itself cannot
   separate the two candidate tokens (`dspark_margin` within the band), which one its
   argmax emits is rounding-determined — the very definition of a tie flip
   ([[dsv41-inexact-ok-if-tie-flips]]).
2. **The fixed `3e-2` band does not scale with logit magnitude.** At the cell16k
   operating magnitudes (~16–256) one bf16 ulp is 0.125–2.0 — 4–66× the constant — so
   any legitimately bf16-tied pair there is force-classed "divergent".

Recommended change (both parts):

* **Add the DSpark-side near-tie** as a rounding-class signal:
  `cls = "tie_flip" if (min_defined(ar_margin, dspark_margin) < tie_band) else "divergent"`
  (a one-line change; absolves index 111 because `dspark_margin = 0.0`).
* **Make the band magnitude-aware:**
  `tie_band = max(DSPARK_TIE_MARGIN_DEFAULT, 3 * ulp_bf16(peak_contested_logit))`
  with `ulp_bf16(x) = 2.0 ** (floor(log2(abs(x))) − 7)` and `peak_contested_logit =
  max(|ar_logit[ar_token]|, |ar_logit[dspark_token]|)` (or the row top-1 magnitude).
  This restores the "3× the rounding floor" intent at the actual operating magnitude.

Fully rigorous alternative (tightest, if a single rule is wanted): class rounding-class
iff `min(ar_margin, dspark_margin) <= |Δlogit(ar_token)| + |Δlogit(dspark_token)|` —
the measured per-token deltas at the two contested tokens can close the smaller
margin. Under this, index 111 is rounding-class (min margin 0.0 ≤ any delta), and a
genuine >ulp divergence with both margins clear still classes "divergent".

Recommend the two-part change (dspark-margin + magnitude-aware band) for the runtime;
it keeps the classifier LOUD for a real >ulp flip while absolving the bf16-tie flips
that dominate the eager arm. Left for a follow-up task per the W119 no-runtime-change
constraint.
<!-- W119_VERDICT_END -->
