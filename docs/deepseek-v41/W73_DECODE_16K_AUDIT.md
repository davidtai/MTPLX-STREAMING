# W73 — where the 16K decode second goes (DeepSeek-V4.1-Flash streaming)

**Question.** Streaming decode measures ~6.0 tok/s at a 1,024-token context but
only 0.55 tok/s at 16,384 (receipt `docs/deepseek-v41/receipts/gpu-windows/
window-28b/prefill-16384-sel-80g.json`, arm `prefill_lean_sel`, 32 greedy tokens:
`decode_tok_s=0.5545`, `decode_wall_s=57.71`, so **1.82 s/token**; the 1K baseline
is ~0.167 s/token — an **extra ~1.65 s/token** at 16K). Expert streaming is
context-independent (6 experts/token), and with selected keys the reference
attention only touches `k = min(T,128)+min(512,T/r)` keys, so decode at 16K should
be nearly as fast as at 1K. It is not. This audit finds the O(T) work, statically
and with CPU microtests scaled to T=16384.

> **Program note (window 29):** the 16,384-token cell is the *only* benchmark for
> this model; the 1K decode numbers are void. The 11× gap below is the headline.

---

## Window-29b UPDATE — measured on the real model; the ranking was wrong

Window 29b ran the fenced decode census at 16K on the real model
(`receipts/gpu-windows/window-29b/ar-16k-{control,chunk}.json`). Two conclusions
**supersede the KV-append hypothesis that was ranked #1 below**:

**A. The KV-append is real O(T) but it is NOT the decode headline.** The fenced
census (ratios are the signal, not the inflated totals) per token at 16K:

| stage | ms/tok @16K | 1K (window-16) | ×  | O(T)? |
|---|---|---|---|---|
| `moe.routed_switch` | 263.8 | 82 | 3.2 | no (context-indep experts) |
| `attn.reuse` (30 layers) | 251.9 | 50 | 5.0 | mostly no (K30 bounds the score) |
| `hc.premix_sinkhorn` | 117.7 | 23 | 5.1 | no (fixed 20 iters) |
| `head` | 72.3 | 2.5 | 28.9 | no (1 row) |
| `attn.full` | 49.5 | — | — | partly (compress-append) |
| `attn.reindex` | 39.8 | — | — | partly (select) |
| `hc.combine` | 30.1 | 15 | 2.0 | no |
| **KV-append (all modes)** | **~58** | ~few | — | **YES** |
| indexer `select` (index srcs) | ~10 | — | — | YES (O(n_comp)) |

The genuinely T-scaling work — every `cache_append` (35.5 reuse + 10.7 full-compress
+ 5.1 reindex + 4.7 full + 2.5 swa ≈ **58 ms/tok**) plus the indexer `select`
(≈ 10 ms/tok) — is **~68 ms/tok, ~3–4 % of the ~1820 ms/tok decode**. A perfect
O(1) append cannot move decode more than ~3 %: window-29b measured
`prefill_lean_sel` 1.579 tok/s vs `prefill_lean_sel_chunk` 1.585 — a wash, as
predicted. **Cause 1 below (window append) is real but was mis-ranked as the
headline; it is a ~3 % item.**

**B. Why the append did not even drop: `mx.slice_update` does not donate in-place on
Metal here.** The flag *did* engage (`ar-16k-chunk.json` `arm_env` shows
`MTPLX_DSV41_KV_CHUNK_GROW=1`; `make_cache` reads it at `LayerAttentionCache`
construction — see the engagement test), but the census `cache_append` was unchanged
(reuse 35.5→36.8). CPU probes show donation holds **only when the store view is not
consumed by a graph**; the real decode feeds the view into the attention, so
`slice_update` copies O(cap)=O(T), same class as `concatenate`. So chunk-grow is
byte-identical but ~0 on Metal. **Do not add `MTPLX_DSV41_KV_CHUNK_GROW` to
`cell16k` as a perf lever** — it only takes the append off the O(T) list for the
record; the real append fix is a bounded window ring (cap = `window_size` +
verify_depth), which needs a drop-offset threaded through
`_window_selected_idx`/`_window_attend` and a drop-aware `mark`/`rollback` on the
speculative-verify seam — deferred (exactness-critical, needs Metal parity).

### The real 16K→decode inflation: a hypothesis table for window 31

The 1K→16K multipliers above are large for stages that are *context-independent by
construction* (`routed_switch`, `head`, `sinkhorn`, `hc.combine` all process one
token's worth of work per step). The unifying mechanism is almost certainly **wired-
memory pressure**, not per-token O(T) compute: at 16K the full-history KV stores are
~0.7 GB (window: 40×16384×512×2) + the compressed/index stores, pushing the resident
set toward the 85 GB / 100 GiB ceiling and **evicting expert slots and resident
weights that were resident at 1K**, so context-independent stages balloon from
re-streaming/re-reading off SSD.

| stage | 1K→16K | best-guess mechanism | window-31 check |
|---|---|---|---|
| `moe.routed_switch` 82→264 (3.2×) | **memory pressure**: KV growth evicts expert slots → more miss-I/O per token (experts themselves are context-indep) | does routed_switch track peak_gb / expert-cache eviction count, not context? |
| `attn.reuse` 50→252 (5.0×) | **mixed**: O(T) cache_append (35 ms) + the gather reshaping the full `[1,T,512]` store each token + memory pressure on the resident store | with `KV_CHUNK_GROW` on, does the *non-append* part of attn.reuse still grow? if yes → gather/residency, not append |
| `head` 2.5→72 (28.9×) | **memory pressure**: the ~1.3 GB head weight evicted at 16K → re-read/token (bf16 head off in this arm; the fp32-cast trap also promotes it — [[dsv41-head-fp32-cast-trap]]) | does `HEAD_MODE=bf16` (smaller, resident) collapse this row? |
| `hc.premix_sinkhorn` 23→118 (5.1×) | **memory pressure / bandwidth contention**: fixed 20-iter recurrence, 1 row; kernel off in this arm | does `SINKHORN_METAL=1` (kernel) or freeing KV memory collapse it? |
| `hc.combine` 15→30 (2.0×) | **bandwidth contention** under pressure (fixed hc_mult, 1 row) | tracks total-resident, not context |

If the mechanism is memory pressure, the highest-value lever is **shrinking the 16K
resident KV footprint** (a bounded window ring frees ~0.67 GB), which could recover
the ballooned context-independent stages far beyond the ~58 ms append. Window 31
should run the **same-arm 1K-vs-16K decode census** (`--context-tokens 1024
--stage-timing` and `16384 --stage-timing`, identical arm) and diff the stage ratios
against this table; and check whether `peak_gb` / expert-eviction counters correlate
with the ballooned stages.

### Engagement telemetry (added this window)

The receipt now carries a `kv_chunk_grow` block and the `--stage-timing` census
prints it: `enabled` (a layer cache chose the chunk-grown backing), `layers_chunk_grown`
/ `layers_plain`, `buffers` (geometric allocations), `appends`, `rows_copied`
(logical). Reading it: `enabled=false` ⇒ the flag never reached construction (env
timing / wrong class); `enabled=true` with flat `rows_copied` while the `cache_append`
census stage stays O(T) ⇒ `slice_update` did not donate on Metal. Window-29b is the
latter.

---

## The receipt reframes the problem

The measured arm has **`MTPLX_DSV41_SELECTED_KEYS=1`** and
**`MTPLX_DSV41_DECODE_ATTN_KERNEL=null`**. So at decode the attention **score is
already K30-gathered** (`_attend` → `_sparse_attend_selected`, `k≈window+index_topk
= 128+512 = 640`, T-independent). **The full-T score is NOT the cost here.** The
per-token O(T) that remains is everything K30 does *not* touch: the KV appends, and
the indexer selection that must run so K30 knows *which* rows to gather.

Config (released, per `W59_SELECTED_KEYS.md`): 40 layers, `head_dim=512`,
`window_size=128`, `index_topk=512`, `index_n_heads=32`, `compress_ratios` L0–1 `0`
/ L2–19 `2` / L20–39 `1`, `kv_source=[2,8,14,20]`,
`index_source=[2,8,14,20,24,28,32,36]`, `candidate_source=20`. At T=16384 the
ratio-1 layers (20–39) carry `n_comp = T = 16384` compressed rows.

## Ranked causes

Per-token cost is per layer × 40 layers unless noted. "CPU double" numbers are
`mx.set_default_device(mx.cpu)` microbenches at the real per-row byte widths, scaled
to T=16384; they establish the *scaling shape*, not the GPU constant.

### Cause 1 — window KV append is O(T). **[dominant fixable port artifact]**

`deepseek_v41_cache.py::_grow` (`mx.concatenate` of the whole store) via
`LayerAttentionCache.append_window`, called every token on **all 40 layers**
(`deepseek_v41.py::_attend`, the `cache_append` bracket). The window store is
`[b, T, 512]` bf16 and grows unbounded; the concatenate reads + writes the whole
thing each token.

- **Measured (CPU double):** a full-T window `concatenate` at T=16384 is **~2.1
  ms/layer** vs **~0.15 ms** at T=1024 — **~14× per layer**, linear in T; across 40
  layers ≈ **84 ms/token** at 16K vs ~6 ms at 1K on the CPU double.
- **Why it is pure waste:** the window is a genuine sliding window of 128. The
  decode gather (`_window_selected_idx`) reads only rows `[p−127 .. p]`; the
  masked-full path masks `wp > qp − window_size`. Everything older than the last
  `window_size` rows is never read. The reference keeps a fixed 128-row ring; the
  port keeps full history only to support arbitrary trim/rollback ("phase 1", per
  `deepseek_v41_cache.py` header).
- **Fix (implemented, exact-by-construction):** `MTPLX_DSV41_KV_CHUNK_GROW` — a
  geometric-capacity buffer + logical length, appended with a **donated
  `mx.slice_update`** in-place write. Amortized **O(new-rows)** per token (the
  copy-everything resize fires only on the O(log T) doublings). The `buf[:, :length]`
  view is **byte-identical** to the concatenated store, so every downstream read is
  unchanged.
- **Measured after fix (CPU double):** append wall is **flat ~14 µs** across cap
  4k/16k/64k (vs concat 20 µs→301 µs as T grows); cumulative rows-copied over 2000
  steps is **3,793 (~1.9 N)** vs the concat path's **2,003,001 (~N²/2)**.

### Cause 2 — compressed-KV + index-key append is O(n_comp). **[same fix]**

Same `_grow` concatenate on the **4 kv_source layers**, via `append_compress` /
`append_index_k` (`_publish_compressed`). Layer 20 is ratio-1 so it concatenates
`[b,16384,512]` (compress_kv) + `[b,16384,128]` (index_k) **every token**; layers
2/8/14 are ratio-2 (`n_comp=8192`, half that). ≈ **40–105 MB/token** of concat
traffic on top of Cause 1.

- Unlike the window, compress_kv **cannot** be bounded — the indexer selection is
  over the full compressed history — but the *append copy* is removable.
- **Fix:** the same `MTPLX_DSV41_KV_CHUNK_GROW` flag makes these lanes amortized
  O(1) too (all three lanes share `_GrowBuffer`), leaving only the inherent O(n_comp)
  *scoring* (Cause 3).

### Cause 3 — indexer scoring + top-k sort is O(n_comp). **[inherent, NOT a port bug]**

`Indexer.select` (`deepseek_v41.py`): `score = einsum("bshd,btd->bsht", q, index_k)`
over `n_comp` rows, then `_topk_rows` → **`mx.sort` over `n_comp`**, plus the
candidate-block prefilter on layer 20. Runs on the **8 index_source layers** every
token — **5 of them at `n_comp=16384`** (layers 20/24/28/32/36), 3 at 8192.

- This is O(T) and is the DeepSeek sparse-attention floor — the reference pays it
  too (the selection is by construction over the whole compressed history). **Not
  removable exactly.** The candidate-block prefilter (layer 20, block 8 → 2048
  blocks) already bounds the *second* level.
- **Not fixed** (labelled inherent). It is the residual O(T) after Causes 1+2 are
  removed; the extended stage timing measures it as `attn.<mode>.select`.

### Cause 4 — full-T attention score + double f32 cast. **[ruled out for this arm]**

Without `MTPLX_DSV41_SELECTED_KEYS`, `_sparse_attend_oneshot` scores the full
`KV=[b,T+n_comp,512]` (`scores=[b,s,H,T]`) and casts `KV.astype(f32)` **twice**
(qk + pv einsums) — the biggest O(T) of all (T×512×64 per layer + ~2×T×512×4 bytes
of f32 materialization). **The receipt has selected keys ON, so this is bounded to
k≈640 and is NOT the 16K cost here.** Flagged because any arm that drops
`SELECTED_KEYS` re-exposes it as the dominant O(T); keep selected keys in `cell16k`.

### Cause 5 — eval-barrier full-cache materialization. **[secondary, GPU-only]**

`_eval_cache_state` forces every layer's `window`/`compress_kv`/`index_k` each span;
at 16K that is ~0.7 GB (windows) + compress/index ≈ **>1 GB of arrays materialized
per token**, competing with expert residency near the 85 GB / 100 GiB wired ceiling.
Chunk-grow reduces the transient churn (no full-store realloc per token) but does not
remove the barrier itself.

## Honesty caveat on the absolute second

The summed bandwidth of Causes 1–3 is ~2–4 GB/token; at the box's 614 GB/s that is
**single-digit ms/token**, whereas the *observed* delta is **~1.65 s/token**.
Bandwidth alone does **not** explain 1.65 s. The large multiplier is a **GPU-only
effect not reproducible on the CPU double**: most plausibly (a) `concatenate`
reallocation churn thrashing the Metal allocator near the wired-memory ceiling
(peak 85 GB / 100 GiB; over-limit collapses ~4× — `never-exceed-the-memory-knob`),
and/or (b) the Cause-5 eval barrier forcing full-cache materialization that evicts
streamed expert pages → re-streaming. **This is why the deliverable includes the
extended `--stage-timing` at 16K** (per CSA mode class + `cache_append` +
`compress_append` + `select`): window 29 runs it on the real model to attribute the
constant across the append lanes vs the indexer floor. The chunk-grow fix removes the
one clearly-attributable, T-scaling *port artifact* (the append reallocations,
Causes 1+2), exact-by-construction.

## Fixes & env keys — SUPERSEDED by the Window-29b update above

> The pre-measurement plan below stands as the append analysis, but window-29b
> measured it: the append is ~3 % of decode and `slice_update` does not donate on
> Metal, so **`MTPLX_DSV41_KV_CHUNK_GROW` is byte-identical but ~0 on Metal — do
> NOT add it to `cell16k` as a perf lever.** Keep `MTPLX_DSV41_SELECTED_KEYS=1`
> (it bounds the attention score). The real 16K lever is the memory-pressure
> hypothesis table above, to be confirmed by window 31's same-arm 1K-vs-16K census.

| Key | Class | Fixes | Measured on Metal |
|---|---|---|---|
| `MTPLX_DSV41_KV_CHUNK_GROW=1` | **byte-identical** | Causes 1 + 2 (append O(T)) *in principle* | ~0 (slice_update no-donate); append is ~3 % of decode anyway |
| `MTPLX_DSV41_SELECTED_KEYS=1` | reassoc (already in target arms) | keeps Cause 4 ruled out | pre-existing (K30) |

New A/B arms in `scripts/deepseek_v41/ab_decode_env_levers.py`: **`kv_chunk_grow`**
(isolation vs control) and **`prefill_lean_sel_chunk`** (= the measured
`prefill_lean_sel` arm + the flag). These stay in the table as the byte-identical
A/B of record and to carry the engagement telemetry; they are not throughput levers.

## How to reproduce the attribution on the real model

```
ab_decode_env_levers.py --context-tokens 16384 --decode-tokens 256 --stage-timing \
    --arms prefill_lean_sel,prefill_lean_sel_chunk --out <receipt.jsonl>
```

Prints, per arm, the decode census: attention per CSA mode class (SWA-only / Full /
Reindex / Reuse) and — peeled out of each `attn.<mode>` via the new decode breakdown
(kept out of the flat sum) — `KV-append`, `compress-append` and `indexer-select`
ms/token, plus the cross-mode append totals. The append rows should collapse under
`kv_chunk_grow`; the `indexer-select` rows should not (the inherent floor).

## Evidence (CPU-only, `mx.set_default_device(mx.cpu)`)

- `tests/models/test_deepseek_v41_chunk_grow.py` — byte-identity of `_GrowBuffer`
  vs `_grow` (single-row, multi-row, set/truncate), LayerAttentionCache parity
  through trim/rollback/state for SWA/Full/Reindex/Reuse modes, an **end-to-end
  tiny-model prefill+decode with bit-identical logits flag on vs off**, and the O(T)
  counter (rows-copied amortized O(N), not O(N²); append wall flat vs T).
- `tests/models/test_deepseek_v41_stage_timing.py::test_decode_breakdown_*` — the
  decode breakdown records per-CSA-mode `cache_append`/`compress_append`/`select`,
  kept out of the flat partition sum.
