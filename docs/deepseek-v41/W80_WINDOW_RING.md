# W80 / K34 — bounded window ring + preallocated compress/index stores

**Goal.** Remove the per-token O(T) window store from the 16K decode path and the
memory-pressure it drives. W73/W76/W78 established that at T=16,384 the decode step
is ~506 ms/token with attention ~277 ms of it, and that the attention-*proper* cost
(~6.6 ms/layer, uniform across all four CSA modes) is **not** a per-op O(T) — it is
a fixed-shape cost inflated by **memory-residency pressure**: the full-history
window store is ~40×16384×512×2 ≈ **0.67 GB**, re-realloced every token by the
phase-1 `mx.concatenate`, thrashing the Metal allocator near the wired ceiling and
evicting streamed expert pages. K32 (`MTPLX_DSV41_KV_CHUNK_GROW`) made the append
byte-identical and amortized-O(1) *logically*, but W73 measured it did **not help on
Metal** because `mx.slice_update` did not donate in-place while the store view feeds
the attention graph.

W80 builds the reference's design: a **bounded ring** for the window store (the
window is a genuine sliding window of 128; only the last 128 rows are ever read) and
**preallocated** compress/index stores, all behind `MTPLX_DSV41_WINDOW_RING=1`
(default OFF). The resident window collapses ~128× (0.67 GB → ~5.5 MB across 40
layers), which — per the W76/W78 memory-pressure finding — is the lever that should
recover the ballooned context-independent decode stages far beyond the ~58 ms/token
raw append.

Deliverables: `mtplx/models/deepseek_v41_cache.py` (`_WindowRing`, telemetry, wiring,
drop-aware trim/rollback, eval-fence helper), `mtplx/models/deepseek_v41.py`
(`drop_offset` threaded through `_window_selected_idx` / `_window_attend` /
`_sparse_attend_selected` / `_attend` and `_eval_cache_state`), the `window_ring` /
`cell16k_ring` arms + `window_ring` receipt telemetry in
`scripts/deepseek_v41/ab_decode_env_levers.py`, and
`tests/models/test_deepseek_v41_window_ring.py`.

---

## The donation finding (item 2) — measured on the CPU double, 0.32.2

Tiny bf16 buffers (≤2 MB; the box guard kills a real-dims allocation). The donation
DECISION is a backend-agnostic refcount/graph property of the eval scheduler, so a
graph structure that blocks donation here blocks it on Metal too. Signal is
WALL-TIME vs capacity (µs/step, cap = ring rows), ratio 2K/512:

| pattern | 512 | 1024 | 2048 | 2K/512 | verdict |
|---|---:|---:|---:|---:|---|
| `mx.concatenate` (plain `_grow`) | 69 | 155 | 325 | **4.69×** | O(cap) copy |
| `slice_update`, result NOT read | 26 | 28 | 26 | **0.99×** | **donates**, O(new) |
| `slice_update` + result → matmul, same eval | 143 | 268 | 529 | **3.69×** | copies O(cap) |
| `slice_update` + result → matmul, **separate** eval (fenced) | 94 | 219 | 472 | **5.00×** | **still** copies |
| `slice_update` + result consumed via `mx.take` gather | 68 | 69 | 65 | **0.95×** | **donates**, O(new) |

**`mx.slice_update` DOES donate in 0.32.2** when the store buffer is uniquely
referenced — but donation is **blocked the moment the resulting buffer feeds a
downstream matmul/einsum in the graph**, and (key) **a fence between the append and
the read does not restore it**. This is exactly W73's "no help on Metal". The one
consumer that PRESERVES donation is a **gather (`mx.take`) that copies rows out**.

**Design consequences:**

* **Window ring (item 1):** the K30 selected path consumes the window via
  `_gather_rows` = `mx.take` → donation preserved; and the ring is bounded to ~136
  rows anyway, so even a copy is trivial (~5.5 MB/token across 40 layers vs the
  ~600 MB/token of the full store). The bound itself is the win.
* **Compress/index stores (item 2):** consumed by the indexer's einsum over **all**
  `n_comp` rows (a full read, not a gather) → donation **blocked**, fence won't fix
  it. Preallocation removes the O(log T) geometric resizes, and the CPU double
  proves the per-append write is **logically O(new rows)** (no full-store realloc:
  `buffers == 1` per lane over 200 appends, `rows_copied == N`, no resize copy). But
  making the physical Metal write truly in-place (donated) requires the store not be
  read in the same graph — which the indexer breaks — so the **true-in-place remedy
  is a custom `mx.fast.metal_kernel` that writes new rows into the preallocated
  buffer's storage directly**, bypassing the functional-donation machinery. That
  kernel is Metal-only and cannot be validated in a no-GPU window; it is a
  GPU-window follow-up. The `_GrowBuffer` write is the single choke-point where it
  drops in. On the CPU double the preallocated slice_update path is byte-identical
  and O(new-rows) logical; the append is ~3% of the 16K decode regardless (W73), so
  the window ring — not the compress/index append — carries the memory-pressure win.

---

## Design

### 1. `_WindowRing` — bounded window store

Fixed capacity `cap_keep = window_size(128) + max_verify(default 8) + slack(default
8)` ≈ 144 logical rows, held in a **pair of ping-pong buffers** of `phys_cap =
cap_keep + headroom(default 64)` rows, allocated once at construction. State:

* `_drop` — absolute position of physical slot 0 (rows `[_drop, _drop+_len)` are
  resident); the **logical `drop_offset`** the readers subtract.
* `_len` — resident physical rows. Logical length = `_drop + _len`.

`view()` returns the resident rows as a CONTIGUOUS array whose row *j* is absolute
position `_drop + j` — so a reader translates an absolute index by subtracting
`drop_offset`, addressing the same positions the full store would. Invariant: the
ring always holds a contiguous SUFFIX of the logical history, so `drop_offset` is
recoverable as `logical_len − resident` (used by `reseat` on state restore).

**Append** of `n` rows:
* while `_len + n ≤ phys_cap`: in-place donated `slice_update` at slot `_len`;
  `_drop` unchanged. (No allocation, no compaction.)
* else: one **compaction** copies the last `keep = max(cap_keep, n + window_size−1)`
  rows into the OTHER ping-pong buffer (never the same buffer — no aliasing) and
  advances `_drop`, dropping the older rows. `keep`'s `n + window_size − 1` term
  guarantees the current forward's oldest query (at the append's first position)
  retains its full causal window; dropped rows are strictly older than that window,
  so they are never read again. **No per-token allocation on the decode path**
  (ping-pong reuse); a prefill chunk wider than `phys_cap` is the only transient
  realloc (allowed — prefill fills the ring chunk-wise).

Cost: steady decode does an in-place write for `headroom` tokens, then one
`cap_keep`-row compaction — amortized ~1 row-copy/token, all bounded, no T-sized
array ever built during decode.

### 2. Preallocated compress/index stores

The indexer selects over the FULL compressed history, so these cannot be bounded.
Under the ring they are `_GrowBuffer(init_cap = maxkv)` — preallocated to the
configured `MTPLX_DSV41_WINDOW_RING_MAXKV` (0/unset → geometric fallback, still
byte-identical), so no geometric resize fires during decode; every append is an
in-place `slice_update`. See the donation finding for the Metal caveat + kernel
remedy.

### 3. Drop-aware verify seam (item 3)

The served DSpark loop rolls back a rejected draft via
`trim_verified_window_to_prefix` → `LayerAttentionCache.trim(n)`; the offline
whole-sequence path uses `mark`/`rollback`. Both restore the window ring by its
**LOGICAL** length (`truncate_to_length`), leaving `_drop` advanced: a compaction
during the speculative append is permanent, but the rows it dropped are always
≥ window_size behind the rolled-back query, so the reachable state is exact
(`cap_keep ≥ window + max_verify` guarantees the kept query's window survives). The
verify block (≤ K+1 = 4 rows at DSpark depth 3) is one multi-row append that never
overflows `cap_keep`, so no reachable row is dropped before the trim.

### 4. `_eval_cache_state` (item 4)

The settle fence now forces each lane's **raw preallocated backing buffer**
(`eval_backing()` → `_WindowRing.raw_backing()` / `_GrowBuffer.raw_backing()`), not
the logical `view()` slice — so a per-token fence realises this step's in-place
writes without materialising a fresh `[b, length]` (window) / `[b, n_comp]`
(compress/index) copy per token.

---

## Exactness argument

**Selected-key path (K30, cell16k) — BIT-IDENTICAL.** The path gathers a FIXED
`k = window_size + min(index_topk, n_comp)` keys via `mx.take`. `_window_selected_idx`
computes absolute window indices and subtracts `drop_offset`; every valid window
slot maps to the same absolute row as the full store (dropped rows are never in a
reachable window, so `valid` is unchanged). The compress gather uses the (full,
un-dropped) compress store. Same `k`, same keys, same order → the softmax over `k`
and the PV are bit-for-bit identical. **Measured:** tiny-model prefill + 280 decode,
max|Δ logits| = **0.0**, tokens identical, with the ring actively dropping (layer-0
`drop_offset` = 39 → 260+).

**Masked-full path (selected keys off) — reassociation-level, greedy-identical.**
The dropped rows would be masked to −∞ (contribute exactly 0), so the softmax result
is unchanged in value — but the score/PV reduction **width** shrinks from T to the
resident count, which reorders the f32 reduction tree. **Measured:** max|Δ| =
**8.3e-7**, tokens identical over 200 decode steps. This is in-family with the
model's other score levers (lean / fused-softmax) and is why the isolation arm
`window_ring` pins selected keys on; `cell16k` / `cell16k_ring` already run selected
keys, so the ring adds NO new lossiness vs `cell16k`.

**Verify seam.** DSpark greedy over the ring reproduces the AR reference exactly
(verify authoritative + exact drop-aware rollback), across cycles that exercise BOTH
accept and reject — the full `test_deepseek_v41_dspark_decode.py` suite (incl. the
256-token AR-reproduction and accept/reject tests) passes with `MTPLX_DSV41_WINDOW_RING=1`
on both the selected and masked-full paths.

---

## Arms + telemetry (`ab_decode_env_levers.py`, append-only)

* `window_ring` = `selected_keys=1 + window_ring=1` — the ring in isolation on its
  bit-identical (selected) composition.
* `cell16k_ring` = `cell16k` with the window ring replacing `kv_chunk_grow` — the
  direct A/B against `cell16k` isolating the ring's 16K decode effect.

Receipt `window_ring` block: `enabled`, `capacity` (cap_keep), `phys_capacity`,
`drops`, `rows_dropped`, `rows_copied` (physical rows written; flat-per-token =
amortized O(1)), `appends`, `reallocs` (~0 in decode). Reset after model load,
snapshot after the run — read it alongside the `cache_append` census stage exactly
as the `kv_chunk_grow` block.

Env knobs: `MTPLX_DSV41_WINDOW_RING` (master), `_MAX_VERIFY`, `_SLACK`, `_HEADROOM`,
`_MAXKV` (compress/index prealloc; 0 → geometric fallback).

---

## Tests (`tests/models/test_deepseek_v41_window_ring.py`, CPU-only, 15 tests)

`_WindowRing`: `test_ring_view_matches_full_history_suffix`,
`test_ring_prefill_chunk_then_decode_and_multi_row`,
`test_ring_truncate_and_reseat_exact`,
`test_ring_no_per_step_alloc_and_amortized_rows_copied` (no `mx.zeros`/`concatenate`
per steady step; rows_copied O(N) not O(N²)).
`LayerAttentionCache`: `test_layer_cache_ring_reachable_reads_match_plain`
(ratio 0/1/2), `test_layer_cache_ring_trim_and_rollback_drop_aware`,
`test_eval_backing_returns_raw_buffer_not_slice` (item 4),
`test_compress_index_preallocated_no_resize` (item 2: no geometric resize,
rows_copied O(new)), `test_window_ring_stats_reset_and_snapshot`.
End-to-end tiny model: `test_model_selected_path_bit_identical_with_ring`,
`test_model_masked_full_path_greedy_identical_with_ring`,
`test_no_T_sized_window_concat_during_decode`.
DSpark: `test_dspark_verify_reproduces_ar_with_ring` (depth 3, accept+reject, ring
reproduces AR).

## Follow-ups (GPU window)

1. Run `cell16k` vs `cell16k_ring` at the 16,384-token cell, fenced stage timing:
   the append lanes stop reallocating and, per W76/W78, the attention-*proper*
   ~6.6 ms/layer should drop if it was memory-pressure riding the window realloc
   churn. Report prefill/decode tok/s, peak GB, TTFT, wall (control vs candidate),
   and whether outputs are identical.
2. The custom `mx.fast.metal_kernel` in-place write for the compress/index lanes
   (the donation-blocked-by-indexer remedy) — Metal-only, validate under the flock.
