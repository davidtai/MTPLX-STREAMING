# W90 — in-situ decode-attention overhead: the mechanism

## Verdict (ranked)

The ~5 ms/layer gap between in-model decode attention (window-34 census: reuse
**6.96** ms/layer @16K, expert-stubbed **6.10**) and the isolated Metal microbench
(**~2.0** ms/layer, flat across T=1K/4K/16K, even with 88 GiB ballast) is **not a
host-side cost**. Ranked by contribution:

1. **(DOMINANT) O(T) source-buffer references in the per-layer decode gather.**
   Every decode token, each Reuse / Reindex / Full layer's
   `_sparse_attend_selected` gathers a *bounded* slice out of **two O(T) source
   buffers**: its own window store `window_all` `[1, T, hd]` and the group's
   **shared** `compress_kv` `[1, n_comp≈T/2, hd]`. Only `window_size + index_topk`
   rows are read, but the *source arrays scale linearly with T*. On the CPU this
   costs nothing that scales with T (see E5/E6 — host wall and per-fn time are
   FLAT in T); on Metal each tiny **B=1** dispatch pays a residency/encode cost
   that scales with the *referenced source size* (the W78/W80 "resident churn",
   [[b1-decode-dispatch-removal-hides]]: a B=1 wall is host-encode + big-kernel
   chain). In-model this is amplified because **40 distinct layers each reference
   their own O(T) buffers every token** — a per-token working set ~40× the
   isolated 1-layer bench, which is exactly why the isolated bench is flat 2.0 and
   why *unreferenced* ballast (window 31) never reproduced it. The W80 window ring
   bounded **only** `window_all` (0.7 GB → 5.5 MB) and recovered ~12 ms/token; the
   **shared `compress_kv`**, referenced by ~34 of 40 layers, stays O(T) — the
   residual ~140 ms.

2. **(SMALL, real) the stage/frame `mx.eval` fence is the vehicle.** The fence
   (`_Fence.add` → `mx.eval(fence._arrays)`, deepseek_v41_stage_timing.py) forces
   each layer's O(T)-referencing kernel to complete inside the `attn.<mode>`
   bracket, and additionally drains the *previous* layer's async routed-expert
   gather into the attention time — receipt window-34 expert-stub = **−0.86**
   ms/reuse-layer (6.96 → 6.10). Real but not the bulk; the task's expert-stub
   pass already showed "the switch is not it."

**Ruled out with CPU evidence (candidates #1, #2, #5):**

3. **#1 compile retrace per token — RULED OUT.** The K22 attention tapes
   (`_attn_qkv_prep`, `_attn_out_prep`) take only `s==1`-shaped inputs
   (`x[1,1,hidden]`, `qcos/qsin[1,rd]`, weights) and are keyed by codec/geometry,
   **not T**. E1: instrumenting `mx.compile` to count real traces, each impl
   traces **exactly once** and replays across all 32 decode steps *with T
   advancing 64→96* — zero per-step retrace; `_ATTN_COMPILED` cache stays at 4
   entries. No `shapeless=True` needed; the traced graph has no shape-dependent
   Python on the token axis (`unflatten`/`flatten` are on the fixed head axis).

4. **#2 ATTN_WIN_MEMO miss — RULED OUT for cell16k.** With `SELECTED_KEYS=1`
   (cell16k) the memoized `_window_attend` is **never called** (E2: 0 calls over
   10 decode steps) — the K30 selected-key path uses `_window_selected_idx`
   (bounded `[s, W]`, W=window_size) instead. Even with selected keys OFF the memo
   is not a per-token miss storm (E2: 80 calls / 10 steps, **70 hits** = 7/8; only
   the first layer per token recomputes because `positions` is a fresh object).

5. **#5 Python cache-lane bookkeeping — RULED OUT.** E5 cProfile of 30 decode
   steps: total **0.281 s** (T=1024) vs **0.294 s** (T=4096) — +4.6%, entirely the
   index/kv-source `Indexer.select` O(T) sort (already attributed, ~10 ms/token,
   fine). `_sparse_attend_selected` tottime is **identical** (0.004 s) at both T;
   `_gather_rows`, `_compressed`, `comp_state`/frontier bookkeeping do not scale.

## Evidence (CPU reproduction, tiny fake config, no artifact, T ≤ 4096)

Harness: tiny `ModelArgs` (4 heads, head_dim 16, 8 layers, `compress_ratios=
[0,0,2,2,2,1,1,1]`, kv-source {2,5}, index-source {2,5,6} → reuse layers {3,4,7}),
`mx.set_default_device(mx.cpu)`, cell16k decode env (`SELECTED_KEYS/ATTN_COMPILE/
ATTN_WIN_MEMO/KV_CHUNK_GROW`). The tiny model reaches T=4096 cache lengths cheaply
on the CPU, so any *host-side* T-scaling is directly reproducible; the mechanism is
counted (compile-cache entries / memo calls / source shapes), never timed in ms.
(Capped at T ≤ 4096 after the box guard killed a T=16384 real-shape run at 4.3 GB.)

| Exp | Result | Reads on |
|-----|--------|----------|
| **E1 retrace** | attn qkv/out impls trace **1×**, replay 31× over 32 steps (T 64→96); cache size 4, stable | #1 ruled out |
| **E2 memo** | `SELECTED_KEYS=1`: `_window_attend` **0 calls**; OFF: 80 calls, 70 hits (7/8) | #2 ruled out |
| **E3 T-source** | reuse gather sources scale with T: `window_all` 16 448→65 600→**262 208 B**, `compress_kv` 8 192→32 768→**131 072 B** (T=256/1024/4096); `comp_idx` `[1,1,5]` bounded | #3 confirmed |
| **E3b reshape** | `buf[:,:len].reshape` + eval **flat ~16 µs** across len 256/1024/4096 (b=1 view is contiguous; no O(T) copy) | reshape-copy sub-hyp negative |
| **E5 cProfile** | 30-step decode 0.281 s (1K) vs 0.294 s (4K); `_sparse_attend_selected` tottime 0.004 s both | #5 ruled out |
| **E6 host wall** | per-reuse-layer host wall **923 → 916 → 915 µs** (T=256/1024/4096) — FLAT | discriminator: no host T-scaling |

E6 is the discriminator: reuse per-layer host wall is flat in T while the in-model
Metal census scales 1.7 (1K, window-16) → 6.96 (16K). Flat CPU + scaling Metal ⇒
the cost is Metal-runtime (residency over the O(T) *referenced* sources), not host.
This matches W76 (CPU bisect found no per-op O(T) in attention-proper — the O(T) is
in source *size*, not op output) and explains why the W80 ring under-recovered.

## Fix — `MTPLX_DSV41_ATTN_SHAPE_STABLE` (K35, default OFF, byte-identical)

Bounds the **compressed lane** the ring cannot (the indexer needs `compress_kv` in
full, so the store can't be shrunk — but the *gather* can be shared). All ~34
non-swa layers of a group read the SAME `(compress_kv, selected_idx)` pair, so the
shipped path re-gathers `index_topk` rows out of the O(T) `compress_kv` once per
layer (~34 O(T) compress references/token at 16K). The lever gathers the selected
compressed KV **once per source** and shares the bounded `[b, s, k, hd]` result on
the per-forward `SharedAttentionRuntime` (`_selected_compress_gather`, keyed by the
*identity* of the two source arrays), so only the first layer of a group references
the O(T) store — **~34 → ~3 O(T) compress references/token** (the 3 index sources).

- **Byte-identical**: a pure caching of the deterministic K30 gather — same rows,
  same order. `shared` is fresh per forward (`new_shared_runtime`), so nothing
  leaks across tokens; a new source publishes new arrays whose identity misses the
  cache and re-gathers. Verified `mx.array_equal` over **64 decode steps**
  (max|Δ|=0), and composed with `SELECT_FENCE` / `KV_CHUNK_GROW`.
- **Composes with the ring** = arm **`cell16k_ring_stable`** (bounds *both* O(T)
  lanes: ring → per-layer window store; shape-stable → shared compress store).
- Touched: `deepseek_v41.py` (`_resolve_attn_shape_stable`,
  `_selected_compress_gather`, `_sparse_attend_selected(..., shared=)` + the
  `_attend` call site). Tests: `tests/models/test_deepseek_v41_attn_shape_stable.py`
  (9 passed). Regressions green: selected_keys 15, ab_env_levers 65, window_ring
  15, stage_timing 17, dspark_decode 24, decode_attn_16k 5.

**New A/B arms (append-only, `ab_decode_env_levers.py`):**
- `attn_shape_stable` = `selected_keys=1, attn_shape_stable=1` (isolation).
- `cell16k_ring_stable` = `cell16k_ring` + `attn_shape_stable=1`.

## What this does NOT claim

The Metal recovery is a **GPU-window measurement** — not made here. The CPU proves
the *mechanism* (O(T) source references, host-flat) and the fix's *byte-identity*;
it cannot measure the residency/encode cost (invisible on CPU; unreferenced ballast
did not reproduce it). Next GPU window: A/B `cell16k` vs `cell16k_ring` vs
`cell16k_ring_stable` on the standard 16,384 cell, reading the decode-attention
census per CSA mode (expect the compressed-lane residual to fall on the ~34
non-swa layers; the window lane is the ring's, already ~12 ms). If the residency
model is right, `cell16k_ring_stable` should recover materially more than the ring
alone on reuse/reindex layers.
