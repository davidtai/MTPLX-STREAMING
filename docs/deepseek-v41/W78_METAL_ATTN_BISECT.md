# W78 — Metal decode-attention op bisect (find the 16K O(T) on the GPU)

**Question.** Window 30 step 1 measured, on the real model (DeepSeek-V4.1-Flash
streaming, arm `cell16k`, 256 greedy decode tokens after a 16,384-token prefill,
fenced stage timing), that the **per-layer decode attention** is the 16K cost:
`attn.reuse` 6.6 ms/layer at T≈16.4K vs ~1.7 ms/layer at T≈1K (×30 layers →
197 ms/token), with `attn.full` 9.1, `attn.reindex` 7.6, `attn.swa_only` 6.6;
everything else (expert switch, Sinkhorn, head, HC) sits near its 1K value, so
attention is 277 of the 506 ms/token. W76
(`docs/deepseek-v41/W76_DECODE_ATTN_16K.md`) bisected the same ops on the **CPU**
backend and found **no per-op O(T)** in attention-proper — but the CPU double does
not reproduce Metal kernel behaviour: `mx.take` / gather over a large source,
non-contiguous cache views forcing copies, argsort/topk kernels, compiled-shape
specialisation, the `ATTN_WIN_MEMO` memo, and the K30 selected-key gather from the
T-row cache. **W78 builds a self-contained Metal microbench that peels the real
decode-attention path op-by-op at real dims, so the GPU window can name the O(T)
op.**

## What the script builds

`scripts/deepseek_v41/metal_decode_attn_bisect.py` reconstructs the **production**
decode-attention path from `mtplx/models/deepseek_v41.py` — no model load, no
artifact, no expert bank:

- One real `Attention` layer per CSA mode (`swa_only`=layer 0, `full`=2,
  `reindex`=24, `reuse`=3), with **random** bf16 weights (bf16 so the cache /
  activations carry the served dtype; that is what makes the masked-full path's
  `KV.astype(f32)` a real O(T) cast and the selected path's per-key cast O(k)).
- A **random pre-filled cache** to length T: the window store filled with the real
  window projection over T tokens (bf16, T rows); for `full` the compressed KV /
  index-key stores and the compressor frontier `comp_state.raw_kv` filled by the
  real `_publish_compressed` (f32, T//ratio and T rows respectively); for
  `reindex` / `reuse` the shared compressed KV / index keys / selection filled at
  the source layer's dtype and count (ratio-1 → bf16, n_comp=T; ratio-2 → f32,
  n_comp=T//2).
- Every `cell16k` attention-relevant env armed: `MTPLX_DSV41_SELECTED_KEYS=1`,
  `MTPLX_DSV41_ATTN_COMPILE=1`, `MTPLX_DSV41_ATTN_WIN_MEMO=1`,
  `MTPLX_DSV41_KV_CHUNK_GROW=1`, `MTPLX_DSV41_SELECT_FENCE=1` (the fused decode
  kernel `MTPLX_DSV41_DECODE_ATTN_KERNEL` is **off**, exactly as `cell16k` runs, so
  the eager selected-gather path is what is timed).

For each (mode, T) it times, per decode step (M=1) with `mx.eval` fences:

- **whole** — the production `Attention._attend` (the entry the decode loop calls),
  fenced once.
- **peeled** — a faithful mirror of `_attend` calling the production methods
  directly (`_attn_qkv_prep`/eager qkv, `LayerAttentionCache.append_window`,
  `_window_attend`, `_publish_compressed`, `Indexer.select`, `_mask_to_topk_idx`,
  `_sparse_attend_selected` / `_sparse_attend`, `_attn_out_prep`/eager o-LoRA),
  fencing after each op:
  `qkv_proj`, `cache_append` (window), `mask_build` (masked-full only),
  `compress_append` (`_publish_compressed` incl. the `comp_state.push`
  concatenate), `select` (indexer score + `_topk_rows` + the K30
  `_mask_to_topk_idx` **argsort**), `attend` (the K30 gather + score + sink-softmax
  + PV, one production call), and `out_proj`.
- **attribution rows** (not part of the peeled sum, so no double count):
  `gather_iso` — the selected-key gather in isolation (`_window_selected_idx` +
  `_gather_rows`, i.e. the `mx.take` from the T-row cache), and
  `score` = `attend − gather_iso` (the score+softmax+PV net of the gather).

It prints a ms/step table per op per T with the **16K/1K ratio** column (the O(T)
tell) per mode, and writes a JSON receipt (`--out`).

## Exact command line for the GPU window

Run **inside the exclusive GPU flock window** (all Metal work goes through the
flock — `[[gpu-work-always-through-flock]]`). The script arms the `cell16k` env
itself, uses one layer per mode built and freed sequentially (≤ ~0.4 GB resident
at 16K per layer, well under the 6 GB budget) and finishes in well under 3 min.

Baseline (the K30 selected-key path, matches `cell16k`):

```
PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w78 \
nice -n 19 /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3 \
  scripts/deepseek_v41/metal_decode_attn_bisect.py \
  --gpu --T 1024 4096 16384 --iters 30 --warmup 5 \
  --out docs/deepseek-v41/receipts/W78_metal_attn_bisect_w<WINDOW>_selected.json
```

Lever runs (attribute a cost; each to its own `--out`, append-only —
`[[never-overwrite-a-measurement]]`):

```
# masked-full path: score + KV.astype(f32) over the whole T (what SELECTED_KEYS avoids)
... metal_decode_attn_bisect.py --gpu --T 1024 4096 16384 --iters 30 --warmup 5 \
    --no-selected-keys --out .../W78_metal_attn_bisect_w<WINDOW>_maskedfull.json

# eager qkv/out projections instead of the K22 compiled tapes
... metal_decode_attn_bisect.py --gpu --T 1024 4096 16384 --iters 30 --warmup 5 \
    --compare-no-compile --out .../W78_metal_attn_bisect_w<WINDOW>_nocompile.json

# window-mask memo off (only bites the masked-full path; per-forward memo is a
# no-op for a single-layer microbench, so expect ~flat — kept for completeness)
... metal_decode_attn_bisect.py --gpu --T 1024 4096 16384 --iters 30 --warmup 5 \
    --no-win-memo --out .../W78_metal_attn_bisect_w<WINDOW>_nowinmemo.json
```

## How to read it

- The **16K/1K ratio** column per op is the O(T) tell. An op flat in T (ratio ≈ 1)
  is not the culprit; an op whose ratio tracks the ~3.9× whole-attention growth
  (6.6/1.7) is.
- Prime suspects on the selected path (all four modes do it): `gather_iso` — the
  `mx.take` gathers `window + index_topk` rows from a source that grows to T rows
  (window T+1, or the compressed store T//ratio); its cost on Metal can track
  source residency even though the gathered count is fixed (640/128).
- `select` is O(n_comp) and should grow on the index-source modes (`full`,
  `reindex`) but be ~0 on `reuse` (a shared-dict read) — `reuse` grew to 6.6 ms in
  the census with **no** select, which already rules select out as *reuse*'s O(T)
  and points at the gather.
- `compress_append` on `full` includes the `comp_state.push` concatenate over the
  f32 T-row frontier (an O(T) append **not** covered by `KV_CHUNK_GROW`, which only
  backs the window / compress_kv / index_k lanes) — watch whether it grows.
- Compare `--no-selected-keys`: the masked-full path scores the whole T and casts
  the whole bf16 KV to f32 (`KV.astype(f32)` folded into `attend`), the classic
  O(T) the selected path was meant to remove; its `attend`/`score` growth is the
  size of the lever.

## Unit test (CPU)

`tests/models/test_metal_decode_attn_bisect.py` runs the microbench at tiny dims,
CPU-pinned, T ∈ {256, 1024}, and asserts: all four modes execute both paths, every
op is finite/non-negative, the mode gating is right (swa has no compress/select;
full/reindex own the select; the isolated gather ran), and the **aggregate peeled
sum is within 20% of the whole** over all cells (a faithful decomposition — no op
missing or double-counted). Aggregating cancels the per-fence CPU sync jitter that
sub-millisecond tiny cells carry; the CPU run is a plumbing / scaling-shape check
only — it does **not** reproduce the Metal per-op O(T) (the reason W78 exists).

```
PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w78 \
  nice -n 19 /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3 \
  -m pytest tests/models/test_metal_decode_attn_bisect.py -v
```

## Notes / caveats

- Random weights + random cache: values are irrelevant to timing (only shapes /
  dtypes / counts set the memory traffic). Greedy identity is **not** a goal here.
- The peeled `attend` is one production `_sparse_attend_selected` call (gather +
  score bundled); `gather_iso` re-runs the gather in isolation for attribution and
  is deliberately excluded from the peeled sum.
- `head_dim` is the port's 512 (the released reference splits it 512 nope + 64
  rope into a 576 latent; the MTPLX port folds the rope tail into the 512-wide
  `wkv`/window store, so the cache row is 512 — this is the served path).
- Default device is **CPU** so an accidental worker run never touches the Metal GPU
  during a benchmark window; the GPU run needs the explicit `--gpu`.
