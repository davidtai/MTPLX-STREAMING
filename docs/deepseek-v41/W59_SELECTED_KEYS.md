# W59 — K30 prefill selected-key gather (`MTPLX_DSV41_SELECTED_KEYS`)

Worker `feat/deepseek-v41-w59` (branched off `feat/deepseek-v41-streaming`). CPU-only,
MLX pinned to `mx.cpu`, no artifact load, <3 GB RSS. No GPU/Metal. Analysis + a
prefill-only, default-OFF lever + CPU exactness proof; no GPU window run here.

## 0. The question, answered from the reference

> In Reuse and Reindex/Full modes, does the reference attend over ALL compressed
> keys with a mask, or does it gather only the `k` selected candidate keys so the
> score is `[rows, 64, k]`?

**It gathers only the `k` selected keys.** The score is `[rows, 64, k]`, never
`[rows, 64, T]`. Evidence, from the reference used for the torchref goldens
(`~/models/DeepSeek-V4.1-Flash-src/inference/`):

- **The kernel gathers.** `sparse_attn_kernel` (`kernel.py:311`) sizes its work as
  `num_blocks = tilelang.cdiv(topk, block)` (`kernel.py:325`) and loads only the
  selected rows: `kv_shared[i,j] = if idxs[i] != -1: kv[by, idxs[i], j] else 0`
  (`kernel.py:362`), scoring `q_shared @ kv_shared^T` over exactly those `topk`
  columns. `topk` is the **width of `topk_idxs`**, not `T`.
- **`topk_idxs` is window ∪ compressed selection.** `Attention.forward` concatenates
  `kv = cat(window_kv, compress_kv)` and `topk_idxs = cat(window_idxs, compress_idxs)`
  then calls `sparse_attn(q, kv, sink, topk_idxs, scale)` (`model.py:777-780`).
  - window part: `get_window_topk_idxs` (`model.py:410`) → per query `p`, indices
    `{max(0,p-W+1) .. p}` (`clamp(p-W+1,0)+arange(W)`, future `idx>p → -1`,
    `model.py:420`); width `min(seqlen, W)`.
  - compressed part: `Indexer.forward` picks `topk = min(index_topk, end_pos//ratio)`
    (`model.py:578`) via `index_score.topk(topk).indices.sort()` (`model.py:579`),
    padded to `-1` where unreachable. Candidate prefilter (`candidate_source_layer`)
    only **masks which** rows are eligible; it does not change the final count.
- **The golden gathers identically.** The pure-torch oracle `_k_sparse_attn`
  (`scripts/deepseek_v41/torchref/ref_forward.py:135`) does
  `torch.gather(kv_e, 2, idxc…)` (`:145`) then `einsum("bmhd,bmtd->bmht")` over the
  gathered `k` — so the golden semantics are the gather, not a `T`-wide mask.

This holds for **every mode that attends compressed rows** — Full (`kv_source`,
computes its own KV + indexer), Reindex (`index_source` not `kv_source`, own indexer
over the shared KV), and Reuse (reads the source's `topk_idxs`) — because they all
route through the one `sparse_attn` call above. SWA-only layers (ratio 0) gather just
the window band.

### Is the port's form a faithful-but-wasteful transliteration? **Yes.**

The MLX port (`mtplx/models/deepseek_v41.py`, `_attend` → `_sparse_attend_oneshot`)
builds `KV = cat(window_all, compress_kv)` (the full history) and `attend =
cat(window_mask, topk_mask)`, then scores the **full `[rows, 64, T]`** and applies
the mask. Masked keys get `-inf` → `exp = 0`, so they contribute **exactly 0** to the
softmax numerator and denominator: the result is **mathematically identical** to the
reference gather. The port simply materializes and computes over `T` columns where
the reference touches `k`. That is the source of the W47/W50-measured cost: the score
transient grows with `T` (per chunk), while the reference's work is fixed.

## 1. Exact key count `k` per mode as a function of T and the config

Config (`~/models/DeepSeek-V4.1-Flash-src/config.json`): `window_size=128`,
`index_topk=512`, `compress_ratios` = L0-1 `0`, L2-19 `2`, L20-39 `1` (MTP L40-42 `0`),
`kv_source_layers=[2,8,14,20]` (ratios 2,2,2,1), `index_source_layers=[2,8,14,20,24,28,32,36]`,
`candidate_source_layer=20`, `candidate_topk_blocks=2048`, `candidate_block_size=8`.

For a layer of ratio `r` at context `T` (full prefill, `start_pos=0`):

```
k = min(T, window_size)  +  ( min(index_topk, T // r)   if r > 0 else 0 )
  = min(T, 128)          +  ( min(512, T // r)           if r > 0 else 0 )
```

| mode | layers | ratio | window | compressed | **k @ T=1024** | **k @ T=16384** |
|---|---|---|---|---|---|---|
| SWA-only | 0-1 (+MTP) | 0 | 128 | — | **128** | **128** |
| Full / Reuse | 2-19 | 2 | 128 | min(512, T/2) | **640** | **640** |
| Full / Reindex / Reuse | 20-39 | 1 | 128 | min(512, T) | **640** | **640** |

**`k` saturates and is independent of T** for all these shapes: `T//r ≥ 512` already
at 1K (ratio-1: 1024≥512; ratio-2: 512≥512), so the compressed count pins at
`index_topk=512` and the window at 128 → **k = 640** (compress layers) / **128** (SWA)
at both 1K and 16K. The candidate prefilter is a no-op at ≤16K (`candidate_topk_blocks
× block_size = 2048×8 = 16384` ≥ every reachable compressed count), so it never
reduces `k` in the standard benchmark shapes.

## 2. Score FLOPs / bytes before vs after, and the per-chunk growth curve

16K, layer-major, default 8 GB chunk target → chunk 953, 18 chunks. Control score
width for a ratio-1 layer **grows** per chunk; K30's is flat:

| chunk | cum tokens | control width (`cum + cum/r`) | K30 width (`128 + min(512, …)`) | ratio |
|---|---|---|---|---|
| 1 | 953 | 1,906 | 640 | 3.0× |
| 5 | 4,765 | 9,530 | 640 | 14.9× |
| 9 | 8,577 | 17,154 | 640 | 26.8× |
| 13 | 12,389 | 24,778 | 640 | 38.7× |
| 17 | 16,201 | 32,402 | 640 | 50.6× |
| 18 | 16,384 | 32,768 | 640 | 51.2× |

Totals over the 40 backbone layers (QK^T + PV, `4·rows·H·width·d`):

| metric | control | K30 | ratio |
|---|---|---|---|
| total score FLOPs (16K) | 6.42e14 | 2.64e13 | **24.4×** |
| reuse-layer score FLOPs (30 of 40) | 4.88e14 | 2.06e13 | **23.7×** |
| peak fp32 score transient / chunk (ratio-1 layer) | 7.91 GB | 0.156 GB | **50.6×** |

Reuse layers are **76 %** of the all-layer control score. The gathered KV operand
`KVg [rows, k, 512]` (`953×640×512×4 ≈ 1.25 GB`) becomes the largest attention
transient under K30 — still a big peak-GB cut vs the 7.91 GB score, and **flat across
chunks** (control's grows to 7.91 GB at the last chunk). The KV is one shared head, so
the gather is `[rows, k, 512]` (not per-head) via a single flat `mx.take`, never the
`[rows, T, 512]` broadcast a naive `take_along_axis` would materialize.

### Expected seconds saved at 16K

The K25 stage timing put the reuse score at **~144–184 s** of the 265–354 s TTFT (the
T-growing term, 4 s → 26 s over the chunk sweep). At the 23.7× reuse ratio K30 leaves
**~6–8 s → saves ~138–176 s** on the reuse layers alone; the Full/Reindex (8) + SWA (2)
layers (the other 24 % of the score) shrink ~24× on top. This is a GPU-window estimate
(softmax/mask/dispatch and the non-score attention stages are unchanged); the actual
figure needs a KG-g window. K30 stacks under `prefill_lean` (K16 layer-major + K26
dense experts + K25 lean): the lean pass-cuts still apply, now over a `k`-wide (not
`T`-wide) score, and `prefill_lean_sel` arms the combination.

## 3. Implementation

- `MTPLX_DSV41_SELECTED_KEYS` (default OFF), read at use (`_resolve_selected_keys`).
  Prefill only (`rows > 1`); decode (`rows == 1`) always takes the shipped masked-full
  path, byte-identical.
- `Attention._sparse_attend_selected` — gathers window (`_window_selected_idx`) +
  selected compressed rows into `[rows, k, 512]` and runs one softmax with the value-0
  sink (reference `_k_sparse_attn` form: max includes the sink, normalize after PV,
  finite-max floor so an all-invalid row → 0, never NaN). All f32.
- `_mask_to_topk_idx` converts the `select` stage's bool `topk_mask` to ascending
  gather indices (padded `-1`) — computed **once per index source**, published on
  `SharedAttentionRuntime.selected_idx`, reused by the Reuse layers exactly like
  `topk_mask`. The gathered index set is exactly the mask's True set, so the softmax is
  over the identical keys (reassociation-level equal, never bit-identical).
- `_gather_rows` — single flat `mx.take` over `b*n` rows (invalid slots clamped to 0,
  masked out in the softmax); no `[b, s, n, d]` broadcast.
- **Merge-safe:** `_sparse_attend_oneshot` / `_sparse_attend_chunked` are untouched, so
  the change composes with a fused-softmax rewrite of those (W58). Only the
  key-selection level (what is handed to the attention math) changed.

## 4. Exactness (CPU-proven)

`tests/models/test_deepseek_v41_selected_keys.py` (14 tests, all pass, <3 GB RSS):

- Helper units: `_gather_rows` vs numpy gather; `_mask_to_topk_idx` is the True set,
  ascending, `-1`-padded, and its gathered set equals the mask; `_window_selected_idx`
  reproduces the reference window band.
- **Unit — selected-gather == masked-full one-shot** over the same window band +
  selection: max |Δ| ≤ 1e-5 (compressed and SWA-only); NaN-free on a fully-invalid row.
- **Integration — tiny 8-layer CSA model** (every mode: swa / full-r2 / reuse /
  full-r1+candidate / reindex): K30 prefill logits vs control **max |Δ| ≈ 5–6e-6,
  greedy argmax identical** across one-shot, chunk {4,7,8}, layer-major, layer-major-
  chunked.
- **Decode (rows=1) byte-identical** — never takes the K30 path.

Not bit-identical (reassociation of the softmax sum over a different key order), like
K25 `lean` — greedy-identical, exactness is not the ship bar (a task eval gates the
lossy prefill stack; K30 itself is stricter, being a re-ordered identical sum).

## 5. Arms / status

`ab_decode_env_levers.py`: `selected_keys` (`MTPLX_DSV41_SELECTED_KEYS=1`) and
`prefill_lean_sel` (`prefill_lean` + K30). Both pin all 20 lever keys. Tests:
`tests/test_deepseek_v41_ab_env_levers.py` (45 pass, includes the two new arms + the
dry-run recording). **STATUS:** IMPLEMENTED + CPU-proven, default OFF. GPU gate KG-g.
See `KERNEL_LEDGER.md` K30.
