# W76 — the 16K decode "attention proper" is a constant, not an O(T) op

**Question (from window 29b step 1, real model, arm `prefill_lean_sel`).** With
K30 selected keys on, the per-layer decode attention at T≈16.4K measured, after
peeling `cache_append` / `compress_append` / `select`, a residual **"attention
proper" of ~7.3 ms/layer that is the SAME across all four CSA modes** —
`attn.full` 12.4 (−1.18 append −2.68 compress −1.18 select → 7.36), `attn.reindex`
9.95 (−1.28 −1.40 → 7.27), `attn.reuse` 8.40 (−1.18 −0.002 → 7.22), and crucially
`attn.swa_only` 8.5 (−1.24 → 7.26). SWA-only layers hold a 128-key window and no
indexer; their attention-proper work is T-independent by construction, yet they
too cost ~7.3 ms. W73 ruled the score out (K30-gathered, T-independent). The task:
find the O(T) op that remains.

**Answer.** On the CPU double there is **no additional per-op O(T) in attention-
proper** — for *any* CSA mode. Bisected op-by-op at the real head/index widths and
in the real integrated decode, attention-proper (qkv projection, selected-key
gather, QK/softmax/PV over `k = window + index_topk`, output projection) is **flat
in T**. The only per-op O(T) decode-attention costs are the ones W73 already named:
the KV **appends** (Causes 1+2, fixed byte-identically by
`MTPLX_DSV41_KV_CHUNK_GROW`, amortized O(1)) and the indexer **`select`** (Cause 3,
inherent, correctly fenced). The measured ~7.3 ms/layer "attention proper" is a
**T-independent per-layer constant inflated on GPU by memory-residency pressure at
16K** (W73 Cause 5), *not* a per-op O(T). **The cross-mode uniformity is the proof:
a per-op O(T) would differ by mode (index-source layers do strictly more work); a
memory-pressure-inflated fixed cost is mode-independent — exactly what is
measured.**

One genuine, byte-identical defect turned up on the way (a stage-timing
attribution leak of the K30 argsort), fixed under `MTPLX_DSV41_SELECT_FENCE`.

> **Reproducibility note.** Everything below is `mx.set_default_device(mx.cpu)`,
> tiny/real fake config, real T cache rows filled directly (no 16K prefill — a real
> prefill materialises the O(N²) indexer score and trips the memory guard). The CPU
> double establishes the **scaling shape** (per-op O(T) vs flat), not the GPU
> constant. It cannot reproduce the GPU memory-pressure multiplier — which is the
> whole point of the finding.

## Method

Built a CPU fake-config decode-attention microbench per CSA mode: a single real
`Attention` layer at the released head/index widths (`head_dim=512`,
`num_attention_heads=64`, `window_size=128`, `index_topk=512`, `index_head_dim=128`),
a `LayerAttentionCache` filled directly to T = 1024 / 4096 / 16384 rows, and one
decode `_attend`. Each op is `mx.eval`-fenced and timed independently; the window
append is pre-materialised before the score is timed (exactly what the
`cache_append` stage fence does in the real census). Cross-checked against the real
integrated decode (tiny dims, real prefill to T≤2048) reading the decode stage
census.

## Scaling tables

### SWA-only, real widths (hd=512, H=64, win=128) — op-by-op (ms/decode step)

| T | whole `_attend` | qkv_proj | **append+read** | gather | score(qk+sm+pv) | out_proj |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 11.20 | 4.42 | 0.146 | 0.051 | 0.084 | 6.47 |
| 4096 | 12.32 | 4.45 | 0.533 | 0.047 | 0.079 | 6.34 |
| 16384 | 13.25 | 4.52 | **2.083** | 0.053 | 0.080 | 6.84 |
| **16K/1K** | 1.18× | 1.02× | **14.2×** | 1.05× | 0.94× | 1.06× |

The whole step grows 1.18×, and **the entire growth is the `append+read`
concatenate** (14.2×, W73 Cause 1). qkv_proj, gather, score and out_proj are flat.
Attention-proper here is a ~11 ms constant dominated by the two big GEMMs
(qkv_proj's `wq_b`, out_proj's grouped o-LoRA + `wo_b`), *not* by anything O(T).

### Reuse, real widths — the compress-gather + score over k=640 (ms/step)

| T | whole `_attend` | window gather | compress gather | score(k=640) |
|---:|---:|---:|---:|---:|
| 1024 | 11.85 | 0.050 | 0.060 | 0.329 |
| 4096 | 11.76 | 0.032 | 0.042 | 0.189 |
| 16384 | 13.34 | 0.029 | 0.041 | 0.180 |
| **16K/1K** | 1.13× | flat | flat | flat |

The compressed-row gather (512 of T rows) and the score over `k=640` are flat; the
1.13× on the whole step is again just the window append.

### Integrated real decode (tiny dims, real prefill), per-mode census (ms/token)

attention-**proper** = `attn.<mode>` − (`cache_append` + `compress_append` + `select`):

| mode | proper@512 | proper@2048 | scale (4×T) | (T-dependent: `select`@512 → @2048) |
|---|---:|---:|---:|---|
| swa_only | 0.476 | 0.461 | **0.97×** | — |
| full | 0.531 | 0.507 | **0.95×** | 0.330 → 0.430 (grows: Cause 3) |
| reindex | 0.261 | 0.243 | **0.93×** | 0.198 → 0.235 (grows: Cause 3) |
| reuse | 0.774 | 0.773 | **1.00×** | 0.003 (reads source mask) |

Attention-proper is flat for **every** mode; the T-dependence lives in `select`
(the indexer, Cause 3) and the appends — both peeled out and both already in W73's
ledger.

### The append is genuinely amortised by chunk-grow (O(T) counter)

Window-lane rows copied over 300 sequential decode appends from T=1024:

| backing | rows copied | shape |
|---|---:|---|
| plain `_grow` (concatenate) | 4,960,350 | ~O(N²) |
| `MTPLX_DSV41_KV_CHUNK_GROW` | 33,068 | ~O(N) |

## The ops and the lines

1. **KV append — the sole dominant CPU O(T)** (W73 Causes 1+2).
   `deepseek_v41_cache.py::_grow` (`mx.concatenate` of the whole store, L106) via
   `append_window` / `append_compress` / `append_index_k`. **Already fixed
   byte-identically** by `MTPLX_DSV41_KV_CHUNK_GROW` (amortized O(1)) and **already
   wired into `cell16k`** through the `prefill_lean_sel_chunk` prefill lane. This is
   the fixable per-op O(T); the CPU double confirms it is the only one that matters.

2. **Indexer select — inherent O(n_comp)** (W73 Cause 3). `Indexer.select` einsum
   over `n_comp` + `_topk_rows`' `mx.sort` (`deepseek_v41.py` L~527/L558), on the 8
   index-source layers. Correctly fenced into `attn.<mode>.select`. Not removable.

3. **NEW (W76) — the K30 `selected_idx` argsort leaks out of the `select` fence.**
   `deepseek_v41.py`, in `_compressed`'s select bracket:
   `shared.selected_idx = _mask_to_topk_idx(mask, …)` (an `mx.argsort` over
   `n_comp`, O(n_comp log n_comp)) is published but the bracket fences only `mask`
   (`_sp.add(mask); _sd.add(mask)`). The argsort therefore stays lazy and is forced
   later by the first downstream compress-gather **inside the `score` stage** —
   mis-charging that index-source O(T) cost into decode attention-**proper**. On the
   CPU double at real widths this leak is `_mask_to_topk_idx` ≈ 0.043 → 0.137 ms
   (1K → 16K, 3.2×); small, but it is the one stray O(T) that lands in
   attention-proper's bracket, so it is exactly the kind of thing that made the
   7.3 ms look suspicious in stage timing. **Fix (byte-identical):** under
   `MTPLX_DSV41_SELECT_FENCE`, also fence `shared.selected_idx` in the select
   bracket so the argsort is timed in `select` where it belongs. The fenced array is
   the *same object* the gather would force and the extra `add` is a no-op outside a
   recording decode session, so the token / cache / logits are byte-identical on or
   off — it changes only the census, never the output. (Not a production speedup, so
   **not** added to `cell16k`.)

## Why 7.3 ms is a constant, not O(T)

At real widths attention-proper on the CPU double is ~11 ms and **flat in T**,
dominated by the qkv and output GEMMs — fixed-shape work independent of the cache
length. On GPU the same fixed work is ~1.7 ms at 1K. It rises to ~7.3 ms at 16K
**uniformly across all four modes**. A per-op O(T) cannot produce that uniformity
(Full does strictly more per-op work than SWA-only). A per-layer fixed cost
inflated by system memory pressure does: at 16K the resident cache is ~40×16 MB
(windows) + the compressed/index lanes ≈ 0.7 GB+, each token's plain-`_grow`
concatenate reallocates a fresh full-length store per layer (~640 MB alloc/free
churn/token), and the decode forward's evals materialise the full cache — near the
85 GB / 100 GiB wired ceiling this thrashes the Metal allocator and evicts streamed
expert pages (W73 Cause 5 / honesty caveat). That is a memory-residency O(cache-size)
effect on a fixed-FLOP op, invisible on the CPU double by construction.

**Lever:** the memory-pressure amplifier is reduced by the append fix already in
`cell16k` (`MTPLX_DSV41_KV_CHUNK_GROW` removes the per-token full-store realloc, cut
the append rows-copied ~150×). There is **no additional per-op O(T)** to remove from
attention-proper. Confirming the residual on the real model needs a GPU window (see
below); it cannot be reproduced on the CPU double.

## Env keys / arms (for the `cell16k` composite preset)

| Key | Class | What it does | In cell16k? |
|---|---|---|---|
| `MTPLX_DSV41_KV_CHUNK_GROW=1` | byte-identical | append O(T)→amortized O(1) (W73 Causes 1+2) — **the** fixable decode-attn O(T) | **yes** (via `prefill_lean_sel_chunk`) |
| `MTPLX_DSV41_SELECT_FENCE=1` | byte-identical (census-only) | charges the K30 argsort to `select`, not `score` (W76 attribution fix) | no (measurement-only) |

New arms in `scripts/deepseek_v41/ab_decode_env_levers.py` (append-only):
`select_fence` (isolation: selected keys + fence), `prefill_lean_sel_fence`
(= the measured `prefill_lean_sel` arm + fence — the direct census A/B that shows
how much of that arm's per-mode attention-proper was the mis-attributed argsort).

## How to confirm on the real model (GPU window)

```
ab_decode_env_levers.py --context-tokens 16384 --decode-tokens 256 --stage-timing \
    --arms prefill_lean_sel,prefill_lean_sel_fence,prefill_lean_sel_chunk --out <receipt.jsonl>
```
- `prefill_lean_sel` vs `prefill_lean_sel_fence`: the per-mode `attn.<mode>` decode
  proper should be **unchanged in total**, with the argsort moving from the score
  sub-stage into `select` on the index-source modes (attribution only).
- `prefill_lean_sel` vs `prefill_lean_sel_chunk`: the append lanes collapse; if the
  ~7.3 ms attention-proper *also* drops, that confirms it was memory-pressure
  (Cause 5) riding the append reallocation churn, not a per-op O(T).

## Evidence (tests, CPU-only)

`tests/models/test_deepseek_v41_decode_attn_16k.py` (5 tests, all pass):
- `test_swa_only_operand_T_independent`, `test_reuse_operand_T_independent` — the
  gathered attention operand `k` is identical at T=1024 and T=16384 (SWA = window;
  Reuse = window + index_topk): the structural, non-flaky proof that no
  attention-proper op is O(T).
- `test_swa_attention_proper_wall_flat_in_T` — SWA-only attention-proper wall at
  16K stays under 5× its 1K wall on the CPU double (a true O(T) op would be ~16×).
- `test_select_fence_byte_identical` — decode logits byte-identical with
  `MTPLX_DSV41_SELECT_FENCE` on vs off, even under an active decode stage-timing
  session (so the fence actually fires).
- `test_select_fence_charges_argsort_to_select` — the fence records a non-empty
  `select` decode sub-stage on the index-source modes and does not change the
  selection.
