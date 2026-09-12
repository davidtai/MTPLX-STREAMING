# W107 — Bounded / preallocated KV growth

David: **"controlling kv growth is crucial for everything."** Every KV lane must be
bounded and preallocated, so the per-token write is O(new rows) in place — no
per-token `mx.concatenate`, no per-token realloc.

This window audits every KV lane at the standard cell shape (16,384 prompt + 256
decode, `--max-kv 17408`), then adds one master switch — `MTPLX_DSV41_KV_BOUNDED`
— that preallocates **all** lanes to `max_kv` at prefill and writes each token in
place, byte-identical by construction. Default ON for the `cell16k_ring*` arms.

Code touched:
- `mtplx/models/deepseek_v41_cache.py` — the bounded lanes, per-lane counters,
  `kv_bytes_at_max_kv`.
- `scripts/deepseek_v41/ab_decode_env_levers.py` — the `MTPLX_DSV41_KV_BOUNDED`
  lever, default-ON on `cell16k_ring*`, `max_kv` stamping, receipt counters block.
- `tests/test_deepseek_v41_w107_kv_growth.py` — CPU proof.

---

## §1 — Audit: every KV lane at the cell shape

Cell dims (from the released config, `DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4`):
`num_hidden_layers=40`, `head_dim=512` (1 shared KV latent head), `sliding_window=128`,
`index_head_dim=128`, `compress_ratios ∈ {0,1,2}`, `kv_source_layer_ids=[2,8,14,20]`
with ratios `{2:2, 8:2, 14:2, 20:1}` (so **3** layers carry a ratio>1 compressor
frontier; layer 20 is ratio 1). Per-lane dtype (review MEDIUM-A, measured on the
bf16 model): **only the layer-0 window store is bf16**; every other store is **fp32**.
Layer 0's post-attention o-LoRA einsum promotes the residual to fp32 (`_o_lora_down`:
`o.astype(mx.float32)`) and it stays fp32 down the stack, so layers 1..39's window KV,
every kv-source layer's compress_kv/index_k (ratio>1 pools in fp32; ratio==1 follows
the now-fp32 residual), and the compressor frontier are all fp32. (This supersedes the
round-1 MEDIUM-2 wording, which wrongly costed layers 1+ window and the ratio==1
compress/index at bf16.)

The five lanes David named map to the cache like this. "Growth strategy today" is
what the **`cell16k_ring` arm actually runs** (W80 window ring on; W73 chunk-grow
off; the compressor frontier untouched by W80). "control" (shipped) is `_grow`
(concatenate) on all four array lanes.

| Lane | Cache field | Rows at cell | Bytes @ max_kv (bounded) | Growth **today** (cell16k_ring) | Per-token write | Alias that defeats donation |
|------|-------------|--------------|--------------------------|--------------------------------|-----------------|-----------------------------|
| **SWA ring** (window) | `LayerAttentionCache._window` (L0 bf16 / L1+ fp32) | bounded to `cap_keep`=window+mv+slack ≈ 144 (phys_cap 208) | L0 `2×208×512×2` + L1..39 `2×208×512×4` = **33.7 MB** | W80 `_WindowRing`: bounded ping-pong, `slice_update` | **in place** (O(1)/tok) | `window_all = layer_cache.window` = `buf[:, :len]` **view**, held across the forward; on Metal a live view keeps `_buf` non-uniquely-referenced → `slice_update` copies instead of donating |
| **compress store** | `LayerAttentionCache._compress_kv` (fp32, all kv-src) | `ceil(max_kv/ratio)`: 8712 (r2) / 17416 (r1) | r2 **fp32** 17.8 MB ×3 + r1 fp32 35.7 MB = **89.2 MB** | W80 `_GrowBuffer` but `maxkv` **unset** in the arm → `init_cap=256` → **geometric doubling** | in place between doublings; each doubling copies the whole prefix (O(cap), O(log T) times) | `shared.compress_kv = layer_cache.compress_kv` (view) published to the group's Reuse/Reindex/Full layers; the indexer's full-store read holds it live |
| **index store** | `LayerAttentionCache._index_k` (fp32, all kv-src) | same rows as compress | r2 **fp32** 4.5 MB ×3 + r1 fp32 8.9 MB = **22.3 MB** | same as compress (geometric doubling) | same as compress | `shared.index_k = layer_cache.index_k` (view) |
| **main latent KV** (compressor frontier) | `CompressorState.raw_kv` / `raw_score` (fp32) | `n_fed` = **every fed token** (up to max_kv) | `2×17416×512×4` ×3 = **214.0 MB** | **`_grow` (concatenate) EVERY token** — untouched by W80 | **COPY, O(n_fed)/tok** = O(T²) over the cell (≈33.5 MB copied per token per layer at T=16384) | n/a — it is a genuine `concatenate`, not a `slice_update` at all |
| **DSpark verify rows** | (write *pattern* into the four lanes above) | K+1 rows appended in one verify forward | — | one `append` of `n=K+1` rows via the lanes above | in place (ring/`_GrowBuffer` slice_update `n` rows) if the buffer has room; `concatenate` under the plain latent | inherits the source lane's alias |

**Total bounded KV @ max_kv = 359.2 MB per sequence** (window 33.7 + compress 89.2 +
index 22.3 + latent 214.0; only layer-0 window is bf16, everything else fp32) —
corrected across two review rounds from the original 286.8 MB (MEDIUM-2: ratio>1
compress/index fp32) → 320.2 MB → **359.2 MB** (MEDIUM-A: layers 1+ window and
ratio==1 compress/index are also fp32, from the fp32 residual). Storing every store
bf16 would cut it to ~179.8 MB. Negligible vs the 110 GB box budget.

**Findings.**
1. The **main latent KV** (compressor frontier) is the one lane W80 never bounded:
   it grows with a per-token `concatenate` — O(n_fed) copy per token, O(T²) over the
   cell — and is the **largest** lane (214 MB, 3 layers × 71 MB fp32). This is the
   primary target.
2. compress / index are only *amortized* O(1) in the arm (geometric doubling,
   `init_cap=256`), because the `cell16k_ring` preset never set
   `WINDOW_RING_MAXKV`; each of the O(log T) doublings copies the whole live prefix.
3. The window ring is already bounded + in-place (W80). Its remaining risk is
   **donation on Metal**: `view()` returns `buf[:, :length]` (a slice that aliases
   `_buf`), published to the model / SharedAttentionRuntime and read there; while
   that view is live, `_buf` is not uniquely referenced, so `mx.slice_update` copies
   instead of donating (the census "slice_update not donating on Metal"). W80's
   `raw_backing()` eval-fence dodges the settle-fence copy; whether the *append*
   donates on Metal still needs a GPU window (§6). On CPU, donation is observed
   (memory stays flat across appends — §5).

---

## §2 — Design: `MTPLX_DSV41_KV_BOUNDED`

One switch that bounds/preallocates every lane. Precedence: **KV_BOUNDED > WINDOW_RING
> chunk-grow.** Read at cache construction (per request, after the harness stamps the
key — never frozen at import). When on, `LayerAttentionCache` builds:

- **window** → `_WindowRing` (the W80 bounded sliding-window ring; genuine sliding
  window, so bounded to `cap_keep`, independent of `max_kv`). A prefill chunk wider
  than the base ping-pong buffers grows them transiently; the ring then **shrinks back
  to its base `phys_cap`** once decode compactions no longer need the extra width
  (review MEDIUM-1), so the decode-steady window allocation matches the §4 formula.
  Rollback of the window lane is bounded to the resident sliding window: a rollback
  that would need rows already compacted away **raises** rather than silently reading
  masked rows (review MEDIUM-3) — shallow verify/device-route rollbacks are always safe.
- **compress / index** → `_GrowBuffer(bounded_cap = ceil(max_kv/ratio) + slack)`,
  preallocated at the first (prefill) append. No geometric doubling — one allocation,
  then every write is a donated `slice_update` of just the new rows. Trim/rollback
  truncate **length-only** (they keep the preallocated buffer — review HIGH-1), so a
  rejected verify cycle never reallocates.
- **latent frontier** → `CompressorState(bounded=True, maxkv)`: `raw_kv` / `raw_score`
  become `_GrowBuffer(bounded_cap = max_kv + slack)`. This removes the last per-token
  `concatenate`. It is preallocated to `max_kv` (not shrunk to a verify-depth ring),
  keeping full history, so the **latent** lane's trim/rollback is exact to any depth
  (unlike the window ring, whose depth is bounded by its resident window — see §6).

`max_kv` reaches the cache via `MTPLX_DSV41_KV_BOUNDED_MAXKV` (the harness stamps it
from the resolved cell `max_kv` after `_apply_arm_env`, before `make_cache`; it falls
back to `MTPLX_DSV41_WINDOW_RING_MAXKV`). If neither is set, the lanes fall back to
the geometric `_GrowBuffer` — still in place, but the `kv_realloc_*` counters flag it
as not-preallocated.

**Bound enforcement.** A bounded `_GrowBuffer` never geometrically resizes: an append
that would exceed `bounded_cap` **raises `ValueError`** ("append … would exceed
preallocated cap …") — no silent growth past `max_kv`.

**Byte-identity.** Pure preallocation + in-place reorder. The `_GrowBuffer` `view()`
(`buf[:, :length]`) is byte-identical to the equivalent `_grow` (concatenate) store,
so every downstream reader (attention score/gather, indexer, pooling math,
trim/rollback, mlx_lm `state`) is unchanged. The window lane's byte-identity was
already proven by W80's drop_offset seam. Verified end-to-end in §5.

---

## §3 — Counters & env

**Env** (all read at use, not import):
- `MTPLX_DSV41_KV_BOUNDED=1|0` — master switch. Default ON on `cell16k_ring*`.
- `MTPLX_DSV41_KV_BOUNDED_MAXKV=<int>` — preallocation cap; harness-stamped from the
  cell `max_kv`, falls back to `MTPLX_DSV41_WINDOW_RING_MAXKV`.

**Per-lane engagement counters** (`kv_bounded_stats()`, reset by
`reset_kv_bounded_stats()`, surfaced in the receipt's `kv_bounded` block next to
`window_ring` / `kv_chunk_grow`). Lanes: `window`, `compress`, `index`, `latent`.
- `kv_inplace_writes_<lane>` — appends that wrote in place into an existing buffer
  (donated `slice_update` of the new rows, or a ping-pong compaction) — the
  O(new-rows) decode path.
- `kv_realloc_<lane>` — appends that **allocated** a buffer. For compress/index/latent
  in a truly preallocated arm this is the one-time prefill count (compress 1, index 1,
  latent 2 = kv+score) and **stays there** across decode and across rejected verify
  cycles (trim/rollback truncate length-only — review HIGH-1); `> 1` growing over
  *decode* means the lane was not preallocated (max_kv unset). The **window** lane's
  count can legitimately exceed 1: it counts the prefill init plus any transient
  grow for a wide prefill chunk plus the one shrink-back to base (review MEDIUM-1);
  what matters is that it stops growing once decode reaches steady state.
- `rows_<lane>` — logical rows written; `layers_bounded` — engagement; `maxkv` — the
  resolved cap; `alloc_bytes` — total buffer bytes allocated (≈ `kv_bytes_at_max_kv`
  once the window has shrunk back to base).

The stamping follows the existing `prefetch_issued_verify` / `kv_chunk_grow` /
`window_ring` pattern (`_run_arm` resets after model load; the receipt snapshots after
the run).

---

## §4 — `kv_bytes_at_max_kv` (memory-plan input for W106)

`kv_bytes_at_max_kv(config, max_kv, **overrides) -> int` (and
`kv_bytes_breakdown_at_max_kv(...)` for the per-lane dict) — a **pure** function of
the model config and `max_kv`, matching exactly what a bounded cache preallocates.

Formula (bytes, batch `B`, summed over layers `L`; `COMP_SLACK`/`LATENT_SLACK` = 8):

```
# per-layer store dtype (review MEDIUM-A, MEASURED): only layer-0 window is the
# model compute dtype (bf16); every other store is fp32 (layer 0's o-LoRA einsum
# promotes the residual to fp32 and it stays fp32 down the stack).
w_bytes(L) = model_dtype_bytes if L == 0 else store_dtype_bytes
window   (every layer)              2 · B · phys_cap · head_dim · w_bytes(L)
   phys_cap = window_size + max_verify + slack + headroom      # bounded, ⟂ max_kv
                                                               # (decode-steady; a wide
                                                               #  prefill chunk grows it
                                                               #  transiently then shrinks
                                                               #  back — review MEDIUM-1)
comp_cap = ceil(max_kv / ratio) + COMP_SLACK                   # compress/index all fp32
compress (kv_source L)              B · comp_cap · head_dim       · store_dtype_bytes
index    (kv_source L)              B · comp_cap · index_head_dim · store_dtype_bytes
latent   (kv_source L, ratio>1)     2 · B · latent_cap · head_dim · store_dtype_bytes
   latent_cap = max_kv + LATENT_SLACK
total = Σ_L (window + compress + index + latent)
```

`model_dtype_bytes` defaults to 2 (the released bf16 model); `store_dtype_bytes` to 4
(fp32). An all-fp32 run is just `model_dtype_bytes == store_dtype_bytes == 4`. The
window term is **independent of `max_kv`** — the win of the sliding-window ring: KV
does not grow with context except through the compress/index/latent lanes.

At the cell (`max_kv=17408`, default dtypes):

| Lane | Bytes |
|------|-------|
| window (L0 bf16, L1..39 fp32) | 33.7 MB |
| compress (fp32) | 89.2 MB |
| index (fp32) | 22.3 MB |
| latent (fp32) | 214.0 MB |
| **total** | **359.2 MB** |

(every store bf16 would drop the total to ~179.8 MB — a future lever, §6.)

The test asserts this formula equals the bytes the bounded cache actually allocates
(`alloc_bytes`) on a **bf16 tiny model** with the defaults (review MEDIUM-A), lane for
lane; the ab receipt stamps `kv_bounded.formula_matches_alloc` so a dtype-model drift
is caught at runtime (exact iff `kv_realloc_window == num_layers`, i.e. no transient
prefill grow — a chunked prefill grows-then-shrinks the window, MEDIUM-1, so the
cumulative `alloc_bytes` then exceeds the steady formula).

---

## §5 — Test evidence (CPU only)

`tests/test_deepseek_v41_w107_kv_growth.py` — 34 tests, `mx.set_default_device(mx.cpu)`,
tiny synthetic dims, run under `nice -n 19`, one file per process (no `-n auto`).
Original + round-1 coverage: formula==alloc, scaling, append-beyond-cap raises,
in-place/no-realloc + flat memory (vs a plain-`_grow` control), bit-identity (vs the
shipped selected path and vs the W80 ring arm), trim/rollback parity, counters,
engagement, and the round-1 HIGH-1/HIGH-2/MEDIUM-1/MEDIUM-3/LOW-1/LOW-2 regressions.

Round-2 review-fix regression tests (one per finding):
- HIGH-A `test_session_bank_deep_prefix_restore_misses_cleanly` (prefill 600, deep
  prefix restore → `_trim_cache_ref_to_prefix` returns False, cache untouched),
  `test_session_bank_shallow_prefix_restore_hits`, `test_rollback_deep_leaves_engram_untouched`.
- MEDIUM-A `test_medium_a_formula_matches_alloc_on_bf16_model` (bf16 tiny model,
  prefill 20 + decode 10 → `alloc_bytes == kv_bytes_at_max_kv` with defaults),
  `test_medium_a_all_kv_source_compress_index_fp32` (supersedes the round-1 MEDIUM-2 test).
- MEDIUM-B `test_server_hard_sets_maxkv_over_stale_env` (a stale smaller AND larger
  env is overridden; supersedes the round-1 setdefault test).
- LOW-A `test_over_cap_chunk_major_prefill_raises_at_offset_zero` (MAXKV 40, prompt
  100, chunk 32 → raises at offset 0, no chunk written).

Real pytest tails (round 2):

```
tests/test_deepseek_v41_w107_kv_growth.py ..................................  [100%]
34 passed, 2 warnings in 8.58s

# no regression in the cache/ring suites or the session-bank suites:
tests/models/test_deepseek_v41_window_ring.py ..... 15 passed in 12.59s
tests/models/test_deepseek_v41_chunk_grow.py  ..... 13 passed
tests/test_session_bank.py                    ..... 18 passed in 0.14s
tests/test_deepseek_v41_engram_state.py       ..... 6 passed in 1.01s
```
(final consolidated tails are re-captured at the end of this window in the worker report.)

---

## §6 — What still needs a GPU measurement (do NOT measure here)

A GPU benchmark window is running on this box; this window is CPU-only and never
loaded the real model. Left for a GPU window:

1. **The 25 ms/token O(T) KV claim.** The census attributes ≈25 ms/token to O(T) KV
   work (KV-append ≈11 + compress-append ≈7 + indexer-select ≈7). KV_BOUNDED targets
   the two **append** lanes (window + compress/index + the latent concatenate); the
   indexer-**select** is a *read/gather* over the full compress store, not a growth
   lane, and is untouched. Receipt placeholder: `<PENDING GPU A/B: cell16k_ring vs
   cell16k_ring + KV_BOUNDED=0 — cache_append / compress_append ms/tok, decode tok/s,
   peak GB, byte-identity>`.
2. **Metal donation.** On CPU the bounded `slice_update` donates (memory flat, §5).
   On Metal, a live `view()` slice published to the model / SharedAttentionRuntime can
   keep the buffer non-uniquely-referenced and force a copy (§1). Whether the bounded
   append donates on Metal — i.e. whether `kv_inplace_writes_*` flat also means the
   `cache_append` census stage drops from O(T) to O(1) — needs the GPU census.
   Placeholder: `<PENDING GPU: cache_append census stage O(T)? with KV_BOUNDED on>`.
3. **latent frontier lever (deferred).** The frontier is preallocated to `max_kv`
   (214 MB fp32) so the **latent** lane's trim/rollback is exact to any depth. (The
   **window** lane is a genuine ring — its rollback is bounded to the resident sliding
   window and raises past that; review MEDIUM-3. Shallow verify/device-route rollbacks
   are always safe on both.) In the served cell, rollback is only verify-depth, so the
   frontier could shrink to a verify-depth ring (older rows are already pooled into
   compress_kv and never re-read) and/or store bf16 (total 320→~186 MB). Both would
   break the whole-sequence deep-trim used by the unit tests / bare-forward path, so
   they are deferred behind their own lever.

### GPU A/B to run later (do NOT run now)

The lever is default ON for the `cell16k_ring*` arms, so the A/B is bounded-ON vs the
same arm with the switch forced OFF:

```
# control (bounded OFF) vs candidate (bounded ON, the default) on the standard cell:
python scripts/deepseek_v41/ab_decode_env_levers.py \
    --model /Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4 \
    --context-tokens 16384 --decode-tokens 256 --max-kv 17408 \
    --arms cell16k_ring cell16k_ring \
    --out <receipt.jsonl> --stage-timing
# force the control arm's switch off via env before its run:
#   MTPLX_DSV41_KV_BOUNDED=0   (cell16k_ring with KV_BOUNDED disabled)
# candidate = cell16k_ring as-is (KV_BOUNDED=1 by default, max_kv auto-stamped 17408)
```

Read on the receipt: `kv_bounded` block (compress/index/latent `kv_realloc_* ==
1/1/2` and staying there across decode + rejected verify cycles; `kv_realloc_window`
settles once the window shrinks back to base; `kv_inplace_writes_*` = decode steps;
`alloc_bytes ≈ 320.2 MB`), the `cache_append` / `compress_append` stage-timing ms/tok
delta, decode tok/s, peak GB, and `byte_identical_vs_ar` (must hold — pure prealloc).

---

## §7 — Review fixes (post-adversarial-review; verdict MERGE WITH FIXES)

Each fix is its own commit with a regression test in `test_deepseek_v41_w107_kv_growth.py`.

### Round 1 (verdict MERGE WITH FIXES — CLOSED)

- **HIGH-1** — DSpark trim/rollback reallocated the "preallocated" compress/index
  lanes every rejected cycle (`= _truncate(...)` → property setter → `_GrowBuffer.set()`
  = fresh `mx.zeros` + full copy; repro `kv_realloc 1→9`). Fixed with
  `LayerAttentionCache._lane_truncate` (length-only `truncate_to` for `_GrowBuffer`
  lanes in trim & rollback).
- **HIGH-2** — the served path never bounded the lanes (only the ab harness stamped
  `MTPLX_DSV41_KV_BOUNDED_MAXKV`). The server now plumbs it from
  `max_live_kv_tokens` at KV-window setup (via `os.environ.setdefault`, mirroring the
  adjacent `MTPLX_CONTEXT_WINDOW_TOKENS`) and registers both keys in
  `_DSV41_LEVER_ENV_KEYS`.
- **MEDIUM-1** — the window ring's `phys_cap` only ever grew, so a chunked prefill
  left it inflated (~114 MB, disagreeing with the formula's 17 MB). It now shrinks
  back to its base `phys_cap` after prefill; the §4 window term is the decode-steady
  bound.
- **MEDIUM-2** — compress/index on the ratio>1 source layers are **fp32** (the
  formula defaulted 2 bytes). Per-layer dtype fixed; cell total 286.8 → **320.2 MB**;
  §1 dtype wording corrected. No append-site cast added (would change bytes vs shipped).
- **MEDIUM-3** — window-ring rollback across a compaction silently masked in-window
  rows. `truncate_to_length` now raises when the rollback reaches below the drop
  frontier; the "arbitrary depth" claim is dropped for the window lane (the latent
  lane keeps full history and stays exact to any depth).
- **LOW-1** — an over-cap forward left the cache half-updated. A whole-cache
  `assert_can_admit(n)` pre-check runs at the top of `_forward_span` /
  `_forward_layer_major`, failing before any lane is written.
- **LOW-2** — `_GrowBuffer.truncate_to(0)` dropped the prealloc; it now keeps the
  buffer (length-only).

### Round 2 (re-review; HIGH-1/2, MEDIUM-1/3, LOW-2 confirmed CLOSED)

- **HIGH-A** — the session-bank prefix restore (`_trim_cache_ref_to_prefix` →
  `entry.trim`) hit MEDIUM-3's window-ring raise whenever the divergence exceeded the
  ring's recoverable depth, and the single-request lane had no guard — so ordinary
  regenerate/edited turns FAILED instead of cold-prefilling. `LayerAttentionCache.trim`
  now returns `0` with **no mutation** when the ring cannot recover
  (`_WindowRing.can_truncate_to_length`), so the bank's "trim ≠ delta" miss contract
  fires (NO_SNAPSHOT_COVERAGE → cold prefill); the raise stays for direct
  `truncate_to_length` callers. `rollback` reorders the ring truncate before the engram
  trim so a failed deep rollback leaves nothing half-rewound.
- **MEDIUM-A** — the formula was still ~33% under on a bf16 model. **Measured**: only
  the layer-0 window store is bf16; every other store is fp32 (layer 0's o-LoRA einsum
  promotes the residual to fp32). New signature `model_dtype_bytes` (default 2) +
  `store_dtype_bytes` (default 4); defaults now match a bf16 model exactly. Cell total
  320.2 → **359.2 MB**. Added the ab receipt gate
  `kv_bounded.formula_matches_alloc = (alloc_bytes == kv_bytes_at_max_kv)` +
  `kv_bytes_formula`. Supersedes round-1 MEDIUM-2's ratio-based dtype model.
- **MEDIUM-B** — the round-1 server plumbing used `setdefault`, letting a stale
  `MTPLX_DSV41_KV_BOUNDED_MAXKV` override `max_live_kv_tokens`. Now hard-set via
  `_plumb_kv_bounded_maxkv` (mirrors `MTPLX_CONTEXT_WINDOW_TOKENS`).
- **LOW-A** — `assert_can_admit` was per-span, so a chunk-major overflow raised at
  span 2 with a partial prefix. Hoisted to `Model.__call__` (whole prompt) before the
  chunk dispatch, so it fails at offset 0.

---

## §8 — GPU window-42 evidence + round-3 fixes

First GPU evidence (window 42, 16K cell, `--stage-timing`; receipts
`.../receipts/gpu-windows/window-42/ar-ring-{ref,v2}.json`, code `0cc4dd7d3` =
round-1 fixes, NOT round-2). **The per-token O(T) KV stages did not move**:
KV-append 11.1 / compress-append 3.3 / indexer-select 6.9 ms/tok (ref) and 11.9 /
4.3 / 7.4 (v2) — same as window 41 (12.0 / 8.3 / 7.8) when the lever did not exist.
The lever preallocated the lanes but the copy stayed.

Receipt `kv_bounded`: `layers_bounded` 80, `maxkv` 17408, `alloc_bytes`
1,422,102,528, `kv_realloc_window` 400, `kv_realloc_compress`/`index` 8,
`kv_realloc_latent` 12, `kv_inplace_writes_window` 21440. `window_ring`: `appends`
21840, `reallocs` 160, `drops` 1520.

**(a) 80 layers, 1.42 GB (≈4× the 359 MB formula).** `--stage-timing` builds a FRESH
cache TWICE per arm — the untimed headline pass (`_generate`) and the fenced pass
(`_stage_timing_pass`), each `make_cache` → 40 `LayerAttentionCache`. The counters are
process-global cumulative, so `layers_bounded` = 2×40 = 80 and `appends` 21840 = 2
passes × 40 layers × ~273 appends (16 prefill chunks + 256 decode + init). It is NOT
the MTP/draft layers (this is the AR arm). `alloc_bytes` 1.42 GB = 2 passes × (~359 MB
steady + ~350 MB window transient churn): the 16,384-token prompt is fed in 16
~1024-token chunks, each wider than the base window `phys_cap` 208, so the ring grows
to chunk width and re-allocs the freed ping-pong slot, and every such transient buffer
accumulates in the cumulative counter. The steady RESIDENT is 359 MB (the formula);
`alloc_bytes` is cumulative-ever across both passes + transients, so
`formula_matches_alloc` correctly reads False here (`kv_realloc_window` 400 ≫
num_layers) — the tripwire working, not a formula error.

**(b) 400 window reallocs (10/layer).** = 2 passes × 5/layer. The 1024-token prefill
chunks exceed the base `phys_cap` 208, so the first chunks each trigger a compaction
that grows the ring, and the following compaction re-allocs the other ping-pong slot —
~5 buffer allocations/layer/pass over the 16-chunk prefill. (`window_ring.reallocs`
160 = 2×2/layer counts only the size-change grows; `kv_realloc_window` 400 also counts
the post-grow ping-pong re-alloc.) This receipt is round-1 (grow-only); round-2
MEDIUM-1 adds a shrink-back at the prefill→decode transition — orthogonal to (c). The
realloc churn is PREFILL-time; it does not touch the per-decode-token `cache_append`.

**(c) KV-append ms/tok unchanged — root cause.** `mx.slice_update` on Metal does NOT
donate into the preallocated buffer: it is a FUNCTIONAL op that allocates a fresh
output plane and copies the whole resident window every append (O(resident)). The
round-1/2 CPU "donation proof" measured cumulative memory flatness (no growth across N
appends — true, each copy's transient is freed) but NOT the per-append copy COST.
`scripts/deepseek_v41/kv_donation_probe.py` measures the per-append ms-vs-T slope +
memory delta and shows, **even on CPU**, `slice_update` is O(T) COPY (ms ×3.8 over N
2048→16384, peak +1 buffer plane) while in-place `buf[:, r:r+n] = new` DONATES (ms
flat ×1.0, peak ≈ 0) when the buffer is uniquely referenced and COPIES safely
(byte-identical, held view untouched) when a live `view()` alias is held. **Both the
primitive AND the read-time alias matter.**

**Round-3 fix.** Switch the BOUNDED lanes to the donating in-place `__setitem__`
(byte-identical — the model bit-identity tests + `test_inplace_row_write_byte_identical_to_slice_update`
+ `test_model_bit_identical_inplace_vs_slice_update`); keep `slice_update` on the
frozen `window_ring`/chunk-grow lanes so the control's basis is unchanged (finding 2).
`MTPLX_DSV41_KV_INPLACE_WRITE` overrides for the A/B (arm `cell16k_ring_bounded_copy`
= bounded + forced `slice_update`).

### GPU probe run plan (orchestrator, in a lock gap; do NOT run here)

```
# window/latent lane width (512), the real store dtype (fp32) at the cell:
python scripts/deepseek_v41/kv_donation_probe.py --gpu \
    --sizes 16384,16896,17408 --dim 512 --dtype fp32 --reps 50 --json <out-512.json>
# index lane width (128):
python scripts/deepseek_v41/kv_donation_probe.py --gpu \
    --sizes 16384,16896,17408 --dim 128 --dtype fp32 --reps 50 --json <out-128.json>
```

Read the verdict (DONATE = ms flat & peak ≪ buffer; COPY(O(T)) otherwise). Decisive rows:
- `slice_update hold_view=False` — if COPY on Metal, confirms (c).
- `setitem hold_view=False` — if DONATE on Metal, the fix works when the buffer is unique.
- `setitem hold_view=True` — if COPY, the read-time `view()` alias must ALSO be removed
  (architectural follow-up: no `window_all` / `shared.compress_kv` slice alive at the
  next append; append before read, or read via `raw_backing`).
- `metal_kernel` — the fallback if no MLX primitive donates on Metal.

### Expected sign once donation works (finding 3)

Window 42's v2 was 1.996 vs ref 2.167 tok/s (−8%) with BOTH arms carrying kv_bounded —
the lever added prealloc cost without removing the copy (c), so it can only be
neutral-to-negative until donation works. Once the donating write lands AND the buffer
is unique at append time, the expected sign is **positive**: `cache_append` (≈11.9) +
`compress_append` (≈4.3) ms/tok fall toward O(new-rows) (~0), lifting decode tok/s by
the fraction those stages are of the ~460 ms/token step. **The receipt line that proves
it:** on `cell16k_ring_bounded` vs `cell16k_ring_bounded_copy`,
`stage_timing.decode_breakdown` `cache_append`/`compress_append` ms/tok DOWN **and**
`kv_bounded.kv_realloc_*` flat (one prealloc per layer, no per-token realloc) **and**
`byte_identical_vs_ar` holds. `indexer-select` (≈6.9) is a READ/gather over the full
compress store, not an append — untouched by the write primitive (a separate lever).

### Arm taxonomy (controls vs candidates)

- **CONTROL** (frozen, windows 39–42): `cell16k_ring` — no `kv_bounded`, no
  `kv_inplace_write` (its `window_ring` lane uses `slice_update`). Its SET env equals
  the window-39 basis (enforced by `test_control_arm_frozen_matches_window39`).
- **Bounded candidates**: `cell16k_ring_bounded` (bounded + donating in-place),
  `cell16k_ring_bounded_copy` (bounded + forced `slice_update` — the write-primitive
  A/B pair).
- **Other candidates** (each isolates its OWN lever vs the control):
  `cell16k_ring_v2` / `_draft` / `_pinned` / `_stable` / `_pool` / `_switch` /
  `_prefetch` / `_v2_draft` — round-2 stripped `kv_bounded` from these so they stay
  clean single-lever isolations (enforced by `test_cell16k_ring_composite_arms`).
- The merge worker's `int/w97f-lanes` commit `0199d92db` added `kv_bounded` to seven
  older attention arms; those live on `int`, not this branch — flag for the coordinator
  to strip if they are meant as clean isolations.
