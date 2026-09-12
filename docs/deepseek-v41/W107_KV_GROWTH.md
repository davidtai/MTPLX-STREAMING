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
frontier; layer 20 is ratio 1). Post-RoPE stores are bf16; the compressor frontier
is **fp32** (the reference projects `wkv`/`wgate` in fp32).

The five lanes David named map to the cache like this. "Growth strategy today" is
what the **`cell16k_ring` arm actually runs** (W80 window ring on; W73 chunk-grow
off; the compressor frontier untouched by W80). "control" (shipped) is `_grow`
(concatenate) on all four array lanes.

| Lane | Cache field | Rows at cell | Bytes @ max_kv (bounded) | Growth **today** (cell16k_ring) | Per-token write | Alias that defeats donation |
|------|-------------|--------------|--------------------------|--------------------------------|-----------------|-----------------------------|
| **SWA ring** (window) | `LayerAttentionCache._window` | bounded to `cap_keep`=window+mv+slack ≈ 144 (phys_cap 208) | `2×208×512×2` ×40 = **17.0 MB** | W80 `_WindowRing`: bounded ping-pong, `slice_update` | **in place** (O(1)/tok) | `window_all = layer_cache.window` = `buf[:, :len]` **view**, held across the forward; on Metal a live view keeps `_buf` non-uniquely-referenced → `slice_update` copies instead of donating |
| **compress store** | `LayerAttentionCache._compress_kv` | `ceil(max_kv/ratio)`: 8712 (r2) / 17416 (r1) | ratio2 8.9 MB ×3 + r1 17.8 MB = **44.6 MB** | W80 `_GrowBuffer` but `maxkv` **unset** in the arm → `init_cap=256` → **geometric doubling** | in place between doublings; each doubling copies the whole prefix (O(cap), O(log T) times) | `shared.compress_kv = layer_cache.compress_kv` (view) published to the group's Reuse/Reindex/Full layers; the indexer's full-store read holds it live |
| **index store** | `LayerAttentionCache._index_k` | same rows as compress | ratio2 2.2 MB ×3 + r1 4.5 MB = **11.1 MB** | same as compress (geometric doubling) | same as compress | `shared.index_k = layer_cache.index_k` (view) |
| **main latent KV** (compressor frontier) | `CompressorState.raw_kv` / `raw_score` (fp32) | `n_fed` = **every fed token** (up to max_kv) | `2×17416×512×4` ×3 = **214.0 MB** | **`_grow` (concatenate) EVERY token** — untouched by W80 | **COPY, O(n_fed)/tok** = O(T²) over the cell (≈33.5 MB copied per token per layer at T=16384) | n/a — it is a genuine `concatenate`, not a `slice_update` at all |
| **DSpark verify rows** | (write *pattern* into the four lanes above) | K+1 rows appended in one verify forward | — | one `append` of `n=K+1` rows via the lanes above | in place (ring/`_GrowBuffer` slice_update `n` rows) if the buffer has room; `concatenate` under the plain latent | inherits the source lane's alias |

**Total bounded KV @ max_kv = 286.8 MB per sequence** (fp32 latent) / 179.8 MB if
the latent frontier were stored bf16 — negligible vs the 110 GB box budget.

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
  window, so bounded to `cap_keep`, independent of `max_kv`).
- **compress / index** → `_GrowBuffer(bounded_cap = ceil(max_kv/ratio) + slack)`,
  preallocated at the first (prefill) append. No geometric doubling — one allocation,
  then every write is a donated `slice_update` of just the new rows.
- **latent frontier** → `CompressorState(bounded=True, maxkv)`: `raw_kv` / `raw_score`
  become `_GrowBuffer(bounded_cap = max_kv + slack)`. This removes the last per-token
  `concatenate`. It is preallocated to `max_kv` (not shrunk to a verify-depth ring)
  so arbitrary-depth trim/rollback stays exact — see §6.

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
- `kv_realloc_<lane>` — appends that **allocated** a buffer. In a truly
  preallocated-bounded arm this is the one-time prefill count (window 1, compress 1,
  index 1, latent 2 = kv+score) and **stays there**; `> 1` growing over the cell means
  the lane was not preallocated (max_kv unset, or a prefill chunk wider than the cap).
- `rows_<lane>` — logical rows written; `layers_bounded` — engagement; `maxkv` — the
  resolved cap; `alloc_bytes` — total buffer bytes allocated (≈ `kv_bytes_at_max_kv`).

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
window   (every layer)              2 · B · phys_cap · head_dim · window_dtype_bytes
   phys_cap = window_size + max_verify + slack + headroom      # bounded, ⟂ max_kv
compress (kv_source L)              B · comp_cap · head_dim       · compress_dtype_bytes
index    (kv_source L)              B · comp_cap · index_head_dim · index_dtype_bytes
   comp_cap = ceil(max_kv / ratio) + COMP_SLACK
latent   (kv_source L, ratio>1)     2 · B · latent_cap · head_dim · latent_dtype_bytes
   latent_cap = max_kv + LATENT_SLACK
total = Σ_L (window + compress + index + latent)
```

`*_dtype_bytes` default to the runtime dtypes (bf16 stores = 2, fp32 latent = 4); pass
measured widths for a specific build. The window term is **independent of `max_kv`** —
the win of the sliding-window ring: KV does not grow with context except through the
compress/index/latent lanes.

At the cell (`max_kv=17408`, default dtypes):

| Lane | Bytes |
|------|-------|
| window | 17.0 MB |
| compress | 44.6 MB |
| index | 11.1 MB |
| latent (fp32) | 214.0 MB |
| **total** | **286.8 MB** |

(latent stored bf16 would drop the total to 179.8 MB — a future lever, §6.)

The test asserts this formula equals the bytes the bounded cache actually allocates
(`alloc_bytes`), lane for lane (§5).

---

## §5 — Test evidence (CPU only)

`tests/test_deepseek_v41_w107_kv_growth.py` — 14 tests, `mx.set_default_device(mx.cpu)`,
tiny synthetic dims, run under `nice -n 19`, one file per process (no `-n auto`):

- `test_kv_bytes_formula_matches_preallocation` — `alloc_bytes` == `kv_bytes_at_max_kv`
  == summed live raw-backing bytes.
- `test_kv_bytes_breakdown_scales_with_max_kv` — compress/index/latent scale with
  max_kv; window does not; dtype widths scale correctly.
- `test_append_beyond_max_kv_raises_{grow_buffer,latent_frontier,via_cache}` — clean
  `ValueError` past the cap (no silent growth).
- `test_inplace_stable_no_realloc_flat_memory` — across 64 decode steps: `kv_realloc_*`
  flat (data pointer stable), raw-backing shapes stable, `get_peak_memory` flat.
- `test_plain_grow_memory_grows_unlike_bounded` — control: the shipped `_grow` frontier's
  active memory grows with N; the bounded one does not.
- `test_model_bounded_bit_identical_vs_legacy_selected_path` — a tiny-model
  prefill+decode (T≈200, ring drops) is **bit-identical** bounded vs shipped.
- `test_model_bounded_bit_identical_vs_window_ring` — bounded vs the W80 ring arm is
  bit-identical (isolates the prealloc/in-place changes).
- `test_bounded_trim_rollback_parity`, `test_bounded_trim_parity` — trim/rollback of
  every lane (incl. the frontier) matches the shipped path.
- `test_counters_increment_and_reset`, `test_bounded_engages_when_env_set_after_import`,
  `test_bounded_maxkv_falls_back_to_window_ring_maxkv`.

Real pytest tails (this window):

```
tests/test_deepseek_v41_w107_kv_growth.py ............                    [100%]
14 passed, 2 warnings in 8.52s

# no regression in the cache/ring suites or the ab-harness suite:
tests/models/test_deepseek_v41_chunk_grow.py ...... 13 passed, 2 warnings in 1.43s
tests/models/test_deepseek_v41_window_ring.py ..... 15 passed, 2 warnings in 13.46s
tests/test_deepseek_v41_w93_lane_b_ring.py ........ 8 passed, 2 warnings in 1.03s
tests/test_deepseek_v41_ab_env_levers.py .......... 65 passed, 2 warnings in 0.78s
```

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
   (214 MB fp32) to keep arbitrary-depth trim/rollback exact. In the served cell,
   rollback is only verify-depth, so the frontier could shrink to a verify-depth ring
   (older rows are already pooled into compress_kv and never re-read) and/or store
   bf16 (total 287→180 MB). Both would break the whole-sequence deep-trim used by the
   unit tests / bare-forward path, so they are deferred behind their own lever.

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

Read on the receipt: `kv_bounded` block (`kv_realloc_* == 1/1/1/2` and staying there;
`kv_inplace_writes_*` = decode steps; `alloc_bytes ≈ 286.8 MB`), the `cache_append` /
`compress_append` stage-timing ms/tok delta, decode tok/s, peak GB, and
`byte_identical_vs_ar` (must hold — pure prealloc).
