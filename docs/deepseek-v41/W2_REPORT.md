# W2 — Engram conditional-memory runtime (DeepSeek-V4.1-Flash)

Branch `feat/deepseek-v41-w2`. Serves the Engram bank the way the Qwen3.8 Flash-Next
resident n-gram table is served: a **bounded LRU of resident rows in unified memory**,
misses read **positionally** from the on-disk affine row bank, **eviction changes residency
only, never values**.

Deliverables:
- `mtplx/ngram_row_cache.py` — generic bounded-row LRU + positional miss reader + MLX packed-row gather.
- `mtplx/engram_bank.py` — `EngramBank` now sits on the generic cache (record decode kept).
- `mtplx/engram_v41.py` — `NgramHashState` (streaming, rollback), compressed-token-map builder, `EngramV41` hook.
- `tests/test_ngram_row_cache.py`, `tests/test_engram_v41.py`.

---

## 1. Lifted from `qwen4_ngram.py` (closed PR #368 branch) vs new

Reference read from `origin/port/qwen38-flash-next-resident-q4`:
`mtplx/qwen4_ngram.py` (3899 lines), `mtplx/models/qwen4_ngram_mlx.py` (163),
`mtplx/native/ngram_cache_mlx.cpp` (92), `docs/qwen4-resident-ngram-cache.md`.

**Reusable core lifted into `mtplx/ngram_row_cache.py`** (algorithm/structure, reimplemented
synchronously — not a copy):

| Concept | Source (qwen4_ngram.py unless noted) | Where it lands |
|---|---|---|
| byte-budget → slot count (`cache_limit // row_bytes`) | `plan_ngram_cache` L2053 | `NGramRowCache.__init__` slot_count |
| row-id-keyed routes + LRU touch/evict-oldest | `_PackedCacheIndex` L2222–2416; `_choose_slots`/`oldest_unpinned_slots` L3391/L2318 | `_lru` OrderedDict + `_alloc_slot` |
| positional `preadv` read loop | `_DescriptorReader.read_into` L2690–2698 | `FileRowReader.read_run` |
| contiguous-run miss coalescing | `NGramRowCache._groups` L3410–3433 | `_contiguous_runs` + `gather_bytes` |
| uint8 gather → byte-range view → `mx.dequantize(mode="affine")` | `AffineQ4NGramRows.__call__` `qwen4_ngram_mlx.py` L46–90 | `RowGeometry.dequantize` |
| byte-budget / residency stats surface | `NGramRowCache` props L3157–3191 | `NGramRowCache.stats`/`resident_bytes` |

**New here (deliberately NOT lifted):**
- Synchronous, single-thread design: dropped the whole async plane — `ThreadPoolExecutor`,
  `NGramAcquireFuture`/`NGramLease` (L2851–2990), `SlotTicket` pins/generations, poisoning,
  page-cache-policy install/rollback, `_VerifiedNGramArtifact` sha256 verification. None of it
  is needed for a synchronous CPU gather and it would have been ~3000 lines of ballast.
- Generic `RowGeometry` for any `(values, bits, group_size)` — the Qwen module hard-codes
  affine-q4-g32; ours also carries engram's affine-q8-g64 (272-byte record).
- Plain **numpy CPU arena** instead of the Metal shared-buffer arena (`ngram_cache_mlx.cpp`
  `allocate_metal_u8_2d`). The `.cpp` is a GPU allocator; W2 is CPU-only (no GPU lock), and the
  pure path is not the bottleneck (§3), so it was not ported.
- **Request larger than the cache**: rows are copied out as their run is read, then may be
  evicted — the Qwen path raises `NGramCacheFull` instead. This preserves `EngramBank.gather`'s
  existing contract (gather 20 rows through a 4-slot budget).
- `MTPLX_ENGRAM_CACHE_LIMIT` / ctor byte-budget path (`cache_bytes_from_env`, Pydantic
  `ByteSize`) in place of the runtime memory planner (`plan_production_ngram_cache` L2120).

`mtplx/engram_bank.py`: the ad-hoc `OrderedDict` LRU + `_read_record` preadv loop were replaced
by `FileRowReader` + `NGramRowCache`; the **record decode** (`_bf16_bits_to_f32`,
`dequantize_rows` = `level*scale+bias` per group, the `gather` split into u32/u16/u16) is kept
verbatim. Public API (`open`, `gather`, `dequantize_rows`, `cache_used_bytes`, `close`, `__len__`)
is unchanged — the pre-existing `tests/test_convert_deepseek_v41_engram.py` passes untouched.

`mtplx/engram_v41.py` is **new**, ported from the model source, not the Qwen module:
`build_compressed_token_map` and `NgramHashState` mirror
`~/models/DeepSeek-V4.1-Flash-src/inference/engram.py` (the streaming history + `trim` rollback
are new); `EngramV41.__call__` mirrors `Engram.forward` (`model.py` L350–364).

---

## 2. Hook signature

```python
class EngramV41(nn.Module):
    def __call__(self, hidden_states: mx.array, token_ids, cache_state) -> mx.array
```

- `hidden_states`: `[B, L, hc_mult, dim]` — the **hc-expanded** residual stream (Hyper-Connections;
  `hc_mult=4`, `dim=5120`). This is exactly what the source passes: `h = layer.engram(h, ...)`
  after `h = h.unsqueeze(2).repeat(1,1,hc_mult,1)` (`model.py` L1258/L1262).
- `token_ids`: `[B, L]` — used for a shape-consistency check against `cache_state`.
- `cache_state`: supplies this layer's row ids via `cache_state.current_row_ids(layer_hash_index)`
  → `[B, L, 24]`. Pass the shared `NgramHashState` (advanced **once per step**, so both engram
  layers reuse one history walk) or an `_StepState(row_ids, token_mask)`. An optional
  `cache_state.token_mask` (`[B, L]`) shuts the gate on image-span positions.
- **Returns** the updated hc-expanded residual stream `(h + gate·value).astype(hidden.dtype)`
  (reference `Engram.forward` output). The pure additive contribution is `result - hidden_states`.
  Integration is `h = layer.engram_hook(h, token_ids, cache_state)`; the hook attribute defaults
  to `None`.

`wkv` is a callable `[.., 24*256] -> [.., dim*(hc_mult+1)]` — `EngramV41.dense_wkv(weight)` for a
dense resident weight; a resident q8 wkv is wrapped as an `mx.quantized_matmul` callable at load.
`q_weight`/`k_weight` are `[hc_mult, dim]`.

> **Resident wkv is currently ABSENT from the artifact.** The streaming artifact's
> `model.safetensors.index.json` has **zero** engram tensors — `layers.{1,14}.engram.wkv.{weight,scale}`,
> `.q_weight`, `.k_weight` were dropped when the embed tables went to the bank. The **source** has
> them (`layers.{1,14}.engram.wkv.weight` FP8-E4M3 + `.wkv.scale` E8M0, `.q_weight`, `.k_weight`,
> shape `[dim*(hc_mult+1)=25600, 24*256=6144]`). So `EngramV41` takes them as ctor args; the
> converter/integration worker must re-add them resident (small: ~157M params/layer + two `[4,5120]`).

---

## 3. Measured CPU cost — 24-row gather (one token, one layer)

`nice -n 19`, CPU-only mlx 0.32.2, per-call over 300–2000 iters after warmup. "hit" = rows resident;
"miss" = rows evicted then re-read (`dequantize` = the full hook path: gather bytes → mx view →
`mx.dequantize` affine → `[24, 256]`).

| Bank | 24-row hit (dequant) | 24-row miss (dequant) | hit gather-only (no dequant) |
|---|---:|---:|---:|
| Synthetic (200k rows, 54 MB, page-cache warm) | 82.4 µs | 92.3 µs | 14.6 µs |
| Real `engram-L1.bin` (384M rows, 104 GB) | 80.7 µs | 198.9 µs | — |
| Real `engram-L14.bin` (384M rows, 104 GB) | 80.5 µs | 192.1 µs | — |

Reading: `mx.dequantize` dominates a hit (~67 µs of ~82; gather-only is ~15 µs). The synthetic
"miss" is cheap because the whole file is in the page cache — it measures the LRU repopulate +
warm read + dequant. The real-bank miss adds ~110 µs of scattered positional `preadv` from a 104 GB
file over the hit. The pure-Python/numpy path is not the bottleneck (dequant is), so the native
`.cpp` was not ported.

---

## 4. Tests (CPU only, `nice -n 19`, no `-n auto`)

`tests/test_ngram_row_cache.py` (10): slot-count/byte-budget respected; hit/miss/dup stats;
gathered bytes match disk with order+duplicates; eviction never changes returned values;
batched contiguous misses coalesce into one `preadv` (and out-of-order requests re-coalesce by
row id); request larger than the cache served correctly with bounded residency; out-of-range
`IndexError`; MLX affine dequant == converter numpy dequant; geometry validation;
`ByteSize` env parsing.

`tests/test_engram_v41.py` (5):
- **compressed token map size == 99092** on the real tokenizer (pad id 2 → valid compressed id).
- **hash parity**: streaming `NgramHashState` fed in irregular chunks vs an independent numpy
  transcription of the reference `NgramHashState.forward` (torch is not installed in this venv,
  so the reference is transcribed, not imported), 200 random sequences, all 24 row ids per
  position per layer identical, incl. start-of-sequence pad and an all-DEAD masked column;
  every id `< num_embeddings`.
- **rollback**: advance, `trim(10)`, re-feed the same tokens → identical row ids.
- **row dequant parity (real bank)**: 64 rows/layer, MLX dequant == converter numpy dequant
  (bf16 rounding only), and cos vs source FP8 (`model-00047/48-of-00048.safetensors`) meets the
  converter tolerance — asserted on mean/median ≥ 0.99994 with a min floor (see caveat).
- **EngramV41 math**: hook output vs a numpy transcription of `Engram.forward`; masked positions
  pass through untouched; `result - hidden` matches the reference delta.

Pre-existing `tests/test_convert_deepseek_v41_engram.py` (7) still passes with the `EngramBank` rewrite.

```
$ PYTHONPATH=<worktree> nice -n 19 python -m pytest \
    tests/test_ngram_row_cache.py tests/test_engram_v41.py tests/test_convert_deepseek_v41_engram.py -q
22 passed, 2 warnings in 1.27s
```
(The 2 warnings are SwigPy `DeprecationWarning`s from the tokenizers/transformers import, not from this code.)

---

## 5. Caveats / handoffs to the integration worker

- **cos tolerance is aggregate, not per-row.** Over 64 random rows/layer: mean 0.99997, median
  0.99997, but ~2–3% of rows dip to ~0.99992 (< 0.99994). The converter's affine-8 requant meets
  0.99994 in aggregate; the test asserts mean/median ≥ 0.99994 + a 0.9995 min floor, and reports
  the distribution rather than a false per-row guarantee.
- **`--engram-cache-limit` CLI not wired.** Per the ≤40-line rule and because the runtime is not
  yet attached to the model worker, only the isolated env/ctor path is provided
  (`MTPLX_ENGRAM_CACHE_LIMIT` via `cache_bytes_from_env`, and `NGramRowCache(cache_bytes=...)` /
  `EngramBank.open(..., cache_bytes=...)`). Wiring the flag through `mtplx/server` is left to the
  integration worker.
- **Resident wkv/q_weight/k_weight absent** (see §2) — must be added to the artifact before the
  hook can run against the real model.
- **Editable-install shadowing**: the venv's `mtplx` editable finder maps to the MAIN repo. It is
  a `sys.meta_path`-**appended** finder (runs after `PathFinder`), so `PYTHONPATH=<worktree>` (or
  cwd=worktree) makes the worktree win; the tests assert `mtplx.*.__file__` is under the worktree.
- **One engram layer per bank / independent row spaces**: L1 and L14 are separate `.bin` files with
  independent row-id spaces (~384M each); `EngramV41` holds one `NGramRowCache` per layer. A shared
  budget across both layers (if desired) is a runtime-planner decision, not a cache-core one.
