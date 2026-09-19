# W13 — DeepSeek-V4.1 per-sequence attention cache

`mtplx/models/deepseek_v41_cache.py` + `tests/models/test_deepseek_v41_cache.py`.

Lifts every piece of per-sequence attention **state** the DeepSeek reference
(`inference/model.py`) keeps on its weight modules into one cache object the
MTPLX serve path owns, so many sequences share one set of weights and the state
trims/rolls back for speculative verify and the resident==stream gate. The module
is weight-free: it stores and mutates state, and does the pure index/roll
mechanics; the wkv/wgate/wk/weights_proj/wq_b/RMSNorm/RoPE/quant math stays in
W10's `deepseek_v41.py`.

## What the reference keeps as state, and where it went

| reference (model.py) | this module |
| --- | --- |
| `Attention.window_kv_cache` `[B,W,head_dim]` ring (L663-668, write L708-719, read L716/719) | `LayerAttentionCache.window` (append-only) + `ring_view` / `.ring(length)` — reproduces the ring slot layout exactly |
| `get_window_topk_idxs` (L409-426) | `window_topk_idxs(...)` (transliterated line-for-line) |
| `Attention.compress_kv_cache` (L669-679, write L761) | `LayerAttentionCache.compress_kv` (append-only, one row / completed group) |
| `Indexer.k_cache` (L520-525, write L547) | `LayerAttentionCache.index_k` |
| `Compressor.kv_state` / `score_state` `[B,ratio,head_dim]` (L449-456, update L466-485) | `CompressorState.raw_kv` / `raw_score` + `.push` |
| `shared_attn` module-global `SharedAttentionRuntime` (L1166-1180) | `SharedAttentionRuntime` (one per cache) |
| `start_pos` bookkeeping (forward signatures) | `DeepseekV41Cache.offset` |
| `Transformer.engram_hash` history (engram.py L155-175) | `DeepseekV41Cache.engram_state` (an `engram_v41.NgramHashState` clone) |

## The two transliteration decisions worth flagging

**Window ring is realised as a view over append-only history.** The reference
ring is a fixed `window_size` (128) buffer with modular writes — bounded, but it
cannot undo a rollback that has already overwritten a slot. The serve path needs
`trim(n)` for speculative verify, so phase 1 keeps the full post-RoPE window
history and `ring_view(window, W, length)` reproduces the reference's exact slot
layout on demand: a full `[B, W, head_dim]` buffer where slot `s` holds the
newest fed position `p < length` with `p % W == s` (empty slots zero while
filling) — exactly what the reference decode reads as `window_kv_cache[:bsz]`,
and what pairs with `window_topk_idxs`' slot indices.
This makes `trim` exact at any depth and generalises the write to multi-token
chunks at nonzero `start_pos` (which the reference, single-token past prefill,
never sees). It reduces to the reference formula on the reference's own call
pattern — verified against a token-by-token oracle. Phase 2 bounds the storage.

**The compressor frontier retains its raw rows.** The reference parks the
partial group in a fixed `ratio`-slot buffer; `CompressorState` retains every fed
`(kv, score)` row so a rollback that crosses a group boundary re-exposes the
shortened partial group exactly (the pooled/`index_k` stores just truncate to
`new_len // ratio`). The reference `-inf`-pads unfilled `score_state` slots, but
it only ever pools a **complete** group, so those pad slots never enter a softmax
and the unpadded tail is equivalent. Pooling reads only a group's own rows, so
chunked feeding equals one-shot feeding.

## Tests (CPU, 22 cases, all green)

`mx.set_default_device(mx.cpu)` at import (a "no GPU" test still defaults MLX to
Metal without it). Every assertion is against an independent numpy transcription
of the reference state updates on random data.

* **window ring rolling across the 128 boundary** — 300 positions fed in chunks
  of 1/7/64/300 all produce a ring identical to a token-by-token reference
  oracle; the still-filling view equals the raw prefix; `window_topk_idxs`
  matches the reference for prefill and decode shapes.
* **compressor pooling** — ratio-2 pooling fed across odd/even chunk boundaries
  (`[13]`, `[1]*13`, `[3,4,5,1]`, `[2]*6+[1]`, `[7,6]`) equals one-shot pooling
  of the whole sequence; `n_fed` / `n_groups` track.
* **index-K append**, **candidate/top-k publication + reuse read** (one slot
  each, identity reads; `topk_mask` ↔ `topk_idxs` alias).
* **trim(n) then re-feed == identical state and identical attention inputs** —
  stored window/compress_kv/index_k/compressor raw and the derived ring +
  `window_topk_idxs` all match a direct feed; a trim that crosses a group
  boundary lands on the exact group count; `mark`/`rollback` matches `trim`.
* **engram_state advances and trims in step** — `NgramHashState.length` tracks
  `cache.offset` through advance and trim; trim-then-refeed restores identical
  n-gram row ids (lookback crosses the boundary).
* **gate / mlx_lm contract** — `mlx_lm.make_prompt_cache(model)` returns this
  cache; the `_greedy_decode` shape (make once, reuse the identical object every
  step, then a speculative `trim`) works; `make_cache` builds from a ModelArgs-
  like object.

Run: `PYTHONPATH=$PWD nice -n 19 .venv/bin/python3 -m pytest tests/models/test_deepseek_v41_cache.py -q` → `22 passed`.

## Integration

W10's call surface is frozen in `PORT_CONTRACT.md` (W13 heading): W10 deletes its
inline `_SharedRuntime` / `_LayerCache` / `DeepseekV41Cache`, imports from this
module (field names kept), and points `Model.make_cache` at `make_cache(self.args, …)`.
No change to `mlx_lm.make_prompt_cache(model)` → `model.make_cache()` →
`model(ids, cache=cache)`, and `runtime.py`'s `configure_*` cache hooks are
pass-throughs unless their env flags are set.

## Scope / not done

Phase-1 float storage only (no fp8 window / fp4 compressed quant — reference
L707/L546/L760; the ring/roll semantics are identical, the storage dtype is
phase 2). Bounded ring storage is phase 2. MTP/DSpark draft-head caches
(`DSparkAttention`, reference L1032-1074) are out of scope. This module was not
wired into a live model forward here (W10 owns that); the state is validated
against the numpy reference and the mlx_lm cache contract.
