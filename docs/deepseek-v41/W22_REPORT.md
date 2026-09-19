# W22 — DeepSeek-V4.1 cache conforms to the served generation path

`mtplx/models/deepseek_v41_cache.py` + `tests/test_deepseek_v41_served_generation.py`
(additive: `tests/models/test_deepseek_v41_cache.py`). Branch `feat/deepseek-v41-w22`
off `feat/deepseek-v41-streaming` @ `5d6dd8ad`.

## Symptom

`mtplx serve --model ~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4` loaded,
`/health` was up and `/v1/models` listed the model, but the first chat completion
died:

```
generation.py restore_or_prefill_prompt_state -> _prefill (:6184)
  -> _cache_has_recurrent_entries (:4331)
     any(not _is_trimmable(entry) for entry in (cache or []))
       TypeError: 'DeepseekV41Cache' object is not iterable
```

## Root cause

The MTPLX serve/generate path consumes the model cache in the **mlx_lm
convention: a list of per-layer cache entries**, not a bespoke container. W13's
`DeepseekV41Cache` was a single object (`.layers`, `.offset`, `.engram_state`,
`.trim`/`.mark`/`.rollback`, `.new_shared_runtime`), so the first thing the serve
path did to it — iterate — raised. It got past load only because the two cache
layout hooks the runtime runs at `make_cache` time are env-gated and return early
when off (`mtplx/cache_state.py:3840`, `:4090`), so nothing iterated the cache
until `_cache_has_recurrent_entries`.

## Convention followed (with references)

The sibling backbone served through the same runtime — `mtplx/models/deepseek_v4.py`
— is the template:

- `Model.make_cache` returns a **plain list** of per-layer cache objects:
  `mtplx/models/deepseek_v4.py:3748` (`return [DeepseekV4Cache(...) for layer in self.layers]`),
  and `make_mtp_cache` the same at `:3703`.
- Each entry (`DeepseekV4Cache`, `mtplx/models/deepseek_v4.py:2434`) is a full
  mlx_lm per-layer cache: own `.offset` (`:2494`), `trim(n) -> int` decrementing it
  by exactly `n` (`:2582`), `state`/`meta_state` get+set (`:2646`/`:2700`),
  `replace_state` (`:2696`), `is_trimmable()` (`:2722`), `size`/`empty`.

That is exactly the surface the generation/runtime code touches on the cache and
on each entry:

| consumer | expects |
| --- | --- |
| `generation.py:4331` `_cache_has_recurrent_entries` | iterate the cache; `entry.is_trimmable()` |
| `generation.py:3767` `_cache_offset` | `cache[0].offset` |
| `generation.py:3776` `_trim_cache_to_offset` | per-entry `entry.offset`, `entry.trim(delta) -> int`, opt. `entry.max_rollback` |
| `cache_state.py:4490` `rollback_after_verify` | per-entry `entry.trim(verified_tokens)` + `restore_cache` |
| `cache_state.py:4567` `trim_verified_window_to_prefix` | per-entry `entry.trim` returns exactly `trim_tokens`; `_entry_offsets(entry)` (`:4652`, reads `entry.offset`) drops by exactly that |
| `cache_state.py:4620` `trim_verified_window_without_snapshot` | every entry `is_trimmable()` (skip-verify-snapshot lane) |
| `cache_state.py:4334`/`:4384` `snapshot_cache` / `snapshot_untrimmable_cache` | iterate; read `entry.state`/`entry.meta_state` (untrimmable variant skips trimmable entries without reading `.state`) |
| `runtime.py:535` `MTPLXRuntime.make_cache` | `inner.make_cache()` then `configure_owned_recurrent_state_cache` / `configure_tail_owned_attention_kv_cache` iterate it (both skip our entries — no `keys`/`values`, not recurrent) |

## What changed (only `mtplx/models/deepseek_v41_cache.py`)

W13's semantics are unchanged — same window ring (`ring_view` / `window_topk_idxs`),
same `CompressorState` frontier, same shared runtime, same append-only trim/rollback
exactness. The shape is reworked to the list convention:

- **`LayerAttentionCache` is now a full mlx_lm per-entry cache.** Added its own
  `offset` + `advance(n)`, `is_trimmable() -> True`, `size`/`empty`, `state`/
  `meta_state` (get+set) + `replace_state`. `trim(n)` now **returns the count**
  and decrements `offset` by exactly `n` (the invariant `trim_verified_window_to_prefix`
  checks), on top of the existing window/compressed/index/frontier truncation.
- **`DeepseekV41Cache` presents as the list**: `__iter__`/`__len__`/`__getitem__`/
  `__setitem__`/`__bool__` over `.layers`; `.offset` is a property = `layers[0].offset`
  (every entry advances/trims in lockstep). The cross-layer state the reference
  keeps process-global stays reachable for the W10 backbone: `new_shared_runtime()`
  and the engram history via the `.engram_state` property. The engram is one object
  per sequence, so the **first entry owns its rewind** — its `trim` moves it exactly
  once even though the serve path trims every entry (`NgramHashState.trim` is exact:
  append-only `_buf` truncation, `engram_v41.py:267`). The `.engram_state` property
  setter keeps that owning entry in sync so re-pointing it (backbone / loader-contract
  test) re-owns the history on the entry that trims it.
- Whole-sequence `trim`/`mark`/`rollback` stay on the container for the W13 unit
  tests and the bare-forward path; the served verify path drives the per-entry
  `trim` instead. Entry `rollback` trims the engram by the offset delta (offset and
  engram length advance in lockstep), so a hook stand-in that only records `trim`
  still rewinds.

**No model change was needed.** The W10 backbone forward
(`mtplx/models/deepseek_v41.py:696-724`) already reads `cache.offset`,
`cache.engram_state`, `cache.new_shared_runtime()`, `cache.layers[i]`,
`cache.advance()` — all preserved on the container — and `Model.make_cache`
(`:868`) is unchanged. `deepseek_v41.py` was not touched.

## Verification (CPU, tiny random-config model, no artifact)

`tests/test_deepseek_v41_served_generation.py` (8 tests) drives the real path:

- `runtime.make_cache()` is the served list shape: iterable, `len == num_hidden_layers`,
  `_cache_has_recurrent_entries(cache) is False`, every entry trimmable with its own
  `.offset`/`.trim` — the regression pin for the original `TypeError`.
- `restore_or_prefill_prompt_state(rt, prompt)` prefills (the exact failing call) and
  a few `forward_ar` decode steps emit in-range tokens; `generate_ar` emits tokens.
- **Rollback exactness / MTP-verify contract**: on the full §0 layer menu (swa /
  full-r2 / reuse / full-r1+cand / reindex), `rollback_after_verify` + refeed is
  **bit-exact** — stored `state` and next-token logits identical to the never-decoded
  path. `trim_verified_window_without_snapshot` (skip-snapshot lane) and
  `_trim_cache_to_offset` (near-prefix restore) both trim to the target offset; a
  cache carrying a non-trimmable entry is correctly refused snapshot-free repair.
- **Engram through the serve loop**: exactly entry 0 owns it, `generate_ar` advances
  it, and a verify rollback rewinds its length in lockstep with the offset (once, not
  once per layer).

Existing suites pass: `tests/models/test_deepseek_v41_cache.py` (22, W13 semantics
preserved), and the V4.1 model/parity/engram/loader-contract sweep — 65 passed, 20
artifact-skipped. The one red in the sweep,
`tests/models/test_deepseek_v41_streaming_clamp.py::test_spec_swiglu_limit_values`,
is **pre-existing on `5d6dd8ad`** (measured on a detached base worktree; that test
does not import the cache and is unrelated to this change).

Peak RSS for the served+cache+parity run: **~0.29 GB** (`/usr/bin/time -l`
`maximum resident set size` 294,289,408 B) — CPU-only, `mx.set_default_device(mx.cpu)`,
well under the 10 GB cap. No artifact loaded.

## Session / SSD prompt-cache save-restore — verdict: keep DISABLED while engram is wired

Verified empirically (tests `test_session_snapshot_roundtrips_kv_but_not_engram_documents_the_gate`
and `test_ssd_on_disk_prompt_cache_is_inoperative_for_this_cache_shape`):

- **In-memory session bank** (`session_bank.py:974` `snapshot_cache_lazy_hybrid` store,
  `_trim_cache_to_offset` near-prefix restore): the KV lanes (window ring, compressed
  KV, index keys, compressor frontier) **and per-entry offsets round-trip bit-exactly**
  through `snapshot_cache` + `restore_cache`. **But the engram n-gram history is not
  part of the mlx_lm `state` contract** — it is per-sequence streaming numpy state
  advanced from token ids and rewound only by `trim` — so a KV-only snapshot restore
  desyncs the engram from the KV on the engram-wired artifact (hashing layers 1 and
  14). A warm-turn near-prefix restore would then compute wrong engram-layer hidden
  states for the suffix.
- **SSD on-disk cache** (mlx_lm `save_prompt_cache`): **raises** (`RuntimeError:
  std::bad_cast`) on this cache's nested/`None`-holed `state` tuples — it **fails
  closed** (no corruption), so the SSD path is inoperative for the V4.1 backend.

The session bank is **auto-enabled** in the engine (`engine_session.py:1622`) and
near-prefix restore/store default **on** (`generation.py:3847`
`_near_prefix_restore_enabled`, `:4526` `MTPLX_SESSION_STORE_ON_PREFILL`), with no
per-model gate. So this is not disabled by construction for engram serving.

- **Operational mitigation (now):** serve V4.1 with near-prefix session restore off —
  `MTPLX_SESSION_NEAR_PREFIX_RESTORE=0` and `MTPLX_SESSION_STORE_ON_PREFILL=0` (or no
  session bank). The KV-only round-trip is correct, so a model served **without engram**
  is already safe.
- **Root-cause follow-up (scoped, out of this task's allowlist):** serialize the engram
  `_buf`/`_len` into the owning entry's `state`/`meta_state` so `snapshot_cache` captures
  it and `restore_cache` restores it (which would make near-prefix restore correct). That
  needs a public serialize/restore hook on `mtplx.engram_v41.NgramHashState`
  (`engram_v41.py`), which is outside this task's file allowlist; the MTP-verify path is
  unaffected either way (it uses `snapshot_untrimmable_cache`, which does not read `state`
  for trimmable entries). The SSD serializer's `None`-hole limitation is separate and
  shared with `deepseek_v4`'s cache.

## Commits (`feat/deepseek-v41-w22`)

1. `31ce3d4d` cache reshaped to the mlx_lm list-of-per-layer-caches.
2. `ffde0cfe` served-order test over a tiny CPU model.
3. `68ebc43c` session/SSD save-restore verification tests.
4. `5a777b2f` container `engram_state` property + per-entry rollback compatibility
   with the W13 API (restores the loader-contract engram rollback test).
