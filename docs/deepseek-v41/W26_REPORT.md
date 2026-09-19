# W26 — Engram history joins the entry-0 cache state (session near-prefix restore / store-on-prefill can be re-enabled)

Branch `feat/deepseek-v41-w26` off `feat/deepseek-v41-streaming` @ `629eb04d`.
Files: `mtplx/engram_v41.py`, `mtplx/models/deepseek_v41_cache.py` (entry-0 state),
`tests/test_deepseek_v41_engram_state.py`, this report. CPU-only, tiny synthetic
configs, no artifact loaded.

## Problem (from W22)

The served path's session bank (`engine_session.py:1622`, auto-enabled) snapshots
and restores the model cache through each entry's `state` / `meta_state` /
`replace_state`. W13/W22 deliberately left the per-sequence engram n-gram history
(`NgramHashState._buf` / `_len`, streaming numpy state in `mtplx/engram_v41.py`)
**out** of that contract — it rewound only through `trim`. So a KV-only snapshot
restore (the session bank's near-prefix store/restore) desynced the engram from
the KV on the engram layers (1 and 14): a warm turn computed wrong engram-layer
hidden states for the suffix. And `mlx_lm.save_prompt_cache` (the SSD on-disk
cache) raised `RuntimeError: std::bad_cast` because the raw numpy `_buf` is not an
`mx.array` leaf. W22's mitigation was to serve V4.1 with the two session knobs
forced off:

- `MTPLX_SESSION_NEAR_PREFIX_RESTORE=0` (`generation.py:3847` `_near_prefix_restore_enabled`)
- `MTPLX_SESSION_STORE_ON_PREFILL=0` (`generation.py:4526`)

## Fix

**(1) `mtplx/engram_v41.py` — public serialise/deserialise on `NgramHashState`.**
A new `state` property returns the streaming history as one `mx.array`
(`[B, T]` int32; `DEAD == -1`), and `replace_state(value)` (also the `state`
setter) reinstates it. Only the *history* travels; the immutable hash config
(compressed token map, multipliers, primes, flat offsets) is shared across
sequences and rebuilt by `fresh()`, so it is not serialised. `fresh()` /
`advance` / `trim` semantics are unchanged: `replace_state` restores `_buf`
exactly (as int64, the working dtype), derives `_len`, and clears the transient
`_current` (the next `advance` recomputes it before any read, exactly as `trim`
does). A never-advanced state serialises to the canonical empty `[1, 0]` buffer;
restoring an empty buffer resets to the fresh history.

**(2) `mtplx/models/deepseek_v41_cache.py` — entry-0 state carries the engram.**
The entry that owns the per-sequence engram (only the first entry of a sequence)
appends `engram_state.state` as a 6th `mx.array` leaf to its `state` tuple; the
`state` setter / `replace_state` accept both the 6-tuple (restore the engram) and
a 5-tuple (KV-only / pre-W26 snapshot; engram left untouched). **Every other
entry, and every entry of a no-engram model, is byte-identical to before W26**
(`engram_state is None` → the plain 5-tuple). Because the leaf is an `mx.array`
(no numpy, no `None`), it round-trips through both
`mtplx.cache_state.snapshot_cache` / `restore_cache` and `mlx_lm.save_prompt_cache`.
A `from_state` classmethod (mlx_lm `load_prompt_cache`'s contract) reconstructs an
entry's KV lanes / compressor frontier / offset and parks any engram leaf on
`loaded_engram_state` for the caller to rehydrate into a fresh `NgramHashState`
(the config is shared and unserialised).

`meta_state` is intentionally left at its 5-field form (`version`, `offset`,
`window_size`, `compress_ratio`, `is_kv_source`): the history rides `state`, so
no meta change is needed.

## Evidence (`tests/test_deepseek_v41_engram_state.py`, 6 tests, all pass)

CPU-pinned, tiny random-config 2-layer model, a real `EngramV41` hook attached to
layer 1 over a synthetic affine-q8 bank (so decode logits genuinely depend on the
engram history), plus direct `NgramHashState` unit coverage. Peak RSS for the
file: **0.11 GB** (`/usr/bin/time -l` `maximum resident set size` 114,999,296 B).

1. **`NgramHashState` round-trip + semantics** — `state` is int32 `[B, T]`;
   `replace_state` restores `_buf` bit-exactly; continued `advance` is bit-exact
   vs the uninterrupted state; the empty/fresh buffer resets; `trim` still works
   after a restore.
2. **Near-prefix restore then decode is BIT-EXACT with engram** — prefill, then
   `snapshot_cache`; restore into a **fresh** cache (fresh engram) and decode the
   same tokens. Entry 0's state is a 6-tuple with the `mx.array` engram leaf; the
   fresh cache's engram length goes 0 → `len(prompt)` on restore; **decode logits
   are identical, token after token**, to the uninterrupted path. This is the
   session-bank warm-turn path made correct.
3. **Production store path (`snapshot_cache_lazy_hybrid`)** — the session bank
   stores with the zero-copy-view snapshot; the engram leaf round-trips
   bit-exactly through it too, even when the live cache is mutated after the
   snapshot (the retained view is not aliased to the live buffer).
4. **KV-only restore desync control (proves the fix matters)** — dropping the
   engram leaf (the pre-W26 5-tuple) leaves the engram fresh and makes decode
   logits **differ**; the full 6-tuple restore makes them identical. This is the
   exact W22 bug and its cure, side by side.
5. **No-engram path unchanged** — a no-engram cache has a 5-tuple `state` on every
   entry, and KV snapshot/restore + decode stays bit-exact.
6. **SSD save/load round-trip** — on an engram-owning entry whose KV lanes are all
   real arrays, `mlx_lm.save_prompt_cache` **succeeds** (where the pre-W26 numpy
   `_buf` raised `std::bad_cast`), and `mlx_lm.load_prompt_cache` reconstructs the
   entry; rehydrating the loaded engram buffer into a `NgramHashState` with the
   shared config reproduces the history bit-exactly and continued hashing matches.

Existing suites (measured this run, CPU, `nice -n 19`, no `-n auto`):

- `tests/models/test_deepseek_v41_cache.py`, `tests/models/test_deepseek_v41_parity.py`,
  `tests/models/test_deepseek_v41_engram_attach.py`,
  `tests/models/test_deepseek_v41_loader_contract.py`, `tests/test_engram_residents.py`,
  `tests/test_engram_v41.py`, `tests/test_deepseek_v41_served_generation.py` — all
  green **except one expected inversion** (below).
- `tests/test_cache_state.py::test_vllm_metal_partitioned_paged_attention_matches_stock_attention`
  fails only in-sweep and passes in isolation; it is a **pre-existing** cross-test
  device-default ordering flake — reproduced identically on the base commit
  `629eb04d` (`1 failed, 115 passed, 3 skipped`), and it imports none of W26's
  code. Not attributable to this change.

## Verdict: the two session knobs can be re-enabled

With the engram history in the entry-0 state contract, `snapshot_cache` /
`restore_cache` now round-trip it (test 2/3), so a session-bank near-prefix store
+ warm-turn restore reproduces the engram-wired decode **bit-exactly** instead of
desyncing layers 1/14. The operational mitigation from W22 can be lifted for the
engram artifact:

- `MTPLX_SESSION_NEAR_PREFIX_RESTORE` back to its default (on)
- `MTPLX_SESSION_STORE_ON_PREFILL` back to its default (on)

These env defaults are set off in the **serving profile files (W21 owns them)**;
W26 does not touch them — it makes the cache contract that they depend on correct.
The MTP-verify path was never affected (it uses `snapshot_untrimmable_cache`,
which does not read `state` for trimmable entries).

## Follow-ups / boundaries

- **One W22 test now asserts the fixed-away limitation and must be updated by its
  owner** (outside this task's allowlist):
  `tests/test_deepseek_v41_served_generation.py::test_session_snapshot_roundtrips_kv_but_not_engram_documents_the_gate`.
  It asserts `int(cache.engram_state.length) == eng_before + 3` (engram NOT
  restored). W26 restores it, so the length is `eng_before` (== `off_before`).
  The corrected assertion is `assert int(cache.engram_state.length) == eng_before`
  and the surrounding comment should say the engram now round-trips with the KV.
  The sibling test `test_ssd_on_disk_prompt_cache_is_inoperative_for_this_cache_shape`
  still passes unchanged: its `save_prompt_cache` raise is from the KV `None`
  holes on non-kv-source layers, not the engram.
- **SSD `None`-hole limitation is separate and out of scope.** A full-cache
  `save_prompt_cache` on a real multi-layer V4.1 cache still raises on the `None`
  KV lanes of sliding-window / non-kv-source entries (shared with
  `deepseek_v4`'s cache). W26 fixes only the engram half (numpy → `mx.array`); the
  `None`-hole fix would touch every entry's state, which is outside "entry-0 state
  only". Production SSD restore would also need `LayerAttentionCache` registered
  into `mlx_lm.models.cache` globals (the `arrays_cache_patch` pattern) — a
  one-liner that belongs with the engine/session wiring, not this task's files.
