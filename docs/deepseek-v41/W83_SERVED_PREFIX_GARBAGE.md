# W83 — served DeepSeek-V4.1 block-prefix (SSD) restore corrupts the cache

Files: `mtplx/cache_bank/codec.py` (fix), `scripts/deepseek_v41/served_cell_bench.sh`
+ `scripts/fable/server_cell_bench.py` (bench hygiene), `tests/test_deepseek_v41_served_prefix_garbage_w83.py`
(reproduction + regression). Branch `w83/served-prefix-garbage` off `6647e0003`.
Server-log evidence copied to `docs/deepseek-v41/receipts/w83/`.

## Symptom

Served DSV4.1-Flash on the Qwen-PR 16,384-token cell produced GARBAGE in the two
runs that reused a 768-token prefix, and coherent text only in the run with no
reuse:

| GPU window / lane | prefill levers | `new_prefill_tokens` | output |
| --- | --- | --- | --- |
| w32 / served-ar   | ON (`KV_CHUNK_GROW=1 SELECTED_KEYS=1 LAYOUT_FIX=1`) | 16384 (no reuse) | COHERENT (270 tok) |
| w32 / served-dspark | ON, native-MTP head | 15616 (768 reused) | GARBAGE (`:\n_ORTP_last_ORTP_LAST_OR…`, 320 tok) |
| w30 / served-ar   | **OFF** (`PREFILL_LAYER_MAJOR=<unset>`, plain backing) | 15616 (768 reused) | GARBAGE (`:\n    "auto…`, 138 tok) |

Garbage tracks the **768-token prefix reuse**, not the levers, the window-ring/chunk-grow
backing, or the DSpark lane (it happens on the plain-backing AR lane too). Every
server flushed SSD session-cache writes at shutdown and reloaded them at the next
start, so the 768-token reuse is served from the **SSD session bank persisted
across launches** — the coherent run is the one that prefilled all 16,384 tokens
cold. 768 = 3 × 256, and 256 is the session-bank/codec block size (block-aligned
restore). Logs: `receipts/w83/serve-20260911T224025Z.log` (w32 dspark garbage),
`serve-20260911T223408Z.log` (w32 ar coherent), `serve-20260911T211827Z.log`
(w30 ar garbage).

## Root cause (task 1) — SSD block-prefix restore slices the wrong tensor axis

The SSD cold tier serves a divergent follow-up by restoring a *block-aligned
prefix* of a long banked entry and re-prefilling only the suffix
(`session_bank._cold_near_prefix_candidate` → `cache_bank/cold_tier.py`
`lookup_prefix_boundary` → `_restore_row`). For a partial restore
(`cold_tier.py:2006-2049`) it decodes the prefix by **slicing persisted tensor
blocks along axis 2** (`codec.decode_payload_prefix` → `_decode_tensor_blocks`,
`_decode_tree_prefix`), gated by `payload_supports_prefix_decode` /
`snapshot_supports_prefix_decode` (`codec.py:520-551`).

Axis 2 is the token axis **only for a standard mlx_lm KVCache** `[B, H, T, D]`.
The DeepSeek-V4.1 per-layer state tensors are `[B, seq, head_dim]` — the token
axis is **axis 1** (probed on the tiny model: window `(1,40,16)`, compress_kv
`(1,20,16)`, index_k `(1,20,8)`, compressor frontier `(1,40,16)`). So the axis-2
prefix slice **never trims the sequence**:

- head_dim < prefix_len (the real model, head_dim ≥ 512, block-sliced along
  axis 2, prefix 767): the head_dim blocks all survive → the lane is returned at
  its **full banked length**, untrimmed.
- head_dim not block-encoded (tiny model, head_dim 16 < 2·256): a plain `tensor`
  spec falls straight through `_decode_tree_prefix` → also returned untrimmed.

Either way `decode_payload_prefix` returns the whole banked entry's lanes (and
an all-`None` `meta_states` tree, `_none_tree_like`), yet labels them
`cache_snapshot_prefix_len == requested_prefix`. Then
`SessionBank.restore_entry_prefix_cache` (`session_bank.py:1898-1903`) sees
`cache_snapshot_prefix_len == required_cache_prefix_len` and installs
`trim_to_target = lambda _cache: True` — **skipping its corrective
`entry.trim`**. The served cache therefore holds the entire banked entry's KV
(wrong length; offset never restored / the `None`-tuple meta is rejected by the
V4.1 `meta_state` setter) while the engine believes it restored the short block
boundary. The suffix prefill runs on desynced KV → token soup.

`snapshot_supports_prefix_decode` already excludes `deepseek-v4` (its 5 fields
couple rolling-window/compressor counters to the tensors) and Gemma's rotating
cache, but **not V4.1**, whose sequence axis is likewise not axis 2 — so V4.1
was wrongly authorised to be block-sliced. **Defect: `codec.py`
`snapshot_supports_prefix_decode` returns `True` for the V4.1 layer-cache meta
version `mtplx-deepseek-v41-layer-cache-v1`.**

Why the RAM path is unaffected: a RAM near/block-prefix candidate keeps the full
snapshot with `cache_snapshot_prefix_len=None`, so `restore_entry_prefix_cache`
takes its real trim (`entry.trim`), which rewinds every V4.1 lane exactly (window
ring, compress/index groups, compressor frontier, engram — W22/W26/W80). The bug
is specific to the SSD cold tier's `decode_payload_prefix` shortcut, i.e. to
cross-launch (persisted-bank) reuse — exactly the observed failure mode.

## Reproduction (task 2) — CPU tiny model, no artifact

`tests/test_deepseek_v41_served_prefix_garbage_w83.py`, CPU-pinned, tiny random
`_csa_args` model (full CSA menu + engram), peak RSS 180 MB, 1.3 s:

1. `test_snapshot_supports_prefix_decode_refuses_v41_layout` — the gate contract:
   `payload_supports_prefix_decode` must be `False` for a V4.1 snapshot. **Failed
   before the fix** (returned `True`).
2. `test_decode_payload_prefix_does_not_trim_v41_token_axis` — mechanism: with a
   40-token banked cache, `decode_payload_prefix(cache_prefix_len=15)` returns
   the window lane still at 40 rows on axis 1 (the axis-2 slice missed the token
   axis). Passes (documents the bug).
3. `test_block_prefix_ssd_restore_matches_ram_reuse` — end-to-end through the
   REAL `SessionBank.restore_entry_prefix_cache` (the exact call
   `generation.py` makes on a near-prefix candidate), decoding the cold entry
   with the SAME dispatch `cold_tier._restore_row` uses. The SSD block-prefix
   restore + suffix decode must equal the RAM reuse of the same banked state
   (in-memory trim to the boundary — the ground-truth reuse, bit-exact, unlike a
   *fresh short prefill* which differs at ~1e-7 from a trim due to prefill kernel
   shape). **Failed before the fix** (untrimmed lanes / rejected `None` meta).
4. `test_full_decode_then_trim_matches_ram_reuse` — control: the full-decode +
   `entry.trim` path (the fix's route) matches RAM reuse bit-exactly. Passes
   before and after → the fix's target path is correct and safe.
5. `test_sampled_verify_rollback_cache_consistent` — Hypothesis B (see below).

## Fix (task 3) — refuse to block-slice the V4.1 layout

`mtplx/cache_bank/codec.py`, `snapshot_supports_prefix_decode`: exclude the V4.1
layer-cache meta version, exactly as `deepseek-v4` is excluded.

```python
if values and values[0] == "mtplx-deepseek-v41-layer-cache-v1":
    return False
```

This routes V4.1 SSD block-prefix restores to the full `decode_payload` +
`entry.trim` path — the same path `deepseek-v4` already uses, exact for every
V4.1 lane. It is the "refuse to reuse when a lane cannot be restored, fall back
to full decode/trim" option. No perf regression: the axis-2 prefix decode never
worked for V4.1 (it read the same blocks and failed to trim), so full decode +
trim is strictly more correct at the same-or-lower SSD read cost.

Post-fix: all 5 W83 tests pass (1.31 s, 180 MB). Regressions clean:
`tests/test_cache_bank.py` 27, `tests/test_session_bank.py` 18,
`tests/test_cold_prefix_ram_shadow.py` 9, `tests/test_cold_tier_min_useful_matched.py`
3, `tests/test_deepseek_v41_served_generation.py` 8,
`tests/models/test_deepseek_v41_cache.py` 22, `tests/test_deepseek_v41_bench_scripts.py`
21 — all passed.

## Benchmark hygiene (task 4)

- `scripts/deepseek_v41/served_cell_bench.sh`: new `DSV41_PREFIX_REUSE` toggle,
  **default `off`**. When off the server is launched with
  `MTPLX_SESSION_NEAR_PREFIX_RESTORE=0 MTPLX_SESSION_BLOCK_PREFIX_RESTORE=0
  MTPLX_SESSION_STORE_ON_PREFILL=0` and an **isolated throwaway SSD cache dir**
  (`--ssd-session-cache-dir <log>/ssd-session-cache-<stamp>`), so no cell — and
  no prior launch's persisted entry — can serve any prefix. Each timed cell is a
  cold prefill, matching the Qwen-PR harness. `DSV41_PREFIX_REUSE=on` measures
  the warm-reuse path deliberately (loud warning logged).
- `scripts/fable/server_cell_bench.py`: every cell receipt now carries a
  `prefix_reuse` field `{prompt_tokens, new_prefill_tokens, reused_prefix_tokens,
  cached_tokens, restore_kind, flagged}`. `flagged` is true iff
  `new_prefill_tokens < prompt_tokens`; `warn_if_prefix_reused` prints a loud
  stderr `WARNING [W83 prefix reuse]` for any flagged row at both cell-print
  sites, so a reused cell is never mistaken for a clean full-prompt measurement.
  Verified: `(16384 → 15616)` flags with 768 reused; `(16384 → 16384)` is clean.

## Hypothesis B (task 5) — sampled DSpark verify/rollback: REFUTED

B is refuted by the evidence and by a direct check, not the cause:

- The garbage occurs on the **plain-backing AR lane** (w30, no DSpark drafter),
  so a DSpark-lane sampled-verify defect cannot explain it.
- `test_sampled_verify_rollback_cache_consistent`: a sampled (temperature-1)
  speculative tail forwarded then rolled back leaves **every lane bit-identical**
  to the never-decoded path, and the continuation logits match. The verify
  rollback is token-driven, not sample-driven — the sampler chooses which tokens
  are proposed, never how the cache trims/restores — so the sampled DSpark verify
  path cannot corrupt the cache. (Greedy exactness is pinned in
  `test_deepseek_v41_served_generation.py`; this adds the sampled tail.) Consistent
  with W77 finding the DSpark divergence sane.
- Scope note: the tiny runtime is the AR trunk (`mtp_enabled=False`); a full
  MTP/DSpark-graft distribution harness was not built because B is already
  refuted by the AR-lane garbage and the rollback bit-exactness — building one
  would be disproportionate to a refuted hypothesis.

## Hypothesis C — chunk-major prefill: not the cause

The two garbage runs bracket the prefill-lever configs (w30 AR had them OFF, w32
DSpark had them ON) and both required the 768 reuse; the coherent run had the
levers ON with no reuse. So the prefill path is not the discriminator — the SSD
block-prefix reuse is.
