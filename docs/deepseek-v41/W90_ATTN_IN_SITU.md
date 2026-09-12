# W90 — in-situ decode-attention overhead: what it is (and isn't)

## Verdict

The ~4-5 ms/layer gap between in-model decode attention and the isolated Metal
microbench (~2.0 ms/layer, flat across T=1K/4K/16K, even with 88 GiB ballast) is a
**mode-independent in-situ floor**, and the strongest-supported cause is **GPU DVFS
downclocking in the idle gaps between B=1 dispatch bursts** — not attention work,
not an O(T) memory reference, not thermal throttling.

This doc supersedes an earlier W90 draft that pinned the gap on the K30 gather
referencing O(T) source buffers. **That claim is falsified** (see below); it is
recorded here so the ledger shows why the shape-stable lever is only a small
dispatch-count cleanup, not the floor fix.

### Ranked

1. **GPU DVFS / clock-gating (strongest hypothesis, telemetry-backed).** `macmon`
   sampled during a live 16,384-token decode read **71 °C (GPU) / 77 °C (CPU) —
   not throttling**, GPU **power ~8.2 W**, **~45 % busy**, **frequency averaging
   ~1030 MHz and swinging 580–1381 MHz second to second** (CPU ~18 %). The GPU is
   idle more than half the time; between the per-token sync/latency gaps DVFS drops
   the clock, so each short kernel burst *after* a gap runs at a reduced frequency.
   The isolated microbench runs a tight loop that keeps the GPU awake at a high
   clock → flat 2.0; the in-model decode (40 layers × per-stage fences + host round
   trips per token) leaves gaps → each burst is slow. This is mode-independent (every
   layer dispatches a similar short chain), which matches the census. **Decisive
   next check:** window 36 runs the in-model 16K bisect `--cooldown-s 180` vs `0`,
   and both scripts now log a `utilization` block (GPU freq/power/busy over the timed
   decode). Cooldown is secondary (macmon already ruled thermal out); the *utilization*
   trace is the primary read.

2. **Fence-serialized B=1 host-encode latency (contributing, not the whole floor).**
   The census fences every stage (`mx.eval`), so decode is fully serialized — no
   host/GPU overlap. At B=1 the wall is a big-kernel chain + host-encode latency
   ([[b1-decode-dispatch-removal-hides]]); the fences remove the overlap the served
   (unfenced) path has. This inflates the *census* absolute numbers but is present in
   the isolated bench too (it also fences per step), so it is not the isolated-vs-
   in-model delta by itself — it is the mechanism that *lets* the DVFS gaps open.

### Falsified / ruled out

- **O(T) source-buffer reference (the earlier W90 claim) — FALSIFIED.** The gather
  does reference O(T) sources (`window_all [1,T,hd]`, shared `compress_kv
  [1,~T/2,hd]`; E3 confirms the shapes scale with T). But: (a) the in-model census
  `docs/deepseek-v41/receipts/gpu-windows/window-34/w78-in-model.json` shows
  **`attn.swa_only` at 7.447 ms/layer vs `attn.reuse` 6.960** — swa_only, a layer
  with **no `compress_kv` at all** and only a ~5 MB window under the ring, costs
  **more** in situ than reuse (which references the O(T) `compress_kv`); (b) the
  isolated microbench **already references O(T) sources per dispatch** yet is flat;
  (c) MLX 0.32.2 residency is per-allocation (ResidencySets), not per-referenced-
  byte; (d) the W80 ring bounded the *larger* (window) lane for only ~0.2–0.5 ms/
  layer. A per-dispatch cost that scaled with referenced source size would not put
  swa_only ≥ reuse. So the O(T) reference is not the floor.

- **#1 compile retrace per token — RULED OUT (E1).** The K22 attention tapes
  (`_attn_qkv_prep`/`_attn_out_prep`) take only `s==1`-shaped inputs and are keyed
  by codec/geometry, not T. Instrumenting `mx.compile`, each impl traces **exactly
  once** and replays across all decode steps *with T advancing 64→96*; the compiled
  cache stays at 4 entries. `shapeless` unneeded; no shape-dependent Python on the
  token axis.

- **#2 ATTN_WIN_MEMO miss — RULED OUT (E2).** With `SELECTED_KEYS=1` (cell16k)
  `_window_attend` is **never called** (0 calls over 10 decode steps) — the K30
  selected-key path uses bounded `_window_selected_idx` instead. Even with it on,
  the memo hits 7/8 (only the first layer per token recomputes).

- **#5 Python cache-lane bookkeeping — RULED OUT (E5/E6).** cProfile of 30 decode
  steps totals 0.281 s (T=1024) vs 0.294 s (T=4096) — the +4.6 % is the index/kv-
  source `Indexer.select` O(T) sort only (already attributed, ~10 ms/token). Per-fn
  `_sparse_attend_selected` tottime is identical (0.004 s) at both T; per-reuse-layer
  host wall is FLAT in T (923 → 916 → 915 µs at T=256/1024/4096). No host T-scaling.

## Evidence (CPU reproduction, tiny fake config, no artifact/GPU, T ≤ 4096)

Tiny `ModelArgs` (4 heads, head_dim 16, 8 layers, `compress_ratios=[0,0,2,2,2,1,1,1]`,
kv-source {2,5}, index-source {2,5,6} → reuse {3,4,7}), `mx.set_default_device(mx.cpu)`,
cell16k decode env. Capped at T ≤ 4096 after the box guard killed a T=16384 real-shape
run at 4.3 GB. The host-side candidates reproduce at any size (counted, not timed).

| Exp | Result | Bears on |
|-----|--------|----------|
| E1 retrace | attn tapes trace **1×**, replay across 32 steps (T 64→96); cache size 4 | #1 ruled out |
| E2 memo | `SELECTED_KEYS=1`: `_window_attend` **0 calls**; off: 70/80 hits | #2 ruled out |
| E3 T-source | gather sources scale with T (`window_all` 16→262 KB, `compress_kv` 8→131 KB, T=256→4096); `comp_idx` bounded | source *shapes* are O(T)… |
| E5 cProfile | 30-step decode 0.281 s (1K) vs 0.294 s (4K); `_sparse_attend_selected` 0.004 s both | …but not host-costly (#5 ruled out) |
| E6 host wall | per-reuse-layer host wall FLAT: 923 → 916 → 915 µs | …and not host-scaling |
| macmon (GPU, coordinator) | 71 °C, ~45 % busy, 580–1381 MHz swing @ 8.2 W during 16K decode | DVFS floor (rank 1) |

## `MTPLX_DSV41_ATTN_SHAPE_STABLE` (default OFF) — a dispatch-count cleanup, not the floor fix

All Reuse/Reindex/Full layers of a group read the SAME `(compress_kv, selected_idx)`
pair, so the shipped K30 path re-issues the compressed-lane gather (~3 tiny host
dispatches) once per layer. This lever gathers it **once per source** and shares the
bounded `[b,s,k,hd]` result (identity-keyed on the per-forward `SharedAttentionRuntime`),
so only the first layer of a group issues it.

- **Effect: a per-token dispatch-count reduction, expected ≤ ~1 % of the token.** On
  the tiny 8-layer config the T-scaling compress gathers drop **6 → 3 / token**
  (measured); on the real ~40-layer backbone the compress-referencing layers collapse
  to the few distinct index-source groups (estimate ≈ **38 → 8** references/token,
  ~150 fewer tiny dispatches/token). It does **not** address the DVFS floor.
- **Byte-identical**: a pure caching of the deterministic K30 gather (same rows, same
  order). Verified `mx.array_equal` over **64 decode steps** (max|Δ|=0), an **s=4
  verify-shaped** forward, and prefill (one-shot + chunked), composed with
  `SELECT_FENCE`, `KV_CHUNK_GROW`, `WINDOW_RING`, and layer-major prefill.
- **Small-M gated (rows = b·s ≤ 8):** armed at decode/verify only. Under layer-major
  chunked prefill a per-chunk cache would pin every chunk's `[chunk,k,hd]` operand for
  the group (~17 GB at the 16K cell, OOM at 64K), so prefill always uses the plain
  per-layer transient gather (freed after the layer) — the review's HIGH-severity fix.
- Exact hit accounting (test): per token, hits == #Reuse layers, misses == #Full+#Reindex.

**A/B arms (append-only, `ab_decode_env_levers.py`):** `attn_shape_stable`
(`selected_keys=1, attn_shape_stable=1`) and `cell16k_ring_stable` (`cell16k_ring` +
`attn_shape_stable=1`).

**Served-log:** the lever (and the previously-missing `SELECT_FENCE`/`WINDOW_RING*`/…)
are in `mtplx/server/openai._DSV41_LEVER_ENV_KEYS`; a test pins `ALL_LEVER_ENVS ⊆`
that snapshot.

## Telemetry added (window 36)

- `util_macmon.py` (shared, mockable): `macmon pipe` background sampler →
  `utilization` block (min/mean/max + per-sample series of gpu freq/power, the two
  distinct macmon occupancy metrics `gpu_usage_ratio` and `gpu_active_ratio`, cpu
  usage, temps; the census "busy %" is `gpu_usage_ratio`) + a one-line census.
  Sudo-free (reuses `/opt/homebrew/bin/macmon` like
  `server_cell_bench.py`), graceful no-op when macmon is absent, `MTPLX_MACMON_BIN`
  override for tests. Unit-tested with the reader mocked.
- `--utilization` + `--util-interval-ms` + `--cooldown-s` on BOTH
  `ab_decode_env_levers.py` and `metal_decode_attn_bisect.py --in-model`. The sampler
  wraps the **timed decode only**; `--cooldown-s` idles after prefill, before decode
  (TTFT, from prefill, unaffected). The bisect's isolated Reuse timing now resets the
  per-forward shared gather cache each iteration so the shape-stable lever is measured
  as a genuine first-touch, not a cross-iteration cache hit.

Window 36: in-model 16K, `--cooldown-s 180` vs `0`, per-CSA-mode census with
`swa_only` as the control, reading the `utilization` block — the discriminator for
the DVFS-downclock floor.
