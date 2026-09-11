# W30 — Layer-major chunked prefill (kernel-ledger K16): read the bank once across chunks

Branch `feat/deepseek-v41-w30` off `feat/deepseek-v41-streaming` @ `106fdd48e`.
Files: `mtplx/models/deepseek_v41.py`,
`tests/models/test_deepseek_v41_layer_major_prefill.py` (new),
`tests/models/test_deepseek_v41_dspark.py` (one added gate), this report,
`docs/deepseek-v41/KERNEL_LEDGER.md` (§K16 status line). CPU-only, tiny synthetic
configs, no artifact loaded. Peak RSS of the K16 test file: **0.11 GB**
(`/usr/bin/time -l` maximum resident set size 118,292,480 B).

## Verdict

W20 chunked prefill is **chunk-major**: every chunk runs through all 40 layers,
so each chunk's MoE re-streams ~the whole routed bank per layer — up to **~13×**
the ~20 s bank read at 16,384 tokens (13 chunks at the auto chunk 1271). K16
restructures the chunked forward to **layer-major**: iterate every layer over all
chunks before the next layer, so a layer's routed experts stream **once** across
the whole prompt. Proven on CPU: layer-major logits and full cache state equal
one-shot within the existing chunked tolerance (≤ 1e-5), a counting fake switch
shows each `(layer, expert)` is fetched **once** (vs C=chunks under chunk-major),
the engram history and every KV lane end identical to one-shot, and DSpark
spec == AR still holds under the layer-major schedule.

**Chosen: option (a)** — concatenate the chunks' MoE inputs per layer into **one
`switch_mlp` call** (row-capped). Option (b) — per-chunk calls relying on the
expert cache retaining a layer's experts across consecutive chunks — **cannot
work on this streaming bank** (see §3).

## 1. What changed (schedule, not math)

The one-shot path and W20's chunk-major driver are untouched and stay the
default; layer-major is an **opt-in** mode (default OFF — see §6). The decoder
layer is split at the MoE seam so a layer can run its attention half for every
chunk before its one MoE call:

* `DecoderLayer.attn_and_moe_input(h, pre_mix, positions, layer_cache, shared)`
  — the layer up to and including the routed-expert input projection (the
  attention Hyper-Connection **writes this layer's KV** here), returning
  `(moe_input, carry, ffn_pre)`.
* `DecoderLayer.moe_combine(moe_output, carry)` — the ffn Hyper-Connection `post`
  that folds the routed output back in.
* `DecoderLayer.__call__` now composes those two around `self.mlp(...)` — the
  one-shot / chunk-major path is **byte-for-byte** the old body (same ops, same
  order).

`DeepseekV41Backbone._forward_layer_major(input_ids, cache, chunk)` is the new
driver. For a prefill span it:

1. builds each chunk's resident state once — the `[b, chunk, hc_mult, hidden]`
   Hyper-Connection stream, the identity `pre_mix`, the absolute `positions`
   (from the entry offset, so no per-chunk `cache.advance` is needed), a per-chunk
   `SharedAttentionRuntime`, and — advancing the engram history **in position
   order** — each chunk's captured engram row ids;
2. iterates layer `L` over chunks `c = 0..C-1`: runs the engram hook (with a
   per-chunk `_ChunkEngramView` carrying that chunk's row ids), captures the
   DSpark `main_hidden` if `L` is a target, runs `attn_and_moe_input`, and
   **evals the chunk's `moe_input` + this layer's KV stores** so its
   `[chunk, H, T]` attention score frees before the next chunk's graph is built
   (only one score live at a time);
3. issues **one** MoE call over the concatenated chunk inputs (row-capped, §4),
   splits the result back per chunk, and `moe_combine`s each;
4. after all layers, `cache.advance(s)`, collapses each chunk's hc copies with its
   final `pre_mix`, RMSNorms, and concatenates outputs (and `main_hidden` parts)
   in position order.

### Why it is exact (vs one-shot)

* **Attention stays causal and per chunk.** Within a layer the chunks run in
  order `0..C-1`; chunk `c` appends its post-RoPE KV to the same append-only
  layer store and reads the accumulated window, so it attends over chunks `< c`
  exactly as one-shot — identical to how chunk-major's later spans see earlier
  spans, just reordered. The `[chunk, H, T]` fp32 score that motivated W20 is
  **never concatenated** — it stays one chunk wide.
* **The per-chunk `SharedAttentionRuntime`** threads a chunk's compressed-KV /
  index-selection down the layer stack exactly as its W20 span would (a
  kv_source layer publishes for chunk `c`, a later reuse layer reads chunk `c`'s
  slot); one runtime per chunk, kept across the whole layer loop.
* **MoE is per-token.** `MoE.__call__` reshapes to `[rows, dim]` and routes /
  gathers / weights / shares per row; concatenating chunks' rows into one call
  changes only the M (row) dimension, not any per-row reduction, so each row's
  routed sum is unchanged. Residual is fp32 matmul-tiling noise (batch-vs-span M
  tiling), the same class W20 documents — measured ≤ ~1.6e-6 at the test scale,
  100× under the port's 1e-3 attention bar.
* **Engram order preserved.** The history is advanced once per chunk in position
  order (identical `_buf`/`_len` to chunk-major), and each chunk's captured row
  ids are replayed to the hook — the shared `_current` cache only holds the last
  advance, so under layer-major it is never read; the per-chunk view supplies it.
* **DSpark.** `main_hidden` captures the target-layer input per chunk and
  concatenates in position order, so the draft-history seed
  (`prompt_hidden[:, :-1, :]`) still spans the whole prompt and the draft's
  `[:, -1:, :]` slice is still the final prompt token.

## 2. Fetch count — the K16 claim (measured, tiny config)

`test_layer_major_reads_each_expert_once_vs_chunk_major` installs a **counting
fake switch** (records, per call, the unique expert ids it is asked to gather —
exactly what `partition_route_waves` streams once per unique expert per call) on
an 8-layer config, prompt `s = 25`, chunk `7` → **C = 4 chunks**:

| metric | chunk-major (W20) | layer-major (K16) |
|---|---:|---:|
| `switch_mlp` calls total | **32** = 8 layers × 4 chunks | **8** = 8 layers × 1 |
| rows per call (max) | 7 (= chunk) | 25 (= whole prompt) |
| distinct `(layer, expert)` fetched | 56 | 56 (same set) |
| **max fetches per `(layer, expert)`** | **4** (= C) | **1** |
| min fetches per `(layer, expert)` | 1 | 1 |

Hard assertions: `max(layer_major) == 1`, `max(chunk_major) == C == 4`, same
`(layer, expert)` set. At the released geometry (16,384 tokens, auto chunk 1271 →
**13 chunks**, 40 layers × 384 experts, top-6) the same structure collapses the
bank-read multiplier **~13× → 1×**: chunk-major issues ~`40 × 13` switch calls
whose union re-gathers a hot expert up to 13 times; layer-major issues `40` calls,
each gathering the layer's experts once — matching KERNEL_LEDGER §3.2's
`~260 s → ~20 s`.

### Why NOT option (b)

`partition_route_waves(max_unique_experts=plan.transient_slots)` bounds a call's
waves by the transient slot pool, which **defaults to `top_k` (6)**
(`mtplx/expert_runtime.py`) — six experts resident at a time, released between
waves. The routed bank is 269 GiB against a 100 GiB knob, so a layer's 384
experts are **streamed**, never all resident. So per-chunk calls (option b) would
re-stream a layer's experts on every chunk regardless of an intervening cache —
the cache holds ≪ 384/layer — and a counting fake switch (no real cache) could
not even model retention. Option (a) makes the read-once structural and provable.

## 3. Memory arithmetic (analytical, released geometry)

Released config: `hidden = 5120`, `hc_mult (streams) = 4`, `H = 64`,
`top_k = 6`, `n_routed = 384`.

**Resident Hyper-Connection state** (the "keep every chunk's hc state resident"
term — all chunks' `[b, chunk, hc_mult, hidden]` bf16 held across the whole layer
loop = the whole prompt): `hidden × streams × tokens × 2 B`.

| tokens | resident hc state | note |
|---|---:|---|
| 1,024 | 5120·4·1024·2 = **41,943,040 B ≈ 0.042 GB** | (1,024 stays one-shot; arithmetic only) |
| 16,384 | 5120·4·16384·2 = **671,088,640 B ≈ 0.671 GB** | **< 1 GB** — the design bound |

The ffn `carry` (post-attention residual) is the same size (0.671 GB) but is
**transient within a layer** — built per chunk, dropped after `moe_combine` — not
resident across layers. `pre_mix` is `5120·4·16384·... ` no — `[b, s, streams]`
fp32 = 262,144 B, negligible.

**MoE routed-output transient** (one call at 16 K): `rows × top_k × hidden × 4` =
16384·6·5120·4 = **2,013,265,920 B ≈ 2.01 GB** — bounded, and freed once the
combine evaluates. **Attention score per chunk** (unchanged from W20): 1271·64·
24576·4 = 7.996 GB, one chunk live at a time. So the layer-major peak beyond the
~10 GB resident weights and ~1 GB accumulating KV is a single 8 GB attention
score **or** the 2 GB routed transient (they do not coexist — the MoE runs after
all chunks' attention halves have been evaluated and freed) — comfortably under
the 100 GB knob. (The real GPU peak/TTFT is the KG-b window's measurement.)

**Row cap.** `_derive_moe_row_cap` = `floor(budget / (top_k·hidden·4))` =
`8e9 / 122880` = **65,104 rows** at the default 8 GB budget. The 16,384-token
prompt is far under it → one MoE call per layer (bank read once). Only beyond
~65 K tokens does the concatenated call split into ≤-cap groups (each group
re-reads its own expert union — a bounded-transient fallback; attention is
already chunked finer there). `MTPLX_DSV41_PREFILL_MOE_TARGET_GB` overrides the
budget (floored at 1 GB).

## 4. Tests (`tests/models/`, CPU, tiny synthetic; no artifact)

`test_deepseek_v41_layer_major_prefill.py` — **9 tests**:

1. `test_layer_major_matches_one_shot` — logits **and** full cache state (window /
   compress_kv / index_k / compressor frontier / offset) == one-shot (≤ 1e-5)
   across chunks `{1,2,3,5,7,8,9,13,24,25}` (non-divisors and the ratio-2 /
   window straddlers included).
2. `test_layer_major_matches_one_shot_swa_only` — same for a pure sliding-window
   model (no CSA).
3. `test_layer_major_reads_each_expert_once_vs_chunk_major` — the §2 fetch-count
   gate (hard: layer-major max fetch == 1, chunk-major == C).
4. `test_layer_major_engram_and_kv_identical_to_one_shot` — with a synthetic
   `NgramHashState` + a fake engram hook that reads this chunk's row ids: the
   engram history buffer (length **and** contents), every KV lane length/value,
   and the engram-wired logits all == one-shot.
5. `test_layer_major_flag_resolution` — env / arg precedence, default OFF.
6. `test_layer_major_is_inert_on_short_prompt_and_decode` — one-shot byte-identical
   with the flag on vs off.
7. `test_moe_row_cap_bounds_routed_transient_at_real_geometry` — cap == 65,104,
   16 K prompt one call, routed transient == 2.01 GB < 8 GB.
8. `test_resident_hc_state_is_well_under_1gb_at_16k` — 0.042 GB / 0.671 GB, < 1 GB.
9. `test_layer_major_moe_row_cap_splits_and_reassembles_exactly` — the row cap
   splits the concatenated call, reassembles the same rows as a single call
   (≤ 1e-5), and no split call exceeds the cap.

`test_deepseek_v41_dspark.py::test_dspark_lossless_under_layer_major_prefill`
(added) — `generate_mtpk` depth 3 with `MTPLX_DSV41_PREFILL_CHUNK=4` and
`MTPLX_DSV41_PREFILL_LAYER_MAJOR=1`: spec == AR.

**Verification (CPU, `nice -n 19`, `mx.set_default_device(mx.cpu)`, no `-n auto`):**

- `test_deepseek_v41_layer_major_prefill.py` — 9 passed (peak RSS 0.11 GB).
- `test_deepseek_v41_dspark.py` — 13 passed (incl. the new layer-major gate).
- Full `tests/models/test_deepseek_v41_*.py` + `tests/test_deepseek_v41_*.py` —
  **166 passed, 39 skipped** (artifact-gated), **1 deselected**:
  `test_deepseek_v41_streaming_clamp.py::test_spec_swiglu_limit_values` is a
  **pre-existing** registry-spec red (a second key,
  `deepseek-v41-flash-expert-mxfp4`, also carries `swiglu_limit=10.0`) — it fails
  identically with this branch's `deepseek_v41.py` change stashed, imports none of
  W30's code, and is not attributable to this change.

## 5. Not changed

- `deepseek_v41_cache.py` — unchanged (frozen API: offset, advance, layers,
  new_shared_runtime, engram_state; the layer store is append-only and order-safe).
- `deepseek_v41_moe.py` / the streamed switch (`expert_mlx.py`) / `expert_runtime.py`
  — unchanged; layer-major hands the seam a wider `[rows, hidden]` and relies on
  `partition_route_waves`'s existing per-call dedup.
- W20's chunk-major driver and one-shot path — unchanged and still the default.

## 6. Default OFF — why, and the follow-up

Layer-major is opt-in (`MTPLX_DSV41_PREFILL_LAYER_MAJOR` env / `prefill_layer_major`
kwarg, default OFF) so:

- the one-shot and chunk-major paths — and every existing W20 test, incl.
  `test_forward_feeds_moe_at_most_chunk_rows` (which asserts chunk-major's
  `calls == layers × chunks` and `max_rows == chunk`) — stay green unchanged;
- the ~13× 16 K-TTFT win is a **GPU-window measurement** (KERNEL_LEDGER **KG-b**:
  16 K one-shot bank read A/B, pass if TTFT −≥ 40 %, byte-identical, peak under
  the knob) — flipping the serving default belongs with that receipt, not a
  CPU-only exactness task.

KG-b runs the layer-major branch (this flag on) against chunk-major; on a pass a
one-line default flip (and an owner update to the W20 MoE-feeding test) lands it
in serving.
