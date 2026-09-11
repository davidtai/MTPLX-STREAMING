# W30 — Layer-major chunked prefill (kernel-ledger K16): read the bank once across chunks

Branch `feat/deepseek-v41-w30` off `feat/deepseek-v41-streaming` @ `106fdd48e`.
Files: `mtplx/models/deepseek_v41.py`, `mtplx/models/deepseek_v41_moe.py`
(`combine_routed` split; §8 fix),
`tests/models/test_deepseek_v41_layer_major_prefill.py` (new),
`tests/models/test_deepseek_v41_dspark.py` (one added gate), this report,
`docs/deepseek-v41/KERNEL_LEDGER.md` (§K16 status line). CPU-only, tiny synthetic
configs, no artifact loaded. Peak RSS of the K16 test file: **0.11 GB**
(`/usr/bin/time -l` maximum resident set size 118,292,480 B); every worker
process ≤ 3 GB.

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

**Chosen: option (a)** — batch the chunks' rows into **one `switch_mlp` call** per
layer (row-capped) so the streamed bank is read once. Option (b) — per-chunk
calls relying on the expert cache retaining a layer's experts across consecutive
chunks — **cannot work on this streaming bank** (see §3).

> **Addendum (2026-09-11, GPU window 14 follow-up).** The first cut batched the
> **whole** MoE per layer, including the *resident* router gate and shared expert.
> Those two are not invariant to the row (M) batch size, so batching them flipped
> a greedy top-k near-tie and diverged one token on the real model. Fixed: the
> resident gate + shared expert now run **per chunk** (byte-identical routing);
> only the streamed `switch_mlp` is batched. Layer-major is now **byte-for-byte
> equal to chunk-major** on CPU (`mx.array_equal`, greedy argmax, full cache
> state). See **§8**.

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
3. runs the MoE with the **resident gate + shared expert per chunk** and **one
   row-capped `switch_mlp` call** over the concatenated rows (bank read once; the
   per-chunk gate/shared is the §8.3 byte-identity fix), splits the routed output
   back per chunk, and `moe_combine`s each;
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
* **MoE routing is byte-identical (post-fix).** The resident gate and shared
  expert run per chunk (M == the chunk, exactly as chunk-major), so `(weights,
  indices)` and the shared output are byte-identical; only the streamed
  `switch_mlp` is batched across chunks, and that per-expert gather is M-invariant
  (byte-identical on CPU). The first cut batched the whole MoE — including the
  resident gate — and its ~1e-6 gate-score perturbation flipped a greedy near-tie
  on the real model; §8 has the root cause and this fix. Net: layer-major is now
  **byte-for-byte == chunk-major** on CPU (chunk-major itself carries W20's
  ≤ 1e-5 attention-tiling noise vs one-shot, which layer-major inherits exactly).
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

`test_deepseek_v41_layer_major_prefill.py` — **10 tests**:

1. `test_layer_major_matches_one_shot` — logits **and** full cache state (window /
   compress_kv / index_k / compressor frontier / offset) == one-shot (≤ 1e-5)
   across chunks `{1,2,3,5,7,8,9,13,24,25}` (non-divisors and the ratio-2 /
   window straddlers included).
1b. `test_layer_major_is_byte_identical_to_chunk_major` (§8 strict gate) —
   layer-major == chunk-major **byte-for-byte** (`mx.array_equal` logits, greedy
   argmax at every position, full cache state) across chunks `{1,3,7,8,13,16}`,
   prompts 25/40, weight scales 0.1/0.3. This is the token-flip regression gate.
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
   splits the streamed `switch_mlp` call and reassembles **byte-identically**
   (`mx.array_equal`) to a per-chunk `mlp(chunk)`, and no split call exceeds the
   cap.

`test_deepseek_v41_dspark.py::test_dspark_lossless_under_layer_major_prefill`
(added) — `generate_mtpk` depth 3 with `MTPLX_DSV41_PREFILL_CHUNK=4` and
`MTPLX_DSV41_PREFILL_LAYER_MAJOR=1`: spec == AR.

**Verification (CPU, `nice -n 19`, `mx.set_default_device(mx.cpu)`, no `-n auto`;
each file in its own process under the 3 GB worker cap):**

- `test_deepseek_v41_layer_major_prefill.py` — 10 passed (peak RSS 0.11 GB).
- `test_deepseek_v41_dspark.py` — 13 passed (incl. the new layer-major gate).
- Every `tests/models/test_deepseek_v41_*.py` + `tests/test_deepseek_v41_*.py`
  run one file at a time — all **passed or skipped** (artifact-gated) except the
  one **pre-existing** red `test_deepseek_v41_streaming_clamp.py::
  test_spec_swiglu_limit_values` (a second key `deepseek-v41-flash-expert-mxfp4`
  also carries `swiglu_limit=10.0`) — it fails identically with this branch's
  `deepseek_v41.py`/`deepseek_v41_moe.py` changes stashed, imports none of W30's
  code, and is not attributable to this change. `moe.py`'s `combine_routed`
  refactor keeps `MoE.__call__` byte-identical (parity + moe suites green).

## 5. Not changed

- `deepseek_v41_cache.py` — unchanged (frozen API: offset, advance, layers,
  new_shared_runtime, engram_state; the layer store is append-only and order-safe).
- `deepseek_v41_moe.py` — `MoE.combine_routed` split out of `__call__` (which
  stays byte-identical); the gate/shared/switch seams themselves are unchanged.
  The streamed switch (`expert_mlx.py`) / `expert_runtime.py` are unchanged;
  layer-major hands the seam a wider `[rows, hidden]` and relies on
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

## 8. Addendum — GPU window 14 divergence: root cause, fix, and the residual

GPU window 14 (integration @ `7770960a9`, 16,384-token prompt, 32 greedy decode,
`MTPLX_DSV41_PREFILL_CHUNK=1024`) ran control (chunk-major) vs `layer_major`.
Receipt: `docs/deepseek-v41/receipts/gpu-windows/window-14/ab-16384-k16.json`.

| arm | TTFT (s) | prefill tok/s | decode tok/s | peak (GB) | token sha |
|---|---:|---:|---:|---:|---|
| control (chunk-major) | 494.88 | 33.11 | 1.4376 | 71.91 | `eb4c163a…` |
| layer_major (first cut) | **372.98** | **43.93** | 1.3007 | 76.60 | `9a69b9f9…` |

**The prefill win is real:** TTFT −24.6 % (494.88 → 372.98 s), prefill throughput
+32.7 % (33.11 → 43.93 tok/s) — the K16 read-once lever pays off. But the first
cut had two problems.

### 8.1 First differing token = decode position 4

`first_token_ids` differ in exactly one slot (0-indexed **position 4**):

```
control    : [1, 5, 4245, 65, 1449, 23042, 201, 5356, 7854, 70925, 848, 1662, 71134, 271, 1897, 3844]
layer_major: [1, 5, 4245, 65,   18, 23042, 201, 5356, 7854, 70925, 848, 1662, 71134, 271, 1897, 3844]
```

Control **1449** vs layer_major **18** at position 4; every other listed token is
identical. A single greedy argmax flip — the signature of a near-tie tipped by
fp rounding, not a systematic corruption.

### 8.2 Root cause — the *resident* gate/shared matmuls are not M-invariant

CPU localisation (`tests/models/test_deepseek_v41_layer_major_prefill.py` plus a
throwaway probe): with a recording switch capturing every `switch_mlp` input,

- the switch **input** `xf` at layer 0 is **byte-identical** between chunk-major
  and layer-major (max\|Δ\| = 0), and so are the router **indices** — so
  attention, the Hyper-Connections, the engram replay and the compressor frontier
  are all exactly reproduced; the divergence is born **inside the MoE**;
- `SwitchGLU` (the routed-expert gather+GEMM) over 25 rows in one call vs split
  `[7,7,7,4]` and concatenated is **byte-identical on CPU** (max\|Δ\| = 0) —
  the streamed per-expert gather is M-invariant;
- the router **gate** `xf @ weight.T` over 25 rows vs the same split is **NOT**
  byte-identical (max\|Δ\| = 1.9e-6) — a plain resident GEMM whose fp32 reduction
  reassociates with the M (row) tile.

So batching the *whole* MoE across chunks perturbed the gate scores by ~1e-6;
on the real 5120-wide, 384-expert model that is enough to flip a **top-k
selection near-tie**, routing a token to a different expert → a different hidden
state → the position-4 argmax flip. (The shared `Expert` is the same class of
resident GEMM and shares the hazard.)

### 8.3 Fix — gate + shared per chunk, only `switch_mlp` batched

`MoE` grew `combine_routed(routed, weights, xf)` (the weighted routed-sum + shared
expert; `__call__` composes it, byte-identical). `_forward_layer_major`'s MoE step
(`_layer_major_moe`) now:

1. computes the **gate** and keeps `xf` **per chunk** (M == the chunk, exactly as
   chunk-major → byte-identical `(weights, indices)`);
2. concatenates only the rows + indices for the **streamed `switch_mlp`** call
   (row-capped) — the bank is still read once;
3. splits the routed output per chunk and runs `combine_routed` **per chunk**, so
   the shared expert's M is the chunk too.

Result on CPU (`test_layer_major_is_byte_identical_to_chunk_major`): layer-major
== chunk-major **byte-for-byte** — logits (`mx.array_equal`), greedy argmax at
every position, and the full cache state — across chunk sizes `{1,3,7,8,13,16}`,
prompts 25/40, weight scales 0.1/0.3.

**Residual GPU risk (must be confirmed by the next window).** CPU proves the gate
was the flip source and that `SwitchGLU` is M-invariant. On GPU the streamed
native-mxfp4 `gather_qmm` may itself carry a small M-variance, but the **routing
is now byte-identical**, so no expert-selection flip remains — only sub-ULP
rounding in the routed GEMM, far less likely to tip an argmax. If a flip survives,
the switch would need a *gather-once, apply-per-chunk-boundary* mode in the
streamed runtime (`expert_mlx.py`/`expert_runtime.py`, outside this worktree) —
that would make even the routed GEMM's M match chunk-major exactly.

### 8.4 The +4.7 GB peak and slower decode

- **Peak (76.60 vs 71.91 GB, +4.69).** The one batched `switch_mlp` over 16,384
  rows holds a `[16384, 6, 5120]` fp32 routed transient = **2.01 GB** at once,
  vs chunk-major's 16 sequential `[1024,6,5120]` = 0.126 GB calls, plus wider
  per-wave working sets and `prepare_prefill_seed` admitting the layer's expert
  union into empty persistent slots in one shot. Inherent to reading the bank
  once; the **row cap** (`MTPLX_DSV41_PREFILL_MOE_TARGET_GB`, default 8 GB → cap
  65,104 rows) is the knob — lowering it splits the switch call (smaller
  transient, at the cost of re-reading each split's expert union).
- **Decode (1.30 vs 1.44 tok/s).** K16 changes only prefill. The decode **hot-set
  census observes DECODE routes only** (`expert_runtime.py:3146` — "prefill
  routing shape does not predict the decode hot set"), so layer-major does **not**
  change the decode residency policy. What differs is the *post-prefill* seed:
  `prepare_prefill_seed` (`expert_streaming.py:501`) fills empty persistent slots
  by **prompt frequency**, and layer-major ranks over the whole prompt at once
  while chunk-major ranks over the first chunk then finds slots filling — a
  different (not obviously worse) starting resident set that the decode census
  re-optimises over the run. The ~10 % delta over only 32 decode tokens, sitting
  after a 373 s vs 495 s prefill, is within host-encode/thermal drift between arms
  (memory: `cpu-heavy-work-voids-flock-windows`); a clean read needs an
  interleaved-pairs decode A/B, not a single ordered pair. This did not change
  with the §8.3 fix (the switch is still batched for read-once).

### 8.5 Recommendation

Re-run KG-b with the fixed branch: expect the prefill win to hold and the greedy
tokens to now match control (or, if a residual GPU flip remains, it will be the
routed-GEMM M-variance of §8.3, which needs the runtime-side gather-once change,
not a model-side fix). Keep the default OFF until that A/B is byte-clean.
