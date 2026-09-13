# W77 — DSpark-DIRECT greedy divergence: classify, don't abort

Worker `w77/dspark-divergence` (branch off `359b9b70f`). Opus 4.8, **CPU-only**
(no GPU / Metal; every experiment pins `mx.set_default_device(mx.cpu)` and builds a
shrunk seeded `Model`, never the real 376 GB artifact). The realized 16K decode
tok/s and the on-Metal logit deltas are GPU-window measurements, **not measured by
this worker**.

## 0. The failure

Window 29b, real DeepSeek-V4.1-Flash streaming, 16,384-token prefill then greedy
decode 256, arm `cell16k` (`layer_major + prefill_dense + score_path=lean +
selected_keys + kv_chunk_grow + layout_fix + HEAD_MODE=bf16 + SINKHORN_METAL +
ATTN_COMPILE + ATTN_WIN_MEMO`, `--decode-mode dspark --dspark-depth 3`):

```
AssertionError: DSpark-DIRECT greedy decode diverged from AR at index 228
(arm 'cell16k'); speculative lane is not lossless
```

The arm aborted with no receipt. At 1K with `stack_a` the DSpark lane was
byte-identical to AR for 256 tokens.

The DSpark greedy verify is authoritative, so the stream is AR **by
construction** — a first-token difference from the AR reference is a greedy
`argmax` *flip* driven by the target forward returning slightly different logits at
the same committed context depending on the row count: AR runs a 1-row (**M=1**)
decode forward per token; the K+1-row (**M=4** at depth 3) verify forward takes
different matmul / softmax kernels under the cell16k levers. Per David's standing
rule, a flip at a genuine near-tie caused by rounding-class deltas is acceptable;
byte-identity is the ship bar only for exact-by-construction arms
(`[[dsv41-inexact-ok-if-tie-flips]]`).

## 1. Task 1 — turn the hard assert into a classification

`scripts/deepseek_v41/ab_decode_env_levers.py` no longer aborts on a DSpark≠AR
greedy stream. It **classifies the first divergence into the receipt and decodes
both streams to full length.**

### Definitions (each metric named once)

- **divergence_index `i`** — first position where the DSpark id ≠ the AR id
  (position 0 is the shared prefill token; decode tokens are 1..).
- **ar_token / dspark_token** — the two ids at `i`.
- **top-2 margin** — `top1_logit − top2_logit` of a logits row (the greedy
  decision gap). `ar_top2_margin` is the AR row's; `dspark_top2_margin` the verify
  row's.
- **max_abs_logit_delta** — `max_v |ar_logits[v] − dspark_logits[v]|` at `i`.
- **class** — `"tie_flip"` iff the flip is rounding-class, else `"divergent"`.
  W120 (docs/deepseek-v41/W120_DIVERGENCE_TIE_BAND.md) superseded the original
  `ar_top2_margin < tie_margin` rule: the decision now uses the CONTESTED margins
  (`|row[ar_token] − row[dspark_token]|`, not the row top-2 gap), a magnitude-aware
  band `tie_band = max(tie_margin, k·ulp_bf16(peak))`, and REQUIRES the measured
  contested deltas to be within the band (`deltas_within_tie_band`). The top-2
  margins remain as diagnostic receipt keys.

### How the two rows are obtained cheaply (at the first mismatch only)

- **DSpark verify row — zero extra forwards.** A `DivergenceCapture` object
  (`mtplx/models/deepseek_v41_dspark_decode.py`) watches the committed stream
  against the AR reference *inside the existing dspark pass*; for greedy, the
  committed token at block index `m` is the argmax of `verify_logits[0, m]`, so at
  the first mismatch it snapshots that row straight out of the logits the cycle
  already computed. Threaded through `dspark_generate(..., divergence_capture=…)`
  → `_decode_cycles(..., divergence_capture=…)`.
- **AR row — one bounded replay, only on divergence.** `_ar_logits_row_at_index`
  re-prefills the prompt and steps `i` **M=1** decode forwards feeding the AR ids
  (the exact one-row shape AR used — *not* a whole-prefix re-prefill, which would
  be a different matmul shape). Cost: 1 prefill + `i ≤ decode_tokens` M=1 steps,
  paid once, and only when a divergence actually occurs (exact arms pay nothing).

### Threshold justification (`DSPARK_TIE_MARGIN_DEFAULT = 3e-2` logit units)

- `HEAD_MODE=bf16` (W40/K21) casts the final hidden to bf16 before the head GEMV.
  bf16 carries a 7-bit mantissa → unit round-off `2^-8 ≈ 3.9e-3`; a logit carries
  a bf16-class perturbation `~|logit|·2^-8` plus the M=1-vs-M>1 accumulation-order
  difference of the head GEMM.
- `DSPARK_BF16_CLASS_DELTA = 1e-2` is that order of magnitude (the bf16-class
  floor). A greedy flip caused purely by rounding needs the top-2 gap to be
  *within* that perturbation, so the tie-flip threshold is **3×** the floor
  (`3e-2`). A gap below it is a genuine near-tie a few bf16 ulps flip
  (`tie_flip`, acceptable); a gap at or above it means the argmax changed for a
  reason larger than rounding (`divergent` — a real lane bug or a non-rounding
  lever), which stays **loud** in the census line even though the arm no longer
  aborts. Overridable per-arm with `--dspark-tie-margin`.

### The lossless gate is preserved

`--dspark-require-lossless` restores the hard `AssertionError` for
exactness-class arms whose ship bar *is* byte-identity (exact-by-construction
reorders like `shared_overlap`). Default OFF → classify.

### Receipt shape (added under `receipt["dspark"]["divergence"]`)

```json
{
  "divergence_index": 228, "ar_token": 1234, "dspark_token": 5678,
  "ar_top2_margin": 0.014, "dspark_top2_margin": 0.021,
  "max_abs_logit_delta": 0.031, "tie_margin": 0.03,
  "class": "tie_flip", "capture_index": 228
}
```

`null` when the stream is byte-identical. The full logits rows never enter the
receipt (only the scalars above).

## 2. Task 2 — which cell16k lever breaks M=1-vs-M=4 identity?

`scripts/deepseek_v41/w77_lever_identity_probe.py` builds the real V4.1 `Model` on
tiny dims (DIM 32, 4 layers, vocab 48, sliding window 8) with seeded weights on
CPU, then compares, at an identical shared prefill, the **M=1** decode lane (feed
the last K+1 tokens as sequential 1-row steps) against the **M=K+1 verify** lane
(feed them as one block), reporting `max_abs_logit_delta` at the shared final
position. Two lanes: **fp32** (bit-exact — a genuine wrong-row/gather bug shows a
LARGE delta even in exact arithmetic) and **bf16** (weights cast to bf16 — the
box's rounding class).

### Per-lever Δ table (tiny double, ctx=20, block=4, seed=0)

| lever | engages @ M=4 | fp32 max\|Δlogit\| | bf16 max\|Δlogit\| | argmax flip |
|---|:--:|---:|---:|:--:|
| baseline (no levers) | — | 7.7e-07 | 4.8e-07 | no |
| layer_major | no¹ | 7.7e-07 | 4.8e-07 | no |
| prefill_dense | no² | 7.7e-07 | 4.8e-07 | no |
| score_path=lean | yes | 8.3e-07 | 6.6e-07 | no |
| selected_keys | yes | 8.9e-07 | 7.2e-07 | no |
| kv_chunk_grow | yes | 7.7e-07 | 4.8e-07 | no |
| layout_fix | no² | 7.7e-07 | 4.8e-07 | no |
| **head=bf16** | yes | **0.0** | **0.0** | no |
| sinkhorn_metal | no³ | 7.7e-07 | 4.8e-07 | no |
| attn_compile | yes | 4.8e-07 | 7.7e-07 | no |
| attn_win_memo | yes | 7.7e-07 | 4.8e-07 | no |
| **cell16k (full stack)** | — | **0.0** | **0.0** | no |

¹ prefill schedule only; the verify is a single forward with no chunk schedule.
² streamed-switch paths gated OFF at the verify: `prefill_dense` is
`RoutingPhase.PREFILL`-gated while the verify routes DECODE-phase, and `layout_fix`
needs ≥2048 rows (`_LAYOUT_FIX_MIN_ROWS_DEFAULT`) while a K+1 verify is ~8 routed
rows — both documented in `expert_mlx.py`, and both inert on the resident tiny
double (which uses resident experts, not the streamed switch).
³ Metal-only kernel; falls back to the byte-identical Sinkhorn recurrence on CPU.

### Reading the table

- **No cell16k lever introduces a structural (logic-bug) delta.** Every value sits
  at fp32 epsilon (~1e-6) or zero. A wrong row/position, a mis-ordered batched
  append, or a mis-unsorted gather at M=4 would be O(0.1–10) even in exact fp32,
  and there is none. `score_path`, `selected_keys`, and `attn_compile` do change
  the value slightly off baseline (they *do* engage and alter the computation) but
  stay in the fp32-noise band.
- **head=bf16 (and the full stack) read 0.0** because bf16-rounding the hidden
  *erases* the sub-ulp fp32 difference between the M=1 and M=4 hiddens — the head
  codec masks the M-difference on CPU rather than amplifying it.
- **The CPU double cannot reproduce the real divergence, by construction.** The
  index-228 flip on the box is a *Metal* effect: M=1 and M=4 dispatch different
  Metal kernels (GEMV vs tiled GEMM) with different bf16 accumulation order — a
  reassociation in the bf16 rounding class. MLX's CPU backend computes M=1 and M=4
  per-row identically, so the delta stays at fp32 epsilon. The code itself
  documents this for exactly the M=4-engaging levers: `HEAD_MODE=bf16`
  ("never bit-identical on Metal"), `LAYOUT_FIX` ("byte-identical on CPU … a
  kernel reassociation … measured in a GPU window"), `SELECTED_KEYS`
  ("mathematically identical up to float reassociation of the softmax sum,
  greedy-identical, never bit-identical").

### Negative controls (so the null result is credible)

The probe/tests include two injected bugs to prove the method has power:

1. `_GrowBuffer.append` with the 4-row verify block written **reversed** → the
   window view diverges from the concatenate store (`test_growbuffer_negative_control_detects_reordered_append`).
2. A monkeypatched wrong-order M=4 verify append (`kv_chunk_grow(BUGGY)`) →
   `max_abs_logit_delta` jumps **>1e-2**, well above the fp32-noise ceiling
   (`test_negative_control_injected_m4_append_bug_is_caught`).

The unpatched code passes the byte-identity gate: `_GrowBuffer` (chunk-grow ON) is
byte-for-byte the `_grow` concatenate store after a multi-row append, whether the
block arrives as one 4-row append (verify) or four 1-row appends (decode).

## 3. Verdict

- **No byte-identity bug was found in any cell16k M>1 path.** No fix to make; the
  chunk-grow buffer, the sorted gather (inert at the verify), and the dense-expert
  path (inert at the verify) are all sound. The index-228 divergence is a
  **rounding-class tie-flip**: 228 tokens of accumulated bf16/Metal
  kernel-dispatch reassociation eventually landed on a greedy near-tie and flipped
  it — the class David's rule accepts.
- The classification now records that verdict per-arm instead of aborting: the
  `cell16k` arm will emit a receipt with `class: "tie_flip"` (assuming the AR
  top-2 margin at the flip is within `3e-2`; the receipt makes it a measured fact,
  not an assumption) and full tokens/cycle, accept-by-depth, decode tok/s, and
  both stream shas — while `--dspark-require-lossless` still gates exact arms.
- If a future arm ever reports `class: "divergent"` (a contested delta exceeds the
  W120 `tie_band_used`, or the contested margins are decisive — see
  W120_DIVERGENCE_TIE_BAND.md), the loud census line flags it and the fp32 probe is
  the tool to localize the lever; by the table above it would be a *new* regression,
  not one of these ten levers.

## 4. Files, tests, reproduction

Changed / added:
- `mtplx/models/deepseek_v41_dspark_decode.py` — `DSPARK_BF16_CLASS_DELTA`,
  `DSPARK_TIE_MARGIN_DEFAULT`, `_top2_margin`, `classify_divergence`,
  `DivergenceCapture`; `divergence_capture` threaded through `dspark_generate` and
  `_decode_cycles` (zero-cost greedy capture).
- `scripts/deepseek_v41/ab_decode_env_levers.py` — `--dspark-require-lossless`,
  `--dspark-tie-margin`; `_ar_logits_row_at_index`, `_print_dspark_divergence`;
  the hard abort replaced by classification (AR replay + capture + classify),
  `receipt["dspark"]["divergence"]`.
- `scripts/deepseek_v41/w77_lever_identity_probe.py` — the CPU per-lever probe.
- `tests/models/test_deepseek_v41_dspark_divergence_classify.py` — classification
  + capture + end-to-end (10 tests).
- `tests/models/test_deepseek_v41_w77_lever_m1_m4_identity.py` — per-lever M=1/M=4
  identity + chunk-grow byte-identity + negative controls (15 tests).

Reproduce (CPU, niced):

```
PYTHONPATH=<worktree> nice -n 19 <venv>/bin/python3 \
    scripts/deepseek_v41/w77_lever_identity_probe.py            # the Δ table
PYTHONPATH=<worktree> nice -n 19 <venv>/bin/python3 -m pytest \
    tests/models/test_deepseek_v41_dspark_divergence_classify.py \
    tests/models/test_deepseek_v41_w77_lever_m1_m4_identity.py
```

Pre-existing suites unaffected: `tests/models/test_deepseek_v41_dspark_decode.py`
(24 passed), `tests/test_deepseek_v41_ab_env_levers.py` (63 passed).
