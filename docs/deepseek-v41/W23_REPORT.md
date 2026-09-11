# W23 — DeepSeek-V4.1 DSpark MTP (speculative draft head) through the MTPLX runtime

Branch `feat/deepseek-v41-w23` (worktree `.worktrees/dsv41-w23`), off
`feat/deepseek-v41-streaming` @ 5d6dd8ad, then merged
`feat/deepseek-v41-w22` (mlx_lm per-layer cache conformance) to build the served
spec path on it. CPU only; peak RSS ≈ **0.12 GiB** (synthetic configs only — the
376 GiB artifact was never loaded, well under the 12 GiB cap).

David's goal (verbatim): *"get the mtp running and get tps over 20, get the mtp
working at a decent clip and minimize k/v memory while maximized streaming moes"*
— through the MTPLX runner's own speculative path, not a bespoke loop.

## Result

DSpark drafts + AR verify run **losslessly on CPU through the runtime's own MTP
path** (`generate_mtpk`): greedy MTP output == greedy AR output over 64 tokens for
K=1, 2, 3, and the served mode is selectable. The head is a faithful MLX
transliteration of the reference DSpark classes; K is capped at the 3 physical
stages (K>3 dead). The **>20 tok/s cost model is written below**: MTP alone does
**not** clear 20 tok/s on the streaming bank — record dedup (R2) + a residency
lever must land alongside it.

## 1. DSpark head — `mtplx/models/deepseek_v41_dspark.py` (new)

A faithful transliteration of the reference `inference/model.py` DSpark classes
(class names + method structure mirrored, reference line refs on each method):

| this module | reference (`inference/model.py`) |
|---|---|
| `DSparkMarkovHead` | `DSparkMarkovHead` L1077-1086 — low-rank per-token logit bias (`embed` vocab→rank, `head` rank→vocab) |
| `DSparkConfidenceHead` | `DSparkConfidenceHead` L1089-1097 — fp32 scalar over `[hidden ; markov_embed]` |
| `DSparkAttention` | `DSparkAttention` L1032-1074 — SWA MLA, KV from the *main* hiddens, Q from the draft; reuses the backbone `Attention` sink-softmax + grouped o-LoRA |
| `DSparkBlock` | `DSparkBlock(Block)` L1100-1156 — 3 stages, `main_proj`/`main_norm` (stage 0), `markov_head`/`confidence_head` (last stage), MTP 128-expert top-3 MoE |
| `DSparkBlock.forward_embed` | L1128-1135 — `main_x`, `[real, noise…]` block, hc expand |
| `DSparkBlock.forward_head` | L1137-1156 — head logits + block-size markov autoregression + confidence |
| `DSparkHead.draft_block` / `.seed_main` | `Transformer.forward_spec` L1274-1282 |
| `DSparkStageCache` | the reference `window_kv_cache` ring L474/L1046-1065 (W23-owned; no MTP-stage cache is contracted) |

Reuse: the entire V4.1 backbone (`Attention` projections, `_sparse_attend`,
`_o_lora_down`, `DecoderLayer`'s Hyper-Connection `_mixes`/`_hc_pre` + the reused
`_hc_post_impl`, and the `MoE` switch seam) is imported from
`mtplx.models.deepseek_v41`; only the DSpark-specific leaves are new.

**Resident expert execution.** The 3 stages' 128 routed experts are RESIDENT
mxfp4 gs32 (W18_REPORT). `mx.gather_qmm` in **mlx 0.32.2 has no `mode=` argument**
(verified), so the resident mxfp4 path is the mlx-lm quantised `SwitchGLU` carrying
the reference clamped SwiGLU (the `MoE` seam this module reuses), **not**
`gather_qmm(mode="mxfp4")`. The DSpark experts are quantised in place
(`Model._build_mtp_head`: mxfp8 dense + mxfp4 experts) and, because the backbone
switch binder walks only `model.model.layers`, they stay resident (never rebound
to the stream bank).

## 2. Integration — the runtime's native MTP path (no bespoke loop)

The runtime MTP path is generic: `MTPLXRuntime.draft_mtp`/`update_mtp_cache`/
`make_mtp_cache` call `model.mtp_forward`/`mtp_update_cache`/`make_mtp_cache`, and
`generate_mtpk` drives the per-depth draft chain + batched verify + accept +
rollback. W23 adds the model-side surface and the dispatch, all mirroring
`deepseek_v4`:

- **`Model` surface** (`deepseek_v41.py`, minimal hook): `__call__(return_hidden/
  emit_logits/logits_keep/input_embeddings)` returns `main_hidden` (the
  concatenated `dspark_target_layer_ids` hiddens, captured in the backbone at the
  attention input, mean over the hc copies — reference L1265-1266);
  `hc_hidden`, `mtp_blocks`, `has_mtp`, `make_mtp_cache`, `mtp_forward`,
  `mtp_update_cache`.
- **block-draft → per-depth adapter.** DSpark drafts a whole block in one
  `forward_spec`; `mtp_forward` runs it on the first depth of a cycle and serves
  the block column-by-column across the runtime's per-depth chain (stash on the
  stage cache). The runtime's greedy target verify is authoritative
  (`generation.py:9137` — "draft cache conditions acceptance only … the reset is
  correctness-free"), so this stays lossless regardless of the adapter.
- **config predicate + injector** `is_deepseek_v41_mtp_config` /
  `inject_deepseek_v41_mtp_support` (publish-only; degrade-to-AR when the head is
  absent), and a **runtime dispatch arm** in `mtplx/runtime.py` after the v4 arm.
  The runtime.py edit is justified: `is_deepseek_v4_mtp_config` never matches a
  V4.1 config, and the generic `inject_mtp_support` builds a qwen3_5 `_MTPModule`
  graft, not this native head — neither is reusable.
- **registry**: base `deepseek-v41` note updated + a `deepseek-v41-mtp` catalog row.
- **loader** `mtp=True` opt-in (`deepseek_v41_loader.py`): `partition_text_residents
  (with_mtp=)` keeps `mtp.*` residents, `resolve_with_mtp` is opt-in (default OFF
  so phase-1 AR is unchanged; `MTPLX_DSV41_MTP=1` / explicit arg opt in, with a
  loud error if the artifact can't honour it), and `Model.sanitize` /
  `_map_mtp_residents` map `mtp.{i}.*` onto the DSpark head paths (dense renames +
  128-expert stacking into the SwitchGLU projections).

## 3. Losslessness (the P3.0 done-when)

`tests/models/test_deepseek_v41_dspark.py` — 11 tests, all green on CPU:

- **`test_dspark_greedy_verify_reproduces_ar_over_64_tokens[1/2/3]`** — through the
  real `generate_mtpk` engine (prefill, per-depth DSpark draft chain, batched
  verify, accept, reject, rollback on W22's conformant cache), the greedy spec
  sequence == the greedy AR sequence over 64 tokens for K=1, 2, 3.
- Sweep evidence (vocab 8, 48 tokens, seeds 0-7): **all 8 seeds lossless**; 6/8
  also exercise both accept and reject (e.g. seed 3: 11 accepted / 34 rejected;
  seeds 0/5 accept nothing — the untrained separate-net head can be degenerate).
- **accept + reject** both exercised and the **engine's acceptance counters
  populate per depth** (seed 3, vocab 8): `accepted_drafts>0`, `rejected_drafts>0`,
  `drafted_by_depth`/`accepted_by_depth` filled, `mtp_forward_calls>0`.
- forward/MTP surface contract; `input_embeddings` rejected; a head-less model
  degrades to AR; the `mtp.*` resident name mapping covers the head parameter tree.

**Acceptance accounting** is exactly the engine's shared counters
(`GenerationOutput.stats`: `accepted_drafts`, `rejected_drafts`, `drafted_tokens`,
`accepted_by_depth`, `drafted_by_depth`, `accept_probability_sum_by_depth`) — the
same the bench receipt / health payload read for every MTP backend, so a DSpark
window reports acceptance the same way as A3B/Hy3/V4.

## 4. Cost model for David's target (report only)

**Inputs.** AR decode = **4.79 tok/s**, SSD-bound, ~**3 GB expert records/token**,
1,024-token shape. Backbone: 40 layers × 384 routed experts, top-6 → 6 records/
layer/token; expert record = 10.55 MiB (2.5 bpw); cold experts/token = 40×6×10.55
MiB = **2.472 GiB** (≈ the ~3 GB figure). A K+1-row verify gathers the **UNION** of
the rows' routed experts; `u` = mean union size per layer.

**Currency = expert bytes streamed (SSD-bound).** Let `a` = drafts accepted per
cycle (0..K), `h` = resident hit rate on the union, `B` = effective SSD bandwidth.
AR pins `B = 4.79 × 2.472 GiB/s`. With **record dedup (R2)** the verify streams
`40 × u × (1−h) × record` cold bytes per cycle and commits `a+1` tokens, so:

```
MTP_tps  =  4.79 × 6·(a+1) / ( u · (1−h) )
```

**Arithmetic / regimes (the ~3 GB/token bytes-per-cycle picture):**

| regime | u | h | a | bytes/cycle (cold) | tok/s |
|---|---|---|---|---|---|
| **no dedup** (naive per-row gather) | (K+1)·6 = 24/layer | 0 | 3 | 9.89 GiB (= 4 AR tokens) | ≈ 4.8 (break-even; **net-negative** below full accept) |
| dedup, no residency (Gate-0 union) | 10 | 0 | 3 | 4.12 GiB | **11.5** |
| dedup, no residency | 10 | 0 | 2 | 4.12 GiB | 8.6 |
| dedup + 50 % residency | 10 | 0.5 | 3 | 2.06 GiB | **23.0** ✓ |
| dedup + 50 % residency, real accept | 10 | 0.5 | ~2.7 (d1≈90 %) | 2.06 GiB | ~21.3 ✓ (marginal) |
| dedup + 53 % residency, tight union | 7 | 0.53 | 3 | 1.38 GiB | **35** ✓ |

**What clears 20 tok/s.** `6·(a+1)/(u·(1−h)) > 20/4.79 = 4.18`. Three levers must
land together:

1. **R2 record dedup is the ENABLER, not an optimisation.** Without it the verify
   re-gathers up to `(K+1)·6 = 24` records/layer — ~4× AR bytes for ≤4× tokens, so
   MTP is break-even at full accept and net-negative otherwise (this is the GLM-5.2
   precedent: union widened misses 135→180/token, I/O ceiling 11.2→8.4 tps, MTP
   only +11 %). Dedup collapses that to `u` records/layer.
2. **Acceptance must be near-full** (`a ≈ 3`, i.e. d1≈90 % as the DSpark-V4 verdict
   found), because throughput is linear in `a+1` and the union cost is fixed per
   cycle. This is what the DSpark head quality buys.
3. **A residency / hit-rate lever** (`h ≈ 0.4–0.5`) is mandatory. With dedup but
   `h=0`, MTP tops out at ~11.5 tok/s at full accept (u≈10) — the GLM +11 % ceiling.
   Only keeping ~half the (small, cycle-repeating) union resident pushes past 20.
   NOTE: prefetch *across the decision sync* is dead (measured +3.27 % slower); the
   live streaming lever is **residency of the recurring union** (island placement /
   hot-set cache) and overlapping the union gather with compute *within* the verify
   forward — not prefetch across the sync.

**Union bar (Gate 0):** median `u ≤ 10` at K=3. **K>3 and tree/wide verify are
DEAD on a streaming bank** — a wider verify only widens `u` (the denominator) while
marginal acceptance → 0, so `6·(a+1)/(u·(1−h))` falls. W23 builds neither.

**Verify path already satisfies the R2 precondition.** The batched verify runs one
backbone forward over the K+1 tokens, so each layer's `MoE.__call__` reshapes to
`[K+1·b, dim]` and issues **one** `switch_mlp(xf, indices)` per layer with all rows
— the precondition for dedup across the union. `test_kplus1_verify_reaches_each_
moe_layer_in_one_call_and_bounds_the_union` gates exactly this (one gate call per
layer, all K+1 rows, and it measures `u`). The dedup itself lives in the streamed
switch (`mtplx/models/expert_mlx.py`, W16 — outside W23's allowlist and needing the
bank), so **whether the runtime's gather issues each unique record once per cycle
must be confirmed on the real bank** (records-requested vs unique) — see gaps.

## 5. Verification gaps (honest scope under the 12 GiB CPU cap)

The 376 GiB artifact was never loaded (12 GiB cap), so these are implemented but
**not verified end to end against the artifact** — flagged for a follow-up run
under `/tmp/dsv41-cpu-model-load.lock`:

- the mxfp4 **stacked-expert resident load** (`_map_mtp_residents` stacks 128
  per-expert tensors into the SwitchGLU projections; the dense name mapping IS
  gated against the model's own parameter tree, the stacked mxfp4 load is not);
- the **R2 dedup count** on the real streamed switch (records requested vs unique
  per cycle) and the **real u** distribution (Gate 0: median u ≤ 10 at K=3);
- the serve-path glue mapping `--generation-mode mtp` → `MTPLX_DSV41_MTP=1` /
  `with_mtp=True` lives in `cli.py` / `resident_loader.py` (outside W23's
  allowlist). Today the operator sets `MTPLX_DSV41_MTP=1` alongside
  `--generation-mode mtp`; a one-line glue in the serve path removes the env step.

## Files
- `mtplx/models/deepseek_v41_dspark.py` (new) — the DSpark head.
- `mtplx/models/deepseek_v41.py` — ModelArgs DSpark fields; backbone `main_hidden`
  capture; `Model` MTP surface + opt-in head build; `sanitize`/`_map_mtp_residents`;
  `is_/inject_deepseek_v41_mtp_support`.
- `mtplx/models/deepseek_v41_loader.py` — `with_mtp` opt-in path.
- `mtplx/runtime.py` — the deepseek_v41 MTP dispatch arm (+ import).
- `mtplx/backends/registry.py` — `deepseek-v41-mtp` row + base note.
- `tests/models/test_deepseek_v41_dspark.py` (new) — 11 tests.
- `docs/deepseek-v41/W23_REPORT.md`, `docs/deepseek-v41/PORT_CONTRACT.md` (W23).
