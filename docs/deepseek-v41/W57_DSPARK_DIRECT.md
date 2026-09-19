# W57 — DSpark-DIRECT decode lane

Scope: a self-contained speculative-decode loop that drives W23's DSpark 3-stage
drafter against the V4.1 target forward, **bypassing MTPLX's generic native-MTP
machinery** (`generate_mtpk`, `draft_lm_head` install, `mtp_patch`, `runtime.py`'s
streaming-block MTP injection, `model_scheduler`'s MTP cycle). Window 21 measured
that generic pathway serving V4.1 MTP at **2.45–2.94 tok/s** vs served AR **2.2–5.0**
on the Qwen-PR prompt (`docs/deepseek-v41/receipts/gpu-windows/window-21/{ar-1k,mtp-1k}/`
in the integration worktree) — it costs more than it accepts. Decode on this lane
is dispatch-bound (~160 ms for a 1-row forward; a K+1-row verify forward costs
little more — window 17: M=4 gather 2.4× M=1, the rest of the layer is dispatch),
so a lean DSpark loop turns accepted tokens into near-free throughput.

Allowed-writes touched: `mtplx/models/deepseek_v41_dspark_decode.py` (new),
`mtplx/models/deepseek_v41_dspark.py` (one-line drafter bug fix, below),
`mtplx/server/openai.py` (serve dispatch + mode), `mtplx/cli.py` (CLI choices),
`scripts/deepseek_v41/ab_decode_env_levers.py`, `scripts/deepseek_v41/bench_standard_shape.py`,
`tests/models/test_deepseek_v41_dspark_decode.py` (new), this report, `OPTIMIZATION_LEDGER.md`.

---

## 1. Reuse (coordinator W57 directive)

The acceptance / verify / cache-rollback **structure** is the proven DeepSeek-V4
native-MTP K3 loop and its cache primitives, adapted to V4.1's DSpark drafter:

- **`generate_mtp1` / `generate_mtpk`** (`mtplx/generation.py`, PR
  [#216](https://github.com/youssofal/MTPLX/pull/216), the V4 native `deepseek_v4`
  backend measured **25.86 tok/s @ K3**, memory `deepseek-v4-mtplx-port`): the
  verify shape (`[primary, d1..dK]` → `K+1` logit rows, row 0 = the token after
  `primary`), the greedy longest-prefix accept, and the speculative-sampling accept
  on the temp>0 path (`compute_acceptance_probability` + `residual_distribution`).
- **Cache primitives** `snapshot_untrimmable_cache` + `trim_verified_window_to_prefix`
  / `rollback_after_verify` (`mtplx/cache_state.py`): the "all-trimmable caches
  repair by trimming the uncommitted verify tail, no re-forward" path the batched
  `generate_mtpk` uses.
- **DSpark drafter** `mtplx/models/deepseek_v41_dspark.py` (W23): `DSparkHead.seed_main`
  / `draft_block`, `DSparkStageCache` (the W26 sliding-window main-KV leaf).

The V4 DSpark K5 *engine* (`mtplx/deepseek_v4_mia_engine.py`, PR #312, the "Mia"
K5 loop) was **read but not ported**: it is a Metal/NVFP4/kernel-fused engine
(1,038 lines, `MiaM6Cycle`/`acceptance`/`commit_*`) whose value is device
scheduling, not the acceptance algorithm — the algorithm it carries is the same
standard speculative loop this lane reuses from `generate_mtp1`/`generate_mtpk`.

**What this lane drops vs the generic machinery:** the whole `generate_mtpk`
feature surface (FR-Spec, online correctors, adapter ensembles, session-bank warm
prefix, capture-commit/graphbank verify strategies, adaptive width) and the
`draft_lm_head`/`mtp_patch` sidecar install. It keeps only draft → K+1-row verify
→ accept → trim.

**What differs in V4.1's DSpark vs V4's single MTP block** (all owned by W23's
drafter, driven verbatim here): 3 stages threading `main_x`, the noise-token block
embed (`[real_token, noise, ..., noise]`), and the markov + confidence heads. The
reference DeepSeek `inference/model.py` carries the DSpark *forward* but explicitly
leaves the decode loop out of scope (its L129-131: "nothing calls `forward_spec` …
the speculative-decoding loop itself is out of scope for this repo"), so the
loop/accept/rollback are standard speculative decoding, not a reference transcription.

---

## 2. Drafter bug found + fixed (blocking)

W23's `DSparkAttention.__init__` never set `self.mode`, but a later window's
`Attention._sparse_attend_oneshot` (`deepseek_v41.py` L715) reads `self.mode` for
its stage-timing labels. So **any** draft-block forward (T=block_size>1 rows → the
score path) raised `'DSparkAttention' object has no attribute 'mode'` — the W23
greedy verify-reproduces-AR gate was **failing on this branch** (regression from
the W37/W50 stage-timing labels landing after W23). Fixed by setting
`self.mode = MODE_SWA_ONLY` in `DSparkAttention.__init__` (DSpark attention is
always a pure sliding window; the label has no numeric effect). This also restores
`generate_mtpk` DSpark on this branch. Verified: `tests/models/test_deepseek_v41_dspark.py`
now passes (13/13), including the depth 1/2/3 greedy gate.

---

## 3. Algorithm as implemented

State invariant: after prefill the target cache holds the prompt; `primary` is the
sampled-but-not-yet-forwarded next token; the DSpark stage windows hold the main
KV of every committed token.

Per cycle:

1. **Draft length.** `draft_block(main_h, primary, …)` emits `block_size` drafts in
   one 3-stage `forward_spec` (noise-token block embed + markov autoregression). We
   propose `K = min(speculative_depth, block_size)`. The **confidence head** drives
   an optional early stop: keep the leading run whose `sigmoid(confidence) ≥
   MTPLX_DSV41_DSPARK_CONF_THRESHOLD` (default off → verify all K; a pure latency
   lever — it changes verify width, never output).
2. **Verify.** One target forward over `[primary, d1..dK]` (`K+1` rows) →
   `verify_logits[i]` = target distribution for the token *after* block position `i`.
3. **Accept.**
   - *Greedy* (`temperature ≤ 0`): accept the longest draft prefix equal to the
     target argmax at the preceding row; the correction/bonus is the target argmax
     at the first unaccepted row. **Byte-for-byte AR regardless of draft quality.**
   - *Sampled* (`temperature > 0`): standard speculative sampling. The DSpark draft
     is greedy (`DSparkBlock.temperature == 0`), so its proposal is the deterministic
     point mass `q = δ_d`; accept `d` w.p. `min(1, p(d)/q(d)) = p(d)`, and on reject
     draw the correction from `norm(max(0, p − q))`. Output marginal = `p` (the AR
     sampler distribution). `K=0` is therefore **exactly AR sampling** under the
     same seed (proven byte-identical vs `generate_ar` in the tests).
4. **Rollback.** Keep `[primary, d1..da]` (`a+1` rows) and trim the `K−a` rejected
   tail with `trim_verified_window_to_prefix` — no re-forward, no snapshot restore
   (the V4.1 cache is all-trimmable: `LayerAttentionCache.is_trimmable`, so
   window/compressed/index/compressor-frontier/engram rewind together). The DSpark
   stage windows are seeded with the committed tokens' main hiddens via `seed_main`.
5. **Advance.** `primary ← correction`; `main_h ← verify_hidden[:, a:a+1, :]` (the
   row that predicted the correction — the reference's "draft from the hidden of the
   token that predicted the next token").

**3-stage boundary:** the reference runs all 3 stages sequentially in `forward_spec`
threading `main_x`; W23's `draft_block` does the same in one call. No special
handling — stage 0 owns `main_proj`, the last stage owns the head + markov + confidence.

---

## 4. Cost model

Dispatch-bound decode (window 17): a 1-row forward ≈ `T1`; a `(K+1)`-row verify
forward ≈ `T_{K+1}`, with `T_{K+1}/T1` small when the layer is dispatch- (not byte-)
bound.

```
expected tok/s  =  AR tok/s  ×  tokens_per_cycle  ×  (T1 / T_{K+1})
tokens_per_cycle = a + 1   (accepted drafts + the correction/bonus), a ∈ [0, K]
```

At full accept (`a = K`) and `T_{K+1} ≈ T1` (dispatch-bound), the ceiling is
`≈ (K+1) × AR tok/s`. The realized number is set by the **acceptance α** (d1 accept
rate) on the programming shape, which is **unmeasured on this box** (V4 got 5.7× at
K3; DSV4.1 α is the W23/`OPTIMIZATION_LEDGER §8` open unknown "lever L").

**Tension with the streaming-bank bytes model (W23 §4):** if the K+1-row verify
re-gathers the *union* of the rows' routed experts from a **streaming** bank, the
cycle is bytes-bound (`(K+1)·top6` records/layer) and the `T_{K+1} ≈ T1` assumption
fails — MTP is then net-negative below full accept (W23's finding). The dispatch-bound
regime holds when the per-cycle expert bytes do **not** scale with K: a resident
bank, or record-dedup (R2) across the verify union. So this lane's win is gated by
the **same** residency/dedup precondition as the generic MTP lane — what it removes
is only the generic machinery's *overhead*, not the bytes wall. Measure α **and**
`T_{K+1}/T1` in a GPU window before crediting a decode delta.

---

## 5. Served + bench commands

Served lane — two equivalent selectors on a `--load-mtp` deepseek_v41 runtime:

```bash
# explicit mode
mtplx serve --model ~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4 \
  --load-mtp --generation-mode dspark --depth 3 --host 127.0.0.1 --port 8080

# or reuse mtp mode + the flag (per-request generation_mode=mtp also works)
MTPLX_DSV41_DSPARK_DIRECT=1 mtplx serve \
  --model ~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4 \
  --load-mtp --generation-mode mtp --depth 3 --host 127.0.0.1 --port 8080
```

The lane streams tokens as they are committed, honours stop tokens / max_tokens /
usage counts, and reports accept stats (accepted/drafted/rejected, by-depth,
verify_calls) in the `mtplx_openai_generation` telemetry (`stats.to_dict()`).
Constrained decoding and vision splice are **not** supported on the lean lane — a
request carrying them stays on the generic `mtp`/`ar` lanes (dspark never carries them).

A/B decode levers + bench (greedy byte-identity asserted, tokens/cycle +
accept-by-depth in the receipt):

```bash
# under the GPU flock; nice -n 19 for any CPU-side build
scripts/deepseek_v41/ab_decode_env_levers.py --model <mxfp4> \
  --decode-mode dspark --dspark-depth 3 --arms control --out <receipt>.jsonl

scripts/deepseek_v41/bench_standard_shape.py --model <mxfp4> \
  --decode-mode dspark --dspark-depth 3 --context-tokens 1024 --steps 256
```

---

## 6. Tests (CPU, `mx.cpu`, tiny seeded double — no artifact)

`tests/models/test_deepseek_v41_dspark_decode.py` — 14 tests:

- greedy dspark == AR over **256 tokens** at depth {0,1,2,3} (the extended P3.0 bar);
- sampled **K=0 == `generate_ar`** byte-for-byte (fixed seed); sampled K>0 sane +
  distribution-matching construction;
- accept + reject both exercised, counters populate per depth;
- **cache offset == prompt + committed − 1** after a run (rollback trims the
  speculative tail exactly; DSpark windows advance in lockstep);
- served `generate_dspark` end to end (greedy == AR, streamed delta == committed,
  stop / max_tokens / usage, accept stats in `stats.to_dict()`);
- server dispatch wiring (`_normalize_generation_mode("dspark")`, the
  `_dspark_direct_selected` gate: dspark mode, or mtp + `MTPLX_DSV41_DSPARK_DIRECT`,
  only for a dsv41 MTP runtime);
- confidence early-stop is lossless.

Plus the restored W23 suite (13/13) and the serve-glue / gate / head suites.

## 6a. Window-23 follow-ups (integration wiring)

Two gaps surfaced when window 23 first ran the lane against the real artifact:

- **Bench loader built the model with_mtp=False → no DSpark head.** `_load_model`
  in `ab_decode_env_levers.py` / `bench_standard_shape.py` now passes `with_mtp=True`
  when `--decode-mode dspark`, via the shared pure helper
  `deepseek_v41_dspark_decode.dspark_bench_loader_overrides` (unit-tested). Because
  the streaming planner applies `text_only_resident_discount` unconditionally (it
  frees the MTP+vision residents' slots), a with_mtp load would over-commit the
  expert slot pool by exactly the MTP residents it then loads; the helper reprices
  **~7.4 GiB** (the MTP-only residents; vision is never on the text/MTP path) out of
  the memory budget so the 82 GiB plan still fits, and both scripts assert the head
  is present with an actionable message before the run. The greedy AR byte-identity
  reference is produced by the **same loaded model** (the AR cell `_generate` /
  `bench_one_cell` AR pass runs on the identical `resident.model`).

- **Served-path validators only accepted `mtp|ar`.** `--generation-mode dspark`
  died in the daemon command layer (`ValueError: generation mode must be 'mtp' or
  'ar'`). `dspark` is now in every generation-mode validator/choice on the served
  path: `mtplx/cli.py` serve+bench `--generation-mode` choices; `mtplx/server/openai.py`
  parser choices + both `_normalize_generation_mode` sites (request + arg-setter) +
  the `available_generation_modes` health list (dsv41 only); and
  `mtplx/commands/public.py` `GENERATION_MODES` (with `_streamed_mtp_flag_requested`
  recognising an explicit `dspark` mode). The `MTPLX_DSV41_DSPARK_DIRECT=1` +
  `--generation-mode mtp` fallback is unchanged. Covered by CPU tests that the full
  serve argv parses to `generation_mode=dspark`, the daemon normalizer keeps it
  (not forced to AR), and it resolves to the DSpark-direct lane for a dsv41 MTP
  runtime.

## 6b. Window-24 root-cause: 1.5 tok/s despite 3.78 tokens/cycle

Window 24 (integration eff795f75, 1,024-prompt, 256 tokens, receipt
`receipts/gpu-windows/window-24/dspark-1024.json`) showed the lane byte-identical
to AR with **superb acceptance** — 68 cycles, 195 drafted, 189 accepted
(94/98/98% by depth), **3.78 tokens/cycle** — yet only **1.53 tok/s** (168 s), and
its own AR reference pass only **1.705 tok/s** (control) / 2.08 (stack_a) vs the
6.24 tok/s AR baseline of window 15. Two separate causes:

### (1) AR-with-head is 1.7 tok/s — the head steals expert cache, not the forward call

The dspark harness's AR reference and the plain `--decode-mode ar` path call the
**identical** forward — `model(ids, cache=cache)`, one row, no `return_hidden`, no
`logits_keep`, same engram, same K19 (K19 only fires at multi-row prefill; decode
is one row), same per-step `argmax`. A 1-row forward is `token_count == 1` →
`RoutingPhase.DECODE`, so AR is already on the decode path. The slowdown is **not
the call** — it is the **model plan**:

- `with_mtp=True` materializes the DSpark residents (~6.7 GiB) that otherwise are
  expert-cache slots, and the harness reprices a further ~7.4 GiB out of the
  budget (§6a). Net, the resident expert cache is ~14 GiB smaller than window 15's
  AR-only load. On a decode whose cost is dominated by **per-miss SERVICE**
  (dispatch/gather per not-resident expert — [[island-placement-beats-tuning]],
  not raw SSD bytes — [[dsv41-decode-not-ssd-bound]]), fewer resident experts means
  many more misses/token, and that is the 2.4–3.7× AR regression.
- `with_mtp=True` also sets the backbone's `_mtp_target_layer_ids`, so every
  forward captures `main_hidden` at 3 target layers (`mean` over the HC copies).
  This is 3 small reductions/token — real but minor, not the 3.7×.

`--with-mtp` was added to `--decode-mode ar` (both bench scripts) so window 25 can
A/B **"AR + head loaded"** (`--decode-mode ar --with-mtp`, same reprice) vs plain
**"AR"** (`--no-with-mtp`) and attribute the head's expert-cache cost directly. The
head's cache cost is intrinsic to MTP; the lever is expert residency/placement
(R2/R3), not the loop. Consider reserving only the true MTP resident bytes (~6.7)
rather than 7.4 (window 24 peaked at 78 GiB, ~4 GiB of head-room the cache could
reclaim).

### (2) 2.47 s/cycle — the K+1-row verify took the PREFILL routing phase

`current_expert_routing_phase(token_count)` (`mtplx/models/expert_mlx.py`) returns
`RoutingPhase.PREFILL` for `token_count > 1` and `DECODE` for 1. The bench harness
calls `model(...)` **directly** (no `rt.forward_ar`, no
`_expert_routing_context`), so the K+1 = 4-row verify defaulted to **PREFILL** —
the wave/admission machinery, `prepare_prefill_seed`, and (armed or not) the
dense-expert re-read path, which cost seconds per call. The runtime's own
`_expert_routing_context` already routes MTP verify batches as `DECODE`
("MTP verify batches are decode traffic regardless of width"); the direct-call
lane bypassed it.

**Fix (implemented, byte-identity proven on CPU, default ON):** the verify forward
is wrapped in `attention_phase("decode_verify")` + `expert_routing_phase(DECODE)`
so both the served (`rt.forward_ar`) and bench (direct `model(...)`) paths use the
DECODE persistent-slot small-M gather. The phase changes only the routing
machinery — same experts, same mxfp4 weights, same matmul — so greedy output is
byte-identical (tests assert this with the phase forced both ways); `DECODE`
services misses too (`expert_runtime.py:2713` `plan.phase is DECODE and plan.misses`),
so a 4-row verify with a few missed union experts is handled by the decode
geometry, not a prefill wave. `MTPLX_DSV41_DSPARK_VERIFY_DECODE_PHASE=0` forces
PREFILL for the window-25 A/B.

### Cycle cost model + instrumentation

Per-cycle wall is now recorded (coarse `perf_counter`) and emitted in the receipt
under `dspark.per_cycle_ms` / `phase_time_s`:

```
cycle_ms  =  draft_ms (3 resident MTP stage forwards)
           + verify_ms (the (K+1)-row target forward)   <-- the PREFILL-phase cost
           + accept_ms (greedy/spec decision, ~0)
           + commit_ms (trim + seed, small)
tok/s     =  tokens_per_cycle / (cycle_s)   [+ prefill amortized]
```

`--decode-mode dspark --stage-timing` additionally arms the W37 probe around the
decode cycles and emits `dspark.verify_stage_timing` — the verify forward's
internal `attn.<mode>` / `moe.routed_switch` breakdown, the census that shows
whether the fix moved the verify off the prefill switch. Expected after the fix:
`verify_ms` drops toward ~1.2× a 1-row decode forward
([[spec-decode-cycle-anatomy]]), so at 3.78 tokens/cycle the lane approaches
`3.78 × decode_tok_s`. **Both α and the verify/1-row ratio still need a GPU-window
A/B** (`MTPLX_DSV41_DSPARK_VERIFY_DECODE_PHASE` on/off, `--with-mtp` on/off); the
CPU double cannot exercise the streamed switch, so the fix is proven byte-identical
and correct here, and its throughput is a window-25 measurement.

## 6c. Window-25 root-cause: head-load cost + the verify still 1.68 s/cycle

Window 25 (integration f622f91f6): AR stack_a **5.86 tok/s** plain vs **2.12 tok/s**
`--with-mtp` (peak 75.9 vs 76.3 GB); dspark on the DECODE phase still 1.68 s/cycle
(draft 216 ms, verify 1,677 ms).

### (a) Head-load 2.8× — the reprice shrinks the COLD expert cache

Ruled out, from the code (not measurable further on CPU):
- **`main_hidden` capture is identical** with/without the head: `_mtp_target_layer_ids`
  is set from the **config** (`deepseek_v41.py:2177`), not from the head build, and
  `Model.__call__` always passes `return_main_hidden=True`, so plain AR and
  `--with-mtp` AR both capture the target-layer hiddens — not the diff.
- **The MTP experts stay resident, never bound to the streamed switch**: the loader's
  `bind_streamed_switches` walks only `model.model.layers` and asserts
  `bound == routed_layer_count` (40 backbone layers); the head's 3×128 experts are
  resident mxfp4 (`_build_mtp_head` `nn.quantize`), and `partition_text_residents`
  keeps them off the streamed bank. So the streamed slot plan has 40 layers
  regardless of the head.

What remains: at the **same** memory budget the streamed slot plan is identical
(the planner's `text_only_resident_discount` is applied regardless of `with_mtp`),
so the only levers are (i) the head loading ~6.7 GiB of residents the plan
discounted, and (ii) the harness reprice removing a further ~7.4 GiB. Peak was
nearly unchanged (76.3 vs 75.9) — the reprice traded ~7 GiB of **expert-cache
slots** for the MTP residents, so the COLD decode has ~7 GiB fewer resident
experts and pays far more cold misses. "Misses nearly free" (window 13) was a WARM
repeat; the window-25 pass is COLD. **`--with-mtp --no-reprice` was added** (both
scripts, via `dspark_bench_loader_overrides(reprice=False)`): it loads the head at
the FULL budget so window 26 can separate the budget/slots effect (should recover
most of the 2.8× if it is purely slots) from any residual head-load code-path
effect (a slowdown at the same budget would be a code path, since the slot plan is
identical). Follow-up if it is slots: thread `with_mtp` into the loader plan so it
discounts only vision (not MTP) when the head is kept, instead of the blanket
harness reprice.

### (b) Verify 1.68 s/cycle — attention took the prefill path; the switch is host-sync-bound

The window-25 verify stage census:
- **attn (`attn.reuse`) ~830 ms per 4-row verify** (vs ~50 ms at M=1): the verify's
  attention ran the **rows>1 prefill score path**. K29 (fused decode/verify
  attention, `b*s ≤ 8`) and K30 (selected-key gather) were **not armed** in the
  window-25 arm. `_sparse_attend` checks `_decode_attn_kernel_use(q)` *before* the
  `q.shape[1] <= 1` branch, so a 4-row verify **does** take the decode kernel when
  armed — the gating was fine, the flags were just off. **Fix:** the lane now arms
  `MTPLX_DSV41_DECODE_ATTN_KERNEL` + `MTPLX_DSV41_SELECTED_KEYS` for the whole
  dspark run (`arm_dspark_decode_kernels`; the ab_decode/bench harness `setdefault`s
  them for `--decode-mode dspark`, the served `generate_dspark` wraps its cycles).
  They are armed for the AR reference too so the (greedy-identical, not
  bit-identical) kernels stay consistent and `byte_identical_vs_ar` holds.
  `MTPLX_DSV41_DSPARK_DECODE_KERNELS=0` opts out for the A/B.
- **switch (`moe.routed_switch`) ~630 ms per verify** (vs ~74 ms at M=1): the
  `route_stage` census pins it to **host-sync barriers**, not the gather (which the
  microbench puts at ~1 ms for 24 rows): `hot.eval_indices` = the dominant stage
  (~10 ms each — a device→host sync of the per-layer routing indices before the
  gather; memory `queued-vs-eager-metal-microbench`), and `hot.allhit_fence_eval`
  (~3.6 ms each). The M=4 verify issues **many more** of these per layer than an
  M=1 step because the all-hit/split-route path evaluates indices per split rather
  than once for the whole 4-row wave. **This is a streamed-runtime fix (GPU-only,
  cannot be validated under the ≤3 GB CPU cap), so it is proposed, not shipped
  here:** keep the verify's per-layer routing indices on-device (one gather, no
  host sync) or batch the M=4 index eval into a single barrier per layer, so the
  verify approaches ~1.3× an M=1 step. Owner: the streamed-switch decode path
  (`expert_streaming.py` DECODE branch / `expert_mlx` all-hit dispatch), not this
  lane.

### New per-cycle model

```
cycle_ms = draft_ms + verify_ms + accept_ms(~0) + commit_ms
verify_ms ≈ attn_ms + switch_ms + (hc/head/misc)
   attn_ms  : ~830 ms with prefill attention  ->  K29/K30 target ~1.3× M=1 (~65 ms)
   switch_ms: ~630 ms host-sync-bound (hot.eval_indices) -> target ~1.3× M=1 (~95 ms)
```
With both addressed the verify should fall from ~1.68 s toward ~0.2–0.3 s, i.e.
at 3.78 tokens/cycle roughly `3.78 × (decode_tok_s of a head-loaded AR step)`.
Both the K29/K30 win and the switch host-sync fix need a GPU-window A/B
(`MTPLX_DSV41_DSPARK_DECODE_KERNELS` on/off; `--no-reprice` on/off); the CPU double
has no streamed switch, so the lane changes here are proven byte-identical and the
throughput is window-26 measurement.

## 6d. Window-27: probe fix, the rows>1 audit, and the K29-off knob

Window 27 (integration b582b73a3, W57d + W60/K29 + W61 single-barrier merged):
dspark verify still **1,876 ms/cycle** with K29 engaged for every verify row
(10,280 kernel calls, 0 fallbacks), and `verify_stage_timing` was **empty** — the
W37 probe recorded only `s == 1` steps, so the 4-row verify forward was invisible.

### (1) Probe records the verify; route-stage + W61 engagement in the receipt
- **`enter_forward`** (decode kind) now arms recording for `1 ≤ s ≤ 8`
  (`_DECODE_STAGE_MAX_ROWS`), not only `s == 1`; a genuine prefill (`s > 8`) stays
  unrecorded. `--decode-mode dspark --stage-timing` now yields the verify's
  internal stage table (`moe.*`, `attn.*`, `hc.*`, `head`) plus coarse
  `dspark.draft` / `dspark.verify` brackets; `arm_recording()` is called at each
  cycle start so both record from cycle 0 (the draft doesn't call `enter_forward`).
- The route-stage probe (`expert_route_probe.ENABLED`, read at use) is armed and
  cleared for the dspark pass, and its **W61 engagement counter**
  (`hot.verify_single_barrier`, plus `hot.all_hit` / `hot.eval_indices` /
  `hot.begin_split_route`) is surfaced in the receipt as `dspark.w61_engagement` —
  so window 28 sees whether W61's single-barrier fast path actually fires (it
  engages only on an **all-hit** verify; a cold/missing verify falls back to the
  multi-barrier `route_waves` loop).

### (2) Audit — every 4-row verify branch keyed on rows>1

| Branch (file) | rows>1 behavior | armed? | expected cost | note |
|---|---|---|---|---|
| `_sparse_attend` q.shape[1]>1 (deepseek_v41.py:686) | prefill score path (one-shot/lean/chunked) | **K29 default now ON** → fused decode kernel (rows≤8), checked *before* the s>1 branch (`:682`) | K29 on: one dispatch; K29 off: ~830 ms | the window-25 attn cost; K29 is −38% vs eager at M=1 so it may hurt the wider verify — `MTPLX_DSV41_DSPARK_VERIFY_K29=0` isolates it (task 3) |
| streamed switch DECODE phase, M=4 (expert_streaming.py) | union routed via `plan()` (deduped, not per-row), then **route_waves** unless W61 all-hit | W61 default ON but engages ONLY all-hit | ~630 ms when W61 misses: `hot.eval_indices` ~10 ms × several barriers/layer | the dominant residual; **streamed-runtime fix, GPU-only** — keep indices on-device / one barrier/layer |
| prefill softmax kernel `s>1 and _prefill_softmax_kernel_use()` (:776) | prefill tiled softmax | default OFF | 0 (off) | would apply to the verify if `MTPLX_DSV41_PREFILL_SOFTMAX_KERNEL=1` |
| HC compile `rows ≤ _HC_COMPILE_MAX_ROWS=32` (:1874) | compiled HC tape engages for the verify | on when `MTPLX_DSV41_HC_COMPILE` | one-time trace per new (4-row) shape, then replay | verify is *inside* the cap — helps, not a penalty |
| attn compile `rows ≤ _ATTN_COMPILE_MAX_ROWS=32` (:1625) | compiled attn prep engages | on when `MTPLX_DSV41_ATTN_COMPILE` | one-time trace, then replay | same |
| backbone one-shot vs chunked | 4 rows ≪ the chunk byte-threshold → one-shot span | n/a | 0 | the verify is never chunk-major / layer-major (those are for large prefills) |
| K19 logits (Model.__call__) | verify passes NO `logits_keep` → keeps all K+1 rows | n/a | tiny `[1,4,vocab]` transient (not the 16k prefill narrowing) | correct — MTP verify must keep every row |
| main_hidden capture (`_mtp_target_layer_ids`) | mean over hc at 3 target layers, ×4 rows | always (config) | ~µs, same as AR per row | not the diff |
| engram advance + cache appends (window/compress/index) | advanced/appended for 4 rows, trimmed on rollback | when engram wired | small; O(rows) | rollback restores exactly |
| MoE main path (draft) — see below | — | — | — | — |

**The draft's 261 ms** is NOT `3 × 40` layers: `DSparkHead` is **3 shallow stages**
(`self.layers = [DSparkBlock(...) for i in range(n_mtp_layers==3)]`), each ONE
decoder block — DSparkAttention (sliding window over 4 rows) + a **resident**
128-expert top-3 MoE — plus `forward_embed` and `forward_head`'s markov
autoregression (`block_size==4` sequential steps). The markov samples are lazy
`mx.argmax` (no per-step host sync — `_sample`), so the 261 ms is the 3-stage
forward's compute/dispatch over 4 rows (resident weights, no streamed misses) +
the head, evaluated once. Being on resident experts it is not the SSD/host-sync
problem the verify has; ~87 ms/stage is dispatch-bound and a candidate for the
same K29/decode-attention treatment (the drafter's DSparkAttention also hits
`_sparse_attend` at 4 rows).

### (3) K29-off knob (task 3)
`MTPLX_DSV41_DSPARK_VERIFY_K29=0` arms K30 (selected keys) but leaves K29
unarmed, so window 29 can separate the fused-kernel's cost from the routing/switch
cost. Single source `dspark_decode_kernel_env_defaults()` drives both the served
`arm_dspark_decode_kernels()` and the bench harness `setdefault`, honoring the
toggle and `MTPLX_DSV41_DSPARK_DECODE_KERNELS=0` (both off).

## 7. Caveats

- **Acceptance α + `T_{K+1}/T1` unmeasured on this box** — the tok/s win is a GPU
  window job (no experts.bin under the 3 GB worker cap). §4's tension with the
  streaming-bank bytes model is real: credit no decode delta without α and the
  verify/1-row cost ratio measured in-window.
- Lean lane only: no session-bank warm prefix (cold prefill each request), no
  constraints, no vision. Those stay on the generic lanes.
- The lane requires an all-trimmable V4.1 cache; a non-trimmable entry raises
  (should not occur for deepseek_v41).
