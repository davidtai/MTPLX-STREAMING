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

## 7. Caveats

- **Acceptance α + `T_{K+1}/T1` unmeasured on this box** — the tok/s win is a GPU
  window job (no experts.bin under the 3 GB worker cap). §4's tension with the
  streaming-bank bytes model is real: credit no decode delta without α and the
  verify/1-row cost ratio measured in-window.
- Lean lane only: no session-bank warm prefix (cold prefill each request), no
  constraints, no vision. Those stay on the generic lanes.
- The lane requires an all-trimmable V4.1 cache; a non-trimmable entry raises
  (should not occur for deepseek_v41).
