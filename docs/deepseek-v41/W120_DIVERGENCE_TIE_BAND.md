# W120 — magnitude-aware (bf16-ulp) DSpark divergence classifier

**Change class.** Small runtime change to the DSpark divergence *classifier* (how a
greedy divergence receipt is *labelled*) + tests + this doc. No decode-path change:
the tokens produced are untouched; only the `class` field and the divergence block
of the receipt change.

## 1. Problem (from W119)

W119 (`W119_EAGER_VERIFY_PARITY.md`) proved the DSpark K+1-row verify equals the
1-row AR forward **bitwise on CPU fp32** (core batch-invariance max|Δ| = 0.000e+00;
full-model verify == AR to ≤ 6.11e-07). So the GPU divergence at window-45 index 111
is **not a bug** — it is bf16 accumulation-order between the s=K+1 and s=1 eager
einsum tiles, compounded through the 40 backbone layers and **quantized by the bf16
head** onto the bf16 dyadic grid. Every observed value there is an integer number of
bf16 ulps:

| quantity | value | in bf16 ulps |
|---|---|---|
| `dspark_top2_margin` (verify side) | 0.0 | 0 — a genuine bf16 tie |
| `ar_top2_margin` | 0.125 | 1 ulp at \|logit\| ∈ [16, 32) |
| `max_abs_logit_delta` | 1.125 | ≈1 ulp at the peak logit + a smaller one |

The old classifier (`deepseek_v41_dspark_decode.py`, W77 rule)
`cls = "tie_flip" if (ar_margin is not None and ar_margin < tie_margin) else "divergent"`
with a **fixed** `tie_margin = 3e-2` mislabels index 111 `divergent` for two reasons:

1. **It ignores the verify-side margin.** The near-tie here is on the *authoritative
   verify* side (`dspark_top2_margin` 0.0), not the AR side (0.125). When the verify
   forward itself cannot separate the two candidate tokens, which one its argmax
   emits is rounding-determined — the definition of a tie flip
   ([[dsv41-inexact-ok-if-tie-flips]]).
2. **The fixed 3e-2 band does not scale with logit magnitude.** At the cell16k
   operating magnitudes (~16–256) one bf16 ulp is 0.125–2.0 — 4–66× the constant —
   so any legitimately bf16-tied pair there is forced to `divergent`.

## 2. Rule implemented

`bf16 ulp` for a logit of magnitude `x`: `ulp_bf16(x) = 2**(floor(log2|x|)) * 2**-7`
(bf16 has a 7-bit mantissa). `ulp_bf16(0)` and non-finite → `0.0` (fall back to the
fixed floor). Grid: 8→0.0625, 16–24→0.125, 32–48→0.25, 64–96→0.5, 128–192→1.0,
256→2.0.

`peak_contested_logit` = `max(|logit[ar_token]|, |logit[dspark_token]|)` over
whichever rows are available (W119 definition), falling back to a row's top-1
magnitude when a contested index is out of range.

`class = "tie_flip"` iff the **AR reference row is present** and EITHER:

* **(a)+(b) magnitude-aware near-tie:** `min_defined(ar_top2_margin,
  dspark_top2_margin) < tie_band`, where
  `tie_band = max(tie_margin, k * ulp_bf16(peak_contested_logit))` and `k` defaults
  to **3**. The near-tie may be on either side — the authoritative verify side is the
  W119 index-111 case; or
* **(c) rounding-class by delta:** `rounding_class_by_delta = min(ar_top2_margin,
  dspark_top2_margin) <= |Δ(ar_token)| + |Δ(dspark_token)|` (the measured per-token
  deltas at the two contested tokens can close the smaller margin) **AND** those
  closing deltas are themselves within the band (`deltas_within_tie_band =
  max(|Δ(ar_token)|, |Δ(dspark_token)|) <= tie_band`).

otherwise `class = "divergent"`. With **no** AR reference row the class is
`divergent` (conservative: a failed M=1 replay is not silently absolved, even on a
hard verify tie).

### Why (c) is gated (the one deviation from W119's literal wording)

W119 offered rule (c) as a *fully rigorous alternative* stated as the bare formula
`min(margin) <= |Δ(ar_token)| + |Δ(dspark_token)|`. Combined with (a) into one
classifier (this task) the **bare** (c) over-fires: any *genuine* argmax swap has a
contested-token delta ≥ its margin (that is what caused the swap), so the bare (c)
would call **every** real divergence `tie_flip`. That directly contradicts W119's own
criterion — *"a genuine >ulp divergence with both margins clear still classes
divergent"* — and would regress the existing gate
`test_divergent_when_ar_margin_above_threshold` (margins 0.5, deltas 0.5, at
\|logit\|≈2 where one ulp is 0.0156, i.e. 32 ulps — not rounding).

So (c) contributes to the class only when the closing deltas are themselves
**rounding-scale** (`deltas_within_tie_band`). The receipt records the raw W119
signal (`rounding_class_by_delta`) *and* the gate (`deltas_within_tie_band`)
separately, so a reader sees exactly why (c) did or did not fire. In practice the
magnitude-aware near-tie (a) is what fires for the real cell16k arm; (c) is a
belt-and-braces confirmation that never absolves a >band delta.

## 3. `k` knob — a classifier setting, not a decode lever

`k` is read (at use) from `MTPLX_DSV41_DIVERGENCE_TIE_ULPS` (default 3), overridable
per-call via the `tie_ulps` argument (argument wins over env; invalid env → default).

It is **deliberately NOT** registered in `ab_decode_env_levers.ALL_LEVER_ENVS`.
`ALL_LEVER_ENVS` is the DECODE-lever registry (its members change what the forward
computes / how it is scheduled, and are snapshotted into the served-log lever
superset `openai._DSV41_LEVER_ENV_KEYS`). `MTPLX_DSV41_DIVERGENCE_TIE_ULPS` changes
**only how a divergence receipt is labelled** — never a token, never a kernel — so
registering it would plant a dead served "lever". Its value is instead **stamped in
the divergence block** as `tie_ulps`, where the classification is auditable. A gate
(`test_env_knob_is_not_a_decode_lever`) locks this out of the registry.

## 4. Receipt keys

Existing (W77) keys are **kept unchanged** (`divergence_index`, `ar_token`,
`dspark_token`, `ar_top2_margin`, `dspark_top2_margin`, `max_abs_logit_delta`,
`tie_margin`, `class`). `tie_margin` stays the **floor** parameter (default 3e-2),
distinct from the effective band. W120 **adds** (never renames):

| key | meaning |
|---|---|
| `tie_band_used` | the effective `max(tie_margin, k*ulp_bf16(peak))` used |
| `tie_ulps` | `k` actually used (arg / env / default) |
| `peak_contested_logit` | magnitude driving the ulp |
| `ulp_bf16_at_peak` | one bf16 ulp at that magnitude |
| `delta_at_ar_token` | \|Δlogit\| at `ar_token` (verify − AR) |
| `delta_at_dspark_token` | \|Δlogit\| at `dspark_token` |
| `rounding_class_by_delta` | W119 literal rule (c) signal (bool/None) |
| `deltas_within_tie_band` | the (c) gate (bool/None) |

## 5. Test evidence

Test: `tests/test_deepseek_v41_w120_divergence_tie_band.py` (CPU, fp32, pure
classifier — no model, no checkpoint, no GPU). Run under `nice -n 19`,
`PYTHONPATH=<worktree>` (the editable install targets `main`, which lacks
`deepseek_v41_dspark_decode.py`'s streaming-branch state), one file per process:

<!-- W120_EVIDENCE_START -->
```
$ PYTHONPATH=$WT nice -n 19 .venv/bin/python3 -m pytest \
      tests/test_deepseek_v41_w120_divergence_tie_band.py -q
............. [100%]
13 passed
```

Cases:

* **`ulp_bf16`** lands on the W119 dyadic grid at 8 / 16 / 24 / 32 / 48 / 64 / 96 /
  128 / 192 / 256 and 16.125; sign-independent; 0 / inf / nan → 0.0.
* **window-45** (ar 0.125 / dspark 0.0 / Δ 1.125 / peak 16.125) → `tie_flip`
  (`tie_band_used` 0.375, `ulp_bf16_at_peak` 0.125); fires by both (a) and (c).
* **real divergence, small delta** (margins 2.0 / 1.5, contested Δ 0.1) → `divergent`
  (`rounding_class_by_delta` False — 1.5 can't be closed by 0.1).
* **genuine large-delta swap** (margins 2.0 / 2.0, contested Δ 2.0 each) →
  `divergent` **via the gate**: `rounding_class_by_delta` True but
  `deltas_within_tie_band` False.
* **fixed-band legacy** (low magnitude ~1, floor dominates) → `tie_flip`
  (`tie_band_used` == 3e-2).
* **verify-side near-tie** (ar margin 0.5 wide, dspark margin 0.1 < band) →
  `tie_flip` (part (a), the authoritative side).
* **magnitude-aware reclassification** (ar margin 0.125 > 3e-2, band 0.375) →
  `tie_flip` (the core W119 fix the old fixed band missed).
* **k knob** — default 3; `tie_ulps` arg wins; env `MTPLX_DSV41_DIVERGENCE_TIE_ULPS`
  read at use (0 → divergent, 10 → band 1.25, invalid → default); NOT in
  `ALL_LEVER_ENVS`.
* **missing AR row** stays `divergent` even on a hard verify tie (conservative).
* **receipt** is JSON scalars only; every W77 key preserved.

Regression: the existing `tests/models/test_deepseek_v41_dspark_divergence_classify.py`
passes **unchanged** — the `have_reference` gate preserves the old missing-AR
conservatism, and the (c) gate preserves the decisive-margin `divergent` case.
<!-- W120_EVIDENCE_END -->

## 6. Verdict

The classifier now labels the W119 bf16-tie flips `tie_flip` (rounding-class,
acceptable per [[dsv41-inexact-ok-if-tie-flips]]) while staying **loud** for a real
> band divergence. The decode path is byte-for-byte unchanged; this is a receipt
label + diagnostic-key change only.
