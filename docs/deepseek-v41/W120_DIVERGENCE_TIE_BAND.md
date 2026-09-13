# W120 — magnitude-aware (bf16-ulp) contested-token DSpark divergence classifier

**Change class.** Small runtime change to the DSpark divergence *classifier* (how a
greedy divergence receipt is *labelled*) + tests + this doc. No decode-path change:
the tokens produced are untouched; only the `class` field and the divergence block
of the receipt change. This revision incorporates the W120 red-team (2 HIGH, 3
MEDIUM, 3 LOW).

## 1. Problem (from W119)

W119 (`W119_EAGER_VERIFY_PARITY.md`) proved the DSpark K+1-row verify equals the
1-row AR forward **bitwise on CPU fp32**. The GPU divergence at window-45 index 111
is **not a bug** — it is bf16 accumulation-order between the s=K+1 and s=1 eager
einsum tiles, compounded through the backbone and **quantized by the bf16 head**
onto the bf16 dyadic grid. The old classifier
(`cls = "tie_flip" if ar_top2_margin < 3e-2 else "divergent"`) mislabels it: at the
cell16k magnitudes (~16–256) one bf16 ulp is 0.125–2.0, 4–66× the fixed 3e-2.

## 2. Rule implemented (red-team-hardened)

`ulp_bf16(x) = 2**(floor(log2|x|)) * 2**-7` (bf16, 7-bit mantissa); `0` / non-finite
→ `0.0`. Grid: 8→0.0625, 16–24→0.125, 32–48→0.25, 64–96→0.5, 128–192→1.0, 256→2.0.

A flip `ar_token → dspark_token` is `class = "tie_flip"` iff it is **absolvable**
AND (**near-tie by band (a)** OR **rounding-class by delta (c)**):

* **absolvable** — a REAL AR reference (`ar_top2_margin` exists ⇒ ≥ 2 logits) AND
  both **contested margins** computable (both tokens in range on both rows) AND
  `deltas_within_tie_band` — `max(|Δ(ar_token)|, |Δ(dspark_token)|) ≤ tie_band`,
  where `Δ(t) = |ar_row[t] − dspark_row[t]|`. **A > band contested delta is not
  rounding** (HIGH-1): a verify row's own tight top-2, or a decisive AR top-2 at
  *other* tokens, never absolves on its own.
* **band** — `tie_band = max(tie_margin, k · ulp_bf16(peak))`, `peak =
  max(|ar_row[ar_token]|, |ar_row[dspark_token]|)` from the **AR reference row
  only** (MEDIUM-1: a garbage verify logit must not widen the band). `k` default 3.
* **contested margin** — `|row[ar_token] − row[dspark_token]|` per row (HIGH-2), NOT
  the row's top-1/top-2 gap (which can be between two entirely different tokens).
  `ar_top2_margin` / `dspark_top2_margin` stay as **diagnostic receipt keys**.
* **(a) near_tie_by_band** — `min(ar_contested_margin, dspark_contested_margin) <
  tie_band`.
* **(c) rounding_class_by_delta** — `min(ar_contested_margin,
  dspark_contested_margin) <= |Δ(ar_token)| + |Δ(dspark_token)|` (recorded as its
  own key; contributes to the class only through the `absolvable` delta gate).

Otherwise `class = "divergent"` (loud, conservative — a failed/partial M=1 replay,
MEDIUM-2, is never silently absolved).

**Why the delta gate matters.** A *genuine* argmax flip has a contested delta ≥ its
margin (that is what caused the flip). Absolving on a small margin alone (the first
W120 draft) therefore absolved real divergences — e.g. AR `[20.0, 17.0]` vs verify
`[18.5, 18.5]` (verify tie, but 1.5/1.5 = 12-ulp deltas), or a verify row shifted
+40 (every contested delta 40). Requiring the *closing deltas* to be within the band
keeps those `divergent` while still absolving a true bf16 tie whose deltas are a few
ulps.

**Operators (`<` vs `<=`).** (a) uses strict `<` on the contested margin; (c) uses
`<=` (a delta exactly equal to the margin closes it). Both require
`deltas_within_tie_band` (`<=` band). Code and this doc agree (LOW).

## 3. `k` knob — a bounded classifier setting, not a decode lever

`k` is read (at use) from `MTPLX_DSV41_DIVERGENCE_TIE_ULPS` (default 3), overridable
per-call via `tie_ulps` (argument wins). Only a **non-negative integer in
`[0, 64]`** is accepted (MEDIUM-3): a negative / non-integer / malformed value
(`"-1"`, `"1_0"`, `"3.0"`, `"nan"`, `"1e20"`, a fractional float, a bool) is
**rejected with a WARN line** and the default (never silently coerced); a value
above 64 is clamped with a WARN (a k of 64 at |logit| 256 is already a 128-logit
band — well past any real bf16 perturbation). `k = 0` is a valid explicit "disable
magnitude scaling" (band collapses to the fixed floor).

It is **deliberately NOT** in `ab_decode_env_levers.ALL_LEVER_ENVS` (the decode-lever
registry, snapshotted into the served-log superset `openai._DSV41_LEVER_ENV_KEYS`):
it changes only how a receipt is *labelled*, never a token or kernel. Its value is
stamped in the divergence block as `tie_ulps`; `test_env_knob_is_not_a_decode_lever`
locks it out of the registry.

## 4. Receipt keys

Existing (W77) keys **kept unchanged** (`divergence_index`, `ar_token`,
`dspark_token`, `ar_top2_margin`, `dspark_top2_margin`, `max_abs_logit_delta`,
`tie_margin`, `class`). `tie_margin` stays the **floor** parameter. W120 **adds**
(never renames):

| key | meaning |
|---|---|
| `tie_band_used` | effective `max(tie_margin, k·ulp_bf16(peak))` |
| `tie_ulps` | `k` actually used (arg / env / default, bounded) |
| `peak_contested_logit` | magnitude driving the ulp — **AR row only** |
| `ulp_bf16_at_peak` | one bf16 ulp at that magnitude |
| `ar_contested_margin` | `\|ar[ar_token] − ar[dspark_token]\|` |
| `dspark_contested_margin` | `\|dsp[dspark_token] − dsp[ar_token]\|` |
| `delta_at_ar_token` | `\|ar[ar_token] − dsp[ar_token]\|` |
| `delta_at_dspark_token` | `\|ar[dspark_token] − dsp[dspark_token]\|` |
| `rounding_class_by_delta` | W119 rule (c) signal on the contested margins |
| `deltas_within_tie_band` | the delta gate (both contested deltas ≤ band) |
| `ar_logit_at_ar_token`, `ar_logit_at_dspark_token`, `dspark_logit_at_ar_token`, `dspark_logit_at_dspark_token` | the four raw contested logits, so a **serialized receipt is self-decidable without the rows** |

The AB census line (`ab_decode_env_levers._print_dspark_divergence`) now prints
`tie_band_used`, both contested margins, both contested deltas, and **which rule
fired** (the old `ar_top2_margin < tie_margin` line was false for a W120 tie_flip).

## 5. Test evidence

Test: `tests/test_deepseek_v41_w120_divergence_tie_band.py` (CPU, fp32, pure
classifier — no model, no checkpoint, no GPU). `nice -n 19`, `PYTHONPATH=<worktree>`,
one file per process:

<!-- W120_EVIDENCE_START -->
```
$ PYTHONPATH=$WT nice -n 19 .venv/bin/python3 -m pytest \
      tests/test_deepseek_v41_w120_divergence_tie_band.py -q
.................. [100%]
18 passed
```

Covers: `ulp_bf16` on the W119 grid; `_peak_contested_logit` AR-row-only; the
window-45 reconstruction → `tie_flip`; the red-team divergence cases (verify-tie
large delta, +40-shifted verify row, uncontested near-tie/HIGH-2, garbage +300
verify logit/MEDIUM-1, degenerate 0-/1-element AR row/MEDIUM-2) → `divergent`;
legitimate 1-ulp bf16 tie and the low-magnitude fixed-band case → `tie_flip`; a
real flip whose within-band deltas close the smaller contested margin (rule (c));
bf16 `mx.array` rows do not raise; `k` default/arg/env-at-use and all the MEDIUM-3
bounds (rejects `-1`, `1_0`, `3.0`, `nan`, `1e20`, fractional/bool; clamps > 64;
`0` valid); knob NOT in `ALL_LEVER_ENVS`; receipt is JSON scalars with all keys.

Regression: `tests/models/test_deepseek_v41_dspark_divergence_classify.py` passes
**unchanged** (10 tests) — the `absolvable` gate keeps the missing-AR conservative
case and the decisive-margin case `divergent`; the contested-margin switch matches
its two-hot-token rows exactly.
<!-- W120_EVIDENCE_END -->

## 6. Re-check of the real window-45 index-111 (eager arm)

Receipt `docs/deepseek-v41/receipts/gpu-windows/window-45/dspark-d5-draft-attn-eager.json`,
`dspark.divergence`: `ar_token 8049`, `dspark_token 1975`, `ar_top2_margin 0.125`,
`dspark_top2_margin 0.0`, `max_abs_logit_delta 1.125`, old `class "divergent"`.

**The receipt stores only these scalars — the full logit rows are not serialized**
(`DivergenceCapture` keeps them in-process only), so the *contested-token deltas*
`Δ(8049)` and `Δ(1975)` that the corrected rule needs **cannot be recomputed from
it**. What *is* pinned:

* `ar_top2_margin = 0.125` is exactly **one bf16 ulp**, which is representable only
  in binade `[16, 32)` (at |logit| ≥ 32 the ulp is ≥ 0.25, so a 0.125 gap cannot
  exist on the bf16 grid). Hence the AR contested logits are ~16–32 and
  **`tie_band_used = max(0.03, 3 × 0.125) = 0.375`** — pinned.
* Both contested deltas are bounded by the vocab-wide max: `Δ(8049), Δ(1975) ≤
  1.125`. W119 attributes the 1.125 to the **peak-magnitude** logit (~128–256, where
  1 ulp ≈ 1.0–2.0), *not* the ~16 contested tokens, so the contested deltas are very
  likely a few ulps at 16 (≤ ~0.375) — but this is **not provable from the receipt**.

**Verdict: indeterminate from the stored receipt.** Under the corrected rule the
class is `tie_flip` iff both contested deltas ≤ 0.375 (band pinned), else
`divergent`. The "tie flip" reading in `[[dsv41-inexact-ok-if-tie-flips]]` /
W119 is *plausible but now UNCONFIRMED* under the stricter rule. To decide
definitively, re-run the eager arm — receipts now carry the four contested logits
(`{ar,dspark}_logit_at_{ar,dspark}_token`), so the next run's receipt is
self-decidable; if either contested delta exceeds 0.375, `divergent` is the correct
label and the memory note should be corrected. (A worker cannot run that GPU window
here; flagged for the coordinator.)

## 7. Verdict

The classifier now absolves a bf16 tie flip **only** when the measured
contested-token deltas are rounding-scale (≤ the magnitude-aware band computed from
the AR reference), using the contested margins — closing the HIGH/MEDIUM holes where
a tight verify top-2, a decisive top-2 at other tokens, a shifted/garbage verify
row, or a degenerate AR row wrongly absolved. The decode path is byte-for-byte
unchanged; this is a receipt label + diagnostic-key change only.
