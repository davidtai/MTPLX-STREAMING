# W95 prefetch precision/economics (one-ahead gate oracle, layers >= 4)

Score scale (mean over layers/decode rows): s1=19.1057, s6=17.6337, gap(s1-s6)=1.4720. objective = hits - 0.67 * wasted (per layer-route; hidden_misses proportional to hits).

| k | margin | issued | hits | wasted | precision | recall | objective |
|--:|-------:|-------:|-----:|-------:|----------:|-------:|----------:|
| 6 | -0.30 | 2.61 | 2.27 | 0.34 | 0.900 | 0.379 | 2.042 |
| 6 | -0.15 | 3.32 | 2.75 | 0.57 | 0.849 | 0.458 | 2.365 |
| 6 | -0.05 | 4.13 | 3.21 | 0.93 | 0.780 | 0.535 | 2.588 |
| 6 | +0.00 | 6.00 | 3.88 | 2.12 | 0.647 | 0.647 | 2.467 |
| 6 | +0.05 | 6.00 | 3.88 | 2.12 | 0.647 | 0.647 | 2.467 |
| 6 | +0.15 | 6.00 | 3.88 | 2.12 | 0.647 | 0.647 | 2.467 |
| 6 | +0.30 | 6.00 | 3.88 | 2.12 | 0.647 | 0.647 | 2.467 |
| 8 | -0.30 | 2.61 | 2.27 | 0.34 | 0.900 | 0.379 | 2.042 |
| 8 | -0.15 | 3.32 | 2.75 | 0.57 | 0.849 | 0.458 | 2.365 |
| 8 | -0.05 | 4.13 | 3.21 | 0.93 | 0.780 | 0.535 | 2.588 |
| 8 | +0.00 | 6.00 | 3.88 | 2.12 | 0.647 | 0.647 | 2.466 |
| 8 | +0.05 | 6.89 | 4.08 | 2.81 | 0.607 | 0.680 | 2.199 |
| 8 | +0.15 | 7.39 | 4.20 | 3.19 | 0.581 | 0.700 | 2.066 |
| 8 | +0.30 | 7.68 | 4.27 | 3.42 | 0.564 | 0.711 | 1.979 |
| 12 | -0.30 | 2.61 | 2.27 | 0.34 | 0.900 | 0.379 | 2.042 |
| 12 | -0.15 | 3.32 | 2.75 | 0.57 | 0.849 | 0.458 | 2.365 |
| 12 | -0.05 | 4.13 | 3.21 | 0.93 | 0.780 | 0.535 | 2.588 |
| 12 | +0.00 | 6.00 | 3.88 | 2.12 | 0.647 | 0.647 | 2.466 |
| 12 | +0.05 | 7.73 | 4.18 | 3.55 | 0.585 | 0.696 | 1.799 |
| 12 | +0.15 | 9.12 | 4.41 | 4.71 | 0.529 | 0.735 | 1.252 |
| 12 | +0.30 | 10.14 | 4.57 | 5.57 | 0.486 | 0.761 | 0.835 |

**Pick (max objective, precision >= 0.5): k=6, margin=-0.05 (= -0.03 x the s1-s6 gap) -> precision 0.780, recall 0.535, issued 4.13/layer, objective 2.588.**

**W95f (review LOW) — the per-row set is effectively `<=5`, not 6.** At margin=-0.05 the trim threshold is the 6th-highest score + 0.05, which sits strictly *above* the 6th score, so the 6th-ranked candidate is always dropped to the `-1` sentinel — every row keeps at most its top 5. This is why the `k=6`, `k=8` and `k=12` rows are byte-for-byte identical at margin `-0.05` (all `issued 4.13 / hits 3.21`): ranks `6..k` are trimmed by construction. Read "k=6" here as "top-6 argpartition candidates, `<=5` surviving the confidence gate per row". Because of this the verify's dedicated per-row width `_RUNNER_V2_VERIFY_K_PER_ROW = 8` was dead (identical surviving set to 6) and has been removed; the verify uses the resolved AR `k`.
