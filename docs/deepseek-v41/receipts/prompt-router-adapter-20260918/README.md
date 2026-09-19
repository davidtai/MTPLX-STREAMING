# Learned look-ahead router screens

Prompt-trained score correction improves prediction quality and transfers
partly to the exact workload. It is unpromoted: no prefetch, predictor latency,
full-model parity or TPS is measured. The additional current-route feature is
not selected because its small coverage gain costs more false reads and twice
the adapter storage.

## Prompt-only training

The W35 trace contains a different 16K prompt and 256 AR decode rows. Target
layers 4–39 use the preceding layer's post-attention router input, available
before that layer's expert reads. The native next gate produces a baseline.
An affine ridge model learns its score residual from the final 1,024 captured
prefill rows: first 768 fit, final 256 select regularization 0.1 or 1.0, then
all 1,024 refit. No decode label fits or selects adapter parameters.

Native CPU top-six self-alignment is 99.9476%. Across all 256 decode rows,
top-six recall rises from 72.0775% to 75.1736%. The 36 affine adapters occupy
21,399,552 B. The CPU screen takes 3.3879 seconds with two BLAS threads and
real MLX imports blocked.

An auxiliary AR cache proxy replays captured routes through the native
110-slot transition-window policy, warming from the 2,048 saved prefill rows.
It is not the complete original prefill, an M6 cache snapshot or a latency
simulation. At two proposed experts per row, excluding residents:

| Predictor | Useful / issued | Precision | Coverage of 6,573 proxy misses |
| --- | ---: | ---: | ---: |
| Direct next gate | 1,423 / 1,876 | 75.8529% | 21.6492% |
| Prompt mean bias | 1,430 / 1,763 | 81.1117% | 21.7557% |
| Prompt ridge residual | 1,386 / 1,593 | 87.0056% | 21.0863% |

## Separate online label adapter

The existing exact 16K/1K M6 capture contains predicted scores and actual
routes, but no true gate scores. `exact/` therefore tests a different adapter:
the smallest margin correction that would place known top-six labels above
the predicted selection boundary. Cycles 0–15 fit, 16–31 calibrate issuance,
and 32–63 evaluate. This does not validate or refute prompt score regression.

On 4,878 held-out physical misses, direct prediction covers 495 at 84.3271%
precision; mean correction covers 532 at 87.0704%; ridge covers 519 at 84.5277%.
These modest gains do not justify a full run. The different calibration span
also makes this direct control distinct from the 763-miss control below.

## Transfer to the exact workload

`transfer/` trains the original score adapters only on the independent W35
prompt and applies them unchanged to the saved exact M6 capture. Inverting
`sqrt(softplus(z)) + bias` reconstructs the predictor features; forward
reconstruction differs by at most 9.536743e-7 in score space. Target routes
never change. Exact cycles 0–31 choose each layer's issue settings at at least
85% training precision, then cycles 32–63 score those fixed settings.

| Predictor | Useful / issued | Extra reads | Precision | Miss coverage |
| --- | ---: | ---: | ---: | ---: |
| Direct control | 763 / 897 | 134 | 85.0613% | 15.6417% |
| Transferred mean bias | 930 / 1,106 | 176 | 84.0868% | 19.0652% |
| Transferred ridge | 1,062 / 1,261 | 199 | 84.2189% | 21.7712% |

The ridge estimates 299 more useful reads with 65 more false reads than the
control. This is unlimited-lead-time prediction evidence. It excludes finite
prefetch capacity, predictor cost, demand contention and owner replacement.
The capture has 105 slots, while the best full run uses 110. No throughput
claim or automatic production policy follows these numbers.

## Current-expert conditioning

`route-conditioned/` adds the already-known current layer's six expert IDs to
the score features. Training and model selection remain on the independent
prompt; exact-run labels still only calibrate read issuance. The sparse route
term could be evaluated by summing six coefficient rows.

Its 42,743,808 B adapters cover 1,095/4,878 misses (22.4477%) but issue 219 false
reads at 83.3333% precision. This adds only 33 useful and 20 false reads over
the smaller ridge predictor. It is not selected for integration. The exact
holdout has been reused during research; it is not an untouched final gate.

## Complete memory and lifecycle

All four screens have a 1 GiB complete incremental CPU bound. Layers are
processed sequentially, source ranges use F_NOCACHE, and only metrics/hashes
survive each layer. No expert payload, target trunk or Metal operation loads.

| Screen / guard | Sampled process peak, B | Sampled machine peak, B | Samples |
| --- | ---: | ---: | ---: |
| Prompt / 45880 | 264,946,336 | 11,263,098,880 | 4 |
| Online labels / 35770 | 13,926,880 | 10,855,284,736 | 1 |
| Transfer / 27916 | 433,963,800 | 11,023,679,488 | 5 |
| Route-conditioned / 92506 | 432,947,992 | 10,979,508,224 | 5 |

The online-label screen finishes between polls: its endpoint process footprint
is 250,364,552 B and machine usage 11,067,080,704 B. The final route-conditioned
endpoint process footprint is 464,569,136 B, also above its sampled peak.
Guard-accounted peak for that screen is 11,014,112,024 B, distinct from machine
physical use. Limits, endpoint observations and sampled peaks are kept
separate; these overlapping quantities must not be added together. All guards
observe zero compressor growth.

Measured source: `61b6899f27fee89f63bf7eed9f1c8aa77e73a53a`. All four guards exit
0 and restore exact Qwen identity/warmup before releasing the lock. Transfer
waits for another owner, then releases at 12:49:13 UTC. An independent health
check subsequently overlaps another owner's Qwen shutdown and sees connection
refused; that owner is left alone. The successful retry at 12:51:40 UTC verifies
healthy/idle/warmed/free state. Final guard 92506 releases at 12:53:34 UTC;
independent verification passes at 12:55:42 UTC. No owned child remains.

No new regression tests or full benchmark runs follow these unpromoted
screens. A higher-capacity predictor can next learn directly from the full
hidden vector instead of its 384-score projection; it first needs a bounded
CPU quality check and an explicit parameter-memory cost.
