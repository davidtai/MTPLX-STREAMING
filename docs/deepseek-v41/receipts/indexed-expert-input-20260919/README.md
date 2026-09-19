# Direct token addressing in packed expert gate/up kernels

The candidate removes the intermediate gathered token rows: each gate/up
assignment indexes the original token buffer directly. Dot products, tiling,
reductions, rounding, activation, down projection and expert-read scheduling
remain unchanged. Construction selects the candidate kernel; the measured
path has no eligibility check or silent fallback.

Both arms have 110 expert rows, replay layer 34's 206 saved M6 routes, and
rotate all 40 native packed output projections with synthetic inputs. Target
attention and full generation are absent. The unchanged/candidate/unchanged
warmed times are 1.363446458 / 1.347701750 / 1.351989792 seconds. The candidate
is 0.7377% shorter than the control mean, within the 0.8438% control spread.
All 206 outputs per arm match exactly; each arm reads the same 759 records.
This is inconclusive and is not promoted or sent to a full-model run. No new
optimization regression test is added.

The first launch fails its source pin before the MLX child starts: staging
updated a `/tmp` alias without replacing the old `/private/tmp` pin. Root
files retain this refusal. The fresh `v2/` staging canonicalizes paths and
validates 33 unique pins before launch. Its guard and child exit zero; final
active MLX ownership is 8 bytes. The static incremental bound is 26 GiB,
including 10 GiB Metal/cache/compilation, 4 GiB host and 12 GiB source cache.
Whole-machine usage remains capped at 110,000,000,000 bytes.

The first guard restores and releases at 16:27:24 UTC. The measured v2 guard
restores exact Qwen identity, health and completed warmup and releases at
16:37:15 UTC. Its sampled physical peak is 19,150,700,544 bytes, sampled child
footprint 7,050,745,568 bytes, and compressor growth zero. Later independent
checks observe subsequent foreign windows and no owned child; they do not
replace or contradict the guard's restoration evidence.

Measured source is `b5f41c51d74a770069048dddd0a644f07886a999`.
`archive-sha256.json` pins the original diagnostic bytes. The retained complete
Q4 result remains 13.8688167379 TPS; 20 TPS is still open.
