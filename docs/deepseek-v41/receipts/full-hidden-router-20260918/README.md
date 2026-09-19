# Full-hidden causal router screen

The larger full-hidden predictor is rejected for this fit and workload. It
does not improve useful missed-expert predictions over the compact score
adapter and would require 283,170,816 B of folded parameters instead of
21,399,552 B. No target model, real prefetch, Metal operation, full benchmark
or new regression test follows this screen. The retained full result remains
13.4141517619 TPS; 20 TPS is unmet.

## Mechanism and scope

The previous layer's native post-attention router input is available before
its expert reads. A dual ridge regression maps this full 5,120-element input
to the residual in the next gate's 384 raw logits. It preserves the native
gate as a prior. The affine correction folds into one effective prediction
matrix and raw-logit bias, followed by the native biased sqrt-softplus scores.
The actual target router and generation arithmetic never change.

The historical W35 trace uses a different 16K prompt and 256 AR decode rows.
For each layer 4–39, the final 1,024 captured prefill rows train the model:
768 fit, 256 select lambda 1 or 10 by top-six overlap then score MSE; all
1,024 then refit before evaluating the 256 decode rows. No decode label fits
or selects parameters. The existing compact score model is freshly refitted
using its original lambda 0.1/1 selection. Folded versus unfolded raw-logit
error is at most 0.000118256; the screen uses the folded predictions.

| Predictor | Decode top-six overlap | Useful / issued at two proposals | Precision | Proxy miss coverage |
| --- | ---: | ---: | ---: | ---: |
| Direct next gate | 72.0775% | 1,423 / 1,876 | 75.8529% | 21.6492% |
| Compact score residual | 75.1736% | 1,386 / 1,593 | 87.0056% | 21.0863% |
| Full-hidden raw residual | 75.0778% | 1,366 / 1,588 | 86.0202% | 20.7820% |

Native CPU self-alignment is 55,267/55,296 = 99.9476%. The cache proxy replays
the captured 2,048 prefill routes into a native 110-slot transition-window
bank with 48 transient slots, then evaluates 256 AR routes with 6,573 misses.
It excludes resident experts from proposed reads. It is not the complete
original prefill, the exact M6 physical cache or a timing simulation.

This rejects the measured affine raw-residual variant, not every possible
full-hidden/nonlinear predictor. Both feature dimension and regression target
differ from the compact score model; the result does not isolate either.
The saved exact M6 capture lacks full hidden vectors, so it cannot validate
this variant by the compact adapter's score inversion method.

## Memory and lifecycle

Measured source: `61b6899f27fee89f63bf7eed9f1c8aa77e73a53a`. The complete
incremental CPU bound is 1 GiB; the child blocks real MLX imports, uses two
BLAS threads, reads ranges with F_NOCACHE and processes layers sequentially.
Only metrics and fitted-parameter hashes survive each layer. The 36 folded
matrices are priced for hypothetical installation, not retained by this probe.

Elapsed CPU-screen wall time is 5.592968 s. Guard session 56562 exits 0, with
six samples: process footprint peak 333,365,968 B, machine physical peak
12,367,265,792 B and guard-accounted peak 12,381,422,288 B. These overlapping
metrics remain separate. Endpoint footprint is 333,431,504 B, slightly above
the sampled peak; endpoint machine usage is 12,317,032,448 B. Compressor
growth is zero. No source model pages remain cached.

The guard restores exact Qwen and warmup before releasing the lock. An
independent healthy/idle/warmed/exact-model/free-lock check passes at
13:06:45 UTC. No owned child remains.
