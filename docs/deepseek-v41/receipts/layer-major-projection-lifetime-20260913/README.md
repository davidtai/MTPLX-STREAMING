# Release completed layers' prefill projection caches

Layer-major prefill previously accumulated all target fp32 wo_a caches, although
each layer's final chunk had already consumed its projection. It now clears that
layer's dense cache after the existing `mx.eval(hs)` fence. Chunks within a layer
still reuse one materialized weight. Calls with at most eight logical rows keep
their caches, including verification with an explicitly small prefill chunk.
No new environment reads, device eligibility checks, or decode counters.

Two guarded CPU-pinned MLX checks use an eight-layer native mxfp8 group32 model.
They cover two requests, within-layer reuse, release before the next layer,
small verification retention, and subsequent nonfused decode cache rebuilding.
Logits, target hidden captures, and KV state match the uncached reference exactly.
The first attempt had a test-only nested-NumPy-array construction error; corrected
before observing the meaningful red result. Red2: large prefill fails because
the completed layer retains its cache, while small verification passes. Green:
both cases pass. All three guard runs restored exact Qwen health/warmup and
released the lock; green exited 0. These tiny shapes remain below the independent
3 GiB process cap and load no artifact.

Native MTP stage caches are outside the backbone loop and remain unchanged.
Nonfused decode rebuilds target fp32 caches on its first call after a large
prefill. The generic plan retains the full fp32 reserve for that lifetime and
other schedules. This proves ownership and exact state behavior, not a
full-model memory saving or throughput improvement. Next full-model admission
must use the previous measured peak plus exact slot delta and graph headroom,
without spending an unmeasured prefill-saving estimate.
