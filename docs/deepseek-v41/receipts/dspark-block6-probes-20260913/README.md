# Bounded DSpark block-6 and target M7 screens

Source `8a8803ad86108c3448331a8e880fe2a0b891e809`. These are small guarded
screens, not a full-model run or a throughput result for the 16K Python task.

The native-head-only probe uses the exact 10,597,621,640 B of MTP, embedding and
output-head parameters from the pinned artifact. It changes only the constructed
draft block length from 5 to 6 (all three stages and the head). The archived
head-probe loading bound of 25,249,119,226 B remains above the observed
15,415,845,136 B MLX loading peak. Synthetic 16K main-hidden seeding added
2,473,893,804 B to the active peak, as in the block-5 control. Eager and
compiled seven-ID outputs (one input plus six drafts) were exactly equal. The
first five draft suggestions differed from the block-5 synthetic probe after
the bidirectional draft attention saw an extra position. Real-context
acceptance is therefore unmeasured.

Median warm head-only latency was 11.104 ms eager and 10.132 ms compiled,
versus 10.304/9.358 ms in the earlier block-5 probe. The compiled cold call
added 21,086,017 B above the active baseline, versus 18,638,211 B at block 5.
These synthetic head timings do not include target verification or expert SSD
reads. The current sampled physical peak was 25,886,015,488 B. Guard exit 0,
exact Qwen restored with warmup done, and a fresh health/lock check passed.

The separate tiny target check uses one identical parameter set per arm and
compares K22 eager versus compiled verification at M=1,4,6,7. With the 12-row
prefill eager in both arms, logits and all 12 sampled KV/cache arrays were
bitwise equal at every size. There were no argmax changes. The first exploratory
M7 run used a 32-row compile cap and unintentionally compiled the 12-row
prefill in one arm; it compared different starting cache states and failed.
A second diagnostic showed the same small discrepancies already at M1. The
controlled cap-7 run resolved them. All three raw guard logs are retained; no
numerical tolerance was widened. Each guard restored Qwen and released the
exclusive lock, verified independently before the next GPU run.

The target check is tiny and CPU-pinned MLX, under the GPU guard. It does not
establish a native-geometry M7 compile/graph peak, cache rollback across real
ring and compressor boundaries, or block-6 acceptance on the actual Python
context. The subsequent paired 16K/64-output screen measured acceptance and
rejected block 6 for that slice; see
[its receipt](../dspark-block6-short-pair-20260913/README.md). No block-6
performance lane was installed.

One separate synthetic attention layer at native DeepSeek dimensions and MXFP8
gs32 projection layout ran M6 and M7 over 16,384 cached positions in all four
attention modes. Its MLX peak was 1,132,996,826 B under an 8 GiB cap; no new
swapouts. This is a shape/memory screen, not arithmetic parity or target TPS.
The observed wrapper recorded `complete:false` because the probe CLI normally
raises `SystemExit(0)` after writing its valid result. That raw bound is intact;
`verify-attn-real-m7.completion-validation.json` separately checks the receipt,
source/wrapper identity, exit 0, exact Qwen health/warmup and free lock. The
archived fixed wrapper handles that normal exit for future runs and was not
used to produce these observations.
