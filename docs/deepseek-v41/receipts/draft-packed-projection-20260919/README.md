# Draft-only packed projection: measured memory and head-time reduction

One native/candidate/native head comparison at source
`a15ba7c40b6e59c7e3c3fdae852112085bddf40e` preserves all 198 commitment
boundaries and 1,242 verification rows. Controls take 1.867688667 and
1.866354375 seconds; the packed candidate takes 1.748210542 seconds, a 6.3637%
wall-time reduction against their mean. Control spread is 0.0715%.

The candidate uses original MXFP8 packed output weights with FP32 inputs and
outputs in the three native draft stages. Native inverse RoPE, the following
output projection, all 93/58/32 compact Q4 experts, and their lookup tables are
retained. The target trunk does not execute. The one changed proposal is at
position 315, depth 5: control 1926, candidate 2921. Both are rejected by the
teacher after four accepted proposals, so the commitment boundary is unchanged.
This is not a proof of fresh full-model parity or throughput.

Cache retirement releases exactly 402,653,184 bytes by both owner inventory
and before/after active-allocation measurement. Active memory after replay is
7,264,170,248 / 6,861,506,268 / 7,264,158,780 bytes. Cumulative MLX peak is
21,299,586,448 bytes, including setup; it is not an arm-specific peak. The
static incremental bound remains 49 GiB. Sampled whole-machine peak is
26,827,735,040 bytes and child footprint is 16,292,060,816 bytes; do not sum them.
No compressor growth is observed. The physical ceiling remains 110 decimal GB.

Guard and child exit 0. Post-child reclamation removes 15,342,174,208 cached
source bytes with zero remaining. Exact Qwen identity, health and warmup are
restored before lock release at 15:58:23 UTC. Independent 15:59:53 UTC evidence
confirms healthy idle Qwen, completed warmup, a free lock, and no owned process.

The retained full Q4 result remains 13.8688167379 TPS. This component saves
about 0.119 seconds of head work on the teacher trajectory; it does not by
itself close the 22.61-second full-workload gap to 20 TPS. Integrating it must
price memory separately for prefill, seed, bank allocation and steady decode;
there is no credit in a phase that never owned the removed dense caches. No
full model run or broad regression suite was added for this component receipt.
