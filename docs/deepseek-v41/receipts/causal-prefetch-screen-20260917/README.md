# CPU causal prefetch screen: rejected

A 1.535-second CPU-only screen uses the complete saved 206-cycle M6 route
trajectory and the actual transition-window bank at capacity102. The first
103 cycles train online predictors and select layers; the last103 evaluate
without looking ahead. Real MLX imports are forbidden. Initial residency is
expanded from the historical prefill state and is not the later packed run's
exact initial residency; the first half warms the banks.

Previous-route, same-layer transition, cross-layer transition and blended
predictors issue at most1,2,4 or6 currently nonresident experts per layer.
Previous-route entries are almost all resident already. The best nontrivial
precision is cross-layer top1:550 useful predictions out of4017 (13.69%).
Against15544 held-out physical reads it could move3.54% earlier while adding
3467 reads (22.30%). No layer meets the first-half80% precision gate.

This deliberately optimistic screen gives correct predictions unlimited lead
time and excludes contention and bank pollution. It does not measure latency,
throughput, or actual prefetch completion. It is sufficient to reject these
predictors without changing production code, allocating a prefetch ring,
interrupting Qwen or adding optimization tests. The two transition matrices and
denominators total47,308,800 CPU bytes.

Existing GPU gate prediction is a different mechanism, with earlier results
in `W93_GATE_PREFETCH.md` and `/tmp/dsv41-gate-prefetch-cache-screen.json`.
The current runtime explicitly rejects prefetch with transition-window; this
screen does not weaken that construction guard. The20TPS goal remains open.
