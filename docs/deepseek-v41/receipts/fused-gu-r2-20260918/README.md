# Smaller fused gate/up tile

The R2 gate plus R2 up fusion is 0.5130% slower in the layer replay, with
0.4406% control spread. It is not promoted; no full-model run or new regression
tests follow it.

This differs from the older R4 plus R4 fusion. The new kernel uses four FP32
accumulators per thread, matching one native R4 projection, rather than eight.
It shares the input load within each thread and combines two launches. Its
aggregate threadgroup count equals the two native projections combined; it
does not halve aggregate input loads. There is no measured register-spill
claim. V16/K5120, scale decoding, dot and accumulation order, SIMD reduction,
separate BF16 gate/up outputs, native clamped activation and down kernel stay
unchanged. Grouping, physical-slot versus expert-scale identity, reader pool,
part readiness and owner lifetimes remain native.

Five warm totals, native/candidate/native/candidate/native:
1,227,682,043; 1,228,201,969; 1,222,282,629; 1,235,416,626; 1,225,522,617 ns.
All 206 layer-34 outputs and physical read counts match in every arm, with
734 warm reads. The replay uses synthetic BF16 inputs and captured routes,
110 persistent/48 transient slots, per-call timers and outside-timer hashing.
It does not measure full-model throughput.

The static incremental bound remains 9 GiB, including host and compiler
allowances. Fusion adds no output allocation or threadgroup scratch. MLX peak
is 3,047,281,161 B; final owners total 8 B. Across 14 observations, the guard
samples a 3,372,780,928 B process-tree peak and 15,597,895,680 B machine peak,
with zero compressor growth. Those scopes overlap and are not added together.

Measured source: `61b6899f27fee89f63bf7eed9f1c8aa77e73a53a`. Guard 39230 exits 0,
reclaims source pages and restores exact Qwen identity/warmup. Lock release is
12:21:17 UTC; independent health/idle/warmup/free verification passes at
12:22:20 UTC. The native default remains installed.
