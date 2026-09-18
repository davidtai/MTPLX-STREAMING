# Mixed single/pair expert-row kernel

**Rejected: candidate/control latency ratio 1.0038815, control spread 0.6348%.**
All 206 native M6 operator outputs and physical reads match in every arm.
No full-model run or new regression tests follow this result. The retained full
result remains 13.1509467 TPS; 20 TPS is unmet.

The earlier row-pair integration used 32 slots and synthetic alternating routes,
and launched separate paired and singleton kernels. A CPU census on the native
206-cycle trace at 109 slots finds 31.58 logical cache-hit rows per layer call,
including 16.06 pairable rows. At the existing median-miss layer 34, those figures
are 31.94 and 15.94, with 16.00 singleton rows. Physical transient reuse is not
included in the census. It restores the captured 73-slot seed and empty growth.

The new operator handles both pairs and singletons in one dispatch per cache
bank. It keeps native V16/V8, four output rows, two SIMD groups, packed scales,
BF16 boundaries and each row's float2 accumulation order. An absent second row
has a sentinel, evaluates a zero second dot lane and suppresses its output write.
Every real output row has one writer. Miss parts retain the native operator;
the installed route table selects the new cache-hit operator only for M6.

The bounded real-runtime replay uses layer 34, all 206 exact native routes,
109 persistent slots, 48 transients and physically loaded 73-slot warm records.
Inputs and shared work are identical across native/mixed/native/mixed/native.
The first four calls warm kernels; later calls decide the sum of operator latency.
Output hashing occurs after each timed call; demand work is completed by that
call's final evaluation and no speculative reads run. This is an operator screen,
not full-model timing or an exact replay of the latest prefill-84 cache state.

The complete incremental bound is 9 GiB: 5 GiB Metal/cache/compiler and 4 GiB host.
Raw bank bound is 2,951,700,480 B. Guard process peak is 3,520,317,696 B and machine
peak 14,080,950,272 B, with zero compressor growth. Final Metal owners are 8 B.
Guard 58512 exits 0, reclaims source pages, restores exact Qwen, completes warmup
and releases the lock at 09:17:49 UTC. Independent health/model/free-lock checks
pass. Two preparation errors (a metadata key and a doubled temporary-path prefix)
were corrected before a GPU child was launched.

Measured source: `039e3bd811c64aa645dd89b5b8c85e1c3cf5ab53`. Installation hashes
pin the reused pair kernel, earlier evidence, runtime and final candidate helpers.
