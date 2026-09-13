# Attention projection cache ownership

Prefill retained an fp32 wo_a cache while fused decode built a separate bf16
transpose. Both remained owned across later requests. Each cache builder now
releases the opposite representation; cache hits keep their existing fast path.
Fused cache invalidation now covers scales and biases as well as packed weights.

The plan still reserves the full 5 GiB target fp32 cache when enabled, covering
prefill on later requests. Fused-only configurations now reserve their missing
2.5 GiB target cache. Native MTP's three fp32 caches remain separately priced.
No expert bank is resized after construction.

Focused CPU checks reproduced five failures before the fix. All nine ownership,
reload, and accounting cases plus five related MTP budget cases pass afterward
with real MLX imports blocked. The real-shape probe runs the actual two Attention
methods with one native mxfp8 group32 [8192,4096] projection on Metal. It verifies
exact dequantized weights through two prefill/decode transitions and a scales
reload. Source commit plus tracked-diff SHA are recorded in the JSON.

| State | MLX active bytes | Module-owned dense bytes | Module-owned fused bytes |
| --- | ---: | ---: | ---: |
| Packed weights | 34,603,024 | 0 | 0 |
| Prefill, both cycles | 168,820,752 | 134,217,728 | 0 |
| Decode, both cycles | 101,711,888 | 0 | 67,108,864 |

Peak MLX allocation including comparison temporaries: 403,701,908 B. The probe
uses a 3 GiB MLX limit, 128 MiB allocator cache, and independent 4 GiB process
guard. Guard exit 0; exact Qwen restored healthy and warm, lock released.

Across 40 target layers this removes 5,368,709,120 B of module-owned fp32 storage
after fused decode has visited every layer. These small-probe checkpoints prove
buffer reclamation, not a full-model peak or throughput improvement. In-flight
graphs can retain their inputs until they complete; the plan retains prefill
headroom. A full-workload measurement is still required before spending the
released memory on more expert slots.
