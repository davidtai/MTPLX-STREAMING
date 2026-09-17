## 2026-07-13 06:47 [saved]
Goal: Make benchmark bottleneck diagnosis resource-based and agent-readable.
Decisions:
- Correlate uncached reader throughput with queue, worker, byte, fence, CPU, and GPU evidence on one clock.
- Keep telemetry opt-in and label enabled token rates diagnostic until reproduced without instrumentation.
- Report absent GPU and DRAM measurements as unavailable; emit candidates only when their required evidence exists.
- Treat cached reader bytes as logical demand; require F_NOCACHE for SSD-ceiling attribution.
Rejected:
- Infer serialization from elapsed time or low SSD use alone.
- Treat pending Metal fences as GPU utilization or routed bytes as DRAM traffic.
Open: Capture authorized process GPU samples when attribution remains incomplete.

## 2026-09-17 [saved]
Goal: Reduce DeepSeek V4.1 memory and exact-workload decode cost within 110 GB.
Decisions:
- Evaluate captured hidden means at the existing prefill fence; lazy source graphs otherwise retain earlier layer states.
- Retain depth five for the complete Python workload; the short prefix favored depth three.
- Reuse validated AR references with null current-run measurements and hashed diagnostic logits; candidate logits remain fresh.
Rejected:
- Staged 3+3 verification and its dependent allocation ladder.
- Tuned transition policy on full six-row route replay.
- Early projection reclamation with no measured benefit.
Open:
- Reach 20 TPS by reducing I/O and verification cost.

## 2026-09-17 12:55 UTC [saved]
Goal: Improve DeepSeek memory accuracy and reduce exact-workload decode cost.
Decisions:
- Preserve unavailable MLX peaks as null; headline and detailed memory values share one observation.
- Report effective DSpark depth separately from requested depth; requesting six does not extend the native five-token head.
- Keep cache policy and capacity changes conditional on measured full-workload allocation and throughput evidence.
Rejected:
- Serial rANS decoding costs more than its saved I/O time.
- Native HC chain compilation without verified numerical compatibility.
- More prefill fences without a measured whole-run peak reduction.
Open:
- Isolate prefill combine allocations when the exclusive GPU lane is available.
- Reach 20 TPS within the 110 GB whole-machine ceiling.
