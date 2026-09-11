# W28 — Per-layer routing-barrier overlap-fill (K1 Rank 1)

Status: SKELETON (worker feat/deepseek-v41-w28). CPU-only static + tiny-synthetic
analysis; no GPU, no real-artifact load. Author: Opus 4.8 worker.

Task: reduce the GPU-idle cost of the ~40 per-token `mx.eval(indices)` routing
barriers in the DSV4.1 streamed forward, behind a switch (default off), without
changing any output (byte-identical greedy tokens). Acts on KERNEL_LEDGER K1.

## 0. Headline
(to be filled)

## 1. Sync census (measured on the tiny synthetic streamed switch)
(to be filled)

## 2. Root cause: shared-overlap is NOT wired for the DSV4.1 MoE
(to be filled)

## 3. Implementation (behind `MTPLX_DSV41_SHARED_OVERLAP`, default off)
(to be filled)

## 4. Byte-identical proof
(to be filled)

## 5. GPU A/B arm for the orchestrator
(to be filled)

## 6. Risks / caveats
(to be filled)
