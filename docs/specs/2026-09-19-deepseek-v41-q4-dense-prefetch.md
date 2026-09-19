# Predictable dense-weight prefetch for Q4

The user stopped Q2 and requested returning to Q4, specifically questioning
whether predictable non-MoE weights can be unloaded to make more room for MoE.
Keep the original 16,384-input/1,024-output workload and 110,000,000,000-byte
whole-machine limit. The best saved Q4 result is 13.4141517619 TPS, one full
run with all native output IDs unchanged; it is not a repeated speedup result.

## Candidate

The forty target query-up projections (`attn.wq_b`) have identical native MXFP8
storage shapes and total 1,730,150,400 bytes. They execute in fixed layer order.
Two rotating buffers require 86,507,520 bytes, giving a payload reduction of
1,643,642,880 bytes. This can fund two additional 17,694,720-byte expert slots
per layer, subject to pricing staging, allocator padding, growth copies, and
fresh baseline/wired-memory admission. Moving weights to ordinary Python RAM
does not provide this saving on unified memory: the persistent source arrays
must retire and future weights must be read from disk.

Keep the native quantized values, shapes, scales and arithmetic. Install the
streaming route at the decode transition after prefill; generic prefill retains
its existing route. Prebind file offsets, array layouts and per-layer callables
at construction. Prefetch known future layers into a fixed buffer ring. Reuse
a buffer only after the GPU has consumed its last reader. All-hit/device-only
routes cannot silently substitute for a required retirement event.

The existing Q4 input-row cache already releases 1,323,827,200 embedding bytes
after prefill. Target `wo_a` sources are already retired after native BF16
materialization; its actual remaining representation is 2,684,354,560 bytes.
Do not count either saving twice or use the smaller retired source to price a
larger streamed representation.

## Measured evidence and limits

The CPU inventory replays the historical 73-slot native route trace exactly:
53,999 physical reads. On the same 206-cycle route, 110-to-112 slots saves
13,430,292,480 expert-read bytes. Streaming wq_b adds 356,410,982,400 dense-read
bytes, a net increase of 342,980,689,920 bytes. The full best run has 198 hybrid
cycles; these older native routes are sensitivity evidence, not a replay of
the hybrid run or a current TPS prediction.

Predictability makes scheduling possible; this byte calculation shows that
success requires substantial overlap with existing computation. It does not
prove that overlap exists. Do not claim a memory-capacity projection as a TPS
win. Streaming all non-MoE families at once would hide attribution and add much
more traffic, so begin with one uniform family.

## Next bounded stage

1. Identify an existing completion boundary that proves query weights have
   been consumed, preserving the current GPU submission schedule. The ordinary
   CPU-routed MoE path needs router values dependent on attention output; verify
   that this boundary covers every installed decode route before reusing it.
2. Build a bounded real-weight comparison of resident versus two-buffer reads
   with identical query matmuls and representative expert I/O. Test distinct
   layers and buffer wraparound. Prove exact outputs and memory bounds before
   timing; use a short alternating comparison rather than a broad test suite.
3. Price all I/O concurrency, direct-destination buffers, file-cache effects and
   compilation headroom before the guarded component run. Preserve demand
   expert-read priority; predictive dense reads must not starve that queue.
4. Promote only a measured component win to the full 16K/1K runner, update its
   admission from actual retirement and fresh baseline, then inspect output.
   Add focused regressions only after successful optimization.

Risks to resolve are buffer reuse before Metal completion, retained graph or
module references defeating retirement, source file-cache duplication, and
shared-SSD contention. No production route, GPU experiment, or new test has
been installed for this candidate yet. Q2 remains stopped.
