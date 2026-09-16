# DeepSeek V4.1 20 TPS Stage Design

## Goal

Move the exact 16,384-token Python prompt and 1,024-token DeepSeek V4.1
generation from the measured 9.6589676 decode tok/s toward at least 20 tok/s,
while keeping whole-machine physical use below 110,000,000,000 bytes.

## Scope

This stage corrects projection-cache lifetime accounting, fixes the benchmark's
model-owned Engram telemetry, installs a bounded causal cache-admission
candidate for the MTP verification route, and submits the resident shared
branch while target-expert miss reads are open. Each performance candidate
remains construction-selected so an unchanged control can be measured without
a hot-path fallback.

The stage does not load MLX or run Metal while another GPU job owns the machine.
It does not build a compressed expert artifact. Component-separated lossless
compression is a later stage only if the cache and scheduling candidates win
but the exact workload remains below 20 tok/s.

## Measured basis

- Exact best: 1,023 decode steps in 105.9119402 seconds, or 9.6589676 tok/s.
- Verification: 102.8307011 seconds over 206 cycles.
- Expert I/O: 49,298 records and 926,833,950,720 bytes in 70.4950996 seconds.
- Cap-92 causal replay: 39,580 records. Longer-history and row-position
  policies did not improve held-out behavior materially.
- Component-separated rANS: 10.4792% smaller on the exact record, but the
  resulting 50.67-second I/O floor leaves too little compute margin by itself.

## Memory accounting

For non-direct, layer-major target attention with both the fp32 prefill cache
and bf16 fused-decode cache enabled, the live target-cache peak is:

```text
max(
    NUM_TEXT_LAYERS * fused_bytes,
    dense_bytes + (NUM_TEXT_LAYERS - 1) * fused_bytes,
)
```

It is not `NUM_TEXT_LAYERS * max(dense_bytes, fused_bytes)`: only the current
layer owns the fp32 prefill form while completed layers own the bf16 decode
form. Other cache modes keep their existing formulas. This removes
2,617,245,696 bytes of false reserve for the exact configuration.

With the unchanged 88,690,793,288-byte engine budget and 7 GiB in-plan runtime
reserve, the corrected fixed footprint is 25,535,486,792 bytes and admits 83
target slots per layer. That is the default safe candidate.

The outer 110 GB target planner separately reserves the measured transient band,
while the inner plan still carries the historical 7 GiB transient reserve. A
staged reserve rebalance can admit at most cap 91, but only behind the exact
runner's construction-time bound. Relative to the measured cap-80 run, cap 91
projects to 97,894,449,392 allocator bytes, leaving just 211,466,960 bytes under
the 98,105,916,352-byte allocator limit. Its external whole-machine projection
is 107,876,024,320 bytes, leaving 2,123,975,680 bytes under the hard ceiling.
Cap 92 is excluded: its 98,646,474,992-byte allocator projection exceeds the
limit by 540,558,640 bytes even though its whole-machine projection remains
below 110 GB. The first staged capacity arm is cap 89; cap 91 is attempted only
if cap 89 confirms the projection and leaves the required measured headroom.
Python remains bounded to 2,131,214,336 bytes for Engram payload, Engram
metadata, and other host reserve.

The runners expose the in-plan portion as `--runtime-reserve-gib`. The default
stays at 7 GiB (cap 83); 3 GiB selects cap 89, and 2 GiB selects cap 91 for the
staged screen. The runner refuses values below 2 GiB and records the installed
reserve in bytes. The outer measured transient band and 110 GB guard remain in
force for every arm.

## Runner telemetry

The benchmark passes a bare expert runtime to the shared counter collector, but
the Engram banks belong to the model. The snapshot interface therefore accepts
the model explicitly while retaining the runtime for expert and I/O counters.
This restores `engram_row_cache` without importing MLX or adding measured-path
instrumentation.

## Causal cache admission

The candidate is selected when the streaming runtime is constructed. It keeps
per-layer bounded state for 384 experts:

- a one-step expert transition table;
- a 16-route frequency window;
- existing last-use epochs.

After observing a decode route, it scores the next-route cache candidates with
0.7 transition probability, 0.2 normalized window frequency, and 0.1 recency.
Current misses below the persistent-admission cutoff use the already bounded
shared transient slots instead of displacing a stronger resident. Prefill
seeding and non-selected cache policies retain their current behavior.

The benchmark selects this lane with `--cache-policy transition-window`.
Omitting the option inherits the served profile policy. The receipt records the
installed policy and whether the single slot pool was constructed.

No environment read, invariant validation, proof counter, fallback, or retry is
added to the measured route. Configuration validation happens once. The true
router indices still determine every expert gather, so cache decisions cannot
change model arithmetic.

## M6 shared-work overlap

The single-barrier verification path currently submits hit gathers and then
blocks in `iter_ready_misses()` before invoking `shared_work()`. When decode
misses are pending, it will claim the existing pipeline work item, construct
the shared branch once, submit it through `mx.async_eval`, and close the claim
before waiting for miss readiness. The existing post-route call remains for
all-hit and non-decode cases. A non-null shared result prevents duplicate work.

This lane defaults off and is installed with `--verify-shared-overlap`. Switch
construction binds the submit callable once and refuses the candidate if
`mx.async_eval` is unavailable. The hot path does not read an environment flag
or fall back to the control. Both runners stamp the installed boolean in the
resolved-plan receipt, so the matched control and candidate are auditable.

## Validation and promotion

No new optimization regression tests are added before measurement, following
the requested workflow. The planner correction updates its existing arithmetic
contract because incorrect memory admission is a safety bug.

When the GPU lane is available, the guard must acquire
`/tmp/mtplx-gpu-exclusive.lock` before stopping Qwen. A short matched arm batch
will compare the corrected cap-83 control, causal cache admission, and
shared-work overlap one change at a time. The winning scheduling/cache stack is
then screened at cap 89. Cap 91 is eligible only after cap 89's measured peak
validates the static projection; cap 92 is not an arm. Only the winning stack
receives an exact 16K/1K run and focused regression tests. Every full run must
report MLX peak, process `phys_footprint`, whole-machine physical peak, token
digest or an allowed tie flip, physical record reads, wall time, and exact Qwen
restoration.

## Failure modes

- The single trace may overstate cache gains. The construction-selected lane is
  rejected and removed unless a matched workload A/B reduces wall time.
- A changed machine baseline may make cap 89 or cap 91 unsafe. Both allocator
  and whole-machine projections are gates; live admission must lower or refuse
  the cap rather than cross either limit.
- Shared work could be submitted twice or outlive its pipeline claim. A single
  stored result and the existing claim/close protocol prevent duplication.
- Lossless decode cost may erase compression savings. No compressed artifact is
  promoted without a real-record decoder microbenchmark that clears the
  end-to-end required margin.
