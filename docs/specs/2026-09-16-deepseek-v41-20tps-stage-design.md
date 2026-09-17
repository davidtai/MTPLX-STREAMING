# DeepSeek V4.1 20 TPS Stage Design

## Goal

Move the exact 16,384-token Python prompt and 1,024-token DeepSeek V4.1
generation from the measured 9.6589676 decode tok/s toward at least 20 tok/s,
while keeping whole-machine physical use below 110,000,000,000 bytes.

## Scope

This stage corrects projection-cache lifetime accounting, fixes the benchmark's
model-owned Engram telemetry, installs a bounded causal cache-admission
candidate for the MTP verification route, and submits the resident shared
branch while target-expert miss reads are open. It also adds a bounded decode
miss-part candidate so completed expert records can reach Metal before the
slowest read in the layer finishes. A staged target-verification candidate
stops before later verify rows when an earlier draft rejects. Each performance candidate remains
construction-selected so an unchanged control can be measured without a
hot-path fallback.

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
- Python raw-record L2: at the same 1,504,051,200-byte cost as two target slots
  per layer, an idealized cap-89 victim cache still needs 40,239 SSD reads in
  the saved-route replay; cap 91 with those bytes in Metal needs 40,111 and
  avoids eviction copies. Keep the 2 GiB host allowance for bounded Python
  caches and metadata rather than adding a duplicate expert-record cache.
- A cap-83 scheduling-shape replay has 8,033 miss-bearing layer calls across
  8,240 calls. Those calls average 5.90 physical records; 6,768 layer calls
  have at least three misses. A three-record bound therefore exposes about
  2.25 completion groups per layer call instead of one layer-wide completion.
- Exact-route replay with the measured acceptance-depth distribution projects
  49,270 target record reads for a six-row verify at cap 80. A `3,3` schedule
  projects 46,355 reads (5.92% fewer) and 1,107 evaluated rows (10.4% fewer);
  `2,2,2` projects 45,616 reads (7.42% fewer) and 1,076 rows (12.9% fewer).
  These are CPU route projections, not GPU throughput results.

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

## Bounded decode miss completion

The current overlap route submits every decode miss in a layer as one outer
future. Its inner reader futures preserve SSD queue depth, but the outer future
becomes ready only after every record finishes. The switch therefore cannot
dispatch any miss gather while a slower record from the same layer is still in
flight.

The candidate keeps all miss parts submitted concurrently and caps each outer
part with `--decode-miss-records-per-part`. Before generation, construction
requires the DeepSeek-V4.1 component-bank path, raw sidecar records, the
existing overlap route, and verified placement for every record. At runtime,
loads are ordered by sidecar part and byte offset; boundaries prefer physical
gaps so a contiguous run remains one scatter read unless the run itself exceeds
the configured cap. Router assignment order, slot ownership, policy commit,
rollback, and gather recombination retain the existing split-route contracts.

`None` is the unchanged one-part control. The first candidate is three records
per part; two records is a follow-up only if the three-record arm confirms that
earlier completion outweighs the extra gather submissions. The actual value is
stamped in the resolved plan and runtime snapshot. No environment read,
eligibility probe, retry, or fallback is added to the enabled route.

## Staged target verification

The default DSpark route retains its single `K+1` target forward. The candidate
partitions those input rows at construction with `--dspark-verify-chunks`; for
the exact depth-five workload the first arm is `3,3`. Each target chunk is
evaluated and accepted before the next chunk is submitted. A rejection ends the
cycle immediately, so later target rows and their expert reads do not run.

Chunk boundaries preserve the causal sequence. For `3,3`, the first forward
consumes `[primary, d1, d2]` and verifies `d1`, `d2`, and `d3`. If all three
drafts pass, the second forward consumes `[d3, d4, d5]`, verifies `d4` and `d5`,
and produces the bonus row. The target cache therefore contains the same six
logical inputs after a full accept. On rejection, the pre-cycle snapshot trims
the total rows actually forwarded back to `[primary, accepted drafts]`.
Concatenated target hiddens seed the DSpark windows from that committed prefix.

Greedy and sampled acceptance both run at the chunk boundary. Sampled mode
retains the original RNG order: it draws once per reached draft and draws the
same residual correction or final bonus, while unreachable later chunks consume
no random values. `cycles` remains the logical speculative-cycle count;
`verify_calls` reports actual target forwards and can exceed `cycles` only on a
staged route. The installed schedule is stamped in DSpark statistics and the
benchmark receipt.

Configuration must be a non-empty positive partition whose sum is exactly
`speculative_depth + 1`; invalid schedules fail before model work in the
runners. The enabled loop does not read environment state, recheck model
metadata, or fall back to the one-shot route.

## Validation and promotion

No new optimization regression tests are added before measurement, following
the requested workflow. The planner correction updates its existing arithmetic
contract because incorrect memory admission is a safety bug.

When the GPU lane is available, the guard must acquire
`/tmp/mtplx-gpu-exclusive.lock` before stopping Qwen. A short matched arm batch
will compare the corrected cap-83 control, three-record miss parts, causal cache
admission, and shared-work overlap one change at a time. Two-record miss parts
are screened only after the three-record arm improves wall time. The `3,3`
staged-verify arm is screened separately; `2,2,2` is attempted only if `3,3`
improves decode wall time. The winning scheduling/cache stack is then screened
at cap 89. Cap 91 is eligible only after
cap 89's measured peak validates the static projection; cap 92 is not an arm.
Only the winning stack receives an exact 16K/1K run and focused regression
tests. Every full run must report MLX peak, process `phys_footprint`,
whole-machine physical peak, token digest or an allowed tie flip, physical
record reads, wall time, and exact Qwen restoration.

## Failure modes

- The single trace may overstate cache gains. The construction-selected lane is
  rejected and removed unless a matched workload A/B reduces wall time.
- A changed machine baseline may make cap 89 or cap 91 unsafe. Both allocator
  and whole-machine projections are gates; live admission must lower or refuse
  the cap rather than cross either limit.
- Shared work could be submitted twice or outlive its pipeline claim. A single
  stored result and the existing claim/close protocol prevent duplication.
- Smaller miss parts add outer-future and gather-dispatch overhead. The
  construction flag defaults off, and the candidate is removed unless its
  matched wall time improves despite that overhead.
- Staged verification adds target-forward and host acceptance boundaries on
  cycles whose first chunk fully accepts. The route is rejected unless avoided
  expert reads outweigh those boundaries in matched decode wall time.
- Lossless decode cost may erase compression savings. No compressed artifact is
  promoted without a real-record decoder microbenchmark that clears the
  end-to-end required margin.
