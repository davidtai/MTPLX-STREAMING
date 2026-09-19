# Q4 compact router correction with finite prefetch lead time

Neither schedule is promoted. The first-GU schedule is slower; moving the
issue point earlier is essentially flat. The retained full-model result
remains **13.4141517619 TPS**, and **20 TPS remains unmet**.

Measured source: `3dcc1605404c860342601479dad1d4a5bc10f9bf`. Scratch root:
`/tmp/dsv41-ridge-prefetch-20260919`. Q2 remains stopped.

## Workload and frozen predictor

This extends the earlier adjacent-layer cost screen with the compact ridge
correction selected by the independent W35 prompt. The two required adapters
reproduce their prior fitted parameter hashes exactly. Preparation takes
0.111325 seconds on the CPU and exits before GPU loading. The parameters and
saved corrected scores occupy a 2,371,114-byte NPZ. No new tuning occurs.

The replay uses real Q4 layers 30–32, 105 persistent slots per layer, 48 shared
transient slots and one shared 16-slot prefetch ring. Each arm physically loads
the saved 73 resident experts per layer and restores the same policy state.
The first 32 native M6 cycles warm the cache; the last 32 are the held-out
timing cohort. There are no per-layer diagnostic gaps. Final timing includes
speculative-read draining and GPU completion.

The GPU pays for native FP32 router projection, sqrt-softplus and the affine
correction on synthetic BF16 inputs. Recorded corrected scores select reads;
attention is omitted. This measures scheduling and operator cost, not live
full-model prediction quality or throughput. Target routes, weights, arithmetic
and native slot publication stay unchanged. CPU ranking and the existing
demand-priority reader queue are included in candidate time.

## One A/B/A comparison per schedule

| Issue point | Native A (ms) | Candidate B (ms) | Native A (ms) | Change vs native median | Control spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| First missing-part gate/up witness | 657.924459 | 663.384000 | 654.422875 | +1.09885% | 0.53364% |
| After demand and resident/shared GPU submission | 659.422791 | 657.999583 | 660.648750 | −0.30850% | 0.18574% |

All 192 layer outputs per arm match exactly in both comparisons. The earlier
issue point improves completed-before-demand prefetches from 81 to 92;
awaited-inflight prefetches fall from 128 to 117 out of 209 issued. Both
candidates still read 1,250 records versus 1,214 for their controls, adding
637,009,920 bytes. These counters include all 64 cycles; reader counters also
include initial bank seeding. They are not held-out-only measurements, and
assignment hits must not be confused with physical records.

Over all 64 continuously timed cycles, the first candidate is 30.5941% longer
and the earlier-issue candidate is 0.02358% longer than its controls' median.
The first cohort includes warmup and potential compilation/cache effects.
Do not compare the two candidates across separate windows as an isolated
speedup. The second run's small held-out improvement and flat whole replay
do not justify full-model integration or new optimization regression tests.

## Ownership and memory

The earlier issue point follows submission of current demand reads and
resident/shared GPU work. It can overlap more source-layer work, but a running
speculative read can still delay demand enqueued later.

The shared ring never recycles a target-minus-one tenant, protecting the
current source layer. Physical slot ownership also waits for loading writers
and GPU-consumer pins. Previous probe outputs are evaluated before the next
layer releases deferred leases. Ticketed completion, route transactions and
failure cleanup retain their native ownership rules. `summary.json` records
the source review and file identities; output parity is additional evidence,
not a substitute for buffer-lifetime reasoning.

Both GPU runs use a **14 GiB incremental bound**: 10 GiB for Metal, allocator
cache and compilation, plus 4 GiB for host, reader and compiler memory. The
379 total slots represent 7,125,442,560 raw bytes or 6,706,298,880 packed bytes.
The 70,778,880 retained output bytes and 1,188,864 adapter bytes are inside the
Metal envelope. CPU preparation has a separate 1 GiB bound and finishes first.
The 110,000,000,000-byte whole-machine and 100 GiB wired ceilings are unchanged.

| Sampled metric | First-GU | Earlier issue |
| --- | ---: | ---: |
| MLX allocator peak (B) | 7,365,486,097 | 7,365,486,097 |
| Child-tree Darwin footprint peak (B) | 7,526,357,160 | 7,872,499,744 |
| Whole-machine physical peak (B) | 19,159,941,120 | 18,627,559,424 |
| Guard accounted peak (B) | 19,159,941,120 | 18,899,767,328 |
| Guard memory samples | 14 | 13 |
| Compressor growth (B) | 0 | 0 |

These metrics overlap. Guard accounting is the maximum of measured physical
use and its baseline-plus-footprint estimate; it is not always the physical
measurement. Final MLX active ownership is 8 bytes in each run. The roughly
240 MB retained between arms belongs to shared gates/scales/scores and is
released at final cleanup.

Some copied descriptive fields in the raw installation records still refer
to older 157/274-slot and row-permutation screens. Raw evidence is preserved
unchanged. `summary.json` explicitly identifies those stale fields and gives
the corrected 379-slot inventory. Actual admission uses the correct 14 GiB
top-level bound, current `budget_components`, 10 GiB allocator limit and
379-slot plan; the no-MLX construction check verifies these agree.

## Lifecycle and preservation

Both children and guards exit zero. Exact Qwen identity, health and completed
warmup are restored before lock release at 13:14:54 and 13:26:40 UTC. Source
file-cache bytes are zero after each run. The independent 13:29:31 observation
finds another job holding the lock and its service temporarily unavailable;
no owned GPU child or waiter remains. Foreign jobs are left alone.

The local checkpoint `q4-ridge-prefetch-inputs.tar.gz` preserves 61 exact
scratch files, including both parameter and route inputs. All archived bytes
were hash-verified; its location and digest are in `input-checkpoint.json`.
Curated source and result files are included here; large model/scale artifacts
remain at their pinned original paths. These historical scripts retain their
original paths and require restaging before replay. No production default or
target arithmetic change is included.
