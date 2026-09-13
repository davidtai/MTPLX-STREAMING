# 110 GB memory calibration

These receipts use decimal GB for the machine budget. They are memory and
correctness calibration, not evidence of the 20 TPS target.

## Findings measured on 2026-09-13

The machine is an Apple M5 Max with 128 GiB physical RAM and an unchanged
100 GiB wired limit. The unloaded baseline initially included about 38 GB of
file-backed pages. That cache belongs in physical memory reporting even though
the OS can reclaim it.

The OS-only purge probe made no change: unprivileged `purge` was refused, and
`sudo -n /usr/sbin/purge` required a password. A successful `sudo -l` permission
query did not establish password-free execution. The guarded attempt restored
the exact service with warmup complete. The memory budget continues to include
the measured file cache.

The benchmark omitted the expert profile's `bypass_page_cache` setting. Engram
row reads also used buffered descriptors, duplicating the bounded Python LRU
in the macOS file cache. Both now install `F_NOCACHE` at construction on macOS.
The resolved expert and Engram modes are included in new receipts.

| Measurement | Result |
| --- | ---: |
| Random Engram payload read per probe | 4,325,376 bytes |
| Buffered probe file-cache growth | 273,039,360 bytes |
| Uncached probe file-cache growth | 0 bytes |
| Corrected 16K calibration sampled physical peak | 99,188,310,016 bytes |
| Corrected calibration sampled process peak | 42,363,028,160 bytes |
| Corrected calibration prefill | 134.350 s |
| Corrected calibration decode | 3.047 TPS over 32 decode steps |

The row probes use different seeded rows, so their timings are not a controlled
speed comparison. The 250 ms OS sampler observed 564 samples during the full
calibration. Swapout counts did not increase. Sampling does not establish an
instantaneous peak bound.

The calibration used the historical window-54 16K prompt and
`cell16k_ring_v2_attn_pf0`, I/O fanout 4, a 100 GB allocation target, a 16 GiB
transient band, and a 2 GiB freed-allocator cache. Its 33 generated token IDs
exactly match the first 33 IDs from window 54. Its smaller expert cache makes
the 3.047 TPS result unsuitable as a regression comparison against that run's
larger, incompletely accounted memory plan.

Before these cache fixes, the guard aborted the same calibration when sampled
physical use reached about 110.8 GB despite a roughly 25.4 GB process footprint.
The guard restored the previous Qwen service and released its lock. No completed
generation receipt exists for that aborted attempt.

`load-phases.jsonl` records another remaining source of duplication: path-based
resident loading grew file-backed pages by about 9.737 GB. The load-only physical
sampled peak was 77,415,890,944 bytes (77,388,152,832 at the load-end boundary).
These files describe the earlier loader.

The `resident-uncached-load-*` files measure the corrected resident loader with
the same reduced allocation settings. Its sampled physical peak is
**68,360,978,432 bytes**. Used file-cache growth through load is 113,655,808 bytes,
versus 9,600,466,944 in the earlier load. File-backed pages themselves still grow:
the new pages are predominantly speculative, which the OS counts as free. Do
not use `file_backed_bytes` alone as physical used memory. The process sampled
peak is 21,129,175,200 bytes; the Engram sidecar is now materialized during load,
instead of waiting for the first forward. Swapouts did not increase. The exact
previous service returned healthy with background warmup done and the lock
released. `resident-load-comparison.json` contains the calculations.

The subsequent `calibration-resident-uncached-*` run uses committed source
`607995471` with the same 16K input, 32 decode steps, 520 persistent slots,
100 GB target, 16 GiB transient band and 2 GiB allocator cache. It produces the
identical token digest and reaches **89,416,974,336 bytes** of sampled physical
usage, down 9,771,335,680 bytes from the earlier full calibration. Its 134.590 s
prefill and 3.066 TPS decode are single-run measurements, not a speed claim.
The typed Engram report and corrected shared transient-pool total are present
in this new raw receipt. All 18 focused resident I/O tests passed under the same
guard before the benchmark process started. Service restoration and unchanged
swapouts were verified afterward.

The larger-cache Python measurement starts from that bound, adds about 10 GB
to the allocation target, and retains the conservative 16 GiB transient band
and 2 GiB allocator cache. At the calibration baseline, the 110 GB split is
46.748 GB baseline + 2.147 GB Python + 61.105 GB Metal. Of Metal, 17.180 GB remains
outside the 43.925 GB engine plan for transients; the freed allocator cache is
inside this allocation. The real guard remeasures the baseline and derives the
actual plan again before load. `coding-budget-preflight.json` is the prospective
calculation, not a usage receipt or a pinned plan.

## Receipt provenance and corrections

`calibration-fixed.jsonl` and its OS trace and guard log are unchanged raw
observations from base commit `a0d36575b` plus the three cache changes listed in
`calibration-summary.json`. Two reporting fixes landed after that process had
imported the runner:

- The transient pool is global: 48 slots × 18,800,640 bytes = **902,430,720
  bytes**. The raw receipt incorrectly multiplies it by 40 layers.
- The new typed `resident_load_report` field is absent. The guard log's
  `STORAGE_POLICY` line records expert and both Engram readers as `f-nocache`.

`calibration-summary.json` preserves these errata explicitly. Do not silently
rewrite the raw receipt or use its transient total for memory accounting.

## Python workload for the performance target

The historical prompt ends with a markdown-report instruction that says
"No code." The new pinned fixtures retain the repository's naturalistic Python
patch request, including its helper and pytest requirements, and omit that
contradictory suffix. They use the artifact's tokenizer and exact chat template
with thinking disabled. The first complete measurement on the 16K fixture is
recorded below.

| Input tokens | SHA-256 of JSON token IDs |
| --- | --- |
| 1,024 | `f915ac7f2373ef94fbc86d944da740e0772453f39d6efba0152bd47dfa43905d` |
| 16,384 | `38894d01011a0146f8621dfdaf4bc4e618092d772d8cc28a70e21163a16799a2` |

`python-prompt-ids.json` contains both inputs, the instruction's 1,024-token
output cap, tokenizer/template hashes, and context provenance. The plain and
rendered text files make the task reviewable. From the repository root,
`PYTHONPATH=. .venv/bin/python
docs/deepseek-v41/receipts/memory-budget-110/build_coding_prompts.py` regenerates
them using the already installed local artifact (use the shared repository venv
when running from a worktree). This generator does not import MLX.

Full-model measurements still require the exclusive GPU guard, a bounded load
and compile peak, service restoration, and an explicit prompt IDs file. Report
actual tokens through EOS and distinguish decode steps from the first token
produced by prefill.

## Full Python workload and revised default reserve

`python-16k-1024-band16-*` records source `607995471` on the 16,384-token
Python fixture, with 1,023 timed decode steps plus the first token from prefill.
The 110 GB target, 16 GiB transient reserve, and 2 GiB allocator cache admitted
27 expert slots per layer. Decode was **3.778795 TPS** over 270.721 s; prefill
was 134.578 s. Sampled physical usage peaked at **100,142,891,008 bytes**.
The summary contains the exact sampled peak and counter deltas. Swapouts did
not increase, and the exact service returned healthy with warmup done.
The output is a Python unified diff, truncated at the 1,024-token cap; its code
was not applied or tested. This is throughput evidence, not code-quality proof.

These receipts exposed an unsafe default projection: the previous 5.54 GiB
transient reserve and 6 GiB allocator cache would admit enough extra expert
slots to require about 115.21 GB when the measured active peak and full cache
capacities are included. The revised defaults reserve 10 GiB for transients
and bound the allocator cache to 2 GiB. At the same baseline, they admit 35
slots per layer and project a 107.91 GB envelope. CPU regressions cover both
benchmark and serving session-bank reservations. The guarded validation below covers the measured 16K AR geometry; it does
not establish safety for arbitrary contexts or MTP.

`python-16k-1024-defaults-*` validates source `1fbe425ca` with the same exact
input and 1,024-token output cap. The guard measured a 47.1807 GB baseline;
actual allocation was 35 slots per layer (1,400 total), 2 GiB Python capacity,
2 GiB retained Metal cache and a 10 GiB transient band. Decode was **4.201765 TPS**
over 243.469 s; prefill was 135.005 s. All 1,024 output token IDs are identical
to the 27-slot control. This single paired observation is not repeatability
evidence or fulfillment of the 20 TPS target.

The true MLX active peak was **56,845,381,092 bytes**. The 250 ms external sampler
observed **106,349,838,336 bytes** of physical usage and **58,733,392,096 bytes**
of process footprint. The runner's 1 s sampler observed a lower physical peak
(104.55 GB), demonstrating why every report labels its sampling method. Both
raw traces are retained; do not replace one observation with the other or claim
an instantaneous OS peak. Swapouts remained unchanged. The guard ran 332 tests
plus seven subtests before the model and restored the exact service with
background warmup complete afterward.

## Corrected runner and cache diagnostic

`python-16k-1024-oracle.*` uses source `f743d1bdc` and the same exact Python
workload. It exercises the corrected static KV reservation by admitting 17,407
live tokens around generation and verifies release afterward. All 1,024 output
IDs match both earlier Python runs. The 250 ms trace records 1,553 samples,
**106,163,322,880 bytes** peak physical use, **58,709,717,072 bytes** peak process
footprint, and **56,845,378,900 bytes** MLX active peak. Swapouts are unchanged.
The exact service returned healthy, warmup completed, and the lock was released.

This pass enabled the passive route observer. Its launch also mistakenly set
the unused `MTPLX_EXPERT_IO_FANOUT=4`; the actual
`MTPLX_DSV41_IO_READ_FANOUT` was unset, so the reader used fanout 1. Its 3.783595
TPS is diagnostic timing, not a regression comparison with the clean fanout-4
4.201765 TPS result. Raw receipts remain unchanged. New runner reports include
the installed reader's `io_read_fanout`, independently of config or environment.

`replay_route_policies.py` reconstructs the captured warm cache and reproduces
all **88,637** observed misses exactly. It must probe the resident-only path
before the regular planner: their hit-touch ordering affects later recency.
Using only the regular planner differed by four misses. Twelve synthetic
warm-state cases also matched complete plans and final slot identities. Actual
MLX imports are blocked in this offline replay.

| Offline result at 35 slots/layer | Misses/token | SSD GB/s needed at 20 TPS |
| --- | ---: | ---: |
| Installed single-pool policy | 86.644 | 32.579 |
| Best screened existing frequency policy, decay 0.97 | 84.282 | 31.691 |
| Clairvoyant per-layer policy with bypass | 51.740 | 19.455 |
| Clairvoyant global 1,400-slot policy with bypass | 49.091 | 18.459 |

These are cache screens with the same captured initial contents, not speed
measurements or deployable future-aware policies. The frequency candidate
disables the single-pool policy only for replay and does not reproduce a new
whole-run prefill. It has not been promoted. The clean run observed 12.657 GB/s
over its I/O windows; that is a measurement, not a hardware bandwidth ceiling.
At that 35-slot capacity, the evidence does not support reaching 20 TPS through
eviction tuning alone. The larger capacities below change the cache bounds and
need their own analysis.
MTP/verify scheduling needs separate arithmetic, peak-memory and performance
gates before any full-model run.

Runner verification logs include the initial test-device contamination and
the final **313-test guarded pass**. A final three-case installed-fanout fix
passed with the 35 reporting/admission cases and three offline cache cases
(38 CPU tests total), with real MLX imports blocked. No test tolerance changed.

## Automatic reclamation and larger resident caches

Source `995522859` integrates clean file-cache reclamation into normal guarded
Qwen shutdown, using its reported model directory. Both following runs use the
same exact 16,384-input Python prompt, 1,023 decode steps and 1,024 output tokens,
pf0, fanout 4, 110 decimal GB, 2 GiB Python capacity and 2 GiB allocator cache.

| Receipt | Baseline GB | Transient GiB | Slots/layer | Decode TPS | Physical sampled peak GB |
|---|---:|---:|---:|---:|---:|
| `python-16k-1024-defaults` | 47.1807 | 10 | 35 | 4.201765 | 106.349838 |
| `python-16k-1024-reclaimed` | 12.9284 | 16 | 72 | 5.834646 | 100.216291 |
| `python-16k-1024-reclaimed-defaults-v2` | 10.2463 | 10 | 84 | **6.281322** | **106.591666** |

The normal default reserve therefore runs **49.49% faster** than the earlier
35-slot allocation on this workload. All 1,024 output IDs match, with digest
`2bd0ad017b9580c8fec340e297696a0bd81a7759b6c5dfe7c5d64de6d40c1090`.
These are measured allocation outcomes with changing live baselines, not a
controlled attribution of every improvement to the transient-band setting.

The 84-slot pass took 134.470 s prefill and 162.864 s decode. MLX active peak was
93,693,612,196 B; the external 250 ms sampler recorded 1,153 samples and a
95,845,554,032 B process-footprint peak. The higher external physical peak is
retained alongside the runner's 1 s observation. Swapouts did not increase.
The guard exited 0, restored the exact healthy Qwen service with warmup complete,
and released the lock; health and lock were independently checked afterward.
The vanished tool session handle after continuation was not an OOM or crash.

At 84 slots, decode streamed 44,848 records, or 43.840 misses/token, down from
86.644 at 35 slots. I/O-window throughput was 12.201 GB/s, with 67.554 ms of read
windows per token. That is not an additive critical-path breakdown. **20 TPS
remains unmet**; cache allocation alone has not established the target.

The first default-reserve wrapper rejected an 11.8336 GB baseline before model
load because its diagnostic interval had an unnecessary 12 GB lower bound.
The retained failed guard log shows exit 1 and successful restoration. The v2
wrapper accepts lower baselines while retaining slot, physical-peak and wired
bounds. No production budget guard was weakened. The 72- and 84-slot active peaks
were about 1 MB below the exact persistent-storage-delta projection from 35 slots,
supporting the bounded capacity adjustment.

## Short profile and decode protection transition

`python-16k-128-timeline-largecache` uses source `0e57cc0ff`, the same 16K
Python input and 128 decode steps, with 82 slots/layer from the live baseline.
The profile and passive route capture are enabled, so its 5.627879 TPS is a
short diagnostic, not a replacement for the full-length performance receipt.
All 129 output IDs and every captured route match the prior control's prefix.
Its 250 ms physical peak was 105,513,058,304 B, with no added swapouts; the guard
exited 0 and restored healthy Qwen/warmup and the free lock.

The existing timeline recorded 128 tokens with estimated overhead 0.057 ms/token.
Mean exposed SSD wait was 75.060 ms/token, routing barriers 46.162 ms/token,
and gather fences 36.153 ms/token. The latter two include GPU work and sync;
they are not isolated GPU execution times. Post-barrier host time excluding
fences was 83.849 ms/token, including that SSD wait. These are short-prefix
measurements; they do not establish an immutable model compute ceiling.

Warm-state replay reproduces all 7,012 observed misses. The seed protects the
entire resident pool during prefill, and previously decode inherited that set
without enforcing its intended 80% protected-segment cap. Trimming only policy
membership at the phase boundary reduced this prefix replay to 6,281 misses
(10.43% fewer), without changing residency, pins or recency stamps. Alternative
segment fractions were less effective. A hypothetical continuation using the
previous complete route trace projects 45,883 to 42,926 misses; routes after
the captured prefix were not observed in this larger-cache run, so this is a
screening estimate, not a measured full-run gain.

The implementation now performs that trim on the first successful decode route
in both the normal and all-hit planners. Subsequent prefill chunks remain fully
protected, and a new request can seed its whole pool again. The existing
transaction snapshots restore the phase and protection on rollback. Five focused
CPU regression cases reproduced the missing transition before the fix; those
and 37 existing policy cases pass with MLX imports blocked. Full-run performance
validation of the transition is pending.
