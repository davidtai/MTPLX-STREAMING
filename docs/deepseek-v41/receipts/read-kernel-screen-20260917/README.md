# Read reduction and native down-projection screen

The retained full-model result remains **12.1146645 decode TPS**; the 20 TPS
target is unmet. No full model was loaded for this stage. The only runtime
change is the GPU guard's memory-summary correction described below.

## Cross-expert coding: rejected

A bounded CPU screen reads 15 manifest-verified records from layers 0, 10, 20,
30 and 39: reference expert 0, compared with experts 1 and 128. Total bytes read
are 282,009,600. Ideal average payload fractions of each native record are:

| Symbol model | Ideal fraction of raw record |
| --- | ---: |
| Independent bytes, separate component tables | 0.892151 |
| XOR bytes against reference expert | 0.947763 |
| Reference-conditioned symbols | 0.895487 |

The reference-conditioned weights use nibbles; scales use bytes. The raw
receipt's `conditional_nibble_bits_per_byte` key also holds the scale result.
These are empirical entropy bounds, excluding tables and reference storage;
they are not encoded sizes or decoder benchmarks. Neither tested cross-expert
model improves on independent component byte coding. No codec was installed.

## Sorting: ruled out for this bank geometry

The local `mlx-fork` checkout is based on 0.31.2 and is not the installed
runtime's source. The installed packages report MLX and mlx-metal 0.32.2.
[MLX v0.32.2's gather dispatcher](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/quantized.cpp#L1785)
requires `M == 1`, `B >= 16`, sorted RHS indices and integer `B/E >= 4` to
select its sorted weight-reuse route. Native D5/M6 has at most 36 assignment
rows and 98–100 persistent bank slots, or 48 transient bank slots. Sorting the
existing bank therefore cannot select that route. A different bank/view design
would be a separate experiment; no sorting benchmark was needed.

## Specialized down projection: limited operator win, not installed

The down matrix is native MXFP4, group size 32, `[5120, 2304]`. Specializing its
nine 256-element K steps avoids padded cache records. The prototype retains the
native eight values per lane, two four-value dot sums, four output rows per SIMD
group and SIMD reduction. Gate/up projections and the clamped SwiGLU are stock.
It changes no stored weights and adds no padded bank allocation.

The first probe had incorrect ushort pointer strides and failed byte parity.
It timed no candidate. The corrected version uses two ushorts per lane and 64
per K step; both fixed-loop and unrolled versions match all output bytes in the
four sampled down-only cases. Early timings vary enough that they are screening
evidence only.

The follow-up measures the complete three-projection MLP using 36 real layer-20
experts and deterministic synthetic BF16 activations. It warms stock first and
interleaves stock/fixed/unrolled controls. Each arm block has 15 timed calls.
The following are medians of the arm-block medians:

| Assignment rows / distinct experts | Stock MLP, ms | Unrolled down MLP, ms | Latency reduction |
| --- | ---: | ---: | ---: |
| 18 / 3 | 0.576958 | 0.532708 | 7.67% |
| 36 / 12 | 1.326208 | 1.313730 | 0.94% |
| 36 / 36 | 1.392416 | 1.380938 | 0.82% |

Every sampled output byte matches for both candidates. This does not establish
full-model parity or a decode gain. The small benefit on larger MLP cases does
not justify another full-model window at this stage. The prototype remains
outside production; expert-read waiting is still the measured primary cost.

The whole-MLP probe has a conservative 6 GiB incremental allowance, a 2 GiB MLX
policy limit and 256 MiB allocator-cache policy. Measured MLX peak is
678,724,030 bytes. Process footprint at the final boundary is 1,628,718,832
bytes; physical machine use is 10,035,527,680 before and 11,758,698,496 bytes
after. Boundary observations are not continuous peaks. These overlapping
memory measures must not be added together.

## Guard memory-summary fix

The short first probes illustrate polling's limits: their guard summaries
round sampled process footprints to `0.0 GiB`, while child boundary receipts
contain substantial allocations. A child that finishes before the first valid
poll also previously printed initialized zero peaks. The updated exit summary:

- Labels every peak as sampled and includes authoritative bytes beside GiB.
- Reports the complete child sample count and poll interval.
- Prints `n/a` when there is no complete child observation, even if the
  pre-step machine baseline was measured.

The footprint reader, cadence, admission arithmetic, 110,000,000,000-byte
ceiling, shutdown reclamation and restoration logic are unchanged. A focused
CPU regression distinguishes missing samples, a measured zero and nonzero
measurements: the previous source fails all three subcases, the fix passes.
The existing hermetic memory-accounting script passes 19 checks; its stale
93 GiB default expectation now matches the existing 100 GiB child policy.
It uses fake readers and a temporary lock, and never touches Metal or Qwen.

## Lifecycle and provenance

All four windows exit zero and restore exact Qwen identity, health and warmup
before lock release. The final GPU window releases at **18:31:39 UTC**; the
independent check at **18:37:02 UTC** confirms the correct model, zero active
requests, warmup done and a free lock. No unrelated process was signaled.

`commands.txt` gives exact commands. `summary.json` records upstream source
URLs and hashes, arithmetic summaries and scope. Raw JSON, scripts and gzipped
logs are retained with a manifest. The measured source is `8f6c84d394`; the
guard reporting fix follows these measurements. The prototype scripts retain
their original temporary paths and require their pinned upstream header for
hashing; the receipt records its public URL and exact hash.
