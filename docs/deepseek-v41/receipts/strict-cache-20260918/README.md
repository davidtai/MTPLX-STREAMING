# Strict allocator cache, 2026-09-18

One complete 16,384-input / 1,024-output Python run reached **13.1509467 TPS**
in **77.7890768 s**, with all 1,024 native output IDs identical. The sampled
whole-machine peak was **109.671973 GB**, below the **110 GB** ceiling.
This is the best single full result so far; **20 TPS remains unmet**.

Measured repository source: `5ba18552cb81f85793469a8d20fe2ff4af4f79d8`.
Native KV16, D5/M6, 206 cycles, part3 reads, 48 shared transients, and charged
post-prefill growth remain. Capacity grows from 84 to 109 slots per layer.

## Change and evidence

MLX 0.32.2's stock free-buffer cache can exceed its configured limit when a
freed buffer is added. Lowering the limit also leaves existing cached buffers.
The isolated allocator patch evicts old inactive buffers before inserting a
fitting buffer, releases oversized buffers, and trims immediately when the
limit is lowered. Both operations retain the existing allocator mutex.

This establishes an inactive-cache bound and removes the separately priced
2,258,155,644-byte overshoot allowance. It does not reduce any active-array,
copy, seed, prefill, KV, compiler, Python or wired-memory reserve. The cache
limit stays 1 GiB through prefill and becomes 256 MiB before growth/decode.
Two fixed 64 MiB Engram arenas have zero evictions in the completed run.

At the actual 10,240,868,352-byte background baseline, CPU-only accounting
admits 105 slots under the preceding cache policy and 109 under the strict
policy. Their bounds are 109,370,132,712 and 109,979,515,100 bytes. The capacity
search extends to 110 with unchanged page-aligned component geometry; the
original phase and wired-memory inequalities still select the actual size.
See [same-baseline admission](full-v1/same-baseline-admission.json).

The official v0.32.2 source archive is pinned by SHA-256. All 237 installed
headers match it. Matched stock and strict host libraries were built with
identical Metal shader bytes; only the allocator object differs between the
two builds. The installed Python extension and production packages were not
replaced. `DYLD_LIBRARY_PATH` is set only inside guarded child execution.
Every measured child attests the actual loaded library path and SHA-256.

The bounded attention comparison matches all five layers' output, cache state,
selected indices and metadata across the wheel and both builds. Stable
Full/Reuse and Full/Reindex timings show no material regression; SWA controls
are noisy. The actual layer34 expert replay matches all 206 physical reads and
outputs, with strict/stock elapsed ratio 0.9960 and 0.324% stock control spread.
These are operator checks, not full-model speed claims.

## Full result

| Measurement | Retained part3 result | Strict allocator |
|---|---:|---:|
| Decode TPS | 12.8091055 | 13.1509467 |
| Decode wall seconds | 79.8650617 | 77.7890768 |
| Decode slots/layer | 104 | 109 |
| Expert records read | 34,259 | 32,316 |
| Expert read bytes | 606,203,412,480 | 571,822,571,520 |
| Read union seconds | 46.9966761 | 45.1256016 |

The observed gain is 2.669%, saving 2.076 seconds and 34.381 GB of expert reads.
This comparison includes different live baselines and maximum-admitted
capacities. It does not isolate allocator latency or establish repeatability.
Read union time is not a GPU-idle measurement. No unchanged full rerun was made.

The active allocation bound is 98,098,546,908 bytes. Measured MLX peak is
97,614,305,072 bytes; process footprint peak is 98,610,301,552 bytes; internal
machine peak is 109,671,972,864 bytes. The guard independently samples a
109,664,894,976-byte machine peak, with 219 samples and zero compressor growth.
Final active MLX memory is 97,461,880,244 bytes and inactive cache is 256,770,058
bytes. The configured 268,435,456-byte cache limit is reported separately.
See [full summary](full-v1/full-summary.json) and its raw sidecars.

Output SHA-256 is
`0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac`.
The existing native-AR difference at 297 remains classified as the same allowed
tie flip. Reused AR timing/memory fields are not current-run measurements.

## Focused regression and lifecycle

Only after the full improvement, three cache regressions were added: oversized
free, accumulated fitting frees, and immediate trimming on a lower limit.
All three fail on matched stock and pass on strict, taking 0.014/0.013 seconds.
No broad suite was run. Test patch and results are included.

Every GPU operation and compilation used the parent-held exclusive guard.
Full guard42497 exits0; exact Qwen identity, health and warmup precede lock
release at07:20:06UTC. The independent full-run check passes at07:20:32UTC.
The final regression guard15580 exits0 and restores/releases at07:24:09UTC;
the independent07:24:19UTC check confirms healthy, idle, warmed Qwen and a free
lock. No owned child remains. Other jobs were preserved. Source/packed file
cache reclamation reports zero remaining cached pages after the full run.

## Artifacts and remaining work

`build/` contains the allocator, build-only shader reuse and test patches,
source provenance, matched build commands, hashes and logs. The upstream MIT
license is included. Large binaries and model payloads remain outside Git.
`attention-screen/`, `expert-screen/`, `regression/` and `full-v1/` preserve
their helpers, construction proofs and receipts. `sha256.json` hashes the
archived evidence. Condensed health receipts retain relevant verified fields.

Live root: `/tmp/dsv41-strict-cache-20260918`. Full raw prefix:
`/tmp/dsv41-110-stage/full-strict-cache-20260918-v1`. Helpers reject stale source
pins and existing outputs; a future run needs a verified source-proof refresh
and a fresh output prefix. The strict binary remains isolated and is required
before taking its admission credit. Stock MLX must retain the old allowance.

Continue reducing expert traffic or exposed verification cost from this result.
Another 26.639 seconds must be removed from this workload to reach20TPS.
Fixed Q8 storage already supports a262144-token configuration, but a complete
256K prefill/rollover envelope remains unverified and secondary to throughput.
