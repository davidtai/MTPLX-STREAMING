# Native output projection with one retained weight representation

The native target output route retains packed MXFP8 `wo_a` weights alongside
the BF16 transpose consumed by fused decode. Across 40 layers, the packed
weights and scales occupy 1,384,120,320 bytes. The staged candidate preserves
the original first-use cache materialization, then retires the packed holder
and cache-key tuple and binds the BF16 output callable directly. A weak parent
reference avoids adding an ownership cycle. No steady eligibility branch,
fallback, counter or changed matrix arithmetic is introduced.

The output function's AST matches native after binding the existing weight,
validated group count and equivalent module alias. The real layer-0 operator
loads four native tensors totaling 77,856,768 bytes from shard 3. Query/KV and
other unused attention parameters are neither loaded nor evaluated. Native
RoPE removal, grouped BF16 matmul and native MXFP8 `wo_b` remain unchanged.

| Measurement | Native warm | Candidate warm |
| --- | ---: | ---: |
| MLX active bytes | 145,952,648 | 111,349,640 |
| Peak bytes, including first materialization | 213,061,512 | 213,061,512 |

The exact active reduction is **34,603,008 bytes**, the original packed matrix
and scales. All output bytes agree at M1, M6 and M8, including a cold first use
of the candidate. Eight active bytes remain after cleanup. This establishes
one-layer ownership, not full-model throughput or cache capacity.

Cold peak remains unchanged because materialization initially needs the packed
source. Relative to the retained warm state, the candidate's cold overlap is
101,711,872 bytes versus 67,108,864 bytes for native. A full steady bound must
therefore retain at least one layer's 34,603,008-byte source overlap while
crediting the retired packed weights. Expert-bank growth happens earlier and
receives no projection-retirement credit.

The 4 GiB incremental bound includes 2 GiB for MLX policy plus 2 GiB for Python,
cache/compiler and temporary owners. The source reader uses final MLX buffers,
bounded uncached ranges, validated native shard/tensor metadata and source
identity checks. After the child exits, the controller reclaims 66,387,968
cached source bytes to zero. The guard samples 403,309,816 bytes of process-tree
footprint and 10,487,103,488 bytes of whole-machine use. These are separate
overlapping observations, not a sum.

The first attempt fails during input setup because native `_cos_sin` requires
flat positions. The corrected harness supplies `[S]` positions and native
`[S,32]` cosine/sine tables. No projection comparison ran in the refused
attempt. Its guard exits 1 and restores/releases at 03:51:16 UTC on September
18; independent 03:54:19 verification finds healthy/idle/warmed/free Qwen.

The completed operator is pinned to source `72b0d219e`. Guard session 10513
exits 0, restores exact Qwen identity, health and warmup, and releases at
03:55:19 UTC. Independent verification at 03:59:04 UTC finds healthy, idle,
warmed Qwen, a free lock and no owned child. Both attempts have SHA-256 archives.

The full candidate retains the original prefill and bank-growth bounds and
separately pins the unretired native MTP seed peak. Only steady projection
storage receives the ownership credit, retaining one cold packed source and
16 MiB of additional host metadata. Installation follows prefill and bank
growth, so the first-use route cannot engage during prompt processing.

## Full 16K/1K result

Source `2488ebf648` completes 1,024 identical native output IDs in 206 cycles,
at **12.4516125 TPS / 82.1580336 seconds**, including 3.4252662 seconds of
charged cache growth. The live 10,881,843,200-byte baseline admits 84 prefill
slots and 101 decode slots, with a 109,600,670,040-byte physical bound.
The best 12.6731624-TPS control used 102 decode slots at a lower baseline.
This run establishes a memory improvement, not a throughput win.

All 40 target layers retire their packed owners and retain the original BF16
matrices. Final MLX active memory is 91,799,553,460 bytes. Adjusting the control
only for the exact 707,788,800-byte slot-band difference gives a reduction of
**1,384,120,320 bytes**, exactly the packed source inventory. Peak MLX memory
is 92,545,655,960 bytes; the corresponding capacity-normalized reduction is
790,843,596 bytes because full MTP seeding now determines the peak. Prefill
active and peak bytes are unchanged. No additional prefill or seed credit was
used to admit this run.

The internal process-footprint peak is 93,711,383,384 bytes and the internal
whole-machine peak is 105,971,466,240 bytes. The independent guard observes
93,742,431,064 bytes for the process tree and 105,996,976,128 bytes for the
whole machine, with 225 samples and no compressor growth. These overlapping
measurements are reported separately. Source and packed artifact cache cleanup
ends at zero cached pages.

Guard session 38104 exits 0, verifies exact Qwen identity, health and background
warmup, and releases the lock at 04:17:52 UTC on September 18. Independent
04:18:49 verification finds healthy/idle/warmed Qwen, a free lock and no owned
child. The `full/` archive includes the exact helpers, source and budget
fingerprints, raw receipts, OS samples, cleanup and lifecycle records, and a
SHA-256 inventory. No broad test suite or general-serving default is added.

Next, compose the two measured memory improvements with a genuinely smaller
bounded Python row cache, preserving all native decode arithmetic and the
110,000,000,000-byte ceiling. The best full result remains 12.6731624 TPS;
20 TPS remains unmet.
