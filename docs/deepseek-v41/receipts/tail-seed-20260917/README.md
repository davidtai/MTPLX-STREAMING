# Bounded DSpark prefill seeding: exact 2,048-row memory result

The native DSpark seed projects all prompt hiddens, computes three pointwise
KV projections/norms/RoPE, and retains only the last 128 rows in each draft
cache. The staged operator compares full 16K seeding with those final rows at
the correct absolute offsets. It loads only the 12 native seed tensors
(89,224,192 bytes) and tiles authentic captured target hiddens into a synthetic
16K input. This is not the real prompt seed or target generation.

The first attempt stopped at the standard resident loader's existing 64 MiB
aggregate unselected-tensor limit. Selecting only these tensors would discard
too much of their three shards. No seed comparison ran; the limit is unchanged.
The controller observed child exit 1 and zero residual source-file cache. The
original guard restored exact Qwen identity, health and warmup and released
the lock at 02:34:21 UTC on September 18. Independent verification at
02:34:38 UTC found healthy/idle/warmed Qwen, a free lock and no owned child.
The complete refusal is preserved in `loader-refusal/`.

Variant 2 at `/tmp/dsv41-tail-seed-20260917/v2` reads only the admitted tensor
ranges into final MLX buffers. It validates all three native shard headers and
selected tensor metadata before allocation, uses uncached positional reads
with at most an 8 MiB view, and verifies source identity after loading. It
retains the 8 GiB bound and process-exit cleanup controller. No production
loader, target arithmetic or memory ceiling is changed.

The 128-row variant completes two interleaved comparisons. Native full seeding
reproduces its cache hashes exactly. Its maximum allocator peak is
2,279,214,764 bytes versus 142,877,484 bytes for the retained-row variant,
saving 2,136,337,280 bytes in this operator. Median seed time falls from
0.0711085 to 0.0017979 seconds. These exclude input construction and do not
measure full-model decode speed.

The final main-hidden row and all absolute offsets match, but the three cache
windows do not match native bytes. Changing the quantized matrix batch size
therefore needs further numeric investigation. This is not an exact native
replacement and is not installed. The next bounded candidate retains 2,048
rows; it is a proposed batch geometry, not a proven dispatch threshold.

Variant 2 leaves eight active allocator bytes. Its controller reclaims
74,366,976 clean source-file bytes to zero after child exit. Guard 38266 exits
0, restores exact Qwen identity/health/warmup and releases at 02:37:42 UTC.
Another guarded benchmark then legitimately owns the lock and stops Qwen.
Work stays on CPU during that window. Independent verification after its
completion, at 02:42:32 UTC, finds Qwen healthy, idle and warmed and the lock
free. `tail128/sha256.json` binds the completed memory/numeric evidence.

The 2,048-row candidate matches the final main-hidden row, all three window
byte sequences and all absolute offsets in both interleaved comparisons.
Its maximum allocator peak is 416,511,276 bytes versus 2,279,214,764 bytes for
the full seed: a saving of 1,862,703,488 bytes. Median seed time is 0.0086178
seconds versus 0.0715489 seconds. This proves the bounded seed operator only;
full prefill capture, cache capacity and decode TPS are not yet measured.

The controller leaves eight active allocator bytes, reclaims 74,366,976 clean
source-file bytes to zero, and exits 0. Guard 36042 restores exact Qwen
identity, health and warmup and releases the lock at 02:44:16 UTC. Independent
verification at 02:50:06 UTC finds Qwen healthy, idle and warmed, a free lock,
and no owned child. `tail2048/sha256.json` preserves the complete result.

The isolated integration at `/tmp/dsv41-tail-seed-20260917/integration` narrows
only the native layer-major DSpark prefill captures and seeds native draft
caches at absolute position 14,336. The target attention, MoE, HC, cache and
verification code stays inherited. A construction-bound subclass avoids a
stored bound-method ownership cycle. It is restricted to one 16K native-KV
request. Q8 partial seeding needs its own offset handling. Initial full-run
admission must keep the old allowances plus metadata; the operator saving
alone is not evidence for more expert slots.

The small integration check completes all four cases: FP32 and BF16 backbones,
33 prompt rows in chunks of seven, and retained tails of seven and 17 rows.
Logits, retained hiddens, cache bytes and offsets, and the next decode step all
match exactly. Allocator peak is 1,591,702 bytes, with 28 active bytes after
cleanup. The first attempt failed only because NumPy cannot directly convert
BF16 cache arrays; the completed check compares raw byte views instead.
Both attempts are preserved in `capture-check-refusal/` and `capture-check/`.

Guard 32403 exits 0, restores exact Qwen identity, health and warmup, and
releases at 02:59:14 UTC. Independent verification at 03:00:51 UTC finds
healthy/idle/warmed Qwen, a free lock and no owned child. The next stage is one
complete native D5/M6 16K/1K run with the best packed-plane lane. Its admission
retains all original allowances and adds 16 MiB of host metadata reserve in
every phase; no operator-derived capacity discount is applied.

The first full attempt refuses before model allocation: the CLI still reserves
2 GiB for the host while admission reserves 2 GiB plus 16 MiB. The derived
engine is consequently 16 MiB too large. The retry passes `--host-overhead-gib
2.015625` and checks that its byte value equals admission. A CPU check, with
MLX imports blocked, verifies the actual runtime budget resolver and admission
produce identical engine and allocator limits at both saved baselines.
The refused guard exits 4, restores Qwen and releases at 03:09:30 UTC;
independent verification at 03:10:00 UTC confirms healthy/idle/warmed/free.

The completed full candidate at source `dcbec19dc` generates all 1,024 native
output IDs exactly in 206 cycles. It measures **12.4289961 TPS / 82.3075322 s**,
including 3.4589748 seconds of phase growth. The live baseline is
10,603,659,264 bytes; admission allows 84 prefill and 101 decode slots at a
109,849,188,584-byte bound. Allocator peak is 93,336,409,632 bytes, process
footprint peak is 95,127,698,216 bytes, and internal sampled whole-machine
peak is 106,151,673,856 bytes. These are overlapping measurements, not sums.

Compared with the best packed-plane run, prefill peak decreases by
591,643,308 bytes and boundary active allocation by 1,536,180,224 bytes.
The best run has one more expert slot per layer, totaling 707,788,800 bytes.
After that exact storage normalization, end-of-decode active allocation is
identical and the overall allocator peak differs by only 89,924 bytes.
Decode still determines peak capacity. This is a prefill memory improvement,
not a measured throughput win or evidence for relaxing steady admission.
The best full result remains 12.6731624 TPS; 20 TPS remains unmet.

The candidate reads 35,494 weight records / 628,056,391,680 bytes, plus the
existing packed-scale installation. The guard completes 224 samples, reports
106,145,562,624 bytes of sampled physical use, and observes no compressor
growth. Source and packed-file cleanup finds zero retained cached pages.
Guard 17051 exits 0, restores exact Qwen identity, health and warmup, and
releases at 03:16:02 UTC. Independent verification at 03:16:54 UTC finds
healthy/idle/warmed Qwen, a free lock and no owned child. `full-host-refusal/`,
`full-tail2048/`, and `full-summary.json` preserve both attempts and comparison.
The helper stays isolated; no generic prefill or Q8 seeding API is changed.
# Native full-run byte-field correction, September18

The composed full run in `../memory-compose-20260918/` measures its retained
`[1,2048,15360]` main hidden tensor as FP32 and125,829,120 bytes. The earlier
tail helper's62,914,560-byte report field assumes BF16, as used in its synthetic
operator. It is not the real full-run tensor size. The composed wrapper now
reports actual dtype and nbytes at the existing seed boundary. Earlier full
admission retained the original memory bounds and did not use this nominal
field for credit. Historical receipts and hashes remain unchanged.
