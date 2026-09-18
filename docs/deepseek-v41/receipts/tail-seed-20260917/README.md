# Bounded DSpark prefill seeding: 128-row memory result

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
