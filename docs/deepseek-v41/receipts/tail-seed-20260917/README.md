# Bounded DSpark prefill seeding: staged

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
