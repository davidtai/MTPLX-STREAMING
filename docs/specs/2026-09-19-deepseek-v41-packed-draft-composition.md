# Compose packed draft output with predictable target projection scheduling

Continue Q4 only on the exact 16,384-input / 1,024-output Python workload.
The best complete result remains 13.8688167379 TPS. This composition requires
a new full output and timing receipt; component wall time cannot establish
20 TPS or a full-model speedup.

## Construction

Keep the native 93/58/32 draft expert banks, native KV16 and D5 plus at most
two causal lookup tokens. Install the measured packed FP32 draft output route
once at model construction. Only `DSparkAttention` output preparation changes;
native seed returns before that branch. Its 2,048-row tail cannot reach the
new packed output operation, whose measured maximum draft width is six.

Retain all 40 native packed target output projections. Expand their exact
BF16 transposes in known layer order, keeping the existing two-buffer schedule.
Prime the first output after native MTP seeding while only the original 84
expert rows exist. Then allocate the independent extension bank while retaining
that single initialized projection. No original expert weight array is copied.
All priming and extension costs remain inside decode wall time.

The smaller 80/40/24 alias table is excluded: its combined component adds a
target call. Direct expert input addressing is also excluded: its nominal gain
is below control variation. Neither experiment justifies a full run.

## Memory accounting

The whole-machine ceiling stays 110,000,000,000 decimal bytes; the wired ceiling
stays 100 GiB. The existing strict allocator identity is pinned. No MLX import,
load or compilation occurs outside the canonical parent-held GPU window.

The new draft route avoids 402,653,184 bytes of native FP32 dense cache in
steady decode. This credit is zero during prefill, seed and extension: those
phases did not own the caches. Add 128 MiB for new GPU temporaries and 16 MiB
for host state. The packed Metal gather route allocates contiguity copies and
outputs, without full dense dequantized weights; the pinned backend source
and a three-stage temporary inventory accompany the installation.

Cold target priming still prices three 67,108,864-byte arrays, now at 84 rows.
Extension retains one initialized array. Steady decode continues to price all
three arrays for replacement. This separates allocation phases without
reducing the cold-compilation allowance.

Reserve another 256 MiB for observed background variation, inside the 110 GB
ceiling. Process host reserve is 1,455,550,464 bytes; host plus background is
1,723,985,920 bytes. Derive all phases from fresh physical and wired snapshots.
Require at least 111 expert rows before full model loading so a capacity drop
does not obscure the comparison. CPU cases admit 111 rows at the former
10,983,129,088-byte baseline with a 109,896,193,272-byte estimate, but only 110
at 11,560,878,080 bytes. A former baseline is not current admission evidence.

## Evidence gate

One guarded full run may proceed after static source, seed-route, ownership,
phase-budget and CLI-budget checks. Preserve exact output IDs, the existing
AR tie classification, readable Python output, decode wall time, separate
allocator/process/machine peaks, reclamation, service identity, warmup and lock
release. Refusal or inconclusive timing does not trigger an unchanged retry.
Add regression tests only after a measured optimization win.

Scratch staging is `/tmp/dsv41-packed-draft-full-20260919`; its source revision
is `b5f41c51d74a770069048dddd0a644f07886a999`. No production default changes.
