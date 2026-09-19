# Release draft projection expansions without additional SSD traffic

The user's observation is correct: dense layer order is known. The retained
Q4 target already exploits that order by expanding the next packed output
projection during current expert demand, retaining two BF16 buffers and pricing
three for replacement. Its conservative saving is 1,098,907,648 bytes. The
query-weight SSD prototype instead saved 1,643,642,880 resident bytes but took
50.0689% longer in its bounded comparison: extra dense reads exceeded avoided
expert reads. That result rejects that schedule, not deterministic scheduling.

The three native draft stages still retain 402,653,184 bytes of expanded FP32
output-projection weights. One draft-only experiment replaces this expansion
with native MXFP8 packed projection, preserving FP32 inverse-RoPE inputs and
outputs and the original following projection. All 93/58/32 compact Q4 experts
and their existing lookup tables remain. The target route is unchanged. Packed
accumulation may alter draft proposals, so no exact-output or full-TPS benefit
is assumed.

Validate geometry, native source identity, and the replacement boundary once
at construction. Bind the candidate directly, with no runtime eligibility or
fallback branch. Retire dense caches between arms and record allocator changes
outside timing; compiled references must not be assumed to release memory.

Use the existing authenticated 16K/1K teacher states for one native/candidate/
native head comparison. Both controls must reproduce the 198 pinned commitment
boundaries. Report candidate cycles, verification rows, head wall time, and
actual memory retirement separately. Teacher output scores proposals only;
this component neither executes the target nor establishes full parity or TPS.

Keep the previously bounded 49 GiB incremental envelope, 110,000,000,000-byte
whole-machine ceiling, and 100 GiB wired ceiling. Run only inside the canonical
parent-held GPU window, after fresh admission, and restore exact Qwen identity,
health, and completed warmup before releasing the lock. No broad test suite is
added; full composition requires a useful component result first.

Scratch: `/tmp/dsv41-draft-packed-projection-20260919`.
Source: `a15ba7c40b6e59c7e3c3fdae852112085bddf40e`.
