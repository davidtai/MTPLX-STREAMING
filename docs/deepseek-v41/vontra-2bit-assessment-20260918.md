# Vontra 2-bit checkpoint assessment

Reviewed 2026-09-18 in response to the user's link, at Hugging Face revision
`802f1a00982705d81b79ad1c83aa0ccc0b863ebc`. This is a read-only assessment of
published metadata and documentation. No model weights were downloaded or
executed, and the current target artifact remains unchanged.

The [model card](https://huggingface.co/Vontra/DeepSeek-V4.1-Flash-MLX-2bit-MTP/blob/802f1a00982705d81b79ad1c83aa0ccc0b863ebc/README.md)
describes ordinary affine 2-bit/group64 requantization from the official
mixed-precision checkpoint, retaining all three native DSpark stages. It
reports 9.46 TPS without MTP on a tiny arithmetic prompt and slower serial
MTP. The resident backbone alone is 160.88 GiB, beyond our 110 decimal GB
whole-machine ceiling. Reported process RSS excludes filesystem cache and
other machine use. Broad quality and long-context validation are absent.

The [runtime guide](https://huggingface.co/Vontra/DeepSeek-V4.1-Flash-MLX-2bit-MTP/blob/802f1a00982705d81b79ad1c83aa0ccc0b863ebc/runtime/README.md)
caps prompt plus output at 128 tokens and output at 64. Its fully resident
headline result does not apply to its slow streaming mode. Consequently the
supplied runner is not directly suitable for our 16K/1K or eventual 256K task.
Its measurements are not comparable with the retained 13.4141518 TPS result.

The useful research direction is reduced expert/draft storage. Switching
the target to 2-bit would change target arithmetic and potentially quality,
far beyond the user's allowance for tie breakers. A separate 2-bit draft
with the original target verifier could preserve target verification while
saving memory; this is an inference, not a tested optimization. Draft
acceptance, block verification, kernel cost and complete memory accounting
would determine whether it helps. No artifact or runtime substitution is
authorized merely by this assessment request.
