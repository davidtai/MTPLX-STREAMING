# Draft-conditioned suffix screen

The new proposal strategy keeps all five native draft tokens and predicts a
suffix in a second, read-only draft pass. Its short variant is a small
conditional candidate; its long variant is rejected. Neither is installed in
the target runner, and no new TPS or target parity result is claimed.

## Mechanism and causal boundary

The native draft embeds the current token followed by noise placeholders.
This experiment replaces the first five following placeholders with the
already-generated native proposals, runs the same three draft stages at width
7 or 13, and projects only the suffix rows. The Markov recurrence starts with
the last native proposal. All original five decisions stay intact. The second
pass reads the existing draft cache without committing speculative state.

Issuance requires native minimum sigmoid confidence of at least 0.9. That
threshold selects 44/44 correct prefixes in the first 103 historical native
boundaries and 40/42 in the remaining 103. It is chosen before the suffix
screen. Future target IDs only score completed proposals and seed states after
commit; they never select the suffix tokens. This is not tree verification.

## Opportunity screen

All 206 native boundaries/proposals reproduce the authenticated reference.
At the 86 eligible boundaries:

| Suffix | Added verification rows | Additional matched tokens |
| --- | ---: | ---: |
| Conditioned width 7, two-token suffix | 172 | 108 |
| Conditioned width 13, eight-token suffix | 688 | 124 |
| Unconditioned width 13, native prefix retained | 688 | 124 |
| Conditioned width 7, suffix confidence >=0.5 | 46 | 40 |

These are independent-boundary opportunities, not new trajectory cycle counts.
Long suffixes add too much incorrect target work. Conditioning also changes
the confidence distribution: the native head's confidence is not assumed to
remain calibrated on filled-placeholder inputs.

## Independent-boundary trajectories

The follow-up recomputes draft proposals at each candidate's own boundaries.
Existing causal lookup has priority; the new head runs only when lookup has no
extension and native confidence passes the gate. Maximum target width is 8.

| Proposal strategy | Target calls | Verification rows | Extra draft calls | Head-only wall, s |
| --- | ---: | ---: | ---: | ---: |
| Retained hybrid control | 198 | 1,242 | 0 | 1.9097665 |
| Conditioned suffix, untrimmed | 190 | 1,316 | 61 | 2.3750340 |
| Conditioned suffix, confidence >=0.5 | 193 | 1,235 | 61 | 2.3930142 |

The filtered form removes five calls and seven rows but adds roughly half a
second of head work. It is a possible secondary contribution, not the main
path to the remaining 25.11-second reduction. No target read count, full
trajectory parity or throughput gain follows from these head-only results.
The untrimmed form increases target rows by 74 and is not selected for a full
run. Production defaults and regression tests remain unchanged.

## Memory and lifecycle

Both screens retain the existing 49 GiB complete incremental head-only bound:
41 GiB active allocation, 4 GiB cache and 4 GiB host. Parameter arrays are
shared across shape-specific owners; sequential graphs and their small output
projection caches fit the existing workspace allowance. The target trunk and
streaming expert banks are never loaded. MLX peak is 21,299,586,448 B in both.

The first guard samples 15,898,120,056 B process-tree footprint and
27,923,136,512 B machine physical usage across nine observations. The trajectory
guard samples 15,426,129,856 B and 26,622,656,512 B across eleven. Both observe
zero compressor growth. MLX allocation, process and machine scopes remain
separate; sampled footprints need not equal the allocator's recorded peak.

Measured source: `61b6899f27fee89f63bf7eed9f1c8aa77e73a53a`. Guards 51071 and
4964 exit 0, reclaim 15,342,174,208 cached source bytes each, and restore exact
Qwen identity and warmup. Final lock release is 12:34:04 UTC. Independent
healthy/idle/warmed/free-lock verification passes at 12:34:43 UTC. No child
or GPU window remains owned by this experiment.
