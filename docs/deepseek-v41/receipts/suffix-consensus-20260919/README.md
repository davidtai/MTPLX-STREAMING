# Q4 causal consensus suffix

The native D5 plus original lookup control remains authoritative. A new
proposal-only extension uses a repeated suffix of committed text when the
original lookup adds nothing and all native confidence scores are at least 0.9.
At least two occurrences must unanimously support each additional token.
The suffix search covers lengths six down to three and adds at most two tokens.
No target weights, target verification, acceptance, or committed KV change.

Source: `4b5591e8245addba2c96105cc37c34e8a63233a5`. Scratch root:
`/tmp/dsv41-q4-suffix-consensus-20260919`. Q2 remains stopped.

## Selection and measured head trajectory

The CPU screen selects suffix length three on the first half only. At the
198 saved independent control boundaries it adds 16 matched tokens for 18
additional rows; the second half contributes 9/9. This cannot predict new
cycle boundaries or full throughput.

The subsequent native head replay evaluates each candidate at its own new
boundaries using authenticated exact target states:

| Arm | Target calls | Target rows | Verification widths |
| --- | ---: | ---: | --- |
| Original hybrid control | 198 | 1,242 | M6:171; M8:27 |
| Consensus suffix | 195 | 1,240 | M6:158; M7:4; M8:33 |

All original control positions, proposals, and commits reproduce exactly.
Teacher future is available only to the scoring/seeding code after proposals
are constructed. It does not enter the consensus index. Single sequential
head times of 1.93508 and 1.87364 seconds are not a repeated speed comparison.
This screen executes no target trunk and establishes no full TPS result.

## Memory and lifecycle

The unique-token 17,408-token host inventory contains 15,215,316 additional
Python payload bytes. A separate **32 MiB** reserve covers the new index,
bringing original lookup plus consensus metadata to **48 MiB**. Interpreter,
allocator arenas, and other runtime memory retain their existing reserves.
This allowance is charged during prefill, growth, seed, and steady decode.

The head-only complete incremental bound remains 49 GiB: 41 GiB active,
4 GiB allocator cache, and 4 GiB host. At a 10,846,617,600-byte baseline, the
successful guard samples 27,138,686,976 whole-machine bytes and 15,832,851,264
child-tree footprint bytes. MLX peak is 21,299,586,448 bytes. These metrics
overlap and must not be added. Nine guard samples show no compressor growth.
Child and guard exit zero. Source cache reclamation removes 15,342,174,208
bytes; exact Qwen identity, health, and warmup are restored before release
at 12:21:39 UTC. Independent post-checks encounter later foreign lock owners
and an unavailable API. A later independent idle/warmed/free-lock observation
before the full candidate is preserved as `head/later-idle-observation.json`.

The first head attempt times out after 120 seconds waiting for the lock and
launches no child. Its log is preserved separately from the successful retry.
The CPU screen also preserves its initial pre-lock rejection of a 512 MiB
child cap below the guard's 1 GiB minimum.

## Full candidate construction

`consensus_install.py` composes the original lookup and consensus before
generation. It reads the five already-evaluated FP32 confidence logits on the
host and compares their minimum to `log(9)`, avoiding a new GPU sigmoid.
Every saved sigmoid decision lies more than 0.002 from 0.9. A CPU construction
audit reproduces all 195 saved proposals and reverse-audits the decode rewrite,
including unchanged native MLX calls. This is not fresh target parity.

The existing packed lane uses at most 48 expert assignments at M8. Its grids
and output allocations depend on row count; M7 is inside that envelope. The
owned native output projection uses the same row-dependent matmul. Installation
now checks fused projection routing for every width one through eight.
Existing compiler, cache, KV, copy, page, and wired allowances remain charged.

The no-MLX preflight admits 110 slots at the historical baseline with a
109,459,159,260-byte physical bound and 1,455,550,464-byte host reserve.
The full runner must recompute this with the fresh post-reclamation baseline;
that preflight is not permission to force 110 slots on the live machine.
Production defaults are unchanged.

## Complete 16K/1K result

The full candidate completes at **13.3997126089 TPS / 76.3449209590 seconds**,
with 195 target calls. All 1,024 native output IDs match exactly; SHA-256 remains
`0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac`.
The output is the same coherent beginning of the requested Python helper diff,
truncated at the 1,024-token cap. It is not a complete validated patch.
The exact text, including unified-diff context whitespace, is preserved in
the full JSONL receipt's `dspark.decoded_text` field.

The fresh 11,047,714,816-byte baseline admits 84→109 slots, with a
109,546,420,444-byte physical bound. Independent guard peak is 109,272,825,856 B;
the runner samples 109,262,798,848 B. Runner process footprint peaks at
97,311,606,592 B and MLX at 96,299,123,174 B. These are overlapping metrics.
The guard records 217 samples and zero compressor growth. Host reserve is
1,455,550,464 B, including the new 32 MiB allowance.

Expert traffic is 32,357 reads / 572,548,055,040 B, with 44.3192 seconds of read
activity. Draft time is 2.02812 s, target verification 70.21019 s, and charged
bank growth 3.65343 s. The historical 110-slot control is 13.4141517619 TPS /
76.2627423750 s with 198 calls and 31,961 reads. The current result is 0.08218 s
longer and has a different capacity; it establishes neither an isolated gain
nor a regression caused by consensus. **20 TPS is unmet.** Keep the prior full
winner and do not promote this candidate or add optimization regressions.

Guard 1944 and its child exit 0. Exact Qwen identity, health, and completed
warmup are verified before lock release at 12:45:47 UTC. A later independent
health request encounters another foreign GPU window with an unavailable API.
`full-v1/lifecycle.json` records that distinction. No owned GPU child or waiter
remains; subsequent jobs are left alone. The raw result, bound, guard log,
installer, and source hashes are preserved under `full-v1/`.
