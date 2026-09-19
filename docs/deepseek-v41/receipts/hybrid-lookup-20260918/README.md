# Native MTP draft with causal lookup extension

The previous standalone prompt-lookup screen was rejected at 575 cycles versus
native D5's 206. This candidate retains all five native MTP proposals. It appends
at most two more tokens only when that complete proposal occurs in known prompt
or committed output text with at least two matching context tokens. Longest
context wins, with the earliest occurrence breaking ties. Source text contains
no uncommitted future tokens. The unchanged target accept/commit path verifies
the resulting six-to-eight input rows.

## Opportunity and draft-state screens

The CPU screen uses saved native proposals at each of 206 native boundaries.
Teacher futures only score the selected continuations. These are independent
opportunities, not a hybrid cycle count: changed boundaries require fresh draft
head evaluation. Longer extensions add many rejected rows. The first half
supports the conservative two-token, two-context candidate: 32 extra correct
tokens across 40 added rows. The second half gives 12 across 28. No parameters
are selected from the second half.

The guarded draft-head replay then follows actual new cycle positions using
the saved exact target hidden states and initialized MTP windows. Its native
control reproduces all 206 original commit lengths. The hybrid needs **198
cycles**, with **1,242 target rows** versus native **1,236**: 171 six-row and
27 eight-row batches. Target execution, new target routes and full throughput
remain unmeasured by this replay. Batch-dependent target/seed arithmetic can
still change the complete candidate's trajectory.

The complete incremental bound is 49 GiB: 41 GiB active, 4 GiB allocator/cache
allowance and 4 GiB host. All native draft ASTs match the prior validated screen
except an unused prefill-detachment method. The source model arithmetic file
is byte-identical. The changed runtime constant standardizes the same 110 GB
budget; changed decoder diagnostics are not used by this head-only replay.

MLX peak is 21,299,586,448 B. Guard samples peak at 14,989,332,344 B process
footprint and 26,319,831,040 B machine usage. Those overlapping measures are
not added. Guard 7091 exits 0 and the supervisor reclaims 15,342,174,208 source
page bytes to zero after the child exits. Exact Qwen restoration and warmup
complete before release; independent health/idle/warmup/free-lock checks pass
at 10:29:27 UTC.

## Full candidate construction

The full candidate composes with the exact input-row cache and attested strict
allocator. It retains the proven native M8 tensor bound and adds 16 MiB host
allowance through both admission and the real CLI resolver. The lookup's complete
17,408-token history uses 3,700,278 B in the native output inventory; a conservative
10,661,888-byte structural bound fits the allowance even when keys differ.

Installation binds the single greedy request before prefill. The source rewrite
retains the exact native MLX calls and accept/commit arithmetic. The draft head
still produces five tokens; reported maximum proposal depth is seven, configured
maximum verify width is eight, and KV admission prices eight lookahead rows.
The actual proposal length determines each target width. Engagement is derived
from existing depth statistics, without new per-cycle proof counters. Only the
known prompt and emitted target-verified tokens enter the lookup index.

## Full result and limits

Measured source is `48de2aaac8c13c5d31cfbeb8ee5bc0f92f78f9b3`. One complete
16,384-input/1,024-output run reaches **13.4141517619 TPS / 76.2627423750 s**,
with all 1,024 native output IDs unchanged. SHA-256 is
`0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac`.
The prior input-row result was 13.3195300354 TPS / 76.8045116670 s: this saves
0.5417692920 s, a 0.7104% TPS increase. Background differs and neither arm is
repeated here, so this is a best single result, not an isolated or repeatable
speedup claim. **20 TPS remains unmet.**

The full run confirms 198 cycles and target calls, versus 206 native. Existing
accepted-by-depth statistics end in `[23, 19]`, showing both extension positions
are used. Physical expert reads increase by 25 records to 31,961, or
565,540,945,920 B. Read union is 44.111376676 s; that is not GPU idle. The small
gain therefore does not establish a material expert-I/O reduction.

Admission grows 84->110 slots/layer with 48 shared transients and native KV16.
At baseline 10,447,192,064 B, the bound is 109,631,928,540 B, including
1,421,996,032 B host reserve. MLX peak is 97,006,846,938 B, process footprint
98,027,571,368 B and machine physical peak 109,238,927,360 B. These overlapping
measures are kept separate. Final inactive cache is 239,688,936 B, below the
268,435,456-byte strict allocator limit. Guard machine peak is 109,181,698,048 B
across 217 samples, with zero compressor growth.

Guard 39139 exits 0. Candidate source cache is zero after cleanup; exact Qwen
restoration, health and warmup complete before lock release at 10:38:22 UTC.
Independent model/health/idle/warmup/free-lock checks pass at 10:39:43 UTC.
Two focused host regressions, added after this run, pass with real MLX imports
blocked. They cover causal continuation ownership, complete proposal matching,
context/tie ordering, query immutability and incremental index updates.

## Longer-extension follow-up

A CPU-only screen retains these 198 saved boundaries and verifies longer causal
extensions in existing eight-row chunks. Maximum extensions of 6/10/18/26 add
46/74/103/106 accepted tokens but require 16/16/22/24 additional target calls and
increase total target rows from 1,242 to 1,306/1,370/1,418/1,434. Counts overlap
across independent boundaries: they are not a new trajectory, cycle forecast or
measured latency. Their added work does not justify another GPU window now.
No longer extension is installed or promoted.

The implementation remains an explicit, one-request experimental runner. No
production service default changes. Fixed Q8 and full 256K prefill verification
remain secondary to the 20 TPS goal.
