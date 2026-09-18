# Adjacent-layer prefetch follow-up

**All three variants are rejected.** The retained full run remains13.1509467TPS;
20TPS remains unmet. These are bounded cost/scheduling screens, not full runs.
Measured source:`039e3bd811c64aa645dd89b5b8c85e1c3cf5ab53`.

| Candidate | Candidate/control latency | Control spread | Outcome |
|---|---:|---:|---|
| Last-GU issue, live gate cost | 1.022744 | 0.7514% | 2.27% slower |
| First-GU issue, demand-priority queue | 1.003543 | 0.6348% | no clear win |
| First-GU/priority plus NumPy ranking | 1.004293 | 0.2844% | no win |

Every screen uses real layers30/31/32 and the first64 native M6 routes. Each
layer has105 persistent slots;48 transient slots and16 prefetch slots are
shared. Initial73 resident records are physically loaded before policy restore.
Target31 and the adjacent target32 use settings selected only on training
cycles0–31. Continuous cycles32–63 determine timing, through final speculative
read drain and GPU completion. All192 layer outputs match in every arm.
Output hashing and metrics snapshots occur after timed work finishes.

The real native gate prefix runs on synthetic BF16 inputs. Ranked predictions
come from saved exact-workload scores. This charges gate computation and ranking
but does not establish live predictor parity. Attention is excluded. The shared
three-layer ring also differs from40 layers, so no full TPS projection is valid.
The natural warm/held-out boundary retains any warm speculative work already in
flight; no unmeasured hashing gaps occur anywhere within the64-cycle cohort.

The original last-GU version waits for126–132 of189 issued reads in candidate
arms. Earlier issue gives queued demand reads priority, while active positional
reads finish normally. A fixed four-worker priority queue owns futures and joins
writers before memoryviews release. Native controls retain their original queue.
NumPy ranking prices a9,216-byte6x384 score block. No production defaults change,
and no regression tests or full-model runs follow these non-winning candidates.

Each complete incremental bound is14GiB:10GiB Metal/cache/compiler plus4GiB
host/reader/compiler. Raw banks7,125,442,560B, packed banks6,706,298,880B; retained
outputs70,778,880B are included in the envelope. MLXpeak7,364,297,233B; final
owners16B. Maximum guard process peak7,513,839,760B and machine peak18,749,620,224B,
with zero compressor growth. Guards48399,99230,41743 all exit0, reclaim source
file cache, restore exactQwen and finish warmup before releasing the lock.
Independent health/model/free-lock checks pass after each terminal guard.

`routes.json.gz` stores each exact helper input compressed; its uncompressed
SHA is still pinned by that variant's installation manifest. Other helper hashes,
commands, raw arm results and compact health receipts are retained alongside it.

The paired screen's optimistic5.11% result is therefore not promoted. Its
unmeasured gaps, omitted predictor cost and two-layer ring retention made it an
insufficient performance gate; see the prior lookahead-io receipt.
