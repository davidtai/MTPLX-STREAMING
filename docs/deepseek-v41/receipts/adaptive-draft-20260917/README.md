# Adaptive draft acceptance screen and service guard fixes

**20 TPS remains unmet.** The adaptive full run completed at **12.3057465 TPS**
(83.1318930 seconds), below the prior best of 12.6731624 TPS. All 1,024 output
tokens match the previous native result. The full candidate used 99 expert
slots per layer versus the best run's 102, so this does not isolate the draft
policy's effect. It is not promoted as a throughput improvement.

The selected policy starts at depth 5. After a completely accepted draft it
uses depth 7; otherwise it uses depth 3. It selects the next width using only
the previous committed prefix. Target verification still decides every output
token. The staged runner creates three views of the same compact native draft
parameters and retains the existing native M<=8 verification envelope.

| Draft-only policy | Verification cycles | Verification rows |
|---|---:|---:|
| Fixed depth 5 | 206 | 1,236 |
| Full acceptance: 7; otherwise: 3 | 195 | 1,242 |
| Full acceptance: 7; short acceptance: 3; otherwise: 5 | 195 | 1,282 |

The fixed-depth control reproduces every captured native commit boundary.
The selected candidate reduces cycles by 5.34% with 0.49% more verification
rows. These are acceptance measurements on saved exact target states, not
target execution, physical expert reads, or a throughput improvement.
`draft-only/` preserves the script, hashes, complete proposals and lifecycle.

The draft-only incremental bound was 52,613,349,376 bytes, including separate
active, cache and host allowances. Its allocator peak was 21,299,586,448 bytes;
the guard sampled a process footprint peak of 15,376,608,560 bytes and machine
physical peak of 31,218,286,592 bytes. These are separate measurements with
different sampling. The child exited 0; exact Qwen identity, health and warmup
returned at 23:49:22 UTC. A final independent check confirmed the lock free.

## Full-run admission and cleanup

The first full attempt stopped before model loading: post-Qwen physical RAM
was about 28.6 GB, including 19.9 GB of file-backed memory. Known Qwen,
DeepSeek resident and compact auxiliary cache reclamation completed. A later
read-only scan found zero cached bytes in DeepSeek's `experts.bin` and only
22,790,144 bytes in Qwen's n-gram file. Those snapshots do not identify the
owner of the remaining file-backed RAM.

The second attempt requested an OS disk-cache purge before admission.
`sudo` required a password, so the guard refused the workload and restored
Qwen. No GPU child ran in that attempt. Both refusals preserved the 110 GB
ceiling; neither was an OOM. Their raw logs are in `full-refusals/`.

The runner now waits under the exclusive lock for observable zero active and
queued requests before Qwen shutdown. Unknown activity or unfinished warmup
also prevents shutdown; timeout leaves the service loaded. This is a fresh
activity check, not an atomic HTTP admission drain.

`GPU_WINDOW_PURGE_DISK_CACHE=1` opts into the OS disk-cache flush after shutdown
and foreign-worker checks, before the new memory baseline. It requires cached
administrator authentication and now refuses before taking the lock or
changing the service when authentication is unavailable. The default remains
the existing targeted model-file cleanup. Three focused CPU guard regressions
pass; no unrelated application was terminated.

## Full adaptive result after RAM reclamation

The user purged the OS file cache. The third attempt then measured a fresh
11,518,132,224-byte post-shutdown baseline and admitted prefill 84 / decode 99
slots with a 109,348,083,944-byte peak bound under the 110,000,000,000-byte cap.
The optional purge flag was off. The full target needed 196 cycles versus the
draft-only replay's 195; saved target states do not reproduce every arithmetic
effect of a new verification schedule.

The run read 652,545,884,160 weight bytes / 36,878 records, versus the best
run's 620,943,114,240 bytes / 35,092 records. Read-union time was 50.9830782
seconds, verification 77.3694152 seconds, and drafting 2.0935138 seconds.
These scopes overlap. The 3.2995715-second cache installation is included in
decode wall time. Capacity and schedule both differ from the earlier run.

| Separate memory measure | Bytes |
|---|---:|
| MLX allocator peak | 91,931,939,555 |
| Internal process footprint peak | 93,697,472,216 |
| Guard process footprint peak | 93,730,879,384 |
| Internal machine physical peak | 108,799,164,416 |
| Guard machine physical peak | 108,678,807,552 |

Both machine samplers stayed below 110 GB. The output SHA-256 is
`0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac`.
The fresh candidate's native-AR diagnostic remains the authorized index-297
tie flip; cached AR throughput and memory remain null. The child exited 0,
Qwen identity/health/warmup were restored, and the lock was released at
00:56:30 UTC on September 18. Independent HTTP, process and lock observations
confirmed restoration and no remaining candidate. `full-v3/sha256.json` binds
the 30 preserved result, harness and lifecycle files. No additional tests
were added for this unsuccessful throughput candidate.

## Retained staging

The complete candidate is staged at
`/tmp/dsv41-adaptive-depth-20260917/full`. Its native admission search can use
85..100 slots instead of the old 96-slot floor; all byte, allocator, wired,
copy and prefill bounds remain. The packed plan adds 16 MiB for draft-view
metadata. CPU accounting admits 102 packed slots at the old 9.955 GB baseline,
93 at 15.745 GB, and 92 at 16.5 GB. The live baseline remains authoritative.

The complete v3 harness is now archived with the result. Do not rerun this
unchanged candidate solely because the draft-only replay predicted fewer
cycles. Further throughput work must reduce expert-read or verification cost.

The earlier CPU cache replacement screen rejected both batch-pinned ARC and
S3-FIFO: held-out policy demand misses were 17,411 and 17,193 versus 15,544
for transition-window. This replay used the captured 73-slot residency plus
29 empty slots and does not measure physical transient reuse. Its compact
ledger is `cpu-replacement-screen.json`; no GPU testing followed those losses.

`sha256.json` binds the 20 preserved evidence files. The existing teacher
arrays remain in the previously verified ignored artifact directory.

## Additional read-only RAM audit

The next continuation checked two possible sources of reclaimable file pages.
All 68,403 cold-session blobs (10,424,846,448 bytes on disk) had zero resident
pages. The bounded inventory of 289 task-owned artifact files
(13,576,497,432 bytes on disk) also had zero resident pages. A private 2 MiB
buffered-file control reported exactly 2 MiB resident before the existing
invalidation helper and zero afterward, with unchanged file size and mtime.
These findings do not justify adding session or temporary-artifact cleanup.

Qwen remained healthy, idle and warmed throughout this CPU-only audit. Machine
physical usage was 133,107,056,640 bytes with Qwen serving, including
37,497,257,984 file-backed bytes. Noninteractive administrator validation still
required a password. No service restart, application termination or GPU run
occurred. `ram-cache-audit/sha256.json` separately binds these five evidence
files. This was the historical blocker; the user subsequently purged the
cache and full v3 completed as recorded above.
