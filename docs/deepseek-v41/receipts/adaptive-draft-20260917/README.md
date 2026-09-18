# Adaptive draft acceptance screen and service guard fixes

**20 TPS remains unmet.** The best completed 16,384-input / 1,024-output run
remains 12.6731624 TPS. This receipt contains a promising draft-only result;
the adaptive candidate has no full-target throughput measurement yet.

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

## Retained staging and next step

The complete candidate is staged at
`/tmp/dsv41-adaptive-depth-20260917/full`. Its native admission search can use
85..100 slots instead of the old 96-slot floor; all byte, allocator, wired,
copy and prefill bounds remain. The packed plan adds 16 MiB for draft-view
metadata. CPU accounting admits 102 packed slots at the old 9.955 GB baseline,
93 at 15.745 GB, and 92 at 16.5 GB. The live baseline remains authoritative.

The user has been asked to run `sudo /usr/sbin/purge` in their own terminal;
do not request or collect the password. After reclamation, refresh the staged
commit and source hashes, use a new `v3` output prefix with the optional purge
flag off, and run through the canonical guard. Require all 1,024 native output
tokens or the authorized index-matched tie gate, complete cleanup, and a fresh
throughput receipt before retaining this as a speed improvement.

The earlier CPU cache replacement screen rejected both batch-pinned ARC and
S3-FIFO: held-out policy demand misses were 17,411 and 17,193 versus 15,544
for transition-window. This replay used the captured 73-slot residency plus
29 empty slots and does not measure physical transient reuse. Its compact
ledger is `cpu-replacement-screen.json`; no GPU testing followed those losses.

`sha256.json` binds the 20 preserved evidence files. The existing teacher
arrays remain in the previously verified ignored artifact directory.
