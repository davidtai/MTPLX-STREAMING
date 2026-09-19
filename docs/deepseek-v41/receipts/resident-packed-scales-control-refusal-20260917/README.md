# Native control admission refusal, 2026-09-17

The native D5/M6 16,384-input / 1,024-output control was attempted at source
`9c8c1237d483eeb244ba8415dc40145b601bbfcf`. The guarded child exited 1 before
loading the model: its post-Qwen physical baseline was 28,352,921,600 bytes.
There is no new throughput or token-parity result from this attempt.

The wrapper reported `invalid measured baseline` because this valid reading
exceeded its 20 GB staging range. A CPU-only check of the unchanged admission
formula independently confirms refusal: even cap91 prefill projects
126,890,293,828 bytes. The native cap91->100 control requires a baseline no
higher than 9,672,294,676 bytes under its 109.5 GB admission target and recorded
wired memory. The 110,000,000,000-byte ceiling and all margins are unchanged.
Do not retry this command with the same baseline or reuse its evidence prefix.

Automatic stopped-Qwen reclamation covered all 22 safetensor files and reduced
cached model pages from 22,790,144 bytes to zero. These cached-page counts are
separate from whole-machine physical use. A later read-only mincore scan found
zero cached pages in DeepSeek's 288,777,830,400-byte experts.bin and all 360
packed-scale payload files. It read no payload and invalidated no pages. The
remaining baseline was not attributed to a particular other file or job.

The guard restored the exact Qwen model, waited for background warmup, and
released the GPU lock at 20:11:04 UTC. Independent checks at 20:11:41 UTC found
Qwen healthy/idle, warmup done, the lock free, and no native-control child left.
A request served during restoration delayed warmup; it was not interrupted.

The retained complete candidate remains 12.4439935 TPS at 106,215,473,152 bytes
sampled machine usage. Its fresh native comparison and the 20 TPS goal remain
open. All 55 files in its prior archive were verified against committed Git
blobs before this attempt; the measured archive is unchanged.
