# Current Goal

DeepSeek V4.1: correct memory reporting and runner bugs, then reach 20 decode
TPS on exact 16,384-input / 1,024-output Python under 110 decimal GB.
Latest complete candidate: **12.4439935 TPS; 20 TPS remains unmet.** Native historical best: 12.1146645 TPS; fresh paired control pending.

# Decisions

- Keep allocator, process phys_footprint and machine physical usage separate.
  Missing measurements stay null/n/a. Limits and plans are not measured usage.
- Ceiling 110,000,000,000B includes baseline, Python, Metal, caches and peaks.
  Keep 2 GiB host reserve and bounded allocator overshoot / copy / graph space.
- Use scripts/deepseek_v41/gpu_window.sh directly, never nested; acquire the
  exclusive /tmp/mtplx-gpu-exclusive.lock before MLX or Qwen shutdown.
  Never steal another job's lane or shut down Qwen while requests are active.
- Qwen shutdown reclaims its clean model-file cache automatically. Restore exact
  identity, health and warmup before release; verify recovery after failed runs.
- Work inline, no agents; minimal checks, optimization tests after wins.
  Tie breakers are allowed; arbitrary or unclassified output drift is not.
- Preserve Claude W126/W127/W128 worktrees; keep unvalidated bounded KV off.

# Plan Status

Executing docs/plans/2026-09-16-deepseek-v41-20tps-stage.md; Task 4 remains open.
Memory/runner fixes include actual measured-pass summaries, phase budgets,
fresh Mach footprint reads with cached bindings, short-reply rejection,
missing diagnostic logits remaining unclassified, and sampled guard peaks.
Newest fix: current storage and plan transient bytes are reported separately
from source_expert_record_bytes. 37 CPU reporting cases pass with real MLX
imports forbidden; generation ASTs are unchanged.

# Latest Complete Candidate

Measured source 4a5acf9d4: lossless global packed scales, native FP4 weights,
native BF16 target head, compact MTP 93/58/32, D5/M6, 91 prefill -> 102 decode,
48 shared transients, pf0, transition-window, miss parts 3, shared overlap,
fanout setting 4 but three weight-plane reads per record; max KV 17664.
HC/attention/window compile flags false; prefill HC post remains compiled.

12.4439935459 TPS / 82.2083357910s, 206 cycles, all 1,024 output IDs identical.
Installation 3.4511022910s is charged to decode. 35,058 expert records read
620,341,493,760 weight bytes; packed-scale installation reads another
3,086,136,060B. Read union 47.811560985s; software concurrency 10.9616, not HW QD.
Historical native 93->100: 12.1146645 TPS / 84.4431145420s, 35,880 records /
674,566,963,200B; resize 1.954109s. This is not a fresh paired comparison.

Allocator peak 94,044,462,082B; internal sampled process 95,847,005,496B and
machine 106,215,473,152B. Baseline 9,613,836,288B; conservative bound
109,483,268,328B. Guard machine peak 106,198,253,568B, 226 complete samples,
zero reported compressor growth. Physical weight slots 73,043,804,160B;
resident packed scales 3,086,136,060B; raw scales released 4,078,632,960B.

Receipt: docs/deepseek-v41/receipts/resident-packed-scales-20260917/README.md.
Original slot-summary record/transient fields are stale; the derived corrected
receipt preserves every timing, memory observation and token. The inherited
capacity_selection label was stale too; the live helper's label is corrected.
Full output SHA: 0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac.

# Artifact and Lifecycle

Complete source-record SHA coverage and exact scale reconstruction: all 15,360
records / 288,777,830,400B. Packed inventory is 3.086 GB versus 16.987 GB raw.
Payload: ignored benchmarks/raw/deepseek-v41-resident-scales/20260917;
/tmp/dsv41-resident-scales-20260917/artifact is a symlink to it. 360 files hashed.
Small proof covers persistent103 / transient48, nonidentity slot/expert indices,
all 384 layer-20 scales and four exact MLP cases. Close leaves 8 MLX bytes;
peak allocator 2.841 GB within an 8 GiB admission bound.

CPU export accumulated source file cache despite F_NOCACHE, invalidating its
proposed 20 GiB incremental bound. Physical used stayed below 110 GB, but Qwen
restore timed out (exit 10). Verified-source reclamation removed 122.343 GB of
cached pages including speculative/free pages; exact Qwen recovery was checked.
Do not rerun that exporter unchanged. The runtime loader uses direct Metal
buffers and disables read-ahead. Its guarded finally reclaims source and packed
files before Qwen restore. The full candidate exited 0; exact restore/warmup and
lock release completed 19:39:28 UTC. A native control retry at 20:08 UTC was
refused before model loading: baseline 28.353 GB, allowed <=9.672 GB for cap 100.
Exact Qwen restore/warmup and free lock were independently verified at 20:11:41 UTC.
All our children are terminal. See docs/deepseek-v41/receipts/resident-packed-scales-control-refusal-20260917.

# Open Work and Retained Context

- Live helpers: /tmp/dsv41-resident-scales-20260917. One-request benchmark only;
  general serving needs a prefill reload/shrink design. Do not enable by default.
- Native D5 cap91->100 control needs <=9.672 GB baseline at the recorded wired
  usage. Use a new evidence prefix only when headroom changes; keep all bounds.
- Refresh live installation HEAD/source hashes after commits. Measured archives
  are immutable; phase_memory_control_sha256 names a receipt, never a wrapper.
- Native growth helpers: /tmp/dsv41-cache-growth-20260917; native M6/M8 bounds:
  /tmp/dsv41-depth7-full-20260917. The cap93 attempt was refused before loading.
- Valid attribution: receipts/decode-read-attribution-20260917. Native M6 loop
  84.671768s includes 52.271397s missing-read wait and 19.940574s Metal eval/encode.
  Do not repeat cProfile; its threaded Python 3.12 attribution was corrupt.
- Do not repeat unchanged rejected fanout8, D7-full, staged3+3, D3-full, rANS,
  XOR/reference coding, cache-policy, prompt-lookup, confidence0.5, HC or q8-head
  candidates. Down-only specialization gives small larger-case MLP gains.
- Teacher: /tmp/dsv41-depth-replay-20260917 and ignored
  benchmarks/raw/deepseek-v41-depth-teacher/20260917. Do not recapture.
- AR cached logits cover only divergence 297; never reuse them for 376 or 480.
