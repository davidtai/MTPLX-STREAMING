# DeepSeek V4.1: captured-hidden lifetime and 110 GB results

The exact 16,384-input / 1,024-output Python workload reaches **11.7203483
decode tokens/s**, versus the earlier 9.6589676 result. **20 tokens/s remains
unmet.** These are individual full-workload measurements; the combined gain
includes cache capacity, admission policy, miss partitioning, and shared-work
overlap, not just the lifetime fix.

## Measured change

Layer-major prefill captures the mean of each selected target layer's
Hyper-Connection state. Leaving those means lazy retained earlier layers'
larger source arrays until final concatenation. Evaluate the captures at the
existing per-chunk fence, together with the attention and MoE inputs. The
reduction arithmetic, complete prompt context, captured values, and fence count
are preserved. The promoted module's AST matches the measured candidate exactly;
see `hidden-capture-promotion-equivalence.json`.

All model runs below used base source
`e589c1e4b17856f506d90d9fb2bbb5ce45711648`. The archived wrappers contain the
exact construction-time patches and have hashes matching their bounds receipts.

| Full run | Depth | Slots/layer | Decode seconds | Decode tokens/s | Expert reads | MLX peak, GB |
|---|---:|---:|---:|---:|---:|---:|
| Full AR + MTP reference | 3 | 91 | 92.7570 | 11.0288 | 38,010 | 97.8954 |
| Reused-AR control | 3 | 91 | 93.4632 | 10.9455 | 38,010 | 97.9015 |
| Captures at existing fence | 3 | 90 | 94.2624 | 10.8527 | 38,474 | 94.1382 |
| Released memory assigned to slots | 3 | 94 | 91.6564 | 11.1613 | 36,692 | 97.1463 |
| Matched full-depth comparison | 5 | 94 | 87.2841 | **11.7203** | 38,613 | 97.1463 |

The live baseline required 90 slots for the first capture experiment. One slot
in each of 40 layers costs exactly 752,025,600 bytes. After accounting for that
one-slot difference, the allocator peak falls by **3,011,286,868 bytes**. The
94-slot run then confirms the resulting allocation: four additional slots per
layer add 3,008,102,400 bytes. This is a storage-normalized memory comparison,
not a speed comparison at equal capacity.

The complete MTP token digest is unchanged in every row:
`0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac`.
The AR digest is
`2bd0ad017b9580c8fec340e297696a0bd81a7759b6c5dfe7c5d64de6d40c1090`.
Their first difference remains the accepted tie at index 297: AR's contested
logits are equal and MTP's margin is 0.25. Fresh MTP logits pass the
index-matched classification in each run.

## Memory and lifecycle

The final depth-five run retains native BF16 target arithmetic, compact MTP
expert banks `(93, 58, 32)`, 94 persistent slots per routed layer, 48 shared
transient slots, full six-row verification, transition-window admission,
three-record miss parts, shared-work overlap, fanout four, and no prefetch.
`max_kv` is 17,664. Admission is specific to this artifact and workload.

| Measure or allocation | Bytes |
|---|---:|
| Whole-machine ceiling | 110,000,000,000 |
| Measured machine baseline | 8,835,645,440 |
| Python/host reserve | 2,147,483,648 |
| Derived MLX allocator limit | 99,016,870,912 |
| Engine allocation budget | 90,946,870,088 |
| Outer transient/cache allowance | 8,070,000,824 |
| Retained allocator cache budget, included above | 1,073,741,824 |
| Admitted allocator peak bound | 97,213,381,724 |
| Admitted whole-machine peak bound | 108,645,754,112 |
| Measured MLX peak | 97,146,257,816 |
| Sampled process `phys_footprint` peak | 98,103,862,608 |
| Sampled internal whole-machine peak | 107,370,692,608 |
| Sampled external whole-workflow peak | **107,453,988,864** |

Allocator, process, and whole-machine measures overlap; do not add them.
Physical usage includes active/inactive file cache. The external sampler runs
every 250 ms and covers loading and diagnostics as well as generation.
Admission adds a 64 MiB allocation margin, a 500 MB physical margin, positive
baseline drift, and a separate 2 GiB graph/wired headroom check. An earlier
4 MiB estimate missed one allocator peak by 2.84 MB and was increased before
subsequent admission. No observed physical usage crossed 110 GB.

The guard automatically reclaims Qwen's clean model and n-gram file cache after
the owned service processes exit. The reference-reuse window reclaimed
22,103,064,576 cached bytes and independently measured a 22,211,362,816-byte
physical reduction. File identities, sizes, and modification times remained
unchanged. Cache-resident bytes and physical reduction are distinct measures.

Every completed window exited zero, restored `com.tea.qwen` with model ID
`mtplx-flash-next-optimized-speed`, completed background warmup, and released
`/tmp/mtplx-gpu-exclusive.lock`. See the guard logs and
`promotion-service-health.json`. No other GPU job was terminated.

## Reporting and validation

Commit `e589c1e4b` fixes DSpark's `mlx_active_*_at_decode_start`: sample after
prefill, immediately before the runner starts its decode clock. Earlier
receipts mislabeled a before-prefill value. Their other peak measures and
decode-end active fields are unaffected. Eighteen CPU memory-reporting checks
passed. Commit `c9f090573` fixes Bash 3.2 nounset handling of an empty auxiliary
path array in the guard; empty, spaced, plural, duplicate, and conflicting
argument cases were checked without a model load.

After the successful full model experiment, the existing tiny quantized
prefill checks passed for both large-prefill and small-verification cases,
including hidden/logit/KV parity across two requests and subsequent decode.
No new optimization test module or broad GPU test suite was added.

The temporary benchmark workflow validates and reuses the full AR reference,
then caches the single AR diagnostic logits row with source, prompt, token, and
payload hashes. Candidate MTP logits are always fresh. Public AR timing,
memory, and counters are null in reused-reference receipts; reference timing
appears only in explicit provenance. This removes repeated AR generation and
diagnostic replay from later experiments.

## Rejected work and remaining constraint

- Staged `3,3` verification lost despite fewer reads. Its old cache and
  eighteen-transient allocation projections do not apply to full M4/M6.
- Prefill chunk 256 saved only 17 MB before the capture fix and changed the AR
  output. Its owned child was intentionally stopped; Qwen was restored.
- Reclaiming all unused decode projections at prefill entry had no peak or
  speed benefit. Neither rejected prefill variant was promoted.
- Full-M6 CPU replay rejects `transition-window-tuned`: at 91 slots it needs
  40,145 reads versus 40,052 for the existing policy. Uneven allocation with
  five bank shapes projects about 3% fewer reads, not a measured GPU gain.
- MTP seed setup did not increase the measured allocator peak. Seed truncation
  is not supported as the next peak-memory optimization by this receipt.

The winner reads 725,949,112,320 expert bytes in an active I/O window of
55.644538 seconds. A 20-token/s result permits only 51.15 seconds for all 1,023
decode steps. The next stage must reduce I/O cost as well as the remaining
verification work; cache-policy replay alone does not establish that result.

## Evidence use

`screen-summary-20260917.json` summarizes this batch and the preceding short
screens. `manifest.json` records stored hashes and original-byte hashes. JSONL
receipts and exact generated-text sidecars are gzip-compressed losslessly. `predecessor/` retains the cap-80
memory anchors required by the initial bounded wrapper. Reproduction needs
the pinned artifact, compact residents, original paths or restored copies of
the recorded predecessors, and the parent-held GPU guard. The wrapper snapshots
are tied to their recorded base source; they are not general runner defaults.
