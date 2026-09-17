# DeepSeek V4.1 prefill HC post memory reduction

The full 16,384-input / 1,024-output Python workload retains the validated MTP
output digest with only the post-MoE Hyper-Connection combine compiled during
layer-major prefill. The existing chunk/layer fences, routing, hidden captures,
and decode implementation are preserved. This is a memory improvement;
**the 20 decode TPS target remains unmet**.

Measured source: `ca207c3d7f923678f5355ca63a0b89a483120dff` plus the archived
installation wrapper. The promoted arithmetic matches after callable renaming.
It retains the original `hc.combine` diagnostic timing hook; the inactive hook
is stateless and does not retain or evaluate arrays. No per-token checks or
counters are added.

## Measurements

All figures below are bytes. Process and machine figures are sampled; they
must not be added together. Machine usage includes file cache. The full run's
external sampler has a shorter interval than its internal memory sampler.

| Run | Slots/layer | Baseline | MLX allocator peak | Sampled process peak | Sampled machine peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical prefill control | 94 | 8,734,113,792 | 97,146,256,508 | 98,258,299,608 | 107,824,005,120 |
| Compiled post, prefill profile | 93 | 9,240,838,144 | 95,208,123,264 | 96,359,949,288 | 105,955,819,520 |
| Compiled post, complete D5 run | 93 | 9,586,573,312 | 95,208,121,956 | 95,512,926,728 | 105,458,794,496 |

One slot in each of forty layers costs exactly 752,025,600 bytes. After this
capacity normalization, the prefill allocator-peak difference is 1,186,107,644
bytes. Physical/process peaks also vary with the baseline, file-cache residency
and sampling; their raw differences are not an exact allocation formula.

The native-shaped synthetic post-only probe was bit-exact. Temporary-plus-output
allocation fell from 83,886,080 to 41,943,040 bytes. Its compiled median was
0.708 ms versus 1.521 ms eager; these small-probe timings are not production
throughput. The custom rounded-tail variant saved no memory and was rejected.
The separate whole-HC-chain compile experiment is not this change.

The complete run measured 11.6513006265 decode TPS / 87.8013565 seconds for
1,023 timed decode steps, 206 cycles, plus the first prefill token. Its post-prefill
peak, including MTP seed and all decode, was **88,755,252,592 bytes**. A single
counter reset before the decode timer isolates this phase; the wrapper's probe
preserves `max(prefill, post-prefill)` for the full-run headline and detail fields.
The prefill-only and full-run allocator peaks differ by just 1,308 bytes.

The complete run's internal sampler measured process 95,471,050,472 and machine
104,870,838,272 bytes. Its admission used the historical conservative full-run
bound, with no savings credited: physical bound 108,512,945,408 bytes under the
110,000,000,000-byte ceiling. The requested allocator cache limit is a retention
policy: instantaneous cache in the prefill profile reached about 2.7 GB despite
a 1 GiB request. It is not a hard measured-cache bound.

## Output and flag provenance

MTP digest:
`0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac`.
AR reference digest:
`2bd0ad017b9580c8fec340e297696a0bd81a7759b6c5dfe7c5d64de6d40c1090`.
The authenticated AR diagnostic at index 297 and freshly captured candidate
logits classify the existing difference as `tie_flip`, with matching index and
consistent rows. This wrapper rejects a new diagnostic-cache miss instead of
replaying reference logits through candidate arithmetic. Public current-run AR
timing/memory fields are null. The earlier AR source is documented in the
[preceding receipts](../hidden-capture-110gb-20260917/README.md).

Both new runs explicitly bind HC compile, attention compile and window memo
false through the corrected construction-time runner. This does not disable
the separately installed post-only prefill callable. Historical staged wrappers
imported the model before applying their preset; their `arm_env` fields cannot
prove the actual globals. HC/attention prep compilation is row-capped below the
512-row prefill chunks, while window memo can affect prefill retention. The
capacity-normalized comparison above is historical, not a fresh paired run with
recorded identical globals. A conservative upper bound on retained boolean
window masks is 268,435,456 bytes; do not present the whole measured difference
as an isolated causal estimate or reuse the old advertised flags for new decode
performance claims.

## Lifecycle and verification

The prefill profile returned zero; Qwen's exact identity, health and completed
background warmup were restored, then the lock released at 13:22:36 UTC.
The full-output run returned zero; the same checks completed and the lock
released at 13:31:16 UTC. Live API checks confirmed restoration. Another
benchmark acquired the lane afterward; it was not interrupted. Swap use did
not increase during either run.

The archive manifest records source and stored SHA-256 values. `.gz` files are
lossless copies of the original JSONL receipts. Commands retain their original
worktree and temporary paths for provenance; wrappers pin the measured source.
The existing `test_layer_major_is_byte_identical_to_chunk_major` regression
passed under the guard: logits, greedy outputs and every KV lane matched over
its existing seeds and chunk sizes. No new test module or broad suite was added.
The check used a 512 MiB MLX limit and 1 GiB child guard. Qwen restoration and
lock release completed at 13:39:41 UTC; a subsequent live check confirmed health,
exact model identity and completed warmup. Another benchmark then owned the lane.

After the numeric regression, the original timing bracket was retained in the
final loop. A CPU-only check exercises its output registration, and the
normalized method AST matches the measured arithmetic with the real inactive
stateless timing classes. These checks avoid another model/GPU run for restoring
a no-op diagnostic hook. See `promotion-verification.json`.
