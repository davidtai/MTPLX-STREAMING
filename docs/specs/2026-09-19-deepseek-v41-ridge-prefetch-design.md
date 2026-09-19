# Compact router correction with real prefetch scheduling

The best complete Q4 run spends 70.22 seconds in target verification and
records 44.11 seconds of expert-read activity. The existing compact predictor
improves saved physical-miss coverage from 15.64% to 21.77%, but that quality
screen has no finite lead time or predictor cost. Measure that missing cost
before considering a full integration.

Reuse the source-pinned three-layer continuous replay for layers 30–32, with
105 slots per layer, 48 shared transient slots, and a shared 16-slot prefetch
ring. Keep the unchanged native control and the existing first-GU issue point
with demand-priority reads. Change only the proposal scores and the charged
predictor computation to the compact prompt-trained ridge correction. Target
routes, expert weights, kernels, and ownership rules remain unchanged.

Reconstruct the two required adapters from the independent W35 prompt using
the already-selected regularization; require their exact saved parameter
hashes. Use the previously selected settings for target31 (width6, margin0.1,
at most12 reads) and target32 (width6, margin0.1, at most4 reads). No new tuning
on the held-out cohort occurs. The two simultaneous budgets fit the 16 slots.
Prepare CPU parameters under the guard before the GPU child; CPU fitting has
its existing 1 GiB bound and terminates before the 14 GiB GPU phase begins.

The GPU computes the native FP32 gate projection, sqrt-softplus scores, and
the affine score correction on synthetic BF16 inputs. Saved exact-workload
prediction scores select reads, as in the predecessor. This prices the live
predictor operation but does not establish its live full-model predictions.
Attention remains omitted. Prefetch errors cause extra reads, never altered
target weights or routes. All layer outputs must match the native control.

Run one A/B/A comparison, including continuous warm and held-out cohorts,
final speculative-read drain, and GPU completion. Keep measurements outside
the hot path and use existing reader/runtime counters. Do not add regression
tests before a win. A material gain beyond control spread can justify the
next bounded integration step; a flat result does not justify a full run of
this same schedule. Neither component timing nor prediction coverage proves
20 TPS on the full 16K/1K workload.

Failure modes: late predictions can leave reads unfinished; false reads can
consume shared bandwidth or evict useful ring owners; a three-layer ring and
omitted attention can differ from full-model timing. Preserve these limits
in the receipt and retain the original guard, writer-join, and buffer-lifetime
rules. No production default changes are part of this experiment.

The first comparison completes with exact outputs but a 1.0988% latency loss
against 0.5336% control spread. Of 209 prefetch records, 128 still require a wait
at the next route; total physical reads rise 1214→1250. Do not run this full.

A source review identifies a distinct scheduling opportunity: first-GU issue
currently means the first missing part's gate/up witness. Resident-hit work
does not trigger it. The next bounded variant issues once after current demand
parts are submitted and resident/shared GPU roots are enqueued, removing the
per-part issue trigger. Keep the identical frozen predictor, queue, ring, and
target arithmetic. This adds lead time but can let an already-started
speculative read delay a subsequently enqueued demand read; measure it in one
A/B/A comparison. Reuse the verified parameter artifact instead of refitting.

The earlier issue point finishes with exact outputs, a 0.3085% held-out latency
improvement and 0.1857% control spread. Its whole 64-cycle replay is 0.0236%
longer. Late reads fall to 117, but total traffic is unchanged at 1250 records.
This is not a material gain. Neither schedule is promoted or run full-model;
no optimization regression tests follow. Preserve the two results and the
corrected memory inventory in `../deepseek-v41/receipts/ridge-prefetch-20260919`.
