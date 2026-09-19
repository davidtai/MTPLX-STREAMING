# Bounded paired prefetch I/O screen

Measured source `0f7146768f93423956ac6ce48ebe9b549cdde4a8`.
This is an optimistic scheduling screen, not a full-model performance result.
The retained full run remains 13.1509467 TPS; 20 TPS is unmet.

Layers 30 and 31 replay the first 64 exact native M6 routes with real packed
weights, 105 persistent slots per layer, 48 shared transient slots and a shared
16-slot prefetch ring. Target 31 was selected by useful predictions in training
cycles 0–31; cycles 32–63 decide timing. Predictions replay recorded causal
scores. Inputs are identical synthetic BF16 values in every arm.

| Arm | Held-out pair latency (ms) | All reads | Exact outputs |
|---|---:|---:|---|
| Native | 410.971 | 793 | yes |
| Prefetch | 395.448 | 762 | yes |
| Native | 414.805 | 793 | yes |
| Prefetch | 386.923 | 762 | yes |
| Native | 412.258 | 793 | yes |

Median ratio 0.9488844, control spread 0.9301%. All 128 layer outputs match in
each arm. Live router computation and attention are excluded. Per-pair hashing
and terminal speculative drain occur outside the timers, allowing background
reads to progress during unmeasured intervals. Two-layer ring retention also
differs from the full 40-layer model. Do not extrapolate this ratio to TPS.

The isolated configuration type admits only this fixed experiment; the public
transition-window/prefetch exclusion is unchanged. Speculative workers provide
a no-op early-GU witness and retain full-record READY publication. Issuing after
all current GU witnesses preserves queued demand priority. A source-pinned copy
of the issue method removes stale recent-miss suppression and filters actual
READY physical owners. Native tickets, leases and completion handling remain.

The complete incremental bound is 11 GiB: 7 GiB Metal/cache/compiler plus 4 GiB
host/reader/compiler. Raw bank bound 5,151,375,360 B; packed banks 4,848,353,280 B.
MLX peak 5,304,624,137 B; guard process peak 5,797,661,048 B and machine peak
17,113,088,000 B. Final Metal owners fall to 8 B. Guard 40813 exits 0; exact Qwen
restoration/warmup and lock release finish at 08:46:12 UTC. Independent health,
model identity and free lock are verified. Source file cache is reclaimed.

Next: three adjacent layers sharing the ring, continuous cohort timing through
terminal drain, and explicitly priced live gate-shaped computation. Captured
router inputs were not saved, so a cost screen cannot establish live predictor
parity. No production default or full-model prefetch is promoted here.
