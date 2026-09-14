# Actual native-MTP target verify routes on the pinned Python workload

This diagnostic uses source `bd542a39f7ba0bcd17bc5667d519baaca1dd7203`, the
native MXFP4 artifact, the pinned 16,384-token Python prompt and 1,023 decode
steps. It records target expert IDs after the switch's existing routing barrier
and snapshots the 40 expert slot banks at the prefill/decode boundary. It does
not report a comparable TPS: the wrapper records routes and runs only the MTP
pass. The complete earlier 78-slot benchmark remains **9.498635 decode TPS**;
the 20 TPS goal remains open.

The first guarded attempt was **refused before loading the model**: a 14.4005 GB
live baseline put its proposed physical bound over the admission threshold.
The corrected wrapper credits the exact negative slot-storage delta when its
plan selects fewer slots. Its bound cross-checks the measured 72-to-78-slot MLX
peak difference against the corresponding 40-layer storage delta (16,276 B
difference). The refused wrapper and guard log are preserved separately.

| Corrected guarded diagnostic | Measurement |
| --- | ---: |
| Live machine baseline | 13,010,600,000 B |
| Target persistent slots/layer | 73 |
| Admitted physical bound, including graph/capture margin | 107,589,956,504 B |
| Actual sampled physical peak, 250 ms | 100,292,935,680 B |
| Actual MLX active peak | 88,137,917,696 B |
| Swapouts, first → last | 4,399,765 → 4,399,765 pages |
| Target layer routes | 40 layers × 206 verify cycles, complete |
| Native MTP output IDs | 1,024, identical to earlier full run |
| Target expert record reads | 53,999 = 1,015,215,759,360 B |

The guard exited zero, restored the exact Qwen service with completed warmup,
and released the GPU lock. Independent `/health`, `/v1/models`, warmup and
nonblocking lock checks passed. The guard saw no compressor growth.

CPU-only replay of the captured *warm* banks and actual verify routes reproduces
**53,999 record reads exactly**. The runtime's 61,912 `expert_misses` count is
assignment-level and includes repeated expert IDs within a verify batch;
`records_read` is the distinct physical-record count relevant to I/O and
replay. The traces contain 199,773 unique per-layer route requests. At this
73-slot capacity, a clairvoyant policy with temporary service and optional
bypass needs at least **29,812** reads from the same warm state, a 44.79% lower
bound gap. It is not an executable cache policy. At the independently measured
~13.1 GB/s uncached read rate, even this lower bound represents ~42.8 seconds
of I/O for a 51.2-second 20-TPS decode budget, so a useful lane also needs
substantial I/O/GPU overlap.

The installed 73-slot policy would require **19.83 GB/s** sustained reads to
finish its bytes inside that 51.2-second budget even with perfect compute
overlap; the [uncached eight-record proxy](../verify-io-uncached-20260913/README.md)
repeatedly measured ~13.1 GB/s across fanouts 1/4/8. At that observed rate the
target is at most ~35,675 records
before charging any compute. This is a workload-specific planning gate, not a
hardware impossibility proof.

Two causal CPU screens did not justify a production cache change. A sweep of
decayed-frequency admission (tuned on this same trace, therefore optimistic)
bottomed at **52,716** reads, only 2.38% below the installed 2Q policy. A
chronological 100-cycle train / 106-cycle evaluation pairing the previously
captured native draft routes with these target routes improved top-two miss
prediction precision from 11.2% to 14.8%, but would waste over 85% of issued
prefetches. This is not a safe bandwidth trade under the 110 GB limit. The
captured MTP target trace, warm-state replay and these screens should guide a
measured scheduling/compute investigation, not an unmeasured policy install.

The exact route JSON is archived as `mtp-verify-routes-16k-1024-v2.json.gz`
(`gzip.open(..., "rt")`); the wrapper's original uncompressed output remains
under `/tmp/dsv41-110-preflight/`. The 250 ms OS samples, bounds, guard logs,
exact wrapper and CPU-only screen scripts are retained here. The earlier
full-run control is
[`target-slot-band6-20260913`](../target-slot-band6-20260913/README.md), and the
draft route trace is [`mtp-route-capture-20260913`](../mtp-route-capture-20260913/README.md).
