# Source-matched destination alignment screen

Rejected: matching destination page offset to the source offset changes CPU
read latency by only0.15188%, inside0.82697% control spread. No GPU bank-layout
change, full run or new tests are justified.

Measured source:`19ea3ac2f888edf5035e3a43bc314bea64f3cb6f`.
This CPU-only screen blocks MLX imports and uses the native three-preadv reader
with fanout4. The two destination choices are16KiB page-aligned mmap buffers
and offsets matching each source plane modulo16KiB. Neither choice changes
record counts, requested bytes, source packing or the native reader implementation.
Actual Metal-buffer base alignment is not measured.

Five interleaved arms process128 batches of3 records. Source coverage contains
576 planes at offset0 and576 at8192 modulo16KiB. The final three complete
records are checked against their native payload digests, and their weight
planes match across all arms. Intermediate buffers are not individually hashed.

| Arm | Read wall time, seconds |
| --- | ---: |
| Page-aligned before | 0.5493738340 |
| Source-matched 1 | 0.5437927500 |
| Page-aligned middle | 0.5448622500 |
| Source-matched 2 | 0.5456648750 |
| Page-aligned after | 0.5455574170 |

Median candidate/control ratio:0.9984811782. This is CPU reader evidence, not a
full decode-throughput measurement or proof of actual Metal destination geometry.

The2GiB host bound covers53,231,616B of mmap destination storage, an18,800,640B
verification buffer, the34,191,301B manifest and reader/host overhead. Guard38108
records322,471,160B peak process footprint and11,048,566,784B peak machine physical
usage across9 samples, with zero compressor growth. No MLX allocation is made.

The guard exits0, source clean pages are reclaimed to0, and exact Qwen identity
and warmup are restored before lock release at11:15:34UTC. Independent health,
idle, warmup and free-lock verification passes at11:19:55UTC on2026-09-18.
