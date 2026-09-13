# Sampled MXFP4 compression headroom

Twelve authenticated 18,800,640-byte records: layers 0/8/16/24/32/39, experts
0/191, read with F_NOCACHE. Original bank and manifest remain unchanged. All
source hashes and local zlib round trips match exactly. No MLX import or model
allocation; this is a sampled screen, not a full-bank ratio or serving benchmark.

| Measure | Result |
| --- | ---: |
| Raw sample bytes | 225,607,680 |
| zlib level1 bytes | 204,963,287 |
| Actual sampled compression ratio | 1.10072x |
| Actual sampled byte saving | 9.1506% |
| Ideal whole-record order0 entropy saving | 6.7587% |
| Ideal separately modeled component entropy saving | 10.7525% |
| Median CPU zlib decode per record | 31.315 ms |

Entropy estimates omit tables, framing, and decode cost. zlib can beat the
whole-record order0 estimate because it exploits structure beyond that model.
Even the measured byte saving is too small to close the current gap to 20 TPS.
Do not enable zlib in serving: its measured CPU decode is much slower than the
current uncached read service per record.

Existing `expert_streamed_codec.py` preserves opaque packed bytes, but its
streamed integration allocates compressed staging, decodes each record through
Metal, copies into NumPy, then copies into the destination slot. Batch decode
is serial. It needs a separately measured direct-slot decoder and scratch-memory
bound before becoming a candidate. Slot capacity remains uncompressed.

Independent process guard: 2 GiB. Boundary snapshots show physical used up to
10,606,280,704 B and process footprint up to 428,262,168 B; these are not samples
of every histogram transient. Swapouts stayed at 4,399,765 pages. Guard exit 0;
exact Qwen restored healthy and warm, lock released, independently verified.
