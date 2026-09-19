# Paired short native-MTP draft-length screen

Source `8a8803ad86108c3448331a8e880fe2a0b891e809`; pinned 16,384-ID Python
prompt, 64 decode steps plus the prefill token, native model files, pf0, I/O
fanout 4, 71 target expert slots per layer and 48 shared transient slots.
The block-6 head was constructed with an explicit experimental size override;
the block-5 control kept the released size. Both runs used the same benchmark
generation code and compared their 65 IDs with the archived AR reference.
**Block 6 is rejected for this slice**: it needed the same 14 verify cycles and
read 12.35% more expert bytes, yielding 8.59% lower decode TPS. This does not
establish full 1,024-token performance for either arm.

| 64-step diagnostic | Block 5 control | Block 6 experiment |
|---|---:|---:|
| Decode wall seconds | 9.242825 | 10.111764 |
| Decode TPS | 6.924290 | 6.329262 |
| Verify cycles | 14 | 14 |
| Output tokens per cycle | 4.642857 | 4.642857 |
| Draft acceptance | 87.9310% | 87.5000% |
| Decode expert records read | 4,460 | 5,011 |
| Decode SSD bytes read | 83,850,854,400 | 94,210,007,040 |
| Full-pass MLX peak, bytes | 86,633,864,412 | 86,633,865,612 |
| Sampled system physical peak during child, bytes | 99,251,191,808 | 98,581,299,200 |

Both generated streams are identical to each other and to the first 65 IDs of
the archived AR run; no divergence classification was needed. The block-6
head changed earlier draft suggestions, so matching cycle count and acceptance
were measured rather than assumed from block 5. The different physical peaks
also reflect distinct guard baselines (10.2689 versus 10.5936 decimal GB),
file-cache state and 250 ms sampling. They are not evidence that block 6 has a
lower steady memory requirement.

The admission bounds used the previous full-workload MLX peak at 72 slots,
charged a separate 3 GiB for the new M7 attention/graph shape, and credited no
short-run saving. Physical bounds were 105,169,989,004 B for block 5 and
105,494,689,004 B for block 6 under the 110,000,000,000 B ceiling. The
shared target transient pool had 48 records, enough for M7's maximum 42
top-6 assignments. Both runs left swapouts unchanged at 4,399,765.

The wrappers intercepted the AR entrypoint after normal model loading to run
one diagnostic MTP pass and exited through the runner's existing cleanup. They
did not fabricate an AR benchmark result or change production code. Each
guard exited 0, automatically reclaimed clean file cache on Qwen shutdown,
restored the exact Qwen server, completed warmup and released the GPU lock.
Fresh health and nonblocking lock checks passed afterward.

Raw MTP results, 250 ms OS traces, admission bounds, guard logs and exact
diagnostic wrappers are archived here. The prior complete full-workload
champion remains [8.938810168 TPS](../dspark-layer-prefill-20260913/README.md).
