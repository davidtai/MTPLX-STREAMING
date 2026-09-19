# Uncached verification I/O proxy

Source `d133cb694dc8a2d886ba21ac5d94a2dc05706ec2`. Eight concurrent real
18,800,640-byte expert records per batch; 256 identical batches per sample.
F_NOCACHE enabled, native reader disabled, no MLX import or model allocation.
Reader fanout order: 4, 1, 2, 8, 8, 2, 1, 4. Final-batch hashes match the manifest
in every sample; the bank identity is unchanged. This is I/O evidence, not TPS.

| Fanout | First aggregate GB/s | Second aggregate GB/s |
| --- | ---: | ---: |
| 4 | 13.084 | 13.091 |
| 1 | 13.140 | 13.143 |
| 2 | 10.566 | 13.100 |
| 8 | 13.031 | 12.934 |

No meaningful gain from increasing fanout; keep the current model setting of 4.
The earlier buffered-read fanout receipts do not establish uncached throughput.

250 ms samples: physical peak 13,197,328,384 B; process footprint peak
388,498,800 B. Swapouts unchanged at 4,399,765 pages. Guard exit 0; exact Qwen
restored healthy and warm, lock released. Independent health/lock check followed.
