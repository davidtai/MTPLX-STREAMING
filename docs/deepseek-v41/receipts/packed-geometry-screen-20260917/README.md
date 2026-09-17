# Packed expert geometry screen

All three candidates preserve exact output bytes on the four real layer-20
expert shapes, but none produces a useful latency gain. No candidate was
installed and no full-model rerun was added.

| Rows / unique experts | R8 / SG2 | R4 / SG4 | FP4 float-bit conversion |
| --- | ---: | ---: | ---: |
| 6 / 6 | -8.37% | -0.32% | -15.12% |
| 18 / 3 | -7.57% | -0.82% | -52.78% |
| 36 / 12 | -1.62% | -0.69% | -27.84% |
| 36 / 36 | -2.02% | +0.07% | -19.62% |

Values are latency reductions against interleaved unchanged controls; negative
values are slower. Inputs are synthetic BF16; weights, nonidentity expert/slot
mapping, whole-MLP arithmetic and layouts are real. Source is
041b93f8b58c11d82e0b2ee3ddc79b75f543aa78. The screen does not measure full-model
throughput.

The guarded child passed within its 6 GiB incremental bound. MLX allocator
peak was 928,797,486 bytes, active memory after close was eight bytes, and the
sampled machine peak was 12,877,561,856 bytes. Qwen was restored; independent
verification at 21:02:27 UTC found it healthy, idle and warmed, with the lock
free and no owned child. Sources, samples, command and restoration are archived.
