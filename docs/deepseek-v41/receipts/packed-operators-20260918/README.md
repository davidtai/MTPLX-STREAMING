# Two rejected packed expert operators

Measured source: `013ba48db733659d287d40e7304e1683a0d7e179`.
Both screens use the attested strict MLX allocator from
[the strict-cache receipt](../strict-cache-20260918/README.md), native kernels,
109 persistent and 48 transient slots, and all 206 saved native M6 routes for
layer 34. The replay restores the captured 73 physical residents before growing
to 109. This is an operator workload, not a reproduction of the latest full
prefill's 84-slot cache state. Incremental memory bound: 9 GiB, including 5 GiB
Metal/cache/compiler and 4 GiB host/reader/compiler.

| Candidate | Candidate/control elapsed ratio | Control spread | Decision |
| --- | ---: | ---: | --- |
| Group each physical-bank batch by expert slot; compute inverse permutation on CPU | 1.017239 | 0.629% | Reject: slower |
| Compile the complete native clamped SwiGLU body | 1.001504 | 1.882% | Reject: flat within noise |

Each uses an interleaved native/candidate/native/candidate/native batch. All
206 outputs and physical read counts match in each arm; each warm arm reads
742 records. Neither changes the retained default. No full-model run or new
regression tests followed these rejected screens.

Row ordering peaks at 3,028,505,097 MLX bytes and releases to 8 active bytes;
activation peaks at 3,028,505,101 and releases to 12. Guard whole-machine peaks
are 13,772,292,096 and 13,843,136,512 bytes respectively, with zero compressor
growth. Guard sessions 93608 and 13482 exited 0. Their recorded independent
checks at 07:36:52 and 07:39:44 UTC confirm exact Qwen identity, healthy/idle
service, completed background warmup and a free lock.

The two subdirectories preserve measured helpers, construction proofs,
per-call data, commands, guard logs, reclamation and recovery receipts. The
large packed artifact and isolated library are referenced by identity and are
not copied here. Helpers retain their original measured /tmp paths; refresh
construction pins deliberately before reuse. `SHA256SUMS.json` covers every
other file in this directory.
