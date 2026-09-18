# Prefill-derived layer capacity screen

**Full result: 12.5526155 TPS / 81.4969600 seconds**, below the prior best
12.6731624 TPS / 80.7217623 seconds. All 1,024 native output tokens match, with
206 cycles and the same authorized native-AR index-297 tie classification.
The fresh 11.050860544 GB baseline admitted 4,000 total slots, assigned 84..128
per layer. The prior best used 4,080 uniform slots. No TPS promotion is claimed.

The CPU screen reduces policy demand misses from 36,421 to 35,631 (2.17%) at
the same 3,960 total persistent slots. It assigns 84 through 128 slots per
layer using only the completed prompt's per-expert routing counts. The second
half of the saved decode trace improves from 16,187 to 15,636 misses (3.41%).
This is not a physical-read or throughput result; 20 TPS remains unmet.

All arms replay the same 206 native M6 cycles, with the same captured 73
residents per layer and empty added slots. The installed transition-window
policy receives each recorded route directly. No decode rows select the
capacities. This seed is a CPU comparison control, not a claim to reproduce
the latest prefill-84 cache. Physical transient reuse is outside this screen.

The original screen allowed up to 192 slots and selected at most 140; it
produced 35,586 misses. The staged candidate limits each layer to 128 so its
largest one-component resize can be bounded before loading the model. An
independent-row M6 coverage score chooses exactly the same vector, so it was
not replayed twice. MLX imports were blocked throughout.

The staged full candidate is `/tmp/dsv41-prefill-allocation-20260917/full`.
Its total slot count comes from fresh 110 GB admission. At the existing timed
prefill/decode transition, a heap chooses marginal frequency coverage from
the actual prefill counts. The vector then remains fixed throughout decode.
Existing indices, stored experts, read order, kernels and target arithmetic
remain authoritative. Each bank only grows; layers assigned 84 skip resizing.
The maximum 128-slot copy is charged up front, together with 16 MiB for
phase-only selection metadata and the existing plane/KV/host/cache margins.
Exact per-layer capacities must appear in both physical ownership and reports.

The full run read 35,097 records / 621,031,587,840 bytes, versus the prior
best's 35,092 records / 620,943,114,240 bytes. The allocation therefore achieved
nearly the same traffic with 80 fewer slots in this comparison, while its
48.6853-second read union was slower than the prior 47.9346 seconds. The
3.2473-second phase transition is included in decode wall time. Different
capacity and background conditions prevent an isolated throughput attribution.

The pre-admitted physical bound was 109,588,601,064 bytes. Measured allocator
peak was 92,628,854,244 bytes; internal process footprint 94,404,245,928 bytes;
internal machine peak 106,146,136,064 bytes. The guard separately sampled
94,410,766,808 process bytes and 106,135,830,528 machine bytes. These measures
remain separate; both machine samplers stayed below the 110 GB ceiling.
The reports carry all 40 capacities and label the scalar 100 as their uniform
equivalent, not a capacity every layer owns.

A read-only pre-shutdown scan found 36,051,369,984 cached bytes in Qwen model
files covered by automatic reclamation. The guarded workload subsequently
admitted its fresh baseline without another privileged purge. Guard session
28436 exited 0; Qwen identity, health and background warmup were restored and
the lock released at 01:19:26 UTC on September 18. Independent checks passed.
No additional tests were added after the full result fell short of the best.

`sha256.json` binds five CPU evidence files. `full/sha256.json` separately binds
the complete result, static admission checks, exact harness and lifecycle.
