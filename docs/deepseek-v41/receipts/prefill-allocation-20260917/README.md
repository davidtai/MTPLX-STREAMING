# Prefill-derived layer capacity screen

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

No full GPU result exists yet. `sha256.json` binds the five CPU evidence files.
