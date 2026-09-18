# Native output projection with one retained weight representation

The native target output route retains packed MXFP8 `wo_a` weights alongside
the BF16 transpose consumed by fused decode. Across 40 layers, the packed
weights and scales occupy 1,384,120,320 bytes. The staged candidate preserves
the original first-use cache materialization, then retires the packed holder
and cache-key tuple and binds the BF16 output callable directly. A weak parent
reference avoids adding an ownership cycle. No steady eligibility branch,
fallback, counter or changed matrix arithmetic is introduced.

The output function's AST matches native after binding the existing weight,
validated group count and equivalent module alias. The real layer-0 operator
loads four native tensors totaling 77,856,768 bytes from shard 3. Query/KV and
other unused attention parameters are neither loaded nor evaluated. Native
RoPE removal, grouped BF16 matmul and native MXFP8 `wo_b` remain unchanged.

| Measurement | Native warm | Candidate warm |
| --- | ---: | ---: |
| MLX active bytes | 145,952,648 | 111,349,640 |
| Peak bytes, including first materialization | 213,061,512 | 213,061,512 |

The exact active reduction is **34,603,008 bytes**, the original packed matrix
and scales. All output bytes agree at M1, M6 and M8, including a cold first use
of the candidate. Eight active bytes remain after cleanup. This establishes
one-layer ownership, not full-model throughput or cache capacity.

Cold peak remains unchanged because materialization initially needs the packed
source. Relative to the retained warm state, the candidate's cold overlap is
101,711,872 bytes versus 67,108,864 bytes for native. A full steady bound must
therefore retain at least one layer's 34,603,008-byte source overlap while
crediting the retired packed weights. Expert-bank growth happens earlier and
receives no projection-retirement credit.

The 4 GiB incremental bound includes 2 GiB for MLX policy plus 2 GiB for Python,
cache/compiler and temporary owners. The source reader uses final MLX buffers,
bounded uncached ranges, validated native shard/tensor metadata and source
identity checks. After the child exits, the controller reclaims 66,387,968
cached source bytes to zero. The guard samples 403,309,816 bytes of process-tree
footprint and 10,487,103,488 bytes of whole-machine use. These are separate
overlapping observations, not a sum.

The first attempt fails during input setup because native `_cos_sin` requires
flat positions. The corrected harness supplies `[S]` positions and native
`[S,32]` cosine/sine tables. No projection comparison ran in the refused
attempt. Its guard exits 1 and restores/releases at 03:51:16 UTC on September
18; independent 03:54:19 verification finds healthy/idle/warmed/free Qwen.

The completed operator is pinned to source `72b0d219e`. Guard session 10513
exits 0, restores exact Qwen identity, health and warmup, and releases at
03:55:19 UTC. Independent verification at 03:59:04 UTC finds healthy, idle,
warmed Qwen, a free lock and no owned child. Both attempts have SHA-256 archives.

The next full candidate retains the original prefill and bank-growth bounds.
Only steady projection storage receives the measured ownership credit, with
the cold source overlap and additional Python metadata charged explicitly.
The best full result remains 12.6731624 TPS; 20 TPS remains unmet.
