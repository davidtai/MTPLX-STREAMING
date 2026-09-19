# Native DSpark block 6/7 feasibility

Source inspected: `1f3b9bca7ae5aab6a4bf30eb969b1f6dd0368711`, with the installed `mlx_lm` source identified in `census.json`. Source/header review only; no implementation, model run or performance claim.

**Construction is weight-shape compatible. Start with block 6 / verify M7; block 7 / verify M8 needs an additional numerical boundary check.**

| Geometry, batch 1 | Control | First extension | Further extension |
|---|---:|---:|---:|
| Native draft block size / requested depth | 5 | 6 | 7 |
| Maximum target verify rows | 6 | 7 | 8 |
| Maximum target expert assignments, top-6 | 36 | 42 | 48 |
| Native MTP assignments per stage, top-3 | 15 | 18 | 21 |
| Shared target transient record capacity | 48 | 48 | 48 |

## Compatibility and boundaries

- `DSparkHead` and all three `DSparkBlock` instances copy `dspark_block_size` from construction args (`deepseek_v41_dspark.py:629,848`). Their parameter dimensions depend on hidden/vocab/Markov rank, not block length (`:375,394,647`). Set the resolved argument before constructing the head and every stage. Changing only `speculative_depth` still clamps to the old head block size (`deepseek_v41_dspark_decode.py:1084`); changing only the head field leaves stage embedding/output loops inconsistent. Preserve original artifact files and record the explicit experimental override.
- Embedding/noise, RoPE positions, attention masks, Markov loops and confidence output derive their lengths from the block (`dspark.py:670,568,590,803,820`). The 7-row draft stays within the draft compile cap 32 and the HC small-row band 7. Native expert gathers remain unsorted: 21 assignments are below installed `SwitchGLU`'s sorting threshold 64. Fixed-shape compiled functions can trace the new inputs; their shared Python cache keys do not imply reuse of a five-row execution graph. Cold compilation/workspace still needs measurement.
- **Adding a draft row changes the earlier draft predictions too.** The draft attention mask includes every block row (`dspark.py:590`), so an extra noise/KV position affects all earlier queries. There is no inherited acceptance claim for the first five positions. This changes the draft approximation; target verification, correction and rollback must remain unchanged.
- Current `cell16k_ring_v2_draft_attn_pf0` pins fused projections/selected keys on, K29 core off, and target `ATTN_COMPILE=1`; HC_COMPILE and SMALL_STAGES_FUSED are unset/off (`ab_decode_env_levers.py:1530`). Both M7 and M8 retain these route choices: fused projections admit <=8 (`deepseek_v41.py:2426`), and streamed single-barrier verification admits 2..8 (`expert_mlx.py:2989`). At M7, 42 assignments fit pool48; M8 reaches its 48-assignment maximum. Keep the current layout, top-k and live capacity checks; no larger pool is implied.
- **M8 is outside the existing K22 test exactness band.** `test_deepseek_v41_attn_compile.py:28,52` documents reassociation at >=8; production `_ATTN_COMPILE_MAX_ROWS` is 32. The target gate prefix and coefficient combine use that same resolver (`deepseek_v41_moe.py:85,204`), so M7/M8 compile without a route change. M7 is within the documented <=7 band, but the direct K22 verify test uses M4 (`test_deepseek_v41_attn_compile.py:155`), not native M7. HC and K35 have actual tiny-model M7 equality tests (`test_deepseek_v41_hc_compile.py:383`, `test_deepseek_v41_small_stages_fused.py:387`); those are supporting evidence, not a substitute for current-preset/native-shape target validation. Do not raise their seven-row caps for M8.
- Ring defaults cover verify8 (`deepseek_v41_cache.py:378`); rejected-tail trimming accepts arbitrary valid counts and checks every resulting offset (`cache_state.py:4567`). MTP windows are updated only from the committed target prefix (`dspark_decode.py:1211,1222`), not speculative drafts. With prompt16384/steps1023, conservative admission is 17414 tokens for depth6 or 17415 for depth7, below maxKV17664. Retain the established fixed-storage reservation and the refusal of the unvalidated KV_BOUNDED lane. Exercise ring wrap and partial compressor groups; do not infer rollback correctness only from final token IDs.

## Smallest meaningful screen, not implemented

1. Extend the existing tiny three-stage fixture at construction to block6, then7: unchanged parameter names/shapes, eager/compiled draft shapes and values, target all-accept plus first/last rejection, next-step logits and complete KV state after rollback across ring/compressor boundaries. Include output-limit/EOS tails. Use the existing decode/compile fixtures; avoid a broad new suite.
2. Reuse the bounded native head-only probe with identical real weights and a recomputed T6/T7 activation/compile allowance. Measure cold and warm draft peaks and latency against T5. Add a bounded native-dimension target M7 check of gate indices/coefficients, selected attention, streamed gather and rollback; M8 needs its own check before block7 advances. Permit only justified target tie-break differences, not unexplained numerical/state errors. Head-only synthetic inputs establish shape/peak behavior, not acceptance.
3. After those bounds pass, a short paired block5/block6 run on the actual 16K Python context (32–64 output tokens, identical target settings and cache capacity) is the smallest useful acceptance screen. Use existing per-depth acceptance/cycle timing and first-divergence evidence; no production counters. Block7 follows only after M8 validation and useful block6 evidence. A short screen cannot establish the full 1,024-token speed target.

Unchanged weights mean unchanged resident payload, not unchanged peak memory: extra draft/head rows, target verify rows, graph specialization and rollback temporaries require a fresh bound. Existing depth5 wrappers deliberately reject these geometries and must not be reused as approval. **Longer-block acceptance, memory peaks and net speed benefit remain unmeasured.**
