# W24 — DeepSeek-V4.1-Flash decode routing-locality census + decode-streaming levers

Status: **IN PROGRESS** (skeleton committed).

Branch `feat/deepseek-v41-w24` off `feat/deepseek-v41-streaming` @ 5d6dd8ad.
CPU-designed, GPU-measured by the orchestrator inside `gpu_window.sh`.

## Goal
Decode > 20 tok/s on the DSV4.1-Flash native artifact at the 1,024- and
16,384-token prompt shapes, MTP on, KV minimal, expert cache maximal, never past
110 GB. Measured baseline (GPU, 1,024 prompt, greedy, 256 tokens, loader path):
prefill 37 tok/s, TTFT 27 s, **decode 4.79 tok/s** @ 72 GiB planner budget, 4.85
@ 92 GiB — cache capacity is not the lever within one prompt; cold misses
dominate.

## Deliverables
1. `scripts/deepseek_v41/routing_census.py` — CPU census tool. **[skeleton]**
2. `scripts/deepseek_v41/ab_decode_levers.py` — GPU A/B lever harness. **[skeleton]**
3. `tests/test_deepseek_v41_decode_levers.py` — byte-identity lever tests. **[skeleton]**
4. This report — census numbers, cost model to 20 tok/s, ranked levers, A/B commands.

## Census numbers
_TBD — census run pending._

## Cost model to 20 tok/s
_TBD._

## Levers (ranked)
_TBD._

## A/B commands
_TBD._

## Peak RSS
_TBD._
