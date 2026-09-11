# W21 report — DeepSeek-V4.1-Flash mxfp4 serve profile + text-only planner pricing

Status: IN PROGRESS (skeleton committed first to survive session restarts).

Scope: make `mtplx serve --model ~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4`
resolve a correct, well-tuned config with NO flags on the MTPLX runner, price the
TEXT-ONLY residents in the planner, add the mxfp4 selfcheck signature, and document
the served-path measurement commands.

Sections to fill:
1. Resolved default config table (with reasons).
2. Planner table (residents, engram, KV, reserve, cache, slots/layer) at 82 GiB, text-only pricing.
3. Served-bench command list.
4. What remains flagged.
