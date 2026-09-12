# W100 — why the DSpark verify pass showed `prefetch_issued 0`, and what actually fixes it

**Status:** root-cause + fix. Worker branch `w100/dspark-verify-prefetch`, based on the
w95 head `5245a90db` (= `182bb21b9` + the three W95f fixes below), authored CPU-only with
MLX pinned to `mx.cpu`; no GPU touched (a benchmark window holds the Metal lock).

## TL;DR

The window-39 receipt's `dspark.serve_stream_counters.expert_cache.prefetch_issued 0` is
**stale evidence, not a live bug in the verify prefetch path**. Window 39 ran at
`5795f0b36`, which **predates the DSpark-verify prefetch extension** — at that commit the
gate-oracle prefetch was AR-only and returned early for any multi-row (`T>1`) forward, so it
was inert on the `K+1`-row verify. The extension (predict the per-row union under
`MTPLX_DSV41_RUNNER=v2` for `T∈[2,8]`) landed in `182bb21b9` (my base). On the base+w95
foundation the verify path **does** reach `prefetch_experts` and issue (proven on a real
CPU runtime below). What was missing is any way to *prove* verify-phase engagement from the
receipt, and a harness wall misreport. This window adds verify-phase counters and fixes the
wall.

## The receipt evidence is pre-extension

Window 39 code is `5795f0b36` (task prompt; receipts under
`docs/deepseek-v41/receipts/gpu-windows/window-39/`). At that commit the stash gate in
`DecoderLayer._maybe_stash_gate_prefetch` was AR-only:

```
# 5795f0b36 mtplx/models/deepseek_v41.py (via git show):
#   single-token AR decode only: h is [B, T, hc_mult, dim]; T == 1.
if h.ndim != 4 or int(h.shape[1]) != 1:
    return
```

`git show 5795f0b36:mtplx/models/deepseek_v41.py | grep -n '2 <= _T\|VERIFY_K_PER_ROW\|_verify = True'`
returns **nothing** — the `T∈[2,8]` verify branch, `_RUNNER_V2_VERIFY_K_PER_ROW`, and the
per-row union did not exist yet. `git diff 5795f0b36 182bb21b9 -- mtplx/models/deepseek_v41.py`
is the commit that ADDS them ("extend it to the DSpark verify"). So on `5795f0b36` the verify
forward (`T = K+1 = 6`) fell into the `else: return` branch and stashed nothing → the switch's
issue site had no pending → `prefetch_issued 0` for the whole DSpark pass. The AR headline in
the same receipt issued 45,658 because AR is `T=1` and the AR-only path fired.

Net: the receipt's `0` is explained entirely by "no verify extension at that commit," which
is simpler and more certain than either of the two hypotheses the review floated (a
structural verify-path failure, or the byte-budget latch — see below).

## At my base the verify path is sound (code trace)

The verify forward reaches `prefetch_experts` end-to-end on `182bb21b9`:

- The DSpark verify is one target forward over `mx.array([block_ids])` shaped `[1, K+1]`
  (`mtplx/models/deepseek_v41_dspark_decode.py:822`), run under an explicit DECODE phase
  override (`_verify_routing_context` → `expert_routing_phase(RoutingPhase.DECODE)`,
  `mtplx/models/deepseek_v41_dspark_decode.py:190-208`).
- `DeepseekV41Model.__call__` → `_forward_span` iterates `layer(...)` =
  `DecoderLayer.__call__` (`mtplx/models/deepseek_v41.py:3300`), whose eager path calls
  `_maybe_stash_gate_prefetch(h)` (`:2952`). Fused small-stages is OFF in the window-39
  arm (`MTPLX_DSV41_SMALL_STAGES_FUSED = null` in both receipts), so the eager path runs.
- `_maybe_stash_gate_prefetch` (`:2896-2942`): `h` is `[1, 6, hc, dim]`, `_T = 6`,
  `_runner_v2_enabled()` and `2 <= 6 <= _RUNNER_V2_VERIFY_MAX_ROWS(8)` → `_verify=True`;
  it predicts the per-row union (`k_eff = _RUNNER_V2_VERIFY_K_PER_ROW = 8`,
  `margin=_resolve_gate_prefetch_margin()`) and sets `switch._mtplx_gate_prefetch_pending`.
- The switch `_run` consumes it (`mtplx/models/expert_mlx.py:2536`) and, at the verify's
  route exit — all-hit single-barrier (`:3082`) or split single-barrier (`:3230`) — calls
  `_issue_pending_gate_prefetch()` → `_issue_gate_prefetch(runtime, pending)` →
  `runtime.prefetch_experts(next_layer, ids)`.

Because the whole chain is reached, `prefetch_issued` should be `>0` on the base. It was `0`
in window 39 only because the *stash* (bullet 3) did not exist at `5795f0b36`.

## Reproduced on a real CPU runtime (no GPU)

Using the W95 test harness (`tests/test_deepseek_v41_w95_runner_v2.py::_open_runtime`, a
real `ExpertStreamingRuntime` + fake bank) and `HotExpertSwitchGLU`, stashing a prediction
of non-resident experts and running a route through the switch under
`MTPLX_DSV41_RUNNER=v2`:

| route | rows | `prefetch_experts` issued |
|-------|------|---------------------------|
| AR all-hit | 1 | **2** |
| AR miss | 1 | **2** |
| verify all-hit | 6 | **2** |
| verify miss (split single-barrier) | 6 | **0** (single-layer fake artifact) |

The verify **all-hit** branch issues exactly like AR. The verify **miss** case returns 0 in
this fake **only because the fake has a single routed layer** (`spec.routed_layer_indices ==
(1,)`), so the gate-oracle's `next_layer` equals the layer just routed. The split
single-barrier path defers that layer's release holding its lock
(`_DeferredSplitClose`), and `prefetch_experts` does a **non-blocking** acquire of the target
layer's lock and returns 0 when held (`mtplx/expert_runtime.py:4202-4210`,
`note_skipped_lock_held` — the documented "~3827 deferred-split hazard"). In the real model
`next_layer = L+1`, a different lock that is free (prior deferred closes are flushed at each
layer's `mx.eval(indices)` before the next-layer prefetch issues), so the split verify
issues too. AR-miss on the *same* single layer issues (it does not defer), which confirms the
0 is the deferred-split same-layer collision, not a verify-path defect.

## The two review hypotheses, resolved

- **"structural verify-path failure"** — refuted: the chain reaches `prefetch_experts` and
  issues (table above). The only 0 is a single-layer fake artifact absent in the real
  cross-layer model.
- **"byte-budget latch"** — this was a **real latent bug** and I reproduced it: at HEAD
  `reset()` cleared `self.counters` but NOT `speculative_bytes_read`/`demand_bytes_read`, and
  `demand_bytes_read` had a single writer (the reconcile fallback), so after one fallback the
  budget `speculative >= 0.5*demand` latched OFF for the whole process and survived the
  `_cold_reset_expert_streaming` between the AR and DSpark passes. **But it does not explain
  window 39**, because window 39 predates the verify extension entirely, and its exact-0 is
  inconsistent with the latch (the AR pass reached 45,658 issues, which requires
  `demand_bytes_read ≈ 0` throughout AR, i.e. no early trip). This latch is **already fixed on
  my foundation** by the w95 worker's commit `5093fb328` ("make the speculative-byte budget
  real and recovering" — a per-decode-token window reset at the token boundary and at
  `reset()`, plus a real cold-demand-miss denominator). I did not touch the budget.

**Which explanation does the evidence support?** Neither of the review's two — the receipt is
stale (pre-extension). The cell receipt cannot arbitrate the latch directly because the ab
harness filtered the runner/bytes counters out (`_EXPERT_CACHE_KEYS` only); that filtering is
also fixed on my foundation by w95 commit `5245a90db` (routes the `runner`/`gate_prefetch`
blocks to the receipt).

## Foundation (cherry-picked / fast-forwarded from w95)

My branch fast-forwards to the current `w95/barrier-free-resident` head `5245a90db`, which is
`182bb21b9` + three W95f fixes the coordinator flagged, so I build on them rather than
conflict with them:

- `e24f546d9` — drop the `-0.05`-margin `-1` sentinels + dedup the verify union in
  `_issue_gate_prefetch` (the crash the coordinator warned about; `GlobalPrefetchRing._key`
  rejected `-1`).
- `5093fb328` — the byte-budget latch fix (above).
- `5245a90db` — route the `runner`/`gate_prefetch` receipt blocks to the harness + daemon.

## What this window changes

1. **Verify-phase engagement counters** (`prefetch_issued_verify` / `prefetch_committed_verify`):
   the merged `prefetch_issued`/`prefetch_committed` cannot tell AR (`M=1`) from verify
   (`M=K+1`) prefetch, so the next window still could not *prove* the verify engaged. New
   `CacheCounters` fields (`mtplx/expert_streaming.py`), incremented in
   `ExpertStreamingRuntime.prefetch_experts(..., verify=True)` and attributed on commit via a
   `(layer, expert, ticket)` tag set (`mtplx/expert_runtime.py`), surfaced in the `runner`
   receipt block (`_runner_snapshot`) so w95's routing carries them to the cell receipt. The
   verify flag is threaded from the switch's issue site
   (`_issue_pending_gate_prefetch` → `_issue_gate_prefetch(..., verify=)`,
   `mtplx/models/expert_mlx.py`) where the phase and row count are known
   (`phase is DECODE and 2 <= tokens.shape[0] <= 8`). Byte-identity is preserved — `verify`
   is telemetry only, it never changes which ids issue or the gathered math. Env flags are
   read at use (the flag is a call parameter, not an import-time read).

2. **Harness wall fix** (`scripts/deepseek_v41/ab_decode_env_levers.py`): `_generate_dspark`
   timed the whole `dspark_generate` call (re-prefill + decode) as `decode_wall_s`, so window
   39 divided 257 tokens by a 297.36 s wall (`decode_tok_s 0.864`) whose decode phase was
   ~93.4 s (2.75 tok/s). Factored out `_dspark_decode_wall_accounting(...)` (unit-testable):
   `decode_wall_s` now excludes the re-prefill (measured from the prefill→decode boundary the
   `prefill_callback` marks), `decode_tok_s = generated / decode_wall_s`, and the whole-call
   figure is preserved as `pass_wall_s`. The AR headline timing (`_generate`) is untouched.

## Validation

`tests/test_deepseek_v41_w100_verify_prefetch.py` (CPU-pinned MLX, real runtime + fake bank):
(1) a verify-shaped route under v2 issues prefetch and `prefetch_issued_verify` increments
while an AR route leaves it at 0; (2) the switch output is byte-identical with the prefetch
prediction stashed vs not; (3) the wall accounting helper excludes the re-prefill.
