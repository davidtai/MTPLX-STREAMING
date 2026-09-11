# W28 — Per-layer routing-barrier overlap-fill (KERNEL_LEDGER K1, Rank 1)

Status: implemented behind a switch (default off), byte-identity proven on the
tiny synthetic model, GPU A/B arm ready for the orchestrator. Author: Opus 4.8
worker (feat/deepseek-v41-w28, from integration @ 373ec9f9). CPU-only; no GPU
lock, no real-artifact load. mlx 0.32.2.

## 0. Headline

The DSV4.1 streamed forward forces **one `mx.eval(indices)` routing barrier per
streamed layer** (40 routed text layers → ~40/token, matching the ledger), each
a device→host round-trip (`.tolist()` the ids to issue the SSD/DRAM gather) with
the GPU idle during it → the ~25 tok/s barrier ceiling (§0 KERNEL_LEDGER).

**Root cause found:** the existing shared-overlap seam (`run_switch_with_shared_
overlap`, used by hy3 and glm) was **never wired for the DSV4.1 MoE.** W11's
`MoE.__call__` called `self.switch_mlp(xf, indices)` through the plain `__call__`
(`shared_work=None`) and then computed `self.shared_experts(xf)` **after** the
routed gather — so the shared expert, which depends only on `xf` and never on the
routed indices, could not fill the barrier's GPU-idle window.

**Fix (behind `MTPLX_DSV41_SHARED_OVERLAP`, default off):** `MoE.__call__` now
hands the shared expert to the streamed switch as `shared_work`, and the switch's
existing hoist (already proven bitwise-identical for hy3) also fires under the new
flag, so the shared branch is `async_eval`-dispatched **before** the routing
barrier. The routing barrier is **preserved** (a3b measured −3.27% when the
decision sync was *deleted* — this is overlap-fill, not deletion) and **no
blocking sync is added**. Pure execution reorder → byte-identical greedy tokens.

The measured GPU throughput delta is the orchestrator's GPU-window job (arm in
§5); this worker delivers the mechanism, the sync census, the byte-identity
proof, and the arm.

## 1. Sync census (measured, tiny synthetic streamed switch + fake bank)

Instrumented one `HotExpertSwitchGLU._run` (one streamed layer, single decode
token) over the `_BankOverlap*` doubles (`tests/test_streamed_models.py`): 3
assignments = 1 resident hit + 2 SSD-miss waves. Counters wrap `mx.eval`
(blocking host sync), `mx.async_eval` (non-blocking dispatch) and `.tolist`
(device→host); the routing barrier is counted via the `hot.eval_indices` stage
bracket. Locked in `tests/test_deepseek_v41_sync_overlap.py`.

| mode | deferred_pin_release | blocking `mx.eval` | `async_eval` | `.tolist` | routing barriers |
|---|---|---:|---:|---:|---:|
| control (shipped) | True | 4 | 0 | 1 | 1 |
| shared_overlap (hoist) | True | 4 | **1** | 1 | 1 |
| control (shipped) | False (fenced) | 4 | 0 | 1 | 1 |
| shared_overlap (hoist) | False (fenced) | 4 | **1** | 1 | 1 |

Reading of the 4 blocking evals in this fixture: **1 routing barrier**
(`mx.eval(indices)`) + **3 wave-completion fences** (1 hit + 2 miss waves,
`force_sync=True`; governed by `MTPLX_EXPERT_SLOT_FENCES` / `deferred_pin_release`,
a slot-lifecycle mechanism this lever does not touch). The real top-6 model has a
different wave count, but the invariant this census proves holds for every shape:

- **exactly one routing barrier per streamed layer** (asserted); the lever never
  adds a second and never deletes it;
- the lever adds the shared expert as **one non-blocking `async_eval`** and **zero
  blocking syncs** (`overlap eval == control eval`), and with the hoist it is
  submitted **before** the barrier — filling the barrier's GPU-idle round-trip;
- the shared expert is computed **exactly once** (no double-compute);
- exactly one device→host route materialization (`.tolist`) either way.

Per decoded token (40 routed text layers): **~40 routing barriers** + 1 sampler
argmax host sync (`.item()`) + the trunk's per-layer attention/norm evals —
matching the ledger's "~40 + the sampler sync". Shipped, each of the 40 barriers
exposes its ~1 ms p50 device→host round-trip as GPU-idle time. The lever
dispatches the shared expert's GEMV work (w1/w2/w3; ~0.44 ms recovered on the
analogous hy3 shared-hoist) into each of those windows.

Caution recorded from the census: **without the hoist**, in strict fenced mode
(`deferred_pin_release=False`) the shared branch is dispatched with a blocking
`mx.eval(shared)` in the miss window → a 5th blocking eval. The hoist (armed by
the same flag) is what makes single-token AR add zero blocking syncs, so the
lever is designed to run with the promoted `deferred_pin_release=True` decode
default (where the miss-window shared dispatch is `async_eval` anyway).

## 2. Root cause detail

`mtplx/models/deepseek_v41_moe.py::MoE.__call__` (shipped):

```python
routed = self.switch_mlp(xf, indices)              # plain __call__ -> shared_work=None
y = (routed.astype(mx.float32) * weights[..., None]).sum(axis=-2)
y = y + self.shared_experts(xf).astype(mx.float32) # shared AFTER routed gather
```

`self.switch_mlp(xf, indices)` runs the routed gather (barrier → I/O → routed
compute) to completion, then the shared expert runs — serialised, its GPU work
unavailable to hide the barrier. hy3 (`hy3_mlx.py:939`) and glm
(`glm52_mlx.py:358`) already route this through `run_switch_with_shared_overlap`;
DSV4.1 did not. The `HotExpertSwitchGLU` machinery to overlap (`run_with_shared_
overlap`, the `shared_work` seam, the `MTPLX_HY3_SHARED_HOIST` hoist, the
miss-window shared dispatch) all existed and was simply not reached for DSV4.1.

## 3. Implementation (behind `MTPLX_DSV41_SHARED_OVERLAP`, default off)

Two edits, both pure execution reorders:

1. **`mtplx/models/deepseek_v41_moe.py`** (call-order only): when the flag is set,
   `MoE.__call__` hands the shared expert to the switch as
   `lambda: self.shared_experts(xf).astype(mx.float32)` via
   `run_switch_with_shared_overlap(self.switch_mlp, xf, indices, shared_work)`,
   then combines `y = (routed.f32 * weights).sum(-2); y = y + shared`. Flag off →
   the shipped path verbatim. A switch without `run_with_shared_overlap` (the
   resident `SwitchGLU` / test default) falls back to `switch_mlp(x, indices),
   shared_work()` — identical arithmetic — so only the streamed path overlaps.

2. **`mtplx/models/expert_mlx.py`** (`HotExpertSwitchGLU._run` hoist gate): the
   existing shared-branch hoist now also fires when `MTPLX_DSV41_SHARED_OVERLAP
   == "1"` (alongside `MTPLX_HY3_SHARED_HOIST`), a distinct env name so DSV4.1 can
   be A/B'd without arming hy3/glm. The hoist `async_eval`s the shared branch
   before `mx.eval(indices)`; the late miss-window dispatch is then skipped
   (`shared is not None`), so no double-compute. hy3/glm never set the DSV4.1 flag
   → their paths are unchanged.

The routed output is independent of `shared_work` in `_run` (the waves, gather,
concatenate and reshape never read `shared`), so `routed` is identical whether
`shared_work` is `None` or supplied; `shared` is exactly the same
`shared_experts(xf).astype(f32)` array. Byte-identity therefore holds by
construction, and is asserted below.

Exactly one routing sync per layer is preserved; the sync is not deleted
(a3b: [[a3b-decode-roundtrip-is-the-lever]] measured deletion +3.27% slower —
MLX async-submission backpressure conserves the block regardless of the Python
sync, so the win is utilization/overlap, priced by the GPU A/B, not the roofline).

## 4. Byte-identical proof (`tests/test_deepseek_v41_sync_overlap.py`, 8 cases, green)

- `test_exactly_one_routing_barrier_per_streamed_layer` — 1 `hot.eval_indices`
  barrier per `_run` in every mode.
- `test_overlap_hoist_adds_no_blocking_sync[deferred True/False]` — overlap adds
  0 blocking `mx.eval`, +1 `async_eval` submitted before the barrier, shared
  computed once, 1 `.tolist`.
- `test_streamed_switch_routed_and_shared_bitwise_identical[deferred True/False]`
  — `HotExpertSwitchGLU` routed **and** shared arrays bitwise-equal, lever off vs
  on (shared branch is a non-trivial function of `x`).
- `test_moe_call_order_bitwise_identical_off_vs_on[n=1, n=4]` — W11 `MoE.__call__`
  output bitwise-equal off vs on for AR (n=1) and the **K=3 MTP verify row-batch
  shape (n=4)**.
- `test_served_ar_tokens_and_logits_bitwise_identical` — end-to-end over the tiny
  real model (every DecoderLayer is a W11 `MoE`, deepseek_v41.py:633): greedy
  tokens **and** per-step logits bitwise-equal off vs on across prefill + 8 AR
  decode steps.

W23 (MTP verify) is merged into the integration branch; the verify seam's MoE
byte-identity is covered by the n=4 case (one verify forward feeds 4 query rows
through each layer's MoE). A full streamed K=3 verify e2e needs the 269 GiB bank
and belongs in the GPU window (§5), not a CPU unit test.

Regression (no hy3/glm/expert drift from the shared expert_mlx.py edit):
`tests/test_shared_hoist.py` (3) + `tests/test_expert_overlap_split.py` (9)
green; `tests/models/test_deepseek_v41_moe.py` 2 pass / 4 artifact-skip;
`test_streamed_models.py::test_hy3_shared_mlp_uses_streamed_overlap_hook` and
`::test_glm_shared_mlp_uses_streamed_overlap_hook` green (hy3/glm still route
through the shared-overlap seam byte-identically).

Pre-existing (NOT this worker): four `test_streamed_models.py` cases fail on the
clean integration base (measured detached at 373ec9f9 and at the current tip
2ed30c9c3, both without any W28 change) — stale test doubles: `fake_q4()` lacks
the `swiglu_limit`/`codec` kwargs `_run_q4_expert`/`_run_component_bank_q4` now
pass, and `_OverlapPending` lacks `.abort`. My `expert_mlx.py` diff touches
neither call site (it is only the flag-gated hoist `or`), so these are not
attributable to W28 and are outside the W28 allowlist to fix
(`test_streamed_decode_evaluates_shared_work_before_waiting_for_misses`,
`test_component_bank_overlaps_hit_and_shared_work_with_incremental_misses`,
`test_128k_prefill_preserves_bounded_routed_then_shared_order`,
`test_component_bank_all_hit_decode_preserves_route_waves_counters_and_shared_order`).

## 5. GPU A/B arm for the orchestrator

`scripts/deepseek_v41/ab_decode_levers.py` (feat/deepseek-v41-w24's harness had
not landed on the integration branch, so this is the small self-contained arm the
task specifies; fold the `shared_overlap` preset into w24's `ARM_PRESETS` when it
merges). Run inside `scripts/deepseek_v41/gpu_window.sh`:

```
ab_decode_levers.py --arms control shared_overlap \
    --context-tokens 1024 --decode-tokens 256 --syncs 16 --out <receipt.jsonl>
```

`control` = lever off (shipped); `shared_overlap` = `MTPLX_DSV41_SHARED_OVERLAP=1`.
Per arm it records: prefill tok/s, TTFT, **decode tok/s**, wall, **peak GB**, the
decoded token-id **sha256** (byte-identity is a recorded fact — a differing sha
FAILS the arm), and with `--syncs N` a probe pass reporting **routing barriers per
decoded token** (`hot.eval_indices`, expected ~40) plus any **GPU-overlap
telemetry** the runtime exposes (`overlap_gpu_dispatch_ns` / `overlap_exposed_
wait_ns` → idle share; populates only with `overlap_miss_reads` armed, else
reported as None rather than fabricated). Reuses `bench_standard_shape.py`'s
loader + cell harness so numbers are apples-to-apples with the standard-shape
receipts. CPU-safe at import / `--help`.

## 6. Risks / caveats

- **Value is A/B-pending, not asserted.** Per the a3b caution, overlap-fill can
  fail to pay if the fill does not actually reduce exposed GPU-idle time. This
  worker proves the mechanism is exact and adds no blocking sync; whether the
  filled window nets decode tok/s (and by how much, vs the ~25→~15–20 ms/token
  the ledger models) is the GPU A/B's call. Default off until measured.
- **Pair with `deferred_pin_release=True`** (the promoted decode default): in
  strict fenced mode, multi-row verify (n=4, hoist inactive since it requires
  n==1) dispatches the shared via a blocking `mx.eval(shared)` in the miss window
  — correct but adds one blocking sync/layer. Single-token AR is clean in both
  modes (the hoist async-dispatches). No change to shipped defaults is made here.
- **Shared-expert cost must exceed the barrier window to fully hide it.** The
  shared GEMV is ~0.44 ms (hy3 measurement); a barrier is ~1 ms p50, so one
  shared expert cannot fully hide one barrier. The larger post-streaming gain is
  the shared no longer serialising *after* the routed gather (its ~0.44 ms leaves
  the critical path). Additional route-independent prologue work (next layer's
  HC-mix/attn-norm) is the further K1 lever, not attempted here (scope: shared
  overlap only).
- **Not wired for MTP-graft residents.** The lever lives in W11's text MoE; the
  DSpark MTP stage residents (if they use a different MoE module) are out of scope
  and unaffected.
