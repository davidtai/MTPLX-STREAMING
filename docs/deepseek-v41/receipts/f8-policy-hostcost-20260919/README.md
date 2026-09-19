# F8 — transition-window decode policy host-cost reduction

**Purpose.** Cut the main-thread host cost of the DeepSeek-V4.1 Q4 decode
transition-window cache policy in `mtplx/expert_streaming.py`
(`LayerExpertSlotBank`), which runs on the main thread BEFORE any SSD read is
submitted (SSD and GPU idle during it). **Decision-identical**: every admitted
set, victim slot, tie-break and float32 score is byte-for-byte unchanged; proven
against a verbatim pre-optimisation oracle over the real route trace and by
randomized property tests (see *Identity proof*). **No throughput claim, no
production code path re-routed, no GPU** — this is a host-side, CPU-only screen.

## Scope of the change (what got faster, all in `expert_streaming.py`)

1. `_transition_window_scores` no longer rebuilds `last_used` with
   `np.fromiter(history.last_used for history in self._history)` over 384 Python
   objects on every routed call; it reads a maintained int64 mirror
   `_last_used_arr`.
2. `_transition_window_admissions` ranks candidates with one `np.lexsort`
   instead of `sorted(..., key=lambda: (score, last_used, prefill_route_freq,
   -expert), reverse=True)` — eliminating ~100 Python `_transition_window_rank`
   calls (each a numpy-scalar extract + `dict.get`) per routed call. The victim
   sort (≈one entry per admitted miss) stays the exact Python sort.
3. The two slot scans (`evictable` + `empty_slots`) are one slot-order pass.
4. `_prefill_route_freq` is now a property whose setter rebuilds an int64 mirror
   `_prefill_freq_arr` (used as the lexsort tie-break key); it is refreshed at the
   in-place `.clear()`/`.update()` sites too.

The mirrors are kept in lockstep with the Python objects at every write site
(`_touch_decode`, decode-hit refresh, both transaction rollbacks, `reset`,
`prepare_prefill_seed`, restore/reopen via the property). `_observe_transition_window`,
`_reset_transition_window`, `_rollback_transition_window` and the score
arithmetic are **unchanged** (float32 summation order preserved).

`_pool_touch`, `_validate_experts`/`_integer`, `_touch_decode`/`_score` and
`_observe_transition_window` are shared / arithmetic-locked and were left as-is;
they are the bulk of the remaining per-call cost (see the after-profile).

## Method

- **CPU-only, MLX hard-blocked** by a meta-path finder in both the harness and
  the identity test (the policy layer is pure Python/numpy; `expert_streaming.py`
  imports no MLX). No GPU, no Metal, no lock.
- **Workload**: the real M6 verify route trace (206 cycles × 40 layers = 8,240
  routed decode calls, route width 36 / unique-mean 24.24), replayed through the
  real `LayerExpertSlotBank` at **111 persistent + 48 transient**,
  cache_policy `transition-window`, `single_pool` — i.e. the f1-overlap-sim
  control config. Loader/replay/`restore`/`make_bank` copied minimally from
  `.worktrees/dsv41-f1-overlap-sim/scripts/deepseek_v41/overlap_schedule_sim.py`
  (branch f1/overlap-sim).
- **Two call paths**: `plan()` (what the published anchors use) and
  `plan_transaction()` (what the runtime's `_plan_route_transaction` actually
  calls; committed each token). `observe_route`/`begin_split_route`/
  `_plan_route_transaction` are in `expert_runtime.py` (out of scope, MLX).
- **Wall-clock**: fresh banks each rep (state mutates), time only the replay,
  report **best-of-21** (min = least OS interference); `× 7,920` (40 × 198) for
  seconds/run. **cProfile** used only for per-function attribution (it inflates
  the call-heavy path ~2×; the wall-clock is the real number).
- All commands run under `nice -n 19`, `pytest` without `-n auto`.

## Inputs (sha256)

| file | sha256 |
| --- | --- |
| `docs/deepseek-v41/receipts/mtp-verify-routes-20260913/mtp-verify-routes-16k-1024-v2.json.gz` | `07b4b720bae831421bbbab6d4cb770ade577096c1e85733ad949350edf66d0da` |
| `docs/deepseek-v41/receipts/memory-budget-110/replay_route_policies.py` (restore helper) | `7be3475934f5d2b6049d046afb30c64024aab595225bc0d8f28fd49fc03fdccc` |
| `tests/_f8_expert_streaming_oracle.py` (verbatim PRE-change oracle) | `de1771923bdaaf275f0ea834dbb1407336c288724b088a811a61e235f0571c39` |
| `mtplx/expert_streaming.py` (POST-change) | `32d8586ab1a37f6aae7743631e887271e99c29ae9be9b5282a4778a1c36e0854` |
| `scripts/deepseek_v41/f8_policy_hostcost_bench.py` | `ed74d88cdfdd1a04cb49082466186a55d32dea7376b65ca55721923cf7de548e` |
| `tests/test_f8_policy_hostcost_identity.py` | `396b0d2df397804c54631b6dd47a503b7fe3e43a967a5abb579d9a8a4ec22561` |

Pre-change `expert_streaming.py` == the oracle sha (`de177192…`). Base commit
`2026906b0`.

## Exact commands

```sh
VENV=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3   # 3.12.13, numpy 2.4.4
cd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f8-hostpath

# Profile (before = oracle, after = package); path in {plan, transaction}
nice -n 19 "$VENV" scripts/deepseek_v41/f8_policy_hostcost_bench.py profile --impl oracle  --path plan
nice -n 19 "$VENV" scripts/deepseek_v41/f8_policy_hostcost_bench.py profile --impl package --path plan

# Wall-clock microbench, best-of-21
nice -n 19 "$VENV" scripts/deepseek_v41/f8_policy_hostcost_bench.py bench --impl oracle  --path plan        --reps 21
nice -n 19 "$VENV" scripts/deepseek_v41/f8_policy_hostcost_bench.py bench --impl package --path plan        --reps 21
nice -n 19 "$VENV" scripts/deepseek_v41/f8_policy_hostcost_bench.py bench --impl oracle  --path transaction --reps 21
nice -n 19 "$VENV" scripts/deepseek_v41/f8_policy_hostcost_bench.py bench --impl package --path transaction --reps 21

# Identity + property tests (isolated; MLX blocked at module import)
nice -n 19 "$VENV" -m pytest tests/test_f8_policy_hostcost_identity.py -q

# Existing CPU-only expert_streaming test modules (mlx-free; the mlx-importing
# bank tests are skipped to honour the no-GPU rule)
nice -n 19 "$VENV" -m pytest tests/test_expert_streaming.py \
  tests/test_expert_streaming_models.py tests/test_bank_history_policy.py \
  tests/test_expert_admission.py tests/test_expert_hot_path_invariants.py \
  tests/test_expert_profiles.py tests/test_expert_io_metrics.py \
  tests/test_analyze_expert_route_trace.py -q
```

## Results — wall-clock (best-of-21, `nice -n 19`; per-run = per-call × 7,920)

Metric definitions: **per_layer_call_us** = min replay wall-time / 8,240 calls,
in microseconds. **per_run_s** = per_layer_call_us × 7,920 / 1e6. **demand
records** = Σ `len(plan.misses)` over the replay (identity anchor).

| path | impl | per_layer_call µs | per_run s (×7920) | demand records |
| --- | --- | ---: | ---: | ---: |
| `plan` | oracle (before) | 85.245 | 0.6751 | 31,636 |
| `plan` | package (after) | **70.836** | **0.5610** | 31,636 |
| `plan` | **delta** | **−14.41 µs (1.20×)** | **−0.1141 s** | identical |
| `plan_transaction` (runtime path) | oracle (before) | 102.024 | 0.8080 | 31,636 |
| `plan_transaction` | package (after) | **88.150** | **0.6981** | 31,636 |
| `plan_transaction` | **delta** | **−13.87 µs (1.16×)** | **−0.1099 s** | identical |

Run-to-run variance is ~10%; a same-process back-to-back A/B (`bench … both`)
gave 1.23–1.25×. Headline uses the more conservative separate-process best-of-21.

### Measured total per call vs the ~0.19 ms receipt figure

The pre-change pure `plan()` decode policy is **~0.085 ms/call** wall-clock
(`plan_transaction` **~0.102 ms**), i.e. **below** the receipt's ~0.19 ms
"cache policy transaction" figure. Under cProfile instrumentation the same
`plan()` measures **0.189 ms/call** (cum) — matching ~0.19 ms. So the receipt's
0.19 ms reflects instrumentation / a busier host and (for the runtime) work
beyond `plan_transaction` (the separate ~0.15 ms "route preparation" lives in
`expert_runtime.py`, out of scope and MLX-bound). The dominant **in-scope** cost
is the policy itself, not plumbing, so the optimization targets it directly.

## Results — cProfile per-layer-call attribution (`--path plan`)

cProfile (2× inflated vs wall-clock) — for RELATIVE attribution only.

BEFORE (oracle), cumulative µs/layer-call:

```
function                                    ncalls   tot_us/call   cum_us/call
plan                                          8240        20.683       188.652
_transition_window_admissions                 8240        15.066        96.088
_transition_window_scores                     7658        12.130        40.340
<lambda> (candidate sort key, L1410)        721409         7.640        25.171
_pool_touch                                 168137         2.772        20.100
_transition_window_rank                     751514        13.346        18.276
_observe_transition_window                    8240         8.775        14.568
<genexpr> (last_used feed, L1347)           2948330         9.978         9.978
```

AFTER (package):

```
function                                    ncalls   tot_us/call   cum_us/call
plan                                          8240        20.169       141.937
_transition_window_admissions                 8240        22.129        50.842   (lexsort now inline)
_pool_touch                                 168137         2.679        19.802   (unchanged, shared)
_transition_window_scores                     7658        11.170        16.420   (last_used fromiter gone)
_validate_experts                             8240         2.598        15.301   (unchanged, shared)
_observe_transition_window                    8240         8.527        14.163   (unchanged)
_transition_window_rank                      30105         …            2.5xx    (victims only; was 751,514 calls)
```

`plan` cum 188.7→141.9 µs (−25%); `_transition_window_scores` cum 40.3→16.4;
`_transition_window_rank` calls 751,514→30,105; the 2.95M-call last_used
generator and the 721k-call candidate-sort lambda are eliminated. Remaining cost
is shared code (`_pool_touch`, `_validate_experts`/`_touch_decode`/`_score`) and
the arithmetic-locked `_observe_transition_window`.

## Identity proof

`tests/test_f8_policy_hostcost_identity.py` — **8 passed in ~5.4 s** (isolated,
`nice -n 19`, MLX blocked). Drives the verbatim pre-change oracle and the live
package in lockstep:

- **Full 206×40 trace @ 111+48 transition-window**: per-layer-call identical
  `RoutePlan` (hits, misses, per-expert slots, loads incl. persistent/transient,
  victim-slot evictions, counters), identical bank state, and **bit-exact float32
  scores** every call; total demand records 31,636 == 31,636.
- **Published anchors on OLD and NEW**: cap102 transition-window plain `plan` =
  **35,164 misses / 8,240 routes**; cap73+48 frequency resident-first probe =
  **53,999 records**. Both exact on both implementations.
- **lexsort tie-break equivalence** (200 crafted trials): experts colliding at
  each rank level (equal score → last_used; equal (score,last_used) → freq; equal
  (score,last_used,freq) → −expert), including never-used residents (last_used
  = −1). NEW admissions/victims/slot-map == OLD.
- **Randomized lockstep fuzz** across `frequency`, `lru`, `transition-window`,
  `transition-window-tuned` (60 trials each): random routes (with duplicates),
  capacities 8..128, prefill+decode plans, `prepare_prefill_seed`, pins,
  `plan_transaction`/`try_plan_all_hits_transaction` rollback, `invalidate_expert`,
  `reset` — full state + mirror equality after every op.
- **Mirror invariants**: `_last_used_arr == [h.last_used for h]` and
  `_prefill_freq_arr == _prefill_route_freq` asserted after every op, across
  rollback and reset.

Robustness note: `_last_used_arr` is maintained at ctor/reset/all write sites and
is authoritative-equal in production (the runtime constructs banks via the
constructor and never restores `_history` wholesale). The offline `restore`
helper loads prefill-boundary warm states whose `last_used` are all −1 (prefill
never touches `last_used`) = the constructor default, so the mirror stays exact
there too (verified: all 40 initial banks have `last_used == −1`).
`_prefill_freq_arr` is restore-robust by construction (the property setter
rebuilds it on the populated-counter assignment).

## Existing CPU-only expert_streaming suite

```
173 passed, 2 warnings in 1.71s
```
(test_expert_streaming, test_expert_streaming_models, test_bank_history_policy,
test_expert_admission, test_expert_hot_path_invariants, test_expert_profiles,
test_expert_io_metrics, test_analyze_expert_route_trace. The 2 warnings are
unrelated SwigPy `__module__` deprecations.)

## Honest assessment

The gain is **modest**: ~1.16–1.20× on the pure policy path, saving ~0.11 s of
main-thread time per 1,024-token run. It removes the two biggest transition-window
host hotspots (the 384-object `last_used` rebuild and the ~100-key Python rank
sort per routed call) with provably identical decisions. It does **not** reduce
the shared `_pool_touch`/validation/`_touch_decode` cost or the arithmetic-locked
`_observe_transition_window`, which now dominate the remaining per-call time and
would need separate, higher-risk work to touch.
