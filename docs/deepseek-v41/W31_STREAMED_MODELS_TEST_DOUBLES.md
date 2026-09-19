# W31 — streamed-model test doubles refreshed for the W11 clamp + W15 codec

## Summary

The DeepSeek-V4.1 W11 (SwiGLU clamp) and W15 (native-mxfp4 codec) work added two
keyword arguments — `swiglu_limit` and `codec` — to the streamed expert helpers
`_run_q4_expert`, `_run_component_bank_q4`, and `_run_mapped_q4` in
`mtplx/models/expert_mlx.py`, and every production call site now passes them. The
in-test monkeypatch doubles for those helpers were not updated, so they raised
`TypeError: unexpected keyword argument 'swiglu_limit'` the moment the switch
called them. A related enumeration test (`test_spec_swiglu_limit_values`) still
expected exactly one spec to carry the clamp, but W15 added a second DeepSeek-V4.1
spec that legitimately inherits it.

Every failure is **test-double / test-expectation staleness**, not a production
regression. The production callers are correct: with `swiglu_limit=None` the clamp
is a no-op (`_clamped_swiglu` returns plain `swiglu`, so hy3/glm are byte-for-byte
unchanged) and `codec="affine"` is the default path.

All fixes are in the test files only; no production file was touched.

## Failures (measured on the integration base 5f0987837, CPU-pinned)

The task brief cited 4 failing cases (an earlier count at 373ec9f9 / 2ed30c9c3).
The current integration base carries **9** double-staleness failures in
`tests/test_streamed_models.py` (W28 merged more streamed tests that reuse the
same strict double signature) plus **1** in the clamp file the coordinator added
to scope — 10 total.

| # | Test | Patched helper | Stale double (line) | Error |
|---|------|----------------|---------------------|-------|
| 1 | `test_streamed_decode_evaluates_shared_work_before_waiting_for_misses` | `_run_q4_expert` | `fake_q4` (493) | `TypeError: unexpected kwarg 'swiglu_limit'`, then `AttributeError: '_OverlapPending' object has no attribute 'abort'` |
| 2 | `test_component_bank_overlaps_hit_and_shared_work_with_incremental_misses` | `_run_component_bank_q4` | `fake_q4` (668) | `TypeError: unexpected kwarg 'swiglu_limit'` |
| 3 | `test_component_bank_claims_runnable_work_immediately_before_dispatch` | `_run_component_bank_q4` | `fake_q4` (714) | `TypeError: unexpected kwarg 'swiglu_limit'` |
| 4 | `test_128k_prefill_preserves_bounded_routed_then_shared_order` | `_run_q4_expert` | `fake_q4` (934) | `TypeError`, then `_OverlapPending` has no `abort` |
| 5 | `test_descriptor_bits_reach_direct_slot_switch` | `_run_q4_expert` | `observe_qmm` (1921) | `TypeError`, then `_OverlapPending` has no `abort` |
| 6 | `test_descriptor_bits_reach_component_bank_all_hit_switch` | `_run_component_bank_q4` | `observe_qmm` (1960) | `TypeError`, then `UnboundLocalError: deferred_release` (see note) |
| 7 | `test_descriptor_bits_reach_component_bank_split_hit_and_misses` | `_run_component_bank_q4` | `observe_qmm` (1981) | `TypeError: unexpected kwarg 'swiglu_limit'` |
| 8 | `test_descriptor_bits_reach_mapped_switch` | `_run_mapped_q4` | `observe_qmm` (2006) | `TypeError: unexpected kwarg 'swiglu_limit'` |
| 9 | `test_component_bank_all_hit_decode_keeps_router_order_without_split_route_ops` | `_run_component_bank_q4` | `assignment_marker` (2648) | `TypeError: unexpected kwarg 'swiglu_limit'` |
| 10 | `tests/models/test_deepseek_v41_streaming_clamp.py::test_spec_swiglu_limit_values` | — (spec enumeration) | test expectation | `AssertionError: {None, 10.0} == {None}` |

Three further tests (`test_global_component_bank_all_hit_decode_binds_without_reads`,
`..._preserves_route_waves_counters_and_shared_order`,
`..._releases_pins_on_q4_error`) carry the same strict double
signature (2746, 2826, 2964). They did not surface in the base failure list because
they run a real cold pass first, but their doubles were refreshed identically so
they cannot fail on the same TypeError when run on Metal / with the real artifact.

## Cause, per case

### Cases 1–9 (`tests/test_streamed_models.py`) — stale monkeypatch doubles

- **New production kwargs.** `swiglu_limit: float | None = None` was added to
  `_run_q4_expert` / `_run_component_bank_q4` / `_gather_component_bank` /
  `_run_mapped_q4` by **`afcd87e19`** (`deepseek_v41 W11: apply the reference
  SwiGLU clamp on the STREAMING path`). `codec: str = "affine"` was added by
  **`0b80dc5d1`** (`deepseek_v41 W15: native-mxfp4 runtime codec through the
  streaming switches`). `git log -S swiglu_limit` / `git log -S 'codec: str =
  "affine"'` on `mtplx/models/expert_mlx.py` confirm both.
- **Production still passes them at every call site.** `_dispatch_component_bank`
  (expert_mlx.py:1889) and `evaluate_direct_bindings` (2165) both pass
  `swiglu_limit=self.swiglu_limit, codec=self.codec`, where
  `self.swiglu_limit = getattr(runtime.spec, "swiglu_limit", None)` and
  `self.codec = getattr(runtime.spec, "expert_codec", "affine")`. The mock
  runtimes in these tests do not set those spec fields, so production passes
  `swiglu_limit=None, codec="affine"` — the unclamped affine path.
- **hy3 / glm production callers are unaffected.** `_clamped_swiglu` returns the
  plain `swiglu(gate, up)` whenever `swiglu_limit is None` (or `<= 0`); `codec`
  defaults to `"affine"`. Every hy3 / glm spec leaves `swiglu_limit` unset, so
  those served paths are byte-for-byte identical (proved separately by
  `test_none_limit_keeps_plain_swiglu_byte_identical`).
- The doubles were declared `def f(selected, ..., *, group_size, bits)` with no
  `swiglu_limit` / `codec` parameter, so the production call raised `TypeError`.

### `_OverlapPending` missing `.abort` (cases 1, 4, 5)

Once the double raised `TypeError`, the switch's split-route error handler
(`except BaseException as exc: pending.abort(exc)`, expert_mlx.py:2307 / 2639)
ran, and `_OverlapPending` — unlike its siblings `_BankOverlapPending` (line 564)
and `_OwnedMissPending` (line 992) — had no `abort`, so an `AttributeError`
masked the original `TypeError`. This double was stale for the same production
change: the failure-safe abort path was added by `e7c22691a` /
`bd36da767`. With the doubles refreshed, no exception is raised, so `abort` is
not reached — but it is added for parity with the other pending doubles.

### Case 10 (`test_spec_swiglu_limit_values`) — stale test expectation

`DEEPSEEK_V41_FLASH_EXPERT_MXFP4 = replace(DEEPSEEK_V41_FLASH_EXPERT_Q2, ...)`
(added by **`3b7346a3c`**, W15 native-mxfp4 spec) does not override
`swiglu_limit`, so it correctly inherits `10.0` from the Q2 spec. **Both**
DeepSeek-V4.1 streaming specs legitimately need the reference `±10` clamp
(`config text_config.swiglu_limit=10.0`); hy3 / glm must stay `None`. The spec
table is correct — the test's expectation (exactly one spec ≠ `None`) is the
stale side. Fixed by exempting both DSV4.1 clamp keys and asserting every other
spec is `None`.

## Fix

Test files only (`git diff --stat`: 2 files, +29/-10):

- `tests/test_streamed_models.py`
  - 8 single-line doubles (`fake_q4` ×4, `observe_qmm` ×4) and 4 multi-line
    doubles (`assignment_marker` ×2, `observe_component_bindings`,
    `fail_second_wave_with_identity`): added `swiglu_limit=None, codec="affine"`
    to each signature. The doubles keep their existing `group_size` / `bits`
    assertions; since production passes `swiglu_limit=None` / `codec="affine"`
    here, honouring the kwargs is the identity/unclamped-affine path the doubles
    already model, so no body change was needed.
  - `_OverlapPending.abort(self, error)` added, mirroring
    `_BankOverlapPending.abort` (`self.events.append(f"abort:{type(error).__name__}")`).
- `tests/models/test_deepseek_v41_streaming_clamp.py`
  - `test_spec_swiglu_limit_values` now exempts
    `{deepseek-v41-flash-expert-q2, deepseek-v41-flash-expert-mxfp4}`, asserts
    each is `10.0`, and asserts every other spec is `None`.

No production file changed.

## Test counts (CPU-pinned: `mx.set_default_device(mx.cpu)`; no GPU; no `~/models` artifact loaded)

Interpreter: `PYTHONPATH=$PWD nice -n 19 .venv/bin/python3 -m pytest <file> -p no:warnings -p cpu_pin`

| File | Result |
|------|--------|
| `tests/test_streamed_models.py` | **80 passed** |
| `tests/test_expert_overlap_split.py` | 9 passed |
| `tests/test_shared_hoist.py` | 3 passed |
| `tests/test_expert_mlx_mxfp4.py` | 3 passed |
| `tests/models/test_deepseek_v41_streaming_clamp.py` | 3 passed, 2 skipped (artifact-gated: `experts.bin not present`) |
| **Combined** | **98 passed, 2 skipped** |

Baseline before the fix: 9 failed in `tests/test_streamed_models.py`, 1 failed in
the clamp file.

## Note — latent production robustness gap (NOT a regression; no test left red)

Case 6's traceback exposed a secondary, pre-existing error-path fragility in
`mtplx/models/expert_mlx.py` around the all-hit component-bank branch (line
~2393–2429): `deferred_release` is assigned (2401) only **after**
`self._dispatch_component_bank(...)` (2393) returns, but the enclosing `finally`
references it (`if not deferred_release: ready.release(...)`, 2429). If the
dispatch raises before 2401 — as the stale double did — the `finally` fails with
`UnboundLocalError: cannot access local variable 'deferred_release'`, masking the
real error.

This is **not** the cause of any test's assertion failing and does **not**
require leaving a test red: with the double refreshed, `_dispatch_component_bank`
returns normally, `deferred_release` is bound, and case 6 passes. It is only
reachable when the all-hit dispatch itself raises (a real kernel/OOM error), in
which case the cleanup would crash instead of propagating the true cause — the
same failure-safety concern `e7c22691a` addressed for the miss/split path.
Flagged for a possible W-follow-up (initialise `deferred_release = False` before
the `try`); out of scope for this test-double task and deliberately not patched
here.

## Commit

Branch `feat/deepseek-v41-w31` off `feat/deepseek-v41-streaming` (base
`5f0987837`); this fix is the tip of `feat/deepseek-v41-w31` (SHA reported in the
task handoff). `scripts/check_ai_attribution.py --range
feat/deepseek-v41-streaming..HEAD`: clean.
