# Fix: `maybe_run_model_selfcheck() got an unexpected keyword argument 'expert_spec'`

Branch: `fix/selfcheck-expert-spec` (from `origin/main` @ `3f7d6fe9`).
Scope: CPU-only, `nice -n 19`, no GPU lock, no real model loaded (peak RSS 54.5 MB).

## Symptom

On fork main (`davidtai/MTPLX-STREAMING` @ `3f7d6fe9`), `mtplx serve --model <any model>`
died at load:

```
File mtplx/runtime.py, line 1169, in _load_impl:
    maybe_run_model_selfcheck(model, expert_spec=getattr(expert_runtime, "spec", None))
TypeError: maybe_run_model_selfcheck() got an unexpected keyword argument 'expert_spec'
```

The keyword is bound before the function body runs, so the load crashed even when the
self-check was disabled by env.

## Root cause

Commit `c5f80852` ("Re-thread expert-streaming/SSD-MoE runtime wiring onto upstream
runtime") re-added the streaming self-check **call site** with the fork's `expert_spec`
keyword, but the parity reset had left `mtplx/kernel_selfcheck.py` at **upstream's**
single-argument signature `def maybe_run_model_selfcheck(model)` (upstream
`youssofal/MTPLX` @ `21be78b3`, runtime.py:857 calls `maybe_run_model_selfcheck(model)`).
Same class of bug as any re-thread that updates a caller but not its callee.

The fork's original implementation lived in the pre-parity commit
`7a46abad` ("kernel selfcheck: expert-gather lane derived from the streaming spec").
That behaviour was restored faithfully and re-threaded onto the current upstream file.

## Fix — `mtplx/kernel_selfcheck.py` only

`mtplx/runtime.py`'s call site is already correct (`expert_spec=getattr(expert_runtime,
"spec", None)`), so it was **not** touched. Restored/re-threaded four pieces:

- **`_check_expert_gather(mx, dtype, bits, group_size, *, bank_group_size=None)`** — a
  synthetic `mx.gather_qmm` round-trip at a streamed expert bank's own `(bits,
  group_size)`, referenced against a per-expert `mx.quantized_matmul` on the selected
  slot (the exact stock op the gather fans out over). Returns the worst per-row max-abs
  diff; a broken lane lands at O(1) or raises. Stock ops only, so it is CPU- and
  GPU-safe.
- **`run_kernel_selfcheck(dtype, bits, group_size, *, expert_signature=None)`** — when a
  signature is given, records an `expert_gather` lane at that format alongside the
  resident lanes (using the same fail-closed `_record` tripwire). When `None`, **no lane
  is added and the report is byte-identical to the resident-only path**.
- **`_expert_quant_signature(spec)`** — derives `(dtype, bits, group_size)` from an
  `ExpertStreamingModelSpec`'s affine quant format (BF16 scale/bias leaves →
  `quant_parameter_bytes == 2` → BF16 gather; `== 4` → FP32). Returns `None` for a
  missing spec or a non-affine shadow-codec (q1 `t158`/`b1`) bank.
- **`maybe_run_model_selfcheck(model, *, expert_spec=None)`** — threads the streamed
  spec's signature into the probe. `expert_spec=None` behaves **exactly** as upstream, so
  the dense/non-streaming call site (`runtime.py:1378`, bare `maybe_run_model_selfcheck(model)`)
  is unchanged.

Net: `mtplx/kernel_selfcheck.py`, +118 / −3.

## Test — `tests/test_runtime_selfcheck_expert_spec.py` (new)

Per the repo lesson *rebase proof must cover call sites → served-order behavioural tests
per observable*, the two headline tests drive the **real served load path**
(`mtplx.runtime.load` → `_load_impl` streaming branch, `mtp=False`, heavy
allocation/construction stubbed) so the genuine `runtime.py:1169` call site invokes the
genuine `maybe_run_model_selfcheck`. A forwarding spy records the keyword the call site
actually passed.

- `test_streamed_serve_load_invokes_selfcheck_with_expert_spec` — a streamed runtime
  carrying `HY3_EXPERT_OQ2E`: the call site passes `expert_spec=<spec>`, the
  `expert_gather` lane runs at 2-bit gs128, the load completes.
- `test_streamed_serve_load_without_spec_matches_dense_selfcheck` — a streamed runtime
  with `spec=None`: the call site passes `expert_spec=None`, the report is byte-identical
  to the resident-only path (no `expert_gather` lane), the load completes.

**Both fail on `origin/main`'s code and pass with the fix.** Verified by swapping in
`git show origin/main:mtplx/kernel_selfcheck.py` and re-running:

```
FAILED ...::test_streamed_serve_load_invokes_selfcheck_with_expert_spec
FAILED ...::test_streamed_serve_load_without_spec_matches_dense_selfcheck
  E  TypeError: maybe_run_model_selfcheck() got an unexpected keyword argument 'expert_spec'
     (raised through the served load path at the real call site)
```

With the fix: `16 passed`. The file also carries unit coverage for
`_expert_quant_signature` (affine specs, `None`, shadow codec), the `expert_gather` lane
(pass, corrupt-kernel fallback, non-dividing group_size fallback, mismatched-group-size
raise), and `maybe_run_model_selfcheck` with/without a spec and when disabled. MLX pinned
to CPU throughout.

## Existing tests (CPU, `nice -n 19`, no `-n auto`)

- `tests/test_kernel_selfcheck.py` — **18/18 passed** (directly exercises the changed module).
- `tests/test_glm52_streamed_mtp.py`, `tests/test_hy3_streamed_mtp.py`,
  `tests/test_server_obs_caps_and_health.py`, `tests/test_server_openai.py`,
  `tests/test_runtime_selfcheck_expert_spec.py` — all pass **except 2 pre-existing
  failures** in `test_glm52_streamed_mtp.py`:
  `test_runtime_finish_mtp_cycle_delegates_when_supported` and
  `test_runtime_finish_mtp_cycle_is_compatible_with_other_backends`, both
  `AttributeError: 'MTPLXRuntime' object has no attribute 'finish_mtp_cycle'`.

  These are **unrelated to this fix** and pre-exist on `origin/main`: this diff touches
  only `mtplx/kernel_selfcheck.py` (runtime.py byte-identical to `origin/main`), and
  `finish_mtp_cycle` is not defined anywhere in `mtplx/runtime.py`. It is a **separate
  dropped-method gap** from the same re-thread (`c5f80852`) — listed below, not fixed
  here (different observable, needs the `finish_mtp_cycle` delegation restored on
  `MTPLXRuntime`; out of scope for this brief).

## Audit of `c5f80852` for other call-site / signature mismatches

`c5f80852` touched only `mtplx/runtime.py` (+612 / −38). Every kwarg-passing call site it
added was introspected against its real callee's signature. **No other mismatch of the
self-check class** (fork kwargs into an upstream-signature callee):

| Call site (kwargs) | Callee | Result |
| --- | --- | --- |
| `maybe_run_model_selfcheck(expert_spec=)` | `kernel_selfcheck` | **was broken → fixed** |
| `construct_resident_model(config=)` | `resident_loader` | OK |
| `ExpertStreamingRuntime.open(spec, buffer_allocator, device_synchronize, apply_memory_cap, mx_module, expert_admission_receipt, additional_resident_bytes)` | `expert_runtime` | OK |
| `apply_mlx_memory_cap(mx_module=)` | `expert_runtime` | OK |
| `inject_hy3_streamed_mtp_support(expected_revision, mtp_precision, shared_kernel, shared_kernel_depth, mtp_module)` | `hy3_mtp_patch` | OK |
| `build_hy3_mtp_module(expected_revision, precision, shared_kernel, shared_kernel_depth, verified_artifacts)` | `hy3_mtp_patch` | OK (`verified_artifacts` plural is correct) |
| `open_verified_hy3_mtp_artifacts(precision, expected_revision)` | `hy3_mtp_patch` | OK |
| `inject_glm52_streamed_mtp_support(expected_revision, verified_artifact, mtp_module)` | `glm52_mtp_patch` | OK |
| `build_glm52_mtp_module(expected_revision, precision, verified_artifact)` | `glm52_mtp_patch` | OK (`verified_artifact` singular is correct) |
| `configure_hy3_router_kernels(sigmoid_mode, splitk_m1_scope)` | `models/hy3_mlx` | OK |
| `estimate_hy3_router_kernel_incremental_bytes(include_mtp)` | `models/hy3_mlx` | OK |
| `MTPContract.with_runtime_metadata(preserve_explicit)` | `mtp_patch` | OK |
| `MTPContract.with_config_defaults()` | `mtp_patch` | OK |
| `ExpertStreamingRuntime.close(timeout)` | `expert_runtime` | OK |
| `open_verified_glm52_mtp_layer78(deep)` | `glm52_mtp_artifact` | OK |

**Other gap found (listed, not fixed — different class/observable):** `c5f80852`'s
re-threaded `MTPLXRuntime` does not define `finish_mtp_cycle` (the streaming-backend
delegation method its tests expect). It is not a kwarg/signature mismatch and does not
break the serve load path; two `test_glm52_streamed_mtp.py` tests are red on `origin/main`
because of it.

## Commits (branch `fix/selfcheck-expert-spec`, no attribution trailers per MTPLX policy)

1. `73f0327e` — Restore expert_spec self-check lane broken by the upstream re-thread
2. `c6d43cdc` — Add served-order regression test for the expert_spec self-check call site
3. (this report)

`scripts/check_ai_attribution.py --range origin/main..HEAD` clean.

## Related deliverable (separate branch — cherry-pick pending)

`scripts/deepseek_v41/gpu_window.sh` phase-4 guard hardened to be **system-wide** (poll
wired+active+compressed used memory, abort+restore over a configurable ceiling default
105 GiB, refuse to start when other >2 GiB `mtplx`/`python`/`mlx` workers are resident;
default child RSS cap lowered 100→90 GiB), plus `scripts/deepseek_v41/test_gpu_window_guard.sh`
(12 assertions against fake `vm_stat`/`ps`, bash 3.2, no GPU/lock/launchctl).

Committed as **`c93514e6`** on branch **`fix/gpu-window-systemwide-guard`** (cut from the
`feat/deepseek-v41-streaming` tip). It was NOT cherry-picked into the integration worktree
directly: `gpu_window.sh` is executing there right now (the orchestrator's live CPU-phase
window, holding the GPU lock), and rewriting a running bash script corrupts its
continuation reads once the poll loop ends — which would break the resident-agent restore
and lock release. Cherry-pick onto the integration branch once that window closes:

```
git -C .worktrees/deepseek-v41 cherry-pick c93514e6   # after PID 6250's window exits
```
