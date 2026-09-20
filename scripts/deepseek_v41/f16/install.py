"""F16 install: wire the verify-row-group pipeline as the LAST step of
``observe_seed_prefill`` (run_full.py:737, after ``prime_model``), no-op-armed unless
``MTPLX_DSV41_F16 == '1'``.

What arming does (all at this quiescent post-prime boundary, once, never per call):
  * VERIFY the pinned-runtime sources the per-group forward clones still match their
    SHAs (``pipeline.verify_source_pins``).
  * REFUSE unless every streamed switch's ``_run`` is the retained SCHEDULED run
    (F16 stage 1 is mutually exclusive with F2b and the F5 stamp probe -- the enabled
    hot path must be the scheduled run, not a wrapped/stamped one).
  * REFUSE unless the per-layer route locks are genuinely per-layer
    (``runtime._global_bank is None``; a global bank collapses them to ONE lock and
    the baton would serialize/deadlock -- expert_runtime.py:2620-2632) and the
    device route is OFF (the clone omits its cold-recovery block).
  * REBIND each runner's ``switch._run`` to the yield-capable run (derived from the
    retained scheduled source by one round-trip-checked line insertion) and give each
    runner an ``_f16_yield`` (the shared baton hand-off).
  * WRAP each runner's ``issue_next`` so it is a no-op on the trailer group's thread
    (the leader already issued that layer's next projection; the trailer re-issuing
    would clobber a live buffer -- design step 5).

The projection store's 2->4 buffer growth is a STAGED edit to projection_install.py
(stage_f16_runner); this module only adds the trailer ``issue_next`` no-op, because
detecting the trailer needs the F16 thread-local and staging that into
projection_install would couple it to F16.  Extra owner: 2 x 67,108,864 bytes
(``F16_EXTRA_PROJECTION_BYTES``), reported for memory admission.

The ``Pipeline`` object is ALWAYS stashed on ``target._f16_pipeline`` (a passthrough
when not armed) so the staged hybrid injection ``_F16_PIPELINE = model._f16_pipeline``
is always valid and an F16=0 run on a staged tree is a stock A/B control.
"""
from __future__ import annotations

import json
import os

from .pipeline import (
    Pipeline,
    bind_yield_run,
    issue_suppressed,
    scheduled_run_cocode,
    verify_source_pins,
)

# Two extra bf16 transpose buffers vs the retained 2-buffer store (each
# (8,4096,1024) bf16 = 67,108,864 bytes); charged to memory admission by the launcher.
F16_EXTRA_PROJECTION_BYTES = 2 * 67_108_864

_DEVICE_ROUTE_ENVS = ("MTPLX_DSV41_DEVICE_ROUTE", "MTPLX_DSV41_DEVICE_ROUTE_PINNED")


def _assert_scheduled_lane(target) -> int:
    """Refuse unless every streamed switch runs the retained scheduled run."""
    ref = scheduled_run_cocode()
    n = 0
    for layer in target.model.layers:
        run = layer.mlp.switch_mlp._run
        fn = getattr(run, "__func__", None)
        if fn is None or fn.__code__.co_code != ref:
            raise RuntimeError(
                f"F16 requires the retained scheduled run on layer {n}; found a "
                "wrapped/stamped/other lane (F2b or F5 not composable with F16)"
            )
        n += 1
    if n == 0:
        raise RuntimeError("F16: no streamed switches found on the target")
    return n


def _collect_runners(target) -> dict:
    runners = {}
    for index, layer in enumerate(target.model.layers):
        switch = layer.mlp.switch_mlp
        runner = getattr(switch._run, "__self__", None)
        if runner is None:
            raise RuntimeError(f"F16: no bound runner on layer {index}")
        runners[index] = (switch, runner)
    return runners


def _wrap_issue_next_for_trailer(runners) -> int:
    """Make each runner's ``issue_next`` a no-op on the trailer group's thread."""
    wrapped = 0
    for _layer, (_switch, runner) in runners.items():
        orig = runner.issue_next  # partial(store.issue, (index+1) % 40)

        def guarded(_orig=orig):
            if issue_suppressed():
                return None
            return _orig()

        runner.issue_next = guarded
        wrapped += 1
    return wrapped


def install(target, *, armed: bool, baton_timeout: float | None = None) -> dict:
    """Wire (or, when not armed, passthrough-stash) the F16 pipeline on ``target``."""
    from .pipeline import BATON_TIMEOUT_S

    timeout = BATON_TIMEOUT_S if baton_timeout is None else float(baton_timeout)
    runtime = target._mtplx_expert_runtime

    if not armed:
        pipeline = Pipeline(target, armed=False, baton_timeout=timeout)
        target._f16_pipeline = pipeline
        return {"installed": False, "armed": False, "reason": "MTPLX_DSV41_F16 != 1"}

    # -- arm: correctness gates, all at construction (never per call) -------
    source_pins = verify_source_pins()
    if getattr(runtime, "_global_bank", None) is not None:
        raise RuntimeError(
            "F16 requires per-layer route locks (runtime._global_bank is None); a "
            "global bank collapses the per-layer locks to one and the baton would "
            "serialize/deadlock (expert_runtime.py:2620-2632)"
        )
    on = [e for e in _DEVICE_ROUTE_ENVS if os.environ.get(e) == "1"]
    if on:
        raise RuntimeError(
            f"F16 requires the device route OFF; these are set: {on} "
            "(the clone omits the device-route cold-recovery block)"
        )
    layers = _assert_scheduled_lane(target)
    runners = _collect_runners(target)
    yield_shas = bind_yield_run(runners)
    wrapped = _wrap_issue_next_for_trailer(runners)

    pipeline = Pipeline(target, armed=True, baton_timeout=timeout)
    target._f16_pipeline = pipeline

    report = {
        "installed": True,
        "armed": True,
        "target_layers": layers,
        "issue_next_wrapped": wrapped,
        "per_layer_locks": True,
        "device_route_off": True,
        "extra_projection_bytes": F16_EXTRA_PROJECTION_BYTES,
        "baton_timeout_s": timeout,
        "a_rows": pipeline_a_rows(),
        "source_pins": source_pins,
    }
    report.update(yield_shas)
    return report


def pipeline_a_rows() -> int:
    from .pipeline import A_ROWS

    return A_ROWS


def install_from_env(target) -> dict:
    """Called from the staged ``observe_seed_prefill`` after ``prime_model``.  Always
    stashes ``target._f16_pipeline`` (armed iff ``MTPLX_DSV41_F16 == '1'``); prints
    one provenance line; registers an at-exit counter dump when a path is set."""
    armed = os.environ.get("MTPLX_DSV41_F16") == "1"
    baton_timeout = os.environ.get("MTPLX_DSV41_F16_BATON_TIMEOUT_S")
    report = install(
        target,
        armed=armed,
        baton_timeout=float(baton_timeout) if baton_timeout else None,
    )
    counters_path = os.environ.get("MTPLX_DSV41_F16_COUNTERS")
    if counters_path:
        import atexit

        atexit.register(dump_counters, target, counters_path)
    _print_install_report(report)
    return report


def _print_install_report(report: dict) -> None:
    print("F16_INSTALL " + json.dumps(report, sort_keys=True, default=str), flush=True)


def dump_counters(target, path) -> dict:
    pipeline = getattr(target, "_f16_pipeline", None)
    data = dict(pipeline.counters) if pipeline is not None else {"installed": False}
    data["armed"] = bool(getattr(pipeline, "armed", False))
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    return data
