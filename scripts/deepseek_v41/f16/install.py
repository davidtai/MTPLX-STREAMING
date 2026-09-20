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
    the greenlet driver would hard-deadlock -- a blocking ``lock.acquire`` on the one
    thread can never be released by the suspended holder; expert_runtime.py:2620-2632) and the
    device route is OFF (the clone omits its cold-recovery block).
  * REBIND each runner's ``switch._run`` to the yield-capable run (derived from the
    retained scheduled source by one round-trip-checked line insertion) and give each
    runner an ``_f16_yield`` (the shared greenlet hand-off).
  * WRAP each runner's ``issue_next`` so it is a no-op on the trailer group's greenlet
    (the leader already issued that layer's next projection; the trailer re-issuing
    would clobber a live buffer -- design step 5).

The projection store's 2->4 buffer growth is a STAGED edit to projection_install.py
(stage_f16_runner); this module only adds the trailer ``issue_next`` no-op, because
detecting the trailer needs the F16 greenlet attribute and staging that into
projection_install would couple it to F16.  Extra owner: 2 x 67,108,864 bytes
(``F16_EXTRA_PROJECTION_BYTES``), reported for memory admission.

The ``Pipeline`` object is ALWAYS stashed on ``target._f16_pipeline`` (a passthrough
when not armed) so the staged hybrid injection ``_F16_PIPELINE = model._f16_pipeline``
is always valid and an F16=0 run on a staged tree is a stock A/B control.
"""
from __future__ import annotations

import collections
import json
import os

from .pipeline import (
    Pipeline,
    bind_barrier_run,
    bind_stamped_run,
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
    """Make each runner's ``issue_next`` a no-op on the trailer group's greenlet."""
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


def install(target, *, armed: bool, split: str = "fixed4", handoff: str = "reads",
            stamps_stem: str | None = None, engram_lookahead: bool = False,
            policy_window: int | None = None) -> dict:
    """Wire (or, when not armed, passthrough-stash) the F16 pipeline on ``target``.

    ``handoff`` selects WHICH derived run + hand-off helpers are compiled and bound,
    once, at this quiescent post-prime boundary: ``reads`` (today's single hand-off) or
    ``barrier`` (F18: submit the routing barrier early, hand off, block after the
    partner's slice).  ``stamps_stem`` (``MTPLX_DSV41_F16_STAMPS``) instead binds the
    STAMPED variant of that run (a separate compiled function; host-only diagnostic).
    ``engram_lookahead`` (``MTPLX_DSV41_F20_ENGRAM_LOOKAHEAD``) binds the F20 engram
    read-lookahead callable on the pipeline; it REFUSES here (during construction) unless
    the F6 parallel gather is already installed on every engram hook cache.  Only meaningful
    when armed (the passthrough never drives the pipeline), so it is ignored when not armed.
    No per-call mode branch: the run function, the driver strategy, the hand-off helpers,
    and the lookahead callable are all fixed here.

    ``policy_window`` (``MTPLX_DSV41_F24_POLICY_WINDOW``) widens the transition-window
    expert-cache horizon on every per-layer bank from the shipped 16 routes to the given
    value.  The two-group verify pipeline issues two ``plan()`` calls per layer per forward,
    so the shipped window spans half as many forwards as designed; 32 restores the 16-forward
    horizon.  It is a cache decision only (which records prefetch, never a routed output), so
    outputs stay bit-exact.  Applied ONCE here (``install_policy_window``), after the gates,
    validating every bank before touching any; only meaningful when armed, so it is ignored
    when not armed."""
    runtime = target._mtplx_expert_runtime

    if not armed:
        pipeline = Pipeline(target, armed=False)
        target._f16_pipeline = pipeline
        return {"installed": False, "armed": False, "reason": "MTPLX_DSV41_F16 != 1"}

    # -- arm: correctness gates, all at construction (never per call) -------
    source_pins = verify_source_pins()
    if getattr(runtime, "_global_bank", None) is not None:
        raise RuntimeError(
            "F16 requires per-layer route locks (runtime._global_bank is None); a "
            "global bank collapses the per-layer locks to one and a blocking acquire "
            "on the single greenlet thread would hard-deadlock (expert_runtime.py:2620-2632)"
        )
    on = [e for e in _DEVICE_ROUTE_ENVS if os.environ.get(e) == "1"]
    if on:
        raise RuntimeError(
            f"F16 requires the device route OFF; these are set: {on} "
            "(the clone omits the device-route cold-recovery block)"
        )
    layers = _assert_scheduled_lane(target)
    runners = _collect_runners(target)

    # Pipeline first (barrier binds `_f16_barrier = pipeline.barrier`); the run function
    # + hand-off helpers are then bound once per the construction-time mode.  The F20
    # lookahead is bound here too (refuses if enabled without the F6 gather installed).
    pipeline = Pipeline(target, armed=True, split=split, handoff=handoff,
                        engram_lookahead=engram_lookahead)
    target._f16_pipeline = pipeline

    stamps_on = bool(stamps_stem)
    if stamps_on:
        from . import stamps as _stamps

        _stamps.configure(stamps_stem)
        run_shas = bind_stamped_run(runners, pipeline, barrier=(handoff == "barrier"))
        import atexit

        atexit.register(_stamps.write)
    elif handoff == "barrier":
        run_shas = bind_barrier_run(runners, pipeline)
    else:
        run_shas = bind_yield_run(runners)
    wrapped = _wrap_issue_next_for_trailer(runners)

    report = {
        "installed": True,
        "armed": True,
        "driver": "greenlet",
        "handoff": handoff,
        "leader_skip": pipeline._skip,
        "stamps": stamps_on,
        "engram_lookahead": pipeline.engram_lookahead,
        "engram_lookahead_layers": len(getattr(pipeline, "_f20_layers", ())),
        "target_layers": layers,
        "issue_next_wrapped": wrapped,
        "per_layer_locks": True,
        "device_route_off": True,
        "extra_projection_bytes": F16_EXTRA_PROJECTION_BYTES,
        "a_rows": pipeline_a_rows(),
        "split": pipeline.split,
        "source_pins": source_pins,
    }
    report.update(run_shas)
    # F24: widen the transition-window cache horizon on every bank (construction
    # time, after the gates; a passthrough {None, 0} fragment when not requested).
    report.update(install_policy_window(runtime, policy_window))
    return report


def install_policy_window(runtime, policy_window: int | None) -> dict:
    """Widen the transition-window expert-cache horizon on every per-layer bank (F24).

    A cache-only decision -- it changes which expert records prefetch, never a routed
    output -- applied ONCE here at construction (never per call).  ``policy_window is
    None`` leaves every bank untouched and reports ``{None, 0}``.  Otherwise EVERY bank
    is validated first (the shipped ``transition-window`` policy, the shipped limit 16,
    a live ``deque`` window and non-``None`` counts) and only then is the new limit
    applied, so a rejected bank leaves no bank half-changed; a mismatch raises before
    generation (AGENTS.md correct-by-design: fail once, clearly, at construction).

    The window deque has no ``maxlen`` (the bank pops from the left once its length
    exceeds the limit), so widening the limit is sufficient; counts, weights and the
    window contents are left untouched.  Returns the install-report fragment."""
    if policy_window is None:
        return {"policy_window": None, "policy_window_banks": 0}

    from mtplx.expert_streaming import TRANSITION_WINDOW_CACHE_POLICY

    banks = runtime._banks
    if not banks:
        raise RuntimeError(
            "F24 policy_window: runtime._banks is empty; the F16 pipeline requires "
            "per-layer transition-window banks (global bank scope is refused upstream)"
        )
    for layer, bank in banks.items():
        if bank.cache_policy != TRANSITION_WINDOW_CACHE_POLICY:
            raise RuntimeError(
                f"F24 policy_window: layer {layer} bank.cache_policy is "
                f"{bank.cache_policy!r}, not {TRANSITION_WINDOW_CACHE_POLICY!r}"
            )
        if bank._transition_window_limit != 16:
            raise RuntimeError(
                f"F24 policy_window: layer {layer} bank._transition_window_limit is "
                f"{bank._transition_window_limit!r}, not the shipped 16"
            )
        if not isinstance(bank._transition_window, collections.deque):
            raise RuntimeError(
                f"F24 policy_window: layer {layer} bank._transition_window is a "
                f"{type(bank._transition_window).__name__}, not a collections.deque"
            )
        if bank._transition_counts is None:
            raise RuntimeError(
                f"F24 policy_window: layer {layer} bank._transition_counts is None"
            )
    if isinstance(policy_window, bool) or not isinstance(policy_window, int):
        raise ValueError(
            f"policy_window must be an int in [16, 256]; got {policy_window!r}"
        )
    if not 16 <= policy_window <= 256:
        raise ValueError(
            f"policy_window must be in [16, 256]; got {policy_window}"
        )

    for bank in banks.values():
        bank._transition_window_limit = policy_window
    return {"policy_window": policy_window, "policy_window_banks": len(banks)}


def pipeline_a_rows() -> int:
    from .pipeline import A_ROWS

    return A_ROWS


def install_from_env(target) -> dict:
    """Called from the staged ``observe_seed_prefill`` after ``prime_model``.  Always
    stashes ``target._f16_pipeline`` (armed iff ``MTPLX_DSV41_F16 == '1'``); prints
    one provenance line; registers an at-exit counter dump when a path is set."""
    armed = os.environ.get("MTPLX_DSV41_F16") == "1"
    # Leader-group size, chosen ONCE here: "fixed4" (oracle = sequential chunks 4+rest) or
    # "balanced" (ceil/floor halves; its own oracle digest).  Hand-off mode chosen ONCE:
    # "reads" (today) or "barrier" (F18).  MTPLX_DSV41_F16_STAMPS=<stem> arms the
    # host-only stamped variant (diagnostic; default OFF).  MTPLX_DSV41_F20_ENGRAM_LOOKAHEAD=1
    # arms the F20 engram read lookahead (refuses unless the F6 parallel gather is installed).
    # MTPLX_DSV41_F24_POLICY_WINDOW=<int> widens the transition-window cache horizon (default
    # unset -> None -> banks unchanged); read here at use, not at import.
    raw_policy_window = os.environ.get("MTPLX_DSV41_F24_POLICY_WINDOW")
    if raw_policy_window is None or raw_policy_window == "":
        policy_window = None
    else:
        try:
            policy_window = int(raw_policy_window)
        except ValueError as exc:
            raise ValueError(
                "MTPLX_DSV41_F24_POLICY_WINDOW must be an integer; got "
                f"{raw_policy_window!r}"
            ) from exc
    report = install(
        target,
        armed=armed,
        split=os.environ.get("MTPLX_DSV41_F16_SPLIT", "fixed4"),
        handoff=os.environ.get("MTPLX_DSV41_F16_HANDOFF", "reads"),
        stamps_stem=os.environ.get("MTPLX_DSV41_F16_STAMPS") or None,
        engram_lookahead=os.environ.get("MTPLX_DSV41_F20_ENGRAM_LOOKAHEAD") == "1",
        policy_window=policy_window,
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
