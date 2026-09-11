"""W42 (KERNEL_LEDGER K23) -- DSV4.1 streamed-switch fast-path.

CPU-pinned, tiny synthetic streamed switch + fake bank (reuses the fake-runtime
pattern from ``tests/test_streamed_models.py``), plus one real streamed runtime
(``_integrated_hy3_artifact``) for the slot-safe byte-identity proof.  No GPU, no
artifact, no real weights; MLX pinned to the CPU device (memory/worker-tests-must-
pin-mlx-cpu.md: "no GPU" is not enough -- MLX defaults to Metal).  Run under
``nice -n 19`` and without ``pytest -n auto``.

The finding (W37 stage-timing + route probe): the DSV4.1 streamed switch decodes
at ~2.5 ms/layer with warm-repeat == cold (3.80 vs 3.73 tok/s), so miss I/O is
NOT the cost.  The route probe attributes the all-hit cost to
``hot.allhit_fence_eval`` -- one BLOCKING ``mx.eval(wave_output)`` per all-hit
layer (``synchronous_fence``), a SECOND device->host round-trip on top of the
``mx.eval(indices)`` routing barrier.  hy3/glm never pay it: their profiles ship
``deferred_pin_release=True`` (+ ``split_route_release="deferred"``); DSV4.1's
``build_streaming_config`` leaves both at the fenced default.  The fast-path
(env ``MTPLX_DSV41_SWITCH_FASTPATH``, default off) promotes the identical,
already-shipped deferred mechanism for the DSV4.1 lane.

These tests lock:

  1. host-sync census: the fast-path removes the per-all-hit-layer blocking
     ``mx.eval`` wave fence (all-hit) and the per-wave blocking fences (split /
     all-miss), preserving EXACTLY ONE routing barrier per layer, and queues one
     deferred slot release instead;
  2. the switch output is bitwise-identical, flag off vs on, for all-hit /
     split-route / all-miss at M=1 and M=4 (the MTP verify row batch);
  3. it falls back to the shipped blocking fence (never crashes) when the runtime
     cannot defer;
  4. on a REAL streamed runtime the flag-driven deferral is bitwise-identical to
     the fenced default and actually holds its slot pins until the covering flush
     (the slot-safety guarantee).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

pytest.importorskip("mlx.core")

mx.set_default_device(mx.cpu)

import mtplx.models.expert_mlx as expert_mlx  # noqa: E402
from mtplx.expert_runtime import RouteWave  # noqa: E402
from mtplx.models.expert_mlx import HotExpertSwitchGLU  # noqa: E402

FASTPATH_FLAG = "MTPLX_DSV41_SWITCH_FASTPATH"

_REAL_EVAL = mx.eval
_REAL_ASYNC = mx.async_eval
_REAL_BRACKET = expert_mlx._route_probe.bracket
_REAL_BANK_Q4 = expert_mlx._run_component_bank_q4

_BANK = object()


@pytest.fixture(autouse=True)
def _cpu_default_device_and_flag():
    """Pin MLX to CPU and isolate the fast-path env flag around every test."""
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    import os

    saved = os.environ.get(FASTPATH_FLAG)
    os.environ.pop(FASTPATH_FLAG, None)
    try:
        yield
    finally:
        mx.set_default_device(previous_device)
        if saved is None:
            os.environ.pop(FASTPATH_FLAG, None)
        else:
            os.environ[FASTPATH_FLAG] = saved


# ---------------------------------------------------------------------------
# Self-contained fake runtime + route doubles (component-bank layout, DECODE).
# The config intentionally OMITS deferred_pin_release / split_route_release, so
# getattr() falls back to (False / "fenced") -- exactly the DSV4.1 shipped
# default the fast-path promotes.
# ---------------------------------------------------------------------------
def _binding(expert: int, index: int) -> SimpleNamespace:
    return SimpleNamespace(
        expert=int(expert),
        buffer=SimpleNamespace(bank=_BANK, bank_index=int(index)),
    )


def _all_hit_ready(experts, events) -> SimpleNamespace:
    bindings = tuple(_binding(e, i) for i, e in enumerate(experts))
    return SimpleNamespace(
        plan=SimpleNamespace(hits=tuple(range(len(experts)))),
        bindings=bindings,
        release=lambda **_kw: events.append("allhit-release"),
    )


class _FakePending:
    """A split-route pending: the first ``hit_experts`` are resident hits, the
    rest are one miss part.  Duck-types the surface ``_run`` touches."""

    def __init__(self, events, experts, hit_experts) -> None:
        self.events = events
        experts = tuple(experts)
        hit_set = set(hit_experts)
        self.plan = SimpleNamespace(
            hits=tuple(e for e in experts if e in hit_set),
            misses=tuple(e for e in experts if e not in hit_set),
        )
        self.misses_pending = bool(self.plan.misses)
        hit_bindings = tuple(
            _binding(e, i) for i, e in enumerate(experts) if e in hit_set
        )
        self.hit_ready = (
            SimpleNamespace(bindings=hit_bindings) if hit_bindings else None
        )
        miss_bindings = tuple(
            _binding(e, i) for i, e in enumerate(experts) if e not in hit_set
        )
        self._miss_ready = (
            SimpleNamespace(
                bindings=miss_bindings,
                plan=SimpleNamespace(experts=tuple(self.plan.misses)),
            )
            if miss_bindings
            else None
        )

    def iter_ready_misses(self):
        if self._miss_ready is not None:
            self.events.append("ready-miss")
            self.misses_pending = False
            yield self._miss_ready

    def release_hits(self) -> None:
        self.events.append("release-hits")

    def release_miss(self, _part) -> None:
        self.events.append("release-miss")

    def abort(self, error) -> None:
        self.events.append(f"abort:{type(error).__name__}")

    def close(self) -> None:
        self.events.append("close")


class _FastpathRuntime:
    """Component-bank DECODE runtime with a working deferred-release queue."""

    def __init__(self, events, *, all_hit_ready=None, pending=None) -> None:
        self.events = events
        self.spec = SimpleNamespace(
            top_k=1,
            hidden_size=2,
            quant_group_size=64,
            quant_bits=4,
        )
        self.manifest = SimpleNamespace(sidecar=None)
        # deferred_pin_release / split_route_release deliberately absent.
        self.config = SimpleNamespace(
            slot_layout="component-banks",
            resource_telemetry=False,
        )
        self._pipeline_ledger = None
        self._all_hit_ready = all_hit_ready
        self._pending = pending
        self.deferred = []

    def observe_route(self, *_a, **_k) -> None:
        return None

    def prepare_prefill_seed(self, *_a, **_k):
        return ()

    def route_waves(self, expert_ids, **_k):
        experts = tuple(expert_ids)
        return (RouteWave(positions=tuple(range(len(experts))), experts=experts),)

    def peek_resident_experts(self, *_a, **_k):
        return frozenset()

    def try_all_hit_route(self, *_a, **_k):
        return self._all_hit_ready

    def begin_split_route(self, _layer, _experts, **_k):
        self.events.append("begin-split")
        return self._pending

    # Deferred-release machinery (present -> ``_fastpath_can_defer`` is True).
    def defer_slot_release(self, ready, wave_output) -> None:
        self.events.append("defer")
        self.deferred.append((ready, wave_output))

    def flush_deferred_slot_releases(self, *, evaluate: bool = False) -> None:
        if evaluate and self.deferred:
            _REAL_EVAL(*[wo for _r, wo in self.deferred])
        while self.deferred:
            ready, _wo = self.deferred.pop(0)
            release = getattr(ready, "release", None)
            if callable(release):
                try:
                    release(synchronize=False)
                except Exception:  # noqa: BLE001 - fake, drain regardless
                    pass
            self.events.append("flush")


class _NoDeferRuntime(_FastpathRuntime):
    """A runtime that CANNOT defer (no defer_slot_release): the fast-path must
    fall back to the shipped blocking fence rather than crash."""

    defer_slot_release = None  # type: ignore[assignment]
    flush_deferred_slot_releases = None  # type: ignore[assignment]


def _fake_gather(selected, bindings, *, group_size, bits, swiglu_limit=None, codec="affine"):
    """Deterministic, non-identity, order-sensitive stand-in for the component-
    bank gather: a stale / dropped / reordered wave shows as a bit difference."""
    ids = mx.array([[float(b.expert)] for b in bindings], dtype=mx.float32)
    return (selected.astype(mx.float32) * 2.0 + ids + 1.0).astype(selected.dtype)


def _inputs(m: int, experts):
    mx.random.seed(101 + m)
    x = (0.5 * mx.random.normal((m, 1, 2))).astype(mx.bfloat16)
    idx = mx.array([[[int(e)]] for e in experts], dtype=mx.int32)
    mx.eval(x, idx)
    return x, idx


def _build_runtime(outcome: str, experts, events):
    if outcome == "all_hit":
        return _FastpathRuntime(events, all_hit_ready=_all_hit_ready(experts, events))
    if outcome == "split":
        pending = _FakePending(events, experts, hit_experts={experts[0]})
        return _FastpathRuntime(events, pending=pending)
    if outcome == "all_miss":
        pending = _FakePending(events, experts, hit_experts=set())
        return _FastpathRuntime(events, pending=pending)
    raise ValueError(outcome)


def _drive(outcome, experts, x, idx, *, fastpath, runtime_cls=None):
    """Run one switch call; return (output, runtime, events).  Counting of host
    syncs is done by the caller via _census; this helper just executes."""
    import os

    events: list[str] = []
    if runtime_cls is _NoDeferRuntime:
        # all-hit only for the no-defer fallback probe.
        runtime = _NoDeferRuntime(
            events, all_hit_ready=_all_hit_ready(experts, events)
        )
    else:
        runtime = _build_runtime(outcome, experts, events)
    if fastpath:
        os.environ[FASTPATH_FLAG] = "1"
    else:
        os.environ.pop(FASTPATH_FLAG, None)
    prev = expert_mlx._run_component_bank_q4
    expert_mlx._run_component_bank_q4 = _fake_gather
    try:
        output = HotExpertSwitchGLU(runtime, 1)(x, idx)
    finally:
        expert_mlx._run_component_bank_q4 = prev
    # Cover the (possibly deferred) wave output, then drain the queue.
    _REAL_EVAL(output)
    flush = getattr(runtime, "flush_deferred_slot_releases", None)
    if callable(flush):
        flush()
    return output, runtime, events


def _census(outcome, experts, x, idx, *, fastpath):
    """Drive one switch call counting blocking mx.eval / async_eval and the
    routing barrier (via the hot.eval_indices bracket).  Returns a dict."""
    import os

    counts = {"eval": 0, "async_eval": 0, "barrier": 0}

    def c_eval(*a, **k):
        counts["eval"] += 1
        return _REAL_EVAL(*a, **k)

    def c_async(*a, **k):
        counts["async_eval"] += 1
        return _REAL_ASYNC(*a, **k)

    @contextmanager
    def c_bracket(stage: str):
        if stage == "hot.eval_indices":
            counts["barrier"] += 1
        with _REAL_BRACKET(stage):
            yield

    events: list[str] = []
    runtime = _build_runtime(outcome, experts, events)
    if fastpath:
        os.environ[FASTPATH_FLAG] = "1"
    else:
        os.environ.pop(FASTPATH_FLAG, None)
    expert_mlx.mx.eval = c_eval
    expert_mlx.mx.async_eval = c_async
    expert_mlx._route_probe.bracket = c_bracket
    expert_mlx._run_component_bank_q4 = _fake_gather
    try:
        output = HotExpertSwitchGLU(runtime, 1)(x, idx)
    finally:
        expert_mlx.mx.eval = _REAL_EVAL
        expert_mlx.mx.async_eval = _REAL_ASYNC
        expert_mlx._route_probe.bracket = _REAL_BRACKET
        expert_mlx._run_component_bank_q4 = _REAL_BANK_Q4
    _REAL_EVAL(output)
    counts["deferred"] = len(runtime.deferred)
    runtime.flush_deferred_slot_releases()
    return counts, output


# ---------------------------------------------------------------------------
# 1. host-sync census: fence removed, exactly one barrier preserved
# ---------------------------------------------------------------------------
def test_all_hit_removes_the_blocking_wave_fence() -> None:
    """All-hit: OFF = routing barrier + 1 blocking wave fence; ON = routing
    barrier ONLY (the fence becomes one deferred slot release)."""
    experts = [0]
    x, idx = _inputs(1, experts)
    off, _ = _census("all_hit", experts, x, idx, fastpath=False)
    on, _ = _census("all_hit", experts, x, idx, fastpath=True)

    assert off["barrier"] == on["barrier"] == 1, (off, on)
    assert off["eval"] == 2, off          # barrier + synchronous_fence
    assert on["eval"] == 1, on            # barrier only
    assert off["deferred"] == 0 and on["deferred"] == 1, (off, on)
    assert on["async_eval"] == 0, on      # all-hit defers, never async-dispatches


def test_split_route_removes_the_per_wave_fences() -> None:
    """Split (1 hit + misses): OFF fences the hit wave and every miss part
    (blocking); ON async-dispatches them and defers the release."""
    experts = [0, 1, 2, 3]
    x, idx = _inputs(4, experts)
    off, _ = _census("split", experts, x, idx, fastpath=False)
    on, _ = _census("split", experts, x, idx, fastpath=True)

    assert off["barrier"] == on["barrier"] == 1, (off, on)
    # OFF: barrier + hit fence + one miss fence = 3 blocking evals.
    assert off["eval"] == 3, off
    assert off["async_eval"] == 0, off
    # ON: barrier ONLY blocks; hit + miss dispatch via async_eval; one deferred.
    assert on["eval"] == 1, on
    assert on["async_eval"] == 2, on
    assert on["deferred"] == 1 and off["deferred"] == 0, (off, on)


def test_all_miss_removes_the_per_wave_fence() -> None:
    experts = [0, 1, 2, 3]
    x, idx = _inputs(4, experts)
    off, _ = _census("all_miss", experts, x, idx, fastpath=False)
    on, _ = _census("all_miss", experts, x, idx, fastpath=True)

    assert off["barrier"] == on["barrier"] == 1, (off, on)
    assert off["eval"] == 2, off          # barrier + one miss fence
    assert on["eval"] == 1, on            # barrier only
    assert on["async_eval"] == 1 and on["deferred"] == 1, on


# ---------------------------------------------------------------------------
# 2. byte-identity: flag off vs on, every route outcome, M=1 and M=4
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize("outcome", ["all_hit", "split", "all_miss"])
def test_switch_output_bitwise_identical_off_vs_on(outcome, m) -> None:
    """The fast-path is a pure fence/release-timing reorder, so the switch
    output must be bitwise-identical off vs on -- for all-hit / split-route /
    all-miss at M=1 (AR) and M=4 (the MTP verify row batch)."""
    if outcome == "split" and m == 1:
        pytest.skip("a mixed hit+miss split is undefined for a single assignment")
    experts = list(range(m))
    x, idx = _inputs(m, experts)

    out_off, _rt_off, _e_off = _drive(outcome, experts, x, idx, fastpath=False)
    out_on, _rt_on, _e_on = _drive(outcome, experts, x, idx, fastpath=True)

    assert out_off.shape == out_on.shape == (m, 1, 1, 2), (out_off.shape, m)
    assert mx.array_equal(out_off, out_on), (
        f"{outcome} M={m}: switch output changed under the fast-path"
    )


def test_fastpath_falls_back_to_fence_when_runtime_cannot_defer() -> None:
    """A runtime without ``defer_slot_release`` must NOT engage the fast-path
    (``_fastpath_can_defer`` guards it): the flag-on run fences exactly like the
    shipped path and is byte-identical -- never an AttributeError."""
    experts = [0]
    x, idx = _inputs(1, experts)

    out_off, _rt, _e = _drive(
        "all_hit", experts, x, idx, fastpath=False, runtime_cls=_NoDeferRuntime
    )
    out_on, rt_on, _e2 = _drive(
        "all_hit", experts, x, idx, fastpath=True, runtime_cls=_NoDeferRuntime
    )
    assert mx.array_equal(out_off, out_on)
    # No defer was queued -> the flag was correctly ignored (fenced fallback).
    assert rt_on.deferred == []


# ---------------------------------------------------------------------------
# 3. real streamed runtime: flag-driven deferral is byte-identical + slot-safe
# ---------------------------------------------------------------------------
def test_integrated_streamed_fastpath_matches_fenced_bitwise(tmp_path: Path) -> None:
    """On a REAL streamed runtime whose config is the DSV4.1 fenced default
    (``deferred_pin_release=False``, ``split_route_release="fenced"``), the
    fast-path flag must produce bitwise-identical logits AND actually hold its
    slot pins until the covering flush (proving the blocking fence was removed,
    not just skipped) -- the slot-safety guarantee the fake bank cannot give.

    This mirrors ``test_deferred_split_route_release_matches_fenced_bitwise`` but
    drives the deferral through ``MTPLX_DSV41_SWITCH_FASTPATH`` instead of the
    config field."""
    import os

    from mtplx.expert_runtime import ExpertStreamingConfig, ExpertStreamingRuntime
    from mtplx.models.expert_mlx import make_mlx_slot_buffer_allocator
    from mtplx.resident_loader import construct_resident_model
    from tests.test_streamed_models import _integrated_hy3_artifact

    root, config, spec, manifest_path = _integrated_hy3_artifact(tmp_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes

    def run(fastpath: bool):
        # DSV4.1 shipped default: fenced.  Only the env flag promotes deferral.
        stream_config = ExpertStreamingConfig(
            model_key=spec.key,
            memory_limit_bytes=fixed + spec.persistent_cache_bytes(1),
            max_live_kv_tokens=0,
            runtime_reserve_bytes=0,
            deferred_pin_release=False,
            split_route_release="fenced",
        )
        runtime = ExpertStreamingRuntime.open(
            root,
            manifest_path,
            stream_config,
            spec=spec,
            buffer_allocator=make_mlx_slot_buffer_allocator(
                stream_config.memory_plan(spec), spec
            ),
            device_synchronize=mx.synchronize,
            apply_memory_cap=False,
        )
        if fastpath:
            os.environ[FASTPATH_FLAG] = "1"
        else:
            os.environ.pop(FASTPATH_FLAG, None)
        try:
            resident = construct_resident_model(root, runtime, config=config)
            logits = resident.model(mx.array([[1]], dtype=mx.int32))
            mx.eval(logits)
            held = runtime.snapshot(mx_module=mx)["slots"]["pins"]
            runtime.flush_deferred_slot_releases(evaluate=True)
            drained = runtime.snapshot(mx_module=mx)["slots"]["pins"]
            return logits, held, drained
        finally:
            os.environ.pop(FASTPATH_FLAG, None)
            runtime.close()

    fenced_logits, fenced_held, fenced_drained = run(False)
    fast_logits, fast_held, fast_drained = run(True)

    assert fenced_drained == 0
    assert fast_held > 0, "fast-path must hold slot pins until the covering flush"
    assert fast_drained == 0
    assert mx.array_equal(fenced_logits, fast_logits).item()
