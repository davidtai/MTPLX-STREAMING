"""F2b install: wire the host ring + speculative pool + intercepted reader + per-layer
run wrappers, as the LAST step of ``observe_seed_prefill`` (run_full.py:737, after
``prime_model``). The runtime keeps ``prefetch_slots == 0``; nothing in the pinned
runtime sources is edited (the reader intercept is DERIVED from the retained
``plane_lane.bind_reader`` and the run wrappers are outer wraps).

Install-point safety (projection_install.py): ``install_model`` validates
``switch._run.__func__ is PackedDecode.run`` and rebinds it to the scheduled run
(``self.issue_next()``) during growth_transition; ``prime_model`` (:124-130) and
``verify_retirement`` (:133-151, post-request) do NOT re-validate ``switch._run`` or the
reader. So wrapping ``switch._run`` and re-binding the reader after ``prime_model`` trips
no later check. The wrapper reuses the runner instance, so ``runner.issue_next`` stays
wired, and it calls the original scheduled run whose ``mx.eval(indices)`` becomes a
no-op after the wrapper's ``mx.eval(indices, merged)`` (barriers per layer unchanged).
"""
from __future__ import annotations

import os

import mlx.core as mx

from .host_ring import PLANE_OFFSETS, F2bCounters, HostRing
from .predictor import FIRST_TARGET_LAYER, GatePredictor, rank_targets, select_prefetch_sources
from .reader_intercept import install_intercept
from .speculative import SpeculativePool

_PLANE_NAMES = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
_HIDDEN = 5120


def _live_plane_lengths(runtime, layers) -> tuple[int, int, int]:
    """Read the decode plane VIEW lengths (weights only) from a live persistent slot --
    never hard-coded (scales are resident, so the weight plane is what the reader fills)."""
    slot = runtime.slots._persistent[(layers[0], 0)]
    view = slot.buffer
    return tuple(int(len(view.component_view(name))) for name in _PLANE_NAMES)


def install(target, *, ring_records: int = 32, workers: int = 3, k: int = 3,
            first_target: int = FIRST_TARGET_LAYER):
    runtime = target._mtplx_expert_runtime
    if runtime.config.prefetch_slots != 0:
        raise RuntimeError("F2b requires the runtime's own prefetch ring OFF (prefetch_slots==0)")
    layers = sorted(int(x) for x in runtime.spec.routed_layer_indices)
    switches = {L: target.model.layers[L].mlp.switch_mlp for L in layers}

    plane_lengths = _live_plane_lengths(runtime, layers)
    plane_specs = tuple(zip(PLANE_OFFSETS, plane_lengths))     # [(0,glen),(6266880,ulen),(12533760,dlen)]
    counters = F2bCounters()
    ring = HostRing(records=ring_records, planes=len(plane_specs),
                    plane_bytes=max(plane_lengths), counters=counters)
    pool = SpeculativePool(runtime.reader, ring, workers=workers)

    # The shared witness threading.local lives on any runner's PartExecutor; grab it
    # BEFORE wrapping switch._run (the wrapper replaces switch._run.__self__).
    a_runner = getattr(switches[layers[0]]._run, "__self__", None)
    if a_runner is None or not hasattr(a_runner, "executor"):
        raise RuntimeError("installed packed runner not found on switch._run")
    local = a_runner.executor.local
    intercept = install_intercept(runtime.reader, local, ring)

    manifest = runtime.manifest
    sources = set(select_prefetch_sources(layers, first_target=first_target))   # 3..38
    wrapped_layers = sorted(L for L in layers if L in sources or L >= first_target)  # 3..39
    for L in wrapped_layers:
        is_source = L in sources
        target_layer = L + 1 if is_source else None
        predictor = GatePredictor(target.model.layers[target_layer].mlp.gate) if is_source else None
        _wrap_run(switches[L], runtime, pool, ring, manifest, plane_specs,
                  own=L, is_target=(L >= first_target), predictor=predictor,
                  target_layer=target_layer, k=k)

    target._f2b = {"ring": ring, "pool": pool, "counters": counters}
    return {
        "installed": True, "ring_records": int(ring_records), "workers": int(workers),
        "plane_lengths": list(plane_lengths), "sources": sorted(sources),
        "wrapped_layers": wrapped_layers, **intercept,
    }


def _wrap_run(switch, runtime, pool, ring, manifest, plane_specs, *, own, is_target,
              predictor, target_layer, k):
    original_run = switch._run                       # bound scheduled run (issue_next variant)
    target_bank = runtime._banks[target_layer] if target_layer is not None else None

    def wrapped(x, indices, *, shared_work):
        if predictor is not None:
            merged = predictor.merged(x.reshape(-1, _HIDDEN))
            mx.eval(indices, merged)                 # THE routing barrier (indices + prediction)
        else:
            merged = None                            # target-only layer: original run owns the barrier
        if is_target:
            pool.note_demand_imminent(own)           # stop starting new speculative planes for this layer
        result = original_run(x, indices, shared_work=shared_work)  # its mx.eval(indices) is a no-op
        if predictor is not None:
            _rank_and_enqueue(merged, target_bank, pool, ring, manifest, plane_specs, target_layer, k)
        return result

    switch._run = wrapped


def _rank_and_enqueue(merged, target_bank, pool, ring, manifest, plane_specs, target_layer, k):
    resident = set(target_bank.resident_experts)
    ring_has = ring.has

    def skip(expert):
        if expert in resident:
            return True
        return ring_has(int(manifest.record(target_layer, expert).sidecar_offset))

    for expert in rank_targets(merged, skip, k):
        record = manifest.record(target_layer, expert)
        pool.enqueue_record(target_layer, int(record.sidecar_offset), plane_specs)


def install_from_env(target) -> dict:
    """Called from the staged ``observe_seed_prefill`` after ``prime_model``. No-op unless
    ``MTPLX_DSV41_F2B == '1'``. Registers an atexit dump of the aggregate counters to
    ``MTPLX_DSV41_F2B_COUNTERS`` (once, after decode)."""
    if os.environ.get("MTPLX_DSV41_F2B") != "1":
        return {"installed": False, "reason": "MTPLX_DSV41_F2B != 1"}
    report = install(
        target,
        ring_records=int(os.environ.get("MTPLX_DSV41_F2B_RECORDS", "32")),
        workers=int(os.environ.get("MTPLX_DSV41_F2B_WORKERS", "3")),
    )
    counters_path = os.environ.get("MTPLX_DSV41_F2B_COUNTERS")
    if counters_path:
        import atexit

        atexit.register(dump_counters, target, counters_path)
    return report


def dump_counters(target, path) -> dict:
    """Dump the aggregate F2b counters ONCE after decode to ``path``."""
    import json

    state = getattr(target, "_f2b", None)
    data = state["counters"].as_dict() if state else {"installed": False}
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    return data
