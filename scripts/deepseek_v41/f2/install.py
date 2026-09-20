"""F2b install: wire the host ring + speculative pool (coordinator + workers) + intercepted
reader + per-layer run wrappers, as the LAST step of ``observe_seed_prefill`` (run_full.py:
737, after ``prime_model``). The runtime keeps ``prefetch_slots == 0``; nothing in the
pinned runtime sources is edited (the reader intercept is DERIVED from the retained
``plane_lane.bind_reader`` and the run wrappers are outer wraps).

Install-point safety (projection_install.py): ``install_model`` validates
``switch._run.__func__ is PackedDecode.run`` and rebinds it to the scheduled run
(``self.issue_next()``) during growth_transition; ``prime_model`` (:124-130) and
``verify_retirement`` (:133-151, post-request) do NOT re-validate ``switch._run`` or the
reader. The wrapper reuses the runner instance (issue_next stays wired) and calls the
original scheduled run whose ``mx.eval(indices)`` becomes a no-op after the wrapper's
``mx.eval(indices, merged)``. Ranking/enqueue run on the pool's coordinator thread, so the
measured main thread only copies the prediction and hands off one item.
"""
from __future__ import annotations

import os

import numpy as np
import mlx.core as mx

from .host_ring import PLANE_OFFSETS, WEIGHT_RECORD_BYTES, F2bCounters, HostRing
from .predictor import (
    FIRST_TARGET_LAYER,
    GatePredictor,
    HostGatePredictor,
    rank_targets,
    select_prefetch_sources,
    words_from_mx,
)
from .reader_intercept import install_intercept
from .speculative import SpeculativePool

_PLANE_NAMES = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
_HIDDEN = 5120


def _live_plane_length(runtime, layers) -> int:
    """The decode weight-plane VIEW length (weights only) from a live persistent slot.
    Asserts the three planes are equal-length and sum to the decode record bytes (fix 2)."""
    slot = runtime.slots._persistent[(layers[0], 0)]
    view = slot.buffer
    lengths = tuple(int(len(view.component_view(name))) for name in _PLANE_NAMES)
    if len(set(lengths)) != 1:
        raise RuntimeError(f"F2b requires equal weight-plane lengths; got {lengths}")
    length = lengths[0]
    if 3 * length != WEIGHT_RECORD_BYTES:
        raise RuntimeError(f"3 x plane length {3 * length} != decode record bytes {WEIGHT_RECORD_BYTES}")
    rep = getattr(runtime, "_representative_record_bytes", None)
    if rep is not None and int(rep) != 3 * length:
        raise RuntimeError(f"3 x plane length {3 * length} != runtime record bytes {rep}")
    return length


def _open_direct_fd(reader) -> int:
    """A private descriptor on experts.bin with the retained reader's own flags
    (O_RDONLY | O_CLOEXEC | O_NOFOLLOW, F_NOCACHE). Kept open for the process lifetime."""
    import fcntl
    from pathlib import Path

    if not getattr(reader, "bypass_page_cache", False):
        raise RuntimeError("F2b direct I/O requires the F_NOCACHE reader (bypass_page_cache)")
    path = Path(reader.root) / "experts.bin"
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
    return fd


def install(target, *, ring_records: int = 32, workers: int = 3, k: int = 3,
            first_target: int = FIRST_TARGET_LAYER, predictor_mode: str | None = None,
            direct_io: bool = True):
    runtime = target._mtplx_expert_runtime
    if runtime.config.prefetch_slots != 0:
        raise RuntimeError("F2b requires the runtime's own prefetch ring OFF (prefetch_slots==0)")
    # Predictor lane chosen ONCE here (never per call): 'lean' = fused compiled tape
    # (default), 'native' = the prior gate-prefix path (kept for A/B), 'cpu' = host-side
    # prediction off the GPU/barrier entirely (F2d).
    if predictor_mode is None:
        predictor_mode = os.environ.get("MTPLX_DSV41_F2B_PREDICTOR", "lean")
    predictor_mode = str(predictor_mode)
    if predictor_mode not in ("lean", "native", "cpu"):
        raise RuntimeError(
            f"MTPLX_DSV41_F2B_PREDICTOR must be 'lean', 'native' or 'cpu'; got {predictor_mode!r}"
        )
    layers = sorted(int(x) for x in runtime.spec.routed_layer_indices)
    switches = {L: target.model.layers[L].mlp.switch_mlp for L in layers}

    plane_len = _live_plane_length(runtime, layers)
    plane_specs = tuple((off, plane_len) for off in PLANE_OFFSETS)     # equal-length planes
    counters = F2bCounters()
    ring = HostRing(records=ring_records, planes=len(PLANE_OFFSETS),
                    plane_bytes=plane_len, counters=counters)

    sources = set(select_prefetch_sources(layers, first_target=first_target))   # 3..38
    target_layers = sorted({L + 1 for L in sources})                            # 4..39
    manifest = runtime.manifest
    expert_count = int(runtime.spec.expert_count)
    # Precompute sidecar offsets for every (target, expert) once (no per-call manifest hit).
    off_of = {
        (t, e): int(manifest.record(t, e).sidecar_offset)
        for t in target_layers for e in range(expert_count)
    }

    def plan_fn(target_layer, scores):
        """Coordinator thread: resident-filter (racy read is fine) + rank + offset map."""
        bank = runtime._banks[target_layer]
        resident = {e for e in bank._slot_to_expert if e is not None}
        base = off_of

        def skip(expert):
            if expert in resident:
                return True
            return ring.has(base[(target_layer, expert)])

        if k == 0:                       # predictor-cost isolation arm: evaluate, issue nothing
            return []
        return [base[(target_layer, e)] for e in rank_targets(scores, skip, k)]

    if not 1 <= int(first_target) <= max(layers) or not 0 <= int(k) <= 8:
        raise RuntimeError(f"F2b first_target/k out of range: first_target={first_target} k={k}")

    # 'cpu' mode: materialize every TARGET gate's params on the host ONCE (validates
    # score_func here, before any generation) and bind the coordinator's host score fn.
    # ``predictor_host_bytes`` (0 for lean/native) is returned + printed so the launcher can
    # charge it to memory admission.
    host_predictors: dict[int, HostGatePredictor] = {}
    predictor_host_bytes = 0
    cpu_score_fn = None
    if predictor_mode == "cpu":
        for L in sorted(sources):
            hp = HostGatePredictor(target.model.layers[L + 1].mlp.gate)
            host_predictors[L + 1] = hp
            predictor_host_bytes += hp.host_bytes

        def cpu_score_fn(tgt, rows, words):
            return host_predictors[int(tgt)].scores_from_words(words, rows)

    direct_fd = _open_direct_fd(runtime.reader) if direct_io else None
    pool = SpeculativePool(runtime.reader, ring, plane_specs=plane_specs,
                           plan_fn=plan_fn, workers=workers, direct_fd=direct_fd,
                           cpu_score_fn=cpu_score_fn)

    # The shared witness threading.local lives on any runner's PartExecutor; grab it
    # BEFORE wrapping switch._run (the wrapper replaces switch._run.__self__).
    a_runner = getattr(switches[layers[0]]._run, "__self__", None)
    if a_runner is None or not hasattr(a_runner, "executor"):
        raise RuntimeError("installed packed runner not found on switch._run")
    local = a_runner.executor.local
    intercept = install_intercept(runtime.reader, local, ring)

    wrapped_layers = sorted(L for L in layers if L in sources or L >= first_target)  # 3..39
    predictors = {}
    for L in wrapped_layers:
        is_source = L in sources
        is_target = L >= first_target
        target_layer = (L + 1) if is_source else None
        if predictor_mode == "cpu":
            _wrap_run_cpu(switches[L], pool, own=L, is_source=is_source, is_target=is_target,
                          target_layer=target_layer,
                          dim=(host_predictors[target_layer].dim if is_source else 0))
        else:
            pred = GatePredictor(target.model.layers[L + 1].mlp.gate, mode=predictor_mode) if is_source else None
            predictors[L] = pred
            _wrap_run(switches[L], pool, own=L, is_source=is_source,
                      is_target=is_target, predictor=pred, target_layer=target_layer)

    target._f2b = {"ring": ring, "pool": pool, "counters": counters}
    return {
        "installed": True, "ring_records": int(ring_records), "workers": int(workers),
        "plane_length": int(plane_len), "sources": sorted(sources),
        "wrapped_layers": wrapped_layers, "predictor_mode": predictor_mode,
        "predictor_host_bytes": int(predictor_host_bytes),
        "k": int(k), "first_target": int(first_target), "direct_io": bool(direct_io), **intercept,
    }


def _wrap_run(switch, pool, *, own, is_source, is_target, predictor, target_layer):
    original_run = switch._run                       # bound scheduled run (issue_next variant)

    def wrapped(x, indices, *, shared_work):
        if is_source:
            merged = predictor.merged(x)             # reshape rides the predictor's tape
            mx.eval(indices, merged)                 # THE routing barrier (indices + prediction)
        elif is_target:
            mx.eval(indices)                         # fix 5: complete the barrier BEFORE noting imminent
            merged = None
        else:
            merged = None
        if is_target:
            pool.note_demand_imminent(own)           # window-stop this layer (epoch++)
        result = original_run(x, indices, shared_work=shared_work)  # its mx.eval(indices) no-ops
        if is_source:
            # Off-main-thread hand-off: copy the evaluated prediction (freeing the device
            # array) and submit ONE item; the coordinator ranks + enqueues (fix 4).
            epoch = pool.current_epoch(target_layer)
            scores = np.asarray(merged, dtype=np.float32).copy()
            pool.submit_prediction(target_layer, epoch, scores)
        return result

    switch._run = wrapped


def _wrap_run_cpu(switch, pool, *, own, is_source, is_target, target_layer, dim):
    """'cpu' mode wrapper: the routing barrier is exactly ``mx.eval(indices)`` -- NO
    predictor op and no extra eval output, so the predictor's f32 upcast + GEMM leave the
    barrier entirely.  After the scheduled run returns, a source layer's main-thread work is
    the MINIMUM: copy the raw bf16 router-input words out of ``x`` into a fresh uint16 array
    (``words_from_mx``; the coordinator holds the only reference, so the next call for this
    layer cannot overwrite it -- no timing/sequence assumption) and hand
    ``(target, epoch, rows, words)`` to the coordinator with ONE ``queue.put``.  The upcast +
    GEMM + sqrtsoftplus run on the coordinator thread, off the measured path and off the GPU.
    """
    original_run = switch._run                       # bound scheduled run (issue_next variant)

    def wrapped(x, indices, *, shared_work):
        mx.eval(indices)                             # THE routing barrier (no predictor op)
        if is_target:
            pool.note_demand_imminent(own)           # window-stop this layer (epoch++), after the barrier
        result = original_run(x, indices, shared_work=shared_work)  # its mx.eval(indices) no-ops
        if is_source:
            epoch = pool.current_epoch(target_layer)
            words = words_from_mx(x)                  # fresh per-call copy (~1 us; provably race-safe)
            pool.submit_words(target_layer, epoch, words.size // dim, words)
        return result

    switch._run = wrapped


def apply_switch_interval_from_env() -> dict:
    """Optional GIL hand-off tuning, applied ONCE at install (never per call).

    The retained lane issues every expert read from Python threads (part executors ->
    fanout pool) that need the GIL to reach ``os.preadv``; the main thread holds the GIL
    through its host work (hit-expert graph build) right after it submits the reads, and
    CPython only forces a hand-off after ``sys.getswitchinterval()`` (default 5 ms -- longer
    than a whole layer call). ``MTPLX_DSV41_GIL_SWITCH_S`` (seconds, e.g. 0.00005) lets the
    reader threads issue their I/O while the main thread is still busy. Output-exact: it
    changes thread scheduling only.
    """
    import sys

    raw = os.environ.get("MTPLX_DSV41_GIL_SWITCH_S")
    if not raw:
        return {"gil_switch_interval_s": sys.getswitchinterval(), "gil_switch_interval_set": False}
    value = float(raw)
    if not 1e-6 <= value <= 0.005:
        raise RuntimeError(f"MTPLX_DSV41_GIL_SWITCH_S out of range [1e-6, 0.005]: {value}")
    sys.setswitchinterval(value)
    return {"gil_switch_interval_s": sys.getswitchinterval(), "gil_switch_interval_set": True}


def install_from_env(target) -> dict:
    """Called from the staged ``observe_seed_prefill`` after ``prime_model``. No-op unless
    ``MTPLX_DSV41_F2B == '1'``. Registers an atexit dump of the counters (once, after
    decode) to ``MTPLX_DSV41_F2B_COUNTERS``."""
    gil = apply_switch_interval_from_env()
    if os.environ.get("MTPLX_DSV41_F2B") != "1":
        report = {"installed": False, "reason": "MTPLX_DSV41_F2B != 1", **gil}
        _print_install_report(report)
        return report
    report = install(
        target,
        ring_records=int(os.environ.get("MTPLX_DSV41_F2B_RECORDS", "32")),
        workers=int(os.environ.get("MTPLX_DSV41_F2B_WORKERS", "3")),
        k=int(os.environ.get("MTPLX_DSV41_F2B_K", "3")),
        first_target=int(os.environ.get("MTPLX_DSV41_F2B_FIRST_TARGET", str(FIRST_TARGET_LAYER))),
        direct_io=os.environ.get("MTPLX_DSV41_F2B_DIRECT_IO", "1") != "0",
    )
    counters_path = os.environ.get("MTPLX_DSV41_F2B_COUNTERS")
    if counters_path:
        import atexit

        atexit.register(dump_counters, target, counters_path)
    report.update(gil)
    _print_install_report(report)
    return report


def _print_install_report(report: dict) -> None:
    """One provenance line in the guard log, printed once at install."""
    import json

    print("F2B_INSTALL " + json.dumps(report, sort_keys=True, default=str), flush=True)


def dump_counters(target, path) -> dict:
    import json

    state = getattr(target, "_f2b", None)
    data = state["counters"].as_dict() if state else {"installed": False}
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    return data
