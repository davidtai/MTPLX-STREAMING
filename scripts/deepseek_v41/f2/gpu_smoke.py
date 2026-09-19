"""WRITE-ONLY bounded Metal smoke for the F2 prefetch lane's execution body.

DO NOT RUN THIS FROM A WORKER. It is launched only under the parent-held GPU/service
guard (scripts/deepseek_v41/gpu_window.sh, which acquires
/tmp/mtplx-gpu-exclusive.lock, unloads the production server and sets
``_GPU_WINDOW_LOCKED=1``), in the style of the ridge-prefetch screen
(docs/deepseek-v41/receipts/ridge-prefetch-20260919/v2/run_screen.py).

Why this exists (the ONE interaction the CPU seam tests cannot cover):
``PrefetchDecode.run``'s execution body drives the plane-split reader
(``plane_lane_prefetch.bind_priority_reader``) and the Metal gather (``PackedOps``).
Both are pinned to production geometry:
  * ``bind_priority_reader`` reads three WEIGHT planes of the production ``experts.bin``
    sidecar at offsets (0, 6,266,880, 12,533,760) with a 17,694,720-byte record
    (plane_lane_prefetch._PLANE_OFFSETS; sources/packed/packed_storage.py WEIGHTS).
    A CPU-buildable tiny record is 6,912 bytes, so those offsets are out of bounds --
    the plane-split read has no valid tiny artifact.
  * ``PackedOps`` builds ``mx.fast.metal_kernel`` projections
    (sources/packed/paired_kernels.py ``make_projection``) that only admit the
    (2304,5120)/(5120,2304) geometry -- Metal-only, no CPU backend.
Materialising production-geometry (5120/2304) records to run this on CPU is both
impossible without Metal and CPU/IO-heavy enough to perturb a concurrent measurement
window, so it is deferred here rather than faked on CPU.

This smoke asserts, on 3 routed layers WITH attention (a real compute window between
layers) at the retained M<=8 verify shape, the properties the CPU tests prove at the
runtime/predictor seam, now end to end through the packed execution body:
  1. OUTPUT IDENTITY: every layer's routed output is byte-identical with the prefetch
     lane installed vs. the stock ``PackedDecode`` control (the gather uses the TRUE
     indices; the ring only warms the cache).
  2. ENGAGEMENT: the aggregate prefetch counters move -- prefetch_issued (verify),
     prefetch_committed, prefetch_hit_on_true_route / prefetch_first_consumption_hits,
     and (when a demanded read is still in flight) prefetch_awaited_inflight -- read
     ONCE after the forwards from ``runtime.counters`` (existing statistics; AGENTS.md).
  3. BARRIER PARITY: the generation-thread ``mx.eval`` count per source layer-call is
     unchanged vs. the control -- ``Issue.prepare`` rides the indices barrier and
     ``Issue()`` / ``prefetch_experts`` add no host sync.

The runtime construction (the strict-allocator libmlx, the 3-layer sidecar slice with
attention, the packed scales) is the staged tree's, reused unchanged; this module
supplies only the lane install and the parity assertions. Fill ``build_three_layer_runtime``
against the staged 3-layer harness before the run.
"""
from __future__ import annotations

import os
import threading


def _require_guard() -> None:
    if os.environ.get("_GPU_WINDOW_LOCKED") != "1":
        raise RuntimeError(
            "gpu_smoke requires the parent-held GPU/service guard; launch it under "
            "scripts/deepseek_v41/gpu_window.sh, never directly"
        )


def build_three_layer_runtime():  # pragma: no cover - staged-tree GPU harness
    """Return ``(runtime, model, switches, scales_by_layer, source_layers, targets)``
    for a 3-consecutive-routed-layer slice WITH attention, built exactly as the retained
    run builds the full model (strict-allocator libmlx, transition-window + ring R config
    via ``full_config.FullPrefetchConfig``, packed scales per layer). This is the staged
    tree's construction; it is not reproduced here."""
    raise NotImplementedError(
        "wire against the staged 3-layer harness (ridge-prefetch run_screen.py/probe.py)"
    )


def _count_main_thread_evals(fn):
    """Run ``fn`` and return the number of generation-thread ``mx.eval`` calls it makes."""
    import mlx.core as mx

    main = threading.main_thread()
    count = {"n": 0}
    real = mx.eval

    def counting(*a, **k):
        if threading.current_thread() is main:
            count["n"] += 1
        return real(*a, **k)

    mx.eval = counting
    try:
        result = fn()
    finally:
        mx.eval = real
    return result, count["n"]


def run_smoke():  # pragma: no cover - runs only under the GPU window guard
    _require_guard()
    import mlx.core as mx

    from .plane_lane_prefetch import PackedOps, install as install_lane
    from .run_full_install import install_f2_growth

    # 1) Control: stock PackedDecode on the 3-layer slice; capture per-layer outputs
    #    and the per-source-layer generation-thread eval count.
    runtime, model, switches, scales, source_layers, targets = build_three_layer_runtime()
    control_ops = {layer: PackedOps(scales[layer]) for layer in switches}
    install_lane(runtime, switches, control_ops, prefetch_sources={})  # no ring issue
    control_out = _run_verify_forwards(runtime, model)   # {layer: output array}
    control_evals = _per_source_layer_eval_counts(runtime, model, source_layers)
    baseline_counters = _snapshot_prefetch_counters(runtime)

    # 2) Candidate: the F2 prefetch lane on the SAME slice + inputs (fresh runtime).
    runtime2, model2, switches2, scales2, source_layers2, targets2 = build_three_layer_runtime()
    install_f2_growth(runtime2, switches2, scales2, model=model2)
    candidate_out = _run_verify_forwards(runtime2, model2)
    candidate_evals = _per_source_layer_eval_counts(runtime2, model2, source_layers2)
    counters = _snapshot_prefetch_counters(runtime2)

    # 1. output identity, layer by layer.
    for layer in control_out:
        assert mx.array_equal(control_out[layer], candidate_out[layer]), (
            f"layer {layer}: prefetch changed the routed output"
        )
    # 2. engagement (existing counters, read once).
    assert counters["prefetch_issued"] > baseline_counters["prefetch_issued"]
    assert counters["prefetch_committed"] >= 1
    assert counters["prefetch_bytes"] >= counters["prefetch_committed"] * 17_694_720
    # 3. barrier parity per source layer-call.
    for layer in source_layers2:
        assert candidate_evals[layer] == control_evals[layer], (
            f"layer {layer}: prefetch added host syncs "
            f"({candidate_evals[layer]} vs {control_evals[layer]})"
        )
    return {"counters": counters, "layers": sorted(control_out)}


def _run_verify_forwards(runtime, model):  # pragma: no cover
    raise NotImplementedError("staged 3-layer M<=8 verify forwards")


def _per_source_layer_eval_counts(runtime, model, source_layers):  # pragma: no cover
    raise NotImplementedError("wrap each source switch._run and count main-thread mx.eval")


def _snapshot_prefetch_counters(runtime):  # pragma: no cover
    c = runtime.counters
    return {
        "prefetch_issued": c.prefetch_issued,
        "prefetch_issued_verify": c.prefetch_issued_verify,
        "prefetch_committed": c.prefetch_committed,
        "prefetch_awaited_inflight": c.prefetch_awaited_inflight,
        "prefetch_wasted": c.prefetch_wasted,
        "prefetch_bytes": c.prefetch_bytes,
        "prefetch_first_consumption_hits": c.prefetch_first_consumption_hits,
    }


if __name__ == "__main__":  # pragma: no cover
    _require_guard()
    import json

    print("F2_GPU_SMOKE", json.dumps(run_smoke()), flush=True)
