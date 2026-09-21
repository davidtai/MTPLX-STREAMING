"""F39: the tcq3 growth-transition mechanics — the phase-boundary step that turns the prefill-loaded bank into the
decode bank and installs the decode-verify lane, mirroring ``packed_phase.install_growth`` but for the tcq3 slot.

The mxfp4 transition retires the raw per-group scales, installs the 3.09 GB packed-scale bank, rebinds the mxfp4
dispatch, keeps 17,694,720-B weight-only slots, and installs the mxfp4 plane lane.  For tcq3, three of those steps
COLLAPSE (validated against the real F38 bank in the tests):

  * raw-scale retirement -> **no-op**: a tcq3 record is code + routs only (``transcode_bank.record_layout`` has no
    ``.scales`` segment), so there are no per-group raw scales to release (``retire_scales_tcq3`` returns 0).
  * packed-scale install -> **already resident**: the whole-bank routs (298,844,160 B) are loaded ONCE at
    construction (``tcq_runtime.load_resident_routs``); the transition installs no per-layer scale owner.
  * slot bytes -> **whole record** (13,290,496 B), not the mxfp4 17,694,720-B weight code.

What stays: the codec-agnostic quiescence + row growth to the admitted decode capacity, then the tcq3 reader
(``tcq.install.bind_tcq_reader``, one contiguous record read) and the tcq3 decode-verify lane
(``tcq.install.install`` — TcqPackedOps: stride kernel + t128 + routs).

The pure accounting (``tcq3_growth_report`` / ``retire_scales_tcq3``) is unit-tested against the real records.  The
runtime bank manipulation in ``transition`` is exercised only in a guarded GPU window (deferred until the bank is
complete and the GPU is free) — the same boundary as every other tcq3 runtime path.
"""
from __future__ import annotations

SLOT_BYTES = 13_290_496                 # whole tcq3 record per decode slot (code + routs)
RESIDENT_ROUT_BYTES = 298_844_160       # whole-bank routs, resident (load_resident_routs)
SCALE_BYTES_MXFP4 = 1_105_920           # mxfp4 per-record raw scale bytes (retired in the mxfp4 transition)
N_LAYERS = 40
TRANSIENT_SLOTS = 48


def retire_scales_tcq3(bank, *, mx=None) -> int:
    """tcq3 raw-scale retirement: a no-op.  Returns 0 released bytes.

    A tcq3 slot holds the whole record; there is no separate ``.scales`` component to remove.  If a bank is passed
    that still exposes ``.scales`` arrays it is NOT a tcq3 bank — raise, so a misrouted mxfp4 bank fails loudly here
    rather than silently skipping a real retirement (AGENTS.md "fail once, clearly").
    """
    arrays = getattr(bank, "arrays", None)
    if arrays is not None and any(str(name).endswith(".scales") for name in arrays):
        raise RuntimeError("retire_scales_tcq3 called on a bank with raw .scales components (not a tcq3 bank)")
    return 0


def tcq3_growth_report(capacity: int, old_capacity: int) -> dict:
    """The transition report for the tcq3 lane, mirroring the mxfp4 ``state`` dict with tcq3 accounting.

    Pure: no scales installed (routs resident), whole-record slots, released raw scales == 0.
    """
    physical_after = (capacity * N_LAYERS + TRANSIENT_SLOTS) * SLOT_BYTES
    return dict(
        phase="installed",
        prefill_slots_per_layer=old_capacity,
        decode_slots_per_layer=capacity,
        decode_weight_record_bytes=SLOT_BYTES,
        source_record_bytes=SLOT_BYTES,               # tcq3 record IS the slot (code + routs)
        resident_packed_scales_bytes=0,
        resident_rout_bytes=RESIDENT_ROUT_BYTES,
        raw_scale_backing_released_bytes=0,
        physical_allocated_bytes=physical_after,
        persistent_cache_bytes=capacity * N_LAYERS * SLOT_BYTES,
        timing_scope=("tcq3 decode transition: no raw-scale retirement (none exist), routs already resident, "
                      "whole-record slots; row growth + tcq3 decode-verify lane install, inside decode wall time"),
    )


def install_growth_tcq3(model, capacity, *, mx, admission, routs=None, tables=None, old_capacity=None):
    """tcq3 analogue of ``packed_phase.install_growth``, grown straight to the FINAL decode ``capacity``.

    ``routs`` (tcq_runtime.load_resident_routs) and ``tables`` (warp tables) are loaded internally from the tcq3
    artifact if not supplied.  Returns ``(transition, report)``.  The returned ``transition()`` performs, at the
    quiescent post-prefill boundary: quiescence, whole-record slot sizing + row growth to ``capacity``
    (codec-agnostic), a tcq3 plan update, the tcq3 reader binding, and the tcq3 decode-verify lane install.  Because
    it grows to the final capacity, the retained ``grow_rows`` step is a no-op for tcq3 (routed off).

    RUNTIME (GPU-deferred): the MLX bank manipulation runs only inside a guarded window on the complete bank; the
    accounting it reports (``tcq3_growth_report``) and the no-op retirement are validated on CPU against real records.
    """
    from dataclasses import replace
    rt = model._mtplx_expert_runtime
    layers = tuple(sorted(rt.spec.routed_layer_indices))
    if old_capacity is None:
        old_capacity = rt.plan.slots_per_layer
    report = tcq3_growth_report(capacity, old_capacity)

    def transition():
        import threading
        from mtplx.models.expert_mlx import MlxComponentSlot
        from mtplx.expert_slots import _PhysicalSlot
        import tcq.install as tcq_install
        import tcq_runtime as R

        nonlocal routs, tables
        if routs is None or tables is None:
            _manifest, routs, tables = tcq_install._load_tcq_resources(rt)
        routs_by_layer = {layer: routs for layer in layers}
        pool, allocator = rt.slots, rt.slots._allocator
        rt.flush_deferred_slot_releases(evaluate=True)
        rt._drain_prefetch_loads()
        pool._drain_completion_fences()
        mx.synchronize()
        # switches per layer, derived exactly as packed_phase.install_growth does
        switches = {layer: model.model.layers[layer].mlp.switch_mlp for layer in layers}
        # 1) raw-scale retirement is a no-op for every tcq3 bank (routs resident, whole-record slots)
        released = retire_scales_tcq3(allocator.banks["transient", -1], mx=mx)
        for layer in layers:
            released += retire_scales_tcq3(allocator.banks["persistent", layer], mx=mx)
        # 2) whole-record slot bytes + grow rows old_capacity..capacity (codec-agnostic slot machinery)
        for physical in (*pool._persistent.values(), *pool._transient):
            physical.buffer.nbytes = SLOT_BYTES
        for layer in layers:
            bank = allocator.banks["persistent", layer]
            for row in range(old_capacity, capacity):
                label = f"layer-{layer}-persistent-{row}"
                buffer = MlxComponentSlot(bank, row, label=label)
                allocator.slots[label] = buffer
                pool._persistent[layer, row] = _PhysicalSlot(label, buffer)
        for policy in rt._banks.values():
            policy._slot_to_expert.extend([None] * (capacity - old_capacity))
            policy.persistent_slots = policy._persistent_capacity = capacity
            policy.slot_count = capacity + policy.transient_slots + policy.prefetch_slots
            policy._protected_cap = max(1, int(capacity * 0.8))
        # tcq3 plan update: whole-record persistent cache at the admitted capacity (no packed-scale residents)
        new_plan = replace(rt.plan, slots_per_layer=capacity, persistent_slots=capacity * N_LAYERS,
                           persistent_cache_bytes=capacity * N_LAYERS * SLOT_BYTES)
        pool.plan = rt.plan = new_plan
        # 3) tcq3 reader (whole-record) + tcq3 decode-verify lane (TcqPackedOps: stride kernel + t128 + routs)
        tcq_install.bind_tcq_reader(rt.reader, threading.local())
        runners = tcq_install.install(rt, switches, routs_by_layer, tables=tables)
        mx.synchronize()
        report.update(phase="decode", plane_layers=len(runners),
                      raw_scale_backing_released_bytes=released)
        return report

    return transition, report
