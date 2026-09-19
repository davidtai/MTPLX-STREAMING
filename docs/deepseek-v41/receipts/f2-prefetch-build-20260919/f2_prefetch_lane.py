"""F2 next-layer expert prefetch -- GPU install glue for the D5/M6 verify decode.

GPU-ONLY.  This module imports ``mlx`` and binds the live runtime, so it is NOT
imported by the CPU test suite (MLX would grab the Metal device).  Every piece of
policy it drives -- the merge/rank/exclude predictor, the window-stop arithmetic,
the plane-granular speculative scheduler and the ring budget charge -- lives in
``scripts/deepseek_v41/f2_predictor.py`` and is pinned on CPU against the offline
scorer ``rescore_router_capture.py`` and the published f1 discrete-event numbers.
What is UNVERIFIED until the guarded GPU window: the MLX gate evaluation, the
reader plane reads, the real slot transactions, and any throughput/latency claim.

Install style mirrors the lookahead-io ``plane_lane.py`` screen: a one-time
install at a quiescent post-prefill boundary that (1) validates the exact frozen
extension-bank configuration and fails loudly otherwise (fail once, before
measured generation -- AGENTS.md), (2) charges the speculative ring in the
admission plan, (3) wires each source layer's switch to the next layer's gate,
and (4) replaces the verify decode ``run`` so the next-layer prediction rides the
SAME ``mx.eval(indices)`` the switch already performs (no new host sync) and its
predicted records stream plane-by-plane behind demand priority with window-stop.

Stock behaviour is the un-installed path: with the lane not installed the runner
is byte-for-byte the retained best run.  There is NO eligible-or-stock or
try-then-fallback branch inside the enabled hot path -- the route is chosen once,
at install, by which ``switch._run`` is bound.  Outputs are computed only from the
TRUE route (``indices``); a mispredict only wastes a speculative read.
"""
from __future__ import annotations

import threading
from typing import Callable

import mlx.core as mx

from mtplx.expert_streaming import GlobalPrefetchRing, RoutingPhase
from mtplx.models.deepseek_v41_moe import _gate_prefix, _gate_prefix_impl

# The CPU-verified core (ranking, window-stop, plane scheduler, budget).
import f2_predictor as F  # noqa: E402  (staged on PYTHONPATH beside scripts/)


# ---------------------------------------------------------------------------
# Next-layer predictor -- rides the source layer's routing-index eval
# ---------------------------------------------------------------------------
class NextLayerGateLink:
    """Plain (non-Module) holder for the NEXT routed layer's gate and target id,
    bound on the SOURCE layer's switch.  A plain ``__slots__`` object rides outside
    the mlx parameter tree (``Module.__setattr__`` would register an array/tuple
    as a child); mirrors ``deepseek_v41._GatePrefetchLink`` and
    ``lookahead_prefetch.LookaheadRouters``."""

    __slots__ = ("target_layer", "next_gate")

    def __init__(self, target_layer: int, next_gate) -> None:
        self.target_layer = int(target_layer)
        self.next_gate = next_gate


def _biased_scores(gate, x):
    """The gate's native selection score ``sqrtsoftplus(x@W.T/temp)+bias`` -- the
    exact ``[rows, n_routed]`` tensor :meth:`Gate.__call__` argmaxes to pick the
    route (deepseek_v41_moe.gate_predict_topk ranks the same quantity).  Riding the
    router's own compiled/eager prefix means the predicted route matches the gate's
    selection with no separately-fitted correction (contrast ridge-prefetch).
    Reads only ``x`` and the frozen gate weights, so it is never an ancestor of the
    layer's own output."""

    xf = x.reshape(-1, gate.dim)
    from mtplx.models.deepseek_v41_moe import _attn_compile_gate

    if _attn_compile_gate(int(xf.shape[0])):
        _scores, biased = _gate_prefix(gate)(
            xf, gate.weight, gate.e_score_correction_bias
        )
    else:
        _scores, biased = _gate_prefix_impl(
            xf, gate.weight, gate.e_score_correction_bias,
            float(gate.gate_temp), str(gate.score_func),
        )
    return biased  # [rows, n_routed] f32


def next_layer_topk_ids(gate, x, resident_experts, k: int = F.DEFAULT_TOP_K):
    """Rank the next layer's top-``k`` non-resident experts from the CURRENT
    layer's post-attention router input ``x`` (max-over-rows, READY residents of
    the target layer excluded).  Returns an ``[k]`` int32 device array; NOT
    eval'd here -- the caller evals it together with the layer's own ``indices``
    so the barrier count per layer is unchanged.

    Device algebra == :func:`f2_predictor.merge_rank_exclude` (max over rows, set
    residents to -inf, top-``k``), which the CPU suite pins bit-for-bit against the
    offline scorer on the real capture.  ``argpartition`` returns the SET (order
    within the top-k is irrelevant to cache warming)."""

    biased = _biased_scores(gate, x)                    # [rows, n_routed]
    merged = biased.max(axis=0)                          # [n_routed] max over rows
    n = int(merged.shape[-1])
    resident = sorted(set(int(e) for e in resident_experts))
    if resident:
        # READY residents of the TARGET layer -> -inf before top-k, exactly as the
        # offline scorer masks ``physical`` (f2_predictor.merge_rank_exclude).
        idx = mx.array(resident, dtype=mx.int32)
        mask = (mx.arange(n)[None, :] == idx[:, None]).any(axis=0)  # [n] bool
        merged = mx.where(mask, mx.array(float("-inf"), dtype=merged.dtype), merged)
    kk = max(1, min(int(k), n))
    return mx.argpartition(-merged, kth=kk - 1)[:kk].astype(mx.int32)


# ---------------------------------------------------------------------------
# Plane reader binding (speculative plane reads via the shipped fanout reader)
# ---------------------------------------------------------------------------
#: gate/up/down component byte offsets inside one packed sidecar record, proven by
#: the lookahead-io plane_lane read offsets.  A speculative read of one plane reads
#: exactly this component's bytes into that component's destination view.
_PLANE_NAMES = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
_PLANE_OFFSETS = (0, 6_266_880, 12_533_760)


def bind_plane_issue(runtime, scheduler_by_layer):
    """Return an ``issue_plane(layer, expert, plane)`` that submits one component
    read for one predicted record onto the runtime's speculative reader queue,
    behind demand priority.  Uses the same ``reader._readv_range_into`` primitive
    plane_lane.py used; the destination view + slot come from the ring assignment
    the scheduler already made.  UNVERIFIED until the GPU window."""

    reader = runtime.reader
    submit = runtime._prefetch_executor.submit  # demand-priority behind the pool

    def issue_plane(layer, expert, plane):
        sched = scheduler_by_layer[layer]
        record, dest = runtime.prefetch_destination(layer, expert)  # slot view
        offset = record.sidecar_offset + _PLANE_OFFSETS[plane]
        view = dest.component_view(_PLANE_NAMES[plane])

        def read():
            ok = True
            try:
                reader._readv_range_into("experts.bin", offset, (view,))
            except BaseException:
                ok = False
            finally:
                view.release()
            sched.on_plane_complete(layer, expert, plane, ok=ok)

        return submit(read)

    return issue_plane


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
def _validate_frozen_config(runtime) -> None:
    """Fail once, loudly, before any measured generation if this is not the exact
    extension-bank composition the lane was designed and bounded for (AGENTS.md:
    validate invariant metadata at the installation boundary, never in the hot
    path)."""

    s, c = runtime.spec, runtime.config
    got = (
        s.hidden_size, s.expert_hidden_size, s.top_k, s.quant_bits,
        s.quant_group_size, s.expert_codec, s.swiglu_limit,
    )
    want = (5120, 2304, 6, 4, 32, "mxfp4", 10.0)
    problems = []
    if got != want:
        problems.append(f"model geometry {got} != {want}")
    if c.slot_layout != "component-banks":
        problems.append(f"slot_layout {c.slot_layout!r} != 'component-banks'")
    if c.cache_scope != "layer":
        problems.append("cache_scope must be 'layer'")
    if getattr(c, "decode_miss_records_per_part", None) != 3:
        problems.append("decode_miss_records_per_part must be 3")
    if str(getattr(c, "cache_policy", "")) not in (
        "transition-window", "tuned-transition-window"
    ):
        problems.append("cache_policy must be the transition-window family")
    if runtime.reader is None or getattr(runtime.reader, "_fanout_executor", None) is None:
        problems.append("reader fanout executor required for plane reads")
    if problems:
        raise RuntimeError("f2 prefetch lane refuses: " + "; ".join(problems))


def install(
    runtime,
    *,
    ring_records: int = F.DEFAULT_RING_RECORDS,
    top_k: int = F.DEFAULT_TOP_K,
    min_target_layer: int = F.FIRST_TARGET_LAYER,
    base_launch_bytes: int | None = None,
    window: F.WindowConstants | None = None,
):
    """One-time install of the F2 verify next-layer prefetch lane.

    Composes onto the retained best run: it does not resize any expert bank, move
    any row, or change the demand route.  Steps:

    1. Validate the frozen extension-bank configuration (fail once if not).
    2. Charge the speculative ring in the admission plan; refuse before model use
       if the 111-slot launch estimate + the ring does not fit under 110e9 B.
    3. Build ONE ``GlobalPrefetchRing`` (``ring_records`` slots, shared across
       layers -- decode runs layers sequentially one step ahead) and a
       per-source-layer :class:`SpeculativePlaneScheduler`.
    4. Wire each source layer L (3..38, i.e. target L+1 in ``min_target_layer``
       ..last-1) to layer L+1's gate.
    5. Replace the verify decode ``run`` so that after the existing
       ``mx.eval(indices)`` (extended to eval the predicted ids in the same
       barrier) the predicted records stream plane-by-plane behind demand.

    Returns the ``{layer: scheduler}`` map for telemetry.  UNVERIFIED until the
    GPU window: the MLX gate eval, plane reads and slot transactions."""

    _validate_frozen_config(runtime)

    # (2) admission: charge the ring on top of the retained 111-slot launch.
    if base_launch_bytes is not None and not F.admits_with_ring(
        base_launch_bytes=base_launch_bytes,
        ring_records=ring_records,
    ):
        raise RuntimeError(
            "f2 prefetch lane refuses: 111 slots + "
            f"{ring_records}-record ring "
            f"({F.ring_charge_bytes(ring_records)} B) exceeds the "
            f"{F.MACHINE_CEILING_BYTES} B whole-machine ceiling"
        )

    routed = tuple(getattr(runtime.spec, "routed_layer_indices", ()))
    if len(routed) < 2:
        return {}
    base_slot = runtime.plan.persistent_slots + runtime.plan.transient_slots
    ring = GlobalPrefetchRing(
        ring_size=ring_records, base=base_slot, expert_count=runtime.spec.n_routed_experts
    )
    window = window or F.WindowConstants()
    layers = runtime.model_layers()  # [(layer_index, mlp, gate)] in trunk order
    scheduler_by_layer: dict[int, F.SpeculativePlaneScheduler] = {}
    last_routed = routed[-1]

    # (4) wire each source L to target L+1's gate; targets < min_target_layer and
    # the last routed layer are unpredicted (layers 0..3 unpredicted per the brief).
    gate_by_layer = {li: gate for li, _mlp, gate in layers}
    for prev, nxt in zip(routed[:-1], routed[1:]):
        if nxt != prev + 1 or nxt < min_target_layer or nxt == last_routed:
            continue
        source_mlp = next((mlp for li, mlp, _g in layers if li == prev), None)
        if source_mlp is None or nxt not in gate_by_layer:
            continue
        source_mlp._mtplx_f2_next = NextLayerGateLink(nxt, gate_by_layer[nxt])
        scheduler_by_layer[nxt] = F.SpeculativePlaneScheduler(
            ring, window=window, top_k=top_k
        )

    issue_plane = bind_plane_issue(runtime, scheduler_by_layer)
    for sched in scheduler_by_layer.values():
        sched._issue_plane = issue_plane

    # (5) replace the verify decode run for each source layer's switch.  The demand
    # compute is the unchanged extension-bank packed runner (``prior_run``); this
    # wrapper only (a) evals the prediction in the SAME barrier and (b) issues the
    # speculative planes after demand + resident/shared GPU work is submitted.
    for prev in routed[:-1]:
        source_mlp = next((mlp for li, mlp, _g in layers if li == prev), None)
        link = getattr(source_mlp, "_mtplx_f2_next", None) if source_mlp else None
        if link is None:
            continue
        switch = source_mlp.switch_mlp
        prior_run = switch._run
        switch._run = _make_run(runtime, switch, prior_run, link, scheduler_by_layer)

    return scheduler_by_layer


def _make_run(runtime, switch, prior_run, link: NextLayerGateLink, scheduler_by_layer):
    target = link.target_layer
    gate = link.next_gate

    def run(x, indices, *, shared_work=None):
        # Only speculate in single-wave verify decode (RoutingPhase.DECODE, the
        # D5/M6 2..K+1 rows); a prefill of the same shape must not (phase gate,
        # mirroring _maybe_stash_gate_prefetch).  A plain AR M=1 also rides this.
        rows = int(x.reshape(-1, gate.dim).shape[0])
        speculate = runtime.expert_routing_phase(rows) is RoutingPhase.DECODE
        predicted_ids = None
        if speculate:
            bank = runtime.bank(target)
            resident = bank.resident_experts() if bank is not None else ()
            predicted_ids = next_layer_topk_ids(gate, x, resident, k=scheduler_by_layer[target].top_k)
            # PIGGYBACK: one host sync covering the layer's own indices AND the
            # predicted ids -- the barrier count per layer is unchanged.
            mx.eval(indices, predicted_ids)
        # Demand compute runs the unchanged extension-bank packed runner.  It
        # performs its own indices consume; when we already eval'd above the
        # runner's mx.eval(indices) is a no-op (indices already realized).
        out = prior_run(x, indices, shared_work=shared_work)
        if speculate and predicted_ids is not None:
            sched = scheduler_by_layer[target]
            sched.note_decode(target)
            ranked = [int(v) for v in predicted_ids.tolist()]
            # Issued AFTER demand + resident/shared GPU work is submitted by
            # prior_run; window-stop derives from the construction-time window.
            sched.issue_predictions(
                target, ranked,
                resident=runtime.bank(target).resident_experts()
                if runtime.bank(target) is not None else (),
                remaining_window_ns=runtime.remaining_compute_window_ns(),
                demand_reads_outstanding=runtime.demand_reads_outstanding(),
            )
        return out

    return run
