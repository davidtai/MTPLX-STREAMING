"""Construction-time selection of the F2 prefetch lane inside run_full (stage edit).

The retained CONTROL installs the packed decode lane in
``sources/packed/packed_phase.py`` ``install_growth`` (:201-202):

    from plane_lane import install as install_plane_lane
    plane_runners.update(install_plane_lane(rt, dict(zip(layers, switches)), owners))

The F2 CANDIDATE is a single-site stage edit (sources/stage_full.py style: assert the
line occurs exactly once, then replace) that routes that ONE call to
``install_f2_growth`` under a construction-time flag. The stock control path is
byte-for-byte untouched when the candidate is off -- the edit adds an alternative
install call selected once, never a per-token / per-layer branch (AGENTS.md).

The candidate config is ``full_config.FullPrefetchConfig`` (transition-window + ring R),
and the ring reserve is charged into the retained admission bound through the SAME
admission code, ``sources/packed/packed_admission.py`` (see the receipt's memory
arithmetic: adding ``full_config.ring_reserve_bytes(R)`` to the capacity search's
``active``/``physical``/``wired`` fit checks at packed_admission.py:126-131 drops the
admitted rows from 111 to 110 for R=32).
"""
from __future__ import annotations

from .issue import DEFAULT_TOP_K, GatePredictor, Issue
from .plane_lane_prefetch import PackedOps, install

# Verify predictor geometry: targets 4..39 predicted from sources 3..38; 0..3 unpredicted
# (f1-real-predictor / f2_predictor.FIRST_TARGET_LAYER).
FIRST_TARGET_LAYER = 4


def select_prefetch_sources(routed_layers, *, first_target: int = FIRST_TARGET_LAYER):
    """Source layers that predict: ``L`` predicts ``L+1`` iff ``L+1`` is routed and
    ``L+1 >= first_target``.

    For the retained 0..39 routed set with ``first_target=4``: sources 3..38 ->
    targets 4..39; layers 0..3 (whose targets 1..3 are < 4) and the final routed layer
    (no successor) run the stock ``PackedDecode``.
    """
    routed = {int(layer) for layer in routed_layers}
    first_target = int(first_target)
    return sorted(
        layer for layer in routed
        if (layer + 1) in routed and (layer + 1) >= first_target
    )


def install_f2_growth(
    runtime,
    switches,
    scales_by_layer,
    *,
    model,
    reader_workers=None,
    first_target: int = FIRST_TARGET_LAYER,
    top_k: int = DEFAULT_TOP_K,
):
    """Drop-in for ``plane_lane.install`` in ``packed_phase.install_growth``.

    ``switches``          ``{layer: model.model.layers[layer].mlp.switch_mlp}``.
    ``scales_by_layer``   ``{layer: packed component scales}`` (packed_phase ``owners``).
    ``model``             the loaded model; each source binds to its target's live gate,
                          ``model.model.layers[L+1].mlp.gate`` (a ``deepseek_v41_moe.Gate``).
    ``reader_workers``    isolated priority-reader worker count (default = native; >= it).

    The ring R is read from ``runtime.config.prefetch_slots`` by ``install`` (it must
    equal ``runtime.plan.prefetch_ring_slots``); this function does not re-derive it.
    """
    layers = sorted(switches)
    ops_by_layer = {layer: PackedOps(scales_by_layer[layer]) for layer in layers}
    source_layers = select_prefetch_sources(layers, first_target=first_target)
    prefetch_sources = {
        layer: Issue(
            runtime,
            layer + 1,
            GatePredictor(model.model.layers[layer + 1].mlp.gate),
            top_k=top_k,
        )
        for layer in source_layers
    }
    return install(
        runtime, switches, ops_by_layer,
        prefetch_sources=prefetch_sources, reader_workers=reader_workers,
    )
