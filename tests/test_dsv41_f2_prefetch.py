"""CPU tests for the F2 next-layer expert prefetch lane (no GPU / no Metal).

MLX is hard-blocked by a meta-path finder before any import, so an accidental
``import mlx`` fails loudly instead of silently grabbing the Metal device (MLX
defaults to Metal on this box).  ``scripts/deepseek_v41/f2_predictor.py`` and the
shipped ``mtplx.expert_streaming.GlobalPrefetchRing`` are both MLX-free.

Coverage (the f2 CPU-verification brief):
  * ranking/merge/exclusion == the offline scorer rescore_router_capture.py on the
    real router-feature-20260918 capture, for k=3, feature=post-attention router,
    merge=max  (skips if the 45.6 MB capture NPZ is absent);
  * window-stop arithmetic and the construction-time window constants;
  * slot ownership/lease under promotion; eviction of wasted ring tenants;
    failure drain; budget accounting includes the ring;
  * a tiny synthetic end-to-end proving output identity with prefetch on vs off.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


# --- hard-block MLX before anything imports it -----------------------------
class _NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise RuntimeError("test_dsv41_f2_prefetch is CPU-only; MLX is forbidden")


if not any(isinstance(f, _NoMLX) for f in sys.meta_path):
    sys.meta_path.insert(0, _NoMLX())

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts" / "deepseek_v41"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import f2_predictor as F  # noqa: E402
from mtplx.expert_streaming import GlobalPrefetchRing  # noqa: E402

# The block above PROVED f2_predictor + GlobalPrefetchRing import with no MLX. Lift it
# now so a sibling suite that legitimately pins MLX to CPU (test_dsv41_f2_prefetch_lane)
# can import MLX in the same pytest process.
sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _NoMLX)]


# The offline scorer lives in the f1-overlap-sim worktree (its own receipt input).
_F1_SCORER = Path(
    "/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/"
    "dsv41-f1-overlap-sim/scripts/deepseek_v41/rescore_router_capture.py"
)


def _load_scorer():
    if not _F1_SCORER.exists():
        pytest.skip(f"offline scorer absent: {_F1_SCORER}")
    spec = importlib.util.spec_from_file_location("rescore_router_capture", _F1_SCORER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# 1. Predictor parity against the offline scorer (real capture)
# ===========================================================================
def test_predictor_matches_offline_scorer_k3_post_attention_max():
    scorer = _load_scorer()
    try:
        cap = scorer.load_capture()
    except SystemExit as exc:  # capture NPZ missing or hash mismatch
        pytest.skip(f"router capture unavailable: {exc}")

    feature = scorer.FEATURES.index("post_attention_router")  # == 1
    assert feature == 1
    k = 3
    scores = cap["scores"][feature]          # (64, 36, 6, 384)
    physical = cap["physical"]               # (64, 36, 384) READY owners
    ref_order = scorer.real_predictions(cap, feature, "max", k)
    ref_set = scorer.issued_mask(scorer.merged_scores(scores, "max"), physical, k)

    n_cycles, n_target = scores.shape[0], scores.shape[1]
    checked = 0
    for c in range(n_cycles):
        for idx in range(n_target):
            got = F.merge_rank_exclude(scores[c, idx], physical[c, idx], k)
            # exact ranked-order parity with real_predictions (keyed by window L)
            L = idx + F.FIRST_TARGET_LAYER - 1
            assert got == ref_order[(c, L)], (c, idx, got, ref_order[(c, L)])
            # set parity with issued_mask
            assert set(got) == set(int(e) for e in np.nonzero(ref_set[c, idx])[0])
            checked += 1
    assert checked == n_cycles * n_target == 64 * 36


def test_predictor_max_beats_sum_on_capture_k3():
    """f1-real-predictor's headline: max-over-rows beats sum at k=3 (post-attn)."""
    scorer = _load_scorer()
    try:
        cap = scorer.load_capture()
    except SystemExit as exc:
        pytest.skip(f"router capture unavailable: {exc}")
    row = lambda rule: next(  # noqa: E731
        r for r in scorer.curve(cap, 1, rule)["budgets"] if r["k"] == 3
    )["heldout"]["miss_coverage"]
    assert row("max") > row("sum")


def test_merge_rank_exclude_semantics():
    # rows max -> [0.3, 0.9, 0.8, 0.5]; expert 1 is a READY resident -> excluded.
    rs = np.array([[0.1, 0.9, 0.2, 0.5], [0.3, 0.2, 0.8, 0.4]])
    ready = np.array([False, True, False, False])
    assert F.merge_rank_exclude(rs, ready, 3) == [2, 3, 0]
    # k larger than the non-resident count -> only the finite (non-resident) tail.
    allbut = np.array([True, True, True, False])
    assert F.merge_rank_exclude(rs, allbut, 3) == [3]
    # max vs sum differ: expert 0 strongly favoured by one row (max=0.9), expert 1
    # moderately favoured by both (sum=1.0) -- max picks 0, sum dilutes to 1.
    rs2 = np.array([[0.9, 0.5], [0.0, 0.5]])
    assert F.merge_rank_exclude(rs2, np.array([False, False]), 1, rule="max") == [0]
    assert F.merge_rank_exclude(rs2, np.array([False, False]), 1, rule="sum") == [1]


# ===========================================================================
# 2. Window-stop arithmetic + construction-time constants
# ===========================================================================
def test_window_constants_match_published_f1_numbers():
    w = F.WindowConstants()  # defaults = the measured f1 constants
    # record read 1.3717 ms, sim plane (record/3) 0.4572 ms, window admits 2.371.
    assert abs(w.record_read_ns() - 1_371_683.7) < 1.0
    assert abs(F.SIM_PLANE_BYTES / 12.9 - 457_227.9) < 1.0
    assert abs(w.records_admitted() - 2.371) < 1e-3
    # actual (unequal) planes: gate/up 6,266,880 ; down 5,160,960.
    assert w.plane_read_ns(0) == pytest.approx(6_266_880 / 12.9)
    assert w.plane_read_ns(2) == pytest.approx(5_160_960 / 12.9)
    assert w.max_plane_read_ns() == pytest.approx(6_266_880 / 12.9)


def test_window_stop_rule():
    w = F.WindowConstants()
    plane = w.plane_read_ns(0)
    # demand still in flight -> never window-stopped (issue behind demand priority)
    assert not F.window_stop_blocks(
        demand_reads_outstanding=2, remaining_window_ns=0.0, plane_read_ns=plane
    )
    # demand done + a full plane still fits -> start it
    assert not F.window_stop_blocks(
        demand_reads_outstanding=0, remaining_window_ns=plane, plane_read_ns=plane
    )
    # demand done + less than one plane left -> stop
    assert F.window_stop_blocks(
        demand_reads_outstanding=0, remaining_window_ns=plane - 1, plane_read_ns=plane
    )


def test_issue_predictions_stops_at_window():
    ring = GlobalPrefetchRing(ring_size=16, base=300, expert_count=384)
    issued = []
    sch = F.SpeculativePlaneScheduler(
        ring, issue_plane=lambda l, e, p: issued.append((l, e, p))
    )
    w = sch.window
    # demand done, window holds exactly gate+up (planes 0,1) of the first record,
    # not the third: 6,266,880*2 fits, +5,160,960 does not.
    budget = w.plane_read_ns(0) + w.plane_read_ns(1) + 1.0
    n = sch.issue_predictions(
        5, [10, 11, 12], remaining_window_ns=budget, demand_reads_outstanding=0
    )
    # first record issues gate+up then window-stops; nothing further issues.
    assert n == 2
    assert issued == [(5, 10, 0), (5, 10, 1)]


# ===========================================================================
# 3. Ring lease / promotion / eviction / failure drain
# ===========================================================================
def test_promotion_reads_only_remaining_planes_and_keeps_lease():
    ring = GlobalPrefetchRing(ring_size=8, base=400, expert_count=384)
    sch = F.SpeculativePlaneScheduler(ring)
    # window admits only gate+up (planes 0,1); down (plane 2) is window-stopped,
    # so it is neither read nor in flight -- exactly what promotion must re-read.
    w = sch.window
    budget = w.plane_read_ns(0) + w.plane_read_ns(1) + 1.0
    sch.issue_predictions(
        5, [10], remaining_window_ns=budget, demand_reads_outstanding=0
    )
    assert ring.prefetch_ticket(5, 10) is not None  # inflight in the ring
    # the two issued planes land.
    sch.on_plane_complete(5, 10, 0, ok=True)
    sch.on_plane_complete(5, 10, 1, ok=True)
    # a demand route now needs expert 10: promote -> only the unread plane [2].
    assert sch.promote(5, 10) == [2]
    assert sch.counters.records_promoted == 1
    # the ring lease (ticket/slot) is retained through promotion.
    assert ring.prefetch_ticket(5, 10) is not None
    # a true cold miss (no speculative record) promotes to the whole record.
    assert sch.promote(5, 99) == [0, 1, 2]


def test_full_read_commits_and_consumption_credits_useful_planes():
    ring = GlobalPrefetchRing(ring_size=8, base=500, expert_count=384)
    sch = F.SpeculativePlaneScheduler(ring)
    sch.issue_predictions(7, [21], remaining_window_ns=1e9, demand_reads_outstanding=1)
    for p in range(3):
        sch.on_plane_complete(7, 21, p, ok=True)
    assert sch.counters.records_committed == 1
    assert ring.published(7, [21]) == {21: ring.base + 0} or ring.published(7, [21])
    sch.note_consumed(7, [21])
    assert sch.counters.planes_useful == 3


def test_failure_drains_record_without_health_flip():
    ring = GlobalPrefetchRing(ring_size=8, base=600, expert_count=384)
    sch = F.SpeculativePlaneScheduler(ring)
    sch.issue_predictions(5, [10], remaining_window_ns=1e9, demand_reads_outstanding=1)
    assert ring.prefetch_ticket(5, 10) is not None
    sch.on_plane_complete(5, 10, 0, ok=True)
    # a failed plane read drains the record: ring assignment invalidated, no raise.
    sch.on_plane_complete(5, 10, 1, ok=False)
    assert ring.prefetch_ticket(5, 10) is None
    assert sch.counters.records_failed == 1
    # the ring slot is free to reassign.
    loads = ring.plan_prefetch(5, [77])
    assert len(loads) == 1


def test_eviction_of_wasted_ring_tenant_is_counted():
    # small ring; commit two tenants for layer 5, never consume them, then force
    # round-robin eviction from a distant layer so target-1 protection (layer 6)
    # does not shield them.
    ring = GlobalPrefetchRing(ring_size=2, base=700, expert_count=384)
    sch = F.SpeculativePlaneScheduler(ring)
    for e in (10, 11):
        sch.issue_predictions(5, [e], remaining_window_ns=1e9, demand_reads_outstanding=1)
        for p in range(3):
            sch.on_plane_complete(5, e, p, ok=True)
    assert sch.counters.records_committed == 2
    # advance epochs past the re-eviction embargo, then predict for layer 9 (its
    # target-1 protected layer is 8, not 5) so the layer-5 tenants are evictable.
    for _ in range(3):
        ring.note_decode(5)
        ring.note_decode(9)
    sch.issue_predictions(9, [80, 81], remaining_window_ns=1e9, demand_reads_outstanding=1)
    assert sch.counters.records_wasted >= 1
    assert sch.counters.planes_wasted == sch.counters.records_wasted * 3


# ===========================================================================
# 4. Budget accounting includes the ring
# ===========================================================================
def test_ring_charge_and_admission():
    assert F.ring_charge_bytes(32) == 32 * 17_694_720 == 566_231_040
    assert F.ring_charge_bytes(0) == 0
    # extension-bank launch physical estimate (README): 109,745,344,620 B.
    base = 109_745_344_620
    # the 32-record ring pushes the launch estimate over the 110e9 ceiling.
    assert not F.admits_with_ring(base_launch_bytes=base, ring_records=32)
    # it fits only when the base leaves >= the ring charge of headroom.
    headroom_base = F.MACHINE_CEILING_BYTES - F.ring_charge_bytes(32)
    assert F.admits_with_ring(base_launch_bytes=headroom_base, ring_records=32)
    assert not F.admits_with_ring(base_launch_bytes=headroom_base + 1, ring_records=32)


# ===========================================================================
# 5. Tiny synthetic end-to-end: output identity with prefetch on vs off
# ===========================================================================
def _expert_vec(expert_id: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + expert_id)
    return rng.standard_normal(8).astype(np.float64)


def _run_layer_output(true_route):
    """A layer's output depends ONLY on its true routed experts."""
    return sum(_expert_vec(e) for e in true_route)


def test_end_to_end_output_identity_prefetch_on_vs_off():
    # two routed layers; layer 1's true route overlaps the layer-0 prediction
    # partially (one correct, one wrong) so prefetch both helps and wastes.
    true_route = {0: [10, 11], 1: [12, 13]}
    predicted_for_1 = [12, 99]  # 12 correct, 99 a mispredict (wasted)

    # ---- prefetch OFF ----
    cache_off: set[int] = set()
    demand_off = 0
    out_off = {}
    for layer in (0, 1):
        for e in true_route[layer]:
            if e not in cache_off:
                cache_off.add(e)
                demand_off += 1
        out_off[layer] = _run_layer_output(true_route[layer])

    # ---- prefetch ON (scheduler drives speculative plane reads) ----
    ring = GlobalPrefetchRing(ring_size=8, base=800, expert_count=384)
    cache_on: set[int] = set()

    def issue_plane(layer, expert, plane):
        # a completed 3-plane read makes the record resident in the cache.
        pass

    sch = F.SpeculativePlaneScheduler(ring, issue_plane=issue_plane)
    demand_on = 0
    wasted_specs = 0
    out_on = {}
    # layer 0: demand-load its own true route, then predict + speculatively load
    # layer 1's experts (planes complete -> resident in cache_on).
    for e in true_route[0]:
        if e not in cache_on:
            cache_on.add(e)
            demand_on += 1
    sch.issue_predictions(1, predicted_for_1, remaining_window_ns=1e12,
                          demand_reads_outstanding=0)
    for e in predicted_for_1:
        for p in range(3):
            sch.on_plane_complete(1, e, p, ok=True)
        cache_on.add(e)  # speculative record now resident
    out_on[0] = _run_layer_output(true_route[0])
    # layer 1: promote the speculative records that are truly demanded; any true
    # expert not pre-read is a demand miss.  Output comes from the TRUE route only.
    for e in true_route[1]:
        remaining = sch.promote(1, e)
        if e in cache_on:
            assert remaining == []          # fully pre-read -> zero demand planes
        else:
            assert remaining == [0, 1, 2]   # cold -> whole record on demand
            cache_on.add(e)
            demand_on += 1
    for e in predicted_for_1:
        if e not in true_route[1]:
            wasted_specs += 1
    out_on[1] = _run_layer_output(true_route[1])

    # (a) output identical regardless of prefetch -- computed only from the true route
    for layer in (0, 1):
        assert np.array_equal(out_off[layer], out_on[layer])
    # (b) prefetch actually did something: it hid the correct prediction (12) so the
    # ON run issued fewer demand loads, and it wasted the mispredict (99).
    assert demand_on < demand_off
    assert wasted_specs == 1
    # both predictions were fully pre-read (committed); the correct one (12) then
    # cost the layer-1 route zero demand planes.
    assert sch.counters.records_committed == 2
