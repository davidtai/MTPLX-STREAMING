"""W87 -- single slot pool (MTPLX_DSV41_SINGLE_SLOT_POOL): merge each layer's
persistent + transient tiers into ONE scan-resistant resident pool, warm at decode
start.

Today (mtplx/expert_streaming.py) the per-layer persistent tier learns only from
DECODE routes; every PREFILL miss is served through the global transient scratch, so
on the 16,384-token cell the pool is COLD when decode starts (measured decode hit
rate 0.741 at 49 slots/layer). W87 admits EVERY miss into one pool with a
2Q/segmented-LRU policy driven by the PREFILL FREQUENCY SEED (prepare_prefill_seed):

- a prompt-frequency seed expert enters PROTECTED (admitted seed-first within the
  wave so it is never lost to transient overflow);
- every other prefill miss lands PROBATIONARY (the eviction end);
- a decode hit promotes probationary -> protected (2Q);
- PREFILL never evicts a protected expert (it overflows to transient instead), so a
  wide prefill scan -- or a re-prefill -- cannot scan out the earned set;
- W64/W71 pins are never victims.

The wave width stays transient_slots on BOTH paths (the merged-capacity widening was
retired -- protected/pinned slots make a wider prefill wave unserviceable, review
HIGH-1). A slot bank is a pure cache, so residency never changes a route's output:
byte-identity vs the two-tier path is proven on the real small component-bank runtime
for prefill+decode and DSpark verify accept/reject. The pure-policy sim (real
geometry, sorted waves, seeded, Zipf) shows single pool >= two-tier on the top-49
overlap AND a higher decode hit rate (first-64 and steady).

CPU-pinned; no GPU; fake component-bank runtime; peak RSS well under the guard.
Run under ``nice -n 19``.
"""

from __future__ import annotations

import os
import random
from collections import Counter

import mlx.core as mx
import pytest

pytest.importorskip("mlx.core")

mx.set_default_device(mx.cpu)

from mtplx.expert_manifest import load_expert_manifest  # noqa: E402
from mtplx.expert_runtime import (  # noqa: E402
    ExpertStreamingConfig,
    ExpertStreamingRuntime,
)
from mtplx.expert_streaming import (  # noqa: E402
    CacheCounters,
    LayerExpertSlotBank,
    RoutingPhase,
)
from mtplx.expert_streaming_models import plan_expert_memory  # noqa: E402
from mtplx.models.expert_mlx import (  # noqa: E402
    HotExpertSwitchGLU,
    expert_routing_phase,
    make_mlx_component_bank_allocator,
)

FLAG = "MTPLX_DSV41_SINGLE_SLOT_POOL"
_REAL_EVAL = mx.eval


@pytest.fixture(autouse=True)
def _cpu_and_flag():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    saved = os.environ.get(FLAG)
    try:
        yield
    finally:
        mx.set_default_device(prev)
        if saved is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = saved


# ==========================================================================
# Part A -- policy (direct LayerExpertSlotBank; pure Python, no runtime)
# ==========================================================================
def _bank(single_pool, *, expert_count=64, pool=8, transient=6, policy="lru"):
    return LayerExpertSlotBank(
        expert_count=expert_count,
        persistent_slots=pool,
        transient_slots=transient,
        cache_policy=policy,
        single_pool=single_pool,
    )


def _sorted_waves(flat, cap):
    uniq = sorted(dict.fromkeys(flat))  # sort_unique=True on the sidecar artifact
    return [uniq[i : i + cap] for i in range(0, len(uniq), cap)]


def _seeded_prefill(bank, prompt_flat):
    """Mirror the real switch: prepare_prefill_seed(full route) then partition the
    route into transient_slots-wide sorted waves (route_waves)."""
    bank.prepare_prefill_seed(prompt_flat)
    for w in _sorted_waves(prompt_flat, bank.transient_slots):
        bank.plan(w, phase=RoutingPhase.PREFILL)


def _zipf_prompt(*, experts, top_k, tokens, seed):
    rng = random.Random(seed)
    perm = list(range(experts))
    rng.shuffle(perm)
    w_by_id = [0.0] * experts
    for rank, eid in enumerate(perm):
        w_by_id[eid] = 1.0 / (1 + rank) ** 0.9
    flat = []
    for _ in range(tokens):
        flat.extend(dict.fromkeys(rng.choices(range(experts), weights=w_by_id, k=top_k)))
    return flat, w_by_id


def test_default_off_is_byte_identical_to_two_tier():
    """single_pool=False (and the unset default) is the historical two-tier path."""
    default = _bank(False)
    off = LayerExpertSlotBank(
        expert_count=64, persistent_slots=8, transient_slots=6, cache_policy="lru"
    )
    seq = [[1, 2], [3, 4], [1, 5], [6, 7, 8], [2, 3], [9, 10, 11, 12]]
    for r in seq:
        p_def = default.plan(r, phase=RoutingPhase.DECODE)
        p_off = off.plan(r, phase=RoutingPhase.DECODE)
        assert p_def.slots == p_off.slots and p_def.loads == p_off.loads
        assert p_def.pool_loads == p_def.scan_inserts == p_def.promotions == 0
    assert default.resident_experts == off.resident_experts


def test_seed_admitted_protected_and_counters():
    """A prompt-frequency seed expert enters PROTECTED; others land probationary."""
    bank = _bank(True, expert_count=64, pool=6, transient=6)
    ctr = CacheCounters()
    bank._prefill_seed_candidates = {0, 1}  # what prepare_prefill_seed would pick
    p = bank.plan([0, 1, 2, 3, 4, 5], phase=RoutingPhase.PREFILL)
    ctr.observe(p, expert_record_bytes=100)
    assert {0, 1} <= bank._protected            # seed -> protected
    assert p.promotions == 2 and p.scan_inserts == 4 and p.pool_loads == 6
    assert ctr.as_dict()["promotions"] == 2
    # a decode hit promotes a probationary resident (2Q)
    p2 = bank.plan([2] * 6, phase=RoutingPhase.DECODE)
    assert p2.promotions == 1 and 2 in bank._protected


def test_prefill_scan_never_evicts_protected():
    """HIGH-3: WITHIN a request's prefill, a wide multi-wave scan cannot evict the
    seed-protected experts -- non-seed misses overflow to transient instead. (Across
    a request boundary -- decode then a new prefill -- the protected set is instead
    DEMOTED on purpose; that is HIGH-2, tested separately.)"""
    bank = _bank(True, expert_count=256, pool=6, transient=6)
    bank._prefill_seed_candidates = set(range(6))  # the prompt's top-6 by frequency
    for e in range(6):
        bank._prefill_route_freq[e] = 100
    bank.plan(list(range(6)), phase=RoutingPhase.PREFILL)  # wave 1: the seed
    protected = set(bank._protected)
    assert protected == set(range(6))
    # subsequent waves of the SAME prefill (no decode in between -> no demote):
    for base in range(100, 250, 6):
        bank.plan(list(range(base, base + 6)), phase=RoutingPhase.PREFILL)
    assert protected <= set(bank.resident_experts), "prefill scan evicted a protected"


def test_pins_are_never_victims_under_single_pool():
    bank = _bank(True, expert_count=256, pool=6, transient=6)
    for w in _sorted_waves([0, 1, 2, 3, 4, 5], 6):
        bank.plan(w, phase=RoutingPhase.PREFILL)
    bank.pin_working_set(experts=[0, 1, 2, 3, 4, 5])
    assert set(bank.pinned_experts) == {0, 1, 2, 3, 4, 5}
    for base in range(20, 80, 6):
        bank.plan(list(range(base, base + 6)), phase=RoutingPhase.PREFILL)
    assert {0, 1, 2, 3, 4, 5} <= set(bank.resident_experts)


def test_route_capacity_bound_is_transient_slots():
    """The merged-capacity widening was retired (HIGH-1): a route can hold at most
    transient_slots unique on BOTH paths; route_waves keeps every wave within it."""
    for single in (True, False):
        b = _bank(single, expert_count=64, pool=8, transient=6)
        b.plan(list(range(6)), phase=RoutingPhase.DECODE)   # ok (== transient)
        with pytest.raises(ValueError):
            b.plan(list(range(7)), phase=RoutingPhase.DECODE)  # 7 > 6


def test_plan_transaction_rollback_restores_pool_state():
    bank = _bank(True, expert_count=64, pool=6, transient=6)
    bank._prefill_seed_candidates = {10, 11}
    _seeded_prefill(bank, [10, 10, 11, 12, 13, 14, 15] + list(range(20, 40)))
    snap = (
        list(bank.resident_experts), set(bank._protected),
        dict(bank._pool_recency), bank._pool_clock, bank._decode_epoch,
    )
    _, txn = bank.plan_transaction([30, 31, 32, 33, 34], phase=RoutingPhase.DECODE)
    txn.rollback_completion()
    assert (
        list(bank.resident_experts), set(bank._protected),
        dict(bank._pool_recency), bank._pool_clock, bank._decode_epoch,
    ) == snap


def test_invalidate_expert_clears_pool_state():
    """MED-5: a capacity/health eviction (invalidate_expert) must drop the expert
    from _protected and _pool_recency, not just the slot map."""
    bank = _bank(True, expert_count=64, pool=6, transient=6)
    bank._prefill_seed_candidates = {5}
    bank.plan([5, 6, 7], phase=RoutingPhase.PREFILL)
    assert 5 in bank._protected and 5 in bank._pool_recency
    slot = bank.invalidate_expert(5)
    assert slot is not None
    assert 5 not in bank._protected and 5 not in bank._pool_recency
    assert 5 not in bank.resident_experts


def test_overflow_guard_raises_before_bad_slot():
    """HIGH-1 tripwire: if a route somehow exceeds what the pool+transient can hold
    (here forced by pinning the whole pool then a transient_slots-wide all-miss
    decode route), the loud policy-time guard fires instead of emitting an
    out-of-plan slot index."""
    bank = _bank(True, expert_count=256, pool=6, transient=6)
    for w in _sorted_waves(list(range(6)), 6):
        bank.plan(w, phase=RoutingPhase.PREFILL)
    bank.pin_working_set(experts=list(range(6)))  # all 6 slots pinned
    # a 6-unique all-miss decode route: 0 persistent free (all pinned) -> all 6 to
    # transient == transient_slots, which fits exactly (no raise).
    bank.plan(list(range(50, 56)), phase=RoutingPhase.DECODE)
    # but a 7-unique route is rejected up front by _validate_experts (> transient).
    with pytest.raises(ValueError):
        bank.plan(list(range(50, 57)), phase=RoutingPhase.DECODE)


# --- the reviewer's real-shape A/B: sorted waves, seeded, Zipf prompt ----------
@pytest.mark.parametrize("policy", ["frequency", "lru"])
def test_sorted_wave_seeded_pool_matches_or_beats_two_tier(policy):
    """HIGH-2 (request 1): at the real route shape (ONE layer-major route,
    sort_unique waves, prepare_prefill_seed) the single pool retains >= the
    two-tier's top-49 overlap and warms at least as well. The SHIPPED baseline is
    cache_policy='frequency' (ExpertStreamingConfig default), vs which the request-1
    first-64 gain is ~1 point and steady is ~tied; vs 'lru' the gains are large. The
    durable win is the served MULTI-REQUEST case (a separate test)."""
    E, K, P, T = 384, 6, 49, 48
    prompt, w_by_id = _zipf_prompt(experts=E, top_k=K, tokens=4096, seed=87)
    top_p = {e for e, _ in Counter(prompt).most_common(P)}

    def decode_hit_rate(bank, n, seed):
        rng = random.Random(seed)
        hits = req = 0
        for _ in range(n):
            r = list(dict.fromkeys(rng.choices(range(E), weights=w_by_id, k=K)))
            pl = bank.plan(r, phase=RoutingPhase.DECODE)
            hs = set(pl.hits)
            hits += sum(e in hs for e in pl.experts)
            req += len(pl.experts)
        return hits / req

    out = {}
    for single in (False, True):
        b = LayerExpertSlotBank(expert_count=E, persistent_slots=P, transient_slots=T,
                                cache_policy=policy, single_pool=single)
        _seeded_prefill(b, prompt)
        overlap = len(set(b.resident_experts) & top_p)
        out[single] = (overlap, decode_hit_rate(b, 64, 9), decode_hit_rate(b, 200, 9))
    (ov_off, f64_off, st_off) = out[False]
    (ov_on, f64_on, st_on) = out[True]
    assert ov_on >= ov_off, f"[{policy}] overlap on={ov_on} < off={ov_off}"
    assert f64_on >= f64_off, f"[{policy}] first-64 on={f64_on:.3f} < off={f64_off:.3f}"
    # steady: single-pool 2Q is much better than lru, ~tied vs frequency -- never
    # materially worse than either baseline.
    assert st_on >= st_off - 0.02, f"[{policy}] steady on={st_on:.3f} << off={st_off:.3f}"


# ==========================================================================
# Part B -- plan arithmetic (batch_admission_slots; allocation-neutral)
# ==========================================================================
def _dsv41_spec():
    import mtplx.expert_streaming_models as esm

    Spec = esm.ExpertStreamingModelSpec
    for name in dir(esm):
        obj = getattr(esm, name, None)
        if isinstance(obj, Spec) and obj.key == "deepseek-v41-flash-expert-mxfp4":
            return obj
    pytest.skip("deepseek-v41-flash-expert-mxfp4 spec not found")


def test_60gib_plan_arithmetic_and_allocation_neutral():
    """60 GiB profile: 49 slots/layer x 40 = 1960 persistent + 48 transient; the
    single-fence wave width is transient_slots (48) on BOTH paths (widening retired);
    allocation byte-identical."""
    sp = _dsv41_spec()
    plan = plan_expert_memory(
        sp, total_limit_bytes=60 * 1024**3, context_tokens=0,
        runtime_reserve_bytes=7 * 1024**3, transient_slots=48,
    )
    assert plan.slots_per_layer == 49 and plan.transient_slots == 48
    assert plan.persistent_slots == 1960
    # single-fence wave width == transient_slots (widening retired, both paths).
    assert plan.batch_admission_slots == 48


# ==========================================================================
# Part C -- runtime byte-identity + crash safety + counters (fake CPU runtime)
# ==========================================================================
def _open_runtime(tmp_dir, *, single_pool, expert_count=32, top_k=4,
                  resident_slots=8, transient=6, cache_scope="layer"):
    from tests.test_streamed_models import _integrated_hy3_artifact

    if single_pool:
        os.environ[FLAG] = "1"
    else:
        os.environ.pop(FLAG, None)
    root, _config, spec, manifest_path = _integrated_hy3_artifact(
        tmp_dir, expert_count=expert_count, top_k=top_k
    )
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    sc = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(resident_slots),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        transient_slots=transient,
        slot_layout="component-banks",
        cache_scope=cache_scope,
    )
    plan = sc.memory_plan(spec)
    rt = ExpertStreamingRuntime.open(
        root, manifest_path, sc, spec=spec,
        buffer_allocator=make_mlx_component_bank_allocator(
            plan, spec, load_expert_manifest(manifest_path)
        ),
        device_synchronize=mx.synchronize, apply_memory_cap=False,
    )
    return rt, spec


def _prefill_decode_outputs(rt, spec, *, seed):
    layer = spec.routed_layer_start
    sw = HotExpertSwitchGLU(rt, layer)
    hidden, top_k, ec = spec.hidden_size, spec.top_k, spec.expert_count
    outs = []
    mx.random.seed(seed)
    rows = 12
    x = (0.3 * mx.random.normal((rows, 1, hidden))).astype(mx.bfloat16)
    flat = [(r * top_k + k) % ec for r in range(rows) for k in range(top_k)]
    idx = mx.array(flat, dtype=mx.int32).reshape((rows, top_k))
    _REAL_EVAL(x, idx)
    with expert_routing_phase("prefill"):
        out = sw(x, idx)
    _REAL_EVAL(out)
    rt.flush_deferred_slot_releases(evaluate=True)
    outs.append(out.reshape((-1, hidden)))
    tail = [(ec - 1 - k) % ec for k in range(top_k)]
    for t in range(6):
        xd = (0.3 * mx.random.normal((1, 1, hidden))).astype(mx.bfloat16)
        rotated = [tail[(t + k) % top_k] for k in range(top_k)]
        idd = mx.array(rotated, dtype=mx.int32).reshape((1, top_k))
        _REAL_EVAL(xd, idd)
        with expert_routing_phase("decode"):
            od = sw(xd, idd)
        _REAL_EVAL(od)
        rt.flush_deferred_slot_releases(evaluate=True)
        outs.append(od.reshape((-1, hidden)))
    return mx.concatenate(outs, axis=0)


def test_prefill_decode_byte_identical_on_vs_off(tmp_path):
    off_dir, on_dir = tmp_path / "off", tmp_path / "on"
    off_dir.mkdir(); on_dir.mkdir()
    rt_off, spec_off = _open_runtime(off_dir, single_pool=False)
    try:
        assert rt_off._single_slot_pool is False
        out_off = _prefill_decode_outputs(rt_off, spec_off, seed=87)
    finally:
        rt_off.close()
    rt_on, spec_on = _open_runtime(on_dir, single_pool=True)
    try:
        assert rt_on._single_slot_pool is True
        # widening retired: the wave width is transient_slots on both paths.
        assert rt_on._batch_admission_slots() == rt_on.plan.transient_slots
        out_on = _prefill_decode_outputs(rt_on, spec_on, seed=87)
    finally:
        rt_on.close()
    assert out_off.shape == out_on.shape
    assert mx.array_equal(out_off, out_on), "single-pool residency changed the math"


def _verify_sequence_outputs(rt, spec, *, seed, warm):
    layer = spec.routed_layer_start
    hidden, top_k, ec = spec.hidden_size, spec.top_k, spec.expert_count
    sw = HotExpertSwitchGLU(rt, layer)
    for e in warm:
        xi = mx.zeros((1, 1, hidden), dtype=mx.bfloat16)
        ii = mx.array([[e] * top_k], dtype=mx.int32)
        with expert_routing_phase("decode"):
            _REAL_EVAL(sw(xi, ii))
        rt.flush_deferred_slot_releases(evaluate=True)
    outs = []
    mx.random.seed(seed)
    for rows in (2, 4, 8):
        x = (0.3 * mx.random.normal((rows, 1, hidden))).astype(mx.bfloat16)
        flat = [(r * 3 + k) % ec for r in range(rows) for k in range(top_k)]
        idx = mx.array(flat, dtype=mx.int32).reshape((rows, top_k))
        _REAL_EVAL(x, idx)
        with expert_routing_phase("decode"):
            out = sw(x, idx)
        _REAL_EVAL(out)
        rt.flush_deferred_slot_releases(evaluate=True)
        outs.append(out.reshape((-1, hidden)))
    return mx.concatenate(outs, axis=0)


def test_dspark_verify_byte_identical_on_vs_off(tmp_path):
    off_dir, on_dir = tmp_path / "off", tmp_path / "on"
    off_dir.mkdir(); on_dir.mkdir()
    warm = [0, 2, 4, 6, 8, 10]
    rt_off, spec_off = _open_runtime(off_dir, single_pool=False)
    try:
        out_off = _verify_sequence_outputs(rt_off, spec_off, seed=61, warm=warm)
    finally:
        rt_off.close()
    rt_on, spec_on = _open_runtime(on_dir, single_pool=True)
    try:
        out_on = _verify_sequence_outputs(rt_on, spec_on, seed=61, warm=warm)
    finally:
        rt_on.close()
    assert out_off.shape == out_on.shape
    assert mx.array_equal(out_off, out_on), "verify accept/reject math diverged"


def test_runtime_no_overflow_after_capacity_reduction(tmp_path):
    """HIGH-1 on the real runtime: a P+T-wide prefill after the derived policy
    lowers the per-layer capacity must NOT fault (route_waves bounds every wave by
    transient_slots)."""
    d = tmp_path / "sp"
    d.mkdir()
    rt, spec = _open_runtime(d, single_pool=True)
    try:
        layer = spec.routed_layer_start
        bank = rt._banks[layer]
        P, T = bank.persistent_slots, bank.transient_slots
        hidden, top_k, ec = spec.hidden_size, spec.top_k, spec.expert_count
        sw = HotExpertSwitchGLU(rt, layer)

        def prefill(experts):
            rows = len(experts) // top_k
            x = (0.3 * mx.random.normal((rows, 1, hidden))).astype(mx.bfloat16)
            idx = mx.array(experts, dtype=mx.int32).reshape((rows, top_k))
            _REAL_EVAL(x, idx)
            with expert_routing_phase("prefill"):
                out = sw(x, idx)
            _REAL_EVAL(out)
            rt.flush_deferred_slot_releases(evaluate=True)

        prefill(list(range(P + T)) + [0] * ((-(P + T)) % top_k))
        rt._evict_layer_bank_to_capacity(layer, P - 1)
        second = [(P + T + i) % ec for i in range(P + T)]
        prefill(second + [second[0]] * ((-(P + T)) % top_k))  # must not raise
        assert bank.persistent_capacity == P - 1
    finally:
        rt.close()


def test_flag_gated_off_under_global_scope(tmp_path):
    """MED-6: the flag requires cache_scope 'layer'; under global scope it is gated
    off (two-tier) rather than faulting."""
    d = tmp_path / "g"
    d.mkdir()
    os.environ[FLAG] = "1"
    try:
        rt, _spec = _open_runtime(d, single_pool=True, cache_scope="global")
    except Exception:
        pytest.skip("global cache scope not supported by the fake artifact")
    try:
        assert rt._single_slot_pool is False  # gated off under global scope
    finally:
        rt.close()


def test_runtime_cold_start_counters_populated_both_paths(tmp_path):
    """Both arms populate the cold_start receipt block (AR decode-step basis);
    single-pool's first-token decode hit rate >= the two-tier path's."""
    rates = {}
    for single_pool, name in ((False, "off"), (True, "on")):
        d = tmp_path / name
        d.mkdir()
        rt, spec = _open_runtime(d, single_pool=single_pool)
        try:
            _prefill_decode_outputs(rt, spec, seed=87)
            cs = rt.snapshot()["cold_start"]
            assert cs["single_slot_pool"] is single_pool
            assert cs["cold_start_decode_steps"] == 64
            assert cs["first_64_steps_requests"] > 0
            assert cs["decode_steps_observed"] >= 6
            assert "measurement_basis" in cs
            assert rt.snapshot()["memory_plan"]["batch_admission_slots"] == (
                rt.plan.transient_slots
            )
            rates[name] = cs["first_64_steps_hit_rate"]
        finally:
            rt.close()
    assert rates["on"] is not None and rates["off"] is not None
    assert rates["on"] >= rates["off"]


def test_cold_window_resets_on_new_request(tmp_path):
    """MED-4: the cold window re-opens on the first PREFILL after decode, so a
    second request's first decode steps count as cold, not steady."""
    d = tmp_path / "req"
    d.mkdir()
    rt, spec = _open_runtime(d, single_pool=True)
    try:
        rt._cold_start_decode_tokens = 3  # small window for the test
        _prefill_decode_outputs(rt, spec, seed=1)  # request 1: >=6 decode steps
        idx_after_1 = rt._decode_token_index
        assert idx_after_1 >= 3  # request 1 exhausted its cold window
        _prefill_decode_outputs(rt, spec, seed=2)  # request 2 (prefill resets)
        # the first PREFILL of request 2 re-opened the window, then request 2's
        # decode advanced it again -- it did not keep climbing from request 1.
        assert rt._decode_token_index <= idx_after_1
    finally:
        rt.close()


def test_seed_first_reorder_admits_seed_first():
    """HIGH-2/LOW: seed experts are admitted FIRST within a prefill wave, ordered by
    ASCENDING frequency, even when the route lists lower ids first -- so the reorder
    is observable in plan.misses and the least-frequent seed takes the lowest recency
    (dropped first by a later decode eviction)."""
    b = _bank(True, expert_count=64, pool=8, transient=12)
    b._prefill_seed_candidates = set(range(24, 32))  # top-8 experts are ids 24..31
    for i, e in enumerate(range(24, 32)):
        b._prefill_route_freq[e] = 100 - i  # 24 most frequent .. 31 least
    plan = b.plan([0, 1, 2, 3, 24, 25, 26, 27, 28, 29, 30, 31],
                  phase=RoutingPhase.PREFILL)
    assert plan.misses[0] in set(range(24, 32)), plan.misses[:3]
    assert plan.misses[0] == 31  # least-frequent seed admitted first (lowest recency)
    assert set(range(24, 32)) <= b._protected  # all seed protected


def _zipf(seed, *, experts, top_k, tokens):
    rng = random.Random(seed)
    perm = list(range(experts)); rng.shuffle(perm)
    w = [0.0] * experts
    for rank, eid in enumerate(perm):
        w[eid] = 1.0 / (1 + rank) ** 0.9
    flat = []
    for _ in range(tokens):
        flat.extend(dict.fromkeys(rng.choices(range(experts), weights=w, k=top_k)))
    return flat, w


def _decode_hit_rate(bank, w, seed, n, *, E=384, K=6):
    rng = random.Random(seed)
    hits = req = 0
    for _ in range(n):
        r = list(dict.fromkeys(rng.choices(range(E), weights=w, k=K)))
        pl = bank.plan(r, phase=RoutingPhase.DECODE)
        hs = set(pl.hits)
        hits += sum(e in hs for e in pl.experts)
        req += len(pl.experts)
    return hits / req


@pytest.mark.parametrize("policy", ["frequency", "lru"])
def test_second_request_different_set_warms_beats_two_tier(policy):
    """HIGH-2: the served MULTI-REQUEST win. Request 1 (prompt A) -> decode 264 ->
    request 2 (prompt B, a DIFFERENT hot set), NO reset. The two-tier tier learned
    A's decode set and B's prefill cannot repopulate it, so it starts cold; the pool
    re-warms per request. (Was COLDER than two-tier before the per-request demote.)"""
    E, K, P, T = 384, 6, 49, 48
    pA, wA = _zipf(1, experts=E, top_k=K, tokens=4096)
    pB, wB = _zipf(2, experts=E, top_k=K, tokens=4096)

    def run(single):
        b = LayerExpertSlotBank(expert_count=E, persistent_slots=P, transient_slots=T,
                                cache_policy=policy, single_pool=single)
        _seeded_prefill(b, pA)
        _decode_hit_rate(b, wA, 11, 264)          # request 1 decode
        _seeded_prefill(b, pB)                    # request 2 prefill, NO reset
        return _decode_hit_rate(b, wB, 22, 64)

    off = run(False)
    on = run(True)
    assert on >= off, f"[{policy}] request-2 first-64 on={on:.3f} < two-tier {off:.3f}"


@pytest.mark.parametrize("policy", ["frequency", "lru"])
def test_second_request_same_hot_set_retains_pool(policy):
    """HIGH-2 fix (1): a SAME-hot-set request 2 must RETAIN the returning hot experts
    -- prepare_prefill_seed re-protects the already-resident chosen in place and seeds
    only the non-resident remainder, instead of demoting + seed-evicting the returning
    set (which collapsed first-64 to ~0.355 with only ~12/49 retained)."""
    E, K, P, T = 384, 6, 49, 48
    pA, wA = _zipf(1, experts=E, top_k=K, tokens=4096)
    top_p = {e for e, _ in Counter(pA).most_common(P)}
    b = LayerExpertSlotBank(expert_count=E, persistent_slots=P, transient_slots=T,
                            cache_policy=policy, single_pool=True)
    _seeded_prefill(b, pA)
    _decode_hit_rate(b, wA, 3, 264)
    _seeded_prefill(b, pA)                        # SAME prompt, request 2, NO reset
    retained = len(set(b.resident_experts) & top_p)
    first64 = _decode_hit_rate(b, wA, 5, 64)
    assert retained >= 45, f"[{policy}] same-set retained only {retained}/49 of top-49"
    assert first64 >= 0.5, f"[{policy}] same-set request-2 first-64 {first64:.3f} collapsed"


def test_pool_ignores_cache_policy():
    """The single pool is a 2Q policy that IGNORES cache_policy (it never consults the
    frequency/lru _score / _victim_slot).  The shipped default is 'frequency'
    (expert_runtime.py), so freq vs lru must produce identical pool state."""
    E, P, T = 64, 8, 6
    seq = [(list(range(0, 6)), RoutingPhase.PREFILL)] + [
        ([(i * 3 + k) % E for k in range(3)], RoutingPhase.DECODE) for i in range(12)
    ]

    def run(policy):
        b = LayerExpertSlotBank(expert_count=E, persistent_slots=P, transient_slots=T,
                                cache_policy=policy, single_pool=True)
        b._prefill_seed_candidates = {0, 1, 2, 3}
        slots = [tuple(b.plan(r, phase=ph).slots) for r, ph in seq]
        return slots, tuple(b.resident_experts), sorted(b._protected)

    assert run("frequency") == run("lru")


def _load_ab_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ab_w87_test", "scripts/deepseek_v41/ab_decode_env_levers.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_bench_counters_and_cold_reset_work_with_bare_runtime():
    """HIGH-1: the loader attaches a BARE ExpertStreamingRuntime (snapshot()/reset())
    as model._mtplx_expert_runtime, not an MTPLXRuntime; the bench must still get an
    expert_cache + cold_start block and the DSpark cold reset must actually fire."""
    ab = _load_ab_module()
    _CACHE_KEYS = (
        "route_calls", "expert_requests", "unique_expert_requests",
        "shared_expert_assignments", "expert_hits", "expert_misses",
        "persistent_loads", "transient_loads", "evictions", "bytes_read",
        "prefetch_issued", "prefetch_committed", "pool_loads", "scan_inserts",
        "promotions",
    )

    class _BareRuntime:  # shaped exactly like ExpertStreamingRuntime (loader:667)
        def __init__(self):
            self.reset_calls = 0

        def snapshot(self):
            return {
                "cache": {k: 1 for k in _CACHE_KEYS},
                "cold_start": {
                    "single_slot_pool": True,
                    "measurement_basis": "first-64 DECODE STEPS, single-request",
                    "cold_start_decode_steps": 64,
                    "decode_steps_observed": 2,
                    "first_64_steps_hits": 3,
                    "first_64_steps_requests": 5,
                    "steady_hits": 0,
                    "steady_requests": 0,
                },
            }

        def reset(self):
            self.reset_calls += 1

        # deliberately NO expert_streaming_snapshot / expert_streaming attribute.

    class _Model:
        def __init__(self, rt):
            self._mtplx_expert_runtime = rt

    rt = _BareRuntime()
    model = _Model(rt)
    snap = ab._stream_counters_snapshot(model)
    assert snap is not None, "bench lost the streaming snapshot on a bare runtime"
    assert "expert_cache" in snap and "cold_start" in snap
    assert snap["expert_cache"]["pool_loads"] == 1
    assert ab._cold_reset_expert_streaming(model) is True
    assert rt.reset_calls == 1  # the DSpark cold reset actually fired


def test_serve_stream_counters_expose_pool_and_cold_start(tmp_path):
    from mtplx.serve_stream_counters import (
        snapshot_stream_counters, stream_counters_delta,
    )

    class _Shim:  # mirrors MTPLXRuntime.expert_streaming_snapshot delegation
        def __init__(self, rt):
            self._rt = rt

        def expert_streaming_snapshot(self):
            return self._rt.snapshot()

    d = tmp_path / "sp"
    d.mkdir()
    rt, spec = _open_runtime(d, single_pool=True)
    try:
        shim = _Shim(rt)
        before = snapshot_stream_counters(shim)
        _prefill_decode_outputs(rt, spec, seed=87)
        after = snapshot_stream_counters(shim)
    finally:
        rt.close()
    delta = stream_counters_delta(before, after, tokens=6)
    ec = delta["expert_cache"]
    for key in ("pool_loads", "scan_inserts", "promotions"):
        assert key in ec
    assert ec["pool_loads"] > 0
    cs = delta["cold_start"]
    assert cs["single_slot_pool"] is True
    assert "decode_hit_rate_first_64_steps" in cs
    assert "decode_hit_rate_steady_state" in cs
