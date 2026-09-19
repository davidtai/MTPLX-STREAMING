"""CPU seam tests for the F2 next-layer prefetch lane (scripts/deepseek_v41/f2/).

Constructs the REAL runtime classes (ExpertStreamingRuntime + LayerExpertSlotBank +
the shared GlobalPrefetchRing + PositionalExpertReader + component-bank slot pool)
over a tiny sidecar-backed synthetic artifact, and drives the f2 lane's live
predictor (issue.py), its priority reader (priority_reads.py), its config
(full_config.py) and its install (plane_lane_prefetch.install) against them. No GPU,
no Metal, no real weights. Run under ``nice -n 19`` and without ``pytest -n auto``.

MLX defaults to Metal on this box, so this file pins the CPU device before importing
anything that imports MLX (a sibling suite hard-blocks MLX with a meta-path finder;
we remove it defensively so the pin, not the block, governs here).

What runs on CPU vs. what is deferred to the guarded GPU window
(docs/deepseek-v41/receipts/f2-prefetch-build-20260919/README.md, "what stays
unverified"):
  * REAL on CPU: the whole runtime/slot/ring/reconcile machinery -- prefetch_experts,
    slots.load_speculative, the shared ring, _reconcile_prefetch_for_route, the plan's
    ring-hit resolution, the counters -- plus the live predictor, the priority reader
    and the config; the demand path reads the sidecar through the shipped native
    reader.
  * DEFERRED to the GPU smoke (f2/gpu_smoke.py): the packed lane's EXECUTION BODY
    (PrefetchDecode.run) with the plane-split ``bind_priority_reader`` + Metal
    ``PackedOps`` gather at production geometry. bind_priority_reader reads three
    weight planes of the production ``experts.bin`` at offsets (0, 6,266,880,
    12,533,760) -- out of bounds for the 6,912-byte tiny record -- and PackedOps is a
    ``mx.fast.metal_kernel``. Both need production-geometry (5120/2304) records, which
    cannot be materialised tiny on CPU. install() acceptance is still proven here
    (it rewires; it does not run a route), and the schedule/barrier parity of
    PrefetchDecode.run is asserted at the predictor seam (one host sync).
"""
from __future__ import annotations

import hashlib
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Defensively drop any sibling _NoMLX meta-path finder, then pin CPU BEFORE MLX loads.
sys.meta_path[:] = [
    finder for finder in sys.meta_path if type(finder).__name__ != "_NoMLX"
]
import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts" / "deepseek_v41"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from mtplx.expert_manifest import (  # noqa: E402
    ExpertManifest,
    ExpertRecord,
    ResidentTensor,
    ShardInfo,
    TensorSegment,
    build_expert_sidecar,
    load_expert_manifest,
    save_expert_manifest,
)
from mtplx.expert_runtime import (  # noqa: E402
    ExpertStreamingConfig,
    ExpertStreamingRuntime,
    RoutingPhase,
)
from mtplx.expert_slots import ExpertSlotState  # noqa: E402
from mtplx.expert_streaming_models import ExpertStreamingModelSpec  # noqa: E402
from mtplx.models.deepseek_v41_moe import Gate, _gate_prefix_impl  # noqa: E402
from mtplx.models.expert_mlx import (  # noqa: E402
    HotExpertSwitchGLU,
    make_mlx_component_bank_allocator,
)

import f2.issue as fi  # noqa: E402
import f2.full_config as fc  # noqa: E402
import f2.plane_lane_prefetch as pl  # noqa: E402
from f2.priority_reads import PriorityReads, native_worker_count  # noqa: E402
from f2.run_full_install import select_prefetch_sources  # noqa: E402
import f2_predictor as F  # noqa: E402  (the sound offline core the live lane matches)

_REAL_EVAL = mx.eval
_SSP_FLAG = "MTPLX_DSV41_SINGLE_SLOT_POOL"

# Tiny component-bank record: 3 weight planes (U32) + per-plane scales/biases (BF16).
COMPONENTS = (
    ("gate_proj.weight", 2048, "U32", (64, 8)), ("gate_proj.scales", 128, "BF16", (64, 1)),
    ("gate_proj.biases", 128, "BF16", (64, 1)), ("up_proj.weight", 2048, "U32", (64, 8)),
    ("up_proj.scales", 128, "BF16", (64, 1)), ("up_proj.biases", 128, "BF16", (64, 1)),
    ("down_proj.weight", 2048, "U32", (64, 8)), ("down_proj.scales", 128, "BF16", (64, 1)),
    ("down_proj.biases", 128, "BF16", (64, 1)),
)
_RECORD_BYTES = sum(item[1] for item in COMPONENTS)
_HIDDEN = 64
_N_ROUTED = 16
_TOP_K = 2


@pytest.fixture(autouse=True)
def _cpu_and_flag():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    saved = os.environ.get(_SSP_FLAG)
    os.environ[_SSP_FLAG] = "1"  # arms runtime._single_slot_pool (with cache_scope=layer)
    try:
        yield
    finally:
        mx.set_default_device(prev)
        if saved is None:
            os.environ.pop(_SSP_FLAG, None)
        else:
            os.environ[_SSP_FLAG] = saved


# ---------------------------------------------------------------------------
# Real sidecar-backed tiny deepseek-v41 runtime (4 routed layers x 16 experts).
# ---------------------------------------------------------------------------
def _build_artifact(tmp_path, *, layers=4, experts=_N_ROUTED, top_k=_TOP_K):
    root = tmp_path / "artifact"
    root.mkdir(parents=True, exist_ok=True)
    total = layers + 2
    spec = ExpertStreamingModelSpec(
        key="deepseek-v41-f2-cpu-test", display_name="F2 CPU", source_model="t/s",
        source_revision="r", quant_model="t/q", quant_revision="qr",
        total_tensor_bytes=layers * experts * _RECORD_BYTES + 1, total_layers=total,
        routed_layer_start=1, routed_layer_count=layers, expert_count=experts, top_k=top_k,
        hidden_size=_HIDDEN, expert_hidden_size=_HIDDEN, quant_bits=4, quant_group_size=64,
        quant_parameter_bytes=2, router_storage="bfloat16", router_matmul_dtype="float32",
        router_bytes=0, kv_bytes_per_token=16, mtp_layer_index=total, mtp_included=False,
    )
    raw = bytearray()
    records: list[ExpertRecord] = []
    for layer in spec.routed_layer_indices:
        for expert in range(experts):
            segments = []
            payload = bytearray()
            for ci, (comp, length, dtype, shape) in enumerate(COMPONENTS):
                blob = bytes([(layer * 64 + expert * 16 + ci + 1) % 251]) * length
                off = len(raw)
                raw.extend(blob)
                payload.extend(blob)
                segments.append(TensorSegment(
                    component=comp, tensor=f"model.layers.{layer}.mlp.switch_mlp.{comp}",
                    shard="source.bin", offset=off, length=length, dtype=dtype, shape=shape))
            records.append(ExpertRecord(
                layer=layer, expert=expert, logical_bytes=len(payload),
                segments=tuple(segments), sha256=hashlib.sha256(payload).hexdigest()))
    resident_off = len(raw)
    raw.append(123)
    (root / "source.bin").write_bytes(raw)
    manifest = ExpertManifest(
        model_key=spec.key, source_repo=spec.quant_model, source_revision=spec.quant_revision,
        quant_bits=4, quant_group_size=64, quant_mode="affine",
        artifact_tensor_bytes=spec.total_tensor_bytes, resident_tensor_bytes=1,
        routed_expert_bytes=spec.routed_expert_bytes,
        shards=(ShardInfo(name="source.bin", size=len(raw), header_bytes=1, header_sha256="fixture-header"),),
        resident_tensors=(ResidentTensor(tensor="model.norm.flag", shard="source.bin",
                          offset=resident_off, length=1, dtype="U8", shape=(1,)),),
        records=tuple(records),
    ).with_digest()
    manifest.validate_structure()
    manifest = build_expert_sidecar(manifest, root, root / "experts.bin", alignment=1)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    return root, spec, manifest_path


def _open_f2_runtime(tmp_path, *, ring=16, transient=48, resident_slots=8,
                     resource_telemetry=False, **overrides):
    root, spec, manifest_path = _build_artifact(tmp_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    pad = (transient + ring + resident_slots + 8) * spec.expert_record_bytes
    kw = dict(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(resident_slots) + pad,
        max_live_kv_tokens=0, runtime_reserve_bytes=0, transient_slots=transient,
        slot_layout="component-banks", cache_scope="layer", prefetch_slots=ring,
        decode_miss_records_per_part=3, overlap_miss_reads=True, split_route_release="deferred",
        io_read_fanout=4, resource_telemetry=resource_telemetry, verify_artifact_headers=False,
        verify_record_hashes=False,
    )
    kw.update(overrides)
    config = ExpertStreamingConfig(**kw)
    runtime = ExpertStreamingRuntime.open(
        root, manifest_path, config, spec=spec,
        buffer_allocator=make_mlx_component_bank_allocator(
            config.memory_plan(spec), spec, load_expert_manifest(manifest_path)),
        device_synchronize=mx.synchronize, apply_memory_cap=False,
    )
    return runtime, spec


def _gate_args(spec):
    return SimpleNamespace(
        hidden_size=spec.hidden_size, num_experts_per_tok=spec.top_k,
        scoring_func="sqrtsoftplus", gate_temp=1.0, norm_topk_prob=True,
        routed_scaling_factor=1.0, n_routed_experts=spec.expert_count,
    )


def _random_gate(spec, *, seed):
    mx.random.seed(seed)
    gate = Gate(spec.routed_layer_start + 1, _gate_args(spec))
    gate.weight = (0.5 * mx.random.normal((spec.expert_count, spec.hidden_size))).astype(mx.bfloat16)
    gate.e_score_correction_bias = (0.1 * mx.random.normal((spec.expert_count,))).astype(mx.float32)
    _REAL_EVAL(gate.weight, gate.e_score_correction_bias)
    return gate


def _settle_prefetch(runtime):
    with runtime._prefetch_lock:
        pending = tuple(runtime._prefetch_futures)
    for future in pending:
        try:
            future.result()
        except BaseException:
            pass


def _cpu_ops(spec):
    class _CpuOps:
        contract = pl.OpsContract(
            spec.hidden_size, spec.expert_hidden_size, spec.top_k,
            spec.quant_bits, spec.quant_group_size, spec.expert_codec, spec.swiglu_limit,
        )

        def gate_up(self, *a, **k):  # pragma: no cover - never called on CPU
            raise NotImplementedError("CPU ops stub: the packed gather runs only on the GPU smoke")

        def down(self, *a, **k):  # pragma: no cover
            raise NotImplementedError("CPU ops stub")

    return _CpuOps


# ===========================================================================
# A. full_config.py -- transition-window + ring R admission
# ===========================================================================
def _paired_config_fields(ring):
    # PairedPrefetchConfig's proven-valid values for a deepseek-v41 transition-window
    # config, generalised: any deepseek-v41 key + the retained structural fields.
    return dict(
        model_key="deepseek-v41-flash-expert-mxfp4", memory_limit_bytes=10 * 1024 ** 3,
        expert_cache_limit_bytes=315 * 18800640, cache_policy="transition-window",
        cache_scope="layer", slot_layout="component-banks", transient_slots=48,
        decode_miss_records_per_part=3, split_route_release="deferred",
        overlap_miss_reads=True, prefetch_slots=ring, max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
    )


@pytest.mark.parametrize("ring", [16, 32])
def test_full_config_admits_transition_window_plus_ring(ring):
    cfg = fc.FullPrefetchConfig(**_paired_config_fields(ring))
    assert cfg.prefetch_slots == ring  # restored after native validation
    assert cfg.cache_policy == "transition-window"


@pytest.mark.parametrize("field,bad", [
    ("prefetch_slots", 8),            # not an admitted ring size
    ("prefetch_slots", 0),            # ring disabled
    ("transient_slots", 32),          # off the retained 48
    ("cache_policy", "frequency"),    # not transition-window
    ("cache_scope", "global"),
    ("slot_layout", "flat"),
    ("decode_miss_records_per_part", 2),
    ("split_route_release", "fenced"),
    ("model_key", "qwen3-8"),
])
def test_full_config_refuses_off_geometry(field, bad):
    fields = _paired_config_fields(32)
    fields[field] = bad
    with pytest.raises((ValueError, TypeError)):
        fc.FullPrefetchConfig(**fields)


def test_ring_reserve_bytes():
    assert fc.EXPERT_WEIGHT_RECORD_BYTES == 17_694_720 == F.EXPERT_RECORD_BYTES
    assert fc.ring_reserve_bytes(32) == 566_231_040
    assert fc.ring_reserve_bytes(16) == 283_115_520
    assert fc.ring_reserve_bytes(0) == 0


# ===========================================================================
# B. priority_reads.py -- N workers, demand priority, in-flight finishes
# ===========================================================================
def test_priority_reads_worker_count_and_native_default():
    pr = PriorityReads(7)
    try:
        assert pr.workers == 7
    finally:
        pr.shutdown()
    with pytest.raises(ValueError):
        PriorityReads(0)
    assert native_worker_count(SimpleNamespace(_fanout_pool_workers=15)) == 15
    assert native_worker_count(SimpleNamespace(_fanout_pool_workers=0)) == 1  # floor


def test_demand_preempts_queued_speculative_and_active_finishes():
    pr = PriorityReads(1)  # one worker: ordering is observable
    started = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def blocker(_job):
        started.set()
        release.wait(timeout=5.0)
        order.append("blocker")

    def record(tag):
        return lambda _job: order.append(tag)

    try:
        f_block = pr.submit(blocker, None, priority=1)   # occupy the worker
        assert started.wait(timeout=5.0)
        f_spec = pr.submit(record("speculative"), None, priority=1)  # queued behind
        f_dem = pr.submit(record("demand"), None, priority=0)        # jumps the queue
        release.set()
        f_block.result(timeout=5.0); f_dem.result(timeout=5.0); f_spec.result(timeout=5.0)
    finally:
        pr.shutdown()
    # the active blocker finished; the demand read preempted the queued speculative one.
    assert order == ["blocker", "demand", "speculative"]


def test_cancelled_future_is_skipped():
    pr = PriorityReads(1)
    started = threading.Event(); release = threading.Event()
    ran: list[str] = []

    def blocker(_j):
        started.set(); release.wait(timeout=5.0)

    try:
        pr.submit(blocker, None, priority=1)
        assert started.wait(timeout=5.0)
        f = pr.submit(lambda _j: ran.append("x"), None, priority=1)
        assert f.cancel()  # cancel while still queued
        release.set()
        pr.shutdown()
    finally:
        release.set()
    assert ran == []  # a cancelled queued read is skipped


def test_shutdown_closes_queue():
    pr = PriorityReads(2)
    pr.shutdown()
    with pytest.raises(RuntimeError):
        pr.submit(lambda _j: None, None, priority=0)


# ===========================================================================
# C. issue.py -- the live predictor (device biased-gate max + host top-k)
# ===========================================================================
def test_gate_predictor_merged_is_max_over_rows_of_biased(tmp_path):
    runtime, spec = _open_f2_runtime(tmp_path)
    try:
        gate = _random_gate(spec, seed=7)
        rows = 6  # M=6 verify shape
        mx.random.seed(11)
        tokens = (0.3 * mx.random.normal((rows, spec.hidden_size))).astype(mx.bfloat16)
        _REAL_EVAL(tokens)
        # reference: the exact eager prefix Gate.__call__ ranks, maxed over rows.
        _scores, biased = _gate_prefix_impl(
            tokens, gate.weight, gate.e_score_correction_bias, 1.0, "sqrtsoftplus")
        ref = np.asarray(mx.max(biased, axis=0).tolist(), dtype=np.float64)
        got = np.asarray(fi.GatePredictor(gate).merged(tokens).tolist(), dtype=np.float64)
        assert np.array_equal(got, ref), "merged != max-over-rows of the native biased gate score"
    finally:
        runtime.close()


def test_issue_prepare_adds_no_host_sync(tmp_path):
    """fix #2 / barrier parity: prepare rides the indices barrier -- exactly ONE
    main-thread mx.eval, the same the control spends on mx.eval(indices)."""
    runtime, spec = _open_f2_runtime(tmp_path)
    try:
        gate = _random_gate(spec, seed=3)
        target = spec.routed_layer_start
        issue = fi.Issue(runtime, target, fi.GatePredictor(gate), top_k=3)
        rows = 6
        tokens = mx.zeros((rows, spec.hidden_size), dtype=mx.bfloat16)
        indices = mx.zeros((rows, spec.top_k), dtype=mx.int32)
        main = threading.main_thread()
        count = {"n": 0}
        real = fi.mx.eval

        def counting(*a, **k):
            if threading.current_thread() is main:
                count["n"] += 1
            return real(*a, **k)

        fi.mx.eval = counting
        try:
            issue.prepare(tokens, indices)
        finally:
            fi.mx.eval = real
        assert count["n"] == 1, f"prepare must ride ONE indices barrier, spent {count['n']}"
    finally:
        runtime.close()


def test_issue_ranking_matches_offline_scorer_and_excludes_ready(tmp_path):
    """The live device-merge + host-rank pipeline == the offline merge_rank_exclude,
    and READY owners of the target layer are dropped from the issue set."""
    runtime, spec = _open_f2_runtime(tmp_path, ring=16)
    try:
        target = spec.routed_layer_start
        # Warm experts 0,1 to READY-resident (persistent) for the target layer.
        sw = HotExpertSwitchGLU(runtime, target)
        for e in (0, 1):
            _REAL_EVAL(sw(mx.zeros((1, 1, spec.hidden_size), dtype=mx.bfloat16),
                          mx.array([[e] * spec.top_k], dtype=mx.int32)))
        runtime.flush_deferred_slot_releases(evaluate=True)
        ready = {
            s.expert for s in (tuple(runtime.slots._persistent.values())
                               + tuple(runtime.slots._transient)
                               + tuple(runtime.slots._prefetch.values()))
            if s.state is ExpertSlotState.READY and s.layer == target
        }
        assert {0, 1} <= ready

        gate = _random_gate(spec, seed=21)
        issue = fi.Issue(runtime, target, fi.GatePredictor(gate), top_k=3)
        rows = 6
        mx.random.seed(5)
        tokens = (0.4 * mx.random.normal((rows, spec.hidden_size))).astype(mx.bfloat16)
        indices = mx.zeros((rows, spec.top_k), dtype=mx.int32)
        _REAL_EVAL(tokens)

        # Offline reference: the biased ROWS + the same READY mask.
        _scores, biased = _gate_prefix_impl(
            tokens, gate.weight, gate.e_score_correction_bias, 1.0, "sqrtsoftplus")
        biased_np = np.asarray(biased.tolist(), dtype=np.float64)
        ready_mask = np.zeros(spec.expert_count, dtype=bool)
        for e in ready:
            ready_mask[e] = True
        ref_ids = F.merge_rank_exclude(biased_np, ready_mask, 3)

        issue.prepare(tokens, indices)
        n = issue()
        _settle_prefetch(runtime)
        bank = runtime._banks[target]
        issued = [e for e in ref_ids if bank.prefetch_ticket(e) is not None
                  or e in bank._prefetch_expert_to_slot]
        assert n == len(ref_ids), (n, ref_ids)
        # every issued id is a top-k non-READY expert; none is a READY owner.
        assert set(issued) == set(ref_ids)
        assert not (set(ref_ids) & ready), "a READY owner was issued"
    finally:
        runtime.close()


# ===========================================================================
# D. ring / reconcile seams on the real runtime (issued via the live path)
# ===========================================================================
def test_speculative_reads_land_in_ring_slots(tmp_path):
    runtime, spec = _open_f2_runtime(tmp_path, ring=16)
    try:
        target = spec.routed_layer_start
        assert runtime.prefetch_experts(target, [6, 7], verify=True) == 2
        _settle_prefetch(runtime)
        bank = runtime._banks[target]
        # publish the settled reads (generation-thread step) and confirm ring tenancy.
        runtime.prefetch_experts(target, [])
        published = bank._prefetch_expert_to_slot
        assert set([6, 7]) <= set(published), published
        base = runtime.plan.transient_slots + runtime.slots.global_persistent_slots
        for e in (6, 7):
            assert published[e] >= base  # a shared-ring slot, above pool + transient
        assert runtime.counters.prefetch_issued >= 2
        assert runtime.counters.prefetch_bytes >= 2 * spec.expert_record_bytes
    finally:
        runtime.close()


def test_demanded_inflight_tenant_is_awaited_not_reread(tmp_path):
    runtime, spec = _open_f2_runtime(tmp_path, ring=16)
    target = spec.routed_layer_start
    gate = threading.Event()
    real_load = runtime.slots.load_speculative

    def gated_load(layer, load):
        gate.wait(timeout=5.0)
        return real_load(layer, load)

    runtime.slots.load_speculative = gated_load
    try:
        assert runtime.prefetch_experts(target, [9], verify=True) == 1
        bank = runtime._banks[target]
        assert bank.prefetch_ticket(9) is not None  # in flight, not committed
        assert 9 not in bank.published_experts([9])
        releaser = threading.Timer(0.15, gate.set)
        releaser.start()
        lock = runtime._layer_locks[target]
        lock.acquire()
        try:
            runtime._reconcile_prefetch_for_route(target, (9,))
        finally:
            lock.release()
        releaser.join()
        # awaited + committed in place -> the true route hit-resolves, no re-read.
        assert runtime.counters.prefetch_awaited_inflight >= 1
        assert 9 in bank.published_experts([9])
    finally:
        gate.set()
        runtime.slots.load_speculative = real_load
        runtime.close()


def test_ring_hit_excluded_from_pool_and_stays_a_ring_tenant(tmp_path):
    """A committed ring tenant a true route needs resolves in place (no re-read, no
    pool load) AND stays a ring tenant -- it is NOT promoted into the persistent pool
    (the ring-tenancy finding)."""
    runtime, spec = _open_f2_runtime(tmp_path, ring=16, resident_slots=2)
    try:
        target = spec.routed_layer_start
        # experts 6,7 committed to the ring; never routed before.
        assert runtime.prefetch_experts(target, [6, 7], verify=True) == 2
        _settle_prefetch(runtime)
        runtime.prefetch_experts(target, [])  # publish -> committed ring tenants
        bank = runtime._banks[target]
        assert set([6, 7]) <= set(bank._prefetch_expert_to_slot)
        assert not (set([6, 7]) & set(bank.resident_experts))  # not in the pool

        before = runtime.counters.prefetch_hit_on_true_route
        sw = HotExpertSwitchGLU(runtime, target)
        out = sw(mx.zeros((2, 1, spec.hidden_size), dtype=mx.bfloat16),
                 mx.array([[6, 7], [6, 7]], dtype=mx.int32))
        _REAL_EVAL(out)
        runtime.flush_deferred_slot_releases(evaluate=True)
        # the ring commits were consumed as hits (not re-read as pool misses)...
        assert runtime.counters.prefetch_hit_on_true_route > before
        # ...and the experts REMAIN ring tenants -- a ring hit never enters the pool.
        assert set([6, 7]) <= set(bank._prefetch_expert_to_slot)
        assert not (set([6, 7]) & set(bank.resident_experts))
    finally:
        runtime.close()


def test_wasted_tenant_recycled(tmp_path):
    runtime, spec = _open_f2_runtime(tmp_path, ring=2)  # tiny ring forces recycling
    try:
        target = spec.routed_layer_start
        assert runtime.prefetch_experts(target, [4, 5], verify=True) == 2
        _settle_prefetch(runtime)
        runtime.prefetch_experts(target, [])  # commit 4,5 (never consumed)
        _settle_prefetch(runtime)
        assert runtime.prefetch_experts(target, [10, 11], verify=True) == 2  # recycle 4,5
        _settle_prefetch(runtime)
        runtime.prefetch_experts(target, [])
        assert runtime.counters.prefetch_wasted >= 2, runtime.counters.prefetch_wasted
    finally:
        runtime.close()


def test_failure_drain_leaves_no_leased_slot(tmp_path):
    runtime, spec = _open_f2_runtime(tmp_path, ring=16)
    target = spec.routed_layer_start
    real_load = runtime.slots.load_speculative

    def failing_load(layer, load):
        raise RuntimeError("synthetic speculative read failure")

    runtime.slots.load_speculative = failing_load
    try:
        assert runtime.prefetch_experts(target, [12], verify=True) == 1
        _settle_prefetch(runtime)
        runtime.prefetch_experts(target, [])  # apply completions -> invalidate the failed read
        bank = runtime._banks[target]
        # the ring assignment is forgotten (no ticket, not published); the slot is free.
        assert bank.prefetch_ticket(12) is None
        assert 12 not in bank.published_experts([12])
        assert 12 not in bank._prefetch_expert_to_slot
        runtime.slots.load_speculative = real_load
        # the freed ring slot is reassignable and the runtime is healthy.
        assert runtime.prefetch_experts(target, [13], verify=True) == 1
        _settle_prefetch(runtime)
    finally:
        runtime.slots.load_speculative = real_load
        runtime.close()


def test_output_identical_prefetch_on_vs_off(tmp_path):
    """The gather always uses the TRUE indices, so a stashed/issued prediction cannot
    change a routed output -- byte-identical at M=6 with prefetch engaged vs not."""
    target = None
    experts = [6, 7]
    idx = mx.array([[6, 7]] * 6, dtype=mx.int32)
    mx.random.seed(41)
    x = (0.3 * mx.random.normal((6, 1, _HIDDEN))).astype(mx.bfloat16)
    _REAL_EVAL(x, idx)

    # prefetch OFF: experts streamed on the true route itself.
    runtime, spec = _open_f2_runtime(tmp_path / "off", ring=16, resident_slots=2)
    target = spec.routed_layer_start
    try:
        out_off = HotExpertSwitchGLU(runtime, target)(x, idx)
        _REAL_EVAL(out_off)
        runtime.flush_deferred_slot_releases(evaluate=True)
    finally:
        runtime.close()

    # prefetch ON: the live predictor issues 6,7 first (committed), then the same
    # true route consumes them as ring hits.
    runtime2, spec2 = _open_f2_runtime(tmp_path / "on", ring=16, resident_slots=2)
    try:
        assert runtime2.prefetch_experts(target, experts, verify=True) == 2
        _settle_prefetch(runtime2)
        before = runtime2.counters.prefetch_hit_on_true_route
        out_on = HotExpertSwitchGLU(runtime2, target)(x, idx)
        _REAL_EVAL(out_on)
        runtime2.flush_deferred_slot_releases(evaluate=True)
        assert runtime2.counters.prefetch_hit_on_true_route > before, "ring not consumed"
        assert mx.array_equal(out_off, out_on), "prefetch changed the routed output"
    finally:
        runtime2.close()


# ===========================================================================
# E. plane_lane_prefetch.install -- gate + wiring on the real runtime
# ===========================================================================
def _install_kwargs(runtime, spec):
    ops = _cpu_ops(spec)
    routed = list(spec.routed_layer_indices)
    switches = {L: HotExpertSwitchGLU(runtime, L) for L in routed}
    ops_by_layer = {L: ops() for L in routed}
    gate_args = _gate_args(spec)
    sources = {
        L: fi.Issue(runtime, L + 1, fi.GatePredictor(Gate(L + 1, gate_args)), top_k=3)
        for L in routed if (L + 1) in routed
    }
    return switches, ops_by_layer, sources


def test_install_accepts_and_wires(tmp_path):
    runtime, spec = _open_f2_runtime(tmp_path, ring=16)
    try:
        native = native_worker_count(runtime.reader)
        assert native == 15  # fanout 4 -> max(4, 5*3)
        switches, ops_by_layer, sources = _install_kwargs(runtime, spec)
        runners = pl.install(runtime, switches, ops_by_layer, prefetch_sources=sources)
        # fix #1: the isolated reader has the native worker count, not four.
        assert isinstance(runtime.reader._fanout_executor, PriorityReads)
        assert runtime.reader._fanout_executor.workers == native
        assert isinstance(runtime._split_executor, pl.PartExecutor)
        assert isinstance(runtime.slots._executor, pl.ReaderExecutor)
        assert isinstance(runtime._prefetch_executor, pl.SpeculativeExecutor)
        routed = list(spec.routed_layer_indices)
        for L in routed:
            expected = pl.PrefetchDecode if (L + 1) in routed else pl.PackedDecode
            assert isinstance(runners[L], expected)
            assert switches[L]._run == runners[L].run
    finally:
        runtime.close()


def test_install_reader_workers_override_floor(tmp_path):
    runtime, spec = _open_f2_runtime(tmp_path, ring=16)
    try:
        switches, ops_by_layer, sources = _install_kwargs(runtime, spec)
        with pytest.raises(RuntimeError, match="native"):
            pl.install(runtime, switches, ops_by_layer, prefetch_sources=sources, reader_workers=4)
    finally:
        runtime.close()


# ===========================================================================
# F. run_full_install.select_prefetch_sources -- the source/target geometry
# ===========================================================================
def test_window_preflight_resolves_every_seam():
    """The CPU window preflight resolves every lane seam on the real classes (the
    machine-checked side of the seam table: zero unresolved names)."""
    import f2.window_preflight as wp
    assert wp._seam_checks() == [], "unresolved lane seam(s)"
    assert wp.main([]) == 0
    # a missing dependency is refused (before the service would be unloaded).
    assert wp.main(["--dep", "/no/such/f2/dep"]) == 1


def test_select_prefetch_sources_full_model():
    # Retained 0..39 routed; targets 4..39 predicted from sources 3..38; 0..3 unpredicted.
    sources = select_prefetch_sources(range(40), first_target=4)
    assert sources == list(range(3, 39))
    assert 39 not in sources          # no successor
    assert set(sources) & {0, 1, 2} == set()  # targets 1..3 are below first_target
    # tiny 4-layer analog (routed 1..4), first_target=2: sources 1..3 -> targets 2..4.
    assert select_prefetch_sources([1, 2, 3, 4], first_target=2) == [1, 2, 3]


@pytest.mark.parametrize("mutate", [
    "prefetch_zero", "transient_small", "resource_telemetry", "ops_mismatch",
    "dmrpp_wrong", "sources_not_subset",
])
def test_install_refuses_each_violated_invariant(tmp_path, mutate):
    if mutate == "prefetch_zero":
        # A ring is mandatory: open without one and install must refuse.
        runtime, spec = _open_f2_runtime(tmp_path, ring=16)
        try:
            switches, ops_by_layer, sources = _install_kwargs(runtime, spec)
            object.__setattr__(runtime.config, "prefetch_slots", 0)
            with pytest.raises(RuntimeError):
                pl.install(runtime, switches, ops_by_layer, prefetch_sources=sources)
        finally:
            runtime.close()
        return
    if mutate == "resource_telemetry":
        runtime, spec = _open_f2_runtime(tmp_path, ring=16, resource_telemetry=True)
    else:
        runtime, spec = _open_f2_runtime(tmp_path, ring=16)
    try:
        switches, ops_by_layer, sources = _install_kwargs(runtime, spec)
        if mutate == "transient_small":
            object.__setattr__(runtime.plan, "transient_slots", 32)
        elif mutate == "ops_mismatch":
            class _Bad:
                contract = pl.OpsContract(128, 128, spec.top_k, 4, 64, "affine", 0.0)
            ops_by_layer = {L: _Bad() for L in ops_by_layer}
        elif mutate == "dmrpp_wrong":
            object.__setattr__(runtime.config, "decode_miss_records_per_part", 2)
        elif mutate == "sources_not_subset":
            sources = dict(sources)
            sources[9999] = list(sources.values())[0]
        with pytest.raises(RuntimeError):
            pl.install(runtime, switches, ops_by_layer, prefetch_sources=sources)
    finally:
        runtime.close()
