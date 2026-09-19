"""One-request phase installation, with coherent storage, policy and cleanup owners."""
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

from packed_storage import WEIGHT_BYTES, SCALE_BYTES, PROJECTIONS, load_layer, remove_raw_scales, bind_weight_reader, make_dispatch

ROOT = Path(__file__).resolve().parent


def install_growth(model, capacity, *, mx, admission):
    from mtplx.models.expert_mlx import HotExpertSwitchGLU, MlxComponentSlot
    from mtplx.expert_slots import _PhysicalSlot, ExpertSlotState
    from bank_growth_final import grow_bank

    rt = model._mtplx_expert_runtime
    pool, old_plan, old_config = rt.slots, rt.plan, rt.config
    allocator = pool._allocator
    layers = tuple(sorted(rt.spec.routed_layer_indices))
    old_capacity = old_plan.slots_per_layer
    inventory_blob = (ROOT / 'artifact/manifest.json').read_bytes()
    if hashlib.sha256(inventory_blob).hexdigest() != admission['packed_inventory_manifest_sha256']:
        raise RuntimeError('packed inventory changed')
    inventory = json.loads(inventory_blob)
    packed_bytes = inventory['packed_bytes']
    source = Path(inventory['source_path'])
    source_stat = source.stat()
    if {k: getattr(source_stat, k) for k in inventory['source_identity']} != inventory['source_identity']:
        raise RuntimeError('verified expert source identity changed')
    if hashlib.sha256((source.parent / 'expert-manifest.json').read_bytes()).hexdigest() != inventory['source_manifest_sha256']:
        raise RuntimeError('verified expert manifest changed')
    if (layers != tuple(range(40)) or rt.spec.expert_count != 384 or rt.spec.top_k != 6
        or rt.spec.expert_record_bytes != 18800640 or rt.spec.expert_codec != 'mxfp4'
        or old_capacity != admission['prefill_slots_per_layer'] or old_capacity != 84
        or not old_capacity < capacity <= 106
        or old_plan.persistent_slots_by_layer or old_plan.cache_scope != 'layer'
        or old_plan.transient_slots != 48 or old_plan.prefetch_ring_slots
        or pool.island_layers or pool._prefetch or rt._prefetch_ring is not None
        or rt._global_bank is not None or rt._derived_cache_policy or not rt._single_slot_pool
        or old_config.cache_policy != 'transition-window' or old_config.slot_layout != 'component-banks'
        or old_config.miss_shadow or old_config.expert_cache_limit_bytes is not None
        or old_config.max_live_kv_tokens != 17664 or not pool.prefer_sidecar or pool.verify_hashes
        or rt.reader._codec_record_map is not None or rt.reader.io_read_fanout != 4
        or not rt.reader.bypass_page_cache
        or any(os.environ.get(k, '0') != '0' for k in ('MTPLX_DSV41_DEVICE_ROUTE', 'MTPLX_DSV41_DEVICE_ROUTE_PINNED'))):
        raise RuntimeError('packed-scale installation requires the exact native single-request lane')
    if (allocator.plan != old_plan or pool.plan != old_plan
        or set(allocator.banks) != {('persistent', layer) for layer in layers} | {('transient', -1)}
        or set(rt._banks) != set(layers)
        or pool.allocated_bytes != (old_capacity * 40 + 48) * 18800640):
        raise RuntimeError('physical pool, allocator or policy ownership differs')
    switches = []
    for layer in layers:
        switch = model.model.layers[layer].mlp.switch_mlp
        bank, policy = allocator.banks['persistent', layer], rt._banks[layer]
        if (type(switch) is not HotExpertSwitchGLU or switch.runtime is not rt
            or switch.layer_index != layer or switch.codec != 'mxfp4'
            or switch.bits != 4 or switch.group_size != 32 or switch.swiglu_limit != 10.0
            or bank.capacity != old_capacity or policy.persistent_slots != old_capacity
            or len(policy._slot_to_expert) != old_capacity or policy.prefetch_slots
            or policy._prefetch_ring is not None or policy.transient_slots != 48):
            raise RuntimeError('layer geometry or installed switch differs')
        for row in range(old_capacity):
            physical = pool._persistent[layer, row]
            if physical.buffer.bank is not bank or physical.buffer.bank_index != row:
                raise RuntimeError('physical row ownership differs')
        switches.append(switch)
    expected_components = {p + s for p in PROJECTIONS for s in ('.weight', '.scales')}
    for bank in allocator.banks.values():
        if (set(bank.arrays) != expected_components or bank.record_bytes != 18800640
            or sum(value.nbytes for value in bank.arrays.values()) != bank.capacity * 18800640):
            raise RuntimeError('native component ownership differs')

    physical_after = (capacity * 40 + 48) * WEIGHT_BYTES
    delta = physical_after + packed_bytes - pool.allocated_bytes
    if delta != admission['growth_payload_bytes']:
        raise RuntimeError('phase allocation differs from the admitted exact delta')
    new_fixed = old_plan.fixed_bytes + packed_bytes - 48 * SCALE_BYTES
    new_limit = old_plan.total_limit_bytes + delta
    new_plan = replace(old_plan, total_limit_bytes=new_limit,
        resident_bytes=old_plan.resident_bytes + packed_bytes,
        transient_bytes=48 * WEIGHT_BYTES,
        persistent_budget_bytes=new_limit - new_fixed,
        persistent_cache_bytes=capacity * 40 * WEIGHT_BYTES,
        persistent_slots=capacity * 40, slots_per_layer=capacity)
    new_config = replace(old_config, memory_limit_bytes=new_limit)
    if (new_plan.allocated_bytes - old_plan.allocated_bytes != delta
        or new_plan.total_limit_bytes - new_plan.allocated_bytes != new_plan.unallocated_bytes):
        raise RuntimeError('fixed residents and physical slot accounting disagree')

    owners = {}
    plane_runners = {}
    state = dict(phase='installed', prefill_slots_per_layer=old_capacity,
        decode_slots_per_layer=capacity, growth_payload_bytes=delta,
        resident_packed_scales_bytes=packed_bytes, source_record_bytes=18800640,
        decode_weight_record_bytes=WEIGHT_BYTES,
        timing_scope='scale installation, raw backing release and one weight resize are inside decode wall time')

    def reject_prefill(*args, **kwargs):
        raise RuntimeError('one-request packed-scale benchmark cannot perform another prefill; reload')

    def closed_dispatch(*args, **kwargs):
        raise RuntimeError('packed-scale runtime is closed')

    def fixed_allocator(size, label):
        raise RuntimeError('packed decode storage is fixed at the installed capacity')

    def close():
        for switch in switches:
            object.__setattr__(switch, '_dispatch_component_bank', closed_dispatch)
            object.__setattr__(switch, '_run', closed_dispatch)
        plane_runners.clear()
        for arrays in owners.values():
            arrays.clear()
        owners.clear()
        allocator.close()

    fixed_allocator.banks = allocator.banks
    fixed_allocator.slots = allocator.slots
    fixed_allocator.plan = new_plan
    fixed_allocator.backend = 'mlx-metal-weight-banks+resident-packed-scales'
    fixed_allocator.close = close

    def transition():
        if state['phase'] != 'installed':
            raise RuntimeError('packed phase can execute only once')
        start = time.perf_counter()
        state['phase'] = 'transitioning'
        rt.flush_deferred_slot_releases(evaluate=True)
        rt._drain_prefetch_loads()
        pool._drain_completion_fences()
        mx.synchronize()
        rt._raise_if_unhealthy()
        with ExitStack() as stack:
            for locks in (rt._layer_locks, pool._ensure_locks):
                for layer in layers:
                    if not locks[layer].acquire(blocking=False):
                        raise RuntimeError('layer or I/O setup is still owned at the phase transition')
                    stack.callback(locks[layer].release)
            stack.enter_context(pool._lifecycle)
            if pool._closing or pool._closed or pool._cleanup_owners or pool.metrics.as_dict()['active_routes']:
                raise RuntimeError('physical pool is not quiescent')
            for physical in (*pool._persistent.values(), *pool._transient):
                if physical.pins or physical.pin_claims or physical.state is ExpertSlotState.LOADING:
                    raise RuntimeError('slot still has an I/O or GPU owner')
            if rt._device_route_probes:
                raise RuntimeError('pending route probes can retain old backing arrays')
            for field in ('_device_route_lut', '_device_route_lut_snapshot', '_device_route_pinned_lut', '_device_route_pinned_snapshot'):
                getattr(rt, field).clear()
            rt._device_route_lut_dirty.update({layer: True for layer in layers})
            rt._device_route_pinned_lut_dirty.update({layer: True for layer in layers})
            mx.clear_cache()
            state['active_before_bytes'] = int(mx.get_active_memory())
            if state['active_before_bytes'] > admission['transition_start_active_bound_bytes']:
                raise RuntimeError('transition start exceeds its pre-admitted bound')
            # Register cleanup before allocating any packed owner or changing a
            # bank. A partial transition is fatal and remains fully owned.
            pool._allocator = fixed_allocator
            try:
                released = remove_raw_scales(allocator.banks['transient', -1], mx=mx)
                for layer, switch in zip(layers, switches):
                    bank = allocator.banks['persistent', layer]
                    released += remove_raw_scales(bank, mx=mx)
                    owners[layer] = load_layer(ROOT / 'artifact', inventory['layers'][layer], mx=mx)
                    grow_bank(bank, capacity, mx=mx)
                    object.__setattr__(switch, '_dispatch_component_bank', make_dispatch(owners[layer], mx=mx))
                    mx.clear_cache()
                for physical in (*pool._persistent.values(), *pool._transient):
                    physical.buffer.nbytes = WEIGHT_BYTES
                for layer in layers:
                    bank = allocator.banks['persistent', layer]
                    for row in range(old_capacity, capacity):
                        label = f'layer-{layer}-persistent-{row}'
                        buffer = MlxComponentSlot(bank, row, label=label)
                        allocator.slots[label] = buffer
                        pool._persistent[layer, row] = _PhysicalSlot(label, buffer)
                for policy in rt._banks.values():
                    policy._slot_to_expert.extend([None] * (capacity - old_capacity))
                    policy.persistent_slots = policy._persistent_capacity = capacity
                    policy.slot_count = capacity + policy.transient_slots + policy.prefetch_slots
                    policy._protected_cap = max(1, int(capacity * 0.8))
                bind_weight_reader(rt.reader)
                pool.plan = rt.plan = new_plan
                rt.config = new_config
                from plane_lane import install as install_plane_lane
                plane_runners.update(install_plane_lane(rt, dict(zip(layers, switches)), owners))
                pool.allocated_bytes = physical_after
                pool.buffer_backend = fixed_allocator.backend
                pool._persistent_route_capacity = capacity
                pool._persistent_route_capacities = {layer: capacity for layer in layers}
                rt._per_layer_record_bytes = {layer: WEIGHT_BYTES for layer in layers}
                rt._representative_record_bytes = WEIGHT_BYTES
                object.__setattr__(model.model, '_forward_layer_major', reject_prefill)
                if released != (40 * old_capacity + 48) * SCALE_BYTES:
                    raise RuntimeError('released raw scales differ from the admitted inventory')
                actual_banks = sum(a.nbytes for bank in allocator.banks.values() for a in bank.arrays.values())
                actual_scales = sum(a.nbytes for scales in owners.values() for component in scales.values() for a in component)
                if actual_banks != physical_after or actual_scales != packed_bytes:
                    raise RuntimeError('phase report differs from physical storage owners')
            except BaseException:
                state['phase'] = 'failed'
                raise
        mx.synchronize()
        mx.clear_cache()
        state.update(phase='decode', plane_readiness='gate_up_then_complete_record', plane_layers=len(plane_runners), growth_seconds=time.perf_counter() - start,
            active_after_bytes=int(mx.get_active_memory()), cache_after_bytes=int(mx.get_cache_memory()),
            transition_peak_bytes=int(mx.get_peak_memory()), raw_scale_backing_released_bytes=released,
            physical_allocated_bytes=physical_after, plan_limit_bytes=new_limit,
            plan_persistent_bytes=new_plan.persistent_cache_bytes)
        return state

    return transition, state
