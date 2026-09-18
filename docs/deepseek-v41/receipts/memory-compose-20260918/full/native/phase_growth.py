"""One-request benchmark installation: retain rows while growing after prefill.

This is deliberately outside general serving. A later prefill is rejected;
partial allocation failure aborts the child instead of resuming generation.
"""
from contextlib import ExitStack
from dataclasses import replace
import time


def install_growth(model, capacity, *, mx, admission):
    from mtplx.models.expert_mlx import make_mlx_component_bank_allocator
    from mtplx.expert_slots import _PhysicalSlot, ExpertSlotState
    from bank_growth_final import grow_bank

    rt = model._mtplx_expert_runtime
    pool, old_plan, old_config = rt.slots, rt.plan, rt.config
    layers = tuple(sorted(rt.spec.routed_layer_indices))
    old_capacity = old_plan.slots_per_layer
    if (len(layers) != 40 or rt.spec.expert_record_bytes != 18800640
        or old_capacity != admission['prefill_slots_per_layer'] or old_capacity != 84 or capacity <= old_capacity
        or old_plan.persistent_slots_by_layer or old_plan.cache_scope != 'layer'
        or old_plan.transient_slots != 48 or old_plan.prefetch_ring_slots
        or pool.island_layers or pool._prefetch or rt._prefetch_ring is not None
        or rt._global_bank is not None or rt._derived_cache_policy
        or not rt._single_slot_pool or old_config.cache_policy != 'transition-window'
        or old_config.slot_layout != 'component-banks' or old_config.miss_shadow
        or old_config.expert_cache_limit_bytes is not None
        or rt.config.max_live_kv_tokens != 17664):
        raise RuntimeError('post-prefill growth requires the exact static benchmark geometry')
    old_allocator = pool._allocator
    if (old_allocator.plan != old_plan or pool.plan != old_plan
        or set(old_allocator.banks) != {('persistent', layer) for layer in layers} | {('transient', -1)}
        or set(rt._banks) != set(layers)
        or pool.allocated_bytes != (old_capacity * len(layers) + 48) * rt.spec.expert_record_bytes):
        raise RuntimeError('allocator, physical pool and policy ownership differ')
    for layer in layers:
        bank, policy = old_allocator.banks['persistent', layer], rt._banks[layer]
        if (bank.capacity != old_capacity or policy.persistent_slots != old_capacity
            or len(policy._slot_to_expert) != old_capacity or policy.prefetch_slots
            or policy._prefetch_ring is not None or policy.transient_slots != 48):
            raise RuntimeError('bank geometry differs before installing growth')
        for row in range(old_capacity):
            physical = pool._persistent[layer, row]
            if physical.buffer.bank is not bank or physical.buffer.bank_index != row:
                raise RuntimeError('physical slot does not own the expected bank row')
    delta = (capacity - old_capacity) * len(layers) * rt.spec.expert_record_bytes
    new_config = replace(old_config, memory_limit_bytes=old_config.memory_limit_bytes + delta)
    new_plan = replace(old_plan,
        total_limit_bytes=old_plan.total_limit_bytes + delta,
        persistent_budget_bytes=old_plan.persistent_budget_bytes + delta,
        persistent_cache_bytes=old_plan.persistent_cache_bytes + delta,
        persistent_slots=capacity * len(layers), slots_per_layer=capacity)
    if (new_plan.fixed_bytes != old_plan.fixed_bytes
        or new_plan.allocated_bytes - old_plan.allocated_bytes != delta
        or new_plan.total_limit_bytes != new_config.memory_limit_bytes
        or delta != admission['growth_payload_bytes']):
        raise RuntimeError('phase plans disagree with the admitted exact byte delta')
    # The factory validates all manifest geometry now and binds the new immutable
    # plan. Its empty maps own no Metal allocation until the boundary transfer.
    new_allocator = make_mlx_component_bank_allocator(new_plan, rt.spec, rt.manifest)
    state = {'phase': 'installed', 'prefill_slots_per_layer': old_capacity,
             'decode_slots_per_layer': capacity, 'growth_payload_bytes': delta,
             'timing_scope': 'inside decode wall time, after the original prefill callback'}

    def reject_second_prefill(*args, **kwargs):
        raise RuntimeError('one-request growth benchmark cannot perform another prefill; reload at the prefill capacity')

    def transition():
        if state['phase'] != 'installed':
            raise RuntimeError('post-prefill storage transition can execute only once')
        state['phase'] = 'transitioning'
        start = time.perf_counter()
        rt.flush_deferred_slot_releases(evaluate=True)
        rt._drain_prefetch_loads()
        pool._drain_completion_fences()
        mx.synchronize()
        rt._raise_if_unhealthy()
        # The caller is the only generation owner. Refuse any outstanding owner
        # rather than waiting on a lock which this same thread could be holding.
        with ExitStack() as stack:
            for layer in layers:
                lock = rt._layer_locks[layer]
                if not lock.acquire(blocking=False):
                    raise RuntimeError('generation layer is still owned at the transition')
                stack.callback(lock.release)
            for layer in layers:
                lock = pool._ensure_locks[layer]
                if not lock.acquire(blocking=False):
                    raise RuntimeError('slot setup is still owned at the transition')
                stack.callback(lock.release)
            stack.enter_context(pool._lifecycle)
            if (pool._closing or pool._closed or pool._cleanup_owners
                or pool.metrics.as_dict()['active_routes']):
                raise RuntimeError('physical pool is not quiescent')
            for physical in (*pool._persistent.values(), *pool._transient):
                if physical.pins or physical.pin_claims or physical.state is ExpertSlotState.LOADING:
                    raise RuntimeError('an expert row still has an I/O or Metal owner')
            if rt._device_route_probes:
                raise RuntimeError('pending device-route probes can retain old bank storage')
            for field in ('_device_route_lut','_device_route_lut_snapshot',
                          '_device_route_pinned_lut','_device_route_pinned_snapshot'):
                getattr(rt, field).clear()
            rt._device_route_lut_dirty.update({layer: True for layer in layers})
            rt._device_route_pinned_lut_dirty.update({layer: True for layer in layers})
            mx.clear_cache()
            state['active_before_bytes'] = int(mx.get_active_memory())
            state['cache_before_bytes'] = int(mx.get_cache_memory())
            if state['active_before_bytes'] > admission['transition_start_active_bound_bytes']:
                raise RuntimeError('actual transition active memory exceeds its pre-admitted bound')
            try:
                for layer in layers:
                    grow_bank(old_allocator.banks['persistent', layer], capacity, mx=mx)
                    # This is an allocation boundary, outside all decode routes.
                    mx.clear_cache()
                new_allocator.banks.update(old_allocator.banks)
                new_allocator.slots.update(old_allocator.slots)
                new_slots = {}
                for layer in layers:
                    for row in range(old_capacity, capacity):
                        label = f'layer-{layer}-persistent-{row}'
                        new_slots[layer, row] = _PhysicalSlot(label, new_allocator(rt.spec.expert_record_bytes, label))
                pool._persistent.update(new_slots)
                pool._allocator = new_allocator
                pool.plan = new_plan
                pool.allocated_bytes += delta
                pool._persistent_route_capacity = capacity
                pool._persistent_route_capacities = {layer: capacity for layer in layers}
                for policy in rt._banks.values():
                    policy._slot_to_expert.extend([None] * (capacity - old_capacity))
                    policy.persistent_slots = capacity
                    policy._persistent_capacity = capacity
                    policy.slot_count = capacity + policy.transient_slots + policy.prefetch_slots
                    policy._protected_cap = max(1, int(capacity * 0.8))
                rt.plan, rt.config = new_plan, new_config
                # Ownership is transferred; calling old_allocator.close would
                # close the very same banks now owned by the new allocator.
                old_allocator.banks.clear()
                old_allocator.slots.clear()
            except BaseException:
                state['phase'] = 'failed'
                # Partial storage is still owned by the current pool allocator;
                # the full runner closes that pool and aborts the child.
                raise
            object.__setattr__(model.model, '_forward_layer_major', reject_second_prefill)
        mx.synchronize()
        mx.clear_cache()
        state.update(phase='decode', growth_seconds=time.perf_counter()-start,
            active_after_bytes=int(mx.get_active_memory()),
            cache_after_bytes=int(mx.get_cache_memory()),
            transition_peak_bytes=int(mx.get_peak_memory()),
            physical_allocated_bytes=pool.allocated_bytes,
            plan_limit_bytes=new_plan.total_limit_bytes,
            plan_persistent_bytes=new_plan.persistent_cache_bytes)
        return state
    return transition, state
