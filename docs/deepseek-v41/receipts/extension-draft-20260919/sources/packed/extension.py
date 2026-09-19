"""Grow the original bank or add a separate extension at a quiescent boundary."""
from contextlib import ExitStack
from dataclasses import replace
import os
import time


def grow_rows(runtime, *, capacity, layout, mx):
    from mtplx.models.expert_mlx import MlxComponentBank, MlxComponentSlot
    from mtplx.expert_slots import _PhysicalSlot, ExpertSlotState
    from packed_storage import remove_raw_scales, WEIGHT_BYTES

    pool,old_plan,config=runtime.slots,runtime.plan,runtime.config
    allocator=pool._allocator
    layers=tuple(runtime.spec.routed_layer_indices)
    old=old_plan.slots_per_layer
    if old!=84 or not old<capacity<=113 or layout not in ('resize','extension'):
        raise RuntimeError('exact first-growth geometry required')
    added=capacity-old
    expected={("persistent",layer) for layer in layers}|{('transient',-1)}
    if (set(allocator.banks)!=expected or old_plan.cache_scope!='layer'
            or old_plan.persistent_slots_by_layer or old_plan.transient_slots!=48
            or old_plan.prefetch_ring_slots or pool._prefetch or runtime._prefetch_ring is not None
            or runtime._global_bank is not None or not runtime._single_slot_pool
            or config.slot_layout!='component-banks' or config.cache_policy!='transition-window'
            or config.split_route_release!='deferred'
            or (runtime.spec.hidden_size,runtime.spec.expert_hidden_size,runtime.spec.top_k,runtime.spec.expert_codec)!=(5120,2304,6,'mxfp4')
            or any(os.environ.get(k,'0')!='0' for k in ('MTPLX_DSV41_DEVICE_ROUTE','MTPLX_DSV41_DEVICE_ROUTE_PINNED'))):
        raise RuntimeError('overflow row requires the exact packed single-request lane')
    expected_bytes=(len(layers)*old+48)*WEIGHT_BYTES
    if (old_plan.persistent_cache_bytes!=len(layers)*old*WEIGHT_BYTES
            or old_plan.transient_bytes!=48*WEIGHT_BYTES
            or pool.allocated_bytes!=expected_bytes
            or old_plan.allocated_bytes+old_plan.unallocated_bytes!=old_plan.total_limit_bytes):
        raise RuntimeError('packed plan/physical inventory differs')
    for layer in layers:
        bank=allocator.banks['persistent',layer]
        policy=runtime._banks[layer]
        if (bank.capacity!=old or policy.persistent_slots!=old
                or bank.record_bytes!=WEIGHT_BYTES or policy.prefetch_slots):
            raise RuntimeError('existing packed bank/policy capacity differs')
    before_rows={(layer,row):(id(pool._persistent[layer,row].buffer),
        id(pool._persistent[layer,row].buffer.bank),pool._persistent[layer,row].buffer.bank_index)
        for layer in layers for row in range(old)}
    delta=len(layers)*added*WEIGHT_BYTES
    plan=replace(old_plan,persistent_slots=old_plan.persistent_slots+len(layers)*added,
        slots_per_layer=capacity,persistent_cache_bytes=old_plan.persistent_cache_bytes+delta,
        persistent_budget_bytes=old_plan.persistent_budget_bytes+delta,
        expert_cache_limit_bytes=(None if old_plan.expert_cache_limit_bytes is None
                                  else old_plan.expert_cache_limit_bytes+delta),
        total_limit_bytes=old_plan.total_limit_bytes+delta)
    started=time.perf_counter_ns()
    runtime.flush_deferred_slot_releases(evaluate=True)
    runtime._drain_prefetch_loads()
    pool._drain_completion_fences()
    mx.synchronize()
    runtime._raise_if_unhealthy()
    with ExitStack() as stack:
        for locks in (runtime._layer_locks,pool._ensure_locks):
            for layer in layers:
                if not locks[layer].acquire(blocking=False):
                    raise RuntimeError('layer still owns a transaction')
                stack.callback(locks[layer].release)
        stack.enter_context(pool._lifecycle)
        if pool._closed or pool._closing or pool._cleanup_owners or pool.metrics.as_dict()['active_routes']:
            raise RuntimeError('pool is not quiescent')
        for physical in (*pool._persistent.values(),*pool._transient):
            if physical.pins or physical.pin_claims or physical.state is ExpertSlotState.LOADING:
                raise RuntimeError('physical slot has a live consumer or writer')
        for layer in layers:
            if layout=='resize':
                from bank_growth_final import grow_bank
                bank=allocator.banks['persistent',layer]
                grow_bank(bank,capacity,mx=mx)
                first_index=old
            else:
                bank=MlxComponentBank(capacity=added,record=pool._record_map[layer,0],
                                      label=f'layer-{layer}-extension')
                allocator.banks['overflow',layer]=bank
                remove_raw_scales(bank,mx=mx)
                first_index=0
            for offset in range(added):
                label=f'layer-{layer}-added-{old+offset}'
                buffer=MlxComponentSlot(bank,first_index+offset,label=label)
                allocator.slots[label]=buffer
                pool._persistent[layer,old+offset]=_PhysicalSlot(label,buffer)
        for layer in layers:
            policy=runtime._banks[layer]
            policy._slot_to_expert.extend([None]*added)
            policy.persistent_slots=policy._persistent_capacity=capacity
            policy.slot_count=capacity+48
            policy._protected_cap=max(1,int(capacity*.8))
        pool._persistent_route_capacity=capacity
        pool._persistent_route_capacities={layer:capacity for layer in layers}
        pool.allocated_bytes=expected_bytes+delta
        pool.plan=runtime.plan=allocator.plan=plan
        runtime.config=replace(config,memory_limit_bytes=plan.total_limit_bytes,
            expert_cache_limit_bytes=(None if config.expert_cache_limit_bytes is None
                                     else plan.expert_cache_limit_bytes))
        for field in ('_device_route_lut','_device_route_lut_snapshot','_device_route_pinned_lut','_device_route_pinned_snapshot'):
            getattr(runtime,field).clear()
        runtime._device_route_lut_dirty.update({layer:True for layer in layers})
        runtime._device_route_pinned_lut_dirty.update({layer:True for layer in layers})
    mx.synchronize()
    elapsed=time.perf_counter_ns()-started
    after_rows={(layer,row):(id(pool._persistent[layer,row].buffer),
        id(pool._persistent[layer,row].buffer.bank),pool._persistent[layer,row].buffer.bank_index)
        for layer in layers for row in range(old)}
    if before_rows!=after_rows:
        raise RuntimeError('an existing physical row moved')
    actual=sum(value.nbytes for bank in allocator.banks.values() for value in bank.arrays.values())
    if actual!=pool.allocated_bytes:
        raise RuntimeError('overflow allocation differs from physical accounting')
    return {'initial_capacity':old,'capacity':capacity,'layout':layout,'layers':len(layers),
        'added_payload_bytes':delta,'elapsed_ns':elapsed,'existing_row_owners_unchanged':True,
        'physical_allocated_bytes':actual,'scope':'Construction-only allocation; original row objects/indices preserved. Extension leaves all original bank backing arrays in place.'}
