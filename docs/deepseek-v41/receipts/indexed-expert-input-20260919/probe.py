"""Continuous Q4 output-projection expansion during expert I/O; no dense decode reads."""
import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import time

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent-held GPU guard required before MLX import')
signal.alarm(180)
ROOT = Path(__file__).resolve().parent
proof = json.loads((ROOT/'installation.json').read_text())
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
before = host_memory_snapshot()
bound = proof['static_incremental_bound_bytes']
if (not before['box']['ok'] or before['box']['used_bytes'] + bound > 110000000000
        or before['box']['wired_bytes'] + bound > 100*1024**3):
    raise RuntimeError('complete component memory bound does not fit')

import numpy as np
import mlx.core as mx
from library_identity import identify
from mtplx.expert_manifest import load_expert_manifest
from mtplx.expert_io import PositionalExpertReader
from mtplx.expert_slots import ExpertSlotPool
from mtplx.expert_runtime import ExpertStreamingRuntime, ExpertStreamingConfig
from mtplx.expert_streaming import RoutePlan, RoutingPhase, SlotLoad
from mtplx.expert_streaming_models import ExpertMemoryPlan, ExpertStreamingModelSpec
from mtplx.models.expert_mlx import make_mlx_component_bank_allocator, HotExpertSwitchGLU, expert_routing_phase
from packed_storage import load_layer, remove_raw_scales, bind_weight_reader, WEIGHT_BYTES
from restore_bank import restore
from output_store import OutputStore
import plane_lane
from indexed_ops import IndexedOps
NATIVE_OPS = plane_lane.PackedOps

library = identify(proof['strict_allocator'])
mx.set_memory_limit(10*1024**3)
mx.set_cache_limit(256*1024**2)
model = Path(proof['model_path'])
LAYER = proof['layer']
spec = ExpertStreamingModelSpec(**{**proof['spec'], 'full_indexer_layers':(), 'island_pin_order':()})
base_plan = ExpertMemoryPlan(**{**proof['plan'], 'persistent_slots_by_layer':()})
base_config = ExpertStreamingConfig(**{**proof['config'], 'persistent_slots_by_layer':(), 'island_layers':(), 'mmap_island_layers':()})
full_manifest = load_expert_manifest(model/'expert-manifest.json')
manifest = dataclasses.replace(full_manifest, records=tuple(r for r in full_manifest.records if r.layer==LAYER),
    routed_expert_bytes=spec.routed_expert_bytes, artifact_tensor_bytes=spec.total_tensor_bytes,
    resident_tensor_bytes=4096, resident_tensors=())

route_payload = json.loads((ROOT/'routes.json').read_text())
routes = route_payload['routes']
assert len(routes) == 206 and all(len(r) == 36 for r in routes)
artifact = json.loads((ROOT/'artifact/manifest.json').read_text())
scales = load_layer(ROOT/'artifact', artifact['layers'][LAYER], mx=mx)
reference = None
report = {'complete':False, 'construction':proof, 'library':library, 'before':before, 'arms':[]}


def run_arm(mode, sequence):
    global reference
    resident = False
    capacity = 84
    plan = dataclasses.replace(base_plan, persistent_slots=capacity, slots_per_layer=capacity,
        persistent_cache_bytes=capacity*spec.expert_record_bytes,
        expert_cache_limit_bytes=capacity*spec.expert_record_bytes,
        persistent_budget_bytes=capacity*spec.expert_record_bytes,
        total_limit_bytes=10*1024**3)
    config = dataclasses.replace(base_config, expert_cache_limit_bytes=plan.expert_cache_limit_bytes,
                                 memory_limit_bytes=10*1024**3)
    mx.synchronize()
    mx.clear_cache()
    reader = allocator = slots = runtime = switch = runners = store = None
    outputs = queries = inputs = route_arrays = None
    q = x = indices = y = shared = previous = shared_input = None
    result = {'mode':mode, 'sequence':sequence, 'capacity':110, 'initial_capacity':84}
    allocation_ns = 0
    try:
        reader = PositionalExpertReader(model, bypass_page_cache=True, use_native=False, io_read_fanout=4)
        allocator = make_mlx_component_bank_allocator(plan, spec, manifest)
        slots = ExpertSlotPool(spec, plan, manifest, reader, buffer_allocator=allocator,
            max_inflight_io_bytes=plan.transient_bytes, verify_hashes=False,
            device_synchronize=mx.synchronize, cache_scope='layer', resource_telemetry=False,
            batch_decode_reads=True)
        for bank in allocator.banks.values():
            remove_raw_scales(bank, mx=mx)
        for slot in allocator.slots.values():
            slot.nbytes = WEIGHT_BYTES
        runtime = ExpertStreamingRuntime(model, spec, config, manifest, plan, reader, slots, single_slot_pool=True)
        bind_weight_reader(reader)
        state = json.loads(json.dumps(route_payload['initial_bank']))
        entries = sorted(((int(e),s) for e,s in state['_expert_to_slot'].items()), key=lambda p:p[1])
        for start in range(0,len(entries),24):
            group = entries[start:start+24]
            ids = tuple(e for e,s in group)
            seed = RoutePlan(phase=RoutingPhase.PREFILL, experts=ids, slots=tuple(s for e,s in group),
                hits=(), misses=ids, loads=tuple(SlotLoad(e,s,True) for e,s in group), evictions=())
            ready = slots.ensure_route(LAYER, seed)
            ready.release(synchronize=False)
        restored = restore(state, policy='transition-window', single_pool=True)
        extra = capacity-restored.persistent_slots
        restored._slot_to_expert.extend([None]*extra)
        restored.persistent_slots = restored._persistent_capacity = capacity
        restored.slot_count += extra
        restored._protected_cap = max(1,int(capacity*.8))
        runtime._banks[LAYER] = restored
        switch = HotExpertSwitchGLU(runtime,LAYER)
        plane_lane.PackedOps = IndexedOps if mode == 'indexed-input' else NATIVE_OPS
        runners = plane_lane.install(runtime,{LAYER:switch},{LAYER:scales},early=True)
        store = OutputStore(model, full_manifest, proof, resident=resident)
        result['projection_storage_mode'] = 'cached-bf16' if resident else 'resident-packed-next-bf16'
        # Canonicalize the component's stripped-weight physical plan before
        # the phase extension; this also runs for the unchanged control.
        plan = dataclasses.replace(plan,persistent_cache_bytes=capacity*WEIGHT_BYTES,
            persistent_budget_bytes=capacity*WEIGHT_BYTES,expert_cache_limit_bytes=capacity*WEIGHT_BYTES,
            transient_bytes=48*WEIGHT_BYTES)
        plan = dataclasses.replace(plan,unallocated_bytes=plan.total_limit_bytes-plan.allocated_bytes)
        runtime.plan = slots.plan = allocator.plan = plan
        slots.allocated_bytes = (capacity+48)*WEIGHT_BYTES
        from extension import grow_rows
        from overflow import append_rows
        initial_arrays={name:id(value) for name,value in allocator.banks['persistent',LAYER].arrays.items()}
        phases=[grow_rows(runtime,capacity=110,layout='extension',mx=mx)]
        allocation_ns=sum(p['elapsed_ns'] for p in phases)
        result['allocation_phases']=phases
        result['allocation_ns']=allocation_ns
        result['existing_backings_unchanged']=all(id(allocator.banks['persistent',LAYER].arrays[name])==value
                                                for name,value in initial_arrays.items())
        if not result['existing_backings_unchanged']:
            raise RuntimeError('bank backing replacement does not match selected layout')
        result['layout']={str(key):bank.capacity for key,bank in allocator.banks.items()}
        if runtime.plan.slots_per_layer!=110 or runtime.slots.allocated_bytes!=(110+48)*WEIGHT_BYTES:
            raise RuntimeError('final physical capacity differs')
        rng = np.random.default_rng(20260919)
        inputs = [mx.array(rng.normal(0,.15,(1,6,32768)).astype(np.float32)).astype(mx.bfloat16) for _ in routes]
        route_arrays = [mx.array(ids,mx.int32).reshape(1,6,6) for ids in routes]
        shared_input = mx.array(rng.normal(0,.15,(1,6,5120)).astype(np.float32)).astype(mx.bfloat16)
        mx.eval(inputs,route_arrays,shared_input)
        outputs, queries = [], []
        previous = mx.array(0,mx.bfloat16)
        mx.eval(previous)
        def resident_shared():
            return mx.tanh(shared_input)
        def issue_next(next_step):
            def submit():
                store.issue(next_step)
                return mx.tanh(shared_input)
            return submit
        callbacks = ([resident_shared]*len(routes) if resident else
                     [issue_next(i+1) for i in range(len(routes)-1)] + [resident_shared])
        acquire = store.resident_acquire if resident else store.acquire
        before_reads = reader.metrics.as_dict()
        started = time.perf_counter_ns()
        if not resident:
            store.issue(0)
        for call in range(len(routes)):
            if call == 4:
                warm_started = time.perf_counter_ns()
            weights = acquire(call)
            # Both arms retain the real previous-layer dependency. The saved
            # routes replace routing arithmetic only; their barrier still covers
            # the entire query projection before a buffer may be retired.
            qi = inputs[call] + previous*0
            q = store.project(qi,weights,call)
            x = q
            indices = route_arrays[call] + (q[0,0,0]*0).astype(mx.int32)
            y,shared = switch._run(x,indices,shared_work=callbacks[call])
            previous = y[0,0,0,0]
            outputs.append(y)
            queries.append(q)
        mx.eval(outputs,queries,shared)
        ended = time.perf_counter_ns()
        runtime.flush_deferred_slot_releases(evaluate=True)
        after_reads = reader.metrics.as_dict()
        result.update(wall_ns=ended-started, warm_wall_ns=ended-warm_started,
                      charged_warm_wall_ns=ended-warm_started+allocation_ns,
                      charged_wall_ns=ended-started+allocation_ns,
                      expert_records=after_reads['records_read']-before_reads['records_read'],
                      expert_read_bytes=after_reads['read_bytes']-before_reads['read_bytes'],
                      dense_read_bytes=0,
                      mlx_peak_bytes=int(mx.get_peak_memory()))
        extra_start=84
        extra=[slots._persistent[LAYER,i] for i in range(extra_start,110)]
        if not any(p.expert is not None for p in extra):
            raise RuntimeError('extension bank was not used by the measured routes')
        result['extension_final_experts']=[p.expert for p in extra]
        digests = []
        for output,query in zip(outputs,queries):
            a,b = np.array(output.view(mx.uint16)),np.array(query.view(mx.uint16))
            # Raw BF16 exponent==255 rejects either infinity or NaN.
            if np.any((a&0x7f80)==0x7f80) or np.any((b&0x7f80)==0x7f80):
                raise RuntimeError('nonfinite component output')
            digests.append({'expert':hashlib.sha256(a.tobytes()).hexdigest(),
                            'query':hashlib.sha256(b.tobytes()).hexdigest()})
        if reference is None:
            reference = digests
        if digests != reference:
            raise RuntimeError('predictably expanded output projection or expert output differs from resident control')
        result.update(all_outputs_exact=True, output_digests=digests,
                      active_after_eval_bytes=int(mx.get_active_memory()))
    finally:
        mx.synchronize()
        if store is not None:
            store.close()
        if runtime is not None:
            runtime.close()
        elif slots is not None:
            slots.close()
        if reader is not None:
            reader.close()
        if allocator is not None:
            allocator.close()
        report['arms'].append(result)
        (ROOT/'probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print('INDEXED_INPUT_ARM',json.dumps({k:v for k,v in result.items() if k!='output_digests'}),flush=True)


try:
    for sequence,mode in enumerate(proof['arms']):
        with expert_routing_phase('decode'):
            run_arm(mode,sequence)
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
    controls = [a['warm_wall_ns'] for a in report['arms'] if a['mode']=='native-gather']
    candidates = [a['warm_wall_ns'] for a in report['arms'] if a['mode']=='indexed-input']
    report.update(complete=True,control_median_ns=statistics.median(controls),
                  candidate_median_ns=statistics.median(candidates),
                  candidate_over_control=statistics.median(candidates)/statistics.median(controls),
                  control_spread_fraction=(max(controls)-min(controls))/statistics.median(controls),
                  after=host_memory_snapshot())
finally:
    scales.clear()
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    report['active_after_close_bytes'] = int(mx.get_active_memory())
    (ROOT/'probe.json').write_text(json.dumps(report,indent=2)+'\n')
if report['active_after_close_bytes'] > 2*1024**2:
    raise RuntimeError('component retained Metal owners after close')
print('INDEXED_INPUT_COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ('construction','arms')}),flush=True)
