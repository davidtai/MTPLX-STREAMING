"""Continuous native Q4 query/expert component replay with bounded prefetch."""
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
from dense_io import QueryStore
import plane_lane

library = identify(proof['strict_allocator'])
mx.set_memory_limit(8*1024**3)
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
del full_manifest
route_payload = json.loads((ROOT/'routes.json').read_text())
routes = route_payload['routes']
assert len(routes) == 206 and all(len(r) == 36 for r in routes)
artifact = json.loads((ROOT/'artifact/manifest.json').read_text())
scales = load_layer(ROOT/'artifact', artifact['layers'][LAYER], mx=mx)
reference = None
report = {'complete':False, 'construction':proof, 'library':library, 'before':before, 'arms':[]}


def run_arm(mode, sequence):
    global reference
    resident = mode == 'resident-110'
    capacity = 110 if resident else 112
    plan = dataclasses.replace(base_plan, persistent_slots=capacity, slots_per_layer=capacity,
        persistent_cache_bytes=capacity*spec.expert_record_bytes,
        expert_cache_limit_bytes=capacity*spec.expert_record_bytes,
        persistent_budget_bytes=capacity*spec.expert_record_bytes,
        total_limit_bytes=8*1024**3)
    config = dataclasses.replace(base_config, expert_cache_limit_bytes=plan.expert_cache_limit_bytes,
                                 memory_limit_bytes=8*1024**3)
    mx.synchronize()
    mx.clear_cache()
    reader = allocator = slots = runtime = switch = runners = store = None
    outputs = queries = inputs = route_arrays = None
    q = x = indices = y = shared = previous = shared_input = None
    result = {'mode':mode, 'sequence':sequence, 'capacity':capacity}
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
        runners = plane_lane.install(runtime,{LAYER:switch},{LAYER:scales},early=True)
        store = QueryStore(model, proof['query_tensors'], mx=mx, resident=resident)
        result['query_backing_bytes'] = sum(a.nbytes for b in store.buffers for a in b.values())
        rng = np.random.default_rng(20260919)
        inputs = [mx.array(rng.normal(0,.15,(1,6,1280)).astype(np.float32)).astype(mx.bfloat16) for _ in routes]
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
        acquire = (lambda step: store.buffers[step%40]) if resident else store.acquire
        before_reads = reader.metrics.as_dict()
        started = time.perf_counter_ns()
        if not resident:
            store.start()
        for call in range(len(routes)):
            if call == 4:
                warm_started = time.perf_counter_ns()
            weights = acquire(call)
            # Both arms retain the real previous-layer dependency. The saved
            # routes replace routing arithmetic only; their barrier still covers
            # the entire query projection before a buffer may be retired.
            qi = inputs[call] + previous*0
            q = mx.quantized_matmul(qi, weights['weight'], weights['scales'],
                                    transpose=True, group_size=32, bits=8, mode='mxfp8')
            x = q[...,:5120]
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
                      expert_records=after_reads['records_read']-before_reads['records_read'],
                      expert_read_bytes=after_reads['read_bytes']-before_reads['read_bytes'],
                      dense_read_bytes=0 if resident else len(routes)*43253760,
                      mlx_peak_bytes=int(mx.get_peak_memory()))
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
            raise RuntimeError('streamed query or expert output differs from resident control')
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
    print('DENSE_PREFETCH_ARM',json.dumps({k:v for k,v in result.items() if k!='output_digests'}),flush=True)


try:
    for sequence,mode in enumerate(proof['arms']):
        with expert_routing_phase('decode'):
            run_arm(mode,sequence)
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
    controls = [a['warm_wall_ns'] for a in report['arms'] if a['mode']=='resident-110']
    candidates = [a['warm_wall_ns'] for a in report['arms'] if a['mode']=='stream-112']
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
print('DENSE_PREFETCH_COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ('construction','arms')}),flush=True)
