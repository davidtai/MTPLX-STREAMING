"""One-layer native route replay for a smaller fused gate/up output tile."""
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
    raise RuntimeError('parent GPU guard required before MLX import')
signal.alarm(180)
ROOT = Path(__file__).resolve().parent
proof = json.loads((ROOT/'installation.json').read_text())
LAYER = proof['layer']
if (ROOT/'probe.json').exists():
    raise RuntimeError('refusing prior evidence overwrite')
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
before = host_memory_snapshot()
if (not before['box']['ok']
        or before['box']['used_bytes'] + proof['static_incremental_bound_bytes'] > 110000000000
        or before['box']['wired_bytes'] + proof['static_incremental_bound_bytes'] > 100*1024**3):
    raise RuntimeError('complete one-layer bound does not fit')
inventory = (ROOT/'artifact/manifest.json').read_bytes()
if hashlib.sha256(inventory).hexdigest() != proof['artifact_manifest_sha256']:
    raise RuntimeError('packed inventory changed')
artifact = json.loads(inventory)
model = Path(proof['model_path'])
if hashlib.sha256((model/'expert-manifest.json').read_bytes()).hexdigest() != artifact['source_manifest_sha256']:
    raise RuntimeError('native source manifest changed')
st = (model/'experts.bin').stat()
if {k: getattr(st,k) for k in artifact['source_identity']} != artifact['source_identity']:
    raise RuntimeError('native expert source changed')

import numpy as np
import mlx.core as mx
from library_identity import identify
library=identify(proof['strict_allocator'])
from mtplx.expert_manifest import load_expert_manifest
from mtplx.expert_io import PositionalExpertReader
from mtplx.expert_slots import ExpertSlotPool
from mtplx.expert_runtime import ExpertStreamingRuntime, ExpertStreamingConfig
from mtplx.expert_streaming import RoutePlan, RoutingPhase, SlotLoad
from mtplx.expert_streaming_models import ExpertMemoryPlan, ExpertStreamingModelSpec
from mtplx.models.expert_mlx import make_mlx_component_bank_allocator, HotExpertSwitchGLU, expert_routing_phase
from packed_storage import load_layer, remove_raw_scales, bind_weight_reader, WEIGHT_BYTES
from restore_bank import restore
import plane_lane
from fused_ops import FusedPackedOps

mx.set_memory_limit(5*1024**3)
mx.set_cache_limit(256*1024**2)
spec = ExpertStreamingModelSpec(**{**proof['spec'], 'full_indexer_layers':(), 'island_pin_order':()})
plan = ExpertMemoryPlan(**{**proof['plan'], 'persistent_slots_by_layer':()})
config = ExpertStreamingConfig(**{**proof['config'], 'persistent_slots_by_layer':(), 'island_layers':(), 'mmap_island_layers':()})
full_manifest = load_expert_manifest(model/'expert-manifest.json')
manifest = dataclasses.replace(full_manifest, records=tuple(r for r in full_manifest.records if r.layer==LAYER),
    routed_expert_bytes=spec.routed_expert_bytes, artifact_tensor_bytes=spec.total_tensor_bytes,
    resident_tensor_bytes=4096, resident_tensors=())
route_payload = json.loads((ROOT/'routes.json').read_text())
assert route_payload['trace_sha256'] == proof['trace_sha256']
routes = route_payload['routes']
assert len(routes)==206 and all(len(ids)==36 for ids in routes)
report = {'source_commit':proof['source_commit'], 'scope':proof['scope'], 'construction':proof,
          'before':before, 'library':library, 'arms':[], 'complete':False}
references = {}
scales = None


def run_arm(mode, sequence):
    cache_bytes = 256 * 1024**2
    part_size = 3
    mx.synchronize()
    mx.clear_cache()
    mx.set_cache_limit(cache_bytes)
    reader = allocator = slots = runtime = switch = runners = None
    x = shared_input = indices = y = shared = None
    result = {'mode':mode, 'part_size':part_size, 'cache_limit_bytes':cache_bytes, 'sequence':sequence, 'cases':[]}
    try:
        reader = PositionalExpertReader(model,bypass_page_cache=True,use_native=False,io_read_fanout=4)
        allocator = make_mlx_component_bank_allocator(plan,spec,manifest)
        slots = ExpertSlotPool(spec,plan,manifest,reader,buffer_allocator=allocator,
            max_inflight_io_bytes=plan.transient_bytes,verify_hashes=False,
            device_synchronize=mx.synchronize,cache_scope='layer',resource_telemetry=False,batch_decode_reads=True)
        if sum(bank.capacity for bank in allocator.banks.values()) != proof['bank_capacity']:
            raise RuntimeError('bank allocation differs from construction')
        for bank in allocator.banks.values():
            remove_raw_scales(bank,mx=mx)
        for slot in allocator.slots.values():
            slot.nbytes = WEIGHT_BYTES
        arm_config = dataclasses.replace(config,decode_miss_records_per_part=part_size)
        runtime = ExpertStreamingRuntime(model,spec,arm_config,manifest,plan,reader,slots,single_slot_pool=True)
        bind_weight_reader(reader)
        # Restore only captured persistent residents. Each physical slot is
        # loaded before publishing the matching policy snapshot; no pretend hits.
        state = json.loads(json.dumps(route_payload['initial_bank']))
        entries = sorted(((int(e),s) for e,s in state['_expert_to_slot'].items()),key=lambda p:p[1])
        for start in range(0,len(entries),24):
            group = entries[start:start+24]
            ids = tuple(e for e,s in group)
            seed = RoutePlan(phase=RoutingPhase.PREFILL,experts=ids,slots=tuple(s for e,s in group),
                hits=(),misses=ids,loads=tuple(SlotLoad(e,s,True) for e,s in group),evictions=())
            ready = slots.ensure_route(LAYER,seed)
            ready.release(synchronize=False)
        restored = restore(state,policy='transition-window',single_pool=True)
        extra = plan.persistent_slots-restored.persistent_slots
        restored._slot_to_expert.extend([None]*extra)
        restored.persistent_slots=restored._persistent_capacity=plan.persistent_slots
        restored.slot_count+=extra
        restored._protected_cap=max(1,int(plan.persistent_slots*.8))
        runtime._banks[LAYER]=restored
        switch = HotExpertSwitchGLU(runtime,LAYER)
        runners = plane_lane.install(runtime,{LAYER:switch},{LAYER:scales},early=True)
        if mode == 'fused-r2':
            runners[LAYER].ops = FusedPackedOps(scales)
        rng = np.random.default_rng(20260918)
        x=mx.array(rng.normal(0,.15,(1,6,5120)).astype(np.float32)).astype(mx.bfloat16)
        shared_input=mx.array(rng.normal(0,.15,(1,6,5120)).astype(np.float32)).astype(mx.bfloat16)
        mx.eval(x,shared_input)
        for call,ids in enumerate(routes):
            indices=mx.array(ids,mx.int32).reshape(1,6,6)
            mx.eval(indices)
            previous=reader.metrics.as_dict()
            start=time.perf_counter_ns()
            y,shared=switch._run(x,indices,shared_work=lambda:mx.tanh(shared_input))
            mx.eval(y,shared)
            elapsed=time.perf_counter_ns()-start
            current=reader.metrics.as_dict()
            digest=hashlib.sha256(np.array(y.view(mx.uint16)).tobytes()).hexdigest()
            row={'call':call,'elapsed_ns':elapsed,'unique_experts':len(set(ids)),
                 'records_read':current['records_read']-previous['records_read'],
                 'read_bytes':current['read_bytes']-previous['read_bytes'],'output_sha256':digest,
                 'cache_bytes_after_eval':int(mx.get_cache_memory())}
            if cache_bytes==0 and row['cache_bytes_after_eval']!=0:
                raise RuntimeError('disabled allocator cache retained buffers')
            identity=(row['records_read'],row['read_bytes'],digest)
            if sequence==0:
                references[call]=identity
            if identity != references[call]:
                raise RuntimeError(f'part{part_size} changes output or physical reads at call{call}')
            result['cases'].append(row)
        runtime.flush_deferred_slot_releases(evaluate=True)
        result['reader_metrics']=reader.metrics.as_dict()
        result['slot_metrics']=slots.metrics.as_dict()
        result['mlx_peak_bytes']=int(mx.get_peak_memory())
        measured=result['cases'][4:]
        result['warm_total_ns']=sum(c['elapsed_ns'] for c in measured)
        result['warm_median_ns']=statistics.median(c['elapsed_ns'] for c in measured)
        result['warm_read_records']=sum(c['records_read'] for c in measured)
        result['all_outputs_and_reads_exact']=True
    finally:
        mx.synchronize()
        if runtime is not None:
            runtime.close()
        elif slots is not None:
            slots.close()
        if reader is not None:
            reader.close()
        if allocator is not None:
            allocator.close()
        x=shared_input=indices=y=shared=None
        runners=switch=runtime=slots=allocator=reader=None
        gc.collect()
        mx.clear_cache()
        result['active_after_close_bytes']=int(mx.get_active_memory())
        report['arms'].append(result)
        (ROOT/'probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FUSED_GU_R2_ARM',json.dumps({k:v for k,v in result.items() if k not in ('cases','reader_metrics','slot_metrics')}),flush=True)


try:
    scales=load_layer(ROOT/'artifact',artifact['layers'][LAYER],mx=mx)
    report['packed_scale_bytes']=sum(a.nbytes for parts in scales.values() for a in parts)
    for sequence,mode in enumerate(proof['arms']):
        with expert_routing_phase('decode'):
            run_arm(mode,sequence)
    controls=[a['warm_total_ns'] for a in report['arms'] if a['mode']=='native']
    candidates=[a['warm_total_ns'] for a in report['arms'] if a['mode']=='fused-r2']
    report['control_median_warm_total_ns']=statistics.median(controls)
    report['candidate_median_warm_total_ns']=statistics.median(candidates)
    report['latency_ratio']=statistics.median(candidates)/statistics.median(controls)
    report['control_spread_fraction']=(max(controls)-min(controls))/statistics.median(controls)
    report['complete']=True
    report['after']=host_memory_snapshot()
finally:
    if scales is not None:
        scales.clear()
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    report['active_after_close_bytes']=int(mx.get_active_memory())
    (ROOT/'probe.json').write_text(json.dumps(report,indent=2)+'\n')
if report['active_after_close_bytes']>2*1024**2:
    raise RuntimeError('one-layer replay retains Metal owners after close')
print('ALLOCATOR_CACHE_COMPLETE',json.dumps({k:report[k] for k in ('complete','latency_ratio','control_spread_fraction','active_after_close_bytes')}),flush=True)
