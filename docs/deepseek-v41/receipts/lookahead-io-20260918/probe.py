"""Optimistic paired-layer I/O screen; no full model or live router timing."""
import dataclasses,gc,hashlib,inspect,json,os,signal,statistics,textwrap,types
from pathlib import Path
import time

if os.environ.get('_GPU_WINDOW_LOCKED')!='1':
    raise RuntimeError('parent GPU guard required before MLX import')
signal.alarm(180)
ROOT=Path(__file__).resolve().parent
proof=json.loads((ROOT/'installation.json').read_text())
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
before=host_memory_snapshot()
if (not before['box']['ok'] or before['box']['used_bytes']+proof['static_incremental_bound_bytes']>110000000000
        or before['box']['wired_bytes']+proof['static_incremental_bound_bytes']>100*1024**3):
    raise RuntimeError('complete paired-layer memory bound does not fit')
if (ROOT/'probe.json').exists():raise RuntimeError('refusing prior evidence overwrite')
inventory=(ROOT/'artifact/manifest.json').read_bytes()
assert hashlib.sha256(inventory).hexdigest()==proof['artifact_manifest_sha256']
artifact=json.loads(inventory)
model=Path(proof['model_path'])
assert hashlib.sha256((model/'expert-manifest.json').read_bytes()).hexdigest()==artifact['source_manifest_sha256']
st=(model/'experts.bin').stat()
assert {k:getattr(st,k) for k in artifact['source_identity']}==artifact['source_identity']

import numpy as np
import mlx.core as mx
from library_identity import identify
library=identify(proof['strict_allocator'])
from mtplx.expert_manifest import load_expert_manifest
from mtplx.expert_io import PositionalExpertReader
from mtplx.expert_slots import ExpertSlotPool,ExpertSlotState
from mtplx.expert_runtime import ExpertStreamingRuntime,ExpertStreamingConfig
from mtplx import expert_runtime as er
from mtplx.expert_streaming import RoutePlan,RoutingPhase,SlotLoad
from mtplx.expert_streaming_models import ExpertMemoryPlan,ExpertStreamingModelSpec
from mtplx.models.expert_mlx import make_mlx_component_bank_allocator,HotExpertSwitchGLU,expert_routing_phase
from packed_storage import load_layer,remove_raw_scales,bind_weight_reader,WEIGHT_BYTES
from restore_bank import restore
from paired_config import PairedPrefetchConfig
import plane_lane

mx.set_memory_limit(7*1024**3)
mx.set_cache_limit(256*1024**2)
spec=ExpertStreamingModelSpec(**{**proof['spec'],'full_indexer_layers':(),'island_pin_order':()})
plan=ExpertMemoryPlan(**{**proof['plan'],'persistent_slots_by_layer':()})
config=PairedPrefetchConfig(**{**proof['config'],'persistent_slots_by_layer':(),'island_layers':(),'mmap_island_layers':()})
LAYERS=tuple(proof['layers'])
assert LAYERS==(30,31) and spec.routed_layer_indices==LAYERS
assert plan.slots_per_layer==105 and plan.prefetch_ring_slots==16
full_manifest=load_expert_manifest(model/'expert-manifest.json')
manifest=dataclasses.replace(full_manifest,records=tuple(r for r in full_manifest.records if r.layer in LAYERS),
    routed_expert_bytes=spec.routed_expert_bytes,artifact_tensor_bytes=spec.total_tensor_bytes,
    resident_tensor_bytes=4096,resident_tensors=())
data=json.loads((ROOT/'routes.json').read_text())

# The old "recent misses" hint belongs to a prior visit of the target layer;
# other layers can already have replaced its shared transient owners. Filter
# actual READY owners at this explicit issue boundary instead. All native ring
# tickets, completion publication, reconciliation and cleanup remain unchanged.
source=textwrap.dedent(inspect.getsource(ExpertStreamingRuntime.prefetch_experts))
old='''        recent_misses = self._recent_route_misses.get(layer)
        if recent_misses:
            expert_ids = [
                expert
                for expert in expert_ids
                if expert not in recent_misses
            ]
'''
assert source.count(old)==1
namespace=dict(er.__dict__)
exec(compile(source.replace(old,''),'physical_prefetch_experts','exec'),namespace)
prefetch_function=namespace['prefetch_experts']

report={'complete':False,'scope':proof['scope'],'source_commit':proof['source_commit'],
    'construction':proof,'before':before,'library':library,'arms':[]}
scales={}
references={}


class Issue:
    def __init__(self,runtime):
        self.runtime=runtime
        self.ranked=()
        self.slots=tuple(runtime.slots._persistent.values())+runtime.slots._transient+tuple(runtime.slots._prefetch.values())

    def __call__(self):
        ready={s.expert for s in self.slots if s.state is ExpertSlotState.READY and s.layer==31}
        ids=[e for e in self.ranked if e not in ready][:8]
        self.runtime.prefetch_experts(31,ids,verify=True)


def run_arm(mode,sequence):
    mx.synchronize();mx.clear_cache()
    reader=allocator=slots=runtime=switches=runners=issue=None
    x=shared_input=indices=y0=y1=s0=s1=None
    result={'mode':mode,'sequence':sequence,'cases':[]}
    try:
        reader=PositionalExpertReader(model,bypass_page_cache=True,use_native=False,io_read_fanout=4)
        allocator=make_mlx_component_bank_allocator(plan,spec,manifest)
        slots=ExpertSlotPool(spec,plan,manifest,reader,buffer_allocator=allocator,
            max_inflight_io_bytes=plan.transient_bytes,verify_hashes=False,
            device_synchronize=mx.synchronize,cache_scope='layer',resource_telemetry=False,batch_decode_reads=True)
        assert sum(b.capacity for b in allocator.banks.values())==proof['bank_capacity']
        for b in allocator.banks.values():remove_raw_scales(b,mx=mx)
        for slot in allocator.slots.values():slot.nbytes=WEIGHT_BYTES
        runtime=ExpertStreamingRuntime(model,spec,config,manifest,plan,reader,slots,single_slot_pool=True)
        runtime.prefetch_experts=types.MethodType(prefetch_function,runtime)
        runtime._per_layer_record_bytes={l:WEIGHT_BYTES for l in LAYERS}
        runtime._representative_record_bytes=WEIGHT_BYTES
        bind_weight_reader(reader)
        for layer in LAYERS:
            state=json.loads(json.dumps(data['initial_banks'][str(layer)]))
            entries=sorted(((int(e),s) for e,s in state['_expert_to_slot'].items()),key=lambda p:p[1])
            for start in range(0,len(entries),24):
                group=entries[start:start+24];ids=tuple(e for e,s in group)
                seed=RoutePlan(phase=RoutingPhase.PREFILL,experts=ids,slots=tuple(s for e,s in group),
                    hits=(),misses=ids,loads=tuple(SlotLoad(e,s,True) for e,s in group),evictions=())
                ready=slots.ensure_route(layer,seed);ready.release(synchronize=False)
            extra=plan.slots_per_layer-state['persistent_slots']
            state['_slot_to_expert'].extend([None]*extra)
            state['persistent_slots']=plan.slots_per_layer
            restored=restore(state,policy='transition-window',single_pool=True,layer_id=layer,
                prefetch_ring=runtime._prefetch_ring,prefetch_slots=16)
            runtime._banks[layer]=restored
        switches={l:HotExpertSwitchGLU(runtime,l) for l in LAYERS}
        issue=Issue(runtime)
        runners=plane_lane.install(runtime,switches,scales,early=True,
            prefetch_source=(30,issue) if mode=='prefetch' else None)
        rng=np.random.default_rng(20260918)
        x=mx.array(rng.normal(0,.15,(1,6,5120)).astype(np.float32)).astype(mx.bfloat16)
        shared_input=mx.array(rng.normal(0,.15,(1,6,5120)).astype(np.float32)).astype(mx.bfloat16)
        mx.eval(x,shared_input)
        for call in range(64):
            indices=[mx.array(data['routes'][str(l)][call],mx.int32).reshape(1,6,6) for l in LAYERS]
            mx.eval(indices)
            issue.ranked=data['predictions'][call]
            previous=reader.metrics.as_dict()
            start=time.perf_counter_ns()
            y0,s0=switches[30]._run(x,indices[0],shared_work=lambda:mx.tanh(shared_input))
            # Require current expert output completion before the target route;
            # omit attention lead time and live predictor compute explicitly.
            mx.eval(y0,s0)
            y1,s1=switches[31]._run(x,indices[1],shared_work=lambda:mx.tanh(shared_input))
            mx.eval(y1,s1)
            elapsed=time.perf_counter_ns()-start
            current=reader.metrics.as_dict()
            digest=hashlib.sha256(np.array(y0.view(mx.uint16)).tobytes()+np.array(y1.view(mx.uint16)).tobytes()).hexdigest()
            if sequence==0:references[call]=digest
            if digest!=references[call]:raise RuntimeError(f'prefetch changed native output at pair{call}')
            result['cases'].append({'call':call,'elapsed_ns':elapsed,'output_sha256':digest,
                'records_read':current['records_read']-previous['records_read'],
                'read_bytes':current['read_bytes']-previous['read_bytes']})
        runtime.flush_deferred_slot_releases(evaluate=True)
        runtime._drain_prefetch_loads()
        result['reader_metrics']=reader.metrics.as_dict()
        result['slot_metrics']=slots.metrics.as_dict()
        result['runtime_counters']=runtime.counters.as_dict()
        result['mlx_peak_bytes']=int(mx.get_peak_memory())
        result['heldout_total_ns']=sum(x['elapsed_ns'] for x in result['cases'][32:])
        result['heldout_read_records']=sum(x['records_read'] for x in result['cases'][32:])
        result['all_outputs_exact']=True
    finally:
        mx.synchronize()
        if runtime is not None:runtime.close()
        elif slots is not None:slots.close()
        if reader is not None:reader.close()
        if allocator is not None:allocator.close()
        x=shared_input=indices=y0=y1=s0=s1=None
        issue=runners=switches=runtime=slots=allocator=reader=ready=None
        gc.collect();mx.clear_cache()
        result['active_after_close_bytes']=int(mx.get_active_memory())
        report['arms'].append(result)
        (ROOT/'probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PAIR_ARM',json.dumps({k:v for k,v in result.items() if k not in ('cases','reader_metrics','slot_metrics','runtime_counters')}),flush=True)


try:
    scales={l:load_layer(ROOT/'artifact',artifact['layers'][l],mx=mx) for l in LAYERS}
    for sequence,mode in enumerate(proof['arms']):
        with expert_routing_phase('decode'):run_arm(mode,sequence)
    controls=[a['heldout_total_ns'] for a in report['arms'] if a['mode']=='native']
    candidates=[a['heldout_total_ns'] for a in report['arms'] if a['mode']=='prefetch']
    report['latency_ratio']=statistics.median(candidates)/statistics.median(controls)
    report['control_spread_fraction']=(max(controls)-min(controls))/statistics.median(controls)
    report['complete']=True
    report['after']=host_memory_snapshot()
finally:
    scales.clear();gc.collect();mx.synchronize();mx.clear_cache()
    report['active_after_close_bytes']=int(mx.get_active_memory())
    (ROOT/'probe.json').write_text(json.dumps(report,indent=2)+'\n')
if report['active_after_close_bytes']>2*1024**2:
    raise RuntimeError('paired replay retains Metal owners after close')
print('PAIR_COMPLETE',json.dumps({k:report[k] for k in ('complete','latency_ratio','control_spread_fraction','active_after_close_bytes')}),flush=True)
