"""Bounded demand-read / gate-up / down-plane overlap screen."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
from types import SimpleNamespace

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent GPU guard required before MLX import')
signal.alarm(180)
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'integration-probe.json'
if OUT.exists():
    raise RuntimeError('refusing to overwrite measured evidence')
proof = json.loads((ROOT / 'integration-construction.json').read_text())
before = host_memory_snapshot()
if (not before['box']['ok']
        or before['box']['used_bytes'] + proof['static_incremental_bound_bytes'] > 110000000000
        or before['box']['wired_bytes'] + proof['static_incremental_bound_bytes'] > 100 * 1024**3):
    raise RuntimeError('bounded probe does not fit current physical/wired memory')
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if source != proof['source_commit']:
    raise RuntimeError('source changed after construction accounting')
for name, digest in proof['helper_sha256'].items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('unchanged helper differs: ' + name)
inventory_bytes = (ROOT / 'artifact/manifest.json').read_bytes()
if hashlib.sha256(inventory_bytes).hexdigest() != proof['artifact_manifest_sha256']:
    raise RuntimeError('packed inventory changed')
artifact = json.loads(inventory_bytes)
model = Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
manifest_path = model / 'expert-manifest.json'
if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != artifact['source_manifest_sha256']:
    raise RuntimeError('source manifest changed')
st = (model / 'experts.bin').stat()
if {k: getattr(st, k) for k in artifact['source_identity']} != artifact['source_identity']:
    raise RuntimeError('weight source identity changed')

import dataclasses
import gc
import numpy as np
import mlx.core as mx
from mtplx.expert_manifest import load_expert_manifest
from mtplx.expert_io import PositionalExpertReader
from mtplx.expert_slots import ExpertSlotPool
from mtplx.expert_runtime import ExpertStreamingRuntime, ExpertStreamingConfig
from mtplx.expert_streaming_models import ExpertMemoryPlan, ExpertStreamingModelSpec
from mtplx.models.expert_mlx import make_mlx_component_bank_allocator, HotExpertSwitchGLU, expert_routing_phase
from packed_storage import load_layer, remove_raw_scales, make_dispatch, bind_weight_reader, WEIGHT_BYTES
import plane_lane
import native_plane_lane

if hashlib.sha256((ROOT/'plane_lane.py').read_bytes()).hexdigest() != proof['lane_sha256']:
    raise RuntimeError('lane differs from construction proof')
mx.set_memory_limit(2*1024**3)
mx.set_cache_limit(256*1024**2)
spec = ExpertStreamingModelSpec(**{**proof['spec'], 'full_indexer_layers':(), 'island_pin_order':()})
plan = ExpertMemoryPlan(**{**proof['plan'], 'persistent_slots_by_layer':()})
config = ExpertStreamingConfig(**{**proof['config'], 'persistent_slots_by_layer':(), 'island_layers':(), 'mmap_island_layers':()})
full_manifest = load_expert_manifest(manifest_path)
manifest = dataclasses.replace(full_manifest, records=tuple(r for r in full_manifest.records if r.layer==20),
    routed_expert_bytes=spec.routed_expert_bytes, artifact_tensor_bytes=spec.total_tensor_bytes,
    resident_tensor_bytes=4096, resident_tensors=())
report = dict(source_commit=source, construction=proof, before=before,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), arms=[],
    scope='One-layer real Runtime/SlotPool/HotExpertSwitchGLU integration, native weights/packed scales, synthetic BF16 inputs. Not full-model throughput.')
scales = None
references = {}


def arm(mode, sequence):
    reader = allocator = slots = runtime = switch = runners = None
    result = dict(mode=mode, sequence=sequence, cases=[])
    try:
        reader = PositionalExpertReader(model, bypass_page_cache=True, use_native=False, io_read_fanout=4)
        allocator = make_mlx_component_bank_allocator(plan,spec,manifest)
        slots = ExpertSlotPool(spec,plan,manifest,reader,buffer_allocator=allocator,
            max_inflight_io_bytes=plan.transient_bytes,verify_hashes=False,
            device_synchronize=mx.synchronize,cache_scope='layer',resource_telemetry=False,batch_decode_reads=True)
        if sum(b.capacity for b in allocator.banks.values()) != 80:
            raise RuntimeError('allocation inventory differs from construction')
        for bank in allocator.banks.values():
            remove_raw_scales(bank,mx=mx)
        for slot in allocator.slots.values():
            slot.nbytes=WEIGHT_BYTES
        runtime=ExpertStreamingRuntime(model,spec,config,manifest,plan,reader,slots,single_slot_pool=True)
        switch=HotExpertSwitchGLU(runtime,20)
        lane={'native':native_plane_lane,'paired':plane_lane}[mode]
        runners=lane.install(runtime,{20:switch},{20:scales},early=True)
        rng=np.random.default_rng(202641)
        permutation=rng.permutation(384)
        # Adjacent calls cover reuse; stride4 yields26 experts for36 M6
        # assignments, near the saved trace's24.24-expert mean. Different
        # groups force real demand reads; paired work is confined to hits.
        for rows in (1,6):
            x=mx.array(rng.standard_normal((1,rows,5120)).astype(np.float32)).astype(mx.bfloat16)
            shared_input=mx.array(rng.standard_normal((1,rows,5120)).astype(np.float32)).astype(mx.bfloat16)
            mx.eval(x,shared_input)
            for call in range(8):
                start_expert=40*(call//2)
                ids=np.array([[permutation[(start_expert+4*r+k)%384] for k in range(6)] for r in range(rows)],dtype=np.int32)
                indices=mx.array(ids[None])
                mx.eval(indices)
                before_metrics=reader.metrics.as_dict()
                t0=time.perf_counter_ns()
                y,shared=switch._run(x,indices,shared_work=lambda:mx.tanh(shared_input))
                mx.eval(y,shared)
                elapsed=(time.perf_counter_ns()-t0)/1e9
                after_metrics=reader.metrics.as_dict()
                bits=np.array(y.view(mx.uint16))
                key=(rows,call)
                if sequence==0:
                    references[key]=bits.copy()
                exact=bool(np.array_equal(bits,references[key]))
                if not exact:
                    raise RuntimeError(f'{mode} differs at M{rows} call{call}: {np.count_nonzero(bits!=references[key])} elements')
                result['cases'].append(dict(rows=rows,call=call,elapsed_s=elapsed,exact_bytes=exact,
                    unique_experts=int(np.unique(ids).size),
                    records_read=after_metrics['records_read']-before_metrics['records_read'],
                    read_bytes=after_metrics['read_bytes']-before_metrics['read_bytes'],
                    output_sha256=hashlib.sha256(bits.tobytes()).hexdigest()))
            del x,shared_input,indices,y,shared,bits
        runtime.flush_deferred_slot_releases(evaluate=True)
        result['reader_metrics']=reader.metrics.as_dict()
        result['slot_metrics']=slots.metrics.as_dict()
        result['mlx_peak_bytes']=int(mx.get_peak_memory())
        result['median_by_rows_s']={str(rows):statistics.median(c['elapsed_s'] for c in result['cases'] if c['rows']==rows and c['call']>=2)
                                    for rows in (1,6)}
        result['miss_median_by_rows_s']={str(rows):statistics.median(c['elapsed_s'] for c in result['cases'] if c['rows']==rows and c['call']>=2 and c['records_read']>0)
                                    for rows in (1,6)}
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
        runners=switch=runtime=slots=allocator=reader=None
        gc.collect()
        mx.clear_cache()
        result['active_after_close_bytes']=int(mx.get_active_memory())
        report['arms'].append(result)
        OUT.write_text(json.dumps(report,indent=2)+'\n')
    print('INTEGRATION_ARM',json.dumps({k:v for k,v in result.items() if k not in ('cases','reader_metrics','slot_metrics')}),flush=True)


try:
    scales=load_layer(ROOT/'artifact',artifact['layers'][20],mx=mx)
    report['packed_scale_bytes']=sum(a.nbytes for parts in scales.values() for a in parts)
    for sequence,mode in enumerate(('native','paired','paired','native')):
        with expert_routing_phase('decode'):
            arm(mode,sequence)
    report['complete']=True
    report['after']=host_memory_snapshot()
finally:
    if scales is not None:
        scales.clear()
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    report['active_after_close_bytes']=int(mx.get_active_memory())
    module_spec=importlib.util.spec_from_file_location('owned_cache','scripts/deepseek_v41/reclaim_file_cache.py')
    reclaim=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(reclaim)
    paths=[model/'experts.bin']+[ROOT/'artifact'/c[f]['file'] for c in artifact['layers'][20]['components'].values()
        for f in ('descriptors','payload','bases')]
    reclaimed=[reclaim.reclaim_file(p) for p in paths]
    report['cleanup']=dict(files=len(reclaimed),cached_page_bytes_before=sum(x['cached_page_bytes_before'] for x in reclaimed),
        cached_page_bytes_after=sum(x['cached_page_bytes_after'] for x in reclaimed))
    OUT.write_text(json.dumps(report,indent=2)+'\n')
if report['active_after_close_bytes']>2*1024**2:
    raise RuntimeError('integration retains Metal owners after close')
print('INTEGRATION_COMPLETE',json.dumps({k:report[k] for k in ('active_after_close_bytes','cleanup')}),flush=True)
