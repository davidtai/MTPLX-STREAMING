"""Bounded native versus pair-of-rows expert MLP screen; no full-model claim."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent guard must own the GPU lock before MLX import')
signal.alarm(180)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'probe.json'
if OUT.exists():
    raise RuntimeError('refusing to overwrite evidence')
proof = json.loads((ROOT / 'construction.json').read_text())
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
before = host_memory_snapshot()
if (not before['box']['ok']
    or before['box']['used_bytes'] + proof['static_incremental_bound_bytes'] > 110000000000
    or before['box']['wired_bytes'] + proof['static_incremental_bound_bytes'] > 100*1024**3):
    raise RuntimeError('bounded operator cannot fit physical/wired headroom')
source = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
if source != proof['source_commit']:
    raise RuntimeError('candidate source changed')
for name, digest in proof['helper_sha256'].items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('operator helper changed: '+name)
blob=(ROOT/'artifact/manifest.json').read_bytes()
if hashlib.sha256(blob).hexdigest() != proof['artifact_manifest_sha256']:
    raise RuntimeError('packed inventory changed')
artifact=json.loads(blob)
model=Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
if hashlib.sha256((model/'expert-manifest.json').read_bytes()).hexdigest()!=artifact['source_manifest_sha256']:
    raise RuntimeError('source manifest changed')
st=(model/'experts.bin').stat()
if {k:getattr(st,k) for k in artifact['source_identity']}!=artifact['source_identity']:
    raise RuntimeError('weight source identity changed')

import numpy as np
import mlx.core as mx
from mtplx.expert_manifest import load_expert_manifest
from mtplx.expert_io import PositionalExpertReader
from mtplx.models.expert_mlx import MlxComponentBank, MlxComponentSlot, _clamped_swiglu
from packed_storage import load_layer, remove_raw_scales, WEIGHT_BYTES
from paired_kernels import make_projection as make_native
from row_pair_kernels import make_projection as make_pair

mx.set_memory_limit(2*1024**3)
mx.set_cache_limit(256*1024**2)
reader=bank=scales=None
report={'scope':'Three real layer20 experts; synthetic BF16 inputs; native and paired MLPs, including CPU grouping/gathers/restore order',
        'construction':proof,'before':before,'source_commit':source,'cases':[],'complete':False}


def run():
    global reader,bank,scales
    manifest=load_expert_manifest(model/'expert-manifest.json')
    records={r.expert:r for r in manifest.records if r.layer==20}
    reader=PositionalExpertReader(model,bypass_page_cache=True,use_native=False,io_read_fanout=4)
    bank=MlxComponentBank(capacity=3,record=records[0],label='row-pair-screen')
    experts=proof['experts']
    slots={e:MlxComponentSlot(bank,(i+1)%3,label=f'pair-{e}') for i,e in enumerate(experts)}
    for e in experts:reader.read_record_into(manifest,records[e],slots[e],verify_hash=True)
    report['record_sha256']={str(e):records[e].sha256 for e in experts}
    remove_raw_scales(bank,mx=mx)
    for slot in slots.values():slot.nbytes=WEIGHT_BYTES
    scales=load_layer(ROOT/'artifact',artifact['layers'][20],mx=mx)
    native_gu,native_down=make_native(2304,5120),make_native(5120,2304)
    pair_gu,pair_down=make_pair(2304,5120),make_pair(5120,2304)

    def mlp(x, ids, paired):
        rows=len(ids)
        pairs=mx.array([(slots[e].bank_index,e) for e in (ids[::2] if paired else ids)],mx.int32)
        group_rows=rows//2 if paired else rows
        gu,down=(pair_gu,pair_down) if paired else (native_gu,native_down)
        x=x.reshape(rows,1,1,5120)
        common=dict(template=[('T',mx.bfloat16)],grid=(32,576,group_rows),threadgroup=(32,2,1),
                    output_shapes=[(rows,1,1,2304)],output_dtypes=[mx.bfloat16])
        g=gu(inputs=[x,pairs,bank.arrays['gate_proj.weight'],*scales['gate_proj']],**common)[0]
        u=gu(inputs=[x,pairs,bank.arrays['up_proj.weight'],*scales['up_proj']],**common)[0]
        h=_clamped_swiglu(g,u,10.0)
        return down(inputs=[h,pairs,bank.arrays['down_proj.weight'],*scales['down_proj']],
                    template=[('T',mx.bfloat16)],grid=(32,1280,group_rows),threadgroup=(32,2,1),
                    output_shapes=[(rows,1,1,5120)],output_dtypes=[mx.bfloat16])[0].reshape(rows,5120)

    def candidate(x,ids):
        members={}
        for i,e in enumerate(ids):members.setdefault(e,[]).append(i)
        paired_positions=[];solo_positions=[]
        for positions in members.values():
            count=len(positions)//2*2
            paired_positions.extend(positions[:count]);solo_positions.extend(positions[count:])
        outputs=[];order=[]
        if paired_positions:
            px=mx.take(x,mx.array(paired_positions,mx.int32),axis=0)
            outputs.append(mlp(px,[ids[p] for p in paired_positions],True));order.extend(paired_positions)
        if solo_positions:
            sx=mx.take(x,mx.array(solo_positions,mx.int32),axis=0)
            outputs.append(mlp(sx,[ids[p] for p in solo_positions],False));order.extend(solo_positions)
        inverse=[0]*len(order)
        for i,p in enumerate(order):inverse[p]=i
        y=outputs[0] if len(outputs)==1 else mx.concatenate(outputs,axis=0)
        return mx.take(y,mx.array(inverse,mx.int32),axis=0)

    rng=np.random.default_rng(641)
    for counts in ((1,1,2),(2,2,2),(3,3,3),(6,6,6)):
        ids=[e for e,count in zip(experts,counts) for _ in range(count)]
        rng.shuffle(ids)
        x=mx.array(rng.standard_normal((len(ids),5120)).astype(np.float32)).astype(mx.bfloat16)
        mx.eval(x)
        a=mlp(x,ids,False);b=candidate(x,ids);mx.eval(a,b)
        a_bits=np.array(a.view(mx.uint16));b_bits=np.array(b.view(mx.uint16))
        equal=bool(np.array_equal(a_bits,b_bits))
        case={'rows':len(ids),'expert_multiplicities':counts,'exact_bytes':equal,'timings':[]}
        if not equal:
            case['differing_elements']=int(np.count_nonzero(a_bits!=b_bits))
            first=tuple(np.argwhere(a_bits!=b_bits)[0])
            case['first_difference']={'index':[int(i) for i in first],
                                      'native_bits':int(a_bits[first]),'candidate_bits':int(b_bits[first])}
            report['cases'].append(case)
            raise RuntimeError('pair-of-rows MLP differs from native bits')
        for order in (('native','paired'),('paired','native'),('native','paired')):
            for arm in order:
                samples=[]
                for _ in range(5):
                    start=time.perf_counter_ns()
                    y=mlp(x,ids,False) if arm=='native' else candidate(x,ids)
                    mx.eval(y);samples.append((time.perf_counter_ns()-start)/1e9)
                case['timings'].append({'arm':arm,'samples_s':samples,'median_s':statistics.median(samples)})
        case['medians_s']={arm:statistics.median(t['median_s'] for t in case['timings'] if t['arm']==arm)
                           for arm in ('native','paired')}
        case['latency_reduction_pct']=100*(1-case['medians_s']['paired']/case['medians_s']['native'])
        report['cases'].append(case)
        OUT.write_text(json.dumps(report,indent=2)+'\n')
        print('ROW_PAIR_CASE',json.dumps({k:v for k,v in case.items() if k!='timings'}),flush=True)
    report['mlx_allocator_peak_bytes']=int(mx.get_peak_memory())
    report['after']=host_memory_snapshot();report['complete']=True


try:
    run()
finally:
    if reader is not None:reader.close()
    if scales is not None:scales.clear()
    if bank is not None:bank.close()
    mx.synchronize();mx.clear_cache()
    report['active_after_close_bytes']=int(mx.get_active_memory())
    spec=importlib.util.spec_from_file_location('reclaimer','scripts/deepseek_v41/reclaim_file_cache.py')
    reclaim=importlib.util.module_from_spec(spec);spec.loader.exec_module(reclaim)
    paths=[model/'experts.bin']+[ROOT/'artifact'/c[f]['file'] for c in artifact['layers'][20]['components'].values()
                               for f in ('descriptors','payload','bases')]
    reclaimed=[reclaim.reclaim_file(p) for p in paths]
    report['cleanup']={key:sum(v[key] for v in reclaimed) for key in ('cached_page_bytes_before','cached_page_bytes_after')}
    OUT.write_text(json.dumps(report,indent=2)+'\n')
if report['active_after_close_bytes']>2*1024**2:raise RuntimeError('Metal owners remain after close')
print('ROW_PAIR_COMPLETE',report['mlx_allocator_peak_bytes'],report['active_after_close_bytes'],flush=True)
