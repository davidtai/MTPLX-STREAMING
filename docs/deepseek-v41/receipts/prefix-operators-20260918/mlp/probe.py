"""Real packed MLP partition cost; no inter-layer scheduling or TPS inference."""
import hashlib,json,os,signal,statistics,subprocess,time
from pathlib import Path
from types import SimpleNamespace

if os.environ.get('_GPU_WINDOW_LOCKED')!='1':raise RuntimeError('parent GPU guard required before MLX')
signal.alarm(240)
ROOT=Path(__file__).resolve().parent
proof=json.loads((ROOT/'installation.json').read_text())
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
before=host_memory_snapshot()
if (not before['box']['ok'] or before['box']['used_bytes']+proof['static_incremental_bound_bytes']>110000000000
    or before['box']['wired_bytes']+proof['static_incremental_bound_bytes']>100*1024**3):
 raise RuntimeError('complete bounded operator footprint does not fit')
if (ROOT/'probe.json').exists():raise RuntimeError('refusing prior evidence overwrite')
if hashlib.sha256((ROOT/'artifact/manifest.json').read_bytes()).hexdigest()!=proof['artifact_manifest_sha256']:
 raise RuntimeError('packed scale inventory changed')
import numpy as np
import mlx.core as mx
from mtplx.expert_manifest import load_expert_manifest
from mtplx.expert_io import PositionalExpertReader
from mtplx.models.expert_mlx import MlxComponentBank,MlxComponentSlot,_run_component_bank_q4
from packed_storage import load_layer,remove_raw_scales,make_dispatch,WEIGHT_BYTES
mx.set_memory_limit(proof['allocator_limit_bytes']);mx.set_cache_limit(proof['allocator_cache_limit_bytes'])
model=Path(proof['model_path']);inventory=json.loads((ROOT/'artifact/manifest.json').read_text())
if hashlib.sha256((model/'expert-manifest.json').read_bytes()).hexdigest()!=inventory['source_manifest_sha256']:
 raise RuntimeError('source manifest changed')
st=(model/'experts.bin').stat()
if {k:getattr(st,k) for k in inventory['source_identity']}!=inventory['source_identity']:
 raise RuntimeError('native expert file identity changed')
manifest=load_expert_manifest(model/'expert-manifest.json')
records={v.expert:v for v in manifest.records if v.layer==20}
report={'source_commit':proof['source_commit'],'scope':proof['scope'],'construction':proof,'before':before,'cases':[],'complete':False}
bank=reader=scales=None

def run():
 global bank,reader,scales
 reader=PositionalExpertReader(model,bypass_page_cache=True,use_native=False,io_read_fanout=4)
 bank=MlxComponentBank(capacity=48,record=records[0],label='prefix-mlp-proof')
 experts=proof['selected_experts']
 slots={e:MlxComponentSlot(bank,(i*17+11)%48,label=f'prefix-{e}') for i,e in enumerate(experts)}
 for expert,slot in slots.items():reader.read_record_into(manifest,records[expert],slot,verify_hash=True)
 report['source_record_sha256']={str(e):records[e].sha256 for e in experts}
 rng=np.random.default_rng(20260918);cases=[]
 for case in proof['cases']:
  route=tuple(case['route']);assert len(route)==36
  # Repeat each token once per routed expert, exactly as PackedOps.gate_up.
  tokens=mx.array(rng.standard_normal((6,5120)).astype(np.float32)).astype(mx.bfloat16)
  x=mx.repeat(tokens,6,axis=0);mx.eval(x)
  bindings=tuple(SimpleNamespace(buffer=slots[e],expert=e) for e in route)
  y=_run_component_bank_q4(x,bindings,group_size=32,bits=4,swiglu_limit=10.0,codec='mxfp4');mx.eval(y)
  reference=np.array(y.view(mx.uint16))
  cases.append((case,x,bindings,reference))
 del y,tokens
 mx.synchronize();released=remove_raw_scales(bank,mx=mx)
 assert released==48*1105920
 for slot in slots.values():slot.nbytes=WEIGHT_BYTES
 scales=load_layer(ROOT/'artifact',inventory['layers'][20],mx=mx)
 dispatch=make_dispatch(scales,mx=mx)
 report['released_raw_scale_bytes']=released
 report['resident_packed_scale_bytes']=inventory['layers'][20]['packed_bytes']
 for info,x,bindings,reference in cases:
  arms={}
  for name,widths in [('full6',(6,)),('prefix1_tail5',(1,5)),('prefix3_tail3',(3,3))]:
   start=0;parts=[]
   for width in widths:
    end=start+width*6
    part_x=x[start:end];mx.eval(part_x)
    parts.append((part_x,bindings[start:end]));start=end
   assert start==36
   arms[name]=tuple(parts)
  def execute(name):
   values=[];spans=[];start=time.perf_counter_ns()
   for part_x,part_bindings in arms[name]:
    t=time.perf_counter_ns();y=dispatch(part_x,part_bindings);mx.eval(y)
    spans.append(time.perf_counter_ns()-t);values.append(y)
   end=time.perf_counter_ns()
   return tuple(values),spans,end-start
  for name in arms:
   for _ in range(3):execute(name)
  result={'cycle':info['cycle'],'route':info['route'],'unique_experts':info['unique_experts'],'timings':[],'parity':{}}
  # Interleaved control/candidates, short enough to avoid thermal drift.
  for name in ('full6','prefix1_tail5','full6','prefix3_tail3','full6','prefix3_tail3','full6','prefix1_tail5','full6'):
   execute(name);totals=[];prefix=[];segments=[]
   for _ in range(11):
    values,spans,total=execute(name);totals.append(total);prefix.append(spans[0]);segments.append(spans)
   result['timings'].append({'arm':name,'median_total_ns':statistics.median(totals),'median_first_part_ns':statistics.median(prefix),'samples_total_ns':totals,'samples_parts_ns':segments})
  # One exact-byte check per measured partition, outside all timed calls.
  for name in arms:
   values,_,_=execute(name);bits=np.concatenate([np.array(y.view(mx.uint16)) for y in values],axis=0)
   equal=bool(np.array_equal(bits,reference))
   result['parity'][name]={'exact_native_bytes':equal,'differing_elements':int(np.count_nonzero(bits!=reference)),'sha256':hashlib.sha256(bits.tobytes()).hexdigest()}
   if not equal:raise RuntimeError('row partition changes routed MLP arithmetic')
  medians={name:statistics.median(v['median_total_ns'] for v in result['timings'] if v['arm']==name) for name in arms}
  first={name:statistics.median(v['median_first_part_ns'] for v in result['timings'] if v['arm']==name) for name in arms}
  result['arm_median_ns']=medians;result['first_part_median_ns']=first
  result['total_cost_ratio']={k:v/medians['full6'] for k,v in medians.items()}
  result['prefix_available_earlier_ns']={k:medians['full6']-first[k] for k in arms if k!='full6'}
  ctrl=[v['median_total_ns'] for v in result['timings'] if v['arm']=='full6']
  result['control_spread_fraction']=(max(ctrl)-min(ctrl))/statistics.median(ctrl)
  report['cases'].append(result)
  (ROOT/'probe.json').write_text(json.dumps(report,indent=2)+'\n')
  print('PREFIX_MLP_CASE',json.dumps({k:v for k,v in result.items() if k not in ('timings','route')}),flush=True)
 report['mlx_peak_bytes']=int(mx.get_peak_memory());report['after']=host_memory_snapshot();report['complete']=True

try:run()
finally:
 if reader is not None:reader.close()
 if scales is not None:scales.clear()
 if bank is not None:bank.close()
 mx.synchronize();mx.clear_cache()
 report['active_after_close_bytes']=int(mx.get_active_memory())
 (ROOT/'probe.json').write_text(json.dumps(report,indent=2)+'\n')
if report['active_after_close_bytes']>2*1024**2:raise RuntimeError('operator allocations remain after close')
print('PREFIX_MLP_COMPLETE',json.dumps({k:report[k] for k in ('complete','mlx_peak_bytes','active_after_close_bytes')}),flush=True)
