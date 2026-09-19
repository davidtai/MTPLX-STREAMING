"""Bounded native attention partition cost and cache-state comparison."""
import gc,hashlib,json,os,signal,statistics,time
from pathlib import Path
if os.environ.get('_GPU_WINDOW_LOCKED')!='1':raise RuntimeError('parent GPU guard required before MLX')
signal.alarm(240)
ROOT=Path(__file__).resolve().parent;proof=json.loads((ROOT/'installation.json').read_text())
if (ROOT/'probe.json').exists():raise RuntimeError('refusing prior evidence overwrite')
for k,v in proof['arm_env'].items():os.environ[k]=v
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
before=host_memory_snapshot()
if (not before['box']['ok'] or before['box']['used_bytes']+proof['incremental_bound_bytes']>110000000000
    or before['box']['wired_bytes']+proof['incremental_bound_bytes']>100*1024**3):
 raise RuntimeError('complete attention bound does not fit')
model=Path(proof['model_path'])
for name,key in (('expert-manifest.json','model_manifest_sha256'),('config.json','config_sha256')):
 if hashlib.sha256((model/name).read_bytes()).hexdigest()!=proof[key]:raise RuntimeError('native artifact changed')
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mtplx.expert_manifest import load_expert_manifest
from mtplx.models import deepseek_v41 as dv
from mtplx.models.deepseek_v41_cache import LayerAttentionCache,SharedAttentionRuntime
from attention_reader import load_attention_tensors
from owned_projection import install_attention
mx.set_memory_limit(proof['allocator_limit_bytes']);mx.set_cache_limit(proof['allocator_cache_limit_bytes'])
args=dv.ModelArgs.from_dict(json.loads((model/'config.json').read_text()))
report={'source_commit':proof['source_commit'],'scope':proof['scope'],'construction':proof,'before':before,'cases':[],'complete':False}
P=proof['seed_offset'];N=proof['rows']

def compare(a,b):
 if a is None or b is None:return {'exact':a is None and b is None,'none':True}
 if a.shape!=b.shape:return {'exact':False,'shape_a':list(a.shape),'shape_b':list(b.shape)}
 aa_bits=np.array(a.view(mx.uint8));bb_bits=np.array(b.view(mx.uint8))
 exact=bool(a.dtype==b.dtype and np.array_equal(aa_bits,bb_bits))
 aa=np.array(a.astype(mx.float32));bb=np.array(b.astype(mx.float32))
 da=aa.astype(np.float64);db=bb.astype(np.float64)
 return {'exact':exact,'shape':list(aa.shape),'dtype':str(a.dtype),'different_values':int(np.count_nonzero(aa!=bb)),'max_abs':float(np.max(np.abs(da-db),initial=0)),'rms_error':float(np.sqrt(np.mean((da-db)**2))),'sha256_a':hashlib.sha256(aa_bits.tobytes()).hexdigest(),'sha256_b':hashlib.sha256(bb_bits.tobytes()).hexdigest()}

def load_attentions():
 manifest=load_expert_manifest(model/'expert-manifest.json')
 wanted=set(proof['resident_names']);kept=tuple(t for t in manifest.resident_tensors if t.tensor in wanted)
 raw=load_attention_tensors(model,manifest,kept,mx=mx);out={}
 for layer in sorted({l for c in proof['cases'] for l in c}):
  attn=dv.Attention(args,layer);prefix=f'layers.{layer}.attn.'
  weights={n.removeprefix(prefix):a for n,a in raw.items() if n.startswith(prefix)}
  for native,target in (('q_norm.weight','q_norm_weight'),('kv_norm.weight','kv_norm_weight'),('compressor.norm.weight','compressor.norm_weight'),('indexer.k_norm.weight','indexer.k_norm_weight')):
   if native in weights:weights[target]=weights.pop(native)
  for name in [n.removesuffix('.weight') for n in weights if n.endswith('.weight') and n.removesuffix('.weight')+'.scales' in weights]:
   w=weights[name+'.weight'];s=weights[name+'.scales']
   if w.dtype!=mx.uint32 or s.dtype!=mx.uint8:raise RuntimeError('native MXFP8 layout differs')
   parent=attn;parts=name.split('.')
   for part in parts[:-1]:parent=getattr(parent,part)
   original=getattr(parent,parts[-1]);shape=(w.shape[0],w.shape[1]*4)
   if type(original) is not nn.Linear or tuple(original.weight.shape)!=shape:raise RuntimeError('native linear geometry differs')
   setattr(parent,parts[-1],nn.QuantizedLinear(shape[1],shape[0],bias=False,group_size=32,bits=8,mode='mxfp8'))
  if set(dict(tree_flatten(attn.parameters())))!=set(weights):raise RuntimeError('native attention inventory differs')
  attn.load_weights(list(weights.items()),strict=True);mx.eval(attn.parameters())
  if any(a is not weights[n] for n,a in tree_flatten(attn.parameters())):raise RuntimeError('unreplaced attention parameters remain')
  install_attention(attn);out[layer]=attn
 return out

def run():
 attentions=load_attentions();rng=np.random.default_rng(20260918)
 for layers in proof['cases']:
  inputs={l:mx.array(rng.normal(0,.15,(1,N,5120)).astype(np.float32)) for l in layers}
  mx.eval(list(inputs.values()))
  seeds={};window_dtypes={}
  for l in layers:
   a=attentions[l];cos,sin=dv._cos_sin(a.inv_freq,mx.arange(P,P+N))
   q,qr,kv=a._qkv_prep_fused(inputs[l],cos,sin,1,N,64,512);mx.eval(q,qr,kv)
   window_dtypes[l]=kv.dtype
   ratio=a.compress_ratio;source=a.is_kv_source
   shapes=[(1,256,512),(1,P//ratio,512) if source else None,(1,P//ratio,128) if source else None,
           (1,P,512) if source and ratio>1 else None,(1,P,512) if source and ratio>1 else None]
   seeds[l]=tuple(None if shape is None else rng.normal(0,.1,shape).astype(np.float32) for shape in shapes)
  del q,qr,kv,cos,sin
  def fresh():
   caches={}
   for l in layers:
    a=attentions[l];c=LayerAttentionCache(args.window_size,a.compress_ratio,a.is_kv_source)
    values=tuple(None if x is None else mx.array(x,dtype=window_dtypes[l] if i==0 else mx.float32) for i,x in enumerate(seeds[l]))
    c.state=values;c.meta_state=(c.meta_state[0],str(P),str(args.window_size),str(a.compress_ratio),'1' if a.is_kv_source else '0')
    mx.eval(c.eval_backing());caches[l]=c
   return caches
  batches={}
  for mode,widths in (('full6',(6,)),('prefix1_tail5',(1,5))):
   start=0;chunks=[]
   for width in widths:
    xs={l:inputs[l][:,start:start+width] for l in layers};pos=mx.arange(P+start,P+start+width)
    mx.eval(list(xs.values()),pos);chunks.append((width,xs,pos));start+=width
   batches[mode]=tuple(chunks)
  def execute(mode,capture=False):
   caches=fresh();shared=SharedAttentionRuntime();outputs={l:[] for l in layers};selected={l:[] for l in layers};parts=[]
   start=time.perf_counter_ns()
   for width,xs,pos in batches[mode]:
    part=time.perf_counter_ns()
    for l in layers:
     y=attentions[l](xs[l],pos,caches[l],shared);mx.eval(y);outputs[l].append(y)
     if capture and shared.selected_idx is not None:
      idx=shared.selected_idx;mx.eval(idx);selected[l].append(np.array(idx))
    for cache in caches.values():cache.offset+=width
    parts.append(time.perf_counter_ns()-part)
   total=time.perf_counter_ns()-start
   if capture:
    mx.eval([x for c in caches.values() for x in c.state if x is not None])
   return outputs,caches,selected,parts,total
  for mode in batches:
   for _ in range(2):execute(mode)
  result={'layers':layers,'modes':[attentions[l].mode for l in layers],'seed_offset':P,'window_dtypes':{str(l):str(t) for l,t in window_dtypes.items()},'input_dtype':'float32','timings':[]}
  for mode in ('full6','prefix1_tail5','full6','prefix1_tail5','full6'):
   totals=[];first=[]
   for _ in range(7):
    _,_,_,parts,total=execute(mode);totals.append(total);first.append(parts[0])
   result['timings'].append({'arm':mode,'median_total_ns':statistics.median(totals),'median_first_part_ns':statistics.median(first),'samples_total_ns':totals,'samples_first_part_ns':first})
  native,nc,ns,_,_=execute('full6',True);candidate,cc,cs,_,_=execute('prefix1_tail5',True)
  parity={}
  for l in layers:
   ny=mx.concatenate(native[l],axis=1);cy=mx.concatenate(candidate[l],axis=1);mx.eval(ny,cy)
   out=compare(ny,cy);state=[compare(a,b) for a,b in zip(nc[l].state,cc[l].state)]
   sel_equal=(len(ns[l])==0 and len(cs[l])==0) or (bool(ns[l]) and bool(cs[l]) and np.array_equal(np.concatenate(ns[l],axis=1),np.concatenate(cs[l],axis=1)))
   parity[str(l)]={'output':out,'state':state,'offsets':[nc[l].offset,cc[l].offset],'meta_state_equal':nc[l].meta_state==cc[l].meta_state,'selected_indices_exact':bool(sel_equal)}
  result['parity']=parity
  meds={mode:statistics.median(v['median_total_ns'] for v in result['timings'] if v['arm']==mode) for mode in batches}
  first={mode:statistics.median(v['median_first_part_ns'] for v in result['timings'] if v['arm']==mode) for mode in batches}
  result['arm_median_ns']=meds;result['first_part_median_ns']=first;result['total_cost_ratio']=meds['prefix1_tail5']/meds['full6'];result['prefix_available_earlier_ns']=meds['full6']-first['prefix1_tail5']
  ctrl=[v['median_total_ns'] for v in result['timings'] if v['arm']=='full6'];result['control_spread_fraction']=(max(ctrl)-min(ctrl))/statistics.median(ctrl)
  report['cases'].append(result);(ROOT/'probe.json').write_text(json.dumps(report,indent=2)+'\n')
  print('PREFIX_ATTENTION_CASE',json.dumps({k:v for k,v in result.items() if k not in ('timings','parity')}),flush=True)
  del native,nc,ns,candidate,cc,cs,ny,cy,inputs,seeds,batches
  gc.collect();mx.synchronize();mx.clear_cache()
 report['mlx_peak_bytes']=int(mx.get_peak_memory());report['after']=host_memory_snapshot();report['complete']=True

try:run()
finally:
 gc.collect();mx.synchronize();mx.clear_cache();report['active_after_close_bytes']=int(mx.get_active_memory())
 (ROOT/'probe.json').write_text(json.dumps(report,indent=2)+'\n')
if report['active_after_close_bytes']>2*1024**2:raise RuntimeError('attention operator arrays remain after close')
print('PREFIX_ATTENTION_COMPLETE',json.dumps({k:report[k] for k in ('complete','mlx_peak_bytes','active_after_close_bytes')}),flush=True)
