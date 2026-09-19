import hashlib,json,os,signal,statistics,time
from pathlib import Path
if os.environ.get('_GPU_WINDOW_LOCKED') != '1': raise RuntimeError('exclusive guard required before MLX')
signal.alarm(120)
control=json.loads(Path('/tmp/dsv41-110-stage/full-cache-control-d405-20260917.jsonl').read_text())
for key,value in control['arm_env'].items():
    if value is None: os.environ.pop(key,None)
    else: os.environ[key]=str(value)
import mlx.core as mx
mx.set_memory_limit(512*1024**2);mx.set_cache_limit(64*1024**2)
from mtplx.models import deepseek_v41 as d
c=json.loads(Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4/config.json').read_text())['text_config']
H,C,M=c['hidden_size'],c['hc_mult'],6
consts=(C,c['hc_sinkhorn_iters'],c['rms_norm_eps'],c['hc_eps'])
mx.random.seed(617)
h=mx.random.normal((1,M,C,H)).astype(mx.float32)
pm=mx.concatenate([mx.ones((1,M,1)),mx.zeros((1,M,C-1))],axis=-1)
fn1=mx.random.normal((C*C+2*C,C*H)).astype(mx.float32)*0.01
fn2=mx.random.normal(fn1.shape).astype(mx.float32)*0.01
base=mx.zeros((C*C+2*C,));scale=mx.ones((3,));norm=mx.ones((H,))
ao=mx.random.normal((1,M,H)).astype(mx.float32);mo=mx.random.normal((1,M,H)).astype(mx.float32)
inputs=(h,pm,fn1,fn2,base,scale,norm,ao,mo)
mx.eval(inputs);mx.synchronize();mx.clear_cache()
report={'scope':'Synthetic native M6 HC attention prep, FFN prep and post combine; exact functions/constants and K3 route; no model generation', 'shape':[1,M,C,H], 'consts':consts, 'resident_inputs_bytes':sum(int(x.nbytes) for x in inputs), 'source_module_sha256':hashlib.sha256(Path(d.__file__).read_bytes()).hexdigest(), 'sinkhorn_kernel':d._sinkhorn_use_kernel(),'premix_kernel':d._hc_premix_use_kernel(),'allocator_limit_bytes':512*1024**2,'cache_limit_bytes':64*1024**2,'results':{}}
references=None
for mode in ['eager','compiled']:
 if mode=='eager':
  prep=lambda *a:d._hc_attn_prep_impl(*a,*consts)
  ffn=lambda *a:d._hc_ffn_prep_impl(*a,*consts)
  post=d._hc_post_impl
 else:
  keys=(*consts,d._sinkhorn_use_kernel(),d._hc_premix_use_kernel())
  prep=d._hc_compiled('attn_prep',*keys);ffn=d._hc_compiled('ffn_prep',*keys);post=d._hc_compiled('moe_combine')
 def run():
  ai,ap,at,ac=prep(h,pm,fn1,base,scale,norm)
  mi,res,fp,fc,fpre=ffn(ao,h,ap,at,ac,fn2,base,scale,norm)
  return ai,mi,post(mo,res,fp,fc)
 input_active=int(mx.get_active_memory());mx.reset_peak_memory()
 start=time.perf_counter();out=run();mx.eval(out);mx.synchronize();first=time.perf_counter()-start
 peak_first=int(mx.get_peak_memory())
 if references is None: references=out
 maxabs=[float(mx.max(mx.abs(a-b)).item()) for a,b in zip(references,out)]
 identical=all(bool(mx.array_equal(a,b).item()) for a,b in zip(references,out))
 samples=[]
 for _ in range(3):
  start=time.perf_counter()
  for _ in range(20): out=run();mx.eval(out)
  mx.synchronize();samples.append((time.perf_counter()-start)/20)
 report['results'][mode]={'first_compile_and_eval_s':first,'median_s':statistics.median(samples),'samples_s':samples,'input_active_bytes':input_active,'peak_first_bytes':peak_first,'peak_all_bytes':int(mx.get_peak_memory()),'max_abs_by_output':maxabs,'bit_exact':identical}
path=Path('/tmp/dsv41-hc-decode-20260917/probe-results.json');path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
