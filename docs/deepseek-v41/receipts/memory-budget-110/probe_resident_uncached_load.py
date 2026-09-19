import importlib.util, json, pathlib, signal, threading, time
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot, memory_profile_snapshot
signal.alarm(180)
out=pathlib.Path('/tmp/dsv41-110-preflight/resident-uncached-load-phases.jsonl'); stop=threading.Event(); phase='before_import'
def sample():
 with out.open('a') as f:
  f.write(json.dumps({'phase':phase,'snapshot':host_memory_snapshot()})+'\n'); f.flush()
def monitor():
 while not stop.wait(0.25): sample()
def mark(name):
 global phase
 phase=name;sample();print('LOAD_PHASE',name,flush=True)
sample();t=threading.Thread(target=monitor,daemon=True);t.start()
try:
 import mlx.core as mx
 spec=importlib.util.spec_from_file_location('ab','scripts/deepseek_v41/ab_decode_env_levers.py');ab=importlib.util.module_from_spec(spec);spec.loader.exec_module(ab)
 args=ab.build_parser().parse_args(['--context-tokens','16384','--decode-tokens','32','--max-kv','17408','--box-target-gb','100','--transient-band-gib','16','--allocator-cache-gib','2','--out','/tmp/dsv41-110-preflight/unused.jsonl'])
 ab._apply_arm_env('cell16k_ring_v2_attn_pf0')
 from mtplx.models import deepseek_v41_loader as loader
 for name in ('open_deepseek_v41_runtime','load_text_only_resident_arrays','construct_deepseek_v41_resident_model'):
  original=getattr(loader,name)
  def traced(*a,_original=original,_name=name,**kw):
   mark(_name+':start');r=_original(*a,**kw);mark(_name+':end');return r
  setattr(loader,name,traced)
 mark('load_model:start');resident=ab._load_model(args,ab._load_bench_module(),mx);mark('load_model:end')
 runtime=resident.model._mtplx_expert_runtime
 report=memory_profile_snapshot(phase='load_end',runtime=runtime,plan=runtime.plan,mx_module=mx)
 pathlib.Path('/tmp/dsv41-110-preflight/resident-uncached-load-result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 runtime.close()
finally:
 stop.set();t.join(2);sample()
