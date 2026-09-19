import importlib.util, json, pathlib, signal, threading, time
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
signal.alarm(900)
path=pathlib.Path('/tmp/dsv41-110-preflight/python-16k-1024-defaults.os.jsonl');stop=threading.Event();phase='startup'
def sample():
 with path.open('a') as f: f.write(json.dumps({'phase':phase,'snapshot':host_memory_snapshot()})+'\n')
def monitor():
 while not stop.wait(0.25): sample()
t=threading.Thread(target=monitor,daemon=True);sample();t.start()
try:
 spec=importlib.util.spec_from_file_location('ab','scripts/deepseek_v41/ab_decode_env_levers.py');ab=importlib.util.module_from_spec(spec);spec.loader.exec_module(ab)
 original=ab._load_model
 def load(*a,**kw):
  global phase
  phase='loading';r=original(*a,**kw);phase='loaded'
  rt=r.model._mtplx_expert_runtime;report=r.model._mtplx_resident_load_report
  print('STORAGE_POLICY',json.dumps({'expert':rt.reader.cache_mode,'engram':report.get('engram_io_cache_modes')}),flush=True)
  if rt.reader.cache_mode!='f-nocache' or set(report.get('engram_io_cache_modes',{}).values())!={'f-nocache'}: raise RuntimeError('cache bypass not installed')
  sample();return r
 ab._load_model=load
 generate=ab._generate
 def traced_generate(*a,**kw):
  global phase
  phase='generation';sample();r=generate(*a,**kw);phase='generation_end';sample();return r
 ab._generate=traced_generate
 raise SystemExit(ab.main())
finally:
 stop.set();t.join(2);sample()
