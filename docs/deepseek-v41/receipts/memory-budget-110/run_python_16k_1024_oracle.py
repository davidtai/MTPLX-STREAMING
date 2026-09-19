import dataclasses, hashlib, importlib.util, json, os, pathlib, signal, subprocess, threading, time
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
signal.alarm(900)
source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
 raise RuntimeError('tracked source must be clean before the diagnostic load')
wrapper_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
if os.environ.get('MTPLX_BELADY_ORACLE') != '1': raise RuntimeError('oracle diagnostic must be explicitly enabled')
path=pathlib.Path('/tmp/dsv41-110-preflight/python-16k-1024-oracle.os.jsonl');stop=threading.Event();phase='startup'
def sample():
 with path.open('a') as f: f.write(json.dumps({'phase':phase,'snapshot':host_memory_snapshot()})+'\n')
def monitor():
 while not stop.wait(0.25): sample()
t=threading.Thread(target=monitor,daemon=True);sample();t.start()
try:
 spec=importlib.util.spec_from_file_location('ab','scripts/deepseek_v41/ab_decode_env_levers.py');ab=importlib.util.module_from_spec(spec);spec.loader.exec_module(ab)
 args=ab.build_parser().parse_args()
 if (args.arms != ['cell16k_ring_v2_attn_pf0'] or args.decode_mode != 'ar'
     or args.context_tokens != 16384 or args.decode_tokens != 1023
     or args.max_kv != 17664 or args.warm_repeat or args.stage_timing
     or args.prefill_stage_timing or args.syncs or args.with_mtp
     or ab._device_sample_resolved(args)):
  raise RuntimeError('this capture requires one 16K/1023-step pf0 AR pass, no extra passes or device sampling')
 original=ab._load_model
 def load(*a,**kw):
  global phase
  phase='loading';r=original(*a,**kw);phase='loaded'
  rt=r.model._mtplx_expert_runtime;report=r.model._mtplx_resident_load_report
  print('STORAGE_POLICY',json.dumps({'expert':rt.reader.cache_mode,'engram':report.get('engram_io_cache_modes')}),flush=True)
  if rt.reader.cache_mode!='f-nocache' or set(report.get('engram_io_cache_modes',{}).values())!={'f-nocache'}: raise RuntimeError('cache bypass not installed')
  sample();return r
 ab._load_model=load
 initial_banks={}
 snapshot=ab._stream_counters_snapshot
 def capture_snapshot(model):
  rt=model._mtplx_expert_runtime
  if not initial_banks:
   for layer,bank in rt._banks.items():
    if bank._prefetch_ring is not None: raise RuntimeError('offline replay requires pf0')
    initial_banks[str(layer)]={
     'expert_count':bank.expert_count,'persistent_slots':bank.persistent_slots,
     'transient_slots':bank.transient_slots,'single_pool':bank.single_pool,
     'cache_policy':bank.cache_policy,'frequency_decay':bank.frequency_decay,
     '_expert_to_slot':dict(bank._expert_to_slot),'_slot_to_expert':list(bank._slot_to_expert),
     '_pool_recency':dict(bank._pool_recency),'_pool_clock':bank._pool_clock,
     '_protected':sorted(bank._protected),'_decode_epoch':bank._decode_epoch,
     '_history':[dataclasses.asdict(h) for h in bank._history],
     '_prefill_seed_candidates':sorted(bank._prefill_seed_candidates),
     '_prefill_route_freq':dict(bank._prefill_route_freq),
     '_saw_decode_since_prefill':bank._saw_decode_since_prefill,
    }
  return snapshot(model)
 ab._stream_counters_snapshot=capture_snapshot
 generate=ab._generate
 generate_started=False
 def traced_generate(*a,**kw):
  global phase, generate_started
  if generate_started: raise RuntimeError('a second generation is not supported by this capture')
  generate_started=True
  phase='generation';sample()
  rt=kw['model']._mtplx_expert_runtime
  if rt._derived_cache_policy: raise RuntimeError('fixed-storage reservation was not installed')
  admitted_tokens=len(kw['prompt_ids'])+int(kw['steps'])
  with rt.admit_kv_tokens(admitted_tokens):
   r=generate(*a,**kw)
  if rt._live_kv_tokens != 0: raise RuntimeError('KV admission did not release')
  phase='generation_end';sample()
  oracle=rt._belady_oracle
  if oracle is None: raise RuntimeError('oracle was disabled during observation')
  sequences={str(layer):seq for layer,seq in oracle._sequences.items()}
  if set(sequences)!=set(initial_banks): raise RuntimeError('incomplete layer trace')
  expected=len(r['generated'])-1
  if any(len(seq)!=expected for seq in sequences.values()): raise RuntimeError('incomplete decode trace')
  payload={'purpose':'diagnostic ordered routes; instrumented throughput is not a promotion receipt',
   'source_commit':source_commit,'wrapper_sha256':wrapper_sha256,'record_bytes':rt.spec.expert_record_bytes,
   'decode_steps':expected,'kv_admission_tokens':admitted_tokens,'kv_released':True,
   'initial_banks':initial_banks,'sequences':sequences,
   'runtime_oracle_semantics':'cold start, mandatory batch admission; not a bypass-capable floor'}
  pathlib.Path('/tmp/dsv41-110-preflight/python-16k-1024-oracle.routes.json').write_text(json.dumps(payload))
  return r
 ab._generate=traced_generate
 raise SystemExit(ab.main())
finally:
 stop.set();t.join(2);sample()
