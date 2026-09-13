"""Bounded larger-residency experiment; same AR graph and pinned workload."""
import dataclasses, hashlib, importlib.util, json, os, pathlib, signal, subprocess, threading, time
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
signal.alarm(900)
PREFIX=pathlib.Path('/tmp/dsv41-110-preflight/python-16k-128-timeline-largecache')
source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
 raise RuntimeError('tracked source must be clean')
base=float(os.environ['MTPLX_DSV41_BOX_BASELINE_GB'])*1e9
if not 0 <= base <= 20e9: raise RuntimeError('invalid measured baseline')
if os.environ.get('MTPLX_DSV41_IO_READ_FANOUT')!='4': raise RuntimeError('explicit fanout4 required')
if os.environ.get('MTPLX_BELADY_ORACLE')!='1' or os.environ.get('MTPLX_DSV41_DECODE_TIMELINE')!='1': raise RuntimeError('explicit diagnostic instrumentation required')
GIB=1024**3; RECORD=18800640; FIXED=23578252736
engine=int(110e9-base-2*GIB-10*GIB)
slots=(engine-FIXED)//(40*RECORD)
if not 1 <= slots <= 100: raise RuntimeError(f'unexpected slot geometry: {slots}')
# Same graph/codec/length and shared 48-record transient pool as measured control.
# Price added persistent bytes against the measured 72-slot peak. Static audit
# confirms banks are filled through memoryviews, with no capacity-dependent copies.
# The measured 72-slot peak was 1,007,640 B below the storage-delta projection.
# Retain that higher projection for the next capacity rather than spending it.
active_bound=84670328292+(slots-72)*40*RECORD
diagnostic_reserve=16*1024**2
physical_bound=base+active_bound+2*GIB+2*GIB+diagnostic_reserve
wired_now=host_memory_snapshot()['box']['wired_bytes']
if physical_bound > 109e9 or wired_now+active_bound+2*GIB > 100*GIB:
 raise RuntimeError('static active/cache/host/wired bound lacks headroom')
bounds={'source_commit':source_commit,'wrapper_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
 'baseline_bytes':base,'engine_budget_bytes':engine,'expected_slots_per_layer':slots,
 'active_bound_bytes':active_bound,'physical_bound_bytes':physical_bound,'wired_before_bytes':wired_now,
 'diagnostic_reserve_bytes':diagnostic_reserve,
 'scope':'same 16K AR pf0 geometry with shorter 128-step diagnostic forward geometry; exact persistent delta beyond conservative 72-slot peak projection; full 2GiB allocator cache and 2GiB Python capacity; 10GiB transient reserve'}
PREFIX.with_suffix('.bounds.json').write_text(json.dumps(bounds,indent=2)+'\n')
print('STATIC_BOUND',json.dumps(bounds),flush=True)
stop=threading.Event();phase='startup'
def sample():
 with PREFIX.with_suffix('.os.jsonl').open('a') as f:f.write(json.dumps({'phase':phase,'snapshot':host_memory_snapshot()})+'\n')
def monitor():
 while not stop.wait(.25):sample()
t=threading.Thread(target=monitor,daemon=True);sample();t.start()
try:
 spec=importlib.util.spec_from_file_location('ab','scripts/deepseek_v41/ab_decode_env_levers.py');ab=importlib.util.module_from_spec(spec);spec.loader.exec_module(ab)
 args=ab.build_parser().parse_args()
 fixture=pathlib.Path('docs/deepseek-v41/receipts/memory-budget-110/python-prompt-ids.json')
 prompt_rows=[r for r in json.loads(fixture.read_text())['prompts'] if r['target_tokens']==16384]
 if len(prompt_rows)!=1 or hashlib.sha256(json.dumps(prompt_rows[0]['token_ids']).encode()).hexdigest()!='38894d01011a0146f8621dfdaf4bc4e618092d772d8cc28a70e21163a16799a2':
  raise RuntimeError('pinned 16K prompt digest mismatch')
 if (pathlib.Path(args.model).resolve()!=pathlib.Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
     or pathlib.Path(args.prompt_ids_file).resolve()!=fixture.resolve()
     or args.box_target_gb!=110 or args.allocator_cache_gib!=2 or args.host_overhead_gib!=2
     or args.memory_plan_from or args.memory_limit_gib is not None or args.expert_cache_limit_gib is not None):
  raise RuntimeError('this bound requires exact artifact, fixture and live 110/10/2/2 allocation')
 if (args.arms!=['cell16k_ring_v2_attn_pf0'] or args.decode_mode!='ar' or args.context_tokens!=16384
     or args.decode_tokens!=128 or args.max_kv!=17664 or args.transient_band_gib!=10
     or args.warm_repeat or args.stage_timing or args.prefill_stage_timing or args.syncs or args.with_mtp
     or ab._device_sample_resolved(args)):
  raise RuntimeError('this bound requires the exact unchanged AR graph and workload')
 original=ab._load_model
 def load(*a,**kw):
  global phase
  plan=ab._resolve_target_plan(a[0])
  if plan['engine_budget_bytes']!=engine or plan['session_bank_reserve_bytes']!=0:
   raise RuntimeError('effective allocation differs from the static bound')
  phase='loading';r=original(*a,**kw);phase='loaded'
  rt=r.model._mtplx_expert_runtime;report=r.model._mtplx_resident_load_report
  if rt.plan.persistent_slots!=slots*40:raise RuntimeError('installed slot count differs from bound')
  if rt.reader.cache_mode!='f-nocache' or set(report.get('engram_io_cache_modes',{}).values())!={'f-nocache'}:raise RuntimeError('uncached I/O required')
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
 def traced_generate(*a,**kw):
  global phase
  phase='generation';sample();rt=kw['model']._mtplx_expert_runtime
  with rt.admit_kv_tokens(len(kw['prompt_ids'])+int(kw['steps'])):r=generate(*a,**kw)
  if rt._live_kv_tokens!=0:raise RuntimeError('KV admission not released')
  phase='generation_end';sample()
  sequences={str(layer):seq for layer,seq in rt._belady_oracle._sequences.items()}
  expected=len(r['generated'])-1
  if set(sequences)!=set(initial_banks) or any(len(seq)!=expected for seq in sequences.values()):raise RuntimeError('incomplete route capture')
  payload={'purpose':'short instrumented profile, not a throughput promotion receipt',
   'source_commit':source_commit,'record_bytes':rt.spec.expert_record_bytes,
   'decode_steps':expected,'kv_admission_tokens':len(kw['prompt_ids'])+int(kw['steps']),
   'kv_released':True,'initial_banks':initial_banks,'sequences':sequences}
  PREFIX.with_suffix('.routes.json').write_text(json.dumps(payload))
  return r
 ab._generate=traced_generate
 raise SystemExit(ab.main())
finally:
 stop.set();t.join(2);sample()
