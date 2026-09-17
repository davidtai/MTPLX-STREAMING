"""Native 16K-context M8 allocation probe at cap16; no throughput promotion."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
if os.environ.get('_GPU_WINDOW_LOCKED')!='1':
    raise RuntimeError('parent-held GPU/service guard required')
ROOT=Path('/tmp/dsv41-depth-replay-20260917')
OUT=ROOT/'target-m8-probe.json'
if OUT.exists(): raise RuntimeError('refusing to overwrite native M8 probe')
proof=json.loads((ROOT/'target-probe-installation.json').read_text())
for path,sha in proof['runtime_source_sha256'].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:
        raise RuntimeError('runtime source differs from native teacher')
if hashlib.sha256((ROOT/'compact_install.py').read_bytes()).hexdigest()!=proof['compact_install_sha256']:
    raise RuntimeError('compact loader installation changed')
reference=json.loads(Path('/tmp/dsv41-110-stage/full-cache-growth-d405-20260917.jsonl').read_text())
for key,value in reference['arm_env'].items():
    if value is None: os.environ.pop(key,None)
    else: os.environ[key]=str(value)
for key in ('MTPLX_DSV41_HC_COMPILE','MTPLX_DSV41_ATTN_COMPILE','MTPLX_DSV41_ATTN_WIN_MEMO'):
    os.environ[key]='0'
os.environ['MTPLX_DSV41_IO_READ_FANOUT']='4'
os.environ['MTPLX_BELADY_ORACLE']='0'
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
GIB=1024**3
saved=(93-16)*40*18800640
# Remove only fixed persistent bank storage from the previous conservative bound.
# Add 16GiB for new-shape workspace AND another 16GiB to its cache allowance.
# This low-capacity probe has ample physical headroom even at those margins.
active_bound=96461356124-saved+16*GIB
cache_allowance=3331897468+16*GIB
before=host_memory_snapshot();base=before['box']['used_bytes']
physical_bound=base+2*GIB+active_bound+cache_allowance
if (physical_bound>109500000000
    or before['box']['wired_bytes']+active_bound+cache_allowance>100*GIB):
    raise RuntimeError(f'native M8 probe cannot fit conservative bound {physical_bound}')
import compact_install as installed
mx=installed.mx;loader=installed.loader;ds=installed.dspark;dv=installed.dsv41
from dataclasses import replace
from mtplx.models.deepseek_v41_dspark_decode import DSparkDecodeStats,dspark_generate,_confidence_threshold_from_env
from mtplx.sampling import SamplerConfig
if _confidence_threshold_from_env(None) is not None:
    raise RuntimeError('M8 allocation probe requires the unchanged untrimmed draft width')
signal.alarm(900)
mx.set_memory_limit(80*GIB);mx.set_cache_limit(GIB)
dv._HC_COMPILE=False;dv._ATTN_COMPILE=False;dv._ATTN_WIN_MEMO=False
original_head_init=ds.DSparkHead.__init__
def head7(self,args):
    if args.dspark_block_size!=5: raise RuntimeError('native draft declaration changed')
    original_head_init(self,replace(args,dspark_block_size=7))
ds.DSparkHead.__init__=head7
ENGINE=20166777672+16*40*18800640+89686016
original_allocator=loader._component_bank_allocator_for
def checked_allocator(config,spec,root,manifest_path,manifest=None,*,additional_resident_bytes=None):
    if manifest is None: manifest=loader.load_expert_manifest(manifest_path)
    plan=config.memory_plan(spec,additional_resident_bytes=additional_resident_bytes,
        resident_discount_bytes=installed.expert_runtime.text_only_resident_discount(manifest,spec))
    if (plan.slots_per_layer!=16 or plan.fixed_bytes!=20166777672 or config.transient_slots!=48
        or config.prefetch_slots!=0 or config.cache_policy!='transition-window'
        or config.decode_miss_records_per_part!=3 or not config.verify_shared_overlap):
        raise RuntimeError(f'native M8 allocation plan changed: {plan}')
    return original_allocator(config,spec,root,manifest_path,manifest,additional_resident_bytes=additional_resident_bytes)
loader._component_bank_allocator_for=checked_allocator
report={'scope':'Native 16K/128 memory probe with cap16 persistent banks, D7/M8; no full-workload throughput claim',
 'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'runtime_source_sha256':proof['runtime_source_sha256'],
 'active_bound_bytes':active_bound,'cache_allowance_bytes':cache_allowance,
 'physical_bound_bytes':physical_bound,'host_reserve_bytes':2*GIB,'before':before,
 'engine_budget_bytes':ENGINE,'allocator_limit_bytes':80*GIB,'diagnostic_only':True,'performance_eligible':False}
print('M8_PROBE_BOUND',json.dumps(report),flush=True)
resident=loader.load_deepseek_v41_streaming(
 '/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4',
 memory_limit_bytes=ENGINE,max_live_kv_tokens=17664,runtime_reserve_bytes=2*GIB,
 admit=False,apply_memory_cap=False,with_mtp=True,slot_layout='component-banks',
 cache_scope='layer',island_layers=(),verify_record_hashes=False,transient_slots=48,
 prefetch_slots=0,cache_policy='transition-window',decode_miss_records_per_part=3,
 verify_shared_overlap=True,split_route_release='deferred',bypass_page_cache=True)
rt=resident.model._mtplx_expert_runtime
try:
    if (rt.plan.slots_per_layer!=16 or rt.reader.io_read_fanout!=4
        or resident.model.mtp.block_size!=7 or any(s.block_size!=7 for s in resident.model.mtp.layers)):
        raise RuntimeError('installed native M8 route differs')
    fixture=json.loads(Path('docs/deepseek-v41/receipts/memory-budget-110/python-prompt-ids.json').read_text())
    prompts=[p for p in fixture['prompts'] if p['target_tokens']==16384]
    if len(prompts)!=1: raise RuntimeError('prompt fixture changed')
    prompt=prompts[0]['token_ids']
    if hashlib.sha256(json.dumps(prompt).encode()).hexdigest()!='38894d01011a0146f8621dfdaf4bc4e618092d772d8cc28a70e21163a16799a2':
        raise RuntimeError('native prompt identity changed')
    def prefill_done(info):
        report['prefill']={'info':info,'active_bytes':int(mx.get_active_memory()),'cache_bytes':int(mx.get_cache_memory()),'peak_bytes':int(mx.get_peak_memory()),'host':host_memory_snapshot()}
        mx.reset_peak_memory()
        print('M8_PROBE_PREFILL',json.dumps(report['prefill']),flush=True)
    stats=DSparkDecodeStats()
    with rt.admit_kv_tokens(16384+128+8):
        ids=dspark_generate(resident.model,prompt,max_tokens=129,sampler=SamplerConfig(temperature=0.0),
            seed=0,stop_ids={1},speculative_depth=7,verify_decode_phase=True,stats=stats,prefill_callback=prefill_done)
    mx.synchronize()
    report['post_prefill']={'active_bytes':int(mx.get_active_memory()),'cache_bytes':int(mx.get_cache_memory()),'peak_bytes':int(mx.get_peak_memory()),'host':host_memory_snapshot()}
    expected=reference['dspark']['token_ids'][:129]
    report['generated_tokens']=len(ids);report['token_ids']=ids
    report['prefix_identical_to_retained_control']=(ids==expected)
    report['stats']=stats.to_dict()
    report['resolved_plan']={'prefill_slots':rt.plan.slots_per_layer,'decode_slots':rt.plan.slots_per_layer,'transient_slots':rt.plan.transient_slots,'effective_depth':stats.speculative_depth,'verify_chunks':stats.verify_chunks}
    OUT.write_text(json.dumps(report,indent=2)+'\n')
    print('M8_PROBE_COMPLETE',json.dumps({'post_prefill':report['post_prefill'],'generated_tokens':len(ids),'prefix_identical':ids==expected,'resolved_plan':report['resolved_plan']}),flush=True)
    if len(ids)!=129 or stats.speculative_depth!=7 or stats.verify_chunks!=[8]:
        raise RuntimeError('native M8 probe did not execute the intended work')
finally:
    rt.close()
