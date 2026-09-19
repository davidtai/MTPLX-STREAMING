"""Compose one bounded full request after the exact component schedule wins."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parent
repo = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
base = Path('/tmp/dsv41-hybrid-lookup-20260918/full-v1').resolve()
r = root / 'full-v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert head == 'd569990aa606bb86c877645f8d10f1d0d1450c58'
screen = json.loads((root/'overflow/probe.json').read_text())
assert screen['complete'] and all(a['all_outputs_exact'] for a in screen['arms'])
assert screen['candidate_over_control'] < .99
assert screen['candidate_over_control'] < 1-screen['control_spread_fraction']
r.mkdir()
for sub in ('native','compat','packed'):
    old = json.loads((base/sub/'installation.json').read_text())
    for name,digest in old.get('runtime_source_sha256',{}).items():
        assert sha(repo/name) == digest,name
    for name,digest in old['helper_sha256'].items():
        assert sha(base/sub/name) == digest,name
    shutil.copytree(base/sub,r/sub,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
for p in r.rglob('*.py'):
    p.write_text(p.read_text().replace(str(base),str(r)))
for origin,dest in ((root/'scheduled_projection.py','projection_install.py'),
                    (root/'overflow/overflow.py','overflow.py'),
                    (root/'overflow/fused_transpose.py','fused_transpose.py')):
    shutil.copyfile(origin,r/'packed'/dest)

def edit(path, edits):
    source = path.read_text()
    for old,new in edits:
        assert source.count(old)==1,(str(path),old)
        source=source.replace(old,new)
    path.write_text(source)

edit(r/'packed/overflow.py',[
 ("expert_cache_limit_bytes=old_plan.expert_cache_limit_bytes+delta,",
  "expert_cache_limit_bytes=(None if old_plan.expert_cache_limit_bytes is None\n"
  "                                  else old_plan.expert_cache_limit_bytes+delta),"),
])

# Initial packed growth and MTP seed retain their old envelopes. Only then is
# one row added independently. No copy of the first expert bank is allocated.
edit(r/'packed/packed_admission.py',[
 ("    allocator_limit = original['allocator_limit_bytes'] - embedding_host - lookup_host",
  "    expansion_host = 16 * 1024**2\n"
  "    expansion_credit = 40*67108864 - 40*34603008 - 3*67108864\n"
  "    overflow_payload = 40 * WEIGHTS\n"
  "    allocator_limit = original['allocator_limit_bytes'] - embedding_host - lookup_host - expansion_host"),
 ("original['prefill_physical_bound_bytes'] + embedding_host + lookup_host > DEFAULT_BOX_BUDGET_BYTES",
  "original['prefill_physical_bound_bytes'] + embedding_host + lookup_host + expansion_host > DEFAULT_BOX_BUDGET_BYTES"),
 ("        active = max(steady, resize, seed)",
  "        steady += overflow_payload - expansion_credit\n"
  "        # Price all new packed rows, one temporary native scale owner, three\n"
  "        # BF16 expansions including cold compilation, and extra page padding.\n"
  "        append_peak = seed + overflow_payload + (RAW-WEIGHTS) + 3*67108864 + original['page_padding_allowance_bytes']\n"
  "        active = max(steady, resize, seed, append_peak)"),
 ("physical = base + original['host_reserve_bytes'] + embedding_host + lookup_host + active",
  "physical = base + original['host_reserve_bytes'] + embedding_host + lookup_host + expansion_host + active"),
 ("host_reserve_bytes=original['host_reserve_bytes'] + embedding_host + lookup_host,",
  "host_reserve_bytes=original['host_reserve_bytes'] + embedding_host + lookup_host + expansion_host,"),
 ("prefill_physical_bound_bytes=original['prefill_physical_bound_bytes'] + embedding_host + lookup_host,",
  "prefill_physical_bound_bytes=original['prefill_physical_bound_bytes'] + embedding_host + lookup_host + expansion_host,"),
 ("        projection_source_bytes_retired=projection_source_bytes,",
  "        projection_source_bytes_retired=0,\n"
  "        projection_source_bytes_retained=projection_source_bytes,\n"
  "        predictable_expansion_host_allowance_bytes=expansion_host,\n"
  "        predictable_expansion_steady_credit_vs_cached_bf16_bytes=expansion_credit,\n"
  "        predictable_expansion_retained_bf16_bytes=2*67108864,\n"
  "        predictable_expansion_replacement_bf16_bound_bytes=3*67108864,"),
 ("        decode_slots_per_layer=capacity, growth_payload_bytes=net_growth,",
  "        initial_decode_slots_per_layer=capacity, initial_growth_payload_bytes=net_growth,\n"
  "        decode_slots_per_layer=capacity+1, growth_payload_bytes=net_growth+overflow_payload,\n"
  "        overflow_payload_bytes=overflow_payload, overflow_append_active_bound_bytes=append_peak,"),
 ("        projection_steady_credit_bytes=projection_credit,",
  "        inherited_projection_bound_normalization_bytes=projection_credit,\n"
  "        projection_steady_credit_bytes=projection_credit+expansion_credit,\n"
  "        projection_steady_credit_reference='Native predecessor retaining packed and all BF16 target projections; includes the inherited cold-source slack.',"),
 ("physical_bound_bytes=max(original['prefill_physical_bound_bytes'] + embedding_host + lookup_host, physical),",
  "physical_bound_bytes=max(original['prefill_physical_bound_bytes'] + embedding_host + lookup_host + expansion_host, physical),"),
 ("capacity_selection='largest admitted packed capacity above prefill through110; exact weight and resident-scale bytes',",
  "capacity_selection='largest admitted initial packed bank through110, then one independent row per layer after seed; no second resize',"),
 ("bound_scope='Adds16MiB fixed-workload lookup metadata within the retained native M8 tensor envelope.",
  "bound_scope='Adds16MiB projection scheduling/overflow host reserve. Original prefill and first-growth limits retained; native seed completes before overflow allocation; append/prime charged with all new rows, one native scale owner, three BF16 arrays and padding. Steady bound additionally credits only1098907648B vs the prior cached-BF16 route. The combined projection credit is normalized against the native predecessor retaining packed and full BF16 weights; source owners remain packed in this candidate. Adds16MiB fixed-workload lookup metadata within the retained native M8 tensor envelope."),
 ("native steady envelope credits packed projection retirement, with no tail credit;",
  "steady projection credit uses the packed-plus-bounded-expansion inventory, with no tail credit;"),
])

edit(r/'packed/run_full.py',[
 ("        # This function runs only during prefill. No added eval or peak reset.",
  "        # Record the unchanged native seed, then perform the bounded storage\n"
  "        # extension and prime. The complete boundary is charged to decode."),
 ("growth_transition, growth_report = install_growth(resident.model, growth_admission['decode_slots_per_layer'], mx=mx, admission=growth_admission)",
  "initial_admission = dict(growth_admission, decode_slots_per_layer=growth_admission['initial_decode_slots_per_layer'], growth_payload_bytes=growth_admission['initial_growth_payload_bytes'])\n"
  "            growth_transition, growth_report = install_growth(resident.model, initial_admission['decode_slots_per_layer'], mx=mx, admission=initial_admission)"),
 ("        return result\n\n    decode_module._seed_prefill_state = observe_seed_prefill",
  "        from overflow import append_rows\n"
  "        from projection_install import prime_model\n"
  "        target = a[0] if a else kw['model']\n"
  "        runtime = target._mtplx_expert_runtime\n"
  "        first_growth = dict(growth_report)\n"
  "        overflow_report = append_rows(runtime, mx=mx)\n"
  "        if runtime.plan.slots_per_layer != growth_admission['decode_slots_per_layer']:\n"
  "            raise RuntimeError('overflow capacity differs from admission')\n"
  "        projection_owner_report.update(prime_model(target))\n"
  "        seed_boundary_memory['after_overflow_and_prime'] = {\n"
  "            'mlx_active_bytes': int(mx.get_active_memory()),\n"
  "            'mlx_peak_bytes': int(mx.get_peak_memory()),\n"
  "        }\n"
  "        growth_report.update(first_packed_growth=first_growth, overflow=overflow_report,\n"
  "            decode_slots_per_layer=runtime.plan.slots_per_layer,\n"
  "            growth_payload_bytes=growth_admission['growth_payload_bytes'],\n"
  "            growth_seconds=first_growth['growth_seconds']+overflow_report['elapsed_ns']/1e9,\n"
  "            active_after_bytes=int(mx.get_active_memory()),\n"
  "            cache_after_bytes=int(mx.get_cache_memory()),\n"
  "            transition_peak_bytes=int(mx.get_peak_memory()),\n"
  "            physical_allocated_bytes=runtime.slots.allocated_bytes,\n"
  "            plan_limit_bytes=runtime.plan.total_limit_bytes,\n"
  "            plan_persistent_bytes=runtime.plan.persistent_cache_bytes,\n"
  "            timing_scope='Original first growth plus independent overflow allocation, projection installation and priming are all inside decode wall time; growth_seconds sums the two bank phases only.')\n"
  "        return result\n\n    decode_module._seed_prefill_state = observe_seed_prefill"),
])

scope = ('Exact16K/1024 Q4 nativeKV16, retained D5 plus original lookup. '
         'Packed target output weights remain resident; exact next-layer BF16 expansion overlaps expert reads. '
         'One independent packed expert row per layer added after native seed; no second bank copy. '
         'All transition time charged; additional16MiB host reserve.')
for sub in ('native','compat','packed'):
    p = r/sub/'installation.json'
    d = json.loads(p.read_text())
    d['source_commit'],d['scope'] = head,scope
    if sub=='packed':
        d['native_admission_sha256']=sha(r/'native/admission.py')
        for name in ('overflow.py','fused_transpose.py'):d['helper_sha256'][name]='pending'
        d['projection_ownership']={
            'helper_sha256':sha(r/'packed/projection_install.py'),
            'component_receipt':str(root/'overflow/probe.json'),
            'component_receipt_sha256':sha(root/'overflow/probe.json'),
            'source_packed_bytes_retained':1384120320,
            'maximum_live_bf16_payload_bytes':3*67108864,
            'conservative_credit_vs_cached_bf16_bytes':1098907648,
            'extra_host_bytes':16*1024**2,
            'initial_capacity_ceiling':110,'final_capacity_ceiling':111,
            'scope':scope}
    d['helper_sha256']={name:sha(r/sub/name) for name in d['helper_sha256']}
    p.write_text(json.dumps(d,indent=2)+'\n')
(r/'launch_full.py').write_text((base/'launch_full.py').read_text().replace(str(base),str(r)))
command=(base/'command.sh').read_text().replace(str(base),str(r))
command=command.replace('full-hybrid-lookup-20260918-v1','full-predictable-expansion-20260919-v1')
command=command.replace('--host-overhead-gib 1.3243370056152344','--host-overhead-gib 1.3399620056152344')
command=command.replace('GPU_WINDOW_LOCK_TIMEOUT=120','GPU_WINDOW_LOCK_TIMEOUT=600')
(r/'command.sh').write_text(command)
(r/'wait_and_run.py').write_text((root/'overflow/wait_and_run.py').read_text())
preflight=(base/'preflight.py').read_text()
preflight=preflight.replace("a['decode_slots_per_layer']==110", "a['decode_slots_per_layer']==111 and a['initial_decode_slots_per_layer']==110")
preflight=preflight.replace('hybrid-admission.json','expansion-admission.json')
(r/'preflight.py').write_text(preflight)
for p in r.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
audit={'source_commit':head,'base':str(base),'scope':scope,
       'component_sha256':sha(root/'overflow/probe.json'),
       'helper_sha256':{str(p.relative_to(r)):sha(p) for p in r.rglob('*')
            if p.is_file() and p.suffix in ('.py','.json') and 'artifact' not in p.parts}}
(r/'source-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({'source_commit':head,'root':str(r),'cpu_only':True}))
