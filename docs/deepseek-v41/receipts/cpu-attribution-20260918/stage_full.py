"""Clone the retained exact full lane and add coarse CPU diagnostics."""
import ast,hashlib,json,shutil,subprocess
from pathlib import Path

root=Path(__file__).resolve().parent
repo=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
base=Path('/tmp/dsv41-strict-cache-20260918/full-v1')
r=root/'full-v1';r.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
for sub in ('native','compat','packed'):
    p=json.loads((base/sub/'installation.json').read_text())
    for name,digest in p.get('runtime_source_sha256',{}).items():assert sha(repo/name)==digest,name
    for name,digest in p['helper_sha256'].items():assert sha(base/sub/name)==digest,name
    shutil.copytree(base/sub,r/sub,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
for p in r.rglob('*.py'):p.write_text(p.read_text().replace(str(base),str(r)))
for name in ('clock_probe.py','cpu_install.py'):shutil.copyfile(root/name,r/'packed'/name)

# Reuse the already-proved diagnostic composition, with only16MiB extra host
# ownership, no added GPU arrays and the retained through110 capacity search.
p=r/'packed/packed_admission.py'
s=Path('/tmp/dsv41-router-feature-20260918/full-v2/packed/packed_admission.py').read_text()
s=s.replace('/tmp/dsv41-router-feature-20260918/full-v2',str(r))
s=s.replace('diagnostic_host = 384 * 1024**2','diagnostic_host = 16 * 1024**2')
s=s.replace('diagnostic_active = 64 * 1024**2','diagnostic_active = 0')
s=s.replace('range(105, old_capacity, -1)',"range(strict['capacity_search_ceiling'], old_capacity, -1)")
s=s.replace('capacity_search_ceiling=105',"capacity_search_ceiling=strict['capacity_search_ceiling']")
s=s.replace('through105; explicit384MiB diagnostic host and64MiB Metal allowances',
            'through110; explicit16MiB diagnostic host and no additional tensor owners')
p.write_text(s)
p=r/'packed/run_full.py';s=p.read_text()
old='    original_dspark_generate = decode_module.dspark_generate'
new='''    from clock_probe import Recorder
    from cpu_install import install_decode,install_model as install_cpu_model
    cpu_recorder = Recorder()
    cpu_installation = install_decode(decode_module,cpu_recorder)
    original_dspark_generate = decode_module.dspark_generate'''
assert s.count(old)==1;s=s.replace(old,new)
old='            projection_owner_report.update(install_model(target_ref(), backbone_type=backbone_type))'
new=old+'\n            cpu_installation["model_entrypoints"] = install_cpu_model(target_ref(),cpu_recorder)'
assert s.count(old)==1;s=s.replace(old,new)
old="        receipt['strict_allocator'] = dict(STRICT_ALLOCATOR)"
new=old+'''
        receipt['cpu_attribution'] = cpu_recorder.snapshot()
        receipt['cpu_instrumentation'] = dict(cpu_installation)
        receipt['performance_claim'] = False
        receipt['dspark']['instrumented_decode_wall_s'] = receipt['dspark']['decode_wall_s']
        receipt['dspark']['instrumented_decode_tok_s'] = receipt['dspark']['decode_tok_s']
        receipt['dspark']['decode_wall_s'] = None
        receipt['dspark']['decode_tok_s'] = None'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace("'ar': 'external_receipt', 'dspark': 'current_run'", "'ar': 'external_receipt', 'dspark': 'instrumented_diagnostic'")
s=s.replace(" 'decode_steps':decode_steps, 'dspark_depth':stage_depth,", " 'decode_steps':decode_steps, 'dspark_depth':stage_depth,\n 'performance_claim':False, 'diagnostic_scope':'coarse wall/main-thread/process CPU clocks;16MiB extra host, no added GPU fences or tensor owners',")
anchor = "    with PREFIX.with_suffix('.passes.jsonl').open('a') as f:"
replacement = "    row['performance_claim'] = False\n    for field in ('decode_tok_s','decode_wall_s'):\n        row['instrumented_'+field] = row[field]\n        row[field] = None\n" + anchor
assert s.count(anchor)==1;s=s.replace(anchor,replacement)
p.write_text(s)
for sub in ('native','compat','packed'):
    p=r/sub/'installation.json';d=json.loads(p.read_text());d['source_commit']=head
    d['scope']='Exact16K/1024 nativeKV16 D5/M6 diagnostic with coarse CPU clocks. No native tensor operations or GPU fences change;16MiB extra host reserve and no added GPU tensor ownership. Not a TPS candidate.'
    if sub=='packed':
        d['native_admission_sha256']=sha(r/'native/admission.py')
        d['helper_sha256'].update({'clock_probe.py':'pending','cpu_install.py':'pending'})
        d['cpu_clock_validation']={'path':str(root/'clock-validation.json'),'sha256':sha(root/'clock-validation.json')}
    d['helper_sha256']={name:sha(r/sub/name) for name in d['helper_sha256']}
    p.write_text(json.dumps(d,indent=2)+'\n')
for name in ('launch_full.py',):
    (r/name).write_text((base/name).read_text().replace(str(base),str(r)))
s=(base/'command.sh').read_text().replace(str(base),str(r))
s=s.replace('full-strict-cache-20260918-v1','full-cpu-attribution-20260918-v1')
s=s.replace('--host-overhead-gib 1.2774620056152344','--host-overhead-gib 1.2930870056152344')
(r/'command.sh').write_text(s)
pre=Path('/tmp/dsv41-router-feature-20260918/full-v2/preflight.py').read_text()
pre=pre.replace("a['decode_slots_per_layer']==105","a['decode_slots_per_layer']==109")
pre=pre.replace('384*1024**2','16*1024**2').replace("a['diagnostic_active_allowance_bytes']==64*1024**2","a['diagnostic_active_allowance_bytes']==0")
(r/'preflight.py').write_text(pre)
for p in r.rglob('*.py'):ast.parse(p.read_text())
audit={'source_commit':head,'base':str(base),'diagnostic_host_allowance_bytes':16*1024**2,
    'new_gpu_tensor_owners_bytes':0,'clock_validation_sha256':sha(root/'clock-validation.json'),
    'helper_sha256':{str(p.relative_to(r)):sha(p) for p in r.rglob('*') if p.is_file() and p.suffix in ('.py','.json') and 'artifact' not in p.parts}}
(r/'source-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({'root':str(r),'source':head,'helper_count':len(audit['helper_sha256']),'extra_host_bytes':16*1024**2}))
