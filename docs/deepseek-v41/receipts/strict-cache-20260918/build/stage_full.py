import ast,hashlib,json,shutil,subprocess
from pathlib import Path
repo=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
base=Path('/tmp/dsv41-cache-budget-20260918/full-v1')
root=Path('/tmp/dsv41-strict-cache-20260918')
r=root/'full-v1';r.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
for sub in ('native','compat','packed'):
 p=json.loads((base/sub/'installation.json').read_text())
 for name,digest in p.get('runtime_source_sha256',{}).items():assert sha(repo/name)==digest,name
 for name,digest in p['helper_sha256'].items():assert sha(base/sub/name)==digest,name
 shutil.copytree(base/sub,r/sub,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
for p in r.rglob('*.py'):
 p.write_text(p.read_text().replace(str(base),str(r)))
shutil.copy2(root/'library_identity.py',r/'packed/library_identity.py')
identity=json.loads((root/'attention-screen/installation.json').read_text())['binary_choices']['strict']
identity['path']=str(Path(identity['path']).resolve())
assert sha(Path(identity['path']))==identity['sha256']
proof={'identity':identity,'source_archive_sha256':'bf892386ccac1d249acc9da21574fd7b95a995f4c58b03310105fe05f077b175','cache_invariant':'free and set_cache_limit hold the existing allocator mutex and evict before publishing an inactive-cache total above the configured limit. Oversized freed buffers are released directly.','decode_cache_limit_bytes':256*1024**2,'overshoot_allowance_removed_bytes':2258155644,'capacity_search_ceiling':110,'capacity_geometry':'Page-aligned5898240-byte components; existing concatenate, pointer/row-index and int32 route arithmetic unchanged. Maximum110 persistent rows plus48 shared transients; every selected capacity still satisfies the original active/copy/seed/wired inequalities.','operators':{}}
for kind,key in (('attention','all_cross_binary_outputs_and_states_exact'),('expert','all206_cross_binary_reads_and_outputs_exact')):
 p=root/(kind+'-screen')/'summary.json';d=json.loads(p.read_text());assert d['complete'] and d[key]
 proof['operators'][kind]={'path':str(p),'sha256':sha(p),'exact_field':key}
p=r/'packed/packed_admission.py';s=p.read_text()
s=s.replace('def resolve_admission(base, wired, *, grow, expected_receipt_hash):','def resolve_admission(base, wired, *, grow, expected_receipt_hash, strict_allocator):')
a="    prefill_requested_cache = original['requested_allocator_cache_limit_bytes']"
b="""    strict = installation['strict_allocator']
    if strict_allocator != strict['identity']:
        raise RuntimeError('strict allocator was not attested at construction')
    for operator in strict['operators'].values():
        blob = Path(operator['path']).read_bytes()
        row = json.loads(blob)
        if hashlib.sha256(blob).hexdigest() != operator['sha256'] or not row['complete'] or not row[operator['exact_field']]:
            raise RuntimeError('matched allocator operator evidence changed')
    removed_overshoot = original['decode_cache_overshoot_allowance_bytes']
    if removed_overshoot != strict['overshoot_allowance_removed_bytes']:
        raise RuntimeError('original free-cache overshoot allowance changed')
    original['decode_cache_overshoot_allowance_bytes'] = 0
    prefill_requested_cache = original['requested_allocator_cache_limit_bytes']"""
assert s.count(a)==1;s=s.replace(a,b)
s=s.replace('for capacity in range(106, old_capacity, -1):',"for capacity in range(strict['capacity_search_ceiling'], old_capacity, -1):")
s=s.replace("        physical_ceiling_bytes=DEFAULT_BOX_BUDGET_BYTES,","        physical_ceiling_bytes=DEFAULT_BOX_BUDGET_BYTES,\n        strict_allocator=dict(strict_allocator),\n        strict_cache_overshoot_credit_bytes=removed_overshoot,\n        capacity_search_ceiling=strict['capacity_search_ceiling'],")
s=s.replace('above prefill through106;','above prefill through110;')
s=s.replace('original overshoot/KV/compiler/wired allowances and32MiB helper host reserve retained;','original KV/compiler/wired allowances and32MiB helper host reserve retained; only the separately added inactive-cache overshoot allowance is removed under the attested strict allocator;')
p.write_text(s)
p=r/'packed/packed_phase.py';s=p.read_text();assert s.count('old_capacity < capacity <= 106')==1;s=s.replace('old_capacity < capacity <= 106',"old_capacity < capacity <= admission['capacity_search_ceiling']");p.write_text(s)
p=r/'packed/run_full.py';s=p.read_text();a='import mlx.core as mx';assert s.count(a)==1
s=s.replace(a,a+"\nfrom library_identity import identify\nSTRICT_ALLOCATOR=identify(PACKED_INSTALLATION['strict_allocator']['identity'])")
a="grow=GROWTH_ENABLED, expected_receipt_hash=COMPATIBILITY['phase_memory_control_sha256'])";assert s.count(a)==1
s=s.replace(a,"grow=GROWTH_ENABLED, expected_receipt_hash=COMPATIBILITY['phase_memory_control_sha256'], strict_allocator=STRICT_ALLOCATOR)")
s=s.replace(" 'packed_scales_installation':PACKED_INSTALLATION,"," 'strict_allocator':dict(STRICT_ALLOCATOR),\n 'packed_scales_installation':PACKED_INSTALLATION,")
s=s.replace("        receipt['bounded_engram'] = dict(bounded_engram_report)","        receipt['bounded_engram'] = dict(bounded_engram_report)\n        receipt['strict_allocator'] = dict(STRICT_ALLOCATOR)")
p.write_text(s)
for sub in ('native','compat','packed'):
 p=r/sub/'installation.json';d=json.loads(p.read_text());d['source_commit']=head
 d['scope']='One complete native KV16 D5/M6 strict-cache allocator candidate. Exact native arithmetic and all active/copy/seed/prefill/KV/wired reserves retained; only the inactive-cache overshoot reserve is removed after loaded-library attestation.'
 if sub=='packed':
  d['strict_allocator']=proof
  d['native_admission_sha256']=sha(r/'native/admission.py')
  d['helper_sha256']['library_identity.py']='pending'
  d['projection_ownership']['capacity_search_ceiling']=110
  d['projection_ownership']['capacity_geometry']=proof['capacity_geometry']
  d['cache_budget_candidate'].update(decode_cache_overshoot_allowance_bytes=0,overshoot_credit_bytes=2258155644,slot_capacity_ceiling=110,scope=d['scope'])
 d['helper_sha256']={name:sha(r/sub/name) for name in d['helper_sha256']}
 p.write_text(json.dumps(d,indent=2)+'\n')
launcher='''"""Select an isolated host library only inside the already-held GPU window."""
import hashlib,json,os,sys
from pathlib import Path
if os.environ.get('_GPU_WINDOW_LOCKED')!='1':raise RuntimeError('parent GPU guard required')
root=Path(__file__).resolve().parent
proof=json.loads((root/'packed/installation.json').read_text())
identity=proof['strict_allocator']['identity']
if hashlib.sha256(Path(identity['path']).read_bytes()).hexdigest()!=identity['sha256']:raise RuntimeError('strict library changed')
env=os.environ.copy();env['DYLD_LIBRARY_PATH']=str(Path(identity['path']).parent)
os.execve(sys.executable,[sys.executable,str(root/'packed/run_full.py'),*sys.argv[1:]],env)
'''
(r/'launch_full.py').write_text(launcher)
s=(base/'command.sh').read_text().replace(str(base),str(r)).replace('full-cache-budget-20260918-v1','full-strict-cache-20260918-v1').replace(str(r/'packed/run_full.py'),str(r/'launch_full.py'))
(r/'command.sh').write_text(s)
s=(base/'summarize.py').read_text().replace('full-cache-budget-20260918-v1','full-strict-cache-20260918-v1').replace("assert result['decode_cache_overshoot_allowance_bytes']==2258155644","assert result['decode_cache_overshoot_allowance_bytes']==0\nassert bound['growth_admission']['strict_cache_overshoot_credit_bytes']==2258155644\nresult['strict_allocator']=receipt['strict_allocator']").replace('One complete cache-budget candidate','One complete strict-cache allocator candidate')
(r/'summarize.py').write_text(s)
for p in r.rglob('*.py'):ast.parse(p.read_text())
audit={'source_commit':head,'base':str(base),'strict_allocator':proof,'helper_sha256':{str(p.relative_to(r)):sha(p) for p in r.rglob('*') if p.is_file() and p.suffix in ('.py','.json') and 'artifact' not in p.parts},'production_package_replaced':False}
(r/'source-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({'root':str(r),'source_commit':head,'helper_count':len(audit['helper_sha256']),'production_package_replaced':False}))
