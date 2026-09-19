import ast,hashlib,json,shutil,subprocess
from pathlib import Path
repo=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
base=Path('/tmp/dsv41-allocator-cache-20260918')
attn=Path('/tmp/dsv41-strict-cache-20260918/attention-screen')
r=Path('/tmp/dsv41-strict-cache-20260918/expert-screen');r.mkdir()
proof=json.loads((base/'installation.json').read_text())
for name,digest in proof['runtime_source_sha256'].items():
 assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,name
for name,digest in proof['helper_sha256'].items():
 assert hashlib.sha256((base/name).read_bytes()).hexdigest()==digest,name
for name in ('probe.py','plane_lane_part2.py','plane_lane.py','restore_bank.py','paired_kernels.py','kernels.py','packed_storage.py','routes.json'):
 shutil.copy2(base/name,r/name)
shutil.copy2(attn/'library_identity.py',r/'library_identity.py')
(r/'artifact').symlink_to((base/'artifact').resolve(),target_is_directory=True)
p=r/'probe.py';s=p.read_text()
def change(a,b):
 global s
 assert s.count(a)==1,a
 s=s.replace(a,b)
change("LAYER = proof['layer']", "CASE=ROOT/'cases'/os.environ['DSV41_OPERATOR_CASE']; BINARY=os.environ['DSV41_OPERATOR_BINARY']\nLAYER = proof['layer']")
s=s.replace("(ROOT/'probe.json')","(CASE/'probe.json')")
change('import mlx.core as mx',"import mlx.core as mx\nfrom library_identity import identify, observe_cache_policy\nlibrary=identify(proof['binary_choices'][BINARY])")
change("'before':before, 'arms':[], 'complete':False", "'before':before, 'binary':BINARY, 'library':library, 'arms':[], 'complete':False")
a=s.index("    controls=[a['warm_total_ns']")
b=s.index("    report['complete']=True",a)
s=s[:a]+"    report['cache_policy_observation']=observe_cache_policy(mx,strict=BINARY=='strict')\n"+s[b:]
change("('complete','latency_ratio','control_spread_fraction','active_after_close_bytes')", "('complete','binary','active_after_close_bytes')")
p.write_text(s)
s=(attn/'run_comparison.py').read_text()
a="files = [Path(proof['model_path']) / name for name in proof['shards']]"
b="""inventory=json.loads((ROOT/'artifact/manifest.json').read_text())
files=[Path(proof['model_path'])/'experts.bin']
files += [ROOT/'artifact'/component[field]['file']
          for component in inventory['layers'][proof['layer']]['components'].values()
          for field in ('descriptors','payload','bases')]"""
assert s.count(a)==1;s=s.replace(a,b)
a="""        life['medians_ns'] = [statistics.median(t['median_total_ns'] for t in c['timings'])
                              for c in row['cases']]"""
b="        life['warm_total_ns']=row['arms'][0]['warm_total_ns']"
assert s.count(a)==1;s=s.replace(a,b)
(r/'run_comparison.py').write_text(s)
proof['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
proof['binary_choices']=json.loads((attn/'installation.json').read_text())['binary_choices']
proof['arms']=[256*1024**2]
proof['scope']='Layer34 actual104-slot native M6 route replay, unchanged part3 plane overlap and256MiB inactive-cache limit. Compare wheel, rebuilt stock, strict allocator, rebuilt stock. Actual physical reads and native outputs are compared across all206 routes; no backbone or full-model TPS claim.'
proof['comparison']='Matched isolated host libraries with byte-identical Metal shaders; one206-route native expert replay per process.'
proof['allocator_sources']['rule']='Strict library evicts older inactive buffers before inserting a fitting buffer and immediately trims on cache-limit reduction. No credit is taken from active, graph or copy allocations.'
proof['helper_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in r.iterdir() if p.suffix in ('.py','.json')}
for p in r.glob('*.py'):ast.parse(p.read_text())
(r/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
s=(attn/'command.sh').read_text().replace(str(attn),str(r))
(r/'command.sh').write_text(s)
print(json.dumps({'root':str(r),'helpers':len(proof['helper_sha256']),'static_incremental_bound_bytes':proof['static_incremental_bound_bytes'],'source_commit':proof['source_commit']}))
