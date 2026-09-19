import ast,hashlib,json,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent;repo=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
base=Path('/tmp/dsv41-allocator-cache-20260918');p=json.loads((base/'installation.json').read_text())
for name,digest in p['runtime_source_sha256'].items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,name
s=(r/'probe.py').read_text();s=s.replace('import plane_lane_part2','import plane_lane_ordered')
a='import mlx.core as mx';s=s.replace(a,a+"\nfrom library_identity import identify\nlibrary=identify(proof['strict_allocator'])")
s=s.replace("'before':before, 'arms':[], 'complete':False","'before':before, 'library':library, 'arms':[], 'complete':False")
s=s.replace('def run_arm(cache_bytes, sequence):','def run_arm(mode, sequence):\n    cache_bytes = 256 * 1024**2')
s=s.replace("result = {'part_size':part_size,", "result = {'mode':mode, 'part_size':part_size,")
s=s.replace('lane = plane_lane if part_size==3 else plane_lane_part2',"lane = plane_lane if mode=='native' else plane_lane_ordered")
s=s.replace("for sequence,cache_bytes in enumerate(proof['arms']):","for sequence,mode in enumerate(proof['arms']):")
s=s.replace('run_arm(cache_bytes,sequence)','run_arm(mode,sequence)')
s=s.replace("if a['cache_limit_bytes']>0","if a['mode']=='native'").replace("if a['cache_limit_bytes']==0","if a['mode']=='ordered'")
(r/'probe.py').write_text(s)
s=(r/'run_screen.py').read_text()
a="child = subprocess.Popen([sys.executable, str(ROOT / 'probe.py')])";assert s.count(a)==1
b="env = os.environ.copy()\nenv['DYLD_LIBRARY_PATH'] = str(Path(installation['strict_allocator']['path']).parent)\nchild = subprocess.Popen([sys.executable, str(ROOT / 'probe.py')], env=env)";s=s.replace(a,b);(r/'run_screen.py').write_text(s)
raw=18800640;packed=17694720;capacity=109;delta=(capacity-104)*raw
for key in ('expert_cache_limit_bytes','persistent_budget_bytes','persistent_cache_bytes'):p['plan'][key]=capacity*raw
p['plan'].update(persistent_slots=capacity,slots_per_layer=capacity,unallocated_bytes=p['plan']['unallocated_bytes']-delta)
p['config']['expert_cache_limit_bytes']=capacity*raw
p['bank_capacity']=capacity+48;p['native_slot_bytes']=(capacity+48)*raw;p['packed_slot_bytes']=(capacity+48)*packed
p['bound_components'].update(raw_bank_bytes=p['native_slot_bytes'],packed_bank_bytes=p['packed_slot_bytes'])
p['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
p['strict_allocator']=json.loads(Path('/tmp/dsv41-strict-cache-20260918/full-v1/packed/installation.json').read_text())['strict_allocator']['identity']
p['scope']='One layer34 actual109-slot native M6 replay under the attested strict allocator. Compare native token-order rows/GPU argsort with physical-slot grouped rows/CPU inverse permutation. Every projection, dtype, reduction, batch, read and final expert output position is preserved. No full-model TPS claim.'
p['comparison']='Native versus grouped-row/CPU-permutation packed decode at256MiB strict inactive-cache limit.'
p['arms']=['native','ordered','native','ordered','native']
p['layer_selection']='Same independently preselected median-miss layer34; expand the physically restored saved73-slot bank to109 before each arm. Do not select layers from candidate timing.'
p['bound_scope']='Unchanged9GiB envelope:5GiB Metal/cache/compile plus4GiB host/reader/compiler. Exact157-slot raw bank updated in inventory; all kernels/temporary shapes unchanged and actual strict inactive cache256MiB. One runtime at a time.'
p['allocator_sources']['rule']='Use the already-verified pinned strict allocator. This screen changes row permutation only; no cache-policy experiment.'
p['helper_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in r.iterdir() if f.is_file() and f.suffix in ('.py','.json') and f.name!='installation.json'}
for f in r.glob('*.py'):ast.parse(f.read_text())
(r/'installation.json').write_text(json.dumps(p,indent=2)+'\n')
s=(base/'command.sh').read_text().replace(str(base),str(r));(r/'command.sh').write_text(s)
print(json.dumps({'root':str(r),'source_commit':p['source_commit'],'capacity':capacity,'bank_capacity':p['bank_capacity'],'raw_bank_bytes':p['native_slot_bytes'],'packed_bank_bytes':p['packed_slot_bytes'],'incremental_bound_bytes':p['static_incremental_bound_bytes'],'helpers':len(p['helper_sha256'])}))
