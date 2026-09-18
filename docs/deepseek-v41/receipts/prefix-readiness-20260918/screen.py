"""CPU route diagnostic; no implementation or predicted TPS is inferred."""
import gzip, hashlib, json, runpy, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(Path.cwd()))
HELPER=Path('docs/deepseek-v41/receipts/memory-budget-110/replay_route_policies.py')
restore=runpy.run_path(str(HELPER))['restore']  # installs NoMLX import guard
TRACE=Path('docs/deepseek-v41/receipts/mtp-verify-routes-20260913/mtp-verify-routes-16k-1024-v2.json.gz')
control=json.loads(Path('/tmp/dsv41-row-weighted-cache-20260917/screen.json').read_text())
assert hashlib.sha256(TRACE.read_bytes()).hexdigest()==control['trace_sha256']
trace=json.loads(gzip.decompress(TRACE.read_bytes()))
assert trace['complete'] and trace['cycles']==206
start=time.perf_counter()
stats={r:{'all_hit_prefix':0,'prefix_misses':0,'prefix_requires_every_miss':0} for r in (1,2,3,4,5,6)}
misses=routes=unique_requests=0
for layer,sequence in trace['target_routes_by_layer'].items():
 bank=restore(trace['initial_banks'][layer],policy='transition-window',single_pool=True)
 extra=102-bank.persistent_slots
 bank._slot_to_expert.extend([None]*extra)
 bank.persistent_slots=bank._persistent_capacity=102
 bank.slot_count+=extra;bank._protected_cap=max(1,int(102*.8))
 counts=[]
 for route in sequence:
  assert len(route)==36
  missing=set(bank.plan(route,phase='decode').misses)
  counts.append(len(missing));misses+=len(missing);routes+=1
  unique_requests+=len(set(route))
  for r in stats:
   needed=missing.intersection(route[:r*6]);s=stats[r]
   s['all_hit_prefix']+=not needed
   s['prefix_misses']+=len(needed)
   s['prefix_requires_every_miss']+=bool(missing) and needed==missing
 assert counts==control['arms']['native']['per_layer'][layer]['per_cycle']
assert misses==35164 and routes==8240
for s in stats.values():
 s['all_hit_prefix_fraction']=s['all_hit_prefix']/routes
 s['prefix_share_of_demand_reads']=s['prefix_misses']/misses
 s['fraction_waiting_for_every_read']=s['prefix_requires_every_miss']/routes
report={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'scope':'Current-and-past native policy replay from captured73 residents plus29 empty slots. Policy demand misses only; no transient reuse, read scheduling, partial-row arithmetic, or full-model TPS proof.', 'mlx_imports_blocked':True,'trace_sha256':hashlib.sha256(TRACE.read_bytes()).hexdigest(),'replay_helper_sha256':hashlib.sha256(HELPER.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'exact_native_per_layer_per_cycle_counts':True,'routes':routes,'demand_misses':misses,'unique_expert_requests':unique_requests,'one_row_expert_requests':routes*36,'one_row_request_amplification':routes*36/unique_requests,'prefixes':stats,'elapsed_s':time.perf_counter()-start}
(ROOT/'screen.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
