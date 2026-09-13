"""CPU-only cache screen; full continuation is hypothetical beyond measured prefix."""
import hashlib, importlib.util, json
from pathlib import Path
spec=importlib.util.spec_from_file_location('replay','/tmp/dsv41-110-preflight/replay_route_policies.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
root=Path('/tmp/dsv41-110-preflight')
trace=json.loads((root/'python-16k-128-timeline-largecache.routes.json').read_text())
old=json.loads((root/'python-16k-1024-oracle.routes.json').read_text())
assert all(seq==old['sequences'][layer][:len(seq)] for layer,seq in trace['sequences'].items())

def segmented(payload, fraction):
    total=0
    for layer,seq in payload['sequences'].items():
        bank=m.restore(payload['initial_banks'][layer])
        bank._protected_cap=max(1,int(bank.persistent_slots*fraction))
        bank._protected=set(sorted(bank._protected,key=lambda e:bank._pool_recency.get(e,0),reverse=True)[:bank._protected_cap])
        for step in seq:
            p=bank.try_plan_all_hits(step,phase='decode')
            if p is None:p=bank.plan(step,phase='decode')
            total+=len(p.misses)
    return {'misses':total,'misses_per_token':total/payload['decode_steps']}

result={'source_commit':trace['source_commit'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'purpose':'causal short prefix replay and hypothetical long continuation; no throughput claim',
        'route_prefix_equal_to_previous_full_trace':True,'screens':{}}
for label,payload in [('measured_128',trace),('hypothetical_1023',{**trace,'decode_steps':old['decode_steps'],'sequences':old['sequences']})]:
    candidates={f'protected-{fraction}':segmented(payload,fraction) for fraction in (.1,.2,.3,.4,.5,.6,.7,.8,.9,1.)}
    for policy in ('lru','frequency'):
        candidates[policy]=m.replay(payload,policy=policy,single_pool=False)
    control=m.replay(payload)
    result['screens'][label]={'control':control,'candidates':candidates}
    print(label,'control',control['misses'],'best',sorted([(k,v['misses']) for k,v in candidates.items()],key=lambda x:x[1])[:4])
(root/'python-16k-128-timeline-largecache.policy-screen.json').write_text(json.dumps(result,indent=2)+'\n')
