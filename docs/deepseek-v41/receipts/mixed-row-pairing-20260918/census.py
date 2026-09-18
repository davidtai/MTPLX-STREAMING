"""Count native C109 cache-hit pairing geometry without loading MLX."""
from pathlib import Path
from collections import Counter
import gzip,hashlib,importlib.abc,json,statistics,sys

class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('CPU geometry cannot import MLX')
sys.meta_path.insert(0,NoMLX())
sys.path.insert(0,'/tmp/dsv41-lookahead-io-20260918')
from restore_bank import restore
from mtplx.expert_streaming import RoutingPhase

repo=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
path=repo/'docs/deepseek-v41/receipts/mtp-verify-routes-20260913/mtp-verify-routes-16k-1024-v2.json.gz'
with gzip.open(path,'rt') as f:data=json.load(f)
out={'scope':'Native logical policy hits at109 slots, captured73 resident seed,206 M6 routes; physical transient reuse is excluded. Geometry only, no performance claim.',
     'trace_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'layers':[]}
for layer in range(40):
    state=data['initial_banks'][str(layer)]
    state['_slot_to_expert'].extend([None]*(109-state['persistent_slots']));state['persistent_slots']=109
    bank=restore(state,policy='transition-window',single_pool=True)
    cases=[]
    for ids in data['target_routes_by_layer'][str(layer)]:
        route=bank.plan(ids,phase=RoutingPhase.DECODE)
        counts=Counter(e for e in ids if e in route.hits)
        rows=sum(counts.values());paired=sum(n//2*2 for n in counts.values())
        cases.append({'hit_rows':rows,'paired_rows':paired,'solo_rows':rows-paired,'miss_experts':len(route.misses)})
    out['layers'].append({'layer':layer,'mean_hit_rows':statistics.mean(c['hit_rows'] for c in cases),
        'mean_paired_rows':statistics.mean(c['paired_rows'] for c in cases),'mean_solo_rows':statistics.mean(c['solo_rows'] for c in cases),
        'paired_hit_row_fraction':sum(c['paired_rows'] for c in cases)/sum(c['hit_rows'] for c in cases),
        'cases':cases})
out['mean_hit_rows']=statistics.mean(l['mean_hit_rows'] for l in out['layers'])
out['mean_paired_rows']=statistics.mean(l['mean_paired_rows'] for l in out['layers'])
out['layer34']={k:v for k,v in out['layers'][34].items() if k!='cases'}
target=Path(__file__).with_name('census.json');target.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='layers'}))
