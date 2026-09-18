"""Small source/geometry/policy construction proof; no MLX imports."""
import ast,hashlib,importlib.abc,inspect,json,sys,textwrap
from pathlib import Path

class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('MLX is forbidden in preflight')

sys.meta_path.insert(0,NoMLX())
from mtplx.expert_streaming_models import ExpertMemoryPlan,ExpertStreamingModelSpec
from mtplx.expert_streaming import GlobalPrefetchRing,RoutingPhase
from mtplx.expert_runtime import ExpertStreamingRuntime
from paired_config import PairedPrefetchConfig
from restore_bank import restore

r=Path(__file__).resolve().parent
p=json.loads((r/'installation.json').read_text())
d=json.loads((r/'routes.json').read_text())
spec=ExpertStreamingModelSpec(**{**p['spec'],'full_indexer_layers':(),'island_pin_order':()})
plan=ExpertMemoryPlan(**{**p['plan'],'persistent_slots_by_layer':()})
config=PairedPrefetchConfig(**{**p['config'],'persistent_slots_by_layer':(),'island_layers':(),'mmap_island_layers':()})
assert spec.routed_layer_indices==(30,31)
assert plan.persistent_slots==210 and plan.slots_per_layer==105 and plan.prefetch_ring_slots==16
ring=GlobalPrefetchRing(ring_size=16,base=105+48,expert_count=384)
banks={}
for layer in (30,31):
    state=d['initial_banks'][str(layer)]
    state['_slot_to_expert'].extend([None]*(105-state['persistent_slots']))
    state['persistent_slots']=105
    bank=restore(state,policy='transition-window',single_pool=True,layer_id=layer,
                 prefetch_ring=ring,prefetch_slots=16)
    assert bank.slot_count==169 and bank._prefetch_ring is ring and len(bank._expert_to_slot)==73
    banks[layer]=bank
ids=d['routes']['31'][0]
e=next(x for x in ids if x not in banks[31]._expert_to_slot)
load=banks[31].plan_prefetch([e])[0]
ticket=banks[31].prefetch_ticket(e)
assert banks[31].commit_prefetch(e,ticket=ticket)
before=(dict(banks[31]._expert_to_slot),banks[31]._decode_epoch)
route,txn=banks[31].plan_transaction(ids,phase=RoutingPhase.DECODE)
assert e in route.prefetch_hits and e not in route.misses
assert all(s==load.slot for x,s in zip(route.experts,route.slots) if x==e)
txn.rollback_completion()
assert (banks[31]._expert_to_slot,banks[31]._decode_epoch)==before
tree=ast.parse((r/'probe.py').read_text())
old=next(n.value.value for n in tree.body if isinstance(n,ast.Assign)
         and any(isinstance(t,ast.Name) and t.id=='old' for t in n.targets))
source=textwrap.dedent(inspect.getsource(ExpertStreamingRuntime.prefetch_experts))
assert source.count(old)==1
for f in r.glob('*.py'):
    compile(f.read_text(),str(f),'exec')
p['helper_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in r.iterdir()
    if f.is_file() and f.name not in ('installation.json','stage.py','preflight.json','command.sh')}
p['paired_capability']={'scope':'two-layer operator only','native_config_guard_unchanged':True,
    'native_published_ring_transaction_observed':True,'published_expert':e,'published_slot':load.slot,
    'policy_mapping_and_epoch_rollback_exact':True}
(r/'installation.json').write_text(json.dumps(p,indent=2)+'\n')
result={'cpu_only':True,'raw_bank_bytes':(210+48+16)*18800640,
    'persistent_slots_per_layer':105,'prefetch_slots':16,'physical_ring_shared':True,
    'warm_residents_per_layer':73,'capability':p['paired_capability']}
(r/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
