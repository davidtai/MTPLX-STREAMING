"""No-MLX validation of the cloned diagnostic accounting and source pins."""
import importlib.abc
import importlib.util
import sys
import os
import json
import hashlib
from pathlib import Path

class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('MLX forbidden in preflight')

sys.meta_path.insert(0,NoMLX())
os.environ['MTPLX_ENGRAM_CACHE_LIMIT']='67108864'
r=Path(__file__).resolve().parent
p=json.loads((r/'packed/installation.json').read_text())
for sub in ('packed','native','compat'):
    d=json.loads((r/sub/'installation.json').read_text())
    for n,h in d['helper_sha256'].items():
        assert hashlib.sha256((r/sub/n).read_bytes()).hexdigest()==h,(sub,n)
b=json.loads(Path('/tmp/dsv41-110-stage/full-strict-cache-20260918-v1.bounds.json').read_text())
c=json.loads((r/'compat/installation.json').read_text())
s=importlib.util.spec_from_file_location('admission',r/'packed/packed_admission.py')
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)
a=m.resolve_admission(b['baseline_bytes'],b['growth_admission']['wired_before_bytes'],
    grow=True,expected_receipt_hash=c['phase_memory_control_sha256'],
    strict_allocator=p['strict_allocator']['identity'])
assert a['decode_slots_per_layer']==113 and a['initial_decode_slots_per_layer']==84 and a['physical_bound_bytes']<=110000000000
assert a['embedding_host_allowance_bytes']==32*1024**2 and a['lookup_host_allowance_bytes']==16*1024**2
assert a['embedding_post_prefill_credit_bytes']==1323827200 and a['embedding_prefill_credit_bytes']==0
from mtplx.expert_runtime import resolve_box_target_mlx_limit_bytes
import shlex
argv=shlex.split((r/'command.sh').read_text())
host_arg=argv[argv.index('--host-overhead-gib')+1]
engine=82693389128
env={'MTPLX_ENGRAM_CACHE_LIMIT':'67108864',
     'MTPLX_DSV41_BOX_TARGET_GB':'110',
     'MTPLX_DSV41_BOX_BASELINE_GB':format(b['baseline_bytes']/1e9,'.17g'),
     'MTPLX_DSV41_MLX_CACHE_LIMIT_GIB':'1',
     'MTPLX_DSV41_TRANSIENT_BAND_GIB':format((a['allocator_limit_bytes']-engine)/1024**3,'.17g'),
     'MTPLX_DSV41_HOST_OVERHEAD_GIB':host_arg}
resolved=resolve_box_target_mlx_limit_bytes(env)
assert resolved['engine_budget_bytes']==engine,resolved
assert resolved['mlx_limit_bytes']==a['allocator_limit_bytes'],resolved
result={'cpu_only':True,'source_hashes_valid':True,'baseline_bytes':b['baseline_bytes'],'admission':a}
result['cli_budget_resolution']=resolved
(r/'draft-extension-admission.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:a[k] for k in ('decode_slots_per_layer','physical_bound_bytes',
    'allocator_limit_bytes','active_bound_bytes','host_reserve_bytes',
    'lookup_host_allowance_bytes','embedding_post_prefill_credit_bytes')}))
