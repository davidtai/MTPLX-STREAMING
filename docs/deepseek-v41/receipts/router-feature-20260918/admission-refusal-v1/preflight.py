"""No-MLX validation of the cloned diagnostic accounting and source pins."""
import importlib.abc
import importlib.util
import sys
import json
import hashlib
from pathlib import Path

class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('MLX forbidden in preflight')

sys.meta_path.insert(0,NoMLX())
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
assert a['decode_slots_per_layer']==105 and a['physical_bound_bytes']<=110000000000
assert a['diagnostic_host_allowance_bytes']==384*1024**2
assert a['diagnostic_active_allowance_bytes']==64*1024**2
result={'cpu_only':True,'source_hashes_valid':True,'baseline_bytes':b['baseline_bytes'],'admission':a}
(r/'diagnostic-admission.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:a[k] for k in ('decode_slots_per_layer','physical_bound_bytes',
    'allocator_limit_bytes','active_bound_bytes','host_reserve_bytes',
    'diagnostic_host_allowance_bytes','diagnostic_active_allowance_bytes')}))
