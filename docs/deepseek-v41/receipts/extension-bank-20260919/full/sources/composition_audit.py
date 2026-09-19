"""CPU-only construction audit; no model load or extra benchmark."""
import ast
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import textwrap


class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('MLX forbidden in composition audit')


sys.meta_path.insert(0, NoMLX())
r = Path(__file__).resolve().parent
base = Path('/tmp/dsv41-hybrid-lookup-20260918/full-v1')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def method(path, cls, name):
    node = ast.parse(path.read_text())
    c = next(n for n in node.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)


old = method(base/'packed/owned_projection.py','BF16Output','__call__')
new = method(r/'packed/projection_install.py','ScheduledOutput','__call__')
assert ast.dump(ast.Module(body=old.body[1:],type_ignores=[])) == ast.dump(
    ast.Module(body=new.body[1:],type_ignores=[]))
module = ast.parse((r/'packed/projection_install.py').read_text())
rewrite = next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='scheduled_run_source')
namespace = {}
exec(compile(ast.Module(body=[rewrite],type_ignores=[]),'<cpu_schedule_rewrite>','exec'),namespace)
lane_path = r/'packed/plane_lane.py'
lane = method(lane_path,'PackedDecode','run')
source = textwrap.dedent('\n'.join(lane_path.read_text().splitlines()[lane.lineno-1:lane.end_lineno]))
updated = namespace['scheduled_run_source'](source)
ast.parse(updated)
assert updated.replace('        self.issue_next()\n','') == source
assert sha(r/'packed/fused_transpose.py') == sha(r.parent/'fused_transpose.py')
for name in ('plane_lane.py','packed_storage.py','paired_kernels.py','kernels.py','hybrid_install.py','lookup.py'):
    assert sha(r/'packed'/name)==sha(base/'packed'/name),name

os.environ['MTPLX_ENGRAM_CACHE_LIMIT']='67108864'
spec = importlib.util.spec_from_file_location('new_admission',r/'packed/packed_admission.py')
admission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(admission)
proof = json.loads((r/'packed/installation.json').read_text())
compat = json.loads((r/'compat/installation.json').read_text())
cases = []
for baseline in (10447192064,11027267584,11820597248):
    a = admission.resolve_admission(baseline,3511418880,grow=True,
        expected_receipt_hash=compat['phase_memory_control_sha256'],
        strict_allocator=proof['strict_allocator']['identity'])
    assert a['initial_decode_slots_per_layer']==84 and 85<=a['decode_slots_per_layer']<=112
    assert a['physical_bound_bytes']<=110000000000
    assert a['host_reserve_bytes']==1438773248
    assert a['active_bound_bytes']+a['decode_cache_allowance_bytes']<=a['allocator_limit_bytes']
    assert a['projection_source_bytes_retired']==0
    cases.append({k:a[k] for k in ('baseline_bytes','initial_decode_slots_per_layer',
        'decode_slots_per_layer','physical_bound_bytes','active_bound_bytes',
        'resize_active_bound_bytes','seed_active_bound_bytes','overflow_append_active_bound_bytes',
        'steady_decode_active_bound_bytes','allocator_limit_bytes')})

result = {'complete':True,'cpu_only':True,
    'native_projection_body_identical_after_weight_acquire':True,
    'native_expert_body_identical_except_prebound_issue':True,
    'expert_kernels_and_hybrid_proposals_unchanged':True,
    'fused_transpose_matches_all40_component':True,
    'admission_cases':cases,
    'scope':'Static construction and phase accounting only; full model timing and output remain unmeasured.'}
(r/'composition-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
