"""Compare whole-machine admission at the actual full-run baseline, without MLX."""
import hashlib,importlib.abc,importlib.util,json,os,sys
from pathlib import Path
class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('CPU admission imported MLX')
sys.meta_path.insert(0,NoMLX())
os.environ['MTPLX_ENGRAM_CACHE_LIMIT']='67108864'
root=Path(__file__).resolve().parent
bound=json.loads(Path('/tmp/dsv41-110-stage/full-strict-cache-20260918-v1.bounds.json').read_text())
proof=json.loads((root/'packed/installation.json').read_text())
compat=json.loads((root/'compat/installation.json').read_text())
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
strict=load(root/'packed/packed_admission.py','strict_admission')
stock=load(Path('/tmp/dsv41-cache-budget-20260918/full-v1/packed/packed_admission.py'),'stock_admission')
base=bound['baseline_bytes'];wired=bound['growth_admission']['wired_before_bytes']
kw=dict(grow=True,expected_receipt_hash=compat['phase_memory_control_sha256'])
a=strict.resolve_admission(base,wired,strict_allocator=proof['strict_allocator']['identity'],**kw)
b=stock.resolve_admission(base,wired,**kw)
assert a==bound['growth_admission']
for field in ('host_reserve_bytes','allocation_margin_bytes','full_logical_kv_extra_allowance_bytes','page_padding_allowance_bytes','transition_start_active_bound_bytes','projection_steady_credit_bytes','tail_transition_active_credit_bytes','prefill_active_bound_bytes','prefill_cache_allowance_bytes','prefill_physical_bound_bytes'):
    assert a[field]==b[field],field
assert a['physical_bound_bytes']<=110000000000
assert a['decode_cache_allowance_bytes']==256*1024**2
result={'cpu_only':True,'baseline_bytes':base,'wired_before_bytes':wired,'stock_slots':b['decode_slots_per_layer'],'strict_slots':a['decode_slots_per_layer'],'stock_physical_bound_bytes':b['physical_bound_bytes'],'strict_physical_bound_bytes':a['physical_bound_bytes'],'inactive_cache_overshoot_allowance_removed_bytes':a['strict_cache_overshoot_credit_bytes'],'all_other_reserve_components_equal':True,'stock_admission':b,'strict_admission':a}
(root/'same-baseline-admission.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if not k.endswith('_admission')}))
