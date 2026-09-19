"""CPU-only maximum-history object ownership inventory for the new proposer."""
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys


class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('MLX prohibited in host index inventory')


sys.meta_path.insert(0,NoMLX())
from consensus import SuffixConsensus
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
ROOT = Path(__file__).resolve().parent
out = ROOT/'host-inventory.json'
if out.exists():
    raise RuntimeError('refusing evidence overwrite')
before = host_memory_snapshot()
if not before['box']['ok'] or before['box']['used_bytes']+512*1024**2>110000000000:
    raise RuntimeError('bounded CPU inventory does not fit')
spec = importlib.util.spec_from_file_location('lookup','/tmp/dsv41-hybrid-lookup-20260918/lookup.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def deep_size(value,seen=None):
    if seen is None:
        seen=set()
    identity=id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    total=sys.getsizeof(value)
    if isinstance(value,dict):
        total+=sum(deep_size(k,seen)+deep_size(v,seen) for k,v in value.items())
    elif isinstance(value,(list,tuple)):
        total+=sum(deep_size(v,seen) for v in value)
    elif hasattr(value,'__dict__'):
        total+=deep_size(vars(value),seen)
    return total


# Unique valid-vocabulary IDs maximize keys and one-element occurrence lists;
# repeated histories replace many such lists/keys with cheaper integer entries.
# This structural check is separate from proposal quality and uses no target IDs.
history=list(range(16384+1024))
baseline=module.LookupExtension(history,minimum_context=2,extra_tokens=2)
candidate=SuffixConsensus(history,min_suffix=3,min_count=2,max_extra=2)
baseline.append_committed([])
candidate.append_committed([])
base=deep_size(baseline)
both=deep_size((baseline,candidate))
extra=both-base
assert extra<32*1024**2
report={'complete':True,'cpu_only':True,'history_tokens':len(history),
        'baseline_deep_bytes':base,'consensus_incremental_deep_bytes':extra,
        'combined_deep_bytes':both,'extra_host_allowance_bytes':32*1024**2,
        'consensus_key_count':len(candidate.ends),
        'consensus_entries':sum(len(v) for v in candidate.ends.values()),
        'before':before,'after':host_memory_snapshot(),
        'incremental_screen_bound_bytes':512*1024**2,
        'scope':'Python object payload inventory on unique-token maximum history. Interpreter, allocator arenas and temporary sets remain covered by separate host reserve; this is not total process usage.',
        'mlx_imported':any(k=='mlx' or k.startswith('mlx.') for k in sys.modules)}
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('before','after')}))
