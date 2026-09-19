"""Compare the final process reader against pinned source, without importing MLX."""
import ast
import ctypes
import importlib.abc
import json
import hashlib
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch


class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('MLX is forbidden in this CPU screen')


sys.meta_path.insert(0, NoMLX())
from mtplx import deepseek_v41_memory_profile as p

SOURCE = '8f30fc002f3af27e0fec5afd56c522d3abf1da90'
old_source = subprocess.check_output(
    ['git', 'show', SOURCE + ':mtplx/deepseek_v41_memory_profile.py'], text=True)
function = next(n for n in ast.parse(old_source).body
                if isinstance(n, ast.FunctionDef) and n.name == '_mach_task_vm_info')
namespace = dict(vars(p))
exec(compile(ast.Module(body=[function], type_ignores=[]), '<pinned process reader>', 'exec'), namespace)
old = namespace['_mach_task_vm_info']
new = p._mach_task_vm_info
assert new()['phys_footprint_bytes'] > 0
timing = {}
for label, reader in [('control_before', old), ('candidate', new), ('control_after', old)]:
    samples = []
    for _ in range(100):
        started = time.perf_counter_ns()
        snapshot = reader()
        samples.append(time.perf_counter_ns() - started)
        assert snapshot['phys_footprint_bytes'] > 0
    timing[label] = dict(median_ns=statistics.median(samples), samples_ns=samples)
before = new()
payload = bytearray(16 * 1024**2)
after = new()
control_after = old()
growth = after['phys_footprint_bytes'] - before['phys_footprint_bytes']
assert growth >= 15 * 1024**2


def task_self():
    return 42


def short_response(task, flavor, info, count):
    info._obj.resident_size = 50000
    count._obj.value = 36  # successful rev0 response lacks phys_footprint
    return 0


p._mach_task_vm_reader.cache_clear()
with patch.object(ctypes, 'CDLL', return_value=SimpleNamespace(
    mach_task_self=task_self, task_info=short_response,
)):
    old_short = old()
    new_short = new()
p._mach_task_vm_reader.cache_clear()
assert old_short['phys_footprint_bytes'] == 0 and new_short is None
result = dict(scope='Own-process memory-reader CPU optimization only; no decode TPS claim',
    source_head=SOURCE, candidate_sha256=hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest(),
    timing=timing, allocation_bytes=len(payload), before=before, after=after,
    control_after=control_after, immediate_footprint_growth_bytes=growth,
    incomplete_response=dict(old=old_short, candidate=new_short),
    system_reader='Unchanged platform vm_stat; do not replace with rate-limited host_statistics64')
Path('/tmp/dsv41-online-cache-20260917/process-reader-production-screen.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v['median_ns'] for k,v in timing.items()}))
print('immediate_footprint_growth_bytes', growth)
print('incomplete response:', result['incomplete_response'])
