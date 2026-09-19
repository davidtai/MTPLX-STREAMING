"""CPU capacity sensitivity on native routes, not a Q2 trajectory prediction."""
import gzip
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import time


class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('MLX forbidden in CPU capacity screen')


sys.meta_path.insert(0, NoMLX())
from savings_restore import restore
from mtplx.expert_streaming import RoutingPhase

ROOT = Path(__file__).resolve().parent
REPO = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
TRACE = REPO/'docs/deepseek-v41/receipts/mtp-verify-routes-20260913/mtp-verify-routes-16k-1024-v2.json.gz'
OUT = ROOT/'savings-proxy.json'
if OUT.exists():
    raise RuntimeError('refusing evidence overwrite')
data = json.loads(gzip.decompress(TRACE.read_bytes()))
assert data['complete'] and data['cycles'] == 206 and data['slots_per_layer'] == 73
started = time.perf_counter()
rows = []
for capacity in (73, 110, 112, 113):
    counts = []
    for layer in range(40):
        bank = restore(data['initial_banks'][str(layer)],
                       policy='transition-window', single_pool=True, layer_id=layer)
        extra = capacity - bank.persistent_slots
        bank._slot_to_expert.extend([None]*extra)
        bank.persistent_slots = bank._persistent_capacity = capacity
        bank.slot_count += extra
        bank._protected_cap = max(1, int(capacity*.8))
        count = 0
        for route in data['target_routes_by_layer'][str(layer)]:
            plan = bank.plan(route, phase=RoutingPhase.DECODE)
            count += len(plan.loads)
        counts.append(count)
    reads = sum(counts)
    if capacity == 73 and reads != data['decode_records_read']:
        raise RuntimeError(f'original trace census differs: {reads}')
    rows.append({'capacity': capacity, 'reads': reads, 'reads_by_layer': counts,
                 'packed_weight_bytes': reads * 17694720})
base = rows[1]['reads']
for row in rows[1:]:
    row['saved_records_vs110'] = base-row['reads']
    row['traffic_reduction_fraction_vs110'] = (base-row['reads'])/base
    row['saved_bytes_vs110'] = (base-row['reads'])*17694720
report = {'complete': True, 'cpu_only': True, 'elapsed_s': time.perf_counter()-started,
          'trace_sha256': hashlib.sha256(TRACE.read_bytes()).hexdigest(),
          'restore_sha256': hashlib.sha256((ROOT/'savings_restore.py').read_bytes()).hexdigest(),
          'policy_sha256': hashlib.sha256((REPO/'mtplx/expert_streaming.py').read_bytes()).hexdigest(),
          'scope': 'Native206-cycle M6 trace with actual73-slot initial residents, expanded with empty capacity; no Q2 target routes or complete new prefill. Sensitivity only, not latency/throughput or current full-run replay.',
          'q2_expert_payload_savings_bytes': 1416683520,
          'two_additional_packed_slots_per_layer_bytes': 2*40*17694720,
          'complete_model_admission_proved': False, 'rows': rows,
          'mlx_imported': any(x=='mlx' or x.startswith('mlx.') for x in sys.modules)}
OUT.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
for row in rows: print(json.dumps({k:v for k,v in row.items() if k!='reads_by_layer'}))
