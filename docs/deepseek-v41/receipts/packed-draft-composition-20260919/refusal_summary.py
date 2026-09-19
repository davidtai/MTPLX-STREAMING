"""Record live refusal and a CPU-only physical-budget counterfactual."""
import importlib.abc
import importlib.util
import json
from pathlib import Path
import re
import sys


class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('MLX forbidden in refusal reconstruction')


sys.meta_path.insert(0, NoMLX())
root = Path(__file__).resolve().parent
r = root / 'full-v1'
log_path = Path('/tmp/dsv41-110-stage/full-packed-draft-20260919-v1.guard.log')
log = log_path.read_text()
baseline = int(re.search(r'exported exact (\d+)-byte baseline', log).group(1))
assert 'packed draft comparison requires at least111 target rows' in log
assert 'GPU step exited with code 1' in log
assert 'model identity ["mtplx-flash-next-optimized-speed"] verified, background warmup ready' in log
assert 'released exclusive GPU lock' in log
packed = json.loads((r / 'packed/installation.json').read_text())
compat = json.loads((r / 'compat/installation.json').read_text())
spec = importlib.util.spec_from_file_location('packed_admission', r / 'packed/packed_admission.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Zero wired memory removes that constraint entirely. Even this favorable
# counterfactual cannot fit111 rows under the physical ceiling at the live
# baseline; do not present it as the unrecorded live wired snapshot.
a = module.resolve_admission(baseline, 0, grow=True,
    expected_receipt_hash=compat['phase_memory_control_sha256'],
    strict_allocator=packed['strict_allocator']['identity'])
extra_rows = 111 - a['decode_slots_per_layer']
steady111 = a['steady_decode_active_bound_bytes'] + extra_rows * 40 * module.WEIGHTS
append111 = a['overflow_append_active_bound_bytes'] + extra_rows * (40 * module.WEIGHTS + module.RAW - module.WEIGHTS)
active111 = max(steady111, append111, a['resize_active_bound_bytes'],
    a['seed_active_bound_bytes'], a['early_prime_active_bound_bytes'])
physical111 = max(a['prefill_physical_bound_bytes'], baseline + a['host_reserve_bytes'] + active111 + a['decode_cache_allowance_bytes'])
assert physical111 > 110000000000
prefix = log_path.with_suffix('').with_suffix('')
assert not prefix.with_suffix('.bounds.json').exists()
assert not prefix.with_suffix('.jsonl').exists()
result = {'complete': True, 'source_commit': packed['source_commit'],
    'live_baseline_bytes': baseline, 'guard_exit_code': 1,
    'model_load_started': False, 'prefill_started': False, 'decode_started': False,
    'live_bounds_file_emitted': False, 'physical_ceiling_bytes': 110000000000,
    'counterfactual_wired_bytes': 0, 'maximum_rows_with_wired_constraint_removed': a['decode_slots_per_layer'],
    'counterfactual_111_row_physical_bound_bytes': physical111,
    'counterfactual_111_row_over_budget_bytes': physical111 - 110000000000,
    'counterfactual_scope': 'CPU physical accounting at the live baseline, with no wired constraint. Not a live admission receipt or GPU measurement.',
    'exact_qwen_identity_health_and_warmup_restored': True,
    'guard_lock_released_at': '2026-09-19T17:04:14Z',
    'independent_health': json.loads((root / 'post-run-health.json').read_text()),
    'unchanged_retry': False, 'new_full_tps': None}
(root / 'refusal-summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k != 'independent_health'}))
