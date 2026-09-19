"""CPU-only construction and phase-budget checks before a guarded full run."""
import ast
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import shlex
import sys


class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('MLX forbidden in CPU preflight')


sys.meta_path.insert(0, NoMLX())
root = Path(__file__).resolve().parent
r = root / 'full-v1'
repo = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
base = Path('/tmp/dsv41-extension-bank-20260919/full-v1').resolve()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
pins = 0
for sub in ('packed', 'native', 'compat'):
    installation = json.loads((r / sub / 'installation.json').read_text())
    for name, digest in installation['helper_sha256'].items():
        assert sha(r / sub / name) == digest, (sub, name)
        pins += 1
for path in r.rglob('*.py'):
    compile(path.read_text(), str(path), 'exec')
packed = json.loads((r / 'packed/installation.json').read_text())
compat = json.loads((r / 'compat/installation.json').read_text())
draft = packed['draft_projection']
assert draft['workspace_inventory_bytes'] < draft['additional_gpu_workspace_bytes'] == 128 * 1024**2
assert sha(Path(draft['backend_source'])) == draft['backend_source_sha256']
assert sha(r / 'packed/draft_projection.py') == draft['source_sha256']
assert sha(Path(draft['component_receipt'])) == draft['component_sha256']
for name in ('plane_lane.py', 'paired_kernels.py', 'kernels.py', 'fused_transpose.py', 'hybrid_install.py', 'extension.py'):
    assert sha(r / 'packed' / name) == sha(base / 'packed' / name), name

run = (r / 'packed/run_full.py').read_text()
assert 'tuple(map(len, SELECTED_MTP_EXPERTS)) != (93, 58, 32)' in run
assert run.count('projection_owner_report.update(prime_model(target))') == 1
assert run.index('projection_owner_report.update(prime_model(target))') < run.index('overflow_report = grow_rows(')
assert run.index("if growth_admission['decode_slots_per_layer'] < 111:") < run.index('resident = original_load(*a, **kw)')
assert 'draft_projection_report = {}' in run

# Native seed_only returns before the changed projection, so the new packed
# route cannot see the 2048-row prompt tail. Its measured maximum is T6.
tree = ast.parse((repo / 'mtplx/models/deepseek_v41_dspark.py').read_text())
attention = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DSparkAttention')
call = next(n for n in attention.body if isinstance(n, ast.FunctionDef) and n.name == '__call__')
seed = next(n for n in call.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'seed_only')
assert isinstance(seed.body[-1], ast.Return)
assert seed.lineno < next(n.lineno for n in ast.walk(call) if isinstance(n, ast.Attribute) and n.attr == '_o_lora_dense_weight')

spec = importlib.util.spec_from_file_location('packed_admission', r / 'packed/packed_admission.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from mtplx.expert_runtime import resolve_box_target_mlx_limit_bytes

argv = shlex.split((r / 'command.sh').read_text())
host_arg = argv[argv.index('--host-overhead-gib') + 1]
cases = []
for baseline in (10240868352, 10983129088, 11053318144, 11560878080, 12250005504):
    a = module.resolve_admission(baseline, 3400744960, grow=True,
        expected_receipt_hash=compat['phase_memory_control_sha256'],
        strict_allocator=packed['strict_allocator']['identity'])
    assert a['physical_ceiling_bytes'] == 110000000000
    assert a['physical_bound_bytes'] <= 110000000000
    assert a['host_reserve_bytes'] == 1723985920 == round(float(host_arg) * 1024**3)
    assert a['draft_prefill_seed_and_extension_credit_bytes'] == 0
    assert a['draft_dense_cache_steady_credit_bytes'] == 402653184
    assert a['background_variation_allowance_bytes'] == 256 * 1024**2
    assert a['early_prime_active_bound_bytes'] == a['seed_active_bound_bytes'] + 3 * 67108864
    assert a['retained_projection_during_extension_bytes'] == 67108864
    expected_active = max(a[k] for k in ('steady_decode_active_bound_bytes', 'resize_active_bound_bytes',
        'seed_active_bound_bytes', 'early_prime_active_bound_bytes', 'overflow_append_active_bound_bytes', 'prefill_active_bound_bytes'))
    assert a['active_bound_bytes'] == expected_active
    env = {'MTPLX_ENGRAM_CACHE_LIMIT': '67108864', 'MTPLX_DSV41_BOX_TARGET_GB': '110',
        'MTPLX_DSV41_BOX_BASELINE_GB': format(baseline / 1e9, '.17g'),
        'MTPLX_DSV41_MLX_CACHE_LIMIT_GIB': '1', 'MTPLX_DSV41_HOST_OVERHEAD_GIB': host_arg,
        'MTPLX_DSV41_TRANSIENT_BAND_GIB': format((a['allocator_limit_bytes'] - 83426614088) / 1024**3, '.17g')}
    resolved = resolve_box_target_mlx_limit_bytes(env)
    assert resolved['engine_budget_bytes'] == 83426614088
    assert resolved['mlx_limit_bytes'] == a['allocator_limit_bytes']
    cases.append({'baseline_bytes': baseline, 'admission': a,
        'meets_111_row_launch_gate': a['decode_slots_per_layer'] >= 111, 'cli_budget': resolved})

assert not any(name == 'mlx' or name.startswith('mlx.') for name in sys.modules)
result = {'complete': True, 'cpu_only': True, 'mlx_imported': False,
    'source_pin_count': pins, 'preflight_sha256': sha(Path(__file__)),
    'target_kernels_unchanged': True, 'native_draft_experts': [93, 58, 32],
    'native_seed_returns_before_output_projection': True, 'cases': cases}
(root / 'preflight.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'complete': True, 'cpu_only': True, 'pins': pins,
    'cases': [{k: row[k] for k in ('baseline_bytes', 'meets_111_row_launch_gate')} | {
        'rows': row['admission']['decode_slots_per_layer'],
        'physical_bound_bytes': row['admission']['physical_bound_bytes']} for row in cases]}))
