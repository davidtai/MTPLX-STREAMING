"""Pin one hybrid control and two predefined draft-only alias screens."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from pruning import choose, TRAIN_CYCLES

ROOT = Path(__file__).resolve().parent
BASE = Path('/tmp/dsv41-hybrid-lookup-20260918')
REPO = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
prior = json.loads((BASE / 'installation.json').read_text())
for path, expected in prior['runtime_source_sha256'].items():
    if sha(Path(path)) != expected:
        raise RuntimeError('runtime source changed since proven draft replay')
for name in ('head_screen.py', 'run_head.py', 'lookup.py', 'library_identity.py'):
    if sha(BASE / name) != prior['helper_sha256'][name]:
        raise RuntimeError('prior measured helper changed')
for name in ('run_head.py', 'lookup.py', 'library_identity.py'):
    shutil.copyfile(BASE / name, ROOT / name)

route_path = REPO / 'docs/deepseek-v41/receipts/mtp-route-capture-20260913/mtp-routes-16k-1024.json'
if sha(route_path) != '6a90006c9c4829dac2a548298c1d237bfd1beb793c0bc6eaf99315fad8c9c3dd':
    raise RuntimeError('training route identity changed')
routes = json.loads(route_path.read_text())['draft_routes_by_cycle_stage_row']
compact_path = Path('/tmp/dsv41-compact-residents/receipt.json')
selected = json.loads(compact_path.read_text())['selected_experts_by_stage']
selection = choose(routes, selected)
(ROOT / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
prior_result = json.loads((BASE / 'head-screen.json').read_text())
control = next(a for a in prior_result['arms'] if a['mode'] == 'hybrid_m8')
if not prior_result['complete'] or control['cycles'] != 198:
    raise RuntimeError('prior hybrid control is incomplete')
(ROOT / 'control-commit-lengths.json').write_text(json.dumps(control['commit_lengths']) + '\n')

s = (BASE / 'head_screen.py').read_text()
s = s.replace(str(BASE), str(ROOT)).replace('/private/tmp/dsv41-hybrid-lookup-20260918', str(ROOT))
s = s.replace("for key,value in reference['arm_env'].items():", """for name, key in (('selection.json','selection_sha256'), ('control-commit-lengths.json','control_commit_lengths_sha256')):
    if hashlib.sha256((CURRENT_ROOT/name).read_bytes()).hexdigest() != installation[key]:
        raise RuntimeError('screen configuration identity changed')
if hashlib.sha256(Path('/tmp/dsv41-compact-residents/receipt.json').read_bytes()).hexdigest() != installation['compact_inventory_sha256']:
    raise RuntimeError('compact inventory identity changed')
for key,value in reference['arm_env'].items():""")
needle = "for mode in ('native','hybrid_m8'):"
if s.count(needle) != 1:
    raise RuntimeError('native replay boundary changed')
setup = '''from pruning import alias_slots
selection = json.loads((CURRENT_ROOT/'selection.json').read_text())
control_lengths = json.loads((CURRENT_ROOT/'control-commit-lengths.json').read_text())
original_luts = LUTS
candidate_luts = {}
for name, info in selection.items():
    mapped = []
    aliases = []
    for stage in range(3):
        gate = owner.mtp.layers[stage].mlp.gate.weight.astype(mx.float32)
        mx.eval(gate)
        slots, experts = alias_slots(np.asarray(gate), info['selected_experts_by_stage'][stage], selected[stage])
        mapped.append(mx.array(slots, dtype=mx.int32))
        aliases.append(experts)
    candidate_luts[name] = mapped
    info['alias_experts_by_stage'] = aliases
del gate
mx.eval(candidate_luts)
report['draft_pruning'] = selection
report['physically_compacted'] = False
report['target_execution'] = False
report['training_cycles'] = installation['training_cycles']
report['heldout_start_position'] = sum(teacher['commit_lengths'][:installation['training_cycles']])
for mode in ('hybrid_m8', 'one_band', 'two_bands'):
    LUTS = original_luts if mode == 'hybrid_m8' else candidate_luts[mode]
'''
s = s.replace(needle, setup.rstrip())
s = s.replace("""        if mode=='hybrid_m8':
            lookup.append_committed(ids[history_count:pos+1])
            history_count=pos+1
            proposed=lookup.extend(native_proposed)""", """        lookup.append_committed(ids[history_count:pos+1])
        history_count=pos+1
        proposed=lookup.extend(native_proposed)""")
start = s.index("    if mode=='native':")
end = s.index("    row={'mode':mode", start)
s = s[:start] + '''    if mode=='hybrid_m8' and commit_lengths != control_lengths:
        raise RuntimeError('hybrid control does not reproduce the pinned 198 boundaries')
''' + s[end:]
s = s.replace("    report['arms'].append(row)", "    row['heldout_cycles_starting_after_split'] = sum(r['position'] >= report['heldout_start_position'] for r in rows)\n    report['arms'].append(row)")
s = s.replace("print('HYBRID_HEAD_ARM'", "print('DRAFT_SURROGATE_ARM'")
(ROOT / 'head_screen.py').write_text(s)

proof = dict(prior)
proof.update(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
    scope='Head-only draft router-nearest aliases trained on the first 103 native cycles. Hybrid control plus two fixed candidate sizes. Target states score proposals only; no target execution, physical compaction, or TPS claim.',
    training_cycles=TRAIN_CYCLES,
    training_route_sha256=sha(route_path), compact_inventory_sha256=sha(compact_path),
    prior_head_script_sha256=sha(BASE/'head_screen.py'),
    prior_head_result_sha256=sha(BASE/'head-screen.json'),
    control_commit_lengths_sha256=sha(ROOT/'control-commit-lengths.json'),
    selection_sha256=sha(ROOT/'selection.json'),
    selection_rule='Per-stage assignment frequency from cycles [0,103), ascending expert ID tie; sorted retained weights. Removed experts alias nearest retained router vector; target acceptance stays native.',
    physically_compacted=False)
proof['bound_components']['router_similarity_host_bytes_inside_host_reserve'] = 32 * 1024**2
proof['helper_sha256'] = {p.name:sha(p) for p in ROOT.glob('*.py')}
for p in ROOT.glob('*.py'):
    ast.parse(p.read_text())
(ROOT/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
command = shlex.join(['env','GPU_WINDOW_LOCK_TIMEOUT=120','GPU_WINDOW_TOTAL_MEM_CEILING_BYTES=110000000000',
    f'GPU_WINDOW_CHILD_RSS_CAP_BYTES={49*1024**3}','GPU_WINDOW_MIN_AVAIL_GB=52','GPU_WINDOW_RESTORE_QWEN_ALWAYS=1',
    'GPU_WINDOW_CANDIDATE_MODEL_DIR='+proof['model_path'],'PYTHONHASHSEED=0','PYTHONUNBUFFERED=1',
    f'PYTHONPATH={REPO}:{ROOT}','scripts/deepseek_v41/gpu_window.sh',
    '/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python',str(ROOT/'run_head.py')])
(ROOT/'command.sh').write_text(command+' > '+shlex.quote(str(ROOT/'head.guard.log'))+' 2>&1\n')
print(json.dumps({'source':proof['source_commit'],'bound':proof['static_incremental_bound_bytes'],
                  'candidates':{k:{p:v for p,v in x.items() if p!='selected_experts_by_stage'} for k,x in selection.items()}}))
