"""Pin native hybrid control and a bounded affine-Q2 draft-only screen."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
BASE = Path('/tmp/dsv41-hybrid-lookup-20260918')
REPO = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
prior = json.loads((BASE/'installation.json').read_text())
for path, expected in prior['runtime_source_sha256'].items():
    assert sha(Path(path)) == expected, path
for name in ('head_screen.py', 'run_head.py', 'lookup.py', 'library_identity.py'):
    assert sha(BASE/name) == prior['helper_sha256'][name], name
for name in ('run_head.py', 'lookup.py', 'library_identity.py'):
    shutil.copyfile(BASE/name, ROOT/name)
result = json.loads((BASE/'head-screen.json').read_text())
control = next(a for a in result['arms'] if a['mode'] == 'hybrid_m8')
assert result['complete'] and control['cycles'] == 198
(ROOT/'control-commit-lengths.json').write_text(json.dumps(control['commit_lengths'])+'\n')
s = (BASE/'head_screen.py').read_text()
s = s.replace('/private/tmp/dsv41-hybrid-lookup-20260918', str(ROOT)).replace(str(BASE), str(ROOT))
s = s.replace('def __init__(self,width,compact):', 'def __init__(self,width,compact,expert_bits=4):')
old = "nn.quantize(self.mtp,group_size=32,bits=4,mode='mxfp4',class_predicate=dv._make_mtp_expert_quant_predicate(32))"
new = """if expert_bits == 4:
            nn.quantize(self.mtp,group_size=32,bits=4,mode='mxfp4',class_predicate=dv._make_mtp_expert_quant_predicate(32))
        elif expert_bits == 2:
            nn.quantize(self.mtp,group_size=64,bits=2,mode='affine',class_predicate=dv._make_mtp_expert_quant_predicate(64))
        else:
            raise RuntimeError('unpriced draft expert codec')"""
assert s.count(old) == 1
s = s.replace(old, new)
start = s.index('owner = DraftOnly(5, compact=True)')
end = s.index('    caches = ', start)
s = s[:start] + '''# Retire the unused full128 native bank before allocating Q2 compact storage.
del weights['full']
gc.collect(); mx.synchronize(); mx.clear_cache()
from requantize import convert_experts
weights['q2'], quantization = convert_experts(weights['compact'], mx)
report['draft_quantization'] = quantization
report['target_execution'] = False
report['target_weights_changed'] = False
report['new_regression_tests'] = False
report['compact_missing_expert_route'] = 'Existing construction table maps nonresident draft IDs to slot0; retained unchanged in both arms. This is not the full128-expert draft.'
print('Q2_DRAFT_CONSTRUCTION', json.dumps(quantization), flush=True)
control_path = CURRENT_ROOT/'control-commit-lengths.json'
if digest_file(control_path) != installation['control_commit_lengths_sha256']:
    raise RuntimeError('control boundary identity changed')
control_lengths = json.loads(control_path.read_text())
for mode in ('hybrid_control', 'q2_experts'):
    key = 'compact' if mode == 'hybrid_control' else 'q2'
    owner = DraftOnly(5, compact=True, expert_bits=4 if mode == 'hybrid_control' else 2)
    current = dict(tree_flatten(owner.parameters()))
    if set(current) != set(weights[key]):
        raise RuntimeError('draft parameter coverage differs from codec inventory')
    del current
    owner.load_weights(list(weights[key].items()), strict=True)
    if any(value is not weights[key][name] for name,value in tree_flatten(owner.parameters())):
        raise RuntimeError('random parameters survived draft installation')
    mx.eval(owner.parameters())
''' + s[end:]
s = s.replace("""        if mode=='hybrid_m8':
            lookup.append_committed(ids[history_count:pos+1])
            history_count=pos+1
            proposed=lookup.extend(native_proposed)""", """        lookup.append_committed(ids[history_count:pos+1])
        history_count=pos+1
        proposed=lookup.extend(native_proposed)""")
start = s.index("    if mode=='native':")
end = s.index("    row={'mode':mode", start)
s = s[:start] + '''    if mode == 'hybrid_control' and commit_lengths != control_lengths:
        raise RuntimeError('control differs from pinned198-cycle hybrid trajectory')
''' + s[end:]
s = s.replace('    del caches,main_h,out,logits,conf', '    del owner,caches,main_h,out,logits,conf')
s = s.replace("print('HYBRID_HEAD_ARM'", "print('Q2_DRAFT_ARM'")
ast.parse(s)
(ROOT/'head_screen.py').write_text(s)
paths = {}
for node in ast.parse(s).body:
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in ('OUT', 'CURRENT_ROOT'):
                paths[target.id] = Path(ast.literal_eval(node.value.args[0]))
assert paths == {'OUT': ROOT/'head-screen.json', 'CURRENT_ROOT': ROOT}
proof = dict(prior)
proof.update(
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
    scope='Head-only affine2/64 requantization of the183 compact draft experts, native dense draft and target unchanged. Teacher-state independent trajectories; no target execution or TPS.',
    parent_head_script_sha256=sha(BASE/'head_screen.py'),
    parent_head_result_sha256=sha(BASE/'head-screen.json'),
    control_commit_lengths_sha256=sha(ROOT/'control-commit-lengths.json'),
    compact_inventory_sha256=sha(Path('/tmp/dsv41-compact-residents/receipt.json')),
    conversion='Native mx.dequantize toBF16 then mx.quantize affine2/64, one expert projection at a time; no whole-model dequantization or file export.',
    target_weights_changed=False,
    bound_scope='Existing49GiB head bound retained. Retire unused7,219,445,760B full expert bank before adding2,023,833,600B Q2 compact bank. Current+Q2 expert storage5,464,350,720B plus<=512MiB stack-copy/one-projection conversion remains below existing11GiB expert allowance. Same<=18GiB text residents,<=3GiB shard,<=1GiB teacher/layout,<=8GiB native graph/compile allowance and4GiB each cache/host. No target trunk/banks, no output-file cache.',
    new_regression_tests=False,
)
proof['helper_sha256'] = {p.name:sha(p) for p in ROOT.glob('*.py')}
for p in ROOT.glob('*.py'): ast.parse(p.read_text())
(ROOT/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
command = shlex.join([
    'env','GPU_WINDOW_LOCK_TIMEOUT=120','GPU_WINDOW_TOTAL_MEM_CEILING_BYTES=110000000000',
    f'GPU_WINDOW_CHILD_RSS_CAP_BYTES={49*1024**3}','GPU_WINDOW_MIN_AVAIL_GB=52',
    'GPU_WINDOW_RESTORE_QWEN_ALWAYS=1', 'GPU_WINDOW_CANDIDATE_MODEL_DIR='+proof['model_path'],
    'PYTHONHASHSEED=0','PYTHONUNBUFFERED=1',f'PYTHONPATH={REPO}:{ROOT}',
    'scripts/deepseek_v41/gpu_window.sh',
    '/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python',str(ROOT/'run_head.py')])
(ROOT/'command.sh').write_text(command+' > '+shlex.quote(str(ROOT/'head.guard.log'))+' 2>&1\n')
print(json.dumps({k:proof[k] for k in ('source_commit','scope','static_incremental_bound_bytes','bound_scope')}))
