"""Stage one bounded mixed-Q2 draft screen with warm repeated trajectories."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
REPO = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
prior = json.loads((BASE/'installation.json').read_text())
for path, expected in prior['runtime_source_sha256'].items():
    assert sha(Path(path)) == expected, path
for name in ('head_screen.py','run_head.py','lookup.py','library_identity.py','requantize.py'):
    assert sha(BASE/name) == prior['helper_sha256'][name], name
for name in ('run_head.py','lookup.py','library_identity.py','requantize.py','control-commit-lengths.json'):
    shutil.copyfile(BASE/name, ROOT/name)
s = (BASE/'head_screen.py').read_text().replace(str(BASE), str(ROOT))
s = s.replace('def __init__(self,width,compact,expert_bits=4):',
              'def __init__(self,width,compact,expert_bits=4,dense_q2=False):')
old = "nn.quantize(self.mtp,group_size=32,bits=8,mode='mxfp8',class_predicate=dv._make_mtp_dense_quant_predicate(32))"
new = '''base_predicate = dv._make_mtp_dense_quant_predicate(32)
        def dense_predicate(path, module):
            if not base_predicate(path, module):
                return False
            if dense_q2 and not (path.endswith('.main_proj') or path.endswith('.attn.wkv')):
                return {'group_size': 64, 'bits': 2, 'mode': 'affine'}
            return True
        nn.quantize(self.mtp,group_size=32,bits=8,mode='mxfp8',class_predicate=dense_predicate)'''
assert s.count(old) == 1
s = s.replace(old, new)
old = "weights['q2'], quantization = convert_experts(weights['compact'], mx)"
s = s.replace(old, old + '''
from dense_quantize import convert_dense
weights['mixed_q2'], dense_quantization = convert_dense(weights['q2'], mx)
report['dense_quantization'] = dense_quantization
report['warm_scope'] = 'Fresh cache per arm; repeated native and Q2 trajectories after each codec first executes. No target trunk or end-to-end TPS.'
print('Q2_DENSE_CONSTRUCTION', json.dumps(dense_quantization), flush=True)''')
s = s.replace("for mode in ('hybrid_control', 'q2_experts'):",
              "for mode in ('hybrid_control_cold', 'q2_experts_cold', 'mixed_q2_cold', 'hybrid_control_warm', 'q2_experts_warm', 'mixed_q2_warm', 'hybrid_control_final'):")
s = s.replace("    key = 'compact' if mode == 'hybrid_control' else 'q2'",
              "    key = 'compact' if mode.startswith('hybrid_control') else ('mixed_q2' if mode.startswith('mixed_q2') else 'q2')")
s = s.replace("owner = DraftOnly(5, compact=True, expert_bits=4 if mode == 'hybrid_control' else 2)",
              "owner = DraftOnly(5, compact=True, expert_bits=4 if key == 'compact' else 2, dense_q2=key == 'mixed_q2')")
s = s.replace("if mode == 'hybrid_control' and commit_lengths != control_lengths:",
              "if key == 'compact' and commit_lengths != control_lengths:")
s = s.replace("report['complete']=True", '''for prefix in ('q2_experts', 'mixed_q2'):
    repeated = [a for a in report['arms'] if a['mode'].startswith(prefix)]
    if repeated[0]['commit_lengths'] != repeated[1]['commit_lengths']:
        raise RuntimeError('repeated draft trajectory is not deterministic')
report['complete']=True''')
ast.parse(s)
(ROOT/'head_screen.py').write_text(s)
paths = {}
for node in ast.parse(s).body:
    if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call):
        for target in node.targets:
            if isinstance(target,ast.Name) and target.id in ('OUT','CURRENT_ROOT'):
                paths[target.id] = Path(ast.literal_eval(node.value.args[0]))
assert paths == {'OUT':ROOT/'head-screen.json','CURRENT_ROOT':ROOT}
proof = dict(prior)
proof.update(
    scope='Head-only Q2/64 compact draft experts plus a mixed dense variant: query/output and shared FFN Q2; main projection,KV builders,routers,norms,Markov,confidence,shared token embedding/output head and target remain native. Saved initial cache remains valid. No target execution/TPS.',
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
    parent_head_script_sha256=sha(BASE/'head_screen.py'),
    parent_head_result_sha256=sha(BASE/'head-screen.json'),
    bound_scope=prior['bound_scope']+' Mixed dense arrays add<=256MiB packed storage and<=256MiB one-matrix conversion scratch; combined expert/dense candidates still fit old11GiB stacked expert allowance after unused full native bank retirement. Owner creation/evaluation is sequential; previous native graph/compile allowance retained.',
)
proof['helper_sha256'] = {p.name:sha(p) for p in ROOT.glob('*.py')}
for p in ROOT.glob('*.py'):ast.parse(p.read_text())
(ROOT/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
cmd = shlex.join(['env','GPU_WINDOW_LOCK_TIMEOUT=120','GPU_WINDOW_TOTAL_MEM_CEILING_BYTES=110000000000',
    f'GPU_WINDOW_CHILD_RSS_CAP_BYTES={49*1024**3}','GPU_WINDOW_MIN_AVAIL_GB=52','GPU_WINDOW_RESTORE_QWEN_ALWAYS=1',
    'GPU_WINDOW_CANDIDATE_MODEL_DIR='+proof['model_path'],'PYTHONHASHSEED=0','PYTHONUNBUFFERED=1',
    f'PYTHONPATH={REPO}:{ROOT}','scripts/deepseek_v41/gpu_window.sh',
    '/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python',str(ROOT/'run_head.py')])
(ROOT/'command.sh').write_text(cmd+' > '+shlex.quote(str(ROOT/'head.guard.log'))+' 2>&1\n')
print(json.dumps({k:proof[k] for k in ('source_commit','scope','static_incremental_bound_bytes')}))
