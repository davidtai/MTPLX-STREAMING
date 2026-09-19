"""Stage a bounded native-gather/indirect-input/native-gather expert comparison."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parent
base = Path('/tmp/dsv41-extension-bank-20260919').resolve()
repo = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
proof = json.loads((base/'installation.json').read_text())
assert not (root/'installation.json').exists()
for name, digest in proof['sha256'].items():
    assert sha(Path(name)) == digest, name
head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert head == 'b5f41c51d74a770069048dddd0a644f07886a999'
for name in ('plane_lane.py','packed_storage.py','paired_kernels.py','kernels.py',
             'restore_bank.py','library_identity.py','routes.json','fused_transpose.py',
             'attention_reader.py','output_store.py','wait_and_run.py','run_screen.py',
             'extension.py','overflow.py','bank_growth_final.py'):
    shutil.copy2(base/name,root/name)
(root/'artifact').symlink_to((base/'artifact').resolve(),target_is_directory=True)

kernel = (base/'paired_kernels.py').read_text()
edits = [
    ('def make_projection(n, k):','def make_indexed_projection(n, k):'),
    ('if (n, k) not in ((2304, 5120), (5120, 2304)):', 'if (n, k) != (2304, 5120):'),
    ('const device T* xp=x+assignment*K+lane*V;',
     'const device T* xp=x+size_t(token_rows[assignment])*K+lane*V;'),
    ("name=f'dsv41_packed_scales_{n}_{k}_paired'", "name=f'dsv41_packed_scales_{n}_{k}_indexed_input'"),
    ("input_names=['x', 'pairs', 'weights',", "input_names=['x', 'pairs', 'token_rows', 'weights',"),
]
for old,new in edits:
    assert kernel.count(old)==1,old
    kernel=kernel.replace(old,new)
restored=kernel
for old,new in reversed(edits):restored=restored.replace(new,old)
assert restored==(base/'paired_kernels.py').read_text()
(root/'indexed_kernel.py').write_text(kernel)

source=(base/'plane_lane.py').read_text()
cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='PackedOps')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='gate_up')
body='\n'.join(source.splitlines()[method.lineno-1:method.end_lineno])+'\n'
old='x = mx.take(tokens,mx.array([p//6 for p in positions],mx.int32),axis=0).reshape(rows,1,1,5120)'
new='token_rows = mx.array([p//6 for p in positions],mx.int32)'
assert body.count(old)==1
body=body.replace(old,new).replace('inputs=[x,pairs,','inputs=[tokens,pairs,token_rows,')
assert body.count('inputs=[tokens,pairs,token_rows,')==2
ops='''"""Direct token-row addressing; original grouping, dot order and activation."""
import mlx.core as mx
from mtplx.models.expert_mlx import _clamped_swiglu
from plane_lane import PackedOps, GateUpWork
from indexed_kernel import make_indexed_projection

class IndexedOps(PackedOps):
    def __init__(self, scales):
        super().__init__(scales)
        self.gu_kernel = make_indexed_projection(2304,5120)

'''+body
(root/'indexed_ops.py').write_text(ops)

s=(base/'probe.py').read_text()
s=s.replace('import plane_lane\n','import plane_lane\nfrom indexed_ops import IndexedOps\nNATIVE_OPS = plane_lane.PackedOps\n')
anchor='        runners = plane_lane.install(runtime,{LAYER:switch},{LAYER:scales},early=True)'
assert s.count(anchor)==1
s=s.replace(anchor,"        plane_lane.PackedOps = IndexedOps if mode == 'indexed-input' else NATIVE_OPS\n"+anchor)
start=s.index("        if mode=='resize109-plus1':")
end=s.index('        allocation_ns=',start)
s=s[:start]+"        phases=[grow_rows(runtime,capacity=110,layout='extension',mx=mx)]\n"+s[end:]
s=s.replace("if result['existing_backings_unchanged'] != (mode=='extension84-plus26'):","if not result['existing_backings_unchanged']:")
s=s.replace("extra_start=109 if mode=='resize109-plus1' else 84",'extra_start=84')
s=s.replace("a['charged_warm_wall_ns']", "a['warm_wall_ns']")
s=s.replace("if a['mode']=='resize109-plus1'", "if a['mode']=='native-gather'")
s=s.replace("if a['mode']=='extension84-plus26'", "if a['mode']=='indexed-input'")
s=s.replace('PREDICTABLE_EXPANSION_', 'INDEXED_INPUT_')
(root/'probe.py').write_text(s)
(root/'command.sh').write_text((base/'command.sh').read_text().replace(str(base),str(root)))
proof.update(source_commit=head,arms=['native-gather','indexed-input','native-gather'],
    scope='Equal110 rows, original84 plus26 extension. Replace only gate/up token materialization with direct row addressing in unchanged native packed dot loops. Same weights, scales, BF16 casts, activation, down projection, read schedule and projection expansion. Require all206 outputs exact; primary metric warmed continuous replay, allocations separately retained.',
    predecessor={'component_result_sha256':sha(base/'probe.json'),'kernel_source_sha256':sha(base/'paired_kernels.py')})
proof['sha256']={p:h for p,h in proof['sha256'].items() if not p.startswith(str(base)+'/')}
for p in root.iterdir():
    if p.is_file() and p.name!='installation.json':
        if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
        proof['sha256'][str(p)]=sha(p)
(root/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
(root/'cpu-preflight.json').write_text(json.dumps({'cpu_only':True,'source_commit':head,
    'kernel_changes_invert_to_native':True,'dot_loop_and_reduction_unchanged':True,
    'static_incremental_bound_bytes':proof['static_incremental_bound_bytes'],
    'new_live_data':'one int32 token row per assignment; removes copied activation rows',
    'max_assignment_rows':48,'scope':proof['scope']},indent=2)+'\n')
print(json.dumps({'source_commit':head,'static_incremental_bound_bytes':proof['static_incremental_bound_bytes'],'scope':proof['scope']}))
