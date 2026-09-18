"""Construct a bounded R2 gate/up fusion screen from authenticated helpers."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
BASE = Path('/tmp/dsv41-packed-reader-pool-20260918').resolve()
OLD_FUSED = Path('/tmp/dsv41-fused-gu-20260917/fused_gu.py')
REPO = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
proof = json.loads((BASE/'installation.json').read_text())
for name, expected in proof['runtime_source_sha256'].items():
    assert sha(REPO/name) == expected, name
for name, expected in proof['helper_sha256'].items():
    assert sha(BASE/name) == expected, name
    shutil.copyfile(BASE/name, ROOT/name)
(ROOT/'artifact').symlink_to((BASE/'artifact').resolve(), target_is_directory=True)

# Preserve the original per-output block/dot/reduction order and BF16 casts.
# R2 gate plus R2 up uses four accumulators, versus eight in the earlier R4
# fused candidate. The input load is shared within each thread; aggregate
# threadgroup count is unchanged from the two native projections combined.
s = OLD_FUSED.read_text().split('\ndef make_dispatch(')[0]
s = s.replace('from paired_kernels import make_projection\n', '')
s = s.replace('from mtplx.models.expert_mlx import _clamped_swiglu\n', '')
assert s.count('threadgroup_position_in_grid.y*8+simdgroup_index_in_threadgroup*4') == 1
s = s.replace('threadgroup_position_in_grid.y*8+simdgroup_index_in_threadgroup*4',
              'threadgroup_position_in_grid.y*4+simdgroup_index_in_threadgroup*2')
assert s.count('r<4') == 3
s = s.replace('r<4', 'r<2')
s = s.replace('float gr[4]={0,0,0,0};', 'float gr[2]={0,0};')
s = s.replace('float ur[4]={0,0,0,0};', 'float ur[2]={0,0};')
s = s.replace('uint gd[4],ud[4];', 'uint gd[2],ud[2];')
s = s.replace('dsv41_packed_gate_up_shared_input', 'dsv41_packed_gate_up_r2_shared_input')
(ROOT/'fused_gu_r2.py').write_text(s)

# Derive the grouping and down operator verbatim from the native type.
s = (ROOT/'plane_lane.py').read_text()
start = s.index('class PackedOps:')
end = s.index('\n\nclass PackedDecode:', start)
s = s[start:end]
s = s.replace('class PackedOps:', 'class FusedPackedOps:')
s = s.replace('self.gu_kernel = make_projection(2304,5120)', 'self.gu_kernel = make_gate_up()')
old = '''            args = dict(template=[('T',mx.bfloat16)],grid=(32,576,rows),threadgroup=(32,2,1),
                        output_shapes=[(rows,1,1,2304)],output_dtypes=[mx.bfloat16])
            g = self.gu_kernel(inputs=[x,pairs,bank.arrays['gate_proj.weight'],*self.gs],**args)[0]
            u = self.gu_kernel(inputs=[x,pairs,bank.arrays['up_proj.weight'],*self.us],**args)[0]'''
new = '''            g,u = self.gu_kernel(
                inputs=[x,pairs,bank.arrays['gate_proj.weight'],*self.gs,bank.arrays['up_proj.weight'],*self.us],
                template=[('T',mx.bfloat16)],grid=(32,1152,rows),threadgroup=(32,2,1),
                output_shapes=[(rows,1,1,2304)]*2,output_dtypes=[mx.bfloat16]*2)'''
assert s.count(old) == 1
s = s.replace(old,new)
(ROOT/'fused_ops.py').write_text('''"""Native ownership/grouping and down projection with construction-bound R2 GU."""
import mlx.core as mx
from mtplx.models.expert_mlx import _clamped_swiglu
from paired_kernels import make_projection
from plane_lane import GateUpWork
from fused_gu_r2 import make_gate_up

''' + s + '\n')

p = ROOT/'probe.py'
s = p.read_text().replace('from concurrent.futures import ThreadPoolExecutor', 'from fused_ops import FusedPackedOps')
start = s.index("        if reader._fanout_executor._max_workers != 15:")
end = s.index('        rng = np.random.default_rng', start)
s = s[:start] + '''        runners = plane_lane.install(runtime,{LAYER:switch},{LAYER:scales},early=True)
        if mode == 'fused-r2':
            runners[LAYER].ops = FusedPackedOps(scales)
''' + s[end:]
s = s.replace("a['mode']=='six-workers'", "a['mode']=='fused-r2'")
s = s.replace('PACKED_READER_POOL', 'FUSED_GU_R2')
s = s.replace('One-layer native route replay for six auxiliary plane-reader workers.',
              'One-layer native route replay for a smaller fused gate/up output tile.')
p.write_text(s)

proof['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
proof['arms'] = ['native','fused-r2','native','fused-r2','native']
proof['comparison'] = 'Native separate R4 gate/up projections versus one R2+R2 fused projection. Original 15 auxiliary reader workers, cache policy, physical reads, input grouping, BF16 boundaries, native activation and down kernel are unchanged.'
proof['bound_scope'] = '9GiB incremental:5GiB Metal/cache/compile plus4GiB host/reader/compiler.158 raw rows, one packed-scale layer,16MiB scratch. The fused projection replaces two outputs with two equally sized outputs, has no extra GPU buffer or threadgroup scratch, and uses the same inputs and owners. Both kernel variants may remain compiled inside the existing compiler allowance.'
proof['allocator_sources']['rule'] = 'Pinned strict allocator unchanged. The plane lane and readers remain byte identical; only the candidate projection type is installed once before replay.'
proof['helper_sha256'] = {p.name:sha(p) for p in ROOT.iterdir() if p.is_file()
    and p.suffix in ('.py','.json') and p.name not in ('installation.json','stage.py')}
(ROOT/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
command = (BASE/'command.sh').read_text().replace(str(BASE),str(ROOT)).replace('/tmp/dsv41-packed-reader-pool-20260918',str(ROOT))
(ROOT/'command.sh').write_text(command)
for p in ROOT.glob('*.py'):
    ast.parse(p.read_text())
assert (ROOT/'plane_lane.py').read_bytes() == (BASE/'plane_lane.py').read_bytes()
audit = {'source_commit':proof['source_commit'],
    'static_incremental_bound_bytes':proof['static_incremental_bound_bytes'],
    'ancestor_r4_fused_sha256':sha(OLD_FUSED),
    'native_plane_sha256':sha(ROOT/'plane_lane.py'),
    'candidate_kernel_sha256':sha(ROOT/'fused_gu_r2.py'),
    'native_tile':{'rows_per_simd_per_projection':4,'simds_per_group':2,'accumulators_per_thread':4,'gu_dispatches':2,'grid_per_dispatch':[32,576,'assignments']},
    'candidate_tile':{'rows_per_simd_per_projection':2,'simds_per_group':2,'accumulators_per_thread':4,'gu_dispatches':1,'grid_per_dispatch':[32,1152,'assignments']},
    'arithmetic':'V16 K5120 STEP512; unchanged scale decoding and scale_dot<16>; ten block accumulations in native order; simd_sum; separate BF16 gate/up stores followed by unchanged clamped_swiglu and native down.',
    'ownership':'Physical bank slots and expert scale IDs remain separate; same grouped assignments and tensors; native part3 GU witness, full-ready publication, pins, failure draining and deferred releases.',
    'extra_array_bytes':0,'extra_threadgroup_scratch_bytes':0,
    'max_gu_outputs_bytes':36*2304*2*2,
    'full_model_promotion':False}
(ROOT/'construction.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit))
