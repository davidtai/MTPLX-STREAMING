"""Construct a bounded geometry/ownership follow-up after a measured win."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
REPO=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
base=ROOT.parent
proof=json.loads((base/'installation.json').read_text())
result=json.loads((base/'probe.json').read_text())
assert result['complete'] and result['candidate_over_control'] < 1-result['control_spread_fraction']
assert all(a['all_outputs_exact'] for a in result['arms'])
assert not (ROOT/'probe.json').exists()
for name in ('packed_storage.py','library_identity.py','wait_and_run.py','run_screen.py'):
    shutil.copyfile(base/name,ROOT/name)
source=Path('/tmp/dsv41-hybrid-lookup-20260918/full-v1/compat/bank_growth_final.py')
expected=json.loads((source.parent.parent/'native/probe-final-results.json').read_text())['helper_sha256']
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
shutil.copyfile(source,ROOT/source.name)
proof={k:proof[k] for k in ('source_commit','model_path','strict_allocator')}
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==proof['source_commit']
proof.update(static_incremental_bound_bytes=8*1024**3,
    bound_components={'metal_cache_compiler_bytes':4*1024**3,'host_reader_compiler_bytes':4*1024**3,
      'raw_initial_bank_bytes':110*18800640,'final_packed_bank_bytes':111*17694720,
      'copy_overlap_allowance_bytes':112*5898240,'inactive_cache_inside_metal_bytes':256*1024**2,
      'compiler_extra_metal_headroom_bytes':512*1024**2},
    touched_files=[],component_result_sha256=hashlib.sha256((base/'probe.json').read_bytes()).hexdigest(),
    scope='No model weight reads or target execution. Synthetic marker data in the exact native packed bank shapes; existing growth helper unchanged.')
b=proof['bound_components']
assert b['final_packed_bank_bytes']+b['copy_overlap_allowance_bytes']+b['inactive_cache_inside_metal_bytes']+b['compiler_extra_metal_headroom_bytes'] < b['metal_cache_compiler_bytes']
for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
paths=list(ROOT.glob('*.py'))+[REPO/p for p in ('mtplx/models/expert_mlx.py','mtplx/expert_manifest.py','mtplx/deepseek_v41_memory_profile.py')]
paths += [Path(proof['model_path'])/'expert-manifest.json',Path(proof['strict_allocator']['path'])]
proof['sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(ROOT/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
argv=['env','GPU_WINDOW_CHILD_RSS_CAP_BYTES='+str(8*1024**3),'GPU_WINDOW_LOCK_TIMEOUT=600',
      'GPU_WINDOW_TOTAL_MEM_CEILING_BYTES=110000000000','GPU_WINDOW_MIN_AVAIL_GB=8',
      'GPU_WINDOW_RESTORE_QWEN_ALWAYS=1','PYTHONHASHSEED=0','PYTHONUNBUFFERED=1',
      'PYTHONPATH='+str(REPO)+':'+str(ROOT),'scripts/deepseek_v41/gpu_window.sh',
      '/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python',str(ROOT/'run_screen.py')]
(ROOT/'command.sh').write_text(shlex.join(argv)+' > '+shlex.quote(str(ROOT/'guard.log'))+' 2>&1\n')
print(json.dumps({'source_commit':proof['source_commit'],'incremental_bound_bytes':proof['static_incremental_bound_bytes'],'cpu_only':True}))
