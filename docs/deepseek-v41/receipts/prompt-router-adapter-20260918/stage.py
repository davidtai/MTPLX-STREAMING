"""Build a CPU-only prompt-trained router residual screen."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parent
BASE=Path('/tmp/dsv41-completed-input-feature-20260918')
REPO=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prior=json.loads((BASE/'installation.json').read_text())
assert sha(BASE/'screen.py')==prior['script_sha256']
s=(BASE/'screen.py').read_text()
s=s[:s.index('start = time.monotonic()')]
s=s.replace('512*1024**2','1024**3').replace('512MiB CPU envelope','1GiB CPU envelope')
s=s.replace('        f.seek(2048*stride, 1)\n        raw = f.read(256*stride)\n        assert len(raw) == 256*stride',
            '        raw = f.read(2304*stride)\n        assert len(raw) == 2304*stride')
s=s.replace("+ ':decode256'", "+ ':all2304'").replace('reshape(256, *shape[1:])','reshape(2304, *shape[1:])')
assert 'f.seek(2048*stride' not in s
s+=(ROOT/'screen_body.py').read_text()
(ROOT/'screen.py').write_text(s)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
proof={'source_commit':head,'base_cpu_script_sha256':sha(BASE/'screen.py'),
    'script_sha256':sha(ROOT/'screen.py'),'static_incremental_bound_bytes':1024**3,
    'source_manifest_sha256':sha(REPO/'.benchmark-artifacts/deepseek-v41/route-traces-w35/manifest.json'),
    'runtime_sources':{'mtplx/expert_streaming.py':sha(REPO/'mtplx/expert_streaming.py'),
                       'mtplx/deepseek_v41_memory_profile.py':sha(REPO/'mtplx/deepseek_v41_memory_profile.py')},
    'training':'Last1024 captured prefill rows. First768 fit, last256 choose ridge0.1/1.0 by top6 overlap; refit all1024 before evaluating256 decode rows. No decode labels fit or select parameters.',
    'prediction':'Previous layer native router input -> next gate raw logits -> native biased scores plus prompt-learned affine residual in standardized raw-logit space.',
    'scope':'Different W35 16K prompt, AR rows. Exact self-alignment gate; generic feature and warm causal AR-cache proxy only. No real prefetch, target execution, Metal import or TPS.',
    'bound':'1GiB CPU incremental including two2304x5120 float32 inputs, source byte buffers, gate weights, per-layer scores/design/solve arrays and BLAS workspace. Process layers sequentially; retain only metrics and hashes.',
    'new_regression_tests':False,'production_install':False}
(ROOT/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
for p in ROOT.glob('*.py'):ast.parse(p.read_text())
command=shlex.join(['env','GPU_WINDOW_CHILD_RSS_CAP_BYTES=1073741824','GPU_WINDOW_LOCK_TIMEOUT=120',
    'GPU_WINDOW_TOTAL_MEM_CEILING_BYTES=110000000000','GPU_WINDOW_MIN_AVAIL_GB=1','GPU_WINDOW_RESTORE_QWEN_ALWAYS=1',
    'GPU_WINDOW_CANDIDATE_MODEL_DIR=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4',
    'PYTHONHASHSEED=0','PYTHONUNBUFFERED=1',f'PYTHONPATH={REPO}',
    'scripts/deepseek_v41/gpu_window.sh','/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python',str(ROOT/'screen.py')])
(ROOT/'command.sh').write_text(command+' > '+shlex.quote(str(ROOT/'guard.log'))+' 2>&1\n')
print(json.dumps(proof))
