"""Use the existing exact-workload labels for a separate online margin adapter."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parent
REPO=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
BASE=Path('/tmp/dsv41-router-feature-20260918/full-v2/packed/router_analysis.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=BASE.read_text()
s=s.replace("('existing_pre_attention_mean','post_attention_router')","('direct','online_bias','online_ridge')")
s=s.replace("('train',slice(0,32))","('train',slice(16,32))")
s=s.replace('First32 cycles choose each layer configuration',
    'Cycles0..15 fit only the adapters; cycles16..31 choose each layer configuration')
(ROOT/'analysis.py').write_text(s)
proof={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
    'capture_path':str(REPO/'.benchmark-artifacts/deepseek-v41/router-feature-20260918/full-router-feature-20260918-v2.router-capture.npz'),
    'capture_sha256':'5d8dd85c412f0c8e332733843e6eb9ed36ac6c8a8e5a7c71c8c2615e71edca3e',
    'capture_bytes':45592534,'static_incremental_bound_bytes':1024**3,
    'source_analysis_sha256':sha(BASE),'script_sha256':sha(ROOT/'screen.py'),
    'analysis_sha256':sha(ROOT/'analysis.py'),
    'training_cycles':[0,16],'calibration_cycles':[16,32],'heldout_cycles':[32,64],
    'ridge':1.0,'ranking_margin':0.025,
    'scope':'Exact16K/1K native M6 capture, first64 cycles at105 slots. No true gate logits were stored, so this is a separate label-trained minimum-margin residual adapter, not a transfer or validation of the W35 score-regression model. No target execution, reads, predictor timing or TPS.',
    'construction':'Fit per-layer affine residual from previous-layer predicted scores to the minimal score changes needed to put observed top6 labels above a0.025 margin. Fit only first16 past cycles. All admission settings use the following16 cycles and freeze before heldout32.'}
(ROOT/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
for p in ROOT.glob('*.py'):ast.parse(p.read_text())
command=shlex.join(['env','GPU_WINDOW_CHILD_RSS_CAP_BYTES=1073741824','GPU_WINDOW_LOCK_TIMEOUT=120',
    'GPU_WINDOW_TOTAL_MEM_CEILING_BYTES=110000000000','GPU_WINDOW_MIN_AVAIL_GB=1','GPU_WINDOW_RESTORE_QWEN_ALWAYS=1',
    'PYTHONHASHSEED=0','PYTHONUNBUFFERED=1',f'PYTHONPATH={REPO}',
    'scripts/deepseek_v41/gpu_window.sh','/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python',str(ROOT/'screen.py')])
(ROOT/'command.sh').write_text(command+' > '+shlex.quote(str(ROOT/'guard.log'))+' 2>&1\n')
print(json.dumps(proof))
