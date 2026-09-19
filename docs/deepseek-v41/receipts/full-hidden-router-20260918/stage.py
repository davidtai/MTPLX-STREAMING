"""Stage one CPU-only full-hidden causal predictor quality screen."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
BASE = Path('/tmp/dsv41-prompt-router-adapter-20260918')
REPO = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
prior = json.loads((BASE / 'installation.json').read_text())
assert sha(BASE / 'screen.py') == prior['script_sha256']
source = (BASE / 'screen.py').read_text()
source = source[:source.index('started=time.monotonic()')]
source += (ROOT / 'screen_body.py').read_text()
ast.parse(source)
(ROOT / 'screen.py').write_text(source)
proof = {
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
    'base_cpu_script_sha256': sha(BASE / 'screen.py'),
    'script_sha256': sha(ROOT / 'screen.py'),
    'static_incremental_bound_bytes': 1024**3,
    'source_manifest_sha256': prior['source_manifest_sha256'],
    'runtime_sources': prior['runtime_sources'],
    'training': 'Last1024 captured prefill rows: first768 fit, last256 select lambda1/10 by top6 overlap then score MSE; refit all1024. No decode labels fit or select parameters.',
    'prediction': 'Previous native router input5120 -> native next-gate raw logits plus affine full-hidden residual. Dual ridge avoids a5120-square Gram. Fold into one5120x384 effective prediction matrix and384 raw bias.',
    'scope': 'Historical W35 different16K prompt,256 AR decode rows,layers4..39. Native target gate unchanged. Quality and warm AR-cache proxy only; no real prefetch, Metal or TPS. Existing exact M6 capture lacks full hidden vectors.',
    'bound': '1GiB incremental: sequential layers, two2304x5120 float32 hidden inputs94MiB; raw conversion<=70MiB; gate,coefficients,folded matrices<=48MiB; training,validation,Gram,solve,score arrays<=160MiB; Python/BLAS and allocator reserve<=640MiB. Retain only metrics and hashes, never all fitted matrices.',
    'folded_runtime_parameter_bytes': 36 * (5120 * 384 + 384) * 4,
    'new_regression_tests': False,
    'production_install': False,
}
(ROOT / 'installation.json').write_text(json.dumps(proof, indent=2) + '\n')
command = shlex.join([
    'env', 'GPU_WINDOW_CHILD_RSS_CAP_BYTES=1073741824', 'GPU_WINDOW_LOCK_TIMEOUT=120',
    'GPU_WINDOW_TOTAL_MEM_CEILING_BYTES=110000000000', 'GPU_WINDOW_MIN_AVAIL_GB=1',
    'GPU_WINDOW_RESTORE_QWEN_ALWAYS=1',
    'GPU_WINDOW_CANDIDATE_MODEL_DIR=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4',
    'PYTHONHASHSEED=0', 'PYTHONUNBUFFERED=1', f'PYTHONPATH={REPO}',
    'scripts/deepseek_v41/gpu_window.sh',
    '/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python', str(ROOT / 'screen.py'),
])
(ROOT / 'command.sh').write_text(command + ' > ' + shlex.quote(str(ROOT / 'guard.log')) + ' 2>&1\n')
print(json.dumps(proof))
