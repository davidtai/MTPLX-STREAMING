"""Build a CPU-only optimistic feature screen, before a live partial-MoE capture."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
BASE = Path('/tmp/dsv41-router-feature-20260918/screen.py')
sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s = BASE.read_text()
s = s.replace('ROOT = Path(__file__).resolve().parent', "if os.environ.get('_GPU_WINDOW_LOCKED') != '1':\n    raise RuntimeError('parent-held guard required for the CPU memory envelope')\nROOT = Path(__file__).resolve().parent")
s = s.replace("temp = float(cfg.get('gate_temp', 1.0) or 1.0)", "temp = float(cfg.get('gate_temp', 1.0) or 1.0)\neps = float(cfg.get('rms_norm_eps', 1e-6))")
old = '''    features = {'self_alignment': router,
                'existing_pre_attention_mean': prev_layer,
                'post_attention_router': prev_router,
                'post_attention_norm_adjusted': prev_router * (norm / prev_norm)}'''
new = '''    # The same-layer mean already contains the preceding layer's complete
    # expert output. It has NO early-I/O lead time and is not deployable as a
    # prefetch feature. Screen its predictive quality before considering an
    # earlier approximation made from cached expert contributions.
    rms_mean = layer_in / np.sqrt(np.mean(layer_in*layer_in,axis=-1,keepdims=True)+eps)
    features = {'self_alignment': router,
                'post_attention_router': prev_router,
                'completed_source_mean_raw': layer_in,
                'completed_source_mean_rms': rms_mean * norm}'''
assert s.count(old) == 1
s = s.replace(old,new)
s = s.replace("result = {name: metrics(biased_scores(x, weight, bias), true)",
              "result = {name: metrics(biased_scores(x[128:], weight, bias), true[128:])")
s = s.replace("'scope':'256 historical AR rows, target layers4..39; candidate screens only'", "'scope':'Last128 of256 historical AR rows, target layers4..39. Completed predecessor means have no early-read lead time; this is an optimistic feature-quality screen, not a mathematical bound, deployable prefetch, or TPS result.'")
head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
s = s.replace("'source_commit':'013ba48db733659d287d40e7304e1683a0d7e179'",f"'source_commit':{head!r}")
s = s.replace("'elapsed_s':time.monotonic()-start,'native_self_alignment':alignment", "'elapsed_s':time.monotonic()-start,'selected_rows':[128,256],'no_new_expert_or_target_execution':True,'native_self_alignment':alignment")
ast.parse(s)
(ROOT/'screen.py').write_text(s)
proof = {'source_commit':head,'base_cpu_script_sha256':sha(BASE),
    'script_sha256':sha(ROOT/'screen.py'),'static_incremental_bound_bytes':512*1024**2,
    'scope':'Model-native RMS normalization of completed-source mean; no actual partial-expert compute or GPU operations.',
    'source_manifest_sha256':sha(REPO/'.benchmark-artifacts/deepseek-v41/route-traces-w35/manifest.json'),
    'exact_acceptance_prompt':False,'new_regression_tests':False}
(ROOT/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
command = shlex.join(['env','GPU_WINDOW_CHILD_RSS_CAP_BYTES=1073741824','GPU_WINDOW_LOCK_TIMEOUT=120',
    'GPU_WINDOW_TOTAL_MEM_CEILING_BYTES=110000000000','GPU_WINDOW_MIN_AVAIL_GB=1','GPU_WINDOW_RESTORE_QWEN_ALWAYS=1',
    'GPU_WINDOW_CANDIDATE_MODEL_DIR=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4',
    'PYTHONHASHSEED=0','PYTHONUNBUFFERED=1',f'PYTHONPATH={REPO}',
    'scripts/deepseek_v41/gpu_window.sh','/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python',str(ROOT/'screen.py')])
(ROOT/'command.sh').write_text(command+' > '+shlex.quote(str(ROOT/'guard.log'))+' 2>&1\n')
print(json.dumps(proof))
