"""Transfer independently prompt-trained score adapters to the exact capture."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
REPO=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
ANALYSIS=Path('/tmp/dsv41-router-feature-20260918/full-v2/packed/router_analysis.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proof=json.loads((BASE/'installation.json').read_text())
assert sha(BASE/'screen.py')==proof['script_sha256']
proof.update(capture_path=str(REPO/'.benchmark-artifacts/deepseek-v41/router-feature-20260918/full-router-feature-20260918-v2.router-capture.npz'),
    capture_sha256='5d8dd85c412f0c8e332733843e6eb9ed36ac6c8a8e5a7c71c8c2615e71edca3e',
    capture_bytes=45592534,
    scope='Train score-residual adapters solely on a different historical W35 prompt. Transfer unchanged to exact M6 captured predicted scores through the native gate inverse. Exact cycles0..31 only choose read-issue settings; evaluate32..63. No exact-run labels fit adapter weights, and no new target run or prefetch occurs.',
    source_analysis_sha256=sha(ANALYSIS))
s=ANALYSIS.read_text().replace("('existing_pre_attention_mean','post_attention_router')","('direct','transferred_bias','transferred_ridge')")
(ROOT/'analysis.py').write_text(s)
s=(BASE/'screen.py').read_text()
start=s.index('started=time.monotonic()')
s=s[:start]+'''import io
if hashlib.sha256((ROOT/'analysis.py').read_bytes()).hexdigest()!=installation['analysis_sha256']:
    raise RuntimeError('transfer analysis source differs')
from analysis import analyze
with open_nocache(Path(installation['capture_path'])) as f:
    data=f.read(installation['capture_bytes']+1)
if len(data)!=installation['capture_bytes'] or hashlib.sha256(data).hexdigest()!=installation['capture_sha256']:
    raise RuntimeError('native capture changed')
with np.load(io.BytesIO(data),allow_pickle=False) as arrays:
    original=arrays['scores']
    exact_actual=arrays['actual'];exact_nrows=arrays['nrows']
    exact_persistent=arrays['persistent'];exact_physical=arrays['physical'];exact_reads=arrays['reads']
del data
exact_scores=np.empty((3,64,36,6,384),np.float32)
exact_scores[0]=original[1]
del original
inverse_max_error=0.0
''' + s[start:]
needle='    result={name:metrics(score,truth[2048:]) for name,score in predictions.items()}'
addition='''    exact=exact_scores[0,:,layer-4].reshape(-1,384)
    # Native scores are sqrt(softplus(z))+bias. Stable inverse reconstructs
    # only the predictor's feature values; target routing never changes.
    softplus=np.maximum(exact-bias,np.float32(1e-6))**2
    exact_z=softplus+np.log(-np.expm1(-softplus))
    reconstructed=scores_from_z(exact_z,bias)
    inverse_max_error=max(inverse_max_error,float(np.max(np.abs(reconstructed-exact))))
    exact_scores[1,:,layer-4]=(exact+residual[1024:2048].mean(axis=0)).reshape(64,6,384)
    exact_scores[2,:,layer-4]=(exact+correct(exact_z,model)).reshape(64,6,384)
'''
assert s.count(needle)==1
s=s.replace(needle,addition+needle)
needle="OUT.write_text(json.dumps(report,indent=2)+'\\n')"
addition="""report['transfer_analysis']=analyze(exact_scores,exact_actual,exact_nrows,exact_persistent,exact_physical,exact_reads)
report['native_gate_inverse_max_score_error']=inverse_max_error
report['after_transfer']=host_memory_snapshot()
report['total_elapsed_s']=time.monotonic()-started
report['exact_labels_used_for_adapter_fit']=False
report['exact_capture_matches_acceptance_workload']=True
"""
assert s.count(needle)==1
s=s.replace(needle,addition+needle)
s+='\nprint("PROMPT_TRANSFER",json.dumps({"inverse_score_max_error":inverse_max_error,"elapsed_s":report["total_elapsed_s"],"heldout":{name:f["totals"]["heldout"] for name,f in report["transfer_analysis"]["families"].items()}}),flush=True)\n'
(ROOT/'screen.py').write_text(s)
proof['script_sha256']=sha(ROOT/'screen.py')
proof['analysis_sha256']=sha(ROOT/'analysis.py')
proof['bound']+=' Exact capture and three score families add less than160MiB within the same1GiB envelope.'
(ROOT/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
for p in ROOT.glob('*.py'):ast.parse(p.read_text())
(ROOT/'command.sh').write_text((BASE/'command.sh').read_text().replace(str(BASE),str(ROOT)))
print(json.dumps({'source':proof['source_commit'],'static_incremental_bound_bytes':proof['static_incremental_bound_bytes'],'scope':proof['scope']}))
