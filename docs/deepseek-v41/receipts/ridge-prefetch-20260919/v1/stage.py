"""Construct a bounded compact-ridge scheduling experiment without MLX."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import subprocess

r = Path(__file__).resolve().parent
b = Path('/tmp/dsv41-lookahead-cpu-rank-20260918').resolve()
repo = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
reference = Path('/tmp/dsv41-prompt-router-adapter-20260918/transfer/screen.json').resolve()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
old = json.loads((b / 'installation.json').read_text())
for name, digest in old['helper_sha256'].items():
    assert sha(b / name) == digest, name
for name, digest in old['runtime_source_sha256'].items():
    assert sha(repo / name) == digest, name
names = ('probe.py', 'predictor_cost.py', 'paired_config.py', 'plane_lane.py',
         'priority_reads.py', 'packed_storage.py', 'paired_kernels.py', 'kernels.py',
         'restore_bank.py', 'library_identity.py', 'run_screen.py', 'routes.json', 'preflight.py')
for name in names:
    assert not (r / name).exists(), name
    shutil.copyfile(b / name, r / name)
(r / 'artifact').symlink_to(b / 'artifact', target_is_directory=True)
prefix = Path('/tmp/dsv41-prompt-router-adapter-20260918/screen.py').read_text().split('# Appended after')[0]
prefix = prefix.replace("OUT = ROOT / 'screen.json'", "OUT = ROOT / 'preparation.json'")
(r / 'prepare.py').write_text(prefix + (r / 'prepare_body.py').read_text())
p = r / 'predictor_cost.py'
s = p.read_text()
s = s.replace('import mlx.core as mx', 'import mlx.core as mx\nimport mlx.nn as nn\nfrom pathlib import Path')
s = s.replace("    return gates,identities", '''    prepared = json.loads((Path(__file__).resolve().parent/'preparation.json').read_text())
    parameter_path = Path(__file__).resolve().parent/'ridge-parameters.npz'
    if not prepared['complete'] or hashlib.sha256(parameter_path.read_bytes()).hexdigest()!=prepared['parameters_sha256']:
        raise RuntimeError('ridge parameter preparation changed')
    if cfg.get('scoring_func','sqrtsoftplus')!='sqrtsoftplus':
        raise RuntimeError('ridge prefix requires the measured native scoring function')
    temp=float(cfg.get('gate_temp',1.) or 1.)
    def ridge_prefix(x,weight,bias,mean,scale,center,coef):
        z=(x.astype(mx.float32)@weight.astype(mx.float32).T)/temp
        return mx.sqrt(nn.softplus(z))+bias+(((z-mean)/scale)@coef+center)
    compiled=mx.compile(ridge_prefix)
    with np.load(parameter_path,allow_pickle=False) as arrays:
        for layer in (31,32):
            values=[arrays[f'layer{layer}_{name}'] for name in ('mean','scale','center','coef')]
            if [v.shape for v in values]!=[(384,),(384,),(384,),(384,384)] or any(v.dtype!=np.float32 for v in values):
                raise RuntimeError('ridge adapter geometry changed')
            params=tuple(mx.array(v) for v in values)
            weight,bias,_=gates[layer]
            gates[layer]=(weight,bias,compiled,params)
            mx.eval(params)
    identities['ridge_parameters']={'sha256':prepared['parameters_sha256'],'bytes':prepared['parameters_bytes']}
    return gates,identities''')
assert s.count('self.weight,self.bias,self.prefix=gate') == 1
s = s.replace('self.weight,self.bias,self.prefix=gate', 'self.weight,self.bias,self.prefix,self.adapter=gate')
assert s.count('live=self.prefix(tokens,self.weight,self.bias)[1]') == 1
s = s.replace('live=self.prefix(tokens,self.weight,self.bias)[1]',
              'live=self.prefix(tokens,self.weight,self.bias,*self.adapter)')
s = s.replace('Pay real gate and ranking costs', 'Pay real gate, ridge correction and ranking costs')
p.write_text(s)
p = r / 'probe.py'
s = p.read_text()
anchor="data=json.loads((ROOT/'routes.json').read_text())"
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + "\ndata['predictor_config']=proof['ridge_configs']")
old_scores="    scores={l:tuple(mx.array(row,mx.float32) for row in data['captured_scores'][str(l)]) for l in (31,32)}"
new_scores="""    with np.load(ROOT/'ridge-parameters.npz',allow_pickle=False) as parameters:
        scores={l:tuple(mx.array(row,mx.float32) for row in parameters[f'layer{l}_scores']) for l in (31,32)}"""
assert s.count(old_scores) == 1
s = s.replace(old_scores, new_scores)
p.write_text(s)
proof = old
proof['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
proof['arms'] = ['native', 'prefetch', 'native']
proof['scope'] = ('Three adjacent real Q4 expert layers, continuous held-out timing and final read/GPU drain. '
                  'Native gate plus compact ridge correction runs on synthetic BF16 inputs; saved exact-workload '
                  'corrected scores choose prefetch reads. First-GU issue, demand-priority queue, CPU ranking. '
                  'Attention omitted; this is scheduling/cost evidence, not live predictor parity or full TPS.')
proof['ridge_reference_path'] = str(reference)
proof['ridge_reference_sha256'] = sha(reference)
proof['training_manifest_sha256'] = sha(repo / '.benchmark-artifacts/deepseek-v41/route-traces-w35/manifest.json')
prior = json.loads(reference.read_text())
proof['ridge_configs'] = {str(v['layer']): v['config'] for v in prior['transfer_analysis']['families']['transferred_ridge']['selected_layers']
                          if v['layer'] in (31,32)}
assert proof['ridge_configs'] == {'31': {'width':6,'margin':.1,'max_records':12}, '32': {'width':6,'margin':.1,'max_records':4}}
proof['budget_components']['ridge_parameter_payload_bytes'] = 2 * 594432
proof['budget_components']['cpu_preparation_bound_bytes'] = 1024**3
proof['budget_components']['phase_scope'] = 'CPU preparation exits before GPU phase. The additional1.19MB of adapters fits the existing10GiB device envelope;4GiB host reserve retained.'
proof['helper_sha256'] = {}
(r / 'installation.json').write_text(json.dumps(proof, indent=2) + '\n')
p = r / 'preflight.py'
s = p.read_text().replace("'probe.py','priority_reads.py'", "'prepare.py','prepare_body.py','prepare_and_run.py','probe.py','priority_reads.py'")
p.write_text(s)
cmd = (b / 'command.sh').read_text().replace(str(b), str(r))
cmd = cmd.replace('GPU_WINDOW_LOCK_TIMEOUT=120', 'GPU_WINDOW_LOCK_TIMEOUT=600')
cmd = cmd.replace(str(r / 'run_screen.py'), str(r / 'prepare_and_run.py'))
(r / 'command.sh').write_text(cmd)
for p in r.glob('*.py'):
    ast.parse(p.read_text())
print(json.dumps({'root':str(r),'source':proof['source_commit'],'static_incremental_bound_bytes':proof['static_incremental_bound_bytes'],'arms':proof['arms']}))
