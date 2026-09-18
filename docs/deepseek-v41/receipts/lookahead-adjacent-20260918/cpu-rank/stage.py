"""Stage CPU ranking of tiny recorded scores, preserving native gate cost."""
from pathlib import Path
import hashlib,json,shutil,subprocess

r=Path(__file__).resolve().parent
b=Path('/tmp/dsv41-lookahead-priority-20260918')
repo=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
names=('probe.py','predictor_cost.py','paired_config.py','plane_lane.py','priority_reads.py','packed_storage.py','paired_kernels.py','kernels.py','restore_bank.py','library_identity.py','run_screen.py','routes.json','preflight.py')
for name in names:shutil.copyfile(b/name,r/name)
(r/'artifact').symlink_to(b/'artifact',target_is_directory=True)
p=r/'predictor_cost.py';s=p.read_text()
begin=s.index('        biased=self.scores[self.call]');end=s.index('        candidates={}',begin)
s=s[:begin]+'''        mx.eval(indices,live)
        biased=np.asarray(self.scores[self.call])
        part=np.argpartition(-biased,kth=5,axis=-1)[...,:6]
        values=np.take_along_axis(biased,part,axis=-1)
        order=np.argsort(-values,axis=-1,kind='stable')
        ranked=np.take_along_axis(part,order,axis=-1)
        values=np.take_along_axis(values,order,axis=-1)
        gaps=values-values[...,-1:]
''' +s[end:]
p.write_text(s)
proof=json.loads((b/'installation.json').read_text());proof['helper_sha256']={}
proof['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
proof['scope']+=' Candidate ranks the 6x384 FP32 score block in NumPy after the existing gate/route barrier, avoiding small GPU selection dispatches.'
(r/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
(r/'command.sh').write_text((b/'command.sh').read_text().replace(str(b.resolve()),str(r)))
for p in r.glob('*.py'):compile(p.read_text(),str(p),'exec')
print(json.dumps({'root':str(r),'source':proof['source_commit'],'new_score_payload_bytes':6*384*4}))
