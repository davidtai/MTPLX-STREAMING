"""Follow independent hybrid trajectories for the two-token suffix candidate."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
HYBRID=Path('/tmp/dsv41-hybrid-lookup-20260918').resolve()
REPO=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proof=json.loads((BASE/'installation.json').read_text())
for name,expected in proof['runtime_source_sha256'].items():
    assert sha(Path(name))==expected,name
for name,expected in proof['helper_sha256'].items():
    assert sha(BASE/name)==expected,name
for name in ('tail.py','run_head.py','library_identity.py'):
    shutil.copyfile(BASE/name,ROOT/name)
hp=json.loads((HYBRID/'installation.json').read_text())
assert sha(HYBRID/'lookup.py')==hp['helper_sha256']['lookup.py']
shutil.copyfile(HYBRID/'lookup.py',ROOT/'lookup.py')
s=(BASE/'head_screen.py').read_text()
s=s[:s.index('# This body is appended')]
s=s.replace(f'OUT=Path({str(BASE/"head-screen.json")!r})',f'OUT=Path({str(ROOT/"head-screen.json")!r})')
s=s.replace(f'CURRENT_ROOT=Path({str(BASE)!r})',f'CURRENT_ROOT=Path({str(ROOT)!r})')
assert f'OUT=Path({str(ROOT/"head-screen.json")!r})' in s
s+=(ROOT/'screen_body.py').read_text()
(ROOT/'head_screen.py').write_text(s)
proof['scope']='Independent head-only hybrid boundaries, original D5 with existing causal lookup priority. Conditioned D7 two-token suffix used only when lookup supplies none and native min-confidence>=0.9. Compare raw suffix and threshold0.5 truncation. Future target IDs only score and seed committed states; no target run or TPS claim.'
proof['opportunity_receipt_sha256']=sha(BASE/'head-screen.json')
proof['tail_cases']=[{'name':'hybrid_control','tail':False},
    {'name':'conditioned_raw','tail':True,'tail_confidence_threshold':None},
    {'name':'conditioned_conf05','tail':True,'tail_confidence_threshold':0.5}]
proof['max_proposed_target_width']=8
proof['helper_sha256']={p.name:sha(p) for p in ROOT.glob('*.py')}
for p in ROOT.glob('*.py'):ast.parse(p.read_text())
(ROOT/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
(ROOT/'command.sh').write_text((BASE/'command.sh').read_text().replace(str(BASE),str(ROOT)))
print(json.dumps({'source':proof['source_commit'],'incremental_bound_bytes':proof['static_incremental_bound_bytes'],'max_target_width':8}))
