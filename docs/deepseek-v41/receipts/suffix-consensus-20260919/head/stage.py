"""CPU-only binding of the selected consensus proposal to a bounded head replay."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
BASE = Path('/tmp/dsv41-draft-conditioned-tail-20260918').resolve()
HYBRID = Path('/tmp/dsv41-hybrid-lookup-20260918').resolve()
REPO = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proof = json.loads((BASE/'installation.json').read_text())
for path,expected in proof['runtime_source_sha256'].items():
    assert sha(Path(path)) == expected,path
for name in ('head_screen.py','run_head.py','library_identity.py'):
    assert sha(BASE/name) == proof['helper_sha256'][name],name
for name in ('run_head.py','library_identity.py'):
    shutil.copyfile(BASE/name,ROOT/name)
cpu = json.loads((ROOT.parent/'screen.json').read_text())
assert cpu['complete'] and cpu['control_exact'] and cpu['selected_minimum_suffix']==3
shutil.copyfile(ROOT.parent/'consensus.py',ROOT/'consensus.py')
shutil.copyfile(HYBRID/'lookup.py',ROOT/'lookup.py')
s = (BASE/'head_screen.py').read_text().split('# This body is appended')[0]
s = s.replace(f"OUT=Path({str(BASE/'head-screen.json')!r})",f"OUT=Path({str(ROOT/'head-screen.json')!r})")
s = s.replace(f'CURRENT_ROOT=Path({str(BASE)!r})',f'CURRENT_ROOT=Path({str(ROOT)!r})')
assert f"OUT=Path({str(ROOT/'head-screen.json')!r})" in s
assert f'CURRENT_ROOT=Path({str(ROOT)!r})' in s
s += (ROOT/'screen_body.py').read_text()
(ROOT/'head_screen.py').write_text(s)
proof['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
proof['scope'] = ('Independent hybrid-control and consensus candidate trajectories using the '
    'authenticated M6 teacher states. Five native proposals and original lookup preserved. '
    'Consensus uses committed history only; min suffix3, two unanimous sources, native min '
    'confidence0.9, at most two extra tokens. Fixed maximum target width8; no target execution '
    'or throughput claim. First-half CPU selection is frozen before this replay.')
proof['consensus_config'] = {'min_suffix':3,'min_count':2,'max_extra':2,'minimum_confidence':.9}
proof['cpu_selection_sha256'] = sha(ROOT.parent/'screen.json')
proof['bound_components']['consensus_index_inside_host_reserve_bytes'] = 32*1024**2
proof['bound_scope'] = ('Same49GiB complete head-only bound:41GiB active,4GiB cache,4GiB host. '
    'One D5 owner shares authenticated arrays. A <=17408-token consensus index adds at most '
    'four n-gram entries per history token;32MiB fits inside host reserve. No target bank. '
    'All GPU shapes and arithmetic match the unchanged hybrid control.')
proof['arithmetic_scope'] = 'Native draft and target arithmetic unchanged; only pure Python causal proposal extension differs.'
proof.pop('tail_cases',None)
proof['helper_sha256'] = {p.name:sha(p) for p in ROOT.glob('*.py')}
for path in ROOT.glob('*.py'):
    ast.parse(path.read_text())
(ROOT/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
(ROOT/'command.sh').write_text((BASE/'command.sh').read_text().replace(str(BASE),str(ROOT)))
print(json.dumps({'source':proof['source_commit'],'incremental_bound_bytes':proof['static_incremental_bound_bytes'],
                  'proposal_config':proof['consensus_config']}))
