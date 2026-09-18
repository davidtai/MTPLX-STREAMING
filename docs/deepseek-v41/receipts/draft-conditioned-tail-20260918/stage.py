"""Bound a creative suffix proposal screen without modifying the verifier."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
BASE = Path('/tmp/dsv41-hybrid-lookup-20260918').resolve()
REPO = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
proof = json.loads((BASE/'installation.json').read_text())
for name, expected in proof['runtime_source_sha256'].items():
    assert sha(Path(name)) == expected, name
for name, expected in proof['helper_sha256'].items():
    assert sha(BASE/name) == expected, name
for name in ('run_head.py','library_identity.py'):
    shutil.copyfile(BASE/name, ROOT/name)
s = (BASE/'head_screen.py').read_text()
s = s[:s.index('from lookup import LookupExtension')]
s = s.replace("OUT=Path('/private/tmp/dsv41-hybrid-lookup-20260918/head-screen.json')",
              f'OUT=Path({str(ROOT / "head-screen.json")!r})')
s = s.replace("CURRENT_ROOT=Path('/private/tmp/dsv41-hybrid-lookup-20260918')",
              f'CURRENT_ROOT=Path({str(ROOT)!r})')
assert str(ROOT / 'head-screen.json') in s and f'CURRENT_ROOT=Path({str(ROOT)!r})' in s
s += (ROOT/'screen_body.py').read_text()
(ROOT/'head_screen.py').write_text(s)
proof['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
proof['scope'] = 'Native206 independent boundaries, original D5 retained. At native min-confidence>=0.9, score conditioned D7/D13 suffixes and a masked D13 suffix. Committed target states are inputs; future IDs only score already-created proposals. Not a trajectory, tree verifier, target run or TPS result.'
proof['control_proposal_receipt'] = str(BASE/'head-screen.json')
proof['control_proposal_sha256'] = sha(BASE/'head-screen.json')
proof['native_prefix_length'] = 5
proof['suffix_gate'] = {'feature':'minimum native sigmoid confidence', 'threshold':0.9,
    'rule':'Broadest tested min-confidence threshold with no native-prefix failures in first103 archived boundaries; selected before this suffix screen.',
    'training_selected':44,'training_correct':44,'heldout_selected':42,'heldout_correct':40}
proof['tail_cases'] = [{'name':'conditioned7','width':7,'conditioned':True},
    {'name':'conditioned13','width':13,'conditioned':True},
    {'name':'masked13','width':13,'conditioned':False}]
proof['bound_components']['tail_logits_and_inputs_inside_workspace_bytes'] = 32*1024**2
proof['bound_scope'] = 'Existing49GiB head-only inventory, T<=13:41GiB active plus4GiB cache and4GiB host. Three owner objects share authenticated parameter arrays. Their at-most three stage output-transpose caches per owner and sequential draft graphs fit the existing8GiB active workspace/compiler allowance. No target trunk or expert streaming bank is constructed; second-pass outputs evaluated before the next case.'
proof['arithmetic_scope'] = 'Native D5, target weights, target arithmetic, verifier and committed-state seeding remain unchanged. Only proposal generation differs. Draft attention writes no cache state; the screen checks window identity and offsets around every extra pass.'
proof['prior_head_script_sha256'] = sha(BASE/'head_screen.py')
proof['helper_sha256'] = {p.name:sha(p) for p in ROOT.glob('*.py')}
for p in ROOT.glob('*.py'):
    ast.parse(p.read_text())
(ROOT/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
command = (BASE/'command.sh').read_text().replace(str(BASE),str(ROOT)).replace('/tmp/dsv41-hybrid-lookup-20260918',str(ROOT))
(ROOT/'command.sh').write_text(command)
print(json.dumps({'source':proof['source_commit'],'incremental_bound_bytes':proof['static_incremental_bound_bytes'],
    'max_draft_width':13,'target_execution':False,'root':str(ROOT)}))
