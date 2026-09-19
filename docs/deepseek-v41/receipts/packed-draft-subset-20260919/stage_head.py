"""Qualify the measured packed draft projection with the existing smaller draft."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parent
out = root / 'head'
base = Path('/tmp/dsv41-draft-packed-projection-20260919').resolve()
repo = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
prior = json.loads((base / 'installation.json').read_text())
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
assert head == 'b5f41c51d74a770069048dddd0a644f07886a999'
for path, digest in prior['runtime_source_sha256'].items():
    assert sha(Path(path)) == digest, path
for name, digest in prior['helper_sha256'].items():
    assert sha(base / name) == digest, name
assert json.loads((base / 'head-screen.json').read_text())['complete']
out.mkdir()
for name in ('projection.py', 'run_head.py', 'lookup.py', 'library_identity.py',
             'selection.json', 'control-commit-lengths.json', 'wait_and_run.py'):
    shutil.copy2(base / name, out / name)
config_source = Path('/tmp/dsv41-extension-draft-20260919/full-v1/packed/draft-config.json')
config = json.loads(config_source.read_text())
assert tuple(map(len, config['selected_experts_by_stage'])) == (80, 40, 24)
assert all(len(row) == 128 for row in config['alias_experts_by_stage'])
shutil.copy2(config_source, out / 'draft-config.json')
s = (base / 'head_screen.py').read_text().replace(str(base), str(out))
anchor = 'projection = Installation(owner)'
assert s.count(anchor) == 1
s = s.replace(anchor, '''draft_config = json.loads((CURRENT_ROOT/'draft-config.json').read_text())
original_luts = LUTS
candidate_luts = []
for stage, aliases in enumerate(draft_config['alias_experts_by_stage']):
    positions = {expert: slot for slot, expert in enumerate(selected[stage])}
    retained = set(draft_config['selected_experts_by_stage'][stage])
    if len(aliases) != 128 or any(e not in retained for e in aliases):
        raise RuntimeError('saved subset alias table differs')
    candidate_luts.append(mx.array([positions[e] for e in aliases], dtype=mx.int32))
mx.eval(candidate_luts)
report['draft_config'] = draft_config
report['draft_config_sha256'] = digest_file(CURRENT_ROOT/'draft-config.json')
projection = Installation(owner)''')
s = s.replace("('hybrid_m8','packed_f32','hybrid_m8')", "('hybrid_m8','packed_subset','hybrid_m8')")
s = s.replace("    report['projection_installations'].append(projection.select(mode=='packed_f32'))",
              "    LUTS = candidate_luts if mode == 'packed_subset' else original_luts\n"
              "    report['projection_installations'].append(projection.select(mode=='packed_subset'))")
s = s.replace("print('DRAFT_PACKED_ARM'", "print('PACKED_SUBSET_ARM'")
(out / 'head_screen.py').write_text(s)
scope = ('Composition qualification: native/packed-plus-80-40-24-aliases/native. '
         'Original183 physical expert owners remain for comparison; only proposals '
         'use the144-expert subset. No target execution or full TPS. Fresh causal '
         'lookup per arm, original target teacher only scores proposals.')
proof = dict(prior)
proof.update(source_commit=head, scope=scope,
             previous_head_quality_reused=False,
             subset_config_sha256=sha(out / 'draft-config.json'),
             packed_component_result_sha256=sha(base / 'head-screen.json'),
             physically_compacted=False,
             selection_rule='Saved80/40/24 subset and router-nearest aliases; no new selection.')
proof['helper_sha256'] = {p.name: sha(p) for p in out.glob('*.py')}
proof['helper_sha256']['draft-config.json'] = sha(out / 'draft-config.json')
(out / 'installation.json').write_text(json.dumps(proof, indent=2) + '\n')
(out / 'command.sh').write_text((base / 'command.sh').read_text().replace(str(base), str(out)))
for p in out.glob('*.py'):
    ast.parse(p.read_text())
    assert str(base) not in p.read_text(), p
audit = {'cpu_only': True, 'source_commit': head, 'scope': scope,
         'static_incremental_bound_bytes': proof['static_incremental_bound_bytes'],
         'runtime_source_pins_verified': len(proof['runtime_source_sha256']),
         'helper_pins': proof['helper_sha256'], 'stage_sha256': sha(Path(__file__))}
(out / 'cpu-preflight.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps({k: audit[k] for k in ('source_commit', 'scope', 'static_incremental_bound_bytes')}))
