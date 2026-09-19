"""CPU-only independent-boundary opportunity screen; no target TPS claim."""
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import time


class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('MLX prohibited in CPU proposal screen')


sys.meta_path.insert(0,NoMLX())
from consensus import SuffixConsensus
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot

ROOT = Path(__file__).resolve().parent
REPO = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
HYBRID = Path('/tmp/dsv41-hybrid-lookup-20260918')
TEACHER = Path('/tmp/dsv41-depth-replay-20260917/teacher.json')
PROMPT = REPO/'docs/deepseek-v41/receipts/memory-budget-110/python-prompt-ids.json'
sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out = ROOT/'screen.json'
if out.exists():
    raise RuntimeError('refusing evidence overwrite')
before = host_memory_snapshot()
if not before['box']['ok'] or before['box']['used_bytes'] + 512*1024**2 > 110000000000:
    raise RuntimeError('CPU bound does not fit')
spec = importlib.util.spec_from_file_location('lookup',HYBRID/'lookup.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
teacher = json.loads(TEACHER.read_text())
expected = teacher['token_ids']
prior = json.loads((HYBRID/'head-screen.json').read_text())
assert prior['teacher_sha256'] == sha(TEACHER)
assert teacher['token_ids_sha256'] == '0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac'
control = next(a for a in prior['arms'] if a['mode']=='hybrid_m8')
assert control['cycles'] == 198 and control['verify_rows'] == 1242
payload = json.loads(PROMPT.read_text())
prompt = next(p['token_ids'] for p in payload['prompts'] if len(p['token_ids']) == 16384)


def match(proposed,future):
    n = 0
    for a,b in zip(proposed,future):
        if a != b:
            break
        n += 1
    return n


started = time.perf_counter()
arms = []
# Selection uses only the first half. The confidence threshold was already
# calibrated by the earlier native draft study, not by this screen's futures.
for minimum in (4,3):
    baseline = module.LookupExtension(prompt, minimum_context=2, extra_tokens=2)
    backoff = SuffixConsensus(prompt,min_suffix=minimum,min_count=2,max_extra=2)
    rows = []
    history_count = 0
    stats = {k:{'issued':0,'extra_rows':0,'extra_matches':0,'productive':0,'bad_native_prefix':0}
             for k in ('first_half','second_half','all')}
    for row in control['rows']:
        pos = row['position']
        committed = expected[history_count:pos+1]
        baseline.append_committed(committed)
        backoff.append_committed(committed)
        history_count = pos+1
        native = row['native_proposed_ids']
        current = baseline.extend(native)
        assert current == row['proposed_ids']
        candidate = current
        if len(current)==5 and min(row['confidence']) >= .9:
            candidate = backoff.extend(native)
        # Only this scoring block receives the target future. Neither proposer
        # receives the teacher object, future IDs, or the match count.
        future = expected[pos+1:]
        control_match,candidate_match = match(current,future),match(candidate,future)
        if control_match != row['teacher_matched_prefix']:
            raise RuntimeError('saved hybrid boundary does not reproduce')
        added = len(candidate)-len(current)
        gain = min(candidate_match-control_match,max(0,len(expected)-1-pos-row['committed_tokens']))
        if added:
            for cohort in ('all','first_half' if pos<512 else 'second_half'):
                d = stats[cohort]
                d['issued'] += 1
                d['extra_rows'] += added
                d['extra_matches'] += gain
                d['productive'] += gain>0
                d['bad_native_prefix'] += control_match<5
        rows.append({'position':pos,'control':current,'candidate':candidate,'extra_rows':added,
                     'extra_matches':gain,'confidence_min':min(row['confidence'])})
    arms.append({'minimum_suffix':minimum,'minimum_occurrences':2,'confidence_minimum':.9,
                 'max_extra':2,'stats':stats,'rows':rows,
                 'index_keys':len(backoff.ends),'index_entries':sum(len(v) for v in backoff.ends.values())})
eligible = [a for a in arms if a['stats']['first_half']['extra_matches'] >= 5
            and a['stats']['first_half']['extra_matches'] >= .75*a['stats']['first_half']['extra_rows']]
selected = max(eligible,key=lambda a:(a['stats']['first_half']['extra_matches'],
                                     -a['stats']['first_half']['extra_rows'],a['minimum_suffix'])) if eligible else None
report = {'complete':True,'cpu_only':True,'elapsed_s':time.perf_counter()-started,'before':before,
          'after':host_memory_snapshot(),'incremental_bound_bytes':512*1024**2,
          'source_sha256':{str(p):sha(p) for p in (Path(__file__),ROOT/'consensus.py',TEACHER,PROMPT,HYBRID/'head-screen.json',HYBRID/'lookup.py')},
          'control_boundaries':198,'control_exact':True,'teacher_futures_only_score':True,
          'selection_rule':'First half >=5 additional matched tokens and >=75% additional-row efficiency; maximize first-half gain. No second-half selection.',
          'selected_minimum_suffix':None if selected is None else selected['minimum_suffix'],
          'arms':arms,'scope':__doc__,'mlx_imported':any(k=='mlx' or k.startswith('mlx.') for k in sys.modules)}
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('complete','elapsed_s','control_exact','selected_minimum_suffix','mlx_imported')}))
for a in arms:
    print(json.dumps({k:v for k,v in a.items() if k!='rows'}))
