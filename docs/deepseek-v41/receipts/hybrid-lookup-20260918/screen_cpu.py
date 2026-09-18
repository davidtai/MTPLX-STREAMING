"""Causal lookup extensions matching all five native MTP proposals.

Each saved native boundary is scored independently. This is an opportunity
screen, not a hybrid cycle count, route replay or throughput measurement.
"""
from collections import defaultdict
import gzip
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import time


class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('MLX is forbidden in the CPU lookup screen')


sys.meta_path.insert(0, NoMLX())
ROOT = Path(__file__).resolve().parent
PROMPT = Path('docs/deepseek-v41/receipts/memory-budget-110/python-prompt-ids.json')
TEACHER = Path('/tmp/dsv41-depth-replay-20260917/teacher.json')
DRAFT = Path('docs/deepseek-v41/receipts/decode-read-attribution-20260917/confidence-screen/screen.json.gz')
FULL = Path('/tmp/dsv41-110-stage/full-embedding-rows-20260918-v2.jsonl')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
teacher = json.loads(TEACHER.read_text())
expected = teacher['token_ids']
full = json.loads(FULL.read_text())
draft_report = json.loads(gzip.decompress(DRAFT.read_bytes()))
native = next(arm for arm in draft_report['arms'] if arm['threshold'] is None)
assert expected == full['dspark']['token_ids'] and len(expected) == 1024
assert draft_report['teacher_sha256'] == sha(TEACHER) and native['cycles'] == 206
assert teacher['token_ids_sha256'] == '0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac'
payload = json.loads(PROMPT.read_text())
prompt = next(p['token_ids'] for p in payload['prompts'] if len(p['token_ids']) == 16384)
history = list(prompt)
ends = defaultdict(list)
indexed_end = 5
rows = []
started = time.perf_counter()
for cycle, row in enumerate(native['rows']):
    pos = row['position']
    history = prompt + expected[:pos+1]
    # Index only occurrences with an already-observed continuation token.
    for end in range(indexed_end, len(history)):
        ends[tuple(history[end-5:end])].append(end)
    indexed_end = len(history)
    proposal = tuple(row['proposed_ids'])
    assert len(proposal) == 5
    matches = []
    for end in ends.get(proposal, ()):
        context = 0
        while (context < 32 and end-5-context > 0
               and history[end-6-context] == history[-1-context]):
            context += 1
        matches.append((context,end))
    native_match = 0
    for a,b in zip(proposal, expected[pos+1:]):
        if a != b:
            break
        native_match += 1
    assert native_match == row['teacher_matched_prefix']
    candidate = {'cycle':cycle,'position':pos,'native_commit':row['committed_tokens'],
                 'native_matched_drafts':native_match,'history_length':len(history),'choices':{}}
    for selection in ('latest','earliest'):
        # Longest context is primary; source recency breaks ties, never futures.
        ranked = sorted(matches,key=lambda p:(p[0],p[1] if selection=='latest' else -p[1]),reverse=True)
        if not ranked:
            candidate['choices'][selection] = None
            continue
        context,end = ranked[0]
        extension = history[end:min(end+32,len(history))]
        extra = 0
        if native_match == 5:
            for a,b in zip(extension,expected[pos+6:]):
                if a != b:
                    break
                extra += 1
        candidate['choices'][selection] = {'context_tokens':context,'source_end':end,
            'extension':extension,'matched_extension_tokens':extra,
            'native_prefix_fully_correct':native_match==5}
    rows.append(candidate)
arms = []
for selection in ('latest','earliest'):
    for min_context in (0,2,4,8,16):
        for extra_cap in (2,4,8,16,32):
            stats = {half:{'opportunities':0,'extra_verify_rows':0,'extra_accepted_tokens':0,
                          'productive_extensions':0,'bad_native_prefix':0}
                     for half in ('first_half','second_half','all')}
            for row in rows:
                choice = row['choices'][selection]
                if choice is None or choice['context_tokens'] < min_context:
                    continue
                proposed = min(extra_cap,len(choice['extension']))
                gained = min(proposed,choice['matched_extension_tokens'],max(0,1023-row['position']-row['native_commit']))
                for half in ('all','first_half' if row['position']<512 else 'second_half'):
                    d=stats[half]
                    d['opportunities']+=proposed>0
                    d['extra_verify_rows']+=proposed
                    d['extra_accepted_tokens']+=gained
                    d['productive_extensions']+=gained>0
                    d['bad_native_prefix']+=not choice['native_prefix_fully_correct']
            arms.append({'selection':selection,'minimum_context_tokens':min_context,
                         'max_extra_rows':extra_cap,**stats})
report = {'scope':__doc__,'cpu_only':True,'elapsed_s':time.perf_counter()-started,
          'teacher_sha256':sha(TEACHER),'prompt_sha256':sha(PROMPT),
          'native_draft_receipt_sha256':sha(DRAFT),'native_full_sha256':sha(FULL),
          'script_sha256':sha(Path(__file__)),'native_cycles':206,'rows':rows,'arms':arms,
          'selection_is_causal':True,'teacher_future_used_only_for_scoring':True,
          'width_cost_and_hybrid_cycle_positions_unmeasured':True}
(ROOT/'screen.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'elapsed_s':report['elapsed_s'],'best_training_gain':sorted(arms,
    key=lambda a:(a['first_half']['extra_accepted_tokens'],-a['first_half']['extra_verify_rows']),reverse=True)[:6],
    'M8_configurations':[a for a in arms if a['max_extra_rows']==2]},indent=2))
