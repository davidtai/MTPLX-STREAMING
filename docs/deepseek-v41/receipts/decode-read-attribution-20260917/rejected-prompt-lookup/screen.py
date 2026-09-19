"""CPU-only causal prompt lookup screen; teacher tokens evaluate proposals only."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEACHER = Path('/tmp/dsv41-depth-replay-20260917/teacher.json')
PROMPT = Path('docs/deepseek-v41/receipts/memory-budget-110/python-prompt-ids.json')
t = json.loads(TEACHER.read_text())
p = json.loads(PROMPT.read_text())
prompt = next(v['token_ids'] for v in p['prompts'] if len(v['token_ids']) == 16384)
expected = t['token_ids']
assert len(expected) == 1024
assert t['token_ids_sha256'] == '0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac'
history = list(prompt)
last = [{} for _ in range(17)]
first = [{} for _ in range(17)]
for end in range(2, len(history)):
    for n in range(2, min(16, end) + 1):
        key = tuple(history[end-n:end])
        last[n][key] = end
        first[n].setdefault(key, end)
rows = []
for pos, committed in enumerate(expected[:-1]):
    # Record suffixes that now have at least one known continuation token.
    end = len(history)
    for n in range(2, 17):
        key = tuple(history[end-n:end])
        last[n][key] = end
        first[n].setdefault(key, end)
    history.append(committed)
    row = {'position': pos}
    for selection, index in (('latest', last), ('earliest', first)):
        for n in range(16, 1, -1):
            key = tuple(history[-n:])
            match = index[n].get(key)
            if match is not None:
                proposal = history[match:min(match+63, len(history))]
                accepted = 0
                for actual, draft in zip(expected[pos+1:], proposal):
                    if actual != draft:
                        break
                    accepted += 1
                row[selection] = {'suffix_length': n, 'source_end': match,
                                  'history_length': len(history),
                                  'proposed': len(proposal), 'accepted': accepted}
                break
        else:
            row[selection] = {'suffix_length': 0, 'proposed': 0, 'accepted': 0}
    rows.append(row)
arms = []
for selection in ('latest', 'earliest'):
    for min_suffix in (2, 4, 8, 16):
        for width in (5, 7, 15, 31, 63):
            pos = 0
            cycles = 0
            proposed = 0
            accepted = 0
            lookup_cycles = 0
            while pos < len(expected) - 1:
                row = rows[pos][selection]
                k = min(width, row['proposed']) if row['suffix_length'] >= min_suffix else 0
                a = min(k, row['accepted'], len(expected)-1-pos)
                proposed += k
                accepted += a
                lookup_cycles += k > 0
                pos += min(a + 1, len(expected)-1-pos)
                cycles += 1
            arms.append({'selection': selection, 'min_suffix': min_suffix, 'width': width,
                         'cycles': cycles, 'verify_rows': cycles + proposed,
                         'proposed_tokens': proposed, 'accepted_tokens': accepted,
                         'lookup_cycles': lookup_cycles})
control_positions = []
pos = 0
for length in t['commit_lengths']:
    control_positions.append(pos)
    pos += min(length, 1023-pos)
comparison = []
for selection in ('latest', 'earliest'):
    for width in (5, 7, 15, 31, 63):
        kept = sum(min(rows[pos][selection]['accepted'], width) for pos in control_positions)
        proposed = sum(min(rows[pos][selection]['proposed'], width) for pos in control_positions)
        comparison.append({'selection': selection, 'width': width,
                           'native_boundary_lookup_accepted': kept,
                           'native_boundary_lookup_proposed': proposed})
report = {'scope': 'Causal token-only screen; no target or draft model execution. Teacher futures score proposals but never select them. Standalone lookup falls back to one target token when no match; not a measured hybrid or throughput result.',
          'prompt_file_sha256': hashlib.sha256(PROMPT.read_bytes()).hexdigest(),
          'teacher_file_sha256': hashlib.sha256(TEACHER.read_bytes()).hexdigest(),
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'native_cycles': len(t['commit_lengths']), 'arms': arms,
          'native_boundary_comparisons': comparison, 'proposals_by_position': rows}
(ROOT/'screen.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'best_by_cycles': sorted(arms, key=lambda a: a['cycles'])[:8],
                  'native_boundary_comparisons': comparison}, indent=2))
