"""CPU screen: delay final admission until current hit consumers finish."""
import gzip
import hashlib
import json
from pathlib import Path
import time
from screen_online import ROOT, TRACE, restore


def replay(state, sequence, capacity, repair):
    bank = restore(state, policy='transition-window', single_pool=True)
    extra = capacity - bank.persistent_slots
    bank._slot_to_expert.extend([None] * extra)
    bank.persistent_slots = bank._persistent_capacity = capacity
    bank.slot_count += extra
    bank._protected_cap = max(1, int(capacity * .8))
    reads, copies = [], 0
    for step in sequence:
        plan = bank.try_plan_all_hits(step, phase='decode')
        if plan is None:
            plan = bank.plan(step, phase='decode')
        reads.append(len(plan.misses))
        if repair:
            # All route consumers have hypothetically finished. Its bypassed
            # misses still occupy transient slots, so their rows can be copied
            # into a now-recyclable current-hit slot without another SSD read.
            bypassed = set(step) - bank._expert_to_slot.keys()
            if not bypassed:
                continue
            scores = bank._transition_window_scores()
            rank = lambda e: bank._transition_window_rank(e, scores)
            candidates = set(bank._expert_to_slot) | bypassed
            keep = set(sorted(candidates, key=rank, reverse=True)[:capacity])
            promote = sorted(bypassed & keep, key=rank, reverse=True)
            victims = sorted(set(bank._expert_to_slot) - keep, key=rank)
            assert len(promote) == len(victims)
            for expert, victim in zip(promote, victims):
                bank._assign_transition_window_slot(expert=expert,
                    slot=bank._expert_to_slot[victim], evictions=[])
                copies += 1
    return dict(total_reads=sum(reads), train_reads=sum(reads[:100]),
                heldout_reads=sum(reads[100:]), record_copies=copies)


with gzip.open(TRACE,'rt') as f:
    trace = json.load(f)
report = dict(scope='Causal CPU post-service admission screen; copies and mandatory wait are not timed; no GPU/runtime change',
    trace_sha256=hashlib.sha256(TRACE.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), arms=[])
for capacity, repair in ((100,False),(100,True)):
    start=time.perf_counter()
    rows={layer:replay(trace['initial_banks'][layer],seq,capacity,repair)
          for layer,seq in trace['target_routes_by_layer'].items()}
    row=dict(capacity=capacity,post_service_repair=repair,elapsed_s=time.perf_counter()-start,
        **{key:sum(x[key] for x in rows.values()) for key in ('total_reads','train_reads','heldout_reads','record_copies')},per_layer=rows)
    report['arms'].append(row)
    print(json.dumps({k:v for k,v in row.items() if k!='per_layer'}),flush=True)
(ROOT/'post-service-screen.json').write_text(json.dumps(report,indent=2)+'\n')
