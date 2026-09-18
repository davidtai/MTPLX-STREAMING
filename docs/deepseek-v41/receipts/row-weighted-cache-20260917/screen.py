"""CPU screen of position-weighted retention; all requested experts still run."""
from collections import deque
import gzip
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import time
from types import MethodType

restore=runpy.run_path('docs/deepseek-v41/receipts/memory-budget-110/replay_route_policies.py')['restore']
import numpy as np

ROOT=Path(__file__).resolve().parent
TRACE=Path('docs/deepseek-v41/receipts/mtp-verify-routes-20260913/mtp-verify-routes-16k-1024-v2.json.gz')
CAPACITY=102


class WeightedHistory:
    def __init__(self,bank,decay,weight_prediction):
        self.bank=bank
        self.row_weights=np.repeat(np.float32(decay)**np.arange(6,dtype=np.float32),6)
        self.weight_prediction=weight_prediction
        self.current_weights=np.ones(384,dtype=np.float32)
        self.previous_weights=None
        self.window=deque()

    def set_route(self,route):
        self.current_weights.fill(0)
        np.maximum.at(self.current_weights,np.asarray(route,dtype=np.intp),self.row_weights)

    def observe(self,current):
        b=self.bank
        indices=np.asarray(current,dtype=np.intp)
        weights=self.current_weights[indices].copy()
        previous=b._transition_previous
        if previous is not None:
            previous_indices=np.asarray(previous,dtype=np.intp)
            if self.weight_prediction:
                # Learn physical future demand (all requested experts), with
                # weighted preceding features; no future acceptance is read.
                b._transition_counts[np.ix_(previous_indices,indices)]+=self.previous_weights[:,None]
                b._transition_denominators[previous_indices]+=self.previous_weights*np.float32(len(current))
            else:
                b._transition_counts[np.ix_(previous_indices,indices)]+=1.0
                b._transition_denominators[previous_indices]+=float(len(current))
        self.window.append((indices,weights))
        b._transition_window_frequency[indices]+=weights
        if len(self.window)>b._transition_window_limit:
            expired,expired_weights=self.window.popleft()
            b._transition_window_frequency[expired]-=expired_weights
        b._transition_previous=current
        self.previous_weights=weights

    def scores(self):
        b=self.bank
        current=np.asarray(b._transition_previous,dtype=np.intp)
        valid=b._transition_denominators[current]>0
        if np.any(valid):
            rows=current[valid]
            probabilities=b._transition_counts[rows]/b._transition_denominators[rows,None]
            if self.weight_prediction:
                weights=self.previous_weights[valid]
                # Keep the original predictor/frequency mixture scale.
                weights=weights*(np.float32(len(rows))/np.sum(weights,dtype=np.float32))
                probabilities=probabilities*weights[:,None]
            prediction=np.sum(probabilities,axis=0,dtype=np.float32)
        else:prediction=np.zeros(384,dtype=np.float32)
        freq=b._transition_window_frequency
        last=np.fromiter((h.last_used for h in b._history),dtype=np.int64,count=384)
        recency=np.zeros(384,dtype=np.float32);seen=last>=0
        recency[seen]=1.0/(1.0+b._decode_epoch-last[seen])
        return (b._transition_prediction_weight*prediction
                +b._transition_frequency_weight*(freq/max(float(freq.max()),1.0))
                +b._transition_recency_weight*recency)


def replay(trace,mode,decay):
    results={}
    for layer,sequence in trace['target_routes_by_layer'].items():
        bank=restore(trace['initial_banks'][layer],policy='transition-window',single_pool=True)
        extra=CAPACITY-bank.persistent_slots
        bank._slot_to_expert.extend([None]*extra)
        bank.persistent_slots=bank._persistent_capacity=CAPACITY
        bank.slot_count+=extra
        bank._protected_cap=max(1,int(CAPACITY*.8))
        history=None
        if mode!='native':
            history=WeightedHistory(bank,decay,mode=='weighted_prediction_and_frequency')
            bank._observe_transition_window=history.observe
            bank._transition_window_scores=history.scores
        counts=[]
        for route in sequence:
            if history is not None:history.set_route(route)
            counts.append(len(bank.plan(route,phase='decode').misses))
        results[layer]={'first_half_misses':sum(counts[:103]),'heldout_misses':sum(counts[103:]),
                        'total_misses':sum(counts),'per_cycle':counts}
    return {**{k:sum(row[k] for row in results.values()) for k in ('first_half_misses','heldout_misses','total_misses')},
            'per_layer':results}


def main():
    trace=json.loads(gzip.decompress(TRACE.read_bytes()))
    assert trace['complete'] and trace['cycles']==206
    assert all(len(row)==36 for seq in trace['target_routes_by_layer'].values() for row in seq)
    report={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'scope':'Policy demand misses only; no physical transient-reuse or throughput claim',
            'information':'Current and past requested rows only. No target logits, accepted lengths or future routes select retention.',
            'seed':'Identical 73 captured residents plus29 empty slots; native transition-window admission and pins',
            'selection':'Two fixed0.9-per-row candidate designs; chronological first103 cycles then103 heldout cycles',
            'capacity':CAPACITY,'trace_sha256':hashlib.sha256(TRACE.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'arms':{},'complete':False}
    for name,mode,decay in (('native','native',1.0),('unit_weight_control','weighted_prediction_and_frequency',1.0),
                            ('weighted_frequency','weighted_frequency',.9),
                            ('weighted_prediction_and_frequency','weighted_prediction_and_frequency',.9)):
        start=time.perf_counter();arm=replay(trace,mode,decay);arm['elapsed_s']=time.perf_counter()-start
        if name=='unit_weight_control':
            assert arm['per_layer']==report['arms']['native']['per_layer'],'CPU control no longer reproduces native admission'
        report['arms'][name]=arm
        print(json.dumps({'arm':name,**{k:v for k,v in arm.items() if k!='per_layer'}}),flush=True)
        (ROOT/'screen.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;report['unit_weight_control_exact']=True
    (ROOT/'screen.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
