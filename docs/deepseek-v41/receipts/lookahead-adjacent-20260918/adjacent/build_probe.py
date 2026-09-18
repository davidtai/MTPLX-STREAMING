"""Adapt the completed paired operator, preserving its source arithmetic."""
from pathlib import Path

r=Path(__file__).resolve().parent
s=Path('/tmp/dsv41-lookahead-io-20260918/probe.py').read_text()
s=s.replace('Optimistic paired-layer I/O screen; no full model or live router timing.',
    'Three adjacent layers with continuous timing and explicitly labeled gate cost.')
s=s.replace('mx.set_memory_limit(7*1024**3)','mx.set_memory_limit(10*1024**3)')
s=s.replace('LAYERS==(30,31)','LAYERS==(30,31,32)')
s=s.replace('import plane_lane','import plane_lane\nfrom predictor_cost import Issue,load_gates')
s=s.replace('scales={}\nreferences={}','scales={}\ngates={}\nscores={}\nreferences={}')
begin=s.index('\nclass Issue:');end=s.index('\ndef run_arm(',begin)
s=s[:begin]+s[end:]
s=s.replace('\n    x=shared_input=indices=y0=y1=s0=s1=None','\n    x=shared_input=indices=y=shared=all_indices=None\n    kept=[]')
s=s.replace('        issue=Issue(runtime)',"        issue={l:Issue(runtime,l+1,gates[l+1],data['predictor_config'][str(l+1)],scores[l+1]) for l in (30,31)}")
s=s.replace("prefetch_source=(30,issue) if mode=='prefetch' else None)","prefetch_sources=issue if mode=='prefetch' else None)")
begin=s.index('        for call in range(64):');end=s.index("        result['reader_metrics']",begin)
s=s[:begin]+'''        all_indices=[[mx.array(data['routes'][str(l)][call],mx.int32).reshape(1,6,6) for l in LAYERS] for call in range(64)]
        mx.eval(all_indices)
        cohort_start=heldout_start=None
        for call in range(64):
            if call==0:cohort_start=time.perf_counter_ns()
            if call==32:
                # Start at the natural warm/cohort boundary. There are no hash,
                # metrics or CPU conversion gaps between any layer or cycle.
                heldout_start=time.perf_counter_ns()
            for layer,indices in zip(LAYERS,all_indices[call]):
                if layer in issue:issue[layer].call=call
                y,shared=switches[layer]._run(x,indices,shared_work=lambda:mx.tanh(shared_input))
                mx.eval(y,shared)
                kept.append(y)
        runtime.flush_deferred_slot_releases(evaluate=True)
        runtime._drain_prefetch_loads()
        mx.synchronize()
        end=time.perf_counter_ns()
        result['heldout_total_ns']=end-heldout_start
        result['continuous_all64_ns']=end-cohort_start
        # Only after ALL speculative reads and GPU work finish, inspect outputs.
        for n,value in enumerate(kept):
            digest=hashlib.sha256(np.array(value.view(mx.uint16)).tobytes()).hexdigest()
            if sequence==0:references[n]=digest
            if digest!=references[n]:raise RuntimeError(f'prefetch changed native output at layer-call{n}')
            result['cases'].append({'call':n//3,'layer':LAYERS[n%3],'output_sha256':digest})
        value=None
''' +s[end:]
s=s.replace("        result['heldout_total_ns']=sum(x['elapsed_ns'] for x in result['cases'][32:])\n        result['heldout_read_records']=sum(x['records_read'] for x in result['cases'][32:])\n",'')
s=s.replace('        x=shared_input=indices=y0=y1=s0=s1=None', '        kept.clear()\n        x=shared_input=indices=y=shared=all_indices=value=None')
s=s.replace("try:\n    scales={l:load_layer", "try:\n    gates,report['gate_tensor_identities']=load_gates(model,proof)\n    scores={l:tuple(mx.array(row,mx.float32) for row in data['captured_scores'][str(l)]) for l in (31,32)}\n    mx.eval(scores)\n    scales={l:load_layer")
s=s.replace('    scales.clear();gc.collect();mx.synchronize();mx.clear_cache()', '    scales.clear();gates.clear();scores.clear();gc.collect();mx.synchronize();mx.clear_cache()')
s=s.replace("'PAIR_ARM'","'ADJACENT_ARM'").replace("'PAIR_COMPLETE'","'ADJACENT_COMPLETE'")
(r/'probe.py').write_text(s)
compile(s,str(r/'probe.py'),'exec')
