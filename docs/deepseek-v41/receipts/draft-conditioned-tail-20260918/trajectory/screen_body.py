# This body is appended after the authenticated native head-only loader.
from lookup import LookupExtension
from tail import TailProposal
from collections import Counter

prior=json.loads(Path(installation['control_proposal_receipt']).read_text())
if digest_file(Path(installation['control_proposal_receipt']))!=installation['control_proposal_sha256']:
    raise RuntimeError('hybrid reference proposal identity differs')
control_reference=next(a for a in prior['arms'] if a['mode']=='hybrid_m8')
prompt_payload=json.loads(Path('docs/deepseek-v41/receipts/memory-budget-110/python-prompt-ids.json').read_text())
prompt=next(p['token_ids'] for p in prompt_payload['prompts'] if len(p['token_ids'])==16384)
ids=teacher['token_ids']
owners={}
for width in (5,7):
    model=DraftOnly(width,compact=True)
    current=dict(tree_flatten(model.parameters()))
    if set(current)!=set(weights['compact']):
        raise RuntimeError('draft parameter coverage mismatch')
    del current
    model.load_weights(list(weights['compact'].items()),strict=True)
    if any(v is not weights['compact'][n] for n,v in tree_flatten(model.parameters())):
        raise RuntimeError('random draft parameters remain after installation')
    mx.eval(model.parameters())
    owners[width]=model
del model
tail=TailProposal(owners[7],conditioned=True)
report={'scope':installation['scope'],'source_commit':installation['source_commit'],
    'teacher_sha256':digest_file(TEACHER_PATH),'baseline':snap,'library_identity':library_identity,
    'static_incremental_bound_bytes':installation['static_incremental_bound_bytes'],
    'future_used_only_for_scoring':True,'original_five_proposals_preserved':True,
    'arms':[],'complete':False}
for case in installation['tail_cases']:
    caches=[ds.DSparkStageCache(args.sliding_window,args.head_dim) for _ in range(3)]
    for c,w,offset in zip(caches,initial_windows,teacher['initial_mtp_offsets']):
        c.window=w;c.offset=offset
    lookup=LookupExtension(prompt,minimum_context=2,extra_tokens=2)
    history_count=0
    pos=0;main_h=initial_main;rows=[]
    started=time.perf_counter()
    while pos<len(ids)-1:
        root_ids=mx.array([ids[pos]])
        out,logits,conf=owners[5].mtp.draft_block(main_h,root_ids,caches,owners[5].model.embed_tokens,owners[5].head)
        mx.eval(out,conf)
        prefix=out[:,1:]
        native=[int(v) for v in np.asarray(prefix)[0]]
        confidence=np.asarray(mx.sigmoid(conf.astype(mx.float32))).reshape(-1).tolist()
        lookup.append_committed(ids[history_count:pos+1])
        history_count=pos+1
        proposed=lookup.extend(native)
        source='lookup' if len(proposed)>5 else 'native'
        row={'position':pos,'native_proposed_ids':native,'native_confidence':confidence,
             'tail_head_called':False}
        if case['tail'] and len(proposed)==5 and min(confidence)>=0.9:
            suffix,tail_conf=tail(main_h,root_ids,prefix,caches)
            mx.eval(suffix,tail_conf)
            suffix_ids=[int(v) for v in np.asarray(suffix)[0]]
            suffix_confidence=np.asarray(mx.sigmoid(tail_conf.astype(mx.float32))).reshape(-1).tolist()
            keep=len(suffix_ids)
            threshold=case['tail_confidence_threshold']
            if threshold is not None:
                keep=0
                for conf_value in suffix_confidence:
                    if conf_value<threshold:break
                    keep+=1
            proposed=native+suffix_ids[:keep]
            source='conditioned' if keep else 'native'
            row.update(tail_head_called=True,suffix_ids=suffix_ids,suffix_confidence=suffix_confidence)
            del suffix,tail_conf
        matched=0
        for draft,actual in zip(proposed,ids[pos+1:]):
            if draft!=actual:break
            matched+=1
        advance=min(matched+1,len(ids)-1-pos)
        row.update(proposed_ids=proposed,source=source,teacher_matched_prefix=matched,
                   committed_tokens=advance,verify_width=len(proposed)+1)
        if case['name']=='hybrid_control':
            reference_row=control_reference['rows'][len(rows)]
            if (reference_row['position']!=pos or reference_row['proposed_ids']!=proposed
                    or reference_row['committed_tokens']!=advance):
                raise RuntimeError('retained hybrid control does not reproduce proposals and boundaries')
        rows.append(row)
        owners[5].mtp.seed_main(hidden[pos:pos+advance][None,:,:],caches)
        mx.eval([c.window for c in caches])
        main_h=hidden[pos+advance-1:pos+advance][None,:,:]
        pos+=advance
    arm={'mode':case['name'],'cycles':len(rows),'verify_rows':sum(r['verify_width'] for r in rows),
         'verify_width_counts':dict(Counter(r['verify_width'] for r in rows)),
         'proposal_source_counts':dict(Counter(r['source'] for r in rows)),
         'tail_head_calls':sum(r['tail_head_called'] for r in rows),
         'head_replay_wall_s':time.perf_counter()-started,'rows':rows,
         'mlx_peak_bytes':int(mx.get_peak_memory())}
    report['arms'].append(arm)
    OUT.write_text(json.dumps(report,indent=2)+'\n')
    print('CONDITIONED_TAIL_ARM',json.dumps({k:v for k,v in arm.items() if k!='rows'}),flush=True)
    del caches,main_h,out,logits,conf
    gc.collect();mx.synchronize();mx.clear_cache()
report.update(complete=True,after=host_memory_snapshot())
OUT.write_text(json.dumps(report,indent=2)+'\n')
