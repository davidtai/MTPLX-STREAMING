# Appended after the authenticated native draft-only loader.
from lookup import LookupExtension
from consensus import SuffixConsensus
from collections import Counter

prior = json.loads(Path(installation['control_proposal_receipt']).read_text())
if digest_file(Path(installation['control_proposal_receipt'])) != installation['control_proposal_sha256']:
    raise RuntimeError('hybrid reference identity changed')
control_reference = next(a for a in prior['arms'] if a['mode']=='hybrid_m8')
prompt_payload = json.loads(Path('docs/deepseek-v41/receipts/memory-budget-110/python-prompt-ids.json').read_text())
prompt = next(p['token_ids'] for p in prompt_payload['prompts'] if len(p['token_ids'])==16384)
ids = teacher['token_ids']
model = DraftOnly(5,compact=True)
current = dict(tree_flatten(model.parameters()))
if set(current) != set(weights['compact']):
    raise RuntimeError('draft parameter coverage mismatch')
del current
model.load_weights(list(weights['compact'].items()),strict=True)
if any(v is not weights['compact'][n] for n,v in tree_flatten(model.parameters())):
    raise RuntimeError('unreplaced random parameters')
mx.eval(model.parameters())
report = {'complete':False,'scope':installation['scope'],'source_commit':installation['source_commit'],
          'teacher_sha256':digest_file(TEACHER_PATH),'baseline':snap,'library_identity':library_identity,
          'static_incremental_bound_bytes':installation['static_incremental_bound_bytes'],
          'future_used_only_for_scoring':True,'original_five_proposals_preserved':True,'arms':[]}
for mode in ('hybrid_control','consensus'):
    caches = [ds.DSparkStageCache(args.sliding_window,args.head_dim) for _ in range(3)]
    for c,w,offset in zip(caches,initial_windows,teacher['initial_mtp_offsets']):
        c.window = w
        c.offset = offset
    lookup = LookupExtension(prompt,minimum_context=2,extra_tokens=2)
    backoff = SuffixConsensus(prompt,min_suffix=3,min_count=2,max_extra=2) if mode=='consensus' else None
    history_count = 0
    pos = 0
    main_h = initial_main
    rows = []
    started = time.perf_counter()
    while pos < len(ids)-1:
        root_ids = mx.array([ids[pos]])
        out,logits,conf = model.mtp.draft_block(main_h,root_ids,caches,model.model.embed_tokens,model.head)
        mx.eval(out,conf)
        native = [int(v) for v in np.asarray(out[:,1:])[0]]
        confidence = np.asarray(mx.sigmoid(conf.astype(mx.float32))).reshape(-1).tolist()
        committed = ids[history_count:pos+1]
        lookup.append_committed(committed)
        if backoff is not None:
            backoff.append_committed(committed)
        history_count = pos+1
        proposed = lookup.extend(native)
        source = 'lookup' if len(proposed)>5 else 'native'
        if backoff is not None and len(proposed)==5 and min(confidence)>=.9:
            proposed = backoff.extend(native)
            if len(proposed)>5:
                source = 'consensus'
        # Proposal construction above has only committed history and the native
        # draft. The target teacher is consulted below for scoring and seeding.
        matched = 0
        for a,b in zip(proposed,ids[pos+1:]):
            if a != b:
                break
            matched += 1
        advance = min(matched+1,len(ids)-1-pos)
        row = {'position':pos,'native_proposed_ids':native,'confidence':confidence,
               'proposed_ids':proposed,'source':source,'teacher_matched_prefix':matched,
               'committed_tokens':advance,'verify_width':len(proposed)+1}
        if mode=='hybrid_control':
            ref = control_reference['rows'][len(rows)]
            if any(ref[k]!=row[k] for k in ('position','proposed_ids','committed_tokens')):
                raise RuntimeError('unchanged hybrid control differs')
        rows.append(row)
        model.mtp.seed_main(hidden[pos:pos+advance][None,:,:],caches)
        mx.eval([c.window for c in caches])
        main_h = hidden[pos+advance-1:pos+advance][None,:,:]
        pos += advance
    arm = {'mode':mode,'cycles':len(rows),'verify_rows':sum(r['verify_width'] for r in rows),
           'verify_width_counts':dict(Counter(r['verify_width'] for r in rows)),
           'source_counts':dict(Counter(r['source'] for r in rows)),
           'head_replay_wall_s':time.perf_counter()-started,'rows':rows,
           'mlx_peak_bytes':int(mx.get_peak_memory())}
    report['arms'].append(arm)
    OUT.write_text(json.dumps(report,indent=2)+'\n')
    print('CONSENSUS_HEAD_ARM',json.dumps({k:v for k,v in arm.items() if k!='rows'}),flush=True)
    del caches,main_h,out,logits,conf,lookup,backoff
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
report.update(complete=True,after=host_memory_snapshot())
OUT.write_text(json.dumps(report,indent=2)+'\n')
