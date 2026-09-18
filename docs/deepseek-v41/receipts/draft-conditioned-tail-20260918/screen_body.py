# This body is appended after the authenticated native head-only loader.
from tail import TailProposal

prior = json.loads(Path(installation['control_proposal_receipt']).read_text())
if digest_file(Path(installation['control_proposal_receipt'])) != installation['control_proposal_sha256']:
    raise RuntimeError('native reference proposal identity differs')
native_reference = next(a for a in prior['arms'] if a['mode'] == 'native')
ids = teacher['token_ids']
owners = {}
for width in (5,7,13):
    model = DraftOnly(width, compact=True)
    current = dict(tree_flatten(model.parameters()))
    if set(current) != set(weights['compact']):
        raise RuntimeError('draft parameter coverage mismatch')
    del current
    model.load_weights(list(weights['compact'].items()), strict=True)
    if any(v is not weights['compact'][n] for n,v in tree_flatten(model.parameters())):
        raise RuntimeError('random draft parameters remain after installation')
    mx.eval(model.parameters())
    owners[width] = model
del model
variants = {case['name']:TailProposal(owners[case['width']],conditioned=case['conditioned'])
            for case in installation['tail_cases']}
caches = [ds.DSparkStageCache(args.sliding_window,args.head_dim) for _ in range(3)]
for c,w,offset in zip(caches,initial_windows,teacher['initial_mtp_offsets']):
    c.window = w
    c.offset = offset
pos = 0
main_h = initial_main
rows = []
started = time.perf_counter()
report = {'scope':installation['scope'],'source_commit':installation['source_commit'],
    'teacher_sha256':digest_file(TEACHER_PATH), 'baseline':snap, 'library_identity':library_identity,
    'static_incremental_bound_bytes':installation['static_incremental_bound_bytes'],
    'future_used_only_for_scoring':True, 'original_five_proposals_preserved':True,
    'rows':rows,'complete':False}
while pos < len(ids)-1:
    root_ids = mx.array([ids[pos]])
    out,logits,conf = owners[5].mtp.draft_block(main_h,root_ids,caches,
        owners[5].model.embed_tokens,owners[5].head)
    mx.eval(out,conf)
    prefix = out[:,1:]
    proposed = [int(v) for v in np.asarray(prefix)[0]]
    confidence = np.asarray(mx.sigmoid(conf.astype(mx.float32))).reshape(-1).tolist()
    reference_row = native_reference['rows'][len(rows)]
    if reference_row['position'] != pos or reference_row['native_proposed_ids'] != proposed:
        raise RuntimeError('native D5 proposals do not reproduce the exact reference')
    row = {'cycle':len(rows),'position':pos,'native_proposed_ids':proposed,
        'native_confidence':confidence,'suffix_issued':min(confidence)>=0.9,'variants':{}}
    # Produce every candidate before looking at this boundary's future IDs.
    if row['suffix_issued']:
        for name,variant in variants.items():
            windows = tuple(c.window for c in caches)
            offsets = tuple(c.offset for c in caches)
            suffix,tail_conf = variant(main_h,root_ids,prefix,caches)
            mx.eval(suffix,tail_conf)
            if any(c.window is not w for c,w in zip(caches,windows)) or offsets != tuple(c.offset for c in caches):
                raise RuntimeError('draft suffix modified committed MTP state')
            row['variants'][name] = {'suffix_ids':[int(v) for v in np.asarray(suffix)[0]],
                'suffix_confidence':np.asarray(mx.sigmoid(tail_conf.astype(mx.float32))).reshape(-1).tolist()}
            del suffix,tail_conf,windows
    native_match = 0
    for draft,actual in zip(proposed,ids[pos+1:]):
        if draft != actual:
            break
        native_match += 1
    advance = min(native_match+1,len(ids)-1-pos)
    if advance != reference_row['committed_tokens']:
        raise RuntimeError('native acceptance boundaries differ')
    row.update(native_matched=native_match,native_commit=advance)
    for variant in row['variants'].values():
        matched = 0
        if native_match == 5:
            for draft,actual in zip(variant['suffix_ids'],ids[pos+6:]):
                if draft != actual:
                    break
                matched += 1
        variant['extra_matches'] = matched
        variant['added_committed_tokens_at_native_boundary'] = min(matched,max(0,1023-pos-advance))
    rows.append(row)
    owners[5].mtp.seed_main(hidden[pos:pos+advance][None,:,:],caches)
    mx.eval([c.window for c in caches])
    main_h = hidden[pos+advance-1:pos+advance][None,:,:]
    pos += advance

report['native_cycles'] = len(rows)
report['native_control_exact'] = len(rows)==206
report['summary'] = []
for name in variants:
    for threshold in (None,0.5,0.75,0.9):
        parts = {half:{'native_boundaries':0,'tail_calls':0,'extra_verify_rows':0,
                       'extra_committed_tokens':0,'productive_tails':0}
                 for half in ('training','heldout','all')}
        for row in rows:
            for half in ('all','training' if row['cycle']<103 else 'heldout'):
                parts[half]['native_boundaries'] += 1
            if name not in row['variants']:
                continue
            candidate = row['variants'][name]
            keep = len(candidate['suffix_ids'])
            if threshold is not None:
                keep = 0
                for conf_value in candidate['suffix_confidence']:
                    if conf_value < threshold:
                        break
                    keep += 1
            extra = min(keep,candidate['added_committed_tokens_at_native_boundary'])
            for half in ('all','training' if row['cycle']<103 else 'heldout'):
                parts[half]['tail_calls'] += 1
                parts[half]['extra_verify_rows'] += keep
                parts[half]['extra_committed_tokens'] += extra
                parts[half]['productive_tails'] += extra>0
        report['summary'].append({'variant':name,'tail_confidence_threshold':threshold,**parts})
report.update(complete=True,elapsed_s=time.perf_counter()-started,
    mlx_peak_bytes=int(mx.get_peak_memory()),after=host_memory_snapshot())
OUT.write_text(json.dumps(report,indent=2)+'\n')
print('CONDITIONED_TAIL_SCREEN',json.dumps({k:report[k] for k in ('complete','native_cycles','native_control_exact','elapsed_s','mlx_peak_bytes','summary')}),flush=True)
