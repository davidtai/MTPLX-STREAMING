"""NumPy-only retrospective scoring; chronological calibration and holdout."""
import numpy as np


def analyze(scores,actual,nrows,persistent,physical,reads):
    cycles,layers,rows,experts = scores.shape[1:]
    assert (cycles,layers,rows,experts)==(64,36,6,384)
    true = np.zeros((cycles,layers,experts),bool)
    c,l,r,k = np.indices(actual[:,4:].shape)
    true[c,l,actual[:,4:]] = True
    missing = reads[:,4:] > 0
    if np.any(missing & ~true):
        raise RuntimeError('observed target physical read not in the native route')
    families = {}
    for feature,name in enumerate(('direct','transferred_bias','transferred_ridge')):
        sc = scores[feature]
        top = np.argsort(-sc,axis=-1,kind='stable')[...,:6].copy()
        values = np.take_along_axis(sc,top,axis=-1)
        gap = values-values[...,-1:]
        configs = []
        for width in (1,2,3,4,6):
            for margin in (0.,.025,.05,.1,.2):
                keep = gap[...,:width] >= margin
                cc,ll,rr,kk = np.nonzero(keep)
                confidence = np.full((cycles,layers,experts),-np.inf,np.float32)
                np.maximum.at(confidence,(cc,ll,top[cc,ll,rr,kk]),gap[cc,ll,rr,kk])
                confidence[physical] = -np.inf
                rank = np.argsort(-confidence,axis=-1,kind='stable')
                for budget in (4,8,12):
                    ids = rank[...,:budget]
                    issued = np.zeros_like(true)
                    valid = np.isfinite(np.take_along_axis(confidence,ids,axis=-1))
                    np.put_along_axis(issued,ids,valid,axis=-1)
                    hit = issued & missing
                    extra = issued & ~missing
                    stats = {}
                    for phase,sl in (('train',slice(0,32)),('heldout',slice(32,64))):
                        a = issued[sl].sum(axis=(0,2)); h = hit[sl].sum(axis=(0,2))
                        e = extra[sl].sum(axis=(0,2)); m = missing[sl].sum(axis=(0,2))
                        stats[phase] = [dict(layer=i+4,issued=int(a[i]),useful=int(h[i]),extra=int(e[i]),
                            actual_misses=int(m[i]),precision=float(h[i]/a[i]) if a[i] else None,
                            miss_coverage=float(h[i]/m[i]) if m[i] else 0.) for i in range(layers)]
                    configs.append(dict(width=width,margin=margin,max_records=budget,stats=stats))
        chosen=[]
        for layer in range(layers):
            eligible=[v for v in configs if v['stats']['train'][layer]['issued']>=8
                      and v['stats']['train'][layer]['precision']>=.85]
            if not eligible:
                continue
            best=max(eligible,key=lambda v:(v['stats']['train'][layer]['useful'],
                       -v['stats']['train'][layer]['extra'],-v['max_records'],-v['width']))
            chosen.append({'layer':layer+4,'config':{k:best[k] for k in ('width','margin','max_records')},
                           'train':best['stats']['train'][layer],
                           'heldout':best['stats']['heldout'][layer]})
        totals={}
        for phase,sl in (('train',slice(0,32)),('heldout',slice(32,64))):
            total={k:sum(x[phase][k] for x in chosen) for k in ('issued','useful','extra')}
            total['actual_misses_all_target_layers']=int(missing[sl].sum())
            total['precision']=total['useful']/total['issued'] if total['issued'] else None
            total['miss_coverage']=total['useful']/total['actual_misses_all_target_layers']
            total['added_traffic_fraction']=total['extra']/total['actual_misses_all_target_layers']
            totals[phase]=total
        families[name]={'selected_layers':chosen,'totals':totals,'configs':configs}
    return {'calibration':'First32 cycles choose each layer configuration at >=85% physical-miss precision and >=8 issued records; evaluate unchanged on next32.',
            'scope':'Unlimited-lead-time upper bound for missing records predicted one layer ahead; excludes currently READY physical records. No speculative reads issued; no latency claim.',
            'physical_filter_note':'Snapshot at prediction time; current-layer demand may subsequently replace shared transient owners.',
            'actual_reads_total':int(reads[:,4:].sum()),'duplicate_read_records':int((reads[:,4:]>1).sum()),
            'families':families}
