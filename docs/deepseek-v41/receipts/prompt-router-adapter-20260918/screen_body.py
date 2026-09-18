# Appended after the audited CPU-only trace and native gate tensor readers.
from mtplx.expert_streaming import LayerExpertSlotBank
import subprocess

installation=json.loads((ROOT/'installation.json').read_text())
if subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()!=installation['source_commit']:
    raise RuntimeError('source commit differs from CPU screen')
if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=installation['script_sha256']:
    raise RuntimeError('CPU screen source identity differs')
for name,expected in installation['runtime_sources'].items():
    if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('runtime source differs')
if hashlib.sha256((TRACE/'manifest.json').read_bytes()).hexdigest()!=installation['source_manifest_sha256']:
    raise RuntimeError('trace manifest differs')


def scores_from_z(z,bias):
    return np.sqrt(np.logaddexp(np.float32(0),z))+bias


def fit(z,y,lam):
    mean=z.mean(axis=0)
    scale=np.maximum(z.std(axis=0),np.float32(0.1))
    center=y.mean(axis=0)
    design=(z-mean)/scale
    gram=design.T@design
    gram.flat[::385]+=np.float32(lam*len(z))
    coef=np.linalg.solve(gram,design.T@(y-center))
    return mean,scale,center,coef


def correct(z,model):
    mean,scale,center,coef=model
    return ((z-mean)/scale)@coef+center


def proxy_misses(predictions,truth,layer):
    # Causal native policy replay of the captured AR rows. This is not the
    # M6 cache state, complete original prefill, or a prefetch timing simulation.
    bank=LayerExpertSlotBank(expert_count=384,persistent_slots=110,transient_slots=48,
        single_pool=True,cache_policy='transition-window',layer_id=layer)
    for route in truth[:2048]:
        bank.plan((int(v) for v in route),phase='prefill')
    ranked={name:np.argsort(-score,axis=1,kind='stable') for name,score in predictions.items()}
    result={name:{str(k):{'issued':0,'hits':0,'misses':0} for k in (1,2,4,6,10)}
            for name in predictions}
    for i,route in enumerate(truth[2048:]):
        required=set(int(v) for v in route)
        resident=set(bank._expert_to_slot)
        missing=required-resident
        for name,rank in ranked.items():
            for k in (1,2,4,6,10):
                selected=set(int(v) for v in rank[i,:k])-resident
                record=result[name][str(k)]
                record['issued']+=len(selected)
                record['hits']+=len(selected & missing)
                record['misses']+=len(missing)
        plan=bank.plan((int(v) for v in route),phase='decode')
        if set(plan.misses)!=missing:
            raise RuntimeError('cache proxy miss census differs from the native plan')
    return result


started=time.monotonic()
rows=[]
prev=hidden(3,'router_in')
for layer in range(4,40):
    current=hidden(layer,'router_in')
    truth=hidden(layer,'top6')
    weight=tensor(f'layers.{layer}.ffn.gate.weight')
    bias=tensor(f'layers.{layer}.ffn.gate.bias')
    z=(prev@weight.T)/temp
    target_z=(current@weight.T)/temp
    baseline=scores_from_z(z,bias)
    target=scores_from_z(target_z,bias)
    residual=target-baseline
    fit_range=slice(1024,1792)
    validation=slice(1792,2048)
    choices=[]
    for lam in (0.1,1.0):
        model=fit(z[fit_range],residual[fit_range],lam)
        prediction=baseline[validation]+correct(z[validation],model)
        scored=metrics(prediction,truth[validation])['6']
        choices.append((scored['hits'],-float(np.mean((prediction-target[validation])**2)),lam))
    chosen=max(choices)[2]
    model=fit(z[1024:2048],residual[1024:2048],chosen)
    predictions={'direct':baseline[2048:],
        'prompt_bias':baseline[2048:]+residual[1024:2048].mean(axis=0),
        'prompt_ridge':baseline[2048:]+correct(z[2048:],model)}
    result={name:metrics(score,truth[2048:]) for name,score in predictions.items()}
    result['self_alignment']=metrics(target[2048:],truth[2048:])
    miss=proxy_misses(predictions,truth,layer)
    rows.append({'layer':layer,'selected_ridge':chosen,'validation_choices':choices,
        'metrics':result,'miss_proxy':miss,
        'adapter_bytes':sum(v.nbytes for v in model),
        'fitted_adapter_sha256':hashlib.sha256(b''.join(v.tobytes() for v in model)).hexdigest()})
    prev=current
    if layer%8==7:
        print(json.dumps({'completed_through_layer':layer,'elapsed_s':time.monotonic()-started}),flush=True)
summary={}
for name in rows[0]['metrics']:
    summary[name]={}
    for k in rows[0]['metrics'][name]:
        counts={key:sum(r['metrics'][name][k][key] for r in rows) for key in ('hits','issued','truth')}
        counts.update(precision=counts['hits']/counts['issued'],recall=counts['hits']/counts['truth'])
        summary[name][k]=counts
miss_summary={}
for name in rows[0]['miss_proxy']:
    miss_summary[name]={}
    for k in rows[0]['miss_proxy'][name]:
        counts={key:sum(r['miss_proxy'][name][k][key] for r in rows) for key in ('hits','issued','misses')}
        counts.update(precision=counts['hits']/max(1,counts['issued']),coverage=counts['hits']/max(1,counts['misses']))
        miss_summary[name][k]=counts
alignment=summary['self_alignment']['6']['recall']
report={'complete':True,'cpu_only':True,'source_commit':installation['source_commit'],
    'scope':installation['scope'],'training':installation['training'],'static_incremental_bound_bytes':1024**3,
    'before':before,'after':host_memory_snapshot(),'elapsed_s':time.monotonic()-started,
    'native_self_alignment':alignment,'cpu_alignment_usable':alignment>.995,
    'same_prompt_as_acceptance':False,'decode_labels_used_for_training_or_selection':False,
    'all_layer_adapter_bytes':sum(r['adapter_bytes'] for r in rows),
    'summary':summary,'miss_proxy_summary':miss_summary,'rows':rows,'consumed_range_sha256':digests}
OUT.write_text(json.dumps(report,indent=2)+'\n')
print('PROMPT_ROUTER_ADAPTER',json.dumps({k:report[k] for k in ('complete','elapsed_s','native_self_alignment','all_layer_adapter_bytes','miss_proxy_summary')}),flush=True)
