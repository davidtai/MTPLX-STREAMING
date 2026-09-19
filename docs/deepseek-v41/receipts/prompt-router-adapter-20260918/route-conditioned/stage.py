"""Add the already-known current expert route to the transferred predictor."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'transfer'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proof=json.loads((BASE/'installation.json').read_text())
assert sha(BASE/'screen.py')==proof['script_sha256']
assert sha(BASE/'analysis.py')==proof['analysis_sha256']
s=(BASE/'analysis.py').read_text().replace("('direct','transferred_bias','transferred_ridge')",
    "('direct','transferred_ridge','route_conditioned_ridge')")
(ROOT/'analysis.py').write_text(s)
s=(BASE/'screen.py').read_text()
s=s.replace('gram.flat[::385]', 'gram.flat[::gram.shape[0]+1]')
s=s.replace("prev=hidden(3,'router_in')", "prev=hidden(3,'router_in')\nprev_routes=hidden(3,'top6')")
needle='    predictions={\'direct\':baseline[2048:],'
start=s.index(needle)
end=s.index('    exact=exact_scores',start)
s=s[:start]+'''    route_features=np.zeros_like(z)
    np.put_along_axis(route_features,prev_routes.astype(np.int64),np.float32(1),axis=1)
    extended=np.concatenate([z,route_features],axis=1)
    route_choices=[]
    for lam in (0.1,1.0):
        candidate=fit(extended[fit_range],residual[fit_range],lam)
        prediction=baseline[validation]+correct(extended[validation],candidate)
        scored=metrics(prediction,truth[validation])['6']
        route_choices.append((scored['hits'],-float(np.mean((prediction-target[validation])**2)),lam))
    route_chosen=max(route_choices)[2]
    route_model=fit(extended[1024:2048],residual[1024:2048],route_chosen)
    predictions={'direct':baseline[2048:],
        'prompt_ridge':baseline[2048:]+correct(z[2048:],model),
        'route_ridge':baseline[2048:]+correct(extended[2048:],route_model)}
''' + s[end:]
old='''    exact_scores[1,:,layer-4]=(exact+residual[1024:2048].mean(axis=0)).reshape(64,6,384)
    exact_scores[2,:,layer-4]=(exact+correct(exact_z,model)).reshape(64,6,384)'''
new='''    current_route=exact_actual[:,layer-1].reshape(-1,6)
    exact_route_features=np.zeros_like(exact_z)
    np.put_along_axis(exact_route_features,current_route.astype(np.int64),np.float32(1),axis=1)
    exact_extended=np.concatenate([exact_z,exact_route_features],axis=1)
    exact_scores[1,:,layer-4]=(exact+correct(exact_z,model)).reshape(64,6,384)
    exact_scores[2,:,layer-4]=(exact+correct(exact_extended,route_model)).reshape(64,6,384)'''
assert s.count(old)==1
s=s.replace(old,new)
s=s.replace("'adapter_bytes':sum(v.nbytes for v in model),", "'adapter_bytes':sum(v.nbytes for v in model),\n        'route_adapter_bytes':sum(v.nbytes for v in route_model),\n        'route_selected_ridge':route_chosen,'route_validation_choices':route_choices,\n        'route_adapter_sha256':hashlib.sha256(b''.join(v.tobytes() for v in route_model)).hexdigest(),")
s=s.replace('    prev=current\n', '    prev=current\n    prev_routes=truth\n')
s=s.replace("'all_layer_adapter_bytes':sum(r['adapter_bytes'] for r in rows),", "'all_layer_adapter_bytes':sum(r['adapter_bytes'] for r in rows),\n    'all_layer_route_adapter_bytes':sum(r['route_adapter_bytes'] for r in rows),")
(ROOT/'screen.py').write_text(s)
proof['script_sha256']=sha(ROOT/'screen.py')
proof['analysis_sha256']=sha(ROOT/'analysis.py')
proof['scope']='Prompt-only training from independent W35 trace. Add current-layer top6 indicators to next-gate score features; transfer to exact M6 capture using current-layer routes already known at prediction time. Exact labels only calibrate read issuance on first32 cycles, then score reused heldout32. No target run, prefetch or TPS.'
proof['prediction']='Affine residual from384 standardized next-gate raw logits plus384 current-layer route indicators. The sparse route term can be applied by summing six prebound coefficient rows; target routing and outputs stay unchanged.'
proof['bound']+=' Extended768-column solves and route indicator arrays add less than64MiB inside the1GiB host envelope.'
proof['parent_screen_sha256']=sha(BASE/'screen.py')
proof['heldout_reused_across_research_iterations']=True
(ROOT/'installation.json').write_text(json.dumps(proof,indent=2)+'\n')
for p in ROOT.glob('*.py'):ast.parse(p.read_text())
(ROOT/'command.sh').write_text((BASE/'command.sh').read_text().replace(str(BASE),str(ROOT)))
print(json.dumps({'source':proof['source_commit'],'static_incremental_bound_bytes':proof['static_incremental_bound_bytes'],'scope':proof['scope']}))
