"""Stage a source-pinned three-layer cost screen without MLX imports."""
from pathlib import Path
import array, ast, fcntl, gzip, hashlib, json, shutil, struct, subprocess, zipfile

ROOT=Path(__file__).resolve().parent
BASE=Path('/tmp/dsv41-lookahead-io-20260918')
REPO=Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
CAPTURE=REPO/'.benchmark-artifacts/deepseek-v41/router-feature-20260918/full-router-feature-20260918-v2.router-capture.npz'
for name in ('packed_storage.py','paired_kernels.py','kernels.py','restore_bank.py','library_identity.py','run_screen.py'):
    shutil.copyfile(BASE/name,ROOT/name)
(ROOT/'artifact').symlink_to(BASE/'artifact',target_is_directory=True)
config=(BASE/'paired_config.py').read_text().replace('two-layer','three-layer').replace('two-layer','three-layer')
config=config.replace('7*1024**3','10*1024**3').replace('210*18800640','315*18800640')
(ROOT/'paired_config.py').write_text(config)
lane=(BASE/'plane_lane.py').read_text()
start=lane.index('class PrefetchDecode(')
lane=lane[:start]+lane[start:].replace('        mx.eval(indices)','        self.issue.prepare(tokens,indices)',1)
lane=lane.replace('prefetch_source=None','prefetch_sources=None')
lane=lane.replace('if prefetch_source is not None and layer == prefetch_source[0]:','if prefetch_sources is not None and layer in prefetch_sources:')
lane=lane.replace('issue=prefetch_source[1]','issue=prefetch_sources[layer]')
(ROOT/'plane_lane.py').write_text(lane)

with gzip.open(REPO/'docs/deepseek-v41/receipts/mtp-verify-routes-20260913/mtp-verify-routes-16k-1024-v2.json.gz','rt') as f:
    trace=json.load(f)
old=json.loads((BASE/'routes.json').read_text())
data={k:v for k,v in old.items() if k not in ('predictions','predictor_config','selection')}
data.update(layers=[30,31,32],routes={str(l):trace['target_routes_by_layer'][str(l)][:64] for l in (30,31,32)},
    initial_banks={str(l):trace['initial_banks'][str(l)] for l in (30,31,32)},
    predictor_config={'31':{'width':4,'margin':.1,'max_records':8},'32':{'width':6,'margin':.2,'max_records':8}},
    selection='Training-selected target31 and adjacent target32; no held-out layer selection.',captured_scores={})
with CAPTURE.open('rb',buffering=0) as source:
    fcntl.fcntl(source.fileno(),48,1)
    with zipfile.ZipFile(source) as z, z.open('scores.npy') as f:
        assert f.read(6)==b'\x93NUMPY'
        v=f.read(2);n=struct.unpack('<H' if v==b'\x01\x00' else '<I',f.read(2 if v==b'\x01\x00' else 4))[0]
        h=ast.literal_eval(f.read(n).decode());base=f.tell()
        assert h['shape']==(2,64,36,6,384) and h['descr']=='<f4'
        for target in (31,32):
            rows=[]
            for cycle in range(64):
                f.seek(base+(((64+cycle)*36+target-4)*6)*384*4)
                a=array.array('f');a.frombytes(f.read(6*384*4))
                rows.append([a[r*384:(r+1)*384].tolist() for r in range(6)])
            data['captured_scores'][str(target)]=rows
(ROOT/'routes.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
p=json.loads((BASE/'installation.json').read_text())
raw=18800640; limit=10*1024**3; persist=315*raw
p.update(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),layers=[30,31,32],
    bank_capacity=315+48+16,static_incremental_bound_bytes=14*1024**3,
    scope='Three adjacent real expert layers; continuous held-out timing includes final drain and native gate compute on synthetic BF16 inputs. Ranked predictions use captured exact-workload scores; this prices computation but is not live predictor parity. No attention or full-model TPS claim.',
    helper_sha256={})
p.pop('paired_capability',None)
p['spec'].update(routed_layer_start=30,routed_layer_count=3,total_tensor_bytes=3*384*raw+4096)
p['plan'].update(total_limit_bytes=limit,expert_cache_limit_bytes=persist,persistent_budget_bytes=persist,
    persistent_slots=315,persistent_cache_bytes=persist,unallocated_bytes=limit-persist-48*raw-16*raw-4096)
p['config'].update(memory_limit_bytes=limit,expert_cache_limit_bytes=persist)
p['budget_components']={'metal_cache_compiler_bytes':10*1024**3,'host_reader_compiler_bytes':4*1024**3,
    'raw_banks_bytes':379*raw,'packed_banks_bytes':379*17694720,'retained_output_bytes':64*3*6*6*5120*2,
    'note':'Raw bank accounting dominates packed allocation plus resident scale owners; remaining Metal envelope prices outputs, gates, workspaces, cache and compilation.'}
for name,digest in p['runtime_source_sha256'].items():assert hashlib.sha256((REPO/name).read_bytes()).hexdigest()==digest,name
for name in ('mtplx/models/deepseek_v41_moe.py','mtplx/expert_streaming.py'):
    p['runtime_source_sha256'][name]=hashlib.sha256((REPO/name).read_bytes()).hexdigest()
model=Path(p['model_path'])
p['gate_config_sha256']=hashlib.sha256((model/'config.json').read_bytes()).hexdigest()
p['gate_index_sha256']=hashlib.sha256((model/'model.safetensors.index.json').read_bytes()).hexdigest()
(ROOT/'installation.json').write_text(json.dumps(p,indent=2)+'\n')
cmd=(BASE/'command.sh').read_text().replace(str(BASE),str(ROOT)).replace('11811160064',str(14*1024**3))
(ROOT/'command.sh').write_text(cmd)
print(json.dumps({'root':str(ROOT),'budget_components':p['budget_components'],'source':p['source_commit']}))
