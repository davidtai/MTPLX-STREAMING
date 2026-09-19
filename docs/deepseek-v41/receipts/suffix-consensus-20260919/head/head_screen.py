"""Guarded draft-only acceptance screen using exact captured target states.

This never executes the target trunk and cannot measure full decode throughput.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

if os.environ.get('_GPU_WINDOW_LOCKED')!='1':
    raise RuntimeError('parent-held GPU/service guard required before MLX')
ROOT=Path('/tmp/dsv41-depth-replay-20260917')
OUT=Path('/private/tmp/dsv41-q4-suffix-consensus-20260919/head/head-screen.json')
if OUT.exists(): raise RuntimeError('refusing to overwrite draft screen')
TEACHER_PATH=ROOT/'teacher.json'
teacher=json.loads(TEACHER_PATH.read_text())
if (teacher['schema']!='dsv41-exact-target-teacher-v2' or teacher['prompt_tokens']!=16384
    or teacher['token_ids_sha256']!='0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac'
    or len(teacher['token_ids'])!=1024 or teacher['target_decode_width']!=6):
    raise RuntimeError('teacher is not the complete validated native trajectory')
reference_path=Path('/tmp/dsv41-110-stage/teacher-cap84-7537-v3-20260917.jsonl')
reference=json.loads(reference_path.read_text())
if (reference.get('aborted') or reference['dspark']['cycles']!=teacher['cycles']
    or reference['teacher_trace']['sha256']!=hashlib.sha256(TEACHER_PATH.read_bytes()).hexdigest()):
    raise RuntimeError('teacher completion receipt does not match')
CURRENT_ROOT=Path('/private/tmp/dsv41-q4-suffix-consensus-20260919/head')
installation=json.loads((CURRENT_ROOT/'installation.json').read_text())
if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=installation['source_commit']:
    raise RuntimeError('source commit differs from pinned head screen')
for path,expected in installation['runtime_source_sha256'].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('native source differs from head-screen audit')
for key,value in reference['arm_env'].items():
    if value is None: os.environ.pop(key,None)
    else: os.environ[key]=str(value)
for key in ('MTPLX_DSV41_HC_COMPILE','MTPLX_DSV41_ATTN_COMPILE','MTPLX_DSV41_ATTN_WIN_MEMO'):
    os.environ[key]='0'
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
GIB=1024**3
# Static allocation inventory: <=18GiB eager text residents, <=11GiB combined
# full/compact MTP stacks, <=3GiB current shard, <=1GiB teacher/layout copies,
# <=8GiB conservative native attention/head/workspace/compile allowance.
# Initial random parameter graphs must be entirely replaced before evaluation.
# T<=13 keeps resident top3 SwitchGLU on its existing unsorted gather M=1 path.
ACTIVE_BOUND=41*GIB
CACHE_ALLOWANCE=4*GIB
HOST_ALLOWANCE=4*GIB
snap=host_memory_snapshot()
if (snap['box']['used_bytes']+ACTIVE_BOUND+CACHE_ALLOWANCE+HOST_ALLOWANCE>110000000000
    or snap['box']['wired_bytes']+ACTIVE_BOUND+CACHE_ALLOWANCE>100*GIB):
    raise RuntimeError('draft-only static allocation bound does not fit current baseline')
import mlx.core as mx
import mlx.nn as nn
from library_identity import identify
library_identity=identify(installation["strict_allocator"])
from mlx.utils import tree_flatten
from mlx_lm.models.switch_layers import SwitchGLU
import numpy as np
from mtplx.expert_manifest import load_expert_manifest
from mtplx.models import deepseek_v41 as dv
from mtplx.models import deepseek_v41_dspark as ds
from mtplx.models import deepseek_v41_loader as loader
from dataclasses import replace
signal.alarm(600)
mx.set_memory_limit(48*GIB)
mx.set_cache_limit(GIB)
dv._HC_COMPILE=False
dv._ATTN_COMPILE=False
dv._ATTN_WIN_MEMO=False
ds._DRAFT_COMPILE=True
ds._DRAFT_HEAD_BF16=True
ARTIFACT=Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
manifest_path=ARTIFACT/'expert-manifest.json'
if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!='44c340845990b4ec21342d914204d69eb9e7cf371d6b18944f8dfd2e945df3e9':
    raise RuntimeError('native manifest changed')
manifest=load_expert_manifest(manifest_path)
partition=loader.partition_text_residents(manifest,with_mtp=True)
if partition.kept_bytes>18*GIB:
    raise RuntimeError('text resident payload exceeds the static inventory')
config=json.loads((ARTIFACT/'config.json').read_text())
args=dv.ModelArgs.from_dict(config)
if (args.hidden_size!=5120 or args.moe_intermediate_size!=2304 or args.dspark_block_size!=5
    or ds.n_mtp_layers(args)!=3 or ds.dspark_target_layer_ids(args)!=(37,38,39)):
    raise RuntimeError('native draft geometry differs')
compact_meta=json.loads(Path('/tmp/dsv41-compact-residents/receipt.json').read_text())
selected=tuple(tuple(x) for x in compact_meta['selected_experts_by_stage'])
if tuple(map(len,selected))!=(93,58,32): raise RuntimeError('compact inventory changed')
LUTS=[]
for ids in selected:
    positions={expert:slot for slot,expert in enumerate(ids)}
    LUTS.append(mx.array([positions.get(e,0) for e in range(128)],dtype=mx.int32))
class CompactSwitch(SwitchGLU):
    def __init__(self,stage,activation):
        super().__init__(args.hidden_size,args.moe_intermediate_size,len(selected[stage]),activation=activation,bias=False)
        self._compact_stage=stage
    def __call__(self,x,indices):
        return super().__call__(x,mx.take(LUTS[self._compact_stage],indices))
class DraftOnly(nn.Module):
    def __init__(self,width,compact):
        super().__init__()
        self.args=replace(args,dspark_block_size=width)
        self.model=nn.Module()
        self.model.embed_tokens=nn.Embedding(args.vocab_size,args.hidden_size)
        self.head=nn.Linear(args.hidden_size,args.vocab_size,bias=False)
        self.mtp=ds.DSparkHead(self.args)
        if compact:
            for stage,block in enumerate(self.mtp.layers):
                block.mlp.switch_mlp=CompactSwitch(stage,block.mlp.switch_mlp.activation)
        nn.quantize(self.mtp,group_size=32,bits=8,mode='mxfp8',class_predicate=dv._make_mtp_dense_quant_predicate(32))
        nn.quantize(self.mtp,group_size=32,bits=4,mode='mxfp4',class_predicate=dv._make_mtp_expert_quant_predicate(32))
        self.eval()
def digest_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''): h.update(block)
    return h.hexdigest()
def load_teacher(name):
    info=teacher['files'][name];path=ROOT/name
    if path.stat().st_size!=info['bytes'] or digest_file(path)!=info['sha256']:
        raise RuntimeError('teacher data identity changed')
    data=np.load(path,allow_pickle=False)
    if list(data.shape)!=info['shape'] or str(data.dtype)!=info['dtype']:
        raise RuntimeError('teacher storage geometry differs')
    arr=mx.array(data)
    if info['mlx_dtype']=='bfloat16':
        if data.dtype!=np.uint16: raise RuntimeError('BF16 teacher is not stored as raw uint16')
        arr=arr.view(mx.bfloat16)
    elif info['mlx_dtype']!='float32' or data.dtype!=np.float32:
        raise RuntimeError('teacher dtype is not supported')
    return arr
initial_main=load_teacher('initial-main.npy')
initial_windows=[load_teacher(f'initial-window-{stage}.npy') for stage in range(3)]
hidden=load_teacher('committed-hidden.npy')
mx.eval(initial_main,initial_windows,hidden)
print('DRAFT_SCREEN_LOADING',json.dumps({'text_resident_payload_bytes':partition.kept_bytes,'static_active_bound_bytes':ACTIVE_BOUND,'host_allowance_bytes':HOST_ALLOWANCE,'cache_allowance_bytes':CACHE_ALLOWANCE}),flush=True)
all_raw=loader.load_text_only_resident_arrays(ARTIFACT,manifest,mx_module=mx,partition=partition)
raw={name:value for name,value in all_raw.items() if name.startswith('mtp.') or name in ('embed.weight','head.weight')}
del all_raw
gc.collect();mx.synchronize();mx.clear_cache()
if not {'embed.weight','head.weight'}.issubset(raw): raise RuntimeError('native shared embeddings are absent')
expert_re=re.compile(r'mtp\.(\d+)\.ffn\.experts\.(\d+)\.')
compact_raw={}
for name,value in raw.items():
    match=expert_re.match(name)
    if match is None or int(match[2]) in selected[int(match[1])]: compact_raw[name]=value
stem={dv._sanitize_name(name):value for name,value in raw.items() if not name.startswith('mtp.')}
weights={'compact':{**stem,**dv._map_mtp_residents(compact_raw)},'full':{**stem,**dv._map_mtp_residents(raw)}}
if sum(int(a.nbytes) for k in ('compact','full') for n,a in weights[k].items() if '.switch_mlp.' in n)>11*GIB:
    raise RuntimeError('stacked expert payload exceeds static inventory')
mx.eval(weights)
del raw,compact_raw,stem,manifest,partition
# No eager source tensors or unevaluated initial random graphs survive install.
gc.collect();mx.synchronize();mx.clear_cache()
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
