"""Pay real gate, ridge correction and ranking costs while replaying saved prediction scores.

The gate sees synthetic inputs. Its results are evaluated but do not choose
reads. This is a cost/scheduling screen, not a live predictor parity result.
"""
import fcntl, hashlib, json, struct
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from mtplx.models.deepseek_v41_moe import _gate_prefix
from mtplx.expert_slots import ExpertSlotState


def load_gates(model,proof):
    cfg_bytes=(model/'config.json').read_bytes()
    index_bytes=(model/'model.safetensors.index.json').read_bytes()
    assert hashlib.sha256(cfg_bytes).hexdigest()==proof['gate_config_sha256']
    assert hashlib.sha256(index_bytes).hexdigest()==proof['gate_index_sha256']
    cfg=json.loads(cfg_bytes);cfg=cfg.get('text_config',cfg)
    index=json.loads(index_bytes)['weight_map'];identities={}

    def tensor(name):
        with open(model/index[name],'rb',buffering=0) as f:
            fcntl.fcntl(f.fileno(),48,1)
            size=struct.unpack('<Q',f.read(8))[0]
            assert size<64*1024**2
            entry=json.loads(f.read(size))[name];lo,hi=entry['data_offsets']
            assert hi-lo<=8*1024**2
            f.seek(8+size+lo);raw=f.read(hi-lo)
            assert len(raw)==hi-lo
        identities[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'dtype':entry['dtype'],'shape':entry['shape']}
        if entry['dtype']=='BF16':a=mx.array(np.frombuffer(raw,np.uint16)).view(mx.bfloat16)
        elif entry['dtype']=='F32':a=mx.array(np.frombuffer(raw,np.float32))
        else:raise RuntimeError(entry['dtype'])
        return a.reshape(entry['shape'])

    prefix=_gate_prefix(SimpleNamespace(score_func=cfg.get('scoring_func','sqrtsoftplus'),gate_temp=float(cfg.get('gate_temp',1.) or 1.)))
    gates={l:(tensor(f'layers.{l}.ffn.gate.weight'),tensor(f'layers.{l}.ffn.gate.bias'),prefix) for l in (31,32)}
    for weight,bias,_ in gates.values():
        assert weight.shape==(384,5120) and bias.shape==(384,)
        mx.eval(weight,bias)
    prepared = json.loads((Path(__file__).resolve().parent/'preparation.json').read_text())
    parameter_path = Path(__file__).resolve().parent/'ridge-parameters.npz'
    if not prepared['complete'] or hashlib.sha256(parameter_path.read_bytes()).hexdigest()!=prepared['parameters_sha256']:
        raise RuntimeError('ridge parameter preparation changed')
    if cfg.get('scoring_func','sqrtsoftplus')!='sqrtsoftplus':
        raise RuntimeError('ridge prefix requires the measured native scoring function')
    temp=float(cfg.get('gate_temp',1.) or 1.)
    def ridge_prefix(x,weight,bias,mean,scale,center,coef):
        z=(x.astype(mx.float32)@weight.astype(mx.float32).T)/temp
        return mx.sqrt(nn.softplus(z))+bias+(((z-mean)/scale)@coef+center)
    compiled=mx.compile(ridge_prefix)
    with np.load(parameter_path,allow_pickle=False) as arrays:
        for layer in (31,32):
            values=[arrays[f'layer{layer}_{name}'] for name in ('mean','scale','center','coef')]
            if [v.shape for v in values]!=[(384,),(384,),(384,),(384,384)] or any(v.dtype!=np.float32 for v in values):
                raise RuntimeError('ridge adapter geometry changed')
            params=tuple(mx.array(v) for v in values)
            weight,bias,_=gates[layer]
            gates[layer]=(weight,bias,compiled,params)
            mx.eval(params)
    identities['ridge_parameters']={'sha256':prepared['parameters_sha256'],'bytes':prepared['parameters_bytes']}
    return gates,identities


class Issue:
    def __init__(self,runtime,target,gate,config,scores):
        self.runtime,self.target=runtime,target
        self.weight,self.bias,self.prefix,self.adapter=gate
        self.width=config['width'];self.margin=float(np.float32(config['margin']))
        self.limit=config['max_records'];self.scores=scores;self.call=0
        self.ranked=()
        self.slots=tuple(runtime.slots._persistent.values())+runtime.slots._transient+tuple(runtime.slots._prefetch.values())

    def prepare(self,tokens,indices):
        # Evaluate the real native prefix alongside selection from recorded
        # scores, piggybacking on the source route's existing evaluation barrier.
        live=self.prefix(tokens,self.weight,self.bias,*self.adapter)
        mx.eval(indices,live)
        biased=np.asarray(self.scores[self.call])
        part=np.argpartition(-biased,kth=5,axis=-1)[...,:6]
        values=np.take_along_axis(biased,part,axis=-1)
        order=np.argsort(-values,axis=-1,kind='stable')
        ranked=np.take_along_axis(part,order,axis=-1)
        values=np.take_along_axis(values,order,axis=-1)
        gaps=values-values[...,-1:]
        candidates={}
        for ids,differences in zip(ranked.tolist(),gaps.tolist()):
            for expert,gap in zip(ids[:self.width],differences[:self.width]):
                if gap>=self.margin:candidates[expert]=max(candidates.get(expert,float('-inf')),gap)
        self.ranked=sorted(candidates,key=lambda expert:(-candidates[expert],expert))

    def __call__(self):
        ready={s.expert for s in self.slots if s.state is ExpertSlotState.READY and s.layer==self.target}
        self.runtime.prefetch_experts(self.target,[e for e in self.ranked if e not in ready][:self.limit],verify=True)
