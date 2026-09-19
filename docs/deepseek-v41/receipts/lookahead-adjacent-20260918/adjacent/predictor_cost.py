"""Pay real gate and ranking costs while replaying saved prediction scores.

The gate sees synthetic inputs. Its results are evaluated but do not choose
reads. This is a cost/scheduling screen, not a live predictor parity result.
"""
import fcntl, hashlib, json, struct
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
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
    return gates,identities


class Issue:
    def __init__(self,runtime,target,gate,config,scores):
        self.runtime,self.target=runtime,target
        self.weight,self.bias,self.prefix=gate
        self.width=config['width'];self.margin=float(np.float32(config['margin']))
        self.limit=config['max_records'];self.scores=scores;self.call=0
        self.ranked=()
        self.slots=tuple(runtime.slots._persistent.values())+runtime.slots._transient+tuple(runtime.slots._prefetch.values())

    def prepare(self,tokens,indices):
        # Evaluate the real native prefix alongside selection from recorded
        # scores, piggybacking on the source route's existing evaluation barrier.
        live=self.prefix(tokens,self.weight,self.bias)[1]
        biased=self.scores[self.call]
        part=mx.argpartition(-biased,kth=5,axis=-1)[...,:6].astype(mx.int32)
        values=mx.take_along_axis(biased,part,axis=-1)
        order=mx.argsort(-values,axis=-1)
        ranked=mx.take_along_axis(part,order,axis=-1)
        values=mx.take_along_axis(values,order,axis=-1)
        gaps=values-values[...,-1:]
        mx.eval(indices,live,ranked,gaps)
        candidates={}
        for ids,differences in zip(ranked.tolist(),gaps.tolist()):
            for expert,gap in zip(ids[:self.width],differences[:self.width]):
                if gap>=self.margin:candidates[expert]=max(candidates.get(expert,float('-inf')),gap)
        self.ranked=sorted(candidates,key=lambda expert:(-candidates[expert],expert))

    def __call__(self):
        ready={s.expert for s in self.slots if s.state is ExpertSlotState.READY and s.layer==self.target}
        self.runtime.prefetch_experts(self.target,[e for e in self.ranked if e not in ready][:self.limit],verify=True)
