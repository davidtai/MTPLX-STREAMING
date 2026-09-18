"""Temporary, bounded observation of causal native router predictions.

Installed after prefill/growth; removed after 64 native verify cycles. Never
issues reads, changes a route, updates cache policy or changes native results.
Extra fences make this a diagnostic, never throughput evidence.
"""
import hashlib
import json
from pathlib import Path
import weakref

import numpy as np
import mlx.core as mx
from mtplx.models import deepseek_v41 as dv
from mtplx.models import deepseek_v41_moe as moe
from mtplx.expert_slots import ExpertSlotState

CYCLES = 64
FIRST, LAST, ROWS, EXPERTS = 4, 39, 6, 384
FEATURES = ('existing_pre_attention_mean', 'post_attention_router')


class Link:
    __slots__ = ('owner', 'layer')
    def __init__(self, owner, layer):
        self.owner, self.layer = weakref.ref(owner), layer


class ObservedLayer(dv.DecoderLayer):
    def attn_and_moe_input(self, h, pre_mix, positions, layer_cache, shared):
        link = self._router_capture
        owner = link.owner()
        owner.incoming[link.layer] = h
        return dv.DecoderLayer.attn_and_moe_input(
            self, h, pre_mix, positions, layer_cache, shared)


class ObservedGate(moe.Gate):
    def __call__(self, x, image_mask=None):
        native = moe.Gate.__call__(self, x, image_mask=image_mask)
        link = self._router_capture
        link.owner().observe(link.layer, x, native[1])
        return native


class Capture:
    def __init__(self, model, prefix):
        layers = model.model.layers
        rt = model._mtplx_expert_runtime
        if len(layers) != 40 or not 84 < rt.plan.slots_per_layer <= 105:
            raise RuntimeError('diagnostic capacity/geometry differs')
        if any(type(x) is not dv.DecoderLayer or type(x.mlp.gate) is not moe.Gate for x in layers):
            raise RuntimeError('native layer/gate types required')
        if dv._small_stages_fused_enabled() or dv._SMALL_STAGES_FUSED_CALLS:
            raise RuntimeError('capture requires the native unfused decoder stages')
        compile_modes = {moe._attn_compile_gate(n) for n in range(1, ROWS+1)}
        if len(compile_modes) != 1:
            raise RuntimeError('prediction prefix differs across admitted row counts')
        self.prefix = Path(prefix)
        self.capacity = rt.plan.slots_per_layer
        self.runtime = weakref.ref(rt)
        self.layers = tuple(weakref.ref(x) for x in layers)
        self.incoming = {}
        self.counts = [-1]*40
        self.restored = False
        self.scores = np.zeros((2,CYCLES,LAST-FIRST+1,ROWS,EXPERTS),np.float32)
        self.actual = np.full((CYCLES,40,ROWS,6),-1,np.int32)
        self.nrows = np.zeros((CYCLES,40),np.uint8)
        self.persistent = np.zeros((CYCLES,LAST-FIRST+1,EXPERTS),bool)
        self.physical = np.zeros_like(self.persistent)
        self.reads = np.zeros((CYCLES,40,EXPERTS),np.uint8)
        self.prefixes = {}
        for layer in range(FIRST-1,LAST):
            gate = layers[layer+1].mlp.gate
            if (gate.dim,gate.topk,gate.n_routed)!=(5120,6,EXPERTS):
                raise RuntimeError('native target gate shape differs')
            if True in compile_modes:
                fn = moe._gate_prefix(gate)
            else:
                temp, sf = gate.gate_temp, gate.score_func
                fn = lambda x,w,b,t=temp,s=sf:moe._gate_prefix_impl(x,w,b,t,s)
            self.prefixes[layer] = (fn,weakref.ref(gate))
        self.physical_slots = {
            layer:tuple(slot for (lid,_),slot in rt.slots._persistent.items() if lid==layer)
            for layer in range(FIRST,LAST+1)}
        self.transients = tuple(rt.slots._transient)
        self.original_read_one = rt.reader.read_record_into
        self.original_read_batch = rt.reader.read_component_records_into

        def record_read(record):
            cycle = self.counts[record.layer]
            if 0 <= cycle < CYCLES:
                self.reads[cycle,record.layer,record.expert] += 1

        def read_one(manifest,record,destination,**kw):
            record_read(record)
            return self.original_read_one(manifest,record,destination,**kw)

        def read_batch(manifest,items,**kw):
            for record,_ in items:
                record_read(record)
            return self.original_read_batch(manifest,items,**kw)

        rt.reader.read_record_into = read_one
        rt.reader.read_component_records_into = read_batch
        for layer in range(FIRST-1,LAST+1):
            block = layers[layer]
            link = Link(self,layer)
            object.__setattr__(block.mlp.gate,'_router_capture',link)
            object.__setattr__(block.mlp.gate,'__class__',ObservedGate)
            if layer < LAST:
                object.__setattr__(block,'_router_capture',link)
                object.__setattr__(block,'__class__',ObservedLayer)
        self.last_switch = weakref.ref(layers[LAST].mlp.switch_mlp)
        self.last_run = layers[LAST].mlp.switch_mlp._run

        def final_layer_run(x,indices,*,shared_work):
            result = self.last_run(x,indices,shared_work=shared_work)
            if self.counts[LAST] == CYCLES-1:
                self.restore()
            return result

        object.__setattr__(layers[LAST].mlp.switch_mlp,'_run',final_layer_run)
        self.payload_bytes = sum(x.nbytes for x in (
            self.scores,self.actual,self.nrows,self.persistent,self.physical,self.reads))
        if self.payload_bytes > 64*1024**2:
            raise RuntimeError('capture payload exceeds the construction envelope')

    def observe(self, layer, x, indices):
        cycle = self.counts[layer]+1
        self.counts[layer] = cycle
        n = int(x.shape[0])
        if cycle >= CYCLES or not 1 <= n <= ROWS:
            raise RuntimeError('capture row/cycle shape exceeds the admitted bound')
        predicted = []
        if layer < LAST:
            h = self.incoming.pop(layer)
            mean = mx.mean(h.astype(mx.float32),axis=2).astype(mx.bfloat16).astype(mx.float32)
            fn,gate_ref = self.prefixes[layer]
            gate = gate_ref()
            predicted = [fn(feature.reshape(-1,5120),gate.weight,gate.e_score_correction_bias)[1]
                         for feature in (mean,x)]
        mx.eval(indices,*predicted)
        self.nrows[cycle,layer] = n
        self.actual[cycle,layer,:n] = np.asarray(indices)
        if layer < LAST:
            target, idx = layer+1, layer+1-FIRST
            for feature,values in enumerate(predicted):
                self.scores[feature,cycle,idx,:n] = np.asarray(values)
            rt = self.runtime()
            for expert in rt._banks[target]._expert_to_slot:
                self.persistent[cycle,idx,expert] = True
            for slot in (*self.physical_slots[target],*self.transients):
                if slot.state is ExpertSlotState.READY and slot.layer==target and slot.expert is not None:
                    self.physical[cycle,idx,slot.expert] = True

    def restore(self):
        if self.restored:
            return
        for layer in range(FIRST-1,LAST+1):
            block = self.layers[layer]()
            object.__setattr__(block.mlp.gate,'__class__',moe.Gate)
            object.__delattr__(block.mlp.gate,'_router_capture')
            if layer < LAST:
                object.__setattr__(block,'__class__',dv.DecoderLayer)
                object.__delattr__(block,'_router_capture')
        rt = self.runtime()
        rt.reader.read_record_into = self.original_read_one
        rt.reader.read_component_records_into = self.original_read_batch
        object.__setattr__(self.last_switch(),'_run',self.last_run)
        self.incoming.clear()
        self.prefixes.clear()
        self.physical_slots.clear()
        self.transients = ()
        self.restored = True

    def finish(self):
        if not self.restored or self.counts[FIRST-1:] != [CYCLES-1]*(40-FIRST+1):
            raise RuntimeError('native capture did not finish all 64 cycles')
        if not np.all(self.nrows[:,FIRST-1:]==ROWS):
            raise RuntimeError('captured workload was not the expected native M6 route')
        path = self.prefix.with_suffix('.router-capture.npz')
        if path.exists():
            raise RuntimeError('refusing capture overwrite')
        np.savez(path,scores=self.scores,actual=self.actual,nrows=self.nrows,
                 persistent=self.persistent,physical=self.physical,reads=self.reads)
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):
                digest.update(block)
        from router_analysis import analyze
        analysis = analyze(self.scores,self.actual,self.nrows,self.persistent,self.physical,self.reads)
        result = {'complete':True,'performance_claim':False,'cycles':CYCLES,'target_layers':list(range(FIRST,LAST+1)),
                  'features':list(FEATURES),'payload_bytes':self.payload_bytes,
                  'capture_path':str(path),'capture_sha256':digest.hexdigest(),
                  'decode_slots_per_layer':self.capacity,'hooks_removed':self.restored,
                  'analysis':analysis,'scope':'Diagnostic fences, no prefetch reads or route changes; first64 exact-workload native M6 cycles.'}
        self.prefix.with_suffix('.router-capture.json').write_text(json.dumps(result,indent=2)+'\n')
        print('ROUTER_CAPTURE',json.dumps({k:v for k,v in result.items() if k!='analysis'}),flush=True)
        return {k:v for k,v in result.items() if k!='analysis'}
