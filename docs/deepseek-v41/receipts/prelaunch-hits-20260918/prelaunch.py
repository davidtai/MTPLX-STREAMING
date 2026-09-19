"""Prelaunch persistent-hit arithmetic before the native CPU route transaction."""
import ast
import hashlib
import inspect
from pathlib import Path

import numpy as np
import mlx.core as mx
import plane_lane
from mtplx.models.expert_mlx import _clamped_swiglu
from prelaunch_kernels import make_projection


class HitOps:
    def __init__(self, bank, scales):
        self.bank = bank
        self.gu = make_projection(2304,5120)
        self.down = make_projection(5120,2304)
        self.gs,self.us,self.ds = (scales[p] for p in ('gate_proj','up_proj','down_proj'))

    def __call__(self, tokens, indices, mapping):
        table = [-1]*384
        for expert,slot in mapping.items():
            table[expert] = slot
        ids = indices.reshape(-1)
        pairs = mx.stack([mx.take(mx.array(table,mx.int32),ids),ids],axis=1)
        rows = int(ids.size)
        x = mx.broadcast_to(tokens[:,None,:],(int(tokens.shape[0]),6,5120)).reshape(rows,1,1,5120)
        args = dict(template=[('T',mx.bfloat16)],grid=(32,576,rows),threadgroup=(32,2,1),
                    output_shapes=[(rows,1,1,2304)],output_dtypes=[mx.bfloat16])
        g = self.gu(inputs=[x,pairs,self.bank.arrays['gate_proj.weight'],*self.gs],**args)[0]
        u = self.gu(inputs=[x,pairs,self.bank.arrays['up_proj.weight'],*self.us],**args)[0]
        h = _clamped_swiglu(g,u,10.0)
        return self.down(inputs=[h,pairs,self.bank.arrays['down_proj.weight'],*self.ds],
            template=[('T',mx.bfloat16)],grid=(32,1280,rows),threadgroup=(32,2,1),
            output_shapes=[(rows,1,1,5120)],output_dtypes=[mx.bfloat16])[0].reshape(rows,5120)


class PrelaunchDecode(plane_lane.PackedDecode):
    def run(self, x, indices, *, shared_work):
        # Early consumers also cover errors before PendingSplitRoute exists.
        # Drain them before the outer teardown can release bank owners.
        try:
            return self.run_prelaunch(x,indices,shared_work=shared_work)
        except BaseException:
            mx.synchronize()
            raise


def install(runtime,switches,scales_by_layer,*,runtime_sha256):
    for name,expected in runtime_sha256.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('prelaunch native ownership source changed: '+name)
    old = plane_lane.install(runtime,switches,scales_by_layer,early=True)
    source = inspect.getsource(plane_lane.PackedDecode.run)
    import textwrap
    source = textwrap.dedent(source)
    changes = [
        ('def run(self, x, indices, *, shared_work):','def run_prelaunch(self, x, indices, *, shared_work):'),
        ('    mx.eval(indices)',
         '    mx.async_eval(indices)\n    early_hits = self.hit_ops(tokens,indices,self.policy_bank._expert_to_slot)\n    mx.async_eval(early_hits)\n    mx.eval(indices)'),
        ('    experts = tuple(int(e) for e in indices.reshape(-1).tolist())',
         '    experts = tuple(int(e) for e in np.asarray(indices).reshape(-1))'),
        ("            buffers = {b.expert:b.buffer for b in pending.hit_ready.bindings}\n            finish(ops.gate_up(tokens,experts,buffers))",
         "            hit_set = set(pending.hit_ready.plan.experts)\n            hit_positions = [p for p,e in enumerate(experts) if e in hit_set]\n            outputs.append(mx.take(early_hits,mx.array(hit_positions,mx.int32),axis=0))\n            positions.extend(hit_positions)")]
    updated = source
    for before,after in changes:
        if updated.count(before)!=1:raise RuntimeError('native plane construction changed')
        updated = updated.replace(before,after)
    restored = updated
    for before,after in reversed(changes):restored=restored.replace(after,before)
    if restored!=source:raise RuntimeError('prelaunch rewrite does not restore source')
    ast.parse(updated)
    namespace=dict(plane_lane.__dict__);namespace['np']=np
    exec(compile(updated,'<prelaunch_native_plane>','exec'),namespace)
    PrelaunchDecode.run_prelaunch=namespace['run_prelaunch']
    runners={}
    for layer,previous in old.items():
        policy=runtime._banks[layer]
        physical=runtime.slots._physical(layer,0).buffer.bank
        if (policy.expert_count!=384 or not policy.single_pool
                or policy.cache_policy!='transition-window'
                or policy._prefetch_ring is not None
                or policy.persistent_slots!=physical.capacity
                or any(not 0<=slot<physical.capacity for slot in policy._expert_to_slot.values())):
            raise RuntimeError('persistent-hit ownership does not fit the fixed route')
        runner=PrelaunchDecode(runtime,layer,previous.ops,previous.executor,early=True)
        runner.policy_bank=policy
        runner.hit_ops=HitOps(physical,scales_by_layer[layer])
        switches[layer]._run=runner.run
        runners[layer]=runner
    return runners
