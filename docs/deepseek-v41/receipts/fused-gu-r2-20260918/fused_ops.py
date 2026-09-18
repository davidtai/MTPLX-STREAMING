"""Native ownership/grouping and down projection with construction-bound R2 GU."""
import mlx.core as mx
from mtplx.models.expert_mlx import _clamped_swiglu
from paired_kernels import make_projection
from plane_lane import GateUpWork
from fused_gu_r2 import make_gate_up

class FusedPackedOps:
    def __init__(self, scales):
        self.gu_kernel = make_gate_up()
        self.down_kernel = make_projection(5120,2304)
        self.gs, self.us, self.ds = (scales[p] for p in ('gate_proj','up_proj','down_proj'))

    def gate_up(self, tokens, experts, buffers):
        groups = {}
        for pos, expert in enumerate(experts):
            if expert in buffers:
                dest = buffers[expert]
                groups.setdefault(id(dest.bank), []).append((pos,expert,dest))
        work = []
        for group in groups.values():
            positions = [v[0] for v in group]
            bank = group[0][2].bank
            pairs = mx.array([(v[2].bank_index,v[1]) for v in group], mx.int32)
            rows = len(group)
            x = mx.take(tokens,mx.array([p//6 for p in positions],mx.int32),axis=0).reshape(rows,1,1,5120)
            g,u = self.gu_kernel(
                inputs=[x,pairs,bank.arrays['gate_proj.weight'],*self.gs,bank.arrays['up_proj.weight'],*self.us],
                template=[('T',mx.bfloat16)],grid=(32,1152,rows),threadgroup=(32,2,1),
                output_shapes=[(rows,1,1,2304)]*2,output_dtypes=[mx.bfloat16]*2)
            h = _clamped_swiglu(g,u,10.0)
            work.append(GateUpWork(positions,bank,pairs,h))
        return work

    def down(self, group):
        rows = len(group.positions)
        return self.down_kernel(inputs=[group.hidden,group.pairs,group.bank.arrays['down_proj.weight'],*self.ds],
            template=[('T',mx.bfloat16)],grid=(32,1280,rows),threadgroup=(32,2,1),
            output_shapes=[(rows,1,1,5120)],output_dtypes=[mx.bfloat16])[0].reshape(rows,5120)

