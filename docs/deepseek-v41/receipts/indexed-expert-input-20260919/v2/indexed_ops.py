"""Direct token-row addressing; original grouping, dot order and activation."""
import mlx.core as mx
from mtplx.models.expert_mlx import _clamped_swiglu
from plane_lane import PackedOps, GateUpWork
from indexed_kernel import make_indexed_projection

class IndexedOps(PackedOps):
    def __init__(self, scales):
        super().__init__(scales)
        self.gu_kernel = make_indexed_projection(2304,5120)

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
            token_rows = mx.array([p//6 for p in positions],mx.int32)
            args = dict(template=[('T',mx.bfloat16)],grid=(32,576,rows),threadgroup=(32,2,1),
                        output_shapes=[(rows,1,1,2304)],output_dtypes=[mx.bfloat16])
            g = self.gu_kernel(inputs=[tokens,pairs,token_rows,bank.arrays['gate_proj.weight'],*self.gs],**args)[0]
            u = self.gu_kernel(inputs=[tokens,pairs,token_rows,bank.arrays['up_proj.weight'],*self.us],**args)[0]
            h = _clamped_swiglu(g,u,10.0)
            work.append(GateUpWork(positions,bank,pairs,h))
        return work
