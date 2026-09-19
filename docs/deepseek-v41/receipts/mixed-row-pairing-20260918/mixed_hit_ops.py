"""One dispatch handles both paired and single rows in M6 cache hits."""
from dataclasses import dataclass
import mlx.core as mx
from mtplx.models.expert_mlx import _clamped_swiglu
from mixed_row_kernels import make_projection


@dataclass
class HitWork:
    positions: list
    bank: object
    pairs: object
    hidden: object
    grid_rows: int


class MixedHitOps:
    def __init__(self,native):
        self.gs,self.us,self.ds=native.gs,native.us,native.ds
        self.gu=make_projection(2304,5120)
        self.down_kernel=make_projection(5120,2304)

    def gate_up(self,tokens,experts,buffers):
        groups={}
        for pos,expert in enumerate(experts):
            if expert in buffers:
                dest=buffers[expert]
                groups.setdefault(id(dest.bank),[]).append((pos,expert,dest))
        work=[]
        for group in groups.values():
            positions=[entry[0] for entry in group]
            bank=group[0][2].bank
            members={}
            for row,(_,expert,dest) in enumerate(group):
                members.setdefault((dest.bank_index,expert),[]).append(row)
            assignments=[]
            for (slot,expert),rows in members.items():
                for i in range(0,len(rows),2):
                    assignments.append((slot,expert,rows[i],rows[i+1] if i+1<len(rows) else -1))
            pairs=mx.array(assignments,mx.int32)
            rows=len(group)
            x=mx.take(tokens,mx.array([p//6 for p in positions],mx.int32),axis=0).reshape(rows,1,1,5120)
            args=dict(template=[('T',mx.bfloat16)],grid=(32,576,len(assignments)),threadgroup=(32,2,1),
                output_shapes=[(rows,1,1,2304)],output_dtypes=[mx.bfloat16])
            g=self.gu(inputs=[x,pairs,bank.arrays['gate_proj.weight'],*self.gs],**args)[0]
            u=self.gu(inputs=[x,pairs,bank.arrays['up_proj.weight'],*self.us],**args)[0]
            h=_clamped_swiglu(g,u,10.0)
            work.append(HitWork(positions,bank,pairs,h,len(assignments)))
        return work

    def down(self,group):
        rows=len(group.positions)
        return self.down_kernel(inputs=[group.hidden,group.pairs,group.bank.arrays['down_proj.weight'],*self.ds],
            template=[('T',mx.bfloat16)],grid=(32,1280,group.grid_rows),threadgroup=(32,2,1),
            output_shapes=[(rows,1,1,5120)],output_dtypes=[mx.bfloat16])[0].reshape(rows,5120)
