"""Pair repeated expert rows inside the declared M6 cache-hit phase."""
from dataclasses import dataclass
import mlx.core as mx
from mtplx.models.expert_mlx import _clamped_swiglu
from row_pair_kernels import make_projection


@dataclass
class HitWork:
    positions: list
    bank: object
    pairs: object
    hidden: object
    down_kernel: object
    grid_rows: int


class RowPairOps:
    def __init__(self, native):
        self.gs,self.us,self.ds=native.gs,native.us,native.ds
        self.routes={1:(native.gu_kernel,native.down_kernel),
                     2:(make_projection(2304,5120),make_projection(5120,2304))}

    def gate_up(self,tokens,experts,buffers):
        groups={}
        for pos,expert in enumerate(experts):
            if expert in buffers:
                dest=buffers[expert]
                groups.setdefault(id(dest.bank),{}).setdefault(expert,[]).append((pos,expert,dest))
        work=[]
        for members in groups.values():
            paired=[];solo=[]
            for entries in members.values():
                count=len(entries)//2*2
                paired.extend(entries[:count]);solo.extend(entries[count:])
            for factor,group in ((2,paired),(1,solo)):
                if not group:
                    continue
                positions=[v[0] for v in group]
                bank=group[0][2].bank
                pairs=mx.array([(v[2].bank_index,v[1]) for v in group[::factor]],mx.int32)
                rows=len(group);grid_rows=rows//factor
                x=mx.take(tokens,mx.array([p//6 for p in positions],mx.int32),axis=0).reshape(rows,1,1,5120)
                gu,down=self.routes[factor]
                args=dict(template=[('T',mx.bfloat16)],grid=(32,576,grid_rows),threadgroup=(32,2,1),
                          output_shapes=[(rows,1,1,2304)],output_dtypes=[mx.bfloat16])
                g=gu(inputs=[x,pairs,bank.arrays['gate_proj.weight'],*self.gs],**args)[0]
                u=gu(inputs=[x,pairs,bank.arrays['up_proj.weight'],*self.us],**args)[0]
                h=_clamped_swiglu(g,u,10.0)
                work.append(HitWork(positions,bank,pairs,h,down,grid_rows))
        return work

    def down(self,group):
        rows=len(group.positions)
        return group.down_kernel(inputs=[group.hidden,group.pairs,group.bank.arrays['down_proj.weight'],*self.ds],
            template=[('T',mx.bfloat16)],grid=(32,1280,group.grid_rows),threadgroup=(32,2,1),
            output_shapes=[(rows,1,1,5120)],output_dtypes=[mx.bfloat16])[0].reshape(rows,5120)
