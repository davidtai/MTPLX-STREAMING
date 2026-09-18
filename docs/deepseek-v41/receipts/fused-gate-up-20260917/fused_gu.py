"""Native reduction order; gate and up share only the input load and launch."""
import mlx.core as mx
from kernels import HEADER
from paired_kernels import make_projection
from mtplx.models.expert_mlx import _clamped_swiglu


def make_gate_up():
    source = r"""
constexpr uint N=2304;
constexpr uint K=5120;
constexpr uint V=16;
constexpr uint STEP=512;
const uint lane=thread_index_in_simdgroup;
const uint outrow=threadgroup_position_in_grid.y*8+simdgroup_index_in_threadgroup*4;
const uint assignment=threadgroup_position_in_grid.z;
const uint slot=pairs[2*assignment];
const uint expert=pairs[2*assignment+1];
const device ushort* gp=(const device ushort*)gate_weights+size_t(slot)*N*(K/4)+outrow*(K/4)+lane*(V/4);
const device ushort* up=(const device ushort*)up_weights+size_t(slot)*N*(K/4)+outrow*(K/4)+lane*(V/4);
const device T* xp=x+assignment*K+lane*V;
float gr[4]={0,0,0,0};
float ur[4]={0,0,0,0};
uint gd[4],ud[4];
for(int r=0;r<4;r++) {
    gd[r]=gate_desc[size_t(expert)*N+outrow+r];
    ud[r]=up_desc[size_t(expert)*N+outrow+r];
}
const uint gb=gate_offsets[expert];
const uint ub=up_offsets[expert];
#pragma clang loop unroll(disable)
for(uint block=0;block<K/STEP;block++) {
    float xv[V];
    for(uint i=0;i<V;i++) xv[i]=float(xp[block*STEP+i]);
    uint group=lane/(32/V)+block*(STEP/32);
    for(int r=0;r<4;r++) {
        uint bits=(gd[r]>>8)&15;
        uint exponent=gd[r]&255;
        if(bits) {
            uint bit=group*bits;
            uint code=gate_payload[gb+(gd[r]>>12)+(bit>>5)];
            exponent += (code>>(bit&31)) & ((1u<<bits)-1);
        }
        float scale=as_type<float>(exponent==0 ? uint(0x400000) : exponent<<23);
        gr[r] += scale_dot<V>(gp+r*(K/4)+block*(STEP/4),xv,scale);
        bits=(ud[r]>>8)&15;
        exponent=ud[r]&255;
        if(bits) {
            uint bit=group*bits;
            uint code=up_payload[ub+(ud[r]>>12)+(bit>>5)];
            exponent += (code>>(bit&31)) & ((1u<<bits)-1);
        }
        scale=as_type<float>(exponent==0 ? uint(0x400000) : exponent<<23);
        ur[r] += scale_dot<V>(up+r*(K/4)+block*(STEP/4),xv,scale);
    }
}
for(int r=0;r<4;r++) {
    gr[r]=simd_sum(gr[r]);
    ur[r]=simd_sum(ur[r]);
    if(lane==0) {
        gate_out[assignment*N+outrow+r]=T(gr[r]);
        up_out[assignment*N+outrow+r]=T(ur[r]);
    }
}
"""
    return mx.fast.metal_kernel(name='dsv41_packed_gate_up_shared_input',
        input_names=['x','pairs','gate_weights','gate_desc','gate_payload','gate_offsets',
                     'up_weights','up_desc','up_payload','up_offsets'],
        output_names=['gate_out','up_out'],header=HEADER,source=source)


def make_dispatch(scales):
    gate_up=make_gate_up()
    down=make_projection(5120,2304)
    gs,us,ds=(scales[p] for p in ('gate_proj','up_proj','down_proj'))
    def dispatch(selected,bindings):
        bank=bindings[0].buffer.bank
        pairs=mx.array([(b.buffer.bank_index,b.expert) for b in bindings],mx.int32)
        rows=len(bindings)
        x=selected.reshape(rows,1,1,5120)
        g,u=gate_up(inputs=[x,pairs,bank.arrays['gate_proj.weight'],*gs,bank.arrays['up_proj.weight'],*us],
            template=[('T',mx.bfloat16)],grid=(32,576,rows),threadgroup=(32,2,1),
            output_shapes=[(rows,1,1,2304)]*2,output_dtypes=[mx.bfloat16]*2)
        h=_clamped_swiglu(g,u,10.0)
        return down(inputs=[h,pairs,bank.arrays['down_proj.weight'],*ds],
            template=[('T',mx.bfloat16)],grid=(32,1280,rows),threadgroup=(32,2,1),
            output_shapes=[(rows,1,1,5120)],output_dtypes=[mx.bfloat16])[0].reshape(rows,5120)
    return dispatch
