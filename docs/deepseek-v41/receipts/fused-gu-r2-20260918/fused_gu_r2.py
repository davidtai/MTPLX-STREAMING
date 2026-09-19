"""Native reduction order; gate and up share only the input load and launch."""
import mlx.core as mx
from kernels import HEADER


def make_gate_up():
    source = r"""
constexpr uint N=2304;
constexpr uint K=5120;
constexpr uint V=16;
constexpr uint STEP=512;
const uint lane=thread_index_in_simdgroup;
const uint outrow=threadgroup_position_in_grid.y*4+simdgroup_index_in_threadgroup*2;
const uint assignment=threadgroup_position_in_grid.z;
const uint slot=pairs[2*assignment];
const uint expert=pairs[2*assignment+1];
const device ushort* gp=(const device ushort*)gate_weights+size_t(slot)*N*(K/4)+outrow*(K/4)+lane*(V/4);
const device ushort* up=(const device ushort*)up_weights+size_t(slot)*N*(K/4)+outrow*(K/4)+lane*(V/4);
const device T* xp=x+assignment*K+lane*V;
float gr[2]={0,0};
float ur[2]={0,0};
uint gd[2],ud[2];
for(int r=0;r<2;r++) {
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
    for(int r=0;r<2;r++) {
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
for(int r=0;r<2;r++) {
    gr[r]=simd_sum(gr[r]);
    ur[r]=simd_sum(ur[r]);
    if(lane==0) {
        gate_out[assignment*N+outrow+r]=T(gr[r]);
        up_out[assignment*N+outrow+r]=T(ur[r]);
    }
}
"""
    return mx.fast.metal_kernel(name='dsv41_packed_gate_up_r2_shared_input',
        input_names=['x','pairs','gate_weights','gate_desc','gate_payload','gate_offsets',
                     'up_weights','up_desc','up_payload','up_offsets'],
        output_names=['gate_out','up_out'],header=HEADER,source=source)

