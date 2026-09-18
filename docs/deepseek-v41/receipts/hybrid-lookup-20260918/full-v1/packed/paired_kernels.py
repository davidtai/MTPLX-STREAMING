"""Exact native geometry with separate physical weight slot and scale expert ID."""
import mlx.core as mx
from kernels import HEADER


def make_projection(n, k):
    if (n, k) not in ((2304, 5120), (5120, 2304)):
        raise ValueError('only native target geometry is admitted')
    values = 16 if k == 5120 else 8
    step = values * 32
    source = f'''
constexpr uint N={n};
constexpr uint K={k};
constexpr uint V={values};
constexpr uint STEP={step};
const uint lane=thread_index_in_simdgroup;
const uint outrow=threadgroup_position_in_grid.y*8+simdgroup_index_in_threadgroup*4;
const uint assignment=threadgroup_position_in_grid.z;
const uint slot=pairs[2*assignment];
const uint expert=pairs[2*assignment+1];
const device ushort* wp=(const device ushort*)weights+size_t(slot)*N*(K/4)+outrow*(K/4)+lane*(V/4);
const device T* xp=x+assignment*K+lane*V;
float result[4]={{0,0,0,0}};
uint desc[4];
for(int r=0;r<4;r++) desc[r]=row_desc[size_t(expert)*N+outrow+r];
const uint expert_base=expert_offsets[expert];
#pragma clang loop unroll(disable)
for(uint block=0;block<K/STEP;block++) {{
    float xv[V];
    for(uint i=0;i<V;i++) xv[i]=float(xp[block*STEP+i]);
    uint group=lane/(32/V)+block*(STEP/32);
    for(int r=0;r<4;r++) {{
        uint bits=(desc[r]>>8)&15;
        uint exponent=desc[r]&255;
        if(bits) {{
            uint bit=group*bits;
            uint code=scale_payload[expert_base+(desc[r]>>12)+(bit>>5)];
            exponent += (code>>(bit&31)) & ((1u<<bits)-1);
        }}
        float scale=as_type<float>(exponent==0 ? uint(0x400000) : exponent<<23);
        result[r] += scale_dot<V>(wp+r*(K/4)+block*(STEP/4),xv,scale);
    }}
}}
for(int r=0;r<4;r++) {{
    result[r]=simd_sum(result[r]);
    if(lane==0) out[assignment*N+outrow+r]=T(result[r]);
}}
'''
    return mx.fast.metal_kernel(
        name=f'dsv41_packed_scales_{n}_{k}_paired',
        input_names=['x', 'pairs', 'weights', 'row_desc', 'scale_payload', 'expert_offsets'],
        output_names=['out'], header=HEADER, source=source)
