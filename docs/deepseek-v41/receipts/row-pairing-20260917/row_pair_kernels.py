"""Two input rows share FP4 conversion; each row keeps native dot order."""
import mlx.core as mx
from kernels import HEADER

PAIR_HEADER = HEADER + '''
template<int V> inline float2 pair_scale_dot(
    const device ushort* w, thread const float* x0,
    thread const float* x1, float scale) {
    float2 accum = float2(0);
    for(int i=0; i<V/4; i++) {
        ushort word = w[i];
        float w0=scale_fp4(uchar(word));
        float w1=scale_fp4(uchar(word >> 4));
        float w2=scale_fp4(uchar(word >> 8));
        float w3=scale_fp4(uchar(word >> 12));
        accum += (float2(x0[4*i], x1[4*i])*w0 +
                  float2(x0[4*i+1], x1[4*i+1])*w1 +
                  float2(x0[4*i+2], x1[4*i+2])*w2 +
                  float2(x0[4*i+3], x1[4*i+3])*w3);
    }
    return scale*accum;
}
'''


def make_projection(n, k):
    if (n, k) not in ((2304, 5120), (5120, 2304)):
        raise ValueError('only native target projection shapes are admitted')
    values = 16 if k == 5120 else 8
    source = f'''
constexpr uint N={n};
constexpr uint K={k};
constexpr uint V={values};
constexpr uint STEP=V*32;
const uint lane=thread_index_in_simdgroup;
const uint outrow=threadgroup_position_in_grid.y*8+simdgroup_index_in_threadgroup*4;
const uint assignment=threadgroup_position_in_grid.z;
const uint slot=pairs[2*assignment];
const uint expert=pairs[2*assignment+1];
const device ushort* wp=(const device ushort*)weights+size_t(slot)*N*(K/4)+outrow*(K/4)+lane*(V/4);
const device T* xp=x+(2*assignment)*K+lane*V;
float2 result[4]={{float2(0),float2(0),float2(0),float2(0)}};
uint desc[4];
for(int r=0;r<4;r++) desc[r]=row_desc[size_t(expert)*N+outrow+r];
const uint expert_base=expert_offsets[expert];
#pragma clang loop unroll(disable)
for(uint block=0;block<K/STEP;block++) {{
    float x0[V],x1[V];
    for(uint i=0;i<V;i++) {{
        x0[i]=float(xp[block*STEP+i]);
        x1[i]=float(xp[K+block*STEP+i]);
    }}
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
        result[r] += pair_scale_dot<V>(wp+r*(K/4)+block*(STEP/4),x0,x1,scale);
    }}
}}
for(int r=0;r<4;r++) {{
    float a=simd_sum(result[r].x);
    float b=simd_sum(result[r].y);
    if(lane==0) {{
        out[(2*assignment)*N+outrow+r]=T(a);
        out[(2*assignment+1)*N+outrow+r]=T(b);
    }}
}}
'''
    return mx.fast.metal_kernel(
        name=f'dsv41_row_pair_{n}_{k}',
        input_names=['x','pairs','weights','row_desc','scale_payload','expert_offsets'],
        output_names=['out'], header=PAIR_HEADER, source=source)
