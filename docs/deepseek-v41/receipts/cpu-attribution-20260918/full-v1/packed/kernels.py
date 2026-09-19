"""Native-geometry scale-codec operator prototype; construction-bound kernels."""
import mlx.core as mx

HEADER='''
// Dot arithmetic follows MLX v0.32.2 fp_quantized.h (Apache-2.0).
inline float scale_fp4(uchar bits) {
    half converted=as_type<half>(ushort((bits & 7) << 9));
    converted *= 16384.0;
    return float(bits & 8 ? -converted : converted);
}
template<int V> inline float scale_dot(
    const device ushort* w, thread const float* x, float scale) {
    float accum=0;
    for(int i=0;i<V/4;i++) {
        accum += (x[4*i]*scale_fp4(uchar(w[i])) +
                  x[4*i+1]*scale_fp4(uchar(w[i] >> 4)) +
                  x[4*i+2]*scale_fp4(uchar(w[i] >> 8)) +
                  x[4*i+3]*scale_fp4(uchar(w[i] >> 12)));
    }
    return scale*accum;
}
'''


def make_projection(n,k,packed):
    if (n,k) not in ((2304,5120),(5120,2304)):
        raise ValueError('only native target expert geometry is admitted')
    values=16 if k==5120 else 8
    step=values*32
    assert k%step==0 and n%8==0
    if packed:
        prep='''
uint desc[4];
for(int r=0;r<4;r++) desc[r]=row_desc[size_t(expert)*N+outrow+r];
const uint expert_base=expert_offsets[expert];
'''
        scale='''
uint bits=(desc[r]>>8)&15;
uint exponent=desc[r]&255;
if(bits) {
    uint bit=group*bits;
    uint code=scale_payload[expert_base+(desc[r]>>12)+(bit>>5)];
    exponent += (code>>(bit&31)) & ((1u<<bits)-1);
}
'''
    else:
        prep=''
        scale='''uint exponent=raw_scales[(size_t(expert)*N+outrow+r)*(K/32)+group];'''
    source=f'''
constexpr uint N={n};
constexpr uint K={k};
constexpr uint V={values};
constexpr uint STEP={step};
const uint lane=thread_index_in_simdgroup;
const uint outrow=threadgroup_position_in_grid.y*8+simdgroup_index_in_threadgroup*4;
const uint assignment=threadgroup_position_in_grid.z;
const uint expert=ids[assignment];
const device ushort* wp=(const device ushort*)weights+size_t(expert)*N*(K/4)+outrow*(K/4)+lane*(V/4);
const device T* xp=x+assignment*K+lane*V;
float result[4]={{0,0,0,0}};
{prep}
#pragma clang loop unroll(disable)
for(uint block=0;block<K/STEP;block++) {{
    float xv[V];
    for(uint i=0;i<V;i++) xv[i]=float(xp[block*STEP+i]);
    uint group=lane/(32/V)+block*(STEP/32);
    for(int r=0;r<4;r++) {{
        {scale}
        float scale=as_type<float>(exponent==0 ? uint(0x400000) : exponent<<23);
        result[r] += scale_dot<V>(wp+r*(K/4)+block*(STEP/4),xv,scale);
    }}
}}
for(int r=0;r<4;r++) {{
    result[r]=simd_sum(result[r]);
    if(lane==0) out[assignment*N+outrow+r]=T(result[r]);
}}
'''
    return mx.fast.metal_kernel(name=f'dsv41_scale_{n}_{k}_{int(packed)}',
        input_names=['x','ids','weights','raw_scales','row_desc','scale_payload','expert_offsets'],
        output_names=['out'],header=HEADER,source=source)
