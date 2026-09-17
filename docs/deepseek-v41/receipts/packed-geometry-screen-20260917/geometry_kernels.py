"""Construction-bound packed FP4 geometry alternatives; native dot order."""
import mlx.core as mx
from kernels import HEADER

FLOAT_FP4 = '''
inline float scale_fp4(uchar bits) {
    uint magnitude = uint(bits) & 7u;
    uint word = magnitude < 2u ? magnitude * 0x3f000000u
                              : 0x3f000000u + (magnitude << 22);
    word |= (uint(bits) & 8u) << 28;
    return as_type<float>(word);
}
'''


def make_projection(n, k, *, results, simdgroups, fp4_float_bits=False):
    if (n, k) not in ((2304, 5120), (5120, 2304)):
        raise ValueError('native target geometry required')
    if results not in (4, 8) or simdgroups not in (2, 4) or n % (results * simdgroups):
        raise ValueError('unsupported native output tiling')
    header = HEADER
    if fp4_float_bits:
        start = header.index('inline float scale_fp4(')
        end = header.index('template<int V>', start)
        header = header[:start] + FLOAT_FP4 + header[end:]
    values = 16 if k == 5120 else 8
    source = f'''
constexpr uint N={n};
constexpr uint K={k};
constexpr uint V={values};
constexpr uint STEP={values * 32};
constexpr uint R={results};
const uint lane=thread_index_in_simdgroup;
const uint outrow=threadgroup_position_in_grid.y*{results * simdgroups}+simdgroup_index_in_threadgroup*R;
const uint assignment=threadgroup_position_in_grid.z;
const uint slot=pairs[2*assignment];
const uint expert=pairs[2*assignment+1];
const device ushort* wp=(const device ushort*)weights+size_t(slot)*N*(K/4)+outrow*(K/4)+lane*(V/4);
const device T* xp=x+assignment*K+lane*V;
float result[R]={{0}};
uint desc[R];
for(uint r=0;r<R;r++) desc[r]=row_desc[size_t(expert)*N+outrow+r];
const uint expert_base=expert_offsets[expert];
#pragma clang loop unroll(disable)
for(uint block=0;block<K/STEP;block++) {{
    float xv[V];
    for(uint i=0;i<V;i++) xv[i]=float(xp[block*STEP+i]);
    uint group=lane/(32/V)+block*(STEP/32);
    for(uint r=0;r<R;r++) {{
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
for(uint r=0;r<R;r++) {{
    result[r]=simd_sum(result[r]);
    if(lane==0) out[assignment*N+outrow+r]=T(result[r]);
}}
'''
    return mx.fast.metal_kernel(
        name=f'dsv41_packed_geometry_{n}_{k}_r{results}_sg{simdgroups}_f{int(fp4_float_bits)}',
        input_names=['x', 'pairs', 'weights', 'row_desc', 'scale_payload', 'expert_offsets'],
        output_names=['out'], header=header, source=source)


def make_dispatch(scales, *, results, simdgroups, fp4_float_bits=False):
    from mtplx.models.expert_mlx import _clamped_swiglu
    gate_kernel = make_projection(2304, 5120, results=results,
        simdgroups=simdgroups, fp4_float_bits=fp4_float_bits)
    down_kernel = make_projection(5120, 2304, results=results,
        simdgroups=simdgroups, fp4_float_bits=fp4_float_bits)
    gate_scales, up_scales, down_scales = (scales[p] for p in ('gate_proj', 'up_proj', 'down_proj'))
    gate_grid_y, down_grid_y = 2304 // results, 5120 // results
    group = (32, simdgroups, 1)

    def dispatch(selected, bindings, *, dense_prefill=False):
        bank = bindings[0].buffer.bank
        pairs = mx.array([(b.buffer.bank_index, b.expert) for b in bindings], dtype=mx.int32)
        rows = len(bindings)
        x = selected.reshape(rows, 1, 1, 5120)
        gate = gate_kernel(inputs=[x, pairs, bank.arrays['gate_proj.weight'], *gate_scales],
            template=[('T', mx.bfloat16)], grid=(32, gate_grid_y, rows), threadgroup=group,
            output_shapes=[(rows, 1, 1, 2304)], output_dtypes=[mx.bfloat16])[0]
        up = gate_kernel(inputs=[x, pairs, bank.arrays['up_proj.weight'], *up_scales],
            template=[('T', mx.bfloat16)], grid=(32, gate_grid_y, rows), threadgroup=group,
            output_shapes=[(rows, 1, 1, 2304)], output_dtypes=[mx.bfloat16])[0]
        hidden = _clamped_swiglu(gate, up, 10.0)
        return down_kernel(inputs=[hidden, pairs, bank.arrays['down_proj.weight'], *down_scales],
            template=[('T', mx.bfloat16)], grid=(32, down_grid_y, rows), threadgroup=group,
            output_shapes=[(rows, 1, 1, 5120)], output_dtypes=[mx.bfloat16])[0].reshape(rows, 5120)

    return dispatch
