"""One-request native BF16 output projection with a single weight owner."""
import weakref

import mlx.core as mx
import mlx.nn as nn
from mtplx.models import deepseek_v41 as dv
from mtplx.models import deepseek_v41_fused_proj_kernels as fp


class BF16Output:
    __slots__ = ('weight', 'wo_b', 'groups')

    def __init__(self, weight, wo_b, groups):
        self.weight, self.wo_b, self.groups = weight, wo_b, groups

    def __call__(self, o, qcos, qsin, b, s):
        # Same operations, dtypes, layouts and kernels as the native dense route.
        wT = self.weight
        o = fp.rope_heads(o, qcos, qsin, inverse=True, out_dtype=wT.dtype)
        g = self.groups
        o = o.reshape(b * s, g, -1).swapaxes(0, 1)
        o = mx.matmul(o, wT)
        o = o.swapaxes(0, 1).reshape(b, s, -1)
        out = self.wo_b(o)
        fp.note_out()
        return out


class FirstOutput:
    __slots__ = ('attention',)

    def __init__(self, attention):
        # The owning Attention call keeps this object alive. Do not create an
        # Attention -> callable -> Attention cycle while awaiting first use.
        self.attention = weakref.ref(attention)

    def __call__(self, o, qcos, qsin, b, s):
        attn = self.attention()
        weight = attn._o_lora_fused_weight()
        owned = BF16Output(weight, attn.wo_b, attn.n_groups)
        object.__setattr__(attn, '_out_prep_fused_impl', owned)
        attn._wo_a_bf16T_cache = None
        attn._wo_a_dense_cache = None
        del attn['wo_a']
        return owned(o, qcos, qsin, b, s)


def install_attention(attn):
    # Validate all invariants before measured execution. FirstOutput only does
    # the native first-use materialization, ownership transfer and retirement.
    if (type(attn) is not dv.Attention or attn.n_heads != 64
        or attn.head_dim != 512 or attn.n_groups != 8 or attn.o_lora_rank != 1024
        or attn.dim != 5120
        or getattr(attn._out_prep_fused_impl, '__func__', None) is not dv.Attention._out_prep_fused_dense):
        raise RuntimeError('native target BF16 output route and geometry required')
    wo, wb = attn.wo_a, attn.wo_b
    for linear, weight_shape, scale_shape in (
        (wo, (8192, 1024), (8192, 128)),
        (wb, (5120, 2048), (5120, 256)),
    ):
        if (type(linear) is not nn.QuantizedLinear or linear.bits != 8
            or linear.group_size != 32 or linear.mode != 'mxfp8'
            or linear.get('bias') is not None or linear.get('biases') is not None
            or linear.weight.dtype != mx.uint32 or linear.scales.dtype != mx.uint8
            or tuple(linear.weight.shape) != weight_shape or tuple(linear.scales.shape) != scale_shape):
            raise RuntimeError('native output projection storage differs')
    raw_bytes = int(wo.weight.nbytes + wo.scales.nbytes)
    if raw_bytes != 34603008:
        raise RuntimeError('packed projection release differs from accounting')
    object.__setattr__(attn, '_out_prep_fused_impl', FirstOutput(attn))
    return {'raw_bytes_released_on_first_use': raw_bytes,
            'retained_bf16_bytes': 67108864,
            'scope': 'One request, fused native output projection only; original packed wo_a is retired after materialization. Generic prefill/reload is not installed.'}
