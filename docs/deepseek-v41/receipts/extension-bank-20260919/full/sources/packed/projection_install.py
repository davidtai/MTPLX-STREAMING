"""Construction-bound exact projection expansion during target expert reads."""
from functools import partial
import hashlib
import inspect
import textwrap
from types import MethodType

import mlx.core as mx
import mlx.nn as nn
from mtplx.models import deepseek_v41 as dv
from mtplx.models import deepseek_v41_fused_proj_kernels as fp
import plane_lane
from fused_transpose import make_transpose


class PackedProjectionStore:
    __slots__ = ('packed', 'buffers', 'expand')

    def __init__(self, packed):
        self.packed = tuple(packed)
        self.buffers = [None, None]
        self.expand = make_transpose()

    def issue(self, layer):
        # The source router's covering eval has consumed the previous layer's
        # projection. Each replacement owns a fresh immutable GPU allocation.
        value = self.expand(*self.packed[layer])
        mx.async_eval(value)
        self.buffers[layer % 2] = value


class ScheduledOutput:
    __slots__ = ('store', 'buffer_index', 'wo_b', 'groups')

    def __init__(self, store, layer, wo_b, groups):
        self.store = store
        self.buffer_index = layer % 2
        self.wo_b, self.groups = wo_b, groups

    def __call__(self, o, qcos, qsin, b, s):
        wT = self.store.buffers[self.buffer_index]
        o = fp.rope_heads(o, qcos, qsin, inverse=True, out_dtype=wT.dtype)
        g = self.groups
        o = o.reshape(b * s, g, -1).swapaxes(0, 1)
        o = mx.matmul(o, wT)
        o = o.swapaxes(0, 1).reshape(b, s, -1)
        out = self.wo_b(o)
        fp.note_out()
        return out


def scheduled_run_source(source):
    anchor = '        if shared_work is not None:'
    if source.count(anchor) != 1:
        raise RuntimeError('packed demand/shared submission boundary changed')
    inserted = '        self.issue_next()\n' + anchor
    updated = source.replace(anchor, inserted)
    if updated.replace(inserted, anchor) != source:
        raise RuntimeError('projection schedule changed the native expert path')
    return updated


def install_model(model, *, backbone_type=dv.DeepseekV41Backbone):
    runtime = model._mtplx_expert_runtime
    if (type(model) is not dv.Model or type(model.model) is not backbone_type
            or len(model.model.layers) != 40 or model.args.hidden_size != 5120
            or model.mtp is None or not all(dv._fused_proj_use(m) for m in range(1, 9))
            or tuple(runtime.spec.routed_layer_indices) != tuple(range(40))
            or not runtime.config.verify_shared_overlap):
        raise RuntimeError('exact sequential native M<=8 target required')
    packed, runners = [], []
    for layer in model.model.layers:
        attn = layer.attn
        if (type(attn) is not dv.Attention or attn.n_heads != 64
                or attn.head_dim != 512 or attn.n_groups != 8
                or attn.o_lora_rank != 1024 or attn.dim != 5120
                or getattr(attn._out_prep_fused_impl, '__func__', None)
                    is not dv.Attention._out_prep_fused_dense
                or getattr(attn, '_wo_a_dense_cache', None) is not None
                or getattr(attn, '_wo_a_bf16T_cache', None) is not None):
            raise RuntimeError('native output route or post-prefill owners differ')
        for linear, ws, ss in ((attn.wo_a, (8192, 1024), (8192, 128)),
                               (attn.wo_b, (5120, 2048), (5120, 256))):
            if (type(linear) is not nn.QuantizedLinear or linear.bits != 8
                    or linear.group_size != 32 or linear.mode != 'mxfp8'
                    or linear.get('bias') is not None or linear.get('biases') is not None
                    or linear.weight.dtype != mx.uint32 or linear.scales.dtype != mx.uint8
                    or tuple(linear.weight.shape) != ws or tuple(linear.scales.shape) != ss):
                raise RuntimeError('native packed output projection differs')
        packed.append((attn.wo_a.weight, attn.wo_a.scales))
        switch = layer.mlp.switch_mlp
        runner = getattr(switch._run, '__self__', None)
        if (type(runner) is not plane_lane.PackedDecode
                or runner.runtime is not runtime or runner.layer != len(runners)
                or getattr(switch._run, '__func__', None) is not plane_lane.PackedDecode.run):
            raise RuntimeError('native packed target runner differs')
        runners.append(runner)
    total = sum(w.nbytes + s.nbytes for w, s in packed)
    if total != 1384120320 or len({(id(w), id(s)) for w, s in packed}) != 40:
        raise RuntimeError('packed target projection inventory differs')
    source = textwrap.dedent(inspect.getsource(plane_lane.PackedDecode.run))
    updated = scheduled_run_source(source)
    namespace = dict(plane_lane.__dict__)
    exec(compile(updated, '<packed_next_projection>', 'exec'), namespace)
    store = PackedProjectionStore(packed)
    for index, (layer, runner) in enumerate(zip(model.model.layers, runners)):
        runner.issue_next = partial(store.issue, (index + 1) % 40)
        layer.mlp.switch_mlp._run = MethodType(namespace['run'], runner)
        object.__setattr__(layer.attn, '_out_prep_fused_impl',
                           ScheduledOutput(store, index, layer.attn.wo_b, 8))
    object.__setattr__(model, '_predictable_projection_store', store)
    return {'target_layers': 40, 'retained_packed_projection_bytes': total,
            'retained_bf16_buffer_bytes': 2 * 67108864,
            'replacement_bf16_buffer_bound_bytes': 3 * 67108864,
            'conservative_steady_credit_vs_cached_bf16_bytes': 1098907648,
            'prefill_growth_and_seed_credit_bytes': 0,
            'original_expert_run_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'scheduled_expert_run_sha256': hashlib.sha256(updated.encode()).hexdigest(),
            'scope': 'One sequential target request; packed weights remain resident. '
                     'Next BF16 transpose issued after demand and resident submission; '
                     'native output arithmetic and original expert path retained.'}


def prime_model(model):
    store = model._predictable_projection_store
    if any(v is not None for v in store.buffers):
        raise RuntimeError('projection prime can execute only once')
    store.issue(0)
    mx.eval(store.buffers[0])
    return {'primed_after_native_seed_and_overflow': True}


def verify_retirement(model):
    """Verify bounded owners after the measured request; no hot-path counters."""
    store = model._predictable_projection_store
    for index, layer in enumerate(model.model.layers):
        attn = layer.attn
        lane = attn._out_prep_fused_impl
        if (type(lane) is not ScheduledOutput or lane.store is not store
                or lane.buffer_index != index % 2 or lane.wo_b is not attn.wo_b
                or getattr(attn, '_wo_a_bf16T_cache', None) is not None
                or getattr(attn, '_wo_a_dense_cache', None) is not None
                or store.packed[index][0] is not attn.wo_a.weight
                or store.packed[index][1] is not attn.wo_a.scales):
            raise RuntimeError('scheduled projection ownership differs')
    if (len(store.buffers) != 2 or any(tuple(v.shape) != (8,4096,1024)
            or v.dtype != mx.bfloat16 for v in store.buffers)):
        raise RuntimeError('bounded BF16 expansion owners differ')
    return {'module_ownership_verified': True, 'retired_module_source_bytes': 0,
            'retained_module_packed_bytes': sum(w.nbytes+s.nbytes for w,s in store.packed),
            'retained_module_bf16_bytes': sum(v.nbytes for v in store.buffers)}
