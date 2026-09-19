"""Construction-bound packed projection for the three draft stages only."""
import hashlib
import inspect
import textwrap
import mlx.core as mx
from mtplx.models import deepseek_v41 as dv
from mtplx.models import deepseek_v41_dspark as ds


class PackedDraftOutput:
    def __init__(self, attention):
        route = dv._DirectMXFP8OLoraOut(attention, attention._wo_a_quant())
        if ((route.groups, route.rank, route.per_group_input) != (8, 1024, 4096)
                or route.weight.dtype != mx.uint32 or route.scales.dtype != mx.uint8):
            raise RuntimeError('native draft output geometry differs')
        weight, scales, wo_b = route.weight, route.scales, route.wo_b

        def run(o, cos, sin):
            b, rows = o.shape[:2]
            o = dv._rope_last(o, cos, sin, inverse=True)
            grouped = o.astype(mx.float32).reshape(b * rows, 8, 4096).swapaxes(0, 1)
            projected = mx.gather_qmm(grouped, weight, scales, None,
                transpose=True, group_size=32, bits=8, mode='mxfp8')
            return wo_b(projected.swapaxes(0, 1).reshape(b, rows, 8192))

        self.run = mx.compile(run)

    def __call__(self, o, cos, sin):
        return self.run(o, cos, sin)


class Installation:
    def __init__(self, owner):
        self.attentions = tuple(b.attn for b in owner.mtp.layers)
        if len(self.attentions) != 3 or any(type(a) is not ds.DSparkAttention for a in self.attentions):
            raise RuntimeError('only the three native draft attention owners are supported')
        self.original = ds.DSparkAttention.__call__
        source = textwrap.dedent(inspect.getsource(self.original))
        start = source.index('        if d_use:', source.index('# Output prep:'))
        end = source.index('        _st.add(out)', start)
        changed = source[:start] + '        out = self._draft_packed_out(o, dcos, dsin)\n' + source[end:]
        namespace = dict(ds.__dict__)
        exec(compile(changed, '<draft_packed_projection>', 'exec'), namespace)
        self.candidate = namespace['__call__']
        self.routes = tuple(PackedDraftOutput(a) for a in self.attentions)
        self.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        self.candidate_sha256 = hashlib.sha256(changed.encode()).hexdigest()

    def select(self, packed):
        mx.synchronize()
        before = int(mx.get_active_memory())
        retired = 0
        for attention, route in zip(self.attentions, self.routes):
            cache = getattr(attention, '_wo_a_dense_cache', None)
            if cache is not None:
                retired += int(cache[3].nbytes)
            attention._wo_a_dense_cache = None
            attention._wo_a_bf16T_cache = None
            attention._draft_packed_out = route
        del cache
        mx.synchronize()
        mx.clear_cache()
        after = int(mx.get_active_memory())
        ds.DSparkAttention.__call__ = self.candidate if packed else self.original
        return {'packed': packed, 'dense_cache_bytes_removed': retired,
                'active_before_bytes': before, 'active_after_bytes': after,
                'native_call_sha256': self.source_sha256,
                'candidate_call_sha256': self.candidate_sha256}
