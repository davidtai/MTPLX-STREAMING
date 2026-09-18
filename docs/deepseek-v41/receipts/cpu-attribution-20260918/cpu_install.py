"""Install diagnostic clocks without changing native tensor operations."""
import ast,hashlib,inspect


def instrument_source(source):
    updated=source
    replacements=[]

    def replace(old,new,count=1):
        nonlocal updated
        assert updated.count(old)==count,(old,updated.count(old))
        updated=updated.replace(old,new)
        replacements.append((old,new))

    replace('_t = _time.perf_counter()','_t = _CPU_RECORDER.start()',4)
    for field,name in (('stats.draft_time_s','draft'),('stats.verify_time_s','verify'),
                       ('accept_time_s','accept'),('stats.commit_time_s','commit')):
        replace(f'{field} += _time.perf_counter() - _t',f'{field} += _CPU_RECORDER.finish({name!r},_t)')
    for statement,name in (
        ('chunk_logits, chunk_hidden = forward(mx.array([chunk_ids]), cache)','target_forward'),
        ('mx.eval(chunk_logits, chunk_hidden)','final_eval')):
        replace('                '+statement,
                '                _cpu_t = _CPU_RECORDER.start()\n'
                '                '+statement+'\n'
                f'                _CPU_RECORDER.finish({name!r},_cpu_t)')
    restored=updated
    for old,new in reversed(replacements):restored=restored.replace(new,old)
    assert restored==source
    return updated


def install_decode(module,recorder):
    original=module._decode_cycles
    source=inspect.getsource(original)
    updated=instrument_source(source)
    namespace=dict(module.__dict__)
    namespace['_CPU_RECORDER']=recorder
    exec(compile(updated,'<diagnostic_decode_cycles>','exec'),namespace)
    timed=namespace['_decode_cycles']
    span=recorder.phases['decode_cycles']

    def observed(**kwargs):
        start=span.start()
        try:return timed(**kwargs)
        finally:span.finish(start)

    module._decode_cycles=observed
    return {'original_function_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'diagnostic_function_sha256':hashlib.sha256(updated.encode()).hexdigest(),
            'native_source_recovered_exactly_after_removing_clock_edits':True,
            'added_gpu_fences':0}


def install_model(model,recorder):
    from plane_lane import PackedDecode
    layers=model.model.layers
    assert len(layers)==40
    base=type(layers[0])
    assert all(type(layer) is base for layer in layers)
    native=base.attn_and_moe_input

    class ClockedLayer(base):
        def attn_and_moe_input(self,*args,**kwargs):
            span=self._cpu_span
            start=span.start()
            try:return native(self,*args,**kwargs)
            finally:span.finish(start)

    for index,layer in enumerate(layers):
        switch=layer.mlp.switch_mlp
        original=switch._run
        assert original.__func__ is PackedDecode.run
        span=recorder.expert[index]

        def run(*args,_native=original,_span=span,**kwargs):
            start=_span.start()
            try:return _native(*args,**kwargs)
            finally:_span.finish(start)

        object.__setattr__(switch,'_run',run)
        object.__setattr__(layer,'_cpu_span',recorder.attention[index])
        object.__setattr__(layer,'__class__',ClockedLayer)
    return {'layers':40,'install_phase':'after bank growth and projection installation',
            'tensor_owners_added':0,'added_gpu_fences':0,
            'scope':'Attention preparation and packed expert entry-point CPU/elapsed time, including any deferred work consumed inside those calls.'}
