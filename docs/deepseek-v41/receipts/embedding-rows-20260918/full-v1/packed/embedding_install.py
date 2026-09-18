"""Bind one-request input-row ownership before prefill and retire at its fence."""
import gc
import importlib.util
import time
from weakref import ref

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from file_embedding import FileRowEmbedding, ARENA_BYTES, HOST_BOUND_BYTES


def prepare(model, source, admission):
    native = model.model.embed_tokens
    if (type(native) is not nn.Embedding or native.weight.dtype != mx.bfloat16
            or list(native.weight.shape) != source['shape']
            or int(native.weight.nbytes) != source['nbytes']
            or model.args.tie_word_embeddings
            or sum(value is native.weight for _, value in tree_flatten(model.parameters())) != 1):
        raise RuntimeError('native input embedding has unexpected type/layout/owners')
    if (admission['embedding_post_prefill_credit_bytes'] != source['nbytes']
            or admission['embedding_host_allowance_bytes'] != HOST_BOUND_BYTES):
        raise RuntimeError('input-row ownership differs from admission')
    cache = FileRowEmbedding(source)
    rt = model._mtplx_expert_runtime
    original_close = rt.close

    def close(*args, **kwargs):
        try:
            return original_close(*args, **kwargs)
        finally:
            mx.synchronize()
            cache.close()

    # Establish cleanup before any model owner can be replaced.
    rt.close = close
    target = ref(model)
    state = {'phase': 'prefill_native', 'source_bytes': source['nbytes'],
             'arena_bytes': ARENA_BYTES, 'host_allowance_bytes': HOST_BOUND_BYTES,
             'prefill_credit_bytes': 0, 'resident_plan_retains_source_reserve': True}

    def retire():
        if state['phase'] != 'prefill_native':
            raise RuntimeError('input embedding can transition only once')
        start = time.perf_counter()
        current = target()
        rt.flush_deferred_slot_releases(evaluate=True)
        mx.synchronize()
        gc.collect()
        mx.clear_cache()
        before = int(mx.get_active_memory())
        old = current.model.embed_tokens
        source_id = id(old.weight)
        current.model.embed_tokens = cache
        del old
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        after = int(mx.get_active_memory())
        if any(id(value) == source_id for _, value in tree_flatten(current.parameters())):
            raise RuntimeError('native input tensor remains in the model tree')
        if before - after < source['nbytes']:
            raise RuntimeError('native input table did not release its admitted bytes')
        if after > admission['transition_start_active_bound_bytes']:
            raise RuntimeError('retired input-table transition exceeds its bound')
        # No mapping is owned by this cache. Remove any older clean source pages
        # before spending the released capacity, keeping file cache explicit.
        spec = importlib.util.spec_from_file_location('embedding_reclaim',
            'scripts/deepseek_v41/reclaim_file_cache.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reclaimed = module.reclaim_file(source['path'])
        if reclaimed['cached_page_bytes_after']:
            raise RuntimeError('embedding source cache retained physical pages')
        state.update(phase='decode_rows', active_before_bytes=before,
                     active_after_bytes=after, released_active_bytes=before-after,
                     source_file_cache=reclaimed, transition_seconds=time.perf_counter()-start)
        return state

    def completed():
        current = target()
        if current.model.embed_tokens is not cache or state['phase'] != 'decode_rows':
            raise RuntimeError('file embedding is not the completed decode owner')
        if len(cache._arena) != ARENA_BYTES or len(cache._lru) > cache._capacity:
            raise RuntimeError('input row cache exceeded its fixed allocation')
        state.update(module_ownership_verified=True, retained_metal_table_bytes=0,
                     cached_rows=len(cache._lru), row_capacity=cache._capacity)
        return dict(state)

    return retire, completed, state
