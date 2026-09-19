"""Install a native D5 draft with bounded causal lookup proposals up to D7."""
import ast
import hashlib
import inspect

from lookup import LookupExtension


def rewrite(source):
    replacements = []
    updated = source

    def replace(old, new):
        nonlocal updated
        if updated.count(old) != 1:
            raise RuntimeError('native decode construction changed: ' + old)
        updated = updated.replace(old, new)
        replacements.append((old, new))

    replace('    verify_chunks = _normalize_verify_chunks(verify_chunks, k_cap)',
            '    verify_chunks = (8,)')
    replace('    stats.speculative_depth = k_cap', '    stats.speculative_depth = 7')
    replace('    stats._ensure_depth(k_cap)', '    stats._ensure_depth(7)')
    replace('    new_tokens: List[int] = []',
            '    lookup = _LOOKUP_EXTENSION\n    lookup.append_committed([primary])\n    new_tokens: List[int] = []')
    replace('            drafts = [int(out_np[1 + i]) for i in range(k_eff)]',
            '            drafts = [int(out_np[1 + i]) for i in range(k_eff)]\n'
            '            drafts = lookup.extend(drafts)\n            k_eff = len(drafts)')
    replace('        if token_callback is not None and delta:',
            '        lookup.append_committed(delta)\n        if token_callback is not None and delta:')
    restored = updated
    for old, new in reversed(replacements):
        restored = restored.replace(new, old)
    if restored != source:
        raise RuntimeError('hybrid edits do not recover the native decode source')

    def mlx_calls(text):
        return [ast.dump(node,include_attributes=False) for node in ast.walk(ast.parse(text))
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                and isinstance(node.func.value,ast.Name) and node.func.value.id=='mx']
    if mlx_calls(source) != mlx_calls(updated):
        raise RuntimeError('hybrid construction changed native MLX operations')
    return updated


def install(module, model, prompt, *, requested_depth, verify_chunks, confidence_threshold):
    if (requested_depth != 5 or verify_chunks is not None or confidence_threshold is not None
            or model.mtp.block_size != 5 or len(prompt) != 16384
            or model._mtplx_expert_runtime.config.max_live_kv_tokens != 17664):
        raise RuntimeError('lookup extension requires the exact D5/M6-M8 single-request lane')
    lookup = LookupExtension(prompt,minimum_context=2,extra_tokens=2)
    source = inspect.getsource(module._decode_cycles)
    updated = rewrite(source)
    namespace = dict(module.__dict__)
    namespace['_LOOKUP_EXTENSION'] = lookup
    exec(compile(updated,'<hybrid_lookup_decode>','exec'),namespace)
    module._decode_cycles = namespace['_decode_cycles']
    return {'native_head_depth':5,'maximum_proposal_depth':7,'maximum_verify_rows':8,
            'minimum_context_tokens':2,'maximum_lookup_tokens':2,
            'selection':'earliest_longest_context_matching_all_five_native_drafts',
            'host_allowance_bytes':16*1024**2,'additional_gpu_arrays_bytes':0,
            'original_decode_function_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'hybrid_decode_function_sha256':hashlib.sha256(updated.encode()).hexdigest(),
            'native_mlx_operations_unchanged':True,
            'scope':'One greedy exact16K/1024 request. Native MTP proposals retained; causal past-text extension verified by the unchanged target accept/commit path.'}
