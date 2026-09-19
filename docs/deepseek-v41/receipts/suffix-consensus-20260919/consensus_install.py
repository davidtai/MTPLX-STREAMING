"""Construct the bounded native D5 plus causal consensus proposal lane."""
import ast
import hashlib
import inspect
import math

import numpy as np

from consensus import SuffixConsensus
from lookup import LookupExtension


class ConsensusExtension:
    def __init__(self, prompt):
        self.lookup = LookupExtension(prompt, minimum_context=2, extra_tokens=2)
        self.consensus = SuffixConsensus(prompt, min_suffix=3, min_count=2, max_extra=2)
        self.consensus.append_committed(())
        self.confidence_logit = math.log(9.0)

    def append_committed(self, tokens):
        self.lookup.append_committed(tokens)
        self.consensus.append_committed(tokens)

    def extend(self, native, confidence):
        proposed = self.lookup.extend(native)
        if len(proposed) > 5:
            return proposed
        # The native draft block has already evaluated confidence. Reading its
        # five FP32 logits introduces no new GPU operation or synchronization.
        if float(np.asarray(confidence).min()) < self.confidence_logit:
            return proposed
        return self.consensus.extend(native)


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
            '            drafts = lookup.extend(drafts, conf)\n            k_eff = len(drafts)')
    replace('        if token_callback is not None and delta:',
            '        lookup.append_committed(delta)\n        if token_callback is not None and delta:')
    restored = updated
    for old, new in reversed(replacements):
        restored = restored.replace(new, old)
    if restored != source:
        raise RuntimeError('consensus edits do not recover the native decode source')

    def mlx_calls(text):
        return [ast.dump(node, include_attributes=False) for node in ast.walk(ast.parse(text))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'mx']

    if mlx_calls(source) != mlx_calls(updated):
        raise RuntimeError('consensus construction changed native MLX operations')
    return updated


def install(module, model, prompt, *, requested_depth, verify_chunks, confidence_threshold):
    if (requested_depth != 5 or verify_chunks is not None or confidence_threshold is not None
            or model.mtp.block_size != 5 or len(prompt) != 16384
            or model._mtplx_expert_runtime.config.max_live_kv_tokens != 17664):
        raise RuntimeError('consensus requires the exact D5/M6-M8 single-request lane')
    lookup = ConsensusExtension(prompt)
    source = inspect.getsource(module._decode_cycles)
    updated = rewrite(source)
    namespace = dict(module.__dict__)
    namespace['_LOOKUP_EXTENSION'] = lookup
    exec(compile(updated, '<consensus_suffix_decode>', 'exec'), namespace)
    module._decode_cycles = namespace['_decode_cycles']
    return {'native_head_depth': 5, 'maximum_proposal_depth': 7, 'maximum_verify_rows': 8,
            'minimum_context_tokens': 2, 'maximum_lookup_tokens': 2,
            'selection': 'original full-proposal lookup then unanimous repeated suffix',
            'consensus_minimum_suffix': 3, 'consensus_minimum_occurrences': 2,
            'consensus_maximum_suffix': 6, 'minimum_native_confidence': 0.9,
            'confidence_logit_threshold': lookup.confidence_logit,
            'host_allowance_bytes': 48 * 1024**2,
            'consensus_extra_host_allowance_bytes': 32 * 1024**2,
            'additional_gpu_arrays_bytes': 0,
            'original_decode_function_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'hybrid_decode_function_sha256': hashlib.sha256(updated.encode()).hexdigest(),
            'native_mlx_operations_unchanged': True,
            'scope': 'One greedy exact16K/1024 request. Five native proposals and original lookup retained. Consensus reads committed history and evaluated native confidence only. Original target accept and commit remain authoritative.'}
