"""Read fixed Python cache ownership only at request boundaries."""
import os
from mtplx.deepseek_v41_memory_profile import python_cache_budget

def describe_engram(model):
    if os.environ.get('MTPLX_ENGRAM_CACHE_LIMIT') != '67108864':
        raise RuntimeError('bounded Engram environment changed')
    budget = python_cache_budget(os.environ)
    if budget['required_host_bytes'] != 1338109952 or len(model._engram_banks) != 2:
        raise RuntimeError('bounded Engram host inventory differs')
    banks = []
    for bank in model._engram_banks:
        cache = bank.cache
        if (cache.budget_bytes != 67108864 or cache.row_bytes < 264
            or cache.slot_count != 67108864 // cache.row_bytes
            or cache._arena.shape != (cache.slot_count, cache.row_bytes)
            or cache._arena.dtype.name != 'uint8'
            or cache._arena.nbytes > 67108864):
            raise RuntimeError('loaded Engram capacity differs from its host bound')
        banks.append({'layer': bank.layer_id, 'budget_bytes': cache.budget_bytes,
                      'row_bytes': cache.row_bytes, 'slot_count': cache.slot_count,
                      'arena_bytes': int(cache._arena.nbytes),
                      'resident_rows': cache.resident_rows, 'stats': dict(cache.stats)})
    return {'python_budget': budget, 'banks': banks, 'extra_helper_host_bytes': 33554432,
            'scope': 'Two fixed native byte arenas and original row LRU; metadata priced at256B per maximum row plus1GiB other Python capacity. Boundary observations only.'}
