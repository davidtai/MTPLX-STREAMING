from collections import Counter
from mtplx.expert_streaming import LayerExpertSlotBank, _ExpertHistory

def restore(state, *, policy=None, decay=None, single_pool=None):
    kwargs = {key: state[key] for key in (
        'expert_count', 'persistent_slots', 'transient_slots',
        'frequency_decay', 'cache_policy', 'single_pool')}
    if policy is not None:
        kwargs['cache_policy'] = policy
    if decay is not None:
        kwargs['frequency_decay'] = decay
    if single_pool is not None:
        kwargs['single_pool'] = single_pool
    bank = LayerExpertSlotBank(**kwargs)
    for key in ('_slot_to_expert', '_pool_clock', '_decode_epoch',
                '_saw_decode_since_prefill'):
        setattr(bank, key, state[key])
    bank._slot_to_expert = list(bank._slot_to_expert)
    for key in ('_expert_to_slot', '_pool_recency'):
        setattr(bank, key, {int(k): v for k, v in state[key].items()})
    for key in ('_protected', '_prefill_seed_candidates'):
        setattr(bank, key, set(state[key]))
    bank._prefill_route_freq = Counter({int(k): v for k, v in state['_prefill_route_freq'].items()})
    bank._history = [_ExpertHistory(**h) for h in state['_history']]
    return bank
