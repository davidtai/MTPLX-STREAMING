"""Pure-Python checks for seeded-pool transition into bounded decode protection."""
import pytest
from mtplx.expert_streaming import LayerExpertSlotBank


def seeded_bank():
    bank = LayerExpertSlotBank(expert_count=16, persistent_slots=5,
                               transient_slots=6, single_pool=True)
    seed = bank.prepare_prefill_seed([0] * 5 + [1] * 4 + [2] * 3 + [3] * 2 + [4])
    bank.plan(seed, phase='prefill')
    assert bank._protected == set(range(5))
    return bank


@pytest.mark.parametrize('all_hit', [False, True])
def test_decode_opens_probation_without_eviction(all_hit):
    bank = seeded_bank()
    before = dict(bank._expert_to_slot)
    route = bank.try_plan_all_hits if all_hit else bank.plan
    plan = route([0], phase='decode')
    assert plan.misses == ()
    assert bank._expert_to_slot == before
    assert bank._protected == {0, 1, 2, 3}
    # The demoted seed can earn protection on a subsequent hit.
    route([4], phase='decode')
    assert 4 in bank._protected and len(bank._protected) == bank._protected_cap


@pytest.mark.parametrize('all_hit', [False, True])
def test_failed_decode_transaction_restores_prefill_protection(all_hit):
    bank = seeded_bank()
    before = dict(bank._expert_to_slot), dict(bank._pool_recency)
    route = bank.try_plan_all_hits_transaction if all_hit else bank.plan_transaction
    plan, transaction = route([0], phase='decode')
    assert len(bank._protected) == bank._protected_cap
    transaction.rollback_completion()
    assert bank._protected == set(range(5))
    assert (bank._expert_to_slot, bank._pool_recency) == before
    assert not bank._saw_decode_since_prefill


def test_prefill_chunks_remain_protected_until_decode_and_reopen_next_request():
    bank = seeded_bank()
    bank.plan([5, 6, 7], phase='prefill')
    assert bank._protected == set(range(5))
    bank.try_plan_all_hits([0], phase='decode')
    assert len(bank._protected) == bank._protected_cap
    bank.prepare_prefill_seed([8] * 5 + [9] * 4 + [10] * 3 + [11] * 2 + [12])
    bank.plan([8, 9, 10, 11, 12], phase='prefill')
    assert bank._protected == {8, 9, 10, 11, 12}
    bank.plan([8], phase='decode')
    assert bank._protected == {8, 9, 10, 11}
