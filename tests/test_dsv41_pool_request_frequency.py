"""Single-pool admission must rank the current prompt, with no MLX import."""

from mtplx.expert_streaming import LayerExpertSlotBank


def _previous_request():
    bank = LayerExpertSlotBank(
        expert_count=8, persistent_slots=2, transient_slots=4, single_pool=True,
    )
    bank.prepare_prefill_seed([0] * 100 + [1])
    bank.plan([0, 1], phase="prefill")
    bank.plan([2, 3], phase="decode")  # ordinary churn removes the first seed
    return bank


def test_new_request_seeds_and_evicts_by_its_own_frequency():
    bank = _previous_request()
    bank.prepare_prefill_seed([0] + [1] * 10)
    seeded = bank.plan([0, 1], phase="prefill")
    assert seeded.misses == (0, 1)  # colder seed first, hotter seed most recent
    assert bank.pin_working_set(top_k=1) == (1,)
    bank.clear_pins()
    decoded = bank.plan([2], phase="decode")
    assert [eviction.previous_expert for eviction in decoded.evictions] == [0]


def test_same_prompt_chunks_still_accumulate_frequency():
    bank = _previous_request()
    bank.prepare_prefill_seed([0] + [1] * 10)
    bank.plan([0, 1], phase="prefill")
    bank.prepare_prefill_seed([0] * 20 + [1])
    assert bank._prefill_route_freq == {0: 21, 1: 11}
    assert bank.pin_working_set(top_k=1) == (0,)


def test_unseeded_new_request_clears_frequency_and_rollback_restores_it():
    bank = _previous_request()
    before = dict(bank._prefill_route_freq)
    plan, transaction = bank.plan_transaction([4], phase="prefill")
    assert plan.misses == (4,)
    assert not bank._prefill_route_freq
    transaction.rollback_publication()
    assert bank._prefill_route_freq == before
    assert bank._saw_decode_since_prefill
    assert set(bank.resident_experts) == {2, 3}
    bank.prepare_prefill_seed([0] + [1] * 10)
    assert bank._prefill_route_freq == {0: 1, 1: 10}
