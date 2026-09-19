"""Derive this exact phase's bound from the retained native growth bound."""
import hashlib
import importlib.util
import json
from pathlib import Path
from mtplx.deepseek_v41_memory_profile import DEFAULT_BOX_BUDGET_BYTES

ROOT = Path(__file__).resolve().parent
WEIGHTS = 17694720
RAW = 18800640
PACKED = 3086136060


def resolve_admission(base, wired, *, grow, expected_receipt_hash, strict_allocator):
    if not grow:
        raise RuntimeError('packed phase requires its explicit one-request growth lane')
    # The separately proved native M8 envelope includes cap84 prefill and bounds
    # both native M6 and M8. Retain its extra KV, padding and compiler margins.
    spec = importlib.util.spec_from_file_location('native_growth_admission', '/private/tmp/dsv41-extension-bank-20260919/full-v1/native/admission.py')
    installation = json.loads((ROOT / 'installation.json').read_text())
    if hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest() != installation['native_admission_sha256']:
        raise RuntimeError('native allocation-bound source changed')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.resolve_admission(base, wired, grow=True, expected_receipt_hash=expected_receipt_hash)
    inventory = json.loads((ROOT / 'artifact/manifest.json').read_text())
    proof = json.loads((ROOT / 'storage-probe.json').read_text())
    if (not inventory['complete'] or inventory['packed_bytes'] != PACKED
        or not proof['complete'] or proof['active_after_close_bytes'] > 2 * 1024**2
        or proof['artifact_manifest_sha256'] != hashlib.sha256((ROOT / 'artifact/manifest.json').read_bytes()).hexdigest()):
        raise RuntimeError('full inventory or physical ownership proof changed')
    for name, digest in proof['helper_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('bounded operator/storage helper changed')
    strict = installation['strict_allocator']
    if strict_allocator != strict['identity']:
        raise RuntimeError('strict allocator was not attested at construction')
    for operator in strict['operators'].values():
        blob = Path(operator['path']).read_bytes()
        row = json.loads(blob)
        if hashlib.sha256(blob).hexdigest() != operator['sha256'] or not row['complete'] or not row[operator['exact_field']]:
            raise RuntimeError('matched allocator operator evidence changed')
    removed_overshoot = original['decode_cache_overshoot_allowance_bytes']
    if removed_overshoot != strict['overshoot_allowance_removed_bytes']:
        raise RuntimeError('original free-cache overshoot allowance changed')
    original['decode_cache_overshoot_allowance_bytes'] = 0
    prefill_requested_cache = original['requested_allocator_cache_limit_bytes']
    original['requested_allocator_cache_limit_bytes'] = 256 * 1024**2
    original['prefill_requested_allocator_cache_limit_bytes'] = prefill_requested_cache
    original['decode_cache_allowance_bytes'] = 256 * 1024**2 + original['decode_cache_overshoot_allowance_bytes']
    old_capacity = original['prefill_slots_per_layer']
    native_capacity = original['decode_slots_per_layer']
    raw_before = (40 * old_capacity + 48) * RAW
    packed_peak = max(layer['packed_bytes'] for layer in inventory['layers'])
    # Processing one layer at a time never owns a second full packed bank.
    # Charge one complete packed layer on top of the final net allocation and
    # the established one-component replacement+tail peak. 16MiB covers the
    # small paired-index/output buffers and partial-page artifact reads.
    extra = (16 + 64) * 1024**2  # price all additional plane roots and scheduler state
    # Two completed, exact runs at cap101 isolate the shorter retained prompt
    # state before bank growth. Projection retirement has not engaged here.
    phase_proof = installation['memory_composition']
    controls = {}
    for name in ('native_owner', 'tail'):
        path = Path(phase_proof[name + '_receipt'])
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != phase_proof[name + '_sha256']:
            raise RuntimeError('full phase ownership proof changed')
        row = json.loads(blob)
        if (row['post_prefill_growth']['prefill_slots_per_layer'] != 84
            or row['post_prefill_growth']['decode_slots_per_layer'] != 101
            or row['dspark']['n_generated'] != 1024
            or row['dspark']['token_ids_sha256'] != '0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac'):
            raise RuntimeError('full phase ownership shape or output changed')
        controls[name] = row
    native_growth = controls['native_owner']['post_prefill_growth']
    tail_growth = controls['tail']['post_prefill_growth']
    tail_credit = native_growth['active_before_bytes'] - tail_growth['active_before_bytes']
    if (tail_credit != 880803840
        or native_growth['active_after_bytes'] - tail_growth['active_after_bytes'] != tail_credit
        or tail_growth['growth_payload_bytes'] != native_growth['growth_payload_bytes']):
        raise RuntimeError('shorter prompt ownership saving changed')
    seed_peak = controls['tail']['dspark']['memory']['seed_prefill_boundary']['after']['mlx_peak_bytes']
    if seed_peak != 90793313528:
        raise RuntimeError('shorter native seed peak changed')
    projection_source_bytes = 40 * 34603008
    projection_cold_overlap = 34603008
    projection_credit = projection_source_bytes - projection_cold_overlap
    embedding = installation['embedding_optimization']
    blob = (ROOT / 'embedding-probe.json').read_bytes()
    operator = json.loads(blob)
    if (hashlib.sha256(blob).hexdigest() != embedding['probe_sha256']
            or not operator['complete'] or not operator['all_outputs_exact']
            or not operator['eviction_ownership_exact']
            or operator['released_native_bytes'] != 1323827200
            or operator['active_after_close_bytes'] != 0):
        raise RuntimeError('bounded input-row ownership proof changed')
    embedding_credit = 1323827200
    embedding_host = 32 * 1024**2
    lookup_host = 16 * 1024**2
    expansion_host = 16 * 1024**2
    expansion_credit = 40*67108864 - 40*34603008 - 3*67108864
    initial_net_growth = (40*old_capacity+48)*WEIGHTS + PACKED - raw_before
    allocator_limit = original['allocator_limit_bytes'] - embedding_host - lookup_host - expansion_host
    if (original['prefill_active_bound_bytes'] + original['prefill_cache_allowance_bytes'] > allocator_limit
            or original['prefill_physical_bound_bytes'] + embedding_host + lookup_host + expansion_host > DEFAULT_BOX_BUDGET_BYTES):
        raise RuntimeError('input-row host allowance does not fit unchanged prefill')
    for capacity in range(112, old_capacity, -1):
        weight_after = (40 * capacity + 48) * WEIGHTS
        net_growth = weight_after + PACKED - raw_before
        overflow_payload = (capacity-old_capacity)*40*WEIGHTS
        steady = original['steady_decode_active_bound_bytes'] - original['growth_payload_bytes'] + net_growth + extra - projection_credit
        # Negative final scale delta cannot be credited at the start. Price
        # the entire packed-scale inventory plus a layer staging allowance
        # on top of the unchanged84-row transition start. No weight copies.
        resize = (original['transition_start_active_bound_bytes'] + PACKED + packed_peak
                  + original['page_padding_allowance_bytes'] + extra - tail_credit)
        seed = seed_peak + (old_capacity - 101) * 40 * WEIGHTS + original['allocation_margin_bytes'] + original['page_padding_allowance_bytes']
        steady -= embedding_credit
        resize -= embedding_credit
        seed -= embedding_credit
        steady -= expansion_credit
        # Price all new packed rows, one temporary native scale owner, three
        # BF16 expansions including cold compilation, and extra page padding.
        append_peak = seed + overflow_payload + (capacity-old_capacity)*(RAW-WEIGHTS) + 3*67108864 + original['page_padding_allowance_bytes']
        active = max(steady, resize, seed, append_peak)
        physical = base + original['host_reserve_bytes'] + embedding_host + lookup_host + expansion_host + active + original['decode_cache_allowance_bytes']
        if (physical <= DEFAULT_BOX_BUDGET_BYTES
            and active + original['decode_cache_allowance_bytes'] <= allocator_limit
            and wired + active + original['decode_cache_allowance_bytes'] + 1024**3 <= 100 * 1024**3):
            break
    else:
        raise RuntimeError('no packed decode capacity fits the live memory envelope')
    result = dict(original)
    result.update(
        physical_ceiling_bytes=DEFAULT_BOX_BUDGET_BYTES,
        allocator_limit_bytes=allocator_limit,
        native_host_reserve_bytes=original['host_reserve_bytes'],
        host_reserve_bytes=original['host_reserve_bytes'] + embedding_host + lookup_host + expansion_host,
        prefill_physical_bound_bytes=original['prefill_physical_bound_bytes'] + embedding_host + lookup_host + expansion_host,
        embedding_host_allowance_bytes=embedding_host,
        lookup_host_allowance_bytes=lookup_host,
        embedding_post_prefill_credit_bytes=embedding_credit,
        embedding_prefill_credit_bytes=0,
        transition_start_active_bound_bytes=original['transition_start_active_bound_bytes'] - embedding_credit,
        strict_allocator=dict(strict_allocator),
        strict_cache_overshoot_credit_bytes=removed_overshoot,
        capacity_search_ceiling=112,
        existing_bank_rows=84, maximum_extension_bank_rows=28,
        plane_overlap_extra_live_allowance_bytes=64 * 1024**2,
        projection_source_bytes_retired=0,
        projection_source_bytes_retained=projection_source_bytes,
        predictable_expansion_host_allowance_bytes=expansion_host,
        predictable_expansion_steady_credit_vs_cached_bf16_bytes=expansion_credit,
        predictable_expansion_retained_bf16_bytes=2*67108864,
        predictable_expansion_replacement_bf16_bound_bytes=3*67108864,
        projection_cold_source_overlap_allowance_bytes=projection_cold_overlap,
        inherited_projection_bound_normalization_bytes=projection_credit,
        projection_steady_credit_bytes=projection_credit+expansion_credit,
        projection_steady_credit_reference='Native predecessor retaining packed and all BF16 target projections; includes the inherited cold-source slack.',
        projection_prefill_and_transition_credit_bytes=0,
        projection_seed_credit_bytes=0, seed_active_bound_bytes=seed,
        seed_control_receipt_sha256=phase_proof['tail_sha256'],
        tail_transition_active_credit_bytes=tail_credit, tail_steady_credit_bytes=0,
        bounded_engram_host_inventory_bytes=1338109952, helper_host_allowance_bytes=32 * 1024**2,
        resident_plan_retains_source_reserve=True,
        native_predecessor_decode_slots_per_layer=native_capacity,
        native_predecessor_growth_payload_bytes=original['growth_payload_bytes'],
        initial_decode_slots_per_layer=old_capacity, initial_growth_payload_bytes=initial_net_growth,
        decode_slots_per_layer=capacity, growth_payload_bytes=net_growth,
        overflow_payload_bytes=overflow_payload, overflow_append_active_bound_bytes=append_peak,
        resident_packed_scales_bytes=PACKED, source_record_bytes=RAW,
        decode_weight_record_bytes=WEIGHTS, maximum_packed_layer_bytes=packed_peak,
        packed_transition_payload_and_copy_bound_bytes=PACKED + packed_peak + extra,
        steady_decode_active_bound_bytes=steady, resize_active_bound_bytes=resize,
        active_bound_bytes=max(original['prefill_active_bound_bytes'], active),
        physical_bound_bytes=max(original['prefill_physical_bound_bytes'] + embedding_host + lookup_host + expansion_host, physical),
        packed_inventory_manifest_sha256=proof['artifact_manifest_sha256'],
        native_capacity_selection=original['capacity_selection'],
        capacity_selection='largest admitted85..112 final slots;84 original rows plus1..28 extension rows after native seed, no expert-bank copies',
        bound_scope='Unchanged84-row prefill. First transition retains84 expert rows, retires raw scales and installs packed scales; price the entire packed inventory plus one packed layer above transition-start instead of crediting its negative final delta. Native MTP seed completes at84 rows. Extension peak includes all final added packed rows, one full extension-bank raw-scale temporary, three BF16 projection arrays and page padding. Existing256MiB allocation margin, nativeM8/KV envelope, 256MiB strict allocator cache,100GiB wired ceiling and110GB whole-machine ceiling retained. Packed projection credit only in steady state. Extra16MiB schedule/bank metadata reserve remains in every phase. No existing expert backing array is resized or copied.',
    )
    return result
