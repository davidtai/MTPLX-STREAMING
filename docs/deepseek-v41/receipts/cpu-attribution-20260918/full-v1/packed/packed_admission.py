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
    spec = importlib.util.spec_from_file_location('native_growth_admission', '/private/tmp/dsv41-cpu-attribution-20260918/full-v1/native/admission.py')
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
    diagnostic_host = 16 * 1024**2
    diagnostic_active = 0
    allocator_limit = original['allocator_limit_bytes'] - diagnostic_host
    if original['prefill_active_bound_bytes'] + original['prefill_cache_allowance_bytes'] > allocator_limit:
        raise RuntimeError('diagnostic reserve does not fit the unchanged prefill envelope')
    if original['prefill_physical_bound_bytes'] + diagnostic_host > DEFAULT_BOX_BUDGET_BYTES:
        raise RuntimeError('diagnostic host reserve does not fit the prefill physical bound')
    for capacity in range(strict['capacity_search_ceiling'], old_capacity, -1):
        weight_after = (40 * capacity + 48) * WEIGHTS
        net_growth = weight_after + PACKED - raw_before
        packed_copy = (2 * capacity - old_capacity) * 5898240
        steady = original['steady_decode_active_bound_bytes'] - original['growth_payload_bytes'] + net_growth + extra - projection_credit
        resize = (original['transition_start_active_bound_bytes'] + net_growth + packed_peak
                  + packed_copy + original['page_padding_allowance_bytes'] + extra - tail_credit)
        seed = seed_peak + (capacity - 101) * 40 * WEIGHTS + original['allocation_margin_bytes'] + original['page_padding_allowance_bytes']
        steady += diagnostic_active
        resize += diagnostic_active
        seed += diagnostic_active
        active = max(steady, resize, seed)
        physical = base + original['host_reserve_bytes'] + diagnostic_host + active + original['decode_cache_allowance_bytes']
        if (physical <= DEFAULT_BOX_BUDGET_BYTES
            and active + original['decode_cache_allowance_bytes'] <= allocator_limit
            and wired + active + original['decode_cache_allowance_bytes'] + 1024**3 <= 100 * 1024**3):
            break
    else:
        raise RuntimeError('no packed decode capacity fits the live memory envelope')
    result = dict(original)
    result.update(
        physical_ceiling_bytes=DEFAULT_BOX_BUDGET_BYTES,
        strict_allocator=dict(strict_allocator),
        strict_cache_overshoot_credit_bytes=removed_overshoot,
        capacity_search_ceiling=strict['capacity_search_ceiling'],
        allocator_limit_bytes=allocator_limit,
        host_reserve_bytes=original['host_reserve_bytes'] + diagnostic_host,
        native_host_reserve_bytes=original['host_reserve_bytes'],
        prefill_physical_bound_bytes=original['prefill_physical_bound_bytes'] + diagnostic_host,
        diagnostic_host_allowance_bytes=diagnostic_host,
        diagnostic_active_allowance_bytes=diagnostic_active,
        plane_overlap_extra_live_allowance_bytes=64 * 1024**2,
        projection_source_bytes_retired=projection_source_bytes,
        projection_cold_source_overlap_allowance_bytes=projection_cold_overlap,
        projection_steady_credit_bytes=projection_credit,
        projection_prefill_and_transition_credit_bytes=0,
        projection_seed_credit_bytes=0, seed_active_bound_bytes=seed,
        seed_control_receipt_sha256=phase_proof['tail_sha256'],
        tail_transition_active_credit_bytes=tail_credit, tail_steady_credit_bytes=0,
        bounded_engram_host_inventory_bytes=1338109952, helper_host_allowance_bytes=32 * 1024**2,
        resident_plan_retains_source_reserve=True,
        native_predecessor_decode_slots_per_layer=native_capacity,
        native_predecessor_growth_payload_bytes=original['growth_payload_bytes'],
        decode_slots_per_layer=capacity, growth_payload_bytes=net_growth,
        resident_packed_scales_bytes=PACKED, source_record_bytes=RAW,
        decode_weight_record_bytes=WEIGHTS, maximum_packed_layer_bytes=packed_peak,
        packed_transition_payload_and_copy_bound_bytes=net_growth + packed_peak + packed_copy + extra,
        steady_decode_active_bound_bytes=steady, resize_active_bound_bytes=resize,
        active_bound_bytes=max(original['prefill_active_bound_bytes'], active),
        physical_bound_bytes=max(original['prefill_physical_bound_bytes'] + diagnostic_host, physical),
        packed_inventory_manifest_sha256=proof['artifact_manifest_sha256'],
        native_capacity_selection=original['capacity_selection'],
        capacity_selection='largest admitted packed capacity above prefill through110; explicit16MiB diagnostic host and no additional tensor owners',
        bound_scope='Original cap84 prefill envelope retained; exact full tail capture credit applies only during growth; independently measured tail seed bound, with no projection credit; native steady envelope credits packed projection retirement, with no tail credit; original KV/compiler/wired allowances and32MiB helper host reserve retained; only the separately added inactive-cache overshoot allowance is removed under the attested strict allocator; actual256MiB allocator cache applies only after prefill, with64MiB native Engram arenas.',
    )
    return result
