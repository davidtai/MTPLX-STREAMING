"""Construction-only bounds for native D7/M8 and its matched D5/M6 control."""
import hashlib
import json
import os
from pathlib import Path
from mtplx.deepseek_v41_memory_profile import DEFAULT_BOX_BUDGET_BYTES

GIB = 1024**3
SLOT_BAND = 40 * 18800640
ROOT = Path('/tmp/dsv41-woa-owner-20260917/full/native')


def resolve_admission(base, wired, *, grow, expected_receipt_hash):
    if not grow:
        raise RuntimeError('this width comparison requires charged post-prefill growth')
    proof = json.loads((ROOT/'installation.json').read_text())
    config_path = Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4/config.json')
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != proof['native_config_sha256']:
        raise RuntimeError('native configuration differs from the allocation proof')
    probe_path = Path(proof['native_m8_probe_path'])
    probe_bytes = probe_path.read_bytes()
    if hashlib.sha256(probe_bytes).hexdigest() != proof['native_m8_probe_sha256']:
        raise RuntimeError('native M8 allocation evidence changed')
    probe = json.loads(probe_bytes)
    if (probe['generated_tokens'] != 129
        or not probe['prefix_identical_to_retained_control']
        or probe['resolved_plan'] != dict(prefill_slots=16, decode_slots=16,
            transient_slots=48, effective_depth=7, verify_chunks=[8])):
        raise RuntimeError('native M8 allocation probe did not cover the required route')
    control_path = Path('/tmp/dsv41-110-stage/full-compiled-hc-post-ca207-20260917.jsonl')
    control_bytes = control_path.read_bytes()
    if hashlib.sha256(control_bytes).hexdigest() != expected_receipt_hash:
        raise RuntimeError('complete phase-memory predecessor changed')
    control = json.loads(control_bytes)
    mem = control['dspark']['memory']
    if (control['resolved_plan']['slots_per_layer'] != 93
        or mem['mlx_peak_after_prefill_bytes'] != 88755252592
        or mem['mlx_active_bytes_at_decode_start'] != 86497096948):
        raise RuntimeError('complete native memory geometry changed')
    copy_probe = json.loads((ROOT/'probe-final-results.json').read_text())
    if (copy_probe['helper_sha256'] != hashlib.sha256((ROOT/'bank_growth_final.py').read_bytes()).hexdigest()
        or copy_probe['active_after_close_bytes'] > 1024**2):
        raise RuntimeError('bounded bank-copy ownership evidence changed')

    initial = 84
    margin = 256 * 1024**2
    padding = 40 * 6 * 16384
    host_reserve = 2 * GIB + 16 * 1024**2  # price projection-owner Python metadata in every phase
    allocator_limit = DEFAULT_BOX_BUDGET_BYTES - base - host_reserve
    # Preserve the larger original prefill bound, also checking the native M8
    # probe normalized only by exact persistent-bank bytes. Price extra scale
    # pages, additional allocator variation, and retained cache separately.
    prefill_active = max(96461356124 + (initial - 93) * SLOT_BAND,
        probe['prefill']['peak_bytes'] + (initial - 16) * SLOT_BAND) + margin + padding
    prefill_cache = max(GIB, probe['prefill']['cache_bytes']) + 64 * 1024**2
    prefill_physical = base + host_reserve + prefill_active + prefill_cache
    if (prefill_physical > DEFAULT_BOX_BUDGET_BYTES
        or prefill_active + prefill_cache > allocator_limit
        or wired + prefill_active + prefill_cache + GIB > 100 * GIB):
        raise RuntimeError(f'cap84 prefill cannot fit the current baseline: {prefill_physical}B')

    # The short probe already crosses the first post-16K grow-buffer boundary.
    # Still add the ENTIRE 17,664-token logical compressed-KV allowance to the
    # larger of full D5 and normalized native M8 peaks, not just 896 extra rows.
    # Config-pinned kv_source layers2/8/14 use ratio2, layer20 ratio1:
    # (3/2+1)*(512+128)*bf16 = 3,200B per token. Window rings are fixed.
    kv_allowance = 17664 * 3200
    overshoot = max(mem['mlx_peak_after_prefill_bytes'] - mem['mlx_active_bytes_at_decode_start'],
        probe['post_prefill']['peak_bytes'] - probe['prefill']['active_bytes'])
    cache_allowance = GIB + overshoot
    transition_start = max(mem['mlx_active_bytes_at_decode_start'] + (initial - 93) * SLOT_BAND,
        probe['prefill']['active_bytes'] + (initial - 16) * SLOT_BAND) + margin + padding
    required = os.environ.get('DSV41_REQUIRE_DECODE_CAP')
    capacities = [int(required)] if required is not None else range(100, 84, -1)
    for capacity in capacities:
        if capacity not in range(85, 101):
            raise RuntimeError('explicit matched decode capacity must be85..100')
        delta = (capacity - initial) * SLOT_BAND
        steady = max(mem['mlx_peak_after_prefill_bytes'] + (capacity - 93) * SLOT_BAND,
            probe['post_prefill']['peak_bytes'] + (capacity - 16) * SLOT_BAND)
        steady += margin + padding + kv_allowance
        # One component replacement/zero-tail at a time; synchronize and release
        # old backing before constructing the next component (measured helper).
        resize = transition_start + delta + padding + (2 * capacity - initial) * 5898240
        active = max(steady, resize)
        physical = base + host_reserve + active + cache_allowance
        if (physical <= DEFAULT_BOX_BUDGET_BYTES and active + cache_allowance <= allocator_limit
            and wired + active + cache_allowance + GIB <= 100 * GIB):
            break
    else:
        raise RuntimeError('no allowed matched native decode capacity fits the current baseline')
    return dict(physical_ceiling_bytes=DEFAULT_BOX_BUDGET_BYTES, prefill_slots_per_layer=initial, decode_slots_per_layer=capacity,
        growth_payload_bytes=delta, transition_start_active_bound_bytes=transition_start,
        steady_decode_active_bound_bytes=steady, resize_active_bound_bytes=resize,
        active_bound_bytes=max(prefill_active, active),
        physical_bound_bytes=max(prefill_physical, physical),
        prefill_active_bound_bytes=prefill_active, prefill_physical_bound_bytes=prefill_physical,
        prefill_cache_allowance_bytes=prefill_cache, host_reserve_bytes=host_reserve, projection_host_allowance_bytes=16 * 1024**2,
        requested_allocator_cache_limit_bytes=GIB, decode_cache_overshoot_allowance_bytes=overshoot,
        decode_cache_allowance_bytes=cache_allowance, allocation_margin_bytes=margin,
        full_logical_kv_extra_allowance_bytes=kv_allowance, page_padding_allowance_bytes=padding,
        allocator_limit_bytes=allocator_limit, wired_before_bytes=wired, baseline_bytes=base,
        control_receipt=str(control_path), control_receipt_sha256=expected_receipt_hash,
        native_m8_probe=str(probe_path), native_m8_probe_sha256=proof['native_m8_probe_sha256'],
        capacity_selection='largest admitted85..100; explicit fixed value required for a fresh matched control')
