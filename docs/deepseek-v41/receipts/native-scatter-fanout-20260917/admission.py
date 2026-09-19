"""Construction-time bounds for the one-request cache-growth experiment."""
import hashlib
import json
from pathlib import Path

GIB = 1024**3
SLOT_BAND = 40 * 18800640
# Additional host/kernel headroom for 20 fanout workers; no new record buffers.
IO_HEADROOM_BYTES = 512 * 1024**2
ROOT = Path('/tmp/dsv41-io-fanout8-20260917')


def resolve_admission(base, wired, *, grow, expected_receipt_hash):
    control_path = Path('/tmp/dsv41-110-stage/full-compiled-hc-post-ca207-20260917.jsonl')
    if hashlib.sha256(control_path.read_bytes()).hexdigest() != expected_receipt_hash:
        raise RuntimeError('phase-memory predecessor changed')
    row = json.loads(control_path.read_text())
    mem = row['dspark']['memory']
    if (row['resolved_plan']['slots_per_layer'] != 93
        or mem['mlx_peak_bytes'] != 95208121956
        or mem['mlx_peak_after_prefill_bytes'] != 88755252592
        or mem['mlx_active_bytes_at_decode_start'] != 86497096948):
        raise RuntimeError('phase predecessor no longer has the expected memory observations')
    allocator_limit = 110000000000 - base - 2 * GIB
    # Retain the prior, deliberately higher prefill bounds. They already price
    # Python, retained allocator storage, all prefill graphs and baseline drift.
    prefill_active_bound = 96461356124
    prefill_physical_bound = 108512945408 + max(0, base - 9586573312) + IO_HEADROOM_BYTES
    if (prefill_active_bound + GIB > allocator_limit
        or prefill_physical_bound > 109500000000
        or wired + prefill_active_bound + 2 * GIB + IO_HEADROOM_BYTES > 100 * GIB):
        raise RuntimeError('current baseline cannot admit the unchanged cap93 prefill')
    margin = 256 * 1024**2
    # After the cleared-cache boundary, overprice a possible retained buffer by
    # the entire observed seed/decode active increase, not merely the largest
    # resized component. This also covers the 1,006,632,960-byte hidden capture.
    cache_overshoot = mem['mlx_peak_after_prefill_bytes'] - mem['mlx_active_bytes_at_decode_start']
    cache_allowance = GIB + cache_overshoot
    transition_start = mem['mlx_active_bytes_at_decode_start'] + margin
    decode_capacity = 93
    decode_bound = mem['mlx_peak_after_prefill_bytes'] + margin
    resize_bound = transition_start
    if grow:
        probe = json.loads((ROOT/'probe-final-results.json').read_text())
        if (probe['helper_sha256'] != hashlib.sha256((ROOT/'bank_growth_final.py').read_bytes()).hexdigest()
            or probe['active_after_close_bytes'] > 1024**2
            or probe['row_sha256'][0] != '53891b5541b11c41c8d51aff28b53e19685897474dbf47ea0f4a57f967f95561'):
            raise RuntimeError('bounded byte-copy/ownership evidence changed')
        for capacity in (100,):
            delta = (capacity - 93) * SLOT_BAND
            padding = 40 * 6 * 16384
            steady = mem['mlx_peak_after_prefill_bytes'] + delta + padding + margin
            # At most one replacement plus its zero tail is under construction;
            # every old input is synchronized/released before the next copy.
            resize = transition_start + delta + padding + (2 * capacity - 93) * 5898240
            active = max(steady, resize)
            physical = base + 2 * GIB + active + cache_allowance + IO_HEADROOM_BYTES
            if (physical <= 109500000000
                and active + cache_allowance <= allocator_limit
                and wired + active + cache_allowance + GIB + IO_HEADROOM_BYTES <= 100 * GIB):
                decode_capacity, decode_bound, resize_bound = capacity, steady, resize
                break
        else:
            raise RuntimeError('the matched cap100 plus fanout8 headroom does not fit this baseline')
    delta = (decode_capacity - 93) * SLOT_BAND
    active_bound = max(prefill_active_bound, decode_bound, resize_bound)
    physical_bound = max(prefill_physical_bound,
        base + 2 * GIB + max(decode_bound, resize_bound) + cache_allowance + IO_HEADROOM_BYTES)
    return dict(prefill_slots_per_layer=93, decode_slots_per_layer=decode_capacity,
        growth_payload_bytes=delta, transition_start_active_bound_bytes=transition_start,
        steady_decode_active_bound_bytes=decode_bound, resize_active_bound_bytes=resize_bound,
        active_bound_bytes=active_bound, physical_bound_bytes=physical_bound,
        prefill_active_bound_bytes=prefill_active_bound,
        prefill_physical_bound_bytes=prefill_physical_bound,
        host_reserve_bytes=2 * GIB, additional_io_host_headroom_bytes=IO_HEADROOM_BYTES,
        requested_allocator_cache_limit_bytes=GIB,
        decode_cache_overshoot_allowance_bytes=cache_overshoot,
        decode_cache_allowance_bytes=cache_allowance, allocation_margin_bytes=margin,
        allocator_limit_bytes=allocator_limit, wired_before_bytes=wired,
        baseline_bytes=base, control_receipt=str(control_path),
        control_receipt_sha256=expected_receipt_hash)
