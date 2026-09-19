"""Conservative admission for one D5/M6 teacher capture at cap84; no growth."""
import hashlib
import json
from pathlib import Path
GIB=1024**3
SLOT_BAND=40*18800640
TRACE_ALLOWANCE=256*1024**2

def resolve_admission(base,wired,*,grow,expected_receipt_hash):
    if grow: raise RuntimeError('teacher capture uses fixed cap84, no growth')
    control_path=Path('/tmp/dsv41-110-stage/full-compiled-hc-post-ca207-20260917.jsonl')
    if hashlib.sha256(control_path.read_bytes()).hexdigest()!=expected_receipt_hash:
        raise RuntimeError('phase memory predecessor changed')
    row=json.loads(control_path.read_text()); mem=row['dspark']['memory']
    if (row['resolved_plan']['slots_per_layer']!=93 or mem['mlx_peak_bytes']!=95208121956
        or mem['mlx_peak_after_prefill_bytes']!=88755252592
        or mem['mlx_active_bytes_at_decode_start']!=86497096948):
        raise RuntimeError('phase predecessor observations differ')
    saved=(93-84)*SLOT_BAND
    # Price the full previous conservative peak, less only fixed bank storage,
    # plus a further 1GiB geometry margin. Keep the previous oversized-cache
    # allowance and add 256MiB for fixed CPU trace storage plus output file cache.
    active=96461356124-saved+GIB
    cache_allowance=GIB+mem['mlx_peak_after_prefill_bytes']-mem['mlx_active_bytes_at_decode_start']
    limit=110000000000-base-2*GIB-TRACE_ALLOWANCE
    physical=base+2*GIB+TRACE_ALLOWANCE+active+cache_allowance
    if (physical>109500000000 or active+cache_allowance>limit
        or wired+active+cache_allowance+2*GIB+TRACE_ALLOWANCE>100*GIB):
        raise RuntimeError(f'cap84 teacher bound refused: baseline={base}, physical_bound={physical}, allocator_limit={limit}, wired={wired}')
    return dict(prefill_slots_per_layer=84,decode_slots_per_layer=84,growth_payload_bytes=0,
        transition_start_active_bound_bytes=mem['mlx_active_bytes_at_decode_start']-saved+GIB,
        steady_decode_active_bound_bytes=mem['mlx_peak_after_prefill_bytes']-saved+GIB,
        resize_active_bound_bytes=0,active_bound_bytes=active,physical_bound_bytes=physical,
        prefill_active_bound_bytes=active,prefill_physical_bound_bytes=physical,
        host_reserve_bytes=2*GIB,trace_host_and_file_cache_allowance_bytes=TRACE_ALLOWANCE,
        extra_geometry_margin_bytes=GIB,fixed_bank_storage_saving_bytes=saved,
        requested_allocator_cache_limit_bytes=GIB,decode_cache_allowance_bytes=cache_allowance,
        allocator_limit_bytes=limit,wired_before_bytes=wired,baseline_bytes=base,
        control_receipt=str(control_path),control_receipt_sha256=expected_receipt_hash)
