"""Conservative Q8 extension of the established native/packed envelope."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
Q8_RESERVE = 503316480
REFERENCE_HOST = 640 * 1024**2


def resolve_admission(base, wired, *, grow, expected_receipt_hash):
    mode=os.environ.get('DSV41_Q8_MODE')
    if mode not in ('ar','dspark'):
        raise RuntimeError('explicit Q8 reference/candidate mode required')
    proof=json.loads((ROOT/'compat/installation.json').read_text())
    path=Path('/tmp/dsv41-q8-lifetimes-20260917/probe.json')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=proof['q8_change']['lifetime_probe_sha256']:
        raise RuntimeError('cache lifetime evidence changed')
    measured=json.loads(path.read_text())
    if (not measured['complete'] or measured['arms'][1]['prefill_peak_bytes'] > Q8_RESERVE
            or any(a['active_after_close_bytes'] > 2*1024**2 for a in measured['arms'])):
        raise RuntimeError('Q8 lifetime probe exceeds its additional reserve')
    spec=importlib.util.spec_from_file_location('native_packed_envelope',ROOT/'packed/packed_admission.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extra_host=REFERENCE_HOST if mode=='ar' else 0
    # Nominate capacity with both additions included. Then attribute the same
    # bytes correctly to actual baseline, allocator buffers, and Python/file I/O.
    original=module.resolve_admission(base+Q8_RESERVE+extra_host,
        wired+Q8_RESERVE+extra_host,grow=grow,expected_receipt_hash=expected_receipt_hash)
    r=dict(original)
    for key in ('prefill_active_bound_bytes','transition_start_active_bound_bytes',
                'steady_decode_active_bound_bytes','resize_active_bound_bytes','active_bound_bytes'):
        r[key]+=Q8_RESERVE
    r.update(baseline_bytes=base,wired_before_bytes=wired,
        host_reserve_bytes=original['host_reserve_bytes']+extra_host,
        allocator_limit_bytes=original['allocator_limit_bytes']+Q8_RESERVE,
        q8_additional_allocator_reserve_bytes=Q8_RESERVE,
        ar_reference_host_logit_reserve_bytes=extra_host,
        bound_scope='Unchanged native/packed M6/M8 envelope plus full fixed Q8 storage/copy/view reserve and separately reserved reference-logit file I/O; no native KV discount.')
    assert r['physical_bound_bytes'] <= 110000000000
    assert r['baseline_bytes']+r['host_reserve_bytes']+r['active_bound_bytes']+r['decode_cache_allowance_bytes'] <= 110000000000
    assert r['active_bound_bytes']+r['decode_cache_allowance_bytes'] <= r['allocator_limit_bytes']
    return r
