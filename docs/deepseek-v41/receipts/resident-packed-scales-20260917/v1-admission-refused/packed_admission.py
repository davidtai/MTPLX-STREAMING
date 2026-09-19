"""Derive this exact phase's bound from the retained native growth bound."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS = 17694720
RAW = 18800640
PACKED = 3086136060


def resolve_admission(base, wired, *, grow, expected_receipt_hash):
    if not grow:
        raise RuntimeError('packed phase requires its explicit one-request growth lane')
    spec = importlib.util.spec_from_file_location('native_growth_admission', '/tmp/dsv41-cache-growth-20260917/admission.py')
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
    old_capacity = original['prefill_slots_per_layer']
    native_capacity = original['decode_slots_per_layer']
    capacity = native_capacity + 1
    if capacity > 103:
        # The native-layout ownership/compile proof covered this maximum bank.
        capacity = 103
    raw_before = (40 * old_capacity + 48) * RAW
    weight_after = (40 * capacity + 48) * WEIGHTS
    net_growth = weight_after + PACKED - raw_before
    packed_peak = max(layer['packed_bytes'] for layer in inventory['layers'])
    native_copy = (2 * native_capacity - old_capacity) * 5898240
    packed_copy = (2 * capacity - old_capacity) * 5898240
    # Processing one layer at a time never owns a second full packed bank.
    # Charge one complete packed layer on top of the final net allocation and
    # the established one-component replacement+tail peak. 16MiB covers the
    # small paired-index/output buffers and partial-page artifact reads.
    extra = 16 * 1024**2
    if (net_growth + extra > original['growth_payload_bytes']
        or net_growth + packed_peak + packed_copy + extra > original['growth_payload_bytes'] + native_copy):
        raise RuntimeError('packed transition does not fit inside the admitted native envelope')
    result = dict(original)
    result.update(
        native_predecessor_decode_slots_per_layer=native_capacity,
        native_predecessor_growth_payload_bytes=original['growth_payload_bytes'],
        decode_slots_per_layer=capacity, growth_payload_bytes=net_growth,
        resident_packed_scales_bytes=PACKED, source_record_bytes=RAW,
        decode_weight_record_bytes=WEIGHTS, maximum_packed_layer_bytes=packed_peak,
        packed_transition_payload_and_copy_bound_bytes=net_growth + packed_peak + packed_copy + extra,
        native_transition_payload_and_copy_bound_bytes=original['growth_payload_bytes'] + native_copy,
        packed_inventory_manifest_sha256=proof['artifact_manifest_sha256'],
        bound_scope='unchanged prefill; packed decode and phase transition conservatively contained in native growth bounds',
    )
    return result
