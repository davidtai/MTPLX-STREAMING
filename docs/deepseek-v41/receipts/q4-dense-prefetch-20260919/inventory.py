"""CPU-only Q4 dense-prefetch byte inventory and native-route sensitivity."""
import collections
import gzip
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import time


class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('MLX prohibited in the CPU dense-prefetch inventory')


sys.meta_path.insert(0, NoMLX())
ROOT = Path(__file__).resolve().parent
REPO = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
ART = Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
MANIFEST = ART / 'expert-manifest.json'
TRACE = REPO / 'docs/deepseek-v41/receipts/mtp-verify-routes-20260913/mtp-verify-routes-16k-1024-v2.json.gz'
BASELINE = Path('/tmp/dsv41-hybrid-lookup-20260918/full-summary.json')
OUT = ROOT / 'inventory-result.json'


def main():
    if OUT.exists():
        raise RuntimeError('refusing to overwrite inventory evidence')
    started = time.perf_counter()
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    baseline = json.loads(BASELINE.read_text())
    groups = collections.defaultdict(list)
    for tensor in manifest['resident_tensors']:
        parts = tensor['tensor'].split('.')
        if parts[0] == 'layers' and parts[2] == 'attn' and parts[3] in ('wq_a', 'wq_b', 'wkv', 'wo_a', 'wo_b'):
            groups[parts[3]].append(tensor)
    candidates = {}
    weight_record = baseline['growth']['decode_weight_record_bytes']
    assert weight_record == 17694720 and baseline['cycles'] == 198
    for name, tensors in groups.items():
        by_layer = collections.defaultdict(list)
        for tensor in tensors:
            by_layer[int(tensor['tensor'].split('.')[1])].append(tensor)
        signature = lambda ts: sorted((t['tensor'].split('.')[-1], t['dtype'], tuple(t['shape']), t['length']) for t in ts)
        uniform = set(by_layer) == set(range(40)) and all(signature(ts) == signature(by_layer[0]) for ts in by_layer.values())
        layer_bytes = max(sum(t['length'] for t in ts) for ts in by_layer.values())
        source_total = sum(t['length'] for t in tensors)
        # wo_a is already materialized to BF16 by the saved Q4 route; count
        # its real resident representation, not the retired packed source.
        if name == 'wo_a':
            layer_bytes = 67108864
        resident_total = layer_bytes * 40 if name == 'wo_a' else source_total
        rotating = 2 * layer_bytes
        freed = resident_total - rotating
        added_slots = freed // (40 * weight_record)
        candidates[name] = {
            'uniform_layers': uniform, 'source_layer0_signature': signature(by_layer[0]),
            'source_bytes': source_total, 'resident_bytes': resident_total,
            'two_buffer_bytes': rotating, 'payload_freed_bytes': freed,
            'additional_uniform_expert_slots_payload_only': added_slots,
            'payload_projection_slots_per_layer': 110 + added_slots,
            'bytes_per_verify_if_streaming_resident_representation': resident_total,
            'additional_dense_bytes_for_historical_198_calls': resident_total * baseline['cycles'],
            'sources': tensors,
        }
    del manifest, raw
    from restore import restore
    from mtplx.expert_streaming import RoutingPhase
    trace = json.loads(gzip.decompress(TRACE.read_bytes()))
    assert trace['complete'] and trace['cycles'] == 206 and trace['slots_per_layer'] == 73
    capacities = sorted({73, 110, *(c['payload_projection_slots_per_layer'] for c in candidates.values())})
    replay = {}
    for capacity in capacities:
        counts = []
        for layer in range(40):
            bank = restore(trace['initial_banks'][str(layer)],
                           policy=None if capacity == 73 else 'transition-window',
                           single_pool=True, layer_id=layer)
            extra = capacity - bank.persistent_slots
            bank._slot_to_expert.extend([None] * extra)
            bank.persistent_slots = bank._persistent_capacity = capacity
            bank.slot_count += extra
            bank._protected_cap = max(1, int(capacity * .8))
            count = 0
            for route in trace['target_routes_by_layer'][str(layer)]:
                plan = bank.try_plan_all_hits(route, phase=RoutingPhase.DECODE)
                if plan is None:
                    plan = bank.plan(route, phase=RoutingPhase.DECODE)
                count += len(plan.loads)
            counts.append(count)
        if capacity == 73 and sum(counts) != trace['decode_records_read']:
            raise RuntimeError('unchanged historical replay differs')
        replay[capacity] = {'reads': sum(counts), 'reads_by_layer': counts}
    for name, candidate in candidates.items():
        capacity = candidate['payload_projection_slots_per_layer']
        saved = (replay[110]['reads'] - replay[capacity]['reads']) * weight_record
        dense = candidate['resident_bytes'] * trace['cycles']
        candidate['trace206_saved_expert_read_bytes'] = saved
        candidate['trace206_added_dense_read_bytes'] = dense
        candidate['trace206_net_added_read_bytes'] = dense - saved
    report = {
        'complete': True, 'cpu_only': True, 'elapsed_s': time.perf_counter() - started,
        'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        'trace_sha256': hashlib.sha256(TRACE.read_bytes()).hexdigest(),
        'baseline_sha256': hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), ROOT / 'restore.py', REPO / 'mtplx/expert_streaming.py')},
        'historical_best': {k: baseline[k] for k in ('tps', 'wall_s', 'cycles', 'capacity', 'io')},
        'candidates': candidates, 'capacity_replay': replay,
        'scope': 'Metadata and fixed native206-cycle route sensitivity only. The full winner has198 hybrid cycles. No measured streaming, overlap, fresh admission, or throughput prediction. Staging/padding/growth copies must be priced separately.',
        'mlx_imported': any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules),
    }
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('complete', 'elapsed_s', 'mlx_imported')}))
    for name, candidate in candidates.items():
        print(json.dumps({'family': name, **{k: v for k, v in candidate.items() if k not in ('sources', 'source_layer0_signature')}}))


if __name__ == '__main__':
    main()
