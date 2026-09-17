import hashlib, json
from pathlib import Path


def summarize(stem):
    p=Path(stem)
    output=p.with_suffix('.jsonl')
    if not output.exists():
        return {'path':str(output),'complete_receipt':False}
    row=json.loads(output.read_text()); d=row['dspark']; m=d['memory']
    counters=d['serve_stream_counters']; io=counters['io']; cache=counters['expert_cache']
    samples=[json.loads(line)['snapshot'] for line in p.with_suffix('.os.jsonl').read_text().splitlines()]
    growth=row.get('post_prefill_growth',{})
    return dict(path=str(output),receipt_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        complete_receipt=True,decode_tok_s=d['decode_tok_s'],decode_wall_s=d['decode_wall_s'],
        generated=d['n_generated'],cycles=d['cycles'],phase_time_s=d['phase_time_s'],
        prefill_slots=row.get('phase_memory_plans',{}).get('prefill',row['resolved_plan'])['slots_per_layer'],
        decode_slots=d['serve_stream_counters']['slot_plan']['slots_per_layer'],growth=growth,
        phase_plan_memory_limits={phase:plan['memory_limit_bytes'] for phase,plan in row.get('phase_memory_plans',{}).items()},
        allocator_peak_bytes=m['mlx_peak_bytes'],allocator_post_prefill_peak_bytes=m['mlx_peak_after_prefill_bytes'],
        sampled_process_peak_bytes=m['process_footprint_peak_bytes'],sampled_system_peak_bytes=m['system_used_peak_bytes'],
        external_sampled_process_peak_bytes=max(x['process']['phys_footprint_bytes'] for x in samples),
        external_sampled_system_peak_bytes=max(x['box']['used_bytes'] for x in samples),
        output_sha256=d['token_ids_sha256'],divergence=d.get('divergence'),
        physical_reads=io.get('records_read'), physical_read_operations=io.get('read_operations'),physical_bytes=io.get('read_bytes'),
        io_read_wall_ns=io.get('read_wall_ns'),io_realized_qd=io.get('read_realized_qd'),
        expert_hits=cache.get('expert_hits'),expert_misses=cache.get('expert_misses'))

if __name__=='__main__':
    import sys
    for stem in sys.argv[1:]: print(json.dumps(summarize(stem),indent=2))
