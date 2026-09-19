"""Summarize a terminal full-run receipt without loading MLX."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
PREFIX=Path('/tmp/dsv41-110-stage/full-miss-part1-20260918-v1')
receipt_path=Path(str(PREFIX)+'.jsonl')
receipt=json.loads(receipt_path.read_text())
prior=json.loads(Path('/tmp/dsv41-110-stage/full-memory-compose-20260918-v1.jsonl').read_text())
guard=Path(str(PREFIX)+'.guard.log').read_text()
bound=json.loads(next(line.split(' ',1)[1] for line in guard.splitlines() if line.startswith('MTP_BOUND ')))
line=next(line for line in guard.splitlines() if 'GPU step exited with code ' in line)
def capture(pattern):
    match=re.search(pattern,line)
    if not match:
        raise RuntimeError('missing terminal guard field: '+pattern)
    return int(match.group(1))

d=receipt['dspark'];old=prior['dspark'];m=d['memory'];io=d['serve_stream_counters']['io']
g=receipt['post_prefill_growth']
result={
    'source_commit':bound['source_commit'],
    'receipt_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    'input_tokens':receipt['prompt_tokens'],
    'output_tokens':d['n_generated'],
    'timed_decode_steps':receipt['decode_tokens'],
    'decode_tok_s':d['decode_tok_s'],
    'decode_wall_s':d['decode_wall_s'],
    'prior_best_decode_tok_s':old['decode_tok_s'],
    'prior_best_decode_wall_s':old['decode_wall_s'],
    'tps_change_percent':100*(d['decode_tok_s']/old['decode_tok_s']-1),
    'wall_reduction_s':old['decode_wall_s']-d['decode_wall_s'],
    'cycles':d['cycles'],
    'all1024_native_ids_identical':len(d['token_ids'])==1024 and d['token_ids']==old['token_ids'],
    'output_sha256':d['token_ids_sha256'],
    'ar_divergence':d['divergence'],
    'prefill_slots_per_layer':g['prefill_slots_per_layer'],
    'decode_slots_per_layer':g['decode_slots_per_layer'],
    'decode_miss_records_per_part':bound['decode_miss_records_per_part'],
    'baseline_bytes':bound['baseline_bytes'],
    'physical_bound_bytes':bound['physical_bound_bytes'],
    'mlx_active_bound_bytes':bound['active_bound_bytes'],
    'mlx_peak_bytes':m['mlx_peak_bytes'],
    'process_footprint_peak_bytes':m['process_footprint_peak_bytes'],
    'system_used_peak_bytes':m['system_used_peak_bytes'],
    'mlx_active_end_bytes':m['mlx_active_bytes_at_decode_end'],
    'mlx_cache_end_bytes':m['mlx_cache_bytes_at_decode_end'],
    'read_bytes':io['read_bytes'],
    'read_records':io['records_read'],
    'read_union_s':io['read_wall_ns']/1e9,
    'growth_seconds':g['growth_seconds'],
    'phase_time_s':d['phase_time_s'],
    'guard_child_exit_code':capture(r'GPU step exited with code (\d+)'),
    'guard_process_tree_peak_bytes':capture(r'sampled peak step footprint .*?\((\d+) bytes\)'),
    'guard_physical_peak_bytes':capture(r'sampled peak physical used .*?\((\d+) bytes\)'),
    'guard_samples':capture(r'complete step memory samples (\d+)'),
    'compressor_growth_bytes':capture(r'sampled peak compressor delta .*?\((\d+) bytes\)'),
    'performance_scope':'One complete candidate versus retained part3 result, same84-to104 capacity and native output; different baseline and time window. Bounded one-layer interleaved comparisons exist; no full-model repeatability claim.',
    'goal_20_tps_reached':d['decode_tok_s']>=20,
}
assert result['input_tokens']==16384 and result['output_tokens']==1024
assert result['decode_miss_records_per_part']==1
assert result['all1024_native_ids_identical']
assert max(result['system_used_peak_bytes'],result['guard_physical_peak_bytes'])<=result['physical_bound_bytes']<=110000000000
(ROOT/'full-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
