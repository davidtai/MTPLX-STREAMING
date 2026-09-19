"""Summarize completed target evidence without importing MLX."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
path = Path('/tmp/dsv41-110-stage/full-suffix-consensus-20260919-v1.jsonl')
row = json.loads(path.read_text())
old = json.loads(Path('/tmp/dsv41-110-stage/full-hybrid-lookup-20260918-v1.jsonl').read_text())
ds = row['dspark']
control = old['dspark']
admission = row['phase_budget']['admission']
io = ds['serve_stream_counters']['io']
memory = ds['memory']
result = {
    'source_commit': '4b5591e8245addba2c96105cc37c34e8a63233a5',
    'receipt_path': str(path), 'single_run_only': True,
    'prompt_tokens': row['prompt_tokens'], 'generated_tokens': ds['n_generated'],
    'all_native_output_ids_exact': ds['token_ids'] == control['token_ids'],
    'token_ids_sha256': ds['token_ids_sha256'],
    'tps': ds['decode_tok_s'], 'wall_s': ds['decode_wall_s'],
    'old_tps': control['decode_tok_s'], 'old_wall_s': control['decode_wall_s'],
    'wall_saved_s': control['decode_wall_s'] - ds['decode_wall_s'],
    'cycles': ds['cycles'], 'verify_calls': ds['verify_calls'],
    'drafted_by_depth': ds['drafted_by_depth'], 'accepted_by_depth': ds['accepted_by_depth'],
    'baseline_bytes': admission['baseline_bytes'],
    'physical_bound_bytes': admission['physical_bound_bytes'],
    'host_reserve_bytes': admission['host_reserve_bytes'],
    'capacity': admission['decode_slots_per_layer'],
    'memory': {k: v for k, v in memory.items() if k not in ('samples', 'seed_prefill_boundary')},
    'io': {k: io[k] for k in ('records_read', 'read_bytes', 'read_wall_ns', 'read_gb_per_s_window')},
    'phase_time_s': ds['phase_time_s'], 'growth': row['post_prefill_growth'],
    'proposal_installation': row['hybrid_lookup'],
    'divergence': ds.get('divergence'),
    'goal_20tps_achieved': (ds['decode_tok_s'] >= 20 and ds['n_generated'] == 1024
                            and row['prompt_tokens'] == 16384
                            and ds['token_ids'] == control['token_ids']),
    'claim_scope': 'One full candidate; historical control and current live capacity are not an isolated repeated A/B.'}
assert row['prompt_tokens'] == 16384 and ds['n_generated'] == 1024
(root / 'full-summary.json').write_text(json.dumps(result, indent=2) + '\n')
(root / 'full-output.txt').write_text(ds['decoded_text'])
print(json.dumps({k: v for k, v in result.items()
                  if k not in ('memory', 'growth', 'proposal_installation', 'divergence')}, indent=2))
