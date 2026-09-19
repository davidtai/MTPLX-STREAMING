"""Archive the completed component/full result without changing measured inputs."""
from pathlib import Path
import hashlib
import json
import re
import shutil

scratch = Path(__file__).resolve().parent
repo = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
out = repo / 'docs/deepseek-v41/receipts/extension-bank-20260919'
out.mkdir(parents=True, exist_ok=True)
prefix = Path('/tmp/dsv41-110-stage/full-extension-bank-20260919-v1')
read = lambda path: json.loads(path.read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
result = read(Path(str(prefix) + '.jsonl'))
bound = read(Path(str(prefix) + '.bounds.json'))
d = result['dspark']
memory = d['memory']
component = read(scratch / 'probe.json')
previous = read(Path('/tmp/dsv41-predictable-expansion-20260919/full-summary.json'))
best = read(Path('/tmp/dsv41-110-stage/full-hybrid-lookup-20260918-v1.jsonl'))['dspark']
assert result['context_tokens'] == 16384 and d['n_generated'] == 1024
assert d['token_ids'] == best['token_ids']
assert d['token_ids_sha256'] == '0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac'
assert all(a['all_outputs_exact'] and len(a['output_digests']) == 206 for a in component['arms'])
assert component['arms'][1]['existing_backings_unchanged']
assert result['projection_ownership']['module_ownership_verified']
assert result['post_prefill_growth']['overflow']['existing_row_owners_unchanged']
assert component['active_after_close_bytes'] == 8

def lifecycle(path):
    lines = path.read_text().splitlines()
    return {
        'restore': next(s for s in reversed(lines) if 'healthy, model identity' in s),
        'release': next(s for s in reversed(lines) if 'released exclusive GPU lock' in s),
        'step_exit': next(s for s in reversed(lines) if 'GPU step exited with code' in s),
    }

life = lifecycle(Path(str(prefix) + '.guard.log'))
guard_peak = int(re.search(r'peak physical used .*?\((\d+) bytes\)', life['step_exit'])[1])
assert 'code 0' in life['step_exit']
assert max(guard_peak, memory['system_used_peak_bytes']) < 110000000000
io = d['serve_stream_counters']['io']
growth = result['post_prefill_growth']
full = {
    'source_commit': bound['source_commit'],
    'decode_tps': d['decode_tok_s'], 'decode_wall_s': d['decode_wall_s'],
    'input_tokens': 16384, 'output_tokens': 1024, 'timed_decode_steps': 1023,
    'all_native_control_ids_exact': True, 'output_ids_sha256': d['token_ids_sha256'],
    'target_calls': d['verify_calls'], 'prefill_capacity': 84,
    'initial_decode_capacity': 84, 'final_decode_capacity': 111,
    'baseline_bytes': bound['baseline_bytes'],
    'admitted_physical_bound_bytes': bound['physical_bound_bytes'],
    'guard_system_peak_bytes': guard_peak,
    'internal_system_peak_bytes': memory['system_used_peak_bytes'],
    'process_footprint_peak_bytes': memory['process_footprint_peak_bytes'],
    'mlx_peak_bytes': memory['mlx_peak_bytes'],
    'internal_peak_minus_launch_estimate_bytes': memory['system_used_peak_bytes'] - bound['physical_bound_bytes'],
    'remaining_sampled_machine_headroom_bytes': 110000000000 - memory['system_used_peak_bytes'],
    'memory_scope': 'Sampled whole-machine physical use includes file cache. Launch estimate was exceeded by 107409300 B; physical ceiling was not. Background and allocator/footprint overlap prevent assigning the entire difference to one application. Samples do not establish an unsampled hard peak.',
    'historical_best_tps': best['decode_tok_s'],
    'historical_best_wall_s': best['decode_wall_s'],
    'wall_saved_vs_historical_best_s': best['decode_wall_s'] - d['decode_wall_s'],
    'tps_ratio_vs_historical_best': d['decode_tok_s'] / best['decode_tok_s'],
    'growth_saved_vs_previous_projection_s': previous['growth']['growth_seconds'] - growth['growth_seconds'],
    'expert_records_saved_vs_historical_best': best['serve_stream_counters']['io']['records_read'] - io['records_read'],
    'expert_bytes_saved_vs_historical_best': best['serve_stream_counters']['io']['read_bytes'] - io['read_bytes'],
    'io': io, 'phase_time_s': d['phase_time_s'],
    'growth': growth, 'projection': result['projection_ownership'],
    'output_sanity': 'Identical coherent Python unified-diff opening adding generation rate and percentile/summary helpers. Truncated at 1024 tokens; not a complete validated patch.',
    'comparison_scope': 'New best single complete result. Same native output and 198 calls; historical comparison has different background and 110 versus 111 slots. No repeated isolated full speedup is established.',
    'lifecycle': life, 'independent_health': read(scratch / 'full-v1/post-run-health.json'),
    'goal_tps': 20, 'goal_achieved': False,
    'seconds_still_to_save_for_20tps': d['decode_wall_s'] - 1023 / 20,
}
(scratch / 'full-summary.json').write_text(json.dumps(full, indent=2) + '\n')
summary = {
    'source_commit': bound['source_commit'], 'q4_only': True,
    'physical_ceiling_bytes': 110000000000, 'wired_ceiling_bytes': 100 * 1024**3,
    'current_best_complete_tps': d['decode_tok_s'], 'target_tps': 20,
    'component': {
        **{k: component[k] for k in ('complete', 'control_median_ns', 'candidate_median_ns', 'candidate_over_control', 'control_spread_fraction', 'active_after_close_bytes')},
        'arms': [{k: a[k] for k in ('mode', 'capacity', 'allocation_ns', 'existing_backings_unchanged', 'wall_ns', 'warm_wall_ns', 'charged_wall_ns', 'charged_warm_wall_ns', 'expert_records', 'expert_read_bytes', 'dense_read_bytes', 'mlx_peak_bytes', 'all_outputs_exact', 'active_after_eval_bytes')} for a in component['arms']],
        'lifecycle': lifecycle(scratch / 'guard.log'),
        'scope': 'Equal110 capacity, all40 packed projections, 206 saved M6 routes from layer34, synthetic inputs, no attention. Allocation charged. MLX peak is cumulative for the process, not per arm.',
    },
    'full': full,
    'regression': read(scratch / 'regression-results.json'),
    'raw_files': {str(p): sha(p) for p in sorted(prefix.parent.glob(prefix.name + '*')) if p.is_file()},
    'next_gate': 'Any full follow-up needs fresh admission that accounts for observed background variation. Do not repeat rejected GU-read coalescing. Production defaults remain unchanged.',
}

for path in sorted(scratch.iterdir()):
    if path.is_file() and not path.is_symlink() and path.name not in ('curate.py', 'full-summary.json', 'stage_full.py'):
        dest = out / 'component' / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
for path in sorted((scratch / 'full-v1').rglob('*')):
    rel = path.relative_to(scratch / 'full-v1')
    if path.is_file() and not path.is_symlink() and not {'__pycache__', 'artifact'} & set(rel.parts):
        dest = out / 'full/sources' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
artifact_manifest = out / 'full/sources/packed/artifact/manifest.json'
artifact_manifest.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(scratch / 'full-v1/packed/artifact/manifest.json', artifact_manifest)
for suffix, name in [('.jsonl', 'result.json'), ('.bounds.json', 'bounds.json'), ('.passes.jsonl', 'passes.ndjson'), ('.guard.log', 'guard.log'), ('.reclamation.json', 'reclamation.json'), ('.target-plan.json', 'target-plan.json')]:
    shutil.copy2(Path(str(prefix) + suffix), out / 'full' / name)
shutil.copy2(scratch / 'stage_full.py', out / 'full/stage_full.py')
shutil.copy2(scratch / 'curate.py', out / 'curate.py')
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: full[k] for k in ('decode_tps', 'decode_wall_s', 'wall_saved_vs_historical_best_s', 'growth_saved_vs_previous_projection_s', 'expert_records_saved_vs_historical_best', 'internal_system_peak_bytes', 'internal_peak_minus_launch_estimate_bytes', 'seconds_still_to_save_for_20tps')}))
