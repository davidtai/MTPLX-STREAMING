"""One guarded full-workload native/packed comparison; parent never imports MLX."""
import datetime
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('the exclusive GPU guard must own this batch')

class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'mlx_lm'}:
            raise RuntimeError('the batch parent must not initialize MLX')

sys.meta_path.insert(0, NoMLX())
ROOT = Path(__file__).resolve().parent
REPO = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
sys.path.insert(0, str(REPO))
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot

config = json.loads((ROOT / 'pair-config.json').read_text())
proof = json.loads((ROOT / 'construction-proof.json').read_text())
if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() != proof['source_commit']:
    raise RuntimeError('batch source commit changed')
if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO, text=True).strip():
    raise RuntimeError('tracked source is dirty')
if hashlib.sha256((REPO / proof['prefill84_evidence']).read_bytes()).hexdigest() != proof['prefill84_evidence_sha256']:
    raise RuntimeError('complete cap84 prefill evidence changed')
for mode in ('native', 'packed'):
    install = json.loads((ROOT / mode / 'installation.json').read_text())
    if install['source_commit'] != proof['source_commit']:
        raise RuntimeError('installation commit changed')
    for name, digest in install.get('runtime_source_sha256', {}).items():
        if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'runtime source changed: {name}')
    for name, digest in install['helper_sha256'].items():
        if hashlib.sha256((ROOT / mode / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'{mode} helper changed: {name}')

cache_spec = importlib.util.spec_from_file_location('owned_cache', REPO / 'scripts/deepseek_v41/reclaim_file_cache.py')
cache = importlib.util.module_from_spec(cache_spec)
cache_spec.loader.exec_module(cache)
model_root = Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
source = model_root / 'experts.bin'
inventory = json.loads((ROOT / 'packed/artifact/manifest.json').read_text())

def cleanup_owned_cache():
    source_stat = source.stat()
    if {key: getattr(source_stat, key) for key in inventory['source_identity']} != inventory['source_identity']:
        raise RuntimeError('verified expert source identity changed before cleanup')
    rows = cache.reclaim_model(model_root)
    rows += cache.reclaim_receipt_safetensors(Path('/tmp/dsv41-compact-residents'))
    rows.append(cache.reclaim_file(source))
    return dict(files=len(rows), cached_page_bytes_before=sum(x['cached_page_bytes_before'] for x in rows),
                cached_page_bytes_after=sum(x['cached_page_bytes_after'] for x in rows))

results = []
for arm in config['arms']:
    mode = arm['mode']
    output = Path(arm['argv'][arm['argv'].index('--out') + 1])
    if any(output.parent.glob(output.stem + '*')):
        raise RuntimeError(f'refusing to overwrite {output}')
    baseline = host_memory_snapshot()['box']['used_bytes']
    env = os.environ.copy()
    env.update(config['env'])
    env.pop('DSV41_REQUIRE_DECODE_CAP', None)
    env['MTPLX_DSV41_BOX_BASELINE_GB'] = format(baseline / 1e9, '.17g')
    print(json.dumps(dict(event='arm_start', mode=mode, baseline_bytes=baseline,
                         utc=datetime.datetime.now(datetime.timezone.utc).isoformat())), flush=True)
    try:
        completed = subprocess.run(arm['argv'], cwd=REPO, env=env, timeout=900)
    finally:
        cleanup = cleanup_owned_cache()
        print(json.dumps(dict(event='owned_cache_cleanup', mode=mode, **cleanup)), flush=True)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    receipt = json.loads(output.read_text())
    target = receipt['dspark']
    ids = target['token_ids'] if 'token_ids' in target else receipt['dspark_token_ids']
    digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    if len(ids) != 1024 or digest != proof['prefill84_evidence_output_sha256']:
        raise RuntimeError('complete output differs from the retained native stream')
    result = dict(mode=mode, receipt=str(output), receipt_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  output_tokens=len(ids), output_sha256=digest, baseline_bytes=baseline)
    results.append(result)
    (ROOT / 'pair-results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(dict(event='arm_complete', **result)), flush=True)
