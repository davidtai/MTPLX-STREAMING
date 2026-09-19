"""Build CPU-only, identity-pinned admission for the input-row screen."""
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import struct
import subprocess

root = Path(__file__).resolve().parent
repo = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
model = Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
base = Path('/tmp/dsv41-row-pair-native109-20260918')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
config = json.loads((model / 'config.json').read_text())['text_config']
index_path = model / 'model.safetensors.index.json'
index = json.loads(index_path.read_text())
tensor_name = 'embed.weight'
path = model / index['weight_map'][tensor_name]
fd = os.open(path, os.O_RDONLY)
try:
    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
    size = struct.unpack('<Q', os.pread(fd, 8, 0))[0]
    header_bytes = os.pread(fd, size, 8)
    tensor = json.loads(header_bytes)[tensor_name]
    st = os.fstat(fd)
finally:
    os.close(fd)
assert tensor['shape'] == [129280, 5120] and tensor['dtype'] == 'BF16'
assert config['tie_word_embeddings'] is False
source = {'path': str(path), 'tensor': tensor_name, 'shape': tensor['shape'],
          'tensor_header': tensor, 'header_sha256': hashlib.sha256(header_bytes).hexdigest(),
          'offset': 8 + size + tensor['data_offsets'][0],
          'nbytes': tensor['data_offsets'][1] - tensor['data_offsets'][0],
          'identity': {k: getattr(st, k) for k in ('st_dev','st_ino','st_size','st_mtime_ns')}}
shutil.copyfile(base / 'library_identity.py', root / 'library_identity.py')
screen = (base / 'run_screen.py').read_text()
start = screen.index("inventory = json.loads((ROOT / 'artifact/manifest.json').read_text())")
end = screen.index('\ndef now():', start)
screen = screen[:start] + "files = [Path(installation['embedding']['path'])]\n" + screen[end:]
screen = screen.replace('PREFIX_MLP_SOURCE_CACHE_RECLAIMED', 'EMBEDDING_SOURCE_CACHE_RECLAIMED')
(root / 'run_screen.py').write_text(screen)
proof = {'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
         'scope': 'Only the native 1.324GB BF16 input table. Exact file-row cache,16MiB arena/32MiB host bound; native M1/M5/M6 outputs, retirement and eviction ownership. No full model.',
         'static_incremental_bound_bytes': 8 * 1024**3,
         'bound_components': {'metal_cache_compiler_envelope_bytes': 4 * 1024**3,
                              'host_source_table_io_compiler_envelope_bytes': 4 * 1024**3,
                              'native_embedding_bytes': source['nbytes'],
                              'host_arena_bytes': 16 * 1024**2,
                              'host_embedding_total_bound_bytes': 32 * 1024**2,
                              'source_read_chunk_bytes': 32 * 1024**2,
                              'max_output_bytes': 8 * 5120 * 2},
         'embedding': source, 'draft_block_size': config['dspark_block_size'],
         'noise_token_id': config['dspark_noise_token_id'],
         'model_path': str(model), 'config_sha256': sha(model / 'config.json'),
         'model_index_sha256': sha(index_path),
         'strict_allocator': json.loads((base / 'installation.json').read_text())['strict_allocator']}
py = [p for p in root.glob('*.py')]
for p in py:
    ast.parse(p.read_text())
proof['helper_sha256'] = {p.name: sha(p) for p in py}
runtime = [repo / 'mtplx/deepseek_v41_memory_profile.py',
           repo / 'mtplx/models/deepseek_v41.py', repo / 'mtplx/models/deepseek_v41_dspark.py',
           repo / 'scripts/deepseek_v41/reclaim_file_cache.py',
           Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/lib/python3.12/site-packages/mlx/nn/layers/embedding.py')]
proof['runtime_source_sha256'] = {str(p):sha(p) for p in runtime}
(root / 'installation.json').write_text(json.dumps(proof, indent=2) + '\n')
command = shlex.join(['env', 'GPU_WINDOW_CHILD_RSS_CAP_BYTES=8589934592',
    'GPU_WINDOW_LOCK_TIMEOUT=120', 'GPU_WINDOW_TOTAL_MEM_CEILING_BYTES=110000000000',
    'GPU_WINDOW_MIN_AVAIL_GB=10', 'GPU_WINDOW_RESTORE_QWEN_ALWAYS=1',
    'PYTHONHASHSEED=0', 'PYTHONUNBUFFERED=1', f'PYTHONPATH={repo}:{root}',
    'scripts/deepseek_v41/gpu_window.sh',
    '/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python',
    str(root / 'run_screen.py')]) + ' > ' + shlex.quote(str(root / 'guard.log')) + ' 2>&1\n'
(root / 'command.sh').write_text(command)
print(json.dumps({'root': str(root), 'embedding_bytes': source['nbytes'],
                  'complete_bound_bytes': proof['static_incremental_bound_bytes'],
                  'shapes': [1, config['dspark_block_size'], 6], 'source': proof['source_commit']}))
