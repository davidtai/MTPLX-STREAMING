"""Bounded, CPU-only export of every exact E8M0 row; no model execution."""
import fcntl
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time


class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'mlx_lm'}:
            raise RuntimeError('scale export must not import MLX')


sys.meta_path.insert(0, NoMLX())
if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent memory/service guard required')
signal.alarm(600)
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot

before = host_memory_snapshot()
HOST_BOUND = 2 * 1024**3
MACHINE_INCREMENT_BOUND = 20 * 1024**3
if not before['box']['ok'] or before['box']['used_bytes'] + MACHINE_INCREMENT_BOUND > 109500000000:
    raise RuntimeError('insufficient headroom for CPU buffers and worst-case output file cache')
import numpy as np
from codec import pack_rows, unpack_rows

root = Path(__file__).resolve().parent
out = root / 'artifact'
if out.exists():
    raise RuntimeError('refusing to overwrite a partial or complete artifact')
model = Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
blob = (model / 'expert-manifest.json').read_bytes()
manifest_hash = hashlib.sha256(blob).hexdigest()
if manifest_hash != '44c340845990b4ec21342d914204d69eb9e7cf371d6b18944f8dfd2e945df3e9':
    raise RuntimeError('native manifest changed')
manifest = json.loads(blob)
del blob
records = {(r['layer'], r['expert']): r for r in manifest['records']}
if len(records) != 40 * 384:
    raise RuntimeError('unexpected expert inventory')
cursor = 0
for layer in range(40):
    for expert in range(384):
        r = records[layer, expert]
        if r['sidecar_offset'] != cursor or r['sidecar_length'] != 18800640 or not r['sha256']:
            raise RuntimeError('source records do not cover the exact contiguous sidecar')
        cursor += r['sidecar_length']
if cursor != manifest['sidecar']['size'] or cursor != 288777830400:
    raise RuntimeError('native source size changed')
RAW_SCALE_BYTES = 40 * 384 * 3 * 368640
MAX_PACKED_BYTES = RAW_SCALE_BYTES + 40 * 384 * (9728 * 4 + 3 * 4)
if MAX_PACKED_BYTES + HOST_BOUND > MACHINE_INCREMENT_BOUND:
    raise RuntimeError('static export allocation bound is inconsistent')
disk_before = shutil.disk_usage(root)
if disk_before.free < MAX_PACKED_BYTES + 8 * 1024**3:
    raise RuntimeError('insufficient disk for worst-case packed scales and reserve')
out.mkdir()
source = model / 'experts.bin'


def identity(st):
    return {k: getattr(st, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns')}


class Output:
    def __init__(self, name):
        self.name = name
        self.fd = os.open(out / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        fcntl.fcntl(self.fd, fcntl.F_NOCACHE, 1)
        self.hash = hashlib.sha256()
        self.size = 0

    def write(self, array):
        data = array.tobytes(order='C')
        view = memoryview(data)
        while view:
            n = os.write(self.fd, view)
            if n <= 0:
                raise RuntimeError('short artifact write')
            view = view[n:]
        self.hash.update(data)
        self.size += len(data)

    def close(self):
        os.fsync(self.fd)
        os.close(self.fd)
        return {'file': self.name, 'bytes': self.size, 'sha256': self.hash.hexdigest(), 'dtype': 'uint32_le'}


report = {
    'format': 'deepseek-v41-exact-row-scales-v1',
    'scope': 'complete target-scale inventory, CPU-only; no inference timing',
    'source_commit': os.environ['DSV41_SOURCE_COMMIT'],
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'codec_sha256': hashlib.sha256((root / 'codec.py').read_bytes()).hexdigest(),
    'source_manifest_sha256': manifest_hash,
    'source_path': str(source),
    'source_bank_sha256_from_manifest': manifest['sidecar']['sha256'],
    'source_bank_sha256_recomputed': False,
    'source_verification': 'all record SHA256 values verified; records cover every source byte exactly once',
    'static_host_bound_bytes': HOST_BOUND,
    'static_machine_increment_bound_bytes': MACHINE_INCREMENT_BOUND,
    'worst_case_packed_bytes': MAX_PACKED_BYTES,
    'raw_scale_bytes': RAW_SCALE_BYTES,
    'memory_before': before,
    'disk_free_before_bytes': disk_before.free,
    'layers': [], 'read_bytes': 0, 'verified_records': 0,
}
fd = os.open(source, os.O_RDONLY)
fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
report['source_identity'] = identity(os.fstat(fd))
start = time.perf_counter()
try:
    for layer in range(40):
        writers = {}
        for projection, rows, columns in (('gate_proj', 2304, 160), ('up_proj', 2304, 160), ('down_proj', 5120, 72)):
            prefix = f'layer{layer:02d}-{projection}'
            writers[projection] = {
                'rows': rows, 'columns': columns,
                'descriptors': Output(prefix + '-descriptors.bin'),
                'payload': Output(prefix + '-payload.bin'),
                'bases': Output(prefix + '-bases.bin'),
                'expert_bases': np.zeros(384, np.uint32),
                'width_counts': {str(w): 0 for w in (0, 1, 2, 4, 8)},
            }
        for expert in range(384):
            record = records[layer, expert]
            data = os.pread(fd, record['sidecar_length'], record['sidecar_offset'])
            if len(data) != 18800640 or hashlib.sha256(data).hexdigest() != record['sha256']:
                raise RuntimeError(f'source integrity failure: {layer}/{expert}')
            report['read_bytes'] += len(data)
            report['verified_records'] += 1
            offset = 0
            for segment in record['segments']:
                if segment['offset'] != record['sidecar_offset'] + offset or segment['shard'] != 'experts.bin':
                    raise RuntimeError('source segment geometry changed')
                if segment['component'].endswith('.scales'):
                    projection = segment['component'].split('.')[0]
                    w = writers[projection]
                    scales = np.frombuffer(data, np.uint8, count=segment['length'], offset=offset).reshape(w['rows'], w['columns'])
                    descriptors, payload = pack_rows(scales)
                    if not np.array_equal(scales, unpack_rows(descriptors, payload, w['columns'])):
                        raise RuntimeError(f'lossy scale encoding: {layer}/{expert}/{projection}')
                    base = w['payload'].size // 4
                    if base + payload.size >= 2**32:
                        raise RuntimeError('layer payload exceeds base index range')
                    w['expert_bases'][expert] = base
                    w['descriptors'].write(descriptors)
                    w['payload'].write(payload)
                    widths = (descriptors >> 8) & 15
                    for width in (0, 1, 2, 4, 8):
                        w['width_counts'][str(width)] += int(np.count_nonzero(widths == width))
                offset += segment['length']
            if offset != len(data):
                raise RuntimeError('source record geometry changed')
        entry = {'layer': layer, 'components': {}}
        for projection, w in writers.items():
            w['bases'].write(w['expert_bases'])
            c = {k: w[k].close() for k in ('descriptors', 'payload', 'bases')}
            c['descriptors']['shape'] = [384, w['rows']]
            c['payload']['shape'] = [c['payload']['bytes'] // 4]
            c['bases']['shape'] = [384]
            c['scale_shape_per_expert'] = [w['rows'], w['columns']]
            c['width_row_counts'] = w['width_counts']
            c['packed_bytes'] = sum(c[k]['bytes'] for k in ('descriptors', 'payload', 'bases'))
            entry['components'][projection] = c
        entry['packed_bytes'] = sum(c['packed_bytes'] for c in entry['components'].values())
        report['layers'].append(entry)
        print('EXPORTED_LAYER', layer, 'packed_bytes', entry['packed_bytes'], 'elapsed_s', round(time.perf_counter() - start, 3), flush=True)
    if identity(os.fstat(fd)) != report['source_identity'] or identity(source.stat()) != report['source_identity']:
        raise RuntimeError('source changed during complete inventory')
finally:
    os.close(fd)
report['packed_bytes'] = sum(layer['packed_bytes'] for layer in report['layers'])
if report['packed_bytes'] > MAX_PACKED_BYTES or report['read_bytes'] != cursor or report['verified_records'] != 15360:
    raise RuntimeError('inventory totals inconsistent')
report['elapsed_s'] = time.perf_counter() - start
report['memory_after'] = host_memory_snapshot()
report['disk_free_after_bytes'] = shutil.disk_usage(root).free
report['complete'] = True
(out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
print('EXPORT_COMPLETE', json.dumps({k: report[k] for k in ('packed_bytes', 'raw_scale_bytes', 'read_bytes', 'verified_records', 'elapsed_s')}), flush=True)
