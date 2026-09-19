"""Authenticate the existing draft artifact with bounded uncached reads."""
import fcntl
import hashlib
import json
import os
from pathlib import Path


def verify(proof, output):
    if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
        raise RuntimeError('parent GPU guard required before artifact I/O')
    p = Path(proof['artifact_receipt_path'])
    blob = p.read_bytes()
    if hashlib.sha256(blob).hexdigest() != proof['artifact_receipt_sha256']:
        raise RuntimeError('draft artifact receipt changed')
    receipt = json.loads(blob)
    verified = []
    for row in receipt['files']:
        path = Path(row['path'])
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            fcntl.fcntl(fd, 45, 0)
            before = os.fstat(fd)
            if before.st_size != row['file_bytes']:
                raise RuntimeError('draft file size changed')
            digest = hashlib.sha256()
            while chunk := os.read(fd, 8 * 1024**2):
                digest.update(chunk)
            after = os.fstat(fd)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise RuntimeError('draft file changed during validation')
            if digest.hexdigest() != row['file_sha256']:
                raise RuntimeError('draft file digest changed')
            verified.append({'path': str(path), 'file_bytes': before.st_size,
                             'sha256': digest.hexdigest()})
        finally:
            os.close(fd)
    report = {'complete': True, 'files': verified, 'cpu_buffer_bound_bytes': 16 * 1024**2,
              'nocache': True, 'phase': 'guarded construction before MLX import'}
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    print('DRAFT_ARTIFACT_VERIFIED', json.dumps({'files': len(verified),
        'bytes': sum(r['file_bytes'] for r in verified)}), flush=True)
