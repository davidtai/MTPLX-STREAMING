"""Preserve exact scratch inputs locally and curate sources/receipts for the PR."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

root = Path(__file__).resolve().parent
repo = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
dest = repo / 'docs/deepseek-v41/receipts/ridge-prefetch-20260919'
checkpoint = Path('/Users/davidtai/projects/OpenSourceWTF/checkpoints/deepseek-v41-20260918T161816Z')
archive = checkpoint / 'q4-ridge-prefetch-inputs.tar.gz'
assert not dest.exists() and not archive.exists()
assert json.loads((root / 'summary.json').read_text())['runs'][1]['child']['returncode'] == 0
assert not json.loads((root / 'lifecycle.json').read_text())['owned_processes']
dest.mkdir()
files = [p for p in root.iterdir() if p.is_file() and p.name != 'archive.py']
files += [p for name in ('v1', 'v2') for p in (root / name).iterdir() if p.is_file() and not p.is_symlink()]
files.append(root / 'archive.py')
sources = {}
for path in files:
    sources[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
with tarfile.open(archive, 'w:gz') as target:
    for path in sorted(files):
        target.add(path, arcname=str(path.relative_to(root)), recursive=False)
    for name in ('v1', 'v2'):
        path = root / name / 'artifact/manifest.json'
        target.add(path, arcname=f'{name}/artifact/manifest.json', recursive=False)
record = {'source_root': str(root), 'archive': str(archive), 'bytes': archive.stat().st_size,
          'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'files': sources,
          'external_artifact_root': str((root / 'v1/artifact').resolve()),
          'external_artifact_manifest_sha256': hashlib.sha256((root / 'v1/artifact/manifest.json').read_bytes()).hexdigest()}
with tarfile.open(archive, 'r:gz') as target:
    for name, identity in sources.items():
        content = target.extractfile(name).read()
        assert len(content) == identity['bytes']
        assert hashlib.sha256(content).hexdigest() == identity['sha256']
(archive.with_suffix('.json')).write_text(json.dumps(record, indent=2) + '\n')
omitted = []
for path in files:
    relative = path.relative_to(root)
    if path.name in ('routes.json', 'ridge-parameters.npz'):
        omitted.append({'relative_path': str(relative), **sources[str(relative)]})
        continue
    target = dest / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
(dest / 'input-checkpoint.json').write_text(json.dumps({
    'local_archive': str(archive), 'archive_sha256': record['sha256'],
    'archive_bytes': record['bytes'], 'inputs_preserved_in_archive': omitted,
    'model_artifacts': 'Large Q4 weights and scale artifacts remain at source-pinned original paths.',
    'artifact_root': record['external_artifact_root'],
    'artifact_manifest_sha256': record['external_artifact_manifest_sha256'],
    'replay_note': 'Sources retain original absolute paths and require restaging the exact artifacts; curated receipts alone are not a portable model bundle.'
}, indent=2) + '\n')
print(json.dumps({'archive': str(archive), 'bytes': record['bytes'], 'sha256': record['sha256'], 'verified_files': len(sources), 'curated_files': sum(p.is_file() for p in dest.rglob('*'))}))
