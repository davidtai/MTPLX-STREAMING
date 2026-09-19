"""Preserve both component outcomes without copying model payloads."""
import hashlib
import json
from pathlib import Path
import shutil

repo = Path('/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41')
for source, name in [
    (Path('/tmp/dsv41-packed-subset-20260919'), 'packed-draft-subset-20260919'),
    (Path('/tmp/dsv41-indexed-input-20260919'), 'indexed-expert-input-20260919'),
]:
    dest = repo / 'docs/deepseek-v41/receipts' / name
    dest.mkdir(exist_ok=True)
    if (dest / 'archive-sha256.json').exists():
        raise RuntimeError(f'refusing to overwrite an archived receipt: {dest}')
    hashes, links = {}, {}
    for path in sorted(source.rglob('*')):
        rel = path.relative_to(source)
        if '__pycache__' in rel.parts:
            continue
        if path.is_symlink():
            links[str(rel)] = str(path.resolve())
            continue
        if not path.is_file():
            continue
        if path.suffix not in ('.py', '.sh', '.json', '.jsonl', '.log', '.md'):
            raise RuntimeError(f'unexpected evidence type: {path}')
        if path.stat().st_size > 2 * 1024**2:
            raise RuntimeError(f'large payload excluded from receipt: {path}')
        output = dest / rel
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output)
        hashes[str(rel)] = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {'source_root': str(source.resolve()), 'files': hashes,
        'external_symlinks_not_copied': links,
        'scope': 'Original diagnostic bytes preserved; paths in source proofs refer to the measured scratch tree.'}
    (dest / 'archive-sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'receipt': name, 'files': len(hashes), 'external_links': len(links)}))
