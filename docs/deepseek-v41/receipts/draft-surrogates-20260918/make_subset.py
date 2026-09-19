"""Stream a true draft-bank subset using bounded, uncached CPU I/O only."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import time

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/tmp/dsv41-compact-residents')
DEST = ROOT / 'artifact'
prior = json.loads((SOURCE / 'receipt.json').read_text())
screen = json.loads((ROOT / 'v2/head-screen.json').read_text())
if not screen['complete'] or screen['physically_compacted']:
    raise RuntimeError('completed draft quality screen required')
if [(a['cycles'], a['verify_rows']) for a in screen['arms'][:2]] != [(198, 1242), (198, 1242)]:
    raise RuntimeError('small candidate failed the predefined acceptance gate')
chosen = screen['draft_pruning']['one_band']
selected = chosen['selected_experts_by_stage']
if tuple(map(len, selected)) != (80, 40, 24):
    raise RuntimeError('candidate inventory differs')
if shutil.disk_usage(ROOT).free < 4 * 1024**3:
    raise RuntimeError('insufficient space for the 2.71 GB subset and margin')
DEST.mkdir()
receipt = {k:v for k,v in prior.items() if k not in ('files','selected_experts_by_stage')}
receipt.update(selected_experts_by_stage=selected, files=[],
    source_receipt_sha256=hashlib.sha256((SOURCE/'receipt.json').read_bytes()).hexdigest(),
    head_screen_sha256=hashlib.sha256((ROOT/'v2/head-screen.json').read_bytes()).hexdigest(),
    cpu_working_buffer_bound_bytes=16*1024**2, nocache=True,
    removed_payload_bytes=chosen['projected_retired_payload_bytes'])
pattern = re.compile(r'mtp\.(\d+)\.ffn\.experts\.(\d+)\.')
for record in prior['files']:
    stage = record['stage']
    path = SOURCE / f'mtp-selected-stage{stage}.safetensors'
    output = DEST / path.name
    started = time.perf_counter()
    with path.open('rb', buffering=0) as src:
        fcntl.fcntl(src.fileno(), 48, 1)  # Darwin F_NOCACHE
        header_len = struct.unpack('<Q', src.read(8))[0]
        if header_len > 1024**2 or path.stat().st_size != record['file_bytes']:
            raise RuntimeError('source file inventory changed')
        raw_header = src.read(header_len)
        header = json.loads(raw_header)
        ordered = sorted(((n,d) for n,d in header.items() if n != '__metadata__'),
                         key=lambda pair:pair[1]['data_offsets'][0])
        position = 0
        kept = []
        new_header = {}
        new_position = 0
        for name, info in ordered:
            start, end = info['data_offsets']
            match = pattern.match(name)
            if start != position or end <= start or match is None or int(match[1]) != stage:
                raise RuntimeError('source tensor layout differs')
            position = end
            retain = int(match[2]) in selected[stage]
            kept.append((name, end-start, retain))
            if retain:
                new_header[name] = dict(info, data_offsets=[new_position,new_position+end-start])
                new_position += end-start
        if position != record['payload_bytes'] or 8+header_len+position != record['file_bytes']:
            raise RuntimeError('source payload has unaccounted bytes')
        if len(new_header) != 6*len(selected[stage]) or new_position != len(selected[stage])*18_800_640:
            raise RuntimeError('subset tensor geometry differs')
        encoded = json.dumps(new_header, separators=(',', ':')).encode()
        encoded += b' ' * (-len(encoded) % 8)
        prefix = struct.pack('<Q',len(encoded)) + encoded
        old_hash, payload_hash, file_hash = (hashlib.sha256() for _ in range(3))
        with output.with_suffix('.partial').open('xb', buffering=0) as dst:
            fcntl.fcntl(dst.fileno(), 48, 1)
            def write_all(data):
                view = memoryview(data)
                while view:
                    count = dst.write(view)
                    if not count: raise RuntimeError('short output write')
                    view = view[count:]
            write_all(prefix)
            file_hash.update(prefix)
            for name, count, retain in kept:
                while count:
                    data = src.read(min(count, 1024**2))
                    if not data: raise RuntimeError('short source read')
                    old_hash.update(data)
                    if retain:
                        write_all(data)
                        payload_hash.update(data)
                        file_hash.update(data)
                    count -= len(data)
            if old_hash.hexdigest() != record['payload_sha256'] or src.read(1):
                raise RuntimeError('authenticated source payload differs')
            os.fsync(dst.fileno())
        output.with_suffix('.partial').replace(output)
    receipt['files'].append(dict(stage=stage,path=str(output),source_shard=record['source_shard'],
        tensor_count=len(new_header),payload_bytes=new_position,file_bytes=output.stat().st_size,
        payload_sha256=payload_hash.hexdigest(),file_sha256=file_hash.hexdigest(),
        source_payload_sha256=old_hash.hexdigest(),source_header_sha256=hashlib.sha256(raw_header).hexdigest(),
        elapsed_s=time.perf_counter()-started))
receipt['complete'] = True
(DEST/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'complete':True,'payload_bytes':sum(f['payload_bytes'] for f in receipt['files']),
                  'removed_payload_bytes':receipt['removed_payload_bytes'],'root':str(DEST)}))
