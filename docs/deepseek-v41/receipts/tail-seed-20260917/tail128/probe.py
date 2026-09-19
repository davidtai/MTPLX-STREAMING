"""Bounded native-window seeding from retained rows; target trunk never executes."""
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import time
from dataclasses import replace

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('parent GPU guard required before MLX import')
signal.alarm(180)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'probe.json'
if OUT.exists():
    raise RuntimeError('refusing to overwrite evidence')
proof = json.loads((ROOT / 'installation.json').read_text())
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
before = host_memory_snapshot()
bound = proof['incremental_bound_bytes']
if (not before['box']['ok'] or before['box']['used_bytes'] + bound > 110000000000
    or before['box']['wired_bytes'] + bound > 100 * 1024**3):
    raise RuntimeError('seed operator cannot fit the current baseline')
MODEL = Path(proof['model_path'])
if hashlib.sha256((MODEL / 'expert-manifest.json').read_bytes()).hexdigest() != proof['model_manifest_sha256']:
    raise RuntimeError('native manifest identity changed')
TEACHER = Path('/tmp/dsv41-depth-replay-20260917/teacher.json')
teacher = json.loads(TEACHER.read_text())
if hashlib.sha256(TEACHER.read_bytes()).hexdigest() != proof['teacher_sha256']:
    raise RuntimeError('native target-state provenance changed')
info = teacher['files']['committed-hidden.npy']
path = TEACHER.parent / 'committed-hidden.npy'
if path.stat().st_size != info['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
    raise RuntimeError('captured hidden-state bytes changed')

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mtplx.expert_manifest import load_expert_manifest
from mtplx.models import deepseek_v41 as dv
from mtplx.models import deepseek_v41_dspark as ds
from seed_reader import load_seed_tensors
from mtplx.models.deepseek_v41_dspark_decode import _seed_prefill_state

mx.set_memory_limit(4 * 1024**3)
mx.set_cache_limit(256 * 1024**2)
args = dv.ModelArgs.from_dict(json.loads((MODEL / 'config.json').read_text()))
if (args.hidden_size, args.head_dim, args.sliding_window, ds.n_mtp_layers(args), ds.dspark_target_layer_ids(args)) != (5120, 512, 128, 3, (37, 38, 39)):
    raise RuntimeError('native seed geometry changed')


class SeedAttention(nn.Module):
    # Native seed_only execution reads only these four attributes. No unused
    # query/output/expert parameters or draft heads are constructed or loaded.
    __call__ = ds.DSparkAttention.__call__

    def __init__(self):
        super().__init__()
        self.eps = args.rms_norm_eps
        self.inv_freq = np.asarray(ds._swa_inv_freq(args), dtype=np.float32)
        self.wkv = nn.Linear(5120, 512, bias=False)
        self.kv_norm_weight = mx.ones((512,))


class SeedStage(nn.Module):
    main_project = ds.DSparkBlock.main_project

    def __init__(self, stage):
        super().__init__()
        self.attn = SeedAttention()
        if stage == 0:
            self.main_proj = nn.Linear(15360, 5120, bias=False)
            self.main_norm_weight = mx.ones((5120,))
            self.norm_eps = args.rms_norm_eps


class SeedHead(nn.Module):
    seed_main = ds.DSparkHead.seed_main

    def __init__(self):
        super().__init__()
        self.layers = [SeedStage(i) for i in range(3)]


def run():
    model = nn.Module()
    model.mtp = SeedHead()
    nn.quantize(model, group_size=32, bits=8, mode='mxfp8')
    wanted = set(proof['resident_names'])
    manifest = load_expert_manifest(MODEL / 'expert-manifest.json')
    kept = tuple(t for t in manifest.resident_tensors if t.tensor in wanted)
    if len(kept) != len(wanted) or sum(t.length for t in kept) != proof['resident_payload_bytes']:
        raise RuntimeError('native seed resident inventory changed')
    raw = load_seed_tensors(MODEL, manifest, kept, mx=mx)
    weights = dv._map_mtp_residents(raw)
    current = dict(tree_flatten(model.parameters()))
    if set(current) != set(weights):
        raise RuntimeError('seed parameter coverage differs')
    del current
    model.load_weights(list(weights.items()), strict=True)
    if any(v is not weights[n] for n, v in tree_flatten(model.parameters())):
        raise RuntimeError('unreplaced seed parameters remain')
    mx.eval(model.parameters())
    del raw, weights, manifest
    data = np.load(path, allow_pickle=False)
    if list(data.shape) != info['shape'] or str(data.dtype) != info['dtype']:
        raise RuntimeError('hidden-state storage shape changed')
    captured = mx.array(data).view(mx.bfloat16)
    mx.eval(captured)
    del data
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    report['resident_active_bytes'] = int(mx.get_active_memory())
    reference = None
    for arm in ('full', 'tail', 'full', 'tail'):
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        first = 0 if arm == 'full' else 16384 - 128
        # Tile authentic committed hidden rows into a synthetic 16K prompt.
        # Both arms see identical final rows and absolute positions; this is
        # neither the real prompt seed nor target generation.
        hidden = mx.take(captured, mx.arange(first, 16384) % captured.shape[0], axis=0)[None, :, :]
        mx.eval(hidden)
        caches = [ds.DSparkStageCache(128, 512) for _ in range(3)]
        for cache in caches:
            cache.offset = first
        started = time.perf_counter()
        main_h = _seed_prefill_state(model, hidden, caches)
        elapsed = time.perf_counter() - started
        bits = [np.asarray(main_h.view(mx.uint16))] + [np.asarray(c.window.view(mx.uint16)) for c in caches]
        if reference is None:
            reference = [a.copy() for a in bits]
        equal = [bool(np.array_equal(a, b)) for a, b in zip(reference, bits)]
        differences = [int(np.count_nonzero(a != b)) for a, b in zip(reference, bits)]
        row = {'arm': arm, 'seed_seconds': elapsed, 'peak_bytes': int(mx.get_peak_memory()),
               'hidden_rows': int(hidden.shape[1]), 'cache_offsets': [c.offset for c in caches],
               'equal_final_row_and_windows': equal, 'differing_elements': differences,
               'state_sha256': [hashlib.sha256(a.tobytes()).hexdigest() for a in bits]}
        report['arms'].append(row)
        OUT.write_text(json.dumps(report, indent=2) + '\n')
        print('TAIL_SEED_ARM', json.dumps(row), flush=True)
        if row['cache_offsets'] != [16384] * 3 or (arm == 'full' and not all(equal)):
            raise RuntimeError('native control state or offsets differ')
        del caches, cache, main_h, hidden, bits
    report['complete'] = True
    report['tail_exact'] = all(all(r['equal_final_row_and_windows']) for r in report['arms'] if r['arm'] == 'tail')
    report['median_seed_seconds'] = {arm: statistics.median(r['seed_seconds'] for r in report['arms'] if r['arm'] == arm) for arm in ('full', 'tail')}
    report['after'] = host_memory_snapshot()


report = {'scope': 'Native dense seed weights only; synthetic16K sequence from captured native hidden rows. Compares full seeding with only the final128 rows at correct absolute offsets. No target/draft generation or TPS claim.',
          'installation': proof, 'before': before, 'arms': [], 'complete': False}
try:
    run()
finally:
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    report['active_after_close_bytes'] = int(mx.get_active_memory())
    OUT.write_text(json.dumps(report, indent=2) + '\n')
if report['active_after_close_bytes'] > 2 * 1024**2:
    raise RuntimeError('native seed owners remain after close')
print('TAIL_SEED_COMPLETE', report.get('tail_exact'), report['active_after_close_bytes'], flush=True)
