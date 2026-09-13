"""Guarded real-weight draft-only allocation/timing measurement; no target MoE."""
import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import threading
import time

ROOT = Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
OUT = Path('/tmp/dsv41-110-preflight/dspark-head-peak.json')
GIB = 1024**3
signal.alarm(300)
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
    raise RuntimeError('tracked source must be clean')
baseline = float(os.environ['MTPLX_DSV41_BOX_BASELINE_GB']) * 1e9
if not 0 <= baseline <= 20e9:
    raise RuntimeError('draft-only baseline exceeds admitted envelope')

env = {
    'MTPLX_DSV41_ATTN_COMPILE': '1', 'MTPLX_DSV41_ATTN_WO_A_CACHE': '1',
    'MTPLX_DSV41_ATTN_LEAN_CASTS': '1', 'MTPLX_DSV41_ATTN_FUSED_PROJ': '1',
    'MTPLX_DSV41_DRAFT_COMPILE': '1', 'MTPLX_DSV41_DRAFT_HEAD_BF16': '1',
    'MTPLX_DSV41_DECODE_ATTN_KERNEL': '0', 'MTPLX_DSV41_KV_BOUNDED': '0',
}
os.environ.update(env)

from mtplx.expert_manifest import load_expert_manifest
from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
from mtplx.models.deepseek_v41_loader import partition_text_residents

manifest = load_expert_manifest(ROOT / 'expert-manifest.json')
partition = partition_text_residents(manifest, with_mtp=True)
kept = tuple(t for t in partition.kept if t.tensor.startswith('mtp.')
             or t.tensor in ('embed.weight', 'head.weight'))
payload = sum(t.length for t in kept)
mtp_payload = sum(t.length for t in kept if t.tensor.startswith('mtp.'))
largest_shard = max((ROOT / t.shard).stat().st_size for t in kept)
assert payload == 10597621640 and mtp_payload == 7949967240
assert largest_shard < 3 * GIB
partition = dataclasses.replace(partition, kept=kept, kept_bytes=payload, kept_count=len(kept))

# Constructors/quantizers are lazy and replaced by strict checkpoint loading
# before any parameter evaluation. During loading allow all selected tensors,
# all stacked MTP copies, one full shard, and 4 GiB of graph/operator workspace.
# Seeding and draft use <=16K main rows / five draft rows and three stages; the
# extra 4 GiB exceeds the full main hidden + projected hidden + all KV copies.
load_active_bound = payload + mtp_payload + largest_shard + 4 * GIB
assert load_active_bound < 24 * GIB
assert baseline + load_active_bound + 2 * GIB + GIB < 110e9
print('DRAFT_ONLY_BOUND', json.dumps({'load_active_bound_bytes': load_active_bound,
      'baseline_bytes': baseline, 'payload_bytes': payload, 'largest_shard_bytes': largest_shard}), flush=True)

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mtplx.models.deepseek_v41 import (Model, ModelArgs,
    _make_mtp_dense_quant_predicate, _make_mtp_expert_quant_predicate)
from mtplx.models.deepseek_v41_dspark import DSparkHead, DSparkStageCache
from mtplx.models.deepseek_v41_dspark_decode import _seed_prefill_state
from mtplx.models.deepseek_v41_loader import load_text_only_resident_arrays
from mtplx.models import deepseek_v41_dspark as dsp

mx.set_memory_limit(24 * GIB)
mx.set_wired_limit(24 * GIB)
mx.set_cache_limit(GIB)
mx.random.seed(0)
report = {'source_commit': source, 'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scope': 'real MTP head plus shared embedding/head only; no target layers or expert bank',
          'env': env, 'load_active_bound_bytes': load_active_bound, 'phases': [], 'os_samples': []}
stop = threading.Event()
phase = 'construction'
def monitor():
    while not stop.wait(.25):
        report['os_samples'].append({'phase': phase, 'snapshot': host_memory_snapshot()})
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
def snapshot(name, elapsed=None):
    item = {'phase': name, 'elapsed_s': elapsed, 'active_bytes': mx.get_active_memory(),
            'peak_bytes': mx.get_peak_memory(), 'cache_bytes': mx.get_cache_memory()}
    report['phases'].append(item)
    print('DRAFT_PHASE', json.dumps(item), flush=True)

try:
    cfg = json.loads((ROOT / 'config.json').read_text())
    args = ModelArgs.from_dict(cfg)
    assert (args.hidden_size, args.vocab_size, args.n_mtp_layers, args.dspark_block_size) == (5120, 129280, 3, 5)
    probe = nn.Module()
    probe.mtp = DSparkHead(args)
    probe.model = nn.Module()
    probe.model.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
    probe.head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
    nn.quantize(probe.mtp, group_size=32, bits=8, mode='mxfp8',
                class_predicate=_make_mtp_dense_quant_predicate(32))
    nn.quantize(probe.mtp, group_size=32, bits=4, mode='mxfp4',
                class_predicate=_make_mtp_expert_quant_predicate(32))
    snapshot('lazy_construction')
    if mx.get_active_memory() > 512 * 1024**2:
        raise RuntimeError('constructor unexpectedly materialized placeholder weights')
    phase = 'load'
    started = time.perf_counter()
    weights = load_text_only_resident_arrays(ROOT, manifest, mx_module=mx, partition=partition)
    weights = Model.sanitize(probe, weights)
    probe.eval()
    probe.load_weights(list(weights.items()), strict=True)
    del weights
    mx.eval(probe.parameters())
    mx.synchronize()
    gc.collect()
    snapshot('load', time.perf_counter() - started)
    report['parameter_bytes'] = sum(v.nbytes for _, v in tree_flatten(probe.parameters()))
    assert report['parameter_bytes'] == payload
    if mx.get_peak_memory() > load_active_bound:
        raise RuntimeError('observed load peak exceeded the static bound')

    phase = 'prefill_seed'
    mx.clear_cache()
    mx.reset_peak_memory()
    report['seed_baseline_active_bytes'] = mx.get_active_memory()
    started = time.perf_counter()
    hidden = mx.random.normal((1, 16384, 3 * args.hidden_size))
    caches = [DSparkStageCache(args.window_size, args.head_dim) for _ in probe.mtp.layers]
    main_h = _seed_prefill_state(probe, hidden, caches)
    del hidden
    mx.synchronize()
    gc.collect()
    snapshot('prefill_seed_16k', time.perf_counter() - started)
    assert all(c.offset == 16384 and c.window.shape == (1, 128, 512) for c in caches)
    token = mx.array([1])
    mx.eval(token)
    ids = []
    for compiled in (False, True):
        phase = 'draft_compiled' if compiled else 'draft_eager'
        dsp._DRAFT_COMPILE = compiled
        mx.reset_peak_memory()
        start_active = mx.get_active_memory()
        times = []
        for i in range(4):
            t0 = time.perf_counter()
            out, logits, confidence = probe.mtp.draft_block(
                main_h, token, caches, probe.model.embed_tokens, probe.head)
            mx.eval(out, logits, confidence)
            times.append(time.perf_counter() - t0)
            if i == 0:
                assert bool(mx.all(mx.isfinite(logits)).item())
                ids.append(out.tolist())
            del out, logits, confidence
        mx.synchronize()
        snapshot(phase, statistics.median(times[1:]))
        report[phase] = {'cold_s': times[0], 'warm_s': times[1:],
                         'baseline_active_bytes': start_active,
                         'peak_delta_bytes': mx.get_peak_memory() - start_active}
    report['eager_compiled_draft_ids_equal'] = ids[0] == ids[1]
    report['draft_ids'] = ids
    report['ok'] = True
finally:
    stop.set()
    thread.join(2)
    report['os_samples'].append({'phase': phase, 'snapshot': host_memory_snapshot()})
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print('DRAFT_RECEIPT', str(OUT), flush=True)
