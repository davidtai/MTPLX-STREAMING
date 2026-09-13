"""Full workload after per-layer prefill projection release; no saving credited."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import threading

from mtplx.deepseek_v41_memory_profile import host_memory_snapshot

signal.alarm(1200)
PREFIX = Path('/tmp/dsv41-110-preflight/python-16k-1024-dspark-layer-prefill')
GIB = 1024**3
RECORD = 18800640
FIXED_MTP = 31931661128
MTP_ADDITIONAL = 8353408392
MTP_PEAK = 89611285956
BASE_SLOTS = 68
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
    raise RuntimeError('tracked source must be clean')
base = float(os.environ['MTPLX_DSV41_BOX_BASELINE_GB']) * 1e9
if not 0 <= base <= 20e9:
    raise RuntimeError('invalid measured baseline')
if os.environ.get('MTPLX_DSV41_IO_READ_FANOUT') != '4':
    raise RuntimeError('explicit fanout4 required')
if os.environ.get('MTPLX_BELADY_ORACLE', '0') != '0':
    raise RuntimeError('no route instrumentation')

prior = json.loads(Path('docs/deepseek-v41/receipts/dspark-cache-owned-20260913/summary.json').read_text())
assert prior['source_commit'] == '3a8b17284ef3ed6fa06cd130a0a56700e926ba62'
assert prior['slots_per_layer'] == BASE_SLOTS
assert round(prior['dspark']['peak_gb'] * 1e9) == MTP_PEAK
# Measured full-workload peak plus exact expert storage delta. No credit for
# reclaimed projection caches or smaller AR prefill outputs. Retain 1 GiB for
# graph/workspace variation and separately price both Python and allocator cache.
engine = int(110e9 - base - 2 * GIB - 11 * GIB)
slots = (engine - FIXED_MTP) // (40 * RECORD)
assert 1 <= slots <= 75
active_bound = MTP_PEAK + (slots - BASE_SLOTS) * 40 * RECORD + GIB
prior_full_physical_bound = prior['physical_peak_bytes'] + base - 10367600000 + (slots - BASE_SLOTS) * 40 * RECORD + GIB
physical_bound = max(base + active_bound + 2 * GIB + 2 * GIB, prior_full_physical_bound)
wired = host_memory_snapshot()['box']['wired_bytes']
if physical_bound > 109e9 or wired + active_bound + 2 * GIB > 100 * GIB:
    raise RuntimeError('bounded phase peak lacks physical or wired headroom')
bounds = {
    'source_commit': source, 'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'baseline_bytes': base, 'engine_budget_bytes': engine, 'expected_slots_per_layer': slots,
    'active_bound_bytes': active_bound, 'physical_bound_bytes': physical_bound,
    'wired_before_bytes': wired, 'graph_workspace_margin_bytes': GIB,
    'prior_measured_mtp_peak_bytes': MTP_PEAK, 'prior_whole_workflow_projection_bytes': prior_full_physical_bound, 'prior_slots_per_layer': BASE_SLOTS,
    'prior_source_commit': prior['source_commit'], 'projection_peak_saving_credited_bytes': 0,
    'scope': 'exact native artifact,16K prompt,1023 steps,depth5,pf0,eager target attention,compiled draft,bf16 heads; 11GiB band,2GiB cache,2GiB Python',
}
PREFIX.with_suffix('.bounds.json').write_text(json.dumps(bounds, indent=2) + '\n')
print('MTP_BOUND', json.dumps(bounds), flush=True)
def record_pass(kind, result):
    row = {'pass':kind, 'decode_tok_s':(len(result['generated'])-1)/result['decode_wall_s'],
           'decode_wall_s':result['decode_wall_s'], 'mlx_peak_bytes':result['memory']['mlx_peak_bytes'],
           'output_ids_sha256':hashlib.sha256(json.dumps(result['generated']).encode()).hexdigest()}
    with PREFIX.with_suffix('.passes.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print('PASS_COMPLETE',json.dumps(row),flush=True)
stop = threading.Event()
phase = 'startup'
def sample():
    with PREFIX.with_suffix('.os.jsonl').open('a') as f:
        f.write(json.dumps({'phase': phase, 'snapshot': host_memory_snapshot()}) + '\n')
def monitor():
    while not stop.wait(.25):
        sample()
thread = threading.Thread(target=monitor, daemon=True)
sample()
thread.start()
try:
    spec = importlib.util.spec_from_file_location('ab', 'scripts/deepseek_v41/ab_decode_env_levers.py')
    ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ab)
    args = ab.build_parser().parse_args()
    fixture = Path('docs/deepseek-v41/receipts/memory-budget-110/python-prompt-ids.json')
    rows = [r for r in json.loads(fixture.read_text())['prompts'] if r['target_tokens'] == 16384]
    assert len(rows) == 1
    assert hashlib.sha256(json.dumps(rows[0]['token_ids']).encode()).hexdigest() == '38894d01011a0146f8621dfdaf4bc4e618092d772d8cc28a70e21163a16799a2'
    if (Path(args.model).resolve() != Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
        or Path(args.prompt_ids_file).resolve() != fixture.resolve()
        or args.arms != ['cell16k_ring_v2_draft_attn_pf0'] or args.decode_mode != 'dspark'
        or args.context_tokens != 16384 or args.decode_tokens != 1023 or args.dspark_depth != 5
        or args.max_kv != 17664 or args.box_target_gb != 110 or args.transient_band_gib != 11
        or args.allocator_cache_gib != 2 or args.host_overhead_gib != 2
        or args.memory_plan_from or args.memory_limit_gib is not None or args.expert_cache_limit_gib is not None
        or args.slot_layout != 'component-banks' or args.expert_profile != 'deepseek-v41-mxfp4-75'
        or args.transient_slots not in (None, 48) or not args.apply_memory_cap
        or args.mlx_limit_headroom_gib is not None or args.dry_run
        or args.prompt_seed != 20260829 or not args.stop_on_eos
        or args.warm_repeat or args.stage_timing or args.prefill_stage_timing or args.syncs
        or ab._device_sample_resolved(args)):
        raise RuntimeError('arguments differ from the bounded MTP workload')
    # Validate the resolved configuration at the allocation boundary, BEFORE the
    # component-bank factory or runtime can allocate anything. CLI validation
    # alone cannot cover profile/env changes that alter the actual pool plan.
    from mtplx.models import deepseek_v41_loader as loader
    from mtplx.expert_runtime import text_only_resident_discount
    original_allocator = loader._component_bank_allocator_for
    def checked_allocator(config, spec, root, manifest_path, manifest=None, *, additional_resident_bytes=None):
        if (config.slot_layout != 'component-banks' or config.transient_slots != 48
            or config.prefetch_slots != 0 or config.max_live_kv_tokens != 17664
            or config.memory_limit_bytes != engine or config.runtime_reserve_bytes != 7 * GIB
            or config.proj_quant is not None or config.proj_requant is not None
            or config.island_layers or config.mmap_island_layers
            or not spec.mtp_included or spec.key != 'deepseek-v41-flash-expert-mxfp4'):
            raise RuntimeError('resolved allocation configuration differs from admitted geometry')
        if manifest is None:
            manifest = loader.load_expert_manifest(manifest_path)
        plan = config.memory_plan(spec, additional_resident_bytes=additional_resident_bytes,
                                  resident_discount_bytes=text_only_resident_discount(manifest, spec))
        if plan.fixed_bytes != FIXED_MTP or plan.slots_per_layer != slots:
            raise RuntimeError('resolved memory plan differs BEFORE allocation')
        return original_allocator(config, spec, root, manifest_path, manifest,
                                  additional_resident_bytes=additional_resident_bytes)
    loader._component_bank_allocator_for = checked_allocator
    original_load = ab._load_model
    def checked_load(*a, **kw):
        resident = original_load(*a, **kw)
        rt = resident.model._mtplx_expert_runtime
        if not rt.spec.mtp_included or rt.plan.slots_per_layer != slots:
            raise RuntimeError('loaded MTP plan differs from static accounting')
        return resident
    ab._load_model = checked_load
    original_ar = ab._generate
    original_mtp = ab._generate_dspark
    def admitted_ar(*a, **kw):
        global phase
        phase = 'ar_reference'
        sample()
        rt = kw['model']._mtplx_expert_runtime
        with rt.admit_kv_tokens(len(kw['prompt_ids']) + int(kw['steps'])):
            result = original_ar(*a, **kw)
        assert rt._live_kv_tokens == 0
        phase = 'ar_reference_end'
        sample()
        record_pass('ar_reference', result)
        return result
    def admitted_mtp(*a, **kw):
        global phase
        phase = 'dspark'
        sample()
        rt = kw['model']._mtplx_expert_runtime
        with rt.admit_kv_tokens(len(kw['prompt_ids']) + int(kw['steps']) + int(kw['depth']) + 1):
            result = original_mtp(*a, **kw)
        assert rt._live_kv_tokens == 0
        phase = 'dspark_end'
        sample()
        record_pass('dspark', result)
        return result
    ab._generate = admitted_ar
    ab._generate_dspark = admitted_mtp
    raise SystemExit(ab.main())
finally:
    stop.set()
    thread.join(2)
    sample()
