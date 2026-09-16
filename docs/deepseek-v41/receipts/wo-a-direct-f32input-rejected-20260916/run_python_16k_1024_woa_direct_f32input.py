"""Exact full workload for packed-MXFP8 target ``wo_a`` with FP32 QMM input."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import threading

from mtplx.deepseek_v41_memory_profile import host_memory_snapshot

# Candidate-only installation: preserve the production tree until this exact gate wins.
import mlx.core as mx
from mtplx.models import deepseek_v41 as dsv41

def _project_grouped_f32_input(self, grouped):
    return mx.gather_qmm(
        grouped.astype(mx.float32),
        self.weight,
        self.scales,
        None,
        transpose=True,
        group_size=self.group_size,
        bits=self.bits,
        mode=self.mode,
    ).astype(mx.bfloat16)

dsv41._DirectMXFP8OLoraOut.project_grouped = _project_grouped_f32_input

signal.alarm(1200)
PREFIX = Path('/tmp/dsv41-110-preflight/python-16k-1024-woa-direct-f32input')
PREFIX.parent.mkdir(parents=True, exist_ok=True)
GIB = 1024**3
RECORD = 18800640
FIXED_MTP = 26697169736
MTP_PEAK = 91897033560
BASE_SLOTS = 78
TARGET_BF16_CACHE_BYTES = 2684354560
TRANSIENT_BAND_GIB = 8.5
GRAPH_MARGIN = 2 * GIB
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

prior = json.loads(Path('docs/deepseek-v41/receipts/target-slot-band6-20260913/summary.json').read_text())
assert prior['source_commit'] == '5c7661db48a1bed22cef753e336f9aba911add24'
assert prior['slots_per_layer'] == BASE_SLOTS
assert prior['budget']['mlx_peak_bytes'] == MTP_PEAK
# The prior complete 78-slot run bounds every phase. Credit only the exact
# 40-layer BF16 target cache removed by the construction-bound direct route,
# then charge every added expert slot and a 2 GiB graph/workspace margin. The
# 8.5 GiB planning band lets the measured baseline choose 78-83 slots while the
# independent physical and wired bounds below enforce the 110 GB ceiling.
baseline_bytes = int(round(base))
engine = (
    110_000_000_000
    - baseline_bytes
    - 2 * GIB
    - int(round(TRANSIENT_BAND_GIB * GIB))
)
slots = (engine - FIXED_MTP) // (40 * RECORD)
assert 78 <= slots <= 83
active_bound = (
    MTP_PEAK
    - TARGET_BF16_CACHE_BYTES
    + (slots - BASE_SLOTS) * 40 * RECORD
    + GRAPH_MARGIN
)
prior_baseline = int(round(prior['budget']['measured_baseline_bytes']))
prior_full_physical_bound = (
    prior['budget']['sampled_physical_peak_bytes_250ms']
    + baseline_bytes
    - prior_baseline
    - TARGET_BF16_CACHE_BYTES
    + (slots - BASE_SLOTS) * 40 * RECORD
    + GRAPH_MARGIN
)
physical_bound = max(
    baseline_bytes + active_bound + 2 * GIB + 2 * GIB,
    prior_full_physical_bound,
)
wired = host_memory_snapshot()['box']['wired_bytes']
if physical_bound > 109e9 or wired + active_bound + 2 * GIB > 100 * GIB:
    raise RuntimeError('bounded phase peak lacks physical or wired headroom')
bounds = {
    'source_commit': source, 'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'baseline_bytes': baseline_bytes, 'engine_budget_bytes': engine, 'expected_slots_per_layer': slots,
    'active_bound_bytes': active_bound, 'physical_bound_bytes': physical_bound,
    'wired_before_bytes': wired, 'graph_workspace_margin_bytes': GRAPH_MARGIN,
    'prior_measured_mtp_peak_bytes': MTP_PEAK, 'prior_whole_workflow_projection_bytes': prior_full_physical_bound, 'prior_slots_per_layer': BASE_SLOTS,
    'prior_source_commit': prior['source_commit'],
    'projection_peak_saving_credited_bytes': TARGET_BF16_CACHE_BYTES,
    'transient_planning_band_gib': TRANSIENT_BAND_GIB,
    'scope': 'exact native artifact,16K prompt,1023 steps,depth5,pf0,direct packed MXFP8 target wo_a with FP32 QMM input and BF16 output,compiled draft,bf16 heads; 8.5GiB planning band,2GiB cache,2GiB Python',
}
PREFIX.with_suffix('.bounds.json').write_text(json.dumps(bounds, indent=2) + '\n')
print('MTP_BOUND', json.dumps(bounds), flush=True)
def record_pass(kind, result, route_state):
    row = {'pass':kind, 'decode_tok_s':(len(result['generated'])-1)/result['decode_wall_s'],
           'decode_wall_s':result['decode_wall_s'], 'mlx_peak_bytes':result['memory']['mlx_peak_bytes'],
           'output_ids_sha256':hashlib.sha256(json.dumps(result['generated']).encode()).hexdigest(),
           'attention_routes':route_state}
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
        or args.arms != ['cell16k_ring_v2_draft_attn_pf0_woa_direct'] or args.decode_mode != 'dspark'
        or args.context_tokens != 16384 or args.decode_tokens != 1023 or args.dspark_depth != 5
        or args.max_kv != 17664 or args.box_target_gb != 110
        or args.transient_band_gib != TRANSIENT_BAND_GIB
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
    def direct_route_state(model):
        report = dict(getattr(model, '_mtplx_resident_load_report', {}))
        route = report.get('attention_routes')
        expected = {
            'wo_a_mode': 'direct_mxfp8_gather_qmm',
            'layers_installed': 40,
            'groups': 8,
            'rank': 1024,
            'input_per_group': 4096,
        }
        if route != expected:
            raise RuntimeError(f'direct attention route was not installed: {route!r}')
        for layer in model.model.layers:
            attn = layer.attn
            if attn._out_prep_fused_impl.__class__.__name__ != '_DirectMXFP8OLoraOut':
                raise RuntimeError('target layer lost its prebound direct route')
            if type(attn._out_prep_fused_impl).project_grouped is not _project_grouped_f32_input:
                raise RuntimeError('target layer lost its FP32-input QMM candidate')
            if getattr(attn, '_wo_a_bf16T_cache', None) is not None:
                raise RuntimeError('direct route materialized a target BF16 cache')
            if getattr(attn, '_wo_a_dense_cache', None) is not None:
                raise RuntimeError('layer-major prefill retained a target F32 cache')
        return route
    def checked_load(*a, **kw):
        resident = original_load(*a, **kw)
        rt = resident.model._mtplx_expert_runtime
        if not rt.spec.mtp_included or rt.plan.slots_per_layer != slots:
            raise RuntimeError('loaded MTP plan differs from static accounting')
        direct_route_state(resident.model)
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
        route_state = direct_route_state(kw['model'])
        phase = 'ar_reference_end'
        sample()
        record_pass('ar_reference', result, route_state)
        return result
    def admitted_mtp(*a, **kw):
        global phase
        phase = 'dspark'
        sample()
        rt = kw['model']._mtplx_expert_runtime
        with rt.admit_kv_tokens(len(kw['prompt_ids']) + int(kw['steps']) + int(kw['depth']) + 1):
            result = original_mtp(*a, **kw)
        assert rt._live_kv_tokens == 0
        route_state = direct_route_state(kw['model'])
        phase = 'dspark_end'
        sample()
        record_pass('dspark', result, route_state)
        return result
    ab._generate = admitted_ar
    ab._generate_dspark = admitted_mtp
    raise SystemExit(ab.main())
finally:
    stop.set()
    thread.join(2)
    sample()
