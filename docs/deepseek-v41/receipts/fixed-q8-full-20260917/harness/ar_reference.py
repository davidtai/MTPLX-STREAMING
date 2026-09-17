"""Guarded 16K DeepSeek-V4.1 stage with compact resident MTP banks."""
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading

if os.environ.get('_GPU_WINDOW_LOCKED') != '1':
    raise RuntimeError('GPU guard must hold lock before MLX import')

PACKED_ROOT = Path('/private/tmp/dsv41-q8-full-20260917/packed')
PACKED_INSTALLATION = json.loads((PACKED_ROOT / 'installation.json').read_text())
for _name, _digest in PACKED_INSTALLATION['helper_sha256'].items():
    if hashlib.sha256((PACKED_ROOT / _name).read_bytes()).hexdigest() != _digest:
        raise RuntimeError('packed phase helper changed')
if os.environ.get('DSV41_CACHE_GROWTH') != '1':
    raise RuntimeError('packed phase requires explicit growth installation')

from mtplx.deepseek_v41_memory_profile import host_memory_snapshot
import mlx.core as mx
from mlx_lm.models.switch_layers import SwitchGLU
from mtplx.models import deepseek_v41 as dsv41
from mtplx.models import deepseek_v41_dspark as dspark
from mtplx.models import deepseek_v41_loader as loader
from mtplx import expert_runtime


# Candidate-only installation. The selected banks contain every MTP expert used
# by the prior exact 16K/1K control trace. Missing experts map to compact slot 0;
# target verification remains authoritative, so an unseen draft route can only
# affect acceptance, never the committed target token.
TRACE_PATH = Path(
    "docs/deepseek-v41/receipts/mtp-route-capture-20260913/"
    "mtp-routes-16k-1024.json"
)
TRACE_SHA256 = "6a90006c9c4829dac2a548298c1d237bfd1beb793c0bc6eaf99315fad8c9c3dd"
if hashlib.sha256(TRACE_PATH.read_bytes()).hexdigest() != TRACE_SHA256:
    raise RuntimeError("MTP route trace identity changed")
_trace = json.loads(TRACE_PATH.read_text())
SELECTED_MTP_EXPERTS = tuple(
    tuple(
        sorted(
            {
                int(expert)
                for cycle in _trace["draft_routes_by_cycle_stage_row"]
                for row in cycle[stage]
                for expert in row
            }
        )
    )
    for stage in range(3)
)
if tuple(map(len, SELECTED_MTP_EXPERTS)) != (93, 58, 32):
    raise RuntimeError("unexpected compact MTP expert geometry")
_COMPACT_LUTS = []
for selected in SELECTED_MTP_EXPERTS:
    positions = {expert: slot for slot, expert in enumerate(selected)}
    _COMPACT_LUTS.append(
        mx.array([positions.get(expert, 0) for expert in range(128)], dtype=mx.int32)
    )


class _CompactMTPExpertSwitch(SwitchGLU):
    def __init__(self, input_dims, hidden_dims, selected, stage, activation):
        super().__init__(
            input_dims,
            hidden_dims,
            len(selected),
            activation=activation,
            bias=False,
        )
        self._compact_stage = int(stage)

    def __call__(self, x, indices):
        mapped = mx.take(_COMPACT_LUTS[self._compact_stage], indices)
        return super().__call__(x, mapped)


_original_dspark_block_init = dspark.DSparkBlock.__init__


def _compact_dspark_block_init(self, args, stage_id, n_stages):
    _original_dspark_block_init(self, args, stage_id, n_stages)
    original = self.mlp.switch_mlp
    self.mlp.switch_mlp = _CompactMTPExpertSwitch(
        args.hidden_size,
        args.moe_intermediate_size,
        SELECTED_MTP_EXPERTS[stage_id],
        stage_id,
        original.activation,
    )


dspark.DSparkBlock.__init__ = _compact_dspark_block_init

_MTP_EXPERT_RE = re.compile(r"mtp\.(\d+)\.ffn\.experts\.(\d+)\.")


def _kept_mtp_expert_name(name):
    match = _MTP_EXPERT_RE.match(name)
    if match is None:
        return True
    stage, expert = int(match.group(1)), int(match.group(2))
    return expert in SELECTED_MTP_EXPERTS[stage]


_original_map_mtp_residents = dsv41._map_mtp_residents


def _compact_map_mtp_residents(items):
    return _original_map_mtp_residents(
        {name: value for name, value in items.items() if _kept_mtp_expert_name(name)}
    )


dsv41._map_mtp_residents = _compact_map_mtp_residents

_original_partition_text_residents = loader.partition_text_residents


def _compact_partition_text_residents(manifest, *, with_mtp=False):
    partition = _original_partition_text_residents(manifest, with_mtp=with_mtp)
    if not with_mtp:
        return partition
    removed = tuple(
        tensor
        for tensor in partition.kept
        if not _kept_mtp_expert_name(tensor.tensor)
    )
    removed_bytes = sum(tensor.length for tensor in removed)
    kept = tuple(tensor for tensor in partition.kept if tensor not in removed)
    return replace(
        partition,
        kept=kept,
        skipped=partition.skipped + removed,
        kept_bytes=partition.kept_bytes - removed_bytes,
        kept_count=partition.kept_count - len(removed),
        skipped_bytes=partition.skipped_bytes + removed_bytes,
        skipped_count=partition.skipped_count + len(removed),
        skipped_mtp_bytes=partition.skipped_mtp_bytes + removed_bytes,
        skipped_mtp_count=partition.skipped_mtp_count + len(removed),
    )


loader.partition_text_residents = _compact_partition_text_residents

_original_load_text_only_resident_arrays = loader.load_text_only_resident_arrays
_COMPACT_RESIDENT_ROOT = Path('/tmp/dsv41-compact-residents')
_COMPACT_RESIDENT_RECEIPT = json.loads(
    (_COMPACT_RESIDENT_ROOT / 'receipt.json').read_text()
)
if _COMPACT_RESIDENT_RECEIPT['trace_sha256'] != TRACE_SHA256:
    raise RuntimeError('compact resident trace identity changed')
if tuple(
    tuple(map(int, values))
    for values in _COMPACT_RESIDENT_RECEIPT['selected_experts_by_stage']
) != SELECTED_MTP_EXPERTS:
    raise RuntimeError('compact resident expert selection changed')


def _compact_load_text_only_resident_arrays(
    root, manifest, *, mx_module=None, partition=None
):
    manifest_path = Path(root).resolve() / 'expert-manifest.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != (
        _COMPACT_RESIDENT_RECEIPT['source_manifest_sha256']
    ):
        raise RuntimeError('compact resident source manifest identity changed')
    if partition is None:
        partition = loader.partition_text_residents(manifest)
    compact = tuple(
        tensor for tensor in partition.kept
        if _MTP_EXPERT_RE.match(tensor.tensor)
    )
    regular = tuple(
        tensor for tensor in partition.kept
        if not _MTP_EXPERT_RE.match(tensor.tensor)
    )
    regular_bytes = sum(tensor.length for tensor in regular)
    regular_partition = replace(
        partition,
        kept=regular,
        kept_bytes=regular_bytes,
        kept_count=len(regular),
    )
    selected = _original_load_text_only_resident_arrays(
        root,
        manifest,
        mx_module=mx_module,
        partition=regular_partition,
    )
    mx_runtime = mx if mx_module is None else mx_module
    by_stage = {
        stage: tuple(
            tensor for tensor in compact
            if int(_MTP_EXPERT_RE.match(tensor.tensor).group(1)) == stage
        )
        for stage in range(3)
    }
    receipt_files = {
        int(item['stage']): item
        for item in _COMPACT_RESIDENT_RECEIPT['files']
    }
    paths = {
        f'mtp-selected-stage{stage}.safetensors':
            _COMPACT_RESIDENT_ROOT / f'mtp-selected-stage{stage}.safetensors'
        for stage in range(3)
    }
    retained_names = {
        f'mtp-selected-stage{stage}.safetensors':
            {tensor.tensor for tensor in by_stage[stage]}
        for stage in range(3)
    }
    for stage, path in enumerate(paths.values()):
        record = receipt_files.get(stage)
        if record is None or path.stat().st_size != int(record['file_bytes']):
            raise RuntimeError(f'compact resident stage {stage} provenance changed')
        if int(record['payload_bytes']) != sum(
            tensor.length for tensor in by_stage[stage]
        ):
            raise RuntimeError(f'compact resident stage {stage} byte plan changed')
    from mtplx.resident_io import ResidentShardReader
    with ResidentShardReader(paths, retained_names=retained_names) as reader:
        for stage, (name, path) in enumerate(paths.items()):
            loaded = reader.load(name, mx_runtime)
            expected = {tensor.tensor: tensor for tensor in by_stage[stage]}
            if set(loaded) != set(expected):
                raise RuntimeError(f'compact resident stage {stage} inventory changed')
            for tensor_name, tensor in expected.items():
                value = loaded[tensor_name]
                if tuple(map(int, value.shape)) != tensor.shape:
                    raise RuntimeError(f'compact resident shape mismatch for {tensor_name}')
                if loader._dtype_name(value) != tensor.dtype:
                    raise RuntimeError(f'compact resident dtype mismatch for {tensor_name}')
                if int(value.nbytes) != tensor.length:
                    raise RuntimeError(f'compact resident byte mismatch for {tensor_name}')
                if tensor_name in selected:
                    raise RuntimeError(f'duplicate compact resident {tensor_name}')
                selected[tensor_name] = value
            del loaded
    if len(selected) != partition.kept_count:
        raise RuntimeError('compact resident allowlist was not loaded completely')
    return selected


loader.load_text_only_resident_arrays = _compact_load_text_only_resident_arrays

MTP_PRUNED_EXPERTS = 3 * 128 - sum(map(len, SELECTED_MTP_EXPERTS))
MTP_PRUNED_BYTES = MTP_PRUNED_EXPERTS * 18_800_640
if (MTP_PRUNED_EXPERTS, MTP_PRUNED_BYTES) != (201, 3_778_928_640):
    raise RuntimeError("unexpected compact MTP resident saving")

_original_text_only_resident_discount = expert_runtime.text_only_resident_discount


def _compact_text_only_resident_discount(manifest, spec):
    discount = _original_text_only_resident_discount(manifest, spec)
    if getattr(spec, "mtp_included", False):
        discount += MTP_PRUNED_BYTES
    return discount


expert_runtime.text_only_resident_discount = _compact_text_only_resident_discount

# Validate source and phase helpers before loading a model.
import inspect
import textwrap
REFERENCE_SOURCE_COMMIT = 'e589c1e4b17856f506d90d9fb2bbb5ce45711648'
GROWTH_ROOT = Path('/private/tmp/dsv41-q8-full-20260917/compat')
COMPATIBILITY = json.loads((GROWTH_ROOT/'installation.json').read_text())
if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() != COMPATIBILITY['source_commit']:
    raise RuntimeError('candidate source commit changed')
for _path, _digest in COMPATIBILITY['runtime_source_sha256'].items():
    if hashlib.sha256(Path(_path).read_bytes()).hexdigest() != _digest:
        raise RuntimeError('candidate runtime source changed')
for _name, _digest in COMPATIBILITY['helper_sha256'].items():
    if hashlib.sha256((GROWTH_ROOT/_name).read_bytes()).hexdigest() != _digest:
        raise RuntimeError('candidate phase helper changed')
_candidate_prefill_source = textwrap.dedent(inspect.getsource(dsv41.DeepseekV41Backbone._forward_layer_major))
if _candidate_prefill_source.count('_PREFILL_HC_POST(moe_outputs[c], *carries[c])') != 1:
    raise RuntimeError('promoted prefill arithmetic is absent')
GROWTH_ENABLED = os.environ.get('DSV41_CACHE_GROWTH') == '1'
if os.environ.get('DSV41_CACHE_GROWTH') not in ('0','1'):
    raise RuntimeError('explicit DSV41_CACHE_GROWTH=0 or 1 required')
signal.alarm(1200)
GIB = 1024**3
RECORD = 18800640
SLOT_BAND_BYTES = 40 * RECORD
CONTROL_ENGINE_BUDGET_BYTES = 87_938_767_688
MAX_ENGINE_BUDGET_BYTES = 90_946_870_088
ENGINE_BUDGET_BYTES = CONTROL_ENGINE_BUDGET_BYTES
ALLOCATOR_LIMIT_BYTES = 98_105_916_352
CONTROL_SLOTS = 90
CONTROL_BASELINE_BYTES = 8_967_356_416
CONTROL_MLX_PEAK_BYTES = 94_138_170_460
CONTROL_PROCESS_PEAK_BYTES = 92_571_554_296
CONTROL_INTERNAL_SYSTEM_PEAK_BYTES = 101_926_273_024
CONTROL_EXTERNAL_SYSTEM_PEAK_BYTES = 105_137_651_712
CONTROL_OUTPUT_SHA256 = "0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac"
RESERVE_PLANS = {
    7: (25_535_486_792, 83, 737_181_696),
    3: (21_240_519_496, 89, 519_995_392),
    2: (20_166_777_672, 94, 89_686_016),
}
GRAPH_MARGIN = 2 * GIB


def _argv_value(flag, *, required=True, default=None):
    try:
        index = sys.argv.index(flag)
    except ValueError:
        if required:
            raise RuntimeError(f'explicit {flag} argument required') from None
        return default
    if index + 1 >= len(sys.argv):
        raise RuntimeError(f'{flag} has no value')
    return sys.argv[index + 1]


output_path = Path(_argv_value('--out')).resolve()
stage_root = Path('/tmp/dsv41-110-stage').resolve()
if output_path.parent != stage_root or output_path.suffix != '.jsonl':
    raise RuntimeError(f'--out must be a fresh .jsonl path under {stage_root}')
stage_root.mkdir(parents=True, exist_ok=True)
PREFIX = output_path.with_suffix('')
for candidate in (
    output_path,
    PREFIX.with_suffix('.bounds.json'),
    PREFIX.with_suffix('.passes.jsonl'),
    PREFIX.with_suffix('.os.jsonl'),
):
    if candidate.exists():
        raise RuntimeError(f'refusing to overwrite stage evidence: {candidate}')

runtime_reserve_raw = float(_argv_value('--runtime-reserve-gib'))
runtime_reserve_gib = int(runtime_reserve_raw)
if runtime_reserve_raw != runtime_reserve_gib or runtime_reserve_gib not in RESERVE_PLANS:
    raise RuntimeError('--runtime-reserve-gib must be exactly 7, 3, or 2')
FIXED_MTP, TARGET_SLOTS, TARGET_REMAINDER = RESERVE_PLANS[runtime_reserve_gib]
FIXED_MTP += 503316480
stage_depth = int(_argv_value('--dspark-depth'))
if stage_depth != 5 or runtime_reserve_gib != 2:
    raise RuntimeError('this matched capacity screen requires depth5/reserve2')
decode_steps = int(_argv_value('--decode-tokens'))
if decode_steps not in (128, 1023):
    raise RuntimeError('--decode-tokens must be 128 for a screen or 1023 for acceptance')
cache_policy = str(_argv_value('--cache-policy'))
if cache_policy not in ('frequency', 'transition-window'):
    raise RuntimeError('stage wrapper accepts frequency or transition-window only')
miss_part_raw = _argv_value(
    '--decode-miss-records-per-part', required=False, default=None
)
miss_records_per_part = None if miss_part_raw is None else int(miss_part_raw)
if miss_records_per_part not in (None, 2, 3):
    raise RuntimeError('stage wrapper accepts decode miss parts of 2 or 3 records')
shared_overlap = '--verify-shared-overlap' in sys.argv
if '--no-verify-shared-overlap' in sys.argv:
    if shared_overlap:
        raise RuntimeError('conflicting verify-shared-overlap flags')
    shared_overlap = False

source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
    raise RuntimeError('tracked source must be clean')
base = int(round(float(os.environ['MTPLX_DSV41_BOX_BASELINE_GB']) * 1e9))
if not 0 <= base <= 20e9:
    raise RuntimeError('invalid measured baseline')

from q8_admission import resolve_admission
if decode_steps != 1023 or cache_policy != 'transition-window' or miss_records_per_part != 3 or not shared_overlap:
    raise RuntimeError('growth comparison requires the complete unchanged D5/M6 workload')
if os.environ.get('MTPLX_DSV41_IO_READ_FANOUT') != '4' or os.environ.get('MTPLX_BELADY_ORACLE','0') != '0':
    raise RuntimeError('fanout4 and no route instrumentation required')
growth_admission = resolve_admission(base, host_memory_snapshot()['box']['wired_bytes'],
    grow=GROWTH_ENABLED, expected_receipt_hash=COMPATIBILITY['phase_memory_control_sha256'])
ALLOCATOR_LIMIT_BYTES = growth_admission['allocator_limit_bytes']
slots = TARGET_SLOTS = growth_admission['prefill_slots_per_layer']
TARGET_REMAINDER = 89686016
engine = ENGINE_BUDGET_BYTES = FIXED_MTP + slots * SLOT_BAND_BYTES + TARGET_REMAINDER
if engine != 90194844488 + 503316480 + (slots - 93) * SLOT_BAND_BYTES:
    raise RuntimeError('prefill engine geometry changed')
transient_band_bytes = ALLOCATOR_LIMIT_BYTES - engine
if not 6 * GIB <= transient_band_bytes <= 20 * GIB:
    raise RuntimeError('prefill transient band lacks the established minimum')
transient_band_gib = transient_band_bytes / GIB
transient_arg = sys.argv.index('--transient-band-gib')
if sys.argv[transient_arg + 1] != 'auto':
    raise RuntimeError('dynamic admitted transient band required')
sys.argv[transient_arg + 1] = f'{transient_band_gib:.17g}'
bounds = {
 'source_commit':source, 'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'baseline_bytes':base, 'engine_budget_bytes':engine, 'expected_slots_per_layer':slots,
 'decode_expected_slots_per_layer':growth_admission['decode_slots_per_layer'],
 'post_prefill_growth':GROWTH_ENABLED, 'growth_admission':growth_admission,
 'packed_scales_installation':PACKED_INSTALLATION,
 'runtime_reserve_gib':runtime_reserve_gib, 'runtime_reserve_bytes':runtime_reserve_gib * GIB,
 'fixed_footprint_bytes':FIXED_MTP, 'plan_remainder_bytes':TARGET_REMAINDER,
 'fixed_q8_cache':COMPATIBILITY['q8_change'],
 'transient_band_gib':transient_band_gib, 'transient_band_bytes':transient_band_bytes,
 'allocator_limit_bytes':ALLOCATOR_LIMIT_BYTES, 'requested_allocator_cache_limit_bytes':GIB,
 'active_bound_bytes':growth_admission['active_bound_bytes'],
 'physical_bound_bytes':growth_admission['physical_bound_bytes'],
 'decode_steps':decode_steps, 'dspark_depth':stage_depth,
 'prefill_mtp_hidden_capture_at_existing_fence':True, 'prefill_hc_post_compiled':True,
 'source_compatibility':COMPATIBILITY,
 'candidate_prefill_sha256':hashlib.sha256(_candidate_prefill_source.encode()).hexdigest(),
 'cache_policy':cache_policy, 'decode_miss_records_per_part':miss_records_per_part,
 'verify_shared_overlap':shared_overlap,
 'selected_mtp_experts_by_stage':list(map(len,SELECTED_MTP_EXPERTS)),
 'mtp_pruned_experts':MTP_PRUNED_EXPERTS, 'mtp_pruned_bytes':MTP_PRUNED_BYTES,
 'scope':COMPATIBILITY['scope'], 'actual_bound_model_levers':COMPATIBILITY['explicit_bound_model_levers'],
}
PREFIX.with_suffix('.bounds.json').write_text(json.dumps(bounds, indent=2) + '\n')

def record_pass(kind, result):
    row = {'pass':kind, 'decode_tok_s':(len(result['generated'])-1)/result['decode_wall_s'],
           'decode_wall_s':result['decode_wall_s'], 'mlx_peak_bytes':result['memory']['mlx_peak_bytes'],
           'process_footprint_peak_bytes':result['memory']['process_footprint_peak_bytes'],
           'system_used_peak_bytes':result['memory']['system_used_peak_bytes'],
           'output_ids_sha256':hashlib.sha256(json.dumps(result['generated']).encode()).hexdigest(),
           'prefill_slots_per_layer':slots, 'decode_slots_per_layer':growth_admission['decode_slots_per_layer'], 'growth_payload_bytes':growth_admission['growth_payload_bytes'], 'post_prefill_growth':dict(growth_report), 'runtime_reserve_gib':runtime_reserve_gib,
           'cache_policy':cache_policy,
           'decode_miss_records_per_part':miss_records_per_part,
           'verify_shared_overlap':shared_overlap,
           'selected_mtp_experts_by_stage': list(map(len, SELECTED_MTP_EXPERTS))}
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
    _bound_prefill_flags = COMPATIBILITY['explicit_bound_model_levers']
    ab.ARM_PRESETS[args.arms[0]] = {
        **ab.ARM_PRESETS[args.arms[0]],
        **{key: '0' for key in _bound_prefill_flags},
    }
    fixture = Path('docs/deepseek-v41/receipts/memory-budget-110/python-prompt-ids.json')
    rows = [r for r in json.loads(fixture.read_text())['prompts'] if r['target_tokens'] == 16384]
    assert len(rows) == 1
    assert hashlib.sha256(json.dumps(rows[0]['token_ids']).encode()).hexdigest() == '38894d01011a0146f8621dfdaf4bc4e618092d772d8cc28a70e21163a16799a2'
    if (Path(args.model).resolve() != Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')
        or Path(args.prompt_ids_file).resolve() != fixture.resolve()
        or args.arms != ['cell16k_ring_v2_draft_attn_pf0'] or args.decode_mode != 'ar' or not args.with_mtp
        or args.context_tokens != 16384 or args.decode_tokens != decode_steps or args.dspark_depth != stage_depth
        or args.max_kv != 17664 or args.box_target_gb != 110 or args.kv_cache_bits != 8 or args.kv_max_append != 953
        or int(round(args.transient_band_gib * GIB)) != transient_band_bytes
        or args.allocator_cache_gib != 1 or args.host_overhead_gib != 2.625
        or float(args.runtime_reserve_gib) != runtime_reserve_gib
        or args.cache_policy != cache_policy
        or args.decode_miss_records_per_part != miss_records_per_part
        or bool(args.verify_shared_overlap) != shared_overlap
        or Path(args.out).resolve() != output_path
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
            or config.memory_limit_bytes != engine
            or config.runtime_reserve_bytes != runtime_reserve_gib * GIB
            or config.cache_policy != cache_policy
            or config.decode_miss_records_per_part != miss_records_per_part
            or bool(config.verify_shared_overlap) != shared_overlap
            or not config.overlap_miss_reads or config.streamed_codec != 'none'
            or config.proj_quant is not None or config.proj_requant is not None
            or config.island_layers or config.mmap_island_layers
            or not spec.mtp_included or spec.key != 'deepseek-v41-flash-expert-mxfp4'):
            raise RuntimeError(
                'resolved allocation configuration differs from admitted geometry: '
                f'slot_layout={config.slot_layout!r}, transient_slots={config.transient_slots!r}, '
                f'prefetch_slots={config.prefetch_slots!r}, max_live_kv_tokens={config.max_live_kv_tokens!r}, '
                f'memory_limit_bytes={config.memory_limit_bytes!r}, runtime_reserve_bytes={config.runtime_reserve_bytes!r}, '
                f'cache_policy={config.cache_policy!r}, decode_miss_records_per_part={config.decode_miss_records_per_part!r}, '
                f'verify_shared_overlap={config.verify_shared_overlap!r}, overlap_miss_reads={config.overlap_miss_reads!r}, '
                f'proj_quant={config.proj_quant!r}, proj_requant={config.proj_requant!r}, '
                f'island_layers={config.island_layers!r}, mmap_island_layers={config.mmap_island_layers!r}, '
                f'mtp_included={spec.mtp_included!r}, spec_key={spec.key!r}'
            )
        if manifest is None:
            manifest = loader.load_expert_manifest(manifest_path)
        plan = config.memory_plan(spec, additional_resident_bytes=additional_resident_bytes,
                                  resident_discount_bytes=text_only_resident_discount(manifest, spec))
        if plan.fixed_bytes != FIXED_MTP or plan.slots_per_layer != slots:
            raise RuntimeError('resolved memory plan differs BEFORE allocation')
        return original_allocator(config, spec, root, manifest_path, manifest,
                                  additional_resident_bytes=additional_resident_bytes)
    loader._component_bank_allocator_for = checked_allocator
    from packed_phase import install_growth
    growth_transition = None
    growth_report = {}
    prefill_resolved_plan = {}
    original_load = ab._load_model
    def checked_load(*a, **kw):
        global growth_transition, growth_report, prefill_resolved_plan
        resident = original_load(*a, **kw)
        loaded_args = a[0] if a else kw['args']
        if loaded_args._dsv41_bound_model_levers != _bound_prefill_flags:
            raise RuntimeError('explicit flags were not bound at construction')
        rt = resident.model._mtplx_expert_runtime
        if (not rt.spec.mtp_included or rt.plan.slots_per_layer != slots
            or rt.config.cache_policy != cache_policy
            or rt.config.decode_miss_records_per_part != miss_records_per_part
            or bool(rt.config.verify_shared_overlap) != shared_overlap
            or not rt.config.overlap_miss_reads
            or not rt._single_slot_pool):
            raise RuntimeError('loaded MTP plan differs from static accounting')
        actual = []
        for stage, selected in zip(resident.model.mtp.layers, SELECTED_MTP_EXPERTS):
            switch = stage.mlp.switch_mlp
            if not isinstance(switch, _CompactMTPExpertSwitch):
                raise RuntimeError('compact MTP switch was not installed')
            shapes = {
                int(switch.gate_proj.weight.shape[0]),
                int(switch.up_proj.weight.shape[0]),
                int(switch.down_proj.weight.shape[0]),
            }
            if shapes != {len(selected)}:
                raise RuntimeError(f'compact MTP bank shape mismatch: {shapes}')
            actual.append(next(iter(shapes)))
        if tuple(actual) != tuple(map(len, SELECTED_MTP_EXPERTS)):
            raise RuntimeError('compact MTP stage geometry mismatch')
        prefill_resolved_plan = ab._resolved_plan(rt, loaded_args)
        if GROWTH_ENABLED:
            growth_transition, growth_report = install_growth(resident.model, growth_admission['decode_slots_per_layer'], mx=mx, admission=growth_admission)
        return resident
    ab._load_model = checked_load
    original_ar = ab._generate
    logit_report = {'path': str(PREFIX.with_suffix('.ar-logits.f32')), 'shape': [1024,129280],
                    'dtype': 'float32', 'row_bytes':129280*4, 'row_sha256':[],
                    'scope':'Complete Q8 M1 reference rows; capture adds diagnostic time to AR only'}
    prefill_boundary_memory = {}
    def admitted_ar(*a, **kw):
        global phase
        import fcntl
        import numpy as np
        phase = 'q8_ar_reference'
        sample()
        rt=kw['model']._mtplx_expert_runtime
        ops=kw['ops']
        fd=os.open(logit_report['path'],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        fcntl.fcntl(fd,fcntl.F_NOCACHE,1)
        class CaptureOps:
            def __getattr__(self,name):
                return getattr(ops,name)
            def argmax_last(self,logits):
                token=ops.argmax_last(logits)
                row=np.asarray(logits[0,-1].astype(mx.float32))
                if row.shape!=(129280,) or row.dtype!=np.float32:
                    raise RuntimeError('reference logit row shape/dtype changed')
                payload=row.tobytes()
                logit_report['row_sha256'].append(hashlib.sha256(payload).hexdigest())
                view=memoryview(payload)
                while view:
                    n=os.write(fd,view)
                    if n<=0:
                        raise RuntimeError('short Q8 reference logit write')
                    view=view[n:]
                return token
        kw['ops']=CaptureOps()
        callback=kw.get('mem_profile')
        def at_boundary(name,token=None):
            if name=='after_prefill':
                prefill_boundary_memory.update(active_bytes=int(mx.get_active_memory()),peak_bytes=int(mx.get_peak_memory()))
                growth_transition()
            if callback is not None:
                callback(name,token=token)
        kw['mem_profile']=at_boundary
        try:
            with rt.admit_kv_tokens(len(kw['prompt_ids'])+int(kw['steps'])):
                result=original_ar(*a,**kw)
            os.fsync(fd)
        finally:
            os.close(fd)
        if len(logit_report['row_sha256'])!=1024:
            raise RuntimeError('Q8 reference did not produce all 1024 logit rows')
        assert rt._live_kv_tokens==0
        phase='q8_ar_reference_end'
        sample()
        record_pass('q8_ar_reference',result)
        return result
    original_run_arm=ab._run_arm
    def reference_receipt(*a,**kw):
        receipt=original_run_arm(*a,**kw)
        if len(receipt['token_ids'])!=1024 or growth_report.get('phase')!='decode':
            raise RuntimeError('incomplete Q8 reference or packed phase transition')
        if receipt['fixed_q8_cache']['bits']!=8:
            raise RuntimeError('reference cache precision differs')
        receipt['reference_only']=True
        receipt['measurement_origin']={'ar':'current_run_with_logit_capture','dspark':None}
        receipt['reference_logit_storage']=logit_report
        receipt['post_prefill_growth']=dict(growth_report)
        receipt['prefill_boundary_memory']=prefill_boundary_memory
        receipt['phase_memory_plans']={'prefill':prefill_resolved_plan,'decode':receipt['resolved_plan']}
        receipt['phase_budget']={'allocator_limit_bytes':ALLOCATOR_LIMIT_BYTES,'prefill_engine_bytes':engine,
                                'decode_engine_bytes':engine+growth_admission['growth_payload_bytes'],
                                'admission':growth_admission}
        receipt['source_installation']=COMPATIBILITY
        return receipt
    ab._generate=admitted_ar
    ab._run_arm=reference_receipt
    raise SystemExit(ab.main())
finally:
    stop.set()
    thread.join(2)
    sample()
    # The export showed that clean source pages can block exact Qwen restore.
    # All GPU work and reader ownership have ended when ab.main unwinds here;
    # keep reclamation inside the same parent-owned GPU window.
    _cleanup_spec = importlib.util.spec_from_file_location('candidate_reclaim', 'scripts/deepseek_v41/reclaim_file_cache.py')
    _cleanup_module = importlib.util.module_from_spec(_cleanup_spec)
    _cleanup_spec.loader.exec_module(_cleanup_module)
    _inventory = json.loads((PACKED_ROOT / 'artifact/manifest.json').read_text())
    _source_path = Path(_inventory['source_path'])
    _source_stat = _source_path.stat()
    if {k: getattr(_source_stat, k) for k in _inventory['source_identity']} != _inventory['source_identity']:
        raise RuntimeError('candidate source identity changed before clean-cache reclamation')
    _cleanup = [_cleanup_module.reclaim_file(_source_path)]
    for _layer in _inventory['layers']:
        for _component in _layer['components'].values():
            for _field in ('descriptors', 'payload', 'bases'):
                _cleanup.append(_cleanup_module.reclaim_file(PACKED_ROOT / 'artifact' / _component[_field]['file']))
    _owned_logits = PREFIX.with_suffix('.ar-logits.f32')
    if _owned_logits.exists():
        _cleanup.append(_cleanup_module.reclaim_file(_owned_logits))
    PREFIX.with_suffix('.reclamation.json').write_text(json.dumps(_cleanup, indent=2) + '\n')
    print('PACKED_CACHE_RECLAIMED', json.dumps({'cached_page_bytes_before': sum(x['cached_page_bytes_before'] for x in _cleanup), 'cached_page_bytes_after': sum(x['cached_page_bytes_after'] for x in _cleanup)}), flush=True)
