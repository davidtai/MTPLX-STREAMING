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

# Verify the promoted source before installing the lifetime-only candidate.
import inspect
import textwrap
REFERENCE_SOURCE_COMMIT = 'e589c1e4b17856f506d90d9fb2bbb5ce45711648'
COMPATIBILITY = json.loads(Path('/tmp/dsv41-110-stage/hc-post-full-installation-ca207-20260917.json').read_text())
if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != COMPATIBILITY['source_commit']:
    raise RuntimeError('candidate source commit changed')
for _path, _digest in COMPATIBILITY['runtime_source_sha256'].items():
    if hashlib.sha256(Path(_path).read_bytes()).hexdigest() != _digest:
        raise RuntimeError('candidate runtime source changed')
_prefill_source = 'def _forward_layer_major(self, input_ids, cache, chunk, *,\n                         return_main_hidden: bool = False):\n    """Layer-major chunked prefill (K16): iterate every layer over all chunks\n    before the next layer, so each layer\'s routed-expert bank is streamed\n    ONCE across the whole prompt instead of once per chunk.\n\n    Correctness vs the chunk-major driver (:meth:`_forward_span` per span):\n\n    * **Attention stays causal + per chunk.** Within a layer the chunks run in\n      order 0..C-1; chunk ``c`` appends its post-RoPE KV to the same\n      append-only layer store and reads the accumulated window, so it attends\n      over chunks ``< c`` exactly as one-shot -- and the ``[chunk, H, T]`` score\n      transient that motivated W20 stays bounded to one chunk (never\n      concatenated).  Each chunk\'s attention half is evaluated before the next\n      chunk builds its graph, so only one score is live at a time.\n    * **The MoE reads the bank once.** After a layer\'s C attention halves, the\n      chunks\' routed-expert inputs are concatenated and fed to ``mlp`` in one\n      ``switch_mlp`` call (row-capped, below), so ``partition_route_waves``\n      gathers each of the layer\'s experts exactly once for the whole prompt.\n    * **Hyper-Connection state is resident per chunk.** Every chunk keeps its\n      own ``[b, chunk, hc_mult, hidden]`` stream and ``pre_mix`` across the\n      whole layer loop (all C together are ``hidden * hc_mult * s * bf16`` --\n      0.67 GB at 16 K, well under 1 GB); the ffn ``carry`` is transient within\n      a layer.  A per-chunk :class:`SharedAttentionRuntime` threads each\n      chunk\'s compressed-KV / index selection down the stack exactly as its\n      span would.\n    * **Engram + DSpark unchanged in order.** The engram history is advanced\n      once per chunk in position order up front (identical ``_buf``/``_len`` to\n      chunk-major) and each chunk\'s row ids are replayed to the engram hook via\n      a per-chunk view, so layers 1/14 write the same residual for the same\n      rows.  ``main_hidden`` captures the target-layer input per chunk and\n      concatenates in position order, so the DSpark draft seed spans the whole\n      prompt (its ``[:, -1:, :]`` slice is still the final prompt token)."""\n    _stime.set_schedule("layer_major")\n    b, s = input_ids.shape\n    # W107 (review LOW-1): fail an over-cap prefill BEFORE any lane is written.\n    _admit = getattr(cache, "assert_can_admit", None)\n    if callable(_admit):\n        _admit(s)\n    offset0 = int(cache.offset)\n    spans = [(start, min(start + chunk, s)) for start in range(0, s, chunk)]\n    n_chunks = len(spans)\n\n    engram_state = getattr(cache, "engram_state", None)\n    want_main = return_main_hidden and bool(self._mtp_target_layer_ids)\n\n    # Per-chunk resident state, built once.  The engram history is advanced in\n    # position order here (so `_buf`/`_len` end identical to chunk-major) and\n    # each chunk\'s returned row ids are captured for the hook replay below --\n    # the shared `_current` only holds the last advance, so we never read it.\n    hs: List[mx.array] = []\n    pre_mixes: List[mx.array] = []\n    positions_all: List[mx.array] = []\n    engram_currents: List[Optional[np.ndarray]] = []\n    shareds = [cache.new_shared_runtime() for _ in range(n_chunks)]\n    main_hiddens: List[List[mx.array]] = [[] for _ in range(n_chunks)]\n    for c, (start, end) in enumerate(spans):\n        ids_c = input_ids[:, start:end]\n        n_c = end - start\n        positions_all.append(mx.arange(offset0 + start, offset0 + end))\n        # W47: embed + engram.advance per chunk (tagged by chunk index), so the\n        # layer-major flat stages match chunk-major.  ``stage``/``chunk`` are\n        # no-ops off / decode; this loop is layer-major-only regardless.\n        with _stime.chunk(c):\n            with _stime.stage("embed") as _st:\n                h_c = self.embed_tokens(ids_c)\n                h_c = mx.broadcast_to(\n                    h_c[:, :, None, :], (b, n_c, self.hc_mult, h_c.shape[-1])\n                )\n                _st.add(h_c)\n            hs.append(h_c)\n            pre_mixes.append(\n                mx.concatenate(\n                    [mx.ones((b, n_c, 1)), mx.zeros((b, n_c, self.hc_mult - 1))],\n                    axis=-1,\n                ).astype(mx.float32)\n            )\n            if engram_state is not None:\n                with _stime.stage("engram.advance"):\n                    engram_currents.append(engram_state.advance(ids_c))\n            else:\n                engram_currents.append(None)\n\n    row_cap = _derive_moe_row_cap(self.args, _prefill_moe_row_target_bytes())\n    # Small verification calls can select this schedule with an explicit\n    # tiny chunk. Keep their decode caches; large prefills need each dense\n    # projection only until that layer\'s final chunk has completed.\n    release_dense_projection = b * s > _DECODE_ATTN_KERNEL_MAX_ROWS\n\n    for layer in self.layers:\n        lc = cache.layers[layer.layer_id]\n        is_target = want_main and layer.layer_id in self._mtp_target_layer_ids\n        moe_inputs: List[mx.array] = []\n        carries: List[tuple] = []\n        for c, (start, end) in enumerate(spans):\n            # W47: tag this (layer, chunk) attention half with the chunk index\n            # so ``by_chunk`` accumulates attention/HC/engram per chunk across\n            # every layer (the MoE is batched below, outside any chunk tag).\n            with _stime.chunk(c):\n                h_c = hs[c]\n                if layer.engram_hook is not None and engram_state is not None:\n                    h_c = layer.engram_hook(\n                        h_c, input_ids[:, start:end],\n                        _ChunkEngramView(engram_currents[c]),\n                    )\n                if is_target:\n                    main_hiddens[c].append(\n                        mx.mean(h_c.astype(mx.float32), axis=2).astype(h_c.dtype)\n                    )\n                moe_in_c, carry_c, ffn_pre_c = layer.attn_and_moe_input(\n                    h_c, pre_mixes[c], positions_all[c], lc, shareds[c]\n                )\n                moe_inputs.append(moe_in_c)\n                carries.append(carry_c)\n                pre_mixes[c] = ffn_pre_c\n                # Free this chunk\'s attention score before the next chunk\'s\n                # graph is built (only one [chunk, H, T] transient live at once).\n                self._eval_layer_transients(lc, moe_in_c, ffn_pre_c, main_hiddens[c])\n\n        # One routed-expert call per layer over every chunk\'s rows -> the bank\n        # is streamed once.  Split only if the row cap (routed-output transient\n        # budget) would be exceeded; at 16 K the whole prompt is one call.\n        moe_outputs = self._layer_major_moe(layer, moe_inputs, spans, row_cap)\n        for c in range(n_chunks):\n            hs[c] = layer.moe_combine(moe_outputs[c], carries[c])\n        mx.eval(hs)\n        if release_dense_projection:\n            # This existing fence has consumed the attention and FFN\n            # carries. Preserve reuse across chunks, then release before\n            # the next layer instead of retaining 40 fp32 weight copies.\n            layer.attn._wo_a_dense_cache = None\n\n    cache.advance(s)\n\n    outputs: List[mx.array] = []\n    main_parts: List[Optional[mx.array]] = []\n    for c in range(n_chunks):\n        with _stime.chunk(c), _stime.stage("final_norm") as _st:\n            h = mx.sum(\n                pre_mixes[c][..., None] * hs[c].astype(mx.float32), axis=2\n            ).astype(hs[c].dtype)\n            out_c = _rmsnorm(h, self.norm_weight, self.args.rms_norm_eps)\n            _st.add(out_c)\n        outputs.append(out_c)\n        main_parts.append(\n            mx.concatenate(main_hiddens[c], axis=-1) if main_hiddens[c] else None\n        )\n    out = mx.concatenate(outputs, axis=1)\n    if not return_main_hidden:\n        return out\n    main_hidden = (\n        None\n        if any(p is None for p in main_parts)\n        else mx.concatenate(main_parts, axis=1)\n    )\n    return out, main_hidden\n'
_installed_prefill_source = textwrap.dedent(inspect.getsource(
    dsv41.DeepseekV41Backbone._forward_layer_major))
_fence = 'self._eval_layer_transients(\n                    lc, moe_in_c, ffn_pre_c, main_hiddens[c]\n                )'
if _installed_prefill_source.count(_fence) != 1:
    raise RuntimeError('promoted prefill fence changed')
# Compile only post-MoE HC in the layer-major prefill path. Decode unchanged.
_combine = 'hs[c] = layer.moe_combine(moe_outputs[c], carries[c])'
if _installed_prefill_source.count(_combine) != 1:
    raise RuntimeError('prefill HC post call changed')
dsv41._prefill_hc_post = mx.compile(dsv41._hc_post_impl)
_candidate_prefill_source = _installed_prefill_source.replace(
    _combine, 'hs[c] = _prefill_hc_post(moe_outputs[c], *carries[c])')
_candidate_prefill_namespace = {}
exec(compile(_candidate_prefill_source, '<prefill-compiled-hc-post>', 'exec'),
     dsv41.__dict__, _candidate_prefill_namespace)
dsv41.DeepseekV41Backbone._forward_layer_major = _candidate_prefill_namespace['_forward_layer_major']

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

# The prior fixed slot cap assumed a 9.75 GB machine baseline. Keep its
# transient floor and host/cache reserves, but size the bank against today's
# measured baseline before any model or expert-bank allocation.
ALLOCATOR_LIMIT_BYTES = 110_000_000_000 - base - 2 * GIB
engine_ceiling = min(MAX_ENGINE_BUDGET_BYTES, ALLOCATOR_LIMIT_BYTES - 6 * GIB)
matched_control_bounds = None
matched_control_bounds_sha256 = None
matched_path = os.environ.get('DSV41_STAGE_MEMORY_CONTROL')
if matched_path:
    matched_path = Path(matched_path).resolve(strict=True)
    if matched_path.parent != stage_root or matched_path.suffixes[-2:] != ['.bounds', '.json']:
        raise RuntimeError('matched control bounds must belong to this stage')
    matched_bytes = matched_path.read_bytes()
    matched = json.loads(matched_bytes)
    if (matched.get('source_commit') != source
        or matched.get('runtime_reserve_gib') != runtime_reserve_gib
        or matched.get('decode_steps') != decode_steps
        or matched.get('cache_policy') != 'frequency'
        or matched.get('decode_miss_records_per_part') is not None
        or matched.get('verify_shared_overlap') is not False
        or matched.get('live_baseline_slot_cap') is not True
        or matched.get('fixed_footprint_bytes') != FIXED_MTP):
        raise RuntimeError('matched control is not the unchanged live-baseline control')
    ENGINE_BUDGET_BYTES = int(matched['engine_budget_bytes'])
    if ENGINE_BUDGET_BYTES > engine_ceiling:
        raise RuntimeError('current baseline cannot fit the measured control budget')
    matched_control_bounds = str(matched_path)
    matched_control_bounds_sha256 = hashlib.sha256(matched_bytes).hexdigest()
else:
    ENGINE_BUDGET_BYTES = engine_ceiling
TARGET_SLOTS, TARGET_REMAINDER = divmod(
    ENGINE_BUDGET_BYTES - FIXED_MTP, SLOT_BAND_BYTES
)
if not 1 <= TARGET_SLOTS <= RESERVE_PLANS[runtime_reserve_gib][1]:
    raise RuntimeError('live baseline leaves no admitted expert-bank geometry')
if os.environ.get('MTPLX_DSV41_IO_READ_FANOUT') != '4':
    raise RuntimeError('explicit fanout4 required')
if os.environ.get('MTPLX_BELADY_ORACLE', '0') != '0':
    raise RuntimeError('no route instrumentation')

control_path = Path('/tmp/dsv41-110-stage/live-combined-depth3-hidden-capture-1023.jsonl')
control_os_path = Path('/tmp/dsv41-110-stage/live-combined-depth3-hidden-capture-1023.os.jsonl')
control_bounds_path = Path('/tmp/dsv41-110-stage/live-combined-depth3-hidden-capture-1023.bounds.json')
control = json.loads(next(control_path.open()))
control_bounds = json.loads(control_bounds_path.read_text())
if (control_bounds['source_commit'] != REFERENCE_SOURCE_COMMIT
    or control_bounds.get('prefill_mtp_hidden_capture_at_existing_fence') is not True
    or control_bounds.get('patched_prefill_sha256') != hashlib.sha256(_prefill_source.encode()).hexdigest()
    or control_bounds.get('baseline_bytes') != CONTROL_BASELINE_BYTES
    or control_bounds.get('dspark_depth') != 3
    or control_bounds.get('cache_policy') != 'transition-window'
    or control_bounds.get('decode_miss_records_per_part') != 3
    or control_bounds.get('verify_shared_overlap') is not True):
    raise RuntimeError('capacity predecessor has different model or allocation behavior')

control_dspark = control['dspark']
control_memory = control_dspark['memory']
control_os_peak = max(
    int(row['snapshot']['box']['used_bytes'])
    for row in map(json.loads, control_os_path.open())
    if row['snapshot']['box'].get('ok')
)
if (
    control['resolved_plan']['slots_per_layer'] != CONTROL_SLOTS
    or control['resolved_plan']['memory_limit_bytes'] != CONTROL_ENGINE_BUDGET_BYTES
    or control_dspark['token_ids_sha256'] != CONTROL_OUTPUT_SHA256
    or int(control_memory['mlx_peak_bytes']) != CONTROL_MLX_PEAK_BYTES
    or int(control_memory['process_footprint_peak_bytes']) != CONTROL_PROCESS_PEAK_BYTES
    or int(control_memory['system_used_peak_bytes']) != CONTROL_INTERNAL_SYSTEM_PEAK_BYTES
    or control_os_peak != CONTROL_EXTERNAL_SYSTEM_PEAK_BYTES
):
    raise RuntimeError('measured hidden-capture cap-90 control evidence changed')

# Resolve the largest admissible bank at construction, including the
# retained allocator cache and actual wired headroom. A higher live baseline
# can reduce capacity instead of admitting an oversized bank or wasting a run.
nominal_slots = TARGET_SLOTS
wired = host_memory_snapshot()['box']['wired_bytes']
ACTIVE_ESTIMATE_MARGIN_BYTES = 64 * 1024**2
for admitted_slots in range(TARGET_SLOTS, 0, -1):
    delta = (admitted_slots - CONTROL_SLOTS) * SLOT_BAND_BYTES
    candidate_active = CONTROL_MLX_PEAK_BYTES + delta + ACTIVE_ESTIMATE_MARGIN_BYTES
    candidate_physical = (CONTROL_EXTERNAL_SYSTEM_PEAK_BYTES + delta
                          + 500_000_000 + max(0, base - CONTROL_BASELINE_BYTES))
    if (candidate_active + GIB <= ALLOCATOR_LIMIT_BYTES
        and candidate_physical <= 109_500_000_000
        and wired + candidate_active + GRAPH_MARGIN <= 100 * GIB):
        break
else:
    raise RuntimeError('live memory headroom cannot admit any expert-bank geometry')
if admitted_slots != TARGET_SLOTS:
    if matched_control_bounds is not None:
        raise RuntimeError('live headroom cannot fit the matched control capacity')
    TARGET_SLOTS = admitted_slots
    ENGINE_BUDGET_BYTES = (FIXED_MTP + TARGET_SLOTS * SLOT_BAND_BYTES
                           + TARGET_REMAINDER)

engine = FIXED_MTP + TARGET_SLOTS * SLOT_BAND_BYTES + TARGET_REMAINDER
if engine != ENGINE_BUDGET_BYTES:
    raise RuntimeError('corrected fixed footprint does not reproduce engine budget')
transient_band_bytes = int(110e9) - base - 2 * GIB - engine
if not 6 * GIB <= transient_band_bytes <= 20 * GIB:
    raise RuntimeError('measured baseline cannot safely produce the bounded slot geometry')
transient_band_gib = transient_band_bytes / GIB
try:
    transient_arg = sys.argv.index('--transient-band-gib')
except ValueError:
    raise RuntimeError('explicit transient-band argument required') from None
if transient_arg + 1 >= len(sys.argv) or sys.argv[transient_arg + 1] != 'auto':
    raise RuntimeError('compact candidate requires the guarded dynamic transient band')
sys.argv[transient_arg + 1] = f'{transient_band_gib:.17g}'
slots = TARGET_SLOTS
slot_delta_bytes = (slots - CONTROL_SLOTS) * SLOT_BAND_BYTES
projected_physical_peak = CONTROL_EXTERNAL_SYSTEM_PEAK_BYTES + slot_delta_bytes
baseline_drift_bytes = max(0, base - CONTROL_BASELINE_BYTES)
physical_bound = projected_physical_peak + 500_000_000 + baseline_drift_bytes
active_bound = CONTROL_MLX_PEAK_BYTES + slot_delta_bytes + ACTIVE_ESTIMATE_MARGIN_BYTES
process_footprint_projection = CONTROL_PROCESS_PEAK_BYTES + slot_delta_bytes
wired = host_memory_snapshot()['box']['wired_bytes']
if (
    active_bound + GIB > ALLOCATOR_LIMIT_BYTES
    or physical_bound > 109_500_000_000
    or wired + active_bound + GRAPH_MARGIN > 100 * GIB
):
    raise RuntimeError('bounded phase peak lacks physical or wired headroom')
bounds = {
    'source_commit': source, 'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'capacity_predecessor': str(control_path),
    'capacity_predecessor_sha256': hashlib.sha256(control_path.read_bytes()).hexdigest(),
    'capacity_predecessor_bounds_sha256': hashlib.sha256(control_bounds_path.read_bytes()).hexdigest(),
    'baseline_bytes': base, 'engine_budget_bytes': engine, 'expected_slots_per_layer': slots,
    'live_baseline_slot_cap': True,
    'nominal_slots_per_layer': nominal_slots,
    'active_estimate_margin_bytes': ACTIVE_ESTIMATE_MARGIN_BYTES,
    'admission_includes_retained_cache_and_wired_headroom': True,
    'matched_control_bounds': matched_control_bounds,
    'matched_control_bounds_sha256': matched_control_bounds_sha256,
    'requested_allocator_cache_limit_bytes': GIB,
    'runtime_reserve_gib': runtime_reserve_gib,
    'runtime_reserve_bytes': runtime_reserve_gib * GIB,
    'fixed_footprint_bytes': FIXED_MTP,
    'plan_remainder_bytes': TARGET_REMAINDER,
    'transient_band_gib': transient_band_gib,
    'transient_band_bytes': transient_band_bytes,
    'allocator_limit_bytes': ALLOCATOR_LIMIT_BYTES,
    'active_bound_bytes': active_bound,
    'process_footprint_projection_bytes': process_footprint_projection,
    'physical_bound_bytes': physical_bound,
    'wired_before_bytes': wired, 'graph_workspace_margin_bytes': GRAPH_MARGIN,
    'control_mlx_peak_bytes': CONTROL_MLX_PEAK_BYTES,
    'control_process_footprint_peak_bytes': CONTROL_PROCESS_PEAK_BYTES,
    'control_internal_system_peak_bytes': CONTROL_INTERNAL_SYSTEM_PEAK_BYTES,
    'control_external_system_peak_bytes': CONTROL_EXTERNAL_SYSTEM_PEAK_BYTES,
    'control_baseline_bytes': CONTROL_BASELINE_BYTES,
    'positive_baseline_drift_bytes': baseline_drift_bytes,
    'control_slots_per_layer': CONTROL_SLOTS,
    'control_source_commit': control_bounds['source_commit'],
    'slot_delta_bytes': slot_delta_bytes,
    'projected_physical_peak_bytes': projected_physical_peak,
    'decode_steps': decode_steps,
    'dspark_depth': stage_depth,
    'prefill_mtp_hidden_capture_at_existing_fence': True,
    'prefill_consumed_inputs_retired': False,
    'prefill_hc_post_compiled': True,
    'source_compatibility': COMPATIBILITY,
    'candidate_prefill_sha256': hashlib.sha256(_candidate_prefill_source.encode()).hexdigest(),
    'patched_prefill_sha256': hashlib.sha256(_prefill_source.encode()).hexdigest(),
    'max_target_rows': stage_depth + 1,
    'max_target_misses': (stage_depth + 1) * 6,
    'cache_policy': cache_policy,
    'decode_miss_records_per_part': miss_records_per_part,
    'verify_shared_overlap': shared_overlap,
    'selected_mtp_experts_by_stage': list(map(len, SELECTED_MTP_EXPERTS)),
    'mtp_pruned_experts': MTP_PRUNED_EXPERTS,
    'mtp_pruned_bytes': MTP_PRUNED_BYTES,
    'scope': COMPATIBILITY['scope'],
    'actual_bound_model_levers': COMPATIBILITY['explicit_bound_model_levers'],
}
# A prior complete AR run may supply only the comparison token stream.
# All current-run AR timings and memory fields are null; DSpark is measured
# normally and still captures/classifies its first divergence against AR.
reference_path = Path(os.environ['DSV41_STAGE_AR_REFERENCE']).resolve(strict=True)
if reference_path.parent != stage_root or reference_path.suffix != '.jsonl':
    raise RuntimeError('AR reference must be a receipt in this stage directory')
reference_bytes = reference_path.read_bytes()
reference_rows = [json.loads(line) for line in reference_bytes.decode().splitlines() if line]
if len(reference_rows) != 1:
    raise RuntimeError('AR reference must contain exactly one complete arm')
reference = reference_rows[0]
reference_bounds_path = reference_path.with_suffix('.bounds.json')
reference_bounds_bytes = reference_bounds_path.read_bytes()
reference_bounds = json.loads(reference_bounds_bytes)
reference_ids = reference.get('token_ids', [])
reference_digest = hashlib.sha256(json.dumps(reference_ids).encode()).hexdigest()
if (reference_bounds.get('source_commit') != REFERENCE_SOURCE_COMMIT
    or reference.get('ar_reference_reuse') is not None
    or reference.get('arm') != 'cell16k_ring_v2_draft_attn_pf0'
    or reference.get('prompt_ids_sha256') != '38894d01011a0146f8621dfdaf4bc4e618092d772d8cc28a70e21163a16799a2'
    or reference.get('prompt_tokens') != 16384
    or len(reference_ids) != decode_steps + 1
    or reference.get('token_ids_sha256') != reference_digest
    or not all(type(token) is int and 0 <= token < 129280 for token in reference_ids)
    or reference.get('decode_wall_s', 0) <= 0
    or reference.get('ttft_s', 0) <= 0
    or reference.get('resident_load_report', {}).get('head_mode') != 'bf16'
    or reference_bounds.get('decode_steps') != decode_steps
    or reference_bounds.get('selected_mtp_experts_by_stage') != [93, 58, 32]
    or reference_bounds.get('reclaim_decode_projection_at_prefill')
    or reference_bounds.get('prefill_chunk_tokens') is not None):
    raise RuntimeError('AR reference is not a complete matched native-target run')
for key in ('MTPLX_DSV41_PREFILL_CHUNK', 'MTPLX_DSV41_PREFILL_CHUNK_TARGET_GB',
            'MTPLX_DSV41_PREFILL_MOE_TARGET_GB'):
    if os.environ.get(key) is not None:
        raise RuntimeError('AR-reference reuse requires the unchanged prefill environment')
reference_provenance = {
    'receipt': str(reference_path),
    'receipt_sha256': hashlib.sha256(reference_bytes).hexdigest(),
    'bounds_sha256': hashlib.sha256(reference_bounds_bytes).hexdigest(),
    'source_commit': REFERENCE_SOURCE_COMMIT,
    'measurement_source_commit': source,
    'token_ids_sha256': reference_digest,
    'current_run_ar_measured': False,
    'reference_ar_decode_wall_s': reference['decode_wall_s'],
    'reference_ar_ttft_s': reference['ttft_s'],
}
bounds['ar_reference_reuse'] = reference_provenance
PREFIX.with_suffix('.bounds.json').write_text(json.dumps(bounds, indent=2) + '\n')
print('MTP_BOUND', json.dumps(bounds), flush=True)
def record_pass(kind, result):
    row = {'pass':kind, 'decode_tok_s':(len(result['generated'])-1)/result['decode_wall_s'],
           'decode_wall_s':result['decode_wall_s'], 'mlx_peak_bytes':result['memory']['mlx_peak_bytes'],
           'process_footprint_peak_bytes':result['memory']['process_footprint_peak_bytes'],
           'system_used_peak_bytes':result['memory']['system_used_peak_bytes'],
           'output_ids_sha256':hashlib.sha256(json.dumps(result['generated']).encode()).hexdigest(),
           'slots_per_layer':slots, 'runtime_reserve_gib':runtime_reserve_gib,
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
        or args.arms != ['cell16k_ring_v2_draft_attn_pf0'] or args.decode_mode != 'dspark'
        or args.context_tokens != 16384 or args.decode_tokens != decode_steps or args.dspark_depth != stage_depth
        or args.max_kv != 17664 or args.box_target_gb != 110
        or int(round(args.transient_band_gib * GIB)) != transient_band_bytes
        or args.allocator_cache_gib != 1 or args.host_overhead_gib != 2
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
    original_load = ab._load_model
    def checked_load(*a, **kw):
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
        probe = kw['mem_probe']
        original_peak = probe.peak_bytes
        def full_run_peak():
            value = original_peak()
            if value is None:
                return None
            return max(value, prefill_boundary_memory.get('mlx_peak_bytes', 0))
        probe.peak_bytes = full_run_peak
        phase = 'dspark'
        sample()
        rt = kw['model']._mtplx_expert_runtime
        with rt.admit_kv_tokens(len(kw['prompt_ids']) + int(kw['steps']) + int(kw['depth']) + 1):
            try:
                result = original_mtp(*a, **kw)
                result['memory']['mlx_peak_after_prefill_bytes'] = int(mx.get_peak_memory())
                result['memory']['mlx_peak_after_prefill_scope'] = 'MTP seed plus complete decode, reset before the decode timer starts'
            finally:
                probe.peak_bytes = original_peak
        assert rt._live_kv_tokens == 0
        phase = 'dspark_end'
        sample()
        record_pass('dspark', result)
        return result
    expected_arm_env = {
        key: ab.ARM_PRESETS[args.arms[0]].get(key)
        for key in ab._arm_env_snapshot()
    }
    if {k: v for k, v in reference.get('arm_env', {}).items() if k not in _bound_prefill_flags} != {
        k: v for k, v in expected_arm_env.items() if k not in _bound_prefill_flags
    }:
        raise RuntimeError('target environment differs outside explicit bound flags')

    def reused_ar_reference(**kw):
        global phase
        phase = 'ar_reference_reused'
        sample()
        print('AR_REFERENCE_REUSED', json.dumps(reference_provenance), flush=True)
        # These private placeholders pass the legacy receipt builder. The
        # public receipt below replaces every AR measurement with null.
        return {'generated': list(reference_ids), 'ttft_s': 0.0,
                'decode_wall_s': 0.0, 'peak_gb': None, 'memory': None,
                'decode_steps_run': len(reference_ids) - 1,
                'extra_forward_steps': 0}

    # One reset outside the decode clock isolates seed/decode peak. The memory
    # probe above retains max(prefill, post-prefill) for full-run headline fields.
    from mtplx.models import deepseek_v41_dspark_decode as decode_module
    original_dspark_generate = decode_module.dspark_generate
    prefill_boundary_memory = {}
    seed_boundary_memory = {}
    original_seed_prefill = decode_module._seed_prefill_state

    def observe_seed_prefill(*a, **kw):
        # This function runs only during prefill. No added eval or peak reset.
        seed_boundary_memory['before'] = {
            'mlx_active_bytes': int(mx.get_active_memory()),
            'mlx_peak_bytes': int(mx.get_peak_memory()),
        }
        result = original_seed_prefill(*a, **kw)
        seed_boundary_memory['after'] = {
            'mlx_active_bytes': int(mx.get_active_memory()),
            'mlx_peak_bytes': int(mx.get_peak_memory()),
        }
        return result

    decode_module._seed_prefill_state = observe_seed_prefill

    def dspark_with_boundary_observation(*a, **kw):
        callback = kw['prefill_callback']

        def observe_prefill_boundary(info):
            prefill_boundary_memory.update({
                'mlx_active_bytes': int(mx.get_active_memory()),
                'mlx_cache_bytes': int(mx.get_cache_memory()),
                'mlx_peak_bytes': int(mx.get_peak_memory()),
            })
            mx.reset_peak_memory()
            callback(info)

        kw['prefill_callback'] = observe_prefill_boundary
        return original_dspark_generate(*a, **kw)

    decode_module.dspark_generate = dspark_with_boundary_observation
    # Cache only the exact AR diagnostic row, after measurement. Subsequent
    # candidates still supply their own freshly captured DSpark logits to the
    # classifier. The key includes the complete source-pinned AR receipt.
    original_ar_logits_row = ab._ar_logits_row_at_index
    ar_logits_cache_report = {}

    def cached_ar_logits_row(**kw):
        import io
        index = int(kw['index'])
        if list(kw['ar_tokens']) != reference_ids:
            raise RuntimeError('diagnostic AR prefix differs from the validated reference')
        key = reference_provenance['receipt_sha256']
        cache_stem = stage_root / f'ar-logits-{key[:24]}-{index}'
        data_path = cache_stem.with_suffix('.npy')
        meta_path = cache_stem.with_suffix('.json')
        expected = {'reference_receipt_sha256': key,
                    'source_commit': REFERENCE_SOURCE_COMMIT, 'index': index,
                    'prompt_ids_sha256': reference['prompt_ids_sha256'],
                    'ar_token_ids_sha256': reference_digest}
        if data_path.exists() or meta_path.exists():
            if not (data_path.is_file() and meta_path.is_file()):
                raise RuntimeError('incomplete AR diagnostic cache entry')
            if data_path.stat().st_size > 1024**2 or meta_path.stat().st_size > 16384:
                raise RuntimeError('AR diagnostic cache exceeds its bounded row size')
            metadata = json.loads(meta_path.read_text())
            if any(metadata.get(k) != v for k, v in expected.items()):
                raise RuntimeError('AR diagnostic cache provenance differs')
            blob = data_path.read_bytes()
            if hashlib.sha256(blob).hexdigest() != metadata.get('npy_sha256'):
                raise RuntimeError('AR diagnostic cache digest differs')
            row = ab.np.load(io.BytesIO(blob), allow_pickle=False)
            reused = True
        else:
            raise RuntimeError('new AR diagnostic index requires a fresh unchanged reference; no replay through candidate arithmetic')
        if row.shape != (129280,) or row.dtype != ab.np.dtype('float32'):
            raise RuntimeError('cached AR diagnostic row shape or dtype differs')
        ar_logits_cache_report.update({**metadata, 'path': str(data_path),
                                       'reused': reused, 'payload_bytes': int(row.nbytes)})
        return row

    ab._ar_logits_row_at_index = cached_ar_logits_row
    original_run_arm = ab._run_arm

    def run_arm_with_reference(*a, **kw):
        receipt = original_run_arm(*a, **kw)
        if receipt['dspark']['token_ids_sha256'] != CONTROL_OUTPUT_SHA256:
            PREFIX.with_suffix('.rejected-output.json').write_text(json.dumps(receipt, indent=2) + '\n')
            raise RuntimeError('compiled prefill changed the full validated MTP token digest')
        receipt['prefill_hc_post_compiled'] = True
        receipt['source_installation'] = COMPATIBILITY
        receipt['ar_reference_reuse'] = reference_provenance
        receipt['measurement_origin'] = {
            'ar': 'external_receipt', 'dspark': 'current_run',
        }
        for key in ('ttft_s', 'prefill_tok_s', 'decode_wall_s', 'decode_tok_s',
                    'peak_gb', 'peak_process_gb', 'memory', 'utilization',
                    'cooldown', 'overlap_telemetry', 'serve_stream_counters',
                    'decode_attn_kernel_engagement', 'attn_core_compile_engagement',
                    'fused_proj_engagement', 'switch_dispatch', 'decode_timeline'):
            receipt[key] = None
        receipt['dspark']['ar_reference_reuse'] = reference_provenance
        receipt['dspark']['memory']['prefill_boundary'] = dict(prefill_boundary_memory)
        receipt['dspark']['memory']['seed_prefill_boundary'] = dict(seed_boundary_memory)
        receipt['dspark']['ar_logits_reference_cache'] = dict(ar_logits_cache_report)
        return receipt

    ab._run_arm = run_arm_with_reference
    ab._generate = reused_ar_reference
    ab._generate_dspark = admitted_mtp
    raise SystemExit(ab.main())
finally:
    stop.set()
    thread.join(2)
    sample()
