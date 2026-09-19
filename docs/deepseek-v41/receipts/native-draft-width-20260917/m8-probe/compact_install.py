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

