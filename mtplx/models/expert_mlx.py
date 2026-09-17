"""MLX execution adapters for slot-backed affine-quantized routed experts."""

from __future__ import annotations

import gc
import hashlib
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.activations import swiglu

from mtplx import expert_route_probe as _route_probe
# W125 (red-team HIGH-1b): the decode timeline probe brackets the blocking gather
# fence here so host_gap can subtract it. No-op unless MTPLX_DSV41_DECODE_TIMELINE=1.
from mtplx import dsv41_decode_timeline as _tl
from mtplx.models import deepseek_v41_stage_timing as _stime

from mtplx.expert_io import PositionalExpertReader
from mtplx.expert_runtime import ExpertStreamingRuntime
from mtplx.expert_manifest import ExpertManifest, ExpertRecord
from mtplx.expert_slots import ExpertSlotBinding, ReadyRoute
from mtplx.expert_streaming import RoutingPhase
from mtplx.expert_streaming_models import (
    MIXED_OFFICIAL_CODEC,
    ExpertMemoryPlan,
    ExpertStreamingModelSpec,
)
from mtplx.mmap_mlx import mmap_u32

# Mixed-official (issue #51, M2) per-tier affine bit widths: gate/up is 2-bit
# affine when the layer's tier is "affine2"; down is always 3-bit affine.
_MIXED_AFFINE_BITS = {"affine2": 2, "affine3": 3}
_MIXED_DOWN_BITS = 3

# W51 / KERNEL_LEDGER K26 -- prefill-only "dequantize once, matmul dense" expert
# path.  The mxfp4 gs32 ``gather_qmm`` is ALU/dequant-bound at the large per-expert
# row counts of the 16 K layer-major prefill (W47: ~2.9 TFLOPS vs a dense bf16
# matmul's ~15-25 TFLOPS on this box; [[metal-sub4bit-alu-bound]]).  When
# ``MTPLX_DSV41_PREFILL_DENSE_EXPERTS`` is armed, a prefill wave dequantizes each
# expert's three matrices from the resident component bank to bf16 ONCE (mx.dequantize
# mode "mxfp4", weight + E8M0 scales, no bias) and runs gate/up/down as dense bf16
# matmuls over that expert's rows, scattering back; experts with fewer than
# ``MTPLX_DSV41_PREFILL_DENSE_MIN_ROWS`` rows keep ``gather_qmm``.  Default OFF; never
# engages at decode (the switch gates it to ``RoutingPhase.PREFILL``, so M=1 is
# byte-identical).  NOT bit-identical to gather_qmm -- fp32 matmul accumulation order
# differs -- within the tolerance the W51 CPU test measures.
_PREFILL_DENSE_ENV = "MTPLX_DSV41_PREFILL_DENSE_EXPERTS"
_PREFILL_DENSE_MIN_ROWS_ENV = "MTPLX_DSV41_PREFILL_DENSE_MIN_ROWS"
_PREFILL_DENSE_BATCH_ENV = "MTPLX_DSV41_PREFILL_DENSE_BATCH"
# W51 window-20 follow-up: the bf16 dense matmul won only −20 s (not the modelled
# −70-80 s). W50 found bf16 score matmuls 34% slower than f32 at 16K on this box, so
# the dense matmul may be paying the same slow bf16 kernel.  This knob runs the dense
# gate/up/down (and the dequant that feeds them) in f32 as an A/B variant; the mxfp4
# dequant is lossless in either dtype (FP4 x 2^E8M0 is exact in bf16 and f32), and
# dequantizing straight to the compute dtype means NO f32->bf16 (or bf16->f32) cast
# and no doubled write.  Default "bf16".
_PREFILL_DENSE_MATMUL_DTYPE_ENV = "MTPLX_DSV41_PREFILL_DENSE_MATMUL_DTYPE"
_PREFILL_DENSE_MIN_ROWS_DEFAULT = 128
_PREFILL_DENSE_BATCH_DEFAULT = 8


def _prefill_dense_experts_enabled() -> bool:
    """Read at use (not import): the server stamps optimization keys after importing
    modules ([[env-flags-read-at-use-not-import]])."""
    return os.environ.get(_PREFILL_DENSE_ENV) == "1"


def _prefill_dense_matmul_dtype() -> "mx.Dtype":
    """The compute dtype for the dense path's dequant + matmuls (W51 window-20 A/B):
    ``f32`` runs the whole expert MLP in float32 (dequant included), else bf16."""
    raw = os.environ.get(_PREFILL_DENSE_MATMUL_DTYPE_ENV, "").strip().lower()
    return mx.float32 if raw in ("f32", "float32", "fp32") else mx.bfloat16


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 1 else default


# W56 / KERNEL_LEDGER K27 -- shape/tiling layout fix.  The routed-expert switch
# calls ``mx.gather_qmm(x[rows,1,1,K], w, s, rhs_indices=slot, transpose=True)`` with
# M==1 and the rows in router (token x top_k) order, i.e. NOT sorted by expert.  In
# mlx 0.32.2 (quantized.cpp GatherQMM::eval_gpu) the fused ``gather_qmm_rhs`` kernel --
# which streams each expert's weight ONCE over its contiguous block of rows -- fires
# ONLY when ``M==1 && B>=16 && right_sorted_ && B/E>=4``, and ``right_sorted_ =
# sorted_indices && lhs_indices is None`` (ops.cpp:5632).  With the flag unset the
# unsorted call takes the per-row ``gather_qmv`` (each output row re-reads its expert's
# full mxfp4 weight -- the W47 ~2.9 TFLOPS memory thrash / 105 s prefill switch).  When
# ``MTPLX_DSV41_LAYOUT_FIX`` is armed AND the wave has at least
# ``MTPLX_DSV41_LAYOUT_FIX_MIN_ROWS`` rows (prefill; decode/verify stay below it, so
# the served M=1 path is the exact shipped call), the gather sorts rows by bank slot on
# device (``mx.argsort``), runs the three projections with ``sorted_indices=True``, and
# unsorts the output.  A permutation + its inverse with an M-independent per-row matmul
# is BYTE-IDENTICAL on CPU (there is one gather_qmm impl); on Metal it swaps
# gather_qmv -> gather_qmm_rhs_nax, a kernel reassociation in the same documented FP
# class as K26 (measured in a GPU window, not bit-identical there).  Default OFF.
_LAYOUT_FIX_ENV = "MTPLX_DSV41_LAYOUT_FIX"
_LAYOUT_FIX_MIN_ROWS_ENV = "MTPLX_DSV41_LAYOUT_FIX_MIN_ROWS"
# ``gather_qmm_rhs`` needs B/E>=4 (E == bank slot count, up to n_routed_experts 384),
# so ~1536 rows minimum; default 2048 keeps decode (6) / verify (24) / small waves on
# the exact shipped unsorted path and only sorts the many-row prefill waves.
_LAYOUT_FIX_MIN_ROWS_DEFAULT = 2048
# Optional per-call row cap for the routed gather (memory bound + microbench sweep).
# 0 (default) = one call over the whole wave.  Only honoured on the sorted path; each
# chunk is a contiguous sub-range of the sorted rows so ``sorted_indices`` stays valid,
# and the per-row math is unchanged (byte-identical to a single call).
_GATHER_ROWS_PER_CALL_ENV = "MTPLX_DSV41_GATHER_ROWS_PER_CALL"


def _layout_fix_enabled() -> bool:
    """Read at use (not import): the server stamps optimization keys after importing
    modules ([[env-flags-read-at-use-not-import]])."""
    return os.environ.get(_LAYOUT_FIX_ENV) == "1"


def _layout_fix_min_rows() -> int:
    return _positive_env_int(_LAYOUT_FIX_MIN_ROWS_ENV, _LAYOUT_FIX_MIN_ROWS_DEFAULT)


def _gather_rows_per_call() -> int:
    raw = os.environ.get(_GATHER_ROWS_PER_CALL_ENV)
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        return 0
    return value if value >= 1 else 0


# W56 / KERNEL_LEDGER K27 (F2) -- expert down-projection K padding.  The expert
# down-proj contracts over K = moe_intermediate_size = 2304, and 2304 % 512 = 256,
# so the mxfp4 `gather_qmv` fast kernel (which needs K % qmv_fast_k_alignment(4)=512)
# is DISABLED for the down gather (quantized.cpp:1337, 147-148) -- gate/up (K=hidden
# 5120, %512==0) hit it.  Padding the down K to the next multiple of 512 (2560) with
# ZERO columns re-enables the fast kernel; a zero column contributes exactly 0, so the
# result is byte-identical (verified: a zeroed mxfp4 gs32 group dequantizes to exactly
# 0.0 for any E8M0 scale byte except 0xFF/NaN, and a `mx.zeros` bank tail uses scale
# byte 0 -> exact).  This requires the down bank slot to be laid out 2560-wide (its
# weight/scales 256 columns / 8 scale groups wider, tail zeroed) AND the down-gather
# activation zero-padded to 2560.  When `MTPLX_DSV41_DOWN_K_PAD` is set,
# `_gather_component_bank` pads the SwiGLU activation to the bank's actual down-K
# (a no-op when the bank is unpadded -> byte-identical; the fast kernel only engages
# once the bank itself is padded).  Default OFF.  See `pad_mxfp4_down_component` for
# the exact layout transform and `down_k_pad_slot_bytes` for the slot arithmetic.
_DOWN_K_PAD_ENV = "MTPLX_DSV41_DOWN_K_PAD"
_DOWN_K_FAST_ALIGN = 512  # mxfp4 (bits 4) qmv_fast_k_alignment == 512


def _down_k_pad_enabled() -> bool:
    """Read at use (not import) ([[env-flags-read-at-use-not-import]])."""
    return os.environ.get(_DOWN_K_PAD_ENV) == "1"


def _down_k_pad_width(k: int, *, align: int = _DOWN_K_FAST_ALIGN) -> int:
    """The down-K rounded up to the next `align` multiple (the fast-qmv width)."""
    k = int(k)
    if k % align == 0:
        return k
    return ((k + align - 1) // align) * align


def pad_mxfp4_down_component(
    weight: mx.array,
    scales: mx.array,
    *,
    group_size: int = 32,
    bits: int = 4,
    align: int = _DOWN_K_FAST_ALIGN,
) -> tuple[mx.array, mx.array]:
    """Lay one expert's mxfp4 down-proj ``[out, K]`` out with a zero K-tail so the
    stored K is a multiple of ``align`` (the fast-qmv width).

    ``weight`` is the packed uint32 ``[out, K/8]`` FP4-code tensor, ``scales`` the
    E8M0 uint8 ``[out, K/group_size]`` exponents (no bias -- mxfp4).  The returned
    tensors have the real data in the leading columns and **zero** packed nibbles +
    **zero** E8M0 scale bytes in the tail.  A zeroed FP4 group dequantizes to
    EXACTLY 0.0 for scale byte 0 (2^-127 * 0 == 0; only 0xFF/NaN is unsafe), so a
    ``gather_qmm`` over the padded weight with a zero-padded activation is
    byte-identical to the unpadded gather.  Used by the admission/offline bank
    builder and the F2 byte-identity tests."""
    packs_per_word = 32 // int(bits)  # mxfp4 -> 8 nibbles / uint32
    k_packed = int(weight.shape[-1])
    k_logical = k_packed * packs_per_word
    k_pad = _down_k_pad_width(k_logical, align=align)
    if k_pad == k_logical:
        return weight, scales
    w_pad_cols = (k_pad - k_logical) // packs_per_word     # extra uint32 columns
    s_pad_cols = (k_pad - k_logical) // int(group_size)    # extra E8M0 scale bytes
    weight_out = mx.pad(weight, [(0, 0)] * (weight.ndim - 1) + [(0, w_pad_cols)])
    scales_out = mx.pad(scales, [(0, 0)] * (scales.ndim - 1) + [(0, s_pad_cols)])
    return weight_out, scales_out


def down_k_pad_slot_bytes(
    *,
    hidden: int,
    inter: int,
    bits: int = 4,
    group_size: int = 32,
    align: int = _DOWN_K_FAST_ALIGN,
) -> dict[str, int]:
    """Byte accounting for padding one expert's mxfp4 down-proj K to ``align``.

    Down-proj is ``[out=hidden, K=inter]``: packed weight ``hidden * K / (32/bits) * 4``
    bytes + E8M0 scales ``hidden * K / group_size`` bytes (no bias).  Returns the
    unpadded / padded down-component bytes and the delta so the memory planner can
    price the slot growth."""
    k_pad = _down_k_pad_width(inter, align=align)
    def _bytes(k: int) -> int:
        w = hidden * (k // (32 // bits)) * 4   # packed uint32 bytes
        s = hidden * (k // group_size)         # 1 E8M0 byte per group
        return w + s
    unpadded = _bytes(inter)
    padded = _bytes(k_pad)
    return {
        "k_real": int(inter),
        "k_pad": int(k_pad),
        "down_unpadded_bytes": int(unpadded),
        "down_padded_bytes": int(padded),
        "down_delta_bytes": int(padded - unpadded),
    }


_ROUTING_PHASE: ContextVar[RoutingPhase | None] = ContextVar(
    "mtplx_expert_routing_phase",
    default=None,
)
_SLOT_INDEX_PATTERN = r"(?:0|[1-9][0-9]*)"
_LAYER_PERSISTENT_LABEL = re.compile(
    rf"layer-({_SLOT_INDEX_PATTERN})-persistent-({_SLOT_INDEX_PATTERN})"
)
_GLOBAL_PERSISTENT_LABEL = re.compile(rf"global-persistent-({_SLOT_INDEX_PATTERN})")
_GLOBAL_TRANSIENT_LABEL = re.compile(rf"global-transient-({_SLOT_INDEX_PATTERN})")
# Mixed-official (issue #51 M2b): one transient service bank per gate/up tier so
# partial-residency misses on either tier class land in a matching geometry.
_MIXED_TRANSIENT_LABEL = re.compile(
    rf"mixed-transient-([a-z0-9]+)-({_SLOT_INDEX_PATTERN})"
)
_LAYER_PREFETCH_LABEL = re.compile(
    rf"layer-({_SLOT_INDEX_PATTERN})-prefetch-({_SLOT_INDEX_PATTERN})"
)
# W93: the prefetch ring is now a single SHARED tier across all layers (like the
# global transient pool), so its slots carry a layer-less label.
_GLOBAL_PREFETCH_LABEL = re.compile(rf"global-prefetch-({_SLOT_INDEX_PATTERN})")


@contextmanager
def expert_routing_phase(phase: RoutingPhase | str) -> Iterator[None]:
    token = _ROUTING_PHASE.set(RoutingPhase(phase))
    try:
        yield
    finally:
        _ROUTING_PHASE.reset(token)


def current_expert_routing_phase(*, token_count: int) -> RoutingPhase:
    explicit = _ROUTING_PHASE.get()
    if explicit is not None:
        return explicit
    return RoutingPhase.PREFILL if token_count > 1 else RoutingPhase.DECODE


def _mark_pipeline_incomplete(ledger: Any, phase: RoutingPhase) -> None:
    try:
        ledger.mark_incomplete(phase=phase)
    except Exception:
        pass


def _begin_pipeline_work(
    ledger: Any,
    method: str,
    *args: Any,
    phase: RoutingPhase,
) -> Any | None:
    """Open optional diagnostics without changing model execution outcomes."""

    try:
        return getattr(ledger, method)(*args, phase=phase)
    except Exception:
        _mark_pipeline_incomplete(ledger, phase)
        return None


def _pipeline_work_call(
    ledger: Any,
    target: Any,
    method: str,
    *args: Any,
    phase: RoutingPhase,
) -> None:
    """Publish one optional work transition while preserving data-path errors."""

    try:
        getattr(target, method)(*args)
    except Exception:
        _mark_pipeline_incomplete(ledger, phase)


class _DeferredSplitClose:
    """Replay a split route's release/close sequence at deferred-flush time.

    The split-route lease bookkeeping (consumer leases, finalize, the layer
    lock) must run through release_miss/close — releasing raw routes would
    corrupt the state machine. This adapter duck-types the ``release``
    surface ``defer_slot_release`` expects and performs today's exact call
    order, one covering eval later.
    """

    def __init__(self, pending: Any, parts: tuple[Any, ...]) -> None:
        self._pending = pending
        self._parts = parts

    def release(self, *, synchronize: bool = False) -> None:
        del synchronize
        first_error: BaseException | None = None
        for part in self._parts:
            try:
                self._pending.release_miss(part)
            except BaseException as exc:  # noqa: BLE001 - drain all parts
                if first_error is None:
                    first_error = exc
        try:
            self._pending.close()
        except BaseException as exc:  # noqa: BLE001 - propagate after close
            if first_error is None:
                first_error = exc
        if first_error is not None:
            raise first_error


class UnboundExpertSwitch(nn.Module):
    """Parameter-free placeholder installed before resident-only loading."""

    def __init__(self, layer_index: int):
        super().__init__()
        self.layer_index = int(layer_index)

    def __call__(self, _x: mx.array, _indices: mx.array) -> mx.array:
        raise RuntimeError(
            f"streamed expert layer {self.layer_index} has no bound runtime"
        )


def _component_array(binding: ExpertSlotBinding, component: str) -> mx.array:
    segment = None
    offset = 0
    for candidate in binding.record.segments:
        if candidate.component == component:
            segment = candidate
            break
        offset += candidate.length
    if segment is None:
        raise KeyError(component)
    if isinstance(binding.buffer, mx.array):
        raw = binding.buffer[offset : offset + segment.length]
        if segment.dtype == "U32":
            return raw.view(mx.uint32).reshape(segment.shape)
        if segment.dtype == "BF16":
            return raw.view(mx.bfloat16).reshape(segment.shape)
        if segment.dtype == "U16":
            return raw.view(mx.uint16).reshape(segment.shape)
        if segment.dtype == "U8":
            return raw.reshape(segment.shape)
        raise TypeError(f"unsupported streamed component dtype {segment.dtype}")
    view = binding.component_view(component)
    if segment.dtype == "U32":
        host = np.frombuffer(view, dtype=np.dtype("<u4")).reshape(segment.shape)
        value = mx.array(host)
    elif segment.dtype == "BF16":
        host = np.frombuffer(view, dtype=np.dtype("<u2")).reshape(segment.shape)
        value = mx.array(host).view(mx.bfloat16)
    elif segment.dtype == "U16":
        host = np.frombuffer(view, dtype=np.dtype("<u2")).reshape(segment.shape)
        value = mx.array(host)
    elif segment.dtype == "U8":
        host = np.frombuffer(view, dtype=np.dtype("u1")).reshape(segment.shape)
        value = mx.array(host)
    else:
        raise TypeError(f"unsupported streamed component dtype {segment.dtype}")
    return value


def mlx_slot_buffer_allocator(size: int, _label: str) -> mx.array:
    """Allocate one stable writable MLX/Metal byte buffer for direct ``pread``."""

    value = mx.zeros((int(size),), dtype=mx.uint8)
    mx.eval(value)
    view = memoryview(value)
    if view.readonly or not view.c_contiguous or view.nbytes != int(size):
        raise RuntimeError("MLX slot buffer is not writable contiguous shared memory")
    view.release()
    return value


def make_mlx_slot_buffer_allocator(
    plan: ExpertMemoryPlan,
    spec: ExpertStreamingModelSpec,
) -> Callable[[int, str], mx.array]:
    """Create stable direct MLX/Metal buffers without materialized bank slices.

    MLX integer indexing does not expose a writable view: evaluating
    ``bank[slot]`` allocates a second buffer.  Keeping both the bank and all
    evaluated slices therefore doubled the expert-cache allocation.  Direct
    fixed slots preserve positional-I/O and generation semantics while making
    physical allocation match the memory plan.
    """

    slots: dict[str, mx.array] = {}
    backend = "mlx-metal-direct-slots"

    def allocate(size: int, label: str) -> mx.array:
        if size != spec.expert_record_bytes:
            raise ValueError("slot allocator size differs from the model descriptor")
        parts = label.split("-")
        if label.startswith("layer-") and "-persistent-" in label:
            layer = int(parts[1])
            slot = int(parts[-1])
            count = plan.slots_for_layer(layer)
            if layer not in spec.routed_layer_indices:
                raise ValueError(f"persistent slot layer {layer} is not routed")
        elif label.startswith("layer-") and "-prefetch-" in label:
            layer = int(parts[1])
            slot = int(parts[-1])
            count = plan.prefetch_ring_slots
            if layer not in spec.routed_layer_indices:
                raise ValueError(f"prefetch slot layer {layer} is not routed")
        elif label.startswith("global-persistent-"):
            slot = int(parts[-1])
            count = plan.persistent_slots
        elif label.startswith("global-transient-"):
            slot = int(parts[-1])
            count = plan.transient_slots
        elif label.startswith("global-prefetch-"):
            # W93: shared prefetch ring (one tier across all layers).
            slot = int(parts[-1])
            count = plan.prefetch_ring_slots
        else:
            raise ValueError(f"unknown expert slot label {label!r}")
        if count <= 0:
            raise ValueError(f"slot {label} has no planned capacity")
        if not 0 <= slot < count:
            raise ValueError(f"slot {label} is outside planned capacity {count}")
        if label in slots:
            raise ValueError(f"slot {label} was allocated twice")
        value = mlx_slot_buffer_allocator(size, label)
        slots[label] = value
        return value

    setattr(allocate, "backend", backend)
    setattr(allocate, "slots", slots)
    return allocate


class MlxComponentBank:
    """Component-major writable MLX storage for a fixed expert-slot tier."""

    def __init__(
        self,
        *,
        capacity: int,
        record: ExpertRecord,
        label: str,
    ) -> None:
        self.capacity = int(capacity)
        self.label = str(label)
        self.record_bytes = int(record.logical_bytes)
        self.arrays: dict[str, mx.array] = {}
        self._views: dict[str, memoryview] = {}
        self._segment_bytes: dict[str, int] = {}
        if self.capacity <= 0:
            raise ValueError("component bank capacity must be positive")
        try:
            for segment in record.segments:
                if segment.component in self.arrays:
                    raise ValueError(
                        f"duplicate component {segment.component!r} in expert record"
                    )
                dtype = {
                    "U32": mx.uint32,
                    "BF16": mx.bfloat16,
                    "U16": mx.uint16,
                    "U8": mx.uint8,
                }.get(segment.dtype)
                if dtype is None:
                    raise TypeError(f"unsupported component-bank dtype {segment.dtype}")
                value = mx.zeros((self.capacity, *segment.shape), dtype=dtype)
                mx.eval(value)
                view = memoryview(value)
                if view.readonly or not view.c_contiguous:
                    raise RuntimeError(
                        f"component bank {label}/{segment.component} is not writable"
                    )
                raw = view.cast("B")
                expected = self.capacity * segment.length
                if raw.nbytes != expected:
                    raise RuntimeError(
                        f"component bank {label}/{segment.component} has "
                        f"{raw.nbytes} bytes; expected {expected}"
                    )
                self.arrays[segment.component] = value
                self._views[segment.component] = raw
                self._segment_bytes[segment.component] = segment.length
        except Exception:
            for view in self._views.values():
                view.release()
            self._views.clear()
            self.arrays.clear()
            raise

    def component_view(self, slot: int, component: str) -> memoryview:
        if not 0 <= int(slot) < self.capacity:
            raise IndexError("component-bank slot is outside capacity")
        length = self._segment_bytes[component]
        start = int(slot) * length
        return self._views[component][start : start + length]

    def close(self) -> None:
        for view in self._views.values():
            try:
                view.release()
            except Exception:
                pass
        self._views.clear()
        self.arrays.clear()
        self._segment_bytes.clear()


class MlxComponentSlot:
    """One logical slot backed by a row in nine component-major MLX arrays."""

    def __init__(
        self,
        bank: MlxComponentBank,
        bank_index: int,
        *,
        label: str,
    ) -> None:
        self.bank = bank
        self.bank_index = int(bank_index)
        self.label = str(label)
        self.nbytes = bank.record_bytes

    def record_views(self, record: ExpertRecord) -> tuple[memoryview, ...]:
        if int(record.logical_bytes) != self.nbytes:
            raise ValueError("record size differs from component-bank slot")
        return tuple(
            self.bank.component_view(self.bank_index, segment.component)
            for segment in record.segments
        )

    def component_view(self, component: str) -> memoryview:
        return self.bank.component_view(self.bank_index, component)


class MappedExpertRecord:
    """One expert record backed directly by its sidecar file pages."""

    def __init__(self, record: ExpertRecord, base: mx.array) -> None:
        self.record = record
        self.base = base
        self._arrays: dict[str, mx.array] | None = None

    @property
    def arrays(self) -> dict[str, mx.array]:
        arrays = self._arrays
        if arrays is not None:
            return arrays
        arrays = {}
        cursor = 0
        for segment in self.record.segments:
            if segment.dtype == "U32":
                typed = self.base
                item_size = 4
            elif segment.dtype == "BF16":
                typed = mx.view(self.base, mx.bfloat16)
                item_size = 2
            elif segment.dtype == "U8":
                # mxfp4 E8M0 scales (one byte per group); base is already uint8.
                typed = self.base
                item_size = 1
            else:
                raise TypeError(f"unsupported mapped component dtype {segment.dtype}")
            if cursor % item_size:
                raise ValueError(f"component {segment.component} is not dtype-aligned")
            arrays[segment.component] = mx.as_strided(
                typed,
                shape=segment.shape,
                offset=cursor // item_size,
            )
            cursor += segment.length
        if cursor != self.record.logical_bytes:
            raise ValueError("mapped component layout does not cover the record")
        self._arrays = arrays
        return arrays


class MappedExpertStore:
    """Virtual-map every sidecar record without adding it to MLX residency.

    The MTLBuffers remain addressable for the life of the model, but their
    pages are not in MLX's process-wide wired residency set. Metal binds only
    the routed record buffers for a QMM command; macOS can retain or evict the
    corresponding file pages through its normal page cache.
    """

    def __init__(
        self,
        root: Path | str,
        manifest: ExpertManifest,
        *,
        workers: int = 96,
    ) -> None:
        if manifest.sidecar is None:
            raise ValueError("metal-mmap execution requires a sidecar manifest")
        self.root = Path(root).resolve()
        self.sidecar = manifest.sidecar
        # One mapped path per bank part; ``path`` stays meaningful for the
        # single-part banks every existing artifact uses.
        self.paths = tuple(self.root / part.file for part in manifest.sidecar.parts)
        self.path = self.paths[0]
        self.records = tuple(manifest.records)
        self.workers = max(1, min(int(workers), 256))
        self._mapped: dict[tuple[int, int], MappedExpertRecord] = {}
        self._lock = threading.Lock()
        self._mapping_seconds = 0.0
        self._qmm_experts = 0
        self._closed = False
        expected = {(record.layer, record.expert) for record in self.records}
        if len(expected) != len(self.records):
            raise ValueError("sidecar contains duplicate layer/expert records")
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        for record in self.records:
            if record.sidecar_offset is None or record.sidecar_length is None:
                raise ValueError("metal-mmap record has no sidecar range")
            if record.sidecar_length != record.logical_bytes:
                raise ValueError("metal-mmap sidecar length differs from record")
            if not 0 <= record.part < len(self.paths):
                raise ValueError("metal-mmap record names a missing sidecar part")
            part = manifest.sidecar.part(record.part)
            if part.data_start % page_size:
                raise ValueError("metal-mmap sidecar parts must start on a page")
            if record.sidecar_offset % page_size or record.sidecar_length % page_size:
                raise ValueError("metal-mmap sidecar records must be page aligned")

    def prepare(self) -> None:
        if self._closed:
            raise RuntimeError("mapped expert store is closed")
        if len(self._mapped) == len(self.records):
            return
        started = time.perf_counter()

        def map_record(
            record: ExpertRecord,
        ) -> tuple[tuple[int, int], MappedExpertRecord]:
            assert record.sidecar_offset is not None
            assert record.sidecar_length is not None
            part = self.sidecar.part(record.part)
            base = mmap_u32(
                self.paths[record.part],
                part.data_start + record.sidecar_offset,
                record.sidecar_length,
                wired=False,
            )
            return (record.layer, record.expert), MappedExpertRecord(record, base)

        with ThreadPoolExecutor(
            max_workers=self.workers,
            thread_name_prefix="mtplx-mmap",
        ) as executor:
            mapped = dict(executor.map(map_record, self.records))
        with self._lock:
            self._mapped = mapped
            self._mapping_seconds += time.perf_counter() - started

    def get(self, layer: int, expert: int) -> MappedExpertRecord:
        try:
            return self._mapped[(int(layer), int(expert))]
        except KeyError as exc:
            raise KeyError(f"mapped expert ({layer}, {expert}) is unavailable") from exc

    def observe_qmm(self, count: int) -> None:
        with self._lock:
            self._qmm_experts += int(count)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "backend": "metal-mmap-unwired-records",
                "record_count": len(self.records),
                "mapped_records": len(self._mapped),
                "virtual_bytes": sum(
                    int(record.logical_bytes) for record in self.records
                ),
                "mapping_seconds": self._mapping_seconds,
                "workers": self.workers,
                "qmm_experts": self._qmm_experts,
                "globally_wired_bytes": 0,
            }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._mapped.clear()
        # Evaluated MLX arrays can retain graph-input cycles until cyclic GC.
        # Collect now so every external MTLBuffer releases before its mmap.
        gc.collect()


class DenseIslandStore:
    """Capacity-guaranteed dense per-layer expert banks (issue #63, C5).

    An island layer holds all of its experts resident for the model's
    lifetime in one component-major bank whose row index IS the expert id.
    Router indices therefore address ``gather_qmm`` directly: no
    expert-to-slot translation, no residency probe, no pins, no fences.
    A miss is impossible by construction, so the streamed route machinery
    never runs for these layers.
    """

    def __init__(
        self,
        manifest: ExpertManifest,
        layers: Iterable[int],
        *,
        expert_count: int,
    ) -> None:
        self.layers = tuple(sorted({int(layer) for layer in layers}))
        self.expert_count = int(expert_count)
        if not self.layers:
            raise ValueError("dense island store requires at least one layer")
        records = {
            (record.layer, record.expert): record for record in manifest.records
        }
        self._records: dict[int, tuple[ExpertRecord, ...]] = {}
        self._banks: dict[int, MlxComponentBank] = {}
        self._fill_seconds = 0.0
        self._filled_layers: set[int] = set()
        self._closed = False
        try:
            for layer in self.layers:
                layer_records = []
                for expert in range(self.expert_count):
                    record = records.get((layer, expert))
                    if record is None:
                        raise ValueError(
                            f"manifest has no record for island layer {layer} "
                            f"expert {expert}"
                        )
                    layer_records.append(record)
                self._records[layer] = tuple(layer_records)
                self._banks[layer] = MlxComponentBank(
                    capacity=self.expert_count,
                    record=layer_records[0],
                    label=f"island-layer-{layer}",
                )
        except Exception:
            self.close()
            raise

    @property
    def island_bytes(self) -> int:
        return sum(
            bank.capacity * bank.record_bytes for bank in self._banks.values()
        )

    def fill(
        self,
        manifest: ExpertManifest,
        reader: PositionalExpertReader,
        *,
        verify_hash: bool = True,
    ) -> None:
        """Bulk-read every island expert into its bank row (one-time cost)."""

        if self._closed:
            raise RuntimeError("dense island store is closed")
        started = time.perf_counter()
        for layer in self.layers:
            if layer in self._filled_layers:
                continue
            bank = self._banks[layer]
            items = tuple(
                (
                    record,
                    MlxComponentSlot(
                        bank,
                        expert,
                        label=f"island-layer-{layer}-expert-{expert}",
                    ),
                )
                for expert, record in enumerate(self._records[layer])
            )
            reader.read_component_records_into(
                manifest,
                items,
                verify_hash=verify_hash,
            )
            self._filled_layers.add(layer)
        self._fill_seconds += time.perf_counter() - started

    def bank_for_layer(self, layer: int) -> MlxComponentBank:
        if self._closed:
            raise RuntimeError("dense island store is closed")
        if layer not in self._filled_layers:
            raise RuntimeError(f"island layer {layer} has not been filled")
        return self._banks[layer]

    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": "dense-island-banks",
            "layers": list(self.layers),
            "expert_count": self.expert_count,
            "island_bytes": self.island_bytes,
            "filled_layers": len(self._filled_layers),
            "fill_seconds": self._fill_seconds,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for bank in self._banks.values():
            bank.close()
        self._banks.clear()
        self._records.clear()
        self._filled_layers.clear()


class DenseIslandSwitchGLU(nn.Module):
    """Expert dispatch for a dense island layer: raw indices, zero host asks.

    The layer's bank row index equals the expert id, so the router's device
    indices are the ``gather_qmm`` rhs_indices as-is. No host sync, no route
    planning, no pin lifecycle: the wave stays inside the lazy graph and
    materializes with the next streamed layer's blocking sync (or the
    row-end fence).
    """

    def __init__(
        self,
        runtime: ExpertStreamingRuntime,
        store: DenseIslandStore,
        layer_index: int,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.layer_index = int(layer_index)
        self.group_size = runtime.spec.quant_group_size
        self.bits = runtime.spec.quant_bits
        self.swiglu_limit = getattr(runtime.spec, "swiglu_limit", None)
        self.codec = getattr(runtime.spec, "expert_codec", "affine")
        self._bank = store.bank_for_layer(self.layer_index)
        # Lazily-built compiled expert gather (issue #51, 70 tps full-residency
        # goal). Full residency rebuilds the 79-layer graph in Python every
        # decode step; compiling the per-layer gather traces it once and
        # replays. The island bank is fully resident and never mutated, so it
        # is safe to capture in the compiled closure. A plain callable is not
        # registered as an nn.Module child.
        # NUMERICS: NOT bitwise vs eager. fp32 matmul compiles exactly, but
        # mx.compile selects a different fused kernel for the quantized
        # gather_qmm that diverges ~0.1-0.6% — non-associative FP, the same
        # class as the vk_k split-K divergence (#171), tagged as a documented
        # FP issue per David's 2026-07-18 ruling. Whether it holds token-sha
        # end-to-end is a guarded-A/B question; if it flips tokens it is a
        # divergent secondary line, not a bug. Default OFF.
        self._compiled_gather = None

    def wave_call(
        self,
        x: mx.array,
        indices: mx.array,
        scores: mx.array,
    ) -> mx.array | None:
        """Fused K3 expert wave over the island bank (issue #65).

        Owns the routing multiply and reduction in the wave kernel's BF16
        combine mode, which bitwise-matches the block's external combine.
        Returns None for any shape the fixed-M4 wave does not cover; the
        caller then runs the classic dispatch unchanged.
        """

        # The fused M4 wave computes SwiGLU inside the Metal kernel with no
        # clamp, so a spec that clamps (spec.swiglu_limit set) must not take it:
        # return None to fall back to the eager clamped __call__ dispatch. No
        # streamed model both clamps and runs the hy3 wave today (the wave is
        # hy3-only, hy3 leaves swiglu_limit None), so this only guards the
        # contract; adding the clamp to the kernel is a future Metal change.
        if self.swiglu_limit is not None and self.swiglu_limit > 0:
            return None

        from mtplx.hy3_expert_wave_m4 import (
            HY3_M4_BATCH,
            HY3_M4_ROWS,
            HY3_M4_TOP_K,
            Hy3M4ExpertWaveIneligible,
            hy3_q2_m4_expert_wave,
        )

        spec = self.runtime.spec
        shape = tuple(int(dim) for dim in x.shape)
        if shape != (HY3_M4_BATCH, HY3_M4_ROWS, spec.hidden_size):
            return None
        if tuple(int(dim) for dim in indices.shape) != (
            HY3_M4_BATCH,
            HY3_M4_ROWS,
            HY3_M4_TOP_K,
        ):
            return None
        phase = current_expert_routing_phase(token_count=int(x.shape[-2]))
        try:
            output = hy3_q2_m4_expert_wave(
                x,
                indices.astype(mx.int32),
                scores,
                self._bank.arrays,
                validated_slot_bounds=(0, spec.expert_count - 1),
                combine_mode="bf16",
                hidden_size=spec.hidden_size,
                intermediate_size=spec.expert_hidden_size,
                group_size=self.group_size,
                bits=self.bits,
            )
        except Hy3M4ExpertWaveIneligible:
            return None
        _route_probe.count(
            f"hot.island.wave.{phase.name.lower()}.layer{self.layer_index:02d}"
        )
        return output.hidden_rows

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        if indices.ndim < 1:
            raise ValueError("expert indices must include a top-k dimension")
        if int(indices.shape[-1]) != self.runtime.spec.top_k:
            raise ValueError(
                f"router selected {indices.shape[-1]} experts; expected "
                f"{self.runtime.spec.top_k}"
            )
        hidden_size = int(x.shape[-1])
        if hidden_size != self.runtime.spec.hidden_size:
            raise ValueError(
                f"expert input width {hidden_size} does not match "
                f"{self.runtime.spec.hidden_size}"
            )
        tokens = x.reshape(-1, hidden_size)
        top_k = int(indices.shape[-1])
        rows = int(tokens.shape[0])
        phase = current_expert_routing_phase(token_count=int(x.shape[-2]))
        _route_probe.count(
            f"hot.island.{phase.name.lower()}.layer{self.layer_index:02d}"
        )
        assignment_inputs = mx.broadcast_to(
            tokens[:, None, :],
            (rows, top_k, hidden_size),
        ).reshape(-1, hidden_size)
        slot_indices = indices.reshape((-1, 1)).astype(mx.int32)
        if os.environ.get("MTPLX_HY3_COMPILE_ISLAND") == "1":
            output = self._compiled_island_gather()(assignment_inputs, slot_indices)
        else:
            output = _gather_component_bank(
                assignment_inputs,
                self._bank,
                slot_indices,
                group_size=self.group_size,
                bits=self.bits,
                swiglu_limit=self.swiglu_limit,
                codec=self.codec,
            )
        return output.reshape((*indices.shape, hidden_size))

    def _compiled_island_gather(self):
        """Build-once compiled expert gather closing over this layer's bank."""
        if self._compiled_gather is None:
            bank = self._bank
            group_size = self.group_size
            bits = self.bits
            swiglu_limit = self.swiglu_limit

            def gather(assignment_inputs: mx.array, slot_indices: mx.array) -> mx.array:
                return _gather_component_bank(
                    assignment_inputs,
                    bank,
                    slot_indices,
                    group_size=group_size,
                    bits=bits,
                    swiglu_limit=swiglu_limit,
                )

            self._compiled_gather = mx.compile(gather)
        return self._compiled_gather


class BankedMmapBank:
    """Duck-typed component bank whose mapped row index is the expert id."""

    __slots__ = ("arrays",)

    def __init__(self, arrays: dict[str, mx.array]) -> None:
        self.arrays = arrays


class BankedMmapIslandStore:
    """Dense island banks served from a mapped banked sidecar (issue #51, C6).

    Same dispatch contract as ``DenseIslandStore`` — bank row == expert id,
    raw router indices feed ``gather_qmm`` — but physical residency belongs
    to the macOS pager: the banked file regions are mapped into Metal
    without copies, pages arrive on first touch, and the page cache keeps
    or evicts them under normal kernel policy. No slot pool, no reads, no
    misses exist for these layers. ``prepare`` verifies each region hash by
    reading the file once, which doubles as the page-cache warmup.
    """

    def __init__(
        self,
        banked_manifest: Path | str,
        layers: Iterable[int],
        *,
        expert_count: int,
        verify_hash: bool = True,
        wired: bool = True,
    ) -> None:
        from mtplx.expert_banked import BankedManifestError, load_banked_manifest

        self.wired = bool(wired)
        self.layers = tuple(sorted({int(layer) for layer in layers}))
        if not self.layers:
            raise ValueError("banked island store requires at least one layer")
        self._banked = load_banked_manifest(banked_manifest)
        # ``none`` maps raw bank bytes into Metal zero-copy; ``rans32x-v1``
        # reads the compressed region and rebuilds each bank with the in-kernel
        # rANS decoder in ``prepare`` (issue #51, C7). Any other codec has no
        # decode path here.
        if self._banked.codec not in ("none", "rans32x-v1"):
            raise BankedManifestError(
                f"banked codec {self._banked.codec!r} has no island decode path"
            )
        self._codec = self._banked.codec
        if self._banked.expert_count != int(expert_count):
            raise BankedManifestError(
                f"banked manifest holds {self._banked.expert_count} experts "
                f"per layer; the model routes {expert_count}"
            )
        missing = [
            layer for layer in self.layers if layer not in self._banked.layer_set
        ]
        if missing:
            raise BankedManifestError(
                f"banked manifest does not cover layers {missing}"
            )
        self.expert_count = int(expert_count)
        self._verify_hash = bool(verify_hash)
        self._banks: dict[int, BankedMmapBank] = {}
        self._bases: list[mx.array] = []
        self._verify_seconds = 0.0
        self._closed = False

    def prepare(self) -> None:
        from mtplx.expert_banked import BankedManifestError

        if self._closed:
            raise RuntimeError("banked island store is closed")
        if self._banks:
            return
        path = self._banked.bin_path()
        alignment = self._banked.alignment
        file_size = path.stat().st_size
        for layer in self.layers:
            entry = self._banked.layer_entry(layer)
            if self._verify_hash:
                started = time.perf_counter()
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    handle.seek(entry.offset)
                    remaining = entry.length
                    while remaining:
                        chunk = handle.read(min(remaining, 8 << 20))
                        if not chunk:
                            raise BankedManifestError(
                                f"banked layer {layer} region is truncated"
                            )
                        digest.update(chunk)
                        remaining -= len(chunk)
                if digest.hexdigest() != entry.sha256:
                    raise BankedManifestError(
                        f"banked layer {layer} region hash mismatch"
                    )
                self._verify_seconds += time.perf_counter() - started
            if self._codec != "none":
                self._banks[layer] = BankedMmapBank(
                    self._decode_layer_banks(path, entry)
                )
                continue
            mapped_length = -(-entry.length // alignment) * alignment
            if entry.offset + mapped_length > file_size:
                raise BankedManifestError(
                    f"banked layer {layer} extent exceeds {path.name}"
                )
            # wired=True registers the mapping in MLX's process-wide residency
            # set: Metal keeps it permanently resident like ordinary weights.
            # Untracked (unwired) buffers make Metal rebuild residency around
            # every submission — measured as the ENTIRE MLX residency set
            # wiring/unwiring (~85 GiB swings) per wave, ~4x decode slowdown.
            # Unwired remains selectable for bands larger than RAM (Q4/GLM).
            base = mmap_u32(path, entry.offset, mapped_length, wired=self.wired)
            arrays: dict[str, mx.array] = {}
            for component in entry.components:
                if component.dtype == "U32":
                    typed = base
                    item_size = 4
                elif component.dtype == "BF16":
                    typed = mx.view(base, mx.bfloat16)
                    item_size = 2
                else:
                    raise TypeError(
                        f"unsupported banked component dtype {component.dtype}"
                    )
                if component.offset % item_size:
                    raise BankedManifestError(
                        f"banked component {component.component} is not "
                        "dtype-aligned"
                    )
                arrays[component.component] = mx.as_strided(
                    typed,
                    shape=(self.expert_count, *component.shape),
                    offset=component.offset // item_size,
                )
            self._banks[layer] = BankedMmapBank(arrays)
            self._bases.append(base)

    def _decode_layer_banks(self, path, entry) -> dict[str, mx.array]:
        """Rebuild one layer's component banks from its compressed region.

        Reads the entropy-coded region, runs the in-kernel rANS decoder per
        component, and materializes each bank as ``[expert_count, *shape]`` in
        its native dtype — row index == expert id, identical to the mmap path.
        Decode happens once at prepare; the banks are then plain resident
        arrays consumed by ``gather_qmm`` with no per-call host work.
        """

        from mtplx.expert_banked import BankedManifestError
        from mtplx.expert_rans_metal import decode_container

        with path.open("rb") as handle:
            handle.seek(entry.offset)
            region = handle.read(entry.length)
        if len(region) != entry.length:
            raise BankedManifestError(
                f"banked layer {entry.layer} region is truncated"
            )
        arrays: dict[str, mx.array] = {}
        for component in entry.components:
            blob = region[component.offset : component.offset + component.length]
            decoded = decode_container(blob)  # uint8[expert_count * seg_len]
            if component.dtype == "U32":
                typed = mx.view(decoded, mx.uint32)
            elif component.dtype == "BF16":
                typed = mx.view(decoded, mx.bfloat16)
            else:
                raise TypeError(
                    f"unsupported banked component dtype {component.dtype}"
                )
            bank = typed.reshape(self.expert_count, *component.shape)
            mx.eval(bank)  # materialize; releases the compressed payload graph
            arrays[component.component] = bank
        return arrays

    def prefetch_all(self) -> int:
        """Issue one MADV_WILLNEED batch across every mapped layer region."""

        from mtplx.mmap_mlx import prefetch

        if not self._bases:
            return 0
        return prefetch(list(self._bases))

    def bank_for_layer(self, layer: int) -> BankedMmapBank:
        if self._closed:
            raise RuntimeError("banked island store is closed")
        bank = self._banks.get(layer)
        if bank is None:
            raise RuntimeError(f"banked island layer {layer} is not prepared")
        return bank

    def snapshot(self) -> dict[str, Any]:
        stored_bytes = sum(
            self._banked.layer_entry(layer).length for layer in self.layers
        )
        resident_bytes = sum(
            component.raw_bytes
            for layer in self.layers
            for component in self._banked.layer_entry(layer).components
        )
        return {
            "backend": (
                "banked-mmap-island-banks"
                if self._codec == "none"
                else "banked-rans-island-banks"
            ),
            "codec": self._codec,
            "wired": self.wired,
            "layers": list(self.layers),
            "expert_count": self.expert_count,
            "mapped_bytes": stored_bytes,
            "resident_bytes": resident_bytes,
            "verify_seconds": self._verify_seconds,
            "prepared_layers": len(self._banks),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._banks.clear()
        self._bases.clear()
        # Evaluated MLX arrays can retain graph-input cycles until cyclic GC.
        # Collect now so every external MTLBuffer releases before its mmap.
        gc.collect()


def make_mlx_component_bank_allocator(
    plan: ExpertMemoryPlan,
    spec: ExpertStreamingModelSpec,
    manifest: ExpertManifest,
) -> Callable[[int, str], MlxComponentSlot]:
    """Allocate slot bytes as component-major banks usable by ``gather_qmm``.

    Unlike a record-major byte bank, these arrays are both directly writable
    through unified-memory views and directly consumable by MLX grouped QMM
    kernels. No persistent slice or stacked weight copy is materialized.
    """

    record_by_key: dict[tuple[int, int], ExpertRecord] = {}
    duplicate_keys: set[tuple[int, int]] = set()
    record_by_layer: dict[int, ExpertRecord] = {}
    for record in manifest.records:
        key = (record.layer, record.expert)
        if key in record_by_key:
            duplicate_keys.add(key)
        else:
            record_by_key[key] = record
        record_by_layer.setdefault(record.layer, record)
    missing = set(spec.routed_layer_indices) - set(record_by_layer)
    if missing:
        raise ValueError(
            f"manifest has no exemplar records for layers {sorted(missing)}"
        )
    if plan.cache_scope == "global":
        expected_keys = {
            (layer, expert)
            for layer in spec.routed_layer_indices
            for expert in range(spec.expert_count)
        }
        actual_keys = set(record_by_key)
        missing_keys = sorted(expected_keys - actual_keys)
        extra_keys = sorted(actual_keys - expected_keys)
        if missing_keys or extra_keys or duplicate_keys:
            raise ValueError(
                "manifest routed expert keys differ from model descriptor: "
                f"missing={missing_keys}, extra={extra_keys}, "
                f"duplicates={sorted(duplicate_keys)}"
            )

    def component_signature(record: ExpertRecord) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                segment.component,
                segment.dtype,
                tuple(segment.shape),
                int(segment.length),
            )
            for segment in record.segments
        )

    expert_codec = getattr(spec, "expert_codec", "affine")
    is_mixed = expert_codec == MIXED_OFFICIAL_CODEC
    routed_layers = set(spec.routed_layer_indices)
    exemplar_layer = spec.routed_layer_indices[0]
    record_by_gate_up_tier: dict[str, ExpertRecord] = {}

    if is_mixed:
        # Per-layer bank geometry (D3): every routed layer has its OWN expected
        # signature, derived from that layer's tier via the single source of
        # truth (plan_record_segments), and enforced within the layer. There is
        # no cross-layer exemplar because t158 and affine2 layers differ.
        from mtplx.expert_mixed_official import plan_record_segments

        mixed_dims = {
            "gate_proj": (spec.expert_hidden_size, spec.hidden_size),
            "up_proj": (spec.expert_hidden_size, spec.hidden_size),
            "down_proj": (spec.hidden_size, spec.expert_hidden_size),
        }
        # One exemplar record per gate/up tier, for the per-tier transient banks.
        layer_signatures: dict[int, tuple[tuple[Any, ...], ...]] = {}
        for layer in spec.routed_layer_indices:
            gate_up_tier, down_tier = manifest.mixed_tier_for_layer(layer)
            record_by_gate_up_tier.setdefault(gate_up_tier, record_by_layer[layer])
            planned = plan_record_segments(
                {"gate_up": gate_up_tier, "down": down_tier}, mixed_dims
            )
            layer_signatures[layer] = tuple(
                (seg.component, seg.dtype, tuple(seg.shape), seg.length)
                for seg in planned
            )
            if component_signature(record_by_layer[layer]) != layer_signatures[layer]:
                raise ValueError(
                    f"manifest component geometry for layer {layer} does not "
                    "match the mixed-official tier layout"
                )
        for record in manifest.records:
            if (
                record.layer in routed_layers
                and component_signature(record) != layer_signatures[record.layer]
            ):
                raise ValueError(
                    "routed-layer component geometry differs for expert "
                    f"({record.layer}, {record.expert}) from its layer's tier"
                )
    else:
        expected_signature: list[tuple[str, str, tuple[int, ...], int]] = []
        for projection in ("gate_proj", "up_proj", "down_proj"):
            output_size = (
                spec.expert_hidden_size
                if projection in {"gate_proj", "up_proj"}
                else spec.hidden_size
            )
            input_size = (
                spec.hidden_size
                if projection in {"gate_proj", "up_proj"}
                else spec.expert_hidden_size
            )
            if expert_codec == "mxfp4":
                # Native mxfp4: packed FP4 codes (uint32) + one uint8 E8M0
                # exponent per 32-column group, no bias leaf.
                expected_signature.extend(
                    (
                        (
                            f"{projection}.weight",
                            "U32",
                            (output_size, input_size * spec.quant_bits // 32),
                            output_size * input_size * spec.quant_bits // 8,
                        ),
                        (
                            f"{projection}.scales",
                            "U8",
                            (output_size, input_size // spec.quant_group_size),
                            output_size * (input_size // spec.quant_group_size),
                        ),
                    )
                )
                continue
            if expert_codec != "affine":
                # Shadow-codec (q1) records: packed sign/trit words plus one
                # bf16-bit scale per g64 group, no bias leaf (gate 3 of
                # research/streamed-q1-codec-gap-analysis.md).
                from mtplx.expert_shadow import (
                    SHADOW_GROUP,
                    _B1_WORDS_PER_GROUP,
                    _T158_BYTES_PER_GROUP,
                )

                groups = input_size // SHADOW_GROUP
                if expert_codec == "b1":
                    packed_dtype = "U32"
                    packed_shape = (output_size, groups * _B1_WORDS_PER_GROUP)
                    packed_length = output_size * groups * _B1_WORDS_PER_GROUP * 4
                else:
                    packed_dtype = "U8"
                    packed_shape = (output_size, groups * _T158_BYTES_PER_GROUP)
                    packed_length = output_size * groups * _T158_BYTES_PER_GROUP
                expected_signature.extend(
                    (
                        (
                            f"{projection}.packed",
                            packed_dtype,
                            packed_shape,
                            packed_length,
                        ),
                        (
                            f"{projection}.scales",
                            "U16",
                            (output_size, groups),
                            output_size * groups * 2,
                        ),
                    )
                )
                continue
            weight_shape = (output_size, input_size * spec.quant_bits // 32)
            parameter_shape = (output_size, input_size // spec.quant_group_size)
            expected_signature.extend(
                (
                    (
                        f"{projection}.weight",
                        "U32",
                        weight_shape,
                        output_size * input_size * spec.quant_bits // 8,
                    ),
                    (
                        f"{projection}.scales",
                        "BF16",
                        parameter_shape,
                        output_size
                        * (input_size // spec.quant_group_size)
                        * spec.quant_parameter_bytes,
                    ),
                    (
                        f"{projection}.biases",
                        "BF16",
                        parameter_shape,
                        output_size
                        * (input_size // spec.quant_group_size)
                        * spec.quant_parameter_bytes,
                    ),
                )
            )

        exemplar_signature = component_signature(record_by_layer[exemplar_layer])
        if exemplar_signature != tuple(expected_signature):
            raise ValueError(
                "manifest component geometry does not match the model descriptor"
            )
        for record in manifest.records:
            if (
                record.layer in routed_layers
                and component_signature(record) != exemplar_signature
            ):
                raise ValueError(
                    "routed-layer component geometry differs for "
                    f"expert ({record.layer}, {record.expert}) from canonical "
                    f"layer {exemplar_layer}"
                )

    banks: dict[tuple[str, object], MlxComponentBank] = {}
    slots: dict[str, MlxComponentSlot] = {}
    backend = "mlx-metal-component-banks"

    def bank_for(kind: str, discriminator: object = -1) -> MlxComponentBank:
        keyed = {"persistent", "prefetch", "mixed-transient"}
        key = (kind, discriminator if kind in keyed else -1)
        bank = banks.get(key)
        if bank is not None:
            return bank
        if kind == "persistent":
            capacity = plan.slots_for_layer(int(discriminator))
            record = record_by_layer[discriminator]
            label = f"layer-{discriminator}-persistent-bank"
        elif kind == "prefetch":
            capacity = plan.prefetch_ring_slots
            record = record_by_layer[discriminator]
            label = f"layer-{discriminator}-prefetch-bank"
        elif kind == "mixed-transient":
            # One transient bank per gate/up tier (issue #51 M2b): a miss on a
            # differing-tier layer lands in a bank of ITS geometry.
            capacity = plan.transient_slots
            record = record_by_gate_up_tier[discriminator]
            label = f"mixed-transient-bank-{discriminator}"
        elif kind == "global-persistent":
            capacity = plan.persistent_slots
            record = record_by_layer[exemplar_layer]
            label = "global-persistent-bank"
        elif kind == "global-prefetch":
            # W93: one shared prefetch-ring bank across all layers (uniform record).
            capacity = plan.prefetch_ring_slots
            record = record_by_layer[exemplar_layer]
            label = "global-prefetch-bank"
        else:
            capacity = plan.transient_slots
            record = record_by_layer[exemplar_layer]
            label = "global-transient-bank"
        bank = MlxComponentBank(capacity=capacity, record=record, label=label)
        banks[key] = bank
        return bank

    def allocate(size: int, label: str) -> MlxComponentSlot:
        # The per-record byte size is validated against the resolved bank below:
        # for mixed banks it is the layer's (or transient exemplar's) record
        # size, not a single uniform ``spec.expert_record_bytes`` (which raises).
        layer_persistent = _LAYER_PERSISTENT_LABEL.fullmatch(label)
        global_persistent = _GLOBAL_PERSISTENT_LABEL.fullmatch(label)
        global_transient = _GLOBAL_TRANSIENT_LABEL.fullmatch(label)
        global_prefetch = _GLOBAL_PREFETCH_LABEL.fullmatch(label)
        mixed_transient = _MIXED_TRANSIENT_LABEL.fullmatch(label)
        layer_prefetch = _LAYER_PREFETCH_LABEL.fullmatch(label)
        if mixed_transient is not None:
            tier = mixed_transient.group(1)
            slot_index = int(mixed_transient.group(2))
            if tier not in record_by_gate_up_tier:
                raise ValueError(f"mixed transient tier {tier!r} is not routed")
            if not 0 <= slot_index < plan.transient_slots:
                raise ValueError("transient slot is outside planned capacity")
            bank = bank_for("mixed-transient", tier)
        elif layer_prefetch is not None:
            if plan.cache_scope != "layer":
                raise ValueError(
                    "prefetch slot label conflicts with global cache scope"
                )
            layer = int(layer_prefetch.group(1))
            slot_index = int(layer_prefetch.group(2))
            if layer not in spec.routed_layer_indices:
                raise ValueError(f"prefetch slot layer {layer} is not routed")
            if not 0 <= slot_index < plan.prefetch_ring_slots:
                raise ValueError("prefetch slot is outside planned capacity")
            bank = bank_for("prefetch", layer)
        elif layer_persistent is not None:
            if plan.cache_scope != "layer":
                raise ValueError(
                    "layer-persistent slot label conflicts with global cache scope"
                )
            layer = int(layer_persistent.group(1))
            slot_index = int(layer_persistent.group(2))
            if layer not in spec.routed_layer_indices:
                raise ValueError(f"persistent slot layer {layer} is not routed")
            if not 0 <= slot_index < plan.slots_for_layer(layer):
                raise ValueError("persistent slot is outside planned capacity")
            bank = bank_for("persistent", layer)
        elif label.startswith("layer-") and "-persistent-" in label:
            raise ValueError(f"unknown expert slot label {label!r}")
        elif global_persistent is not None:
            if plan.cache_scope != "global":
                raise ValueError(
                    "global-persistent slot label conflicts with layer cache scope"
                )
            slot_index = int(global_persistent.group(1))
            if not 0 <= slot_index < plan.persistent_slots:
                raise ValueError("global persistent slot is outside planned capacity")
            bank = bank_for("global-persistent", -1)
        elif label.startswith("global-persistent-"):
            raise ValueError(f"unknown expert slot label {label!r}")
        elif global_transient is not None:
            slot_index = int(global_transient.group(1))
            if not 0 <= slot_index < plan.transient_slots:
                raise ValueError("transient slot is outside planned capacity")
            bank = bank_for("transient", -1)
        elif label.startswith("global-transient-"):
            raise ValueError(f"unknown expert slot label {label!r}")
        elif global_prefetch is not None:
            # W93: shared prefetch ring (one bank across all layers).
            if plan.cache_scope != "layer":
                raise ValueError(
                    "prefetch slot label conflicts with global cache scope"
                )
            slot_index = int(global_prefetch.group(1))
            if not 0 <= slot_index < plan.prefetch_ring_slots:
                raise ValueError("prefetch slot is outside planned capacity")
            bank = bank_for("global-prefetch", -1)
        elif label.startswith("global-prefetch-"):
            raise ValueError(f"unknown expert slot label {label!r}")
        else:
            raise ValueError(f"unknown expert slot label {label!r}")
        if int(size) != bank.record_bytes:
            raise ValueError("slot allocator size differs from the bank record")
        if label in slots:
            raise ValueError(f"slot {label} was allocated twice")
        slot = MlxComponentSlot(bank, slot_index, label=label)
        slots[label] = slot
        return slot

    def close_banks() -> None:
        for bank in tuple(banks.values()):
            bank.close()
        banks.clear()
        slots.clear()
        _release_mlx_cache()

    setattr(allocate, "backend", backend)
    setattr(allocate, "slots", slots)
    setattr(allocate, "banks", banks)
    setattr(allocate, "close", close_banks)
    # The plan this allocator sizes its per-bank capacities from. Exposed so a
    # caller can confirm each allocator bank matches the slot pool's exact plan
    # (they must agree or a persistent slot the pool enumerates is rejected as
    # "outside planned capacity"); no bank is allocated to read it.
    setattr(allocate, "plan", plan)
    return allocate


def _release_mlx_cache() -> None:
    try:
        mx.clear_cache()
    except Exception:  # pragma: no cover - compatibility with older MLX
        pass


def _clamped_swiglu(
    gate: mx.array, up: mx.array, swiglu_limit: float | None = None
) -> mx.array:
    """SwiGLU with the reference's optional asymmetric clamp.

    ``swiglu_limit is None`` (or <= 0) is the plain ``swiglu(gate, up)`` -- every
    codec/model that leaves ``spec.swiglu_limit`` unset is byte-for-byte
    unchanged. When set, the clamp is applied to the pre-activation projections
    exactly as DeepSeek ``inference/model.py`` ``Expert.forward`` L846-847 (and
    the resident ``deepseek_v41_moe.ClampedSwiGLU``): the *up* branch is clipped
    two-sided ``[-limit, +limit]`` and the *gate* branch only from above at
    ``+limit``, before the SiLU."""
    if swiglu_limit is not None and swiglu_limit > 0:
        up = mx.clip(up, -swiglu_limit, swiglu_limit)
        gate = mx.minimum(gate, swiglu_limit)
    return swiglu(gate, up)


def _run_mxfp4_expert(
    x: mx.array,
    binding: ExpertSlotBinding,
    *,
    group_size: int,
    bits: int = 4,
    swiglu_limit: float | None = None,
) -> mx.array:
    """Single-expert native-mxfp4 MLP: weight+scales (E8M0, no bias) via mxfp4 qmm."""

    def qmm(values: mx.array, projection: str) -> mx.array:
        return mx.quantized_matmul(
            values,
            _component_array(binding, f"{projection}.weight"),
            scales=_component_array(binding, f"{projection}.scales"),
            group_size=group_size,
            bits=bits,
            mode="mxfp4",
        )

    gate = qmm(x, "gate_proj")
    up = qmm(x, "up_proj")
    return qmm(_clamped_swiglu(gate, up, swiglu_limit), "down_proj")


def _run_q4_expert(
    x: mx.array,
    binding: ExpertSlotBinding,
    *,
    group_size: int,
    bits: int = 4,
    swiglu_limit: float | None = None,
    codec: str = "affine",
) -> mx.array:
    if codec == "mxfp4":
        return _run_mxfp4_expert(
            x, binding, group_size=group_size, bits=bits, swiglu_limit=swiglu_limit
        )
    gate_weight = _component_array(binding, "gate_proj.weight")
    gate_scales = _component_array(binding, "gate_proj.scales")
    gate_biases = _component_array(binding, "gate_proj.biases")
    up_weight = _component_array(binding, "up_proj.weight")
    up_scales = _component_array(binding, "up_proj.scales")
    up_biases = _component_array(binding, "up_proj.biases")
    down_weight = _component_array(binding, "down_proj.weight")
    down_scales = _component_array(binding, "down_proj.scales")
    down_biases = _component_array(binding, "down_proj.biases")

    gate = mx.quantized_matmul(
        x,
        gate_weight,
        scales=gate_scales,
        biases=gate_biases,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )
    up = mx.quantized_matmul(
        x,
        up_weight,
        scales=up_scales,
        biases=up_biases,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )
    hidden = _clamped_swiglu(gate, up, swiglu_limit)
    return mx.quantized_matmul(
        hidden,
        down_weight,
        scales=down_scales,
        biases=down_biases,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )


def _run_component_bank_q4(
    x: mx.array,
    bindings: tuple[ExpertSlotBinding, ...],
    *,
    group_size: int,
    bits: int = 4,
    swiglu_limit: float | None = None,
    codec: str = "affine",
) -> mx.array:
    """Execute assignment-aligned rows from one component-major slot bank."""

    if not bindings or int(x.shape[0]) != len(bindings):
        raise ValueError(
            "component-bank inputs and bindings must be non-empty and aligned"
        )
    bank = getattr(bindings[0].buffer, "bank", None)
    if bank is None or any(
        getattr(binding.buffer, "bank", None) is not bank for binding in bindings
    ):
        raise ValueError("component-bank execution requires one shared bank")
    slot_indices = mx.array(
        [int(binding.buffer.bank_index) for binding in bindings],
        dtype=mx.int32,
    ).reshape((-1, 1))
    return _gather_component_bank(
        x,
        bank,
        slot_indices,
        group_size=group_size,
        bits=bits,
        swiglu_limit=swiglu_limit,
        codec=codec,
    )


def _dequantize_mxfp4_slot(
    bank: MlxComponentBank,
    slot: int,
    projection: str,
    *,
    group_size: int,
    bits: int,
    dtype: "mx.Dtype",
) -> mx.array:
    """Dense ``[out, in]`` matrix for one expert's mxfp4 gs32 projection, at ``dtype``.

    The native-mxfp4 component bank stores ``{projection}.weight`` (packed FP4
    codes, uint32) and ``{projection}.scales`` (one E8M0 exponent byte per
    ``group_size`` columns, no bias leaf) row-major by slot, exactly the leaves
    :func:`_gather_component_bank`'s ``mode="mxfp4"`` gather feeds ``gather_qmm``.
    ``mx.dequantize`` reverses the same codec once, straight into ``dtype`` -- so
    there is no intermediate-precision materialization or cast, and the dequant is
    lossless in bf16 *or* f32 (FP4 code x 2^E8M0 is exactly representable in both);
    only the later dense matmul reassociates vs the fused gather (see
    :func:`_run_component_bank_dense_prefill`)."""
    weight = bank.arrays[f"{projection}.weight"][slot]
    scales = bank.arrays[f"{projection}.scales"][slot]
    return mx.dequantize(
        weight,
        scales,
        group_size=group_size,
        bits=bits,
        mode="mxfp4",
        dtype=dtype,
    )


def _run_component_bank_dense_prefill(
    x: mx.array,
    bindings: tuple[ExpertSlotBinding, ...],
    *,
    group_size: int,
    bits: int = 4,
    swiglu_limit: float | None = None,
    min_rows: int,
    batch: int,
    matmul_dtype: "mx.Dtype | None" = None,
) -> mx.array:
    """Prefill "dequantize once, matmul dense" expert wave (W51, K26).

    Groups this wave's assignment-aligned rows by expert (bank slot; the host-side
    ``binding.buffer.bank_index`` needs no device sync).  For every expert with at
    least ``min_rows`` rows it dequantizes gate/up/down from mxfp4 gs32 to
    ``matmul_dtype`` ONCE and runs three dense matmuls over that expert's rows
    (ClampedSwiGLU unchanged), instead of the per-row ALU/dequant-bound mxfp4
    ``gather_qmm``.  Experts below the threshold keep ``gather_qmm`` (one grouped
    gather).  Results are scattered back into the input row order, so the return
    matches :func:`_gather_component_bank`'s ``[rows, out]`` contract exactly.

    ``matmul_dtype`` (default bf16; ``mx.float32`` for the W51 window-20 A/B) is the
    dense-compute precision -- the dequant lands straight in it (no cast, no doubled
    write).  Bounded transient: each dequantized expert is ~3 x 2 x hidden x inter
    bytes at bf16 (~71 MB at 5120x2304; ~142 MB at f32); experts are processed in
    batches of ``batch`` with an ``mx.eval`` between batches, so at most ``batch``
    experts' dequantized copies are live at once (~0.57 GB bf16 at the default 8) and
    none survives the call.

    NOT bit-identical to the pure ``gather_qmm`` wave: the dequant is exact, but the
    dense matmul accumulation order differs from the fused gather's, so outputs
    match only within the tolerance the W51 CPU test measures.  When no expert clears
    the threshold this is a pure ``gather_qmm`` fall-through (byte-identical).

    W47 prefill stage timing (no-op off-session) breaks the wave into nested brackets
    ``switch.dense.{group_rows,dequant,matmul,scatter}`` and
    ``switch.gather_qmm_fallback`` and tallies rows/experts routed dense vs gather, so
    the next 16K window attributes the residual switch cost."""

    if not bindings or int(x.shape[0]) != len(bindings):
        raise ValueError(
            "component-bank inputs and bindings must be non-empty and aligned"
        )
    bank = getattr(bindings[0].buffer, "bank", None)
    if bank is None or any(
        getattr(binding.buffer, "bank", None) is not bank for binding in bindings
    ):
        raise ValueError("component-bank execution requires one shared bank")

    cdt = matmul_dtype if matmul_dtype is not None else mx.bfloat16

    with _stime.stage_nested("switch.dense.group_rows"):
        slot_of_row = [int(binding.buffer.bank_index) for binding in bindings]
        groups: dict[int, list[int]] = {}
        for row, slot in enumerate(slot_of_row):
            groups.setdefault(slot, []).append(row)
        dense_slots = [slot for slot, rows in groups.items() if len(rows) >= min_rows]
        dense_set = set(dense_slots)
        small_rows = [
            row
            for slot, slot_rows in groups.items()
            if slot not in dense_set
            for row in slot_rows
        ]

    # Per-layer census (no-op off a prefill-timing session): rows/experts routed
    # dense vs the sub-threshold gather fall-through, and one call per invocation.
    _stime.tally("dense.calls", 1)
    _stime.tally("dense.experts_total", len(groups))
    _stime.tally("dense.experts_dense", len(dense_slots))
    _stime.tally("dense.experts_under_threshold", len(groups) - len(dense_slots))
    _stime.tally("dense.rows_dense", len(slot_of_row) - len(small_rows))
    _stime.tally("dense.rows_gather", len(small_rows))

    if not dense_slots:
        # No expert clears the threshold (small-M waves, decode-shaped inputs):
        # a pure gather_qmm wave, byte-identical to the flag-off path.
        with _stime.stage_nested("switch.gather_qmm_fallback"):
            slot_indices = mx.array(slot_of_row, dtype=mx.int32).reshape((-1, 1))
            return _gather_component_bank(
                x,
                bank,
                slot_indices,
                group_size=group_size,
                bits=bits,
                swiglu_limit=swiglu_limit,
                codec="mxfp4",
            )

    parts: list[mx.array] = []
    part_positions: list[int] = []
    pending: list[mx.array] = []
    for slot in dense_slots:
        rows = groups[slot]
        x_slot = mx.take(x, mx.array(rows, dtype=mx.int32), axis=0)
        if x_slot.dtype != cdt:
            x_slot = x_slot.astype(cdt)
        with _stime.stage_nested("switch.dense.dequant") as _dq_fence:
            dq_gate = _dequantize_mxfp4_slot(
                bank, slot, "gate_proj", group_size=group_size, bits=bits, dtype=cdt
            )
            dq_up = _dequantize_mxfp4_slot(
                bank, slot, "up_proj", group_size=group_size, bits=bits, dtype=cdt
            )
            dq_down = _dequantize_mxfp4_slot(
                bank, slot, "down_proj", group_size=group_size, bits=bits, dtype=cdt
            )
            _dq_fence.add(dq_gate, dq_up, dq_down)
        with _stime.stage_nested("switch.dense.matmul") as _mm_fence:
            gate = mx.matmul(x_slot, dq_gate.T)
            up = mx.matmul(x_slot, dq_up.T)
            hidden = _clamped_swiglu(gate, up, swiglu_limit)
            y_slot = mx.matmul(hidden, dq_down.T)
            if y_slot.dtype != x.dtype:
                y_slot = y_slot.astype(x.dtype)
            _mm_fence.add(y_slot)
        parts.append(y_slot)
        part_positions.extend(rows)
        pending.append(y_slot)
        # Bound the peak: materialize each batch of experts so their dequantized
        # transients (unreferenced past this iteration) are freed before the next
        # batch dequantizes.  The small [rows, out] outputs stay in ``parts``.
        if len(pending) >= batch:
            mx.eval(pending)
            pending = []
    if pending:
        mx.eval(pending)

    if small_rows:
        with _stime.stage_nested("switch.gather_qmm_fallback"):
            small_idx = mx.array(small_rows, dtype=mx.int32)
            sub_x = mx.take(x, small_idx, axis=0)
            sub_slots = mx.array(
                [slot_of_row[row] for row in small_rows], dtype=mx.int32
            ).reshape((-1, 1))
            parts.append(
                _gather_component_bank(
                    sub_x,
                    bank,
                    sub_slots,
                    group_size=group_size,
                    bits=bits,
                    swiglu_limit=swiglu_limit,
                    codec="mxfp4",
                )
            )
            part_positions.extend(small_rows)

    with _stime.stage_nested("switch.dense.scatter"):
        if len(parts) == 1 and part_positions == list(range(len(slot_of_row))):
            return parts[0]
        joined = mx.concatenate(parts, axis=0)
        order = mx.argsort(mx.array(part_positions, dtype=mx.int32))
        return mx.take(joined, order, axis=0)


def _gather_component_bank(
    x: mx.array,
    bank: MlxComponentBank,
    slot_indices: mx.array,
    *,
    group_size: int,
    bits: int,
    swiglu_limit: float | None = None,
    codec: str = "affine",
) -> mx.array:
    """Row-gathered three-matrix expert MLP against one component bank.

    ``codec="affine"`` uses weight+scales+biases through ``gather_qmm(mode=
    "affine")``; ``codec="mxfp4"`` uses weight+scales only (E8M0, no bias) through
    ``gather_qmm(mode="mxfp4")``.  Both preserve the optional SwiGLU clamp.
    """

    rows = int(x.shape[0])
    width = int(x.shape[-1])

    # W56/K27 layout fix: sort rows by bank slot so gather_qmm takes the fused
    # weight-streamed-once kernel (see the _LAYOUT_FIX_* env docs above).  Gated to
    # many-row prefill waves; decode/verify stay on the exact shipped unsorted call.
    use_sorted = _layout_fix_enabled() and rows >= _layout_fix_min_rows()
    if use_sorted:
        perm = mx.argsort(slot_indices.reshape(-1))
        inv_perm = mx.argsort(perm)
        x = mx.take(x, perm, axis=0)
        slot_indices = mx.take(slot_indices, perm, axis=0)

    def _wave(values_x: mx.array, slot_col: mx.array) -> mx.array:
        """Three-projection expert MLP over one (already-oriented) row block."""
        n = int(values_x.shape[0])
        selected = values_x.reshape((n, 1, 1, width))
        if codec == "mxfp4":
            def qmm(values: mx.array, projection: str) -> mx.array:
                # W92 dispatch census: one grouped gather_qmm per component over
                # ALL routed slots (never per-expert). Probe-gated (zero cost when
                # MTPLX_ROUTE_STAGE_PROBE is off); the receipt divides this by
                # hot.all_hit to prove the all-hit switch is 3 dispatches/call.
                _route_probe.count("hot.switch_gather_qmm")
                return mx.gather_qmm(
                    values,
                    bank.arrays[f"{projection}.weight"],
                    bank.arrays[f"{projection}.scales"],
                    rhs_indices=slot_col,
                    transpose=True,
                    group_size=group_size,
                    bits=bits,
                    mode="mxfp4",
                    sorted_indices=use_sorted,
                )
        else:
            def qmm(values: mx.array, projection: str) -> mx.array:
                # W92 dispatch census: see the mxfp4 branch above.
                _route_probe.count("hot.switch_gather_qmm")
                return mx.gather_qmm(
                    values,
                    bank.arrays[f"{projection}.weight"],
                    bank.arrays[f"{projection}.scales"],
                    bank.arrays[f"{projection}.biases"],
                    rhs_indices=slot_col,
                    transpose=True,
                    group_size=group_size,
                    bits=bits,
                    mode="affine",
                    sorted_indices=use_sorted,
                )

        gate = qmm(selected, "gate_proj")
        up = qmm(selected, "up_proj")
        swig = _clamped_swiglu(gate, up, swiglu_limit)
        # F2 down-K pad: match the SwiGLU activation width to the bank's down-proj K.
        # When the bank is padded (down weight 2560-wide) this zero-pads the activation
        # so gather_qmm sees K%512==0 and takes the fast mxfp4 qmv; a zero activation
        # tail against the zero weight tail contributes exactly 0 -> byte-identical.
        # No-op when off or when the bank is unpadded (k_wt == k_act).
        if _down_k_pad_enabled():
            dw = bank.arrays["down_proj.weight"]
            k_wt = int(dw.shape[-1]) * (32 // int(bits))  # packed uint32 -> logical K
            k_act = int(swig.shape[-1])
            if k_wt > k_act:
                pad = [(0, 0)] * (swig.ndim - 1) + [(0, k_wt - k_act)]
                swig = mx.pad(swig, pad)
        out = qmm(swig, "down_proj")
        return out.reshape((n, int(out.shape[-1])))

    per_call = _gather_rows_per_call() if use_sorted else 0
    if per_call and per_call < rows:
        parts = [
            _wave(x[c0:c0 + per_call], slot_indices[c0:c0 + per_call])
            for c0 in range(0, rows, per_call)
        ]
        output = mx.concatenate(parts, axis=0)
    else:
        output = _wave(x, slot_indices)

    if use_sorted:
        output = mx.take(output, inv_perm, axis=0)
    return output


def _run_component_bank_shadow(
    x: mx.array,
    bindings: tuple[ExpertSlotBinding, ...],
    *,
    codec: str,
    swiglu_limit: float | None = None,
) -> mx.array:
    """Execute assignment-aligned rows of a shadow-codec (q1) slot bank.

    The q2 lane's equal (gate 4 of the gap analysis): records were read
    into component-bank slots by the ordinary slot machinery, and the rows
    execute through ``shadow_gather_mm`` — the bank row fed to the kernel
    is the slot index, not the expert id.  Eager-only, like the miss-shadow
    lane: the shadow kernel is not traceable under ``mx.compile``.
    """

    if not bindings or int(x.shape[0]) != len(bindings):
        raise ValueError(
            "component-bank inputs and bindings must be non-empty and aligned"
        )
    bank = getattr(bindings[0].buffer, "bank", None)
    if bank is None or any(
        getattr(binding.buffer, "bank", None) is not bank for binding in bindings
    ):
        raise ValueError("component-bank execution requires one shared bank")
    slot_rows = mx.array(
        [int(binding.buffer.bank_index) for binding in bindings],
        dtype=mx.int32,
    )
    return _shadow_gather_component_bank(
        x, bank, slot_rows, codec=codec, swiglu_limit=swiglu_limit
    )


def _shadow_gather_component_bank(
    x: mx.array,
    bank: MlxComponentBank,
    slot_rows: mx.array,
    *,
    codec: str,
    swiglu_limit: float | None = None,
) -> mx.array:
    """Row-gathered three-projection shadow MLP against one component bank."""

    from mtplx.kernels.shadow_gather import shadow_gather_mm

    def projection(values: mx.array, name: str) -> mx.array:
        return shadow_gather_mm(
            values,
            slot_rows,
            bank.arrays[f"{name}.packed"],
            bank.arrays[f"{name}.scales"],
            codec=codec,
        )

    gate = projection(x, "gate_proj")
    up = projection(x, "up_proj")
    return projection(_clamped_swiglu(gate, up, swiglu_limit), "down_proj")


def _run_component_bank_mixed(
    x: mx.array,
    bindings: tuple[ExpertSlotBinding, ...],
    *,
    gate_up_tier: str,
    group_size: int,
    swiglu_limit: float | None = None,
) -> mx.array:
    """Execute a mixed-official (issue #51, M2) component-bank wave.

    Per-projection-group dispatch (D4): the layer's gate/up pair runs through
    ``shadow_gather_mm`` (t158-tier) or ``gather_qmm(bits=2)`` (affine2-tier),
    and ``down`` always runs through ``gather_qmm(bits=3)`` — all in one pass,
    against one shared per-layer bank. Eager-only, like the shadow lane.
    """

    if not bindings or int(x.shape[0]) != len(bindings):
        raise ValueError(
            "component-bank inputs and bindings must be non-empty and aligned"
        )
    bank = getattr(bindings[0].buffer, "bank", None)
    if bank is None or any(
        getattr(binding.buffer, "bank", None) is not bank for binding in bindings
    ):
        raise ValueError("component-bank execution requires one shared bank")
    slot_rows = [int(binding.buffer.bank_index) for binding in bindings]
    slot_rows_1d = mx.array(slot_rows, dtype=mx.int32)
    slot_indices_2d = mx.array(slot_rows, dtype=mx.int32).reshape((-1, 1))
    return _gather_component_bank_mixed(
        x,
        bank,
        slot_indices_2d,
        slot_rows_1d,
        gate_up_tier=gate_up_tier,
        group_size=group_size,
        swiglu_limit=swiglu_limit,
    )


def _gather_component_bank_mixed(
    x: mx.array,
    bank: MlxComponentBank,
    slot_indices_2d: mx.array,
    slot_rows_1d: mx.array,
    *,
    gate_up_tier: str,
    group_size: int,
    swiglu_limit: float | None = None,
) -> mx.array:
    """Row-gathered mixed-tier expert MLP against one shared per-layer bank."""

    from mtplx.kernels.shadow_gather import shadow_gather_mm

    rows = int(x.shape[0])
    in_dim = int(x.shape[-1])

    def affine(values: mx.array, projection: str, bits: int) -> mx.array:
        selected = values.reshape((rows, 1, 1, int(values.shape[-1])))
        out = mx.gather_qmm(
            selected,
            bank.arrays[f"{projection}.weight"],
            bank.arrays[f"{projection}.scales"],
            bank.arrays[f"{projection}.biases"],
            rhs_indices=slot_indices_2d,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode="affine",
        )
        return out.reshape((rows, int(out.shape[-1])))

    def shadow(values: mx.array, projection: str) -> mx.array:
        return shadow_gather_mm(
            values,
            slot_rows_1d,
            bank.arrays[f"{projection}.packed"],
            bank.arrays[f"{projection}.scales"],
            codec="t158",
        )

    if gate_up_tier == "t158":
        gate = shadow(x.reshape((rows, in_dim)), "gate_proj")
        up = shadow(x.reshape((rows, in_dim)), "up_proj")
    elif gate_up_tier == "affine2":
        bits = _MIXED_AFFINE_BITS["affine2"]
        gate = affine(x, "gate_proj", bits)
        up = affine(x, "up_proj", bits)
    else:  # pragma: no cover - guarded by the manifest tier contract
        raise ValueError(f"unsupported mixed gate/up tier {gate_up_tier!r}")
    hidden = _clamped_swiglu(gate, up, swiglu_limit)
    return affine(hidden, "down_proj", _MIXED_DOWN_BITS)


def _run_shadow_bank(
    x: mx.array,
    expert_rows: mx.array,
    bank: Any,
    swiglu_limit: float | None = None,
) -> mx.array:
    """Three-projection expert MLP against a low-precision shadow bank.

    ``expert_rows`` are raw expert ids — the shadow bank row index is the
    expert id (island-bank convention). Output is a quality tier, close to
    the exact quantized path but not bitwise. Eager-only: the shadow
    kernel is not traceable under ``mx.compile``.
    """

    from mtplx.kernels.shadow_gather import shadow_gather_mm

    def projection(values: mx.array, name: str) -> mx.array:
        return shadow_gather_mm(
            values,
            expert_rows,
            bank.arrays[f"{name}.packed"],
            bank.arrays[f"{name}.scales"],
            codec=bank.codec,
        )

    gate = projection(x, "gate_proj")
    up = projection(x, "up_proj")
    return projection(_clamped_swiglu(gate, up, swiglu_limit), "down_proj")


def _run_mapped_q4(
    x: mx.array,
    mapped: MappedExpertRecord,
    *,
    group_size: int,
    bits: int = 4,
    swiglu_limit: float | None = None,
    codec: str = "affine",
) -> mx.array:
    arrays = mapped.arrays

    if codec == "mxfp4":
        def qmm(values: mx.array, projection: str) -> mx.array:
            return mx.quantized_matmul(
                values,
                arrays[f"{projection}.weight"],
                scales=arrays[f"{projection}.scales"],
                group_size=group_size,
                bits=bits,
                mode="mxfp4",
            )
    else:
        def qmm(values: mx.array, projection: str) -> mx.array:
            return mx.quantized_matmul(
                values,
                arrays[f"{projection}.weight"],
                scales=arrays[f"{projection}.scales"],
                biases=arrays[f"{projection}.biases"],
                group_size=group_size,
                bits=bits,
                mode="affine",
            )

    return qmm(
        _clamped_swiglu(qmm(x, "gate_proj"), qmm(x, "up_proj"), swiglu_limit),
        "down_proj",
    )


class MappedExpertSwitchGLU(nn.Module):
    """Execute routed quantized experts from record-sized file-backed MTLBuffers."""

    def __init__(
        self,
        runtime: ExpertStreamingRuntime,
        store: MappedExpertStore,
        layer_index: int,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.store = store
        self.layer_index = int(layer_index)
        self.group_size = runtime.spec.quant_group_size
        self.bits = runtime.spec.quant_bits
        self.swiglu_limit = getattr(runtime.spec, "swiglu_limit", None)
        self.codec = getattr(runtime.spec, "expert_codec", "affine")

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        hidden_size = int(x.shape[-1])
        top_k = int(indices.shape[-1])
        if top_k != self.runtime.spec.top_k:
            raise ValueError("mapped router top-k differs from the model descriptor")
        tokens = x.reshape((-1, hidden_size))
        mx.eval(indices)
        expert_ids = tuple(int(value) for value in indices.reshape(-1).tolist())
        phase = current_expert_routing_phase(token_count=int(x.shape[-2]))
        self.runtime.observe_route(
            self.layer_index,
            phase,
            expert_ids,
            token_count=int(tokens.shape[0]),
        )
        by_expert: dict[int, list[int]] = {}
        for position, expert in enumerate(expert_ids):
            by_expert.setdefault(expert, []).append(position)

        outputs: list[mx.array] = []
        output_positions: list[int] = []
        for expert, positions in by_expert.items():
            token_positions = mx.array(
                [position // top_k for position in positions],
                dtype=mx.int32,
            )
            selected = mx.take(tokens, token_positions, axis=0)
            outputs.append(
                _run_mapped_q4(
                    selected,
                    self.store.get(self.layer_index, expert),
                    group_size=self.group_size,
                    bits=self.bits,
                    swiglu_limit=self.swiglu_limit,
                    codec=self.codec,
                )
            )
            output_positions.extend(positions)
        mx.eval(outputs)
        self.store.observe_qmm(len(by_expert))
        joined = mx.concatenate(outputs, axis=0)
        order = mx.argsort(mx.array(output_positions, dtype=mx.int32))
        return mx.take(joined, order, axis=0).reshape((*indices.shape, hidden_size))


class HotExpertSwitchGLU(nn.Module):
    """Correctness-first slot-backed replacement for ``SwitchGLU``.

    This portable path reconstructs MLX arrays from each fixed host slot and
    evaluates a bounded wave before releasing it.  The native extension can
    replace the component binding without changing router or cache semantics.
    """

    def __init__(self, runtime: ExpertStreamingRuntime, layer_index: int):
        super().__init__()
        self.runtime = runtime
        self.layer_index = int(layer_index)
        self.group_size = runtime.spec.quant_group_size
        self.bits = runtime.spec.quant_bits
        # Streamed record codec (issue #51 q1 lane): "affine" executes
        # slot-resident records through gather_qmm; a shadow codec ("b1",
        # "t158") executes them through shadow_gather_mm instead.  The read
        # path is identical either way — only the component-bank dispatch
        # differs (gate 4 of the gap analysis).
        self.codec = getattr(runtime.spec, "expert_codec", "affine")
        # Reference SwiGLU clamp for models that set it (DeepSeek-V4.1: 10.0);
        # None -> plain SwiGLU, so every other streamed model is unchanged.
        self.swiglu_limit = getattr(runtime.spec, "swiglu_limit", None)
        # Mixed-official (issue #51, M2): resolve this layer's (gate_up, down)
        # tier from the loaded manifest's layer_tier_map — never the spec (D1).
        # Fail closed if the layer has no tier entry.
        self._gate_up_tier: str | None = None
        self._down_tier: str | None = None
        if self.codec == MIXED_OFFICIAL_CODEC:
            self._gate_up_tier, self._down_tier = (
                runtime.manifest.mixed_tier_for_layer(self.layer_index)
            )
        # Shadow miss fallback (issue #51): when the runtime opened with
        # miss_shadow, decode misses on this streamed layer are served from
        # a resident low-precision bank instead of waiting on SSD.
        shadow_lookup = getattr(runtime, "shadow_bank_for_layer", None)
        self._shadow_bank = (
            shadow_lookup(self.layer_index) if callable(shadow_lookup) else None
        )
        # M6 is an explicit construction-time route. The control binds no
        # submitter and retains routed-then-shared ordering; the candidate binds
        # the verify-miss submitter once, with no environment read or invariant
        # revalidation in the measured path.
        self._verify_shared_async_eval = None
        self._verify_shared_submit = None
        if getattr(runtime.config, "verify_shared_overlap", False):
            async_eval = getattr(mx, "async_eval", None)
            if not callable(async_eval):
                raise RuntimeError(
                    "verify_shared_overlap requires callable mlx.core.async_eval"
                )
            self._verify_shared_async_eval = async_eval
            self._verify_shared_submit = self._submit_verify_shared_overlap

    def _submit_verify_shared_overlap(
        self,
        shared_work: Callable[[], mx.array],
    ) -> mx.array:
        shared = shared_work()
        self._verify_shared_async_eval(shared)
        return shared

    def _dispatch_component_bank(
        self,
        selected: mx.array,
        bindings: tuple[ExpertSlotBinding, ...],
        *,
        dense_prefill: bool = False,
    ) -> mx.array:
        """Execute one component-bank wave under this layer's record codec.

        ``dense_prefill`` (W51/K26) is threaded True only from the streamed switch's
        PREFILL split path; the decode all-hit / device-route / shadow callers leave
        it at the default, so the dense expert path can never engage at M=1."""

        # W44: capture this layer's component bank from a routed binding the
        # first time we see one, so the barrier-free device-route path can gather
        # against it without a binding of its own. Idempotent + guarded (fakes
        # and non-streamed runtimes simply lack the hook).
        if bindings:
            register = getattr(self.runtime, "register_component_bank", None)
            if callable(register):
                register(self.layer_index, getattr(bindings[0].buffer, "bank", None))

        if self.codec in ("affine", "mxfp4"):
            if dense_prefill and self.codec == "mxfp4":
                return _run_component_bank_dense_prefill(
                    selected,
                    bindings,
                    group_size=self.group_size,
                    bits=self.bits,
                    swiglu_limit=self.swiglu_limit,
                    min_rows=_positive_env_int(
                        _PREFILL_DENSE_MIN_ROWS_ENV, _PREFILL_DENSE_MIN_ROWS_DEFAULT
                    ),
                    batch=_positive_env_int(
                        _PREFILL_DENSE_BATCH_ENV, _PREFILL_DENSE_BATCH_DEFAULT
                    ),
                    matmul_dtype=_prefill_dense_matmul_dtype(),
                )
            return _run_component_bank_q4(
                selected,
                bindings,
                group_size=self.group_size,
                bits=self.bits,
                swiglu_limit=self.swiglu_limit,
                codec=self.codec,
            )
        if self.codec == MIXED_OFFICIAL_CODEC:
            assert self._gate_up_tier is not None
            return _run_component_bank_mixed(
                selected,
                bindings,
                gate_up_tier=self._gate_up_tier,
                group_size=self.group_size,
                swiglu_limit=self.swiglu_limit,
            )
        return _run_component_bank_shadow(
            selected, bindings, codec=self.codec, swiglu_limit=self.swiglu_limit
        )

    def _run_device_route(
        self,
        x: mx.array,
        indices: mx.array,
        tokens: mx.array,
        top_k: int,
        hidden_size: int,
        shared_work: Callable[[], mx.array] | None,
        *,
        pinned: bool = False,
    ) -> tuple[mx.array, mx.array | None] | None:
        """W44 barrier-free all-hit path (env ``MTPLX_DSV41_DEVICE_ROUTE``); W71
        pinned variant (env ``MTPLX_DSV41_DEVICE_ROUTE_PINNED``) when ``pinned``.

        Gathers ``lut[indices]`` on the device -- NO ``mx.eval(indices)``, no
        ``.tolist()``, zero host syncs on this layer -- and defers verification to
        an ``async_eval`` read enqueued for the next flush.  ``pinned`` swaps the
        resident LUT/snapshot for the PINNED-only ones: the gather is then exact
        only on an all-PINNED route (whose slots W64 keeps stable, so the deferred
        read cannot race a recycle -- the W44 §8 fix), and any non-pinned expert
        reads a void row and is caught by the deferred flush, whose caller
        recomputes that layer on the fenced path (W71_DEVICE_ROUTE_PINNED.md).  For
        every kept assignment the LUT slot equals the fenced ``bank_index``, so the
        routed output is byte-identical.  Returns ``None`` to fall back to the
        fenced path when the bank is not yet captured."""

        bank = self.runtime.component_bank_for_layer(self.layer_index)
        if bank is None:
            return None
        if pinned:
            lut = self.runtime.device_route_pinned_lut(self.layer_index)
            snapshot = self.runtime.device_route_pinned_snapshot(self.layer_index)
        else:
            lut = self.runtime.device_route_lut(self.layer_index)
            snapshot = self.runtime.device_route_snapshot(self.layer_index)
        if lut is None:
            # W92: the LUT builder could not take this layer's lock (a pending
            # deferred route holds it) -- use the fenced path this token, which
            # runs the covering flush that releases the deferral.
            return None
        # Device-side expert -> slot: no host round-trip on ``indices``.
        slot = mx.take(lut, indices.reshape(-1))
        safe = mx.maximum(slot, 0).reshape((-1, 1))
        rows = int(tokens.shape[0])
        assignment_inputs = mx.broadcast_to(
            tokens[:, None, :], (rows, top_k, hidden_size)
        ).reshape((-1, hidden_size))
        with _route_probe.bracket("hot.device_route_gather"):
            routed = _gather_component_bank(
                assignment_inputs,
                bank,
                safe,
                group_size=self.group_size,
                bits=self.bits,
                swiglu_limit=self.swiglu_limit,
                codec=self.codec,
            )
        routed = routed.reshape((*indices.shape, hidden_size))
        # Deferred verification: submit ``indices`` (non-blocking) and enqueue the
        # probe against the snapshot the LUT was built from. Never read here --
        # that would be the barrier this path exists to remove.
        _async_eval = getattr(mx, "async_eval", None)
        if callable(_async_eval):
            _async_eval(indices)
        # Resident (W44) path keeps the 3-arg call so a runtime double predating
        # W71 is unaffected; the pinned path opts in explicitly.
        if pinned:
            self.runtime.enqueue_device_route_probe(
                self.layer_index, indices, snapshot, pinned=True
            )
        else:
            self.runtime.enqueue_device_route_probe(
                self.layer_index, indices, snapshot
            )
        _route_probe.count("hot.device_route_pinned" if pinned else "hot.device_route")
        shared = shared_work() if shared_work is not None else None
        return routed, shared

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        output, _overlap_result = self._run(
            x,
            indices,
            shared_work=None,
        )
        return output

    def run_with_shared_overlap(
        self,
        x: mx.array,
        indices: mx.array,
        shared_work: Callable[[], mx.array],
    ) -> tuple[mx.array, mx.array]:
        """Evaluate resident shared work while decode misses stream from SSD."""

        output, shared = self._run(
            x,
            indices,
            shared_work=shared_work,
        )
        assert shared is not None
        return output, shared

    def _run(
        self,
        x: mx.array,
        indices: mx.array,
        *,
        shared_work: Callable[[], mx.array] | None,
    ) -> tuple[mx.array, mx.array | None]:
        if indices.ndim < 1:
            raise ValueError("expert indices must include a top-k dimension")
        if int(indices.shape[-1]) != self.runtime.spec.top_k:
            raise ValueError(
                f"router selected {indices.shape[-1]} experts; expected "
                f"{self.runtime.spec.top_k}"
            )
        hidden_size = int(x.shape[-1])
        if hidden_size != self.runtime.spec.hidden_size:
            raise ValueError(
                f"expert input width {hidden_size} does not match "
                f"{self.runtime.spec.hidden_size}"
            )
        tokens = x.reshape(-1, hidden_size)
        top_k = int(indices.shape[-1])
        # W93 gate-oracle prefetch: read-and-clear any prediction the DecoderLayer
        # stashed (the NEXT layer's top-k ids, computed from THIS layer's input
        # residual before attention). Consumed exactly once here regardless of the
        # route path taken below: the fenced path rides it on its indices barrier
        # and issues the reads; the barrier-free device-route path drops it (there
        # is no host sync to piggyback -- W93_GATE_PREFETCH.md §3).
        _gate_prefetch_pending = getattr(self, "_mtplx_gate_prefetch_pending", None)
        if _gate_prefetch_pending is not None:
            self._mtplx_gate_prefetch_pending = None
        # DSV4.1 switch fast-path (W42, KERNEL_LEDGER K23; env
        # ``MTPLX_DSV41_SWITCH_FASTPATH``, default off).  W37 measured the
        # streamed switch at ~2.5 ms/layer with warm-repeat decode == cold decode
        # (3.80 vs 3.73 tok/s), so miss I/O is NOT the cost.  The route probe
        # attributes the exposed all-hit cost to ``hot.allhit_fence_eval``: one
        # BLOCKING ``mx.eval(wave_output)`` per all-hit layer (``synchronous_
        # fence``) -- a SECOND device->host round-trip on top of the
        # ``mx.eval(indices)`` routing barrier, serialising the decode so the next
        # layer cannot overlap this layer's gather.  hy3/glm never pay it: their
        # profiles ship ``deferred_pin_release=True`` (+ ``split_route_release=
        # "deferred"``), so the all-hit release defers to the next barrier's
        # covering eval and the split waves dispatch via ``async_eval``.  DSV4.1's
        # ``build_streaming_config`` leaves both at the dataclass default
        # (fenced/False), so this lane -- and only this lane -- eats the extra
        # per-layer sync.  The fast-path promotes the identical, already-shipped
        # deferred mechanism for DSV4.1 WITHOUT mutating the config (other lanes
        # and callers are untouched; the flag A/Bs cleanly).  It is a pure
        # fence/release-timing reorder -- the gather math is unchanged -- so the
        # output is byte-identical (asserted for all-hit / split / all-miss at
        # M=1 and M=4).  It engages only when the runtime actually implements the
        # deferred-release machinery (the real ExpertStreamingRuntime does; a fake
        # without it falls back to the shipped fence, never crashing).
        _switch_fastpath = os.environ.get("MTPLX_DSV41_SWITCH_FASTPATH") == "1"
        # Variant B (W42 window-14 diagnosis): pure defer measured -13.4% decode.
        # Root cause -- the all-hit deferred branch below submitted NOTHING (no
        # eval, no async_eval), unlike the split defer branch
        # (``evaluate_component_bindings`` async_evals its wave outputs: "the GPU
        # still needs the part submitted now -- without it the device idles").
        # DSV4.1's backbone (deepseek_v41.py, W41, out of this allowlist) has no
        # submit cadence either (hy3 pairs deferred_pin_release with
        # ``MTPLX_HY3_SUBMIT_CADENCE=8``, hy3_mlx.py:1165), so on all-hit layers
        # the lazy graph accrues and the device idles until the next routing
        # barrier drains it in one lump (the a3b backpressure trap,
        # [[a3b-decode-roundtrip-is-the-lever]]).  ``MTPLX_DSV41_SWITCH_SUBMIT``
        # (companion to the fast-path, default off) makes the all-hit deferred
        # branch async_eval its wave output -- a non-blocking per-layer submit
        # that keeps the GPU fed WITHOUT the blocking host round-trip -- matching
        # the split path.  Scheduling only; byte-identical.
        _switch_submit = os.environ.get("MTPLX_DSV41_SWITCH_SUBMIT") == "1"
        _fastpath_can_defer = (
            _switch_fastpath
            and callable(getattr(self.runtime, "defer_slot_release", None))
            and callable(
                getattr(self.runtime, "flush_deferred_slot_releases", None)
            )
        )
        # OR the env promotion with the config field at every decision site. With
        # the flag OFF, ``_fastpath_can_defer`` is False and these collapse to the
        # exact shipped expressions -> byte-identical, zero behavioural change.
        _deferred_pin_active = (
            getattr(self.runtime.config, "deferred_pin_release", False)
            or _fastpath_can_defer
        )
        _split_deferred_active = (
            getattr(self.runtime.config, "split_route_release", "fenced")
            == "deferred"
            or _fastpath_can_defer
        )
        # W44 device-route (K24, env ``MTPLX_DSV41_DEVICE_ROUTE``) + W71 pinned
        # variant (env ``MTPLX_DSV41_DEVICE_ROUTE_PINNED``): the barrier-free path.
        # Engages only for a DECODE component-bank layer whose bank has been
        # captured (a prior fenced route) and whose codec gathers through
        # gather_qmm (affine/mxfp4).  It issues the gather over the device LUT
        # WITHOUT ``mx.eval(indices)`` and defers verification
        # (``flush_device_route_probes`` at the token boundary).  W71 (pinned) uses
        # the PINNED-only LUT, so the gather is exact only on an all-pinned route
        # (slot-stable under W64) and any non-pinned expert is recovered fenced;
        # this closes the W44 window-19 slot-recycle race.  Falls through to the
        # fenced path otherwise -- including the first route for a layer, which
        # populates residency and captures the bank.  See W71_DEVICE_ROUTE_PINNED.md.
        _dr_pinned = os.environ.get("MTPLX_DSV41_DEVICE_ROUTE_PINNED") == "1"
        _dr = os.environ.get("MTPLX_DSV41_DEVICE_ROUTE") == "1"
        _dr_lut_attr = "device_route_pinned_lut" if _dr_pinned else "device_route_lut"
        if (
            (_dr or _dr_pinned)
            and self.codec in ("affine", "mxfp4")
            and self.runtime.config.slot_layout == "component-banks"
            and current_expert_routing_phase(token_count=int(x.shape[-2]))
            is RoutingPhase.DECODE
            and callable(getattr(self.runtime, _dr_lut_attr, None))
            and callable(getattr(self.runtime, "component_bank_for_layer", None))
            # Cold recovery: the backbone forces specific layers back onto the
            # fenced path (barrier + admit + gather) on a recovery pass, so a miss
            # is repaired byte-identically. Those layers skip the device path here.
            and self.layer_index
            not in getattr(self.runtime, "_device_route_force_fenced", frozenset())
        ):
            device_result = self._run_device_route(
                x, indices, tokens, top_k, hidden_size, shared_work,
                pinned=_dr_pinned,
            )
            if device_result is not None:
                return device_result
        # Shared-branch hoist (issue #51, flag-gated, exact-quality). Normally
        # the resident shared MLP is submitted AFTER begin_split_route so it
        # overlaps the miss reads (~0.44 ms). But the router host-sync below
        # (mx.eval(indices), the per-streamed-layer barrier, p50 ~1 ms) is the
        # bigger exposed cost, and the shared branch depends only on x — not on
        # the routed indices — so it can be dispatched HERE to fill the GPU-idle
        # window of the sync round-trip instead. Pure execution reorder: same
        # shared_mlp(x), same combine, bitwise-identical output. Decode single
        # token only; requires async_eval so the dispatch does not itself block.
        # ``MTPLX_DSV41_SHARED_OVERLAP`` (W28, K1) drives the identical reorder
        # for the DeepSeek-V4.1 MoE, whose W11 module now hands its shared expert
        # in as ``shared_work``.  Same hoist, same bitwise-identical combine; a
        # distinct env name so DSV4.1 can be A/B'd without arming hy3/glm.
        _hoisted_shared: mx.array | None = None
        if (
            shared_work is not None
            and int(x.shape[-2]) == 1
            and (
                os.environ.get("MTPLX_HY3_SHARED_HOIST") == "1"
                or os.environ.get("MTPLX_DSV41_SHARED_OVERLAP") == "1"
            )
        ):
            _async_eval = getattr(mx, "async_eval", None)
            if callable(_async_eval):
                _hoisted_shared = shared_work()
                _async_eval(_hoisted_shared)
        with _route_probe.bracket("hot.eval_indices"):
            # W93: the next layer's predicted ids ride THIS one host sync (they are
            # independent of ``indices``, so both materialize in one device->host
            # round-trip -- no second barrier). Exactly one mx.eval either way.
            if _gate_prefetch_pending is not None:
                mx.eval(indices, _gate_prefetch_pending[1])
            else:
                mx.eval(indices)
        # This eval materialized every earlier layer's wave output, so any
        # deferred pin releases are now covered without their own fence.
        flush_deferred = getattr(self.runtime, "flush_deferred_slot_releases", None)
        if flush_deferred is not None:
            flush_deferred()
        with _route_probe.bracket("hot.route_host"):
            expert_ids = tuple(int(value) for value in indices.reshape(-1).tolist())
        # W93 issue-order fix (lane C, HIGH-3): the NEXT layer's gate-oracle
        # prefetch is issued only AFTER this layer's own demand misses have been
        # submitted by ``begin_split_route`` -- so the speculative ring reads never
        # queue ahead of the current layer's demand reads at the SSD (the
        # cannibalism hazard: a ~216 MiB speculative burst racing this layer's
        # misses on a separate executor). ONLY THE ISSUE POINT MOVES: the eval that
        # materializes the prediction still happened above, on the single
        # ``mx.eval(indices, predicted)`` routing barrier, so ``.tolist()`` at the
        # deferred issue site is host-resident with no extra sync.
        #
        # Ordering invariant at the deferred issue site (below, before each route
        # return): the issue happens AFTER the covering flush
        # (``flush_deferred_slot_releases`` above) and AFTER ``begin_split_route``
        # has submitted L's misses; L's gather is an ancestor of indices_{L+1} (the
        # prediction was formed from L's input residual ``h`` and materialized on
        # L's own indices barrier), so issuing later adds no host sync and cannot
        # reorder ahead of the demand reads. Fired exactly once per ``_run`` via the
        # guard. The barrier-free device-route path returns earlier (above) and
        # drops the prediction unevaluated -- there is no barrier to piggyback and
        # so nothing to issue (W93_GATE_PREFETCH.md §3).
        _gate_prefetch_issued = False

        def _issue_pending_gate_prefetch() -> None:
            nonlocal _gate_prefetch_issued
            if _gate_prefetch_pending is None or _gate_prefetch_issued:
                return
            _gate_prefetch_issued = True
            # W100: this route is a DSpark verify when it is a small-M (2..8-row)
            # DECODE forward -- the same shape ``_maybe_stash_gate_prefetch``
            # stashed the per-row union for. Tag the issue so the runtime can
            # prove verify-phase engagement apart from the AR (M=1) total. ``phase``
            # / ``tokens`` are bound above before any call site.
            _verify_phase = (
                phase is RoutingPhase.DECODE and 2 <= int(tokens.shape[0]) <= 8
            )
            _issue_gate_prefetch(
                self.runtime, _gate_prefetch_pending, verify=_verify_phase
            )
        # Batch size is not a generation phase. A batched decode has shape
        # ``[B, 1, H]`` and must still train/use the persistent decode hot set;
        # only the sequence length distinguishes prefill from decode here.
        phase = current_expert_routing_phase(token_count=int(x.shape[-2]))
        self.runtime.observe_route(
            self.layer_index,
            phase,
            expert_ids,
            token_count=int(tokens.shape[0]),
        )
        if phase is RoutingPhase.PREFILL:
            # W47 switch breakdown (prefill stage timing only; no-op otherwise):
            # the admission / prefill-seed bookkeeping that primes the persistent
            # hot set for this layer's routed experts.  A host-side stage.
            with _stime.stage_nested("switch.admission"):
                self.runtime.prepare_prefill_seed(self.layer_index, expert_ids)
        # W64 (R3-pin): pin this layer's post-prefill working set (on the first
        # DECODE route after prefill, refreshed per MTPLX_DSV41_PIN_REFRESH_TOKENS)
        # and record the all-pinned-hit telemetry. No-op -- one env read -- unless
        # MTPLX_DSV41_PIN_WORKING_SET is armed, so the fenced decode path stays
        # byte-identical when the lever is off. It runs before the route below
        # executes, so this token's route already respects the fresh pins.
        # Guarded so a runtime double without the hook is unaffected.
        _pin_hook = getattr(self.runtime, "pin_working_set_hook", None)
        if callable(_pin_hook):
            _pin_hook(self.layer_index, expert_ids, phase)

        # W51 / K26 -- the prefill-only "dequantize once, matmul dense" expert path.
        # Gated to PREFILL (so the decode M=1 path is structurally excluded and stays
        # byte-identical) + native mxfp4 + the env flag, read at use.  Threaded into
        # ``_dispatch_component_bank`` from the split-route waves below.
        _dense_prefill_active = (
            phase is RoutingPhase.PREFILL
            and self.codec == "mxfp4"
            and _prefill_dense_experts_enabled()
        )

        outputs: list[mx.array] = []
        output_positions: list[int] = []
        # Seed from the hoist (above): when set, the shared branch is already
        # computed and submitted, so the late dispatch sites skip recomputing it.
        shared: mx.array | None = _hoisted_shared

        def update_fence_metrics(ready: ReadyRoute, **values: int) -> None:
            metrics = getattr(getattr(ready, "pool", None), "metrics", None)
            update = getattr(metrics, "update", None)
            if callable(update):
                update(**values)

        def synchronous_fence(ready: ReadyRoute, values: Any) -> None:
            if self.runtime.config.resource_telemetry:
                update_fence_metrics(
                    ready,
                    synchronous_fences=1,
                    synchronous_fence_slots=len(ready.bindings),
                )
            # W125 HIGH-1b: this blocking mx.eval is the routed-gather fence taken on
            # the shipped regime (all-hit + fenced split path); it is GPU execution
            # the host waits on, so the timeline records it as fence_total (subtracted
            # from host_gap). self.layer_index is in scope (synchronous_fence is a
            # closure of _run). No-op unless the probe is recording this decode token.
            _tl_f = _tl.now()
            try:
                try:
                    mx.eval(values)
                finally:
                    _tl.add_fence(self.layer_index, _tl_f)
            except BaseException as exc:
                update_fence_metrics(ready, completion_fence_failures=1)
                record = getattr(
                    getattr(ready, "pool", None),
                    "_record_completion_error",
                    None,
                )
                if callable(record):
                    record(exc)
                raise

        def fence_bindings(
            ready: ReadyRoute,
            bindings: tuple[ExpertSlotBinding, ...],
            wave_outputs: list[mx.array],
            *,
            force_sync: bool = False,
        ) -> None:
            raw = os.environ.get("MTPLX_EXPERT_SLOT_FENCES", "1")
            enabled = raw.strip().lower() not in {"0", "false", "no", "off"}
            async_eval = getattr(mx, "async_eval", None)
            if not enabled:
                synchronous_fence(ready, wave_outputs)
                return
            if not callable(async_eval):
                update_fence_metrics(ready, completion_fence_fallbacks=1)
                synchronous_fence(ready, wave_outputs)
                return
            if force_sync:
                update_fence_metrics(
                    ready,
                    completion_fences=1,
                    completion_fence_slots=len(bindings),
                )
                synchronous_fence(ready, wave_outputs)
                return
            try:
                async_eval(wave_outputs)
            except Exception:
                # Older/stripped MLX builds may expose the name without a
                # usable asynchronous evaluator. Preserve the generation
                # fence with the original synchronous barrier.
                update_fence_metrics(ready, completion_fence_fallbacks=1)
                synchronous_fence(ready, wave_outputs)
                return
            roots = tuple(wave_outputs)
            defer = getattr(ready, "defer_bindings_until", None)
            if not callable(defer):
                synchronous_fence(ready, roots)
                return
            try:
                defer(
                    bindings,
                    lambda: mx.eval(roots),
                )
            except Exception:
                # If the completion lane rejects or cannot represent this
                # binding set, do not release the route on an async promise.
                update_fence_metrics(ready, completion_fence_fallbacks=1)
                synchronous_fence(ready, roots)
                raise_completion_error = getattr(
                    getattr(ready, "pool", None),
                    "_raise_completion_error",
                    None,
                )
                if callable(raise_completion_error):
                    raise_completion_error()

        def evaluate_component_bindings(
            positions: tuple[int, ...] | list[int],
            bindings: tuple[ExpertSlotBinding, ...],
            ready: ReadyRoute,
            *,
            force_sync: bool = False,
            defer: bool = False,
        ) -> None:
            if not positions:
                return
            by_bank: dict[int, list[tuple[int, ExpertSlotBinding]]] = {}
            for global_position, binding in zip(positions, bindings, strict=True):
                by_bank.setdefault(id(binding.buffer.bank), []).append(
                    (global_position, binding)
                )
            wave_outputs: list[mx.array] = []
            wave_positions: list[int] = []
            for assignments in by_bank.values():
                grouped_positions = [position for position, _binding in assignments]
                grouped_bindings = tuple(binding for _position, binding in assignments)
                token_positions = mx.array(
                    [position // top_k for position in grouped_positions],
                    dtype=mx.int32,
                )
                selected = mx.take(tokens, token_positions, axis=0)
                wave_outputs.append(
                    self._dispatch_component_bank(
                        selected,
                        grouped_bindings,
                        dense_prefill=_dense_prefill_active,
                    )
                )
                wave_positions.extend(grouped_positions)
            if defer:
                # No release obligation here (the lease replay is queued),
                # but the GPU still needs the part submitted now — without
                # it the device idles through every miss wait.
                async_eval = getattr(mx, "async_eval", None)
                if callable(async_eval):
                    async_eval(wave_outputs)
                else:
                    mx.eval(wave_outputs)
            else:
                fence_bindings(
                    ready,
                    bindings,
                    wave_outputs,
                    force_sync=force_sync,
                )
            outputs.extend(wave_outputs)
            output_positions.extend(wave_positions)

        def evaluate_direct_bindings(
            positions: tuple[int, ...] | list[int],
            bindings: tuple[ExpertSlotBinding, ...],
            ready: ReadyRoute,
            *,
            force_sync: bool = False,
            defer: bool = False,
        ) -> None:
            if not positions:
                return
            by_expert: dict[int, list[int]] = {}
            binding_by_expert: dict[int, ExpertSlotBinding] = {}
            for global_position, binding in zip(positions, bindings, strict=True):
                by_expert.setdefault(binding.expert, []).append(global_position)
                binding_by_expert.setdefault(binding.expert, binding)
            wave_outputs: list[mx.array] = []
            wave_positions: list[int] = []
            for expert, expert_positions in by_expert.items():
                token_positions = mx.array(
                    [position // top_k for position in expert_positions],
                    dtype=mx.int32,
                )
                selected = mx.take(tokens, token_positions, axis=0)
                wave_outputs.append(
                    _run_q4_expert(
                        selected,
                        binding_by_expert[expert],
                        group_size=self.group_size,
                        bits=self.bits,
                        swiglu_limit=self.swiglu_limit,
                        codec=self.codec,
                    )
                )
                wave_positions.extend(expert_positions)
            if defer:
                # No release obligation here (the lease replay is queued),
                # but the GPU still needs the part submitted now — without
                # it the device idles through every miss wait.
                async_eval = getattr(mx, "async_eval", None)
                if callable(async_eval):
                    async_eval(wave_outputs)
                else:
                    mx.eval(wave_outputs)
            else:
                fence_bindings(
                    ready,
                    bindings,
                    wave_outputs,
                    force_sync=force_sync,
                )
            outputs.extend(wave_outputs)
            output_positions.extend(wave_positions)

        pipeline_ledger = getattr(self.runtime, "_pipeline_ledger", None)
        shared_pipeline_work = None
        if (
            pipeline_ledger is not None
            and shared_work is not None
            and phase is RoutingPhase.DECODE
        ):
            shared_pipeline_work = _begin_pipeline_work(
                pipeline_ledger,
                "begin_shared_work",
                phase=phase,
            )

        # W61 -- verify single-barrier fast path (env MTPLX_DSV41_VERIFY_SINGLE_
        # BARRIER, default ON; byte-identical -- see below).  A small-M DECODE
        # forward (the DSpark K+1 verify: 2..8 rows) otherwise splits its
        # rows*top_k assignments across several transient-bounded ``route_waves``,
        # each paying its own device->host fence -- window 25 measured a 4-row
        # verify at ~630 ms in moe.routed_switch (M=1: ~74 ms), several barriers
        # per layer.  But an ALL-HIT route lives entirely in PERSISTENT slots (no
        # transient bound), so the whole route can be pinned by ONE
        # ``try_all_hit_route`` and gathered in ONE wave over rows*top_k via the
        # K27 sorted ``gather_qmm``, released once (deferred / variant-B) -- exactly
        # ONE routing barrier (the ``mx.eval(indices)`` above) per layer.
        # ``gather_qmm`` is row-independent, so the single gather is bit-for-bit the
        # split+concat the loop below would produce (same token, same expert
        # weights per assignment); this is the existing all-hit branch applied to
        # one full wave.  A miss declines the all-hit probe and takes the W66 split
        # single-barrier path (below): when the whole route -- hits AND misses --
        # fits transient capacity in one transaction it is admitted in ONE
        # ``begin_split_route`` submission and gathered deferred, still exactly ONE
        # routing barrier.  A route whose unique experts exceed transient capacity
        # falls through to the bounded ``route_waves`` loop (unchanged), which
        # admits it in fenced capacity-bounded waves.  M=1 is excluded (kept
        # byte-for-byte).
        _verify_single_barrier = (
            os.environ.get("MTPLX_DSV41_VERIFY_SINGLE_BARRIER", "1") == "1"
            and phase is RoutingPhase.DECODE
            and self.runtime.config.slot_layout == "component-banks"
            and self.codec in ("affine", "mxfp4")
            and self._shadow_bank is None
            and 2 <= int(tokens.shape[0]) <= 8
            and int(len(expert_ids)) == int(tokens.shape[0]) * top_k
        )
        # W81 engagement census: a "verify shape" is any small-M (2..8 rows) DECODE
        # component-bank route -- the DSpark K+1 verify.  Count every such route as
        # a fast-path candidate so the ab census (and the next GPU window) can prove
        # engaged / declined with the decline reason, instead of inferring it from
        # cumulative hot.* counters.  Engagements are counted below
        # (hot.verify_single_barrier for W61 all-hit, hot.verify_single_barrier_split
        # for the W66/W81 batched split); here we tick the candidate + any decline.
        _verify_shape = (
            phase is RoutingPhase.DECODE
            and self.runtime.config.slot_layout == "component-banks"
            and 2 <= int(tokens.shape[0]) <= 8
        )
        if _verify_shape:
            _route_probe.count("hot.verify_candidate")
            if not _verify_single_barrier:
                if os.environ.get("MTPLX_DSV41_VERIFY_SINGLE_BARRIER", "1") != "1":
                    _route_probe.count("hot.verify_decline.flag_off")
                elif self.codec not in ("affine", "mxfp4"):
                    _route_probe.count("hot.verify_decline.codec")
                elif self._shadow_bank is not None:
                    _route_probe.count("hot.verify_decline.shadow_bank")
                elif int(len(expert_ids)) != int(tokens.shape[0]) * top_k:
                    _route_probe.count("hot.verify_decline.assignment_shape")
                else:
                    _route_probe.count("hot.verify_decline.other")
        if _verify_single_barrier:
            with _route_probe.bracket("hot.try_all_hit"):
                ready = self.runtime.try_all_hit_route(
                    self.layer_index, tuple(expert_ids), phase=phase
                )
            if ready is not None:
                _route_probe.count("hot.all_hit")
                _route_probe.count("hot.verify_single_barrier")
                hit_pipeline_work = None
                if pipeline_ledger is not None:
                    hit_pipeline_work = _begin_pipeline_work(
                        pipeline_ledger,
                        "begin_hit_work",
                        ready.plan.hits,
                        phase=phase,
                    )
                deferred_release = False
                try:
                    # positions == range(len(expert_ids)) for the full route, so
                    # the assignment inputs are the tokens broadcast over top_k --
                    # identical to the all-hit branch's broadcast case.
                    assignment_inputs = mx.broadcast_to(
                        tokens[:, None, :],
                        (int(tokens.shape[0]), top_k, hidden_size),
                    ).reshape((-1, hidden_size))
                    if pipeline_ledger is not None and hit_pipeline_work is not None:
                        _pipeline_work_call(
                            pipeline_ledger, hit_pipeline_work, "claim", phase=phase
                        )
                    # W92 dispatch census: the W61 (multi-row verify) all-hit gather
                    # is attributed to hot.allhit_gather_qmm too (a delta on the
                    # shared counter), so gather_qmm_per_all_hit_call stays 3 across
                    # BOTH the AR M=1 general-loop and the M=2..8 verify all-hit.
                    _qmm_before = _route_probe.peek("hot.switch_gather_qmm")
                    with _route_probe.bracket("hot.allhit_dispatch_build"):
                        wave_output = self._dispatch_component_bank(
                            assignment_inputs, ready.bindings
                        )
                    _route_probe.count(
                        "hot.allhit_gather_qmm",
                        _route_probe.peek("hot.switch_gather_qmm") - _qmm_before,
                    )
                    # Variant-B release: submit the one gather (non-blocking) so the
                    # GPU is fed, and DEFER the slot release to the next layer's
                    # routing barrier (its covering eval materializes this gather
                    # before the pinned slots are released -- the W42-proven, race-
                    # free deferred-pin mechanism; the slots stay pinned meanwhile,
                    # so unlike the W44 device route there is no unpinned recycle).
                    # This leaves exactly ONE blocking sync per layer (the barrier).
                    # Falls back to a single blocking fence only if the runtime
                    # cannot defer (fakes without the seam); either way it is ONE
                    # fence for the whole route, never one per split wave.
                    _verify_can_defer = callable(
                        getattr(self.runtime, "defer_slot_release", None)
                    ) and callable(
                        getattr(self.runtime, "flush_deferred_slot_releases", None)
                    )
                    if _deferred_pin_active or _verify_can_defer:
                        _async_eval = getattr(mx, "async_eval", None)
                        if callable(_async_eval):
                            _async_eval(wave_output)
                            _route_probe.count("hot.allhit_defer_submit")
                        self.runtime.defer_slot_release(ready, wave_output)
                        deferred_release = True
                        _route_probe.count("hot.allhit_defer")
                    else:
                        with _route_probe.bracket("hot.allhit_fence_eval"):
                            synchronous_fence(ready, wave_output)
                finally:
                    if pipeline_ledger is not None and hit_pipeline_work is not None:
                        _pipeline_work_call(
                            pipeline_ledger, hit_pipeline_work, "close", phase=phase
                        )
                    if not deferred_release:
                        ready.release(synchronize=False)
                output = wave_output.reshape((*indices.shape, hidden_size))
                if shared_work is not None and shared is None:
                    shared = shared_work()
                # All-hit route: no demand misses were submitted, so ordering is
                # vacuously satisfied -- issue the next layer's prefetch on the way
                # out (W93 issue-order fix; guarded to fire once).
                _issue_pending_gate_prefetch()
                return output, shared
            # W66/W81 split single-barrier (batched): this small-M DECODE verify
            # has misses, so W61's all-hit probe declined.  W66 covered the case
            # where the WHOLE route (hits + misses) fit transient capacity in ONE
            # ``begin_split_route`` -- one admission, one deferred gather, exactly
            # ONE routing barrier.  W81 keeps that and adds the wider case measured
            # on the real model (window 31): the DSpark 4-row verify routes
            # rows*top_k=24 assignments with ~20 unique experts, but the bench
            # runtime's transient capacity was ``spec.top_k`` (=6, the loader's
            # unset-``transient_slots`` default), so ~20 > 6 declined W66 and the
            # route fell to the bounded ``route_waves`` loop, which fences EVERY
            # transient-bounded split wave part (hit fence + a fence per miss part
            # + the shared fence) -- ~3.6 begin_split_route/layer-verify at 733 ms
            # in moe.routed_switch.  W81 instead partitions the route into the
            # SAME capacity-bounded waves ``route_waves`` would produce (identical
            # partition for DECODE: ``sort_unique`` off), and admits each wave as
            # ONE ``begin_split_route`` whose hits + all miss parts are gathered
            # deferred (async-submitted) and fenced by a SINGLE ``synchronous_fence``
            # -- ONE fence per batch, never one per wave part.  Pin-safety (W44 /
            # issue #120): the layer lock is not reentrant and non-final waves must
            # recycle their transient slots, so each NON-final wave materializes its
            # own gather (one fence) and releases (release_hits + release_miss +
            # close, freeing the lock) BEFORE the next wave re-enters; only the
            # FINAL wave defers its whole release to the next routing barrier's
            # covering eval (the W42/W61 deferred-pin proof -- pins held until the
            # eval, no unpinned recycle).  Net: (num_waves - 1) blocking fences + the
            # ONE routing barrier, versus the legacy loop's ~2x-that per-part fences;
            # unique <= capacity collapses to a single wave = the original W66
            # single-barrier promise.  Byte-identical to the bounded loop: the waves,
            # the per-assignment (token x expert-weight) gathers and the output-
            # position recombination are the same; only fence/release TIMING moves,
            # and the gather is row-independent.  A runtime without the defer/flush
            # seam (a fake double) leaves ``_verify_can_defer`` False and falls
            # through to the bounded loop unchanged.  M=1 never reaches here.
            _verify_can_defer = callable(
                getattr(self.runtime, "defer_slot_release", None)
            ) and callable(
                getattr(self.runtime, "flush_deferred_slot_releases", None)
            )
            # W87: the LIVE single-fence capacity -- runtime._batch_admission_slots()
            # is transient + non-pinned persistent capacity under the single slot
            # pool (shrinking with pins / a lowered derived cap), else the transient
            # scratch bound.  Falls back to plan.transient_slots on a runtime double
            # without the seam (byte-identical off).
            _bas = getattr(self.runtime, "_batch_admission_slots", None)
            if callable(_bas):
                _capacity = int(_bas())
            else:
                _plan_obj = getattr(self.runtime, "plan", None)
                _capacity = int(getattr(_plan_obj, "transient_slots", 0) or 0)
            if _verify_can_defer and _capacity >= 1:
                # Identical partition to the bounded loop (DECODE: sort_unique off);
                # unique <= capacity yields a single wave (the W66 fast path).
                _waves = tuple(self.runtime.route_waves(expert_ids))
                _final_wave = len(_waves) - 1
                _route_probe.count("hot.verify_single_barrier_split")
                if len(_waves) > 1:
                    _route_probe.count("hot.verify_single_barrier_batched")
                for _wave_index, _wave in enumerate(_waves):
                    _is_final = _wave_index == _final_wave
                    with _stime.stage_nested("switch.miss_submit"), \
                            _route_probe.bracket("hot.begin_split_route"):
                        _pending = self.runtime.begin_split_route(
                            self.layer_index,
                            _wave.experts,
                            phase=phase,
                        )
                    _wave_start = len(outputs)
                    _deferred_parts: list[ReadyRoute] = []
                    _fence_ready: ReadyRoute | None = None
                    _completed = False
                    try:
                        _hit_set = set(_pending.plan.hits)
                        if _pending.hit_ready is not None:
                            _hit_positions = tuple(
                                position
                                for position, expert in zip(
                                    _wave.positions, _wave.experts, strict=True
                                )
                                if expert in _hit_set
                            )
                            evaluate_component_bindings(
                                _hit_positions,
                                _pending.hit_ready.bindings,
                                _pending.hit_ready,
                                force_sync=True,
                                defer=True,
                            )
                            _fence_ready = _pending.hit_ready
                        # The resident shared branch depends only on ``x``.  Its
                        # graph can run while the native readers satisfy this
                        # verify wave's demand misses, rather than after the
                        # blocking miss iterator has drained.  Keep the result so
                        # the post-route once-only guard handles all-hit and
                        # non-decode routes without duplicating the branch.
                        if (
                            self._verify_shared_submit is not None
                            and shared_work is not None
                            and shared is None
                            and phase is RoutingPhase.DECODE
                            and _pending.misses_pending
                        ):
                            if (
                                pipeline_ledger is not None
                                and shared_pipeline_work is not None
                            ):
                                _pipeline_work_call(
                                    pipeline_ledger,
                                    shared_pipeline_work,
                                    "claim",
                                    phase=phase,
                                )
                            try:
                                shared = self._verify_shared_submit(shared_work)
                            finally:
                                if (
                                    pipeline_ledger is not None
                                    and shared_pipeline_work is not None
                                ):
                                    _pipeline_work_call(
                                        pipeline_ledger,
                                        shared_pipeline_work,
                                        "close",
                                        phase=phase,
                                    )
                                    shared_pipeline_work = None
                        for _miss_ready in _pending.iter_ready_misses():
                            _ready_experts = set(_miss_ready.plan.experts)
                            _miss_positions = tuple(
                                position
                                for position, expert in zip(
                                    _wave.positions, _wave.experts, strict=True
                                )
                                if expert not in _hit_set and expert in _ready_experts
                            )
                            evaluate_component_bindings(
                                _miss_positions,
                                _miss_ready.bindings,
                                _miss_ready,
                                force_sync=True,
                                defer=True,
                            )
                            _deferred_parts.append(_miss_ready)
                            if _fence_ready is None:
                                _fence_ready = _miss_ready
                        _completed = True
                    except BaseException as exc:
                        _pending.abort(exc)
                        raise
                    finally:
                        if not _completed:
                            _pending.close()
                        elif _is_final:
                            # One deferred release covers the final wave: the
                            # adapter replays release_miss + close (hit pins, miss
                            # leases, and the layer lock) one covering eval later.
                            self.runtime.defer_slot_release(
                                _DeferredSplitClose(
                                    _pending, tuple(_deferred_parts)
                                ),
                                tuple(outputs[_wave_start:]),
                            )
                        else:
                            # Non-final wave: ONE fence materializes this wave's
                            # gathers, then release recycles its transient slots and
                            # frees the (non-reentrant) layer lock for the next wave.
                            if _fence_ready is not None:
                                synchronous_fence(
                                    _fence_ready, tuple(outputs[_wave_start:])
                                )
                            if _pending.hit_ready is not None:
                                _pending.release_hits()
                            for _mr in _deferred_parts:
                                _pending.release_miss(_mr)
                            _pending.close()
                joined = mx.concatenate(outputs, axis=0)
                order = mx.argsort(mx.array(output_positions, dtype=mx.int32))
                joined = mx.take(joined, order, axis=0)
                output = joined.reshape((*indices.shape, hidden_size))
                if shared_work is not None and shared is None:
                    shared = shared_work()
                # W93 issue-order fix: every wave's begin_split_route has now
                # submitted this layer's demand misses; issue the next layer's
                # speculative prefetch after them (guarded to fire once).
                _issue_pending_gate_prefetch()
                return output, shared
            # No defer/flush seam (fake double): fall through to the bounded
            # route_waves loop below (byte-identical, fenced per wave part).
            _route_probe.count("hot.verify_decline.no_defer_seam")

        try:
            # W47 switch breakdown (prefill only; no-op otherwise): the host-side
            # route planning that groups this layer's expert ids into gather waves.
            with _stime.stage_nested("switch.route_plan"):
                waves = tuple(
                    self.runtime.route_waves(
                        expert_ids,
                        sort_unique=(
                            phase is RoutingPhase.PREFILL
                            and self.runtime.manifest.sidecar is not None
                        ),
                    )
                )
            # Deferring hands this layer's lock and slot pins to the next
            # generation-thread flush, so it is only legal on the wave that
            # ends the layer's routing.  A wide verify batch splits into
            # several waves, and the layer lock is not reentrant: a deferred
            # non-final wave would park the generation thread on its own lock
            # when the next wave re-entered the same layer (issue #120,
            # GLM depth>=4 at 6-row verify / 32 transient slots).
            final_wave = len(waves) - 1
            for wave_index, wave in enumerate(waves):
                # Shadow miss fallback (issue #51): with a shadow bank bound,
                # a decode wave never waits on SSD.  Resident experts run
                # exactly from cache; every other assignment is served from
                # the resident low-precision shadow bank, and the exact tier
                # warms asynchronously through the speculative-prefetch lane
                # so future routes turn these serves into exact hits.
                # Prefill is untouched (exact, transient-served).
                shadow_bank = self._shadow_bank
                resident: frozenset[int] = frozenset()
                if shadow_bank is not None and phase is RoutingPhase.DECODE:
                    resident = self.runtime.peek_resident_experts(
                        self.layer_index, wave.experts
                    )
                if (
                    shadow_bank is not None
                    and phase is RoutingPhase.DECODE
                    and not all(expert in resident for expert in wave.experts)
                ):
                    exact_assignments = tuple(
                        (position, expert)
                        for position, expert in zip(
                            wave.positions, wave.experts, strict=True
                        )
                        if expert in resident
                    )
                    shadow_positions = [
                        position
                        for position, expert in zip(
                            wave.positions, wave.experts, strict=True
                        )
                        if expert not in resident
                    ]
                    shadow_experts = [
                        expert for expert in wave.experts if expert not in resident
                    ]
                    if exact_assignments:
                        with _route_probe.bracket("hot.begin_shadow_route"):
                            pending = self.runtime.begin_split_route(
                                self.layer_index,
                                tuple(
                                    expert for _position, expert in exact_assignments
                                ),
                                phase=phase,
                            )
                        try:
                            hit_set = set(pending.plan.hits)
                            hit_positions = tuple(
                                position
                                for position, expert in exact_assignments
                                if expert in hit_set
                            )
                            if pending.hit_ready is not None:
                                evaluate_component_bindings(
                                    hit_positions,
                                    pending.hit_ready.bindings,
                                    pending.hit_ready,
                                    force_sync=True,
                                )
                                pending.release_hits()
                            # The residency peek is a snapshot, not a
                            # reservation: an expert evicted between peek and
                            # pin becomes a planned miss here and is serviced
                            # exactly (slower on this route, never wrong).
                            for miss_ready in pending.iter_ready_misses():
                                try:
                                    ready_experts = set(miss_ready.plan.experts)
                                    miss_positions = tuple(
                                        position
                                        for position, expert in exact_assignments
                                        if expert not in hit_set
                                        and expert in ready_experts
                                    )
                                    evaluate_component_bindings(
                                        miss_positions,
                                        miss_ready.bindings,
                                        miss_ready,
                                        force_sync=True,
                                    )
                                finally:
                                    pending.release_miss(miss_ready)
                        except BaseException as exc:
                            pending.abort(exc)
                            raise
                        finally:
                            pending.close()
                    with _route_probe.bracket("hot.shadow_serve"):
                        token_positions = mx.array(
                            [position // top_k for position in shadow_positions],
                            dtype=mx.int32,
                        )
                        selected = mx.take(tokens, token_positions, axis=0)
                        outputs.append(
                            _run_shadow_bank(
                                selected,
                                mx.array(shadow_experts, dtype=mx.int32),
                                shadow_bank,
                                swiglu_limit=self.swiglu_limit,
                            )
                        )
                        output_positions.extend(shadow_positions)
                    unique_shadowed = tuple(dict.fromkeys(shadow_experts))
                    _route_probe.count("hot.shadow_route")
                    self.runtime.note_shadow_serve(
                        self.layer_index,
                        assignments=len(shadow_positions),
                        experts=len(unique_shadowed),
                    )
                    self.runtime.prefetch_experts(
                        self.layer_index, unique_shadowed
                    )
                    continue
                # Keep the all-hit optimization inside the authoritative bounded
                # route-wave loop.  A successful probe avoids split-route futures
                # and per-expert grouping while retaining the normal policy epoch,
                # counters, and assignment order for this wave.
                if (
                    phase is RoutingPhase.DECODE
                    and self.runtime.config.slot_layout == "component-banks"
                ):
                    with _route_probe.bracket("hot.try_all_hit"):
                        ready = self.runtime.try_all_hit_route(
                            self.layer_index,
                            wave.experts,
                            phase=phase,
                        )
                    outcome = "all_hit" if ready is not None else "split_route"
                    _route_probe.count(f"hot.{outcome}")
                    _route_probe.count(
                        f"hot.{outcome}.{phase.name.lower()}"
                        f".layer{self.layer_index:02d}"
                    )
                    if ready is not None:
                        hit_pipeline_work = None
                        if pipeline_ledger is not None:
                            hit_pipeline_work = _begin_pipeline_work(
                                pipeline_ledger,
                                "begin_hit_work",
                                ready.plan.hits,
                                phase=phase,
                            )
                        deferred_release = False
                        try:
                            if wave.positions == tuple(range(len(expert_ids))):
                                assignment_inputs = mx.broadcast_to(
                                    tokens[:, None, :],
                                    (int(tokens.shape[0]), top_k, hidden_size),
                                ).reshape((-1, hidden_size))
                            else:
                                token_positions = mx.array(
                                    [position // top_k for position in wave.positions],
                                    dtype=mx.int32,
                                )
                                assignment_inputs = mx.take(
                                    tokens,
                                    token_positions,
                                    axis=0,
                                )
                            if (
                                pipeline_ledger is not None
                                and hit_pipeline_work is not None
                            ):
                                _pipeline_work_call(
                                    pipeline_ledger,
                                    hit_pipeline_work,
                                    "claim",
                                    phase=phase,
                                )
                            # W92 dispatch census: attribute ONLY this all-hit
                            # call's gather_qmm dispatches to hot.allhit_gather_qmm
                            # (a delta on the shared counter), so the receipt's
                            # per-all-hit-call ratio is exactly 3 (gate/up/down) and
                            # is not inflated by split parts / prefill waves that
                            # also flow through _gather_component_bank.
                            _qmm_before = _route_probe.peek("hot.switch_gather_qmm")
                            with _route_probe.bracket("hot.allhit_dispatch_build"):
                                wave_output = self._dispatch_component_bank(
                                    assignment_inputs,
                                    ready.bindings,
                                )
                            _route_probe.count(
                                "hot.allhit_gather_qmm",
                                _route_probe.peek("hot.switch_gather_qmm")
                                - _qmm_before,
                            )
                            # Slot pins may be released only after the lazy graph
                            # has consumed the currently bound bank generations.
                            # Deferred mode: the next generation-thread eval is
                            # that consumption proof; no per-layer fence runs.
                            deferred_release = False
                            # Promoted default via ExpertStreamingConfig after
                            # the C3 matrix; fakes without the field keep the
                            # fence path.  ``_deferred_pin_active`` also fires when
                            # the W42 fast-path (``MTPLX_DSV41_SWITCH_FASTPATH``)
                            # is armed on a runtime that can defer -- removing this
                            # per-all-hit-layer blocking ``mx.eval`` is the whole
                            # point of the fast-path.
                            # DECODE-only (self-guarding, not just the enclosing
                            # branch): a multi-group layer-major PREFILL has no
                            # covering eval between groups, so a deferred gather
                            # could read a slot the next group reloaded before its
                            # release -- always fence in prefill (W92 review).
                            if (
                                wave_index == final_wave
                                and _deferred_pin_active
                                and phase is RoutingPhase.DECODE
                            ):
                                # Variant B: submit the gather now (non-blocking)
                                # so the GPU is fed during the host graph-build of
                                # the next layers, instead of idling until the
                                # next barrier drains the backlog. Same array,
                                # same value -> byte-identical; the deferred
                                # release still waits for the barrier's covering
                                # eval. Only under the fast-path env (never for
                                # the hy3 config-deferred path, which pairs with
                                # its own submit cadence).
                                if _switch_submit and _fastpath_can_defer:
                                    _async_eval = getattr(mx, "async_eval", None)
                                    if callable(_async_eval):
                                        _async_eval(wave_output)
                                        _route_probe.count("hot.allhit_defer_submit")
                                self.runtime.defer_slot_release(ready, wave_output)
                                deferred_release = True
                                _route_probe.count("hot.allhit_defer")
                            else:
                                with _route_probe.bracket("hot.allhit_fence_eval"):
                                    synchronous_fence(ready, wave_output)
                            outputs.append(wave_output)
                            output_positions.extend(wave.positions)
                        finally:
                            if (
                                pipeline_ledger is not None
                                and hit_pipeline_work is not None
                            ):
                                _pipeline_work_call(
                                    pipeline_ledger,
                                    hit_pipeline_work,
                                    "close",
                                    phase=phase,
                                )
                            if not deferred_release:
                                ready.release(synchronize=False)
                        continue

                # Both layouts pin hits and start miss reads first, then run the
                # resident experts on the GPU while the misses stream from SSD.
                evaluate_bindings = (
                    evaluate_component_bindings
                    if self.runtime.config.slot_layout == "component-banks"
                    else evaluate_direct_bindings
                )
                # Deferred split release: no per-part fence; lease
                # bookkeeping replays at the next generation-thread eval
                # (the same coverage proof deferred pins already rely on).
                # A deferred split also keeps the layer lock until that
                # flush, so only the final wave may take this path.
                # ``_split_deferred_active`` / ``_deferred_pin_active`` fold in the
                # W42 fast-path promotion (``MTPLX_DSV41_SWITCH_FASTPATH``) so a
                # split layer's hit + miss waves dispatch via ``async_eval`` and
                # release at the next barrier, instead of one blocking
                # ``synchronous_fence`` per wave part.  Only the final wave may
                # defer (issue #120: the layer lock is not reentrant).
                deferred_split = (
                    wave_index == final_wave
                    and phase is RoutingPhase.DECODE
                    and _split_deferred_active
                    and _deferred_pin_active
                )
                deferred_parts: list[ReadyRoute] = []
                split_completed = False
                wave_output_start = len(outputs)
                # W47 switch breakdown (prefill only; no-op otherwise): submitting
                # the miss reads for this wave (host-side; the SSD reads then stream
                # asynchronously and the blocking wait is folded into the fenced
                # moe.routed_switch total, since reads overlap the gather).
                with _stime.stage_nested("switch.miss_submit"), \
                        _route_probe.bracket("hot.begin_split_route"):
                    pending = self.runtime.begin_split_route(
                        self.layer_index,
                        wave.experts,
                        phase=phase,
                    )
                # Fix (B) overlap telemetry (issue #130): measured
                # coactivity, never inferred from tok/s. The dispatch span
                # counts only when the miss reads were still open when the
                # GPU work went down; the exposed span is the residual
                # blocking wait the overlap could not hide.
                overlap_track = phase is RoutingPhase.DECODE and bool(
                    getattr(self.runtime.config, "overlap_miss_reads", False)
                )
                route_ready_ns = time.monotonic_ns() if overlap_track else 0
                dispatch_done_ns = 0
                dispatched_open = False
                first_miss_ready_ns = 0
                try:
                    hit_pipeline_work = None
                    if pipeline_ledger is not None and pending.hit_ready is not None:
                        hit_pipeline_work = _begin_pipeline_work(
                            pipeline_ledger,
                            "begin_hit_work",
                            pending.plan.hits,
                            phase=phase,
                        )
                    try:
                        hit_set = set(pending.plan.hits)
                        hit_positions = tuple(
                            position
                            for position, expert in zip(
                                wave.positions, wave.experts, strict=True
                            )
                            if expert in hit_set
                        )
                        if pending.hit_ready is not None:
                            # Split parts feed one shared lazy graph. Keep every MLX
                            # eval on the generation thread; evaluating a fence on the
                            # completion lane can race the next part's graph traversal.
                            if (
                                pipeline_ledger is not None
                                and hit_pipeline_work is not None
                            ):
                                _pipeline_work_call(
                                    pipeline_ledger,
                                    hit_pipeline_work,
                                    "claim",
                                    phase=phase,
                                )
                            evaluate_bindings(
                                hit_positions,
                                pending.hit_ready.bindings,
                                pending.hit_ready,
                                force_sync=True,
                                defer=deferred_split,
                            )
                    finally:
                        if (
                            pipeline_ledger is not None
                            and hit_pipeline_work is not None
                        ):
                            _pipeline_work_call(
                                pipeline_ledger,
                                hit_pipeline_work,
                                "close",
                                phase=phase,
                            )
                    if pending.hit_ready is not None and not deferred_split:
                        pending.release_hits()
                    # The resident shared branch depends only on ``x``.  Force it
                    # on Metal while the native readers own miss futures, so
                    # all-miss layers have useful GPU work instead of an empty
                    # device.  Keep prefill unchanged: at 128K, eagerly retaining
                    # the full shared output across routed waves would violate the
                    # bounded-memory execution contract.
                    if (
                        shared_work is not None
                        and shared is None
                        and phase is RoutingPhase.DECODE
                        and pending.misses_pending
                    ):
                        if (
                            pipeline_ledger is not None
                            and shared_pipeline_work is not None
                        ):
                            _pipeline_work_call(
                                pipeline_ledger,
                                shared_pipeline_work,
                                "claim",
                                phase=phase,
                            )
                        try:
                            shared = shared_work()
                            # Submit the shared branch so the GPU has work
                            # during miss I/O. Deferred-pin mode already
                            # trusts the next generation-thread eval for
                            # coverage, so the blocking barrier serves no
                            # release obligation there; strict mode keeps it.
                            async_eval = getattr(mx, "async_eval", None)
                            if _deferred_pin_active and callable(async_eval):
                                async_eval(shared)
                            else:
                                mx.eval(shared)
                        finally:
                            if (
                                pipeline_ledger is not None
                                and shared_pipeline_work is not None
                            ):
                                _pipeline_work_call(
                                    pipeline_ledger,
                                    shared_pipeline_work,
                                    "close",
                                    phase=phase,
                                )
                                shared_pipeline_work = None
                    if overlap_track:
                        dispatch_done_ns = time.monotonic_ns()
                        dispatched_open = pending.misses_pending
                    for miss_ready in pending.iter_ready_misses():
                        if overlap_track and first_miss_ready_ns == 0:
                            first_miss_ready_ns = time.monotonic_ns()
                        part_error: BaseException | None = None
                        try:
                            ready_experts = set(miss_ready.plan.experts)
                            miss_positions = tuple(
                                position
                                for position, expert in zip(
                                    wave.positions, wave.experts, strict=True
                                )
                                if expert not in hit_set and expert in ready_experts
                            )
                            if pipeline_ledger is not None:
                                _pipeline_work_call(
                                    pipeline_ledger,
                                    pending,
                                    "claim_misses",
                                    miss_ready,
                                    phase=phase,
                                )
                            evaluate_bindings(
                                miss_positions,
                                miss_ready.bindings,
                                miss_ready,
                                force_sync=True,
                                defer=deferred_split,
                            )
                        except BaseException as exc:
                            part_error = exc
                            raise
                        finally:
                            if deferred_split and part_error is None:
                                deferred_parts.append(miss_ready)
                            else:
                                try:
                                    pending.release_miss(miss_ready)
                                except BaseException:
                                    if part_error is None:
                                        raise
                    if overlap_track and pending.plan.misses:
                        self.runtime.slots.metrics.update(
                            overlap_split_routes=1,
                            overlap_gpu_dispatch_ns=(
                                dispatch_done_ns - route_ready_ns
                                if dispatched_open
                                else 0
                            ),
                            overlap_exposed_wait_ns=(
                                max(0, first_miss_ready_ns - dispatch_done_ns)
                                if first_miss_ready_ns
                                else 0
                            ),
                        )
                    split_completed = True
                except BaseException as exc:
                    pending.abort(exc)
                    raise
                finally:
                    if deferred_split and split_completed:
                        self.runtime.defer_slot_release(
                            _DeferredSplitClose(
                                pending, tuple(deferred_parts)
                            ),
                            tuple(outputs[wave_output_start:]),
                        )
                    else:
                        pending.close()

            if not outputs:
                raise ValueError("router produced no expert assignments")
            if len(outputs) == 1 and output_positions == list(range(len(expert_ids))):
                joined = outputs[0]
            else:
                joined = mx.concatenate(outputs, axis=0)
                order = mx.argsort(mx.array(output_positions, dtype=mx.int32))
                joined = mx.take(joined, order, axis=0)
            output = joined.reshape((*indices.shape, hidden_size))
            if shared_work is not None and shared is None:
                # No physical wait remained to hide, or this was a bounded-memory
                # prefill. Preserve the original routed-then-shared ordering.
                if pipeline_ledger is not None and shared_pipeline_work is not None:
                    _pipeline_work_call(
                        pipeline_ledger,
                        shared_pipeline_work,
                        "claim",
                        phase=phase,
                    )
                try:
                    if shared is None:
                        shared = shared_work()
                finally:
                    if pipeline_ledger is not None and shared_pipeline_work is not None:
                        _pipeline_work_call(
                            pipeline_ledger,
                            shared_pipeline_work,
                            "close",
                            phase=phase,
                        )
                        shared_pipeline_work = None
            # W93 issue-order fix: the bounded route_waves loop (and the shadow
            # path) has submitted every demand miss for this layer via
            # begin_split_route above; issue the next layer's speculative prefetch
            # only now, after the demand reads (guarded to fire once).
            _issue_pending_gate_prefetch()
            return output, shared
        finally:
            if pipeline_ledger is not None and shared_pipeline_work is not None:
                _pipeline_work_call(
                    pipeline_ledger,
                    shared_pipeline_work,
                    "close",
                    phase=phase,
                )


def run_switch_with_shared_overlap(
    switch_mlp: Any,
    x: mx.array,
    indices: mx.array,
    shared_work: Callable[[], mx.array],
) -> tuple[mx.array, mx.array]:
    """Use streamed miss overlap when supported, otherwise preserve ordering."""

    overlap = getattr(switch_mlp, "run_with_shared_overlap", None)
    if callable(overlap):
        return overlap(x, indices, shared_work)
    return switch_mlp(x, indices), shared_work()


def _issue_gate_prefetch(runtime: Any, pending: tuple, *, verify: bool = False) -> None:
    """Hand a stashed W93 gate-oracle prediction to the speculative ring.

    ``pending`` is ``(next_layer, predicted_ids_array)``; the array was already
    materialized on the switch's indices barrier, so ``.tolist()`` here is a host
    read with no new sync. Best-effort: a runtime without the ring (or a
    prediction the ring drops as resident/inflight) costs only the prediction.

    ``verify`` (W100) marks a DSpark verify-phase issue so the runtime attributes
    ``prefetch_issued_verify`` / ``prefetch_committed_verify`` separately from the
    AR (M=1) total -- the caller (the switch) knows the phase; this only forwards
    the flag, it never changes which ids issue."""

    next_layer, predicted = pending
    prefetch = getattr(runtime, "prefetch_experts", None)
    if prefetch is None:
        return
    # W95: drop the confidence-gated -1 sentinels (rank 6..k below the margin
    # threshold are trimmed to -1 by ``gate_predict_topk``) and dedup across the
    # verify's per-row union, preserving first-seen order. Without the drop the
    # ring's ``_key`` rejects -1 (``expert id must be at least 0``) and kills the
    # decode step; ``prefetch_experts`` further skips already-resident ids, so the
    # issued speculative set is {gated, non-resident} experts.
    ids = list(
        dict.fromkeys(
            v for v in (int(value) for value in predicted.reshape(-1).tolist()) if v >= 0
        )
    )
    note = getattr(runtime, "note_gate_prefetch_predicted", None)
    if callable(note):
        note(next_layer, len(ids))
    try:
        prefetch(next_layer, ids, verify=verify)
    except TypeError:
        # A runtime whose prefetch_experts predates the W100 verify kwarg (older
        # build / minimal fake): fall back to the phase-agnostic call so the
        # gate-oracle prefetch still issues; only the verify attribution is lost.
        prefetch(next_layer, ids)


def bind_streamed_switches(model: Any, runtime: ExpertStreamingRuntime) -> int:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError("model does not expose transformer layers")
    bound = 0
    mapped_store = None
    if runtime.config.slot_layout == "metal-mmap":
        workers_text = os.environ.get("MTPLX_MMAP_WORKERS", "96")
        try:
            workers = int(workers_text)
        except ValueError as exc:
            raise ValueError("MTPLX_MMAP_WORKERS must be an integer") from exc
        mapped_store = MappedExpertStore(
            runtime.root,
            runtime.manifest,
            workers=workers,
        )
        mapped_store.prepare()
        runtime._mapped_expert_store = mapped_store
    island_layers = frozenset(runtime.config.island_layers)
    island_store = None
    if island_layers:
        island_store = DenseIslandStore(
            runtime.manifest,
            island_layers,
            expert_count=runtime.spec.expert_count,
        )
        island_store.fill(
            runtime.manifest,
            runtime.reader,
            verify_hash=(
                runtime.config.verify_record_hashes
                and not runtime.config.verify_sidecar_hash_at_open
            ),
        )
        runtime._island_store = island_store
    banked_layers = frozenset(runtime.config.mmap_island_layers)
    banked_store = None
    if banked_layers:
        banked_store = BankedMmapIslandStore(
            Path(runtime.config.banked_manifest),
            banked_layers,
            expert_count=runtime.spec.expert_count,
            wired=runtime.config.mmap_island_wired,
        )
        banked_store.prepare()
        banked_store.prefetch_all()
        runtime._banked_island_store = banked_store
    for layer_index in runtime.spec.routed_layer_indices:
        layer = layers[layer_index]
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "switch_mlp"):
            raise TypeError(f"layer {layer_index} has no switch_mlp seam")
        if island_store is not None and layer_index in island_layers:
            mlp.switch_mlp = DenseIslandSwitchGLU(
                runtime,
                island_store,
                layer_index,
            )
        elif banked_store is not None and layer_index in banked_layers:
            mlp.switch_mlp = DenseIslandSwitchGLU(
                runtime,
                banked_store,
                layer_index,
            )
        elif mapped_store is None:
            mlp.switch_mlp = HotExpertSwitchGLU(runtime, layer_index)
        else:
            mlp.switch_mlp = MappedExpertSwitchGLU(
                runtime,
                mapped_store,
                layer_index,
            )
        bound += 1
    # W93: wire each routed layer's MoE to the next layer's gate for one-ahead
    # gate-oracle prefetch. DSV4.1-only + self-guarding (non-DSV4.1 trunks carry
    # no `mlp.gate` with a `score_func`, so nothing is installed for them), and
    # inert until MTPLX_DSV41_GATE_PREFETCH arms the ring.
    from .deepseek_v41 import install_gate_prefetch_links

    install_gate_prefetch_links(model, runtime)
    return bound
