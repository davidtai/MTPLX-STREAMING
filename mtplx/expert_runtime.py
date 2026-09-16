"""Model-independent orchestration for bounded SSD expert streaming."""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from contextlib import nullcontext
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .expert_io import ExpertIOError, PositionalExpertReader
# W125 host-side decode timeline probe (env MTPLX_DSV41_DECODE_TIMELINE=1). No-op
# import when unset; the marks below self-gate on the single-row decode window.
from . import dsv41_decode_timeline as _tl
from .expert_manifest import (
    ExpertManifest,
    ExpertManifestError,
    load_expert_manifest,
    validate_expert_manifest_spec,
    verify_expert_manifest,
)
from .expert_slots import (
    ExpertCompletionFenceError,
    ExpertSlotError,
    ExpertSlotPool,
    ReadyRoute,
    RouteIOAdmission,
)
from .expert_streaming import (
    CacheCounters,
    GlobalExpertSlotBank,
    GlobalPrefetchRing,
    LayerExpertSlotBank,
    RoutePlan,
    RoutePolicyTxn,
    RoutingPhase,
    TRANSITION_WINDOW_CACHE_POLICY,
)
from .expert_streaming_models import (
    MIXED_OFFICIAL_CODEC,
    ExpertMemoryPlan,
    ExpertStreamingModelSpec,
    get_model_spec,
    plan_expert_memory,
    proj_quant_covers,
    affine_quant_kept_bytes,
)
from .optimization_profiles import (
    ENFORCEABLE_KNOBS,
    not_applicable_violations,
)
from .resource_metrics import ExpertPipelineLedger, ExpertPipelineRoute
from .belady_oracle import BeladyOracle
from .route_census import (
    ISLAND_PLACEMENT_FILENAME,
    ROUTE_CENSUS_FILENAME,
    RouteCensus,
    RouteCensusError,
    derive_placement,
    load_census,
    load_placement,
    save_census,
    save_placement,
)


_LOGGER = logging.getLogger(__name__)

_MEMORY_RE = re.compile(r"^([0-9]+)([kmgt]i?b?|b)?$", re.IGNORECASE)


class ExpertStreamingConfigurationError(ValueError):
    pass


def validate_deepseek_v41_runtime_env(model_key: str) -> None:
    """Reject unvalidated full-model lanes before load-time caps or allocations.

    Read the actual process environment used by cache construction. Isolated cache
    classes remain available for numerical investigation; this is installation-only.
    """
    if model_key.startswith("deepseek-v41") and (
        os.environ.get("MTPLX_DSV41_KV_BOUNDED") or ""
    ).strip().lower() in ("1", "true", "yes", "on"):
        raise ExpertStreamingConfigurationError(
            "MTPLX_DSV41_KV_BOUNDED has unvalidated Metal parity and cannot "
            "be installed for DeepSeek-V4.1 streaming; disable it before "
            "loading. Use isolated cache tests to investigate the lane."
        )


def _pipeline_call(
    ledger: ExpertPipelineLedger | None,
    route: ExpertPipelineRoute,
    method: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Publish optional diagnostics without changing runtime outcomes."""

    try:
        getattr(route, method)(*args, **kwargs)
    except Exception:
        if ledger is not None:
            try:
                ledger.mark_incomplete(phase=route.phase)
            except Exception:
                pass


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def parse_memory_bytes(value: str | int) -> int:
    if isinstance(value, bool):
        raise ExpertStreamingConfigurationError("memory size must not be bool")
    if isinstance(value, int):
        if value <= 0:
            raise ExpertStreamingConfigurationError("memory size must be positive")
        return value
    if not isinstance(value, str):
        raise ExpertStreamingConfigurationError(
            "memory size must be bytes or a suffixed string"
        )
    normalized = value.strip().lower()
    match = _MEMORY_RE.fullmatch(normalized)
    if match is None:
        raise ExpertStreamingConfigurationError(f"invalid memory size {value!r}")
    number = int(match.group(1))
    suffix = (match.group(2) or "b").lower()
    multipliers = {
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
        "tib": 1024**4,
    }
    result = number * multipliers[suffix]
    if result <= 0:
        raise ExpertStreamingConfigurationError("memory size must be positive")
    return result


@dataclass(frozen=True)
class ExpertStreamingConfig:
    model_key: str
    memory_limit_bytes: int
    max_live_kv_tokens: int
    runtime_reserve_bytes: int = 16 * 1024**3
    expert_cache_limit_bytes: int | None = None
    transient_slots: int | None = None
    io_staging_bytes: int = 0
    execution_workspace_bytes: int = 0
    max_inflight_io_bytes: int | None = None
    max_open_files: int = 16
    max_read_chunk_bytes: int = 8 * 1024 * 1024
    # W24 R4 (Factor D) lever, default 1 (OFF): split each large record's
    # positional read into N concurrent contiguous sub-reads to raise SSD queue
    # depth on the serially-layer-dependent decode path. Byte-identical.
    io_read_fanout: int = 1
    frequency_decay: float = 0.995
    prefer_sidecar: bool = True
    verify_record_hashes: bool = True
    verify_artifact_headers: bool = True
    verify_sidecar_hash_at_open: bool = False
    prefill_admission: bool = False
    # ``None`` (the default, i.e. the user set no --expert-slot-layout) derives the
    # layout from the streamed record codec in __post_init__: a non-affine artifact
    # (mxfp4/shadow/mixed-official) can only be read by the component-banks dispatch,
    # every other dispatch assumes the affine triple; affine keeps the historical
    # direct-slots default.  An explicit value is honoured verbatim.
    slot_layout: str | None = None
    trace_routes: bool = False
    cache_policy: str = "frequency"
    cache_scope: str = "layer"
    bypass_page_cache: bool = False
    resource_telemetry: bool = False
    q2_expert_kernel: str = "stock"
    hy3_router_kernel: str = "mpp-r1-fused-r2"
    hy3_router_sigmoid: str = "precise"
    hy3_mtp_shared_kernel: str = "stock"
    hy3_mtp_shared_kernel_depth: int = 3
    deferred_pin_release: bool = False
    island_layers: tuple[int, ...] = ()
    island_layer_count: int | None = None
    mmap_island_layers: tuple[int, ...] = ()
    banked_manifest: str | None = None
    banked_codec: str = "none"
    # Streamed compressed sidecar (issue #113): when "rans32x-v1", streamed
    # miss reads pull per-record rANS containers and decode them in-kernel
    # before slot residency -- fewer bytes/token off SSD at zero quality cost.
    # "none" leaves every streamed read byte-unchanged.
    streamed_codec: str = "none"
    streamed_codec_manifest: str | None = None
    # Post-decode sha256 of rANS-decoded records. Default ON until the 16k
    # long-context validation passes (David 2026-07-17); the container's own
    # structural guards remain regardless.
    streamed_codec_verify: bool = True
    mmap_island_wired: bool = True
    proj_quant: str | None = None
    # EXPERIMENT knob (oq2e resident arm): re-quantize residents that already
    # loaded quantized (e.g. q8/gs64) DOWN to a lower-bit affine format. This
    # is a separate mechanism from proj_quant (which only touches BF16
    # Linears) and does NOT trip its not_applicable enforcement — q8 -> q4
    # double quantization is deliberate. Only "q4" is sanctioned.
    proj_requant: str | None = None
    kv_quant: str | None = None
    split_route_release: str = "fenced"
    # Fix (B), issue #130: submit a decode route's misses as ONE batched
    # part (single future, admission before the wait, adjacent records
    # coalesced into scatter reads) so resident-expert compute overlaps the
    # miss reads. OFF leaves the per-expert split-part path byte-identical.
    overlap_miss_reads: bool = False
    # DeepSeek-V4.1 decode scheduling arm.  When overlap_miss_reads is enabled,
    # cap each independently completable miss part to this many records while
    # still submitting every part at once.  ``None`` keeps the current one-part
    # layer batch.  The cap is construction-selected so the enabled hot path
    # never probes or falls back.
    decode_miss_records_per_part: int | None = None
    # DeepSeek-V4.1 verify-only scheduling arm: after a split route has submitted
    # its demand misses, enqueue the resident shared MLP before waiting for those
    # reads. Installed once at switch construction; OFF preserves the unchanged
    # routed-then-shared control ordering.
    verify_shared_overlap: bool = False
    prefetch_slots: int = 0
    speculative_io_fraction: float = 0.25
    route_census: bool = True
    miss_shadow: str | None = None
    miss_shadow_layers: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_key, str) or not self.model_key:
            raise TypeError("model_key must be a non-empty string")
        # Slot-layout default derives from the streamed record codec, in the one
        # place both the loader (build_streaming_config) and the serve path pass
        # through (this __post_init__).  When the user set no explicit layout, a
        # non-affine artifact (mxfp4/shadow/mixed-official) defaults to the
        # component-banks dispatch — the only one that carries the codec branch;
        # the direct-slot / mapped / dense-island dispatches assume the affine
        # triple and would misread the record.  An explicit direct-slots stays
        # verbatim and ExpertStreamingRuntime.open rejects it loudly for a
        # non-affine artifact.  Unregistered synthetic keys keep the affine
        # default (their spec is carried into open() directly).
        if self.slot_layout is None:
            try:
                codec = get_model_spec(self.model_key).expert_codec
            except ValueError:
                codec = "affine"
            object.__setattr__(
                self,
                "slot_layout",
                "component-banks" if codec != "affine" else "direct-slots",
            )
        for name, minimum in (
            ("memory_limit_bytes", 1),
            ("max_live_kv_tokens", 0),
            ("runtime_reserve_bytes", 0),
            ("io_staging_bytes", 0),
            ("execution_workspace_bytes", 0),
            ("max_open_files", 1),
            ("max_read_chunk_bytes", 1),
            ("io_read_fanout", 1),
        ):
            object.__setattr__(
                self, name, _integer(name, getattr(self, name), minimum=minimum)
            )
        for name in (
            "expert_cache_limit_bytes",
            "transient_slots",
            "max_inflight_io_bytes",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _integer(name, value, minimum=0))
        if self.max_inflight_io_bytes == 0:
            raise ValueError("max_inflight_io_bytes must be positive when supplied")
        if isinstance(self.frequency_decay, bool):
            raise TypeError("frequency_decay must be numeric")
        decay = float(self.frequency_decay)
        if not 0.0 < decay <= 1.0:
            raise ValueError("frequency_decay must be in (0, 1]")
        object.__setattr__(self, "frequency_decay", decay)
        if self.cache_policy not in {
            "frequency",
            "lru",
            TRANSITION_WINDOW_CACHE_POLICY,
        }:
            raise ValueError(
                "cache_policy must be 'frequency', 'lru', or "
                f"{TRANSITION_WINDOW_CACHE_POLICY!r}"
            )
        if self.cache_scope not in {"layer", "global"}:
            raise ValueError("cache_scope must be 'layer' or 'global'")
        if self.q2_expert_kernel not in {
            "stock",
            "nax",
            "fused",
            "fused-nax",
        }:
            raise ValueError(
                "q2_expert_kernel must be 'stock', 'nax', 'fused', or 'fused-nax'"
            )
        if self.hy3_router_kernel not in {
            "stock",
            "steel-r1-fused-r2",
            "mpp-r1-fused-r2",
            "mpp-fp32-splitk-r1-fused-r2",
            "mpp-r1-last-arrival-fused-r2",
            "mpp-row-owned-fused",
        }:
            raise ValueError(
                "hy3_router_kernel must be 'stock', 'steel-r1-fused-r2', "
                "'mpp-r1-fused-r2', 'mpp-fp32-splitk-r1-fused-r2', "
                "'mpp-r1-last-arrival-fused-r2', or 'mpp-row-owned-fused'"
            )
        if not isinstance(self.deferred_pin_release, bool):
            raise ValueError("deferred_pin_release must be bool")
        if self.proj_quant not in {None, "q8", "q4"}:
            raise ValueError("proj_quant must be None, 'q8', or 'q4'")
        # q8 -> q4 is the only sanctioned requant experiment; reject "q8"
        # (no-op) and anything else so a typo cannot silently pass through.
        if self.proj_requant not in {None, "q4"}:
            raise ValueError("proj_requant must be None or 'q4'")
        if self.kv_quant not in {None, "q8", "q4"}:
            raise ValueError("kv_quant must be None, 'q8', or 'q4'")
        if self.miss_shadow is not None:
            if self.miss_shadow not in {"b1", "t158"}:
                raise ValueError("miss_shadow must be None, 'b1', or 't158'")
            if self.cache_scope != "layer":
                raise ValueError("miss_shadow requires cache_scope 'layer'")
            if self.slot_layout != "component-banks":
                raise ValueError(
                    "miss_shadow requires the component-banks slot layout"
                )
        if self.miss_shadow_layers is not None:
            if self.miss_shadow is None:
                raise ValueError(
                    "miss_shadow_layers requires miss_shadow to be set"
                )
            object.__setattr__(
                self,
                "miss_shadow_layers",
                _integer(
                    "miss_shadow_layers", self.miss_shadow_layers, minimum=1
                ),
            )
        if self.split_route_release not in {"fenced", "deferred"}:
            raise ValueError(
                "split_route_release must be 'fenced' or 'deferred'"
            )
        if not isinstance(self.overlap_miss_reads, bool):
            raise ValueError("overlap_miss_reads must be bool")
        if self.overlap_miss_reads and self.slot_layout != "component-banks":
            raise ValueError(
                "overlap_miss_reads requires the component-banks slot layout"
            )
        if self.decode_miss_records_per_part is not None:
            object.__setattr__(
                self,
                "decode_miss_records_per_part",
                _integer(
                    "decode_miss_records_per_part",
                    self.decode_miss_records_per_part,
                    minimum=1,
                ),
            )
            if not self.model_key.startswith("deepseek-v41"):
                raise ValueError(
                    "decode_miss_records_per_part is only installed for "
                    "DeepSeek-V4.1"
                )
            if not self.overlap_miss_reads:
                raise ValueError(
                    "decode_miss_records_per_part requires overlap_miss_reads"
                )
            if self.slot_layout != "component-banks":
                raise ValueError(
                    "decode_miss_records_per_part requires the component-banks "
                    "slot layout"
                )
        if not isinstance(self.verify_shared_overlap, bool):
            raise ValueError("verify_shared_overlap must be bool")
        if self.verify_shared_overlap:
            if not self.model_key.startswith("deepseek-v41"):
                raise ValueError(
                    "verify_shared_overlap is only installed for DeepSeek-V4.1"
                )
            if self.slot_layout != "component-banks":
                raise ValueError(
                    "verify_shared_overlap requires the component-banks slot layout"
                )
        object.__setattr__(
            self,
            "prefetch_slots",
            _integer("prefetch_slots", self.prefetch_slots, minimum=0),
        )
        if self.prefetch_slots and self.cache_scope != "layer":
            raise ValueError("prefetch_slots require cache_scope 'layer'")
        if self.cache_policy == TRANSITION_WINDOW_CACHE_POLICY:
            if not self.model_key.startswith("deepseek-v41"):
                raise ValueError(
                    "transition-window cache policy is only installed for "
                    "DeepSeek-V4.1"
                )
            if self.cache_scope != "layer":
                raise ValueError(
                    "transition-window cache policy requires cache_scope 'layer'"
                )
            if self.slot_layout != "component-banks":
                raise ValueError(
                    "transition-window cache policy requires the component-banks "
                    "slot layout"
                )
            if self.prefetch_slots:
                raise ValueError(
                    "transition-window cache policy cannot be combined with "
                    "speculative prefetch"
                )
        if isinstance(self.speculative_io_fraction, bool) or not isinstance(
            self.speculative_io_fraction, (int, float)
        ):
            raise TypeError("speculative_io_fraction must be a number")
        fraction = float(self.speculative_io_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("speculative_io_fraction must be in (0, 1]")
        object.__setattr__(self, "speculative_io_fraction", fraction)
        if self.island_layer_count is not None:
            if self.island_layers:
                raise ValueError(
                    "island_layers and island_layer_count are mutually "
                    "exclusive; give the count OR the explicit list"
                )
            count = _integer(
                "island_layer_count", self.island_layer_count, minimum=1
            )
            object.__setattr__(self, "island_layer_count", count)
            try:
                spec = get_model_spec(self.model_key)
            except ValueError:
                # Unregistered model keys carry their spec into open()
                # directly; count resolution defers there with the rest of
                # the placement precedence chain.
                spec = None
            if spec is not None and spec.island_pin_order:
                if count > len(spec.island_pin_order):
                    raise ValueError(
                        f"island_layer_count {count} exceeds the "
                        f"{self.model_key} pin order "
                        f"({len(spec.island_pin_order)} layers)"
                    )
                object.__setattr__(
                    self,
                    "island_layers",
                    tuple(sorted(spec.island_pin_order[:count])),
                )
            # else: unresolved (island_layers stays empty while the count is
            # set). resolve_island_placement() maps the count to layers from
            # the model root's island-placement.json; precedence is explicit
            # island_layers > spec.island_pin_order > placement file > error.
        island_layers = self.island_layers
        if isinstance(island_layers, (str, bytes)) or not isinstance(
            island_layers, (tuple, list)
        ):
            raise TypeError("island_layers must be a tuple of layer indices")
        normalized_islands = tuple(
            sorted(
                {
                    _integer("island_layers entry", layer, minimum=0)
                    for layer in island_layers
                }
            )
        )
        object.__setattr__(self, "island_layers", normalized_islands)
        if normalized_islands:
            if self.cache_scope != "layer":
                raise ValueError("island_layers require cache_scope 'layer'")
            if self.slot_layout != "component-banks":
                raise ValueError(
                    "island_layers require the component-banks slot layout"
                )
            if self.trace_routes:
                raise ValueError(
                    "island_layers execute without host route observation; "
                    "trace_routes must be disabled"
                )
        if not isinstance(self.mmap_island_wired, bool):
            raise TypeError("mmap_island_wired must be bool")
        mmap_islands = self.mmap_island_layers
        if isinstance(mmap_islands, (str, bytes)) or not isinstance(
            mmap_islands, (tuple, list)
        ):
            raise TypeError("mmap_island_layers must be a tuple of layer indices")
        normalized_mmap = tuple(
            sorted(
                {
                    _integer("mmap_island_layers entry", layer, minimum=0)
                    for layer in mmap_islands
                }
            )
        )
        object.__setattr__(self, "mmap_island_layers", normalized_mmap)
        if self.banked_codec not in {"none", "rans32x-v1"}:
            raise ValueError(
                "banked_codec must be 'none' or 'rans32x-v1'"
            )
        if self.streamed_codec not in {"none", "rans32x-v1"}:
            raise ValueError(
                "streamed_codec must be 'none' or 'rans32x-v1'"
            )
        if not isinstance(self.streamed_codec_verify, bool):
            raise TypeError("streamed_codec_verify must be bool")
        if self.streamed_codec != "none":
            if self.streamed_codec_manifest is None:
                raise ValueError(
                    "streamed_codec requires a streamed_codec_manifest path"
                )
            if self.slot_layout == "metal-mmap":
                raise ValueError(
                    "streamed_codec decodes compressed records into slots; it is "
                    "incompatible with the metal-mmap zero-copy layout"
                )
            if self.decode_miss_records_per_part is not None:
                raise ValueError(
                    "decode_miss_records_per_part requires raw sidecar records; "
                    "disable streamed_codec for this scheduling arm"
                )
        elif self.streamed_codec_manifest is not None:
            raise ValueError(
                "streamed_codec_manifest requires streamed_codec 'rans32x-v1'"
            )
        if normalized_mmap:
            if self.banked_manifest is None:
                raise ValueError(
                    "mmap_island_layers require a banked_manifest path"
                )
            if self.cache_scope != "layer":
                raise ValueError(
                    "mmap_island_layers require cache_scope 'layer'"
                )
            if self.slot_layout != "component-banks":
                raise ValueError(
                    "mmap_island_layers require the component-banks slot layout"
                )
            if self.trace_routes:
                raise ValueError(
                    "mmap_island_layers execute without host route observation; "
                    "trace_routes must be disabled"
                )
            overlap = set(normalized_mmap) & set(normalized_islands)
            if overlap:
                raise ValueError(
                    "island_layers and mmap_island_layers must be disjoint; "
                    f"both claim {sorted(overlap)}"
                )
        if self.hy3_router_sigmoid not in {"precise", "fast"}:
            raise ValueError("hy3_router_sigmoid must be 'precise' or 'fast'")
        if (
            self.hy3_router_sigmoid == "fast"
            and self.hy3_router_kernel != "mpp-row-owned-fused"
        ):
            raise ValueError(
                "fast router sigmoid is selectable only with mpp-row-owned-fused"
            )
        if self.hy3_mtp_shared_kernel not in {"stock", "metal-exact"}:
            raise ValueError(
                "hy3_mtp_shared_kernel must be 'stock' or 'metal-exact'"
            )
        object.__setattr__(
            self,
            "hy3_mtp_shared_kernel_depth",
            _integer(
                "hy3_mtp_shared_kernel_depth",
                self.hy3_mtp_shared_kernel_depth,
                minimum=1,
            ),
        )
        if self.slot_layout not in {
            "direct-slots",
            "component-banks",
            "metal-mmap",
        }:
            raise ValueError(
                "slot_layout must be 'direct-slots', 'component-banks', or 'metal-mmap'"
            )
        for name in (
            "prefer_sidecar",
            "verify_record_hashes",
            "verify_artifact_headers",
            "verify_sidecar_hash_at_open",
            "prefill_admission",
            "trace_routes",
            "bypass_page_cache",
            "resource_telemetry",
            "route_census",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.verify_sidecar_hash_at_open and not self.prefer_sidecar:
            raise ValueError(
                "verify_sidecar_hash_at_open requires prefer_sidecar: source-shard "
                "reads are not covered by the sidecar digest"
            )
        if self.slot_layout == "metal-mmap" and not self.verify_sidecar_hash_at_open:
            raise ValueError(
                "metal-mmap executes mapped weights without per-record hashing; "
                "it requires verify_sidecar_hash_at_open"
            )
        if self.cache_scope == "global" and self.slot_layout == "metal-mmap":
            raise ValueError(
                "global expert caching requires direct-slots or component-banks"
            )
        if self.prefill_admission:
            raise ValueError(
                "prefill admission is not implemented; prefill must use transient slots"
            )

    @property
    def derived_expert_cache_policy(self) -> bool:
        """Current layouts require a static maximum-context reservation.

        Direct slots and component banks allocate their full backing storage
        at construction. Evicting an expert only changes its logical mapping;
        it cannot fund growing KV storage. Reserve ``max_live_kv_tokens`` before
        sizing either pool, including when no expert-cache cap is supplied.
        The mapped layout also retains its existing static reservation.
        """

        return False

    def memory_plan(
        self,
        spec: ExpertStreamingModelSpec,
        *,
        additional_resident_bytes: int = 0,
        live_kv_tokens: int | None = None,
        resident_discount_bytes: int = 0,
        layer_record_bytes: "Mapping[int, int] | None" = None,
    ) -> ExpertMemoryPlan:
        if self.island_layer_count is not None and not self.island_layers:
            raise ExpertStreamingConfigurationError(
                f"island_layer_count {self.island_layer_count} is unresolved "
                f"for {self.model_key}; resolve_island_placement() (or "
                "ExpertStreamingRuntime.open) maps the count to layers. "
                "Selection precedence: explicit island_layers > "
                "spec.island_pin_order > island-placement.json > error"
            )
        # File-backed Metal records use the OS page cache as their physical
        # tier and never consume fixed MLX expert slots. Retain only a tiny
        # unreachable transient pool so the generic runtime invariants and
        # diagnostics remain valid while the mapped switch owns execution.
        expert_cache_limit_bytes = self.expert_cache_limit_bytes
        transient_slots = self.transient_slots
        if self.slot_layout == "metal-mmap":
            expert_cache_limit_bytes = 0
            transient_slots = spec.top_k
        if live_kv_tokens is not None and not self.derived_expert_cache_policy:
            raise ExpertStreamingConfigurationError(
                "live_kv_tokens applies only to the derived expert-cache "
                "policy; static plans always reserve max_live_kv_tokens"
            )
        if self.derived_expert_cache_policy:
            # Derived single-limit policy: the plan carries one KV boundary,
            # not a whole-context reservation. The post-load boundary is zero
            # live tokens; admit_kv_tokens re-plans before every growth.
            context_tokens = (
                0
                if live_kv_tokens is None
                else _integer("live_kv_tokens", live_kv_tokens, minimum=0)
            )
        else:
            context_tokens = self.max_live_kv_tokens
        return plan_expert_memory(
            spec,
            total_limit_bytes=self.memory_limit_bytes,
            context_tokens=context_tokens,
            runtime_reserve_bytes=self.runtime_reserve_bytes,
            expert_cache_limit_bytes=expert_cache_limit_bytes,
            transient_slots=transient_slots,
            io_staging_bytes=self.io_staging_bytes,
            execution_workspace_bytes=self.execution_workspace_bytes,
            additional_resident_bytes=additional_resident_bytes,
            resident_discount_bytes=resident_discount_bytes,
            kv_quant=self.kv_quant,
            cache_scope=self.cache_scope,
            island_layer_count=len(self.island_layers),
            mmap_island_layer_count=len(self.mmap_island_layers),
            mmap_islands_wired=self.mmap_island_wired,
            prefetch_ring_slots=self.prefetch_slots,
            miss_shadow=self.miss_shadow,
            miss_shadow_layers=self.miss_shadow_layers,
            layer_record_bytes=layer_record_bytes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _pipeline_ledger_for_config(
    config: ExpertStreamingConfig,
) -> ExpertPipelineLedger | None:
    """Enable pipeline attribution only on the instrumented slot-backed path."""

    if not config.resource_telemetry or config.slot_layout == "metal-mmap":
        return None
    return ExpertPipelineLedger(strict=False)


@dataclass(frozen=True)
class RouteWave:
    positions: tuple[int, ...]
    experts: tuple[int, ...]


def partition_route_waves(
    expert_ids: Iterable[int],
    *,
    max_unique_experts: int,
    sort_unique: bool = False,
) -> tuple[RouteWave, ...]:
    """Greedily partition flattened assignments into bounded expert unions."""

    capacity = _integer("max_unique_experts", max_unique_experts, minimum=1)
    experts = tuple(expert_ids)
    ordered_unique: list[int] = []
    seen: set[int] = set()
    for expert in experts:
        if isinstance(expert, bool) or not isinstance(expert, int):
            raise TypeError("expert ids must be exact integers")
        if expert not in seen:
            seen.add(expert)
            ordered_unique.append(expert)
    if sort_unique:
        ordered_unique.sort()
    waves: list[RouteWave] = []
    for start in range(0, len(ordered_unique), capacity):
        selected = set(ordered_unique[start : start + capacity])
        positions = tuple(
            position for position, expert in enumerate(experts) if expert in selected
        )
        waves.append(
            RouteWave(
                positions=positions,
                experts=tuple(experts[position] for position in positions),
            )
        )
    return tuple(waves)


@dataclass
class KVAdmission:
    runtime: ExpertStreamingRuntime
    tokens: int
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.runtime.release_kv_tokens(self.tokens)
        self.released = True

    def __enter__(self) -> KVAdmission:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class _ReadyRouteGroup:
    """Several independently completed route parts in original route order."""

    def __init__(self, plan: RoutePlan, parts: tuple[ReadyRoute, ...]) -> None:
        if not parts:
            raise ExpertSlotError("incremental miss group must contain a route part")
        pool = parts[0].pool
        if any(part.pool is not pool for part in parts[1:]):
            raise ExpertSlotError("incremental miss parts belong to different pools")
        self.plan = plan
        self.parts = parts
        self.pool = pool
        bindings_by_slot: dict[tuple[int, int], list[Any]] = {}
        for part in parts:
            for binding in part.bindings:
                bindings_by_slot.setdefault(
                    (binding.expert, binding.logical_slot),
                    [],
                ).append(binding)
        try:
            self.bindings = tuple(
                bindings_by_slot[(expert, slot)].pop(0)
                for expert, slot in zip(plan.experts, plan.slots, strict=True)
            )
        except (KeyError, IndexError) as exc:
            raise ExpertSlotError(
                "incremental miss parts do not cover the original route"
            ) from exc
        if any(bindings_by_slot.values()):
            raise ExpertSlotError(
                "incremental miss parts exceed the original route coverage"
            )

    @property
    def slots(self) -> tuple[int, ...]:
        return self.plan.slots

    @property
    def generations(self) -> tuple[int, ...]:
        return tuple(binding.generation for binding in self.bindings)

    def validate(self) -> None:
        for part in self.parts:
            part.validate()


class _RouteCancel:
    """One route-local cancellation source composed with its caller."""

    def __init__(
        self,
        caller: threading.Event | None,
        internal: threading.Event,
    ) -> None:
        self.caller = caller
        self.internal = internal

    def is_set(self) -> bool:
        return self.internal.is_set() or (
            self.caller is not None and self.caller.is_set()
        )


class PendingSplitRoute:
    """One layer transaction with pinned hits and asynchronously loading misses."""

    def __init__(
        self,
        runtime: "ExpertStreamingRuntime",
        layer: int,
        plan: RoutePlan,
        layer_lock: threading.Lock,
        hit_ready: ReadyRoute | None,
        miss_futures: dict[Future[ReadyRoute], RoutePlan],
        policy_txn: RoutePolicyTxn | None = None,
        io_admission: RouteIOAdmission | None = None,
        miss_cancel_event: threading.Event | None = None,
        lifecycle_release: Callable[[], None] | None = None,
        miss_parts: tuple[RoutePlan, ...] | None = None,
        pipeline_route: ExpertPipelineRoute | None = None,
    ) -> None:
        self.runtime = runtime
        self.layer = layer
        self.plan = plan
        self._policy_txn = policy_txn or RoutePolicyTxn(rollback=lambda: None)
        self._io_admission = io_admission
        self._policy_observed = False
        self.hit_ready = hit_ready
        self._miss_futures = dict(miss_futures)
        self._miss_ordinals = {
            future: ordinal for ordinal, future in enumerate(self._miss_futures)
        }
        self._all_miss_ordinals = set(self._miss_ordinals.values())
        initial_parts = tuple(self._miss_futures.values())
        self._all_miss_parts = {
            ordinal: part
            for ordinal, part in enumerate(
                initial_parts if miss_parts is None else miss_parts
            )
        }
        self._submitted_miss_ordinals = set(self._miss_ordinals.values())
        self._miss_admissions: dict[int, RouteIOAdmission] = {}
        self._completed_miss_ordinals: set[int] = set()
        self._miss_ready_parts: dict[int, ReadyRoute] = {}
        self._claimed_miss_futures: dict[Future[ReadyRoute], int] = {}
        self._consumer_leases: set[int] = set()
        self._releasing_consumer_leases: set[int] = set()
        self._miss_ready: _ReadyRouteGroup | None = None
        self._aggregate_lease: _ReadyRouteGroup | None = None
        self._aggregate_lease_ordinals: set[int] = set()
        self._miss_cancel_event = miss_cancel_event or threading.Event()
        self._lifecycle_release = lifecycle_release
        self._pipeline_route = pipeline_route
        self._layer_lock = layer_lock
        self._state_lock = threading.Lock()
        self._failure: BaseException | None = None
        self._failure_callbacks = 0
        self._failure_finalizing = False
        self._failure_finalized = False
        self._cleanup_error: BaseException | None = None
        self._ready_cleanup_complete = False
        self._ready_cleanup_finalizing = False
        self._hits_released = hit_ready is None
        self._close_requested = False
        self._finalized = False
        self._closed = False

    def release_hits(self) -> None:
        ready = self.hit_ready
        if ready is None:
            return
        self.hit_ready = None
        self._hits_released = True
        try:
            ready.release(synchronize=False)
        except BaseException as exc:
            self._record_cleanup_error(exc)
            raise

    @property
    def misses_pending(self) -> bool:
        """Whether miss I/O still offers useful work-overlap headroom."""

        with self._state_lock:
            return any(not future.done() for future in self._miss_futures)

    def claim_misses(self, ready: ReadyRoute | _ReadyRouteGroup) -> None:
        """Record the point at which runnable miss bindings are consumed."""

        route = self._pipeline_route
        if route is None:
            return
        experts = tuple(dict.fromkeys(load.expert for load in ready.plan.loads))
        if experts:
            _pipeline_call(
                self.runtime._pipeline_ledger,
                route,
                "claim_misses",
                experts,
            )

    def _attach_miss_future(
        self,
        future: Future[ReadyRoute],
        plan: RoutePlan,
        *,
        ordinal: int,
        io_admission: RouteIOAdmission | None = None,
    ) -> None:
        with self._state_lock:
            self._miss_futures[future] = plan
            self._miss_ordinals[future] = ordinal
            self._all_miss_ordinals.add(ordinal)
            self._all_miss_parts[ordinal] = plan
            self._submitted_miss_ordinals.add(ordinal)
            if io_admission is not None:
                self._miss_admissions[ordinal] = io_admission

    def _retain_lifecycle_after_admission(self) -> None:
        with self._state_lock:
            if (
                self._lifecycle_release is not None
                or self._failure_finalized
                or self._policy_observed
            ):
                return
            lifecycle = self.runtime.slots.retain_admitted_split_lifecycle()
            self._lifecycle_release = lifecycle.release

    def _record_cleanup_error(self, error: BaseException) -> None:
        promote = False
        with self._state_lock:
            if self._cleanup_error is None:
                self._cleanup_error = error
                promote = True
        if promote:
            # Never nest the runtime health lock under Pending state.
            recorder = getattr(self.runtime, "_record_cleanup_error", None)
            if callable(recorder):
                recorder(error)

    def _release_lifecycle(self) -> None:
        with self._state_lock:
            release = self._lifecycle_release
            self._lifecycle_release = None
        if release is None:
            return
        try:
            release()
        except BaseException as exc:
            with self._state_lock:
                if self._lifecycle_release is None:
                    self._lifecycle_release = release
            self._record_cleanup_error(exc)

    @staticmethod
    def _release_routes(routes: Iterable[ReadyRoute]) -> BaseException | None:
        first_error: BaseException | None = None
        for ready in routes:
            try:
                ready.release(synchronize=False)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        return first_error

    def _store_completed_future(
        self,
        future: Future[ReadyRoute],
        ordinal: int,
    ) -> None:
        try:
            ready = future.result()
        except BaseException:
            return
        with self._state_lock:
            self._miss_ready_parts[ordinal] = ready

    def _failure_callback(
        self,
        future: Future[ReadyRoute],
        ordinal: int,
    ) -> None:
        self._store_completed_future(future, ordinal)
        with self._state_lock:
            self._failure_callbacks -= 1
        self._finish_failure_if_ready()

    def abort(self, error: BaseException) -> None:
        """Cancel this transaction without waiting for running miss workers."""

        # Cancellation is the first observable failure action. Every miss part
        # sees this same route-local event, composed with the caller event.
        self._miss_cancel_event.set()
        with self._state_lock:
            if self._failure is not None:
                return
            self._failure = error
            pending = tuple(
                sorted(
                    self._miss_futures,
                    key=lambda future: self._miss_ordinals[future],
                )
            )
            ordinals = {future: self._miss_ordinals[future] for future in pending}
            self._miss_futures.clear()
            self._miss_ordinals.clear()
            self._miss_ready = None
            # Claim every future before failure becomes observable without the
            # state lock. Completed paths and callbacks each consume one claim.
            self._failure_callbacks = len(pending)

        for future in pending:
            future.cancel()
        completed: list[Future[ReadyRoute]] = []
        running: list[Future[ReadyRoute]] = []
        for future in pending:
            (completed if future.done() else running).append(future)
        for future in completed:
            self._failure_callback(future, ordinals[future])
        for future in running:
            future.add_done_callback(
                lambda done, ordinal=ordinals[future]: self._failure_callback(
                    done,
                    ordinal,
                )
            )
        self._finish_failure_if_ready()

    def _finish_failure_if_ready(self) -> None:
        with self._state_lock:
            if (
                self._failure is None
                or self._failure_callbacks
                or self._claimed_miss_futures
                or self._consumer_leases
                or self._releasing_consumer_leases
                or self._aggregate_lease is not None
                or self._ready_cleanup_finalizing
                or self._failure_finalizing
                or self._failure_finalized
            ):
                return
            self._failure_finalizing = True
            failure = self._failure
            policy_observed = self._policy_observed

        try:
            with self._state_lock:
                routes = tuple(
                    self._miss_ready_parts[ordinal]
                    for ordinal in sorted(self._miss_ready_parts)
                )
                self._miss_ready_parts.clear()
                self._miss_ready = None
            release_error = self._release_routes(routes)
            if release_error is not None:
                self._record_cleanup_error(release_error)
            if not policy_observed:
                try:
                    accepted_ordinals = {
                        ordinal
                        for ordinal, admission in self._miss_admissions.items()
                        if admission.any_accepted
                    }
                    if not self._miss_admissions and (
                        self._io_admission is not None
                        and self._io_admission.any_accepted
                    ):
                        accepted_ordinals = set(self._submitted_miss_ordinals)
                    accepted_parts = tuple(
                        self._all_miss_parts[ordinal]
                        for ordinal in sorted(accepted_ordinals)
                    )
                    self.runtime._handle_split_route_failure(
                        self.layer,
                        self.plan,
                        self._policy_txn,
                        failure,
                        accepted_parts=accepted_parts,
                        io_admission=self._io_admission,
                    )
                except BaseException as exc:
                    self._record_cleanup_error(exc)
        finally:
            self._release_lifecycle()
            with self._state_lock:
                self._ready_cleanup_complete = True
                self._failure_finalized = True
                self._failure_finalizing = False
            self._finalize_if_ready()

    def _iter_pipeline_miss_completions(
        self,
        snapshot: tuple[Future[ReadyRoute], ...],
    ) -> Iterable[Future[ReadyRoute]]:
        """Bound a completion step that may block on the existing iterator.

        The readiness scan and ``next(as_completed(...))`` cannot be atomic through
        the public Future API. The measured span is therefore an upper bound that
        can include completion races, iterator work, and telemetry-lock delay.
        """

        route = self._pipeline_route
        assert route is not None
        completions = iter(as_completed(snapshot))
        remaining = set(snapshot)
        while remaining:
            try:
                may_block_for_next = not any(future.done() for future in remaining)
            except Exception:
                may_block_for_next = False
                ledger = self.runtime._pipeline_ledger
                if ledger is not None:
                    try:
                        ledger.mark_incomplete(phase=route.phase)
                    except Exception:
                        pass
            if may_block_for_next:
                _pipeline_call(
                    self.runtime._pipeline_ledger,
                    route,
                    "begin_potentially_blocking_next_miss_step",
                )
            try:
                future = next(completions)
            finally:
                if may_block_for_next:
                    _pipeline_call(
                        self.runtime._pipeline_ledger,
                        route,
                        "end_potentially_blocking_next_miss_step",
                    )
            remaining.discard(future)
            yield future

    def iter_ready_misses(self) -> Iterable[ReadyRoute]:
        """Yield authoritative miss bindings in physical completion order."""

        with self._state_lock:
            snapshot = tuple(self._miss_futures)
        completion_order = (
            as_completed(snapshot)
            if self._pipeline_route is None
            else self._iter_pipeline_miss_completions(snapshot)
        )
        # W125: from here to each yield the generation thread is blocked on the miss
        # futures (as_completed / future.result) -- the EXPOSED SSD read wait the
        # overlap could not hide, and the biggest suspected host gap. Reset after
        # every yield so a multi-part route accumulates only its own blocks.
        _tl_w = _tl.now()
        for future in completion_order:
            with self._state_lock:
                if future not in self._miss_futures:
                    continue
                self._miss_futures.pop(future)
                ordinal = self._miss_ordinals.pop(future)
                self._claimed_miss_futures[future] = ordinal
            try:
                ready = future.result()
            except BaseException as exc:
                with self._state_lock:
                    self._claimed_miss_futures.pop(future, None)
                    failure = self._failure
                if failure is not None:
                    self._finish_failure_if_ready()
                    raise failure
                self.abort(exc)
                raise
            with self._state_lock:
                self._claimed_miss_futures.pop(future, None)
                self._miss_ready_parts[ordinal] = ready
                self._completed_miss_ordinals.add(ordinal)
                failure = self._failure
            if failure is not None:
                self._finish_failure_if_ready()
                raise failure
            try:
                self.runtime.slots.raise_if_unhealthy()
            except BaseException as exc:
                self.abort(exc)
                raise
            with self._state_lock:
                is_final_part = (
                    not self._miss_futures and not self._claimed_miss_futures
                )
            if is_final_part:
                try:
                    self.runtime.slots.commit_if_healthy(
                        lambda ordinal=ordinal: self._validate_and_commit_policy(
                            lease_ordinal=ordinal
                        )
                    )
                except BaseException as exc:
                    self.abort(exc)
                    raise
            else:
                with self._state_lock:
                    if self._failure is None and not self._close_requested:
                        self._consumer_leases.add(ordinal)
            with self._state_lock:
                leased = ordinal in self._consumer_leases
                failure = self._failure
                if failure is not None and leased:
                    self._consumer_leases.remove(ordinal)
                    leased = False
            if failure is not None:
                self._finish_failure_if_ready()
                raise failure
            if not leased:
                failure = ExpertSlotError("split route closed before miss yield")
                self.abort(failure)
                self._finish_failure_if_ready()
                raise failure
            _tl.add_miss_wait(self.layer, _tl_w)  # W125: exposed miss wait for this part
            _tl.miss_ready(self.layer)  # W125: a miss part's slots are ready
            yield ready
            _tl_w = _tl.now()  # W125: start timing the next part's wait
        if not self._policy_observed:
            try:
                self.runtime.slots.raise_if_unhealthy()
                self.runtime.slots.commit_if_healthy(self._validate_and_commit_policy)
            except BaseException as exc:
                self.abort(exc)
                raise

    def _validate_completed_misses(self) -> None:
        with self._state_lock:
            if self._completed_miss_ordinals != self._all_miss_ordinals:
                raise ExpertSlotError(
                    "incremental miss completion does not cover every route part"
                )

    def _validate_and_commit_policy(self, *, lease_ordinal: int | None = None) -> None:
        """Validate and publish only host policy state and counters."""

        self._validate_completed_misses()
        self._commit_policy(lease_ordinal=lease_ordinal)

    def _prepare_ready_group(self) -> _ReadyRouteGroup | None:
        miss_plan = self.runtime._subset_route_plan(self.plan, hits=False)
        if miss_plan is None:
            return None
        with self._state_lock:
            if self._miss_ready is not None:
                return self._miss_ready
            if self._failure is not None:
                raise self._failure
            if self._close_requested:
                raise ExpertSlotError("split route closed before miss aggregation")
            if self._completed_miss_ordinals != self._all_miss_ordinals:
                raise ExpertSlotError(
                    "incremental miss completion does not cover every route part"
                )
            expected_ordinals = set(self._all_miss_ordinals)
            if set(self._miss_ready_parts) != expected_ordinals:
                raise ExpertSlotError(
                    "incremental miss parts do not cover the original route"
                )
            if self._consumer_leases != expected_ordinals:
                raise ExpertSlotError(
                    "incremental miss aggregation does not own every route part"
                )
            parts = tuple(
                self._miss_ready_parts[ordinal] for ordinal in sorted(expected_ordinals)
            )
            group = _ReadyRouteGroup(miss_plan, parts)
            self._consumer_leases.clear()
            self._aggregate_lease = group
            self._aggregate_lease_ordinals = expected_ordinals
            self._miss_ready = group
            return group

    def finish_misses(self) -> _ReadyRouteGroup | None:
        with self._state_lock:
            if self._miss_ready is not None:
                return self._miss_ready
            empty = not self._miss_futures and not self._miss_ready_parts
        if empty:
            return None
        try:
            for _ready in self.iter_ready_misses():
                pass
            return self._prepare_ready_group()
        except BaseException as exc:
            self.abort(exc)
            with self._state_lock:
                leased_readies = tuple(
                    self._miss_ready_parts[ordinal]
                    for ordinal in sorted(self._consumer_leases)
                )
            for ready in leased_readies:
                try:
                    self.release_miss(ready)
                except BaseException:
                    pass
            self._finish_failure_if_ready()
            raise

    def _commit_policy(self, *, lease_ordinal: int | None = None) -> None:
        committed = False
        with self._state_lock:
            if self._failure is not None:
                return
            if not self._policy_observed:
                incremental_parts = len(self._completed_miss_ordinals)
                self.runtime._publish_route_transaction(
                    self.layer,
                    self.plan,
                    self._policy_txn,
                    incremental_parts=incremental_parts,
                )
                self._policy_observed = True
                committed = True
            if lease_ordinal is not None:
                self._consumer_leases.add(lease_ordinal)
        if committed:
            self._release_lifecycle()

    def release_miss(self, ready: ReadyRoute) -> None:
        """Release one streamed part while retaining sole route ownership."""

        with self._state_lock:
            matches = tuple(
                ordinal
                for ordinal, candidate in self._miss_ready_parts.items()
                if candidate is ready
            )
            if len(matches) != 1:
                raise ExpertSlotError("miss part is not owned by this split route")
            ordinal = matches[0]
            if ordinal not in self._consumer_leases:
                raise ExpertSlotError("miss part has no active consumer lease")
            self._consumer_leases.remove(ordinal)
            self._releasing_consumer_leases.add(ordinal)
            self._miss_ready_parts.pop(ordinal)
            self._miss_ready = None
        release_error: BaseException | None = None
        try:
            ready.release(synchronize=False)
        except BaseException as exc:
            release_error = exc
            self._record_cleanup_error(exc)
        with self._state_lock:
            self._releasing_consumer_leases.remove(ordinal)
        success_error = self._finish_success_close_if_ready()
        self._finish_failure_if_ready()
        self._finalize_if_ready()
        if release_error is not None:
            raise release_error
        if success_error is not None:
            raise success_error

    def release_misses(self, ready: _ReadyRouteGroup) -> None:
        """Return one aggregate consumer lease to Pending ownership."""

        with self._state_lock:
            if self._aggregate_lease is not ready:
                raise ExpertSlotError("miss aggregate is not owned by this split route")
            ordinals = tuple(sorted(self._aggregate_lease_ordinals))
            routes = tuple(self._miss_ready_parts[ordinal] for ordinal in ordinals)
            self._aggregate_lease = None
            self._aggregate_lease_ordinals.clear()
            self._miss_ready = None
            self._releasing_consumer_leases.update(ordinals)
            for ordinal in ordinals:
                self._miss_ready_parts.pop(ordinal)
        release_error = self._release_routes(routes)
        if release_error is not None:
            self._record_cleanup_error(release_error)
        with self._state_lock:
            self._releasing_consumer_leases.difference_update(ordinals)
        success_error = self._finish_success_close_if_ready()
        self._finish_failure_if_ready()
        self._finalize_if_ready()
        if release_error is not None:
            raise release_error
        if success_error is not None:
            raise success_error

    def _finish_success_close_if_ready(self) -> BaseException | None:
        with self._state_lock:
            if (
                self._failure is not None
                or not self._close_requested
                or self._claimed_miss_futures
                or self._consumer_leases
                or self._releasing_consumer_leases
                or self._aggregate_lease is not None
                or self._ready_cleanup_finalizing
                or self._ready_cleanup_complete
            ):
                return None
            self._ready_cleanup_finalizing = True
            routes = tuple(
                self._miss_ready_parts[ordinal]
                for ordinal in sorted(self._miss_ready_parts)
            )
            self._miss_ready_parts.clear()
            self._miss_ready = None
        release_error = self._release_routes(routes)
        if release_error is not None:
            self._record_cleanup_error(release_error)
        with self._state_lock:
            self._ready_cleanup_complete = True
            self._ready_cleanup_finalizing = False
        self._finalize_if_ready()
        return release_error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._state_lock:
            needs_abort = self._failure is None and not self._policy_observed
        if needs_abort:
            self.abort(ExpertSlotError("split route closed before commit"))
        first_error: BaseException | None = None
        try:
            self.release_hits()
        except BaseException as exc:
            first_error = exc
        with self._state_lock:
            failure = self._failure
            self._close_requested = True
        release_error = self._finish_success_close_if_ready()
        if first_error is None:
            first_error = release_error
        self._finish_failure_if_ready()
        self._finalize_if_ready()
        if failure is None and first_error is not None:
            raise first_error

    def _finalize_if_ready(self) -> None:
        with self._state_lock:
            if (
                self._finalized
                or not self._close_requested
                or not self._hits_released
                or not self._ready_cleanup_complete
            ):
                return
            self._finalized = True
            pipeline_route = self._pipeline_route
            self._pipeline_route = None
        try:
            if pipeline_route is not None:
                _pipeline_call(
                    self.runtime._pipeline_ledger,
                    pipeline_route,
                    "close",
                )
        finally:
            self._layer_lock.release()

    def __enter__(self) -> "PendingSplitRoute":
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: object,
    ) -> None:
        if exc is not None:
            self.abort(exc)
        self.close()


def derived_expert_cache_allowance_bytes(plan: ExpertMemoryPlan) -> int:
    """Byte allowance for the streamed slot cache under one boundary plan.

    A pure function of the plan: ``plan.fixed_bytes`` is authoritative for
    everything permanently resident (model weights, live KV, transient miss
    service, workspace, staging, allocator headroom, and any dense-island or
    wired mmap-band layers a plan may carry), so the derivation stays correct
    as fixed-side categories evolve.  A paged (unwired) mmap band lives in
    the page cache outside MLX and additionally shrinks the MLX cap by its
    own bytes (see :func:`reconcile_mlx_memory_cap`); the slot cache must fit
    under that reduced cap too, or the pager evicts band pages on every
    decode wave.

    Returns a negative value when even an empty cache oversubscribes the
    limit: the caller must fail the admission rather than clamp.
    """

    paged_band_bytes = (
        0
        if getattr(plan, "mmap_islands_wired", True)
        else getattr(plan, "mmap_island_bytes", 0)
    )
    available = plan.total_limit_bytes - plan.fixed_bytes - paged_band_bytes
    if available < 0:
        return available
    return min(plan.persistent_budget_bytes, available)
def resolve_island_placement(
    config: ExpertStreamingConfig,
    root: Path | str,
    *,
    spec: ExpertStreamingModelSpec | None = None,
) -> ExpertStreamingConfig:
    """Resolve a pending island_layer_count into explicit island_layers.

    Selection precedence (2026-07-21, census-first): explicit
    ``island_layers`` > ``<root>/island-placement.json`` (the auto-census
    artifact, issue #98 — the bank's OWN measured ranking) when sound >
    ``spec.island_pin_order`` (version-controlled bootstrap default) >
    ADVISORY census (noise-flagged; used only when no spec order exists,
    with a loud warning) > error. Rationale: a census is measured on the
    concrete bank at this root; a spec order is a static snapshot and, for
    derived banks, often borrowed from a sibling. Idempotent: a config
    whose count already resolved (or that requests no count) returns
    unchanged.
    """

    count = config.island_layer_count
    if count is None or config.island_layers:
        return config
    model_spec = get_model_spec(config.model_key) if spec is None else spec
    if model_spec.key != config.model_key:
        raise ExpertStreamingConfigurationError(
            "config and spec model keys differ"
        )
    placement_path = Path(root) / ISLAND_PLACEMENT_FILENAME
    placement = None
    if placement_path.is_file():
        try:
            placement = load_placement(placement_path)
        except RouteCensusError as exc:
            raise ExpertStreamingConfigurationError(
                f"island_layer_count {count} cannot be resolved: {exc}"
            ) from exc
        if placement.model_key != model_spec.key:
            raise ExpertStreamingConfigurationError(
                f"{placement_path} was derived for "
                f"{placement.model_key!r}, not {model_spec.key!r}"
            )
    if placement is not None and not placement.advisory:
        order = placement.layer_pin_order
        source = str(placement_path)
        _LOGGER.info(
            "island placement resolved from census %s (%d routed "
            "assignments)",
            placement_path,
            placement.census_total_routed_assignments,
        )
    elif model_spec.island_pin_order:
        order = model_spec.island_pin_order
        source = f"{model_spec.key} spec island_pin_order"
        if placement is not None:
            _LOGGER.warning(
                "ignoring ADVISORY census %s (only %d routed assignments; "
                "rankings may be noise-ordered) in favor of the spec pin "
                "order",
                placement_path,
                placement.census_total_routed_assignments,
            )
    elif placement is not None:
        order = placement.layer_pin_order
        source = str(placement_path)
        _LOGGER.warning(
            "island placement %s is advisory: derived from only %d routed "
            "assignments; rankings may still be noise-ordered (no spec pin "
            "order exists to prefer)",
            placement_path,
            placement.census_total_routed_assignments,
        )
    else:
        raise ExpertStreamingConfigurationError(
            f"island_layer_count {count} cannot be resolved for "
            f"{model_spec.key}: no census at {placement_path} and the spec "
            "has no measured island pin order. Selection precedence: "
            f"explicit island_layers > {ISLAND_PLACEMENT_FILENAME} "
            "(auto-census, issue #98) > spec.island_pin_order > advisory "
            "census > this error. Run the model once without islands — the "
            "route census records decode routes and writes the placement "
            "at close — or pass explicit island_layers."
        )
    if count > len(order):
        raise ExpertStreamingConfigurationError(
            f"island_layer_count {count} exceeds the {model_spec.key} pin "
            f"order from {source} ({len(order)} layers)"
        )
    try:
        return dataclass_replace(
            config,
            island_layer_count=None,
            island_layers=tuple(sorted(order[:count])),
        )
    except (TypeError, ValueError) as exc:
        raise ExpertStreamingConfigurationError(str(exc)) from exc


def proj_quant_plan_discount(manifest: Any, proj_quant: str | None) -> int:
    """Bytes the load-time resident quantization removes from the fixed side.

    Computed per manifest tensor with the same scope predicate the loader
    applies, so the plan prices the post-quantization footprint instead of
    the stored BF16 one (kept bytes round up — the discount understates).
    """

    if not proj_quant:
        return 0
    discount = 0
    for tensor in manifest.resident_tensors:
        if tensor.dtype.upper() not in {"BF16", "BFLOAT16"}:
            continue
        name = tensor.tensor
        if name.endswith(".weight"):
            name = name[: -len(".weight")]
        else:
            continue
        if proj_quant_covers(name):
            discount += tensor.length - affine_quant_kept_bytes(
                tensor.length, proj_quant
            )
    return discount


def proj_requant_plan_discount(manifest: Any, proj_requant: str | None) -> int:
    """Bytes the q8->q4 resident requant removes from the fixed side.

    Scope matches the loader (``proj_quant_covers`` over ``*_proj`` weights).
    Only the packed U32 weight shrinks (q8 packs 4 values/u32, q4 packs 8 →
    exactly half the bytes); scales/biases keep their group count at gs64 and
    are unchanged. Non-quantized (BF16) residents are ignored — the requant
    pass only converts already-quantized modules.
    """

    if proj_requant != "q4":
        return 0
    discount = 0
    for tensor in manifest.resident_tensors:
        if tensor.dtype.upper() != "U32":
            continue
        name = tensor.tensor
        if name.endswith(".weight"):
            name = name[: -len(".weight")]
        else:
            continue
        if proj_quant_covers(name):
            discount += tensor.length // 2
    return discount


# Residents a text-only autoregressive forward never wires. The DeepSeek-V4.1
# converter keeps the MTP residents (mxfp8 dense + mxfp4 experts) and the
# vision/aligner/image residents inside the streamed artifact, but phase-1 AR
# serving loads only the text backbone
# (``mtplx.models.deepseek_v41_loader.partition_text_residents`` drops exactly
# these dotted-name prefixes). Kept here so the memory-plan discount and the
# loader's text-only filter share one prefix set; a test locks the two equal.
TEXT_ONLY_SKIP_PREFIXES = ("mtp.", "vision.", "aligner.", "image_")


def text_only_resident_discount(manifest: Any, spec: Any) -> int:
    """Bytes of residents the text-only AR forward skips at load, for planner pricing.

    ``spec.resident_bytes`` counts every resident the converter emitted, but the
    serve path materializes only the text backbone; without this discount the
    planner over-reserves the fixed side. For the shipped DeepSeek-V4.1-Flash
    mxfp4 artifact the skipped MTP + vision residents are 8.31 GiB, worth +11
    resident expert slots/layer at the 82 GiB envelope (W3/W21).

    Zero for every spec whose streamed manifest carries no such residents
    (hy3/glm keep their MTP head in a separate external artifact and have no
    vision residents), so their resolved memory plans stay byte-identical. When
    ``spec.mtp_included`` is True (a future MTP serve path wires the MTP
    residents), only vision/aligner/image are discounted.
    """

    mtp_wired = bool(getattr(spec, "mtp_included", False))
    discount = 0
    for tensor in manifest.resident_tensors:
        name = tensor.tensor
        if not name.startswith(TEXT_ONLY_SKIP_PREFIXES):
            continue
        if mtp_wired and name.startswith("mtp."):
            continue
        discount += tensor.length
    return discount


def reconcile_mlx_memory_cap(
    plan: ExpertMemoryPlan,
    *,
    env: Mapping[str, str] | None = None,
) -> int:
    """Resolve the MLX-owned portion and reject a conflicting env cap."""

    # A wired band is registered in MLX's residency set and counted by MLX
    # memory accounting (it already sits on the plan's fixed side); only a
    # paged band lives in the page cache outside the cap.
    paged_band_bytes = (
        0
        if getattr(plan, "mmap_islands_wired", True)
        else getattr(plan, "mmap_island_bytes", 0)
    )
    mlx_limit = (
        plan.total_limit_bytes
        - plan.runtime_reserve_bytes
        - plan.io_staging_bytes
        - paged_band_bytes
    )
    if mlx_limit <= 0:
        raise ExpertStreamingConfigurationError(
            "memory plan leaves no MLX allocation budget"
        )
    # A paged band's pages are the pager's first eviction victims. If MLX's
    # buffer cache may balloon into them, every decode wave re-faults from
    # SSD — fail fast instead. (Measured: 10.85 -> 2.77 tok/s under exactly
    # this squeeze.)
    wired_need = (
        plan.fixed_bytes
        - plan.runtime_reserve_bytes
        - plan.io_staging_bytes
        + plan.persistent_cache_bytes
    )
    if paged_band_bytes and mlx_limit < wired_need:
        raise ExpertStreamingConfigurationError(
            "MLX budget cannot hold the wired footprint next to the paged "
            f"mmap island band: cap {mlx_limit} < wired need {wired_need}; "
            "raise memory_limit_bytes or shrink the band"
        )
    source = os.environ if env is None else env
    existing = source.get("MTPLX_MEMORY_LIMIT_BYTES")
    if existing:
        parsed = parse_memory_bytes(existing)
        if parsed != mlx_limit:
            raise ExpertStreamingConfigurationError(
                "MTPLX_MEMORY_LIMIT_BYTES conflicts with expert streaming plan: "
                f"env={parsed}, planned={mlx_limit}"
            )
    return mlx_limit


# W118 (H7): the MLX allocator-limit headroom lever.  Read at USE (never frozen at
# import), matching every other DSV4.1 lever ([[env-flags-read-at-use-not-import]]).
# Default 0 == today.  It adds N GiB to the value passed to ``set_memory_limit`` ABOVE
# the residency plan WITHOUT changing the plan (residents, expert-cache slots, prefetch
# ring) or the ``MTPLX_MEMORY_LIMIT_BYTES`` engine budget -- so bytes/routing/outputs
# stay byte-identical.  The finding (W112 receipt window-44b): every real window runs
# the model OVER its own MLX limit (plan 69.2 -> mlx_peak 74.3).
#
# Mechanism (mlx 0.32.2 allocator.cpp): set_memory_limit sets block_limit_ and
# gc_limit_ = min(limit, 0.95 x recommendedMaxWorkingSetSize).  On a cache MISS with
# active + cache + size >= gc_limit_, MetalAllocator::malloc calls
# release_cached_buffers(...); once active >= gc_limit_ the release argument exceeds the
# pool, so the ENTIRE buffer cache is cleared, and every later allocation is a fresh
# newBuffer (page zero-fill) + residency-set insert.  There is NO scheduler wait and NO
# error -- it is cache-clear-on-miss thrash, the (f) allocator-pressure regime (5.2x
# in-model attention).  Raising ONLY the limit above the steady-state peak lifts
# gc_limit_ so the miss path stops clearing the cache, without touching what is
# resident.  (Proven for prefill, where mlx_peak > limit; whether decode is also over
# the limit is what window 46 measures -- see W118_MLX_LIMIT_HEADROOM.md.)
MLX_LIMIT_HEADROOM_ENV = "MTPLX_DSV41_MLX_LIMIT_HEADROOM_GIB"
_MLX_LIMIT_HEADROOM_GIB = 1024**3


def resolve_mlx_limit_headroom_bytes(env: Mapping[str, str] | None = None) -> int:
    """Extra bytes added to the value passed to ``mx.set_memory_limit`` ABOVE the
    residency plan (W118 / H7).  Reads ``MTPLX_DSV41_MLX_LIMIT_HEADROOM_GIB`` at USE;
    default 0 (unset / empty == today).  Does NOT change the plan or
    ``MTPLX_MEMORY_LIMIT_BYTES``.  Rejects a non-numeric or negative value."""

    source = os.environ if env is None else env
    raw = source.get(MLX_LIMIT_HEADROOM_ENV)
    if raw is None or str(raw).strip() == "":
        return 0
    stripped = str(raw).strip()
    # W118 review MEDIUM-3: reject underscore-obfuscated values ("1_0" is 10.0 under
    # PEP 515, which silently misreads an operator typo) BEFORE float().
    if "_" in stripped:
        raise ExpertStreamingConfigurationError(
            f"{MLX_LIMIT_HEADROOM_ENV} must be a plain number of GiB, got {raw!r} "
            "(underscores are not allowed)"
        )
    try:
        value = float(stripped)
    except (TypeError, ValueError) as exc:
        raise ExpertStreamingConfigurationError(
            f"{MLX_LIMIT_HEADROOM_ENV} must be a number of GiB, got {raw!r}"
        ) from exc
    # W118 review MEDIUM-3: nan/inf must be rejected -- ``float('nan') < 0`` is False,
    # so a non-finite value would otherwise pass and crash in-window.
    if not math.isfinite(value):
        raise ExpertStreamingConfigurationError(
            f"{MLX_LIMIT_HEADROOM_ENV} must be finite, got {raw!r}"
        )
    if value < 0:
        raise ExpertStreamingConfigurationError(
            f"{MLX_LIMIT_HEADROOM_ENV} must be non-negative, got {value}"
        )
    return int(round(value * _MLX_LIMIT_HEADROOM_GIB))


# W121: target-based MLX allocator limit.  David's directive after window 47 proved
# the mechanism: with set_memory_limit ABOVE the steady active peak the freed-buffer
# LRU HOLDS across misses (decode 2.99 -> 5.49; cache_at_decode_end 0 -> 5.0 GiB),
# whereas at-the-plan it clears on every miss.  Instead of a hand-picked headroom, the
# allocator limit is derived from David's TOTAL box target so box_used stays <= target
# BY CONSTRUCTION: the box holds macOS+agent (the baseline, measured once with the
# resident agent booted out) plus this process, so the process may use
# Allocation target and baseline are decimal GB; all cache/reserve bands are GiB.
# Serving and benchmark constructors derive one shared target plan before loading.
# An unset target in the low-level runtime preserves the legacy explicit-plan path.
BOX_TARGET_ENV = "MTPLX_DSV41_BOX_TARGET_GB"
BOX_BASELINE_ENV = "MTPLX_DSV41_BOX_BASELINE_GB"
DEFAULT_BOX_TARGET_GB = 110.0  # total decimal GB, including host allocations and baseline
_DECIMAL_GB = 1_000_000_000
_GIB = 1024**3

# The allocator policy accounts for active allocations plus retained cache jointly.
# MLX get_peak_memory reports ACTIVE allocations only; it does not measure this sum.
# Reserve Python arenas/indices outside the allocator. Inside it, reserve the larger
# of observed prefill transients and decode active overshoot + retained allocator
# cache. These are planning bands from prior receipts, not a hard process peak proof.
# The OS guard independently measures physical system usage, including file cache.
BOX_ALLOC_CACHE_ENV = "MTPLX_DSV41_MLX_CACHE_LIMIT_GIB"
BOX_TRANSIENT_BAND_ENV = "MTPLX_DSV41_TRANSIENT_BAND_GIB"
BOX_HOST_OVERHEAD_ENV = "MTPLX_DSV41_HOST_OVERHEAD_GIB"
BOX_ACTIVE_OVERSHOOT_ENV = "MTPLX_DSV41_ACTIVE_OVERSHOOT_GIB"
BOX_SESSION_BANK_ENV = "MTPLX_DSV41_SESSION_BANK_GIB"
# Uncached 16K receipts show ~6.53 GiB active overhang above allocated plan
# storage. The prefill band includes the full 2 GiB freed cache plus headroom.
DEFAULT_ALLOC_CACHE_GIB = 2.0
DEFAULT_TRANSIENT_BAND_GIB = 10.0
DEFAULT_ACTIVE_OVERSHOOT_GIB = 1.45  # active_at_decode_start - PLAN (W47 receipt 1.4416; round up to cover it)
DEFAULT_HOST_OVERHEAD_GIB = 2.0  # full default Engram arenas, indices, and other Python


def _physical_ram_bytes() -> int | None:
    import subprocess
    import sys
    if sys.platform != "darwin":
        return None
    result = subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"],
                            capture_output=True, text=True, timeout=5, check=True)
    return int(result.stdout.strip())


def prepare_deepseek_v41_memory_config(config, *, env=None, preserve_memory_limit=False,
                                     args=None):
    """Resolve the serving allocation once, before any model or cache allocation."""
    if not config.model_key.startswith("deepseek-v41"):
        return config, None
    from .deepseek_v41_memory_profile import DEFAULT_ENGRAM_CACHE_BYTES, box_memory_snapshot
    target_env = os.environ if env is None else env
    if args is not None:
        if getattr(args, "memory_budget", None):
            raise ExpertStreamingConfigurationError(
                "--memory-budget conflicts with the DeepSeek box target; configure "
                "MTPLX_DSV41_BOX_TARGET_GB in decimal GB instead")
        for key in ("MTPLX_MEMORY_LIMIT_BYTES", "MTPLX_WIRED_LIMIT_BYTES", "MTPLX_MEMORY_BUDGET"):
            if target_env.get(key):
                raise ExpertStreamingConfigurationError(
                    f"{key} conflicts with the DeepSeek box target; configure "
                    "MTPLX_DSV41_BOX_TARGET_GB or --expert-memory-limit instead")
        # The SSD writer admits oversized entries even above its backlog limit.
        # Until encode/restore capacity is bounded, it cannot share this budget.
        explicit_ssd = ("ssd-session-cache" in getattr(args, "_cli_flags", set()) or
                        bool(target_env.get("MTPLX_SSD_SESSION_CACHE")))
        if explicit_ssd and getattr(args, "ssd_session_cache", "off") != "off":
            raise ExpertStreamingConfigurationError(
                "SSD session cache has no bounded host allocation under the DeepSeek "
                "box target; use --ssd-session-cache off")
        args.ssd_session_cache = "off"
        from pydantic import ByteSize, TypeAdapter
        target_env.setdefault("MTPLX_SESSION_BANK_MAX_BYTES", "2GiB")
        bank_bytes = int(TypeAdapter(ByteSize).validate_python(
            target_env["MTPLX_SESSION_BANK_MAX_BYTES"]))
        if bank_bytes <= 0:
            raise ExpertStreamingConfigurationError("session bank capacity must be positive")
        # Use bytes for the downstream parser so GB/GiB aliases cannot disagree.
        target_env["MTPLX_SESSION_BANK_MAX_BYTES"] = str(bank_bytes)
        target_env[BOX_SESSION_BANK_ENV] = str(bank_bytes / _GIB)
        raw_cache = getattr(args, "mlx_cache_limit", None) or target_env.get("MTPLX_MLX_CACHE_LIMIT")
        if raw_cache:
            from pydantic import ByteSize, TypeAdapter
            cache_bytes = int(TypeAdapter(ByteSize).validate_python(raw_cache))
            target_env[BOX_ALLOC_CACHE_ENV] = str(cache_bytes / _GIB)
    target_env.setdefault(BOX_TARGET_ENV, str(int(DEFAULT_BOX_TARGET_GB)))
    target_env.setdefault("MTPLX_ENGRAM_CACHE_LIMIT", str(DEFAULT_ENGRAM_CACHE_BYTES))
    if not target_env.get(BOX_BASELINE_ENV):
        snap = box_memory_snapshot()
        baseline = snap.get("used_bytes")
        if not snap.get("ok") or baseline is None or baseline <= 0:
            raise ExpertStreamingConfigurationError("cannot measure pre-load system baseline")
        target_env[BOX_BASELINE_ENV] = f"{baseline / _DECIMAL_GB:.12g}"
    budget = resolve_box_target_mlx_limit_bytes(target_env)
    engine_bytes = budget["engine_budget_bytes"]
    if preserve_memory_limit:
        if config.memory_limit_bytes > engine_bytes:
            raise ExpertStreamingConfigurationError(
                "explicit engine budget exceeds the target allocation budget")
        engine_bytes = config.memory_limit_bytes
    budget["configured_engine_budget_bytes"] = engine_bytes
    return dataclass_replace(config, memory_limit_bytes=engine_bytes), budget


def _env_pos_float(env: Mapping[str, str], key: str) -> float | None:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "" or str(raw).strip().lower() == "default":
        return None
    stripped = str(raw).strip()
    if "_" in stripped:
        raise ExpertStreamingConfigurationError(
            f"{key} must be a plain number, got {raw!r} (underscores not allowed)"
        )
    try:
        value = float(stripped)
    except (TypeError, ValueError) as exc:
        raise ExpertStreamingConfigurationError(
            f"{key} must be a number, got {raw!r}"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ExpertStreamingConfigurationError(f"{key} must be finite and positive, got {value}")
    return value


def _env_nonneg_float_or_default(
    env: Mapping[str, str], key: str, default: float
) -> float:
    """A GiB band value from the env (>= 0, finite), or ``default`` when unset/empty."""

    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return float(default)
    stripped = str(raw).strip()
    if "_" in stripped:
        raise ExpertStreamingConfigurationError(
            f"{key} must be a plain number of GiB, got {raw!r} (underscores not allowed)"
        )
    try:
        value = float(stripped)
    except (TypeError, ValueError) as exc:
        raise ExpertStreamingConfigurationError(
            f"{key} must be a number of GiB, got {raw!r}"
        ) from exc
    if not math.isfinite(value) or value < 0:
        raise ExpertStreamingConfigurationError(
            f"{key} must be finite and non-negative GiB, got {value}"
        )
    return value


def resolve_box_target_mlx_limit_bytes(
    env: Mapping[str, str] | None = None,
    *,
    baseline_bytes: int | None = None,
    memsize_bytes: int | None = None,
) -> dict[str, Any] | None:
    """Resolve a construction-time budget, or None when the target is unset.

    allocator = target - measured system baseline - full Python capacity reserve.
    engine = allocator - max(prefill transient, decode overshoot + allocator cache).
    The cache lives within the allocator budget. MLX peak is active-only and is
    not a process/system measurement. Bands are forecasts; the guarded runner
    must still verify actual usage and full-model peak headroom before execution.
    Target/baseline are decimal GB; cache and reserve settings are GiB.
    """

    source = os.environ if env is None else env
    target_gb = _env_pos_float(source, BOX_TARGET_ENV)
    if target_gb is None:
        # The env may be set to a bare marker meaning "use the default target".  Only
        # arm when the key is present at all; absent -> legacy path (return None).
        if BOX_TARGET_ENV not in source:
            return None
        target_gb = DEFAULT_BOX_TARGET_GB
    baseline_gb = _env_pos_float(source, BOX_BASELINE_ENV)
    if baseline_gb is not None:
        baseline_b = int(round(baseline_gb * _DECIMAL_GB))
    elif baseline_bytes is not None:
        baseline_b = int(baseline_bytes)
        if baseline_b <= 0:
            raise ExpertStreamingConfigurationError("baseline_bytes must be positive")
        baseline_gb = baseline_b / _DECIMAL_GB
    else:
        raise ExpertStreamingConfigurationError(
            f"{BOX_TARGET_ENV} is set but no baseline is available: set "
            f"{BOX_BASELINE_ENV} (decimal GB, measured physical system usage with the "
            "resident agent booted out) or pass baseline_bytes"
        )
    cache_gib = _env_nonneg_float_or_default(
        source, BOX_ALLOC_CACHE_ENV, DEFAULT_ALLOC_CACHE_GIB
    )
    transient_gib = _env_nonneg_float_or_default(
        source, BOX_TRANSIENT_BAND_ENV, DEFAULT_TRANSIENT_BAND_GIB
    )
    from .deepseek_v41_memory_profile import python_cache_budget
    try:
        python_budget = python_cache_budget(source)
    except ValueError as exc:
        raise ExpertStreamingConfigurationError(str(exc)) from exc
    host_gib = _env_nonneg_float_or_default(source, BOX_HOST_OVERHEAD_ENV,
        max(DEFAULT_HOST_OVERHEAD_GIB, python_budget["required_host_bytes"] / _GIB))
    if round(host_gib * _GIB) < python_budget["required_host_bytes"]:
        raise ExpertStreamingConfigurationError(
            "host overhead does not cover configured Python caches and metadata")
    active_gib = _env_nonneg_float_or_default(
        source, BOX_ACTIVE_OVERSHOOT_ENV, DEFAULT_ACTIVE_OVERSHOOT_GIB
    )
    cache_b = int(round(cache_gib * _GIB))
    transient_b = int(round(transient_gib * _GIB))
    host_b = int(round(host_gib * _GIB))
    active_b = int(round(active_gib * _GIB))
    session_b = int(round(_env_nonneg_float_or_default(source, BOX_SESSION_BANK_ENV, 0) * _GIB))
    target_b = int(round(target_gb * _DECIMAL_GB))
    # (HIGH-2 (1)) the allocator/wired limit reserves only the host overhead the MLX
    # allocator can't see; the cache lives WITHIN this joint active+cache limit.
    mlx_limit = target_b - baseline_b - host_b
    if mlx_limit <= 0:
        raise ExpertStreamingConfigurationError(
            f"{BOX_TARGET_ENV} {target_gb:g} GB minus baseline {baseline_gb:.4g} GB "
            f"minus host overhead {host_gib:g} GiB leaves no MLX budget "
            f"({mlx_limit} bytes)"
        )
    if mlx_limit > 100 * _GIB:
        raise ExpertStreamingConfigurationError("derived allocator exceeds 100 GiB wired cap")
    physical = _physical_ram_bytes() if memsize_bytes is None else memsize_bytes
    if physical is not None and target_b + 8 * _DECIMAL_GB > physical:
        raise ExpertStreamingConfigurationError("target plus 8 GB headroom exceeds physical RAM")
    # (HIGH-A) the engine budget (persistent slots == the plan) reserves the WORST of the
    # two regimes under the allocator limit -- they are NOT additive (each is a distinct execution phase): prefill peak = engine + transient_band; decode peak = active
    # (engine + active_overshoot) + cache_room.  engine = allocator - max(band,
    # active_overshoot + cache).
    # Retained owners, an admitted candidate, and one strided-copy scratch leaf.
    engine_reserve_b = max(transient_b, active_b + cache_b) + 3 * session_b
    engine_budget_b = mlx_limit - engine_reserve_b
    if engine_budget_b <= 0:
        raise ExpertStreamingConfigurationError(
            f"{BOX_TARGET_ENV} derives allocator limit {mlx_limit} bytes but the engine "
            f"reserve max(transient band {transient_gib:g}, active overshoot {active_gib:g} "
            f"+ cache {cache_gib:g}) GiB leaves no engine budget ({engine_budget_b} "
            f"bytes); raise {BOX_TARGET_ENV}"
        )
    return {
        "mlx_limit_bytes": mlx_limit,
        "box_target_gb": target_gb,
        "box_baseline_gb": baseline_gb,
        "box_baseline_bytes": baseline_b,
        "host_overhead_gib": host_gib,
        "host_overhead_bytes": host_b,
        "allocator_cache_limit_gib": cache_gib,
        "allocator_cache_limit_bytes": cache_b,
        "transient_band_gib": transient_gib,
        "transient_band_bytes": transient_b,
        "active_overshoot_gib": active_gib,
        "active_overshoot_bytes": active_b,
        "engine_reserve_bytes": engine_reserve_b,
        "session_bank_capacity_bytes": session_b,
        "session_bank_reserve_bytes": 3 * session_b,
        "engine_budget_bytes": engine_budget_b,
        "python_cache_budget": python_budget,
        "physical_ram_bytes": physical,
        "wired_hard_limit_bytes": 100 * _GIB,
    }


def apply_mlx_memory_cap(
    plan: ExpertMemoryPlan,
    *,
    mx_module: Any | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Install allocator, wired and cache limits before model allocation.

    Target mode requires every cap to succeed. Its engine budget has already
    reserved Python capacity and Metal transient/cache room at construction.
    Legacy callers retain their explicit plan/headroom behavior. Limits are
    configuration policies, not measured peaks or hard process-memory bounds.
    """

    target_env = os.environ if env is None else env
    plan_limit = reconcile_mlx_memory_cap(plan, env=target_env)
    # The engine budget env stays the reconciled PLAN value (the config that built the
    # plan already sized it -- to allocator_limit - max(band, active_overshoot + cache)
    # when the target is armed, see the bench).  reconcile stamps it for the cross-check; this
    # function never moves it, so bytes/routing follow the config's slot count.
    target_env["MTPLX_MEMORY_LIMIT_BYTES"] = str(plan_limit)
    headroom = resolve_mlx_limit_headroom_bytes(target_env)
    # W121 + HIGH-2: when the box target is armed, set the allocator limit from the
    # target (target - baseline - host_overhead) instead of the plan, and bound the
    # freed-buffer LRU with set_cache_limit(allocator_cache_limit) ALONE (the cache lives
    # within the joint active+cache policy). The OS guard measures actual usage;
    # this allocation policy is not itself a hard physical-memory bound.  Headroom stays an explicit add-on override.
    box_target = None
    try:
        box_target = resolve_box_target_mlx_limit_bytes(target_env)
    except ExpertStreamingConfigurationError:
        raise
    if box_target is not None:
        # HIGH-B: an explicit headroom ADDED to the target-derived limit pushes box_used
        # ABOVE the target (an *_hr8 arm under an armed target -> +8 GiB -> ~108.6 GB box,
        # 1.4 GB from the panic line, and the receipt/sidecar omit the +8).  The target
        # IS the box budget, so refuse rather than silently overshoot it.
        if headroom > 0:
            raise ExpertStreamingConfigurationError(
                f"{MLX_LIMIT_HEADROOM_ENV} ({headroom} bytes) cannot be combined with an "
                f"armed {BOX_TARGET_ENV}: the target already IS the box budget, so adding "
                "headroom would push box_used above the target. Drop the headroom or the "
                "target."
            )
        base_limit = box_target["mlx_limit_bytes"]
        limit_source = "box_target"
        if base_limit < plan_limit:
            raise ExpertStreamingConfigurationError(
                f"{BOX_TARGET_ENV} derives an MLX allocator limit of {base_limit} "
                f"bytes (target - baseline - host overhead), BELOW the engine budget "
                f"{plan_limit} (residents + KV + expert slots do not fit under the "
                f"target); raise {BOX_TARGET_ENV} or lower --max-kv / the expert-cache "
                "budget"
            )
    else:
        base_limit = plan_limit
        limit_source = "plan"
    limit = base_limit + headroom
    if mx_module is None:
        try:
            import mlx.core as mx
        except Exception as exc:
            return {
                "applied": False,
                "reason": "mlx_unavailable",
                "error": repr(exc),
                "limit": limit,
            }
    else:
        mx = mx_module
    setter = getattr(mx, "set_memory_limit", None)
    if not callable(setter):
        metal = getattr(mx, "metal", None)
        setter = getattr(metal, "set_memory_limit", None)
    if not callable(setter):
        raise ExpertStreamingConfigurationError("MLX memory limit API is unavailable")
    setter(limit)
    # Target-mode admission depends on wiring the working set. A refused or
    # unavailable wired/cache API therefore aborts before any model allocation.
    wired_report: dict[str, Any] = {}
    wired_setter = getattr(mx, "set_wired_limit", None)
    wired_api = "mx.set_wired_limit"
    if not callable(wired_setter):
        metal = getattr(mx, "metal", None)
        wired_setter = getattr(metal, "set_wired_limit", None)
        wired_api = "mx.metal.set_wired_limit"
    if callable(wired_setter):
        try:
            previous = wired_setter(limit)
            wired_report = {
                "wired_limit_applied": True,
                "wired_limit_bytes": limit,
                "wired_limit_api": wired_api,
                "previous_wired_limit_bytes": (
                    int(previous) if previous is not None else None
                ),
            }
        except Exception as exc:  # pragma: no cover - OS/driver refusal path
            wired_report = {
                "wired_limit_applied": False,
                "wired_limit_bytes": limit,
                "wired_limit_error": repr(exc),
            }
    else:
        wired_report = {
            "wired_limit_applied": False,
            "wired_limit_reason": "set_wired_limit_unavailable",
        }
    if box_target is not None and not wired_report["wired_limit_applied"]:
        raise ExpertStreamingConfigurationError(
            f"required wired limit could not be applied: {wired_report}")
    # The retained allocator cache lives within the target-derived limit.
    cache_report: dict[str, Any] = {}
    if box_target is not None:
        cache_bytes = int(box_target["allocator_cache_limit_bytes"])
        cache_setter = getattr(mx, "set_cache_limit", None)
        cache_api = "mx.set_cache_limit"
        if not callable(cache_setter):
            metal = getattr(mx, "metal", None)
            cache_setter = getattr(metal, "set_cache_limit", None)
            cache_api = "mx.metal.set_cache_limit"
        if callable(cache_setter):
            try:
                prev_cache = cache_setter(cache_bytes)
                cache_report = {
                    "cache_limit_applied": True,
                    "cache_limit_bytes": cache_bytes,
                    "cache_limit_api": cache_api,
                    "previous_cache_limit_bytes": (
                        int(prev_cache) if prev_cache is not None else None
                    ),
                }
            except Exception as exc:  # pragma: no cover - OS/driver refusal path
                cache_report = {
                    "cache_limit_applied": False,
                    "cache_limit_bytes": cache_bytes,
                    "cache_limit_error": repr(exc),
                }
        else:
            cache_report = {
                "cache_limit_applied": False,
                "cache_limit_reason": "set_cache_limit_unavailable",
                "cache_limit_bytes": cache_bytes,
            }
        if not cache_report["cache_limit_applied"]:
            raise ExpertStreamingConfigurationError(
                f"required cache limit could not be applied: {cache_report}")
    report: dict[str, Any] = {
        "applied": True,
        "limit": limit,
        "limit_source": limit_source,
        **wired_report,
        **cache_report,
    }
    if box_target is not None:
        report["box_target_gb"] = box_target["box_target_gb"]
        report["box_baseline_gb"] = round(box_target["box_baseline_gb"], 6)
        report["box_target_mlx_limit_bytes"] = int(box_target["mlx_limit_bytes"])
        report["host_overhead_bytes"] = int(box_target["host_overhead_bytes"])
        report["host_overhead_gib"] = round(box_target["host_overhead_gib"], 6)
        report["allocator_cache_limit_bytes"] = int(
            box_target["allocator_cache_limit_bytes"]
        )
        report["allocator_cache_limit_gib"] = round(
            box_target["allocator_cache_limit_gib"], 6
        )
        report["transient_band_bytes"] = int(box_target["transient_band_bytes"])
        report["transient_band_gib"] = round(box_target["transient_band_gib"], 6)
        report["active_overshoot_bytes"] = int(box_target["active_overshoot_bytes"])
        report["active_overshoot_gib"] = round(box_target["active_overshoot_gib"], 6)
        report["engine_reserve_bytes"] = int(box_target["engine_reserve_bytes"])
        report["session_bank_capacity_bytes"] = int(box_target["session_bank_capacity_bytes"])
        report["session_bank_reserve_bytes"] = int(box_target["session_bank_reserve_bytes"])
        report["engine_budget_bytes"] = int(box_target["engine_budget_bytes"])
        report["python_cache_budget"] = box_target["python_cache_budget"]
        report["slot_derivation"] = _slot_derivation_report(plan, box_target)
    return report


def _slot_derivation_report(
    plan: ExpertMemoryPlan, box_target: Mapping[str, Any]
) -> dict[str, Any]:
    """The W121 persistent-slot arithmetic for the receipt (David: "track the memory
    usage so we can efficiently use it").

        engine_budget = allocator_limit - max(transient_band, active_overshoot + cache_room)
        persistent expert slots = (engine_budget - fixed_footprint) / record_bytes

    ``fixed_footprint_source`` is ``plan.fixed_bytes`` -- the plan's ESTIMATE (residents
    + KV + reserves + transient service), NOT a live measurement (LOW: labelled as such).
    ``record_bytes`` is one streamed expert record.  Records what the target SUPPORTS and
    what the plan (built by the config with ``memory_limit_bytes = engine_budget``)
    actually resolved, so a receipt shows both agree; it never re-sizes the live plan."""

    allocator_limit = int(box_target["mlx_limit_bytes"])
    transient_band = int(box_target["transient_band_bytes"])
    active_overshoot = int(box_target["active_overshoot_bytes"])
    cache_room = int(box_target["allocator_cache_limit_bytes"])
    engine_reserve = int(box_target["engine_reserve_bytes"])
    engine_budget = int(box_target["engine_budget_bytes"])
    fixed_bytes = int(getattr(plan, "fixed_bytes", 0) or 0)
    persistent_slots_plan = int(getattr(plan, "persistent_slots", 0) or 0)
    persistent_cache_bytes = int(getattr(plan, "persistent_cache_bytes", 0) or 0)
    record_bytes = (
        persistent_cache_bytes // persistent_slots_plan
        if persistent_slots_plan > 0
        else None
    )
    slot_allowance = engine_budget - fixed_bytes
    slots_at_target = (
        max(0, slot_allowance) // record_bytes if record_bytes else None
    )
    return {
        "formula": (
            "engine_budget = allocator_limit - max(transient_band, active_overshoot + "
            "cache_room) - session_bank_reserve; persistent_slots = "
            "(engine_budget - fixed_footprint) / record_bytes"
        ),
        "allocator_limit_bytes": allocator_limit,
        "transient_band_bytes": transient_band,
        "active_overshoot_bytes": active_overshoot,
        "cache_room_bytes": cache_room,
        "engine_reserve_bytes": engine_reserve,
        "session_bank_reserve_bytes": int(box_target.get("session_bank_reserve_bytes", 0)),
        "engine_budget_bytes": engine_budget,
        # NB: the plan's ESTIMATE of the fixed footprint, not a live measurement.
        "fixed_footprint_bytes_plan_estimate": fixed_bytes,
        "slot_allowance_bytes": slot_allowance,
        "record_bytes": record_bytes,
        "persistent_slots_at_target": slots_at_target,
        "persistent_slots_plan": persistent_slots_plan,
        "persistent_cache_bytes_plan": persistent_cache_bytes,
    }


def mlx_memory_telemetry(mx_module: Any | None = None) -> dict[str, int | str]:
    if mx_module is None:
        try:
            import mlx.core as mx
        except Exception as exc:
            return {"error": repr(exc)}
    else:
        mx = mx_module
    report: dict[str, int | str] = {}
    for name in ("get_active_memory", "get_peak_memory", "get_cache_memory"):
        getter = getattr(mx, name, None)
        if not callable(getter):
            getter = getattr(getattr(mx, "metal", None), name, None)
        if callable(getter):
            try:
                report[name.removeprefix("get_") + "_bytes"] = int(getter())
            except Exception as exc:
                report[name + "_error"] = repr(exc)
    return report


# W64 (R3-pin) env flags. Read at USE (never frozen at import), matching every
# other DSV4.1 lever ([[env-flags-read-at-use-not-import]]): the server stamps
# the optimization keys AFTER importing the runtime module.
PIN_WORKING_SET_ENV = "MTPLX_DSV41_PIN_WORKING_SET"
PIN_REFRESH_TOKENS_ENV = "MTPLX_DSV41_PIN_REFRESH_TOKENS"
# W71 (K24 revived): the barrier-free device route guarded by the W64 pins. When
# "1" the switch takes the pinned device path (gather over the pinned-only LUT,
# defer the all-pinned check) and the backbone establishes pins out-of-band at the
# prefill->decode boundary + arms the cold-recovery flush. Default off.
DEVICE_ROUTE_PINNED_ENV = "MTPLX_DSV41_DEVICE_ROUTE_PINNED"

# Reusable no-op context for the "no per-layer lock" branch (global-lock banks
# are already excluded from W64, so this only stands in for a missing lock).
_NULL_CTX = nullcontext()


def parse_pin_working_set(value: str | None) -> tuple[str, float | None] | None:
    """Parse ``MTPLX_DSV41_PIN_WORKING_SET`` into a pin spec, or None (off).

    - unset / ``0`` / ``off`` / ``false`` / ``no`` / empty  -> None (default off)
    - ``all`` / ``keys``                                    -> ``("all", None)``
      (pin every resident expert -> a fully static layer; the ``pin_ws`` arm)
    - a fraction in ``(0, 1]`` (``0.5``, ``50%``, ``1.0``)  -> ``("frac", f)``
      (pin ``round(f * persistent_slots)`` experts)
    - a positive integer ``K``                              -> ``("slots", K)``
      (pin the top ``K`` resident experts)

    Ambiguity rule: a bare integer is a slot COUNT; write ``1.0`` / ``100%`` /
    ``all`` for "the whole set". Anything unparseable is treated as off.
    """

    if value is None:
        return None
    raw = value.strip().lower()
    if not raw or raw in {"0", "off", "false", "no"}:
        return None
    if raw in {"all", "keys", "1.0", "100%"}:
        return ("all", None)
    try:
        if raw.endswith("%"):
            frac = float(raw[:-1]) / 100.0
            return ("frac", frac) if 0.0 < frac <= 1.0 else None
        if "." in raw:
            frac = float(raw)
            return ("frac", frac) if 0.0 < frac <= 1.0 else None
        count = int(raw)
        return ("slots", float(count)) if count >= 1 else None
    except ValueError:
        return None


def parse_pin_refresh_tokens(value: str | None) -> int:
    """Decode ``MTPLX_DSV41_PIN_REFRESH_TOKENS`` (>=1 re-ranks every N decode
    epochs; 0 / unset / bad = pin once after prefill and never refresh)."""

    if not value:
        return 0
    try:
        n = int(value.strip())
    except ValueError:
        return 0
    return n if n >= 1 else 0


# W87: first N DECODE tokens counted as "cold start" for the first-N-token vs
# steady-state decode hit-rate receipt fields (both slot-pool paths populate them).
_COLD_START_DECODE_TOKENS = 64


class ExpertStreamingRuntime:
    """Connect cache policy, checked I/O, fixed slots, and KV admission."""

    def __init__(
        self,
        root: Path,
        spec: ExpertStreamingModelSpec,
        config: ExpertStreamingConfig,
        manifest: ExpertManifest,
        plan: ExpertMemoryPlan,
        reader: PositionalExpertReader,
        slots: ExpertSlotPool,
        *,
        memory_cap_report: dict[str, Any] | None = None,
        integrity_report: dict[str, Any] | None = None,
        pipeline_ledger: ExpertPipelineLedger | None = None,
        single_slot_pool: bool = False,
    ) -> None:
        self.root = root
        self.spec = spec
        self.config = config
        self.manifest = manifest
        self.plan = plan
        self.reader = reader
        self.slots = slots
        self.memory_cap_report = memory_cap_report
        self.integrity_report = integrity_report
        self._pipeline_ledger = pipeline_ledger
        self._decode_miss_records_per_part = int(
            config.decode_miss_records_per_part or 0
        )
        self._decode_miss_placement = (
            {
                (record.layer, record.expert): (
                    int(record.part),
                    int(record.sidecar_offset),
                    int(record.sidecar_length),
                )
                for record in manifest.records
            }
            if config.decode_miss_records_per_part is not None
            else {}
        )
        # Mixed-official (issue #51, M2): per-layer record bytes for telemetry
        # accounting, since ``spec.expert_record_bytes`` raises for mixed. The
        # representative (exemplar) value stands in where a single scalar is
        # needed for a coarse aggregate.
        self._per_layer_record_bytes = (
            manifest.record_bytes_by_layer()
            if spec.is_mixed_official
            else None
        )
        self._representative_record_bytes = (
            self._per_layer_record_bytes[spec.routed_layer_indices[0]]
            if self._per_layer_record_bytes is not None
            else spec.expert_record_bytes
        )
        self.counters = CacheCounters()
        self._layer_counters = {
            layer: CacheCounters() for layer in spec.routed_layer_indices
        }
        self._phase_counters = {phase: CacheCounters() for phase in RoutingPhase}
        self._counter_lock = threading.Lock()
        self._global_bank = (
            GlobalExpertSlotBank(
                layer_indices=spec.routed_layer_indices,
                expert_count=spec.expert_count,
                persistent_slots=plan.persistent_slots,
                transient_slots=plan.transient_slots,
                prefill_slots_per_layer=plan.slots_per_layer,
                frequency_decay=config.frequency_decay,
                cache_policy=config.cache_policy,
            )
            if config.cache_scope == "global"
            else None
        )
        # Wired and mmap-banked islands share the routing contract: streamed
        # route entry points reject both, and neither owns slot-pool state.
        self.island_layer_set = frozenset(config.island_layers) | frozenset(
            config.mmap_island_layers
        )
        # W87 single-slot pool (env MTPLX_DSV41_SINGLE_SLOT_POOL): merge each
        # layer's persistent + transient tiers into ONE scan-resistant resident
        # pool (only layer cache scope; the two-tier path is byte-identical off).
        self._single_slot_pool = bool(single_slot_pool)
        # W87 cold-start decode telemetry: first-N-token vs steady-state decode
        # hit rate, ALWAYS ON so both paths report the same receipt fields.
        self._streamed_layer_set = (
            frozenset(spec.routed_layer_indices) - self.island_layer_set
        )
        self._cold_start_decode_tokens = _COLD_START_DECODE_TOKENS
        self._decode_token_index = 0
        self._decode_layers_seen: set[int] = set()
        self._cold_decode_hits = 0
        self._cold_decode_requests = 0
        self._steady_decode_hits = 0
        self._steady_decode_requests = 0
        # MED-4: re-open the cold window on the first PREFILL route after decode
        # (a new request), so each request's first decode steps are measured fresh.
        self._saw_decode_since_prefill = False
        # W93: ONE shared prefetch ring across all routed layers (the memory bound
        # -- ``prefetch_ring_slots`` records TOTAL, not per-layer). Every bank
        # delegates its ring bookkeeping to this instance, keyed by its layer.
        self._prefetch_ring = (
            GlobalPrefetchRing(
                ring_size=plan.prefetch_ring_slots,
                base=plan.slots_per_layer + plan.transient_slots,
                expert_count=spec.expert_count,
            )
            if plan.prefetch_ring_slots > 0 and self._global_bank is None
            else None
        )
        self._banks = (
            {}
            if self._global_bank is not None
            else {
                layer: LayerExpertSlotBank(
                    expert_count=spec.expert_count,
                    persistent_slots=plan.slots_per_layer,
                    transient_slots=plan.transient_slots,
                    frequency_decay=config.frequency_decay,
                    cache_policy=config.cache_policy,
                    prefetch_slots=plan.prefetch_ring_slots,
                    single_pool=self._single_slot_pool,
                    layer_id=layer,
                    prefetch_ring=self._prefetch_ring,
                )
                for layer in spec.routed_layer_indices
                if layer not in self.island_layer_set
            }
        )
        if self._global_bank is not None:
            # A route holds this lock through hit execution and miss loading.
            # Physical pinning remains the final overwrite fence, while this
            # lock prevents another layer from selecting the same global
            # victim before the current transaction publishes its mapping.
            global_lock = threading.Lock()
            self._layer_locks = {
                layer: global_lock for layer in spec.routed_layer_indices
            }
        else:
            self._layer_locks = {
                layer: threading.Lock() for layer in spec.routed_layer_indices
            }
        self._kv_lock = threading.Lock()
        self._live_kv_tokens = 0
        self._live_kv_peak = 0
        self._pending_kv_tokens = 0
        # Derived single-limit policy state. The allowance is recomputed only
        # at KV boundaries; cache hits never touch any of this.
        # Mixed-official specs (issue #51, M2) have no uniform
        # ``spec.resident_bytes`` (it raises), so reconstruct the resident
        # baseline from the manifest-derived per-layer record sizes exactly as
        # ``plan_expert_memory`` does; the additional-resident accounting must
        # hold on both the uniform and mixed lanes.
        if spec.is_mixed_official:
            spec_resident_baseline = spec.total_tensor_bytes - spec.expert_count * sum(
                self._per_layer_record_bytes[layer]
                for layer in spec.routed_layer_indices
            )
        else:
            spec_resident_baseline = spec.resident_bytes
        self._additional_resident_bytes = plan.resident_bytes - spec_resident_baseline
        self._derived_cache_policy = config.derived_expert_cache_policy
        self._allowance_lock = threading.Lock()
        self._derived_allowance_bytes: int | None = None
        self._derived_capacity_slots: int | None = None
        if self._derived_cache_policy:
            # Post-load boundary: the open-time plan is the zero-KV plan, so
            # its budget is the initial allowance and the pool size is the
            # initial capacity.
            self._derived_allowance_bytes = max(
                0, derived_expert_cache_allowance_bytes(plan)
            )
            self._derived_capacity_slots = (
                plan.persistent_slots
                if config.cache_scope == "global"
                else plan.slots_per_layer
            )
        self._close_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._cleanup_error_lock = threading.Lock()
        self._cleanup_error: BaseException | None = None
        self._mapped_expert_store: Any | None = None
        self._island_store: Any | None = None
        self._banked_island_store: Any | None = None
        self._shadow_store: Any | None = None
        # W44 device-route (barrier-free all-hit, K24): per-layer device LUT
        # (expert -> persistent bank row, -1 = non-resident), rebuilt on the host
        # ONLY when that layer's residency changes; the resident snapshot each LUT
        # was built from; a dirty flag per layer; the per-layer component bank
        # captured from the first fenced binding (so the device path gathers
        # without a routed binding); and the async verification queue.
        self._device_route_lut: dict[int, Any] = {}
        self._device_route_lut_snapshot: dict[int, frozenset[int]] = {}
        self._device_route_lut_dirty: dict[int, bool] = {}
        self._device_route_bank: dict[int, Any] = {}
        self._device_route_probes: list[tuple[int, Any, frozenset[int], bool]] = []
        # W71 (K24 revived, env ``MTPLX_DSV41_DEVICE_ROUTE_PINNED``): a parallel
        # PINNED-only LUT (expert -> slot for PINNED experts, -1 for everything
        # else, so the device gather is exact only on an all-pinned route -- whose
        # slots W64 guarantees cannot be recycled by normal decode admission). It
        # is rebuilt on the host only when the layer's PIN set changes (a pin /
        # refresh, or a memory-forced capacity eviction that unpins an expert),
        # tracked by its own dirty flag so it never rebuilds off a mere LRU churn
        # of the unpinned free tail. Telemetry counts the barrier-free (all-pinned,
        # kept) vs recovered (not-all-pinned, refenced) layers per flush.
        self._device_route_pinned_lut: dict[int, Any] = {}
        self._device_route_pinned_snapshot: dict[int, frozenset[int]] = {}
        self._device_route_pinned_lut_dirty: dict[int, bool] = {}
        self._device_route_pinned_flushes = 0
        self._device_route_pinned_barrier_free_layers = 0
        self._device_route_pinned_recovered_layers = 0
        # Layers the backbone has forced back onto the fenced path for a W44 cold
        # recovery pass (the switch skips the device path for these). Empty in the
        # steady state; set/cleared around a recovery re-run by the decode forward.
        self._device_route_force_fenced: frozenset[int] = frozenset()
        # W64 (R3-pin): post-prefill pinned working set (env
        # ``MTPLX_DSV41_PIN_WORKING_SET``; default off -> every path below is a
        # byte-identical no-op). Per-layer banks only. ``_pin_last_epoch`` gates
        # the pin trigger to the first decode route after prefill (and to every
        # ``MTPLX_DSV41_PIN_REFRESH_TOKENS`` decode epochs when set); the two
        # counters feed the all-pinned-hit-rate telemetry surfaced in the
        # snapshot / A/B receipt / served event.
        self._pin_last_epoch: dict[int, int] = {}
        self._pin_telemetry_lock = threading.Lock()
        self._pin_decode_routes = 0
        self._pin_all_pinned_routes = 0
        self._shadow_serve_routes = 0
        self._shadow_serve_assignments = 0
        self._shadow_serve_experts = 0
        # Decode-route census (issue #98): pure host-side counters, flushed
        # to the model root at close(). Never touches routing or numerics.
        self._census_lock = threading.Lock()
        self._route_census = (
            RouteCensus(spec.key) if config.route_census else None
        )
        # Runtime Belady oracle (diagnostic, env MTPLX_BELADY_ORACLE=1): the
        # eviction-ceiling analog of the census. Records decode routes for
        # streamed layers and reports the clairvoyant fetch floor at snapshot.
        # Zero cost when unset; never touches the decode data path.
        self._belady_oracle = (
            BeladyOracle(island_layers=self.island_layer_set)
            if BeladyOracle.enabled()
            else None
        )
        self._route_trace_lock = threading.Lock()
        self._route_trace: list[dict[str, Any]] = []
        self._route_trace_epoch = 0
        self._route_trace_decode_step = 0
        self._route_trace_decode_layers_seen: set[int] = set()
        self._incremental_miss_routes = 0
        self._incremental_miss_parts = 0
        self._split_executor = ThreadPoolExecutor(
            max_workers=max(1, plan.transient_slots),
            thread_name_prefix="mtplx-route-miss",
        )
        # Speculative ring loads run off the route-miss executor so a
        # prediction burst can never starve a real miss read.
        self._prefetch_lock = threading.Lock()
        self._prefetch_futures: set[Future] = set()
        # Settled speculative loads awaiting publication, per layer.
        # Workers only enqueue here; the generation thread applies them
        # at its next prefetch_experts call for the layer.
        self._prefetch_completions: dict[
            int, list[tuple[int, int | None, bool]]
        ] = {}
        # W93: (layer, expert) -> the inflight speculative read's future, so the
        # DEMAND route can await a needed expert's in-flight prefetch instead of
        # issuing a duplicate read (``_reconcile_prefetch_for_route``). Registered
        # under ``_prefetch_lock`` alongside ``_prefetch_futures``; dropped when the
        # read settles (``_finish_prefetch_load``).
        self._prefetch_inflight_futures: dict[tuple[int, int], Future] = {}
        # W100: (layer, expert, ticket) of speculative reads issued during a
        # DSpark VERIFY-phase prefetch, so the async commit path can attribute
        # ``prefetch_committed_verify`` to the verify without the completion
        # carrying its issuing phase. Written in ``prefetch_experts`` (verify=True)
        # and consumed/discarded in ``_apply_prefetch_completions``; guarded by
        # ``_prefetch_lock`` (same as the completions/futures indices).
        self._verify_prefetch_tags: set[tuple[int, int, int | None]] = set()
        # Each layer's most recent route-plan misses (updated under the
        # layer lock): their bytes are streaming in on the demand path or
        # freshly transient-resident, so predicting them again would only
        # duplicate the read.
        self._recent_route_misses: dict[int, frozenset[int]] = {}
        # Sustained decode can predict faster than the executor reads.
        # Without a ceiling the queue grows without bound: every queued
        # load is stale by the time it runs (its assignment recycled many
        # times over), reads are wasted, and reset/close must drain the
        # whole backlog. Beyond this ceiling prefetch_experts plans
        # nothing — speculation is best-effort.
        self._prefetch_backlog_limit = 4 * max(
            1, plan.prefetch_ring_slots
        )
        # Speculative I/O admission: speculation may occupy at most a
        # configured fraction of the inflight read budget, so a demand
        # miss read never queues behind a burst of predictions at the
        # SSD. Concurrency floors at one read (speculation is throttled,
        # never starved) and the ring width still bounds it above.
        #
        # W93 lane C (HIGH-3): the fraction budget above can still resolve to a
        # large executor (~12 reads == ~216 MiB in flight on the 75 GiB profile),
        # which -- on its own bypass-admission executor -- let a prediction burst
        # race the current layer's demand misses. Cap the ABSOLUTE number of
        # concurrent speculative reads (executor max_workers, and thus in-flight
        # single-record reads) to a small, strictly-smaller-than-demand bound,
        # default 4. Configurable via MTPLX_DSV41_GATE_PREFETCH_MAX_INFLIGHT, read
        # here at construction (use, not import); floored at 1 so speculation is
        # throttled, never starved. The demand-miss executor (_split_executor,
        # max_workers=transient_slots) is not bounded by this cap.
        _spec_cap_raw = os.environ.get(
            "MTPLX_DSV41_GATE_PREFETCH_MAX_INFLIGHT", "4"
        )
        try:
            _spec_inflight_cap = max(1, int(_spec_cap_raw))
        except (TypeError, ValueError):
            _spec_inflight_cap = 4
        if config.prefetch_slots > 0:
            inflight_budget = (
                config.max_inflight_io_bytes
                if config.max_inflight_io_bytes is not None
                else slots.max_inflight_io_bytes
            )
            budget_reads = int(
                inflight_budget * config.speculative_io_fraction
            ) // max(1, spec.expert_record_bytes)
            self._prefetch_max_reads = max(
                1,
                min(
                    budget_reads,
                    max(1, plan.prefetch_ring_slots),
                    _spec_inflight_cap,
                ),
            )
        else:
            self._prefetch_max_reads = 0
        self._prefetch_executor = (
            ThreadPoolExecutor(
                max_workers=self._prefetch_max_reads,
                thread_name_prefix="mtplx-prefetch",
            )
            if config.prefetch_slots > 0
            else None
        )
        # W93 lane C: split I/O-byte accounting for the receipt (lane B's snapshot
        # block reads these by name). ``speculative_bytes_read`` is bumped by the
        # speculative executor (``_run_speculative_load``); ``demand_bytes_read``
        # by the reconcile fallback (``_reconcile_prefetch_for_route``) when a
        # gate-predicted expert's speculative read timed out or failed and the
        # route had to stream it on the demand path instead.
        self.speculative_bytes_read = 0
        self.demand_bytes_read = 0
        # W95g (HIGH-1 re-review): strict demand priority via a per-token
        # speculative SHARE with a floor -- NOT a cumulative spec<=(budget x demand)
        # cap. The old rule (budget 0.5, cumulative) combined with the recovering
        # window and the negative feedback of every hidden miss shrinking the demand
        # denominator settled to spec/(spec+demand) <= 1/3, so a real-runtime probe
        # showed only the first call in a token issuing and every later call
        # skipped, throttling to ~16-25 issued/token vs the ~149 the design cost
        # model (§5) needs. The rule is now: skip only when this token's speculative
        # bytes exceed a SHARE ``f`` of the token's total (spec+demand) SSD bytes,
        # PLUS a floor of ``floor_records`` records so early speculation (small or
        # zero demand) is never throttled. ``MTPLX_DSV41_GATE_PREFETCH_BYTE_BUDGET``
        # keeps its name but now sets the share ``f``; 0 disables the budget
        # entirely (what the live windows use right now). Default f = 0.85 under the
        # v2 runner, else 0.0 (no budget -- the GATE_PREFETCH-only path).
        _budget_raw = os.environ.get("MTPLX_DSV41_GATE_PREFETCH_BYTE_BUDGET")
        if _budget_raw is not None:
            try:
                self._prefetch_byte_budget = max(0.0, float(_budget_raw))
            except (TypeError, ValueError):
                self._prefetch_byte_budget = 0.0
        elif os.environ.get("MTPLX_DSV41_RUNNER") == "v2":
            self._prefetch_byte_budget = 0.85
        else:
            self._prefetch_byte_budget = 0.0
        # W95g: the floor (in expert records) below which speculation is never
        # throttled regardless of the share -- so a token's first prefetch calls
        # always issue. Env-overridable (MTPLX_DSV41_GATE_PREFETCH_BYTE_FLOOR);
        # default 8 records.
        _floor_raw = os.environ.get("MTPLX_DSV41_GATE_PREFETCH_BYTE_FLOOR")
        if _floor_raw is not None:
            try:
                self._prefetch_byte_floor_records = max(0, int(_floor_raw))
            except (TypeError, ValueError):
                self._prefetch_byte_floor_records = 8
        else:
            self._prefetch_byte_floor_records = 8
        self._prefetch_budget_skips = 0
        # W95g: prefetch_experts calls that reached the budget decision, so the
        # receipt exposes budget_skips / prefetch_calls (the visible throttle ratio).
        self._prefetch_calls = 0
        # W95f: the speculative-byte BUDGET is a RECOVERING per-decode-token window,
        # not a cumulative-since-open latch. ``demand_bytes_read`` and
        # ``speculative_bytes_read`` stay cumulative (the receipt reports them and
        # normalises per decode token); the budget in ``prefetch_experts`` compares
        # the DELTAS since the last token boundary, marked here and re-snapshotted
        # whenever ``_decode_token_index`` advances (``_observe_cold_start_unlocked``)
        # / at ``reset``. Without the window the budget latched off after the first
        # fallback (spec >> demand forever); with it each token races demand afresh.
        self._demand_bytes_at_token_start = 0
        self._speculative_bytes_at_token_start = 0
        # W93 lane C (MED-a): bound the wait ``_reconcile_prefetch_for_route`` holds
        # under the layer lock on an in-flight speculative read. Default 2.0 s;
        # configurable via MTPLX_DSV41_GATE_PREFETCH_RECONCILE_TIMEOUT_S (read at
        # construction). On timeout the route falls back to a demand load.
        _reconcile_timeout_raw = os.environ.get(
            "MTPLX_DSV41_GATE_PREFETCH_RECONCILE_TIMEOUT_S", "2.0"
        )
        try:
            self._prefetch_reconcile_timeout_s = max(
                0.0, float(_reconcile_timeout_raw)
            )
        except (TypeError, ValueError):
            self._prefetch_reconcile_timeout_s = 2.0

    @classmethod
    def open(
        cls,
        root: Path | str,
        manifest_path: Path | str,
        config: ExpertStreamingConfig,
        *,
        spec: ExpertStreamingModelSpec | None = None,
        buffer_allocator: Callable[[int, str], Any] | None = None,
        device_synchronize: Callable[[], None] | None = None,
        apply_memory_cap: bool = True,
        mx_module: Any | None = None,
        env: dict[str, str] | None = None,
        additional_resident_bytes: int = 0,
        expert_admission_receipt: Mapping[str, Any] | None = None,
    ) -> ExpertStreamingRuntime:
        artifact_root = Path(root).resolve()
        model_spec = get_model_spec(config.model_key) if spec is None else spec
        if model_spec.key != config.model_key:
            raise ExpertStreamingConfigurationError("config and spec model keys differ")
        validate_deepseek_v41_runtime_env(config.model_key)
        # Optimization-profile enforcement (issue #99): knobs the profile
        # marks not_applicable fail loudly when explicitly forced. This
        # generalizes resident_loader's _verify_kv_quant_honored probe to
        # config time — pure config inspection, no new runtime probes.
        profile_violations = not_applicable_violations(
            config.model_key,
            {knob: getattr(config, knob, None) for knob in ENFORCEABLE_KNOBS},
        )
        if profile_violations:
            raise ExpertStreamingConfigurationError(
                "; ".join(profile_violations)
            )
        config = resolve_island_placement(config, artifact_root, spec=model_spec)
        streamed_codec_spec = getattr(model_spec, "expert_codec", "affine")
        mixed_official = streamed_codec_spec == MIXED_OFFICIAL_CODEC
        if streamed_codec_spec != "affine":
            # q1 (shadow-codec) artifacts (issue #51): records are packed
            # sign/trit words plus one bf16 scale, executed through
            # shadow_gather_mm. Only the component-banks streamed dispatch
            # carries the codec branch (gate 4). The direct-slot, mapped,
            # and dense-island dispatches all assume the affine triple and
            # would silently misread the record, so reject them loudly.
            if config.slot_layout != "component-banks":
                raise ExpertStreamingConfigurationError(
                    f"{model_spec.key} is a {model_spec.expert_codec} streamed "
                    "artifact; it requires the component-banks slot layout, "
                    f"not {config.slot_layout!r}"
                )
            if config.miss_shadow is not None:
                raise ExpertStreamingConfigurationError(
                    "miss_shadow shadows an exact affine artifact with a "
                    f"low-precision bank; a {model_spec.expert_codec} artifact "
                    "is already low precision, so shadowing it is nonsense — "
                    "set miss_shadow to None"
                )
            if config.island_layers or config.mmap_island_layers:
                raise ExpertStreamingConfigurationError(
                    "dense-island execution dispatches through the affine "
                    f"gather kernel; {model_spec.expert_codec} artifacts cannot "
                    "serve island or mmap-island layers"
                )
        if mixed_official:
            # Mixed-official MVP serving lane (issue #51 M2, D6): the
            # component-banks lane with per-layer banks only. The prefetch ring
            # and global cache scope are perf lanes not yet wired for
            # non-uniform records; reject them loudly so E1 runs the exact path.
            if config.prefetch_slots:
                raise ExpertStreamingConfigurationError(
                    "mixed-official banks do not serve the prefetch ring in the "
                    "MVP lane; set prefetch_slots to 0"
                )
            if config.cache_scope != "layer":
                raise ExpertStreamingConfigurationError(
                    "mixed-official banks require cache_scope 'layer' (per-layer "
                    "banks); global scope shares slots across differing records"
                )
        manifest = load_expert_manifest(manifest_path)
        cls._validate_manifest_identity(manifest, model_spec)
        if config.decode_miss_records_per_part is not None:
            if manifest.sidecar is None or any(
                record.sidecar_offset is None or record.sidecar_length is None
                for record in manifest.records
            ):
                raise ExpertStreamingConfigurationError(
                    "decode_miss_records_per_part requires construction-verified "
                    "sidecar placement for every expert record"
                )
        if config.island_layers and not set(config.island_layers) <= set(
            model_spec.routed_layer_indices
        ):
            raise ExpertStreamingConfigurationError(
                "island_layers must be routed layers of "
                f"{model_spec.key}: {sorted(model_spec.routed_layer_indices)[:3]}"
                f"..{sorted(model_spec.routed_layer_indices)[-1]}"
            )
        if config.mmap_island_layers:
            if not set(config.mmap_island_layers) <= set(
                model_spec.routed_layer_indices
            ):
                raise ExpertStreamingConfigurationError(
                    "mmap_island_layers must be routed layers of "
                    f"{model_spec.key}"
                )
            from mtplx.expert_banked import (
                BankedManifestError,
                load_banked_manifest,
            )

            try:
                banked = load_banked_manifest(Path(config.banked_manifest))
            except BankedManifestError as exc:
                raise ExpertStreamingConfigurationError(
                    f"banked manifest is unusable: {exc}"
                ) from exc
            uncovered = sorted(
                set(config.mmap_island_layers) - banked.layer_set
            )
            if uncovered:
                raise ExpertStreamingConfigurationError(
                    f"banked manifest does not cover mmap island layers "
                    f"{uncovered}"
                )
            if banked.expert_count != model_spec.expert_count:
                raise ExpertStreamingConfigurationError(
                    f"banked manifest holds {banked.expert_count} experts per "
                    f"layer; {model_spec.key} routes {model_spec.expert_count}"
                )
        integrity_report = None
        if config.verify_artifact_headers or config.verify_sidecar_hash_at_open:
            if config.verify_sidecar_hash_at_open and manifest.sidecar is None:
                raise ExpertStreamingConfigurationError(
                    "verified-sidecar mode requires a sidecar manifest"
                )
            integrity_report = verify_expert_manifest(
                manifest,
                artifact_root,
                verify_sidecar_hash=config.verify_sidecar_hash_at_open,
            )
        # W95 v2 runner (MTPLX_DSV41_RUNNER=v2) arms the single scan-resistant pool
        # (admit every miss) as part of its one composed switch, without the user
        # stacking MTPLX_DSV41_SINGLE_SLOT_POOL. Inline env read (no import of the
        # deepseek_v41 helper) keeps this low-level module free of that cycle; the
        # value is byte-identical to the pre-v2 path when MTPLX_DSV41_RUNNER is unset.
        _ssp_env = (
            os.environ.get("MTPLX_DSV41_SINGLE_SLOT_POOL") == "1"
            or os.environ.get("MTPLX_DSV41_RUNNER") == "v2"
        )
        # MED-6: the single slot pool is implemented only for per-layer banks
        # (GlobalExpertSlotBank has no pool policy and would fault every prefill
        # wave). Gate it on layer scope and warn rather than crash a served config.
        # The transition-window policy is itself a construction-time request for
        # the per-layer merged pool; it does not depend on a second environment
        # switch.  Other policies retain the existing runner/env selection.
        single_slot_pool = (
            config.cache_policy == TRANSITION_WINDOW_CACHE_POLICY
            or (_ssp_env and config.cache_scope == "layer")
        )
        if _ssp_env and config.cache_scope != "layer":
            _LOGGER.warning(
                "MTPLX_DSV41_SINGLE_SLOT_POOL ignored: it requires cache_scope "
                "'layer' (got %r); running the two-tier path.",
                config.cache_scope,
            )
        if (
            config.cache_policy == TRANSITION_WINDOW_CACHE_POLICY
            and model_spec.expert_count != 384
        ):
            raise ExpertStreamingConfigurationError(
                "transition-window cache policy requires exactly 384 experts"
            )
        plan = config.memory_plan(
            model_spec,
            additional_resident_bytes=additional_resident_bytes,
            resident_discount_bytes=proj_quant_plan_discount(
                manifest, config.proj_quant
            )
            + proj_requant_plan_discount(manifest, config.proj_requant)
            + text_only_resident_discount(manifest, model_spec),
            layer_record_bytes=(
                manifest.record_bytes_by_layer() if mixed_official else None
            ),
        )
        if not plan.fits_fixed:
            raise ExpertStreamingConfigurationError(
                f"fixed expert-streaming footprint exceeds limit by {-plan.unallocated_bytes} bytes"
            )
        cap_report = (
            apply_mlx_memory_cap(plan, mx_module=mx_module, env=env)
            if apply_memory_cap
            else None
        )
        codec_sidecar = None
        if config.streamed_codec != "none":
            from mtplx.expert_streamed_codec import (
                StreamedCodecError,
                load_streamed_codec_manifest,
                validate_against_base,
            )

            codec_manifest_path = Path(config.streamed_codec_manifest)
            if not codec_manifest_path.is_absolute():
                codec_manifest_path = artifact_root / codec_manifest_path
            try:
                codec_sidecar = load_streamed_codec_manifest(codec_manifest_path)
                if codec_sidecar.codec != config.streamed_codec:
                    raise StreamedCodecError(
                        f"streamed codec sidecar codec {codec_sidecar.codec!r} "
                        f"does not match config {config.streamed_codec!r}"
                    )
                validate_against_base(codec_sidecar, manifest)
            except StreamedCodecError as exc:
                raise ExpertStreamingConfigurationError(
                    f"streamed codec sidecar is unusable: {exc}"
                ) from exc
        pipeline_ledger = _pipeline_ledger_for_config(config)
        pipeline_kwargs = (
            {} if pipeline_ledger is None else {"pipeline_ledger": pipeline_ledger}
        )
        admission_kwargs: dict[str, Any] = {}
        if expert_admission_receipt is not None:
            admitted_banks = expert_admission_receipt.get("banks")
            admitted_identities = (
                [(bank.get("file"), bank.get("sha256")) for bank in admitted_banks]
                if isinstance(admitted_banks, list)
                and all(isinstance(bank, Mapping) for bank in admitted_banks)
                else None
            )
            expected_identities = (
                [(part.file, part.sha256) for part in manifest.sidecar.parts]
                if manifest.sidecar is not None
                else []
            )
            if (
                expert_admission_receipt.get("manifest_sha256")
                != manifest.manifest_sha256
                or admitted_identities != expected_identities
            ):
                raise ExpertStreamingConfigurationError(
                    "expert admission receipt banks do not match the manifest"
                )
            admission_kwargs["expert_admission_receipt"] = (
                expert_admission_receipt
            )
        # W123: io read-fanout arm/kill-switch. The reader's ``io_read_fanout``
        # (default 1 == OFF, byte-identical single scatter) splits a record's
        # contiguous read into N concurrent sub-reads to raise SSD queue depth
        # off the QD1 floor. An inline env override lets a window arm or kill the
        # lever without a code change (like MTPLX_DSV41_SINGLE_SLOT_POOL above):
        # set MTPLX_DSV41_IO_READ_FANOUT to an int >= 1 (1 disables the fanout).
        # Unset -> config.io_read_fanout (default 1), so the shipped path is
        # unchanged. Gated to DeepSeek-V4.1 configs so the DSV4.1-named env never
        # reshapes another model's reader (red-team MEDIUM); config.io_read_fanout
        # still applies to any model.
        _io_read_fanout = config.io_read_fanout
        _fanout_env = (
            os.environ.get("MTPLX_DSV41_IO_READ_FANOUT")
            if str(config.model_key).startswith("deepseek-v41")
            else None
        )
        if _fanout_env is not None:
            try:
                _parsed_fanout = int(_fanout_env)
            except ValueError:
                _LOGGER.warning(
                    "MTPLX_DSV41_IO_READ_FANOUT=%r is not an integer; using "
                    "config io_read_fanout=%d",
                    _fanout_env,
                    config.io_read_fanout,
                )
            else:
                if _parsed_fanout >= 1:
                    _io_read_fanout = _parsed_fanout
                else:
                    _LOGGER.warning(
                        "MTPLX_DSV41_IO_READ_FANOUT=%d must be >= 1; using "
                        "config io_read_fanout=%d",
                        _parsed_fanout,
                        config.io_read_fanout,
                    )
        try:
            reader = PositionalExpertReader(
                artifact_root,
                max_open_files=config.max_open_files,
                max_read_chunk_bytes=config.max_read_chunk_bytes,
                bypass_page_cache=config.bypass_page_cache,
                codec_sidecar=codec_sidecar,
                codec_verify=config.streamed_codec_verify,
                io_read_fanout=_io_read_fanout,
                **pipeline_kwargs,
                **admission_kwargs,
            )
        except ExpertIOError as exc:
            raise ExpertStreamingConfigurationError(
                f"expert admission receipt is unusable: {exc}"
            ) from exc
        # Expert reads are on the decode path, so make the active I/O backend
        # visible: without the optional compiled extension the portable
        # ``preadv`` fallback is used, which works but is slower and otherwise
        # gives no sign that a faster path exists.
        if reader.backend == "native":
            _LOGGER.info("expert I/O backend: native positional reader")
        else:
            _LOGGER.info(
                "expert I/O backend: %s (portable fallback). For faster expert "
                "reads, install the native reader: "
                "uv pip install -e native_extensions/expert_io",
                reader.backend,
            )
        try:
            slots = ExpertSlotPool(
                model_spec,
                plan,
                manifest,
                reader,
                buffer_allocator=buffer_allocator,
                max_inflight_io_bytes=config.max_inflight_io_bytes,
                prefer_sidecar=config.prefer_sidecar,
                verify_hashes=(
                    config.verify_record_hashes
                    and not config.verify_sidecar_hash_at_open
                ),
                device_synchronize=device_synchronize,
                cache_scope=config.cache_scope,
                resource_telemetry=config.resource_telemetry,
                island_layers=(
                    config.island_layers + config.mmap_island_layers
                ),
                batch_decode_reads=config.overlap_miss_reads,
                **pipeline_kwargs,
            )
        except Exception:
            reader.close()
            raise
        runtime = cls(
            artifact_root,
            model_spec,
            config,
            manifest,
            plan,
            reader,
            slots,
            memory_cap_report=cap_report,
            integrity_report=integrity_report,
            single_slot_pool=single_slot_pool,
            **pipeline_kwargs,
        )
        if config.miss_shadow is not None:
            from mtplx.expert_shadow import ShadowBankStore

            try:
                streamed_layers = tuple(
                    layer
                    for layer in model_spec.routed_layer_indices
                    if layer not in runtime.island_layer_set
                )
                if config.miss_shadow_layers is not None:
                    # Worst layers first: pin-order-first streamed layers
                    # route flattest, so their exact-cache hit rate is
                    # lowest and shadows displace the most stall time.
                    streamed_set = set(streamed_layers)
                    ranked = [
                        layer
                        for layer in model_spec.island_pin_order
                        if layer in streamed_set
                    ]
                    ranked += [
                        layer for layer in streamed_layers if layer not in ranked
                    ]
                    streamed_layers = tuple(
                        sorted(ranked[: config.miss_shadow_layers])
                    )
                # The store constructor is the loud fence: it rejects a plan
                # whose shadow pricing does not match this codec/layer set.
                shadow_store = ShadowBankStore(
                    model_spec,
                    streamed_layers,
                    codec=config.miss_shadow,
                    plan=plan,
                )
                shadow_store.fill(
                    manifest,
                    artifact_root,
                    verify_hash=(
                        config.verify_record_hashes
                        and not config.verify_sidecar_hash_at_open
                    ),
                )
                runtime._shadow_store = shadow_store
            except BaseException:
                runtime.close()
                raise
        return runtime

    @staticmethod
    def _validate_manifest_identity(
        manifest: ExpertManifest,
        spec: ExpertStreamingModelSpec,
    ) -> None:
        try:
            validate_expert_manifest_spec(manifest, spec)
        except ExpertManifestError as exc:
            raise ExpertStreamingConfigurationError(
                "manifest does not match pinned model descriptor: " + str(exc)
            ) from exc

    def _record_cleanup_error(self, error: BaseException) -> None:
        with self._cleanup_error_lock:
            if self._cleanup_error is None:
                self._cleanup_error = error

    def _raise_cleanup_error(self) -> None:
        with self._cleanup_error_lock:
            error = self._cleanup_error
        if error is not None:
            raise ExpertSlotError("expert streaming runtime cleanup failed") from error

    def _raise_if_unhealthy(self) -> None:
        self.slots.raise_if_unhealthy()
        self._raise_cleanup_error()

    def _derived_expert_plan(self, live_kv_tokens: int) -> ExpertMemoryPlan:
        """One boundary plan of the single-limit policy at a live KV level."""

        # ``_per_layer_record_bytes`` is the manifest-derived per-layer record
        # map for mixed-official specs and ``None`` otherwise, matching what the
        # open-time plan passed; a mixed spec raises without it (issue #51 M2).
        return self.config.memory_plan(
            self.spec,
            additional_resident_bytes=self._additional_resident_bytes,
            live_kv_tokens=live_kv_tokens,
            layer_record_bytes=self._per_layer_record_bytes,
        )

    def _apply_derived_allowance(self) -> None:
        """Recompute the derived expert-cache allowance and evict down to it.

        Runs only at KV boundaries (admission, release, reset); cache hits
        never reach this path.  Those boundaries run on the generation thread,
        like reset()/close() -- the same thread that holds a deferred route's
        layer lock -- which is why the flush below (not a blocking re-acquire) is
        what breaks the self-deadlock.  Byte accounting is record-granular: every
        streamed slot holds exactly ``spec.expert_record_bytes``, so a byte
        allowance maps to an exact entry capacity.
        """

        # A deferred split/all-hit route (split_route_release="deferred",
        # deferred_pin_release, or the W42/W92 switch fast-path) keeps its layer
        # lock held until the next covering flush -- but the eviction loop below
        # takes EVERY layer lock, so on the generation thread (which also runs the
        # deferral) this KV boundary would self-deadlock waiting on a lock only a
        # later forward would release.  Drain the deferred releases first (they are
        # this thread's own, already async-submitted), exactly as reset()/close()
        # do at their boundaries.  No-op when nothing is deferred.
        self.flush_deferred_slot_releases(evaluate=True)
        with self._allowance_lock:
            with self._kv_lock:
                admitted = self._live_kv_tokens + self._pending_kv_tokens
            plan = self._derived_expert_plan(admitted)
            allowance = derived_expert_cache_allowance_bytes(plan)
            if allowance < 0:
                raise ExpertStreamingConfigurationError(
                    f"admitting {admitted} live KV tokens oversubscribes "
                    f"memory_limit_bytes={self.config.memory_limit_bytes} by "
                    f"{-allowance} bytes even with an empty expert cache; "
                    "the admission is refused before any allocation"
                )
            if self._global_bank is not None:
                # Global scope is forbidden for mixed-official specs, so a
                # single uniform record size is always valid on this branch.
                record_bytes = self.spec.expert_record_bytes
                capacity = allowance // record_bytes
                lock = self._layer_locks[self.spec.routed_layer_indices[0]]
                with lock:
                    self._evict_global_bank_to_capacity(capacity)
            else:
                # Uniform per-layer capacity over the streamed banks, exactly
                # mirroring the plan's uniform slots-per-layer derivation. For
                # mixed-official specs the streamed banks differ in record size
                # (issue #51 D2), so the denominator is the plan's
                # ``streamed_bytes_sum`` (per-layer record bytes summed over the
                # streamed banks) rather than a uniform ``layers * record`` term;
                # for uniform specs the two are identical.
                if self._per_layer_record_bytes is not None:
                    streamed_slot_bytes = sum(
                        self._per_layer_record_bytes[layer] for layer in self._banks
                    )
                else:
                    streamed_slot_bytes = (
                        len(self._banks) * self.spec.expert_record_bytes
                    )
                capacity = (
                    allowance // streamed_slot_bytes if streamed_slot_bytes else 0
                )
                for layer in sorted(self._banks):
                    with self._layer_locks[layer]:
                        self._evict_layer_bank_to_capacity(layer, capacity)
            self._derived_allowance_bytes = allowance
            self._derived_capacity_slots = capacity

    def _evict_layer_bank_to_capacity(self, layer: int, capacity: int) -> None:
        """Synchronously evict one layer bank's policy victims to a cap.

        A failure below leaves this and already-processed banks at the
        stricter cap with the admission unpublished, which under-uses but
        never oversubscribes the limit; the next boundary relaxes it.
        """

        bank = self._banks[layer]
        bank.set_persistent_capacity(capacity)
        skipped: set[int] = set()
        while bank.occupancy > capacity:
            # A KV-growth boundary lowers the cap: memory is the hard constraint,
            # so a W64 pinned expert may be evicted here as a last resort
            # (``respect_pins=False``). ``invalidate_expert`` unpins it, and the
            # device-route dirty marks below rebuild the layer's LUT views.
            victim = bank.peek_victim(excluded=skipped, respect_pins=False)
            if victim is None:
                raise ExpertStreamingConfigurationError(
                    f"cannot evict streamed experts on layer {layer} below "
                    f"the derived allowance: {bank.occupancy} resident "
                    f"entries exceed the {capacity}-entry cap and every "
                    "candidate is pinned or loading"
                )
            expert, slot = victim
            was_pinned = expert in bank.pinned_experts
            try:
                self.slots.invalidate(layer, slot, expert=expert)
            except ExpertSlotError:
                skipped.add(expert)
                continue
            bank.invalidate_expert(expert)
            self._mark_device_route_dirty(layer)  # W44: residency changed
            if was_pinned:
                # W71 (3): a pinned expert was force-evicted -> its pinned-only LUT
                # is now stale (slot recycled); invalidate so the next pinned route
                # touching it is recomputed on the fenced path.
                self._mark_device_route_pinned_dirty(layer)

    def _evict_global_bank_to_capacity(self, capacity: int) -> None:
        """Synchronously evict the global bank's policy victims to a cap."""

        bank = self._global_bank
        assert bank is not None
        bank.set_persistent_capacity(capacity)
        skipped: set[tuple[int, int]] = set()
        while bank.occupancy > capacity:
            victim = bank.peek_victim(excluded=skipped)
            if victim is None:
                raise ExpertStreamingConfigurationError(
                    "cannot evict streamed experts below the derived "
                    f"allowance: {bank.occupancy} resident entries exceed "
                    f"the {capacity}-entry cap and every candidate is "
                    "pinned or loading"
                )
            layer, expert, slot = victim
            try:
                self.slots.invalidate(layer, slot, expert=expert)
            except ExpertSlotError:
                skipped.add((layer, expert))
                continue
            bank.invalidate_expert(layer, expert)
    def _reject_island_layer(self, layer: int) -> None:
        # Island layers execute dense against their own full-resident bank;
        # reaching the streamed route machinery for one is a wiring bug, not
        # a runtime condition to fall back from.
        if layer in self.island_layer_set:
            raise ExpertStreamingConfigurationError(
                f"layer {layer} is a dense island layer; streamed routing is "
                "not available for it"
            )

    def admit_kv_tokens(self, tokens: int) -> KVAdmission:
        # Lifecycle order is close -> slot/runtime health -> KV accounting.
        # Release takes only the KV lock so an existing lease can always drain.
        with self._close_lock:
            if self._closed:
                raise ExpertSlotError("expert streaming runtime is closed")
            if self._closing:
                raise ExpertSlotError("expert streaming runtime is closing")
            self._raise_if_unhealthy()
            count = _integer("tokens", tokens, minimum=1)
            if not self._derived_cache_policy:
                with self._kv_lock:
                    requested = self._live_kv_tokens + count
                    if requested > self.config.max_live_kv_tokens:
                        raise ExpertStreamingConfigurationError(
                            f"live KV admission {requested} exceeds planned "
                            f"{self.config.max_live_kv_tokens} tokens"
                        )
                    self._live_kv_tokens = requested
                    self._live_kv_peak = max(self._live_kv_peak, requested)
                return KVAdmission(self, count)
            # Derived single-limit policy: shrink the expert-cache allowance
            # and synchronously evict down to it first, so KV growth is
            # admitted only once the cache is within budget.
            with self._kv_lock:
                requested = self._live_kv_tokens + count
                if requested > self.config.max_live_kv_tokens:
                    raise ExpertStreamingConfigurationError(
                        f"live KV admission {requested} exceeds planned "
                        f"{self.config.max_live_kv_tokens} tokens"
                    )
                self._pending_kv_tokens += count
            try:
                self._apply_derived_allowance()
            except BaseException:
                with self._kv_lock:
                    self._pending_kv_tokens -= count
                raise
            with self._kv_lock:
                self._pending_kv_tokens -= count
                self._live_kv_tokens += count
                self._live_kv_peak = max(self._live_kv_peak, self._live_kv_tokens)
            return KVAdmission(self, count)

    def release_kv_tokens(self, tokens: int) -> None:
        count = _integer("tokens", tokens, minimum=1)
        with self._kv_lock:
            if count > self._live_kv_tokens:
                raise RuntimeError("KV admission accounting underflow")
            self._live_kv_tokens -= count
        if self._derived_cache_policy and not self._closed and not self._closing:
            # KV shrink recomputes the larger allowance but allocates
            # nothing: later misses refill the cache naturally up to the
            # raised cap.
            self._apply_derived_allowance()

    def ensure_route(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
        cancel_event: threading.Event | None = None,
        deadline_ns: int | None = None,
    ) -> ReadyRoute:
        if self._closed:
            raise ExpertSlotError("expert streaming runtime is closed")
        if self._closing:
            raise ExpertSlotError("expert streaming runtime is closing")
        try:
            lock = self._layer_locks[layer]
        except KeyError as exc:
            raise ValueError(
                f"layer {layer} is not routed for {self.spec.key}"
            ) from exc
        with lock:
            self._raise_if_unhealthy()
            route_plan, policy_txn = self._plan_route_transaction(
                layer,
                expert_ids,
                phase=phase,
            )
            ready: ReadyRoute | None = None
            io_admission = RouteIOAdmission()
            try:
                ready = self.slots.ensure_route(
                    layer,
                    route_plan,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                    io_admission=io_admission,
                )
                policy_txn.commit()
            except BaseException as exc:
                if ready is not None:
                    ready.release(synchronize=False)
                self._handle_route_failure(
                    layer,
                    route_plan,
                    policy_txn,
                    exc,
                    io_admission=io_admission,
                )
                raise
            self._observe_plan(layer, route_plan)
            assert ready is not None
            return ready

    def defer_slot_release(self, ready, wave_output) -> None:
        """Queue a pinned route for release after the next generation eval.

        The wave output is an ancestor of the next layer's router indices, so
        the next ``mx.eval`` on the generation thread materializes it; pins
        release then without a per-layer blocking fence. The output reference
        is retained so the lazy graph cannot drop it before that eval.
        """

        pending = getattr(self, "_deferred_slot_releases", None)
        if pending is None:
            pending = []
            self._deferred_slot_releases = pending
        pending.append((ready, wave_output))

    def flush_deferred_slot_releases(self, *, evaluate: bool = False) -> None:
        """Release queued routes; the caller just completed a covering eval.

        ``evaluate=True`` fences the pending wave outputs first and is safe at
        any generation-thread boundary (row end, reset, close, KV admit/release)
        where no later eval is guaranteed to cover them.  All of these run on the
        generation thread -- the same thread that enqueues the deferrals -- so the
        pop/release below is not racing a concurrent append.
        """

        pending = getattr(self, "_deferred_slot_releases", None)
        if not pending:
            return
        if evaluate:
            # Local import: this module stays MLX-free at import time so the
            # I/O layer never touches MLX from worker contexts.
            import mlx.core as mx

            mx.eval(*[wave_output for _ready, wave_output in pending])
        first_error: BaseException | None = None
        while pending:
            ready, _wave_output = pending.pop(0)
            try:
                ready.release(synchronize=False)
            except BaseException as error:  # noqa: BLE001 - propagate after drain
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def try_all_hit_route(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
        cancel_event: threading.Event | None = None,
        deadline_ns: int | None = None,
    ) -> ReadyRoute | None:
        """Pin one fully resident route without wave or split execution.

        The policy probe is side-effect free when any assignment misses,
        allowing the caller to use the regular split route unchanged.
        """

        if self._closed:
            raise ExpertSlotError("expert streaming runtime is closed")
        if self._closing:
            raise ExpertSlotError("expert streaming runtime is closing")
        self._reject_island_layer(layer)
        try:
            lock = self._layer_locks[layer]
        except KeyError as exc:
            raise ValueError(
                f"layer {layer} is not routed for {self.spec.key}"
            ) from exc
        with lock:
            self._raise_if_unhealthy()
            planned = (
                self._global_bank.try_plan_all_hits_transaction(
                    layer,
                    expert_ids,
                    phase=phase,
                )
                if self._global_bank is not None
                else self._banks[layer].try_plan_all_hits_transaction(
                    expert_ids,
                    phase=phase,
                )
            )
            if planned is None:
                return None
            route_plan, policy_txn = planned
            ready: ReadyRoute | None = None

            def publish_route() -> None:
                self._publish_route_transaction(layer, route_plan, policy_txn)

            try:
                ready = self.slots.ensure_route(
                    layer,
                    route_plan,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                )
                self.slots.commit_if_healthy(publish_route)
            except BaseException:
                # A successful all-hit probe has no loads and therefore can
                # never cross the destructive I/O boundary.  Any pin-path
                # failure must restore its decode history and epoch exactly.
                if ready is not None:
                    try:
                        ready.release(synchronize=False)
                    except BaseException as cleanup_error:
                        self._record_cleanup_error(cleanup_error)
                        try:
                            ready.release(synchronize=False)
                        except BaseException as retry_error:
                            self._record_cleanup_error(retry_error)
                try:
                    policy_txn.rollback_publication()
                except BaseException as rollback_error:
                    self._record_cleanup_error(rollback_error)
                raise
            assert ready is not None
            _tl.all_hit(layer)  # W125: fully resident route, no miss I/O this layer
            return ready

    def _observe_plan(self, layer: int, plan: RoutePlan) -> None:
        with self._counter_lock:
            self._observe_plan_unlocked(layer, plan)

    def _record_bytes_for_layer(self, layer: int) -> int:
        """Streamed record bytes for a routed layer (per-layer for mixed)."""

        if self._per_layer_record_bytes is not None:
            return self._per_layer_record_bytes[int(layer)]
        return self.spec.expert_record_bytes

    def _observe_plan_unlocked(self, layer: int, plan: RoutePlan) -> None:
        # This plan is for one layer, so the layer's record size is the correct
        # byte weight for all three counter scopes (exact for mixed too).
        record_bytes = self._record_bytes_for_layer(layer)
        self.counters.observe(plan, expert_record_bytes=record_bytes)
        self._layer_counters[layer].observe(plan, expert_record_bytes=record_bytes)
        self._phase_counters[plan.phase].observe(
            plan, expert_record_bytes=record_bytes
        )
        self._observe_cold_start_unlocked(layer, plan)

    def _snapshot_prefetch_byte_window(self) -> None:
        """W95f: open a fresh speculative-byte budget window by marking the current
        cumulative demand/speculative byte totals as this window's origin. Called at
        every decode-token boundary (and at ``reset``); the budget in
        ``prefetch_experts`` throttles on the delta from here, so each token's
        speculation races that token's demand afresh and the budget cannot latch off
        for the life of the process. Caller holds ``_counter_lock``."""

        self._demand_bytes_at_token_start = self.demand_bytes_read
        self._speculative_bytes_at_token_start = self.speculative_bytes_read

    def _observe_cold_start_unlocked(self, layer: int, plan: RoutePlan) -> None:
        """W87: attribute each DECODE layer-route's assignment hits to the
        first-N-step (cold-start) or steady-state bucket, advancing the decode-step
        index at the token boundary (all streamed layers seen, or a layer repeats).
        Assignment-grained to match ``CacheCounters.hit_rate``.  Always on; both
        slot-pool paths populate the same fields.  A decode STEP == one full layer
        sweep == one token in --decode-mode ar (the mode the window A/B runs in);
        MED-4 re-opens the cold window on the first PREFILL route after decode so
        each request is measured fresh."""

        if plan.phase is not RoutingPhase.DECODE:
            if plan.phase is RoutingPhase.PREFILL and self._saw_decode_since_prefill:
                self._decode_token_index = 0
                self._decode_layers_seen = set()
                self._saw_decode_since_prefill = False
                self._snapshot_prefetch_byte_window()  # W95f: new request window
            return
        self._saw_decode_since_prefill = True
        hit_experts = set(plan.hits)
        assignment_hits = sum(expert in hit_experts for expert in plan.experts)
        requests = len(plan.experts)
        if self._decode_token_index < self._cold_start_decode_tokens:
            self._cold_decode_hits += assignment_hits
            self._cold_decode_requests += requests
        else:
            self._steady_decode_hits += assignment_hits
            self._steady_decode_requests += requests
        seen = self._decode_layers_seen
        if layer in seen:
            self._decode_token_index += 1
            self._decode_layers_seen = {layer}
            self._snapshot_prefetch_byte_window()  # W95f: new decode-token window
        else:
            seen.add(layer)
            if self._streamed_layer_set and seen >= self._streamed_layer_set:
                self._decode_token_index += 1
                self._decode_layers_seen = set()
                self._snapshot_prefetch_byte_window()  # W95f: new decode-token window

    def _cold_start_telemetry_locked(self) -> dict[str, Any]:
        """W87 cold-start decode telemetry (caller holds the counter lock)."""

        cold_h, cold_r = self._cold_decode_hits, self._cold_decode_requests
        warm_h, warm_r = self._steady_decode_hits, self._steady_decode_requests
        return {
            "cold_start_decode_steps": self._cold_start_decode_tokens,
            "decode_steps_observed": self._decode_token_index,
            "measurement_basis": (
                "first-64 DECODE STEPS, single-request (verify calls under DSpark, "
                "one token per step under --decode-mode ar; the cold window "
                "re-opens per request). Run the window A/B in --decode-mode ar."
            ),
            "single_slot_pool": self._single_slot_pool,
            "first_64_steps_hits": cold_h,
            "first_64_steps_requests": cold_r,
            "first_64_steps_hit_rate": (cold_h / cold_r) if cold_r else None,
            "steady_hits": warm_h,
            "steady_requests": warm_r,
            "steady_hit_rate": (warm_h / warm_r) if warm_r else None,
        }

    def _observe_incremental_unlocked(self, *, routes: int, parts: int) -> None:
        self._incremental_miss_routes += routes
        self._incremental_miss_parts += parts

    def _publish_route_transaction(
        self,
        layer: int,
        plan: RoutePlan,
        policy_txn: RoutePolicyTxn,
        *,
        incremental_parts: int = 0,
    ) -> None:
        counters = (
            self.counters,
            self._layer_counters[layer],
            self._phase_counters[plan.phase],
        )
        with self._counter_lock:
            counter_snapshots = tuple(counter.__dict__.copy() for counter in counters)
            incremental_snapshot = (
                self._incremental_miss_routes,
                self._incremental_miss_parts,
            )
            try:
                self._observe_plan_unlocked(layer, plan)
                if plan.phase is RoutingPhase.DECODE and plan.misses:
                    self._observe_incremental_unlocked(
                        routes=1,
                        parts=incremental_parts,
                    )
                # Publish policy last.  Counter observation is reversible and
                # cannot expose a partial snapshot while this lock is held.
                # Deferring the commit also means an all-hit global route does
                # not need to copy the entire LRU merely to undo a later
                # counter failure.
                policy_txn.commit()
                # W44: a route that loads/evicts/misses changed this layer's
                # persistent residency, so its device-route LUT is now stale.
                if plan.loads or plan.evictions or plan.misses:
                    self._mark_device_route_dirty(layer)
            except BaseException:
                for counter, snapshot in zip(counters, counter_snapshots, strict=True):
                    counter.__dict__.clear()
                    counter.__dict__.update(snapshot)
                (
                    self._incremental_miss_routes,
                    self._incremental_miss_parts,
                ) = incremental_snapshot
                try:
                    policy_txn.rollback_publication()
                except BaseException as rollback_error:
                    self._record_cleanup_error(rollback_error)
                raise

    def _plan_route(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
    ) -> RoutePlan:
        if self._global_bank is not None:
            return self._global_bank.plan(layer, expert_ids, phase=phase)
        return self._banks[layer].plan(expert_ids, phase=phase)

    def _plan_route_transaction(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
    ) -> tuple[RoutePlan, RoutePolicyTxn]:
        if self._global_bank is not None:
            return self._global_bank.plan_transaction(
                layer,
                expert_ids,
                phase=phase,
            )
        plan, policy_txn = self._banks[layer].plan_transaction(
            expert_ids, phase=phase
        )
        # Advisory record for the speculative lane (caller holds the layer
        # lock): the demand path is about to stream these records, so
        # predicting them again would duplicate the read. An all-hit plan
        # clears the record — its previous misses are resident by now.
        self._recent_route_misses[layer] = frozenset(
            load.expert for load in plan.loads
        )
        return plan, policy_txn

    def _invalidate_policy_expert(self, layer: int, expert: int) -> int | None:
        if self._global_bank is not None:
            return self._global_bank.invalidate_expert(layer, expert)
        bank = self._banks[layer]
        was_pinned = int(expert) in bank.pinned_experts
        slot = bank.invalidate_expert(expert)
        if was_pinned:
            # W71 (3): forgetting a pinned mapping unpins it -> its pinned-only LUT
            # entry is stale; invalidate so a later pinned route recomputes fenced.
            self._mark_device_route_pinned_dirty(layer)
        return slot

    @staticmethod
    def _subset_route_plan(
        plan: RoutePlan,
        *,
        hits: bool,
    ) -> RoutePlan | None:
        hit_set = set(plan.hits)
        selected_indices = [
            index
            for index, expert in enumerate(plan.experts)
            if (expert in hit_set) is hits
        ]
        selected = tuple(
            (plan.experts[index], plan.slots[index]) for index in selected_indices
        )
        if not selected:
            return None
        return RoutePlan(
            phase=plan.phase,
            experts=tuple(expert for expert, _slot in selected),
            slots=tuple(slot for _expert, slot in selected),
            hits=plan.hits if hits else (),
            misses=() if hits else plan.misses,
            loads=() if hits else plan.loads,
            evictions=() if hits else plan.evictions,
            generations=(
                tuple(plan.generations[index] for index in selected_indices)
                if plan.generations
                else ()
            ),
        )

    @staticmethod
    def _grouped_miss_route_parts(
        plan: RoutePlan,
        *,
        expert_groups: tuple[tuple[int, ...], ...],
    ) -> tuple[RoutePlan, ...]:
        loads_by_expert = {load.expert: load for load in plan.loads}
        parts: list[RoutePlan] = []
        for group in expert_groups:
            group_set = set(group)
            positions = tuple(
                index
                for index, candidate in enumerate(plan.experts)
                if candidate in group_set
            )
            parts.append(
                RoutePlan(
                    phase=plan.phase,
                    experts=tuple(plan.experts[index] for index in positions),
                    slots=tuple(plan.slots[index] for index in positions),
                    hits=(),
                    misses=tuple(
                        expert for expert in plan.misses if expert in group_set
                    ),
                    loads=tuple(loads_by_expert[expert] for expert in group),
                    evictions=tuple(
                        eviction
                        for eviction in plan.evictions
                        if eviction.next_expert in group_set
                    ),
                    generations=(
                        tuple(plan.generations[index] for index in positions)
                        if plan.generations
                        else ()
                    ),
                )
            )
        return tuple(parts)

    @classmethod
    def _miss_route_parts(cls, plan: RoutePlan) -> tuple[RoutePlan, ...]:
        """Split a miss plan by expert while preserving assignment duplicates."""

        unique_experts = tuple(dict.fromkeys(plan.experts))
        load_experts = tuple(load.expert for load in plan.loads)
        if len(set(load_experts)) != len(load_experts) or set(load_experts) != set(
            unique_experts
        ):
            raise ExpertSlotError(
                "incremental miss experts and slot loads must match one-to-one"
            )
        if len({load.slot for load in plan.loads}) != len(plan.loads):
            raise ExpertSlotError("incremental miss parts must own disjoint slots")
        return cls._grouped_miss_route_parts(
            plan,
            expert_groups=tuple((expert,) for expert in unique_experts),
        )

    def _bounded_decode_miss_route_parts(
        self,
        layer: int,
        plan: RoutePlan,
    ) -> tuple[RoutePlan, ...]:
        """Build bounded, sidecar-ordered parts for early miss completion."""

        records_per_part = self._decode_miss_records_per_part
        placement = self._decode_miss_placement
        ordered_experts = tuple(
            load.expert
            for load in sorted(
                plan.loads,
                key=lambda load: placement[(layer, load.expert)][:2],
            )
        )

        # Prefer a boundary at a physical gap.  This retains a contiguous
        # sidecar run in one scatter read unless the run itself exceeds the
        # construction-time bound.
        groups: list[tuple[int, ...]] = []
        start = 0
        while start < len(ordered_experts):
            end = min(start + records_per_part, len(ordered_experts))
            if end < len(ordered_experts):
                gap = None
                for index in range(start + 1, end + 1):
                    left = placement[(layer, ordered_experts[index - 1])]
                    right = placement[(layer, ordered_experts[index])]
                    adjacent = (
                        left[0] == right[0]
                        and left[1] + left[2] == right[1]
                    )
                    if not adjacent:
                        gap = index
                if gap is not None:
                    end = gap
            groups.append(ordered_experts[start:end])
            start = end
        return self._grouped_miss_route_parts(
            plan,
            expert_groups=tuple(groups),
        )

    def _rollback_route_loads(self, layer: int, plan: RoutePlan) -> None:
        for load in plan.loads:
            if load.persistent:
                self._invalidate_policy_expert(layer, load.expert)
            try:
                self.slots.invalidate(
                    layer,
                    load.slot,
                    expert=load.expert,
                    generation=load.generation,
                )
            except ExpertSlotError:
                pass

    def _handle_route_failure(
        self,
        layer: int,
        plan: RoutePlan,
        policy_txn: RoutePolicyTxn,
        error: BaseException,
        *,
        io_admission: RouteIOAdmission | None = None,
    ) -> None:
        rollback_safe = (
            not io_admission.any_accepted
            if io_admission is not None
            else (
                isinstance(error, ExpertCompletionFenceError)
                and error.policy_rollback_safe
            )
        )
        if rollback_safe:
            policy_txn.rollback_completion()
            return
        self._rollback_route_loads(layer, plan)

    def _handle_split_route_failure(
        self,
        layer: int,
        plan: RoutePlan,
        policy_txn: RoutePolicyTxn,
        error: BaseException,
        *,
        accepted_parts: tuple[RoutePlan, ...],
        io_admission: RouteIOAdmission | None,
    ) -> None:
        """Restore untouched victims while quarantining accepted split loads."""

        if io_admission is None or not io_admission.any_accepted:
            policy_txn.rollback_completion()
            return
        if not accepted_parts:
            self._handle_route_failure(
                layer,
                plan,
                policy_txn,
                error,
                io_admission=io_admission,
            )
            return

        # Every submitted future has settled before this runs. Remove only the
        # physical records that crossed their part-local admission boundary,
        # then restore the full policy snapshot. Accepted evictions cannot be
        # restored physically, so quarantine those victims again afterward.
        for part in accepted_parts:
            self._rollback_route_loads(layer, part)
        policy_txn.rollback_completion()
        for part in accepted_parts:
            for eviction in part.evictions:
                previous_layer = (
                    layer
                    if eviction.previous_layer is None
                    else eviction.previous_layer
                )
                self._invalidate_policy_expert(
                    previous_layer,
                    eviction.previous_expert,
                )
        if self._global_bank is not None:
            for part in accepted_parts:
                for load in part.loads:
                    if load.persistent and load.generation is not None:
                        self._global_bank.reconcile_slot_generation(
                            load.slot,
                            load.generation,
                        )

    def begin_split_route(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
        cancel_event: threading.Event | None = None,
        deadline_ns: int | None = None,
    ) -> PendingSplitRoute:
        """Pin hits now and load misses while the caller evaluates hit work."""

        if self._closed:
            raise ExpertSlotError("expert streaming runtime is closed")
        if self._closing:
            raise ExpertSlotError("expert streaming runtime is closing")
        self._reject_island_layer(layer)
        try:
            lock = self._layer_locks[layer]
        except KeyError as exc:
            raise ValueError(
                f"layer {layer} is not routed for {self.spec.key}"
            ) from exc
        lock.acquire()
        # W93: materialize once so the reconcile scan below and the planner see the
        # same ids (expert_ids may be a one-shot iterable).
        expert_ids = tuple(expert_ids)
        plan: RoutePlan | None = None
        policy_txn: RoutePolicyTxn | None = None
        hit_ready: ReadyRoute | None = None
        pending: PendingSplitRoute | None = None
        pipeline_route: ExpertPipelineRoute | None = None
        io_admission = RouteIOAdmission()
        miss_cancel_event = threading.Event()
        combined_cancel = _RouteCancel(cancel_event, miss_cancel_event)
        try:
            self._raise_if_unhealthy()
            # W93 gate-oracle prefetch reconcile (inert unless the ring is armed):
            # publish settled ring reads and await a needed in-flight one before
            # planning, so a gate-predicted expert resolves as a hit rather than a
            # duplicate demand read. Under the layer lock already held here.
            _tl_rc = _tl.now()  # W125: time the reconcile await (host gap component)
            self._reconcile_prefetch_for_route(layer, expert_ids)
            _tl.add_reconcile(layer, _tl_rc)
            _tl.reconcile_done(layer)
            plan, policy_txn = self._plan_route_transaction(
                layer,
                expert_ids,
                phase=phase,
            )
            hit_plan = self._subset_route_plan(plan, hits=True)
            miss_plan = self._subset_route_plan(plan, hits=False)
            # W95f: account this layer-route's DEMAND miss bytes -- the reads that
            # touch the SSD on the blocking path -- WHERE the demand plan is made,
            # so the speculative-byte budget has a real demand denominator. At HEAD
            # the only writer was the reconcile fallback below, so cold demand
            # misses never advanced it: the budget was inert until the first
            # fallback and then latched off. ``_reconcile_prefetch_for_route`` above
            # already promoted settled ring reads to hits and invalidated fell-back
            # predictions, so ``miss_plan.loads`` is exactly the {cold miss,
            # fell-back} demand set -- counting it here SUPERSEDES the fallback
            # increment (which is removed, so no double count). Gated on the ring
            # being armed so the shipped (ring-off) path stays byte-identical.
            if (
                self.config.prefetch_slots > 0
                and miss_plan is not None
                and miss_plan.loads
            ):
                _demand_record_bytes = self._record_bytes_for_layer(layer)
                with self._counter_lock:
                    self.demand_bytes_read += (
                        len(miss_plan.loads) * _demand_record_bytes
                    )
            # Fix (B) overlap submits the layer's decode misses together.  The
            # unchanged mode uses one part; the construction-selected bounded
            # mode exposes several completion groups while still submitting all
            # groups before the caller starts resident work.
            batch_misses = (
                self.config.overlap_miss_reads
                and miss_plan is not None
                and plan.phase is RoutingPhase.DECODE
            )
            if miss_plan is None:
                miss_parts = ()
            elif (
                batch_misses
                and self.config.decode_miss_records_per_part is not None
            ):
                miss_parts = self._bounded_decode_miss_route_parts(
                    layer,
                    miss_plan,
                )
            elif batch_misses:
                miss_parts = (miss_plan,)
            elif plan.phase is RoutingPhase.DECODE:
                miss_parts = self._miss_route_parts(miss_plan)
            else:
                miss_parts = (miss_plan,)
            pipeline_ledger = self._pipeline_ledger
            if pipeline_ledger is not None:
                try:
                    load_experts = tuple(
                        dict.fromkeys(
                            load.expert
                            for load in (() if miss_plan is None else miss_plan.loads)
                        )
                    )
                    pipeline_route = pipeline_ledger.begin_route(
                        layer=layer,
                        phase=plan.phase,
                        load_experts=load_experts,
                        load_logical_bytes=tuple(
                            self.manifest.record(layer, expert).logical_bytes
                            for expert in load_experts
                        ),
                    )
                except Exception:
                    pipeline_route = None
                    try:
                        pipeline_ledger.mark_incomplete(phase=plan.phase)
                    except Exception:
                        pass
            hit_ready = (
                self.slots.ensure_route(
                    layer,
                    hit_plan,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                )
                if hit_plan is not None
                else None
            )
            pending = PendingSplitRoute(
                runtime=self,
                layer=layer,
                plan=plan,
                layer_lock=lock,
                hit_ready=hit_ready,
                miss_futures={},
                policy_txn=policy_txn,
                io_admission=io_admission,
                miss_cancel_event=miss_cancel_event,
                miss_parts=miss_parts,
                pipeline_route=pipeline_route,
            )
            if miss_plan is not None:
                ensure = (
                    self.slots.ensure_route_part
                    if plan.phase is RoutingPhase.DECODE
                    else self.slots.ensure_route
                )
                for ordinal, miss_part in enumerate(miss_parts):
                    part_admission = io_admission.child()
                    if pipeline_route is None:
                        future = self._split_executor.submit(
                            ensure,
                            layer,
                            miss_part,
                            cancel_event=combined_cancel,
                            deadline_ns=deadline_ns,
                            io_admission=part_admission,
                            route_admitted=pending._retain_lifecycle_after_admission,
                        )
                    else:
                        future = self._split_executor.submit(
                            ensure,
                            layer,
                            miss_part,
                            cancel_event=combined_cancel,
                            deadline_ns=deadline_ns,
                            io_admission=part_admission,
                            route_admitted=pending._retain_lifecycle_after_admission,
                            pipeline_route=pipeline_route,
                        )
                    pending._attach_miss_future(
                        future,
                        miss_part,
                        ordinal=ordinal,
                        io_admission=part_admission,
                    )
                if batch_misses:
                    self.slots.metrics.update(
                        batched_miss_parts=len(miss_parts),
                        batched_miss_records=sum(
                            len(part.loads) for part in miss_parts
                        ),
                    )
            else:
                pending._commit_policy()
            # W125: the demand miss set is planned and its reads submitted to the
            # split executor (the miss futures above); this is the "miss set
            # computed + reads issued" host event for this layer/token.
            _tl.miss_issue(layer)
            return pending
        except BaseException as setup_error:
            # Mirror the sync-path rollback: without it, a failed hit pin or
            # submit leaves the bank mapping experts to never-loaded slots,
            # wedging every later route on this layer until reset().
            miss_cancel_event.set()
            if pending is not None:
                pending.abort(setup_error)
                pending.close()
            else:
                if pipeline_route is not None:
                    _pipeline_call(
                        self._pipeline_ledger,
                        pipeline_route,
                        "close",
                    )
                if hit_ready is not None:
                    try:
                        hit_ready.release(synchronize=False)
                    except BaseException:
                        pass
                if policy_txn is not None:
                    try:
                        self._handle_route_failure(
                            layer,
                            plan,
                            policy_txn,
                            setup_error,
                            io_admission=io_admission,
                        )
                    except BaseException:
                        pass
                lock.release()
            raise

    def route_waves(
        self,
        expert_ids: Iterable[int],
        *,
        sort_unique: bool = False,
    ) -> tuple[RouteWave, ...]:
        return partition_route_waves(
            expert_ids,
            max_unique_experts=self._batch_admission_slots(),
            sort_unique=sort_unique,
        )

    def _batch_admission_slots(self) -> int:
        """W87: max unique experts admitted in ONE transaction (one fence) =
        ``plan.batch_admission_slots`` (== transient_slots on both paths; the merged-
        capacity widening was retired, review HIGH-1).  Consumed by ``route_waves``
        and the expert_mlx verify single-fence gate."""

        # W87: the single-fence wave width is plan.batch_admission_slots
        # (== transient_slots; the merged-capacity widening was retired, review
        # HIGH-1) -- transient_slots on both paths (byte-identical).  The single-pool
        # win is the admission policy (prefill warms the pool, decode 2Q), not width.
        return int(self.plan.batch_admission_slots or self.plan.transient_slots)

    def observe_route(
        self,
        layer: int,
        phase: RoutingPhase | str,
        expert_ids: Iterable[int],
        *,
        token_count: int,
    ) -> None:
        # W125: this call fires immediately after the switch's mx.eval(indices)
        # routing barrier (the device->host sync / device-LUT resolve), so it is
        # the "routing indices barrier done" host event for this layer.
        _tl.barrier_done(layer)
        # W125 (red-team HIGH-2/MEDIUM): stamp which switch path this run uses, once.
        # deferred_pin_release / split_route_release are NOT in _PROFILE_PLAN_FIELDS,
        # so on the shipped default (fenced) with SWITCH_FASTPATH off the split path
        # blocks on mx.eval of the gather -- meaning ready_to_dispatch and the
        # miss_issue_to_ready PHASE include GPU gather time; only miss_wait_total
        # (ACC) is the pure host SSD wait. A reader needs this flag to interpret them.
        if _tl.enabled() and not _tl.config_noted():
            _cfg = self.config
            _dpr = bool(getattr(_cfg, "deferred_pin_release", False))
            _srr = str(getattr(_cfg, "split_route_release", "fenced"))
            _fastpath = os.environ.get("MTPLX_DSV41_SWITCH_FASTPATH") == "1"
            # HIGH-2b: mirror the switch's _deferred_pin_active exactly. The route
            # fences unless deferred_pin_release OR fastpath_can_defer is set;
            # split_route_release alone (even "deferred") does NOT stop the fence.
            # fastpath_can_defer = SWITCH_FASTPATH armed AND this runtime has the
            # defer/flush seam (the real ExpertStreamingRuntime does).
            _can_defer = callable(getattr(self, "defer_slot_release", None)) and callable(
                getattr(self, "flush_deferred_slot_releases", None)
            )
            _fastpath_can_defer = _fastpath and _can_defer
            _tl.note_switch_config(
                deferred_pin_release=_dpr,
                split_route_release=_srr,
                switch_fastpath=_fastpath,
                fastpath_can_defer=_fastpath_can_defer,
                overlap_miss_reads=bool(getattr(_cfg, "overlap_miss_reads", False)),
                device_route=(
                    os.environ.get("MTPLX_DSV41_DEVICE_ROUTE") == "1"
                    or os.environ.get("MTPLX_DSV41_DEVICE_ROUTE_PINNED") == "1"
                ),
                fenced_split_path=not (_dpr or _fastpath_can_defer),
            )
        census = self._route_census
        if (
            census is None
            and not self.config.trace_routes
            and self._belady_oracle is None
        ):
            return
        normalized_phase = RoutingPhase(phase)
        routed_experts = [int(expert) for expert in expert_ids]
        if (
            self._belady_oracle is not None
            and normalized_phase is RoutingPhase.DECODE
        ):
            # Diagnostic only; the oracle self-excludes island layers. Failure
            # disables it for the session rather than perturbing decode.
            try:
                with self._census_lock:
                    self._belady_oracle.observe(layer, routed_experts)
            except Exception:
                self._belady_oracle = None
        if census is not None and normalized_phase is RoutingPhase.DECODE:
            # Placement census: decode routes only (prefill routing shape
            # does not predict the decode hot set). Accumulation is pure
            # host-side counting; any failure disables the census for the
            # session rather than perturbing the decode path.
            try:
                with self._census_lock:
                    census.observe(layer, routed_experts)
            except Exception:
                self._route_census = None
                _LOGGER.warning(
                    "route census recording failed; census disabled for "
                    "this session",
                    exc_info=True,
                )
        if not self.config.trace_routes:
            return
        with self._route_trace_lock:
            entry = {
                "layer": int(layer),
                "phase": normalized_phase.value,
                "trace_epoch": self._route_trace_epoch,
                "token_count": int(token_count),
                "expert_ids": routed_experts,
            }
            if normalized_phase is RoutingPhase.DECODE:
                routed_layers = set(self.spec.routed_layer_indices)
                if layer in self._route_trace_decode_layers_seen:
                    # Preserve evidence rather than silently assigning a
                    # duplicate layer to the current step. The analyzer will
                    # reject the resulting incomplete step sequences.
                    self._route_trace_decode_step += 1
                    self._route_trace_decode_layers_seen.clear()
                entry["decode_step"] = self._route_trace_decode_step
                self._route_trace_decode_layers_seen.add(int(layer))
                if self._route_trace_decode_layers_seen == routed_layers:
                    self._route_trace_decode_step += 1
                    self._route_trace_decode_layers_seen.clear()
            self._route_trace.append(entry)

    def route_trace(self) -> list[dict[str, Any]]:
        with self._route_trace_lock:
            return [dict(entry) for entry in self._route_trace]

    def prepare_prefill_seed(
        self,
        layer: int,
        expert_ids: Iterable[int],
    ) -> tuple[int, ...]:
        self._reject_island_layer(layer)
        try:
            lock = self._layer_locks[layer]
        except KeyError as exc:
            raise ValueError(
                f"layer {layer} is not routed for {self.spec.key}"
            ) from exc
        with lock:
            self._raise_if_unhealthy()
            if self._global_bank is not None:
                return self._global_bank.prepare_prefill_seed(layer, expert_ids)
            return self._banks[layer].prepare_prefill_seed(expert_ids)

    def shadow_bank_for_layer(self, layer: int) -> Any | None:
        """The layer's shadow miss-fallback bank, or None when disabled."""

        store = self._shadow_store
        if store is None:
            return None
        try:
            return store.bank_for_layer(layer)
        except Exception:
            return None

    def peek_resident_experts(
        self, layer: int, expert_ids: Iterable[int]
    ) -> frozenset[int]:
        """Which of ``expert_ids`` are hit-eligible right now (no pinning).

        A snapshot, not a reservation: residency can change between this
        peek and a subsequent route, in which case the route machinery
        simply services the expert exactly (slower, never wrong).
        """

        bank = self._banks.get(layer)
        if bank is None:
            return frozenset()
        lock = self._layer_locks[layer]
        with lock:
            return bank.published_experts(expert_ids)

    # ------------------------------------------------------------------
    # W44 device-route (barrier-free all-hit; env MTPLX_DSV41_DEVICE_ROUTE).
    # See docs/deepseek-v41/W44_DEVICE_ROUTE.md.
    # ------------------------------------------------------------------
    def _mark_device_route_dirty(self, layer: int) -> None:
        """Residency for ``layer`` changed -> its device LUT must be rebuilt on
        the next device-route gather. No-op until the device path is used."""

        if self._device_route_lut:
            self._device_route_lut_dirty[int(layer)] = True

    def _mark_device_route_pinned_dirty(self, layer: int) -> None:
        """The PINNED set for ``layer`` changed (a pin, a refresh re-rank, or a
        force-eviction unpin) -> its pinned-only LUT must be rebuilt on the next
        pinned device-route gather (W71 (1)/(3)). Kept SEPARATE from the residency
        dirty mark: a pinned expert's slot never moves while it stays pinned (W64),
        so a mere free-tail LRU churn -- an unpinned residency change -- leaves the
        pinned LUT exact and must NOT trigger a rebuild. No-op until the pinned
        device path has been used."""

        if self._device_route_pinned_lut:
            self._device_route_pinned_lut_dirty[int(layer)] = True

    def register_component_bank(self, layer: int, bank: Any) -> None:
        """Capture ``layer``'s component bank from a fenced binding (idempotent),
        so the device-route path can gather without a routed binding of its own."""

        if bank is not None and int(layer) not in self._device_route_bank:
            self._device_route_bank[int(layer)] = bank

    def component_bank_for_layer(self, layer: int) -> Any | None:
        return self._device_route_bank.get(int(layer))

    def device_route_lut(self, layer: int, *, mx_module: Any | None = None) -> Any:
        """Device expert->slot LUT for ``layer`` (int32 ``[expert_count]``; -1 =
        non-resident).  Rebuilt on the host only when the layer's persistent
        residency has changed since the last build; otherwise the cached mx.array
        is returned unchanged (no host work).  Only per-layer component banks
        populate the table; under other layouts every expert reads as -1 (a miss),
        so the device path never fabricates an all-hit it cannot back with the
        exact resident slot."""

        layer = int(layer)
        cached = self._device_route_lut.get(layer)
        if cached is not None and not self._device_route_lut_dirty.get(layer, False):
            return cached
        mx = mx_module
        if mx is None:  # local import: keep this module MLX-free at import time
            import mlx.core as mx
        n = int(self.spec.expert_count)
        table = [-1] * n
        snapshot: set[int] = set()
        bank = self._banks.get(layer) if self._banks else None
        lock = self._layer_locks.get(layer)

        def _fill(source_bank: Any) -> None:
            slot_map = getattr(source_bank, "_expert_to_slot", None)
            if not slot_map:
                return
            for expert, slot in slot_map.items():
                e = int(expert)
                if 0 <= e < n:
                    table[e] = int(slot)
                    snapshot.add(e)

        if bank is not None:
            if lock is not None:
                # Non-blocking: this rebuild is reached from _run_device_route
                # BEFORE the per-layer covering flush, so a pending deferred route
                # (split_route_release="deferred" / deferred_pin_release / the W42
                # switch fast-path / a W61/W81 verify defer) may hold this layer's
                # lock. Blocking here would self-deadlock the generation thread; on
                # contention return None so the caller uses the fenced path (which
                # flushes) and the LUT stays dirty to rebuild next token. (W92 review.)
                if not lock.acquire(blocking=False):
                    return None
                try:
                    _fill(bank)
                finally:
                    lock.release()
            else:
                _fill(bank)
        arr = mx.array(table, dtype=mx.int32)
        mx.eval(arr)
        self._device_route_lut[layer] = arr
        self._device_route_lut_snapshot[layer] = frozenset(snapshot)
        self._device_route_lut_dirty[layer] = False
        return arr

    def device_route_snapshot(self, layer: int) -> frozenset[int]:
        """The resident-expert set the current LUT for ``layer`` was built from
        (what an all-hit device gather for ``layer`` is exact against)."""

        return self._device_route_lut_snapshot.get(int(layer), frozenset())

    def device_route_pinned_lut(self, layer: int, *, mx_module: Any | None = None) -> Any:
        """W71: device expert->slot LUT for ``layer`` built from PINNED experts
        ONLY (int32 ``[expert_count]``; -1 for every non-pinned expert, resident or
        not). Rebuilt on the host only when the layer's PIN set changes -- a pin /
        refresh re-rank or a memory-forced capacity eviction that unpins an expert
        (``_mark_device_route_pinned_dirty``); otherwise the cached mx.array is
        returned unchanged. A pinned expert's slot is never recycled by normal
        decode admission (W64), so an all-pinned route's deferred gather over this
        table cannot race a recycle -- the safety property W44 §8 lacked. A route
        touching any non-pinned expert reads -1 -> void row (clamped) and is caught
        by the deferred flush, which recomputes that layer on the fenced path."""

        layer = int(layer)
        cached = self._device_route_pinned_lut.get(layer)
        if cached is not None and not self._device_route_pinned_lut_dirty.get(
            layer, False
        ):
            return cached
        mx = mx_module
        if mx is None:  # local import: keep this module MLX-free at import time
            import mlx.core as mx
        n = int(self.spec.expert_count)
        table = [-1] * n
        snapshot: set[int] = set()
        bank = self._banks.get(layer) if self._banks else None
        lock = self._layer_locks.get(layer)

        def _fill(source_bank: Any) -> None:
            slot_map = getattr(source_bank, "_expert_to_slot", None)
            if not slot_map:
                return
            pinned = getattr(source_bank, "pinned_experts", frozenset())
            for expert in pinned:
                e = int(expert)
                slot = slot_map.get(e)
                if slot is not None and 0 <= e < n:
                    table[e] = int(slot)
                    snapshot.add(e)

        if bank is not None:
            if lock is not None:
                # Non-blocking (see device_route_lut): a pending deferred route may
                # hold this layer's lock before the covering flush; on contention
                # return None so _run_device_route falls to the fenced path. (W92.)
                if not lock.acquire(blocking=False):
                    return None
                try:
                    _fill(bank)
                finally:
                    lock.release()
            else:
                _fill(bank)
        arr = mx.array(table, dtype=mx.int32)
        mx.eval(arr)
        self._device_route_pinned_lut[layer] = arr
        self._device_route_pinned_snapshot[layer] = frozenset(snapshot)
        self._device_route_pinned_lut_dirty[layer] = False
        return arr

    def device_route_pinned_snapshot(self, layer: int) -> frozenset[int]:
        """The PINNED-expert set the current pinned LUT for ``layer`` was built
        from (what an all-pinned device gather for ``layer`` is exact against)."""

        return self._device_route_pinned_snapshot.get(int(layer), frozenset())

    def set_device_route_force_fenced(self, layers: Iterable[int]) -> None:
        """Force ``layers`` onto the fenced path (barrier + admit + gather) for a
        W44 cold-recovery re-run; pass ``()`` to clear. The switch reads this and
        skips its device path for those layers, so a flagged miss is repaired
        byte-identically while every other layer keeps the barrier-free route."""

        self._device_route_force_fenced = frozenset(int(x) for x in layers)

    def enqueue_device_route_probe(
        self,
        layer: int,
        indices: Any,
        snapshot: frozenset[int],
        *,
        pinned: bool = False,
    ) -> None:
        """Queue a routed ``indices`` array (already ``async_eval``'d by the
        switch) for deferred verification. ``pinned`` (W71) selects the check the
        flush applies: a pinned probe is kept only if every routed expert is
        CURRENTLY pinned (so a mid-token force-eviction of a pinned expert flips it
        to a recompute), a resident (W44) probe only if every expert is in the
        build ``snapshot``."""

        self._device_route_probes.append((int(layer), indices, snapshot, bool(pinned)))

    def flush_device_route_probes(self) -> list[tuple[int, tuple[int, ...]]]:
        """Read back the queued route indices and return, per probed layer, the
        experts that were NOT in that layer's LUT snapshot -- the misses whose
        optimistic gather is void and must be recovered on the fenced path.  An
        empty result means every probed layer was all-hit (its device gather is
        byte-identical to the fenced path)."""

        probes = self._device_route_probes
        if not probes:
            return []
        self._device_route_probes = []
        # ONE batched host sync for the whole span's verification: the ids were
        # async_eval'd per layer during the barrier-free pass (so they are usually
        # already resident by now), and evaluating them together here forces a
        # single device->host round-trip -- not one per layer, which would
        # reintroduce the ~40 syncs the device route exists to remove. (The
        # warm/all-hit token therefore costs this ONE verify sync, not 40.)
        import mlx.core as mx  # local: keep this module MLX-free at import

        mx.eval(*[indices for _layer, indices, _snapshot, _pinned in probes])
        misses: list[tuple[int, tuple[int, ...]]] = []
        pinned_probes = 0
        pinned_recovered = 0
        for layer, indices, snapshot, pinned in probes:
            ids = [int(v) for v in indices.reshape(-1).tolist()]
            if pinned:
                # W71: verify against the CURRENT pinned set, not the build
                # snapshot -- so an expert that was pinned when the LUT was built
                # but has since been force-evicted (unpinned) is flagged for a
                # fenced recompute, closing the W44 §8 recycle race for the one
                # case W64 allows a pinned slot to move (memory hard constraint).
                bank = self._banks.get(layer) if self._banks else None
                lock = self._layer_locks.get(layer)
                if bank is not None and lock is not None:
                    with lock:
                        pinned_now = getattr(bank, "pinned_experts", frozenset())
                else:
                    pinned_now = getattr(bank, "pinned_experts", frozenset())
                missed = tuple(sorted({e for e in ids if e not in pinned_now}))
                pinned_probes += 1
                if missed:
                    pinned_recovered += 1
            else:
                missed = tuple(sorted({e for e in ids if e not in snapshot}))
            if missed:
                misses.append((layer, missed))
        if pinned_probes:
            self._device_route_pinned_flushes += 1
            self._device_route_pinned_barrier_free_layers += (
                pinned_probes - pinned_recovered
            )
            self._device_route_pinned_recovered_layers += pinned_recovered
        return misses

    # ------------------------------------------------------------------
    # W64 (R3-pin): post-prefill pinned working set + all-pinned telemetry.
    # See docs/deepseek-v41/W64_PINNED_WORKING_SET.md.
    # ------------------------------------------------------------------
    def pin_working_set_hook(
        self,
        layer: int,
        expert_ids: Iterable[int],
        phase: RoutingPhase | str,
    ) -> None:
        """Switch-side per-route hook (env ``MTPLX_DSV41_PIN_WORKING_SET``).

        Default off -> immediate no-op (one env read), so decode is byte-identical
        to the pre-W64 path. When armed and ``phase`` is DECODE it (1) pins the
        layer's working set on the first decode route after prefill -- and every
        ``MTPLX_DSV41_PIN_REFRESH_TOKENS`` decode epochs when set -- and (2)
        records whether every routed expert was pinned (the all-pinned-hit rate a
        later device route can turn barrier-free). Never raises: a pin failure
        disables W64 for the session rather than perturbing decode."""

        spec = parse_pin_working_set(os.environ.get(PIN_WORKING_SET_ENV))
        if spec is None:
            return
        if RoutingPhase(phase) is not RoutingPhase.DECODE:
            return
        if self._global_bank is not None:
            return  # per-layer banks only; global scope is out of W64 scope
        try:
            self._maybe_pin_layer(int(layer), spec)
            self._observe_pin_route(int(layer), expert_ids)
        except Exception:  # pragma: no cover - defensive; never perturb decode
            _LOGGER.warning(
                "W64 pin_working_set hook failed; pinning left as-is for the "
                "remaining decode",
                exc_info=True,
            )

    def _maybe_pin_layer(
        self, layer: int, spec: tuple[str, float | None]
    ) -> None:
        bank = self._banks.get(layer)
        if bank is None:
            return
        lock = self._layer_locks.get(layer)
        refresh = parse_pin_refresh_tokens(os.environ.get(PIN_REFRESH_TOKENS_ENV))
        with lock if lock is not None else _NULL_CTX:
            epoch = bank._decode_epoch
            last = self._pin_last_epoch.get(layer)
            if last is not None and (refresh <= 0 or epoch - last < refresh):
                return
            mode, value = spec
            if mode == "all":
                bank.pin_working_set(experts=bank.resident_experts)
            else:
                slots = bank.persistent_slots
                if mode == "frac":
                    assert value is not None
                    k = max(1, int(round(value * slots)))
                else:
                    k = int(value or 0)
                free_tail = max(0, slots - k)
                bank.pin_working_set(top_k=k, free_tail=free_tail)
            self._pin_last_epoch[layer] = epoch
        # Residency contract changed (pinned set is now this layer's static set);
        # a device-route LUT built before the pin must be rebuilt.
        self._mark_device_route_dirty(layer)
        # W71: the pin set changed -> the pinned-only LUT must rebuild too.
        self._mark_device_route_pinned_dirty(layer)

    def _observe_pin_route(self, layer: int, expert_ids: Iterable[int]) -> None:
        bank = self._banks.get(layer)
        if bank is None:
            return
        all_pinned = bank.route_all_pinned(expert_ids)
        with self._pin_telemetry_lock:
            self._pin_decode_routes += 1
            if all_pinned:
                self._pin_all_pinned_routes += 1

    def pin_working_set(self, layer: int | None = None) -> dict[int, tuple[int, ...]]:
        """Force-pin the working set now (out-of-band from the switch hook).

        Ranks each per-layer bank's resident experts by prefill frequency and
        pins the top set per the env spec. Returns ``{layer: pinned ids}``. A
        no-op (``{}``) when the lever is off or the runtime uses a global bank.
        Exposed for a backbone that prefers to pin explicitly at the prefill ->
        decode boundary rather than lazily on the first decode route."""

        spec = parse_pin_working_set(os.environ.get(PIN_WORKING_SET_ENV))
        if spec is None or self._global_bank is not None:
            return {}
        layers = (
            [int(layer)] if layer is not None else sorted(self._banks)
        )
        pinned: dict[int, tuple[int, ...]] = {}
        for lyr in layers:
            bank = self._banks.get(lyr)
            if bank is None:
                continue
            self._maybe_pin_layer(lyr, spec)
            pinned[lyr] = tuple(sorted(bank.pinned_experts))
        return pinned

    def clear_working_set_pins(self) -> None:
        """Drop every layer's pinned working set (and the pin-epoch gates)."""

        if self._global_bank is not None:
            return
        for lyr, bank in self._banks.items():
            lock = self._layer_locks.get(lyr)
            with lock if lock is not None else _NULL_CTX:
                bank.clear_pins()
            self._mark_device_route_dirty(lyr)
            self._mark_device_route_pinned_dirty(lyr)  # W71: pin set changed
        self._pin_last_epoch.clear()

    def layer_pinned_static(self, layer: int) -> bool:
        """True iff ``layer`` has an active pinned working set (its pinned
        experts are never recycled on normal admission). A device route may take
        the barrier-free path for a route only when this holds AND every routed
        expert is pinned -- see :meth:`route_all_pinned`."""

        bank = self._banks.get(int(layer))
        return bool(bank is not None and bank.pinned_static)

    def route_all_pinned(self, layer: int, expert_ids: Iterable[int]) -> bool:
        """True iff every routed expert on ``layer`` is pinned (slot-stable).
        The per-token gate a barrier-free device route consults (W44 §8)."""

        bank = self._banks.get(int(layer))
        return bool(bank is not None and bank.route_all_pinned(expert_ids))

    def pinned_working_set_telemetry(self) -> dict[str, Any]:
        """Snapshot of the W64 pin state (pinned count per layer + the
        all-pinned-hit rate per decode route). Surfaced in the runtime snapshot,
        the A/B receipt, and the served event; empty-ish when the lever is off."""

        enabled = parse_pin_working_set(os.environ.get(PIN_WORKING_SET_ENV)) is not None
        pinned_by_layer: dict[str, int] = {}
        static_layers: list[int] = []
        if self._global_bank is None:
            for lyr in sorted(self._banks):
                bank = self._banks[lyr]
                count = bank.pinned_count
                if count:
                    pinned_by_layer[str(lyr)] = count
                    static_layers.append(lyr)
        with self._pin_telemetry_lock:
            routes = self._pin_decode_routes
            all_pinned = self._pin_all_pinned_routes
        return {
            "enabled": enabled,
            "spec": os.environ.get(PIN_WORKING_SET_ENV),
            "refresh_tokens": parse_pin_refresh_tokens(
                os.environ.get(PIN_REFRESH_TOKENS_ENV)
            ),
            "pinned_by_layer": pinned_by_layer,
            "pinned_total": sum(pinned_by_layer.values()),
            "static_layer_count": len(static_layers),
            "decode_routes": routes,
            "all_pinned_routes": all_pinned,
            "all_pinned_hit_rate": (
                round(all_pinned / routes, 6) if routes else None
            ),
        }

    def device_route_pinned_telemetry(self) -> dict[str, Any]:
        """W71 barrier-free-layer telemetry for the pinned device route (env
        ``MTPLX_DSV41_DEVICE_ROUTE_PINNED``). Counts, per probe flush (~= per
        decode token in the steady no-recovery state), how many probed layers
        were kept barrier-free (every routed expert pinned -> slot-stable, so the
        deferred gather stands) vs recovered on the fenced path (a routed expert
        was not pinned or was force-evicted). Cumulative counters, so the
        served-event delta reports the window's barrier-free-layers-per-token."""

        enabled = os.environ.get(DEVICE_ROUTE_PINNED_ENV) == "1"
        flushes = self._device_route_pinned_flushes
        return {
            "enabled": enabled,
            "flushes": flushes,
            "barrier_free_layers": self._device_route_pinned_barrier_free_layers,
            "recovered_layers": self._device_route_pinned_recovered_layers,
            "barrier_free_layers_per_flush": (
                round(self._device_route_pinned_barrier_free_layers / flushes, 6)
                if flushes
                else None
            ),
        }

    def note_shadow_serve(
        self, layer: int, *, assignments: int, experts: int
    ) -> None:
        with self._counter_lock:
            self._shadow_serve_routes += 1
            self._shadow_serve_assignments += int(assignments)
            self._shadow_serve_experts += int(experts)

    @property
    def speculation_saturated(self) -> bool:
        """True when the speculative lane should not take more predictions.

        The lookahead hook consults this before spending router compute
        and between per-layer issues, so under admission pressure the
        farther (lower-overlap) lookahead layers are dropped first.
        """

        if self._prefetch_executor is None:
            return True
        with self._prefetch_lock:
            return len(self._prefetch_futures) >= self._prefetch_backlog_limit

    def note_gate_prefetch_predicted(self, layer: int, count: int) -> None:
        """W93: record the id volume a gate-oracle prediction handed to the ring
        for ``layer`` (before the ring dedups it against residency/inflight/
        recent-miss). No-op when the ring is off or the layer is not routed."""

        if self.config.prefetch_slots <= 0 or count <= 0:
            return
        layer_counter = self._layer_counters.get(layer)
        with self._counter_lock:
            self.counters.prefetch_predicted += int(count)
            if layer_counter is not None:
                layer_counter.prefetch_predicted += int(count)

    def _reconcile_prefetch_for_route(
        self, layer: int, expert_ids: tuple[int, ...]
    ) -> None:
        """W93 (item 3): before a DEMAND route plans ``layer``, promote settled
        ring reads to committed hits and AWAIT any needed expert whose ring read
        is still in flight -- so the true route reads the already-issued bytes
        rather than issuing a duplicate demand read.

        Runs under the layer lock the caller (``begin_split_route``) already
        holds. Deadlock-free: the awaited read runs on a prefetch worker whose
        ``_run_speculative_load`` takes the layer lock only advisorily
        (``blocking=False``) and reads through the slot state machine's own locks,
        never this layer lock. Inert unless the ring is armed, so the shipped
        demand path is byte-identical with the flag off.

        MED-a: the await is BOUNDED (``_prefetch_reconcile_timeout_s``, default
        2.0 s). On timeout the route can use a demand load while the ring retains
        the unfinished writer's assignment. Only a terminal failed/cancelled
        read can release its ticket; a running read must keep its physical slot
        until completion, even after the demand route stops waiting."""

        if self.config.prefetch_slots <= 0:
            return
        bank = self._banks.get(layer)
        if bank is None:
            return
        # The common case: the read finished during the one-layer overlap window
        # and only needs publishing (cheap, non-blocking) -> it becomes a hit.
        self._apply_prefetch_completions(layer, bank)
        awaited = 0
        fell_back = 0
        for expert in dict.fromkeys(int(value) for value in expert_ids):
            # A committed expert already hit-resolves; only a still-INFLIGHT ring
            # read has a live ticket here.
            ticket = bank.prefetch_ticket(expert)
            if ticket is None:
                continue
            with self._prefetch_lock:
                future = self._prefetch_inflight_futures.get((layer, expert))
            if future is None:
                continue
            # MED-a: bound the wait held under the layer lock. An unbounded
            # ``future.result()`` parked the generation thread on a stuck
            # speculative read for the whole route. With a timeout, ``result``
            # raises on timeout OR read failure. An unpublished ring assignment
            # is already invisible to demand planning, so a timeout need not
            # release it. Retaining it prevents a stale physical writer from
            # overwriting a newer expert in the same ring slot.
            try:
                future.result(timeout=self._prefetch_reconcile_timeout_s)
                ok = True
            except BaseException:
                ok = False
            if ok and bank.prefetch_ticket(expert) == ticket:
                # Publish the settled read directly (do not race the done
                # callback's completion record): the true route then hit-resolves
                # this ring slot instead of issuing a second read.
                if bank.commit_prefetch(expert, ticket=ticket):
                    awaited += 1
            elif not ok:
                # A finished read cannot write again. An unfinished read still
                # owns its ring slot; its completion callback will publish or
                # invalidate it once recycling is safe.
                if future.done():
                    bank.invalidate_prefetch(expert, ticket=ticket)
                fell_back += 1
        # Catch any reads that settled while we awaited above.
        self._apply_prefetch_completions(layer, bank)
        if awaited:
            with self._counter_lock:
                self.counters.prefetch_awaited_inflight += awaited
                self._layer_counters[layer].prefetch_awaited_inflight += awaited
        # W95f: unpublished experts are re-planned as demand misses by ``_plan_route_transaction``
        # (begin_split_route), where their bytes are now accounted via
        # ``miss_plan.loads``. Adding them here as well double-counted the demand
        # denominator -- and, being fallback-only, this was the sole demand writer,
        # so cold demand misses went uncounted and the budget stayed inert until a
        # fallback then latched. The demand increment moved wholesale to the plan
        # site. (``fell_back`` is left counted above for readability/future
        # telemetry; it no longer feeds any counter.)

    def prefetch_experts(
        self, layer: int, expert_ids: Iterable[int], *, verify: bool = False
    ) -> int:
        """Speculatively load predicted experts into the layer's ring tier.

        ``verify`` (W100) flags a DSpark verify-phase issue (RoutingPhase.DECODE,
        2..8-row target forward) so the issued/committed reads also accrue to
        ``prefetch_issued_verify`` / ``prefetch_committed_verify`` -- the AR (M=1)
        and verify (M=K+1) prefetch would otherwise be indistinguishable in the
        merged totals. Pure telemetry: ``verify`` never changes which reads issue
        or the gathered math.

        Returns the number of asynchronous loads issued. Zero when prefetch
        is disabled, the layer is an island or unrouted, or every prediction
        is already resident, published, or inflight. Completed loads are
        NOT published by their worker callbacks — they queue, and this
        call batch-applies them under the layer lock it already holds, so
        workers never touch layer locks. An empty ``expert_ids`` is a pure
        flush. A failed load only invalidates its ring assignment —
        speculation never marks the runtime unhealthy.
        """

        if self.config.prefetch_slots <= 0:
            return 0
        if layer in self.island_layer_set:
            return 0
        bank = self._banks.get(layer)
        if bank is None:
            return 0
        if self._closed or self._closing:
            return 0
        executor = self._prefetch_executor
        if executor is None:
            return 0
        # W95g (HIGH-1 re-review): per-token speculative SHARE with a floor (see the
        # __init__ budget note). Skip only when this token's speculative bytes exceed
        # a share ``f`` of the token's total (spec+demand) SSD bytes PLUS a floor of
        # ``floor_records`` records -- so a token's first prefetch calls always issue
        # (small/zero demand no longer latches speculation off after one call), while
        # a sustained speculative burst that dominates the drive still backs off. The
        # previous rule (spec >= budget x demand) settled to spec/(spec+demand) <= 1/3
        # -- every hidden miss shrank the demand denominator -- so a probe issued 4 on
        # the first call and skipped every later one, ~16-25 issued/token vs the ~149
        # the design table needs. ``prefetch_calls`` counts every call reaching this
        # decision so budget_skips/prefetch_calls is the visible throttle fraction.
        # Off (share 0) -> no budget (what the live windows use).
        #
        # W95f: the windows are RECOVERING per-decode-token deltas (bytes since the
        # last token boundary), not cumulative-since-open. The demand denominator
        # advances on every cold miss (plan site), and the marks re-snapshot at each
        # token boundary (``_observe_cold_start_unlocked``) / at ``reset``, so a token
        # that ran speculation ahead races demand afresh next token; the budget can
        # never latch off for the life of the process.
        self._prefetch_calls += 1
        _demand_window = self.demand_bytes_read - self._demand_bytes_at_token_start
        _spec_window = (
            self.speculative_bytes_read - self._speculative_bytes_at_token_start
        )
        if self._prefetch_byte_budget > 0.0:
            _record_bytes = self._record_bytes_for_layer(layer)
            _floor_bytes = self._prefetch_byte_floor_records * _record_bytes
            _share_cap = (
                self._prefetch_byte_budget * (_spec_window + _demand_window)
                + _floor_bytes
            )
            if _spec_window > _share_cap:
                self._prefetch_budget_skips += 1
                return 0
        # W93 HIGH-2 (self-starving ring): a speculative read that settles AFTER
        # its own layer's reconcile stays queued as an unapplied completion, and
        # its ring slot stays inflight (so unrecyclable) until that layer is
        # visited again — a whole token later. On the shared ring that pins one
        # of the 2*k slots per stuck read and starves every other layer's
        # prefetch. So before planning THIS layer, drain the settled completions
        # of every OTHER layer whose lock we can take without blocking — each
        # under its own lock, one at a time (never two layer locks held at once,
        # so this cannot deadlock; all acquisitions are non-blocking) — turning
        # settled reads into committed, recyclable entries and freeing their
        # shared slots.
        with self._prefetch_lock:
            other_pending = [
                other for other in self._prefetch_completions if other != layer
            ]
        for other in other_pending:
            other_bank = self._banks.get(other)
            other_lock = self._layer_locks.get(other)
            if other_bank is None or other_lock is None:
                continue
            if other_lock.acquire(blocking=False):
                try:
                    self._apply_prefetch_completions(other, other_bank)
                finally:
                    other_lock.release()
        lock = self._layer_locks[layer]
        # Never block the generation thread on a layer transaction: under
        # deferred split-route release the previous token's pending split
        # holds this lock until the next covering-eval flush, and that
        # flush runs on the very thread calling here. Speculation is
        # best-effort — skip the step instead of waiting.
        if not lock.acquire(blocking=False):
            # ~3827 deferred-split hazard: the predictions never issued. Count
            # them so the receipt shows the skip instead of silent loss.
            if self._prefetch_ring is not None:
                self._prefetch_ring.note_skipped_lock_held(layer)
            return 0
        try:
            self._apply_prefetch_completions(layer, bank)
            with self._prefetch_lock:
                backlog = len(self._prefetch_futures)
            if backlog >= self._prefetch_backlog_limit:
                if self._prefetch_ring is not None:
                    self._prefetch_ring.note_skipped_backlog(layer)
                return 0
            recent_misses = self._recent_route_misses.get(layer)
            if recent_misses:
                expert_ids = [
                    expert
                    for expert in expert_ids
                    if expert not in recent_misses
                ]
            loads = bank.plan_prefetch(expert_ids)
            # W93: drain the wasted-read count the plan_prefetch eviction accrued
            # while we still hold the layer lock. MED-b: the ring attributes each
            # waste to the VICTIM's layer, so drain the per-layer breakdown (the
            # victim may be a DIFFERENT layer than this evicting caller).
            wasted_by_layer = (
                self._prefetch_ring.consume_wasted_by_layer()
                if self._prefetch_ring is not None
                else {}
            )
            # Assignment tickets bind each load's completion to the exact
            # assignment it filled: the same expert can be recycled and
            # re-assigned while a callback is still queued, and that stale
            # callback must not publish (or retire) the newer assignment.
            tickets = {
                load.expert: bank.prefetch_ticket(load.expert)
                for load in loads
            }
        finally:
            lock.release()
        issued = 0
        for load in loads:
            try:
                future = executor.submit(
                    self._run_speculative_load,
                    layer,
                    load,
                    tickets[load.expert],
                )
            except RuntimeError:
                # The executor shut down mid-flight; forget the assignment
                # so a later plan_prefetch can reuse the ring slot.
                with lock:
                    bank.invalidate_prefetch(
                        load.expert,
                        ticket=tickets[load.expert],
                    )
                continue
            with self._prefetch_lock:
                self._prefetch_futures.add(future)
                # W93: index the future by (layer, expert) so a demand route can
                # await this exact read (see _reconcile_prefetch_for_route).
                self._prefetch_inflight_futures[(layer, load.expert)] = future
                # W100: tag verify-phase reads so the async commit path can
                # attribute prefetch_committed_verify without the completion
                # carrying its issuing phase.
                if verify:
                    self._verify_prefetch_tags.add(
                        (layer, load.expert, tickets[load.expert])
                    )
            future.add_done_callback(
                lambda completed, layer=layer, expert=load.expert, ticket=(
                    tickets[load.expert]
                ): self._finish_prefetch_load(layer, expert, ticket, completed)
            )
            issued += 1
        record_bytes = self._record_bytes_for_layer(layer)
        wasted_total = sum(wasted_by_layer.values())
        if issued or wasted_total:
            with self._counter_lock:
                self.counters.prefetch_issued += issued
                self._layer_counters[layer].prefetch_issued += issued
                # W100: the DSpark verify-phase slice of the same issues.
                if verify and issued:
                    self.counters.prefetch_issued_verify += issued
                    self._layer_counters[layer].prefetch_issued_verify += issued
                self.counters.prefetch_bytes += issued * record_bytes
                self._layer_counters[layer].prefetch_bytes += issued * record_bytes
                if wasted_total:
                    self.counters.prefetch_wasted += wasted_total
                    # MED-b: charge each waste to the victim layer that predicted
                    # it, not to this evicting caller.
                    for victim_layer, count in wasted_by_layer.items():
                        victim_counter = self._layer_counters.get(victim_layer)
                        if victim_counter is not None:
                            victim_counter.prefetch_wasted += count
        return issued

    def _run_speculative_load(
        self,
        layer: int,
        load: Any,
        ticket: int | None,
    ) -> None:
        """Read one predicted expert unless its assignment already recycled.

        A load can sit in the executor queue while later plan_prefetch
        calls recycle its ring assignment; performing the read anyway
        would waste SSD bandwidth on bytes whose ticketed commit is
        guaranteed to be refused. The check is advisory; physical assignment
        ownership lasts until this worker's Future is terminal. A stale commit
        check alone cannot prevent a late writer from overwriting recycled bytes.
        """

        bank = self._banks.get(layer)
        lock = self._layer_locks.get(layer)
        if bank is None or lock is None:
            return
        # Advisory only, so never block a worker on a route-held layer
        # lock: when the lock is contended the still-owned assignment permits
        # the read to proceed without waiting for generation.
        if lock.acquire(blocking=False):
            try:
                live = bank.prefetch_ticket(load.expert) == ticket
            finally:
                lock.release()
            if not live:
                return
        self.slots.load_speculative(layer, load)
        # W93 lane C: this speculative read actually touched the SSD -- account its
        # bytes on the speculative side of the demand/speculative split (the early
        # ``return`` above skips reads whose assignment already recycled).
        record_bytes = self._record_bytes_for_layer(layer)
        with self._counter_lock:
            self.speculative_bytes_read += record_bytes

    def _finish_prefetch_load(
        self,
        layer: int,
        expert: int,
        ticket: int | None,
        future: Future,
    ) -> None:
        """Record a settled load; never touch banks or layer locks here.

        This runs on a prefetch worker. Publication happens on the
        generation thread at its next ``prefetch_experts`` call for the
        layer (``_apply_prefetch_completions``), under the layer lock it
        already holds — a route-held lock therefore cannot hold a worker
        captive (measured: a 5 ms route hold turned a 34 us
        issue-to-publish path into 7.6 ms of captive-worker convoy).
        """

        with self._prefetch_lock:
            self._prefetch_futures.discard(future)
            # W93: drop the (layer, expert) index only if it still points at THIS
            # future -- a newer prefetch of the same expert may have replaced it.
            if self._prefetch_inflight_futures.get((layer, expert)) is future:
                del self._prefetch_inflight_futures[(layer, expert)]
        try:
            future.result()
        except BaseException:
            succeeded = False
        else:
            succeeded = True
        with self._prefetch_lock:
            self._prefetch_completions.setdefault(layer, []).append(
                (expert, ticket, succeeded)
            )

    def _apply_prefetch_completions(self, layer: int, bank: Any) -> None:
        """Publish settled loads for one layer; caller holds its lock.

        Ticketed commits keep the settled-tenant invariant: an assignment
        stays inflight (and its slot unrecyclable) until applied here, so
        publication still implies the physical slot holds the expert's
        bytes.
        """

        with self._prefetch_lock:
            completions = self._prefetch_completions.pop(layer, None)
        if not completions:
            return
        # W100: snapshot which of THIS layer's settling reads were verify-tagged
        # and drop every settled tag (committed or not) so the tag set cannot grow.
        # Done under _prefetch_lock, without any bank op held under it.
        keys = [(layer, expert, ticket) for expert, ticket, _ in completions]
        with self._prefetch_lock:
            verify_keys = {k for k in keys if k in self._verify_prefetch_tags}
            self._verify_prefetch_tags.difference_update(keys)
        committed = 0
        committed_verify = 0
        for expert, ticket, succeeded in completions:
            if succeeded:
                if bank.commit_prefetch(expert, ticket=ticket):
                    committed += 1
                    if (layer, expert, ticket) in verify_keys:
                        committed_verify += 1
            else:
                bank.invalidate_prefetch(expert, ticket=ticket)
        if committed:
            with self._counter_lock:
                self.counters.prefetch_committed += committed
                self._layer_counters[layer].prefetch_committed += committed
                if committed_verify:
                    self.counters.prefetch_committed_verify += committed_verify
                    self._layer_counters[
                        layer
                    ].prefetch_committed_verify += committed_verify

    def _drain_prefetch_loads(self) -> None:
        """Wait out in-flight speculative loads (bounded single-record reads).

        Reset and close call this before touching pool or bank state, then
        discard the settled-but-unapplied completions: their assignments
        die with the bank state, and applying them later would only be a
        chain of ticket-refused no-ops.
        """

        with self._prefetch_lock:
            pending = tuple(self._prefetch_futures)
        for future in pending:
            try:
                future.result()
            except BaseException:
                pass
        with self._prefetch_lock:
            self._prefetch_completions.clear()
            # W93: the futures have all settled; their (layer, expert) index dies
            # with the bank state along with the completions.
            self._prefetch_inflight_futures.clear()
            # W100: the verify-attribution tags die with the drained reads.
            self._verify_prefetch_tags.clear()

    def reset(self) -> None:
        # Deferred pin releases must flush (with a covering fence) before the
        # pool resets, or reset waits forever on the final routes' pins.
        self.flush_deferred_slot_releases(evaluate=True)
        # W44: residency is cleared below, so every device-route LUT is stale and
        # any un-flushed route probes belong to the pre-reset row. Keep the
        # captured component banks (physical arrays survive the reset).
        self._device_route_lut.clear()
        self._device_route_lut_snapshot.clear()
        self._device_route_lut_dirty.clear()
        self._device_route_pinned_lut.clear()
        self._device_route_pinned_snapshot.clear()
        self._device_route_pinned_lut_dirty.clear()
        self._device_route_probes = []
        # Straggler speculative loads hold pool lifecycle claims; without
        # this drain the pool reset below would reject them as active
        # routes. The bank reset then retires any not-yet-committed
        # assignment, so a late commit callback publishes nothing.
        self._drain_prefetch_loads()
        locks = tuple(dict.fromkeys(self._layer_locks.values()))
        for lock in locks:
            lock.acquire()
        try:
            self._raise_if_unhealthy()
            self.slots.reset()
            if self._global_bank is not None:
                self._global_bank.reset()
            else:
                for bank in self._banks.values():
                    bank.reset()
            # W93: the shared ring is reset once here (the banks delegate to it and
            # do not reset a shared ring themselves).
            if self._prefetch_ring is not None:
                self._prefetch_ring.reset()
            self._recent_route_misses.clear()
            with self._counter_lock:
                self.counters = CacheCounters()
                self._layer_counters = {
                    layer: CacheCounters() for layer in self.spec.routed_layer_indices
                }
                self._phase_counters = {
                    phase: CacheCounters() for phase in RoutingPhase
                }
                self._incremental_miss_routes = 0
                self._incremental_miss_parts = 0
                self._decode_token_index = 0
                self._decode_layers_seen = set()
                self._cold_decode_hits = 0
                self._cold_decode_requests = 0
                self._steady_decode_hits = 0
                self._steady_decode_requests = 0
                self._saw_decode_since_prefill = False
                # W95g (MEDIUM-1): zero the demand/speculative byte totals and the
                # budget-skip / prefetch-call counters HERE, matching ``counters``.
                # Leaving them cumulative made ``_runner_snapshot`` divide bytes
                # accrued across the AR pass + prefill by the post-reset decode step
                # count, so the DSpark block's per-token figures folded in the prior
                # phases. Then re-mark the budget window origin (now 0) -> an empty
                # window right after reset.
                self.demand_bytes_read = 0
                self.speculative_bytes_read = 0
                self._prefetch_budget_skips = 0
                self._prefetch_calls = 0
                self._snapshot_prefetch_byte_window()
            if self.config.trace_routes:
                with self._route_trace_lock:
                    previous_epoch = self._route_trace_epoch
                    self._route_trace_epoch += 1
                    self._route_trace_decode_step = 0
                    self._route_trace_decode_layers_seen.clear()
                    self._route_trace.append(
                        {
                            "phase": "reset",
                            "previous_trace_epoch": previous_epoch,
                            "trace_epoch": self._route_trace_epoch,
                        }
                    )
        finally:
            for lock in reversed(locks):
                lock.release()
        if self._derived_cache_policy:
            # A bank reset drops residency but keeps each bank's capacity
            # cap; re-derive it from the current KV accounting outside the
            # layer locks (the allowance path acquires them itself).
            self._apply_derived_allowance()

    def snapshot(self, *, mx_module: Any | None = None) -> dict[str, Any]:
        with self._kv_lock:
            live_kv = self._live_kv_tokens
            peak_kv = self._live_kv_peak
        with self._counter_lock:
            cache = self.counters.as_dict()
            cache_by_layer = {
                str(layer): counters.as_dict()
                for layer, counters in self._layer_counters.items()
            }
            cache_by_phase = {
                phase.value: counters.as_dict()
                for phase, counters in self._phase_counters.items()
            }
            incremental_misses = {
                "routes": self._incremental_miss_routes,
                "parts": self._incremental_miss_parts,
            }
            cold_start = self._cold_start_telemetry_locked()
        # Never hold the counter lock across slot health/fence inspection.
        slots = self.slots.snapshot()
        snapshot = {
            "model_key": self.spec.key,
            "expert_codec": self.spec.expert_codec,
            "manifest_sha256": self.manifest.manifest_sha256,
            "memory_plan": {
                "total_limit_bytes": self.plan.total_limit_bytes,
                "fixed_bytes": self.plan.fixed_bytes,
                "persistent_cache_bytes": self.plan.persistent_cache_bytes,
                "slots_per_layer": self.plan.slots_per_layer,
                "cache_scope": self.config.cache_scope,
                "global_persistent_slots": (
                    self.plan.persistent_slots
                    if self.config.cache_scope == "global"
                    else None
                ),
                "transient_slots": self.plan.transient_slots,
                "batch_admission_slots": self._batch_admission_slots(),
                "single_slot_pool": self._single_slot_pool,
                "allocated_bytes": self.plan.allocated_bytes,
                "unallocated_bytes": self.plan.unallocated_bytes,
                "miss_shadow": self.plan.miss_shadow,
                "shadow_bytes": self.plan.shadow_bytes,
            },
            "expert_cache_policy": {
                "derived": self._derived_cache_policy,
                "allowance_bytes": self._derived_allowance_bytes,
                "persistent_capacity": self._derived_capacity_slots,
                "cached_bytes": (
                    self._global_bank.occupancy
                    if self._global_bank is not None
                    else sum(bank.occupancy for bank in self._banks.values())
                )
                * self._representative_record_bytes,
            },
            "memory_cap": self.memory_cap_report,
            "integrity": self.integrity_report,
            "mlx_memory": mlx_memory_telemetry(mx_module),
            "live_kv_tokens": live_kv,
            "live_kv_tokens_peak": peak_kv,
            "cache": cache,
            "cache_by_layer": cache_by_layer,
            "cache_by_phase": cache_by_phase,
            "incremental_misses": incremental_misses,
            "cold_start": cold_start,
            "slots": slots,
            "pin_working_set": self.pinned_working_set_telemetry(),
            "device_route_pinned": self.device_route_pinned_telemetry(),
        }
        # W110: the io-thread reader metrics (per-record sha256 engagement +
        # bytes/reads) so the MTPLX_DSV41_VERIFY_RECORD_HASHES lever's counters
        # (records_hashed / records_unhashed / hash_thread_ns_total) travel on the receipt.
        # Guarded: a stub reader without metrics just omits the block.
        try:
            snapshot["io"] = self.reader.metrics.as_dict()
        except Exception:
            pass
        if self._belady_oracle is not None:
            # The clairvoyant fetch floor over the full decode window, at the
            # actual per-layer slot budget — the runtime analog of the offline
            # replay, reported alongside the measured loads for a live gap.
            try:
                with self._census_lock:
                    snapshot["belady_oracle"] = self._belady_oracle.report(
                        self.plan.slots_per_layer
                    )
            except Exception:
                pass
        if self._global_bank is not None:
            snapshot["global_cache"] = {
                **self._global_bank.snapshot(),
                "resident_experts_by_layer": {
                    str(layer): list(experts)
                    for layer, experts in self._global_bank.resident_experts_by_layer.items()
                },
            }
        if self._mapped_expert_store is not None:
            snapshot["mapped_experts"] = self._mapped_expert_store.snapshot()
        if self._island_store is not None:
            snapshot["island_experts"] = self._island_store.snapshot()
        if self._banked_island_store is not None:
            snapshot["banked_island_experts"] = (
                self._banked_island_store.snapshot()
            )
        if self._shadow_store is not None:
            shadow = self._shadow_store.snapshot()
            with self._counter_lock:
                shadow["serve_routes"] = self._shadow_serve_routes
                shadow["served_assignments"] = self._shadow_serve_assignments
                shadow["served_experts"] = self._shadow_serve_experts
            snapshot["shadow_experts"] = shadow
        snapshot["speculative_io"] = {
            "fraction": self.config.speculative_io_fraction,
            "max_concurrent_reads": self._prefetch_max_reads,
            "max_inflight_bytes": (
                self._prefetch_max_reads * self._representative_record_bytes
            ),
        }
        if self._pipeline_ledger is not None:
            snapshot["expert_pipeline"] = self._pipeline_ledger.snapshot()
        # W95f (review HIGH-3): carry the gate-oracle prefetch + v2 runner receipt
        # blocks on THIS snapshot too (not only resource_telemetry_snapshot), so the
        # served daemon's stream-counter path (expert_streaming_snapshot -> snapshot
        # -> serve_stream_counters.snapshot_stream_counters) logs the SSD-hiding
        # counters. Guarded, so the shipped/off snapshot is byte-unchanged.
        if self.config.prefetch_slots > 0:
            snapshot["gate_prefetch"] = self._gate_prefetch_snapshot(
                cache, cache_by_layer
            )
        if os.environ.get("MTPLX_DSV41_RUNNER") == "v2":
            snapshot["runner"] = self._runner_snapshot(cache, cold_start)
        # W125: served-path host decode timeline (additive; only when armed).
        if _tl.enabled():
            snapshot["decode_timeline"] = _tl.snapshot()
        self._raise_if_unhealthy()
        return snapshot

    def resource_telemetry_snapshot(
        self,
        *,
        mx_module: Any | None = None,
    ) -> dict[str, Any]:
        """Return cheap cumulative counters for the benchmark sampler."""

        with self._counter_lock:
            cache = self.counters.as_dict()
            cache_by_layer = {
                str(layer): counters.as_dict()
                for layer, counters in self._layer_counters.items()
            }
            cache_by_phase = {
                phase.value: counters.as_dict()
                for phase, counters in self._phase_counters.items()
            }
            incremental_misses = {
                "routes": self._incremental_miss_routes,
                "parts": self._incremental_miss_parts,
            }
            cold_start = self._cold_start_telemetry_locked()
        # Pool occupancy has independent locks; do not hold the counter lock
        # across that snapshot.
        slots = self.slots.resource_telemetry_snapshot()
        snapshot = {
            "model_key": self.spec.key,
            "quant_bits": self.spec.quant_bits,
            "expert_record_bytes": self._representative_record_bytes,
            "mlx_memory": mlx_memory_telemetry(mx_module),
            "cache": cache,
            "cache_by_layer": cache_by_layer,
            "cache_by_phase": cache_by_phase,
            "incremental_misses": incremental_misses,
            "cold_start": cold_start,
            "pin_working_set": self.pinned_working_set_telemetry(),
            "device_route_pinned": self.device_route_pinned_telemetry(),
            **slots,
        }
        # W110: io-thread reader metrics (per-record sha256 engagement) for the
        # bench sampler's receipt too (records_hashed / records_unhashed /
        # hash_thread_ns_total). Guarded; a stub reader just omits it.
        try:
            snapshot["io"] = self.reader.metrics.as_dict()
        except Exception:
            pass
        # W125: per-token/per-layer host decode timeline (only when the probe is
        # armed via MTPLX_DSV41_DECODE_TIMELINE=1, so the shipped snapshot is
        # byte-unchanged off). Additive key.
        if _tl.enabled():
            snapshot["decode_timeline"] = _tl.snapshot()
        if self._pipeline_ledger is not None:
            snapshot["expert_pipeline"] = self._pipeline_ledger.snapshot()
        # W93: the gate-oracle prefetch receipt block (only when the ring is armed,
        # so the shipped snapshot is unchanged with the flag off).
        if self.config.prefetch_slots > 0:
            snapshot["gate_prefetch"] = self._gate_prefetch_snapshot(
                cache, cache_by_layer
            )
        # W95: the v2 runner receipt block -- ONE rolled-up view of the composed
        # SSD-hiding runner (single pool + prefetch ring + overlap_miss_reads).
        # Only when MTPLX_DSV41_RUNNER=v2 is armed, so the shipped snapshot is
        # byte-unchanged off.
        if os.environ.get("MTPLX_DSV41_RUNNER") == "v2":
            snapshot["runner"] = self._runner_snapshot(cache, cold_start)
        return snapshot

    def _runner_snapshot(
        self, cache: dict[str, Any], cold_start: dict[str, Any]
    ) -> dict[str, Any]:
        """W95 v2 runner receipt block, built from the already-collected cache
        counters + W87 cold-start telemetry. Emitted by BOTH ``snapshot`` (the
        served daemon's stream-counter path) and ``resource_telemetry_snapshot``
        (the bench sampler), so the paired window reads the SAME SSD-hiding counters
        whether it scrapes the daemon or the harness receipt.

        W95f (review HIGH-3): carries the ``committed+awaited`` denominator so a
        prefetch HIT RATE can be derived, and PER-DECODE-TOKEN normalisations
        (misses/token, speculative-bytes/token) rather than only cumulative-since-
        open -- the window targets are per token."""

        # W95f (review MEDIUM): report the RESOLVED prefetch width / margin -- the
        # values the run actually uses -- not the raw env. Re-parsing the env here
        # (``int(env)`` / ``float(env)``) raised ValueError on a malformed
        # MTPLX_DSV41_GATE_PREFETCH_MARGIN (e.g. "abc") and lost the whole receipt,
        # even though the run proceeded because ``_resolve_gate_prefetch_margin``
        # swallows the bad value. The resolvers also apply the v2 default + explicit-
        # override precedence, so the receipt matches the routed behaviour. Lazy
        # import keeps this low-level module free of the deepseek_v41 import cycle
        # (see the single_slot_pool note in ``open``); a resolver failure falls back
        # to the v2 defaults rather than dropping the receipt.
        # W95g (review LOW-1): also resolve the DSpark verify-phase prefetch gate so
        # the receipt stamps whether the verify speculates and its row bound -- a
        # self-describing receipt (the verify shares the AR width/margin and the
        # speculative-byte throttle, stamped in ``verify_prefetch`` below).
        try:
            from mtplx.models.deepseek_v41 import (
                _resolve_gate_prefetch_k,
                _resolve_gate_prefetch_margin,
                _runner_v2_enabled,
                _RUNNER_V2_VERIFY_MAX_ROWS,
            )

            _k_resolved = int(_resolve_gate_prefetch_k())
            _margin_resolved = float(_resolve_gate_prefetch_margin())
            _v2_on = bool(_runner_v2_enabled())
            _verify_max_rows = int(_RUNNER_V2_VERIFY_MAX_ROWS)
        except Exception:
            _k_resolved, _margin_resolved = 6, -0.05
            _v2_on, _verify_max_rows = True, 8
        # committed+awaited denominator: a settled ring read is counted in
        # prefetch_committed XOR prefetch_awaited_inflight (the demand-await publish
        # path), so their sum is the settled+published total and a prefetch hit rate
        # (hit_on_true_route / this) is <= 1.0.
        _committed = int(cache.get("prefetch_committed", 0))
        _awaited = int(cache.get("prefetch_awaited_inflight", 0))
        _committed_settled = _committed + _awaited
        _hit = int(cache.get("prefetch_hit_on_true_route", 0))
        # W95g (review MEDIUM-2): ``prefetch_hit_on_true_route`` counts EVERY
        # consumption of a prefetched record, so a resident ring record re-consumed
        # across tokens makes ``prefetch_hit_rate`` (hit / committed+awaited) exceed
        # 1.0. ``prefetch_first_consumption_hits`` counts each record's hit at most
        # once, and ``prefetch_first_hit_rate`` = it / records ISSUED is bounded
        # [0,1]. Both keys are ADDED; ``prefetch_hit_rate`` keeps its meaning.
        _first_hit = int(cache.get("prefetch_first_consumption_hits", 0))
        _issued = int(cache.get("prefetch_issued", 0))
        # per-decode-token normalisations. ``decode_steps`` is the W87 decode-step
        # index (one full routed-layer sweep == one token under --decode-mode ar).
        _steps = int(cold_start.get("decode_steps_observed", 0))
        _misses = int(cache.get("expert_misses", 0))
        _bytes_read = int(cache.get("bytes_read", 0))
        _demand = int(getattr(self, "demand_bytes_read", 0))
        _spec = int(getattr(self, "speculative_bytes_read", 0))

        def _per_tok(value: int) -> float | None:
            return (value / _steps) if _steps else None

        return {
            "mode": "v2",
            "single_pool": bool(getattr(self, "_single_slot_pool", False)),
            "overlap_miss_reads": bool(
                getattr(self.config, "overlap_miss_reads", False)
            ),
            "decode_miss_records_per_part": getattr(
                self.config,
                "decode_miss_records_per_part",
                None,
            ),
            # W123: resolved io read-fanout (1 == OFF; the reader holds the
            # env-override-applied value). Pairs with io.read_inflight_max and
            # io.read_ns/decode_wall_s to price the realized read-pool queue
            # depth in the receipt.
            "io_read_fanout": int(
                getattr(getattr(self, "reader", None), "io_read_fanout", 1)
            ),
            # the retuned prefetch knobs (W95): AR predict width, confidence
            # margin, global ring size, and the demand-priority byte budget.
            "prefetch_k": _k_resolved,
            "prefetch_margin": _margin_resolved,
            "ring_slots": int(getattr(self.config, "prefetch_slots", 0)),
            # W95g: byte_budget is now the per-token speculative SHARE f (0 = off),
            # with a floor of byte_floor_records records; budget_skips / prefetch_calls
            # is the visible throttle fraction.
            "byte_budget": float(getattr(self, "_prefetch_byte_budget", 0.0)),
            "byte_floor_records": int(
                getattr(self, "_prefetch_byte_floor_records", 0)
            ),
            "budget_skips": int(getattr(self, "_prefetch_budget_skips", 0)),
            "prefetch_calls": int(getattr(self, "_prefetch_calls", 0)),
            # W95g (review LOW-1): the DSpark verify-phase prefetch gate, stamped so
            # the receipt self-describes what the verify did. The verify predicts the
            # per-row UNION of gate_L one layer ahead ONLY under v2, on a
            # <= ``max_rows`` batch in the DECODE routing phase; it reuses the AR
            # predict width (``k`` == prefetch_k) + confidence ``margin`` (==
            # prefetch_margin) and shares the same speculative-byte throttle
            # (``byte_budget`` / ``byte_floor_records`` == the runner-level keys
            # above, by construction). ``enabled`` = the gate will speculate on a
            # verify (v2 armed, width > 0, ring present).
            "verify_prefetch": {
                "enabled": bool(
                    _v2_on
                    and _k_resolved > 0
                    and int(getattr(self.config, "prefetch_slots", 0)) > 0
                ),
                "max_rows": _verify_max_rows,
                "k": _k_resolved,
                "margin": _margin_resolved,
                "byte_budget": float(getattr(self, "_prefetch_byte_budget", 0.0)),
                "byte_floor_records": int(
                    getattr(self, "_prefetch_byte_floor_records", 0)
                ),
            },
            # SSD read (demand vs speculative split shows the drive contention).
            "expert_misses": _misses,
            "bytes_read": _bytes_read,
            "hit_rate": float(cache.get("hit_rate", 0.0)),
            "demand_bytes_read": _demand,
            "speculative_bytes_read": _spec,
            "prefetch_issued": _issued,
            # W100: the DSpark verify-phase slice, so the paired window can prove
            # the multi-row verify engaged the prefetch independent of the AR total
            # (prefetch_issued above merges AR M=1 and verify M=K+1).
            "prefetch_issued_verify": int(cache.get("prefetch_issued_verify", 0)),
            "prefetch_committed_verify": int(
                cache.get("prefetch_committed_verify", 0)
            ),
            "prefetch_hit_on_true_route": _hit,
            "prefetch_wasted": int(cache.get("prefetch_wasted", 0)),
            "prefetch_bytes": int(cache.get("prefetch_bytes", 0)),
            "pool_promotions": int(cache.get("promotions", 0)),
            "pool_loads": int(cache.get("pool_loads", 0)),
            # committed+awaited denominator + the derived prefetch hit rate.
            # NOTE (W95g MEDIUM-2): this key counts every consumption in the
            # numerator and CAN exceed 1.0; use ``prefetch_first_hit_rate`` below for
            # the bounded [0,1] rate. Kept unchanged for continuity (existing key).
            "prefetch_committed": _committed_settled,
            "prefetch_awaited_inflight": _awaited,
            "prefetch_hit_rate": (
                _hit / _committed_settled if _committed_settled else 0.0
            ),
            # W95g (review MEDIUM-2): first-consumption hits (each prefetched record
            # counted at most once) and the bounded [0,1] rate over records issued.
            "prefetch_first_consumption_hits": _first_hit,
            "prefetch_first_hit_rate": (
                _first_hit / _issued if _issued else 0.0
            ),
            # per-decode-token normalisations (the window targets are per token).
            "decode_steps": _steps,
            "expert_misses_per_token": _per_tok(_misses),
            "bytes_read_per_token": _per_tok(_bytes_read),
            "demand_bytes_per_token": _per_tok(_demand),
            "speculative_bytes_per_token": _per_tok(_spec),
            # Not timed on this path; the receipt derives SSD ms/token from
            # expert_misses x record_bytes / realized BW.
            "ssd_wait_ms_per_token": None,
        }

    def _gate_prefetch_snapshot(
        self, cache: dict[str, Any], cache_by_layer: dict[str, Any]
    ) -> dict[str, Any]:
        """Assemble the W93 ``gate_prefetch`` receipt block from the already
        collected cache counters: totals, per-layer hit rate, and a one-line
        census. ``min_layer`` reads the DSV4.1 lever env (gate-prefetch is a
        DSV4.1 feature); a bad value falls back to the shipped default 4."""

        try:
            min_layer = max(
                0, int(os.environ.get("MTPLX_DSV41_GATE_PREFETCH_MIN_LAYER") or 4)
            )
        except (TypeError, ValueError):
            min_layer = 4
        committed = int(cache.get("prefetch_committed", 0))
        awaited = int(cache.get("prefetch_awaited_inflight", 0))
        hit = int(cache.get("prefetch_hit_on_true_route", 0))
        issued = int(cache.get("prefetch_issued", 0))
        # W95g (review MEDIUM-2): first-consumption hits (each record at most once).
        # ``hit_rate`` below counts every consumption in its numerator and can exceed
        # 1.0; ``first_hit_rate`` = first_consumption_hits / issued is bounded [0,1].
        first_hit = int(cache.get("prefetch_first_consumption_hits", 0))
        bytes_prefetched = int(cache.get("prefetch_bytes", 0))
        # W93 MED-b: an awaited-inflight commit is a ring read that SETTLED and
        # PUBLISHED — the demand route blocked on the in-flight read and committed
        # it (``_reconcile_prefetch_for_route``) — which is exactly the meaning of
        # ``committed``, but it published on the demand-await path rather than the
        # async-completion path that increments ``prefetch_committed``. It then
        # counts as a ``hit_on_true_route``, so without folding it in, hit can
        # exceed committed and hit_rate > 1.0. A settle is counted in committed
        # XOR awaited (never both), so the sum is the true settled+published total
        # and hit_rate <= 1.0.
        committed_settled = committed + awaited
        # W93 HIGH-2: predictions that never issued, by reason (cumulative, read
        # from the shared ring; keyed by target layer).
        skips = (
            self._prefetch_ring.prefetch_skip_snapshot()
            if self._prefetch_ring is not None
            else {
                "dropped_no_slot": {},
                "skipped_lock_held": {},
                "skipped_backlog": {},
            }
        )
        dropped_no_slot = sum(skips["dropped_no_slot"].values())
        skipped_lock_held = sum(skips["skipped_lock_held"].values())
        skipped_backlog = sum(skips["skipped_backlog"].values())
        block: dict[str, Any] = {
            "k": int(self.config.prefetch_slots),
            "min_layer": min_layer,
            "predicted": int(cache.get("prefetch_predicted", 0)),
            "issued": issued,
            "committed": committed_settled,
            "hit_on_true_route": hit,
            "first_consumption_hits": first_hit,
            "wasted": int(cache.get("prefetch_wasted", 0)),
            "awaited_inflight": awaited,
            "dropped_no_slot": dropped_no_slot,
            "skipped_lock_held": skipped_lock_held,
            "skipped_backlog": skipped_backlog,
            "bytes_prefetched": bytes_prefetched,
            "hit_rate": (hit / committed_settled) if committed_settled else 0.0,
            "first_hit_rate": (first_hit / issued) if issued else 0.0,
        }
        per_layer: dict[str, Any] = {}
        for layer, lc in cache_by_layer.items():
            lcommitted = int(lc.get("prefetch_committed", 0))
            lawaited = int(lc.get("prefetch_awaited_inflight", 0))
            lhit = int(lc.get("prefetch_hit_on_true_route", 0))
            lfirst_hit = int(lc.get("prefetch_first_consumption_hits", 0))
            lissued = int(lc.get("prefetch_issued", 0))
            ldropped = int(skips["dropped_no_slot"].get(int(layer), 0))
            lskip_lock = int(skips["skipped_lock_held"].get(int(layer), 0))
            lskip_backlog = int(skips["skipped_backlog"].get(int(layer), 0))
            if not (
                lc.get("prefetch_predicted")
                or lc.get("prefetch_issued")
                or lhit
                or ldropped
                or lskip_lock
                or lskip_backlog
            ):
                continue
            lcommitted_settled = lcommitted + lawaited
            per_layer[str(layer)] = {
                "predicted": int(lc.get("prefetch_predicted", 0)),
                "issued": lissued,
                "committed": lcommitted_settled,
                "hit_on_true_route": lhit,
                "first_consumption_hits": lfirst_hit,
                "wasted": int(lc.get("prefetch_wasted", 0)),
                "awaited_inflight": lawaited,
                "dropped_no_slot": ldropped,
                "skipped_lock_held": lskip_lock,
                "skipped_backlog": lskip_backlog,
                "hit_rate": (lhit / lcommitted_settled) if lcommitted_settled else 0.0,
                "first_hit_rate": (lfirst_hit / lissued) if lissued else 0.0,
            }
        block["per_layer"] = per_layer
        block["census"] = (
            f"gate_prefetch k={block['k']} min_layer={min_layer}: "
            f"predicted={block['predicted']} issued={block['issued']} "
            f"committed={committed_settled} hit={hit} (rate {block['hit_rate']:.3f}) "
            f"first_hit={first_hit} (rate {block['first_hit_rate']:.3f}) "
            f"wasted={block['wasted']} awaited={awaited} "
            f"dropped={dropped_no_slot} skipped_lock={skipped_lock_held} "
            f"skipped_backlog={skipped_backlog} "
            f"bytes={bytes_prefetched / (1024 * 1024):.1f}MiB"
        )
        return block

    def _flush_route_census(self) -> None:
        """Merge session decode counts to disk and re-derive the placement.

        Runs once, off the decode path, at close(). Merges into any
        existing ``route-census.json`` (windowed decay lives in
        ``RouteCensus.merge``) and rewrites ``island-placement.json`` from
        the merged census. Every write is atomic (tempfile + rename) and
        best-effort: failures log and never fail close(). ``reset()``
        deliberately does not clear the census — it is session-scoped
        diagnostic state, not routing policy.
        """

        with self._census_lock:
            census = self._route_census
            self._route_census = None
        if census is None or census.total_routed_assignments == 0:
            return
        try:
            updated_at = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            census_path = self.root / ROUTE_CENSUS_FILENAME
            merged = census
            if census_path.is_file():
                try:
                    existing = load_census(census_path)
                except RouteCensusError:
                    _LOGGER.warning(
                        "existing %s is unusable; rebuilding from this "
                        "session's census",
                        census_path,
                        exc_info=True,
                    )
                else:
                    if existing.model_key == census.model_key:
                        existing.merge(census)
                        merged = existing
                    else:
                        _LOGGER.warning(
                            "existing %s belongs to %r, not %r; rebuilding",
                            census_path,
                            existing.model_key,
                            census.model_key,
                        )
            save_census(merged, census_path, updated_at=updated_at)
            placement = derive_placement(
                merged,
                expert_count=self.spec.expert_count,
                slots_per_layer_hint=(
                    self.plan.slots_per_layer or self.spec.top_k
                ),
            )
            save_placement(
                placement,
                self.root / ISLAND_PLACEMENT_FILENAME,
                updated_at=updated_at,
            )
        except Exception:
            _LOGGER.warning(
                "route census flush failed; placement artifacts unchanged",
                exc_info=True,
            )

    def close(self, *, timeout: float | None = None) -> None:
        self.flush_deferred_slot_releases(evaluate=True)
        deadline = None if timeout is None else time.monotonic() + timeout
        if deadline is None:
            self._close_lock.acquire()
        else:
            remaining = max(0.0, deadline - time.monotonic())
            if not self._close_lock.acquire(timeout=remaining):
                raise TimeoutError(
                    "expert streaming runtime close already in progress at deadline"
                )
        try:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            if self._closed:
                slots_error: BaseException | None = None
                try:
                    self.slots.close(timeout=remaining)
                except BaseException as exc:
                    slots_error = exc
                if slots_error is not None:
                    raise slots_error
                self._raise_cleanup_error()
                return
            self._closing = True
            # In-flight speculative loads hold pool lifecycle claims; drain
            # them (and stop accepting more) before the pool close waits on
            # active routes.
            self._drain_prefetch_loads()
            if self._prefetch_executor is not None:
                self._prefetch_executor.shutdown(
                    wait=deadline is None,
                    cancel_futures=True,
                )
            slots_error: BaseException | None = None
            try:
                self.slots.close(timeout=remaining)
            except BaseException as exc:
                if not self.slots._closed:
                    raise
                slots_error = exc
            self._split_executor.shutdown(
                wait=deadline is None,
                cancel_futures=True,
            )
            if self._mapped_expert_store is not None:
                self._mapped_expert_store.close()
                self._mapped_expert_store = None
            if self._island_store is not None:
                self._island_store.close()
                self._island_store = None
            if self._banked_island_store is not None:
                self._banked_island_store.close()
                self._banked_island_store = None
            self._flush_route_census()
            self._closed = True
            self._closing = False
            if slots_error is not None:
                raise slots_error
            self._raise_cleanup_error()
        finally:
            self._close_lock.release()

    def __enter__(self) -> ExpertStreamingRuntime:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def load_configured_expert_runtime(
    root: Path | str,
    manifest_path: Path | str,
    config: ExpertStreamingConfig,
    **kwargs: Any,
) -> ExpertStreamingRuntime:
    try:
        return ExpertStreamingRuntime.open(root, manifest_path, config, **kwargs)
    except ExpertManifestError as exc:
        raise ExpertStreamingConfigurationError(str(exc)) from exc
