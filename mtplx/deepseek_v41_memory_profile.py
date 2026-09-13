"""DeepSeek-V4.1 explicit, self-limiting memory profile (W62).

Three concerns live here, all MLX-optional and ``psutil``-free so the CPU test
suite can exercise them without a Metal device or the 195 GB artifact:

1. **Instrumentation** -- :func:`memory_profile_snapshot` gathers, at a named
   phase (load-end / after-prefill / per-N-decode), the three MLX allocator
   accessors (:func:`mlx_memory_snapshot`), the process footprint via ``mach``
   ``task_info`` (:func:`process_rss_snapshot`, no ``psutil``), the box totals
   from ``vm_stat`` (:func:`box_memory_snapshot`), and the planner's byte
   breakdown (:func:`plan_breakdown`).  :func:`format_memory_profile_table`
   renders one table for a window receipt.

2. **The allocator overhang.**  ``mtplx.expert_runtime.apply_mlx_memory_cap``
   sets ``mx.set_memory_limit`` (active-allocation ceiling) but never bounds the
   allocator's *freed-buffer* cache, so decode/prefill transients that are freed
   are retained by MLX for the process lifetime -- memory the plan never priced.
   :func:`apply_allocator_cache_limit` calls ``mx.set_cache_limit`` from the
   plan/derivation so the retained cache stays inside the runtime reserve.  This
   is exactly the served path's ``_configure_mlx_cache_limit`` machinery
   (``mtplx/server/openai.py``), applied to the CLI/bench lane that lacked it.

3. **A budget-derived plan.**  The window operators hand-picked ``82``/``92``
   GiB ``--memory-limit-gib`` and drove the box into the kernel-panic zone
   (window 28: 92 GiB plan -> 102 GB box, 0.1 GB free, 7 GB compressed, decode
   paging).  :func:`derive_plan_from_budget` inverts that: start from David's
   TOTAL box budget (``MTPLX_DSV41_BOX_BUDGET_GB``, default 100) and subtract the
   macOS floor, the measured host overhead, and the allocator cache limit to get
   the plan.  ``--memory-limit-gib`` stays as an explicit override.

The byte model (where a token's memory lands), all held simultaneously in the
resident-set of the DSV4.1 process during a window with the Qwen server booted
out:

    box_total  =  macOS_floor            (~6 GiB, kept free/file-backed)
               +  mlx_active             (weights + KV + live transients;
                                          bounded by set_memory_limit = plan -
                                          reserve - io_staging)
               +  mlx_cache              (freed-buffer retention; bounded by
                                          set_cache_limit -- item (2))
               +  host_overhead          (python heap, positional bank read
                                          buffers, engram host-side row LRU
                                          ~2 GiB, tokenizer; NOT counted by
                                          mx.get_peak_memory -- the "8-12 GB
                                          outside the plan" David measured)

    => plan  =  box_budget - macOS_floor - host_overhead - cache_limit   (item 3)

The runtime reserve inside the plan (7 GiB) is headroom for prefill transients
-- the K30 selected-key attention bounds the 16K score transient to
``[rows, H, window + index_topk]`` (T-independent, ``mtplx/models/deepseek_v41.py``)
and the K26 dense-expert prefill batch is bounded to ~0.57 GB
(``mtplx/models/expert_mlx.py``), both well inside the reserve.
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

GIB = 1024**3

# --------------------------------------------------------------------------
# Derivation constants (documented; a few are env-overridable for tuning).
# --------------------------------------------------------------------------

#: David's TOTAL box-use safety budget in GB (the resident Qwen server is booted
#: out during a GPU window, so the DSV4.1 process + macOS is all that runs).
#: The box panics at ~110 GB; 100 keeps a margin below that.
DEFAULT_BOX_BUDGET_GIB = 100.0

#: macOS + file-cache pages that must stay free/file-backed or the compressor
#: starts (measured: at a 92 GiB plan the box sat at 102 GB with 0.1 GB free and
#: 7 GB already in the compressor).  ~6 GiB is the assumed floor.
MACOS_FLOOR_GIB = 6.0

#: Process RSS *above* the MLX allocator's own accounting -- the python heap,
#: the positional-expert bank read buffers, the engram host-side row LRU
#: (``DEFAULT_ENGRAM_CACHE_BYTES`` = 2 GiB), and the tokenizer.  David measured
#: 8-12 GB of process memory outside the plan; 10 GiB is the midpoint and is the
#: profile constant the derivation subtracts.  ``mx.get_peak_memory`` never sees
#: any of this, so it must come off the box budget explicitly.
HOST_OVERHEAD_GIB = 10.0

#: Bound on the MLX allocator's freed-buffer cache (``mx.set_cache_limit``).
#: Kept strictly below the 7 GiB runtime reserve so freed transients never
#: accumulate past the reserve.  Mirrors the served ``_default_mlx_cache_limit``
#: tier for a <=100 GB box.
ALLOCATOR_CACHE_LIMIT_GIB = 6.0

#: In-plan headroom (``runtime_reserve_bytes``) for prefill transients; matches
#: ``mtplx.models.deepseek_v41_loader.DEFAULT_RUNTIME_RESERVE_BYTES`` (7 GiB).
RUNTIME_RESERVE_GIB = 7.0

#: K26 (W51) dense-expert prefill batch bound: at the default batch of 8 experts
#: at most 8 dequantized bf16 copies (~71 MB each at 5120x2304) are live at once
#: (~0.57 GB) and none survives the call -- ``mtplx/models/expert_mlx.py``
#: ``_gather_component_bank_dense``.  Priced into the reserve, not the plan body.
PREFILL_DENSE_TRANSIENT_GIB = 0.57

# Env overrides (all optional; unset -> the constants above).
ENV_BOX_BUDGET = "MTPLX_DSV41_BOX_BUDGET_GB"
ENV_MACOS_FLOOR = "MTPLX_DSV41_MACOS_FLOOR_GB"
ENV_HOST_OVERHEAD = "MTPLX_DSV41_HOST_OVERHEAD_GB"
ENV_CACHE_LIMIT = "MTPLX_DSV41_MLX_CACHE_LIMIT_GB"


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{key} must be positive, got {value}")
    return value


# --------------------------------------------------------------------------
# (3) Budget -> plan derivation.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetDerivation:
    """The resolved plan and its derivation from a TOTAL box budget."""

    source: str  # "budget" | "override"
    box_budget_gib: float
    macos_floor_gib: float
    host_overhead_gib: float
    cache_limit_gib: float
    runtime_reserve_gib: float
    prefill_transient_gib: float
    plan_gib: float
    memory_limit_bytes: int
    cache_limit_bytes: int
    runtime_reserve_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "box_budget_gib": round(self.box_budget_gib, 4),
            "macos_floor_gib": round(self.macos_floor_gib, 4),
            "host_overhead_gib": round(self.host_overhead_gib, 4),
            "cache_limit_gib": round(self.cache_limit_gib, 4),
            "runtime_reserve_gib": round(self.runtime_reserve_gib, 4),
            "prefill_transient_gib": round(self.prefill_transient_gib, 4),
            "plan_gib": round(self.plan_gib, 4),
            "memory_limit_bytes": int(self.memory_limit_bytes),
            "cache_limit_bytes": int(self.cache_limit_bytes),
            "runtime_reserve_bytes": int(self.runtime_reserve_bytes),
            "formula": self.formula(),
        }

    def formula(self) -> str:
        if self.source == "override":
            return (
                f"plan={self.plan_gib:.4g} GiB (explicit --memory-limit-gib "
                f"override); cache_limit={self.cache_limit_gib:.4g} GiB"
            )
        return (
            f"plan = budget({self.box_budget_gib:.4g}) "
            f"- macOS_floor({self.macos_floor_gib:.4g}) "
            f"- host_overhead({self.host_overhead_gib:.4g}) "
            f"- cache_limit({self.cache_limit_gib:.4g}) "
            f"= {self.plan_gib:.4g} GiB"
        )


def derive_plan_from_budget(
    *,
    box_budget_gib: float | None = None,
    macos_floor_gib: float | None = None,
    host_overhead_gib: float | None = None,
    cache_limit_gib: float | None = None,
    runtime_reserve_gib: float = RUNTIME_RESERVE_GIB,
    prefill_transient_gib: float = PREFILL_DENSE_TRANSIENT_GIB,
    override_memory_limit_gib: float | None = None,
    env: Mapping[str, str] | None = None,
) -> BudgetDerivation:
    """Resolve the expert-streaming plan from David's TOTAL box budget.

    ``plan = budget - macOS_floor - host_overhead - cache_limit``.  Any argument
    left ``None`` falls back to its env var (``MTPLX_DSV41_*``) and then its
    profile constant.  ``override_memory_limit_gib`` (the explicit
    ``--memory-limit-gib``) bypasses the subtraction and pins the plan; the cache
    limit is still resolved so the overhang fix applies to overrides too.
    """

    source_env: Mapping[str, str] = os.environ if env is None else env
    budget = (
        _env_float(source_env, ENV_BOX_BUDGET, DEFAULT_BOX_BUDGET_GIB)
        if box_budget_gib is None
        else float(box_budget_gib)
    )
    floor = (
        _env_float(source_env, ENV_MACOS_FLOOR, MACOS_FLOOR_GIB)
        if macos_floor_gib is None
        else float(macos_floor_gib)
    )
    host = (
        _env_float(source_env, ENV_HOST_OVERHEAD, HOST_OVERHEAD_GIB)
        if host_overhead_gib is None
        else float(host_overhead_gib)
    )
    cache = (
        _env_float(source_env, ENV_CACHE_LIMIT, ALLOCATOR_CACHE_LIMIT_GIB)
        if cache_limit_gib is None
        else float(cache_limit_gib)
    )
    reserve = float(runtime_reserve_gib)
    transient = float(prefill_transient_gib)

    if override_memory_limit_gib is not None:
        plan_gib = float(override_memory_limit_gib)
        source = "override"
    else:
        plan_gib = budget - floor - host - cache
        source = "budget"

    if plan_gib <= 0:
        raise ValueError(
            f"derived plan is non-positive ({plan_gib:.4g} GiB): budget "
            f"{budget:.4g} - floor {floor:.4g} - host {host:.4g} - cache "
            f"{cache:.4g}; raise {ENV_BOX_BUDGET} or lower the deductions"
        )
    # The reserve + the bounded prefill transient must fit under the plan, or the
    # fixed footprint alone overflows before a single expert slot is priced.
    if plan_gib <= reserve + transient:
        raise ValueError(
            f"derived plan {plan_gib:.4g} GiB does not cover the runtime "
            f"reserve {reserve:.4g} GiB + bounded prefill transient "
            f"{transient:.4g} GiB; raise {ENV_BOX_BUDGET}"
        )

    return BudgetDerivation(
        source=source,
        box_budget_gib=budget,
        macos_floor_gib=floor,
        host_overhead_gib=host,
        cache_limit_gib=cache,
        runtime_reserve_gib=reserve,
        prefill_transient_gib=transient,
        plan_gib=plan_gib,
        memory_limit_bytes=int(round(plan_gib * GIB)),
        cache_limit_bytes=int(round(cache * GIB)),
        runtime_reserve_bytes=int(round(reserve * GIB)),
    )


def budget_child_env(
    env: Mapping[str, str],
    *,
    default_box_budget_gib: float = DEFAULT_BOX_BUDGET_GIB,
) -> dict[str, str]:
    """Stamp David's box budget into a served child's env if it is unset.

    The served daemon's ``apply_expert_profile_child_env`` calls this for the
    DeepSeek-V4.1 profile so the SAME single knob (``MTPLX_DSV41_BOX_BUDGET_GB``)
    reaches the child that the bench scripts derive from -- nobody hand-picks a
    ceiling on either path.  Returns only the keys to ADD (never overwrites an
    operator's explicit value); it is deliberately advisory (it does not stamp
    the load-bearing ``MTPLX_MEMORY_LIMIT_BYTES``), so it cannot conflict with
    the served plan's own memory-cap reconciliation.
    """

    if env.get(ENV_BOX_BUDGET):
        return {}
    return {ENV_BOX_BUDGET: f"{float(default_box_budget_gib):g}"}


# --------------------------------------------------------------------------
# (2) Allocator cache-limit application (the overhang fix).
# --------------------------------------------------------------------------


def apply_allocator_cache_limit(
    cache_limit_bytes: int,
    *,
    mx_module: Any | None = None,
) -> dict[str, Any]:
    """Bound the MLX allocator's freed-buffer cache via ``mx.set_cache_limit``.

    ``mx.set_cache_limit`` returns the *previous* limit.  Returns a report dict
    for the receipt.  MLX-optional: with no ``mx`` importable, reports
    ``applied=False`` rather than raising, so a CPU dry-run stays green.
    """

    cache_limit_bytes = int(cache_limit_bytes)
    if cache_limit_bytes < 0:
        raise ValueError("cache_limit_bytes must be non-negative")
    mx = mx_module
    if mx is None:
        try:
            import mlx.core as mx  # type: ignore
        except Exception as exc:  # pragma: no cover - env guard
            return {
                "applied": False,
                "reason": "mlx_unavailable",
                "error": repr(exc),
                "cache_limit_bytes": cache_limit_bytes,
            }
    setter = getattr(mx, "set_cache_limit", None)
    if not callable(setter):
        setter = getattr(getattr(mx, "metal", None), "set_cache_limit", None)
    if not callable(setter):
        return {
            "applied": False,
            "reason": "set_cache_limit_unavailable",
            "cache_limit_bytes": cache_limit_bytes,
        }
    previous = setter(cache_limit_bytes)
    return {
        "applied": True,
        "cache_limit_bytes": cache_limit_bytes,
        "previous_cache_limit_bytes": int(previous)
        if previous is not None
        else None,
    }


# --------------------------------------------------------------------------
# (1) Instrumentation: MLX / process / box / plan snapshots.
# --------------------------------------------------------------------------


def mlx_memory_snapshot(mx_module: Any | None = None) -> dict[str, Any]:
    """The three MLX allocator accessors: active, cache, peak (bytes)."""

    mx = mx_module
    if mx is None:
        try:
            import mlx.core as mx  # type: ignore
        except Exception as exc:  # pragma: no cover - env guard
            return {"ok": False, "error": repr(exc)}
    out: dict[str, Any] = {"ok": True}
    for public, attr in (
        ("active_bytes", "get_active_memory"),
        ("cache_bytes", "get_cache_memory"),
        ("peak_bytes", "get_peak_memory"),
    ):
        getter = getattr(mx, attr, None)
        if not callable(getter):
            getter = getattr(getattr(mx, "metal", None), attr, None)
        if callable(getter):
            try:
                out[public] = int(getter())
            except Exception as exc:  # pragma: no cover - defensive
                out[public + "_error"] = repr(exc)
    return out


class _TaskVMInfo:
    """Lazily-built ctypes ``task_vm_info`` struct definition (darwin only)."""

    _struct = None

    @classmethod
    def struct(cls):
        if cls._struct is not None:
            return cls._struct
        import ctypes

        class task_vm_info(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("region_count", ctypes.c_int32),
                ("page_size", ctypes.c_int32),
                ("resident_size", ctypes.c_uint64),
                ("resident_size_peak", ctypes.c_uint64),
                ("device", ctypes.c_uint64),
                ("device_peak", ctypes.c_uint64),
                ("internal", ctypes.c_uint64),
                ("internal_peak", ctypes.c_uint64),
                ("external", ctypes.c_uint64),
                ("external_peak", ctypes.c_uint64),
                ("reusable", ctypes.c_uint64),
                ("reusable_peak", ctypes.c_uint64),
                ("purgeable_volatile_pmap", ctypes.c_uint64),
                ("purgeable_volatile_resident", ctypes.c_uint64),
                ("purgeable_volatile_virtual", ctypes.c_uint64),
                ("compressed", ctypes.c_uint64),
                ("compressed_peak", ctypes.c_uint64),
                ("compressed_lifetime", ctypes.c_uint64),
                ("phys_footprint", ctypes.c_uint64),
            ]

        cls._struct = task_vm_info
        return task_vm_info


def _mach_task_vm_info() -> dict[str, int] | None:
    """Current ``resident_size`` / ``phys_footprint`` via ``mach`` task_info.

    ``psutil``-free: reads the kernel's ``TASK_VM_INFO`` for this task directly
    through ctypes.  Returns ``None`` off darwin or on any mach failure.
    """

    if sys.platform != "darwin":
        return None
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        mach_task_self = libc.mach_task_self
        mach_task_self.restype = ctypes.c_uint
        task_info = libc.task_info
        TASK_VM_INFO = 22
        info = _TaskVMInfo.struct()()
        count = ctypes.c_uint32(ctypes.sizeof(info) // 4)
        kr = task_info(
            mach_task_self(),
            TASK_VM_INFO,
            ctypes.byref(info),
            ctypes.byref(count),
        )
        if kr != 0:
            return None
        return {
            "resident_bytes": int(info.resident_size),
            "phys_footprint_bytes": int(info.phys_footprint),
            "compressed_bytes": int(info.compressed),
        }
    except Exception:  # pragma: no cover - defensive
        return None


def process_rss_snapshot() -> dict[str, Any]:
    """Process footprint: current RSS + phys_footprint (mach) and peak maxrss."""

    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = int(maxrss) if sys.platform == "darwin" else int(maxrss) * 1024
    out: dict[str, Any] = {"peak_maxrss_bytes": peak}
    mach = _mach_task_vm_info()
    if mach is not None:
        out.update(mach)
        out["source"] = "mach_task_vm_info"
    else:
        # Fallback: peak maxrss is the only portable figure.
        out["resident_bytes"] = None
        out["phys_footprint_bytes"] = None
        out["source"] = "getrusage_only"
    return out


def _parse_vm_stat(text: str) -> dict[str, int]:
    """Parse ``vm_stat`` output into byte counters."""

    page_size = None
    header = text.splitlines()[0] if text else ""
    # "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
    for token in header.replace("(", " ").replace(")", " ").split():
        if token.isdigit():
            page_size = int(token)
            break
    pages: dict[str, int] = {}
    for line in text.splitlines()[1:]:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip().rstrip(".")
        if not value.isdigit():
            continue
        pages[key.strip().lower()] = int(value)

    required = {"pages free", "pages wired down", "pages active", "pages inactive",
                "pages speculative", "pages occupied by compressor", "anonymous pages",
                "file-backed pages"}
    if not page_size or not required.issubset(pages):
        raise ValueError("incomplete vm_stat snapshot")

    def _bytes(*names: str) -> int:
        for name in names:
            if name in pages:
                return pages[name] * page_size
        return 0

    return {
        "page_size": page_size,
        "free_bytes": _bytes("pages free"),
        "wired_bytes": _bytes("pages wired down"),
        "active_bytes": _bytes("pages active"),
        "inactive_bytes": _bytes("pages inactive"),
        "speculative_bytes": _bytes("pages speculative"),
        "anonymous_bytes": _bytes("anonymous pages"),
        "file_backed_bytes": _bytes("file-backed pages"),
        "compressor_bytes": _bytes("pages occupied by compressor"),
        "compressed_bytes": _bytes("pages stored in compressor"),
        # Matches top's PhysMem used; includes reclaimable file cache. Do not add
        # process footprint or MLX allocations: those pages are already included.
        "used_bytes": sum(_bytes(k) for k in (
            "pages wired down", "pages active", "pages inactive",
            "pages occupied by compressor")),
        # Diagnostic only; this excludes file cache but can also miss unwired
        # driver allocations. It is not a complete whole-machine measurement.
        "non_file_used_bytes": sum(_bytes(k) for k in (
            "pages wired down", "anonymous pages", "pages occupied by compressor")),
        "swapins_pages": pages.get("swapins"),
        "swapouts_pages": pages.get("swapouts"),
    }


def box_memory_snapshot() -> dict[str, Any]:
    """Physical page counters; used includes file cache, compressor is physical."""

    if sys.platform != "darwin":
        return {"ok": False, "reason": "not_darwin"}
    try:
        proc = subprocess.run(
            ["/usr/bin/vm_stat"], capture_output=True, text=True, timeout=2
        )
    except Exception as exc:  # pragma: no cover - env guard
        return {"ok": False, "error": repr(exc)}
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip() or "vm_stat failed"}
    try:
        out = _parse_vm_stat(proc.stdout)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    out["ok"] = True
    out["source"] = "vm_stat"
    out["used_includes_file_cache"] = True
    return out


def host_memory_snapshot() -> dict[str, Any]:
    """OS-only observation shared by health and benchmark sampling, in bytes.

    The start/end timestamps bound sequential kernel reads, not an atomic sample.
    Process footprint includes Metal and must never be added to system used.
    """
    started = time.monotonic_ns()
    process = process_rss_snapshot()
    box = box_memory_snapshot()
    return {"sample_start_monotonic_ns": started,
            "sample_end_monotonic_ns": time.monotonic_ns(),
            "process": process, "box": box}


def plan_breakdown(
    plan: Any,
    *,
    runtime: Any | None = None,
    mx_module: Any | None = None,
    engram_cache_bytes: int | None = None,
) -> dict[str, Any]:
    """The planner's byte breakdown: residents, engram, KV, reserve, cache.

    ``plan`` is an ``ExpertMemoryPlan``.  ``expert_cache_planned_bytes`` is the
    plan's ``persistent_cache_bytes``; when a live ``runtime`` is supplied the
    actual ``allocated_cache_bytes`` (occupied slots x record bytes) comes from
    ``runtime.snapshot()``.  ``engram_bytes`` is the host-side row LRU budget --
    it lives in host overhead, NOT in ``plan.resident_bytes`` -- and is reported
    for completeness (defaults to ``resolve_engram_cache_bytes()``).
    """

    if engram_cache_bytes is None:
        try:
            from .models.deepseek_v41_loader import resolve_engram_cache_bytes

            engram_cache_bytes = int(resolve_engram_cache_bytes())
        except Exception:  # pragma: no cover - defensive
            engram_cache_bytes = None

    out: dict[str, Any] = {
        "total_limit_bytes": int(getattr(plan, "total_limit_bytes", 0)),
        "resident_bytes": int(getattr(plan, "resident_bytes", 0)),
        "engram_host_lru_bytes": engram_cache_bytes,
        "kv_planned_bytes": int(getattr(plan, "kv_bytes", 0)),
        "runtime_reserve_bytes": int(getattr(plan, "runtime_reserve_bytes", 0)),
        "io_staging_bytes": int(getattr(plan, "io_staging_bytes", 0)),
        "transient_bytes": int(getattr(plan, "transient_bytes", 0)),
        "island_bytes": int(getattr(plan, "island_bytes", 0)),
        "fixed_bytes": int(getattr(plan, "fixed_bytes", 0)),
        "expert_cache_planned_bytes": int(
            getattr(plan, "persistent_cache_bytes", 0)
        ),
        "slots_per_layer": int(getattr(plan, "slots_per_layer", 0)),
        "unallocated_bytes": int(getattr(plan, "unallocated_bytes", 0)),
        "fits_fixed": bool(getattr(plan, "fits_fixed", False)),
        "context_tokens": int(getattr(plan, "context_tokens", 0)),
    }
    if runtime is not None:
        try:
            snap = runtime.snapshot(mx_module=mx_module)
            policy = snap.get("expert_cache_policy", {}) if isinstance(snap, dict) else {}
            out["expert_cache_allocated_bytes"] = policy.get("cached_bytes")
            out["expert_cache_allowance_bytes"] = policy.get("allowance_bytes")
        except Exception:  # pragma: no cover - defensive
            out["expert_cache_allocated_bytes"] = None
    return out


def memory_profile_snapshot(
    *,
    phase: str,
    token: int | None = None,
    plan: Any | None = None,
    runtime: Any | None = None,
    mx_module: Any | None = None,
    derivation: BudgetDerivation | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One structured memory snapshot at a named ``phase``.

    Schema (top-level keys)::

        {"phase", "token", "mlx", "process", "box", "plan"?, "derivation"?}

    ``phase`` is the callsite label -- ``"load_end"``, ``"after_prefill"``,
    ``"decode"`` -- and ``token`` the decode-token index for per-N snapshots.
    Every sub-snapshot is best-effort: a missing MLX or a non-darwin box yields
    an ``ok=False`` sub-dict, never an exception.
    """

    snap: dict[str, Any] = {
        **host_memory_snapshot(),
        "phase": str(phase),
        "token": int(token) if token is not None else None,
        "mlx": mlx_memory_snapshot(mx_module),
    }
    if plan is not None:
        snap["plan"] = plan_breakdown(plan, runtime=runtime, mx_module=mx_module)
    if derivation is not None:
        snap["derivation"] = (
            derivation.as_dict()
            if isinstance(derivation, BudgetDerivation)
            else dict(derivation)
        )
    return snap


# --------------------------------------------------------------------------
# Table rendering for a window receipt.
# --------------------------------------------------------------------------


def _gib(value: Any) -> str:
    if value is None:
        return "  --  "
    try:
        return f"{float(value) / GIB:7.2f}"
    except (TypeError, ValueError):
        return "  --  "


def format_memory_profile_table(
    snapshots: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> str:
    """Render one or more snapshots as an aligned GiB table for a receipt."""

    if isinstance(snapshots, Mapping):
        rows = [snapshots]
    else:
        rows = list(snapshots)

    header = (
        f"{'phase':<16}{'tok':>6}  "
        f"{'mlx_act':>8}{'mlx_cache':>10}{'mlx_peak':>9}  "
        f"{'rss':>8}{'phys_fp':>9}  "
        f"{'box_wired':>10}{'box_comp':>9}{'box_free':>9}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        mlx = row.get("mlx", {}) or {}
        proc = row.get("process", {}) or {}
        box = row.get("box", {}) or {}
        token = row.get("token")
        lines.append(
            f"{str(row.get('phase', '?')):<16}"
            f"{('' if token is None else str(token)):>6}  "
            f"{_gib(mlx.get('active_bytes')):>8}"
            f"{_gib(mlx.get('cache_bytes')):>10}"
            f"{_gib(mlx.get('peak_bytes')):>9}  "
            f"{_gib(proc.get('resident_bytes') or proc.get('peak_maxrss_bytes')):>8}"
            f"{_gib(proc.get('phys_footprint_bytes')):>9}  "
            f"{_gib(box.get('wired_bytes')):>10}"
            f"{_gib(box.get('compressor_bytes')):>9}"
            f"{_gib(box.get('free_bytes')):>9}"
        )
    # A derivation line, if any snapshot carries one (all share the same plan).
    for row in rows:
        deriv = row.get("derivation")
        if isinstance(deriv, Mapping) and deriv.get("formula"):
            lines.append("")
            lines.append("derivation: " + str(deriv["formula"]))
            break
    return "\n".join(lines)
