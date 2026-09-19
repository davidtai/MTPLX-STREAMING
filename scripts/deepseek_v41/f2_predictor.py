"""F2 next-layer expert prefetch -- pure predictor, window-stop and plane ring.

CPU-only, no MLX.  Importable in unit tests without touching Metal (MLX defaults
to the Metal device on this box, so the tests either pin it to CPU or, as here,
avoid importing it at all).  The MLX / runtime install glue lives beside the
receipt in ``docs/deepseek-v41/receipts/f2-prefetch-build-20260919/
f2_prefetch_lane.py`` (staged for the GPU window, same style as the
lookahead-io ``plane_lane.py`` install).  That glue imports the ranking,
window-stop, ring-scheduling and budget logic from HERE, so the exact selection
and schedule the GPU lane issues are the ones these CPU tests pin against
``rescore_router_capture.py`` and the published f1 discrete-event numbers.

Provenance of every constant below is a measured receipt, not a guess:

* record / plane bytes            -- extension-bank-20260919, f1-stack-sim-20260919
* compute window c_layer 3.25201ms -- f1-overlap-sim / extension-bank 111-slot run
* SSD rate 12.9 GB/s (window)      -- f1-stack-sim (measured single-drive 12.873)
* predictor = post-attn router, max-over-rows, k=3, READY-resident exclusion
                                   -- f1-real-predictor-20260919 (curve.json)
* schedule = issue after demand submit, plane granularity, window-stop
                                   -- f1-stack-sim-20260919 knob 1 + knob 2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Callable, Iterable, Sequence

import numpy as np

# The shipped record-slot ring: proven round-robin replacement, target-1
# protection, re-eviction embargo and per-victim-layer waste accounting.  Import
# is MLX-free (verified), so the plane scheduler layers plane state on top of the
# shipped ring instead of re-deriving its slot rules.
from mtplx.expert_streaming import GlobalPrefetchRing, SlotLoad

# ---------------------------------------------------------------------------
# Measured constants
# ---------------------------------------------------------------------------
#: One packed mxfp4 expert record on the deepseek-v41-mxfp4-75 profile.  The
#: sidecar record is three weight planes; the plane_lane read offsets prove the
#: byte layout (gate @0, up @6,266,880, down @12,533,760; record end 17,694,720).
EXPERT_RECORD_BYTES = 17_694_720
#: gate / up / down plane bytes (unequal; sum == EXPERT_RECORD_BYTES).  The f1
#: discrete-event sim modelled equal thirds (5,898,240) for its window arithmetic;
#: both are exposed so the receipt can reconcile them.
PLANE_BYTES = (6_266_880, 6_266_880, 5_160_960)
SIM_PLANE_BYTES = EXPERT_RECORD_BYTES // 3  # 5,898,240 -- f1-stack-sim's plane
assert sum(PLANE_BYTES) == EXPERT_RECORD_BYTES
assert SIM_PLANE_BYTES * 3 == EXPERT_RECORD_BYTES

#: Compute window per layer call (non-read compute distributed uniformly), from
#: the 111-slot 198-cycle extension-bank run, inherited unchanged by f1.
COMPUTE_WINDOW_NS = 3_252_010.0  # 3.25201 ms
#: Primary measured drive rate used for the f1 window arithmetic (GB/s -> B/ns).
DEFAULT_SSD_RATE_GB_S = 12.9

#: Default speculative ring, in RECORDS (f1-stack-sim base ring; each record is
#: EXPERT_RECORD_BYTES resident).  Every prior GPU screen used 16; the build
#: default is 32 per the f2 brief.
DEFAULT_RING_RECORDS = 32
#: Whole-machine ceiling the runner's admission plan must never exceed.
MACHINE_CEILING_BYTES = 110_000_000_000

#: Verify predictor geometry (f1-real-predictor).  Targets 4..39 predicted from
#: sources 3..38; 0..3 unpredicted.  Merge over verify rows by MAX; keep top k=3.
FIRST_TARGET_LAYER = 4
DEFAULT_TOP_K = 3
DEFAULT_MERGE = "max"


# ---------------------------------------------------------------------------
# Predictor: ranking / merge / READY-resident exclusion
# ---------------------------------------------------------------------------
def merge_rows(row_scores: np.ndarray, rule: str = DEFAULT_MERGE) -> np.ndarray:
    """(rows, n_experts) -> (n_experts,): merge the next layer's per-row gate
    scores per expert.  ``max`` is the f1-real-predictor winner (max beats sum at
    every budget: sum dilutes an expert one row favours strongly)."""

    a = np.asarray(row_scores)
    if a.ndim != 2:
        raise ValueError(f"row_scores must be (rows, n_experts); got {a.shape}")
    if rule == "max":
        return a.max(axis=0)
    if rule == "sum":
        return a.sum(axis=0)
    raise ValueError(f"unknown merge rule {rule!r}")


def merge_rank_exclude(
    row_scores: np.ndarray,
    ready_mask: Sequence[bool] | np.ndarray,
    k: int = DEFAULT_TOP_K,
    *,
    rule: str = DEFAULT_MERGE,
) -> list[int]:
    """The f2 predictor's ranked top-``k`` ids for ONE target layer call.

    ``row_scores`` -- (rows, n_experts) the NEXT layer's gate applied to the
    current layer's post-attention router input, one score row per verify row.
    ``ready_mask`` -- (n_experts,) True where the expert is a READY resident of
    the TARGET layer at prediction time (persistent + READY transient), excluded.

    Bit-for-bit the ranking ``rescore_router_capture.py`` performs: merge over
    rows, set READY residents to -inf, stable descending argsort, top-``k``, drop
    any non-finite (all-resident) tail.  Returned in ranked (descending) order --
    the SET is what warms the cache; the order only decides which few are issued
    first when the window truncates issuance."""

    merged = merge_rows(row_scores, rule)
    ready = np.asarray(ready_mask, dtype=bool)
    if ready.shape != merged.shape:
        raise ValueError(
            f"ready_mask {ready.shape} must match n_experts {merged.shape}"
        )
    m = merged.astype(np.float64, copy=True)
    m[ready] = -np.inf
    order = np.argsort(-m, kind="stable")[: max(0, int(k))]
    return [int(e) for e in order if isfinite(m[int(e)])]


# ---------------------------------------------------------------------------
# Window-stop arithmetic (constants derived once at construction)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WindowConstants:
    """Per-layer compute window and per-plane read time, all derived once from
    configuration (never measured per call).  ``ssd_rate_gb_s`` is a decimal GB/s
    rate; bytes/ns = rate (since 1 GB/s == 1 B/ns for decimal GB)."""

    compute_window_ns: float = COMPUTE_WINDOW_NS
    ssd_rate_gb_s: float = DEFAULT_SSD_RATE_GB_S
    record_bytes: int = EXPERT_RECORD_BYTES
    plane_bytes: tuple[int, int, int] = PLANE_BYTES

    @property
    def rate_bytes_per_ns(self) -> float:
        return float(self.ssd_rate_gb_s)  # decimal GB/s == B/ns

    def record_read_ns(self) -> float:
        return self.record_bytes / self.rate_bytes_per_ns

    def plane_read_ns(self, plane_index: int) -> float:
        return self.plane_bytes[plane_index] / self.rate_bytes_per_ns

    def max_plane_read_ns(self) -> float:
        return max(self.plane_bytes) / self.rate_bytes_per_ns

    def records_admitted(self) -> float:
        """How many whole records the compute window admits at this rate."""
        return self.compute_window_ns / self.record_read_ns()


def window_stop_blocks(
    *,
    demand_reads_outstanding: int,
    remaining_window_ns: float,
    plane_read_ns: float,
) -> bool:
    """The f1-stack-sim window-stop rule (knob 2, the selected win).

    Do NOT start a speculative plane once the layer's demand reads have ALL
    completed AND the remaining compute window is shorter than one plane read.
    While demand is still in flight the window is not the binding limit -- issue
    continues behind demand priority -- so this returns False.  Refusing to start
    a plane that cannot finish inside the window removes the in-flight demand-delay
    penalty entirely, reaching the ideal-preempt read wait without a preemptible
    drive."""

    if demand_reads_outstanding > 0:
        return False
    return remaining_window_ns < plane_read_ns


# ---------------------------------------------------------------------------
# Budget: the ring is resident, so admission must charge it
# ---------------------------------------------------------------------------
def ring_charge_bytes(ring_records: int = DEFAULT_RING_RECORDS) -> int:
    """Resident bytes the speculative ring reserves: one full record per slot."""
    if ring_records < 0:
        raise ValueError("ring_records must be >= 0")
    return int(ring_records) * EXPERT_RECORD_BYTES


def admits_with_ring(
    *,
    base_launch_bytes: int,
    ring_records: int = DEFAULT_RING_RECORDS,
    machine_ceiling_bytes: int = MACHINE_CEILING_BYTES,
) -> bool:
    """True iff the retained-best launch estimate PLUS the ring charge fits under
    the whole-machine ceiling.  ``base_launch_bytes`` is the extension-bank
    111-slot launch estimate the runner's own admission produces; this only adds
    the ring reserve on top and re-checks the same ceiling."""

    return base_launch_bytes + ring_charge_bytes(ring_records) <= machine_ceiling_bytes


# ---------------------------------------------------------------------------
# Plane-granular speculative scheduler (state machine over the shipped ring)
# ---------------------------------------------------------------------------
_N_PLANES = 3
_STATUS_SPECULATIVE = "speculative"
_STATUS_PROMOTED = "promoted"
_STATUS_COMMITTED = "committed"
_STATUS_FAILED = "failed"


@dataclass
class _PlaneRecord:
    layer: int
    expert: int
    slot: int
    ticket: int | None
    status: str = _STATUS_SPECULATIVE
    planes_read: set[int] = field(default_factory=set)
    planes_inflight: set[int] = field(default_factory=set)

    def remaining_planes(self) -> list[int]:
        done = self.planes_read | self.planes_inflight
        return [p for p in range(_N_PLANES) if p not in done]

    def fully_read(self) -> bool:
        return len(self.planes_read) == _N_PLANES


@dataclass
class SchedulerCounters:
    """AGGREGATE only, collected outside the measured device path (they never
    gate the math and add no per-token/per-dispatch device counter).  The receipt
    reports these once."""

    planes_issued: int = 0
    planes_useful: int = 0     # planes of records a true route later consumed
    planes_wasted: int = 0     # planes charged to records evicted unconsumed (<=3 each)
    records_promoted: int = 0  # speculative records a demand route took over
    records_committed: int = 0
    records_failed: int = 0
    records_wasted: int = 0    # committed ring records recycled without consumption

    def as_dict(self) -> dict[str, int]:
        return {
            "planes_issued": self.planes_issued,
            "planes_useful": self.planes_useful,
            "planes_wasted": self.planes_wasted,
            "records_promoted": self.records_promoted,
            "records_committed": self.records_committed,
            "records_failed": self.records_failed,
            "records_wasted": self.records_wasted,
        }


class SpeculativePlaneScheduler:
    """Plane-granular speculative reads over the shipped record-slot ring.

    Record-slot assignment (which physical ring slot a predicted record takes,
    round-robin with target-1 protection, re-eviction embargo and per-victim
    waste accounting) is delegated UNCHANGED to :class:`GlobalPrefetchRing`.  This
    class adds the f1-stack-sim plane behaviour on top:

    * a predicted record's three planes (gate/up/down) are issued one at a time,
      behind demand priority, each gated by :func:`window_stop_blocks`;
    * a speculative record a demand route later needs is PROMOTED -- only its
      unread planes are read on the demand path, the already-read planes are
      reused (:meth:`promote`);
    * a plane read that fails DRAINS the record: its ring assignment is
      invalidated and no further plane is issued; the runtime is never marked
      unhealthy (there is no health flag to flip here -- speculation is
      best-effort by construction);
    * waste (a committed-but-never-consumed record recycled round-robin) is
      charged to the victim's own layer via the ring and drained here.

    The scheduler holds NO lock and performs NO I/O: the caller injects an
    ``issue_plane(layer, expert, plane) -> handle`` callable so the CPU tests can
    drive it with a fake reader and the GPU lane can drive it with the runtime's
    demand-priority reader queue."""

    def __init__(
        self,
        ring: GlobalPrefetchRing,
        *,
        window: WindowConstants | None = None,
        top_k: int = DEFAULT_TOP_K,
        issue_plane: Callable[[int, int, int], object] | None = None,
    ) -> None:
        self.ring = ring
        self.window = window or WindowConstants()
        self.top_k = int(top_k)
        self._issue_plane = issue_plane
        self.counters = SchedulerCounters()
        # (layer, expert) -> _PlaneRecord for every live speculative/promoted rec.
        self._records: dict[tuple[int, int], _PlaneRecord] = {}

    # -- issue --------------------------------------------------------------
    def issue_predictions(
        self,
        layer: int,
        ranked_ids: Sequence[int],
        *,
        resident: Iterable[int] = (),
        remaining_window_ns: float,
        demand_reads_outstanding: int = 0,
        is_slot_pinned: Callable[[int], bool] | None = None,
    ) -> int:
        """Assign ring slots for ``layer``'s ranked predicted experts and issue
        their planes, in rank order, plane-by-plane, honouring window-stop.

        Returns the number of PLANES issued.  A prediction already resident,
        committed or inflight is skipped by the ring.  Issuance stops at the first
        plane window-stop refuses (rank order means the farthest/lowest-value
        record is the one dropped)."""

        loads: tuple[SlotLoad, ...] = self.ring.plan_prefetch(
            layer, ranked_ids, resident=resident, is_slot_pinned=is_slot_pinned
        )
        self._drain_ring_waste()
        issued_planes = 0
        # The window is consumed as speculative planes are issued into it, so a
        # running remainder tracks how much compute window is left for the NEXT
        # plane (the constants themselves are construction-time; only the elapsed
        # is runtime).  While a demand read is still outstanding window-stop is
        # inert (issue behind demand priority), so the remainder is not charged.
        remaining = float(remaining_window_ns)
        stopped = False
        for load in loads:
            if stopped:
                break
            key = (int(layer), int(load.expert))
            ticket = self.ring.prefetch_ticket(*key)
            rec = _PlaneRecord(
                layer=int(layer), expert=int(load.expert), slot=load.slot,
                ticket=ticket,
            )
            self._records[key] = rec
            for plane in range(_N_PLANES):
                plane_ns = self.window.plane_read_ns(plane)
                if window_stop_blocks(
                    demand_reads_outstanding=demand_reads_outstanding,
                    remaining_window_ns=remaining,
                    plane_read_ns=plane_ns,
                ):
                    stopped = True
                    break
                rec.planes_inflight.add(plane)
                if self._issue_plane is not None:
                    self._issue_plane(rec.layer, rec.expert, plane)
                self.counters.planes_issued += 1
                issued_planes += 1
                if demand_reads_outstanding <= 0:
                    remaining -= plane_ns
        return issued_planes

    # -- completion / failure ----------------------------------------------
    def on_plane_complete(
        self, layer: int, expert: int, plane: int, *, ok: bool = True
    ) -> None:
        """Record a settled plane read.  On failure, DRAIN the record."""

        key = (int(layer), int(expert))
        rec = self._records.get(key)
        if rec is None:
            return
        rec.planes_inflight.discard(int(plane))
        if not ok:
            # Failure drain: invalidate the ring assignment and forget the record.
            # No health flag flips -- speculation never marks the runtime unhealthy.
            self.ring.invalidate_prefetch(rec.layer, rec.expert, ticket=rec.ticket)
            rec.status = _STATUS_FAILED
            self.counters.records_failed += 1
            self._records.pop(key, None)
            return
        rec.planes_read.add(int(plane))
        if rec.fully_read() and rec.status == _STATUS_SPECULATIVE:
            if self.ring.commit_prefetch(rec.layer, rec.expert, ticket=rec.ticket):
                rec.status = _STATUS_COMMITTED
                self.counters.records_committed += 1

    # -- promotion ----------------------------------------------------------
    def promote(self, layer: int, expert: int) -> list[int]:
        """A demand route needs ``(layer, expert)``.  Promote the speculative
        record so only its UNREAD planes are read on the demand path; the
        already-read (and in-flight) planes are reused.  Returns the plane indices
        the demand path still has to read (empty if the record is already fully
        read).  Returns ``[0, 1, 2]`` when there is no speculative record (a true
        cold miss -- the demand path reads the whole record)."""

        key = (int(layer), int(expert))
        rec = self._records.get(key)
        if rec is None:
            return list(range(_N_PLANES))
        if rec.status == _STATUS_SPECULATIVE:
            rec.status = _STATUS_PROMOTED
            self.counters.records_promoted += 1
        return rec.remaining_planes()

    def note_consumed(self, layer: int, experts: Iterable[int]) -> None:
        """A true route consumed these committed ring records (first consumption):
        mark them used so a later round-robin eviction is not miscounted as
        wasted, and credit their read planes as useful."""

        layer = int(layer)
        experts = [int(e) for e in experts]
        self.ring.mark_used(layer, experts)
        for e in experts:
            rec = self._records.get((layer, e))
            if rec is not None:
                self.counters.planes_useful += len(rec.planes_read)

    def note_decode(self, layer: int) -> None:
        """Advance the ring's per-layer decode epoch (drives its embargo)."""
        self.ring.note_decode(int(layer))

    # -- internals ----------------------------------------------------------
    def _drain_ring_waste(self) -> None:
        """Fold the ring's per-victim-layer waste (committed records recycled
        round-robin without a true route consuming them) into the aggregate
        counters.  The ring charges each wasted RECORD to its own predicting
        layer; we do not retain an evicted record's plane count, so charge an
        upper bound of ``_N_PLANES`` planes per wasted record (the receipt reports
        both the record count and this plane upper bound)."""

        wasted = self.ring.consume_wasted_by_layer()
        wasted_records = sum(int(c) for c in wasted.values())
        self.counters.records_wasted += wasted_records
        self.counters.planes_wasted += wasted_records * _N_PLANES


__all__ = [
    "EXPERT_RECORD_BYTES",
    "PLANE_BYTES",
    "SIM_PLANE_BYTES",
    "COMPUTE_WINDOW_NS",
    "DEFAULT_SSD_RATE_GB_S",
    "DEFAULT_RING_RECORDS",
    "MACHINE_CEILING_BYTES",
    "FIRST_TARGET_LAYER",
    "DEFAULT_TOP_K",
    "merge_rows",
    "merge_rank_exclude",
    "WindowConstants",
    "window_stop_blocks",
    "ring_charge_bytes",
    "admits_with_ring",
    "SchedulerCounters",
    "SpeculativePlaneScheduler",
]
