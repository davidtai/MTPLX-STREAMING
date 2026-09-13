"""Policy primitives for SSD-backed routed-expert slot banks.

This module deliberately has no MLX dependency.  It owns the cache decision
that sits between a resident MoE router and a future native Metal slot-bank
loader.  Keeping the policy pure makes route traces reproducible and lets us
size a cache before allocating multi-gigabyte Metal buffers.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from operator import index
from typing import Callable, Iterable


def _integer(name: str, value: object, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        normalized = index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


class RoutingPhase(str, Enum):
    """Inference phase used to select the admission policy."""

    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class SlotLoad:
    """One expert record that must be loaded before dispatch."""

    expert: int
    slot: int
    persistent: bool
    generation: int | None = None


@dataclass(frozen=True)
class SlotEviction:
    """A persistent slot reassigned to a hotter expert."""

    slot: int
    previous_expert: int
    next_expert: int
    previous_layer: int | None = None
    next_layer: int | None = None


@dataclass(frozen=True)
class RoutePlan:
    """Resolved slot mapping for one layer invocation.

    ``slots`` preserves the router's input order.  A native executor can use
    it in place of the original expert ids after all ``loads`` have completed.
    """

    phase: RoutingPhase
    experts: tuple[int, ...]
    slots: tuple[int, ...]
    hits: tuple[int, ...]
    misses: tuple[int, ...]
    loads: tuple[SlotLoad, ...]
    evictions: tuple[SlotEviction, ...]
    generations: tuple[int | None, ...] = ()
    # W87 single-pool admission telemetry (0 on the two-tier path).
    pool_loads: int = 0
    scan_inserts: int = 0
    promotions: int = 0
    # W93: the subset of ``hits`` that resolved from the speculative prefetch
    # ring (a gate-oracle prediction that paid off), so the counters can price
    # ``hit_on_true_route`` distinctly from the persistent-tier hits it is folded
    # into. Empty on every non-ring path (byte-identical to the pre-W93 plan).
    prefetch_hits: tuple[int, ...] = ()
    # W95g (review MEDIUM-2): the subset of ``prefetch_hits`` this route consumes
    # for the FIRST time -- ring entries not yet marked used since they were
    # committed. ``prefetch_hits`` re-lists a resident ring record on every token
    # that routes it (so ``prefetch_hit_on_true_route`` counts every consumption
    # and can exceed the number of records committed/issued); ``prefetch_first_hits``
    # counts each prefetched record's hit AT MOST ONCE, giving the bounded [0,1]
    # ``prefetch_first_hit_rate = first-consumption hits / prefetch_issued``. Empty
    # on every non-ring path.
    prefetch_first_hits: tuple[int, ...] = ()


class RoutePolicyTxn:
    """Idempotent policy publication or completion-failure rollback."""

    def __init__(
        self,
        *,
        commit: Callable[[], None] | None = None,
        rollback: Callable[[], None],
    ) -> None:
        self._commit = commit
        self._rollback = rollback
        self._finished = False
        self._rolled_back = False

    def commit(self) -> None:
        if self._finished:
            return
        if self._commit is not None:
            self._commit()
        self._finished = True

    def rollback_completion(self) -> None:
        if self._finished:
            return
        self._rollback()
        self._finished = True
        self._rolled_back = True

    def rollback_publication(self) -> None:
        """Undo a publication that committed before a later host-side failure."""

        if self._rolled_back:
            return
        self._rollback()
        self._finished = True
        self._rolled_back = True


@dataclass
class CacheCounters:
    """Aggregate counters suitable for CLI/server metrics."""

    route_calls: int = 0
    expert_requests: int = 0
    unique_expert_requests: int = 0
    shared_expert_assignments: int = 0
    expert_hits: int = 0
    expert_misses: int = 0
    persistent_loads: int = 0
    transient_loads: int = 0
    evictions: int = 0
    bytes_read: int = 0
    prefetch_issued: int = 0
    prefetch_committed: int = 0
    # W100: the DSpark-verify slice of the gate-oracle prefetch. ``prefetch_issued``
    # / ``prefetch_committed`` above merge the AR (M=1) and verify (M=K+1) phases,
    # so a window cannot tell whether the multi-row verify actually engaged the
    # prefetch. These count ONLY the verify-phase (RoutingPhase.DECODE, 2..8-row)
    # issues/commits, threaded from the switch's issue site via
    # ``prefetch_experts(..., verify=True)`` -- so verify engagement is provable
    # independent of the AR total. Never gate the math; scheduling telemetry only.
    prefetch_issued_verify: int = 0
    prefetch_committed_verify: int = 0
    # W87 single-pool (0 on the two-tier path): pool_loads = misses admitted into
    # the merged resident pool; scan_inserts = of those, the prefill/scan-resistant
    # inserts; promotions = probationary->protected transitions on a later hit.
    pool_loads: int = 0
    scan_inserts: int = 0
    promotions: int = 0
    # W93 gate-oracle prefetch telemetry. ``predicted`` (ids handed to the ring),
    # ``hit_on_true_route`` (commits a true route consumed as a hit), ``wasted``
    # (commits evicted round-robin without ever being hit), ``awaited_inflight``
    # (true-route experts awaited in flight instead of re-read) and ``bytes``
    # (issued * record) are incremented by the runtime; ``hit_on_true_route`` also
    # accrues through :meth:`observe` from ``RoutePlan.prefetch_hits``.
    prefetch_predicted: int = 0
    prefetch_hit_on_true_route: int = 0
    prefetch_wasted: int = 0
    prefetch_awaited_inflight: int = 0
    prefetch_bytes: int = 0
    # W95g (review MEDIUM-2): first-consumption prefetch hits. ``hit_on_true_route``
    # accrues ``len(plan.prefetch_hits)`` on EVERY route, so a resident ring record
    # re-consumed across N tokens is counted N times and ``hit_on_true_route`` can
    # exceed the records committed/issued (an unbounded rate). This accrues
    # ``len(plan.prefetch_first_hits)`` -- each prefetched record's hit at most once
    # -- and is the numerator of the bounded ``prefetch_first_hit_rate`` (over
    # ``prefetch_issued``).
    prefetch_first_consumption_hits: int = 0

    def observe(self, plan: RoutePlan, *, expert_record_bytes: int) -> None:
        expert_record_bytes = _integer(
            "expert_record_bytes", expert_record_bytes, minimum=0
        )
        hit_experts = set(plan.hits)
        unique_request_count = len(set(plan.experts))
        assignment_hits = sum(expert in hit_experts for expert in plan.experts)
        self.route_calls += 1
        self.expert_requests += len(plan.experts)
        self.unique_expert_requests += unique_request_count
        self.shared_expert_assignments += len(plan.experts) - unique_request_count
        self.expert_hits += assignment_hits
        self.expert_misses += len(plan.experts) - assignment_hits
        self.persistent_loads += sum(load.persistent for load in plan.loads)
        self.transient_loads += sum(not load.persistent for load in plan.loads)
        self.evictions += len(plan.evictions)
        self.bytes_read += len(plan.loads) * expert_record_bytes
        self.pool_loads += plan.pool_loads
        self.scan_inserts += plan.scan_inserts
        self.promotions += plan.promotions
        # W93: a prefetch-ring commit consumed by this true route (unique experts).
        # Every consumption counts (can exceed committed/issued -- see the field doc).
        self.prefetch_hit_on_true_route += len(plan.prefetch_hits)
        # W95g (review MEDIUM-2): only FIRST consumptions -- each prefetched record's
        # hit at most once -- so first-consumption hits / prefetch_issued is <= 1.
        self.prefetch_first_consumption_hits += len(plan.prefetch_first_hits)

    @property
    def hit_rate(self) -> float:
        total = self.expert_hits + self.expert_misses
        return self.expert_hits / total if total else 0.0

    def as_dict(self) -> dict[str, int | float]:
        return {
            "route_calls": self.route_calls,
            "expert_requests": self.expert_requests,
            "unique_expert_requests": self.unique_expert_requests,
            "shared_expert_assignments": self.shared_expert_assignments,
            "expert_hits": self.expert_hits,
            "expert_misses": self.expert_misses,
            "hit_rate": self.hit_rate,
            "persistent_loads": self.persistent_loads,
            "transient_loads": self.transient_loads,
            "evictions": self.evictions,
            "bytes_read": self.bytes_read,
            "prefetch_issued": self.prefetch_issued,
            "prefetch_committed": self.prefetch_committed,
            "prefetch_issued_verify": self.prefetch_issued_verify,
            "prefetch_committed_verify": self.prefetch_committed_verify,
            "pool_loads": self.pool_loads,
            "scan_inserts": self.scan_inserts,
            "promotions": self.promotions,
            "prefetch_predicted": self.prefetch_predicted,
            "prefetch_hit_on_true_route": self.prefetch_hit_on_true_route,
            "prefetch_wasted": self.prefetch_wasted,
            "prefetch_awaited_inflight": self.prefetch_awaited_inflight,
            "prefetch_bytes": self.prefetch_bytes,
            "prefetch_first_consumption_hits": self.prefetch_first_consumption_hits,
        }


@dataclass
class _ExpertHistory:
    score: float = 0.0
    score_epoch: int = 0
    last_used: int = -1


@dataclass
class _GlobalDirectoryEntry:
    slot: int
    generation: int
    state: str
    lru_rank: int


class GlobalPrefetchRing:
    """One speculative lookahead ring SHARED across all routed layers (W93).

    In decode the layers run sequentially and each layer is prefetched exactly
    one step ahead, so a ring of ``ring_size`` slots (sized 2*k for a width-k
    predictor -- layer L's committed hits keep resolving while layer L+1's
    fill) bounds the resident ring to ``ring_size`` records TOTAL, a small fixed
    reserve like the transient pool, rather than ``ring_size * n_layers`` carved
    out of the persistent LRU (docs/deepseek-v41/W93_GATE_PREFETCH.md §4).

    Entries are keyed by ``(layer, expert)``; the physical slot handed to the
    pool is ``base + ring_index`` where ``base = persistent_slots +
    transient_slots`` (uniform across layers, so a ring slot resolves to the
    shared ``ExpertSlotPool._prefetch[ring_index]`` regardless of layer). The
    round-robin replacement, inflight-safety, ticketing and reeviction embargo
    are the per-layer ring's, generalised: the embargo is measured in each
    layer's OWN decode epoch (advanced by :meth:`note_decode` from the bank's
    ``plan``), so a shared ring reproduces the per-layer churn guard exactly.

    Every call is made under the caller's per-layer transaction lock (the
    runtime serialises a layer's prefetch planning and route resolution), so the
    ring holds no lock of its own. A ``ticket`` binds a completion to one exact
    assignment and, like the per-layer ring, SURVIVES :meth:`reset` so a
    callback straddling a reset can never collide with a fresh assignment.
    """

    reeviction_window = 2

    def __init__(self, *, ring_size: int, base: int, expert_count: int) -> None:
        self.ring_size = _integer("ring_size", ring_size, minimum=0)
        self.base = _integer("base", base, minimum=0)
        self.expert_count = _integer("expert_count", expert_count, minimum=1)
        self._slot_to_key: list[tuple[int, int] | None] = [None] * self.ring_size
        self._key_to_slot: dict[tuple[int, int], int] = {}
        self._inflight: dict[tuple[int, int], tuple[int, int]] = {}
        self._cursor = 0
        self._ticket = 0
        self._evicted: dict[tuple[int, int], int] = {}
        self._used: set[tuple[int, int]] = set()
        # W93 MED-b: wasted (committed-but-never-consumed) reads are counted per
        # the VICTIM's layer (the layer that predicted the evicted entry), not
        # the layer whose ``plan_prefetch`` call happened to evict it. Drained by
        # the runtime after each plan under the layer lock.
        self._wasted: dict[int, int] = {}
        self._epoch_by_layer: dict[int, int] = {}
        # W93 HIGH-2 receipt counters, keyed by the TARGET layer (the layer the
        # prediction was for). Cumulative until :meth:`reset`; read (not drained)
        # by the runtime's ``gate_prefetch`` snapshot so a self-starving or
        # pressured ring is visible instead of silently dropping predictions.
        #   dropped_no_slot   -- a prediction found no free ring slot (every slot
        #                        inflight, target-1 protected, or pinned);
        #   skipped_lock_held -- the runtime skipped a prefetch because the layer
        #                        lock was held (~3827 deferred-split hazard);
        #   skipped_backlog   -- the runtime skipped because the read backlog was
        #                        already at its ceiling.
        self._dropped_no_slot: dict[int, int] = {}
        self._skipped_lock_held: dict[int, int] = {}
        self._skipped_backlog: dict[int, int] = {}

    def _key(self, layer: int, expert: int) -> tuple[int, int]:
        expert = _integer("expert id", expert, minimum=0)
        if expert >= self.expert_count:
            raise ValueError(f"expert id {expert} is outside [0, {self.expert_count})")
        return (int(layer), expert)

    def note_decode(self, layer: int) -> None:
        """Advance ``layer``'s decode epoch (drives that layer's reeviction
        embargo). The owning bank calls this once per DECODE ``plan``."""

        layer = int(layer)
        self._epoch_by_layer[layer] = self._epoch_by_layer.get(layer, 0) + 1

    def plan_prefetch(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        resident: Iterable[int] = (),
        is_slot_pinned: Callable[[int], bool] | None = None,
    ) -> tuple[SlotLoad, ...]:
        """Assign ring slots for ``layer``'s predicted experts; return the loads.

        Skips experts already resident (persistent) for this layer, already
        committed or inflight in the ring, or under the reeviction embargo.
        Ring replacement is round-robin over the WHOLE ring (across layers). A
        candidate slot is skipped as a victim when it is:

        * inflight -- two fills on one buffer would race;
        * held by layer ``layer - 1`` -- the immediately-next layer's prediction,
          whose OWN true route has not run yet and is about to consume it. We
          prefetch for ``layer`` during layer ``layer - 1``'s decode forward, so
          recycling ``layer - 1``'s entries here would drop a read the very next
          route needs (the lane-C ring rule);
        * pinned -- when the caller passes ``is_slot_pinned``, a slot it reports
          pinned (an earned/committed slot a demand route has claimed) is never a
          ring victim.

        A committed victim recycled without ever being consumed by a true route
        is priced as a wasted read against the VICTIM's layer. A prediction that
        finds no eligible slot is counted as ``dropped_no_slot`` for ``layer``."""

        if not self.ring_size:
            return ()
        layer = int(layer)
        resident_set = {int(e) for e in resident}
        epoch = self._epoch_by_layer.get(layer, 0)
        loads: list[SlotLoad] = []
        for expert in dict.fromkeys(int(e) for e in expert_ids):
            key = self._key(layer, expert)
            if (
                expert in resident_set
                or key in self._key_to_slot
                or key in self._inflight
            ):
                continue
            evicted_epoch = self._evicted.get(key)
            if evicted_epoch is not None:
                if epoch - evicted_epoch < self.reeviction_window:
                    continue
                del self._evicted[key]
            ring_index: int | None = None
            # Exactly ``ring_size`` probes advance the cursor across every slot
            # once (round-robin), so if any eligible slot exists it is found.
            for _probe in range(self.ring_size):
                candidate = self._cursor % self.ring_size
                self._cursor += 1
                tenant = self._slot_to_key[candidate]
                if tenant is not None:
                    if tenant in self._inflight:
                        continue
                    if tenant[0] == layer - 1:
                        # target-1 protection: never evict the imminent-route
                        # layer's prediction (lane-C ring rule).
                        continue
                    if is_slot_pinned is not None and is_slot_pinned(
                        self.base + candidate
                    ):
                        continue
                ring_index = candidate
                break
            if ring_index is None:
                # Every slot is inflight, target-1 protected, or pinned: the
                # prediction is dropped for want of a slot.
                self._dropped_no_slot[layer] = self._dropped_no_slot.get(layer, 0) + 1
                continue
            victim = self._slot_to_key[ring_index]
            if victim is not None:
                was_committed = self._key_to_slot.pop(victim, None) is not None
                if was_committed and victim not in self._used:
                    # attribute the wasted read to the victim's OWN layer.
                    self._wasted[victim[0]] = self._wasted.get(victim[0], 0) + 1
                self._used.discard(victim)
                # embargo the victim in ITS layer's epoch scale.
                self._evicted[victim] = self._epoch_by_layer.get(victim[0], 0)
            self._slot_to_key[ring_index] = key
            ticket = self._ticket
            self._ticket += 1
            self._inflight[key] = (self.base + ring_index, ticket)
            loads.append(
                SlotLoad(expert=expert, slot=self.base + ring_index, persistent=False)
            )
        return tuple(loads)

    def prefetch_ticket(self, layer: int, expert: int) -> int | None:
        entry = self._inflight.get(self._key(layer, expert))
        return None if entry is None else entry[1]

    def commit_prefetch(
        self, layer: int, expert: int, *, ticket: int | None = None
    ) -> bool:
        key = self._key(layer, expert)
        entry = self._inflight.get(key)
        if entry is None:
            return False
        slot, assignment_ticket = entry
        if ticket is not None and ticket != assignment_ticket:
            return False
        del self._inflight[key]
        if self._slot_to_key[slot - self.base] != key:
            return False
        self._key_to_slot[key] = slot
        return True

    def invalidate_prefetch(
        self, layer: int, expert: int, *, ticket: int | None = None
    ) -> int | None:
        key = self._key(layer, expert)
        if ticket is not None:
            entry = self._inflight.get(key)
            if entry is None or entry[1] != ticket:
                return None
            del self._inflight[key]
            slot = entry[0]
            if self._slot_to_key[slot - self.base] == key:
                self._slot_to_key[slot - self.base] = None
            self._used.discard(key)
            return slot
        slot = self._key_to_slot.pop(key, None)
        if slot is None:
            entry = self._inflight.pop(key, None)
            slot = None if entry is None else entry[0]
        if slot is not None:
            if self._slot_to_key[slot - self.base] == key:
                self._slot_to_key[slot - self.base] = None
        self._used.discard(key)
        return slot

    def published(self, layer: int, expert_ids: Iterable[int]) -> dict[int, int]:
        """{expert: ring slot} for this layer's committed (hit-eligible) entries."""

        layer = int(layer)
        return {
            int(expert): self._key_to_slot[(layer, int(expert))]
            for expert in expert_ids
            if (layer, int(expert)) in self._key_to_slot
        }

    def mark_used(self, layer: int, experts: Iterable[int]) -> None:
        """Record that a true route consumed these committed ring entries, so a
        later round-robin eviction of them is not miscounted as wasted."""

        layer = int(layer)
        for expert in experts:
            self._used.add((layer, int(expert)))

    def first_consumption(self, layer: int, experts: Iterable[int]) -> frozenset[int]:
        """W95g (review MEDIUM-2): of ``experts`` (already published/hit-eligible
        for ``layer``), those this layer has NOT yet consumed as a hit since they
        were committed -- i.e. not in ``_used``. Non-mutating; call :meth:`mark_used`
        to record the consumption. A committed entry stays published across tokens,
        so :meth:`published` re-lists it every routing step; this bounds the hit to
        the FIRST such step. ``_used`` is cleared for a key only on eviction /
        invalidation (which drop it from ``_key_to_slot`` too), so a record must be
        re-committed -- and thus re-issued -- to be first-consumed again."""

        layer = int(layer)
        return frozenset(
            int(expert)
            for expert in experts
            if (layer, int(expert)) not in self._used
        )

    def consume_wasted(self) -> int:
        """Return and zero the TOTAL wasted-read count across all layers.

        Kept for the per-layer-bank wrapper and standalone unit tests; the
        runtime drains the per-layer breakdown via
        :meth:`consume_wasted_by_layer` to attribute each waste to its victim
        layer."""

        count = sum(self._wasted.values())
        self._wasted = {}
        return count

    def consume_wasted_by_layer(self) -> dict[int, int]:
        """Return and zero the wasted-read count keyed by the VICTIM's layer.

        A wasted read is a committed ring entry recycled round-robin without a
        true route ever consuming it; it is charged to the layer that predicted
        it (the victim), not to the layer whose ``plan_prefetch`` evicted it."""

        drained = self._wasted
        self._wasted = {}
        return drained

    def note_skipped_lock_held(self, layer: int) -> None:
        """Record that a prefetch for ``layer`` was skipped (layer lock held)."""

        layer = int(layer)
        self._skipped_lock_held[layer] = self._skipped_lock_held.get(layer, 0) + 1

    def note_skipped_backlog(self, layer: int) -> None:
        """Record that a prefetch for ``layer`` was skipped (read backlog full)."""

        layer = int(layer)
        self._skipped_backlog[layer] = self._skipped_backlog.get(layer, 0) + 1

    def prefetch_skip_snapshot(self) -> dict[str, dict[int, int]]:
        """A read-only, cumulative snapshot of the three skip/drop counters.

        Copies so a concurrent telemetry read cannot observe a dict mutating.
        Not drained -- these accrue until :meth:`reset` (like the runtime's
        cumulative cache counters)."""

        return {
            "dropped_no_slot": dict(self._dropped_no_slot),
            "skipped_lock_held": dict(self._skipped_lock_held),
            "skipped_backlog": dict(self._skipped_backlog),
        }

    def reset(self) -> None:
        self._slot_to_key = [None] * self.ring_size
        self._key_to_slot.clear()
        self._inflight.clear()
        self._cursor = 0
        self._evicted.clear()
        self._used.clear()
        self._wasted = {}
        self._epoch_by_layer.clear()
        self._dropped_no_slot.clear()
        self._skipped_lock_held.clear()
        self._skipped_backlog.clear()
        # ``_ticket`` is deliberately NOT reset (see class docstring).


class LayerExpertSlotBank:
    """Per-layer TinyLFU-like hot bank with transient service slots.

    The persistent tier learns only from decode routes.  Prefill misses use the
    transient tier and therefore cannot evict a useful decode hot set.  Decode
    misses are admitted only once their decayed frequency beats the coldest
    unpinned resident; otherwise they are served through a transient slot.

    This class plans I/O but never performs it.  The future native layer owns
    fixed Metal buffers and applies the returned ``SlotLoad`` operations using
    aligned ``pread``.

    W87: with ``single_pool=True`` (env ``MTPLX_DSV41_SINGLE_SLOT_POOL``) the two
    tiers collapse into ONE resident pool.  Every miss -- prefill or decode -- is
    admitted into the pool with 2Q/segmented-LRU scan-resistant insertion (new
    entries probationary, promoted to protected on a later hit, victims taken from
    the probationary segment first, pins never victims), so a long prefill scan
    flows through without evicting the earned set and the prompt's tail is left
    resident for decode.  A slot bank is a pure cache, so this changes only which
    loads happen, never a route's computed output.
    """

    # Decode epochs a ring-evicted expert stays unpredictable. Ring
    # turnover at high miss volume otherwise evicts and re-reads the same
    # hot experts every couple of tokens, multiplying SSD traffic.
    prefetch_reeviction_window = 2

    def __init__(
        self,
        *,
        expert_count: int,
        persistent_slots: int,
        transient_slots: int,
        frequency_decay: float = 0.995,
        cache_policy: str = "frequency",
        prefetch_slots: int = 0,
        single_pool: bool = False,
        layer_id: int = 0,
        prefetch_ring: "GlobalPrefetchRing | None" = None,
    ) -> None:
        expert_count = _integer("expert_count", expert_count, minimum=1)
        persistent_slots = _integer("persistent_slots", persistent_slots, minimum=0)
        transient_slots = _integer("transient_slots", transient_slots, minimum=1)
        prefetch_slots = _integer("prefetch_slots", prefetch_slots, minimum=0)
        if persistent_slots > expert_count:
            raise ValueError("persistent_slots cannot exceed expert_count")
        if isinstance(frequency_decay, bool):
            raise TypeError("frequency_decay must be a finite number")
        try:
            frequency_decay = float(frequency_decay)
        except (TypeError, ValueError) as exc:
            raise TypeError("frequency_decay must be a finite number") from exc
        if not isfinite(frequency_decay) or not 0.0 < frequency_decay <= 1.0:
            raise ValueError("frequency_decay must be in (0, 1]")

        self.expert_count = expert_count
        self.persistent_slots = persistent_slots
        self.transient_slots = transient_slots
        self.prefetch_slots = prefetch_slots
        self.slot_count = persistent_slots + transient_slots + prefetch_slots
        self.frequency_decay = frequency_decay
        if cache_policy not in {"frequency", "lru"}:
            raise ValueError("cache_policy must be 'frequency' or 'lru'")
        self.cache_policy = cache_policy

        # Prefill traffic must neither age nor refresh decode admission state.
        # A separate decode-only epoch keeps a long prompt from erasing the
        # hot set immediately before generation starts.
        self._decode_epoch = 0
        self._slot_to_expert: list[int | None] = [None] * self.persistent_slots
        self._expert_to_slot: dict[int, int] = {}
        self._history = [_ExpertHistory() for _ in range(expert_count)]
        self._prefill_seed_candidates: set[int] = set()
        self._persistent_capacity = self.persistent_slots
        # W93: the speculative lookahead ring is now a SHARED GlobalPrefetchRing
        # (keyed by (layer, expert)), so its resident cost is one small fixed
        # reserve across all layers rather than a per-layer ring times the layer
        # count. The runtime passes ONE ring to every bank; a bank constructed
        # standalone with prefetch_slots>0 (unit tests) owns a single-ring of that
        # width, behaving exactly like the old per-layer ring. The ring is inert
        # (None) when prefetch is disabled -- every path below is then byte-
        # identical to the pre-ring code.
        self._layer_id = int(layer_id)
        if prefetch_slots > 0:
            self._prefetch_ring = prefetch_ring or GlobalPrefetchRing(
                ring_size=prefetch_slots,
                base=self.persistent_slots + self.transient_slots,
                expert_count=self.expert_count,
            )
            self._owns_prefetch_ring = prefetch_ring is None
        else:
            self._prefetch_ring = None
            self._owns_prefetch_ring = False
        # W64 (R3-pin): a post-prefill PINNED WORKING SET. Experts in
        # ``_pinned`` are never chosen as a *decode-admission* eviction victim
        # (``_victim_slot`` with ``respect_pins``), so a pinned expert's
        # persistent slot is never recycled in place. That is the exact safety
        # property the W44 barrier-free device route needs: its deferred,
        # unpinned gather cannot race an LRU slot recycle for a pinned expert
        # (W44_DEVICE_ROUTE.md §8). Empty until ``pin_working_set`` is called;
        # while empty every path below is byte-identical to the pre-pin code
        # (``_pinned`` widens no blocked set and touches no output). A memory-
        # forced capacity eviction may still evict a pinned expert
        # (``respect_pins=False``) -- memory is the hard constraint -- and
        # ``invalidate_expert`` then drops it from the pinned set.
        self._pinned: set[int] = set()
        # Per-expert prefill routing frequency, accumulated in
        # ``prepare_prefill_seed`` from the prompt's routed ids so
        # ``pin_working_set`` can rank the resident set by prompt-frequent
        # experts with no caller-supplied count. Decode ``_score`` breaks ties.
        self._prefill_route_freq: Counter[int] = Counter()
        # W87 single-pool (env MTPLX_DSV41_SINGLE_SLOT_POOL): merge the persistent
        # + transient tiers into ONE per-layer resident pool with 2Q/segmented-LRU
        # scan-resistant admission.  All state below is inert (never read) when
        # ``single_pool`` is False, so the two-tier path stays byte-identical.
        self.single_pool = bool(single_pool)
        # Experts promoted out of probation by a later hit (protected segment).
        self._protected: set[int] = set()
        # Resident-expert recency stamp on a pool-local monotonic clock that
        # PREFILL advances too (unlike ``_decode_epoch``, which prefill must not
        # touch): this is what leaves the prompt's tail resident for decode.
        self._pool_recency: dict[int, int] = {}
        self._pool_clock = 0
        # Segmented-LRU protected-segment cap (~80%), leaving a probation landing
        # zone so a scan cannot starve; promotion past the cap demotes the coldest
        # protected expert back to probation.
        self._protected_cap = max(1, int(self.persistent_slots * 0.8))
        # W87 HIGH-2 (per-request warming): True once a DECODE route is planned;
        # the next PREFILL (a new request) demotes the prior request's protected
        # set so this prompt can re-warm.  Reset by that demote and by reset().
        self._saw_decode_since_prefill = False

    @property
    def resident_experts(self) -> tuple[int, ...]:
        return tuple(expert for expert in self._slot_to_expert if expert is not None)

    def published_experts(self, expert_ids: Iterable[int]) -> frozenset[int]:
        """Subset of ``expert_ids`` that would hit-resolve right now.

        Persistent residents plus committed prefetch-ring entries; inflight
        ring assignments are excluded because a route may not consume a
        slot mid-write. Pure peek — no history, epoch, or slot mutation.
        """

        expert_ids = tuple(expert_ids)
        ring_published: dict[int, int] = (
            self._prefetch_ring.published(self._layer_id, expert_ids)
            if self._prefetch_ring is not None
            else {}
        )
        return frozenset(
            expert
            for expert in expert_ids
            if expert in self._expert_to_slot or expert in ring_published
        )

    @property
    def occupancy(self) -> int:
        return len(self._expert_to_slot)

    @property
    def persistent_capacity(self) -> int:
        """Resident-entry cap currently admitted by the memory policy."""

        return self._persistent_capacity

    def set_persistent_capacity(self, capacity: int) -> int:
        """Cap resident persistent entries without changing physical slots.

        The runtime lowers this at a KV-growth boundary (then evicts down to
        it) and raises it again on KV shrink.  Entries above the cap are
        never admitted; existing entries above it may only be replaced.
        """

        capacity = _integer("capacity", capacity, minimum=0)
        self._persistent_capacity = min(capacity, self.persistent_slots)
        return self._persistent_capacity

    def peek_victim(
        self, *, excluded: Iterable[int] = (), respect_pins: bool = True
    ) -> tuple[int, int] | None:
        """Return the policy's next eviction candidate without mutating state.

        ``respect_pins`` (default) skips the W64 pinned working set; a
        memory-forced capacity eviction passes ``respect_pins=False`` so it can
        evict a pinned expert as a last resort (memory is the hard constraint).
        """

        slot = self._victim_slot(pinned=set(excluded), respect_pins=respect_pins)
        if slot is None:
            return None
        expert = self._slot_to_expert[slot]
        assert expert is not None
        return expert, slot

    def invalidate_expert(self, expert_id: int) -> int | None:
        """Forget a failed/stale persistent mapping and return its slot."""

        expert = _integer("expert id", expert_id, minimum=0)
        if expert >= self.expert_count:
            raise ValueError(f"expert id {expert} is outside [0, {self.expert_count})")
        slot = self._expert_to_slot.pop(expert, None)
        if slot is not None:
            self._slot_to_expert[slot] = None
        # A forgotten mapping is no longer resident, so it cannot stay pinned,
        # protected, or carry a stale pool-recency stamp (W87: a capacity/health
        # eviction goes through here, so the pool bookkeeping must follow).
        self._pinned.discard(expert)
        self._protected.discard(expert)
        self._pool_recency.pop(expert, None)
        return slot

    def reset(self) -> None:
        """Clear residency and decode history without changing capacity."""

        self._decode_epoch = 0
        self._slot_to_expert = [None] * self.persistent_slots
        self._expert_to_slot.clear()
        self._history = [_ExpertHistory() for _ in range(self.expert_count)]
        self._prefill_seed_candidates.clear()
        # A SHARED ring is reset once by the runtime (not per bank, or the first
        # bank's reset would wipe the others' entries); a bank-owned ring resets
        # here.
        if self._owns_prefetch_ring and self._prefetch_ring is not None:
            self._prefetch_ring.reset()
        self._pinned.clear()
        self._prefill_route_freq.clear()
        self._protected.clear()
        self._pool_recency.clear()
        self._pool_clock = 0
        self._saw_decode_since_prefill = False

    def plan_prefetch(self, expert_ids: Iterable[int]) -> tuple[SlotLoad, ...]:
        """Assign shared-ring slots for this layer's predicted experts.

        Delegates to the shared :class:`GlobalPrefetchRing` keyed by this bank's
        layer; skips experts already resident for this layer (its persistent
        tier), committed, inflight, or under the reeviction embargo. Ring
        replacement is round-robin over the WHOLE (cross-layer) ring but never
        recycles an inflight slot, and never touches this bank's persistent
        tier. Empty when prefetch is disabled."""

        if self._prefetch_ring is None:
            return ()
        return self._prefetch_ring.plan_prefetch(
            self._layer_id, expert_ids, resident=self._expert_to_slot.keys()
        )

    def prefetch_ticket(self, expert_id: int) -> int | None:
        """The inflight assignment ticket for ``(this layer, expert)``, if any."""

        if self._prefetch_ring is None:
            return None
        return self._prefetch_ring.prefetch_ticket(self._layer_id, expert_id)

    def commit_prefetch(
        self,
        expert_id: int,
        *,
        ticket: int | None = None,
    ) -> bool:
        """Publish a completed shared-ring load for this layer as hit-eligible."""

        if self._prefetch_ring is None:
            return False
        return self._prefetch_ring.commit_prefetch(
            self._layer_id, expert_id, ticket=ticket
        )

    def invalidate_prefetch(
        self,
        expert_id: int,
        *,
        ticket: int | None = None,
    ) -> int | None:
        """Forget a failed or stale shared-ring assignment for this layer."""

        if self._prefetch_ring is None:
            return None
        return self._prefetch_ring.invalidate_prefetch(
            self._layer_id, expert_id, ticket=ticket
        )

    def consume_prefetch_wasted(self) -> int:
        """Return and zero the shared ring's wasted-read count.

        A wasted read is a committed ring entry recycled round-robin without
        ever being consumed by a true route (a mispredicted one-layer-ahead
        prediction). The count is ring-global (across layers); the runtime
        drains it after each ``plan_prefetch`` under the layer lock. Failed
        reads and reset are excluded -- only settled, never-hit commits count."""

        if self._prefetch_ring is None:
            return 0
        return self._prefetch_ring.consume_wasted()

    @property
    def _prefetch_expert_to_slot(self) -> dict[int, int]:
        """This layer's committed ring entries as ``{expert: slot}`` -- a read-only
        per-layer projection of the shared ring (keyed globally by (layer,
        expert)), matching the pre-W93 per-layer dict for introspection/tests."""

        if self._prefetch_ring is None:
            return {}
        layer = self._layer_id
        return {
            expert: slot
            for (ring_layer, expert), slot in self._prefetch_ring._key_to_slot.items()
            if ring_layer == layer
        }

    @property
    def _prefetch_inflight(self) -> dict[int, tuple[int, int]]:
        """This layer's inflight ring assignments as ``{expert: (slot, ticket)}``
        -- the read-only per-layer projection of the shared ring."""

        if self._prefetch_ring is None:
            return {}
        layer = self._layer_id
        return {
            expert: entry
            for (ring_layer, expert), entry in self._prefetch_ring._inflight.items()
            if ring_layer == layer
        }

    def prepare_prefill_seed(self, expert_ids: Iterable[int]) -> tuple[int, ...]:
        """Choose prompt-frequent experts for empty slots without eviction."""

        # W64: accumulate the prompt's routing frequency BEFORE the capacity
        # early-return, so ``pin_working_set`` can rank the resident set by
        # prompt-frequent experts even when every persistent slot is already
        # full (the return value/seed selection below is otherwise unchanged).
        experts = self._validate_experts_for_seed(expert_ids)
        # W87 HIGH-2: a new request demotes the prior protected set here, so the
        # seed budget below is the whole (now-unprotected) pool rather than the
        # zero empty slots a full pool would report.
        self._reopen_pool_for_new_request()
        self._prefill_route_freq.update(experts)
        if self.single_pool:
            # Budget = capacity - protected: demoted residents are probationary and
            # the seed evicts them, so the seed can span the whole pool per request.
            empty = self._persistent_capacity - len(self._protected)
        else:
            empty = self._persistent_capacity - self.occupancy
        if empty <= 0:
            self._prefill_seed_candidates.clear()
            return ()
        counts = Counter(experts)
        ranked = sorted(counts, key=lambda expert: (-counts[expert], expert))
        if self.single_pool:
            # W87 IMPORTANT: rank the top-``empty`` by frequency REGARDLESS of
            # residency.  A same-hot-set request 2 returns experts still resident
            # (demoted to probation by the reopen); RE-PROTECT those in place so the
            # seed of the low-frequency remainder cannot evict them, and seed only
            # the NON-resident chosen for admission.  Without this the demote + seed
            # evicts the returning hot set (only ~12/49 survive; first-64 0.355).
            # Recency is refreshed by ASCENDING frequency (least-frequent lowest),
            # matching the seed-first admission order.
            chosen_all = ranked[:empty]
            resident_chosen = [e for e in chosen_all if e in self._expert_to_slot]
            for expert in sorted(resident_chosen, key=lambda e: counts[e]):
                self._protected.add(expert)
                self._pool_clock += 1
                self._pool_recency[expert] = self._pool_clock
            chosen = tuple(e for e in chosen_all if e not in self._expert_to_slot)
        else:
            chosen = tuple(
                expert for expert in ranked if expert not in self._expert_to_slot
            )[:empty]
        self._prefill_seed_candidates = set(chosen)
        return chosen

    # ------------------------------------------------------------------
    # W64 (R3-pin): post-prefill pinned working set.
    # ------------------------------------------------------------------
    def _pin_rank(self, expert: int) -> float:
        """Ranking key for pinning: prefill routing frequency, then the decayed
        decode score (so a refresh mid-decode tracks the live hot set)."""

        return float(self._prefill_route_freq.get(expert, 0)) + self._score(expert)

    def pin_working_set(
        self,
        *,
        top_k: int | None = None,
        free_tail: int = 0,
        experts: Iterable[int] | None = None,
    ) -> tuple[int, ...]:
        """Pin a post-prefill working set: mark the top-ranked resident experts
        never-recyclable so their persistent slot is stable for the whole decode.

        ``experts`` pins exactly that resident subset (the ``pin_ws`` arm passes
        the full resident set -> a fully static layer). Otherwise the currently
        resident persistent experts are ranked by ``_pin_rank`` and the top
        ``top_k`` are pinned, capped so at least ``free_tail`` persistent slots
        stay unpinned for decode misses to admit into. Pinning never moves a slot
        or changes any gather output -- only which experts a later eviction may
        choose -- so decode stays byte-identical with pinning on or off. Replaces
        any prior pin set for this layer. Returns the pinned expert ids (sorted).
        """

        resident = [expert for expert in self._slot_to_expert if expert is not None]
        if experts is not None:
            requested = {int(expert) for expert in experts}
            chosen = [expert for expert in resident if expert in requested]
        else:
            ranked = sorted(resident, key=lambda expert: (-self._pin_rank(expert), expert))
            limit = len(resident)
            if free_tail > 0:
                limit = min(limit, max(0, self.persistent_slots - int(free_tail)))
            if top_k is not None:
                limit = min(limit, max(0, int(top_k)))
            chosen = ranked[:limit]
        self._pinned = set(chosen)
        return tuple(sorted(self._pinned))

    def clear_pins(self) -> None:
        """Drop the pinned working set (return to pure LRU/frequency eviction)."""

        self._pinned.clear()

    @property
    def pinned_experts(self) -> frozenset[int]:
        """The never-recycled working set (empty unless pinning is active)."""

        return frozenset(self._pinned)

    @property
    def pinned_count(self) -> int:
        return len(self._pinned)

    @property
    def pinned_static(self) -> bool:
        """True when a non-empty pinned working set is active for this layer.

        Its experts are never recycled on normal admission, so any all-hit route
        whose experts are a subset of :attr:`pinned_experts` is slot-stable for
        the whole decode -- the precondition the W44 barrier-free device route
        needs to gather without a pin/fence (see :meth:`route_all_pinned`)."""

        return bool(self._pinned)

    def route_all_pinned(self, expert_ids: Iterable[int]) -> bool:
        """True iff every routed expert is pinned (and thus slot-stable). A
        device route may take the barrier-free path for exactly these routes."""

        if not self._pinned:
            return False
        return all(int(expert) in self._pinned for expert in expert_ids)

    def _validate_experts_for_seed(self, expert_ids: Iterable[int]) -> tuple[int, ...]:
        try:
            experts = tuple(
                _integer("expert id", expert, minimum=0) for expert in expert_ids
            )
        except TypeError as exc:
            raise TypeError("expert ids must be exact integers") from exc
        for expert in experts:
            if expert >= self.expert_count:
                raise ValueError(
                    f"expert id {expert} is outside [0, {self.expert_count})"
                )
        return experts

    def _validate_experts(self, expert_ids: Iterable[int]) -> tuple[int, ...]:
        try:
            experts = tuple(
                _integer("expert id", expert, minimum=0) for expert in expert_ids
            )
        except TypeError as exc:
            raise TypeError("expert ids must be exact integers") from exc
        if not experts:
            raise ValueError("a route must select at least one expert")
        for expert in experts:
            if not 0 <= expert < self.expert_count:
                raise ValueError(
                    f"expert id {expert} is outside [0, {self.expert_count})"
                )
        unique_count = len(dict.fromkeys(experts))
        # W87 (review HIGH-1): the single-pool wave width is transient_slots (the
        # merged-capacity widening was retired -- protected/pinned slots make a
        # wider prefill wave unserviceable), so the route bound is transient_slots
        # on BOTH paths, exactly as before.  route_waves keeps every wave within
        # it; the plan() overflow guard below is the final tripwire.
        if unique_count > self.transient_slots:
            raise ValueError(
                "transient_slots must cover the maximum unique experts in one route"
            )
        return experts

    def _score(self, expert: int) -> float:
        history = self._history[expert]
        age = self._decode_epoch - history.score_epoch
        if age <= 0 or history.score == 0.0:
            return history.score
        return history.score * (self.frequency_decay**age)

    def _touch_decode(self, expert: int) -> None:
        history = self._history[expert]
        history.score = self._score(expert) + 1.0
        history.score_epoch = self._decode_epoch
        history.last_used = self._decode_epoch

    def _empty_persistent_slot(self) -> int | None:
        if self.occupancy >= self._persistent_capacity:
            return None
        for slot, expert in enumerate(self._slot_to_expert):
            if expert is None:
                return slot
        return None

    def _victim_slot(
        self, *, pinned: set[int], respect_pins: bool = True
    ) -> int | None:
        # The route's own hits are always protected; the W64 pinned working set
        # is additionally protected on normal decode admission (``respect_pins``)
        # but NOT on a memory-forced capacity eviction. When ``_pinned`` is empty
        # (the default / lever-off state) ``blocked`` is exactly ``pinned``, so
        # victim selection -- and therefore every output and counter -- is
        # bitwise-identical to the pre-W64 behaviour.
        blocked = pinned | self._pinned if (respect_pins and self._pinned) else pinned
        candidates: list[tuple[float, int, int]] = []
        for slot, expert in enumerate(self._slot_to_expert):
            if expert is None or expert in blocked:
                continue
            history = self._history[expert]
            if self.cache_policy == "lru":
                candidates.append((float(history.last_used), 0, slot))
            else:
                candidates.append((self._score(expert), history.last_used, slot))
        return min(candidates)[2] if candidates else None

    def _assign_persistent(
        self,
        *,
        slot: int,
        expert: int,
        evictions: list[SlotEviction],
    ) -> None:
        previous = self._slot_to_expert[slot]
        if previous is not None:
            del self._expert_to_slot[previous]
            # Normal admission never selects a pinned slot (``_victim_slot``
            # excludes ``_pinned``); this discard only matters if a pinned slot
            # is reused through some other path, keeping ``_pinned`` truthful.
            self._pinned.discard(previous)
            evictions.append(
                SlotEviction(
                    slot=slot,
                    previous_expert=previous,
                    next_expert=expert,
                )
            )
        self._slot_to_expert[slot] = expert
        self._expert_to_slot[expert] = slot

    # ------------------------------------------------------------------
    # W87 single-pool (env MTPLX_DSV41_SINGLE_SLOT_POOL) 2Q admission.
    # ------------------------------------------------------------------
    def _pool_victim_slot(
        self, *, pinned: set[int], allow_protected: bool
    ) -> int | None:
        """Coldest PROBATIONARY resident; only if ``allow_protected`` (decode) and
        probation is empty, the coldest PROTECTED resident; never a route hit
        (``pinned``) nor a W64 pin (``_pinned``).  Segmented-LRU: a scan-landing
        (probationary) expert is evicted before any promoted one.  During PREFILL
        the caller passes ``allow_protected=False`` (HIGH-3), so a wide prefill
        wave can NEVER evict the earned/seeded set -- it overflows to transient
        instead (bounded by :meth:`pool_admission_capacity`)."""

        blocked = pinned | self._pinned if self._pinned else pinned
        probation: list[tuple[int, int]] = []
        protected: list[tuple[int, int]] = []
        for slot, expert in enumerate(self._slot_to_expert):
            if expert is None or expert in blocked:
                continue
            recency = self._pool_recency.get(expert, 0)
            if expert in self._protected:
                protected.append((recency, slot))
            else:
                probation.append((recency, slot))
        if probation:
            return min(probation)[1]
        if allow_protected and protected:
            return min(protected)[1]
        return None

    def _pool_admit(
        self,
        *,
        expert: int,
        evictions: list[SlotEviction],
        pinned: set[int],
        protect: bool,
        allow_protected: bool,
    ) -> int | None:
        """Admit a miss into the merged resident pool.  ``protect`` marks it
        PROTECTED on entry (a frequency-seed expert -- HIGH-2); otherwise it lands
        PROBATIONARY.  ``allow_protected`` lets the victim search fall back to the
        protected segment (decode) or not (prefill -- HIGH-3).  Returns the slot,
        or None when no admissible slot exists (the miss overflows to transient)."""

        slot = self._empty_persistent_slot()
        if slot is None:
            slot = self._pool_victim_slot(
                pinned=pinned, allow_protected=allow_protected
            )
        if slot is None:
            return None
        victim = self._slot_to_expert[slot]
        if victim is not None:
            self._protected.discard(victim)
            self._pool_recency.pop(victim, None)
        self._assign_persistent(slot=slot, expert=expert, evictions=evictions)
        self._pool_clock += 1
        self._pool_recency[expert] = self._pool_clock
        if protect:
            self._protected.add(expert)
        else:
            self._protected.discard(expert)
        return slot

    def _pool_touch(self, expert: int) -> int:
        """Refresh a resident expert's pool recency and promote it out of
        probation on this (re-)reference.  Returns 1 iff it was promoted now."""

        self._pool_clock += 1
        self._pool_recency[expert] = self._pool_clock
        if expert in self._protected:
            return 0
        self._protected.add(expert)
        if len(self._protected) > self._protected_cap:
            # Demote the coldest OTHER protected expert back to probation (still
            # resident) so the probation landing zone never starves under a
            # promote-heavy workload.
            coldest = min(
                (e for e in self._protected if e != expert),
                key=lambda e: self._pool_recency.get(e, 0),
                default=None,
            )
            if coldest is not None:
                self._protected.discard(coldest)
        return 1

    def _reopen_pool_for_new_request(self) -> None:
        """W87 HIGH-2: on the first PREFILL after a DECODE (a new request), demote
        this layer's protected hot set -- clear ``_protected`` but KEEP
        ``_pool_recency`` (LRU order) -- so the new prompt's frequency seed can
        re-warm the pool and its prefill scan can evict the prior request's set.
        Without this, a served daemon warms only the FIRST request: the seed budget
        would be zero (pool full) and the stale protected set would never demote, so
        request 2 (a different hot set) starts colder than the two-tier LRU.  No-op
        unless a decode route was seen since the last reopen (and off the flag)."""

        if self.single_pool and self._saw_decode_since_prefill:
            self._protected.clear()
            # Prompt frequency belongs to this request. Keep decode recency,
            # but do not let a prior prompt reverse the new seed's priority.
            # Replace rather than clear so a route transaction can restore the
            # old counter by reference if this unseeded reopen is rolled back.
            self._prefill_route_freq = Counter()
            self._saw_decode_since_prefill = False

    def plan(
        self,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
    ) -> RoutePlan:
        """Resolve router expert ids to persistent or transient slots."""

        experts = self._validate_experts(expert_ids)
        phase = RoutingPhase(phase)
        unique_experts = tuple(dict.fromkeys(experts))

        if phase is RoutingPhase.DECODE:
            self._decode_epoch += 1
            if self.single_pool:
                self._saw_decode_since_prefill = True
            for expert in experts:
                self._touch_decode(expert)
            # W93: advance this layer's epoch in the shared ring so its reeviction
            # embargo tracks the layer's own decode cadence.
            if self._prefetch_ring is not None:
                self._prefetch_ring.note_decode(self._layer_id)

        hit_set = {
            expert for expert in unique_experts if expert in self._expert_to_slot
        }
        # W93: committed shared-ring entries for THIS layer that the true route
        # needs resolve as hits reading the ring slot in place (no re-read, no
        # copy). The ring is keyed by (layer, expert), so this bank sees only its
        # own layer's commits.
        ring_published: dict[int, int] = (
            self._prefetch_ring.published(
                self._layer_id,
                [expert for expert in unique_experts if expert not in hit_set],
            )
            if self._prefetch_ring is not None
            else {}
        )
        prefetch_hits = set(ring_published)
        miss_order = [
            expert
            for expert in unique_experts
            if expert not in hit_set and expert not in prefetch_hits
        ]
        resolved: dict[int, int] = {
            expert: self._expert_to_slot[expert] for expert in hit_set
        }
        for expert in prefetch_hits:
            resolved[expert] = ring_published[expert]
        # Mark these ring commits consumed, so a later round-robin eviction is not
        # miscounted as wasted. W95g (review MEDIUM-2): capture the FIRST-consumption
        # subset (not yet marked used) BEFORE marking, so a resident ring record
        # re-consumed on a later token counts a hit at most once.
        prefetch_first_hits: frozenset[int] = frozenset()
        if prefetch_hits and self._prefetch_ring is not None:
            prefetch_first_hits = self._prefetch_ring.first_consumption(
                self._layer_id, prefetch_hits
            )
            self._prefetch_ring.mark_used(self._layer_id, prefetch_hits)
        hit_set |= prefetch_hits
        loads: list[SlotLoad] = []
        evictions: list[SlotEviction] = []
        pinned = set(hit_set)
        transient_experts: list[int] = []
        pool_loads = 0
        scan_inserts = 0
        promotions = 0
        if self.single_pool:
            # Decode hits promote (2Q); PREFILL hits only refresh recency --
            # prefill promotion is FREQUENCY-driven via the seed (HIGH-2), not
            # "hit in a later wave" (which never fires on the real layer-major
            # single sorted route).  Prefetch-ring hits live outside the pool.
            promote_hits = phase is RoutingPhase.DECODE
            for expert in hit_set:
                if expert in self._expert_to_slot:
                    if promote_hits:
                        promotions += self._pool_touch(expert)
                    else:
                        self._pool_clock += 1
                        self._pool_recency[expert] = self._pool_clock

        if self.single_pool and phase is RoutingPhase.PREFILL:
            # HIGH-2 fallback: a prefill route that skipped prepare_prefill_seed
            # still reopens the pool for the new request here (idempotent).
            self._reopen_pool_for_new_request()
            if self._prefill_seed_candidates:
                # HIGH-2: admit the frequency-seed experts FIRST within the wave so
                # they claim (protected) slots before the low-frequency scan fills
                # it -- a seed appearing late in an id-sorted wave would otherwise
                # find the pool full and overflow to transient (lost, since
                # sorted-unique waves never revisit it).  LOW: order the seed by
                # ASCENDING frequency so the least-frequent seed gets the lowest
                # recency and is the first protected expert a later decode eviction
                # drops (not merely the lowest id).  Reordering ADMISSION never
                # changes the route output: ``resolved`` maps each expert to its
                # slot by identity and the gather is recombined by original position.
                _seed_first = sorted(
                    (e for e in miss_order if e in self._prefill_seed_candidates),
                    key=lambda e: self._prefill_route_freq.get(e, 0),
                )
                if _seed_first:
                    _rest = [
                        e
                        for e in miss_order
                        if e not in self._prefill_seed_candidates
                    ]
                    miss_order = _seed_first + _rest

        for expert in miss_order:
            persistent_slot: int | None = None
            if self.single_pool:
                is_prefill = phase is RoutingPhase.PREFILL
                # HIGH-2: a prompt-frequency seed expert enters PROTECTED so a
                # later prefill wave (or a re-prefill) cannot scan it out; every
                # other prefill miss lands probationary.  HIGH-3: prefill never
                # evicts a protected/seeded expert (allow_protected=False) -- it
                # overflows to transient instead.
                is_seed = is_prefill and expert in self._prefill_seed_candidates
                persistent_slot = self._pool_admit(
                    expert=expert,
                    evictions=evictions,
                    pinned=pinned,
                    protect=is_seed,
                    allow_protected=not is_prefill,
                )
                if is_seed:
                    self._prefill_seed_candidates.discard(expert)
                if persistent_slot is None:
                    transient_experts.append(expert)
                    continue
                pool_loads += 1
                if is_seed:
                    promotions += 1
                elif is_prefill:
                    scan_inserts += 1
                pinned.add(expert)
                resolved[expert] = persistent_slot
                loads.append(
                    SlotLoad(expert=expert, slot=persistent_slot, persistent=True)
                )
                continue
            if (
                phase is RoutingPhase.PREFILL
                and expert in self._prefill_seed_candidates
            ):
                persistent_slot = self._empty_persistent_slot()
                self._prefill_seed_candidates.discard(expert)
            elif phase is RoutingPhase.DECODE and self.persistent_slots:
                persistent_slot = self._empty_persistent_slot()
                if persistent_slot is None:
                    victim_slot = self._victim_slot(pinned=pinned)
                    if victim_slot is not None:
                        victim = self._slot_to_expert[victim_slot]
                        assert victim is not None
                        # Do not let a first-seen singleton evict a resident
                        # merely because decay made the resident score < 1.
                        # A second decode observation (or older history) must
                        # first lift the candidate above this admission floor.
                        if self.cache_policy == "lru" or self._score(expert) > max(
                            1.0, self._score(victim)
                        ):
                            persistent_slot = victim_slot

            if persistent_slot is None:
                transient_experts.append(expert)
                continue

            self._assign_persistent(
                slot=persistent_slot,
                expert=expert,
                evictions=evictions,
            )
            pinned.add(expert)
            resolved[expert] = persistent_slot
            loads.append(SlotLoad(expert=expert, slot=persistent_slot, persistent=True))

        if self.single_pool and len(transient_experts) > self.transient_slots:
            # HIGH-1 tripwire: a correctly bounded wave (route_waves ->
            # pool_admission_capacity) can never overflow the transient scratch.
            # Raise BEFORE emitting an out-of-plan slot index (which would fault
            # deep in ExpertSlotPool._physical) so a bounding bug is loud here.
            raise ValueError(
                "single-pool route overflowed the transient scratch: "
                f"{len(transient_experts)} experts spilled but only "
                f"{self.transient_slots} transient slots exist "
                f"(persistent_capacity={self._persistent_capacity}, "
                f"pinned={len(self._pinned)}); the route was not bounded by "
                "pool_admission_capacity()."
            )
        transient_base = self.persistent_slots
        for offset, expert in enumerate(transient_experts):
            slot = transient_base + offset
            resolved[expert] = slot
            loads.append(SlotLoad(expert=expert, slot=slot, persistent=False))

        if phase is RoutingPhase.DECODE:
            for expert in hit_set:
                self._history[expert].last_used = self._decode_epoch

        return RoutePlan(
            phase=phase,
            experts=experts,
            slots=tuple(resolved[expert] for expert in experts),
            hits=tuple(expert for expert in unique_experts if expert in hit_set),
            misses=tuple(miss_order),
            loads=tuple(loads),
            evictions=tuple(evictions),
            pool_loads=pool_loads,
            scan_inserts=scan_inserts,
            promotions=promotions,
            prefetch_hits=tuple(prefetch_hits),
            prefetch_first_hits=tuple(prefetch_first_hits),
        )

    def plan_transaction(
        self,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
    ) -> tuple[RoutePlan, RoutePolicyTxn]:
        experts = self._validate_experts(expert_ids)
        unique_experts = tuple(dict.fromkeys(experts))
        decode_epoch = self._decode_epoch
        histories = {
            expert: (
                self._history[expert].score,
                self._history[expert].score_epoch,
                self._history[expert].last_used,
            )
            for expert in unique_experts
        }
        seed_candidates = set(self._prefill_seed_candidates)
        pool_protected = set(self._protected)
        pool_recency = dict(self._pool_recency)
        pool_clock = self._pool_clock
        saw_decode = self._saw_decode_since_prefill
        prefill_route_freq = self._prefill_route_freq
        plan = self.plan(experts, phase=phase)

        def rollback() -> None:
            evictions = {eviction.slot: eviction for eviction in plan.evictions}
            for load in reversed(plan.loads):
                if not load.persistent:
                    continue
                if self._expert_to_slot.get(load.expert) == load.slot:
                    self._expert_to_slot.pop(load.expert, None)
                eviction = evictions.get(load.slot)
                if eviction is None:
                    self._slot_to_expert[load.slot] = None
                else:
                    self._slot_to_expert[load.slot] = eviction.previous_expert
                    self._expert_to_slot[eviction.previous_expert] = load.slot
            self._decode_epoch = decode_epoch
            for expert, values in histories.items():
                history = self._history[expert]
                history.score, history.score_epoch, history.last_used = values
            self._prefill_seed_candidates = set(seed_candidates)
            self._protected = pool_protected
            self._pool_recency = pool_recency
            self._pool_clock = pool_clock
            self._saw_decode_since_prefill = saw_decode
            self._prefill_route_freq = prefill_route_freq

        return plan, RoutePolicyTxn(rollback=rollback)

    def try_plan_all_hits(
        self,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
    ) -> RoutePlan | None:
        """Plan a fully resident route without transient-capacity splitting.

        A failed probe is side-effect free so callers can fall back to the
        normal bounded-wave planner.  A successful probe applies the same
        decode frequency/recency updates as :meth:`plan` and preserves every
        router assignment, including duplicate expert IDs, in ``slots``.
        """

        experts = self._validate_experts_for_seed(expert_ids)
        if not experts:
            raise ValueError("a route must select at least one expert")
        phase = RoutingPhase(phase)
        unique_experts = tuple(dict.fromkeys(experts))
        if any(expert not in self._expert_to_slot for expert in unique_experts):
            return None

        if phase is RoutingPhase.DECODE:
            self._decode_epoch += 1
            if self.single_pool:
                self._saw_decode_since_prefill = True
            for expert in experts:
                self._touch_decode(expert)
            for expert in unique_experts:
                self._history[expert].last_used = self._decode_epoch

        promotions = 0
        if self.single_pool:
            # Phase-guard: a DECODE all-hit route promotes (2Q); a PREFILL all-hit
            # only refreshes recency -- prefill promotion is frequency-seed-driven,
            # not hit-driven (matches the plan() hit handling).
            promote = phase is RoutingPhase.DECODE
            for expert in unique_experts:
                if expert in self._expert_to_slot:
                    if promote:
                        promotions += self._pool_touch(expert)
                    else:
                        self._pool_clock += 1
                        self._pool_recency[expert] = self._pool_clock

        return RoutePlan(
            phase=phase,
            experts=experts,
            slots=tuple(self._expert_to_slot[expert] for expert in experts),
            hits=unique_experts,
            misses=(),
            loads=(),
            evictions=(),
            promotions=promotions,
        )

    def try_plan_all_hits_transaction(
        self,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
    ) -> tuple[RoutePlan, RoutePolicyTxn] | None:
        experts = self._validate_experts_for_seed(expert_ids)
        unique_experts = tuple(dict.fromkeys(experts))
        decode_epoch = self._decode_epoch
        histories = {
            expert: (
                self._history[expert].score,
                self._history[expert].score_epoch,
                self._history[expert].last_used,
            )
            for expert in unique_experts
        }
        pool_protected = set(self._protected)
        pool_recency = dict(self._pool_recency)
        pool_clock = self._pool_clock
        saw_decode = self._saw_decode_since_prefill
        plan = self.try_plan_all_hits(experts, phase=phase)
        if plan is None:
            return None

        def rollback() -> None:
            self._decode_epoch = decode_epoch
            for expert, values in histories.items():
                history = self._history[expert]
                history.score, history.score_epoch, history.last_used = values
            self._protected = pool_protected
            self._pool_recency = pool_recency
            self._pool_clock = pool_clock
            self._saw_decode_since_prefill = saw_decode

        return plan, RoutePolicyTxn(rollback=rollback)


class GlobalExpertSlotBank:
    """One fixed expert-record cache shared by every routed model layer.

    Keys include both layer and expert ID because equal expert IDs in different
    layers name unrelated weights.  The physical buffers are allocated once;
    this policy only changes the generation-safe indirection from a key to a
    global slot.  Prefill initially admits at most the legacy uniform quota per
    layer so early layers cannot consume the entire empty pool.  Decode then
    allows the replacement policy to move capacity between layers.
    """

    def __init__(
        self,
        *,
        layer_indices: Iterable[int],
        expert_count: int,
        persistent_slots: int,
        transient_slots: int,
        prefill_slots_per_layer: int,
        frequency_decay: float = 0.995,
        cache_policy: str = "lru",
    ) -> None:
        self.layer_indices = tuple(
            _integer("layer index", layer, minimum=0) for layer in layer_indices
        )
        if not self.layer_indices or len(set(self.layer_indices)) != len(
            self.layer_indices
        ):
            raise ValueError("layer_indices must contain unique routed layers")
        self._layer_set = set(self.layer_indices)
        self.expert_count = _integer("expert_count", expert_count, minimum=1)
        self.persistent_slots = _integer(
            "persistent_slots", persistent_slots, minimum=0
        )
        self.transient_slots = _integer("transient_slots", transient_slots, minimum=1)
        self.prefill_slots_per_layer = _integer(
            "prefill_slots_per_layer", prefill_slots_per_layer, minimum=0
        )
        if self.prefill_slots_per_layer > self.expert_count:
            raise ValueError("prefill_slots_per_layer cannot exceed expert_count")
        maximum_keys = len(self.layer_indices) * self.expert_count
        if self.persistent_slots > maximum_keys:
            raise ValueError("persistent_slots cannot exceed routed expert count")
        if isinstance(frequency_decay, bool):
            raise TypeError("frequency_decay must be a finite number")
        try:
            self.frequency_decay = float(frequency_decay)
        except (TypeError, ValueError) as exc:
            raise TypeError("frequency_decay must be a finite number") from exc
        if not isfinite(self.frequency_decay) or not 0.0 < self.frequency_decay <= 1.0:
            raise ValueError("frequency_decay must be in (0, 1]")
        if cache_policy not in {"frequency", "lru"}:
            raise ValueError("cache_policy must be 'frequency' or 'lru'")
        self.cache_policy = cache_policy
        self.slot_count = self.persistent_slots + self.transient_slots

        self._decode_epoch = 0
        self._slot_to_key: list[tuple[int, int] | None] = [None] * self.persistent_slots
        self._key_to_slot: dict[tuple[int, int], int] = {}
        self._directory: dict[tuple[int, int], _GlobalDirectoryEntry] = {}
        self._slot_generations: list[int] = [0] * self.persistent_slots
        self._free_slots = deque(range(self.persistent_slots))
        self._free_slot_set = set(range(self.persistent_slots))
        self._lru: OrderedDict[tuple[int, int], int] = OrderedDict()
        self._lru_clock = 0
        self._history: dict[tuple[int, int], _ExpertHistory] = {}
        self._layer_occupancy: Counter[int] = Counter()
        self._evictions = 0
        self._cross_layer_evictions = 0
        self._prefill_seed_candidates: dict[int, set[int]] = {
            layer: set() for layer in self.layer_indices
        }
        self._persistent_capacity = self.persistent_slots

    @property
    def occupancy(self) -> int:
        return len(self._key_to_slot)

    @property
    def persistent_capacity(self) -> int:
        """Resident-entry cap currently admitted by the memory policy."""

        return self._persistent_capacity

    def set_persistent_capacity(self, capacity: int) -> int:
        """Cap resident persistent entries without changing physical slots.

        The runtime lowers this at a KV-growth boundary (then evicts down to
        it) and raises it again on KV shrink.  Entries above the cap are
        never admitted; existing entries above it may only be replaced.
        """

        capacity = _integer("capacity", capacity, minimum=0)
        self._persistent_capacity = min(capacity, self.persistent_slots)
        return self._persistent_capacity

    def peek_victim(
        self, *, excluded: Iterable[tuple[int, int]] = ()
    ) -> tuple[int, int, int] | None:
        """Return the policy's next (layer, expert, slot) eviction candidate."""

        slot = self._victim_slot(pinned=set(excluded))
        if slot is None:
            return None
        key = self._slot_to_key[slot]
        assert key is not None
        return key[0], key[1], slot

    @property
    def resident_experts_by_layer(self) -> dict[int, tuple[int, ...]]:
        grouped: dict[int, list[int]] = {layer: [] for layer in self.layer_indices}
        for key in self._slot_to_key:
            if key is not None:
                grouped[key[0]].append(key[1])
        return {layer: tuple(experts) for layer, experts in grouped.items()}

    @property
    def occupancy_by_layer(self) -> dict[int, int]:
        return {
            layer: int(self._layer_occupancy[layer]) for layer in self.layer_indices
        }

    def _key(self, layer: int, expert: int) -> tuple[int, int]:
        layer = _integer("layer index", layer, minimum=0)
        expert = _integer("expert id", expert, minimum=0)
        if layer not in self._layer_set:
            raise ValueError(f"layer {layer} is not a routed model layer")
        if expert >= self.expert_count:
            raise ValueError(f"expert id {expert} is outside [0, {self.expert_count})")
        return layer, expert

    def _validate_experts(
        self, layer: int, expert_ids: Iterable[int]
    ) -> tuple[int, tuple[int, ...]]:
        layer, experts = self._validate_experts_without_capacity(layer, expert_ids)
        if len(dict.fromkeys(experts)) > self.transient_slots:
            raise ValueError(
                "transient_slots must cover the maximum unique experts in one route"
            )
        return layer, experts

    def _validate_experts_without_capacity(
        self, layer: int, expert_ids: Iterable[int]
    ) -> tuple[int, tuple[int, ...]]:
        layer = self._key(layer, 0)[0]
        try:
            experts = tuple(
                _integer("expert id", expert, minimum=0) for expert in expert_ids
            )
        except TypeError as exc:
            raise TypeError("expert ids must be exact integers") from exc
        if not experts:
            raise ValueError("a route must select at least one expert")
        for expert in experts:
            self._key(layer, expert)
        return layer, experts

    def _history_for(self, key: tuple[int, int]) -> _ExpertHistory:
        history = self._history.get(key)
        if history is None:
            history = _ExpertHistory()
            self._history[key] = history
        return history

    def _score(self, key: tuple[int, int]) -> float:
        history = self._history_for(key)
        age = self._decode_epoch - history.score_epoch
        if age <= 0 or history.score == 0.0:
            return history.score
        return history.score * (self.frequency_decay**age)

    def _touch_decode(self, key: tuple[int, int]) -> None:
        history = self._history_for(key)
        history.score = self._score(key) + 1.0
        history.score_epoch = self._decode_epoch
        history.last_used = self._decode_epoch

    def _empty_slot(self) -> int | None:
        if self.occupancy >= self._persistent_capacity:
            return None
        if not self._free_slots:
            return None
        slot = self._free_slots.popleft()
        self._free_slot_set.remove(slot)
        return slot

    def _touch_lru(self, key: tuple[int, int]) -> None:
        entry = self._directory[key]
        self._lru_clock += 1
        entry.lru_rank = self._lru_clock
        self._lru[key] = entry.slot
        self._lru.move_to_end(key)

    def _discard_lru(self, key: tuple[int, int]) -> None:
        self._lru.pop(key, None)

    def _rebuild_lru(self) -> None:
        self._lru = OrderedDict(
            (key, entry.slot)
            for key, entry in sorted(
                self._directory.items(), key=lambda item: item[1].lru_rank
            )
        )

    def _victim_slot(self, *, pinned: set[tuple[int, int]]) -> int | None:
        if self.cache_policy == "lru":
            for key, slot in self._lru.items():
                if key not in pinned:
                    return slot
            return None
        candidates: list[tuple[float, int, int]] = []
        for slot, key in enumerate(self._slot_to_key):
            if key is None or key in pinned:
                continue
            history = self._history_for(key)
            if self.cache_policy == "lru":
                candidates.append((float(history.last_used), 0, slot))
            else:
                candidates.append((self._score(key), history.last_used, slot))
        return min(candidates)[2] if candidates else None

    def _assign(
        self,
        *,
        slot: int,
        key: tuple[int, int],
        evictions: list[SlotEviction],
        evicted_entries: dict[int, tuple[tuple[int, int], _GlobalDirectoryEntry]]
        | None = None,
        occupancy_before: dict[int, tuple[bool, int]] | None = None,
    ) -> None:
        previous = self._slot_to_key[slot]
        if previous is not None:
            previous_entry = self._directory.get(previous)
            if previous_entry is None:
                raise RuntimeError(
                    "global resident slot is missing its directory entry"
                )
            if evicted_entries is not None:
                evicted_entries[slot] = (
                    previous,
                    _GlobalDirectoryEntry(
                        previous_entry.slot,
                        previous_entry.generation,
                        previous_entry.state,
                        previous_entry.lru_rank,
                    ),
                )
            if occupancy_before is not None:
                occupancy_before.setdefault(
                    previous[0],
                    (
                        previous[0] in self._layer_occupancy,
                        self._layer_occupancy[previous[0]],
                    ),
                )
            del self._key_to_slot[previous]
            self._directory.pop(previous, None)
            self._discard_lru(previous)
            self._layer_occupancy[previous[0]] -= 1
            self._evictions += 1
            if previous[0] != key[0]:
                self._cross_layer_evictions += 1
            evictions.append(
                SlotEviction(
                    slot=slot,
                    previous_expert=previous[1],
                    next_expert=key[1],
                    previous_layer=previous[0],
                    next_layer=key[0],
                )
            )
        elif slot in self._free_slot_set:
            # Defensive support for callers assigning a specifically chosen
            # empty slot rather than consuming it via _empty_slot().
            self._free_slot_set.remove(slot)
            self._free_slots.remove(slot)
        self._slot_generations[slot] += 1
        if occupancy_before is not None:
            occupancy_before.setdefault(
                key[0],
                (key[0] in self._layer_occupancy, self._layer_occupancy[key[0]]),
            )
        self._slot_to_key[slot] = key
        self._key_to_slot[key] = slot
        self._directory[key] = _GlobalDirectoryEntry(
            slot=slot,
            generation=self._slot_generations[slot],
            state="loading",
            lru_rank=0,
        )
        self._touch_lru(key)
        self._layer_occupancy[key[0]] += 1

    def prepare_prefill_seed(
        self, layer: int, expert_ids: Iterable[int]
    ) -> tuple[int, ...]:
        layer, experts = self._validate_experts_without_capacity(layer, expert_ids)
        remaining_layer = max(
            0, self.prefill_slots_per_layer - self._layer_occupancy[layer]
        )
        empty = self._persistent_capacity - self.occupancy
        available = min(remaining_layer, empty)
        if available <= 0:
            self._prefill_seed_candidates[layer].clear()
            return ()
        counts = Counter(experts)
        ranked = sorted(counts, key=lambda expert: (-counts[expert], expert))
        chosen = tuple(
            expert for expert in ranked if (layer, expert) not in self._key_to_slot
        )[:available]
        self._prefill_seed_candidates[layer] = set(chosen)
        return chosen

    def invalidate_expert(self, layer: int, expert_id: int) -> int | None:
        key = self._key(layer, expert_id)
        slot = self._key_to_slot.pop(key, None)
        if slot is not None:
            self._slot_to_key[slot] = None
            self._directory.pop(key, None)
            self._discard_lru(key)
            self._layer_occupancy[layer] -= 1
            if slot not in self._free_slot_set:
                self._free_slots.append(slot)
                self._free_slot_set.add(slot)
        return slot

    def reconcile_slot_generation(self, slot: int, generation: int) -> None:
        """Advance policy state to a generation already used physically."""

        slot = _integer("slot", slot, minimum=0)
        if slot >= self.persistent_slots:
            raise ValueError("slot is outside the persistent global cache")
        generation = _integer("generation", generation, minimum=0)
        self._slot_generations[slot] = max(
            self._slot_generations[slot],
            generation,
        )

    def publish_ready(self, layer: int, plan: RoutePlan) -> None:
        """Publish successfully filled global generations as cache hits."""

        for load in plan.loads:
            if not load.persistent or load.generation is None:
                continue
            key = self._key(layer, load.expert)
            entry = self._directory.get(key)
            if (
                entry is None
                or entry.slot != load.slot
                or entry.generation != load.generation
                or entry.state != "loading"
            ):
                raise RuntimeError("global cache generation changed before publish")
            entry.state = "ready"

    def rollback(self, layer: int, plan: RoutePlan) -> tuple[tuple[int, int], ...]:
        """Remove only loading directory entries reserved by this plan."""

        removed: list[tuple[int, int]] = []
        for load in plan.loads:
            if not load.persistent or load.generation is None:
                continue
            key = self._key(layer, load.expert)
            entry = self._directory.get(key)
            if (
                entry is None
                or entry.slot != load.slot
                or entry.generation != load.generation
                or entry.state != "loading"
            ):
                continue
            del self._directory[key]
            self._key_to_slot.pop(key, None)
            self._discard_lru(key)
            if self._slot_to_key[load.slot] == key:
                self._slot_to_key[load.slot] = None
                self._layer_occupancy[layer] -= 1
                if load.slot not in self._free_slot_set:
                    self._free_slots.append(load.slot)
                    self._free_slot_set.add(load.slot)
            removed.append((load.slot, load.generation))
        return tuple(removed)

    def plan(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
        _evicted_entries: dict[int, tuple[tuple[int, int], _GlobalDirectoryEntry]]
        | None = None,
        _occupancy_before: dict[int, tuple[bool, int]] | None = None,
    ) -> RoutePlan:
        layer, experts = self._validate_experts(layer, expert_ids)
        phase = RoutingPhase(phase)
        unique_experts = tuple(dict.fromkeys(experts))
        keys = tuple((layer, expert) for expert in unique_experts)

        if phase is RoutingPhase.DECODE:
            self._decode_epoch += 1
            for expert in experts:
                self._touch_decode((layer, expert))

        hit_keys = {
            key
            for key in keys
            if (entry := self._directory.get(key)) is not None
            and entry.state == "ready"
        }
        if self.cache_policy == "lru":
            for key in keys:
                if key in hit_keys:
                    self._touch_lru(key)
        hit_set = {expert for key_layer, expert in hit_keys if key_layer == layer}
        miss_order = [expert for expert in unique_experts if expert not in hit_set]
        resolved = {expert: self._key_to_slot[(layer, expert)] for expert in hit_set}
        loads: list[SlotLoad] = []
        evictions: list[SlotEviction] = []
        pinned = set(hit_keys)
        transient_experts: list[int] = []

        for expert in miss_order:
            key = (layer, expert)
            persistent_slot: int | None = None
            if (
                phase is RoutingPhase.PREFILL
                and expert in self._prefill_seed_candidates[layer]
                and self._layer_occupancy[layer] < self.prefill_slots_per_layer
            ):
                persistent_slot = self._empty_slot()
                self._prefill_seed_candidates[layer].discard(expert)
            elif phase is RoutingPhase.DECODE and self.persistent_slots:
                persistent_slot = self._empty_slot()
                if persistent_slot is None:
                    victim_slot = self._victim_slot(pinned=pinned)
                    if victim_slot is not None:
                        victim = self._slot_to_key[victim_slot]
                        assert victim is not None
                        if self.cache_policy == "lru" or self._score(key) > max(
                            1.0, self._score(victim)
                        ):
                            persistent_slot = victim_slot

            if persistent_slot is None:
                transient_experts.append(expert)
                continue
            self._assign(
                slot=persistent_slot,
                key=key,
                evictions=evictions,
                evicted_entries=_evicted_entries,
                occupancy_before=_occupancy_before,
            )
            pinned.add(key)
            resolved[expert] = persistent_slot
            loads.append(
                SlotLoad(
                    expert=expert,
                    slot=persistent_slot,
                    persistent=True,
                    generation=self._directory[key].generation,
                )
            )

        transient_base = self.persistent_slots
        for offset, expert in enumerate(transient_experts):
            slot = transient_base + offset
            resolved[expert] = slot
            loads.append(SlotLoad(expert=expert, slot=slot, persistent=False))

        if phase is RoutingPhase.DECODE:
            for key in hit_keys:
                self._history_for(key).last_used = self._decode_epoch

        generations = tuple(
            (
                self._directory[(layer, expert)].generation
                if resolved[expert] < self.persistent_slots
                else None
            )
            for expert in experts
        )
        return RoutePlan(
            phase=phase,
            experts=experts,
            slots=tuple(resolved[expert] for expert in experts),
            hits=tuple(expert for expert in unique_experts if expert in hit_set),
            misses=tuple(miss_order),
            loads=tuple(loads),
            evictions=tuple(evictions),
            generations=generations,
        )

    def plan_transaction(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
    ) -> tuple[RoutePlan, RoutePolicyTxn]:
        layer, experts = self._validate_experts(layer, expert_ids)
        route_keys = {(layer, expert) for expert in experts}
        history_keys = set(route_keys)
        if self.cache_policy == "frequency":
            # Frequency victim selection inherently scans every resident and
            # may lazily create history for one. Preserve that policy's exact
            # rollback semantics without charging the LRU hot path for it.
            history_keys.update(key for key in self._slot_to_key if key is not None)
        missing = object()
        histories: dict[tuple[int, int], object] = {}
        for key in history_keys:
            history = self._history.get(key)
            histories[key] = (
                missing
                if history is None
                else (history.score, history.score_epoch, history.last_used)
            )
        decode_epoch = self._decode_epoch
        lru_clock = self._lru_clock
        route_lru_ranks = {
            key: entry.lru_rank
            for key in route_keys
            if (entry := self._directory.get(key)) is not None
        }
        evictions = self._evictions
        cross_layer_evictions = self._cross_layer_evictions
        seed_candidates = set(self._prefill_seed_candidates[layer])
        evicted_entries: dict[int, tuple[tuple[int, int], _GlobalDirectoryEntry]] = {}
        occupancy_before: dict[int, tuple[bool, int]] = {}
        plan = self.plan(
            layer,
            experts,
            phase=phase,
            _evicted_entries=evicted_entries,
            _occupancy_before=occupancy_before,
        )

        def commit() -> None:
            self.publish_ready(layer, plan)

        def rollback() -> None:
            self._decode_epoch = decode_epoch
            empty_slots = [
                load.slot
                for load in plan.loads
                if load.persistent and load.slot not in evicted_entries
            ]
            for load in reversed(plan.loads):
                if not load.persistent or load.generation is None:
                    continue
                key = (layer, load.expert)
                if self._key_to_slot.get(key) == load.slot:
                    self._key_to_slot.pop(key, None)
                self._directory.pop(key, None)
                evicted = evicted_entries.get(load.slot)
                if evicted is None:
                    self._slot_to_key[load.slot] = None
                    self._free_slot_set.add(load.slot)
                else:
                    previous_key, previous_entry = evicted
                    self._slot_to_key[load.slot] = previous_key
                    self._key_to_slot[previous_key] = load.slot
                    self._directory[previous_key] = _GlobalDirectoryEntry(
                        previous_entry.slot,
                        previous_entry.generation,
                        previous_entry.state,
                        previous_entry.lru_rank,
                    )
                self._slot_generations[load.slot] = load.generation - 1
            # Restore empty slots at the exact front positions consumed by
            # _empty_slot(). An accepted partial load may already have
            # appended one of them during physical quarantine.
            for slot in empty_slots:
                try:
                    self._free_slots.remove(slot)
                except ValueError:
                    pass
            for slot in reversed(empty_slots):
                self._free_slots.appendleft(slot)
            for key, rank in route_lru_ranks.items():
                entry = self._directory.get(key)
                if entry is not None:
                    entry.lru_rank = rank
            self._lru_clock = lru_clock
            self._rebuild_lru()
            for affected_layer, (was_present, value) in occupancy_before.items():
                if was_present:
                    self._layer_occupancy[affected_layer] = value
                else:
                    self._layer_occupancy.pop(affected_layer, None)
            self._evictions = evictions
            self._cross_layer_evictions = cross_layer_evictions
            self._prefill_seed_candidates[layer] = set(seed_candidates)
            for key, values in histories.items():
                if values is missing:
                    self._history.pop(key, None)
                    continue
                score, score_epoch, last_used = values
                history = self._history.get(key)
                if history is None:
                    history = _ExpertHistory()
                    self._history[key] = history
                history.score = score
                history.score_epoch = score_epoch
                history.last_used = last_used

        return plan, RoutePolicyTxn(commit=commit, rollback=rollback)

    def try_plan_all_hits_transaction(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
    ) -> tuple[RoutePlan, RoutePolicyTxn] | None:
        """Plan a ready global route and defer policy updates until commit."""

        layer, experts = self._validate_experts_without_capacity(layer, expert_ids)
        phase = RoutingPhase(phase)
        unique_experts = tuple(dict.fromkeys(experts))
        keys = tuple((layer, expert) for expert in unique_experts)
        entries = tuple(self._directory.get(key) for key in keys)
        if any(entry is None or entry.state != "ready" for entry in entries):
            return None

        missing = object()
        histories: dict[tuple[int, int], object] = {}
        for key in keys:
            history = self._history.get(key)
            histories[key] = (
                missing
                if history is None
                else (history.score, history.score_epoch, history.last_used)
            )
        decode_epoch = self._decode_epoch
        lru_clock = self._lru_clock
        lru_ranks = {key: self._directory[key].lru_rank for key in keys}

        resolved = {
            expert: entry.slot
            for expert, entry in zip(unique_experts, entries, strict=True)
            if entry is not None
        }
        generations = {
            expert: entry.generation
            for expert, entry in zip(unique_experts, entries, strict=True)
            if entry is not None
        }
        plan = RoutePlan(
            phase=phase,
            experts=experts,
            slots=tuple(resolved[expert] for expert in experts),
            hits=unique_experts,
            misses=(),
            loads=(),
            evictions=(),
            generations=tuple(generations[expert] for expert in experts),
        )

        def commit() -> None:
            if phase is RoutingPhase.DECODE:
                self._decode_epoch += 1
                for expert in experts:
                    self._touch_decode((layer, expert))
                for key in keys:
                    self._history_for(key).last_used = self._decode_epoch
            if self.cache_policy == "lru":
                for key in keys:
                    self._touch_lru(key)

        def rollback() -> None:
            self._decode_epoch = decode_epoch
            for key, rank in lru_ranks.items():
                self._directory[key].lru_rank = rank
            self._lru_clock = lru_clock
            self._rebuild_lru()
            for key, values in histories.items():
                if values is missing:
                    self._history.pop(key, None)
                    continue
                score, score_epoch, last_used = values
                history = self._history.get(key)
                if history is None:
                    history = _ExpertHistory()
                    self._history[key] = history
                history.score = score
                history.score_epoch = score_epoch
                history.last_used = last_used

        return plan, RoutePolicyTxn(commit=commit, rollback=rollback)

    def reset(self) -> None:
        self._decode_epoch = 0
        self._slot_to_key = [None] * self.persistent_slots
        self._key_to_slot.clear()
        self._directory.clear()
        # Physical reset empties slots without rewinding their generations.
        # Preserve the matching policy watermarks for the next assignment.
        self._free_slots = deque(range(self.persistent_slots))
        self._free_slot_set = set(range(self.persistent_slots))
        self._lru.clear()
        self._lru_clock = 0
        self._history.clear()
        self._layer_occupancy.clear()
        self._evictions = 0
        self._cross_layer_evictions = 0
        for candidates in self._prefill_seed_candidates.values():
            candidates.clear()

    def snapshot(self) -> dict[str, object]:
        return {
            "capacity": self.persistent_slots,
            "occupancy": self.occupancy,
            "occupancy_by_layer": self.occupancy_by_layer,
            "evictions": self._evictions,
            "cross_layer_evictions": self._cross_layer_evictions,
            # Records are read directly into their final fixed slots. There is
            # no live-weight compaction or relocation copy on this path.
            "relocation_bytes": 0,
        }


@dataclass
class ExpertCacheSimulation:
    """Multi-layer cache simulator and aggregate I/O accounting."""

    expert_count: int
    persistent_slots: int
    transient_slots: int
    expert_record_bytes: int
    allocated_layer_count: int | None = None
    frequency_decay: float = 0.995
    counters: CacheCounters = field(default_factory=CacheCounters)
    _layers: dict[int, LayerExpertSlotBank] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.expert_count = _integer("expert_count", self.expert_count, minimum=1)
        self.persistent_slots = _integer(
            "persistent_slots", self.persistent_slots, minimum=0
        )
        self.transient_slots = _integer(
            "transient_slots", self.transient_slots, minimum=1
        )
        self.expert_record_bytes = _integer(
            "expert_record_bytes", self.expert_record_bytes, minimum=1
        )
        if self.persistent_slots > self.expert_count:
            raise ValueError("persistent_slots cannot exceed expert_count")
        if self.allocated_layer_count is not None:
            self.allocated_layer_count = _integer(
                "allocated_layer_count", self.allocated_layer_count, minimum=1
            )
        if isinstance(self.frequency_decay, bool):
            raise TypeError("frequency_decay must be a finite number")
        try:
            self.frequency_decay = float(self.frequency_decay)
        except (TypeError, ValueError) as exc:
            raise TypeError("frequency_decay must be a finite number") from exc
        if not isfinite(self.frequency_decay) or not 0.0 < self.frequency_decay <= 1.0:
            raise ValueError("frequency_decay must be in (0, 1]")

    def layer(self, layer_index: int) -> LayerExpertSlotBank:
        layer_index = _integer("layer_index", layer_index, minimum=0)
        if layer_index not in self._layers:
            if (
                self.allocated_layer_count is not None
                and len(self._layers) >= self.allocated_layer_count
            ):
                raise ValueError("trace exceeds allocated_layer_count")
            self._layers[layer_index] = LayerExpertSlotBank(
                expert_count=self.expert_count,
                persistent_slots=self.persistent_slots,
                transient_slots=self.transient_slots,
                frequency_decay=self.frequency_decay,
            )
        return self._layers[layer_index]

    def observe(
        self,
        *,
        layer_index: int,
        expert_ids: Iterable[int],
        phase: RoutingPhase | str,
    ) -> RoutePlan:
        plan = self.layer(layer_index).plan(expert_ids, phase=phase)
        self.counters.observe(plan, expert_record_bytes=self.expert_record_bytes)
        return plan

    def summary(self, *, effective_ssd_bytes_per_second: float) -> dict[str, object]:
        if isinstance(effective_ssd_bytes_per_second, bool):
            raise TypeError("effective_ssd_bytes_per_second must be finite")
        try:
            effective_ssd_bytes_per_second = float(effective_ssd_bytes_per_second)
        except (TypeError, ValueError) as exc:
            raise TypeError("effective_ssd_bytes_per_second must be finite") from exc
        if (
            not isfinite(effective_ssd_bytes_per_second)
            or effective_ssd_bytes_per_second <= 0
        ):
            raise ValueError("effective_ssd_bytes_per_second must be positive")
        counters = self.counters.as_dict()
        layer_count = (
            len(self._layers)
            if self.allocated_layer_count is None
            else self.allocated_layer_count
        )
        return {
            **counters,
            "estimated_io_seconds": self.counters.bytes_read
            / effective_ssd_bytes_per_second,
            "layers_observed": len(self._layers),
            "allocated_layer_count": layer_count,
            "persistent_cache_scope": (
                "observed_layers_only"
                if self.allocated_layer_count is None
                else "configured_model"
            ),
            "persistent_cache_bytes": layer_count
            * self.persistent_slots
            * self.expert_record_bytes,
            "observed_layer_cache_bytes": len(self._layers)
            * self.persistent_slots
            * self.expert_record_bytes,
            # Native execution reuses one top-k scratch bank across sequential
            # layers; it is intentionally not multiplied by layer count.
            "transient_scratch_bytes": self.transient_slots * self.expert_record_bytes,
            "resident_experts_by_layer": {
                str(layer): list(bank.resident_experts)
                for layer, bank in sorted(self._layers.items())
            },
        }
