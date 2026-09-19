"""F2 next-layer prefetch decode lane for the FULL 40-layer DeepSeek-V4.1 runtime.

Derived from the GPU-proven ridge-prefetch v2 lane (docs/deepseek-v41/receipts/
ridge-prefetch-20260919/v2/plane_lane.py) with the three fixes its flat screens
called for:

  fix #1  the isolated priority reader has >= the native fanout pool's worker count
          (a constructor arg, defaulted at install to ``reader._fanout_pool_workers``),
          so the candidate arm reads with at least the control's parallelism. Codex
          hard-coded four workers.
  fix #2  ``Issue.prepare`` does ONLY the device prediction and evaluates it on the
          source route's existing indices barrier; all host ranking / READY filtering
          / issue happen in ``Issue.__call__`` after the demand reads are submitted
          (see f2/issue.py). Barriers per layer call are unchanged: 1 routing (the
          ``mx.eval(indices, pred)`` inside prepare) + 1 miss-drain (the completion
          loop). ``prefetch_experts`` submits async reads and adds no mx barrier; the
          reconcile await inside ``begin_split_route`` is an I/O join, not a barrier.
  fix #3  the predictor is parameter-free (f2/issue.py): device biased-gate max over
          rows, host top-3 not-READY, no learned correction and no margin threshold.

Installed once at a quiescent post-prefill boundary. Slot READY publication, policy
transactions, pins and deferred releases remain owned by the runtime. Speculative
records are warmed through the shipped ``runtime.prefetch_experts`` -> shared
``GlobalPrefetchRing`` -> ``slots.load_speculative`` path; the true route always
gathers from the TRUE ``indices``, so a mispredict only wastes a speculative read and
can never change a logit (mtplx/expert_streaming.py: ``LayerExpertSlotBank.plan``
resolves committed ring hits in place before the miss set, :1553-1589).

This module is the CANDIDATE lane. The stock CONTROL is the retained
``sources/packed/plane_lane.py`` ``PackedDecode`` (prefetch_slots == 0), byte-for-byte
untouched when the candidate is off. There is NO eligible-or-stock / try-then-fallback
branch in the enabled hot path: the lane is chosen once, at install, by which
``switch._run`` is bound (AGENTS.md).
"""
from __future__ import annotations

from collections import namedtuple
from concurrent.futures import Future, as_completed
from dataclasses import dataclass
import threading

import mlx.core as mx
from mtplx.expert_streaming import RoutingPhase
from mtplx.models.expert_mlx import _clamped_swiglu, _DeferredSplitClose

from .priority_reads import PriorityReads, native_worker_count


# The fixed-shape / fixed-codec contract an ops object was compiled for. install()
# refuses unless the runtime spec matches it exactly (AGENTS.md: encode proven
# invariants in installed types; validate once at construction). The production
# ``PackedOps`` pins (5120, 2304, 6, 4, 32, mxfp4, 10.0) -- the geometry its Metal
# kernels require; a CPU-test ops object declares its own tiny geometry so the same
# install() gate can be exercised on a real runtime without Metal.
OpsContract = namedtuple(
    "OpsContract",
    "hidden_size expert_hidden_size top_k quant_bits quant_group_size expert_codec swiglu_limit",
)

# The retained production geometry (sources/packed/plane_lane.py install gate).
PRODUCTION_CONTRACT = OpsContract(5120, 2304, 6, 4, 32, "mxfp4", 10.0)

# Weight-only plane byte offsets inside one packed mxfp4 record (experts.bin sidecar):
# gate @0, up @6,266,880, down @12,533,760; record end 17,694,720
# (sources/packed/packed_storage.py, packed_admission.py WEIGHTS).
_PLANE_NAMES = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
_PLANE_OFFSETS = (0, 6_266_880, 12_533_760)


class PlanePart:
    """A demand miss part: reads at demand priority 0 and publishes its GU witness."""

    read_priority = 0

    @staticmethod
    def read_first(read, job, submit):
        # The demand path reads its own first sub-range inline (never queued behind a
        # speculative backlog); the tail jobs are submitted at demand priority.
        read(job)

    def __init__(self, plan):
        self.plan = plan
        self.expected = frozenset(load.expert for load in plan.loads)
        self.buffers = {}
        self.publish_lock = threading.Lock()
        self.gate_up_ready = Future()
        self.phase = "waiting"
        self.groups = []

    def publish_read_components(self, items):
        with self.publish_lock:
            self.buffers.update((record.expert, dest) for record, dest in items)
            if self.expected <= self.buffers.keys() and not self.gate_up_ready.done():
                self.gate_up_ready.set_result(tuple(self.buffers.items()))

    def finish(self, ready):
        # A physically resident part may need no read. Its normal READY result is also
        # a complete gate/up witness; it uses the same split compute.
        if not self.gate_up_ready.done():
            self.gate_up_ready.set_result(
                tuple({b.expert: b.buffer for b in ready.bindings}.items())
            )


class IgnoreGUPublication:
    """Witness for a speculative ring read: priority 1, no gate/up publication."""

    read_priority = 1

    @staticmethod
    def read_first(read, job, submit):
        # A speculative read has no inline-first optimisation: submit at priority 1 and
        # wait, so a concurrent demand read (priority 0) preempts the queue.
        submit(read, job, priority=1).result()

    def publish_read_components(self, items):
        pass


class PartExecutor:
    def __init__(self, executor, local):
        self.executor, self.local = executor, local
        self.parts = []

    def submit(self, fn, layer, plan, **kwargs):
        part = PlanePart(plan)

        def execute():
            self.local.part = part
            try:
                ready = fn(layer, plan, **kwargs)
                part.finish(ready)
                return ready
            except BaseException as error:
                if not part.gate_up_ready.done():
                    part.gate_up_ready.set_exception(error)
                raise
            finally:
                del self.local.part

        future = self.executor.submit(execute)
        self.parts.append((future, part))
        return future

    def shutdown(self, *args, **kwargs):
        try:
            return self.executor.shutdown(*args, **kwargs)
        finally:
            self.parts.clear()


class ReaderExecutor:
    """Carry the part witness through the pool's existing I/O executor."""

    def __init__(self, executor, local):
        self.executor, self.local = executor, local

    def submit(self, fn, *args, **kwargs):
        part = self.local.part

        def execute():
            self.local.part = part
            try:
                return fn(*args, **kwargs)
            finally:
                del self.local.part

        return self.executor.submit(execute)

    def shutdown(self, *args, **kwargs):
        return self.executor.shutdown(*args, **kwargs)


class SpeculativeExecutor:
    """Wrap the runtime's prefetch executor so ring reads carry the priority-1
    ignore-publication witness on their worker thread."""

    def __init__(self, executor, local):
        self.executor, self.local = executor, local
        self.witness = IgnoreGUPublication()

    def submit(self, fn, *args, **kwargs):
        def execute():
            self.local.part = self.witness
            try:
                return fn(*args, **kwargs)
            finally:
                del self.local.part

        return self.executor.submit(execute)

    def shutdown(self, *args, **kwargs):
        return self.executor.shutdown(*args, **kwargs)


def bind_priority_reader(reader, local):
    """Rebind the reader's record/component reads to a plane-split, priority-aware
    path over the isolated ``PriorityReads`` pool. gate/up planes read first (early
    GU witness), down last. Demand parts read at priority 0, speculative at 1."""

    read_range = reader._readv_range_into
    submit = reader._fanout_executor.submit
    metrics = reader.metrics
    names = _PLANE_NAMES
    offsets = _PLANE_OFFSETS

    def run(items, cancel_event, deadline_ns, pipeline_phase):
        part = local.part
        jobs, futures = [], []
        error = None
        try:
            for record, dest in items:
                for i, (name, offset) in enumerate(zip(names, offsets)):
                    jobs.append(
                        (i == 2, record.sidecar_offset + offset, dest.component_view(name))
                    )
            jobs.sort(key=lambda j: j[0])
            metrics.update(
                record_requests=len(items),
                records_read=len(items),
                sidecar_record_requests=len(items),
            )

            def read(job):
                _, offset, view = job
                read_range(
                    "experts.bin",
                    offset,
                    (view,),
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                    pipeline_phase=pipeline_phase,
                )

            try:
                for job in jobs[1:]:
                    futures.append((job[0], submit(read, job, priority=part.read_priority)))
                part.read_first(read, jobs[0], submit)
                for down, future in futures:
                    if not down:
                        future.result()
                part.publish_read_components(items)
            except BaseException as exc:
                error = exc
            # Publication of a full record is unchanged: all writers, including
            # failed/cancelled siblings, finish before views or slots can release.
            for _, future in futures:
                try:
                    future.result()
                except BaseException as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error
            metrics.update(records_unhashed=len(items))
            return ("unverified",) * len(items)
        finally:
            for _, _, view in jobs:
                view.release()

    def read_one(manifest, record, destination, *, prefer_sidecar=True, verify_hash=True,
                 cancel_event=None, deadline_ns=None, pipeline_phase=None):
        return run(((record, destination),), cancel_event, deadline_ns, pipeline_phase)[0]

    def read_batch(manifest, items, *, verify_hash=True, cancel_event=None,
                   deadline_ns=None, pipeline_phase=None):
        if not items:
            return ()
        return run(items, cancel_event, deadline_ns, pipeline_phase)

    reader.read_record_into = read_one
    reader.read_component_records_into = read_batch


@dataclass
class GateUpWork:
    positions: list
    bank: object
    pairs: object
    hidden: object


class PackedOps:
    """Production Metal component-bank projection ops (mxfp4, 5120/2304).

    ``make_projection`` (sources/packed/paired_kernels.py) is a ``mx.fast.metal_kernel``
    that only admits the (2304,5120)/(5120,2304) geometry, so the import is lazy: this
    module imports on CPU (for the seam tests) without pulling in Metal, and the kernels
    are built only when the production ops are constructed for a GPU window.
    """

    contract = PRODUCTION_CONTRACT

    def __init__(self, scales):
        from paired_kernels import make_projection  # lazy: Metal-only, window/GPU-smoke

        self.gu_kernel = make_projection(2304, 5120)
        self.down_kernel = make_projection(5120, 2304)
        self.gs, self.us, self.ds = (scales[p] for p in ("gate_proj", "up_proj", "down_proj"))

    def gate_up(self, tokens, experts, buffers):
        groups = {}
        for pos, expert in enumerate(experts):
            if expert in buffers:
                dest = buffers[expert]
                groups.setdefault(id(dest.bank), []).append((pos, expert, dest))
        work = []
        for group in groups.values():
            positions = [v[0] for v in group]
            bank = group[0][2].bank
            pairs = mx.array([(v[2].bank_index, v[1]) for v in group], mx.int32)
            rows = len(group)
            x = mx.take(tokens, mx.array([p // 6 for p in positions], mx.int32), axis=0).reshape(rows, 1, 1, 5120)
            args = dict(template=[("T", mx.bfloat16)], grid=(32, 576, rows), threadgroup=(32, 2, 1),
                        output_shapes=[(rows, 1, 1, 2304)], output_dtypes=[mx.bfloat16])
            g = self.gu_kernel(inputs=[x, pairs, bank.arrays["gate_proj.weight"], *self.gs], **args)[0]
            u = self.gu_kernel(inputs=[x, pairs, bank.arrays["up_proj.weight"], *self.us], **args)[0]
            h = _clamped_swiglu(g, u, 10.0)
            work.append(GateUpWork(positions, bank, pairs, h))
        return work

    def down(self, group):
        rows = len(group.positions)
        return self.down_kernel(
            inputs=[group.hidden, group.pairs, group.bank.arrays["down_proj.weight"], *self.ds],
            template=[("T", mx.bfloat16)], grid=(32, 1280, rows), threadgroup=(32, 2, 1),
            output_shapes=[(rows, 1, 1, 5120)], output_dtypes=[mx.bfloat16])[0].reshape(rows, 5120)


class PackedDecode:
    """The retained control runner (stock). Byte-for-byte the retained lane's run."""

    def __init__(self, runtime, layer, ops, executor, *, early):
        self.runtime, self.layer, self.ops, self.executor = runtime, layer, ops, executor
        self.completions = self.early_completions if early else self.complete_completions

    @staticmethod
    def early_completions(parts):
        owners = {p.gate_up_ready: ("gu", p) for _, p in parts}
        owners.update((f, ("full", p)) for f, p in parts)
        for done in as_completed(owners):
            yield owners[done]

    @staticmethod
    def complete_completions(parts):
        owners = dict(parts)
        for done in as_completed(owners):
            yield "full", owners[done]

    def _open_route(self, x, indices):
        """Hook: perform the layer's routing barrier. The control evaluates indices;
        the prefetch subclass rides the SAME barrier to also force its prediction."""
        mx.eval(indices)

    def _after_demand_submitted(self):
        """Hook: run after ``begin_split_route`` has submitted the demand reads. The
        control does nothing; the prefetch subclass issues its speculative reads here."""

    def run(self, x, indices, *, shared_work):
        runtime, ops = self.runtime, self.ops
        tokens = x.reshape(-1, 5120)
        self._open_route(x, indices)
        runtime.flush_deferred_slot_releases()
        experts = tuple(int(e) for e in indices.reshape(-1).tolist())
        runtime.observe_route(self.layer, RoutingPhase.DECODE, experts, token_count=int(tokens.shape[0]))
        self.executor.parts.clear()
        pending = runtime.begin_split_route(self.layer, experts, phase=RoutingPhase.DECODE)
        parts = tuple(self.executor.parts)
        by_key = {frozenset(p.plan.experts): p for _, p in parts}
        outputs, positions, submitted, leased = [], [], [], []
        shared = None

        def submit_gu(part):
            if part.phase != "waiting":
                return
            buffers = dict(part.gate_up_ready.result())
            part.groups = ops.gate_up(tokens, experts, buffers)
            roots = [g.hidden for g in part.groups]
            mx.async_eval(roots)
            submitted.extend(roots)
            part.phase = "submitted"

        def finish(groups):
            wave = [ops.down(g) for g in groups]
            mx.async_eval(wave)
            outputs.extend(wave)
            positions.extend(p for g in groups for p in g.positions)

        try:
            if pending.hit_ready is not None:
                buffers = {b.expert: b.buffer for b in pending.hit_ready.bindings}
                finish(ops.gate_up(tokens, experts, buffers))
            if shared_work is not None:
                shared = shared_work()
                mx.async_eval(shared)
            # Demand parts already exist and resident/shared GPU roots are enqueued.
            # The prefetch subclass starts speculative reads here, before a miss GU.
            self._after_demand_submitted()
            ready_iter = pending.iter_ready_misses()
            for kind, part in self.completions(parts):
                if kind == "gu":
                    submit_gu(part)
                    continue
                # The runtime iterator performs normal full-record publication, policy
                # commit and pin accounting. It may choose a different already-complete
                # part; match the actual yielded record set.
                ready = next(ready_iter)
                part = by_key[frozenset(ready.plan.experts)]
                submit_gu(part)
                finish(part.groups)
                leased.append(ready)
                part.groups.clear()
                part.phase = "consumed"
            # Complete the iterator's final policy bookkeeping, including the all-hit
            # case where the event list was empty.
            for _ in ready_iter:
                raise RuntimeError("miss completion inventory was not exhausted")
            runtime.defer_slot_release(_DeferredSplitClose(pending, tuple(leased)), tuple(outputs))
        except BaseException as error:
            # Early GU consumers cannot outlive a failed slot transaction. Drain
            # submitted work before abort can recycle any of its owners.
            try:
                mx.synchronize()
            finally:
                pending.abort(error)
                pending.close()
            raise
        finally:
            self.executor.parts.clear()
        joined = mx.concatenate(outputs, axis=0)
        order = mx.argsort(mx.array(positions, mx.int32))
        return mx.take(joined, order, axis=0).reshape((*indices.shape, 5120)), shared


class PrefetchDecode(PackedDecode):
    """The candidate runner: identical to ``PackedDecode`` except its routing barrier
    also forces the next-layer prediction, and it issues the speculative reads after
    the demand reads are submitted. The gather is unchanged (TRUE indices only)."""

    def __init__(self, *args, issue, **kwargs):
        super().__init__(*args, **kwargs)
        self.issue = issue

    def _open_route(self, x, indices):
        # fix #2: ride the layer's own indices barrier; ``prepare`` forces indices AND
        # the device prediction in one ``mx.eval`` -- no extra host sync.
        self.issue.prepare(x.reshape(-1, 5120), indices)

    def _after_demand_submitted(self):
        # fix #2/#3: host ranking + READY filter + single ``prefetch_experts`` issue,
        # AFTER ``begin_split_route`` submitted the demand reads.
        self.issue()


def install(runtime, switches, ops_by_layer, *, prefetch_sources, early=True, reader_workers=None):
    """One-time installation of the F2 prefetch lane for the M<=8 verify decode.

    Mirrors ``sources/packed/plane_lane.py`` ``install``'s checks but REQUIRES a
    prefetch ring (``config.prefetch_slots == plan.prefetch_ring_slots == R > 0``) and
    validates the fixed geometry/codec against the ops objects' contract (the ops are
    the fixed-shape entrypoints that carry the invariant, so a production ops object
    pins 5120/2304/6/mxfp4 while a CPU-test ops object declares its own tiny geometry).

    ``switches``          {layer: switch} for all routed layers.
    ``ops_by_layer``      {layer: ops} (one shared geometry/codec contract).
    ``prefetch_sources``  {source_layer: Issue} for the layers that predict (3..38);
                          the remaining layers run the stock ``PackedDecode``.
    ``reader_workers``    isolated priority-reader worker count; default = the native
                          fanout pool's worker count read here (fix #1). Must be >= it.
    """
    s, c, p = runtime.spec, runtime.config, runtime.plan
    ring = c.prefetch_slots
    contracts = {ops.contract for ops in ops_by_layer.values()}
    if len(contracts) != 1:
        raise RuntimeError("f2 prefetch lane requires one shared ops geometry/codec contract")
    contract = next(iter(contracts))
    if ((s.hidden_size, s.expert_hidden_size, s.top_k)
            != (contract.hidden_size, contract.expert_hidden_size, contract.top_k)
            or (s.quant_bits, s.quant_group_size, s.expert_codec, s.swiglu_limit)
            != (contract.quant_bits, contract.quant_group_size, contract.expert_codec, contract.swiglu_limit)
            or c.slot_layout != "component-banks" or c.cache_scope != "layer"
            or c.decode_miss_records_per_part != 3 or not isinstance(ring, int) or ring <= 0
            or p.prefetch_ring_slots != ring or c.resource_telemetry
            or runtime._pipeline_ledger is not None or not runtime._single_slot_pool
            or c.split_route_release != "deferred" or p.transient_slots < 48
            or set(switches) != set(ops_by_layer) or set(switches) != set(s.routed_layer_indices)
            or runtime.reader._fanout_executor is None):
        raise RuntimeError("f2 prefetch lane requires the retained fixed decode configuration with a ring")
    if prefetch_sources is None or not (set(prefetch_sources) <= set(switches)):
        raise RuntimeError("prefetch_sources must be a subset of the routed switches")
    native_workers = native_worker_count(runtime.reader)
    if reader_workers is None:
        reader_workers = native_workers
    elif int(reader_workers) < native_workers:
        raise RuntimeError(
            f"reader_workers {reader_workers} is below the native fanout pool "
            f"({native_workers}); the candidate must not read with less parallelism"
        )
    mx.synchronize()
    runtime.flush_deferred_slot_releases()
    local = threading.local()
    executor = PartExecutor(runtime._split_executor, local)
    runtime._split_executor = executor
    runtime.slots._executor = ReaderExecutor(runtime.slots._executor, local)
    runtime._prefetch_executor = SpeculativeExecutor(runtime._prefetch_executor, local)
    runtime.reader._fanout_executor.shutdown(wait=True)
    runtime.reader._fanout_executor = PriorityReads(int(reader_workers))
    bind_priority_reader(runtime.reader, local)
    runners = {}
    for layer, switch in switches.items():
        ops = ops_by_layer[layer]
        if layer in prefetch_sources:
            runner = PrefetchDecode(runtime, layer, ops, executor, early=early, issue=prefetch_sources[layer])
        else:
            runner = PackedDecode(runtime, layer, ops, executor, early=early)
        switch._run = runner.run
        runners[layer] = runner
    return runners
