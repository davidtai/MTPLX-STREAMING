"""Explicit packed decode lane with gate/up readiness before full record readiness.

Installed only at a quiescent, post-prefill boundary. Slot READY publication,
policy transactions, pins and deferred releases remain owned by the runtime.
An early witness covers only gate/up components whose writers have completed.
"""
from concurrent.futures import Future, as_completed
from dataclasses import dataclass
import threading

import mlx.core as mx
from mtplx.expert_streaming import RoutingPhase
from mtplx.models.expert_mlx import _clamped_swiglu, _DeferredSplitClose
from paired_kernels import make_projection


class PlanePart:
    def __init__(self, plan):
        self.plan = plan
        self.expected = frozenset(load.expert for load in plan.loads)
        self.buffers = {}
        self.publish_lock = threading.Lock()
        self.gate_up_ready = Future()
        self.phase = 'waiting'
        self.groups = []

    def publish_read_components(self, items):
        with self.publish_lock:
            self.buffers.update((record.expert, dest) for record, dest in items)
            if self.expected <= self.buffers.keys() and not self.gate_up_ready.done():
                self.gate_up_ready.set_result(tuple(self.buffers.items()))

    def finish(self, ready):
        # A physically resident part may need no read. Its normal READY result
        # is also a complete gate/up witness; it uses the same split compute.
        if not self.gate_up_ready.done():
            self.gate_up_ready.set_result(tuple({b.expert:b.buffer for b in ready.bindings}.items()))


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


def bind_reader(reader, local):
    read_range = reader._readv_range_into
    submit = reader._fanout_executor.submit
    metrics = reader.metrics
    names = ('gate_proj.weight', 'up_proj.weight', 'down_proj.weight')
    offsets = (0, 6266880, 12533760)

    def run(items, cancel_event, deadline_ns, pipeline_phase):
        part = local.part
        jobs, futures = [], []
        error = None
        try:
            for record, dest in items:
                for i, (name, offset) in enumerate(zip(names, offsets)):
                    jobs.append((i == 2, record.sidecar_offset+offset, dest.component_view(name)))
            jobs.sort(key=lambda j:j[0])
            metrics.update(record_requests=len(items), records_read=len(items), sidecar_record_requests=len(items))

            def read(job):
                _, offset, view = job
                read_range('experts.bin', offset, (view,), cancel_event=cancel_event,
                           deadline_ns=deadline_ns, pipeline_phase=pipeline_phase)

            try:
                for job in jobs[1:]:
                    futures.append((job[0], submit(read, job)))
                read(jobs[0])
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
            return ('unverified',) * len(items)
        finally:
            for _, _, view in jobs:
                view.release()

    def read_one(manifest, record, destination, *, prefer_sidecar=True, verify_hash=True,
                 cancel_event=None, deadline_ns=None, pipeline_phase=None):
        return run(((record,destination),), cancel_event, deadline_ns, pipeline_phase)[0]

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
    def __init__(self, scales):
        self.gu_kernel = make_projection(2304,5120)
        self.down_kernel = make_projection(5120,2304)
        self.gs, self.us, self.ds = (scales[p] for p in ('gate_proj','up_proj','down_proj'))

    def gate_up(self, tokens, experts, buffers):
        groups = {}
        for pos, expert in enumerate(experts):
            if expert in buffers:
                dest = buffers[expert]
                groups.setdefault(id(dest.bank), []).append((pos,expert,dest))
        work = []
        for group in groups.values():
            positions = [v[0] for v in group]
            bank = group[0][2].bank
            pairs = mx.array([(v[2].bank_index,v[1]) for v in group], mx.int32)
            rows = len(group)
            x = mx.take(tokens,mx.array([p//6 for p in positions],mx.int32),axis=0).reshape(rows,1,1,5120)
            args = dict(template=[('T',mx.bfloat16)],grid=(32,576,rows),threadgroup=(32,2,1),
                        output_shapes=[(rows,1,1,2304)],output_dtypes=[mx.bfloat16])
            g = self.gu_kernel(inputs=[x,pairs,bank.arrays['gate_proj.weight'],*self.gs],**args)[0]
            u = self.gu_kernel(inputs=[x,pairs,bank.arrays['up_proj.weight'],*self.us],**args)[0]
            h = _clamped_swiglu(g,u,10.0)
            work.append(GateUpWork(positions,bank,pairs,h))
        return work

    def down(self, group):
        rows = len(group.positions)
        return self.down_kernel(inputs=[group.hidden,group.pairs,group.bank.arrays['down_proj.weight'],*self.ds],
            template=[('T',mx.bfloat16)],grid=(32,1280,rows),threadgroup=(32,2,1),
            output_shapes=[(rows,1,1,5120)],output_dtypes=[mx.bfloat16])[0].reshape(rows,5120)


class PackedDecode:
    def __init__(self, runtime, layer, ops, executor, *, early):
        self.runtime, self.layer, self.ops, self.executor = runtime, layer, ops, executor
        self.completions = self.early_completions if early else self.complete_completions

    @staticmethod
    def early_completions(parts):
        owners = {p.gate_up_ready:('gu',p) for _,p in parts}
        owners.update((f,('full',p)) for f,p in parts)
        for done in as_completed(owners):
            yield owners[done]

    @staticmethod
    def complete_completions(parts):
        owners = dict(parts)
        for done in as_completed(owners):
            yield 'full',owners[done]

    def run(self, x, indices, *, shared_work):
        runtime, ops = self.runtime, self.ops
        tokens = x.reshape(-1,5120)
        mx.eval(indices)
        runtime.flush_deferred_slot_releases()
        experts = tuple(int(e) for e in indices.reshape(-1).tolist())
        runtime.observe_route(self.layer,RoutingPhase.DECODE,experts,token_count=int(tokens.shape[0]))
        self.executor.parts.clear()
        pending = runtime.begin_split_route(self.layer,experts,phase=RoutingPhase.DECODE)
        parts = tuple(self.executor.parts)
        by_key = {frozenset(p.plan.experts):p for _,p in parts}
        outputs, positions, submitted, leased = [], [], [], []
        shared = None

        def submit_gu(part):
            if part.phase != 'waiting':
                return
            buffers = dict(part.gate_up_ready.result())
            part.groups = ops.gate_up(tokens,experts,buffers)
            roots = [g.hidden for g in part.groups]
            mx.async_eval(roots)
            submitted.extend(roots)
            part.phase = 'submitted'

        def finish(groups):
            wave = [ops.down(g) for g in groups]
            mx.async_eval(wave)
            outputs.extend(wave)
            positions.extend(p for g in groups for p in g.positions)

        try:
            if pending.hit_ready is not None:
                buffers = {b.expert:b.buffer for b in pending.hit_ready.bindings}
                finish(ops.gate_up(tokens,experts,buffers))
            if shared_work is not None:
                shared = shared_work()
                mx.async_eval(shared)
            ready_iter = pending.iter_ready_misses()
            for kind, part in self.completions(parts):
                if kind == 'gu':
                    submit_gu(part)
                    continue
                # The runtime iterator performs normal full-record publication,
                # policy commit and pin accounting. It may choose a different
                # already-complete part; match the actual yielded record set.
                ready = next(ready_iter)
                part = by_key[frozenset(ready.plan.experts)]
                submit_gu(part)
                finish(part.groups)
                leased.append(ready)
                part.groups.clear()
                part.phase = 'consumed'
            # Complete the iterator's final policy bookkeeping, including the
            # all-hit case where the event list was empty.
            for _ in ready_iter:
                raise RuntimeError('miss completion inventory was not exhausted')
            runtime.defer_slot_release(_DeferredSplitClose(pending,tuple(leased)),tuple(outputs))
        except BaseException as error:
            # Early GU consumers cannot outlive a failed slot transaction.
            # Drain submitted work before abort can recycle any of its owners.
            try:
                mx.synchronize()
            finally:
                pending.abort(error)
                pending.close()
            raise
        finally:
            self.executor.parts.clear()
        joined = mx.concatenate(outputs,axis=0)
        order = mx.argsort(mx.array(positions,mx.int32))
        return mx.take(joined,order,axis=0).reshape((*indices.shape,5120)),shared


def install(runtime, switches, scales_by_layer, *, early=True):
    """One-time installation for a declared M<=8, single-wave decode schedule."""
    s,c,p = runtime.spec,runtime.config,runtime.plan
    if ((s.hidden_size,s.expert_hidden_size,s.top_k,s.quant_bits,s.quant_group_size,s.expert_codec,s.swiglu_limit)
            != (5120,2304,6,4,32,'mxfp4',10.0)
            or c.slot_layout!='component-banks' or c.cache_scope!='layer'
            or c.decode_miss_records_per_part!=3 or c.prefetch_slots or c.resource_telemetry
            or runtime._pipeline_ledger is not None or not runtime._single_slot_pool
            or c.split_route_release!='deferred' or p.transient_slots<48
            or set(switches)!=set(scales_by_layer) or set(switches)!=set(s.routed_layer_indices)
            or runtime.reader._fanout_executor is None):
        raise RuntimeError('packed plane lane requires its proven fixed decode configuration')
    mx.synchronize()
    runtime.flush_deferred_slot_releases()
    local = threading.local()
    executor = PartExecutor(runtime._split_executor,local)
    runtime._split_executor = executor
    runtime.slots._executor = ReaderExecutor(runtime.slots._executor, local)
    bind_reader(runtime.reader,local)
    runners = {}
    for layer,switch in switches.items():
        runner = PackedDecode(runtime,layer,PackedOps(scales_by_layer[layer]),executor,early=early)
        switch._run = runner.run
        runners[layer] = runner
    return runners
