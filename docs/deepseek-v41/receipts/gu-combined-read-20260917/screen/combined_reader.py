from concurrent.futures import Future
import threading

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
                jobs.append((False, record.sidecar_offset,
                    (dest.component_view(names[0]), dest.component_view('gate_gap'),
                     dest.component_view(names[1]))))
                jobs.append((True, record.sidecar_offset + offsets[2],
                    (dest.component_view(names[2]),)))
            jobs.sort(key=lambda j:j[0])
            metrics.update(record_requests=len(items), records_read=len(items), sidecar_record_requests=len(items))

            def read(job):
                _, offset, views = job
                read_range('experts.bin', offset, views, cancel_event=cancel_event,
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
            for _, _, views in jobs:
                for view in views:
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
