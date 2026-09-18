"""Phase-bound packed-scale ownership and native weight-only reads."""
import fcntl
import hashlib
import os
from pathlib import Path

WEIGHT_BYTES = 17694720
SCALE_BYTES = 1105920
PROJECTIONS = ('gate_proj', 'up_proj', 'down_proj')


def load_layer(root, entry, *, mx):
    """Read directly into the final owned Metal allocation, one file at a time."""
    arrays = {}
    for projection in PROJECTIONS:
        component = entry['components'][projection]
        values = []
        for field in ('descriptors', 'payload', 'bases'):
            meta = component[field]
            value = mx.zeros(tuple(meta['shape']), dtype=mx.uint32)
            mx.eval(value)
            raw = memoryview(value).cast('B')
            fd = os.open(Path(root) / meta['file'], os.O_RDONLY | os.O_NOFOLLOW)
            try:
                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                # Darwin SDK sys/fcntl.h: F_RDAHEAD=45; Python omits the name.
                fcntl.fcntl(fd, 45, 0)
                if os.fstat(fd).st_size != meta['bytes'] or raw.nbytes != meta['bytes']:
                    raise RuntimeError('packed scale size differs from inventory')
                cursor = 0
                while cursor < len(raw):
                    end = min(cursor + 8 * 1024**2, len(raw))
                    count = os.preadv(fd, [raw[cursor:end]], cursor)
                    if count <= 0:
                        raise RuntimeError('short packed scale read')
                    cursor += count
                if hashlib.sha256(raw).hexdigest() != meta['sha256']:
                    raise RuntimeError('packed scale artifact digest differs')
            finally:
                os.close(fd)
                raw.release()
            values.append(value)
        arrays[projection] = tuple(values)
    return arrays


def remove_raw_scales(bank, *, mx):
    """Only call at a quiescent phase boundary after all consumers have drained."""
    mx.synchronize()
    before = int(mx.get_active_memory())
    released = 0
    for projection in PROJECTIONS:
        name = projection + '.scales'
        raw = bank._views.pop(name)
        released += raw.nbytes
        raw.release()
        del raw
        del bank.arrays[name]
        del bank._segment_bytes[name]
    bank.record_bytes = WEIGHT_BYTES
    mx.synchronize()
    mx.clear_cache()
    after = int(mx.get_active_memory())
    if after > before - released + 1024**2:
        raise RuntimeError('an old scale backing still has a live owner')
    return released


def bind_weight_reader(reader):
    """Constructor-selected three-plane sidecar reader; source geometry proved by installer."""
    read_range = reader._readv_range_into
    submit = reader._fanout_executor.submit
    metrics = reader.metrics
    offsets = (0, 6266880, 12533760)
    names = tuple(p + '.weight' for p in PROJECTIONS)

    def run(items, cancel_event, deadline_ns, pipeline_phase):
        jobs = []
        futures = []
        error = None
        try:
            for record, destination in items:
                for name, offset in zip(names, offsets):
                    view = destination.component_view(name)
                    jobs.append((record.sidecar_offset + offset, view))
            metrics.update(record_requests=len(items), records_read=len(items), sidecar_record_requests=len(items))

            def read(job):
                offset, view = job
                read_range('experts.bin', offset, (view,), cancel_event=cancel_event,
                           deadline_ns=deadline_ns, pipeline_phase=pipeline_phase)

            try:
                for job in jobs[1:]:
                    futures.append(submit(read, job))
                read(jobs[0])
            except BaseException as exc:
                error = exc
            # A submit, read or cancellation failure cannot release a writable
            # slot view until every successfully submitted writer is terminal.
            for future in futures:
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
            for _, view in jobs:
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


def make_dispatch(scales, *, mx):
    """Bind layer scale owners and exact kernels once, retaining normal slot pins."""
    from paired_kernels import make_projection
    from mtplx.models.expert_mlx import _clamped_swiglu
    gate_kernel = make_projection(2304, 5120)
    down_kernel = make_projection(5120, 2304)
    gate_scales, up_scales, down_scales = (scales[p] for p in PROJECTIONS)

    def dispatch(selected, bindings, *, dense_prefill=False):
        bank = bindings[0].buffer.bank
        pairs = mx.array([(b.buffer.bank_index, b.expert) for b in bindings], dtype=mx.int32)
        rows = len(bindings)
        x = selected.reshape(rows, 1, 1, 5120)
        gate = gate_kernel(inputs=[x, pairs, bank.arrays['gate_proj.weight'], *gate_scales],
            template=[('T', mx.bfloat16)], grid=(32, 576, rows), threadgroup=(32, 2, 1),
            output_shapes=[(rows, 1, 1, 2304)], output_dtypes=[mx.bfloat16])[0]
        up = gate_kernel(inputs=[x, pairs, bank.arrays['up_proj.weight'], *up_scales],
            template=[('T', mx.bfloat16)], grid=(32, 576, rows), threadgroup=(32, 2, 1),
            output_shapes=[(rows, 1, 1, 2304)], output_dtypes=[mx.bfloat16])[0]
        hidden = _clamped_swiglu(gate, up, 10.0)
        return down_kernel(inputs=[hidden, pairs, bank.arrays['down_proj.weight'], *down_scales],
            template=[('T', mx.bfloat16)], grid=(32, 1280, rows), threadgroup=(32, 2, 1),
            output_shapes=[(rows, 1, 1, 5120)], output_dtypes=[mx.bfloat16])[0].reshape(rows, 5120)

    return dispatch
