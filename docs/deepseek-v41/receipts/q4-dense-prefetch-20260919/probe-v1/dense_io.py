"""Bounded diagnostic readers for native Q4 model query projections.

Only construction touches MLX storage. Background workers fill stable writable
buffers with positional reads; the caller owns the GPU retirement boundary.
"""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os


class QueryStore:
    def __init__(self, root, tensors, *, mx, resident):
        self.mx = mx
        self.resident = resident
        self.fds = {}
        self.buffers = []
        self.views = []
        self.jobs = []
        self.future = None
        self.pool = None
        try:
            grouped = {}
            headers = {}
            for t in tensors:
                name = t['shard']
                if name not in self.fds:
                    fd = os.open(root / name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
                    self.fds[name] = fd
                    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                    n = int.from_bytes(os.pread(fd, 8, 0), 'little')
                    if not 0 < n <= 1024**2:
                        raise RuntimeError('unexpected safetensors header size')
                    headers[name] = (8+n, json.loads(os.pread(fd, n, 8)))
                base, header = headers[name]
                h = header[t['tensor']]
                lo, hi = h['data_offsets']
                if (h['dtype'], h['shape'], base+lo, hi-lo) != (t['dtype'], t['shape'], t['offset'], t['length']):
                    raise RuntimeError('query source metadata changed')
                if base+hi > os.fstat(self.fds[name]).st_size:
                    raise RuntimeError('query tensor exceeds source file')
                layer = int(t['tensor'].split('.')[1])
                grouped.setdefault(layer, {})[t['tensor'].split('.')[-1]] = t
            signature = lambda row: sorted((k, t['dtype'], t['shape'], t['length']) for k,t in row.items())
            if set(grouped) != set(range(40)) or set(grouped[0]) != {'weight', 'scales'}:
                raise RuntimeError('query family does not cover the exact 40 layers')
            if any(signature(row) != signature(grouped[0]) for row in grouped.values()):
                raise RuntimeError('query family is not uniform')
            if (grouped[0]['weight']['shape'], grouped[0]['scales']['shape']) != ([32768,320], [32768,40]):
                raise RuntimeError('query native MXFP8 geometry changed')
            count = 40 if resident else 2
            for _ in range(count):
                arrays, views = {}, {}
                for name in ('weight', 'scales'):
                    t = grouped[0][name]
                    a = mx.zeros(t['shape'], dtype=mx.uint32 if name == 'weight' else mx.uint8)
                    mx.eval(a)
                    view = memoryview(a).cast('B')
                    if view.readonly or not view.c_contiguous or view.nbytes != t['length']:
                        raise RuntimeError('query destination is not stable writable storage')
                    arrays[name], views[name] = a, view
                self.buffers.append(arrays)
                self.views.append(views)
            for layer in range(40):
                row = grouped[layer]
                self.jobs.append(tuple((self.fds[row[n]['shard']], row[n]['offset'], n)
                                       for n in ('weight', 'scales')))
            if resident:
                for layer in range(40):
                    self.fill(layer, layer)
            else:
                self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='q4-dense-lookahead')
        except BaseException:
            self.close()
            raise

    def fill(self, layer, slot):
        for fd, offset, name in self.jobs[layer]:
            view = self.views[slot][name]
            cursor = 0
            while cursor < len(view):
                end = min(cursor + 8*1024**2, len(view))
                n = os.preadv(fd, [view[cursor:end]], offset+cursor)
                if n <= 0:
                    raise RuntimeError('short query-weight read')
                cursor += n

    def start(self):
        # Include the initial fill in candidate timing; subsequent reads overlap.
        self.fill(0, 0)

    def acquire(self, step):
        if self.future is not None:
            self.future.result()
            self.future = None
        return self.buffers[step % 2]

    def issue(self, step):
        # Caller has crossed the router dependency barrier and submitted demand
        # expert reads. The other slot's last query consumer is already complete.
        self.future = self.pool.submit(self.fill, step % 40, step % 2)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None
        if self.future is not None:
            self.future.result()
            self.future = None
        for views in self.views:
            for view in views.values():
                view.release()
        self.views.clear()
        self.buffers.clear()
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()
