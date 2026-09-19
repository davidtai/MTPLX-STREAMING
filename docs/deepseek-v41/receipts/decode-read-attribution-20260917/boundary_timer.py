"""Explicit main-thread wall-time scopes for one diagnostic decode run."""
import functools
import threading
import time


class BoundaryTimer:
    def __init__(self):
        self.owner = threading.get_ident()
        self.stack = []
        self.stats = {}
        self.edges = {}

    def call(self, name, fn, *args, **kwargs):
        if threading.get_ident() != self.owner:
            return fn(*args, **kwargs)
        parent = self.stack[-1][0] if self.stack else None
        frame = [name, time.perf_counter_ns(), 0]
        self.stack.append(frame)
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter_ns() - frame[1]
            self.stack.pop()
            exclusive = elapsed - frame[2]
            if self.stack:
                self.stack[-1][2] += elapsed
            for table, key in ((self.stats, name), (self.edges, (parent, name))):
                row = table.setdefault(key, [0, 0, 0])
                row[0] += 1
                row[1] += elapsed
                row[2] += exclusive

    def wrap(self, name, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            return self.call(name, fn, *args, **kwargs)
        return wrapped

    def wrap_iterator(self, name, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            iterator = iter(fn(*args, **kwargs))
            try:
                while True:
                    try:
                        item = self.call(name, next, iterator)
                    except StopIteration:
                        return
                    yield item
            finally:
                iterator.close()
        return wrapped

    def snapshot(self):
        if self.stack:
            raise RuntimeError('snapshot requested inside an unfinished scope')
        if any(row[2] < 0 or row[1] < row[2] for row in self.stats.values()):
            raise RuntimeError('invalid inclusive/exclusive timing')
        def row(name, values):
            return dict(name=name, observations=values[0],
                        inclusive_s=values[1]/1e9, exclusive_s=values[2]/1e9)
        root_ns = sum(values[1] for (parent, name), values in self.edges.items() if parent is None)
        exclusive_ns = sum(values[2] for values in self.stats.values())
        if root_ns != exclusive_ns:
            raise RuntimeError('scopes do not partition root wall time')
        return dict(root_wall_ns=root_ns, exclusive_sum_ns=exclusive_ns,
                    owner_thread_id=self.owner,
                    rows=sorted((row(name, values) for name, values in self.stats.items()),
                                key=lambda r:r['exclusive_s'], reverse=True),
                    edges=[dict(parent=parent, **row(name, values))
                           for (parent, name), values in self.edges.items()])
