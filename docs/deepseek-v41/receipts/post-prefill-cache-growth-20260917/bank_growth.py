"""Experimental phase-boundary storage growth; never called by a decode layer."""

def grow_bank(bank, capacity, *, mx, release_slack_bytes=1024**2):
    old_capacity = bank.capacity
    if capacity <= old_capacity:
        raise ValueError('growth requires a larger capacity')
    initial_active = mx.get_active_memory()
    added = 0
    # Each replacement is evaluated before its old exported view is released.
    # The stable bank object and every old logical row index are preserved.
    for component in tuple(bank.arrays):
        old = bank.arrays[component]
        replacement = mx.concatenate([
            old, mx.zeros((capacity - old_capacity, *old.shape[1:]), dtype=old.dtype)
        ], axis=0)
        mx.eval(replacement)
        raw = memoryview(replacement).cast('B')
        expected = capacity * bank._segment_bytes[component]
        if raw.readonly or not raw.c_contiguous or raw.nbytes != expected:
            raw.release()
            raise RuntimeError('new component storage is not an owned writable bank')
        bank._views[component].release()
        bank.arrays[component] = replacement
        bank._views[component] = raw
        added += (capacity - old_capacity) * bank._segment_bytes[component]
        del old, replacement, raw
        # Installation-boundary ownership check: abort after one bounded copy
        # if a stale exported view/tape still owns its old large allocation.
        if mx.get_active_memory() > initial_active + added + release_slack_bytes:
            raise RuntimeError('old component backing remains live after replacement')
    bank.capacity = capacity
    return added
