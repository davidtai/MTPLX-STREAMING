"""CPU tests for the F2b lane-private host ring + reader interception (no GPU/Metal).

Real objects wherever possible: the real HostRing / SpeculativePool, the real derived
reader (built from the retained ``plane_lane.bind_reader`` source), and the real
GatePredictor + offline-scorer parity. The retained reader is exercised against a fake
reader whose ``_readv_range_into`` serves synthetic bytes by offset (so the hard-coded
real offsets 0 / 6,266,880 / 12,533,760 need no giant file), with tiny plane lengths.
MLX is pinned to CPU before import. Run under nice -n 19.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "_NoMLX"]
import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts" / "deepseek_v41"
_ARCH_PACKED = _REPO / "docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed"
for p in (str(_SCRIPTS), str(_ARCH_PACKED)):
    if p not in sys.path:
        sys.path.insert(0, p)

from f2.host_ring import HostRing, F2bCounters  # noqa: E402
from f2 import reader_intercept as ri  # noqa: E402
from f2 import predictor as fp  # noqa: E402
from f2.speculative import SpeculativePool  # noqa: E402
import f2_predictor as F  # noqa: E402  (offline scorer core)

# real record plane offsets (bind_reader hard-codes these); tiny lengths for the test.
OFFS = (0, 6_266_880, 12_533_760)
LENS = (16, 16, 12)                 # gate/up equal, down smaller (mirrors the real shape)
NAMES = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")


@pytest.fixture(autouse=True)
def _cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(prev)


# ---------------------------------------------------------------------------
# A. HostRing state machine
# ---------------------------------------------------------------------------
def _ring(records=2):
    return HostRing(records=records, planes=3, plane_bytes=max(LENS), counters=F2bCounters())


def test_ring_ready_hit_copies_and_no_reread():
    r = _ring()
    off, ln = OFFS[0], LENS[0]
    assert r.enqueue(off, ln)
    buf = r.begin_read(off)
    np.frombuffer(buf, np.uint8)[:] = np.arange(ln, dtype=np.uint8)
    r.end_read(off, ok=True)
    assert r.state_of(off) == "ready"
    dest = bytearray(ln)
    assert r.try_serve(off, memoryview(dest)) is True
    assert bytes(dest) == bytes(range(ln))
    assert r.counters.planes_hits == 1 and r.counters.planes_completed == 1


def test_ring_queued_cancel_then_worker_skips():
    r = _ring()
    off = OFFS[0]
    r.enqueue(off, LENS[0])
    assert r.try_serve(off, memoryview(bytearray(LENS[0]))) is False   # QUEUED -> cancel
    assert r.counters.planes_cancelled == 1
    assert r.begin_read(off) is None                                    # worker skips a cancelled entry


def test_ring_reading_wait_path():
    r = _ring()
    off, ln = OFFS[1], LENS[1]
    r.enqueue(off, ln)
    r.begin_read(off)                     # state READING, event not set
    served = {}

    def demand():
        served["ok"] = r.try_serve(off, memoryview(bytearray(ln)), wait_timeout=5.0)

    t = threading.Thread(target=demand)
    t.start()
    # complete the read shortly after the demand thread starts waiting.
    threading.Timer(0.1, lambda: r.end_read(off, ok=True)).start()
    t.join(timeout=5.0)
    assert served.get("ok") is True
    assert r.counters.planes_waits == 1


def test_ring_failure_evicts_and_demand_falls_back():
    r = _ring()
    off = OFFS[0]
    r.enqueue(off, LENS[0])
    r.begin_read(off)
    r.end_read(off, ok=False)             # speculative read failed -> evict
    assert r.has(off) is False
    assert r.try_serve(off, memoryview(bytearray(LENS[0]))) is False   # absent -> pread


def test_ring_recycle_skips_referenced_and_reading():
    # capacity 3 buffers (records=1). Fill 3 READY, hold one referenced, one READING;
    # a 4th enqueue must recycle only the free-able (READY, unreferenced) one.
    r = HostRing(records=1, planes=3, plane_bytes=max(LENS), counters=F2bCounters())
    offs = [100, 200, 300]
    for o in offs:
        r.enqueue(o, LENS[0]); r.begin_read(o); r.end_read(o, ok=True)   # all READY
    # make offs[0] READING (not recyclable), offs[1] referenced via an in-progress copy.
    with r._lock:
        r._entries[offs[0]].state = "reading"
        r._entries[offs[1]].refcount = 1
    assert r.enqueue(400, LENS[0]) is True          # recycles offs[2] (the only free-able)
    assert r.has(offs[2]) is False and r.has(400) is True
    assert r.has(offs[0]) and r.has(offs[1])         # protected ones survive
    assert r.counters.planes_wasted >= 1             # a READY entry recycled unconsumed


# ---------------------------------------------------------------------------
# B. Derived reader (from the retained bind_reader source)
# ---------------------------------------------------------------------------
class _Metrics:
    def __init__(self):
        self.calls = []

    def update(self, **kw):
        self.calls.append(tuple(sorted(kw.items())))


class _Fut:
    def __init__(self, fn, args):
        self._e = None
        try:
            self._r = fn(*args)
        except BaseException as e:  # noqa: BLE001
            self._r, self._e = None, e

    def result(self):
        if self._e:
            raise self._e
        return self._r


class _Exec:
    def submit(self, fn, *a):
        return _Fut(fn, a)


class _Reader:
    """Fake reader: _readv_range_into fills each view with a deterministic pattern keyed
    by offset, and records every offset it preads."""

    def __init__(self):
        self.metrics = _Metrics()
        self._fanout_executor = _Exec()
        self.preads = []

    def _readv_range_into(self, name, offset, views, *, cancel_event=None,
                          deadline_ns=None, pipeline_phase=None):
        self.preads.append(int(offset))
        for v in views:
            arr = np.frombuffer(v, np.uint8)
            if not arr.flags.writeable:
                arr = arr.view(); arr.flags.writeable = True
            arr[:] = (np.arange(arr.shape[0], dtype=np.uint8) + np.uint8(offset & 0x3F))


class _Dest:
    def __init__(self):
        self.buffers = {name: bytearray(L) for name, L in zip(NAMES, LENS)}

    def component_view(self, name):
        return memoryview(self.buffers[name])


class _Witness:
    read_priority = 0

    def __init__(self):
        self.published = []

    def publish_read_components(self, items):
        self.published.append([rec.sidecar_offset for rec, _ in items])


class _Rec:
    def __init__(self, sidecar_offset):
        self.sidecar_offset = int(sidecar_offset)


def _run_batch(bind_fn, serve, sidecar_offset=0):
    """Bind a reader with ``bind_fn`` (retained plane_lane.bind_reader OR the derived one)
    and run one component-record read; return (reader, dest, witness)."""
    import plane_lane

    reader = _Reader()
    local = threading.local()
    local.part = _Witness()
    if serve is None:
        bind_fn(reader, local)                       # retained signature (reader, local)
    else:
        fn, _b, _d = ri.build_bind_reader(serve)
        fn(reader, local)
    dest = _Dest()
    reader.read_component_records_into(None, ((_Rec(sidecar_offset), dest),))
    return reader, dest, local.part


def test_derived_reader_equals_retained_when_ring_empty():
    import plane_lane

    empty = HostRing(records=2, planes=3, plane_bytes=max(LENS), counters=F2bCounters())
    r_ret, d_ret, w_ret = _run_batch(plane_lane.bind_reader, serve=None)
    r_der, d_der, w_der = _run_batch(None, serve=empty.try_serve)
    # byte-identical destinations, identical metrics, identical early-witness publication.
    assert d_ret.buffers == d_der.buffers
    assert r_ret.metrics.calls == r_der.metrics.calls
    assert w_ret.published == w_der.published
    # ring empty -> the derived reader still preads every plane (no serve).
    assert sorted(r_der.preads) == sorted(r_ret.preads) == sorted(OFFS)
    assert empty.counters.planes_hits == 0


def test_derived_reader_ready_plane_serves_from_ring_no_pread():
    ring = HostRing(records=2, planes=3, plane_bytes=max(LENS), counters=F2bCounters())
    # make the GATE plane (offset 0) READY with a known pattern; up/down absent.
    ring.enqueue(OFFS[0], LENS[0])
    buf = ring.begin_read(OFFS[0])
    np.frombuffer(buf, np.uint8)[:] = np.full(LENS[0], 0xAB, np.uint8)
    ring.end_read(OFFS[0], ok=True)
    reader, dest, _w = _run_batch(None, serve=ring.try_serve)
    # gate served from RAM (no pread at offset 0); up/down preaded.
    assert OFFS[0] not in reader.preads
    assert OFFS[1] in reader.preads and OFFS[2] in reader.preads
    assert bytes(dest.buffers["gate_proj.weight"]) == bytes([0xAB]) * LENS[0]
    assert ring.counters.planes_hits == 1


def test_derived_reader_queued_plane_cancels_and_preads():
    ring = HostRing(records=2, planes=3, plane_bytes=max(LENS), counters=F2bCounters())
    ring.enqueue(OFFS[0], LENS[0])          # QUEUED (never read)
    reader, dest, _w = _run_batch(None, serve=ring.try_serve)
    assert OFFS[0] in reader.preads          # demand cancelled the queued entry and preaded
    assert ring.counters.planes_cancelled == 1


def test_derivation_roundtrips_and_pins_retained_lane():
    import inspect
    import textwrap
    import plane_lane

    src = textwrap.dedent(inspect.getsource(plane_lane.bind_reader))
    derived = ri.derive_bind_reader_source(src)
    assert "F2b ring intercept" in derived
    assert [l for l in derived.splitlines() if l.strip() != ri._INSERT] == src.splitlines()
    import hashlib
    assert hashlib.sha256(inspect.getsource(plane_lane).encode()).hexdigest() == ri.RETAINED_PLANE_LANE_SHA256


# ---------------------------------------------------------------------------
# C. Speculative pool (real threads + fake reader) + window-stop
# ---------------------------------------------------------------------------
def test_pool_fills_ring_and_window_stop_drops_unstarted():
    ring = HostRing(records=4, planes=3, plane_bytes=max(LENS), counters=F2bCounters())
    reader = _Reader()
    pool = SpeculativePool(reader, ring, workers=2)
    try:
        specs = tuple(zip(OFFS, LENS))
        pool.enqueue_record(5, 0, specs)          # record at sidecar 0 -> planes 0/6.27M/12.53M
        pool.drain()
        assert ring.state_of(OFFS[0]) == "ready"
        assert ring.counters.planes_completed == 3
        assert ring.counters.records_full == 1
        # window-stop for target 6: a record enqueued after the flag is set never starts.
        pool.note_demand_imminent(6)
        pool.enqueue_record(6, 20_000_000, specs)
        pool.drain()
        assert ring.state_of(20_000_000 + OFFS[0]) == "queued"   # never read (dropped)
    finally:
        pool.shutdown()


# ---------------------------------------------------------------------------
# D. Predictor ranking == offline scorer
# ---------------------------------------------------------------------------
def test_rank_targets_matches_offline_scorer():
    rng = np.random.default_rng(7)
    rows, n = 6, 24
    row_scores = rng.standard_normal((rows, n)).astype(np.float64)
    merged = row_scores.max(axis=0)                       # device merge == max over rows
    resident = {3, 11}
    ready_mask = np.zeros(n, dtype=bool)
    for e in resident:
        ready_mask[e] = True
    ref = F.merge_rank_exclude(row_scores, ready_mask, 3)
    got = fp.rank_targets(merged, skip=lambda e: e in resident, k=3)
    assert got == ref


def test_select_prefetch_sources_geometry():
    assert fp.select_prefetch_sources(range(40), first_target=4) == list(range(3, 39))
    assert 39 not in fp.select_prefetch_sources(range(40), first_target=4)


def test_gate_predictor_merged_is_max_over_rows_biased():
    from mtplx.models.deepseek_v41_moe import Gate, _gate_prefix_impl
    from types import SimpleNamespace
    args = SimpleNamespace(hidden_size=64, num_experts_per_tok=2, scoring_func="sqrtsoftplus",
                           gate_temp=1.0, norm_topk_prob=True, routed_scaling_factor=1.0,
                           n_routed_experts=16)
    mx.random.seed(3)
    gate = Gate(5, args)
    gate.weight = (0.4 * mx.random.normal((16, 64))).astype(mx.bfloat16)
    gate.e_score_correction_bias = (0.1 * mx.random.normal((16,))).astype(mx.float32)
    mx.eval(gate.weight, gate.e_score_correction_bias)
    tokens = (0.3 * mx.random.normal((6, 64))).astype(mx.bfloat16)
    mx.eval(tokens)
    _s, biased = _gate_prefix_impl(tokens, gate.weight, gate.e_score_correction_bias, 1.0, "sqrtsoftplus")
    ref = np.asarray(mx.max(biased, axis=0).tolist(), dtype=np.float64)
    got = np.asarray(fp.GatePredictor(gate).merged(tokens).tolist(), dtype=np.float64)
    assert np.array_equal(got, ref)


# ---------------------------------------------------------------------------
# E. Barrier parity: predict rides ONE mx.eval(indices, merged)
# ---------------------------------------------------------------------------
def test_wrapper_barrier_parity_one_eval():
    from mtplx.models.deepseek_v41_moe import Gate
    from types import SimpleNamespace
    args = SimpleNamespace(hidden_size=64, num_experts_per_tok=2, scoring_func="sqrtsoftplus",
                           gate_temp=1.0, norm_topk_prob=True, routed_scaling_factor=1.0,
                           n_routed_experts=16)
    gate = Gate(5, args)
    mx.eval(gate.weight, gate.e_score_correction_bias)
    pred = fp.GatePredictor(gate)
    tokens = mx.zeros((6, 64), dtype=mx.bfloat16)
    indices = mx.zeros((6, 2), dtype=mx.int32)
    main = threading.main_thread()
    n = {"eval": 0}
    real = mx.eval

    def counting(*a, **k):
        if threading.current_thread() is main:
            n["eval"] += 1
        return real(*a, **k)

    mx.eval = counting
    try:
        merged = pred.merged(tokens)
        mx.eval(indices, merged)          # the wrapper's single routing barrier
    finally:
        mx.eval = real
    assert n["eval"] == 1


# ---------------------------------------------------------------------------
# F. End-to-end reader: lane on (ring populated) vs off (ring empty) -> identical dest
# ---------------------------------------------------------------------------
def test_end_to_end_lane_on_vs_off_identical_output():
    import plane_lane

    # OFF: retained reader preads all planes.
    _r_off, d_off, _w = _run_batch(plane_lane.bind_reader, serve=None)
    # ON: pre-fill the ring for this record's planes with the SAME bytes the reader would
    # pread (deterministic pattern), so the demand read serves from RAM -> identical dest.
    ring = HostRing(records=2, planes=3, plane_bytes=max(LENS), counters=F2bCounters())
    for off, ln in zip(OFFS, LENS):
        ring.enqueue(off, ln)
        buf = ring.begin_read(off)
        np.frombuffer(buf, np.uint8)[:] = (np.arange(ln, dtype=np.uint8) + np.uint8(off & 0x3F))
        ring.end_read(off, ok=True)
    r_on, d_on, _w2 = _run_batch(None, serve=ring.try_serve)
    assert d_on.buffers == d_off.buffers          # identical routed bytes
    assert r_on.preads == []                       # every plane served from RAM
    assert ring.counters.planes_hits == 3
