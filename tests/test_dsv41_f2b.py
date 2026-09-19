"""CPU tests for the F2b lane-private host ring + reader interception (no GPU/Metal).

Real objects: the real HostRing / SpeculativePool (coordinator + worker threads), the real
derived reader (built from the retained ``plane_lane.bind_reader`` source), and the real
GatePredictor + offline-scorer parity. The retained reader runs against a fake reader whose
``_readv_range_into`` serves synthetic bytes by offset (the hard-coded real offsets need no
giant file), with tiny EQUAL plane lengths (the three decode weight planes are equal-length).
MLX pinned to CPU. Run under nice -n 19.
"""
from __future__ import annotations

import sys
import threading
import time
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
import f2_predictor as F  # noqa: E402

OFFS = (0, 6_266_880, 12_533_760)
LEN = 16                                  # equal weight-plane length (real: 5,898,240)
LENS = (LEN, LEN, LEN)
SPECS = tuple((off, LEN) for off in OFFS)
NAMES = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")


@pytest.fixture(autouse=True)
def _cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(prev)


def _ring(records=2):
    return HostRing(records=records, planes=3, plane_bytes=LEN, counters=F2bCounters())


# ---------------------------------------------------------------------------
# A. HostRing state machine
# ---------------------------------------------------------------------------
def test_ring_ready_hit_copies_and_no_reread():
    r = _ring()
    off = OFFS[0]
    assert r.enqueue(off, LEN)
    np.frombuffer(r.begin_read(off), np.uint8)[:] = np.arange(LEN, dtype=np.uint8)
    r.end_read(off, ok=True)
    dest = bytearray(LEN)
    assert r.try_serve(off, memoryview(dest)) is True
    assert bytes(dest) == bytes(range(LEN))
    assert r.counters.planes_hits == 1 and r.counters.planes_completed == 1
    assert r.counters.ready_reading_hwm >= 1


def test_ring_length_mismatch_refuses_and_counts():
    r = _ring()
    off = OFFS[0]
    r.enqueue(off, LEN)
    np.frombuffer(r.begin_read(off), np.uint8)[:] = np.full(LEN, 9, np.uint8)
    r.end_read(off, ok=True)
    short = bytearray(LEN - 4)                          # demand view disagrees in length
    assert r.try_serve(off, memoryview(short)) is False  # refuse -> pread
    assert r.counters.planes_length_mismatch == 1
    assert bytes(short) == bytes(LEN - 4)               # untouched (no stale partial copy)
    assert r.counters.planes_hits == 0


def test_ring_queued_cancel_then_worker_skips():
    r = _ring()
    off = OFFS[0]
    r.enqueue(off, LEN)
    assert r.try_serve(off, memoryview(bytearray(LEN))) is False
    assert r.counters.planes_cancelled == 1
    assert r.begin_read(off) is None


def test_ring_discard_frees_queued_entry():
    r = _ring()
    off = OFFS[0]
    r.enqueue(off, LEN)
    assert r.has(off) is True
    assert r.discard(off) is True
    assert r.has(off) is False
    # a READY/READING/referenced entry is NOT discardable.
    r.enqueue(off, LEN); r.begin_read(off); r.end_read(off, ok=True)
    assert r.discard(off) is False and r.has(off) is True


def test_ring_reading_wait_path():
    r = _ring()
    off = OFFS[1]
    r.enqueue(off, LEN)
    r.begin_read(off)
    served = {}
    t = threading.Thread(target=lambda: served.__setitem__("ok", r.try_serve(off, memoryview(bytearray(LEN)))))
    t.start()
    threading.Timer(0.1, lambda: r.end_read(off, ok=True)).start()
    t.join(timeout=5.0)
    assert served.get("ok") is True and r.counters.planes_waits == 1


def test_ring_failure_evicts_and_demand_falls_back():
    r = _ring()
    off = OFFS[0]
    r.enqueue(off, LEN); r.begin_read(off); r.end_read(off, ok=False)
    assert r.has(off) is False
    assert r.try_serve(off, memoryview(bytearray(LEN))) is False


def test_ring_recycle_skips_referenced_and_reading():
    r = HostRing(records=1, planes=3, plane_bytes=LEN, counters=F2bCounters())
    offs = [100, 200, 300]
    for o in offs:
        r.enqueue(o, LEN); r.begin_read(o); r.end_read(o, ok=True)
    with r._lock:
        r._entries[offs[0]].state = "reading"
        r._entries[offs[1]].refcount = 1
    assert r.enqueue(400, LEN) is True
    assert r.has(offs[2]) is False and r.has(400) is True
    assert r.has(offs[0]) and r.has(offs[1])
    assert r.counters.planes_wasted >= 1


def test_ring_consumed_ready_not_wasted_on_recycle():
    r = HostRing(records=1, planes=3, plane_bytes=LEN, counters=F2bCounters())
    offs = [10, 20, 30]
    for o in offs:
        r.enqueue(o, LEN); r.begin_read(o); r.end_read(o, ok=True)
    r.try_serve(offs[0], memoryview(bytearray(LEN)))    # consume offs[0]
    # recycle: offs[0] is consumed -> not wasted; offs[1] unconsumed -> wasted.
    r.enqueue(40, LEN); r.enqueue(50, LEN)
    assert r.counters.planes_wasted >= 1
    # the consumed one, when recycled, must not add to wasted beyond the unconsumed ones.


# ---------------------------------------------------------------------------
# B. Derived reader (from the retained bind_reader source)
# ---------------------------------------------------------------------------
class _Metrics:
    def __init__(self): self.calls = []
    def update(self, **kw): self.calls.append(tuple(sorted(kw.items())))


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
    def submit(self, fn, *a): return _Fut(fn, a)


class _Reader:
    def __init__(self):
        self.metrics = _Metrics()
        self._fanout_executor = _Exec()
        self.preads = []
    def _readv_range_into(self, name, offset, views, *, cancel_event=None, deadline_ns=None, pipeline_phase=None):
        self.preads.append(int(offset))
        for v in views:
            arr = np.frombuffer(v, np.uint8)
            if not arr.flags.writeable:
                arr = arr.view(); arr.flags.writeable = True
            arr[:] = (np.arange(arr.shape[0], dtype=np.uint8) + np.uint8(offset & 0x3F))


class _Dest:
    def __init__(self): self.buffers = {n: bytearray(LEN) for n in NAMES}
    def component_view(self, name): return memoryview(self.buffers[name])


class _Witness:
    def __init__(self): self.published = []
    def publish_read_components(self, items): self.published.append([r.sidecar_offset for r, _ in items])


class _Rec:
    def __init__(self, sidecar_offset): self.sidecar_offset = int(sidecar_offset)


def _run_batch(bind_fn, serve):
    reader = _Reader()
    local = threading.local()
    local.part = _Witness()
    if serve is None:
        bind_fn(reader, local)
    else:
        fn, _b, _d = ri.build_bind_reader(serve)
        fn(reader, local)
    dest = _Dest()
    reader.read_component_records_into(None, ((_Rec(0), dest),))
    return reader, dest, local.part


def test_derived_reader_equals_retained_when_ring_empty():
    import plane_lane
    empty = _ring()
    r_ret, d_ret, w_ret = _run_batch(plane_lane.bind_reader, serve=None)
    r_der, d_der, w_der = _run_batch(None, serve=empty.try_serve)
    assert d_ret.buffers == d_der.buffers
    assert r_ret.metrics.calls == r_der.metrics.calls
    assert w_ret.published == w_der.published
    assert sorted(r_der.preads) == sorted(r_ret.preads) == sorted(OFFS)
    assert empty.counters.planes_hits == 0


def test_derived_reader_ready_plane_serves_from_ring_no_pread():
    ring = _ring()
    ring.enqueue(OFFS[0], LEN)
    np.frombuffer(ring.begin_read(OFFS[0]), np.uint8)[:] = np.full(LEN, 0xAB, np.uint8)
    ring.end_read(OFFS[0], ok=True)
    reader, dest, _w = _run_batch(None, serve=ring.try_serve)
    assert OFFS[0] not in reader.preads
    assert OFFS[1] in reader.preads and OFFS[2] in reader.preads
    assert bytes(dest.buffers["gate_proj.weight"]) == bytes([0xAB]) * LEN
    assert ring.counters.planes_hits == 1


def test_derived_reader_queued_plane_cancels_and_preads():
    ring = _ring()
    ring.enqueue(OFFS[0], LEN)
    reader, _d, _w = _run_batch(None, serve=ring.try_serve)
    assert OFFS[0] in reader.preads and ring.counters.planes_cancelled == 1


def test_derivation_roundtrips_pins_lane_and_has_no_eligibility_check():
    import inspect, textwrap, hashlib
    import plane_lane
    src = textwrap.dedent(inspect.getsource(plane_lane.bind_reader))
    derived = ri.derive_bind_reader_source(src)
    assert "F2b ring intercept" in derived and "is not None" not in ri._INSERT
    assert [l for l in derived.splitlines() if l.strip() != ri._INSERT] == src.splitlines()
    assert hashlib.sha256(inspect.getsource(plane_lane).encode()).hexdigest() == ri.RETAINED_PLANE_LANE_SHA256


# ---------------------------------------------------------------------------
# C. Speculative pool: coordinator + workers, multi-cycle epoch (fix 1), window-stop
# ---------------------------------------------------------------------------
def test_pool_prefetch_alive_across_cycles():
    """>= 3 cycles: each cycle predicts a fresh record for T and its planes ARE read
    (the old set-based window-stop killed prefetch from cycle 2)."""
    ring = HostRing(records=8, planes=3, plane_bytes=LEN, counters=F2bCounters())
    reader = _Reader()
    rec = {"sidecar": 0}
    pool = SpeculativePool(reader, ring, plane_specs=SPECS,
                           plan_fn=lambda t, s: [rec["sidecar"]], workers=2)
    T = 5
    try:
        for cycle in range(3):
            rec["sidecar"] = 1_000_000 * (cycle + 1)     # a fresh record each cycle
            ep = pool.current_epoch(T)
            pool.submit_prediction(T, ep, np.zeros(24, np.float32))
            pool.drain()
            assert ring.state_of(rec["sidecar"] + OFFS[0]) == "ready", f"cycle {cycle} not read"
            pool.note_demand_imminent(T)                 # T runs -> epoch++
        assert ring.counters.planes_completed >= 9       # 3 cycles x 3 planes
    finally:
        pool.shutdown()


def test_pool_window_stop_skips_stale_epoch_and_discards():
    """A job left queued from cycle N is skipped in cycle N+1 (epoch mismatch) and the
    ring entry is discarded, not left as a phantom hit (fix 3)."""
    ring = HostRing(records=8, planes=3, plane_bytes=LEN, counters=F2bCounters())
    gate = threading.Event()
    entered = threading.Event()
    reader = _Reader()
    real = reader._readv_range_into

    def gated(name, offset, views, **kw):
        entered.set()
        gate.wait(timeout=5.0)
        return real(name, offset, views, **kw)

    reader._readv_range_into = gated
    pool = SpeculativePool(reader, ring, plane_specs=SPECS,
                           plan_fn=lambda t, s: [0], workers=1)   # one worker: one plane in flight
    T = 5
    try:
        pool.submit_prediction(T, pool.current_epoch(T), np.zeros(24, np.float32))
        assert entered.wait(timeout=5.0)                 # worker started plane 1 (in flight)
        pool.note_demand_imminent(T)                     # epoch++ while planes 2,3 wait in the queue
        gate.set()
        pool.drain()
        assert ring.state_of(OFFS[0]) == "ready"         # in-flight plane 1 finished
        # planes 2,3 (stale epoch) were skipped AND discarded (no phantom QUEUED entries).
        assert ring.counters.planes_epoch_skipped >= 2
        assert ring.has(OFFS[1]) is False and ring.has(OFFS[2]) is False
    finally:
        gate.set()
        pool.shutdown()


# ---------------------------------------------------------------------------
# D. Predictor ranking == offline scorer
# ---------------------------------------------------------------------------
def test_rank_targets_matches_offline_scorer():
    rng = np.random.default_rng(7)
    rows, n = 6, 24
    row_scores = rng.standard_normal((rows, n)).astype(np.float64)
    merged = row_scores.max(axis=0)
    resident = {3, 11}
    ready = np.zeros(n, dtype=bool)
    for e in resident:
        ready[e] = True
    assert fp.rank_targets(merged, skip=lambda e: e in resident, k=3) == F.merge_rank_exclude(row_scores, ready, 3)


def test_select_prefetch_sources_geometry():
    assert fp.select_prefetch_sources(range(40), first_target=4) == list(range(3, 39))


def test_gate_predictor_merged_is_max_over_rows_biased():
    from mtplx.models.deepseek_v41_moe import Gate, _gate_prefix_impl
    from types import SimpleNamespace
    args = SimpleNamespace(hidden_size=64, num_experts_per_tok=2, scoring_func="sqrtsoftplus",
                           gate_temp=1.0, norm_topk_prob=True, routed_scaling_factor=1.0, n_routed_experts=16)
    mx.random.seed(3)
    gate = Gate(5, args)
    gate.weight = (0.4 * mx.random.normal((16, 64))).astype(mx.bfloat16)
    gate.e_score_correction_bias = (0.1 * mx.random.normal((16,))).astype(mx.float32)
    tokens = (0.3 * mx.random.normal((6, 64))).astype(mx.bfloat16)
    mx.eval(gate.weight, gate.e_score_correction_bias, tokens)
    _s, biased = _gate_prefix_impl(tokens, gate.weight, gate.e_score_correction_bias, 1.0, "sqrtsoftplus")
    ref = np.asarray(mx.max(biased, axis=0).tolist(), dtype=np.float64)
    got = np.asarray(fp.GatePredictor(gate).merged(tokens).tolist(), dtype=np.float64)
    assert np.array_equal(got, ref)


# ---------------------------------------------------------------------------
# E. Barrier parity: source rides ONE mx.eval(indices, merged); target-only evals first
# ---------------------------------------------------------------------------
def _count_main_evals(fn):
    main = threading.main_thread()
    n = {"e": 0}
    real = mx.eval

    def counting(*a, **k):
        if threading.current_thread() is main:
            n["e"] += 1
        return real(*a, **k)

    mx.eval = counting
    try:
        fn()
    finally:
        mx.eval = real
    return n["e"]


def test_source_wrapper_one_barrier():
    from mtplx.models.deepseek_v41_moe import Gate
    from types import SimpleNamespace
    args = SimpleNamespace(hidden_size=64, num_experts_per_tok=2, scoring_func="sqrtsoftplus",
                           gate_temp=1.0, norm_topk_prob=True, routed_scaling_factor=1.0, n_routed_experts=16)
    gate = Gate(5, args)
    mx.eval(gate.weight, gate.e_score_correction_bias)
    pred = fp.GatePredictor(gate)
    tokens = mx.zeros((6, 64), dtype=mx.bfloat16)
    indices = mx.zeros((6, 2), dtype=mx.int32)

    def source():
        merged = pred.merged(tokens)
        mx.eval(indices, merged)
    assert _count_main_evals(source) == 1


def test_target_only_wrapper_completes_barrier_before_note(fix5=True):
    # fix 5: a target-only layer must mx.eval(indices) in the wrapper (one barrier),
    # not leave it to the scheduled run after noting imminent.
    indices = mx.zeros((6, 2), dtype=mx.int32)
    order = []

    class _Pool:
        def note_demand_imminent(self, t): order.append("note")

    pool = _Pool()

    def target_only():
        mx.eval(indices)          # wrapper completes the routing barrier first
        order.append("eval")
        pool.note_demand_imminent(39)
    assert _count_main_evals(target_only) == 1
    assert order == ["eval", "note"]   # barrier BEFORE note imminent


# ---------------------------------------------------------------------------
# F. Wrapper main-thread cost microbench (fix 4): only np.asarray + one queue.put
# ---------------------------------------------------------------------------
def test_wrapper_main_thread_cost_microbench(capsys):
    import queue
    merged = mx.zeros((384,), dtype=mx.float32)
    mx.eval(merged)
    q = queue.Queue()

    class _P:
        def current_epoch(self, t): return 0
        def submit_prediction(self, t, e, s): q.put((t, e, s))

    pool = _P()
    N = 2000

    def hand_off():
        scores = np.asarray(merged, dtype=np.float32).copy()
        pool.submit_prediction(5, pool.current_epoch(5), scores)

    hand_off()                                   # warm
    t0 = time.perf_counter()
    for _ in range(N):
        hand_off()
    us = (time.perf_counter() - t0) / N * 1e6
    with capsys.disabled():
        print(f"\n[f2b microbench] wrapper main-thread hand-off (excl. mx.eval): {us:.2f} us/call "
              f"(before: resident-walk + rank + 3x manifest.record + 3x enqueue ~= 100-200 us/call)")
    assert us < 100.0                            # target < 25 us; 100 us guards against CI flakiness


# ---------------------------------------------------------------------------
# G. End-to-end reader: lane on (ring pre-filled) vs off (retained) -> identical bytes
# ---------------------------------------------------------------------------
def test_end_to_end_lane_on_vs_off_identical_output():
    import plane_lane
    _r_off, d_off, _w = _run_batch(plane_lane.bind_reader, serve=None)
    ring = _ring()
    for off in OFFS:
        ring.enqueue(off, LEN)
        np.frombuffer(ring.begin_read(off), np.uint8)[:] = (np.arange(LEN, dtype=np.uint8) + np.uint8(off & 0x3F))
        ring.end_read(off, ok=True)
    r_on, d_on, _w2 = _run_batch(None, serve=ring.try_serve)
    assert d_on.buffers == d_off.buffers and r_on.preads == []
    assert ring.counters.planes_hits == 3


def test_counters_schema_present():
    c = F2bCounters().as_dict()
    for k in ("planes_length_mismatch", "planes_epoch_skipped", "coordinator_batches",
              "ready_reading_hwm", "planes_wasted", "records_full", "records_partial"):
        assert k in c
