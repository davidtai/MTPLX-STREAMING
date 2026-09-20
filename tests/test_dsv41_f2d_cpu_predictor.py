"""CPU tests for the F2d 'cpu' predictor mode (scripts/deepseek_v41/f2/*).

The 'cpu' mode moves the next-layer gate prediction ENTIRELY off the GPU and off the
measured routing barrier: the wrapper's barrier is just ``mx.eval(indices)``, the target
gate's params are materialized as numpy f32 ONCE at install (``HostGatePredictor``), and the
coordinator thread upcasts the raw bf16 router-input words and runs the same
``sqrt(softplus((x@W.T)/temp)) + bias`` max-over-rows in numpy.

Coverage: (a) host scores vs the native ``_gate_prefix_impl`` device path (max|delta|,
top-8 SET, top-3 of non-tied cases); (b) the raw-words hand-off is race-safe (fresh
per-call copy: an inline torn-vs-safe control + a threaded producer/slow-consumer hammer);
(c) main-thread us/call of the cpu hand-off vs the lean hand-off, excluding mx.eval; plus
the pool's bound cpu coordinator path and construction-time validation.

MLX pinned to CPU (MLX defaults to Metal and the GPU lock is NOT ours); run under
nice -n 19, pytest WITHOUT -n auto.
"""
from __future__ import annotations

import gc
import queue
import sys
import threading
import time
from pathlib import Path

# A sibling CPU-only module installs a _NoMLX meta-path finder; strip it before importing MLX.
sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "_NoMLX"]

import numpy as np
import pytest

import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts" / "deepseek_v41"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from f2 import predictor as fp  # noqa: E402
from f2.host_ring import HostRing, F2bCounters  # noqa: E402
from f2.speculative import SpeculativePool  # noqa: E402
from mtplx.models.deepseek_v41_moe import Gate, _gate_prefix_impl  # noqa: E402

N_ROUTED, HIDDEN, TEMP = 384, 5120, 1.0        # DSV4.1 text gate: [384, 5120], sqrtsoftplus
OFFS = (0, 6_266_880, 12_533_760)              # real weight-plane offsets in one record


@pytest.fixture(autouse=True)
def _cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(prev)


def _make_gate(*, dim=HIDDEN, n_routed=N_ROUTED, temp=TEMP, score_func="sqrtsoftplus", seed=11):
    from types import SimpleNamespace
    args = SimpleNamespace(hidden_size=dim, num_experts_per_tok=8, scoring_func=score_func,
                           gate_temp=temp, norm_topk_prob=True, routed_scaling_factor=1.0,
                           n_routed_experts=n_routed)
    mx.random.seed(seed)
    gate = Gate(7, args)
    gate.weight = (0.05 * mx.random.normal((n_routed, dim))).astype(mx.bfloat16)
    gate.e_score_correction_bias = (0.1 * mx.random.normal((n_routed,))).astype(mx.float32)
    mx.eval(gate.weight, gate.e_score_correction_bias)
    return gate


def _np64(a):
    return np.asarray(a.tolist(), dtype=np.float64)


def _rank_stats(ref, cand, k, rel_tol=1e-4):
    """(exact, hard): exact = top-k orderings identical; hard = top-k SETS differ by an
    expert whose ref score is NOT a near-tie with the k-th boundary (a real ranking error).
    Boundary ties (|score - boundary| < rel_tol*scale) are tolerated swaps."""
    ro = np.argsort(-ref, kind="stable")
    co = np.argsort(-cand, kind="stable")
    if list(ro[:k]) == list(co[:k]):
        return True, False
    scale = max(1e-12, float(np.max(np.abs(ref))))
    boundary = float(ref[ro[k - 1]])
    for e in set(ro[:k].tolist()) ^ set(co[:k].tolist()):
        if abs(float(ref[e]) - boundary) >= rel_tol * scale:
            return False, True
    return False, False


# ---------------------------------------------------------------------------
# (a) cpu-mode scores vs the native _gate_prefix_impl device path, rows 4..8
# ---------------------------------------------------------------------------
def test_cpu_scores_match_native_gate_prefix(capsys):
    gate = _make_gate()
    hp = fp.HostGatePredictor(gate)
    n_cases = 240
    max_abs = 0.0
    top8_set_eq = exact8 = hard8 = 0
    top3_nontie = top3_nontie_ok = hard3 = 0
    for i in range(n_cases):
        rows = 4 + (i % 5)                                     # rows 4..8
        x = (0.3 * mx.random.normal((rows, 1, HIDDEN))).astype(mx.bfloat16)
        mx.eval(x)
        # native device path (the router's own arithmetic), max over rows
        _s, biased = _gate_prefix_impl(x.reshape(-1, HIDDEN), gate.weight,
                                       gate.e_score_correction_bias, TEMP, "sqrtsoftplus")
        ref = _np64(mx.max(biased, axis=0))
        # cpu host path via the wrapper's EXACT hand-off copy
        words = fp.words_from_mx(x)
        cand = _np64(hp.scores_from_words(words, words.size // HIDDEN))
        max_abs = max(max_abs, float(np.max(np.abs(cand - ref))))

        ro8, co8 = np.argsort(-ref)[:8], np.argsort(-cand)[:8]
        top8_set_eq += int(set(ro8.tolist()) == set(co8.tolist()))
        e8, h8 = _rank_stats(ref, cand, 8)
        exact8 += int(e8); hard8 += int(h8)

        e3, h3 = _rank_stats(ref, cand, 3)
        hard3 += int(h3)
        # a case is "non-tied" at the top-3 boundary iff swapping there is not a float tie
        # (i.e. _rank_stats would flag a real divergence if the sets differed); count those.
        o = np.argsort(-ref)
        scale = max(1e-12, float(np.max(np.abs(ref))))
        tied3 = abs(float(ref[o[2]]) - float(ref[o[3]])) < 1e-4 * scale
        if not tied3:
            top3_nontie += 1
            top3_nontie_ok += int(set(o[:3].tolist()) == set(np.argsort(-cand)[:3].tolist()))

    with capsys.disabled():
        print(f"\n[f2d] cpu-vs-native ({n_cases} cases, rows 4..8): max|dmerged|={max_abs:.3e}")
        print(f"[f2d]   top-8 SET agreement = {top8_set_eq}/{n_cases}  "
              f"(exact top-8 order = {exact8}/{n_cases}, hard divergences = {hard8})")
        print(f"[f2d]   top-3 non-tied match = {top3_nontie_ok}/{top3_nontie}  "
              f"(hard top-3 divergences = {hard3})")
    assert max_abs < 1e-3                                     # float32 GEMM order only
    assert hard8 == 0                                         # no real top-8 divergence
    assert hard3 == 0                                         # top-3 of every NON-tied case matches
    assert top3_nontie_ok == top3_nontie


def test_cpu_bf16_to_f32_upcast_is_bit_exact():
    """The host upcast (words uint16 -> f32) reconstructs bf16 exactly (bf16 == the high 16
    bits of f32), so the host GEMM sees exactly the bytes the router does."""
    for rows in (4, 6, 8):
        x = (0.3 * mx.random.normal((rows, 1, HIDDEN))).astype(mx.bfloat16)
        mx.eval(x)
        words = fp.words_from_mx(x)
        host_x32 = (words.astype(np.uint32) << 16).view(np.float32).reshape(rows, HIDDEN)
        ref_x32 = np.asarray(x.astype(mx.float32)).reshape(rows, HIDDEN)
        assert np.array_equal(host_x32, ref_x32), f"rows={rows}"


# ---------------------------------------------------------------------------
# (b) hand-off race-safety: fresh per-call copy owns its data and cannot be torn
# ---------------------------------------------------------------------------
def test_view_would_tear_but_fresh_copy_is_safe():
    """Deterministic control proving the copy is load-bearing: a non-copying view of the
    same source buffer TEARS when the source is overwritten; ``words_from_mx``'s fresh copy
    does not."""
    src = np.full(64, 5, dtype=np.uint16)
    view = np.frombuffer(memoryview(src), np.uint16)          # NO copy (the unsafe variant)
    cp = fp.words_from_mx(src)                                # fresh copy (the shipped path)
    src[:] = 9                                                # producer overwrites / reuses
    assert view.tolist() == [9] * 64                          # the view TORE (shows the risk)
    assert cp.tolist() == [5] * 64                            # the copy is intact (safe)
    assert cp.base is None and cp.flags.owndata               # independent, self-owned


def test_words_from_mx_returns_independent_owned_array():
    x = (0.3 * mx.random.normal((6, 1, HIDDEN))).astype(mx.bfloat16)
    mx.eval(x)
    w = fp.words_from_mx(x)
    assert w.dtype == np.uint16 and w.size == 6 * HIDDEN
    assert w.base is None and w.flags.owndata and w.flags.writeable
    snap = w.copy()
    del x                                                     # source gone -> words unaffected
    gc.collect()
    assert np.array_equal(w, snap)


def test_handoff_hammer_producer_overwrites_while_consumer_reads(capsys):
    """Hammer: every word of a REUSED source buffer carries a per-epoch stamp; the producer
    overwrites the source in place immediately after each hand-off, while a deliberately slow
    consumer verifies each handed-off array is internally consistent (a single stamp -- no
    torn/mixed rows). Fresh-copy hand-off => the consumer never reads the buffer being
    overwritten."""
    n = 64 * 16
    src = np.zeros(n, dtype=np.uint16)                        # the reused source buffer
    q: queue.Queue = queue.Queue()
    torn = []
    M = 600

    def consumer():
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                return
            stamp, words = item
            for _ in range(3):
                time.sleep(0)                                 # yield: give the producer a chance to overwrite
            u = np.unique(words)
            if not (u.size == 1 and int(u[0]) == stamp):
                torn.append((stamp, u.tolist()[:4]))
            q.task_done()

    t = threading.Thread(target=consumer)
    t.start()
    try:
        for s in range(1, M + 1):
            src[:] = np.uint16(s)                             # stamp the reused buffer
            words = fp.words_from_mx(src)                     # FRESH copy of this stamp
            src[:] = np.uint16(0xFFFF)                        # immediately clobber / reuse the source
            q.put((s, words))
    finally:
        q.put(None)
        t.join(timeout=15)
    with capsys.disabled():
        print(f"\n[f2d] hand-off hammer: {M} epochs, torn/mixed reads = {len(torn)}")
    assert torn == []


# ---------------------------------------------------------------------------
# (c) main-thread cost microbench: cpu hand-off vs lean hand-off, excluding mx.eval
# ---------------------------------------------------------------------------
def test_main_thread_cost_cpu_vs_lean_excl_eval(capsys):
    import collections

    gate = _make_gate()
    lean = fp.GatePredictor(gate, mode="lean")
    hp = fp.HostGatePredictor(gate)
    x = (0.3 * mx.random.normal((6, 1, HIDDEN))).astype(mx.bfloat16)
    mx.eval(x)

    # Model the coordinator's inbox as already drained (production: calls are ~340 ms apart
    # and the coordinator consumes in ~0.3 ms, so no backlog): a maxlen=1 holder frees the
    # prior array each call, so malloc reuses the block -- the STEADY-STATE main-thread cost,
    # without the two artifacts a full queue.Queue introduces here (a piling-up backlog that
    # defeats malloc reuse, or a background drainer's per-put GIL handshake). The real
    # queue.put lock adds a small constant equally to both paths.
    inbox: collections.deque = collections.deque(maxlen=1)

    class _P:
        def current_epoch(self, t):
            return 0

        def submit_prediction(self, t, e, s):
            inbox.append(s)

        def submit_words(self, t, e, r, w):
            inbox.append(w)

    pool = _P()
    merged = lean.merged(x)
    mx.eval(merged)                                           # lean barrier already forced merged

    def lean_path():
        # lean main-thread work AFTER the barrier: copy the [n_routed] f32 prediction + submit.
        scores = np.asarray(merged, dtype=np.float32).copy()
        pool.submit_prediction(5, pool.current_epoch(5), scores)

    def cpu_path():
        # cpu main-thread work AFTER the barrier: fresh uint16 copy of x's words + submit.
        words = fp.words_from_mx(x)
        pool.submit_words(5, pool.current_epoch(5), words.size // HIDDEN, words)

    # Isolate the one real queue.put lock cost (path-identical), for reference.
    q: queue.Queue = queue.Queue()

    def put_roundtrip():
        q.put(0)
        q.get()

    def bench(fn, N):
        t0 = time.perf_counter()
        for _ in range(N):
            fn()
        return (time.perf_counter() - t0) / N * 1e6

    N = 3000
    for fn in (lean_path, cpu_path, put_roundtrip):
        bench(fn, 300)                                       # warm
    us_lean = bench(lean_path, N)
    us_cpu = bench(cpu_path, N)
    us_put = bench(put_roundtrip, N)

    with capsys.disabled():
        print(f"\n[f2d microbench] main-thread hand-off, EXCL. mx.eval (steady state): "
              f"lean={us_lean:.2f} us/call  cpu={us_cpu:.2f} us/call")
        print(f"[f2d microbench]   (cpu copies rows*{HIDDEN} u16 ~= 60 KB; lean copies "
              f"{N_ROUTED} f32; +~{us_put:.2f} us for the shared queue.put; cpu ALSO removes "
              f"the merged GEMM from the barrier's mx.eval)")
    assert us_cpu < 25.0                                     # target ~1-2 us; guard well under any hot-path budget


# ---------------------------------------------------------------------------
# (d-support) pool cpu coordinator path + bound-decode regression + validation
# ---------------------------------------------------------------------------
class _Reader:
    """Fake reader serving synthetic bytes by offset (mirrors test_dsv41_f2b)."""

    def _readv_range_into(self, name, offset, views, *, cancel_event=None, deadline_ns=None,
                          pipeline_phase=None):
        for v in views:
            arr = np.frombuffer(v, np.uint8)
            if not arr.flags.writeable:
                arr = arr.view()
                arr.flags.writeable = True
            arr[:] = (np.arange(arr.shape[0], dtype=np.uint8) + np.uint8(offset & 0x3F))


def test_pool_cpu_mode_coordinator_computes_scores_and_enqueues():
    """cpu-mode pool: submit_words -> coordinator upcasts+GEMMs via the host predictor ->
    the SAME plan_fn ranks -> planes enqueued & read. Proves the construction-bound cpu
    decode and that the coordinator's scores equal the host predictor's direct output."""
    gate = _make_gate()
    hp = fp.HostGatePredictor(gate)
    ring = HostRing(records=8, planes=3, plane_bytes=16, counters=F2bCounters())
    reader = _Reader()
    specs = tuple((off, 16) for off in OFFS)
    seen = {}
    sidecar = 1_000_000

    def plan_fn(target, scores):
        seen["scores"] = np.asarray(scores, dtype=np.float64)
        return [sidecar]                                     # one fixed record for the test

    def cpu_score_fn(t, rows, words):
        return hp.scores_from_words(words, rows)

    pool = SpeculativePool(reader, ring, plane_specs=specs, plan_fn=plan_fn, workers=2,
                           cpu_score_fn=cpu_score_fn)
    try:
        rows = 6
        x = (0.3 * mx.random.normal((rows, 1, HIDDEN))).astype(mx.bfloat16)
        mx.eval(x)
        words = fp.words_from_mx(x)
        pool.submit_words(5, pool.current_epoch(5), words.size // HIDDEN, words)
        pool.drain()
        assert np.allclose(seen["scores"], _np64(hp.scores_from_words(words, rows)))
        for off in OFFS:
            assert ring.state_of(sidecar + off) == "ready"
        assert ring.counters.planes_completed >= 3
        assert ring.counters.coordinator_batches >= 1
    finally:
        pool.shutdown()


def test_pool_scores_mode_still_works_when_no_cpu_score_fn():
    """Regression: with cpu_score_fn=None the coordinator decode is identity, so the
    'lean'/'native' 3-tuple (target, epoch, scores) path is unchanged."""
    ring = HostRing(records=8, planes=3, plane_bytes=16, counters=F2bCounters())
    reader = _Reader()
    specs = tuple((off, 16) for off in OFFS)
    pool = SpeculativePool(reader, ring, plane_specs=specs, plan_fn=lambda t, s: [500],
                           workers=2)                         # cpu_score_fn defaults to None
    try:
        pool.submit_prediction(5, pool.current_epoch(5), np.zeros(N_ROUTED, np.float32))
        pool.drain()
        assert ring.state_of(500 + OFFS[0]) == "ready"
        assert ring.counters.planes_completed >= 3
    finally:
        pool.shutdown()


def test_host_predictor_rejects_non_sqrtsoftplus_at_construction():
    gate = _make_gate(score_func="sigmoid")
    with pytest.raises(RuntimeError, match="sqrtsoftplus"):
        fp.HostGatePredictor(gate)


def test_host_predictor_bytes_and_param_shapes():
    gate = _make_gate()
    hp = fp.HostGatePredictor(gate)
    assert hp.weight.shape == (N_ROUTED, HIDDEN) and hp.weight.dtype == np.float32
    assert hp.bias.shape == (N_ROUTED,) and hp.bias.dtype == np.float32
    assert hp.dim == HIDDEN and hp.temp == TEMP
    expected = N_ROUTED * HIDDEN * 4 + N_ROUTED * 4           # 7,864,320 + 1,536
    assert hp.host_bytes == expected == 7_865_856            # ~7.9 MB per target layer


def test_scores_from_words_shape_and_dtype():
    gate = _make_gate()
    hp = fp.HostGatePredictor(gate)
    for rows in (4, 5, 6, 7, 8):
        x = mx.zeros((rows, 1, HIDDEN), dtype=mx.bfloat16)
        mx.eval(x)
        out = hp.scores_from_words(fp.words_from_mx(x), rows)
        assert out.shape == (N_ROUTED,) and out.dtype == np.float32


def test_wrap_run_cpu_barrier_is_one_eval_note_before_run_and_handoff():
    """The 'cpu' wrapper: the routing barrier is exactly ONE main-thread ``mx.eval(indices)``
    with a SINGLE arg (no merged / no extra eval output); a target notes demand AFTER the
    barrier and BEFORE the scheduled run; a source hands off (target, epoch, rows, words)
    ONCE after the run."""
    from f2 import install as f2i

    events = []

    class _Pool:
        def note_demand_imminent(self, t):
            events.append(("note", t))

        def current_epoch(self, t):
            return 7

        def submit_words(self, t, e, r, w):
            events.append(("submit", t, e, r, w.dtype, int(w.size)))

    class _Switch:
        def __init__(self):
            self._run = self._orig

        def _orig(self, x, indices, *, shared_work):
            events.append(("run", shared_work))
            return "RESULT"

    sw = _Switch()
    f2i._wrap_run_cpu(sw, _Pool(), own=5, is_source=True, is_target=True,
                      target_layer=6, dim=HIDDEN)

    x = (0.3 * mx.random.normal((6, 1, HIDDEN))).astype(mx.bfloat16)
    indices = mx.zeros((6, 8), dtype=mx.int32)
    mx.eval(x, indices)

    main = threading.main_thread()
    n = {"e": 0}
    arg_counts = []
    real = mx.eval

    def counting(*a, **k):
        if threading.current_thread() is main:
            n["e"] += 1
            arg_counts.append(len(a))
        return real(*a, **k)

    mx.eval = counting
    try:
        out = sw._run(x, indices, shared_work="SW")
    finally:
        mx.eval = real

    assert out == "RESULT"
    assert n["e"] == 1 and arg_counts == [1]                 # one barrier eval, indices only (no merged)
    assert [e[0] for e in events] == ["note", "run", "submit"]
    assert events[0] == ("note", 5)                          # note_demand_imminent(own) AFTER barrier, BEFORE run
    _, tgt, epoch, rows, dtype, size = events[2]
    assert (tgt, epoch, rows, dtype, size) == (6, 7, 6, np.uint16, 6 * HIDDEN)


def test_wrap_run_cpu_target_only_notes_but_does_not_hand_off():
    """A target-only wrapped layer (not a source) completes the barrier + notes demand but
    hands nothing off."""
    from f2 import install as f2i

    events = []

    class _Pool:
        def note_demand_imminent(self, t):
            events.append(("note", t))

        def current_epoch(self, t):  # pragma: no cover - must not be called
            events.append(("epoch", t))
            return 0

        def submit_words(self, *a):  # pragma: no cover - must not be called
            events.append(("submit",))

    class _Switch:
        def __init__(self):
            self._run = lambda x, indices, *, shared_work: events.append(("run",))

    sw = _Switch()
    f2i._wrap_run_cpu(sw, _Pool(), own=39, is_source=False, is_target=True,
                      target_layer=None, dim=0)
    x = mx.zeros((6, 1, HIDDEN), dtype=mx.bfloat16)
    indices = mx.zeros((6, 8), dtype=mx.int32)
    mx.eval(x, indices)
    sw._run(x, indices, shared_work="SW")
    assert [e[0] for e in events] == ["note", "run"]         # no submit, no current_epoch
    assert events[0] == ("note", 39)
