"""CPU tests for the F16 verify-row-group pipeline (scripts/deepseek_v41/f16).

Pins MLX to CPU; tiny random DeepSeek-V4.1 configs (no artifact); no GPU/Metal.
Proves, for M in {5,6,7,8}:
  * ``Pipeline.pipelined_forward`` == running ``forward(A)`` then ``forward(B)``
    sequentially, BYTE-FOR-BYTE (logits, main_hidden, every layer cache store, engram
    state, cache.offset) -- with and without an engram hook wired; and that a
    subsequent trim + next forward is byte-identical too;
  * the greenlet driver produces a causally-valid schedule against a mock switch whose
    run has the retained begin/yield/finish shape and per-layer locks that RAISE on a
    violation (double-acquire), including the throw-into-suspended-partner exception
    path and rows<=4 passthrough, and it spawns NO threads (all ops on the caller);
  * the group greenlets inherit the caller's routing phase (a fresh greenlet starts
    with an EMPTY context, so gr_context=copy_context() carries it) and stay on the
    one calling thread/stream -- a verify group would otherwise route as PREFILL;
  * the three staged edits round-trip on the REAL archived helpers, refuse a
    double-apply, and compose with the F6 and F12 stagers;
  * the cloned per-group forward mirrors the live ``_forward_span`` + Model tail
    line-for-line, and the yield-run derivation inserts exactly one hand-off.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
import textwrap
import threading
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts" / "deepseek_v41"
_PACKED = _REPO / "docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed"
# .f16-site holds greenlet (private target dir, NOT the shared venv); it must be on
# sys.path before f16.pipeline (which imports greenlet) is imported.  For the GPU arm
# the same dir goes on PYTHONPATH.
_F16_SITE = _REPO / ".f16-site"
for _p in (str(_SCRIPTS), str(_PACKED), str(_F16_SITE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import greenlet  # noqa: E402  (from .f16-site)

from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402
from mtplx.models.deepseek_v41_cache import make_cache  # noqa: E402
from mtplx.engram_v41 import NgramHashState, n_hash_cols  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from f16 import pipeline as pl  # noqa: E402
from f16 import stage_f16_runner as stager  # noqa: E402
from f16.pipeline import (  # noqa: E402
    LEADER,
    TRAILER,
    Pipeline,
    _PipelineAborted,
    f16_yield,
    issue_suppressed,
)
_ENG_LAYERS = (1, 3)
_DSPARK_LAYERS = (2, 4)


# ---------------------------------------------------------------------------
# fixtures (mirror tests/models/test_deepseek_v41_layer_major_prefill.py)
# ---------------------------------------------------------------------------
def _args(**over) -> ModelArgs:
    base = dict(
        vocab_size=48, hidden_size=32, num_hidden_layers=8,
        num_attention_heads=4, head_dim=16, qk_rope_head_dim=4,
        q_lora_rank=12, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=2,
        index_n_heads=2, index_head_dim=8, index_topk=5,
        sliding_window=8, window_size=8, swiglu_limit=0.5,
        compress_ratios=[0, 0, 2, 2, 2, 1, 1, 1],
        kv_source_layer_ids=[2, 5], index_source_layer_ids=[2, 5, 6],
        candidate_source_layer_id=5, candidate_topk_blocks=3, candidate_block_size=2,
        rope_scaling={"rope_type": "yarn", "factor": 16, "beta_fast": 32,
                      "beta_slow": 1, "original_max_position_embeddings": 65536},
        dspark_target_layer_ids=list(_DSPARK_LAYERS),
    )
    base.update(over)
    return ModelArgs(**base)


def _randomize(model, seed=1, scale=0.1):
    mx.random.seed(seed)
    new = []
    for name, arr in tree_flatten(model.parameters()):
        if arr.ndim == 1 and ("norm_weight" in name or name.endswith("norm.weight")):
            v = 1.0 + 0.2 * mx.random.normal(arr.shape)
        elif "attn_sink" in name:
            v = 0.5 * mx.random.normal(arr.shape)
        else:
            v = scale * mx.random.normal(arr.shape)
        new.append((name, v.astype(mx.float32)))
    model.update(tree_unflatten(new))
    mx.eval(model.parameters())


def _engram_proto(vocab):
    nl = len(_ENG_LAYERS)
    rs = np.random.RandomState
    return NgramHashState(
        layer_ids=list(_ENG_LAYERS), max_ngram_size=3, n_heads=2,
        token_map=list(range(vocab)),
        multipliers=rs(0).randint(1, 7, size=(nl, 3)),
        primes=rs(1).randint(3, 97, size=(nl, 2, 2)),
        flat_offsets=rs(2).randint(0, 50, size=(nl, n_hash_cols(3, 2))),
        pad_compressed=0,
    )


class _FakeEngramHook:
    """Adds a deterministic function of this chunk's engram row ids (so the hidden
    state genuinely depends on them; a mis-ordered replay would diverge)."""

    def __init__(self, layer_hash_index: int):
        self.lhi = int(layer_hash_index)

    def __call__(self, h, token_ids, cache_state):
        row_ids = cache_state.current_row_ids(self.lhi)
        B, L = int(h.shape[0]), int(h.shape[1])
        assert tuple(row_ids.shape[:2]) == (B, L)
        add = (row_ids.astype(np.float64).sum(-1) % 7).astype(np.float32)
        return h + 0.01 * mx.array(add)[:, :, None, None]


def _attach_fake_engram(model):
    for layer in model.model.layers:
        layer.engram_hook = (
            _FakeEngramHook(_ENG_LAYERS.index(layer.layer_id))
            if layer.layer_id in _ENG_LAYERS else None
        )


def _cache_snapshot(cache):
    out = {}
    for i, lc in enumerate(cache.layers):
        for nm in ("window", "compress_kv", "index_k"):
            a = getattr(lc, nm, None)
            if a is not None:
                out[f"{i}.{nm}"] = np.array(a)
        cs = getattr(lc, "comp_state", None)
        if cs is not None:
            for nm in ("raw_kv", "raw_score"):
                a = getattr(cs, nm, None)
                if a is not None:
                    out[f"{i}.cs.{nm}"] = np.array(a)
    return out, int(cache.offset)


def _forward(model):
    def fwd(ids, cache):
        return model(ids, cache=cache, return_hidden=True)

    return fwd


def _make_cache(args, model, use_engram):
    if use_engram:
        return make_cache(args, engram_state=_engram_proto(args.vocab_size).fresh())
    return model.make_cache()


def _built_model(seed=1, use_engram=False):
    args = _args()
    model = Model(args)
    _randomize(model, seed=seed)
    if use_engram:
        _attach_fake_engram(model)
    return args, model


# ---------------------------------------------------------------------------
# 1. bitwise parity: pipelined_forward == forward(A) then forward(B)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rows", [5, 6, 7, 8])
@pytest.mark.parametrize("use_engram", [False, True])
@pytest.mark.parametrize("seed,scale", [(1, 0.1), (11, 0.3)])
def test_pipelined_forward_is_bitwise_sequential(rows, use_engram, seed, scale):
    args = _args()
    model = Model(args)
    _randomize(model, seed=seed, scale=scale)
    if use_engram:
        _attach_fake_engram(model)
    forward = _forward(model)
    ids = mx.array(np.random.RandomState(rows).randint(0, args.vocab_size, size=(1, rows)))
    ids_a, ids_b = ids[:, :4], ids[:, 4:]

    # oracle: two sequential causal forwards on one cache (the F2b-proved arithmetic).
    c2 = _make_cache(args, model, use_engram)
    la, ma = forward(ids_a, c2)
    lb, mb = forward(ids_b, c2)
    mx.eval(la, ma, lb, mb)
    oracle_logits = mx.concatenate([la, lb], axis=1)
    oracle_mh = mx.concatenate([ma, mb], axis=1)
    mx.eval(oracle_logits, oracle_mh)
    snap2, off2 = _cache_snapshot(c2)
    eng2 = np.array(c2.engram_state.state) if use_engram else None

    # pipeline
    c1 = _make_cache(args, model, use_engram)
    pipe = Pipeline(model, armed=True)
    logits, mh = pipe.pipelined_forward(forward, ids, c1)
    mx.eval(logits, mh)
    snap1, off1 = _cache_snapshot(c1)
    eng1 = np.array(c1.engram_state.state) if use_engram else None

    assert pipe.counters["pipelined_forwards"] == 1
    assert bool(mx.array_equal(logits, oracle_logits).item()), "logits differ"
    assert bool(mx.array_equal(mh, oracle_mh).item()), "main_hidden differs"
    assert off1 == off2 == rows
    assert set(snap1) == set(snap2)
    for k, v in snap2.items():
        assert np.array_equal(snap1[k], v), f"cache[{k}] differs"
    if use_engram:
        assert np.array_equal(eng1, eng2), "engram state differs"


def test_pipelined_forward_then_trim_then_next_is_bitwise():
    """A cycle accepts some drafts and trims the rest; the next forward on the
    trimmed cache must be byte-identical to the sequential path's."""
    args, model = _built_model(seed=7, use_engram=True)
    forward = _forward(model)
    rows, kept = 8, 4  # accepted 3 drafts + the primary; trim rows-kept
    ids = mx.array(np.random.RandomState(2).randint(0, args.vocab_size, size=(1, rows)))
    ida, idb = ids[:, :4], ids[:, 4:]
    nxt = mx.array(np.random.RandomState(3).randint(0, args.vocab_size, size=(1, 1)))

    c2 = _make_cache(args, model, True)
    forward(ida, c2)
    forward(idb, c2)
    c2.trim(rows - kept)
    nl2, _ = forward(nxt, c2)
    mx.eval(nl2)
    snap2, off2 = _cache_snapshot(c2)

    c1 = _make_cache(args, model, True)
    pipe = Pipeline(model, armed=True)
    pipe.pipelined_forward(forward, ids, c1)
    c1.trim(rows - kept)
    nl1, _ = forward(nxt, c1)
    mx.eval(nl1)
    snap1, off1 = _cache_snapshot(c1)

    assert off1 == off2 == kept + 1
    assert bool(mx.array_equal(nl1, nl2).item()), "post-trim next-forward logits differ"
    assert set(snap1) == set(snap2)
    for k, v in snap2.items():
        assert np.array_equal(snap1[k], v), f"post-trim cache[{k}] differs"


def test_multi_cycle_decode_is_bitwise_sequential():
    """Several verify cycles on ONE accumulating cache (verify -> accept -> trim ->
    next verify), through one reused Pipeline, must stay byte-identical to the
    sequential (4,4) path -- catches any cross-call baton / thread-local / cache leak."""
    args, model = _built_model(seed=3, use_engram=True)
    forward = _forward(model)
    pipe = Pipeline(model, armed=True)
    cycles = [(6, 3), (8, 2), (7, 5), (5, 4)]  # (verify rows, kept = accepted+1)

    c_seq = _make_cache(args, model, True)
    c_pipe = _make_cache(args, model, True)
    rng = np.random.RandomState(42)
    for i, (rows, kept) in enumerate(cycles):
        ids = mx.array(rng.randint(0, args.vocab_size, size=(1, rows)))
        ls_a, _ = forward(ids[:, :4], c_seq)
        ls_b, _ = forward(ids[:, 4:], c_seq)
        seq_logits = mx.concatenate([ls_a, ls_b], axis=1)
        pipe_logits, _ = pipe.pipelined_forward(forward, ids, c_pipe)
        mx.eval(seq_logits, pipe_logits)
        assert bool(mx.array_equal(pipe_logits, seq_logits).item()), f"cycle {i} logits differ"
        c_seq.trim(rows - kept)
        c_pipe.trim(rows - kept)
        s_seq, off_seq = _cache_snapshot(c_seq)
        s_pipe, off_pipe = _cache_snapshot(c_pipe)
        assert off_seq == off_pipe
        for k, v in s_seq.items():
            assert np.array_equal(s_pipe[k], v), f"cycle {i} cache[{k}] differs"
    assert pipe.counters["pipelined_forwards"] == len(cycles)


def test_parity_holds_under_the_real_routing_context():
    """The real decode loop runs the verify inside ``_verify_routing_context`` (DECODE
    phase).  Running BOTH the pipeline and the sequential oracle under that context
    must still be byte-identical (and the group threads inherit the phase)."""
    from mtplx.attention_context import attention_phase
    from mtplx.models.expert_mlx import expert_routing_phase
    from mtplx.expert_streaming import RoutingPhase

    args, model = _built_model(seed=11, use_engram=True)
    forward = _forward(model)
    ids = mx.array(np.random.RandomState(7).randint(0, args.vocab_size, size=(1, 6)))
    ctx = lambda: (attention_phase("decode_verify"),  # noqa: E731
                   expert_routing_phase(RoutingPhase.DECODE))

    c2 = _make_cache(args, model, True)
    with ctx()[0], ctx()[1]:
        la, _ = forward(ids[:, :4], c2)
        lb, _ = forward(ids[:, 4:], c2)
        oracle = mx.concatenate([la, lb], axis=1)

    c1 = _make_cache(args, model, True)
    pipe = Pipeline(model, armed=True)
    with ctx()[0], ctx()[1]:
        got, _ = pipe.pipelined_forward(forward, ids, c1)
    mx.eval(oracle, got)
    assert bool(mx.array_equal(got, oracle).item())


# ---------------------------------------------------------------------------
# 2. rows<=4 passthrough + unarmed passthrough
# ---------------------------------------------------------------------------
def test_rows_le_4_passthrough_single_forward():
    args, model = _built_model(seed=5)
    forward = _forward(model)
    pipe = Pipeline(model, armed=True)
    calls = {"n": 0}

    def counting(ids, cache):
        calls["n"] += 1
        return forward(ids, cache)

    for rows in (1, 2, 3, 4):
        ids = mx.array(np.random.RandomState(rows).randint(0, args.vocab_size, size=(1, rows)))
        got = pipe.pipelined_forward(counting, ids, model.make_cache())
        want = forward(ids, model.make_cache())
        mx.eval(got[0], want[0])
        assert bool(mx.array_equal(got[0], want[0]).item())
    assert pipe.counters == {"calls": 4, "single_forwards": 4,
                             "pipelined_forwards": 0, "handoffs": 0}
    assert calls["n"] == 4


def test_unarmed_pipeline_is_passthrough():
    args, model = _built_model(seed=6)
    forward = _forward(model)
    pipe = Pipeline(model, armed=False)
    ids = mx.array(np.random.RandomState(9).randint(0, args.vocab_size, size=(1, 6)))
    got = pipe.pipelined_forward(forward, ids, model.make_cache())
    want = forward(ids, model.make_cache())
    mx.eval(got[0], want[0])
    assert bool(mx.array_equal(got[0], want[0]).item())
    assert pipe.counters == {"calls": 0, "single_forwards": 0,
                             "pipelined_forwards": 0, "handoffs": 0}


# ---------------------------------------------------------------------------
# 3. greenlet hygiene: routing phase (ContextVar via gr_context) + same stream
# ---------------------------------------------------------------------------
def test_group_greenlets_inherit_routing_phase_and_stay_on_calling_stream():
    from mtplx.attention_context import attention_phase, current_attention_phase
    from mtplx.models.expert_mlx import expert_routing_phase, current_expert_routing_phase
    from mtplx.expert_streaming import RoutingPhase

    _args_, model = _built_model(seed=1)
    pipe = Pipeline(model, armed=True)
    main_stream = repr(mx.default_stream(mx.default_device()))

    def probe():
        # token_count=4 (>1): without the DECODE ContextVar this derives PREFILL. A
        # fresh greenlet starts with an EMPTY context, so gr_context=copy_context() is
        # what carries the phase in.
        return (current_attention_phase(),
                current_expert_routing_phase(token_count=4),
                repr(mx.default_stream(mx.default_device())))

    with attention_phase("decode_verify"), expert_routing_phase(RoutingPhase.DECODE):
        results = pipe._run_groups(probe, probe)
    for ap, rp, st in results:
        assert ap == "decode_verify", "attention phase did not propagate to a group greenlet"
        assert rp is RoutingPhase.DECODE, "routing phase leaked to PREFILL on a group greenlet"
        assert st == main_stream, "group greenlet used a different MLX stream"


def test_f16_yield_and_issue_suppressed_are_noops_outside_a_group():
    # The current (main) greenlet has no _f16_group attribute.
    f16_yield()  # no-op, no switch, no raise
    assert issue_suppressed() is False


# ---------------------------------------------------------------------------
# 4. greenlet schedule against a mock switch with per-layer locks that RAISE
# ---------------------------------------------------------------------------
class _ScheduleViolation(Exception):
    pass


class _RaisingLock:
    """Per-layer route lock that RAISES if acquired while already held -- so a driver
    that lets B enter a layer A has not left is caught immediately.  (Greenlets share
    one thread, so identity is by role, not thread ident.)"""

    def __init__(self, idx):
        self.idx = idx
        self._holder = None

    def acquire(self, role):
        if self._holder is not None:
            raise _ScheduleViolation(
                f"layer {self.idx} lock held by role {self._holder} on acquire by {role}"
            )
        self._holder = role

    def release(self):
        self._holder = None


class _MockRuntime:
    """Mimics the retained flush/begin/defer locking: begin(L) takes the per-layer
    lock, defer(L) queues its release, flush drains the deferred releases -- exactly
    plane_lane run head flush + begin_split_route + defer_slot_release."""

    def __init__(self, n_layers):
        self.locks = [_RaisingLock(i) for i in range(n_layers)]
        self.deferred = []
        self.entered = []  # (role, layer) in schedule order

    def flush(self):
        while self.deferred:
            self.deferred.pop(0).release()

    def begin(self, layer, role):
        self.locks[layer].acquire(role)
        self.entered.append((role, layer))

    def defer(self, layer):
        self.deferred.append(self.locks[layer])


def _mock_group(runtime, role, n_layers, fail_at=None):
    def run():
        for layer in range(n_layers):
            runtime.flush()
            runtime.begin(layer, role)  # acquire lock[layer] (raises on violation)
            if fail_at is not None and layer == fail_at:
                runtime.locks[layer].release()  # mimic run() abort/close cleanup
                raise RuntimeError(f"mock fail role={role} layer={layer}")
            try:
                f16_yield()  # the real greenlet hand-off
            except _PipelineAborted:
                runtime.locks[layer].release()  # mimic abort/close on the current route
                raise
            runtime.defer(layer)
        return ("done", role)

    return run


def test_greenlet_schedule_is_causally_valid_and_interleaves():
    _args_, model = _built_model(seed=1)
    pipe = Pipeline(model, armed=True)
    n = 12
    rt = _MockRuntime(n)
    # No _ScheduleViolation is raised => the per-layer locks were never double-held =>
    # B never entered a layer A had not left (the causality the design requires).
    results = pipe._run_groups(_mock_group(rt, LEADER, n), _mock_group(rt, TRAILER, n))
    assert results == [("done", LEADER), ("done", TRAILER)]
    assert pipe.counters["handoffs"] > 2, "driver never alternated -> no interleave"

    # KV causality: when B begins layer L, A has already RUN layer L (leader_seen >=
    # L+1), so A's layer-L KV is written before B reads it.  (The stronger lock order
    # -- A must have entered L+1 -- is proven by the absence of _ScheduleViolation:
    # the per-layer lock would RAISE if B entered a layer A had not released.)
    leader_seen = 0
    for role, layer in rt.entered:
        if role == LEADER:
            leader_seen += 1
        else:
            assert leader_seen >= layer + 1, (
                f"trailer entered layer {layer} with only {leader_seen} leader layers done"
            )
    # both groups eventually ran every layer, interleaved (not fully sequential)
    assert sum(1 for r, _ in rt.entered if r == LEADER) == n
    assert sum(1 for r, _ in rt.entered if r == TRAILER) == n
    first_trailer = next(i for i, (r, _) in enumerate(rt.entered) if r == TRAILER)
    assert any(r == LEADER for r, _ in rt.entered[first_trailer:]), "no interleave after B started"


def test_mock_lock_actually_detects_a_violation():
    """Guard against a vacuous positive test: driving the locks in a bad order (a
    layer entered while still held) must raise."""
    rt = _MockRuntime(3)
    rt.begin(0, LEADER)
    with pytest.raises(_ScheduleViolation):
        rt.begin(0, TRAILER)  # lock 0 still held -> violation


def test_exception_path_throws_into_suspended_partner_and_reraises():
    """When one group raises, the driver throws _PipelineAborted into the SUSPENDED
    partner so its run cleanup runs (lock released), then re-raises the ORIGINAL error.
    No route lock may be left held (it would wedge the layer lock)."""
    _args_, model = _built_model(seed=1)
    pipe = Pipeline(model, armed=True)
    n = 8
    rt = _MockRuntime(n)
    with pytest.raises(RuntimeError, match="mock fail role=0 layer=3"):
        pipe._run_groups(_mock_group(rt, LEADER, n, fail_at=3), _mock_group(rt, TRAILER, n))
    assert all(lk._holder is None for lk in rt.locks), "a route lock was left held on abort"


def test_pipelined_forward_creates_no_threads_and_runs_on_calling_thread():
    """The greenlet driver must not spawn threads, and every group's MLX ops must run
    on the calling thread (MLX GPU streams are thread-bound)."""
    args, model = _built_model(seed=1, use_engram=True)
    forward = _forward(model)
    pipe = Pipeline(model, armed=True)
    ids = mx.array(np.random.RandomState(1).randint(0, args.vocab_size, size=(1, 6)))

    calling = threading.get_ident()
    seen = set()
    orig = pipe._group_forward

    def spy(*a, **k):
        seen.add(threading.get_ident())
        return orig(*a, **k)

    pipe._group_forward = spy
    before = threading.active_count()
    logits, _ = pipe.pipelined_forward(forward, ids, _make_cache(args, model, True))
    mx.eval(logits)
    assert threading.active_count() == before, "pipelined forward spawned a thread"
    assert seen == {calling}, "a group ran off the calling thread"
    assert pipe.counters["pipelined_forwards"] == 1


# ---------------------------------------------------------------------------
# 5. yield-run derivation + source pins (clone faithfulness)
# ---------------------------------------------------------------------------
def test_f16_yield_run_inserts_one_handoff_with_roundtrip():
    import plane_lane
    import projection_install

    base = textwrap.dedent(inspect.getsource(plane_lane.PackedDecode.run))
    sched = projection_install.scheduled_run_source(base)
    yielded = pl.f16_run_source(sched)
    assert yielded.count("self._f16_yield()") == 1
    lines = yielded.splitlines()
    yi = next(i for i, ln in enumerate(lines) if ln.strip() == "self._f16_yield()")
    assert lines[yi + 1].strip() == "ready_iter = pending.iter_ready_misses()"
    # round-trip: removing the one inserted line recovers the scheduled source lines
    recovered = [ln for ln in lines if ln.strip() != "self._f16_yield()"]
    assert recovered == sched.splitlines()
    # double-insert refused
    with pytest.raises(RuntimeError):
        pl.f16_run_source(yielded)


def test_yield_run_compiles_and_is_distinguishable_from_scheduled():
    run_fn, shas = pl.build_yield_run()
    assert callable(run_fn)
    assert run_fn.__code__.co_code != pl.scheduled_run_cocode(), (
        "yield run bytecode must differ from the scheduled run (the install refuse check)"
    )
    assert shas["retained_plane_lane_sha256"] == pl.RETAINED_PLANE_LANE_SHA256


def test_scheduled_run_cocode_matches_a_fresh_scheduled_compile():
    import plane_lane
    import projection_install

    base = textwrap.dedent(inspect.getsource(plane_lane.PackedDecode.run))
    sched = projection_install.scheduled_run_source(base)
    ns = dict(plane_lane.__dict__)
    exec(compile(sched, "<ref>", "exec"), ns)  # noqa: S102
    assert ns["run"].__code__.co_code == pl.scheduled_run_cocode()


def test_source_pins_match_live_runtime():
    report = pl.verify_source_pins()  # raises on drift
    assert report["_forward_span"] == pl.FORWARD_SPAN_SHA256
    assert report["Model.__call__"] == pl.MODEL_CALL_SHA256
    assert report["_ChunkEngramView"] == pl.CHUNK_ENGRAM_VIEW_SHA256


def test_clone_mirrors_live_forward_span_and_model_tail_line_for_line():
    """Diff the clone's arithmetic against the live source: every load-bearing line
    the per-group forward reproduces must appear verbatim in the pinned runtime."""
    from mtplx.models import deepseek_v41 as dv

    span = textwrap.dedent(inspect.getsource(dv.DeepseekV41Backbone._forward_span))
    for line in (
        "positions = mx.arange(cache.offset, cache.offset + s)",
        "h = self.embed_tokens(input_ids)",
        "h = mx.broadcast_to(h[:, :, None, :], (b, s, self.hc_mult, h.shape[-1]))",
        "[mx.ones((b, s, 1)), mx.zeros((b, s, self.hc_mult - 1))], axis=-1",
        "engram_state.advance(input_ids)",
        "main_hiddens.append(mx.mean(h.astype(mx.float32), axis=2).astype(h.dtype))",
        "h, pre_mix = layer(h, pre_mix, positions, cache.layers[layer.layer_id], shared)",
        "cache.advance(s)",
        "h = mx.sum(pre_mix[..., None] * h.astype(mx.float32), axis=2).astype(h.dtype)",
        "out = _rmsnorm(h, self.norm_weight, self.args.rms_norm_eps)",
    ):
        assert line in span, f"clone-mirrored line missing from live _forward_span: {line}"

    call = textwrap.dedent(inspect.getsource(dv.Model.__call__))
    assert "logits = self._apply_head(source)" in call
    assert "return logits, main_hidden" in call

    lm = textwrap.dedent(inspect.getsource(dv.DeepseekV41Backbone._forward_layer_major))
    assert "engram_currents.append(engram_state.advance(ids_c))" in lm
    assert "_ChunkEngramView(engram_currents[c])" in lm
    assert "offset0 = int(cache.offset)" in lm


# ---------------------------------------------------------------------------
# 6. stager: round-trip on the REAL archived helpers, double-apply, composition
# ---------------------------------------------------------------------------
def _archived(name):
    return (_PACKED / name).read_text()


def test_stage_run_full_roundtrips_on_archived_source():
    src = _archived("run_full.py")
    out = stager.stage_run_full(src)
    assert out != src and "_f16_install" in out and "install_from_env(target)" in out
    # explicit reverse: strip the two inserted lines -> recover the original
    inserted = ("        import f16.install as _f16_install  # F16\n"
                "        _f16_install.install_from_env(target)  # F16 (armed only if MTPLX_DSV41_F16=1)\n")
    assert out.replace(inserted, "") == src


def test_stage_hybrid_install_roundtrips_and_preserves_mx_calls():
    src = _archived("hybrid_install.py")
    out = stager.stage_hybrid_install(src)
    assert "_F16_PIPELINE.pipelined_forward(forward, mx.array([chunk_ids]), cache)" in out
    assert "namespace['_F16_PIPELINE'] = _F16Lazy(model)" in out
    # the staged rewrite() still recovers the native source AND keeps the mx-call set:
    # run it against the pinned _decode_cycles source (its own round-trip + AST check).
    ns = dict(vars(__import__("hybrid_install")))
    exec(compile(out, "<staged_hybrid>", "exec"), ns)  # noqa: S102
    from mtplx.models import deepseek_v41_dspark_decode as dec

    dec_src = inspect.getsource(dec._decode_cycles)
    updated = ns["rewrite"](dec_src)  # raises if round-trip or mx-call check fails
    assert "_F16_PIPELINE.pipelined_forward" in updated


def test_stage_projection_install_roundtrips_to_four_buffers():
    src = _archived("projection_install.py")
    out = stager.stage_projection_install(src)
    assert "[None, None, None, None]" in out
    assert out.count("layer % 4") == 2  # issue() + ScheduledOutput.buffer_index
    assert "len(store.buffers) != 4" in out
    assert "lane.buffer_index != index % 4" in out
    assert "layer % 2" not in out and "index % 2" not in out


@pytest.mark.parametrize("fn,name", [
    (stager.stage_run_full, "run_full.py"),
    (stager.stage_hybrid_install, "hybrid_install.py"),
    (stager.stage_projection_install, "projection_install.py"),
])
def test_double_apply_is_refused(fn, name):
    out = fn(_archived(name))
    with pytest.raises(RuntimeError):
        fn(out)


def _load_stage_fn(worktree, subpkg, module):
    path = (Path(f"/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/{worktree}")
            / "scripts" / "deepseek_v41" / subpkg / module)
    spec = importlib.util.spec_from_file_location(f"_{subpkg}_{module}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.stage


def test_composes_with_f6_and_f12_stagers():
    f6_stage = _load_stage_fn("dsv41-f6-engram", "f6", "stage_f6_runner.py")
    f12_stage = _load_stage_fn("dsv41-f12-growth", "f12", "stage_f12_runner.py")

    rf = _archived("run_full.py")
    # F16 (prime_model anchor) and F6 (describe_engram + growth_transition anchors)
    # both edit run_full.py at disjoint anchors -> compose in either order.
    a = stager.stage_run_full(f6_stage(rf))
    b = f6_stage(stager.stage_run_full(rf))
    for out in (a, b):
        assert "_f16_install" in out and "install_from_env(target)" in out
        assert "install_from_env" in out  # F6's install point marker also present
    assert rf.count(stager._RF_ANCHOR) == f6_stage(rf).count(stager._RF_ANCHOR) == 1

    # F12 edits packed_phase.py -- a file F16 never touches; it still stages cleanly.
    pp = _archived("packed_phase.py")
    assert "_f12_finish" in f12_stage(pp)
    assert "_F16_PIPELINE" not in pp  # F16 has no packed_phase edit


# ---------------------------------------------------------------------------
# 7. install refuse mechanism (co_code discrimination), CPU-only
# ---------------------------------------------------------------------------
def test_install_refuse_distinguishes_lanes_by_cocode():
    """The install-time refuse check compares switch._run.__func__.co_code to the
    scheduled run's; a wrapped closure (no __func__, F2b-shape) or a differing
    bytecode (F5/F16-shape) must not match."""
    sched = pl.scheduled_run_cocode()
    yield_fn, _ = pl.build_yield_run()
    assert yield_fn.__code__.co_code != sched          # F16 yield run rejected as "scheduled"
    plain_closure = (lambda x: x)                       # F2b-style wrap: no __func__
    assert getattr(plain_closure, "__func__", None) is None


def test_lazy_pipeline_resolves_the_install_at_call_time():
    """The hybrid rewrite is installed BEFORE prefill; F16's install stashes the pipeline
    AFTER it. The injected object must not touch ``model._f16_pipeline`` until a verify call."""
    import types

    from f16.pipeline import LazyPipeline

    model = types.SimpleNamespace()                      # no _f16_pipeline yet
    lazy = LazyPipeline(model)                           # must not raise
    calls = []
    model._f16_pipeline = types.SimpleNamespace(
        pipelined_forward=lambda forward, ids, cache: calls.append((forward, ids, cache)) or "out"
    )
    assert lazy.pipelined_forward("f", "ids", "cache") == "out"
    assert calls == [("f", "ids", "cache")]


@pytest.mark.parametrize("rows", [5, 6, 7, 8])
@pytest.mark.parametrize("use_engram", [False, True])
def test_balanced_split_is_bitwise_sequential_ceil_floor(rows, use_engram):
    """``split='balanced'``: leader = ceil(rows/2), trailer = floor(rows/2); bit-identical to two
    sequential causal forwards of exactly those sizes (the balanced oracle's arithmetic)."""
    args = _args()
    model = Model(args)
    _randomize(model, seed=7, scale=0.2)
    if use_engram:
        _attach_fake_engram(model)
    forward = _forward(model)
    ids = mx.array(np.random.RandomState(100 + rows).randint(0, args.vocab_size, size=(1, rows)))
    lead = (rows + 1) // 2

    c2 = _make_cache(args, model, use_engram)
    la, ma = forward(ids[:, :lead], c2)
    lb, mb = forward(ids[:, lead:], c2)
    oracle_logits = mx.concatenate([la, lb], axis=1)
    oracle_mh = mx.concatenate([ma, mb], axis=1)
    mx.eval(oracle_logits, oracle_mh)
    snap2, off2 = _cache_snapshot(c2)

    c1 = _make_cache(args, model, use_engram)
    pipe = Pipeline(model, armed=True, split="balanced")
    logits, mh = pipe.pipelined_forward(forward, ids, c1)
    mx.eval(logits, mh)
    snap1, off1 = _cache_snapshot(c1)

    assert pipe.counters["pipelined_forwards"] == 1
    assert bool(mx.array_equal(logits, oracle_logits).item()), "logits differ"
    assert bool(mx.array_equal(mh, oracle_mh).item()), "main_hidden differs"
    assert off1 == off2 == rows
    for k, v in snap2.items():
        assert np.array_equal(snap1[k], v), f"cache[{k}] differs"


def test_unknown_split_is_refused_at_construction():
    args, model = _built_model()
    with pytest.raises(RuntimeError):
        Pipeline(model, armed=True, split="thirds")
