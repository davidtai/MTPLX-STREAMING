"""CPU tests for the F18 barrier hand-off (scripts/deepseek_v41/f16, handoff='barrier').

Pins MLX to CPU at import (MLX defaults to Metal); tiny random DeepSeek-V4.1 configs
(no artifact); no GPU/Metal.  Covers, for the two-hand-off (barrier + reads) interleave:

  1. Bitwise parity vs sequential chunks -- both splits, rows 5-8, engram on/off,
     trim-then-next and multi-cycle -- driven by a TEST-ONLY switch hook that fires
     ``pipeline.barrier(indices)`` / ``mx.eval(indices)`` / ``f16_yield()`` at each
     layer's MoE, so the two-hand-off interleave really runs with real arithmetic.
  2. A mock-runtime schedule with a STREAM model (submission-sequence numbers): a close
     may be released only by a flush whose covering barrier sequence exceeds every wave
     it guards.  Asserts no violation, no double-held lock, the trailer reaches the last
     layer (orphan hand-off), and a NEGATIVE CONTROL: with the driver swap defeated the
     checker fails (guards against a vacuous test).
  3. Abort at each hand-off kind, either group: partner unwound, every close released or
     handed back to the runtime list, no lock leaked, original error re-raised.
  4. Derivation: barrier + stamped run round-trip, unique anchors, no ``mx.`` in inserts,
     double-apply refused; a non-group ``barrier()`` issues no ``async_eval`` (spy).
  5. reads mode unchanged (same digest as barrier + the sequential oracle).
  6. Stamps: sink + readout smoke; stamped hand-off helpers record without perturbing
     arithmetic.
"""
from __future__ import annotations

import inspect
import os
import sys
import textwrap
import threading

import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)  # BEFORE any MLX work; MLX defaults to Metal otherwise

from pathlib import Path  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts" / "deepseek_v41"
_PACKED = _REPO / "docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed"
# .f16-site holds greenlet (private target dir, NOT the shared venv).  Honour F16SITE
# (default = the F16 pipeline worktree's private dir); the new worktree has no .f16-site.
_F16_SITE = os.environ.get(
    "F16SITE",
    "/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f16-pipeline/.f16-site",
)
for _p in (str(_SCRIPTS), str(_PACKED), str(_F16_SITE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import greenlet  # noqa: E402  (from .f16-site)

from mtplx.models.deepseek_v41 import Model, ModelArgs  # noqa: E402
from mtplx.models.deepseek_v41_cache import make_cache  # noqa: E402
from mtplx.engram_v41 import NgramHashState, n_hash_cols  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from f16 import pipeline as pl  # noqa: E402
from f16 import stamps as st  # noqa: E402
from f16 import stamp_readout as sr  # noqa: E402
from f16.pipeline import (  # noqa: E402
    LEADER,
    NONGROUP,
    TRAILER,
    Pipeline,
    _PipelineAborted,
    f16_yield,
    f16_yield_stamped,
)

_ENG_LAYERS = (1, 3)
_DSPARK_LAYERS = (2, 4)


# ---------------------------------------------------------------------------
# fixtures (mirror tests/test_dsv41_f16_pipeline.py)
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


def _built_model(seed=1, use_engram=False, scale=0.1):
    args = _args()
    model = Model(args)
    _randomize(model, seed=seed, scale=scale)
    if use_engram:
        _attach_fake_engram(model)
    return args, model


# ---------------------------------------------------------------------------
# test-only switch hook: fire the two hand-offs at each layer's MoE so the barrier
# interleave really runs with real arithmetic (the CPU model has no streamed switch).
# barrier()/f16_yield() are no-ops outside a group greenlet, so the hook is inert in the
# sequential oracle and only interleaves inside the pipeline's group greenlets.
# ---------------------------------------------------------------------------
class _HookedSwitch:
    def __init__(self, inner, pipe):
        self.inner = inner
        self.pipe = pipe

    def __call__(self, xf, indices):
        self.pipe.barrier(indices)   # barrier hand-off (submit early + park the partner)
        mx.eval(indices)             # retained blocking routing eval
        out = self.inner(xf, indices)
        f16_yield()                  # reads hand-off
        return out


class _StampedHookedSwitch:
    """Mirrors the stamped derived run's in-run stamp sequence + stamped hand-offs, so
    the stamped helpers are exercised with real arithmetic on CPU."""

    def __init__(self, inner, pipe, layer):
        self.inner = inner
        self.pipe = pipe
        self.layer = int(layer)

    def __call__(self, xf, indices):
        g = greenlet.getcurrent()
        rec = st.begin(getattr(g, "_f16_role", NONGROUP), self.layer, int(indices.shape[0]))
        g._f16_rec = rec                          # stamp 0
        self.pipe._barrier_stamped(indices)       # barrier + stamps 1,2
        st.stamp(rec, 3)
        st.annotate(rec, n_parts=1, n_hits=0, n_unique=int(indices.size))
        st.stamp(rec, 4)
        st.stamp(rec, 5)
        out = self.inner(xf, indices)
        f16_yield_stamped()                        # reads hand-off + stamp 6
        st.stamp(rec, 7)
        st.stamp(rec, 8)
        return out


def _install_hook(model, pipe, cls=_HookedSwitch):
    saved = []
    for layer in model.model.layers:
        sm = layer.mlp.switch_mlp
        saved.append((layer, sm))
        layer.mlp.switch_mlp = (cls(sm, pipe, layer.layer_id) if cls is _StampedHookedSwitch
                                else cls(sm, pipe))
    return saved


def _remove_hook(saved):
    for layer, sm in saved:
        layer.mlp.switch_mlp = sm


def _n_layers():
    return 8


# ---------------------------------------------------------------------------
# 1. barrier-mode bitwise parity vs the sequential oracle (hand-offs really fire)
# ---------------------------------------------------------------------------
def _lead_rows(rows, split):
    return 4 if split == "fixed4" else (rows + 1) // 2


@pytest.mark.parametrize("rows", [5, 6, 7, 8])
@pytest.mark.parametrize("use_engram", [False, True])
@pytest.mark.parametrize("split", ["fixed4", "balanced"])
@pytest.mark.parametrize("seed,scale", [(2, 0.1), (13, 0.3)])
def test_barrier_parity_is_bitwise_sequential(rows, use_engram, split, seed, scale):
    args = _args()
    model = Model(args)
    _randomize(model, seed=seed, scale=scale)
    if use_engram:
        _attach_fake_engram(model)
    forward = _forward(model)
    ids = mx.array(np.random.RandomState(rows + seed).randint(0, args.vocab_size, size=(1, rows)))
    lead = _lead_rows(rows, split)

    # oracle: two sequential causal forwards of exactly the group sizes (no hook).
    c2 = _make_cache(args, model, use_engram)
    la, ma = forward(ids[:, :lead], c2)
    lb, mb = forward(ids[:, lead:], c2)
    mx.eval(la, ma, lb, mb)
    oracle_logits = mx.concatenate([la, lb], axis=1)
    oracle_mh = mx.concatenate([ma, mb], axis=1)
    mx.eval(oracle_logits, oracle_mh)
    snap2, off2 = _cache_snapshot(c2)
    eng2 = np.array(c2.engram_state.state) if use_engram else None

    # pipeline: barrier mode, hand-offs fired by the test hook (real interleave).
    c1 = _make_cache(args, model, use_engram)
    pipe = Pipeline(model, armed=True, split=split, handoff="barrier")
    saved = _install_hook(model, pipe)
    try:
        logits, mh = pipe.pipelined_forward(forward, ids, c1)
        mx.eval(logits, mh)
    finally:
        _remove_hook(saved)
    snap1, off1 = _cache_snapshot(c1)
    eng1 = np.array(c1.engram_state.state) if use_engram else None

    assert pipe.counters["pipelined_forwards"] == 1
    # The two-hand-off interleave really ran (not the degenerate sequential fallthrough).
    assert pipe.counters["handoffs"] > 2 * _n_layers(), pipe.counters
    assert bool(mx.array_equal(logits, oracle_logits).item()), "logits differ"
    assert bool(mx.array_equal(mh, oracle_mh).item()), "main_hidden differs"
    assert off1 == off2 == rows
    assert set(snap1) == set(snap2)
    for k, v in snap2.items():
        assert np.array_equal(snap1[k], v), f"cache[{k}] differs"
    if use_engram:
        assert np.array_equal(eng1, eng2), "engram state differs"


def test_barrier_trim_then_next_is_bitwise():
    args, model = _built_model(seed=7, use_engram=True)
    forward = _forward(model)
    rows, kept = 8, 4
    ids = mx.array(np.random.RandomState(2).randint(0, args.vocab_size, size=(1, rows)))
    nxt = mx.array(np.random.RandomState(3).randint(0, args.vocab_size, size=(1, 1)))

    c2 = _make_cache(args, model, True)
    forward(ids[:, :4], c2)
    forward(ids[:, 4:], c2)
    c2.trim(rows - kept)
    nl2, _ = forward(nxt, c2)
    mx.eval(nl2)
    snap2, off2 = _cache_snapshot(c2)

    c1 = _make_cache(args, model, True)
    pipe = Pipeline(model, armed=True, handoff="barrier")
    saved = _install_hook(model, pipe)
    try:
        pipe.pipelined_forward(forward, ids, c1)
    finally:
        _remove_hook(saved)
    c1.trim(rows - kept)
    nl1, _ = forward(nxt, c1)  # next forward runs with the real (unhooked) switch
    mx.eval(nl1)
    snap1, off1 = _cache_snapshot(c1)

    assert off1 == off2 == kept + 1
    assert bool(mx.array_equal(nl1, nl2).item()), "post-trim next-forward logits differ"
    for k, v in snap2.items():
        assert np.array_equal(snap1[k], v), f"post-trim cache[{k}] differs"


def test_barrier_multi_cycle_is_bitwise_sequential():
    args, model = _built_model(seed=3, use_engram=True)
    forward = _forward(model)
    pipe = Pipeline(model, armed=True, handoff="barrier")
    cycles = [(6, 3), (8, 2), (7, 5), (5, 4)]

    c_seq = _make_cache(args, model, True)
    c_pipe = _make_cache(args, model, True)
    rng = np.random.RandomState(42)
    for i, (rows, kept) in enumerate(cycles):
        ids = mx.array(rng.randint(0, args.vocab_size, size=(1, rows)))
        ls_a, _ = forward(ids[:, :4], c_seq)
        ls_b, _ = forward(ids[:, 4:], c_seq)
        seq_logits = mx.concatenate([ls_a, ls_b], axis=1)
        saved = _install_hook(model, pipe)
        try:
            pipe_logits, _ = pipe.pipelined_forward(forward, ids, c_pipe)
        finally:
            _remove_hook(saved)
        mx.eval(seq_logits, pipe_logits)
        assert bool(mx.array_equal(pipe_logits, seq_logits).item()), f"cycle {i} differs"
        c_seq.trim(rows - kept)
        c_pipe.trim(rows - kept)
        s_seq, off_seq = _cache_snapshot(c_seq)
        s_pipe, off_pipe = _cache_snapshot(c_pipe)
        assert off_seq == off_pipe
        for k, v in s_seq.items():
            assert np.array_equal(s_pipe[k], v), f"cycle {i} cache[{k}] differs"
    assert pipe.counters["pipelined_forwards"] == len(cycles)


def test_barrier_no_threads_and_on_calling_thread():
    args, model = _built_model(seed=1, use_engram=True)
    forward = _forward(model)
    pipe = Pipeline(model, armed=True, handoff="barrier")
    ids = mx.array(np.random.RandomState(1).randint(0, args.vocab_size, size=(1, 6)))
    calling = threading.get_ident()
    seen = set()
    orig = pipe._group_forward

    def spy(*a, **k):
        seen.add(threading.get_ident())
        return orig(*a, **k)

    pipe._group_forward = spy
    saved = _install_hook(model, pipe)
    before = threading.active_count()
    try:
        logits, _ = pipe.pipelined_forward(forward, ids, _make_cache(args, model, True))
        mx.eval(logits)
    finally:
        _remove_hook(saved)
    assert threading.active_count() == before, "barrier forward spawned a thread"
    assert seen == {calling}, "a group ran off the calling thread"


# ---------------------------------------------------------------------------
# 2. mock-runtime schedule with a STREAM model + negative control
# ---------------------------------------------------------------------------
class _ScheduleViolation(Exception):
    pass


class _Lock:
    def __init__(self, idx):
        self.idx = idx
        self.holder = None

    def acquire(self, role):
        if self.holder is not None:
            raise _ScheduleViolation(
                f"layer {self.idx} lock held by {self.holder} on acquire by {role}"
            )
        self.holder = role

    def release(self):
        self.holder = None


class _Close:
    """Stand-in for a deferred split close; guards the waves submitted during one run."""

    def __init__(self, layer, role, lock):
        self.layer = layer
        self.role = role
        self.lock = lock
        self.guard_seq = -1  # max submission sequence of the waves this close guards
        self.released = False

    def release(self, *, synchronize=False):  # duck-types the release surface
        del synchronize
        self.lock.release()
        self.released = True


class _StreamRuntime:
    """Runtime stand-in with a single in-order submission stream (a monotonic sequence),
    per-layer locks, and a deferred-close list the driver swaps.  ``defeat_swap`` makes
    ``_deferred_slot_releases`` a shared list the swap cannot separate (the negative
    control)."""

    def __init__(self, n_layers, *, defeat_swap=False):
        self.locks = [_Lock(i) for i in range(n_layers)]
        self.clock = 0
        self.last_barrier_seq = {LEADER: -1, TRAILER: -1}
        self.violations = []
        self.entered = []       # (role, layer) in schedule order
        self.all_closes = []    # every close created
        self.aborted_at = {}    # role -> 'barrier' | 'yield'
        self._defeat = defeat_swap
        self._shared = []       # used only when defeat_swap
        self._own = None        # backing store for the per-group swap

    @property
    def _deferred_slot_releases(self):
        return self._shared if self._defeat else self._own

    @_deferred_slot_releases.setter
    def _deferred_slot_releases(self, value):
        if self._defeat:
            return  # swap cannot separate the lists
        self._own = value

    # -- stream events -------------------------------------------------------
    def on_barrier(self, role):
        self.clock += 1
        self.last_barrier_seq[role] = self.clock  # mx.eval(indices) later covers <= this

    def on_wave(self, close):
        self.clock += 1
        close.guard_seq = self.clock

    def begin(self, layer, role):
        self.locks[layer].acquire(role)
        self.entered.append((role, layer))
        close = _Close(layer, role, self.locks[layer])
        self.all_closes.append(close)
        return close

    def defer(self, close):
        lst = self._deferred_slot_releases
        if lst is None:
            lst = []
            self._deferred_slot_releases = lst
        lst.append(close)

    def flush_check(self, role):
        """evaluate=False flush: release each close, flagging a violation if any of its
        guarded waves was submitted after this group's covering barrier."""
        lst = self._deferred_slot_releases
        if not lst:
            return
        frontier = self.last_barrier_seq[role]
        while lst:
            close = lst.pop(0)
            if close.guard_seq > frontier:
                self.violations.append(
                    (role, close.role, close.layer, close.guard_seq, frontier)
                )
            close.release()

    def boundary_flush(self):
        """evaluate=True boundary flush (no safety check): release whatever remains."""
        lst = self._deferred_slot_releases
        if lst:
            while lst:
                lst.pop(0).release()


def _mock_group(rt, pipe, role, n_layers, fail_at=None):
    def run():
        for layer in range(n_layers):
            idx = mx.array([layer], mx.int32)
            rt.on_barrier(role)
            try:
                pipe.barrier(idx)          # adopt orphans + async_eval + barrier hand-off
            except _PipelineAborted:
                rt.aborted_at[role] = "barrier"  # no lock held here (begin not reached)
                raise
            rt.flush_check(role)           # release prior closes on the installed list
            cur = rt.begin(layer, role)    # acquire this layer's lock (raises on violation)
            rt.on_wave(cur)                # hit waves
            if fail_at is not None and layer == fail_at:
                cur.release()              # mimic the run's except: pending.abort + close
                raise RuntimeError(f"mock fail role={role} layer={layer}")
            try:
                f16_yield()                # reads hand-off
            except _PipelineAborted:
                rt.aborted_at[role] = "yield"
                cur.release()              # mimic the run's except: pending.close
                raise
            rt.on_wave(cur)                # miss waves
            rt.defer(cur)                  # defer this layer's close
        return ("done", role)

    return run


def _pipe_with_runtime(runtime, handoff="barrier"):
    _args_, model = _built_model(seed=1)
    pipe = Pipeline(model, armed=True, handoff=handoff)
    model._mtplx_expert_runtime = runtime
    return pipe


def test_barrier_stream_schedule_is_safe_and_interleaves():
    n = 12
    rt = _StreamRuntime(n)
    pipe = _pipe_with_runtime(rt)
    results = pipe._run_groups_barrier(
        _mock_group(rt, pipe, LEADER, n), _mock_group(rt, pipe, TRAILER, n)
    )
    rt.boundary_flush()  # retained boundary flush releases whatever was handed back

    assert results == [("done", LEADER), ("done", TRAILER)]
    assert rt.violations == [], f"stream-safety violations: {rt.violations}"
    assert all(lk.holder is None for lk in rt.locks), "a layer lock was left held"
    assert all(c.released for c in rt.all_closes), "a close was neither released nor handed back"
    # both groups ran every layer, interleaved (the trailer reached the last layer)
    assert sum(1 for r, _ in rt.entered if r == LEADER) == n
    assert sum(1 for r, _ in rt.entered if r == TRAILER) == n
    assert (TRAILER, n - 1) in rt.entered, "trailer never reached the last layer"
    assert pipe.counters["handoffs"] > 2 * n, "driver did not interleave two hand-offs/layer"
    # KV causality: when B begins layer L, A has already run layer L.
    leader_seen = 0
    for role, layer in rt.entered:
        if role == LEADER:
            leader_seen += 1
        else:
            assert leader_seen >= layer + 1, (
                f"trailer entered layer {layer} with only {leader_seen} leader layers done"
            )


def test_negative_control_shared_list_fails_the_stream_checker():
    """With the driver swap defeated (a single shared deferred list, as reads mode uses)
    the barrier-mode flush releases a partner's close whose waves were submitted after
    this group's barrier -> the stream checker MUST flag it (guards against a vacuous
    positive test)."""
    n = 12
    rt = _StreamRuntime(n, defeat_swap=True)
    pipe = _pipe_with_runtime(rt)
    # A double-held lock (also a real corruption) may surface first; either way the
    # shared list is caught.  Accept the schedule violation OR recorded stream violations.
    try:
        pipe._run_groups_barrier(
            _mock_group(rt, pipe, LEADER, n), _mock_group(rt, pipe, TRAILER, n)
        )
        raced = False
    except _ScheduleViolation:
        raced = True
    assert raced or rt.violations, (
        "defeated swap produced neither a stream-safety violation nor a lock violation"
    )


def test_mock_stream_checker_detects_a_manual_violation():
    """The stream checker is not vacuous: releasing a close whose guarded wave postdates
    the covering barrier is flagged."""
    rt = _StreamRuntime(2)
    rt.on_barrier(LEADER)                  # barrier seq = frontier for the leader
    close = rt.begin(0, LEADER)
    close.guard_seq = rt.last_barrier_seq[LEADER] + 5   # wave submitted AFTER the barrier
    rt.defer(close)
    rt.flush_check(LEADER)
    assert rt.violations, "checker missed a too-late wave"


# ---------------------------------------------------------------------------
# 3. abort at each hand-off kind, either group
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fail_role", [LEADER, TRAILER])
@pytest.mark.parametrize("fail_layer", [1, 3, 5])
def test_abort_unwinds_partner_and_restores_all_closes(fail_role, fail_layer):
    n = 8
    rt = _StreamRuntime(n)
    pipe = _pipe_with_runtime(rt)
    groups = {
        LEADER: _mock_group(rt, pipe, LEADER, n, fail_at=fail_layer if fail_role == LEADER else None),
        TRAILER: _mock_group(rt, pipe, TRAILER, n, fail_at=fail_layer if fail_role == TRAILER else None),
    }
    with pytest.raises(RuntimeError, match=f"mock fail role={fail_role} layer={fail_layer}"):
        pipe._run_groups_barrier(groups[LEADER], groups[TRAILER])

    # rule 3: everything was handed back to the runtime list, nothing dropped; the
    # retained boundary flush then releases every remaining close.
    rt.boundary_flush()
    assert all(lk.holder is None for lk in rt.locks), "a layer lock was leaked on abort"
    assert all(c.released for c in rt.all_closes), "a close was dropped on abort"
    assert rt.violations == [], f"unsafe release during abort: {rt.violations}"
    # the partner was unwound at a hand-off (barrier or yield), not left suspended
    other = TRAILER if fail_role == LEADER else LEADER
    assert other in rt.aborted_at, "partner was not unwound"


def test_abort_exercises_both_handoff_kinds():
    """Across the failing configurations both partner suspend points (barrier and yield)
    are exercised -- the _PipelineAborted delivery path is tested for each hand-off."""
    kinds = set()
    for fail_role in (LEADER, TRAILER):
        for fail_layer in range(1, 7):
            n = 8
            rt = _StreamRuntime(n)
            pipe = _pipe_with_runtime(rt)
            g = {
                LEADER: _mock_group(rt, pipe, LEADER, n,
                                    fail_at=fail_layer if fail_role == LEADER else None),
                TRAILER: _mock_group(rt, pipe, TRAILER, n,
                                     fail_at=fail_layer if fail_role == TRAILER else None),
            }
            with pytest.raises(RuntimeError):
                pipe._run_groups_barrier(g[LEADER], g[TRAILER])
            rt.boundary_flush()
            other = TRAILER if fail_role == LEADER else LEADER
            if other in rt.aborted_at:
                kinds.add(rt.aborted_at[other])
    assert kinds == {"barrier", "yield"}, f"only exercised {kinds}"


# ---------------------------------------------------------------------------
# 4. derivation: barrier + stamped round-trip, unique anchors, non-group spy
# ---------------------------------------------------------------------------
def _sched_source():
    import plane_lane
    import projection_install

    base = textwrap.dedent(inspect.getsource(plane_lane.PackedDecode.run))
    return projection_install.scheduled_run_source(base)


def test_barrier_run_source_inserts_both_handoffs_with_roundtrip():
    sched = _sched_source()
    derived = pl.f16_barrier_run_source(sched)
    assert derived.count("self._f16_barrier(indices)") == 1
    assert derived.count("self._f16_yield()") == 1
    lines = derived.splitlines()
    bi = next(i for i, ln in enumerate(lines) if ln.strip() == "self._f16_barrier(indices)")
    assert lines[bi + 1].strip() == "mx.eval(indices)", "barrier not immediately before mx.eval"
    yi = next(i for i, ln in enumerate(lines) if ln.strip() == "self._f16_yield()")
    assert lines[yi + 1].strip() == "ready_iter = pending.iter_ready_misses()"
    # round-trip: removing both inserted lines recovers the scheduled source.
    recovered = [ln for ln in lines
                 if ln.strip() not in ("self._f16_barrier(indices)", "self._f16_yield()")]
    assert recovered == sched.splitlines()
    # double-apply refused (the barrier line is already present).
    with pytest.raises(RuntimeError):
        pl.f16_barrier_run_source(derived)


def test_stamped_run_source_roundtrips_and_is_mx_free():
    sched = _sched_source()
    for barrier in (False, True):
        base = pl.f16_barrier_run_source(sched) if barrier else pl.f16_run_source(sched)
        stamped = pl.f16_stamp_run_source(base)
        stamp_lines = ("self._f16_stamp0(tokens)", "self._f16_stamp3()",
                       "self._f16_stamproute(experts, parts, pending)",
                       "self._f16_stamp5()", "self._f16_stamp7()", "self._f16_stamp8()")
        for sl in stamp_lines:
            assert stamped.count(sl) == 1, f"{sl} not inserted exactly once (barrier={barrier})"
            assert "mx." not in sl
        recovered = [ln for ln in stamped.splitlines() if ln.strip() not in stamp_lines]
        assert recovered == base.splitlines(), "stamp inserts changed more than their lines"
        # double-apply refused.
        with pytest.raises(RuntimeError):
            pl.f16_stamp_run_source(stamped)


def test_barrier_and_stamped_runs_compile_distinctly():
    reads_fn, _ = pl.build_yield_run()
    barrier_fn, bshas = pl.build_barrier_run()
    assert callable(barrier_fn)
    assert barrier_fn.__code__.co_code != reads_fn.__code__.co_code
    assert barrier_fn.__code__.co_code != pl.scheduled_run_cocode()
    assert bshas["f16_barrier_run_sha256"] != bshas["scheduled_run_sha256"]
    for barrier in (False, True):
        sfn, sshas = pl.build_stamped_run(barrier=barrier)
        assert callable(sfn)
        key = "f16_stamped_barrier_run_sha256" if barrier else "f16_stamped_reads_run_sha256"
        assert key in sshas and sshas[key] != sshas["scheduled_run_sha256"]


def test_nongroup_barrier_issues_no_async_eval(monkeypatch):
    """A non-group caller's barrier must issue NO async_eval (the retained blocking
    mx.eval follows); a group caller's barrier MUST issue one."""
    _args_, model = _built_model(seed=1)
    pipe = Pipeline(model, armed=True, handoff="barrier")
    calls = {"n": 0}
    real_async = pl.mx.async_eval

    def spy(*a, **k):
        calls["n"] += 1
        return real_async(*a, **k)

    monkeypatch.setattr(pl.mx, "async_eval", spy)

    # main (non-group) greenlet: barrier is a no-op, NO async_eval.
    pipe.barrier(mx.array([0], mx.int32))
    assert calls["n"] == 0, "non-group barrier issued an async_eval"

    # a group greenlet: barrier submits exactly one async_eval, then hands off.
    def grp():
        pipe.barrier(mx.array([0], mx.int32))
        return "ok"

    g = greenlet.greenlet(grp)
    g._f16_group = True
    g._f16_skip = 0
    g._f16_role = LEADER
    g.switch()          # runs to the barrier hand-off (parks in g.parent.switch)
    assert calls["n"] == 1, "group barrier did not issue exactly one async_eval"
    g.switch()          # resume to completion


# ---------------------------------------------------------------------------
# 5. reads mode unchanged (same digest as barrier + the sequential oracle)
# ---------------------------------------------------------------------------
def test_reads_mode_construction_is_unchanged():
    _args_, model = _built_model(seed=1)
    reads = Pipeline(model, armed=True)                       # default handoff
    assert reads.handoff == "reads" and reads._skip == 1
    # reads keeps the class driver (no per-group-list driver installed).
    assert "_run_groups" not in reads.__dict__
    barrier = Pipeline(model, armed=True, handoff="barrier")
    assert barrier.handoff == "barrier" and barrier._skip == 2
    assert barrier.__dict__.get("_run_groups") is not None   # barrier driver bound
    with pytest.raises(RuntimeError):
        Pipeline(model, armed=True, handoff="two-hands")      # unknown mode refused


@pytest.mark.parametrize("rows,split", [(6, "fixed4"), (7, "balanced"), (8, "fixed4")])
def test_reads_and_barrier_agree_with_oracle(rows, split):
    """reads-mode output, barrier-mode output (hand-offs fired), and the sequential
    oracle are all bit-identical -- barrier changes only the schedule, not arithmetic."""
    args, model = _built_model(seed=5, use_engram=True, scale=0.2)
    forward = _forward(model)
    ids = mx.array(np.random.RandomState(rows).randint(0, args.vocab_size, size=(1, rows)))
    lead = _lead_rows(rows, split)

    c_or = _make_cache(args, model, True)
    la, _ = forward(ids[:, :lead], c_or)
    lb, _ = forward(ids[:, lead:], c_or)
    oracle = mx.concatenate([la, lb], axis=1)

    c_reads = _make_cache(args, model, True)
    reads = Pipeline(model, armed=True, split=split, handoff="reads")
    reads_logits, _ = reads.pipelined_forward(forward, ids, c_reads)

    c_bar = _make_cache(args, model, True)
    bar = Pipeline(model, armed=True, split=split, handoff="barrier")
    saved = _install_hook(model, bar)
    try:
        bar_logits, _ = bar.pipelined_forward(forward, ids, c_bar)
    finally:
        _remove_hook(saved)

    mx.eval(oracle, reads_logits, bar_logits)
    assert bool(mx.array_equal(reads_logits, oracle).item()), "reads != oracle"
    assert bool(mx.array_equal(bar_logits, oracle).item()), "barrier != oracle"
    assert bool(mx.array_equal(bar_logits, reads_logits).item()), "barrier != reads"


# ---------------------------------------------------------------------------
# 6. stamps: sink + readout smoke; stamped hand-off helpers record + preserve parity
# ---------------------------------------------------------------------------
def test_stamps_sink_and_readout_smoke(tmp_path, capsys):
    st.configure(str(tmp_path / "probe"))
    # synthesize two forwards' worth of records for both roles.
    for fwd in range(2):
        for role in (LEADER, TRAILER):
            for layer in range(4):
                rec = st.begin(role, layer, rows=6)
                for sid in range(1, st.N_STAMPS):
                    st.stamp(rec, sid)
                st.annotate(rec, n_parts=2, n_hits=1, n_unique=5)
    raw = st.write()
    assert raw is not None and Path(raw).exists()
    assert (tmp_path / "probe.summary.json").exists()
    rc = sr.main([str(tmp_path / "probe")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "F16_STAMP_READOUT" in out and "encode" in out and "gpu_wait" in out
    st.configure(str(tmp_path / "unused"))  # reset the global sink for other tests


def test_stamped_handoffs_record_and_preserve_parity(tmp_path):
    args, model = _built_model(seed=9, use_engram=True, scale=0.2)
    forward = _forward(model)
    ids = mx.array(np.random.RandomState(6).randint(0, args.vocab_size, size=(1, 6)))

    c_or = _make_cache(args, model, True)
    la, _ = forward(ids[:, :4], c_or)
    lb, _ = forward(ids[:, 4:], c_or)
    oracle = mx.concatenate([la, lb], axis=1)

    st.configure(str(tmp_path / "run"))
    c1 = _make_cache(args, model, True)
    pipe = Pipeline(model, armed=True, handoff="barrier")
    saved = _install_hook(model, pipe, cls=_StampedHookedSwitch)
    try:
        logits, _ = pipe.pipelined_forward(forward, ids, c1)
        mx.eval(logits, oracle)
    finally:
        _remove_hook(saved)

    assert bool(mx.array_equal(logits, oracle).item()), "stamped hand-offs perturbed arithmetic"
    recs = st.sink().records
    assert recs, "no stamp records captured"
    # the barrier stamps (1,2) and reads-resume stamp (6) fired on group records.
    group_recs = [r for r in recs if r["role"] in (LEADER, TRAILER)]
    assert group_recs
    assert any(r["s"][1] is not None and r["s"][2] is not None for r in group_recs), "no barrier stamps"
    assert any(r["s"][6] is not None for r in group_recs), "no reads-resume stamp"
    st.configure(str(tmp_path / "unused"))  # reset the global sink
