"""F16 verify-row-group pipeline: interleave two causal MTP-verify row groups so
one group's SSD miss-read wait overlaps the other group's routing barrier + submit.

The lever is admissible because splitting the M<=8 verify rows into group A (the
first 4 rows) and group B (the rest), with B attending to A's KV, is ALREADY a
supported arithmetic -- the retained decode loop's ``verify_chunks``.  Fable ran it
sequentially as (4,4): output digest
172830a96d84dbdac631c058fe0dfaf956df15b6c353f41fc05f604bc92c6393.  This module's
``pipelined_forward`` must be BIT-IDENTICAL to running ``forward(A)`` then
``forward(B)`` sequentially (same per-group row counts => same kernels), so a GPU
run reproduces that digest exactly.

How it interleaves WITHOUT restructuring the model:
  * Each group's forward runs in its own Python thread under a STRICT BATON: exactly
    one thread runs at any instant, the other blocked on the baton.  The baton is
    handed over at ONE yield point inside the expert switch -- ``self._f16_yield()``,
    inserted (by :func:`f16_run_source`) into the retained SCHEDULED
    ``PackedDecode.run`` immediately before ``ready_iter =
    pending.iter_ready_misses()`` (after the demand reads + hit gate_up + shared
    expert are submitted, before the completion loop blocks on the SSD reads).
  * ``_f16_yield`` is a no-op unless the current thread is a pipeline group thread
    (it reads a thread-local set only by :meth:`Pipeline.pipelined_forward`).
  * Baton order (leader A one layer ahead of trailer B): A runs [layer 0 to yield]
    then does NOT hand over at that first yield, continues [finish 0, layer 1 to
    yield], then alternates at every yield; B waits to start, then hands over at
    every yield.  Because ``begin_split_route(L)`` takes the per-layer lock and it is
    released only by the NEXT run's ``flush_deferred_slot_releases`` (expert_runtime
    begin_split_route:4149 acquire / _DeferredSplitClose->close:1552 release, flushed
    at plane_lane run head), B may enter layer L only after A has entered layer L+1 --
    so A has always written layer L's KV (via ``layer(...)``; LayerAttentionCache.
    advance:1479-1484 only moves the offset, the layer forward grows the store)
    before B reaches layer L.  Causality holds; ``cache.advance`` runs once, after
    both groups, in the driver.

Faithfulness of the per-group forward: it is a hand clone of
``DeepseekV41Backbone._forward_span`` + the ``Model.__call__`` head/logits/main_hidden
tail, with exactly three differences (positions from a shared ``offset0``; engram
advanced up front for A then B and read per group through ``_ChunkEngramView`` as
``_forward_layer_major`` does; no per-group ``cache.advance``).  The device-route
recovery is asserted OFF (never cloned).  The source of every function this clone
mirrors is SHA-pinned below; :func:`verify_source_pins` fails loudly on drift, and
the CPU test diffs the clone's per-line arithmetic against the live source.

MLX-free at import except ``mlx.core``; CPU-testable.
"""
from __future__ import annotations

import contextvars
import hashlib
import inspect
import textwrap
import threading
import time
from types import MethodType
from typing import Any, Callable, Optional

import mlx.core as mx

# --- SHA pins: the retained/pinned sources this lever derives from or clones. ----
# A drift in any of these breaks an assumption; verify_source_pins / preflight fail.
RETAINED_PLANE_LANE_SHA256 = (
    "1acad9e24c37e5c618b2d8e2e98fb93eb94b5476d5c6de6fa0ee054db468ba54"
)
FORWARD_SPAN_SHA256 = (
    "c5750a4d25a216b562aeeef5d495dd071645704cc3d1ea735c7a7ff485b3f432"
)
MODEL_CALL_SHA256 = (
    "677f0d5025e3fd408e85ff2b37340ef0fee5372876ae4bd889ae43261aa2f2b6"
)
CHUNK_ENGRAM_VIEW_SHA256 = (
    "38567d1f60a766be987769cf3dc840d49e1123d945e743692d4c0945a9618b37"
)
FORWARD_LAYER_MAJOR_SHA256 = (
    "b0cc782462dbc8fb2cf5849f4f7f50d2697a2dc4457e9fca3a2b6685c7cac66e"
)
DEVICE_ROUTE_ACTIVE_SHA256 = (
    "1c8788ac30e9899d0f6799d0f508fc02e1195a0884b636debb36925fb9e31e8a"
)

# The verify row split (design): group A is the first 4 rows, group B is the rest.
A_ROWS = 4
# Default baton / join timeouts (seconds).
BATON_TIMEOUT_S = 120.0

# The one anchor the yield is inserted before, and the inserted line.  The insert
# issues no ``mx`` op (a pure host baton hand-off), so the lane's MLX op sequence is
# unchanged -- the per-group forward is byte-identical to the unyielded scheduled run.
_YIELD_ANCHOR = "ready_iter = pending.iter_ready_misses()"
_YIELD_INSERT = "self._f16_yield()"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _dedented_source(obj) -> str:
    return textwrap.dedent(inspect.getsource(obj))


# ---------------------------------------------------------------------------
# Source pins
# ---------------------------------------------------------------------------
def verify_source_pins() -> dict:
    """Assert the live pinned-runtime sources this lever clones/derives from match
    the SHAs pinned above.  Raises on the FIRST mismatch so a GPU window cannot run
    a lever built against a drifted backbone.  CPU-safe (pure text)."""
    from mtplx.models import deepseek_v41 as dv

    checks = {
        "_forward_span": (dv.DeepseekV41Backbone._forward_span, FORWARD_SPAN_SHA256),
        "Model.__call__": (dv.Model.__call__, MODEL_CALL_SHA256),
        "_ChunkEngramView": (dv._ChunkEngramView, CHUNK_ENGRAM_VIEW_SHA256),
        "_forward_layer_major": (
            dv.DeepseekV41Backbone._forward_layer_major,
            FORWARD_LAYER_MAJOR_SHA256,
        ),
        "_device_route_active": (
            dv.DeepseekV41Backbone._device_route_active,
            DEVICE_ROUTE_ACTIVE_SHA256,
        ),
    }
    report = {}
    for name, (fn, pin) in checks.items():
        got = _sha(_dedented_source(fn))
        if got != pin:
            raise RuntimeError(
                f"F16: pinned-runtime {name} drifted from the cloned source "
                f"(sha {got} != pinned {pin}); re-derive the F16 clone before a GPU run"
            )
        report[name] = got
    return report


# ---------------------------------------------------------------------------
# Yield-capable run derivation (from the retained SCHEDULED PackedDecode.run)
# ---------------------------------------------------------------------------
def f16_run_source(scheduled_source: str) -> str:
    """Insert ``self._f16_yield()`` immediately before the miss-completion iterator,
    asserting a unique anchor and a byte-for-byte round trip (removing the inserted
    line recovers ``scheduled_source``).  The insert adds no ``mx`` op."""
    if "mx." in _YIELD_INSERT:
        raise RuntimeError("F16 yield insert would add an mx op")
    if _YIELD_INSERT in scheduled_source:
        raise RuntimeError("F16 yield already inserted in this run source")
    orig = scheduled_source.splitlines()
    hits = [i for i, ln in enumerate(orig) if ln.strip() == _YIELD_ANCHOR]
    if len(hits) != 1:
        raise RuntimeError(
            f"F16 yield anchor is not unique ({len(hits)}x): {_YIELD_ANCHOR!r} -- "
            "the retained scheduled run changed; re-pin the F16 yield"
        )
    i = hits[0]
    indent = orig[i][: len(orig[i]) - len(orig[i].lstrip())]
    lines = list(orig)
    lines.insert(i, indent + _YIELD_INSERT)
    recovered = [ln for ln in lines if ln.strip() != _YIELD_INSERT]
    if recovered != orig:
        raise RuntimeError("F16 yield insertion changed the scheduled run beyond one line")
    return "\n".join(lines)


def _scheduled_run_source() -> str:
    """The retained scheduled run source (``self.issue_next()`` variant that
    projection_install installs), derived from the pinned plane_lane at run time."""
    import plane_lane
    import projection_install

    lane_src = inspect.getsource(plane_lane)
    if _sha(lane_src) != RETAINED_PLANE_LANE_SHA256:
        raise RuntimeError("plane_lane.py differs from the pinned retained lane")
    base = _dedented_source(plane_lane.PackedDecode.run)
    return projection_install.scheduled_run_source(base)


def build_yield_run():
    """Compile the yield-capable run in plane_lane's namespace (so it closes over the
    identical helpers) from the retained scheduled source.  Returns (run_fn, shas)."""
    import plane_lane

    scheduled = _scheduled_run_source()
    yielded = f16_run_source(scheduled)
    namespace = dict(plane_lane.__dict__)
    exec(compile(yielded, "<f16_yield_run>", "exec"), namespace)  # noqa: S102
    return namespace["run"], {
        "scheduled_run_sha256": _sha(scheduled),
        "f16_yield_run_sha256": _sha(yielded),
        "retained_plane_lane_sha256": RETAINED_PLANE_LANE_SHA256,
    }


def scheduled_run_cocode() -> bytes:
    """Bytecode of the retained scheduled run, for the install-time refuse check
    (F16 stage 1 is mutually exclusive with F2b and the F5 stamp probe: the enabled
    hot path must be the retained scheduled run, not a wrapped/stamped variant)."""
    import plane_lane

    scheduled = _scheduled_run_source()
    namespace = dict(plane_lane.__dict__)
    exec(compile(scheduled, "<f16_scheduled_ref>", "exec"), namespace)  # noqa: S102
    return namespace["run"].__code__.co_code


# ---------------------------------------------------------------------------
# Per-thread state read by the injected run (_f16_yield) and the wrapped issue_next.
# ---------------------------------------------------------------------------
_TLS = threading.local()


class _PipelineAborted(BaseException):
    """Raised inside a group thread's yield to unwind it when the OTHER group failed
    (or the baton timed out).  A BaseException so it is not swallowed by ``except
    Exception`` and reaches ``PackedDecode.run``'s BaseException route cleanup."""


def f16_yield() -> None:
    """The one baton hand-off point, bound onto every runner as ``_f16_yield``.
    A no-op unless the current thread is an armed pipeline group thread."""
    baton = getattr(_TLS, "baton", None)
    if baton is None:
        return
    baton.yield_turn(_TLS.role)


def issue_suppressed() -> bool:
    """True on the trailer group's thread (the leader already issued that layer's
    next-projection; re-issuing would clobber a buffer the leader still needs)."""
    return getattr(_TLS, "suppress_issue_next", False)


# ---------------------------------------------------------------------------
# Strict baton (exactly one group runs at a time; leader stays one layer ahead)
# ---------------------------------------------------------------------------
class _Baton:
    LEADER = 0
    TRAILER = 1

    def __init__(self, *, timeout: float = BATON_TIMEOUT_S) -> None:
        self._cond = threading.Condition()
        self._turn = self.LEADER               # whose turn it is to RUN
        self._leader_skips_first = True        # leader gets one layer ahead
        self._done = [False, False]            # a group's forward returned/aborted
        self._aborted = False
        self.error: Optional[BaseException] = None
        self.handoffs = 0
        self._timeout = float(timeout)

    # -- called by the group thread targets ---------------------------------
    def await_start(self, role: int) -> None:
        """Block until it is ``role``'s turn to begin its forward.  The leader
        returns immediately (turn starts LEADER); the trailer waits for the leader's
        second-yield hand-over so it never begins a layer the leader has not passed."""
        with self._cond:
            self._wait_until_turn_locked(role)

    def finish(self, role: int) -> None:
        """A group's forward returned (or unwound on abort): release the other."""
        with self._cond:
            self._done[role] = True
            self._turn = 1 - role
            self._cond.notify_all()

    def fail(self, role: int, error: BaseException) -> None:
        """A group raised: record the first real error, abort, release the other."""
        with self._cond:
            if self.error is None and not isinstance(error, _PipelineAborted):
                self.error = error
            self._aborted = True
            self._done[role] = True
            self._cond.notify_all()

    def abort(self) -> None:
        with self._cond:
            self._aborted = True
            self._cond.notify_all()

    # -- called from inside the run (via f16_yield) -------------------------
    def yield_turn(self, role: int) -> None:
        with self._cond:
            if self._aborted:
                raise _PipelineAborted()
            other = 1 - role
            if self._done[other]:
                return  # the other group finished; run freely to completion
            if role == self.LEADER and self._leader_skips_first:
                self._leader_skips_first = False
                return  # leader does not hand over at its first yield (get ahead)
            self._turn = other
            self.handoffs += 1
            self._cond.notify_all()
            self._wait_until_turn_locked(role)

    # -- internal (cond held) ----------------------------------------------
    def _wait_until_turn_locked(self, role: int) -> None:
        other = 1 - role
        deadline = time.monotonic() + self._timeout
        while self._turn != role:
            if self._aborted:
                raise _PipelineAborted()
            if self._done[other]:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._aborted = True
                if self.error is None:
                    self.error = TimeoutError(
                        f"F16 baton timed out after {self._timeout}s"
                    )
                self._cond.notify_all()
                raise _PipelineAborted()
            self._cond.wait(remaining)
        if self._aborted:
            raise _PipelineAborted()


# ---------------------------------------------------------------------------
# The pipeline object (injected into the hybrid decode namespace as _F16_PIPELINE)
# ---------------------------------------------------------------------------
class Pipeline:
    """Holds the target model and drives the two-group verify pipeline.

    When ``armed`` is False (MTPLX_DSV41_F16 != 1) ``pipelined_forward`` is a pure
    passthrough to the retained ``forward`` -- an explicit construction-time route,
    so a staged tree run with F16 off is a byte-for-byte stock A/B control.
    """

    def __init__(self, model, *, armed: bool, baton_timeout: float = BATON_TIMEOUT_S):
        self.model = model
        self.backbone = model.model
        self.armed = bool(armed)
        self.baton_timeout = float(baton_timeout)
        self.join_timeout = float(baton_timeout) + 10.0
        self.counters = {
            "calls": 0,
            "single_forwards": 0,
            "pipelined_forwards": 0,
            "handoffs": 0,
        }

    # -- entry (replaces the ONE verify forward call) -----------------------
    def pipelined_forward(self, forward: Callable[[Any, Any], tuple], ids, cache):
        if not self.armed:
            return forward(ids, cache)
        self.counters["calls"] += 1
        rows = int(ids.shape[1])
        if rows <= A_ROWS:
            # Runtime M-route (logical row count genuinely varies): a single group
            # needs no split/threads.  Runs the rebound yield run; _f16_yield no-ops.
            self.counters["single_forwards"] += 1
            return forward(ids, cache)
        self.counters["pipelined_forwards"] += 1
        return self._run_pipeline(ids, cache)

    # -- two-group interleave ----------------------------------------------
    def _run_pipeline(self, ids, cache):
        backbone = self.backbone
        rows = int(ids.shape[1])
        rows_a, rows_b = A_ROWS, rows - A_ROWS
        ids_a, ids_b = ids[:, :A_ROWS], ids[:, A_ROWS:]
        offset0 = int(cache.offset)

        # The clone omits the device-route cold-recovery block, so it MUST be off.
        if backbone._device_route_active(cache, rows_a) or backbone._device_route_active(
            cache, rows_b
        ):
            raise RuntimeError("F16 pipeline requires the device route OFF")

        # Admit the whole forward once, up front (mirrors Model->backbone.__call__).
        admit = getattr(cache, "assert_can_admit", None)
        if callable(admit):
            admit(rows_a + rows_b)

        # Engram history advanced for A then B up front, in position order (identical
        # buffer/length to running the two spans sequentially -- the _forward_layer_
        # major discipline); each group reads its own captured advance via a view.
        engram_state = getattr(cache, "engram_state", None)
        if engram_state is not None:
            cur_a = engram_state.advance(ids_a)
            cur_b = engram_state.advance(ids_b)
        else:
            cur_a = cur_b = None

        baton = _Baton(timeout=self.baton_timeout)
        results = self._run_groups(
            baton,
            lambda: self._group_forward(
                ids_a, cache, offset0=offset0, start=0, engram_current=cur_a
            ),
            lambda: self._group_forward(
                ids_b, cache, offset0=offset0, start=rows_a, engram_current=cur_b
            ),
        )

        # Both groups complete: advance the cache once, then concatenate exactly as
        # the sequential chunk loop concatenates its parts.
        cache.advance(rows_a + rows_b)
        logits_a, mh_a = results[_Baton.LEADER]
        logits_b, mh_b = results[_Baton.TRAILER]
        logits = mx.concatenate([logits_a, logits_b], axis=1)
        main_hidden = (
            None
            if (mh_a is None or mh_b is None)
            else mx.concatenate([mh_a, mh_b], axis=1)
        )
        return logits, main_hidden

    # -- run two group callables under the strict baton (real thread machinery) -
    def _run_groups(self, baton, group_leader, group_trailer) -> list:
        """Run two zero-arg group callables under ``baton``, each on its own Python
        thread, on the main generation thread's stream, and under the main thread's
        routing context (copied per thread).  Returns ``[leader_result,
        trailer_result]`` or raises the first group error (never hangs)."""
        stream = mx.default_stream(mx.default_device())
        results: list = [None, None]
        groups = {_Baton.LEADER: group_leader, _Baton.TRAILER: group_trailer}

        def run_group(role):
            _TLS.baton = baton
            _TLS.role = role
            _TLS.suppress_issue_next = role == _Baton.TRAILER
            try:
                baton.await_start(role)
                # Both groups issue on the main generation thread's stream (MLX gives
                # each thread its OWN default stream, so without this A and B would
                # land on different streams and diverge from the single-stream oracle).
                with mx.stream(stream):
                    results[role] = groups[role]()
                baton.finish(role)
            except _PipelineAborted:
                baton.finish(role)
            except BaseException as error:  # noqa: BLE001 - propagated via the driver
                baton.fail(role, error)
            finally:
                _TLS.baton = None
                _TLS.role = None
                _TLS.suppress_issue_next = False

        # A separate context copy per thread carries the main thread's routing phase
        # (attention_phase / expert_routing_phase are ContextVars that do NOT
        # propagate to a plain Thread -- a verify group routed as PREFILL would be
        # catastrophic).  Two distinct Context objects avoid a run() reentrancy clash.
        ctx_l = contextvars.copy_context()
        ctx_t = contextvars.copy_context()
        tl = threading.Thread(
            target=lambda: ctx_l.run(run_group, _Baton.LEADER), name="f16-verify-A"
        )
        tt = threading.Thread(
            target=lambda: ctx_t.run(run_group, _Baton.TRAILER), name="f16-verify-B"
        )
        tl.start()
        tt.start()
        tl.join(self.join_timeout)
        tt.join(self.join_timeout)
        if tl.is_alive() or tt.is_alive():
            baton.abort()
            raise RuntimeError("F16 pipeline group thread did not terminate")
        self.counters["handoffs"] += baton.handoffs
        if baton.error is not None:
            raise baton.error
        return results

    # -- per-group forward: faithful clone of _forward_span + Model head tail -
    def _group_forward(self, ids_group, cache, *, offset0, start, engram_current):
        """One verify row group through every layer, appending to the shared cache
        WITHOUT advancing it, and returning ``(logits, main_hidden)`` -- the exact
        tuple ``model(ids, cache=cache, return_hidden=True)`` returns for these rows.

        Byte-for-byte vs ``_forward_span`` + ``Model.__call__`` (verify path:
        emit_logits=True, logits_keep/logits_rows=None -> head every row) except:
          (i)  positions from the shared ``offset0`` + ``start`` (not cache.offset),
          (ii) engram advanced up front by the driver; this reads its captured rows
               through ``_ChunkEngramView`` (as ``_forward_layer_major`` does),
          (iii) no ``cache.advance`` here (the driver advances once, after both).
        The device-route recovery is asserted off (never entered).  Host-only stage/
        timeline instrumentation (``_stime``/``_tl``, no mx ops, assumes the single
        generation thread) is omitted; the arithmetic is unchanged."""
        from mtplx.models.deepseek_v41 import _ChunkEngramView, _rmsnorm

        backbone = self.backbone
        model = self.model
        b, s = ids_group.shape

        positions = mx.arange(offset0 + start, offset0 + start + s)  # (i)

        h = backbone.embed_tokens(ids_group)
        h = mx.broadcast_to(h[:, :, None, :], (b, s, backbone.hc_mult, h.shape[-1]))
        pre_mix = mx.concatenate(
            [mx.ones((b, s, 1)), mx.zeros((b, s, backbone.hc_mult - 1))], axis=-1
        ).astype(mx.float32)

        engram_state = getattr(cache, "engram_state", None)
        engram_view = (
            _ChunkEngramView(engram_current) if engram_current is not None else None
        )
        shared = cache.new_shared_runtime()  # own shared runtime per group
        want_main = bool(backbone._mtp_target_layer_ids)
        main_hiddens: list = []

        if backbone._device_route_active(cache, s):  # never cloned; must be off
            raise RuntimeError("F16 group forward requires the device route OFF")

        for layer in backbone.layers:
            if layer.engram_hook is not None and engram_state is not None:
                h = layer.engram_hook(h, ids_group, engram_view)  # (ii)
            if want_main and layer.layer_id in backbone._mtp_target_layer_ids:
                main_hiddens.append(
                    mx.mean(h.astype(mx.float32), axis=2).astype(h.dtype)
                )
            h, pre_mix = layer(
                h, pre_mix, positions, cache.layers[layer.layer_id], shared
            )
        # (iii) no cache.advance here.

        h = mx.sum(pre_mix[..., None] * h.astype(mx.float32), axis=2).astype(h.dtype)
        out = _rmsnorm(h, backbone.norm_weight, backbone.args.rms_norm_eps)
        main_hidden = mx.concatenate(main_hiddens, axis=-1) if main_hiddens else None

        # Model.__call__ tail for the verify path: head every row, return both.
        logits = model._apply_head(out)
        return logits, main_hidden


class LazyPipeline:
    """What the staged hybrid rewrite injects as ``_F16_PIPELINE`` (review 2026-09-19).

    ``hybrid_install.install`` runs at the START of the DSpark pass -- before prefill -- while
    ``f16.install.install_from_env`` stashes ``model._f16_pipeline`` only at the post-prime
    boundary (after prefill + growth + seed). Binding ``model._f16_pipeline`` at hybrid-install
    time would raise AttributeError before the prefill even starts, so the injected object
    resolves the pipeline at CALL time; the first verify forward happens after the install."""

    __slots__ = ("_model",)

    def __init__(self, model) -> None:
        self._model = model

    def pipelined_forward(self, forward, ids, cache):
        return self._model._f16_pipeline.pipelined_forward(forward, ids, cache)


def bind_yield_run(runners) -> dict:
    """Rebind each retained ``PackedDecode`` runner's ``switch._run`` to the
    yield-capable run and give each runner an ``_f16_yield`` (the shared hand-off).
    Reuses the existing runner instances (``issue_next``, executor, ops preserved).
    ``runners`` maps layer -> (switch, runner)."""
    run_fn, shas = build_yield_run()
    bound = 0
    for _layer, (switch, runner) in runners.items():
        runner._f16_yield = f16_yield
        switch._run = MethodType(run_fn, runner)
        bound += 1
    shas["yield_run_bound_layers"] = bound
    return shas
