"""F16 CPU preflight: verify every attribute, method, anchor and pinned source the
install + staged edits depend on EXISTS and resolves on the REAL pinned classes and
the REAL archived helper sources -- so a 2.5-minute GPU prefill cannot then die on an
AttributeError or a moved anchor.

Run (packed sources + scripts dir + the private greenlet dir `.f16-site` on PYTHONPATH),
MLX pinned to CPU:
    python -m f16.preflight
Prints ``F16_PREFLIGHT_OK {json}`` and exits 0, or raises with the first failure.
Pure CPU; never imports/loads an artifact, never touches Metal or the GPU lock.
"""
from __future__ import annotations

import contextvars
import inspect
import json
from pathlib import Path

# greenlet is the pipeline driver; fail loudly (not a late ImportError) if the private
# .f16-site dir is not on PYTHONPATH -- a GPU window must not discover this after prefill.
try:
    import greenlet  # noqa: F401
except ImportError as _exc:  # pragma: no cover
    raise RuntimeError(
        "F16 requires the 'greenlet' package; put the private .f16-site dir on PYTHONPATH "
        "(it is NOT in the shared venv)"
    ) from _exc

import mlx.core as mx

mx.set_default_device(mx.cpu)

from . import stage_f16_runner as stager  # noqa: E402
from . import pipeline as pl  # noqa: E402


def _require(obj, name: str, kind: str) -> None:
    if not hasattr(obj, name):
        raise RuntimeError(f"F16 preflight: {kind} is missing attribute {name!r}")


def _check_pinned_runtime() -> dict:
    """Methods/attributes on the pinned mtplx classes the clone + install use, plus
    the cloned-source SHAs (verify_source_pins)."""
    from mtplx.models import deepseek_v41 as dv
    from mtplx.models import deepseek_v41_cache as dvc
    from mtplx import expert_runtime as er
    from mtplx import attention_context as ac
    from mtplx.models import expert_mlx as em

    # Class-level methods (functions defined in the class body).
    for name in ("_forward_span", "_forward_layer_major", "_device_route_active"):
        _require(dv.DeepseekV41Backbone, name, "DeepseekV41Backbone")
    for name in ("__call__", "_apply_head", "make_cache"):
        _require(dv.Model, name, "Model")
    for name in ("_ChunkEngramView", "_rmsnorm"):
        _require(dv, name, "deepseek_v41 module")
    _require(dv._ChunkEngramView, "current_row_ids", "_ChunkEngramView")
    for name in ("offset", "advance", "new_shared_runtime", "assert_can_admit",
                 "engram_state", "mark", "rollback"):  # layers is instance-set (smoke run)
        _require(dvc.DeepseekV41Cache, name, "DeepseekV41Cache")
    for name in ("begin_split_route", "defer_slot_release", "flush_deferred_slot_releases"):
        _require(er.ExpertStreamingRuntime, name, "ExpertStreamingRuntime")

    # The per-layer-lock premise the baton needs (else a global bank -> one lock).
    rt_src = inspect.getsource(er.ExpertStreamingRuntime.__init__)
    if "self._global_bank is not None" not in rt_src or "self._layer_locks" not in rt_src:
        raise RuntimeError("F16 preflight: expert_runtime per-layer-lock branch not found")

    # The routing phase carriers MUST be ContextVars (do not propagate to a plain
    # thread; the pipeline propagates them via copy_context().run per group thread).
    for mod, vname, ph in ((ac, "_ATTENTION_PHASE", "attention"),
                           (em, "_ROUTING_PHASE", "routing")):
        _require(mod, vname, f"{ph} phase module")
        if not isinstance(getattr(mod, vname), contextvars.ContextVar):
            raise RuntimeError(f"F16 preflight: {ph} phase carrier is not a ContextVar")
    _require(ac, "attention_phase", "attention_context")
    _require(em, "expert_routing_phase", "expert_mlx")

    # Instance attributes the clone reads exist on a real (tiny) backbone, and the
    # whole clone path resolves end-to-end on CPU (the strongest AttributeError guard).
    _smoke_clone_on_tiny_model()
    return pl.verify_source_pins()


def _smoke_clone_on_tiny_model() -> None:
    """Build a tiny DeepSeek-V4.1 model (no artifact) and run ONE armed pipelined
    forward (6 rows) so every backbone/Model/cache attribute the clone touches is
    exercised.  Correctness (bit-parity) is proven by the CPU tests; this only
    guarantees the clone path resolves without an AttributeError."""
    from mtplx.models.deepseek_v41 import Model, ModelArgs

    args = ModelArgs(
        vocab_size=48, hidden_size=32, num_hidden_layers=6, num_attention_heads=4,
        head_dim=16, qk_rope_head_dim=4, q_lora_rank=12, o_lora_rank=8, o_groups=2,
        moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=2,
        index_n_heads=2, index_head_dim=8, index_topk=5, sliding_window=8,
        window_size=8, swiglu_limit=0.5, compress_ratios=[0, 0, 2, 2, 2, 1],
        kv_source_layer_ids=[2, 5], index_source_layer_ids=[2, 5],
        candidate_source_layer_id=5, candidate_topk_blocks=3, candidate_block_size=2,
        rope_scaling={"rope_type": "yarn", "factor": 16, "beta_fast": 32,
                      "beta_slow": 1, "original_max_position_embeddings": 65536},
        dspark_target_layer_ids=[1, 3],
    )
    model = Model(args)
    mx.eval(model.parameters())
    for attr in ("embed_tokens", "layers", "norm_weight", "hc_mult",
                 "_mtp_target_layer_ids", "args"):
        _require(model.model, attr, "DeepseekV41Backbone instance")
    forward = lambda ids, cache: model(ids, cache=cache, return_hidden=True)  # noqa: E731
    import numpy as np

    ids = mx.array(np.random.RandomState(0).randint(0, args.vocab_size, size=(1, 6)))
    # Both hand-off modes must resolve the whole clone path end-to-end (the strongest
    # AttributeError guard).  The CPU model has no streamed switch, so the hand-offs do
    # not fire inside model(...); correctness (bit-parity) is proven by the CPU tests.
    for handoff in ("reads", "barrier"):
        pipeline = pl.Pipeline(model, armed=True, handoff=handoff)
        logits, main_hidden = pipeline.pipelined_forward(forward, ids, model.make_cache())
        mx.eval(logits, main_hidden)
        if tuple(logits.shape) != (1, 6, args.vocab_size):
            raise RuntimeError(
                f"F16 preflight: clone logits shape {tuple(logits.shape)} ({handoff})"
            )
    _smoke_barrier_handoff(model)


def _smoke_barrier_handoff(model) -> None:
    """Resolve the F18 barrier machinery on CPU without a streamed switch: run the
    per-group driver over two trivial group callables that fire ``barrier`` (orphan
    adopt + async_eval + hand off) and ``f16_yield`` per layer, with a stand-in runtime
    whose ``_deferred_slot_releases`` the driver swaps.  Catches an AttributeError in
    ``barrier`` / ``_run_groups_barrier`` before a window ever binds them."""
    import types

    model._mtplx_expert_runtime = types.SimpleNamespace(_deferred_slot_releases=None)
    pipeline = pl.Pipeline(model, armed=True, handoff="barrier")

    def group(_role):
        def run():
            for _layer in range(3):
                pipeline.barrier(mx.array([0], mx.int32))  # submit + hand off
                pl.f16_yield()                              # reads hand off
            return "ok"

        return run

    results = pipeline._run_groups_barrier(group(pl.LEADER), group(pl.TRAILER))
    if results != ["ok", "ok"] or pipeline.counters["handoffs"] <= 2:
        raise RuntimeError(f"F16 preflight: barrier hand-off smoke failed ({results})")
    del model._mtplx_expert_runtime


def _check_derivation() -> dict:
    """The retained plane_lane pin holds and BOTH hand-off modes' runs -- plus the
    stamped variants and the scheduled-run reference -- derive and compile (unique
    anchors + round trip).  So a GPU window in either mode cannot die on a moved anchor.
    Returns the merged SHA report, which includes both derived-run SHAs."""
    report: dict = {}
    reads_fn, reads_shas = pl.build_yield_run()             # reads mode (yield only)
    barrier_fn, barrier_shas = pl.build_barrier_run()       # F18 barrier mode (barrier+yield)
    if not callable(reads_fn) or not callable(barrier_fn):
        raise RuntimeError("F16 preflight: a derived run did not compile")
    if reads_fn.__code__.co_code == barrier_fn.__code__.co_code:
        raise RuntimeError("F16 preflight: barrier run bytecode matches the reads run")
    report.update(reads_shas)
    report["f16_barrier_run_sha256"] = barrier_shas["f16_barrier_run_sha256"]

    # Stamped variants (diagnostic): derive + round-trip so a stamped window cannot die
    # on a moved anchor either.
    for barrier in (False, True):
        sfn, sshas = pl.build_stamped_run(barrier=barrier)
        if not callable(sfn):
            raise RuntimeError("F16 preflight: a stamped run did not compile")
        report.update({k: v for k, v in sshas.items() if k.startswith("f16_stamped_")})

    cocode = pl.scheduled_run_cocode()
    if not isinstance(cocode, bytes) or not cocode:
        raise RuntimeError("F16 preflight: scheduled-run cocode empty")
    return report


def _import_engram_parallel():
    """The F6 module, imported whichever way is on PYTHONPATH: top-level
    ``engram_parallel`` (how the F6 stager imports it -- ``scripts/deepseek_v41/f6``
    on the path) or the ``f6.engram_parallel`` namespace package
    (``scripts/deepseek_v41`` on the path)."""
    try:
        import engram_parallel as ep  # noqa: F811
        return ep
    except ImportError:
        from f6 import engram_parallel as ep  # type: ignore
        return ep


def _check_f20() -> dict:
    """F20 engram read lookahead resolves end-to-end on CPU: the f6 lookahead symbols
    import, the Pipeline refuses to arm the lookahead on a non-F6 cache, and a
    construct -> prefetch(both groups) -> collect(F6 gather) -> clear round-trip runs on
    a synthetic file-backed cache.  So a GPU window with
    ``MTPLX_DSV41_F20_ENGRAM_LOOKAHEAD=1`` cannot die at install on an AttributeError or
    a missing symbol.  Never touches Metal or an artifact (pure numpy + preadv)."""
    import os
    import shutil
    import tempfile
    import types

    import numpy as np
    from mtplx.ngram_row_cache import FileRowReader, NGramRowCache, RowGeometry

    ep = _import_engram_parallel()
    for name in ("prefetch_rows", "clear_pending", "ParallelReadState", "install"):
        if not hasattr(ep, name):
            raise RuntimeError(f"F16 preflight: f6 engram_parallel missing {name!r} (F20 lane)")
    probe = ep.ParallelReadState(workers=2)
    try:
        for key in ("lookahead_calls", "lookahead_rows_submitted"):
            if key not in probe.stats:
                raise RuntimeError(f"F16 preflight: ParallelReadState.stats missing {key!r}")
    finally:
        probe.shutdown()

    geom = RowGeometry(values_per_row=256, bits=8, group_size=32, mode="mxfp8")  # 264 B/row
    row_bytes, num_rows = geom.row_bytes, 64
    tmpdir = tempfile.mkdtemp(prefix="f16-preflight-f20-")
    try:
        path = os.path.join(tmpdir, "engram-preflight.bin")
        blob = (np.arange(num_rows * row_bytes, dtype=np.uint64) % 251).astype(np.uint8)
        with open(path, "wb") as fh:
            fh.write(blob.tobytes())

        # Refuse-gate: arming the lookahead on a cache the F6 gather is NOT installed on
        # must raise at construction (correct-by-design; not per forward).
        plain = NGramRowCache(FileRowReader(path, row_bytes=row_bytes, num_rows=num_rows),
                              geom, num_rows=num_rows, cache_bytes=num_rows * row_bytes)
        plain_layers = [types.SimpleNamespace(
            engram_hook=types.SimpleNamespace(layer_hash_index=0, row_cache=plain))]
        plain_model = types.SimpleNamespace(model=types.SimpleNamespace(layers=plain_layers))
        try:
            pl.Pipeline(plain_model, armed=True, engram_lookahead=True)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("F16 preflight: F20 enable did not refuse a non-F6 cache")
        plain.close()

        # Enabled path: two engram-hook layers sharing one F6-installed cache.
        cache = NGramRowCache(FileRowReader(path, row_bytes=row_bytes, num_rows=num_rows),
                              geom, num_rows=num_rows, cache_bytes=num_rows * row_bytes)
        state = ep.install([cache], workers=2)
        layers = [types.SimpleNamespace(engram_hook=types.SimpleNamespace(layer_hash_index=0, row_cache=cache)),
                  types.SimpleNamespace(engram_hook=types.SimpleNamespace(layer_hash_index=1, row_cache=cache)),
                  types.SimpleNamespace(engram_hook=None)]
        model = types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))
        pipe = pl.Pipeline(model, armed=True, engram_lookahead=True)

        n_layers, cols = 2, 3
        cur_a = np.arange(2 * n_layers * cols).reshape(1, 2, n_layers, cols) % num_rows
        cur_b = (np.arange(2 * n_layers * cols).reshape(1, 2, n_layers, cols) + 5) % num_rows
        pipe._engram_lookahead(cur_a, cur_b)
        if not getattr(cache, "_f20_pending", None):
            raise RuntimeError("F16 preflight: F20 prefetch queued no reads")
        row_ids = np.asarray(cur_a)[:, :, 0, :].reshape(-1).tolist()
        got = cache.gather_bytes(row_ids)  # collect through the F6 gather (uses prefetched bytes)
        if tuple(got.shape) != (len(row_ids), row_bytes):
            raise RuntimeError(f"F16 preflight: F20 collect shape {tuple(got.shape)}")
        pipe._engram_clear()
        if getattr(cache, "_f20_pending", None):
            raise RuntimeError("F16 preflight: F20 clear left pending futures")
        report = {
            "module": getattr(ep, "__name__", "?"),
            "engram_layers": len(pipe._f20_layers),
            "lookahead_stats": sorted(k for k in state.stats if k.startswith("lookahead")),
            "refuse_without_f6": True,
        }
        state.shutdown()
        cache.close()
        return report
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _archived_dir() -> Path:
    import projection_install

    return Path(projection_install.__file__).resolve().parent


def _check_staged_anchors() -> dict:
    """Every staged edit's anchors resolve, are unique, and round-trip on the REAL
    archived helper sources (run the stage functions in memory; do not write)."""
    d = _archived_dir()
    results = {}
    for fname, fn in (("run_full.py", stager.stage_run_full),
                      ("hybrid_install.py", stager.stage_hybrid_install),
                      ("projection_install.py", stager.stage_projection_install)):
        src = (d / fname).read_text()
        out = fn(src)
        if out == src:
            raise RuntimeError(f"F16 preflight: staging {fname} produced no change")
        results[fname] = "staged_ok"

    # The verify forward anchor the hybrid replace() targets must exist exactly once
    # in the pinned _decode_cycles source (rewrite() runs against it at window time).
    from mtplx.models import deepseek_v41_dspark_decode as dec

    dec_src = inspect.getsource(dec._decode_cycles)
    fwd = "                chunk_logits, chunk_hidden = forward(mx.array([chunk_ids]), cache)"
    if dec_src.count(fwd) != 1:
        raise RuntimeError(
            f"F16 preflight: verify-forward anchor count {dec_src.count(fwd)} != 1 "
            "in _decode_cycles"
        )
    results["decode_forward_anchor"] = "unique"
    return results


def preflight() -> dict:
    report = {
        "greenlet_version": greenlet.__version__,
        "pinned_runtime": _check_pinned_runtime(),
        "derivation": _check_derivation(),
        "staged_anchors": _check_staged_anchors(),
        "f20_engram_lookahead": _check_f20(),
        "extra_projection_bytes": 2 * 67_108_864,
    }
    return report


def main() -> int:
    report = preflight()
    deriv = report["derivation"]
    # Both hand-off modes' derived-run SHAs (the reads yield run and the F18 barrier
    # run), explicit so a window log records exactly which run each mode compiled.
    print("F16_DERIVED_RUN_SHA reads=" + deriv["f16_yield_run_sha256"]
          + " barrier=" + deriv["f16_barrier_run_sha256"])
    print("F16_PREFLIGHT_OK " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
