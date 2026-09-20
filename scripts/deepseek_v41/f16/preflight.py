"""F16 CPU preflight: verify every attribute, method, anchor and pinned source the
install + staged edits depend on EXISTS and resolves on the REAL pinned classes and
the REAL archived helper sources -- so a 2.5-minute GPU prefill cannot then die on an
AttributeError or a moved anchor.

Run (packed sources + scripts dir on PYTHONPATH), MLX pinned to CPU:
    python -m f16.preflight
Prints ``F16_PREFLIGHT_OK {json}`` and exits 0, or raises with the first failure.
Pure CPU; never imports/loads an artifact, never touches Metal or the GPU lock.
"""
from __future__ import annotations

import contextvars
import inspect
import json
from pathlib import Path

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
    pipeline = pl.Pipeline(model, armed=True)
    forward = lambda ids, cache: model(ids, cache=cache, return_hidden=True)  # noqa: E731
    import numpy as np

    ids = mx.array(np.random.RandomState(0).randint(0, args.vocab_size, size=(1, 6)))
    logits, main_hidden = pipeline.pipelined_forward(forward, ids, model.make_cache())
    mx.eval(logits, main_hidden)
    if tuple(logits.shape) != (1, 6, args.vocab_size):
        raise RuntimeError(f"F16 preflight: clone logits shape {tuple(logits.shape)}")


def _check_derivation() -> dict:
    """The retained plane_lane pin holds and the yield-capable run + scheduled-run
    reference both derive and compile (unique anchor + round trip)."""
    run_fn, shas = pl.build_yield_run()
    if not callable(run_fn):
        raise RuntimeError("F16 preflight: yield run did not compile")
    cocode = pl.scheduled_run_cocode()
    if not isinstance(cocode, bytes) or not cocode:
        raise RuntimeError("F16 preflight: scheduled-run cocode empty")
    return shas


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
        "pinned_runtime": _check_pinned_runtime(),
        "derivation": _check_derivation(),
        "staged_anchors": _check_staged_anchors(),
        "extra_projection_bytes": 2 * 67_108_864,
    }
    return report


def main() -> int:
    report = preflight()
    print("F16_PREFLIGHT_OK " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
