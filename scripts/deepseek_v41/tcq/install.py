"""F39: install the tcq3 (eschamoe K=3) decode-verify lane at the post-prefill boundary.

This is an EXPLICIT construction-time route, NOT a runtime fallback (AGENTS.md "correct by design"): the caller
(packed_phase.py, at the plane-lane bind site, via ``stage_tcq_runner.py``) picks tcq3 XOR the stock mxfp4
``plane_lane.install`` ONCE, from the model's codec, before any decode.  Nothing here re-checks per token.

What it does (the ONLY hot-path changes vs the retained mxfp4 lane):
  * DECODE-verify: swaps the mxfp4 scale-codec kernels for the stride-aware trellis tile kernel over a WHOLE-RECORD
    bank, wrapping each projection ``t128`` before / ``t128 * rout`` after (``tcq_runtime.TcqPackedOps``).
  * bank READER: one contiguous 13,290,496-B read per record into a single slot view (vs the mxfp4 3-weight-segment
    reader), because a slot now holds the whole record (code + routs).
  * routs: the whole-bank routs (299 MB) are loaded resident at construction and indexed by global record index.
  * cache-row size: a slot is the tcq3 record (13,290,496 B) instead of the mxfp4 weight code (17,694,720 B).

What it does NOT touch: routing, the accept/commit path, KV, engram, and the draft/MTP resident FP4 experts (the
draft head keeps its own resident weights).  Prefill/seed decode-to-bf16 is ``tcq_runtime.decode_expert_to_bf16``,
installed on the prefill streaming path by the runner, not here.

BOUNDARY (documented; needs the real F38 bank + a tcq3-aware model loader to run end-to-end): this install assumes
the loader presents the tcq3 bank as a single whole-record int16 component per slot
(``tcq_runtime.EXPERT_RECORD_ARRAY``) and reports ``spec.expert_codec == 'tcq3'`` /
``spec.expert_record_bytes == 13,290,496``.  The retained mxfp4 growth+admission ladder (packed_admission.py) still
derives capacity from the 17,694,720-B mxfp4 slot; re-deriving it to bank the ~33% headroom is follow-up work.
"""
from __future__ import annotations

import os
import sys

# tcq_runtime lives in the trellis package next door; add it to sys.path at import time.
_HERE = os.path.dirname(os.path.abspath(__file__))
_TRELLIS = os.path.join(os.path.dirname(_HERE), "trellis")
if _TRELLIS not in sys.path:
    sys.path.insert(0, _TRELLIS)

import tcq_runtime as R          # noqa: E402

TCQ3_CODEC = "tcq3"
TCQ3_QUANT_BITS = 3
TCQ3_RECORD_BYTES = 13_290_496
ENV_FLAG = "MTPLX_DSV41_TCQ3"


def validate_tcq_config(runtime, switches, routs_by_layer) -> None:
    """The single construction-time gate.  Raise unless this is the exact fixed tcq3 D5/M<=8 decode lane.

    Mirrors ``plane_lane.install``'s guard, retargeted to tcq3: codec / bits / record bytes are tcq3, the decode
    geometry (hidden sizes, top_k, swiglu limit, miss records) is the proven one, and the shared runtime invariants
    (no prefetch/telemetry/pipeline ledger, single slot pool, deferred split release, fanout executor) hold.
    """
    s, c, p = runtime.spec, runtime.config, runtime.plan
    bad = []
    if s.expert_codec != TCQ3_CODEC:
        bad.append(f"expert_codec={s.expert_codec!r} (need {TCQ3_CODEC!r})")
    if getattr(s, "quant_bits", None) != TCQ3_QUANT_BITS:
        bad.append(f"quant_bits={getattr(s, 'quant_bits', None)} (need {TCQ3_QUANT_BITS})")
    if getattr(s, "expert_record_bytes", None) != TCQ3_RECORD_BYTES:
        bad.append(f"expert_record_bytes={getattr(s, 'expert_record_bytes', None)} (need {TCQ3_RECORD_BYTES})")
    if (s.hidden_size, s.expert_hidden_size, s.top_k, s.swiglu_limit) != (5120, 2304, 6, 10.0):
        bad.append(f"decode geometry {(s.hidden_size, s.expert_hidden_size, s.top_k, s.swiglu_limit)} "
                   "!= (5120, 2304, 6, 10.0)")
    if c.cache_scope != "layer" or c.decode_miss_records_per_part != 3:
        bad.append(f"cache_scope={c.cache_scope!r} decode_miss_records_per_part={c.decode_miss_records_per_part}")
    if c.prefetch_slots or getattr(c, "resource_telemetry", False):
        bad.append("prefetch_slots/resource_telemetry must be off")
    if c.split_route_release != "deferred" or p.transient_slots < 48:
        bad.append(f"split_route_release={c.split_route_release!r} transient_slots={p.transient_slots}")
    if runtime._pipeline_ledger is not None or not runtime._single_slot_pool:
        bad.append("pipeline ledger must be None and the single-slot pool present")
    if runtime.reader._fanout_executor is None:
        bad.append("reader fanout executor is required")
    if set(switches) != set(routs_by_layer) or set(switches) != set(s.routed_layer_indices):
        bad.append("switches must cover exactly the routed layers and match routs_by_layer")
    if bad:
        raise RuntimeError("tcq3 lane requires its proven fixed decode configuration: " + "; ".join(bad))


def bind_tcq_reader(reader, local):
    """Whole-record reader: ONE contiguous ``record_bytes`` read per record into the slot's record view.

    tcq3 stores the whole record (code + routs) contiguously, so unlike the mxfp4 3-weight-segment reader
    (plane_lane.bind_reader) a slot needs a single read.  The early gate/up witness is unavailable (the whole
    record arrives at once), so this binds the plain full-record publication path.
    """
    read_range = reader._readv_range_into
    metrics = reader.metrics

    def run(items, cancel_event, deadline_ns, pipeline_phase):
        views = []
        try:
            for record, dest in items:
                view = dest.component_view(R.EXPERT_RECORD_ARRAY)
                views.append(view)
                read_range("experts.bin", record.sidecar_offset, (view,), cancel_event=cancel_event,
                           deadline_ns=deadline_ns, pipeline_phase=pipeline_phase)
            metrics.update(record_requests=len(items), records_read=len(items),
                           sidecar_record_requests=len(items), records_unhashed=len(items))
            return ("unverified",) * len(items)
        finally:
            for view in views:
                view.release()

    def read_one(manifest, record, destination, *, prefer_sidecar=True, verify_hash=True,
                 cancel_event=None, deadline_ns=None, pipeline_phase=None):
        return run(((record, destination),), cancel_event, deadline_ns, pipeline_phase)[0]

    def read_batch(manifest, items, *, verify_hash=True, cancel_event=None,
                   deadline_ns=None, pipeline_phase=None):
        return run(items, cancel_event, deadline_ns, pipeline_phase) if items else ()

    reader.read_record_into = read_one
    reader.read_component_records_into = read_batch


def install(runtime, switches, routs_by_layer, *, tables, early=False):
    """Install the tcq3 decode-verify lane.  ``routs_by_layer`` maps layer -> the resident whole-bank routs dict
    (all layers share the same three arrays); ``tables`` are the warp tables (tcq_kernel_check.warp_tables()).

    Reuses the retained ``plane_lane`` split-route machinery (PartExecutor / ReaderExecutor / PackedDecode); only
    the per-layer ops (TcqPackedOps) and the reader binding differ.  ``early`` defaults False: the whole-record
    read has no partial gate/up witness.
    """
    import threading
    import mlx.core as mx
    from plane_lane import PartExecutor, ReaderExecutor, PackedDecode

    validate_tcq_config(runtime, switches, routs_by_layer)
    mx.synchronize()
    runtime.flush_deferred_slot_releases()
    local = threading.local()
    executor = PartExecutor(runtime._split_executor, local)
    runtime._split_executor = executor
    runtime.slots._executor = ReaderExecutor(runtime.slots._executor, local)
    bind_tcq_reader(runtime.reader, local)
    runners = {}
    for layer, switch in switches.items():
        ops = R.TcqPackedOps(routs_by_layer[layer], layer, tables=tables)
        runner = PackedDecode(runtime, layer, ops, executor, early=early)
        switch._run = runner.run
        runners[layer] = runner
    return runners


def tcq3_enabled() -> bool:
    """True when the launcher armed the tcq3 lane (``MTPLX_DSV41_TCQ3=1``)."""
    return os.environ.get(ENV_FLAG) == "1"


def _tcq_model_dir() -> str:
    """The tcq3 artifact directory: the launcher's ``GPU_WINDOW_CANDIDATE_MODEL_DIR`` (falls back to MODEL_DIR)."""
    d = os.environ.get("GPU_WINDOW_CANDIDATE_MODEL_DIR") or os.environ.get("MODEL_DIR")
    if not d:
        raise RuntimeError("tcq3 lane armed but GPU_WINDOW_CANDIDATE_MODEL_DIR is unset")
    return d


def _load_tcq_resources(runtime):
    """Load the resident routs (299 MB, once) and the warp tables for the tile kernel from the tcq3 artifact."""
    from tcq_kernel_check import warp_tables
    manifest = R.read_tcq_manifest(os.path.join(_tcq_model_dir(), "expert-manifest.json"))
    routs = R.load_resident_routs(manifest)
    return manifest, routs, warp_tables()


def route_plane_lane(runtime, layers, switches, owners, mxfp4_install):
    """The explicit construction-time route called from packed_phase.py's plane-lane bind site.

    tcq3 armed -> install the tcq3 lane (load routs + tables once, wire TcqPackedOps).  Otherwise -> the stock
    mxfp4 ``plane_lane.install(runtime, switches_by_layer, owners)``.  One decision, at construction, from the
    model's codec and the arm flag; never a per-token fallback.
    """
    switches_by_layer = dict(zip(layers, switches))
    if not tcq3_enabled():
        return mxfp4_install(runtime, switches_by_layer, owners)
    _manifest, routs, tables = _load_tcq_resources(runtime)
    routs_by_layer = {layer: routs for layer in switches_by_layer}     # all layers share the resident bank routs
    return install(runtime, switches_by_layer, routs_by_layer, tables=tables)


def install_from_env(runtime, switches, routs_by_layer, *, tables, early=False):
    """Construction-time route selector.  If ``MTPLX_DSV41_TCQ3=1`` install the tcq3 lane and return its runners;
    otherwise return ``None`` so the caller takes the stock ``plane_lane.install`` route.  Never a per-token branch.
    """
    if not tcq3_enabled():
        return None
    return install(runtime, switches, routs_by_layer, tables=tables, early=early)
