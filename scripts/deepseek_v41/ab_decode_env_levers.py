#!/usr/bin/env python3
"""W28/W30 A/B arms: DeepSeek-V4.1-Flash env-flag levers (K1 shared overlap, K16 layer-major prefill) with the host-sync census.

Kept separate from ab_decode_levers.py (W24), whose arms are ExpertStreamingConfig
overrides (I/O fanout, overlap reads, inflight bytes); this file's arms are model env flags.

The orchestrator runs this INSIDE ``scripts/deepseek_v41/gpu_window.sh`` (holding
the GPU flock, Qwen unloaded, memory-guarded).  ``control`` is the shipped path
(every lever OFF), byte-for-byte; ``shared_overlap`` arms
``MTPLX_DSV41_SHARED_OVERLAP=1`` -- W11's MoE hands its shared expert to the
streamed switch as ``shared_work`` (KERNEL_LEDGER K1), so the switch dispatches
it into the GPU-idle window of the per-layer ``mx.eval(indices)`` routing barrier
+ miss I/O instead of serialising it after the routed gather.  The lever is a
pure execution reorder, so decoded tokens must be byte-identical; a differing
token-id sha256 FAILS the arm.

This is the small self-contained arm the W28 task calls for (feat/deepseek-v41-w24
had not landed on the integration branch).  When W24's ``ab_decode_levers.py``
merges, fold the ``shared_overlap`` preset into its ``ARM_PRESETS`` (a one-line
env-threading addition) and delete this file.

Per arm it reports: prefill tok/s, TTFT, decode tok/s, wall, peak GB, the decoded
token ids + their sha256, and -- with ``--syncs`` -- the per-decoded-token host
sync census off the route-stage probe (``hot.eval_indices`` = the routing barrier
the ledger prices at ~40/token) plus any GPU-overlap telemetry the runtime
exposes.  No GPU work at import; ``--help`` is CPU-safe.

Reuses the proven cell harness in ``bench_standard_shape.py`` (loader, prompt
builder, MLX ops, memory + gather probes) so the numbers are apples-to-apples
with the standard-shape receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

DEFAULT_MODEL = Path(
    "~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4"
).expanduser()
GIB = 1024 ** 3
OVERLAP_ENV = "MTPLX_DSV41_SHARED_OVERLAP"
PROBE_ENV = "MTPLX_ROUTE_STAGE_PROBE"
BARRIER_STAGE = "hot.eval_indices"

LAYER_MAJOR_ENV = "MTPLX_DSV41_PREFILL_LAYER_MAJOR"

ARM_PRESETS = {
    "control": {OVERLAP_ENV: None, LAYER_MAJOR_ENV: None},   # shipped: levers OFF
    "shared_overlap": {OVERLAP_ENV: "1", LAYER_MAJOR_ENV: None},  # W28 K1 lever ON
    "layer_major": {OVERLAP_ENV: None, LAYER_MAJOR_ENV: "1"},     # W30 K16 lever ON
    "both": {OVERLAP_ENV: "1", LAYER_MAJOR_ENV: "1"},
}


def _load_bench_module():
    """Import ``bench_standard_shape`` (a sibling script) for its cell harness.

    Its top level imports only the standard library, so this is CPU-safe.
    """
    path = Path(__file__).resolve().parent / "bench_standard_shape.py"
    spec = importlib.util.spec_from_file_location("_dsv41_bench_standard_shape", path)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise ImportError(f"cannot load bench harness at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--context-tokens", type=int, default=1024, choices=(1024, 16384))
    p.add_argument("--decode-tokens", type=int, default=256)
    p.add_argument(
        "--arms",
        nargs="+",
        default=["control", "shared_overlap"],
        help="preset names from ARM_PRESETS (control, shared_overlap, layer_major, both)",
    )
    p.add_argument("--out", type=Path, required=True, help="append-only JSONL receipt")
    p.add_argument(
        "--syncs",
        type=int,
        default=0,
        metavar="N",
        help="extra probe pass of N decode steps per arm with the route-stage "
        "probe ON, to census host syncs/token (0 = skip; the probe inflates "
        "timing, so it never touches the reported tok/s pass)",
    )
    # GPU-window defaults (agent booted out -> ~82 GiB planner budget).
    p.add_argument("--memory-limit-gib", type=float, default=82.0)
    p.add_argument("--expert-cache-limit-gib", type=float, default=None)
    p.add_argument(
        "--apply-memory-cap", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--slot-layout", default="component-banks")
    p.add_argument("--max-kv", type=int, default=4096)
    p.add_argument("--admit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--admission-receipt", type=Path, default=None)
    p.add_argument(
        "--verify-record-hashes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--seed", type=int, default=0)
    return p


def _apply_arm_env(arm: str) -> None:
    if arm not in ARM_PRESETS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARM_PRESETS)}")
    for key, value in ARM_PRESETS[arm].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _load_model(args, bench, mx):
    from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming

    admission_receipt = None
    if args.admission_receipt is not None:
        admission_receipt = json.loads(Path(args.admission_receipt).read_text())
    max_kv = bench.resolve_max_kv([args.context_tokens], args.decode_tokens, args.max_kv)
    cache_limit = (
        None
        if args.expert_cache_limit_gib is None
        else int(args.expert_cache_limit_gib * GIB)
    )
    resident = load_deepseek_v41_streaming(
        args.model,
        memory_limit_bytes=int(args.memory_limit_gib * GIB),
        max_live_kv_tokens=int(max_kv),
        admit=args.admit,
        admission_receipt=admission_receipt,
        expert_cache_limit_bytes=cache_limit,
        apply_memory_cap=args.apply_memory_cap,
        slot_layout=args.slot_layout,
        cache_scope="layer",
        island_layers=(),
        verify_record_hashes=args.verify_record_hashes,
    )
    return resident


def _generate(*, model, ops, mem_probe, prompt_ids, steps):
    """Greedy prefill + ``steps`` decode; captures the decoded token ids."""
    mem_probe.reset_peak()
    t0 = time.perf_counter()
    cache = model.make_cache()
    logits = model(ops.input([list(prompt_ids)]), cache=cache)
    ops.sync(logits)
    ttft_s = time.perf_counter() - t0
    token = ops.argmax_last(logits)
    generated = [token]

    decode_start = time.perf_counter()
    for _ in range(int(steps)):
        logits = model(ops.input([[token]]), cache=cache)
        ops.sync(logits)
        token = ops.argmax_last(logits)
        generated.append(token)
    decode_wall_s = time.perf_counter() - decode_start
    return {
        "generated": [int(t) for t in generated],
        "ttft_s": ttft_s,
        "decode_wall_s": decode_wall_s,
        "peak_gb": mem_probe.peak_bytes() / GIB,
    }


def _overlap_telemetry(runtime) -> dict | None:
    """Best-effort GPU-overlap census off the runtime slot metrics.

    ``overlap_gpu_dispatch_ns`` (work issued while miss reads were open) and
    ``overlap_exposed_wait_ns`` (residual blocking wait the overlap could not
    hide) populate only when ``overlap_miss_reads`` is armed; return None
    otherwise so the arm never fabricates an idle figure.
    """
    metrics = getattr(getattr(runtime, "slots", None), "metrics", None)
    snap = getattr(metrics, "snapshot", None)
    data = snap() if callable(snap) else getattr(metrics, "__dict__", None)
    if not isinstance(data, dict):
        return None
    dispatch = data.get("overlap_gpu_dispatch_ns")
    exposed = data.get("overlap_exposed_wait_ns")
    if not dispatch and not exposed:
        return None
    total = (dispatch or 0) + (exposed or 0)
    return {
        "overlap_gpu_dispatch_ns": dispatch,
        "overlap_exposed_wait_ns": exposed,
        "gpu_idle_share": (exposed / total) if total else None,
    }


def _run_arm(args, arm, bench, mx) -> dict:
    _apply_arm_env(arm)
    build_prompt = bench._load_build_prompt()
    prompt_ids, prompt_meta = build_prompt(
        _tokenizer(args, bench),
        bench._prompt_args(args, args.context_tokens),
    )
    resident = _load_model(args, bench, mx)
    model = resident.model
    runtime = getattr(model, "_mtplx_expert_runtime", None)
    try:
        ops = bench._MLXOps(mx)
        mem_probe = bench._MLXMemProbe(mx)
        run = _generate(
            model=model,
            ops=ops,
            mem_probe=mem_probe,
            prompt_ids=prompt_ids,
            steps=args.decode_tokens,
        )
        ids = run["generated"]
        receipt = {
            "arm": arm,
            "overlap_env": os.environ.get(OVERLAP_ENV),
            "context_tokens": int(args.context_tokens),
            "decode_tokens": int(args.decode_tokens),
            "prompt_tokens": len(prompt_ids),
            "ttft_s": run["ttft_s"],
            "prefill_tok_s": (len(prompt_ids) / run["ttft_s"])
            if run["ttft_s"] > 0
            else None,
            "decode_wall_s": run["decode_wall_s"],
            "decode_tok_s": (args.decode_tokens / run["decode_wall_s"])
            if run["decode_wall_s"] > 0
            else None,
            "peak_gb": run["peak_gb"],
            "token_ids_sha256": hashlib.sha256(
                json.dumps(ids).encode()
            ).hexdigest(),
            "first_token_ids": ids[:16],
            "overlap_telemetry": _overlap_telemetry(runtime)
            if runtime is not None
            else None,
        }
        if args.syncs > 0:
            receipt["sync_census"] = _sync_census(
                model=model, ops=ops, mem_probe=mem_probe,
                prompt_ids=prompt_ids, steps=args.syncs,
            )
        return receipt
    finally:
        if runtime is not None:
            close = getattr(runtime, "close", None)
            if callable(close):
                close()


def _tokenizer(args, bench):
    from mlx_lm.utils import load_tokenizer

    return load_tokenizer(Path(args.model))


def _sync_census(*, model, ops, mem_probe, prompt_ids, steps) -> dict:
    """A short probe pass (route stage probe must be ENABLED via PROBE_ENV set
    before mtplx import) that counts the routing barrier per decoded token."""
    from mtplx import expert_route_probe as probe

    if not getattr(probe, "ENABLED", False):
        return {"enabled": False, "note": f"set {PROBE_ENV}=1 before launch"}
    # prefill first (routes every layer once), then snapshot and decode.
    cache = model.make_cache()
    logits = model(ops.input([list(prompt_ids)]), cache=cache)
    ops.sync(logits)
    token = ops.argmax_last(logits)
    before = int(probe._COUNTS.get(BARRIER_STAGE, 0))
    for _ in range(int(steps)):
        logits = model(ops.input([[token]]), cache=cache)
        ops.sync(logits)
        token = ops.argmax_last(logits)
    after = int(probe._COUNTS.get(BARRIER_STAGE, 0))
    barriers = after - before
    return {
        "enabled": True,
        "decode_steps": int(steps),
        "routing_barriers_total": barriers,
        "routing_barriers_per_token": (barriers / steps) if steps else None,
        "stages": probe.snapshot().get("stages", {}),
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.syncs > 0:
        # The probe reads its ENABLED flag at import, so arm it before any mtplx
        # import happens inside the arm run.
        os.environ[PROBE_ENV] = "1"
    bench = _load_bench_module()
    import mlx.core as mx

    mx.random.seed(int(args.seed))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    receipts = []
    for arm in args.arms:
        print(f"[ab] arm={arm} ctx={args.context_tokens} decode={args.decode_tokens}")
        receipt = _run_arm(args, arm, bench, mx)
        receipts.append(receipt)
        with args.out.open("a") as fh:
            fh.write(json.dumps(receipt) + "\n")
        print(
            f"[ab]   decode_tok_s={receipt['decode_tok_s']} "
            f"peak_gb={receipt['peak_gb']:.2f} sha={receipt['token_ids_sha256'][:12]}"
        )

    # Control-vs-overlap summary: byte-identity is a recorded fact, not a claim.
    if len(receipts) >= 2:
        base = receipts[0]
        for cand in receipts[1:]:
            identical = cand["token_ids_sha256"] == base["token_ids_sha256"]
            d_base = base["decode_tok_s"] or 0.0
            d_cand = cand["decode_tok_s"] or 0.0
            delta = ((d_cand - d_base) / d_base * 100.0) if d_base else None
            print(
                f"[ab] {cand['arm']} vs {base['arm']}: "
                f"byte_identical={identical} "
                f"decode_tok_s {d_base:.3f} -> {d_cand:.3f} "
                f"({'+' if (delta or 0) >= 0 else ''}{delta:.2f}%)"
                if delta is not None
                else f"[ab] {cand['arm']} vs {base['arm']}: byte_identical={identical}"
            )
            if not identical:
                print(
                    f"[ab] FAIL: {cand['arm']} changed the decoded tokens "
                    "(the lever must be a pure execution reorder)"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
