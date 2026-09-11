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
DEFAULT_BOS_ID = 0
OVERLAP_ENV = "MTPLX_DSV41_SHARED_OVERLAP"
PROBE_ENV = "MTPLX_ROUTE_STAGE_PROBE"
STAGE_TIMING_ENV = "MTPLX_DSV41_STAGE_TIMING"
BARRIER_STAGE = "hot.eval_indices"

LAYER_MAJOR_ENV = "MTPLX_DSV41_PREFILL_LAYER_MAJOR"
SINKHORN_METAL_ENV = "MTPLX_DSV41_SINKHORN_METAL"   # K3, merged @ 8982b93c9
HC_COMPILE_ENV = "MTPLX_DSV41_HC_COMPILE"           # K4, landing

# Every lever env key, in a stable order. Each preset names ALL of them (None =
# force-unset) so applying an arm fully determines the four flags regardless of
# what a prior arm in the same process left set -- the arms are independent.
ALL_LEVER_ENVS = (OVERLAP_ENV, LAYER_MAJOR_ENV, SINKHORN_METAL_ENV, HC_COMPILE_ENV)


def _preset(*, overlap=None, layer_major=None, sinkhorn=None, hc=None) -> dict:
    """A preset that pins ALL four lever keys (None = force-unset)."""
    return {
        OVERLAP_ENV: overlap,
        LAYER_MAJOR_ENV: layer_major,
        SINKHORN_METAL_ENV: sinkhorn,
        HC_COMPILE_ENV: hc,
    }


ARM_PRESETS = {
    "control": _preset(),                                    # shipped: all levers OFF
    "shared_overlap": _preset(overlap="1"),                 # W28 K1 lever ON
    "layer_major": _preset(layer_major="1"),                # W30 K16 lever ON
    "sinkhorn_metal": _preset(sinkhorn="1"),                # K3 lever ON (8982b93c9)
    "hc_compile": _preset(hc="1"),                          # K4 lever ON (landing)
    "both": _preset(overlap="1", layer_major="1"),          # shared_overlap + layer_major
    "all_levers": _preset(overlap="1", layer_major="1", sinkhorn="1", hc="1"),
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
        help="preset names from ARM_PRESETS (control, shared_overlap, layer_major, "
        "sinkhorn_metal, hc_compile, both, all_levers)",
    )
    p.add_argument("--out", type=Path, required=True, help="append-only JSONL receipt")
    # Prompt build: mirrors bench_standard_shape.py exactly, so that
    # ``bench._prompt_args(args, ctx)`` reads the same fields with the same
    # defaults and the reused ``build_prompt`` produces the identical standard-
    # shape prompt (deterministic prefill_bench coding-agent prompt + reference
    # BOS). Missing these was the W11 crash (_prompt_args hit args.prompt).
    p.add_argument(
        "--prompt",
        default=None,
        help="literal prompt text; overrides the prefill_bench builder "
        "(default None = build the deterministic prefill_bench prompt).",
    )
    p.add_argument("--prompt-format", default="raw", choices=("raw", "chat"))
    p.add_argument(
        "--bos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="prepend the reference BOS (id --bos-id, default 0). The artifact "
        "tokenizer does not add it; the reference always does (W8_REPORT.md).",
    )
    p.add_argument("--bos-id", type=int, default=DEFAULT_BOS_ID)
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="CPU-only test double: apply each arm's env and build the standard-"
        "shape prompt with bench's fake tokenizer -- no model load, no MLX/Metal. "
        "Proves argument resolution, prompt build and per-arm env application.",
    )
    p.add_argument(
        "--syncs",
        type=int,
        default=0,
        metavar="N",
        help="extra probe pass of N decode steps per arm with the route-stage "
        "probe ON, to census host syncs/token (0 = skip; the probe inflates "
        "timing, so it never touches the reported tok/s pass)",
    )
    p.add_argument(
        "--stage-timing",
        action="store_true",
        default=False,
        help="extra W37 decode pass with MTPLX_DSV41_STAGE_TIMING fences per "
        "stage (embed / attention-by-CSA-mode / engram / HC / gate+barrier / "
        "routed switch / shared / combine / head / sample), recorded into the "
        "receipt as ``stage_timing`` (mean ms/token per stage + counts). The "
        "fences inflate absolute time, so this pass never touches the reported "
        "tok/s; the ratios between stages are the signal.",
    )
    p.add_argument(
        "--stage-timing-steps",
        type=int,
        default=None,
        metavar="N",
        help="decode steps for the --stage-timing pass (default: --decode-tokens).",
    )
    p.add_argument(
        "--warm-repeat",
        action="store_true",
        default=False,
        help="after the measured cold pass, re-run prefill+decode of the SAME "
        "prompt a second time in the same process (expert cache + engram warm, "
        "fresh KV / engram history via model.make_cache()) and record the second "
        "pass's decode/prefill tok/s + TTFT as ``warm_*``; bounds the no-miss "
        "ceiling. Token ids must match the cold pass (recorded, not asserted).",
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


def _arm_env_snapshot() -> dict:
    """The four lever env keys' current values (None = unset)."""
    return {key: os.environ.get(key) for key in ALL_LEVER_ENVS}


def _dry_run_arm(args, arm, bench) -> dict:
    """CPU-only arm double (no model, no MLX/Metal): apply the arm's env, then
    build the standard-shape prompt with bench's fake tokenizer so the prompt-
    build metadata is byte-for-byte identical to bench_standard_shape's
    ``--dry-run`` for the same context cell."""
    build_prompt = bench._load_build_prompt()
    tokenizer = bench._FakeTokenizer()
    prompt_ids, prompt_meta = build_prompt(
        tokenizer, bench._prompt_args(args, args.context_tokens)
    )
    return {
        "arm": arm,
        "dry_run": True,
        "overlap_env": os.environ.get(OVERLAP_ENV),
        "arm_env": _arm_env_snapshot(),
        "context_tokens": int(args.context_tokens),
        "decode_tokens": int(args.decode_tokens),
        "prompt_tokens": len(prompt_ids),
        "prompt_build": prompt_meta,
        # W37 pass toggles resolved offline (no model / MLX): proves the flags
        # thread through argument resolution before a GPU window burns on them.
        "stage_timing": bool(getattr(args, "stage_timing", False)),
        "stage_timing_steps": (
            int(args.stage_timing_steps)
            if getattr(args, "stage_timing_steps", None) is not None
            else int(args.decode_tokens)
        ),
        "warm_repeat": bool(getattr(args, "warm_repeat", False)),
    }


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
    if getattr(args, "dry_run", False):
        return _dry_run_arm(args, arm, bench)
    build_prompt = bench._load_build_prompt()
    prompt_ids, prompt_meta = build_prompt(
        _tokenizer(args, bench),
        bench._prompt_args(args, args.context_tokens),
    )
    resident = _load_model(args, bench, mx)
    model = resident.model
    runtime = getattr(model, "_mtplx_expert_runtime", None)
    # W38/K3 Sinkhorn engagement: zero the module counters after model load so the
    # receipt reports THIS arm's kernel-vs-recurrence Sinkhorn split. A truthy
    # ``MTPLX_DSV41_SINKHORN_METAL`` arm whose ``kernel_calls`` is 0 means the
    # Metal kernel silently fell back to the recurrence (which would still be
    # byte-identical and ~as fast) -- exactly the "did it actually run?" question.
    # Read cumulatively over the whole arm because under ``MTPLX_DSV41_HC_COMPILE``
    # the Python wrapper runs only during the (cold) trace, not per warm token.
    try:
        from mtplx.models import deepseek_v41 as _dsv41
        _dsv41._reset_sinkhorn_kernel_calls()
    except Exception:  # pragma: no cover - defensive
        _dsv41 = None
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
            "arm_env": _arm_env_snapshot(),
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
        if getattr(args, "warm_repeat", False):
            receipt["warm"] = _warm_repeat_pass(
                model=model, ops=ops, mem_probe=mem_probe,
                prompt_ids=prompt_ids, steps=args.decode_tokens,
                cold_ids=ids,
            )
        if getattr(args, "stage_timing", False):
            steps = (
                int(args.stage_timing_steps)
                if args.stage_timing_steps is not None
                else int(args.decode_tokens)
            )
            receipt["stage_timing"] = _stage_timing_pass(
                model=model, ops=ops, prompt_ids=prompt_ids, steps=steps,
            )
        if args.syncs > 0:
            receipt["sync_census"] = _sync_census(
                model=model, ops=ops, mem_probe=mem_probe,
                prompt_ids=prompt_ids, steps=args.syncs,
            )
        if _dsv41 is not None:
            eng = _dsv41._sinkhorn_kernel_calls()
            receipt["sinkhorn_engagement"] = {
                "kernel_calls": eng["kernel"],
                "recurrence_calls": eng["recurrence"],
                # >0 kernel with 0 recurrence == the Metal Sinkhorn actually ran;
                # 0 kernel with >0 recurrence == it fell back to the recurrence.
                "engaged": eng["kernel"] > 0 and eng["recurrence"] == 0,
                "sinkhorn_metal_env": os.environ.get(SINKHORN_METAL_ENV),
                "hc_compile_env": os.environ.get(HC_COMPILE_ENV),
                "note": "cumulative over this arm (prefill + decode + census)",
            }
        return receipt
    finally:
        if runtime is not None:
            close = getattr(runtime, "close", None)
            if callable(close):
                close()


def _tokenizer(args, bench):
    from mlx_lm.utils import load_tokenizer

    return load_tokenizer(Path(args.model))


def _warm_repeat_pass(*, model, ops, mem_probe, prompt_ids, steps, cold_ids) -> dict:
    """Second prefill+decode of the SAME prompt in the same process.

    A fresh ``model.make_cache()`` resets the KV window and hands a fresh engram
    history clone, but the expert-bank LRU and any engram row cache stay warm from
    the cold pass, so this pass bounds the no-miss decode ceiling.  Greedy decode
    is deterministic, so the warm token ids must match the cold pass -- recorded
    (``token_ids_match`` + both sha256), never asserted, so a mismatch is reported
    instead of crashing the arm."""
    run = _generate(
        model=model, ops=ops, mem_probe=mem_probe,
        prompt_ids=prompt_ids, steps=steps,
    )
    warm_ids = run["generated"]
    cold = [int(t) for t in cold_ids]
    warm_sha = hashlib.sha256(json.dumps(warm_ids).encode()).hexdigest()
    cold_sha = hashlib.sha256(json.dumps(cold).encode()).hexdigest()
    return {
        "warm_ttft_s": run["ttft_s"],
        "warm_prefill_tok_s": (len(prompt_ids) / run["ttft_s"])
        if run["ttft_s"] > 0
        else None,
        "warm_decode_wall_s": run["decode_wall_s"],
        "warm_decode_tok_s": (steps / run["decode_wall_s"])
        if run["decode_wall_s"] > 0
        else None,
        "warm_peak_gb": run["peak_gb"],
        "warm_token_ids_sha256": warm_sha,
        "cold_token_ids_sha256": cold_sha,
        "token_ids_match": warm_ids == cold,
    }


def _stage_timing_pass(*, model, ops, prompt_ids, steps) -> dict:
    """W37 fenced decode pass -> ``model.stage_timing_report()``.

    Prefill once (probe unarmed -> untouched), then arm the stage-timing session
    around the decode loop only, wrapping each step in a ``frame`` and the
    argmax/host round-trip in a ``sample`` stage.  The route-stage probe counters
    (when ``MTPLX_ROUTE_STAGE_PROBE`` is armed) are cleared before the loop so the
    merged ``route_stage`` census reflects this window, not the prefill + prior
    passes.  Fences inflate absolute time, so nothing here feeds the reported
    tok/s."""
    from mtplx.models import deepseek_v41_stage_timing as stime

    cache = model.make_cache()
    logits = model(ops.input([list(prompt_ids)]), cache=cache)
    ops.sync(logits)
    token = ops.argmax_last(logits)
    # Reset the route probe window (best-effort; its snapshot is cumulative).
    try:
        from mtplx import expert_route_probe as route_probe

        if getattr(route_probe, "ENABLED", False):
            route_probe._SUMS.clear()
            route_probe._COUNTS.clear()
    except Exception:
        pass
    stime.begin()
    for _ in range(int(steps)):
        with stime.frame():
            logits = model(ops.input([[token]]), cache=cache)
            with stime.stage("sample"):
                token = ops.argmax_last(logits)
    report = model.stage_timing_report()
    stime.end()
    return report if report is not None else {"enabled": False}


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
    # W38/K3: also delta the Sinkhorn route stages so the census shows whether the
    # Metal kernel or the recurrence carried this decode (eager path only -- under
    # a warm HC-compile tape the Python wrapper does not re-run, so this delta is 0
    # and the arm-level ``sinkhorn_engagement`` counter is the authoritative read).
    sk_before = int(probe._COUNTS.get("hc.sinkhorn_kernel", 0))
    rec_before = int(probe._COUNTS.get("hc.sinkhorn_recurrence", 0))
    for _ in range(int(steps)):
        logits = model(ops.input([[token]]), cache=cache)
        ops.sync(logits)
        token = ops.argmax_last(logits)
    after = int(probe._COUNTS.get(BARRIER_STAGE, 0))
    barriers = after - before
    sk = int(probe._COUNTS.get("hc.sinkhorn_kernel", 0)) - sk_before
    rec = int(probe._COUNTS.get("hc.sinkhorn_recurrence", 0)) - rec_before
    return {
        "enabled": True,
        "decode_steps": int(steps),
        "routing_barriers_total": barriers,
        "routing_barriers_per_token": (barriers / steps) if steps else None,
        "sinkhorn_kernel_calls_decode": sk,
        "sinkhorn_recurrence_calls_decode": rec,
        "stages": probe.snapshot().get("stages", {}),
    }


def _run_dry(args, bench) -> int:
    """CPU-only path: run every requested arm through the dry-run double.

    No MLX/Metal import, no model load; writes the same append-only JSONL
    receipt (with ``prompt_build`` + ``arm_env`` per arm) so the harness'
    argument resolution, prompt build and per-arm env application are all
    exercised offline.
    """
    for arm in args.arms:
        print(
            f"[ab] DRY-RUN arm={arm} ctx={args.context_tokens} "
            f"decode={args.decode_tokens}"
        )
        receipt = _run_arm(args, arm, bench, mx=None)
        with args.out.open("a") as fh:
            fh.write(json.dumps(receipt) + "\n")
        print(
            f"[ab]   prompt_tokens={receipt['prompt_tokens']} "
            f"arm_env={receipt['arm_env']}"
        )
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    bench = _load_bench_module()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        return _run_dry(args, bench)

    if args.syncs > 0 or args.stage_timing:
        # The route-stage probe reads its ENABLED flag at import, so arm it before
        # any mtplx import happens inside the arm run.  --stage-timing arms it too,
        # so model.stage_timing_report() can merge the switch-internal breakdown
        # (hot.eval_indices barrier / miss-I/O / gather) under ``route_stage``.
        os.environ[PROBE_ENV] = "1"
    if args.stage_timing:
        # Advisory marker in the receipt env snapshot; the fenced session is armed
        # in-process by deepseek_v41_stage_timing.begin(), not by this env.
        os.environ[STAGE_TIMING_ENV] = "1"
    import mlx.core as mx

    mx.random.seed(int(args.seed))

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
