#!/usr/bin/env python3
"""P1.7 streamed==resident argmax gate for DeepSeek-V4.1-Flash q2 streaming.

Proves that serving a routed layer's experts *resident* (pinned island) instead
of *streamed* from the on-disk Q2 bank does not change the model's greedy
(argmax) decode. It builds the model twice through the real serve-path loader
(:func:`mtplx.models.deepseek_v41_loader.load_deepseek_v41_streaming`):

  Run A (streamed):  slot_layout=component-banks, island_layers=()      -- every
                     routed layer streams its experts from experts.bin.
  Run B (resident):  slot_layout=component-banks, island_layers=<subset> -- the
                     subset's experts are held resident (DenseIslandStore); all
                     other layers still stream.

Both runs greedily decode ``--steps`` (default 32) tokens from a fixed prompt.
The gate PASSES iff the two argmax token sequences are byte-identical.

WHY A SUBSET, NOT ALL 40 LAYERS (task-mandated caveat)
------------------------------------------------------
Pinning *every* routed layer resident is impossible inside the 100 GiB wired
knob: the full Q2 expert bank is ~158 GiB (169,869,312,000 routed bytes;
40 layers x 384 experts x 11,059,200 B). Each fully-pinned layer costs ~4.15 GiB
resident, so at most ~20 layers fit alongside the 8.67 GiB text residents and the
7 GiB runtime reserve. This gate therefore pins a SUBSET per invocation
(``--pinned-layers``) and, to make the comparison total across all 40 layers,
additionally records the digest of every routed expert record actually gathered
in each run (the manifest SHA-256 of each selected ``(layer, expert)`` record).
Rotate ``--pinned-layers`` across invocations to cover all 40 layers; the
per-run gathered-record digests let a reviewer confirm the same experts were
exercised regardless of where they were served from.

The isolation is deliberate: BOTH runs use the component-banks slot layout, so
residency is the ONLY difference between them (not the dispatch path). Islands
require component-banks and ``cache_scope='layer'`` (mtplx/expert_runtime.py:371).

ASSUMPTIONS / PRECONDITIONS (run later by the orchestrator, on the GPU, inside
gpu_window.sh which holds the exclusive lock -- NOT by the harness author):
  * Worker W1's ``mtplx/models/deepseek_v41.py`` (Model, ModelArgs) exists; until
    then the loader raises ResidentLoadError and this gate reports that cleanly.
  * The model follows the mlx_lm calling convention ``model(ids[None], cache=c)``
    -> logits ``[B, T, vocab]``; if it rejects ``cache=`` we fall back to a
    full-sequence re-feed and note it in the receipt.
  * The local expert-manifest carries the HF identity so admission passes
    (W3 rebased it; see docs/deepseek-v41/W3_REPORT.md). ``--no-admit`` +
    ``--admission-receipt`` are available if admission must be injected.
  * Expert selections are captured by wrapping each bound ``switch_mlp`` (both
    HotExpertSwitchGLU and DenseIslandSwitchGLU are called as
    ``switch_mlp(x, indices)``), so gathered-record digests are available for
    pinned and streamed layers alike.

A receipt JSON is written append-only under
``.benchmark-artifacts/deepseek-v41/<utc-stamp>/`` (never overwritten): git rev,
spec key, manifest sha, cache limits, both argmax sequences, match/mismatch, the
pinned-layer subset per run, and the gathered expert-record digests per run.

This module performs no GPU work at import time; ``--help`` and import are safe
on CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MODEL = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4").expanduser()
DEFAULT_OUT_DIR = Path(".benchmark-artifacts/deepseek-v41")
GIB = 1024**3


def build_prompt(tokenizer, args):
    """Input token ids + build-metadata for the gate.

    Default: the deterministic ``mtplx.prefill_bench`` coding-agent programming
    prompt at ``--context-tokens`` (David's standardized benchmark input), with
    the reference's leading BOS prepended.  A literal ``--prompt`` overrides the
    builder.  ``--prompt-format`` defaults to ``raw`` because the artifact
    tokenizer ships no HF chat template."""
    if args.prompt is not None:
        build_ids = list(tokenizer.encode(args.prompt))
        meta = {
            "prompt_source": "literal",
            "prompt_context_tokens": len(build_ids),
            "prompt_actual_tokens": len(build_ids),
            "prompt_style": "literal",
            "prompt_format": "raw",
            "prompt_release_valid": False,
        }
    else:
        from mtplx.prefill_bench import _prompt_build_for_context

        pb = _prompt_build_for_context(
            tokenizer, int(args.context_tokens), prompt_format=args.prompt_format
        )
        build_ids = list(pb.token_ids)
        meta = dict(pb.metadata)
        meta["prompt_source"] = "prefill_bench"
    prompt_ids = list(build_ids)
    if args.bos:
        prompt_ids = [int(args.bos_id)] + prompt_ids
    meta["bos_prepended"] = bool(args.bos)
    meta["bos_id"] = int(args.bos_id) if args.bos else None
    meta["input_tokens"] = len(prompt_ids)
    meta["tokenizer"] = str(args.model)
    return prompt_ids, meta


def _parse_layers(text: str) -> tuple[int, ...]:
    if not text.strip():
        return ()
    layers = tuple(sorted({int(part) for part in text.split(",") if part.strip()}))
    for layer in layers:
        if layer < 0:
            raise argparse.ArgumentTypeError("layer indices must be >= 0")
    return layers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt",
        default=None,
        help="literal prompt; overrides the prefill_bench builder (default None: "
        "build the deterministic prefill_bench prompt at --context-tokens).",
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=1024,
        choices=(1024, 16384),
        help="prefill_bench programming-prompt size (David's standardized input). "
        "Default 1024; 16384 is the prefill cell.",
    )
    parser.add_argument("--prompt-format", default="raw", choices=("raw", "chat"))
    parser.add_argument(
        "--bos", action=argparse.BooleanOptionalAction, default=True,
        help="prepend the reference BOS (id 0); the model degenerates without it.",
    )
    parser.add_argument("--bos-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument(
        "--pinned-layers",
        type=_parse_layers,
        default=(20,),
        metavar="L1,L2,...",
        help="routed layers to pin resident in Run B (default: 20). Rotate across "
        "invocations to cover all 40 layers; must fit within the memory limit.",
    )
    parser.add_argument("--slot-layout", default="component-banks")
    parser.add_argument("--max-kv", type=int, default=4096)
    parser.add_argument("--memory-limit-gib", type=float, default=100.0)
    parser.add_argument(
        "--expert-cache-limit-gib",
        type=float,
        default=None,
        help="explicit expert cache byte budget (default: runtime plans it)",
    )
    parser.add_argument(
        "--verify-record-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-admit", dest="admit", action="store_false", default=True)
    parser.add_argument("--admission-receipt", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--apply-memory-cap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply the reconciled MLX memory cap before allocation (default: "
        "on for the GPU window; pass --no-apply-memory-cap for a CPU-only "
        "reproduction that must not touch the MLX/Metal memory cap).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="force the MLX default device to the CPU stream (no Metal). Used "
        "to run the gate as a CPU reproduction/regression without the GPU lock; "
        "the real proof runs on the GPU with this off.",
    )
    return parser


def _git_rev(cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=str(cwd),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    rev = out.stdout.strip()
    return rev or None


def _record_digest_index(manifest) -> dict[tuple[int, int], str | None]:
    return {(r.layer, r.expert): r.sha256 for r in manifest.records}


class _CaptureSwitch:
    """Plain callable proxy that records selected experts, then forwards.

    The model calls ``switch_mlp(x, indices)``; both the streamed
    (HotExpertSwitchGLU) and pinned (DenseIslandSwitchGLU) dispatchers share that
    signature, so this proxy captures the routed expert ids for either.
    """

    def __init__(self, inner, layer_index: int, sink: list[tuple[int, int, int]], step_ref: list[int]):
        self._inner = inner
        self._layer_index = int(layer_index)
        self._sink = sink
        self._step_ref = step_ref

    def __call__(self, x, indices):
        try:
            import mlx.core as mx

            ids = mx.array(indices).reshape(-1).tolist()
            step = self._step_ref[0]
            for expert in sorted({int(v) for v in ids}):
                self._sink.append((step, self._layer_index, expert))
        except Exception:
            pass
        return self._inner(x, indices)

    def __getattr__(self, name):  # forward any attribute the model reads
        return getattr(self._inner, name)


def _wrap_switches(model, sink: list[tuple[int, int, int]], step_ref: list[int]) -> int:
    inner = getattr(getattr(model, "model", None), "layers", None)
    if inner is None:
        inner = getattr(model, "layers", None)
    if inner is None:
        raise TypeError("model does not expose transformer layers")
    wrapped = 0
    for layer in inner:
        mlp = getattr(layer, "mlp", None)
        switch = getattr(mlp, "switch_mlp", None) if mlp is not None else None
        if switch is None:
            continue
        layer_index = int(getattr(switch, "layer_index", wrapped))
        mlp.switch_mlp = _CaptureSwitch(switch, layer_index, sink, step_ref)
        wrapped += 1
    return wrapped


def _greedy_decode(model, prompt_ids, steps: int, step_ref: list[int]):
    """Greedy-decode ``steps`` tokens; returns (argmax_ids, used_cache: bool)."""

    import mlx.core as mx

    try:
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(model)
    except Exception:
        cache = None

    def forward(ids_2d):
        if cache is not None:
            return model(ids_2d, cache=cache)
        return model(ids_2d)

    argmax: list[int] = []
    used_cache = cache is not None
    context = list(prompt_ids)
    step_ref[0] = 0
    try:
        logits = forward(mx.array([context]))
    except TypeError:
        # model rejects cache=; fall back to cacheless full re-feed.
        used_cache = False
        cache = None
        logits = model(mx.array([context]))
    token = int(mx.argmax(logits[0, -1]).item())
    argmax.append(token)
    for step in range(1, steps):
        step_ref[0] = step
        if used_cache:
            logits = forward(mx.array([[token]]))
        else:
            context.append(token)
            logits = model(mx.array([context]))
        token = int(mx.argmax(logits[0, -1]).item())
        argmax.append(token)
    return argmax, used_cache


def _run_once(args, pinned_layers, digest_index_holder: dict):
    """Build the model with the given island subset, decode, return a run record."""

    from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming

    receipt = None
    if args.admission_receipt is not None:
        receipt = json.loads(Path(args.admission_receipt).read_text())

    overrides = dict(
        slot_layout=args.slot_layout,
        cache_scope="layer",
        island_layers=tuple(pinned_layers),
        verify_record_hashes=args.verify_record_hashes,
    )
    cache_limit = (
        None
        if args.expert_cache_limit_gib is None
        else int(args.expert_cache_limit_gib * GIB)
    )
    resident = load_deepseek_v41_streaming(
        args.model,
        memory_limit_bytes=int(args.memory_limit_gib * GIB),
        max_live_kv_tokens=int(args.max_kv),
        admit=args.admit,
        admission_receipt=receipt,
        expert_cache_limit_bytes=cache_limit,
        apply_memory_cap=args.apply_memory_cap,
        **overrides,
    )
    model = resident.model
    runtime = getattr(model, "_mtplx_expert_runtime")
    try:
        if not digest_index_holder:
            digest_index_holder.update(_record_digest_index(runtime.manifest))
            digest_index_holder["__spec_key__"] = runtime.spec.key
            digest_index_holder["__manifest_sha__"] = getattr(
                runtime.manifest, "manifest_sha256", None
            )
            digest_index_holder["__memory_limit__"] = runtime.config.memory_limit_bytes
            digest_index_holder["__expert_cache_limit__"] = (
                runtime.config.expert_cache_limit_bytes
            )

        sink: list[tuple[int, int, int]] = []
        step_ref = [0]
        wrapped = _wrap_switches(model, sink, step_ref)

        from mlx_lm.utils import load_tokenizer

        tokenizer = load_tokenizer(Path(args.model))
        prompt_ids, prompt_meta = build_prompt(tokenizer, args)
        if "__prompt_build__" not in digest_index_holder:
            digest_index_holder["__prompt_build__"] = prompt_meta

        argmax, used_cache = _greedy_decode(model, prompt_ids, args.steps, step_ref)

        gathered = []
        index = {k: v for k, v in digest_index_holder.items() if isinstance(k, tuple)}
        for step, layer, expert in sink:
            gathered.append(
                {
                    "step": step,
                    "layer": layer,
                    "expert": expert,
                    "record_sha256": index.get((layer, expert)),
                }
            )
        return {
            "pinned_layers": list(pinned_layers),
            "prompt_token_count": len(prompt_ids),
            "argmax_tokens": argmax,
            "gathered_records": gathered,
            "gathered_record_count": len(gathered),
            "switches_wrapped": wrapped,
            "used_kv_cache": used_cache,
        }
    finally:
        try:
            runtime.close()
        except Exception:
            pass


def _out_dir(base: Path) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    candidate = base / stamp
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = base / f"{stamp}-{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # Heavy imports are deferred to here so --help/import stay CPU-safe.
    try:
        import mlx.core as mx

        if args.cpu:
            # Route every MLX op through the CPU stream (no Metal). Set before
            # any array is built so the whole build+decode stays off the GPU.
            mx.set_default_device(mx.cpu)
        mx.random.seed(int(args.seed))
    except Exception as exc:  # pragma: no cover - environment guard
        print(f"gate_stream_equals_resident: MLX is required to run: {exc}", file=sys.stderr)
        return 2

    if not args.pinned_layers:
        print(
            "gate_stream_equals_resident: --pinned-layers must name >=1 layer for Run B",
            file=sys.stderr,
        )
        return 2

    worktree = Path(__file__).resolve().parents[2]
    digest_index_holder: dict = {}

    try:
        print(f"[gate] Run A (streamed, island_layers=()) on {args.model}", flush=True)
        run_streamed = _run_once(args, (), digest_index_holder)
        print(
            f"[gate] Run B (resident, island_layers={list(args.pinned_layers)})",
            flush=True,
        )
        run_resident = _run_once(args, args.pinned_layers, digest_index_holder)
    except Exception as exc:  # loader/admission/model not ready, OOM, etc.
        # Always surface the full traceback first: the one-line summary below
        # otherwise swallows the actual failure site (a raw AttributeError deep
        # in the streamed dispatch is indistinguishable from a missing-W1
        # ResidentLoadError without it).
        import traceback

        traceback.print_exc(file=sys.stderr)
        print(
            "gate_stream_equals_resident: could not complete a run "
            f"({type(exc).__name__}: {exc}). If this is a ResidentLoadError, "
            "worker W1's mtplx/models/deepseek_v41.py may not be available yet, "
            "or admission failed on the manifest identity (see W3_REPORT.md).",
            file=sys.stderr,
        )
        return 3

    match = run_streamed["argmax_tokens"] == run_resident["argmax_tokens"]

    receipt = {
        "gate": "streamed_equals_resident",
        "port_plan_ref": "P1.7",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_rev": _git_rev(worktree),
        "model_path": str(args.model),
        "spec_key": digest_index_holder.get("__spec_key__"),
        "manifest_sha256": digest_index_holder.get("__manifest_sha__"),
        "memory_limit_bytes": digest_index_holder.get("__memory_limit__"),
        "expert_cache_limit_bytes": digest_index_holder.get("__expert_cache_limit__"),
        "slot_layout": args.slot_layout,
        "prompt": args.prompt,
        "prompt_build": digest_index_holder.get("__prompt_build__"),
        "steps": args.steps,
        "seed": args.seed,
        "pinned_layers_run_b": list(args.pinned_layers),
        "match": match,
        "verdict": "PASS" if match else "MISMATCH",
        "runs": {
            "streamed": run_streamed,
            "resident": run_resident,
        },
        "caveat": (
            "Full 40-layer pinning does not fit in 100 GiB (158 GiB bank); Run B "
            "pins a subset. Rotate --pinned-layers across invocations for full "
            "coverage; gathered_records lists the digest of every routed expert "
            "record actually used in each run."
        ),
    }

    out_dir = _out_dir(args.out_dir)
    receipt_path = out_dir / "gate_stream_equals_resident.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))

    print(f"[gate] verdict={receipt['verdict']} match={match}", flush=True)
    print(f"[gate] receipt -> {receipt_path}", flush=True)
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
