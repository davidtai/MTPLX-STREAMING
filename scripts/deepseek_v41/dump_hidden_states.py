#!/usr/bin/env python3
"""Per-layer hidden-state divergence dump for DeepSeek-V4.1-Flash streaming.

Runs a fixed prompt through the real serve-path model (one PREFILL forward) and
writes, to ``--out``, a small JSON of per-layer hidden-state summaries so the
same dump taken on CPU and on the GPU can be diffed
(:mod:`scripts.deepseek_v41.compare_hidden_states`) to find the first layer that
diverges.

What it records (all deterministic, greedy, no sampling):

  * ``embed``            -- the hc-expanded token-embedding stream entering layer 0.
  * ``layers[i]``        -- for every backbone layer i: the output residual
                            stream stats (``mean``/``std``/``max_abs``) and the
                            first 64 floats of the LAST position (flattened over
                            the hc copies), plus, on the engram layers, the
                            pre-engram and post-engram stream stats.
  * ``final_norm``       -- the collapsed + RMSNorm'd stream feeding the head.
  * ``logits_top8``      -- the final top-8 logits with ids + decoded text.
  * ``engram_gates``     -- the engram gate values (``mean``/``std``/``min``/
                            ``max``) for layers 1 and 14 (the reference
                            ``sigmoid(copysign(sqrt(clamp|dot|), dot))``).

The capture is non-invasive: it wraps the layer ``__call__`` and the two engram
hooks in the running process and restores them afterwards, so the model module
is not modified.  The dump uses the SAME loader the P1.7 gate uses
(``load_deepseek_v41_streaming``, ``slot_layout=component-banks`` by default).

CPU: pass ``--cpu`` (forces the MLX CPU stream) and ``--no-apply-memory-cap``.
GPU (orchestrator, inside the lock): omit ``--cpu``; ``--apply-memory-cap`` is on
by default.  This module does no GPU work at import; ``--help`` is CPU-safe.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MODEL = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4").expanduser()
GIB = 1024**3


def build_prompt(tokenizer, args):
    """Build the input token ids + a build-metadata dict.

    By default the deterministic ``mtplx.prefill_bench`` coding-agent programming
    prompt at ``--context-tokens`` (David's standardized input; the same builder
    the prefill ladder uses), then prepend the BOS the reference always adds.  A
    literal ``--prompt`` overrides the builder.  The artifact tokenizer has no HF
    chat template, so ``--prompt-format`` defaults to ``raw`` (the reference's own
    ``encoding.py`` chat tokens are not an HF template)."""
    if args.prompt is not None:
        build_ids = list(tokenizer.encode(args.prompt))
        meta = {
            "prompt_source": "literal",
            "prompt_text_sha256": None,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt",
        default=None,
        help="literal prompt text; overrides the prefill_bench builder. Default: "
        "None (build the deterministic prefill_bench prompt at --context-tokens).",
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=1024,
        choices=(1024, 16384),
        help="prefill_bench programming-prompt size (David's standardized input). "
        "Default 1024. 16384 is the prefill cell (heavy on CPU; GPU-run).",
    )
    parser.add_argument("--prompt-format", default="raw", choices=("raw", "chat"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--bos-id",
        type=int,
        default=0,
        help="token id prepended to the prompt (the artifact's "
        "'<｜begin▁of▁sentence｜>' == 0).  The reference always prepends it "
        "(encoding.encode_messages add_default_bos_token=True); the tokenizer's "
        "add_bos_token is False, so tokenizer.encode() omits it.  Pass --no-bos "
        "to reproduce the no-BOS input.",
    )
    parser.add_argument("--no-bos", dest="bos", action="store_false", default=True)
    parser.add_argument("--slot-layout", default="component-banks")
    parser.add_argument("--max-kv", type=int, default=4096)
    parser.add_argument("--memory-limit-gib", type=float, default=100.0)
    parser.add_argument("--expert-cache-limit-gib", type=float, default=None)
    parser.add_argument(
        "--verify-record-hashes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--no-admit", dest="admit", action="store_false", default=True)
    parser.add_argument("--admission-receipt", type=Path, default=None)
    parser.add_argument(
        "--engram-ablate",
        action="store_true",
        default=False,
        help="detach the engram hooks (engram_hook=None) before the forward, to "
        "isolate the engram contribution in the dump.",
    )
    parser.add_argument(
        "--apply-memory-cap",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _git_rev() -> str | None:
    try:
        out = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=str(Path(__file__).resolve().parent),
            check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _stats(arr):
    """(mean, std, max_abs) of an mx array, computed in fp32."""
    import mlx.core as mx

    a = arr.astype(mx.float32)
    mean = float(mx.mean(a).item())
    std = float(mx.sqrt(mx.mean((a - mean) ** 2)).item())
    max_abs = float(mx.max(mx.abs(a)).item())
    finite = bool(mx.all(mx.isfinite(a)).item())
    return {"mean": mean, "std": std, "max_abs": max_abs, "finite": finite}


def _last_pos_first64(h):
    """First 64 floats of the last sequence position, flattened over any head/hc
    axes.  ``h`` is [b, s, ...]; take [0, -1] and flatten."""
    import mlx.core as mx

    v = h[0, -1]
    v = v.reshape(-1)[:64].astype(mx.float32)
    return [float(x) for x in v.tolist()]


def _summ(h):
    d = _stats(h)
    d["first64_last_pos"] = _last_pos_first64(h)
    return d


def _engram_gate_stats(hook, hidden_states, cache_state):
    """Recompute the engram gate for one layer exactly as ``EngramV41.__call__``
    does (reference ``Engram.forward``), returning its summary.  Uses only the
    hook's public attributes, so the model module is untouched."""
    import mlx.core as mx
    import numpy as np

    row_ids = cache_state.current_row_ids(hook.layer_hash_index)
    B, L = int(hidden_states.shape[0]), int(hidden_states.shape[1])
    embed = hook.row_cache.dequantize(row_ids)
    kv = hook.wkv(embed.reshape(B, L, -1))
    split = hook.hc_mult * hook.dim
    key = kv[..., :split].astype(mx.float32).reshape(B, L, hook.hc_mult, hook.dim)
    h = hidden_states.astype(mx.float32)
    weight = (hook.q_weight * hook.k_weight).astype(mx.float32)
    eps = hook.norm_eps
    rstd = mx.rsqrt(mx.mean(h * h, axis=-1) + eps) * mx.rsqrt(mx.mean(key * key, axis=-1) + eps)
    dot = mx.sum(h * weight * key, axis=-1) * rstd * (hook.dim ** -0.5)
    mag = mx.sqrt(mx.maximum(mx.abs(dot), hook.clamp_value))
    signed = mx.where(dot < 0, -mag, mag)
    gate = mx.sigmoid(signed)
    g = gate.astype(mx.float32)
    value = kv[..., split:].astype(mx.float32)
    return {
        "gate_mean": float(mx.mean(g).item()),
        "gate_std": float(mx.sqrt(mx.mean((g - mx.mean(g)) ** 2)).item()),
        "gate_min": float(mx.min(g).item()),
        "gate_max": float(mx.max(g).item()),
        "dot_mean": float(mx.mean(dot).item()),
        "dot_max_abs": float(mx.max(mx.abs(dot)).item()),
        "value_max_abs": float(mx.max(mx.abs(value)).item()),
        "key_max_abs": float(mx.max(mx.abs(key)).item()),
        "gate_last_pos": [float(x) for x in g[0, -1].reshape(-1).tolist()],
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    import mlx.core as mx

    if args.cpu:
        mx.set_default_device(mx.cpu)
    mx.random.seed(int(args.seed))

    from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming
    from mtplx.models.deepseek_v41 import DecoderLayer
    from mlx_lm.utils import load_tokenizer

    receipt = None
    if args.admission_receipt is not None:
        receipt = json.loads(Path(args.admission_receipt).read_text())

    cache_limit = (
        None if args.expert_cache_limit_gib is None
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
        slot_layout=args.slot_layout,
        cache_scope="layer",
        island_layers=(),
        verify_record_hashes=args.verify_record_hashes,
    )
    model = resident.model
    runtime = getattr(model, "_mtplx_expert_runtime")
    try:
        tokenizer = load_tokenizer(Path(args.model))
        prompt_ids, prompt_meta = build_prompt(tokenizer, args)
        ids = mx.array([prompt_ids])

        layer_records: dict[int, dict] = {}
        engram_records: dict[int, dict] = {}

        # -- wrap the engram hooks (capture pre/post stream + gate) --------------
        original_hooks: dict[int, object] = {}
        for layer in model.model.layers:
            hook = getattr(layer, "engram_hook", None)
            if hook is None:
                continue
            original_hooks[layer.layer_id] = hook
            if args.engram_ablate:
                layer.engram_hook = None
                continue

            def make_wrapper(_hook, _lid):
                def wrapper(hidden_states, token_ids, cache_state):
                    rec = {"pre_engram": _summ(hidden_states)}
                    try:
                        rec["gate"] = _engram_gate_stats(_hook, hidden_states, cache_state)
                    except Exception as exc:  # pragma: no cover
                        rec["gate_error"] = repr(exc)
                    out = _hook(hidden_states, token_ids, cache_state)
                    rec["post_engram"] = _summ(out)
                    engram_records[_lid] = rec
                    return out
                return wrapper

            layer.engram_hook = make_wrapper(hook, layer.layer_id)

        # -- wrap DecoderLayer.__call__ (capture per-layer output + embed input) --
        orig_call = DecoderLayer.__call__
        embed_holder: dict[str, dict] = {}

        def layer_call(self, h, pre_mix, positions, layer_cache, shared):
            if self.layer_id == 0:
                embed_holder["embed"] = _summ(h)
            out, next_pre = orig_call(self, h, pre_mix, positions, layer_cache, shared)
            layer_records[self.layer_id] = _summ(out)
            return out, next_pre

        DecoderLayer.__call__ = layer_call
        try:
            cache = model.make_cache()
            normed = model.model(ids, cache=cache)  # final collapse + RMSNorm
            logits = model.head(normed.astype(mx.float32))
            mx.eval(logits, normed)
        finally:
            DecoderLayer.__call__ = orig_call
            for lid, hook in original_hooks.items():
                model.model.layers[lid].engram_hook = hook

        # -- top-8 logits --------------------------------------------------------
        last = logits[0, -1].astype(mx.float32)
        order = [int(i) for i in mx.argsort(-last).tolist()[:8]]
        top8 = [
            {"id": tid, "logit": float(last[tid].item()),
             "text": tokenizer.decode([tid])}
            for tid in order
        ]

        dump = {
            "meta": {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "git_rev": _git_rev(),
                "device": "cpu" if args.cpu else "default",
                "slot_layout": args.slot_layout,
                "spec_key": runtime.spec.key,
                "manifest_sha256": getattr(runtime.manifest, "manifest_sha256", None),
                "prompt_build": prompt_meta,
                "prompt_ids_head": prompt_ids[:16],
                "prompt_ids_tail": prompt_ids[-8:],
                "input_token_count": len(prompt_ids),
                "engram_ablate": bool(args.engram_ablate),
                "engram_layer_ids": list(getattr(model, "_mtplx_engram_layer_ids", ()) or ()),
                "n_layers": len(model.model.layers),
            },
            "embed": embed_holder.get("embed"),
            "layers": [
                {"layer": i, **layer_records[i]} for i in sorted(layer_records)
            ],
            "engram": {str(lid): engram_records[lid] for lid in sorted(engram_records)},
            "final_norm": _summ(normed),
            "argmax_token": {
                "id": order[0], "text": tokenizer.decode([order[0]]),
            },
            "logits_top8": top8,
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(dump, indent=2))
        print(f"[dump] wrote {out_path}")
        print(f"[dump] argmax={order[0]} {tokenizer.decode([order[0]])!r}  "
              f"final_norm max_abs={dump['final_norm']['max_abs']:.4g}")
        return 0
    finally:
        try:
            runtime.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
