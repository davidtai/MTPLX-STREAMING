#!/usr/bin/env python3
"""W24 A/B harness for DeepSeek-V4.1-Flash decode-time streaming levers.

The orchestrator runs this INSIDE ``scripts/deepseek_v41/gpu_window.sh`` (holding
the GPU flock, Qwen unloaded, memory-guarded).  One lever per arm; ``control`` is
the shipped path (every lever OFF), byte-for-byte.  Each arm loads the model with
its ``ExpertStreamingConfig`` overrides, runs the standardized ``mtplx.prefill_bench``
prompt (default 1,024 context tokens, BOS) + ``--decode-tokens`` (default 256)
greedy tokens, and appends to an append-only receipt:

  * prefill tok/s, TTFT, decode tok/s, wall;
  * realized decode SSD bandwidth = (decode-phase bytes off SSD) / (decode wall)
    -- **Gate D** (ledger §6); prefill/decode split from the runtime IO telemetry;
  * cache hit rate, bytes read (real SSD bytes, not planned-logical);
  * peak GB (mx.get_peak_memory);
  * the decoded token ids + their sha256 -- so control-vs-candidate byte-identity
    is a recorded fact, not a claim (levers change WHEN/HOW/WHICH-cached, never the
    argmax; a differing sha fails the arm).

Levers (all default OFF; ``ExpertStreamingConfig`` fields threaded through
``load_deepseek_v41_streaming(**config_overrides)``):

  control                         shipped defaults
  overlap_miss_reads=1            batch a decode layer's <=6 miss reads into one
                                  coalesced/concurrent read set (Factor D / R4)
  io_read_fanout=4                split each 18.80 MB record read into N concurrent
                                  chunk sub-reads (Factor D / R4; W24 new lever)
  max_inflight_io_bytes=8G        widen the expert-IO threadpool = queue depth (R4)
  max_read_chunk_bytes=32M        one preadv per record instead of 3x 8 MiB (R4)
  cache_scope=global              global LRU pool instead of per-layer banks (R3)

Arm syntax: ``name`` for a registered preset, or generic ``key=value`` (value
accepts G/M/K suffixes for byte counts, and 1/0/true/false for bools).  Multiple
``key=value`` in one arm are ANDed with ``+`` (e.g. ``overlap_miss_reads=1+io_read_fanout=4``).

No GPU work at import; ``--help`` is CPU-safe.  Pass ``--cpu`` for a tiny CPU
smoke test (does not measure real bandwidth).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

DEFAULT_MODEL = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4").expanduser()
GIB = 1024 ** 3


def _parse_scalar(v: str):
    s = v.strip()
    low = s.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    mult = 1
    if s and s[-1] in "kKmMgG":
        mult = {"k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}[s[-1].lower()]
        s = s[:-1]
    try:
        return int(s) * mult
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return v  # bare string (e.g. cache_scope=global)


# Registered presets -> ExpertStreamingConfig overrides.
ARM_PRESETS = {
    "control": {},
    "overlap": {"overlap_miss_reads": True},
    "fanout4": {"io_read_fanout": 4},
    "fanout8": {"io_read_fanout": 8},
    "inflight8g": {"max_inflight_io_bytes": 8 * GIB},
    "chunk32m": {"max_read_chunk_bytes": 32 * 1024 * 1024},
    "scope_global": {"cache_scope": "global"},
    # the stacked Factor-D arm (issue concurrently + split each record):
    "overlap_fanout4": {"overlap_miss_reads": True, "io_read_fanout": 4},
}


def parse_arm(arm: str) -> tuple[str, dict]:
    """Return (arm_name, config_overrides) for a preset name or key=value[+...]."""
    if arm in ARM_PRESETS:
        return arm, dict(ARM_PRESETS[arm])
    overrides: dict = {}
    for kv in arm.split("+"):
        if "=" not in kv:
            raise ValueError(f"arm {arm!r}: {kv!r} is not a preset or key=value")
        k, v = kv.split("=", 1)
        overrides[k.strip()] = _parse_scalar(v)
    return arm, overrides


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--context-tokens", type=int, default=1024, choices=(1024, 16384))
    p.add_argument("--decode-tokens", type=int, default=256)
    p.add_argument("--arms", nargs="+", default=["control"],
                   help="preset names or key=value[+key=value] override specs")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prompt-format", default="raw", choices=("raw", "chat"))
    p.add_argument("--bos-id", type=int, default=0)
    p.add_argument("--no-bos", dest="bos", action="store_false", default=True)
    p.add_argument("--slot-layout", default="component-banks")
    # GPU window defaults (agent booted out -> ~82 GiB planner); --cpu smoke uses
    # a small plan and apply_memory_cap off.
    p.add_argument("--memory-limit-gib", type=float, default=82.0)
    p.add_argument("--expert-cache-limit-gib", type=float, default=None)
    p.add_argument("--apply-memory-cap", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cpu", action="store_true", default=False)
    p.add_argument("--max-kv", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    return p


def _io_read_bytes(runtime):
    try:
        snap = runtime.snapshot()
        return int(((snap.get("slots") or {}).get("io") or {}).get("read_bytes") or 0)
    except Exception:
        return None


def _peak_gib(mx) -> float | None:
    for fn in ("get_peak_memory",):
        f = getattr(mx, fn, None)
        if callable(f):
            try:
                return float(f()) / GIB
            except Exception:
                pass
    try:
        return float(mx.metal.get_peak_memory()) / GIB
    except Exception:
        return None


def run_arm(args, arm_name: str, overrides: dict, log) -> dict:
    import mlx.core as mx
    if args.cpu:
        mx.set_default_device(mx.cpu)
    mx.random.seed(int(args.seed))
    try:
        mx.reset_peak_memory()
    except Exception:
        pass

    from mtplx.models.deepseek_v41_loader import load_deepseek_v41_streaming
    from mlx_lm.utils import load_tokenizer
    from mlx_lm.models.cache import make_prompt_cache

    cache_limit = (None if args.expert_cache_limit_gib is None
                   else int(args.expert_cache_limit_gib * GIB))
    t_load = time.time()
    resident = load_deepseek_v41_streaming(
        args.model,
        memory_limit_bytes=int(args.memory_limit_gib * GIB),
        max_live_kv_tokens=int(args.max_kv),
        admit=True,
        admission_receipt=None,
        expert_cache_limit_bytes=cache_limit,
        apply_memory_cap=args.apply_memory_cap,
        slot_layout=args.slot_layout,
        cache_scope=overrides.pop("cache_scope", "layer"),
        island_layers=(),
        verify_record_hashes=False,
        **overrides,
    )
    model = resident.model
    runtime = getattr(model, "_mtplx_expert_runtime")
    load_s = time.time() - t_load
    log(f"[{arm_name}] loaded in {load_s:.1f}s")

    try:
        tokenizer = load_tokenizer(Path(args.model))
        from mtplx.prefill_bench import _prompt_build_for_context
        pb = _prompt_build_for_context(tokenizer, int(args.context_tokens),
                                       prompt_format=args.prompt_format)
        prompt_ids = list(pb.token_ids)
        if args.bos:
            prompt_ids = [int(args.bos_id)] + prompt_ids

        cache = make_prompt_cache(model)
        # prefill
        tp = time.time()
        logits = model(mx.array([prompt_ids]), cache=cache)
        mx.eval(logits)
        prefill_s = time.time() - tp
        ttft_s = prefill_s
        token = int(mx.argmax(logits[0, -1]).item())
        decoded = [token]
        io_prefill = _io_read_bytes(runtime)
        # decode
        td = time.time()
        for _ in range(int(args.decode_tokens) - 1):
            logits = model(mx.array([[token]]), cache=cache)
            token = int(mx.argmax(logits[0, -1]).item())
            decoded.append(token)
        decode_s = time.time() - td
        io_total = _io_read_bytes(runtime)

        decode_io = (None if io_total is None or io_prefill is None
                     else max(0, io_total - io_prefill))
        snap = runtime.snapshot()
        cache_stats = snap.get("cache") or {}
        sha = hashlib.sha256(
            ",".join(str(t) for t in decoded).encode()).hexdigest()
        row = {
            "arm": arm_name,
            "overrides": {k: (v if not isinstance(v, bool) else v)
                          for k, v in {**overrides,
                                       "cache_scope": getattr(runtime.config, "cache_scope", None)}.items()},
            "context_tokens": args.context_tokens,
            "decode_tokens": len(decoded),
            "prefill_tok_s": len(prompt_ids) / prefill_s if prefill_s else None,
            "ttft_s": ttft_s,
            "decode_tok_s": (len(decoded) - 1) / decode_s if decode_s else None,
            "decode_wall_s": decode_s,
            "decode_io_bytes": decode_io,
            "decode_bytes_per_token": (decode_io / (len(decoded) - 1)
                                       if decode_io is not None and len(decoded) > 1 else None),
            "realized_decode_gib_per_s": (
                (decode_io / GIB) / decode_s
                if decode_io is not None and decode_s else None),
            "cache_hit_rate": cache_stats.get("hit_rate"),
            "cache_expert_misses": cache_stats.get("expert_misses"),
            "io_read_bytes_total": io_total,
            "peak_gib": _peak_gib(mx),
            "decoded_ids": decoded,
            "decoded_sha256": sha,
            "decoded_text_head": tokenizer.decode(decoded[:32]),
        }
        log(f"[{arm_name}] decode {row['decode_tok_s']:.3f} tok/s  "
            f"realizedBW {row['realized_decode_gib_per_s']}  "
            f"hit {row['cache_hit_rate']}  peak {row['peak_gib']}  sha {sha[:12]}")
        return row
    finally:
        try:
            runtime.close()
        except Exception:
            pass


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logf = open(str(args.out) + ".log", "a", buffering=1)

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        logf.write(s + "\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    reference_sha = None
    for arm in args.arms:
        name, overrides = parse_arm(arm)
        log(f"=== arm {name} overrides={overrides} ===")
        try:
            row = run_arm(args, name, overrides, log)
        except Exception as exc:
            row = {"arm": name, "error": repr(exc)}
            log(f"[{name}] ERROR {exc!r}")
        if name == "control" and "decoded_sha256" in row:
            reference_sha = row["decoded_sha256"]
        if reference_sha and row.get("decoded_sha256"):
            row["byte_identical_to_control"] = (row["decoded_sha256"] == reference_sha)
        results.append(row)
        # append-only: rewrite the receipt with all rows so far.
        payload = {
            "meta": {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "worker": "W24",
                "device": "cpu" if args.cpu else "gpu",
                "model": str(args.model),
                "context_tokens": args.context_tokens,
                "decode_tokens": args.decode_tokens,
                "reference_sha256": reference_sha,
            },
            "arms": results,
        }
        args.out.write_text(json.dumps(payload, indent=2))
    log(f"[ab] wrote {args.out}  ({len(results)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
