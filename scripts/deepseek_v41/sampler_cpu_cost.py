#!/usr/bin/env python3
"""W53: per-token sampler cost on CPU over a DeepSeek-V4.1-shaped logits row.

Measures the host+device work the SERVED decode path adds on top of the
in-process greedy bench (which only does ``mx.argmax`` on already-evaluated
logits). All work runs on ``mx.cpu`` so the script needs no GPU/flock and no
experts.bin; it is an ORDER-OF-MAGNITUDE probe for the host round-trip and the
129,280-vocab argsort/argpartition cost, not a Metal timing.

Arms (all top_k=20, top_p=0.95, temperature=1.0 — the served cell sampler):
  greedy_argmax        : mx.argmax over the full row (what ab_decode does)
  sparse_device_topk   : sparse_distribution_from_mlx_logits + draw
                         (the SERVED path: device argpartition+logsumexp over
                          full vocab, only the k-support crosses to host)
  dense_full_vocab     : distribution_from_logits(_logits_to_numpy(row))
                         (the "unpruned" host reference: full-vocab softmax +
                          argsort on the host every token)

Run:
  PYTHONPATH=$PWD nice -n 19 .venv/bin/python3 \
    scripts/deepseek_v41/sampler_cpu_cost.py --iters 200 --vocab 129280
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MTPLX_DISABLE_METAL", "1")

import mlx.core as mx
import numpy as np

mx.set_default_device(mx.cpu)

from mtplx.sampling import (  # noqa: E402
    SamplerConfig,
    distribution_from_logits,
    sample_from_distribution,
)
from mtplx.fast_sampling import sparse_distribution_from_mlx_logits  # noqa: E402


def _logits_to_numpy(row: mx.array) -> np.ndarray:
    row = row.astype(mx.float32)
    mx.eval(row)
    return np.asarray(row, dtype=np.float32).astype(np.float64).reshape(-1)


def _percentiles(samples: list[float]) -> dict[str, float]:
    arr = np.asarray(samples, dtype=np.float64) * 1e3  # ms
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def _bench(fn, iters: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return _percentiles(samples)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--vocab", type=int, default=129280)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260829)
    args = ap.parse_args(argv)

    rng_np = np.random.default_rng(args.seed)
    # A realistic decode logits row: mostly near-zero with a peaked head.
    base = rng_np.standard_normal(args.vocab).astype(np.float32) * 2.0
    peak = rng_np.choice(args.vocab, size=64, replace=False)
    base[peak] += rng_np.uniform(6.0, 14.0, size=64).astype(np.float32)
    row_mx = mx.array(base)
    mx.eval(row_mx)

    config = SamplerConfig(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k
    )

    def greedy_argmax():
        tok = mx.argmax(row_mx, axis=-1)
        mx.eval(tok)
        return int(tok.item())

    draw_rng = np.random.default_rng(args.seed)

    def sparse_device_topk():
        dist = sparse_distribution_from_mlx_logits(row_mx, config)
        return sample_from_distribution(dist, draw_rng)

    draw_rng2 = np.random.default_rng(args.seed)

    def dense_full_vocab():
        probs = distribution_from_logits(_logits_to_numpy(row_mx), config)
        return sample_from_distribution(probs, draw_rng2)

    results = {
        "device": "cpu",
        "vocab": args.vocab,
        "iters": args.iters,
        "sampler": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k},
        "arms": {
            "greedy_argmax": _bench(greedy_argmax, args.iters, args.warmup),
            "sparse_device_topk": _bench(sparse_device_topk, args.iters, args.warmup),
            "dense_full_vocab": _bench(dense_full_vocab, args.iters, args.warmup),
        },
    }
    g = results["arms"]["greedy_argmax"]["median_ms"]
    for name, r in results["arms"].items():
        r["delta_vs_greedy_ms"] = round(r["median_ms"] - g, 4)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
