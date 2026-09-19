#!/usr/bin/env python3
"""Real-weight one-layer screen: cached BF16 matmul vs packed MXFP8 gather_qmm."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open


SHARD = Path(
    "/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4/"
    "model-00003.safetensors"
)
WEIGHT_KEY = "layers.0.attn.wo_a.weight"
SCALES_KEY = "layers.0.attn.wo_a.scales"
WO_B_WEIGHT_KEY = "layers.0.attn.wo_b.weight"
WO_B_SCALES_KEY = "layers.0.attn.wo_b.scales"
GROUPS = 8
RANK = 1024
INPUT = 4096
GROUP_SIZE = 32
BITS = 8
MODE = "mxfp8"
ROUNDS = 9
ITERS = 30


def run_ms(fn, iterations: int) -> float:
    mx.synchronize()
    start = time.perf_counter_ns()
    for _ in range(iterations):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter_ns() - start) / 1e6 / iterations


def summarize(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "round_ms": values,
    }


def main() -> None:
    mx.set_memory_limit(2_000_000_000)
    mx.random.seed(41)
    with safe_open(str(SHARD), framework="mlx") as handle:
        weight = handle.get_tensor(WEIGHT_KEY)
        scales = handle.get_tensor(SCALES_KEY)
        wo_b_weight = handle.get_tensor(WO_B_WEIGHT_KEY)
        wo_b_scales = handle.get_tensor(WO_B_SCALES_KEY)

    assert tuple(weight.shape) == (GROUPS * RANK, INPUT // 4), weight.shape
    assert tuple(scales.shape) == (GROUPS * RANK, INPUT // GROUP_SIZE), scales.shape
    assert weight.dtype == mx.uint32, weight.dtype
    assert scales.dtype == mx.uint8, scales.dtype
    assert tuple(wo_b_weight.shape) == (5120, GROUPS * RANK // 4)
    assert tuple(wo_b_scales.shape) == (5120, GROUPS * RANK // GROUP_SIZE)

    weight_grouped = weight.reshape(GROUPS, RANK, -1)
    scales_grouped = scales.reshape(GROUPS, RANK, -1)
    dense = mx.dequantize(
        weight,
        scales,
        None,
        group_size=GROUP_SIZE,
        bits=BITS,
        mode=MODE,
    ).astype(mx.bfloat16).reshape(GROUPS, RANK, INPUT)
    dense_t = mx.contiguous(mx.swapaxes(dense, 1, 2))
    mx.eval(weight_grouped, scales_grouped, dense_t)

    results: dict[str, object] = {
        "scope": (
            "one real target attention layer; exact fused-path layout; "
            "wo_a plus common wo_b output projection; no model throughput claim"
        ),
        "shard": str(SHARD),
        "weight_key": WEIGHT_KEY,
        "codec": {"mode": MODE, "bits": BITS, "group_size": GROUP_SIZE},
        "geometry": {"groups": GROUPS, "rank": RANK, "input_per_group": INPUT},
        "dense_cache_bytes_per_layer": GROUPS * RANK * INPUT * 2,
        "dense_cache_bytes_40_layers": 40 * GROUPS * RANK * INPUT * 2,
        "shapes": {},
    }

    for rows in (1, 6):
        # Exact fused-path stride layout: [rows, groups, input] followed by the
        # same swapaxes used after inverse RoPE in Attention._out_prep_fused.
        base = mx.random.normal((rows, GROUPS, INPUT)).astype(mx.bfloat16)
        x = base.swapaxes(0, 1)

        def control_down():
            return mx.matmul(x, dense_t)

        def candidate_down():
            return mx.gather_qmm(
                x,
                weight_grouped,
                scales_grouped,
                None,
                transpose=True,
                group_size=GROUP_SIZE,
                bits=BITS,
                mode=MODE,
            )

        def wo_b(down):
            return mx.quantized_matmul(
                down.swapaxes(0, 1).reshape(rows, GROUPS * RANK),
                wo_b_weight,
                wo_b_scales,
                None,
                transpose=True,
                group_size=GROUP_SIZE,
                bits=BITS,
                mode=MODE,
            )

        def control():
            return wo_b(control_down())

        def candidate():
            return wo_b(candidate_down())

        control_out = control()
        candidate_out = candidate()
        mx.eval(control_out, candidate_out)
        delta = candidate_out.astype(mx.float32) - control_out.astype(mx.float32)
        ref = control_out.astype(mx.float32)
        max_abs, mean_abs, l2_delta, l2_ref = (
            mx.max(mx.abs(delta)),
            mx.mean(mx.abs(delta)),
            mx.sqrt(mx.sum(delta * delta)),
            mx.sqrt(mx.sum(ref * ref)),
        )
        mx.eval(max_abs, mean_abs, l2_delta, l2_ref)

        # Compile/warm both routes before alternating timed rounds. Report the
        # down projection alone and the full down+common wo_b path separately.
        for _ in range(8):
            mx.eval(control(), candidate())
        mx.synchronize()
        control_ms: list[float] = []
        candidate_ms: list[float] = []
        control_down_ms: list[float] = []
        candidate_down_ms: list[float] = []
        for round_id in range(ROUNDS):
            if round_id % 2:
                candidate_ms.append(run_ms(candidate, ITERS))
                control_ms.append(run_ms(control, ITERS))
                candidate_down_ms.append(run_ms(candidate_down, ITERS))
                control_down_ms.append(run_ms(control_down, ITERS))
            else:
                control_ms.append(run_ms(control, ITERS))
                candidate_ms.append(run_ms(candidate, ITERS))
                control_down_ms.append(run_ms(control_down, ITERS))
                candidate_down_ms.append(run_ms(candidate_down, ITERS))

        c_med = statistics.median(control_ms)
        q_med = statistics.median(candidate_ms)
        results["shapes"][str(rows)] = {
            "control": summarize(control_ms),
            "candidate": summarize(candidate_ms),
            "candidate_speedup": c_med / q_med,
            "candidate_time_reduction_pct": 100.0 * (c_med - q_med) / c_med,
            "down_only": {
                "control": summarize(control_down_ms),
                "candidate": summarize(candidate_down_ms),
                "candidate_speedup": (
                    statistics.median(control_down_ms)
                    / statistics.median(candidate_down_ms)
                ),
            },
            "numerics": {
                "control_dtype": str(control_out.dtype),
                "candidate_dtype": str(candidate_out.dtype),
                "max_abs": float(max_abs.item()),
                "mean_abs": float(mean_abs.item()),
                "relative_l2": float((l2_delta / mx.maximum(l2_ref, 1e-30)).item()),
            },
        }

    results["mlx"] = {
        "active_bytes": mx.get_active_memory(),
        "peak_bytes": mx.get_peak_memory(),
        "cache_bytes": mx.get_cache_memory(),
    }
    payload = json.dumps(results, indent=2, sort_keys=True) + "\n"
    Path("/tmp/dsv41-woa-direct-bench.json").write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
