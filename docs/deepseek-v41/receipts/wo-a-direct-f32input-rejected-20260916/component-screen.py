#!/usr/bin/env python3
"""One-layer screen for transient BF16 dequantization of target wo_a."""

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
GROUPS = 8
RANK = 1024
INPUT = 4096
GROUP_SIZE = 32
BITS = 8
MODE = "mxfp8"
ROUNDS = 7
ITERS = 16


def timed(fn) -> float:
    mx.synchronize()
    started = time.perf_counter_ns()
    for _ in range(ITERS):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter_ns() - started) / 1e6 / ITERS


def median(values: list[float]) -> float:
    return statistics.median(values)


def main() -> None:
    mx.set_memory_limit(2_000_000_000)
    mx.random.seed(4102)
    with safe_open(str(SHARD), framework="mlx") as handle:
        weight = handle.get_tensor("layers.0.attn.wo_a.weight")
        scales = handle.get_tensor("layers.0.attn.wo_a.scales")
        wo_b_weight = handle.get_tensor("layers.0.attn.wo_b.weight")
        wo_b_scales = handle.get_tensor("layers.0.attn.wo_b.scales")

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
    mx.eval(dense_t, wo_b_weight, wo_b_scales)

    def fresh_transpose(*, contiguous: bool):
        value = mx.dequantize(
            weight,
            scales,
            None,
            group_size=GROUP_SIZE,
            bits=BITS,
            mode=MODE,
        ).astype(mx.bfloat16).reshape(GROUPS, RANK, INPUT)
        value = mx.swapaxes(value, 1, 2)
        return mx.contiguous(value) if contiguous else value

    result: dict[str, object] = {
        "scope": "one real target layer; cached BF16 transpose versus transient dequantize+transpose",
        "iterations_per_round": ITERS,
        "rounds": ROUNDS,
        "dense_cache_bytes_per_layer": GROUPS * RANK * INPUT * 2,
        "shapes": {},
    }
    for rows in (1, 6):
        base = mx.random.normal((rows, GROUPS, INPUT)).astype(mx.bfloat16)
        x = base.swapaxes(0, 1)

        def down_cached():
            return mx.matmul(x, dense_t)

        def down_transient_view():
            return mx.matmul(x, fresh_transpose(contiguous=False))

        def down_transient_contiguous():
            return mx.matmul(x, fresh_transpose(contiguous=True))

        def down_qmm_f32cast():
            return mx.gather_qmm(
                x.astype(mx.float32),
                weight_grouped,
                scales_grouped,
                None,
                transpose=True,
                group_size=GROUP_SIZE,
                bits=BITS,
                mode=MODE,
            ).astype(mx.bfloat16)

        def down_qmm_chunks(chunks: int):
            width = INPUT // chunks
            parts = []
            for part in range(chunks):
                lo = part * width
                hi = lo + width
                parts.append(
                    mx.gather_qmm(
                        x[..., lo:hi],
                        weight_grouped[..., lo // 4:hi // 4],
                        scales_grouped[..., lo // GROUP_SIZE:hi // GROUP_SIZE],
                        None,
                        transpose=True,
                        group_size=GROUP_SIZE,
                        bits=BITS,
                        mode=MODE,
                    ).astype(mx.float32)
                )
            total = parts[0]
            for part in parts[1:]:
                total = total + part
            return total.astype(mx.bfloat16)

        def wo_b(value):
            return mx.quantized_matmul(
                value.swapaxes(0, 1).reshape(rows, GROUPS * RANK),
                wo_b_weight,
                wo_b_scales,
                None,
                transpose=True,
                group_size=GROUP_SIZE,
                bits=BITS,
                mode=MODE,
            )

        arms = {
            "cached": lambda: wo_b(down_cached()),
            "transient_view": lambda: wo_b(down_transient_view()),
            "transient_contiguous": lambda: wo_b(down_transient_contiguous()),
            "qmm_f32cast": lambda: wo_b(down_qmm_f32cast()),
            "qmm_chunk2": lambda: wo_b(down_qmm_chunks(2)),
            "qmm_chunk4": lambda: wo_b(down_qmm_chunks(4)),
        }
        outputs = {name: fn() for name, fn in arms.items()}
        mx.eval(*outputs.values())
        control = outputs["cached"]
        numerics = {}
        for name, output in outputs.items():
            delta = output.astype(mx.float32) - control.astype(mx.float32)
            max_abs = mx.max(mx.abs(delta))
            equal = mx.all(output == control)
            mx.eval(max_abs, equal)
            numerics[name] = {
                "byte_equal_control": bool(equal.item()),
                "max_abs": float(max_abs.item()),
            }

        for _ in range(4):
            mx.eval(*(fn() for fn in arms.values()))
        samples = {name: [] for name in arms}
        down_samples = {
            "cached": [],
            "transient_view": [],
            "transient_contiguous": [],
            "qmm_f32cast": [],
            "qmm_chunk2": [],
            "qmm_chunk4": [],
        }
        down_arms = {
            "cached": down_cached,
            "transient_view": down_transient_view,
            "transient_contiguous": down_transient_contiguous,
            "qmm_f32cast": down_qmm_f32cast,
            "qmm_chunk2": lambda: down_qmm_chunks(2),
            "qmm_chunk4": lambda: down_qmm_chunks(4),
        }
        names = list(arms)
        for round_id in range(ROUNDS):
            order = names[round_id % len(names):] + names[:round_id % len(names)]
            for name in order:
                samples[name].append(timed(arms[name]))
            for name in reversed(order):
                down_samples[name].append(timed(down_arms[name]))

        cached_ms = median(samples["cached"])
        result["shapes"][str(rows)] = {
            "full_median_ms": {name: median(values) for name, values in samples.items()},
            "down_median_ms": {
                name: median(values) for name, values in down_samples.items()
            },
            "full_slowdown_vs_cached": {
                name: median(values) / cached_ms for name, values in samples.items()
            },
            "numerics": numerics,
        }

    result["mlx"] = {
        "active_bytes": mx.get_active_memory(),
        "peak_bytes": mx.get_peak_memory(),
        "cache_bytes": mx.get_cache_memory(),
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    Path("/tmp/dsv41-woa-transient-screen.json").write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
