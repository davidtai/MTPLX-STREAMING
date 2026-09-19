#!/usr/bin/env python3
"""W56 / KERNEL_LEDGER K27 -- DeepSeek-V4.1-Flash shape / tiling microbench.

Times each hot op on the decode (M=1) and prefill (rows=chunk, T up to 16,384)
paths at the *real* model shapes, in its current form vs the shape-fix form, so a
GPU window can price the K27 layout findings directly.  The cases mirror the W56
audit table:

  score      -- QK^T / PV as einsum vs explicit batched matmul; f32 vs bf16 inputs;
                score chunk 512/1024/2048; split-K key-chunk sizes.
  olora      -- grouped o-LoRA down-proj einsum vs batched matmul (strided vs
                batched-heads layout).
  gather     -- routed mxfp4 gather_qmm rows-per-call 6/12/24/48/96, and
                unsorted per-row gather_qmv vs sorted-indices fused gather_qmm_rhs.
  down_align -- expert down-proj fast gather_qmv, ragged K=2304 vs zero-padded
                K=2560 (K%512==0).
  dense      -- prefill dense expert matmul bf16 vs f32 at rows 128/256/1024,
                5120x2304 (the K26 / W50 dtype question).

CPU-SAFE contract: ``--help`` and ``--dry-run`` never touch Metal (they only print
the planned cases and shapes -- importing mlx.core does not initialise the GPU;
the first array op would).  A real run executes on the default device, which is
Metal inside a GPU window; run it ONLY inside a flock'd window with qwen unloaded
([[all-gpu-work-through-flock]], [[hy3-benchmark-panic-protocol]]).  Emits a JSON
receipt with warmup / iterations / median-ms / achieved GB-s or TFLOPS per variant.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

# Real released DeepSeek-V4.1-Flash text shapes (ModelArgs defaults).
HIDDEN = 5120
INTER = 2304
HEADS = 64
HEAD_DIM = 512
N_ROUTED = 384
TOP_K = 6
O_GROUPS = 8
O_LORA_RANK = 1024
IN_PER_GROUP = HEADS * HEAD_DIM // O_GROUPS  # 4096
Q_LORA_RANK = 1280
VOCAB = 129280
MXFP4_GS = 32


@dataclass
class Variant:
    name: str
    note: str
    # build() -> (fn, flops, bytes_moved).  Built lazily (real run only).
    build: Callable[[], tuple[Callable[[], object], float, float]]


@dataclass
class Case:
    name: str
    shape_note: str
    variants: list[Variant] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Case builders (import mx lazily so --help / --dry-run never initialise Metal)
# --------------------------------------------------------------------------- #
def _build_cases(seq_t: int, score_chunk: int, gather_experts: int) -> list[Case]:
    import mlx.core as mx

    def rn(*shape, dtype=mx.float32):
        return mx.random.normal(shape).astype(dtype)

    cases: list[Case] = []

    # --- score: QK^T then PV, einsum vs explicit batched matmul, f32 vs bf16 --
    for dt_name, dt in (("f32", mx.float32), ("bf16", mx.bfloat16)):
        c = Case(
            f"score.chunk{score_chunk}.T{seq_t}.{dt_name}",
            f"q[1,{score_chunk},{HEADS},{HEAD_DIM}] KV[1,{seq_t},{HEAD_DIM}] "
            f"-> scores[1,{score_chunk},{HEADS},{seq_t}]",
        )
        s, H, hd, T = score_chunk, HEADS, HEAD_DIM, seq_t
        qk_flops = 2.0 * s * H * hd * T
        pv_flops = 2.0 * s * H * T * hd

        def _einsum():
            q = rn(1, s, H, hd, dtype=dt)
            KV = rn(1, T, hd, dtype=dt)
            def run():
                sc = mx.einsum("bshd,btd->bsht", q, KV)
                o = mx.einsum("bsht,btd->bshd", sc, KV)
                return o
            return run, qk_flops + pv_flops, 0.0

        def _matmul():
            q = rn(1, s, H, hd, dtype=dt)
            KV = rn(1, T, hd, dtype=dt)
            def run():
                sc = mx.matmul(q.reshape(1, s * H, hd), mx.swapaxes(KV, 1, 2))
                o = mx.matmul(sc, KV).reshape(1, s, H, hd)
                return o
            return run, qk_flops + pv_flops, 0.0

        c.variants += [
            Variant("einsum", "shipped mx.einsum QK^T/PV", _einsum),
            Variant("matmul", "explicit batched matmul reshape", _matmul),
        ]
        cases.append(c)

    # --- score split-K key-chunk sizes (transient cap) -----------------------
    c = Case(
        f"score_keychunk.chunk{score_chunk}.T{seq_t}.f32",
        f"online-softmax over key-chunk widths of KV[1,{seq_t},{HEAD_DIM}]",
    )
    s, H, hd, T = score_chunk, HEADS, HEAD_DIM, seq_t
    for kc in (0, 512, 1024, 2048):  # 0 = one-shot (no chunk)
        def _mk(kc=kc):
            def build():
                q = rn(1, s, H, hd)
                KV = rn(1, T, hd)
                def run():
                    if kc == 0:
                        sc = mx.einsum("bshd,btd->bsht", q, KV)
                        w = mx.softmax(sc, axis=-1)
                        return mx.einsum("bsht,btd->bshd", w, KV)
                    acc = mx.zeros((1, s, H, hd))
                    denom = mx.zeros((1, s, H, 1))
                    m = mx.full((1, s, H, 1), -1e30)
                    for c0 in range(0, T, kc):
                        KVc = KV[:, c0:c0 + kc, :]
                        sc = mx.einsum("bshd,btd->bsht", q, KVc)
                        m_new = mx.maximum(m, mx.max(sc, axis=-1, keepdims=True))
                        corr = mx.exp(m - m_new)
                        p = mx.exp(sc - m_new)
                        denom = denom * corr + mx.sum(p, axis=-1, keepdims=True)
                        acc = acc * corr + mx.einsum("bsht,btd->bshd", p, KVc)
                        m = m_new
                    return acc / denom
                return run, 2.0 * s * H * hd * T * 2, 0.0
            return build
        c.variants.append(
            Variant(f"keychunk{kc}", "one-shot" if kc == 0 else f"split-K {kc}", _mk())
        )
    cases.append(c)

    # --- o-LoRA grouped down-proj: einsum vs batched matmul ------------------
    for rows in (1, score_chunk):
        c = Case(
            f"olora.rows{rows}.f32",
            f"o[1,{rows},{O_GROUPS},{IN_PER_GROUP}] W[{O_GROUPS},{O_LORA_RANK},"
            f"{IN_PER_GROUP}] -> [1,{rows},{O_GROUPS},{O_LORA_RANK}]",
        )
        g, r, d = O_GROUPS, O_LORA_RANK, IN_PER_GROUP
        flops = 2.0 * rows * g * r * d

        def _oe(rows=rows):
            o = rn(1, rows, g, d)
            W = rn(g, r, d)
            def run():
                return mx.einsum("bsgd,grd->bsgr", o, W)
            return run, flops, 0.0

        def _om(rows=rows):
            o = rn(1, rows, g, d)
            W = rn(g, r, d)
            def run():
                ob = mx.transpose(o, (2, 0, 1, 3)).reshape(g, rows, d)
                out = mx.matmul(ob, mx.swapaxes(W, 1, 2))
                return mx.transpose(out.reshape(g, 1, rows, r), (1, 2, 0, 3))
            return run, flops, 0.0

        c.variants += [
            Variant("einsum", "shipped grouped einsum", _oe),
            Variant("bmm", "transpose-to-batch + matmul", _om),
        ]
        cases.append(c)

    # --- routed mxfp4 gather: rows-per-call + unsorted vs sorted-indices -----
    # E is bounded (default 64) to keep the pre-quantize bf16 transient under the
    # memory cap; sorted vs unsorted kernel selection (gather_qmm_rhs needs B/E>=4,
    # B>=16) is exercised identically.  The served-model A/B (bench_standard_shape +
    # MTPLX_DSV41_LAYOUT_FIX) prices F1 at the true E=384; this prices the ratio.
    E, N, K = gather_experts, INTER, HIDDEN
    dense = rn(E, N, K, dtype=mx.bfloat16)
    for rows in (6, 12, 24, 48, 96, score_chunk * TOP_K):
        c = Case(
            f"gather.rows{rows}.mxfp4",
            f"x[{rows},1,1,{K}] w[{E},{N},{K}] mxfp4 gs{MXFP4_GS} -> [{rows},{N}]",
        )

        def _unsorted(rows=rows):
            w, sc = mx.quantize(dense, group_size=MXFP4_GS, bits=4, mode="mxfp4")
            x = rn(rows, 1, 1, K, dtype=mx.bfloat16)
            slot = (mx.random.randint(0, E, (rows,))).astype(mx.int32).reshape(-1, 1)
            def run():
                return mx.gather_qmm(x, w, sc, rhs_indices=slot, transpose=True,
                                     group_size=MXFP4_GS, bits=4, mode="mxfp4")
            return run, 2.0 * rows * N * K, 0.0

        def _sorted(rows=rows):
            w, sc = mx.quantize(dense, group_size=MXFP4_GS, bits=4, mode="mxfp4")
            x = rn(rows, 1, 1, K, dtype=mx.bfloat16)
            slot0 = (mx.random.randint(0, E, (rows,))).astype(mx.int32)
            perm = mx.argsort(slot0)
            xs = mx.take(x, perm, axis=0)
            slot = mx.take(slot0, perm, axis=0).reshape(-1, 1)
            def run():
                return mx.gather_qmm(xs, w, sc, rhs_indices=slot, transpose=True,
                                     group_size=MXFP4_GS, bits=4, mode="mxfp4",
                                     sorted_indices=True)
            return run, 2.0 * rows * N * K, 0.0

        c.variants += [
            Variant("unsorted", "shipped gather_qmv per row", _unsorted),
            Variant("sorted", "sorted_indices -> gather_qmm_rhs", _sorted),
        ]
        cases.append(c)

    # --- down-proj fast gather_qmv: ragged K=2304 vs padded K=2560 ----------
    # down-proj kernel selection (fast vs slow gather_qmv) is set by N%8 and K%512
    # only, independent of the expert count, so use a small bank to bound memory.
    E_DA = 8
    c = Case(
        "down_align.mxfp4",
        f"down-proj x[.,1,1,K] w[{E_DA},{HIDDEN},K] mxfp4; fast qmv needs K%512==0",
    )
    for K_down, tag in ((INTER, "ragged2304"), (2560, "padded2560")):
        def _mk(K_down=K_down):
            def build():
                dd = rn(E_DA, HIDDEN, K_down, dtype=mx.bfloat16)
                w, sc = mx.quantize(dd, group_size=MXFP4_GS, bits=4, mode="mxfp4")
                rows = TOP_K
                x = rn(rows, 1, 1, K_down, dtype=mx.bfloat16)
                slot = (mx.random.randint(0, E_DA, (rows,))).astype(mx.int32).reshape(-1, 1)
                def run():
                    return mx.gather_qmm(x, w, sc, rhs_indices=slot, transpose=True,
                                         group_size=MXFP4_GS, bits=4, mode="mxfp4")
                return run, 2.0 * rows * HIDDEN * K_down, 0.0
            return build
        c.variants.append(Variant(tag, f"K={K_down} (%512={K_down % 512})", _mk()))
    cases.append(c)

    # --- dense prefill expert matmul: bf16 vs f32 at rows 128/256/1024 -------
    for rows in (128, 256, 1024):
        c = Case(
            f"dense.rows{rows}",
            f"x[{rows},{HIDDEN}] @ gate[{INTER},{HIDDEN}]^T -> [{rows},{INTER}] "
            "(dequantized-once dense matmul)",
        )
        for dt_name, dt in (("bf16", mx.bfloat16), ("f32", mx.float32)):
            def _mk(dt=dt, rows=rows):
                def build():
                    x = rn(rows, HIDDEN, dtype=dt)
                    W = rn(INTER, HIDDEN, dtype=dt)
                    def run():
                        return mx.matmul(x, W.T)
                    return run, 2.0 * rows * INTER * HIDDEN, 0.0
                return build
            c.variants.append(Variant(dt_name, f"dense {dt_name} matmul", _mk()))
        cases.append(c)

    return cases


# --------------------------------------------------------------------------- #
def _time_variant(v: Variant, warmup: int, iters: int) -> dict:
    import mlx.core as mx

    fn, flops, _ = v.build()
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)  # ms
    med = statistics.median(samples)
    rec = {
        "variant": v.name,
        "note": v.note,
        "median_ms": round(med, 4),
        "min_ms": round(min(samples), 4),
        "max_ms": round(max(samples), 4),
    }
    if flops and med > 0:
        rec["tflops"] = round(flops / (med * 1e-3) / 1e12, 3)
    return rec


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                   help="print the planned cases + shapes and exit (no Metal, CPU-safe)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--seq-t", type=int, default=16384,
                   help="running KV length T for the score cases")
    p.add_argument("--score-chunk", type=int, default=1024, choices=[512, 1024, 2048],
                   help="query rows per prefill score chunk (MTPLX_DSV41_PREFILL_CHUNK)")
    p.add_argument("--gather-experts", type=int, default=64,
                   help="bank size E for the routed-gather case (bound the pre-quantize "
                        "transient; raise toward 384 only with a larger --memory-limit-gib)")
    p.add_argument("--memory-limit-gib", type=float, default=8.0,
                   help="cap MLX working-set (real run only)")
    p.add_argument("--out", type=str, default=None, help="JSON receipt path")
    p.add_argument("--only", type=str, default=None,
                   help="substring filter on case name")
    args = p.parse_args(argv)

    if args.dry_run:
        # Describe the plan WITHOUT importing/using mlx ops (no GPU touch).
        plan = _dry_run_plan(args.seq_t, args.score_chunk)
        if args.only:
            plan = [c for c in plan if args.only in c["case"]]
        print(json.dumps({"dry_run": True, "seq_t": args.seq_t,
                          "score_chunk": args.score_chunk, "cases": plan}, indent=2))
        return 0

    import mlx.core as mx
    # Cap the working set so an in-window run stays under budget.
    try:
        mx.set_memory_limit(int(args.memory_limit_gib * (1 << 30)))
    except Exception:
        pass

    cases = _build_cases(args.seq_t, args.score_chunk, args.gather_experts)
    if args.only:
        cases = [c for c in cases if args.only in c.name]

    results = []
    for case in cases:
        recs = [_time_variant(v, args.warmup, args.iters) for v in case.variants]
        results.append({"case": case.name, "shape": case.shape_note, "variants": recs})
        best = min(recs, key=lambda r: r["median_ms"])
        line = "  ".join(f"{r['variant']}={r['median_ms']}ms" for r in recs)
        print(f"[{case.name}] {line}  (fastest: {best['variant']})")

    receipt = {
        "kind": "dsv41_shape_tiling_microbench",
        "kernel_ledger": "K27",
        "device": _device_str(),
        "mlx_version": getattr(mx, "__version__", "?"),
        "warmup": args.warmup,
        "iters": args.iters,
        "seq_t": args.seq_t,
        "score_chunk": args.score_chunk,
        "gather_experts": args.gather_experts,
        "results": results,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(receipt, f, indent=2)
        print(f"receipt -> {args.out}")
    return 0


def _device_str() -> str:
    try:
        import mlx.core as mx
        info = mx.metal.device_info()
        return f"{info.get('device_name', '?')}/{info.get('architecture', '?')}"
    except Exception:
        return platform.platform()


def _dry_run_plan(seq_t: int, score_chunk: int) -> list[dict]:
    """The case/variant/shape catalogue as pure data (no mlx ops)."""
    s, H, hd, T = score_chunk, HEADS, HEAD_DIM, seq_t
    plan = []
    for dt in ("f32", "bf16"):
        plan.append({"case": f"score.chunk{s}.T{T}.{dt}",
                     "shape": f"q[1,{s},{H},{hd}] KV[1,{T},{hd}] -> [1,{s},{H},{T}]",
                     "variants": ["einsum", "matmul"]})
    plan.append({"case": f"score_keychunk.chunk{s}.T{T}.f32",
                 "shape": f"KV[1,{T},{hd}] online-softmax key-chunk widths",
                 "variants": ["keychunk0", "keychunk512", "keychunk1024", "keychunk2048"]})
    for rows in (1, s):
        plan.append({"case": f"olora.rows{rows}.f32",
                     "shape": f"o[1,{rows},{O_GROUPS},{IN_PER_GROUP}] "
                              f"W[{O_GROUPS},{O_LORA_RANK},{IN_PER_GROUP}]",
                     "variants": ["einsum", "bmm"]})
    for rows in (6, 12, 24, 48, 96, s * TOP_K):
        plan.append({"case": f"gather.rows{rows}.mxfp4",
                     "shape": f"x[{rows},1,1,{HIDDEN}] w[{N_ROUTED},{INTER},{HIDDEN}] mxfp4",
                     "variants": ["unsorted", "sorted"]})
    plan.append({"case": "down_align.mxfp4",
                 "shape": f"down-proj w[{N_ROUTED},{HIDDEN},K]; fast qmv needs K%512==0",
                 "variants": [f"ragged2304 (%512={INTER % 512})", "padded2560 (%512=0)"]})
    for rows in (128, 256, 1024):
        plan.append({"case": f"dense.rows{rows}",
                     "shape": f"x[{rows},{HIDDEN}] @ gate[{INTER},{HIDDEN}]^T",
                     "variants": ["bf16", "f32"]})
    return plan


if __name__ == "__main__":
    sys.exit(main())
