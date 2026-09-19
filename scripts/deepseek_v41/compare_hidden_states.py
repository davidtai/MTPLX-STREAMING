#!/usr/bin/env python3
"""Diff two :mod:`scripts.deepseek_v41.dump_hidden_states` dumps.

Prints, layer by layer, the max-abs difference of the per-layer
``first64_last_pos`` vectors (and the deltas of ``mean``/``std``/``max_abs``),
and reports the FIRST layer whose ``first64`` max-abs difference exceeds
``--threshold``.  Also diffs ``embed``, ``final_norm``, the engram gate stats,
and the final top-8 logits.

Intended use: the CPU dump is committed under
``docs/deepseek-v41/receipts/``; the orchestrator takes the same dump on the GPU
inside the guarded window and runs this to localize any GPU-vs-CPU divergence to
its first layer.  Pure Python, no MLX -- safe anywhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p: str | Path) -> dict:
    return json.loads(Path(p).read_text())


def _maxabs_diff(a: list, b: list) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    return max(abs(float(a[i]) - float(b[i])) for i in range(n))


def _fmt_stats_delta(sa: dict, sb: dict) -> str:
    keys = ("mean", "std", "max_abs")
    parts = []
    for k in keys:
        if k in sa and k in sb:
            parts.append(f"d{k}={sb[k] - sa[k]:+.4g}")
    fa, fb = sa.get("finite", True), sb.get("finite", True)
    if not (fa and fb):
        parts.append(f"finite A={fa} B={fb}")
    return " ".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dump_a", help="baseline dump JSON (e.g. the CPU dump)")
    parser.add_argument("dump_b", help="comparison dump JSON (e.g. the GPU dump)")
    parser.add_argument("--threshold", type=float, default=1e-2,
                        help="first64 max-abs diff that counts as a divergence (default 1e-2)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    a = _load(args.dump_a)
    b = _load(args.dump_b)

    print(f"A: {args.dump_a}")
    print(f"   device={a['meta'].get('device')} git={a['meta'].get('git_rev')} "
          f"ablate={a['meta'].get('engram_ablate')} prompt_ids={a['meta'].get('prompt_ids')}")
    print(f"B: {args.dump_b}")
    print(f"   device={b['meta'].get('device')} git={b['meta'].get('git_rev')} "
          f"ablate={b['meta'].get('engram_ablate')} prompt_ids={b['meta'].get('prompt_ids')}")
    if a["meta"].get("prompt_ids") != b["meta"].get("prompt_ids"):
        print("!! WARNING: prompt_ids differ -- the dumps are not comparable")

    print(f"\nthreshold (first64 max-abs) = {args.threshold:g}\n")

    # embed
    if a.get("embed") and b.get("embed"):
        d = _maxabs_diff(a["embed"]["first64_last_pos"], b["embed"]["first64_last_pos"])
        print(f"embed            first64_maxabs_diff={d:.4e}   {_fmt_stats_delta(a['embed'], b['embed'])}")

    la = {r["layer"]: r for r in a.get("layers", [])}
    lb = {r["layer"]: r for r in b.get("layers", [])}
    first_diverging = None
    for i in sorted(set(la) & set(lb)):
        d = _maxabs_diff(la[i]["first64_last_pos"], lb[i]["first64_last_pos"])
        flag = ""
        if d > args.threshold and first_diverging is None:
            first_diverging = i
            flag = "  <== FIRST DIVERGENCE"
        print(f"layer {i:>2}          first64_maxabs_diff={d:.4e}   {_fmt_stats_delta(la[i], lb[i])}{flag}")

    # engram gates
    ega, egb = a.get("engram", {}), b.get("engram", {})
    for lid in sorted(set(ega) & set(egb)):
        ga = ega[lid].get("gate", {})
        gb = egb[lid].get("gate", {})
        if ga and gb:
            for key in ("gate_mean", "gate_max", "dot_max_abs", "value_max_abs"):
                if key in ga and key in gb:
                    print(f"engram L{lid} {key:<14} A={ga[key]:+.4g}  B={gb[key]:+.4g}  d={gb[key]-ga[key]:+.4g}")

    # final norm
    if a.get("final_norm") and b.get("final_norm"):
        d = _maxabs_diff(a["final_norm"]["first64_last_pos"], b["final_norm"]["first64_last_pos"])
        print(f"\nfinal_norm       first64_maxabs_diff={d:.4e}   {_fmt_stats_delta(a['final_norm'], b['final_norm'])}")

    # top-8 logits
    ta = a.get("argmax_token", {})
    tb = b.get("argmax_token", {})
    print(f"\nargmax A: id={ta.get('id')} {ta.get('text')!r}")
    print(f"argmax B: id={tb.get('id')} {tb.get('text')!r}")
    print("top-8 A:", [ (t['id'], round(t['logit'], 3)) for t in a.get("logits_top8", []) ])
    print("top-8 B:", [ (t['id'], round(t['logit'], 3)) for t in b.get("logits_top8", []) ])

    print()
    if first_diverging is None:
        print(f"VERDICT: no layer exceeds the first64 threshold {args.threshold:g} "
              f"(dumps agree to that tolerance).")
        return 0
    print(f"VERDICT: first diverging layer = {first_diverging} "
          f"(first64 max-abs diff > {args.threshold:g}).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
