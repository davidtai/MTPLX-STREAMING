"""F15: classify a CANDIDATE-vs-CONTROL greedy divergence at any index (CPU tool).

Given two arms' dumped verify-logits rows (``row_dump.py``) and their token receipts,
this finds the first global position where the two streams differ and runs the repo's own
``classify_divergence`` rule on that position, with the CONTROL arm's row+token in the
``ar_*`` (reference) role and the CANDIDATE arm's row+token in the ``dspark_*`` role.

This is the offline counterpart to the retained runner's cache-only classifier, which can
only classify against the ONE cached AR row (index 297) and raises on any new index. Here
both arms supply a freshly-captured verify row for the contested position, so a divergence
at e.g. global position 480 gets a proven tie / no-tie verdict without any GPU replay.

CPU only. MLX is pinned to the CPU device before the runtime is imported.

Usage:
  python classify_pair.py \
    --control-dir   <dir with rows.json + row-*.npy from the CONTROL arm> \
    --candidate-dir <dir with rows.json + row-*.npy from the CANDIDATE arm> \
    --control-tokens   <CONTROL arm receipt: *.jsonl (accepted) or *.rejected-output.json> \
    --candidate-tokens <CANDIDATE arm receipt> \
    [--index N] [--tie-ulps K] [--out result.json]
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- CPU pin BEFORE the runtime import (MLX defaults to Metal) ---------------
import mlx.core as mx

mx.set_default_device(mx.cpu)

import numpy as np

# Default pinned runtime worktree (the benchmark's exact sources). Overridable with
# --runtime-path or by putting it on PYTHONPATH; classify_divergence must come from here.
_DEFAULT_RUNTIME = "/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-run-d5f15e7a"


# ---------------------------------------------------------------------------
# rule string -- exact transcription of the pinned
# scripts/deepseek_v41/ab_decode_env_levers.py::_dspark_divergence_rule, used as a
# fallback when the pinned module cannot be reused. (The test pins the two equal.)
# ---------------------------------------------------------------------------
def _local_dspark_divergence_rule(d: dict) -> str:
    unavailable = d.get("unavailable_logits")
    if unavailable:
        return f"unclassified (missing {', '.join(unavailable)} logits)"
    band = d.get("tie_band_used", d.get("tie_margin"))
    arc = d.get("ar_contested_margin")
    dsc = d.get("dspark_contested_margin")
    within = d.get("deltas_within_tie_band")
    if d.get("rows_consistent") is False:
        return "none (row argmax != credited token -> row did not produce it)"
    if within is False:
        return "none (contested delta > band / non-finite -> not rounding)"
    fired = []
    if within and band is not None and arc is not None and dsc is not None and min(arc, dsc) < band:
        fired.append("near_tie_by_band(a)")
    if within and d.get("rounding_class_by_delta"):
        fired.append("rounding_class_by_delta(c)")
    return "+".join(fired) if fired else "none"


def _resolve_runtime(runtime_path: str):
    """Ensure the pinned runtime is importable and return ``classify_divergence``."""
    rp = os.path.abspath(runtime_path)
    if rp not in sys.path:
        sys.path.insert(0, rp)
    from mtplx.models.deepseek_v41_dspark_decode import classify_divergence  # noqa: E402

    return classify_divergence


def _resolve_rule_fn(runtime_path: str):
    """Best-effort reuse of the pinned ``_dspark_divergence_rule``; local copy otherwise.

    Returns ``(fn, source)`` where source is ``"pinned"`` or ``"local-copy"``. The pinned
    module is loaded from an explicit file path (as run_full.py does), so nothing is
    added to ``sys.modules`` under a name that could shadow a real import.
    """
    ab_path = os.path.join(runtime_path, "scripts", "deepseek_v41", "ab_decode_env_levers.py")
    try:
        spec = importlib.util.spec_from_file_location("_f15_pinned_ab", ab_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fn = getattr(mod, "_dspark_divergence_rule", None)
            if callable(fn):
                return fn, "pinned"
    except Exception:
        pass
    return _local_dspark_divergence_rule, "local-copy"


# ---------------------------------------------------------------------------
# receipts + rows
# ---------------------------------------------------------------------------
def _is_int_list(x: Any) -> bool:
    return (
        isinstance(x, list)
        and len(x) > 0
        and all(isinstance(v, int) and not isinstance(v, bool) for v in x)
    )


def _receipt_path(path: str) -> str:
    """Resolve a token receipt: a file as given, or auto-discover inside a directory
    (a ``*.rejected-output.json`` takes precedence, else the primary ``*.jsonl``)."""
    p = Path(path)
    if p.is_file():
        return str(p)
    if p.is_dir():
        rej = sorted(glob.glob(str(p / "*.rejected-output.json")))
        if len(rej) == 1:
            return rej[0]
        if len(rej) > 1:
            raise SystemExit(f"{path}: multiple *.rejected-output.json; pass the file explicitly")
        jsonl = [
            x
            for x in sorted(glob.glob(str(p / "*.jsonl")))
            if not x.endswith((".os.jsonl", ".passes.jsonl"))
        ]
        if len(jsonl) == 1:
            return jsonl[0]
        raise SystemExit(f"{path}: could not pick a unique receipt ({len(jsonl)} candidate .jsonl)")
    raise SystemExit(f"{path}: not a file or directory")


def load_arm_tokens(path: str) -> Tuple[List[int], str, str]:
    """Return ``(token_ids, receipt_path, source_key)`` -- the arm's OWN DSpark stream.

    Prefers ``dspark.token_ids`` (the top-level ``token_ids`` can differ from it), never
    ``dspark.ar_reference.token_ids`` (that is the AR reference, not this arm's stream).
    """
    receipt = _receipt_path(path)
    records: List[Any] = []
    if receipt.endswith(".jsonl"):
        with open(receipt) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        obj = json.load(open(receipt))
        records = obj if isinstance(obj, list) else [obj]

    for rec in records:
        if isinstance(rec, dict):
            ds = rec.get("dspark")
            if isinstance(ds, dict) and _is_int_list(ds.get("token_ids")):
                return list(ds["token_ids"]), receipt, "dspark.token_ids"
    for rec in records:
        if isinstance(rec, dict) and _is_int_list(rec.get("token_ids")):
            return list(rec["token_ids"]), receipt, "token_ids"
    raise SystemExit(f"{receipt}: no 1-D int token_ids found (looked under dspark.token_ids, token_ids)")


def load_rows_index(dir_path: str) -> Dict[int, Dict[str, int]]:
    """Load ``rows.json`` (str-or-int keys) if present; ``{}`` otherwise (the .npy rows
    are sufficient to classify -- rows.json only carries token/m metadata for validation)."""
    p = Path(dir_path) / "rows.json"
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text())
    return {int(k): v for k, v in raw.items()}


def load_row(dir_path: str, gpos: int) -> np.ndarray:
    """Load ``<dir>/row-<gpos>.npy`` as a 1-D float array; a clear error if absent."""
    p = Path(dir_path) / f"row-{gpos}.npy"
    if not p.is_file():
        raise SystemExit(
            f"{p} not found: re-run that arm with global position {gpos} in "
            f"MTPLX_DSV41_F15_ROW_INDICES (e.g. a range around it) so the row is dumped"
        )
    return np.asarray(np.load(p)).reshape(-1)


def _top5(row: np.ndarray) -> List[List[float]]:
    n = int(row.size)
    k = min(5, n)
    top = np.argsort(row)[::-1][:k]
    return [[int(i), float(row[int(i)])] for i in top]


def _first_divergence(a: List[int], b: List[int]) -> Optional[int]:
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Classify a candidate-vs-control divergence at any index")
    ap.add_argument("--control-dir", required=True, help="dir with rows.json + row-*.npy (CONTROL arm)")
    ap.add_argument("--candidate-dir", required=True, help="dir with rows.json + row-*.npy (CANDIDATE arm)")
    ap.add_argument("--control-tokens", required=True, help="CONTROL arm receipt (.jsonl or .rejected-output.json) or its dir")
    ap.add_argument("--candidate-tokens", required=True, help="CANDIDATE arm receipt or its dir")
    ap.add_argument("--index", type=int, default=None, help="classify this global position instead of the first divergence")
    ap.add_argument("--tie-ulps", type=int, default=None, help="override the magnitude-aware tie-band ulp multiplier k")
    ap.add_argument("--runtime-path", default=_DEFAULT_RUNTIME, help="pinned runtime worktree that provides classify_divergence")
    ap.add_argument("--out", default=None, help="write the result JSON here")
    args = ap.parse_args(argv)

    classify_divergence = _resolve_runtime(args.runtime_path)
    rule_fn, rule_source = _resolve_rule_fn(args.runtime_path)

    control_tokens, control_receipt, control_src = load_arm_tokens(args.control_tokens)
    candidate_tokens, candidate_receipt, candidate_src = load_arm_tokens(args.candidate_tokens)

    first = _first_divergence(control_tokens, candidate_tokens)
    if args.index is not None:
        index = int(args.index)
        index_source = "explicit"
    else:
        if first is None:
            summary = {
                "class": "identical",
                "note": "control and candidate streams are identical over the compared range",
                "n_positions_compared": min(len(control_tokens), len(candidate_tokens)),
                "control_receipt": control_receipt,
                "candidate_receipt": candidate_receipt,
            }
            print(json.dumps(summary, indent=2))
            if args.out:
                Path(args.out).write_text(json.dumps(summary, indent=2))
            return 0
        index = first
        index_source = "first_divergence"

    # Validate the two dirs' rows.json (if present) line up with the receipts at `index`.
    ctrl_idx = load_rows_index(args.control_dir)
    cand_idx = load_rows_index(args.candidate_dir)
    for name, meta_map, toks in (
        ("control", ctrl_idx, control_tokens),
        ("candidate", cand_idx, candidate_tokens),
    ):
        meta = meta_map.get(index)
        if meta is not None and index < len(toks) and int(meta.get("token")) != int(toks[index]):
            raise SystemExit(
                f"{name}-dir rows.json token {meta.get('token')} at gpos {index} != receipt token "
                f"{toks[index]}: the dir and the tokens are from different runs (swapped args?)"
            )

    control_row = load_row(args.control_dir, index)      # AR (reference) role
    candidate_row = load_row(args.candidate_dir, index)  # DSpark role

    control_token = int(control_tokens[index]) if index < len(control_tokens) else None
    candidate_token = int(candidate_tokens[index]) if index < len(candidate_tokens) else None

    kwargs = dict(
        index=index,
        ar_token=control_token,
        dspark_token=candidate_token,
        ar_logits_row=control_row,
        dspark_logits_row=candidate_row,
    )
    if args.tie_ulps is not None:
        kwargs["tie_ulps"] = int(args.tie_ulps)
    d = classify_divergence(**kwargs)
    rule = rule_fn(d)

    # positions that still differ after the first divergence (context only).
    n_cmp = min(len(control_tokens), len(candidate_tokens))
    diff_after = sum(1 for i in range(index + 1, n_cmp) if control_tokens[i] != candidate_tokens[i])

    result = {
        "index": index,
        "index_source": index_source,
        "first_divergence_index": first,
        "class": d["class"],
        "rule": rule,
        "rule_source": rule_source,
        "control_token": control_token,
        "candidate_token": candidate_token,
        "ar_contested_margin": d.get("ar_contested_margin"),
        "dspark_contested_margin": d.get("dspark_contested_margin"),
        "tie_band_used": d.get("tie_band_used"),
        "tie_margin": d.get("tie_margin"),
        "tie_ulps": d.get("tie_ulps"),
        "peak_contested_logit": d.get("peak_contested_logit"),
        "delta_at_ar_token": d.get("delta_at_ar_token"),
        "delta_at_dspark_token": d.get("delta_at_dspark_token"),
        "max_abs_logit_delta": d.get("max_abs_logit_delta"),
        "rows_consistent": d.get("rows_consistent"),
        "deltas_within_tie_band": d.get("deltas_within_tie_band"),
        "rounding_class_by_delta": d.get("rounding_class_by_delta"),
        "unavailable_logits": d.get("unavailable_logits"),
        "control_top5": _top5(control_row),
        "candidate_top5": _top5(candidate_row),
        "positions_differ_after_first": diff_after,
        "n_positions_compared": n_cmp,
        "control": {
            "receipt": control_receipt,
            "tokens_source": control_src,
            "dir": os.path.abspath(args.control_dir),
            "row": f"row-{index}.npy",
            "rows_json_meta": ctrl_idx.get(index),
        },
        "candidate": {
            "receipt": candidate_receipt,
            "tokens_source": candidate_src,
            "dir": os.path.abspath(args.candidate_dir),
            "row": f"row-{index}.npy",
            "rows_json_meta": cand_idx.get(index),
        },
        "classify_divergence": d,
    }

    verdict = {
        "tie_flip": "TIE FLIP (acceptable, rounding-class)",
        "divergent": "DIVERGENT (NOT a tie flip -- real difference)",
        "unclassified": "UNCLASSIFIED (a row was missing/empty -- tie unproven)",
    }.get(d["class"], d["class"])
    print(
        f"[f15] control@{index}={control_token} vs candidate@{index}={candidate_token} "
        f"=> {verdict}\n"
        f"[f15]   rule={rule} (source={rule_source}) "
        f"ar_contested={result['ar_contested_margin']} dsp_contested={result['dspark_contested_margin']} "
        f"tie_band_used={result['tie_band_used']}\n"
        f"[f15]   Δ@ar_tok={result['delta_at_ar_token']} Δ@dsp_tok={result['delta_at_dspark_token']} "
        f"max|Δlogit|={result['max_abs_logit_delta']} rows_consistent={result['rows_consistent']}\n"
        f"[f15]   first_divergence={first} positions_differ_after={diff_after}/{n_cmp}",
        flush=True,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"[f15] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
