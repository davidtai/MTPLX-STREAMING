"""Per-arm readout for the F5 compile window (CPU): digest compare, W120 divergence
verdict, per-cycle verify timing, cycles, and engagement counters.

Reads an arm receipt (the runner's ``--out`` JSON, or its ``.rejected-output.json``
written on a digest change) and prints a one-line headline plus a JSON block.  The
divergence CLASS is read from the receipt's ``dspark.divergence`` (the AB driver's
W120 classifier populated it with the LIVE AR-logits row -- EDIT 1 in the staged
runner keeps that live, so an arbitrary-index divergence is classified, not null).
Never runs the model; a pure receipt reader.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CONTROL_DSPARK_SHA256 = "0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac"


def _load(path: Path) -> dict:
    """Load an arm receipt: a single JSON object (possibly pretty-printed) or the
    last row of a JSON-lines file.  Tries whole-file JSON first so a pretty-printed
    object written to a ``.jsonl`` path still parses."""
    text = path.read_text().strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(x) for x in text.splitlines() if x.strip()]
        if not rows:
            raise
        return rows[-1]


def _verdict(dsp: dict, control_sha: str) -> dict:
    sha = dsp.get("token_ids_sha256")
    identical = sha == control_sha
    div = dsp.get("divergence") or {}
    cls = div.get("class")
    # tie_or_identical policy: identical OR an index-matched tie_flip is acceptable.
    if identical:
        outcome = "identical_to_control"
    elif cls == "tie_flip" and div.get("capture_index_matches_first") is True:
        outcome = "tie_flip_acceptable"
    elif cls in (None, "") and div.get("ar_replay_error"):
        outcome = "UNCLASSIFIED (ar replay error -- AR logits row unavailable)"
    elif cls:
        outcome = f"DIVERGENT ({cls})"
    else:
        outcome = "changed_digest_no_divergence_block"
    return {
        "digest_matches_control": identical,
        "dspark_token_ids_sha256": sha,
        "divergence_class": cls,
        "divergence_index": div.get("divergence_index"),
        "ar_contested_margin": div.get("ar_contested_margin"),
        "dspark_contested_margin": div.get("dspark_contested_margin"),
        "tie_band_used": div.get("tie_band_used"),
        "deltas_within_tie_band": div.get("deltas_within_tie_band"),
        "ar_replay_error": div.get("ar_replay_error"),
        "capture_index_matches_first": div.get("capture_index_matches_first"),
        "outcome": outcome,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="F5 per-arm receipt readout")
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--control-sha", default=CONTROL_DSPARK_SHA256)
    ap.add_argument("--probe-summary", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    rp = Path(args.receipt)
    if not rp.is_file():
        print(f"[f5:{args.arm}] MISSING receipt {rp}")
        return 2
    receipt = _load(rp)
    dsp = receipt.get("dspark", receipt)   # rejected-output stores the same shape

    verdict = _verdict(dsp, args.control_sha)
    pcm = dsp.get("per_cycle_ms") or {}
    engage = {
        "small_stages": receipt.get("small_stages_engagement")
        or dsp.get("small_stages_engagement"),
        "attn_core_compile": dsp.get("attn_core_compile_engagement")
        or receipt.get("attn_core_compile_engagement"),
        "bound_model_levers": receipt.get("bound_model_levers"),
    }
    out = {
        "arm": args.arm,
        "verdict": verdict,
        "cycles": dsp.get("cycles"),
        "verify_calls": dsp.get("verify_calls"),
        "tokens_per_cycle": dsp.get("tokens_per_cycle"),
        "accept_rate": dsp.get("accept_rate"),
        "decode_tok_s": dsp.get("decode_tok_s"),
        "per_cycle_ms": {k: pcm.get(k) for k in ("draft_ms", "verify_ms", "accept_ms", "commit_ms")},
        "phase_time_s": dsp.get("phase_time_s"),
        "headline_pass": dsp.get("headline_pass"),
        "engagement": engage,
        "decoded_text_head": dsp.get("decoded_text_head"),
        "decoded_text_tail": dsp.get("decoded_text_tail"),
        "receipt_path": str(rp),
    }
    if args.probe_summary and Path(args.probe_summary).is_file():
        out["timed_probe_summary"] = json.loads(Path(args.probe_summary).read_text())

    verify_ms = out["per_cycle_ms"].get("verify_ms")
    print(
        f"[f5:{args.arm}] {verdict['outcome']} | "
        f"digest={'MATCH' if verdict['digest_matches_control'] else 'DIFF'} "
        f"class={verdict['divergence_class']} @idx={verdict['divergence_index']} | "
        f"verify_ms/cycle={verify_ms} cycles={out['cycles']} "
        f"tok/cyc={out['tokens_per_cycle']} decode_tok_s={out['decode_tok_s']} | "
        f"small_stages={engage['small_stages']}"
    )
    text = json.dumps(out, indent=2)
    if args.report:
        Path(args.report).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
