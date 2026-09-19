#!/usr/bin/env python3
"""A/B/A arm driver for the F2 prefetch window (runs INSIDE gpu_window.sh).

WRITE-ONLY / GPU-ONLY: the orchestrator runs this via run_f2_prefetch_window.sh;
the author does not run it.  It drives the REAL extension-bank entrypoint
(``launch_full.py`` -> ``packed/run_full.py``) once per arm, so the runner, growth,
projection and dspark path are byte-for-byte the retained best run.  The only
per-arm difference is one environment flag:

  * control   arms run with the lane OFF (byte-identical to the retained run);
  * candidate arms set ``MTPLX_DSV41_F2_PREFETCH=1``, which the staged
    ``run_full.py`` reads at its post-prefill boundary to call
    ``f2_prefetch_lane.install(runtime, base_launch_bytes=..., ring_records=...)``
    -- the ONE integration edit the window applies to the staged runner (same
    stage-edit pattern extension-bank's stage_full.py used; see the receipt).

Each arm gets a FRESH receipt dir under ``MTPLX_DSV41_F2_OUTROOT`` (this refuses to
overwrite an existing arm dir -- receipts are never overwritten), stores the full
1,024 output token ids, and asserts their sha256 against ``MTPLX_DSV41_F2_DIGEST``.
A digest mismatch or a non-zero arm rc aborts the chain (so the `&&` launcher
reports the first failing arm).

NOTE ON ENVELOPE: each arm is a full model load + 16,384 prefill + 1,024 decode.
Three arms in one window may exceed the time/thermal envelope; run two arms
(``F2_ARMS='candidate control'``) if so.  Single-load A/B/A (install/uninstall the
lane between arms without reloading) is a later refinement; it needs run_full to
expose a per-arm generate seam, which this driver deliberately does not assume.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LAUNCH_FULL = Path(os.environ.get(
    "MTPLX_DSV41_F2_LAUNCH_FULL",
    "/tmp/dsv41-extension-bank-20260919/full-v1/launch_full.py",
))


def _fresh_arm_dir(outroot: Path, index: int, label: str) -> Path:
    d = outroot / f"arm{index}-{label}"
    if d.exists():
        sys.exit(f"[f2-arms] refuse to overwrite existing receipt dir {d}")
    d.mkdir(parents=True)
    return d


def _output_ids_sha256(passes_path: Path) -> tuple[str, int]:
    """Read the final full-decode pass's stored output ids + sha256."""
    last = None
    for line in passes_path.read_text().splitlines():
        row = json.loads(line)
        if row.get("pass") in ("full", "decode", "mtp", "dspark"):
            last = row
    if last is None:
        sys.exit(f"[f2-arms] no decode pass row in {passes_path}")
    return last["output_ids_sha256"], int(last.get("decode_slots_per_layer", -1))


def main() -> None:
    if os.environ.get("_GPU_WINDOW_LOCKED") != "1":
        sys.exit("[f2-arms] must run inside gpu_window.sh (exclusive GPU lock)")
    arms = os.environ.get("MTPLX_DSV41_F2_ARMS", "control candidate control").split()
    outroot = Path(os.environ["MTPLX_DSV41_F2_OUTROOT"])
    digest = os.environ["MTPLX_DSV41_F2_DIGEST"]
    ring_records = os.environ.get("MTPLX_DSV41_F2_RING_RECORDS", "32")
    base_launch = os.environ.get("MTPLX_DSV41_F2_BASE_LAUNCH_BYTES", "")
    base_argv = sys.argv[1:]  # extension-bank BASE_ARGS from the shell script
    outroot.mkdir(parents=True, exist_ok=True)

    summary = {"arms": [], "digest_expected": digest, "ring_records": int(ring_records)}
    for i, label in enumerate(arms):
        if label not in ("control", "candidate"):
            sys.exit(f"[f2-arms] unknown arm {label!r}")
        arm_dir = _fresh_arm_dir(outroot, i, label)
        out_jsonl = arm_dir / f"{label}.jsonl"
        env = os.environ.copy()
        if label == "candidate":
            env["MTPLX_DSV41_F2_PREFETCH"] = "1"
            env["MTPLX_DSV41_F2_RING_RECORDS"] = str(ring_records)
            if base_launch:
                env["MTPLX_DSV41_F2_BASE_LAUNCH_BYTES"] = base_launch
        else:
            env.pop("MTPLX_DSV41_F2_PREFETCH", None)
        argv = [sys.executable, str(_LAUNCH_FULL), *base_argv, "--out", str(out_jsonl)]
        print(f"[f2-arms] arm {i} ({label}) -> {out_jsonl}", flush=True)
        rc = subprocess.call(argv, env=env)
        if rc != 0:
            sys.exit(f"[f2-arms] arm {i} ({label}) exited rc={rc}; chain aborts")
        passes = out_jsonl.with_suffix(".passes.jsonl")
        got, slots = _output_ids_sha256(passes)
        ok = got == digest
        (arm_dir / "arm-receipt.json").write_text(json.dumps(
            {"arm": i, "label": label, "output_ids_sha256": got,
             "digest_match": ok, "decode_slots_per_layer": slots,
             "prefetch": label == "candidate"}, indent=2) + "\n")
        summary["arms"].append({"arm": i, "label": label, "digest_match": ok,
                                "output_ids_sha256": got})
        print(f"[f2-arms]   sha256 {got} digest_match={ok} slots={slots}", flush=True)
        if not ok:
            (outroot / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            sys.exit(f"[f2-arms] arm {i} ({label}) OUTPUT DIVERGED from the native "
                     f"digest; chain aborts (candidate must be bit-identical)")
    (outroot / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[f2-arms] all {len(arms)} arms complete; summary at {outroot/'summary.json'}")


if __name__ == "__main__":
    main()
