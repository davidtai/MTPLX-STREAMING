"""F39: build a FULL 15,360-record tcq3 dry-run artifact over a sparse (truncate-presized) experts.bin.

Exercises the tcq3 reader/manifest and the +tcq staging at real bank SCALE without 204 GB of real data: the bin is
presized with ``truncate`` (holes read as zeros on APFS) and only a handful of records carry F38 DryEncoder codes.
CPU-only, no GPU.  NEVER writes under ~/models — output goes to a caller-provided dir (a scratchpad).

    python build_sparse_dry_artifact.py --out-dir <scratchpad/tcq3-sparse> [--src-manifest <mxfp4 manifest>] \
        [--written 0-7,15359]

The manifest is written via the F38 ``transcode_bank.build_manifest`` (full 15,360 records); holes share the sha256
of a zero-filled record (they ARE zeros).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_TRELLIS = os.path.join(os.path.dirname(HERE), "trellis")
for p in (_TRELLIS,):
    if p not in sys.path:
        sys.path.insert(0, p)
import transcode_bank as tb   # noqa: E402

DEFAULT_SRC = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4/expert-manifest.json")


def _parse_written(spec: str, n: int) -> list:
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(i for i in out if 0 <= i < n)


def build(out_dir: str, src_manifest: str, written_idx: list) -> dict:
    src = json.load(open(src_manifest))
    records = src["records"]
    n = len(records)
    segs_rel = tb.record_layout(records[0])
    rec_bytes = tb.record_bytes(segs_rel)
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, "experts.bin")
    with open(bin_path, "wb") as fh:
        fh.truncate(n * rec_bytes)                       # sparse: holes read as zeros
    enc = tb.DryEncoder()
    # sha256 of a zero-filled record (every hole shares it — holes ARE zeros)
    zero_sha = hashlib.sha256(bytes(rec_bytes)).hexdigest()
    shas = [zero_sha] * n
    written = set(written_idx)
    with open(bin_path, "r+b") as fh:
        for i in written:
            r = records[i]
            parts = {}
            for comp in tb.COMPONENTS:
                seg = next(x for x in segs_rel if x["component"] == f"{comp}.code")
                nI, nJ, _ = seg["shape"]
                code, rout = enc.encode(r["layer"], r["expert"], comp, nI, nJ, nJ * 16)
                parts[comp] = (code, rout, 0)
            shas[i] = tb.write_record(fh, i * rec_bytes, segs_rel, parts)
        fh.flush()
        os.fsync(fh.fileno())
    manifest = tb.build_manifest(src, segs_rel, rec_bytes, shas, 256, 10, out_dir)
    with open(os.path.join(out_dir, "expert-manifest.json"), "w") as f:
        json.dump(manifest, f)
    # symlink the non-expert files of the source artifact (residents load through these)
    tb.link_rest(os.path.dirname(src_manifest), out_dir)
    st = os.stat(bin_path)
    return {"out_dir": out_dir, "n_records": n, "record_bytes": rec_bytes,
            "logical_bytes": n * rec_bytes, "disk_blocks": st.st_blocks, "written": sorted(written)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--src-manifest", default=DEFAULT_SRC)
    ap.add_argument("--written", default="0-7,15359")
    args = ap.parse_args()
    if os.path.abspath(args.out_dir).startswith(os.path.expanduser("~/models")):
        sys.exit("REFUSE: never write a tcq3 artifact under ~/models")
    src = json.load(open(args.src_manifest))
    written = _parse_written(args.written, len(src["records"]))
    info = build(args.out_dir, args.src_manifest, written)
    print(f"sparse tcq3 dry artifact: {info['n_records']} records x {info['record_bytes']} B = "
          f"{info['logical_bytes'] / 1e9:.1f} GB logical, {info['disk_blocks'] * 512 / 1e6:.1f} MB on disk; "
          f"wrote records {info['written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
