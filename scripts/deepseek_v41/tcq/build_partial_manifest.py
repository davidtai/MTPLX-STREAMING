"""F39 (Task 3): build a tcq3 expert-manifest for the REAL, still-writing bank from its progress.json.

The F38 encoder writes the final ``expert-manifest.json`` only when the bank is complete (~2-4 h).  To use the bank
NOW (read-only), build the manifest in a scratchpad from the real record layout + the per-record sha256 that
``progress.json`` already carries for the finished records (0..next-1); records beyond ``next`` are holes (zeros in
the presized experts.bin) and get the zero-record sha as a placeholder.  Nothing is written under ~/models — the
scratchpad dir gets the manifest plus a read-only symlink to the real experts.bin so the reader can open it.

    python build_partial_manifest.py --bank ~/models/DeepSeek-V4.1-...-tcq3 --out-dir <scratchpad/tcq3-partial> \
        [--src-manifest <mxfp4 manifest>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_TRELLIS = os.path.join(os.path.dirname(HERE), "trellis")
if _TRELLIS not in sys.path:
    sys.path.insert(0, _TRELLIS)
import transcode_bank as tb   # noqa: E402

DEFAULT_SRC = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4/expert-manifest.json")


def build(bank_dir: str, out_dir: str, src_manifest: str) -> dict:
    if os.path.abspath(out_dir).startswith(os.path.expanduser("~/models")):
        raise SystemExit("REFUSE: never write a tcq3 manifest under ~/models")
    src = json.load(open(src_manifest))
    records = src["records"]
    n = len(records)
    segs_rel = tb.record_layout(records[0])
    rec_bytes = tb.record_bytes(segs_rel)
    progress = json.load(open(os.path.join(bank_dir, "progress.json")))
    if progress.get("n_records") != n:
        raise SystemExit(f"progress n_records {progress.get('n_records')} != source records {n}")
    prog_shas = progress.get("sha256", [None] * n)
    nxt = int(progress.get("next", 0))
    zero_sha = hashlib.sha256(bytes(rec_bytes)).hexdigest()          # placeholder for unwritten (hole) records
    finished = 0
    shas = []
    for i in range(n):
        s = prog_shas[i] if i < len(prog_shas) else None
        if i < nxt and s:
            shas.append(s)
            finished += 1
        else:
            shas.append(zero_sha)
    os.makedirs(out_dir, exist_ok=True)
    manifest = tb.build_manifest(src, segs_rel, rec_bytes, shas, 256, 10, out_dir)
    manifest["partial"] = {"next": nxt, "finished_records": finished, "source_progress": os.path.join(bank_dir, "progress.json")}
    with open(os.path.join(out_dir, "expert-manifest.json"), "w") as f:
        json.dump(manifest, f)
    # read-only symlink to the real experts.bin so the reader can open records 0..next-1
    link = os.path.join(out_dir, "experts.bin")
    if not os.path.lexists(link):
        os.symlink(os.path.join(bank_dir, "experts.bin"), link)
    # residents: link the non-expert files from the real bank (already symlinks there -> resolve to the source)
    tb.link_rest(bank_dir, out_dir)
    return {"out_dir": out_dir, "n_records": n, "record_bytes": rec_bytes, "next": nxt,
            "finished_records": finished, "experts_bin": os.path.join(bank_dir, "experts.bin")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-tcq3"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--src-manifest", default=DEFAULT_SRC)
    args = ap.parse_args()
    info = build(args.bank, args.out_dir, args.src_manifest)
    print(f"partial tcq3 manifest: {info['n_records']} records, {info['finished_records']} finished (next={info['next']}); "
          f"experts.bin -> {info['experts_bin']} (read-only symlink) in {info['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
