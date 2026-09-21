"""F38: transcode the DeepSeek-V4.1 mxfp4 routed-expert bank into a tcq3 (eschamoe K=3 trellis) bank on the GPU.

Runs INSIDE gpu_window.sh (single venv-python process, GPU lock held by the guard) with a wall-time budget so the
production server is restored between chunks; resumable: ``progress.json`` in the output dir records the next record
index and the per-record sha256 of everything written so far.

Output artifact (``--out-dir``):
  experts.bin            fixed-size records in the SOURCE manifest's (layer, expert) order; per record the segments
                         gate_proj.code  int16 [in/16, out/16, 48]   (eschamoe tiles of W[in,out], see tcq_encode.py)
                         gate_proj.rout  f16   [out]
                         up_proj.code / up_proj.rout, down_proj.code / down_proj.rout
                         (rin == 1 for every projection and is not stored)
  expert-manifest.json   the source manifest with ``quantization`` replaced by the tcq3 description, ``records``
                         re-pointed at the new offsets/lengths/dtypes/shapes/sha256, ``artifact`` renamed; written
                         when the last record lands (``--finalize`` also writes it from a complete progress file).
  conversion-receipt.json encoder parameters, git revision, per-chunk timings.
Everything else the runtime needs (dense shards, config, tokenizer) stays in the source artifact directory; the
runtime integration points its expert bank at this directory (symlinks for the rest are made by ``--link-rest``).

CPU-only pieces (record layout, offsets, manifest, progress) are testable without Metal via ``--dry-run``: the
encoder is replaced by a deterministic stand-in and no source bytes are read.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np


def nocache(fh) -> None:
    """Bypass the unified buffer cache for a streaming file (macOS F_NOCACHE): the guard's ceiling counts file
    cache as physical memory, and 288 GB of reads + 204 GB of writes through the cache killed chunk 1 at 102.5 GiB."""
    fcntl.fcntl(fh.fileno(), fcntl.F_NOCACHE, 1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

COMPONENTS = ("gate_proj", "up_proj", "down_proj")
NW = 48


def load_source_manifest(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def record_layout(src_rec: dict) -> list:
    """Segments of ONE tcq3 record (relative offsets), derived from the source record's projection shapes.

    Source weight shape is [out, in/8] (mxfp4 U32); the eschamoe code covers W[in, out] as [in/16, out/16, 48] int16.
    """
    src = {s["component"]: s for s in src_rec["segments"]}
    segs = []
    off = 0
    for comp in COMPONENTS:
        out_f, in8 = src[f"{comp}.weight"]["shape"]
        in_f = in8 * 8
        assert in_f % 128 == 0 and out_f % 128 == 0, (comp, in_f, out_f)
        nI, nJ = in_f // 16, out_f // 16
        code_len = nI * nJ * NW * 2
        segs.append({"component": f"{comp}.code", "offset": off, "length": code_len, "dtype": "I16",
                     "shape": [nI, nJ, NW]})
        off += code_len
        segs.append({"component": f"{comp}.rout", "offset": off, "length": out_f * 2, "dtype": "F16",
                     "shape": [out_f]})
        off += out_f * 2
    return segs


def record_bytes(segs: list) -> int:
    return segs[-1]["offset"] + segs[-1]["length"]


def progress_path(out_dir: str) -> str:
    return os.path.join(out_dir, "progress.json")


def load_progress(out_dir: str, n_records: int) -> dict:
    p = progress_path(out_dir)
    if os.path.exists(p):
        with open(p) as f:
            prog = json.load(f)
        assert prog["n_records"] == n_records, "progress file belongs to a different bank"
        return prog
    return {"n_records": n_records, "next": 0, "sha256": [None] * n_records, "chunks": []}


def save_progress(out_dir: str, prog: dict) -> None:
    tmp = progress_path(out_dir) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(prog, f)
    os.replace(tmp, progress_path(out_dir))


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


class DryEncoder:
    """Deterministic stand-in for the GPU encoder (CPU-only tests): code = hash-derived int16, rout = ones."""

    def encode(self, layer: int, expert: int, comp: str, nI: int, nJ: int, out_f: int) -> tuple:
        rng = np.random.default_rng(layer * 1000003 + expert * 1009 + COMPONENTS.index(comp))
        code = rng.integers(-32768, 32767, size=(nI, nJ, NW), dtype=np.int16)
        rout = np.ones(out_f, np.float16)
        return code, rout


class GpuEncoder:
    """Reads the mxfp4 record, dequantizes on the GPU, runs the F38 trellis encoder per projection."""

    def __init__(self, src_manifest: dict, experts_bin: str, W: int, rounds: int, batch: int):
        import mlx.core as mx
        import tcq_encode_metal as gm

        self.mx, self.gm = mx, gm
        self.W, self.rounds, self.batch = W, rounds, batch
        self.fh = open(experts_bin, "rb", buffering=0)
        nocache(self.fh)
        self.timings = {"read_s": 0.0, "prep_s": 0.0, "encode_s": 0.0}

    def _read(self, seg: dict) -> np.ndarray:
        self.fh.seek(seg["offset"])
        buf = self.fh.read(seg["length"])
        assert len(buf) == seg["length"], "short read"
        dt = "<u4" if seg["dtype"] == "U32" else np.uint8
        return np.frombuffer(buf, dtype=dt).reshape(seg["shape"])

    def encode_record(self, src_rec: dict) -> dict:
        mx, gm = self.mx, self.gm
        src = {s["component"]: s for s in src_rec["segments"]}
        out = {}
        for comp in COMPONENTS:
            t0 = time.perf_counter()
            wq = self._read(src[f"{comp}.weight"])
            sc = self._read(src[f"{comp}.scales"])
            t1 = time.perf_counter()
            W_ref = mx.dequantize(mx.array(wq), mx.array(sc), group_size=32, bits=4, mode="mxfp4").astype(mx.float32)
            W_esch = W_ref.T                                          # [in, out]
            res = gm.encode_projection_gpu(W_esch, self.W, self.rounds, self.batch)
            mx.eval(res["code"], res["rout"], res["stats"])
            code = np.array(res["code"]).astype(np.int16)
            rout = np.array(res["rout"]).astype(np.float16)
            ntrunc = int(np.array(res["stats"][:, 0]).sum())
            out[comp] = (code, rout, ntrunc)
            self.timings["read_s"] += t1 - t0
            self.timings["prep_s"] += res["prep_s"]
            self.timings["encode_s"] += res["encode_s"]
        return out


def write_record(fh, base: int, segs: list, parts: dict) -> str:
    """Write one record's segments at absolute offset ``base``; returns the record sha256."""
    h = hashlib.sha256()
    fh.seek(base)
    for seg in segs:
        comp, kind = seg["component"].rsplit(".", 1)
        arr = parts[comp][0] if kind == "code" else parts[comp][1]
        b = np.ascontiguousarray(arr).tobytes()
        assert len(b) == seg["length"], (seg["component"], len(b), seg["length"])
        h.update(b)
        fh.write(b)
    return h.hexdigest()


def build_manifest(src_m: dict, segs_rel: list, rec_bytes: int, shas: list, W: int, rounds: int, out_dir: str) -> dict:
    m = {k: v for k, v in src_m.items() if k not in ("records", "manifest_sha256")}
    m["quantization"] = {"mode": "tcq3", "bits": 3, "K": 3, "tile": 16, "hadamard_block": 128, "rin": 1,
                         "codebook": "eschamoe-hash-3417055213", "encoder": "dsv41-f38-gpu-beam", "beam": W,
                         "bisection_rounds": rounds, "source_quantization": src_m["quantization"]}
    m["artifact"] = os.path.basename(os.path.abspath(out_dir))
    recs = []
    for i, r in enumerate(src_m["records"]):
        base = i * rec_bytes
        segs = []
        for s in segs_rel:
            comp = s["component"].split(".")[0]
            segs.append({"component": s["component"], "tensor": f"layers.{r['layer']}.ffn.experts.{r['expert']}.{comp}",
                         "offset": base + s["offset"], "length": s["length"], "dtype": s["dtype"], "shape": s["shape"]})
        recs.append({"layer": r["layer"], "expert": r["expert"], "logical_bytes": rec_bytes, "sha256": shas[i],
                     "segments": segs})
    m["records"] = recs
    body = json.dumps(m, sort_keys=True).encode()
    m["manifest_sha256"] = hashlib.sha256(body).hexdigest()
    return m


def link_rest(src_dir: str, out_dir: str) -> list:
    made = []
    for name in sorted(os.listdir(src_dir)):
        if name in ("experts.bin", "expert-manifest.json") or name.startswith("."):
            continue
        dst = os.path.join(out_dir, name)
        srcp = os.path.join(src_dir, name)
        if not os.path.lexists(dst):
            # regular files are HARD-linked (same volume, no extra space): the GPU guard's file-cache reclaimer refuses
            # a model directory whose config.json / *.safetensors are symlinks; directories stay symlinks.
            if os.path.isfile(srcp) and not os.path.islink(srcp):
                os.link(srcp, dst)
            else:
                os.symlink(srcp, dst)
            made.append(name)
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default=os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--time-budget-s", type=float, default=3600.0)
    ap.add_argument("--max-records", type=int, default=0, help="stop after this many records in this chunk (0 = budget only)")
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--dry-run", action="store_true", help="CPU stand-in encoder, no source bytes read")
    ap.add_argument("--finalize", action="store_true", help="write the manifest from a complete progress file and exit")
    ap.add_argument("--link-rest", action="store_true", help="symlink the non-expert files of the source artifact")
    ap.add_argument("--progress-every", type=int, default=8)
    args = ap.parse_args()

    src_m = load_source_manifest(os.path.join(args.src_dir, "expert-manifest.json"))
    n = len(src_m["records"])
    segs_rel = record_layout(src_m["records"][0])
    rec_bytes = record_bytes(segs_rel)
    for r in src_m["records"][1:]:
        assert record_layout(r) == segs_rel, "non-uniform record geometry"
    os.makedirs(args.out_dir, exist_ok=True)
    prog = load_progress(args.out_dir, n)
    bin_path = os.path.join(args.out_dir, "experts.bin")

    if args.link_rest:
        print("linked:", link_rest(args.src_dir, args.out_dir), flush=True)

    if args.finalize:
        assert prog["next"] == n and all(prog["sha256"]), f"bank incomplete: next={prog['next']} of {n}"
        m = build_manifest(src_m, segs_rel, rec_bytes, prog["sha256"], args.W, args.rounds, args.out_dir)
        with open(os.path.join(args.out_dir, "expert-manifest.json"), "w") as f:
            json.dump(m, f, indent=1, sort_keys=True)
        print(f"manifest written: {n} records x {rec_bytes} B = {n * rec_bytes / 1e9:.1f} GB", flush=True)
        return 0

    if prog["next"] >= n:
        print("bank already complete; use --finalize", flush=True)
        return 0

    if args.dry_run:
        enc = DryEncoder()
    else:
        enc = GpuEncoder(src_m, os.path.join(args.src_dir, "experts.bin"), args.W, args.rounds, args.batch)

    # pre-size the file once so every chunk writes in place
    with open(bin_path, "ab") as f:
        pass
    if os.path.getsize(bin_path) < n * rec_bytes:
        with open(bin_path, "r+b") as f:
            f.truncate(n * rec_bytes)

    t_start = time.perf_counter()
    done_here = 0
    chunk = {"start_index": prog["next"], "t0": time.time(), "git": git_rev(), "W": args.W, "rounds": args.rounds}
    with open(bin_path, "r+b") as fh:
        nocache(fh)
        i = prog["next"]
        while i < n:
            if (time.perf_counter() - t_start) > args.time_budget_s or (args.max_records and done_here >= args.max_records):
                break
            r = src_m["records"][i]
            if args.dry_run:
                parts = {}
                for comp in COMPONENTS:
                    s = [x for x in segs_rel if x["component"] == f"{comp}.code"][0]
                    nI, nJ, _ = s["shape"]
                    code, rout = enc.encode(r["layer"], r["expert"], comp, nI, nJ, nJ * 16)
                    parts[comp] = (code, rout, 0)
            else:
                parts = enc.encode_record(r)
            sha = write_record(fh, i * rec_bytes, segs_rel, parts)
            prog["sha256"][i] = sha
            i += 1
            done_here += 1
            prog["next"] = i
            if done_here % args.progress_every == 0 or i == n:
                fh.flush()
                os.fsync(fh.fileno())
                save_progress(args.out_dir, prog)
                el = time.perf_counter() - t_start
                extra = "" if args.dry_run else f" read {enc.timings['read_s']:.1f}s prep {enc.timings['prep_s']:.1f}s encode {enc.timings['encode_s']:.1f}s"
                print(f"record {i}/{n} L{r['layer']}E{r['expert']} {el:.1f}s elapsed, {el / done_here:.2f} s/record{extra}", flush=True)
        fh.flush()
        os.fsync(fh.fileno())
    chunk.update({"end_index": prog["next"], "records": done_here, "wall_s": time.perf_counter() - t_start})
    if not args.dry_run:
        chunk["timings"] = dict(enc.timings)
    prog["chunks"].append(chunk)
    save_progress(args.out_dir, prog)
    receipt = {"encoder": "dsv41-f38-gpu-beam", "record_bytes": rec_bytes, "segments": segs_rel,
               "n_records": n, "chunks": prog["chunks"]}
    with open(os.path.join(args.out_dir, "conversion-receipt.json"), "w") as f:
        json.dump(receipt, f, indent=1)
    if prog["next"] == n:
        m = build_manifest(src_m, segs_rel, rec_bytes, prog["sha256"], args.W, args.rounds, args.out_dir)
        with open(os.path.join(args.out_dir, "expert-manifest.json"), "w") as f:
            json.dump(m, f, indent=1, sort_keys=True)
        print("BANK COMPLETE; manifest written", flush=True)
    print(f"chunk done: {done_here} records, next={prog['next']}/{n}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
