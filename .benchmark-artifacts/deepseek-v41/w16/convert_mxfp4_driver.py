#!/usr/bin/env python3
"""W16 native-mxfp4 full-bank driver for DeepSeek-V4.1-Flash.

Repacks the source FP4 (E2M1 + E8M0) routed experts into a native **mxfp4**
streamed expert bank.  This is a *lossless repack*: on mlx 0.32.2,
``mx.quantize(fp32_dequant(source_fp4), group_size=32, bits=4, mode="mxfp4")``
reproduces the source E2M1 codes + E8M0 scales bit-for-bit (W9
``bank_mx_probe.json``: ``bit_exact_vs_source = true`` over 191 experts x 3
weights).  So the driver could equivalently transcode nibbles directly; it goes
through ``mx.quantize`` so the on-disk framing is exactly MLX's native mxfp4
output (what the runtime dequantizes) and re-verifies bit-exactness per sample.

Residents / config / encoding / engram / tokenizer are **hardlinked** from the
q2 artifact (byte-identical; the q8 dense residents do not depend on the routed
codec), so this driver writes ONLY ``experts.bin`` (+ manifests at finalize).

Record framing (mlx 0.32.2 mxfp4; W15 owns the canonical manifest definition):
  per projection gate_proj(w1) / up_proj(w3) / down_proj(w2):
    <proj>.weight  U32 LE  [out, in/8]   (8 x 4-bit E2M1 codes per word)
    <proj>.scales  U8      [out, in/32]  (E8M0 exponent byte; 2**(b-127))
  record = 3*(packed + scales) = 18,800,640 B, no biases.
  record_index(layer L, expert e) = L*384 + e ; offset = index * record_bytes.

CPU only, nice -n 19, RSS-bounded (one expert at a time), resumable per source
shard (journal OUTSIDE the artifact).  Writes are pwrite at deterministic
offsets to a pre-sized experts.bin; atomic rename of manifests at finalize.

Usage:
  python convert_mxfp4_driver.py write   --src <src> --out <artifact> --index <idx.json>
  python convert_mxfp4_driver.py finalize --src <src> --out <artifact> --index <idx.json> --config <config.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w16")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)

from mtplx import deepseek_v41_convert as dc  # noqa: E402

# ---- mxfp4 record geometry (fixed by mlx 0.32.2; asserted against W15 spec) --
MXFP4_BITS = 4
MXFP4_GROUP = 32
N_ROUTED_EXPERTS = dc.N_ROUTED_EXPERTS  # 384
ROUTED_LAYERS = list(dc.ROUTED_LAYERS)  # 0..39
MODEL_KEY = "deepseek-v41-flash-expert-mxfp4"

# component order mirrors the affine COMPONENTS minus biases
MXFP4_COMPONENTS = (
    "gate_proj.weight", "gate_proj.scales",
    "up_proj.weight", "up_proj.scales",
    "down_proj.weight", "down_proj.scales",
)


def mxfp4_record_bytes(hidden=dc.HIDDEN_SIZE, inter=dc.MOE_INTERMEDIATE,
                       bits=MXFP4_BITS, group=MXFP4_GROUP) -> int:
    params = 3 * hidden * inter
    packed = params * bits // 8
    scales = params // group  # 1 byte E8M0 per group, no bias
    return packed + scales


MXFP4_RECORD_BYTES = mxfp4_record_bytes()  # 18_800_640


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sha(buf: bytes) -> str:
    return hashlib.sha256(buf).hexdigest()


# --------------------------------------------------------------------------
# mxfp4 repack of one expert (streams one expert; peak RSS ~ a few hundred MB)
# --------------------------------------------------------------------------
def repack_expert_mxfp4(f32_by_proj: dict[str, np.ndarray]):
    """Return (record_blobs, segment_meta) for one expert under the mxfp4 framing."""
    blobs: list[bytes] = []
    seg_meta: list[dict] = []
    cursor = 0
    for proj in ("gate_proj", "up_proj", "down_proj"):
        w = mx.array(np.ascontiguousarray(f32_by_proj[proj], dtype=np.float32))
        packed, scales = mx.quantize(w, group_size=MXFP4_GROUP, bits=MXFP4_BITS, mode="mxfp4")
        mx.eval(packed, scales)
        pw = np.ascontiguousarray(np.array(packed)).astype("<u4").tobytes()
        sc = np.ascontiguousarray(np.array(scales)).astype(np.uint8).tobytes()
        for leaf, dtype, shp, part in (
            ("weight", "U32", tuple(packed.shape), pw),
            ("scales", "U8", tuple(scales.shape), sc),
        ):
            seg_meta.append({
                "component": f"{proj}.{leaf}", "leaf_name": f"{proj}.{leaf}",
                "dtype": dtype, "shape": list(shp), "rel_offset": cursor, "length": len(part),
            })
            blobs.append(part)
            cursor += len(part)
        del packed, scales, w
    if cursor != MXFP4_RECORD_BYTES:
        raise RuntimeError(f"record built {cursor} B != expected {MXFP4_RECORD_BYTES}")
    return blobs, seg_meta


def write_record(exp_fd: int, record_index: int, blobs: list[bytes]) -> tuple[int, str]:
    offset = record_index * MXFP4_RECORD_BYTES
    payload = b"".join(blobs)
    if len(payload) != MXFP4_RECORD_BYTES:
        raise RuntimeError(f"record {record_index}: {len(payload)} != {MXFP4_RECORD_BYTES}")
    pos = offset
    view = memoryview(payload)
    while view:
        n = os.pwrite(exp_fd, view, pos)
        if n <= 0:
            raise RuntimeError("short pwrite")
        pos += n
        view = view[n:]
    return offset, _sha(payload)


# --------------------------------------------------------------------------
# journal (resumable, OUTSIDE the artifact)
# --------------------------------------------------------------------------
def journal_dir() -> Path:
    d = REPO / ".benchmark-artifacts" / "deepseek-v41" / "w16" / "journal"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_journal() -> dict:
    out: dict[str, dict] = {}
    for p in sorted(journal_dir().glob("shard-*.json")):
        d = json.loads(p.read_text())
        out[int(d["srcidx"])] = d
    return out


def save_journal(data: dict) -> None:
    p = journal_dir() / f"shard-{int(data['srcidx']):05d}.json"
    tmp = p.with_suffix(".partial")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, p)


def shard_path(src: Path, srcidx: int) -> Path:
    return src / f"model-{srcidx:05d}-of-00048.safetensors"


# --------------------------------------------------------------------------
# one source shard -> mxfp4 records for its bank experts
# --------------------------------------------------------------------------
def process_shard(srcidx: int, path: Path, exp_fd: int) -> dict:
    header, data_start = dc.read_safetensors_header(str(path))
    entries = dc.tensor_entries(header)
    fd = os.open(str(path), os.O_RDONLY)
    records_meta: list[dict] = []
    src_bytes = 0
    t0 = time.time()
    try:
        groups: dict[tuple[int, int], dict[str, dc.TensorEntry]] = {}
        for name, entry in entries.items():
            if dc.is_bank_expert(name):
                m = name.split(".")
                groups.setdefault((int(m[1]), int(m[4])), {})[f"{m[5]}.{m[6]}"] = entry
        for (layer, expert) in sorted(groups):
            if layer not in ROUTED_LAYERS:
                continue
            g = groups[(layer, expert)]
            f32_by_proj: dict[str, np.ndarray] = {}
            for proj, w in dc.PROJ_TO_SOURCE_W.items():
                we = g[f"{w}.weight"]; se = g[f"{w}.scale"]
                packed = np.frombuffer(dc.read_tensor_raw(fd, data_start, we), dtype=np.uint8).reshape(we.shape)
                scale = np.frombuffer(dc.read_tensor_raw(fd, data_start, se), dtype=np.uint8).reshape(se.shape)
                src_bytes += (we.end - we.begin) + (se.end - se.begin)
                f32_by_proj[proj] = dc.dequant_fp4(packed, scale)
            blobs, seg = repack_expert_mxfp4(f32_by_proj)
            rec_index = ROUTED_LAYERS.index(layer) * N_ROUTED_EXPERTS + expert
            offset, sha = write_record(exp_fd, rec_index, blobs)
            records_meta.append({"layer": layer, "expert": expert, "index": rec_index,
                                 "sidecar_offset": offset, "sha256": sha, "segments": seg})
            del f32_by_proj, blobs
            # bound RSS: release the mlx CPU buffer pool between experts
            mx.clear_cache()
    finally:
        os.close(fd)
    dt = time.time() - t0
    rate = (src_bytes / 1e6) / dt if dt > 0 else 0.0
    log(f"shard {srcidx:05d}: {len(records_meta)} experts, {src_bytes/1024**3:.2f} GiB "
        f"source in {dt:.1f}s ({rate:.0f} MB/s)")
    return {"srcidx": srcidx, "source_file": path.name,
            "source_sha256": _source_sha(path),
            "records": records_meta, "src_bytes": src_bytes, "seconds": dt}


def _source_sha(path: Path):
    meta = path.parent / ".cache" / "huggingface" / "download" / (path.name + ".metadata")
    try:
        lines = meta.read_text().splitlines()
        return lines[1].strip() if len(lines) >= 2 else None
    except OSError:
        return None


def parse_shards(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


# --------------------------------------------------------------------------
def cmd_write(args) -> int:
    src = args.src.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads(args.index.read_text())["weight_map"]
    names_by_shard: dict[str, set[str]] = {}
    for name, shard in index.items():
        names_by_shard.setdefault(shard, set()).add(name)

    num_records = len(ROUTED_LAYERS) * N_ROUTED_EXPERTS
    want = num_records * MXFP4_RECORD_BYTES
    log(f"target: {num_records} records, record={MXFP4_RECORD_BYTES} B, "
        f"bank={want/1024**3:.1f} GiB ({want/1e9:.1f} GB)")

    exp_path = out / "experts.bin"
    exp_fd = os.open(str(exp_path), os.O_RDWR | os.O_CREAT, 0o644)
    if os.fstat(exp_fd).st_size < want:
        os.ftruncate(exp_fd, want)

    state = load_journal()
    done_records = sum(len(d.get("records", [])) for d in state.values())
    shard_indices = parse_shards(args.shards)
    t_start = time.time()
    try:
        for srcidx in shard_indices:
            if srcidx in state:
                continue
            path = shard_path(src, srcidx)
            expected = names_by_shard.get(path.name, set())
            if not expected:
                save_journal({"srcidx": srcidx, "source_file": path.name,
                              "source_sha256": None, "records": [], "src_bytes": 0, "seconds": 0})
                continue
            if not path.is_file():
                raise SystemExit(f"source shard missing: {path}")
            data = process_shard(srcidx, path, exp_fd)
            os.fsync(exp_fd)
            save_journal(data)
            done_records += len(data["records"])
            frac = done_records / num_records
            elapsed = time.time() - t_start
            eta = (elapsed / frac - elapsed) if frac > 0 else 0
            written = done_records * MXFP4_RECORD_BYTES
            mbps = (written / 1e6) / elapsed if elapsed > 0 else 0
            log(f"progress: {done_records}/{num_records} records ({100*frac:.1f}%), "
                f"{written/1024**3:.1f} GiB written, {mbps:.0f} MB/s, "
                f"elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m")
    finally:
        os.fsync(exp_fd)
        os.close(exp_fd)
    log(f"WRITE DONE: {done_records}/{num_records} records")
    return 0


def cmd_finalize(args) -> int:
    """Build expert-manifest.json + conversion-manifest.json for the mxfp4 bank.

    Reuses W15's committed ``build_mxfp4_manifest`` (identical record framing to
    this driver — verified) against the experts.bin THIS driver wrote.

    The resident section is built from the ACTUAL resident shards present in
    <out> at finalize time (headers + real-file sha256), NOT a preserved sibling
    manifest -- W18 replaces the resident shards (q8 -> mxfp8) and W19 the engram
    banks in place, so the shipped manifest must match the shipped files.
    Re-runnable: run once when the bank completes, again after W18/W19 land.
    """
    out = args.out.expanduser().resolve()
    from mtplx.expert_streaming_models import get_model_spec
    from mtplx.expert_manifest import verify_expert_manifest
    from types import SimpleNamespace
    sys.path.insert(0, str(REPO / "scripts"))
    import convert_deepseek_v41_streamed as conv

    spec = get_model_spec(MODEL_KEY)
    state = load_journal()
    all_records: list[dict] = []
    for d in sorted(state.values(), key=lambda x: x["srcidx"]):
        all_records.extend(d.get("records", []))
    want = len(ROUTED_LAYERS) * N_ROUTED_EXPERTS
    if len(all_records) != want:
        raise SystemExit(f"journal has {len(all_records)} records, expected {want}; write incomplete")

    # Resident section from the shards actually on disk right now (re-scanned each
    # finalize so it tracks the W18 mxfp8 resident / MTP replacements).
    shard_names = sorted(p.name for p in out.glob("model-*.safetensors"))
    if not shard_names:
        raise SystemExit(f"no resident model-*.safetensors in {out}")
    resident_tensors, shard_infos, _weight_map, resident_total = \
        conv.build_resident_index_and_manifest_inputs(out, shard_names)
    resident_manifest = SimpleNamespace(
        resident_tensors=resident_tensors, shards=shard_infos,
        resident_tensor_bytes=resident_total, model_key="present-residents-scan",
    )
    log(f"finalize: {len(all_records)} records; resident scan {len(shard_names)} shards, "
        f"{len(resident_tensors)} tensors, {resident_total/1024**3:.2f} GiB")
    manifest = conv.build_mxfp4_manifest(out, resident_manifest, spec, all_records,
                                         require_pinned=args.require_pinned)
    log(f"manifest: {len(manifest.records)} records, "
        f"resident={manifest.resident_tensor_bytes/1024**3:.2f} GiB, "
        f"routed={manifest.routed_expert_bytes/1024**3:.2f} GiB, digest={manifest.manifest_sha256[:12]}")
    log("verifying (records + shards + sidecar) ...")
    report = verify_expert_manifest(manifest, out, verify_records=True,
                                    verify_shard_hashes=True, verify_sidecar_hash=True)
    log(f"verify report: {json.dumps(report)}")
    log("DONE (finalize)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("write", "finalize"):
        s = sub.add_parser(name)
        s.add_argument("--src", type=Path, required=True)
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--index", type=Path, required=True)
        s.add_argument("--config", type=Path, default=None)
        s.add_argument("--shards", default="1-46")
        s.add_argument("--require-pinned", action="store_true",
                       help="finalize: also pin artifact/resident total bytes to the spec "
                            "(off by default so finalize stays re-runnable while W18/W19 "
                            "change the resident/engram byte totals)")
    args = ap.parse_args()
    return cmd_write(args) if args.cmd == "write" else cmd_finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
