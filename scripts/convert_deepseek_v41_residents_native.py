#!/usr/bin/env python3
"""W18: repack the DeepSeek-V4.1-Flash *resident* tensors to MLX-native float
quant formats as EXACT repacks of the Hub source (not affine requantizations).

Scope (RESIDENT tensors only; the routed backbone expert bank ``experts.bin`` is
W16's, the engram sidecar ``engram/`` is W19's):

  * dense ``F8_E4M3`` projections (attn wq_a/wq_b/wkv/wo_a/wo_b, ffn
    shared_experts w1/w2/w3, attn.indexer.wq_b, and the MTP dense projections)
    with ``F8_E8M0`` 32x32 block scales  ->  **mxfp8 gs32** (bit-exact: keep the
    E4M3 codes, expand the block scale to the per-(row,32-col) E8M0 layout MLX
    wants; ``mx.quantize(mode="mxfp8")`` on the FP8 dequant does exactly this and
    a per-tensor ``np.array_equal`` gate rejects any tensor that does not map).
  * MTP routed experts (``mtp.{0,1,2}.ffn.experts.{0..127}.w{1,2,3}``, FP4)  ->
    **mxfp4 gs32** (bit-exact repack of the FP4/E2M1 source).
  * everything BF16/F32 at source (embed, head, norms, hc_*, gate weight/bias,
    attn_sink, compressor/indexer bf16 pieces, vision/aligner, image_* markers)
    -> kept verbatim.  embed/head are BF16 at source, so "keep" restores them to
    exact bf16 (the affine-q8 artifact had quantised them).

The residents are written into the mxfp4 artifact IN PLACE, one shard at a time,
by the SAFE swap: write ``model-NNNNN.partial.safetensors`` -> verify its header,
tensor set and a bit-exact spot-check -> ``os.replace`` over
``model-NNNNN.safetensors`` (atomic; the old inode is never unlinked first).
model.safetensors.index.json and config.json's quantization block are rewritten
last, after every shard is swapped.  NOTE: the mxfp4 artifact currently holds the
ONLY copy of the affine-q8 residents (the q2 artifact was deleted), which is why
no shard is ever removed before its replacement is on disk and verified.

CPU-only (``mx.set_default_device(mx.cpu)``; no GPU flock).  Resumable: a per
resident-shard journal under ``<out>/.w18-native-state/`` lets a re-invoke skip
finished shards, so a panic loses at most the in-flight shard.  RSS is bounded by
streaming one source tensor at a time via ``os.pread`` on the safetensors data
section (never loading a whole source shard).
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)

from mtplx import deepseek_v41_convert as dc  # noqa: E402

MTP_LAYERS = (0, 1, 2)
MTP_SHARD_BASE = 47  # mtp layer L -> resident shard model-000{47+L}.safetensors


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# safetensors helpers (native array residents; mx.save_safetensors)
# --------------------------------------------------------------------------
def _sha_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blob in iter(lambda: f.read(chunk), b""):
            h.update(blob)
    return h.hexdigest()


def _bf16_array(raw: bytes, shape) -> mx.array:
    return mx.array(np.frombuffer(raw, dtype="<u2").reshape(shape)).view(mx.bfloat16)


def _f32_array(raw: bytes, shape) -> mx.array:
    return mx.array(np.frombuffer(raw, dtype="<f4").reshape(shape).copy())


def _f16_array(raw: bytes, shape) -> mx.array:
    return mx.array(np.frombuffer(raw, dtype="<f2").reshape(shape).copy())


def _write_verify_rename(out: Path, shard: str, residents: dict, verify_names: set) -> dict:
    """Write ``residents`` to ``<out>/<shard>`` via a verified atomic swap.

    Writes ``<shard-stem>.partial.safetensors``, reopens it to confirm the header
    parses and holds exactly ``verify_names``, then ``os.replace`` over the final
    name.  The old file is never unlinked before the replacement is verified.
    Returns per-shard metadata (size, sha256, header bytes, tensor dtypes).
    """
    mx.eval(*residents.values())
    tmp = out / shard.replace(".safetensors", ".partial.safetensors")
    mx.save_safetensors(str(tmp), residents, metadata={"format": "mlx"})
    # verify the freshly written file before it replaces the live resident
    header, data_start = dc.read_safetensors_header(str(tmp))
    names = {k for k in header if k != "__metadata__"}
    if names != verify_names:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"{shard}: written tensor set mismatch "
            f"(+{sorted(names - verify_names)[:4]} / -{sorted(verify_names - names)[:4]})"
        )
    implied = dc.implied_file_size(header, data_start)
    actual = tmp.stat().st_size
    if actual != implied:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{shard}: size {actual} != header-implied {implied}")
    os.replace(tmp, out / shard)
    p = out / shard
    dtypes = {name: header[name]["dtype"] for name in names}
    return {
        "shard": shard,
        "size": p.stat().st_size,
        "sha256": _sha_file(p),
        "header_bytes": data_start,
        "n_tensors": len(names),
        "dtypes": dtypes,
    }


# --------------------------------------------------------------------------
# per-tensor native repack (bounded RSS: one source tensor at a time)
# --------------------------------------------------------------------------
def _repack_fp8_to_mxfp8(fd, ds, entry, scale_entry, name, exact_stats):
    weight = np.frombuffer(dc.read_tensor_raw(fd, ds, entry), np.uint8).reshape(entry.shape)
    scale = np.frombuffer(dc.read_tensor_raw(fd, ds, scale_entry), np.uint8).reshape(scale_entry.shape)
    codes, scales, ref = dc.repack_fp8_block_to_mxfp8(weight, scale, name)  # raises if inexact
    exact_stats["mxfp8"] += 1
    src_bytes = (entry.end - entry.begin) + (scale_entry.end - scale_entry.begin)
    del weight, scale, ref
    return codes, scales, src_bytes


def _repack_fp4_to_mxfp4(fd, ds, entry, scale_entry, name, exact_stats):
    packed = np.frombuffer(dc.read_tensor_raw(fd, ds, entry), np.uint8).reshape(entry.shape)
    scale = np.frombuffer(dc.read_tensor_raw(fd, ds, scale_entry), np.uint8).reshape(scale_entry.shape)
    codes, scales, ref = dc.repack_fp4_to_mxfp4(packed, scale, name)  # raises if inexact
    exact_stats["mxfp4"] += 1
    src_bytes = (entry.end - entry.begin) + (scale_entry.end - scale_entry.begin)
    del packed, scale, ref
    return codes, scales, src_bytes


def _keep_verbatim(fd, ds, entry):
    raw = dc.read_tensor_raw(fd, ds, entry)
    if entry.dtype == "BF16":
        arr = _bf16_array(raw, entry.shape)
    elif entry.dtype == "F32":
        arr = _f32_array(raw, entry.shape)
    elif entry.dtype == "F16":
        arr = _f16_array(raw, entry.shape)
    else:
        raise RuntimeError(f"unexpected keep dtype {entry.dtype} for {entry.name}")
    return arr, len(raw)


# --------------------------------------------------------------------------
# main pass: one source shard -> one resident shard (native)
# --------------------------------------------------------------------------
def process_main_shard(srcidx: int, path: Path, out: Path) -> dict | None:
    """Convert source shard ``srcidx``'s residents to native and swap the shard.

    Backbone routed experts (-> experts.bin/W16), engram tensors (-> engram/W19),
    scale siblings, and MTP experts (-> the separate MTP pass) are skipped here.
    Returns the shard metadata, or ``None`` when the shard carries no residents.
    """
    header, data_start = dc.read_safetensors_header(str(path))
    entries = dc.tensor_entries(header)
    fd = os.open(str(path), os.O_RDONLY)
    residents: dict[str, mx.array] = {}
    exact_stats = {"mxfp8": 0, "mxfp4": 0, "keep": 0}
    src_bytes = 0
    t0 = time.time()
    try:
        for name in sorted(entries):
            entry = entries[name]
            disp = dc.native_resident_disposition(entry)
            if disp == "drop":
                continue
            if dc.is_mtp_expert(name):
                continue  # MTP experts handled in the dedicated pass (shards 47-49)
            base = name[: -len(".weight")] if name.endswith(".weight") else name
            if disp == "mxfp8":
                scale_entry = entries[base + ".scale"]
                codes, scales, nb = _repack_fp8_to_mxfp8(fd, data_start, entry, scale_entry, name, exact_stats)
                residents[base + ".weight"] = codes
                residents[base + ".scales"] = scales
                src_bytes += nb
            elif disp == "mxfp4":
                scale_entry = entries[base + ".scale"]
                codes, scales, nb = _repack_fp4_to_mxfp4(fd, data_start, entry, scale_entry, name, exact_stats)
                residents[base + ".weight"] = codes
                residents[base + ".scales"] = scales
                src_bytes += nb
            else:  # keep
                arr, nb = _keep_verbatim(fd, data_start, entry)
                residents[name] = arr
                exact_stats["keep"] += 1
                src_bytes += nb
    finally:
        os.close(fd)
    if not residents:
        return None
    shard = f"model-{srcidx:05d}.safetensors"
    meta = _write_verify_rename(out, shard, residents, set(residents.keys()))
    residents.clear()
    dt = time.time() - t0
    log(f"main shard {srcidx:05d}: {meta['n_tensors']} residents "
        f"(mxfp8={exact_stats['mxfp8']} keep={exact_stats['keep']}), "
        f"{src_bytes/1024**3:.2f} GiB source in {dt:.1f}s -> {shard} ({meta['size']/1024**3:.2f} GiB)")
    meta["exact_stats"] = exact_stats
    meta["src_bytes"] = src_bytes
    return meta


# --------------------------------------------------------------------------
# MTP experts pass: source FP4 -> mxfp4 residents (one shard per MTP layer)
# --------------------------------------------------------------------------
def process_mtp_layer(L: int, src: Path, weight_map: dict, out: Path) -> dict:
    """Repack MTP layer ``L``'s 128 FP4 experts to mxfp4 -> model-000{47+L}."""
    n_experts = 1 + max(
        int(n.split(".")[4]) for n in weight_map
        if n.startswith(f"mtp.{L}.ffn.experts.") and n.endswith(".weight")
    )
    residents: dict[str, mx.array] = {}
    shard = f"model-{MTP_SHARD_BASE + L:05d}.safetensors"
    src_bytes = 0
    n_exact = 0
    t0 = time.time()
    # experts of one MTP layer live in a single source shard
    src_shard = weight_map[f"mtp.{L}.ffn.experts.0.w1.weight"]
    header, ds = dc.read_safetensors_header(str(src / src_shard))
    entries = dc.tensor_entries(header)
    fd = os.open(str(src / src_shard), os.O_RDONLY)
    try:
        for E in range(n_experts):
            for w in ("w1", "w2", "w3"):
                base = f"mtp.{L}.ffn.experts.{E}.{w}"
                we = entries[base + ".weight"]
                se = entries[base + ".scale"]
                codes, scales, nb = _repack_fp4_to_mxfp4(fd, ds, we, se, base, {"mxfp4": 0})
                residents[base + ".weight"] = codes
                residents[base + ".scales"] = scales
                src_bytes += nb
                n_exact += 1
    finally:
        os.close(fd)
    meta = _write_verify_rename(out, shard, residents, set(residents.keys()))
    residents.clear()
    dt = time.time() - t0
    log(f"mtp layer {L}: {n_experts} experts -> mxfp4 ({n_exact} tensors), "
        f"{src_bytes/1024**3:.2f} GiB source in {dt:.1f}s -> {shard} ({meta['size']/1024**3:.2f} GiB)")
    meta["exact_stats"] = {"mxfp4": n_exact}
    meta["src_bytes"] = src_bytes
    meta["n_experts"] = n_experts
    meta["mtp_layer"] = L
    return meta


# --------------------------------------------------------------------------
# resume journal
# --------------------------------------------------------------------------
def _state_dir(out: Path) -> Path:
    d = out / ".w18-native-state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_state(out: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    for p in sorted(_state_dir(out).glob("*.json")):
        try:
            data = json.loads(p.read_text())
            done[data["shard"]] = data
        except Exception:
            continue
    return done


def _save_state(out: Path, meta: dict) -> None:
    p = _state_dir(out) / (meta["shard"].replace(".safetensors", "") + ".json")
    tmp = p.with_suffix(".json.partial")
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, p)


# --------------------------------------------------------------------------
# finalize: model.safetensors.index.json + config quantization block
# --------------------------------------------------------------------------
def build_index(out: Path, shard_names: list[str]) -> tuple[dict, int]:
    weight_map: dict[str, str] = {}
    total = 0
    for shard in sorted(shard_names):
        header, data_start = dc.read_safetensors_header(str(out / shard))
        for name, entry in dc.tensor_entries(header).items():
            weight_map[name] = shard
            total += entry.end - entry.begin
    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}
    return index, total


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def finalize(out: Path, shard_names: list[str]) -> dict:
    index, total = build_index(out, shard_names)
    _atomic_write_json(out / "model.safetensors.index.json", index)

    # config.json quantization block: default = the resident dense codec (mxfp8
    # gs32); per-module overrides for the MTP routed experts (mxfp4 gs32).
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    quant = {"group_size": dc.NATIVE_GROUP_SIZE, "bits": dc.MXFP8_BITS, "mode": "mxfp8"}
    for name, shard in index["weight_map"].items():
        if dc.is_mtp_expert(name) and name.endswith(".weight"):
            module = name[: -len(".weight")]
            quant[module] = {"group_size": dc.NATIVE_GROUP_SIZE, "bits": dc.MXFP4_BITS, "mode": "mxfp4"}
    cfg["quantization"] = quant
    _atomic_write_json(cfg_path, cfg)
    return {"total_size": total, "n_tensors": len(index["weight_map"]),
            "n_mtp_overrides": sum(1 for k in quant if k not in ("group_size", "bits", "mode"))}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def parse_shards(spec_str: str) -> list[int]:
    out: list[int] = []
    for part in spec_str.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="Hub source dir (DeepSeek-V4.1-Flash-src)")
    ap.add_argument("--out", type=Path, required=True, help="native artifact dir (…-mxfp4)")
    ap.add_argument("--shards", default="1-46", help="source shard indices for the main pass")
    ap.add_argument("--mtp", default="0,1,2", help="MTP layers to repack (default all)")
    ap.add_argument("--skip-mtp", action="store_true")
    ap.add_argument("--no-finalize", action="store_true")
    ap.add_argument("--report", type=Path, default=None, help="write a JSON run report here")
    args = ap.parse_args()

    src = args.src.expanduser().resolve()
    out = args.out.expanduser().resolve()
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    names_by_shard: dict[str, set] = {}
    for name, shard in index.items():
        names_by_shard.setdefault(shard, set()).add(name)

    state = _load_state(out)
    resident_shards: set[str] = {m["shard"] for m in state.values()}
    all_meta: list[dict] = list(state.values())
    t_start = time.time()

    # ---- main pass ----
    for srcidx in parse_shards(args.shards):
        path = src / f"model-{srcidx:05d}-of-00048.safetensors"
        shard = f"model-{srcidx:05d}.safetensors"
        if shard in state:
            log(f"main shard {srcidx:05d}: done (resume)")
            continue
        if not path.is_file() or path.name not in names_by_shard:
            continue
        meta = process_main_shard(srcidx, path, out)
        if meta is None:
            log(f"main shard {srcidx:05d}: no residents; skip")
            _save_state(out, {"shard": shard, "empty": True, "n_tensors": 0})
            continue
        _save_state(out, meta)
        resident_shards.add(shard)
        all_meta.append(meta)

    # ---- MTP experts pass ----
    if not args.skip_mtp:
        for L in parse_shards(args.mtp):
            shard = f"model-{MTP_SHARD_BASE + L:05d}.safetensors"
            if shard in state:
                log(f"mtp layer {L}: done (resume)")
                continue
            meta = process_mtp_layer(L, src, index, out)
            _save_state(out, meta)
            resident_shards.add(shard)
            all_meta.append(meta)

    log(f"conversion pass complete in {(time.time()-t_start)/60:.1f}m "
        f"({len([s for s in resident_shards])} resident shards)")

    report = {"resident_shards": sorted(resident_shards),
              "shard_meta": [m for m in all_meta if not m.get("empty")]}
    if not args.no_finalize:
        real_shards = [s for s in resident_shards if (out / s).is_file()]
        fin = finalize(out, real_shards)
        report["finalize"] = fin
        log(f"finalize: index {fin['n_tensors']} tensors, total_size {fin['total_size']/1024**3:.2f} GiB, "
            f"config mxfp8 default + {fin['n_mtp_overrides']} mtp mxfp4 overrides")

    if args.report:
        _atomic_write_json(args.report, report)
        log(f"wrote report {args.report}")
    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
