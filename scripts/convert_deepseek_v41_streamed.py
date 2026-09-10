#!/usr/bin/env python3
"""Resumable, shard-streaming DeepSeek-V4.1-Flash -> MTPLX Q2 streaming converter.

Routed backbone experts (``layers.{0..39}.ffn.experts.{0..383}.w{1,2,3}``, FP4)
are dequantized and re-quantized to affine Q2/gs64 and packed into an
aligned ``experts.bin`` expert bank (record = the nine ordered affine
components ``mtplx.expert_manifest`` expects, 11,059,200 B).  Everything else
that is not a routed expert and not Engram becomes q8/gs64 MLX resident
safetensors (router / norms / hc_* / attn_sink kept exact).  MTP routed experts
(``mtp.*.ffn.experts``) are excluded by default; Engram is not converted.

CPU-only: ``mx.quantize`` runs on the CPU stream, so no GPU flock is taken.
Resumable: per-source-shard state is journalled under
``<out>/.convert-state/`` and the (fixed-size) expert records are written at
deterministic offsets, so an interrupted run is restarted by re-invoking with
the same arguments.  Output files are append-only: an existing, verifying
output is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)

from mtplx import deepseek_v41_convert as dc  # noqa: E402
from mtplx.expert_manifest import (  # noqa: E402
    EMPTY_SHA256,
    ExpertManifest,
    ExpertRecord,
    ResidentTensor,
    ShardInfo,
    SidecarInfo,
    TensorSegment,
    load_expert_manifest,
    validate_expert_manifest_spec,
    verify_expert_manifest,
)
from mtplx.expert_streaming_models import get_model_spec  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# shard readiness
# --------------------------------------------------------------------------
def shard_path(src: Path, srcidx: int) -> Path:
    return src / f"model-{srcidx:05d}-of-00048.safetensors"


def shard_ready(path: Path, expected_names: set[str]) -> tuple[bool, str]:
    """A shard is ready only when the final file exists, its header parses,
    its size matches what the header implies, and it holds the index's names."""
    if not path.is_file():
        return False, "missing"
    try:
        header, data_start = dc.read_safetensors_header(str(path))
    except Exception as exc:  # truncated / mid-write header
        return False, f"header unparsable ({exc})"
    implied = dc.implied_file_size(header, data_start)
    actual = path.stat().st_size
    if actual != implied:
        return False, f"size {actual} != implied {implied}"
    names = {k for k in header if k != "__metadata__"}
    if names != expected_names:
        return False, f"names mismatch (+{len(names - expected_names)}/-{len(expected_names - names)})"
    return True, "ready"


def source_sha256(src: Path, path: Path) -> str | None:
    """The publisher's file sha256, read from the hf-download .metadata sidecar.

    Line 2 of ``<src>/.cache/huggingface/download/<file>.metadata`` is the
    git-lfs sha256 oid of the blob (= file content hash), so provenance needs
    no re-hash of the multi-GB source shard.
    """
    meta = src / ".cache" / "huggingface" / "download" / (path.name + ".metadata")
    try:
        lines = meta.read_text().splitlines()
        return lines[1].strip() if len(lines) >= 2 else None
    except OSError:
        return None


def wait_for_shard(path: Path, expected_names: set[str], poll: float, wait: bool) -> None:
    while True:
        ok, why = shard_ready(path, expected_names)
        if ok:
            return
        if not wait:
            raise SystemExit(f"shard not ready and --wait is off: {path.name}: {why}")
        log(f"waiting for {path.name}: {why}")
        time.sleep(poll)


# --------------------------------------------------------------------------
# expert bank writing (deterministic offsets, fixed record size)
# --------------------------------------------------------------------------
def _sha(buf: bytes) -> str:
    return hashlib.sha256(buf).hexdigest()


def quantize_expert_components(f32_by_proj: dict[str, np.ndarray]) -> tuple[list[bytes], list[dict]]:
    """Quantize one expert's gate/up/down to Q2 and return record blobs + segment meta."""
    blobs: list[bytes] = []
    seg_meta: list[dict] = []
    cursor = 0
    for proj in ("gate_proj", "up_proj", "down_proj"):
        packed, scales, biases = dc.quantize_affine(
            f32_by_proj[proj], bits=dc.EXPERT_BITS, group_size=dc.GROUP_SIZE
        )
        parts = dc.component_bytes(packed, scales, biases)
        shapes = [tuple(packed.shape), tuple(scales.shape), tuple(biases.shape)]
        dtypes = ["U32", "BF16", "BF16"]
        leaves = ["weight", "scales", "biases"]
        for leaf, dtype, shape, part in zip(leaves, dtypes, shapes, parts):
            seg_meta.append(
                {
                    "component": f"{proj}.{leaf}",
                    "leaf_name": f"{proj}.{leaf}",
                    "dtype": dtype,
                    "shape": list(shape),
                    "rel_offset": cursor,
                    "length": len(part),
                }
            )
            blobs.append(part)
            cursor += len(part)
        del packed, scales, biases
    return blobs, seg_meta


def write_expert_record(exp_fd: int, record_index: int, blobs: list[bytes]) -> tuple[int, str]:
    offset = record_index * dc.EXPERT_RECORD_BYTES
    payload = b"".join(blobs)
    if len(payload) != dc.EXPERT_RECORD_BYTES:
        raise RuntimeError(
            f"record {record_index}: {len(payload)} bytes != {dc.EXPERT_RECORD_BYTES}"
        )
    pos = offset
    view = memoryview(payload)
    while view:
        written = os.pwrite(exp_fd, view, pos)
        if written <= 0:
            raise RuntimeError("short pwrite to experts.bin")
        pos += written
        view = view[written:]
    return offset, _sha(payload)


# --------------------------------------------------------------------------
# per-source-shard processing
# --------------------------------------------------------------------------
def process_shard(
    srcidx: int,
    path: Path,
    out: Path,
    routed_layers: list[int],
    exp_fd: int,
    include_mtp_experts: bool,
) -> dict:
    """Convert one source shard: experts -> bank, residents -> a q8 safetensors."""
    header, data_start = dc.read_safetensors_header(str(path))
    entries = dc.tensor_entries(header)
    fd = os.open(str(path), os.O_RDONLY)
    residents: dict[str, mx.array] = {}
    resident_meta: list[dict] = []
    records_meta: list[dict] = []
    resident_shard = f"model-{srcidx:05d}.safetensors"
    src_bytes = 0
    t0 = time.time()
    try:
        # ---- 1. routed experts -> bank ---------------------------------
        expert_groups: dict[tuple[int, int], dict[str, dc.TensorEntry]] = {}
        for name, entry in entries.items():
            if dc.is_bank_expert(name):
                m = name.split(".")
                layer = int(m[1]); expert = int(m[4]); w = m[5]; leaf = m[6]
                expert_groups.setdefault((layer, expert), {})[f"{w}.{leaf}"] = entry
        for (layer, expert) in sorted(expert_groups.keys()):
            if layer not in routed_layers:
                continue
            g = expert_groups[(layer, expert)]
            f32_by_proj: dict[str, np.ndarray] = {}
            for proj, w in dc.PROJ_TO_SOURCE_W.items():
                we = g[f"{w}.weight"]; se = g[f"{w}.scale"]
                packed = np.frombuffer(dc.read_tensor_raw(fd, data_start, we), dtype=np.uint8).reshape(we.shape)
                scale = np.frombuffer(dc.read_tensor_raw(fd, data_start, se), dtype=np.uint8).reshape(se.shape)
                src_bytes += (we.end - we.begin) + (se.end - se.begin)
                f32_by_proj[proj] = dc.dequant_fp4(packed, scale)
            blobs, seg_meta = quantize_expert_components(f32_by_proj)
            layer_pos = routed_layers.index(layer)
            record_index = layer_pos * dc.N_ROUTED_EXPERTS + expert
            offset, sha = write_expert_record(exp_fd, record_index, blobs)
            records_meta.append(
                {"layer": layer, "expert": expert, "index": record_index,
                 "sidecar_offset": offset, "sha256": sha, "segments": seg_meta}
            )
            del f32_by_proj, blobs
        # ---- 2. residents ------------------------------------------------
        for name in sorted(entries):
            entry = entries[name]
            if dc.is_bank_expert(name) or dc.is_engram(name):
                continue
            if dc.is_mtp_expert(name) and not include_mtp_experts:
                continue
            disp = dc.resident_disposition(entry)
            if disp == "drop":
                continue
            if disp == "keep":
                raw = dc.read_tensor_raw(fd, data_start, entry)
                src_bytes += len(raw)
                if entry.dtype == "BF16":
                    arr = mx.array(np.frombuffer(raw, dtype="<u2").reshape(entry.shape)).view(mx.bfloat16)
                elif entry.dtype == "F32":
                    arr = mx.array(np.frombuffer(raw, dtype="<f4").reshape(entry.shape))
                elif entry.dtype == "F16":
                    arr = mx.array(np.frombuffer(raw, dtype="<f2").reshape(entry.shape))
                else:
                    raise RuntimeError(f"unexpected keep dtype {entry.dtype} for {name}")
                mx.eval(arr)
                residents[name] = arr
                resident_meta.append({"tensor": name, "quantized": False})
            else:  # quantize -> q8
                raw = dc.read_tensor_raw(fd, data_start, entry)
                src_bytes += len(raw)
                if entry.dtype == "F8_E4M3":
                    sib = entries[name[:-len(".weight")] + ".scale"] if name.endswith(".weight") else None
                    if sib is None:
                        raise RuntimeError(f"no .scale sibling for fp8 {name}")
                    weight_u8 = np.frombuffer(raw, dtype=np.uint8).reshape(entry.shape)
                    scale_u8 = np.frombuffer(dc.read_tensor_raw(fd, data_start, sib), dtype=np.uint8).reshape(sib.shape)
                    src_bytes += (sib.end - sib.begin)
                    f32 = dc.dequant_fp8_block(weight_u8, scale_u8)
                else:
                    f32 = dc.raw_to_f32(entry, raw)
                packed, scales, biases = dc.quantize_affine(f32, bits=dc.RESIDENT_BITS, group_size=dc.GROUP_SIZE)
                base = name[:-len(".weight")] if name.endswith(".weight") else name
                residents[f"{base}.weight"] = packed
                residents[f"{base}.scales"] = scales
                residents[f"{base}.biases"] = biases
                resident_meta.append({"tensor": name, "quantized": True})
                del f32
        # ---- 3. flush resident safetensors ------------------------------
        if residents:
            mx.eval(*residents.values())
            # mx.save_safetensors enforces a .safetensors extension, so the
            # temp name must keep it; rename is atomic within the same dir.
            tmp = out / resident_shard.replace(".safetensors", ".partial.safetensors")
            mx.save_safetensors(str(tmp), residents, metadata={"format": "mlx"})
            os.replace(tmp, out / resident_shard)
        residents.clear()
    finally:
        os.close(fd)
    dt = time.time() - t0
    rate = (src_bytes / 1e6) / dt if dt > 0 else 0.0
    log(f"shard {srcidx:05d}: {len(records_meta)} experts, "
        f"{len([m for m in resident_meta])} residents, "
        f"{src_bytes/1024**3:.2f} GiB source in {dt:.1f}s ({rate:.0f} MB/s)")
    return {
        "srcidx": srcidx,
        "source_file": path.name,
        "source_sha256": source_sha256(path.parent, path),
        "resident_shard": resident_shard if resident_meta else None,
        "records": records_meta,
        "src_bytes": src_bytes,
        "seconds": dt,
    }


# --------------------------------------------------------------------------
# finalize: manifest, config, index, aux files, verification
# --------------------------------------------------------------------------
def _hash_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blob in iter(lambda: f.read(chunk), b""):
            h.update(blob)
    return h.hexdigest()


def build_resident_index_and_manifest_inputs(out: Path, resident_shards: list[str]):
    """Parse written resident shards -> (resident_tensors, shard_infos, index, total)."""
    resident_tensors: list[ResidentTensor] = []
    shard_infos: list[ShardInfo] = []
    weight_map: dict[str, str] = {}
    total = 0
    for shard in sorted(resident_shards):
        p = out / shard
        header, data_start = dc.read_safetensors_header(str(p))
        header_bytes = data_start
        with open(p, "rb") as f:
            head_raw = f.read(header_bytes)
        shard_infos.append(
            ShardInfo(
                name=shard,
                size=p.stat().st_size,
                header_bytes=header_bytes,
                header_sha256=_sha(head_raw),
                sha256=_hash_file(p),
                kind="safetensors",
            )
        )
        for name, entry in dc.tensor_entries(header).items():
            length = entry.end - entry.begin
            resident_tensors.append(
                ResidentTensor(
                    tensor=name, shard=shard, offset=data_start + entry.begin,
                    length=length, dtype=entry.dtype, shape=entry.shape,
                )
            )
            weight_map[name] = shard
            total += length
    resident_tensors.sort(key=lambda t: t.tensor)
    return resident_tensors, shard_infos, weight_map, total


def finalize(out: Path, src: Path, config: dict, spec, records_meta: list[dict],
             resident_shards: list[str], source_shas: dict[str, str],
             require_pinned: bool) -> ExpertManifest:
    exp_path = out / "experts.bin"
    exp_size = exp_path.stat().st_size
    exp_sha = _hash_file(exp_path)

    resident_tensors, resident_shard_infos, weight_map, resident_total = \
        build_resident_index_and_manifest_inputs(out, resident_shards)

    # model.safetensors.index.json (authoritative verify requires it)
    index = {"metadata": {"total_size": resident_total}, "weight_map": weight_map}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    # config.json with the resident quantization block (scales-presence driven)
    cfg = dict(config)
    cfg["quantization"] = {"group_size": dc.GROUP_SIZE, "bits": dc.RESIDENT_BITS, "mode": "affine"}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    # authoritative expert records + sidecar
    records_meta = sorted(records_meta, key=lambda r: (r["layer"], r["expert"]))
    records: list[ExpertRecord] = []
    routed_bytes = 0
    for r in records_meta:
        base = r["sidecar_offset"]
        segments = tuple(
            TensorSegment(
                component=s["component"],
                tensor=f"layers.{r['layer']}.ffn.experts.{r['expert']}.{s['leaf_name']}",
                shard="experts.bin",
                offset=base + s["rel_offset"],
                length=s["length"],
                dtype=s["dtype"],
                shape=tuple(s["shape"]),
            )
            for s in r["segments"]
        )
        records.append(
            ExpertRecord(
                layer=r["layer"], expert=r["expert"],
                logical_bytes=dc.EXPERT_RECORD_BYTES, segments=segments,
                sha256=r["sha256"], sidecar_offset=base,
                sidecar_length=dc.EXPERT_RECORD_BYTES, part=0,
            )
        )
        routed_bytes += dc.EXPERT_RECORD_BYTES

    sidecar_shard = ShardInfo(
        name="experts.bin", size=exp_size, header_bytes=0,
        header_sha256=EMPTY_SHA256, sha256=exp_sha, kind="sidecar",
    )
    manifest = ExpertManifest(
        model_key=spec.key,
        source_repo=spec.quant_model,
        source_revision=spec.quant_revision,
        quant_bits=spec.quant_bits,
        quant_group_size=spec.quant_group_size,
        quant_mode="affine",
        artifact_tensor_bytes=resident_total + routed_bytes,
        resident_tensor_bytes=resident_total,
        routed_expert_bytes=routed_bytes,
        shards=tuple(resident_shard_infos) + (sidecar_shard,),
        resident_tensors=tuple(resident_tensors),
        records=tuple(records),
        sidecar=SidecarInfo(file="experts.bin", alignment=dc.ALIGNMENT,
                            size=exp_size, sha256=exp_sha),
    ).with_digest()
    manifest.validate_structure()
    validate_expert_manifest_spec(manifest, spec, require_pinned_tensor_bytes=require_pinned)

    manifest_path = out / "expert-manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))

    # conversion-manifest.json (provenance)
    conv = {
        "model_key": spec.key,
        "kind": "fp4_e2m1_to_q2",
        "source_model": spec.source_model,
        "source_revision": dc.SOURCE_REVISION,
        "group_size": dc.GROUP_SIZE,
        "expert_bits": dc.EXPERT_BITS,
        "resident_bits": dc.RESIDENT_BITS,
        "mode": "affine",
        "mlx_version": mx.__version__,
        "expert_record_bytes": dc.EXPERT_RECORD_BYTES,
        "routed_layers": sorted({r["layer"] for r in records_meta}),
        "resident_tensor_bytes": resident_total,
        "routed_expert_bytes": routed_bytes,
        "artifact_tensor_bytes": resident_total + routed_bytes,
        "records": len(records),
        "engram_included": False,
        "mtp_experts_included": False,
        "source_shard_sha256": dict(sorted(source_shas.items())),
    }
    (out / "conversion-manifest.json").write_text(json.dumps(conv, indent=2))

    # copy tokenizer / license / encoding verbatim (append-only: skip if present)
    for name in ("tokenizer.json", "tokenizer_config.json", "LICENSE"):
        s = src / name
        d = out / name
        if s.is_file() and not d.exists():
            shutil.copyfile(s, d)
    enc_src = src / "encoding"
    enc_dst = out / "encoding"
    if enc_src.is_dir() and not enc_dst.exists():
        shutil.copytree(enc_src, enc_dst)
    return manifest


# --------------------------------------------------------------------------
# state journal (resume)
# --------------------------------------------------------------------------
def state_dir(out: Path) -> Path:
    d = out / ".convert-state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_state(out: Path) -> dict[int, dict]:
    done: dict[int, dict] = {}
    for p in sorted(state_dir(out).glob("src-*.json")):
        try:
            data = json.loads(p.read_text())
            done[int(data["srcidx"])] = data
        except Exception:
            continue
    return done


def save_state(out: Path, data: dict) -> None:
    p = state_dir(out) / f"src-{data['srcidx']:05d}.json"
    tmp = p.with_suffix(".json.partial")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, p)


# --------------------------------------------------------------------------
# cosine round-trip (pilot QA)
# --------------------------------------------------------------------------
def cosine_check(out: Path, src: Path, records_meta: list[dict], n: int,
                 weight_map: dict[str, str]) -> list[dict]:
    """Dequantize Q2 records and compare cosine vs the FP4-dequantized source."""
    exp_fd = os.open(str(out / "experts.bin"), os.O_RDONLY)
    rows = []
    sample = records_meta[:: max(1, len(records_meta) // n)][:n]
    # locate the source shard holding each record's layer via the index
    for r in sample:
        layer, expert = r["layer"], r["expert"]
        w1_name = f"layers.{layer}.ffn.experts.{expert}.w1.weight"
        srcshard = src / weight_map[w1_name]
        header, data_start = dc.read_safetensors_header(str(srcshard))
        entries = dc.tensor_entries(header)
        sfd = os.open(str(srcshard), os.O_RDONLY)
        cos_by_proj = {}
        try:
            base = r["sidecar_offset"]
            cursor = base
            for proj, w in dc.PROJ_TO_SOURCE_W.items():
                we = entries[f"layers.{layer}.ffn.experts.{expert}.{w}.weight"]
                se = entries[f"layers.{layer}.ffn.experts.{expert}.{w}.scale"]
                packed = np.frombuffer(dc.read_tensor_raw(sfd, data_start, we), dtype=np.uint8).reshape(we.shape)
                scale = np.frombuffer(dc.read_tensor_raw(sfd, data_start, se), dtype=np.uint8).reshape(se.shape)
                ref = dc.dequant_fp4(packed, scale).astype(np.float32)
                # matching q2 component read from the bank
                seg = {s["component"]: s for s in r["segments"]}
                pw = seg[f"{proj}.weight"]; ps = seg[f"{proj}.scales"]; pb = seg[f"{proj}.biases"]
                def _read(seg):
                    raw = dc._pread_exact(exp_fd, base + seg["rel_offset"], seg["length"])
                    return raw
                out_shape = tuple(pw["shape"])
                pk = mx.array(np.frombuffer(_read(pw), dtype="<u4").reshape(out_shape))
                sc = mx.array(np.frombuffer(_read(ps), dtype="<u2").reshape(tuple(ps["shape"]))).view(mx.bfloat16)
                bi = mx.array(np.frombuffer(_read(pb), dtype="<u2").reshape(tuple(pb["shape"]))).view(mx.bfloat16)
                dq = np.array(mx.dequantize(pk, sc, bi, group_size=dc.GROUP_SIZE, bits=dc.EXPERT_BITS).astype(mx.float32))
                a = ref.ravel(); b = dq.ravel()
                cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                rel = float(np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-12))
                cos_by_proj[proj] = (cos, rel)
        finally:
            os.close(sfd)
        rows.append({"layer": layer, "expert": expert, "cos": cos_by_proj})
    os.close(exp_fd)
    return rows


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
MTP_BITS = 8
MTP_GROUP = 32  # aligns q8 groups with the FP4 source's per-32-column scale


def add_mtp_residents(out: Path, src: Path, weight_map: dict[str, str], spec,
                      cosine_experts: int) -> None:
    """Append the DSpark MTP routed experts to v1 as affine q8/gs32 residents.

    Append-only: writes NEW safetensors shards (model-000{47+L}), then updates
    model.safetensors.index.json, config.json (per-module quantization
    overrides), expert-manifest.json, and conversion-manifest.json in place.
    experts.bin and the existing resident shards are never rewritten.
    """
    manifest_path = out / "expert-manifest.json"
    manifest = load_expert_manifest(manifest_path)
    if any(dc.is_mtp_expert(t.tensor) for t in manifest.resident_tensors):
        log("MTP experts already present in the manifest; nothing to do")
        return

    mtp_layers = sorted({int(n.split(".")[1]) for n in weight_map
                         if dc.is_mtp_expert(n) and n.endswith(".weight")})
    n_experts = 1 + max(int(n.split(".")[4]) for n in weight_map if dc.is_mtp_expert(n))
    log(f"MTP: layers {mtp_layers}, {n_experts} experts each -> affine q8 gs{MTP_GROUP} residents")

    st = state_dir(out)
    new_shard_files: list[str] = []
    overrides: dict[str, dict] = {}
    source_shas: dict[str, str] = {}
    receipts: list[tuple[str, float, float]] = []

    for L in mtp_layers:
        shard_file = f"model-{47 + L:05d}.safetensors"
        journal = st / f"mtp-{L:05d}.json"
        if journal.exists() and (out / shard_file).is_file():
            data = json.loads(journal.read_text())
            overrides.update(data["overrides"])
            source_shas[data["source_file"]] = data["source_sha256"]
            new_shard_files.append(shard_file)
            log(f"MTP layer {L}: already done (resume)")
            continue
        srcshard = src / weight_map[f"mtp.{L}.ffn.experts.0.w1.weight"]
        header, ds = dc.read_safetensors_header(str(srcshard))
        entries = dc.tensor_entries(header)
        fd = os.open(str(srcshard), os.O_RDONLY)
        buffer: dict[str, mx.array] = {}
        layer_overrides: dict[str, dict] = {}
        src_bytes = 0
        t0 = time.time()
        try:
            for E in range(n_experts):
                for w in ("w1", "w2", "w3"):
                    base = f"mtp.{L}.ffn.experts.{E}.{w}"
                    we = entries[base + ".weight"]; se = entries[base + ".scale"]
                    packed = np.frombuffer(dc.read_tensor_raw(fd, ds, we), np.uint8).reshape(we.shape)
                    scale = np.frombuffer(dc.read_tensor_raw(fd, ds, se), np.uint8).reshape(se.shape)
                    src_bytes += (we.end - we.begin) + (se.end - se.begin)
                    f32 = dc.dequant_fp4(packed, scale)
                    pk, sc, bi = dc.quantize_affine(f32, bits=MTP_BITS, group_size=MTP_GROUP)
                    buffer[base + ".weight"] = pk
                    buffer[base + ".scales"] = sc
                    buffer[base + ".biases"] = bi
                    layer_overrides[base] = {"bits": MTP_BITS, "group_size": MTP_GROUP, "mode": "affine"}
                    if E < cosine_experts:
                        deq = np.array(mx.dequantize(pk, sc, bi, group_size=MTP_GROUP,
                                                     bits=MTP_BITS).astype(mx.float32))
                        a = f32.ravel(); b = deq.ravel()
                        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                        receipts.append((base, cos, float(np.max(np.abs(b - a)))))
                    del f32
            mx.eval(*buffer.values())
            tmp = out / shard_file.replace(".safetensors", ".partial.safetensors")
            mx.save_safetensors(str(tmp), buffer, metadata={"format": "mlx"})
            os.replace(tmp, out / shard_file)
        finally:
            os.close(fd)
        sha = source_sha256(src, srcshard)
        journal.write_text(json.dumps({"layer": L, "shard": shard_file,
                                       "overrides": layer_overrides,
                                       "source_file": srcshard.name, "source_sha256": sha}))
        overrides.update(layer_overrides)
        source_shas[srcshard.name] = sha
        new_shard_files.append(shard_file)
        dt = time.time() - t0
        log(f"MTP layer {L}: {n_experts} experts -> {shard_file}, "
            f"{src_bytes/1024**3:.2f} GiB source in {dt:.1f}s")
        buffer.clear()

    # ---- parse new shards -> resident tensors + shard infos ----
    new_res, new_shard_infos, new_wm, new_total = \
        build_resident_index_and_manifest_inputs(out, new_shard_files)
    added_bytes = new_total

    # ---- update manifest in place (append residents + shards) ----
    combined_res = sorted(list(manifest.resident_tensors) + list(new_res),
                          key=lambda t: t.tensor)
    safetensors_shards = [s for s in manifest.shards if s.kind == "safetensors"]
    sidecar_shards = [s for s in manifest.shards if s.kind == "sidecar"]
    combined_shards = safetensors_shards + list(new_shard_infos) + sidecar_shards
    resident_bytes = manifest.resident_tensor_bytes + added_bytes
    manifest2 = replace(
        manifest,
        resident_tensors=tuple(combined_res),
        shards=tuple(combined_shards),
        resident_tensor_bytes=resident_bytes,
        artifact_tensor_bytes=resident_bytes + manifest.routed_expert_bytes,
        manifest_sha256=None,
    ).with_digest()
    manifest2.validate_structure()
    validate_expert_manifest_spec(manifest2, spec, require_pinned_tensor_bytes=False)
    manifest_path.write_text(json.dumps(manifest2.to_dict(), indent=2))

    # ---- update model.safetensors.index.json ----
    index_path = out / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].update(new_wm)
    index["metadata"]["total_size"] = resident_bytes
    index_path.write_text(json.dumps(index, indent=2))

    # ---- update config.json quantization module overrides ----
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.setdefault("quantization", {"group_size": dc.GROUP_SIZE, "bits": dc.RESIDENT_BITS, "mode": "affine"})
    cfg["quantization"].update(overrides)
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # ---- update conversion-manifest.json ----
    conv_path = out / "conversion-manifest.json"
    conv = json.loads(conv_path.read_text())
    conv["mtp_experts_included"] = True
    conv["mtp_expert_mode"] = "affine-q8-gs32"
    conv["mtp_expert_bits"] = MTP_BITS
    conv["mtp_expert_group_size"] = MTP_GROUP
    conv["mtp_expert_layers"] = mtp_layers
    conv["mtp_expert_bytes_added"] = added_bytes
    conv["resident_tensor_bytes"] = resident_bytes
    conv["artifact_tensor_bytes"] = resident_bytes + manifest.routed_expert_bytes
    conv.setdefault("source_shard_sha256", {}).update(source_shas)
    conv_path.write_text(json.dumps(conv, indent=2))

    log(f"MTP residents added: {len(new_res)} tensors across {len(new_shard_files)} shards, "
        f"{added_bytes/1024**3:.2f} GiB; {len(overrides)} config overrides")
    if receipts:
        coses = [c for _n, c, _m in receipts]
        maxes = [m for _n, _c, m in receipts]
        log(f"MTP q8gs32 receipt: cos mean={np.mean(coses):.6f} min={np.min(coses):.6f} "
            f"max-abs-err={np.max(maxes):.5f} over {len(receipts)} components")
        for name, cos, mx_err in receipts[:24]:
            log(f"  {name}: cos={cos:.6f} maxabs={mx_err:.5f}")

    # experts.bin is byte-untouched by this append (records verified at v1
    # finalize); re-verify the authoritative inventory + all shard hashes
    # (incl. the new MTP shards) + the sidecar hash.
    log("verifying v1+MTP (shards + sidecar; records unchanged) ...")
    report = verify_expert_manifest(manifest2, out, verify_records=False,
                                    verify_shard_hashes=True, verify_sidecar_hash=True)
    log(f"verify report: {json.dumps(report)}")


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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True, help="model.safetensors.index.json")
    ap.add_argument("--config", type=Path, required=True, help="source config.json")
    ap.add_argument("--shards", default="1-46", help="source shard indices to process")
    ap.add_argument("--wait", action="store_true", help="poll for shard readiness")
    ap.add_argument("--poll-interval", type=float, default=30.0)
    ap.add_argument("--pilot", action="store_true", help="pilot: strict spec over the processed layers only")
    ap.add_argument("--cosine-experts", type=int, default=0, help="run cosine QA over N sampled records")
    ap.add_argument("--include-mtp-experts", action="store_true")
    ap.add_argument("--no-finalize", action="store_true")
    ap.add_argument("--require-pinned", action="store_true",
                    help="require spec total/resident byte match (default off; total is provisional)")
    ap.add_argument("--add-mtp-residents", action="store_true",
                    help="append-only: add MTP routed experts to an existing v1 artifact as "
                         "affine q8/gs32 residents; update index/config/manifests in place")
    args = ap.parse_args()

    src: Path = args.src.expanduser().resolve()
    out: Path = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    spec = get_model_spec(dc.MODEL_KEY)
    config = json.loads(args.config.read_text())
    index = json.loads(args.index.read_text())["weight_map"]

    if args.add_mtp_residents:
        add_mtp_residents(out, src, index, spec,
                          cosine_experts=max(args.cosine_experts, 8))
        log("DONE (mtp residents)")
        return 0

    # index names grouped by source shard
    names_by_shard: dict[str, set[str]] = {}
    for name, shard in index.items():
        names_by_shard.setdefault(shard, set()).add(name)

    shard_indices = parse_shards(args.shards)
    # routed layers whose expert tensors map to a processed source shard
    # (defines the record-index -> offset ordering in experts.bin)
    routed_set = set()
    for name, shard in index.items():
        if dc.is_bank_expert(name):
            si = int(shard[len("model-"):len("model-") + 5])
            if si in shard_indices:
                routed_set.add(int(name.split(".")[1]))
    routed_layers = sorted(routed_set)
    num_records = len(routed_layers) * dc.N_ROUTED_EXPERTS
    log(f"routed layers to process: {routed_layers} ({num_records} records, "
        f"record={dc.EXPERT_RECORD_BYTES} B, bank~{num_records*dc.EXPERT_RECORD_BYTES/1024**3:.1f} GiB)")

    # open (or create) experts.bin sized to the full record count
    exp_path = out / "experts.bin"
    exp_fd = os.open(str(exp_path), os.O_RDWR | os.O_CREAT, 0o644)
    want = num_records * dc.EXPERT_RECORD_BYTES
    if os.fstat(exp_fd).st_size < want:
        os.ftruncate(exp_fd, want)

    state = load_state(out)
    all_records: list[dict] = []
    resident_shards: list[str] = []
    source_shas: dict[str, str] = {}
    for data in state.values():
        all_records.extend(data.get("records", []))
        if data.get("resident_shard"):
            resident_shards.append(data["resident_shard"])
        if data.get("source_file") and data.get("source_sha256"):
            source_shas[data["source_file"]] = data["source_sha256"]

    t_start = time.time()
    processed = 0
    for srcidx in shard_indices:
        if srcidx in state:
            log(f"shard {srcidx:05d}: already done (resume)")
            continue
        path = shard_path(src, srcidx)
        expected = names_by_shard.get(path.name, set())
        if not expected:
            log(f"shard {srcidx:05d}: no tensors in index; skipping")
            save_state(out, {"srcidx": srcidx, "resident_shard": None, "records": [], "src_bytes": 0, "seconds": 0})
            continue
        wait_for_shard(path, expected, args.poll_interval, args.wait)
        data = process_shard(srcidx, path, out, routed_layers, exp_fd, args.include_mtp_experts)
        os.fsync(exp_fd)
        save_state(out, data)
        all_records.extend(data["records"])
        if data["resident_shard"]:
            resident_shards.append(data["resident_shard"])
        if data.get("source_file") and data.get("source_sha256"):
            source_shas[data["source_file"]] = data["source_sha256"]
        processed += 1
        done_records = len(all_records)
        if num_records:
            elapsed = time.time() - t_start
            frac = done_records / num_records
            eta = (elapsed / frac - elapsed) if frac > 0 else 0
            log(f"progress: {done_records}/{num_records} records ({100*frac:.1f}%), "
                f"elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m")
    os.fsync(exp_fd)
    os.close(exp_fd)

    if args.no_finalize:
        log("skipping finalize (--no-finalize)")
        return 0

    fin_spec = spec
    if args.pilot:
        fin_spec = replace(spec, routed_layer_start=routed_layers[0],
                           routed_layer_count=len(routed_layers))
    log("finalizing: manifest + config + index + aux ...")
    manifest = finalize(out, src, config, fin_spec, all_records, resident_shards,
                        source_shas, require_pinned=args.require_pinned)
    log(f"manifest: {len(manifest.records)} records, "
        f"resident={manifest.resident_tensor_bytes/1024**3:.2f} GiB, "
        f"routed={manifest.routed_expert_bytes/1024**3:.2f} GiB, digest={manifest.manifest_sha256[:12]}")

    log("verifying (repo verify_expert_manifest: records + sidecar) ...")
    report = verify_expert_manifest(manifest, out, verify_records=True,
                                    verify_shard_hashes=True, verify_sidecar_hash=True)
    log(f"verify report: {json.dumps(report)}")

    if args.cosine_experts:
        log(f"cosine round-trip QA over {args.cosine_experts} records ...")
        rows = cosine_check(out, src, all_records, args.cosine_experts, index)
        coses = [c for row in rows for (c, _r) in row["cos"].values()]
        rels = [r for row in rows for (_c, r) in row["cos"].values()]
        log(f"cosine mean={np.mean(coses):.4f} min={np.min(coses):.4f} "
            f"rel_err mean={np.mean(rels):.4f}")
        for row in rows:
            parts = " ".join(f"{p}={row['cos'][p][0]:.3f}" for p in ("gate_proj", "up_proj", "down_proj"))
            log(f"  L{row['layer']} E{row['expert']}: {parts}")
    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
