#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash Engram tables -> MTPLX disk-backed affine-8 row bank.

The Engram conditional-memory tables (``layers.{1,14}.engram.embed.{weight,scale}``,
~189 GiB FP8 E4M3 with E8M0 scales) stay on disk and stream row-by-row at inference,
exactly like the routed-expert bank.  This converts each table into a flat array of
fixed-size affine-8 records -- record index == source row index -- so a runtime lookup
is ``pread(fd, record_bytes, row * record_bytes)``.

One source row is ``head_dim`` (256) values (mirrors the source shape
``[num_embeddings, head_dim]``; the table is per-head, NOT ``n_heads*head_dim`` wide).
Each record is that row re-quantized with ``mx.quantize(bits=8, group_size=64,
mode="affine")``: packed U32 weights + BF16 scales + BF16 biases == 272 bytes.

Cleanliness (David): the artifact's ``engram/`` folder holds ONLY the finished
``engram-L1.bin``, ``engram-L14.bin`` and ``engram-manifest.json``.  The resume journal
and the in-progress ``.bin`` live under ``--state-dir`` (default
``/Users/davidtai/models/dsv41-convert-state/engram``); a completed layer is
``os.replace``-d (atomic, same filesystem) into the artifact, so no partial file ever
rests in ``engram/``.

Resumable: per-layer chunk state is journalled; completed chunks are never rewritten.
CPU-only (``mx.quantize`` on the CPU stream, no GPU flock).  Streaming reads of the
source (``os.pread`` by offset), never mmap of the 95 GiB shard.

Run under ``nice -n 19``.
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

# import the worktree's converter primitives (editable-install CWD shadowing guard)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mtplx.deepseek_v41_convert as dc  # noqa: E402

DEFAULT_STATE_DIR = "/Users/davidtai/models/dsv41-convert-state/engram"
MANIFEST_FORMAT = "mtplx-engram-manifest-v1"

# ---- mxfp8 row codec (EXACT byte repack of the Hub source) ----------------
# The source engram rows already ARE mxfp8: F8_E4M3 code bytes + one F8_E8M0 scale byte per
# 32-col group.  The ``mxfp8`` codec keeps those bytes verbatim -- 256 code bytes (U32[64]) + 8
# scale bytes (U8[8]) == 264 B/row -- so the bank dequantizes bit-exactly with
# ``mx.dequantize(mode="mxfp8", group_size=32, bits=8)``.  No requantization, unlike the affine
# codec.  Contrast the affine record (272 B): U32[64] levels + BF16[4] scales + BF16[4] biases.
MXFP8_GROUP_SIZE = dc.ENGRAM_FP8_BLOCK            # 32 (source scale block == mlx mxfp8 group)


def mxfp8_record_bytes(head_dim: int = dc.ENGRAM_HEAD_DIM,
                       group: int = MXFP8_GROUP_SIZE) -> int:
    """Bytes of one mxfp8 record: E4M3 code bytes (== head_dim) + one E8M0 scale byte / group."""
    return head_dim + head_dim // group           # 256 + 8 == 264


def mxfp8_record_layout(head_dim: int = dc.ENGRAM_HEAD_DIM,
                        group: int = MXFP8_GROUP_SIZE) -> dict:
    """Byte layout of one mxfp8 record (offsets within the fixed-size record; no biases)."""
    groups = head_dim // group
    return {
        "weight": {"offset": 0, "length": head_dim, "dtype": "U32",
                   "shape": [head_dim // 4]},
        "scales": {"offset": head_dim, "length": groups, "dtype": "U8",
                   "shape": [groups]},
        "record_bytes": head_dim + groups,
    }


MXFP8_RECORD_BYTES = mxfp8_record_bytes()         # 264


def _mxfp8_chunk_records(weight_u8: np.ndarray, scale_u8: np.ndarray) -> np.ndarray:
    """Repack a chunk of source rows into mxfp8 records: ``[rows, 264]`` uint8.

    ``weight_u8``: uint8 ``[rows, head_dim]`` (F8_E4M3 code bytes, kept verbatim).
    ``scale_u8``:  uint8 ``[rows, head_dim//32]`` (F8_E8M0 scale bytes, kept verbatim).
    The record is ``code_bytes | scale_bytes`` -- an EXACT copy of the source bytes, no
    dequant/requant, so the row is bit-exact to the Hub source.
    """
    rows, hd = weight_u8.shape
    groups = hd // MXFP8_GROUP_SIZE
    if scale_u8.shape != (rows, groups):
        raise ValueError(f"scale shape {scale_u8.shape} != {(rows, groups)}")
    rec = np.concatenate(
        [np.ascontiguousarray(weight_u8, dtype=np.uint8),
         np.ascontiguousarray(scale_u8, dtype=np.uint8)], axis=1)
    exp = mxfp8_record_bytes(hd)
    if rec.shape[1] != exp:
        raise RuntimeError(f"mxfp8 record width {rec.shape[1]} != expected {exp}")
    return np.ascontiguousarray(rec, dtype=np.uint8)

# ---- resident Engram projections (W4 sidecar) ----------------------------
# The streaming artifact carries the row banks but NOT the small resident Engram
# projection tensors.  This sidecar re-materialises them next to the banks:
#   layers.{1,14}.engram.wkv.{weight,scale}  F8_E4M3 [25600,6144] + F8_E8M0 32x32
#       -> dequantized per the same FP8 block-scale formula the streamed converter
#          uses for every other FP8 resident, then re-quantized to affine q8/gs64
#          (packed U32 weight + BF16 scales + BF16 biases), naming
#          layers.L.engram.wkv.{weight,scales,biases} -- the exact convention of the
#          other resident q8 tensors in the artifact shards (e.g. aligner.w1.*).
#   layers.{1,14}.engram.{q_weight,k_weight}  [4,5120] -> kept EXACT as F32.
#       (the pinned source stores these BF16; BF16 -> F32 is a lossless widening, and
#        EngramV41 upcasts them to F32 anyway, so no information is lost.)
RESIDENT_SIDECAR = "engram-residents.safetensors"
RESIDENT_BITS = dc.ENGRAM_BITS          # 8
RESIDENT_GROUP = dc.ENGRAM_GROUP_SIZE   # 64
RESIDENT_WKV = "wkv"                     # layers.L.engram.wkv.{weight,scale}
RESIDENT_EXACT = ("q_weight", "k_weight")  # layers.L.engram.{q,k}_weight, kept F32 exact


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# source shard readiness / provenance
# --------------------------------------------------------------------------
def load_index(index_path: Path) -> dict:
    return json.loads(Path(index_path).read_text())


def shard_names(weight_map: dict, shard_file: str) -> set:
    return {name for name, f in weight_map.items() if f == shard_file}


def shard_ready(path: Path, expected_names: set) -> tuple[bool, str]:
    """Ready only when the final file exists, header parses, size matches the header,
    and it holds exactly the index's names for that shard."""
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
    if not expected_names.issubset(names):
        missing = expected_names - names
        return False, f"missing tensors {sorted(missing)[:3]}"
    return True, "ready"


def wait_for_shard(path: Path, expected_names: set, poll: float, wait: bool) -> None:
    while True:
        ok, why = shard_ready(path, expected_names)
        if ok:
            return
        if not wait:
            raise SystemExit(f"shard not ready and --wait is off: {path.name}: {why}")
        log(f"waiting for {path.name}: {why}")
        time.sleep(poll)


def source_sha256(src: Path, shard_file: str) -> str | None:
    """Publisher's sha256 from the hf-download ``.metadata`` sidecar (line 2 = git-lfs oid)."""
    meta = src / ".cache" / "huggingface" / "download" / (shard_file + ".metadata")
    try:
        lines = meta.read_text().splitlines()
        return lines[1].strip() if len(lines) >= 2 else None
    except OSError:
        return None


# --------------------------------------------------------------------------
# journal (resume): lives under state_dir, never inside the artifact
# --------------------------------------------------------------------------
def journal_path(state_dir: Path, layer: int) -> Path:
    return state_dir / f"engram-L{layer}.journal.json"


def build_bin_path(state_dir: Path, layer: int) -> Path:
    return state_dir / f"engram-L{layer}.bin"


def load_journal(state_dir: Path, layer: int) -> dict:
    p = journal_path(state_dir, layer)
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def save_journal(state_dir: Path, layer: int, data: dict) -> None:
    p = journal_path(state_dir, layer)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, p)


# --------------------------------------------------------------------------
# per-layer conversion
# --------------------------------------------------------------------------
def convert_layer(
    layer: int,
    src: Path,
    out: Path,
    weight_map: dict,
    *,
    state_dir: Path,
    chunk_rows: int,
    max_rows: int,
    wait: bool,
    poll: float,
    fsync_every: int = 8,
    expected_rows: int | None = None,
    row_codec: str = "affine",
    finalize: bool = True,
) -> dict:
    """Convert one engram table to a row bank.  Returns the manifest layer entry.

    ``row_codec="affine"`` (default): requantize the FP8-dequantized rows to affine-8
    (272 B/row).  In-progress bin lives under ``state_dir``; the completed layer is
    ``os.replace``-d into ``out/engram-L{layer}.bin`` (the artifact holds only finished files).

    ``row_codec="mxfp8"``: EXACT byte repack of the source (264 B/row).  In-progress bytes are
    written to ``out/engram-L{layer}.bin.new`` (alongside the affine bank, same filesystem so
    the final rename is atomic); the journal still lives outside the artifact under
    ``state_dir``.  With ``finalize=True`` the ``.bin.new`` is renamed over
    ``out/engram-L{layer}.bin`` here; with ``finalize=False`` it is left in place so the caller
    can rename every layer and rewrite the manifest together (see :func:`finalize_mxfp8`).
    The returned entry carries the bank file's ``sha256``.
    """
    if row_codec not in ("affine", "mxfp8"):
        raise ValueError(f"unknown row_codec {row_codec!r}")
    li = dc.ENGRAM_LAYER_IDS.index(layer)
    wname = f"layers.{layer}.engram.embed.weight"
    sname = f"layers.{layer}.engram.embed.scale"
    shard_file = weight_map[wname]
    if weight_map[sname] != shard_file:
        raise RuntimeError(f"weight/scale of layer {layer} in different shards")
    shard = src / shard_file

    expected = {wname, sname}
    wait_for_shard(shard, expected, poll, wait)

    header, data_start = dc.read_safetensors_header(str(shard))
    entries = dc.tensor_entries(header)
    we, se = entries[wname], entries[sname]
    if we.dtype != "F8_E4M3":
        raise RuntimeError(f"{wname} dtype {we.dtype} != F8_E4M3")
    if se.dtype != "F8_E8M0":
        raise RuntimeError(f"{sname} dtype {se.dtype} != F8_E8M0")
    if len(we.shape) != 2 or we.shape[1] != dc.ENGRAM_HEAD_DIM:
        raise RuntimeError(f"{wname} shape {we.shape} != [N, {dc.ENGRAM_HEAD_DIM}]")
    n_rows = we.shape[0]
    groups32 = dc.ENGRAM_HEAD_DIM // dc.ENGRAM_FP8_BLOCK
    if se.shape != (n_rows, groups32):
        raise RuntimeError(f"{sname} shape {se.shape} != {(n_rows, groups32)}")
    cfg_rows = dc.ENGRAM_NUM_EMBEDDINGS[li] if expected_rows is None else expected_rows
    if n_rows != cfg_rows:
        raise RuntimeError(f"layer {layer} rows {n_rows} != expected num_embeddings {cfg_rows}")
    num_embeddings = dc.ENGRAM_NUM_EMBEDDINGS[li] if expected_rows is None else n_rows

    rec_bytes = MXFP8_RECORD_BYTES if row_codec == "mxfp8" else dc.ENGRAM_RECORD_BYTES
    n_eff = n_rows if max_rows <= 0 else min(n_rows, max_rows)
    total_bytes = n_eff * rec_bytes
    n_chunks = (n_eff + chunk_rows - 1) // chunk_rows

    final = out / f"engram-L{layer}.bin"

    def _entry(bank_file: Path | None) -> dict:
        if row_codec == "mxfp8":
            sha = _hash_file(bank_file) if bank_file is not None and bank_file.is_file() else None
            return _layer_manifest_entry_mxfp8(
                layer, n_eff, num_embeddings, shard_file, src, we, se, sha)
        return _layer_manifest_entry(layer, n_eff, num_embeddings, shard_file, src, we, se)

    if row_codec == "mxfp8":
        # staging lives IN the artifact (.bin.new); the affine bank at `final` has a different
        # size (272 B/row) so it never masquerades as a finished mxfp8 bank.
        out.mkdir(parents=True, exist_ok=True)
        bin_path = out / f"engram-L{layer}.bin.new"
        if final.is_file() and final.stat().st_size == total_bytes:
            log(f"L{layer}: mxfp8 bank already finalized ({total_bytes} B) -- skip")
            return _entry(final)
    else:
        if final.is_file() and final.stat().st_size == total_bytes:
            log(f"L{layer}: final bin already present and correct size ({total_bytes} B) -- skip")
            return _entry(final)
        state_dir.mkdir(parents=True, exist_ok=True)
        bin_path = build_bin_path(state_dir, layer)

    state_dir.mkdir(parents=True, exist_ok=True)
    journal = load_journal(state_dir, layer)
    ok = (journal.get("n_eff") == n_eff and journal.get("record_bytes") == rec_bytes
          and journal.get("row_codec", "affine") == row_codec)
    done = set(journal.get("done_chunks", [])) if ok else set()
    if not done:
        journal = {"layer": layer, "n_eff": n_eff, "record_bytes": rec_bytes,
                   "chunk_rows": chunk_rows, "n_chunks": n_chunks, "done_chunks": [],
                   "row_codec": row_codec}

    if not (bin_path.is_file() and os.stat(bin_path).st_size == total_bytes and
            len(done) == n_chunks):
        wfd = os.open(str(shard), os.O_RDONLY)
        bfd = os.open(str(bin_path), os.O_RDWR | os.O_CREAT, 0o644)
        w_base = data_start + we.begin
        s_base = data_start + se.begin
        started = time.time()
        rows_done_before = len(done) * chunk_rows
        try:
            for c in range(n_chunks):
                if c in done:
                    continue
                a = c * chunk_rows
                b = min(a + chunk_rows, n_eff)
                rows = b - a
                wraw = dc._pread_exact(wfd, w_base + a * dc.ENGRAM_HEAD_DIM, rows * dc.ENGRAM_HEAD_DIM)
                sraw = dc._pread_exact(wfd, s_base + a * groups32, rows * groups32)
                wu8 = np.frombuffer(wraw, dtype=np.uint8).reshape(rows, dc.ENGRAM_HEAD_DIM)
                su8 = np.frombuffer(sraw, dtype=np.uint8).reshape(rows, groups32)
                if row_codec == "mxfp8":
                    rec = _mxfp8_chunk_records(wu8, su8)  # exact byte repack, [rows, 264]
                else:
                    f32 = dc.dequant_engram_embed(wu8, su8)
                    rec = dc.engram_chunk_records(f32)  # [rows, rec_bytes] uint8
                payload = rec.tobytes()
                if len(payload) != rows * rec_bytes:
                    raise RuntimeError(f"chunk {c}: payload {len(payload)} != {rows * rec_bytes}")
                _pwrite_all(bfd, payload, a * rec_bytes)
                done.add(c)
                journal["done_chunks"] = sorted(done)
                if (c + 1) % fsync_every == 0 or c == n_chunks - 1:
                    os.fsync(bfd)
                    save_journal(state_dir, layer, journal)
                    elapsed = max(1e-6, time.time() - started)
                    rows_this = (b) - rows_done_before
                    mbps = (rows_this * rec_bytes) / elapsed / 1e6
                    eta = (n_eff - b) * rec_bytes / max(1e-6, (rows_this * rec_bytes) / elapsed)
                    log(f"L{layer}[{row_codec}]: chunk {c+1}/{n_chunks} rows {b}/{n_eff} "
                        f"({100*b/n_eff:.1f}%) {mbps:.1f} MB/s ETA {eta/60:.1f} min")
            os.fsync(bfd)
            save_journal(state_dir, layer, journal)
        finally:
            os.close(wfd)
            os.close(bfd)

    actual = os.stat(bin_path).st_size
    if actual != total_bytes:
        raise RuntimeError(f"L{layer}: bin size {actual} != expected {total_bytes}")

    if row_codec == "mxfp8" and not finalize:
        journal["completed_staging"] = True
        save_journal(state_dir, layer, journal)
        log(f"L{layer}: mxfp8 staging complete -> {bin_path} ({total_bytes} B, {n_eff} rows) "
            f"[awaiting finalize]")
        return _entry(bin_path)

    out.mkdir(parents=True, exist_ok=True)
    os.replace(bin_path, final)  # atomic move into the artifact (same filesystem)
    journal["completed"] = True
    save_journal(state_dir, layer, journal)
    log(f"L{layer}[{row_codec}]: complete -> {final} ({total_bytes} B, {n_eff} rows)")
    return _entry(final)


def _pwrite_all(fd: int, payload: bytes, offset: int) -> None:
    view = memoryview(payload)
    pos = offset
    while view:
        n = os.pwrite(fd, view, pos)
        if n <= 0:
            raise RuntimeError("short pwrite")
        pos += n
        view = view[n:]


def _layer_manifest_entry(layer: int, rows: int, num_embeddings: int, shard_file: str,
                          src: Path, we, se) -> dict:
    return {
        "layer_id": layer,
        "file": f"engram-L{layer}.bin",
        "rows": rows,
        "num_embeddings": num_embeddings,
        "record_bytes": dc.ENGRAM_RECORD_BYTES,
        "total_bytes": rows * dc.ENGRAM_RECORD_BYTES,
        "quant": {"bits": dc.ENGRAM_BITS, "group_size": dc.ENGRAM_GROUP_SIZE,
                  "mode": "affine", "head_dim": dc.ENGRAM_HEAD_DIM},
        "record_layout": dc.engram_record_layout(),
        "source": {
            "shard_file": shard_file,
            "weight_tensor": f"layers.{layer}.engram.embed.weight",
            "scale_tensor": f"layers.{layer}.engram.embed.scale",
            "weight_dtype": we.dtype,
            "weight_shape": list(we.shape),
            "scale_dtype": se.dtype,
            "scale_shape": list(se.shape),
            "fp8_block": dc.ENGRAM_FP8_BLOCK,
            "sha256": source_sha256(src, shard_file),
        },
    }


def _layer_manifest_entry_mxfp8(layer: int, rows: int, num_embeddings: int, shard_file: str,
                                src: Path, we, se, bank_sha256: str | None) -> dict:
    """Manifest entry for an mxfp8 bank: 264 B/row, U32[64] codes + U8[8] e8m0 scales, no bias.

    ``bank_sha256`` is the sha256 of the written ``.bin`` (streamed), added per David's request.
    """
    return {
        "layer_id": layer,
        "file": f"engram-L{layer}.bin",
        "rows": rows,
        "num_embeddings": num_embeddings,
        "record_bytes": MXFP8_RECORD_BYTES,
        "total_bytes": rows * MXFP8_RECORD_BYTES,
        "sha256": bank_sha256,
        "quant": {"bits": 8, "group_size": MXFP8_GROUP_SIZE,
                  "mode": "mxfp8", "head_dim": dc.ENGRAM_HEAD_DIM},
        "record_layout": mxfp8_record_layout(),
        "source": {
            "shard_file": shard_file,
            "weight_tensor": f"layers.{layer}.engram.embed.weight",
            "scale_tensor": f"layers.{layer}.engram.embed.scale",
            "weight_dtype": we.dtype,
            "weight_shape": list(we.shape),
            "scale_dtype": se.dtype,
            "scale_shape": list(se.shape),
            "fp8_block": dc.ENGRAM_FP8_BLOCK,
            "sha256": source_sha256(src, shard_file),
            "exact_repack": True,
        },
    }


def finalize_mxfp8(out: Path, layers: list[int]) -> None:
    """Atomically rename each layer's ``engram-L{L}.bin.new`` over ``engram-L{L}.bin``.

    Call after every layer's staging file is complete + verified, immediately before rewriting
    the manifest, so the artifact flips from the affine banks to the mxfp8 banks in one step
    (each rename is atomic; the manifest is updated right after).  Idempotent: a layer whose
    ``.bin.new`` is already gone (renamed on a prior run) is left as-is.
    """
    for L in layers:
        staging = out / f"engram-L{L}.bin.new"
        final = out / f"engram-L{L}.bin"
        if staging.is_file():
            os.replace(staging, final)
            log(f"L{L}: mxfp8 finalize -> {final}")


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------
def build_hashing_block() -> dict:
    """Every constant the runtime needs to map a token to its engram row indices,
    so it never has to reread the reference ``inference/engram.py``."""
    primes = dc.engram_prime_layout()
    per_layer = []
    for li, lid in enumerate(dc.ENGRAM_LAYER_IDS):
        offsets, flat, total = dc.engram_flat_offsets(primes[li])
        per_layer.append({
            "layer_id": lid,
            "primes": [list(pn) for pn in primes[li]],   # [n-gram size 2..max][head]
            "flat_offsets": offsets,                       # per (n-gram,head), row-space base
            "total_rows": total,                           # == num_embeddings
        })
    mult = dc.compute_engram_hash_multipliers()
    return {
        "layer_ids": list(dc.ENGRAM_LAYER_IDS),
        "max_ngram_size": dc.ENGRAM_MAX_NGRAM_SIZE,
        "n_heads": dc.ENGRAM_N_HEADS,
        "head_dim": dc.ENGRAM_HEAD_DIM,
        "vocab_size": dc.ENGRAM_VOCAB_SIZE,
        "compressed_vocab_size": dc.ENGRAM_COMPRESSED_VOCAB_SIZE,
        "pad_id": dc.ENGRAM_PAD_ID,
        "n_hash_cols": dc.engram_n_hash_cols(),
        "rows_gathered_per_token_per_layer": dc.engram_n_hash_cols(),
        "hash_multipliers": mult.astype(object).tolist(),   # int64, [n_layers][max_ngram_size]
        "per_layer": per_layer,
        "hash_recipe": (
            "compressed = token_map[input_ids]; DEAD(-1) for masked tokens. For shift in "
            "0..max_ngram_size-1: gather cache[pos-shift] (clamp>=0), block when pos<shift or "
            "source==DEAD, fill blocked with pad_id (=token_map[engram_pad_id]). "
            "products = tokens * hash_multipliers[layer]; rolling XOR across shifts 1..n; "
            "row = (rolling %% primes[layer][ngram][head]) + flat_offsets. Row is a global index "
            "into the layer's bank (0..num_embeddings)."
        ),
        "compressed_token_map": {
            "note": (
                "Built at load from the tokenizer (engram.build_compressed_token_map): decode each "
                "token id via backend_tokenizer (skip_special_tokens=False), normalize, and collapse "
                "ids that normalize alike. Partial-UTF8 tokens ('\\ufffd') keyed by raw id_to_token."
            ),
            "expected_size": dc.ENGRAM_COMPRESSED_VOCAB_SIZE,
            "pad_id_raw": dc.ENGRAM_PAD_ID,
            "compressed_pad_id": "token_map[engram_pad_id]  (computed by the port)",
            "normalizer_sequence": [
                "NFKC", "NFD", "StripAccents", "Lowercase",
                "Replace(Regex('[ \\t\\r\\n]+'), ' ')",
                "Replace(Regex('^ $'), sentinel='\\ue000')",
                "Strip",
                "Replace(sentinel='\\ue000', ' ')",
            ],
        },
    }


def read_existing_residents(out: Path) -> dict | None:
    """Return the ``residents`` block of an existing manifest, if any (else ``None``).

    The residents sidecar (``engram-residents.safetensors``) is a separate W4 artifact that
    lives beside the banks; rewriting the manifest for a bank-codec change must not drop it.
    """
    p = out / "engram-manifest.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get("residents")
    except Exception:
        return None


def write_manifest(out: Path, layer_entries: list[dict], *, row_codec: str = "affine",
                   residents: dict | None = None) -> Path:
    dequant = {
        "source_weight_dtype": "F8_E4M3",
        "source_scale_dtype": "F8_E8M0",
        "source_scale_group": dc.ENGRAM_FP8_BLOCK,
        "formula": "f32 = E4M3_LUT[weight] * 2**(scale_byte-127) per (row, 32-col group)",
    }
    if row_codec == "mxfp8":
        quant = {"bits": 8, "group_size": MXFP8_GROUP_SIZE, "mode": "mxfp8",
                 "head_dim": dc.ENGRAM_HEAD_DIM, "record_bytes": MXFP8_RECORD_BYTES}
        dequant["note"] = (
            "mxfp8: the bank stores the source F8_E4M3 code bytes (U32[64]) and F8_E8M0 scale "
            "bytes (U8[8]) verbatim; dequantize with mx.dequantize(mode='mxfp8', group_size=32, "
            "bits=8), bit-exact to the formula above."
        )
    else:
        quant = {"bits": dc.ENGRAM_BITS, "group_size": dc.ENGRAM_GROUP_SIZE,
                 "mode": "affine", "head_dim": dc.ENGRAM_HEAD_DIM,
                 "record_bytes": dc.ENGRAM_RECORD_BYTES}
    manifest = {
        "format": MANIFEST_FORMAT,
        "model_key": dc.MODEL_KEY,
        "source_repo": "deepseek-ai/DeepSeek-V4.1-Flash",
        "source_revision": dc.SOURCE_REVISION,
        "dequant": dequant,
        "quant": quant,
        "layers": sorted(layer_entries, key=lambda e: e["layer_id"]),
    }
    if residents is not None:               # preserve the W4 residents sidecar entry (after layers)
        manifest["residents"] = residents
    manifest["hashing"] = build_hashing_block()
    payload = json.dumps(manifest, indent=2).encode()
    manifest["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    final = out / "engram-manifest.json"
    tmp = final.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, final)
    return final


# --------------------------------------------------------------------------
# resident Engram projections -> one q8/exact sidecar + manifest `residents` entry
# --------------------------------------------------------------------------
def _hash_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blob in iter(lambda: f.read(chunk), b""):
            h.update(blob)
    return h.hexdigest()


def _wkv_parity(src_f32: np.ndarray, packed, scales, biases) -> dict:
    """Dequant-roundtrip parity of the q8 wkv vs the FP8-dequantized source.

    Returns per-row min cosine, flattened cosine, and max-abs error against
    ``src_f32`` (the exact ``dequant_fp8_block`` output).
    """
    mx = dc._cpu_mx()
    deq = np.array(
        mx.dequantize(packed, scales, biases, group_size=RESIDENT_GROUP,
                      bits=RESIDENT_BITS, mode="affine").astype(mx.float32)
    )
    diff = np.abs(deq - src_f32)
    num = np.sum(src_f32 * deq, axis=1)
    den = np.linalg.norm(src_f32, axis=1) * np.linalg.norm(deq, axis=1) + 1e-30
    cos_row = num / den
    a, b = src_f32.reshape(-1), deq.reshape(-1)
    cos_flat = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return {
        "wkv_cos_row_min": float(cos_row.min()),
        "wkv_cos_row_mean": float(cos_row.mean()),
        "wkv_cos_flat": cos_flat,
        "wkv_max_abs_err": float(diff.max()),
        "wkv_mean_abs_err": float(diff.mean()),
    }


def convert_residents(
    src: Path,
    out: Path,
    weight_map: dict,
    *,
    layers=dc.ENGRAM_LAYER_IDS,
    wait: bool,
    poll: float,
    verify: bool = True,
) -> tuple[dict, dict, Path]:
    """Build ``engram/engram-residents.safetensors`` (does NOT touch the manifest).

    Streams only the four resident tensors per layer from the source shard (never
    mmaps the 95 GiB shard): wkv weight/scale (157 MB) + q/k (40 KB each).  Returns
    ``(residents_manifest_entry, parity_by_layer, sidecar_path)``.
    """
    mx = dc._cpu_mx()
    residents: dict = {}
    per_layer_meta: list[dict] = []
    parity: dict = {}

    for L in layers:
        wkv_w = f"layers.{L}.engram.{RESIDENT_WKV}.weight"
        wkv_s = f"layers.{L}.engram.{RESIDENT_WKV}.scale"
        q_name = f"layers.{L}.engram.q_weight"
        k_name = f"layers.{L}.engram.k_weight"
        for nm in (wkv_w, wkv_s, q_name, k_name):
            if nm not in weight_map:
                raise RuntimeError(f"{nm} absent from source index")
        shard_file = weight_map[wkv_w]
        if any(weight_map[nm] != shard_file for nm in (wkv_s, q_name, k_name)):
            raise RuntimeError(f"layer {L} resident tensors split across shards")
        shard = src / shard_file
        expected = {wkv_w, wkv_s, q_name, k_name}
        wait_for_shard(shard, expected, poll, wait)

        header, data_start = dc.read_safetensors_header(str(shard))
        entries = dc.tensor_entries(header)
        we, se = entries[wkv_w], entries[wkv_s]
        qe, ke = entries[q_name], entries[k_name]
        if we.dtype != "F8_E4M3":
            raise RuntimeError(f"{wkv_w} dtype {we.dtype} != F8_E4M3")
        if se.dtype != "F8_E8M0":
            raise RuntimeError(f"{wkv_s} dtype {se.dtype} != F8_E8M0")
        if len(we.shape) != 2 or we.shape[1] % RESIDENT_GROUP:
            raise RuntimeError(f"{wkv_w} shape {we.shape} not q8/gs{RESIDENT_GROUP} quantizable")

        fd = os.open(str(shard), os.O_RDONLY)
        try:
            w_raw = dc._pread_exact(fd, data_start + we.begin, we.end - we.begin)
            s_raw = dc._pread_exact(fd, data_start + se.begin, se.end - se.begin)
            q_raw = dc._pread_exact(fd, data_start + qe.begin, qe.end - qe.begin)
            k_raw = dc._pread_exact(fd, data_start + ke.begin, ke.end - ke.begin)
        finally:
            os.close(fd)

        w_u8 = np.frombuffer(w_raw, dtype=np.uint8).reshape(we.shape)
        s_u8 = np.frombuffer(s_raw, dtype=np.uint8).reshape(se.shape)
        wkv_f32 = dc.dequant_fp8_block(w_u8, s_u8)             # [out, in] f32
        packed, scales, biases = dc.quantize_affine(
            wkv_f32, bits=RESIDENT_BITS, group_size=RESIDENT_GROUP)
        base = f"layers.{L}.engram.{RESIDENT_WKV}"
        residents[f"{base}.weight"] = packed
        residents[f"{base}.scales"] = scales
        residents[f"{base}.biases"] = biases

        # q/k: kept exact, widened losslessly to F32 (source is BF16 for this checkpoint)
        q_f32 = np.ascontiguousarray(dc.raw_to_f32(qe, q_raw), dtype=np.float32)
        k_f32 = np.ascontiguousarray(dc.raw_to_f32(ke, k_raw), dtype=np.float32)
        residents[q_name] = mx.array(q_f32)
        residents[k_name] = mx.array(k_f32)

        lmeta = {
            "layer_id": L,
            "source_shard": shard_file,
            "source_sha256": source_sha256(src, shard_file),
            "wkv": {
                "weight": f"{base}.weight", "scales": f"{base}.scales",
                "biases": f"{base}.biases",
                "source_tensor": wkv_w, "scale_tensor": wkv_s,
                "source_dtype": we.dtype, "source_scale_dtype": se.dtype,
                "source_shape": list(we.shape), "source_scale_shape": list(se.shape),
                "fp8_block": [we.shape[0] // se.shape[0], we.shape[1] // se.shape[1]],
            },
            "q_weight": {"name": q_name, "source_tensor": q_name,
                         "source_dtype": qe.dtype, "shape": list(qe.shape)},
            "k_weight": {"name": k_name, "source_tensor": k_name,
                         "source_dtype": ke.dtype, "shape": list(ke.shape)},
        }
        if verify:
            p = _wkv_parity(wkv_f32, packed, scales, biases)
            q_err = float(np.max(np.abs(np.array(residents[q_name]) - q_f32)))
            k_err = float(np.max(np.abs(np.array(residents[k_name]) - k_f32)))
            p["q_max_abs_err"] = q_err
            p["k_max_abs_err"] = k_err
            p["q_exact"] = q_err == 0.0
            p["k_exact"] = k_err == 0.0
            parity[L] = p
            lmeta["parity"] = p
            log(f"L{L}: wkv q8 parity cos_row_min={p['wkv_cos_row_min']:.6f} "
                f"cos_flat={p['wkv_cos_flat']:.6f} max_abs_err={p['wkv_max_abs_err']:.4e} "
                f"| q/k exact={p['q_exact'] and p['k_exact']}")
        per_layer_meta.append(lmeta)
        del wkv_f32, q_f32, k_f32

    # -- write the sidecar atomically (temp keeps the .safetensors extension) --
    mx.eval(*residents.values())
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / RESIDENT_SIDECAR.replace(".safetensors", ".partial.safetensors")
    mx.save_safetensors(str(tmp), residents, metadata={"format": "mlx"})
    sidecar = out / RESIDENT_SIDECAR
    os.replace(tmp, sidecar)

    # -- authoritative tensor list from the written file header ----------------
    header, _ = dc.read_safetensors_header(str(sidecar))
    tensors_meta = [
        {"name": nm, "dtype": header[nm]["dtype"], "shape": list(header[nm]["shape"])}
        for nm in sorted(k for k in header if k != "__metadata__")
    ]
    entry = {
        "file": RESIDENT_SIDECAR,
        "total_bytes": sidecar.stat().st_size,
        "sha256": _hash_file(sidecar),
        "quant": {
            "wkv": {"bits": RESIDENT_BITS, "group_size": RESIDENT_GROUP, "mode": "affine"},
            "q_weight": "f32-exact", "k_weight": "f32-exact",
        },
        "dequant": {
            "source_weight_dtype": "F8_E4M3",
            "source_scale_dtype": "F8_E8M0",
            "formula": "f32 = E4M3_LUT[weight] * 2**(scale_byte-127) per source block "
                       "(same FP8 block-scale dequant as the streamed residents)",
        },
        "tensors": tensors_meta,
        "layers": per_layer_meta,
    }
    log(f"sidecar -> {sidecar} ({entry['total_bytes']} B, {len(tensors_meta)} tensors)")
    return entry, parity, sidecar


def update_manifest_with_residents(out: Path, residents_entry: dict,
                                   backup_path: Path | None = None) -> Path:
    """Add/replace the manifest ``residents`` entry and recompute ``manifest_sha256``.

    The sha recipe is identical to :func:`write_manifest`: sha256 over
    ``json.dumps(manifest, indent=2)`` of the manifest *without* the ``manifest_sha256``
    field, then that field appended.  Only ``engram-manifest.json`` is rewritten (the
    banks / shards / experts.bin are never touched).  A copy of the pre-update manifest
    is written to ``backup_path`` first (outside the artifact).
    """
    final = out / "engram-manifest.json"
    manifest = json.loads(final.read_text())
    if backup_path is not None:
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(final.read_text())
        log(f"manifest backup -> {backup_path}")

    manifest.pop("manifest_sha256", None)
    manifest.pop("residents", None)  # idempotent re-run: replace, never duplicate
    # insert `residents` right after `layers` for a stable, logical key order
    rebuilt: dict = {}
    for k, v in manifest.items():
        rebuilt[k] = v
        if k == "layers":
            rebuilt["residents"] = residents_entry
    if "residents" not in rebuilt:
        rebuilt["residents"] = residents_entry
    manifest = rebuilt

    payload = json.dumps(manifest, indent=2).encode()
    manifest["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    tmp = final.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, final)
    log(f"manifest updated (manifest_sha256={manifest['manifest_sha256']}) -> {final}")
    return final


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("banks", "residents"), default="banks",
                    help="banks: convert the row banks + write the manifest (default). "
                         "residents: build the engram-residents.safetensors sidecar and "
                         "add a `residents` entry to the existing manifest.")
    ap.add_argument("--src", type=Path, required=True, help="source dir with model-000{47,48}-of-00048.safetensors")
    ap.add_argument("--out", type=Path, required=True, help="artifact engram/ output dir")
    ap.add_argument("--index", type=Path, required=True, help="source model.safetensors.index.json")
    ap.add_argument("--layers", default="1,14", help="engram layer ids to convert")
    ap.add_argument("--state-dir", type=Path, default=Path(DEFAULT_STATE_DIR))
    ap.add_argument("--chunk-rows", type=int, default=65536)
    ap.add_argument("--max-rows", type=int, default=0, help="pilot: limit rows per layer (0 = all)")
    ap.add_argument("--wait", action="store_true", help="poll for shard readiness")
    ap.add_argument("--poll-interval", type=float, default=30.0)
    ap.add_argument("--no-manifest", action="store_true", help="skip manifest write")
    ap.add_argument("--manifest-backup", type=Path, default=None,
                    help="residents mode: write a copy of the pre-update manifest here "
                         "(outside the artifact) before adding the residents entry")
    ap.add_argument("--no-verify", action="store_true",
                    help="residents mode: skip the q8 dequant-roundtrip parity check")
    ap.add_argument("--row-codec", choices=("affine", "mxfp8"), default="affine",
                    help="banks mode: 'affine' (272 B/row requantize, default) or 'mxfp8' "
                         "(264 B/row EXACT repack of the source E4M3 codes + E8M0 scales; "
                         "writes engram-L{L}.bin.new then atomically renames over the bank "
                         "and rewrites the manifest in one step).")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    for L in layers:
        if L not in dc.ENGRAM_LAYER_IDS:
            raise SystemExit(f"layer {L} not an engram layer {dc.ENGRAM_LAYER_IDS}")

    idx = load_index(args.index)
    weight_map = idx["weight_map"]

    if args.mode == "residents":
        log(f"pid {os.getpid()} | mode residents | layers {layers} | out {args.out} | "
            f"sidecar {RESIDENT_SIDECAR}")
        entry, parity, sidecar = convert_residents(
            args.src, args.out, weight_map, layers=layers,
            wait=args.wait, poll=args.poll_interval, verify=not args.no_verify,
        )
        if not args.no_manifest:
            update_manifest_with_residents(args.out, entry, backup_path=args.manifest_backup)
        log("done")
        return 0

    codec = args.row_codec
    rec_bytes = MXFP8_RECORD_BYTES if codec == "mxfp8" else dc.ENGRAM_RECORD_BYTES
    log(f"pid {os.getpid()} | mode banks | row_codec {codec} | layers {layers} | "
        f"chunk_rows {args.chunk_rows} | max_rows {args.max_rows or 'all'} | out {args.out} | "
        f"state {args.state_dir} | record_bytes {rec_bytes}")

    # mxfp8: stage every layer to .bin.new, then flip all banks + manifest together (one step).
    finalize_each = codec != "mxfp8"
    # preserve the W4 residents sidecar entry across a manifest rewrite (captured pre-flip)
    residents = read_existing_residents(args.out) if codec == "mxfp8" else None
    entries = []
    for L in layers:
        entry = convert_layer(
            L, args.src, args.out, weight_map,
            state_dir=args.state_dir, chunk_rows=args.chunk_rows,
            max_rows=args.max_rows, wait=args.wait, poll=args.poll_interval,
            row_codec=codec, finalize=finalize_each,
        )
        entries.append(entry)

    if codec == "mxfp8":
        finalize_mxfp8(args.out, layers)   # atomic rename each .bin.new over the affine bank

    if not args.no_manifest:
        mpath = write_manifest(args.out, entries, row_codec=codec, residents=residents)
        if residents is not None:
            log(f"manifest: preserved residents entry ({residents.get('file')})")
        log(f"manifest -> {mpath}")

    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
