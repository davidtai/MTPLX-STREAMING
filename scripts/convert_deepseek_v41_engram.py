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
) -> dict:
    """Convert one engram table to an affine-8 bank.  Returns the manifest layer entry."""
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

    rec_bytes = dc.ENGRAM_RECORD_BYTES
    n_eff = n_rows if max_rows <= 0 else min(n_rows, max_rows)
    total_bytes = n_eff * rec_bytes
    n_chunks = (n_eff + chunk_rows - 1) // chunk_rows

    final = out / f"engram-L{layer}.bin"
    if final.is_file() and final.stat().st_size == total_bytes:
        log(f"L{layer}: final bin already present and correct size ({total_bytes} B) -- skip")
        return _layer_manifest_entry(layer, n_eff, num_embeddings, shard_file, src, we, se)

    state_dir.mkdir(parents=True, exist_ok=True)
    bin_path = build_bin_path(state_dir, layer)
    journal = load_journal(state_dir, layer)
    done = set(journal.get("done_chunks", [])) if journal.get(
        "n_eff") == n_eff and journal.get("record_bytes") == rec_bytes else set()
    if not done:
        journal = {"layer": layer, "n_eff": n_eff, "record_bytes": rec_bytes,
                   "chunk_rows": chunk_rows, "n_chunks": n_chunks, "done_chunks": []}

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
                log(f"L{layer}: chunk {c+1}/{n_chunks} rows {b}/{n_eff} "
                    f"({100*b/n_eff:.1f}%) {mbps:.1f} MB/s ETA {eta/60:.1f} min")
        os.fsync(bfd)
        save_journal(state_dir, layer, journal)
    finally:
        os.close(wfd)
        os.close(bfd)

    actual = os.stat(bin_path).st_size
    if actual != total_bytes:
        raise RuntimeError(f"L{layer}: bin size {actual} != expected {total_bytes}")
    out.mkdir(parents=True, exist_ok=True)
    os.replace(bin_path, final)  # atomic move into the artifact (same filesystem)
    journal["completed"] = True
    save_journal(state_dir, layer, journal)
    log(f"L{layer}: complete -> {final} ({total_bytes} B, {n_eff} rows)")
    return _layer_manifest_entry(layer, n_eff, num_embeddings, shard_file, src, we, se)


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


def write_manifest(out: Path, layer_entries: list[dict]) -> Path:
    manifest = {
        "format": MANIFEST_FORMAT,
        "model_key": dc.MODEL_KEY,
        "source_repo": "deepseek-ai/DeepSeek-V4.1-Flash",
        "source_revision": dc.SOURCE_REVISION,
        "dequant": {
            "source_weight_dtype": "F8_E4M3",
            "source_scale_dtype": "F8_E8M0",
            "source_scale_group": dc.ENGRAM_FP8_BLOCK,
            "formula": "f32 = E4M3_LUT[weight] * 2**(scale_byte-127) per (row, 32-col group)",
        },
        "quant": {"bits": dc.ENGRAM_BITS, "group_size": dc.ENGRAM_GROUP_SIZE,
                  "mode": "affine", "head_dim": dc.ENGRAM_HEAD_DIM,
                  "record_bytes": dc.ENGRAM_RECORD_BYTES},
        "layers": sorted(layer_entries, key=lambda e: e["layer_id"]),
        "hashing": build_hashing_block(),
    }
    payload = json.dumps(manifest, indent=2).encode()
    manifest["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    final = out / "engram-manifest.json"
    tmp = final.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, final)
    return final


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
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
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    for L in layers:
        if L not in dc.ENGRAM_LAYER_IDS:
            raise SystemExit(f"layer {L} not an engram layer {dc.ENGRAM_LAYER_IDS}")

    idx = load_index(args.index)
    weight_map = idx["weight_map"]

    log(f"pid {os.getpid()} | layers {layers} | chunk_rows {args.chunk_rows} | "
        f"max_rows {args.max_rows or 'all'} | out {args.out} | state {args.state_dir} | "
        f"record_bytes {dc.ENGRAM_RECORD_BYTES}")

    entries = []
    for L in layers:
        entry = convert_layer(
            L, args.src, args.out, weight_map,
            state_dir=args.state_dir, chunk_rows=args.chunk_rows,
            max_rows=args.max_rows, wait=args.wait, poll=args.poll_interval,
        )
        entries.append(entry)

    if not args.no_manifest:
        mpath = write_manifest(args.out, entries)
        log(f"manifest -> {mpath}")

    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
