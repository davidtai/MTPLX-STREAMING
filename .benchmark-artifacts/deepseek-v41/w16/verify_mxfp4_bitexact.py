#!/usr/bin/env python3
"""W16 bit-exact sample verification of the mxfp4 experts.bin vs the FP4 source.

For >=64 sampled records spread across all 40 layers: read the record from the
new ``experts.bin``, dequantize each projection with ``mx.dequantize(...,
mode="mxfp4")``, and assert it equals the fp32 dequant of the *source* FP4
expert bit-for-bit (``np.array_equal``).  Native mxfp4 is a lossless repack of
the E2M1+E8M0 source, so equality must be exact (not merely high-cosine).

CPU only, RSS-bounded (one expert at a time).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w16")
sys.path.insert(0, str(REPO))
import mlx.core as mx  # noqa: E402
mx.set_default_device(mx.cpu)
from mtplx import deepseek_v41_convert as dc  # noqa: E402

RECORD_BYTES = 18_800_640
GROUP, BITS = 32, 4
DIM, INTER = dc.HIDDEN_SIZE, dc.MOE_INTERMEDIATE  # 5120, 2304
# (out, in) per projection; packed [out, in/8] u32, scales [out, in/32] u8
PROJ_SHAPE = {"gate_proj": (INTER, DIM), "up_proj": (INTER, DIM), "down_proj": (DIM, INTER)}
PROJ_ORDER = ("gate_proj", "up_proj", "down_proj")


def parse_record(buf: memoryview):
    """Split a record into {proj: (packed_u32[out,in/8], scales_u8[out,in/32])}."""
    out = {}
    cur = 0
    for proj in PROJ_ORDER:
        o, i = PROJ_SHAPE[proj]
        pw_len = o * (i // 8) * 4
        sc_len = o * (i // GROUP) * 1
        packed = np.frombuffer(buf[cur:cur + pw_len], dtype="<u4").reshape(o, i // 8); cur += pw_len
        scales = np.frombuffer(buf[cur:cur + sc_len], dtype=np.uint8).reshape(o, i // GROUP); cur += sc_len
        out[proj] = (packed, scales)
    assert cur == RECORD_BYTES, (cur, RECORD_BYTES)
    return out


def dequant_record_proj(packed, scales):
    q = mx.array(np.ascontiguousarray(packed))
    s = mx.array(np.ascontiguousarray(scales))
    deq = mx.dequantize(q, s, group_size=GROUP, bits=BITS, mode="mxfp4")
    return np.array(deq.astype(mx.float32))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="mxfp4 artifact dir")
    ap.add_argument("--src", type=Path, required=True, help="source model dir")
    ap.add_argument("--index", type=Path, required=True, help="source safetensors index.json")
    ap.add_argument("--per-layer", type=int, default=2, help="experts sampled per layer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default="0-39",
                    help="layer range/list to sample (e.g. '0-4' for already-written shards)")
    args = ap.parse_args()

    src = args.src.expanduser().resolve()
    index = json.loads(args.index.read_text())["weight_map"]
    exp_path = args.out.expanduser().resolve() / "experts.bin"
    exp_fd = os.open(str(exp_path), os.O_RDONLY)

    def _parse(spec):
        out = []
        for part in spec.split(","):
            if "-" in part:
                a, b = part.split("-"); out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
        return sorted(set(out))

    rng = np.random.default_rng(args.seed)
    layers = _parse(args.layers)
    samples = []
    for L in layers:
        for e in sorted(rng.choice(384, size=args.per_layer, replace=False).tolist()):
            samples.append((L, int(e)))
    n_ok = 0
    n_total = 0
    max_abs = 0.0
    t0 = time.time()
    # cache: source shard header per shard file (small)
    hdr_cache: dict[str, tuple] = {}
    for (L, e) in samples:
        rec_index = L * 384 + e
        buf = memoryview(os.pread(exp_fd, RECORD_BYTES, rec_index * RECORD_BYTES))
        got = parse_record(buf)
        for proj, w in dc.PROJ_TO_SOURCE_W.items():
            wname = f"layers.{L}.ffn.experts.{e}.{w}.weight"
            sname = f"layers.{L}.ffn.experts.{e}.{w}.scale"
            shard = index[wname]
            spath = src / shard
            if shard not in hdr_cache:
                hdr_cache[shard] = dc.read_safetensors_header(str(spath))
            header, data_start = hdr_cache[shard]
            entries = dc.tensor_entries(header)
            we, se = entries[wname], entries[sname]
            sfd = os.open(str(spath), os.O_RDONLY)
            try:
                packed = np.frombuffer(dc.read_tensor_raw(sfd, data_start, we), dtype=np.uint8).reshape(we.shape)
                scale = np.frombuffer(dc.read_tensor_raw(sfd, data_start, se), dtype=np.uint8).reshape(se.shape)
            finally:
                os.close(sfd)
            src_f32 = dc.dequant_fp4(packed, scale)
            got_f32 = dequant_record_proj(*got[proj])
            n_total += 1
            exact = np.array_equal(got_f32, src_f32)
            d = float(np.abs(got_f32 - src_f32).max()) if got_f32.shape == src_f32.shape else float("inf")
            max_abs = max(max_abs, d)
            if exact:
                n_ok += 1
            else:
                print(f"  MISMATCH L{L} e{e} {proj}: maxabs={d:.3g} shapes {got_f32.shape} {src_f32.shape}", flush=True)
    os.close(exp_fd)
    dt = time.time() - t0
    result = {
        "sampled_records": len(samples), "checked_weights": n_total,
        "bitexact_weights": n_ok, "all_bitexact": n_ok == n_total,
        "max_abs_diff": max_abs, "seconds": round(dt, 1),
        "layers_covered": len(layers),
    }
    print(json.dumps(result, indent=2))
    return 0 if result["all_bitexact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
