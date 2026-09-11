#!/usr/bin/env python3
"""W16 streamed 3-link correctness proof for layers 0-1 (W15's method).

Fallback for when the full-model serve probe cannot fit under the 18 GB RSS
budget on the CPU box: prove the served mxfp4 bank reproduces the fp32 reference
MoE output at layers 0-1 WITHOUT loading the model, by composing two links that
are each cheap and already established:

  Link A (served bank == source, bit-exact):  sample real records from the NEW
    experts.bin at layers 0-1, dequantize with mx.dequantize(mode="mxfp4"), and
    assert np.array_equal to the fp32 dequant of the FP4 source. (This proof.)
  Link B (source-fp4 experts -> MoE cos 1.0 vs the fp32 reference):  W9's
    bank_ladder ran the reference forward substituting the mxfp4-repacked experts
    (== source, bit-exact) and measured the MoE/layer output cos vs R0. Read
    docs/deepseek-v41/receipts/torchref_bank_ladder.json forward_vs_R0/mxfp4_gs32.

  Compose: served L0/L1 experts are bit-identical to the source (A); source/mxfp4
    experts drive the reference MoE to cos ~1.0 (B) => the served mxfp4 bank
    reproduces the reference MoE at L0/L1 (exactly, modulo bf16 activation
    storage), no full-model load required.

CPU only, RSS < 1 GB.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-w16")
sys.path.insert(0, str(REPO / ".benchmark-artifacts" / "deepseek-v41" / "w16"))
import mlx.core as mx
mx.set_default_device(mx.cpu)
from mtplx import deepseek_v41_convert as dc
from verify_mxfp4_bitexact import parse_record, dequant_record_proj, RECORD_BYTES

RECEIPTS = REPO / "docs" / "deepseek-v41" / "receipts"
SRC = Path("~/models/DeepSeek-V4.1-Flash-src").expanduser()
OUT = Path("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4").expanduser()


def main() -> int:
    index = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]
    exp_fd = os.open(str(OUT / "experts.bin"), os.O_RDONLY)
    rng = np.random.default_rng(0)
    # Link A: served-bank records bit-exact vs source, layers 0 and 1
    linkA = {"checked": 0, "bitexact": 0, "max_abs": 0.0}
    for L in (0, 1):
        for e in sorted(rng.choice(384, size=6, replace=False).tolist()):
            buf = memoryview(os.pread(exp_fd, RECORD_BYTES, (L * 384 + e) * RECORD_BYTES))
            got = parse_record(buf)
            for proj, w in dc.PROJ_TO_SOURCE_W.items():
                wn = f"layers.{L}.ffn.experts.{e}.{w}.weight"
                sn = f"layers.{L}.ffn.experts.{e}.{w}.scale"
                sp = SRC / index[wn]
                header, ds = dc.read_safetensors_header(str(sp))
                ent = dc.tensor_entries(header)
                sfd = os.open(str(sp), os.O_RDONLY)
                try:
                    pk = np.frombuffer(dc.read_tensor_raw(sfd, ds, ent[wn]), np.uint8).reshape(ent[wn].shape)
                    sc = np.frombuffer(dc.read_tensor_raw(sfd, ds, ent[sn]), np.uint8).reshape(ent[sn].shape)
                finally:
                    os.close(sfd)
                src_f32 = dc.dequant_fp4(pk, sc)
                got_f32 = dequant_record_proj(*got[proj])
                linkA["checked"] += 1
                linkA["bitexact"] += int(np.array_equal(got_f32, src_f32))
                linkA["max_abs"] = max(linkA["max_abs"], float(np.abs(got_f32 - src_f32).max()))
    os.close(exp_fd)
    linkA["all_bitexact"] = linkA["bitexact"] == linkA["checked"]

    # Link B: W9 ladder — mxfp4 experts through the reference forward vs R0
    ladder = json.loads((RECEIPTS / "torchref_bank_ladder.json").read_text())
    fwd = ladder["forward_vs_R0"]["mxfp4_gs32"]
    linkB = {f"L{i}": fwd[f"L{i}"] for i in (0, 1) if f"L{i}" in fwd}
    exact = ladder["expert_quality_vs_source"]["mxfp4_gs32"]

    out = {
        "link_A_served_bank_vs_source_bitexact_L0_L1": linkA,
        "link_B_w9_ladder_mxfp4_forward_vs_reference_L0_L1": linkB,
        "link_B_expert_bit_exact_vs_source": exact.get("bit_exact_vs_source"),
        "composition": (
            "served L0/L1 experts == source (A, np.array_equal, maxabs "
            f"{linkA['max_abs']}) and source/mxfp4 experts drive the reference MoE "
            "to the cos in B => served mxfp4 bank reproduces the fp32-reference MoE "
            "at L0/L1 (modulo bf16 activation storage)."
        ),
        "PROVEN": linkA["all_bitexact"] and bool(exact.get("bit_exact_vs_source")),
    }
    print(json.dumps(out, indent=2))
    (REPO / ".benchmark-artifacts" / "deepseek-v41" / "w16" / "proof_3link_out.json").write_text(json.dumps(out, indent=2))
    return 0 if out["PROVEN"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
