"""CPU tests for the DSV4.1 F39 tcq3 runtime reader (tcq_runtime.py) + the stride-aware tile kernel.

MLX is pinned to CPU by conftest.py AND defensively in each test that touches arrays ('no GPU' == CPU, not
Metal).  No test opens a GPU window; the stride kernel is only *constructed* (mx.fast.metal_kernel), never launched.

Covers the spec's four CPU tests:
  1. manifest / record reader on a dry-run tcq3 artifact (built here via F38 transcode_bank's own dry-run path);
  2. effective-weight path (t128 -> matmul -> t128 * rout) == tcq_encode.effective_weight on the F34 real sample
     (escha_code[0], escha_rout[0]) to 1e-5 relative, using the vendor decoder for W_q;
  3. the stride-aware kernel source is generated with the right stride / per-projection offset constants;
  4. row / slot accounting for the tcq3 record.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# trellis dir (for tcq_runtime / tcq_encode / tcq_kernels / transcode_bank) and worktree root (for `mtplx`).
_TRELLIS = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[4]
for p in (str(_TRELLIS), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import tcq_encode as enc          # noqa: E402
import tcq_kernels as tk          # noqa: E402
import tcq_runtime as R           # noqa: E402

REAL_MANIFEST = os.environ.get(
    "MTPLX_DSV41_SRC_MANIFEST",
    os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4/expert-manifest.json"))
F34_SAMPLE = os.environ.get(
    "MTPLX_DSV41_F34_SAMPLE",
    "/Users/davidtai/projects/OpenSourceWTF/reports/dsv41-f34-trellis/dsv41_f34_L20E3_down_proj_full_beam.npz")

N_DRY = 6


@pytest.fixture(scope="module")
def dry_artifact(tmp_path_factory):
    """Build a small tcq3 artifact with the F38 dry-run encoder (deterministic random codes), CPU-only, no GPU."""
    if not os.path.exists(REAL_MANIFEST):
        pytest.skip(f"source manifest absent: {REAL_MANIFEST}")
    import transcode_bank as tb
    src = json.load(open(REAL_MANIFEST))
    src["records"] = src["records"][:N_DRY]
    out = tmp_path_factory.mktemp("tcq3-dry")
    segs_rel = tb.record_layout(src["records"][0])
    rec_bytes = tb.record_bytes(segs_rel)
    encoder = tb.DryEncoder()
    with open(out / "experts.bin", "wb") as fh:
        fh.truncate(N_DRY * rec_bytes)
    shas = []
    with open(out / "experts.bin", "r+b") as fh:
        for i, r in enumerate(src["records"]):
            parts = {}
            for comp in tb.COMPONENTS:
                seg = next(x for x in segs_rel if x["component"] == f"{comp}.code")
                nI, nJ, _ = seg["shape"]
                code, rout = encoder.encode(r["layer"], r["expert"], comp, nI, nJ, nJ * 16)
                parts[comp] = (code, rout, 0)
            shas.append(tb.write_record(fh, i * rec_bytes, segs_rel, parts))
    manifest = tb.build_manifest(src, segs_rel, rec_bytes, shas, 256, 10, str(out))
    json.dump(manifest, open(out / "expert-manifest.json", "w"))
    return str(out), src


# ---------------------------------------------------------------- 1. manifest / record reader

def test_manifest_reader_accepts_tcq3_and_validates_geometry(dry_artifact):
    art, _src = dry_artifact
    man = R.read_tcq_manifest(os.path.join(art, "expert-manifest.json"))
    assert man.n_records == N_DRY
    g = man.geometry
    assert g.record_bytes == 13_290_496 and g.record_words == 6_645_248
    # segments in record order: gate.code, gate.rout, up.code, up.rout, down.code, down.rout
    assert [s.component for s in g.segments] == [
        "gate_proj.code", "gate_proj.rout", "up_proj.code", "up_proj.rout", "down_proj.code", "down_proj.rout"]
    assert g.code_segment("gate_proj").shape == (320, 144, 48)
    assert g.code_segment("down_proj").shape == (144, 320, 48)
    assert g.rout_segment("gate_proj").shape == (2304,) and g.rout_segment("down_proj").shape == (5120,)


def test_slice_record_round_trips_dry_codes(dry_artifact):
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    art, _src = dry_artifact
    man = R.read_tcq_manifest(os.path.join(art, "expert-manifest.json"))
    g = man.geometry
    rec = man.records[3]
    with open(os.path.join(art, "experts.bin"), "rb") as fh:
        fh.seek(R.record_base_offset(rec, g))
        buf = fh.read(g.record_bytes)
    segs = R.slice_record(buf, g)
    # the F38 DryEncoder is deterministic: code = rng(layer*1000003 + expert*1009 + comp_index)
    for ci, comp in enumerate(R.COMPONENTS):
        rng = np.random.default_rng(rec["layer"] * 1000003 + rec["expert"] * 1009 + ci)
        nI, nJ, _ = g.code_segment(comp).shape
        expected = rng.integers(-32768, 32767, size=(nI, nJ, 48), dtype=np.int16)
        assert np.array_equal(segs[f"{comp}.code"], expected), f"{comp}.code differs"
        assert segs[f"{comp}.rout"].shape == g.rout_segment(comp).shape


def test_resident_routs_shapes_and_index(dry_artifact):
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    art, _src = dry_artifact
    man = R.read_tcq_manifest(os.path.join(art, "expert-manifest.json"))
    routs = R.load_resident_routs(man)
    mx.eval(*routs.values())
    assert tuple(routs["gate_proj"].shape) == (N_DRY, 2304)
    assert tuple(routs["up_proj"].shape) == (N_DRY, 2304)
    assert tuple(routs["down_proj"].shape) == (N_DRY, 5120)
    # DryEncoder routs are ones; global index for L0E3 == 3
    gi = R.global_index(man.records[3]["layer"], man.records[3]["expert"], man.geometry)
    assert gi == 3
    assert bool((np.array(routs["down_proj"][gi]) == 1).all())


def test_manifest_reader_rejects_non_tcq3(dry_artifact, tmp_path):
    art, _src = dry_artifact
    m = json.load(open(os.path.join(art, "expert-manifest.json")))
    m["quantization"] = {"mode": "mxfp4", "bits": 4, "group_size": 32}
    bad = tmp_path / "mxfp4-manifest.json"
    json.dump(m, open(bad, "w"))
    with pytest.raises(ValueError, match="not a tcq3"):
        R.read_tcq_manifest(str(bad))


def test_manifest_reader_rejects_wrong_geometry(dry_artifact, tmp_path):
    art, _src = dry_artifact
    m = json.load(open(os.path.join(art, "expert-manifest.json")))
    # corrupt the first record's gate code length -> geometry validation must fail
    for s in m["records"][0]["segments"]:
        if s["component"] == "gate_proj.code":
            s["length"] += 2
    bad = tmp_path / "bad-geom-manifest.json"
    json.dump(m, open(bad, "w"))
    with pytest.raises(ValueError):
        R.read_tcq_manifest(str(bad))


# ---------------------------------------------------------------- 2. effective-weight path == tcq_encode reference

def _load_f34():
    if not os.path.exists(F34_SAMPLE):
        pytest.skip(f"F34 sample absent: {F34_SAMPLE}")
    z = np.load(F34_SAMPLE)
    return z["escha_code"][0], z["escha_rin"][0], z["escha_rout"][0]


def test_effective_weight_matches_tcq_encode_on_f34_sample():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    code, rin, rout = _load_f34()
    assert float(rin.min()) == 1.0 and float(rin.max()) == 1.0, "rin must be 1 (tcq3)"
    W_q = R.decode_wq(code)                                   # vendor decoder, fp16 [2304, 5120]
    mx.eval(W_q)
    E_mine = np.array(R.effective_weight_mx(W_q, mx.array(rout.astype(np.float32))).astype(mx.float32))
    E_ref = enc.effective_weight(np.array(W_q).astype(np.float32),
                                 np.ones(W_q.shape[0], np.float32), rout.astype(np.float32))
    rel = float(np.sqrt(((E_mine - E_ref) ** 2).sum()) / np.sqrt((E_ref ** 2).sum()))
    assert rel < 1e-5, f"effective-weight rel err {rel:.3e}"


def test_forward_decode_verify_equals_x_at_effective_weight_on_f34():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    code, _rin, rout = _load_f34()
    W_q = R.decode_wq(code)
    mx.eval(W_q)
    E_ref = enc.effective_weight(np.array(W_q).astype(np.float32),
                                 np.ones(W_q.shape[0], np.float32), rout.astype(np.float32))
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, W_q.shape[0])).astype(np.float32)   # [rows, IN]
    y_fwd = np.array(R.forward_decode_verify(mx.array(x), W_q, mx.array(rout.astype(np.float32))))
    y_ref = x @ E_ref
    rel = float(np.sqrt(((y_fwd - y_ref) ** 2).sum()) / np.sqrt((y_ref ** 2).sum()))
    assert rel < 1e-5, f"forward rel err {rel:.3e}"


def test_prefill_decode_to_bf16_matches_effective_weight_on_f34():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    code, _rin, rout = _load_f34()
    E_bf16 = np.array(R.decode_expert_to_bf16(code, rout.astype(np.float32)).astype(mx.float32))
    W_q = R.decode_wq(code)
    mx.eval(W_q)
    E_ref = enc.effective_weight(np.array(W_q).astype(np.float32),
                                 np.ones(W_q.shape[0], np.float32), rout.astype(np.float32))
    rel = float(np.sqrt(((E_bf16 - E_ref) ** 2).sum()) / np.sqrt((E_ref ** 2).sum()))
    assert rel < 5e-3, f"bf16 prefill decode rel err {rel:.3e}"    # bf16 rounding, ~1e-2 abs slack


# ---------------------------------------------------------------- 3. stride-aware kernel source constants

def test_stride_kernel_source_has_correct_constants():
    stride = tk.TCQ3_RECORD_WORDS
    assert stride == 6_645_248 and stride * 2 == 13_290_496
    offsets = {"gate_proj": 0, "up_proj": 2_214_144, "down_proj": 4_428_288}
    assert tk.TCQ3_CODE_WORD_OFFSETS == offsets
    for comp, (OUT, IN) in {"gate_proj": (2304, 5120), "up_proj": (2304, 5120), "down_proj": (5120, 2304)}.items():
        src = tk._tile_strided_source(IN, OUT, stride, offsets[comp])
        base = next(l.strip() for l in src.splitlines() if "const device short* base" in l)
        assert base == f"const device short* base = code + ulong(ids[row]) * {stride}ul + {offsets[comp]}ul;"
        # body identical to the contiguous tile variant except the single base line
        s_body = [l for l in src.splitlines() if "base =" not in l]
        c_body = [l for l in tk._tile_source(IN, OUT).splitlines() if "base =" not in l]
        assert s_body == c_body, f"{comp} body diverges beyond the base pointer"


def test_stride_kernel_constructs_and_rejects_bad_geometry():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)                              # construct only; no Metal launch on CPU
    for OUT, IN, comp in [(2304, 5120, "gate_proj"), (2304, 5120, "up_proj"), (5120, 2304, "down_proj")]:
        assert tk.make_tcq_projection_strided(OUT, IN, comp) is not None
    with pytest.raises(ValueError):
        tk.make_tcq_projection_strided(999, 5120, "gate_proj")          # bad OUT/IN
    with pytest.raises(ValueError):
        tk.make_tcq_projection_strided(2304, 5120, "no_such_proj")      # bad component


# ---------------------------------------------------------------- 4. row / slot accounting

def test_row_slot_accounting_matches_expected():
    g = R.validate_geometry(R._expected_geometry())
    assert g.record_bytes == 13_290_496
    assert g.slot_bytes() == 13_290_496                        # whole record per decode slot
    # a cache 'row' = one slot in every one of the 40 MoE layers (how the retained bank counts a dropped row)
    assert g.cache_row_bytes() == 40 * 13_290_496 == 531_619_840
    # mxfp4 today: 40 x 17,694,720 = 707,788,800 B/row
    assert 40 * R.MXFP4_SLOT_BYTES == 707_788_800
    # ~33% more experts resident for the same code memory
    assert abs(g.resident_fraction_vs_mxfp4() - 17_694_720 / 13_290_496) < 1e-9
    assert 1.33 < g.resident_fraction_vs_mxfp4() < 1.34
    # whole-bank routs resident: 15,360 x (2304+2304+5120) x 2 B
    assert g.n_records == 15_360
    assert g.resident_rout_bytes() == 15_360 * (2304 + 2304 + 5120) * 2 == 298_844_160
    # per-projection code word offsets == the kernel's stride constants
    assert {p: g.code_word_offset(p) for p in R.COMPONENTS} == tk.TCQ3_CODE_WORD_OFFSETS
    # rout word offsets sit between the code segments
    assert g.rout_segment("gate_proj").word_offset == 2_211_840
    assert g.rout_segment("up_proj").word_offset == 4_425_984
    assert g.rout_segment("down_proj").word_offset == 6_640_128
