"""CPU tests for the F39 Task-3 growth-transition mechanics (tcq/growth.py) and the partial-manifest builder read
against the REAL still-writing bank.  No GPU.  MLX pinned to CPU by conftest.py.

The runtime bank manipulation in install_growth_tcq3.transition() is GPU-deferred; here we test the PURE accounting
(retire = no-op, whole-record slots, routs resident, no packed scales) and validate the tcq3 decode path on REAL
encoder codes read through a partial manifest built from the bank's progress.json.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_TRELLIS = Path(__file__).resolve().parents[1]
_DSV41 = Path(__file__).resolve().parents[2]
_ROOT = Path(__file__).resolve().parents[4]
for p in (str(_TRELLIS), str(_DSV41), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tcq import growth as G                    # noqa: E402

REAL_BANK = os.environ.get("MTPLX_DSV41_TCQ3_BANK",
                           os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-tcq3"))
REAL_SRC = os.environ.get("MTPLX_DSV41_SRC_MANIFEST",
                          os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4/expert-manifest.json"))


# ---------------------------------------------------------------- growth-transition accounting (pure)

def test_growth_report_whole_record_slots_no_scales():
    r = G.tcq3_growth_report(capacity=145, old_capacity=84)
    assert r["decode_slots_per_layer"] == 145 and r["prefill_slots_per_layer"] == 84
    assert r["decode_weight_record_bytes"] == 13_290_496 and r["source_record_bytes"] == 13_290_496
    assert r["resident_packed_scales_bytes"] == 0            # no packed scales
    assert r["resident_rout_bytes"] == 298_844_160           # routs resident instead
    assert r["raw_scale_backing_released_bytes"] == 0        # nothing to retire
    assert r["physical_allocated_bytes"] == (145 * 40 + 48) * 13_290_496
    assert r["persistent_cache_bytes"] == 145 * 40 * 13_290_496


def test_retire_scales_is_a_noop_and_rejects_mxfp4_banks():
    class TcqBank:
        arrays = {"expert_record": object()}
    class Mxfp4Bank:
        arrays = {"gate_proj.weight": object(), "gate_proj.scales": object()}
    assert G.retire_scales_tcq3(TcqBank()) == 0
    assert G.retire_scales_tcq3(object()) == 0               # a bank with no .arrays: still a no-op
    with pytest.raises(RuntimeError, match="not a tcq3 bank"):
        G.retire_scales_tcq3(Mxfp4Bank())


def test_growth_report_scale_credit_vs_mxfp4():
    # mxfp4 would carry 3,086,136,060 B of resident packed scales + 17,694,720 B slots;
    # tcq3 carries 298,844,160 B routs + 13,290,496 B slots. The report reflects the swap.
    r = G.tcq3_growth_report(capacity=101, old_capacity=84)
    assert r["resident_packed_scales_bytes"] == 0
    assert 17_694_720 - r["decode_weight_record_bytes"] == 4_404_224           # smaller slot
    assert 3_086_136_060 - r["resident_rout_bytes"] == 2_787_291_900           # freed scale bytes


# ---------------------------------------------------------------- partial manifest built from the real bank

@pytest.fixture(scope="module")
def partial_manifest(tmp_path_factory):
    if not (os.path.isdir(REAL_BANK) and os.path.exists(os.path.join(REAL_BANK, "progress.json"))
            and os.path.exists(REAL_SRC)):
        pytest.skip("real tcq3 bank / progress.json / source manifest not present")
    from tcq import build_partial_manifest as B
    out = tmp_path_factory.mktemp("tcq3-partial")
    info = B.build(REAL_BANK, str(out), REAL_SRC)
    return str(out), info


def test_partial_manifest_matches_progress(partial_manifest):
    part, info = partial_manifest
    progress = json.load(open(os.path.join(REAL_BANK, "progress.json")))
    assert info["n_records"] == 15360
    # finished count equals progress 'next' (records with a real sha below next)
    assert info["finished_records"] <= progress["next"] and info["finished_records"] >= 1
    man = json.load(open(os.path.join(part, "expert-manifest.json")))
    assert man["quantization"]["mode"] == "tcq3"
    assert man["partial"]["next"] == progress["next"]
    # a finished record carries its real sha (not the zero placeholder)
    assert man["records"][0]["sha256"] == progress["sha256"][0]


def test_partial_manifest_reader_on_real_records(partial_manifest):
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import tcq_runtime as R
    import tcq_encode as E
    part, info = partial_manifest
    man = R.read_tcq_manifest(os.path.join(part, "expert-manifest.json"))
    assert man.n_records == 15360
    g = man.geometry
    nxt = info["next"]

    def read_down(idx):
        rec = man.records[idx]
        with open(os.path.join(part, "experts.bin"), "rb") as fh:
            fh.seek(R.record_base_offset(rec, g))
            buf = fh.read(g.record_bytes)
        return R.slice_record(buf, g)

    seg0 = read_down(0)                                        # record 0 is finished from the first chunk
    assert int((seg0["down_proj.code"] != 0).sum()) > 0
    # decode-verify on REAL encoder codes: runtime MLX path == tcq_encode reference to 1e-5
    W_q = R.decode_wq(seg0["down_proj.code"])
    mx.eval(W_q)
    rout = np.array(seg0["down_proj.rout"]).astype(np.float32)
    E_mine = np.array(R.effective_weight_mx(W_q, mx.array(rout)).astype(mx.float32))
    E_ref = E.effective_weight(np.array(W_q).astype(np.float32), np.ones(2304, np.float32), rout)
    rel = float(np.sqrt(((E_mine - E_ref) ** 2).sum()) / np.sqrt((E_ref ** 2).sum()))
    assert rel < 1e-5, f"real-record effective-weight rel err {rel:.3e}"
    # a record beyond 'next' is an unwritten hole -> zeros
    if nxt < 15360:
        hole = read_down(min(nxt + 50, 15359))
        assert int((hole["down_proj.code"] != 0).sum()) == 0
