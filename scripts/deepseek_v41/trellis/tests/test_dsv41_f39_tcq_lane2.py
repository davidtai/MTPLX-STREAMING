"""CPU tests for the F39 boundary-closing work: tcq3 admission re-derivation, loader monkeypatch, the two-file
stager (round-trip on the REAL retained sources + stock-preserving growth gate), and the reader on a FULL
15,360-record SPARSE dry artifact.  No GPU.  MLX pinned to CPU by conftest.py.
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

from tcq import tcq_admission as A            # noqa: E402
from tcq import loader_install as L           # noqa: E402
from tcq import stage_tcq_runner as S         # noqa: E402

_RECEIPTS = _ROOT / "docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed"
REAL_MANIFEST = os.environ.get(
    "MTPLX_DSV41_SRC_MANIFEST",
    os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4/expert-manifest.json"))
GIB = 1024 ** 3


def _mxfp4_admission(cap=101, fixed=10_000_000_000, cache=256 * 1024 ** 2, allocator=90_000_000_000,
                     host_reserve=1_400_000_000):
    """A self-consistent mxfp4 admission result dict (steady = fixed + slots + PACKED)."""
    return dict(
        decode_slots_per_layer=cap,
        steady_decode_active_bound_bytes=A._slot_term(cap, A.WEIGHTS) + A.PACKED + fixed,
        decode_cache_allowance_bytes=cache,
        allocator_limit_bytes=allocator,
        host_reserve_bytes=host_reserve,
        growth_payload_bytes=5_000_000_000,
        prefill_active_bound_bytes=60_000_000_000,
    )


# ---------------------------------------------------------------- admission re-derivation

def test_admission_admits_more_rows_and_holds_ceilings():
    base, wired = 20_000_000_000, 15_000_000_000
    mxfp4 = _mxfp4_admission()
    res = A.retarget(mxfp4, base=base, wired=wired)
    assert res["decode_slots_per_layer"] > mxfp4["decode_slots_per_layer"]          # more experts resident
    assert res["extra_rows_vs_mxfp4"] == res["decode_slots_per_layer"] - mxfp4["decode_slots_per_layer"]
    # every unchanged ceiling holds
    active, cache = res["active_bound_bytes"], mxfp4["decode_cache_allowance_bytes"]
    assert res["physical_bound_bytes"] <= A.DEFAULT_BOX_BUDGET_BYTES
    assert active + cache <= mxfp4["allocator_limit_bytes"]
    assert wired + active + cache + GIB <= 100 * GIB
    # charge routs, credit retired scales
    assert res["resident_rout_bytes"] == 298_844_160 and res["resident_packed_scales_bytes"] == 0
    assert res["retired_scale_credit_bytes"] == A.PACKED
    assert res["tcq3_scale_credit_vs_mxfp4_bytes"] == A.PACKED - A.ROUTS == 2_787_291_900
    assert res["decode_weight_record_bytes"] == 13_290_496 and res["expert_codec"] == "tcq3"


def test_admission_gain_at_least_slot_ratio_when_scales_dominated_out():
    # the per-slot shrink alone (17,694,720 / 13,290,496 = 1.331x) is the floor; freed scales add more
    base, wired = 20_000_000_000, 10_000_000_000
    mxfp4 = _mxfp4_admission()
    res = A.retarget(mxfp4, base=base, wired=wired)
    ratio = res["decode_slots_per_layer"] / mxfp4["decode_slots_per_layer"]
    assert ratio >= A.WEIGHTS / A.TCQ3_SLOT - 0.02, ratio        # >= ~1.33x (freed scales push it higher)


def test_admission_tighter_envelope_admits_fewer():
    base, wired = 20_000_000_000, 15_000_000_000
    loose = A.retarget(_mxfp4_admission(allocator=90_000_000_000), base=base, wired=wired)["decode_slots_per_layer"]
    # a tighter allocator limit binds sooner -> fewer rows admitted (still a positive capacity)
    tight = A.retarget(_mxfp4_admission(allocator=78_000_000_000), base=base, wired=wired)["decode_slots_per_layer"]
    assert loose > tight > 0, (loose, tight)


def test_admission_raises_when_nothing_fits():
    # an envelope with no room even at the mxfp4 capacity
    with pytest.raises(RuntimeError, match="no tcq3 decode capacity"):
        A.retarget(_mxfp4_admission(), base=110_000_000_000, wired=99 * GIB)


# ---------------------------------------------------------------- loader monkeypatch

def test_manifest_support_accepts_tcq3_and_preserves_mxfp4():
    from mtplx import expert_manifest as em
    L.install_manifest_support()
    assert em.expert_components_for_mode("tcq3") == L.TCQ3_MANIFEST_COMPONENTS
    # mxfp4 (and the 6-component order) unchanged
    assert em.expert_components_for_mode("mxfp4") == (
        "gate_proj.weight", "gate_proj.scales", "up_proj.weight", "up_proj.scales",
        "down_proj.weight", "down_proj.scales")
    assert L.install_manifest_support() is False          # idempotent


def test_spec_support_reports_tcq3_record_bytes():
    from mtplx.expert_streaming_models import ExpertStreamingModelSpec as Spec
    L.install_spec_support()
    assert getattr(Spec, "_tcq3_installed", False) is True
    # the patched property returns 13,290,496 for a tcq3-codec object (delegates otherwise)
    class _Fake:
        expert_codec = "tcq3"
    assert Spec.expert_record_bytes.fget(_Fake()) == 13_290_496
    assert L.install_spec_support() is False              # idempotent


def test_loader_facts():
    assert L.TCQ3_RECORD_BYTES == 13_290_496
    assert L.tcq3_manifest_components() == (
        "gate_proj.code", "gate_proj.rout", "up_proj.code", "up_proj.rout", "down_proj.code", "down_proj.rout")
    assert L.WHOLE_RECORD_COMPONENT == "expert_record"


# ---------------------------------------------------------------- stager (both files, real sources)

def test_stager_round_trips_packed_phase():
    src = (_RECEIPTS / "packed_phase.py").read_text()
    updated = S.rewrite_packed_phase(src)
    assert "route_plane_lane" in updated and "MTPLX_DSV41_TCQ3" in updated
    assert updated.replace(S._PL_ROUTED, S._PL_ANCHOR).replace(S._GG_ROUTED, S._GG_ANCHOR) == src


def test_stager_round_trips_run_full():
    src = (_RECEIPTS / "run_full.py").read_text()
    updated = S.rewrite_run_full(src)
    for tok in ("install_tcq_loader", "stamp_spec_tcq3", "_tcq_adm.retarget",
                "install_growth_tcq3", "grown to decode capacity in install_growth_tcq3",
                "MTPLX_DSV41_TCQ3') == '1' else Path('/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4')"):
        assert tok in updated
    rev = updated
    for old, new in ((S._MODEL_ANCHOR, S._MODEL_ROUTED), (S._LOAD_ANCHOR, S._LOAD_ROUTED),
                     (S._STAMP_ANCHOR, S._STAMP_ROUTED), (S._ADM_ANCHOR, S._ADM_ROUTED),
                     (S._GROWTH_ANCHOR, S._GROWTH_ROUTED), (S._GROWROWS_ANCHOR, S._GROWROWS_ROUTED)):
        rev = rev.replace(new, old)
    assert rev == src


def test_stager_rejects_missing_anchor():
    with pytest.raises(RuntimeError, match="anchor not found exactly once"):
        S.rewrite_packed_phase("def f():\n    return 1\n")
    with pytest.raises(RuntimeError, match="anchor not found exactly once"):
        S.rewrite_run_full("x = 1\n")


def test_growth_gate_truth_table_preserves_stock():
    """The staged growth codec gate must be behaviorally identical to the original when the lane is off."""
    cond = ("(rt_record != 18800640 or rt_codec != 'mxfp4') and not "
            "(tcq3_env == '1' and rt_codec == 'tcq3' and rt_record == 13290496)")
    orig = "(rt_record != 18800640 or rt_codec != 'mxfp4')"

    def ev(expr, record, codec, env):
        return eval(expr, {}, {"rt_record": record, "rt_codec": codec, "tcq3_env": env})

    # flag OFF: identical to the original for every codec (stock preserved)
    for record, codec in [(18800640, "mxfp4"), (13290496, "tcq3"), (999, "affine")]:
        assert ev(cond, record, codec, "0") == ev(orig, record, codec, "0")
    # flag ON: mxfp4 still accepted (gate False), tcq3 now accepted (gate False), junk still rejected (True)
    assert ev(cond, 18800640, "mxfp4", "1") is False
    assert ev(cond, 13290496, "tcq3", "1") is False
    assert ev(cond, 999, "affine", "1") is True


# ---------------------------------------------------------------- sparse full-scale dry artifact

@pytest.fixture(scope="module")
def sparse_artifact(tmp_path_factory):
    if not os.path.exists(REAL_MANIFEST):
        pytest.skip(f"source manifest absent: {REAL_MANIFEST}")
    from tcq import build_sparse_dry_artifact as B
    out = tmp_path_factory.mktemp("tcq3-sparse")
    info = B.build(str(out), REAL_MANIFEST, written_idx=[0, 15359])
    return str(out), info


def test_sparse_artifact_full_scale_reader(sparse_artifact):
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import tcq_runtime as R
    art, info = sparse_artifact
    assert info["n_records"] == 15360 and info["logical_bytes"] == 15360 * 13_290_496
    # sparse: on-disk far smaller than logical (holes)
    assert info["disk_blocks"] * 512 < info["logical_bytes"] // 100
    man = R.read_tcq_manifest(os.path.join(art, "expert-manifest.json"))
    assert man.n_records == 15360
    g = man.geometry

    def down_nonzero(idx):
        rec = man.records[idx]
        with open(os.path.join(art, "experts.bin"), "rb") as fh:
            fh.seek(R.record_base_offset(rec, g))
            buf = fh.read(g.record_bytes)
        return int((R.slice_record(buf, g)["down_proj.code"] != 0).sum())

    assert down_nonzero(0) > 0 and down_nonzero(15359) > 0        # written records
    assert down_nonzero(5000) == 0                               # hole reads as zeros
    assert R.global_index(man.records[15359]["layer"], man.records[15359]["expert"], g) == 15359
