"""CPU tests for the F39 tcq3 serve-decode install (tcq/serve_install.py): the decode wrapper math equals
tcq_encode.effective_weight on the F34 sample, the monkeypatch installs on the real expert_mlx module and round-trips
(and is a no-op when the env is unset), and the record-validator truth table.  No serve, no GPU (the tile-kernel
dispatch is Metal — validated by serve_parity.py).  MLX pinned to CPU by conftest.py.
"""
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_TRELLIS = Path(__file__).resolve().parents[1]
_DSV41 = Path(__file__).resolve().parents[2]
_ROOT = Path(__file__).resolve().parents[4]
for p in (str(_TRELLIS), str(_DSV41), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tcq import serve_install as S      # noqa: E402

F34_SAMPLE = os.environ.get(
    "MTPLX_DSV41_F34_SAMPLE",
    "/Users/davidtai/projects/OpenSourceWTF/reports/dsv41-f34-trellis/dsv41_f34_L20E3_down_proj_full_beam.npz")
SPEC = types.SimpleNamespace(hidden_size=5120, expert_hidden_size=2304)


# ---------------------------------------------------------------- decode wrapper math == tcq_encode.effective_weight

def test_decode_wrapper_math_matches_tcq_encode_on_f34():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import tcq_runtime as R
    import tcq_encode as enc
    if not os.path.exists(F34_SAMPLE):
        pytest.skip(f"F34 sample absent: {F34_SAMPLE}")
    z = np.load(F34_SAMPLE)
    code, rout = z["escha_code"][0], z["escha_rout"][0].astype(np.float32)
    W_q = R.decode_wq(code)
    mx.eval(W_q)
    out_dim = int(W_q.shape[1])                                   # 5120 (down_proj)
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((4, int(W_q.shape[0]))).astype(np.float32))
    # the SERVE wrapper's t128/rout math around a CPU matmul (= xh @ W_q); the GPU tile kernel is serve_parity's job
    y = S._tcq3_project(x, code_bank=None, rout_bank=mx.array(rout[None, :]),
                        slot_ids=mx.array([0, 0, 0, 0], mx.uint32), out_dim=out_dim, tables=None,
                        matmul=lambda xh: xh @ W_q)
    E_ref = enc.effective_weight(np.array(W_q).astype(np.float32), np.ones(int(W_q.shape[0]), np.float32), rout)
    rel = float(np.sqrt(((np.array(y) - np.array(x) @ E_ref) ** 2).sum()) / np.sqrt(((np.array(x) @ E_ref) ** 2).sum()))
    assert rel < 1e-5, f"serve decode-wrapper rel err {rel:.3e}"


# ---------------------------------------------------------------- validator truth table

def test_validator_accepts_tcq3_and_rejects_variants():
    good = S.tcq3_expected_signature(SPEC)
    # correct tcq3 record: 6 segments, code I16 + rout F16 per projection, no scales
    assert [c for c, *_ in good] == ["gate_proj.code", "gate_proj.rout", "up_proj.code", "up_proj.rout",
                                     "down_proj.code", "down_proj.rout"]
    S.validate_tcq3_record(good, SPEC)                            # accepts
    # wrong dtype
    bad_dtype = list(good); bad_dtype[0] = ("gate_proj.code", "U32", (320, 144, 48), 320 * 144 * 48 * 2)
    with pytest.raises(ValueError):
        S.validate_tcq3_record(tuple(bad_dtype), SPEC)
    # mxfp4-style record (weight+scales) must be rejected
    mxfp4 = tuple((f"{p}.weight", "U32", (2304, 640), 5898240) for p in ("gate_proj", "up_proj", "down_proj"))
    with pytest.raises(ValueError):
        S.validate_tcq3_record(mxfp4, SPEC)
    # a scale leaf present -> reject (tcq3 has no scales)
    with_scales = good[:1] + (("gate_proj.scales", "U8", (2304, 160), 368640),) + good[1:]
    with pytest.raises(ValueError):
        S.validate_tcq3_record(with_scales, SPEC)


def test_expected_signature_matches_transcode_bank_layout():
    # code lengths: gate/up [320,144,48], down [144,320,48]; routs [2304]/[5120] f16 — the F38 record.
    sig = dict((c, (dt, sh, ln)) for c, dt, sh, ln in S.tcq3_expected_signature(SPEC))
    assert sig["gate_proj.code"] == ("I16", (320, 144, 48), 320 * 144 * 48 * 2)
    assert sig["down_proj.code"] == ("I16", (144, 320, 48), 144 * 320 * 48 * 2)
    assert sig["gate_proj.rout"] == ("F16", (2304,), 2304 * 2)
    assert sig["down_proj.rout"] == ("F16", (5120,), 5120 * 2)
    # whole record = sum of segment lengths = 13,290,496 B
    assert sum(ln for _, _, _, ln in S.tcq3_expected_signature(SPEC)) == 13_290_496


# ---------------------------------------------------------------- monkeypatch installs + round-trips + no-op when unset

def test_install_rebinds_dispatch_and_uninstall_round_trips():
    from mtplx.models import expert_mlx  # noqa: F401
    classes = S._switch_classes()
    assert classes, "expected at least one streamed switch class with _dispatch_component_bank"
    before = {c: c._dispatch_component_bank for c in classes}
    S.install(tables="DUMMY")                                     # no GPU: tables unused unless a tcq3 switch decodes
    try:
        assert all(c._dispatch_component_bank is not before[c] for c in classes)   # rebound
        # idempotent: second install is a no-op
        assert S.install(tables="DUMMY") == {"already": True}
    finally:
        S.uninstall()
    assert all(c._dispatch_component_bank is before[c] for c in classes)           # restored byte-for-byte


def test_install_from_env_is_noop_when_flag_unset(monkeypatch):
    monkeypatch.delenv(S.ENV_FLAG, raising=False)
    assert S.install_from_env() is None
    classes = S._switch_classes()
    before = {c: c._dispatch_component_bank for c in classes}
    S.install_from_env()                                          # flag unset -> must not touch the stock decode
    assert all(c._dispatch_component_bank is before[c] for c in classes)


def test_patched_dispatch_leaves_non_tcq3_switch_on_stock_path():
    # a fake mxfp4 switch: the patched dispatch must delegate to the original (not the tcq3 branch)
    classes = S._switch_classes()
    cls = classes[0]
    calls = {}
    before = cls._dispatch_component_bank

    def fake_orig(self, selected, bindings, *, dense_prefill=False):
        calls["orig"] = (self.codec, dense_prefill)
        return "stock"
    cls._dispatch_component_bank = fake_orig
    try:
        S._INSTALLED["orig"] = {}          # ensure a clean install over the fake original
        S.install(tables="DUMMY")
        fake = types.SimpleNamespace(codec="mxfp4", swiglu_limit=10.0)
        out = cls._dispatch_component_bank(fake, None, (), dense_prefill=False)
        assert out == "stock" and calls["orig"] == ("mxfp4", False)   # delegated to stock
    finally:
        S.uninstall()
        cls._dispatch_component_bank = before
