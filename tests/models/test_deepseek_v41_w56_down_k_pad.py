"""W56 / KERNEL_LEDGER K27 (F2) -- MTPLX_DSV41_DOWN_K_PAD down-projection K padding.

The expert down-proj contracts over K = moe_intermediate_size = 2304, and
2304 % 512 = 256, so the mxfp4 fast `gather_qmv` (K % 512 == 0) is disabled for the
down gather while gate/up (K = hidden 5120) hit it. Padding the down K to 2560 with
ZERO columns re-enables the fast kernel and is byte-identical (a zero column
contributes exactly 0; a zeroed mxfp4 gs32 group dequantizes to exactly 0.0 for the
scale byte 0 a `mx.zeros` bank tail carries -- only 0xFF/NaN is unsafe).

These tests build a fake padded down bank with `pad_mxfp4_down_component` and prove
the flag-on gather over the padded bank equals the flag-off gather over the unpadded
bank, for the M=1 decode wave, the M=4 (K+1) verify batch, and a prefill wave. CPU
only ([[worker-tests-must-pin-mlx-cpu]]); no Metal.
"""

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mtplx.models import expert_mlx


class _Bank:
    def __init__(self, arrays):
        self.arrays = arrays


def _make_down_bank(E, hidden, inter, *, pad, gs=32, bits=4):
    """A fake mxfp4 bank; down-proj optionally K-padded to the fast-qmv width."""
    mx.random.seed(20260911)
    arrays = {}
    # gate/up: [inter, hidden] (K = hidden 5120, already %512==0)
    for proj in ("gate_proj", "up_proj"):
        dense = mx.random.normal((E, inter, hidden)).astype(mx.bfloat16)
        w, sc = mx.quantize(dense, group_size=gs, bits=bits, mode="mxfp4")
        arrays[f"{proj}.weight"] = w
        arrays[f"{proj}.scales"] = sc
    # down: [hidden, inter] (K = inter 2304)
    dense = mx.random.normal((E, hidden, inter)).astype(mx.bfloat16)
    w, sc = mx.quantize(dense, group_size=gs, bits=bits, mode="mxfp4")
    if pad:
        # pad each expert's down component; vmap-free: pad the whole [E, hidden, K]
        w, sc = expert_mlx.pad_mxfp4_down_component(w, sc, group_size=gs, bits=bits)
    arrays["down_proj.weight"] = w
    arrays["down_proj.scales"] = sc
    return _Bank(arrays)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(expert_mlx._DOWN_K_PAD_ENV, raising=False)
    monkeypatch.delenv(expert_mlx._LAYOUT_FIX_ENV, raising=False)
    yield


_WAVES = [("decode_m1", 6), ("verify_k1", 24), ("prefill", 512)]


@pytest.mark.parametrize("name,rows", _WAVES)
def test_down_k_pad_byte_identical(monkeypatch, name, rows):
    E, hidden, inter = 8, 96, 2304  # real inter (2304 % 512 = 256), small hidden
    unpadded = _make_down_bank(E, hidden, inter, pad=False)
    padded = _make_down_bank(E, hidden, inter, pad=True)
    # padded down weight must be 2560-wide (2560/8 = 320 packed uint32)
    assert int(padded.arrays["down_proj.weight"].shape[-1]) * 8 == 2560
    assert int(unpadded.arrays["down_proj.weight"].shape[-1]) * 8 == 2304

    mx.random.seed(rows)
    x = mx.random.normal((rows, hidden)).astype(mx.bfloat16)
    slot = mx.random.randint(0, E, (rows,)).astype(mx.int32).reshape((-1, 1))

    # control: flag off, unpadded bank
    out_ctrl = expert_mlx._gather_component_bank(
        x, unpadded, slot, group_size=32, bits=4, swiglu_limit=10.0, codec="mxfp4"
    )
    # fix: flag on, padded bank (gather pads the activation to 2560 to match)
    monkeypatch.setenv(expert_mlx._DOWN_K_PAD_ENV, "1")
    out_pad = expert_mlx._gather_component_bank(
        x, padded, slot, group_size=32, bits=4, swiglu_limit=10.0, codec="mxfp4"
    )
    mx.eval(out_ctrl, out_pad)
    assert out_ctrl.shape == (rows, hidden)
    assert mx.array_equal(out_ctrl, out_pad), (
        f"{name}: down-K pad not byte-identical "
        f"(max|d|={float(mx.max(mx.abs(out_ctrl - out_pad)))})"
    )


def test_flag_on_unpadded_bank_is_noop(monkeypatch):
    """Flag on but the bank is unpadded (real runtime before admission padding):
    k_wt == k_act -> no activation pad -> byte-identical to flag off."""
    E, hidden, inter = 8, 96, 2304
    bank = _make_down_bank(E, hidden, inter, pad=False)
    mx.random.seed(3)
    x = mx.random.normal((10, hidden)).astype(mx.bfloat16)
    slot = mx.random.randint(0, E, (10,)).astype(mx.int32).reshape((-1, 1))
    out_off = expert_mlx._gather_component_bank(
        x, bank, slot, group_size=32, bits=4, swiglu_limit=10.0, codec="mxfp4"
    )
    monkeypatch.setenv(expert_mlx._DOWN_K_PAD_ENV, "1")
    out_on = expert_mlx._gather_component_bank(
        x, bank, slot, group_size=32, bits=4, swiglu_limit=10.0, codec="mxfp4"
    )
    mx.eval(out_off, out_on)
    assert mx.array_equal(out_off, out_on)


def test_pad_transform_and_slot_bytes():
    # A zeroed mxfp4 group dequantizes to exactly 0.0 (scale byte 0), so the padded
    # tail is exact; verify the transform widens to the next 512 multiple.
    hidden, inter = 96, 2304
    dense = mx.random.normal((hidden, inter)).astype(mx.bfloat16)
    w, sc = mx.quantize(dense, group_size=32, bits=4, mode="mxfp4")
    wp, sp = expert_mlx.pad_mxfp4_down_component(w, sc, group_size=32, bits=4)
    assert int(wp.shape[-1]) * 8 == 2560
    assert int(sp.shape[-1]) * 32 == 2560
    # the padded tail dequantizes to exactly zero
    dq = mx.dequantize(wp, sp, group_size=32, bits=4, mode="mxfp4")
    mx.eval(dq)
    assert bool(mx.all(dq[:, 2304:2560] == 0.0))
    assert not bool(mx.any(mx.isnan(dq)))

    a = expert_mlx.down_k_pad_slot_bytes(hidden=5120, inter=2304)
    assert a["k_pad"] == 2560
    assert a["down_delta_bytes"] == 696320               # +0.664 MiB
    assert a["down_padded_bytes"] == a["down_unpadded_bytes"] + 696320
    # +11.1% on the down component
    assert abs(a["down_delta_bytes"] / a["down_unpadded_bytes"] - 0.1111) < 1e-3
