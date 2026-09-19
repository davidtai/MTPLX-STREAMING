"""W56 / KERNEL_LEDGER K27 -- MTPLX_DSV41_LAYOUT_FIX sorted-indices routed gather.

The layout fix sorts the routed-expert gather's rows by bank slot and passes
``sorted_indices=True`` so ``mx.gather_qmm`` takes the fused weight-streamed-once
kernel on Metal.  On CPU there is a single ``gather_qmm`` implementation and the
change is a permutation + its inverse over an M-independent per-row matmul, so it
must be BYTE-IDENTICAL to the shipped unsorted call for every wave shape --
decode (M=1, 6 rows), the K+1 verify batch (24 rows), a chunk-major prefill wave
and a layer-major prefill wave.  These tests pin MLX to CPU
([[worker-tests-must-pin-mlx-cpu]]) and never touch Metal.
"""

import os

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mtplx.models import expert_mlx


class _Bank:
    """Minimal component-bank stand-in: only ``.arrays`` is read by the gather."""

    def __init__(self, arrays):
        self.arrays = arrays


def _make_bank(E, N, K, *, codec, group_size=32, bits=4):
    """One synthetic expert bank quantized in the switch's codec."""
    mode = "mxfp4" if codec == "mxfp4" else "affine"
    arrays = {}
    mx.random.seed(1234)
    for proj, (out, inp) in {
        "gate_proj": (N, K),
        "up_proj": (N, K),
        "down_proj": (K, N),
    }.items():
        dense = mx.random.normal((E, out, inp)).astype(mx.bfloat16)
        if mode == "mxfp4":
            w, sc = mx.quantize(dense, group_size=group_size, bits=bits, mode="mxfp4")
            arrays[f"{proj}.weight"] = w
            arrays[f"{proj}.scales"] = sc
        else:
            w, sc, bi = mx.quantize(dense, group_size=group_size, bits=bits)
            arrays[f"{proj}.weight"] = w
            arrays[f"{proj}.scales"] = sc
            arrays[f"{proj}.biases"] = bi
    return _Bank(arrays)


def _gather(x, bank, slot, *, codec, group_size=32, bits=4):
    return expert_mlx._gather_component_bank(
        x, bank, slot, group_size=group_size, bits=bits,
        swiglu_limit=10.0, codec=codec,
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        expert_mlx._LAYOUT_FIX_ENV,
        expert_mlx._LAYOUT_FIX_MIN_ROWS_ENV,
        expert_mlx._GATHER_ROWS_PER_CALL_ENV,
    ):
        monkeypatch.delenv(key, raising=False)
    yield


# (name, rows) covering decode / verify-batch / chunk-major / layer-major waves.
_WAVES = [
    ("decode_m1", 6),          # 1 token x top_k 6
    ("verify_k1", 24),         # 4 rows x top_k 6
    ("prefill_chunk", 3072),   # ~512 tok x 6, clears the 2048 default
    ("prefill_layer_major", 8192),
]


@pytest.mark.parametrize("codec", ["mxfp4", "affine"])
@pytest.mark.parametrize("name,rows", _WAVES)
def test_layout_fix_byte_identical(monkeypatch, codec, name, rows):
    E, N, K = 12, 128, 256
    bank = _make_bank(E, N, K, codec=codec)
    mx.random.seed(rows)
    x = mx.random.normal((rows, K)).astype(mx.bfloat16)
    slot = mx.random.randint(0, E, (rows,)).astype(mx.int32).reshape((-1, 1))

    # Control: flag off -> exact shipped unsorted gather.
    out_off = _gather(x, bank, slot, codec=codec)

    # Layout fix on, threshold low enough that even the 6-row decode wave sorts,
    # so the sorted path itself is exercised at every shape (not just skipped).
    monkeypatch.setenv(expert_mlx._LAYOUT_FIX_ENV, "1")
    monkeypatch.setenv(expert_mlx._LAYOUT_FIX_MIN_ROWS_ENV, "1")
    out_on = _gather(x, bank, slot, codec=codec)

    mx.eval(out_off, out_on)
    assert out_off.shape == (rows, K)
    assert mx.array_equal(out_off, out_on), (
        f"{codec}/{name}: sorted gather not byte-identical "
        f"(max|d|={float(mx.max(mx.abs(out_off - out_on)))})"
    )


@pytest.mark.parametrize("codec", ["mxfp4", "affine"])
def test_default_threshold_leaves_small_waves_untouched(monkeypatch, codec):
    """With the default 2048 min-rows, decode/verify never sort: flag on == off."""
    E, N, K = 12, 128, 256
    bank = _make_bank(E, N, K, codec=codec)
    for rows in (6, 24):
        mx.random.seed(rows)
        x = mx.random.normal((rows, K)).astype(mx.bfloat16)
        slot = mx.random.randint(0, E, (rows,)).astype(mx.int32).reshape((-1, 1))
        out_off = _gather(x, bank, slot, codec=codec)
        monkeypatch.setenv(expert_mlx._LAYOUT_FIX_ENV, "1")  # default threshold 2048
        out_on = _gather(x, bank, slot, codec=codec)
        monkeypatch.delenv(expert_mlx._LAYOUT_FIX_ENV, raising=False)
        mx.eval(out_off, out_on)
        assert mx.array_equal(out_off, out_on)


@pytest.mark.parametrize("per_call", [512, 1000, 4096])
def test_rows_per_call_is_byte_identical(monkeypatch, per_call):
    """Chunking the sorted wave by rows-per-call must not change per-row math."""
    codec = "mxfp4"
    E, N, K = 12, 128, 256
    bank = _make_bank(E, N, K, codec=codec)
    rows = 3072
    mx.random.seed(7)
    x = mx.random.normal((rows, K)).astype(mx.bfloat16)
    slot = mx.random.randint(0, E, (rows,)).astype(mx.int32).reshape((-1, 1))

    out_off = _gather(x, bank, slot, codec=codec)

    monkeypatch.setenv(expert_mlx._LAYOUT_FIX_ENV, "1")
    monkeypatch.setenv(expert_mlx._LAYOUT_FIX_MIN_ROWS_ENV, "1")
    monkeypatch.setenv(expert_mlx._GATHER_ROWS_PER_CALL_ENV, str(per_call))
    out_on = _gather(x, bank, slot, codec=codec)

    mx.eval(out_off, out_on)
    assert mx.array_equal(out_off, out_on)
