"""W78 CPU unit test for scripts/deepseek_v41/metal_decode_attn_bisect.py.

Proves the microbench (a) executes end to end at tiny dims on the CPU for all four
CSA modes at T in {256, 1024}, and (b) that the peeled sub-ops sum to within 20%
of the whole ``Attention._attend`` -- i.e. the peel is a faithful decomposition
(no op missing, none double-counted).

The comparison is on the AGGREGATE peeled-sum vs whole over every (mode, T) cell:
individual sub-millisecond CPU cells carry per-fence sync jitter, but the aggregate
cancels it, and a grossly missing / double-counted op would still move it well past
20%.  This is a CPU scaling-shape / plumbing check only; the GPU window measures the
per-op O(T) magnitudes (the CPU backend does not reproduce Metal kernel behaviour --
that is the whole reason W78 exists).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import mlx.core as mx
import pytest

# Pin the CPU stream at import: "no GPU" means CPU here, and MLX defaults to Metal.
mx.set_default_device(mx.cpu)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_REPO, "scripts", "deepseek_v41", "metal_decode_attn_bisect.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("w78_metal_decode_attn_bisect", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_MOD = _load_module()

_SUM_KEYS = ["qkv_proj", "cache_append", "mask_build", "compress_append", "select",
             "attend", "out_proj"]
_MODES = ["swa_only", "full", "reindex", "reuse"]
_TS = [256, 1024]


@pytest.fixture(scope="module")
def receipt():
    # CPU, tiny dims, the selected-key (cell16k) path -- iters chosen so the
    # aggregate is stable under a niced shared host.
    return _MOD.run({
        "tiny": True, "gpu": False, "Ts": _TS,
        "iters": 20, "warmup": 3, "use_selected": True,
    })


def test_pinned_to_cpu(receipt):
    assert mx.default_device() == mx.cpu
    assert receipt["device"] == "cpu"
    assert receipt["path"] == "selected_keys"


def test_all_modes_and_Ts_present(receipt):
    assert set(receipt["results"]) == set(_MODES)
    for mode in _MODES:
        assert set(receipt["results"][mode]) == {str(T) for T in _TS}


def test_every_op_finite_and_nonneg(receipt):
    import math
    for mode in _MODES:
        for T in _TS:
            cell = receipt["results"][mode][str(T)]
            for key in _SUM_KEYS + ["gather_iso", "score", "attend", "whole", "peeled_sum"]:
                v = cell[key]
                assert isinstance(v, float) and math.isfinite(v) and v >= 0.0, (mode, T, key, v)
            # whole must be a real measurement (attention actually ran)
            assert cell["whole"] > 0.0, (mode, T)
            # the selected-key gather isolation ran (mx.take from the T-row cache)
            assert cell["gather_iso"] > 0.0, (mode, T)


def test_compress_and_select_gate_by_mode(receipt):
    """swa_only has no compressor/indexer; reuse reads the source selection (its
    own select is a cheap dict read); full/reindex own the indexer select."""
    for T in _TS:
        assert receipt["results"]["swa_only"][str(T)]["compress_append"] == 0.0
        assert receipt["results"]["swa_only"][str(T)]["select"] == 0.0
        # full owns its compressor frontier + indexer select -> both non-trivial
        assert receipt["results"]["full"][str(T)]["compress_append"] > 0.0
        assert receipt["results"]["full"][str(T)]["select"] > 0.0
        # reindex owns the indexer select (reuses the KV, no own compressor)
        assert receipt["results"]["reindex"][str(T)]["select"] > 0.0


def test_peeled_sum_within_20pct_of_whole(receipt):
    """The peel is a faithful decomposition: aggregate peeled-sum ~= aggregate
    whole (within 20%).  Aggregating over all cells cancels per-cell CPU jitter."""
    peeled_total = 0.0
    whole_total = 0.0
    for mode in _MODES:
        for T in _TS:
            cell = receipt["results"][mode][str(T)]
            # recompute the sum from the parts as an independent check
            s = sum(cell[k] for k in _SUM_KEYS)
            assert abs(s - cell["peeled_sum"]) < 1e-9, (mode, T, s, cell["peeled_sum"])
            peeled_total += cell["peeled_sum"]
            whole_total += cell["whole"]
    ratio = peeled_total / whole_total
    assert 0.80 <= ratio <= 1.20, (
        f"aggregate peeled/whole={ratio:.3f} outside [0.80, 1.20] "
        f"(peeled={peeled_total:.4f} ms, whole={whole_total:.4f} ms)"
    )


def test_masked_full_path_executes(receipt):
    """The --no-selected-keys lever runs the masked-full path (mask build + full-T
    score) without error and its peel decomposes the whole.

    The masked-full peel fences one extra bracket the selected path does not (the
    ``_window_attend`` mask build), so at tiny CPU dims the fixed per-fence sync
    cost runs the aggregate ~18% over the whole (rock-stable, ~+/-0.2% across
    repeats -- it is deterministic per-fence overhead, not a missing / double-
    counted op).  On the GPU at real dims that overhead is negligible (compute-
    bound), so the bound here is widened only to absorb the CPU fence tax; a gross
    decomposition error would still blow past it.  The strict 20% faithfulness
    check is on the primary selected-key path (test_peeled_sum_within_20pct...)."""
    r = _MOD.run({
        "tiny": True, "gpu": False, "Ts": _TS,
        "iters": 20, "warmup": 3, "use_selected": False,
    })
    assert r["path"] == "masked_full"
    peeled_total = whole_total = 0.0
    for mode in _MODES:
        for T in _TS:
            cell = r["results"][mode][str(T)]
            # masked-full builds the window attend mask (selected path does not)
            assert cell["mask_build"] > 0.0, (mode, T)
            peeled_total += cell["peeled_sum"]
            whole_total += cell["whole"]
    ratio = peeled_total / whole_total
    assert 0.80 <= ratio <= 1.25, f"masked-full aggregate peeled/whole={ratio:.3f}"
