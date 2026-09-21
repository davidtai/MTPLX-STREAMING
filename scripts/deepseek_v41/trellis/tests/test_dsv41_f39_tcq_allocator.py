"""CPU test for the F39 sha-safe make_mlx_component_bank_allocator monkeypatch (tcq.loader_install): a tcq3 record is
accepted (returns a record-driven allocate callable), an mxfp4 record still goes through the ORIGINAL unchanged, and
a bad tcq3 record is rejected.  The monkeypatch rebinds the module function (no source edit -> passes run_full.py's
pinned-mtplx sha assert).  No serve, no model, no GPU (the allocate callable is returned without allocating a bank).
"""
import sys
import types
from pathlib import Path

import pytest

_DSV41 = Path(__file__).resolve().parents[2]
_ROOT = Path(__file__).resolve().parents[4]
for p in (str(_DSV41), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tcq import loader_install as L      # noqa: E402


def _seg(c, d, sh, ln):
    return types.SimpleNamespace(component=c, dtype=d, shape=sh, length=ln)


def _manifest(segs):
    return types.SimpleNamespace(records=[types.SimpleNamespace(layer=0, expert=0, segments=segs)])


def _spec(codec):
    return types.SimpleNamespace(expert_codec=codec, routed_layer_indices=(0,), hidden_size=5120,
                                 expert_hidden_size=2304, quant_bits=(3 if codec == "tcq3" else 4),
                                 quant_group_size=32, expert_count=384)


_PLAN = types.SimpleNamespace(cache_scope="layer")
_TCQ3 = [_seg("gate_proj.code", "I16", (320, 144, 48), 4423680), _seg("gate_proj.rout", "F16", (2304,), 4608),
         _seg("up_proj.code", "I16", (320, 144, 48), 4423680), _seg("up_proj.rout", "F16", (2304,), 4608),
         _seg("down_proj.code", "I16", (144, 320, 48), 4423680), _seg("down_proj.rout", "F16", (5120,), 10240)]
_MXFP4 = [_seg("gate_proj.weight", "U32", (2304, 640), 5898240), _seg("gate_proj.scales", "U8", (2304, 160), 368640),
          _seg("up_proj.weight", "U32", (2304, 640), 5898240), _seg("up_proj.scales", "U8", (2304, 160), 368640),
          _seg("down_proj.weight", "U32", (5120, 288), 5898240), _seg("down_proj.scales", "U8", (5120, 72), 368640)]


@pytest.fixture
def installed():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from mtplx.models import expert_mlx as em
    original = em.make_mlx_component_bank_allocator
    em._tcq3_allocator_installed = False
    L.install_component_bank_allocator_support()
    try:
        yield em, original
    finally:
        em.make_mlx_component_bank_allocator = original
        em._tcq3_allocator_installed = False


def test_allocator_monkeypatch_accepts_tcq3_returns_callable(installed):
    em, _ = installed
    alloc = em.make_mlx_component_bank_allocator(_PLAN, _spec("tcq3"), _manifest(_TCQ3))
    assert callable(alloc)
    for attr in ("banks", "slots", "close", "plan"):
        assert hasattr(alloc, attr)          # same allocator contract as the original


def test_allocator_monkeypatch_delegates_mxfp4_to_original(installed):
    em, _ = installed
    # a valid mxfp4 record must pass unchanged (the patched wrapper delegates non-tcq3 to the captured original,
    # which runs the stock mxfp4 signature validation and returns the record-driven allocate callable).
    alloc = em.make_mlx_component_bank_allocator(_PLAN, _spec("mxfp4"), _manifest(_MXFP4))
    assert callable(alloc)
    # and the original still REJECTS a genuinely malformed mxfp4 record (delegation preserves strictness)
    bad = list(_MXFP4)
    bad[0] = _seg("gate_proj.weight", "U32", (2304, 999), 5898240)
    with pytest.raises(ValueError):
        em.make_mlx_component_bank_allocator(_PLAN, _spec("mxfp4"), _manifest(bad))


def test_allocator_monkeypatch_rejects_bad_tcq3(installed):
    em, _ = installed
    with pytest.raises(ValueError):
        em.make_mlx_component_bank_allocator(_PLAN, _spec("tcq3"), _manifest(_MXFP4))  # mxfp4 segs vs tcq3 spec


def test_install_is_idempotent_and_round_trips():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from mtplx.models import expert_mlx as em
    original = em.make_mlx_component_bank_allocator
    em._tcq3_allocator_installed = False
    try:
        assert L.install_component_bank_allocator_support() is True
        assert L.install_component_bank_allocator_support() is False      # idempotent
        assert em.make_mlx_component_bank_allocator is not original       # rebound
    finally:
        em.make_mlx_component_bank_allocator = original
        em._tcq3_allocator_installed = False
    assert em.make_mlx_component_bank_allocator is original               # restored
