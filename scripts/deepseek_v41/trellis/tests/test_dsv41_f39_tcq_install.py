"""CPU tests for the F39 tcq3 install route (tcq/install.py) and the packed_phase stager (tcq/stage_tcq_runner.py).

No GPU / no runtime: validate_tcq_config runs against a duck-typed runtime; the stager round-trips on the REAL
retained packed_phase.py source.  MLX pinned to CPU by conftest.py.
"""
import os
import sys
import types
from pathlib import Path

import pytest

_TRELLIS = Path(__file__).resolve().parents[1]
_DSV41 = Path(__file__).resolve().parents[2]                 # scripts/deepseek_v41 -> `import tcq.install`
_ROOT = Path(__file__).resolve().parents[4]
for p in (str(_TRELLIS), str(_DSV41), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import tcq.install as ti                                     # noqa: E402
import tcq.stage_tcq_runner as stager                        # noqa: E402

RETAINED_PACKED_PHASE = str(_ROOT / "docs/deepseek-v41/receipts/extension-bank-20260919/full/sources/packed/packed_phase.py")


def _good_runtime():
    spec = types.SimpleNamespace(expert_codec="tcq3", quant_bits=3, expert_record_bytes=13_290_496,
                                 hidden_size=5120, expert_hidden_size=2304, top_k=6, swiglu_limit=10.0,
                                 routed_layer_indices={0, 1, 2})
    config = types.SimpleNamespace(cache_scope="layer", decode_miss_records_per_part=3, prefetch_slots=0,
                                   resource_telemetry=False, split_route_release="deferred")
    plan = types.SimpleNamespace(transient_slots=48)
    reader = types.SimpleNamespace(_fanout_executor=object())
    return types.SimpleNamespace(spec=spec, config=config, plan=plan, reader=reader,
                                 _pipeline_ledger=None, _single_slot_pool=True)


SWITCHES = {0: object(), 1: object(), 2: object()}
ROUTS = {0: {}, 1: {}, 2: {}}


def test_validate_accepts_good_tcq3_config():
    ti.validate_tcq_config(_good_runtime(), SWITCHES, ROUTS)      # must not raise


@pytest.mark.parametrize("field,value", [
    ("expert_codec", "mxfp4"),
    ("quant_bits", 4),
    ("expert_record_bytes", 17_694_720),
    ("hidden_size", 4096),
    ("top_k", 8),
    ("swiglu_limit", 7.0),
])
def test_validate_rejects_wrong_spec(field, value):
    rt = _good_runtime()
    setattr(rt.spec, field, value)
    with pytest.raises(RuntimeError, match="tcq3 lane requires"):
        ti.validate_tcq_config(rt, SWITCHES, ROUTS)


@pytest.mark.parametrize("field,value", [
    ("decode_miss_records_per_part", 2),
    ("prefetch_slots", 4),
    ("split_route_release", "immediate"),
])
def test_validate_rejects_wrong_config(field, value):
    rt = _good_runtime()
    setattr(rt.config, field, value)
    with pytest.raises(RuntimeError, match="tcq3 lane requires"):
        ti.validate_tcq_config(rt, SWITCHES, ROUTS)


def test_validate_rejects_switch_layer_mismatch():
    rt = _good_runtime()
    with pytest.raises(RuntimeError, match="tcq3 lane requires"):
        ti.validate_tcq_config(rt, {0: object(), 1: object()}, ROUTS)   # missing layer 2


def test_tcq3_enabled_reads_env(monkeypatch):
    monkeypatch.delenv(ti.ENV_FLAG, raising=False)
    assert ti.tcq3_enabled() is False
    monkeypatch.setenv(ti.ENV_FLAG, "1")
    assert ti.tcq3_enabled() is True
    monkeypatch.setenv(ti.ENV_FLAG, "0")
    assert ti.tcq3_enabled() is False


def test_install_from_env_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv(ti.ENV_FLAG, raising=False)
    # must return None WITHOUT importing plane_lane / touching the runtime (stock route)
    assert ti.install_from_env(object(), SWITCHES, ROUTS, tables=None) is None


def test_route_plane_lane_falls_through_to_mxfp4_when_disabled(monkeypatch):
    monkeypatch.delenv(ti.ENV_FLAG, raising=False)
    calls = {}

    def fake_mxfp4_install(rt, switches_by_layer, owners):
        calls["args"] = (rt, switches_by_layer, owners)
        return {"stock": True}

    rt = object()
    layers = (0, 1, 2)
    switches = (object(), object(), object())
    owners = {"scales": 1}
    out = ti.route_plane_lane(rt, layers, switches, owners, fake_mxfp4_install)
    assert out == {"stock": True}
    # mxfp4 gets exactly the retained call shape: dict(zip(layers, switches)) + owners
    assert calls["args"][1] == dict(zip(layers, switches)) and calls["args"][2] is owners


# ---------------------------------------------------------------- stager round-trip on the real source

def test_stager_round_trips_on_retained_packed_phase():
    if not os.path.exists(RETAINED_PACKED_PHASE):
        pytest.skip(f"retained packed_phase.py absent: {RETAINED_PACKED_PHASE}")
    source = Path(RETAINED_PACKED_PHASE).read_text()
    updated = stager.rewrite(source)
    assert updated != source
    assert "route_plane_lane" in updated
    assert "import tcq.install as _tcq_route" in updated
    # the stock mxfp4 call still present verbatim inside route_plane_lane's argument list
    assert "install_plane_lane" in updated
    # reversing the edit recovers the original byte-for-byte (rewrite() asserts this internally too)
    assert updated.replace(stager._ROUTED, stager._ANCHOR) == source


def test_stager_rejects_missing_anchor():
    with pytest.raises(RuntimeError, match="anchor changed"):
        stager.rewrite("def f():\n    return 1\n")
