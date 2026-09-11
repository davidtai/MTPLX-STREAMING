"""Served-order: the streamed slot-layout default derives from the record codec.

A native-mxfp4 (or shadow / mixed-official) artifact can only be read by the
component-banks dispatch; the direct-slot / mapped / dense-island dispatches assume
the affine triple and would misread the record.  So when the user set no explicit
``--expert-slot-layout``, an mxfp4 spec must default to component-banks in the one
place both the loader and the serve path pass through (``ExpertStreamingConfig``),
while affine specs keep the historical direct-slots default.  An explicit
direct-slots is honoured verbatim and rejected loudly at ``open()``.

CPU only, no model load, trivial RSS (the guard fires before any bank/manifest read).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.cpu)

import pytest

from mtplx.expert_runtime import (
    ExpertStreamingConfig,
    ExpertStreamingConfigurationError,
    ExpertStreamingRuntime,
)
from mtplx.expert_streaming_models import get_model_spec

MXFP4 = "deepseek-v41-flash-expert-mxfp4"


def _cfg(key, **kw):
    return ExpertStreamingConfig(
        model_key=key, memory_limit_bytes=8 * 1024**3, max_live_kv_tokens=1024, **kw
    )


def test_config_default_slot_layout_derives_from_codec():
    # mxfp4 (and any non-affine codec) -> component-banks when unset.
    assert _cfg(MXFP4).slot_layout == "component-banks"
    assert get_model_spec(MXFP4).expert_codec == "mxfp4"
    # affine specs keep the historical direct-slots default (unchanged).
    assert _cfg("deepseek-v41-flash-expert-q2").slot_layout == "direct-slots"
    assert _cfg("hy3-q4").slot_layout == "direct-slots"
    # shadow-codec spec also needs component-banks (consistent with the guard).
    assert _cfg("glm52-expert-q1t").slot_layout == "component-banks"
    # an unregistered synthetic key keeps the affine default (spec carried into open()).
    assert _cfg("totally-made-up-key").slot_layout == "direct-slots"


def test_explicit_slot_layout_is_honoured_verbatim():
    assert _cfg(MXFP4, slot_layout="direct-slots").slot_layout == "direct-slots"
    assert _cfg(MXFP4, slot_layout="component-banks").slot_layout == "component-banks"
    assert (
        _cfg("hy3-q4", slot_layout="component-banks").slot_layout == "component-banks"
    )


def test_serve_open_selects_component_banks_for_mxfp4(tmp_path: Path):
    """The real serve-path open() for the mxfp4 spec (default config) clears the
    codec/slot-layout guard — it fails LATER on the (empty) manifest, proving the
    default resolved to component-banks rather than the direct-slots that would
    have tripped the guard."""
    spec = get_model_spec(MXFP4)
    cfg = _cfg(spec.key)  # no explicit slot_layout -> component-banks
    assert cfg.slot_layout == "component-banks"
    with pytest.raises(Exception) as excinfo:
        ExpertStreamingRuntime.open(
            tmp_path, tmp_path / "expert-manifest.json", cfg, spec=spec,
            apply_memory_cap=False,
        )
    assert "component-banks slot layout" not in str(excinfo.value)


def test_serve_open_rejects_forced_direct_slots_on_mxfp4(tmp_path: Path):
    """Forcing direct-slots on a mxfp4 artifact is still rejected loudly at open()."""
    spec = get_model_spec(MXFP4)
    cfg = _cfg(spec.key, slot_layout="direct-slots")
    assert cfg.slot_layout == "direct-slots"
    with pytest.raises(ExpertStreamingConfigurationError, match="component-banks slot layout"):
        ExpertStreamingRuntime.open(
            tmp_path, tmp_path / "expert-manifest.json", cfg, spec=spec,
            apply_memory_cap=False,
        )


def test_affine_open_defaults_direct_slots(tmp_path: Path):
    """An affine spec's default config stays direct-slots (byte-identical to before)."""
    spec = get_model_spec("deepseek-v41-flash-expert-q2")
    cfg = _cfg(spec.key)
    assert cfg.slot_layout == "direct-slots"
