"""MTP selection must own residency before allocating the expert banks (no MLX)."""
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mtplx.expert_streaming_models import get_model_spec
from mtplx.models import deepseek_v41_loader as loader


@pytest.mark.parametrize("explicit,env,included", [
    (True, "0", True), (None, "1", True), (False, "1", False),
    (None, "0", False),
])
def test_mtp_is_priced_before_allocation(monkeypatch, tmp_path, explicit, env, included):
    config = {"model_type": "deepseek_v41", "text_config": {"n_mtp_layers": 3}}
    manifest = SimpleNamespace(model_key="deepseek-v41-flash-expert-mxfp4",
                               resident_tensors=[SimpleNamespace(tensor="mtp.0.ffn")])
    opened, built = {}, {}
    monkeypatch.setenv("MTPLX_DSV41_MTP", env)
    monkeypatch.setattr(loader, "load_expert_manifest", lambda path: manifest)
    monkeypatch.setattr(loader, "resolve_artifact_member", lambda root, name: root / name)
    monkeypatch.setattr(loader, "resolve_gate_prefetch_ring_slots", lambda current: current)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", SimpleNamespace(load_config=lambda root: config))
    monkeypatch.setitem(sys.modules, "mtplx.models.deepseek_v41",
                        SimpleNamespace(_resolve_wo_a_cache=lambda: True))

    def open_runtime(root, path, runtime_config, **kwargs):
        opened.update(kwargs)
        return SimpleNamespace(spec=kwargs["spec"], manifest=manifest, close=lambda: None)

    def build(root, runtime, **kwargs):
        built.update(kwargs)
        return runtime

    monkeypatch.setattr(loader.ExpertStreamingRuntime, "open", open_runtime)
    monkeypatch.setattr(loader, "construct_deepseek_v41_resident_model", build)
    # Deliberately supply the opposite mode: explicit resolution must own it.
    spec = replace(get_model_spec(manifest.model_key), mtp_included=not included)
    loader.load_deepseek_v41_streaming(
        tmp_path, memory_limit_bytes=80 * 1024**3, max_live_kv_tokens=16384,
        admit=False, apply_memory_cap=False, spec=spec, with_mtp=explicit,
        slot_layout="direct-slots",
    )
    assert opened["spec"].mtp_included is included
    expected = loader.SWA_WINDOW_BYTES + 40 * loader.WO_A_DENSE_F32_BYTES
    if included:
        expected += 3 * (loader.WO_A_DENSE_F32_BYTES + 128 * 512 * 4)
    assert opened["additional_resident_bytes"] == expected
    assert built["with_mtp"] is included
    assert built["config"] == config


def test_separate_construction_refuses_an_unpriced_mtp_head(tmp_path):
    manifest = SimpleNamespace(resident_tensors=[SimpleNamespace(tensor="mtp.0.ffn")])
    runtime = SimpleNamespace(manifest=manifest, spec=SimpleNamespace(mtp_included=False))

    def no_model_import():
        pytest.fail("unpriced MTP reached model construction")

    with pytest.raises(loader.ResidentLoadError, match="MTP.*memory plan"):
        loader.construct_deepseek_v41_resident_model(
            tmp_path, runtime, with_mtp=True,
            config={"model_type": "deepseek_v41", "n_mtp_layers": 3},
            model_class_resolver=no_model_import,
        )
