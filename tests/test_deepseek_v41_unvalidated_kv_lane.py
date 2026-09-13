"""CPU-only admission and A/B reporting regressions for unvalidated bounded KV.

The benchmark tests run its real CLI/summary with saved or synthetic receipts;
the model runner and the MLX module are inert doubles. No model is loaded.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

from mtplx import expert_runtime as runtime

ROOT = Path(__file__).resolve().parents[1]
KV_BOUNDED = "MTPLX_DSV41_KV_BOUNDED"
FUSED_PROJ = "MTPLX_DSV41_ATTN_FUSED_PROJ"


@pytest.fixture
def ab(monkeypatch):
    path = ROOT / "scripts/deepseek_v41/ab_decode_env_levers.py"
    spec = importlib.util.spec_from_file_location("_kv_parity_ab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # main imports MLX only to seed it before calling the replaced model runner.
    fake_core = types.ModuleType("mlx.core")
    fake_core.random = types.SimpleNamespace(seed=lambda seed: None)
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    monkeypatch.setattr(module, "_load_bench_module", lambda: None)
    monkeypatch.setattr(module, "_apply_cell_prompt_guard", lambda args: None)
    monkeypatch.setattr(module, "_write_output_sidecars", lambda *args: None)
    return module


def _receipt(ab, arm, sha, **env):
    return {
        "arm": arm,
        "arm_env": dict(ab.ARM_PRESETS[arm], **env),
        "rounding_class": ab._is_rounding_class(arm),
        "rounding_class_keys": ab._rounding_class_keys(arm),
        "token_ids_sha256": sha,
        "decode_tok_s": 4.0,
        "memory": {"plan_limit_gib_effective": 60.0},
    }


def _summary(ab, receipts, monkeypatch, tmp_path, capsys):
    pending = iter(receipts)
    monkeypatch.setattr(ab, "_run_arm", lambda *args: next(pending))
    rc = ab.main([
        "--arms", *(r["arm"] for r in receipts),
        "--context-tokens", "1024", "--decode-tokens", "1",
        "--out", str(tmp_path / "receipts.jsonl"),
    ])
    return rc, capsys.readouterr().out


def test_unchanged_rounding_lever_does_not_excuse_bounded_mismatch(
    ab, monkeypatch, tmp_path, capsys,
):
    base = _receipt(ab, "cell16k_ring_v2_attn", "control")
    candidate = _receipt(ab, "cell16k_ring_v2_attn_bounded", "candidate")
    rc, output = _summary(ab, [base, candidate], monkeypatch, tmp_path, capsys)
    assert "token-id sha differs -- expected" not in output
    assert "FAIL" in output
    assert rc != 0


def test_actual_env_overrides_preset_and_stale_rounding_labels(
    ab, monkeypatch, tmp_path, capsys,
):
    # Different arm names/preset labels, but the effective rounding lever is on
    # in BOTH runs. Only bounded KV changes in the recorded runtime environment.
    base = _receipt(ab, "control", "control", **{FUSED_PROJ: "1"})
    candidate = _receipt(ab, "attn_fused_proj", "candidate", **{KV_BOUNDED: "1"})
    rc, output = _summary(ab, [base, candidate], monkeypatch, tmp_path, capsys)
    assert "token-id sha differs -- expected" not in output
    assert "FAIL" in output
    assert rc != 0


def test_new_rounding_lever_does_not_validate_bounded_kv(
    ab, monkeypatch, tmp_path, capsys,
):
    base = _receipt(ab, "control", "control")
    candidate = _receipt(ab, "cell16k_ring_v2_attn_bounded", "candidate")
    rc, output = _summary(ab, [base, candidate], monkeypatch, tmp_path, capsys)
    assert "token-id sha differs -- expected" not in output
    assert "FAIL" in output
    assert rc != 0


@pytest.mark.parametrize("base_value,candidate_value", [(None, "1"), ("1", "0")])
def test_changed_rounding_lever_is_classified_in_both_directions(
    ab, monkeypatch, tmp_path, capsys, base_value, candidate_value,
):
    base = _receipt(ab, "control", "control", **{FUSED_PROJ: base_value})
    candidate = _receipt(ab, "control", "candidate", **{FUSED_PROJ: candidate_value})
    rc, output = _summary(ab, [base, candidate], monkeypatch, tmp_path, capsys)
    assert "token-id sha differs -- expected" in output
    assert FUSED_PROJ in output
    assert "FAIL" not in output
    assert rc == 0


def test_equivalent_boolean_spelling_is_not_a_changed_lever(
    ab, monkeypatch, tmp_path, capsys,
):
    base = _receipt(ab, "attn_fused_proj", "control", **{FUSED_PROJ: "1"})
    candidate = _receipt(ab, "attn_fused_proj", "candidate", **{FUSED_PROJ: " TRUE "})
    rc, output = _summary(ab, [base, candidate], monkeypatch, tmp_path, capsys)
    assert "token-id sha differs -- expected" not in output
    assert "FAIL" in output
    assert rc != 0


def test_dspark_margin_is_preserved_but_not_attributed_to_another_pair(
    ab, monkeypatch, tmp_path, capsys,
):
    base = _receipt(ab, "control", "control")
    candidate = _receipt(ab, "attn_fused_proj", "candidate")
    divergence = {"divergence_index": 33, "ar_top2_margin": 0.25,
                  "dspark_top2_margin": 0.0, "class": "tie_flip"}
    candidate["dspark"] = {"divergence": divergence}
    rc, output = _summary(ab, [base, candidate], monkeypatch, tmp_path, capsys)
    assert "token-id sha differs -- expected" in output
    assert "first divergence @" not in output
    assert "control top-2 logit margin" not in output
    saved = [json.loads(line) for line in (tmp_path / "receipts.jsonl").read_text().splitlines()]
    assert saved[1]["dspark"]["divergence"] == divergence
    assert rc == 0


def test_saved_windows_keep_capacity_confound_visible(
    ab, monkeypatch, tmp_path, capsys,
):
    receipt_dir = ROOT / "docs/deepseek-v41/receipts/gpu-windows"
    rows = [json.loads((receipt_dir / f"window-{w}/ar-v2-attn.json").read_text())
            for w in (47, 48)]
    assert rows[0]["prompt_ids_sha256"] == rows[1]["prompt_ids_sha256"]
    assert rows[0]["token_ids"][:33] == rows[1]["token_ids"][:33]
    assert (rows[0]["token_ids"][33], rows[1]["token_ids"][33]) == (832, 790)
    assert [r["resolved_plan"]["slots_per_layer"] for r in rows] == [66, 71]
    rc, output = _summary(ab, rows, monkeypatch, tmp_path, capsys)
    assert "equal budgets cannot be established" in output
    assert "token-id sha differs -- expected" not in output
    assert "FAIL" in output
    assert rc != 0


def test_resolved_plan_stamps_actual_budget(ab, monkeypatch):
    monkeypatch.setenv("MTPLX_DSV41_GATE_PREFETCH", "0")
    engine = types.SimpleNamespace(
        plan=types.SimpleNamespace(total_limit_bytes=49_934_398_112),
        config=types.SimpleNamespace(), spec=types.SimpleNamespace(),
    )
    report = ab._resolved_plan(engine, types.SimpleNamespace())
    assert report.get("memory_limit_bytes") == 49_934_398_112


@pytest.mark.parametrize("field", ["resolved_plan", "memory_cap"])
def test_unequal_modern_budgets_do_not_report_reproducible_plan(
    ab, monkeypatch, tmp_path, capsys, field,
):
    rows = [_receipt(ab, "control", "same") for _ in range(2)]
    key = "memory_limit_bytes" if field == "resolved_plan" else "engine_budget_bytes"
    for row, budget in zip(rows, (40_000_000_000, 50_000_000_000)):
        row["memory"] = {"schema_version": 2}
        row[field] = {key: budget}
    rc, output = _summary(ab, rows, monkeypatch, tmp_path, capsys)
    assert "DIFFERENT plan_limit" in output
    assert "40000000000" in output and "50000000000" in output
    assert "plan reproducibility: all arms" not in output
    assert rc == 0  # intentional budget comparisons remain usable


def test_unknown_plans_are_not_reported_as_equal(ab, monkeypatch, tmp_path, capsys):
    rows = [_receipt(ab, "control", "same") for _ in range(2)]
    for row in rows:
        row["memory"] = {"schema_version": 2}
    rc, output = _summary(ab, rows, monkeypatch, tmp_path, capsys)
    assert "unknown" in output.lower()
    assert "plan reproducibility: all arms" not in output
    assert "plan_limit=None" not in output
    assert rc == 0


def test_plan_precedence_and_legacy_gib_conversion(ab, monkeypatch, tmp_path, capsys):
    rows = [_receipt(ab, "control", "same") for _ in range(3)]
    rows[0]["resolved_plan"] = {"memory_limit_bytes": 60 * 1024**3}
    rows[0]["memory_cap"] = {"engine_budget_bytes": 1}
    rows[0]["memory"]["plan_limit_gib_effective"] = 1
    rows[1]["memory_cap"] = {"engine_budget_bytes": 60 * 1024**3}
    rows[1]["memory"]["plan_limit_gib_effective"] = 1
    rc, output = _summary(ab, rows, monkeypatch, tmp_path, capsys)
    assert "all arms ran plan_limit_bytes=64424509440" in output
    assert "DIFFERENT" not in output
    assert rc == 0


@pytest.mark.parametrize("mtp", [False, True])
def test_served_entry_rejects_before_native_mtp_cap_or_bank_creation(
    ab, monkeypatch, tmp_path, mtp,
):
    """Real public load wrapper/implementation, with only heavy imports stubbed."""
    def stub_module(name, **members):
        module = types.ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)

    def forbidden(*args, **kwargs):
        pytest.fail("cap or bank allocation reached before served lane rejection")

    stub_module("mtplx.mtp_adapters", **dict.fromkeys((
        "install_saved_mtp_lora_adapter", "merge_installed_mtp_lora_adapters",
        "mtp_adapter_depth",
    ), forbidden))
    stub_module("mtplx.a3b_whole_moe",
                validate_a3b_whole_moe_load_options=lambda **kwargs: None)
    stub_module("mtplx.models.expert_mlx",
                make_mlx_component_bank_allocator=forbidden,
                make_mlx_slot_buffer_allocator=forbidden)
    stub_module("mtplx.models.deepseek_v41",
                is_deepseek_v41_mtp_config=lambda config: True)
    spec = importlib.util.spec_from_file_location(
        "mtplx._kv_admission_runtime", ROOT / "mtplx/runtime.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_load_runtime_metadata", lambda path: {})
    monkeypatch.setattr(module, "_install_architectures_declared_module_alias", lambda config: False)
    monkeypatch.setattr(runtime, "apply_mlx_memory_cap", forbidden)
    for name in ("proj_quant_plan_discount", "proj_requant_plan_discount", "text_only_resident_discount"):
        monkeypatch.setattr(runtime, name, lambda *args, **kwargs: 0)
    from mtplx import expert_manifest
    monkeypatch.setattr(expert_manifest, "load_expert_manifest", lambda path: object())
    monkeypatch.setattr(runtime.ExpertStreamingConfig, "memory_plan",
                        lambda *args, **kwargs: types.SimpleNamespace(fits_fixed=True))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    config = runtime.ExpertStreamingConfig(
        model_key="deepseek-v41-flash-expert-mxfp4", memory_limit_bytes=1,
        max_live_kv_tokens=0, runtime_reserve_bytes=0,
    )
    monkeypatch.setenv(KV_BOUNDED, "1")
    with pytest.raises(runtime.ExpertStreamingConfigurationError,
                       match="unvalidated Metal parity"):
        module.load(tmp_path, mtp=mtp, expert_streaming_config=config,
                    expert_manifest=tmp_path / "expert-manifest.json")
    assert os.environ[KV_BOUNDED] == "1"


class _PastAdmission(Exception):
    pass


def _open_to_admission(monkeypatch, tmp_path, model_key):
    def reached(*args, **kwargs):
        raise _PastAdmission("optimization profile validation reached")

    def forbidden(*args, **kwargs):
        pytest.fail("manifest, allocator, or MLX cap reached before lane rejection")

    monkeypatch.setattr(runtime, "not_applicable_violations", reached)
    monkeypatch.setattr(runtime, "load_expert_manifest", forbidden)
    monkeypatch.setattr(runtime, "apply_mlx_memory_cap", forbidden)
    return runtime.ExpertStreamingRuntime.open(
        tmp_path, tmp_path / "unused-manifest.json",
        types.SimpleNamespace(model_key=model_key),
        spec=types.SimpleNamespace(key=model_key),
        buffer_allocator=forbidden, mx_module=types.SimpleNamespace(),
    )


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", " TRUE "])
@pytest.mark.parametrize("model_key", ["deepseek-v41-flash-expert-mxfp4", "deepseek-v41-flash-expert-q2"])
def test_bounded_kv_rejected_before_loading_or_allocating(
    monkeypatch, tmp_path, value, model_key,
):
    monkeypatch.setenv(KV_BOUNDED, value)
    with pytest.raises(runtime.ExpertStreamingConfigurationError,
                       match="unvalidated Metal parity"):
        _open_to_admission(monkeypatch, tmp_path, model_key)
    assert os.environ[KV_BOUNDED] == value


@pytest.mark.parametrize("value", [None, "", "0", "false", "off", "no"])
def test_disabled_bounded_lane_preserves_normal_admission(monkeypatch, tmp_path, value):
    if value is None:
        monkeypatch.delenv(KV_BOUNDED, raising=False)
    else:
        monkeypatch.setenv(KV_BOUNDED, value)
    with pytest.raises(_PastAdmission):
        _open_to_admission(monkeypatch, tmp_path, "deepseek-v41-flash-expert-mxfp4")


def test_deepseek_lane_env_does_not_reject_another_family(monkeypatch, tmp_path):
    monkeypatch.setenv(KV_BOUNDED, "1")
    with pytest.raises(_PastAdmission):
        _open_to_admission(monkeypatch, tmp_path, "hy3-test-model")
