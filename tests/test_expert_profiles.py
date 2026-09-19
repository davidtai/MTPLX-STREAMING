import json
import subprocess
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

import mtplx.expert_profiles as expert_profiles
from mtplx.expert_profiles import (
    build_expert_streaming_config,
    load_expert_profiles,
    select_expert_profile,
)


GiB = 1024**3


def _profile_resource_document():
    resource = expert_profiles.files("mtplx").joinpath(
        "data/expert_profiles.json"
    )
    return json.loads(resource.read_text(encoding="utf-8"))


def _parse_profile_document(document):
    return expert_profiles._parse_expert_profiles_resource(
        json.dumps(document)
    )


def test_only_promoted_oq2e_profiles_are_installed():
    # hy3 oQ2e promoted profiles plus the DeepSeek-V4.1-Flash mxfp4 serve
    # profile (W21), which auto-resolves the streamed config with no flags.
    assert set(load_expert_profiles()) == {
        "hy3-oq2e-64",
        "hy3-oq2e-88",
        "hy3-oq2e-96",
        "deepseek-v41-mxfp4-75",
    }


def test_loaded_profiles_are_immutable():
    profiles = load_expert_profiles()
    profile = profiles["hy3-oq2e-64"]

    with pytest.raises(TypeError):
        profiles["candidate"] = profile
    with pytest.raises(TypeError):
        profile.config["memory_limit_bytes"] = 1
    with pytest.raises(TypeError):
        profile.child_env["MTPLX_SUSTAINED_PREFILL"] = "0"
    with pytest.raises(FrozenInstanceError):
        profile.name = "candidate"


def test_auto_selects_largest_profile_that_passes_both_memory_gates():
    selected = select_expert_profile(
        "auto",
        model_key="hy3-expert-oq2e",
        installed_ram_bytes=128 * GiB,
        available_bytes=97 * GiB,
    )
    assert selected.name == "hy3-oq2e-88"


def test_auto_selection_observes_the_installed_ram_gate():
    selected = select_expert_profile(
        "auto",
        model_key="hy3-expert-oq2e",
        installed_ram_bytes=87 * GiB,
        available_bytes=128 * GiB,
    )
    assert selected.name == "hy3-oq2e-64"


def test_auto_selection_measures_memory_once_at_construction(monkeypatch):
    calls = {"installed": 0, "available": 0}

    def installed_ram_bytes():
        calls["installed"] += 1
        return 128 * GiB

    def available_memory_bytes():
        calls["available"] += 1
        return 97 * GiB

    monkeypatch.setattr(
        expert_profiles, "_installed_ram_bytes", installed_ram_bytes
    )
    monkeypatch.setattr(
        expert_profiles, "available_memory_bytes", available_memory_bytes
    )

    selected = select_expert_profile(
        "auto",
        model_key="hy3-expert-oq2e",
    )

    assert selected.name == "hy3-oq2e-88"
    assert calls == {"installed": 1, "available": 1}


def test_64_profile_installs_measured_cache_heavy_geometry(monkeypatch):
    profile = load_expert_profiles()["hy3-oq2e-64"]
    monkeypatch.setattr(
        expert_profiles,
        "resolve_island_placement",
        lambda *_args, **_kwargs: pytest.fail(
            "zero-island profile must not infer islands"
        ),
    )

    config = build_expert_streaming_config(profile)

    assert profile.process_ceiling_bytes == 71 * GiB
    assert profile.weight_envelope_bytes == 64 * GiB
    assert profile.generation_mode == "ar"
    assert profile.evidence_receipts == (
        "evals/tier2/t3_64x16k_armB_frequency.json",
        "research/envelope-admission-sweep-2026-07-22.json",
    )
    assert config.memory_limit_bytes == 71 * GiB
    assert config.runtime_reserve_bytes == 7 * GiB
    assert config.expert_cache_limit_bytes == 53_678_702_592
    assert config.max_live_kv_tokens == 4096
    assert config.island_layers == ()
    assert config.island_layer_count is None
    assert config.proj_requant == "q4"
    assert config.verify_record_hashes is False
    assert config.verify_sidecar_hash_at_open is False


def test_88_and_96_profiles_install_exact_island_counts():
    profiles = load_expert_profiles()

    config_88 = build_expert_streaming_config(profiles["hy3-oq2e-88"])
    assert profiles["hy3-oq2e-88"].process_ceiling_bytes == 95 * GiB
    assert profiles["hy3-oq2e-88"].weight_envelope_bytes == 88 * GiB
    assert config_88.island_layer_count == 74
    assert len(config_88.island_layers) == 74
    assert config_88.expert_cache_limit_bytes == 2 * GiB
    assert config_88.split_route_release == "deferred"
    assert config_88.deferred_pin_release is True

    config_96 = build_expert_streaming_config(profiles["hy3-oq2e-96"])
    assert profiles["hy3-oq2e-96"].process_ceiling_bytes == 103 * GiB
    assert profiles["hy3-oq2e-96"].weight_envelope_bytes == 96 * GiB
    assert config_96.island_layer_count == 79
    assert len(config_96.island_layers) == 79
    assert config_96.expert_cache_limit_bytes == 2 * GiB
    assert config_96.split_route_release == "fenced"
    assert config_96.deferred_pin_release is False


def test_deepseek_v41_profile_defaults_to_a_lean_60gib_plan():
    # W79: the served default engine plan drops to 60 GiB -- window 30 measured the
    # 16K cell as fast at a 60 GiB plan as at 80 (2.24 vs 2.21 tok/s) at a far lower
    # peak (67.5 vs 87.8 GB) -- while the admission ceiling and --expert-memory-limit
    # override cap stay at the 82 GiB envelope (75 GiB weights + 7 GiB reserve).
    profile = load_expert_profiles()["deepseek-v41-mxfp4-75"]
    assert profile.process_ceiling_bytes == 82 * GiB
    assert profile.weight_envelope_bytes == 75 * GiB
    assert profile.config["runtime_reserve_bytes"] == 7 * GiB
    # envelope + reserve == ceiling still holds; the default plan sits BELOW it.
    assert (
        profile.weight_envelope_bytes + profile.config["runtime_reserve_bytes"]
        == profile.process_ceiling_bytes
    )
    assert profile.config["memory_limit_bytes"] == 60 * GiB

    config = build_expert_streaming_config(profile)
    assert config.memory_limit_bytes == 60 * GiB

    # --expert-memory-limit still overrides UP TO the 82 GiB ceiling (the 80 GiB
    # A/B arm window 30 ran) ...
    raised = build_expert_streaming_config(
        profile, overrides={"memory_limit_bytes": "80GiB"}
    )
    assert raised.memory_limit_bytes == 80 * GiB
    # ... but never beyond it.
    with pytest.raises(ValueError, match="memory_limit_bytes"):
        build_expert_streaming_config(
            profile, overrides={"memory_limit_bytes": "90GiB"}
        )


def test_profile_overrides_normalize_memory_values():
    profile = load_expert_profiles()["hy3-oq2e-64"]

    config = build_expert_streaming_config(
        profile,
        overrides={
            "memory_limit_bytes": "70GiB",
            "expert_cache_limit_bytes": "50GiB",
        },
    )

    assert config.memory_limit_bytes == 70 * GiB
    assert config.expert_cache_limit_bytes == 50 * GiB


def test_profile_override_cannot_exceed_admitted_process_ceiling():
    profile = load_expert_profiles()["hy3-oq2e-64"]

    with pytest.raises(ValueError) as excinfo:
        build_expert_streaming_config(
            profile,
            overrides={"memory_limit_bytes": "192GiB"},
        )

    message = str(excinfo.value)
    assert "memory_limit_bytes" in message
    assert str(192 * GiB) in message
    assert str(profile.process_ceiling_bytes) in message


def test_profile_override_cannot_replace_model_identity():
    profile = load_expert_profiles()["hy3-oq2e-64"]

    with pytest.raises(ValueError, match="model_key"):
        build_expert_streaming_config(
            profile,
            overrides={"model_key": "hy3-expert-q2"},
        )


def test_profile_resource_rejects_duplicate_json_keys():
    resource = """\
{"schema": 1, "schema": 1, "profiles": []}
"""

    with pytest.raises(ValueError, match="duplicate JSON key 'schema'"):
        expert_profiles._parse_expert_profiles_resource(resource)


def test_profile_resource_rejects_config_ceiling_above_process_ceiling():
    # W79: memory_limit_bytes is the DEFAULT engine plan and may sit AT or BELOW
    # process_ceiling_bytes (the admission RAM + the --expert-memory-limit override
    # cap).  Only a default ABOVE the ceiling is incoherent (it could not be
    # admitted), so that alone is rejected.
    document = _profile_resource_document()
    row = document["profiles"][0]
    row["config"]["memory_limit_bytes"] += 1

    with pytest.raises(ValueError, match="config.memory_limit_bytes"):
        _parse_profile_document(document)


def test_profile_resource_accepts_default_plan_below_ceiling():
    # A default plan strictly below the ceiling is valid (W79): the profile ships a
    # proven-lean plan while leaving operators headroom to A/B a larger cap.
    document = _profile_resource_document()
    row = document["profiles"][0]
    row["config"]["memory_limit_bytes"] -= 1

    profiles = _parse_profile_document(document)
    parsed = profiles[row["name"]]
    assert parsed.config["memory_limit_bytes"] == row["config"]["memory_limit_bytes"]
    assert parsed.config["memory_limit_bytes"] < parsed.process_ceiling_bytes


def test_profile_resource_rejects_weight_and_reserve_mismatch():
    document = _profile_resource_document()
    row = document["profiles"][0]
    row["config"]["runtime_reserve_bytes"] -= 1

    with pytest.raises(ValueError, match="weight_envelope_bytes"):
        _parse_profile_document(document)


def test_profile_resource_rejects_config_identity_shadowing():
    document = _profile_resource_document()
    row = document["profiles"][0]
    row["config"]["model_key"] = "hy3-expert-q2"

    with pytest.raises(ValueError, match="config.*model_key"):
        _parse_profile_document(document)


def test_model_key_mismatch_fails_before_selection():
    with pytest.raises(ValueError) as excinfo:
        select_expert_profile(
            "hy3-oq2e-64",
            model_key="hy3-expert-q2",
            installed_ram_bytes=128 * GiB,
            available_bytes=128 * GiB,
        )

    message = str(excinfo.value)
    assert "hy3-oq2e-64" in message
    assert "hy3-expert-q2" in message
    assert "hy3-expert-oq2e" in message


def test_explicit_profile_fails_instead_of_downgrading():
    available = 70 * GiB

    with pytest.raises(ValueError) as excinfo:
        select_expert_profile(
            "hy3-oq2e-88",
            model_key="hy3-expert-oq2e",
            installed_ram_bytes=128 * GiB,
            available_bytes=available,
        )

    message = str(excinfo.value)
    assert str(95 * GiB) in message
    assert str(available) in message
    assert "required" in message
    assert "available" in message


def test_auto_fails_when_no_promoted_profile_fits():
    available = 70 * GiB

    with pytest.raises(ValueError) as excinfo:
        select_expert_profile(
            "auto",
            model_key="hy3-expert-oq2e",
            installed_ram_bytes=128 * GiB,
            available_bytes=available,
        )

    message = str(excinfo.value)
    assert str(71 * GiB) in message
    assert str(available) in message
    assert "hy3-oq2e-64" in message
    assert "hy3-oq2e-88" in message
    assert "hy3-oq2e-96" in message


def test_available_memory_counts_reclaimable_vm_stat_pages(monkeypatch):
    output = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  10.
Pages active:                                20.
Pages inactive:                              30.
Pages speculative:                           40.
Pages throttled:                              0.
Pages wired down:                            60.
Pages purgeable:                             50.
"""
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(expert_profiles.subprocess, "run", run)

    assert expert_profiles.available_memory_bytes() == 130 * 16384
    assert calls == [
        (
            ["/usr/bin/vm_stat"],
            {
                "check": True,
                "capture_output": True,
                "text": True,
                "timeout": 2.0,
            },
        )
    ]


def test_available_memory_rejects_missing_page_size(monkeypatch):
    output = """\
Mach Virtual Memory Statistics:
Pages free: 10.
Pages inactive: 30.
Pages speculative: 40.
Pages purgeable: 50.
"""
    monkeypatch.setattr(
        expert_profiles.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=output),
    )

    with pytest.raises(RuntimeError, match="page size"):
        expert_profiles.available_memory_bytes()


def test_available_memory_rejects_zero_page_size(monkeypatch):
    output = """\
Mach Virtual Memory Statistics: (page size of 0 bytes)
Pages free: 10.
Pages inactive: 30.
Pages speculative: 40.
Pages purgeable: 50.
"""
    monkeypatch.setattr(
        expert_profiles.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=output),
    )

    with pytest.raises(RuntimeError, match="page size.*positive"):
        expert_profiles.available_memory_bytes()


def test_available_memory_rejects_missing_required_counter(monkeypatch):
    output = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 10.
Pages inactive: 30.
Pages speculative: 40.
"""
    monkeypatch.setattr(
        expert_profiles.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=output),
    )

    with pytest.raises(RuntimeError, match="Pages purgeable"):
        expert_profiles.available_memory_bytes()


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.TimeoutExpired(["/usr/bin/vm_stat"], timeout=2.0),
        subprocess.CalledProcessError(1, ["/usr/bin/vm_stat"]),
    ],
    ids=["timeout", "nonzero-exit"],
)
def test_available_memory_reports_subprocess_failure(monkeypatch, failure):
    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(expert_profiles.subprocess, "run", fail)

    with pytest.raises(RuntimeError, match="vm_stat preflight failed"):
        expert_profiles.available_memory_bytes()
