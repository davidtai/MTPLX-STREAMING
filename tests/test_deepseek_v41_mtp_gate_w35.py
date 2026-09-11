"""W35: DSpark MTP serve-gate + AR-profile-engagement tests (CPU-only, no artifact).

Covers the serve-path AR-only gates that blocked `--generation-mode mtp` for the
DeepSeek-V4.1 DSpark artifact, and the promoted-profile knobs that shape the AR
lane (context = max_live_kv_tokens, session bank yields to the expert cache).
"""

from __future__ import annotations

import types

import pytest

MODEL_KEY = "deepseek-v41-flash-expert-mxfp4"
PROFILE_NAME = "deepseek-v41-mxfp4-75"
BIG = 200 * 10**9


def _mtp_args(**kw):
    ns = types.SimpleNamespace(
        _cli_flags={"generation-mode"}, generation_mode="mtp", load_mtp=True
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


# --------------------------------------------------------------------------
# the SECOND AR-only gate (commands/public.py) that fired at daemon startup
# --------------------------------------------------------------------------
def test_public_streamed_gen_mode_allows_native_mtp(monkeypatch, tmp_path) -> None:
    from mtplx.commands import public as pub
    import mtplx.expert_cli as ec

    monkeypatch.setattr(ec, "_authoritative_manifest_path", lambda r: tmp_path / "m.json")
    monkeypatch.setattr(ec, "_is_native_streamed_mtp", lambda root, m: True)
    # native MTP artifact + --generation-mode mtp -> no error (allowed)
    assert pub._streamed_generation_mode_error(_mtp_args(), tmp_path) is None


def test_public_streamed_gen_mode_rejects_non_native_mtp(monkeypatch, tmp_path) -> None:
    from mtplx.commands import public as pub
    import mtplx.expert_cli as ec

    monkeypatch.setattr(ec, "_authoritative_manifest_path", lambda r: tmp_path / "m.json")
    monkeypatch.setattr(ec, "_is_native_streamed_mtp", lambda root, m: False)
    err = pub._streamed_generation_mode_error(_mtp_args(), tmp_path)
    assert err == "promoted streamed profiles are AR-only in MTPLX 2.3.1rc1"


def test_public_streamed_gen_mode_ar_is_allowed(monkeypatch, tmp_path) -> None:
    from mtplx.commands import public as pub
    import mtplx.expert_cli as ec

    consulted = {"native": False}

    def _native(root, m):
        consulted["native"] = True
        return True

    monkeypatch.setattr(ec, "_authoritative_manifest_path", lambda r: tmp_path / "m.json")
    monkeypatch.setattr(ec, "_is_native_streamed_mtp", _native)
    ar_args = types.SimpleNamespace(_cli_flags=set(), generation_mode="ar", load_mtp=False)
    assert pub._streamed_generation_mode_error(ar_args, tmp_path) is None
    assert consulted["native"] is False  # no mtp requested -> native gate not consulted


# --------------------------------------------------------------------------
# native-MTP predicate (shared by both AR-only gates)
# --------------------------------------------------------------------------
def test_is_native_streamed_mtp_predicate(monkeypatch, tmp_path) -> None:
    import mtplx.expert_cli as ec

    class _T:
        def __init__(self, name):
            self.tensor = name

    class _Man:
        def __init__(self, names):
            self.resident_tensors = [_T(n) for n in names]

    def _cfg(cfg):
        monkeypatch.setattr("mlx_lm.utils.load_config", lambda root: cfg)

    # deepseek_v41 + MTP stages + mtp.* residents -> native
    _cfg({"model_type": "deepseek_v41", "text_config": {"n_mtp_layers": 3}})
    monkeypatch.setattr(ec, "load_expert_manifest", lambda p: _Man(["mtp.0.x", "model.y"]),
                        raising=False)
    monkeypatch.setattr("mtplx.expert_manifest.load_expert_manifest",
                        lambda p: _Man(["mtp.0.x", "model.y"]))
    assert ec._is_native_streamed_mtp(tmp_path, tmp_path / "m.json") is True

    # deepseek_v41 but no mtp.* residents -> not native
    monkeypatch.setattr("mtplx.expert_manifest.load_expert_manifest",
                        lambda p: _Man(["model.y"]))
    assert ec._is_native_streamed_mtp(tmp_path, tmp_path / "m.json") is False

    # hy3 config -> not native (external MTP)
    _cfg({"model_type": "hy_v3", "n_mtp_layers": 1})
    monkeypatch.setattr("mtplx.expert_manifest.load_expert_manifest",
                        lambda p: _Man(["mtp.0.x"]))
    assert ec._is_native_streamed_mtp(tmp_path, tmp_path / "m.json") is False


# --------------------------------------------------------------------------
# promoted profile shapes the AR lane (issues #2/#3)
# --------------------------------------------------------------------------
def test_profile_auto_resolves_for_model_key() -> None:
    from mtplx.expert_profiles import select_expert_profile

    p = select_expert_profile("auto", model_key=MODEL_KEY,
                              installed_ram_bytes=BIG, available_bytes=BIG)
    assert p.name == PROFILE_NAME


def test_profile_context_is_16384_and_session_bank_yields() -> None:
    from mtplx.expert_profiles import build_expert_streaming_config, load_expert_profiles

    profile = load_expert_profiles()[PROFILE_NAME]
    cfg = build_expert_streaming_config(profile)
    # served context for this lane == the profile KV plan (the daemon caps
    # args.context_window to this so it is not the machine-bound 696K window)
    assert cfg.max_live_kv_tokens == 16384
    child_env = dict(profile.child_env)
    # session bank yields to the expert cache (48G auto-size -> 2 GiB cap)
    assert child_env.get("MTPLX_SESSION_BANK_MAX_BYTES") == "2GiB"
    assert child_env.get("MTPLX_ENGRAM_CACHE_LIMIT") == "2GiB"


def test_ar_profile_does_not_request_mtp() -> None:
    # The promoted profile is AR by construction (generation_mode 'ar'); the MTP
    # residents are a byte regression (W24 census), so the AR lane is default and
    # must not force MTP on.
    from mtplx.expert_profiles import load_expert_profiles

    assert load_expert_profiles()[PROFILE_NAME].generation_mode == "ar"
