"""W21-followup: DSpark MTP serve glue + planner pricing (CPU-only, no artifact).

Covers

1. the memory planner prices the wired ``mtp.*`` residents when MTP is served
   (spec ``mtp_included=True``) and is byte-identical to AR (mtp off);
2. the discount gates on ``mtp_included`` so the spec swap in the serve path is
   the only trigger needed;
3. ``resolve_with_mtp`` / ``is_deepseek_v41_mtp_config`` resolve ``--generation-mode
   mtp`` to with_mtp True without the ``MTPLX_DSV41_MTP`` env;
4. ``construct_resident_model`` threads ``with_mtp`` to the DeepSeek loader;
5. ``expert_cli`` lifts the AR-only rule only for the native-MTP artifact.

All CPU config work: no real artifact is loaded (measured manifest byte totals are
pinned constants; manifests are synthetic stand-ins).
"""

from __future__ import annotations

import dataclasses
import types

import pytest

from mtplx.expert_runtime import text_only_resident_discount
from mtplx.expert_streaming_models import get_model_spec, plan_expert_memory

GiB = 1024**3
MODEL_KEY = "deepseek-v41-flash-expert-mxfp4"

# Envelope (W21 profile) + measured mxfp4 manifest resident skips.
LIMIT = 82 * GiB
RESERVE = 7 * GiB
SWA_WINDOW = 40 * 128 * (512 * 2)
TRANSIENT = 48
DISCOUNT_AR = 8_920_505_736     # AR skips mtp (7,949,968,776) + vision (970,536,960)
DISCOUNT_MTP = 970_536_960      # MTP wires the mtp.* residents; only vision skipped
MTP_RESIDENT_BYTES = 7_949_968_776  # what MTP adds back (all mtp.* residents kept)


@dataclasses.dataclass(frozen=True)
class _T:
    tensor: str
    length: int


@dataclasses.dataclass(frozen=True)
class _Manifest:
    resident_tensors: tuple[_T, ...]


@dataclasses.dataclass(frozen=True)
class _Spec:
    mtp_included: bool = False


# --------------------------------------------------------------------------
# 1 + 2. planner prices the mtp.* residents when served with MTP
# --------------------------------------------------------------------------
def _plan(discount, ctx=16384):
    return plan_expert_memory(
        get_model_spec(MODEL_KEY),
        total_limit_bytes=LIMIT,
        context_tokens=ctx,
        runtime_reserve_bytes=RESERVE,
        additional_resident_bytes=SWA_WINDOW,
        resident_discount_bytes=discount,
        transient_slots=TRANSIENT,
        cache_scope="layer",
    )


def test_discount_gates_on_mtp_included() -> None:
    # The only serve-path trigger is spec.mtp_included: MTP-on keeps mtp.* (only
    # vision discounted), MTP-off discounts mtp.* + vision.
    manifest = _Manifest(
        (_T("mtp.0.ffn", 500), _T("vision.enc", 300), _T("model.layers.0", 9))
    )
    assert text_only_resident_discount(manifest, _Spec(mtp_included=False)) == 800
    assert text_only_resident_discount(manifest, _Spec(mtp_included=True)) == 300


def test_planner_prices_mtp_residents_and_ar_unchanged() -> None:
    ar = _plan(DISCOUNT_AR)
    mtp = _plan(DISCOUNT_MTP)
    # AR row is the W21 text-only envelope, unchanged.
    assert ar.slots_per_layer == 92
    # MTP prices all wired mtp.* residents (+7.95 GB): 10 fewer resident slots.
    assert mtp.resident_bytes - ar.resident_bytes == MTP_RESIDENT_BYTES
    assert mtp.slots_per_layer == 82
    assert ar.slots_per_layer - mtp.slots_per_layer == 10
    assert mtp.fits_fixed and ar.fits_fixed


def test_spec_swap_is_valid_and_drives_the_discount() -> None:
    # dataclasses.replace(spec, mtp_included=True) is the exact swap the serve
    # path applies; it is a valid spec and flips the discount to vision-only.
    spec = get_model_spec(MODEL_KEY)
    swapped = dataclasses.replace(spec, mtp_included=True)
    assert swapped.mtp_included is True
    assert swapped.resident_bytes == spec.resident_bytes  # same routed bank
    manifest = _Manifest((_T("mtp.a", 11), _T("vision.b", 22), _T("model.x", 33)))
    assert text_only_resident_discount(manifest, spec) == 33
    assert text_only_resident_discount(manifest, swapped) == 22


# --------------------------------------------------------------------------
# 3. --generation-mode mtp resolves with_mtp without the env
# --------------------------------------------------------------------------
def test_resolve_with_mtp_explicit_wins_over_env(monkeypatch) -> None:
    from mtplx.models.deepseek_v41_loader import resolve_with_mtp

    config = {"text_config": {"model_type": "deepseek_v41_text", "n_mtp_layers": 3}}
    manifest = _Manifest((_T("mtp.0.x", 10), _T("model.layers.0", 5)))
    monkeypatch.setenv("MTPLX_DSV41_MTP", "0")  # explicit wins over env
    assert resolve_with_mtp(config, manifest, True) is True
    assert resolve_with_mtp(config, manifest, False) is False


def test_resolve_with_mtp_defaults_off_without_env(monkeypatch) -> None:
    from mtplx.models.deepseek_v41_loader import resolve_with_mtp

    config = {"text_config": {"model_type": "deepseek_v41_text", "n_mtp_layers": 3}}
    manifest = _Manifest((_T("mtp.0.x", 10), _T("model.layers.0", 5)))
    monkeypatch.delenv("MTPLX_DSV41_MTP", raising=False)
    assert resolve_with_mtp(config, manifest, None) is False


def test_is_deepseek_v41_mtp_config_predicate() -> None:
    from mtplx.models.deepseek_v41 import is_deepseek_v41_mtp_config

    assert is_deepseek_v41_mtp_config(
        {"model_type": "deepseek_v41", "text_config": {"n_mtp_layers": 3}}
    )
    # no stages -> not native MTP
    assert not is_deepseek_v41_mtp_config({"model_type": "deepseek_v41"})
    # hy3/glm never match
    assert not is_deepseek_v41_mtp_config({"model_type": "hy_v3", "n_mtp_layers": 1})


# --------------------------------------------------------------------------
# 4. construct_resident_model threads with_mtp to the DeepSeek loader
# --------------------------------------------------------------------------
def test_construct_resident_model_threads_with_mtp(monkeypatch, tmp_path) -> None:
    import mtplx.resident_loader as rl
    import mtplx.models.deepseek_v41_loader as loader

    captured: dict[str, object] = {}

    def _fake_construct(root, runtime, *, config=None, mx_module=None,
                        switch_binder=None, strict=True, with_mtp=None):
        captured["with_mtp"] = with_mtp
        return "resident-model"

    monkeypatch.setattr(loader, "construct_deepseek_v41_resident_model", _fake_construct)
    out = rl.construct_resident_model(
        tmp_path, object(), config={"model_type": "deepseek_v41"}, with_mtp=True
    )
    assert out == "resident-model"
    assert captured["with_mtp"] is True


# --------------------------------------------------------------------------
# 5. expert_cli lifts the AR-only rule only for the native-MTP artifact
# --------------------------------------------------------------------------
def _args(**kw):
    ns = types.SimpleNamespace(
        _cli_flags={"generation-mode"}, generation_mode="mtp", load_mtp=True,
        expert_streaming_config=None, expert_manifest=None,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_expert_cli_ar_only_still_rejects_non_native_mtp(monkeypatch, tmp_path) -> None:
    import mtplx.expert_cli as ec

    monkeypatch.setattr(ec, "expert_streaming_requested", lambda a: True)
    monkeypatch.setattr(ec, "_authoritative_manifest_path", lambda r: tmp_path / "m.json")
    # hy3/glm streamed profile: not native MTP -> AR-only rule stands.
    monkeypatch.setattr(ec, "_is_native_streamed_mtp", lambda root, m: False)
    with pytest.raises(ValueError, match="AR-only"):
        ec.expert_streaming_load_kwargs(_args(), tmp_path)


def test_expert_cli_allows_generation_mode_mtp_for_native_head(monkeypatch, tmp_path) -> None:
    import mtplx.expert_cli as ec

    monkeypatch.setattr(ec, "expert_streaming_requested", lambda a: True)
    monkeypatch.setattr(ec, "_authoritative_manifest_path", lambda r: tmp_path / "m.json")
    monkeypatch.setattr(ec, "_is_native_streamed_mtp", lambda root, m: True)
    # Short-circuit the rest of the resolution once the native gate passes.
    sentinel = RuntimeError("reached admission with native MTP allowed")

    def _boom(root):
        raise sentinel

    monkeypatch.setattr(ec, "ensure_expert_admitted", _boom)
    with pytest.raises(RuntimeError) as excinfo:
        ec.expert_streaming_load_kwargs(_args(), tmp_path)
    assert excinfo.value is sentinel  # no AR-only ValueError was raised first


def test_expert_cli_ar_default_is_not_mtp(monkeypatch, tmp_path) -> None:
    # No mtp flags -> native gate never consulted, mtp stays off (AR).
    import mtplx.expert_cli as ec

    called = {"native": False}

    def _native(root, m):
        called["native"] = True
        return True

    monkeypatch.setattr(ec, "expert_streaming_requested", lambda a: True)
    monkeypatch.setattr(ec, "_authoritative_manifest_path", lambda r: tmp_path / "m.json")
    monkeypatch.setattr(ec, "_is_native_streamed_mtp", _native)
    monkeypatch.setattr(ec, "ensure_expert_admitted",
                        lambda root: (_ for _ in ()).throw(RuntimeError("stop")))
    ar_args = types.SimpleNamespace(_cli_flags=set(), generation_mode="ar",
                                    load_mtp=False, expert_streaming_config=None,
                                    expert_manifest=None)
    with pytest.raises(RuntimeError, match="stop"):
        ec.expert_streaming_load_kwargs(ar_args, tmp_path)
    assert called["native"] is False  # native gate not consulted when mtp not requested
