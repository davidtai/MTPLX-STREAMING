"""W21: DeepSeek-V4.1-Flash mxfp4 serve-profile + planner pricing (CPU-only).

Pure configuration/planner/selfcheck work: no real artifact is loaded, no MLX
device is touched. Covers

1. the promoted profile is auto-selected by model key and resolves the full
   tuned serve config with no flags;
2. the memory planner at the 82 GiB envelope with text-only resident pricing
   yields 92 slots/layer (and +11 over full-resident pricing);
3. serve-config resolution for the mxfp4 spec yields the profile and leaves the
   hy3 promoted configs byte-identical;
4. kernel_selfcheck._expert_quant_signature handles the mxfp4 codec (returns
   None cleanly, never raises).
"""

from __future__ import annotations

import dataclasses

import pytest

from mtplx.expert_profiles import (
    build_expert_streaming_config,
    load_expert_profiles,
    select_expert_profile,
)
from mtplx.expert_runtime import (
    TEXT_ONLY_SKIP_PREFIXES,
    text_only_resident_discount,
)
from mtplx.expert_streaming_models import get_model_spec, plan_expert_memory
from mtplx.kernel_selfcheck import _expert_quant_signature
from mtplx.models.deepseek_v41_loader import (
    DEFAULT_ENGRAM_CACHE_BYTES,
    TEXT_ONLY_SKIP_PREFIXES as LOADER_SKIP_PREFIXES,
    resolve_engram_cache_bytes,
)

GiB = 1024**3
MODEL_KEY = "deepseek-v41-flash-expert-mxfp4"
PROFILE_NAME = "deepseek-v41-mxfp4-75"
BIG = 200 * 10**9  # installed/available RAM plenty for selection

# Memory-plan envelope (coordinator 2026-09-10: 82 GiB planner default on the
# 100 GB box). Text-only discount is the measured MTP + vision resident skip of
# the shipped mxfp4 artifact (expert-manifest.json: mtp 7,949,968,776 + vision
# 970,536,960 = 8,920,505,736); the SWA sliding window is a fixed 5 MiB reserve.
LIMIT = 82 * GiB
RESERVE = 7 * GiB
SWA_WINDOW = 40 * 128 * (512 * 2)          # 5,242,880
TEXT_ONLY_DISCOUNT = 8_920_505_736
TRANSIENT_SLOTS = 48


# --------------------------------------------------------------------------
# stand-ins (no real manifest / no artifact load)
# --------------------------------------------------------------------------
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
# 1. profile selected by model key + resolved config
# --------------------------------------------------------------------------
def test_profile_auto_selected_by_model_key() -> None:
    profile = select_expert_profile(
        "auto", model_key=MODEL_KEY, installed_ram_bytes=BIG, available_bytes=BIG
    )
    assert profile.name == PROFILE_NAME
    assert profile.model_key == MODEL_KEY
    assert profile.generation_mode == "ar"
    assert profile.process_ceiling_bytes == 82 * GiB
    assert profile.weight_envelope_bytes == 75 * GiB
    # envelope + reserve == ceiling (profile invariant)
    assert (
        profile.weight_envelope_bytes + profile.config["runtime_reserve_bytes"]
        == profile.process_ceiling_bytes
    )


def test_explicit_profile_selection_and_model_key_guard() -> None:
    profile = select_expert_profile(
        PROFILE_NAME, model_key=MODEL_KEY, installed_ram_bytes=BIG, available_bytes=BIG
    )
    assert profile.name == PROFILE_NAME
    # wrong model key for the named profile is rejected
    with pytest.raises(ValueError, match="requires model key"):
        select_expert_profile(
            PROFILE_NAME,
            model_key="hy3-expert-oq2e",
            installed_ram_bytes=BIG,
            available_bytes=BIG,
        )


def test_resolved_serve_config_no_flags() -> None:
    profile = select_expert_profile(
        "auto", model_key=MODEL_KEY, installed_ram_bytes=BIG, available_bytes=BIG
    )
    cfg = build_expert_streaming_config(profile)
    assert cfg.model_key == MODEL_KEY
    # W79 dropped the mxfp4-75 plan ceiling 82 -> 60 GiB (the process ceiling is
    # kept at 82 and explicit overrides are still clamped <= 82). Assert the config
    # carries the profile's CURRENT declared values, read from expert_profiles.json,
    # not a stale literal.
    assert cfg.memory_limit_bytes == profile.config["memory_limit_bytes"]
    assert cfg.runtime_reserve_bytes == profile.config["runtime_reserve_bytes"]
    # the process ceiling (RSS guard) is a SEPARATE knob from the plan ceiling and
    # stays 82 GiB.
    assert profile.process_ceiling_bytes == 82 * GiB
    assert cfg.max_live_kv_tokens == 16384
    # KV stays bf16 and minimal
    assert cfg.kv_quant is None
    # LRU/layer default (no W24 routing census on this branch yet)
    assert cfg.cache_policy == "lru"
    assert cfg.cache_scope == "layer"
    # mxfp4 is a non-affine bank: component-banks is the only dispatch that reads
    # it, and it is derived from the codec even with no explicit slot_layout.
    assert cfg.slot_layout == "component-banks"
    assert cfg.transient_slots == TRANSIENT_SLOTS
    assert cfg.bypass_page_cache is True          # F_NOCACHE on
    assert cfg.max_read_chunk_bytes == 8 * 1024 * 1024
    assert cfg.split_route_release == "deferred"
    assert cfg.prefetch_slots == 0
    assert cfg.streamed_codec == "none"
    assert cfg.verify_record_hashes is False
    assert cfg.verify_sidecar_hash_at_open is False
    assert cfg.prefer_sidecar is True
    # remainder ("derived") expert-cache policy: no explicit cache cap
    assert cfg.expert_cache_limit_bytes is None
    assert cfg.derived_expert_cache_policy is True
    # islands are off (mxfp4 cannot serve dense/mmap islands)
    assert cfg.island_layers == ()
    assert cfg.island_layer_count is None
    assert cfg.mmap_island_layers == ()
    # engram row cache 2 GiB via env in the profile child_env
    assert profile.child_env.get("MTPLX_ENGRAM_CACHE_LIMIT") == "256MiB"


def test_slot_layout_derives_from_codec_without_explicit_flag() -> None:
    # Even if the profile did not pin slot_layout, the mxfp4 codec forces
    # component-banks in ExpertStreamingConfig.__post_init__.
    from mtplx.expert_runtime import ExpertStreamingConfig

    cfg = ExpertStreamingConfig(
        model_key=MODEL_KEY, memory_limit_bytes=82 * GiB, max_live_kv_tokens=16384
    )
    assert cfg.slot_layout == "component-banks"


# --------------------------------------------------------------------------
# 2. planner numbers at 82 GiB with text-only pricing
# --------------------------------------------------------------------------
@pytest.fixture()
def spec():
    return get_model_spec(MODEL_KEY)


def _plan(spec, ctx, *, discount):
    return plan_expert_memory(
        spec,
        total_limit_bytes=LIMIT,
        context_tokens=ctx,
        runtime_reserve_bytes=RESERVE,
        additional_resident_bytes=SWA_WINDOW,
        resident_discount_bytes=discount,
        transient_slots=TRANSIENT_SLOTS,
        cache_scope="layer",
    )


@pytest.mark.parametrize("ctx", [0, 1024, 16384, 65536])
def test_planner_text_only_slots_per_layer(spec, ctx) -> None:
    plan = _plan(spec, ctx, discount=TEXT_ONLY_DISCOUNT)
    assert plan.fits_fixed
    assert plan.slots_per_layer == 92
    assert plan.persistent_slots == 92 * spec.routed_layer_count
    # resident priced is the text-only backbone plus the SWA window
    assert plan.resident_bytes == (spec.resident_bytes - TEXT_ONLY_DISCOUNT) + SWA_WINDOW
    # fixed footprint fits under the envelope
    assert plan.fixed_bytes <= LIMIT


def test_planner_kv_is_bf16_and_minimal(spec) -> None:
    plan16k = _plan(spec, 16384, discount=TEXT_ONLY_DISCOUNT)
    assert plan16k.kv_bytes == 16384 * spec.kv_bytes_per_token == 52_428_800


def test_text_only_pricing_gains_eleven_slots(spec) -> None:
    text_only = _plan(spec, 16384, discount=TEXT_ONLY_DISCOUNT)
    full = _plan(spec, 16384, discount=0)
    assert full.slots_per_layer == 81
    assert text_only.slots_per_layer - full.slots_per_layer == 11


# --------------------------------------------------------------------------
# 3. text-only discount hook (synthetic manifests; hy3/glm stay identical)
# --------------------------------------------------------------------------
def test_skip_prefixes_match_loader_filter() -> None:
    # The planner discount and the loader's text-only filter must use one set.
    assert TEXT_ONLY_SKIP_PREFIXES == LOADER_SKIP_PREFIXES


def test_discount_counts_mtp_and_vision_when_mtp_off() -> None:
    manifest = _Manifest(
        (
            _T("model.embed_tokens.weight", 100),
            _T("mtp.0.ffn.w", 500),
            _T("mtp.1.dense", 250),
            _T("vision.encoder", 300),
            _T("aligner.proj", 40),
            _T("image_newline", 8),
            _T("model.layers.0.self_attn.q_proj.weight", 999),
        )
    )
    assert text_only_resident_discount(manifest, _Spec(mtp_included=False)) == 1098


def test_discount_keeps_mtp_when_included() -> None:
    manifest = _Manifest(
        (_T("mtp.0.x", 500), _T("vision.enc", 300), _T("model.layers.1.mlp", 9))
    )
    # MTP wired -> only vision/aligner/image discounted
    assert text_only_resident_discount(manifest, _Spec(mtp_included=True)) == 300
    assert text_only_resident_discount(manifest, _Spec(mtp_included=False)) == 800


def test_discount_zero_for_hy3_like_manifest() -> None:
    # hy3/glm keep their MTP head external and have no vision residents, so the
    # discount is 0 and their resolved memory plans stay byte-identical.
    manifest = _Manifest(
        (
            _T("model.embed_tokens.weight", 100),
            _T("model.layers.0.self_attn.q_proj.weight", 200),
            _T("model.layers.0.mlp.gate_proj.weight", 300),
            _T("norm.weight", 8),
        )
    )
    assert text_only_resident_discount(manifest, _Spec(mtp_included=False)) == 0


def test_session_bank_knobs_no_longer_forced_off_for_mxfp4() -> None:
    # W26 (engram history rides the entry-0 cache state): restore_cache now
    # rewinds the engram with the KV, so in-memory near-prefix restore no longer
    # desyncs the layer-1/14 engram hash. The profile therefore stops forcing
    # MTPLX_SESSION_NEAR_PREFIX_RESTORE / MTPLX_SESSION_STORE_ON_PREFILL to 0
    # (they fall back to the engine default = on). The SSD prompt-cache cold
    # tier stays off separately (serve --ssd-session-cache off; None-KV lanes +
    # LayerAttentionCache registration are still unsupported).
    profiles = load_expert_profiles()
    child_env = dict(profiles[PROFILE_NAME].child_env)
    assert "MTPLX_SESSION_NEAR_PREFIX_RESTORE" not in child_env
    assert "MTPLX_SESSION_STORE_ON_PREFILL" not in child_env
    assert child_env.get("MTPLX_ENGRAM_CACHE_LIMIT") == "256MiB"


def test_session_bank_features_default_on_under_the_profile_env(monkeypatch) -> None:
    # With the kill switches removed from child_env, applying the profile env
    # leaves both features at the engine default (on).
    from mtplx.expert_cli import apply_expert_profile_child_env
    from mtplx.generation import (
        _near_prefix_restore_enabled,
        _store_on_prefill_env_enabled,
    )

    profiles = load_expert_profiles()

    class _Args:
        _resolved_expert_profile = profiles[PROFILE_NAME]

    monkeypatch.delenv("MTPLX_SESSION_NEAR_PREFIX_RESTORE", raising=False)
    monkeypatch.delenv("MTPLX_SESSION_STORE_ON_PREFILL", raising=False)
    environ: dict[str, str] = {}
    apply_expert_profile_child_env(_Args(), environ)
    assert "MTPLX_SESSION_NEAR_PREFIX_RESTORE" not in environ
    assert "MTPLX_SESSION_STORE_ON_PREFILL" not in environ
    for key, value in environ.items():
        monkeypatch.setenv(key, value)
    assert _near_prefix_restore_enabled() is True
    assert _store_on_prefill_env_enabled() is True


def test_head_last_row_prefill_enabled_for_mxfp4() -> None:
    # W29 / K19: the mxfp4 lane heads only the last prefill row. W20 chunking
    # bounds the attention/score transients but the lm head still built the
    # [1, s, vocab] logits (8.47 GB at 16,384 tokens) until this env turned the
    # runner's final-logits-only prefill on for the lane. Output is unchanged
    # (decode seeds from the last token only); score_prompt_logprobs stays all-rows.
    profiles = load_expert_profiles()
    child_env = dict(profiles[PROFILE_NAME].child_env)
    assert child_env.get("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS") == "0"


def test_head_last_row_gate_flips_when_child_env_applied(monkeypatch) -> None:
    # The child_env, applied to the serve daemon, actually turns the
    # last-row-head prefill on at its generation-path gate.
    from mtplx.expert_cli import apply_expert_profile_child_env
    from mtplx.generation import _env_falsey, _final_logits_prefill_enabled

    profiles = load_expert_profiles()

    class _Args:
        _resolved_expert_profile = profiles[PROFILE_NAME]

    # without the lane env the last-row lever is off (read live, not frozen)
    monkeypatch.delenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", raising=False)
    assert _env_falsey("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS") is False

    environ: dict[str, str] = {}
    apply_expert_profile_child_env(_Args(), environ)
    for key, value in environ.items():
        monkeypatch.setenv(key, value)
    assert _env_falsey("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS") is True
    assert _final_logits_prefill_enabled() is True


def test_hy3_profiles_do_not_gate_the_session_bank() -> None:
    # hy3/glm are untouched: they never set the session-bank kill switches, so
    # the engine's default (bank on) still applies to them.
    profiles = load_expert_profiles()
    for name in ("hy3-oq2e-64", "hy3-oq2e-88", "hy3-oq2e-96"):
        child_env = dict(profiles[name].child_env)
        assert "MTPLX_SESSION_NEAR_PREFIX_RESTORE" not in child_env
        assert "MTPLX_SESSION_STORE_ON_PREFILL" not in child_env


def test_expert_profile_choices_are_registry_derived() -> None:
    # No-flags serve forwards the auto-resolved profile name to the daemon
    # child, which re-parses it against these choices.
    from mtplx.expert_cli import expert_profile_choices

    choices = expert_profile_choices()
    assert "auto" in choices
    assert PROFILE_NAME in choices
    assert {"hy3-oq2e-64", "hy3-oq2e-88", "hy3-oq2e-96"} <= set(choices)


def test_hy3_promoted_configs_unchanged() -> None:
    # Adding the deepseek profile must not perturb the hy3 promoted profiles.
    profiles = load_expert_profiles()
    assert {"hy3-oq2e-64", "hy3-oq2e-88", "hy3-oq2e-96"} <= set(profiles)
    hy3_64 = build_expert_streaming_config(profiles["hy3-oq2e-64"])
    assert hy3_64.model_key == "hy3-expert-oq2e"
    assert hy3_64.memory_limit_bytes == 76235669504
    assert hy3_64.runtime_reserve_bytes == 7516192768
    assert hy3_64.expert_cache_limit_bytes == 53678702592
    assert hy3_64.cache_policy == "frequency"
    assert hy3_64.proj_requant == "q4"
    assert hy3_64.slot_layout == "component-banks"
    # all three hy3 profiles still build cleanly
    for name in ("hy3-oq2e-64", "hy3-oq2e-88", "hy3-oq2e-96"):
        assert build_expert_streaming_config(profiles[name]).model_key == "hy3-expert-oq2e"


# --------------------------------------------------------------------------
# 4. mxfp4 selfcheck signature
# --------------------------------------------------------------------------
def test_selfcheck_signature_none_for_mxfp4() -> None:
    spec = get_model_spec(MODEL_KEY)
    # never raises; the affine expert_gather lane cannot validate a non-affine
    # bank, so the signature is None (lane skipped).
    assert _expert_quant_signature(spec) is None


def test_selfcheck_signature_affine_bank_unchanged() -> None:
    import mlx.core as mx

    affine = get_model_spec("deepseek-v41-flash-expert-q2")
    assert _expert_quant_signature(affine) == (mx.bfloat16, 2, 64)
    assert _expert_quant_signature(None) is None


# --------------------------------------------------------------------------
# engram cache-limit plumbing (CPU-only)
# --------------------------------------------------------------------------
def test_engram_cache_default_is_two_gib(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_ENGRAM_CACHE_LIMIT", raising=False)
    assert DEFAULT_ENGRAM_CACHE_BYTES == 256 * 1024**2
    assert resolve_engram_cache_bytes() == 256 * 1024**2


def test_engram_cache_env_override(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_ENGRAM_CACHE_LIMIT", "3GiB")
    assert resolve_engram_cache_bytes() == 3 * GiB
