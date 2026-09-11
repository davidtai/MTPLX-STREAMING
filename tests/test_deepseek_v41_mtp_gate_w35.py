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


# --------------------------------------------------------------------------
# window-12 follow-up: the parent AR-forcing, the session-bank plan, the cap
# --------------------------------------------------------------------------
import os  # noqa: E402
from pathlib import Path  # noqa: E402

_MXFP4 = Path(os.path.expanduser(
    "~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4"
))


@pytest.mark.skipif(not _MXFP4.exists(), reason="mxfp4 artifact not on this box")
def test_real_artifact_generation_mode_resolves_to_mtp() -> None:
    # Builds the serve decision from the real artifact's metadata (config.json +
    # expert-manifest.json; no experts.bin) + the argv. The parent AR-forcing
    # block that reported generation_mode='ar' in window 12 is now gated on this.
    from mtplx.commands import public as pub

    mtp_args = types.SimpleNamespace(
        _cli_flags={"generation-mode"}, generation_mode="mtp", load_mtp=True
    )
    assert pub._streamed_native_mtp_requested(mtp_args, _MXFP4) is True
    # the gated forcing block keeps MTP (mirrors public.py):
    if pub._streamed_native_mtp_requested(mtp_args, _MXFP4):
        mtp_args.generation_mode = pub.GENERATION_MODE_MTP
        mtp_args.load_mtp = True
        mtp_args.no_mtp = False
    else:  # pragma: no cover
        mtp_args.generation_mode = pub.GENERATION_MODE_AR
    assert mtp_args.generation_mode == "mtp"
    # AR (no flag) on the same artifact stays AR
    ar_args = types.SimpleNamespace(
        _cli_flags=set(), generation_mode="ar", load_mtp=False
    )
    assert pub._streamed_native_mtp_requested(ar_args, _MXFP4) is False


def test_memory_plan_bank_yields_to_explicit_cap() -> None:
    # The profile child_env's MTPLX_SESSION_BANK_MAX_BYTES=2GiB now governs the
    # advertised plan bank (was the module's 48G cap), matching the engine bank.
    from mtplx.memory_plan import plan_memory

    GiB = 1024**3
    base = dict(
        total_ram_bytes=128 * GiB, model_weights_bytes=int(9.7 * GiB),
        usable_bytes_override=75 * GiB, usable_bytes_explicit=True,
        kv_bytes_per_token=3200, requested_context=16384,
    )
    assert round(plan_memory(**base).bank_idle_max_bytes / GiB) == 48
    capped = plan_memory(**base, session_bank_max_bytes=2 * GiB)
    assert round(capped.bank_idle_max_bytes / GiB) == 2
    assert round(capped.bank_steady_bytes / GiB) == 2


def test_explicit_session_bank_env_wins_over_the_plan(monkeypatch) -> None:
    # engine_session is the process that reads MTPLX_SESSION_BANK_MAX_BYTES; an
    # explicit value wins over the auto/plan sizing (so the child_env reaches the
    # actual bank, not just the display).
    from mtplx.engine_session import resolve_session_bank_max_bytes

    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_BYTES", "2GiB")
    max_bytes, auto = resolve_session_bank_max_bytes(int(9.7 * 1024**3))
    assert auto is False
    assert max_bytes == 2 * 1024**3


def test_engine_budget_is_ceiling_minus_reserve() -> None:
    # issue #3: the 75 GiB engine budget (MTPLX_MEMORY_LIMIT_BYTES) IS the 82 GiB
    # profile ceiling minus the 7 GiB reserve — the two are consistent, not
    # competing. The Metal limit governs weights + expert cache.
    from mtplx.expert_profiles import build_expert_streaming_config, load_expert_profiles
    from mtplx.expert_runtime import reconcile_mlx_memory_cap
    from mtplx.expert_streaming_models import get_model_spec

    cfg = build_expert_streaming_config(load_expert_profiles()[PROFILE_NAME])
    plan = cfg.memory_plan(get_model_spec(cfg.model_key))
    engine_budget = reconcile_mlx_memory_cap(plan)
    assert engine_budget == cfg.memory_limit_bytes - cfg.runtime_reserve_bytes
    assert engine_budget == 75 * 1024**3
    assert cfg.memory_limit_bytes == 82 * 1024**3


# --------------------------------------------------------------------------
# window-14: served MTP construct == what the resident load provides
# (the mtp.*_vl gate-bias drop). Real config + manifest metadata; the partition
# tensors are lazy mx.zeros that are never eval'd, so no experts.bin is loaded.
# --------------------------------------------------------------------------
_DTYPE_MAP = {
    "bfloat16": "bfloat16", "bf16": "bfloat16", "float32": "float32", "f32": "float32",
    "float16": "float16", "uint32": "uint32", "u32": "uint32", "uint8": "uint8",
    "int8": "int8", "int32": "int32", "uint16": "uint16", "bool": "bool_",
}


@pytest.mark.skipif(not _MXFP4.exists(), reason="mxfp4 artifact not on this box")
def test_served_mtp_partition_matches_constructed_model_params() -> None:
    import mlx.core as mx

    mx.set_default_device(mx.cpu)  # no GPU
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_config

    from mtplx.expert_manifest import load_expert_manifest
    from mtplx.models.deepseek_v41_loader import (
        deepseek_v41_model_classes,
        partition_text_residents,
    )

    cfg = load_config(_MXFP4)
    Model, ModelArgs = deepseek_v41_model_classes()
    # Construct exactly what the served MTP path constructs (worker W23 head).
    model = Model(
        ModelArgs.from_dict(cfg),
        engram_bank_path=None,
        quantization=cfg.get("quantization"),
        mtp=True,
    )
    model.eval()
    model_params = {k for k, _ in tree_flatten(model.parameters())}

    manifest = load_expert_manifest(_MXFP4 / "expert-manifest.json")
    partition = partition_text_residents(manifest, with_mtp=True)
    # lazy zeros -> sanitize maps/stacks names without materializing weights
    weights = {
        t.tensor: mx.zeros(
            tuple(t.shape), dtype=getattr(mx, _DTYPE_MAP.get(str(t.dtype).lower(), "bfloat16"))
        )
        for t in partition.kept
    }
    weights = model.sanitize(weights)
    provided = set(weights)

    # THE fix: no resident is passed that the model does not declare (strict
    # load_weights would otherwise raise "N parameters not in model").
    extra = provided - model_params
    assert extra == set(), f"resident keys not in model: {sorted(extra)}"

    # The only model params the residents do not provide are the streamed
    # backbone routed experts (bound via bind_streamed_switches, not residents).
    missing = model_params - provided
    assert missing, "expected the streamed switch_mlp experts among model params"
    assert all(".switch_mlp." in name for name in missing), sorted(missing)[:5]


@pytest.mark.skipif(not _MXFP4.exists(), reason="mxfp4 artifact not on this box")
def test_mtp_vl_gate_bias_is_dropped_and_absent_from_the_head() -> None:
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_config

    from mtplx.expert_manifest import load_expert_manifest
    from mtplx.models.deepseek_v41_loader import (
        deepseek_v41_model_classes,
        partition_text_residents,
    )

    manifest = load_expert_manifest(_MXFP4 / "expert-manifest.json")
    kept = {t.tensor for t in partition_text_residents(manifest, with_mtp=True).kept}
    # the text-only DSpark head omits the VL gate bias, so its 3 residents are
    # dropped from the MTP partition (else strict load_weights sees them as
    # parameters not in the model)
    assert not any(n.startswith("mtp.") and n.endswith("_vl") for n in kept)
    # the backbone's VL gate-bias residents are UNTOUCHED (40 kept); the model's
    # sanitize handles them, so the fix is scoped to the mtp.* namespace only
    assert sum(
        1 for n in kept if n.startswith("layers.") and n.endswith("_vl")
    ) == 40

    cfg = load_config(_MXFP4)
    Model, ModelArgs = deepseek_v41_model_classes()
    model = Model(
        ModelArgs.from_dict(cfg), engram_bank_path=None,
        quantization=cfg.get("quantization"), mtp=True,
    )
    model.eval()
    params = {k for k, _ in tree_flatten(model.parameters())}
    # the constructed model declares no ``*_vl`` gate bias at all (sanitize folds
    # the backbone's, and the text-only head omits its own), which is exactly why
    # the mtp.*_vl residents have no home and must be dropped before strict load
    assert not any(n.endswith("_vl") for n in params)


# --------------------------------------------------------------------------
# window-16: the streamed native path installs the DSpark drafter itself
# (the streaming block returns before the general is_deepseek_v41_mtp_config
# dispatch, so it must publish the in-model head -- not raise "unresolved
# streamed MTP backend None").
# --------------------------------------------------------------------------
def test_native_mtp_install_degrades_without_a_head() -> None:
    # inject returns False (degrade-to-AR) rather than raising when the model
    # carries no head or the config declares no stages -- the runtime branch
    # then raises the clear "injection failed", never "unresolved backend".
    from mtplx.models.deepseek_v41 import inject_deepseek_v41_mtp_support

    cfg = {"model_type": "deepseek_v41", "text_config": {"n_mtp_layers": 3}}

    class _NoHead:
        mtp_blocks = None

    assert inject_deepseek_v41_mtp_support(_NoHead(), None, cfg, None) is False
    # a head-bearing model but a config with no MTP stages also degrades
    class _Head:
        mtp_blocks = [1, 2, 3]

    assert inject_deepseek_v41_mtp_support(_Head(), None, {"model_type": "deepseek_v41"}, None) is False


@pytest.mark.skipif(not _MXFP4.exists(), reason="mxfp4 artifact not on this box")
def test_served_native_mtp_constructs_and_publishes_the_drafter() -> None:
    # Runs the served MTP setup far enough to construct the DSpark drafter (the
    # mtp=True resident construct, no experts.bin) and publish it exactly as the
    # runtime.py streaming block's native branch does, then validate it.
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from mlx_lm.utils import load_config

    from mtplx.models.deepseek_v41 import (
        inject_deepseek_v41_mtp_support,
        is_deepseek_v41_mtp_config,
    )
    from mtplx.models.deepseek_v41_loader import deepseek_v41_model_classes
    from mtplx.mtp_patch import validate_mtp_support

    cfg = load_config(_MXFP4)
    assert is_deepseek_v41_mtp_config(cfg)  # the served native-MTP gate

    Model, ModelArgs = deepseek_v41_model_classes()
    model = Model(
        ModelArgs.from_dict(cfg), engram_bank_path=None,
        quantization=cfg.get("quantization"), mtp=True,
    )
    model.eval()
    # the streaming block's native branch: publish the in-model head + validate
    assert inject_deepseek_v41_mtp_support(model, _MXFP4, cfg, None) is True
    assert validate_mtp_support(model) is True


# --------------------------------------------------------------------------
# window-17: served MTP resolves the output head via lm_head (DSV4.1 names it
# `head`, W40 HEAD_MODE codec) -> _install_draft_lm_head no longer raises
# "model has no lm_head and does not tie output projection to embeddings".
# --------------------------------------------------------------------------
def test_served_mtp_resolves_head_via_lm_head_on_a_stub() -> None:
    import types

    import mlx.core as mx
    import mlx.nn as nn

    mx.set_default_device(mx.cpu)
    from mlx.utils import tree_flatten

    from mtplx.draft_lm_head import _install_draft_lm_head

    class _StubDSV41(nn.Module):
        # a stub DSV4.1 model: untied output projection named `head`, exposed
        # under the lm_head alias, plus the DSpark mtp surface (stages carry no
        # shared_head_head, so _install falls to the lm_head lookup).
        def __init__(self, hidden: int = 64, vocab: int = 128):
            super().__init__()
            self.head = nn.Linear(hidden, vocab, bias=False)
            self._head_mode = "bf16"
            self.mtp = types.SimpleNamespace(layers=[types.SimpleNamespace()])

        @property
        def lm_head(self):
            return self.head

        def _apply_head(self, source):  # the real bf16 codec head path
            return self.head(source.astype(self.head.weight.dtype)).astype(mx.float32)

    mx.random.seed(0)
    model = _StubDSV41()
    mx.eval(model.parameters())

    # lm_head is a plain alias -> parameters() counts the head once (under head.*)
    keys = {k for k, _ in tree_flatten(model.parameters())}
    assert model.lm_head is model.head
    assert not any(k.startswith("lm_head") for k in keys)

    # the served MTP setup resolves the head instead of raising AttributeError
    rt = types.SimpleNamespace(model=model)
    _install_draft_lm_head(rt, bits=4, group_size=64, mode="affine")
    assert getattr(model, "_mtplx_draft_lm_head", None) is not None

    # the resolved head callable equals the model's own AR head path output
    h = mx.random.normal((3, 64)).astype(mx.bfloat16)
    assert mx.allclose(model.lm_head(h).astype(mx.float32), model._apply_head(h))


@pytest.mark.skipif(not _MXFP4.exists(), reason="mxfp4 artifact not on this box")
def test_real_dsv41_model_exposes_lm_head_and_served_attrs() -> None:
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_config

    from mtplx.models.deepseek_v41_loader import deepseek_v41_model_classes

    cfg = load_config(_MXFP4)
    Model, ModelArgs = deepseek_v41_model_classes()
    model = Model(
        ModelArgs.from_dict(cfg), engram_bank_path=None,
        quantization=cfg.get("quantization"), mtp=True,
    )
    model.eval()
    # lm_head aliases the untied `head`; no duplicate lm_head parameter subtree
    assert model.lm_head is model.head
    keys = {k for k, _ in tree_flatten(model.parameters())}
    assert not any(k.startswith("lm_head") for k in keys)
    # the served MTP/verify path's model-attribute assumptions hold for DSV4.1:
    # rt.embed_tokens -> model.model.embed_tokens; model.model.layers exists.
    assert model.model.embed_tokens is not None
    assert len(model.model.layers) == int(cfg["text_config"]["num_hidden_layers"])
