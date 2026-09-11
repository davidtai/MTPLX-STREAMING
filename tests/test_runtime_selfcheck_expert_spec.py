"""Served-order regression coverage for the expert-streaming kernel self-check.

Background (the bug this file locks down): commit c5f80852 re-threaded the
expert-streaming load onto the upstream runtime and re-added the streaming
self-check call site in ``mtplx/runtime.py``::

    maybe_run_model_selfcheck(model, expert_spec=getattr(expert_runtime, "spec", None))

but kept upstream's single-argument ``maybe_run_model_selfcheck(model)``
signature, so *every* ``mtplx serve`` died at load with::

    TypeError: maybe_run_model_selfcheck() got an unexpected keyword argument 'expert_spec'

The keyword is bound before the function body runs, so even a disabled
self-check crashed the load.  A per-function unit test of
``maybe_run_model_selfcheck`` would not have caught a call site that passed a
differently-named kwarg, so the primary tests here drive the REAL served load
path (``mtplx.runtime.load`` -> ``_load_impl``) through its streaming branch and
assert the actual call site invokes the self-check with the runtime's spec (and
without one for the dense/None case), and that the load completes.  Both
served-order tests fail on origin/main's code (TypeError at the call site) and
pass with the restored kernel_selfcheck implementation.

All work is stock ``mx.gather_qmm`` / ``mx.quantized_matmul`` / ``mx.quantize``
and pins MLX to the CPU, so the whole file runs GPU-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import pytest

# Worker tests must pin MLX to CPU (MLX defaults to Metal); do it at import and
# again per test so no lane silently runs on the GPU.
mx.set_default_device(mx.cpu)

from mtplx import kernel_selfcheck as ks
from mtplx.expert_runtime import ExpertStreamingConfig
from mtplx.expert_streaming_models import (
    GLM52_EXPERT_Q1T,
    HY3_EXPERT_ONLY_Q4,
    HY3_EXPERT_OQ2E,
    HY3_EXPERT_Q2,
)
from mtplx.kernel_selfcheck import (
    lane_disabled,
    report_for_health,
    run_kernel_selfcheck,
)
from mtplx.runtime import load


@pytest.fixture(autouse=True)
def _cpu_and_clean_selfcheck(monkeypatch: pytest.MonkeyPatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setenv("MTPLX_KERNEL_SELFCHECK", "1")
    monkeypatch.delenv("MTPLX_NAX_VERIFY", raising=False)
    monkeypatch.delenv("MTPLX_GQA_PACKED_SDPA", raising=False)
    ks._reset_for_tests()
    yield
    ks._reset_for_tests()


# --------------------------------------------------------------------------- #
# Served-order tests: drive the real streaming load path to its self-check call
# --------------------------------------------------------------------------- #


class _PlainTarget:
    """A resident model with no quantized layers (generic-dtype self-check)."""


class _FakeExpertRuntime:
    """Stands in for an opened ExpertStreamingRuntime; carries the spec the
    call site reads via ``getattr(expert_runtime, "spec", None)``."""

    def __init__(self, spec: Any) -> None:
        self.spec = spec
        self.closed = False

    def close(self, *, timeout: float | None = None) -> None:
        del timeout
        self.closed = True


def _streaming_config(model_key: str = "hy3-expert-oq2e") -> ExpertStreamingConfig:
    return ExpertStreamingConfig(
        model_key=model_key,
        memory_limit_bytes=1,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
    )


def _model_root(tmp_path: Path) -> Path:
    root = tmp_path / "streamed-model"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "glm_moe_dsa"}), encoding="utf-8"
    )
    return root


def _drive_streamed_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    runtime_spec: Any,
    get_model_spec_return: Any,
):
    """Run the REAL ``load()`` streaming branch (mtp=False) with the heavy
    allocation/construction internals stubbed, letting the genuine call site
    invoke the genuine ``maybe_run_model_selfcheck``.

    Returns ``(runtime, recorded)`` where ``recorded`` captures the kwargs the
    self-check call site actually passed.
    """
    root = _model_root(tmp_path)
    model = _PlainTarget()
    fake_runtime = _FakeExpertRuntime(runtime_spec)

    def open_stub(*_args, **kwargs):
        # Mirror ExpertStreamingRuntime.open: the runtime carries the streamed
        # spec the call site later reads.  Prefer the explicit test spec but
        # fall back to the spec the load actually threaded in.
        fake_runtime.spec = runtime_spec
        del kwargs
        return fake_runtime

    monkeypatch.setattr(
        ExpertStreamingConfig,
        "memory_plan",
        lambda _self, _spec, **_kw: SimpleNamespace(
            fits_fixed=True, unallocated_bytes=0
        ),
    )
    monkeypatch.setattr(
        "mtplx.expert_streaming_models.get_model_spec",
        lambda _model_key: get_model_spec_return,
    )
    monkeypatch.setattr(
        "mtplx.models.expert_mlx.make_mlx_slot_buffer_allocator",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(
        "mtplx.models.expert_mlx.make_mlx_component_bank_allocator",
        lambda *_a, **_k: object(),
    )
    from mtplx import expert_runtime as expert_runtime_module

    monkeypatch.setattr(
        expert_runtime_module.ExpertStreamingRuntime,
        "open",
        staticmethod(open_stub),
    )
    monkeypatch.setattr(
        "mtplx.resident_loader.construct_resident_model",
        lambda *_a, **_k: SimpleNamespace(
            model=model, report=SimpleNamespace(as_dict=lambda: {})
        ),
    )
    monkeypatch.setattr(
        "mtplx.runtime._load_tokenizer_resilient", lambda *_a, **_k: object()
    )
    monkeypatch.setattr(
        "mtplx.attention_split.configure_split_full_attention",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "mtplx.native_mlp.configure_native_mlp", lambda *_a, **_k: None
    )
    monkeypatch.setattr("mtplx.nax_verify.nax_env_enabled", lambda: False)

    # Spy that forwards to the REAL self-check: proves which keyword the call
    # site passed while still exercising the genuine probe.  On origin/main the
    # forwarded call raises TypeError (the callee rejects expert_spec).
    recorded: dict[str, Any] = {}
    real_selfcheck = ks.maybe_run_model_selfcheck

    def selfcheck_spy(model_arg, **kwargs):
        recorded["called"] = True
        recorded["expert_spec"] = kwargs.get("expert_spec", "MISSING")
        return real_selfcheck(model_arg, **kwargs)

    monkeypatch.setattr(
        "mtplx.kernel_selfcheck.maybe_run_model_selfcheck", selfcheck_spy
    )

    runtime = load(
        root,
        mtp=False,
        expert_streaming_config=_streaming_config(),
        expert_manifest=root / "expert-manifest.json",
    )
    return runtime, model, recorded


def test_streamed_serve_load_invokes_selfcheck_with_expert_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Streaming load: the served call site passes the runtime's expert spec,
    the routed-expert gather_qmm lane runs at the bank's 2-bit gs128 format, and
    the load completes.  FAILS on origin/main (TypeError at the call site)."""
    runtime, model, recorded = _drive_streamed_load(
        monkeypatch,
        tmp_path,
        runtime_spec=HY3_EXPERT_OQ2E,
        get_model_spec_return=HY3_EXPERT_OQ2E,
    )

    assert runtime.model is model  # load ran to completion
    # The call site threaded the runtime's spec (not None) into the self-check.
    assert recorded["called"] is True
    assert recorded["expert_spec"] is HY3_EXPERT_OQ2E
    # ...which engaged the expert-gather lane at the bank's own quant format.
    health = report_for_health()
    assert health["ran"] is True
    assert health["expert_gather"] == "ok", health
    assert not lane_disabled("expert_gather")


def test_streamed_serve_load_without_spec_matches_dense_selfcheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A streamed runtime with no spec passes expert_spec=None to the served
    call site; the self-check then behaves exactly as the dense/resident-only
    path (no expert_gather lane) and the load completes.  Also FAILS on
    origin/main, since the call site passes the expert_spec keyword regardless
    of its value."""
    runtime, model, recorded = _drive_streamed_load(
        monkeypatch,
        tmp_path,
        runtime_spec=None,
        get_model_spec_return=HY3_EXPERT_OQ2E,
    )

    assert runtime.model is model
    assert recorded["called"] is True
    assert recorded["expert_spec"] is None
    health = report_for_health()
    assert health["ran"] is True
    assert "expert_gather" not in health  # byte-identical to upstream
    assert not lane_disabled("expert_gather")


# --------------------------------------------------------------------------- #
# Unit coverage for the restored kernel_selfcheck helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spec,expected_bits,expected_group_size",
    [
        (HY3_EXPERT_OQ2E, 2, 128),
        (HY3_EXPERT_Q2, 2, 64),
        (HY3_EXPERT_ONLY_Q4, 4, 64),
    ],
)
def test_expert_signature_derived_from_spec(
    spec, expected_bits, expected_group_size
) -> None:
    assert ks._expert_quant_signature(spec) == (
        mx.bfloat16,
        expected_bits,
        expected_group_size,
    )


def test_expert_signature_none_without_spec() -> None:
    assert ks._expert_quant_signature(None) is None


def test_expert_signature_none_for_shadow_codec() -> None:
    # Shadow-codec (q1 lane) banks do not run the affine gather_qmm path, so
    # they must yield no expert signature and add no gather lane.
    assert GLM52_EXPERT_Q1T.expert_codec != "affine"
    assert ks._expert_quant_signature(GLM52_EXPERT_Q1T) is None


@pytest.mark.parametrize("group_size", [64, 128], ids=["gs64", "gs128"])
def test_expert_gather_lane_passes_on_healthy_bank(group_size) -> None:
    report = run_kernel_selfcheck(
        mx.bfloat16, 4, 64, expert_signature=(mx.bfloat16, 2, group_size)
    )
    assert report["lanes"]["expert_gather"] == "ok"
    assert report["dmax"]["expert_gather"] <= ks._QMM_TOLERANCE
    assert not lane_disabled("expert_gather")


def test_expert_gather_lane_fails_closed_on_corrupt_kernel(monkeypatch) -> None:
    original = mx.gather_qmm

    def corrupted(*args, **kwargs):
        return original(*args, **kwargs) + 1000.0

    monkeypatch.setattr(mx, "gather_qmm", corrupted)
    report = run_kernel_selfcheck(
        mx.bfloat16, 4, 64, expert_signature=(mx.bfloat16, 2, 128)
    )
    assert report["lanes"]["expert_gather"] == "fallback"
    assert lane_disabled("expert_gather")
    assert report_for_health()["expert_gather"] == "fallback"


def test_expert_gather_lane_fails_closed_on_wrong_group_size() -> None:
    # A group_size that does not divide the synthetic bank's K raises inside the
    # probe; the recorder catches it as a hard fallback (dmax == inf).
    report = run_kernel_selfcheck(
        mx.bfloat16, 4, 64, expert_signature=(mx.bfloat16, 2, 96)
    )
    assert report["lanes"]["expert_gather"] == "fallback"
    assert report["dmax"]["expert_gather"] == float("inf")
    assert lane_disabled("expert_gather")


def test_check_expert_gather_mismatched_group_size_raises() -> None:
    # Bank quantized at gs=64, gather run at gs=128: gather_qmm's weight/scales
    # contract fails closed.
    with pytest.raises(Exception):
        y = ks._check_expert_gather(mx, mx.bfloat16, 2, 128, bank_group_size=64)
        mx.eval(y)


def test_no_expert_lane_without_signature() -> None:
    # Non-streaming path: no expert_signature -> the report carries no expert
    # lane at all (not even "skipped"), keeping dense loads byte-identical.
    report = run_kernel_selfcheck(mx.bfloat16, 4, 64)
    assert "expert_gather" not in report["lanes"]
    assert "expert_gather" not in report["dmax"]
    assert not lane_disabled("expert_gather")


def test_maybe_run_dense_model_adds_no_expert_lane() -> None:
    report = ks.maybe_run_model_selfcheck(_PlainTarget())
    assert report is not None
    assert "expert_gather" not in report["lanes"]
    assert not lane_disabled("expert_gather")


def test_maybe_run_streaming_spec_adds_expert_lane() -> None:
    report = ks.maybe_run_model_selfcheck(_PlainTarget(), expert_spec=HY3_EXPERT_OQ2E)
    assert report is not None
    assert report["lanes"]["expert_gather"] == "ok"
    assert not lane_disabled("expert_gather")
    assert report_for_health()["expert_gather"] == "ok"


def test_maybe_run_disabled_skips_expert_lane(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_KERNEL_SELFCHECK", "0")
    assert (
        ks.maybe_run_model_selfcheck(_PlainTarget(), expert_spec=HY3_EXPERT_OQ2E)
        is None
    )
