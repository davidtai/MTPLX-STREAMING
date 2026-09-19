"""W53 served-decode stage timer: zero-overhead-when-off, table when armed,
receipt export, and exactness of the lazy-trace-totals saving.

CPU only: pins mx.cpu, uses a tiny deterministic stub model (no experts.bin,
no Metal), well under the 3 GB RSS budget.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np

mx.set_default_device(mx.cpu)

from mtplx.generation import generate_ar  # noqa: E402
from mtplx.mtp_patch import MTPContract  # noqa: E402
from mtplx.runtime import MTPLXRuntime  # noqa: E402
from mtplx.sampling import SamplerConfig  # noqa: E402
from mtplx.serve_stage_timing import (  # noqa: E402
    StageTimer,
    stage_timing_enabled,
    write_stage_timing_receipt,
)

VOCAB = 8
MARGIN = 10.0


class _CyclicTokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(str(int(token)) for token in tokens)


class _CyclicModel:
    """After token t the model deterministically wants (t + 1) % VOCAB."""

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def _logits_for(self, last_tokens):
        rows = []
        for token in last_tokens:
            row = [0.0] * VOCAB
            row[(int(token) + 1) % VOCAB] = MARGIN
            rows.append(row)
        return mx.array([rows], dtype=mx.float32)

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden=False,
        hidden_variant=None,
        emit_logits=True,
        logits_keep=None,
    ):
        tokens = [int(t) for t in np.asarray(input_ids).reshape(-1)]
        keep = len(tokens) if logits_keep is None else min(len(tokens), max(1, int(logits_keep)))
        logits = self._logits_for(tokens[-keep:]) if emit_logits else None
        hidden = mx.zeros((1, len(tokens), 2), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        if return_hidden:
            return logits, hidden
        return logits


def _cyclic_runtime():
    return MTPLXRuntime(
        model=_CyclicModel(),
        tokenizer=_CyclicTokenizer(),
        model_path=Path("tiny-cyclic"),
        mtp_enabled=False,
        contract=MTPContract(),
    )


def _run_ar(max_tokens=48, seed=7):
    return generate_ar(
        _cyclic_runtime(),
        [0],
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.6, top_p=0.95, top_k=4),
        seed=seed,
        stop_token_ids=set(),
    )


# ---- StageTimer unit behaviour -------------------------------------------


def test_stage_timer_disabled_is_a_noop():
    timer = StageTimer(enabled=False)
    timer.begin()
    timer.lap("sample")
    timer.add("forward", 1.0)
    timer.tick_token(5)
    assert timer.summary() == {}


def test_stage_timer_enabled_builds_a_table():
    timer = StageTimer(enabled=True)
    for _ in range(3):
        timer.begin()
        timer.lap("sample")
        timer.add("forward", 0.01)
        timer.tick_token()
    summary = timer.summary()
    assert summary["enabled"] is True
    assert summary["tokens"] == 3
    assert "sample" in summary["stages"]
    assert "forward" in summary["stages"]
    assert summary["stages"]["forward"]["count"] == 3
    # per_token_ms reported for every stage; all JSON-primitive.
    for stage in summary["stages"].values():
        assert set(stage) >= {"total_s", "count", "mean_ms", "per_token_ms", "share"}


def test_stage_timing_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("MTPLX_SERVE_STAGE_TIMING", raising=False)
    assert stage_timing_enabled() is False
    monkeypatch.setenv("MTPLX_SERVE_STAGE_TIMING", "1")
    assert stage_timing_enabled() is True
    monkeypatch.setenv("MTPLX_SERVE_STAGE_TIMING", "off")
    assert stage_timing_enabled() is False


# ---- receipt export -------------------------------------------------------


def test_receipt_is_noop_without_env(monkeypatch):
    monkeypatch.delenv("MTPLX_SERVE_STAGE_TIMING_RECEIPT", raising=False)
    assert write_stage_timing_receipt({"tokens": 1}, request_id="r", mode="ar") is None


def test_receipt_is_noop_on_empty_summary(monkeypatch, tmp_path):
    monkeypatch.setenv("MTPLX_SERVE_STAGE_TIMING_RECEIPT", str(tmp_path))
    assert write_stage_timing_receipt({}, request_id="r", mode="ar") is None


def test_receipt_written_to_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MTPLX_SERVE_STAGE_TIMING_RECEIPT", str(tmp_path))
    path = write_stage_timing_receipt(
        {"tokens": 2, "stages": {}}, request_id="abc", mode="ar"
    )
    assert path is not None and Path(path).exists()
    import json

    payload = json.loads(Path(path).read_text())
    assert payload["event"] == "mtplx_serve_stage_timing"
    assert payload["generation_mode"] == "ar"
    assert payload["tokens"] == 2


def test_receipt_written_to_file_path(monkeypatch, tmp_path):
    target = tmp_path / "stage.json"
    monkeypatch.setenv("MTPLX_SERVE_STAGE_TIMING_RECEIPT", str(target))
    path = write_stage_timing_receipt(
        {"tokens": 1, "stages": {}}, request_id="x", mode="mtp"
    )
    assert Path(path) == target and target.exists()


# ---- generate_ar wiring ---------------------------------------------------


def test_generate_ar_stage_timing_off_by_default(monkeypatch):
    monkeypatch.delenv("MTPLX_SERVE_STAGE_TIMING", raising=False)
    out = _run_ar()
    assert out.stats.serve_stage_timing == {}


def test_generate_ar_stream_counters_absent_without_streaming_runtime():
    # The cyclic stub has no expert_streaming and no engram banks, so the
    # per-request streaming-counter delta is empty (never raises).
    out = _run_ar()
    assert out.stats.serve_stream_counters == {}


def test_generate_ar_stream_counters_delta_when_snapshot_present():
    # Inject a streaming snapshot that advances every call: generate_ar's
    # decode-phase bracket must surface a non-empty expert_cache delta.
    rt = _cyclic_runtime()
    state = {"hits": 0, "misses": 0, "loads": 0, "bytes": 0}

    def _snapshot():
        # ~1 miss + 4 hits + 12 bytes per call; call count tracks decode steps.
        state["hits"] += 4
        state["misses"] += 1
        state["loads"] += 1
        state["bytes"] += 12
        return {
            "cache": {
                "expert_hits": state["hits"],
                "expert_misses": state["misses"],
                "transient_loads": state["loads"],
                "persistent_loads": 0,
                "bytes_read": state["bytes"],
                "route_calls": state["hits"] + state["misses"],
            }
        }

    rt.expert_streaming_snapshot = _snapshot
    out = generate_ar(
        rt,
        [0],
        max_tokens=32,
        sampler=SamplerConfig(temperature=0.6, top_p=0.95, top_k=4),
        seed=7,
        stop_token_ids=set(),
    )
    ec = out.stats.serve_stream_counters["expert_cache"]
    assert ec["expert_misses"] > 0
    assert ec["records_streamed"] > 0
    assert ec["bytes_read_per_token"] > 0
    assert out.stats.serve_stream_counters["phase"] == "decode"


def test_generate_ar_stage_timing_populates_table(monkeypatch):
    monkeypatch.setenv("MTPLX_SERVE_STAGE_TIMING", "1")
    out = _run_ar()
    table = out.stats.serve_stage_timing
    assert table.get("enabled") is True
    assert table["tokens"] == len(out.tokens)
    stages = table["stages"]
    # The AR loop banks these stages every step it runs a forward.
    assert "sample" in stages
    assert "emit" in stages
    assert "forward" in stages
    assert "eval" in stages
    assert stages["sample"]["count"] >= 1
    # forward runs once fewer than tokens (final token is not forwarded).
    assert stages["forward"]["count"] == max(0, len(out.tokens) - 1)


# ---- lazy-trace-totals is exactness-neutral -------------------------------


def test_lazy_trace_totals_is_byte_identical(monkeypatch):
    monkeypatch.setenv("MTPLX_AR_LAZY_TRACE_TOTALS", "1")
    lazy = _run_ar(seed=11)
    monkeypatch.setenv("MTPLX_AR_LAZY_TRACE_TOTALS", "0")
    eager = _run_ar(seed=11)
    assert list(lazy.tokens) == list(eager.tokens)
    assert lazy.text == eager.text
