"""W110 -- CPU tests for the DECODE-path per-record sha256 lever + counters.

Covers ``docs/deepseek-v41/W110_DECODE_RECORD_HASH.md``:

  * ``MTPLX_DSV41_VERIFY_RECORD_HASHES`` parsing in
    ``deepseek_v41_loader.build_streaming_config`` -- read AT USE, AUTHORITATIVE
    when set (0/1) over an explicit caller value, current behaviour when unset;
  * the lever touches ONLY the DECODE-path ``verify_record_hashes`` and never the
    admission/open integrity fields (``verify_artifact_headers`` /
    ``verify_sidecar_hash_at_open``), so open-time verification stays on;
  * the io-thread engagement counters (``records_hashed`` / ``records_unhashed`` /
    ``hash_ns_total``) reflect whether the per-record sha256 ran, on both the
    single-record and batched (v2 overlap) sidecar read paths;
  * bytes are byte-identical with hashing ON vs OFF (hashing never mutates bytes);
  * a corrupt record (on-disk bytes disagree with the manifest's trusted hash) is
    REJECTED with hashing ON and -- documented -- lands silently with it OFF;
  * the W110 receipt-bug fix in ``ab_decode_env_levers._resolved_plan``: the
    gate-prefetch armed state / predict width come from the RUNTIME object (the v2
    auto-arm) and the raw env, not from ``os.environ`` alone;
  * the receipt surfacing: ``serve_stream_counters`` passes the io block through and
    the decode-scoped delta reports the hashing counters.

No GPU, no Metal, no model load, no ``~/models`` read, no server, no network. The
synthetic bank is a few bytes in ``tmp_path``. MLX is pinned to the CPU device at
import per memory/worker-tests-must-pin-mlx-cpu.md ("no GPU" is not enough -- MLX
defaults to Metal). Run under ``nice -n 19`` and without ``pytest -n auto`` (one
file per process).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mtplx.expert_io import ExpertIOIntegrityError, PositionalExpertReader
from mtplx.expert_streaming_models import get_model_spec
from mtplx.models import deepseek_v41_loader as loader
from mtplx import serve_stream_counters

_VRH_ENV = "MTPLX_DSV41_VERIFY_RECORD_HASHES"
# A file-free, in-memory pinned descriptor for the v2 cell's codec (mxfp4). No
# artifact/model bytes are ever read -- build_streaming_config only builds a config.
_SPEC_KEY = "deepseek-v41-flash-expert-mxfp4"
_KNOB = 60 * 1024**3
_RESERVE = 7 * 1024**3

_WT = Path(__file__).resolve().parents[1]
_AB_PATH = _WT / "scripts" / "deepseek_v41" / "ab_decode_env_levers.py"


# --------------------------------------------------------------------------
# Synthetic sidecar bank helpers (no model, a few bytes on disk).
# --------------------------------------------------------------------------
class _Comp:
    """A component-slot destination: N buffers whose views land record bytes."""

    def __init__(self, lengths: tuple[int, ...]) -> None:
        self.buffers = tuple(bytearray(n) for n in lengths)

    def record_views(self, _record: object) -> tuple[memoryview, ...]:
        return tuple(memoryview(b) for b in self.buffers)

    def payload(self) -> bytes:
        return b"".join(self.buffers)


def _record(layer: int, expert: int, offset: int, payload: bytes, sha: str | None):
    # Two equal segments so both the flat and component views are exercised.
    half = len(payload) // 2
    return SimpleNamespace(
        layer=layer,
        expert=expert,
        logical_bytes=len(payload),
        segments=(SimpleNamespace(length=half), SimpleNamespace(length=half)),
        sidecar_offset=offset,
        sidecar_length=len(payload),
        sha256=sha,
    )


def _bank(tmp_path: Path, *payloads: bytes):
    blob = b"".join(payloads)
    (tmp_path / "experts.bin").write_bytes(blob)
    manifest = SimpleNamespace(sidecar=SimpleNamespace(file="experts.bin"))
    records, offset = [], 0
    for i, p in enumerate(payloads):
        records.append(_record(1, i, offset, p, hashlib.sha256(p).hexdigest()))
        offset += len(p)
    return manifest, records


def _cfg(**overrides):
    spec = get_model_spec(_SPEC_KEY)
    return loader.build_streaming_config(
        spec,
        memory_limit_bytes=_KNOB,
        max_live_kv_tokens=17408,
        runtime_reserve_bytes=_RESERVE,
        **overrides,
    )


def _load_ab():
    spec = importlib.util.spec_from_file_location("dsv41_ab_w110", _AB_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ==========================================================================
# 1. Env parsing in build_streaming_config (read at use, authoritative on set).
# ==========================================================================
def test_env_unset_leaves_current_behaviour(monkeypatch):
    monkeypatch.delenv(_VRH_ENV, raising=False)
    # Config default (True) survives when neither the caller nor the env sets it.
    assert _cfg().verify_record_hashes is True
    # An explicit caller value wins when the env is unset.
    assert _cfg(verify_record_hashes=False).verify_record_hashes is False


def test_env_zero_is_authoritative_over_explicit_true(monkeypatch):
    monkeypatch.setenv(_VRH_ENV, "0")
    assert _cfg(verify_record_hashes=True).verify_record_hashes is False


def test_env_one_is_authoritative_over_explicit_false(monkeypatch):
    monkeypatch.setenv(_VRH_ENV, "1")
    assert _cfg(verify_record_hashes=False).verify_record_hashes is True


def test_env_empty_string_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv(_VRH_ENV, "")
    # Empty is not a value: the caller (or default) still decides.
    assert _cfg(verify_record_hashes=False).verify_record_hashes is False
    assert _cfg().verify_record_hashes is True


def test_lever_does_not_touch_admission_open_integrity_fields(monkeypatch):
    """The lever is DECODE-path only: with hashing forced off, the open/admission
    integrity fields (verify_artifact_headers / verify_sidecar_hash_at_open) keep
    their defaults, so ExpertStreamingRuntime.open still verifies the manifest."""
    monkeypatch.setenv(_VRH_ENV, "0")
    cfg = _cfg(verify_record_hashes=True)
    assert cfg.verify_record_hashes is False  # decode path off
    assert cfg.verify_artifact_headers is True  # open path untouched (default)
    assert cfg.verify_sidecar_hash_at_open is False  # untouched (default)


# ==========================================================================
# 2. io-thread engagement counters + byte-identity (single + batch paths).
# ==========================================================================
def test_batch_hashing_on_counts_records_hashed_and_times(tmp_path):
    manifest, (r0, r1) = _bank(tmp_path, b"A" * 8, b"B" * 8)
    d0, d1 = _Comp((4, 4)), _Comp((4, 4))
    with PositionalExpertReader(tmp_path, use_native=False) as reader:
        digests = reader.read_component_records_into(
            manifest, ((r0, d0), (r1, d1)), verify_hash=True
        )
        m = reader.metrics.as_dict()
    assert digests == (r0.sha256, r1.sha256)
    assert d0.payload() == b"A" * 8 and d1.payload() == b"B" * 8
    assert m["records_hashed"] == 2
    assert m["records_unhashed"] == 0
    assert m["hash_ns_total"] > 0


def test_batch_hashing_off_skips_and_counts_unhashed(tmp_path):
    manifest, (r0, r1) = _bank(tmp_path, b"A" * 8, b"B" * 8)
    d0, d1 = _Comp((4, 4)), _Comp((4, 4))
    with PositionalExpertReader(tmp_path, use_native=False) as reader:
        digests = reader.read_component_records_into(
            manifest, ((r0, d0), (r1, d1)), verify_hash=False
        )
        m = reader.metrics.as_dict()
    assert digests == ("unverified", "unverified")
    assert m["records_hashed"] == 0
    assert m["records_unhashed"] == 2
    assert m["hash_ns_total"] == 0


def test_single_record_hashing_on_and_off_counts(tmp_path):
    manifest, (r0,) = _bank(tmp_path, b"C" * 8)
    on, off = _Comp((4, 4)), _Comp((4, 4))
    with PositionalExpertReader(tmp_path, use_native=False) as reader:
        dig_on = reader.read_record_into(manifest, r0, on, verify_hash=True)
        m_on = dict(reader.metrics.as_dict())
    with PositionalExpertReader(tmp_path, use_native=False) as reader:
        dig_off = reader.read_record_into(manifest, r0, off, verify_hash=False)
        m_off = dict(reader.metrics.as_dict())
    assert dig_on == r0.sha256
    assert m_on["records_hashed"] == 1 and m_on["records_unhashed"] == 0
    assert m_on["hash_ns_total"] > 0
    assert dig_off == "unverified"
    assert m_off["records_hashed"] == 0 and m_off["records_unhashed"] == 1
    assert m_off["hash_ns_total"] == 0


def test_bytes_are_identical_hashing_on_vs_off(tmp_path):
    """Hashing never mutates bytes: the slot payload is identical either way."""
    manifest, (r0, r1) = _bank(tmp_path, b"XY" * 4, b"ZW" * 4)
    on0, on1 = _Comp((4, 4)), _Comp((4, 4))
    off0, off1 = _Comp((4, 4)), _Comp((4, 4))
    with PositionalExpertReader(tmp_path, use_native=False) as reader:
        reader.read_component_records_into(
            manifest, ((r0, on0), (r1, on1)), verify_hash=True
        )
    with PositionalExpertReader(tmp_path, use_native=False) as reader:
        reader.read_component_records_into(
            manifest, ((r0, off0), (r1, off1)), verify_hash=False
        )
    assert on0.payload() == off0.payload()
    assert on1.payload() == off1.payload()


# ==========================================================================
# 3. Corruption: rejected with hashing ON; documented behaviour with it OFF.
# ==========================================================================
def _corrupt_record(tmp_path: Path):
    # On-disk bytes are b"A"*8, but the manifest's trusted hash is of DIFFERENT
    # bytes -- i.e. a record whose SSD bytes were mutated after admission.
    (tmp_path / "experts.bin").write_bytes(b"A" * 8)
    manifest = SimpleNamespace(sidecar=SimpleNamespace(file="experts.bin"))
    wrong_sha = hashlib.sha256(b"Z" * 8).hexdigest()
    return manifest, _record(1, 0, 0, b"A" * 8, wrong_sha)


def test_corrupt_record_rejected_with_hashing_on(tmp_path):
    manifest, bad = _corrupt_record(tmp_path)
    dest = _Comp((4, 4))
    with PositionalExpertReader(tmp_path, use_native=False) as reader:
        with pytest.raises(ExpertIOIntegrityError):
            reader.read_component_records_into(
                manifest, ((bad, dest),), verify_hash=True
            )
        # The mismatch is recorded as an integrity error (fail-closed).
        assert reader.metrics.as_dict()["integrity_errors"] == 1


def test_corrupt_record_lands_silently_with_hashing_off(tmp_path):
    """DOCUMENTED behaviour with the lever OFF: no per-record re-check runs, so a
    record whose SSD bytes disagree with the manifest's trusted hash is NOT
    detected -- the (corrupt) on-disk bytes land in the slot and no error is
    raised. This is the residual risk the ADMITTED_DESCRIPTOR_SECURITY_BOUNDARY
    declares out of the local-artifact threat model; the admission receipt at open
    (pinned fd + header/size checks) still stands."""
    manifest, bad = _corrupt_record(tmp_path)
    dest = _Comp((4, 4))
    with PositionalExpertReader(tmp_path, use_native=False) as reader:
        digests = reader.read_component_records_into(
            manifest, ((bad, dest),), verify_hash=False
        )
        m = reader.metrics.as_dict()
    assert digests == ("unverified",)
    assert dest.payload() == b"A" * 8  # the on-disk bytes, unverified
    assert m["integrity_errors"] == 0
    assert m["records_unhashed"] == 1


# ==========================================================================
# 4. Receipt-bug fix: _resolved_plan reflects the ACTUAL armed state.
# ==========================================================================
def _stub_runtime(prefetch_slots: int, runner_block: dict | None):
    return SimpleNamespace(
        plan=SimpleNamespace(
            transient_slots=48, persistent_slots=10, slots_per_layer=49
        ),
        spec=SimpleNamespace(expert_record_bytes=18_800_000, routed_layer_count=40),
        config=SimpleNamespace(prefetch_slots=prefetch_slots, split_route_release="deferred"),
        resource_telemetry_snapshot=lambda: (
            {"runner": runner_block} if runner_block else {}
        ),
    )


_STUB_ARGS = SimpleNamespace(
    transient_slots=None, _dsv41_plan_overrides=None, expert_profile="none"
)


def test_resolved_plan_reports_v2_auto_arm_from_runtime(monkeypatch):
    """The window-41 bug: RUNNER=v2 auto-arms the ring (prefetch_slots=48) with
    MTPLX_DSV41_GATE_PREFETCH UNSET; the receipt must show armed=True / k=6 (the
    runner block's resolved prefetch_k), not armed=False / k=0 from the env."""
    ab = _load_ab()
    monkeypatch.delenv(ab.GATE_PREFETCH_ENV, raising=False)
    rp = ab._resolved_plan(_stub_runtime(48, {"prefetch_k": 6}), _STUB_ARGS)
    assert rp["gate_prefetch_armed"] is True
    assert rp["gate_prefetch_k"] == 6  # the resolved predict width, NOT 48//2
    assert rp["gate_prefetch_env"] is None  # raw env for provenance
    assert rp["prefetch_slots"] == 48


def test_resolved_plan_control_has_no_ring(monkeypatch):
    ab = _load_ab()
    monkeypatch.delenv(ab.GATE_PREFETCH_ENV, raising=False)
    rp = ab._resolved_plan(_stub_runtime(0, None), _STUB_ARGS)
    assert rp["gate_prefetch_armed"] is False
    assert rp["gate_prefetch_k"] == 0


def test_resolved_plan_explicit_gate_falls_back_to_half_ring(monkeypatch):
    """An explicit GATE_PREFETCH-only arm (no v2 runner block) sizes the ring 2*k,
    so //2 recovers the explicit predict width when no runner block is present."""
    ab = _load_ab()
    monkeypatch.setenv(ab.GATE_PREFETCH_ENV, "10")
    rp = ab._resolved_plan(_stub_runtime(20, None), _STUB_ARGS)
    assert rp["gate_prefetch_armed"] is True
    assert rp["gate_prefetch_k"] == 10


def test_resolved_plan_explicit_env_armed_but_no_ring_raises(monkeypatch):
    """The W93 explicit-lever guard survives: an explicit env armed with NO ring
    built would measure control-vs-control, so fail loud."""
    ab = _load_ab()
    monkeypatch.setenv(ab.GATE_PREFETCH_ENV, "10")
    with pytest.raises(AssertionError):
        ab._resolved_plan(_stub_runtime(0, None), _STUB_ARGS)


# ==========================================================================
# 5. Receipt surfacing: serve_stream_counters passes io through + deltas it.
# ==========================================================================
def test_snapshot_stream_counters_passes_io_block_through():
    snap = {
        "cache": {},
        "io": {"records_hashed": 5, "records_unhashed": 0, "hash_ns_total": 1000},
    }
    rt = SimpleNamespace(snapshot=lambda: snap)
    out = serve_stream_counters.snapshot_stream_counters(rt)
    assert out.get("io") == snap["io"]


def test_stream_counters_delta_reports_decode_scoped_hashing():
    before = {"io": {"records_hashed": 10, "records_unhashed": 0, "hash_ns_total": 2_000_000}}
    after = {"io": {"records_hashed": 100, "records_unhashed": 0, "hash_ns_total": 20_000_000}}
    d = serve_stream_counters.stream_counters_delta(before, after, tokens=9, phase="decode")
    io = d["io"]
    assert io["records_hashed"] == 90
    assert io["records_unhashed"] == 0
    assert io["hash_fraction"] == 1.0
    assert io["hash_ms"] == pytest.approx(18.0, abs=1e-6)  # 18,000,000 ns


def test_stream_counters_delta_reports_hashing_off_window():
    before = {"io": {"records_hashed": 0, "records_unhashed": 10, "hash_ns_total": 0}}
    after = {"io": {"records_hashed": 0, "records_unhashed": 100, "hash_ns_total": 0}}
    d = serve_stream_counters.stream_counters_delta(before, after, tokens=9, phase="decode")
    io = d["io"]
    assert io["records_hashed"] == 0
    assert io["records_unhashed"] == 90
    assert io["hash_fraction"] == 0.0
    assert io["hash_ms"] == 0.0
