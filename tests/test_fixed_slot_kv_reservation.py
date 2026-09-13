"""KV admission must price allocated storage, not logical expert occupancy.

Real slot pools and file reads, with tiny bytearray backing and no MLX import.
"""

import pytest

from test_expert_slots_runtime import _artifact, _global_artifact
from mtplx.expert_manifest import save_expert_manifest
from mtplx.expert_runtime import (
    ExpertStreamingConfig,
    ExpertStreamingConfigurationError,
    ExpertStreamingRuntime,
)


@pytest.mark.parametrize("layout", ("direct-slots", "component-banks"))
@pytest.mark.parametrize("scope", ("layer", "global"))
def test_admission_accounts_for_retained_physical_slots(tmp_path, monkeypatch, layout, scope):
    monkeypatch.setenv("MTPLX_DSV41_RUNNER", "v2")
    root, spec, manifest, expected = _global_artifact(tmp_path, expert_count=4)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    # One byte short of funding four cached records plus the full KV reserve.
    # Logical eviction cannot make the original four backing arrays smaller.
    limit = (spec.resident_bytes + spec.transient_scratch_bytes
             + 4 * spec.expert_record_bytes + spec.kv_bytes_per_token - 1)
    buffers = []

    def allocate(size, label):
        buffer = bytearray(size)
        buffers.append(buffer)
        return buffer

    config = ExpertStreamingConfig(
        model_key=spec.key, memory_limit_bytes=limit, max_live_kv_tokens=1,
        runtime_reserve_bytes=0, verify_artifact_headers=False,
        slot_layout=layout, cache_scope=scope,
    )
    runtime = ExpertStreamingRuntime.open(
        root, manifest_path, config, spec=spec, apply_memory_cap=False,
        buffer_allocator=allocate,
    )
    try:
        ready = runtime.ensure_route(1, [0], phase="decode")
        try:
            before = tuple((id(buffer), len(buffer)) for buffer in buffers)
            admission = runtime.admit_kv_tokens(1)
            try:
                actual_owned = sum(len(buffer) for buffer in buffers)
                assert actual_owned + spec.resident_bytes + spec.kv_bytes_per_token <= limit
                assert runtime.plan.context_tokens == 1
                assert runtime.plan.kv_bytes == spec.kv_bytes_per_token
                assert tuple((id(buffer), len(buffer)) for buffer in buffers) == before
                assert bytes(ready.bindings[0].buffer) == expected[(1, 0)]
                with pytest.raises(ExpertStreamingConfigurationError, match="exceeds planned"):
                    runtime.admit_kv_tokens(1)
                assert runtime._live_kv_tokens == 1
            finally:
                admission.release()
            assert runtime._live_kv_tokens == 0
            assert tuple((id(buffer), len(buffer)) for buffer in buffers) == before
        finally:
            ready.release(synchronize=False)
    finally:
        runtime.close()


def test_insufficient_max_context_fails_before_slot_allocation(tmp_path):
    root, spec, manifest, _ = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=(spec.resident_bytes + spec.transient_scratch_bytes
                            + spec.kv_bytes_per_token - 1),
        max_live_kv_tokens=1, runtime_reserve_bytes=0, verify_artifact_headers=False,
    )

    def allocate(size, label):
        pytest.fail("fixed-footprint admission must precede backing allocation")

    with pytest.raises(ExpertStreamingConfigurationError, match="footprint exceeds"):
        ExpertStreamingRuntime.open(
            root, manifest_path, config, spec=spec, apply_memory_cap=False,
            buffer_allocator=allocate,
        )
