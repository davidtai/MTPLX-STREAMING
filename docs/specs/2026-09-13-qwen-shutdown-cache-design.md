# Automatic Qwen shutdown cache reclamation

User authorization: after the measured 33.75 GB reclamation experiment, the user
requested automatic reclamation when shutting down Qwen. Apply the verified
mechanism to the normal DeepSeek guarded shutdown, before its baseline sample.

The guard captures `model_path` from Qwen health before bootout. After the stopped
process tree is gone, it invokes a stdlib-only helper over that model directory's
flat safetensors files. Read-only descriptors, read-only shared mappings and
MS_SYNC|MS_INVALIDATE discard clean cached pages without editing model bytes.
The helper reports measured physical-used before/after and cached-page counts
separately. The existing parent lock, 110 GB ceiling and exact restore remain.

Risks and handling: reject a missing/non-model path or symlink before touching
cache; reject unreadable process ancestry before shutdown; wait for captured
descendants before reclamation; bound mappings to 1 GiB, file count to 512 and
execution to 30 seconds. Helper failure stops the workload and runs normal restore.
Do not derive a new budget from estimated reclaimable bytes: sample the real
baseline after the helper exits. No hot-path or production model changes.

This integrates the repository's `scripts/deepseek_v41/gpu_window.sh` shutdown
workflow. Other independent tools that directly issue `launchctl bootout` are
outside this repository change. A shutdown hook inside Qwen was considered but
would need to run after its own full process tree exits, when the launchd job
cannot reliably finish such a hook. The lock-owning guard supplies that boundary.

Verification: failing CPU regressions for automatic ordering/failure propagation,
path validation, unchanged scratch-file digest, and bounded mapping cleanup; full
existing guard regression bundle; independent safety review; guarded full-model
run with fresh baseline, exact token parity, sampled memory and restored health.
