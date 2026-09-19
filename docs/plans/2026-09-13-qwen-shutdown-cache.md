# Qwen shutdown cache implementation plan

Goal: automatically reclaim the stopped Qwen model's clean file cache before
the DeepSeek guard measures its allocation baseline.
Architecture: capture the service model path and process tree; after full stop,
run a bounded read-only helper; preserve fail-closed workload admission and restore.
Tech: Bash, Python stdlib, Darwin mmap/mincore/msync.
Assumes a flat safetensors model folder reported by service health. Reject symlink
files, malformed paths and uninspectable process state before reclamation.

- [x] Add failing CPU tests in `tests/test_qwen_shutdown_file_cache.py` for the
  real scratch-file digest and cache release, invalid roots/symlinks before any
  invalidation, syscall cleanup and failures. Run directly with Python unittest.
- [x] Implement `scripts/deepseek_v41/reclaim_file_cache.py` using the measured
  readonly mapping mechanism, 1 GiB windows, 512 files and 30-second alarm.
- [x] Extend `tests/test_gpu_window_runner_safety.py` with model-path parsing,
  automatic post-stop/pre-baseline ordering and failed-helper refusal. Observe
  failures, then wire `gpu_window.sh`; captured descendants must also be gone.
- [x] Run both CPU files and the existing shell guard regressions. Review the
  helper and lifecycle diff. Commit the verified source before GPU measurement.
- [x] Run the exact 16K Python workload under the normal guard, fanout 4 and
  110 GB. Use the reviewed larger-residency bound and 16 GiB transient reserve;
  compare all output IDs, retain memory samples and verify exact service restore.
