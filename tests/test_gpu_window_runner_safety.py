"""Hermetic guard regressions: stdlib only, fake readers/service, no MLX.

Run directly with Python; no process allocates model-sized memory and every
service command is replaced before the guard can execute.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts/deepseek_v41/gpu_window.sh"
TREE = ROOT / "scripts/deepseek_v41/tree_footprint.py"
VALID_VM = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages wired down: 10.
Pages active: 100.
Pages inactive: 200.
Anonymous pages: 20.
Pages occupied by compressor: 30.
"""


class GuardSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.env = dict(os.environ)
        self.env.pop("GPU_WINDOW_TOTAL_MEM_CEILING_GB", None)
        self.env.pop("GPU_WINDOW_TOTAL_MEM_CEILING_BYTES", None)
        self.env.update({
            "GPU_WINDOW_TEST_MODE": "1",
            "MTPLX_GPU_LOCK": str(self.path / "exclusive.lock"),
            "GPU_WINDOW_LOCK_TIMEOUT": "2",
            "GPU_WINDOW_RESTORE_TIMEOUT": "1",
            "GPU_WINDOW_FOREIGN_WORKER_RSS_GB": "100000",
            "GPU_WINDOW_VM_STAT_CMD": self.command("vm", "cat <<'VM'\n" + VALID_VM + "VM\n"),
            "GPU_WINDOW_COMPRESSOR_CMD": self.command("comp", "echo 0\n"),
            "GPU_WINDOW_FOOTPRINT_READER": self.command("fp", "echo 1048576\n"),
            "GPU_WINDOW_LAUNCHCTL_CMD": self.command("launchctl", "exit 0\n"),
            "GPU_WINDOW_CURL_CMD": self.command("curl", 'case "${!#}" in */v1/models) cat "$FAKE_MODELS";; *) cat "$FAKE_HEALTH";; esac\n'),
            "GPU_WINDOW_HEALTH_URL": "http://fake.invalid/health",
            "GPU_WINDOW_MODELS_URL": "http://fake.invalid/v1/models",
        })
        health = self.path / "health.json"
        health.write_text(json.dumps({"ok": True, "startup": {"warmup": {"background": {"state": "done"}}}}))
        self.env["FAKE_HEALTH"] = str(health)
        models = self.path / "models.json"
        models.write_text('{"data": [{"id": "original"}]}')
        self.env["FAKE_MODELS"] = str(models)

    def command(self, name, body):
        path = self.path / name
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)
        return str(path)

    def run_guard(self, *args):
        return subprocess.run(["/bin/bash", str(GUARD), *args], env=self.env,
                              capture_output=True, text=True, timeout=8)

    def configured_ceiling(self):
        # Configuration prefix only: no lock, service, sysctl, or child launch.
        prefix = GUARD.read_text().split('UID_NUM="$(id -u)"', 1)[0]
        return subprocess.run(["/bin/bash", "-c", prefix + '\nprintf "%s" "$TOTAL_MEM_CEILING_BYTES"'],
                              env=self.env, capture_output=True, text=True, timeout=3)

    def test_default_ceiling_is_exactly_110_decimal_gb(self):
        result = self.configured_ceiling()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(int(result.stdout), 110_000_000_000)

    def test_explicit_legacy_ceiling_remains_gib(self):
        self.env["GPU_WINDOW_TOTAL_MEM_CEILING_GB"] = "100"
        self.assertEqual(int(self.configured_ceiling().stdout), 100 * 1024**3)

    def test_exact_byte_ceiling_override(self):
        self.env["GPU_WINDOW_TOTAL_MEM_CEILING_BYTES"] = "109000000000"
        self.assertEqual(int(self.configured_ceiling().stdout), 109_000_000_000)

    def test_ambiguous_ceiling_overrides_are_rejected(self):
        self.env["GPU_WINDOW_TOTAL_MEM_CEILING_GB"] = "100"
        self.env["GPU_WINDOW_TOTAL_MEM_CEILING_BYTES"] = "109000000000"
        self.assertNotEqual(self.configured_ceiling().returncode, 0)

    def test_physical_used_includes_active_and_inactive_file_pages(self):
        result = self.run_guard("--selftest", "used-mem-bytes")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(int(result.stdout), (10 + 100 + 200 + 30) * 16384)

    def test_capture_actual_model_directory(self):
        Path(self.env['FAKE_HEALTH']).write_text(json.dumps({'model_path': str(self.path)}))
        result = self.run_guard('--selftest', 'model-path')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(self.path.resolve()))

    def test_reject_missing_or_ambiguous_model_directory(self):
        for value in (None, '', 'relative/model', '/tmp/model\nwrong'):
            Path(self.env['FAKE_HEALTH']).write_text(json.dumps({'model_path': value}))
            result = self.run_guard('--selftest', 'model-path')
            self.assertNotEqual(result.returncode, 0)

    def test_automatic_reclamation_precedes_fresh_baseline(self):
        source = GUARD.read_text()
        start = source.index('# ---------------- phase 3:')
        call = source.index('if ! _reclaim_qwen_file_cache;', start)
        baseline = source.index('if ! USED_START=', start)
        self.assertLess(start, call)
        self.assertLess(call, baseline)
        self.assertIn('QWEN_PROCESS_IDS', source[start:call])

    def test_failed_reclamation_refuses_workload(self):
        source = GUARD.read_text()
        start = source.index('_reclaim_qwen_file_cache() {')
        end = source.index('\n_restored_api_ready()', start)
        helper = self.path / 'reclaim_file_cache.py'
        helper.write_text('raise SystemExit(17)\n')
        script = 'SCRIPT_PATH="$FAKE_SCRIPT"; QWEN_MODEL_PATH=/tmp/model; TOTAL_MEM_CEILING_BYTES=110000000000\n'
        script += 'used_mem_bytes() { echo 1000; }; log() { :; }; err() { :; }; _check_abort() { :; };\n'
        script += source[start:end]
        script += '\nif ! _reclaim_qwen_file_cache; then exit 8; fi\nprintf WORKLOAD_STARTED\n'
        env = dict(self.env, FAKE_SCRIPT=str(self.path / 'guard.sh'))
        result = subprocess.run(['/bin/bash', '-c', script], env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 8, result.stderr)
        self.assertNotIn('WORKLOAD_STARTED', result.stdout)

    def test_registered_but_unhealthy_service_does_not_restore(self):
        Path(self.env["FAKE_HEALTH"]).write_text('{"ok": false}')
        result = self.run_guard("--selftest", "restore-run", "1", "")
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_optimized_python_cannot_disable_restore_health_validation(self):
        self.env["PYTHONOPTIMIZE"] = "1"
        Path(self.env["FAKE_HEALTH"]).write_text('{"ok": false}')
        result = self.run_guard("--selftest", "restore-run", "1", "", '["original"]')
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_restore_retries_until_matching_service_finishes_warmup(self):
        self.env["GPU_WINDOW_RESTORE_TIMEOUT"] = "3"
        self.env["FAKE_HEALTH_PROBED"] = str(self.path / "health_probed")
        self.env["GPU_WINDOW_CURL_CMD"] = self.command("retry_curl", '''
case "${!#}" in
  */v1/models) cat "$FAKE_MODELS" ;;
  *) if [[ ! -f "$FAKE_HEALTH_PROBED" ]]; then
       touch "$FAKE_HEALTH_PROBED"; echo '{"ok":false}'
     else cat "$FAKE_HEALTH"; fi ;;
esac
''')
        result = self.run_guard("--selftest", "restore-run", "1", "", '["original"]')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(Path(self.env["FAKE_HEALTH_PROBED"]).exists())

    def test_registered_but_warming_service_does_not_restore(self):
        Path(self.env["FAKE_HEALTH"]).write_text(json.dumps({
            "ok": True, "startup": {"warmup": {"background": {"state": "running"}}}}))
        result = self.run_guard("--selftest", "restore-run", "1", "")
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_teardown_propagates_restore_failure(self):
        source = GUARD.read_text()
        start = source.index("teardown() {")
        end = source.index("\n# W106 abort item (a)", start)
        script = "STEP_PID=''; _STEP_TAG=''; restore_qwen() { return 17; }; err() { :; }; _step_children_exited() { return 0; };\n"
        script += source[start:end] + "\ntrap teardown EXIT\nexit 0\n"
        result = subprocess.run(["/bin/bash", "-c", script], env=self.env,
                                capture_output=True, text=True, timeout=3)
        self.assertNotEqual(result.returncode, 0)

    def test_teardown_does_not_restore_over_surviving_qwen_descendant(self):
        source = GUARD.read_text()
        start = source.index('teardown() {')
        end = source.index('\n# W106 abort item (a)', start)
        script = "STEP_PID=''; _STEP_TAG=''; QWEN_STOP_REQUESTED=1; QWEN_PROCESS_IDS=$$;\n"
        script += 'restore_qwen() { printf UNSAFE_BOOTSTRAP; }; err() { printf "%s\\n" "$*"; }; _step_children_exited() { return 0; };\n'
        script += source[start:end] + '\ntrap teardown EXIT\nexit 5\n'
        result = subprocess.run(['/bin/bash', '-c', script], env=self.env,
                                capture_output=True, text=True, timeout=3)
        self.assertEqual(result.returncode, 10)
        self.assertIn('RESTORE_FAILED', result.stdout)
        self.assertNotIn('UNSAFE_BOOTSTRAP', result.stdout)

    def test_healthy_wrong_model_does_not_restore(self):
        result = self.run_guard("--selftest", "restore-run", "1", "", '["expected"]')
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_healthy_matching_model_restores(self):
        result = self.run_guard("--selftest", "restore-run", "1", "", '["original"]')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_teardown_does_not_restore_over_a_live_owned_child(self):
        source = GUARD.read_text()
        start = source.index("teardown() {")
        end = source.index("\n# W106 abort item (a)", start)
        script = "STEP_PID=''; _STEP_TAG='owned'; _kill_step_tree() { :; }; err() { :; };\n"
        script += "_step_children_exited() { return 1; }; restore_qwen() { echo BOOTSTRAPPED; };\n"
        script += source[start:end] + "\ntrap teardown EXIT\nexit 0\n"
        result = subprocess.run(["/bin/bash", "-c", script], env=self.env,
                                capture_output=True, text=True, timeout=3)
        self.assertNotIn("BOOTSTRAPPED", result.stdout)
        self.assertNotEqual(result.returncode, 0)

    def test_final_check_retains_children_known_before_reparenting(self):
        source = GUARD.read_text()
        start = source.index("_step_children_exited() {")
        end = source.index("\n_kill_step_tree()", start)
        self.env["GPU_WINDOW_PS_CMD"] = self.command("ps", "echo S\n")
        script = 'LAST_STEP_PIDS="$$"; PS_CMD="$GPU_WINDOW_PS_CMD"; _collect_step_pids() { :; };\n'
        script += source[start:end] + "\n_step_children_exited\n"
        result = subprocess.run(["/bin/bash", "-c", script], env=self.env,
                                capture_output=True, text=True, timeout=3)
        self.assertNotEqual(result.returncode, 0)

    def test_invalid_vm_observation_refuses_to_start_step(self):
        for name, body in (("empty", "exit 0\n"), ("garbage", "echo garbage\n"),
                           ("missing", "cat <<'VM'\n" + VALID_VM.replace("Pages active: 100.\n", "") + "VM\n"),
                           ("missing_inactive", "cat <<'VM'\n" + VALID_VM.replace("Pages inactive: 200.\n", "") + "VM\n"),
                           ("status", "cat <<'VM'\n" + VALID_VM + "VM\nexit 2\n")):
            with self.subTest(name=name):
                marker = self.path / (name + ".started")
                self.env["GPU_WINDOW_VM_STAT_CMD"] = self.command("bad_vm", body)
                result = self.run_guard("/usr/bin/touch", str(marker))
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse(marker.exists(), "step started without a valid baseline")

    def test_invalid_compressor_observation_refuses_to_start_step(self):
        self.env["GPU_WINDOW_COMPRESSOR_CMD"] = self.command("bad_comp", "exit 2\n")
        marker = self.path / "comp.started"
        result = self.run_guard("/usr/bin/touch", str(marker))
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(marker.exists())

    def test_failed_foreign_worker_scan_refuses_to_start_step(self):
        for body in ("exit 2\n", "echo garbage\n"):
            with self.subTest(body=body):
                self.env["GPU_WINDOW_PS_CMD"] = self.command("bad_ps", body)
                marker = self.path / "ps.started"
                result = self.run_guard("/usr/bin/touch", str(marker))
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse(marker.exists())

    def test_unreadable_process_tree_is_not_an_empty_owned_set(self):
        for body in ("exit 2\n", "echo garbage\n"):
            with self.subTest(body=body):
                self.env["GPU_WINDOW_PS_CMD"] = self.command("bad_ps_tree", body)
                result = self.run_guard("--selftest", "tree-pids", "101")
                self.assertNotEqual(result.returncode, 0)

    def test_malformed_tag_snapshot_cannot_confirm_child_exit(self):
        source = GUARD.read_text()
        start = source.index("_pids_with_tag() {")
        end = source.index("\n# The FULL set", start)
        self.env["GPU_WINDOW_PS_CMD"] = self.command("bad_tag_ps", "echo garbage\n")
        script = '_STEP_TAG="owned"; PS_CMD="$GPU_WINDOW_PS_CMD";\n'
        script += source[start:end] + "\n_pids_with_tag\n"
        result = subprocess.run(["/bin/bash", "-c", script], env=self.env,
                                capture_output=True, text=True, timeout=3)
        self.assertNotEqual(result.returncode, 0)

    def test_unreadable_live_step_state_aborts_instead_of_waiting_unmonitored(self):
        self.env["GPU_WINDOW_PS_CMD"] = self.command(
            "bad_state", 'if [[ "$*" == "-o state="* ]]; then exit 2; fi\nexec /bin/ps "$@"\n')
        self.env["GPU_WINDOW_KILL_GRACE_SECONDS"] = "0"
        marker = self.path / "unmonitored.finished"
        result = self.run_guard("/bin/bash", "-c", 'sleep 3; touch "$1"', "guard-child", str(marker))
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(marker.exists(), "step completed after its live state became unreadable")

    def exiting_state_reader(self):
        # Darwin ps can report ?E while task-thread inspection is unavailable
        # during exit. Keep the real zombie/gone transition for the final reap.
        self.env["GPU_WINDOW_PS_CMD"] = self.command("exiting_state", '''
if [[ "$*" == "-o state="* ]]; then
    state="$(/bin/ps "$@")"; rc=$?
    if (( rc == 0 )) && [[ -n "$state" && "$state" != *Z* ]]; then
        echo '?E'
    else
        printf '%s\n' "$state"
    fi
    exit "$rc"
fi
exec /bin/ps "$@"
''')

    def test_darwin_exiting_state_preserves_child_exit_code(self):
        self.exiting_state_reader()
        result = self.run_guard("/bin/bash", "-c", "sleep .15; exit 7")
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertIn("GPU step exited with code 7", result.stdout + result.stderr)

    def test_darwin_exiting_state_still_checks_memory(self):
        self.exiting_state_reader()
        self.env["GPU_WINDOW_CHILD_RSS_CAP_BYTES"] = str(1024**3)
        self.env["GPU_WINDOW_FOOTPRINT_READER"] = self.command("over_cap", "echo 2147483648\n")
        self.env["GPU_WINDOW_KILL_GRACE_SECONDS"] = "0"
        marker = self.path / "over_cap.finished"
        result = self.run_guard("/bin/bash", "-c", 'sleep 3; touch "$1"', "guard-child", str(marker))
        self.assertEqual(result.returncode, 6, result.stdout + result.stderr)
        self.assertFalse(marker.exists())

    def test_baseline_at_or_over_ceiling_refuses_to_start_step(self):
        baseline = (10 + 100 + 200 + 30) * 16384
        for ceiling in (baseline, baseline - 1):
            with self.subTest(ceiling=ceiling):
                self.env["GPU_WINDOW_TOTAL_MEM_CEILING_BYTES"] = str(ceiling)
                marker = self.path / "unsafe_baseline.started"
                result = self.run_guard("/usr/bin/touch", str(marker))
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse(marker.exists(), "step launched with no physical-memory headroom")


class TreeSafetyTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("tree_guard_safety", TREE)
        self.tree = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.tree)

    def test_failed_ps_cannot_be_a_root_only_measurement(self):
        out = subprocess.CompletedProcess([], 2, stdout="", stderr="failed")
        with mock.patch.object(self.tree.subprocess, "run", return_value=out):
            with self.assertRaises(RuntimeError):
                self.tree._child_map()

    def test_malformed_ps_cannot_be_a_root_only_measurement(self):
        out = subprocess.CompletedProcess([], 0, stdout="garbage\n", stderr="")
        with mock.patch.object(self.tree.subprocess, "run", return_value=out):
            with self.assertRaises(RuntimeError):
                self.tree._child_map()

    def test_unreadable_live_child_is_not_omitted(self):
        with mock.patch.object(self.tree, "tree_pids", return_value=[101, 102]), \
             mock.patch.object(self.tree, "phys_footprint", side_effect=[100, None]), \
             mock.patch("os.kill", return_value=None):
            with self.assertRaises(RuntimeError):
                self.tree.tree_footprint([101])

    def test_exited_child_can_be_omitted(self):
        with mock.patch.object(self.tree, "tree_pids", return_value=[101, 102]), \
             mock.patch.object(self.tree, "phys_footprint", side_effect=[100, None]), \
             mock.patch("os.kill", side_effect=ProcessLookupError):
            self.assertEqual(self.tree.tree_footprint([101]), 100)


if __name__ == "__main__":
    unittest.main()
