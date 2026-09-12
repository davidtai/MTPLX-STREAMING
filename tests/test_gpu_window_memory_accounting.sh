#!/usr/bin/env bash
# W106 hermetic test for gpu_window.sh's phase-4 process-TREE RSS accounting.
#
# The pre-W106 phase-4 loop polled `ps -o rss= -p STEP_PID` where STEP_PID is the
# `bash -c "..."` chain the wrapper launches, not the python benchmark underneath
# it, so every window logged "peak step RSS 0.0 GiB" and the child RSS cap could
# never fire.  This test runs the REAL wrapper in GPU_WINDOW_TEST_MODE=1 (which
# skips phases 1-3 + restore and uses a temp lock -- NO sysctl, NO launchctl, NO
# real GPU lock, NO Metal) against a FAKE step: a `bash -c` chain that launches a
# python child which allocates ~500 MB and sleeps.  It asserts the exit summary
# reports the whole-tree RSS (the ~0.5 GiB python child), not the few-MB bash
# parent, plus the 30s-cadence memory sample line.
#
#   nice -n 19 bash tests/test_gpu_window_memory_accounting.sh
#
# Exit 0 = all assertions pass; exit 1 = at least one failed.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${HERE}/../scripts/deepseek_v41/gpu_window.sh"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); printf 'ok   - %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf 'FAIL - %s\n     %s\n' "$1" "$2"; }

# --- a fake vm_stat: (wired + anonymous + occupied-compressor) = 50 GiB ---------
# Deterministic and well under the 105 GiB ceiling, so the system-wide guard never
# trips regardless of the box's real memory state while this test runs.
#   wired 1500000 + anon 1500000 + comp 276800 = 3276800 pages * 16384 = 50 GiB
FAKE_VMSTAT="${TMP}/vm_stat_50gib"
cat > "${FAKE_VMSTAT}" <<'EOF'
#!/bin/bash
cat <<'V'
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  100000.
Anonymous pages:                            1500000.
Pages inactive:                              200000.
Pages wired down:                           1500000.
Pages occupied by compressor:                276800.
V
EOF
chmod +x "${FAKE_VMSTAT}"

# --- the python child the fake step launches (~500 MB resident, then sleeps) ----
CHILD_PY="${TMP}/mem_child.py"
cat > "${CHILD_PY}" <<'EOF'
import sys, time
n = 500 * 1024 * 1024
buf = bytearray(n)
# Touch one byte per 4 KiB page so the pages are actually resident (RSS reflects
# the whole allocation, not a lazily-committed mapping).
for i in range(0, n, 4096):
    buf[i] = 1
sys.stderr.write("child resident ~500MB, sleeping\n")
sys.stderr.flush()
time.sleep(6)
EOF

# The step is a bash -c chain (STEP_PID = this bash) that runs python as a CHILD
# and waits on it -- exactly the shape that fooled the pre-W106 single-pid poll.
STEP_CHAIN="python3 '${CHILD_PY}' & _cp=\$!; wait \$_cp"

LOCK="${TMP}/testmode.lock"
LOG="${TMP}/window.log"

GPU_WINDOW_TEST_MODE=1 \
MTPLX_GPU_LOCK="${LOCK}" \
GPU_WINDOW_VM_STAT_CMD="${FAKE_VMSTAT}" \
GPU_WINDOW_FOREIGN_WORKER_RSS_GB=100000 \
GPU_WINDOW_RSS_POLL_SECONDS=1 \
  bash "${SCRIPT}" bash -c "${STEP_CHAIN}" >"${LOG}" 2>&1
RC=$?

echo "----- gpu_window.sh log (TEST MODE) -----"
cat "${LOG}"
echo "-----------------------------------------"

# 1. clean exit (the fake step exits 0)
if [[ "${RC}" -eq 0 ]]; then
  ok "wrapper exits 0 on a clean fake step"
else
  bad "wrapper exits 0 on a clean fake step" "exit code was ${RC}"
fi

# 2. TEST MODE actually skipped phases 1-3 (no sysctl / launchctl touched)
if grep -q "TEST MODE: skipping phases 1-3" "${LOG}"; then
  ok "TEST MODE skipped phases 1-3 + restore"
else
  bad "TEST MODE skipped phases 1-3 + restore" "skip line not found"
fi

# 3. the exit summary reports the whole-tree peak RSS, not the bash-parent-only 0.0
PEAK_LINE="$(grep 'peak step tree RSS' "${LOG}" | tail -1)"
if [[ -n "${PEAK_LINE}" ]]; then
  ok "exit summary has a 'peak step tree RSS' line"
  echo "     -> ${PEAK_LINE}"
else
  bad "exit summary has a 'peak step tree RSS' line" "line not found"
fi

# Extract the GiB number after 'peak step tree RSS' and assert it captured the
# ~0.5 GiB python child (>= 0.3 GiB); a single-pid poll of the bash chain would be
# ~0.0 GiB, so this is the direct proof the tree walk works.
PEAK_GIB="$(printf '%s\n' "${PEAK_LINE}" | sed -n 's/.*peak step tree RSS \([0-9.]*\) GiB.*/\1/p')"
if [[ -n "${PEAK_GIB}" ]] && awk -v v="${PEAK_GIB}" 'BEGIN{exit !(v+0 >= 0.3)}'; then
  ok "peak step tree RSS captured the python child (${PEAK_GIB} GiB >= 0.3)"
else
  bad "peak step tree RSS captured the python child (>= 0.3 GiB)" "parsed '${PEAK_GIB}' GiB"
fi

# 4. the max-single-process figure is present in the same summary
if [[ "${PEAK_LINE}" == *"max single process"* ]]; then
  ok "exit summary reports the max single-process RSS"
else
  bad "exit summary reports the max single-process RSS" "phrase not found"
fi

# 5. peak system used is the running max (50 GiB from the fake vm_stat), not 0
SYS_GIB="$(printf '%s\n' "${PEAK_LINE}" | sed -n 's/.*peak system used \([0-9.]*\) GiB.*/\1/p')"
if [[ -n "${SYS_GIB}" ]] && awk -v v="${SYS_GIB}" 'BEGIN{exit !(v+0 >= 49.0)}'; then
  ok "peak system used is the running max (${SYS_GIB} GiB ~= 50 from fake vm_stat)"
else
  bad "peak system used is the running max (~50 GiB)" "parsed '${SYS_GIB}' GiB"
fi

# 6. a periodic memory-envelope sample line was logged during the step
if grep -q "phase 4: mem sample -- step tree RSS" "${LOG}"; then
  ok "logged a periodic 'mem sample' envelope line during the step"
else
  bad "logged a periodic 'mem sample' envelope line during the step" "no sample line"
fi

# 7. it never touched the real exclusive lock
if [[ "$(grep -c 'acquired exclusive GPU lock' "${LOG}")" -ge 1 ]] && grep -q "${LOCK}" "${LOG}"; then
  ok "used the temp lock path, not the real /tmp/mtplx-gpu-exclusive.lock"
else
  bad "used the temp lock path" "lock line did not reference ${LOCK}"
fi

# =============================================================================
# W106 item 4: TREE-KILL on abort.  A fake step is a `bash -c` chain that launches
# a python child which spawns a long-lived `sleep` GRANDCHILD, then allocates
# enough to trip a (low) child-tree RSS cap so the wrapper aborts.  After the
# abort NO descendant may survive: pre-W106 the abort killed only STEP_PID (the
# `bash -c` chain), orphaning the python + sleep (reparented to launchd) and
# letting a chained next step still run.  The tree-kill must reap the whole tree.
# =============================================================================
TK_PIDFILE="${TMP}/treekill_pids"
TK_CHILD_PY="${TMP}/treekill_child.py"
cat > "${TK_CHILD_PY}" <<'EOF'
import os, subprocess, sys, time
# A long-lived grandchild -- the reparent target that pre-W106 survived the abort.
sleeper = subprocess.Popen(["sleep", "600"])
with open(os.environ["TK_PIDFILE"], "w") as fh:
    fh.write("%d %d\n" % (os.getpid(), sleeper.pid))
    fh.flush()
    os.fsync(fh.fileno())
# ~500 MB resident so the whole-tree RSS trips the low child cap and aborts.
n = 500 * 1024 * 1024
buf = bytearray(n)
for i in range(0, n, 4096):
    buf[i] = 1
sys.stderr.write("treekill child resident ~500MB, sleeping\n")
sys.stderr.flush()
time.sleep(600)
EOF

# STEP_PID = this bash -c; python is its child; sleep is the grandchild.
TK_STEP="python3 '${TK_CHILD_PY}' & _cp=\$!; wait \$_cp"
TK_LOG="${TMP}/treekill.log"

GPU_WINDOW_TEST_MODE=1 \
MTPLX_GPU_LOCK="${TMP}/treekill.lock" \
GPU_WINDOW_VM_STAT_CMD="${FAKE_VMSTAT}" \
GPU_WINDOW_FOREIGN_WORKER_RSS_GB=100000 \
GPU_WINDOW_RSS_POLL_SECONDS=1 \
GPU_WINDOW_CHILD_RSS_CAP_BYTES=$(( 200 * 1024 * 1024 )) \
GPU_WINDOW_KILL_GRACE_SECONDS=1 \
TK_PIDFILE="${TK_PIDFILE}" \
  bash "${SCRIPT}" bash -c "${TK_STEP}" >"${TK_LOG}" 2>&1
TK_RC=$?

echo "----- gpu_window.sh log (TREE-KILL) -----"
cat "${TK_LOG}"
echo "-----------------------------------------"

# 8. the RSS-cap abort fired (exit 6) and killed the child
if [[ "${TK_RC}" -eq 6 ]]; then
  ok "tree-kill scenario aborted on the child-tree RSS cap (exit 6)"
else
  bad "tree-kill scenario aborted on the RSS cap (exit 6)" "exit code was ${TK_RC}"
fi

# 9. the fake step recorded its descendant pids
TK_PY=""; TK_SLEEP=""
if [[ -s "${TK_PIDFILE}" ]]; then
  read -r TK_PY TK_SLEEP < "${TK_PIDFILE}"
  ok "fake step recorded its descendant pids (python=${TK_PY} sleep=${TK_SLEEP})"
else
  bad "fake step recorded its descendant pids" "pidfile empty: ${TK_PIDFILE}"
fi

# Poll up to 5 s for a pid to be gone (KILL is async; give it a moment).
_tk_wait_dead() {
  local pid="$1" i
  [[ -n "${pid}" ]] || return 1
  for (( i = 0; i < 20; i++ )); do
    kill -0 "${pid}" 2>/dev/null || return 0
    sleep 0.25
  done
  return 1
}

# 10. the python child (the reparented middle of the tree) is dead
if _tk_wait_dead "${TK_PY}"; then
  ok "python child (pid ${TK_PY}) was tree-killed on abort"
else
  bad "python child tree-killed on abort" "pid ${TK_PY} still alive after abort"
  kill -KILL "${TK_PY}" 2>/dev/null || true
fi

# 11. the sleep GRANDCHILD (the pre-W106 orphan) is dead -- the core of the fix
if _tk_wait_dead "${TK_SLEEP}"; then
  ok "sleep grandchild (pid ${TK_SLEEP}) was tree-killed on abort"
else
  bad "sleep grandchild tree-killed on abort (the pre-W106 orphan)" "pid ${TK_SLEEP} still alive"
  kill -KILL "${TK_SLEEP}" 2>/dev/null || true
fi

# 12. the log shows the tree-RSS-cap abort path (not a single-pid poll)
if grep -q "step tree RSS.*exceeded cap" "${TK_LOG}"; then
  ok "abort log names the step-TREE RSS cap breach"
else
  bad "abort log names the step-TREE RSS cap breach" "cap line not found"
fi

printf '\n%d passed, %d failed\n' "${PASS}" "${FAIL}"
[[ "${FAIL}" -eq 0 ]]
