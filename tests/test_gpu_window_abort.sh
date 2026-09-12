#!/usr/bin/env bash
# W106 hermetic test for gpu_window.sh's ABORT path (real-window incident, window
# 42): a TERM to the bash gpu_window.sh must abort PROMPTLY while it is in the
# phase-4 poll loop (bash defers a heavy trap during `sleep`/`wait`), and the whole
# step process tree -- INCLUDING a python descendant reparented to launchd BEFORE
# the tree snapshot -- must be killed (no orphan loading outside the lock).
#
# GPU_WINDOW_TEST_MODE=1 + a temp lock: NO sysctl, NO launchctl, NO real GPU lock,
# NO Metal.
#
#   nice -n 19 bash tests/test_gpu_window_abort.sh
#
# Exit 0 = all assertions pass; exit 1 = at least one failed.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${HERE}/../scripts/deepseek_v41/gpu_window.sh"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

# HARD SAFETY NET: never the real lock / launchctl.
export GPU_WINDOW_TEST_MODE=1
export MTPLX_GPU_LOCK="${TMP}/hermetic.lock"
export GPU_WINDOW_LOCK_TIMEOUT=20
export GPU_WINDOW_VM_STAT_CMD="${TMP}/fake_vmstat"
export GPU_WINDOW_FOREIGN_WORKER_RSS_GB=100000
export GPU_WINDOW_RSS_POLL_SECONDS=1
export GPU_WINDOW_KILL_GRACE_SECONDS=1

PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); printf 'ok   - %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf 'FAIL - %s\n     %s\n' "$1" "$2"; }

# fake vm_stat: 50 GiB used, well under the ceiling (no ceiling abort).
cat > "${GPU_WINDOW_VM_STAT_CMD}" <<'EOF'
#!/bin/bash
cat <<'V'
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  100000.
Anonymous pages:                            1500000.
Pages wired down:                           1500000.
Pages occupied by compressor:                276800.
V
EOF
chmod +x "${GPU_WINDOW_VM_STAT_CMD}"

# The step: a python child that -- AFTER 1 s (so the fork races the first tree
# snapshot) -- spawns a `sleep` GRANDCHILD, writes both pids, and then the child
# re-parents the grandchild by design (the grandchild outlives an early chain
# death). Both must be dead after the abort.
CHILD_PY="${TMP}/abort_child.py"
cat > "${CHILD_PY}" <<'EOF'
import os, subprocess, sys, time
time.sleep(1.0)  # fork the grandchild AFTER the wrapper's first snapshot
sleeper = subprocess.Popen(["sleep", "600"])
with open(os.environ["ABORT_PIDFILE"], "w") as fh:
    fh.write("%d %d\n" % (os.getpid(), sleeper.pid))
    fh.flush(); os.fsync(fh.fileno())
sys.stderr.write("abort child + grandchild up\n"); sys.stderr.flush()
time.sleep(600)
EOF

ABORT_PIDFILE="${TMP}/abort_pids"
export ABORT_PIDFILE
STEP="python3 '${CHILD_PY}' & _c=\$!; wait \$_c"
LOG="${TMP}/abort.log"

# Launch the wrapper in the background. The launched pid is the python fcntl
# lock-holder (it execs into python and IGNORES INT/TERM); its child is the bash
# gpu_window.sh that OWNS the trap -- that is the pid we must TERM.
bash "${SCRIPT}" bash -c "${STEP}" >"${LOG}" 2>&1 &
HOLDER=$!

# Wait for the step to be up (pidfile written) -> the poll loop is active.
for _ in $(seq 1 60); do [[ -s "${ABORT_PIDFILE}" ]] && break; sleep 0.25; done
if [[ -s "${ABORT_PIDFILE}" ]]; then
  read -r AB_PY AB_SLEEP < "${ABORT_PIDFILE}"
  ok "step + grandchild started (python=${AB_PY} sleep=${AB_SLEEP})"
else
  bad "step started" "pidfile never written"; AB_PY=""; AB_SLEEP=""
fi

# Find the bash gpu_window.sh child of the holder (the trap owner).
BASH_WRAPPER=""
for _ in $(seq 1 20); do
  BASH_WRAPPER="$(ps -axo pid=,ppid=,command= | awk -v h="${HOLDER}" '$2==h && /gpu_window\.sh/ {print $1; exit}')"
  [[ -n "${BASH_WRAPPER}" ]] && break
  sleep 0.25
done
if [[ -n "${BASH_WRAPPER}" ]]; then
  ok "found the bash gpu_window.sh trap owner (pid=${BASH_WRAPPER})"
else
  bad "found the trap-owner bash" "no gpu_window.sh child of holder ${HOLDER}"
fi

# The log must print the abort recipe naming that pid.
if grep -q "abort: to abort this window cleanly, kill -TERM ${BASH_WRAPPER}" "${LOG}"; then
  ok "startup log printed the abort recipe naming the trap-owner pid"
else
  bad "startup abort-recipe log" "$(grep 'abort: to abort' "${LOG}" || echo 'no abort-recipe line')"
fi

# --- ABORT: TERM the trap owner while it is in the poll loop ---
T0=$(date +%s)
kill -TERM "${BASH_WRAPPER}" 2>/dev/null || true

# The wrapper must EXIT promptly (not hang for 12 s+). Poll the holder.
EXITED=0
for _ in $(seq 1 40); do   # up to 10 s
  kill -0 "${HOLDER}" 2>/dev/null || { EXITED=1; break; }
  sleep 0.25
done
T1=$(date +%s)
if (( EXITED == 1 )); then
  ok "wrapper aborted promptly on TERM (~$((T1 - T0)) s, not a 12s+ hang)"
else
  bad "wrapper aborted promptly on TERM" "still alive after ~$((T1 - T0)) s"
  kill -KILL "${HOLDER}" 2>/dev/null || true
fi

echo "----- gpu_window.sh log (ABORT) -----"; cat "${LOG}"; echo "-------------------------------------"

# The abort path logged the signal + tree-kill.
if grep -q "abort requested (INT/TERM); killing the step tree" "${LOG}"; then
  ok "abort path fired from the poll loop (flag-checked, not deferred)"
else
  bad "abort path fired" "$(grep -i abort "${LOG}" | tail -3)"
fi

# Poll up to 5 s for BOTH descendants to be dead (tree-kill on abort).
_wait_dead() { local p="$1" i; [[ -n "$p" ]] || return 1; for (( i=0;i<20;i++ )); do kill -0 "$p" 2>/dev/null || return 0; sleep 0.25; done; return 1; }
if _wait_dead "${AB_PY}"; then ok "python step killed on abort (pid ${AB_PY})"; else bad "python step killed on abort" "pid ${AB_PY} alive"; kill -KILL "${AB_PY}" 2>/dev/null||true; fi
if _wait_dead "${AB_SLEEP}"; then ok "sleep grandchild killed on abort (pid ${AB_SLEEP})"; else bad "sleep grandchild killed on abort" "pid ${AB_SLEEP} alive"; kill -KILL "${AB_SLEEP}" 2>/dev/null||true; fi

# =============================================================================
# W106 abort item (b): a python REPARENTED to launchd (its `bash -c` chain exits
# first) must still be reaped -- window 42's 27 GB python kept loading experts.bin
# outside the lock.  The ppid walk misses it (ppid=1); the env-TAG scan catches it.
# Here the step's bash -c backgrounds+disowns the python and exits 0 (NORMAL exit):
# on teardown the wrapper must reap the tagged orphan before releasing the lock.
# =============================================================================
ORPHAN_PY="${TMP}/orphan_child.py"
cat > "${ORPHAN_PY}" <<'EOF'
import os, sys, time
with open(os.environ["ORPHAN_PIDFILE"], "w") as fh:
    fh.write("%d\n" % os.getpid()); fh.flush(); os.fsync(fh.fileno())
sys.stderr.write("orphan up (ppid=%d, tag=%s)\n" % (os.getppid(), os.environ.get("_GPU_WINDOW_STEP_TAG",""))); sys.stderr.flush()
time.sleep(600)
EOF
ORPHAN_PIDFILE="${TMP}/orphan_pid"; export ORPHAN_PIDFILE
# bash -c: background + disown the python, then EXIT 0 -> python reparents to launchd.
ORPHAN_STEP="python3 '${ORPHAN_PY}' & disown; exit 0"
OLOG="${TMP}/orphan.log"

bash "${SCRIPT}" bash -c "${ORPHAN_STEP}" >"${OLOG}" 2>&1 &
OHOLDER=$!
for _ in $(seq 1 60); do [[ -s "${ORPHAN_PIDFILE}" ]] && break; sleep 0.25; done
ORPHAN_PID="$(cat "${ORPHAN_PIDFILE}" 2>/dev/null | tr -d ' \n')"
if [[ -n "${ORPHAN_PID}" ]]; then ok "orphan python started (pid=${ORPHAN_PID})"; else bad "orphan started" "no pidfile"; fi

# Wait for the wrapper to finish (the step exits 0 quickly; teardown then reaps).
for _ in $(seq 1 60); do kill -0 "${OHOLDER}" 2>/dev/null || break; sleep 0.25; done

echo "----- gpu_window.sh log (ORPHAN reap) -----"; cat "${OLOG}"; echo "-------------------------------------------"

# The reparented python (ppid=1 after the chain exits) must be DEAD -- only the
# env-tag scan can find it, so this is the direct proof of the (b) fix.
if _wait_dead "${ORPHAN_PID}"; then
  ok "reparented orphan python was reaped via the env tag (pid ${ORPHAN_PID})"
else
  bad "reparented orphan reaped" "pid ${ORPHAN_PID} STILL ALIVE outside the lock"
  kill -KILL "${ORPHAN_PID}" 2>/dev/null || true
fi
# It reparented for real (ppid became 1) -- the ppid walk would have missed it.
if grep -q "orphan up (ppid=1" "${OLOG}"; then
  ok "orphan really reparented to launchd (ppid=1) -- ppid walk would miss it"
else
  ok "orphan reap verified (ppid line not captured, non-fatal)"
fi

# The abort-a run (a real tree-kill of live pids) must produce NO false ORPHAN
# lines once the tree is dead (MEDIUM-1: the rescan re-confirms kill -0).
if ! grep -q "ORPHAN survived tree-kill" "${LOG}"; then
  ok "MEDIUM-1: abort-a killed the tree cleanly, no false ORPHAN lines"
else
  bad "MEDIUM-1 no false ORPHAN on abort-a" "$(grep 'ORPHAN survived' "${LOG}")"
fi

# =============================================================================
# W106 MEDIUM-1: _step_tree_pids must emit ONLY present pids (an absent root prints
# nothing), and a DEAD step (clean fast exit) must be reaped with ZERO false ORPHAN
# lines and no KILL of a dead pid.
# =============================================================================
if [[ -z "$(bash "${SCRIPT}" --selftest tree-pids 999999 | tr -d ' \n')" ]]; then
  ok "MEDIUM-1: --selftest tree-pids 999999 (absent root) prints nothing"
else
  bad "MEDIUM-1 absent root not emitted" "got '$(bash "${SCRIPT}" --selftest tree-pids 999999)'"
fi

DEAD_LOG="${TMP}/dead.log"
GPU_WINDOW_TEST_MODE=1 MTPLX_GPU_LOCK="${TMP}/dead.lock" \
GPU_WINDOW_VM_STAT_CMD="${GPU_WINDOW_VM_STAT_CMD}" GPU_WINDOW_FOREIGN_WORKER_RSS_GB=100000 \
  bash "${SCRIPT}" bash -c "true" >"${DEAD_LOG}" 2>&1
DEAD_RC=$?
if [[ "${DEAD_RC}" -eq 0 ]] && ! grep -q "ORPHAN survived tree-kill" "${DEAD_LOG}"; then
  ok "MEDIUM-1: a dead step is reaped with ZERO false ORPHAN lines (rc 0)"
else
  bad "MEDIUM-1 dead-step zero ORPHAN" "rc=${DEAD_RC}; $(grep 'ORPHAN' "${DEAD_LOG}" || echo none)"
fi

# W106 LOW (round 4): teardown must IGNORE (not default) INT/TERM so a second
# signal mid-restore cannot kill it and leave the agent down.  Deterministic source
# guard for the exact disposition (a behaviour race-test would be flaky).
if grep -q "trap '' INT TERM" "${SCRIPT}"; then
  ok "LOW: teardown ignores further INT/TERM (trap '' INT TERM), not reset to default"
else
  bad "LOW teardown trap ''" "no \"trap '' INT TERM\" in gpu_window.sh"
fi

printf '\n%d passed, %d failed\n' "${PASS}" "${FAIL}"
[[ "${FAIL}" -eq 0 ]]
