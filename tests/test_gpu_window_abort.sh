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

printf '\n%d passed, %d failed\n' "${PASS}" "${FAIL}"
[[ "${FAIL}" -eq 0 ]]
