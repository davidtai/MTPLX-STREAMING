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

# HARD SAFETY NET: a GPU window may be holding the REAL exclusive lock
# (/tmp/mtplx-gpu-exclusive.lock).  Export a hermetic temp lock + TEST MODE for the
# WHOLE test, so even a malformed per-invocation env line can NEVER fall back to the
# real lock and block on it.  Per-scenario MTPLX_GPU_LOCK= still overrides this.
export GPU_WINDOW_TEST_MODE=1
export MTPLX_GPU_LOCK="${TMP}/hermetic_testmode.lock"
# Never block: if some invocation still reached a held lock, fail fast instead of
# hanging the suite (0 = block forever; a few seconds is plenty for a temp lock).
export GPU_WINDOW_LOCK_TIMEOUT="${GPU_WINDOW_LOCK_TIMEOUT:-20}"

PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); printf 'ok   - %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf 'FAIL - %s\n     %s\n' "$1" "$2"; }

# --- a fake vm_stat whose BASELINE (wired + anonymous + occupied-by-compressor) =
# 50 GiB, with DELIBERATELY HUGE active/inactive (file page cache) to prove the
# W121 baseline IGNORES them (an active+inactive formula false-aborts on the 269 GiB
# expert bank's reclaimable file cache -- the window-47 abort).  The step's own
# growing memory is added on top as phys_footprint, not read from vm_stat.
#   baseline: wired 2000000 + anon 1000000 + comp 276800 = 3276800 pages * 16384 = 50 GiB
#   active 4000000 + inactive 300000 (~67 GiB file cache) are NOT counted.
FAKE_VMSTAT="${TMP}/vm_stat_50gib"
cat > "${FAKE_VMSTAT}" <<'EOF'
#!/bin/bash
cat <<'V'
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  100000.
Pages active:                               4000000.
Pages inactive:                              300000.
Pages speculative:                                0.
Anonymous pages:                            1000000.
Pages wired down:                           2000000.
Pages occupied by compressor:                276800.
V
EOF
chmod +x "${FAKE_VMSTAT}"

# A fake step-tree phys_footprint reader (constant 3 GiB) so box_used is
# deterministic in the clean-step scenario.  3 GiB = 3221225472 bytes.
FAKE_FP="${TMP}/fp_reader"
printf '#!/bin/bash\necho 3221225472\n' > "${FAKE_FP}"
chmod +x "${FAKE_FP}"
# A flat compressor sysctl (delta 0, never trips).
FAKE_COMP="${TMP}/comp_flat"
printf '#!/bin/bash\necho 1073741824\n' > "${FAKE_COMP}"
chmod +x "${FAKE_COMP}"

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
GPU_WINDOW_FOOTPRINT_READER="${FAKE_FP}" \
GPU_WINDOW_COMPRESSOR_CMD="${FAKE_COMP}" \
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

# 3. the exit summary reports the peak step phys_footprint (from the reader)
PEAK_LINE="$(grep 'peak step footprint' "${LOG}" | tail -1)"
if [[ -n "${PEAK_LINE}" ]]; then
  ok "exit summary has a 'peak step footprint' line"
  echo "     -> ${PEAK_LINE}"
else
  bad "exit summary has a 'peak step footprint' line" "line not found"
fi

# 3b. peak step footprint == the fake reader's 3.0 GiB (the guard reads phys_footprint,
# not ps RSS, so it captures Metal too).
PEAK_GIB="$(printf '%s\n' "${PEAK_LINE}" | sed -n 's/.*peak step footprint \([0-9.]*\) GiB.*/\1/p')"
if [[ -n "${PEAK_GIB}" ]] && awk -v v="${PEAK_GIB}" 'BEGIN{exit !(v+0 >= 2.9 && v+0 <= 3.1)}'; then
  ok "peak step footprint == the reader's 3.0 GiB (${PEAK_GIB})"
else
  bad "peak step footprint == 3.0 GiB" "parsed '${PEAK_GIB}' GiB"
fi

# 4. peak box used = baseline 50 + footprint 3 = 53 GiB (the plain sum)
BOX_GIB="$(printf '%s\n' "${PEAK_LINE}" | sed -n 's/.*peak box used \([0-9.]*\) GiB.*/\1/p')"
if [[ -n "${BOX_GIB}" ]] && awk -v v="${BOX_GIB}" 'BEGIN{exit !(v+0 >= 52.9 && v+0 <= 53.1)}'; then
  ok "peak box used = baseline + footprint (${BOX_GIB} GiB = 50 + 3)"
else
  bad "peak box used = 53 GiB" "parsed '${BOX_GIB}' GiB"
fi

# 5. peak compressor delta is reported (flat here -> 0.0)
if [[ "${PEAK_LINE}" == *"peak compressor delta"* ]]; then
  ok "exit summary reports the peak compressor delta"
else
  bad "exit summary reports the peak compressor delta" "phrase not found"
fi

# 6. a periodic memory-envelope sample line was logged during the step (baseline +
# step footprint = box used; compressor delta)
if grep -q "phase 4: mem sample -- baseline .* step footprint .* box used" "${LOG}"; then
  ok "logged a periodic 'mem sample' envelope line (baseline + footprint = box used)"
else
  bad "logged a periodic 'mem sample' envelope line during the step" "no sample line"
fi

# 7. it never touched the real exclusive lock
if [[ "$(grep -c 'acquired exclusive GPU lock' "${LOG}")" -ge 1 ]] && grep -q "${LOCK}" "${LOG}"; then
  ok "used the temp lock path, not the real /tmp/mtplx-gpu-exclusive.lock"
else
  bad "used the temp lock path" "lock line did not reference ${LOCK}"
fi

# 7b. W106 HIGH-2 + MEDIUM-B: the step-start log states BOTH guard caps, and since
#     this scenario's baseline is 50 GiB used with the DEFAULT 93 GiB child cap and the
#     W121 HIGH-3 DEFAULT 100 GiB ceiling (50 + 93 > 100), MEDIUM-B LOWERS the effective
#     child cap to ceiling - used_start = 50 GiB. Assert the lowering line (naming the
#     original 93 GiB default) + the guard-caps line showing the 50 GiB effective cap.
if grep -q "phase 4: effective child-tree footprint cap 50.0 GiB (lowered from 93.0 GiB" "${LOG}" \
   && grep -q "phase 4: guard caps -- child-tree footprint cap 50.0 GiB" "${LOG}" \
   && grep -q "box-used ceiling 100 GiB" "${LOG}"; then
  ok "MEDIUM-B: effective child cap lowered to ceiling-used_start (50 GiB); box-used ceiling 100 GiB (HIGH-3 default)"
else
  bad "MEDIUM-B effective child cap 50 GiB + 100 GiB ceiling" \
      "$(grep -E 'effective child-tree footprint cap|guard caps' "${LOG}" || echo 'no cap lines')"
fi

# =============================================================================
# W106 item 4: TREE-KILL on abort.  A fake step is a `bash -c` chain that launches
# a python child which spawns a long-lived `sleep` GRANDCHILD.  The wrapper aborts
# on the SYSTEM CEILING (fake vm_stat 50 GiB > a 40 GiB ceiling -- exit 8; the
# child-tree RSS cap now requires >= 1 GiB per HIGH-3 so it is not used to trip
# here).  After the abort NO descendant may survive: pre-W106 the abort killed only
# STEP_PID (the `bash -c` chain), orphaning the python + sleep.
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

# W121: a stateful fake phys_footprint reader that returns 0 for the first two polls
# (so the python child + sleep grandchild are spawned before any abort -- race-free)
# then 60 GiB.  With baseline 50 GiB + a 100 GiB ceiling, box_used = 50 -> 50 -> 110,
# so the BOX guard fires on the third poll and tree-kills the whole step subtree.
TK_FP="${TMP}/tk_fp_reader"
cat > "${TK_FP}" <<EOF
#!/bin/bash
C="${TMP}/tk_fp_count"
n=\$(cat "\$C" 2>/dev/null || echo 0); echo \$((n + 1)) > "\$C"
if [ "\$n" -lt 2 ]; then echo 0; else echo 64424509440; fi
EOF
chmod +x "${TK_FP}"
rm -f "${TMP}/tk_fp_count"

# W106 LOW-2: pass a NON-INTEGER grace so we also assert the wrapper validates it
# (falls back to 2) instead of breaking the KILL loop's bash arithmetic.  (NB: no
# comments INSIDE the backslash-continued env chain below -- a `#` there truncates
# the command and drops GPU_WINDOW_TEST_MODE/MTPLX_GPU_LOCK.)
GPU_WINDOW_TEST_MODE=1 \
MTPLX_GPU_LOCK="${TMP}/treekill.lock" \
GPU_WINDOW_VM_STAT_CMD="${FAKE_VMSTAT}" \
GPU_WINDOW_FOOTPRINT_READER="${TK_FP}" \
GPU_WINDOW_COMPRESSOR_CMD="${FAKE_COMP}" \
GPU_WINDOW_FOREIGN_WORKER_RSS_GB=100000 \
GPU_WINDOW_RSS_POLL_SECONDS=1 \
GPU_WINDOW_TOTAL_MEM_CEILING_GB=100 \
GPU_WINDOW_KILL_GRACE_SECONDS=abc \
TK_PIDFILE="${TK_PIDFILE}" \
  bash "${SCRIPT}" bash -c "${TK_STEP}" >"${TK_LOG}" 2>&1
TK_RC=$?

echo "----- gpu_window.sh log (TREE-KILL) -----"
cat "${TK_LOG}"
echo "-----------------------------------------"

# 8. the box-ceiling abort fired (exit 8) and killed the child
if [[ "${TK_RC}" -eq 8 ]]; then
  ok "tree-kill scenario aborted on the box ceiling (exit 8)"
else
  bad "tree-kill scenario aborted on the box ceiling (exit 8)" "exit code was ${TK_RC}"
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

# 12. the log shows the box-ceiling abort path (baseline + step footprint)
if grep -q "BOX used memory .* exceeded ceiling" "${TK_LOG}"; then
  ok "abort log names the box-ceiling breach (baseline + step footprint)"
else
  bad "abort log names the box-ceiling breach" "ceiling line not found"
fi

# 13. W106 LOW-2: the non-integer GPU_WINDOW_KILL_GRACE_SECONDS was validated
#     (warned + fell back to 2) rather than breaking the tree-kill's arithmetic.
if grep -q "GPU_WINDOW_KILL_GRACE_SECONDS='abc' is not a non-negative integer; using 2" "${TK_LOG}"; then
  ok "non-integer kill-grace is validated and falls back to 2 (LOW-2)"
else
  bad "non-integer kill-grace validation" "warning line not found"
fi

# =============================================================================
# W106 HIGH-3: a FRACTIONAL safety cap (e.g. GPU_WINDOW_TOTAL_MEM_CEILING_GB=95.5, a
# GiB/GB confusion) must REFUSE (exit 2 BEFORE phase 0), not silently fall back to
# the 102 default and RAISE the ceiling over the operator's intent.
# =============================================================================
MA_LOG="${TMP}/mediuma.log"
GPU_WINDOW_TEST_MODE=1 \
MTPLX_GPU_LOCK="${TMP}/mediuma.lock" \
GPU_WINDOW_VM_STAT_CMD="${FAKE_VMSTAT}" \
GPU_WINDOW_TOTAL_MEM_CEILING_GB=95.5 \
  bash "${SCRIPT}" bash -c "true" >"${MA_LOG}" 2>&1
MA_RC=$?

echo "----- gpu_window.sh log (HIGH-3 fractional ceiling refusal) -----"
cat "${MA_LOG}"
echo "-----------------------------------------------------------------"

# 14. the fractional ceiling is REFUSED with exit 2 (before phase 0)
if [[ "${MA_RC}" -eq 2 ]] && grep -q "GPU_WINDOW_TOTAL_MEM_CEILING_GB=95.5 is invalid" "${MA_LOG}"; then
  ok "HIGH-3: fractional system ceiling REFUSED (exit 2), not silently raised"
else
  bad "HIGH-3 fractional ceiling refusal (exit 2)" "rc=${MA_RC}; log: $(cat "${MA_LOG}")"
fi

# 15. HIGH-3: CHILD_RSS_CAP_BYTES "93" (93 BYTES, < 1 GiB) is REFUSED, not treated
#     as 93 GiB nor left to kill the step after bootout.
CAP_LOG="${TMP}/cap.log"
GPU_WINDOW_TEST_MODE=1 \
MTPLX_GPU_LOCK="${TMP}/cap.lock" \
GPU_WINDOW_VM_STAT_CMD="${FAKE_VMSTAT}" \
GPU_WINDOW_CHILD_RSS_CAP_BYTES=93 \
  bash "${SCRIPT}" bash -c "true" >"${CAP_LOG}" 2>&1
CAP_RC=$?
if [[ "${CAP_RC}" -eq 2 ]] && grep -q "GPU_WINDOW_CHILD_RSS_CAP_BYTES=93 is invalid" "${CAP_LOG}"; then
  ok "HIGH-3: CHILD_RSS_CAP_BYTES=93 (bytes, < 1 GiB) REFUSED (exit 2)"
else
  bad "HIGH-3 child-cap bytes refusal" "rc=${CAP_RC}; log: $(cat "${CAP_LOG}")"
fi

# 15b. W106 LOW (round 4): GPU_WINDOW_RSS_POLL_SECONDS=0 is REFUSED (a `sleep 0`
#      busy-loop would pin a core and inflate the host-encode-sensitive window).
POLL_LOG="${TMP}/poll.log"
GPU_WINDOW_TEST_MODE=1 \
MTPLX_GPU_LOCK="${TMP}/poll.lock" \
GPU_WINDOW_VM_STAT_CMD="${FAKE_VMSTAT}" \
GPU_WINDOW_RSS_POLL_SECONDS=0 \
  bash "${SCRIPT}" bash -c "true" >"${POLL_LOG}" 2>&1
POLL_RC=$?
if [[ "${POLL_RC}" -eq 2 ]] && grep -q "GPU_WINDOW_RSS_POLL_SECONDS=0 is invalid" "${POLL_LOG}"; then
  ok "LOW: GPU_WINDOW_RSS_POLL_SECONDS=0 REFUSED (exit 2)"
else
  bad "LOW poll=0 refusal" "rc=${POLL_RC}; log: $(cat "${POLL_LOG}")"
fi

# (W121: the old MEDIUM-1 "ps-vs-Metal RSS undercount" adjustment is retired -- the
# guard now measures phys_footprint directly, which already includes Metal, so there
# is no ps-RSS undercount to correct.  Compressor-tripwire coverage lives in
# tests/test_gpu_window_used_mem_w121.sh.)

printf '\n%d passed, %d failed\n' "${PASS}" "${FAIL}"
[[ "${FAIL}" -eq 0 ]]
