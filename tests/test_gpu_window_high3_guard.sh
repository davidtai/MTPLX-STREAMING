#!/usr/bin/env bash
# W121 HIGH-3 hermetic test for gpu_window.sh's hardened phase-4 box guard:
#   * a LIVE physical-used term (wired+active+inactive+compressor) catches all resident pages
#     to OTHER processes growing (box_used = max(baseline + step footprint, live used));
#   * the step-footprint reader FAILING (exit != 0 / no valid footprint) ABORTS instead
#     of continuing at box_used == baseline (the old fail-open at '0').
#
# GPU_WINDOW_TEST_MODE=1 + a temp lock: NO sysctl, NO launchctl, NO real GPU lock, NO
# Metal, NO agent bootout.  Every reader (vm_stat, compressor, footprint) is injected.
#
#   nice -n 19 bash tests/test_gpu_window_high3_guard.sh
# Exit 0 = all pass; 1 = a failure.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${HERE}/../scripts/deepseek_v41/gpu_window.sh"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

export GPU_WINDOW_TEST_MODE=1
export MTPLX_GPU_LOCK="${TMP}/hermetic.lock"
export GPU_WINDOW_LOCK_TIMEOUT=20
export GPU_WINDOW_FOREIGN_WORKER_RSS_GB=100000   # skip the foreign-worker scan
export GPU_WINDOW_RSS_POLL_SECONDS=1
export GPU_WINDOW_KILL_GRACE_SECONDS=1
export GPU_WINDOW_COMPRESSOR_TRIP_GB=100000       # never trip on the compressor here
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$(( 200 * 1024 * 1024 * 1024 ))  # huge; box guard is the one under test

PASS=0; FAIL=0
ok()  { PASS=$((PASS + 1)); printf 'ok   - %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf 'FAIL - %s\n     %s\n' "$1" "$2"; }

# fake compressor sysctl: always 0 bytes (no compressor tripwire).
FAKE_COMP="${TMP}/fake_sysctl"
cat > "${FAKE_COMP}" <<'EOF'
#!/bin/bash
echo 0
EOF
chmod +x "${FAKE_COMP}"
export GPU_WINDOW_COMPRESSOR_CMD="${FAKE_COMP}"

# a small, constant footprint reader: 1 GiB regardless of the pid arg.
FAKE_FP_OK="${TMP}/fp_ok"
cat > "${FAKE_FP_OK}" <<'EOF'
#!/bin/bash
echo 1073741824
EOF
chmod +x "${FAKE_FP_OK}"

# a BROKEN footprint reader: exits 2 (the tree_footprint.py subprocess-timeout signature).
FAKE_FP_BAD="${TMP}/fp_bad"
cat > "${FAKE_FP_BAD}" <<'EOF'
#!/bin/bash
exit 2
EOF
chmod +x "${FAKE_FP_BAD}"

# a static low vm_stat (~11.5 GiB): wired 640000 + active 40000 + comp 74000.
FAKE_VMS_LOW="${TMP}/vms_low"
cat > "${FAKE_VMS_LOW}" <<'EOF'
#!/bin/bash
cat <<'V'
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  100000.
Pages active:                                 40000.
Pages inactive:                                   0.
Anonymous pages:                              40000.
Pages wired down:                            640000.
Pages occupied by compressor:                 74000.
V
EOF
chmod +x "${FAKE_VMS_LOW}"

# a STATEFUL vm_stat: first call (baseline) ~11.5 GiB, every later call (mid-step)
# active +30 GiB (1966080 pages) -> ~41.5 GiB used, to simulate ANOTHER process growing.
FAKE_VMS_JUMP="${TMP}/vms_jump"
cat > "${FAKE_VMS_JUMP}" <<EOF
#!/bin/bash
CNT="${TMP}/vms_calls"
n=\$(cat "\${CNT}" 2>/dev/null || echo 0)
echo \$(( n + 1 )) > "\${CNT}"
if [ "\${n}" -eq 0 ]; then anon=40000; else anon=2006080; fi   # +1966080 pages = +30 GiB
cat <<V
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  100000.
Pages active:                                 \${anon}.
Pages inactive:                                   0.
Anonymous pages:                              \${anon}.
Pages wired down:                            640000.
Pages occupied by compressor:                 74000.
V
EOF
chmod +x "${FAKE_VMS_JUMP}"

# --- 1. positive control: reader OK + low vm_stat -> step completes clean (no false trip)
GPU_WINDOW_TOTAL_MEM_CEILING_GB=30 \
GPU_WINDOW_VM_STAT_CMD="${FAKE_VMS_LOW}" \
GPU_WINDOW_FOOTPRINT_READER="${FAKE_FP_OK}" \
  bash "${SCRIPT}" sleep 2 >/dev/null 2>&1
rc=$?
if [[ "${rc}" -eq 0 ]]; then
  ok "control: baseline 11.5 + footprint 1 GiB < 30 GiB ceiling, live also low -> clean exit"
else
  bad "control clean exit" "expected rc 0, got ${rc}"
fi

# --- 2. HIGH-3 fail-closed: a broken footprint reader (exit 2) ABORTS (rc != 0),
#        never continues at box_used == baseline.
GPU_WINDOW_TOTAL_MEM_CEILING_GB=100 \
GPU_WINDOW_VM_STAT_CMD="${FAKE_VMS_LOW}" \
GPU_WINDOW_FOOTPRINT_READER="${FAKE_FP_BAD}" \
  bash "${SCRIPT}" sleep 30 >"${TMP}/out2.log" 2>&1
rc=$?
if [[ "${rc}" -ne 0 ]] && grep -qi "footprint reader UNREADABLE\|footprint.*unreadable" "${TMP}/out2.log"; then
  ok "reader exits 2 -> window ABORTS (rc ${rc} != 0), logs 'footprint reader UNREADABLE'"
else
  bad "reader failure aborts" "rc=${rc}; log:$(tail -3 "${TMP}/out2.log" | tr '\n' '|')"
fi

# --- 3. HIGH-3 live term: footprint stays small (< ceiling) but vm_stat jumps +30 GiB
#        mid-step -> the live system term trips the box guard (rc != 0).  Without the
#        live term box_used = baseline+footprint ~12.5 GiB < 30 and the step would finish.
rm -f "${TMP}/vms_calls"
GPU_WINDOW_TOTAL_MEM_CEILING_GB=30 \
GPU_WINDOW_VM_STAT_CMD="${FAKE_VMS_JUMP}" \
GPU_WINDOW_FOOTPRINT_READER="${FAKE_FP_OK}" \
  bash "${SCRIPT}" sleep 30 >"${TMP}/out3.log" 2>&1
rc=$?
if [[ "${rc}" -ne 0 ]] && grep -qi "GUARD accounted memory" "${TMP}/out3.log"; then
  ok "vm_stat anon +30 GiB mid-step -> live term trips the box guard (rc ${rc} != 0)"
else
  bad "live term catches other-process growth" "rc=${rc}; log:$(tail -3 "${TMP}/out3.log" | tr '\n' '|')"
fi

printf '\n%d passed, %d failed\n' "${PASS}" "${FAIL}"
[[ "${FAIL}" -eq 0 ]]
