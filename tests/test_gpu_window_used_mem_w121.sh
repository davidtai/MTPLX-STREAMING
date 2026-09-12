#!/usr/bin/env bash
# W121 hermetic test for gpu_window.sh's corrected box-used guard.
#
# The guard quantity is the PLAIN SUM David asked for:
#   box_used = used_at_start (one-time baseline wired+anon+comp, file cache excluded)
#            + Σ phys_footprint over the step tree (proc_pid_rusage ri_phys_footprint,
#              which INCLUDES Metal/IOAccelerator wired-or-not but NOT the shared file
#              page cache).
# Plus a compressor tripwire: abort if vm.compressor_bytes_used grows > trip GiB over
# its at-start value (the swap-collapse signature).
#
# This replaces the earlier top-equivalent (wired+active+inactive+spec+comp) attempt,
# which summed the 269 GiB expert bank's reclaimable FILE CACHE and false-aborted the
# window-47 load 10 s in.
#
#   nice -n 19 bash tests/test_gpu_window_used_mem_w121.sh
# Exit 0 = all pass; 1 = a failure.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${HERE}/../scripts/deepseek_v41/gpu_window.sh"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
export GPU_WINDOW_TEST_MODE=1
export MTPLX_GPU_LOCK="${TMP}/hermetic_testmode.lock"
export GPU_WINDOW_LOCK_TIMEOUT=20

PASS=0; FAIL=0
ok()  { PASS=$((PASS + 1)); printf 'ok   - %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf 'FAIL - %s\n     %s\n' "$1" "$2"; }

# fake vm_stat: baseline = wired+anon+comp.  wired 700000 + anon 53248 + comp 65536
#   = 818784 pages * 16384 = 13413236736 B = 12.49 GiB.  Choose ~11.5 GiB (window 46
#   baseline): wired 640000 + anon 40000 + comp 74000 = 754000 * 16384 = 11.51 GiB.
FAKE_VMS="${TMP}/vmstat"
cat > "${FAKE_VMS}" <<'EOF'
#!/bin/bash
cat <<'V'
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  100000.
Pages active:                               4300000.
Pages inactive:                              300000.
Anonymous pages:                              40000.
Pages wired down:                            640000.
Pages occupied by compressor:                 74000.
V
EOF
chmod +x "${FAKE_VMS}"
# baseline (wired 640000 + anon 40000 + comp 74000) * 16384 = 12,353,536,000 B = 11.50 GiB
# NOTE: active/inactive are HUGE here (the 269 GiB bank's file cache); the baseline
# formula must IGNORE them (that is the whole point).

# --- 1. baseline ignores file cache (active/inactive) ---
BASE_GIB="$(GPU_WINDOW_VM_STAT_CMD="${FAKE_VMS}" bash "${SCRIPT}" --selftest used-mem-gib | tr -d '[:space:]')"
if [[ "${BASE_GIB}" == "11.5" ]]; then
  ok "baseline = wired+anon+comp only (${BASE_GIB} GiB), ignores the file-cache active/inactive"
else
  bad "baseline ignores file cache" "got ${BASE_GIB} GiB (expected 11.5; active/inactive must NOT count)"
fi

# --- 2. box_used = baseline + step footprint ---
FP74="${TMP}/fp74"; printf '#!/bin/bash\necho 79456894976\n' > "${FP74}"; chmod +x "${FP74}"  # 74 GiB
BOX_GIB="$(GPU_WINDOW_VM_STAT_CMD="${FAKE_VMS}" GPU_WINDOW_FOOTPRINT_READER="${FP74}" \
           bash "${SCRIPT}" --selftest box-used-gib 4242 | tr -d '[:space:]')"
# 11.5 + 74.0 = 85.5 GiB
if [[ "${BOX_GIB}" == "85.5" ]]; then
  ok "box_used = baseline + step footprint (${BOX_GIB} GiB = 11.5 + 74.0), window-46 vector"
else
  bad "box_used sum" "got ${BOX_GIB} (expected 85.5)"
fi

# --- helper: a fake step that sleeps; run the wrapper end-to-end in TEST MODE ---
STEP="${TMP}/step.sh"
cat > "${STEP}" <<'EOF'
#!/bin/bash
python3 -c "import time; time.sleep(8)" &
wait $!
EOF
chmod +x "${STEP}"

run_window() {  # $1=footprint_reader $2=sysctl_cmd -> prints "rc=<code>" + log path
  local fpr="$1" sysctl_cmd="$2" log="${TMP}/log.$RANDOM"
  GPU_WINDOW_VM_STAT_CMD="${FAKE_VMS}" \
  GPU_WINDOW_FOOTPRINT_READER="${fpr}" \
  GPU_WINDOW_COMPRESSOR_CMD="${sysctl_cmd}" \
  GPU_WINDOW_TOTAL_MEM_CEILING_GB=100 \
  GPU_WINDOW_RSS_POLL_SECONDS=1 \
  GPU_WINDOW_COMPRESSOR_TRIP_GB=8 \
  GPU_WINDOW_FOREIGN_WORKER_RSS_GB=9999 \
  MTPLX_GPU_LOCK="${TMP}/w.$RANDOM.lock" \
  bash "${SCRIPT}" bash "${STEP}" >"${log}" 2>&1
  local rc=$?
  printf 'rc=%s log=%s\n' "${rc}" "${log}"
}

# flat compressor sysctl (delta 0)
SYSCTL_FLAT="${TMP}/sysctl_flat"
printf '#!/bin/bash\necho 1073741824\n' > "${SYSCTL_FLAT}"; chmod +x "${SYSCTL_FLAT}"

# --- 3. window-46 vector (footprint 74, baseline 11.5, flat compressor) -> NO abort ---
eval "$(run_window "${FP74}" "${SYSCTL_FLAT}")"
if [[ "${rc}" == "0" ]] && grep -q "box used 85.5 GiB" "${log}"; then
  ok "box 85.5 GiB (< 100 ceiling) does NOT abort; mem sample logs the three numbers"
else
  bad "no false abort at box 85.5" "rc=${rc}; $(grep -m1 'mem sample\|ERROR' "${log}" || echo 'no sample')"
fi

# --- 4. footprint 95 GiB -> box 106.5 > 100 ceiling -> abort exit 8 ---
FP95="${TMP}/fp95"; printf '#!/bin/bash\necho 102005473280\n' > "${FP95}"; chmod +x "${FP95}"  # 95 GiB
eval "$(run_window "${FP95}" "${SYSCTL_FLAT}")"
if [[ "${rc}" == "8" ]] && grep -q "BOX used memory" "${log}"; then
  ok "box 106.5 GiB (> 100 ceiling) aborts exit 8 (baseline + footprint over ceiling)"
else
  bad "box over ceiling aborts" "rc=${rc}; $(grep -m1 'ERROR' "${log}" || echo none)"
fi

# --- 5. compressor +33 GiB over start -> abort exit 8 (even with small footprint) ---
# stateful fake sysctl: first call (COMPRESSOR_START) returns 1 GiB, later calls +33 GiB.
SYSCTL_JUMP="${TMP}/sysctl_jump"
cat > "${SYSCTL_JUMP}" <<EOF
#!/bin/bash
C="${TMP}/compcount"
n=\$(cat "\$C" 2>/dev/null || echo 0); echo \$((n+1)) > "\$C"
if [ "\$n" -eq 0 ]; then echo 1073741824; else echo 36507222016; fi   # 1 GiB -> 34 GiB (+33)
EOF
chmod +x "${SYSCTL_JUMP}"
FP10="${TMP}/fp10"; printf '#!/bin/bash\necho 10737418240\n' > "${FP10}"; chmod +x "${FP10}"  # 10 GiB
rm -f "${TMP}/compcount"
eval "$(run_window "${FP10}" "${SYSCTL_JUMP}")"
if [[ "${rc}" == "8" ]] && grep -q "COMPRESSOR grew" "${log}"; then
  ok "compressor +33 GiB over start aborts exit 8 (swap-collapse tripwire), footprint small"
else
  bad "compressor tripwire aborts" "rc=${rc}; $(grep -m1 'ERROR' "${log}" || echo none)"
fi

echo
printf 'W121 gpu_window box-used: %d passed, %d failed\n' "${PASS}" "${FAIL}"
[[ "${FAIL}" -eq 0 ]]
