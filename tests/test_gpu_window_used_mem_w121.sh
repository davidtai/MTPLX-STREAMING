#!/usr/bin/env bash
# W121 hermetic test for gpu_window.sh's corrected used-memory formula and its
# `top` cross-check self-test.  NO GPU, NO lock, NO launchctl -- the --selftest
# hooks run the pure guard math against a FAKE vm_stat / FAKE top and exit before
# the lock phase.
#
# Proves:
#   1. used_mem_bytes() = (wired + active + inactive + speculative + compressor)
#      * page size (the top-equivalent figure), on a known vector.
#   2. top-cross-check prints "ok" when the formula agrees with `top` within 1 GB.
#   3. top-cross-check prints "MISMATCH" when they diverge by > 1 GB.
#   4. Regression: a NON-WIRED Metal scenario (low wired, big active/inactive --
#      exactly window 46, where the bench never called set_wired_limit) is caught
#      by the new formula, while the OLD wired+anonymous+compressor formula would
#      have undercounted it by tens of GiB.
#
#   nice -n 19 bash tests/test_gpu_window_used_mem_w121.sh
#
# Exit 0 = all pass; exit 1 = a failure.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${HERE}/../scripts/deepseek_v41/gpu_window.sh"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

# Never touch the real lock even though --selftest exits before the lock phase.
export GPU_WINDOW_TEST_MODE=1
export MTPLX_GPU_LOCK="${TMP}/hermetic_testmode.lock"
export GPU_WINDOW_LOCK_TIMEOUT=20

PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); printf 'ok   - %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf 'FAIL - %s\n     %s\n' "$1" "$2"; }

PS=16384  # page size (bytes) used by all fixtures

make_vmstat() {  # $1=file wired $2 active $3 inactive $4 speculative $5 anon $6 comp
  cat > "$1" <<EOF
#!/bin/bash
cat <<'V'
Mach Virtual Memory Statistics: (page size of ${PS} bytes)
Pages free:                                  100000.
Pages active:                               ${3}.
Pages inactive:                             ${4}.
Pages speculative:                          ${5}.
Anonymous pages:                            ${6}.
Pages wired down:                           ${2}.
Pages occupied by compressor:               ${7}.
V
EOF
  chmod +x "$1"
}

make_top() {  # $1=file  $2="102G"  (the used token top prints)
  cat > "$1" <<EOF
#!/bin/bash
cat <<'V'
Processes: 700 total
PhysMem: ${2} used (82G wired, 1091M compressor), 22G unused.
VM: 200T vsize
V
EOF
  chmod +x "$1"
}

# --- 1. exact formula on a known vector ---------------------------------------
# wired 1500000 + active 1300000 + inactive 200000 + spec 0 + comp 276800
#   = 3276800 pages * 16384 = 53687091200 bytes = 50.0 GiB
VMS="${TMP}/vmstat_50"
make_vmstat "${VMS}" 1500000 1300000 200000 0 900000 276800
GOT="$(GPU_WINDOW_VM_STAT_CMD="${VMS}" bash "${SCRIPT}" --selftest used-mem-bytes 2>/dev/null | tr -d '[:space:]')"
EXP=53687091200
if [[ "${GOT}" == "${EXP}" ]]; then
  ok "used_mem_bytes = wired+active+inactive+spec+comp (${GOT} == ${EXP}, 50 GiB)"
else
  bad "used_mem_bytes exact formula" "got ${GOT}, want ${EXP}"
fi

# The OLD formula (wired+anonymous+comp) on the SAME vector would be
#   1500000 + 900000 + 276800 = 2676800 pages * 16384 = 43.85 GiB -- 6+ GiB lower,
# confirming anonymous(=internal) is NOT the right term.
GIB_NEW="$(GPU_WINDOW_VM_STAT_CMD="${VMS}" bash "${SCRIPT}" --selftest used-mem-gib 2>/dev/null | tr -d '[:space:]')"
if [[ "${GIB_NEW}" == "50.0" ]]; then
  ok "used-mem-gib renders 50.0 GiB"
else
  bad "used-mem-gib renders 50.0 GiB" "got ${GIB_NEW}"
fi

# --- 2. top cross-check: agreement -> ok --------------------------------------
# 50 GiB used == 53.69 GB; top prints "53G used" (parsed as 53*1024^3 = 56908316672),
# delta = 53.69 - 53.0 GiB = ... keep it well within 1 GiB: make top say "50G".
TOPF="${TMP}/top_agree"
make_top "${TOPF}" "50G"
CC="$(GPU_WINDOW_VM_STAT_CMD="${VMS}" GPU_WINDOW_TOP_CMD="${TOPF}" bash "${SCRIPT}" --selftest top-cross-check 2>/dev/null)"
if [[ "${CC}" == ok* ]]; then
  ok "top-cross-check agrees within 1 GiB (${CC})"
else
  bad "top-cross-check agreement" "got '${CC}'"
fi

# --- 3. top cross-check: divergence > 1 GiB -> MISMATCH ------------------------
TOPF_BAD="${TMP}/top_disagree"
make_top "${TOPF_BAD}" "80G"   # 80 GiB vs formula's 50 GiB -> 30 GiB apart
CCB="$(GPU_WINDOW_VM_STAT_CMD="${VMS}" GPU_WINDOW_TOP_CMD="${TOPF_BAD}" bash "${SCRIPT}" --selftest top-cross-check 2>/dev/null)"
if [[ "${CCB}" == MISMATCH* ]]; then
  ok "top-cross-check flags a >1 GiB divergence (${CCB})"
else
  bad "top-cross-check divergence" "got '${CCB}'"
fi

# --- 4. regression: non-wired Metal (window 46) -------------------------------
# Metal held NON-WIRED sits in active/inactive, not wired.  Model it: wired only
# 330000 (~5 GiB, like window 46's "5406M wired"), but active 4300000 + inactive
# 300000 (~70 GiB of Metal in the LRU) + comp 2100000 (~32 GiB compressed).
#   NEW: (330000 + 4300000 + 300000 + 0 + 2100000) * 16384 = 7030000 * 16384
#      = 115180994560 B = 107.3 GiB  -> OVER the 102 GiB ceiling (guard aborts).
#   OLD (wired+anon+comp): anon is small here (Metal is not "internal"): say
#   400000 -> (330000 + 400000 + 2100000)*16384 = 2830000*16384 = 44.3 GiB -> the
#   old guard would have seen only 44 GiB and NEVER aborted (the window-46 bug).
VMS46="${TMP}/vmstat_w46"
make_vmstat "${VMS46}" 330000 4300000 300000 0 400000 2100000
OVER="$(GPU_WINDOW_TOTAL_MEM_CEILING_GB=102 GPU_WINDOW_VM_STAT_CMD="${VMS46}" \
        bash "${SCRIPT}" --selftest over-ceiling 2>/dev/null | tr -d '[:space:]')"
if [[ "${OVER}" == "yes" ]]; then
  ok "non-wired Metal (window 46) trips the 102 GiB ceiling under the new formula"
else
  bad "non-wired Metal trips the ceiling" "over-ceiling said '${OVER}' (regression: guard would not abort)"
fi
NEWGIB="$(GPU_WINDOW_VM_STAT_CMD="${VMS46}" bash "${SCRIPT}" --selftest used-mem-gib 2>/dev/null | tr -d '[:space:]')"
# 7030000 * 16384 / 1073741824 = 107.3 GiB
if [[ "${NEWGIB}" == 107.* ]]; then
  ok "window-46 non-wired Metal reads ~107 GiB used (${NEWGIB}), not the old ~44 GiB blind spot"
else
  bad "window-46 used ~107 GiB" "got ${NEWGIB}"
fi

echo
printf 'W121 gpu_window used-mem: %d passed, %d failed\n' "${PASS}" "${FAIL}"
[[ "${FAIL}" -eq 0 ]]
