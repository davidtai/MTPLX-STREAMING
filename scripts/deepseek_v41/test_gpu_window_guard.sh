#!/usr/bin/env bash
# Shell-level unit test for gpu_window.sh's phase-4 system-wide memory guard.
#
# Exercises the pure guard math through the script's `--selftest` hooks against
# FAKE `vm_stat` / `ps` output -- no GPU, no lock, no launchctl, no real memory
# state.  The hooks run and exit before the lock phase, so this is safe to run
# anywhere (CPU only).
#
#   bash scripts/deepseek_v41/test_gpu_window_guard.sh
#
# Exit 0 = all assertions pass; exit 1 = at least one failed.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${HERE}/gpu_window.sh"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

PASS=0
FAIL=0
ok()   { PASS=$((PASS + 1)); printf 'ok   - %s\n' "$1"; }
bad()  { FAIL=$((FAIL + 1)); printf 'FAIL - %s\n     expected: %s\n     actual:   %s\n' "$1" "$2" "$3"; }
eq()   { if [[ "$2" == "$3" ]]; then ok "$1"; else bad "$1" "$2" "$3"; fi; }

# --- a fake vm_stat whose (wired + active + occupied-by-compressor) pages ------
# sum to EXACTLY 100 GiB at the 16384-byte page size:
#   wired 2500000 + active 3500000 + comp 553600 = 6553600 pages
#   6553600 * 16384 = 107374182400 bytes = 100.0 GiB
# "stored in compressor" is deliberately large and MUST be ignored (it is the
# pre-compression logical count, not the physical footprint).
FAKE_VMSTAT="${TMP}/vm_stat_100gib"
cat > "${FAKE_VMSTAT}" <<'EOF'
#!/bin/bash
cat <<'V'
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  100000.
Anonymous pages:                            3500000.
Pages inactive:                              200000.
Pages speculative:                            10000.
Pages throttled:                                  0.
Pages wired down:                           2500000.
Pages stored in compressor:                 9000000.
Pages occupied by compressor:                553600.
V
EOF
chmod +x "${FAKE_VMSTAT}"

run_vmstat() { GPU_WINDOW_VM_STAT_CMD="${FAKE_VMSTAT}" "$@"; }

# used_mem_bytes = exactly 100 GiB in bytes
eq "used_mem_bytes sums wired+anonymous+occupied-compressor at the page size" \
   "107374182400" \
   "$(run_vmstat bash "${SCRIPT}" --selftest used-mem-bytes)"

eq "used-mem-gib renders 100.0" \
   "100.0" \
   "$(run_vmstat bash "${SCRIPT}" --selftest used-mem-gib)"

# ceiling decisions
eq "100 GiB used is UNDER the default 105 GiB ceiling -> no" \
   "no" \
   "$(run_vmstat bash "${SCRIPT}" --selftest over-ceiling)"

eq "100 GiB used is OVER a 90 GiB ceiling -> yes" \
   "yes" \
   "$(GPU_WINDOW_TOTAL_MEM_CEILING_GB=90 run_vmstat bash "${SCRIPT}" --selftest over-ceiling)"

eq "100 GiB used exactly AT a 100 GiB ceiling is not strictly over -> no" \
   "no" \
   "$(GPU_WINDOW_TOTAL_MEM_CEILING_GB=100 run_vmstat bash "${SCRIPT}" --selftest over-ceiling)"

# A vm_stat that omits the compressor line still parses (comp defaults to 0).
FAKE_NOCOMP="${TMP}/vm_stat_nocomp"
cat > "${FAKE_NOCOMP}" <<'EOF'
#!/bin/bash
cat <<'V'
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Anonymous pages:                            3500000.
Pages wired down:                           2500000.
V
EOF
chmod +x "${FAKE_NOCOMP}"
# (3500000 + 2500000) * 16384 = 98304000000
eq "missing compressor line -> comp treated as 0" \
   "98304000000" \
   "$(GPU_WINDOW_VM_STAT_CMD="${FAKE_NOCOMP}" bash "${SCRIPT}" --selftest used-mem-bytes)"

# --- a fake ps for the foreign-worker scan ------------------------------------
FAKE_PS="${TMP}/ps_workers"
cat > "${FAKE_PS}" <<'EOF'
#!/bin/bash
cat <<'P'
 4242 31000000 /worktrees/other/.venv/bin/python3
 4243   900000 /usr/bin/python3
 5001 40000000 /Applications/SomeBigApp.app/Contents/MacOS/SomeBigApp
 5002  3000000 mtplx-serve
 5003  5000000 /opt/homebrew/bin/mlx_worker
P
EOF
chmod +x "${FAKE_PS}"

run_ps() { GPU_WINDOW_PS_CMD="${FAKE_PS}" "$@"; }

# cap 2 GiB: the three python/mtplx/mlx procs above 2 GiB, sorted as emitted;
# NOT the 900 MB python (under cap) and NOT the 40 GiB non-matching app.
heavy_2g="$(run_ps bash "${SCRIPT}" --selftest heavy-workers | awk '{print $1}' | tr '\n' ',')"
eq "heavy-workers @2GiB lists only python/mtplx/mlx procs above the cap" \
   "4242,5002,5003," \
   "${heavy_2g}"

eq "heavy-workers @2GiB does NOT list the 40 GiB non-python app" \
   "" \
   "$(run_ps bash "${SCRIPT}" --selftest heavy-workers | awk '$1==5001')"

eq "heavy-workers @2GiB does NOT list the sub-cap python" \
   "" \
   "$(run_ps bash "${SCRIPT}" --selftest heavy-workers | awk '$1==4243')"

# cap 6 GiB: only the 31 GiB python remains.
heavy_6g="$(GPU_WINDOW_FOREIGN_WORKER_RSS_GB=6 run_ps bash "${SCRIPT}" --selftest heavy-workers | awk '{print $1}' | tr '\n' ',')"
eq "heavy-workers @6GiB lists only the single >6 GiB worker" \
   "4242," \
   "${heavy_6g}"

# empty ps -> empty result (nothing resident).
FAKE_EMPTY="${TMP}/ps_empty"
printf '#!/bin/bash\necho ""\n' > "${FAKE_EMPTY}"
chmod +x "${FAKE_EMPTY}"
eq "heavy-workers with no resident workers -> empty" \
   "" \
   "$(GPU_WINDOW_PS_CMD="${FAKE_EMPTY}" bash "${SCRIPT}" --selftest heavy-workers)"

# --- dispatch guards ----------------------------------------------------------
bash "${SCRIPT}" --selftest bogus-target >/dev/null 2>&1
eq "unknown --selftest target exits 2" "2" "$?"

# --- summary ------------------------------------------------------------------
printf '\n%d passed, %d failed\n' "${PASS}" "${FAIL}"
[[ "${FAIL}" -eq 0 ]]
