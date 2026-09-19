#!/usr/bin/env bash
# W106 hermetic test for gpu_window.sh's resident-agent RESTORE (real-window
# incidents, windows 42/43): restore_qwen bootstrapped a TRANSIENT guard-dir plist
# that was gone by restore time, leaving com.tea.qwen DOWN.  This drives the pure
# _resolve_restore_plist + the _do_restore core through `--selftest restore-plist`
# and `--selftest restore-run` against a FAKE launchctl -- NO real launchctl, NO
# sysctl, NO GPU lock, NO Metal.  It NEVER touches the real service.
#
#   nice -n 19 bash tests/test_gpu_window_restore.sh
#
# Exit 0 = all assertions pass; exit 1 = at least one failed.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${HERE}/../scripts/deepseek_v41/gpu_window.sh"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

# HARD SAFETY NET: never touch the real launchctl or lock even if a code path slips.
export GPU_WINDOW_LAUNCHCTL_CMD="${TMP}/fake_launchctl"
export GPU_WINDOW_CURL_CMD="${TMP}/fake_curl"
export MTPLX_GPU_LOCK="${TMP}/hermetic.lock"
export GPU_WINDOW_TEST_MODE=1
export GPU_WINDOW_RESTORE_TIMEOUT=3

PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); printf 'ok   - %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf 'FAIL - %s\n     %s\n' "$1" "$2"; }

# --- a fake launchctl: records `bootstrap` targets; `print` reflects a state file --
#   print   -> exit 0 iff ${TMP}/loaded exists (service present), else exit 1
#   bootstrap <domain> <plist> -> record the plist, create ${TMP}/loaded, exit 0
cat > "${GPU_WINDOW_LAUNCHCTL_CMD}" <<EOF
#!/bin/bash
BOOTREC="${TMP}/bootstrap_calls"
LOADED="${TMP}/loaded"
case "\$1" in
  print)     [[ -f "\${LOADED}" ]] && exit 0 || exit 1 ;;
  bootstrap) printf '%s\n' "\$3" >> "\${BOOTREC}"; : > "\${LOADED}"; exit 0 ;;
  *)         exit 0 ;;
esac
EOF
chmod +x "${GPU_WINDOW_LAUNCHCTL_CMD}"
cat > "${GPU_WINDOW_CURL_CMD}" <<'EOF'
#!/bin/bash
case "${!#}" in
  */v1/models) echo '{"data":[{"id":"hermetic-model"}]}' ;;
  *) echo '{"ok":true,"startup":{"warmup":{"background":{"state":"done"}}}}' ;;
esac
EOF
chmod +x "${GPU_WINDOW_CURL_CMD}"

_reset_state() { rm -f "${TMP}/bootstrap_calls" "${TMP}/loaded"; }
_bootstrapped() { [[ -f "${TMP}/bootstrap_calls" ]] && cat "${TMP}/bootstrap_calls" || printf ''; }

CANON="${TMP}/LaunchAgents.com.tea.qwen.plist"   # the durable canonical plist
: > "${CANON}"
GUARD_GONE="${TMP}/.mtplx-qwen-guard-rnd/com.tea.qwen.plist"  # never created (vanished)

# ============================ _resolve_restore_plist ==========================
: > "${TMP}/disc.plist"
eq_plist() { if [[ "$2" == "$3" ]]; then ok "$1"; else bad "$1" "want '$3' got '$2'"; fi; }
eq_plist "discovered gone -> falls back to canonical" \
  "$(bash "${SCRIPT}" --selftest restore-plist "${GUARD_GONE}" "${CANON}")" "${CANON}"
eq_plist "discovered present -> uses discovered" \
  "$(bash "${SCRIPT}" --selftest restore-plist "${TMP}/disc.plist" "${CANON}")" "${TMP}/disc.plist"
eq_plist "neither present -> empty" \
  "$(bash "${SCRIPT}" --selftest restore-plist "${GUARD_GONE}" "${TMP}/nope")" ""

# ============================ _do_restore (restore-run) =======================

# A) guard dir vanished, WAS_LOADED=1 -> bootstraps the CANONICAL plist, rc 0.
_reset_state
GPU_WINDOW_QWEN_PLIST="${CANON}" GPU_WINDOW_RESTORE_QWEN_ALWAYS=1 \
  bash "${SCRIPT}" --selftest restore-run 1 "${GUARD_GONE}" >"${TMP}/a.log" 2>&1
A_RC=$?
if [[ "${A_RC}" -eq 0 ]] && [[ "$(_bootstrapped)" == "${CANON}" ]]; then
  ok "A: guard-dir gone + was_loaded=1 -> bootstraps canonical, rc 0"
else
  bad "A: guard-dir gone -> bootstrap canonical" "rc=${A_RC} bootstrapped='$(_bootstrapped)'; log: $(cat "${TMP}/a.log")"
fi
grep -q "falling back to '${CANON}'" "${TMP}/a.log" \
  && ok "A: logged the fallback from the vanished guard-dir plist" \
  || bad "A: fallback log" "$(cat "${TMP}/a.log")"

# B) RESTORE_QWEN_ALWAYS=1, was_loaded=0, not currently loaded -> bootstraps anyway.
_reset_state
GPU_WINDOW_QWEN_PLIST="${CANON}" GPU_WINDOW_RESTORE_QWEN_ALWAYS=1 \
  bash "${SCRIPT}" --selftest restore-run 0 "" >"${TMP}/b.log" 2>&1
B_RC=$?
if [[ "${B_RC}" -eq 0 ]] && [[ "$(_bootstrapped)" == "${CANON}" ]]; then
  ok "B: RESTORE_QWEN_ALWAYS=1 + not-loaded-at-entry -> bootstraps canonical anyway"
else
  bad "B: RESTORE_QWEN_ALWAYS bootstraps anyway" "rc=${B_RC} bootstrapped='$(_bootstrapped)'; log: $(cat "${TMP}/b.log")"
fi

# C) RESTORE_QWEN_ALWAYS=1, was_loaded=0, ALREADY loaded -> no bootstrap, rc 0.
_reset_state; : > "${TMP}/loaded"   # pretend the service is already up
GPU_WINDOW_QWEN_PLIST="${CANON}" GPU_WINDOW_RESTORE_QWEN_ALWAYS=1 \
  bash "${SCRIPT}" --selftest restore-run 0 "" >"${TMP}/c.log" 2>&1
C_RC=$?
if [[ "${C_RC}" -eq 0 ]] && [[ -z "$(_bootstrapped)" ]]; then
  ok "C: already-loaded -> no bootstrap, rc 0"
else
  bad "C: already-loaded no-op" "rc=${C_RC} bootstrapped='$(_bootstrapped)'"
fi

# D) no plist exists at all, was_loaded=1 -> rc 1 + manual-recovery command printed.
_reset_state
GPU_WINDOW_QWEN_PLIST="${TMP}/missing_canonical.plist" GPU_WINDOW_RESTORE_QWEN_ALWAYS=1 \
  bash "${SCRIPT}" --selftest restore-run 1 "${GUARD_GONE}" >"${TMP}/d.log" 2>&1
D_RC=$?
if [[ "${D_RC}" -ne 0 ]] && [[ -z "$(_bootstrapped)" ]] \
   && grep -q "manual recovery: .*bootstrap ${DOMAIN:-gui/$(id -u)} ${TMP}/missing_canonical.plist" "${TMP}/d.log"; then
  ok "D: no plist exists -> rc!=0, no bootstrap, exact manual command printed"
else
  bad "D: no-plist manual command" "rc=${D_RC} bootstrapped='$(_bootstrapped)'; log: $(cat "${TMP}/d.log")"
fi

# E) RESTORE_QWEN_ALWAYS=0, was_loaded=0 -> leaves stopped, rc 0, no bootstrap.
_reset_state
GPU_WINDOW_QWEN_PLIST="${CANON}" GPU_WINDOW_RESTORE_QWEN_ALWAYS=0 \
  bash "${SCRIPT}" --selftest restore-run 0 "" >"${TMP}/e.log" 2>&1
E_RC=$?
if [[ "${E_RC}" -eq 0 ]] && [[ -z "$(_bootstrapped)" ]] \
   && grep -q "leaving it stopped" "${TMP}/e.log"; then
  ok "E: RESTORE_QWEN_ALWAYS=0 + not-loaded -> left stopped, no bootstrap"
else
  bad "E: ALWAYS=0 leaves stopped" "rc=${E_RC} bootstrapped='$(_bootstrapped)'; log: $(cat "${TMP}/e.log")"
fi

# F) LOW (round 4): was_loaded=1 but the service is ALREADY loaded now (a bootout
#    that failed and never stopped it) -> short-circuit: rc 0, NO bootstrap, NO
#    false "may be DOWN".
_reset_state; : > "${TMP}/loaded"   # already up
GPU_WINDOW_QWEN_PLIST="${CANON}" GPU_WINDOW_RESTORE_QWEN_ALWAYS=1 \
  bash "${SCRIPT}" --selftest restore-run 1 "${CANON}" >"${TMP}/f.log" 2>&1
F_RC=$?
if [[ "${F_RC}" -eq 0 ]] && [[ -z "$(_bootstrapped)" ]] \
   && grep -q "already loaded; verifying API readiness" "${TMP}/f.log" \
   && ! grep -q "may be DOWN" "${TMP}/f.log"; then
  ok "F: was_loaded=1 + already-loaded -> short-circuit, no bootstrap, no false DOWN"
else
  bad "F: was_loaded=1 already-loaded short-circuit" "rc=${F_RC} bootstrapped='$(_bootstrapped)'; log: $(cat "${TMP}/f.log")"
fi

printf '\n%d passed, %d failed\n' "${PASS}" "${FAIL}"
[[ "${FAIL}" -eq 0 ]]
