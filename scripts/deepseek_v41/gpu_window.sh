#!/usr/bin/env bash
# Guarded GPU window for the first real-model DeepSeek-V4.1-Flash q2 runs.
#
# Runs ONE step (its argv) inside a serialized exclusive-GPU window and restores
# the resident agent (com.tea.qwen on :8080) exactly on exit -- including on
# failure, timeout, memory-guard kill, or Ctrl-C.
#
# Phases (each prints a UTC-timestamped line):
#   0. Acquire the exclusive GPU lock (fcntl advisory lock on
#      /tmp/mtplx-gpu-exclusive.lock -- the SAME lock mtplx.qwen_guard takes via
#      fcntl.flock; macOS ships no flock(1)).  BLOCKING wait with a printed queue
#      notice; it NEVER SIGSTOP/SIGKILLs whoever holds the lock (see
#      memory/never-signal-flock-queue.md).
#   1. Verify the wired-memory knob (iogpu.wired_limit_mb) is in (0, 100 GiB].
#      READ ONLY -- this script NEVER writes the sysctl and never raises it
#      (memory/never-exceed-the-memory-knob.md, qwen-serve-crash-loop-guards.md).
#   2. Discover the resident agent's plist from `launchctl print` and capture its
#      pid + whether it is loaded.
#   3. `launchctl bootout gui/<uid>/com.tea.qwen` (NO kickstart -- kickstart -k
#      races the port yield and leaves it down; memory/guarded-window-launch-
#      protocol.md).  Poll until its pid is gone AND free memory rises by the
#      expected resident-agent release, or fail loudly (no hidden retries).
#   4. Run the step (argv).  Memory guard: if the step child's RSS exceeds the
#      cap (default 100 GiB), kill it and restore.
#   5. EXIT/INT/TERM trap: `launchctl bootstrap gui/<uid> <plist>` to restore the
#      resident agent, then the lock is released by the fcntl holder (phase 0).
#      Qwen is restored BEFORE the lock is released, so a queued window never
#      acquires while the reload races.
#
# Launch rules (memory/guarded-window-launch-protocol.md): launch as the DIRECT
# command of a run_in_background:true Bash call -- NEVER a shell `&` one-liner,
# which reaps the process group and leaves qwen down.  Do NOT stop the launching
# agent mid-window (it kills this wrapper AND its lock holder).
#
# Borrowed shape:
#   - lock path + fcntl advisory lock:     mtplx/qwen_guard.py:27,408 (LOCK_EX)
#   - bootout (not kickstart) / bootstrap: mtplx/qwen_guard.py:1136,1078
#   - read-only lsof/no-signal lock rule:  tools/mixofficial_governor.sh:19,29
#   - hold-lock-run-child-restore flow:    scripts/run_with_qwen_stopped.py
#
# NOTE: this wrapper does not itself execute any MLX/GPU code -- the step's argv
# does.  It is the harness the orchestrator launches; it is NOT run by the
# harness author.

set -uo pipefail  # intentionally NOT -e: exit codes are managed explicitly so
                  # the teardown trap always runs and restores the resident agent.

# ------------------------------- configuration -------------------------------
LOCK_PATH="${MTPLX_GPU_LOCK:-/tmp/mtplx-gpu-exclusive.lock}"
LOCK_TIMEOUT="${GPU_WINDOW_LOCK_TIMEOUT:-0}"          # seconds; 0 = block forever
QWEN_LABEL="${GPU_WINDOW_QWEN_LABEL:-com.tea.qwen}"
WIRED_CAP_MB="${GPU_WINDOW_WIRED_CAP_MB:-102400}"     # 100 GiB, never exceeded/raised
STOP_TIMEOUT="${GPU_WINDOW_STOP_TIMEOUT:-180}"        # seconds to confirm the stop
RESTORE_TIMEOUT="${GPU_WINDOW_RESTORE_TIMEOUT:-300}"  # seconds to confirm restore
MIN_FREED_GB="${GPU_WINDOW_MIN_FREED_GB:-60}"         # expect ~86 GiB; gate below it
CHILD_RSS_CAP_BYTES="${GPU_WINDOW_CHILD_RSS_CAP_BYTES:-$(( 100 * 1024 * 1024 * 1024 ))}"
RSS_POLL_SECONDS="${GPU_WINDOW_RSS_POLL_SECONDS:-2}"

UID_NUM="$(id -u)"
DOMAIN="gui/${UID_NUM}"

ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s [gpu_window] %s\n' "$(ts)" "$*"; }
err() { printf '%s [gpu_window] ERROR: %s\n' "$(ts)" "$*" >&2; }
gib() { awk -v b="${1:-0}" 'BEGIN{printf "%.1f", b/1073741824}'; }

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

if [[ $# -lt 1 ]]; then
  err "usage: gpu_window.sh <step> [args...]   (the step is run inside the guarded GPU window)"
  exit 2
fi

# ---------------- phase 0: acquire the exclusive GPU lock, then re-exec --------
# macOS has no flock(1); hold the same fcntl advisory lock mtplx.qwen_guard uses.
# The Python holder keeps the lock fd open for the whole window and releases it
# only after this wrapper (its child) has restored the resident agent and exited.
if [[ -z "${_GPU_WINDOW_LOCKED:-}" ]]; then
  exec /usr/bin/env python3 - "$LOCK_PATH" "$LOCK_TIMEOUT" "$SCRIPT_PATH" "$@" <<'PYHOLDER'
import errno, fcntl, os, signal, subprocess, sys, time

lock_path, timeout, script = sys.argv[1], float(sys.argv[2]), sys.argv[3]
step = sys.argv[4:]


def log(msg):
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"{stamp} [gpu_window] {msg}", flush=True)


fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
deadline = None if timeout <= 0 else time.monotonic() + timeout
queued = False
while True:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        break
    except OSError as exc:
        if exc.errno not in (errno.EAGAIN, errno.EACCES, getattr(errno, "EWOULDBLOCK", errno.EAGAIN)):
            raise
        if not queued:
            log(f"GPU lock held by another window; QUEUED (blocking wait, will NOT signal the holder) lock={lock_path}")
            queued = True
        if deadline is not None and time.monotonic() >= deadline:
            log(f"ERROR: timed out after {timeout:.0f}s waiting for the exclusive GPU lock {lock_path}")
            os.close(fd)
            sys.exit(75)
        time.sleep(0.5)
log(f"acquired exclusive GPU lock: {lock_path}")

# Ignore signals here so a group Ctrl-C reaches the child, which does the ordered
# teardown (restore qwen) before we release the lock.
for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    try:
        signal.signal(signum, signal.SIG_IGN)
    except (ValueError, OSError):
        pass

env = dict(os.environ)
env["_GPU_WINDOW_LOCKED"] = "1"
child = subprocess.Popen(["/bin/bash", script, *step], env=env)
try:
    rc = child.wait()
finally:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    os.close(fd)
    log(f"released exclusive GPU lock: {lock_path}")
if rc is None:
    rc = 1
sys.exit(rc if rc >= 0 else 128 - rc)
PYHOLDER
fi

# ================= from here on, the exclusive lock is held ====================

avail_bytes() {
  # Best-effort "available" memory: free + inactive + speculative + purgeable
  # pages * page size.  A resident-agent unload shows up here as a large rise.
  /usr/bin/vm_stat 2>/dev/null | awk '
    /page size of/ { for (i = 1; i <= NF; i++) if ($i == "of") ps = $(i + 1) }
    /^Pages free/         { gsub(/\./, "", $3); free = $3 }
    /^Pages inactive/     { gsub(/\./, "", $3); inact = $3 }
    /^Pages speculative/  { gsub(/\./, "", $3); spec = $3 }
    /^Pages purgeable/    { gsub(/\./, "", $3); purg = $3 }
    END { if (ps == "") ps = 16384; printf "%.0f", (free + inact + spec + purg) * ps }
  '
}

WAS_LOADED=0
RESTORED=0
STEP_PID=""
PEAK_RSS_BYTES=0
PLIST=""
QWEN_PID=""

restore_qwen() {
  if (( RESTORED == 1 )); then return; fi
  RESTORED=1
  if (( WAS_LOADED == 0 )); then
    log "restore: ${QWEN_LABEL} was not loaded at entry; leaving it stopped (as found)"
    return
  fi
  log "restore: launchctl bootstrap ${DOMAIN} ${PLIST}"
  if ! /bin/launchctl bootstrap "${DOMAIN}" "${PLIST}"; then
    err "restore: 'launchctl bootstrap ${DOMAIN} ${PLIST}' FAILED; ${QWEN_LABEL} may be DOWN on :8080 -- manual recovery required"
    return
  fi
  local deadline
  deadline=$(( $(date +%s) + RESTORE_TIMEOUT ))
  while (( $(date +%s) < deadline )); do
    if /bin/launchctl print "${DOMAIN}/${QWEN_LABEL}" >/dev/null 2>&1; then
      log "restore: ${QWEN_LABEL} is loaded again (launchctl service present); /health may still be warming"
      return
    fi
    sleep 1
  done
  err "restore: ${QWEN_LABEL} did not reappear within ${RESTORE_TIMEOUT}s; verify :8080 manually"
}

teardown() {
  local ec=$?
  trap - EXIT INT TERM
  if [[ -n "${STEP_PID}" ]] && kill -0 "${STEP_PID}" 2>/dev/null; then
    log "teardown: terminating step child pid=${STEP_PID}"
    kill -TERM "${STEP_PID}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${STEP_PID}" 2>/dev/null || break
      sleep 0.25
    done
    if kill -0 "${STEP_PID}" 2>/dev/null; then
      log "teardown: SIGKILL step child pid=${STEP_PID}"
      kill -KILL "${STEP_PID}" 2>/dev/null || true
    fi
    wait "${STEP_PID}" 2>/dev/null || true
  fi
  restore_qwen
  exit "${ec}"
}
trap teardown EXIT INT TERM

# ---------------- phase 1: verify the wired-memory knob (read only) -----------
log "phase 1: verifying iogpu.wired_limit_mb is in (0, ${WIRED_CAP_MB}] MB (<= 100 GiB); never raising it"
WIRED_MB="$(/usr/sbin/sysctl -n iogpu.wired_limit_mb 2>/dev/null || true)"
if [[ -z "${WIRED_MB}" || ! "${WIRED_MB}" =~ ^[0-9]+$ ]]; then
  err "phase 1: could not read iogpu.wired_limit_mb (got '${WIRED_MB}'); refusing to open a GPU window"
  exit 3
fi
if (( WIRED_MB == 0 || WIRED_MB > WIRED_CAP_MB )); then
  err "phase 1: iogpu.wired_limit_mb=${WIRED_MB} MB is not in (0, ${WIRED_CAP_MB}] -- the knob must be <= 100 GiB and this script never raises it; refusing"
  exit 3
fi
log "phase 1: wired-memory knob OK: iogpu.wired_limit_mb=${WIRED_MB} MB ($(( WIRED_MB / 1024 )) GiB); left UNCHANGED"

# ---------------- phase 2: discover the resident agent's plist + pid ----------
log "phase 2: inspecting ${DOMAIN}/${QWEN_LABEL} via launchctl print"
PRINT_OUT="$(/bin/launchctl print "${DOMAIN}/${QWEN_LABEL}" 2>/dev/null || true)"
if [[ -n "${PRINT_OUT}" ]]; then
  WAS_LOADED=1
  PLIST="$(printf '%s\n' "${PRINT_OUT}" | awk -F ' = ' '/^[[:space:]]*path = /{print $2; exit}')"
  QWEN_PID="$(printf '%s\n' "${PRINT_OUT}" | awk -F ' = ' '/^[[:space:]]*pid = /{print $2; exit}')"
fi
PLIST="${GPU_WINDOW_QWEN_PLIST:-${PLIST:-${HOME}/Library/LaunchAgents/${QWEN_LABEL}.plist}}"
if (( WAS_LOADED == 0 )); then
  log "phase 2: ${QWEN_LABEL} is NOT loaded at entry; nothing to boot out; it will be left stopped on exit"
else
  log "phase 2: ${QWEN_LABEL} loaded (pid=${QWEN_PID:-unknown}); plist=${PLIST}"
fi

# ---------------- phase 3: bootout + confirm the release ----------------------
if (( WAS_LOADED == 1 )); then
  AVAIL_BEFORE="$(avail_bytes)"
  log "phase 3: available memory before bootout: $(gib "${AVAIL_BEFORE}") GiB"
  log "phase 3: launchctl bootout ${DOMAIN}/${QWEN_LABEL} (NO kickstart)"
  if ! /bin/launchctl bootout "${DOMAIN}/${QWEN_LABEL}" 2>/dev/null; then
    # Fall back to the domain + plist path spelling of bootout.
    if ! /bin/launchctl bootout "${DOMAIN}" "${PLIST}" 2>/dev/null; then
      err "phase 3: 'launchctl bootout ${QWEN_LABEL}' FAILED"
      exit 5
    fi
  fi
  MIN_FREED_BYTES=$(( MIN_FREED_GB * 1024 * 1024 * 1024 ))
  deadline=$(( $(date +%s) + STOP_TIMEOUT ))
  pid_gone=0
  freed=0
  while (( $(date +%s) < deadline )); do
    if (( pid_gone == 0 )); then
      if [[ -z "${QWEN_PID}" ]] || ! kill -0 "${QWEN_PID}" 2>/dev/null; then
        pid_gone=1
        log "phase 3: ${QWEN_LABEL} pid ${QWEN_PID:-<none>} is gone"
      fi
    fi
    AVAIL_NOW="$(avail_bytes)"
    freed=$(( AVAIL_NOW - AVAIL_BEFORE ))
    if (( pid_gone == 1 && freed >= MIN_FREED_BYTES )); then
      log "phase 3: stop confirmed -- pid gone and $(gib "${freed}") GiB freed (>= ${MIN_FREED_GB} GiB)"
      break
    fi
    sleep 1
  done
  if (( pid_gone == 0 )); then
    err "phase 3: ${QWEN_LABEL} pid ${QWEN_PID} still alive after ${STOP_TIMEOUT}s; aborting (no hidden retries)"
    exit 5
  fi
  AVAIL_NOW="$(avail_bytes)"
  freed=$(( AVAIL_NOW - AVAIL_BEFORE ))
  if (( freed < MIN_FREED_BYTES )); then
    err "phase 3: only $(gib "${freed}") GiB freed after stop (need >= ${MIN_FREED_GB} GiB); memory did not reflect the resident-agent release; aborting"
    exit 5
  fi
fi

# ---------------- phase 4: run the step under the RSS memory guard ------------
log "phase 4: starting GPU step under RSS cap $(gib "${CHILD_RSS_CAP_BYTES}") GiB: $*"
"$@" &
STEP_PID=$!
while :; do
  read -r rss_kb state < <(ps -o rss=,state= -p "${STEP_PID}" 2>/dev/null || true)
  if [[ -z "${state:-}" || "${state}" == Z* ]]; then
    break
  fi
  if [[ "${rss_kb:-}" =~ ^[0-9]+$ ]]; then
    rss_bytes=$(( rss_kb * 1024 ))
    if (( rss_bytes > PEAK_RSS_BYTES )); then
      PEAK_RSS_BYTES=${rss_bytes}
    fi
    if (( rss_bytes > CHILD_RSS_CAP_BYTES )); then
      err "phase 4: step child RSS $(gib "${rss_bytes}") GiB exceeded cap $(gib "${CHILD_RSS_CAP_BYTES}") GiB; killing child and restoring"
      kill -TERM "${STEP_PID}" 2>/dev/null || true
      sleep 2
      if kill -0 "${STEP_PID}" 2>/dev/null; then
        kill -KILL "${STEP_PID}" 2>/dev/null || true
      fi
      wait "${STEP_PID}" 2>/dev/null || true
      STEP_PID=""
      exit 6
    fi
  fi
  sleep "${RSS_POLL_SECONDS}"
done
wait "${STEP_PID}"
step_rc=$?
STEP_PID=""
log "phase 4: GPU step exited with code ${step_rc}; peak step RSS $(gib "${PEAK_RSS_BYTES}") GiB"

# phase 5 (restore + lock release) runs in the teardown trap on this exit.
exit "${step_rc}"
