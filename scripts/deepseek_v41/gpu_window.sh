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
#   4. Run the step (argv) under a SYSTEM-WIDE memory guard.  Before starting it
#      refuses to open if other mtplx/python workers above the foreign cap
#      (default 2 GiB RSS) are resident (prints them).  While it runs, aborts +
#      restores if EITHER the step child's RSS exceeds its cap (default 90 GiB)
#      OR total system used memory (wired+active+compressed) exceeds the ceiling
#      (default 105 GiB) -- the box panicked on 2026-09-10 when the aggregate
#      crossed the box limit while no single child had.
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

# W106 MEDIUM-A: several GiB caps feed bash arithmetic (`GB * 1024^3`, `X * 4`) that
# a fractional/non-integer value would break -- and with `set -u` an unset derived
# var would kill the window at the first poll AFTER Qwen is already booted out.
# Validate them to a non-negative INTEGER BEFORE phase 0 (warn + fall back to the
# default, like KILL_GRACE).  Defined before the config block so it can guard it.
# NOTE: the env var names carry GiB despite the historical `_GB` suffix (see the
# W106 doc "Units" section); the value is GiB (1 GiB = 1024^3 bytes).
_int_or_default() {  # $1=value $2=default $3=env-name -> a valid non-negative int
  if [[ "$1" =~ ^[0-9]+$ ]]; then
    printf '%s' "$1"
  else
    printf '%s [gpu_window] WARN: %s=%s is not a non-negative integer (GiB); using %s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$3" "$1" "$2" >&2
    printf '%s' "$2"
  fi
}

# ------------------------------- configuration -------------------------------
LOCK_PATH="${MTPLX_GPU_LOCK:-/tmp/mtplx-gpu-exclusive.lock}"
LOCK_TIMEOUT="${GPU_WINDOW_LOCK_TIMEOUT:-0}"          # seconds; 0 = block forever
QWEN_LABEL="${GPU_WINDOW_QWEN_LABEL:-com.tea.qwen}"
WIRED_CAP_MB="${GPU_WINDOW_WIRED_CAP_MB:-102400}"     # 100 GiB, never exceeded/raised
STOP_TIMEOUT="${GPU_WINDOW_STOP_TIMEOUT:-180}"        # seconds to confirm the stop
RESTORE_TIMEOUT="${GPU_WINDOW_RESTORE_TIMEOUT:-300}"  # seconds to confirm restore
# W106 restore hardening (real-window incident, windows 42/43): the plist that
# `launchctl print` reports for a running com.tea.qwen is often a TRANSIENT guard-dir
# copy (~/.mtplx-qwen-guard-<rand>/com.tea.qwen.plist) written by mtplx.qwen_guard
# when IT bootstrapped the agent; that dir is gone by restore time, so `bootstrap`
# FAILS and David's agent is left DOWN.  CANONICAL_PLIST is the DURABLE fallback
# (~/Library/LaunchAgents/<label>.plist).  RESTORE_QWEN_ALWAYS bootstraps it at exit
# even if the agent was not loaded at entry, so a previous failed restore cannot
# cascade.  LAUNCHCTL_CMD is overridable so restore is unit-testable with a fake.
CANONICAL_PLIST="${GPU_WINDOW_QWEN_PLIST:-${HOME}/Library/LaunchAgents/${QWEN_LABEL}.plist}"
RESTORE_QWEN_ALWAYS="${GPU_WINDOW_RESTORE_QWEN_ALWAYS:-1}"   # default ON on this box
LAUNCHCTL_CMD="${GPU_WINDOW_LAUNCHCTL_CMD:-/bin/launchctl}"  # overridable for tests
MIN_AVAIL_GB="$(_int_or_default "${GPU_WINDOW_MIN_AVAIL_GB:-100}" 100 GPU_WINDOW_MIN_AVAIL_GB)"  # GiB the step needs available after the stop
# W106 HIGH-2: all guard caps are GiB (bytes = N * 1024^3), stated explicitly.
# Default child-tree RSS cap = 93 GiB ~= 100 GB (David's "100 GB total for
# everything").  This cap is LIVE for the first time (pre-W106 the poll read the
# few-MB `bash -c` shell RSS ~= 0); at 93 GiB it sits just under the 100 GB budget.
CHILD_RSS_CAP_BYTES="${GPU_WINDOW_CHILD_RSS_CAP_BYTES:-$(( 93 * 1024 * 1024 * 1024 ))}"  # 93 GiB ~= 100 GB (David's total budget)
RSS_POLL_SECONDS="${GPU_WINDOW_RSS_POLL_SECONDS:-2}"

# System-wide phase-4 guard (2026-09-10 panic hardening): the box kernel-panicked
# and rebooted when TOTAL used memory crossed the box limit, even though no single
# child breached its own RSS cap.  So phase 4 also polls SYSTEM used memory (wired
# + active + compressed, from vm_stat) and aborts+restores over this ceiling, and
# it REFUSES to open the window while other mtplx/python workers above the foreign
# cap are resident (their footprint co-resides with the step's).
# W106 HIGH-2: GiB. Default system ceiling = 102 GiB ~= 109.5 GB, just under the
# 110 GB hard line (was 105 GiB ~= 112.7 GB, OVER the hard line).
TOTAL_MEM_CEILING_GB="$(_int_or_default "${GPU_WINDOW_TOTAL_MEM_CEILING_GB:-102}" 102 GPU_WINDOW_TOTAL_MEM_CEILING_GB)"  # 102 GiB ~= 109.5 GB, under the 110 GB hard limit
TOTAL_MEM_CEILING_BYTES=$(( TOTAL_MEM_CEILING_GB * 1024 * 1024 * 1024 ))
FOREIGN_WORKER_RSS_GB="$(_int_or_default "${GPU_WINDOW_FOREIGN_WORKER_RSS_GB:-2}" 2 GPU_WINDOW_FOREIGN_WORKER_RSS_GB)"   # GiB: refuse to start if another mtplx/python worker exceeds this RSS
VM_STAT_CMD="${GPU_WINDOW_VM_STAT_CMD:-/usr/bin/vm_stat}"        # overridable so the guard math is unit-testable
PS_CMD="${GPU_WINDOW_PS_CMD:-/bin/ps}"                           # overridable for the foreign-worker scan unit test
                                                                # (also used for the phase-4 step-tree RSS walk below)

# W106 hermetic test mode: GPU_WINDOW_TEST_MODE=1 skips phases 1-3 (the wired-knob
# sysctl read, the launchctl inspect + bootout) and the resident-agent restore --
# i.e. it touches NO sysctl and NO launchctl -- and defaults the exclusive lock to
# a throwaway temp path so tests/test_gpu_window_memory_accounting.sh can exercise
# the phase-4 process-tree RSS accounting against a fake step without touching the
# real GPU lock or the resident agent.  Phase 4 (the polling loop under test) still
# runs.  An explicit MTPLX_GPU_LOCK still wins over the temp default.
GPU_WINDOW_TEST_MODE="${GPU_WINDOW_TEST_MODE:-}"
if [[ "${GPU_WINDOW_TEST_MODE}" == "1" && -z "${MTPLX_GPU_LOCK:-}" ]]; then
  LOCK_PATH="${TMPDIR:-/tmp}/gpu_window_testmode.lock"
fi

UID_NUM="$(id -u)"
DOMAIN="gui/${UID_NUM}"

ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s [gpu_window] %s\n' "$(ts)" "$*"; }
err() { printf '%s [gpu_window] ERROR: %s\n' "$(ts)" "$*" >&2; }
gib() { awk -v b="${1:-0}" 'BEGIN{printf "%.1f", b/1073741824}'; }

# Total *used* physical memory in bytes = (wired down + anonymous + occupied-by-
# compressor) pages * page size, parsed from vm_stat.  Anonymous (not active):
# active includes file-backed page cache, which a 269 GiB mmap'd expert bank
# fills within seconds and macOS reclaims on demand (false abort 2026-09-11).  This is the system-wide
# pressure signal the phase-4 guard aborts on: a runaway allocation ANYWHERE on
# the box (not only the step child) is what panicked the machine on 2026-09-10.
# "occupied by compressor" is the physical compressed footprint (NOT "stored in
# compressor", which is the larger pre-compression logical count).
used_mem_bytes() {
  "${VM_STAT_CMD}" 2>/dev/null | awk '
    /page size of/ { for (i = 1; i <= NF; i++) if ($i == "of") ps = $(i + 1) }
    /^Pages wired down/             { gsub(/\./, "", $NF); wired = $NF }
    /^Anonymous pages/              { gsub(/\./, "", $NF); anon = $NF }
    /^Pages occupied by compressor/ { gsub(/\./, "", $NF); comp = $NF }
    END {
      if (ps == "") ps = 16384
      printf "%.0f", (wired + anon + comp) * ps
    }
  '
}

# Print "<pid> <rssGiB> <command>" for every python/mtplx/mlx process whose RSS
# exceeds the foreign-worker cap, EXCLUDING this script's own pids (the bash
# wrapper and its parent python lock holder).  Used to refuse to open a window
# while another worker session holds gigabytes that would co-reside with the
# step and push the box over the ceiling.
list_heavy_foreign_workers() {
  local self_pids
  self_pids=" $$ ${PPID:-} ${_GPU_WINDOW_HOLDER_PID:-} "
  "${PS_CMD}" -axo pid=,rss=,comm= 2>/dev/null | awk \
    -v cap_kb="$(( FOREIGN_WORKER_RSS_GB * 1024 * 1024 ))" \
    -v self="${self_pids}" '
    {
      pid = $1; rss = $2 + 0;
      cmd = $3; for (i = 4; i <= NF; i++) cmd = cmd " " $i;
      if (rss <= cap_kb) next;
      if (index(self, " " pid " ") > 0) next;
      lc = tolower(cmd);
      if (lc ~ /python|mtplx|mlx/) printf "%s %.1fGiB %s\n", pid, rss / 1048576, cmd;
    }
  '
}

# Walk the process tree rooted at $1 (that pid AND every descendant) from a SINGLE
# `ps` snapshot and print "<sum_rss_kb> <max_rss_kb>": the SUM of RSS across the
# whole tree, and the MAX single-process RSS.  The phase-4 step is launched as a
# `bash -c "..."` chain whose own RSS is a few MB while the python benchmark
# underneath it holds ~65-70 GiB, so polling STEP_PID alone (pre-W106) logged
# "peak step RSS 0.0 GiB" and the child cap could never fire.  A `seen` guard makes
# the walk robust against pid reuse cycles.
_step_tree_rss() {
  local root="${1:-}"
  if [[ -z "${root}" ]]; then printf '0 0'; return; fi
  "${PS_CMD}" -axo pid=,ppid=,rss= 2>/dev/null | awk -v root="${root}" '
    { pid = $1 + 0; ppid = $2 + 0; rss = $3 + 0; RSS[pid] = rss; kids[ppid] = kids[ppid] " " pid }
    END {
      head = 1; tail = 0; wl[++tail] = root + 0; sum = 0; maxp = 0;
      while (head <= tail) {
        p = wl[head]; head++;
        if (p in seen) continue;
        seen[p] = 1;
        if (p in RSS) { sum += RSS[p]; if (RSS[p] > maxp) maxp = RSS[p]; }
        if (p in kids) {
          n = split(kids[p], cc, " ");
          for (i = 1; i <= n; i++) if (cc[i] != "") wl[++tail] = cc[i] + 0;
        }
      }
      printf "%d %d", sum, maxp;
    }
  '
}

# W106 item 4: print every pid in the process tree rooted at $1 (that pid AND every
# descendant), space-separated, root first, from a SINGLE `ps` snapshot.  Used by
# the tree-kill on abort: the step is a `bash -c "a; b; c"` chain whose python
# descendants must ALL be signalled, or an abort of the chain leaves a running
# python orphaned (reparented to launchd) and the next chained command could still
# start.  Snapshot the pids BEFORE signalling (killing reparents/removes members).
# A `seen` guard makes the walk robust against pid-reuse cycles.
_step_tree_pids() {
  local root="${1:-}"
  [[ -n "${root}" ]] || return 0
  "${PS_CMD}" -axo pid=,ppid= 2>/dev/null | awk -v root="${root}" '
    { pid = $1 + 0; ppid = $2 + 0; kids[ppid] = kids[ppid] " " pid }
    END {
      head = 1; tail = 0; wl[++tail] = root + 0; out = "";
      while (head <= tail) {
        p = wl[head]; head++;
        if (p in seen) continue;
        seen[p] = 1;
        out = out " " p;
        if (p in kids) {
          n = split(kids[p], cc, " ");
          for (i = 1; i <= n; i++) if (cc[i] != "") wl[++tail] = cc[i] + 0;
        }
      }
      sub(/^ /, "", out);
      print out;
    }
  '
}

# W106 restore hardening: choose a plist that EXISTS at restore time.  Prefer the
# launchctl-discovered path ($1) if it still exists, else the durable canonical
# plist ($2); echo the chosen path (empty if neither exists).  Defined before the
# selftest block so `--selftest restore-plist` can exercise it hermetically.
_resolve_restore_plist() {
  local discovered="${1:-}" canonical="${2:-}"
  if [[ -n "${discovered}" && -f "${discovered}" ]]; then
    printf '%s' "${discovered}"
  elif [[ -n "${canonical}" && -f "${canonical}" ]]; then
    printf '%s' "${canonical}"
  else
    printf ''
  fi
}

# Core of the resident-agent restore (bootstrap), split out so a fake ${LAUNCHCTL_CMD}
# can unit-test it.  $1 = was_loaded (1/0), $2 = the launchctl-discovered plist path
# (may be a vanished guard-dir copy).  Bootstraps a plist that EXISTS, falling back to
# ${CANONICAL_PLIST}; with RESTORE_QWEN_ALWAYS=1 it bootstraps even when the agent was
# not loaded at entry (so a previous failed restore cannot cascade), unless the agent
# is already loaded now.  On failure it prints the exact manual command.
_do_restore() {
  local was_loaded="${1:-0}" discovered="${2:-}"
  local want=0
  if (( was_loaded == 1 )); then
    want=1
  elif [[ "${RESTORE_QWEN_ALWAYS}" == "1" ]]; then
    if "${LAUNCHCTL_CMD}" print "${DOMAIN}/${QWEN_LABEL}" >/dev/null 2>&1; then
      log "restore: ${QWEN_LABEL} already loaded; nothing to do (RESTORE_QWEN_ALWAYS=1)"
      return 0
    fi
    want=1
    log "restore: ${QWEN_LABEL} was NOT loaded at entry, but GPU_WINDOW_RESTORE_QWEN_ALWAYS=1 -> bootstrapping anyway (guards against a cascaded prior failure)"
  else
    log "restore: ${QWEN_LABEL} was not loaded at entry; leaving it stopped (as found)"
    return 0
  fi

  local plist
  plist="$(_resolve_restore_plist "${discovered}" "${CANONICAL_PLIST}")"
  if [[ -n "${discovered}" && "${discovered}" != "${plist}" ]]; then
    log "restore: discovered plist '${discovered}' is gone; falling back to '${plist:-<none>}'"
  fi
  if [[ -z "${plist}" ]]; then
    err "restore: NO plist file exists to bootstrap (discovered '${discovered}' gone, canonical '${CANONICAL_PLIST}' missing); ${QWEN_LABEL} may be DOWN -- manual recovery: ${LAUNCHCTL_CMD} bootstrap ${DOMAIN} ${CANONICAL_PLIST}"
    return 1
  fi

  log "restore: launchctl bootstrap ${DOMAIN} ${plist}"
  if ! "${LAUNCHCTL_CMD}" bootstrap "${DOMAIN}" "${plist}"; then
    err "restore: 'launchctl bootstrap ${DOMAIN} ${plist}' FAILED; ${QWEN_LABEL} may be DOWN on :8080 -- manual recovery: ${LAUNCHCTL_CMD} bootstrap ${DOMAIN} ${CANONICAL_PLIST}"
    return 1
  fi
  local deadline
  deadline=$(( $(date +%s) + RESTORE_TIMEOUT ))
  while (( $(date +%s) < deadline )); do
    if "${LAUNCHCTL_CMD}" print "${DOMAIN}/${QWEN_LABEL}" >/dev/null 2>&1; then
      log "restore: ${QWEN_LABEL} is loaded again (launchctl service present); /health may still be warming"
      return 0
    fi
    sleep 1
  done
  err "restore: ${QWEN_LABEL} did not reappear within ${RESTORE_TIMEOUT}s; verify :8080 manually (${LAUNCHCTL_CMD} bootstrap ${DOMAIN} ${CANONICAL_PLIST})"
  return 1
}

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# ---- test/introspection hooks (no GPU, no lock, no launchctl) ----------------
# Let a shell unit test exercise the pure guard math against fake vm_stat / ps
# output without opening a window.  Runs BEFORE the lock phase and exits.
if [[ "${1:-}" == "--selftest" ]]; then
  shift
  case "${1:-}" in
    used-mem-bytes) used_mem_bytes; echo ;;
    used-mem-gib)   gib "$(used_mem_bytes)"; echo ;;
    over-ceiling)
      _u="$(used_mem_bytes)"
      if [[ "${_u}" =~ ^[0-9]+$ ]] && (( _u > TOTAL_MEM_CEILING_BYTES )); then
        echo yes
      else
        echo no
      fi
      ;;
    heavy-workers)  list_heavy_foreign_workers ;;
    tree-pids)      _step_tree_pids "${2:-}" ; echo ;;   # W106 item 4: tree walk
    restore-plist)  _resolve_restore_plist "${2:-}" "${3:-}" ; echo ;;  # discovered, canonical
    restore-run)
      # W106 restore test: run _do_restore against a fake ${LAUNCHCTL_CMD} with
      # env-injected inputs, then exit with its status.  $2 = was_loaded (1/0),
      # $3 = discovered plist path (may be a vanished guard-dir copy).
      _do_restore "${2:-0}" "${3:-}"
      exit $?
      ;;
    *) err "unknown --selftest target: ${1:-} (used-mem-bytes|used-mem-gib|over-ceiling|heavy-workers|tree-pids|restore-plist|restore-run)"; exit 2 ;;
  esac
  exit 0
fi

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

# Ignore signals HERE (in the lock holder) so a group Ctrl-C / a TERM to the holder
# does not kill it before the child bash has restored qwen and released the lock.
_ABORT_SIGS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
for signum in _ABORT_SIGS:
    try:
        signal.signal(signum, signal.SIG_IGN)
    except (ValueError, OSError):
        pass


# W106 abort fix (real-window incident, window 42): a signal that is SIG_IGN at
# bash startup CANNOT be trapped ("signals ignored on entry to a non-interactive
# shell cannot be trapped or reset") -- so the child bash, inheriting the holder's
# SIG_IGN, silently ignored TERM and its teardown trap never fired.  RESET
# INT/TERM/HUP to SIG_DFL in the child (after fork, before exec) so the child bash
# starts with the default disposition and its `trap` installs.  The holder itself
# stays ignoring them (above), so the operator TERMs the child bash (the abort
# recipe prints its pid), not the holder.
def _reset_child_signals():  # runs in the child between fork and exec
    for _s in _ABORT_SIGS:
        try:
            signal.signal(_s, signal.SIG_DFL)
        except (ValueError, OSError):
            pass


env = dict(os.environ)
env["_GPU_WINDOW_LOCKED"] = "1"
child = subprocess.Popen(
    ["/bin/bash", script, *step], env=env, preexec_fn=_reset_child_signals
)
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

# W106 item 4: grace period (seconds) between the tree-wide TERM and the KILL of
# any survivor.  Overridable so the tree-kill unit test does not wait the full 2 s.
# LOW-2: must be a non-negative INTEGER (it drives bash arithmetic `KILL_GRACE
# _SECONDS * 4`); a non-integer would break the loop, so fall back to 2.
KILL_GRACE_SECONDS="${GPU_WINDOW_KILL_GRACE_SECONDS:-2}"
if [[ ! "${KILL_GRACE_SECONDS}" =~ ^[0-9]+$ ]]; then
  err "GPU_WINDOW_KILL_GRACE_SECONDS='${KILL_GRACE_SECONDS}' is not a non-negative integer; using 2"
  KILL_GRACE_SECONDS=2
fi

_kill_step_tree() {
  # W106 item 4: TERM then (after a grace) KILL the ENTIRE process tree rooted at
  # STEP_PID -- the `bash -c "..."` chain AND every python/sleep descendant -- so
  # an aborted step can NEVER start its next chained command and no grandchild is
  # left orphaned (reparented to launchd) still holding GPU/host memory.  The pids
  # are snapshotted BEFORE any signal (killing reparents/removes tree members), and
  # signalled by pid so reparenting mid-teardown does not let one escape.  Reaps
  # STEP_PID (a job of this shell) and clears it so the teardown trap does not
  # re-run; the restore of the resident agent then runs from the EXIT trap.
  [[ -n "${STEP_PID}" ]] || return 0
  local pids _p _i _alive
  pids="$(_step_tree_pids "${STEP_PID}")"
  [[ -n "${pids}" ]] || pids="${STEP_PID}"
  for _p in ${pids}; do
    kill -TERM "${_p}" 2>/dev/null || true
  done
  # Poll for the whole tree to exit, up to KILL_GRACE_SECONDS (0.25 s cadence).
  for (( _i = 0; _i < KILL_GRACE_SECONDS * 4; _i++ )); do
    _alive=0
    for _p in ${pids}; do
      if kill -0 "${_p}" 2>/dev/null; then _alive=1; break; fi
    done
    (( _alive == 0 )) && break
    sleep 0.25
  done
  # KILL any survivor of the grace period.
  for _p in ${pids}; do
    if kill -0 "${_p}" 2>/dev/null; then
      kill -KILL "${_p}" 2>/dev/null || true
    fi
  done
  wait "${STEP_PID}" 2>/dev/null || true
  STEP_PID=""
}

# Back-compat alias for the phase-4 abort call sites (RSS cap / system ceiling).
_kill_step_child() { _kill_step_tree; }

WAS_LOADED=0
RESTORED=0
STEP_PID=""
PEAK_TREE_RSS_BYTES=0       # running peak of the SUM of RSS across the step process tree
PEAK_MAX_PROC_RSS_BYTES=0   # running peak of the MAX single-process RSS in that tree
PEAK_SYSTEM_USED_BYTES=0    # running peak of system used memory (NOT the value at exit)
PLIST=""
QWEN_PID=""

restore_qwen() {
  # Teardown entry: guard against double-restore + TEST MODE (never touch launchctl
  # in tests), then delegate to _do_restore, which chooses a plist that EXISTS
  # (falling back to the durable CANONICAL_PLIST when the launchctl-discovered guard
  # -dir copy is gone) and, with RESTORE_QWEN_ALWAYS=1, bootstraps even if the agent
  # was not loaded at entry so a prior failed restore cannot cascade.
  if (( RESTORED == 1 )); then return; fi
  RESTORED=1
  if [[ "${GPU_WINDOW_TEST_MODE}" == "1" ]]; then
    log "restore: TEST MODE -- skipping (no launchctl)"
    return
  fi
  _do_restore "${WAS_LOADED}" "${PLIST}"
}

teardown() {
  local ec=$?
  trap - EXIT INT TERM
  # W106 item 4: a TERM/INT to the wrapper (or any non-abort exit with the step
  # still running) tree-kills the WHOLE step process tree, not just STEP_PID, so a
  # `bash -c` chain never starts its next step and no python descendant survives.
  if [[ -n "${STEP_PID}" ]] && kill -0 "${STEP_PID}" 2>/dev/null; then
    log "teardown: terminating step process tree (root pid=${STEP_PID}): $(_step_tree_pids "${STEP_PID}")"
    _kill_step_tree
  fi
  restore_qwen
  exit "${ec}"
}

# W106 abort item (a): INT/TERM must actually abort while bash is in the phase-4
# poll loop.  bash DEFERS a heavy trap until the running foreground command
# (`sleep`, `wait`) returns, so the old `trap teardown INT TERM` could sit for
# many seconds while the step kept loading.  Instead the signal handler only SETS A
# FLAG (async-safe, instant); the phase-3/4 loops check it every poll and abort
# promptly.  teardown still runs from the EXIT trap (so restore always happens).
_ABORT_SIGNAL=0
_on_abort_signal() {
  _ABORT_SIGNAL=1
  log "abort: INT/TERM received; aborting at the next poll (<= ${RSS_POLL_SECONDS}s)"
}
trap teardown EXIT
trap _on_abort_signal INT TERM

# Called at the top of the phase-3/4 loops: if an abort signal came in, kill the
# step tree (if any) and exit -> the EXIT trap restores the agent + releases the lock.
_check_abort() {
  (( _ABORT_SIGNAL )) || return 0
  err "phase 4: abort requested (INT/TERM); killing the step tree and restoring"
  _kill_step_tree
  exit 9
}

# W106 (real-window incident): print the ABORT RECIPE + restore plist up front, so
# an operator aborting by hand signals the RIGHT pid.  The trap owner is THIS bash
# gpu_window.sh process ($$); its parent (the python fcntl lock-holder) IGNORES
# INT/TERM/HUP by design, so `kill -TERM <parent>` does nothing.
log "abort: to abort this window cleanly, kill -TERM $$ (this bash gpu_window.sh pid); the parent python lock-holder ignores signals. The trap tree-kills the step and restores ${QWEN_LABEL}."
log "restore: on exit ${QWEN_LABEL} is bootstrapped from ${CANONICAL_PLIST} (RESTORE_QWEN_ALWAYS=${RESTORE_QWEN_ALWAYS}); a vanished launchctl-discovered guard-dir plist falls back to this path."

if [[ "${GPU_WINDOW_TEST_MODE}" == "1" ]]; then
  log "TEST MODE: skipping phases 1-3 (wired-knob sysctl read, launchctl inspect/bootout) and the resident-agent restore; lock=${LOCK_PATH}"
else
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
fi  # end phases 1-2 (skipped whole in GPU_WINDOW_TEST_MODE=1; phase 3 below is
    # then auto-skipped because WAS_LOADED stays 0, as is restore_qwen)

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
  MIN_AVAIL_BYTES=$(( MIN_AVAIL_GB * 1024 * 1024 * 1024 ))
  deadline=$(( $(date +%s) + STOP_TIMEOUT ))
  pid_gone=0
  freed=0
  while (( $(date +%s) < deadline )); do
    _check_abort   # W106 (a): abort promptly even during the bootout wait
    if (( pid_gone == 0 )); then
      if [[ -z "${QWEN_PID}" ]] || ! kill -0 "${QWEN_PID}" 2>/dev/null; then
        pid_gone=1
        log "phase 3: ${QWEN_LABEL} pid ${QWEN_PID:-<none>} is gone"
      fi
    fi
    AVAIL_NOW="$(avail_bytes)"
    freed=$(( AVAIL_NOW - AVAIL_BEFORE ))
    if (( pid_gone == 1 && AVAIL_NOW >= MIN_AVAIL_BYTES )); then
      log "phase 3: stop confirmed -- pid gone, $(gib "${freed}") GiB freed, $(gib "${AVAIL_NOW}") GiB available (>= ${MIN_AVAIL_GB} GiB)"
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
  if (( AVAIL_NOW < MIN_AVAIL_BYTES )); then
    err "phase 3: only $(gib "${AVAIL_NOW}") GiB available after stop ($(gib "${freed}") GiB freed; need >= ${MIN_AVAIL_GB} GiB available); aborting"
    exit 5
  fi
fi

# ---------------- phase 4: run the step under the SYSTEM-WIDE memory guard -----
# Pre-flight: refuse to open the window while another mtplx/python worker is
# holding more than the foreign cap.  Its footprint co-resides with the step, so
# starting here risks pushing the box over the ceiling (2026-09-10 panic).
log "phase 4: scanning for other heavy mtplx/python workers (> ${FOREIGN_WORKER_RSS_GB} GiB RSS) before opening the window"
HEAVY_WORKERS="$(list_heavy_foreign_workers)"
if [[ -n "${HEAVY_WORKERS}" ]]; then
  err "phase 4: REFUSING to start -- other mtplx/python workers above ${FOREIGN_WORKER_RSS_GB} GiB RSS are resident (they co-reside with the step and could panic the box):"
  printf '%s\n' "${HEAVY_WORKERS}" | while IFS= read -r _hw_line; do
    err "    ${_hw_line}"
  done
  exit 7
fi
USED_START="$(used_mem_bytes)"
log "phase 4: system used memory at start: $(gib "${USED_START}") GiB (ceiling ${TOTAL_MEM_CEILING_GB} GiB)"

# W106 MEDIUM-B: relate the child-tree cap to the measured baseline.  If the step
# grew to the full CHILD_RSS_CAP on top of what is ALREADY used, the box would
# cross the system ceiling before the per-child cap ever fired.  So the EFFECTIVE
# child-tree cap is min(CHILD_RSS_CAP, ceiling - used_start): the most the step can
# add without crossing the ceiling.  We LOWER (never raise) it, and never refuse.
EFFECTIVE_CHILD_CAP_BYTES="${CHILD_RSS_CAP_BYTES}"
if [[ "${USED_START:-}" =~ ^[0-9]+$ ]] && \
   (( USED_START + CHILD_RSS_CAP_BYTES > TOTAL_MEM_CEILING_BYTES )); then
  _headroom=$(( TOTAL_MEM_CEILING_BYTES - USED_START ))
  (( _headroom < 0 )) && _headroom=0
  EFFECTIVE_CHILD_CAP_BYTES="${_headroom}"
  log "phase 4: effective child-tree cap $(gib "${EFFECTIVE_CHILD_CAP_BYTES}") GiB (lowered from $(gib "${CHILD_RSS_CAP_BYTES}") GiB: used_start $(gib "${USED_START}") + cap would cross the ${TOTAL_MEM_CEILING_GB} GiB ceiling)"
fi

# W106 HIGH-2: state BOTH caps explicitly (GiB + the ~GB equivalent) at step start
# so the operator sees the guard envelope next to the step it is about to run.
log "phase 4: guard caps -- child-tree RSS cap $(gib "${EFFECTIVE_CHILD_CAP_BYTES}") GiB (~$(awk -v b="${EFFECTIVE_CHILD_CAP_BYTES}" 'BEGIN{printf "%.0f", b/1e9}') GB); system used ceiling ${TOTAL_MEM_CEILING_GB} GiB (~$(awk -v g="${TOTAL_MEM_CEILING_GB}" 'BEGIN{printf "%.1f", g*1073741824/1e9}') GB, under the 110 GB hard limit)"
log "phase 4: starting GPU step under child-tree RSS cap $(gib "${EFFECTIVE_CHILD_CAP_BYTES}") GiB + system ceiling ${TOTAL_MEM_CEILING_GB} GiB: $*"
"$@" &
STEP_PID=$!
# Seed the system-used running peak with the at-start reading so PEAK_SYSTEM_USED
# is a true max over the window (the pre-W106 exit line re-read used_mem_bytes and
# reported the value AT EXIT, not the peak).
if [[ "${USED_START:-}" =~ ^[0-9]+$ ]]; then
  PEAK_SYSTEM_USED_BYTES="${USED_START}"
fi
_last_mem_sample=0  # 0 => the first poll logs an envelope sample immediately
while :; do
  _check_abort   # W106 (a): abort promptly on a queued INT/TERM (not deferred)
  # Loop terminates when STEP_PID is gone or a zombie (same condition as before);
  # RSS is now measured over its whole tree, not this one (near-empty) pid.
  step_state="$("${PS_CMD}" -o state= -p "${STEP_PID}" 2>/dev/null | tr -d ' \t\n')"
  if [[ -z "${step_state}" || "${step_state}" == Z* ]]; then
    break
  fi
  # Walk STEP_PID + all descendants: SUM RSS across the tree (the python benchmark
  # under the bash-c chain) and the MAX single process.  The cap applies to the SUM.
  read -r tree_kb max_kb < <(_step_tree_rss "${STEP_PID}")
  tree_bytes=$(( ${tree_kb:-0} * 1024 ))
  max_bytes=$(( ${max_kb:-0} * 1024 ))
  if (( tree_bytes > PEAK_TREE_RSS_BYTES )); then
    PEAK_TREE_RSS_BYTES=${tree_bytes}
  fi
  if (( max_bytes > PEAK_MAX_PROC_RSS_BYTES )); then
    PEAK_MAX_PROC_RSS_BYTES=${max_bytes}
  fi
  if (( tree_bytes > EFFECTIVE_CHILD_CAP_BYTES )); then
    err "phase 4: step tree RSS $(gib "${tree_bytes}") GiB (max single process $(gib "${max_bytes}") GiB) exceeded cap $(gib "${EFFECTIVE_CHILD_CAP_BYTES}") GiB; killing child and restoring"
    _kill_step_child
    exit 6
  fi
  # System-wide guard: a runaway allocation anywhere on the box (not only this
  # child) that crosses the ceiling aborts the step and restores the agent.
  used_now="$(used_mem_bytes)"
  if [[ "${used_now:-}" =~ ^[0-9]+$ ]]; then
    if (( used_now > PEAK_SYSTEM_USED_BYTES )); then
      PEAK_SYSTEM_USED_BYTES=${used_now}
    fi
    if (( used_now > TOTAL_MEM_CEILING_BYTES )); then
      err "phase 4: SYSTEM used memory $(gib "${used_now}") GiB exceeded ceiling $(gib "${TOTAL_MEM_CEILING_BYTES}") GiB (step tree RSS $(gib "${PEAK_TREE_RSS_BYTES}") GiB); killing child and restoring"
      _kill_step_child
      exit 8
    fi
  fi
  # One memory-envelope sample every 30 s (and once on the first poll) so the log
  # shows the real footprint of the step as it runs, not just the exit summary.
  _now_epoch="$(date +%s)"
  if (( _now_epoch - _last_mem_sample >= 30 )); then
    log "phase 4: mem sample -- step tree RSS $(gib "${tree_bytes}") GiB (max single process $(gib "${max_bytes}") GiB), system used $(gib "${used_now:-0}") GiB"
    _last_mem_sample=${_now_epoch}
  fi
  sleep "${RSS_POLL_SECONDS}"
done
wait "${STEP_PID}"
step_rc=$?
STEP_PID=""
log "phase 4: GPU step exited with code ${step_rc}; peak step tree RSS $(gib "${PEAK_TREE_RSS_BYTES}") GiB (max single process $(gib "${PEAK_MAX_PROC_RSS_BYTES}") GiB), peak system used $(gib "${PEAK_SYSTEM_USED_BYTES}") GiB"

# phase 5 (restore + lock release) runs in the teardown trap on this exit.
exit "${step_rc}"
