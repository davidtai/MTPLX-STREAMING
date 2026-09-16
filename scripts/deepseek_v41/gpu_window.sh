#!/usr/bin/env bash
# Guarded GPU window for the first real-model DeepSeek-V4.1-Flash q2 runs.
#
# Runs ONE step (its argv) inside a serialized exclusive-GPU window and restores
# the resident agent (com.tea.qwen on :8080) on exit, including on failure,
# memory-guard kill, or Ctrl-C. Restoration verifies model IDs and health.
# If cleanup or restoration fails, exit 10 / RESTORE_FAILED requires manual
# recovery. The holder releases its lock after that failure; no later GPU work
# is safe until the operator has recovered and verified the service.
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
#      Capture its model directory before bootout, wait for captured descendants,
#      then reclaim clean model-file cache before measuring the workload baseline.
#   4. Run the step (argv) under a SYSTEM-WIDE memory guard.  Before starting it
#      refuses to open if other mtplx/python workers above the foreign cap
#      (default 2 GiB RSS) are resident (prints them).  While it runs, aborts +
#      restores if EITHER the step tree footprint exceeds its cap (default 93 GiB)
#      OR total physical used memory (wired+active+inactive+physical compressor)
#      exceeds the ceiling (default 110 decimal GB, including file cache).
#   5. EXIT/INT/TERM trap: `launchctl bootstrap gui/<uid> <plist>` to restore the
#      resident agent, then the lock is released by the fcntl holder (phase 0).
#      Successful windows release only after model identity and health/warmup
#      verification. Failed restoration has the explicit recovery boundary above.
#
# Launch rules (memory/guarded-window-launch-protocol.md): launch as the DIRECT
# command of a run_in_background:true Bash call -- NEVER a shell `&` one-liner,
# which reaps the process group and leaves qwen down.  Do NOT stop the launching
# agent mid-window (it kills this wrapper AND its lock holder).
#
# W106 MEDIUM-3 launcher pattern: chain multiple windows / arms with `&&`, NEVER
# `;`.  A step that aborts exits non-zero (the ab harness exits 4 at the FIRST arm
# that aborts); with `;` the chain would re-open a window and re-abort every later
# step, and the launcher would report only the LAST rc.  With `&&` the first
# non-zero rc stops the chain and is the reported rc.  To abort a running window by
# hand, `kill -TERM` the pid this wrapper prints at start ("abort:" line) -- the
# bash gpu_window.sh, NOT its parent python lock-holder (which ignores signals).
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

# W106 MEDIUM-A/HIGH-3: several GiB caps feed bash arithmetic (`GB * 1024^3`,
# `X * 4`) that a fractional/non-integer value would break -- and with `set -u` an
# unset derived var would kill the window at the first poll AFTER Qwen is already
# booted out.  Validate them to a non-negative INTEGER BEFORE phase 0.  Defined
# before the config block so they can guard it.  NOTE: the env var names carry GiB
# despite the historical `_GB` suffix (see the W106 doc "Units" section); the value
# is GiB (1 GiB = 1024^3 bytes).
#
# _int_or_default warns + falls back (used for non-safety knobs like KILL_GRACE).
_int_or_default() {  # $1=value $2=default $3=env-name -> a valid non-negative int
  if [[ "$1" =~ ^[0-9]+$ ]]; then
    printf '%s' "$1"
  else
    printf '%s [gpu_window] WARN: %s=%s is not a non-negative integer (GiB); using %s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$3" "$1" "$2" >&2
    printf '%s' "$2"
  fi
}

# HIGH-3: the SAFETY caps (system ceiling, min-avail, foreign cap, child-tree cap)
# must REFUSE on an invalid value, not fall back to a default -- a fractional
# GPU_WINDOW_TOTAL_MEM_CEILING_GB=95.5 falling back to 102 would silently RAISE the
# ceiling (over the operator's intent).  _require_int VALIDATES (prints an ERROR to
# stderr and returns non-zero on a bad value) but does NOT exit -- the CALLER does
# `|| exit 2` in the MAIN shell, because an `exit` inside `$(...)` would only leave
# a subshell and the invalid value would slip through.  Refuses BEFORE phase 0.
_require_int() {  # $1=env-name $2=value [min] -> return 0 valid, else err + return 1
  local name="$1" val="$2" min="${3:-0}" reason=""
  if [[ ! "${val}" =~ ^[0-9]+$ ]]; then
    reason="must be a non-negative integer"
  elif (( val < min )); then
    reason="must be >= ${min}"
  else
    return 0
  fi
  printf '%s [gpu_window] ERROR: %s=%s is invalid (%s); refusing to open a GPU window\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${name}" "${val}" "${reason}" >&2
  return 1
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
CURL_CMD="${GPU_WINDOW_CURL_CMD:-/usr/bin/curl}"
HEALTH_URL="${GPU_WINDOW_HEALTH_URL:-http://127.0.0.1:8080/health}"
MODELS_URL="${GPU_WINDOW_MODELS_URL:-http://127.0.0.1:8080/v1/models}"
EXPECTED_MODEL_IDS=""  # captured before bootout, never inferred from the replacement service
QWEN_MODEL_PATH=""     # actual service artifact, captured before bootout
QWEN_PROCESS_IDS=""    # descendants must exit before file-cache reclamation
QWEN_STOP_REQUESTED=0  # set only after bootout succeeds
# Optional candidate artifact whose stale clean safetensor pages must not inflate
# the post-Qwen admission baseline.  The helper validates ownership and contents
# before invalidating pages; this never scans experts.bin.
CANDIDATE_MODEL_PATH="${GPU_WINDOW_CANDIDATE_MODEL_DIR:-}"
if [[ -n "${CANDIDATE_MODEL_PATH}" ]]; then
  if ! CANDIDATE_MODEL_PATH="$(/usr/bin/env python3 -c '
import pathlib, sys
try:
    path = pathlib.Path(sys.argv[1]).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("candidate model path is not a directory")
    print(path)
except (OSError, ValueError):
    sys.exit(1)
' "${CANDIDATE_MODEL_PATH}")"; then
    printf '%s [gpu_window] ERROR: GPU_WINDOW_CANDIDATE_MODEL_DIR is not a valid model directory\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
    exit 2
  fi
fi
# HIGH-3: safety caps REFUSE (exit 2) on an invalid value (never silently fall back).
MIN_AVAIL_GB="${GPU_WINDOW_MIN_AVAIL_GB:-100}"  # GiB the step needs available after the stop
_require_int GPU_WINDOW_MIN_AVAIL_GB "${MIN_AVAIL_GB}" || exit 2
# W106 HIGH-2: all guard caps are GiB (bytes = N * 1024^3), stated explicitly.
# Default child-tree RSS cap = 93 GiB ~= 100 GB (David's "100 GB total for
# everything").  This cap is LIVE for the first time (pre-W106 the poll read the
# few-MB `bash -c` shell RSS ~= 0); at 93 GiB it sits just under the 100 GB budget.
# HIGH-3: CHILD_RSS_CAP_BYTES is BYTES (not GiB) and must be >= 1 GiB -- "93" would
# be 93 BYTES (killing the step right after bootout), and a fractional value would
# break the arithmetic; refuse either.
CHILD_RSS_CAP_BYTES="${GPU_WINDOW_CHILD_RSS_CAP_BYTES:-$(( 93 * 1024 * 1024 * 1024 ))}"
_require_int GPU_WINDOW_CHILD_RSS_CAP_BYTES "${CHILD_RSS_CAP_BYTES}" $(( 1024 * 1024 * 1024 )) || exit 2
# MEDIUM-1: default 1s (streaming grows fast).  LOW (round 4): REFUSE 0 -- `sleep 0`
# is a busy-loop that pins a core and inflates the host-encode-sensitive window.
RSS_POLL_SECONDS="${GPU_WINDOW_RSS_POLL_SECONDS:-1}"
_require_int GPU_WINDOW_RSS_POLL_SECONDS "${RSS_POLL_SECONDS}" 1 || exit 2
# W106 MEDIUM-1: the child-tree cap compares `ps` RSS, which UNDERCOUNTS unified
# Metal memory by ~18 GiB (window 43: tree RSS 51.2 vs system-baseline 69.5), so it
# could never fire before the (accurate, vm_stat-based) system ceiling. Lower the
# effective child cap by this documented undercount so it fires at the real
# footprint; set 0 to disable the correction.  The SYSTEM ceiling remains the
# authoritative guard (vm_stat counts wired + compressed Metal).
RSS_METAL_UNDERCOUNT_GIB="$(_int_or_default "${GPU_WINDOW_RSS_METAL_UNDERCOUNT_GIB:-18}" 18 GPU_WINDOW_RSS_METAL_UNDERCOUNT_GIB)"
# W121: phase 4 measures the step tree's phys_footprint directly (proc_pid_rusage
# ri_phys_footprint, which INCLUDES Metal/IOAccelerator, wired or not), so the ps-RSS
# undercount fudge above is no longer applied to the cap.  RSS_METAL_UNDERCOUNT_GIB is
# retained only for the legacy ps-tree telemetry line.
# W121 compressor tripwire: abort if vm.compressor_bytes_used grows more than this many
# GiB over its at-start value during the step.  A healthy run keeps the compressor
# flat; non-wired Metal spilling into it is the swap-collapse signature (window 46: the
# compressor jumped to ~33 GB).  Set 0 to disable.
COMPRESSOR_TRIP_GB="$(_int_or_default "${GPU_WINDOW_COMPRESSOR_TRIP_GB:-8}" 8 GPU_WINDOW_COMPRESSOR_TRIP_GB)"
COMPRESSOR_TRIP_BYTES=$(( COMPRESSOR_TRIP_GB * 1024 * 1024 * 1024 ))

# Exact default: 110 decimal GB. Explicit historical _GB values retain their
# GiB meaning. The byte override avoids unit ambiguity; specifying both refuses.
# Polling detects breaches, but cannot bound an allocation between samples.
if [[ -n "${GPU_WINDOW_TOTAL_MEM_CEILING_BYTES:-}" && -n "${GPU_WINDOW_TOTAL_MEM_CEILING_GB:-}" ]]; then
  printf 'ERROR: specify only GPU_WINDOW_TOTAL_MEM_CEILING_BYTES or the legacy GiB GPU_WINDOW_TOTAL_MEM_CEILING_GB\n' >&2
  exit 2
elif [[ -n "${GPU_WINDOW_TOTAL_MEM_CEILING_GB:-}" ]]; then
  _require_int GPU_WINDOW_TOTAL_MEM_CEILING_GB "${GPU_WINDOW_TOTAL_MEM_CEILING_GB}" 1 || exit 2
  TOTAL_MEM_CEILING_BYTES=$(( GPU_WINDOW_TOTAL_MEM_CEILING_GB * 1024 * 1024 * 1024 ))
else
  TOTAL_MEM_CEILING_BYTES="${GPU_WINDOW_TOTAL_MEM_CEILING_BYTES:-110000000000}"
  _require_int GPU_WINDOW_TOTAL_MEM_CEILING_BYTES "${TOTAL_MEM_CEILING_BYTES}" 1 || exit 2
fi
# Display only. All comparisons use the authoritative integer byte value.
TOTAL_MEM_CEILING_GB="$(awk -v b="${TOTAL_MEM_CEILING_BYTES}" 'BEGIN {printf "%.9g", b/1073741824}')"
FOREIGN_WORKER_RSS_GB="${GPU_WINDOW_FOREIGN_WORKER_RSS_GB:-2}"   # GiB: refuse to start if another mtplx/python worker exceeds this RSS
_require_int GPU_WINDOW_FOREIGN_WORKER_RSS_GB "${FOREIGN_WORKER_RSS_GB}" || exit 2
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

# Physical used bytes, matching top: wired + active + inactive + physical
# compressor. Includes file cache and unwired pages; speculative pages are free.
# Used for both the post-unload baseline and the live guard observation.
used_mem_bytes() {
  local raw
  raw="$("${VM_STAT_CMD}" 2>/dev/null)" || return 1
  printf '%s\n' "${raw}" | awk '
    /page size of/ { for (i = 1; i <= NF; i++) if ($i == "of") ps = $(i + 1) }
    /^Pages wired down:/             { sub(/\.$/, "", $NF); wired = $NF; w = 1 }
    /^Pages active:/                 { sub(/\.$/, "", $NF); active = $NF; a = 1 }
    /^Pages inactive:/               { sub(/\.$/, "", $NF); inactive = $NF; inactive_seen = 1 }
    /^Pages occupied by compressor:/ { sub(/\.$/, "", $NF); comp = $NF; c = 1 }
    END {
      if (ps !~ /^[0-9]+$/ || ps <= 0 || !w || !a || !inactive_seen || !c ||
          wired !~ /^[0-9]+$/ || active !~ /^[0-9]+$/ || inactive !~ /^[0-9]+$/ || comp !~ /^[0-9]+$/) exit 1
      printf "%.0f", (wired + active + inactive + comp) * ps
    }
  '
}

# Physical bytes the compressor holds right now (sysctl vm.compressor_bytes_used).
# The phase-4 tripwire aborts if this grows > COMPRESSOR_TRIP_GB over its start
# value during the step: a healthy run keeps it flat, but non-wired Metal spilling
# to the compressor (window 46: 33 GB) is the swap-collapse signature.  Overridable
# for the unit test.
COMPRESSOR_SYSCTL_CMD="${GPU_WINDOW_COMPRESSOR_CMD:-/usr/sbin/sysctl}"
compressor_bytes_used() {
  local v
  v="$("${COMPRESSOR_SYSCTL_CMD}" -n vm.compressor_bytes_used 2>/dev/null)" || return 1
  [[ "${v}" =~ ^[0-9]+$ ]] || return 1
  printf '%s' "${v}"
}

# Sum of phys_footprint (mach proc_pid_rusage ri_phys_footprint) over the step's
# whole process tree. phys_footprint is a per-process accounting total INCLUDING
# IOAccelerator/Metal (wired or not) but EXCLUDING the shared file page cache --
# added to the baseline as a conservative guard estimate, separate from measured
# live physical used. Overridable for the unit test.
FOOTPRINT_READER_CMD="${GPU_WINDOW_FOOTPRINT_READER:-/usr/bin/env python3 $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tree_footprint.py}"
# HIGH-3: expose reader success/failure so the phase-4 guard can FAIL CLOSED instead of
# continuing at box_used == baseline when the reader breaks.  Still prints '0' on failure
# (backward-compatible for the box-used-* subcommands' arithmetic), but ALSO sets the
# globals TREE_FOOTPRINT_READ_OK / TREE_FOOTPRINT_BYTES -- the guard reads those by
# calling this function DIRECTLY (not under $(...), which would run it in a subshell and
# discard the globals).
TREE_FOOTPRINT_READ_OK=1
TREE_FOOTPRINT_BYTES=0
tree_footprint_bytes() {
  local root="${1:-}"
  TREE_FOOTPRINT_READ_OK=1
  TREE_FOOTPRINT_BYTES=0
  [[ -z "${root}" ]] && { TREE_FOOTPRINT_READ_OK=0; printf '0'; return; }
  local raw rc v
  raw="$(${FOOTPRINT_READER_CMD} "${root}" 2>/dev/null)"; rc=$?
  v="$(printf '%s' "${raw}" | tr -d '[:space:]')"
  if (( rc != 0 )) || ! [[ "${v}" =~ ^[0-9]+$ ]]; then
    TREE_FOOTPRINT_READ_OK=0; printf '0'; return
  fi
  TREE_FOOTPRINT_BYTES="${v}"
  printf '%s' "${v}"
}

# Print "<pid> <rssGiB> <command>" for every python/mtplx/mlx process whose RSS
# exceeds the foreign-worker cap, EXCLUDING this script's own pids (the bash
# wrapper and its parent python lock holder).  Used to refuse to open a window
# while another worker session holds gigabytes that would co-reside with the
# step and push the box over the ceiling.
list_heavy_foreign_workers() {
  local self_pids raw
  self_pids=" $$ ${PPID:-} ${_GPU_WINDOW_HOLDER_PID:-} "
  raw="$("${PS_CMD}" -axo pid=,rss=,comm= 2>/dev/null)" || return 1
  [[ -n "${raw}" ]] || return 1
  printf '%s\n' "${raw}" | awk \
    -v cap_kb="$(( FOREIGN_WORKER_RSS_GB * 1024 * 1024 ))" \
    -v self="${self_pids}" '
    {
      if (NF < 3 || $1 !~ /^[0-9]+$/ || $2 !~ /^[0-9]+$/) exit 1;
      pid = $1; rss = $2 + 0;
      cmd = $3; for (i = 4; i <= NF; i++) cmd = cmd " " $i;
      if (rss <= cap_kb) next;
      if (index(self, " " pid " ") > 0) next;
      lc = tolower(cmd);
      if (lc ~ /python|mtplx|mlx/) printf "%s %.1fGiB %s\n", pid, rss / 1048576, cmd;
    }
  '
}

# 0 = gone/zombie, 1 = running, 2 = unreadable. A failed ps read while the owned
# root is alive must never turn the polling loop into an unmonitored wait.
_step_finished() {
  local state rc
  state="$("${PS_CMD}" -o state= -p "${STEP_PID}" 2>/dev/null)"; rc=$?
  state="${state//[[:space:]]/}"
  if (( rc != 0 )) || [[ -z "${state}" ]]; then
    kill -0 "${STEP_PID}" 2>/dev/null && return 2
    return 0
  fi
  # Darwin can lose Mach thread information before the process becomes a
  # zombie: ps prints ? plus its P_WEXIT flag (for example ?E or ?NEs).
  # This is still a LIVE step: keep footprint/box monitoring and reap only
  # after the normal gone/zombie transition. A bare ? remains unreadable.
  # Apple adv_cmds/ps/{tasks,print}.c define the Mach state and suffix order.
  if ! [[ "${state}" =~ ^[RSDTZWIXUH][[:alnum:]\<\>+]*$ ]] &&
     ! [[ "${state}" =~ ^\?[\<N]?X?EV?L?s?[+]?$ ]]; then
    return 2
  fi
  [[ "${state}" == Z* ]] && return 0
  return 1
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
  # MEDIUM-1: emit only pids PRESENT in the ps snapshot.  The old walk seeded the
  # root unconditionally, so `--selftest tree-pids 999999` (or an already-dead
  # STEP_PID) printed a bogus pid -> the abort rescan logged false "ORPHAN survived"
  # ERRORs and KILLed a dead pid.
  local raw
  raw="$("${PS_CMD}" -axo pid=,ppid= 2>/dev/null)" || return 1
  [[ -n "${raw}" ]] || return 1
  printf '%s\n' "${raw}" | awk -v root="${root}" '
    {
      if (NF != 2 || $1 !~ /^[0-9]+$/ || $2 !~ /^[0-9]+$/) { bad = 1; exit 1 }
      pid = $1 + 0; ppid = $2 + 0; present[pid] = 1; kids[ppid] = kids[ppid] " " pid
    }
    END {
      if (bad) exit 1;
      head = 1; tail = 0; wl[++tail] = root + 0; out = "";
      while (head <= tail) {
        p = wl[head]; head++;
        if (p in seen) continue;
        seen[p] = 1;
        if (p in present) out = out " " p;   # only live pids
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

# API observations are bounded and have no model/MLX imports. The canonical
# model tuple is captured before unload, then compared on every restore probe.
_read_model_ids() {
  local raw
  raw="$("${CURL_CMD}" --fail --silent --connect-timeout 2 --max-time 2 "${MODELS_URL}")" || return 1
  printf '%s' "${raw}" | /usr/bin/env python3 -c '
import json, sys
try:
    rows = json.load(sys.stdin)["data"]
    ids = [row["id"] for row in rows]
    if not ids or not all(isinstance(i, str) and i for i in ids):
        raise ValueError("missing model IDs")
    print(json.dumps(sorted(ids), separators=(",", ":")))
except (KeyError, TypeError, ValueError):
    sys.exit(1)
'
}

_read_model_path() {
  local raw
  raw="$("${CURL_CMD}" --fail --silent --connect-timeout 2 --max-time 2 "${HEALTH_URL}")" || return 1
  printf '%s' "${raw}" | /usr/bin/env python3 -c '
import json, pathlib, sys
try:
    value = json.load(sys.stdin)["model_path"]
    if not isinstance(value, str) or not value.startswith("/") or any(c in value for c in "\n\r\0"):
        raise ValueError("missing absolute model directory")
    path = pathlib.Path(value).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("model path is not a directory")
    print(path)
except (OSError, KeyError, TypeError, ValueError):
    sys.exit(1)
'
}

_reclaim_qwen_file_cache() {
  local before
  before="$(used_mem_bytes)" || return 1
  if (( before + 1073741824 >= TOTAL_MEM_CEILING_BYTES )); then
    err "cache reclamation lacks its bounded 1 GiB host headroom"
    return 1
  fi
  log "phase 3: reclaiming stopped-service clean file cache from ${QWEN_MODEL_PATH}"
  # Helper is stdlib-only, read-only, bounded to 30 seconds. The shell waits for
  # it before any restore or model load; its exit precedes the fresh baseline.
  /usr/bin/env python3 "$(dirname "${SCRIPT_PATH}")/reclaim_file_cache.py" \
    --operation stopped_service_file_cache_reclamation "${QWEN_MODEL_PATH}" || return 1
  _check_abort
}

_reclaim_candidate_file_cache() {
  local before
  before="$(used_mem_bytes)" || return 1
  if (( before + 1073741824 >= TOTAL_MEM_CEILING_BYTES )); then
    err "candidate cache reclamation lacks its bounded 1 GiB host headroom"
    return 1
  fi
  log "phase 3: reclaiming stale candidate clean file cache from ${CANDIDATE_MODEL_PATH}"
  /usr/bin/env python3 "$(dirname "${SCRIPT_PATH}")/reclaim_file_cache.py" \
    --operation candidate_file_cache_reclamation "${CANDIDATE_MODEL_PATH}" || return 1
  _check_abort
}

_restored_api_ready() {
  local raw ids
  raw="$("${CURL_CMD}" --fail --silent --connect-timeout 2 --max-time 2 "${HEALTH_URL}")" || return 1
  printf '%s' "${raw}" | /usr/bin/env python3 -c '
import json, sys
try:
    health = json.load(sys.stdin)
    if health.get("ok") is not True:
        raise ValueError("service unhealthy")
    warmup = (health.get("startup") or {}).get("warmup", health.get("warmup")) or {}
    background = warmup.get("background")
    if background is not None and background.get("state") not in ("done", "disabled", "skipped"):
        raise ValueError("background warmup unfinished")
except (AttributeError, TypeError, ValueError):
    sys.exit(1)
' || return 1
  ids="$(_read_model_ids)" || return 1
  [[ -z "${EXPECTED_MODEL_IDS}" || "${ids}" == "${EXPECTED_MODEL_IDS}" ]]
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
  local loaded=0
  "${LAUNCHCTL_CMD}" print "${DOMAIN}/${QWEN_LABEL}" >/dev/null 2>&1 && loaded=1
  if (( loaded == 1 )); then
    log "restore: ${QWEN_LABEL} already loaded; verifying API readiness and model identity"
  elif (( was_loaded == 0 )) && [[ "${RESTORE_QWEN_ALWAYS}" == "1" ]]; then
    log "restore: ${QWEN_LABEL} was NOT loaded at entry, but GPU_WINDOW_RESTORE_QWEN_ALWAYS=1 -> bootstrapping anyway (guards against a cascaded prior failure)"
  elif (( was_loaded == 0 )); then
    log "restore: ${QWEN_LABEL} was not loaded at entry; leaving it stopped (as found)"
    return 0
  fi

  local plist
  plist="$(_resolve_restore_plist "${discovered}" "${CANONICAL_PLIST}")"
  if [[ -n "${discovered}" && "${discovered}" != "${plist}" ]]; then
    log "restore: discovered plist '${discovered}' is gone; falling back to '${plist:-<none>}'"
  fi
  if (( loaded == 0 )) && [[ -z "${plist}" ]]; then
    err "restore: NO plist file exists to bootstrap (discovered '${discovered}' gone, canonical '${CANONICAL_PLIST}' missing); ${QWEN_LABEL} may be DOWN -- manual recovery: ${LAUNCHCTL_CMD} bootstrap ${DOMAIN} ${CANONICAL_PLIST}"
    return 1
  fi

  local deadline
  deadline=$(( $(date +%s) + RESTORE_TIMEOUT ))
  while (( $(date +%s) < deadline )); do
    if ! "${LAUNCHCTL_CMD}" print "${DOMAIN}/${QWEN_LABEL}" >/dev/null 2>&1; then
      if [[ -n "${plist}" ]]; then
        log "restore: launchctl bootstrap ${DOMAIN} ${plist}"
        "${LAUNCHCTL_CMD}" bootstrap "${DOMAIN}" "${plist}" || true
      fi
    fi
    if "${LAUNCHCTL_CMD}" print "${DOMAIN}/${QWEN_LABEL}" >/dev/null 2>&1 && _restored_api_ready; then
      if [[ -n "${EXPECTED_MODEL_IDS}" ]]; then
        log "restore: ${QWEN_LABEL} healthy, model identity ${EXPECTED_MODEL_IDS} verified, background warmup ready"
      else
        log "restore: ${QWEN_LABEL} healthy with model IDs available, background warmup ready (no entry model tuple was available to compare)"
      fi
      return 0
    fi
    sleep 1
  done
  err "restore: ${QWEN_LABEL} did not become healthy with the expected model within ${RESTORE_TIMEOUT}s; verify ${HEALTH_URL} and ${MODELS_URL}; manual recovery: ${LAUNCHCTL_CMD} bootstrap ${DOMAIN} ${CANONICAL_PLIST}"
  return 1
}

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# ---- test/introspection hooks (no GPU, no lock, no launchctl) ----------------
# Let a shell unit test exercise the pure guard math against fake vm_stat / ps
# output without opening a window.  Runs BEFORE the lock phase and exits.
if [[ "${1:-}" == "--selftest" ]]; then
  shift
  case "${1:-}" in
    used-mem-bytes) used_mem_bytes || exit 8; echo ;;
    used-mem-gib)   _u="$(used_mem_bytes)" || exit 8; gib "${_u}"; echo ;;
    compressor-bytes) compressor_bytes_used || exit 8; echo ;;
    tree-footprint) tree_footprint_bytes "${2:-}"; echo ;;
    box-used-bytes)
      # W121: box_used = one-time baseline (used_mem_bytes) + step-tree phys_footprint.
      # Inject GPU_WINDOW_VM_STAT_CMD (baseline) + GPU_WINDOW_FOOTPRINT_READER (tree)
      # for a deterministic unit test.  $2 = the step root pid handed to the reader.
      _base="$(used_mem_bytes)" || exit 8; _fp="$(tree_footprint_bytes "${2:-}")"
      printf '%s\n' "$(( _base + _fp ))"
      ;;
    box-used-gib)
      _base="$(used_mem_bytes)" || exit 8; _fp="$(tree_footprint_bytes "${2:-}")"
      gib "$(( _base + _fp ))"; echo
      ;;
    over-ceiling)
      _u="$(used_mem_bytes)" || exit 8
      if [[ "${_u}" =~ ^[0-9]+$ ]] && (( _u > TOTAL_MEM_CEILING_BYTES )); then
        echo yes
      else
        echo no
      fi
      ;;
    heavy-workers)  list_heavy_foreign_workers ;;
    model-path)    _read_model_path || exit 8 ;;
    tree-pids)      _step_tree_pids "${2:-}" || exit 8; echo ;;   # W106 item 4: tree walk
    restore-plist)  _resolve_restore_plist "${2:-}" "${3:-}" ; echo ;;  # discovered, canonical
    restore-run)
      # W106 restore test: run _do_restore against a fake ${LAUNCHCTL_CMD} with
      # env-injected inputs, then exit with its status.  $2 = was_loaded (1/0),
      # $3 = discovered plist path (may be a vanished guard-dir copy).
      EXPECTED_MODEL_IDS="${4:-}"
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
# after this wrapper (its child) exits. Exit 10 indicates failed cleanup/restore
# and requires operator recovery before any later GPU work.
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

# W106 abort item (b): the tree-kill must catch descendants REPARENTED to launchd.
# A ppid walk from STEP_PID misses a python whose `bash -c` chain died first (its
# ppid is now 1) -- window 42: a 27 GB python kept loading experts.bin outside the
# lock for minutes.  So each step is launched with a UNIQUE env tag
# (_GPU_WINDOW_STEP_TAG), inherited by every descendant and UNCHANGED by
# reparenting; `_pids_with_tag` finds them via `ps -E` regardless of ppid.  The
# grep uses the [x]-bracket trick so the grep/awk pipeline never matches itself.
#
# W106 MEDIUM-3 LIMITATION: `ps -E` does NOT expose the environment of macOS
# PLATFORM binaries (SIP-signed: /bin/bash, /bin/sleep, /usr/bin/tee, ...), so a
# reparented platform-binary descendant is INVISIBLE to this tag scan (verified:
# a tagged /bin/sleep shows no env; a tagged .venv python does).  It reliably
# catches the descendant that MATTERS -- the venv python holding the model -- so
# KEEP THE STEP A SINGLE venv-python process (no `python ... | tee`, no wrapping
# `bash -c` that itself outlives the python) to guarantee the heavy orphan is
# reaped.  The ppid tree + post-KILL rescan still catch non-reparented platform
# children; the vm_stat SYSTEM ceiling is the backstop for anything missed.
_pids_with_tag() {
  [[ -n "${_STEP_TAG:-}" ]] || return 0
  local pat="_GPU_WINDOW_STEP_TAG=[${_STEP_TAG:0:1}]${_STEP_TAG:1}"
  local raw
  raw="$("${PS_CMD}" -axEo pid=,command= 2>/dev/null)" || return 1
  [[ -n "${raw}" ]] || return 1
  printf '%s\n' "${raw}" | awk -v pat="${pat}" -v self=" $$ ${PPID:-} ${_GPU_WINDOW_HOLDER_PID:-} " \
    '{ if (NF < 2 || $1 !~ /^[0-9]+$/) exit 1 }
     $0 ~ pat { if (index(self, " " $1 " ") == 0) print $1 }'
}

# The FULL set of step pids: the ppid tree rooted at STEP_PID UNION the env-tagged
# processes (which survive reparenting), one pid per line, deduped.
_collect_step_pids() {
  local tree="" tagged
  if [[ -n "${STEP_PID:-}" ]]; then
    tree="$(_step_tree_pids "${STEP_PID}")" || return 1
  fi
  tagged="$(_pids_with_tag)" || return 1
  printf '%s\n%s\n' "${tree}" "${tagged}" | tr ' ' '\n' | awk '/^[0-9]+$/' | sort -un
}

_step_children_exited() {
  local pids p state
  pids="$(_collect_step_pids)" || return 1
  # Retain the last owned set: a surviving platform binary may have reparented
  # after TERM and no longer expose its environment tag through ps.
  for p in ${pids} ${LAST_STEP_PIDS:-}; do
    if kill -0 "${p}" 2>/dev/null; then
      state="$("${PS_CMD}" -o state= -p "${p}" 2>/dev/null)" || return 1
      state="${state//[[:space:]]/}"
      [[ "${state}" == Z* ]] || return 1
    fi
  done
}

_kill_step_tree() {
  # TERM then (after a grace) KILL every process in the step set (ppid tree UNION
  # env-tag), so an aborted or completed step leaves NO descendant -- including one
  # reparented to launchd -- alive to keep running outside the lock.  Then RE-SCAN
  # (a couple rounds) for late forks / reparents, logging + KILLing any ORPHAN.
  # Robust to an empty STEP_PID (reaps tagged orphans after a "normal" step exit).
  local pids _p _i _alive _r
  pids="$(_collect_step_pids)"
  [[ -z "${pids}" && -n "${STEP_PID:-}" ]] && pids="${STEP_PID}"
  LAST_STEP_PIDS="${LAST_STEP_PIDS:-} ${pids}"
  if [[ -n "${pids}" ]]; then
    for _p in ${pids}; do kill -TERM "${_p}" 2>/dev/null || true; done
    for (( _i = 0; _i < KILL_GRACE_SECONDS * 4; _i++ )); do
      _alive=0
      for _p in ${pids}; do kill -0 "${_p}" 2>/dev/null && { _alive=1; break; }; done
      (( _alive == 0 )) && break
      sleep 0.25
    done
    for _p in ${pids}; do
      kill -0 "${_p}" 2>/dev/null && kill -KILL "${_p}" 2>/dev/null || true
    done
  fi
  # Re-scan: catch anything that forked/reparented AFTER the snapshot (the window-42
  # orphan). Log each survivor as an ORPHAN and KILL it.  MEDIUM-1: re-confirm each
  # pid is ALIVE (kill -0) before logging/KILLing, so a dead pid never produces a
  # false "ORPHAN survived" line.
  for _r in 1 2 3; do
    local survivors; survivors="$(_collect_step_pids)"
    [[ -z "${survivors}" ]] && break
    LAST_STEP_PIDS="${LAST_STEP_PIDS:-} ${survivors}"
    local _any=0
    for _p in ${survivors}; do
      kill -0 "${_p}" 2>/dev/null || continue
      _any=1
      err "phase 4: ORPHAN survived tree-kill: pid ${_p} ($("${PS_CMD}" -o command= -p "${_p}" 2>/dev/null | tr '\n' ' ' | cut -c1-100)); KILLing"
      kill -KILL "${_p}" 2>/dev/null || true
    done
    (( _any == 0 )) && break
    sleep 0.3
  done
  [[ -n "${STEP_PID:-}" ]] && { wait "${STEP_PID}" 2>/dev/null || true; }
  STEP_PID=""
}

# Back-compat alias for the phase-4 abort call sites (RSS cap / system ceiling).
_kill_step_child() { _kill_step_tree; }

WAS_LOADED=0
RESTORED=0
STEP_PID=""
_STEP_TAG=""                # W106 (b): unique env tag on the step, to find reparented descendants
LAST_STEP_PIDS=""           # retained until the final pre-bootstrap liveness check
PEAK_TREE_RSS_BYTES=0       # W121: running peak of the step tree's Σ phys_footprint
PEAK_MAX_PROC_RSS_BYTES=0   # (retained; legacy ps-tree telemetry, no longer polled)
PEAK_GUARD_ACCOUNTED_BYTES=0  # peak of max(baseline + footprint estimate, physical used)
PEAK_SYSTEM_USED_BYTES=0      # sampled physical-used peak, including baseline
PEAK_COMPRESSOR_DELTA_BYTES=0  # running peak of compressor growth over its at-start value
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
  # LOW (round 4): IGNORE further INT/TERM during teardown (do not reset to the
  # DEFAULT disposition -- a second TERM mid-restore would kill the process and
  # leave the agent down).  Remove only the EXIT trap so teardown does not re-enter.
  trap - EXIT
  trap '' INT TERM
  # W106 item 4: a TERM/INT to the wrapper (or any non-abort exit with the step
  # still running) tree-kills the WHOLE step process tree, not just STEP_PID, so a
  # `bash -c` chain never starts its next step and no python descendant survives.
  if [[ -n "${STEP_PID}" ]] && kill -0 "${STEP_PID}" 2>/dev/null; then
    log "teardown: terminating step process tree (root pid=${STEP_PID}): $(_step_tree_pids "${STEP_PID}")"
    _kill_step_tree
  elif [[ -n "${_STEP_TAG:-}" ]]; then
    # W106 (b): the step already returned, but a tagged descendant may be orphaned
    # (reparented to launchd) and still running -- reap it before releasing the lock
    # so nothing keeps loading the model outside the exclusive window.
    _kill_step_tree
  fi
  if ! _step_children_exited; then
    err "RESTORE_FAILED: owned step processes remain alive or cannot be inspected; refusing to bootstrap over them. Manual recovery required before more GPU work."
    exit 10
  fi
  if (( ${QWEN_STOP_REQUESTED:-0} == 1 )); then
    for _old_qwen_pid in ${QWEN_PROCESS_IDS}; do
      if kill -0 "${_old_qwen_pid}" 2>/dev/null; then
        err "RESTORE_FAILED: captured Qwen process ${_old_qwen_pid} survived shutdown; refusing to bootstrap over it. Manual recovery required before more GPU work."
        exit 10
      fi
    done
  fi
  if ! restore_qwen; then
    err "RESTORE_FAILED: service readiness/identity not verified; manual recovery required before more GPU work (step exit ${ec})."
    exit 10
  fi
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
  log "phase 2: ${QWEN_LABEL} is NOT loaded at entry; nothing to boot out; restore-on-exit policy ${RESTORE_QWEN_ALWAYS}"
else
  log "phase 2: ${QWEN_LABEL} loaded (pid=${QWEN_PID:-unknown}); plist=${PLIST}"
  if ! EXPECTED_MODEL_IDS="$(_read_model_ids)"; then
    err "phase 2: cannot capture the served model identity; refusing to boot out the service"
    exit 5
  fi
  log "phase 2: captured model identity ${EXPECTED_MODEL_IDS}"
  if ! QWEN_MODEL_PATH="$(_read_model_path)"; then
    err "phase 2: cannot capture the service model directory; refusing to boot out the service"
    exit 5
  fi
  if ! QWEN_PROCESS_IDS="$(_step_tree_pids "${QWEN_PID}")" || [[ -z "${QWEN_PROCESS_IDS}" ]]; then
    err "phase 2: cannot capture the service process tree; refusing to boot out the service"
    exit 5
  fi
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
  QWEN_STOP_REQUESTED=1
  deadline=$(( $(date +%s) + STOP_TIMEOUT ))
  pid_gone=0
  freed=0
  while (( $(date +%s) < deadline )); do
    _check_abort   # W106 (a): abort promptly even during the bootout wait
    if (( pid_gone == 0 )); then
      pid_gone=1
      for _qwen_pid in ${QWEN_PROCESS_IDS}; do
        if kill -0 "${_qwen_pid}" 2>/dev/null; then pid_gone=0; fi
      done
      if (( pid_gone == 1 )); then log "phase 3: ${QWEN_LABEL} captured process tree is gone"; fi
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
    err "phase 3: ${QWEN_LABEL} process tree still alive after ${STOP_TIMEOUT}s; aborting (no hidden retries)"
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
if ! HEAVY_WORKERS="$(list_heavy_foreign_workers)"; then
  err "phase 4: foreign-worker ps snapshot unreadable or malformed; refusing to start the step"
  exit 8
fi
if [[ -n "${HEAVY_WORKERS}" ]]; then
  err "phase 4: REFUSING to start -- other mtplx/python workers above ${FOREIGN_WORKER_RSS_GB} GiB RSS are resident (they co-reside with the step and could panic the box):"
  printf '%s\n' "${HEAVY_WORKERS}" | while IFS= read -r _hw_line; do
    err "    ${_hw_line}"
  done
  exit 7
fi
if [[ -n "${QWEN_MODEL_PATH}" ]]; then
  if ! _reclaim_qwen_file_cache; then
    err "phase 3: file-cache reclamation failed; refusing workload and restoring service"
    exit 8
  fi
fi
if [[ -n "${CANDIDATE_MODEL_PATH}" && "${CANDIDATE_MODEL_PATH}" != "${QWEN_MODEL_PATH}" ]]; then
  if ! _reclaim_candidate_file_cache; then
    err "phase 3: candidate file-cache reclamation failed; refusing workload and restoring service"
    exit 8
  fi
fi
if ! USED_START="$(used_mem_bytes)"; then
  err "phase 4: baseline vm_stat unreadable or incomplete; refusing to start the step"
  exit 8
fi
if (( USED_START >= TOTAL_MEM_CEILING_BYTES )); then
  err "phase 4: physical used baseline ${USED_START} bytes leaves no headroom below ceiling ${TOTAL_MEM_CEILING_BYTES} bytes; refusing to start the step"
  exit 8
fi
if ! COMPRESSOR_START="$(compressor_bytes_used)"; then
  err "phase 4: compressor baseline unreadable; refusing to start the step"
  exit 8
fi
log "phase 4: physical used at start (wired+active+inactive+compressor, includes file cache): $(gib "${USED_START}") GiB (ceiling ${TOTAL_MEM_CEILING_GB} GiB); compressor at start $(gib "${COMPRESSOR_START}") GiB"
log "phase 4: guard accounting = max(baseline + step footprint estimate, live physical used); compressor tripwire ${COMPRESSOR_TRIP_GB} GiB over start; physical used includes file cache"
# MEDIUM-4: hand the measured baseline (decimal GB) to the step env so the DSV4.1 bench's
# box target derives from the SAME physical-used baseline the guard uses -- measured now
# with the resident agent booted out -- rather than a hand-passed --box-baseline-gb (which
# the bench now treats as a fallback only).  Both units logged.
if [[ "${USED_START:-}" =~ ^[0-9]+$ ]]; then
  export MTPLX_DSV41_BOX_BASELINE_GB="$(awk -v b="${USED_START}" 'BEGIN{printf "%.4f", b/1e9}')"
  log "phase 4: exported MTPLX_DSV41_BOX_BASELINE_GB=${MTPLX_DSV41_BOX_BASELINE_GB} (decimal GB) == $(gib "${USED_START}") GiB baseline to the step env"
fi

# W106 MEDIUM-B: relate the child-tree cap to the measured baseline.  If the step
# grew to the full CHILD_RSS_CAP on top of what is ALREADY used, the box would
# cross the system ceiling before the per-child cap ever fired.  So the EFFECTIVE
# child-tree cap is min(CHILD_RSS_CAP, ceiling - used_start): the most the step can
# add (as phys_footprint) without crossing the ceiling.  We LOWER (never raise) it.
# W121: the cap is compared against the tree phys_footprint DIRECTLY -- footprint is
# the process total incl. Metal, so the old ps-RSS undercount fudge is gone.
EFFECTIVE_CHILD_CAP_BYTES="${CHILD_RSS_CAP_BYTES}"
if [[ "${USED_START:-}" =~ ^[0-9]+$ ]] && \
   (( USED_START + CHILD_RSS_CAP_BYTES > TOTAL_MEM_CEILING_BYTES )); then
  _headroom=$(( TOTAL_MEM_CEILING_BYTES - USED_START ))
  (( _headroom < 0 )) && _headroom=0
  EFFECTIVE_CHILD_CAP_BYTES="${_headroom}"
  log "phase 4: effective child-tree footprint cap $(gib "${EFFECTIVE_CHILD_CAP_BYTES}") GiB (lowered from $(gib "${CHILD_RSS_CAP_BYTES}") GiB: used_start $(gib "${USED_START}") + cap would cross the ${TOTAL_MEM_CEILING_GB} GiB ceiling)"
fi

# W106 HIGH-2: state BOTH caps explicitly (GiB + the ~GB equivalent) at step start
# so the operator sees the guard envelope next to the step it is about to run.
log "phase 4: guard caps -- child-tree footprint cap $(gib "${EFFECTIVE_CHILD_CAP_BYTES}") GiB (~$(awk -v b="${EFFECTIVE_CHILD_CAP_BYTES}" 'BEGIN{printf "%.0f", b/1e9}') GB); physical-used ceiling ${TOTAL_MEM_CEILING_BYTES} bytes ($(awk -v b="${TOTAL_MEM_CEILING_BYTES}" 'BEGIN{printf "%.3f", b/1e9}') decimal GB, ${TOTAL_MEM_CEILING_GB} GiB)"
# W106 (b): tag the step's environment with a unique marker, inherited by EVERY
# descendant and unchanged by reparenting, so _pids_with_tag can find (and kill) a
# python that was reparented to launchd after its `bash -c` chain died.  Set inline
# on the step only (NOT exported in the wrapper), so it never matches the wrapper.
_STEP_TAG="gpuwin-$$-$(date +%s)-${RANDOM}${RANDOM}"
log "phase 4: starting GPU step (tag ${_STEP_TAG}) under child-tree footprint cap $(gib "${EFFECTIVE_CHILD_CAP_BYTES}") GiB + system ceiling ${TOTAL_MEM_CEILING_GB} GiB: $*"
_GPU_WINDOW_STEP_TAG="${_STEP_TAG}" "$@" &
STEP_PID=$!
# Seed both sampled peaks with the pre-step baseline.
if [[ "${USED_START:-}" =~ ^[0-9]+$ ]]; then
  PEAK_SYSTEM_USED_BYTES="${USED_START}"
  PEAK_GUARD_ACCOUNTED_BYTES="${USED_START}"
fi
_last_mem_sample=0  # 0 => the first poll logs an envelope sample immediately
while :; do
  _check_abort   # W106 (a): abort promptly on a queued INT/TERM (not deferred)
  # Loop terminates when STEP_PID is gone or a zombie.
  _step_finished; step_status=$?
  if (( step_status == 0 )); then
    break
  elif (( step_status == 2 )); then
    err "phase 4: live step state unreadable; killing child and restoring"
    _kill_step_child
    exit 8
  fi
  # W121: the step's contribution to box pressure = Σ phys_footprint over its whole
  # tree (proc_pid_rusage ri_phys_footprint, which INCLUDES Metal/IOAccelerator
  # whether or not it is wired, but EXCLUDES the shared file page cache). This
  # footprint plus the baseline is a conservative estimate, separate from the live
  # physical-used observation, which includes the expert bank file cache.
  # Call DIRECTLY (not under $(...)) so TREE_FOOTPRINT_READ_OK / _BYTES set inside the
  # function reach this shell -- a command substitution would run it in a subshell and
  # discard the flag, defeating the HIGH-3 fail-closed check.
  tree_footprint_bytes "${STEP_PID}" >/dev/null
  tree_ok=${TREE_FOOTPRINT_READ_OK}
  tree_bytes=${TREE_FOOTPRINT_BYTES}
  # Fold in LIVE physical used (including file cache) so the
  # guard is not blind to OTHER processes growing (a build, a pytest sweep, a worker):
  #   box_used = max(baseline + step footprint, live system used).
  if ! live_used="$(used_mem_bytes)"; then
    err "phase 4: live vm_stat unreadable or incomplete; killing child and restoring"
    _kill_step_child
    exit 8
  fi
  if (( ! tree_ok )); then
    # A gone/zombie ROOT is a BENIGN step exit, not a reader failure: the step raced us
    # between the step_state check above and this read (e.g. `bash -c true` exiting fast),
    # and tree_footprint exits non-zero on an unreadable root (MEDIUM-2).  Re-check
    # liveness; if the step is gone, break and let the normal post-loop reap run.
    _step_finished; step_status=$?
    if (( step_status == 0 )); then
      break
    fi
    # HIGH-3: FAIL CLOSED.  The reader broke (subprocess error / 15 s timeout) while the
    # step is STILL ALIVE -- do NOT continue at box_used == baseline (fail-open, blind to
    # the whole step); a broken primary guard is not something to run a GPU step under.
    err "phase 4: step-footprint reader UNREADABLE for LIVE step ${STEP_PID} (returned no valid footprint); the box guard cannot see the step -- killing child and restoring"
    _kill_step_child
    exit 8
  fi
  box_used=$(( USED_START + tree_bytes ))
  if (( live_used > box_used )); then
    box_used=${live_used}
  fi
  if (( tree_bytes > PEAK_TREE_RSS_BYTES )); then
    PEAK_TREE_RSS_BYTES=${tree_bytes}
  fi
  if (( box_used > PEAK_GUARD_ACCOUNTED_BYTES )); then
    PEAK_GUARD_ACCOUNTED_BYTES=${box_used}
  fi
  if (( live_used > PEAK_SYSTEM_USED_BYTES )); then
    PEAK_SYSTEM_USED_BYTES=${live_used}
  fi
  # Authoritative box guard (checked FIRST): max(baseline + step footprint, live system
  # used) over the ceiling -- David's plain sum, hardened against other-process growth.
  if (( box_used > TOTAL_MEM_CEILING_BYTES )); then
    err "phase 4: GUARD accounted memory $(gib "${box_used}") GiB (max of baseline $(gib "${USED_START}") + step footprint estimate $(gib "${tree_bytes}"), live physical used $(gib "${live_used}") GiB) exceeded ceiling $(gib "${TOTAL_MEM_CEILING_BYTES}") GiB; killing child and restoring"
    _kill_step_child
    exit 8
  fi
  # Secondary per-step footprint cap: fires only when an operator sets a TIGHTER
  # CHILD_RSS_CAP than the box headroom (otherwise EFFECTIVE_CHILD_CAP == the box
  # headroom and the box guard above already caught it).
  if (( tree_bytes > EFFECTIVE_CHILD_CAP_BYTES )); then
    err "phase 4: step tree phys_footprint $(gib "${tree_bytes}") GiB exceeded cap $(gib "${EFFECTIVE_CHILD_CAP_BYTES}") GiB; killing child and restoring"
    _kill_step_child
    exit 6
  fi
  # Compressor tripwire: a healthy run keeps the compressor flat; a jump is the
  # swap-collapse signature (non-wired Metal spilling to the compressor).
  if ! comp_now="$(compressor_bytes_used)"; then
    err "phase 4: compressor reader unreadable; killing child and restoring"
    _kill_step_child
    exit 8
  fi
  comp_delta=$(( comp_now - COMPRESSOR_START ))
  (( comp_delta < 0 )) && comp_delta=0
  if (( comp_delta > PEAK_COMPRESSOR_DELTA_BYTES )); then
    PEAK_COMPRESSOR_DELTA_BYTES=${comp_delta}
  fi
  if (( COMPRESSOR_TRIP_BYTES > 0 && comp_delta > COMPRESSOR_TRIP_BYTES )); then
    err "phase 4: COMPRESSOR grew $(gib "${comp_delta}") GiB over start (> ${COMPRESSOR_TRIP_GB} GiB tripwire) -- swap/compression collapse; killing child and restoring"
    _kill_step_child
    exit 8
  fi
  # One memory-envelope sample every 30 s (and once on the first poll): the three
  # numbers David asked for (baseline start, step tree footprint, compressor delta).
  _now_epoch="$(date +%s)"
  if (( _now_epoch - _last_mem_sample >= 30 )); then
    log "phase 4: mem sample -- baseline $(gib "${USED_START}") GiB + step footprint $(gib "${tree_bytes}") GiB, live physical used $(gib "${live_used}") GiB => guard accounted $(gib "${box_used}") GiB; compressor +$(gib "${comp_delta}") GiB (trip ${COMPRESSOR_TRIP_GB} GiB)"
    _last_mem_sample=${_now_epoch}
  fi
  sleep "${RSS_POLL_SECONDS}"
done
wait "${STEP_PID}"
step_rc=$?
STEP_PID=""
log "phase 4: GPU step exited with code ${step_rc}; peak step footprint $(gib "${PEAK_TREE_RSS_BYTES}") GiB, peak guard accounted $(gib "${PEAK_GUARD_ACCOUNTED_BYTES}") GiB, peak physical used $(gib "${PEAK_SYSTEM_USED_BYTES}") GiB, peak compressor delta $(gib "${PEAK_COMPRESSOR_DELTA_BYTES}") GiB"

# phase 5 (restore + lock release) runs in the teardown trap on this exit.
exit "${step_rc}"
