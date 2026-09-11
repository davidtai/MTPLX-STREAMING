#!/usr/bin/env bash
# Serve the DeepSeek-V4.1-Flash q2 streaming artifact on a FREE HIGH PORT (never
# :8080), wait for /health, print generation_mode + model key, send one short
# chat completion, print tok/s, then stop the server cleanly.
#
# This is a STEP meant to run *inside* gpu_window.sh, which holds the exclusive
# GPU lock and has already booted out the resident agent.  It therefore does NOT
# take the lock or touch launchctl / :8080 itself.
#
#   bash scripts/deepseek_v41/gpu_window.sh \
#        bash scripts/deepseek_v41/serve_health.sh
#
# Runs the WORKTREE's code (PYTHONPATH + cwd = worktree) and asserts engagement
# before serving, so the editable install cannot silently shadow another tree
# (memory/editable-install-cwd-shadowing.md).  Every phase prints a timestamped
# line.
#
# Requires worker W1's mtplx/models/deepseek_v41.py to exist for the model to
# construct; until then the server fails admission/construction and this script
# reports the failure and exits non-zero (by design).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "${HERE}/../.." && pwd)"
# The campaign venv lives in the MAIN checkout, not the worktree; PYTHONPATH=WORKTREE
# makes the worktree's editable mtplx win (memory/editable-install-cwd-shadowing.md).
CAMPAIGN_VENV="${MTPLX_CAMPAIGN_VENV:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv}"
VENV_PY="${MTPLX_VENV_PY:-${CAMPAIGN_VENV}/bin/python3}"
MODEL="${DSV41_MODEL:-${HOME}/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4}"
HOST="${DSV41_HOST:-127.0.0.1}"
HEALTH_TIMEOUT="${DSV41_HEALTH_TIMEOUT:-600}"   # server load = 8.67 GB residents + admit
STOP_TIMEOUT="${DSV41_STOP_TIMEOUT:-60}"
MAX_TOKENS="${DSV41_MAX_TOKENS:-16}"
# Optional extra `mtplx serve` flags, e.g. DSV41_SERVE_EXTRA_ARGS="--generation-mode mtp".
LOG_DIR="${DSV41_LOG_DIR:-${TMPDIR:-/tmp}/dsv41-serve-health}"
# Stdlib-only response parsers. Bodies are piped into this AS A FILE so the piped
# JSON reaches sys.stdin (a `python3 - <<'HEREDOC'` here-doc would shadow fd 0).
PARSE="${HERE}/serve_health_parse.py"

ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s [serve_health] %s\n' "$(ts)" "$*"; }
err() { printf '%s [serve_health] ERROR: %s\n' "$(ts)" "$*" >&2; }

if [[ ! -x "${VENV_PY}" ]]; then
  err "venv python not found/executable at ${VENV_PY}"
  exit 1
fi
if [[ ! -f "${PARSE}" ]]; then
  err "response parser not found at ${PARSE}"
  exit 1
fi
if [[ ! -d "${MODEL}" ]]; then
  err "model artifact not found at ${MODEL}"
  exit 1
fi
mkdir -p "${LOG_DIR}"
SERVER_LOG="${LOG_DIR}/serve-$(date -u +%Y%m%dT%H%M%SZ).log"

# Never :8080. Pick the first free port in a high range.
PORT="$(
  cd "${WORKTREE}" && PYTHONPATH="${WORKTREE}" "${VENV_PY}" - <<'PYPORT'
import socket, sys
FORBIDDEN = {8080}
for port in range(18080, 18300):
    if port in FORBIDDEN:
        continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
    print(port)
    sys.exit(0)
sys.exit("no free port in 18080-18299")
PYPORT
)" || { err "could not find a free high port"; exit 1; }
if [[ "${PORT}" == "8080" || ! "${PORT}" =~ ^[0-9]+$ ]]; then
  err "refusing to serve on port '${PORT}' (never :8080)"
  exit 1
fi
BASE="http://${HOST}:${PORT}"
log "selected free port ${PORT} (never :8080); server log -> ${SERVER_LOG}"

# Editable-install engagement guard: the served code MUST be this worktree's.
log "asserting mtplx resolves to the worktree ${WORKTREE}"
if ! (cd "${WORKTREE}" && PYTHONPATH="${WORKTREE}" "${VENV_PY}" - "${WORKTREE}" <<'PYASSERT'
import sys, mtplx
root = sys.argv[1]
if not mtplx.__file__.startswith(root):
    sys.exit(f"mtplx resolved to {mtplx.__file__}, not under {root}")
PYASSERT
); then
  err "engagement assertion failed; refusing to serve the wrong tree"
  exit 1
fi

SERVER_PID=""
cleanup() {
  local ec=$?
  trap - EXIT INT TERM
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    log "stopping server pid=${SERVER_PID} (SIGTERM)"
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    local deadline
    deadline=$(( $(date +%s) + STOP_TIMEOUT ))
    while (( $(date +%s) < deadline )); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      log "server did not stop in ${STOP_TIMEOUT}s; SIGKILL pid=${SERVER_PID}"
      kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
    log "server stopped"
  fi
  exit "${ec}"
}
trap cleanup EXIT INT TERM

# Start the server from the worktree. Let the streamed artifact force AR itself;
# --no-auth keeps the localhost health check key-free.
log "starting: mtplx serve --model ${MODEL} --host ${HOST} --port ${PORT}"
(
  cd "${WORKTREE}" || exit 97
  exec env PYTHONPATH="${WORKTREE}" "${VENV_PY}" -m mtplx.cli serve \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --no-auth ${DSV41_SERVE_EXTRA_ARGS:-}
) >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
log "server pid=${SERVER_PID}"

# Wait for /health (or the server dies first).
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
healthy=0
while (( $(date +%s) < deadline )); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    err "server pid=${SERVER_PID} exited before /health came up; tail of ${SERVER_LOG}:"
    tail -n 40 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if curl -sf -m 5 "${BASE}/health" >/dev/null 2>&1; then
    healthy=1
    break
  fi
  sleep 2
done
if (( healthy == 0 )); then
  err "/health did not come up within ${HEALTH_TIMEOUT}s; tail of ${SERVER_LOG}:"
  tail -n 40 "${SERVER_LOG}" >&2 || true
  exit 1
fi
log "/health is up at ${BASE}/health"

# model id from /v1/models; generation_mode + profile + model_key from /health.
# Each body is piped into the parser AS A FILE, so it reaches the parser's stdin.
HEALTH_JSON="$(curl -sf -m 10 "${BASE}/health" 2>/dev/null || true)"
MODELS_JSON="$(curl -sf -m 10 "${BASE}/v1/models" 2>/dev/null || true)"
MODEL_ID="$(printf '%s' "${MODELS_JSON}" | "${VENV_PY}" "${PARSE}" models 2>/dev/null || true)"
printf '%s' "${HEALTH_JSON}" | "${VENV_PY}" "${PARSE}" health || true
[[ -z "${MODEL_ID}" ]] && MODEL_ID="$(basename "${MODEL}")"
log "served model id (/v1/models) = ${MODEL_ID}"

# One short greedy chat completion; time it and report tok/s.
log "sending one short chat completion (max_tokens=${MAX_TOKENS}, temperature=0)"
REQ="$(
  MID="${MODEL_ID}" MT="${MAX_TOKENS}" "${VENV_PY}" - <<'PYREQ'
import json, os
print(json.dumps({
    "model": os.environ["MID"],
    "messages": [{"role": "user", "content": "In one short sentence, what is a transformer in machine learning?"}],
    "max_tokens": int(os.environ["MT"]),
    "temperature": 0,
    "stream": False,
}))
PYREQ
)"
START_NS=$(date +%s%N 2>/dev/null || python3 -c 'import time;print(int(time.time()*1e9))')
RESP="$(curl -sf -m 120 -H 'Content-Type: application/json' -d "${REQ}" "${BASE}/v1/chat/completions" 2>/dev/null || true)"
END_NS=$(date +%s%N 2>/dev/null || python3 -c 'import time;print(int(time.time()*1e9))')
if [[ -z "${RESP}" ]]; then
  err "chat completion returned no body; tail of ${SERVER_LOG}:"
  tail -n 20 "${SERVER_LOG}" >&2 || true
  exit 1
fi
printf '%s' "${RESP}" | WALL_NS=$(( END_NS - START_NS )) "${VENV_PY}" "${PARSE}" chat || true

log "health check complete; stopping server"
exit 0
