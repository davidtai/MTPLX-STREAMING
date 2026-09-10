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
MODEL="${DSV41_MODEL:-${HOME}/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2}"
HOST="${DSV41_HOST:-127.0.0.1}"
HEALTH_TIMEOUT="${DSV41_HEALTH_TIMEOUT:-600}"   # server load = 8.67 GB residents + admit
STOP_TIMEOUT="${DSV41_STOP_TIMEOUT:-60}"
MAX_TOKENS="${DSV41_MAX_TOKENS:-16}"
LOG_DIR="${DSV41_LOG_DIR:-${TMPDIR:-/tmp}/dsv41-serve-health}"

ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s [serve_health] %s\n' "$(ts)" "$*"; }
err() { printf '%s [serve_health] ERROR: %s\n' "$(ts)" "$*" >&2; }

if [[ ! -x "${VENV_PY}" ]]; then
  err "venv python not found/executable at ${VENV_PY}"
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
    --no-auth
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

# generation_mode + model key from /health, model id from /v1/models.
HEALTH_JSON="$(curl -sf -m 10 "${BASE}/health" 2>/dev/null || true)"
MODELS_JSON="$(curl -sf -m 10 "${BASE}/v1/models" 2>/dev/null || true)"
MODEL_ID="$(
  printf '%s' "${MODELS_JSON}" | "${VENV_PY}" - <<'PYMID' 2>/dev/null || true
import json, sys
try:
    data = json.load(sys.stdin).get("data") or []
    print(data[0]["id"] if data else "")
except Exception:
    print("")
PYMID
)"
printf '%s' "${HEALTH_JSON}" | "${VENV_PY}" - <<'PYHEALTH' 2>/dev/null || true
import json, sys, time

def find_first(obj, key):
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if key in cur and cur[key] is not None:
                return cur[key]
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return None

stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
try:
    health = json.load(sys.stdin)
except Exception:
    print(f"{stamp} [serve_health] /health was not valid JSON")
    sys.exit(0)
gen = find_first(health, "generation_mode")
key = find_first(health, "model_key")
print(f"{stamp} [serve_health] generation_mode = {gen!r}")
print(f"{stamp} [serve_health] model_key       = {key!r}")
PYHEALTH
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
printf '%s' "${RESP}" | WALL_NS=$(( END_NS - START_NS )) "${VENV_PY}" - <<'PYRESP' || true
import json, os, sys, time

stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
wall_s = max(1e-9, int(os.environ.get("WALL_NS", "0")) / 1e9)

def find_first(obj, key):
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if key in cur and isinstance(cur[key], (int, float)):
                return cur[key]
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return None

try:
    resp = json.load(sys.stdin)
except Exception:
    print(f"{stamp} [serve_health] chat response was not valid JSON")
    sys.exit(0)

usage = resp.get("usage") or {}
comp = usage.get("completion_tokens")
prompt = usage.get("prompt_tokens")
# Prefer a server-reported decode tok/s if the response carries one.
server_tok_s = find_first(resp, "decode_tok_s") or find_first(resp, "tok_s")
text = ""
try:
    text = (resp["choices"][0]["message"]["content"] or "").strip()
except Exception:
    pass
print(f"{stamp} [serve_health] usage: prompt_tokens={prompt} completion_tokens={comp}")
if isinstance(server_tok_s, (int, float)) and server_tok_s > 0:
    print(f"{stamp} [serve_health] tok/s (server-reported) = {server_tok_s:.2f}")
if isinstance(comp, int) and comp > 0:
    print(f"{stamp} [serve_health] tok/s (wall {wall_s:.2f}s) = {comp / wall_s:.2f}")
else:
    print(f"{stamp} [serve_health] no completion_tokens in usage; wall {wall_s:.2f}s")
print(f"{stamp} [serve_health] completion: {text[:200]!r}")
PYRESP

log "health check complete; stopping server"
exit 0
