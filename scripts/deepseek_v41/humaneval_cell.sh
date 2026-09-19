#!/usr/bin/env bash
# One HumanEval(164) pass@1 cell for the DeepSeek-V4.1-Flash streaming artifact,
# per decode lane: serve it on a FREE HIGH PORT (never :8080), assert /health,
# then run humaneval_cell.py against it at David's sampler and write an
# append-only receipt (strict + completed-task pass@1 + truncation rate) that
# records the lane and the daemon's resolved decode-lever env. Stops the server
# cleanly on exit.
#
# This is a STEP meant to run *inside* gpu_window.sh, which holds the exclusive
# GPU lock and has already booted out the resident agent. It therefore does NOT
# take the lock or touch launchctl / :8080 itself.
#
#   # AR lane (served default; the profile arms the byte-identical decode levers:
#   #   head bf16 + Sinkhorn kernel + attention compile + window memo)
#   DSV41_LANE=ar \
#   DSV41_RECEIPT_DIR=docs/deepseek-v41/receipts/w68-humaneval-ar \
#   bash scripts/deepseek_v41/gpu_window.sh \
#        bash scripts/deepseek_v41/humaneval_cell.sh
#
#   # DSpark-DIRECT lane (W57): mtplx serve --load-mtp --generation-mode dspark
#   #   --depth 3
#   DSV41_LANE=dspark \
#   DSV41_RECEIPT_DIR=docs/deepseek-v41/receipts/w68-humaneval-dspark \
#   bash scripts/deepseek_v41/gpu_window.sh \
#        bash scripts/deepseek_v41/humaneval_cell.sh
#
# Reuses serve_health.sh's port finder + editable-install engagement assertion
# (memory/editable-install-cwd-shadowing.md): runs the WORKTREE's code
# (PYTHONPATH + cwd = worktree) and asserts mtplx resolves under it before
# serving, and served_cell_bench.sh's serve lifecycle (free port, engagement
# assertion, --expert-memory-limit passthrough, clean stop, decode-lever
# startup-line grep). The SSD SessionBank cold tier is turned OFF so the shared
# prod bank never warms this correctness cell
# (memory/ssd-session-bank-warms-benchmarks.md); HumanEval sends 164 distinct
# prompts, so cross-request restore is irrelevant.
#
# Env knobs:
#   DSV41_LANE=ar|dspark          which decode lane to gate (default ar)
#   DSV41_DEPTH=3                 DSpark draft depth (dspark lane only)
#   DSV41_RECEIPT_DIR=<dir>       append-only receipt root (default
#                                 .benchmark-artifacts/deepseek-v41; DSV41_OUT_DIR
#                                 still honoured as a fallback alias)
#   DSV41_MEMORY_LIMIT_GIB=<N>    -> mtplx serve --expert-memory-limit <N>GiB
#   DSV41_MAX_LIVE_KV_TOKENS=<N>  -> mtplx serve --expert-max-live-kv-tokens <N>
#   DSV41_WORKERS=<N>             concurrent in-flight requests (default 4)
#   DSV41_SERVE_EXTRA_ARGS="..."  extra serve args, appended verbatim
#
#   --dry-run : prove wiring on CPU (paths, venv, driver) with NO serve, NO
#               model, NO code execution -- invokes humaneval_cell.py --dry-run.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "${HERE}/../.." && pwd)"
# The campaign venv lives in the MAIN checkout, not the worktree; PYTHONPATH=WORKTREE
# makes the worktree's editable mtplx win (memory/editable-install-cwd-shadowing.md).
CAMPAIGN_VENV="${MTPLX_CAMPAIGN_VENV:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv}"
VENV_PY="${MTPLX_VENV_PY:-${CAMPAIGN_VENV}/bin/python3}"
MODEL="${DSV41_MODEL:-${HOME}/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4}"
HOST="${DSV41_HOST:-127.0.0.1}"
HEALTH_TIMEOUT="${DSV41_HEALTH_TIMEOUT:-600}"   # server load = residents + admit
STOP_TIMEOUT="${DSV41_STOP_TIMEOUT:-60}"
LOG_DIR="${DSV41_LOG_DIR:-${TMPDIR:-/tmp}/dsv41-humaneval-cell}"
DATASET="${HUMANEVAL_DATASET:-/Users/davidtai/projects/OpenSourceWTF/benchmark-archive/datasets/HumanEval.jsonl}"
# Append-only receipt root: DSV41_RECEIPT_DIR is the primary knob; DSV41_OUT_DIR
# stays as a back-compat alias so older window scripts keep working.
RECEIPT_DIR="${DSV41_RECEIPT_DIR:-${DSV41_OUT_DIR:-${WORKTREE}/.benchmark-artifacts/deepseek-v41}}"
LIMIT="${HUMANEVAL_LIMIT:-}"                     # empty = all 164 tasks
WORKERS="${DSV41_WORKERS:-4}"
LANE="${DSV41_LANE:-ar}"
DEPTH="${DSV41_DEPTH:-3}"
DRY_RUN="0"
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN="1" ;;
  esac
done

ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s [humaneval_cell] %s\n' "$(ts)" "$*"; }
err() { printf '%s [humaneval_cell] ERROR: %s\n' "$(ts)" "$*" >&2; }

if [[ ! -x "${VENV_PY}" ]]; then
  err "venv python not found/executable at ${VENV_PY}"
  exit 1
fi

# Resolve the decode lane -> serve flags. AR is the served default (no lane
# flags); DSpark is the W57 DSpark-DIRECT lane.
SERVE_LANE_ARGS=()
case "${LANE}" in
  ar)
    SERVE_FLAGS_DESC="(served AR default)"
    ;;
  dspark)
    SERVE_LANE_ARGS=(--load-mtp --generation-mode dspark --depth "${DEPTH}")
    SERVE_FLAGS_DESC="--load-mtp --generation-mode dspark --depth ${DEPTH}"
    ;;
  *)
    err "unknown DSV41_LANE '${LANE}' (expected ar|dspark)"
    exit 2
    ;;
esac

# -------------------------------- dry run ------------------------------------
# Prove the wiring on CPU with no serve, no model, no code execution.
if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY-RUN: no serve, no model, no code execution"
  log "lane=${LANE} serve flags: ${SERVE_FLAGS_DESC}"
  log "would serve: mtplx serve --model ${MODEL} --host ${HOST} --port <free 18080-18299> --no-auth --ssd-session-cache off ${SERVE_LANE_ARGS[*]:-}"
  log "would score: humaneval_cell.py --dataset-path ${DATASET} --out-dir ${RECEIPT_DIR}"
  exec env PYTHONPATH="${WORKTREE}" "${VENV_PY}" "${HERE}/humaneval_cell.py" \
    --dry-run \
    --out-dir "${RECEIPT_DIR}" \
    --dataset-path "${DATASET}" \
    --lane "${LANE}" \
    --depth "${DEPTH}" \
    --serve-flags "${SERVE_FLAGS_DESC}" \
    --label "dry-run"
fi

if [[ ! -d "${MODEL}" ]]; then
  err "model artifact not found at ${MODEL}"
  exit 1
fi
if [[ ! -f "${DATASET}" ]]; then
  err "HumanEval dataset not found at ${DATASET} (set HUMANEVAL_DATASET)"
  exit 1
fi
mkdir -p "${LOG_DIR}" "${RECEIPT_DIR}"
SERVER_LOG="${LOG_DIR}/serve-${LANE}-$(date -u +%Y%m%dT%H%M%SZ).log"

# Never :8080. Pick the first free port in a high range (serve_health.sh's finder).
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
log "lane=${LANE}; serve flags: ${SERVE_FLAGS_DESC}"
log "selected free port ${PORT} (never :8080); server log -> ${SERVER_LOG}"
log "receipt dir -> ${RECEIPT_DIR}"

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

# --expert-memory-limit passthrough (served_cell_bench.sh contract): the accepted
# flag is a size string like "60GiB"; NOT --memory-budget (the serve parser
# rejects that). Optional --expert-max-live-kv-tokens for finer KV control.
MEM_ARGS=()
if [[ -n "${DSV41_MEMORY_LIMIT_GIB:-}" ]]; then
  MEM_ARGS+=(--expert-memory-limit "${DSV41_MEMORY_LIMIT_GIB}GiB")
  log "expert memory ceiling override: --expert-memory-limit ${DSV41_MEMORY_LIMIT_GIB}GiB"
fi
if [[ -n "${DSV41_MAX_LIVE_KV_TOKENS:-}" ]]; then
  MEM_ARGS+=(--expert-max-live-kv-tokens "${DSV41_MAX_LIVE_KV_TOKENS}")
  log "context window override: --expert-max-live-kv-tokens ${DSV41_MAX_LIVE_KV_TOKENS}"
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

# Start the server from the worktree. The SSD SessionBank cold tier is OFF so
# the shared prod bank never warms this cell (ssd-session-bank-warms-benchmarks).
log "starting: mtplx serve --model ${MODEL} --host ${HOST} --port ${PORT} --no-auth --ssd-session-cache off ${SERVE_LANE_ARGS[*]:-} ${MEM_ARGS[*]:-} ${DSV41_SERVE_EXTRA_ARGS:-}"
(
  cd "${WORKTREE}" || exit 97
  exec env PYTHONPATH="${WORKTREE}" "${VENV_PY}" -m mtplx.cli serve \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --no-auth \
    --ssd-session-cache off \
    "${SERVE_LANE_ARGS[@]}" "${MEM_ARGS[@]}" ${DSV41_SERVE_EXTRA_ARGS:-}
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

# Surface which decode levers engaged (daemon startup line); humaneval_cell.py
# re-scrapes the same line from --server-log into the receipt.
grep -a "DeepSeek-V4.1 decode levers (resolved env):" "${SERVER_LOG}" | tail -n 1 || \
  log "no lever startup line found in ${SERVER_LOG} (older daemon?)"

# generation_mode + model_key from /health; model id from /v1/models.
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
MODEL_KEY="$(
  printf '%s' "${HEALTH_JSON}" | "${VENV_PY}" - <<'PYKEY' 2>/dev/null || true
import json, sys
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
try:
    health = json.load(sys.stdin)
except Exception:
    print("")
    sys.exit(0)
print(find_first(health, "model_key") or "")
PYKEY
)"
[[ -z "${MODEL_ID}" ]] && MODEL_ID="$(basename "${MODEL}")"
log "served model id (/v1/models) = ${MODEL_ID}; model_key = ${MODEL_KEY:-<none>}"

# Run the one HumanEval(164) pass@1 cell against the served endpoint.
CELL_ARGS=(
  --base-url "${BASE}"
  --model "${MODEL_ID}"
  --dataset-path "${DATASET}"
  --out-dir "${RECEIPT_DIR}"
  --lane "${LANE}"
  --depth "${DEPTH}"
  --serve-flags "${SERVE_FLAGS_DESC}"
  --server-log "${SERVER_LOG}"
  --workers "${WORKERS}"
)
[[ -n "${MODEL_KEY}" ]] && CELL_ARGS+=(--spec-key "${MODEL_KEY}")
[[ -n "${LIMIT}" ]] && CELL_ARGS+=(--limit "${LIMIT}")
[[ -n "${DSV41_MEMORY_LIMIT_GIB:-}" ]] && CELL_ARGS+=(--expert-memory-limit "${DSV41_MEMORY_LIMIT_GIB}GiB")

log "running HumanEval(164) pass@1 cell, lane=${LANE} (temperature 1, top-p 0.95, top-k 20, seed 20260829, non-binding cap)"
(
  cd "${WORKTREE}" || exit 97
  exec env PYTHONPATH="${WORKTREE}" "${VENV_PY}" "${HERE}/humaneval_cell.py" \
    "${CELL_ARGS[@]}"
)
CELL_RC=$?
log "HumanEval cell exited with code ${CELL_RC}; stopping server"
exit "${CELL_RC}"
