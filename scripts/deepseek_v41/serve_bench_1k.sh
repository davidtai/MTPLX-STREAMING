#!/usr/bin/env bash
# Serve the DeepSeek-V4.1-Flash mxfp4 streaming artifact on a FREE HIGH PORT
# (never :8080), wait for /health, then send the SAME 1,024-token prefill_bench
# prompt the decode-lever bench harness builds (as a RAW token-id list on
# /v1/completions, max_tokens=256, temperature=0), write a JSON receipt with the
# server-side decode/prefill tok/s + TTFT + completion sha, and stop the server.
#
# This is the served counterpart to serve_health.sh's 18-token smoke: the lever
# wins were measured on a 1,024-prefill + 256-decode shape, so the served rate
# must be re-measured on that shape (a 16-token completion cannot show a decode
# RATE gain). Reuses serve_health.sh's port pick / engagement guard / clean stop.
#
# Runs INSIDE gpu_window.sh (holds the exclusive GPU lock, already booted the
# resident agent out); does NOT take the lock or touch launchctl / :8080.
#
#   # AR (default profile: HEAD_MODE=bf16 + SINKHORN_METAL + ATTN_COMPILE +
#   #     ATTN_WIN_MEMO ship as served defaults via the profile child_env)
#   bash scripts/deepseek_v41/gpu_window.sh \
#        bash scripts/deepseek_v41/serve_bench_1k.sh
#
#   # native MTP head
#   DSV41_SERVE_EXTRA_ARGS="--generation-mode mtp" \
#   bash scripts/deepseek_v41/gpu_window.sh \
#        bash scripts/deepseek_v41/serve_bench_1k.sh
#
# The startup log line "[4/6] DeepSeek-V4.1 decode levers (resolved env): ..."
# (server log) shows exactly which levers engaged inside the daemon.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "${HERE}/../.." && pwd)"
CAMPAIGN_VENV="${MTPLX_CAMPAIGN_VENV:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv}"
VENV_PY="${MTPLX_VENV_PY:-${CAMPAIGN_VENV}/bin/python3}"
MODEL="${DSV41_MODEL:-${HOME}/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4}"
HOST="${DSV41_HOST:-127.0.0.1}"
HEALTH_TIMEOUT="${DSV41_HEALTH_TIMEOUT:-600}"
STOP_TIMEOUT="${DSV41_STOP_TIMEOUT:-60}"
MAX_TOKENS="${DSV41_MAX_TOKENS:-256}"
CONTEXT_TOKENS="${DSV41_CONTEXT_TOKENS:-1024}"
REQUEST_TIMEOUT="${DSV41_REQUEST_TIMEOUT:-600}"
# Fixed-step decode-rate probe (W18/W46): the bench harness force-decodes 256
# steps IGNORING EOS, and the raw prefill_bench prompt's greedy first token is
# EOS -- so an EOS-honouring server stops at 1 blank token. Set (default on)
# MTPLX_IGNORE_STOP_TOKENS so THIS dedicated benchmark server decodes the full
# max_tokens, matching the harness. Never set on the shared :8080 serve.
IGNORE_STOP_TOKENS="${DSV41_IGNORE_STOP_TOKENS:-1}"
LOG_DIR="${DSV41_LOG_DIR:-${TMPDIR:-/tmp}/dsv41-serve-bench-1k}"
BENCH="${HERE}/serve_bench_1k.py"

ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s [serve_bench_1k] %s\n' "$(ts)" "$*"; }
err() { printf '%s [serve_bench_1k] ERROR: %s\n' "$(ts)" "$*" >&2; }

if [[ ! -x "${VENV_PY}" ]]; then err "venv python not found at ${VENV_PY}"; exit 1; fi
if [[ ! -f "${BENCH}" ]]; then err "bench client not found at ${BENCH}"; exit 1; fi
if [[ ! -d "${MODEL}" ]]; then err "model artifact not found at ${MODEL}"; exit 1; fi
mkdir -p "${LOG_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SERVER_LOG="${LOG_DIR}/serve-${STAMP}.log"
# Append-only receipt (never-overwrite-a-measurement): default is stamp-suffixed.
RECEIPT="${DSV41_BENCH_RECEIPT:-${LOG_DIR}/receipt-${STAMP}.json}"

# Never :8080. First free port in a high range.
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
  err "refusing to serve on port '${PORT}' (never :8080)"; exit 1
fi
BASE="http://${HOST}:${PORT}"
log "selected free port ${PORT} (never :8080); server log -> ${SERVER_LOG}"
log "receipt -> ${RECEIPT}"

# Editable-install engagement guard: the served code MUST be this worktree's.
log "asserting mtplx resolves to the worktree ${WORKTREE}"
if ! (cd "${WORKTREE}" && PYTHONPATH="${WORKTREE}" "${VENV_PY}" - "${WORKTREE}" <<'PYASSERT'
import sys, mtplx
root = sys.argv[1]
if not mtplx.__file__.startswith(root):
    sys.exit(f"mtplx resolved to {mtplx.__file__}, not under {root}")
PYASSERT
); then
  err "engagement assertion failed; refusing to serve the wrong tree"; exit 1
fi

SERVER_PID=""
cleanup() {
  local ec=$?
  trap - EXIT INT TERM
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    log "stopping server pid=${SERVER_PID} (SIGTERM)"
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    local deadline; deadline=$(( $(date +%s) + STOP_TIMEOUT ))
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

# Start the server from the worktree. Let the streamed artifact force AR itself
# (or honour --generation-mode mtp via DSV41_SERVE_EXTRA_ARGS). --no-auth keeps
# the localhost health check key-free. The profile child_env carries the lever
# defaults; export MTPLX_DSV41_* in this shell to override one for an A/B.
log "starting: mtplx serve --model ${MODEL} --host ${HOST} --port ${PORT} ${DSV41_SERVE_EXTRA_ARGS:-} (MTPLX_IGNORE_STOP_TOKENS=${IGNORE_STOP_TOKENS})"
(
  cd "${WORKTREE}" || exit 97
  exec env PYTHONPATH="${WORKTREE}" MTPLX_IGNORE_STOP_TOKENS="${IGNORE_STOP_TOKENS}" \
    "${VENV_PY}" -m mtplx.cli serve \
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
    err "server pid=${SERVER_PID} exited before /health; tail of ${SERVER_LOG}:"
    tail -n 40 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if curl -sf -m 5 "${BASE}/health" >/dev/null 2>&1; then healthy=1; break; fi
  sleep 2
done
if (( healthy == 0 )); then
  err "/health did not come up within ${HEALTH_TIMEOUT}s; tail of ${SERVER_LOG}:"
  tail -n 40 "${SERVER_LOG}" >&2 || true
  exit 1
fi
log "/health is up at ${BASE}/health"

# Surface which decode levers engaged (the daemon startup line).
grep -a "DeepSeek-V4.1 decode levers (resolved env):" "${SERVER_LOG}" | tail -n 1 || \
  log "no lever startup line found in ${SERVER_LOG} (older daemon?)"

# Resolve the served model id.
MODEL_ID="$(curl -sf -m 10 "${BASE}/v1/models" 2>/dev/null \
  | "${VENV_PY}" "${HERE}/serve_health_parse.py" models 2>/dev/null || true)"
[[ -z "${MODEL_ID}" ]] && MODEL_ID="$(basename "${MODEL}")"
log "served model id = ${MODEL_ID}"

# Send the SAME 1,024-token prefill_bench prompt (raw token-id list) + 256 greedy
# decode, and write the receipt. build_prompt_ids loads the real tokenizer
# (metadata read, no weights) so the ids are byte-identical to the bench harness.
log "sending 1,024-token prefill_bench prompt (max_tokens=${MAX_TOKENS}, temperature=0)"
if ! (
  cd "${WORKTREE}" && PYTHONPATH="${WORKTREE}" "${VENV_PY}" "${BENCH}" \
    --base-url "${BASE}" \
    --model "${MODEL}" \
    --model-id "${MODEL_ID}" \
    --context-tokens "${CONTEXT_TOKENS}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature 0 \
    --timeout-s "${REQUEST_TIMEOUT}" \
    --out "${RECEIPT}" \
    --print
); then
  err "served bench request failed; tail of ${SERVER_LOG}:"
  tail -n 20 "${SERVER_LOG}" >&2 || true
  exit 1
fi

log "served 1K bench complete; receipt at ${RECEIPT}; stopping server"
exit 0
