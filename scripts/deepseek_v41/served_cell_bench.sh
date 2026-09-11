#!/usr/bin/env bash
# Serve the DeepSeek-V4.1-Flash mxfp4 streaming artifact on a FREE HIGH PORT
# (never :8080), wait for /health, then run the FABLE server-cell harness
# (scripts/fable/server_cell_bench.py) against it with the SAME prompt
# construction as the Qwen3.8 125B MTPLX PRs (#475/#478/#482/#485/#488):
# rotated hash-pinned filler context + the pinned instruction line, sized to an
# exact templated-token target, PR sampler (temperature 1 / top-p 0.95 /
# top-k 20), seeds 20260829/20260830/20260831, max_tokens 1024, natural stop.
#
# For DeepSeek-V4.1 the harness runs in --model-family deepseek-v41: the DSV4.1
# tokenizer has NO chat template and NO thinking mode, so enable_thinking /
# reasoning_effort are OMITTED and the counter/ids replicate the server's plain
# templateless render ("user: " + content + "\nassistant:", add_special_tokens
# =False, NO leading BOS). See docs/deepseek-v41/W49_QWEN_PR_PROMPT_SHAPE.md.
#
# This is a STEP meant to run *inside* gpu_window.sh (which holds the exclusive
# GPU lock and has already booted out the resident agent). It does NOT take the
# lock or touch launchctl / :8080. It reuses serve_health.sh's port pick /
# engagement guard / clean stop. MTPLX_IGNORE_STOP_TOKENS is deliberately NOT
# set: cells stop naturally, like the PRs.
#
#   # AR (default profile: HEAD_MODE=bf16 + SINKHORN_METAL + ATTN_COMPILE ship
#   #     as served defaults via the profile child_env)
#   DSV41_RECEIPT_DIR=docs/deepseek-v41/receipts/w49-served-cells \
#   bash scripts/deepseek_v41/gpu_window.sh \
#        bash scripts/deepseek_v41/served_cell_bench.sh
#
#   # native MTP head
#   DSV41_SERVE_EXTRA_ARGS="--generation-mode mtp" \
#   DSV41_RECEIPT_DIR=docs/deepseek-v41/receipts/w49-served-cells-mtp \
#   bash scripts/deepseek_v41/gpu_window.sh \
#        bash scripts/deepseek_v41/served_cell_bench.sh
#
# The 16K cell must keep the box under 100 GB. The served plan (W35/W46) is an
# 82 GiB cap / ~75 GiB engine, applied by the model's profile child_env. If the
# served 16K prefill needs a smaller cap, set DSV41_MEMORY_LIMIT_GIB=<N> and it
# is passed through as `mtplx serve --memory-budget <N>GiB`.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "${HERE}/../.." && pwd)"
CAMPAIGN_VENV="${MTPLX_CAMPAIGN_VENV:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv}"
VENV_PY="${MTPLX_VENV_PY:-${CAMPAIGN_VENV}/bin/python3}"
MODEL="${DSV41_MODEL:-${HOME}/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4}"
HOST="${DSV41_HOST:-127.0.0.1}"
HEALTH_TIMEOUT="${DSV41_HEALTH_TIMEOUT:-600}"
STOP_TIMEOUT="${DSV41_STOP_TIMEOUT:-60}"
REQUEST_TIMEOUT="${DSV41_REQUEST_TIMEOUT:-3600}"
MAX_TOKENS="${DSV41_MAX_TOKENS:-1024}"
# THE Qwen-PR standard shape: 1,024 and 16,384 templated tokens. Space-separated
# here (env-overridable), converted to the harness's comma list below.
CONTEXTS="${DSV41_CONTEXTS:-1024 16384}"
# All three production seeds for BOTH 1K and 16K unless overridden.
SEEDS="${DSV41_SEEDS:-20260829 20260830 20260831}"
# Which cells: sweep only by default (the Qwen-PR sized cells). "both" adds the
# vanity ~100-token feel cell.
CELLS="${DSV41_CELLS:-sweep}"
MODEL_FAMILY="${DSV41_MODEL_FAMILY:-deepseek-v41}"
# PR sampler.
TEMPERATURE="${DSV41_TEMPERATURE:-1.0}"
TOP_P="${DSV41_TOP_P:-0.95}"
TOP_K="${DSV41_TOP_K:-20}"
# Never above the 16K sanity gate unless deliberately raised.
STOP_AFTER_CONTEXT="${DSV41_STOP_AFTER_CONTEXT:-16384}"

LOG_DIR="${DSV41_LOG_DIR:-${TMPDIR:-/tmp}/dsv41-served-cell-bench}"
RECEIPT_DIR="${DSV41_RECEIPT_DIR:-${LOG_DIR}/receipts}"
HARNESS="${WORKTREE}/scripts/fable/server_cell_bench.py"
PARSE="${HERE}/serve_health_parse.py"

ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s [served_cell_bench] %s\n' "$(ts)" "$*"; }
err() { printf '%s [served_cell_bench] ERROR: %s\n' "$(ts)" "$*" >&2; }

if [[ ! -x "${VENV_PY}" ]]; then err "venv python not found at ${VENV_PY}"; exit 1; fi
if [[ ! -f "${HARNESS}" ]]; then err "harness not found at ${HARNESS}"; exit 1; fi
if [[ ! -f "${PARSE}" ]]; then err "response parser not found at ${PARSE}"; exit 1; fi
if [[ ! -d "${MODEL}" ]]; then err "model artifact not found at ${MODEL}"; exit 1; fi
mkdir -p "${LOG_DIR}" "${RECEIPT_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SERVER_LOG="${LOG_DIR}/serve-${STAMP}.log"
PROMPT_CACHE="${RECEIPT_DIR}/prompts-${MODEL_FAMILY}.json"
PROMPT_IDS="${RECEIPT_DIR}/prompt-ids-${MODEL_FAMILY}.json"

# comma lists for the harness
CONTEXTS_CSV="$(printf '%s' "${CONTEXTS}" | tr ' ' ',')"
SEEDS_CSV="$(printf '%s' "${SEEDS}" | tr ' ' ',')"

log "worktree=${WORKTREE}"
log "receipt dir -> ${RECEIPT_DIR}"
log "server log  -> ${SERVER_LOG}"
log "contexts=${CONTEXTS_CSV} seeds=${SEEDS_CSV} cells=${CELLS} family=${MODEL_FAMILY}"
log "sampler: temperature=${TEMPERATURE} top_p=${TOP_P} top_k=${TOP_K} max_tokens=${MAX_TOKENS}"

# --- Phase 1: build prompts + exact server token-id lists (CPU, BEFORE serve) -
# Done before the server starts so the tokenizer/sizing CPU never contaminates a
# timed cell (server_cell_bench build_prompt_cache: "Run this OUTSIDE the guarded
# window"). --model-family deepseek-v41 => plain templateless counter, no BOS.
log "building prompts (CPU) at contexts ${CONTEXTS_CSV}, seeds ${SEEDS_CSV}"
if ! (
  cd "${WORKTREE}" && PYTHONPATH="${WORKTREE}" "${VENV_PY}" "${HARNESS}" \
    --mode build-prompts \
    --model-family "${MODEL_FAMILY}" \
    --tokenizer "${MODEL}" \
    --contexts "${CONTEXTS_CSV}" \
    --seeds "${SEEDS_CSV}" \
    --reasoning xhigh \
    --prompt-cache "${PROMPT_CACHE}" \
    --prompt-ids-out "${PROMPT_IDS}"
); then
  err "prompt build failed"; exit 1
fi
log "prompts -> ${PROMPT_CACHE}"
log "prompt ids (exact server ids, for --prompt-ids-file A/Bs) -> ${PROMPT_IDS}"

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
log "selected free port ${PORT} (never :8080)"

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

# --- Phase 2: start the server ------------------------------------------------
# The streamed artifact forces AR itself (or honours --generation-mode mtp via
# DSV41_SERVE_EXTRA_ARGS). --no-auth keeps the localhost health check key-free.
# The profile child_env carries the 82 GiB / 75 GiB served plan; pass
# DSV41_MEMORY_LIMIT_GIB to override the cap for a smaller-cap 16K run.
MEM_ARGS=()
if [[ -n "${DSV41_MEMORY_LIMIT_GIB:-}" ]]; then
  MEM_ARGS=(--memory-budget "${DSV41_MEMORY_LIMIT_GIB}GiB")
  log "memory cap override: --memory-budget ${DSV41_MEMORY_LIMIT_GIB}GiB"
fi
log "starting: mtplx serve --model ${MODEL} --host ${HOST} --port ${PORT} --no-auth ${MEM_ARGS[*]:-} ${DSV41_SERVE_EXTRA_ARGS:-}"
(
  cd "${WORKTREE}" || exit 97
  exec env PYTHONPATH="${WORKTREE}" "${VENV_PY}" -m mtplx.cli serve \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --no-auth "${MEM_ARGS[@]}" ${DSV41_SERVE_EXTRA_ARGS:-}
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

# Surface which decode levers engaged (daemon startup line).
grep -a "DeepSeek-V4.1 decode levers (resolved env):" "${SERVER_LOG}" | tail -n 1 || \
  log "no lever startup line found in ${SERVER_LOG} (older daemon?)"

# Resolve the served model id (also drives family auto-detect + the request 'model').
MODEL_ID="$(curl -sf -m 10 "${BASE}/v1/models" 2>/dev/null \
  | "${VENV_PY}" "${PARSE}" models 2>/dev/null || true)"
[[ -z "${MODEL_ID}" ]] && MODEL_ID="$(basename "${MODEL}")"
log "served model id = ${MODEL_ID}"

# --- Phase 3: run the sweep cells against the running endpoint ----------------
# Pure HTTP client (--mode cells): no server started here, no GPU lock taken.
# User-Agent is pinned server-cell-bench/1 inside the harness so the managed-
# client path never overrides the sampler.
log "running cells against ${BASE} (${CELLS}, ${CONTEXTS_CSV})"
if ! (
  cd "${WORKTREE}" && PYTHONPATH="${WORKTREE}" "${VENV_PY}" "${HARNESS}" \
    --mode cells \
    --base-url "${BASE}" \
    --served-model-id "${MODEL_ID}" \
    --model-family "${MODEL_FAMILY}" \
    --server "${MODEL_FAMILY}" \
    --prompt-cache "${PROMPT_CACHE}" \
    --receipt-dir "${RECEIPT_DIR}" \
    --contexts "${CONTEXTS_CSV}" \
    --seeds "${SEEDS_CSV}" \
    --cells "${CELLS}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --stop-after-context "${STOP_AFTER_CONTEXT}" \
    --timeout-s "${REQUEST_TIMEOUT}"
); then
  err "served cells failed; tail of ${SERVER_LOG}:"
  tail -n 20 "${SERVER_LOG}" >&2 || true
  exit 1
fi

log "served cells complete; receipts under ${RECEIPT_DIR}; stopping server"
exit 0
