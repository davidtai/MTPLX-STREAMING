#!/usr/bin/env bash
# Guarded GPU window for the F2 next-layer expert prefetch lane on the exact
# extension-bank 16,384-in / 1,024-out Q4 workload.  WRITE-ONLY artifact: the
# orchestrator launches this; the author does NOT run it (it needs the exclusive
# GPU lock another session holds).
#
# It arms control / candidate / control (A/B/A) in ONE guarded window through the
# existing scripts/deepseek_v41/gpu_window.sh protocol:
#   * A (control)   = the retained extension-bank best run, prefetch OFF (pf0).
#   * B (candidate) = the same runner + the F2 verify next-layer prefetch lane
#                     (f2_prefetch_lane.install at the post-prefill boundary).
#   * A (control)   = control again, to bound background drift across the window.
# Each arm writes a FRESH receipt dir (never overwrites a prior receipt), stores
# the full 1,024 output token ids, and asserts their sha256 against the retained
# native-MTP digest so a candidate that changes the output fails loudly:
#   0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac
#
# Before ANY GPU work it runs a CPU admission preflight: the retained 111-slot
# launch estimate PLUS the speculative ring (32 records x 17,694,720 B) must fit
# under the 110,000,000,000 B whole-machine ceiling; if not it refuses here,
# before model load (AGENTS.md: fail once, clearly, before measured generation).
#
# If three full arms cannot fit one window's time/thermal envelope, run with
# F2_ARMS="candidate control" (two arms) -- see the note printed at the end.
#
# Launch (guarded-window-launch-protocol.md): the DIRECT command of a
# run_in_background:true Bash call -- never a shell `&` one-liner; do not stop the
# launching agent mid-window.
set -euo pipefail

# --- paths (override via env) ---------------------------------------------
WT="${WT:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f2-prefetch}"
DSV41_WT="${DSV41_WT:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41}"
PY="${PY:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python}"
SCRATCH="${SCRATCH:-/tmp/dsv41-extension-bank-20260919/full-v1}"           # staged extension-bank runner
LANE_DIR="${LANE_DIR:-$WT/docs/deepseek-v41/receipts/f2-prefetch-build-20260919}"
PRED_DIR="${PRED_DIR:-$WT/scripts/deepseek_v41}"                            # f2_predictor.py lives here
MODEL="${MODEL:-/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4}"
AUX="${AUX:-/tmp/dsv41-compact-residents}"
PROMPT_IDS="${PROMPT_IDS:-docs/deepseek-v41/receipts/memory-budget-110/python-prompt-ids.json}"
AR_REF="${AR_REF:-/tmp/dsv41-110-stage/live-combined-depth3-reserve2-1023.jsonl}"
OUTROOT="${OUTROOT:-/tmp/dsv41-f2-prefetch-20260919}"
DIGEST="0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac"
RING_RECORDS="${RING_RECORDS:-32}"
F2_ARMS="${F2_ARMS:-control candidate control}"   # A/B/A; set "candidate control" for 2 arms

# --- (0) CPU admission preflight: refuse before ANY GPU work if the ring does
#         not fit on top of the retained 111-slot launch estimate. -----------
# base_launch_bytes is the extension-bank README "Launch physical estimate".
BASE_LAUNCH_BYTES="${BASE_LAUNCH_BYTES:-109745344620}"
echo "[f2-window] CPU admission preflight (no GPU): 111 slots + ${RING_RECORDS}-record ring vs 110e9 ceiling"
PYTHONPATH="$DSV41_WT:$PRED_DIR" nice -n 19 "$PY" - "$BASE_LAUNCH_BYTES" "$RING_RECORDS" <<'PYEOF'
import sys, importlib.abc
class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, n, path=None, target=None):
        if n == "mlx" or n.startswith("mlx."):
            raise RuntimeError("preflight is CPU-only; MLX forbidden")
sys.meta_path.insert(0, NoMLX())
import f2_predictor as F
base = int(sys.argv[1]); ring = int(sys.argv[2])
charge = F.ring_charge_bytes(ring)
ok = F.admits_with_ring(base_launch_bytes=base, ring_records=ring)
print(f"[f2-window]   ring charge = {charge} B; base launch = {base} B; "
      f"sum = {base + charge} B; ceiling = {F.MACHINE_CEILING_BYTES} B; fits = {ok}")
if not ok:
    sys.exit("[f2-window] REFUSED before model load: 111 slots + ring exceeds the "
             f"{F.MACHINE_CEILING_BYTES} B whole-machine ceiling. Reduce RING_RECORDS "
             "or free background memory; do NOT raise the ceiling.")
PYEOF
echo "[f2-window] admission preflight passed."

mkdir -p "$OUTROOT"

# --- shared env for every arm (mirrors extension-bank full/command.sh) ------
COMMON_ENV=(
  MTPLX_ENGRAM_CACHE_LIMIT=67108864
  DSV41_CACHE_GROWTH=1
  DSV41_STAGE_AR_REFERENCE="$AR_REF"
  GPU_WINDOW_LOCK_TIMEOUT=600
  GPU_WINDOW_TOTAL_MEM_CEILING_BYTES=110000000000
  GPU_WINDOW_MIN_AVAIL_GB=100
  GPU_WINDOW_RESTORE_QWEN_ALWAYS=1
  GPU_WINDOW_CANDIDATE_MODEL_DIR="$MODEL"
  GPU_WINDOW_CANDIDATE_AUX_DIR="$AUX"
  MTPLX_DSV41_IO_READ_FANOUT=4
  MTPLX_BELADY_ORACLE=0
  PYTHONHASHSEED=0
  PYTHONUNBUFFERED=1
)
# extension-bank base runner argv (D5 + two lookup tokens, native KV16, 84+27):
BASE_ARGS=(
  --model "$MODEL"
  --arms cell16k_ring_v2_draft_attn_pf0
  --context-tokens 16384 --decode-tokens 1023
  --decode-mode dspark --dspark-depth 5 --dspark-require-tie-class
  --max-kv 17664
  --prompt-ids-file "$PROMPT_IDS" --prompt-seed 20260829 --stop-on-eos
  --box-target-gb 110 --host-overhead-gib 1.3399620056152344
  --allocator-cache-gib 1 --runtime-reserve-gib 2 --transient-band-gib auto
  --expert-profile deepseek-v41-mxfp4-75 --slot-layout component-banks
  --transient-slots 48 --apply-memory-cap --cache-policy transition-window
  --verify-shared-overlap --decode-miss-records-per-part 3 --kv-cache-bits 16
)

# One guarded window; the launcher below loads the model ONCE and runs the A/B/A
# arms in sequence (control uses stock switch._run; candidate installs the F2 lane
# at the post-prefill boundary; the second control uninstalls it).  Chaining the
# arms inside one window (not one gpu_window.sh per arm) keeps a single model load
# and one background baseline for the A/B/A comparison.
run_window() {
  local arms="$1"
  PYTHONPATH="$DSV41_WT:$SCRATCH/packed:$SCRATCH/compat:$LANE_DIR:$PRED_DIR" \
  env "${COMMON_ENV[@]}" \
      MTPLX_DSV41_F2_ARMS="$arms" \
      MTPLX_DSV41_F2_RING_RECORDS="$RING_RECORDS" \
      MTPLX_DSV41_F2_OUTROOT="$OUTROOT" \
      MTPLX_DSV41_F2_DIGEST="$DIGEST" \
      MTPLX_DSV41_F2_BASE_LAUNCH_BYTES="$BASE_LAUNCH_BYTES" \
    "$WT/scripts/deepseek_v41/gpu_window.sh" \
      "$PY" "$LANE_DIR/f2_window_arms.py" "${BASE_ARGS[@]}"
}

echo "[f2-window] arming ONE guarded window, arms: $F2_ARMS"
echo "[f2-window]   fresh per-arm receipt dirs under $OUTROOT ; digest gate $DIGEST"
run_window "$F2_ARMS"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[f2-window] window exited rc=$rc."
  echo "[f2-window] If it aborted on the time/thermal envelope with three full arms,"
  echo "[f2-window] re-run two arms:   F2_ARMS='candidate control' bash $0"
  exit "$rc"
fi
echo "[f2-window] done. Per-arm output ids + sha256 comparisons are under $OUTROOT."
