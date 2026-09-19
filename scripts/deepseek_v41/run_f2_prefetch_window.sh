#!/usr/bin/env bash
# Runnable guarded GPU window for the F2b lane-private-host-ring prefetch on the retained
# DeepSeek-V4.1 Q4 13.87-TPS cell (16,384-in / 1,024-out, D5/M<=8).
#
# WRITE, DO NOT RUN from a worker: another session holds /tmp/mtplx-gpu-exclusive.lock and
# is measuring. The orchestrator launches this (guarded-window-launch-protocol.md).
#
# F2b keeps the runtime's own prefetch OFF (prefetch_slots==0); a decode MISS is fulfilled
# from a private HOST ring by an intercepted reader. No pinned runtime source is edited.
# Facts baked in (mirrors run_f5_compile_window.sh @ e1dd3fcb4):
#   1. SOURCE PIN. The guarded child runs from the DETACHED run worktree ($RUNWT, HEAD ==
#      the pinned source_commit); the F2b package rides on PYTHONPATH; receipts are copied
#      back here. The CPU preflight re-checks the pin BEFORE the service is unloaded.
#   2. STAGING. Copy the receipt-archived sources into a fresh /private/tmp staging dir
#      (NEVER the pinned original) and apply anchored, round-trip edits (stage_f2_runner.py):
#      the equal-capacity admission cap on every arm, plus the F2b install (after prime_model)
#      + traceback surfacing on the candidate arm's run_full.
#   3. EQUAL-CAPACITY LADDER. Admission takes the largest capacity that fits the live
#      post-unload baseline. F2_MAX_ROWS (default 108) caps it: controls at F2_MAX_ROWS,
#      candidate at F2_MAX_ROWS-1 + F2b, and control_low at F2_MAX_ROWS-1. The F2b ring is
#      HOST RAM (~566 MB at R=32), not MLX; one dropped row/layer is 707,788,800 B of MLX
#      active -- larger -- so the candidate's TOTAL physical <= a control's (receipt).
#   4/5. --out must be a fresh /tmp/dsv41-110-stage/<stem>.jsonl (run_full.py:328-338);
#      sidecars <stem>.{bounds,passes,os}.* land beside it and are copied into the arm dir.
#      Guard exit per arm: continue only on 0; 4 (digest != control) is a FAILURE for F2b
#      (prefetch is exact); any other code stops the ladder. Lock wait 7200s.
set -euo pipefail

# --------------------------------------------------------------------------- paths
WT="${WT:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f2-prefetch}"
RUNWT="${RUNWT:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-run-d5f15e7a}"
PYBIN="${PYBIN:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python}"
F2PKG="$WT/scripts/deepseek_v41"                     # parent of the f2/ package
RETAINED_SRC="$WT/docs/deepseek-v41/receipts/extension-bank-20260919/full/sources"
PACKED_INSTALLATION="${PACKED_INSTALLATION:-/private/tmp/dsv41-extension-bank-20260919/full-v1/packed/installation.json}"
COMPAT_INSTALLATION="${COMPAT_INSTALLATION:-/private/tmp/dsv41-extension-bank-20260919/full-v1/compat/installation.json}"
STRICT_LIB="${STRICT_LIB:-/private/tmp/dsv41-strict-cache-20260918/strict-lib}"
MODEL_DIR="${MODEL_DIR:-/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4}"
AUX_DIR="${AUX_DIR:-/tmp/dsv41-compact-residents}"
PROMPT_IDS="docs/deepseek-v41/receipts/memory-budget-110/python-prompt-ids.json"   # relative to RUNWT
AR_REFERENCE="${DSV41_STAGE_AR_REFERENCE:-/tmp/dsv41-110-stage/live-combined-depth3-reserve2-1023.jsonl}"
CONTROL_SHA="${CONTROL_SHA:-0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac}"

F2_MAX_ROWS="${F2_MAX_ROWS:-108}"
F2_RING_RECORDS="${F2_RING_RECORDS:-32}"
F2_WORKERS="${F2_WORKERS:-3}"
STAMP="$(date +%Y%m%d-%H%M%S)"
STAGE_ROOT="/private/tmp/dsv41-f2b-${STAMP}"
OUT_STAGE="/tmp/dsv41-110-stage"                     # run_full.py:330 requires --out here
RECEIPTS="$WT/docs/deepseek-v41/receipts/f2-prefetch-build-20260919/window-${STAMP}"

cd "$WT"
mkdir -p "$RECEIPTS" "$OUT_STAGE"

# --------------------------------------------------- 1. CPU preflight (before unload)
echo "== F2b preflight (CPU; MLX pinned; before any service unload) =="
PYTHONPATH="$RETAINED_SRC/packed:$RUNWT:$F2PKG" nice -n 19 "$PYBIN" -m f2.window_preflight \
  --run-worktree "$RUNWT" \
  --compat-installation "$COMPAT_INSTALLATION" \
  --packed-installation "$PACKED_INSTALLATION" \
  --archived-dir "$RETAINED_SRC" \
  --dep "$STRICT_LIB" --dep "$RUNWT/$PROMPT_IDS" --dep "$MODEL_DIR"

# --------------------------------------------------- 2. stage patched runner copies
stage_tree() {  # $1 dest  $2 max_rows  $3 install_f2b(0/1)
  local dest="$1" rows="$2" f2b="$3"
  if [ -e "$dest" ]; then echo "REFUSE: staged tree exists: $dest"; exit 2; fi
  mkdir -p "$dest"; cp -R "$RETAINED_SRC/." "$dest/"
  nice -n 19 "$PYBIN" "$F2PKG/f2/stage_f2_runner.py" \
    --admission "$dest/packed/packed_admission.py" --max-rows "$rows"
  [ "$f2b" = "1" ] && nice -n 19 "$PYBIN" "$F2PKG/f2/stage_f2_runner.py" \
    --run-full "$dest/packed/run_full.py"
  nice -n 19 "$PYBIN" -m f2.window_preflight --no-seams \
    $(find "$dest" -maxdepth 2 -name '*.py' | sed 's/^/--compile /')
}
STAGE_CONTROL="$STAGE_ROOT/control"       # F2_MAX_ROWS, no F2b
STAGE_CANDIDATE="$STAGE_ROOT/candidate"   # F2_MAX_ROWS-1, F2b install + traceback
STAGE_CTRL_LOW="$STAGE_ROOT/control-low"  # F2_MAX_ROWS-1, no F2b (isolate the lost row)
echo "== stage retained sources -> $STAGE_ROOT (control@${F2_MAX_ROWS}, candidate@$((F2_MAX_ROWS-1))+F2b) =="
stage_tree "$STAGE_CONTROL" "$F2_MAX_ROWS" 0
stage_tree "$STAGE_CANDIDATE" "$((F2_MAX_ROWS-1))" 1
[ "${F2_INCLUDE_CONTROL_LOW:-0}" = "1" ] && stage_tree "$STAGE_CTRL_LOW" "$((F2_MAX_ROWS-1))" 0

# ---------------------------------------------------------- retained arg list (fixed)
retained_args() {  # $1 = absolute --out under /tmp/dsv41-110-stage
  printf '%s ' \
    --model "$MODEL_DIR" --arms cell16k_ring_v2_draft_attn_pf0 \
    --context-tokens 16384 --decode-tokens 1023 \
    --decode-mode dspark --dspark-depth 5 --dspark-require-tie-class \
    --max-kv 17664 --prompt-ids-file "$PROMPT_IDS" --prompt-seed 20260829 \
    --stop-on-eos --box-target-gb 110 \
    --host-overhead-gib 1.3399620056152344 --allocator-cache-gib 1 \
    --runtime-reserve-gib 2 --transient-band-gib auto \
    --expert-profile deepseek-v41-mxfp4-75 --slot-layout component-banks \
    --transient-slots 48 --apply-memory-cap --cache-policy transition-window \
    --verify-shared-overlap --decode-miss-records-per-part 3 --kv-cache-bits 16 \
    --out "$1"
}

# ------------------------------------------------------------------ per-arm launcher
run_arm() {  # $1 arm  $2 staged tree  $3 f2b(0/1)
  local arm="$1" tree="$2" f2b="$3"
  local dir="$RECEIPTS/arm-${arm}"
  if [ -e "$dir" ]; then echo "REFUSE: $dir exists (never overwrite a measurement)"; exit 2; fi
  mkdir -p "$dir"
  local stem="$OUT_STAGE/f2b-${arm}-${STAMP}"
  local out="${stem}.jsonl"
  local pypath="$RUNWT:$tree/packed:$tree/compat:$F2PKG"
  local f2b_env=""
  [ "$f2b" = "1" ] && f2b_env="MTPLX_DSV41_F2B=1 MTPLX_DSV41_F2B_RECORDS=$F2_RING_RECORDS MTPLX_DSV41_F2B_WORKERS=$F2_WORKERS MTPLX_DSV41_F2B_COUNTERS=$dir/f2b_counters.json"
  { echo "cd $RUNWT"; echo "PYTHONPATH=$pypath"; echo "$f2b_env gpu_window.sh $PYBIN $tree/launch_full.py $(retained_args "$out")"; } > "$dir/command.txt"
  echo "== arm ${arm}: tree=$tree f2b=${f2b} out=${out} =="
  cd "$RUNWT"
  # shellcheck disable=SC2086
  env \
    MTPLX_ENGRAM_CACHE_LIMIT=67108864 DSV41_CACHE_GROWTH=1 \
    DSV41_STAGE_AR_REFERENCE="$AR_REFERENCE" \
    GPU_WINDOW_LOCK_TIMEOUT="${F2_LOCK_TIMEOUT:-7200}" \
    GPU_WINDOW_TOTAL_MEM_CEILING_BYTES=110000000000 \
    GPU_WINDOW_MIN_AVAIL_GB=100 GPU_WINDOW_RESTORE_QWEN_ALWAYS=1 \
    GPU_WINDOW_CANDIDATE_MODEL_DIR="$MODEL_DIR" GPU_WINDOW_CANDIDATE_AUX_DIR="$AUX_DIR" \
    MTPLX_DSV41_IO_READ_FANOUT=4 MTPLX_BELADY_ORACLE=0 \
    PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 $f2b_env \
    PYTHONPATH="$pypath" \
    scripts/deepseek_v41/gpu_window.sh "$PYBIN" "$tree/launch_full.py" \
      $(retained_args "$out") > "$dir/guard.log" 2>&1 && rc=0 || rc=$?
  echo "$rc" > "$dir/guard.exit"
  # copy the run_full receipt + sidecars (<stem>.*) into the arm dir.
  cp "${stem}".* "$dir/" 2>/dev/null || true
  if [ "$rc" = "4" ]; then
    echo "FAIL: arm ${arm} guard exit 4 (output digest != control). F2b is EXACT -> this is a BUG. Stopping."
    tail -8 "$dir/guard.log" || true; exit 4
  fi
  if [ "$rc" != "0" ]; then
    echo "ABORT: arm ${arm} guard exit ${rc}; see $dir/guard.log -- remaining arms NOT run."
    tail -8 "$dir/guard.log" || true; exit "$rc"
  fi
}

# ------------------------------------------------------------------------- the arms
ARMS="${F2_ARMS:-control_a candidate control_b}"
for arm in $ARMS; do
  case "$arm" in
    control_a)   run_arm control_a   "$STAGE_CONTROL"  0 ;;
    candidate)   run_arm candidate   "$STAGE_CANDIDATE" 1 ;;
    control_b)   run_arm control_b   "$STAGE_CONTROL"  0 ;;
    control_low) run_arm control_low "$STAGE_CTRL_LOW" 0 ;;
    *) echo "unknown arm '$arm'"; exit 2 ;;
  esac
done

# --------------------------------- readout: rows / digest / F2b counters
echo "== F2b readout: admitted rows / digest / host-ring counters =="
PYTHONPATH="$WT" nice -n 19 "$PYBIN" - "$RECEIPTS" "$CONTROL_SHA" <<'PYEOF'
import json, sys
from pathlib import Path
def _read(p):
    try: return p.read_text().strip()
    except Exception: return "?"
receipts, control_sha = Path(sys.argv[1]), sys.argv[2]
rows_by_arm, ok = {}, True
for arm_dir in sorted(p for p in receipts.iterdir() if p.is_dir()):
    arm = arm_dir.name.replace("arm-", "")
    receipt = next(iter(arm_dir.glob("f2b-*.jsonl")), None)
    if receipt is None:
        print(f"  {arm}: NO RECEIPT (guard.exit={_read(arm_dir/'guard.exit')})"); ok = False; continue
    recs = [json.loads(l) for l in receipt.read_text().splitlines() if l.strip()]
    rows = next((r.get("decode_slots_per_layer") or r.get("post_prefill_growth", {}).get("decode_slots_per_layer")
                 for r in recs if r.get("decode_slots_per_layer") or r.get("post_prefill_growth")), None)
    digest = next((r.get("token_ids_sha256") or r.get("output_ids_sha256") for r in recs
                   if r.get("token_ids_sha256") or r.get("output_ids_sha256")), None)
    ctr_path = arm_dir / "f2b_counters.json"
    ctr = json.loads(ctr_path.read_text()) if ctr_path.exists() else {}
    rows_by_arm[arm] = rows
    match = (digest == control_sha); ok = ok and match
    print(f"  {arm}: rows={rows} sha={'OK' if match else 'MISMATCH ' + str(digest)}"
          + (f" f2b={ctr}" if ctr else ""))
ctrl_rows = {a: r for a, r in rows_by_arm.items() if a.startswith("control")}
if len(set(v for v in ctrl_rows.values() if v is not None)) > 1:
    print(f"  WARNING: control arms landed on UNEQUAL rows {ctrl_rows} -- background drift"); ok = False
print("DIGEST GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 4)
PYEOF
echo "== F2b window complete; receipts under $RECEIPTS =="
