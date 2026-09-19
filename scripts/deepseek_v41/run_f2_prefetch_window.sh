#!/usr/bin/env bash
# Runnable guarded GPU window for the F2 next-layer expert prefetch lane on the
# retained DeepSeek-V4.1 Q4 13.87-TPS cell (16,384-in / 1,024-out, D5/M<=8).
#
# WRITE, DO NOT RUN from a worker: another session holds /tmp/mtplx-gpu-exclusive.lock
# and is measuring. The orchestrator launches this (guarded-window-launch-protocol.md:
# the DIRECT command of a run_in_background:true Bash call, never a shell `&`).
#
# Modelled on the F5 window that ran through preflight+staging today
# (.../dsv41-f5-compile/scripts/deepseek_v41/run_f5_compile_window.sh). Key facts baked in:
#   1. SOURCE PIN. The retained run_full.py refuses unless `git rev-parse HEAD` in its CWD
#      == the pinned source_commit and 11 runtime sources hash identically. So the guarded
#      child runs from the DETACHED run worktree ($RUNWT, HEAD == pin); the F2 package rides
#      on PYTHONPATH; receipts are written back here by absolute --out. The CPU preflight
#      re-checks the pin BEFORE the service is unloaded.
#   2. STAGING. run_full imports helpers by module name (PYTHONPATH order decides), while
#      self-checking the pinned originals. So we copy the receipt-archived sources into a
#      fresh /private/tmp/dsv41-f2-prefetch-<stamp>/ (NEVER the pinned original) and apply
#      anchored, round-trip-checked edits to the STAGED copies (stage_f2_runner.py).
#   3. EQUAL-CAPACITY LADDER. Admission takes the largest capacity that fits the live
#      post-unload baseline (today ~109 rows, not 111). F2_MAX_ROWS (default 108) caps the
#      search: controls at F2_MAX_ROWS, candidate at F2_MAX_ROWS-1 AND +R-record ring, plus
#      an optional control at F2_MAX_ROWS-1 to separate the ring from the lost row. Rows are
#      recorded per arm and unequal rows are flagged.
#   4/5. Guard exit per arm: continue only on 0; 4 (digest rejection) is a FAILURE for F2
#      (prefetch is exact, so the digest MUST equal the control's); any other code stops the
#      ladder (never open another window on an unverified service state). Lock wait 7200s.
#
# NOTE (see the receipt "ring-enable gap"): the CONTROL arms are runnable as-is. The
# CANDIDATE arm additionally needs the runtime built with prefetch_slots=R and
# packed_phase.install_growth taught to accept the ring (it refuses one today,
# packed_phase.py:41-42/54/64-65). Those edits are GPU-unverifiable and NOT shipped blind.
set -euo pipefail

# --------------------------------------------------------------------------- paths
WT="${WT:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f2-prefetch}"
RUNWT="${RUNWT:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-run-d5f15e7a}"
PYBIN="${PYBIN:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python}"
F2DIR="$WT/scripts/deepseek_v41/f2"
RETAINED_SRC="$WT/docs/deepseek-v41/receipts/extension-bank-20260919/full/sources"
PACKED_INSTALLATION="${PACKED_INSTALLATION:-/private/tmp/dsv41-extension-bank-20260919/full-v1/packed/installation.json}"
COMPAT_INSTALLATION="${COMPAT_INSTALLATION:-/private/tmp/dsv41-extension-bank-20260919/full-v1/compat/installation.json}"
STRICT_LIB="${STRICT_LIB:-/private/tmp/dsv41-strict-cache-20260918/strict-lib}"

MODEL_DIR="${MODEL_DIR:-/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4}"
AUX_DIR="${AUX_DIR:-/tmp/dsv41-compact-residents}"
PROMPT_IDS="docs/deepseek-v41/receipts/memory-budget-110/python-prompt-ids.json"   # relative to RUNWT cwd
AR_REFERENCE="${DSV41_STAGE_AR_REFERENCE:-/tmp/dsv41-110-stage/live-combined-depth3-reserve2-1023.jsonl}"
CONTROL_SHA="0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac"

F2_MAX_ROWS="${F2_MAX_ROWS:-108}"
F2_RING_RECORDS="${F2_RING_RECORDS:-32}"
STAMP="$(date +%Y%m%d-%H%M%S)"
STAGE_ROOT="/private/tmp/dsv41-f2-prefetch-${STAMP}"
RECEIPTS="$WT/docs/deepseek-v41/receipts/f2-prefetch-build-20260919/window-${STAMP}"

cd "$WT"
mkdir -p "$RECEIPTS"

# --------------------------------------------------- 1. CPU preflight (before unload)
echo "== F2 preflight (CPU; MLX pinned; before any service unload) =="
PYTHONPATH="$RUNWT:$F2DIR/.." MTPLX_DSV41_SINGLE_SLOT_POOL=1 nice -n 19 "$PYBIN" -m f2.window_preflight \
  --run-worktree "$RUNWT" \
  --compat-installation "$COMPAT_INSTALLATION" \
  --packed-installation "$PACKED_INSTALLATION" \
  --archived-dir "$RETAINED_SRC" \
  --dep "$STRICT_LIB" \
  --dep "$RUNWT/$PROMPT_IDS" \
  --dep "$MODEL_DIR" \
  --dep "$RETAINED_SRC/packed/run_full.py"
# window_preflight exits 1 on any failure; `set -e` aborts BEFORE staging/unload.

# --------------------------------------------------- 2. stage patched runner copies
# A fresh staged tree per capacity/ring geometry (never the pinned original).
stage_tree() {  # $1 dest  $2 max_rows  $3 ring_records  $4 install_swap(0/1)
  local dest="$1" rows="$2" ring="$3" swap="$4"
  if [ -e "$dest" ]; then echo "REFUSE: staged tree exists: $dest"; exit 2; fi
  mkdir -p "$dest"; cp -R "$RETAINED_SRC/." "$dest/"
  nice -n 19 "$PYBIN" "$F2DIR/stage_f2_runner.py" \
    --admission "$dest/packed/packed_admission.py" --max-rows "$rows" --ring-records "$ring"
  [ "$swap" = "1" ] && nice -n 19 "$PYBIN" "$F2DIR/stage_f2_runner.py" \
    --packed-phase "$dest/packed/packed_phase.py"
  # every staged .py must compile before it is ever imported under the lock.
  nice -n 19 "$PYBIN" -m f2.window_preflight --no-seams \
    $(find "$dest" -maxdepth 2 -name '*.py' | sed 's/^/--compile /')
}
STAGE_CONTROL="$STAGE_ROOT/control"      # F2_MAX_ROWS, no ring, retained install
STAGE_CANDIDATE="$STAGE_ROOT/candidate"  # F2_MAX_ROWS-1, +R ring, F2 install swap
STAGE_CTRL_LOW="$STAGE_ROOT/control-low" # F2_MAX_ROWS-1, no ring (isolate the lost row)
echo "== stage retained sources -> $STAGE_ROOT (control@${F2_MAX_ROWS}, candidate@$((F2_MAX_ROWS-1))+R${F2_RING_RECORDS}) =="
stage_tree "$STAGE_CONTROL" "$F2_MAX_ROWS" 0 0
stage_tree "$STAGE_CANDIDATE" "$((F2_MAX_ROWS-1))" "$F2_RING_RECORDS" 1
[ "${F2_INCLUDE_CONTROL_LOW:-0}" = "1" ] && stage_tree "$STAGE_CTRL_LOW" "$((F2_MAX_ROWS-1))" 0 0

# ---------------------------------------------------------- retained arg list (fixed)
retained_args() {  # $1 = absolute out path
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
run_arm() {  # $1 arm  $2 staged tree
  local arm="$1" tree="$2"
  local dir="$RECEIPTS/arm-${arm}"
  if [ -e "$dir" ]; then echo "REFUSE: $dir exists (never overwrite a measurement)"; exit 2; fi
  mkdir -p "$dir"
  local out="$dir/result.jsonl"
  local pypath="$RUNWT:$tree/packed:$tree/compat:$F2DIR/.."
  # record the exact per-arm command line (after expansion) so it can be diffed vs command.sh.
  { echo "cd $RUNWT"; echo "PYTHONPATH=$pypath"; echo "gpu_window.sh $PYBIN $tree/launch_full.py $(retained_args "$out")"; } > "$dir/command.txt"
  echo "== arm ${arm}: tree=$tree =="
  cd "$RUNWT"
  env \
    MTPLX_ENGRAM_CACHE_LIMIT=67108864 DSV41_CACHE_GROWTH=1 \
    DSV41_STAGE_AR_REFERENCE="$AR_REFERENCE" \
    GPU_WINDOW_LOCK_TIMEOUT="${F2_LOCK_TIMEOUT:-7200}" \
    GPU_WINDOW_TOTAL_MEM_CEILING_BYTES=110000000000 \
    GPU_WINDOW_MIN_AVAIL_GB=100 GPU_WINDOW_RESTORE_QWEN_ALWAYS=1 \
    GPU_WINDOW_CANDIDATE_MODEL_DIR="$MODEL_DIR" GPU_WINDOW_CANDIDATE_AUX_DIR="$AUX_DIR" \
    MTPLX_DSV41_IO_READ_FANOUT=4 MTPLX_BELADY_ORACLE=0 MTPLX_DSV41_SINGLE_SLOT_POOL=1 \
    PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 \
    PYTHONPATH="$pypath" \
    scripts/deepseek_v41/gpu_window.sh "$PYBIN" "$tree/launch_full.py" \
      $(retained_args "$out") > "$dir/guard.log" 2>&1 && rc=0 || rc=$?
  echo "$rc" > "$dir/guard.exit"
  if [ "$rc" = "4" ]; then
    echo "FAIL: arm ${arm} guard exit 4 (output digest != control). Prefetch is EXACT, so this is a BUG, not a tie. Stopping."
    tail -8 "$dir/guard.log" || true; exit 4
  fi
  if [ "$rc" != "0" ]; then
    echo "ABORT: arm ${arm} guard exit ${rc}; see $dir/guard.log -- remaining arms NOT run (unverified service state)."
    tail -8 "$dir/guard.log" || true; exit "$rc"
  fi
}

# ------------------------------------------------------------------------- the arms
ARMS="${F2_ARMS:-control_a candidate control_b}"
for arm in $ARMS; do
  case "$arm" in
    control_a)   run_arm control_a   "$STAGE_CONTROL" ;;
    candidate)   run_arm candidate   "$STAGE_CANDIDATE" ;;
    control_b)   run_arm control_b   "$STAGE_CONTROL" ;;
    control_low) run_arm control_low "$STAGE_CTRL_LOW" ;;
    *) echo "unknown arm '$arm'"; exit 2 ;;
  esac
done

# --------------------------------- readout: rows (equal-capacity), digest, counters
echo "== F2 readout: admitted rows / digest / prefetch counters =="
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
    receipt = arm_dir / "result.jsonl"
    if not receipt.exists():
        print(f"  {arm}: NO RECEIPT (guard.exit={_read(arm_dir/'guard.exit')})"); ok = False; continue
    recs = [json.loads(l) for l in receipt.read_text().splitlines() if l.strip()]
    rows = next((r.get("decode_slots_per_layer") or r.get("post_prefill_growth", {}).get("decode_slots_per_layer")
                 for r in recs if r.get("decode_slots_per_layer") or r.get("post_prefill_growth")), None)
    digest = next((r.get("token_ids_sha256") or r.get("output_ids_sha256") for r in recs
                   if r.get("token_ids_sha256") or r.get("output_ids_sha256")), None)
    ctr = next((r.get("prefetch_counters") for r in recs if r.get("prefetch_counters")), {})
    rows_by_arm[arm] = rows
    match = (digest == control_sha)
    ok = ok and match
    print(f"  {arm}: rows={rows} sha={'OK' if match else 'MISMATCH ' + str(digest)}"
          + (f" prefetch={ctr}" if ctr else ""))
# equal-capacity: controls must share a row count; candidate is one lower by design.
ctrl_rows = {a: r for a, r in rows_by_arm.items() if a.startswith("control")}
if len(set(v for v in ctrl_rows.values() if v is not None)) > 1:
    print(f"  WARNING: control arms landed on UNEQUAL rows {ctrl_rows} -- background drift; compare with care"); ok = False
print("DIGEST GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 4)
PYEOF
echo "== F2 window complete; receipts under $RECEIPTS =="
