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
PACKED_ARTIFACT_REAL="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41/benchmarks/raw/deepseek-v41-resident-scales/20260917"
F2_RING_RECORDS="${F2_RING_RECORDS:-32}"
F2_WORKERS="${F2_WORKERS:-3}"
# F2_PROBE=1 composes the F5 critical-path stamp probe (measured zero-cost in F5 windows 1-2)
# under the F2b wrappers: the F5 stager edits the staged run_full FIRST (hook after
# growth_transition), then the F2b stager adds its install after prime_model, so F2b wraps
# the timed run (same runner instance, same executor/witness local).
F2_PROBE="${F2_PROBE:-0}"
F5DIR="${F5DIR:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f5-compile/scripts/deepseek_v41/f5_compile}"
# Arm grammar: <base>[+si][+eg|+egl]   (arm dir = arm-<token with + -> _>)
#   base: control|control_a|control_b (F2_MAX_ROWS, plain) | control_low (F2_MAX_ROWS-1, plain)
#         | candidate (F2_MAX_ROWS-1, F2b prefetch)
#   +si : GIL switch interval MTPLX_DSV41_GIL_SWITCH_S=$F2_GIL_SWITCH_S (applied once at the
#         F2b hook; the hook is staged but F2b itself stays off for control bases)
#   +eg : F6 parallel Engram miss reads, decode-site install; +egl: load-site (prefill too)
F2_GIL_SWITCH_S="${F2_GIL_SWITCH_S:-0.00005}"
F2_ENGRAM_WORKERS="${F2_ENGRAM_WORKERS:-16}"
F6DIR="${F6DIR:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f6-engram/scripts/deepseek_v41/f6}"
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
stage_tree() {  # $1 dest  $2 max_rows  $3 stage the F2b/GIL hook (0/1)  $4 stage F6 engram (0/1)
  local dest="$1" rows="$2" f2b="$3" eng="${4:-0}"
  if [ -e "$dest" ]; then echo "REFUSE: staged tree exists: $dest"; exit 2; fi
  mkdir -p "$dest"; cp -R "$RETAINED_SRC/." "$dest/"
  # The receipt archives only artifact/manifest.json; the 3.09 GB packed-scale binaries live
  # in the integration worktree and Codex's staging symlinked the DIRECTORY (load_layer uses
  # O_NOFOLLOW on the file only). Without this the prefill->decode transition dies with
  # FileNotFoundError after a full prefill (measured 2026-09-19). Read-only use.
  nice -n 19 "$PYBIN" - "$dest/packed/artifact/manifest.json" "$PACKED_ARTIFACT_REAL" <<'ARTPY'
import hashlib, json, os, sys
archived, real = sys.argv[1], sys.argv[2]
if hashlib.sha256(open(archived, "rb").read()).hexdigest() != hashlib.sha256(open(os.path.join(real, "manifest.json"), "rb").read()).hexdigest():
    sys.exit("REFUSE: packed artifact manifest differs from the archived receipt manifest")
missing, total = [], 0
def walk(node):
    global total
    if isinstance(node, dict):
        if isinstance(node.get("file"), str):
            path = os.path.join(real, node["file"])
            if not os.path.isfile(path) or os.path.islink(path):
                missing.append(node["file"])
            else:
                size = os.path.getsize(path); total += size
                if isinstance(node.get("bytes"), int) and node["bytes"] != size:
                    missing.append(f"{node['file']} (size {size} != {node['bytes']})")
        for value in node.values():
            walk(value)
    elif isinstance(node, list):
        for value in node:
            walk(value)
walk(json.load(open(archived)))
if missing:
    sys.exit("REFUSE: packed artifact files missing/mismatched: " + ", ".join(missing[:8]))
print(f"packed artifact OK: manifest identical, every listed file present ({total} bytes)")
ARTPY
  rm -rf "$dest/packed/artifact"
  ln -s "$PACKED_ARTIFACT_REAL" "$dest/packed/artifact"
  nice -n 19 "$PYBIN" "$F2PKG/f2/stage_f2_runner.py" \
    --admission "$dest/packed/packed_admission.py" --max-rows "$rows"
  if [ "$F2_PROBE" = "1" ]; then
    nice -n 19 "$PYBIN" "$F5DIR/stage_f5_runner.py" \
      --retained "$dest/packed/run_full.py" --out "$dest/packed/run_full.py"
  fi
  if [ "$eng" = "1" ]; then   # after F5 (shares its growth_transition anchor), before F2b
    nice -n 19 "$PYBIN" "$F6DIR/stage_f6_runner.py" \
      --retained "$dest/packed/run_full.py" --out "$dest/packed/run_full.py"
  fi
  if [ "$f2b" = "1" ]; then
    nice -n 19 "$PYBIN" "$F2PKG/f2/stage_f2_runner.py" --run-full "$dest/packed/run_full.py"
  fi
  PYTHONPATH="$F2PKG" nice -n 19 "$PYBIN" -m f2.window_preflight --no-seams \
    $(find "$dest" -maxdepth 2 -name '*.py' | sed 's/^/--compile /')
}
parse_arm() {  # $1 arm token -> A_BASE A_SI A_ENG A_ROWS A_F2B A_HOOK A_ENGSTAGE A_DIRNAME
  local tok="$1" mods
  A_BASE="${tok%%+*}"; A_SI=0; A_ENG=0
  mods="+${tok#*+}+"; [ "$tok" = "$A_BASE" ] && mods="+"
  case "$mods" in *"+si+"*) A_SI=1 ;; esac
  case "$mods" in *"+eg+"*) A_ENG=decode ;; esac
  case "$mods" in *"+egl+"*) A_ENG=load ;; esac
  case "$A_BASE" in
    control|control_a|control_b) A_ROWS="$F2_MAX_ROWS"; A_F2B=0 ;;
    control_low)                 A_ROWS="$((F2_MAX_ROWS-1))"; A_F2B=0 ;;
    candidate)                   A_ROWS="$((F2_MAX_ROWS-1))"; A_F2B=1 ;;
    *) echo "unknown arm base '$A_BASE' in '$tok'"; exit 2 ;;
  esac
  A_HOOK=0; { [ "$A_F2B" = "1" ] || [ "$A_SI" = "1" ]; } && A_HOOK=1
  A_ENGSTAGE=0; [ "$A_ENG" != "0" ] && A_ENGSTAGE=1
  A_TREE="$STAGE_ROOT/r${A_ROWS}-h${A_HOOK}-e${A_ENGSTAGE}"
  A_DIRNAME="$(printf '%s' "$tok" | tr '+' '_')"
}
ARMS="${F2_ARMS:-control_a candidate control_b}"
echo "== stage retained sources -> $STAGE_ROOT (arms: $ARMS; probe=$F2_PROBE) =="
for arm in $ARMS; do   # stage EVERY needed tree up-front: fail before the first unload
  parse_arm "$arm"
  [ -d "$A_TREE" ] || stage_tree "$A_TREE" "$A_ROWS" "$A_HOOK" "$A_ENGSTAGE"
done
if [ "${F2_STAGE_ONLY:-0}" = "1" ]; then   # CPU dry run of the whole staging sequence
  for arm in $ARMS; do parse_arm "$arm"; echo "STAGED $arm -> $A_TREE (rows=$A_ROWS f2b=$A_F2B si=$A_SI engram=$A_ENG)"; done
  rmdir "$RECEIPTS" 2>/dev/null || true
  echo "STAGE ONLY: no GPU window opened"; exit 0
fi

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
run_arm() {  # $1 arm token (parse_arm must have run for it)
  local arm="$A_DIRNAME" tree="$A_TREE" f2b="$A_F2B"
  local dir="$RECEIPTS/arm-${arm}"
  if [ -e "$dir" ]; then echo "REFUSE: $dir exists (never overwrite a measurement)"; exit 2; fi
  mkdir -p "$dir"
  local stem="$OUT_STAGE/f2b-${arm}-${STAMP}"
  local out="${stem}.jsonl"
  local pypath="$RUNWT:$tree/packed:$tree/compat:$F2PKG"
  local probe_env=""
  if [ "$F2_PROBE" = "1" ]; then
    pypath="$pypath:$F5DIR"
    probe_env="MTPLX_DSV41_F5_ENABLE= MTPLX_DSV41_F5_CAPS8=0 MTPLX_DSV41_F5_TIMED_PROBE=1 MTPLX_DSV41_F5_TIMED_OUT=$dir/timed_probe"
  fi
  local f2b_env=""
  [ "$f2b" = "1" ] && f2b_env="MTPLX_DSV41_F2B=1 MTPLX_DSV41_F2B_RECORDS=$F2_RING_RECORDS MTPLX_DSV41_F2B_WORKERS=$F2_WORKERS MTPLX_DSV41_F2B_COUNTERS=$dir/f2b_counters.json"
  [ "$A_SI" = "1" ] && f2b_env="$f2b_env MTPLX_DSV41_GIL_SWITCH_S=$F2_GIL_SWITCH_S"
  if [ "$A_ENG" != "0" ]; then
    pypath="$pypath:$F6DIR"
    f2b_env="$f2b_env MTPLX_DSV41_F6_ENGRAM_PARALLEL=1 MTPLX_DSV41_F6_INSTALL=$A_ENG MTPLX_DSV41_F6_ENGRAM_WORKERS=$F2_ENGRAM_WORKERS"
  fi
  { echo "cd $RUNWT"; echo "PYTHONPATH=$pypath"; echo "$f2b_env $probe_env gpu_window.sh $PYBIN $tree/launch_full.py $(retained_args "$out")"; } > "$dir/command.txt"
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
    PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 $f2b_env $probe_env \
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
for arm in $ARMS; do
  parse_arm "$arm"
  run_arm "$arm"
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
    receipt = next(iter(arm_dir.glob("f2b-*.passes.jsonl")), None)
    if receipt is None:
        print(f"  {arm}: NO RECEIPT (guard.exit={_read(arm_dir/'guard.exit')})"); ok = False; continue
    recs = [json.loads(l) for l in receipt.read_text().splitlines() if l.strip()]
    dspark = next((r for r in recs if r.get("pass") == "dspark"), {})
    rows = dspark.get("decode_slots_per_layer")
    digest = dspark.get("output_ids_sha256")
    tps = dspark.get("decode_tok_s"); wall = dspark.get("decode_wall_s")
    ctr_path = arm_dir / "f2b_counters.json"
    ctr = json.loads(ctr_path.read_text()) if ctr_path.exists() else {}
    rows_by_arm[arm] = rows
    match = (digest == control_sha); ok = ok and match
    print(f"  {arm}: rows={rows} decode_tok_s={tps} decode_wall_s={wall} sha={'OK' if match else 'MISMATCH ' + str(digest)}"
          + (f" f2b={ctr}" if ctr else ""))
ctrl_rows = {a: r for a, r in rows_by_arm.items() if a.startswith("control")}
if len(set(v for v in ctrl_rows.values() if v is not None)) > 1:
    print(f"  WARNING: control arms landed on UNEQUAL rows {ctrl_rows} -- background drift"); ok = False
print("DIGEST GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 4)
PYEOF
echo "== F2b window complete; receipts under $RECEIPTS =="
