#!/usr/bin/env bash
# F5 compile-lever GPU window for the retained DeepSeek-V4.1 Q4 13.87-TPS cell.
#
# WRITE, DO NOT RUN blindly.  This script measures the dispatch-reduction compile
# levers (HC_COMPILE / ATTN_COMPILE / SMALL_STAGES_FUSED) for the M<=8 DSpark verify
# decode, which the retained run left OFF (bound false for prefill-memory provenance).
#
# Safety contract (AGENTS.md GPU-and-memory-safety; memory: guarded-window-launch):
#   * The GPU lock (/tmp/mtplx-gpu-exclusive.lock) is taken ONLY by
#     scripts/deepseek_v41/gpu_window.sh, which also stops+restores the Qwen service
#     and enforces the 100 GiB wired cap.  This script NEVER touches Metal directly.
#   * The CPU preflight below refuses BEFORE the service is unloaded if any
#     dependency is missing or background memory makes 111 decode rows inadmissible.
#   * Each arm runs the EXACT retained config/admission via a STAGED patched copy of
#     run_full.py (two anchored, round-trip-checked edits; see stage_f5_runner.py) --
#     the committed receipt is never modified.
#   * Fresh receipt dir per arm; nothing is overwritten (an existing arm dir aborts).
#   * The HEADLINE tok/s pass is UNTIMED (no --stage-timing): the stage recorder
#     forces these levers eager, so a timed pass would measure them OFF.
#   * Every arm's receipt carries engagement counters (fused vs eager) -- arm_env is
#     NOT proof of engagement.  Each arm reports verify_ms/cycle AND cycles (a
#     changed token stream changes the cycle count; TPS alone is not comparable).
#   * On a digest change the receipt's W120 divergence classifier (kept LIVE by the
#     staged runner) yields a tie_flip/divergent VERDICT with the contested-logit /
#     tie-band keys, not a null (the HC-screen gap at index 480).
#
# Arms:  A  control (reproduce 0d54d9b2...)          F  control again (drift check)
#        A2 control + TimedPackedDecode stamp probe (zero-distortion critical-path)
#        B  +HC_COMPILE      C +HC_COMPILE+ATTN_COMPILE
#        D  +SMALL_STAGES_FUSED (+ATTN_COMPILE)      E  best of B-D + caps@8 (optional)
set -euo pipefail

# --------------------------------------------------------------------------- paths
WT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f5-compile"
# The retained runner refuses unless `git rev-parse HEAD` in its CWD equals the measured
# source commit and every pinned runtime source hashes identically (run_full.py ~L280).
# So the guarded child runs from a DETACHED worktree at exactly that commit; the F5
# helpers ride on PYTHONPATH and receipts are written back here by absolute path.
RUNWT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-run-d5f15e7a"
COMPAT_INSTALLATION="/private/tmp/dsv41-extension-bank-20260919/full-v1/compat/installation.json"
PYBIN="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
F5DIR="$WT/scripts/deepseek_v41/f5_compile"
RETAINED_SRC="$WT/docs/deepseek-v41/receipts/extension-bank-20260919/full/sources"
RETAINED_RUNNER="$RETAINED_SRC/packed/run_full.py"

MODEL_DIR="/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4"
AUX_DIR="/tmp/dsv41-compact-residents"
PROMPT_IDS="docs/deepseek-v41/receipts/memory-budget-110/python-prompt-ids.json"
AR_REFERENCE="${DSV41_STAGE_AR_REFERENCE:-/tmp/dsv41-110-stage/live-combined-depth3-reserve2-1023.jsonl}"
CONTROL_SHA="0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac"

STAMP="$(date +%Y%m%d-%H%M%S)"
STAGE="/private/tmp/dsv41-f5-compile-${STAMP}"
RECEIPTS="$WT/docs/deepseek-v41/receipts/f5-compile-levers-20260919/window-${STAMP}"

cd "$WT"
mkdir -p "$RECEIPTS"

# --------------------------------------------------------- 1. CPU preflight (refuse)
echo "== F5 preflight (CPU; before any service unload) =="
PYTHONPATH="$WT" nice -n 19 "$PYBIN" "$F5DIR/f5_preflight.py" \
  --retained-runner "$RETAINED_RUNNER" \
  --ar-reference "$AR_REFERENCE" \
  --prompt-ids "$WT/$PROMPT_IDS" \
  --model-dir "$MODEL_DIR" \
  --aux-dir "$AUX_DIR" \
  --installation-json "$RETAINED_SRC/packed/installation.json" \
  --report "$RECEIPTS/preflight.json"
# f5_preflight exits 3 on any failure; `set -e` aborts here BEFORE staging/unload.
echo "== source pin preflight (the check that killed the first attempt, now BEFORE unload) =="
nice -n 19 "$PYBIN" - "$RUNWT" "$COMPAT_INSTALLATION" <<'PINPY'
import hashlib, json, subprocess, sys
from pathlib import Path
root, compat = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
if head != compat["source_commit"]:
    sys.exit(f"REFUSE: run worktree HEAD {head} != pinned {compat['source_commit']}")
dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
if dirty:
    sys.exit("REFUSE: run worktree has tracked modifications:\n" + dirty)
bad = [p for p, d in compat["runtime_source_sha256"].items()
       if hashlib.sha256((root / p).read_bytes()).hexdigest() != d]
if bad:
    sys.exit(f"REFUSE: pinned runtime sources differ: {bad}")
print("source pin OK:", head, f"({len(compat['runtime_source_sha256'])} runtime sources match)")
PINPY

# --------------------------------------------------- 2. stage the F5-patched runner
echo "== stage retained sources + apply the 2 F5 edits to run_full.py =="
mkdir -p "$STAGE"
cp -R "$RETAINED_SRC/." "$STAGE/"
# The receipt archives only artifact/manifest.json; the 3.09 GB packed-scale binaries
# live in the integration worktree and Codex's own staging symlinked them
# (full-v1/packed/artifact -> .../benchmarks/raw/deepseek-v41-resident-scales/20260917).
# packed_phase loads ROOT/artifact/<file> with O_NOFOLLOW on the FILE, so a directory
# symlink is exactly the retained layout.  Read-only use.
PACKED_ARTIFACT_REAL="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41/benchmarks/raw/deepseek-v41-resident-scales/20260917"
nice -n 19 "$PYBIN" - "$STAGE/packed/artifact/manifest.json" "$PACKED_ARTIFACT_REAL" <<'ARTPY'
import hashlib, json, os, sys
archived, real = sys.argv[1], sys.argv[2]
if hashlib.sha256(open(archived, "rb").read()).hexdigest() != hashlib.sha256(open(os.path.join(real, "manifest.json"), "rb").read()).hexdigest():
    sys.exit("REFUSE: packed artifact manifest differs from the archived receipt manifest")
inventory = json.load(open(archived))
missing, total = [], 0
def walk(node):
    global total
    if isinstance(node, dict):
        if "file" in node and isinstance(node["file"], str):
            path = os.path.join(real, node["file"])
            if not os.path.isfile(path) or os.path.islink(path):
                missing.append(node["file"])
            else:
                total += os.path.getsize(path)
                for key in ("bytes", "size", "nbytes", "file_bytes"):
                    if key in node and isinstance(node[key], int) and node[key] != os.path.getsize(path):
                        missing.append(node["file"] + f" (size {os.path.getsize(path)} != {node[key]})")
                        break
        for value in node.values():
            walk(value)
    elif isinstance(node, list):
        for value in node:
            walk(value)
walk(inventory)
if missing:
    sys.exit("REFUSE: packed artifact files missing/mismatched: " + ", ".join(missing[:8]))
print(f"packed artifact OK: manifest identical, every listed file present ({total} bytes)")
ARTPY
rm -rf "$STAGE/packed/artifact"
ln -s "$PACKED_ARTIFACT_REAL" "$STAGE/packed/artifact"
nice -n 19 "$PYBIN" "$F5DIR/stage_f5_runner.py" \
  --retained "$RETAINED_RUNNER" \
  --out "$STAGE/packed/run_full.py"
# f5_decode_levers + timed_plane_lane are imported by the staged runner at the
# post-prefill boundary; expose them on PYTHONPATH alongside the staged helpers.
PYPATH="$RUNWT:$STAGE/packed:$STAGE/compat:$F5DIR"
if [ -n "${F5_MAX_ROWS:-}" ]; then
  echo "== equal-capacity staging: decode capacity search capped at ${F5_MAX_ROWS} rows/layer =="
  nice -n 19 "$PYBIN" "$F5DIR/stage_f5_runner.py" \
    --retained "$RETAINED_RUNNER" --out "$STAGE/packed/run_full.py" \
    --admission "$STAGE/packed/packed_admission.py" --max-rows "$F5_MAX_ROWS"
fi

# ---------------------------------------------------------- retained arg list (fixed)
retained_args() {  # $1 = out path
  printf '%s ' \
    --model "$MODEL_DIR" \
    --arms cell16k_ring_v2_draft_attn_pf0 \
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
run_arm() {  # $1 arm  $2 F5_ENABLE  $3 F5_CAPS8(0/1)  $4 F5_TIMED_PROBE(0/1)
  local arm="$1" enable="$2" caps8="$3" timed="$4"
  local dir="$RECEIPTS/arm-${arm}"
  if [ -e "$dir" ]; then
    echo "REFUSE: $dir exists (never overwrite a measurement)"; exit 2
  fi
  mkdir -p "$dir"
  # run_full.py gate (~L330): --out must be a FRESH .jsonl directly under
  # /tmp/dsv41-110-stage (its sidecars .bounds.json/.passes.jsonl/.os.jsonl and a
  # .rejected-output.json land beside it).  Everything is copied into the arm dir after.
  local stem="f5-${STAMP}-${arm}"
  local out="/tmp/dsv41-110-stage/${stem}.jsonl"
  if ls /tmp/dsv41-110-stage/${stem}.* >/dev/null 2>&1; then
    echo "REFUSE: stage evidence for ${stem} already exists"; exit 2
  fi
  local probe_out="$dir/timed_probe"
  echo "== arm ${arm}: F5_ENABLE='${enable}' CAPS8=${caps8} TIMED=${timed} =="
  cd "$RUNWT"
  # Guard env is IDENTICAL to the retained command.sh; only the F5 arm env and --out
  # differ.  gpu_window.sh takes the lock, stops Qwen, sets _GPU_WINDOW_LOCKED=1,
  # runs the child, restores Qwen, releases the lock.
  env \
    MTPLX_ENGRAM_CACHE_LIMIT=67108864 \
    DSV41_CACHE_GROWTH=1 \
    DSV41_STAGE_AR_REFERENCE="$AR_REFERENCE" \
    GPU_WINDOW_LOCK_TIMEOUT="${F5_LOCK_TIMEOUT:-1800}" \
    GPU_WINDOW_TOTAL_MEM_CEILING_BYTES=110000000000 \
    GPU_WINDOW_MIN_AVAIL_GB=100 \
    GPU_WINDOW_RESTORE_QWEN_ALWAYS=1 \
    GPU_WINDOW_CANDIDATE_MODEL_DIR="$MODEL_DIR" \
    GPU_WINDOW_CANDIDATE_AUX_DIR="$AUX_DIR" \
    MTPLX_DSV41_IO_READ_FANOUT=4 \
    MTPLX_BELADY_ORACLE=0 \
    PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 \
    MTPLX_DSV41_F5_ENABLE="$enable" \
    MTPLX_DSV41_F5_CAPS8="$caps8" \
    MTPLX_DSV41_F5_TIMED_PROBE="$timed" \
    MTPLX_DSV41_F5_TIMED_OUT="$probe_out" \
    PYTHONPATH="$PYPATH" \
    scripts/deepseek_v41/gpu_window.sh "$PYBIN" "$STAGE/launch_full.py" \
      $(retained_args "$out") \
      > "$dir/guard.log" 2>&1 && rc=0 || rc=$?
  echo "$rc" > "$dir/guard.exit"
  cp -p /tmp/dsv41-110-stage/${stem}.* "$dir/" 2>/dev/null || true
  if [ "$rc" != "0" ] && [ "$rc" != "4" ]; then
    # 4 = output-digest rejection (expected for a rounding-class lever arm; the
    # readout classifies it).  Anything else (2 = refused/lock timeout, 10 =
    # RESTORE_FAILED, memory-guard kills...) must stop the whole window: never open
    # another window on top of an unverified service state.
    echo "ABORT: arm ${arm} guard exit ${rc}; see $dir/guard.log -- remaining arms NOT run"
    tail -5 "$dir/guard.log" || true
    exit "$rc"
  fi
  [ "$rc" = "4" ] && echo "  (guard exit 4 -- output digest differs from control; readout classifies it)"

  # readout: normal receipt if the digest matched, else the .rejected-output.json.
  local receipt="$dir/${stem}.jsonl"
  [ -f "$receipt" ] || receipt="$dir/${stem}.rejected-output.json"
  local probe_summary=""
  [ "$timed" = "1" ] && probe_summary="$probe_out.summary.json"
  PYTHONPATH="$WT" nice -n 19 "$PYBIN" "$F5DIR/f5_readout.py" \
    --receipt "$receipt" --arm "$arm" --control-sha "$CONTROL_SHA" \
    ${probe_summary:+--probe-summary "$probe_summary"} \
    --report "$dir/readout.json" || echo "  (readout could not parse a receipt for arm ${arm})"
}

# ------------------------------------------------------------------------- the arms
# F5_ARMS selects a subset (space separated), default = the full ladder.
ARMS="${F5_ARMS:-A A2 B C D F}"
for arm in $ARMS; do
  case "$arm" in
    A)  run_arm A   ""                          0 0 ;;  # control: reproduce 0d54d9b2...
    A2) run_arm A2  ""                          0 1 ;;  # control + TimedPackedDecode stamp probe
    B)  run_arm B   "hc_compile"                0 0 ;;
    C)  run_arm C   "hc_compile,attn_compile"   0 0 ;;
    D)  run_arm D   "small_stages,attn_compile" 0 0 ;;
    E)  run_arm E   "${F5_E_ENABLE:-hc_compile,attn_compile}" 1 0 ;;  # + caps@8 (rounding-class)
    F)  run_arm F   ""                          0 0 ;;  # control again (session drift check)
    *)  echo "unknown arm '$arm'"; exit 2 ;;
  esac
done

echo "== F5 window complete; receipts under $RECEIPTS =="
echo "== compare per-arm readout.json: verify_ms/cycle, cycles, engagement, verdict =="
