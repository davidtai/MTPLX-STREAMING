#!/usr/bin/env bash
# F39 tcq3 decode-lane window (WRITE; a worker RUNS the F2_STAGE_ONLY dry run only — the guarded GPU window is
# opened by the orchestrator per guarded-window-launch-protocol.md, and only once the REAL F38 bank exists).
#
# The tcq3 lane is the '+tcq' arm: point the candidate model dir at the tcq3 artifact, arm MTPLX_DSV41_TCQ3=1, put
# the tcq package on PYTHONPATH, and stage the EXPLICIT plane-lane route (tcq.stage_tcq_runner) into a copy of the
# retained packed sources.  Stock mxfp4 stays the default route by construction (route_plane_lane calls the same
# install_plane_lane when the flag is off) — not a runtime fallback (AGENTS.md "correct by design").
#
# Two modes:
#   F2_STAGE_ONLY=1  (CPU, no GPU, safe for a worker): copy retained packed sources -> staging dir, apply the tcq
#                    route edit, byte-compile the tree, print, exit.  Validates the arm plumbing.
#   (unset)          guarded GPU window: refuses unless a REAL (non-dry-run) tcq3 bank is present, then launches
#                    gpu_window.sh with the retained 16K args + the tcq env.  Waits for the lock; never runs while
#                    /tmp/dsv41-fable-window.active exists or a lock holder is present.
#
# This launcher is self-contained (it does NOT compose the other lanes' stagers).  To run the full benchmark arm
# 'pipe+bal+egl+gt+plx+la+ct50+fi+tcq' add the +tcq modifier to run_f2_prefetch_window.sh per the F39 report diff.
set -euo pipefail

WT="${WT:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f39-tcq3-runtime}"
RUNWT="${RUNWT:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-run-d5f15e7a}"
PYBIN="${PYBIN:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python}"
TCQPKG="$WT/scripts/deepseek_v41"                         # parent of tcq/ and trellis/
RETAINED_SRC="$WT/docs/deepseek-v41/receipts/extension-bank-20260919/full/sources"
# The tcq3 artifact (F38 output); residents load through its --link-rest symlinks.  Fable provides the real path.
TCQ3_ART="${TCQ3_ART:-${GPU_WINDOW_CANDIDATE_MODEL_DIR:-/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-tcq3}}"
GPU_WINDOW="$RUNWT/scripts/deepseek_v41/gpu_window.sh"
LOCK="/tmp/mtplx-gpu-exclusive.lock"
STAMP="$(date +%Y%m%d-%H%M%S)"
STAGE_ROOT="${STAGE_ROOT:-/private/tmp/dsv41-f39-tcq-${STAMP}}"

echo "== F39 tcq3 lane: artifact=$TCQ3_ART stage_root=$STAGE_ROOT =="

# ------------------------------------------------------- stage the tcq route into a copy of the retained sources
stage_tcq_tree() {   # $1 dest dir
  local dest="$1"
  if [ -e "$dest" ]; then echo "REFUSE: staged tree exists: $dest"; exit 2; fi
  mkdir -p "$dest"; cp -R "$RETAINED_SRC/." "$dest/"
  # 1) stage the whole +tcq route: plane-lane decode route + growth codec gate (packed_phase.py) and the
  #    loader install + spec stamp + admission retarget (run_full.py). All edits round-trip + byte-compile.
  PYTHONPATH="$TCQPKG" nice -n 19 "$PYBIN" "$TCQPKG/tcq/stage_tcq_runner.py" \
    --packed-phase "$dest/packed/packed_phase.py" --out-packed-phase "$dest/packed/packed_phase.py" \
    --run-full "$dest/packed/run_full.py" --out-run-full "$dest/packed/run_full.py"
  # 2) drop the tcq + trellis packages into the staged tree so `import tcq.*` / `import tcq_runtime` resolve
  cp -R "$TCQPKG/tcq" "$dest/packed/tcq"
  cp -R "$TCQPKG/trellis" "$dest/packed/trellis"
  # 3) byte-compile the whole staged tree (catches a broken edit before any unload / GPU)
  nice -n 19 "$PYBIN" -m compileall -q "$dest/packed" >/dev/null
  echo "STAGED tcq route -> $dest/packed/{packed_phase.py, run_full.py}"
  grep -q "route_plane_lane" "$dest/packed/packed_phase.py" || { echo "REFUSE: plane-lane route not staged"; exit 2; }
  grep -q "install_tcq_loader" "$dest/packed/run_full.py" || { echo "REFUSE: loader install not staged"; exit 2; }
  grep -q "_tcq_adm.retarget" "$dest/packed/run_full.py" || { echo "REFUSE: admission retarget not staged"; exit 2; }
}

if [ "${F2_STAGE_ONLY:-0}" = "1" ]; then
  stage_tcq_tree "$STAGE_ROOT"
  echo "STAGE ONLY: no GPU window opened (tcq3 route staged + byte-compiled; run the real window once the F38 bank exists)"
  exit 0
fi

# ------------------------------------------------------------------------------- guarded GPU window (bank required)
# The F38 DryEncoder writes the SAME manifest labels as the real encoder, so real-vs-dry cannot be read from the
# artifact.  Fable provides the real bank path and the operator asserts it with TCQ3_REAL_BANK=1 (prevents a worker
# from benchmarking a dry-run/plumbing bank, whose outputs are garbage by design).
MANIFEST="$TCQ3_ART/expert-manifest.json"
[ -f "$MANIFEST" ] || { echo "BLOCKED: tcq3 manifest absent at $MANIFEST (real F38 bank not built yet). Nothing to run."; exit 3; }
nice -n 19 "$PYBIN" - "$MANIFEST" <<'PY' || { echo "BLOCKED: $MANIFEST is not a tcq3 manifest."; exit 3; }
import json, sys
sys.exit(0 if json.load(open(sys.argv[1])).get("quantization", {}).get("mode") == "tcq3" else 1)
PY
[ "${TCQ3_REAL_BANK:-0}" = "1" ] || { echo "BLOCKED: set TCQ3_REAL_BANK=1 to confirm $TCQ3_ART is the REAL F38 bank (a dry-run bank's outputs are garbage by design; the eval gate is separate)."; exit 3; }

# Never open a window while Fable's flag file exists or the lock is held (wait/poll per the spec).
if [ -e /tmp/dsv41-fable-window.active ]; then echo "BLOCKED: /tmp/dsv41-fable-window.active present; wait for Fable."; exit 3; fi
if command -v lsof >/dev/null && lsof "$LOCK" >/dev/null 2>&1; then echo "BLOCKED: GPU lock $LOCK held; wait for the owner."; exit 3; fi

stage_tcq_tree "$STAGE_ROOT"
RECEIPTS="$WT/docs/deepseek-v41/receipts/f39-tcq-build-${STAMP}"; mkdir -p "$RECEIPTS"
OUT="/tmp/dsv41-110-stage/f39-tcq-${STAMP}.jsonl"; mkdir -p "$(dirname "$OUT")"
PYPATH="$RUNWT:$STAGE_ROOT/packed:$STAGE_ROOT/compat:$TCQPKG"
echo "== F39 tcq3 guarded window: model=$TCQ3_ART out=$OUT =="
cd "$RUNWT"
# shellcheck disable=SC2086
env \
  MTPLX_DSV41_TCQ3=1 \
  GPU_WINDOW_LOCK_TIMEOUT="${F39_LOCK_TIMEOUT:-7200}" \
  GPU_WINDOW_TOTAL_MEM_CEILING_BYTES=110000000000 \
  GPU_WINDOW_MIN_AVAIL_GB=100 GPU_WINDOW_RESTORE_QWEN_ALWAYS=1 \
  GPU_WINDOW_CANDIDATE_MODEL_DIR="$TCQ3_ART" \
  PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 \
  PYTHONPATH="$PYPATH" \
  "$GPU_WINDOW" "$PYBIN" "$STAGE_ROOT/launch_full.py" \
    --model "$TCQ3_ART" --arms cell16k_ring_v2_draft_attn_pf0 \
    --context-tokens 16384 --decode-tokens 1023 \
    --decode-mode dspark --dspark-depth 5 --dspark-require-tie-class \
    --max-kv 17664 --prompt-ids-file "docs/deepseek-v41/receipts/memory-budget-110/python-prompt-ids.json" \
    --prompt-seed 20260829 --stop-on-eos --box-target-gb 110 \
    --expert-profile deepseek-v41-mxfp4-75 --slot-layout component-banks \
    --transient-slots 48 --apply-memory-cap --cache-policy transition-window \
    --verify-shared-overlap --decode-miss-records-per-part 3 --kv-cache-bits 16 \
    --out "$OUT" > "$RECEIPTS/guard.log" 2>&1 && rc=0 || rc=$?
echo "$rc" > "$RECEIPTS/guard.exit"
echo "== F39 tcq3 window exit $rc; receipts under $RECEIPTS =="
