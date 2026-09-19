#!/usr/bin/env bash
# Guarded GPU window for the F2 next-layer expert prefetch lane on the exact
# extension-bank 16,384-in / 1,024-out Q4 D5/M<=8 verify workload.
#
# WRITE-ONLY: the orchestrator launches this under the exclusive GPU lock; the author
# does NOT run it (another session holds /tmp/mtplx-gpu-exclusive.lock and is
# measuring). Follow guarded-window-launch-protocol.md: this is the DIRECT command of a
# run_in_background:true Bash call, never a shell `&` one-liner, and the launching
# agent is not stopped mid-window.
#
# Arms (each a FRESH receipt dir, never overwriting a prior receipt; full 1,024 output
# token ids stored; sha256 gated against the retained native-MTP digest):
#   control_a   111 persistent rows, NATIVE packed lane (prefetch OFF) -- the retained best.
#   candidate   110 persistent rows + R=32 speculative ring, F2 prefetch lane ON.
#   control_b   111 rows, native lane again -- bounds background drift across the window.
#   control_110 (optional, F2_INCLUDE_CONTROL110=1) 110 rows, native lane, prefetch OFF --
#               isolates the ring's effect from the one-row capacity reduction.
# The candidate's rows come from the SAME admission code the retained run uses
# (sources/packed/packed_admission.py): charging the ring reserve
# (f2.full_config.ring_reserve_bytes(32) = 566,231,040 B) into the capacity search's
# fit checks (packed_admission.py:126-131) drops the admitted rows 111 -> 110, because
# 111 rows + ring = 110,311,575,660 B > 110e9 while 110 rows + ring = 109,603,786,860 B
# fits. R=16 (F2_RING_RECORDS=16) is also supported (still needs 110 rows).
#
# The candidate output MUST be byte-identical to the control's (the gather uses the TRUE
# indices; the ring only warms the cache): every arm's token_ids_sha256 is gated to
#   0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac
# and the candidate receipt gains AGGREGATE prefetch counters read ONCE after decode
# from runtime.counters (issued / committed / awaited-in-flight / wasted / extra bytes).
set -euo pipefail

# --- paths (override via env) ---------------------------------------------
WT="${WT:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f2-prefetch}"
DSV41_WT="${DSV41_WT:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41}"
PY="${PY:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python}"
STAGE_CONTROL="${STAGE_CONTROL:-/private/tmp/dsv41-extension-bank-20260919/full-v1}"   # retained staged runner
STAGE_CANDIDATE="${STAGE_CANDIDATE:-/private/tmp/dsv41-extension-bank-20260919/full-v1-f2}"  # + F2 stage edits
STRICT_LIB="${STRICT_LIB:-/private/tmp/dsv41-strict-cache-20260918/strict-lib}"
MODEL="${MODEL:-/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4}"
AUX="${AUX:-/tmp/dsv41-compact-residents}"
PROMPT_IDS="${PROMPT_IDS:-docs/deepseek-v41/receipts/memory-budget-110/python-prompt-ids.json}"
OUTROOT="${OUTROOT:-/tmp/dsv41-f2-prefetch-window-20260919}"
DIGEST="${DIGEST:-0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac}"
RING_RECORDS="${F2_RING_RECORDS:-32}"
GUARD="$WT/scripts/deepseek_v41/gpu_window.sh"

# --- (0) CPU preflight BEFORE the production service is unloaded -----------
# Resolve every runtime/reader/slot/ring/gate seam the lane touches on the real classes,
# import every f2 module (MLX pinned to CPU), and verify the staged-tree + prompt-id
# dependencies exist. Refuse here -- before model load -- if anything is missing
# (AGENTS.md: fail once, clearly, before measured generation).
echo "[f2-window] CPU preflight (no GPU, MLX pinned to CPU): resolving lane seams + deps"
PREFLIGHT_DEPS=(
  --dep "$STAGE_CONTROL/packed/run_full.py"
  --dep "$STAGE_CANDIDATE/packed/run_full.py"
  --dep "$STAGE_CANDIDATE/packed/packed_admission.py"
  --dep "$STRICT_LIB"
  --dep "$DSV41_WT/$PROMPT_IDS"
  --dep "$WT/scripts/deepseek_v41/f2/plane_lane_prefetch.py"
  --dep "$WT/scripts/deepseek_v41/f2/issue.py"
  --dep "$WT/scripts/deepseek_v41/f2/full_config.py"
  --dep "$WT/scripts/deepseek_v41/f2/run_full_install.py"
)
# Add --sha PATH=HEX pins from the staged installation.json when present (the guard
# re-verifies the strict allocator identity too; this is the CPU-side pre-check).
PYTHONPATH="$WT:$WT/scripts/deepseek_v41" MTPLX_DSV41_SINGLE_SLOT_POOL=1 \
  nice -n 19 "$PY" -m f2.window_preflight "${PREFLIGHT_DEPS[@]}"
echo "[f2-window] preflight passed."

mkdir -p "$OUTROOT"

# --- shared env for every arm (mirrors extension-bank full/command.sh) ------
COMMON_ENV=(
  GPU_WINDOW_LOCK_TIMEOUT=600
  GPU_WINDOW_TOTAL_MEM_CEILING_BYTES=110000000000
  GPU_WINDOW_MIN_AVAIL_GB=100
  GPU_WINDOW_RESTORE_QWEN_ALWAYS=1
  GPU_WINDOW_CANDIDATE_MODEL_DIR="$MODEL"
  GPU_WINDOW_CANDIDATE_AUX_DIR="$AUX"
  MTPLX_DSV41_IO_READ_FANOUT=4
  MTPLX_DSV41_SINGLE_SLOT_POOL=1
  PYTHONHASHSEED=0
  PYTHONUNBUFFERED=1
)
# extension-bank base runner argv (D5 + native KV16 + shared overlap, 16,384/1,023).
BASE_ARGS=(
  --model "$MODEL"
  --arms cell16k_ring_v2_draft_attn_pf0
  --context-tokens 16384 --decode-tokens 1023
  --decode-mode dspark --dspark-depth 5 --dspark-require-tie-class
  --max-kv 17664
  --prompt-ids-file "$PROMPT_IDS" --prompt-seed 20260829 --stop-on-eos
  --box-target-gb 110 --slot-layout component-banks --transient-slots 48
  --apply-memory-cap --cache-policy transition-window
  --verify-shared-overlap --decode-miss-records-per-part 3 --kv-cache-bits 16
)

# One arm = one guarded window (one model load), writing a fresh receipt.
# arm_name  stage_tree                cache_policy       extra run_full flags
run_arm() {
  local name="$1" tree="$2" extra_env="$3"; shift 3
  local out="$OUTROOT/$name"
  if [ -e "$out" ]; then
    echo "[f2-window] REFUSING to overwrite existing receipt dir: $out" >&2
    exit 3
  fi
  mkdir -p "$out"
  echo "[f2-window] arm '$name' -> $out (tree: $tree)"
  # shellcheck disable=SC2086
  PYTHONPATH="$DSV41_WT:$tree/packed:$tree/compat:$WT/scripts/deepseek_v41" \
    env "${COMMON_ENV[@]}" $extra_env \
    "$GUARD" "$PY" "$tree/launch_full.py" "${BASE_ARGS[@]}" "$@" \
      --out "$out/receipt.jsonl" > "$out/guard.log" 2>&1
}

# control_a / candidate / control_b (A/B/A); optional control_110 to isolate the row cut.
# The candidate tree carries the F2 stage edits (documented in the receipt README):
#   packed_phase.py  : the single install_plane_lane(...) call -> f2.run_full_install.install_f2_growth(...)
#   packed_admission.py: charge f2.full_config.ring_reserve_bytes(R) into the :126-131 fit checks (111->110 rows)
#   run_full.py      : FullPrefetchConfig + record_pass gains the once-read prefetch counters
# Control arms run the retained tree unchanged (prefetch OFF, 111 rows).
run_arm control_a "$STAGE_CONTROL" ""
run_arm candidate "$STAGE_CANDIDATE" "MTPLX_DSV41_F2_PREFETCH=1 MTPLX_DSV41_F2_RING_RECORDS=$RING_RECORDS"
run_arm control_b "$STAGE_CONTROL" ""
if [ "${F2_INCLUDE_CONTROL110:-0}" = "1" ]; then
  run_arm control_110 "$STAGE_CONTROL" "MTPLX_DSV41_FORCE_DECODE_SLOTS=110"
fi

# --- post-run gate: token-id sha256 + prefetch counters (read once) --------
echo "[f2-window] verifying token-id digests and collecting prefetch counters"
PYTHONPATH="$WT:$WT/scripts/deepseek_v41" nice -n 19 "$PY" - "$OUTROOT" "$DIGEST" <<'PYEOF'
import json, sys
from pathlib import Path
outroot, digest = Path(sys.argv[1]), sys.argv[2]
ok = True
for arm in sorted(p.name for p in outroot.iterdir() if p.is_dir()):
    receipt = outroot / arm / "receipt.jsonl"
    if not receipt.exists():
        print(f"[f2-window]   {arm}: NO RECEIPT ({receipt})"); ok = False; continue
    rows = [json.loads(line) for line in receipt.read_text().splitlines() if line.strip()]
    passes = [r for r in rows if r.get("output_ids_sha256") or r.get("token_ids_sha256")]
    for r in passes:
        got = r.get("token_ids_sha256") or r.get("output_ids_sha256")
        match = (got == digest)
        ok = ok and match
        ctr = r.get("prefetch_counters", {})  # candidate stage edit writes these once after decode
        print(f"[f2-window]   {arm}: sha={'OK' if match else 'MISMATCH '+str(got)}"
              + (f" counters={ctr}" if ctr else ""))
print("[f2-window] DIGEST GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 4)
PYEOF
echo "[f2-window] done. Per-arm receipts + guard logs under $OUTROOT."
