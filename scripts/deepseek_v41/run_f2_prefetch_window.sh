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
#         | pipe (F2_MAX_ROWS-1, F16 two-group verify pipeline: 4+rest row groups interleaved under a
#           thread baton; no F2b, no stamp probe; MUST reproduce digest $F16_ORACLE_SHA = the
#           sequential 4+4 chunk arm, whose first control divergence @480 is a proven tie flip)
#   +si : GIL switch interval MTPLX_DSV41_GIL_SWITCH_S=$F2_GIL_SWITCH_S (applied once at the
#         F2b hook; the hook is staged but F2b itself stays off for control bases)
#   +eg : F6 parallel Engram miss reads, decode-site install; +egl: load-site (prefill too)
#   +gt : F12 parallel packed-scale load in the growth transition (read + sha256 on a pool,
#         joined before first use); +pn : F2b predictor 'native' (default is the F2c 'lean' tape)
#   +k0 : F2b evaluates the predictor but issues NO speculative reads (isolates predictor cost)
#   +ftN : F2b first target layer N (default 4; +ft1 adds targets 1-3 from sources 0-2)
#   +rio : F2b speculative reads go through the retained reader (default: private fd, bare preadv)
#   +rd : F15 verify-logits row dump at global positions $F2_ROW_INDICES (tie classification of a
#         candidate-vs-control divergence; rows land in <arm dir>/rows)
#   +cmp : rounding-class compile levers hc_compile,attn_compile via the F5 hook (needs F2_PROBE=1)
#   +pc : F2d predictor on the CPU coordinator thread (barrier = exactly mx.eval(indices)); its
#         283,170,816 B of host f32 gate weights are charged to admission with the ring
#   +ml : wire (mlock) the F2b ring buffers once at install
#   +plr : F17 per-layer extension rows from the causal prefill rule (same total rows; exact)
#   +plx : F17 per-layer rows from the causal EXCESS rule (rows ∝ stat - p10(stat); prefill-only)
#   +plo : F17 per-layer rows from the F14 ORACLE profile (shape mode; benchmark-derived ceiling)
#   +cmp2 : +cmp with caps8; +cmp3 : +cmp with attn_core_compile; +cmp4 : +cmp with hc_premix_kernel
#   +vcb : sequential BALANCED verify schedule (ceil(n/2)+floor(n/2) rows per cycle, single chunk for
#         n<=4) = the oracle for the balanced pipeline split
#   +bal : (pipe base) F16 balanced leader/trailer split; oracle digest = $F16_BAL_ORACLE_SHA if set
#   +bh  : (pipe base) F18 barrier hand-off: async routing barrier + second greenlet hand-off, per-group
#          deferred slot releases; same oracle digest as the split it rides. Needs F16PKG = the f18 package.
#   +wm  : EXACT decode lever attn_win_memo via the F5 hook (needs F2_PROBE=1): the sliding-window attend mask is built
#          once per forward instead of once per layer (same array object; keyed on the positions object, per group).
#   +opsN / +mbN : MLX_MAX_OPS_PER_BUFFER=N / MLX_MAX_MB_PER_BUFFER=N for the child (Metal command-buffer commit
#          thresholds; M5 Max defaults 50 ops / 50 MB). Scheduling only: same kernels, same arithmetic.
#   +st  : (pipe base) per-slice stamps -> <arm dir>/f16_stamps.{raw.json.gz,summary.json}. Needs the f18 package.
#   +vcA-B : stage the hybrid install's verify schedule as two chunks A+B (=8): row-split
#         EXACTNESS probe (digest decides whether a two-group verify pipeline is exact);
#         slower by construction, never a throughput candidate
F2_GIL_SWITCH_S="${F2_GIL_SWITCH_S:-0.00005}"
F2_ENGRAM_WORKERS="${F2_ENGRAM_WORKERS:-16}"
F6DIR="${F6DIR:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f6-engram/scripts/deepseek_v41/f6}"
F12DIR="${F12DIR:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f12-growth/scripts/deepseek_v41/f12}"
F2_GROWTH_WORKERS="${F2_GROWTH_WORKERS:-8}"
F15DIR="${F15DIR:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f15-tieclass/scripts/deepseek_v41/f15}"
F2_ROW_INDICES="${F2_ROW_INDICES:-470-490}"
F16PKG="${F16PKG:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f16-pipeline/scripts/deepseek_v41}"
F16_ORACLE_SHA="${F16_ORACLE_SHA:-172830a96d84dbdac631c058fe0dfaf956df15b6c353f41fc05f604bc92c6393}"
F16SITE="${F16SITE:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f16-pipeline/.f16-site}"   # private greenlet install
F17DIR="${F17DIR:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-f17-rows/scripts/deepseek_v41/f17}"
F17_ORACLE_SHAPE="${F17_ORACLE_SHAPE:-81,54,59,14,32,25,0,9,18,10,3,15,29,28,0,30,0,9,0,54,0,0,0,15,0,0,0,27,0,6,13,28,27,34,10,7,21,32,55,55}"
STAMP="$(date +%Y%m%d-%H%M%S)"
STAGE_ROOT="/private/tmp/dsv41-f2b-${STAMP}"
OUT_STAGE="/tmp/dsv41-110-stage"                     # run_full.py:330 requires --out here
RECEIPTS="$WT/docs/deepseek-v41/receipts/f2-prefetch-build-20260919/window-${STAMP}"

cd "$WT"
mkdir -p "$RECEIPTS" "$OUT_STAGE"
# Advisory flag for Fable's CPU/SSD workers: while it exists they must not run SSD or
# memory-heavy benches (an overlapping 3 GB microbench pushed the box over the 110e9
# ceiling on 2026-09-19 and the guard killed an arm; `lsof` on the GPU lock races with the
# ~17 s gaps between arms). The launcher only SETS it; Fable removes it when pausing.
# F2_NO_FLAG=1: timing-insensitive arms (e.g. logits row dumps) leave the workers running.
if [ "${F2_STAGE_ONLY:-0}" != "1" ] && [ "${F2_NO_FLAG:-0}" != "1" ]; then touch /tmp/dsv41-fable-window.active; fi

# --------------------------------------------------- 1. CPU preflight (before unload)
echo "== F2b preflight (CPU; MLX pinned; before any service unload) =="
PYTHONPATH="$RETAINED_SRC/packed:$RUNWT:$F2PKG" nice -n 19 "$PYBIN" -m f2.window_preflight \
  --run-worktree "$RUNWT" \
  --compat-installation "$COMPAT_INSTALLATION" \
  --packed-installation "$PACKED_INSTALLATION" \
  --archived-dir "$RETAINED_SRC" \
  --dep "$STRICT_LIB" --dep "$RUNWT/$PROMPT_IDS" --dep "$MODEL_DIR"

# --------------------------------------------------- 2. stage patched runner copies
stage_tree() {  # $1 dest  $2 max_rows  $3 stage the F2b/GIL hook (0/1)  $4 stage F6 engram (0/1)  $5 charge the ring (0/1)
  local dest="$1" rows="$2" f2b="$3" eng="${4:-0}" ring="${5:-0}" vc="${6:-0}" gt="${7:-0}" rd="${8:-0}" pl="${9:-0}" pipe="${10:-0}"
  local ring_arg=""
  [ "$ring" != "0" ] && ring_arg="--ring-bytes $ring"   # total host bytes charged to admission
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
    --admission "$dest/packed/packed_admission.py" --max-rows "$rows" $ring_arg
  if [ "$pl" != "0" ]; then   # F17: per-layer extension rows (staged extension.py)
    nice -n 19 "$PYBIN" "$F17DIR/stage_f17_runner.py" \
      --extension "$RETAINED_SRC/packed/extension.py" --out "$dest/packed/extension.py"
  fi
  if [ "$gt" = "1" ]; then
    nice -n 19 "$PYBIN" "$F12DIR/stage_f12_runner.py" --packed-phase "$dest/packed/packed_phase.py"
  fi
  if [ "$vc" != "0" ]; then
    nice -n 19 "$PYBIN" "$F2PKG/f2/stage_f2_runner.py" \
      --hybrid-install "$dest/packed/hybrid_install.py" --verify-chunks "$([ "$vc" = "b" ] && echo balanced || printf '%s' "$vc" | tr '-' ',')"
  fi
  if [ "$pipe" = "1" ]; then   # F16: run_full hook + hybrid verify call + 4-buffer projection store
    # preflight re-applies the stager to the RETAINED sources, so it needs the unstaged tree
    PYTHONPATH="$F16PKG:$F16SITE:$RETAINED_SRC/packed:$RETAINED_SRC/compat:$RUNWT" nice -n 19 "$PYBIN" -m f16.preflight
    PYTHONPATH="$F16PKG" nice -n 19 "$PYBIN" -m f16.stage_f16_runner \
      --run-full "$dest/packed/run_full.py" --hybrid-install "$dest/packed/hybrid_install.py" \
      --projection-install "$dest/packed/projection_install.py"
  fi
  if [ "$F2_PROBE" = "1" ]; then
    nice -n 19 "$PYBIN" "$F5DIR/stage_f5_runner.py" \
      --retained "$dest/packed/run_full.py" --out "$dest/packed/run_full.py"
  fi
  if [ "$rd" = "1" ]; then    # F15 row dump: shares the growth_transition anchor with F5/F6
    nice -n 19 "$PYBIN" "$F15DIR/stage_f15_runner.py" \
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
  A_CAPS8=0; A_PIPE=0; A_BAL=0; A_BH=0; A_ST=0; A_OPS=0; A_MB=0; A_WM=0
  A_BASE="${tok%%+*}"; A_SI=0; A_ENG=0; A_VC=0; A_GT=0; A_PN=0; A_K0=0; A_FT=0; A_RIO=0; A_RD=0; A_CMP=0; A_PC=0; A_ML=0; A_PL=0; A_CMPSET=""
  mods="+${tok#*+}+"; [ "$tok" = "$A_BASE" ] && mods="+"
  case "$mods" in *"+si+"*) A_SI=1 ;; esac
  case "$mods" in *"+eg+"*) A_ENG=decode ;; esac
  case "$mods" in *"+egl+"*) A_ENG=load ;; esac
  case "$mods" in *"+gt+"*) A_GT=1 ;; esac
  case "$mods" in *"+pn+"*) A_PN=1 ;; esac
  case "$mods" in *"+k0+"*) A_K0=1 ;; esac
  case "$mods" in *"+rd+"*) A_RD=1 ;; esac
  case "$mods" in *"+cmp+"*) A_CMP=1; A_CMPSET="hc_compile,attn_compile" ;; esac
  case "$mods" in *"+cmp2+"*) A_CMP=1; A_CMPSET="hc_compile,attn_compile"; A_CAPS8=1 ;; esac
  case "$mods" in *"+cmp3+"*) A_CMP=1; A_CMPSET="hc_compile,attn_compile,attn_core_compile" ;; esac
  case "$mods" in *"+cmp4+"*) A_CMP=1; A_CMPSET="hc_compile,attn_compile,hc_premix_kernel" ;; esac
  case "$mods" in *"+plr+"*) A_PL=rule ;; esac
  case "$mods" in *"+plo+"*) A_PL=oracle ;; esac
  case "$mods" in *"+plx+"*) A_PL=excess ;; esac
  case "$mods" in *"+bal+"*) A_BAL=1 ;; esac
  case "$mods" in *"+bh+"*) A_BH=1 ;; esac
  case "$mods" in *"+wm+"*) A_WM=1 ;; esac
  case "$mods" in *"+st+"*) A_ST=1 ;; esac
  case "$mods" in *"+pc+"*) A_PC=1 ;; esac
  case "$mods" in *"+ml+"*) A_ML=1 ;; esac
  case "$mods" in *"+rio+"*) A_RIO=1 ;; esac
  case "$mods" in *"+ft"*) A_FT="${mods#*+ft}"; A_FT="${A_FT%%+*}" ;; esac
  case "$mods" in *"+ops"*) A_OPS="${mods#*+ops}"; A_OPS="${A_OPS%%+*}" ;; esac
  case "$mods" in *"+mb"*) A_MB="${mods#*+mb}"; A_MB="${A_MB%%+*}" ;; esac
  case "$mods" in *"+vc"*) A_VC="${mods#*+vc}"; A_VC="${A_VC%%+*}" ;; esac
  case "$A_BASE" in
    control|control_a|control_b) A_ROWS="$F2_MAX_ROWS"; A_F2B=0 ;;
    control_low)                 A_ROWS="$((F2_MAX_ROWS-1))"; A_F2B=0 ;;
    candidate)                   A_ROWS="$((F2_MAX_ROWS-1))"; A_F2B=1 ;;
    pipe)                        A_ROWS="$((F2_MAX_ROWS-1))"; A_F2B=0; A_PIPE=1 ;;
    *) echo "unknown arm base '$A_BASE' in '$tok'"; exit 2 ;;
  esac
  A_HOOK=0; { [ "$A_F2B" = "1" ] || [ "$A_SI" = "1" ]; } && A_HOOK=1
  A_ENGSTAGE=0; [ "$A_ENG" != "0" ] && A_ENGSTAGE=1
  A_HOSTBYTES=0
  [ "$A_F2B" = "1" ] && A_HOSTBYTES=$((F2_RING_RECORDS * 3 * 5898240))
  [ "$A_F2B" = "1" ] && [ "$A_PC" = "1" ] && A_HOSTBYTES=$((A_HOSTBYTES + 283170816))
  [ "$A_PIPE" = "1" ] && A_HOSTBYTES=$((A_HOSTBYTES + 134217728))   # F16: two extra bf16 projection buffers
  [ "$A_PL" != "0" ] && A_HOSTBYTES=$((A_HOSTBYTES + 100663296))   # F17 append-peak under-count (<= 67 MB), charged as 96 MiB
  A_TREE="$STAGE_ROOT/r${A_ROWS}-h${A_HOOK}-e${A_ENGSTAGE}-g${A_HOSTBYTES}-v${A_VC}-t${A_GT}-d${A_RD}-p${A_PL}-q${A_PIPE}"   # g = host ring charged to admission
  A_DIRNAME="$(printf '%s' "$tok" | tr '+' '_')"
}
ARMS="${F2_ARMS:-control_a candidate control_b}"
echo "== stage retained sources -> $STAGE_ROOT (arms: $ARMS; probe=$F2_PROBE) =="
for arm in $ARMS; do   # stage EVERY needed tree up-front: fail before the first unload
  parse_arm "$arm"
  [ -d "$A_TREE" ] || stage_tree "$A_TREE" "$A_ROWS" "$A_HOOK" "$A_ENGSTAGE" "$A_HOSTBYTES" "$A_VC" "$A_GT" "$A_RD" "$A_PL" "$A_PIPE"
done
if [ "${F2_STAGE_ONLY:-0}" = "1" ]; then   # CPU dry run of the whole staging sequence
  for arm in $ARMS; do parse_arm "$arm"; echo "STAGED $arm -> $A_TREE (rows=$A_ROWS f2b=$A_F2B si=$A_SI engram=$A_ENG verify_chunks=$A_VC growth=$A_GT native_predictor=$A_PN k0=$A_K0 first_target=$A_FT reader_io=$A_RIO row_dump=$A_RD compile=$A_CMP cpu_predictor=$A_PC wired_ring=$A_ML per_layer_rows=$A_PL compile_set=$A_CMPSET caps8=$A_CAPS8 host_bytes=$A_HOSTBYTES)"; done
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
    local f5_enable=""; [ "$A_CMP" = "1" ] && f5_enable="$A_CMPSET"
    [ "$A_WM" = "1" ] && f5_enable="${f5_enable:+$f5_enable,}attn_win_memo"
    local timed=1; [ "$A_PIPE" = "1" ] && timed=0   # the stamp probe rebinds run; F16 refuses any non-scheduled lane
    probe_env="MTPLX_DSV41_F5_ENABLE=$f5_enable MTPLX_DSV41_F5_CAPS8=$A_CAPS8 MTPLX_DSV41_F5_TIMED_PROBE=$timed MTPLX_DSV41_F5_TIMED_OUT=$dir/timed_probe"
  fi
  local f2b_env=""
  [ "$f2b" = "1" ] && f2b_env="MTPLX_DSV41_F2B=1 MTPLX_DSV41_F2B_RECORDS=$F2_RING_RECORDS MTPLX_DSV41_F2B_WORKERS=$F2_WORKERS MTPLX_DSV41_F2B_COUNTERS=$dir/f2b_counters.json"
  [ "$A_SI" = "1" ] && f2b_env="$f2b_env MTPLX_DSV41_GIL_SWITCH_S=$F2_GIL_SWITCH_S"
  [ "$A_PN" = "1" ] && f2b_env="$f2b_env MTPLX_DSV41_F2B_PREDICTOR=native"
  [ "$A_PC" = "1" ] && f2b_env="$f2b_env MTPLX_DSV41_F2B_PREDICTOR=cpu"
  [ "$A_ML" = "1" ] && f2b_env="$f2b_env MTPLX_DSV41_F2B_WIRE_RING=1"
  if [ "$A_PIPE" = "1" ]; then
    pypath="$pypath:$F16PKG:$F16SITE"
    f2b_env="$f2b_env MTPLX_DSV41_F16=1 MTPLX_DSV41_F16_COUNTERS=$dir/f16_counters.json"
    [ "$A_BAL" = "1" ] && f2b_env="$f2b_env MTPLX_DSV41_F16_SPLIT=balanced"
    if [ "$A_BH" = "1" ] || [ "$A_ST" = "1" ]; then
      [ -f "$F16PKG/f16/stamps.py" ] || { echo "REFUSE: +bh/+st need the F18 package (set F16PKG to .worktrees/dsv41-f18-handoff/scripts/deepseek_v41)"; exit 2; }
    fi
    [ "$A_BH" = "1" ] && f2b_env="$f2b_env MTPLX_DSV41_F16_HANDOFF=barrier"
    [ "$A_ST" = "1" ] && f2b_env="$f2b_env MTPLX_DSV41_F16_STAMPS=$dir/f16_stamps"
  fi
  if [ "$A_PL" != "0" ]; then
    pypath="$pypath:$F17DIR"
    if [ "$A_PL" = "rule" ]; then f2b_env="$f2b_env MTPLX_DSV41_F17_ALLOC=prefill_rule"
    elif [ "$A_PL" = "excess" ]; then f2b_env="$f2b_env MTPLX_DSV41_F17_ALLOC=prefill_excess"
    else f2b_env="$f2b_env MTPLX_DSV41_F17_ALLOC=shape:$F17_ORACLE_SHAPE"; fi
  fi
  [ "$A_K0" = "1" ] && f2b_env="$f2b_env MTPLX_DSV41_F2B_K=0"
  case "$A_OPS$A_MB" in *[!0-9]*) echo "REFUSE: +ops/+mb need a positive integer (got ops=$A_OPS mb=$A_MB)"; exit 2 ;; esac
  [ "$A_OPS" != "0" ] && f2b_env="$f2b_env MLX_MAX_OPS_PER_BUFFER=$A_OPS"
  [ "$A_MB" != "0" ] && f2b_env="$f2b_env MLX_MAX_MB_PER_BUFFER=$A_MB"
  if [ "$A_CMP" = "1" ] && [ "$F2_PROBE" != "1" ]; then echo "REFUSE: +cmp needs F2_PROBE=1 (F5 hook)"; exit 2; fi
  if [ "$A_WM" = "1" ] && [ "$F2_PROBE" != "1" ]; then echo "REFUSE: +wm needs F2_PROBE=1 (F5 hook)"; exit 2; fi
  if [ "$A_RD" = "1" ]; then
    pypath="$pypath:$F15DIR"; mkdir -p "$dir/rows"
    f2b_env="$f2b_env MTPLX_DSV41_F15_ROW_DUMP_DIR=$dir/rows MTPLX_DSV41_F15_ROW_INDICES=$F2_ROW_INDICES"
  fi
  [ "$A_RIO" = "1" ] && f2b_env="$f2b_env MTPLX_DSV41_F2B_DIRECT_IO=0"
  [ "$A_FT" != "0" ] && f2b_env="$f2b_env MTPLX_DSV41_F2B_FIRST_TARGET=$A_FT"
  if [ "$A_GT" = "1" ]; then
    pypath="$pypath:$F12DIR"
    f2b_env="$f2b_env MTPLX_DSV41_F12_PARALLEL_SCALES=1 MTPLX_DSV41_F12_WORKERS=$F2_GROWTH_WORKERS"
  fi
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
  if [ "$A_PIPE" = "1" ] && [ "$A_CMP" != "1" ]; then   # the pipeline has an exact oracle: the sequential 4+rest digest
    local got; got="$(grep -a -o '"output_ids_sha256": "[0-9a-f]*"' "$dir"/f2b-*.passes.jsonl 2>/dev/null | head -1 | cut -d'"' -f4)"
    local want="$F16_ORACLE_SHA"; [ "$A_BAL" = "1" ] && want="${F16_BAL_ORACLE_SHA:-}"
    if [ -z "$want" ]; then
      echo "  pipe arm ${arm}: no oracle digest configured for this split; got '${got:-none}' (compare by hand)"
    elif [ "$got" != "$want" ]; then
      echo "FAIL: pipe arm ${arm} digest '${got:-none}' != oracle $want (guard exit $rc). Stopping."
      grep -a -E "ABORTED|Traceback|Error" "$dir/guard.log" | tail -5 || true; exit 4
    fi
    [ -n "$want" ] && echo "  pipe arm ${arm}: digest == its sequential row-split oracle (bit-identical arithmetic)"
  fi
  if [ "$rc" = "4" ] && { [ "$A_VC" != "0" ] || [ "$A_CMP" = "1" ] || [ "$A_PIPE" = "1" ]; }; then
    echo "  (guard exit 4 on a rounding-class probe arm: digest differs from control, as expected; continuing)"
    return 0
  fi
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
