#!/bin/bash
# Window 30: the 16K cell — AR cell16k (full prefill+decode stack + KV chunk-grow) stage timing; served AR + DSpark 16K at 80 GiB with the W75 stall fix.
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-30
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 30"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((96*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] AR cell16k @16384, decode 256, stage timing, 80 GiB'; MTPLX_ROUTE_STAGE_PROBE=1 \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 16384 --decode-tokens 256 --max-kv 17408 --arms cell16k --stage-timing --memory-limit-gib 80 --out $OUT/ar-16k-cell16k.json 2>&1 | grep -vE \"\$F\" | tail -14
echo '[step 1b] AR cell16k @16384, decode 256, stage timing, 60 GiB plan (pressure hypothesis A/B vs step 1)'; MTPLX_ROUTE_STAGE_PROBE=1 \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 16384 --decode-tokens 256 --max-kv 17408 --arms cell16k --stage-timing --memory-limit-gib 60 --out $OUT/ar-16k-cell16k-60g.json 2>&1 | grep -vE \"\$F\" | tail -14
echo '[step 2] served AR 16K x1 seed at 80 GiB (W75 fix)'; DSV41_CONTEXTS=16384 DSV41_SEEDS=20260829 DSV41_MEMORY_LIMIT_GIB=80 DSV41_RECEIPT_DIR=$OUT/served-ar-16k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -8
echo '[step 3] served DSpark 16K x1 seed at 80 GiB (W75 fix)'; DSV41_CONTEXTS=16384 DSV41_SEEDS=20260829 DSV41_MEMORY_LIMIT_GIB=80 DSV41_SERVE_EXTRA_ARGS='--load-mtp --generation-mode dspark --depth 3' DSV41_RECEIPT_DIR=$OUT/served-dspark-16k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -8
echo '[window 30 done]'
"
