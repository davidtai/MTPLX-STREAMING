#!/bin/bash
# Window 21: served cells on the Qwen3.8-PR prompt construction (chat template + BOS, thinking off): AR and MTP; 1K × 3 seeds, 16K × 1 seed.
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-21
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 21"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((85*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] served AR cells 1K x3 seeds'; DSV41_CONTEXTS=1024 DSV41_RECEIPT_DIR=$OUT/ar-1k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -30
echo '[step 2] served MTP cells 1K x3 seeds'; DSV41_CONTEXTS=1024 DSV41_SERVE_EXTRA_ARGS='--generation-mode mtp' DSV41_RECEIPT_DIR=$OUT/mtp-1k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -30
echo '[step 3] served AR cell 16K x1 seed'; DSV41_CONTEXTS=16384 DSV41_SEEDS=20260829 DSV41_MEMORY_LIMIT_GIB=60 DSV41_RECEIPT_DIR=$OUT/ar-16k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -24
echo '[step 4] served MTP cell 16K x1 seed'; DSV41_CONTEXTS=16384 DSV41_SEEDS=20260829 DSV41_MEMORY_LIMIT_GIB=60 DSV41_SERVE_EXTRA_ARGS='--generation-mode mtp' DSV41_RECEIPT_DIR=$OUT/mtp-16k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -24
echo '[window 21 done]'
"
