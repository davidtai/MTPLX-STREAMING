#!/bin/bash
# Window 18: served 1K bench — AR candidate (profile defaults) vs AR control (levers off).
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-18
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 18"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((85*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] served AR candidate (profile defaults)'; DSV41_BENCH_RECEIPT=$OUT/served_ar_candidate.json bash scripts/deepseek_v41/serve_bench_1k.sh 2>&1 | grep -vE \"\$F\" | tail -20
echo '[step 2] served AR control (levers off)'; MTPLX_DSV41_HEAD_MODE=default MTPLX_DSV41_SINKHORN_METAL=0 MTPLX_DSV41_ATTN_COMPILE=0 MTPLX_DSV41_ATTN_WIN_MEMO=0 DSV41_BENCH_RECEIPT=$OUT/served_ar_control.json bash scripts/deepseek_v41/serve_bench_1k.sh 2>&1 | grep -vE \"\$F\" | tail -20
echo '[window 18 done]'
"
