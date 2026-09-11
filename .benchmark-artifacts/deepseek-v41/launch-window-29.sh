#!/bin/bash
# Window 29: THE cell only (16,384-token Qwen-PR sweep prompt = 1K task + filler, real prefill, then decode).
# Attribute the 16K-context decode collapse (0.55 tok/s) with stage timing on AR and DSpark, then served AR + DSpark at a 70 GiB cap.
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-29
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 29"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((96*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] AR prefill_lean_sel @16384 (control: window-28b arm), decode 256, stage timing, 80 GiB'; MTPLX_ROUTE_STAGE_PROBE=1 \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 16384 --decode-tokens 256 --arms prefill_lean_sel --stage-timing --memory-limit-gib 80 --out $OUT/ar-16k-control.json 2>&1 | grep -vE \"\$F\" | tail -14
echo '[step 2] AR prefill_lean_sel_chunk @16384 (W73 KV_CHUNK_GROW), decode 256, stage timing, 80 GiB'; MTPLX_ROUTE_STAGE_PROBE=1 \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 16384 --decode-tokens 256 --arms prefill_lean_sel_chunk --stage-timing --memory-limit-gib 80 --out $OUT/ar-16k-chunk.json 2>&1 | grep -vE \"\$F\" | tail -14
echo '[step 2b] DSpark cell16k @16384, decode 256, stage timing, 80 GiB'; MTPLX_ROUTE_STAGE_PROBE=1 \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 16384 --decode-tokens 256 --decode-mode dspark --dspark-depth 3 --arms cell16k --stage-timing --memory-limit-gib 80 --out $OUT/dspark-16k-cell.json 2>&1 | grep -vE \"\$F\" | tail -14
echo '[step 3] served AR 16K x1 seed at 70 GiB cap'; DSV41_CONTEXTS=16384 DSV41_SEEDS=20260829 DSV41_MEMORY_LIMIT_GIB=70 DSV41_RECEIPT_DIR=$OUT/served-ar-16k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -8
echo '[step 4] served DSpark 16K x1 seed at 70 GiB cap'; DSV41_CONTEXTS=16384 DSV41_SEEDS=20260829 DSV41_MEMORY_LIMIT_GIB=70 DSV41_SERVE_EXTRA_ARGS='--load-mtp --generation-mode dspark --depth 3' DSV41_RECEIPT_DIR=$OUT/served-dspark-16k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -8
echo '[window 29 done]'
"
