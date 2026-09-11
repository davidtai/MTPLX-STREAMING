#!/bin/bash
# Window 28: MORE MoE SPACE. 1K arms at a 92 GiB plan (AR stack, AR with head no-reprice, DSpark); 16K prefill_lean_sel at 72 GiB; served DSpark 1K at a 92 GiB cap.
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-28
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 28"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((96*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] AR stack_a @1024, 92 GiB plan'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --arms stack_a --memory-limit-gib 92 --out $OUT/ar-1024-92g.json 2>&1 | grep -vE \"\$F\" | tail -4
echo '[step 2] AR stack_a --with-mtp --no-reprice @1024, 92 GiB plan'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --decode-mode ar --with-mtp --no-reprice --arms stack_a --memory-limit-gib 92 --out $OUT/ar-1024-withmtp-92g.json 2>&1 | grep -vE \"\$F\" | tail -4
echo '[step 3] dspark stack_a @1024, 92 GiB plan'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --decode-mode dspark --dspark-depth 3 --arms stack_a --memory-limit-gib 92 --out $OUT/dspark-1024-92g.json 2>&1 | grep -vE \"\$F\" | tail -8
echo '[step 4] 16K prefill_lean_sel at 80 GiB plan (decode 32)'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 16384 --decode-tokens 32 --max-kv 16640 --arms prefill_lean_sel --memory-limit-gib 80 --out $OUT/prefill-16384-sel-80g.json 2>&1 | grep -vE \"\$F\" | tail -6
echo '[step 5] served DSpark-direct 1K x3 at 92 GiB cap'; DSV41_CONTEXTS=1024 DSV41_MEMORY_LIMIT_GIB=92 DSV41_SERVE_EXTRA_ARGS='--load-mtp --generation-mode dspark --depth 3' DSV41_RECEIPT_DIR=$OUT/dspark-1k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -8
echo '[step 6] served AR 16K x1 at 80 GiB cap (fixed admission)'; DSV41_CONTEXTS=16384 DSV41_SEEDS=20260829 DSV41_MEMORY_LIMIT_GIB=80 DSV41_RECEIPT_DIR=$OUT/ar-16k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -8
echo '[window 28 done]'
"
