#!/bin/bash
# Window 29: DSpark verify attribution at an 80 GiB plan — stage timing + W61 engagement, K29 on vs off; served DSpark 16K.
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
echo '[step 1] dspark stack_a @1024, 80 GiB, stage timing, K29 on'; MTPLX_ROUTE_STAGE_PROBE=1 \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --decode-mode dspark --dspark-depth 3 --arms stack_a --stage-timing --memory-limit-gib 80 --out $OUT/dspark-1024-k29on.json 2>&1 | grep -vE \"\$F\" | tail -10
echo '[step 2] dspark stack_a @1024, 80 GiB, K29 OFF'; MTPLX_DSV41_DSPARK_VERIFY_K29=0 MTPLX_ROUTE_STAGE_PROBE=1 \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --decode-mode dspark --dspark-depth 3 --arms stack_a --stage-timing --memory-limit-gib 80 --out $OUT/dspark-1024-k29off.json 2>&1 | grep -vE \"\$F\" | tail -10
echo '[step 3] device-sample A/B @1024 (K32)'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --arms stack_a --memory-limit-gib 80 --out $OUT/ar-1024-classic.json 2>&1 | grep -vE \"\$F\" | tail -4; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --arms stack_a --device-sample --memory-limit-gib 80 --out $OUT/ar-1024-devsample.json 2>&1 | grep -vE \"\$F\" | tail -4
echo '[step 4] served DSpark-direct 16K x1 at 80 GiB'; DSV41_CONTEXTS=16384 DSV41_SEEDS=20260829 DSV41_MEMORY_LIMIT_GIB=80 DSV41_SERVE_EXTRA_ARGS='--load-mtp --generation-mode dspark --depth 3' DSV41_RECEIPT_DIR=$OUT/dspark-16k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -8
echo '[window 29 done]'
"
