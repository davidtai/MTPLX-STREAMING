#!/bin/bash
# Window 23: DSpark-direct decode @1K; K27 layout_fix A/B @16K; shape/tiling microbench; served AR 1K with counters; served DSpark 1K; served AR 16K.
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-23
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 23"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((85*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] dspark-direct decode @1024 (control, stack_a)'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --decode-mode dspark --dspark-depth 3 --arms control stack_a --memory-limit-gib 82 --out $OUT/dspark-1024.json 2>&1 | grep -vE \"\$F\" | tail -24
echo '[step 2] K27 layout_fix A/B @16384 (60 GiB plan)'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 16384 --decode-tokens 8 --max-kv 16640 --arms layer_major layout_fix --prefill-stage-timing --memory-limit-gib 60 --out $OUT/prefill-16384-layout.json 2>&1 | grep -vE \"\$F\" | tail -16
echo '[step 3] shape/tiling microbench'; \$PY scripts/deepseek_v41/shape_tiling_microbench.py --seq-t 16384 --score-chunk 1024 --warmup 3 --iters 15 --memory-limit-gib 8 --out $OUT/shape_tiling_microbench.json 2>&1 | grep -vE \"\$F\" | tail -40
echo '[step 4] served AR 1K x3 with stage timing + counters'; MTPLX_SERVE_STAGE_TIMING=1 MTPLX_SERVE_STAGE_TIMING_RECEIPT=$OUT/served-ar-1k-stage MTPLX_ROUTE_STAGE_PROBE=1 DSV41_CONTEXTS=1024 DSV41_RECEIPT_DIR=$OUT/ar-1k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -14
echo '[step 5] served DSpark-direct 1K x3'; MTPLX_SERVE_STAGE_TIMING=1 MTPLX_SERVE_STAGE_TIMING_RECEIPT=$OUT/served-dspark-1k-stage DSV41_CONTEXTS=1024 DSV41_SERVE_EXTRA_ARGS='--load-mtp --generation-mode dspark --depth 3' DSV41_RECEIPT_DIR=$OUT/dspark-1k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -14
echo '[step 6] served AR 16K x1 (60 GiB)'; DSV41_CONTEXTS=16384 DSV41_SEEDS=20260829 DSV41_MEMORY_LIMIT_GIB=60 DSV41_RECEIPT_DIR=$OUT/ar-16k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -14
echo '[window 23 done]'
"
