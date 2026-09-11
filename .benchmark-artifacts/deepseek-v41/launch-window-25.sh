#!/bin/bash
# Window 25: DSpark verify-phase A/B @1K; AR with/without MTP head; served DSpark-direct 1K (fixed); served AR 16K (widened KV window).
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-25
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 25"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((85*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] dspark @1024 verify DECODE phase (stack_a)'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --decode-mode dspark --dspark-depth 3 --arms stack_a --stage-timing --memory-limit-gib 82 --out $OUT/dspark-1024-decodephase.json 2>&1 | grep -vE \"\$F\" | tail -16
echo '[step 2] dspark @1024 verify PREFILL phase (stack_a, control for step 1)'; MTPLX_DSV41_DSPARK_VERIFY_DECODE_PHASE=0 \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --decode-mode dspark --dspark-depth 3 --arms stack_a --memory-limit-gib 82 --out $OUT/dspark-1024-prefillphase.json 2>&1 | grep -vE \"\$F\" | tail -8
echo '[step 3] AR stack_a with MTP head loaded vs plain'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --decode-mode ar --with-mtp --arms stack_a --memory-limit-gib 82 --out $OUT/ar-1024-withmtp.json 2>&1 | grep -vE \"\$F\" | tail -6; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --decode-mode ar --arms stack_a --memory-limit-gib 82 --out $OUT/ar-1024-plain.json 2>&1 | grep -vE \"\$F\" | tail -6
echo '[step 4] served DSpark-direct 1K x3 (verify decode phase)'; DSV41_CONTEXTS=1024 DSV41_SERVE_EXTRA_ARGS='--load-mtp --generation-mode dspark --depth 3' DSV41_RECEIPT_DIR=$OUT/dspark-1k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -10
echo '[step 5] served AR 16K x1 (60 GiB, KV 17664)'; DSV41_CONTEXTS=16384 DSV41_SEEDS=20260829 DSV41_MEMORY_LIMIT_GIB=60 DSV41_RECEIPT_DIR=$OUT/ar-16k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -10
echo '[window 25 done]'
"
