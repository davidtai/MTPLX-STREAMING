#!/bin/bash
# Window 32: (1) W78 bisect heap-pressure sweep (ballast 60 / 60+churn / 88+churn); (2) served AR 16K at the 60 GiB profile default (W79 levers); (3) served DSpark 16K at 60.
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-32
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 32"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((96*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
F='transformers\]|█|╗|╝|║|╭|╰|│'
B=scripts/deepseek_v41/metal_decode_attn_bisect.py
echo '[step 1] bisect ballast 60 GiB'; nice -n 19 \$PY \$B --gpu --T 1024 4096 16384 --iters 30 --warmup 5 --ballast-gib 60 --out $OUT/w78-ballast60.json 2>&1 | grep -vE \"\$F\" | tail -60
echo '[step 1b] bisect ballast 60 GiB + churn'; nice -n 19 \$PY \$B --gpu --T 1024 4096 16384 --iters 30 --warmup 5 --ballast-gib 60 --ballast-churn --out $OUT/w78-ballast60-churn.json 2>&1 | grep -vE \"\$F\" | tail -60
echo '[step 1c] bisect ballast 88 GiB + churn'; nice -n 19 \$PY \$B --gpu --T 1024 4096 16384 --iters 30 --warmup 5 --ballast-gib 88 --ballast-churn --out $OUT/w78-ballast88-churn.json 2>&1 | grep -vE \"\$F\" | tail -60
echo '[step 2] served AR 16K x1 seed, profile default plan (60 GiB, cell16k levers)'; DSV41_CONTEXTS=16384 DSV41_SEEDS=20260829 DSV41_RECEIPT_DIR=$OUT/served-ar-16k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -8
echo '[step 3] served DSpark 16K x1 seed, profile default plan'; DSV41_CONTEXTS=16384 DSV41_SEEDS=20260829 DSV41_SERVE_EXTRA_ARGS='--load-mtp --generation-mode dspark --depth 3' DSV41_RECEIPT_DIR=$OUT/served-dspark-16k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -8
echo '[window 32 done]'
"
